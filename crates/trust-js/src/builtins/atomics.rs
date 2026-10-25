//! Split out of builtins/mod.rs (behavior-preserving move).

use super::*;
use crate::interpreter::{AsyncWait, BlockedAgent, WaitOutcome, Waiter};
use crate::value::ValueView;
use std::time::Duration;

/// `Atomics` over integer TypedArrays. A shared buffer's read-modify-write operations hold its
/// Shared Data Block's lock; `wait`/`waitAsync`/`notify` use the process-wide WaiterLists.
pub(super) fn install_atomics(it: &mut Interp) {
    let atomics = Object::new(Some(it.object_proto));

    fn target(i: &mut Interp, args: &[Value]) -> Result<(TaInfo, usize), Value> {
        target_rw(i, args, false)
    }
    fn target_rw(i: &mut Interp, args: &[Value], write: bool) -> Result<(TaInfo, usize), Value> {
        let ptr = map_ptr(&arg(args, 0))
            .ok_or_else(|| i.make_error("TypeError", "Atomics: not an integer TypedArray"))?;
        let info = *i
            .typed_arrays
            .get(&ptr)
            .ok_or_else(|| i.make_error("TypeError", "Atomics: not an integer TypedArray"))?;
        if matches!(
            info.kind,
            TaKind::F16 | TaKind::F32 | TaKind::F64 | TaKind::U8Clamped
        ) {
            return Err(i.make_error("TypeError", "Atomics requires an integer TypedArray"));
        }
        if write && i.immutable_buffers.contains(&info.buffer) {
            return Err(i.make_error(
                "TypeError",
                "Atomics: cannot write into a view over an immutable ArrayBuffer",
            ));
        }
        // ValidateAtomicAccess: the length is retrieved BEFORE the index coercion (which may
        // detach or resize the buffer); ToIndex truncates toward zero (NaN→0); a negative or
        // out-of-bounds result is a RangeError, a fractional request index simply truncated.
        // ValidateIntegerTypedArray: an already-detached buffer is a TypeError up front.
        let len0 = i
            .ta_len(&info)
            .ok_or_else(|| i.make_error("TypeError", "Atomics: detached TypedArray"))?;
        let raw = ab(i.to_number(&arg(args, 1)))?;
        let idx = if raw.is_nan() { 0.0 } else { raw.trunc() };
        if idx < 0.0 || !idx.is_finite() || idx as usize >= len0 {
            return Err(i.make_error("RangeError", "Atomics: index out of range"));
        }
        Ok((info, idx as usize))
    }

    /// RevalidateAtomicAccess after a value coercion (which may detach or shrink the buffer).
    fn revalidate(i: &mut Interp, info: &TaInfo, idx: usize) -> Result<(), Value> {
        if i.ta_len(info).map(|l| idx < l) != Some(true) {
            return Err(i.make_error(
                "TypeError",
                "Atomics: buffer was detached or shrunk during value coercion",
            ));
        }
        Ok(())
    }
    fn operand(i: &mut Interp, info: &TaInfo, v: &Value) -> Result<i128, Value> {
        if info.kind.is_bigint() {
            Ok(ab(i.to_bigint(v))?.to_i128_wrapping())
        } else {
            let n = ab(i.to_number(v))?;
            Ok(if n.is_finite() { n.trunc() as i128 } else { 0 })
        }
    }
    /// DoWait's `expected` (ECMA-262 #sec-dowait): ToBigInt64(value) for a BigInt64Array,
    /// ToInt32(value) for an Int32Array, compared with the element's signed value.
    fn wait_operand(i: &mut Interp, info: &TaInfo, v: &Value) -> Result<i128, Value> {
        Ok(if info.kind.is_bigint() {
            ab(i.to_bigint(v))?.to_i128_wrapping() as i64 as i128
        } else {
            ab(i.to_int32(v))? as i128
        })
    }
    /// The signed value of a `kind` element at `byte_index` of a Shared Data Block's bytes.
    fn element_value(kind: TaKind, bytes: &[u8], byte_index: usize) -> Option<i128> {
        let bytes = bytes.get(byte_index..byte_index + kind.elsize())?;
        Some(if kind.is_bigint() {
            kind.read_bigint(bytes)
        } else {
            kind.read(bytes) as i128
        })
    }
    /// DoWait from EnterCriticalSection through AddWaiter (ECMA-262 #sec-dowait): inside the
    /// critical section of the element's WaiterList, read the element and, when it equals
    /// `expected`, add `waiter`. Otherwise the synchronous result: "not-equal", or "timed-out"
    /// for a `waitAsync` whose timeout is zero, which adds no waiter (`None`).
    fn enqueue_waiter(
        info: &TaInfo,
        byte_index: usize,
        id: u64,
        expected: i128,
        waiter: Option<Waiter>,
    ) -> Result<(), &'static str> {
        let block = crate::interpreter::shared_mem_get(id).ok_or("not-equal")?;
        crate::interpreter::with_waiter_list(id, byte_index, |list| {
            if element_value(info.kind, &block.lock().unwrap(), byte_index) != Some(expected) {
                return Err("not-equal");
            }
            list.push(waiter.ok_or("timed-out")?);
            Ok(())
        })
    }
    /// DoWait's realTimeout from ToNumber(timeout): NaN is +∞, and a negative value 0. `None`
    /// is +∞, as is a finite timeout too long to represent.
    fn wait_timeout(i: &mut Interp, timeout: &Value) -> Result<Option<Duration>, Value> {
        let q = ab(i.to_number(timeout))?;
        Ok(if q.is_nan() {
            None
        } else {
            Duration::try_from_secs_f64(q.max(0.0) / 1000.0).ok()
        })
    }
    fn write_i128(i: &mut Interp, info: &TaInfo, idx: usize, n: i128) {
        if info.kind.is_bigint() {
            i.ta_write_bigint(info, idx, n);
        } else {
            i.ta_write(info, idx, n as f64);
        }
    }
    /// Truncate `v` to the element type's bit width (unsigned domain) for raw comparison.
    fn wrap_bits(kind: TaKind, v: i128) -> u128 {
        let bits = kind.elsize() * 8;
        (v as u128) & (u128::MAX >> (128 - bits))
    }
    fn rmw(i: &mut Interp, args: &[Value], f: fn(i128, i128) -> i128) -> Result<Value, Value> {
        let (info, idx) = target_rw(i, args, true)?;
        let val = operand(i, &info, &arg(args, 2))?;
        revalidate(i, &info, idx)?;
        // One atomic read-modify-write (a shared buffer's lock is held across both halves).
        let old = i.ta_modify(&info, idx, |o| Some(f(o, val))).unwrap_or(0);
        Ok(if info.kind.is_bigint() {
            Value::BigInt(old.into())
        } else {
            Value::Num(old as f64)
        })
    }

    it.def_method(&atomics, "add", 3, |i, _t, a| rmw(i, a, |o, v| o + v));
    it.def_method(&atomics, "sub", 3, |i, _t, a| rmw(i, a, |o, v| o - v));
    it.def_method(&atomics, "and", 3, |i, _t, a| rmw(i, a, |o, v| o & v));
    it.def_method(&atomics, "or", 3, |i, _t, a| rmw(i, a, |o, v| o | v));
    it.def_method(&atomics, "xor", 3, |i, _t, a| rmw(i, a, |o, v| o ^ v));
    it.def_method(&atomics, "exchange", 3, |i, _t, a| rmw(i, a, |_o, v| v));
    it.def_method(&atomics, "load", 2, |i, _t, a| {
        let (info, idx) = target(i, a)?;
        revalidate(i, &info, idx)?;
        Ok(i.ta_read(&info, idx))
    });
    it.def_method(&atomics, "store", 3, |i, _t, a| {
        let (info, idx) = target_rw(i, a, true)?;
        let val = operand(i, &info, &arg(a, 2))?;
        revalidate(i, &info, idx)?;
        write_i128(i, &info, idx, val);
        // store returns the coerced value itself, not the (possibly wrapped) stored representation.
        Ok(if info.kind.is_bigint() {
            Value::BigInt(val.into())
        } else {
            Value::Num(val as f64)
        })
    });
    it.def_method(&atomics, "compareExchange", 4, |i, _t, a| {
        let (info, idx) = target_rw(i, a, true)?;
        let expected = operand(i, &info, &arg(a, 2))?;
        let replacement = operand(i, &info, &arg(a, 3))?;
        revalidate(i, &info, idx)?;
        // The comparison is on the element's raw byte representation, so the expected value
        // wraps to the element type first (e.g. 68547 matches an Int16 2979); the compare and
        // the conditional write happen under one lock hold.
        let old = i
            .ta_modify(&info, idx, |o| {
                if wrap_bits(info.kind, o) == wrap_bits(info.kind, expected) {
                    Some(replacement)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        Ok(if info.kind.is_bigint() {
            Value::BigInt(old.into())
        } else {
            Value::Num(old as f64)
        })
    });
    it.def_method(&atomics, "isLockFree", 1, |i, _t, a| {
        let n = ab(i.to_number(&arg(a, 0)))?;
        Ok(Value::Bool(matches!(n as i64, 1 | 2 | 4 | 8)))
    });
    // ValidateIntegerTypedArray(ta, waitable=true): only a non-detached Int32Array / BigInt64Array,
    // checked BEFORE any index/value coercion (so a poisoned index doesn't run first).
    fn require_waitable(i: &mut Interp, ta: &Value) -> Result<TaInfo, Value> {
        let err = || {
            i.make_error(
                "TypeError",
                "Atomics.wait/notify requires an Int32Array or BigInt64Array",
            )
        };
        let info = map_ptr(ta)
            .and_then(|p| i.typed_arrays.get(&p).copied())
            .ok_or_else(err)?;
        if !matches!(info.kind, TaKind::I32 | TaKind::I64) {
            return Err(err());
        }
        // A detached buffer (regular buffer removed from the store) has no [[ArrayBufferData]].
        if !i.buffer_attached(info.buffer) {
            return Err(i.make_error("TypeError", "Atomics.wait/notify: buffer is detached"));
        }
        Ok(info)
    }
    it.def_method(&atomics, "wait", 4, |i, _t, a| {
        let winfo = require_waitable(i, &arg(a, 0))?;
        // Atomics.wait needs a *shared* buffer — checked before ValidateAtomicAccess (index coercion).
        let id = match i.shared_block_id(winfo.buffer) {
            Some(id) => id,
            None => {
                return Err(i.make_error(
                    "TypeError",
                    "Atomics.wait requires a SharedArrayBuffer-backed array",
                ))
            }
        };
        let (info, idx) = target(i, a)?;
        let expected = wait_operand(i, &info, &arg(a, 2))?;
        let timeout = wait_timeout(i, &arg(a, 3))?;
        // Only an agent with [[CanBlock]] may suspend (the main agent cannot).
        if !i.can_block {
            return Err(i.make_error(
                "TypeError",
                "Atomics.wait: the calling agent cannot be suspended",
            ));
        }
        let byte_index = info.offset + idx * info.kind.elsize();
        let agent = BlockedAgent::default();
        let waiter = Waiter::Blocked(agent.clone());
        if let Err(result) = enqueue_waiter(&info, byte_index, id, expected, Some(waiter)) {
            return Ok(Value::lstr(result));
        }
        let interrupt = i.runtime_interrupt.clone();
        match crate::interpreter::futex_block(&agent, id, byte_index, timeout, &interrupt) {
            WaitOutcome::Woken => Ok(Value::lstr("ok")),
            WaitOutcome::TimedOut => Ok(Value::lstr("timed-out")),
            // Native dispatch polls the host's control again when this returns and replaces this
            // placeholder with the non-catchable interruption.
            WaitOutcome::Interrupted => Err(i.make_error(
                "Error",
                interrupt
                    .current_reason()
                    .map_or("Atomics.wait was interrupted", |reason| reason.message()),
            )),
        }
    });
    it.def_method(&atomics, "notify", 3, |i, _t, a| {
        require_waitable(i, &arg(a, 0))?;
        let (info, idx) = target(i, a)?;
        // count = ToIntegerOrInfinity(arg2), default +Infinity (= all); negative clamps to 0.
        let count: i64 = {
            let __value_match_original = arg(a, 2);
            match (__value_match_original).view() {
                ValueView::Undefined => -1,
                _ => {
                    let v = __value_match_original;
                    {
                        let n = ab(i.to_number(&v))?;
                        if n.is_nan() || n < 0.0 {
                            0
                        } else if n == f64::INFINITY {
                            -1
                        } else {
                            n as i64
                        }
                    }
                }
            }
        };
        // notify is a no-op (returns 0) on a non-shared buffer.
        match i.shared_block_id(info.buffer) {
            Some(id) => {
                let byte_index = info.offset + idx * info.kind.elsize();
                let own = i.gc_heap.agent.work.borrow().inbox.clone();
                let (woken, own_tokens) =
                    crate::interpreter::futex_notify(id, byte_index, count, own.as_ref());
                // NotifyWaiter resolves this agent's own waitAsync promises at once.
                for token in own_tokens {
                    i.resolve_async_wait(token, "ok");
                }
                Ok(Value::Num(woken as f64))
            }
            None => Ok(Value::Num(0.0)),
        }
    });
    it.def_method(&atomics, "waitAsync", 4, |i, _t, a| {
        // ValidateIntegerTypedArray(waitable) runs before any index/value coercion.
        let winfo = require_waitable(i, &arg(a, 0))?;
        let id = match i.shared_block_id(winfo.buffer) {
            Some(id) => id,
            None => {
                return Err(i.make_error(
                    "TypeError",
                    "Atomics.waitAsync requires a SharedArrayBuffer-backed array",
                ))
            }
        };
        let (info, idx) = target(i, a)?;
        let expected = wait_operand(i, &info, &arg(a, 2))?;
        let timeout = wait_timeout(i, &arg(a, 3))?;
        let result = i.new_object();
        // A value that already differs, or a zero timeout, resolves synchronously (not async).
        let byte_index = info.offset + idx * info.kind.elsize();
        let token = crate::interpreter::next_async_wait_token();
        let inbox = i.agent_async_inbox();
        let waiter = (timeout != Some(Duration::ZERO)).then_some(Waiter::Async(inbox, token));
        if let Err(value) = enqueue_waiter(&info, byte_index, id, expected, waiter) {
            set_data(&result, "async", Value::Bool(false));
            set_data(&result, "value", Value::lstr(value));
            return Ok(Value::Obj(result));
        }
        // The waiter joined its list synchronously, so a notify later in this same job sees it.
        // A notify resolves the promise (this agent's at once, another agent's through this
        // agent's inbox), or this agent's job checkpoints time it out (settle_async_waits).
        let promise = i.new_promise();
        let block = i.shared_block(info.buffer).unwrap().clone();
        i.enqueue_async_wait(AsyncWait {
            promise,
            token,
            id,
            block,
            index: byte_index,
            // A timeout too far away to represent never elapses.
            deadline: timeout.and_then(|timeout| std::time::Instant::now().checked_add(timeout)),
        });
        set_data(&result, "async", Value::Bool(true));
        set_data(&result, "value", promise);
        Ok(Value::Obj(result))
    });
    it.def_method(&atomics, "pause", 0, |i, _t, a| {
        // The optional iterationNumber, if present, must be an integral Number.
        match (arg(a, 0)).view() {
            ValueView::Undefined => {}
            ValueView::Num(n) if n.fract() == 0.0 && n.is_finite() => {}
            _ => {
                return Err(i.make_error(
                    "TypeError",
                    "Atomics.pause iterationNumber must be an integer",
                ))
            }
        }
        Ok(Value::Undefined)
    });
    set_to_string_tag(it, &atomics, "Atomics");
    set_builtin(&it.global, "Atomics", Value::Obj(atomics));
}
