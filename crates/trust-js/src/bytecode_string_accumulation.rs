//! A complete before-read declarative string accumulation transaction.
//!
//! ECMA-262 e28783d5 Assignment Evaluation / GetValue / PutValue: the successful
//! region has only a declarative GetValue, a checked local read, primitive string
//! addition and a guaranteed same-record mutable store. Every failed proof leaves
//! the original operations and owners untouched. No post-RHS name re-resolution
//! is used to guess the original Reference.

use crate::value::ValueView;
use std::cell::Cell;

use super::{observe_arithmetic_operands, observe_arithmetic_result, Chunk, Op};
use crate::execution_storage::{SlotAccess, ValueStack};
use crate::interpreter::{Env, Interp, MAX_STR_LEN};
use crate::value::Value;

const MAX_SCOPES: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Plan {
    pub(crate) pc: usize,
    pub(crate) end: usize,
    pub(crate) slot: u16,
    name: u32,
    read_cache: u32,
    write_cache: u32,
    keep: bool,
}

impl Plan {
    pub(crate) fn at(chunk: &Chunk, pc: usize) -> Option<Self> {
        let Op::LoadName(name, read_cache) = *chunk.ops.get(pc)? else {
            return None;
        };
        let Op::LoadLocal(slot) = *chunk.ops.get(pc.checked_add(1)?)? else {
            return None;
        };
        if !matches!(chunk.ops.get(pc.checked_add(2)?), Some(Op::Add)) {
            return None;
        }
        let keep = matches!(chunk.ops.get(pc.checked_add(3)?), Some(Op::Dup));
        let store_pc = pc.checked_add(if keep { 4 } else { 3 })?;
        let Op::StoreNameCached(stored_name, write_cache) = *chunk.ops.get(store_pc)? else {
            return None;
        };
        if stored_name != name
            || name as usize >= chunk.names.len()
            || read_cache as usize >= chunk.name_caches.len()
            || write_cache as usize >= chunk.name_caches.len()
            || slot as usize >= chunk.n_slots
        {
            return None;
        }
        Some(Self {
            pc,
            end: store_pc + 1,
            slot,
            name,
            read_cache,
            write_cache,
            keep,
        })
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_VM_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static TEST_NATIVE_HITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// A code-generation ablation used only to prove actual H_EXEC replay reaches the
    /// transaction. It does not change bytecode or any fallback's semantics.
    pub(crate) static TEST_REPLAY_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct Stored {
    result: Option<Value>,
}

/// Resolution is proved BEFORE the original read. A `with` record, global object/record,
/// missing name, import, TDZ or immutable cell is never bypassed. Taking a mutable borrow of
/// the actual holder pins both the resolution and store through the non-observing transaction.
fn try_store(
    chunk: &Chunk,
    interp: &mut Interp,
    env: &Env,
    plan: Plan,
    right: Value,
) -> Option<Stored> {
    let ValueView::Str(ref suffix) = (right).view() else {
        return None;
    };
    let name = &chunk.names[plan.name as usize];
    let mut current = *env;
    for _ in 0..MAX_SCOPES {
        if Env::ptr_eq(&current, &interp.global_env) {
            return None;
        }
        let parent = {
            let mut record = current.borrow_mut();
            if record.has_with_object() || record.parent.is_none() {
                return None;
            }
            if let Some(binding) = record.vars.get_mut(name) {
                if !binding.initialized
                    || !binding.mutable
                    || binding.import_ref.is_some()
                    || binding.imported
                {
                    return None;
                }
                let original = binding.value();
                let ValueView::Str(prefix) = original.view() else {
                    return None;
                };
                if prefix.len().saturating_add(suffix.len()) > MAX_STR_LEN {
                    return None;
                }
                let profiling =
                    observe_arithmetic_operands(&chunk.feedback, plan.pc + 2, &original, &right);
                // Keep the original binding populated and rooted throughout decode/append.
                // Commit the copied prefix before any observer or interior resume can enter;
                // real aliases, including the RHS and substring views, keep their old text.
                let mut prefix = prefix;
                prefix.append_or_replace(suffix);
                let result = Value::Str(prefix);
                binding.set_value(result);
                observe_arithmetic_result(&chunk.feedback, plan.pc + 2, profiling, &result);
                let stored = Stored {
                    result: plan.keep.then_some(result),
                };
                drop(record);
                // Pure scalar/weak cache population, as after the ordinary checked store.
                // No owner is retained by these caches and no JS operation is replayed.
                let _ = chunk.name_ic_fill(interp, env, plan.name, plan.read_cache);
                let _ = chunk.name_ic_fill(interp, env, plan.name, plan.write_cache);
                return Some(stored);
            }
            record.parent
        };
        current = parent?;
    }
    None
}

pub(super) fn try_vm(
    chunk: &Chunk,
    interp: &mut Interp,
    env: &Env,
    slots: &[Cell<Value>],
    stack: &mut ValueStack,
    pc: usize,
    stop_at: Option<usize>,
) -> Option<usize> {
    let plan = Plan::at(chunk, pc)?;
    if stop_at.is_some_and(|stop| pc < stop && stop < plan.end) {
        return None;
    }
    let stored = try_store(
        chunk,
        interp,
        env,
        plan,
        slots.read_value(plan.slot as usize),
    )?;
    if let Some(result) = stored.result {
        stack.push(result);
    }
    #[cfg(test)]
    TEST_VM_HITS.with(|hits| hits.set(hits.get() + 1));
    Some(plan.end)
}

/// A miss returns the original sp and flag1; success is a complete region with flag0.
/// The emitter branches to `plan.end` only on success and preserves every ordinary interior
/// label. This is not a throwing helper, VM recovery or a specialization miss.
pub(crate) unsafe extern "C" fn jit_try(
    ctx: *mut crate::jit::JitCtx,
    pc: u32,
    sp: *mut Value,
) -> crate::jit::SpFlag {
    let ctx = unsafe { &mut *ctx };
    // OPSTAT counts operations that actually reach their original checked helpers. Keep
    // its existing vocabulary rather than inventing Add/Store calls for a fused commit.
    // Diagnostic execution therefore retains the full original native/helper route.
    if ctx.opstat_enabled {
        return crate::jit::SpFlag { sp, flag: 1 };
    }
    let chunk = unsafe { &*ctx.chunk };
    let Some(plan) = Plan::at(chunk, pc as usize) else {
        return crate::jit::SpFlag { sp, flag: 1 };
    };
    if plan.slot as usize >= ctx.n_slots {
        return crate::jit::SpFlag { sp, flag: 1 };
    }
    // As in jit_exec_inner, the running callee's swapped env_raw is authoritative.
    let env = std::mem::ManuallyDrop::new(unsafe {
        crate::interpreter::Env::from_raw(
            ctx.env_raw as *const std::cell::RefCell<crate::interpreter::Scope>,
        )
    });
    let right = unsafe { ctx.clone_slot(plan.slot as usize) };
    let Some(stored) = try_store(chunk, unsafe { &mut *ctx.interp }, &env, plan, right) else {
        return crate::jit::SpFlag { sp, flag: 1 };
    };
    let end_sp = if let Some(result) = stored.result {
        unsafe {
            sp.write(result);
            sp.add(1)
        }
    } else {
        sp
    };
    #[cfg(test)]
    TEST_NATIVE_HITS.with(|hits| hits.set(hits.get() + 1));
    crate::jit::SpFlag {
        sp: end_sp,
        flag: 0,
    }
}

#[cfg(test)]
#[path = "bytecode_string_accumulation_tests.rs"]
mod tests;
