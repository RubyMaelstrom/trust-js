//! The optimizing tier's exits (P3 stage 1; the plan's Design point 6 and the "Stage 1" record's
//! back end): their recipes, the registry that keeps each table as long as its code, and the
//! deoptimizer.
//!
//! A failing check is one `bl` to its code's shared exit stub (`a64`), V8's form: the stub saves
//! x0-x29 and d0-d31 below the frame and calls [`deoptimize`] with the return address, which
//! names the exit. The exit's recipe gives, for each slot its frame state writes and each
//! operand-stack position, where the value is: a register or spill slot holding a tagged word,
//! an f64, an int32, a Boolean or an object's cell (as the register allocation placed it), or a
//! constant; slots the frame already holds are left as they are. The deoptimizer boxes the
//! values (Numbers as f64 bits with NaN canonicalized, as `emit_exec_number_store` stores them),
//! writes the canonical slots and operand words, rebuilds the handler records, and runs the chunk
//! in the VM from the exit's pc to its completion (`bytecode::run_native_frame_in_vm`, which
//! fills elided rest and `arguments` slots as `run_inner` would); the completion goes to
//! `ctx.ret` or `ctx.error`, and the stub leaves through the matching epilogue. The code stays
//! protected meanwhile: its record's `code_base` names it (`jit::running_code`).
//!
//! Two exits carry a pending throw, for a throw at a pc with a handler (the plan's point 11):
//! a poll's, whose registers its slow path restored before it branches to its exit, and an
//! effect's, whose frame state the code wrote to the frame before the effect ran (the effect
//! clobbers the caller-saved registers), so that its recipe is empty.
//!
//! TRUST_JS_OPT_EXIT_EVERY (the forced-exit gate): when it is compiled in, every check first
//! decrements `Interp::opt_exit_countdown` and exits when it reaches zero; the deoptimizer then
//! counts the exit as forced and starts the countdown again from `Interp::opt_exit_every`.
//!
//! Repeated exits (stage 2's point 12): each record counts its exits and the table their total,
//! forced ones left out, and at the table's limit (`opt::exit_limit`, 100 << R) the deoptimizer
//! discards the code before its VM run (`opt::discard`): new entries take the template JIT's
//! code, and the records' counts become the exit profile. Activations still running a discarded
//! code finish in it, and their exits count without triggering anything again, as JSC's exit
//! returns early for a jettisoned code block (dfg/DFGOperations.cpp:6664-6672).

use super::ir::{ExitReason, Rep};

/// Where an exit finds a value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    /// An x register of the save area, holding a value of `rep` (Tagged, I32, Bool or Cell).
    Gp { reg: u8, rep: Rep },
    /// A d register of the save area, holding an f64.
    Fp { reg: u8 },
    /// The spill slot at `offset` bytes into the frame area, holding a value of `rep`.
    Spill { offset: u32, rep: Rep },
    /// A constant execution word.
    Const(u64),
}

/// What state an exit finds the frame and registers in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExitKind {
    /// A failed check, an unconditional exit, or the forced-exit gate before a check or an
    /// effect: the registers and spill slots hold the state's values.
    Check,
    /// A throw from a loop header's poll: the registers hold the state's values (the slow path
    /// restored them), and the throw is in `ctx.error`.
    PollThrow,
    /// A throw from an effect: the code wrote the state's values to the frame before the effect,
    /// and the throw is in `ctx.error`.
    EffectThrow,
}

/// One exit: its frame state's recipe.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExitRecord {
    /// Where the VM resumes.
    pub(crate) pc: u32,
    pub(crate) kind: ExitKind,
    pub(crate) reason: ExitReason,
    /// The slots the exit writes, with their values' sources.
    pub(crate) slots: Box<[(u16, Source)]>,
    /// The operand stack at `pc`, bottom first.
    pub(crate) stack: Box<[Source]>,
    /// The handler records active at `pc`, innermost last: (push pc, operand depth at the push).
    pub(crate) handlers: Box<[(u32, u32)]>,
    /// A shape check's receiver and expected shapes, which a WrongShape exit's log line shows
    /// (`describe_receiver`) and the exit profile keeps the count of.
    pub(crate) shape_check: Option<Box<ShapeCheck>>,
    /// The exits taken here, forced ones left out: a discard tallies the records with one, as
    /// JSC's tallyFrequentExitSites tallies the exits its code took
    /// (bytecode/CodeBlock.cpp:3094-3120).
    pub(crate) count: std::cell::Cell<u32>,
}

/// What a shape check tested: the receiver word's source at the check, and its shapes.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ShapeCheck {
    pub(crate) receiver: Source,
    pub(crate) shapes: Box<[u32]>,
}

/// A code object's exits.
#[derive(Debug, Default)]
pub(crate) struct ExitTable {
    /// The code's first byte, set when it is mapped.
    pub(crate) code_base: usize,
    /// Each exit's return address from its `bl`, as a byte offset into the code, ascending.
    pub(crate) return_offsets: Vec<u32>,
    pub(crate) exits: Vec<ExitRecord>,
    /// The function's name, for the log.
    pub(crate) name: Box<str>,
    /// The exits taken so far, forced ones left out.
    pub(crate) taken: std::cell::Cell<u32>,
    /// The exits past which the code is discarded (`opt::exit_limit` when it was compiled), so
    /// that an exit reads no chunk to decide.
    pub(crate) limit: u32,
    /// Whether the code was discarded (`opt::discard`).
    pub(crate) discarded: std::cell::Cell<bool>,
}

/// The save area the exit stub makes (ARM64): x0..x29 at words 0..29, d0..d31 at words 30..61;
/// the frame area (the spill slots) starts right above it.
pub(crate) const SAVED_X: usize = 30;
pub(crate) const SAVE_BYTES: u32 = 8 * (30 + 32);

impl ExitTable {
    /// The exit whose `bl` returns to `ret`.
    fn exit_at(&self, ret: usize) -> &ExitRecord {
        let offset = ret.wrapping_sub(self.code_base) as u32;
        let index = self
            .return_offsets
            .binary_search(&offset)
            .unwrap_or_else(|_| panic!("no exit returns to code offset {offset:#x}"));
        &self.exits[index]
    }

    /// Heap bytes the table holds (the memory census's metadata).
    pub(crate) fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.return_offsets.capacity() * 4
            + self.exits.iter().map(record_bytes).sum::<usize>()
            + self.name.len()
    }
}

fn record_bytes(record: &ExitRecord) -> usize {
    std::mem::size_of::<ExitRecord>()
        + record.slots.len() * std::mem::size_of::<(u16, Source)>()
        + record.stack.len() * std::mem::size_of::<Source>()
        + record.handlers.len() * 8
        + record.shape_check.as_ref().map_or(0, |check| {
            std::mem::size_of::<ShapeCheck>() + check.shapes.len() * 4
        })
}

/// The exit tables of the optimized code objects alive in the process, by their code's first
/// byte: each is freed with its code (`release`, from `JitCode`'s drop). The code embeds its
/// table's address, so an exit never looks it up here; Agents on other threads compile too.
struct Registry {
    tables: std::sync::Mutex<crate::fasthash::FastMap<usize, Box<ExitTable>>>,
    live: std::sync::atomic::AtomicUsize,
}

fn registry() -> &'static Registry {
    static REGISTRY: std::sync::OnceLock<Registry> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| Registry {
        tables: std::sync::Mutex::new(Default::default()),
        live: std::sync::atomic::AtomicUsize::new(0),
    })
}

/// Keep `table` while the code at `code_base` lives.
pub(crate) fn register(code_base: usize, table: Box<ExitTable>) {
    let registry = registry();
    let mut tables = registry
        .tables
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if tables.insert(code_base, table).is_none() {
        registry
            .live
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The heap bytes of the exit table of the code at `code_base`, if it has one (the memory census,
/// `JitCode::retained_heap_metadata_bytes`; one atomic load for code without one).
pub(crate) fn table_bytes(code_base: *const u8) -> usize {
    let registry = registry();
    if registry.live.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        return 0;
    }
    registry
        .tables
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&(code_base as usize))
        .map_or(0, |table| table.retained_bytes())
}

/// Free the exit table of the code at `code_base`, if it has one (every code object's drop
/// calls this; the first stage's code has none, and the common case is one atomic load).
pub(crate) fn release(code_base: *const u8) {
    let registry = registry();
    if registry.live.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        return;
    }
    let removed = registry
        .tables
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&(code_base as usize));
    if removed.is_some() {
        registry
            .live
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
thread_local! {
    /// Exits taken on this thread, by pc and reason (tests).
    pub(crate) static TEST_EXITS: std::cell::RefCell<Vec<(u32, ExitReason)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Read a value from the save area or the frame area by its source, boxed as an execution word.
///
/// # Safety
/// `saved` is an exit stub's save area, the frame area right above it.
unsafe fn read(saved: *const u64, source: Source) -> crate::value::Value {
    use crate::value::Value;
    let word = |rep: Rep, bits: u64| -> Value {
        match rep {
            Rep::Tagged => unsafe { Value::from_bits(bits) },
            Rep::F64 => Value::Num(f64::from_bits(bits)),
            Rep::I32 => Value::Num(f64::from(bits as u32 as i32)),
            Rep::Bool => Value::Bool(bits & 1 != 0),
            Rep::Cell => unsafe { Value::from_bits(crate::value::PACK_OBJ | bits) },
            Rep::Ptr => unreachable!("a frame state holds no address (verify.rs)"),
            Rep::Env => unreachable!("a frame state holds no environment (verify.rs)"),
            Rep::None => unreachable!("a frame state holds values"),
        }
    };
    unsafe {
        match source {
            Source::Gp { reg, rep } => word(rep, *saved.add(reg as usize)),
            Source::Fp { reg } => word(Rep::F64, *saved.add(SAVED_X + reg as usize)),
            Source::Spill { offset, rep } => {
                let at = saved.cast::<u8>().add((SAVE_BYTES + offset) as usize);
                word(rep, at.cast::<u64>().read())
            }
            Source::Const(bits) => Value::from_bits(bits),
        }
    }
}

/// A WrongShape exit's log detail: the shapes the check expected, and the receiver's, or what
/// the receiver is when it is no object (a primitive whose wrapper prototype a cache way names,
/// say).
fn describe_receiver(word: crate::value::Value, shapes: &[u32]) -> String {
    use crate::value::ValueView;
    let expected: Vec<String> = shapes.iter().map(|shape| format!("{shape:#x}")).collect();
    let actual = match word.view() {
        ValueView::Obj(object) => match object.try_borrow() {
            Ok(object) => format!(
                "{:#x}{}",
                object.props.shape(),
                if object.ic_plain() {
                    ""
                } else {
                    " (not IC-plain)"
                }
            ),
            Err(_) => "an object in use".to_string(),
        },
        ValueView::Str(_) => "a string".to_string(),
        ValueView::Num(_) => "a number".to_string(),
        ValueView::Bool(_) => "a Boolean".to_string(),
        ValueView::Undefined => "undefined".to_string(),
        ValueView::Null => "null".to_string(),
        _ => "another primitive".to_string(),
    };
    format!(" expected [{}] actual {actual}", expected.join(" "))
}

/// The exit stub's call (`a64`): apply the exit's recipe, then run the chunk in the VM from its pc
/// to completion. Returns 1 with `ctx.ret` the completion's value or 0 with `ctx.error` set,
/// which the stub turns into the matching epilogue.
///
/// # Safety
/// Called only by an exit stub of the code `table` belongs to, with `ctx` its frame's record,
/// `ret` the return address of the exit's `bl` and `saved` the stub's save area.
pub(crate) unsafe extern "C" fn deoptimize(
    ctx: *mut crate::jit::JitCtx,
    table: *const ExitTable,
    ret: usize,
    saved: *const u64,
) -> u64 {
    let ctx = unsafe { &mut *ctx };
    let table = unsafe { &*table };
    let exit = table.exit_at(ret);
    let i = unsafe { &mut *ctx.interp };
    let mut reason = exit.reason;
    if exit.kind == ExitKind::Check && i.opt_exit_every != 0 && i.opt_exit_countdown == 0 {
        // The forced-exit gate's countdown reached zero at this check (see the module header):
        // only a gated check's decrement leaves it there, and this exit starts it again.
        i.opt_exit_countdown = i.opt_exit_every;
        reason = ExitReason::Forced;
    }
    if super::log_enabled() {
        let detail = match &exit.shape_check {
            Some(check) if reason == ExitReason::WrongShape => {
                describe_receiver(unsafe { read(saved, check.receiver) }, &check.shapes)
            }
            _ => String::new(),
        };
        eprintln!(
            "[opt] exit {} pc {} {:?} ({:?}){detail}",
            table.name, exit.pc, reason, exit.kind
        );
    }
    #[cfg(test)]
    TEST_EXITS.with(|exits| exits.borrow_mut().push((exit.pc, reason)));
    let chunk = unsafe { &*ctx.chunk };
    let mut depth = exit.stack.len();
    let pending = match exit.kind {
        ExitKind::Check | ExitKind::PollThrow => {
            // Every value goes to the frame before anything can allocate: from then on the
            // frame's words are what the collector finds.
            let slots = unsafe { std::slice::from_raw_parts(ctx.slots, ctx.n_slots) };
            for &(slot, source) in exit.slots.iter() {
                slots[slot as usize].set(unsafe { read(saved, source) });
            }
            for (k, &source) in exit.stack.iter().enumerate() {
                let word = unsafe { read(saved, source) };
                debug_assert!(!word.is_empty(), "an Empty word on the operand stack");
                unsafe { ctx.stack_base.add(k).write(word) };
            }
            if exit.kind == ExitKind::Check {
                None
            } else {
                Some(())
            }
        }
        ExitKind::EffectThrow => Some(()),
    };
    let pending = match pending {
        None => None,
        // Only a thrown completion is caught: an interruption leaves the function as it is.
        Some(()) => match ctx.error.take() {
            Some(crate::interpreter::Abrupt::Throw(error)) => {
                // The innermost handler restores its depth; the words above it are dead.
                if let Some(&(_, handler_depth)) = exit.handlers.last() {
                    depth = depth.min(handler_depth as usize);
                }
                Some(error)
            }
            other => {
                ctx.error = other;
                return 0;
            }
        },
    };
    if reason != ExitReason::Forced {
        // Repeated exits: at the limit, the code is discarded once (`opt::discard`) and new
        // entries stop using it; this activation finishes in the VM, any other running it in it.
        // The frame holds every value by now, so the discard's compile may allocate.
        exit.count.set(exit.count.get().saturating_add(1));
        let taken = table.taken.get().saturating_add(1);
        table.taken.set(taken);
        if taken >= table.limit && !table.discarded.get() {
            super::discard(i, chunk, table);
        }
    }
    ctx.handlers.clear();
    ctx.handlers
        .extend(exit.handlers.iter().map(|&(push_pc, depth)| {
            crate::bytecode::pushed_handler(chunk, push_pc as usize, depth as usize)
        }));
    let result =
        unsafe { crate::bytecode::run_native_frame_in_vm(ctx, exit.pc as usize, depth, pending) };
    match result {
        Ok(value) => {
            ctx.ret = value;
            1
        }
        Err(error) => {
            ctx.error = Some(error);
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    #[test]
    fn a_table_lives_until_its_code_is_released() {
        let base = 0x7000_0000_0040usize;
        let before = registry().live.load(std::sync::atomic::Ordering::Relaxed);
        register(base, Box::default());
        assert_eq!(
            registry().live.load(std::sync::atomic::Ordering::Relaxed),
            before + 1
        );
        release(base as *const u8);
        release(base as *const u8);
        assert_eq!(
            registry().live.load(std::sync::atomic::Ordering::Relaxed),
            before
        );
    }

    #[test]
    fn recipes_box_each_representation_as_the_frame_holds_it() {
        // x0..x29, d0..d31, then the frame area: one spill slot.
        let mut saved = vec![0u64; SAVED_X + 32 + 1];
        saved[3] = Value::Bool(true).bits();
        saved[4] = (-7i32) as u32 as u64 | 0xdead_0000_0000; // an int32's register, upper half undefined
        saved[5] = 1;
        saved[6] = 0x1230;
        saved[SAVED_X + 9] = f64::NAN.to_bits() | 1; // a NaN payload the code made
        saved[SAVED_X + 32] = 2.5f64.to_bits();
        let read = |source| unsafe { read(saved.as_ptr(), source) }.bits();
        assert_eq!(
            read(Source::Gp {
                reg: 3,
                rep: Rep::Tagged
            }),
            Value::Bool(true).bits()
        );
        assert_eq!(
            read(Source::Gp {
                reg: 4,
                rep: Rep::I32
            }),
            Value::Num(-7.0).bits()
        );
        assert_eq!(
            read(Source::Gp {
                reg: 5,
                rep: Rep::Bool
            }),
            Value::Bool(true).bits()
        );
        assert_eq!(
            read(Source::Gp {
                reg: 6,
                rep: Rep::Cell
            }),
            crate::value::PACK_OBJ | 0x1230
        );
        assert_eq!(read(Source::Fp { reg: 9 }), crate::value::PACK_CANON_NAN);
        assert_eq!(
            read(Source::Spill {
                offset: 0,
                rep: Rep::F64
            }),
            Value::Num(2.5).bits()
        );
        assert_eq!(read(Source::Const(Value::Null.bits())), Value::Null.bits());
    }
}
