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
//!
//! An exited frame that loops (stage 2, step 13, JSC's form with the template JIT as the
//! baseline): each exit at or past the table's loop limit (`opt::loop_exit_limit`, 5 << R, never
//! above the limit) that does not discard arms the loop limit's warm-up again
//! (`opt::arm_loop_limit`: `loop_warm_up` of the chunk's template loop iterations from now), as
//! JSC's exit resets the baseline's execution counter to its long warm-up
//! (dfg/DFGOSRExitCompilerCommon.cpp:105-141); a template loop whose count reaches it asks the
//! next native entry to discard the code (`opt::loop_limit_due`), as the baseline loop's tier-up
//! check jettisons the replacement whose exits reached osrExitCountForReoptimizationFromLoop
//! (jit/JITOperations.cpp:2944-2966). The exited frame leaves the VM at its first back edge, as
//! every exited frame does. Forced exits arm nothing, as they count toward nothing.

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
    /// A free name's check's proof byte (snapshot.rs's NameView::proof), which a WrongName exit
    /// keeps in the exit profile (stage 2, step 4); [`super::ir::TRUNCATION_EXIT`] for a
    /// TruncateNumberOrOddball's NotNumber exit, which the profile keeps likewise; 0 for every
    /// other exit.
    pub(crate) proof: u8,
    /// The exits taken here, forced ones left out: a discard tallies the records with one, as
    /// JSC's tallyFrequentExitSites tallies the exits its code took
    /// (bytecode/CodeBlock.cpp:3094-3120).
    pub(crate) count: std::cell::Cell<u32>,
    /// The frame the recipe rebuilds (stage 3): 0, the compiled function's, or an inlined
    /// frame's, [`ExitTable::frames`]`[frame - 1]`, whose recipe and its parents' rebuild the
    /// frames below it. In the record's padding: a record is no larger than before.
    pub(crate) frame: u16,
}

/// An inlined frame's part of the exits through it (P3 stage 3, step 1), kept once in the table
/// for every exit of the frame and the frames above it: where its callee and receiver are, and
/// its parent's recipe, the caller's state at `call_pc + 1` without the call's operands, which
/// the deoptimizer resumes with the call's result pushed (or at `call_pc` with its throw). JSC
/// keeps one InlineCallFrame per inlined call that every exit through it reads
/// (dfg/DFGOSRExitCompilerCommon.cpp:293-415); a home is a value's for its whole range
/// (`regalloc`), so one recipe of the parent serves every exit.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FrameRecipe {
    /// The callee's first-stage chunk: its index among the code's inlined chunks ([`Inlined`]).
    pub(crate) chunk: u32,
    /// The call's pc in the parent's chunk.
    pub(crate) call_pc: u32,
    /// The parent's frame: 0, the compiled function's, or `frames[parent - 1]`.
    pub(crate) parent: u16,
    pub(crate) callee: Source,
    pub(crate) receiver: Source,
    /// The parent's slots and operand stack, as an exit record's.
    pub(crate) slots: Box<[(u16, Source)]>,
    pub(crate) stack: Box<[Source]>,
    pub(crate) handlers: Box<[(u32, u32)]>,
}

/// What a shape check tested: the receiver word's source at the check, and its shapes; for a
/// prototype's check (CheckProto, P3 stage 3 step L2: `hop`) the source of the object whose
/// [[Prototype]] it checked. `ways` is the access's receiver shapes, the count its WrongShape
/// site keeps in the exit profile (`Rules::stale_ways` compares the site's ways with it).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ShapeCheck {
    pub(crate) receiver: Source,
    pub(crate) shapes: Box<[u32]>,
    pub(crate) ways: u8,
    pub(crate) hop: bool,
}

/// A code object's exits.
#[derive(Debug, Default)]
pub(crate) struct ExitTable {
    /// The code's first byte, set when it is mapped.
    pub(crate) code_base: usize,
    /// Each exit's return address from its `bl`, as a byte offset into the code, ascending.
    pub(crate) return_offsets: Box<[u32]>,
    pub(crate) exits: Box<[ExitRecord]>,
    /// The inlined frames some exit passes through, each a parent's before its own (stage 3);
    /// empty for a code without inlined frames. Boxes, where the two above were vectors, keep
    /// the table's size.
    pub(crate) frames: Box<[FrameRecipe]>,
    /// The function's name, for the log.
    pub(crate) name: Box<str>,
    /// The exits taken so far, forced ones left out.
    pub(crate) taken: std::cell::Cell<u32>,
    /// The exits past which the code is discarded (`opt::exit_limit` when it was compiled), so
    /// that an exit reads no chunk to decide.
    pub(crate) limit: u32,
    /// The exits from which each exit that does not discard arms the loop limit's warm-up on
    /// the chunk (`opt::arm_loop_limit`), whose crossing discards the code at the next entry
    /// (`opt::loop_exit_limit` when it was compiled: 5 << R, never above `limit`; stage 2, step
    /// 13): JSC's exitCountThresholdForReoptimizationFromLoop (bytecode/CodeBlock.cpp:
    /// 2758-2771).
    pub(crate) loop_limit: u32,
    /// The loop limit's warm-up in native runs of template loop iterations
    /// (`opt::loop_warm_up` when it was compiled: a discard's warm-up's first distance, as
    /// JSC's exit and jettison both wait thresholdForOptimizeAfterWarmUp's 1,000,
    /// runtime/OptionsList.h:373-374).
    pub(crate) loop_warm_up: u32,
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
            + self.return_offsets.len() * 4
            + self.exits.iter().map(record_bytes).sum::<usize>()
            + self.frames.iter().map(frame_bytes).sum::<usize>()
            + self.name.len()
    }

    /// Inlined frame `frame`'s recipe (`frame` > 0).
    fn frame(&self, frame: u16) -> &FrameRecipe {
        &self.frames[usize::from(frame) - 1]
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

fn frame_bytes(frame: &FrameRecipe) -> usize {
    std::mem::size_of::<FrameRecipe>()
        + frame.slots.len() * std::mem::size_of::<(u16, Source)>()
        + frame.stack.len() * std::mem::size_of::<Source>()
        + frame.handlers.len() * 8
}

/// What a code with inlined frames holds for as long as it lives (stage 3): the chunks its
/// inlined frames run, by [`FrameRecipe::chunk`], which its exits rebuild frames of, and the weak
/// pins of the identities it embeds, so that no other object takes their addresses (as an
/// InlineTarget's pin keeps a splice's callee, bytecode.rs's `InlineTarget::pin`). Kept with the
/// code's base on the thread that linked the code, which alone runs and drops it (the exit table
/// crosses threads and holds the chunks by index), and released with its exit table (`release`).
#[derive(Default)]
pub(crate) struct Inlined {
    pub(crate) chunks: Box<[std::rc::Rc<crate::bytecode::Chunk>]>,
    pub(crate) pins: Box<[crate::value::WeakGc]>,
}

thread_local! {
    /// Each linked code's [`Inlined`], by its code's base.
    static INLINED: std::cell::RefCell<crate::fasthash::FastMap<usize, Inlined>> =
        std::cell::RefCell::new(Default::default());
}

/// Keep `inlined` as long as the code at `code_base` (its link, on its thread): nothing for a
/// code without inlined frames.
pub(crate) fn hold(code_base: usize, inlined: Inlined) {
    if inlined.chunks.is_empty() && inlined.pins.is_empty() {
        return;
    }
    let replaced = INLINED.with(|held| held.borrow_mut().insert(code_base, inlined));
    debug_assert!(replaced.is_none(), "a released code's holds left with it");
    // Dropped out of the map's borrow, as `drop_held` drops.
    drop(replaced);
}

/// Inlined chunk `index` of the code at `code_base`, which lives as long as the code: its
/// running activation keeps it, and only its drop releases what it holds.
fn inlined_chunk(code_base: usize, index: u32) -> *const crate::bytecode::Chunk {
    INLINED.with(|held| {
        let held = held.borrow();
        let inlined = held
            .get(&code_base)
            .expect("a code with inlined frames holds their chunks");
        std::rc::Rc::as_ptr(&inlined.chunks[index as usize])
    })
}

/// The first-stage chunk of the inlined frame `exit` rebuilds, None for the compiled function's
/// frame (the exit profile's key, `opt::discard`).
pub(crate) fn exit_chunk<'t>(
    table: &'t ExitTable,
    exit: &ExitRecord,
) -> Option<&'t crate::bytecode::Chunk> {
    (exit.frame != 0).then(|| {
        // SAFETY: the code holds the chunk as long as its table lives (`release`).
        unsafe { &*inlined_chunk(table.code_base, table.frame(exit.frame).chunk) }
    })
}

/// The heap bytes the code at `code_base` holds for its inlined frames (the memory census).
fn held_bytes(code_base: usize) -> usize {
    INLINED
        .try_with(|held| {
            held.borrow().get(&code_base).map_or(0, |inlined| {
                std::mem::size_of::<Inlined>()
                    + inlined.chunks.len()
                        * std::mem::size_of::<std::rc::Rc<crate::bytecode::Chunk>>()
                    + inlined.pins.len() * std::mem::size_of::<crate::value::WeakGc>()
            })
        })
        .unwrap_or(0)
}

/// Drop what the code at `code_base` held for its inlined frames, if anything (its release).
fn drop_held(code_base: usize) {
    let held = INLINED
        .try_with(|held| {
            let mut held = held.borrow_mut();
            if held.is_empty() {
                None
            } else {
                held.remove(&code_base)
            }
        })
        .ok()
        .flatten();
    // Dropped out of the map's borrow: a chunk's drop may release other codes.
    drop(held);
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
    let bytes = registry
        .tables
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&(code_base as usize))
        .map(|table| table.retained_bytes());
    match bytes {
        Some(bytes) => bytes + held_bytes(code_base as usize),
        None => 0,
    }
}

/// The exit table of `code`, if it is the tier's code (template JIT code has none). The table
/// lives as long as its code, which the caller holds: only the code's drop releases it
/// (`release`), and the registry's map moves only its boxes.
pub(crate) fn table_of(code: &crate::jit::JitCode) -> Option<&ExitTable> {
    let registry = registry();
    if registry.live.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        return None;
    }
    let table = registry
        .tables
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&(code.mem_ptr() as usize))
        .map(|table| &**table as *const ExitTable)?;
    // SAFETY: the box is freed only by `release`, at `code`'s drop, which the borrow outlives.
    Some(unsafe { &*table })
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
        drop_held(code_base as usize);
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
/// The [[Prototype]] of an object word (null for anything else, or an object in use), for the
/// log of a prototype's check.
fn prototype_of(word: crate::value::Value) -> crate::value::Value {
    use crate::value::{Value, ValueView};
    match word.view() {
        ValueView::Obj(object) => match object.try_borrow() {
            Ok(object) => object.proto.map_or(Value::Null, Value::Obj),
            Err(_) => Value::Null,
        },
        _ => Value::Null,
    }
}

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
                let word = unsafe { read(saved, check.receiver) };
                if check.hop {
                    format!(
                        " (a prototype){}",
                        describe_receiver(prototype_of(word), &check.shapes)
                    )
                } else {
                    describe_receiver(word, &check.shapes)
                }
            }
            _ => String::new(),
        };
        let frame = match exit.frame {
            0 => String::new(),
            frame => format!(" in inlined frame {frame}"),
        };
        eprintln!(
            "[opt] exit {} pc {} {:?} ({:?}){frame}{detail}",
            table.name, exit.pc, reason, exit.kind
        );
    }
    #[cfg(test)]
    TEST_EXITS.with(|exits| exits.borrow_mut().push((exit.pc, reason)));
    if exit.frame != 0 {
        return unsafe { deoptimize_frames(ctx, table, exit, reason, saved) };
    }
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
    // The frame holds every value by now, so the discard's compile may allocate.
    count(i, chunk, table, exit, reason, false);
    ctx.handlers.clear();
    ctx.handlers
        .extend(exit.handlers.iter().map(|&(push_pc, depth)| {
            crate::bytecode::pushed_handler(chunk, push_pc as usize, depth as usize)
        }));
    let result =
        unsafe { crate::bytecode::run_native_frame_in_vm(ctx, exit.pc as usize, depth, pending) };
    complete(ctx, result)
}

/// An exit's count (see the module's doc), forced ones left out: one count and one limit for the
/// code, whichever frame exits, as JSC counts every exit of a code block on it
/// (dfg/DFGOSRExitCompilerCommon.cpp:42-104). Every frame's values are in rooted storage.
/// `from_loop`: the exit passes through an inlined frame whose callee's first stage tried to
/// enter a loop (P3 stage 3, step 2), whose exits the loop limit takes at once.
#[inline(always)]
fn count(
    i: &mut crate::interpreter::Interp,
    chunk: &crate::bytecode::Chunk,
    table: &ExitTable,
    exit: &ExitRecord,
    reason: ExitReason,
    from_loop: bool,
) {
    if reason != ExitReason::Forced {
        // Repeated exits: at the limit, the code is discarded once (`opt::discard`) and new
        // entries stop using it; this activation finishes in the VM, any other running it in it.
        exit.count.set(exit.count.get().saturating_add(1));
        let taken = table.taken.get().saturating_add(1);
        table.taken.set(taken);
        if taken >= table.limit && !table.discarded.get() {
            super::discard(i, chunk, table, super::Limit::Exits);
        }
        // JSC's loop rule for inlined frames (D12; dfg/DFGOSRExitCompilerCommon.cpp:70-104): an
        // exit through an inline call frame whose baseline code tried to enter a loop takes the
        // loop threshold at once, since only the outermost code block's loops count toward it.
        if from_loop && taken >= table.loop_limit && !table.discarded.get() {
            super::discard(i, chunk, table, super::Limit::InlinedLoop);
        }
        // An exited frame that loops (stage 2, step 13; see the module's doc): at the loop limit
        // or past it, the code not discarded, the loop limit's warm-up starts again (JSC's reset
        // of the baseline counter at every exit that does not jettison); below it, nothing.
        if taken >= table.loop_limit && !table.discarded.get() {
            super::arm_loop_limit(chunk, table.loop_warm_up);
        }
    }
}

/// The compiled frame's completion, for the stub's epilogue: 1 with `ctx.ret` its value, or 0
/// with `ctx.error` set.
#[inline(always)]
fn complete(
    ctx: &mut crate::jit::JitCtx,
    result: Result<crate::value::Value, crate::interpreter::Abrupt>,
) -> u64 {
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
thread_local! {
    /// Exits through inlined frames taken on this thread: the exiting frame's depth in its chain
    /// (1 the outermost inlined frame), its pc and the reason (tests).
    pub(crate) static TEST_INLINED_EXITS: std::cell::RefCell<Vec<(usize, u32, ExitReason)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// An inlined frame the deoptimizer rebuilt (`deoptimize_frames`): its record, the callee's
/// bookkeeping a call would make, and its call in its parent.
struct Rebuilt {
    record: std::ptr::NonNull<crate::jit::JitFrame>,
    /// The callee's identity (`Gc::as_ptr`), rooted by its parent's operand words, where the
    /// call's window holds it.
    fn_ptr: usize,
    strict: bool,
    arrow: bool,
    uses_this: bool,
    /// The parent's chunk and the call's pc there.
    parent_chunk: *const crate::bytecode::Chunk,
    call_pc: u32,
    /// The parent's operand depth without the call's operands: where the call's result goes.
    parent_depth: usize,
}

/// Where the innermost frame resumes: the exit's pc, its operand depth and the throw it carries.
struct Top {
    pc: usize,
    depth: usize,
    pending: Option<crate::value::Value>,
}

/// What `deoptimize_frames` checks of its first phase (tests): when every frame's values are
/// written, nothing has been allocated or collected and no frame entered since the exit.
#[cfg(test)]
struct Quiet {
    collections: u64,
    objects: u64,
    frames: usize,
    depth: u32,
}

#[cfg(test)]
impl Quiet {
    fn now(i: &crate::interpreter::Interp) -> Self {
        Quiet {
            collections: i.gc_heap.core.borrow().collections(),
            objects: crate::value::heap_allocated_objects(&i.gc_heap),
            frames: i.fn_frames.len(),
            depth: i.depth,
        }
    }

    fn check(&self, i: &crate::interpreter::Interp) {
        assert!(
            i.gc_heap.core.borrow().collections() == self.collections
                && crate::value::heap_allocated_objects(&i.gc_heap) == self.objects
                && i.fn_frames.len() == self.frames
                && i.depth == self.depth,
            "a frame's values written after an allocation, a collection or a frame's entry"
        );
    }
}

/// An exit through inlined frames (P3 stage 3, step 1; D5): every frame rebuilt first, then the
/// innermost run in the VM and each parent resumed after its call, as V8's deoptimizer builds an
/// interpreter frame per function, the non-topmost ones advancing past their call with the
/// callee's result (deoptimizer/deoptimizer.cc:2080-2094, 2296-2300), SpiderMonkey rebuilds a
/// Baseline frame per inlined frame (jit/Bailouts.h:100-131) and JSC reifies every inline call
/// frame (dfg/DFGOSRExitCompilerCommon.cpp:293-415) before the innermost resumes.
///
/// (a) Every frame's values in rooted storage before anything allocates or collects: the compiled
/// frame's in its slots and operand words, as a one-frame exit writes them; each inlined frame's
/// in a pooled activation record popped for it, which the collector traces while it is out of
/// the pool, its receiver its `this` until (b) binds it; each call's window, the callee and a
/// CallWithThis's receiver, in its parent's operand words above the parent's depth, where a
/// call's operands lie while it runs (SpiderMonkey's caller frame keeps the call's callee and
/// `this` the same way, jit/BaselineBailouts.cpp:1036-1060). (b) For each inlined frame,
/// outermost first, the bookkeeping of a call (`Interp::enter_call_frame`, the call path's own):
/// the depth and its check, an FnFrame with the call's position in its parent, the construct
/// state saved; then `this` bound. (c) The innermost frame runs in the VM from the exit's pc on
/// its record (`run_native_frame_in_vm`). (d) Each parent resumes from its call's pc + 1 with the
/// result pushed, or from the call's pc with the throw pending, which the VM routes to the
/// parent's handler or out; an interruption passes every frame; each frame's FnFrame is popped and
/// its record released as its run ends (`Interp::leave_call_frame`). (e) The compiled frame's
/// completion goes to the stub's epilogue, as a one-frame exit's.
///
/// # Safety
/// As [`deoptimize`]'s, `exit` an exit of `table` through an inlined frame.
#[cold]
#[inline(never)]
unsafe fn deoptimize_frames(
    ctx: &mut crate::jit::JitCtx,
    table: &ExitTable,
    exit: &ExitRecord,
    reason: ExitReason,
    saved: *const u64,
) -> u64 {
    use crate::interpreter::Abrupt;
    use crate::value::Value;
    let i = unsafe { &mut *ctx.interp };
    #[cfg(test)]
    let quiet = Quiet::now(i);
    // A throw the exit carries; an interruption leaves the function as it is, no frame rebuilt.
    let mut pending = match exit.kind {
        ExitKind::Check => None,
        ExitKind::PollThrow => match ctx.error.take() {
            Some(Abrupt::Throw(error)) => Some(error),
            other => {
                ctx.error = other;
                return 0;
            }
        },
        ExitKind::EffectThrow => unreachable!("no effect runs in an inlined frame (step 3's)"),
    };
    // The chain, innermost first.
    let mut chain: Vec<u16> = Vec::new();
    let mut frame = exit.frame;
    while frame != 0 {
        chain.push(frame);
        frame = table.frame(frame).parent;
    }
    #[cfg(test)]
    TEST_INLINED_EXITS.with(|exits| exits.borrow_mut().push((chain.len(), exit.pc, reason)));
    let chunk = unsafe { &*ctx.chunk };
    // (a) Each inlined frame, outermost first: its record and its values, its call's window in
    // its parent's operand words. A frame's own state is the exit's for the innermost, and the
    // parent recipe of the frame above it for the others; the compiled frame's, the outermost
    // inlined frame's parent recipe.
    let state_of = |k: usize| -> (&[(u16, Source)], &[Source], &[(u32, u32)]) {
        // The state of chain[k]: its own exit's, or the parent recipe of chain[k - 1].
        if k == 0 {
            (&exit.slots, &exit.stack, &exit.handlers)
        } else {
            let above = table.frame(chain[k - 1]);
            (&above.slots, &above.stack, &above.handlers)
        }
    };
    let outermost = table.frame(*chain.last().expect("an inlined frame"));
    let mut rebuilt: Vec<Rebuilt> = Vec::with_capacity(chain.len());
    // The compiled frame's values, as a one-frame exit writes them.
    unsafe {
        write_values(
            ctx,
            &outermost.slots,
            &outermost.stack,
            saved,
            #[cfg(test)]
            (&quiet, &*i),
        )
    };
    let (code_base, genv) = (ctx.code_base as usize, ctx.genv);
    for k in (0..chain.len()).rev() {
        let recipe = table.frame(chain[k]);
        let callee_chunk = unsafe { &*inlined_chunk(code_base, recipe.chunk) };
        let callee = unsafe { read(saved, recipe.callee) };
        let receiver = unsafe { read(saved, recipe.receiver) };
        let object = callee
            .as_obj()
            .expect("an inlined frame's callee is a function (its identity guard)");
        let (strict, arrow, env) = match &object.borrow().kind {
            crate::value::ObjectKind::User(user) => {
                (user.func.is_strict, user.func.is_arrow, user.env)
            }
            _ => unreachable!("an inlined frame's callee is an author function"),
        };
        // The call's window in its parent's operand words, above the parent's depth.
        let parent_ctx = match rebuilt.last() {
            None => &mut *ctx,
            Some(parent) => unsafe { &mut (*parent.record.as_ptr()).ctx },
        };
        let parent_chunk = parent_ctx.chunk;
        let parent_depth = state_of(k + 1).1.len();
        unsafe { write_window(parent_ctx, parent_depth, recipe.call_pc, callee, receiver) };
        let record = i
            .frame_pool
            .pop()
            .unwrap_or_else(|| crate::jit::JitFrame::alloc(i));
        let record_ctx = unsafe { &mut (*record.as_ptr()).ctx };
        let (_, n_slots) = callee_chunk.jit_frame();
        record_ctx.chunk = callee_chunk;
        record_ctx.n_slots = n_slots;
        record_ctx.stack_base = unsafe { record_ctx.slots.add(n_slots).cast::<Value>() };
        record_ctx.final_sp = record_ctx.stack_base;
        record_ctx.env_raw = crate::interpreter::Env::as_ptr(&env) as *const u8;
        record_ctx.env_parent_raw = crate::jit::jit_env_parent_raw(&env);
        record_ctx.genv = genv;
        // Unbound until (b): the record keeps it meanwhile.
        record_ctx.set_this(receiver);
        // An inlined frame's every slot: Undefined, then the recipe's (Dead slots stay so).
        let slots = unsafe { std::slice::from_raw_parts(record_ctx.slots, n_slots) };
        for slot in slots {
            slot.set(Value::Undefined);
        }
        let (own_slots, own_stack, _) = state_of(k);
        unsafe {
            write_values(
                record_ctx,
                own_slots,
                own_stack,
                saved,
                #[cfg(test)]
                (&quiet, &*i),
            )
        };
        rebuilt.push(Rebuilt {
            record,
            fn_ptr: crate::value::Gc::as_ptr(&object) as usize,
            strict,
            arrow,
            uses_this: callee_chunk.needs_frame_this(),
            parent_chunk,
            call_pc: recipe.call_pc,
            parent_depth,
        });
    }
    #[cfg(test)]
    quiet.check(i);
    // Tests: each rebuilt frame's callee lies in its call's window, in its parent's operand words,
    // which root it while the frame runs (its FnFrame holds no handle, `FnFrame::fn_ptr`).
    #[cfg(test)]
    for (level, frame) in rebuilt.iter().enumerate() {
        let parent = match level {
            0 => &*ctx,
            _ => unsafe { &(*rebuilt[level - 1].record.as_ptr()).ctx },
        };
        let window =
            unsafe { std::slice::from_raw_parts(parent.stack_base.add(frame.parent_depth), 2) };
        assert!(
            window.iter().any(|word| word
                .as_obj()
                .is_some_and(|callee| crate::value::Gc::as_ptr(&callee) as usize == frame.fn_ptr)),
            "a rebuilt frame's callee in its call's window"
        );
    }
    // Every frame's values are in rooted storage: the count, whose discard may allocate. An
    // inlined frame's callee whose first stage tried to enter a loop (its borrowed-frame code's
    // state, made at the VM's first loop transfer: JSC's didTryToEnterInLoop, set at a loop's OSR
    // entry, dfg/DFGOSREntry.cpp:225-226) puts the exit under the loop limit (P3 stage 3, step 2).
    let from_loop = chain.iter().any(|&frame| {
        // SAFETY: the code holds the chunk as long as its table lives (`release`).
        let callee = unsafe { &*inlined_chunk(code_base, table.frame(frame).chunk) };
        callee.osr.get().is_some()
    });
    count(i, chunk, table, exit, reason, from_loop);
    // Each frame's handler records, rebuilt from its state against its own chunk.
    let frame_handlers = |frame_ctx: &mut crate::jit::JitCtx, handlers: &[(u32, u32)]| {
        let frame_chunk = unsafe { &*frame_ctx.chunk };
        frame_ctx.handlers.clear();
        frame_ctx
            .handlers
            .extend(handlers.iter().map(|&(push_pc, depth)| {
                crate::bytecode::pushed_handler(frame_chunk, push_pc as usize, depth as usize)
            }));
    };
    frame_handlers(ctx, state_of(chain.len()).2);
    for (level, frame) in rebuilt.iter().enumerate() {
        let k = chain.len() - 1 - level;
        frame_handlers(unsafe { &mut (*frame.record.as_ptr()).ctx }, state_of(k).2);
    }
    let mut depth = exit.stack.len();
    if pending.is_some() {
        // The innermost handler restores its depth; the words above it are dead.
        if let Some(&(_, handler_depth)) = exit.handlers.last() {
            depth = depth.min(handler_depth as usize);
        }
    }
    let top = Top {
        pc: exit.pc as usize,
        depth,
        pending: pending.take(),
    };
    let inner = unsafe { run_inlined(i, &rebuilt, 0, top) };
    let result = unsafe { resume(ctx, &rebuilt[0], inner) };
    complete(ctx, result)
}

/// Write a frame's values from the save area: the slots its recipe names and its operand words.
///
/// # Safety
/// As [`deoptimize`]'s; `frame` a frame's record of the recipe's chunk.
unsafe fn write_values(
    frame: &mut crate::jit::JitCtx,
    slots: &[(u16, Source)],
    stack: &[Source],
    saved: *const u64,
    #[cfg(test)] (quiet, i): (&Quiet, &crate::interpreter::Interp),
) {
    #[cfg(test)]
    quiet.check(i);
    let words = unsafe { std::slice::from_raw_parts(frame.slots, frame.n_slots) };
    for &(slot, source) in slots {
        words[slot as usize].set(unsafe { read(saved, source) });
    }
    for (k, &source) in stack.iter().enumerate() {
        let word = unsafe { read(saved, source) };
        debug_assert!(!word.is_empty(), "an Empty word on the operand stack");
        unsafe { frame.stack_base.add(k).write(word) };
    }
}

/// A call's window in its parent's operand words, from `depth` up, as the call left them while it
/// ran: a Call's callee; a CallWithThis's receiver, then its callee (bytecode.rs's Op::Call and
/// Op::CallWithThis). The parent's VM run starts below them; they root the callee meanwhile.
///
/// # Safety
/// `parent` is a frame's record whose operand words hold the call's window.
unsafe fn write_window(
    parent: &mut crate::jit::JitCtx,
    depth: usize,
    call_pc: u32,
    callee: crate::value::Value,
    receiver: crate::value::Value,
) {
    let chunk = unsafe { &*parent.chunk };
    unsafe {
        match chunk.jit_ops()[call_pc as usize] {
            crate::bytecode::Op::Call(..) => parent.stack_base.add(depth).write(callee),
            crate::bytecode::Op::CallWithThis(..) => {
                parent.stack_base.add(depth).write(receiver);
                parent.stack_base.add(depth + 1).write(callee);
            }
            _ => unreachable!("an inlined frame's call is a Call or a CallWithThis (verify.rs)"),
        }
    }
}

/// Inlined frames `frames[level..]` (outermost first) entered as their calls enter them, the
/// innermost run from `top`, and each resumed after the frame above it, back to `frames[level]`'s
/// completion (see `deoptimize_frames`'s (b) to (d)). A frame whose entry fails (the depth past
/// its limit, a collection's throw, an interruption) completes so, as its call would, and the
/// frames above it are released unrun.
///
/// # Safety
/// `frames` were rebuilt by `deoptimize_frames` for this exit; their records are out of the pool.
unsafe fn run_inlined(
    i: &mut crate::interpreter::Interp,
    frames: &[Rebuilt],
    level: usize,
    top: Top,
) -> Result<crate::value::Value, crate::interpreter::Abrupt> {
    let frame = &frames[level];
    // The call's position in its parent, which its frame keeps (`explicit_call`'s).
    i.call_site =
        crate::bytecode::call_op_position(unsafe { &*frame.parent_chunk }, frame.call_pc as usize);
    let entry = match i.enter_call_frame(frame.fn_ptr, frame.strict, frame.arrow) {
        Ok(entry) => entry,
        Err(error) => {
            i.call_site = crate::stack_trace::NO_POSITION;
            for unrun in &frames[level..] {
                unsafe { crate::jit::JitFrame::release(i, unrun.record) };
            }
            return Err(error);
        }
    };
    let record = unsafe { &mut (*frame.record.as_ptr()).ctx };
    let this = i.bind_this_word(record.this_value(), frame.uses_this, frame.strict);
    record.set_this(this);
    let depth = i.depth;
    let result = crate::interpreter::with_execution_stack(depth, || {
        if level + 1 < frames.len() {
            let inner = unsafe { run_inlined(i, frames, level + 1, top) };
            unsafe { resume(record, &frames[level + 1], inner) }
        } else {
            unsafe {
                crate::bytecode::run_native_frame_in_vm(record, top.pc, top.depth, top.pending)
            }
        }
    });
    let result = i.leave_call_frame(entry, result);
    i.call_site = crate::stack_trace::NO_POSITION;
    unsafe { crate::jit::JitFrame::release(i, frame.record) };
    result
}

/// `parent`'s run after its call to `callee` (an inlined frame) completed with `inner`: from the
/// call's pc + 1 with the value pushed where the call's window began, as the call leaves it; from
/// the call's pc with a throw pending, which the VM routes to the parent's handler or out, at the
/// innermost handler's depth; an interruption passes it unrun.
///
/// # Safety
/// `parent` is the record of `callee`'s parent, its values and handlers rebuilt.
unsafe fn resume(
    parent: &mut crate::jit::JitCtx,
    callee: &Rebuilt,
    inner: Result<crate::value::Value, crate::interpreter::Abrupt>,
) -> Result<crate::value::Value, crate::interpreter::Abrupt> {
    use crate::interpreter::Abrupt;
    match inner {
        Ok(value) => {
            unsafe { parent.stack_base.add(callee.parent_depth).write(value) };
            unsafe {
                crate::bytecode::run_native_frame_in_vm(
                    parent,
                    callee.call_pc as usize + 1,
                    callee.parent_depth + 1,
                    None,
                )
            }
        }
        Err(Abrupt::Throw(error)) => {
            let depth = parent
                .handlers
                .last()
                .map_or(callee.parent_depth, |handler| {
                    callee.parent_depth.min(handler.stack_depth)
                });
            unsafe {
                crate::bytecode::run_native_frame_in_vm(
                    parent,
                    callee.call_pc as usize,
                    depth,
                    Some(error),
                )
            }
        }
        Err(other) => Err(other),
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
