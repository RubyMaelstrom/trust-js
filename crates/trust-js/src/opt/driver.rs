//! The tier's driver, where it has a back end: its switches, the settled-feedback rule that picks
//! the checkpoint it compiles at, the pipeline run that makes a chunk's native code, and the
//! fall-back from code that exits too often.

use super::ir::Refusal;
use super::{a64, build, snapshot, verify};
use crate::bytecode::Chunk;
use crate::interpreter::Interp;
use std::cell::RefCell;

/// The switches a test sets for its thread in place of the environment's, which a process reads
/// once (`TestSwitches::set`).
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TestSwitches {
    pub(crate) on: bool,
    pub(crate) eager: bool,
    pub(crate) exit_every: u32,
}

#[cfg(test)]
thread_local! {
    static TEST_SWITCHES: std::cell::Cell<Option<TestSwitches>> =
        const { std::cell::Cell::new(None) };
    /// The chunks the tier compiled on this thread, and the codes repeated exits retired.
    pub(crate) static TEST_COMPILED: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    pub(crate) static TEST_FELL_BACK: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    /// The checkpoints (chunk, number) at which the tier waited for settled feedback.
    pub(crate) static TEST_WAITS: RefCell<Vec<(usize, u8)>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
impl TestSwitches {
    /// Use these switches on this thread from now on (an engine made afterwards takes
    /// `exit_every` for its countdown).
    pub(crate) fn set(self) {
        TEST_SWITCHES.with(|switches| switches.set(Some(self)));
    }
}

#[cfg(test)]
fn test_switches() -> Option<TestSwitches> {
    TEST_SWITCHES.with(std::cell::Cell::get)
}

/// TRUST_JS_OPT: the tier is off unless set to a value other than `0` (the plan's Stage 1: off
/// by default until a stage exit measures a net win).
pub(crate) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(switches) = test_switches() {
        return switches.on;
    }
    matches!(env_value!("TRUST_JS_OPT"), Some(v) if v != "0")
}

/// TRUST_JS_OPT_LOG: one stderr line per compile (its time and code range), refusal (with its
/// reason and pc), exit, wait for settled feedback, fallback, and template compile of a second
/// stage (its code range).
pub(crate) fn log_enabled() -> bool {
    env_flag!("TRUST_JS_OPT_LOG")
}

/// TRUST_JS_OPT_EAGER: compile at the first checkpoint, without waiting for settled feedback (the
/// eager gate; V8's --stress-maglev, SpiderMonkey's eager Ion compilation).
pub(crate) fn eager() -> bool {
    #[cfg(test)]
    if let Some(switches) = test_switches() {
        return switches.eager;
    }
    env_flag!("TRUST_JS_OPT_EAGER")
}

/// TRUST_JS_OPT_EXIT_EVERY=N: fail every Nth check passed (the forced-exit gate; V8's
/// --deopt-every-n-times, SpiderMonkey's bailAfter). 0 or unset: off.
pub(crate) fn exit_every() -> u32 {
    #[cfg(test)]
    if let Some(switches) = test_switches() {
        return switches.exit_every;
    }
    env_value!("TRUST_JS_OPT_EXIT_EVERY")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// The exits of one code object past which new entries stop using it (`fall_back`): the plan's
/// point 12, JSC's 100 x 2^R exits at R = 0, without the recompile stage 2 brings.
pub(crate) const EXIT_LIMIT: u32 = 100;

thread_local! {
    /// The feedback fingerprint the last second-stage checkpoint of each chunk recorded while
    /// the tier waits for its feedback to settle, by the chunk's address: an entry goes when the
    /// chunk's checkpoints end, and with the chunk (`forget`). The tier's own state, so that the
    /// chunk carries nothing for it while the tier is off.
    static SETTLING: RefCell<crate::fasthash::FastMap<usize, u64>> =
        RefCell::new(Default::default());
    /// Whether `fall_back` is compiling the template JIT's code (`note_template`'s label).
    static FALLING_BACK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The settled-feedback rule at a second-stage checkpoint of `chunk` (the plan's Stage 1 record,
/// "When"; V8's wait for a quiet IC): whether its feedback fingerprint (`snapshot::fingerprint`)
/// equals the one its previous checkpoint recorded, or, under TRUST_JS_OPT_EAGER, any
/// checkpoint. Records this checkpoint's fingerprint for the next one.
pub(crate) fn settled(chunk: &Chunk) -> bool {
    let print = snapshot::fingerprint(chunk);
    let key = chunk as *const Chunk as usize;
    let previous = SETTLING.with(|settling| settling.borrow_mut().insert(key, print));
    eager() || previous == Some(print)
}

/// Whether the tier waits for `chunk`'s next checkpoint (`last`: none follows this one): while
/// a checkpoint remains and the feedback has not settled (`settled`). The last checkpoint
/// compiles whatever the fingerprint says, as JSC optimizes after maximumOptimizationDelay
/// delays whatever its profiles say (CodeBlock::shouldOptimizeNowFromBaseline) and V8 pushes a
/// tier-up out at an IC change but never cancels it (TieringManager::NotifyICChanged). Logs
/// each wait and each compile of unsettled feedback under TRUST_JS_OPT_LOG.
pub(crate) fn waits(chunk: &Chunk, last: bool) -> bool {
    if settled(chunk) {
        return false;
    }
    if !last {
        if log_enabled() {
            eprintln!(
                "[opt] waiting {} at checkpoint {}: feedback changed",
                name(chunk),
                chunk.inline_checkpoints()
            );
        }
        #[cfg(test)]
        TEST_WAITS.with(|waits| {
            waits
                .borrow_mut()
                .push((chunk as *const Chunk as usize, chunk.inline_checkpoints()))
        });
        return true;
    }
    if log_enabled() {
        eprintln!(
            "[opt] compiling {} at its last checkpoint ({}) with unsettled feedback",
            name(chunk),
            chunk.inline_checkpoints()
        );
    }
    false
}

/// `chunk`'s checkpoints have ended (its second stage was built, the last one passed, or the
/// chunk drops), or its wait starts afresh.
pub(crate) fn forget(chunk: &Chunk) {
    let key = chunk as *const Chunk as usize;
    // A chunk can drop while the thread's locals are torn down.
    let _ = SETTLING.try_with(|settling| settling.borrow_mut().remove(&key));
}

fn name(chunk: &Chunk) -> String {
    let names: Vec<&str> = chunk
        .jit_slot_names()
        .iter()
        .take(4)
        .map(|name| &**name)
        .collect();
    format!("fn({})", names.join(","))
}

/// Compile `chunk` with the tier (the snapshot, the graph, its verification, the ARM64 code) and
/// make the code the chunk's native code: `chunk` is a second stage, or the function's own chunk
/// compiled again because it had nothing to splice, and has no native code yet. Whether the tier
/// accepted it. A refusal is logged with its reason and pc under TRUST_JS_OPT_LOG; a graph the
/// verifier rejects is the builder's bug, which panics in tests and debug builds and is refused
/// otherwise.
pub(crate) fn compile_into(i: &mut Interp, chunk: &Chunk) -> bool {
    release_retired(i);
    let started = std::time::Instant::now();
    // The slot first, as the template JIT's compiles take it: a refusal drops the permit, which
    // leaves the slot cold.
    let Some(permit) = chunk.jit.begin_compile() else {
        if log_enabled() {
            eprintln!("[opt] refused {}: no code slot", name(chunk));
        }
        return false;
    };
    let compiled = (|| {
        let snapshot = snapshot::Snapshot::take(chunk);
        let graph = build::build(chunk, &snapshot)?;
        if let Err(message) = verify::verify(&graph, chunk) {
            if cfg!(any(test, debug_assertions)) {
                panic!(
                    "the tier's graph of {} fails verification: {message}",
                    name(chunk)
                );
            }
            if log_enabled() {
                eprintln!("[opt] verify {} failed: {message}", name(chunk));
            }
            return Err(Refusal::whole("the graph fails verification"));
        }
        let layout = *i
            .jit_layout
            .get_or_init(|| crate::value::jit_layout(&i.object_proto));
        if !i.interp_layout.get().valid {
            let layout = crate::interpreter::interp_layout(i);
            i.interp_layout.set(layout);
        }
        if !crate::jit::ensure_executable_capacity(0) {
            return Err(Refusal::whole("no executable capacity"));
        }
        let code = a64::compile(chunk, &graph, &layout, &i.interp_layout.get())?;
        Ok((graph.nodes.len(), code))
    })();
    let elapsed = started.elapsed();
    match compiled {
        Ok((nodes, code)) => {
            if log_enabled() {
                eprintln!(
                    "[opt] compiled {} ({} ops, {nodes} nodes, {} bytes) in {:.3} ms \
                     at {:#x} {:#x}",
                    name(chunk),
                    chunk.jit_ops().len(),
                    code.code_bytes(),
                    elapsed.as_secs_f64() * 1e3,
                    code.mem_ptr() as usize,
                    code.code_bytes()
                );
            }
            #[cfg(test)]
            TEST_COMPILED
                .with(|compiled| compiled.borrow_mut().push(chunk as *const Chunk as usize));
            permit.commit(Some(code));
            true
        }
        Err(refusal) => {
            if log_enabled() {
                match refusal.pc {
                    Some(pc) => eprintln!(
                        "[opt] refused {} at pc {pc}: {} ({:.3} ms)",
                        name(chunk),
                        refusal.reason,
                        elapsed.as_secs_f64() * 1e3
                    ),
                    None => eprintln!(
                        "[opt] refused {}: {} ({:.3} ms)",
                        name(chunk),
                        refusal.reason,
                        elapsed.as_secs_f64() * 1e3
                    ),
                }
            }
            false
        }
    }
}

/// Past [`EXIT_LIMIT`] exits of the tier's code for `chunk` (the plan's Stage 1 record, "Repeated
/// exits"): new entries take the template JIT's code for the chunk, compiled now (or the VM where
/// it cannot be), the call caches' epoch moves so that every way re-resolves, and the tier's code
/// stays alive among the interpreter's retired codes while an activation runs it, the exiting one
/// among them. Stage 1 compiles the chunk with the tier no more.
pub(crate) fn fall_back(i: &mut Interp, chunk: &Chunk) {
    let layout = *i
        .jit_layout
        .get_or_init(|| crate::value::jit_layout(&i.object_proto));
    if !i.interp_layout.get().valid {
        let layout = crate::interpreter::interp_layout(i);
        i.interp_layout.set(layout);
    }
    FALLING_BACK.with(|falling| falling.set(true));
    let template = match crate::jit::compile_profiled(chunk, &layout, &i.interp_layout.get()) {
        crate::jit::JitCompileOutcome::Compiled(code) => Some(std::rc::Rc::new(code)),
        _ => None,
    };
    FALLING_BACK.with(|falling| falling.set(false));
    let compiled = template.is_some();
    if let Some(retired) = chunk.jit.replace_resident(template) {
        #[cfg(test)]
        TEST_FELL_BACK.with(|fell| fell.borrow_mut().push(retired.mem_ptr() as usize));
        i.opt_retired.push(retired);
    }
    crate::bytecode::invalidate_call_caches();
    if log_enabled() {
        eprintln!(
            "[opt] {} fell back after {EXIT_LIMIT} exits to the {}",
            name(chunk),
            if compiled {
                "template JIT's code"
            } else {
                "VM"
            }
        );
    }
    release_retired(i);
}

/// The template JIT compiled `chunk`'s code at `base`, `len` bytes: under TRUST_JS_OPT_LOG with
/// the tier on, a second stage's (a chunk with a record map, the tier's own empty-plan chunks
/// among them) is logged with its code range, as the tier's compiles are, so that a profile's
/// samples tell the two kinds of second-stage code apart (the perf map names both `JS:^`).
/// "fallback" is `fall_back`'s compile; "second stage" any other, such as the first native
/// entry's compile of a second stage the tier refused, or a loop's borrowed-frame code.
pub(crate) fn note_template(chunk: &Chunk, base: *mut u8, len: usize) {
    if !log_enabled() || !enabled() || chunk.feedback_vector().record_map().is_none() {
        return;
    }
    let why = if FALLING_BACK.with(std::cell::Cell::get) {
        "fallback"
    } else {
        "second stage"
    };
    eprintln!(
        "[opt] template {:#x} {:#x} {} ({why})",
        base as usize,
        len,
        name(chunk)
    );
}

/// Drop the retired codes no activation runs any more.
pub(crate) fn release_retired(i: &mut Interp) {
    if i.opt_retired.is_empty() {
        return;
    }
    let running = crate::jit::running_code();
    i.opt_retired
        .retain(|code| running.contains(&(code.mem_ptr() as usize)));
}
