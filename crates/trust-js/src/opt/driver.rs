//! The tier's driver, where it has a back end: its switches, the settled-feedback rule that picks
//! the checkpoint it compiles at, the pipeline run that makes a chunk's native code, the discard
//! of code that exits too often, and its reoptimization (P3 stage 2, step 3).
//!
//! The plan's point 12 on TRust's code (the stage 2 design, "Step 3"): past its limit of exits,
//! 100 << R, a code is discarded (`discard`): its exits' sites become the exit profile, R (the
//! function's discards) grows, the template JIT's code takes new entries, and the second stage's
//! warm-up is armed, its checkpoints (100, 400, 1,600, 6,400) << R invocations apart, as JSC
//! waits twice as long after each jettison (bytecode/CodeBlock.cpp:2602-2608), and then one
//! every 500 << R. At a checkpoint whose feedback has settled the tier compiles the chunk again
//! (`reoptimize`), declining what the profile flagged, and its code replaces the template JIT's.
//! Departures from point 12, each for a precondition TRust lacks yet: the recompile takes the
//! same chunk again (a second stage is set once; another splicing is stage 3's); the limit for an
//! exited frame that then loops hot (5 x 2^R) stays with stage 5, since the exited frame finishes
//! in the VM, whose loops count nothing; and a discard whose exits were all unfixable (throws,
//! unsupported operations) arms no recompile, which would build the same code, as SpiderMonkey
//! disables Ion for frequent unfixable bailouts (jit/IonScript.h:336-341), until the in-place
//! resume takes exceptions out of the count. Each discard and each recompile moves the call
//! caches' global epoch, where the exemplars reach only the replaced code's callers; the log
//! counts the moves.
//!
//! The checkpoints count loop iterations as well as native entries (stage 2, step 8), with V8's
//! weight (step 11): with the tier on, an entry counts K native runs (`super::ENTRY_WEIGHT`), and
//! each loop iteration of template code (a first stage's, a discarded code's fallback) its loop's
//! share of an entry, max(1, round(K x B / L)), B the loop's span and L the chunk's length
//! (`loop_weight`), toward its chunk's checkpoint word while a checkpoint or a warm-up is pending
//! (`counts_loops`, `jit::emit_loop_count`), as Ignition's back edge charges its loop's length
//! against a budget of invocations times the bytecode's length (InterpreterAssembler::
//! JumpBackward, interpreter/interpreter-assembler.cc:1252-1256); the next native entry takes the
//! checkpoint, and transfer into a running loop is OSR, stage 5's. Each distance is in
//! invocations, K native runs each, so that a loop-free function keeps its schedule.
//!
//! Nothing forces a compile (step 11): the settled-feedback rule waits at every checkpoint whose
//! feedback changed since the last, and past the fourth one comes every 500 invocations
//! (`super::SETTLED_WINDOW`; a warm-up's, 500 << R), V8's wait after an IC change, which raises a
//! function's budget to minimum_invocations_after_ic_update invocations' worth and forces nothing
//! later (TieringManager::NotifyICChanged, execution/tiering-manager.cc:497-576). A function whose
//! feedback never settles keeps its template code; so does one whose next checkpoint would pass
//! the u32 word, which is not armed. The tier refuses a chunk over V8's size limit
//! (`MAX_OPTIMIZED_OPS`) before its snapshot, or with nothing to splice before its recompile.

use super::exit::{ExitRecord, ExitTable};
use super::ir::{ExitReason, Refusal};
use super::profile::{self, ChunkProfile};
use super::{a64, build, snapshot, verify};
use crate::bytecode::Chunk;
use crate::interpreter::Interp;
use crate::jit::JitCode;
use std::cell::Cell;
use std::rc::Rc;

/// The switches a test sets for its thread in place of the environment's, which a process reads
/// once (`TestSwitches::set`).
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TestSwitches {
    pub(crate) on: bool,
    pub(crate) eager: bool,
    pub(crate) exit_every: u32,
    pub(crate) no_reopt: bool,
    pub(crate) no_exit_profile: bool,
    /// The base exit limit; 0: the default ([`EXIT_LIMIT`]).
    pub(crate) exit_limit: u32,
    /// TRUST_JS_OPT_NO_LOOP_COUNT: template code's loop iterations count toward no checkpoint.
    pub(crate) no_loop_count: bool,
}

#[cfg(test)]
thread_local! {
    static TEST_SWITCHES: std::cell::Cell<Option<TestSwitches>> =
        const { std::cell::Cell::new(None) };
    /// The chunks the tier compiled on this thread (recompiles included), and the codes discards
    /// retired.
    pub(crate) static TEST_COMPILED: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
    pub(crate) static TEST_DISCARDED: std::cell::RefCell<Vec<usize>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// The checkpoints (chunk, number) at which the tier waited for settled feedback, a warm-up's
    /// among them.
    pub(crate) static TEST_WAITS: std::cell::RefCell<Vec<(usize, u8)>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// The graphs the verifier checked on this thread (`generate`).
    pub(crate) static TEST_VERIFIED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The chunks the tier refused on this thread (`generate`, `refuses_unspliced`), with the
    /// refusals' reasons.
    pub(crate) static TEST_REFUSED: std::cell::RefCell<Vec<(usize, &'static str)>> =
        const { std::cell::RefCell::new(Vec::new()) };
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

/// Whether the tier verifies the graphs it builds (`verify`, stage 2 step 10): in debug builds
/// and the library's tests, and in release builds only under TRUST_JS_OPT_VERIFY=1, as Maglev
/// verifies only #ifdef DEBUG (VerifyGraph, maglev/maglev-compiler.cc:74-80), SpiderMonkey's
/// AssertGraphCoherency is empty without DEBUG (jit/MIRGraph.cpp:1812-1822) and JSC validates
/// with assertions or its validateGraph option, false by default (validationEnabled,
/// dfg/DFGCommon.h:88-95). A release build otherwise spends none of its compile time on it.
pub(crate) fn verifies() -> bool {
    verifies_in(
        cfg!(any(test, debug_assertions)),
        env_flag!("TRUST_JS_OPT_VERIFY"),
    )
}

/// [`verifies`] in a build with or without debug assertions, TRUST_JS_OPT_VERIFY set or not.
pub(crate) fn verifies_in(debug: bool, switch: bool) -> bool {
    debug || switch
}

/// TRUST_JS_OPT_LOG: one stderr line per compile (its time and code range), refusal (with its
/// reason and pc), decline, exit, wait for settled feedback, discard, flag, reoptimization, epoch
/// move, template compile of a second stage (its code range), and main-thread part of a tier
/// compile besides `generate` (`log_main_thread`: the recompile, the install, a fallback).
pub(crate) fn log_enabled() -> bool {
    env_flag!("TRUST_JS_OPT_LOG")
}

/// TRUST_JS_OPT_EAGER: compile at the first checkpoint, without waiting for settled feedback (the
/// eager gate; V8's --stress-maglev, SpiderMonkey's eager Ion compilation); a warm-up's
/// recompile too.
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

/// TRUST_JS_OPT_NO_REOPT: stage 1's fallback for good: a discard records no exit profile, adds
/// nothing to R and arms no recompile.
pub(crate) fn no_reopt() -> bool {
    #[cfg(test)]
    if let Some(switches) = test_switches() {
        return switches.no_reopt;
    }
    env_flag!("TRUST_JS_OPT_NO_REOPT")
}

/// TRUST_JS_OPT_NO_EXIT_PROFILE: discards record no exit profile, so that recompiles decline
/// nothing (P2's variant, its plan:6581-6587): the feedback alone guides them.
pub(crate) fn no_exit_profile() -> bool {
    #[cfg(test)]
    if let Some(switches) = test_switches() {
        return switches.no_exit_profile;
    }
    env_flag!("TRUST_JS_OPT_NO_EXIT_PROFILE")
}

/// The base exit limit: TRUST_JS_OPT_EXIT_LIMIT=N, a positive N, else [`EXIT_LIMIT`] (1 cycles
/// reoptimization at every exit, the gates' fourth configuration).
pub(crate) fn exit_limit_base() -> u32 {
    #[cfg(test)]
    if let Some(switches) = test_switches() {
        return if switches.exit_limit == 0 {
            EXIT_LIMIT
        } else {
            switches.exit_limit
        };
    }
    env_value!("TRUST_JS_OPT_EXIT_LIMIT")
        .and_then(|v| v.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(EXIT_LIMIT)
}

/// The base exit limit by default: JSC's 100 x 2^R exits at R = 0 (the plan's point 12;
/// exitCountThresholdForReoptimization, bytecode/CodeBlock.cpp:2738-2771).
pub(crate) const EXIT_LIMIT: u32 = 100;

/// TRUST_JS_OPT_NO_LOOP_COUNT: template code's loop iterations count toward no checkpoint, which
/// then count native entries alone, as before stage 2's step 8 (for comparisons).
pub(crate) fn no_loop_count() -> bool {
    #[cfg(test)]
    if let Some(switches) = test_switches() {
        return switches.no_loop_count;
    }
    env_flag!("TRUST_JS_OPT_NO_LOOP_COUNT")
}

/// Whether the template JIT's code, compiled now, counts its loop iterations toward its chunk's
/// checkpoints (stage 2, step 8; see the module's doc): with the tier on, unless
/// TRUST_JS_OPT_NO_LOOP_COUNT. Tier-off code counts nothing and is emitted as before.
pub(crate) fn counts_loops() -> bool {
    enabled() && !no_loop_count()
}

/// The native runs an iteration of a loop spanning `span` operations (its header to its last
/// back edge) counts in a chunk of `len` (stage 2, step 11): its share of an entry's K
/// (`super::ENTRY_WEIGHT`), max(1, round(K x span / len)), as Ignition's back edge charges its
/// loop's length against a budget of invocations times the bytecode's length (see the module's
/// doc). At most K: a loop spans no more than its chunk.
pub(crate) fn loop_weight(span: usize, len: usize) -> u32 {
    let k = u64::from(super::ENTRY_WEIGHT);
    let (span, len) = (span as u64, len.max(1) as u64);
    ((k * span + len / 2) / len).clamp(1, k) as u32
}

/// The tier's size limit (stage 2, step 11): `generate` refuses a chunk of more operations
/// before its snapshot, and a first stage of more with nothing to splice is not recompiled
/// (`refuses_unspliced`), as V8 never optimizes a function over max_optimized_bytecode_size (60
/// KB; flags/flag-definitions.h:1557), giving it the largest budget
/// (execution/tiering-manager.cc:282-283): 60 KB at three to six bytes of V8's bytecode an
/// operation (Mandreel's initPhysics, 19,580 operations, is 66,944 bytes in V8 13.6). A refusal
/// keeps the second stage, as every refusal does.
pub(crate) const MAX_OPTIMIZED_OPS: usize = 15_000;

/// The exits past which the tier's code for `chunk`, compiled now, is discarded: the base limit
/// << R, R the function's discards so far (on its first stage), so that each discard doubles
/// the next code's limit.
pub(crate) fn exit_limit(chunk: &Chunk) -> u32 {
    let r = ChunkProfile::get(profile::first_stage(chunk)).map_or(0, ChunkProfile::discards);
    (u64::from(exit_limit_base()) << r).min(u64::from(u32::MAX)) as u32
}

/// The warm-up's first distance in invocations at R = 0; each of the next three is four times
/// the last ((100, 400, 1,600, 6,400) << R), the second stage's own schedule
/// (`Chunk::advance_inline_feedback`) doubled at each discard, and then one every 500 << R
/// (`super::SETTLED_WINDOW`), V8's wait after an IC change: nothing forces the recompile. An
/// invocation is K native runs (`super::entry_weight`): an entry, or in the fallback's template
/// code loop iterations (`counts_loops`, `loop_weight`).
const WARM_UP: u32 = 100;
/// The warm-up's checkpoints apart by growing distances; the window follows them.
const WARM_UP_CHECKPOINTS: u8 = 4;

/// The native runs from a warm-up's checkpoint `k` (0 for the discard) to the next, at `r`:
/// None where they would pass the u32 word (never at R up to `profile::MAX_DISCARDS`).
pub(crate) fn warm_up_gap(r: u8, k: u8) -> Option<u32> {
    let invocations = if k < WARM_UP_CHECKPOINTS {
        u64::from(WARM_UP) << (2 * u32::from(k))
    } else {
        u64::from(super::SETTLED_WINDOW)
    };
    u32::try_from((invocations * u64::from(super::entry_weight())) << r).ok()
}

/// The checkpoint word a warm-up's checkpoint `k` (0 for the discard) arms at `runs` native runs,
/// at `r`: None where it would pass u32::MAX, which arms nothing (the template JIT's code stays,
/// as the second stage's past its word does).
pub(crate) fn warm_up_word(runs: u32, r: u8, k: u8) -> Option<u32> {
    runs.checked_add(warm_up_gap(r, k)?)
}

thread_local! {
    /// Whether `discard` is compiling the template JIT's code (`note_template`'s label).
    static DISCARDING: Cell<bool> = const { Cell::new(false) };
    /// The epoch moves the tier made on this thread (the log's count).
    static EPOCH_MOVES: Cell<u64> = const { Cell::new(0) };
}

/// The settled-feedback rule at a checkpoint of `chunk` (the plan's Stage 1 record, "When"; V8's
/// wait for a quiet IC): whether its feedback fingerprint (`snapshot::fingerprint`) equals the
/// one its previous checkpoint recorded, or, under TRUST_JS_OPT_EAGER, any checkpoint. Records
/// this checkpoint's fingerprint for the next one, on the chunk (`opt::profile`).
pub(crate) fn settled(chunk: &Chunk) -> bool {
    let print = snapshot::fingerprint(chunk);
    let previous = ChunkProfile::of(chunk).settle(print);
    eager() || previous == Some(print)
}

/// Whether the tier waits at `chunk`'s second-stage checkpoint: while the feedback has not
/// settled (`settled`), at every checkpoint, the next one coming at most 500 invocations later
/// (`Chunk::advance_inline_feedback`), as V8 delays a tier-up 500 invocations' worth after an IC
/// change and forces nothing (TieringManager::NotifyICChanged; stage 2, step 11). Logs each wait
/// under TRUST_JS_OPT_LOG.
pub(crate) fn waits(chunk: &Chunk) -> bool {
    waits_at(chunk, chunk.inline_checkpoints(), "")
}

/// [`waits`] at `chunk`'s checkpoint `checkpoint`, `what` the log's note of its kind.
fn waits_at(chunk: &Chunk, checkpoint: u8, what: &str) -> bool {
    if settled(chunk) {
        return false;
    }
    if log_enabled() {
        eprintln!(
            "[opt] waiting {} at checkpoint {checkpoint}: feedback changed{what}",
            name(chunk)
        );
    }
    #[cfg(test)]
    TEST_WAITS.with(|waits| {
        waits
            .borrow_mut()
            .push((chunk as *const Chunk as usize, checkpoint))
    });
    true
}

/// V8's size limit before the bytecode recompile (stage 2, step 11): whether `chunk`, a first
/// stage whose plan splices nothing, is over the limit (`MAX_OPTIMIZED_OPS`), so that its
/// recompile, which only the tier would take, is not made, and the function keeps its first
/// stage, as after the tier's refusal of a chunk with nothing spliced. Logged as a refusal under
/// TRUST_JS_OPT_LOG. A chunk with splices is recompiled, and `generate` refuses it, the second
/// stage keeping the template JIT's code.
pub(crate) fn refuses_unspliced(chunk: &Chunk) -> bool {
    let ops = chunk.jit_ops().len();
    if ops <= MAX_OPTIMIZED_OPS {
        return false;
    }
    #[cfg(test)]
    TEST_REFUSED.with(|refused| {
        refused
            .borrow_mut()
            .push((chunk as *const Chunk as usize, "over the size limit"))
    });
    if log_enabled() {
        eprintln!(
            "[opt] refused {}: over the size limit ({ops} ops, before its recompile)",
            name(chunk)
        );
    }
    true
}

/// `chunk`'s checkpoints have ended (its second stage was built, or its recompile was made), or
/// its wait starts afresh.
pub(crate) fn forget(chunk: &Chunk) {
    if let Some(profile) = ChunkProfile::get(chunk) {
        profile.unsettle();
    }
}

/// A timer for a part of a tier compile that runs on the main thread, under TRUST_JS_OPT_LOG
/// (None otherwise: no clock read): the owner's count for the compile bar takes every such part
/// (the plan's "The owner's answers", 2026-11-10), each logged by [`log_main_thread`].
pub(crate) fn main_thread_timer() -> Option<std::time::Instant> {
    log_enabled().then(std::time::Instant::now)
}

/// Log the main-thread `part` of a tier compile of `chunk` (`recompile`: the bytecode recompile
/// that makes the tier's chunk; `install`: the code's installation after `generate`; `fallback`:
/// the template JIT's compile of a discarded code's fallback), timed from `started`
/// ([`main_thread_timer`]): `[opt] main-thread <part> <name> in <ms> ms`. Nothing without a timer.
pub(crate) fn log_main_thread(part: &str, chunk: &Chunk, started: Option<std::time::Instant>) {
    if let Some(started) = started {
        eprintln!(
            "[opt] main-thread {part} {} in {:.3} ms",
            name(chunk),
            started.elapsed().as_secs_f64() * 1e3
        );
    }
}

/// A timer for the template JIT's compile of `chunk` that starts now (`jit::compile_profiled`)
/// where it is a discarded code's fallback, under TRUST_JS_OPT_LOG: the compile `discard` makes,
/// or the one at a later entry where the discard's was deferred for executable capacity.
pub(crate) fn fallback_timer(chunk: &Chunk) -> Option<std::time::Instant> {
    if !log_enabled() {
        return None;
    }
    let deferred = ChunkProfile::get(chunk).is_some_and(ChunkProfile::fallback_deferred);
    (DISCARDING.with(Cell::get) || deferred).then(std::time::Instant::now)
}

/// The fallback's compile that [`fallback_timer`] timed ended, `deferred` if it was deferred for
/// executable capacity: its main-thread line, and the mark that sends the next compile of the
/// chunk to the log while one is deferred.
pub(crate) fn log_fallback(chunk: &Chunk, started: Option<std::time::Instant>, deferred: bool) {
    if started.is_none() {
        return;
    }
    log_main_thread("fallback", chunk, started);
    if deferred {
        ChunkProfile::of(chunk).set_fallback_deferred(true);
    } else if let Some(profile) = ChunkProfile::get(chunk) {
        profile.set_fallback_deferred(false);
    }
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

/// The tier's code for `chunk`: its snapshot (the exit profile's declines among its feedback),
/// its graph, the graph's verification where the build verifies (`verifies`), the ARM64 code.
/// A chunk over the size limit (`MAX_OPTIMIZED_OPS`) is refused first. Logs the compile with its
/// time and code range, each decline, or the refusal with its reason and pc, under
/// TRUST_JS_OPT_LOG; a graph the verifier rejects is the builder's bug, which panics in tests and
/// debug builds and is refused otherwise.
fn generate(i: &mut Interp, chunk: &Chunk) -> Option<Rc<JitCode>> {
    let started = std::time::Instant::now();
    let compiled = (|| {
        // V8's size limit, before any work (stage 2, step 11).
        if chunk.jit_ops().len() > MAX_OPTIMIZED_OPS {
            return Err(Refusal::whole("over the size limit"));
        }
        // The global names resolve in the running realm (stage 2, step 4).
        let snapshot = snapshot::Snapshot::take_in(chunk, Some(&*i));
        // The graph, and the chunk's Cfg and initialized reads, made once for the verifier and
        // the back end too.
        let build::Built {
            graph,
            cfg,
            initialized,
        } = build::build(chunk, &snapshot)?;
        if log_enabled() {
            for decline in &graph.declines {
                eprintln!(
                    "[opt] declined {} pc {} {:?}: {}",
                    name(chunk),
                    decline.pc,
                    decline.reason,
                    decline.form
                );
            }
        }
        if verifies() {
            #[cfg(test)]
            TEST_VERIFIED.with(|verified| verified.set(verified.get() + 1));
            if let Err(message) = verify::verify(&graph, chunk, &cfg) {
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
        let code = a64::compile(
            chunk,
            &graph,
            &cfg,
            initialized,
            &layout,
            &i.interp_layout.get(),
        )?;
        pin_global_environment(i, &graph);
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
            Some(code)
        }
        Err(refusal) => {
            #[cfg(test)]
            TEST_REFUSED.with(|refused| {
                refused
                    .borrow_mut()
                    .push((chunk as *const Chunk as usize, refusal.reason))
            });
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
            None
        }
    }
}

/// Compile `chunk` with the tier and make the code the chunk's native code: `chunk` is a second
/// stage (a stale chain's rebuilt one among them), or the function's own chunk compiled again
/// because it had nothing to splice, and has no native code yet. Whether the tier accepted it.
pub(crate) fn compile_into(i: &mut Interp, chunk: &Chunk) -> bool {
    release_retired(i);
    // The slot first, as the template JIT's compiles take it: a refusal drops the permit, which
    // leaves the slot cold.
    let Some(permit) = chunk.jit.begin_compile() else {
        if log_enabled() {
            eprintln!("[opt] refused {}: no code slot", name(chunk));
        }
        return false;
    };
    match generate(i, chunk) {
        Some(code) => {
            let started = main_thread_timer();
            permit.commit(Some(code));
            log_main_thread("install", chunk, started);
            true
        }
        None => false,
    }
}

/// Move the call caches' global epoch, so that every way re-resolves (a cached call names the
/// replaced code, which is freed once nothing runs it): `why` names the tier's reason in the
/// log, which counts the moves.
fn move_epoch(why: &str) {
    crate::bytecode::invalidate_call_caches();
    let moves = EPOCH_MOVES.with(|moves| {
        moves.set(moves.get() + 1);
        moves.get()
    });
    if log_enabled() {
        eprintln!("[opt] epoch moved ({why}), {moves} by the tier");
    }
}

/// Past its exit limit (`exit_limit`), the tier's code for `chunk`, whose exit table `table` is,
/// is discarded (see the module's doc), before the exiting activation's VM run; activations
/// still running it finish in it, kept among the interpreter's retired codes. Each record that
/// counted exits adds its site to the exit profile (`flag`), as JSC's tallyFrequentExitSites
/// (bytecode/CodeBlock.cpp:3094-3120), but throws and unsupported operations, which flag nothing
/// (there is nothing to decline); R grows; the template JIT's code takes new entries (a compile
/// deferred for capacity leaves the slot cold, compiled at its next entry); the epoch moves;
/// and the warm-up toward the recompile is armed, unless every exit was unfixable. Under
/// TRUST_JS_OPT_NO_REOPT, or where the tier is off (tests that install its code by hand), the
/// discard is stage 1's fallback for good.
pub(crate) fn discard(i: &mut Interp, chunk: &Chunk, table: &ExitTable) {
    table.discarded.set(true);
    let first = profile::first_stage(chunk);
    let reoptimizing = enabled() && !no_reopt();
    let before = ChunkProfile::get(first).map_or(0, ChunkProfile::discards);
    let r = if reoptimizing {
        ChunkProfile::of(first).add_discard()
    } else {
        before
    };
    let taken: Vec<&ExitRecord> = table
        .exits
        .iter()
        .filter(|exit| exit.count.get() != 0)
        .collect();
    let fixable = taken
        .iter()
        .any(|exit| !matches!(exit.reason, ExitReason::Throw | ExitReason::Unsupported));
    let now = swap_in_template(i, chunk, table);
    if log_enabled() {
        // The exits by site and reason (a site's checks of one reason share its pc).
        let mut sites: Vec<(u32, ExitReason, u32)> = Vec::new();
        for exit in &taken {
            match sites
                .iter_mut()
                .find(|(pc, reason, _)| *pc == exit.pc && *reason == exit.reason)
            {
                Some(site) => site.2 = site.2.saturating_add(exit.count.get()),
                None => sites.push((exit.pc, exit.reason, exit.count.get())),
            }
        }
        sites.sort_by_key(|&(pc, _, _)| pc);
        let sites: Vec<String> = sites
            .iter()
            .map(|(pc, reason, count)| format!("pc {pc} {reason:?} x{count}"))
            .collect();
        eprintln!(
            "[opt] discarded {} after {} exits (limit {}, R {before} -> {r}) for {}: {}",
            name(chunk),
            table.taken.get(),
            table.limit,
            now,
            sites.join(", ")
        );
    }
    if reoptimizing && !no_exit_profile() {
        for exit in &taken {
            if !matches!(exit.reason, ExitReason::Throw | ExitReason::Unsupported) {
                flag(chunk, exit);
            }
        }
    }
    move_epoch("discard");
    // The warm-up runs on a recompile's checkpoint word, which no first-stage schedule uses: a
    // chunk with a record map (a second stage, the tier's own chunk of a function with nothing
    // to splice, a stale chain's recompile).
    if reoptimizing && chunk.feedback_vector().record_map().is_some() {
        if !fixable {
            if log_enabled() {
                eprintln!(
                    "[opt] not reoptimized {}: unfixable exits only",
                    name(chunk)
                );
            }
        } else if let Some(at) = warm_up_word(chunk.jit_runs.get(), r, 0) {
            let profile = ChunkProfile::of(chunk);
            profile.arm(1, at);
            chunk.set_inline_retry_at(at);
            // The first checkpoint compares the feedback with this, the exits' VM runs' records
            // included.
            profile.settle(snapshot::fingerprint(chunk));
        } else if log_enabled() {
            eprintln!(
                "[opt] not reoptimized {}: its warm-up would pass the checkpoint word",
                name(chunk)
            );
        }
    }
    release_retired(i);
}

/// The template JIT's code in place of the discarded `table`'s in `chunk`'s slot, which still
/// holds it (the code runs, so no eviction took it); what new entries now run, for the log.
fn swap_in_template(i: &mut Interp, chunk: &Chunk, table: &ExitTable) -> &'static str {
    let holds = matches!(chunk.jit.get(), Some(Some(code))
        if code.mem_ptr() as usize == table.code_base);
    if !holds {
        return "another code";
    }
    let layout = *i
        .jit_layout
        .get_or_init(|| crate::value::jit_layout(&i.object_proto));
    if !i.interp_layout.get().valid {
        let layout = crate::interpreter::interp_layout(i);
        i.interp_layout.set(layout);
    }
    DISCARDING.with(|discarding| discarding.set(true));
    let outcome = crate::jit::compile_profiled(chunk, &layout, &i.interp_layout.get());
    DISCARDING.with(|discarding| discarding.set(false));
    let (retired, now) = match outcome {
        crate::jit::JitCompileOutcome::Compiled(code) => (
            chunk.jit.replace_resident(Some(Rc::new(code))),
            "the template JIT's code",
        ),
        crate::jit::JitCompileOutcome::Deferred { required_bytes } => {
            // No capacity now: the slot goes cold, and the chunk's next entry compiles it once
            // the bytes fit, as a deferred first compile waits.
            chunk.jit_budget_wait_bytes.set(required_bytes);
            (
                chunk.jit.discard(),
                "the template JIT's code at the next entry",
            )
        }
        crate::jit::JitCompileOutcome::Unavailable => (chunk.jit.replace_resident(None), "the VM"),
    };
    if let Some(retired) = retired {
        #[cfg(test)]
        TEST_DISCARDED.with(|discarded| discarded.borrow_mut().push(retired.mem_ptr() as usize));
        i.opt_retired.push(retired);
    }
    now
}

/// Add `exit`'s site to the exit profile (see `opt::profile`): at the baseline site `chunk`'s
/// record map names for its pc, on that first-stage chunk and marked whether it is a splice's,
/// or at the pc on `chunk` itself; a WrongShape site with the shapes its check expected. Logs
/// each new site.
fn flag(chunk: &Chunk, exit: &ExitRecord) {
    let shapes = match exit.reason {
        ExitReason::WrongShape => exit.shape_check.as_ref().map_or(0, |check| {
            check.shapes.len().min(usize::from(u8::MAX)) as u8
        }),
        // The proof's byte its check expected (stage 2, step 4).
        ExitReason::WrongName => exit.proof,
        _ => 0,
    };
    let (at, pc, inlined) = match profile::baseline_site(chunk, exit.pc as usize) {
        Some((baseline, pc, spliced)) => (baseline, pc as u32, spliced),
        None => (chunk, exit.pc, false),
    };
    let latest = exit.reason == ExitReason::WrongName;
    let new = ChunkProfile::of(at).add_site(pc, exit.reason as u8, inlined, shapes, latest);
    if new && log_enabled() {
        let key = if std::ptr::eq(at, chunk) {
            "own".to_string()
        } else {
            format!(
                "at {} pc {pc}{}",
                name(at),
                if inlined { " (spliced)" } else { "" }
            )
        };
        eprintln!(
            "[opt] flag {} pc {} {:?} {key}",
            name(chunk),
            exit.pc,
            exit.reason
        );
    }
}

/// Whether `chunk`'s checkpoint word counts toward a recompile after a discard (its warm-up is
/// armed): `Interp::try_inline_recompile` hands its due checkpoints to [`reoptimize`].
pub(crate) fn reoptimizing(chunk: &Chunk) -> bool {
    ChunkProfile::get(chunk).is_some_and(|profile| profile.warm_up().is_some())
}

/// A due checkpoint of `chunk`'s warm-up (see the module's doc): the next one is armed (none
/// past the u32 word), and where the feedback settled since the last (or under
/// TRUST_JS_OPT_EAGER at any checkpoint), the tier compiles the chunk again, the snapshot holding
/// the exit profile's flags, and installs the code in place of the template JIT's. Nothing forces
/// the recompile: feedback that keeps changing keeps the template JIT's code (stage 2, step 11).
/// A refusal keeps the template JIT's code. No Realm test: the code's guards compare the Realm's
/// global environment.
pub(crate) fn reoptimize(i: &mut Interp, chunk: &Chunk) {
    let profile = ChunkProfile::of(chunk);
    let Some((checkpoint, at)) = profile.warm_up() else {
        return;
    };
    let runs = chunk.jit_runs.get();
    if runs < at {
        // Another request moved the word (a stale chain's, which its branch declined): the
        // warm-up's checkpoint stands.
        chunk.set_inline_retry_at(at);
        return;
    }
    let r = ChunkProfile::get(profile::first_stage(chunk)).map_or(0, ChunkProfile::discards);
    match warm_up_word(runs, r, checkpoint) {
        Some(next) => {
            profile.arm(checkpoint.saturating_add(1), next);
            chunk.set_inline_retry_at(next);
        }
        None => {
            profile.disarm();
            chunk.set_inline_retry_at(0);
        }
    }
    if waits_at(chunk, checkpoint, " (reoptimization)") {
        return;
    }
    profile.disarm();
    chunk.set_inline_retry_at(0);
    forget(chunk);
    if log_enabled() {
        eprintln!(
            "[opt] reoptimizing {} at checkpoint {checkpoint} (R {r})",
            name(chunk)
        );
    }
    release_retired(i);
    if let Some(code) = generate(i, chunk) {
        let started = main_thread_timer();
        let installed = install(i, chunk, code);
        log_main_thread("install", chunk, started);
        if installed {
            move_epoch("reoptimization");
        }
    }
    release_retired(i);
}

/// Keep the running realm's global environment from address reuse while the heap lives, where
/// `graph`'s code compares a record with it (stage 2, step 4): a global read's code embeds its
/// address, and another environment at that address would pass its check. As the call caches'
/// same-realm proof pins it (`Interp::global_env_pins`): a weak pin, which owns no binding.
fn pin_global_environment(i: &mut Interp, graph: &super::ir::Graph) {
    use super::ir::NodeOp;
    let genv = crate::interpreter::Env::as_ptr(&i.global_env) as usize;
    let embeds = graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, NodeOp::CheckEnvIs { env, .. } if env as usize == genv));
    if embeds
        && !i
            .global_env_pins
            .iter()
            .any(|pin| pin.as_ptr() as usize == genv)
    {
        let pin = crate::interpreter::Env::downgrade(&i.global_env);
        i.global_env_pins.push(pin);
    }
}

/// Make `code`, the tier's recompile of `chunk`, the chunk's native code: in place of the
/// resident code, which joins the retired codes until no activation runs it, or in a cold slot
/// (evicted, or left cold by a deferred discard). An unavailable slot keeps the chunk in the VM.
fn install(i: &mut Interp, chunk: &Chunk, code: Rc<JitCode>) -> bool {
    match chunk.jit.get() {
        Some(Some(_)) => {
            if let Some(replaced) = chunk.jit.replace_resident(Some(code)) {
                i.opt_retired.push(replaced);
            }
            true
        }
        Some(None) => {
            if log_enabled() {
                eprintln!("[opt] refused {}: an unavailable code slot", name(chunk));
            }
            false
        }
        None => match chunk.jit.begin_compile() {
            Some(permit) => {
                permit.commit(Some(code));
                true
            }
            None => {
                if log_enabled() {
                    eprintln!("[opt] refused {}: no code slot", name(chunk));
                }
                false
            }
        },
    }
}

/// What the template JIT's code for `chunk`, compiled now, is to the tier: with the tier on, a
/// second stage's (a chunk with a record map, the tier's own empty-plan chunks among them) is
/// "fallback" when a discard compiles it and "second stage" otherwise, such as the first native
/// entry's compile of a second stage the tier refused, or a loop's borrowed-frame code; None for
/// any other code. The log (`note_template`) and the jitdump's names (`jit::profiler`) mark it.
pub(crate) fn template_class(chunk: &Chunk) -> Option<&'static str> {
    if !enabled() || chunk.feedback_vector().record_map().is_none() {
        return None;
    }
    Some(if DISCARDING.with(Cell::get) {
        "fallback"
    } else {
        "second stage"
    })
}

/// The template JIT compiled `chunk`'s code at `base`, `len` bytes: under TRUST_JS_OPT_LOG, a
/// second stage's (`template_class`) is logged with its code range, as the tier's compiles are,
/// so that a profile's samples tell the two kinds of second-stage code apart (the perf map names
/// both `JS:^`).
pub(crate) fn note_template(chunk: &Chunk, base: *mut u8, len: usize) {
    if !log_enabled() {
        return;
    }
    let Some(why) = template_class(chunk) else {
        return;
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
