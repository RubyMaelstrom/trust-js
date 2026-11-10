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
use super::job::{self, Compiled, Finished, Job, Outcome};
use super::profile::{self, ChunkProfile};
use super::{a64, snapshot};
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
    /// The tier compiles off the main thread (`asynchronous`, where the process can): false for
    /// a test that reads a compile's results at once, which compiles on the main thread (P3
    /// stage 2, step 12).
    pub(crate) asynchronous: bool,
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
    /// The chunks whose compiles off the main thread their install dropped (`install_job`), with
    /// the reasons (stage 2, step 12).
    pub(crate) static TEST_DROPPED: std::cell::RefCell<Vec<(usize, &'static str)>> =
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
    debug_assert!(
        !crate::jit::off_main_compile(),
        "a compile off the main thread reads no test switch (they are the main thread's)"
    );
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

/// The tier's code for `chunk`, compiled now (P3 stage 2, step 12's three parts, run in a row):
/// the main thread's preparation (`prepare`: the snapshot, its exit profile's declines among its
/// feedback, and the chunk's frozen view), the compile (`opt::job`: its graph, the graph's
/// verification where the build verifies (`verifies`), the ARM64 code), and the main thread's
/// link of the code (`finish`). A chunk over the size limit (`MAX_OPTIMIZED_OPS`) is refused
/// first. Logs the compile with its time and code range, each decline, or the refusal with its
/// reason and pc, under TRUST_JS_OPT_LOG, and the preparation as a main-thread part; a graph the
/// verifier rejects is the builder's bug, which panics in tests and debug builds and is refused
/// otherwise. With the code, the timer of its install, which the caller ends (`log_main_thread`'s
/// `install`).
fn generate(i: &mut Interp, chunk: &Chunk) -> Option<(Rc<JitCode>, Option<std::time::Instant>)> {
    let started = std::time::Instant::now();
    let timer = main_thread_timer();
    let (job, pin) = match prepare(i, chunk, compiles_on_another_thread()) {
        Ok(prepared) => prepared,
        Err(refusal) => {
            note_refusal(chunk, &refusal, &Timing::main(started, started.elapsed()));
            return None;
        }
    };
    log_main_thread("snapshot", chunk, timer);
    let finished = run_here(job, chunk);
    let install = main_thread_timer();
    let timing = Timing {
        thread: if compiles_on_another_thread() {
            "a scoped"
        } else {
            "the main"
        },
        compiled: finished.elapsed,
        needed: std::time::Duration::ZERO,
        waited: std::time::Duration::ZERO,
        snapshot: started,
        runs: 1,
    };
    finish(i, chunk, finished, pin, &timing).map(|code| (code, install))
}

/// Where and when a compile ran, for its log line (the plan's point 10): the thread its build
/// and code generation ran on (`the main`, `the compile`), their time there (summed over its
/// runs: a job that found stubs or atoms missing runs again), its wait in the compile thread's
/// queue, and when its snapshot started, so that the line gives the time from the snapshot to
/// the install.
struct Timing {
    thread: &'static str,
    compiled: std::time::Duration,
    /// The part of `compiled` its runs before the last spent: the runs that found stubs or atoms
    /// missing.
    needed: std::time::Duration,
    waited: std::time::Duration,
    snapshot: std::time::Instant,
    runs: u32,
}

impl Timing {
    /// A refusal on the main thread before any compile, `elapsed` after `started`.
    fn main(started: std::time::Instant, elapsed: std::time::Duration) -> Self {
        Timing {
            thread: "the main",
            compiled: elapsed,
            needed: std::time::Duration::ZERO,
            waited: std::time::Duration::ZERO,
            snapshot: started,
            runs: 0,
        }
    }

    /// The log's fields after a compile line's own: `on <thread> thread, queued <ms> ms,
    /// installed <ms> ms after its snapshot` (and, past one run, `, <n> runs, <ms> ms before the
    /// last`: the time of the runs that found stubs or atoms missing).
    fn fields(&self) -> String {
        format!(
            "on {} thread, queued {:.3} ms, installed {:.3} ms after its snapshot{}",
            self.thread,
            self.waited.as_secs_f64() * 1e3,
            self.snapshot.elapsed().as_secs_f64() * 1e3,
            if self.runs > 1 {
                format!(
                    ", {} runs, {:.3} ms before the last",
                    self.runs,
                    self.needed.as_secs_f64() * 1e3
                )
            } else {
                String::new()
            }
        )
    }
}

/// A compile's main-thread preparation (the plan's point 2): V8's size limit, the snapshot (its
/// global names resolved in the running realm, stage 2 step 4), the chunk's frozen view, the
/// layouts, the code generator's switches; with `off_thread`, the stub table's snapshot, and the
/// code mapped by the compile. With the job, a pin of the running realm's global environment,
/// where the snapshot reads its global names or the compile runs off the main thread, which keeps
/// its address from reuse until the install (the plan's point 7): the install compares it with
/// the running realm's, and keeps the pin where the code checks it (`pin_global_environment`).
fn prepare(
    i: &mut Interp,
    chunk: &Chunk,
    off_thread: bool,
) -> Result<(Job, Option<crate::interpreter::WeakEnv>), Refusal> {
    // V8's size limit, before any work (stage 2, step 11).
    if chunk.jit_ops().len() > MAX_OPTIMIZED_OPS {
        return Err(Refusal::whole("over the size limit"));
    }
    // The global names resolve in the running realm (stage 2, step 4).
    let snapshot = snapshot::Snapshot::take_in(chunk, Some(&*i));
    let view = chunk.freeze(!off_thread);
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
    let genv = crate::interpreter::Env::as_ptr(&i.global_env) as usize;
    let pin = (off_thread || snapshot.reads_global_names_of(genv))
        .then(|| crate::interpreter::Env::downgrade(&i.global_env));
    let job = Job {
        view,
        snapshot,
        options: a64::Options::from_env(),
        switches: a64::Switches::read(chunk),
        layout,
        ilayout: i.interp_layout.get(),
        verifies: verifies(),
        genv,
        stubs: off_thread.then(|| crate::jit::stub_table_snapshot(&[])),
        stub_copies: Vec::new(),
        maps: off_thread,
        #[cfg(test)]
        test_addresses: crate::jit::TestAddresses::capture(),
    };
    Ok((job, pin))
}

/// Whether a compile runs off the main thread: in test builds under
/// TRUST_JS_OPT_SCOPED_COMPILE, on a scoped thread joined at once (`run_here`), the check that it
/// reads none of the main thread's state.
fn compiles_on_another_thread() -> bool {
    cfg!(test) && env_flag!("TRUST_JS_OPT_SCOPED_COMPILE")
}

/// Run `job`, a compile of `chunk`, until it needs no stub the main thread has not assembled and
/// no atom it has not made: on this thread, or where `compiles_on_another_thread` on a scoped
/// thread joined at once. What it found missing is made here, on the main thread (`provide`,
/// logged as a main-thread part), and the job runs again.
fn run_here(mut job: Job, chunk: &Chunk) -> Finished {
    loop {
        let finished = if compiles_on_another_thread() {
            std::thread::scope(|scope| {
                let compile = std::thread::Builder::new()
                    .name("trust-js-opt-scoped".into())
                    .stack_size(COMPILE_STACK)
                    .spawn_scoped(scope, move || {
                        let _off = crate::jit::OffMainCompile::enter();
                        job::compile(job)
                    })
                    .expect("a scoped compile thread");
                compile
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
        } else {
            job::compile(job)
        };
        let Outcome::Needs { stubs } = &finished.outcome else {
            return finished;
        };
        let timer = main_thread_timer();
        let stubs = stubs.clone();
        job = finished.job;
        provide(&mut job, &stubs);
        log_main_thread("stubs", chunk, timer);
    }
}

/// What a compile off the main thread found missing (main thread): the stubs `stubs` assembled
/// (those whose assembly failed kept per chunk by its next run) and the stub table's snapshot
/// taken again, the names' atoms made.
fn provide(job: &mut Job, stubs: &[u32]) {
    if !stubs.is_empty() {
        let failed = assemble_stubs(job, stubs);
        job.stub_copies.extend(failed);
        job.stubs = Some(crate::jit::stub_table_snapshot(&job.stub_copies));
    }
    job.view.make_missing_keys();
}

/// The stack of a thread that compiles: the builder and the allocator iterate, so this bounds
/// only their callees (V8's background compile threads' stacks are its platform's default).
pub(crate) const COMPILE_STACK: usize = 16 << 20;

/// Assemble the stubs `keys` that `job`'s emission found missing (main thread): those whose
/// assembly failed, which its next run keeps per chunk.
fn assemble_stubs(job: &Job, keys: &[u32]) -> Vec<u32> {
    let direct =
        (job.switches.fast & (1 << 20) != 0 && job.switches.direct_shared_context).then(|| {
            (
                job.view.jit_inline_attempted_off(),
                job.view.jit_runs_off(),
                job.view.jit_inline_retry_at_off(),
            )
        });
    crate::jit::assemble_stubs(keys, &job.layout, &job.ilayout, direct)
}

/// A refusal of `chunk`'s compile, timed as `timing` says: the tests' record and the log's line,
/// `(<ms> ms)` the time on the compile's thread, the thread and the waits after it.
fn note_refusal(chunk: &Chunk, refusal: &Refusal, timing: &Timing) {
    #[cfg(test)]
    TEST_REFUSED.with(|refused| {
        refused
            .borrow_mut()
            .push((chunk as *const Chunk as usize, refusal.reason))
    });
    if log_enabled() {
        let ms = timing.compiled.as_secs_f64() * 1e3;
        match refusal.pc {
            Some(pc) => eprintln!(
                "[opt] refused {} at pc {pc}: {} ({ms:.3} ms) {}",
                name(chunk),
                refusal.reason,
                timing.fields()
            ),
            None => eprintln!(
                "[opt] refused {}: {} ({ms:.3} ms) {}",
                name(chunk),
                refusal.reason,
                timing.fields()
            ),
        }
    }
}

/// The main thread's end of a compile of `chunk` (`finished`): the verifier's and the builder's
/// log lines, and for a compiled code its mapping where the compile could not map it (after the
/// main thread's reclaim), the instruction barrier where another thread mapped it, its link
/// (`a64::link`: the exit table registered, the JitCode, the perf map's entry), the realm's pin
/// where the code compares its global environment, and the log's line; or the refusal's.
fn finish(
    i: &mut Interp,
    chunk: &Chunk,
    finished: Finished,
    pin: Option<crate::interpreter::WeakEnv>,
    timing: &Timing,
) -> Option<Rc<JitCode>> {
    let Finished {
        outcome,
        declines,
        verified,
        verify_failure,
        embeds_genv,
        ..
    } = finished;
    #[cfg(test)]
    if verified {
        TEST_VERIFIED.with(|count| count.set(count.get() + 1));
    }
    let _ = verified;
    if log_enabled() {
        for decline in &declines {
            eprintln!(
                "[opt] declined {} pc {} {:?}: {}",
                name(chunk),
                decline.pc,
                decline.reason,
                decline.form
            );
        }
        if let Some(message) = &verify_failure {
            eprintln!("[opt] verify {} failed: {message}", name(chunk));
        }
    }
    let compiled = match outcome {
        Outcome::Compiled(compiled) => compiled,
        Outcome::Refused(refusal) => {
            note_refusal(chunk, &refusal, timing);
            return None;
        }
        Outcome::Needs { .. } => unreachable!("a run ends with what it needs made"),
    };
    let Compiled {
        assembled,
        executable,
        nodes,
    } = *compiled;
    let executable = match executable {
        Some(executable) => {
            // Mapped by another thread: this core runs none of the code's stale instructions
            // (JSC's crossModifyingCodeFence; the writer's flush synchronizes only the writer).
            crate::jit::sync_instruction_stream();
            executable.0
        }
        None => match crate::jit::ExecutableBuffer::from_bytes(assembled.bytes()) {
            Some(executable) => executable,
            None => {
                note_refusal(chunk, &Refusal::whole("no executable memory"), timing);
                return None;
            }
        },
    };
    let code = a64::link(assembled, executable, chunk);
    if embeds_genv {
        if let Some(pin) = pin {
            pin_global_environment(i, pin);
        }
    }
    if log_enabled() {
        eprintln!(
            "[opt] compiled {} ({} ops, {nodes} nodes, {} bytes) in {:.3} ms \
             at {:#x} {:#x} {}",
            name(chunk),
            chunk.jit_ops().len(),
            code.code_bytes(),
            timing.compiled.as_secs_f64() * 1e3,
            code.mem_ptr() as usize,
            code.code_bytes(),
            timing.fields()
        );
    }
    #[cfg(test)]
    TEST_COMPILED.with(|compiled| compiled.borrow_mut().push(chunk as *const Chunk as usize));
    Some(code)
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
        Some((code, install)) => {
            permit.commit(Some(code));
            log_main_thread("install", chunk, install);
            true
        }
        None => false,
    }
}

// ---------------------------------------------------------------------------------------------
// Compiling off the main thread (stage 2, step 12; the plan's points 3-10)
// ---------------------------------------------------------------------------------------------

/// Whether the tier compiles off the main thread now (the plan's point 8): with more than one
/// CPU available to the process and a compile thread started, unless TRUST_JS_OPT_SYNC is set,
/// or, in test builds, unless the test's switches say otherwise (`TestSwitches::asynchronous`:
/// a test that reads a compile's results at once compiles on the main thread, V8's
/// --single-threaded, SpiderMonkey's runtime switch, JSC's useConcurrentJIT=false).
pub(crate) fn asynchronous() -> bool {
    #[cfg(test)]
    if let Some(switches) = test_switches() {
        return switches.asynchronous && compile_thread_usable();
    }
    !env_flag!("TRUST_JS_OPT_SYNC") && compile_thread_usable()
}

/// TRUST_JS_OPT_SYNC is unset (the environment's mode, which tests' switches copy).
#[cfg(test)]
pub(crate) fn asynchronous_by_environment() -> bool {
    !env_flag!("TRUST_JS_OPT_SYNC")
}

/// Whether a compile can run on the compile thread: more than one CPU available (one gains
/// nothing from it, SpiderMonkey's rule, jit/Ion.cpp:2110-2118) and the thread started (a
/// failed start compiles on the main thread, as the watchdog's degrades).
fn compile_thread_usable() -> bool {
    more_than_one_cpu() && super::compiler::available()
}

/// More than one CPU available to the process, by `std::thread::available_parallelism`, which
/// honors the affinity mask (a process confined to one CPU of many counts one, where
/// SpiderMonkey counts the configured CPUs): read once a process.
fn more_than_one_cpu() -> bool {
    static MANY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *MANY.get_or_init(|| std::thread::available_parallelism().is_ok_and(|n| n.get() > 1))
}

/// How a tier compile ends, which its caller decided (stage 2, step 12): run now on the main
/// thread, or at the install of a compile off it.
pub(crate) enum Landing {
    /// A function's recompile (its second stage, the tier's chunk of a function with nothing to
    /// splice, a stale chain's recompile): published as `Interp::publish_recompile` says.
    Publish(crate::interpreter::Publication),
    /// A discarded code's recompile (`reoptimize`): the code in place of the template JIT's and
    /// the epoch moved; the function, and the warm-up's next checkpoint and its run count, armed
    /// again where the result is dropped.
    Reoptimize {
        func: std::rc::Weak<crate::ast::Function>,
        warm_up: Option<(u8, u32)>,
    },
}

/// An engine's compiles running off the main thread (`Interp::opt_pending`), which keep their
/// chunks alive (the plan's point 7), and the queue the compile thread returns them on. An
/// engine dropped with compiles pending cancels them (the plan's point 9): the compile thread
/// skips them, or drops their results.
#[derive(Default)]
pub(crate) struct PendingCompiles {
    results: Option<std::sync::Arc<super::compiler::Results>>,
    jobs: Vec<PendingJob>,
    next: u64,
}

impl PendingCompiles {
    /// The compiles pending (tests).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.jobs.len()
    }

    /// The heap bytes the pending compiles' records hold, each chunk they keep handed to
    /// `chunk` (the memory census).
    pub(crate) fn measure(&self, mut chunk: impl FnMut(&Rc<Chunk>)) -> usize {
        for job in &self.jobs {
            chunk(&job.chunk);
        }
        self.jobs.capacity() * std::mem::size_of::<PendingJob>()
    }

    /// Cancel every pending compile.
    pub(crate) fn cancel(&mut self) {
        for job in self.jobs.drain(..) {
            job.cancelled
                .store(true, std::sync::atomic::Ordering::Release);
        }
    }
}

impl Drop for PendingCompiles {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// A compile running off the main thread, as its engine keeps it.
struct PendingJob {
    id: u64,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// The chunk the code goes into (its record map keeps the function's first stage).
    chunk: Rc<Chunk>,
    /// The chunk whose checkpoint started it, marked compiling and its checkpoint disarmed until
    /// the install, and the word that checkpoint had armed.
    asking: Rc<Chunk>,
    armed: u32,
    landing: Landing,
    /// The pin of the running realm's global environment at the snapshot, and its address.
    pin: Option<crate::interpreter::WeakEnv>,
    genv: usize,
    /// The revisions of the exit profiles the snapshot read (`revisions`).
    revisions: u64,
    /// The compile's runs and their time on the compile thread and in its queue so far.
    timing: Timing,
}

/// The revisions of the exit profiles a snapshot of `chunk` reads (its own and its record map's
/// targets'): a discard that flags sites or counts R changes them.
fn revisions(chunk: &Chunk) -> u64 {
    let of = |chunk: &Chunk| ChunkProfile::get(chunk).map_or(0, |p| u64::from(p.revision()));
    of(chunk)
        + chunk
            .feedback_vector()
            .record_map()
            .map_or(0, |map| map.targets().iter().map(|target| of(target)).sum())
}

/// Whether a compile `chunk`'s checkpoint started runs off the main thread (its profile says
/// compiling): no checkpoint of it compiles meanwhile.
pub(crate) fn compiling(chunk: &Chunk) -> bool {
    ChunkProfile::get(chunk).is_some_and(ChunkProfile::compiling)
}

/// Compile `chunk`, a function's recompile, with the tier and publish it as `publication` says
/// (`Interp::publish_recompile`): now, where the tier compiles on the main thread
/// (`asynchronous`), or at the install of a compile off it, where the publication then moves
/// (published before its code exists, the chunk would run in the bytecode VM). `asking` is the
/// chunk whose checkpoint asked.
pub(crate) fn compile_recompile(
    i: &mut Interp,
    chunk: Rc<Chunk>,
    asking: &Chunk,
    publication: crate::interpreter::Publication,
) {
    if asynchronous() {
        if let Some(asking_rc) = publication.asking.clone() {
            match start_job(i, chunk, asking_rc, Landing::Publish(publication)) {
                Ok(()) => return,
                Err((chunk, Landing::Publish(publication))) => {
                    i.publish_recompile(&publication, asking, chunk, false);
                    return;
                }
                Err(_) => unreachable!("a publication's landing"),
            }
        }
    }
    let accepted = compile_into(i, &chunk);
    i.publish_recompile(&publication, asking, chunk, accepted);
}

/// Start a compile of `chunk` off the main thread (the plan's points 3 and 4), its ending
/// `landing` at the install: its preparation now (the snapshot, logged as a main-thread part),
/// the job queued for the compile thread, and `asking`, the chunk whose checkpoint asked, marked
/// compiling with no checkpoint armed until then (V8's tiering_in_progress, SpiderMonkey's
/// isIonCompilingOffThread). The chunk and the landing back where the preparation refused the
/// compile (logged), which the caller then ends as a refusal.
#[allow(clippy::result_large_err)]
fn start_job(
    i: &mut Interp,
    chunk: Rc<Chunk>,
    asking: Rc<Chunk>,
    landing: Landing,
) -> Result<(), (Rc<Chunk>, Landing)> {
    release_retired(i);
    let started = std::time::Instant::now();
    let timer = main_thread_timer();
    let (job, pin) = match prepare(i, &chunk, true) {
        Ok(prepared) => prepared,
        Err(refusal) => {
            note_refusal(&chunk, &refusal, &Timing::main(started, started.elapsed()));
            return Err((chunk, landing));
        }
    };
    let pending = &mut i.opt_pending;
    pending.next += 1;
    let id = pending.next;
    let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let results = pending.results.get_or_insert_with(Default::default).clone();
    let genv = job.genv;
    if let Err(job) = super::compiler::enqueue(id, job, cancelled.clone(), results, i.poll.clone())
    {
        // No compile thread after all: compiled here, and ended now.
        log_main_thread("snapshot", &chunk, timer);
        let finished = run_here(job, &chunk);
        let timing = Timing {
            thread: "the main",
            compiled: finished.elapsed,
            needed: std::time::Duration::ZERO,
            waited: std::time::Duration::ZERO,
            snapshot: started,
            runs: 1,
        };
        let install = main_thread_timer();
        let code = finish(i, &chunk, finished, pin, &timing);
        land(i, chunk, &asking, landing, code, install);
        return Ok(());
    }
    let armed = asking.inline_retry_at.get();
    asking.set_inline_retry_at(0);
    ChunkProfile::of(&asking).set_compiling(true);
    let revisions = revisions(&chunk);
    // The main thread's part of starting it: the snapshot, the frozen view, the hand-off.
    log_main_thread("snapshot", &chunk, timer);
    i.opt_pending.jobs.push(PendingJob {
        id,
        cancelled,
        chunk,
        asking,
        armed,
        landing,
        pin,
        genv,
        revisions,
        timing: Timing {
            thread: "the compile",
            compiled: std::time::Duration::ZERO,
            needed: std::time::Duration::ZERO,
            waited: std::time::Duration::ZERO,
            snapshot: started,
            runs: 0,
        },
    });
    Ok(())
}

/// The end of a compile of `chunk` whose code is `code` (None: refused), as `landing` says: the
/// publication, or the reoptimization's install and epoch move; `install` times the main
/// thread's part (`[opt] main-thread install`).
fn land(
    i: &mut Interp,
    chunk: Rc<Chunk>,
    asking: &Chunk,
    landing: Landing,
    code: Option<Rc<JitCode>>,
    install: Option<std::time::Instant>,
) {
    match landing {
        Landing::Publish(publication) => {
            let accepted = code.is_some_and(|code| match chunk.jit.begin_compile() {
                Some(permit) => {
                    permit.commit(Some(code));
                    log_main_thread("install", &chunk, install);
                    true
                }
                None => {
                    if log_enabled() {
                        eprintln!("[opt] refused {}: no code slot", name(&chunk));
                    }
                    false
                }
            });
            i.publish_recompile(&publication, asking, chunk, accepted);
        }
        Landing::Reoptimize { .. } => {
            if let Some(code) = code {
                let installed = install_code(i, &chunk, code);
                log_main_thread("install", &chunk, install);
                if installed {
                    move_epoch("reoptimization");
                }
            }
            release_retired(i);
        }
    }
}

/// Install the compiles of `i` that finished off the main thread (the plan's point 5), at a
/// poll of generated code that found the install bit set (`jit_interrupt`, V8's
/// INSTALL_MAGLEV_CODE interrupt, SpiderMonkey's attach): each result whose job is still pending
/// is checked (`dropped`), then linked and landed as its caller decided, its code run after the
/// instruction barrier (`finish`); a result whose job found stubs or atoms missing has them made
/// here and runs again; a panic on the compile thread is raised here.
pub(crate) fn install_finished(i: &mut Interp) {
    if !i.poll.take_install_request() {
        return;
    }
    let Some(results) = i.opt_pending.results.clone() else {
        return;
    };
    for done in results.take() {
        install_one(i, done, &results);
    }
}

fn install_one(
    i: &mut Interp,
    done: super::compiler::Done,
    results: &std::sync::Arc<super::compiler::Results>,
) {
    let Some(at) = i.opt_pending.jobs.iter().position(|job| job.id == done.id) else {
        return;
    };
    let finished = match done.finished {
        Ok(finished) => finished,
        Err(panic) => {
            let job = i.opt_pending.jobs.remove(at);
            undo(&job);
            std::panic::resume_unwind(panic);
        }
    };
    {
        let timing = &mut i.opt_pending.jobs[at].timing;
        timing.compiled += finished.elapsed;
        timing.waited += done.waited;
        timing.runs += 1;
    }
    if let Outcome::Needs { stubs } = &finished.outcome {
        i.opt_pending.jobs[at].timing.needed += finished.elapsed;
        let stubs = stubs.clone();
        let chunk = i.opt_pending.jobs[at].chunk.clone();
        let timer = main_thread_timer();
        let mut job = finished.job;
        provide(&mut job, &stubs);
        let pending = &i.opt_pending.jobs[at];
        let requeued = super::compiler::enqueue(
            pending.id,
            job,
            pending.cancelled.clone(),
            results.clone(),
            i.poll.clone(),
        );
        log_main_thread("stubs", &chunk, timer);
        if let Err(job) = requeued {
            // No compile thread after all: run here, and install now.
            let finished = run_here(job, &chunk);
            let job = i.opt_pending.jobs.remove(at);
            install_job(i, job, finished);
        }
        return;
    }
    let job = i.opt_pending.jobs.remove(at);
    install_job(i, job, finished);
}

/// Install `finished`, `job`'s result, or drop it where a check fails (`dropped`).
fn install_job(i: &mut Interp, job: PendingJob, finished: Finished) {
    let install = main_thread_timer();
    ChunkProfile::of(&job.asking).set_compiling(false);
    if let Some(reason) = dropped(i, &job) {
        #[cfg(test)]
        TEST_DROPPED.with(|dropped| {
            dropped
                .borrow_mut()
                .push((Rc::as_ptr(&job.chunk) as usize, reason))
        });
        if log_enabled() {
            eprintln!(
                "[opt] dropped {}: {reason} (compiled in {:.3} ms {})",
                name(&job.chunk),
                job.timing.compiled.as_secs_f64() * 1e3,
                job.timing.fields()
            );
        }
        undo(&job);
        // The result goes here, its executable pages with it.
        drop(finished);
        log_main_thread("install", &job.chunk, install);
        return;
    }
    let PendingJob {
        chunk,
        asking,
        landing,
        pin,
        timing,
        ..
    } = job;
    let code = finish(i, &chunk, finished, pin, &timing);
    if code.is_none() {
        // A refusal's ending: its time at the install is the main thread's too.
        let refused = chunk.clone();
        land(i, chunk, &asking, landing, code, None);
        log_main_thread("install", &refused, install);
        return;
    }
    land(i, chunk, &asking, landing, code, install);
}

/// What no run-time guard of the code covers, checked at its install (the plan's point 5): why
/// `job`'s result is dropped, if it is. The realm whose global environment the snapshot pinned
/// is the running one; no discard flagged sites or counted R since the snapshot (`revisions`);
/// the function lives; the chunk is still the one to install into (the asking chunk not
/// superseded, no recompile published meanwhile, a reoptimized chunk not superseded). V8 drops a
/// job whose dependencies fail "s.t. we may reattempt compilation"
/// (maglev/maglev-compiler.cc:359-368): so does `undo`.
fn dropped(i: &Interp, job: &PendingJob) -> Option<&'static str> {
    if job.pin.as_ref().is_some_and(|pin| pin.strong_count() == 0)
        || crate::interpreter::Env::as_ptr(&i.global_env) as usize != job.genv
    {
        return Some("another realm runs");
    }
    if revisions(&job.chunk) != job.revisions {
        return Some("a discard flagged its sites since its snapshot");
    }
    match &job.landing {
        Landing::Publish(publication) => {
            let Some(func) = publication.func.upgrade() else {
                return Some("its function was collected");
            };
            let runtime = func.runtime();
            if !runtime.heap.is_alive() {
                return Some("its function was collected");
            }
            if job.asking.superseded() {
                return Some("its chunk was superseded");
            }
            if publication.slot(&runtime).get().is_some() {
                return Some("a recompile was published meanwhile");
            }
        }
        Landing::Reoptimize { func, .. } => {
            let Some(func) = func.upgrade() else {
                return Some("its function was collected");
            };
            if !func.runtime().heap.is_alive() {
                return Some("its function was collected");
            }
            if job.chunk.superseded() {
                return Some("its chunk was superseded");
            }
        }
    }
    None
}

/// After a dropped result, the function may tier up again: its asking chunk's checkpoint (or a
/// reoptimization's warm-up) armed again as it was, a chain recompile asked for again.
fn undo(job: &PendingJob) {
    ChunkProfile::of(&job.asking).set_compiling(false);
    match &job.landing {
        Landing::Publish(publication) => {
            job.asking.set_inline_retry_at(job.armed);
            if publication.chain() {
                job.asking.reoptimize.set(true);
            }
        }
        Landing::Reoptimize { warm_up, .. } => {
            if let Some((checkpoint, at)) = *warm_up {
                ChunkProfile::of(&job.chunk).arm(checkpoint, at);
                job.chunk.set_inline_retry_at(at);
            }
        }
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
/// global environment. Off the main thread (`asynchronous`), the recompile is installed at its
/// job's install, `chunk` (its Rc, `chunk_rc`, kept meanwhile) marked compiling until then; a
/// dropped result arms the warm-up's next checkpoint again (`undo`). `func` is the chunk's
/// function.
pub(crate) fn reoptimize(
    i: &mut Interp,
    chunk: &Chunk,
    func: std::rc::Weak<crate::ast::Function>,
    chunk_rc: Option<Rc<Chunk>>,
) {
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
    let warm_up = match warm_up_word(runs, r, checkpoint) {
        Some(next) => {
            profile.arm(checkpoint.saturating_add(1), next);
            chunk.set_inline_retry_at(next);
            Some((checkpoint.saturating_add(1), next))
        }
        None => {
            profile.disarm();
            chunk.set_inline_retry_at(0);
            None
        }
    };
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
    if asynchronous() {
        if let Some(chunk_rc) = chunk_rc {
            // A refusal at the preparation keeps the template JIT's code, as one at the compile.
            let landing = Landing::Reoptimize { func, warm_up };
            if start_job(i, chunk_rc.clone(), chunk_rc, landing).is_err() {
                release_retired(i);
            }
            return;
        }
    }
    let _ = (func, warm_up);
    release_retired(i);
    if let Some((code, started)) = generate(i, chunk) {
        let installed = install_code(i, chunk, code);
        log_main_thread("install", chunk, started);
        if installed {
            move_epoch("reoptimization");
        }
    }
    release_retired(i);
}

/// Keep the realm's global environment, `pin`'s, from address reuse while the heap lives, where
/// a code compares a record with it (stage 2, step 4): a global read's code embeds its address,
/// and another environment at that address would pass its check. As the call caches' same-realm
/// proof pins it (`Interp::global_env_pins`): a weak pin, which owns no binding. The pin was
/// taken with the snapshot (`prepare`), so that the realm the code checks is the snapshot's (the
/// plan's step 12, point 7).
fn pin_global_environment(i: &mut Interp, pin: crate::interpreter::WeakEnv) {
    if !i
        .global_env_pins
        .iter()
        .any(|kept| kept.as_ptr() == pin.as_ptr())
    {
        i.global_env_pins.push(pin);
    }
}

/// Make `code`, the tier's recompile of `chunk`, the chunk's native code: in place of the
/// resident code, which joins the retired codes until no activation runs it, or in a cold slot
/// (evicted, or left cold by a deferred discard). An unavailable slot keeps the chunk in the VM.
fn install_code(i: &mut Interp, chunk: &Chunk, code: Rc<JitCode>) -> bool {
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
