//! A tier compile as a job (P3 stage 2, step 12; the plan's "Step 12, compiling off the main
//! thread"): its inputs ([`Job`]), made on the main thread with the snapshot (`driver::prepare`),
//! the compile itself ([`compile`]: the build, the verification, register allocation, emission
//! and, where the budget allows without the main thread's reclaim, the copy into executable
//! memory), and its outputs ([`Finished`]), which the main thread installs (`driver`). The inputs
//! and outputs are `Send`, and the compile reads nothing else: no chunk, cache, record, heap
//! object or interpreter (their addresses travel as words to embed), no thread-local (the test
//! switches, counters and stub table are the main thread's, read into the job), so that it runs
//! on any thread, as V8's Maglev runs Execute (maglev/maglev-compiler.cc:90-313) and
//! SpiderMonkey its CompileBackEnd (jit/Ion.cpp:1750-1771). The process's shared stubs are the
//! exception it takes a lock for: a stub the job's snapshot of their table lacks is looked up
//! again in the table and made there where still missing (P3 stage 2, step 17), as JSC's compiler
//! threads make a missing thunk (JITThunks::ctiStubImpl, jit/JITThunks.cpp:189-222), so that a
//! job runs once.

use super::a64::{self, Assembled, Options, Stubs, Switches};
use super::ir::{Decline, Refusal};
use super::snapshot::{Raw, Snapshot};
use super::view::FrozenChunk;
use super::{build, verify};
use crate::jit::{ExecutableBuffer, MissingStubs, StubTableSnapshot};
use std::sync::Arc;

/// A job's memory (P3 stage 2, step 16, point 5): its frozen view and its snapshot's raw copy,
/// whose vectors the next job's preparation fills again.
pub(crate) type JobMemory = (FrozenChunk, Raw);

/// What a compile inlines (P3 stage 3, step 2; `opt::oracle`): the sites, and each callee chunk's
/// frozen view and raw copy, by `build::InlineWay::chunk`, made on the main thread in job memory
/// as the compiled chunk's are.
pub(crate) struct InlineJob {
    pub(crate) sites: Vec<build::InlineSite>,
    pub(crate) callees: Vec<JobMemory>,
}

/// A compile's inputs (see the module's doc).
pub(crate) struct Job {
    /// The chunk's frozen view.
    pub(crate) view: FrozenChunk,
    /// Its feedback's raw copy, taken at the snapshot, which the compile decodes (P3 stage 2,
    /// step 16).
    pub(crate) snapshot: Raw,
    /// The callees the oracle inlines (TRUST_JS_OPT_INLINE=graph; P3 stage 3, step 2).
    pub(crate) inlining: Option<InlineJob>,
    pub(crate) options: Options,
    pub(crate) switches: Switches,
    pub(crate) layout: crate::value::JitLayout,
    pub(crate) ilayout: crate::interpreter::InterpLayout,
    /// Whether the build verifies its graph (`driver::verifies`).
    pub(crate) verifies: bool,
    /// The running realm's global environment at the snapshot (`Env::as_ptr`), whose identity
    /// the code may check (a word).
    pub(crate) genv: usize,
    /// The main thread's stub table at the snapshot, and the table, for a stub the snapshot
    /// lacks (P3 stage 2, step 17); None where the compile runs on the main thread, which finds
    /// or makes each stub in its table at once, as the template JIT does.
    pub(crate) stubs: Option<Arc<StubTableSnapshot>>,
    /// Whether the compile maps its code itself (off the main thread; the main thread's
    /// synchronous compile maps at the install, as before).
    pub(crate) maps: bool,
    /// Whether the compile reads the clock for its time (`Finished::elapsed`): under
    /// TRUST_JS_OPT_LOG only, the log's (stage 2, step 16).
    pub(crate) timed: bool,
    /// The main thread's addresses of the counters test builds' templates embed.
    #[cfg(test)]
    pub(crate) test_addresses: crate::jit::TestAddresses,
    /// The main thread's perf map capture, which lists the stubs the compile makes (step 17).
    #[cfg(test)]
    pub(crate) test_perf_map: crate::jit::profiler::TestPerfMap,
}

/// What a compile made. It runs once: no stub or atom it needs is missing (the frozen view
/// carries every name's key, P3 stage 2, step 16, and the emission makes a missing stub, step 17).
pub(crate) enum Outcome {
    /// The code, and its nodes' count (the log's).
    Compiled(Box<Compiled>),
    Refused(Refusal),
}

/// A compiled code, before its install.
pub(crate) struct Compiled {
    pub(crate) assembled: Assembled,
    /// The code's executable memory, where the compile mapped it.
    pub(crate) executable: Option<SendExecutable>,
    pub(crate) nodes: usize,
}

/// An executable mapping made on the compile thread, handed to the main thread with its result:
/// the mapping and its reservation are process-wide (the code region's lock, the budget's
/// atomics), and nothing runs the code before the main thread installs it.
pub(crate) struct SendExecutable(pub(crate) ExecutableBuffer);

// SAFETY: an ExecutableBuffer owns a mapping (or the region's pages under its lock) and a
// reservation of the process-wide budget; neither is tied to the thread that made it, and its
// drop frees them through the same process-wide paths from any thread.
unsafe impl Send for SendExecutable {}

/// A compile's outputs (see the module's doc).
pub(crate) struct Finished {
    pub(crate) outcome: Outcome,
    /// The stubs its emission met missing from its snapshot of the stub table, which it found or
    /// made in the table, or kept as copies (P3 stage 2, step 17): the log's and the tests'.
    pub(crate) stubs: MissingStubs,
    /// The builder's declines, which the install logs.
    pub(crate) declines: Vec<Decline>,
    /// The accesses rule K narrowed, which the install logs.
    pub(crate) narrowed: Vec<super::ir::Narrowed>,
    /// Whether a graph was verified, and the verifier's message where it failed (release).
    pub(crate) verified: bool,
    pub(crate) verify_failure: Option<String>,
    /// The compile's time on its thread, where it was timed (`Job::timed`; zero otherwise).
    pub(crate) elapsed: std::time::Duration,
    /// The job's memory, its frozen view and its snapshot's raw copy, given back for the next
    /// job's preparation to fill again (P3 stage 2, step 16, point 5), and its callees' (P3
    /// stage 3, step 2).
    pub(crate) memory: JobMemory,
    pub(crate) callee_memory: Vec<JobMemory>,
    /// What the inliner did with each inlined site's ways, for the log (step 2).
    pub(crate) inlines: Vec<build::InlineNote>,
    /// The callee chunks the code's inlined frames run, by `InlinedFrame::chunk`: their indices
    /// among the oracle's (`oracle::Held::chunks`).
    pub(crate) frame_chunks: Vec<u32>,
    /// Whether the code compares a record with the realm's global environment (`genv`), which
    /// the install then pins (stage 2, step 4).
    pub(crate) embeds_genv: bool,
}

/// Compile `job` (see the module's doc): reads nothing but the job and process constants. Its
/// snapshot's views are decoded here, on the compile's thread, from the raw copy the main thread
/// took (P3 stage 2, step 16: SpiderMonkey's WarpBuilder interprets its oracle's stub copies off
/// thread).
pub(crate) fn compile(job: Job) -> Finished {
    let started = job.timed.then(std::time::Instant::now);
    #[cfg(test)]
    let _addresses = crate::jit::TestAddresses::install(job.test_addresses);
    #[cfg(test)]
    let _perf_map = job.test_perf_map.clone().install();
    let mut declines = Vec::new();
    let mut narrowed = Vec::new();
    let mut met = MissingStubs::default();
    let mut verified = false;
    let mut verify_failure = None;
    let mut embeds_genv = false;
    let mut inlines = Vec::new();
    let mut frame_chunks = Vec::new();
    let outcome = (|| {
        let snapshot = Snapshot::decode(&job.snapshot, &job.view);
        // The inlined callees' snapshots, decoded here as the compiled chunk's (step 2).
        let callee_memory = job
            .inlining
            .as_ref()
            .map_or(&[][..], |inlining| &inlining.callees[..]);
        let callee_snapshots: Vec<Snapshot> = callee_memory
            .iter()
            .map(|(view, raw)| Snapshot::decode(raw, view))
            .collect();
        let callees: Vec<build::Callee<'_>> = callee_memory
            .iter()
            .zip(&callee_snapshots)
            .map(|((view, _), snapshot)| (view, snapshot))
            .collect();
        let inlining = job
            .inlining
            .as_ref()
            .map(|inlining| (&inlining.sites[..], &callees[..]));
        let built = match build::build_inlined(&job.view, &snapshot, inlining) {
            Ok(built) => built,
            Err(refusal) => return Outcome::Refused(refusal),
        };
        let build::Built {
            graph,
            cfg,
            initialized,
            callees: frame_cfgs,
            inlines: notes,
        } = built;
        inlines = notes;
        frame_chunks = frame_cfgs.iter().map(|&(chunk, _)| chunk).collect();
        // The chunks the graph's inlined frames run, with their Cfgs, by `InlinedFrame::chunk`.
        let frames: Vec<verify::FrameChunk<'_>> = frame_cfgs
            .iter()
            .map(|(chunk, cfg)| (&callee_memory[*chunk as usize].0, cfg))
            .collect();
        declines = graph.declines.clone();
        narrowed = graph.narrowed.clone();
        // A guard of a graph-inlined call compares the realm's global environment too (step 2).
        embeds_genv = graph.nodes.iter().any(|node| match node.op {
            super::ir::NodeOp::CheckEnvIs { env, .. } => env as usize == job.genv,
            super::ir::NodeOp::CheckCallee { .. } | super::ir::NodeOp::CalleeMatches { .. } => {
                job.inlining.is_some()
            }
            _ => false,
        });
        if job.verifies {
            verified = true;
            if let Err(message) = verify::verify_frames(&graph, (&job.view, &cfg), &frames) {
                if cfg!(any(test, debug_assertions)) {
                    panic!(
                        "the tier's graph of {} fails verification: {message}",
                        job.view.label()
                    );
                }
                verify_failure = Some(message);
                return Outcome::Refused(Refusal::whole("the graph fails verification"));
            }
        }
        let mut stubs = match &job.stubs {
            Some(table) => Stubs::frozen(table),
            None => Stubs::live(),
        };
        let assembled = a64::assemble(
            &job.view,
            &graph,
            &cfg,
            &frames,
            initialized,
            &job.layout,
            &job.ilayout,
            job.options,
            &job.switches,
            &mut stubs,
        );
        met = stubs.met;
        let mut assembled = match assembled {
            Ok(assembled) => assembled,
            Err(refusal) => return Outcome::Refused(refusal),
        };
        codedump(&assembled);
        let executable = job
            .maps
            .then(|| ExecutableBuffer::from_bytes_quietly(assembled.bytes()))
            .flatten()
            .map(SendExecutable);
        if executable.is_some() {
            // The mapping holds the bytes: the words are freed here, not at the install.
            assembled.words = Vec::new();
        }
        Outcome::Compiled(Box::new(Compiled {
            assembled,
            executable,
            nodes: graph.nodes.len(),
        }))
    })();
    // The job's memory goes back to the thread that prepared it (point 5), the rest of it here,
    // on the compile's thread.
    Finished {
        memory: (job.view, job.snapshot),
        callee_memory: job
            .inlining
            .map_or_else(Vec::new, |inlining| inlining.callees),
        inlines,
        frame_chunks,
        outcome,
        stubs: met,
        declines,
        narrowed,
        verified,
        verify_failure,
        elapsed: started.map_or(std::time::Duration::ZERO, |started| started.elapsed()),
        embeds_genv,
    }
}

/// TRUST_JS_JIT_CODEDUMP=<pattern>: the code's words, where its table's name contains the
/// pattern (`*` for every code).
fn codedump(assembled: &Assembled) {
    if let Some(pat) = env_value!("TRUST_JS_JIT_CODEDUMP") {
        let table = &assembled.table;
        if pat == "*" || table.name.contains(pat) {
            eprintln!(
                "[opt-codedump] {} {} words",
                table.name,
                assembled.words.len()
            );
            for w in &assembled.words {
                eprintln!("[opt-codedump] {w:08x}");
            }
        }
    }
}

/// The job's and the result's types cross threads (the plan's point 2): checked here.
#[allow(dead_code)]
fn assert_send<T: Send>() {}
const _: fn() = assert_send::<Job>;
const _: fn() = assert_send::<Finished>;
