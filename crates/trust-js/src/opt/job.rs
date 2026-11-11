//! A tier compile as a job (P3 stage 2, step 12; the plan's "Step 12, compiling off the main
//! thread"): its inputs ([`Job`]), made on the main thread with the snapshot (`driver::prepare`),
//! the compile itself ([`compile`]: the build, the verification, register allocation, emission
//! and, where the budget allows without the main thread's reclaim, the copy into executable
//! memory), and its outputs ([`Finished`]), which the main thread installs (`driver`). The inputs
//! and outputs are `Send`, and the compile reads nothing else: no chunk, cache, record, heap
//! object or interpreter (their addresses travel as words to embed), no thread-local (the test
//! switches, counters and stub table are the main thread's, read into the job), so that it runs
//! on any thread, as V8's Maglev runs Execute (maglev/maglev-compiler.cc:90-313) and
//! SpiderMonkey its CompileBackEnd (jit/Ion.cpp:1750-1771).

use super::a64::{self, Assembled, Options, Stubs, Switches};
use super::ir::{Decline, Refusal};
use super::snapshot::Snapshot;
use super::view::FrozenChunk;
use super::{build, verify};
use crate::jit::{ExecutableBuffer, StubTableSnapshot};
use std::sync::Arc;

/// A compile's inputs (see the module's doc).
pub(crate) struct Job {
    /// The chunk's frozen view.
    pub(crate) view: FrozenChunk,
    /// Its feedback, read at the snapshot.
    pub(crate) snapshot: Snapshot,
    pub(crate) options: Options,
    pub(crate) switches: Switches,
    pub(crate) layout: crate::value::JitLayout,
    pub(crate) ilayout: crate::interpreter::InterpLayout,
    /// Whether the build verifies its graph (`driver::verifies`).
    pub(crate) verifies: bool,
    /// The running realm's global environment at the snapshot (`Env::as_ptr`), whose identity
    /// the code may check (a word).
    pub(crate) genv: usize,
    /// The main thread's stub table at the snapshot, or None where the compile runs on the main
    /// thread, which assembles a missing stub at once.
    pub(crate) stubs: Option<Arc<StubTableSnapshot>>,
    /// The stubs whose assembly failed on the main thread: the code keeps copies.
    pub(crate) stub_copies: Vec<u32>,
    /// Whether the compile maps its code itself (off the main thread; the main thread's
    /// synchronous compile maps at the install, as before).
    pub(crate) maps: bool,
    /// Whether the compile reads the clock for its time (`Finished::elapsed`): under
    /// TRUST_JS_OPT_LOG only, the log's (stage 2, step 16).
    pub(crate) timed: bool,
    /// The main thread's addresses of the counters test builds' templates embed.
    #[cfg(test)]
    pub(crate) test_addresses: crate::jit::TestAddresses,
}

/// What a compile made.
pub(crate) enum Outcome {
    /// The code, and its nodes' count (the log's).
    Compiled(Box<Compiled>),
    Refused(Refusal),
    /// The emission found stubs missing from the snapshot's table (`stubs`, in the order it
    /// requested them), or names whose atoms the frozen view lacks (`FrozenChunk::missing_keys`):
    /// the main thread assembles and makes them, and the job runs again. Off the main thread
    /// only.
    Needs {
        stubs: Vec<u32>,
    },
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
    /// The job where it runs again (`Outcome::Needs`); otherwise the compile's thread dropped it.
    pub(crate) job: Option<Job>,
    pub(crate) outcome: Outcome,
    /// The builder's declines, which the install logs.
    pub(crate) declines: Vec<Decline>,
    /// Whether a graph was verified, and the verifier's message where it failed (release).
    pub(crate) verified: bool,
    pub(crate) verify_failure: Option<String>,
    /// The compile's time on its thread, where it was timed (`Job::timed`; zero otherwise).
    pub(crate) elapsed: std::time::Duration,
    /// Whether the code compares a record with the realm's global environment (`genv`), which
    /// the install then pins (stage 2, step 4).
    pub(crate) embeds_genv: bool,
}

/// Compile `job` (see the module's doc): reads nothing but the job and process constants.
pub(crate) fn compile(job: Job) -> Finished {
    let started = job.timed.then(std::time::Instant::now);
    #[cfg(test)]
    let _addresses = crate::jit::TestAddresses::install(job.test_addresses);
    let mut declines = Vec::new();
    let mut verified = false;
    let mut verify_failure = None;
    let mut embeds_genv = false;
    let outcome = (|| {
        let built = match build::build_view(&job.view, &job.snapshot) {
            Ok(built) => built,
            Err(refusal) => return Outcome::Refused(refusal),
        };
        let build::Built {
            graph,
            cfg,
            initialized,
        } = built;
        declines = graph.declines.clone();
        embeds_genv = graph.nodes.iter().any(|node| {
            matches!(node.op, super::ir::NodeOp::CheckEnvIs { env, .. } if env as usize == job.genv)
        });
        if job.verifies {
            verified = true;
            if let Err(message) = verify::verify_view(&graph, &job.view, &cfg) {
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
        let mut assembled = match a64::assemble(
            &job.view,
            &graph,
            &cfg,
            initialized,
            &job.layout,
            &job.ilayout,
            job.options,
            &job.switches,
            &mut stubs,
        ) {
            Ok(assembled) => assembled,
            Err(refusal) => return Outcome::Refused(refusal),
        };
        if !stubs.missing.is_empty() || !job.view.missing_keys.borrow().is_empty() {
            return Outcome::Needs {
                stubs: stubs.missing,
            };
        }
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
    // The job's frozen view, snapshot and stub table go here, on the compile's thread, unless
    // the job runs again.
    let job = matches!(outcome, Outcome::Needs { .. }).then_some(job);
    Finished {
        job,
        outcome,
        declines,
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
