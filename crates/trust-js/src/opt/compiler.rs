//! The process's compile thread (P3 stage 2, step 12; the plan's "Step 12, compiling off the main
//! thread", points 3, 5 and 9): one thread a process, named `trust-js-opt`, started at the first
//! job and kept for the process, as V8's Maglev keeps its platform's background workers and JSC
//! its worklist threads. It serves every engine (agents and workers run engines on other
//! threads): a job carries its engine's result queue ([`Results`]) and poll word, and its
//! cancellation flag. The thread takes the jobs in order, compiles each (`job::compile`), queues
//! the result on its engine's queue and sets the engine's install bit (`interrupt::POLL_INSTALL`,
//! with release order), so that the engine's next poll installs it (`driver::install_finished`),
//! as V8's job sets its isolate's INSTALL_MAGLEV_CODE interrupt
//! (maglev/maglev-concurrent-dispatcher.cc:315-320) and SpiderMonkey's helper requests
//! AttachOffThreadCompilations (jit/IonCompileTask.cpp:21-44). A cancelled job (its engine was
//! dropped) is skipped, or its result dropped (point 9). One thread, because the measured queue
//! is short (offthread-input.txt 3.4) and the owner's model has two CPUs.
//!
//! Where no thread can be started the tier compiles on the main thread, as the watchdog's thread
//! degrades (`interrupt::watchdog`): the first failed start is the last.

use super::job::{self, Finished, Job};
use crate::interrupt::PollWord;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// An engine's finished jobs, which its compile thread queues and its main thread installs.
#[derive(Default)]
pub(crate) struct Results {
    done: Mutex<Vec<Done>>,
}

impl Results {
    /// The finished jobs queued so far, in the order they finished.
    pub(crate) fn take(&self) -> Vec<Done> {
        std::mem::take(&mut *self.done.lock().unwrap_or_else(PoisonError::into_inner))
    }
}

/// A finished job: its id, the compile's result (a panic's payload where it panicked, which the
/// install raises again on the main thread), and its wait in the queue.
pub(crate) struct Done {
    pub(crate) id: u64,
    pub(crate) finished: std::thread::Result<Finished>,
    pub(crate) waited: Duration,
}

/// A job in the queue.
struct Queued {
    id: u64,
    job: Job,
    cancelled: Arc<AtomicBool>,
    results: Arc<Results>,
    poll: Arc<PollWord>,
    queued: Instant,
}

/// The thread's queue.
struct Compiler {
    queue: Mutex<VecDeque<Queued>>,
    wake: Condvar,
}

/// The process's compile thread, once its start was tried: None where it could not start.
static COMPILER: OnceLock<Option<&'static Compiler>> = OnceLock::new();

/// Held by a test that needs a compile to stay pending: the thread takes it around each compile
/// (tests).
#[cfg(test)]
pub(crate) static TEST_HOLD: Mutex<()> = Mutex::new(());

/// The cancelled jobs the thread skipped or whose results it dropped (tests).
#[cfg(test)]
pub(crate) static TEST_CANCELLED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Whether this process has a compile thread, started now if this is the first ask (None where
/// it cannot start: the tier compiles on the main thread).
pub(crate) fn available() -> bool {
    compiler().is_some()
}

fn compiler() -> Option<&'static Compiler> {
    *COMPILER.get_or_init(start)
}

/// Start the thread: false where the system refuses it.
fn start() -> Option<&'static Compiler> {
    let compiler: &'static Compiler = Box::leak(Box::new(Compiler {
        queue: Mutex::new(VecDeque::new()),
        wake: Condvar::new(),
    }));
    std::thread::Builder::new()
        .name("trust-js-opt".into())
        .stack_size(super::driver::COMPILE_STACK)
        .spawn(move || run(compiler))
        .ok()
        .map(|_| compiler)
}

/// Queue `job` (id `id` of its engine) for the compile thread, which queues its result on
/// `results` and sets `poll`'s install bit; `cancelled` set means the engine dropped it. The job
/// back where the process has no compile thread.
#[allow(clippy::result_large_err)]
pub(crate) fn enqueue(
    id: u64,
    job: Job,
    cancelled: Arc<AtomicBool>,
    results: Arc<Results>,
    poll: Arc<PollWord>,
) -> Result<(), Job> {
    let Some(compiler) = compiler() else {
        return Err(job);
    };
    compiler
        .queue
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push_back(Queued {
            id,
            job,
            cancelled,
            results,
            poll,
            queued: Instant::now(),
        });
    compiler.wake.notify_one();
    Ok(())
}

fn run(compiler: &'static Compiler) {
    // Debug builds check that a compile here reaches none of a main thread's state.
    let _off = crate::jit::OffMainCompile::enter();
    loop {
        let queued = {
            let mut queue = compiler
                .queue
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            loop {
                if let Some(queued) = queue.pop_front() {
                    break queued;
                }
                queue = compiler
                    .wake
                    .wait(queue)
                    .unwrap_or_else(PoisonError::into_inner);
            }
        };
        let Queued {
            id,
            job,
            cancelled,
            results,
            poll,
            queued,
        } = queued;
        // An engine dropped meanwhile: its job is skipped (the plan's point 9).
        if cancelled.load(Ordering::Acquire) {
            #[cfg(test)]
            TEST_CANCELLED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        let waited = queued.elapsed();
        #[cfg(test)]
        let hold = TEST_HOLD.lock().unwrap_or_else(PoisonError::into_inner);
        // A panic (a test's failed verification, a bug) is the main thread's to raise.
        let finished = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job::compile(job)));
        #[cfg(test)]
        drop(hold);
        if cancelled.load(Ordering::Acquire) {
            // Its executable pages, if any, are freed here, from the process-wide region.
            #[cfg(test)]
            TEST_CANCELLED.fetch_add(1, Ordering::Relaxed);
            continue;
        }
        results
            .done
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Done {
                id,
                finished,
                waited,
            });
        poll.request_install();
    }
}
