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
//!
//! The hand-off (stage 2, step 16): the thread marks itself waiting under the queue's lock before
//! it waits, and an enqueue wakes it only where it finds the mark, which it clears ([`Queue`]): a
//! job queued while the thread compiles costs no wake, as JSC's worklist wakes a thread only
//! while its active threads fall short of its target (JITWorklist::wakeThreads,
//! jit/JITWorklist.cpp:106-148), a worklist thread leaving the active count under the worklist's
//! lock when it finds no plan (JITWorklistThread::poll, jit/JITWorklistThread.cpp:97-126), and
//! SpiderMonkey's dispatch returns while every helper thread runs a task
//! (vm/HelperThreads.cpp:262-265), a helper that finishes taking the next itself
//! (vm/InternalThreadPool.cpp:308-333). A job's clock reads (its queue stamp) are the log's.

use super::job::{self, Finished, Job};
use crate::interrupt::PollWord;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
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
    /// When it was queued, under TRUST_JS_OPT_LOG only (its wait is the log's).
    queued: Option<Instant>,
}

/// A queue one thread takes from ([`Queue::take`]) and any thread adds to ([`Queue::push`]), which
/// wakes the taker only where it waits (see the module's doc). Why no job is left behind: the
/// taker looks at the jobs and marks itself waiting under one hold of the lock, and a push adds
/// its job and reads the mark under another, so a push either comes first, and the taker finds
/// its job before it waits, or finds the mark, and wakes it; the condition variable's wait
/// releases the lock and sleeps as one step, so the wake reaches the waiter even where it is
/// made after the push releases the lock (std's futex condition variable: the waiter sleeps only
/// while the variable's sequence word is the one it read under the lock, which each notify
/// moves). A spurious return or a wake whose job another take found finds the jobs empty and
/// waits again, marked. A taker busy with a job looks at the queue again before it waits, so the
/// jobs pushed meanwhile, unwoken, are taken in order then.
pub(crate) struct Queue<T> {
    state: Mutex<QueueState<T>>,
    wake: Condvar,
}

struct QueueState<T> {
    jobs: VecDeque<T>,
    /// The taker waits for a job: the next push clears it and wakes the taker.
    waiting: bool,
}

impl<T> Queue<T> {
    pub(crate) const fn new() -> Self {
        Queue {
            state: Mutex::new(QueueState {
                jobs: VecDeque::new(),
                waiting: false,
            }),
            wake: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, QueueState<T>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Add `job` at the queue's end, waking the taker where it waits: whether it woke it.
    pub(crate) fn push(&self, job: T) -> bool {
        let mut state = self.lock();
        state.jobs.push_back(job);
        let wake = std::mem::take(&mut state.waiting);
        drop(state);
        if wake {
            self.wake.notify_one();
        }
        wake
    }

    /// The first job, waiting for one where there is none (one taker a queue).
    pub(crate) fn take(&self) -> T {
        let mut state = self.lock();
        loop {
            if let Some(job) = state.jobs.pop_front() {
                return job;
            }
            state.waiting = true;
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Whether the taker waits for a job (tests).
    #[cfg(test)]
    pub(crate) fn waiting(&self) -> bool {
        self.lock().waiting
    }

    /// The jobs queued and not yet taken (tests).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().jobs.len()
    }
}

/// The thread's queue.
struct Compiler {
    queue: Queue<Queued>,
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
        queue: Queue::new(),
    }));
    std::thread::Builder::new()
        .name("trust-js-opt".into())
        .stack_size(super::driver::COMPILE_STACK)
        .spawn(move || run(compiler))
        .ok()
        .map(|_| compiler)
}

/// Queue `job` (id `id` of its engine) for the compile thread, which queues its result on
/// `results` and sets `poll`'s install bit; `cancelled` set means the engine dropped it. The
/// thread is woken only where it waits ([`Queue::push`]). The job back where the process has no
/// compile thread.
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
    compiler.queue.push(Queued {
        id,
        job,
        cancelled,
        results,
        poll,
        queued: super::log_enabled().then(Instant::now),
    });
    Ok(())
}

fn run(compiler: &'static Compiler) {
    // Debug builds check that a compile here reaches none of a main thread's state.
    let _off = crate::jit::OffMainCompile::enter();
    loop {
        let queued = compiler.queue.take();
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
        let waited = queued.map_or(Duration::ZERO, |queued| queued.elapsed());
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

#[cfg(test)]
mod tests {
    use super::Queue;
    use std::sync::mpsc;
    use std::time::Duration;

    const STOP: u32 = u32::MAX;
    const PATIENCE: Duration = Duration::from_secs(60);

    /// Wait until `queue`'s taker waits for a job.
    fn until_waiting(queue: &Queue<u32>) {
        let start = std::time::Instant::now();
        while !queue.waiting() {
            assert!(start.elapsed() < PATIENCE, "the taker waits for a job");
            std::thread::yield_now();
        }
    }

    /// The hand-off (stage 2, step 16): a push wakes the taker only where it waits; jobs pushed
    /// while it is busy are taken without a wake, in order, before it waits again.
    #[test]
    fn a_push_wakes_the_taker_only_where_it_waits() {
        let queue: &'static Queue<u32> = Box::leak(Box::new(Queue::new()));
        let (taken, jobs) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        // The taker reports each job and stays busy with it until released.
        let taker = std::thread::spawn(move || loop {
            let job = queue.take();
            taken.send(job).unwrap();
            if job == STOP {
                break;
            }
            released.recv().unwrap();
        });
        until_waiting(queue);
        assert!(queue.push(1), "a push to a waiting taker wakes it");
        assert_eq!(
            jobs.recv_timeout(PATIENCE),
            Ok(1),
            "the woken taker takes it"
        );
        // Busy with job 1: the pushes meanwhile wake nothing.
        assert!(!queue.waiting());
        assert!(
            !queue.push(2),
            "a push while the taker is busy wakes nothing"
        );
        assert!(!queue.push(3));
        assert_eq!(queue.len(), 2);
        for job in [2, 3] {
            release.send(()).unwrap();
            assert_eq!(
                jobs.recv_timeout(PATIENCE),
                Ok(job),
                "taken unwoken, in order"
            );
        }
        release.send(()).unwrap();
        // Done: it waits again, and the next push wakes it.
        until_waiting(queue);
        assert_eq!(queue.len(), 0);
        assert!(queue.push(STOP));
        assert_eq!(jobs.recv_timeout(PATIENCE), Ok(STOP));
        taker.join().unwrap();
    }

    /// No job is left behind: producers on several threads push many small jobs while the taker
    /// takes them, waiting whenever it runs dry; every job is taken, once, and the queue drains.
    #[test]
    fn many_small_jobs_from_several_threads_are_all_taken() {
        const PRODUCERS: u32 = 4;
        const JOBS: u32 = 20_000;
        let queue: &'static Queue<u32> = Box::leak(Box::new(Queue::new()));
        let (done, result) = mpsc::channel();
        let taker = std::thread::spawn(move || {
            let mut seen = vec![0u8; (PRODUCERS * JOBS) as usize];
            loop {
                let job = queue.take();
                if job == STOP {
                    break;
                }
                seen[job as usize] += 1;
                // A small compile, now and then.
                if job.is_multiple_of(97) {
                    std::thread::sleep(Duration::from_micros(20));
                }
            }
            done.send(seen).unwrap();
        });
        let producers: Vec<_> = (0..PRODUCERS)
            .map(|p| {
                std::thread::spawn(move || {
                    let mut woke = 0u32;
                    for k in 0..JOBS {
                        woke += u32::from(queue.push(p * JOBS + k));
                        if k.is_multiple_of(251) {
                            // Let the taker run dry and wait sometimes.
                            std::thread::sleep(Duration::from_micros(50));
                        }
                    }
                    woke
                })
            })
            .collect();
        let woke: u32 = producers.into_iter().map(|p| p.join().unwrap()).sum();
        queue.push(STOP);
        let seen = result
            .recv_timeout(PATIENCE)
            .expect("the taker takes every job and the last");
        taker.join().unwrap();
        assert!(seen.iter().all(|&n| n == 1), "every job taken once");
        assert_eq!(queue.len(), 0, "the queue drained");
        // A taker that never runs dry takes every job unwoken: the count is the scheduler's.
        assert!(woke <= PRODUCERS * JOBS);
        eprintln!("{woke} of {} pushes woke the taker", PRODUCERS * JOBS);
    }
}
