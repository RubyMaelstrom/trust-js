//! Host-driven script interruption.
//!
//! HTML §8.1.4.5 permits a user agent to abort a running script without executing author
//! `catch`/`finally` machinery; the Worker termination algorithm requires that ability from a
//! thread running in parallel with the worker. This is therefore an engine control completion,
//! not a JavaScript exception.
//!
//! Execution polls one word per interpreter ([`PollWord`]), in V8's shape: its generated code
//! compares the stack pointer with a limit that other threads move to request an interrupt
//! (`StackGuard::RequestInterrupt`), so that "keep running" costs a load and a branch. A
//! [`RuntimeInterrupt`] sets the interrupt bit of the words it controls when the host cancels
//! or asks for navigation, and a watchdog thread does when an armed deadline passes, as
//! embedders of V8 bound execution time from a thread of their own.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::Instant;

/// Why the host stopped the current JavaScript execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterruptReason {
    Cancelled,
    UserNavigation,
    DeadlineExceeded,
}

impl InterruptReason {
    pub fn message(self) -> &'static str {
        match self {
            Self::Cancelled => "JavaScript execution cancelled by the host",
            Self::UserNavigation => "JavaScript execution yielded to user navigation",
            Self::DeadlineExceeded => "JavaScript execution deadline exceeded",
        }
    }
}

/// The word an interpreter's execution polls at its safepoints: native code's function entries,
/// loop back-edges and the ends of allocating helpers' calls (a load through
/// `JitCtx::poll_word`, a load of the word and a branch), and the Rust tiers'
/// `Interp::interrupt_poll`. It is zero while no poll needs its slow path. Two writers keep a
/// bit each, with atomic read-modify-writes, so neither clobbers the other's:
/// - [`POLL_HEAP`], the interpreter's heap's: set exactly while a collection is due, the
///   live-object count above its ceiling included (see `gc_core::Control::attach_poll_word`);
///   written on the interpreter's thread.
/// - [`POLL_INTERRUPT`], the interpreter's [`RuntimeInterrupt`]'s: set while a host
///   interruption may be pending; written from any thread.
///
/// The slow path asks the authorities ([`RuntimeInterrupt::current_reason`] and
/// `Interp::gc_check`), so a bit set spuriously costs one slow path.
#[derive(Debug, Default)]
pub(crate) struct PollWord(AtomicU64);

/// The heap's bit of a [`PollWord`]: a collection is due.
pub(crate) const POLL_HEAP: u64 = 1 << 0;
/// The interrupt handle's bit of a [`PollWord`]: a host interruption may be pending.
pub(crate) const POLL_INTERRUPT: u64 = 1 << 1;

impl PollWord {
    /// The word's address, through which native code loads it.
    pub(crate) fn as_ptr(&self) -> *const AtomicU64 {
        &self.0
    }

    /// Whether the interrupt bit is set: a host interruption may be pending.
    #[inline(always)]
    pub(crate) fn interrupt_pending(&self) -> bool {
        self.0.load(Ordering::Relaxed) & POLL_INTERRUPT != 0
    }

    /// Whether the heap's bit is set: a collection is due.
    pub(crate) fn heap_due(&self) -> bool {
        self.0.load(Ordering::Relaxed) & POLL_HEAP != 0
    }

    /// Set the heap's bit to `due`, leaving the interrupt bit as it is.
    pub(crate) fn set_heap_due(&self, due: bool) {
        self.set(POLL_HEAP, due);
    }

    fn set(&self, bit: u64, on: bool) {
        if on {
            self.0.fetch_or(bit, Ordering::Release);
        } else {
            self.0.fetch_and(!bit, Ordering::Release);
        }
    }
}

/// Thread-safe control handle polled by every execution tier.
///
/// Cancellation is permanent and is used for realm/worker teardown. User-navigation requests and
/// deadlines are reusable: the host clears/rearms them at the next task boundary. A host may
/// share one handle among engines: it keeps the poll word of each interpreter it controls and
/// sets their interrupt bit while a reason may be pending.
#[derive(Debug, Default)]
pub struct RuntimeInterrupt {
    shared: Arc<Shared>,
}

/// A handle's state. The deadline watchdog holds it weakly, so a deadline outliving every
/// handle of its own expires into nothing.
#[derive(Debug)]
struct Shared {
    /// Active host-control reasons, in one word: the authority that
    /// [`RuntimeInterrupt::current_reason`] reads.
    state: AtomicU8,
    deadline: Mutex<Deadline>,
    /// How to wake the thread that is blocked for this execution, if any (an agent suspended in
    /// `Atomics.wait`, or a host waiting for the agent's jobs): it cannot poll.
    blocked: Mutex<Option<BlockedWake>>,
    /// The poll words whose interrupt bit this handle keeps, one per interpreter it controls.
    /// Their bit is written only under this lock, after the change of `state` it reflects, and
    /// says what `state` says at that moment: the last writer's bit is then the current one, so
    /// a request racing a clear is never lost.
    words: Mutex<Vec<Weak<PollWord>>>,
    /// The watchdog that watches this handle's deadlines: the process's, or a test's own.
    watchdog: &'static watchdog::Watchdog,
}

impl Default for Shared {
    fn default() -> Self {
        Self::new(&watchdog::WATCHDOG)
    }
}

/// The armed deadline, if any.
#[derive(Debug, Default)]
struct Deadline {
    at: Option<Instant>,
    /// Counts the deadlines set: the watchdog reports a deadline's expiry only while its
    /// generation is current.
    generation: u64,
    /// The watchdog's entry for `at`, while it has one.
    watch: Option<watchdog::Key>,
}

/// Wakes a blocked thread so that it observes a new interruption request or deadline.
#[derive(Clone)]
pub(crate) struct BlockedWake(pub(crate) Arc<dyn Fn() + Send + Sync>);

impl std::fmt::Debug for BlockedWake {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BlockedWake")
    }
}

const CANCELLED: u8 = 1 << 0;
const USER_NAVIGATION: u8 = 1 << 1;
const DEADLINE_ARMED: u8 = 1 << 2;
/// The armed deadline passed: the watchdog saw it pass, or it was set in the past.
const DEADLINE_EXPIRED: u8 = 1 << 3;
/// The armed deadline has no watchdog, whose thread could not be started: the interrupt bit
/// stays set while it is armed, and the polls' slow path reads the clock.
const DEADLINE_UNWATCHED: u8 = 1 << 4;
const DEADLINE: u8 = DEADLINE_ARMED | DEADLINE_EXPIRED | DEADLINE_UNWATCHED;
/// The states that set the poll words' interrupt bit.
const POLLED: u8 = CANCELLED | USER_NAVIGATION | DEADLINE_EXPIRED | DEADLINE_UNWATCHED;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl RuntimeInterrupt {
    pub fn cancel(&self) {
        self.shared.state.fetch_or(CANCELLED, Ordering::Release);
        self.shared.sync_words();
        self.shared.wake_blocked();
    }

    pub fn request_user_navigation(&self) {
        self.shared
            .state
            .fetch_or(USER_NAVIGATION, Ordering::Release);
        self.shared.sync_words();
        self.shared.wake_blocked();
    }

    pub fn begin_user_interaction(&self) {
        self.shared
            .state
            .fetch_and(!USER_NAVIGATION, Ordering::Release);
        self.shared.sync_words();
    }

    pub fn user_navigation_requested(&self) -> bool {
        self.shared.state.load(Ordering::Acquire) & USER_NAVIGATION != 0
    }

    /// Arm `deadline`, replacing any armed one, or disarm with `None`. A deadline already past
    /// expires at once; a later one when the watchdog thread sees it pass.
    pub fn set_deadline(&self, deadline: Option<Instant>) {
        let shared = &self.shared;
        {
            let mut slot = lock(&shared.deadline);
            if let Some(key) = slot.watch.take() {
                shared.watchdog.forget(key);
            }
            slot.generation = slot.generation.wrapping_add(1);
            slot.at = deadline;
            let armed = match deadline {
                None => 0,
                Some(at) if at <= Instant::now() => DEADLINE_ARMED | DEADLINE_EXPIRED,
                Some(at) => match shared
                    .watchdog
                    .watch(at, slot.generation, Arc::downgrade(shared))
                {
                    Some(key) => {
                        slot.watch = Some(key);
                        DEADLINE_ARMED
                    }
                    None => DEADLINE_ARMED | DEADLINE_UNWATCHED,
                },
            };
            let mut state = shared.state.load(Ordering::Relaxed);
            while let Err(actual) = shared.state.compare_exchange_weak(
                state,
                state & !DEADLINE | armed,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                state = actual;
            }
        }
        shared.sync_words();
        // A blocked thread recomputes when it must wake.
        shared.wake_blocked();
    }

    /// The armed deadline, when a thread blocked for this execution must wake to report it.
    pub(crate) fn armed_deadline(&self) -> Option<Instant> {
        if self.shared.state.load(Ordering::Acquire) & DEADLINE_ARMED == 0 {
            return None;
        }
        lock(&self.shared.deadline).at
    }

    /// Run `wait`, which blocks the calling thread, with `wake` registered: a cancellation,
    /// navigation request or deadline change from another thread calls it, and so does the
    /// expiry of the armed deadline. `wait` must test [`Self::current_reason`] while holding the
    /// lock `wake` takes before it notifies, and block by releasing that lock, so that a request
    /// made after the test still wakes it.
    pub(crate) fn while_blocked<R>(&self, wake: BlockedWake, wait: impl FnOnce() -> R) -> R {
        *lock(&self.shared.blocked) = Some(wake);
        let result = wait();
        *lock(&self.shared.blocked) = None;
        result
    }

    /// The currently active reason, in host-control priority order. An armed deadline counts
    /// from the moment it passes, whether or not the watchdog has seen it yet.
    pub fn current_reason(&self) -> Option<InterruptReason> {
        let state = self.shared.state.load(Ordering::Acquire);
        if state == 0 {
            return None;
        }
        if state & CANCELLED != 0 {
            return Some(InterruptReason::Cancelled);
        }
        if state & USER_NAVIGATION != 0 {
            return Some(InterruptReason::UserNavigation);
        }
        if state & DEADLINE_EXPIRED != 0
            || state & DEADLINE_ARMED != 0
                && lock(&self.shared.deadline)
                    .at
                    .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Some(InterruptReason::DeadlineExceeded);
        }
        None
    }

    /// Keep `word`'s interrupt bit from now on: its interpreter starts answering to this handle.
    pub(crate) fn register(&self, word: &Arc<PollWord>) {
        let mut words = lock(&self.shared.words);
        words.retain(|kept| kept.strong_count() != 0);
        words.push(Arc::downgrade(word));
        word.set(POLL_INTERRUPT, self.shared.polled());
    }

    /// Stop keeping `word`'s interrupt bit: its interpreter changes handles or ends.
    pub(crate) fn unregister(&self, word: &Arc<PollWord>) {
        lock(&self.shared.words)
            .retain(|kept| kept.strong_count() != 0 && !std::ptr::eq(kept.as_ptr(), &**word));
    }

    /// A handle whose deadlines `watchdog` watches instead of the process's (tests).
    #[cfg(test)]
    fn with_watchdog(watchdog: &'static watchdog::Watchdog) -> Self {
        Self {
            shared: Arc::new(Shared::new(watchdog)),
        }
    }

    /// The number of live poll words this handle keeps (tests).
    #[cfg(test)]
    pub(crate) fn registered_words(&self) -> usize {
        lock(&self.shared.words)
            .iter()
            .filter(|word| word.strong_count() != 0)
            .count()
    }
}

impl Shared {
    fn new(watchdog: &'static watchdog::Watchdog) -> Self {
        Self {
            state: AtomicU8::new(0),
            deadline: Mutex::default(),
            blocked: Mutex::default(),
            words: Mutex::default(),
            watchdog,
        }
    }

    /// Whether the poll words' interrupt bit should be set.
    fn polled(&self) -> bool {
        self.state.load(Ordering::Acquire) & POLLED != 0
    }

    /// Write every registered word's interrupt bit from the state as it stands now (see
    /// `words`), forgetting the words whose interpreters have ended.
    fn sync_words(&self) {
        let mut words = lock(&self.words);
        let polled = self.polled();
        words.retain(|word| match word.upgrade() {
            Some(word) => {
                word.set(POLL_INTERRUPT, polled);
                true
            }
            None => false,
        });
    }

    /// The watchdog's report that the deadline armed as `generation` passed.
    fn expire(&self, generation: u64) {
        {
            let mut slot = lock(&self.deadline);
            if slot.generation != generation {
                return; // since replaced or disarmed
            }
            slot.watch = None;
            self.state.fetch_or(DEADLINE_EXPIRED, Ordering::Release);
        }
        self.sync_words();
        self.wake_blocked();
    }

    fn wake_blocked(&self) {
        let wake = lock(&self.blocked).clone();
        if let Some(BlockedWake(wake)) = wake {
            wake();
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        let slot = self
            .deadline
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(key) = slot.watch.take() {
            self.watchdog.forget(key);
        }
    }
}

/// One thread at a time watches the armed deadlines, Node's watchdog thread kept for bursts
/// (Decision 18): Node starts a thread for each timed run and joins it (`node_watchdog.cc`);
/// here the first deadline armed in the future starts the thread, the thread serves the
/// deadlines armed after it, and it exits once none has been armed for [`IDLE_EXIT`]. The next
/// deadline then starts a fresh thread, and no thread exists while deadlines are out of use. The
/// thread sleeps until the earliest deadline, then reports each one that has passed to its
/// handle (`Shared::expire`).
mod watchdog {
    use super::{lock, PoisonError, Shared};
    use std::collections::BTreeMap;
    use std::sync::{Condvar, Mutex, Weak};
    use std::time::{Duration, Instant};

    /// How long the thread outlives the last deadline armed. A host that arms deadlines in a
    /// burst keeps one thread: TRust arms and clears one around every URLPattern regexp
    /// evaluation, and a thread's start, cancellation and join measured 36-90 us on Crystal,
    /// against a few microseconds for the match itself.
    pub(super) const IDLE_EXIT: Duration = Duration::from_secs(5);

    /// An entry's place in the queue: its deadline, then the order of arming.
    pub(super) type Key = (Instant, u64);

    struct Queue {
        /// The armed deadlines: each one's generation and handle.
        entries: BTreeMap<Key, (u64, Weak<Shared>)>,
        next: u64,
        /// When the thread looks at the queue next: the earliest deadline as it last went to
        /// sleep, or `None` while it sleeps until an entry arrives. Only an earlier deadline
        /// needs to wake it; one replaced or disarmed meanwhile costs it one wake-up for
        /// nothing, at most, instead of a host's every arming.
        wake_at: Option<Instant>,
        /// Whether a thread serves the queue. The arm that finds none starts one and sets this,
        /// and the thread clears it as it exits, both under the queue's lock: an arm racing the
        /// exit either lands before it, and the thread serves it, or finds the thread gone and
        /// starts the next.
        running: bool,
    }

    pub(super) struct Watchdog {
        queue: Mutex<Queue>,
        /// Signalled when a deadline earlier than `wake_at` arrives.
        changed: Condvar,
        /// How long the thread outlives the last deadline armed ([`IDLE_EXIT`] but in tests).
        idle: Duration,
        /// Threads started so far (tests).
        #[cfg(test)]
        started: std::sync::atomic::AtomicUsize,
    }

    impl std::fmt::Debug for Watchdog {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("Watchdog")
        }
    }

    /// The process's watchdog.
    pub(super) static WATCHDOG: Watchdog = Watchdog::new(IDLE_EXIT);

    impl Watchdog {
        pub(super) const fn new(idle: Duration) -> Self {
            Self {
                queue: Mutex::new(Queue {
                    entries: BTreeMap::new(),
                    next: 0,
                    wake_at: None,
                    running: false,
                }),
                changed: Condvar::new(),
                idle,
                #[cfg(test)]
                started: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        /// Watch `handle`'s deadline `at`, armed as `generation`, starting the thread if none
        /// runs. `None` if it cannot start.
        pub(super) fn watch(
            &'static self,
            at: Instant,
            generation: u64,
            handle: Weak<Shared>,
        ) -> Option<Key> {
            let mut queue = lock(&self.queue);
            let key = (at, queue.next);
            queue.next += 1;
            queue.entries.insert(key, (generation, handle));
            if !queue.running {
                // Started under the lock, which the thread takes first: no arm or exit can come
                // between the start and its outcome.
                let started = std::thread::Builder::new()
                    .name("trust-js-watchdog".into())
                    .spawn(move || self.run())
                    .is_ok();
                if !started {
                    queue.entries.remove(&key);
                    return None;
                }
                queue.running = true;
                queue.wake_at = Some(at);
                #[cfg(test)]
                self.started
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(key);
            }
            let sooner = queue.wake_at.is_none_or(|wake_at| at < wake_at);
            if sooner {
                queue.wake_at = Some(at);
            }
            drop(queue);
            if sooner {
                self.changed.notify_one();
            }
            Some(key)
        }

        /// Drop the entry `key` (its deadline was replaced or disarmed): a handle that arms and
        /// clears deadlines often leaves no stale entries behind.
        pub(super) fn forget(&self, key: Key) {
            lock(&self.queue).entries.remove(&key);
        }

        fn run(&self) {
            let mut queue = lock(&self.queue);
            // Since when the queue has been empty, while it is.
            let mut idle_since = None;
            loop {
                let now = Instant::now();
                let mut passed = Vec::new();
                while let Some(entry) = queue.entries.first_entry() {
                    if entry.key().0 > now {
                        break;
                    }
                    passed.push(entry.remove());
                }
                if !passed.is_empty() {
                    // Report outside the queue's lock: `Shared::expire` takes the handle's
                    // locks, and a handle dropped here forgets its entry.
                    drop(queue);
                    for (generation, handle) in passed {
                        if let Some(shared) = handle.upgrade() {
                            shared.expire(generation);
                        }
                    }
                    queue = lock(&self.queue);
                    continue;
                }
                queue.wake_at = queue.entries.first_key_value().map(|(&(at, _), _)| at);
                let wait = match queue.wake_at {
                    Some(at) => {
                        idle_since = None;
                        at.saturating_duration_since(now)
                    }
                    None => {
                        let idle = now.saturating_duration_since(*idle_since.get_or_insert(now));
                        if idle >= self.idle {
                            // Exit under the lock (see `Queue::running`).
                            queue.running = false;
                            return;
                        }
                        self.idle - idle
                    }
                };
                queue = self
                    .changed
                    .wait_timeout(queue, wait)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        }

        /// The entries waiting (tests).
        #[cfg(test)]
        pub(super) fn entries(&self) -> Vec<Key> {
            lock(&self.queue).entries.keys().copied().collect()
        }

        /// Whether a thread serves the queue, and how many have been started (tests).
        #[cfg(test)]
        pub(super) fn threads(&self) -> (bool, usize) {
            (
                lock(&self.queue).running,
                self.started.load(std::sync::atomic::Ordering::Relaxed),
            )
        }
    }
}

#[cfg(test)]
#[path = "interrupt_tests.rs"]
mod tests;
