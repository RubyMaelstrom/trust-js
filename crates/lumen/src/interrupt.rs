//! Host-driven script interruption.
//!
//! HTML §8.1.4.5 permits a user agent to abort a running script without executing author
//! `catch`/`finally` machinery; the Worker termination algorithm requires that ability from a
//! thread running in parallel with the worker. This is therefore an engine control completion,
//! not a JavaScript exception.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
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

/// Thread-safe control handle polled by every execution tier.
///
/// Cancellation is permanent and is used for realm/worker teardown. User-navigation requests and
/// deadlines are reusable: the host clears/rearms them at the next task boundary.
#[derive(Debug, Default)]
pub struct RuntimeInterrupt {
    /// Active host-control reasons. Keeping the common state in one word makes every execution
    /// tier's overwhelmingly common "keep running" poll one acquire load; the deadline mutex is
    /// consulted only when its bit is armed.
    state: AtomicU8,
    deadline: Mutex<Option<Instant>>,
    /// How to wake the thread that is blocked for this execution, if any (an agent suspended in
    /// `Atomics.wait`, or a host waiting for the agent's jobs): it cannot poll.
    blocked: Mutex<Option<BlockedWake>>,
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

impl RuntimeInterrupt {
    pub fn cancel(&self) {
        self.state.fetch_or(CANCELLED, Ordering::Release);
        self.wake_blocked();
    }

    pub fn request_user_navigation(&self) {
        self.state.fetch_or(USER_NAVIGATION, Ordering::Release);
        self.wake_blocked();
    }

    pub fn begin_user_interaction(&self) {
        self.state.fetch_and(!USER_NAVIGATION, Ordering::Release);
    }

    pub fn user_navigation_requested(&self) -> bool {
        self.state.load(Ordering::Acquire) & USER_NAVIGATION != 0
    }

    pub fn set_deadline(&self, deadline: Option<Instant>) {
        let armed = deadline.is_some();
        *self
            .deadline
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = deadline;
        if armed {
            self.state.fetch_or(DEADLINE_ARMED, Ordering::Release);
        } else {
            self.state.fetch_and(!DEADLINE_ARMED, Ordering::Release);
        }
        // A blocked thread recomputes when it must wake.
        self.wake_blocked();
    }

    /// The armed deadline, when a thread blocked for this execution must wake to report it.
    pub(crate) fn armed_deadline(&self) -> Option<Instant> {
        if self.state.load(Ordering::Acquire) & DEADLINE_ARMED == 0 {
            return None;
        }
        *self
            .deadline
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    /// Run `wait`, which blocks the calling thread, with `wake` registered: a cancellation,
    /// navigation request or deadline change from another thread calls it. `wait` must test
    /// [`Self::current_reason`] while holding the lock `wake` takes before it notifies, and block
    /// by releasing that lock, so that a request made after the test still wakes it.
    pub(crate) fn while_blocked<R>(&self, wake: BlockedWake, wait: impl FnOnce() -> R) -> R {
        *self
            .blocked
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(wake);
        let result = wait();
        *self
            .blocked
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
        result
    }

    fn wake_blocked(&self) {
        let wake = self
            .blocked
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        if let Some(BlockedWake(wake)) = wake {
            wake();
        }
    }

    /// The currently active reason, in host-control priority order.
    pub fn current_reason(&self) -> Option<InterruptReason> {
        let state = self.state.load(Ordering::Acquire);
        if state & CANCELLED != 0 {
            return Some(InterruptReason::Cancelled);
        }
        if state & USER_NAVIGATION != 0 {
            return Some(InterruptReason::UserNavigation);
        }
        if state & DEADLINE_ARMED != 0
            && self
                .deadline
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return Some(InterruptReason::DeadlineExceeded);
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{InterruptReason, RuntimeInterrupt};
    use std::time::{Duration, Instant};

    #[test]
    fn packed_interrupt_state_preserves_priority_and_reusable_reasons() {
        let interrupt = RuntimeInterrupt::default();
        assert_eq!(interrupt.current_reason(), None);

        interrupt.set_deadline(Some(Instant::now() + Duration::from_secs(60)));
        assert_eq!(interrupt.current_reason(), None);

        interrupt.request_user_navigation();
        assert!(interrupt.user_navigation_requested());
        assert_eq!(
            interrupt.current_reason(),
            Some(InterruptReason::UserNavigation)
        );
        interrupt.begin_user_interaction();
        assert!(!interrupt.user_navigation_requested());
        assert_eq!(interrupt.current_reason(), None);

        interrupt.set_deadline(Some(Instant::now() - Duration::from_secs(1)));
        assert_eq!(
            interrupt.current_reason(),
            Some(InterruptReason::DeadlineExceeded)
        );

        interrupt.request_user_navigation();
        interrupt.cancel();
        assert_eq!(interrupt.current_reason(), Some(InterruptReason::Cancelled));
        interrupt.begin_user_interaction();
        interrupt.set_deadline(None);
        assert_eq!(interrupt.current_reason(), Some(InterruptReason::Cancelled));
    }
}
