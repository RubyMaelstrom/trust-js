//! The interrupt handle's state, the poll words it keeps and the deadline watchdog.

use super::{watchdog, InterruptReason, PollWord, RuntimeInterrupt};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};
use watchdog::Watchdog;

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

fn word() -> Arc<PollWord> {
    Arc::new(PollWord::default())
}

/// Wait up to `limit` for `done`, polling every millisecond.
fn eventually(limit: Duration, done: impl Fn() -> bool) -> bool {
    let start = Instant::now();
    while !done() {
        if start.elapsed() > limit {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

/// A request sets the interrupt bit of every word the handle keeps (a handle shared among
/// engines keeps one per interpreter); clearing a reason clears it unless another remains.
#[test]
fn requests_set_and_clears_recheck_every_registered_word() {
    let interrupt = RuntimeInterrupt::default();
    let (first, second) = (word(), word());
    interrupt.register(&first);
    interrupt.register(&second);
    let bits = || (first.interrupt_pending(), second.interrupt_pending());
    assert_eq!(bits(), (false, false));

    interrupt.request_user_navigation();
    assert_eq!(bits(), (true, true));
    interrupt.begin_user_interaction();
    assert_eq!(bits(), (false, false));

    interrupt.set_deadline(Some(Instant::now() + Duration::from_secs(60)));
    assert_eq!(
        bits(),
        (false, false),
        "an armed deadline is not yet a reason"
    );
    interrupt.set_deadline(Some(Instant::now()));
    assert_eq!(
        bits(),
        (true, true),
        "a deadline set in the past expires at once"
    );
    interrupt.request_user_navigation();
    interrupt.begin_user_interaction();
    assert_eq!(bits(), (true, true), "the expired deadline remains");
    interrupt.set_deadline(None);
    assert_eq!(bits(), (false, false));

    interrupt.request_user_navigation();
    interrupt.cancel();
    interrupt.begin_user_interaction();
    assert_eq!(bits(), (true, true), "cancellation is permanent");
    interrupt.set_deadline(None);
    assert_eq!(bits(), (true, true));
}

/// A word registered late takes the handle's current state; an unregistered word no longer
/// follows it; ended words are forgotten.
#[test]
fn registration_follows_the_handle_and_forgets_ended_words() {
    let old = RuntimeInterrupt::default();
    let new = RuntimeInterrupt::default();
    let word = word();
    old.register(&word);
    new.cancel();
    assert!(!word.interrupt_pending());

    // The swap an interpreter makes in `Interp::set_runtime_interrupt`.
    old.unregister(&word);
    new.register(&word);
    assert!(
        word.interrupt_pending(),
        "registration writes the current state"
    );
    old.request_user_navigation();
    old.begin_user_interaction();
    assert!(
        word.interrupt_pending(),
        "the old handle no longer writes it"
    );
    assert_eq!(old.registered_words(), 0);
    assert_eq!(new.registered_words(), 1);

    let calm = RuntimeInterrupt::default();
    let ended = self::word();
    calm.register(&ended);
    calm.register(&word);
    assert!(!word.interrupt_pending(), "a calm handle clears the bit");
    drop(ended);
    assert_eq!(calm.registered_words(), 1);
    calm.cancel();
    assert!(word.interrupt_pending());
}

/// A deadline in the future expires when the watchdog thread sees it pass: then, and not
/// before, the bit is set and `current_reason` reports it from the expiry alone.
#[test]
fn the_watchdog_expires_an_armed_deadline() {
    let interrupt = RuntimeInterrupt::default();
    let word = word();
    interrupt.register(&word);
    let deadline = Instant::now() + Duration::from_millis(30);
    interrupt.set_deadline(Some(deadline));
    assert!(!word.interrupt_pending());
    assert!(
        eventually(Duration::from_secs(10), || word.interrupt_pending()),
        "the watchdog never reported the deadline"
    );
    assert!(Instant::now() >= deadline, "reported before it passed");
    assert_eq!(
        interrupt.current_reason(),
        Some(InterruptReason::DeadlineExceeded)
    );
    assert_eq!(interrupt.armed_deadline(), Some(deadline));
    interrupt.set_deadline(None);
    assert!(!word.interrupt_pending());
    assert_eq!(interrupt.current_reason(), None);
}

/// A deadline replaced or disarmed before it passes never expires, and leaves no entry with
/// the watchdog; nor does one whose handle ends.
#[test]
fn replaced_disarmed_and_dropped_deadlines_never_expire() {
    let interrupt = RuntimeInterrupt::default();
    let word = word();
    interrupt.register(&word);
    let soon = Instant::now() + Duration::from_millis(20);
    interrupt.set_deadline(Some(soon));
    let later = Instant::now() + Duration::from_secs(3600);
    interrupt.set_deadline(Some(later));
    let disarmed = Instant::now() + Duration::from_millis(20);
    let other = RuntimeInterrupt::default();
    other.set_deadline(Some(disarmed));
    other.set_deadline(None);
    let dropped = Instant::now() + Duration::from_secs(7200);
    let gone = RuntimeInterrupt::default();
    gone.set_deadline(Some(dropped));
    drop(gone);
    let entries = watchdog::WATCHDOG.entries();
    assert!(entries.iter().any(|(at, _)| *at == later));
    assert!(!entries
        .iter()
        .any(|(at, _)| *at == soon || *at == disarmed || *at == dropped));
    std::thread::sleep(Duration::from_millis(60));
    assert!(!word.interrupt_pending());
    assert_eq!(interrupt.current_reason(), None);
    assert_eq!(other.current_reason(), None);
    interrupt.set_deadline(None);
    assert!(!watchdog::WATCHDOG
        .entries()
        .iter()
        .any(|(at, _)| *at == later));
}

/// A navigation request racing an interaction that clears it, from two threads: whichever
/// change comes last, the bit agrees with the state once both have returned.
#[test]
fn a_request_racing_a_clear_is_never_lost() {
    let interrupt = Arc::new(RuntimeInterrupt::default());
    let word = word();
    interrupt.register(&word);
    let rounds = 2000;
    let barrier = Arc::new(Barrier::new(2));
    let clearer = std::thread::spawn({
        let interrupt = interrupt.clone();
        let barrier = barrier.clone();
        move || {
            for _ in 0..rounds {
                barrier.wait();
                interrupt.begin_user_interaction();
                barrier.wait();
                barrier.wait();
            }
        }
    });
    let mut set = 0;
    for _ in 0..rounds {
        barrier.wait();
        interrupt.request_user_navigation();
        barrier.wait();
        // Both have returned: the bit says what the state says.
        let requested = interrupt.user_navigation_requested();
        assert_eq!(word.interrupt_pending(), requested);
        set += usize::from(requested);
        interrupt.begin_user_interaction();
        barrier.wait();
    }
    clearer.join().expect("clearer");
    eprintln!("{set} of {rounds} requests outlived the racing clear");
}

/// A thread blocked for the execution (`while_blocked`) wakes for the watchdog's expiry of the
/// deadline armed while it waits, and then observes it.
#[test]
fn a_blocked_thread_wakes_when_the_watchdog_expires_its_deadline() {
    let interrupt = Arc::new(RuntimeInterrupt::default());
    let state = Arc::new((std::sync::Mutex::new(()), std::sync::Condvar::new()));
    let waiter = std::thread::spawn({
        let interrupt = interrupt.clone();
        let state = state.clone();
        move || {
            let wake = super::BlockedWake(Arc::new({
                let state = state.clone();
                move || {
                    let _held = state.0.lock().unwrap();
                    state.1.notify_all();
                }
            }));
            interrupt.while_blocked(wake, || {
                let mut held = state.0.lock().unwrap();
                // Wait without a timeout of its own: only a wake can end this.
                while interrupt.current_reason().is_none() {
                    held = state.1.wait(held).unwrap();
                }
                interrupt.current_reason()
            })
        }
    });
    std::thread::sleep(Duration::from_millis(10));
    interrupt.set_deadline(Some(Instant::now() + Duration::from_millis(20)));
    assert_eq!(
        waiter.join().expect("waiter"),
        Some(InterruptReason::DeadlineExceeded)
    );
}

/// A watchdog of the test's own, whose thread outlives the last deadline by `idle`.
fn watchdog(idle: Duration) -> &'static Watchdog {
    Box::leak(Box::new(Watchdog::new(idle)))
}

/// The watchdog's lifecycle (Decision 18): no thread before a deadline is armed; the first
/// starts one, which serves a burst of armings and clearings (TRust's around each URLPattern
/// match) and the expiries that follow; it exits once nothing has been armed for its idle
/// time, and the next deadline starts a fresh one.
#[test]
fn the_watchdog_thread_serves_a_burst_and_exits_when_idle() {
    let dog = watchdog(Duration::from_millis(50));
    assert_eq!(dog.threads(), (false, 0), "no thread before a deadline");
    let interrupt = RuntimeInterrupt::with_watchdog(dog);
    let word = word();
    interrupt.register(&word);
    for _ in 0..1000 {
        interrupt.set_deadline(Some(Instant::now() + Duration::from_secs(60)));
        interrupt.set_deadline(None);
    }
    assert_eq!(dog.threads(), (true, 1), "a burst keeps one thread");
    assert!(dog.entries().is_empty());
    for _ in 0..5 {
        interrupt.set_deadline(Some(Instant::now() + Duration::from_millis(2)));
        assert!(eventually(Duration::from_secs(10), || word.interrupt_pending()));
        interrupt.set_deadline(None);
    }
    assert_eq!(dog.threads().1, 1, "expiries keep the same thread");
    assert!(
        eventually(Duration::from_secs(10), || !dog.threads().0),
        "the thread exits once idle"
    );
    interrupt.set_deadline(Some(Instant::now() + Duration::from_millis(2)));
    assert_eq!(
        dog.threads(),
        (true, 2),
        "the next deadline starts a fresh thread"
    );
    assert!(eventually(Duration::from_secs(10), || word.interrupt_pending()));
    interrupt.set_deadline(None);
}

/// A deadline armed while the idle thread decides to exit is never lost: the arming and the
/// exit take the queue's lock, so the deadline lands before the exit and the old thread serves
/// it, or after it and starts the next thread. The rounds arm the next deadline at a spread of
/// moments around the thread's 1 ms idle time.
#[test]
fn a_deadline_armed_as_the_thread_exits_is_never_lost() {
    let dog = watchdog(Duration::from_millis(1));
    let interrupt = RuntimeInterrupt::with_watchdog(dog);
    let word = word();
    interrupt.register(&word);
    for round in 0..300u32 {
        interrupt.set_deadline(Some(Instant::now() + Duration::from_micros(200)));
        assert!(
            eventually(Duration::from_secs(10), || word.interrupt_pending()),
            "round {round}: the deadline was lost"
        );
        interrupt.set_deadline(None);
        std::thread::sleep(Duration::from_micros(250 * u64::from(round % 7)));
    }
    let (_, started) = dog.threads();
    eprintln!("300 deadlines on {started} threads");
    assert!(started > 1, "the thread exited between rounds: {started}");
}

/// What a deadline costs (`--ignored`, in a release build): arming and clearing one while the
/// thread runs, as in a burst, and the time from a deadline to its bit for a thread started by
/// that deadline and for a thread already running.
#[test]
#[ignore]
fn watchdog_costs() {
    fn median(mut samples: Vec<f64>) -> (f64, f64) {
        samples.sort_by(f64::total_cmp);
        (samples[samples.len() / 2], samples[samples.len() * 9 / 10])
    }
    let dog = watchdog(Duration::from_secs(5));
    let interrupt = RuntimeInterrupt::with_watchdog(dog);
    interrupt.set_deadline(Some(Instant::now() + Duration::from_secs(60)));
    interrupt.set_deadline(None);
    let pairs = 200_000;
    let mut runs = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..pairs {
            interrupt.set_deadline(Some(start + Duration::from_secs(60)));
            interrupt.set_deadline(None);
        }
        runs.push(start.elapsed().as_nanos() as f64 / pairs as f64);
    }
    let (burst, _) = median(runs);
    eprintln!(
        "arm and clear during a burst: {burst:.0} ns a pair ({} thread)",
        dog.threads().1
    );

    // From the deadline to its bit: spin on the word, the watchdog on another core.
    let wake = |interrupt: &RuntimeInterrupt, word: &PollWord, ahead: Duration| -> f64 {
        let at = Instant::now() + ahead;
        interrupt.set_deadline(Some(at));
        while !word.interrupt_pending() {
            std::hint::spin_loop();
        }
        let late = Instant::now().saturating_duration_since(at);
        interrupt.set_deadline(None);
        late.as_secs_f64() * 1e6
    };
    let word = word();
    for ahead in [Duration::from_micros(50), Duration::from_millis(2)] {
        let fresh = watchdog(Duration::from_millis(1));
        let interrupt = RuntimeInterrupt::with_watchdog(fresh);
        interrupt.register(&word);
        let mut samples = Vec::new();
        for _ in 0..40 {
            while fresh.threads().0 {
                std::thread::sleep(Duration::from_millis(1));
            }
            samples.push(wake(&interrupt, &word, ahead));
        }
        let (p50, p90) = median(samples);
        eprintln!(
            "fresh thread, deadline {ahead:?} ahead: first wake {p50:.0} us after it (p90 \
             {p90:.0}; {} threads)",
            fresh.threads().1
        );
        interrupt.unregister(&word);
        let warm = watchdog(Duration::from_secs(5));
        let interrupt = RuntimeInterrupt::with_watchdog(warm);
        interrupt.register(&word);
        let samples = (0..40).map(|_| wake(&interrupt, &word, ahead)).collect();
        let (p50, p90) = median(samples);
        eprintln!(
            "running thread, deadline {ahead:?} ahead: wake {p50:.0} us after it (p90 {p90:.0})"
        );
        interrupt.unregister(&word);
    }
}
