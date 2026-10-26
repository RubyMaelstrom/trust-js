//! The heap's sizing and memory-reduction policy, after V8's: how far the heap may grow before
//! its next major collection (V8's `MemoryController`, src/heap/heap-controller.cc), how large
//! the nursery is (V8's young generation growth and shrinking, src/heap/heap.cc), and when an
//! otherwise quiet heap is collected to give memory back (V8's `MemoryReducer`,
//! src/heap/memory-reducer.cc). Constants and rules as of V8 296ba5e0 (2026-10-26). Chrome runs
//! no idle-time or periodic full collection; V8's memory reducer is the only time-based one.
//!
//! The policy is pure: [`Pacing`] holds what the heap measured (`gc_core::Heap` feeds it the
//! collections' times and bytes), and the engine runs the memory reducer's timer when the host
//! asks (`Engine::run_memory_reducer`). A heap without pacing (the heap core's own tests, and
//! targets without a clock) keeps a fixed growth factor and nursery.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// V8's `kTargetMutatorUtilization`: the share of time the heap's growth aims to leave to the
/// program rather than to major collections.
const TARGET_MUTATOR_UTILIZATION: f64 = 0.97;
/// V8's `kMinGrowingFactor`, also its growth for a collection that reduces memory.
pub(crate) const MIN_GROWING_FACTOR: f64 = 1.1;
/// V8's `kConservativeGrowingFactor`: the most a heap grows while the memory reducer, having
/// run, is done (V8's `kSlow` growing mode).
const SLOW_GROWING_FACTOR: f64 = 1.3;
/// V8's `kMaxGrowingFactor`: its factor for a heap whose limit is at least 2 GB on 64-bit
/// targets.
pub(crate) const MAX_GROWING_FACTOR: f64 = 4.0;
/// V8's `MinimumAllocationLimitGrowingStep`: a major collection leaves at least this much room.
pub(crate) const MIN_GROWING_STEP: usize = 8 << 20;

/// The nursery's first and smallest budget, to which a quiet heap returns (V8's semi-space
/// starts at 1 MB; this engine's cells are larger).
pub(crate) const MIN_NURSERY_BYTES: usize = 2 << 20;
/// The nursery's largest budget: V8's largest semi-space with 4 GB of memory or more.
pub(crate) const MAX_NURSERY_BYTES: usize = 32 << 20;
/// V8's `kLowAllocationThroughput`, in bytes per millisecond: below it, the nursery shrinks.
const LOW_ALLOCATION_THROUGHPUT: f64 = 1000.0;

/// V8's `kHighMutatorUtilization` for the memory reducer's "low allocation rate".
const HIGH_MUTATOR_UTILIZATION: f64 = 0.993;
/// V8's `kConservativeGcSpeedInBytesPerMillisecond`, assumed before a speed is measured.
const CONSERVATIVE_GC_SPEED: f64 = 200_000.0;
/// The span of recent allocation the throughputs average (V8 smooths them exponentially,
/// with half-lives of 100 ms rising and 1 s falling).
const THROUGHPUT_WINDOW: Duration = Duration::from_millis(5000);
/// How many recent events of each kind the speeds and throughputs average: V8's ring buffers'.
const EVENTS: usize = 10;
/// No bytes over no time: an average of the recorded events alone.
const NOTHING: (usize, Duration) = (0, Duration::ZERO);

/// V8's `MemoryController::DynamicGrowingFactor`: the factor by which a major collection lets
/// the heap grow so that, at the measured speeds, major collections take about
/// `1 - TARGET_MUTATOR_UTILIZATION` of the time. A collector fast relative to the program's
/// allocation keeps the heap tight; a slow one lets it grow, up to `max_factor`.
pub(crate) fn dynamic_growing_factor(gc_speed: f64, mutator_speed: f64, max_factor: f64) -> f64 {
    if gc_speed <= 0.0 || mutator_speed <= 0.0 {
        return max_factor;
    }
    let speed_ratio = gc_speed / mutator_speed;
    let a = speed_ratio * (1.0 - TARGET_MUTATOR_UTILIZATION);
    let b = a - TARGET_MUTATOR_UTILIZATION;
    // The factor is a / b, but b may be small (or not positive): then the maximum.
    let factor = if a < b * max_factor {
        a / b
    } else {
        max_factor
    };
    factor.min(max_factor).max(MIN_GROWING_FACTOR)
}

/// V8's `Heap::ComputeMutatorUtilization`: the share of time left to a program allocating at
/// `mutator_speed` while a collector working at `gc_speed` keeps up.
fn mutator_utilization(mutator_speed: f64, gc_speed: Option<f64>) -> f64 {
    if mutator_speed <= 0.0 {
        return 1.0;
    }
    let gc_speed = gc_speed.unwrap_or(CONSERVATIVE_GC_SPEED);
    gc_speed / (mutator_speed + gc_speed)
}

/// Bytes over time: the last [`EVENTS`] events.
#[derive(Clone, Debug, Default)]
struct Events {
    events: VecDeque<(usize, Duration)>,
}

impl Events {
    fn push(&mut self, bytes: usize, time: Duration) {
        if self.events.len() == EVENTS {
            self.events.pop_front();
        }
        self.events.push_back((bytes, time));
    }

    /// V8's `GCTracer::AverageSpeed`, in bytes per millisecond: `initial` (bytes over a span
    /// not yet recorded) and then the events, newest first, until they cover `window` (all of
    /// them without one); clamped to 1 B/ms..1 GiB/ms, and `None` while no time was measured.
    fn speed(&self, initial: (usize, Duration), window: Option<Duration>) -> Option<f64> {
        let (mut bytes, mut time) = initial;
        for &(b, t) in self.events.iter().rev() {
            if window.is_some_and(|window| time >= window) {
                break;
            }
            bytes = bytes.saturating_add(b);
            time += t;
        }
        let millis = time.as_secs_f64() * 1000.0;
        (millis > 0.0).then(|| (bytes as f64 / millis).clamp(1.0, (1 << 30) as f64))
    }
}

/// The heap's measurements and the sizes it derives from them (see the module documentation).
#[derive(Debug)]
pub(crate) struct Pacing {
    /// The nursery's budget, `None` without a nursery (every collection major).
    nursery: Option<usize>,
    /// Whether the nursery's budget follows the measurements (off when a diagnostic or an
    /// explicit setting fixes it).
    dynamic_nursery: bool,
    /// Whether a major collection's budget grows by the measured factor (off once an explicit
    /// setting fixes the growth).
    dynamic_growth: bool,
    /// Bytes promoted since the nursery last grew: past its budget, it grows.
    survived_since_growth: usize,
    /// When the current collection started, the bytes allocated since the last one ended, and
    /// the program's time since allocation was last sampled, to which the collection's
    /// promotion belongs.
    started: Option<Instant>,
    allocated_at_start: usize,
    mutator: Option<Duration>,
    /// When allocation was last sampled (V8's `SampleAllocation`: as each collection starts and
    /// ends, and at the memory reducer's timer), and the bytes allocated since the last
    /// collection then.
    last_sample: Option<Instant>,
    sampled: usize,
    /// Allocation between samples: all of it (into the nursery, or the heap without one), and
    /// what reached the old generation.
    allocation: Events,
    old_allocation: Events,
    /// The collections' speeds over their pauses: a minor collection's in the young bytes it
    /// processed, a major one's in the bytes it marked.
    minor_speed: Events,
    major_speed: Events,
    /// The collection in progress reduces memory (the memory reducer's).
    reduce_memory: bool,
    pub(crate) reducer: MemoryReducer,
}

impl Pacing {
    pub(crate) fn new(nursery: Option<usize>, dynamic_nursery: bool) -> Self {
        Self {
            nursery,
            dynamic_nursery: dynamic_nursery && nursery.is_some(),
            dynamic_growth: true,
            survived_since_growth: 0,
            started: None,
            mutator: None,
            allocated_at_start: 0,
            last_sample: None,
            sampled: 0,
            allocation: Events::default(),
            old_allocation: Events::default(),
            minor_speed: Events::default(),
            major_speed: Events::default(),
            reduce_memory: false,
            reducer: MemoryReducer::default(),
        }
    }

    /// The nursery's budget.
    pub(crate) fn nursery(&self) -> Option<usize> {
        self.nursery
    }

    /// Fix the nursery's budget at `nursery` (an explicit setting, `Heap::set_nursery_bytes`).
    pub(crate) fn fix_nursery(&mut self, nursery: Option<usize>) {
        self.nursery = nursery;
        self.dynamic_nursery = false;
    }

    /// Fix the heap's growth (an explicit setting, `Heap::set_heap_growth`): major collections
    /// return no factor, and the heap applies its own.
    pub(crate) fn fix_growth(&mut self) {
        self.dynamic_growth = false;
    }

    /// Make the next collection reduce memory: minimal growth, the smallest nursery.
    pub(crate) fn reduce_memory_next(&mut self) {
        self.reduce_memory = true;
    }

    /// Whether the collection in progress, or the next, reduces memory.
    pub(crate) fn reducing_memory(&self) -> bool {
        self.reduce_memory
    }

    /// Sample allocation at `now` (the memory reducer's timer), `allocated` bytes having been
    /// allocated since the last collection ended: what was allocated since the last sample, and
    /// nothing promoted, over the time since.
    pub(crate) fn sample(&mut self, now: Instant, allocated: usize) {
        if let Some(time) = self.since_sample(now, allocated) {
            self.old_allocation.push(0, time);
        }
    }

    /// Record the allocation since the last sample, as of `now`, and return its time.
    fn since_sample(&mut self, now: Instant, allocated: usize) -> Option<Duration> {
        let time = self
            .last_sample
            .map(|last| now.saturating_duration_since(last));
        if let Some(time) = time {
            self.allocation
                .push(allocated.saturating_sub(self.sampled), time);
        }
        self.last_sample = Some(now);
        self.sampled = allocated;
        time
    }

    /// A collection starts at `now`, `allocated` bytes having been allocated since the last
    /// one ended.
    pub(crate) fn start(&mut self, now: Instant, allocated: usize) {
        self.started = Some(now);
        self.allocated_at_start = allocated;
        self.mutator = self.since_sample(now, allocated);
    }

    /// The collection that `start` began ends at `now`, having marked `marked` bytes (a minor
    /// collection's are the bytes it promoted). For a major one, returns the factor by which
    /// the heap may grow until the next (the caller applies [`MIN_GROWING_STEP`] and the
    /// budget's floor). The heap then reports a major collection's memory to the reducer
    /// ([`MemoryReducer::mark_compact`]), once it has trimmed its spare pages.
    pub(crate) fn end(&mut self, now: Instant, major: bool, marked: usize) -> Option<f64> {
        let pause = self.started.take().map_or(Duration::ZERO, |started| {
            now.saturating_duration_since(started)
        });
        // What reached the old generation while the program ran: a minor collection's
        // promotion, or without a nursery everything allocated. (A major collection with a
        // nursery promotes too, but its share of the heap it marks is unknown: V8 samples its
        // old generation's allocation at minor collections too.)
        let old_growth = match (major, self.nursery) {
            (false, _) => Some(marked),
            (true, None) => Some(self.allocated_at_start),
            (true, Some(_)) => None,
        };
        if let (Some(bytes), Some(time)) = (old_growth, self.mutator.take()) {
            self.old_allocation.push(bytes, time);
        }
        let reduce = std::mem::take(&mut self.reduce_memory);
        let factor = if major {
            self.major_speed.push(marked, pause);
            self.dynamic_growth.then(|| {
                if reduce {
                    MIN_GROWING_FACTOR
                } else {
                    let gc_speed = self.major_speed.speed(NOTHING, None);
                    let mutator_speed = self.old_allocation.speed(NOTHING, Some(THROUGHPUT_WINDOW));
                    let factor = dynamic_growing_factor(
                        gc_speed.unwrap_or(0.0),
                        mutator_speed.unwrap_or(0.0),
                        MAX_GROWING_FACTOR,
                    );
                    if self.reducer.grow_slowly() {
                        factor.min(SLOW_GROWING_FACTOR)
                    } else {
                        factor
                    }
                }
            })
        } else {
            // V8's young generation speed: the young bytes a collection processed (all that was
            // allocated since the last one), not just those it promoted.
            self.minor_speed.push(self.allocated_at_start, pause);
            None
        };
        self.resize_nursery(major, marked, reduce);
        self.last_sample = Some(now);
        self.sampled = 0;
        factor
    }

    /// V8's young generation sizing: the nursery doubles, up to [`MAX_NURSERY_BYTES`], once the
    /// bytes promoted since it last grew pass its budget (a program whose new objects survive),
    /// and returns to [`MIN_NURSERY_BYTES`] while allocation is slow or memory is being reduced.
    fn resize_nursery(&mut self, major: bool, marked: usize, reduce: bool) {
        let Some(nursery) = self.nursery.filter(|_| self.dynamic_nursery) else {
            return;
        };
        if !major {
            self.survived_since_growth = self.survived_since_growth.saturating_add(marked);
        }
        let throughput = self.allocation.speed(NOTHING, Some(THROUGHPUT_WINDOW));
        if reduce || throughput.is_some_and(|speed| speed < LOW_ALLOCATION_THROUGHPUT) {
            self.nursery = Some(MIN_NURSERY_BYTES);
            self.survived_since_growth = 0;
        } else if self.survived_since_growth > nursery && nursery < MAX_NURSERY_BYTES {
            self.nursery = Some((nursery * 2).min(MAX_NURSERY_BYTES));
            self.survived_since_growth = 0;
        }
    }

    /// V8's `Heap::HasLowAllocationRate`: at `now`, `allocated` bytes having been allocated
    /// since the last collection ended, both generations allocate slowly enough that collecting
    /// them would take under 0.7% of the time.
    pub(crate) fn low_allocation_rate(&self, now: Instant, allocated: usize) -> bool {
        let window = Some(THROUGHPUT_WINDOW);
        let since = self
            .last_sample
            .map_or(Duration::ZERO, |last| now.saturating_duration_since(last));
        let young = self
            .allocation
            .speed((allocated.saturating_sub(self.sampled), since), window)
            .unwrap_or(0.0);
        let old = self.old_allocation.speed((0, since), window).unwrap_or(0.0);
        let minor_speed = self.minor_speed.speed(NOTHING, None);
        let major_speed = self.major_speed.speed(NOTHING, None);
        mutator_utilization(young, minor_speed.or(major_speed)) > HIGH_MUTATOR_UTILIZATION
            && mutator_utilization(old, major_speed) > HIGH_MUTATOR_UTILIZATION
    }
}

/// V8's `MemoryReducer` (src/heap/memory-reducer.cc): after a major collection that left the
/// heap holding noticeably more memory than when the reducer last ran, or when the host says
/// much may have died (a document went away), it waits [`LONG_DELAY`]; then, if the program
/// allocates slowly (or [`WATCHDOG_DELAY`] passed without a major collection), it collects to
/// reduce memory, up to [`MAX_GCS`] times, [`SHORT_DELAY`] apart while each frees memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum MemoryReducer {
    /// Never ran; otherwise as `Done`.
    Uninit {
        last_gc: Option<Instant>,
    },
    Done {
        last_gc: Option<Instant>,
        committed_at_last_run: usize,
    },
    Wait {
        started_gcs: u32,
        next_gc_start: Instant,
        last_gc: Option<Instant>,
    },
    Run {
        started_gcs: u32,
    },
}

impl Default for MemoryReducer {
    fn default() -> Self {
        Self::Uninit { last_gc: None }
    }
}

/// V8's `memory_reducer_delay_ms` (formerly `kLongDelayMs`).
pub(crate) const LONG_DELAY: Duration = Duration::from_millis(8000);
/// V8's `kShortDelayMs`.
pub(crate) const SHORT_DELAY: Duration = Duration::from_millis(500);
/// V8's `kWatchdogDelayMs`.
pub(crate) const WATCHDOG_DELAY: Duration = Duration::from_millis(100_000);
/// V8's `memory_reducer_gc_count`.
pub(crate) const MAX_GCS: u32 = 2;
/// V8's `kCommittedMemoryFactor` and `kCommittedMemoryDelta`: a major collection that leaves
/// the heap holding less than max(this factor ×, this much more than) what the reducer's last
/// run left does not start it.
const COMMITTED_FACTOR: f64 = 1.1;
const COMMITTED_DELTA: usize = 10 << 20;
/// V8's `gc_memory_reducer_start_delay_ms`: the wait after possible garbage is reported.
pub(crate) const START_DELAY: Duration = Duration::from_millis(30_000);
/// V8's `HasHighFragmentation` slack: memory held beyond twice the used bytes and this much.
const HIGH_FRAGMENTATION_SLACK: usize = 16 << 20;

impl MemoryReducer {
    /// When the host should next run the reducer's timer.
    pub(crate) fn deadline(&self) -> Option<Instant> {
        match *self {
            Self::Wait { next_gc_start, .. } => Some(next_gc_start),
            Self::Uninit { .. } | Self::Done { .. } | Self::Run { .. } => None,
        }
    }

    /// V8's `ShouldGrowHeapSlowly`: the reducer ran and is done.
    fn grow_slowly(&self) -> bool {
        matches!(self, Self::Done { .. })
    }

    /// At `now`, much of the heap may just have died (the host let a document go, or the heap
    /// grew past its bootstrap without a major collection yet): wait [`START_DELAY`].
    pub(crate) fn possible_garbage(&mut self, now: Instant) {
        if let Self::Uninit { last_gc } | Self::Done { last_gc, .. } = *self {
            *self = Self::Wait {
                started_gcs: 0,
                next_gc_start: now + START_DELAY,
                last_gc,
            };
        }
    }

    /// The timer at `now`, the heap holding `committed` bytes from the system. `start_gc` is
    /// whether the program allocates slowly, `can_start` whether a collection can run now.
    /// Returns whether to start a memory-reducing collection (the state is then `Run`).
    pub(crate) fn timer(
        &mut self,
        now: Instant,
        committed: usize,
        start_gc: bool,
        can_start: bool,
    ) -> bool {
        let Self::Wait {
            started_gcs,
            next_gc_start,
            last_gc,
        } = *self
        else {
            return false;
        };
        if started_gcs >= MAX_GCS {
            *self = Self::Done {
                last_gc,
                committed_at_last_run: committed,
            };
            return false;
        }
        let watchdog = last_gc.is_some_and(|last| now > last + WATCHDOG_DELAY);
        if can_start && (start_gc || watchdog) {
            if next_gc_start > now {
                return false;
            }
            *self = Self::Run {
                started_gcs: started_gcs + 1,
            };
            return true;
        }
        *self = Self::Wait {
            started_gcs,
            next_gc_start: now + LONG_DELAY,
            last_gc,
        };
        false
    }

    /// A major collection, of any cause, ended at `now`, the heap holding `committed` bytes
    /// from the system after it (`used` of them in cells) and `committed_before` as it started.
    pub(crate) fn mark_compact(
        &mut self,
        now: Instant,
        committed_before: usize,
        committed: usize,
        used: usize,
    ) {
        let fragmented = committed.saturating_sub(used) > used + HIGH_FRAGMENTATION_SLACK;
        let likely_more = committed_before > committed + (1 << 20) || fragmented;
        *self = match *self {
            Self::Uninit { .. } | Self::Done { .. } => self.after_mark_compact(now, committed),
            Self::Wait { started_gcs, .. } => Self::Wait {
                started_gcs,
                next_gc_start: now + LONG_DELAY,
                last_gc: Some(now),
            },
            Self::Run { started_gcs } => {
                if started_gcs < MAX_GCS && (likely_more || started_gcs == 1) {
                    Self::Wait {
                        started_gcs,
                        next_gc_start: now + SHORT_DELAY,
                        last_gc: Some(now),
                    }
                } else {
                    Self::Done {
                        last_gc: Some(now),
                        committed_at_last_run: committed,
                    }
                }
            }
        };
    }

    /// From `Uninit` (as a `Done` that never ran) or `Done`: wait after a major collection that
    /// left the heap holding more than max(1.1 ×, 10 MB more than) the last run left.
    fn after_mark_compact(self, now: Instant, committed: usize) -> Self {
        let (ran, committed_at_last_run) = match self {
            Self::Done {
                committed_at_last_run,
                ..
            } => (true, committed_at_last_run),
            _ => (false, 0),
        };
        let threshold = ((committed_at_last_run as f64 * COMMITTED_FACTOR) as usize)
            .max(committed_at_last_run + COMMITTED_DELTA);
        if committed >= threshold {
            Self::Wait {
                started_gcs: 0,
                next_gc_start: now + LONG_DELAY,
                last_gc: Some(now),
            }
        } else if ran {
            Self::Done {
                last_gc: Some(now),
                committed_at_last_run,
            }
        } else {
            Self::Uninit { last_gc: Some(now) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: usize = 1 << 20;

    #[test]
    fn growing_factor_follows_v8s_mutator_utilization_target() {
        // No speeds measured yet: the maximum, as V8.
        assert_eq!(dynamic_growing_factor(0.0, 1.0, 4.0), 4.0);
        assert_eq!(dynamic_growing_factor(1.0, 0.0, 4.0), 4.0);
        // A collector 100 times faster than allocation: a / b = 3 / 2.03.
        let factor = dynamic_growing_factor(100.0, 1.0, 4.0);
        assert!((factor - 3.0 / 2.03).abs() < 1e-9, "{factor}");
        // Very fast: the minimum. Slow (b not positive): the maximum.
        assert_eq!(dynamic_growing_factor(1e9, 1.0, 4.0), MIN_GROWING_FACTOR);
        assert_eq!(dynamic_growing_factor(10.0, 1.0, 4.0), 4.0);
        assert_eq!(dynamic_growing_factor(10.0, 1.0, 2.0), 2.0);
    }

    /// A minor collection at `*now` that promotes `promoted` of `allocated` bytes allocated over
    /// `gap` and pauses 100 us.
    fn minor(
        pacing: &mut Pacing,
        now: &mut Instant,
        gap: Duration,
        allocated: usize,
        promoted: usize,
    ) {
        *now += gap;
        pacing.start(*now, allocated);
        *now += Duration::from_micros(100);
        pacing.end(*now, false, promoted);
    }

    #[test]
    fn the_nursery_grows_while_its_survivors_pass_it_and_shrinks_when_allocation_is_slow() {
        let t0 = Instant::now();
        let mut now = t0;
        let mut pacing = Pacing::new(Some(MIN_NURSERY_BYTES), true);
        // Each minor collection promotes the whole nursery, allocated in 1 ms: the promoted bytes
        // pass the budget at every second collection, and it doubles.
        let full = |pacing: &Pacing| pacing.nursery().unwrap();
        for _ in 0..2 {
            let bytes = full(&pacing);
            minor(
                &mut pacing,
                &mut now,
                Duration::from_millis(1),
                bytes,
                bytes,
            );
        }
        assert_eq!(pacing.nursery(), Some(4 * MB));
        for _ in 0..10 {
            let bytes = full(&pacing);
            minor(
                &mut pacing,
                &mut now,
                Duration::from_millis(1),
                bytes,
                bytes,
            );
        }
        assert_eq!(pacing.nursery(), Some(MAX_NURSERY_BYTES));
        // Allocation then slows below 1 KB/ms over the throughput window: back to the minimum.
        for _ in 0..EVENTS {
            minor(&mut pacing, &mut now, Duration::from_secs(10), MB, 0);
        }
        assert_eq!(pacing.nursery(), Some(MIN_NURSERY_BYTES));
        // A memory-reducing collection returns it to the minimum too, with the minimal growth.
        let mut pacing = Pacing::new(Some(MAX_NURSERY_BYTES), true);
        pacing.reduce_memory_next();
        pacing.start(t0, MB);
        assert_eq!(
            pacing.end(t0 + Duration::from_millis(1), true, MB),
            Some(MIN_GROWING_FACTOR)
        );
        assert_eq!(pacing.nursery(), Some(MIN_NURSERY_BYTES));
        // A fixed nursery stays.
        let mut pacing = Pacing::new(Some(MIN_NURSERY_BYTES), false);
        let mut now = t0;
        for _ in 0..4 {
            minor(
                &mut pacing,
                &mut now,
                Duration::from_millis(1),
                2 * MB,
                2 * MB,
            );
        }
        assert_eq!(pacing.nursery(), Some(MIN_NURSERY_BYTES));
    }

    #[test]
    fn a_major_collections_factor_reflects_the_collectors_speed_against_promotion() {
        // The first major collection, with no promotion measured: the maximum, as V8.
        let t0 = Instant::now();
        let mut pacing = Pacing::new(Some(MIN_NURSERY_BYTES), false);
        pacing.start(t0, 0);
        assert_eq!(
            pacing.end(t0 + Duration::from_millis(10), true, 100 * MB),
            Some(MAX_GROWING_FACTOR)
        );
        // 2 MB promoted per 2 ms against 10 MB marked per ms: a ratio of 10, so a collector slow
        // relative to promotion, and the maximum.
        let mut now = t0 + Duration::from_millis(10);
        for _ in 0..5 {
            minor(
                &mut pacing,
                &mut now,
                Duration::from_millis(2),
                2 * MB,
                2 * MB,
            );
        }
        pacing.start(now, MB);
        let factor = pacing.end(now + Duration::from_millis(10), true, 100 * MB);
        assert_eq!(factor, Some(MAX_GROWING_FACTOR));
        // 1 MB promoted per second: the collector is 10,000 times faster, and keeps the heap
        // tight.
        let mut pacing = Pacing::new(Some(MIN_NURSERY_BYTES), false);
        let mut now = t0;
        pacing.start(now, 0);
        now += Duration::from_millis(10);
        pacing.end(now, true, 100 * MB);
        for _ in 0..4 {
            minor(&mut pacing, &mut now, Duration::from_secs(1), 2 * MB, MB);
        }
        pacing.start(now, MB);
        let factor = pacing
            .end(now + Duration::from_millis(10), true, 100 * MB)
            .unwrap();
        assert!(factor < 1.2, "{factor}");
        // Without a nursery, all allocation counts as old.
        let mut pacing = Pacing::new(None, true);
        let mut now = t0;
        pacing.start(now, 0);
        now += Duration::from_millis(10);
        pacing.end(now, true, 100 * MB);
        now += Duration::from_secs(1);
        pacing.start(now, MB);
        let factor = pacing
            .end(now + Duration::from_millis(10), true, 100 * MB)
            .unwrap();
        assert!(factor < 1.2, "{factor}");
    }

    #[test]
    fn low_allocation_rate_counts_the_quiet_time_since_the_last_collection() {
        let t0 = Instant::now();
        let mut now = t0;
        let mut pacing = Pacing::new(Some(MIN_NURSERY_BYTES), false);
        // 2 MB per ms, against minor collections processing 20 MB per ms: busy.
        for _ in 0..4 {
            minor(
                &mut pacing,
                &mut now,
                Duration::from_millis(1),
                2 * MB,
                2 * MB / 10,
            );
        }
        assert!(!pacing.low_allocation_rate(now, 0));
        // Six quiet seconds later the window holds only quiet time.
        assert!(pacing.low_allocation_rate(now + Duration::from_secs(6), 1000));
        // Unless the program allocated much since: 8 GiB in 6 s against minor collections that
        // process 20 MB per ms.
        assert!(!pacing.low_allocation_rate(now + Duration::from_secs(6), 8192 * MB));
    }

    #[test]
    fn memory_reducer_follows_v8s_states() {
        let t0 = Instant::now();
        let mut reducer = MemoryReducer::default();
        // A small heap's major collection leaves it done.
        reducer.mark_compact(t0, 5 * MB, 5 * MB, 4 * MB);
        assert_eq!(reducer, MemoryReducer::Uninit { last_gc: Some(t0) });
        // One past 10 MB starts the long wait.
        reducer.mark_compact(t0, 20 * MB, 20 * MB, 19 * MB);
        assert_eq!(reducer.deadline(), Some(t0 + LONG_DELAY));
        // An early timer keeps the wait; a busy program waits another long delay.
        assert!(!reducer.timer(t0 + Duration::from_secs(1), 20 * MB, true, true));
        assert_eq!(reducer.deadline(), Some(t0 + LONG_DELAY));
        let busy = t0 + LONG_DELAY;
        assert!(!reducer.timer(busy, 20 * MB, false, true));
        assert_eq!(reducer.deadline(), Some(busy + LONG_DELAY));
        // A quiet program: the first collection, then a second after the short delay, which
        // ends the run (V8 collects at most twice).
        let quiet = busy + LONG_DELAY;
        assert!(reducer.timer(quiet, 20 * MB, true, true));
        reducer.mark_compact(quiet, 20 * MB, 10 * MB, 8 * MB);
        assert_eq!(reducer.deadline(), Some(quiet + SHORT_DELAY));
        assert!(reducer.timer(quiet + SHORT_DELAY, 10 * MB, true, true));
        reducer.mark_compact(quiet + SHORT_DELAY, 9 * MB, 9 * MB, 8 * MB);
        assert_eq!(
            reducer,
            MemoryReducer::Done {
                last_gc: Some(quiet + SHORT_DELAY),
                committed_at_last_run: 9 * MB,
            }
        );
        // Growth to under max(1.1x, +10 MB) of that does not restart it; past it does.
        reducer.mark_compact(quiet, 18 * MB, 18 * MB, 17 * MB);
        assert_eq!(reducer.deadline(), None);
        reducer.mark_compact(quiet, 20 * MB, 20 * MB, 19 * MB);
        assert_eq!(reducer.deadline(), Some(quiet + LONG_DELAY));
        // The watchdog collects a busy program 100 s after the last major collection.
        let late = quiet + WATCHDOG_DELAY + Duration::from_millis(1);
        assert!(reducer.timer(late, 20 * MB, false, true));
        // Possible garbage starts the longer wait; a run ends after MAX_GCS collections.
        let mut reducer = MemoryReducer::default();
        reducer.possible_garbage(t0);
        assert_eq!(reducer.deadline(), Some(t0 + START_DELAY));
        let mut now = t0 + START_DELAY;
        for _ in 0..MAX_GCS {
            assert!(reducer.timer(now, 40 * MB, true, true));
            reducer.mark_compact(now, 40 * MB, 20 * MB, 10 * MB);
            now += SHORT_DELAY;
        }
        assert_eq!(reducer.deadline(), None);
        // A collection that cannot start now waits the long delay.
        let mut reducer = MemoryReducer::default();
        reducer.possible_garbage(t0);
        assert!(!reducer.timer(t0 + START_DELAY, 0, true, false));
        assert_eq!(reducer.deadline(), Some(t0 + START_DELAY + LONG_DELAY));
        // A second run follows the first only if the first freed memory or left the heap
        // fragmented: here it left 40 MB held for 10 MB used (more than twice and 16 MB more).
        let mut reducer = MemoryReducer::default();
        reducer.possible_garbage(t0);
        assert!(reducer.timer(t0 + START_DELAY, 40 * MB, true, true));
        reducer.mark_compact(t0 + START_DELAY, 40 * MB, 40 * MB, 10 * MB);
        assert_eq!(reducer.deadline(), Some(t0 + START_DELAY + SHORT_DELAY));
    }

    #[test]
    fn explicit_settings_fix_the_nursery_and_the_growth() {
        let t0 = Instant::now();
        let mut pacing = Pacing::new(Some(MIN_NURSERY_BYTES), true);
        pacing.fix_nursery(Some(256 << 10));
        pacing.fix_growth();
        let mut now = t0;
        for _ in 0..4 {
            minor(&mut pacing, &mut now, Duration::from_millis(1), MB, MB);
        }
        assert_eq!(pacing.nursery(), Some(256 << 10));
        pacing.start(now, MB);
        assert_eq!(
            pacing.end(now + Duration::from_millis(10), true, 100 * MB),
            None
        );
        // Allocation is still sampled after a major collection that applies no factor.
        assert!(!pacing.low_allocation_rate(now + Duration::from_millis(10), 64 * MB));
    }

    #[test]
    fn a_heap_grows_slowly_once_the_memory_reducer_has_run() {
        let t0 = Instant::now();
        let mut pacing = Pacing::new(Some(MIN_NURSERY_BYTES), false);
        pacing.start(t0, 0);
        assert_eq!(
            pacing.end(t0 + Duration::from_millis(10), true, 100 * MB),
            Some(MAX_GROWING_FACTOR)
        );
        pacing.reducer = MemoryReducer::Done {
            last_gc: Some(t0),
            committed_at_last_run: 100 * MB,
        };
        pacing.start(t0 + Duration::from_secs(1), 0);
        assert_eq!(
            pacing.end(
                t0 + Duration::from_secs(1) + Duration::from_millis(10),
                true,
                100 * MB
            ),
            Some(SLOW_GROWING_FACTOR)
        );
    }
}
