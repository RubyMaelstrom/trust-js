//! trust-js-timers — the timer globals (`setTimeout`, `setInterval`, `clearTimeout`,
//! `clearInterval`, `setImmediate`) as an op crate.
//!
//! The ops only mutate the [`Timers`] heap in `OpState`; nothing here sleeps, spawns, or
//! fires. The runtime's event loop drives everything: it asks [`Timers::next_deadline`] how
//! long it may block, and fires [`Timers::take_due`] callbacks each turn. `setImmediate`
//! doesn't touch the heap at all — it queues on the loop's [`CallbackQueue`] for the next
//! turn.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::time::{Duration, Instant};

use trust_js_host::{ops, CallbackQueue, Ctx, Extension, Value};

/// The extension a runtime installs: the five timer globals plus the [`Timers`] state.
pub fn extension() -> Extension {
    Extension {
        name: "timers",
        globals: ops![
            "setTimeout" (2) => op_set_timeout,
            "setInterval" (2) => op_set_interval,
            "clearTimeout" (1) => op_clear_timer,
            "clearInterval" (1) => op_clear_timer,
            "setImmediate" (1) => op_set_immediate,
        ],
        namespaces: &[],
        state_init: Some(|state| state.put(Timers::default())),
        js_init: None,
        js_init_snapshot: None,
    }
}

#[derive(Clone)]
pub enum TimerHandler {
    Function(Value),
    Code(String),
}

struct Entry {
    handler: TimerHandler,
    args: Vec<Value>,
    timeout: Duration,
    repeat: bool,
    /// Unique internal value for the currently armed invocation. A due task carries this token so
    /// clearing (and eventual ID reuse) after task queuing still makes that task a no-op.
    handle: u64,
    task_nesting: u32,
    armed: bool,
}

/// One queued task from the HTML timer task source. The runtime must call [`Timers::begin_task`]
/// immediately before invocation and [`Timers::finish_task`] immediately afterward.
pub struct TimerTask {
    id: u32,
    handle: u64,
    pub handler: TimerHandler,
    pub args: Vec<Value>,
    nesting: u32,
}

/// The timer heap. Cancellation is lazy: `clear*` removes the entry; stale heap nodes are
/// skipped (and popped) when they surface, so `clearTimeout` is O(1).
#[derive(Default)]
pub struct Timers {
    next_id: u32,
    next_handle: u64,
    heap: BinaryHeap<Reverse<(Instant, u64, u32)>>,
    entries: HashMap<u32, Entry>,
    /// Timer nesting level of the currently running timer task, or zero outside one.
    current_nesting: u32,
}

impl Timers {
    fn allocate_id(&mut self) -> u32 {
        assert!(
            self.entries.len() < i32::MAX as usize,
            "setTimeout/setInterval ID space exhausted"
        );
        loop {
            self.next_id = if self.next_id >= i32::MAX as u32 {
                1
            } else {
                self.next_id + 1
            };
            if !self.entries.contains_key(&self.next_id) {
                return self.next_id;
            }
        }
    }

    fn allocate_handle(&mut self) -> u64 {
        self.next_handle = self
            .next_handle
            .checked_add(1)
            .expect("timer unique-handle space exhausted");
        self.next_handle
    }

    fn schedule(
        &mut self,
        handler: TimerHandler,
        args: Vec<Value>,
        timeout: Duration,
        repeat: bool,
    ) -> u32 {
        let id = self.allocate_id();
        let handle = self.allocate_handle();
        let nesting = self.current_nesting;
        let delay = effective_timeout(timeout, nesting);
        self.entries.insert(
            id,
            Entry {
                handler,
                args,
                timeout,
                repeat,
                handle,
                task_nesting: nesting.saturating_add(1),
                armed: true,
            },
        );
        self.heap
            .push(Reverse((Instant::now() + delay, handle, id)));
        id
    }

    fn clear(&mut self, id: u32) {
        self.entries.remove(&id);
    }

    /// Whether any live timer remains (the loop stays alive while true).
    pub fn has_pending(&self) -> bool {
        !self.entries.is_empty()
    }

    /// When the loop may sleep until. Pops cancelled heap nodes so a cleared timer can't
    /// produce a busy-wakeup loop.
    pub fn next_deadline(&mut self) -> Option<Instant> {
        while let Some(Reverse((deadline, handle, id))) = self.heap.peek().copied() {
            if self
                .entries
                .get(&id)
                .is_some_and(|entry| entry.armed && entry.handle == handle)
            {
                return Some(deadline);
            }
            self.heap.pop();
        }
        None
    }

    /// Tasks due at `now`, earliest first. Repeating timers are deliberately not rearmed here:
    /// HTML recursively runs timer initialization *after* the handler, from a fresh start time.
    /// This also bounds a zero-delay interval to one queued task instead of looping forever or
    /// bulk-emitting every missed period.
    pub fn take_due(&mut self, now: Instant) -> Vec<TimerTask> {
        let mut due = Vec::new();
        while let Some(Reverse((deadline, handle, id))) = self.heap.peek().copied() {
            if deadline > now {
                break;
            }
            self.heap.pop();
            let Some(entry) = self.entries.get_mut(&id) else {
                continue; // cancelled
            };
            if !entry.armed || entry.handle != handle {
                continue;
            }
            entry.armed = false;
            due.push(TimerTask {
                id,
                handle,
                handler: entry.handler.clone(),
                args: entry.args.clone(),
                nesting: entry.task_nesting,
            });
        }
        due
    }

    /// Validate a queued task's unique handle and expose its nesting level to timer calls made by
    /// the handler (including exception reporting). A timer cleared after queuing fails here.
    pub fn begin_task(&mut self, task: &TimerTask) -> bool {
        let valid = self
            .entries
            .get(&task.id)
            .is_some_and(|entry| !entry.armed && entry.handle == task.handle);
        self.current_nesting = if valid { task.nesting } else { 0 };
        valid
    }

    /// Complete the timer task before the event loop's microtask checkpoint. A live interval is
    /// reinitialized from `now`; a one-shot is removed. Clearing from the callback wins because it
    /// removes the entry (or changes its handle) before this check.
    pub fn finish_task(&mut self, task: &TimerTask, now: Instant) {
        self.current_nesting = 0;
        let Some(entry) = self.entries.get(&task.id) else {
            return;
        };
        if entry.armed || entry.handle != task.handle {
            return;
        }
        if !entry.repeat {
            self.entries.remove(&task.id);
            return;
        }
        let delay = effective_timeout(entry.timeout, task.nesting);
        let next_nesting = task.nesting.saturating_add(1);
        let handle = self.allocate_handle();
        let entry = self
            .entries
            .get_mut(&task.id)
            .expect("validated timer disappeared without author code");
        entry.handle = handle;
        entry.task_nesting = next_nesting;
        entry.armed = true;
        self.heap.push(Reverse((now + delay, handle, task.id)));
    }
}

/// HTML §8.7 timer initialization: timer tasks at nesting levels above five clamp sub-4 ms waits.
fn effective_timeout(timeout: Duration, nesting: u32) -> Duration {
    if nesting > 5 && timeout < Duration::from_millis(4) {
        Duration::from_millis(4)
    } else {
        timeout
    }
}

/// Web IDL `long` conversion (ConvertToInt with 32 bits and signedness "signed").
fn webidl_long(number: f64) -> i32 {
    if !number.is_finite() || number == 0.0 {
        return 0;
    }
    let modulo = number.trunc().rem_euclid(4_294_967_296.0);
    if modulo >= 2_147_483_648.0 {
        (modulo - 4_294_967_296.0) as i32
    } else {
        modulo as i32
    }
}

/// WHATWG HTML timer initialization steps. Web IDL converts the handler union and `long` delay
/// before the algorithm clamps negative and deeply nested timeouts.
fn schedule_op(ctx: &mut Ctx, args: &[Value], repeat: bool) -> Result<Value, Value> {
    let handler = match args.first() {
        Some(callback) if callback.is_callable() => TimerHandler::Function(callback.clone()),
        Some(code) => TimerHandler::Code(ctx.coerce_string(code)?.to_string()),
        None => {
            let kind = if repeat { "setInterval" } else { "setTimeout" };
            return Err(ctx.make_error("TypeError", format!("{kind} requires a handler")));
        }
    };
    let timeout = match args.get(1) {
        Some(value) => webidl_long(ctx.coerce_number(value)?),
        None => 0,
    };
    let delay = Duration::from_millis(timeout.max(0) as u64);
    let extra: Vec<Value> = args.iter().skip(2).cloned().collect();
    let timers = ctx.host_mut::<Timers>().expect("timers state installed");
    let id = timers.schedule(handler, extra, delay, repeat);
    Ok(Value::Num(id as f64))
}

fn op_set_timeout(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    schedule_op(ctx, args, false)
}

fn op_set_interval(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    schedule_op(ctx, args, true)
}

/// Shared by `clearTimeout`/`clearInterval` (per spec either clears either kind). Unknown or
/// non-numeric ids are ignored.
fn op_clear_timer(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let id = match args.first() {
        Some(value) => webidl_long(ctx.coerce_number(value)?),
        None => 0,
    };
    let timers = ctx.host_mut::<Timers>().expect("timers state installed");
    timers.clear(id as u32);
    Ok(Value::Undefined)
}

/// Queue for the next loop turn (after microtasks, before timers get another look).
fn op_set_immediate(ctx: &mut Ctx, _this: Value, args: &[Value]) -> Result<Value, Value> {
    let callback = match args.first() {
        Some(cb) if cb.is_callable() => cb.clone(),
        _ => return Err(ctx.make_error("TypeError", "setImmediate expects a function")),
    };
    let extra: Vec<Value> = args.iter().skip(1).cloned().collect();
    CallbackQueue::enqueue(ctx.op_state(), callback, extra);
    Ok(Value::Undefined)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inert_handler() -> TimerHandler {
        TimerHandler::Function(Value::Undefined)
    }

    #[test]
    fn webidl_long_conversion_wraps_before_html_clamping() {
        assert_eq!(webidl_long(f64::NAN), 0);
        assert_eq!(webidl_long(f64::INFINITY), 0);
        assert_eq!(webidl_long(-1.9), -1);
        assert_eq!(webidl_long(4_294_967_296.0), 0);
        assert_eq!(webidl_long(4_294_967_301.0), 5);
        assert_eq!(webidl_long(2_147_483_648.0), i32::MIN);
    }

    #[test]
    fn nesting_clamp_starts_only_above_level_five() {
        assert_eq!(
            effective_timeout(Duration::from_millis(0), 5),
            Duration::ZERO
        );
        assert_eq!(
            effective_timeout(Duration::from_millis(3), 6),
            Duration::from_millis(4)
        );
        assert_eq!(
            effective_timeout(Duration::from_millis(4), 99),
            Duration::from_millis(4)
        );
    }

    #[test]
    fn delayed_interval_queues_once_and_restarts_from_completion_time() {
        let mut timers = Timers::default();
        timers.schedule(inert_handler(), Vec::new(), Duration::from_millis(10), true);
        let long_after_deadline = Instant::now() + Duration::from_secs(10);
        let mut due = timers.take_due(long_after_deadline);
        assert_eq!(due.len(), 1, "missed periods were emitted in bulk");
        assert!(timers.take_due(long_after_deadline).is_empty());

        let task = due.pop().unwrap();
        assert!(timers.begin_task(&task));
        timers.finish_task(&task, long_after_deadline);
        assert_eq!(
            timers.next_deadline(),
            Some(long_after_deadline + Duration::from_millis(10))
        );
    }

    #[test]
    fn zero_delay_interval_produces_at_most_one_task_per_reinitialization() {
        let mut timers = Timers::default();
        timers.schedule(inert_handler(), Vec::new(), Duration::ZERO, true);
        let now = Instant::now();
        let mut due = timers.take_due(now);
        assert_eq!(due.len(), 1);
        assert!(timers.take_due(now).is_empty());

        let task = due.pop().unwrap();
        assert!(timers.begin_task(&task));
        timers.finish_task(&task, now);
        assert_eq!(timers.take_due(now).len(), 1);
    }

    #[test]
    fn clearing_a_queued_timer_invalidates_its_unique_handle_check() {
        let mut timers = Timers::default();
        let id = timers.schedule(inert_handler(), Vec::new(), Duration::ZERO, false);
        let task = timers.take_due(Instant::now()).pop().unwrap();
        timers.clear(id);
        assert!(!timers.begin_task(&task));
        assert!(!timers.has_pending());
    }
}
