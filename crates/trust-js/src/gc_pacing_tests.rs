//! V8's heap sizing and memory reducer in the engine (see `gc_pacing`): the nursery grows while
//! its objects survive, and the memory reducer, armed by a major collection that left the heap
//! large or by possible garbage, collects a quiet heap through the embedder's
//! `Engine::run_memory_reducer` (here its `Interp` methods) and gives its memory back.
//!
//! The engine's unit tests run without pacing (their budgets must not depend on time), so each
//! test turns it on for its own engine, and drives the reducer with instants it chooses.

use std::time::{Duration, Instant};

use crate::gc_pacing::{MemoryReducer, LONG_DELAY, MAX_GCS, START_DELAY};
use crate::{Completion, Engine};

/// An engine whose heap follows V8's sizing policy and runs no stress collections.
fn engine() -> Engine {
    let engine = Engine::new();
    {
        let mut core = engine.interp.gc_heap.core.borrow_mut();
        core.set_stress_every(0);
        core.enable_pacing(true);
    }
    engine
}

fn eval(engine: &mut Engine, source: &str) {
    match engine.eval(source, false).expect("pacing fixture parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn reducer(engine: &Engine) -> MemoryReducer {
    engine
        .interp
        .gc_heap
        .core
        .borrow()
        .memory_reducer()
        .expect("pacing is on")
}

fn committed(engine: &Engine) -> usize {
    engine.interp.gc_heap.core.borrow().committed()
}

#[test]
fn the_nursery_grows_while_new_objects_survive() {
    let mut engine = engine();
    let start = engine.interp.gc_heap.core.borrow().nursery_budget();
    assert_eq!(start, Some(crate::gc_pacing::MIN_NURSERY_BYTES));
    // Every object made survives: the promoted bytes pass the budget, which doubles.
    eval(
        &mut engine,
        "globalThis.keep = []; for (let i = 0; i < 400000; i++) keep.push({ i, next: null });",
    );
    let grown = engine
        .interp
        .gc_heap
        .core
        .borrow()
        .nursery_budget()
        .unwrap();
    assert!(grown > start.unwrap(), "{grown}");
}

#[test]
fn the_memory_reducer_collects_a_quiet_heap_after_a_major_collection_left_it_large() {
    let mut engine = engine();
    // A major collection that keeps tens of MB arms the reducer for 8 s later.
    eval(
        &mut engine,
        "globalThis.big = []; for (let i = 0; i < 200000; i++) big.push({ i, s: 'x' + i });",
    );
    let before = Instant::now();
    engine.interp.collect_garbage_for_host();
    let deadline = engine
        .interp
        .memory_reducer_deadline()
        .expect("the reducer waits");
    assert!(deadline >= before + LONG_DELAY && deadline <= Instant::now() + LONG_DELAY);
    // An early timer leaves it waiting.
    assert!(!engine
        .interp
        .run_memory_reducer(deadline - Duration::from_secs(1)));
    assert_eq!(engine.interp.memory_reducer_deadline(), Some(deadline));
    // The objects die; at the deadline the program has been quiet, so it collects, and gives
    // the memory back.
    eval(&mut engine, "big = null;");
    let held = committed(&engine);
    let (all, minor) = engine.unstable_gc_collection_counts();
    assert!(engine.interp.run_memory_reducer(deadline));
    let (all_after, minor_after) = engine.unstable_gc_collection_counts();
    assert_eq!(
        (all_after - all, minor_after - minor),
        (1, 0),
        "one major collection"
    );
    assert!(
        committed(&engine) * 4 < held,
        "{} of {held} bytes held",
        committed(&engine)
    );
    // It freed memory, so a second collection follows after the short delay, and then it is
    // done: V8 runs at most two.
    let next = engine
        .interp
        .memory_reducer_deadline()
        .expect("a second run is due");
    assert!(engine.interp.run_memory_reducer(next));
    assert_eq!(MAX_GCS, 2);
    assert!(matches!(reducer(&engine), MemoryReducer::Done { .. }));
    assert_eq!(engine.interp.memory_reducer_deadline(), None);
}

#[test]
fn possible_garbage_and_growth_past_bootstrap_start_the_reducer() {
    let mut engine = engine();
    let before = Instant::now();
    engine.interp.notify_possible_garbage();
    let deadline = engine
        .interp
        .memory_reducer_deadline()
        .expect("the reducer waits");
    assert!(deadline >= before + START_DELAY);
    // A heap that grows 1 MB past its bootstrap before any major collection starts it too.
    let mut engine = self::engine();
    engine.interp.gc_heap.core.borrow_mut().bootstrapped();
    assert_eq!(engine.interp.memory_reducer_deadline(), None);
    eval(
        &mut engine,
        "globalThis.keep = []; for (let i = 0; i < 100000; i++) keep.push({ i });",
    );
    assert!(engine.interp.memory_reducer_deadline().is_some());
}

#[test]
fn a_reducer_timer_inside_js_execution_waits() {
    let mut engine = engine();
    engine.interp.notify_possible_garbage();
    let deadline = engine.interp.memory_reducer_deadline().unwrap();
    engine.interp.gc_heap.agent.suppressed.set(1);
    assert!(!engine.interp.run_memory_reducer(deadline));
    engine.interp.gc_heap.agent.suppressed.set(0);
    assert_eq!(
        engine.interp.memory_reducer_deadline(),
        Some(deadline + LONG_DELAY)
    );
}
