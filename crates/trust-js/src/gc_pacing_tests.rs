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

/// An engine whose heap follows V8's sizing policy and runs no stress collections, and does not
/// verify its heap. Verification, on in the engine's unit tests, walks the whole heap at every
/// collection and after every slice of a marking cycle: the collections' measured speeds, which
/// the policy reads, then counted those walks (a 28 MB cycle's 470 slices took 144 s in a debug
/// build, 15 ms each at its start and 555 ms at its end), and on a loaded or thermally capped
/// machine the policy's verdicts turned (the reducer saw a 117 B/ms collector, the nursery a
/// program allocating under 1 MB/s). The heap's integrity has its own tests.
fn engine() -> Engine {
    let engine = Engine::new();
    {
        let mut core = engine.interp.gc_heap.core.borrow_mut();
        core.set_stress_every(0);
        core.enable_pacing(true);
        core.set_heap_verification(false);
    }
    engine
}

fn eval(engine: &mut Engine, source: &str) {
    match engine.eval(source, false).expect("pacing fixture parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

/// The memory reducer's own deadline (the engine's `memory_reducer_deadline`, a host's, also
/// counts the spare page pool's, W6 step 5).
fn reducer_deadline(engine: &Engine) -> Option<Instant> {
    reducer(engine).deadline()
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
    let deadline = reducer_deadline(&engine).expect("the reducer waits");
    assert!(deadline >= before + LONG_DELAY && deadline <= Instant::now() + LONG_DELAY);
    // An early timer leaves it waiting.
    assert!(!engine
        .interp
        .run_memory_reducer(deadline - Duration::from_secs(1)));
    assert_eq!(reducer_deadline(&engine), Some(deadline));
    // The objects die; at the deadline the program has been quiet, so it collects, and gives
    // the memory back.
    eval(&mut engine, "big = null;");
    let held = committed(&engine);
    let (all, minor) = engine.unstable_gc_collection_counts();
    assert!(engine.interp.run_memory_reducer(deadline));
    // With incremental marking the reducer starts a cycle (W6 step 4), which the host's idle
    // steps finish: its final pause is the one major collection.
    while engine.interp.advance_incremental_marking() {}
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
    let next = reducer_deadline(&engine).expect("a second run is due");
    assert!(engine.interp.run_memory_reducer(next));
    while engine.interp.advance_incremental_marking() {}
    assert_eq!(MAX_GCS, 2);
    assert!(matches!(reducer(&engine), MemoryReducer::Done { .. }));
    assert_eq!(engine.interp.memory_reducer_deadline(), None);
}

/// W6 step 4: with incremental marking the memory reducer starts a marking cycle instead of a
/// collection, as V8's starts incremental marking, and the host's idle steps
/// (`Engine::advance_incremental_marking`) take the cycle to its final pause, which gives the
/// memory back as the reducer's collection does.
#[test]
fn the_memory_reducer_starts_a_cycle_that_idle_steps_finish() {
    let mut engine = engine();
    engine
        .interp
        .gc_heap
        .core
        .borrow_mut()
        .set_incremental(crate::gc_core::IncrementalMode::On);
    eval(
        &mut engine,
        "globalThis.big = []; for (let i = 0; i < 200000; i++) big.push({ i, s: 'x' + i });",
    );
    engine.interp.collect_garbage_for_host();
    let deadline = reducer_deadline(&engine).expect("the reducer waits");
    eval(&mut engine, "big = null;");
    let held = committed(&engine);
    let counts = engine.unstable_gc_collection_counts();
    assert!(engine.interp.run_memory_reducer(deadline));
    assert!(engine.interp.incremental_marking_running(), "a cycle marks");
    assert_eq!(
        engine.unstable_gc_collection_counts(),
        counts,
        "no collection yet"
    );
    let mut steps = 0;
    while engine.interp.advance_incremental_marking() {
        steps += 1;
        assert!(steps < 1_000_000, "the idle steps finish the cycle");
    }
    let (all, minor) = engine.unstable_gc_collection_counts();
    assert_eq!(
        (all - counts.0, minor - counts.1),
        (1, 0),
        "the cycle's final pause"
    );
    assert!(
        committed(&engine) * 4 < held,
        "{} of {held} bytes held",
        committed(&engine)
    );
}

/// V8's reducer waits while incremental marking runs (src/heap/memory-reducer.cc:86-87: its
/// timer can start a GC only while marking is stopped): a timer that finds a cycle marking
/// collects nothing, puts its collection off by its long delay, and leaves the cycle running, the
/// program's; the cycle's final pause then rearms the reducer as any major collection does.
#[test]
fn a_reducer_timer_while_a_cycle_marks_waits() {
    let mut engine = engine();
    engine
        .interp
        .gc_heap
        .core
        .borrow_mut()
        .set_incremental(crate::gc_core::IncrementalMode::On);
    eval(
        &mut engine,
        "globalThis.big = []; for (let i = 0; i < 200000; i++) big.push({ i, s: 'x' + i });",
    );
    engine.interp.collect_garbage_for_host();
    let deadline = reducer_deadline(&engine).expect("the reducer waits");
    eval(&mut engine, "big = null;");
    assert!(engine.interp.test_start_cycle_for_host(), "a cycle marks");
    let counts = engine.unstable_gc_collection_counts();
    assert!(!engine.interp.run_memory_reducer(deadline), "no collection");
    assert!(
        engine.interp.incremental_marking_running(),
        "the cycle runs on"
    );
    assert_eq!(engine.unstable_gc_collection_counts(), counts);
    assert!(
        matches!(
            reducer(&engine),
            MemoryReducer::Wait { started_gcs: 0, next_gc_start, .. }
                if next_gc_start == deadline + LONG_DELAY
        ),
        "{:?}",
        reducer(&engine)
    );
    // The cycle's final pause is a major collection: the reducer waits from there.
    while engine.interp.advance_incremental_marking() {}
    let (all, _) = engine.unstable_gc_collection_counts();
    assert_eq!(all, counts.0 + 1, "the cycle's final pause");
    let rearmed = reducer_deadline(&engine).expect("the reducer waits again");
    assert!(rearmed >= Instant::now() + LONG_DELAY - Duration::from_secs(1));
}

#[test]
fn possible_garbage_and_growth_past_bootstrap_start_the_reducer() {
    let mut engine = engine();
    let before = Instant::now();
    engine.interp.notify_possible_garbage();
    let deadline = reducer_deadline(&engine).expect("the reducer waits");
    assert!(deadline >= before + START_DELAY);
    // A heap that grows 1 MB past its bootstrap before any major collection starts it too.
    let mut engine = self::engine();
    engine.interp.gc_heap.core.borrow_mut().bootstrapped();
    assert_eq!(engine.interp.memory_reducer_deadline(), None);
    eval(
        &mut engine,
        "globalThis.keep = []; for (let i = 0; i < 100000; i++) keep.push({ i });",
    );
    assert!(reducer_deadline(&engine).is_some());
}

#[test]
fn a_reducer_timer_inside_js_execution_waits() {
    let mut engine = engine();
    engine.interp.notify_possible_garbage();
    let deadline = reducer_deadline(&engine).unwrap();
    engine.interp.gc_heap.agent.suppressed.set(1);
    assert!(!engine.interp.run_memory_reducer(deadline));
    engine.interp.gc_heap.agent.suppressed.set(0);
    assert_eq!(reducer_deadline(&engine), Some(deadline + LONG_DELAY));
}
