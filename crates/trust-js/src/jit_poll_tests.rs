//! The native JIT's polls see the cell heap's "collection due" (P1 S5 step 7b; ARM64, and
//! x86-64 from S6 step 7): its byte budget, stress interval and deferred requests, none of
//! which moves the live-object count, through the due word beside that count
//! (`value::LiveCount`). Entry and loop polls take their slow path while the word is set, a
//! general region's backedge guard bails to its canonical header (ARM64), and in baseline code
//! the calls of allocating helpers end in an explicit poll (`jit::emit_helper_end_poll`,
//! `x64::Asm::helper_end_poll`). x64 has no regions and no direct call sequence: its loops run
//! in baseline code and its calls through the checked helper.
//!
//! The programs run at `Tier::Jit` with collections on and, in debug builds, the heap verifier
//! after every marking. Each counts the
//! collections generated code's polls ran, by caller (`jit::take_poll_census`), checks that its
//! functions ran natively, and checks the due word against the heap's `collection_due` (as
//! `Interp::gc_check` does at every safepoint in debug builds). "Stress 1" is the heap setting
//! `TRUST_JS_GC_STRESS=1` gives every heap, a collection due after each cell allocation,
//! applied to the test's own heap. `jit::TEST_NO_END_POLLS` compiles without helper-end polls,
//! to watch the entry and loop polls alone.

use crate::bytecode::Tier;
use crate::interpreter::Interp;
use crate::value::Value;
use crate::Engine;

use super::rooting_tests::{assert_tracked_collectable, install, jit_engine, run};
use super::traced_tests::{assert_native, eval, settle};
use super::{take_poll_census, PollCensus, POLL_ENTRY, POLL_HELPER_END, POLL_LOOP};

const LOOP: usize = POLL_LOOP as usize;
const ENTRY: usize = POLL_ENTRY as usize;
const END: usize = POLL_HELPER_END as usize;

/// An engine at `Tier::Jit` with tier threshold 0.
fn engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(engine.interp.tier, Tier::Jit);
    engine
}

/// The heap's stress interval: a collection is due after every `every` cell allocations (0:
/// never).
fn stress(i: &Interp, every: usize) {
    i.gc_heap.core.borrow_mut().set_stress_every(every);
}

fn collections(i: &Interp) -> u64 {
    i.gc_heap.core.borrow().collections()
}

/// The due word generated code reads agrees with the heap's `collection_due`.
fn assert_due_word(i: &Interp) {
    let core = i.gc_heap.core.borrow();
    assert!(core.due_word_agrees());
    assert_eq!(
        crate::value::heap_collection_due(&i.gc_heap),
        core.collection_due()
    );
}

/// Region entries so far on this thread (`regions::executed_regions`); x64 has no regions.
fn executed_regions() -> u64 {
    #[cfg(target_arch = "aarch64")]
    {
        super::regions::executed_regions()
    }
    #[cfg(target_arch = "x86_64")]
    {
        0
    }
}

/// Run `f` with helper-end polls off for every chunk this thread compiles meanwhile.
fn without_end_polls<R>(f: impl FnOnce() -> R) -> R {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            super::TEST_NO_END_POLLS.with(|off| off.set(false));
        }
    }
    super::TEST_NO_END_POLLS.with(|off| off.set(true));
    let _restore = Restore;
    f()
}

/// Evaluate `source`, returning its result and what the polls of generated code did meanwhile.
fn measured(engine: &mut Engine, source: &str) -> (String, PollCensus) {
    take_poll_census();
    let result = eval(engine, source);
    (result, take_poll_census())
}

/// (1) Loop-only code at stress 1: a compiled loop that allocates through helpers and makes no
/// calls collects on (nearly) every iteration. `wide` runs in baseline code (its 16-element
/// literal is deeper than a region's operand stack): with helper-end polls each allocation is
/// collected where its helper returns, with them off at the loop header's poll, which now sees
/// the due word. On ARM64 `narrow` runs as a general region, whose backedge bails to the
/// canonical header's poll either way; on x86-64 it runs in baseline code, as `wide` does.
#[test]
fn loop_only_code_collects_every_iteration_at_stress_1() {
    const SOURCE: &str = r#"
        function wide(n) {
            var sum = 0, row = null;
            for (var i = 0; i < n; i++) {
                row = [{v: i}, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i];
                sum += row[0].v;
            }
            return sum;
        }
        function narrow(n) {
            var sum = 0, last = null;
            for (var i = 0; i < n; i++) {
                last = [i, {v: i}];
                sum += last[1].v;
            }
            return sum;
        }
        wide(3) + narrow(3);
    "#;
    const N: u64 = 200;
    for end_polls in [true, false] {
        let measure = || {
            let mut engine = engine();
            assert_eq!(eval(&mut engine, SOURCE), "6");
            assert_native(&mut engine, &["wide", "narrow"]);
            stress(&engine.interp, 1);
            ["wide", "narrow"].map(|function| {
                let entries = executed_regions();
                let (result, census) = measured(&mut engine, &format!("{function}({N});"));
                assert_eq!(result, (N * (N - 1) / 2).to_string());
                assert_due_word(&engine.interp);
                (census, executed_regions() - entries)
            })
        };
        let [(wide, wide_regions), (narrow, narrow_regions)] = if end_polls {
            measure()
        } else {
            without_end_polls(measure)
        };
        eprintln!("loop-only, end polls {end_polls}: wide {wide:?}, narrow {narrow:?}");
        assert_eq!(wide_regions, 0, "wide runs in baseline code");
        for census in [wide, narrow] {
            assert!(
                census.collections[LOOP] + census.collections[END] >= N - 2,
                "a collection per iteration: {census:?}"
            );
        }
        #[cfg(target_arch = "aarch64")]
        let baseline = {
            assert!(narrow_regions >= N - 2, "narrow re-enters its region");
            assert!(narrow.collections[LOOP] >= N - 2, "{narrow:?}");
            [wide]
        };
        #[cfg(target_arch = "x86_64")]
        let baseline = {
            assert_eq!(narrow_regions, 0, "narrow runs in baseline code");
            [wide, narrow]
        };
        for census in baseline {
            if end_polls {
                // The object's and the array's helpers: two collections per iteration.
                assert!(census.collections[END] >= 2 * N - 2, "{census:?}");
            } else {
                assert_eq!(census.calls[END], 0, "{census:?}");
                assert!(census.collections[LOOP] >= N - 2, "{census:?}");
            }
        }
    }
}

/// (2) Call-only code at stress 1: loop-free recursion through direct calls (on x86-64 through
/// the checked call helper). With helper-end polls off, the object each level allocates is
/// collected at the next callee's entry poll, the direct call's safepoint; with them on, where
/// the allocating helper returns.
#[test]
fn call_only_code_collects_at_callee_entry_polls_at_stress_1() {
    const SOURCE: &str = r#"
        function make(n) { return {n: n}; }
        function down(n) {
            if (n === 0) return 0;
            var o = make(n);
            return o.n + down(n - 1);
        }
        down(3);
    "#;
    const N: u64 = 100;
    for end_polls in [true, false] {
        let measure = || {
            let mut engine = engine();
            assert_eq!(eval(&mut engine, SOURCE), "6");
            assert_native(&mut engine, &["make", "down"]);
            settle(&mut engine, "make");
            settle(&mut engine, "down");
            stress(&engine.interp, 1);
            #[cfg(target_arch = "x86_64")]
            {
                let (result, census) = measured(&mut engine, &format!("down({N});"));
                assert_eq!(result, (N * (N + 1) / 2).to_string());
                assert_due_word(&engine.interp);
                census
            }
            // A concurrent test can exhaust the call-cache epoch, which sends calls through
            // the checked helper until their sites refill: measure a round that returned
            // through the direct sequence at every level.
            #[cfg(target_arch = "aarch64")]
            for _ in 0..100 {
                super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
                let (result, census) = measured(&mut engine, &format!("down({N});"));
                assert_eq!(result, (N * (N + 1) / 2).to_string());
                assert_due_word(&engine.interp);
                if super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.get()) as u64 >= 2 * N {
                    return census;
                }
            }
            #[cfg(target_arch = "aarch64")]
            panic!("down({N}) never returned through direct calls");
        };
        let census = if end_polls {
            measure()
        } else {
            without_end_polls(measure)
        };
        eprintln!("call-only, end polls {end_polls}: {census:?}");
        if end_polls {
            assert!(census.collections[END] >= N - 2, "{census:?}");
        } else {
            assert_eq!(census.calls[END], 0, "{census:?}");
            // x64's calls run through the call helper, whose amortized safepoint collects at
            // every call under TRUST_JS_GC_STRESS, ahead of the callee's entry poll.
            if cfg!(target_arch = "aarch64") || !crate::gc_test::stress_active() {
                assert!(census.collections[ENTRY] >= N - 2, "{census:?}");
            }
        }
    }
}

/// (3) The nursery budget, no stress: a loop allocating only strings, which move no live-object
/// count, collects within one poll of spending its budget, not at the interrupt divider: the
/// loop takes fewer polls than the divider's 16384, and the most bytes a collecting poll found
/// allocated past the budget stays within one iteration's allocations. (Until S6 step 10c the
/// byte budget made allocation's collections due; the nursery budget does now.)
#[test]
fn a_spent_byte_budget_collects_within_one_poll() {
    const SOURCE: &str = r#"
        function strings(n) {
            var s = '', total = 0;
            for (var i = 0; i < n; i++) {
                s = 'k' + i;
                total += s.length;
            }
            return total;
        }
        strings(3);
    "#;
    const N: u64 = 12_000;
    let mut engine = engine();
    stress(&engine.interp, 0);
    assert_eq!(eval(&mut engine, SOURCE), "6");
    assert_native(&mut engine, &["strings"]);
    // A nursery budget of a twentieth of the live heap, from a collection now.
    engine.interp.gc_collect();
    let stats = engine.interp.gc_heap.core.borrow().stats();
    let budget = stats.live_bytes / 20;
    engine
        .interp
        .gc_heap
        .core
        .borrow_mut()
        .set_nursery_bytes(Some(budget));
    let objects = engine.interp.live_object_count();
    let allocations = stats.allocations;
    let before = collections(&engine.interp);
    engine.interp.interrupt_poll_tick = 1; // the divider is due 16383 polls from here
    let (result, census) = measured(&mut engine, &format!("strings({N});"));
    let expected: u64 = (0..N).map(|i| 1 + i.to_string().len() as u64).sum();
    assert_eq!(result, expected.to_string());
    assert_due_word(&engine.interp);
    let collected = collections(&engine.interp) - before;
    let cells = engine.interp.gc_heap.core.borrow().stats().allocations - allocations;
    eprintln!(
        "nursery budget: {budget} bytes, {cells} cells over {N} iterations, {collected} \
         collections, {census:?}"
    );
    assert!(
        engine.interp.live_object_count() <= objects + 2,
        "the loop allocates no objects"
    );
    assert!(
        collected >= 4,
        "{collected} collections over {N} iterations: {census:?}"
    );
    assert_eq!(
        census.collections[LOOP] + census.collections[END] + census.collections[ENTRY],
        collected,
        "every collection ran at a poll of generated code"
    );
    // One iteration allocates two strings (the number's and the concatenation); a poll follows
    // each allocating helper, and a region's backedge each iteration.
    assert!(
        census.max_overshoot <= 256,
        "allocation passed the trigger by {} bytes",
        census.max_overshoot
    );
}

/// (4) A request deferred while JIT code runs (an explicit collection inside a no-GC region,
/// in the native `deferred()`) is honoured at the next poll, of each kind: in baseline code the
/// end of the native's call (`once`, `viaEntry`); with helper-end polls off the next callee's
/// entry (`viaEntry`) or, when `once` returns with no poll after the request, the next
/// activation's entry; in a loop (on ARM64 a region loop) the header its backedge reaches
/// (`honoured`). `count()` reads the collections run so far.
#[test]
fn a_deferred_request_is_honoured_at_the_next_poll() {
    fn deferred(i: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
        let before = collections(i);
        {
            let _region = i.gc_heap.core.borrow().nogc();
            i.gc_collect();
        }
        assert_eq!(collections(i), before, "the request was deferred");
        assert!(crate::value::heap_collection_due(&i.gc_heap));
        assert_due_word(i);
        Ok(Value::Num(before as f64))
    }
    fn count(i: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
        Ok(Value::Num(collections(i) as f64))
    }
    const SOURCE: &str = r#"
        function once() { var before = deferred(); return count() - before; }
        function next() { return count(); }
        function viaEntry() { var before = deferred(); return next() - before; }
        function honoured(n) {
            var seen = 0;
            for (var i = 0; i < n; i++) {
                var before = deferred();
                seen += count() - before;
            }
            return seen;
        }
        once() + viaEntry() + honoured(1);
    "#;
    const N: u64 = 20;
    for end_polls in [true, false] {
        let measure = || {
            let mut engine = engine();
            stress(&engine.interp, 0);
            install(&mut engine, "deferred", 0, deferred);
            install(&mut engine, "count", 0, count);
            eval(&mut engine, SOURCE);
            assert_native(&mut engine, &["once", "next", "viaEntry", "honoured"]);
            settle(&mut engine, "next");
            let regions = executed_regions();
            let calls = [
                "viaEntry();",
                &format!("honoured({N});"),
                "once();",
                "next();",
            ]
            .map(|call| {
                let (result, census) = measured(&mut engine, call);
                assert_due_word(&engine.interp);
                (result, census)
            });
            (calls, executed_regions() > regions)
        };
        let ([via_entry, honoured, once, after], region) = if end_polls {
            measure()
        } else {
            without_end_polls(measure)
        };
        eprintln!(
            "deferred, end polls {end_polls}: viaEntry {via_entry:?}, honoured {honoured:?} \
             (a region: {region}), once {once:?}, then {after:?}"
        );
        if crate::gc_test::stress_active() {
            continue; // the natives' calls collect as well
        }
        // Each request is honoured at its next poll: none waits for a later one.
        assert_eq!(via_entry.0, "1", "honoured before next() returned");
        if end_polls {
            assert_eq!(
                via_entry.1.collections,
                [0, 0, 1],
                "where deferred()'s call ended"
            );
            assert_eq!(once.0, "1", "honoured before count() ran");
            assert_eq!(
                once.1.collections,
                [0, 0, 1],
                "where deferred()'s call ended"
            );
            assert_eq!(after.1.collections, [0; 3], "nothing left pending");
        } else {
            assert_eq!(via_entry.1.collections, [0, 1, 0], "at next()'s entry poll");
            assert_eq!(once.0, "0");
            assert_eq!(
                once.1.collections, [0; 3],
                "no poll follows the request in once()"
            );
            assert_eq!(
                after.1.collections,
                [0, 1, 0],
                "at the next activation's entry"
            );
        }
        let census = honoured.1;
        assert_eq!(
            census.collections[LOOP] + census.collections[END],
            N,
            "{census:?}"
        );
        assert_eq!(
            honoured.0,
            census.collections[END].to_string(),
            "a request not honoured where deferred()'s call returned is honoured at the loop \
             header that follows"
        );
    }
}

/// (5) Operand temporaries live across helper-end polls at stress 1: in `f(a, {x: n},
/// [marks()])`, a collection runs where `{x: n}`'s helper returns while the callee and `a`, a
/// fresh object nothing else holds, wait on the caller's operand stack, and again where the
/// array's does. `marks()` reports what marked `a` first in the collection at `{x: n}`'s
/// poll: a root (the activation's operand words), not a stale stack copy. Every stress
/// collection is major here: a minor one would not mark `a`, old since the collection at the
/// poll after `make`'s allocation. Everything reads back, and nothing tracked outlives the
/// calls.
#[test]
fn operand_temporaries_survive_helper_end_polls() {
    let mut engine = jit_engine(0);
    stress(&engine.interp, 1);
    engine
        .interp
        .gc_heap
        .core
        .borrow_mut()
        .set_stress_major_every(1);
    let source = r#"
        function make(tag) { return {tag: tag}; }
        function f(a, o, arr) { return arr[0] + ':' + a.tag + ':' + o.x; }
        function g(n) { return f(track(make('A' + n)), {x: n}, [marks()]); }
        g(0);
    "#;
    assert_eq!(run(&mut engine, source), "R:A0:0");
    assert_native(&mut engine, &["make", "f", "g"]);
    settle(&mut engine, "make");
    settle(&mut engine, "f");
    for n in 1..4 {
        take_poll_census();
        assert_eq!(run(&mut engine, &format!("g({n});")), format!("R:A{n}:{n}"));
        let census = take_poll_census();
        eprintln!("operand temporaries, g({n}): {census:?}");
        assert!(census.collections[END] >= 2, "{census:?}");
        assert_due_word(&engine.interp);
    }
    assert_tracked_collectable(&mut engine);
}

/// (6) A general region's backedge bails to its canonical header while the due word is set.
/// At stress 1 the header's poll collects on every iteration and the loop re-enters the region.
/// With collections suppressed (`gc_suppressed`) or deferred (a no-GC region) the word stays
/// set: every backedge bails and every header poll calls the helper, which collects nothing,
/// yet each bail follows a whole iteration in the region and the loop completes. The request
/// left pending is honoured at the next poll after the no-GC region ends. On x86-64 the loop
/// runs in baseline code: at stress 1 its object's helper-end poll collects on every iteration,
/// and while the word stays set its header and end polls call the helper on every iteration.
#[test]
fn a_region_loop_completes_when_the_due_word_makes_it_bail() {
    const SOURCE: &str = r#"
        function regionLoop(n) {
            var sum = 0, o = null;
            for (var i = 0; i < n; i++) {
                o = {v: i, w: sum};
                sum += o.v;
            }
            return sum;
        }
        regionLoop(3);
    "#;
    const N: u64 = 200;
    let expected = (N * (N - 1) / 2).to_string();
    let mut engine = engine();
    assert_eq!(eval(&mut engine, SOURCE), "3");
    assert_native(&mut engine, &["regionLoop"]);
    let regions = executed_regions;
    // The loop's region entries: one per bail on ARM64; none on x86-64.
    let reentered = |entered: u64| {
        if cfg!(target_arch = "aarch64") {
            entered >= N - 2
        } else {
            entered == 0
        }
    };

    stress(&engine.interp, 1);
    let entries = regions();
    let (result, census) = measured(&mut engine, &format!("regionLoop({N});"));
    let entered = regions() - entries;
    eprintln!("region, stress 1: {entered} region entries, {census:?}");
    assert_eq!(result, expected);
    assert!(reentered(entered), "re-entered after each bail: {entered}");
    #[cfg(target_arch = "aarch64")]
    assert!(census.collections[LOOP] >= N - 2, "{census:?}");
    #[cfg(target_arch = "x86_64")]
    assert!(census.collections[END] >= N - 2, "{census:?}");
    assert_due_word(&engine.interp);

    engine.interp.suspend_gc();
    let entries = regions();
    let (result, census) = measured(&mut engine, &format!("regionLoop({N});"));
    let entered = regions() - entries;
    engine.interp.resume_gc();
    eprintln!("region, suppressed: {entered} region entries, {census:?}");
    assert_eq!(result, expected);
    assert!(reentered(entered), "an iteration per bail: {entered}");
    assert!(census.calls[LOOP] >= N - 2, "{census:?}");
    assert_eq!(census.collections, [0; 3], "suppressed");
    assert!(crate::value::heap_collection_due(&engine.interp.gc_heap));

    stress(&engine.interp, 0);
    let (result, census) = {
        let _region = engine.interp.gc_heap.core.borrow().nogc();
        engine.interp.gc_collect();
        let entries = regions();
        let outcome = measured(&mut engine, &format!("regionLoop({N});"));
        let entered = regions() - entries;
        eprintln!(
            "region, deferred: {entered} region entries, {:?}",
            outcome.1
        );
        assert!(reentered(entered), "an iteration per bail: {entered}");
        outcome
    };
    assert_eq!(result, expected);
    assert!(census.calls[LOOP] >= N - 2, "{census:?}");
    assert_eq!(census.collections, [0; 3], "deferred");
    assert!(engine.interp.gc_heap.core.borrow().collection_pending());
    assert_due_word(&engine.interp);

    let (result, census) = measured(&mut engine, "regionLoop(4);");
    assert_eq!(result, "6");
    assert!(
        census.collections.iter().sum::<u64>() >= 1,
        "the pending request ran at a poll: {census:?}"
    );
    assert!(!engine.interp.gc_heap.core.borrow().collection_pending());
    assert_due_word(&engine.interp);
}
