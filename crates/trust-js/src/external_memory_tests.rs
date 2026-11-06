//! Memory the collector frees but does not allocate in its pages counts toward its triggers.
//! A dead Map's tables, a bound function's arguments or an error's stack trace live in the Rust
//! heap until a collection finds their owner dead, so a script that keeps replacing one large
//! owner must collect as it goes, not only once its cells fill the nursery: each loop below
//! allocates a few cells per iteration and megabytes of payload.

use crate::{Completion, Engine};

/// An engine whose heap runs no stress collections, so the triggers alone decide.
fn engine() -> Engine {
    let engine = Engine::new();
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

/// (collections so far, external bytes charged now).
fn state(engine: &Engine) -> (u64, usize) {
    let core = engine.interp.gc_heap.core.borrow();
    (core.collections(), core.stats().external_bytes)
}

/// One Map of 100,000 numeric entries live at a time: about 6 MB of tables each, and almost no
/// cells. The tables are charged as they grow, so collections run during the loop and reclaim
/// the replaced Maps; the bytes still charged afterwards are a few Maps' worth, not 30.
#[test]
fn replaced_maps_are_collected_as_their_tables_grow() {
    let mut engine = engine();
    let (collections, _) = state(&engine);
    let result = eval(
        &mut engine,
        "var m, n = 0;
         for (var i = 0; i < 30; i++) {
           m = new Map();
           for (var j = 0; j < 100000; j++) m.set(j, j);
           n += m.size;
         }
         n",
    );
    assert_eq!(result, "3000000");
    let (after, external) = state(&engine);
    assert!(
        after - collections >= 10,
        "only {} collections for 30 replaced 6 MB Maps",
        after - collections
    );
    assert!(
        external < 40 << 20,
        "{external} external bytes still charged after 30 replaced Maps"
    );
}

/// W6: with marking cycles, the replaced Maps' tables are charged while a cycle marks, and the
/// charges bring its slices, as the tables' allocation steps V8's marking (they are V8 heap
/// objects there, src/heap/incremental-marking.cc:54-55, 91-96): each cycle marks in a few
/// slices and its final pause frees the Maps it found dead, well before V8's margin over the
/// byte budget (at least 32 MB) would force it. Before the charges counted, no slice ran (the
/// loop allocates a few hundred bytes of cells per Map), every cycle ended at its margin, and
/// up to the budget and the margin's bytes stayed charged (48.5 MB at the loop's end with V8's
/// sizing, 54 MB at a minor collection with fixed budgets). Checked after each Map, with fixed
/// budgets and with V8's sizing (`gc_pacing`).
#[test]
fn replaced_maps_are_collected_as_their_tables_grow_while_cycles_mark() {
    for pacing in [false, true] {
        let mut engine = engine();
        // `TRUST_JS_GC_PACING=1` gives every engine V8's sizing.
        let pacing = {
            let mut core = engine.interp.gc_heap.core.borrow_mut();
            core.set_incremental(crate::gc_core::IncrementalMode::On);
            if pacing {
                core.enable_pacing(true);
            }
            core.memory_reducer().is_some()
        };
        let (collections, _) = state(&engine);
        eval(&mut engine, "var m, n = 0; 0");
        let mut most = 0;
        for i in 0..30 {
            eval(
                &mut engine,
                "m = new Map(); for (var j = 0; j < 100000; j++) m.set(j, j); n += m.size; 0",
            );
            let (_, external) = state(&engine);
            most = most.max(external);
            assert!(
                external < 40 << 20,
                "pacing {pacing}: {external} external bytes still charged after {} replaced Maps",
                i + 1
            );
        }
        assert_eq!(eval(&mut engine, "n"), "3000000");
        let (after, _) = state(&engine);
        assert!(
            after - collections >= 10,
            "pacing {pacing}: only {} collections for 30 replaced 6 MB Maps",
            after - collections
        );
        let slices = engine
            .interp
            .gc_heap
            .core
            .borrow()
            .stats()
            .last_cycle_slices;
        assert!(
            slices > 1,
            "pacing {pacing}: the last cycle ran {slices} slices (its start alone: no charge \
             brought a slice); most charged after a Map: {most}"
        );
    }
}

/// The ledger releases what its containers charged: a Map, a bound function with many
/// arguments and an error's stack trace charge their payloads, and once their owners are
/// collected the charge returns to where it was.
#[test]
fn collected_owners_release_their_charges() {
    let mut engine = engine();
    engine.interp.gc_collect();
    let (_, before) = state(&engine);
    eval(
        &mut engine,
        "var owners = [new Map([[1, 2], [3, 4]]), Math.max.bind(null, ...new Array(5000).fill(1)),
                       new Error('charged')];
         for (var j = 0; j < 10000; j++) owners[0].set(j, j);
         owners[2].stack; 0",
    );
    let (_, charged) = state(&engine);
    assert!(
        charged > before + 10000 * 24,
        "the Map's entries, the bound arguments and the trace charge their bytes: {before} -> {charged}"
    );
    eval(&mut engine, "owners = null; 0");
    engine.interp.gc_collect();
    let (_, after) = state(&engine);
    assert!(
        after <= before + 4096,
        "collected owners release their charges: {before} -> {charged} -> {after}"
    );
}

/// The engine's own payloads live in the heap (`gc_storage`): element storage counts toward the
/// nursery as cells do, so replacing one large Array, however few cells it takes, collects.
/// Each pattern allocated gigabytes before any collection when element storage lived in the
/// Rust heap uncounted: `new Array(n).fill(v)`, a chain of `map`, and arrays grown by `push`.
///
/// Each pattern makes at least 32 MB of elements, in four evaluations, with at most one array
/// of 800 KB live: a collection comes at least once per nursery's budget of them, however the
/// nursery is sized (V8's sizing, `gc_pacing`, grows it while its objects survive: each minor
/// collection promotes the live array), and the replaced arrays do not pile up.
#[test]
fn replaced_arrays_are_collected_as_their_elements_grow() {
    // The elements each pattern makes at least, and the most one array holds.
    const ELEMENT_BYTES: usize = 40 * 100_000 * 8;
    const ARRAY_BYTES: usize = 100_000 * 8;
    for (setup, step, result, expected) in [
        (
            "var keep, n = 0; 0",
            "for (var i = 0; i < 10; i++) { keep = new Array(100000).fill(i); n += keep.length; } 0",
            "n",
            "4000000",
        ),
        (
            "var a = Array.from({length: 100000}, (_, i) => i); 0",
            "for (var i = 0; i < 10; i++) a = a.map(x => x + 1); 0",
            "a[0]",
            "40",
        ),
        (
            "var s = 0; 0",
            "for (var i = 0; i < 100; i++) {
               var a = [];
               for (var j = 0; j < 10000; j++) a.push(j * 0.5);
               s += a.length;
             }
             0",
            "s",
            "4000000",
        ),
    ] {
        let mut engine = engine();
        // The collections the element blocks' charges bring, each in one pause: with
        // TRUST_JS_GC_INCREMENTAL a marking cycle that starts at the soft limit takes the
        // charges toward its margin instead, and none of its slices counts (W6 step 4).
        engine
            .interp
            .gc_heap
            .core
            .borrow_mut()
            .set_incremental(crate::gc_core::IncrementalMode::Off);
        // The budget the elements spend: the nursery's, or without one the byte budget.
        let budget = |engine: &Engine| {
            let core = engine.interp.gc_heap.core.borrow();
            core.nursery_budget()
                .unwrap_or_else(|| core.stats().trigger_bytes)
        };
        let (collections, _) = state(&engine);
        assert_eq!(eval(&mut engine, setup), "0", "{setup}");
        // The largest budget the evaluations ran with, sampled between them.
        let mut largest = budget(&engine);
        for _ in 0..4 {
            assert_eq!(eval(&mut engine, step), "0", "{step}");
            largest = largest.max(budget(&engine));
        }
        assert_eq!(eval(&mut engine, result), expected, "{step}");
        let (after, _) = state(&engine);
        // At most a budget and an array are allocated between two collections, before the
        // first and after the last: 10 at least with a fixed nursery of 2 MiB.
        let least = (ELEMENT_BYTES / (largest + ARRAY_BYTES)).saturating_sub(1) as u64;
        assert!(
            after - collections >= least,
            "only {} collections while making 32 MB of elements with budgets of {largest} bytes \
             at most (at least {least}):\n{step}",
            after - collections
        );
        let held = engine.interp.gc_heap.core.borrow().stats().live_bytes;
        assert!(
            held < 48 << 20,
            "{held} bytes of cells held after replacing arrays:\n{step}"
        );
    }
}

/// The registry that holds closures' functions (`value::ClosureFunction`) lets a function go at
/// the collection that finds its closures dead, as the closures' finalizers used to: the next
/// minor collection when they all died young, a major one once one had survived. A function
/// made by `new Function` owns its parse tree and code, which the collector does not count, so
/// holding dead ones until a major collection would let a script that compiles a function per
/// call grow without bound. Counts over many functions keep the test independent of stale
/// words in native frames, which may keep any one closure.
#[test]
fn closure_functions_go_with_their_closures() {
    let mut engine = engine();
    let count = |engine: &Engine| crate::value::closure_function_count(&engine.interp.gc_heap);
    engine.interp.gc_collect();
    let base = count(&engine);
    // 2,000 functions, each closure dead at once.
    eval(
        &mut engine,
        "var n = 0; for (var i = 0; i < 2000; i++) n += new Function('x', 'return x + ' + i)(1); n",
    );
    engine.interp.gc_collect_minor();
    let after_minor = count(&engine);
    assert!(
        after_minor < base + 100,
        "a minor collection kept {} of 2,000 functions whose closures died young",
        after_minor - base
    );
    // 200 functions whose closures survive a collection, then die: minor collections cannot
    // tell, a major one can.
    eval(
        &mut engine,
        "var kept = []; for (var i = 0; i < 200; i++) kept.push(new Function('return ' + i)); 0",
    );
    engine.interp.gc_collect_minor();
    let with_old = count(&engine);
    assert!(
        with_old >= after_minor + 200,
        "{with_old} < {after_minor} + 200"
    );
    eval(&mut engine, "kept = null; 0");
    engine.interp.gc_collect_minor();
    assert!(
        count(&engine) >= after_minor + 200,
        "a minor collection judged old closures"
    );
    engine.interp.gc_collect();
    assert!(
        count(&engine) < after_minor + 20,
        "a major collection kept {} functions of dead old closures",
        count(&engine).saturating_sub(after_minor)
    );
}

/// A test's own collection (`gc_collect_precise_for_test`) judges the closure registry as
/// every marking does: the closures it marks are old, and the minor collection after it must
/// not take their functions for young ones. GC stress caught two agent tests calling a freed
/// function after a precise major collection had bypassed the registry.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn closure_functions_outlive_the_minor_collection_after_a_precise_one() {
    let mut engine = engine();
    let count = |engine: &Engine| crate::value::closure_function_count(&engine.interp.gc_heap);
    eval(
        &mut engine,
        "var kept = []; for (var i = 0; i < 200; i++) kept.push(new Function('return ' + i)); 0",
    );
    let held = count(&engine);
    engine
        .interp
        .gc_collect_precise_for_test(crate::gc_core::CollectionKind::Major);
    engine.interp.gc_collect_minor();
    assert!(
        count(&engine) >= held,
        "a minor collection let go of {} functions of old closures",
        held - count(&engine)
    );
    assert_eq!(
        eval(
            &mut engine,
            "var sum = 0; for (var f of kept) sum += f(); sum"
        ),
        "19900"
    );
}
