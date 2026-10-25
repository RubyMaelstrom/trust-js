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
#[test]
fn replaced_arrays_are_collected_as_their_elements_grow() {
    for (source, expected) in [
        (
            "var keep, n = 0;
             for (var i = 0; i < 40; i++) { keep = new Array(100000).fill(i); n += keep.length; }
             n",
            "4000000",
        ),
        (
            "var a = Array.from({length: 100000}, (_, i) => i);
             for (var i = 0; i < 40; i++) a = a.map(x => x + 1);
             a[0]",
            "40",
        ),
        (
            "var s = 0;
             for (var i = 0; i < 400; i++) {
               var a = [];
               for (var j = 0; j < 10000; j++) a.push(j * 0.5);
               s += a.length;
             }
             s",
            "4000000",
        ),
    ] {
        let mut engine = engine();
        let (collections, _) = state(&engine);
        assert_eq!(eval(&mut engine, source), expected, "{source}");
        let (after, _) = state(&engine);
        assert!(
            after - collections >= 10,
            "only {} collections while replacing about 64 MB of elements:\n{source}",
            after - collections
        );
        let held = engine.interp.gc_heap.core.borrow().stats().live_bytes;
        assert!(
            held < 48 << 20,
            "{held} bytes of cells held after replacing arrays:\n{source}"
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
