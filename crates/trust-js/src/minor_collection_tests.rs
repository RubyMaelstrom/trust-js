//! S6 step 10c: minor collections in the engine. A minor collection reclaims young garbage and
//! keeps every old object, those nothing reaches any more included: they wait for a major
//! collection. Weak references, WeakMap entries and FinalizationRegistry cells follow the same
//! rule. Each cause collects with its kind: allocation safepoints and the task boundary
//! minorly unless a major collection is due, explicit requests, ArrayBuffer pressure and the
//! live-object ceiling majorly, and GC stress minorly with every Mth collection major.
//!
//! These tests watch when collections run and what each keeps, so their engines' heaps run no
//! stress collections (`engine`), whatever `TRUST_JS_GC_STRESS` says; the stress test sets its
//! own interval. They collect from Rust, holding nothing they watch in their own frames (see
//! `gc_test`).

use crate::bytecode::Tier;
use crate::gc_core::CollectionKind;
use crate::gc_test::alive;
use crate::value::ValueView;
use crate::value::{Gc, WeakGc};
use crate::{Completion, Engine};

/// An engine at `tier` (threshold 0) whose heap has no stress interval.
fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("minor collection fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

/// A weak handle to the object the global `name` holds, made in a frame of its own (the next
/// explicit collection zeroes it).
#[inline(never)]
fn weak_global(engine: &mut Engine, name: &str) -> WeakGc {
    let env = engine.interp.global_env;
    match (engine.interp.get_var(name, &env)).map(|__value| (__value).view()) {
        Ok(ValueView::Obj(object)) => Gc::downgrade(&object),
        _ => panic!("{name}: not an object"),
    }
}

/// The engine's heap's collections so far: (all, minor).
fn collections(engine: &Engine) -> (u64, u64) {
    let core = engine.interp.gc_heap.core.borrow();
    (core.collections(), core.minor_collections())
}

/// The collections since `before`: (minor, major).
fn since(engine: &Engine, before: (u64, u64)) -> (u64, u64) {
    let (all, minor) = collections(engine);
    let minor = minor - before.1;
    (minor, all - before.0 - minor)
}

/// A minor collection reclaims young garbage and keeps the old objects, whether something still
/// reaches them or not; the next major collection reclaims the old garbage. At every tier.
#[test]
fn a_minor_collection_reclaims_young_garbage_and_keeps_old_objects() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        eval(&mut engine, "var kept = { k: 1 }, dropped = { d: 2 }; 0");
        let old_live = weak_global(&mut engine, "kept");
        let old_dead = weak_global(&mut engine, "dropped");
        engine.interp.gc_collect();
        eval(
            &mut engine,
            "dropped = null; var young = { y: 3 }, garbage = { g: 4 }; 0",
        );
        let young = weak_global(&mut engine, "young");
        let garbage = weak_global(&mut engine, "garbage");
        eval(&mut engine, "garbage = null; 0");
        let before = collections(&engine);
        engine.interp.gc_collect_minor();
        assert_eq!(since(&engine, before), (1, 0), "{tier:?}");
        assert!(!alive(&garbage), "{tier:?}: young garbage");
        assert!(alive(&young) && alive(&old_live), "{tier:?}");
        assert!(
            alive(&old_dead),
            "{tier:?}: an old object that died waits for a major collection"
        );
        assert_eq!(eval(&mut engine, "kept.k + young.y"), "4", "{tier:?}");
        engine.interp.gc_collect();
        assert!(!alive(&old_dead), "{tier:?}");
        assert!(alive(&young) && alive(&old_live), "{tier:?}");
    }
}

/// WeakRef targets, WeakMap entries and FinalizationRegistry cells at minor collections: a
/// young target that dies is cleared by a minor collection (its WeakRef empties, its WeakMap
/// entry goes and its registry's cleanup runs), while an old target that died stays until a
/// major collection clears it the same way. At every tier.
#[test]
fn weak_targets_die_young_at_minor_collections_and_old_at_major_ones() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        let entries = |engine: &Engine| -> usize {
            engine
                .interp
                .weak_collection_data
                .values()
                .map(Vec::len)
                .sum()
        };
        eval(
            &mut engine,
            r#"
            var cleaned = [], registry = new FinalizationRegistry(held => cleaned.push(held));
            var map = new WeakMap(), refs = {}, holder = {};
            function watch(name) {
                var target = { name: name };
                refs[name] = new WeakRef(target);
                map.set(target, name);
                registry.register(target, name);
                return target;
            }
            holder.target = watch("old");
            0
            "#,
        );
        // The old target survives a collection, then dies: only a major collection can tell.
        engine.interp.gc_collect();
        eval(&mut engine, "holder.target = null; watch('young'); 0");
        assert_eq!(entries(&engine), 2, "{tier:?}");
        let before = collections(&engine);
        engine.interp.gc_collect_minor();
        assert_eq!(since(&engine, before), (1, 0), "{tier:?}");
        assert_eq!(
            entries(&engine),
            1,
            "{tier:?}: the young key's entry is gone"
        );
        // The cleanup job runs after this script, so the next one sees it.
        eval(&mut engine, "0");
        assert_eq!(
            eval(
                &mut engine,
                "[refs.young.deref() === undefined, refs.old.deref() !== undefined, \
                 cleaned.join()].join(':')"
            ),
            "true:true:young",
            "{tier:?}"
        );
        engine.interp.gc_collect();
        assert_eq!(entries(&engine), 0, "{tier:?}");
        eval(&mut engine, "0");
        assert_eq!(
            eval(
                &mut engine,
                "[refs.old.deref() === undefined, cleaned.join()].join(':')"
            ),
            "true:young,old",
            "{tier:?}"
        );
    }
}

/// The task boundary collects minorly, on its growth and churn conditions; when a major
/// collection is due (a minor collection left the heap past its byte budget), majorly.
#[test]
fn the_task_boundary_collects_minorly_unless_a_major_collection_is_due() {
    let mut engine = engine(Tier::Bytecode);
    engine.interp.gc_collect();
    // A script that allocates past the churn floor (10,000 objects) collects at its end.
    let before = collections(&engine);
    eval(
        &mut engine,
        "(function () { for (var i = 0; i < 12000; i++) ({ i: i }); })(); 0",
    );
    let (minor, major) = since(&engine, before);
    assert!(minor >= 1 && major == 0, "{minor} minor, {major} major");
    let before = collections(&engine);
    engine.interp.gc_heap.agent.task_pending.set(true);
    engine.interp.gc_task_boundary();
    assert_eq!(since(&engine, before), (1, 0));
    // A byte budget the heap is past: the next minor collection makes a major one due.
    engine
        .interp
        .gc_heap
        .core
        .borrow_mut()
        .set_heap_growth(1, 100);
    engine.interp.gc_collect_minor();
    assert_eq!(
        engine.interp.gc_heap.core.borrow().due_kind(),
        CollectionKind::Major
    );
    let before = collections(&engine);
    engine.interp.gc_heap.agent.task_pending.set(true);
    engine.interp.gc_task_boundary();
    assert_eq!(since(&engine, before), (0, 1));
}

/// Explicit requests collect majorly: the engine's, the host's, `$262.gc`, the idle collection
/// (with `embed`), the performance metrics' exit collection, and ArrayBuffer pressure.
#[test]
fn explicit_collections_are_major() {
    let mut engine = engine(Tier::Interp);
    let requests: [(&str, fn(&mut Engine)); 5] = [
        ("Interp::gc_collect", |engine| engine.interp.gc_collect()),
        ("collect_garbage_for_host", |engine| {
            engine.interp.collect_garbage_for_host();
        }),
        ("$262.gc", |engine| {
            eval(engine, "$262.gc(); 0");
        }),
        ("unstable_collect_for_performance_metrics", |engine| {
            engine.unstable_collect_for_performance_metrics()
        }),
        ("ArrayBuffer pressure", |engine| {
            engine.interp.note_buffer_allocation(usize::MAX / 2)
        }),
    ];
    for (request, collect) in requests {
        let before = collections(&engine);
        collect(&mut engine);
        assert_eq!(since(&engine, before), (0, 1), "{request}");
    }
    #[cfg(feature = "embed")]
    {
        let before = collections(&engine);
        engine.collect_garbage_at_idle();
        assert_eq!(since(&engine, before), (0, 1), "collect_garbage_at_idle");
    }
}

/// The live-object ceiling collects majorly: old objects that died count as live until a major
/// collection, and the ceiling's collection frees them instead of throwing. Retention past the
/// limit still throws its RangeError, after a major collection. No budget makes another
/// collection due here, and the task boundary is deferred. At every tier.
#[test]
fn the_live_object_limit_collects_majorly_before_its_range_error() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        engine.set_live_object_limit(150_000);
        engine.interp.gc_heap.agent.task_deferred.set(true);
        {
            let mut core = engine.interp.gc_heap.core.borrow_mut();
            core.set_nursery_bytes(Some(1 << 40));
            core.set_heap_growth(1 << 40, 100);
        }
        eval(
            &mut engine,
            "var old = []; for (var i = 0; i < 120000; i++) old.push({ i: i }); 0",
        );
        engine.interp.gc_collect();
        let before = collections(&engine);
        let source = "old = null; var kept = [], outcome = 'ok';
            function fill(n) { for (var i = 0; i < n; i++) kept.push({ i: i }); }
            try { fill(60000); } catch (e) { outcome = e.name; }
            outcome + ':' + kept.length";
        assert_eq!(eval(&mut engine, source), "ok:60000", "{tier:?}");
        assert_eq!(
            since(&engine, before),
            (0, 1),
            "{tier:?}: the ceiling's collection"
        );
        let before = collections(&engine);
        let source = "kept = []; outcome = 'ok';
            try { fill(250000); } catch (e) { outcome = e.name + ':' + e.message; }
            kept = null; outcome";
        assert_eq!(
            eval(&mut engine, source),
            "RangeError:allocation limit exceeded",
            "{tier:?}"
        );
        let (minor, major) = since(&engine, before);
        assert!(
            minor == 0 && major >= 1,
            "{tier:?}: {minor} minor, {major} major"
        );
    }
}

/// Allocation safepoints, the JIT's polls among them, collect minorly when the nursery budget is
/// spent, and majorly once a minor collection leaves the heap past its byte budget. At every
/// tier.
#[test]
fn allocation_safepoints_collect_minorly_until_a_major_collection_is_due() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        engine.interp.gc_heap.agent.task_deferred.set(true);
        {
            let mut core = engine.interp.gc_heap.core.borrow_mut();
            core.set_nursery_bytes(Some(256 << 10));
            core.set_heap_growth(1 << 40, 100);
        }
        eval(
            &mut engine,
            "function churn(n) { var o; for (var i = 0; i < n; i++) o = { i: i }; return o.i; } 0",
        );
        let before = collections(&engine);
        // 20,000 objects of 192 bytes: about 15 nursery budgets.
        assert_eq!(eval(&mut engine, "churn(20000)"), "19999", "{tier:?}");
        let (minor, major) = since(&engine, before);
        assert!(
            minor >= 8 && major == 0,
            "{tier:?}: {minor} minor, {major} major"
        );
        engine
            .interp
            .gc_heap
            .core
            .borrow_mut()
            .set_heap_growth(1, 100);
        let before = collections(&engine);
        assert_eq!(eval(&mut engine, "churn(20000)"), "19999", "{tier:?}");
        let (minor, major) = since(&engine, before);
        assert!(
            minor >= 1 && major >= 1,
            "{tier:?}: {minor} minor, {major} major"
        );
    }
}

/// GC stress collects at every safepoint that follows an allocation: minorly, but every Mth
/// collection majorly; what the script keeps survives either kind.
#[test]
fn gc_stress_collects_minorly_with_every_mth_collection_major() {
    let mut engine = engine(Tier::Bytecode);
    engine.interp.gc_heap.agent.task_deferred.set(true);
    {
        let mut core = engine.interp.gc_heap.core.borrow_mut();
        core.set_stress_every(1);
        core.set_stress_major_every(4);
    }
    let before = collections(&engine);
    // A call is a safepoint, and each allocates.
    let source = "function make(i) { return { i: i }; }
        var kept = []; for (var i = 0; i < 64; i++) kept.push(make(i)); kept.length";
    assert_eq!(eval(&mut engine, source), "64");
    let (minor, major) = since(&engine, before);
    // The heap counts its stress collections from its first (under TRUST_JS_GC_STRESS, as the
    // engine was made), so the period may start part-way.
    assert!(minor + major >= 64, "{minor} minor, {major} major");
    assert!(
        major.abs_diff((minor + major) / 4) <= 1,
        "{minor} minor, {major} major"
    );
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
    assert_eq!(
        eval(&mut engine, "kept.reduce((sum, o) => sum + o.i, 0)"),
        "2016"
    );
}
