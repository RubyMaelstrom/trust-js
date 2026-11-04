//! W6: incremental marking's cycles under the engine: what a cycle's slices and final pause keep,
//! with the program running between the slices in every tier.

use crate::bytecode::Tier;
use crate::gc_core::IncrementalMode;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// An engine whose heap marks its major collections in cycles, each slice a few cells at every
/// safepoint (`TRUST_JS_GC_INCREMENTAL=force`), in `tier`. The tests start and end the cycles they
/// watch, so the heap runs no stress collections, whatever `TRUST_JS_GC_STRESS` says: the stress
/// interval's would start cycles of its own.
fn cycling_engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    let mut core = engine.interp.gc_heap.core.borrow_mut();
    core.set_stress_every(0);
    core.set_incremental(IncrementalMode::Force);
    drop(core);
    engine
}

fn cycle_running(engine: &Engine) -> bool {
    engine.interp.gc_heap.core.borrow().cycle_running()
}

/// A graph built before a cycle starts and rewired while its slices mark: new objects stored
/// into old ones (through property, element, closure-variable and Map writes), old objects moved
/// from a holder the marker has not reached to one it has, and objects dropped. The final pause
/// keeps exactly what the program can still reach, and verification (the unit tests verify every
/// collection) checks each slice's barrier invariant and the final marking's graph.
#[test]
fn a_cycle_keeps_what_the_program_reaches_through_its_slices_in_every_tier() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = cycling_engine(tier);
        eval(
            &mut engine,
            r#"
            var roots = [];
            for (let i = 0; i < 300; i++) roots.push({ i, next: null, kids: [] });
            var map = new Map();
            var captured = null;
            function capture(value) { captured = value; }
            true
            "#,
        );
        assert!(engine.interp.gc_start_cycle(), "{tier:?}: the cycle starts");
        assert!(cycle_running(&engine), "{tier:?}");
        let result = eval(
            &mut engine,
            r#"
            var moved = [];
            for (let round = 0; round < 40; round++) {
                for (let i = 0; i < roots.length; i++) {
                    const node = roots[i];
                    // A new object into an old one, by property and by element.
                    node.next = { round, i, payload: 'p' + round + ':' + i };
                    node.kids.push({ round });
                    if (node.kids.length > 4) node.kids.shift();
                    map.set(i, { round });
                    capture({ round, i });
                }
                // An old object moved from the far end (traced last) to the near end.
                const last = roots.pop();
                moved.push(last);
                roots.unshift({ i: -round, next: last, kids: [] });
            }
            let ok = true;
            for (let i = 0; i < roots.length; i++) {
                const node = roots[i];
                if (node.next === null || node.next.round !== 39) {
                    if (!(node.next && typeof node.next.i === 'number')) ok = false;
                }
                for (const kid of node.kids) if (typeof kid.round !== 'number') ok = false;
            }
            for (const m of moved) if (typeof m.i !== 'number') ok = false;
            for (const [k, v] of map) if (v.round !== 39) ok = false;
            ok && captured.round === 39
            "#,
        );
        assert_eq!(result, "true", "{tier:?}");
        // An explicit collection finishes the cycle (its final pause) and keeps the graph.
        engine.interp.gc_collect();
        assert!(!cycle_running(&engine), "{tier:?}: the final pause ends the cycle");
        let result = eval(
            &mut engine,
            r#"
            let total = 0;
            for (const node of roots) total += node.kids.length + (node.next ? 1 : 0);
            for (const m of moved) total += m.kids.length;
            total > 0 && map.size === 300 && captured.i === 299
            "#,
        );
        assert_eq!(result, "true", "{tier:?}");
    }
}

/// An explicit minor collection during a cycle finishes it (V8 collects fully for a young
/// generation request while it marks): the cycle's final pause runs, as a major collection.
#[test]
fn an_explicit_minor_collection_finishes_a_running_cycle() {
    let mut engine = cycling_engine(Tier::Interp);
    eval(&mut engine, "var keep = { a: [1, 2, 3] }; true");
    assert!(engine.interp.gc_start_cycle());
    let majors = {
        let core = engine.interp.gc_heap.core.borrow();
        core.collections() - core.minor_collections()
    };
    engine.interp.gc_collect_minor();
    assert!(!cycle_running(&engine));
    let core = engine.interp.gc_heap.core.borrow();
    assert_eq!(core.collections() - core.minor_collections(), majors + 1);
    drop(core);
    assert_eq!(eval(&mut engine, "keep.a.length"), "3");
}

/// A cycle's start and every slice keep the marker's worklist; the final pause follows once it
/// empties, at the next safepoint, without an explicit request.
#[test]
fn a_cycle_finishes_once_its_worklist_empties() {
    let mut engine = cycling_engine(Tier::Bytecode);
    eval(
        &mut engine,
        "var keep = []; for (let i = 0; i < 500; i++) keep.push({ i, a: [i] }); true",
    );
    assert!(engine.interp.gc_start_cycle());
    let collections = engine.interp.gc_heap.core.borrow().collections();
    eval(
        &mut engine,
        "let n = 0; for (let i = 0; i < 100000 && n < 1e9; i++) n += keep[i % 500].a[0]; n > 0",
    );
    assert!(!cycle_running(&engine), "the cycle's final pause ran on its own");
    assert_eq!(
        engine.interp.gc_heap.core.borrow().collections(),
        collections + 1
    );
    assert_eq!(eval(&mut engine, "keep[499].a[0]"), "499");
}

/// Weak collections across a cycle: entries whose keys die while the cycle marks go at its final
/// pause, entries made meanwhile (in collections the marker traced already, whose cards the
/// writes set) keep their values while their keys live, and a WeakRef to an object that died
/// reads undefined after the job that made it.
#[test]
fn weak_collections_keep_their_live_entries_across_a_cycle_in_every_tier() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = cycling_engine(tier);
        eval(
            &mut engine,
            r#"
            var wm = new WeakMap(), ws = new WeakSet(), keys = [];
            for (let i = 0; i < 200; i++) {
                const k = { i };
                keys.push(k);
                wm.set(k, { v: i });
                ws.add(k);
            }
            var ref = new WeakRef({ dead: true });
            true
            "#,
        );
        assert!(engine.interp.gc_start_cycle(), "{tier:?}");
        eval(
            &mut engine,
            r#"
            for (let i = 0; i < 100; i++) keys[i] = null;
            for (let i = 0; i < 100; i++) {
                const k = { j: i };
                keys.push(k);
                wm.set(k, { w: i, back: k });
                ws.add(k);
            }
            true
            "#,
        );
        engine.interp.gc_collect();
        assert!(!cycle_running(&engine), "{tier:?}");
        let result = eval(
            &mut engine,
            r#"
            let ok = keys.length === 300;
            for (let i = 100; i < 200; i++) ok = ok && wm.get(keys[i]).v === i && ws.has(keys[i]);
            for (let i = 200; i < 300; i++) {
                const entry = wm.get(keys[i]);
                ok = ok && entry.w === i - 200 && entry.back === keys[i] && ws.has(keys[i]);
            }
            ok && ref.deref() === undefined
            "#,
        );
        assert_eq!(result, "true", "{tier:?}");
    }
}

/// A weak handle to the object the global `name` holds, made in a frame of its own.
fn weak_global(engine: &mut Engine, name: &str) -> Box<crate::value::WeakGc> {
    crate::gc_test::in_frame(|| {
        let env = engine.interp.global_env;
        let value = engine
            .interp
            .get_var(name, &env)
            .unwrap_or_else(|_| panic!("fixture global {name}"));
        let object = value.as_obj().expect("an object");
        Box::new(crate::value::Gc::downgrade(&object))
    })
}

/// An explicit collection is precise while a cycle marks (V8's PreciseCollectAllGarbage): an
/// object the cycle marked before the program dropped it is freed, by the collection that
/// follows the cycle's final pause.
#[test]
fn an_explicit_collection_during_a_cycle_frees_what_died_after_the_cycle_marked_it() {
    let mut engine = cycling_engine(Tier::Bytecode);
    eval(
        &mut engine,
        r#"
        var bulk = [];
        for (let i = 0; i < 30000; i++) bulk.push({ i });
        var dropped = { kids: [1, 2, 3] };
        true
        "#,
    );
    let weak = weak_global(&mut engine, "dropped");
    // Kept complemented: a plain address in this frame would keep the object alive.
    let address = crate::gc_test::HiddenAddress::from_fn(|| {
        weak.upgrade()
            .map(|object| crate::value::Gc::as_ptr(&object) as usize)
            .expect("alive")
    });
    assert!(engine.interp.gc_start_cycle());
    #[inline(never)]
    fn marked(engine: &Engine, address: crate::gc_test::HiddenAddress) -> bool {
        engine.interp.gc_heap.core.borrow().cell_age(address.get() as u64)
            == Some(crate::gc_core::CellAge::Old)
    }
    let mut slices = 0;
    while !marked(&engine, address) {
        assert!(engine.interp.gc_slice(), "the cycle runs until the object is marked");
        slices += 1;
        assert!(slices < 1_000_000);
    }
    assert!(cycle_running(&engine), "the bulk keeps the worklist busy");
    eval(&mut engine, "dropped = null; 0");
    assert!(cycle_running(&engine));
    engine.interp.gc_collect();
    assert!(!cycle_running(&engine));
    assert!(!crate::gc_test::alive(&*weak), "the precise collection freed it");
    assert_eq!(eval(&mut engine, "bulk.length"), "30000");
}
