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
        assert!(
            !cycle_running(&engine),
            "{tier:?}: the final pause ends the cycle"
        );
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
    assert!(
        !cycle_running(&engine),
        "the cycle's final pause ran on its own"
    );
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
        engine
            .interp
            .gc_heap
            .core
            .borrow()
            .cell_age(address.get() as u64)
            == Some(crate::gc_core::CellAge::Old)
    }
    let mut slices = 0;
    while !marked(&engine, address) {
        assert!(
            engine.interp.gc_slice(),
            "the cycle runs until the object is marked"
        );
        slices += 1;
        assert!(slices < 1_000_000);
    }
    assert!(cycle_running(&engine), "the bulk keeps the worklist busy");
    eval(&mut engine, "dropped = null; 0");
    assert!(cycle_running(&engine));
    engine.interp.gc_collect();
    assert!(!cycle_running(&engine));
    assert!(
        !crate::gc_test::alive(&*weak),
        "the precise collection freed it"
    );
    assert_eq!(eval(&mut engine, "bulk.length"), "30000");
}

/// A test's precise collection (`Interp::gc_collect_precise_for_test`) while a cycle marks
/// finishes the cycle first, then marks from the roots alone: an object the cycle marked before
/// the program dropped it is freed, as an explicit collection frees it (above). With incremental
/// marking on by default, a stress major's cycle can run as such a collection comes
/// (`reentry_rooting_tests`' `retired`, under `TRUST_JS_GC_STRESS=1`).
#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_precise_test_collection_during_a_cycle_frees_what_died_after_the_cycle_marked_it() {
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
    let address = hidden_address(&weak);
    assert!(engine.interp.gc_start_cycle());
    slice_until_marked(&mut engine, address, "the dropped object");
    assert!(cycle_running(&engine), "the bulk keeps the worklist busy");
    eval(&mut engine, "dropped = null; 0");
    assert!(cycle_running(&engine));
    engine
        .interp
        .gc_collect_precise_for_test(crate::gc_core::CollectionKind::Major);
    assert!(!cycle_running(&engine));
    assert!(
        !crate::gc_test::alive(&*weak),
        "the precise collection freed it"
    );
    assert_eq!(eval(&mut engine, "bulk.length"), "30000");
}

/// The address of the object a weak handle names, kept complemented (a plain address in a test's
/// frame would keep the object alive).
fn hidden_address(weak: &crate::value::WeakGc) -> crate::gc_test::HiddenAddress {
    crate::gc_test::HiddenAddress::from_fn(|| {
        weak.upgrade()
            .map(|object| crate::value::Gc::as_ptr(&object) as usize)
            .expect("alive")
    })
}

/// Whether the running cycle has marked the cell at `address`.
#[inline(never)]
fn marked(engine: &Engine, address: crate::gc_test::HiddenAddress) -> bool {
    engine
        .interp
        .gc_heap
        .core
        .borrow()
        .cell_age(address.get() as u64)
        == Some(crate::gc_core::CellAge::Old)
}

/// Run slices of the running cycle until it has marked the cell at `address`.
fn slice_until_marked(engine: &mut Engine, address: crate::gc_test::HiddenAddress, what: &str) {
    let mut slices = 0;
    while !marked(engine, address) {
        assert!(
            engine.interp.gc_slice(),
            "{what}: the cycle runs until it is marked"
        );
        slices += 1;
        assert!(slices < 1_000_000, "{what}: never marked");
    }
}

/// W6 step 2b: a WeakMap's entries are discovered in the slices that mark it (as V8 discovers an
/// ephemeron table's entries as its marker visits the table): a value only the map holds is
/// marked before the final pause. The entries the program writes after that, under new keys and
/// old ones, are the final pause's, from the cycle's records, and survive it. The final pause
/// alone ends the cycle (an explicit minor request), so no collection after it hides a miss.
#[test]
fn a_weak_maps_entries_are_discovered_in_the_slices_and_merged_at_the_final_pause_in_every_tier() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = cycling_engine(tier);
        eval(
            &mut engine,
            r#"
            var keys = [], wm = new WeakMap();
            for (let i = 0; i < 200; i++) {
                const k = { i };
                keys.push(k);
                wm.set(k, { v: i });
            }
            // A long chain that only the first entry's value holds keeps the marker busy once
            // the entry is discovered.
            wm.set(keys[0], { v: 0, chain: (function () {
                let head = null;
                for (let i = 0; i < 20000; i++) head = { next: head, i };
                return head;
            })() });
            var probe = wm.get(keys[0]);
            true
            "#,
        );
        let weak = weak_global(&mut engine, "probe");
        let probe = hidden_address(&weak);
        eval(&mut engine, "probe = null; 0");
        assert!(engine.interp.gc_start_cycle(), "{tier:?}");
        slice_until_marked(&mut engine, probe, "the first entry's value");
        assert!(
            cycle_running(&engine),
            "{tier:?}: the chain keeps the cycle marking"
        );
        eval(
            &mut engine,
            r#"
            for (let i = 0; i < 100; i++) {
                const k = { j: i };
                keys.push(k);
                wm.set(k, { w: i, back: k });
            }
            for (let i = 1; i < 51; i++) wm.set(keys[i], { replaced: i });
            true
            "#,
        );
        assert!(
            cycle_running(&engine),
            "{tier:?}: the program ran within the cycle"
        );
        engine.interp.gc_collect_minor();
        assert!(!cycle_running(&engine), "{tier:?}");
        let result = eval(
            &mut engine,
            r#"
            let ok = keys.length === 300;
            for (let i = 1; i < 51; i++) ok = ok && wm.get(keys[i]).replaced === i;
            for (let i = 51; i < 200; i++) ok = ok && wm.get(keys[i]).v === i;
            for (let i = 200; i < 300; i++) {
                const entry = wm.get(keys[i]);
                ok = ok && entry.w === i - 200 && entry.back === keys[i];
            }
            let n = 0;
            for (let node = wm.get(keys[0]).chain; node; node = node.next) n++;
            ok && n === 20000
            "#,
        );
        assert_eq!(result, "true", "{tier:?}");
    }
}

/// W6 step 2b: a side-slot table's edges fire in the slices as their owners are marked, and a
/// write during the cycle into an owner the slices traced is the final pause's (the cycle's
/// records): a reaction added to a pending promise after the slices traced its reactions
/// survives the final pause alone, and runs once the promise settles.
#[test]
fn a_promise_reaction_added_during_a_cycle_survives_its_final_pause_in_every_tier() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = cycling_engine(tier);
        eval(
            &mut engine,
            r#"
            var settle, counted = 0, result = 0;
            var p = new Promise(function (resolve) { settle = resolve; });
            // The first reaction alone holds a long chain, which keeps the marker busy once the
            // promise's reactions are traced.
            var first = (function () {
                let head = null;
                for (let i = 0; i < 20000; i++) head = { next: head, i };
                return function () {
                    for (let node = head; node; node = node.next) counted++;
                };
            })();
            p.then(first);
            true
            "#,
        );
        let weak = weak_global(&mut engine, "first");
        let first = hidden_address(&weak);
        eval(&mut engine, "first = null; 0");
        assert!(engine.interp.gc_start_cycle(), "{tier:?}");
        slice_until_marked(&mut engine, first, "the first reaction");
        assert!(
            cycle_running(&engine),
            "{tier:?}: the chain keeps the cycle marking"
        );
        eval(
            &mut engine,
            r#"
            p.then((function () {
                const captured = { v: 41 };
                return function () { result = captured.v + 1; };
            })());
            true
            "#,
        );
        assert!(
            cycle_running(&engine),
            "{tier:?}: the program ran within the cycle"
        );
        engine.interp.gc_collect_minor();
        assert!(!cycle_running(&engine), "{tier:?}");
        assert_eq!(
            eval(&mut engine, "settle(1); 0"),
            "0",
            "{tier:?}: the reactions run"
        );
        assert_eq!(
            eval(&mut engine, "counted + ':' + result"),
            "20000:42",
            "{tier:?}"
        );
    }
}

/// W6 step 2b: a Realm group activates in the slice that marks a member, and the final pause
/// traces every active group again: a template object a Realm's [[TemplateMap]] gains during the
/// cycle (GetTemplateObject's per-Realm cache) survives the final pause alone, and its site keeps
/// returning it.
#[test]
fn a_realm_groups_template_map_gains_an_entry_during_a_cycle_in_every_tier() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = cycling_engine(tier);
        eval(
            &mut engine,
            r#"
            var other = $262.createRealm().global;
            var first = other.eval("(function () { return (function (s) { return s; })`first`; })");
            var second = other.eval("(function () { return (function (s) { return s; })`second`; })");
            var template = first();
            // A long chain that only the first template's entry holds: marking the template
            // (the Realm group's activation) brings it, and it keeps the marker busy.
            var byTemplate = new WeakMap();
            byTemplate.set(template, (function () {
                let head = null;
                for (let i = 0; i < 20000; i++) head = { next: head, i };
                return head;
            })());
            true
            "#,
        );
        let weak = weak_global(&mut engine, "template");
        let template = hidden_address(&weak);
        eval(&mut engine, "template = null; 0");
        assert!(engine.interp.gc_start_cycle(), "{tier:?}");
        slice_until_marked(&mut engine, template, "the first template");
        assert!(
            cycle_running(&engine),
            "{tier:?}: the chain keeps the cycle marking"
        );
        eval(&mut engine, "second(); 0");
        assert!(
            cycle_running(&engine),
            "{tier:?}: the program ran within the cycle"
        );
        engine.interp.gc_collect_minor();
        assert!(!cycle_running(&engine), "{tier:?}");
        let result = eval(
            &mut engine,
            r#"
            let n = 0;
            for (let node = byTemplate.get(first()); node; node = node.next) n++;
            second() === second() && second()[0] === "second" && n === 20000
            "#,
        );
        assert_eq!(result, "true", "{tier:?}");
    }
}

/// The running cycle's slices so far.
fn cycle_slices(engine: &Engine) -> u32 {
    let core = engine.interp.gc_heap.core.borrow();
    core.cycle_progress().expect("a running cycle").slices
}

/// Where the next chunk of the large element block of the owner at `address` starts, while the
/// running cycle traces it in chunks.
#[inline(never)]
fn chunk_progress(engine: &Engine, address: crate::gc_test::HiddenAddress) -> Option<usize> {
    let core = engine.interp.gc_heap.core.borrow();
    core.chunk_progress(address.get())
}

/// Run slices of the running cycle until the large element block of the owner at `address` is
/// between its chunks, past its first 128 words.
fn slice_until_between_chunks(
    engine: &mut Engine,
    address: crate::gc_test::HiddenAddress,
    what: &str,
) {
    let mut slices = 0;
    while !chunk_progress(engine, address).is_some_and(|next| next >= 128) {
        assert!(
            engine.interp.gc_slice(),
            "{what}: the cycle runs until the block is between its chunks"
        );
        slices += 1;
        assert!(slices < 1_000_000, "{what}: never between chunks");
    }
}

/// A weak handle to the object that element `index` of the array the global `name` holds names,
/// made in a frame of its own.
fn weak_element(engine: &mut Engine, name: &str, index: u32) -> Box<crate::value::WeakGc> {
    crate::gc_test::in_frame(|| {
        let env = engine.interp.global_env;
        let array = engine
            .interp
            .get_var(name, &env)
            .unwrap_or_else(|_| panic!("fixture global {name}"));
        let array = array.as_obj().expect("an array");
        let element = array
            .borrow()
            .props
            .data_element(index)
            .expect("an element");
        Box::new(crate::value::Gc::downgrade(
            &element.as_obj().expect("an object"),
        ))
    })
}

/// An engine whose heap marks its major collections in `On` cycles, each slice a few cells (a
/// test-only budget in place of a slice's bytes and time), in `tier`: a large element block's
/// chunks span slices as in an ordinary cycle, whose precleans run only once its worklist is
/// out of work.
fn chunking_engine(tier: Tier) -> Engine {
    let engine = cycling_engine(tier);
    let mut core = engine.interp.gc_heap.core.borrow_mut();
    core.set_incremental(IncrementalMode::On);
    core.set_test_slice_cells(Some(4));
    drop(core);
    engine
}

/// W6 step 6: a cycle traces a large element block a chunk at a time across its slices, its
/// owner queued again for each chunk, and a take of the owner's card traces the block again from
/// its first chunk: a write may have landed behind the chunks traced. Objects stored only behind
/// them by an element write, a splice and an unshift are marked by the slices once the block's
/// chunks end and a preclean, out of work, takes the card (only the block holds them, and slices
/// mark no roots). A last store behind the chunks, which no slice follows, survives the final
/// pause through its take of the card, and verification (the unit tests verify every
/// collection) checks the final marking's closure.
#[test]
fn a_block_marked_in_chunks_keeps_what_is_stored_behind_its_chunks_in_every_tier() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = chunking_engine(tier);
        eval(
            &mut engine,
            "var big = []; for (let i = 0; i < 20000; i++) big.push({ i }); true",
        );
        let big_weak = weak_global(&mut engine, "big");
        let big = hidden_address(&big_weak);
        assert!(engine.interp.gc_start_cycle(), "{tier:?}");
        slice_until_between_chunks(&mut engine, big, "the first chunks");
        eval(
            &mut engine,
            r#"
            big[0] = { kept: 'first' };
            big.splice(1, 0, { kept: 'spliced' });
            big.unshift({ kept: 'unshifted' });
            true
            "#,
        );
        let stored: Vec<_> = (0..3)
            .map(|index| weak_element(&mut engine, "big", index))
            .collect();
        for (index, weak) in stored.iter().enumerate() {
            let what = format!("{tier:?}: the object stored at {index}");
            slice_until_marked(&mut engine, hidden_address(weak), &what);
        }
        slice_until_between_chunks(&mut engine, big, "the chunks after the restart");
        // No slice between the last store and the final pause.
        let slices = cycle_slices(&engine);
        eval(&mut engine, "big[3] = { kept: 'last' }; true");
        assert_eq!(cycle_slices(&engine), slices, "{tier:?}: no slice ran");
        assert!(
            chunk_progress(&engine, big).is_some_and(|next| next > 3),
            "{tier:?}: the store is behind the chunks traced"
        );
        engine.interp.gc_collect_minor();
        assert!(!cycle_running(&engine), "{tier:?}");
        let result = eval(
            &mut engine,
            r#"
            let kept = '';
            for (let i = 0; i < 4; i++) kept += big[i].kept + ' ';
            kept + big.length + ' ' + big[4].i + ' ' + big[20001].i
            "#,
        );
        assert_eq!(
            result, "unshifted first spliced last 20002 2 19999",
            "{tier:?}"
        );
    }
}

/// Whether the card of the cell at `address` is set.
#[inline(never)]
fn card_set(engine: &Engine, address: crate::gc_test::HiddenAddress) -> bool {
    let core = engine.interp.gc_heap.core.borrow();
    core.card_set_at(address.get())
}

/// W6 step 6: under force (verification's mode), a preclean's pass begun at a forced slice's end
/// leaves the card of a chunked block's owner set, as an `On` cycle's precleans, which run only
/// out of work, never meet a block between its chunks: the block keeps its chunks, and objects
/// stored behind them survive the final pause through its take of the card, which traces the
/// block again from its first chunk.
#[test]
fn a_forced_slices_preclean_leaves_a_chunked_blocks_card_for_the_final_pause_in_every_tier() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = cycling_engine(tier);
        eval(
            &mut engine,
            "var big = []; for (let i = 0; i < 2000; i++) big.push({ i }); true",
        );
        let big_weak = weak_global(&mut engine, "big");
        let big = hidden_address(&big_weak);
        assert!(engine.interp.gc_start_cycle(), "{tier:?}");
        slice_until_between_chunks(&mut engine, big, "the first chunks");
        eval(
            &mut engine,
            "big[0] = { kept: 'first' }; big[1] = { kept: 'second' }; true",
        );
        for _ in 0..64 {
            assert!(
                card_set(&engine, big),
                "{tier:?}: the forced passes keep the card"
            );
            assert!(engine.interp.gc_slice(), "{tier:?}");
        }
        assert!(
            cycle_running(&engine),
            "{tier:?}: the block's children keep the cycle marking"
        );
        engine.interp.gc_collect_minor();
        assert!(!cycle_running(&engine), "{tier:?}");
        let result = eval(
            &mut engine,
            "big[0].kept + ' ' + big[1].kept + ' ' + big.length + ' ' + big[1999].i",
        );
        assert_eq!(result, "first second 2000 1999", "{tier:?}");
    }
}

/// W6 step 6: under force, a cycle finishes on its own while the program writes a chunked block
/// at every safepoint: a forced slice's pass restarting the block at every pass would keep its
/// chunks from ever ending, and the cycle marking until its margin forced the final pause.
#[test]
fn a_forced_cycle_finishes_while_the_program_writes_a_chunked_block() {
    let mut engine = cycling_engine(Tier::Bytecode);
    eval(
        &mut engine,
        "var hot = []; for (let i = 0; i < 2000; i++) hot.push({ i }); true",
    );
    assert!(engine.interp.gc_start_cycle());
    let collections = engine.interp.gc_heap.core.borrow().collections();
    eval(
        &mut engine,
        "for (let i = 0; i < 100000; i++) hot[i % 2] = hot[(i + 1) % 2]; true",
    );
    assert!(
        !cycle_running(&engine),
        "the cycle's final pause ran on its own"
    );
    assert_eq!(
        engine.interp.gc_heap.core.borrow().collections(),
        collections + 1
    );
    assert_eq!(
        eval(&mut engine, "hot.length + ' ' + hot[1999].i"),
        "2000 1999"
    );
}

/// W6 step 6 (a step 4 bug, cycles on only, which the default-on suite found): the external
/// bytes a program charges during a cycle count toward its margin as they come. A Map's tables
/// grown past the margin make the cycle's final pause due, which the poll words' heap bit says
/// at once (debug builds check at every safepoint that it agrees with `poll_due`), and a
/// safepoint runs it while the program still grows the Map.
#[test]
fn a_maps_tables_grown_past_a_cycles_margin_bring_its_final_pause() {
    let mut engine = cycling_engine(Tier::Bytecode);
    engine
        .interp
        .gc_heap
        .core
        .borrow_mut()
        .set_incremental(IncrementalMode::On);
    eval(
        &mut engine,
        "var keep = []; for (let i = 0; i < 20000; i++) keep.push({ i }); true",
    );
    {
        let mut core = engine.interp.gc_heap.core.borrow_mut();
        // A margin of 2 MB, half the room to a maximum 4 MB past the budget: it is passed once
        // the heap's bytes and the external bytes exceed the budget by 2 MB, which the Map's
        // tables (tens of bytes an entry, 8 MiB of budget) do before the loop ends.
        let budget = core.stats().trigger_bytes;
        core.set_max_heap_bytes(budget + (4 << 20));
    }
    assert!(engine.interp.gc_start_cycle());
    let collections = engine.interp.gc_heap.core.borrow().collections();
    let result = eval(
        &mut engine,
        "var m = new Map(); for (let j = 0; j < 300000; j++) m.set(j, j); m.size",
    );
    assert_eq!(result, "300000");
    assert!(!cycle_running(&engine), "the margin's final pause ran");
    assert!(engine.interp.gc_heap.core.borrow().collections() > collections);
}

/// The weak collection held by an embedder's handle (`Ctx::handle`), as TRust keeps its platform
/// slots' WeakMaps: a root, so the cycle's start marks it and fires its side-slot owner edge with
/// the start's client, which discovers the collection (step 2b). The values that only its entries
/// hold must survive the cycle's own final pause (an explicit minor request, which collects no
/// further), whatever the program writes meanwhile.
#[cfg(feature = "embed")]
fn rooted_weak_collection_fixture(engine: &mut Engine) -> crate::embed::Handle {
    eval(
        engine,
        r#"
        var keys = [], wm = new WeakMap(), ws = new WeakSet();
        for (let i = 0; i < 64; i++) {
            const k = { i };
            keys.push(k);
            wm.set(k, { v: i, kids: [i, i + 1] });
            ws.add(k);
        }
        var probe = wm.get(keys[7]);
        true
        "#,
    );
    engine.scope(|cx| {
        let global = cx.global_this();
        let map = cx.member_get(&global, "wm").expect("the fixture's WeakMap");
        cx.handle(&map).expect("a handle to the WeakMap")
    })
}

/// Whether the rooted fixture's entries are all intact (`rooted_weak_collection_fixture`).
#[cfg(feature = "embed")]
fn rooted_weak_collection_intact(engine: &mut Engine) -> String {
    eval(
        engine,
        r#"
        let ok = keys.length === 64;
        for (let i = 0; i < 64; i++) {
            const entry = wm.get(keys[i]);
            ok = ok && entry !== undefined && entry.v === i && entry.kids[1] === i + 1;
            ok = ok && ws.has(keys[i]);
        }
        ok
        "#,
    )
}

/// A weak collection that the cycle's start marks (a root) keeps its entries' values through the
/// cycle: the start registers the entries its client discovered, as each slice does. Before, the
/// start's client was dropped with its discoveries, and the final pause freed every value only the
/// entries held (TRust's pages: DOM wrappers' internal-slot records gone, `classList.toggle is not
/// a function`, a dead key reaching `observe_weak_target`).
#[cfg(feature = "embed")]
#[test]
fn a_weak_collection_marked_at_a_cycles_start_keeps_its_entries_in_every_tier() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = cycling_engine(tier);
        let handle = rooted_weak_collection_fixture(&mut engine);
        let weak = weak_global(&mut engine, "probe");
        eval(&mut engine, "probe = null; 0");
        assert!(engine.interp.gc_start_cycle(), "{tier:?}");
        engine.interp.gc_slice_to_completion();
        engine.interp.gc_collect_minor();
        assert!(
            !cycle_running(&engine),
            "{tier:?}: the final pause ended it"
        );
        assert!(
            crate::gc_test::alive(&*weak),
            "{tier:?}: an entry's value survives the cycle that marked its rooted WeakMap at its start"
        );
        assert_eq!(
            rooted_weak_collection_intact(&mut engine),
            "true",
            "{tier:?}"
        );
        drop(handle);
    }
}

/// Two Agents on one thread, each with a weak collection its embedder roots, switching while a
/// cycle is open in one and then in both: each heap's cycle keeps its own entries' values, and the
/// survivors serve as weak targets afterwards (WeakRef, FinalizationRegistry, WeakMap keys), whose
/// registration looks up their heap (`observe_weak_target`). Every direct call into an engine's
/// interpreter first makes its Agent the active one, as the engine's own entries do.
#[cfg(feature = "embed")]
#[test]
fn cycles_open_across_agent_switches_keep_each_heaps_weak_entries() {
    fn on(engine: &mut Engine) -> &mut Engine {
        engine.interp.activate_gc_heap();
        engine
    }
    let mut first = cycling_engine(Tier::Jit);
    let mut second = cycling_engine(Tier::Bytecode);
    let first_handle = rooted_weak_collection_fixture(on(&mut first));
    let first_probe = weak_global(on(&mut first), "probe");
    let second_handle = rooted_weak_collection_fixture(on(&mut second));
    let second_probe = weak_global(on(&mut second), "probe");
    eval(&mut first, "probe = null; 0");
    eval(&mut second, "probe = null; 0");
    let churn = r#"
        var churned = [];
        for (let i = 0; i < 2000; i++) churned.push({ i, s: 'x' + i });
        for (let i = 0; i < 64; i += 3) wm.set(keys[i], { v: i, kids: [i, i + 1], again: true });
        churned.length
    "#;
    // The first heap's cycle opens, and the second Agent runs: an Agent's cycle advances at its own
    // safepoints alone. Then the second's opens too, and each runs while the other's is open.
    assert!(on(&mut first).interp.gc_start_cycle());
    assert_eq!(eval(&mut second, churn), "2000");
    assert!(
        cycle_running(&first),
        "the other Agent's run leaves the cycle open"
    );
    assert!(on(&mut second).interp.gc_start_cycle());
    assert_eq!(eval(&mut first, churn), "2000");
    assert!(
        cycle_running(&second),
        "the other Agent's run leaves the cycle open"
    );
    assert_eq!(eval(&mut second, churn), "2000");
    // Each cycle ends with its own final pause (an explicit minor request), unless its own run
    // finished it already.
    for engine in [&mut first, &mut second] {
        let engine = on(engine);
        if cycle_running(engine) {
            engine.interp.gc_slice_to_completion();
            engine.interp.gc_collect_minor();
        }
        assert!(!cycle_running(engine));
    }
    for (engine, probe) in [(&mut first, &first_probe), (&mut second, &second_probe)] {
        let engine = on(engine);
        assert!(
            crate::gc_test::alive(&**probe),
            "an entry's value survives its heap's cycle"
        );
        assert_eq!(rooted_weak_collection_intact(engine), "true");
        let weak_targets = r#"
            var refs = [], registry = new FinalizationRegistry(() => {}), keyed = new WeakMap();
            for (let i = 0; i < 64; i++) {
                const value = wm.get(keys[i]);
                refs.push(new WeakRef(value));
                registry.register(value, i);
                keyed.set(value, i);
            }
            refs.every((ref, i) => ref.deref() === wm.get(keys[i]) && keyed.get(ref.deref()) === i)
        "#;
        assert_eq!(eval(engine, weak_targets), "true");
    }
    drop((first_handle, second_handle));
}
