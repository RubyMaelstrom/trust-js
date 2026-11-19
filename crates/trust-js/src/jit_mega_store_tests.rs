//! The engine-wide store table probed by generated code (the megamorphic line's piece 1c): a
//! named store whose site's ways miss, at a megamorphic site, takes the store table's entry for
//! the receiver's shape, the site's key and the receiver's prototype (`emit_store_mega`) and
//! commits it as a way's: an update of an own data property, or a creation (one empty shape
//! under eight prototypes, a base constructor's `this.x =`). After the entries are filled every
//! store is native, at a site made cold (the compact store stub) and at a site made warm (the
//! in-line ways, then `SharedStub::StoreMega`); every tier agrees; two engines alive on one thread
//! each store through their own table (the stubs are process-wide).

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn engine(tier: Tier, warm: bool) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(if warm { 1 } else { 0 });
    if warm {
        engine.interp.jit_admit_runs = 20;
    }
    engine
}

fn helpers() -> usize {
    crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(std::cell::Cell::get)
}

const SOURCE: &str = r#"
var protos = []; for (var k = 0; k < 8; k++) protos.push({ ['p' + k]: k });
var objs = []; for (var k = 0; k < 8; k++) { var o = {}; o['s' + k] = k; o.v = 0; objs.push(o); }
function upd(o, v) { o.v = v; }
function mk(k) { var o = Object.create(protos[k]); o.q = k; o.r = k + 1; return o; }
function runu(n) { for (var i = 0; i < n; i++) upd(objs[i & 7], i); var s = 0; for (var k = 0; k < 8; k++) s += objs[k].v; return s; }
function runc(n) { var s = 0; for (var i = 0; i < n; i++) { var o = mk(i & 7); s += o.q + o.r; } return s + ':' + Object.keys(mk(3)).join(','); }
runu(32); runc(32);
"#;

const STEPS: [&str; 2] = ["runu(80)", "runc(80)"];

/// Each step's value; with `native`, a pass after a filling pass must make no store helper call.
/// The creation ways and entries hold the process-wide creation epoch, which concurrent tests move
/// (a pinned prototype dropped with its engine, a prototype redefined): a pass under which it
/// moved refills them, so the check takes a filling pass and a checked pass with the epoch still
/// through both, and runs the two again (bounded) when it was not.
fn steps(engine: &mut Engine, native: Option<&str>) -> Vec<String> {
    let mut values = Vec::new();
    for step in STEPS {
        let first = eval(engine, step);
        let mut tries = 0;
        let again = loop {
            let epoch = crate::value::proto_epoch();
            assert_eq!(eval(engine, step), first);
            let (before, filled) = (helpers(), crate::value::proto_epoch());
            let again = eval(engine, step);
            assert_eq!(first, again);
            let still = epoch == filled && filled == crate::value::proto_epoch();
            if let Some(form) = native {
                if !still && tries < 16 {
                    tries += 1;
                    continue;
                }
                assert_eq!(
                    helpers() - before,
                    0,
                    "{form} `{step}`: every store native once the entries are filled"
                );
            }
            break again;
        };
        values.push(again);
    }
    values
}

#[test]
fn a_megamorphic_store_site_is_served_by_the_store_table_in_generated_code() {
    crate::opt::pin_tier_off_for_test();
    let mut interp = engine(Tier::Interp, false);
    eval(&mut interp, SOURCE);
    let expected = steps(&mut interp, None);
    assert_eq!(expected, ["604", "640:q,r"]);
    let mut vm = engine(Tier::Bytecode, false);
    eval(&mut vm, SOURCE);
    assert_eq!(steps(&mut vm, None), expected, "the VM");
    for (warm, form) in [(false, "cold"), (true, "warm")] {
        let mut jit = engine(Tier::Jit, warm);
        eval(&mut jit, SOURCE);
        assert_eq!(steps(&mut jit, Some(form)), expected, "{form}");
    }
}

#[test]
fn two_engines_alive_on_one_thread_each_store_through_their_own_table() {
    crate::opt::pin_tier_off_for_test();
    let mut a = engine(Tier::Jit, false);
    let mut b = engine(Tier::Jit, false);
    eval(&mut a, SOURCE);
    eval(
        &mut b,
        "var pad = []; for (var i = 0; i < 100; i++) pad.push({ ['z' + i]: i });",
    );
    eval(&mut b, SOURCE);
    for _ in 0..2 {
        assert_eq!(steps(&mut a, Some("a")), ["604", "640:q,r"]);
        assert_eq!(steps(&mut b, Some("b")), ["604", "640:q,r"]);
    }
}

fn global(engine: &mut Engine, name: &str) -> crate::value::Value {
    let env = engine.interp.global_env;
    engine.interp.get_var(name, &env).ok().expect("bound name")
}

/// The global function `function`'s compiled chunk and its one named store's cache index.
fn store_site(engine: &mut Engine, function: &str) -> (std::rc::Rc<crate::bytecode::Chunk>, u32) {
    use crate::bytecode::Op;
    let crate::value::ValueView::Obj(f) = global(engine, function).view() else {
        panic!("{function}")
    };
    let crate::value::ObjectKind::User(user) = &f.borrow().kind else {
        panic!("{function} is not a user function")
    };
    let runtime = user.func.runtime();
    let chunk = std::rc::Rc::clone(
        runtime
            .code
            .get()
            .and_then(Option::as_ref)
            .expect("compiled"),
    );
    let caches: Vec<u32> = chunk
        .jit_ops()
        .iter()
        .filter_map(|op| match *op {
            Op::SetProp(_, c) | Op::SetPropDrop(_, c) | Op::SetPropLocalDrop(_, _, c) => Some(c),
            _ => None,
        })
        .collect();
    assert_eq!(caches.len(), 1, "one named store in {function}");
    (chunk, caches[0])
}

/// The receiver shapes of the global function `function`'s one named store's ways.
fn store_ways(engine: &mut Engine, function: &str) -> [u32; crate::bytecode::PROP_IC_WAYS] {
    let (chunk, cache) = store_site(engine, function);
    chunk.jit_cache_ways(cache).map(|way| way.recv_shape)
}

/// The address of the global function `function`'s one named store's cache (its first way), the
/// token its generated probe of the table leaves when it misses (`Interp::store_probe_token`).
fn store_token(engine: &mut Engine, function: &str) -> usize {
    let (chunk, cache) = store_site(engine, function);
    chunk.jit_cache_ptr(cache)
}

/// `Interp::try_ic_set`'s probes of the store table and their hits so far, on this thread.
fn table_probes() -> [usize; 2] {
    crate::interpreter::TEST_STORE_TABLE_PROBES.with(std::cell::Cell::get)
}

/// A store the table serves in generated code is promoted into its site's ways, as the helper's
/// hit of the table is (`Interp::try_ic_set` -> `ic_insert`): way 0 takes the receiver's shape and
/// the other ways move down one.
#[test]
fn a_store_the_table_serves_in_generated_code_is_promoted_into_its_sites_ways() {
    crate::opt::pin_tier_off_for_test();
    for (tier, warm) in [
        (Tier::Jit, false),
        (Tier::Jit, true),
        (Tier::Bytecode, false),
    ] {
        let mut jit = engine(tier, warm);
        eval(
            &mut jit,
            &(SOURCE.to_string()
                + "for (var k = 0; k < 8; k++) globalThis['o' + k] = objs[k]; runu(64);"),
        );
        let mut promoted = 0;
        for k in [0, 5, 2, 7, 4, 1, 6, 3] {
            let shape = match global(&mut jit, &format!("o{k}")).view() {
                crate::value::ValueView::Obj(o) => o.borrow().props.shape(),
                _ => panic!("o{k}"),
            };
            let before = store_ways(&mut jit, "upd");
            let calls = helpers();
            eval(&mut jit, &format!("upd(o{k}, {k})"));
            let after = store_ways(&mut jit, "upd");
            if before.contains(&shape) {
                continue;
            }
            assert_eq!(helpers(), calls, "warm {warm}: o{k}'s store is native");
            assert_eq!(
                after,
                [shape, before[0], before[1], before[2]],
                "warm {warm}: o{k}'s entry promoted into way 0"
            );
            promoted += 1;
        }
        assert!(promoted >= 2, "warm {warm}: {promoted} promotions");
    }
}

/// Only a megamorphic store site probes the table: a site whose ways hold no generic mark takes
/// the helper, whose fill teaches its own ways, though the table holds another site's entry for
/// the same shape and key.
#[test]
fn a_store_site_short_of_megamorphic_fills_its_own_ways_and_does_not_probe_the_table() {
    crate::opt::pin_tier_off_for_test();
    let mut jit = engine(Tier::Jit, false);
    eval(
        &mut jit,
        &(SOURCE.to_string() + "function upd2(o, v) { o.v = v; }"),
    );
    let calls = helpers();
    eval(&mut jit, "upd2(objs[3], 1)");
    assert_eq!(helpers() - calls, 1, "the helper fills upd2's way");
    eval(&mut jit, "upd2(objs[3], 2)");
    assert_eq!(helpers() - calls, 1, "upd2's own way");
}

/// One probe a miss: a megamorphic creation site whose every object has a new prototype (the
/// always-wrong case) misses the table in generated code on every store, and its helper does not
/// probe the table again (the token the probe left names the site), as no exemplar's miss
/// handler probes its cache again.
#[test]
fn a_store_whose_generated_probe_missed_does_not_probe_the_table_again() {
    crate::opt::pin_tier_off_for_test();
    let mut jit = engine(Tier::Jit, false);
    eval(
        &mut jit,
        r#"
        function mk(p, v) { var o = Object.create(p); o.q = v; return o; }
        function run(n) { var s = 0; for (var j = 0; j < n; j++) s += mk({}, j).q & 1; return s; }
        run(64);
    "#,
    );
    let (calls, probes) = (helpers(), table_probes());
    assert_eq!(eval(&mut jit, "run(200)"), "100");
    assert!(
        helpers() - calls >= 200,
        "every creation misses: {}",
        helpers() - calls
    );
    assert_eq!(
        table_probes()[0] - probes[0],
        0,
        "the generated probe missed: the helper does not probe again"
    );
    assert_eq!(
        jit.interp.store_probe_token[0].get(),
        0,
        "every token taken"
    );
}

/// A token decides nothing but speed. The helper's entry takes it for every store that reaches
/// it, and honours it only for the site it names: a token left for site `sa`, then a store
/// through another site `sb` whose shape the table holds (it hits, and takes the token), then
/// `sa` again with a shape the table holds: it hits too. A token honoured where the table would
/// have hit only loses that hit's fast path: the store takes the full lookup and lands.
#[test]
fn a_stale_store_token_never_suppresses_another_stores_table_hit() {
    crate::opt::pin_tier_off_for_test();
    let mut jit = engine(Tier::Jit, false);
    eval(
        &mut jit,
        r#"
        var shapes = [];
        for (var k = 0; k < 8; k++) { var o = {}; o['s' + k] = k; o.v = 0; shapes.push(o); }
        function upd(o, x) { o.v = x; }
        function sa(o, x) { o.v = x; }
        function sb(o, x) { o.v = x; }
        var mine = { a: 1, v: 0 }, theirs = { b: 1, v: 0 };
        for (var i = 0; i < 64; i++) upd(shapes[i & 7], i);
        for (var i = 0; i < 8; i++) { sa(mine, i); sb(theirs, i); }
    "#,
    );
    let token = store_token(&mut jit, "sa");
    // Left for `sa`, as its probe would leave it; `sb`'s store reaches its helper (its ways hold
    // `theirs` alone) and its probe of the table hits.
    jit.interp.store_probe_token[0].set(token);
    jit.interp.store_probe_token[1].set(0); // no entry met: the fill hashes its slot
    let before = table_probes();
    eval(&mut jit, "sb(shapes[5], 7)");
    assert_eq!(
        table_probes()[1] - before[1],
        1,
        "sb's store hits the table: the token names another site"
    );
    assert_eq!(
        jit.interp.store_probe_token[0].get(),
        0,
        "sb's helper took the token"
    );
    let before = table_probes();
    eval(&mut jit, "sa(shapes[3], 5)");
    assert_eq!(
        table_probes()[1] - before[1],
        1,
        "sa's store hits the table: the stale token decided nothing"
    );
    // Honoured where the table would hit: no probe, the full lookup, the same value.
    jit.interp.store_probe_token[0].set(token);
    jit.interp.store_probe_token[1].set(0); // no entry met: the fill hashes its slot
    let before = table_probes();
    eval(&mut jit, "sa(shapes[4], 6)");
    assert_eq!(table_probes(), before, "an honoured token skips the probe");
    assert_eq!(jit.interp.store_probe_token[0].get(), 0);
    assert_eq!(
        eval(
            &mut jit,
            "shapes[3].v + ':' + shapes[4].v + ':' + shapes[5].v"
        ),
        "5:6:7"
    );
}

/// A creation entry of the store table is taken for its own prototype alone: two live prototypes
/// whose entries share a slot (the empty shape and the key the same; found among candidates that
/// alternate a writable data `q`, under which a creation still creates on the receiver, and a
/// setter `q`), and a megamorphic creation site that fills the first's entry: a creation under
/// the second calls its setter, in generated code (the prototype compare a table entry takes
/// first, `emit_store_mega`) and in the helper. Nothing here moves the process-wide creation
/// epoch, which other tests' counts read.
#[test]
fn a_store_table_creation_entry_serves_its_own_prototype_alone() {
    crate::opt::pin_tier_off_for_test();
    for warm in [false, true] {
        let mut jit = engine(Tier::Jit, warm);
        eval(
            &mut jit,
            r#"
            var seen = 0, cands = [];
            for (var k = 0; k < 3000; k++)
                cands.push(k & 1 ? { set q(v) { seen += v; } } : { q: 0 });
            function mk(p, v) { var o = Object.create(p); o.q = v; return o; }
            var others = []; for (var k = 0; k < 8; k++) others.push({ ['p' + k]: k });
            for (var k = 0; k < 40; k++) mk(others[k & 7], k);
            var empty = Object.create(others[0]);
        "#,
        );
        let cands = global(&mut jit, "cands");
        let crate::value::ValueView::Obj(e) = global(&mut jit, "empty").view() else {
            panic!("empty")
        };
        let shape = e.borrow().props.shape();
        let key = crate::atom::PropKey::from_text("q");
        let mut by_slot: [std::collections::HashMap<usize, usize>; 2] = Default::default();
        let mut pair = None;
        for k in 0..3000 {
            let c = jit.interp.get_member(&cands, &k.to_string()).ok().unwrap();
            let crate::value::ValueView::Obj(o) = c.view() else {
                panic!("candidate")
            };
            let slot =
                crate::interpreter::store_slot(shape, key, crate::value::Gc::as_ptr(&o) as usize);
            by_slot[k & 1].insert(slot, k);
            if let Some(&other) = by_slot[1 - (k & 1)].get(&slot) {
                pair = Some(if k & 1 == 1 { (other, k) } else { (k, other) });
                break;
            }
        }
        let (x, y) = pair.expect("a data and a setter candidate share a store slot");
        let out = eval(
            &mut jit,
            &format!(
                r#"
                var x = cands[{x}], y = cands[{y}];
                mk(x, 1); mk(x, 2);
                var r = mk(y, 5), s = mk(y, 7);
                seen + ':' + r.hasOwnProperty('q') + ':' + s.hasOwnProperty('q') + ':' + mk(x, 3).q
            "#
            ),
        );
        assert_eq!(
            out, "12:false:false:3",
            "warm {warm}: y's setter, x's creation"
        );
    }
}
