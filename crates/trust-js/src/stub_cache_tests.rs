//! The engine-wide property cache (`Interp::stub_cache`, the megamorphic line's piece 1): an
//! entry holds its key beside its resolution, and a probe compares both the key and the
//! receiver's shape. Two keys of one object that share a table slot never read each other's
//! property.
use super::*;
use crate::bytecode::{IcState, PROP_IC_WAYS};
use crate::value::ValueView;
use crate::{Completion, Engine};
use std::cell::Cell;

fn eval(engine: &mut Engine, source: &str) {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

/// Two names whose keys take the same slot of the table whatever the receiver's shape:
/// `stub_slot` mixes the shape in by xor after the key, so equal key mixes collide for every
/// shape.
fn colliding_names() -> (String, String) {
    let mix = |name: &str| {
        let n = crate::atom::PropKey::from_text(name).word() as usize >> 3;
        (n ^ (n >> 7)) & (STUB_CACHE_SIZE - 1)
    };
    let mut seen = std::collections::HashMap::new();
    for k in 0..20_000 {
        let name = format!("k{k}");
        if let Some(other) = seen.insert(mix(&name), name.clone()) {
            return (other, name);
        }
    }
    panic!("no two of 20,000 names share a slot");
}

#[test]
fn two_keys_in_one_slot_never_read_each_other() {
    let mut engine = Engine::new();
    let (first, second) = colliding_names();
    eval(
        &mut engine,
        &format!("var o = {{}}; o.{first} = 1; o.{second} = 2;"),
    );
    let global = Value::Obj(engine.interp.global);
    let o = engine
        .interp
        .get_member_key(&global, "o")
        .unwrap_or_else(|_| panic!("global o"));
    let shape = o.as_obj().expect("an object").borrow().props.shape();
    let key = |name: &str| crate::atom::PropKey::from_text(name);
    assert_eq!(
        stub_slot(shape, key(&first)),
        stub_slot(shape, key(&second))
    );
    let read = |engine: &mut Engine, name: &str| {
        // A fresh site each read: its ways miss, so the read probes the table first.
        let ways = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
        match engine
            .interp
            .get_prop_ic(&o, name, key(name), &ways[0])
            .unwrap_or_else(|_| panic!("o.{name} threw"))
            .view()
        {
            ValueView::Num(n) => n,
            _ => f64::NAN,
        }
    };
    assert_eq!(read(&mut engine, &first), 1.0);
    let entry = engine
        .interp
        .stub_cache_get(shape, key(&first))
        .expect("the first read filled its entry");
    assert!(entry.key == Some(key(&first)));
    assert!(engine.interp.stub_cache_get(shape, key(&second)).is_none());
    // The second name's read finds the first's entry in its slot and must not take it.
    assert_eq!(read(&mut engine, &second), 2.0);
    assert_eq!(read(&mut engine, &first), 1.0);
    // An entry's layout: the key's word first, the state after it (generated code reads both).
    assert_eq!(std::mem::offset_of!(StubEntry, key), 0);
    assert_eq!(std::mem::offset_of!(StubEntry, st), 8);
    assert_eq!(std::mem::size_of::<std::cell::Cell<StubEntry>>(), 40);
}

/// The global `name`'s object.
fn object(engine: &mut Engine, name: &str) -> Gc {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let ValueView::Obj(object) = value.view() else {
        panic!("{name} is not an object")
    };
    object
}

fn proto_of(object: &Gc) -> usize {
    object
        .borrow()
        .proto
        .as_ref()
        .map_or(0, |p| Gc::as_ptr(p) as usize)
}

/// The store table is a table of its own (the megamorphic line's piece 1c): at a megamorphic site
/// that reads a name its receivers lack and then creates it (`if (!o.x) o.x = v`), the read's
/// cached absence stays in the read table and the creations in the store table, for one receiver
/// shape under eight prototypes: neither evicts the other's entry (in one shared table the
/// creation's entry took the absence's slot).
#[test]
fn a_reads_absence_and_a_stores_creation_of_one_shape_keep_their_own_tables() {
    use crate::bytecode::{IC_ABSENT, IC_CREATE};
    let mut engine = Engine::new();
    engine.set_tier(crate::bytecode::Tier::Bytecode);
    eval(
        &mut engine,
        "var protos = []; for (var k = 0; k < 8; k++) protos.push({ ['p' + k]: k });
         function f(o, v) { if (!o.x) o.x = v; return o.x; }
         var s = 0;
         for (var i = 0; i < 64; i++) s += f(Object.create(protos[i & 7]), i + 1);
         for (var k = 0; k < 8; k++) globalThis['s' + k] = Object.create(protos[k]);
         f(s7, 1);
         var last = Object.create(protos[7]);",
    );
    let key = crate::atom::PropKey::from_text("x");
    let last = object(&mut engine, "last");
    let shape = last.borrow().props.shape();
    let read = engine
        .interp
        .stub_cache_get(shape, key)
        .expect("the read's entry for the empty shape");
    assert_eq!(read.st.depth, IC_ABSENT, "the read table keeps the absence");
    let store = engine
        .interp
        .store_cache_get(shape, key, proto_of(&last))
        .expect("the store's entry for the empty shape under protos[7]");
    assert_eq!(
        store.st.depth, IC_CREATE,
        "the store table keeps the creation"
    );
    assert_eq!(store.st.recv_shape, shape);
}

/// R2 (F4 part 2) and the realms' unique global shapes (U1) for the store table: a store at a
/// megamorphic site into a watched global's Constant property finds no entry (only the store fill
/// fills the table, after its write's report), so it reports and moves the property's state;
/// and the entry that store fills, for its realm's global's shape, is not found for another
/// realm's global.
#[test]
fn a_store_table_hit_never_skips_a_watched_globals_report_nor_crosses_realms() {
    use crate::object_fuse::PropertyState;
    let mut engine = Engine::new();
    engine.set_tier(crate::bytecode::Tier::Bytecode);
    eval(
        &mut engine,
        "var gx = 7; function rd() { return globalThis.gx; } rd(); rd();",
    );
    let global = engine.interp.global;
    let key = crate::atom::PropKey::from_text("gx");
    let slot = global.borrow().props.slot_of("gx").expect("gx");
    let shape = global.borrow().props.shape();
    let fuse = engine.interp.global_fuse().expect("the global's fuse");
    let _ = fuse.try_optimize_constant(slot);
    assert_eq!(fuse.state(slot), PropertyState::Constant);
    let ways = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
    engine.interp.ic_mark_generic(&ways[0]);
    let stored = engine
        .interp
        .try_ic_set::<true>(&global, key, &Value::Num(8.0), &ways[0], 0);
    assert!(matches!(stored, IcSet::Stored));
    assert_eq!(
        fuse.state(slot),
        PropertyState::NotConstant,
        "the store reported to the fuse"
    );
    let entry = engine
        .interp
        .store_cache_get(shape, key, proto_of(&global))
        .expect("the store's fill, now NotConstant");
    assert_eq!(entry.st.recv_shape, shape);
    // Another realm's global: its own unique shape, no entry of the first realm's serves it.
    let ValueView::Obj(other) = engine.interp.create_realm().view() else {
        panic!("a realm's global")
    };
    other
        .borrow_mut()
        .props
        .insert("gx", Property::data(Value::Num(0.0), true, true, true));
    let theirs = other.borrow().props.shape();
    assert_ne!(theirs, shape, "each realm's global has its own shape");
    assert!(engine
        .interp
        .store_cache_get(theirs, key, proto_of(&other))
        .is_none_or(|e| e.st.recv_shape != theirs));
}

/// Two keys whose store-table slots coincide for every shape and prototype (`store_slot` mixes
/// the key as `stub_slot` does) never write each other's property: the entry's key is compared.
#[test]
fn two_keys_in_one_store_slot_never_write_each_other() {
    let mut engine = Engine::new();
    engine.set_tier(crate::bytecode::Tier::Bytecode);
    let (a, b) = colliding_names();
    eval(&mut engine, &format!("var o = {{}}; o.{a} = 0; o.{b} = 0;"));
    let o = object(&mut engine, "o");
    let (shape, proto) = (o.borrow().props.shape(), proto_of(&o));
    let slot = |name: &str| store_slot(shape, crate::atom::PropKey::from_text(name), proto);
    assert_eq!(slot(&a), slot(&b), "the two keys share a store slot");
    let set = |engine: &mut Engine, name: &str, v: f64| {
        let ways = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
        engine.interp.ic_mark_generic(&ways[0]);
        let key = crate::atom::PropKey::from_text(name);
        assert!(matches!(
            engine
                .interp
                .try_ic_set::<true>(&o, key, &Value::Num(v), &ways[0], 0),
            IcSet::Stored
        ));
    };
    set(&mut engine, &a, 1.0);
    assert!(
        engine
            .interp
            .store_cache_get(shape, crate::atom::PropKey::from_text(&a), proto)
            .is_some(),
        "the first store filled its entry"
    );
    set(&mut engine, &b, 2.0);
    set(&mut engine, &a, 3.0);
    let both = engine.eval(&format!("o.{a} + ',' + o.{b}"), false);
    assert!(matches!(both, Ok(Completion::Value(ref v)) if v == "3,2"));
}

fn value(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(v) => v,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

/// The store table's probes and hits so far, and Reflect.set's stores that went on to the fill.
fn reflect_counts() -> ([usize; 2], usize) {
    (
        TEST_STORE_TABLE_PROBES.with(Cell::get),
        TEST_REFLECT_SET_FILLS.with(Cell::get),
    )
}

/// Keys from `Object.keys`, which carry their key (`lstr::KEY_MARK`), as a reactive set trap's do.
const REFLECT_KEYS: &str = "var kx = Object.keys({ x: 0 })[0], kc = Object.keys({ c: 0 })[0];";

/// Reflect.set takes the store table before the store path's scratch ways, once a store: a miss
/// probes the table once and hands the entry it met to the fill (no second probe), and the store
/// after it, of the same shape, key and prototype, is the table's (no fill). Generated code's
/// native call reaches the resolution (`native_stub_call`); the VM's call of a native does not.
/// In a process of its own (`run_alone`): a call whose cache the process-wide call caches' epoch
/// staled takes the generic call, which reaches no resolution, and its counts would read none.
#[test]
fn reflect_set_takes_the_store_table_once_before_its_scratch_ways() {
    crate::opt::pin_tier_off_for_test();
    if !crate::jit::construct_tests::run_alone(
        "interpreter::stub_cache_tests::reflect_set_takes_the_store_table_once_before_its_scratch_ways",
    ) {
        return;
    }
    let tier = crate::bytecode::Tier::Jit;
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    eval(&mut engine, REFLECT_KEYS);
    eval(
        &mut engine,
        "var o = { a: 1, x: 0 }; function rs(o, k, v) { return Reflect.set(o, k, v); }
         for (var i = 0; i < 64; i++) rs({ ['w' + (i & 3)]: 0, x: 0 }, kx, i);",
    );
    let ([probes, hits], fills) = reflect_counts();
    assert_eq!(value(&mut engine, "rs(o, kx, 1)"), "true");
    let ([p1, h1], f1) = reflect_counts();
    assert_eq!(
        (p1 - probes, h1 - hits, f1 - fills),
        (1, 0, 1),
        "{tier:?}: a miss probes once and fills"
    );
    assert_eq!(value(&mut engine, "rs(o, kx, 2) + ':' + o.x"), "true:2");
    let ([p2, h2], f2) = reflect_counts();
    assert_eq!(
        (p2 - p1, h2 - h1, f2 - f1),
        (1, 1, 0),
        "{tier:?}: the table serves the next store"
    );
}

/// Reflect.set's fast path takes only what the full path resolves the same way. A receiver other
/// than the target (a plain object, or a Proxy of the target whose handler traps
/// `defineProperty`) gets OrdinarySet's steps on itself, though the table holds the target's
/// entry; a setter on the chain is called every time (no creation is filled under it); a frozen
/// target refuses the store. Nothing here moves the process-wide creation epoch, which other
/// tests' counts read.
#[test]
fn reflect_set_s_table_path_keeps_the_receiver_the_chain_and_the_attributes() {
    crate::opt::pin_tier_off_for_test();
    for tier in [crate::bytecode::Tier::Bytecode, crate::bytecode::Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, REFLECT_KEYS);
        eval(
            &mut engine,
            "function rs(o, k, v) { return Reflect.set(o, k, v); }
             function rs4(o, k, v, r) { return Reflect.set(o, k, v, r); }
             for (var i = 0; i < 64; i++) { var w = { ['w' + (i & 3)]: 0, x: 0 }; rs(w, kx, i); rs4(w, kx, i, w); }
             var o = { a: 1, x: 0 }; rs(o, kx, 1); rs(o, kx, 2); rs4(o, kx, 2, o);
             var r = { b: 1, x: 0 }; var defines = 0;
             var p = new Proxy(o, { defineProperty: function (t, k, d) { defines++; return Reflect.defineProperty(t, k, d); } });",
        );
        assert_eq!(
            value(&mut engine, "rs4(o, kx, 3, r) + ':' + o.x + ':' + r.x"),
            "true:2:3",
            "{tier:?}: a plain receiver's own property"
        );
        assert_eq!(
            value(&mut engine, "rs4(o, kx, 4, p) + ':' + o.x + ':' + defines"),
            "true:4:1",
            "{tier:?}: the receiver Proxy's defineProperty trap"
        );
        eval(
            &mut engine,
            "var P = { q: 1 }, seen = [];
             for (var i = 0; i < 3; i++) rs(Object.create(P), kc, i);",
        );
        let created = value(
            &mut engine,
            "var n = Object.create(P); rs(n, kc, 7) + ':' + Object.keys(n).join(',')",
        );
        assert_eq!(created, "true:c", "{tier:?}: a creation the table serves");
        assert_eq!(
            value(
                &mut engine,
                "var S = { set c(v) { seen.push(v); } };
                 for (var i = 0; i < 3; i++) rs(Object.create(S), kc, 8 + i);
                 var m = Object.create(S); rs(m, kc, 11) + ':' + Object.keys(m).length + ':' + seen.join(',')"
            ),
            "true:0:8,9,10,11",
            "{tier:?}: a setter on the chain is called, every time"
        );
        assert_eq!(
            value(&mut engine, "Object.freeze(o); rs(o, kx, 5) + ':' + o.x"),
            "false:4",
            "{tier:?}: a frozen target refuses"
        );
    }
}

/// The table's hit takes an entry for its own receiver shape and prototype alone. The slot mixes
/// in both (`store_slot`), but two triples can share one: each collision is made here by copying
/// one triple's real entry into the slot the other's probe reads. An update entry of another
/// shape (its `x` elsewhere) is not applied; a creation entry under another prototype is not
/// taken, and that prototype's setter for the key is called. Reflect.set's probe is the Rust
/// path's alone. In a process of its own (`run_alone`): each real entry is one a Reflect.set call
/// filled through the resolution, which a call staled by the call caches' epoch does not reach.
#[test]
fn reflect_set_s_table_hit_takes_its_own_shape_and_prototype_alone() {
    crate::opt::pin_tier_off_for_test();
    if !crate::jit::construct_tests::run_alone(
        "interpreter::stub_cache_tests::reflect_set_s_table_hit_takes_its_own_shape_and_prototype_alone",
    ) {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(crate::bytecode::Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, REFLECT_KEYS);
    eval(
        &mut engine,
        "function rs(o, k, v) { return Reflect.set(o, k, v); }
         for (var i = 0; i < 64; i++) rs({ ['w' + (i & 3)]: 0, x: 0 }, kx, i);
         var seen = [], A = { p: 0, x: 0 }, B = { x: 0, q: 0 };
         var Pa = { q: 1 }, Pb = { set c(v) { seen.push(v); } };
         var ea = Object.create(Pa), eb = Object.create(Pb);
         rs(A, kx, 1); rs(A, kx, 2); rs(Object.create(Pa), kc, 1); rs(Object.create(Pa), kc, 2);",
    );
    let triple = |engine: &mut Engine, name: &str| {
        let o = object(engine, name);
        let shape = o.borrow().props.shape();
        (shape, proto_of(&o))
    };
    let collide = |engine: &mut Engine, from: &str, to: &str, key: &str| {
        let key = crate::atom::PropKey::from_text(key);
        let ((fs, fp), (ts, tp)) = (triple(engine, from), triple(engine, to));
        let entry = engine
            .interp
            .store_cache_get(fs, key, fp)
            .expect("the first triple's entry");
        engine.interp.store_cache[store_slot(ts, key, tp)].set(entry);
    };
    collide(&mut engine, "A", "B", "x");
    assert_eq!(
        value(&mut engine, "rs(B, kx, 5) + ':' + B.x + ':' + B.q"),
        "true:5:0",
        "an update entry of another shape in the slot"
    );
    collide(&mut engine, "ea", "eb", "c");
    assert_eq!(
        value(
            &mut engine,
            "var m = Object.create(Pb); rs(m, kc, 7) + ':' + Object.keys(m).length + ':' + seen.join(',')"
        ),
        "true:0:7",
        "a creation entry under another prototype in the slot"
    );
}
