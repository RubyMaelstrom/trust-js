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
