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
