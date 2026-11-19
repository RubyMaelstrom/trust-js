//! An Array's absent named read cached (P3 stage 3, the megamorphic line's piece 2): the fill
//! (`Interp::ic_get_rederive`) may cross an Array level for a name that cannot be an index,
//! because an Array's elements, its only entries outside its shape, all have index keys, so its
//! shape proves such a name absent as an ordinary object's does. The probe (`ic_shape_probe`)
//! re-walks every level and compares its shape, so a later addition anywhere on the chain, an
//! accessor, a deletion, a subclass level and a swapped prototype are all seen. An index-like
//! name stays out: an element write keeps the array's shape.
use super::*;
use crate::bytecode::{IcState, IC_ABSENT, IC_MID_GENERIC, PROP_IC_WAYS};
use crate::value::ValueView;
use crate::{Completion, Engine};
use std::cell::Cell;

type Ways = [Cell<IcState>; PROP_IC_WAYS];

fn eval(engine: &mut Engine, source: &str) {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(_) => {}
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn global(engine: &mut Engine, name: &str) -> Value {
    let global = Value::Obj(engine.interp.global);
    engine
        .interp
        .get_member_key(&global, name)
        .unwrap_or_else(|_| panic!("global {name}"))
}

/// `receiver[name]` of a global through the named-read cache `ways`, as a named read's helper
/// reads it, shown as its number or `undefined`.
fn read(engine: &mut Engine, receiver: &str, name: &str, ways: &Ways) -> String {
    // The receiver stays reachable from its global binding while the read runs.
    let base = global(engine, receiver);
    let value = engine
        .interp
        .get_prop_ic(&base, name, crate::atom::PropKey::from_text(name), &ways[0])
        .unwrap_or_else(|_| panic!("{receiver}.{name} threw"));
    match value.view() {
        ValueView::Undefined => "undefined".to_string(),
        ValueView::Num(n) => n.to_string(),
        _ => "other".to_string(),
    }
}

fn empty() -> Ways {
    [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS]
}

/// The level count of way 0's cached absence, or None when way 0 holds none.
fn absent_levels(ways: &Ways) -> Option<u32> {
    let way = ways[0].get();
    (way.depth == IC_ABSENT).then_some(way.slot)
}

#[test]
fn an_array_s_absent_name_is_cached_and_every_later_change_is_seen() {
    let mut engine = Engine::new();
    eval(
        &mut engine,
        "var arr = [1, 2, 3]; var other = [4]; var calls = 0;
         class Sub extends Array {} var sub = new Sub(); sub.push(5);",
    );
    let name = "__v_isRef";
    let ways = empty();
    assert_eq!(read(&mut engine, "arr", name, &ways), "undefined");
    // arr, Array.prototype (an Array itself) and Object.prototype.
    assert_eq!(
        absent_levels(&ways),
        Some(3),
        "the fill crosses both Array levels"
    );
    // The probe takes what the fill made: a second read hits way 0 and demotes nothing.
    assert_eq!(read(&mut engine, "arr", name, &ways), "undefined");
    assert_eq!(
        ways[1].get().depth,
        crate::bytecode::IC_EMPTY,
        "the second read hit"
    );
    // Element and `length` writes keep an array's shape: the way still proves the absence.
    eval(&mut engine, "arr.push(4); arr.length = 2; other[1] = 9;");
    for receiver in ["arr", "other", "arr"] {
        assert_eq!(read(&mut engine, receiver, name, &ways), "undefined");
    }
    assert_eq!(absent_levels(&ways), Some(3));
    let steps = [
        // An addition on each level is seen, and a deletion caches the absence again.
        ("Array.prototype.__v_isRef = 1;", "arr", "1"),
        ("delete Array.prototype.__v_isRef;", "arr", "undefined"),
        ("Object.prototype.__v_isRef = 2;", "arr", "2"),
        ("delete Object.prototype.__v_isRef;", "arr", "undefined"),
        ("arr.__v_isRef = 3;", "arr", "3"),
        ("", "other", "undefined"),
        ("delete arr.__v_isRef;", "arr", "undefined"),
        // An accessor on Array.prototype runs once a read.
        (
            "Object.defineProperty(Array.prototype, '__v_isRef', { configurable: true,
                 get: function () { calls++; return 4; } });",
            "arr",
            "4",
        ),
        ("", "calls", "1"),
        ("delete Array.prototype.__v_isRef;", "arr", "undefined"),
        // A prototype swapped under the array: a level that was not there, then the end moved.
        ("Object.setPrototypeOf(arr, { __v_isRef: 8 });", "arr", "8"),
        ("Object.setPrototypeOf(arr, null);", "arr", "undefined"),
        (
            "Object.setPrototypeOf(arr, Array.prototype);",
            "arr",
            "undefined",
        ),
    ];
    for (k, (source, receiver, expected)) in steps.into_iter().enumerate() {
        eval(&mut engine, source);
        let value = if receiver == "calls" {
            match global(&mut engine, "calls").view() {
                ValueView::Num(n) => n.to_string(),
                _ => "other".to_string(),
            }
        } else {
            read(&mut engine, receiver, name, &ways)
        };
        assert_eq!(value, expected, "after `{source}`: {receiver}.{name}");
        if k == 9 {
            // A deletion takes a fresh shape: the read cached the absence again.
            assert_eq!(
                absent_levels(&ways),
                Some(3),
                "cached again after a deletion"
            );
        }
    }
    // A subclass of Array: sub, Sub.prototype, Array.prototype, Object.prototype.
    let ways = empty();
    assert_eq!(read(&mut engine, "sub", name, &ways), "undefined");
    assert_eq!(absent_levels(&ways), Some(4));
    eval(&mut engine, "Sub.prototype.__v_isRef = 6;");
    assert_eq!(read(&mut engine, "sub", name, &ways), "6");
    // A Proxy level is no plain level: the read takes the full [[Get]] and its trap runs.
    eval(
        &mut engine,
        "var px = [1]; Object.setPrototypeOf(px, new Proxy([], { get: function (t, k, r) {
             return k === '__v_isRef' ? 7 : Reflect.get(t, k, r); } }));",
    );
    let ways = empty();
    for _ in 0..2 {
        assert_eq!(read(&mut engine, "px", name, &ways), "7");
    }
    assert_eq!(absent_levels(&ways), None);
}

#[test]
fn an_index_like_name_never_caches_its_absence_across_an_array() {
    let mut engine = Engine::new();
    eval(&mut engine, "var arr = [1, 2];");
    // "2" names an element, and an element write keeps the array's shape.
    let ways = empty();
    assert_eq!(read(&mut engine, "arr", "2", &ways), "undefined");
    assert_eq!(absent_levels(&ways), None);
    eval(&mut engine, "arr.push(9);");
    assert_eq!(read(&mut engine, "arr", "2", &ways), "9");
    // A digit-led name that is no index ("01", "1e3") is kept out as the templates' `arr_ok`
    // gate keeps it, so the fill and the probe agree: no way, and no refill that would mark the
    // site generic.
    for name in ["01", "1e3"] {
        let ways = empty();
        for _ in 0..PROP_IC_WAYS + 2 {
            assert_eq!(read(&mut engine, "arr", name, &ways), "undefined", "{name}");
        }
        assert_eq!(absent_levels(&ways), None, "{name}");
        assert_eq!(
            ways[0].get().mid_ok & IC_MID_GENERIC,
            0,
            "{name}: not generic"
        );
        eval(&mut engine, &format!("arr[{name:?}] = 3;"));
        assert_eq!(read(&mut engine, "arr", name, &ways), "3", "{name}");
    }
    // "-0" is a name (ToString(-0) is "0", so `arr[-0]` writes the element 0).
    let ways = empty();
    assert_eq!(read(&mut engine, "arr", "-0", &ways), "undefined");
    assert_eq!(absent_levels(&ways), Some(3));
    eval(&mut engine, "arr[-0] = 5;");
    assert_eq!(read(&mut engine, "arr", "-0", &ways), "undefined");
    eval(&mut engine, "arr['-0'] = 4;");
    assert_eq!(read(&mut engine, "arr", "-0", &ways), "4");
}
