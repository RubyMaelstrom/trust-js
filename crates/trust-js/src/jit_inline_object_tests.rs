//! W2e step 3: object literals made in line by generated code (`alloc::emit_inline_object`). Each
//! in-line object must be the object the helper makes, field for field, keep the heap's live
//! count and byte charge as the helper's allocation keeps them, and survive the collections its
//! allocations make due (the in-line path declines while one is due, and its helper polls).
use super::*;
use crate::bytecode::Tier;
use crate::value::{Gc, Value};
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

fn global(engine: &mut Engine, name: &str) -> Gc {
    let global = Value::Obj(engine.interp.global);
    engine
        .interp
        .get_member_key(&global, name)
        .unwrap_or_else(|_| panic!("{name}"))
        .as_obj()
        .unwrap_or_else(|| panic!("{name} is an object"))
}

fn in_line() -> usize {
    alloc::INLINE_OBJECTS.with(|count| count.get())
}

/// What a reader of an object's header sees: class, shape, flag bits and element words.
fn header(object: &Gc) -> (u32, u32, bool, bool, usize) {
    let body = object.borrow();
    (
        body.props.class(),
        body.props.shape(),
        body.extensible(),
        body.ic_plain(),
        body.props.words().map_or(0, |words| words.len()),
    )
}

/// An in-line object reads as the object the helper made at the same site (its first execution,
/// which makes the template and the image), in every capacity class: the same header, prototype,
/// keys, attributes and values.
#[test]
fn in_line_objects_are_the_helpers_objects() {
    let literals = [
        "{a: v}",
        "{a: v, b: 'two', c: null}",
        "{a: v, b: v, c: v, d: v, e: v, f: v, g: v}",
        "{k0: v, k1: v, k2: v, k3: v, k4: v, k5: v, k6: v, k7: v, k8: v, k9: v, k10: v, k11: v, \
         k12: v, k13: v, k14: v}",
    ];
    for literal in literals {
        let mut engine = engine(Tier::Jit);
        let before = in_line();
        let source = format!(
            "function make(v) {{ return {literal}; }}
             var first = make(7);
             for (var i = 0; i < 20; i++) make(i);
             var later = make(7);
             var d = function (o) {{
                 return JSON.stringify(Object.getOwnPropertyDescriptors(o)) +
                     (Object.getPrototypeOf(o) === Object.prototype) + Object.isExtensible(o);
             }};
             d(first) === d(later) ? 'same' : d(first) + ' / ' + d(later)"
        );
        assert_eq!(eval(&mut engine, &source), "same", "{literal}");
        assert!(in_line() > before, "{literal}: no in-line object");
        let (first, later) = (global(&mut engine, "first"), global(&mut engine, "later"));
        assert_eq!(header(&first), header(&later), "{literal}");
        assert!(Gc::ptr_eq(
            later.borrow().proto.as_ref().unwrap(),
            &engine.interp.object_proto
        ));
    }
}

/// Objects made in line keep their values and properties through the collections their own
/// allocations make due, and the heap's live count matches its live cells after a collection.
#[test]
fn in_line_objects_survive_collections_and_keep_the_live_count() {
    let mut engine = engine(Tier::Jit);
    let before = in_line();
    let result = eval(
        &mut engine,
        "function make(i) { return {x: i, y: 'v' + (i % 7), z: {w: i}}; }
         var keep = [];
         for (var i = 0; i < 200000; i++) {
             var o = make(i);
             if (i % 1000 === 0) keep.push(o);
         }
         var sum = 0;
         for (var k = 0; k < keep.length; k++) sum += keep[k].x + keep[k].z.w + keep[k].y.length;
         sum + ':' + keep.length + ':' + keep[199].y",
    );
    assert_eq!(result, "39800400:200:v4");
    assert!(in_line() - before > 100_000, "the loop allocated in line");
    engine.interp.gc_collect();
    let heap = crate::value::active_gc_heap();
    let counted = crate::value::live_objects();
    let cells = heap.core.borrow().live_count(heap.object_desc) as i64;
    assert_eq!(counted, cells, "the live count is the live cells'");
}

/// A literal whose template keeps a record, or whose values do not fit a cell, and the first
/// executions before the template exists, take the helper: the results are the same.
#[test]
fn literals_past_the_cell_take_the_helper() {
    let mut engine = engine(Tier::Jit);
    let keys: Vec<String> = (0..20).map(|k| format!("k{k}: v")).collect();
    let source = format!(
        "function big(v) {{ return {{{}}}; }} var o = big(1); var p = big(2);
         Object.keys(p).length + ':' + p.k19",
        keys.join(", ")
    );
    assert_eq!(eval(&mut engine, &source), "20:2");
}

/// `{}`, `[]` and a constant literal's copy (W2e step 4) made in line read as the helper's, and
/// behave as theirs under later writes: properties added to `{}`, elements pushed onto `[]`, and
/// a write to a shared literal's copy, which copies its block first (W2d step 4).
#[test]
fn empty_literals_and_shared_copies_are_the_helpers() {
    let cases = [
        ("{}", "o.a = 1; o.b = 2; Object.keys(o).join()", "a,b"),
        (
            "[]",
            "o.push(1, 2, 3); o.length + ':' + o.join()",
            "3:1,2,3",
        ),
        (
            "[1, 'two', 3]",
            "o[0] = 9; String(o) + ':' + String(make())",
            "9,two,3:1,two,3",
        ),
    ];
    for (literal, after, expected) in cases {
        let mut engine = engine(Tier::Jit);
        let before = in_line();
        let source = format!(
            "function make() {{ return {literal}; }}
             var first = make();
             for (var i = 0; i < 20; i++) make();
             var later = make();
             var d = function (o) {{
                 return JSON.stringify(Object.getOwnPropertyDescriptors(o)) +
                     Object.getPrototypeOf(o) + Array.isArray(o) + Object.isExtensible(o);
             }};
             d(first) === d(later) ? 'same' : d(first) + ' / ' + d(later)"
        );
        assert_eq!(eval(&mut engine, &source), "same", "{literal}");
        assert!(in_line() > before, "{literal}: no in-line object");
        let (first, later) = (global(&mut engine, "first"), global(&mut engine, "later"));
        assert_eq!(header(&first), header(&later), "{literal}");
        let result = eval(&mut engine, &format!("var o = later; {after}"));
        assert_eq!(result, expected, "{literal}");
    }
}
