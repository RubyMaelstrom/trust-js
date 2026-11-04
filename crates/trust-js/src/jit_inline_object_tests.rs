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

/// Whether generated code made objects in line since a counter read `before`: so under GC
/// stress too, which turns the in-line path off (the heap counts each allocation), and the
/// tests then check the helpers' objects alone.
fn grew(now: usize, before: usize) -> bool {
    crate::gc_test::stress_active() || now > before
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
        assert!(grew(in_line(), before), "{literal}: no in-line object");
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
    assert!(
        crate::gc_test::stress_active() || in_line() - before > 100_000,
        "the loop allocated in line"
    );
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
        assert!(grew(in_line(), before), "{literal}: no in-line object");
        let (first, later) = (global(&mut engine, "first"), global(&mut engine, "later"));
        assert_eq!(header(&first), header(&later), "{literal}");
        let result = eval(&mut engine, &format!("var o = later; {after}"));
        assert_eq!(result, expected, "{literal}");
    }
}

fn in_line_constructs() -> usize {
    alloc::INLINE_CONSTRUCTS.with(|count| count.get())
}

/// The `this` of a direct construction made in line (W2e step 6) is the object the helper made
/// before the constructor's learning ended: the same header, prototype and properties; a new
/// `prototype` reaches the next instances; instances survive the collections their allocations
/// make due.
#[test]
fn in_line_this_objects_are_the_helpers() {
    let mut engine = engine(Tier::Jit);
    let before = in_line_constructs();
    let result = eval(
        &mut engine,
        "function P(x, y) { this.x = x; this.y = y; }
         P.prototype.sum = function () { return this.x + this.y; };
         function make(i) { return new P(i, 2 * i); }
         var first = make(1);
         var keep = [];
         for (var i = 0; i < 100000; i++) { var p = make(i); if (i % 1000 === 0) keep.push(p); }
         var later = make(1);
         var d = function (o) {
             return JSON.stringify(Object.getOwnPropertyDescriptors(o)) +
                 (Object.getPrototypeOf(o) === P.prototype) + o.sum() + Object.isExtensible(o);
         };
         var total = 0;
         for (var k = 0; k < keep.length; k++) total += keep[k].sum();
         (d(first) === d(later) ? 'same' : d(first) + ' / ' + d(later)) + ':' + total",
    );
    assert_eq!(result, "same:14850000");
    assert!(grew(in_line_constructs(), before), "no in-line this");
    let (first, later) = (global(&mut engine, "first"), global(&mut engine, "later"));
    assert_eq!(header(&first), header(&later));
    // A new `prototype`: the site's cached shape no longer matches, and the next instance takes
    // it (through the helper, which refills the site).
    let result = eval(
        &mut engine,
        "P.prototype = { sum: function () { return -1; } };
         var after = make(3);
         after.sum() + ':' + (Object.getPrototypeOf(after) === P.prototype)",
    );
    assert_eq!(result, "-1:true");
    engine.interp.gc_collect();
    let heap = crate::value::active_gc_heap();
    assert_eq!(
        crate::value::live_objects(),
        heap.core.borrow().live_count(heap.object_desc) as i64,
        "the live count is the live cells'"
    );
}

/// In-line `this` objects live in the running Agent's heap, whichever heap was active when the
/// code was assembled: the direct construct sequence is a process-wide shared stub, which once
/// built in the heap it was first assembled under and handed later engines that heap's cells
/// (a dropped heap's: a crash). Two engines on one thread, each alive while the other runs.
#[test]
fn in_line_this_objects_live_in_the_running_agents_heap() {
    let source = "function P(x, y) { this.x = x; this.y = y; }
         function make(i) { return new P(i, 2 * i); }
         var made = [];
         for (var i = 0; i < 2000; i++) { var p = make(i); if (i % 50 === 0) made.push(p); }
         made.length";
    let mut engines = [engine(Tier::Jit), engine(Tier::Jit)];
    for round in 0..2 {
        for (index, engine) in engines.iter_mut().enumerate() {
            let before = in_line_constructs();
            let _active = engine.interp.enter_agent();
            assert_eq!(eval(engine, source), "40", "engine {index}, round {round}");
            assert!(
                grew(in_line_constructs(), before),
                "engine {index}: no in-line this"
            );
            let made = global(engine, "made");
            let heap = engine.interp.gc_heap.clone();
            for k in 0..40 {
                let p = engine
                    .interp
                    .get_member_key(&Value::Obj(made), &k.to_string())
                    .ok()
                    .and_then(|value| value.as_obj())
                    .expect("an instance");
                assert!(
                    heap.core
                        .borrow()
                        .owns_cell(Gc::as_ptr(&p) as usize, heap.object_desc),
                    "engine {index}, round {round}: instance {k} lives in another heap"
                );
            }
            engine.interp.gc_collect();
            assert_eq!(
                crate::value::heap_live_objects(&heap),
                heap.core.borrow().live_count(heap.object_desc) as i64,
                "engine {index}: the live count is the live cells'"
            );
        }
    }
}

/// A constructor whose instances reserve a record (more fields than their cell holds) keeps the
/// helper, and a deferred `prototype` is made by the helper's first construction.
#[test]
fn this_objects_past_their_cell_take_the_helper() {
    let mut engine = engine(Tier::Jit);
    let fields: Vec<String> = (0..20).map(|k| format!("this.f{k} = v;")).collect();
    let source = format!(
        "function Big(v) {{ {} }} function make(i) {{ return new Big(i); }}
         for (var i = 0; i < 50; i++) make(i);
         var b = make(7); Object.keys(b).length + ':' + b.f19 + ':' + (b instanceof Big)",
        fields.join(" ")
    );
    assert_eq!(eval(&mut engine, &source), "20:7:true");
}

/// The element words' used length, capacity and kind bits of `name`'s Array.
fn words_of(engine: &mut Engine, name: &str) -> (usize, usize, u32) {
    let handle = global(engine, name);
    let body = handle.borrow();
    let words = body.props.words().expect("element words");
    (words.len(), words.capacity(), words.kind())
}

/// Array literals with elements made in line (W2e step 5) are the helper's Arrays: the same
/// header, an exact block with the same kind bits for every kind of value, and the same behavior
/// under later writes, growth and collections.
#[test]
fn in_line_arrays_with_elements_are_the_helpers() {
    let cases = [
        "[v, v]",
        "[v, v + 0.5, v]",
        "[v, -0, NaN]",
        "[v, 'two', null, undefined]",
        "[{}, [], v]",
        "[v, v, v, v, v, v, v, v]",
    ];
    for literal in cases {
        let mut engine = engine(Tier::Jit);
        let before = in_line();
        let source = format!(
            "function make(v) {{ return {literal}; }}
             var first = make(1);
             for (var i = 0; i < 20; i++) make(i);
             var later = make(1);
             var d = function (o) {{
                 return JSON.stringify(o) + Object.keys(o).length + o.length +
                     Object.is(o[1], -0) + (Object.getPrototypeOf(o) === Array.prototype);
             }};
             d(first) === d(later) ? 'same' : d(first) + ' / ' + d(later)"
        );
        assert_eq!(eval(&mut engine, &source), "same", "{literal}");
        assert!(grew(in_line(), before), "{literal}: no in-line array");
        let (first, later) = (global(&mut engine, "first"), global(&mut engine, "later"));
        assert_eq!(header(&first), header(&later), "{literal}");
        assert_eq!(
            words_of(&mut engine, "first"),
            words_of(&mut engine, "later"),
            "{literal}: the element block"
        );
        let result = eval(
            &mut engine,
            "later.push('x'); later[0] = 5; String(later.length) + ':' + later[0] + ':' +
             later[later.length - 1]",
        );
        assert_eq!(result.split(':').nth(1), Some("5"), "{literal}: {result}");
    }
    let mut engine = engine(Tier::Jit);
    let result = eval(
        &mut engine,
        "function pair(a, b) { return [a, b]; }
         var keep = [];
         for (var i = 0; i < 100000; i++) { var p = pair(i, 'n' + i); if (i % 1000 === 0) keep.push(p); }
         var sum = 0;
         for (var k = 0; k < keep.length; k++) sum += keep[k][0] + keep[k][1].length;
         sum + ':' + keep[99][1]",
    );
    assert_eq!(result, "4950587:n99000");
}
