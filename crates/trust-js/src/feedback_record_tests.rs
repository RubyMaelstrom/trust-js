//! The always-on records (W4 steps 3-5; feedback-schema.txt section 5): the VM records every
//! execution; generated code records only what its in-line path does not take, on its miss
//! paths; equality records nothing. The value record (step 3), the element record (step 4), then
//! the named sites' generic mark (step 5).

use crate::bytecode::*;
use crate::feedback::{
    ValueClass as C, ELEM_ARGUMENTS, ELEM_ARRAY, ELEM_EXOTIC, ELEM_HOLEY, ELEM_KEY_INDEX,
    ELEM_KEY_NUMBER, ELEM_KEY_OTHER, ELEM_KEY_STRING, ELEM_KEY_SYMBOL, ELEM_NON_INT32,
    ELEM_NON_NUMBER, ELEM_ORDINARY, ELEM_OUT_OF_BOUNDS, ELEM_PRIMITIVE, ELEM_STRING,
    ELEM_TYPED_ARRAY, ELEM_UNWRITABLE, RECORD_OPERAND1_SHIFT, RECORD_RESULT_SHIFT,
};
use crate::value::{ObjectKind, Value, ValueView};
use crate::{Completion, Engine};

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("record fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// The global function `name`'s compiled chunk.
fn chunk_of(engine: &mut Engine, name: &str) -> std::rc::Rc<Chunk> {
    let env = engine.interp.global_env;
    let value: Value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let ValueView::Obj(function) = value.view() else {
        panic!("{name} is not an object")
    };
    let ObjectKind::User(user) = &function.borrow().kind else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    let chunk = runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap_or_else(|| panic!("{name} stayed interpreted"));
    std::rc::Rc::clone(chunk)
}

/// The record of `name`'s one operation that `which` accepts, as (operand 0, operand 1, result).
fn record(engine: &mut Engine, name: &str, which: impl Fn(&Op) -> bool) -> (u32, u32, u32) {
    let chunk = chunk_of(engine, name);
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| which(&chunk.jit_ops()[pc]))
        .collect();
    assert_eq!(
        pcs.len(),
        1,
        "one such operation in {name}: {:?}",
        chunk.jit_ops()
    );
    let word = chunk.feedback_record(pcs[0]);
    (
        word & 0x1ff,
        (word >> RECORD_OPERAND1_SHIFT) & 0x1ff,
        (word >> RECORD_RESULT_SHIFT) & 0x1ff,
    )
}

const INT32: u32 = C::NumberInt32.bit();
const DOUBLE: u32 = C::NumberDouble.bit();
const STRING: u32 = C::String.bit();

#[test]
fn the_vm_records_every_class_it_executes() {
    let mut engine = engine(Tier::Bytecode);
    eval(
        &mut engine,
        "function f(a, b) { return a + b; } f(1, 2); f(1.5, 2); f('a', 'b');",
    );
    let (left, right, result) = record(&mut engine, "f", |op| matches!(op, Op::Add));
    assert_eq!(left, INT32 | DOUBLE | STRING);
    assert_eq!(right, INT32 | STRING);
    assert_eq!(result, INT32 | DOUBLE | STRING);
}

#[test]
fn native_code_records_its_misses_and_not_its_in_line_numbers() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function f(a, b) { return a + b; } for (var i = 0; i < 20; i++) f(i, 0.5);",
    );
    assert_eq!(
        record(&mut engine, "f", |op| matches!(op, Op::Add)),
        (0, 0, 0),
        "Number + Number is Add's in-line set"
    );
    eval(&mut engine, "f('x', 'y');");
    assert_eq!(
        record(&mut engine, "f", |op| matches!(op, Op::Add)),
        (STRING, STRING, STRING)
    );
}

#[test]
fn equality_records_nothing() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        eval(
            &mut engine,
            "function g(a, b) { return a === b; } g('a', 'b'); g(1, {}); g('a', 'a' + '');",
        );
        assert_eq!(
            record(&mut engine, "g", |op| matches!(op, Op::StrictEq)),
            (0, 0, 0),
            "{tier:?}"
        );
    }
}

#[test]
fn relational_comparisons_record_their_operands() {
    let mut engine = engine(Tier::Bytecode);
    eval(
        &mut engine,
        "function h(a, b) { return a < b; } h('a', 'b'); h(1, 2.5);",
    );
    assert_eq!(
        record(&mut engine, "h", |op| matches!(op, Op::Lt)),
        (STRING | INT32, STRING | DOUBLE, 0)
    );
    let mut engine = self::engine(Tier::Jit);
    eval(
        &mut engine,
        "function h(a, b) { return a < b; } for (var i = 0; i < 20; i++) h(i, 2.5); h('a', 'b');",
    );
    assert_eq!(
        record(&mut engine, "h", |op| matches!(op, Op::Lt)),
        (STRING, STRING, 0)
    );
}

#[test]
fn the_remainder_records_what_leaves_its_int32_in_line_set() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function m(a, b) { return a % b; } for (var i = 0; i < 20; i++) m(i + 7, 3);",
    );
    assert_eq!(
        record(&mut engine, "m", |op| matches!(op, Op::Mod)),
        (0, 0, 0)
    );
    eval(&mut engine, "m(7.5, 2);");
    assert_eq!(
        record(&mut engine, "m", |op| matches!(op, Op::Mod)),
        (DOUBLE, INT32, DOUBLE)
    );
}

#[test]
fn an_update_records_a_non_number_operand_and_its_result() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function u(s) { s++; return s; } for (var i = 0; i < 20; i++) u(i);",
    );
    let which = |op: &Op| matches!(op, Op::UpdateLocal(..));
    assert_eq!(record(&mut engine, "u", which), (0, 0, 0));
    eval(&mut engine, "u('4');");
    assert_eq!(record(&mut engine, "u", which), (STRING, 0, INT32));
}

/// The element record of `name`'s one element operation that `which` accepts.
fn element(engine: &mut Engine, name: &str, which: impl Fn(&Op) -> bool) -> u32 {
    let chunk = chunk_of(engine, name);
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| which(&chunk.jit_ops()[pc]))
        .collect();
    assert_eq!(
        pcs.len(),
        1,
        "one such operation in {name}: {:?}",
        chunk.jit_ops()
    );
    chunk.feedback_record(pcs[0])
}

fn load(op: &Op) -> bool {
    matches!(op, Op::GetElem | Op::GetElemLocal(..) | Op::GetMethodElem)
}

fn store(op: &Op) -> bool {
    matches!(
        op,
        Op::SetElem | Op::SetElemDrop | Op::SetElemLocal(..) | Op::SetElemLocalDrop(..)
    )
}

/// A packed int32 Array that no literal shares (a constant literal's block is copy-on-write,
/// which the record reports as unwritable).
const PACKED: &str = "var p = []; for (var j = 0; j < 4; j++) p.push(j);";

#[test]
fn the_vm_records_every_element_receiver_and_key() {
    let mut engine = engine(Tier::Bytecode);
    eval(
        &mut engine,
        &format!(
            "{PACKED} function g(a, k) {{ return a[k]; }}
             function h() {{ return g(arguments, 0); }}
             g(p, 0); g(new Int8Array(2), 1); g({{a: 1}}, 'a'); h(5); g(new Proxy({{}}, {{}}), 'x');
             g('ab', 1); g(1, 'x'); g({{}}, 1.5); g({{}}, Symbol.iterator); g({{}}, null);"
        ),
    );
    assert_eq!(
        element(&mut engine, "g", load),
        ELEM_ARRAY
            | ELEM_TYPED_ARRAY
            | ELEM_ORDINARY
            | ELEM_ARGUMENTS
            | ELEM_EXOTIC
            | ELEM_STRING
            | ELEM_PRIMITIVE
            | ELEM_KEY_INDEX
            | ELEM_KEY_NUMBER
            | ELEM_KEY_STRING
            | ELEM_KEY_SYMBOL
            | ELEM_KEY_OTHER
    );
}

#[test]
fn the_vm_records_an_arrays_lost_kind_facts() {
    let mut engine = engine(Tier::Bytecode);
    eval(
        &mut engine,
        &format!(
            "{PACKED} function g(a, i) {{ return a[i]; }} g(p, 1);
             function f(a, i) {{ return a[i]; }} f([1.5, , 'x'], 7);
             function z(a, i) {{ return a[i]; }} z(Object.freeze(p), 0);
             function s(a, i, v) {{ a[i] = v; }} s(p.slice(), 4, 1);"
        ),
    );
    assert_eq!(element(&mut engine, "g", load), ELEM_ARRAY | ELEM_KEY_INDEX);
    assert_eq!(
        element(&mut engine, "f", load),
        ELEM_ARRAY
            | ELEM_HOLEY
            | ELEM_NON_NUMBER
            | ELEM_NON_INT32
            | ELEM_OUT_OF_BOUNDS
            | ELEM_KEY_INDEX
    );
    assert_eq!(
        element(&mut engine, "z", load),
        ELEM_ARRAY | ELEM_UNWRITABLE | ELEM_KEY_INDEX
    );
    assert_eq!(
        element(&mut engine, "s", store),
        ELEM_ARRAY | ELEM_OUT_OF_BOUNDS | ELEM_KEY_INDEX,
        "an append is an index past the used words"
    );
}

#[test]
fn native_element_reads_record_only_what_leaves_the_in_line_set() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        &format!(
            "{PACKED} var t = new Float64Array(4); var o = {{}}; o[0] = 1; o[1] = 2;
             function g(a, i) {{ return a[i]; }}
             for (var i = 0; i < 20; i++) {{ g(p, i % 4); g(t, i % 4); g(o, i % 2); }}"
        ),
    );
    assert_eq!(
        element(&mut engine, "g", load),
        0,
        "an Array's, a typed array's and an ordinary object's own elements at an index"
    );
    eval(&mut engine, "g(p, 'length');");
    assert_eq!(
        element(&mut engine, "g", load),
        ELEM_ARRAY | ELEM_KEY_STRING
    );
    eval(&mut engine, "g([1, , 3], 1); g(p, 10);");
    assert_eq!(
        element(&mut engine, "g", load),
        ELEM_ARRAY | ELEM_HOLEY | ELEM_OUT_OF_BOUNDS | ELEM_KEY_STRING | ELEM_KEY_INDEX
    );
    eval(&mut engine, "g('abc', 1);");
    assert_eq!(
        element(&mut engine, "g", load) & (ELEM_STRING | ELEM_KEY_INDEX),
        ELEM_STRING | ELEM_KEY_INDEX,
        "a String receiver is outside the in-line set"
    );
}

#[test]
fn native_element_stores_record_only_what_leaves_the_in_line_set() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        &format!(
            "{PACKED} var t = new Int32Array(4);
             function s(a, i, v) {{ a[i] = v; }}
             for (var i = 0; i < 20; i++) {{ s(p, i % 4, i); s(t, i % 4, i); }} s(p, 4, 0);"
        ),
    );
    assert_eq!(
        element(&mut engine, "s", store),
        0,
        "overwrites, an append and typed-array stores"
    );
    eval(
        &mut engine,
        "s(p, 'x', 1); s(Object.freeze([1, 2].slice()), 0, 3);",
    );
    assert_eq!(
        element(&mut engine, "s", store),
        ELEM_ARRAY | ELEM_UNWRITABLE | ELEM_KEY_STRING | ELEM_KEY_INDEX
    );
}

/// Whether `name`'s one named operation that `which` accepts carries the generic mark.
fn generic(engine: &mut Engine, name: &str, which: impl Fn(&Op) -> bool) -> bool {
    let chunk = chunk_of(engine, name);
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| which(&chunk.jit_ops()[pc]))
        .collect();
    assert_eq!(
        pcs.len(),
        1,
        "one such operation in {name}: {:?}",
        chunk.jit_ops()
    );
    chunk.named_site_generic(pcs[0])
}

fn named_load(op: &Op) -> bool {
    matches!(op, Op::GetProp(..) | Op::GetPropLocal(..))
}

#[test]
fn a_fifth_receiver_shape_marks_a_named_site_generic() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        eval(
            &mut engine,
            "function g(o) { return o.x; }
             var shapes = [{x: 1}, {x: 1, a: 1}, {x: 1, b: 1}, {x: 1, c: 1}];
             for (var i = 0; i < 20; i++) g(shapes[i % 4]);",
        );
        assert!(
            !generic(&mut engine, "g", named_load),
            "{tier:?}: four shapes fit the ways"
        );
        eval(&mut engine, "g({x: 1, d: 1});");
        assert!(
            generic(&mut engine, "g", named_load),
            "{tier:?}: a fifth evicts a live way"
        );
        eval(
            &mut engine,
            "for (var i = 0; i < 20; i++) g(shapes[i % 4]);",
        );
        assert!(
            generic(&mut engine, "g", named_load),
            "{tier:?}: the mark stays"
        );
    }
}

#[test]
fn a_refused_fill_marks_a_named_site_generic() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function g(o) { return o.x; } function m(o) { return o.x; }
         function s(a) { a.foo = 1; }
         for (var i = 0; i < 20; i++) { g({x: i}); m({x: i}); }",
    );
    assert!(!generic(&mut engine, "g", named_load));
    eval(&mut engine, "g(new Proxy({x: 2}, {})); s([]);");
    assert!(generic(&mut engine, "g", named_load), "a proxy receiver");
    assert!(!generic(&mut engine, "m", named_load), "a monomorphic site");
    assert!(
        generic(&mut engine, "s", |op| matches!(
            op,
            Op::SetProp(..) | Op::SetPropDrop(..) | Op::SetPropLocalDrop(..)
        )),
        "an Array receiver of a store"
    );
}

#[test]
fn a_generic_sites_prototype_ways_still_hit() {
    // Two and three hops to the method: the templates test the intermediate shapes' bits of
    // `mid_ok` alone, beside the generic mark.
    let mut engine = engine(Tier::Jit);
    let total = eval(
        &mut engine,
        "class A { m() { return 1; } } class B extends A {} class C extends B {}
         function c(o) { return o.m(); }
         var others = [{m() { return 2; }}, {a: 1, m() { return 2; }}, {b: 1, m() { return 2; }},
                       {c: 1, m() { return 2; }}, {d: 1, m() { return 2; }}];
         var total = 0;
         for (var i = 0; i < 5; i++) total += c(others[i]);
         for (var i = 0; i < 40; i++) total += c(i % 2 ? new B() : new C());
         total",
    );
    assert_eq!(total, "50");
    assert!(generic(&mut engine, "c", |op| matches!(
        op,
        Op::GetMethod(..)
    )));
}
