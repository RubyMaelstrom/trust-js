//! The always-on value record (W4 step 3; feedback-schema.txt section 5): the VM records every
//! execution; generated code records only what its in-line path does not take, on its miss
//! paths; equality records nothing.

use crate::bytecode::*;
use crate::feedback::{ValueClass as C, RECORD_OPERAND1_SHIFT, RECORD_RESULT_SHIFT};
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
