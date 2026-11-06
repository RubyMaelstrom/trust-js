//! A native's fast entry from generated code (candidate 2 (a), the light classes): the call-hit
//! form's shared stub (`emit_call_hit_stub`) runs a native of a light class on light operands
//! through `bytecode::jit_native_light`, and leaves everything else to the call-hit helper.

use crate::bytecode::{
    native_operands, native_stub_kind, NativeOperands, Tier, INTRINSIC_LIGHT,
    INTRINSIC_LIGHT_PRIMITIVE_ARGS, NATIVE_STUB_CONSTRUCT_STATE, NATIVE_STUB_LIGHT,
    NATIVE_STUB_TEST_NONE, NATIVE_STUB_TEST_NUMBERS, NATIVE_STUB_TEST_PRIMITIVE_ARGS,
    NATIVE_STUB_TEST_PRIMITIVE_OPERANDS, TEST_NATIVE_STUB_CALLS,
};
use crate::value::Value;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// `source`'s completion in the tree-walker and in machine code, which must agree, and the
/// natives the stub ran for the machine-code run's last `rounds` evaluations of `call`.
fn stub_calls(source: &str, call: &str, rounds: usize) -> (String, usize) {
    let mut interp = Engine::new();
    interp.set_tier(Tier::Interp);
    eval(&mut interp, source);
    let expected = eval(&mut interp, call);
    let mut jit = Engine::new();
    jit.set_tier(Tier::Jit);
    jit.set_tier_threshold(0);
    eval(&mut jit, source);
    // Warm up: compile, fill the call caches, settle the inline checkpoints.
    for _ in 0..8 {
        assert_eq!(eval(&mut jit, call), expected, "{call}");
    }
    let before = TEST_NATIVE_STUB_CALLS.with(std::cell::Cell::get);
    for _ in 0..rounds {
        assert_eq!(eval(&mut jit, call), expected, "{call}");
    }
    (
        expected,
        TEST_NATIVE_STUB_CALLS.with(std::cell::Cell::get) - before,
    )
}

const LIGHT: &str = r#"
var m = new Map([['k', 3]]), wm = new WeakMap(), key = {};
wm.set(key, 4);
var s = 'the quick brown fox';
function operandOnly(n) { var t = 0; for (var i = 0; i < n; i++) t += m.get('k') + wm.get(key); return t; }
function light(n) { var t = 0; for (var i = 0; i < n; i++) t += Number.isInteger(i / 2) ? 1 : 0; return t; }
function primitiveArgs(n) { var t = 0; for (var i = 0; i < n; i++) t += parseInt('12', 10) + Number('3'); return t; }
function primitiveOperands(n) {
    var t = 0;
    for (var i = 0; i < n; i++) t += (s.includes('fox') ? 1 : 0) + s.indexOf('q') + s.substring(4, 9).length;
    return t;
}
function numbers(n) { var t = 0; for (var i = 0; i < n; i++) t += Math.round(i / 3) + Math.sign(i - 25); return t; }
"#;

/// Each light class's natives run through the stub, at sites in loops (the in-line form) and
/// outside them (the compact form): nearly every call (a call that meets a recompile's first entry
/// takes the helper).
#[test]
fn light_natives_take_their_fast_entry() {
    for (call, per_round) in [
        ("operandOnly(50)", 100),
        ("light(50)", 50),
        ("primitiveArgs(50)", 100),
        ("primitiveOperands(50)", 150),
        ("numbers(50)", 100),
    ] {
        let (_, served) = stub_calls(LIGHT, call, 3);
        assert!(
            served * 10 >= 3 * per_round * 9,
            "{call}: {served} natives through the stub, about {} expected",
            3 * per_round
        );
    }
    // Outside a loop: a function that makes one call, called many times.
    let (_, served) = stub_calls(
        "var s = 'abc'; function once(t) { return s.indexOf(t); } \
         function drive(n) { var r = 0; for (var i = 0; i < n; i++) r += once('c'); return r; }",
        "drive(40)",
        3,
    );
    assert!(served >= 108, "{served} natives through the stub");
}

/// Operands their class does not prove light take the call-hit helper, which converts them in
/// order (an Object's toString runs; a Symbol's conversion throws).
#[test]
fn non_light_operands_take_the_call_hit_helper() {
    let source = r#"
var s = 'abcdef', log = [];
var probe = { toString() { log.push('toString'); return 'cd'; } };
function objects(n) { var t = 0; for (var i = 0; i < n; i++) t += s.includes(probe) ? 1 : 0; return t + ':' + log.length; }
function strings(n) { var t = 0; for (var i = 0; i < n; i++) t += Math.round('2.5'); return t; }
"#;
    let (result, served) = stub_calls(source, "log.length = 0, objects(20)", 2);
    assert_eq!(result, "20:20");
    assert_eq!(served, 0, "an Object argument takes the helper");
    let (result, served) = stub_calls(source, "strings(20)", 2);
    assert_eq!(result, "60");
    assert_eq!(
        served, 0,
        "a String argument to a numeric native takes the helper"
    );
}

/// A light native's error unwinds through the site's handler, with the call's position: the
/// error's stack is the tree-walker's.
#[test]
fn a_light_natives_error_unwinds_with_its_call_position() {
    let source = r#"
function bad(n) {
    var caught = 0, stack = '';
    for (var i = 0; i < n; i++) {
        try { 'ab'.repeat(-1); } catch (e) { caught++; stack = e.stack; }
    }
    return caught + '|' + /RangeError/.test(stack) + '|' + /bad/.test(stack) + '|' + /:5:/.test(stack);
}
"#;
    let (result, served) = stub_calls(source, "bad(10)", 2);
    assert_eq!(result, "10|true|true|true");
    assert!(served >= 18, "{served} natives through the stub");
}

/// A native called while its caller constructs runs with a call's construct state: the stub
/// leaves such a call to the helper, which clears it (Number and Boolean return primitives).
#[test]
fn a_constructing_caller_takes_the_call_hit_helper() {
    let source = r#"
function Box(v) { this.n = Number(v); this.b = Boolean(v); this.k = typeof this.n + typeof this.b; }
function make(n) { var t = ''; for (var i = 0; i < n; i++) t = new Box('5').k; return t; }
"#;
    let (result, _) = stub_calls(source, "make(20)", 2);
    assert_eq!(result, "numberboolean");
}

/// A constructing caller's natives of the classes that hold no constructor run through the stub
/// (they read no construct state); the constructors' classes take the helper, and every result is
/// the tree-walker's. The fills check that no constructor is in a class the stub runs without its
/// construct state (`bytecode::native_stub_kind`).
#[test]
fn a_constructing_callers_other_light_natives_take_their_fast_entry() {
    let source = r#"
var m = new Map([['k', 3]]), s = ' the quick fox ';
function Box(v) {
    this.r = Math.round(v) + Math.max(v, 1) + Math.sqrt(4) + Math.sign(v) + Math.hypot(v, 1);
    this.p = parseInt('12') + parseFloat('1.5') + isFinite(v) + isNaN(v);
    this.c = Number(v) + String(v) + Boolean(v) + String.fromCharCode(65) + String.fromCodePoint(66);
    this.o = Array.isArray(v) + ':' + Object.is(v, v) + ':' + m.get('k') + ':' + Number.isInteger(v);
    this.s = s.indexOf('q') + s.includes('fox') + s.substring(1, 3) + s.charAt(2) + s.charCodeAt(1)
        + s.codePointAt(0) + s.slice(2, 4) + s.trim() + s.toUpperCase();
}
function make(n) {
    var t = '';
    for (var i = 0; i < n; i++) { var b = new Box(2.5); t = [b.r, b.p, b.c, b.o, b.s].join('|'); }
    return t;
}
"#;
    // Math.round, Math.sign, m.get, indexOf, includes, substring, trim and toUpperCase a
    // construction at least: no template finishes them in line, and their classes hold no
    // constructor (parseInt's, Number.isInteger's and the others take the helper here).
    let (_, served) = stub_calls(source, "make(20)", 2);
    assert!(
        served >= 8 * 20 * 2 * 9 / 10,
        "{served} natives through the stub"
    );
}

/// The stub kinds agree with `native_operands`: a kind's class is light on the operands its test
/// passes and needs the boundary on those it fails; every class that is light for some operands
/// has a kind, except DataView's accessors, which the helper serves; and the construct state is
/// tested for the classes that hold constructors.
#[test]
fn stub_kinds_agree_with_native_operands() {
    let engine = Engine::new();
    let object = Value::Obj(engine.interp.new_object());
    let number = Value::Num(1.5);
    let string = Value::lstr("s");
    let undefined = Value::Undefined;
    let classify = |class: u8, this: &Value, args: &[Value]| unsafe {
        native_operands(class, this, args.as_ptr(), args.len())
    };
    for class in 0..=u8::MAX {
        let kind = native_stub_kind(class);
        if kind == 0 {
            if ![
                crate::bytecode::INTRINSIC_DATAVIEW_GET,
                crate::bytecode::INTRINSIC_DATAVIEW_SET,
            ]
            .contains(&class)
            {
                assert_ne!(
                    classify(class, &string, &[number, string]),
                    NativeOperands::Light,
                    "class {class} is light for primitive operands but has no stub kind"
                );
            }
            continue;
        }
        assert_eq!(kind & NATIVE_STUB_LIGHT, NATIVE_STUB_LIGHT);
        assert_eq!(
            kind & NATIVE_STUB_CONSTRUCT_STATE != 0,
            [INTRINSIC_LIGHT, INTRINSIC_LIGHT_PRIMITIVE_ARGS].contains(&class),
            "class {class}'s construct state"
        );
        match kind & 3 {
            NATIVE_STUB_TEST_NONE => assert_eq!(
                classify(class, &object, &[object, object]),
                NativeOperands::Light
            ),
            NATIVE_STUB_TEST_NUMBERS => {
                assert_eq!(
                    classify(class, &object, &[number, number]),
                    NativeOperands::Light
                );
                assert_eq!(
                    classify(class, &undefined, &[number, string]),
                    NativeOperands::Boundary
                );
            }
            NATIVE_STUB_TEST_PRIMITIVE_ARGS => {
                assert_eq!(
                    classify(class, &object, &[number, string]),
                    NativeOperands::Light
                );
                assert_eq!(
                    classify(class, &undefined, &[string, object]),
                    NativeOperands::Boundary
                );
            }
            NATIVE_STUB_TEST_PRIMITIVE_OPERANDS => {
                assert_eq!(
                    classify(class, &string, &[number, string]),
                    NativeOperands::Light
                );
                assert_eq!(
                    classify(class, &object, &[number]),
                    NativeOperands::Boundary
                );
                assert_eq!(
                    classify(class, &string, &[object]),
                    NativeOperands::Boundary
                );
            }
            _ => unreachable!(),
        }
    }
}
