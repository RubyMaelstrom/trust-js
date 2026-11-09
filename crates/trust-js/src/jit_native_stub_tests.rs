//! A native's fast entry from generated code (candidate 2 (a)): the call-hit form's shared stub
//! (`emit_call_hit_stub`) runs a native of a light class on light operands through
//! `bytecode::jit_native_light`, every other engine native through the lean boundary
//! (`bytecode::jit_native_boundary`, fork B2) and Reflect.get, has and set through
//! `bytecode::jit_native_reflect`, and leaves user callees, scoped natives, Reflect.apply and
//! DataView's accessors to the call-hit helper.

use crate::bytecode::{
    native_operands, native_stub_kind, NativeOperands, Tier, INTRINSIC_DATAVIEW_GET,
    INTRINSIC_DATAVIEW_SET, INTRINSIC_LIGHT, INTRINSIC_LIGHT_PRIMITIVE_ARGS,
    INTRINSIC_REFLECT_APPLY, INTRINSIC_REFLECT_GET, INTRINSIC_REFLECT_SET, NATIVE_STUB_BOUNDARY,
    NATIVE_STUB_CONSTRUCT_STATE, NATIVE_STUB_LIGHT, NATIVE_STUB_REFLECT, NATIVE_STUB_TEST_NONE,
    NATIVE_STUB_TEST_NUMBERS, NATIVE_STUB_TEST_PRIMITIVE_ARGS, NATIVE_STUB_TEST_PRIMITIVE_OPERANDS,
    TEST_NATIVE_BOUNDARY_STUB_CALLS, TEST_NATIVE_STUB_CALLS,
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
/// natives the stub ran on the caller's window for the machine-code run's last `rounds`
/// evaluations of `call`.
fn stub_calls(source: &str, call: &str, rounds: usize) -> (String, usize) {
    served(&TEST_NATIVE_STUB_CALLS, source, call, rounds)
}

/// As [`stub_calls`], for the natives the stub ran through its boundary and Reflect entries.
fn boundary_stub_calls(source: &str, call: &str, rounds: usize) -> (String, usize) {
    served(&TEST_NATIVE_BOUNDARY_STUB_CALLS, source, call, rounds)
}

fn served(
    counter: &'static std::thread::LocalKey<std::cell::Cell<usize>>,
    source: &str,
    call: &str,
    rounds: usize,
) -> (String, usize) {
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
    let before = counter.with(std::cell::Cell::get);
    for _ in 0..rounds {
        assert_eq!(eval(&mut jit, call), expected, "{call}");
    }
    (expected, counter.with(std::cell::Cell::get) - before)
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

/// A native called while its caller constructs runs with a call's construct state: the entry of
/// the classes that hold constructors saves and clears it around the native (Number and Boolean
/// return primitives).
#[test]
fn a_constructing_callers_constructor_natives_run_with_a_calls_construct_state() {
    let source = r#"
function Box(v) { this.n = Number(v); this.b = Boolean(v); this.k = typeof this.n + typeof this.b; }
function make(n) { var t = ''; for (var i = 0; i < n; i++) t = new Box('5').k; return t; }
"#;
    let (result, served) = stub_calls(source, "make(20)", 2);
    assert_eq!(result, "numberboolean");
    assert!(
        served >= 2 * 20 * 2 * 9 / 10,
        "{served} natives through the stub"
    );
}

/// A constructing caller's light natives run through the stub: those of the classes that hold no
/// constructor as they are (they read no construct state), the constructors' classes with a
/// call's construct state; every result is the tree-walker's. The fills check that no
/// constructor is in a class the stub runs without its construct state
/// (`bytecode::native_stub_kind`).
#[test]
fn a_constructing_callers_other_light_natives_take_their_fast_entry() {
    // Counts the natives the stub runs, from the tier's effects too; a forced exit hands the
    // rest of its call to the VM.
    crate::opt::pin_forced_exits_off_for_test();
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
    // Math.round, Math.sign, parseInt, parseFloat, isFinite, Number, String, Boolean,
    // String.fromCodePoint, m.get, Number.isInteger, indexOf, includes, substring, trim and
    // toUpperCase a construction at least: no template finishes them in line.
    let (_, served) = stub_calls(source, "make(20)", 2);
    assert!(
        served >= 16 * 20 * 2 * 9 / 10,
        "{served} natives through the stub"
    );
}

/// The stub kinds agree with `native_operands`: a light kind's class is light on the operands its
/// test passes and needs the boundary on those it fails; every class that is light for some
/// operands has a light kind, except DataView's accessors, which the helper serves with
/// Reflect.apply; Reflect's three have their kind; every other class is the boundary's; and the
/// construct state is tested for the classes that hold constructors.
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
        if kind & NATIVE_STUB_LIGHT == 0 {
            let expected = if [
                INTRINSIC_REFLECT_APPLY,
                INTRINSIC_DATAVIEW_GET,
                INTRINSIC_DATAVIEW_SET,
            ]
            .contains(&class)
            {
                0
            } else if (INTRINSIC_REFLECT_GET..=INTRINSIC_REFLECT_SET).contains(&class) {
                NATIVE_STUB_REFLECT
            } else {
                assert_ne!(
                    classify(class, &string, &[number, string]),
                    NativeOperands::Light,
                    "class {class} is light for primitive operands but has no light kind"
                );
                NATIVE_STUB_BOUNDARY
            };
            assert_eq!(kind, expected, "class {class}");
            continue;
        }
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

/// Natives that may run author code take the stub's lean-boundary entry, in loops: nearly every
/// call (a call that meets a recompile's first entry takes the helper).
#[test]
fn boundary_natives_take_the_lean_boundary_from_the_stub() {
    let source = r#"
var o = {a: 1, b: 2}, arr = [1, 2, 3];
function keys(n) { var t = 0; for (var i = 0; i < n; i++) t += Object.keys(o).length; return t; }
function index(n) { var t = 0; for (var i = 0; i < n; i++) t += arr.indexOf(2); return t; }
function proto(n) { var t = 0; for (var i = 0; i < n; i++) t += Object.getPrototypeOf(o) === Object.prototype ? 1 : 0; return t; }
function parse(n) { var t = 0; for (var i = 0; i < n; i++) t += JSON.parse('[1,2]').length; return t; }
"#;
    for call in ["keys(50)", "index(50)", "proto(50)", "parse(50)"] {
        let (_, served) = boundary_stub_calls(source, call, 3);
        assert!(
            served * 10 >= 150 * 9,
            "{call}: {served} natives through the stub"
        );
    }
    // A light class's native whose operands its test refuses goes there too.
    let (result, served) = boundary_stub_calls(
        "var probe = { toString() { return 'b'; } }; \
         function refused(n) { var t = 0; for (var i = 0; i < n; i++) t += 'abc'.includes(probe) ? 1 : 0; return t; }",
        "refused(50)",
        3,
    );
    assert_eq!(result, "50");
    assert!(served * 10 >= 150 * 9, "{served} natives through the stub");
}

/// Natives re-enter author code through the lean boundary: replace's function argument, a
/// Proxy's trap, a getter Object.assign reads and a ToString that includes runs, each as the
/// tree-walker computes.
#[test]
fn a_boundary_natives_author_code_re_enters() {
    let source = r#"
var calls = 0, s = 'axbxc';
function rep(n) { var t = ''; for (var i = 0; i < n; i++) t = s.replace(/x/g, function (m) { calls++; return m.toUpperCase(); }); return t + ':' + calls; }
var traps = 0, proxy = new Proxy({a: 1, b: 2}, { ownKeys(t) { traps++; return Reflect.ownKeys(t); } });
function trapped(n) { var t = 0; for (var i = 0; i < n; i++) t += Object.keys(proxy).length; return t + ':' + traps; }
var reads = 0, getter = { get v() { reads++; return 2; } };
function assigned(n) { var t = 0; for (var i = 0; i < n; i++) t += Object.assign({}, getter).v; return t + ':' + reads; }
var probe = { toString() { reads++; return 'b'; } };
function converted(n) { var t = 0; for (var i = 0; i < n; i++) t += 'abc'.includes(probe) ? 1 : 0; return t + ':' + reads; }
"#;
    for (call, expected) in [
        ("calls = 0, rep(20)", "aXbXc:40"),
        ("traps = 0, trapped(20)", "40:20"),
        ("reads = 0, assigned(20)", "40:20"),
        ("reads = 0, converted(20)", "20:20"),
    ] {
        let (result, served) = boundary_stub_calls(source, call, 2);
        assert_eq!(result, expected, "{call}");
        assert!(served >= 36, "{call}: {served} natives through the stub");
    }
}

/// A native's pending tail call (the bytecode's proper tail calls leave one, and a native may, as
/// `tailCall` here does), which the lean boundary runs after the native returns.
#[test]
fn the_lean_boundary_runs_a_natives_pending_tail_call() {
    fn tail_call(
        i: &mut crate::interpreter::Interp,
        _: Value,
        args: &[Value],
    ) -> Result<Value, Value> {
        let callee = args.first().cloned().unwrap_or(Value::Undefined);
        let rest = args.get(1..).unwrap_or(&[]).to_vec();
        i.pending_tail = Some(Box::new((callee, Value::Undefined, rest)));
        Ok(Value::Undefined)
    }
    let mut jit = Engine::new();
    jit.set_tier(Tier::Jit);
    jit.set_tier_threshold(0);
    super::rooting_tests::install(&mut jit, "tailCall", 2, tail_call);
    eval(
        &mut jit,
        "function inc(x) { return x + 1; } \
         function tail(n) { var t = 0; for (var i = 0; i < n; i++) t += tailCall(inc, i); return t; }",
    );
    for _ in 0..8 {
        assert_eq!(eval(&mut jit, "tail(50)"), "1275");
    }
    let before = TEST_NATIVE_BOUNDARY_STUB_CALLS.with(std::cell::Cell::get);
    for _ in 0..2 {
        assert_eq!(eval(&mut jit, "tail(50)"), "1275");
    }
    let served = TEST_NATIVE_BOUNDARY_STUB_CALLS.with(std::cell::Cell::get) - before;
    assert!(served * 10 >= 100 * 9, "{served} natives through the stub");
}

/// Reflect.get, has and set take their own entry: answered on the operands as the call-hit
/// helper answers them (keys that carry their key, as `Interp::reflect_resolved` requires), and
/// through the lean boundary for a Proxy, whose trap runs.
#[test]
fn reflect_get_has_and_set_take_their_own_entry() {
    let source = r#"
var o = {a: 1, b: 0}, ka = Object.keys(o)[0], kb = Object.keys(o)[1];
function reflect(n) { var t = 0; for (var i = 0; i < n; i++) { Reflect.set(o, kb, i); t += Reflect.get(o, ka) + (Reflect.has(o, kb) ? 1 : 0); } return t; }
var traps = 0, p = new Proxy({a: 1}, { get(t, k) { traps++; return t[k]; } });
function proxied(n) { var t = 0; for (var i = 0; i < n; i++) t += Reflect.get(p, 'a'); return t + ':' + traps; }
"#;
    let resolved = || crate::interpreter::TEST_REFLECT_RESOLVED.with(std::cell::Cell::get);
    let before = resolved();
    let (result, served) = boundary_stub_calls(source, "reflect(50)", 3);
    assert_eq!(result, "100");
    assert!(
        served * 10 >= 450 * 9,
        "{served} Reflect calls through the stub"
    );
    assert!(resolved() - before >= 450, "answered on the operands");
    let (result, served) = boundary_stub_calls(source, "traps = 0, proxied(20)", 2);
    assert_eq!(result, "20:20");
    assert!(served >= 36, "{served} Reflect calls through the stub");
}

/// A boundary native's error unwinds through the site's handler, with the call's position.
#[test]
fn a_boundary_natives_error_unwinds_with_its_call_position() {
    let source = r#"
function bad(n) {
    var caught = 0, stack = '';
    for (var i = 0; i < n; i++) {
        try { JSON.parse('{'); } catch (e) { caught++; stack = e.stack; }
    }
    return caught + '|' + /SyntaxError/.test(stack) + '|' + /bad/.test(stack) + '|' + /:5:/.test(stack);
}
"#;
    let (result, served) = boundary_stub_calls(source, "bad(10)", 2);
    assert_eq!(result, "10|true|true|true");
    assert!(served >= 18, "{served} natives through the stub");
}

/// A codePointAt site whose String.prototype fuse step misses (`jit::string_calls`: a String
/// object receiver, or an index that is no Number) runs the unfused templates and the call: its
/// native through the stub, the light entry for a primitive receiver and a primitive index, the
/// lean boundary for an object receiver, each as the tree-walker computes.
#[test]
fn fused_string_sites_that_miss_their_fuse_take_the_stub() {
    let source = r#"
var s = 'abcdef', boxed = new String('abcdef'), keys = ['1', '2', '3'];
function boxedAt(n) { var t = 0; for (var i = 0; i < n; i++) t += boxed.codePointAt(i % 6); return t; }
function keyedAt(n) { var t = 0; for (var i = 0; i < n; i++) t += s.codePointAt(keys[i % 3]); return t; }
"#;
    let (result, served) = boundary_stub_calls(source, "boxedAt(60)", 3);
    assert_eq!(result, "5970");
    assert!(
        served * 10 >= 180 * 9,
        "{served} object receivers through the stub"
    );
    let (result, served) = stub_calls(source, "keyedAt(60)", 3);
    assert_eq!(result, "5940");
    assert!(
        served * 10 >= 180 * 9,
        "{served} String keys through the stub"
    );
}
