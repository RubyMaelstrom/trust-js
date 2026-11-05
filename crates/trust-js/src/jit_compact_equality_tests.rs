//! Equality and `!` at first-stage sites (the compact templates' step 2). The compiler puts a
//! literal operand of an equality on the right and folds `!` and `void` of a literal
//! (`bytecode::literal_truthiness`); a first-stage site outside loops tests a constant right
//! operand in line (`jit::emit_eq_constant`) and calls a shared stub for the rest
//! (`jit::emit_equal_stub`, `jit::emit_not_stub`), which calls the checked helper itself for
//! what it cannot decide. Every function here compiles at its first call, so each runs as
//! first-stage native code at `Tier::Jit`; the tree-walker, which keeps the source's evaluation
//! order, gives the expected results.

use crate::bytecode::{Chunk, Op, Tier};
use crate::{Completion, Engine};

/// A fresh engine at `tier` that compiles every function at its first call.
fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// Run `f` on the global function `name`'s first-stage chunk.
fn with_chunk<R>(engine: &mut Engine, name: &str, f: impl FnOnce(&Chunk) -> R) -> R {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let object_handle = value.as_obj().expect("a function object");
    let object = object_handle.borrow();
    let crate::value::ObjectKind::User(user) = &object.kind else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    let chunk = runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap_or_else(|| panic!("{name} has no chunk"));
    f(chunk)
}

/// Each operand kind against each constant kind, on either side, for every operator, as a value
/// and as a branch; two operands; `!` as a value and as a branch. The values include both
/// zeroes, NaN, flat, view and two-byte Strings, Strings that carry their key (a property key
/// from `Object.keys`) and one-unit Strings from `charAt`, Objects (one with [[IsHTMLDDA]]'s
/// ordinary absence), Symbols and BigInts.
const MATRIX: &str = r#"
var long = "";
for (var i = 0; i < 300; i++) long += String.fromCharCode(97 + (i % 26));
var keyed = Object.keys({ abc: 1, length: 2, "0": 3 });
var values = [
  undefined, null, true, false, 0, -0, 1, -1, 2.5, NaN, Infinity, "", "a", "{", "0", "abc",
  "length", "2.5", " 1 ", "été", "été".substring(1), long, long.substring(1, 299),
  long.substring(0, 3), keyed[0], keyed[1], keyed[2], "{abc".charAt(0), "xa".charAt(1),
  {}, [], [0], [""], Symbol("s"), 0n, 1n, { valueOf() { return 1; } },
  { toString() { return "a"; } }, function f() {}, new Number(0), new String(""),
];
var constants = ["undefined", "null", "true", "false", "0", "-0", "1", "2.5", '""', '"a"',
  '"{"', '"0"', '"abc"', '"length"', "void 0", "!0", "!1", "0n"];
var ops = ["===", "!==", "==", "!="];
var fns = [];
for (var c of constants) for (var op of ops) {
  fns.push(new Function("x", "return " + c + " " + op + " x;"));
  fns.push(new Function("x", "if (" + c + " " + op + " x) return 1; return 0;"));
  fns.push(new Function("x", "return x " + op + " " + c + ";"));
  fns.push(new Function("x", "if (x " + op + " " + c + ") return 1; return 0;"));
  fns.push(new Function("x", "return (x " + op + " " + c + ") ? 2 : 3;"));
  fns.push(new Function("x", "if (!(" + c + " " + op + " x)) return 1; return 0;"));
}
for (var op of ops) {
  fns.push(new Function("x", "y", "return x " + op + " y;"));
  fns.push(new Function("x", "y", "if (x " + op + " y) return 1; return 0;"));
  fns.push(new Function("x", "y", "var a = x, b = y; if (a " + op + " b) return 1; return 0;"));
  fns.push(new Function("x", "y", "return !(x " + op + " y);"));
}
fns.push(function (x) { return !x; });
fns.push(function (x) { if (!x) return 1; return 0; });
fns.push(function (x) { return !!x; });
var lines = [];
for (var fn of fns) {
  var line = "";
  for (var i = 0; i < values.length; i++) {
    var r = fn.length === 2 ? fn(values[i], values[(i * 7 + 3) % values.length]) : fn(values[i]);
    line += String(r) + ",";
  }
  lines.push(line);
}
lines.push(String([!0, !1, !"", !"a", !null, !void 0, void 0, !!0, !-0, !(0), !!1]));
var thrower = { valueOf() { throw new Error("boom"); } };
var looseThrow = new Function("x", "try { return 1 == x; } catch (e) { return e.message; }");
var looseBranch = new Function("x", "try { if (x != 1) return 'ne'; return 'eq'; } catch (e) { return e.message; }");
lines.push(looseThrow(thrower) + " " + looseBranch(thrower));
var calls = 0;
var counted = { valueOf() { calls++; return 3; } };
var once = new Function("x", "return 3 == x;");
var swapped = new Function("x", "return x == 3;");
lines.push(once(counted) + " " + swapped(counted) + " " + calls);
var order = [];
var sides = function () { return (order.push(1), "a") === (order.push(2), "a"); };
lines.push(sides() + " " + order.join(""));
lines.join("\n");
"#;

#[test]
fn equality_and_not_agree_with_the_tree_walker() {
    let expected = eval(&mut engine(Tier::Interp), MATRIX);
    assert!(expected.lines().count() > 300, "the matrix ran");
    for tier in [Tier::Bytecode, Tier::Jit] {
        let actual = eval(&mut engine(tier), MATRIX);
        let differ: Vec<usize> = expected
            .lines()
            .zip(actual.lines())
            .enumerate()
            .filter_map(|(line, (want, got))| (got != want).then_some(line))
            .collect();
        assert!(differ.is_empty(), "{tier:?}: lines {differ:?} differ");
        assert_eq!(actual, expected, "{tier:?}");
    }
}

#[test]
fn a_literal_operand_compiles_on_the_right_and_literals_fold() {
    use crate::value::{PACK_BOOL, PACK_NULL};
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        r#"
        function left(x) { return "a" === x; }
        function nil(x) { return null != x; }
        function voided(x) { return void 0 !== x; }
        function negated() { return !0; }
        function both() { return 1 === 2; }
        function none(x, y) { return x === y; }
        function inverted(x, y) { return !(x == y); }
        left("a"); nil(0); voided(0); negated(); both(); none(1, 2); inverted(1, 2);
        "#,
    );
    with_chunk(&mut engine, "left", |chunk| {
        let ops = chunk.jit_ops();
        assert!(
            matches!(ops[..3], [Op::LoadLocal(0), Op::Const(k), Op::StrictEq] if chunk.jit_const_is_str(k)),
            "{ops:?}"
        );
    });
    with_chunk(&mut engine, "nil", |chunk| {
        let ops = chunk.jit_ops();
        assert!(
            matches!(ops[..3], [Op::LoadLocal(0), Op::Const(k), Op::NotEq]
                if chunk.jit_const_scalar_bits(k) == Some(PACK_NULL)),
            "{ops:?}"
        );
    });
    with_chunk(&mut engine, "voided", |chunk| {
        let ops = chunk.jit_ops();
        assert!(
            matches!(ops[..3], [Op::LoadLocal(0), Op::Undef, Op::StrictNotEq]),
            "{ops:?}"
        );
    });
    with_chunk(&mut engine, "negated", |chunk| {
        let ops = chunk.jit_ops();
        assert!(
            matches!(ops[0], Op::Const(k) if chunk.jit_const_scalar_bits(k) == Some(PACK_BOOL | 1)),
            "{ops:?}"
        );
        assert!(!ops.iter().any(|op| matches!(op, Op::Not)), "{ops:?}");
    });
    // Two literals keep their order, as do two operands that are not literals.
    with_chunk(&mut engine, "both", |chunk| {
        let ops = chunk.jit_ops();
        assert!(
            matches!(ops[0], Op::Const(k) if chunk.jit_const_num(k) == Some(1f64.to_bits())),
            "{ops:?}"
        );
    });
    with_chunk(&mut engine, "none", |chunk| {
        let ops = chunk.jit_ops();
        assert!(
            matches!(ops[..3], [Op::LoadLocal(0), Op::LoadLocal(1), Op::StrictEq]),
            "{ops:?}"
        );
    });
    // `!` of an equality is the negated equality.
    with_chunk(&mut engine, "inverted", |chunk| {
        let ops = chunk.jit_ops();
        assert!(
            matches!(ops[..3], [Op::LoadLocal(0), Op::LoadLocal(1), Op::NotEq]),
            "{ops:?}"
        );
        assert!(!ops.iter().any(|op| matches!(op, Op::Not)), "{ops:?}");
    });
}
