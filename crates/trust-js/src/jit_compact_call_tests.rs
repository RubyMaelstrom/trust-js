//! Calls at first-stage sites outside loops (the compact templates' step 4): a native callee's
//! hit enters one shared stub for every intrinsic the site's arity can reach (the one- and
//! two-argument families, Math.max and Math.min, Array.prototype.pop, Function.prototype.call),
//! and the checked call helpers run through their shared stubs. Each function below holds one
//! call site, outside any loop, so its first native code is compact; the driver calls it with
//! values that complete each intrinsic, decline it (a non-object receiver, an argument of
//! another type) and throw from it, then with a user callee and an uncached one. Every tier
//! agrees; the tree-walker's results are the expected ones.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

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

const PROGRAM: &str = r#"
var lines = [];
function show(v) {
  if (typeof v === "object" && v !== null) return JSON.stringify(v);
  if (typeof v === "symbol") return v.toString();
  return String(v);
}
function rec(tag, f) {
  try { lines.push(tag + ":" + show(f())); }
  catch (e) { lines.push(tag + ":throw " + (e && e.constructor ? e.constructor.name : typeof e)); }
}
function pop0(a) { return a.pop(); }
function cc1(s, i) { return s.charCodeAt(i); }
function ca1(s, i) { return s.charAt(i); }
function floor1(x) { return Math.floor(x); }
function push1(a, v) { return a.push(v); }
function push1d(a, v) { a.push(v); return a.length; }
function call1(f, t) { return f.call(t); }
function call2(f, t, x) { return f.call(t, x); }
function call3(f, t, x, y) { return f.call(t, x, y); }
function call5(f, t, a, b, c, d) { return f.call(t, a, b, c, d); }
function call9(f, t, a, b, c, d, e, g, h, k) { return f.call(t, a, b, c, d, e, g, h, k); }
function max3(a, b, c) { return Math.max(a, b, c); }
function min4(a, b, c, d) { return Math.min(a, b, c, d); }
function slice2(s, a, b) { return s.slice(a, b); }
function own1(o) { return o.hasOwnProperty("x"); }
function user1(o, v) { return o.m(v); }
function plain1(f, v) { return f(v); }
function make(C, v) { return new C(v); }
function sum() { var s = this === undefined ? "u" : show(this); for (var i = 0; i < arguments.length; i++) s += "," + show(arguments[i]); return s; }
function thrower() { throw new RangeError("boom"); }
function Point(x) { this.x = x; }
var fakePop = { pop: function () { return "own pop"; } };
var poly = [{ m: function (v) { return v + 1; } }, { m: function (v) { return v * 2; } },
  { m: function (v) { return "s" + v; } }, { m: function (v) { return [v]; } },
  { m: function (v) { return -v; } }];
for (var round = 0; round < 3; round++) {
  rec("pop", function () { return pop0([1, 2, 3]); });
  rec("pop-empty", function () { return pop0([]); });
  rec("pop-own", function () { return pop0(fakePop); });
  rec("pop-prim", function () { return pop0("abc"); });
  rec("cc", function () { return cc1("hello", 1); });
  rec("cc-out", function () { return cc1("hello", 9); });
  rec("cc-two-byte", function () { return cc1("héllo", 1); });
  rec("cc-num-recv", function () { return cc1(12345, 1); });
  rec("ca", function () { return ca1("hello", 4); });
  rec("floor", function () { return floor1(2.7); });
  rec("floor-str", function () { return floor1("3.9"); });
  rec("push", function () { return push1([1], 2); });
  rec("push-d", function () { return push1d([1, 2], 3); });
  rec("call1", function () { return call1(sum, { t: 1 }); });
  rec("call1-prim", function () { return call1(sum, 5); });
  rec("call2", function () { return call2(sum, null, "x"); });
  rec("call3", function () { return call3(sum, [1], 2, 3); });
  rec("call5", function () { return call5(sum, "s", 1, 2, 3, 4); });
  rec("call9", function () { return call9(sum, 0, 1, 2, 3, 4, 5, 6, 7, 8); });
  rec("call-throw", function () { return call2(thrower, null, 1); });
  rec("call-nonfn", function () { return call2({ call: Function.prototype.call }, null, 1); });
  rec("call-bound", function () { return call2(sum.bind("b"), null, 2); });
  rec("max3", function () { return max3(1, 5, 3); });
  rec("max3-nan", function () { return max3(1, NaN, 3); });
  rec("max3-str", function () { return max3("7", 5, 3); });
  rec("min4", function () { return min4(4, -0, 0, 9); });
  rec("min4-obj", function () { return min4(4, { valueOf: function () { return -2; } }, 0, 9); });
  rec("slice2", function () { return slice2("abcdef", 1, 4); });
  rec("slice2-neg", function () { return slice2("abcdef", -3, -1); });
  rec("slice2-arr", function () { return slice2([1, 2, 3, 4], 1, 3); });
  rec("own", function () { return own1({ x: 1 }); });
  rec("own-no", function () { return own1({ y: 1 }); });
  rec("user", function () { return user1(poly[round], 4); });
  for (var k = 0; k < poly.length; k++) rec("poly" + k, function () { return user1(poly[k], 3); });
  rec("plain", function () { return plain1(Math.abs, -4); });
  rec("plain-user", function () { return plain1(function (v) { return v + "!"; }, 4); });
  rec("plain-throw", function () { return plain1(thrower, 0); });
  rec("plain-nonfn", function () { return plain1(17, 0); });
  rec("new", function () { return make(Point, 3).x; });
  rec("new-native", function () { return make(Array, 3).length; });
  rec("new-nonctor", function () { return make(Math.abs, 3); });
}
lines.join("\n");
"#;

#[test]
fn compact_call_sites_agree_with_the_tree_walker() {
    let expected = eval(&mut engine(Tier::Interp), PROGRAM);
    assert!(
        expected.lines().count() > 100,
        "the program ran: {expected}"
    );
    for tier in [Tier::Bytecode, Tier::Jit] {
        let actual = eval(&mut engine(tier), PROGRAM);
        let differ: Vec<String> = expected
            .lines()
            .zip(actual.lines())
            .filter(|(want, got)| got != want)
            .map(|(want, got)| format!("{want} / {got}"))
            .collect();
        assert!(differ.is_empty(), "{tier:?}: {differ:?}");
        assert_eq!(actual, expected, "{tier:?}");
    }
}
