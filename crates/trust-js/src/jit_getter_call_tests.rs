//! Getters called through the direct sequence from the property template's validated accessor
//! landing (ECMA-262 GetValue; §10.1.8.1 OrdinaryGet step 8, Call(getter, Receiver); §10.2.1
//! [[Call]] with no NewTarget). The getter is the accessor's live [[Get]] at each read, even when
//! a replacement keeps the holder's shape; `this` is the read's original receiver (a primitive's
//! own word into a strict getter, boxed for a sloppy one); the getter sees no arguments and
//! new.target undefined, whatever the reading code's construct state. The programs read through
//! every receiver form (an operand, `this`, a local, a method read), at prototype depths 0-3,
//! through polymorphic and code-keyed getters, getters the sequence cannot enter (one that builds
//! an environment), native getters at the same sites, and getters that throw (the error placed at
//! the read, as the helper places it), recurse past the stack limit, collect or delete their own
//! property. Every tier agrees; the expected values are Node's.
//!
//! On ARM64 the warmed reads' getters complete through the sequence (the getter chunk's direct
//! return and throw counters), and while one runs its receiver and the getter itself stay roots
//! that the getter's call cache does not keep alive afterwards.

use crate::bytecode::Tier;
use crate::Engine;

use super::traced_tests::eval;
#[cfg(target_arch = "aarch64")]
use super::traced_tests::{assert_native, settle};

/// The bytecode VM runs a warm engine's loop-free functions make before native code.
const WARM_RUNS: u32 = 20;

/// A fresh engine at `tier` with `collect()`. A read's code calls getters directly when the
/// read's property cache had seen an accessor when the code was made: a `warm` engine runs a
/// loop-free function on the bytecode VM a while before compiling it to native code, as a cold
/// one compiles each function at its first call.
fn engine(tier: Tier, warm: bool) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(u32::from(warm));
    if warm {
        engine.interp.jit_admit_runs = WARM_RUNS;
    }
    let collect = engine.interp.make_native("collect", 0, |interp, _, _| {
        interp.gc_collect();
        Ok(crate::value::Value::Undefined)
    });
    engine.interp.global.borrow_mut().props.insert(
        "collect",
        crate::value::Property::data(crate::value::Value::Obj(collect), true, true, true),
    );
    engine
}

const READS: &str = r#"
function forms() {
  class P { constructor(x) { this.x = x; } get d() { return this.x * 2; } get self() { return this; }
    get m() { return function () { return this === undefined ? 'u' : this.x; }; } sum() { return this.d + this.d; } }
  function operand(o) { return [o][0].d; }
  function local(o) { var t = 0; for (var i = 0; i < 3; i++) t += o.d; return t; }
  function method(o) { return o.m(); }
  var a = new P(3), b = new P(4), s = 0, same = true;
  for (var k = 0; k < 300; k++) { s += operand(a) + local(b) + a.sum() + method(a); if (a.self !== a) same = false; }
  return s + ':' + same;
}
function replaced() {
  var o = {}; Object.defineProperty(o, 'v', { get() { return 1; }, configurable: true });
  function read(x) { return x.v; }
  var t = 0; for (var k = 0; k < 200; k++) t += read(o);
  Object.defineProperty(o, 'v', { get() { return 100; }, configurable: true });
  for (var k = 0; k < 200; k++) t += read(o);
  Object.defineProperty(o, 'v', { get: undefined, set(v) {}, configurable: true });
  var absent = read(o);
  Object.defineProperty(o, 'v', { value: 'data', configurable: true });
  return t + ':' + absent + ':' + read(o);
}
function throwing() {
  var o = { n: 0 }; Object.defineProperty(o, 'boom', { get() { this.n++; if (this.n % 97 === 0) throw new Error('boom' + this.n); return this.n; } });
  function read(x) { var arr = [1, 2, 3]; try { return arr.length + x.boom; } catch (e) { return e.message; } }
  var caught = []; for (var k = 0; k < 300; k++) { var v = read(o); if (typeof v === 'string') caught.push(v); }
  return caught.join(',') + ':' + o.n;
}
function depths() {
  function A() {} Object.defineProperty(A.prototype, 'g', { get() { return 'A' + (this.tag || ''); } });
  function B() {} B.prototype = Object.create(A.prototype);
  function C() {} C.prototype = Object.create(B.prototype);
  var own = {}; Object.defineProperty(own, 'g', { get() { return 'own'; } });
  var c = new C(); c.tag = 'c';
  var objects = [own, new A(), new B(), c], out = '';
  for (var j = 0; j < 4; j++) { var read = function (o) { return o.g; }, v; for (var k = 0; k < 200; k++) v = read(objects[j]); out += v; }
  return out;
}
function primitives() {
  Object.defineProperty(String.prototype, 'strictLen', { get: function () { 'use strict'; return typeof this + this.length; }, configurable: true });
  Object.defineProperty(String.prototype, 'sloppyLen', { get: function () { return typeof this + this.length; }, configurable: true });
  function read(s) { return s.strictLen + '/' + s.sloppyLen; }
  function strictRead(s) { 'use strict'; return s.strictLen; }
  var v; for (var k = 0; k < 200; k++) v = read('abc') + '|' + strictRead('xy');
  delete String.prototype.strictLen; delete String.prototype.sloppyLen;
  return v;
}
function callState() {
  var o = {}; Object.defineProperty(o, 'info', { get: function () { return [arguments.length, new.target === undefined, this === o].join(); } });
  function read(x) { return x.info; }
  var v; for (var k = 0; k < 200; k++) v = read(o);
  var p = {}; Object.defineProperty(p, 'nt', { get: function () { return new.target === undefined ? 'u' : 'd'; } });
  function K() { this.v = p.nt; this.after = new.target === K; }
  var made; for (var k = 0; k < 200; k++) made = new K();
  return v + ':' + made.v + made.after;
}
function shapes() {
  var holder = this;
  var arrow = {}; Object.defineProperty(arrow, 'a', { get: () => holder === undefined });
  var env = { base: 5 }; Object.defineProperty(env, 'e', { get: function () { var b = this.base; var f = function () { return b + 1; }; return f(); } });
  function reactive(obj, key, val) { Object.defineProperty(obj, key, { get: function reactiveGetter() { return val; }, configurable: true }); }
  var many = []; for (var i = 0; i < 20; i++) { var r = {}; reactive(r, 'v', i); many.push(r); }
  class S1 { get area() { return 1; } } class S2 { get area() { return 2; } } class S3 { get area() { return 3; } }
  class S4 { get area() { return 4; } } class S5 { get area() { return 5; } }
  var poly = [new S1(), new S2(), new S3(), new S4(), new S5()];
  var map = new Map([[1, 2]]), typed = new Uint8Array(7), plain = { get size() { return 10; } };
  function readA(o) { return o.a; } function readE(o) { return o.e; } function readV(o) { return o.v; }
  function readArea(o) { return o.area; } function readSize(o) { return o.size === undefined ? o.length : o.size; }
  var t = [0, 0, 0, 0, 0];
  for (var k = 0; k < 1000; k++) {
    t[0] += readA(arrow) ? 1 : 0; t[1] += readE(env); t[2] += readV(many[k % 20]);
    t[3] += readArea(poly[k % 5]); t[4] += readSize(k % 3 === 0 ? map : k % 3 === 1 ? typed : plain);
  }
  return t.join(',');
}
function lifetimes() {
  var o = {}, calls = 0;
  Object.defineProperty(o, 'x', { get: function () {
    calls++; var junk = []; for (var i = 0; i < 20; i++) junk.push({ i: i });
    if (calls === 150) Object.defineProperty(o, 'x', { value: 'data', configurable: true });
    collect(); return junk.length + junk[19].i; }, configurable: true });
  function read(x) { return x.x; }
  var seen = []; for (var k = 0; k < 300; k++) { var v = read(o); if (k >= 148 && k <= 151) seen.push(v); }
  var deep = { n: 0 }; Object.defineProperty(deep, 'down', { get: function () { if (this.n <= 0) return 0; this.n--; return 1 + this.down; } });
  function descend(x, n) { x.n = n; return x.down; }
  var total = 0; for (var k = 0; k < 30; k++) total += descend(deep, 40);
  var overflow = 'skipped';
  if (!stressed) { try { descend(deep, 1e6); } catch (e) { overflow = e.constructor.name; } }
  return seen.join(',') + ':' + calls + ':' + total + ':' + overflow;
}
function stacks() {
  var o = {}; Object.defineProperty(o, 'bad', { get: function badGetter() { throw new Error('bad'); } });
  function readBad(x) {
    return x.bad;
  }
  var stack = ''; for (var k = 0; k < 200; k++) { try { readBad(o); } catch (e) { stack = e.stack; } }
  var lines = stack.split('\n'), at = /:(\d+):(\d+)\)?$/, getter = at.exec(lines[1]), reader = at.exec(lines[2]);
  return /badGetter/.test(lines[1]) + ':' + (reader[1] - getter[1]) + ':' + reader[2];
}
[forms(), replaced(), throwing(), depths(), primitives(), callState(), shapes(), lifetimes(), stacks()].join('\n')
"#;

const READ_RESULTS: &str = "13500:true\n20200:undefined:data\nboom97,boom194,boom291:300\nownAAAc\n\
string3/object3|string2\n0,true,true:utrue\n0,6000,9500,3000,5995\n39,39,data,data:150:1200:RangeError\n\
true:2:14";

#[test]
fn getter_reads_agree_across_tiers() {
    // Under GC stress, a collection (and in debug builds the heap's verification) at every
    // safepoint, the programs run where reads call getters directly, and without the recursion
    // to the stack limit, whose every call would scan the whole native stack.
    let stressed = crate::gc_test::stress_active();
    let source = format!("var stressed = {stressed};{READS}");
    let expected = if stressed {
        READ_RESULTS.replace(":RangeError", ":skipped")
    } else {
        READ_RESULTS.to_string()
    };
    let runs = [
        (Tier::Interp, false),
        (Tier::Bytecode, false),
        (Tier::Jit, false),
        (Tier::Jit, true),
    ];
    for (tier, warm) in runs.into_iter().skip(if stressed { 3 } else { 0 }) {
        let mut engine = engine(tier, warm);
        assert_eq!(
            eval(&mut engine, &source),
            expected,
            "{tier:?}, warm {warm}"
        );
    }
}

/// One class's getters, read through each receiver form, and a getter that throws.
#[cfg(target_arch = "aarch64")]
const FORMS: &str = r#"
class P { constructor(x) { this.x = x; } get d() { return this.x * 2; }
  get m() { return function () { return this.x; }; } twice() { return this.d + this.d; } }
var getterD = Object.getOwnPropertyDescriptor(P.prototype, 'd').get;
var getterM = Object.getOwnPropertyDescriptor(P.prototype, 'm').get;
var ps = [new P(3), new P(4)];
function operand(k) { return ps[k & 1].d; }
function local(k) { var o = ps[k & 1]; return o.d; }
function viaThis(k) { return ps[k & 1].twice(); }
function method(k) { return ps[k & 1].m(); }
var bad = {}; Object.defineProperty(bad, 'v', { get: function badGetter() { throw new Error('bad'); } });
var getterBad = Object.getOwnPropertyDescriptor(bad, 'v').get;
function readBad() {
  return bad.v;
}
function sum(read) { var t = 0; for (var k = 0; k < 50; k++) t += read(k); return t; }
function position() {
  var stack; for (var k = 0; k < 50; k++) { try { readBad(); } catch (e) { stack = e.stack; } }
  return /:(\d+:\d+)\)?$/.exec(stack.split('\n')[2])[1];
}
[sum(operand), sum(local), sum(viaThis), sum(method), position()].join()
"#;

/// The completions `count` reads from the chunks of `function`, the function itself and the
/// one its inline recompile made.
#[cfg(target_arch = "aarch64")]
fn completions(
    function: &crate::ast::Function,
    count: fn(&crate::bytecode::Chunk) -> usize,
) -> usize {
    let runtime = function.runtime();
    [runtime.code.get(), runtime.code2.get()]
        .into_iter()
        .flatten()
        .flatten()
        .map(|chunk| count(chunk))
        .sum()
}

/// The function of the closure the global `name` holds.
#[cfg(target_arch = "aarch64")]
fn function_of(engine: &mut Engine, name: &str) -> std::rc::Rc<crate::ast::Function> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let object = value.as_obj().expect("a function object");
    let object = object.borrow();
    let crate::value::Callable::User(user) = &object.call else {
        panic!("{name} is not a user function")
    };
    user.func.clone()
}

/// Run `round` until the counter `count` of the getter `getter`'s chunks advances, as the
/// sequence's completions count them: a concurrent test can exhaust the process-wide call-cache
/// epoch, which sends reads through the checked helper until their sites refill.
#[cfg(target_arch = "aarch64")]
fn direct_round(
    engine: &mut Engine,
    getter: &str,
    count: fn(&crate::bytecode::Chunk) -> usize,
    round: &str,
) -> String {
    let function = function_of(engine, getter);
    let mut result = String::new();
    for _ in 0..100 {
        let before = completions(&function, count);
        result = eval(engine, round);
        if completions(&function, count) > before {
            return result;
        }
    }
    panic!("{round}: {getter} never completed through the direct sequence ({result})");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn warmed_getters_complete_through_the_direct_sequence() {
    let mut engine = engine(Tier::Jit, true);
    // The read's line and column in FORMS (`bad.v`'s property name, as Node reports it).
    let (line, column) = FORMS
        .lines()
        .enumerate()
        .find_map(|(index, text)| text.find("return bad.v;").map(|at| (index + 1, at + 12)))
        .unwrap();
    let position = format!("{line}:{column}");
    assert_eq!(
        eval(&mut engine, FORMS),
        format!("350,350,700,175,{position}")
    );
    assert_native(
        &mut engine,
        &[
            "operand",
            "local",
            "viaThis",
            "method",
            "readBad",
            "getterD",
            "getterM",
            "getterBad",
        ],
    );
    for getter in ["getterD", "getterM", "getterBad"] {
        settle(&mut engine, getter);
    }
    let returns = |chunk: &crate::bytecode::Chunk| chunk.test_direct_returns.get();
    for (reader, getter, expected) in [
        ("operand", "getterD", "350"),
        ("local", "getterD", "350"),
        ("viaThis", "getterD", "700"),
        ("method", "getterM", "175"),
    ] {
        let round = format!("sum({reader})");
        assert_eq!(
            direct_round(&mut engine, getter, returns, &round),
            expected,
            "{reader}"
        );
    }
    let throws = |chunk: &crate::bytecode::Chunk| chunk.test_direct_throws.get();
    assert_eq!(
        direct_round(&mut engine, "getterBad", throws, "position()"),
        position,
        "a direct getter's throw is placed at the read"
    );
}

/// While a getter runs through the sequence, the getter (no longer its holder's: a replacement
/// keeps the holder's shape, and so the read's cache) and the receiver are roots; afterwards
/// nothing keeps them, the getter's call cache included.
#[cfg(target_arch = "aarch64")]
#[test]
fn a_running_getter_and_its_receiver_are_roots() {
    use super::rooting_tests::{assert_tracked_collectable, jit_engine, run};
    let mut engine = jit_engine(1);
    engine.interp.jit_admit_runs = WARM_RUNS;
    let source = r#"
        function R(armed) { this.armed = armed; }
        function other() { return 'other'; }
        function install() {
            Object.defineProperty(R.prototype, 'g', { configurable: true, get: function self() {
                if (!this.armed) return 'cold';
                Object.defineProperty(R.prototype, 'g', { configurable: true, get: other });
                track(self); track(this);
                var marks = collect();
                return marks + ':' + typeof self + ':' + this.armed;
            } });
        }
        function read(o) { return o.g; }
        install();
        var probe = Object.getOwnPropertyDescriptor(R.prototype, 'g').get;
        var warm = ''; for (var k = 0; k < 100; k++) warm = read(new R(false));
        warm;
    "#;
    assert_eq!(run(&mut engine, source), "cold");
    assert_native(&mut engine, &["read", "probe"]);
    // Every closure the rounds install runs the probe's function, whose chunk counts the
    // sequence's returns into it.
    settle(&mut engine, "probe");
    let function = function_of(&mut engine, "probe");
    run(&mut engine, "probe = null;");
    let returns = || completions(&function, |chunk| chunk.test_direct_returns.get());
    let mut result = String::new();
    for _ in 0..100 {
        let before = returns();
        result = run(&mut engine, "install(); read(new R('A'));");
        if returns() > before {
            break;
        }
    }
    assert_eq!(result, "RR:function:A");
    assert!(
        returns() > 0,
        "the armed read ran its getter through the sequence"
    );
    assert_tracked_collectable(&mut engine);
}
