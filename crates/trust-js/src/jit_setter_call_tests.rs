//! Setters called by property stores whose caches prove the accessor (ECMA-262 PutValue;
//! §10.1.9.2 OrdinarySetWithOwnDescriptor step 6, Call(setter, Receiver, « V »); §10.2.1 [[Call]]
//! with no NewTarget). The setter is the accessor's live [[Set]] at each store, even when a
//! replacement keeps the holder's shape; `this` is the store's original receiver, never the
//! holder; the setter sees one argument and new.target undefined; the assignment's value is the
//! right-hand side, whatever the setter returns; an accessor without a setter fails silently in
//! sloppy code and throws in strict code. The programs store through every form (an operand, a
//! kept value, `this`, a local), at prototype depths 0-6, through polymorphic and code-keyed
//! setters, setters that build an environment, native setters (`__proto__`), Proxies on the
//! chain, frozen and non-extensible receivers, primitives, compound assignments and updates (the
//! getter, then the setter), and setters that throw, recurse past the stack limit, collect, or
//! reshape the receiver and replace themselves. Every tier agrees; the expected values are Node's.
//!
//! On ARM64 the stores' setters complete through the direct sequence from the stores' helper
//! stub (the setter chunk's direct return and throw counters), from a function's first native
//! code as from code made once the store's cache had seen the accessor; while one runs, its
//! receiver, the setter and the value stay roots that the store's call cache does not keep alive
//! afterwards.

use crate::bytecode::Tier;
use crate::Engine;

use super::traced_tests::eval;
#[cfg(target_arch = "aarch64")]
use super::traced_tests::{assert_native, settle};

/// The bytecode VM runs a warm engine's loop-free functions this many times before native code.
const WARM_RUNS: u32 = 20;

/// A fresh engine at `tier` with `collect()`. A cold engine compiles each function at its first
/// call, before its stores' caches have seen anything; a `warm` one first runs a loop-free
/// function on the bytecode VM a while, so its stores' caches hold the accessor when its native
/// code is made.
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

const STORES: &str = r#"
// Setter semantics through every store form, compared across tiers against Node.
var out = [];
function log(x) { out.push(String(x)); }
var N = 300;

// 1. `this` is the original receiver, not the holder; forms: operand drop, keep, this, local.
function forms() {
  class P {
    constructor(x) { this._x = x; this.seen = 0; }
    set x(v) { if (this === P.prototype) throw new Error('holder as this'); this._x = v; this.seen++; return 'ignored'; }
    get x() { return this._x; }
    setViaThis(v) { this.x = v; }
  }
  class Q extends P {}
  class R extends Q {}
  class S extends R {}
  var p = new P(0), q = new Q(0), r = new R(0), s = new S(0), t = 0, kept = '';
  function drop(o, v) { o.x = v; }
  function keep(o, v) { return o.x = v; }
  function local(v) { var o = q; for (var i = 0; i < 2; i++) o.x = v + i; return o.x; }
  for (var k = 0; k < N; k++) {
    drop(p, k); t += p.x;
    kept = keep(r, k * 2); t += r.x;
    s.setViaThis(k + 1); t += s.x;
    t += local(k);
  }
  return t + ':' + kept + ':' + [p.seen, q.seen, r.seen, s.seen].join() + ':' + (keep(p, 'z') + p.x);
}
log(forms());

// 2. Missing setter: sloppy fails silently, strict throws; the assignment's value is still V.
function missing() {
  var o = {}; Object.defineProperty(o, 'g', { get() { return 'G'; }, configurable: true });
  function sloppy(x, v) { return x.g = v; }
  function strict(x, v) { 'use strict'; try { x.g = v; return 'no throw'; } catch (e) { return e.constructor.name; } }
  class C { get ro() { return 1; } assign(v) { try { this.ro = v; return 'no throw'; } catch (e) { return e.constructor.name; } } }
  var c = new C(), a = '', b = '', d = '';
  for (var k = 0; k < N; k++) { a = sloppy(o, k); b = strict(o, k); d = c.assign(k); }
  var inherited = Object.create(o), e = '';
  for (var k = 0; k < N; k++) e = sloppy(inherited, 'v' + k) + strict(inherited, k);
  return [a, o.g, b, d, c.ro, e, Object.keys(inherited).length].join();
}
log(missing());

// 3. The assignment's value is the right-hand side, whatever the setter returns or does to it.
function values() {
  var o = { set v(x) { x.mutated = true; return 42; } };
  var last, chained, obj;
  for (var k = 0; k < N; k++) { obj = { k: k }; last = (o.v = obj); chained = o.v = o.v = k; }
  var f = { set n(x) { return 7; } };
  function inExpr(k) { return (f.n = k) + (f.n = 1); }
  var s = 0; for (var k = 0; k < N; k++) s += inExpr(k);
  return [last === obj, last.mutated, chained, s].join();
}
log(values());

// 4. A Proxy anywhere on the chain takes the generic path; the trap sees the receiver.
function proxies() {
  var calls = [];
  var target = { set x(v) { calls.push('target-setter:' + (this === target ? 'target' : this.tag)); } };
  var trapped = new Proxy({}, { set(t, key, v, receiver) { calls.push('trap:' + key + ':' + receiver.tag); return true; } });
  var forwarding = new Proxy(target, {});
  var a = Object.create(trapped); a.tag = 'a';
  var b = Object.create(forwarding); b.tag = 'b';
  var c = Object.create(Object.create(trapped)); c.tag = 'c';
  function store(o, v) { o.x = v; }
  for (var k = 0; k < N; k++) { store(a, k); store(b, k); store(c, k); }
  var direct = new Proxy({}, { set(t, k2, v) { calls.push('direct:' + v); return true; } });
  for (var k = 0; k < 3; k++) store(direct, k);
  var strictReject = new Proxy({}, { set() { return false; } });
  var rejected = (function () { 'use strict'; try { strictReject.x = 1; return 'no'; } catch (e) { return e.constructor.name; } })();
  return calls.length + ':' + calls.slice(0, 3).join('|') + ':' + calls.slice(-3).join('|') + ':' + rejected;
}
log(proxies());

// 5. A setter may throw; the error unwinds through the store (caught by the caller).
function throwing() {
  var o = { n: 0, set boom(v) { this.n++; if (this.n % 97 === 0) throw new Error('boom' + this.n + ':' + v); } };
  function store(x, v) { var arr = [1, 2, 3]; try { x.boom = v; return arr.length; } catch (e) { return e.message; } }
  var caught = []; for (var k = 0; k < N; k++) { var r = store(o, k); if (typeof r === 'string') caught.push(r); }
  class T { set bad(v) { throw new TypeError('bad ' + v); } run(v) { try { this.bad = v; } catch (e) { return e.message; } } }
  var t = new T(), last; for (var k = 0; k < N; k++) last = t.run(k);
  return caught.join(',') + ':' + o.n + ':' + last;
}
log(throwing());

// 6. A setter may reshape the receiver or replace itself: nothing cached holds afterwards.
function reshape() {
  var log2 = [];
  function C() {}
  Object.defineProperty(C.prototype, 'x', { configurable: true, set(v) {
    log2.length < 5 && log2.push('setter:' + v);
    this['f' + (v % 7)] = v;            // new shapes on the receiver
    if (v % 50 === 0) collect();
    if (v === 150) Object.defineProperty(C.prototype, 'x', { configurable: true, set(w) { this.after = w; } });
    if (v === 200) Object.setPrototypeOf(this, null);
  } });
  var c = new C();
  function store(o, v) { o.x = v; }
  for (var k = 0; k < N; k++) store(c, k);
  var keys = Object.keys(c).sort().join();
  // Own data shadowing appears later: the store writes it, no setter.
  var d = new C(); store(d, 1); Object.defineProperty(d, 'x', { value: 'own', writable: true, configurable: true });
  for (var k = 0; k < 5; k++) store(d, 'data' + k);
  // An accessor replaced by data on the prototype.
  var e = new C(); Object.defineProperty(C.prototype, 'x', { value: 'protodata', writable: true, configurable: true });
  for (var k = 0; k < 5; k++) store(e, k);
  return log2.join('|') + ':' + keys + ':' + c.after + ':' + d.x + ':' + e.x + ':' + Object.getPrototypeOf(c);
}
log(reshape());

// 7. Compound assignments and updates read through the getter and write through the setter, in
// that order.
function compound() {
  var trace = [];
  var o = { _v: 1, get v() { trace.push('get'); return this._v; }, set v(x) { trace.push('set:' + x); this._v = x; } };
  o.v += 1; o.v -= 1; o.v *= 3; o.v++; ++o.v; o.v--; --o.v; o.v **= 2; o.v |= 4; o.v >>= 1;
  var post = o.v++, pre = ++o.v;
  o.v ||= 5; o._v = 0; o.v ||= 5; o.v &&= 0; o.v ??= 9;
  var s = 0, w = { _w: 0, get w() { return this._w; }, set w(x) { this._w = x + 0.5; } };
  for (var k = 0; k < N; k++) { w.w += 1; w.w++; s += w.w; }
  var str = { _s: '', get s() { return this._s; }, set s(x) { this._s = x.slice(-5); } };
  for (var k = 0; k < N; k++) str.s += String.fromCharCode(97 + k % 26);
  return trace.join(',') + ':' + post + ':' + pre + ':' + o._v + ':' + s + ':' + str._s;
}
log(compound());

// 8. Depths 0-5, own setters of object literals, static setters, setters on Object.prototype.
function depths() {
  function chain(n) { var base = {}; Object.defineProperty(base, 'd', { set(v) { this.got = (this.got || 0) + v; } });
    var o = base; for (var i = 0; i < n; i++) o = Object.create(o); return o; }
  var objs = [0, 1, 2, 3, 4, 5, 6].map(chain), t = [];
  for (var j = 0; j < objs.length; j++) { var store = function (o, v) { o.d = v; }; for (var k = 0; k < N; k++) store(objs[j], 1); t.push(objs[j].got); }
  class St { static set s(v) { this.got = (this.got || 0) + v; } }
  function staticStore(v) { St.s = v; }
  for (var k = 0; k < N; k++) staticStore(2);
  Object.defineProperty(Object.prototype, 'zz', { configurable: true, set(v) { Object.defineProperty(this, 'zzv', { value: v, writable: true, configurable: true }); } });
  var plain = {}; function zz(o, v) { o.zz = v; } for (var k = 0; k < N; k++) zz(plain, k);
  var res = t.join() + ':' + St.got + ':' + plain.zzv + ':' + Object.keys(plain).length;
  delete Object.prototype.zz;
  return res;
}
log(depths());

// 9. Polymorphic and code-keyed setters, setters the sequence cannot enter, native setters.
function shapes() {
  class S1 { set a(v) { this.r = 1 * v; } } class S2 { set a(v) { this.r = 2 * v; } } class S3 { set a(v) { this.r = 3 * v; } }
  class S4 { set a(v) { this.r = 4 * v; } } class S5 { set a(v) { this.r = 5 * v; } }
  var poly = [new S1(), new S2(), new S3(), new S4(), new S5()], sum = 0;
  function store(o, v) { o.a = v; }
  for (var k = 0; k < 1000; k++) { var o = poly[k % 5]; store(o, k); sum += o.r; }
  function reactive(obj, key) { var val; Object.defineProperty(obj, key, { get: function () { return val; }, set: function reactiveSetter(v) { val = v * 2; }, configurable: true }); }
  var many = []; for (var i = 0; i < 20; i++) { var r = {}; reactive(r, 'v', i); many.push(r); }
  var t = 0; for (var k = 0; k < 1000; k++) { store2(many[k % 20], k); t += many[k % 20].v; }
  function store2(o, v) { o.v = v; }
  var env = { base: 5, set e(v) { var b = this.base; var f = function () { return b + v; }; this.res = f(); } };
  var u = 0; function storeE(o, v) { o.e = v; } for (var k = 0; k < N; k++) { storeE(env, k); u += env.res; }
  var protoSwap = 0; for (var k = 0; k < N; k++) { var p = {}; p.__proto__ = k % 2 ? Array.prototype : Object.prototype; protoSwap += Array.isArray(p) ? 0 : (p instanceof Array ? 1 : 0); }
  return [sum, t, u, protoSwap].join();
}
log(shapes());

// 10. The setter's call state: one argument, new.target undefined, `this` the receiver.
function callState() {
  var o = { set info(v) { this.state = [arguments.length, new.target === undefined, this === o, v].join(); } };
  function store(x) { x.info = 'v'; }
  for (var k = 0; k < N; k++) store(o);
  var p = { set nt(v) { this.ntv = new.target === undefined ? 'u' : 'd'; } };
  function K() { p.nt = 1; this.after = new.target === K; }
  var made; for (var k = 0; k < N; k++) made = new K();
  return o.state + ':' + p.ntv + made.after;
}
log(callState());

// 11. Primitive receivers: a setter on a wrapper prototype sees the primitive (strict) or its
// wrapper (sloppy); without a setter a strict store to a primitive throws.
function primitives() {
  Object.defineProperty(String.prototype, 'strictSet', { set: function (v) { 'use strict'; this.constructor.lastStrict = typeof this + v; }, configurable: true });
  Object.defineProperty(String.prototype, 'sloppySet', { set: function (v) { String.lastSloppy = typeof this + v; }, configurable: true });
  function store(s, v) { s.strictSet = v; s.sloppySet = v; }
  for (var k = 0; k < N; k++) store('abc', k);
  var thrown = (function () { 'use strict'; try { 'abc'.nope = 1; return 'no'; } catch (e) { return e.constructor.name; } })();
  var r = String.lastStrict + '/' + String.lastSloppy + '/' + thrown;
  delete String.prototype.strictSet; delete String.prototype.sloppySet;
  return r;
}
log(primitives());

// 12. Non-extensible and frozen receivers still call inherited setters.
function frozen() {
  var calls = 0; var proto = { set f(v) { calls++; } };
  var a = Object.preventExtensions(Object.create(proto)), b = Object.freeze(Object.create(proto));
  function store(o, v) { o.f = v; }
  for (var k = 0; k < N; k++) { store(a, k); store(b, k); }
  var own = Object.freeze({ set g(v) { calls += 1000; } });
  for (var k = 0; k < 3; k++) own.g = k;
  return calls;
}
log(frozen());

// 13. Recursion through setters, past the stack limit.
function deep() {
  var o = { n: 0, set down(v) { if (v <= 0) { this.n++; return; } this.down = v - 1; } };
  function descend(x, n) { x.down = n; return x.n; }
  var total = 0; for (var k = 0; k < 30; k++) total = descend(o, 40);
  var overflow = 'skipped';
  if (!stressed) { try { descend(o, 1e6); } catch (e) { overflow = e.constructor.name; } }
  return total + ':' + overflow;
}
log(deep());

// 14. Stack positions: a throw from a setter is placed at the store.
function stacks() {
  var o = {}; Object.defineProperty(o, 'bad', { set: function badSetter(v) { throw new Error('bad'); } });
  function storeBad(x) {
    x.bad = 1;
  }
  var stack = ''; for (var k = 0; k < 200; k++) { try { storeBad(o); } catch (e) { stack = e.stack; } }
  var lines = stack.split('\n'), at = /:(\d+):(\d+)\)?$/, setter = at.exec(lines[1]), storer = at.exec(lines[2]);
  return /badSetter/.test(lines[1]) + ':' + /storeBad/.test(lines[2]) + ':' + (storer[1] - setter[1]) + ':' + storer[2];
}
log(stacks());

// 15. `o.k += v` statements: numbers add in line and store through the template (data and
// setters); strings, BigInts and everything else keep the generic operation.
function appends() {
  var r = [];
  function strictAdd(x, v) { 'use strict'; x.n += v; }
  function sloppyAdd(x, v) { x.n += v; }
  var o = { n: 0 };
  for (var k = 0; k < N; k++) strictAdd(o, 0.5);
  r.push(o.n);
  var f = Object.freeze({ n: 1 });
  try { strictAdd(f, 1); r.push('no'); } catch (e) { r.push(e.constructor.name); }
  sloppyAdd(f, 1); r.push(f.n);
  var p = Object.create({ n: 10 }); strictAdd(p, 1); r.push(p.n, Object.keys(p).join());
  var z = { n: -0 }; strictAdd(z, -0); r.push(Object.is(z.n, -0));
  var nan = { n: NaN }; strictAdd(nan, 1); r.push(nan.n);
  var s = { n: 'a' }; for (var k = 0; k < 5; k++) strictAdd(s, k); r.push(s.n);
  var ro = {}; Object.defineProperty(ro, 'n', { value: 1, writable: false });
  try { strictAdd(ro, 1); r.push('no'); } catch (e) { r.push(e.constructor.name); }
  sloppyAdd(ro, 1); r.push(ro.n);
  try { strictAdd('abc', 1); r.push('no'); } catch (e) { r.push(e.constructor.name); }
  sloppyAdd('abc', 1);
  var big = { n: 1n }; strictAdd(big, 2n); r.push(String(big.n));
  try { strictAdd(big, 1); } catch (e) { r.push(e.constructor.name); }
  var obj = { n: { valueOf() { r.push('valueOf'); return 5; } } }; strictAdd(obj, 1); r.push(obj.n);
  var acc = { _n: 1, get n() { r.push('get'); return this._n; }, set n(v) { r.push('set' + v); this._n = v; } };
  for (var k = 0; k < 3; k++) strictAdd(acc, 1);
  var getOnly = { get n() { return 1; } };
  try { strictAdd(getOnly, 1); r.push('no'); } catch (e) { r.push(e.constructor.name); }
  sloppyAdd(getOnly, 1); r.push(getOnly.n);
  var t = 0, cnt = { n: 0 }; for (var k = 0; k < 10000; k++) { cnt.n += 1; t += cnt.n; } r.push(t);
  class Counter { constructor() { this.c = 0; } bump() { this.c += 2; return this.c; } }
  var cc = new Counter(), u = 0; for (var k = 0; k < N; k++) u += cc.bump(); r.push(u);
  return r.join();
}
log(appends());
out.join('\n')
"#;

const STORE_RESULTS: &str = "224850:598:300,600,300,300:zz\n\
299,G,TypeError,TypeError,1,v299TypeError,0\n\
true,true,299,45150\n\
905:trap:tag:undefined|trap:tag:undefined|trap:x:undefined:direct:0|direct:1|direct:2:TypeError\n\
boom97:96,boom194:193,boom291:290:300:bad 299\n\
setter:0|setter:1|setter:2|setter:3|setter:4:after,f0,f1,f2,f3,f4,f5,f6:299:data4:4:[object Object]\n\
get,set:2,get,set:1,get,set:3,get,set:4,get,set:5,get,set:4,get,set:3,get,set:9,get,set:13,get,set:6,get,set:7,get,set:8,get,get,set:5,get,set:0,get:6:8:0:135450:jklmn\n\
300,300,300,300,300,300,300:600:299:0\n\
1500500,999000,46350,150\n\
1,true,true,v:utrue\n\
string299/object299/TypeError\n\
3600\n\
30:RangeError\n\
true:true:2:11\n\
150,TypeError,1,11,n,true,NaN,a01234,TypeError,1,TypeError,3,TypeError,valueOf,6,get,set2,get,set3,get,set4,TypeError,1,50005000,90300";

#[test]
fn setter_stores_agree_across_tiers() {
    // Under GC stress, a collection (and in debug builds the heap's verification) at every
    // safepoint, the programs run in native code, where stores call setters directly, and
    // without the recursion to the stack limit, whose every call would scan the whole native
    // stack.
    let stressed = crate::gc_test::stress_active();
    let source = format!("var stressed = {stressed};{STORES}");
    let expected = if stressed {
        STORE_RESULTS.replace(":RangeError", ":skipped")
    } else {
        STORE_RESULTS.to_string()
    };
    let runs = [
        (Tier::Interp, false),
        (Tier::Bytecode, false),
        (Tier::Jit, false),
        (Tier::Jit, true),
    ];
    for (tier, warm) in runs.into_iter().skip(if stressed { 2 } else { 0 }) {
        let mut engine = engine(tier, warm);
        let result = eval(&mut engine, &source);
        for (line, (got, want)) in result.lines().zip(expected.lines()).enumerate() {
            assert_eq!(got, want, "{tier:?}, warm {warm}, line {}", line + 1);
        }
        assert_eq!(result, expected, "{tier:?}, warm {warm}");
    }
}

/// One class's setter, stored through each form, and a setter that throws.
#[cfg(target_arch = "aarch64")]
const FORMS: &str = r#"
class P { constructor(x) { this._x = x; } set x(v) { this._x = v * 2; } get x() { return this._x; }
  assign(v) { this.x = v; return this._x; } }
var setterX = Object.getOwnPropertyDescriptor(P.prototype, 'x').set;
var ps = [new P(3), new P(4)];
function drop(k) { ps[k & 1].x = k; return ps[k & 1]._x; }
function keep(k) { return (ps[k & 1].x = k) + ps[k & 1]._x; }
function local(k) { var o = ps[k & 1]; o.x = k; return o._x; }
function viaThis(k) { return ps[k & 1].assign(k); }
var bad = {}; Object.defineProperty(bad, 'v', { set: function badSetter(v) { throw new Error('bad'); } });
var setterBad = Object.getOwnPropertyDescriptor(bad, 'v').set;
function storeBad() {
  bad.v = 1;
}
function sum(store) { var t = 0; for (var k = 0; k < 50; k++) t += store(k); return t; }
function position() {
  var stack; for (var k = 0; k < 50; k++) { try { storeBad(); } catch (e) { stack = e.stack; } }
  return /:(\d+:\d+)\)?$/.exec(stack.split('\n')[2])[1];
}
[sum(drop), sum(keep), sum(local), sum(viaThis), position()].join()
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

/// Run `round` until the counter `count` of the setter `setter`'s chunks advances, as the
/// sequence's completions count them: a concurrent test can exhaust the process-wide call-cache
/// epoch, which sends stores through the checked helper until their sites refill.
#[cfg(target_arch = "aarch64")]
fn direct_round(
    engine: &mut Engine,
    setter: &str,
    count: fn(&crate::bytecode::Chunk) -> usize,
    round: &str,
) -> String {
    let function = function_of(engine, setter);
    let mut result = String::new();
    for _ in 0..100 {
        let before = completions(&function, count);
        result = eval(engine, round);
        if completions(&function, count) > before {
            return result;
        }
    }
    panic!("{round}: {setter} never completed through the direct sequence ({result})");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn setters_complete_through_the_direct_sequence_from_first_and_warm_code() {
    // The helper's shared stub calls the setters: in-line stubs (a diagnostic) keep the helper.
    if !super::shared_stubs_enabled() {
        return;
    }
    for warm in [false, true] {
        setters_complete_through_the_direct_sequence(warm);
    }
}

#[cfg(target_arch = "aarch64")]
fn setters_complete_through_the_direct_sequence(warm: bool) {
    let mut engine = engine(Tier::Jit, warm);
    // The store's line and column in FORMS (`bad.v = 1`'s operator, as Node reports it).
    let (line, column) = FORMS
        .lines()
        .enumerate()
        .find_map(|(index, text)| text.find("bad.v = 1;").map(|at| (index + 1, at + 7)))
        .unwrap();
    let position = format!("{line}:{column}");
    assert_eq!(
        eval(&mut engine, FORMS),
        format!("2450,3675,2450,2450,{position}"),
        "warm {warm}"
    );
    assert_native(
        &mut engine,
        &[
            "drop",
            "keep",
            "local",
            "viaThis",
            "storeBad",
            "setterX",
            "setterBad",
        ],
    );
    for setter in ["setterX", "setterBad"] {
        settle(&mut engine, setter);
    }
    let returns = |chunk: &crate::bytecode::Chunk| chunk.test_direct_returns.get();
    for (store, expected) in [
        ("drop", "2450"),
        ("keep", "3675"),
        ("local", "2450"),
        ("viaThis", "2450"),
    ] {
        let round = format!("sum({store})");
        assert_eq!(
            direct_round(&mut engine, "setterX", returns, &round),
            expected,
            "{store}, warm {warm}"
        );
    }
    let throws = |chunk: &crate::bytecode::Chunk| chunk.test_direct_throws.get();
    assert_eq!(
        direct_round(&mut engine, "setterBad", throws, "position()"),
        position,
        "a direct setter's throw is placed at the store, warm {warm}"
    );
}

/// While a setter runs through the sequence, the setter (no longer its holder's: a replacement
/// keeps the holder's shape, and so the store's cache), the receiver and the value are roots;
/// afterwards nothing keeps them, the setter's call cache included.
#[cfg(target_arch = "aarch64")]
#[test]
fn a_running_setter_its_receiver_and_value_are_roots() {
    use super::rooting_tests::{assert_tracked_collectable, jit_engine, run};
    if !super::shared_stubs_enabled() {
        return;
    }
    // `store`'s first native code, made at its first call.
    let mut engine = jit_engine(0);
    let source = r#"
        function R(armed) { this.armed = armed; }
        function other(v) { this.other = v; }
        function install() {
            Object.defineProperty(R.prototype, 's', { configurable: true, set: function self(v) {
                if (!this.armed) { this.cold = v; return; }
                Object.defineProperty(R.prototype, 's', { configurable: true, set: other });
                track(self); track(this); track(v);
                var marks = collect();
                this.result = marks + ':' + typeof self + ':' + this.armed + ':' + v.tag;
            } });
        }
        function store(o, v) { o.s = v; return o; }
        install();
        var probe = Object.getOwnPropertyDescriptor(R.prototype, 's').set;
        var warm = 0; for (var k = 0; k < 100; k++) warm = store(new R(false), k).cold;
        warm;
    "#;
    assert_eq!(run(&mut engine, source), "99");
    assert_native(&mut engine, &["store", "probe"]);
    // Every closure the rounds install runs the probe's function, whose chunk counts the
    // sequence's returns into it.
    settle(&mut engine, "probe");
    let function = function_of(&mut engine, "probe");
    run(&mut engine, "probe = null;");
    let returns = || completions(&function, |chunk| chunk.test_direct_returns.get());
    let mut result = String::new();
    for _ in 0..100 {
        let before = returns();
        result = run(
            &mut engine,
            "install(); store(new R('A'), { tag: 'V' }).result;",
        );
        if returns() > before {
            break;
        }
    }
    assert_eq!(result, "RRR:function:A:V");
    assert!(
        returns() > 0,
        "the armed store ran its setter through the sequence"
    );
    assert_tracked_collectable(&mut engine);
}

/// Stores in spliced callees: a setter called through the sequence records the store's implicit
/// inline position, as the helper's call does, so a stack captured inside it (and the stack of an
/// error it throws, which the store places) shows the same logical frames as at every other tier.
const SPLICED: &str = r#"
function trace(e) {
  return e.stack.split('\n').slice(0, 5).map(function (line) { return line.trim(); }).join('|');
}
class Q { constructor(x) { this.x = x; } set d(v) { this.x = v * 2; }
  set here(v) { this.stack = trace(new Error('here')); }
  set bad(v) { throw new Error('bad'); } }
var setterQ = Object.getOwnPropertyDescriptor(Q.prototype, 'd').set;
var setterHereQ = Object.getOwnPropertyDescriptor(Q.prototype, 'here').set;
var setterBadQ = Object.getOwnPropertyDescriptor(Q.prototype, 'bad').set;
var q = new Q(5);
function storeD(o, v) { o.d = v; }
function storeHere(o) { o.here = 1; }
function storeBad(o) { o.bad = 1; }
function spliced(k) { storeD(q, k); return q.x + k; }
function splicedHere() { storeHere(q); return q.stack; }
function splicedBad() { storeBad(q); }
var here = '', bad = '';
function drive() {
  var t = 0;
  for (var k = 0; k < 300; k++) {
    t += spliced(k);
    here = splicedHere();
    try { splicedBad(); } catch (e) { bad = trace(e); }
  }
  return t;
}
"#;

#[test]
fn spliced_stores_call_setters_with_their_inline_positions() {
    let mut stacks = Vec::new();
    for tier in [Tier::Interp, Tier::Jit] {
        let mut engine = engine(tier, false);
        eval(&mut engine, SPLICED);
        assert_eq!(eval(&mut engine, "drive()"), "134550", "{tier:?}");
        #[cfg(target_arch = "aarch64")]
        if tier == Tier::Jit && super::shared_stubs_enabled() {
            let returns = |chunk: &crate::bytecode::Chunk| chunk.test_direct_returns.get();
            let throws = |chunk: &crate::bytecode::Chunk| chunk.test_direct_throws.get();
            for (caller, setter, count) in [
                (
                    "spliced",
                    "setterQ",
                    returns as fn(&crate::bytecode::Chunk) -> usize,
                ),
                ("splicedHere", "setterHereQ", returns),
                ("splicedBad", "setterBadQ", throws),
            ] {
                let spliced = function_of(&mut engine, caller)
                    .runtime()
                    .code2
                    .get()
                    .is_some_and(Option::is_some);
                assert!(spliced, "{caller} has a second-stage compile");
                settle(&mut engine, setter);
                direct_round(&mut engine, setter, count, "drive()");
            }
        }
        stacks.push(eval(&mut engine, "here + '\\n' + bad"));
    }
    assert!(
        stacks[0].contains("at storeHere") && stacks[0].contains("at storeBad"),
        "the logical frames of the spliced stores: {}",
        stacks[0]
    );
    assert_eq!(
        stacks[0], stacks[1],
        "the stacks inside spliced stores' setters"
    );
}
