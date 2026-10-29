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

use crate::bytecode::Tier;
use crate::Engine;

use super::traced_tests::eval;

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
true:true:2:11";

#[test]
fn setter_stores_agree_across_tiers() {
    // Under GC stress, a collection (and in debug builds the heap's verification) at every
    // safepoint, the programs run in native code, and without the recursion to the stack limit,
    // whose every call would scan the whole native stack.
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
