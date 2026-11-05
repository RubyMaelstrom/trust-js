//! Proxy `get` traps called by the property reads' helper stub through the direct sequence
//! (ECMA-262 §10.5.8 [[Get]] with the Proxy as the Receiver: GetMethod(handler, "get"),
//! Call(trap, handler, «target, P, Receiver») and the invariant of step 10). The programs read
//! through every receiver form (an operand, a local, `this` in a method called on the Proxy, a
//! method read), through one handler's trap and through a new trap closure for each proxy
//! (chai's proxify), a `get` on the handler's prototype (Vue 3.4's class handlers) or an accessor
//! on the handler, no trap (the target's [[Get]]), a revoked proxy, traps that throw, collect or
//! return what the target's non-configurable properties forbid (also one the trap itself makes
//! non-configurable), arrays' `length` and methods, nested proxies, and a trap whose error is
//! placed at the read. Every tier agrees; the expected values are Node's.
//!
//! On ARM64 the reads' traps complete through the sequence (the trap chunk's direct return and
//! throw counters), from a function's first native code, made before its reads met a Proxy, as
//! from code made once they had.

use crate::bytecode::Tier;
use crate::Engine;

use super::traced_tests::eval;
#[cfg(target_arch = "aarch64")]
use super::traced_tests::{assert_native, settle};

/// The bytecode VM runs a warm engine's loop-free functions make before native code.
const WARM_RUNS: u32 = 20;

/// A fresh engine at `tier` with `collect()`, cold (each function compiled at its first call) or
/// `warm` (a loop-free function first runs on the bytecode VM a while).
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
  function reactive() { return function get(target, key, receiver) { return Reflect.get(target, key, receiver); }; }
  var handler = { get: reactive() };
  class P { constructor(x) { this.x = x; } twice() { return this.x + this.x; } get d() { return this.x * 2; } }
  var ps = [new Proxy(new P(3), handler), new Proxy(new P(4), handler)];
  function operand(k) { return [ps[k & 1]][0].x; }
  function local(k) { var o = ps[k & 1]; return o.x + o.d; }
  function viaThis(k) { return ps[k & 1].twice(); }
  var t = 0; for (var k = 0; k < 300; k++) t += operand(k) + local(k) + viaThis(k);
  return t;
}
function closures() {
  var seen = 0;
  function proxify(o) { return new Proxy(o, { get: function proxyGetter(t, k) { seen++; return Reflect.get(t, k); } }); }
  var ps = []; for (var i = 0; i < 40; i++) ps.push(proxify({ v: i, w: 'w' + i }));
  function read(p) { return p.v; }
  var t = 0; for (var k = 0; k < 400; k++) t += read(ps[k % 40]);
  return t + ':' + seen + ':' + ps[7].w;
}
function inherited() {
  class Base { get(t, k, r) { return k === 'secret' ? 'hidden' : Reflect.get(t, k, r); } }
  class Mutable extends Base { set(t, k, v, r) { return Reflect.set(t, k, v, r); } }
  var p = new Proxy({ a: 1, secret: 's' }, new Mutable());
  function read(o) { return o.a + o.secret; }
  var v; for (var k = 0; k < 200; k++) v = read(p);
  return v;
}
function accessorTrap() {
  var asked = 0, h = {};
  Object.defineProperty(h, 'get', { get() { asked++; return function (t, k) { return t[k] + 1; }; } });
  var p = new Proxy({ n: 10 }, h);
  function read(o) { return o.n; }
  var t = 0; for (var k = 0; k < 200; k++) t += read(p);
  return t + ':' + asked;
}
function forwarded() {
  var proto = { inherited: 'i' }, target = Object.create(proto); target.own = 'o';
  var p = new Proxy(target, {});
  function read(o) { return o.own + o.inherited + o.missing; }
  var v; for (var k = 0; k < 200; k++) v = read(p);
  return v;
}
function late() {
  var h = {}, p = new Proxy({ v: 1 }, h);
  function read(o) { return o.v; }
  var t = 0; for (var k = 0; k < 100; k++) t += read(p);
  h.get = function (t, k) { return 10; }; for (var k = 0; k < 100; k++) t += read(p);
  delete h.get; for (var k = 0; k < 100; k++) t += read(p);
  return t;
}
function revoked() {
  var r = Proxy.revocable({ x: 1 }, { get(t, k) { return t[k]; } });
  function read(o) { return o.x; }
  var t = 0; for (var k = 0; k < 200; k++) t += read(r.proxy);
  r.revoke();
  var e; try { read(r.proxy); } catch (err) { e = err.constructor.name; }
  return t + ':' + e;
}
function throwing() {
  var n = 0, p = new Proxy({}, { get(t, k) { n++; if (n % 97 === 0) throw new Error('trap' + n); return n; } });
  function read(o) { try { return o.anything; } catch (e) { return e.message; } }
  var caught = []; for (var k = 0; k < 300; k++) { var v = read(p); if (typeof v === 'string') caught.push(v); }
  return caught.join(',') + ':' + n;
}
function invariants() {
  var frozen = Object.freeze({ x: 1 });
  var liar = new Proxy(frozen, { get(t, k) { return k === 'x' ? 2 : 0; } });
  var honest = new Proxy(frozen, { get(t, k) { return t[k]; } });
  var writeOnly = {}; Object.defineProperty(writeOnly, 'y', { set(v) {}, configurable: false });
  var five = new Proxy(writeOnly, { get() { return 5; } });
  var none = new Proxy(writeOnly, { get() { return undefined; } });
  function readX(o) { try { return o.x; } catch (e) { return e.constructor.name; } }
  function readY(o) { try { return o.y; } catch (e) { return e.constructor.name; } }
  var out = [];
  for (var k = 0; k < 150; k++) { var a = readX(liar), b = readX(honest), c = readY(five), d = readY(none); if (k === 149) out.push(a, b, c, d); }
  return out.join(',');
}
function changedInTrap() {
  var target = { z: 0 }, made = false;
  var p = new Proxy(target, { get(t, k) {
    if (k === 'z' && !made && count === 120) { made = true; Object.defineProperty(t, 'z', { value: 1, writable: false, configurable: false }); }
    return k === 'z' ? 2 : 0; } });
  var count = 0;
  function read(o) { count++; try { return o.z; } catch (e) { return e.constructor.name; } }
  var out = []; for (var k = 0; k < 200; k++) { var v = read(p); if (k >= 118 && k <= 121) out.push(v); }
  return out.join(',');
}
function arrays() {
  var h = { get(t, k, r) { return Reflect.get(t, k, r); } };
  var a = new Proxy([1, 2, 3], h);
  function len(o) { return o.length; }
  function pushes(o) { return typeof o.push; }
  var t = 0; for (var k = 0; k < 200; k++) t += len(a);
  return t + ':' + pushes(a);
}
function collecting() {
  var p = new Proxy({ v: 'kept' }, { get(t, k, r) { var junk = []; for (var i = 0; i < 20; i++) junk.push({ i: i }); collect(); return Reflect.get(t, k, r) + junk[19].i; } });
  function read(o) { return o.v; }
  var v; for (var k = 0; k < 200; k++) v = read(p);
  return v;
}
function nested() {
  var inner = new Proxy({ x: 'deep' }, { get(t, k, r) { return Reflect.get(t, k, r) + '!'; } });
  var outer = new Proxy(inner, { get(t, k, r) { return '<' + Reflect.get(t, k, r) + '>'; } });
  function read(o) { return o.x; }
  var v; for (var k = 0; k < 200; k++) v = read(outer);
  return v;
}
function receivers() {
  var target = { get who() { return this === p ? 'proxy' : this === target ? 'target' : 'other'; } };
  var p = new Proxy(target, { get(t, k, r) { return Reflect.get(t, k, r); } });
  function read(o) { return o.who; }
  var v; for (var k = 0; k < 200; k++) v = read(p);
  return v;
}
function stacks() {
  var p = new Proxy({}, { get: function badTrap() { throw new Error('bad'); } });
  function readBad(x) {
    return x.bad;
  }
  var stack = ''; for (var k = 0; k < 200; k++) { try { readBad(p); } catch (e) { stack = e.stack; } }
  var lines = stack.split('\n'), at = /:(\d+):(\d+)\)?$/, trap = at.exec(lines[1]), reader = at.exec(lines[2]);
  return /badTrap/.test(lines[1]) + ':' + (reader[1] - trap[1]);
}
[forms(), closures(), inherited(), accessorTrap(), forwarded(), late(), revoked(), throwing(), invariants(), changedInTrap(), arrays(), collecting(), nested(), receivers(), stacks()].join('\n')
"#;

const READ_RESULTS: &str =
    "6300\n7800:400:w7\n1hidden\n2200:200\noiundefined\n1200\n200:TypeError\n\
trap97,trap194,trap291:300\nTypeError,1,TypeError,\n2,TypeError,TypeError,TypeError\n600:function\n\
kept19\n<deep!>\nproxy\ntrue:2";

#[test]
fn proxy_reads_agree_across_tiers() {
    let runs = [
        (Tier::Interp, false),
        (Tier::Bytecode, false),
        (Tier::Jit, false),
        (Tier::Jit, true),
    ];
    // Under GC stress, a collection (and in debug builds the heap's verification) at every
    // safepoint: the native tiers only.
    let stressed = crate::gc_test::stress_active();
    for (tier, warm) in runs.into_iter().skip(if stressed { 2 } else { 0 }) {
        let mut engine = engine(tier, warm);
        assert_eq!(
            eval(&mut engine, READS),
            READ_RESULTS,
            "{tier:?}, warm {warm}"
        );
    }
}

/// A read whose traps are closures made for each Proxy (chai's proxify) pins only the traps its
/// call cache's ways record, and a collection forgets a dead one's identity and drops its pin
/// (`bytecode::TrapCalls::prune_dead`): no trap call cache stays on the interpreter's list once
/// its traps are garbage, and the read keeps answering, its ways matched by code.
#[test]
fn a_reads_trap_pins_go_with_their_closures() {
    const CLOSURES: &str = r#"
function make(i) { return new Proxy({ x: i }, { get: function proxyGetter(t, k, r) { return Reflect.get(t, k, r) * 2; } }); }
function read(p) { return p.x; }
function round() { var t = 0; for (var n = 0; n < 2000; n++) t += read(make(n)); return t; }
round()
"#;
    for (tier, warm) in [(Tier::Jit, false), (Tier::Jit, true)] {
        let mut engine = engine(tier, warm);
        assert_eq!(
            eval(&mut engine, CLOSURES),
            "3998000",
            "{tier:?}, warm {warm}"
        );
        engine.interp.gc_collect();
        engine.interp.gc_collect();
        let listed = engine
            .interp
            .trap_calls
            .iter()
            .filter(|calls| calls.upgrade().is_some())
            .count();
        assert_eq!(
            listed, 0,
            "{tier:?}, warm {warm}: trap caches still pinning"
        );
        assert_eq!(
            eval(&mut engine, "round()"),
            "3998000",
            "{tier:?}, warm {warm}"
        );
    }
}

/// Reflect.get, has and set from generated code, which answers a call on an Object with a key
/// String that carries its key without the native when the resolution runs no author code
/// (`Interp::reflect_resolved`). The keys are such Strings (`K`, a property's key as Object.keys
/// gives it back; a literal is not one) but where a case says otherwise. Reads: own, inherited,
/// absent and index properties, an Array's `length`, holes and methods, getters (with and
/// without a receiver), a Proxy target, a frozen target, keys that carry no key (a literal, a
/// concatenation, a Symbol), a property that becomes an accessor and then moves to the
/// prototype, and the TypeErrors of a primitive target. Stores: an own property through the
/// target and through a Proxy of it without traps, the Proxies whose defineProperty and
/// getOwnPropertyDescriptor traps OrdinarySet calls, other receivers (an object, another
/// target's Proxy, a revoked Proxy, a primitive), creation (refused when not extensible), the
/// chain's read-only property, setter and writable data, read-only and accessor own
/// properties, elements and Arrays' `length` (an append, a hole, truncation, an invalid length,
/// a String, a read-only `length`), a typed array; and reactive Arrays and objects as Vue's
/// handlers make them, whose traps call Reflect with their own key.
const REFLECTS: &str = r#"
function marked(names) {
  var o = {};
  for (var n = 0; n < names.length; n++) o[names[n]] = 0;
  var m = {};
  Object.keys(o).forEach(function (k) { m[k] = k; });
  return m;
}
var K = marked(['own', 'inherited', 'missing', 'viaGetter', '0', '1', '2', '3', '5', '6', 'length', 'extra', 'push',
  'description', 'toString', 'x', 'a', 'fresh', 'yes', 'no', 'roUp', 'setUp', 'dataUp']);
function gets() {
  var proto = { inherited: 'i', get viaGetter() { return 'g:' + (this === receiver ? 'r' : 'o'); } };
  var o = Object.create(proto); o.own = 'o'; o[3] = 'three';
  var receiver = { tag: 'receiver' };
  var arr = [10, , 30]; arr.extra = 'e';
  var key = 'ow' + 'n', sym = Symbol('s'); o[sym] = 'symbol';
  var p = new Proxy({ own: 'proxied' }, { get(t, k, r) { return 'trap:' + String(k); } });
  var frozen = Object.freeze({ own: 'frozen' });
  function g2(t, k) { return Reflect.get(t, k); }
  function g3(t, k, r) { return Reflect.get(t, k, r); }
  var out = [];
  for (var n = 0; n < 200; n++) {
    var row = [g2(o, K.own), g2(o, K.inherited), g2(o, K.missing), g3(o, K.viaGetter, receiver), g2(o, K.viaGetter),
      g2(o, K['3']), g2(arr, K.length), g2(arr, K['0']), g2(arr, K['1']), g2(arr, K['2']), g2(arr, K.extra),
      g2(arr, K.push) === Array.prototype.push, g2(o, key), g2(o, sym), g2(p, K.own), g3(p, K.x, o), g2(frozen, K.own),
      g2(o, K.description), g2(o, K.toString) === Object.prototype.toString, g2(o, 'own'), g2(arr, '2')];
    if (n === 199) out = row;
  }
  var errors = [];
  try { Reflect.get(1, K.x); } catch (e) { errors.push(e.constructor.name); }
  try { g2(null, K.x); } catch (e) { errors.push(e.constructor.name); }
  return out.join(',') + ';' + errors.join(',');
}
function hases() {
  var proto = { inherited: 1 }, o = Object.create(proto); o.own = 1; o[5] = 1;
  var arr = [1, , 3];
  var p = new Proxy({}, { has(t, k) { return k === 'yes'; } });
  function h(t, k) { return Reflect.has(t, k); }
  var out = [];
  for (var n = 0; n < 200; n++) {
    var row = [h(o, K.own), h(o, K.inherited), h(o, K.missing), h(o, K['5']), h(o, K['6']), h(arr, K['0']), h(arr, K['1']),
      h(arr, K['2']), h(arr, K.length), h(arr, K.push), h(p, K.yes), h(p, K.no), h(o, K.toString), h(o, 'ow' + 'n'), h(o, 'own')];
    if (n === 199) out = row;
  }
  var errors = [];
  try { h(1, K.x); } catch (e) { errors.push(e.constructor.name); }
  return out.join(',') + ';' + errors.join(',');
}
function mutations() {
  var o = { a: 1 }, t = 0;
  function g(x) { return Reflect.get(x, K.a); }
  for (var n = 0; n < 100; n++) t += g(o);
  Object.defineProperty(o, 'a', { get() { return 100; }, configurable: true });
  for (var n = 0; n < 100; n++) t += g(o);
  delete o.a; Object.prototype.a = 7;
  for (var n = 0; n < 100; n++) t += g(o);
  delete Object.prototype.a;
  return t + ':' + g(o);
}
function sets() {
  var out = [], log = [];
  var o = { a: 0 }, plain = new Proxy(o, {}), getOnly = new Proxy(o, { get(t, k, r) { return Reflect.get(t, k, r); } });
  var defining = new Proxy(o, { defineProperty(t, k, d) { log.push('define:' + k + '=' + d.value); return Reflect.defineProperty(t, k, d); } });
  var describing = new Proxy(o, { getOwnPropertyDescriptor(t, k) { log.push('describe:' + k); return Reflect.getOwnPropertyDescriptor(t, k); } });
  var other = { a: 'other' }, otherProxy = new Proxy(other, {});
  var revocable = Proxy.revocable(o, {}); revocable.revoke();
  function s3(t, k, v) { return Reflect.set(t, k, v); }
  function s4(t, k, v, r) { return Reflect.set(t, k, v, r); }
  var t = 0;
  for (var n = 0; n < 200; n++) t += s3(o, K.a, n) + s4(o, K.a, n, o) + s4(o, K.a, n + 1, plain) + s4(o, K.a, n + 2, getOnly);
  out.push(t + ':' + o.a);
  for (var n = 0; n < 2; n++) out.push(s4(o, K.a, 'd' + n, defining), s4(o, K.a, 'g' + n, describing));
  out.push(log.join('/') + ':' + o.a);
  out.push(s4(o, K.a, 'r', other) + ':' + o.a + ':' + other.a);
  out.push(s4(o, K.a, 'q', otherProxy) + ':' + o.a + ':' + other.a);
  try { s4(o, K.a, 'x', revocable.proxy); } catch (e) { out.push(e.constructor.name); }
  out.push(s4(o, K.a, 'prim', 1));
  var fresh = {}, closed = Object.preventExtensions({});
  out.push(s3(fresh, K.fresh, 1) + ':' + s4(fresh, K.x, 2, new Proxy(fresh, {})) + ':' + Object.keys(fresh).join('|') + ':' +
    s3(closed, K.fresh, 1) + ':' + Object.keys(closed).length);
  var proto = { dataUp: 'proto' };
  Object.defineProperty(proto, 'roUp', { value: 'ro', writable: false });
  Object.defineProperty(proto, 'setUp', { set(v) { this.tagged = v; } });
  var child = Object.create(proto), childProxy = new Proxy(child, {});
  out.push([s3(child, K.roUp, 1), s3(child, K.setUp, 'set'), child.tagged, s4(child, K.setUp, 'viaProxy', childProxy), child.tagged,
    s3(child, K.dataUp, 'own'), child.dataUp, proto.dataUp].join(','));
  var frozen = Object.freeze({ a: 1 }), acc = { get a() { return 'acc'; }, set a(v) { log.push('acc:' + v); } };
  out.push(s3(frozen, K.a, 2) + ':' + frozen.a + ':' + s3(acc, K.a, 'v') + ':' + log[log.length - 1]);
  var arr = [0, 1, 2], arrProxy = new Proxy(arr, {});
  out.push([s3(arr, K['1'], 'one'), s4(arr, K['3'], 'three', arrProxy), s4(arr, K.length, 4, arrProxy), s3(arr, K['5'], 'five'),
    arr.length, s3(arr, K.length, 2), arr.length, arr.join('|'), s3(arr, K.length, 3), arr.length, String(arr[2])].join(','));
  try { s3(arr, K.length, 1.5); } catch (e) { out.push(e.constructor.name + ':' + arr.length); }
  out.push(s3(arr, K.length, '1') + ':' + arr.length);
  var fixed = [1, 2]; Object.defineProperty(fixed, 'length', { writable: false });
  out.push(s3(fixed, K.length, 2) + ':' + s3(fixed, K['2'], 3) + ':' + fixed.length);
  var ta = new Uint8Array(2), sym = Symbol('s'), so = {};
  out.push(s3(ta, K['1'], 7) + ':' + ta[1] + ':' + s3(ta, K['5'], 7) + ':' + s3(so, 'a', 1) + ':' + s3(so, sym, 2) + ':' + so.a + ':' + so[sym]);
  try { s3(1, K.a, 1); } catch (e) { out.push(e.constructor.name); }
  return out.join(';');
}
function reactive() {
  var log = [];
  var handler = {
    get(t, k, r) { return Reflect.get(t, k, r); },
    set(t, k, v, r) { log.push(String(k)); return Reflect.set(t, k, v, r); },
    has(t, k) { return Reflect.has(t, k); }
  };
  var list = new Proxy([], handler), sum = 0;
  for (var n = 0; n < 100; n++) list.push(n);
  for (var n = 0; n < 50; n++) list.pop();
  list.forEach(function (v) { sum += v; });
  var obj = new Proxy({ count: 0 }, handler);
  for (var n = 0; n < 100; n++) obj.count++;
  return list.length + ':' + sum + ':' + obj.count + ':' + log.length + ':' + ('count' in obj) + ':' + (5 in list) + ':' + (60 in list);
}
[gets(), hases(), mutations(), sets(), reactive()].join('\n')
"#;

const REFLECT_RESULTS: &str = "\
o,i,,g:r,g:o,three,3,10,,30,e,true,o,symbol,trap:own,trap:x,frozen,,true,o,30;TypeError,TypeError\n\
true,true,false,true,false,true,false,true,true,true,true,false,true,true,true;TypeError\n\
10800:undefined\n\
800:201;true;true;true;true;define:a=d0/describe:a/define:a=d1/describe:a:g1;true:g1:r;true:g1:q;TypeError;false;\
true:true:fresh|x:false:0;false,true,set,true,viaProxy,true,own,proto;false:1:true:acc:v;\
true,true,true,true,6,true,2,0|one,true,3,undefined;RangeError:3;true:1;false:false:2;true:7:true:true:true:1:2;TypeError\n\
50:1225:100:350:true:true:false";

#[test]
fn reflect_calls_agree_across_tiers() {
    let runs = [
        (Tier::Interp, false),
        (Tier::Bytecode, false),
        (Tier::Jit, false),
        (Tier::Jit, true),
    ];
    let stressed = crate::gc_test::stress_active();
    for (tier, warm) in runs.into_iter().skip(if stressed { 2 } else { 0 }) {
        let mut engine = engine(tier, warm);
        let before = crate::interpreter::TEST_REFLECT_RESOLVED.with(std::cell::Cell::get);
        assert_eq!(
            eval(&mut engine, REFLECTS),
            REFLECT_RESULTS,
            "{tier:?}, warm {warm}"
        );
        if tier == Tier::Jit {
            assert!(
                crate::interpreter::TEST_REFLECT_RESOLVED.with(std::cell::Cell::get) > before,
                "{tier:?}, warm {warm}: calls answered on their operands"
            );
            // Each kind of call is answered: 300 of each, all but a call cache's warm-up.
            for (script, result) in RESOLVED_KINDS {
                let before = crate::interpreter::TEST_REFLECT_RESOLVED.with(std::cell::Cell::get);
                assert_eq!(eval(&mut engine, script), *result, "{tier:?}, warm {warm}");
                let resolved =
                    crate::interpreter::TEST_REFLECT_RESOLVED.with(std::cell::Cell::get) - before;
                assert!(
                    resolved >= 250,
                    "{tier:?}, warm {warm}: {resolved} of {script} answered on their operands"
                );
            }
        }
    }
}

/// One kind of Reflect call 300 times, with a key that carries its key, through a Proxy of the
/// target as Vue's traps pass it, each answered on its operands once the call cache proves the
/// native.
const RESOLVED_KINDS: &[(&str, &str)] = &[
    (
        "(function () { var o = { a: 1 }, p = new Proxy(o, {}), k = Object.keys(o)[0], t = 0; \
         function g(t, k, r) { return Reflect.get(t, k, r); } \
         for (var n = 0; n < 300; n++) t += g(o, k, p); return t; })()",
        "300",
    ),
    (
        "(function () { var a = [1, 2], k = Object.keys(a)[1], t = 0; \
         function h(t, k) { return Reflect.has(t, k); } \
         for (var n = 0; n < 300; n++) t += h(a, k); return t; })()",
        "300",
    ),
    (
        "(function () { var o = { a: 0 }, p = new Proxy(o, {}), k = Object.keys(o)[0]; \
         function s(t, k, v, r) { return Reflect.set(t, k, v, r); } \
         for (var n = 0; n < 300; n++) s(o, k, n, p); return o.a; })()",
        "299",
    ),
];

/// One handler's trap and a throwing trap, read through each receiver form.
#[cfg(target_arch = "aarch64")]
const FORMS: &str = r#"
function reactive() { return function get(target, key, receiver) { return Reflect.get(target, key, receiver); }; }
var trapGet = reactive();
var handler = { get: trapGet };
class P { constructor(x) { this.x = x; } twice() { return this.x + this.x; } }
var ps = [new Proxy(new P(3), handler), new Proxy(new P(4), handler)];
function operand(k) { return [ps[k & 1]][0].x; }
function local(k) { var o = ps[k & 1]; return o.x; }
function viaThis(k) { return ps[k & 1].twice(); }
var trapBad = function badTrap() { throw new Error('bad'); };
var bad = new Proxy({}, { get: trapBad });
function readBad() {
  return bad.v;
}
function sum(read) { var t = 0; for (var k = 0; k < 50; k++) t += read(k); return t; }
function position() {
  var stack; for (var k = 0; k < 50; k++) { try { readBad(); } catch (e) { stack = e.stack; } }
  return /:(\d+:\d+)\)?$/.exec(stack.split('\n')[2])[1];
}
[sum(operand), sum(local), sum(viaThis), position()].join()
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
    let crate::value::ObjectKind::User(user) = &object.kind else {
        panic!("{name} is not a user function")
    };
    user.func.clone()
}

/// Run `round` until the counter `count` of the trap `trap`'s chunks advances, as the
/// sequence's completions count them: a concurrent test can exhaust the process-wide call-cache
/// epoch, which sends reads through the checked helper until their sites refill.
#[cfg(target_arch = "aarch64")]
fn direct_round(
    engine: &mut Engine,
    trap: &str,
    count: fn(&crate::bytecode::Chunk) -> usize,
    round: &str,
) -> String {
    let function = function_of(engine, trap);
    let mut result = String::new();
    for _ in 0..100 {
        let before = completions(&function, count);
        result = eval(engine, round);
        if completions(&function, count) > before {
            return result;
        }
    }
    panic!("{round}: {trap} never completed through the direct sequence ({result})");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn traps_complete_through_the_direct_sequence_from_first_and_warm_code() {
    // The helper's shared stub calls the traps: in-line stubs (a diagnostic) keep the helper.
    if !super::shared_stubs_enabled() {
        return;
    }
    for warm in [false, true] {
        traps_complete_through_the_direct_sequence(warm);
    }
}

#[cfg(target_arch = "aarch64")]
fn traps_complete_through_the_direct_sequence(warm: bool) {
    let mut engine = engine(Tier::Jit, warm);
    // The read's line and column in FORMS (`bad.v`'s property name, as Node reports it).
    let (line, column) = FORMS
        .lines()
        .enumerate()
        .find_map(|(index, text)| text.find("return bad.v;").map(|at| (index + 1, at + 12)))
        .unwrap();
    let position = format!("{line}:{column}");
    assert_eq!(
        eval(&mut engine, FORMS),
        format!("175,175,350,{position}"),
        "warm {warm}"
    );
    assert_native(
        &mut engine,
        &[
            "operand", "local", "viaThis", "readBad", "trapGet", "trapBad",
        ],
    );
    for trap in ["trapGet", "trapBad"] {
        settle(&mut engine, trap);
    }
    let returns = |chunk: &crate::bytecode::Chunk| chunk.test_direct_returns.get();
    for (reader, expected) in [("operand", "175"), ("local", "175"), ("viaThis", "350")] {
        let round = format!("sum({reader})");
        assert_eq!(
            direct_round(&mut engine, "trapGet", returns, &round),
            expected,
            "{reader}, warm {warm}"
        );
    }
    let throws = |chunk: &crate::bytecode::Chunk| chunk.test_direct_throws.get();
    assert_eq!(
        direct_round(&mut engine, "trapBad", throws, "position()"),
        position,
        "a direct trap's throw is placed at the read, warm {warm}"
    );
}
