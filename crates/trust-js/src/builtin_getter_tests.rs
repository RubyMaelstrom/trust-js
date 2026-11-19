//! Built-in and host accessors read through the property caches (P3 stage 3, the getter line).
//! a1: a typed-array receiver fills a site's ways like an ordinary one where the name is no
//! CanonicalNumericIndexString (ECMA-262 10.4.5.5: [[Get]] is then OrdinaryGet), on ways that
//! record the receiver's kind (`IC_RECEIVER_TYPED_ARRAY`), the one kind of way a probe admits a
//! typed array on; no absence way is made for one. a2: a native getter a way finds is called
//! directly with the receiver (`Interp::try_native_getter`), except another Realm's.
//!
//! The programs' expected lines are Node's (v24, the same programs at N = 300; the line's
//! evidence directory keeps them at N = 3000 for the four configurations of the release binary):
//! the getter replaced on %TypedArray%.prototype or shadowed on a concrete prototype, the
//! prototype changed, an own `length`, subclasses, a Proxy on the chain and as the receiver, a
//! receiver that is no typed array inheriting the getter, numeric-looking names, a detached
//! buffer, a resizable one shrunk and grown under fixed and length-tracking views, native getters
//! that throw and re-enter, and another Realm's typed arrays and natives; and native setters
//! (a3: the store's way, the setter called directly with the receiver and the value): one that
//! throws, `__proto__`'s, another Realm's, a getter-only accessor in strict code, one that
//! re-enters, and one whose behavior a `new.target` would change.

use super::*;
use crate::bytecode::{
    IcState, Tier, IC_ACCESSOR, IC_EMPTY, IC_MID_GENERIC, IC_MID_PRIMITIVE, IC_MID_TYPED_ARRAY,
    PROP_IC_WAYS,
};
use crate::value::ValueView;
use crate::{Completion, Engine};
use std::cell::Cell;

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn engine(tier: Tier, threshold: u32) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(threshold);
    engine
}

fn native_getters() -> usize {
    property_read::TEST_NATIVE_GETTERS.with(Cell::get)
}

fn global(engine: &mut Engine, name: &str) -> Value {
    let env = engine.interp.global_env;
    engine.interp.get_var(name, &env).ok().expect("a global")
}

fn read(engine: &mut Engine, base: &Value, name: &str, cache: &Cell<IcState>) -> Value {
    engine
        .interp
        .get_prop_ic(base, name, crate::atom::PropKey::from_text(name), cache)
        .ok()
        .expect("the read completes")
}

/// The programs' common functions (`log` keeps the line as the completion value).
const PRELUDE: &str = r#"var log = function (s) { globalThis.__out = s; };
var N = 300;
function sumOf(f, a) { var s = 0; for (var i = 0; i < N; i++) s += f(a); return s; }
function len(a) { return a.length; }
function blen(a) { return a.byteLength; }
function boff(a) { return a.byteOffset; }
function size(a) { return a.size; }
function tryRead(f, a) { try { return String(f(a)); } catch (e) { return e.constructor.name; } }
var TAP = Object.getPrototypeOf(Uint8Array.prototype);
"#;

const REPLACED: (&str, &str) = (
    r#"var ta = new Uint8Array(10), out = [];
out.push(sumOf(len, ta));
var d = Object.getOwnPropertyDescriptor(TAP, 'length');
Object.defineProperty(TAP, 'length', { get: function () { return 7; }, configurable: true });
out.push(sumOf(len, ta));
Object.defineProperty(TAP, 'length', { get: Object.getOwnPropertyDescriptor(Map.prototype, 'size').get, configurable: true });
out.push(tryRead(len, ta));
Object.defineProperty(TAP, 'length', { get: Object.getOwnPropertyDescriptor(TAP, 'byteLength').get, configurable: true });
out.push(sumOf(len, new Float64Array(3)));
Object.defineProperty(TAP, 'length', { get: undefined, set: function (v) {}, configurable: true });
out.push(String(sumOf(len, ta)));
Object.defineProperty(TAP, 'length', d);
out.push(sumOf(len, ta));
var bd = Object.getOwnPropertyDescriptor(TAP, 'byteLength');
Object.defineProperty(TAP, 'byteLength', { value: 99, configurable: true });
out.push(sumOf(blen, new Float64Array(3)));
Object.defineProperty(Uint8Array.prototype, 'length', { get: function () { return this.byteOffset + 1000; }, configurable: true });
out.push(sumOf(len, ta), sumOf(len, new Int8Array(2)));
delete Uint8Array.prototype.length;
out.push(sumOf(len, ta));
Object.defineProperty(TAP, 'byteLength', bd);
log('replaced ' + out.join(' '));
"#,
    "replaced 3000 2100 TypeError 7200 NaN 3000 29700 300000 600 3000",
);

const PROTO: (&str, &str) = (
    r#"var ta = new Uint8Array(10), out = [];
out.push(sumOf(len, ta));
Object.setPrototypeOf(ta, Int8Array.prototype);
out.push(sumOf(len, ta));
Object.setPrototypeOf(ta, { length: 3 });
out.push(sumOf(len, ta));
Object.setPrototypeOf(ta, null);
out.push(String(sumOf(len, ta)));
Object.setPrototypeOf(ta, Uint8Array.prototype);
out.push(sumOf(len, ta));
var mixed = [new Uint8Array(1), new Int16Array(2), new Float32Array(3), new Uint8ClampedArray(4), new BigInt64Array(5), new Float64Array(6)];
var s = 0; for (var i = 0; i < N; i++) s += mixed[i % 6].length + mixed[i % 6].byteLength + mixed[i % 6].byteOffset;
out.push(s);
log('proto ' + out.join(' '));
"#,
    "proto 3000 3000 900 NaN 3000 6500",
);

const SHADOW: (&str, &str) = (
    r#"var ta = new Uint8Array(10), out = [];
out.push(sumOf(len, ta));
Object.defineProperty(ta, 'length', { value: 42, configurable: true });
out.push(sumOf(len, ta));
delete ta.length;
out.push(sumOf(len, ta));
class U extends Uint8Array {}
class V extends Uint8Array { get length() { return -1; } }
class W extends U {}
out.push(sumOf(len, new U(5)), sumOf(len, new V(5)), sumOf(len, new W(6)));
var all = [ta, new U(5), new V(5), new W(6)], s = 0;
for (var i = 0; i < N; i++) s += all[i & 3].length;
out.push(s);
ta.foo = 3; var t = 0; for (var i = 0; i < N; i++) t += ta.foo; out.push(t);
log('shadow ' + out.join(' '));
"#,
    "shadow 3000 12600 3000 1500 -300 1800 1500 900",
);

const PROXY: (&str, &str) = (
    r#"var ta = new Uint8Array(10), out = [];
out.push(sumOf(len, ta));
var traps = 0;
Object.setPrototypeOf(ta, new Proxy(Uint8Array.prototype, { get(t, k, r) { if (k === 'length') { traps++; return 5; } return Reflect.get(t, k, r); } }));
out.push(sumOf(len, ta), traps);
var p = new Proxy(new Uint8Array(4), {});
var tp = new Proxy(new Uint8Array(4), { get(t, k, r) { return k === 'length' ? 1 : Reflect.get(t, k, r); } });
var warm = new Uint8Array(4), pr = [warm, tp], ps = 0;
for (var i = 0; i < N; i++) ps += len(warm);
for (var i = 0; i < N; i++) ps += len(pr[i & 1]);
out.push(ps);
out.push(tryRead(len, p));
log('proxy ' + out.join(' '));
"#,
    "proxy 3000 1500 300 1950 TypeError",
);

const RECEIVER: (&str, &str) = (
    r#"var out = [];
var fake = Object.create(Uint8Array.prototype);
out.push(tryRead(len, fake), tryRead(blen, fake), tryRead(boff, fake));
var ta = new Uint8Array(10);
var onTop = Object.create(ta);
out.push(tryRead(len, onTop));
var both = [ta, fake], s = 0, errs = 0;
for (var i = 0; i < N; i++) { try { s += both[i & 1].length; } catch (e) { errs++; } }
out.push(s, errs);
var both2 = [ta, Object.setPrototypeOf({}, Uint8Array.prototype)], s2 = 0, errs2 = 0;
for (var i = 0; i < N; i++) { try { s2 += both2[i & 1].length + both2[i & 1].byteOffset; } catch (e) { errs2++; } }
out.push(s2, errs2);
var m = new Map([[1, 2]]); var notMap = Object.create(Map.prototype);
out.push(sumOf(size, m), tryRead(size, notMap));
var keys = ['-0', '1.5', 'NaN', 'Infinity', '1e+21'], r = [];
for (var k of keys) { Uint8Array.prototype[k] = 7; var plain = Object.create(Uint8Array.prototype); var z = 0, w = 0;
  for (var i = 0; i < 200; i++) { z += (ta[k] === undefined ? 1 : 0); w += plain[k]; } r.push(z + '/' + w); }
out.push(r.join(','));
function rdNaN(o) { return o.NaN; } function rdInf(o) { return o.Infinity; }
var pl = Object.setPrototypeOf({}, Uint8Array.prototype), q = 0;
for (var i = 0; i < N; i++) { q += rdNaN(pl) + rdInf(pl); if (rdNaN(ta) !== undefined || rdInf(ta) !== undefined) q += 1000000; }
out.push(q);
log('receiver ' + out.join(' '));
"#,
    "receiver TypeError TypeError TypeError TypeError 1500 150 1500 150 300 TypeError 200/1400,200/1400,200/1400,200/1400,200/1400 4200",
);

const DETACH: (&str, &str) = (
    r#"var out = [];
var ab = new ArrayBuffer(16), ta = new Uint16Array(ab, 4, 4);
out.push(sumOf(len, ta), sumOf(blen, ta), sumOf(boff, ta));
var moved = ab.transfer();
out.push(sumOf(len, ta), sumOf(blen, ta), sumOf(boff, ta), moved.byteLength, ab.byteLength, ab.detached);
var rab = new ArrayBuffer(16, { maxByteLength: 64 });
var fixed = new Uint16Array(rab, 4, 4), track = new Uint16Array(rab, 4);
function all(a) { return a.length * 10000 + a.byteLength * 100 + a.byteOffset; }
out.push(sumOf(all, fixed), sumOf(all, track));
rab.resize(9);
out.push(sumOf(all, fixed), sumOf(all, track));
rab.resize(2);
out.push(sumOf(all, fixed), sumOf(all, track));
rab.resize(64);
out.push(sumOf(all, fixed), sumOf(all, track));
var gsab = typeof SharedArrayBuffer === 'function' ? new Int32Array(new SharedArrayBuffer(32)) : new Int32Array(8);
out.push(sumOf(all, gsab), sumOf(all, new BigUint64Array(3)));
log('detach ' + out.join(' '));
"#,
    "detach 1200 2400 1200 0 0 0 16 0 true 12241200 18361200 0 6121200 0 0 12241200 91801200 24960000 9720000",
);

const THROW: (&str, &str) = (
    r#"var out = [];
var P = Object.create(null);
Object.defineProperty(P, 'size', Object.getOwnPropertyDescriptor(Map.prototype, 'size'));
var o = Object.create(P), errs = 0;
for (var i = 0; i < N; i++) { try { o.size; } catch (e) { if (e instanceof TypeError) errs++; } }
out.push(errs);
var calls = 0;
var Q = {}; Object.defineProperty(Q, 'joined', { get: Array.prototype.join });
var al = Object.create(Q); al.length = 2; al[0] = { toString() { calls++; return 'a'; } }; al[1] = 'b';
var s = ''; for (var i = 0; i < N; i++) s = al.joined;
out.push(s, calls);
var R = {}; Object.defineProperty(R, 'boom', { get: Function.prototype.call });
var depth = 0; var rr = Object.create(R);
out.push(tryRead(function (a) { return a.boom; }, rr));
var T = {}; Object.defineProperty(T, 'str', { get: Object.prototype.toString });
out.push(sumOf(function (a) { return a.str.length; }, Object.create(T)));
var D = {}; Object.defineProperty(D, 'd', { get: Date });
function F() { var o = Object.create(D); this.r = typeof o.d; }
var types = ''; for (var i = 0; i < N; i++) types = new F().r;
out.push(types);
log('throw ' + out.join(' '));
"#,
    "throw 300 a,b 300 TypeError 4500 string",
);

const REALM: (&str, &str) = (
    r#"var out = [];
var other = typeof $262 !== 'undefined' ? $262.createRealm().global : require('vm').runInNewContext('this');
var ota = new other.Uint8Array(6), mine = new Uint8Array(3);
out.push(sumOf(len, ota), sumOf(len, mine));
var both = [ota, mine], s = 0; for (var i = 0; i < N; i++) s += both[i & 1].length + both[i & 1].byteLength;
out.push(s);
var oget = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(other.Uint8Array.prototype), 'length').get;
var fakeO = Object.create(other.Uint8Array.prototype);
var e1 = 0; for (var i = 0; i < N; i++) { try { fakeO.length; } catch (e) { if (e instanceof other.TypeError && !(e instanceof TypeError)) e1++; } }
var e2; try { oget.call({}); } catch (e) { e2 = (e instanceof other.TypeError) + '/' + (e instanceof TypeError); }
out.push(e1, e2);
var om = new other.Map([[1, 1], [2, 2]]); out.push(sumOf(size, om));
var fm = Object.create(other.Map.prototype); var e3; try { fm.size; } catch (e) { e3 = (e instanceof other.TypeError) + '/' + (e instanceof TypeError); }
out.push(e3);
log('realm ' + out.join(' '));
"#,
    "realm 1800 900 2700 300 true/false 600 true/false",
);

const SETTER: (&str, &str) = (
    r#"var out = [];
var P = {}; Object.defineProperty(P, 'x', { set: Function.prototype.call, configurable: true });
var o = Object.create(P), errs = 0;
for (var i = 0; i < N; i++) { try { o.x = i; } catch (e) { if (e instanceof TypeError) errs++; } }
out.push(errs);
var a = {}, b = {}, protos = [a, b], c = 0;
for (var i = 0; i < N; i++) { var t = {}; t.__proto__ = protos[i & 1]; if (Object.getPrototypeOf(t) === protos[i & 1]) c++; }
out.push(c);
var other = typeof $262 !== 'undefined' ? $262.createRealm().global : require('vm').runInNewContext('this');
var Q = {}; Object.defineProperty(Q, 'y', { set: other.Function.prototype.call });
var q = Object.create(Q), re = 0;
for (var i = 0; i < N; i++) { try { q.y = 1; } catch (e) { if (e instanceof other.TypeError && !(e instanceof TypeError)) re++; } }
out.push(re);
var G = {}; Object.defineProperty(G, 'z', { get: Date });
function sset(o) { 'use strict'; o.z = 1; }
var g = Object.create(G), se = 0;
for (var i = 0; i < N; i++) { try { sset(g); } catch (e) { if (e instanceof TypeError) se++; } }
out.push(se);
var W = {}; Object.defineProperty(W, 'w', { set: Array.prototype.push });
var wo = Object.create(W);
for (var i = 0; i < N; i++) wo.w = i;
out.push(wo.length, wo[N - 1]);
var D = {}; Object.defineProperty(D, 'd', { set: Date });
function F() { var o = Object.create(D); o.d = 1; this.ok = 1; }
var made = 0; for (var i = 0; i < N; i++) made += new F().ok;
out.push(made);
log('setter ' + out.join(' '));
"#,
    "setter 300 300 300 300 300 299 300",
);

const PROGRAMS: [(&str, (&str, &str)); 9] = [
    ("replaced", REPLACED),
    ("proto", PROTO),
    ("shadow", SHADOW),
    ("proxy", PROXY),
    ("receiver", RECEIVER),
    ("detach", DETACH),
    ("throw", THROW),
    ("realm", REALM),
    ("setter", SETTER),
];

#[test]
fn builtin_accessor_programs_agree_with_node_in_every_tier() {
    for (name, (body, expected)) in PROGRAMS {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            for threshold in [0, 32] {
                let mut engine = engine(tier, threshold);
                let source = format!("{PRELUDE}{body}\n__out");
                assert_eq!(
                    evaluate(&mut engine, &source),
                    expected,
                    "{name} at {tier:?}, threshold {threshold}"
                );
            }
        }
    }
}

/// a1's fill: a typed array's `length` is an accessor way two hops up whose kind is the typed
/// array's; a numeric-looking name fills nothing and marks the site generic; a way a plain
/// receiver of the same shape and chain filled never answers a typed array (an object literal's
/// empty shape is a fresh typed array's; `Object.create`'s objects start at another).
#[test]
fn typed_array_ways_record_their_kind_and_numeric_names_fill_none() {
    let mut engine = Engine::new();
    evaluate(
        &mut engine,
        "globalThis.ta = new Uint8Array(4); globalThis.fake = Object.setPrototypeOf({}, Uint8Array.prototype);\
         for (var k of ['-0', '1.5', 'NaN', 'Infinity']) Uint8Array.prototype[k] = 7; 0",
    );
    let ta = global(&mut engine, "ta");
    let fake = global(&mut engine, "fake");
    let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
    for _ in 0..3 {
        assert!(matches!(
            read(&mut engine, &ta, "length", &cache[0]).view(),
            ValueView::Num(4.0)
        ));
        let way = cache[0].get();
        assert_eq!(way.depth, IC_ACCESSOR | 2);
        assert_eq!(way.mid_ok & IC_MID_PRIMITIVE, IC_MID_TYPED_ARRAY);
        assert_eq!(
            cache[1].get().depth,
            IC_EMPTY,
            "one way, hit after its fill"
        );
    }
    for name in ["-0", "1.5", "NaN", "Infinity"] {
        let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
        assert!(matches!(
            read(&mut engine, &ta, name, &cache[0]).view(),
            ValueView::Undefined
        ));
        assert_eq!(cache[0].get().depth, IC_EMPTY, "{name}");
        assert_ne!(cache[0].get().mid_ok & IC_MID_GENERIC, 0, "{name}");
        // The plain receiver's way, then the typed array at the same site.
        let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
        assert!(matches!(
            read(&mut engine, &fake, name, &cache[0]).view(),
            ValueView::Num(7.0)
        ));
        assert_eq!(cache[0].get().mid_ok & IC_MID_PRIMITIVE, 0, "{name}");
        assert!(
            matches!(
                read(&mut engine, &ta, name, &cache[0]).view(),
                ValueView::Undefined
            ),
            "a plain way must not answer a typed array's {name}"
        );
    }
}

/// a1: a typed array's absent name makes no absence way and leaves its site generic, as a
/// refused fill made it.
#[test]
fn typed_array_absent_names_fill_no_way() {
    let mut engine = Engine::new();
    evaluate(&mut engine, "globalThis.ta = new Float64Array(2); 0");
    let ta = global(&mut engine, "ta");
    let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
    for _ in 0..2 {
        assert!(matches!(
            read(&mut engine, &ta, "absentName", &cache[0]).view(),
            ValueView::Undefined
        ));
        assert_eq!(cache[0].get().depth, IC_EMPTY);
        assert_ne!(cache[0].get().mid_ok & IC_MID_GENERIC, 0);
    }
}

/// a2: a native getter found by a way is called directly in the VM and from generated code, a
/// typed array's (a1's way; ARM64's template answers it in line, piece b), a Map's and an
/// embedder's closures alike; another Realm's native takes the full call.
#[test]
fn native_getters_found_by_ways_are_called_directly() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier, 0);
        engine.scope(|cx| {
            use crate::embed::{NativeCaptures, Value as V};
            let proto = V::Obj(cx.new_object());
            let get = cx.new_native_fn("get h", 0, |_, _, _| Ok(V::Num(2.0)), NativeCaptures::None);
            cx.define_accessor_value(&proto, "h", Some(get), None, false);
            fn scoped<'s>(
                _: &mut crate::embed::Ctx<'s>,
                _: crate::embed::Value<'s>,
                _: &[crate::embed::Value<'s>],
            ) -> Result<crate::embed::Value<'s>, crate::embed::Value<'s>> {
                Ok(crate::embed::Value::Num(3.0))
            }
            let get = V::Obj(cx.make_native("get g", 0, scoped));
            cx.define_accessor_value(&proto, "g", Some(get), None, false);
            let global = cx.global_this();
            cx.member_set(&global, "HostProto", proto).expect("global");
        });
        let before = native_getters();
        assert_eq!(
            evaluate(
                &mut engine,
                "var ta = new Int32Array(5), m = new Map([[1, 1]]), h = Object.create(HostProto), s = 0;\
                 function f(a, b, c) { return a.length + b.size + c.h + c.g; }\
                 for (var i = 0; i < 100; i++) s += f(ta, m, h); String(s)"
            ),
            "1100"
        );
        // On ARM64 the template answers a typed array's `length` in line (piece b).
        let calls = if cfg!(target_arch = "aarch64") && tier == Tier::Jit {
            3
        } else {
            4
        };
        assert!(
            native_getters() - before >= calls * 90,
            "{tier:?}: {} direct calls",
            native_getters() - before
        );
        let before = native_getters();
        assert_eq!(
            evaluate(
                &mut engine,
                "var other = $262.createRealm().global, om = new other.Map([[1, 1]]), t = 0;\
                 function g(a) { return a.size; } for (var i = 0; i < 100; i++) t += g(om); String(t)"
            ),
            "100"
        );
        assert_eq!(
            native_getters(),
            before,
            "{tier:?}: another Realm's native takes the call"
        );
    }
}

/// a2: a host getter's closure captures stay rooted through its direct call, even when the
/// getter deletes its own accessor and collects before reading them.
#[test]
fn host_getter_captures_stay_rooted_through_the_direct_call() {
    let mut engine = engine(Tier::Jit, 0);
    engine.scope(|cx| {
        use crate::embed::{NativeCaptures, Value as V};
        let proto = V::Obj(cx.new_object());
        let held = cx
            .eval_value("({ n: 41 })")
            .expect("parses")
            .expect("evaluates");
        let get = cx.new_native_closure_with_captures(
            "get once",
            0,
            |cx, this, _, captures| {
                let proto = cx
                    .eval_value("HostProto2")
                    .expect("parses")
                    .expect("evaluates");
                let _ = this;
                cx.eval_value("delete HostProto2.once; HostProto2 = null; 0")
                    .expect("parses")
                    .expect("evaluates");
                let _ = proto;
                cx.collect_garbage_for_host();
                let n = cx.member_get(&captures[0], "n").expect("a capture");
                Ok(V::Num(n.as_num_opt().unwrap_or(0.0) + 1.0))
            },
            &[held],
            NativeCaptures::None,
        );
        cx.define_accessor_value(&proto, "once", Some(get), None, true);
        let global = cx.global_this();
        cx.member_set(&global, "HostProto2", proto).expect("global");
    });
    let before = native_getters();
    assert_eq!(
        evaluate(
            &mut engine,
            "var o = Object.create(HostProto2); function r(a) { return a.once; } \
             String(r(o)) + ',' + String(r(o))"
        ),
        "42,undefined"
    );
    assert!(native_getters() > before);
}

fn native_setters() -> usize {
    property_read::TEST_NATIVE_SETTERS.with(Cell::get)
}

/// a3: a native setter found by a store's way is called directly with the receiver and the
/// value, in the VM and from generated code, an embedder's closures and bare natives alike.
#[test]
fn native_setters_found_by_ways_are_called_directly() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier, 0);
        engine.scope(|cx| {
            use crate::embed::{NativeCaptures, Value as V};
            let proto = V::Obj(cx.new_object());
            let set = cx.new_native_fn(
                "set h",
                1,
                |cx, this, args| {
                    let n = args.first().and_then(|v| v.as_num_opt()).unwrap_or(0.0);
                    let old = cx.member_get(&this, "sum")?.as_num_opt().unwrap_or(0.0);
                    cx.member_set(&this, "sum", V::Num(old + n))?;
                    Ok(V::Undefined)
                },
                NativeCaptures::None,
            );
            cx.define_accessor_value(&proto, "h", None, Some(set), false);
            fn scoped<'s>(
                cx: &mut crate::embed::Ctx<'s>,
                this: crate::embed::Value<'s>,
                args: &[crate::embed::Value<'s>],
            ) -> Result<crate::embed::Value<'s>, crate::embed::Value<'s>> {
                let n = args.first().and_then(|v| v.as_num_opt()).unwrap_or(0.0);
                let old = cx.member_get(&this, "sum")?.as_num_opt().unwrap_or(0.0);
                cx.member_set(&this, "sum", crate::embed::Value::Num(old + 2.0 * n))?;
                Ok(crate::embed::Value::Undefined)
            }
            let set = V::Obj(cx.make_native("set g", 1, scoped));
            cx.define_accessor_value(&proto, "g", None, Some(set), false);
            let global = cx.global_this();
            cx.member_set(&global, "HostSetters", proto)
                .expect("global");
        });
        let before = native_setters();
        assert_eq!(
            evaluate(
                &mut engine,
                "var h = Object.create(HostSetters); h.sum = 0;\
                 function f(o, i) { o.h = i; o.g = i; }\
                 for (var i = 0; i < 100; i++) f(h, i); String(h.sum)"
            ),
            "14850"
        );
        assert!(
            native_setters() - before >= 2 * 90,
            "{tier:?}: {} direct calls",
            native_setters() - before
        );
    }
}
