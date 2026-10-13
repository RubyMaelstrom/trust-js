//! ECMA-262 e28783d5, CodePointAt / String.prototype.codePointAt:
//! receiver conversion precedes index conversion; surrogate positions are code units.
use crate::{bytecode::Tier, Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("character fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

#[test]
fn character_reads_cover_short_cached_and_smuggled_utf16_positions() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            function point(s,n){return s.codePointAt(n);}
            function unit(s,n){return s.charCodeAt(n);}
            function character(s,n){return s.charAt(n);}
            var s='A\uD83D\uDE00\uD800z\uDC00é',long='x'.repeat(80)+s;
            for(var warm=0;warm<160;warm++){point(s,1);point(long,81);unit(s,2);character(s,3);}
            var points=[65,0x1F600,0xDE00,0xD800,122,0xDC00,233],units=[65,0xD83D,0xDE00,0xD800,122,0xDC00,233],ok=true;
            for(var n=0;n<points.length;n++){
                ok=ok&&point(s,n)===points[n]&&point(long,n+80)===points[n];
                ok=ok&&unit(s,n)===units[n]&&unit(long,n+80)===units[n];
                ok=ok&&character(s,n).charCodeAt(0)===units[n];
            }
            var maximum=String.fromCharCode(0xDBFF,0xDFFF),privateScalar='\u{10F800}';
            ok&&point(maximum,0)===0x10FFFF&&point(maximum,1)===0xDFFF&&point(privateScalar,0)===0x10F800&&point(s,7)===undefined&&character(s,7)===''&&Number.isNaN(unit(s,7))
        "#
            ),
            "true",
            "{tier:?}"
        );
        engine.interp.gc_collect();
        assert_eq!(eval(&mut engine, "point(long,81)"), "128512", "{tier:?}");
    }
}

#[test]
fn code_point_indices_and_method_replacement_keep_coercion_order() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var original=String.prototype.codePointAt;
            function point(s,n){return s.codePointAt(n);}
            for(var warm=0;warm<160;warm++)point('A😀',1);
            var trace='',receiver={toString(){trace+='r';return 'A😀'}},index={valueOf(){trace+='i';return 1}};
            var value=original.call(receiver,index),firstTrace=trace,failures=0;
            trace='';try{original.call(null,index)}catch(e){if(e instanceof TypeError)failures++;}
            for(var bad of [Symbol(),1n])try{point('A😀',bad)}catch(e){if(e instanceof TypeError)failures++;}
            var ok=point('A😀',NaN)===65&&point('A😀',-0.5)===65&&point('A😀',1.9)===0x1F600&&point('A😀',2.9)===0xDE00;
            for(var out of [-1,3,Infinity,-Infinity,Number.MAX_VALUE])ok=ok&&point('A😀',out)===undefined;
            String.prototype.codePointAt=function(n){return n+900};
            var replaced=point('A😀',2);String.prototype.codePointAt=original;
            [value,firstTrace,trace,failures,ok,replaced,point('A😀',1)].join('|')
        "#
            ),
            "128512|ri||3|true|902|128512",
            "{tier:?}"
        );
    }
}

#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn numeric_code_point_calls_enter_the_guarded_native_intrinsic() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        function point(s,n){return s.codePointAt(n);}
        for(var warm=0;warm<160;warm++)point('é😀',1);
    "#,
    );
    // CALL_IC_EPOCH is process-wide: a concurrently running test that recompiles a chunk
    // invalidates every cached call site, and the next few calls here legitimately revalidate
    // through the generic path. Measure a run during which the epoch stayed unchanged.
    let (mut hits, mut epoch_before, mut epoch_after) = (0, 0, 1);
    for _attempt in 0..20 {
        let before = crate::bytecode::TEST_JIT_CODE_POINT_INTRINSICS.with(std::cell::Cell::get);
        epoch_before = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var s='é😀',wide='x'.repeat(80)+s,ok=true;
            for(var n=0;n<40;n++)ok=ok&&point(s,1)===0x1F600&&point(wide,81)===0x1F600&&point('plain',2)===97;
            ok
        "#
            ),
            "true"
        );
        hits = crate::bytecode::TEST_JIT_CODE_POINT_INTRINSICS.with(std::cell::Cell::get) - before;
        epoch_after = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if epoch_before == epoch_after {
            break;
        }
    }
    assert!(hits >= 120,
        "warm String/Number calls bypass generic native argument decoding: {hits} hits, epoch {epoch_before}->{epoch_after}");
    assert!(engine.interp.fn_frames.is_empty());
}

/// charCodeAt on non-ASCII receivers and with any Number index, String.fromCharCode(Number) and
/// isNaN/Number.isNaN(Number) take the JIT's operand-only paths; other operands, replaced
/// functions and fresh (last-owner) receivers keep the full call. Expected values are Node's.
#[test]
fn unit_reads_from_char_code_and_is_nan_match_the_full_calls() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            function unit(s,n){return s.charCodeAt(n);}
            function fcc(n){return String.fromCharCode(n);}
            function nan1(x){return isNaN(x);}
            function nan2(x){return Number.isNaN(x);}
            function nan3(o,x){return o.isNaN(x);}
            var long='é'.repeat(40)+'abc', short='hé';
            for (var w=0; w<300; w++){ unit(long, w%45); unit(short, w&1); fcc(w); nan1(w); nan2(w);
                nan3(globalThis, w); nan3(Number, w); }
            var r=[];
            r.push(unit(long,0), unit(long,40), unit(long,42), unit(long,43), unit(long,-1),
                unit(long,1.9), unit(long,NaN), unit(long,1e300), unit(short,1), unit(short,5),
                unit(long,-0.5));
            r.push(fcc(65).charCodeAt(0), fcc(-1).charCodeAt(0), fcc(65536+66).charCodeAt(0),
                fcc(NaN).charCodeAt(0), fcc(Infinity).charCodeAt(0), fcc(97.9),
                fcc(0xD800).charCodeAt(0), fcc(233), fcc(-0).length);
            r.push(nan1(NaN), nan1(1), nan1('abc'), nan1('5'), nan2(NaN), nan2('abc'),
                nan3(globalThis, NaN), nan3({isNaN: isNaN}, 3), nan3(Number, NaN), nan3(Number, 'x'),
                nan1(undefined), nan1(-Infinity));
            r.push(unit('é'.repeat(70)+String(w), 69), unit(String(w)+'é', 3));
            var saved = String.fromCharCode; String.fromCharCode = function(){return 'patched'};
            r.push(fcc(65)); String.fromCharCode = saved; r.push(fcc(66));
            var savedNaN = isNaN; isNaN = function(){return 'p2'}; r.push(nan1(1));
            isNaN = savedNaN; r.push(nan1(NaN));
            r.join(',');
        "#
            ),
            "233,97,99,NaN,NaN,233,233,NaN,233,NaN,233,65,65535,66,0,0,a,55296,\u{e9},1,\
             true,false,true,false,true,false,true,false,true,false,true,false,233,233,patched,B,p2,true",
            "{tier:?}"
        );
    }
}

/// Map/Set/WeakMap/WeakSet methods and Math's numeric functions called on moved operands
/// without the native-call boundary, and Reflect.apply with such a target, keep the full
/// algorithms: CoerceKey, brand checks, weak-key validation, ToNumber's order for non-Number
/// arguments, and CreateListFromArrayLike's Gets for holes, getters and array-likes. Expected
/// values are Node's.
#[test]
fn operand_only_natives_and_reflect_apply_match_the_full_calls() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
var log = [];
function rec(x) { log.push(String(x)); }
function err(f) { try { return String(f()); } catch (e) { return e.constructor.name; } }
var keys = [{}, {}, 'k', -0, NaN, 1];
var m = new Map(), s = new Set(), wm = new WeakMap(), ws = new WeakSet();
function mg(x, k) { return x.get(k); }
function mh(x, k) { return x.has(k); }
function ms(x, k, v) { return x.set(k, v); }
function md(x, k) { return x.delete(k); }
function sa(x, k) { return x.add(k); }
function ab(x) { return Math.abs(x); }
function fl(x) { return Math.floor(x); }
function mx(a, b) { return Math.max(a, b); }
function ra(f, t, l) { return Reflect.apply(f, t, l); }
for (var i = 0; i < 400; i++) {
  var k = keys[i % keys.length];
  ms(m, k, i); mg(m, k); mh(m, k); sa(s, k); mh(s, k);
  if (typeof k === 'object') { ms(wm, k, i); mg(wm, k); mh(wm, k); sa(ws, k); mh(ws, k); }
  ab(i - 200); fl(i / 3); mx(i, 200 - i); ra(WeakMap.prototype.get, wm, [keys[0]]);
}
rec(mg(m, 0)); rec(mg(m, -0)); rec(mg(m, NaN)); rec(mg(m, 'k')); rec(mg(m, keys[1])); rec(mg(m, 'nope'));
rec(mh(m, +0)); rec(md(m, -0)); rec(mh(m, 0)); rec(ms(m, 'x', 1) === m); rec(m.size);
rec(mh(s, NaN)); rec(sa(s, 7) === s); rec(md(s, 7)); rec(s.size);
rec(mg(wm, keys[0])); rec(mg(wm, 'prim')); rec(mh(wm, 3)); rec(err(function () { return ms(wm, 'prim', 1); }));
rec(md(wm, keys[1])); rec(mh(wm, keys[1])); rec(mh(ws, keys[0])); rec(err(function () { return sa(ws, 5); }));
var sym = Symbol('w'); ms(wm, sym, 'sv'); rec(mg(wm, sym));
rec(err(function () { return mg({ get: Map.prototype.get }, 1); }));
rec(err(function () { return mg(s, 1); }));
rec(err(function () { return mh(m.has ? { has: Set.prototype.has } : 0, 1); }));
rec(err(function () { return mg(wm.get ? { get: WeakMap.prototype.get } : 0, keys[0]); }));
rec(err(function () { return mg(new WeakSet(), keys[0]); }));
var order = []; var o1 = { valueOf: function () { order.push('a'); return -3; } }, o2 = { valueOf: function () { order.push('b'); return 4; } };
rec(ab(o1)); rec(fl('2.7')); rec(mx(o1, o2)); rec(order.join('')); rec(ab(-0) === 0 && 1 / ab(-0)); rec(fl(-0.5)); rec(mx(NaN, 1)); rec(mx(-0, 0));
rec(ra(WeakMap.prototype.get, wm, [keys[0]])); rec(ra(WeakMap.prototype.get, wm, [,])); rec(ra(Map.prototype.get, m, ['k']));
var getterHits = 0; var arr = [0]; Object.defineProperty(arr, 0, { get: function () { getterHits++; return 'k'; } });
rec(ra(Map.prototype.get, m, arr)); rec(getterHits);
rec(ra(Map.prototype.get, m, { length: 1, 0: 'k' })); rec(err(function () { return ra(Map.prototype.get, s, ['k']); }));
rec(err(function () { return ra(5, m, []); })); rec(ra(function (a, b) { return a + b + this.z; }, { z: 1 }, [2, 3]));
rec(ra(Math.max, null, [3, 9, 4])); rec(ra(Math.max, null, [3, o2])); rec(order.join(''));
Array.prototype[0] = 'k'; rec(ra(Map.prototype.get, m, [,])); delete Array.prototype[0];
log.join(',');
        "#
            ),
            "399,399,394,398,397,undefined,true,true,false,true,6,true,true,true,6,396,undefined,false,TypeError,true,false,true,TypeError,sv,TypeError,TypeError,TypeError,TypeError,TypeError,3,2,4,aab,Infinity,-1,NaN,0,396,undefined,398,398,1,398,TypeError,TypeError,6,9,4,aabb,398",
            "{tier:?}"
        );
    }
}

/// Private field, method and accessor operations run in the JIT's operation helper rather than
/// through a materialized activation; they still resolve each Private Name from the running
/// environment (ECMA-262 ResolvePrivateIdentifier), so nested classes and computed keys see
/// their own `#x`, block scopes do not disturb it, and brand checks throw. Expected values are
/// Node's.
#[test]
fn private_names_resolve_from_the_running_environment_in_compiled_code() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
var log = [];
function rec(x) { log.push(String(x)); }
function err(f) { try { return String(f()); } catch (e) { return e.constructor.name; } }
class A {
  #x = 1; static #count = 0; #m() { return this.#x * 10; } get #acc() { return this.#x + 100; } set #acc(v) { this.#x = v; }
  constructor(v) { this.#x = v; A.#count++; }
  get x() { return this.#x; }
  bump() { this.#x = this.#x + 1; return this.#x; }
  callM() { return this.#m(); }
  acc() { this.#acc = this.#acc; return this.#x; }
  static count() { return A.#count; }
  static has(o) { return #x in o; }
  static peek(o) { return o.#x; }
  blocky(fs) { for (let k = 0; k < 2; k++) { fs.push(() => k + this.#x); } { let z = this.#x; fs.push(() => z); } return this.#x; }
  nested() {
    class B { #x = 'inner'; read(o) { return o.#x; } static key(o) { return #x in o; } }
    const b = new B();
    return [b.read(b), err(() => b.read(this)), B.key(this), B.key(b), this.#x].join('/');
  }
  computed(o) { const self = this; class C { #x = 'c'; [o.#x]() { return 1; } static k = Object.getOwnPropertyNames(C.prototype).join(); } return C.k; }
}
var objs = []; for (var i = 0; i < 200; i++) objs.push(new A(i));
var s = 0;
for (var r = 0; r < 30; r++) for (var i = 0; i < 200; i++) { var o = objs[i]; s += o.x + (A.has(o) ? 1 : 0) + A.peek(o) + o.callM(); }
rec(s); rec(A.count());
rec(objs[3].bump()); rec(objs[3].x); rec(objs[3].callM()); rec(objs[3].acc());
rec(A.has({})); rec(err(() => A.peek({}))); rec(err(() => A.has(5))); rec(err(() => Object.getOwnPropertyDescriptor(A.prototype, 'x').get.call({})));
var fs = []; rec(objs[5].blocky(fs)); rec(fs.map(f => f()).join(','));
rec(objs[7].nested());
rec(err(() => objs[2].computed(objs[9])));
for (var r = 0; r < 300; r++) objs[r % 200].bump();
rec(objs[0].x); rec(objs[199].x);
class P { #v; constructor(v) { this.#v = v } static get(o) { return o.#v } }
var proxy = new Proxy(new P(1), {}); rec(err(() => P.get(proxy)));
log.join(',');
        "#
            ),
            "7170000,200,4,4,40,104,false,TypeError,TypeError,TypeError,5,5,6,5,inner/TypeError/false/true/7,TypeError,2,200,TypeError",
            "{tier:?}"
        );
    }
}

/// Method intrinsics at call sites outside loops run in the chunk's shared intrinsic stub; the
/// same intrinsics inside a small loop keep their inline sequences. Both must complete, decline
/// (other operand types, replaced builtins), throw and leave non-intrinsic calls to the ordinary
/// call exactly as before. Expected values are Node's.
#[test]
fn shared_and_inline_call_intrinsics_agree() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
var log = [];
function rec(x) { log[log.length] = (typeof x === 'number' && x !== x ? 'NaN' : String(x)); }
function err(f) { try { return String(f()); } catch (e) { return e.constructor.name + (e instanceof RangeError ? ':' + e.message : ''); } }
// One call site per helper, outside any loop (shared-stub dispatch); the callers loop.
function ca(s, n) { return s.charAt(n); }
function cc(s, n) { return s.charCodeAt(n); }
function cp(s, n) { return s.codePointAt(n); }
function sq(m, x) { return m.sqrt(x); }
function nn(o, x) { return o.isNaN(x); }
function fc(S, n) { return S.fromCharCode(n); }
function pu(a, v) { return a.push(v); }
function ex(re, s) { re.exec(s); return re.lastIndex; }
function sp(s, sep) { s.split(sep); return s.length; }
function sl(s, a, b) { return s.slice(a, b); }
function ho(O, o, k) { return O.hasOwn(o, k); }
function ap(f, t, args) { return f.apply(t, args); }
function rp(s, re, w) { s.replace(re, w); return re.lastIndex; }
function cl1(f, t) { return f.call(t); }
function cl2(f, t, x) { return f.call(t, x); }
function add(y) { return (this && this.base || 0) + (y === undefined ? 1 : y); }
function thrower(y) { throw new RangeError('t' + y); }
var ascii = 'hello world', uni = 'héllo \u{1F600}!';
var acc = [];
for (var r = 0; r < 300; r++) {
  var k = r % 13;
  acc.push(ca(ascii, k), cc(ascii, k), cc(uni, k % 9), cp(uni, k % 9), sq(Math, k * k), nn(globalThis, k), nn(Number, k),
    fc(String, 97 + k), pu([1, 2], k), ex(/o/g, ascii), sp('a,b,c', ','), sl(ascii, 1, 1 + k % 5), ho(Object, { a: 1 }, 'a'),
    ap(add, { base: 10 }, [k]), rp('aXbXc', /X/g, '-'), cl1(add, { base: 2 }), cl2(add, { base: 3 }, k));
}
rec(acc.length); rec(acc.slice(0, 17).join('|')); rec(acc.slice(-17).join('|'));
// Declines and unusual operands through the same warmed sites.
rec(ca(ascii, 1.7)); rec(ca(ascii, -1)); rec(ca(ascii, 99)); rec(ca(uni, 7).charCodeAt(0)); rec(err(() => ca(5, 0)));
rec(cc(ascii, 2.9)); rec(cc(ascii, 400)); rec(cc(uni, 6)); rec(cc(ascii, '3')); rec(cc(new String('ab'), 1));
rec(cp(uni, 7)); rec(cp(uni, 8)); rec(cp(ascii, 50));
rec(sq(Math, '16')); rec(sq(Math, -1)); rec(sq({ sqrt: function (x) { return 'own' + x; } }, 4));
rec(nn(globalThis, 'x')); rec(nn(Number, 'x')); rec(nn(globalThis, undefined));
rec(fc(String, 65.9)); rec(fc(String, 65536 + 66)); rec(fc(String, '67'));
rec(pu({ length: 1, push: Array.prototype.push }, 'v')); rec(err(() => pu(Object.freeze([1]), 2)));
rec(ex(/x/g, ascii)); rec(sp('a1b2c', /\d/)); rec(sl(uni, 1, 3)); rec(sl(ascii, '1', '4'));
rec(ho(Object, { b: 2 }, 'a')); rec(err(() => ho(Object, null, 'a')));
rec(err(() => ap(thrower, null, [7]))); rec(ap(add, null, { length: 1, 0: 41 }));
rec(err(() => cl1(thrower, null))); rec(err(() => cl2(thrower, null, 9))); rec(err(() => cl2(5, null, 1)));
rec(rp('aaa', /a/g, function (m) { return m.toUpperCase(); }));
// Replacing a builtin changes the call IC's identity proof.
var savedCharAt = String.prototype.charAt; String.prototype.charAt = function () { return 'patched'; };
rec(ca(ascii, 0)); String.prototype.charAt = savedCharAt; rec(ca(ascii, 0));
var savedPush = Array.prototype.push; Array.prototype.push = function () { return -1; };
rec(pu([1], 2)); Array.prototype.push = savedPush; rec(pu([1], 2));
// The same intrinsics at sites inside a small loop keep their inline sequences.
function hot(n) {
  var t = 0, parts = [];
  for (var q = 0; q < n; q++) {
    t += ascii.charCodeAt(q % 11) + uni.charCodeAt(q % 9) + Math.sqrt(q) + (isNaN(q) ? 1 : 0) + ascii.codePointAt(q % 3);
    parts.push(String.fromCharCode(97 + q % 26) + ascii.charAt(q % 11) + ascii.slice(q % 3, 5));
    if (Object.hasOwn(parts, q)) t++;
  }
  return t + ':' + parts.length + ':' + parts[n - 1];
}
rec(hot(300)); rec(hot(7));
log.join(',');
"#
            ),
            "5100,h|104|104|104|0|false|false|a|3|5|5||true|10|0|3|3,h|104|104|104|0|false|false|a|3|5|5||true|10|0|3|3,e,,,56832,TypeError,108,NaN,55357,108,98,56832,33,undefined,4,NaN,own4,true,false,true,A,B,C,2,TypeError,0,5,él,ell,false,TypeError,RangeError:t7,41,RangeError:tundefined,RangeError:t9,TypeError,0,patched,h,-1,2,3792239.2358805016:300:nlllo,57483.83182209023:7:gwhello",
            "{tier:?}"
        );
    }
}
