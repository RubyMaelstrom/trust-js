//! ECMA-262 e28783d5 §13.4.2.1-§13.4.5.1 (Postfix/Prefix Increment/Decrement) on element
//! references: GetValue, ToNumeric, Number::add/subtract, PutValue. The ARM64 template for
//! `obj[key]++`/`--` must agree with the checked operation for every receiver, key and value
//! it declines, and keep the element mirror coherent for the ones it handles.

use crate::{bytecode::Tier, Completion, Engine};

fn evaluate(tier: Tier, source: &str) -> String {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    match engine.eval(source, false).expect("update fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
    }
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        assert_eq!(evaluate(tier, source), expected, "{tier:?}");
    }
}

/// Values, holes, prototype elements, non-writable, frozen, accessor, typed-array, proxy,
/// mapped-arguments, string/negative/fractional keys and last-reference receivers (expected
/// answers from Node.js 22).
#[test]
fn element_updates_agree_with_the_checked_operation() {
    check(
        r##"
function run() {
  var out = [];
  var a = [1, 2.5, -0, NaN, "7", undefined, 1n, null, true, {valueOf(){ return 10; }}];
  for (var k = 0; k < a.length; k++) {
    var b = a.slice();
    var r1 = b[k]++; var r2 = ++b[k]; var r3 = b[k]--; var r4 = --b[k];
    out.push([Object.is(r1, -0) ? "-0" : r1, r2, r3, r4, Object.is(b[k], -0) ? "-0" : b[k]].map(String).join("/"));
  }
  var h = [1,,3]; h[1]++; out.push(h.join(",") + ":" + h.length);
  var o = [1,2]; o[5]++; out.push(o.length + ":" + o[5]);
  Array.prototype[7] = 41; var p = []; p.length = 9; var r = p[7]++;
  out.push(r + ":" + p[7] + ":" + Object.prototype.hasOwnProperty.call(p, 7)); delete Array.prototype[7];
  var nw = [1,2]; Object.defineProperty(nw, 0, {writable:false}); nw[0]++; out.push(String(nw[0]));
  out.push((function(){ "use strict"; var x = Object.freeze([1]); try { x[0]++; return "no"; } catch (e) { return e.name; } })());
  var calls = 0; var acc = [0];
  Object.defineProperty(acc, 0, {get(){ calls++; return 5; }, set(v){ calls += 10; }});
  var rr = acc[0]++; out.push(rr + ":" + calls);
  var t = new Int8Array([127, -128]); t[0]++; t[1]--; out.push(t[0] + ":" + t[1]);
  var f = new Float64Array([1.5]); f[0]++; out.push(String(f[0]));
  var obj = {0: 1, 1: 2.5, x: 3}; obj[0]++; obj[1]--; out.push(obj[0] + ":" + obj[1]);
  var trace = [];
  var px = new Proxy([1], {
    get(t, k) { trace.push("g" + String(k)); return t[k]; },
    set(t, k, v) { trace.push("s" + String(k) + "=" + v); t[k] = v; return true; }
  });
  px[0]++; out.push(trace.join(","));
  out.push((function(x){ arguments[0]++; return x; })(5));
  var big = [9007199254740992, 0.1, -1]; big[0]++; big[1]++; big[2]++;
  out.push(big[0] + ":" + big[1] + ":" + Object.is(big[2], 0));
  var sk = [1,2]; sk["1"]++; sk[-1] = 5; sk[-1]++; sk[1.5] = 1; sk[1.5]++;
  out.push(sk[1] + ":" + sk[-1] + ":" + sk[1.5]);
  out.push((function(){ return [41][0]++; })() + ":" + (function(){ return ++[41][0]; })());
  var fr = Object.freeze([1]); fr[0]++; out.push(String(fr[0]));
  var ne = Object.preventExtensions([1]); ne[0]++; out.push(String(ne[0]));
  var boxed = [{}]; boxed[0]++; out.push(typeof boxed[0]);
  return out.join("|");
}
var first = run(), same = true;
for (var i = 0; i < 60; i++) same = same && run() === first;
same + "#" + first;
"##,
        "true#1/3/3/1/1|2.5/4.5/4.5/2.5/2.5|-0/2/2/0/0|NaN/NaN/NaN/NaN/NaN|7/9/9/7/7|\
         NaN/NaN/NaN/NaN/NaN|1/3/3/1/1|0/2/2/0/0|1/3/3/1/1|10/12/12/10/10|1,NaN,3:3|6:NaN|\
         41:42:true|1|TypeError|5:11|-128:127|2.5|2:1.5|g0,s0=2|6|9007199254740992:1.1:true|\
         3:6:2|41:42|1|2|number",
    );
}

/// Updated numeric elements stay visible to every reader of the element mirror.
#[test]
fn element_updates_keep_the_numeric_mirror_coherent() {
    check(
        r#"
        function bump(m, n) { for (var z = 0; z < n; z++) m[z & 15]++; return m; }
        function fall(m, n) { for (var z = 0; z < n; z++) --m[z & 3]; return m; }
        var m = bump(new Array(16).fill(0), 1000);
        var s = 0; for (var z = 0; z < 16; z++) s += m[z];
        var d = fall([0.5, 1, 2, 3], 10);
        var holey = [1, 2, , 4]; bump(holey, 4);
        var objects = bump([{}, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 32);
        [s, m.indexOf(62), m.includes(63), m.reduce(function (x, y) { return x + y; }),
         Math.max.apply(null, m), d.join(), String(holey), holey.length,
         objects.slice(0, 3).join()].join("|");
        "#,
        "1000|8|true|1000|63|-2.5,-2,0,1|2,3,NaN,5|4|NaN,3,4",
    );
}

/// Dense Number elements of ordinary objects and Arrays update without the checked helper.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn dense_number_element_updates_stay_native() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let setup = r#"
        function pre(a, i) { return ++a[i]; }
        function post(a, i) { return a[i]--; }
        function discard(a, i) { a[i]++; }
        var array = [1, 2.5, 3], object = {0: 7, 1: 8}, packed = new Array(8).fill(0);
        for (var w = 0; w < 4; w++) { pre(array, 0); post(object, 1); discard(packed, 3); }
    "#;
    let Completion::Value(_) = engine.eval(setup, false).expect("setup parses") else {
        panic!("setup threw");
    };
    crate::bytecode::TEST_JIT_UPDATE_ELEM_HELPERS.with(|count| count.set(0));
    let source = r#"
        var out = 0;
        for (var k = 0; k < 200; k++) {
            out += pre(array, k % 3) + post(object, k & 1);
            discard(packed, k & 7);
        }
        [out, array.join(), object[0], object[1], packed.join()].join("|");
    "#;
    let Completion::Value(value) = engine.eval(source, false).expect("loop parses") else {
        panic!("loop threw");
    };
    assert_eq!(value, "-1332.5|72,69.5,69|-93|-96|25,25,25,29,25,25,25,25");
    assert_eq!(
        crate::bytecode::TEST_JIT_UPDATE_ELEM_HELPERS.with(std::cell::Cell::get),
        0
    );
}
