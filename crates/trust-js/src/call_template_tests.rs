//! W3 item 2c's call templates for proven natives (ECMA-262 at e28783d5, spec.html): Array.isArray
//! (IsArray, :5878), Object.is (SameValue, :6008), and Math.max and Math.min with three and four
//! arguments (:33847, :33869). Each program runs in every tier, its call sites both outside loops
//! (the JIT's shared stubs) and inside them; the expected values are Node v24.22.0's, and a
//! reference written in comparisons checks every combination of the listed operands.
use crate::{bytecode::Tier, Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("template fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// Array.isArray, as a method and through a captured alias (a plain call, Vue's
/// `const j = Array.isArray`), on every kind of value: a Proxy's target decides, a revoked
/// Proxy throws, Array.prototype and Array subclasses' instances are Arrays, typed arrays and
/// `arguments` are not. Object.is on Numbers (NaN, signed zeros), Strings and BigInts of equal
/// content in distinct cells, identities and mixed kinds. Replaced builtins fail the call
/// cache's proof.
const IS_ARRAY_AND_OBJECT_IS: &str = r#"
var log = [];
function rec(x) { log[log.length] = (typeof x === 'number' && x !== x ? 'NaN' : String(x)); }
function err(f) { try { return String(f()); } catch (e) { return e.constructor.name; } }
// One call site per form, outside any loop (the shared stubs); the callers loop.
function isA(x) { return Array.isArray(x); }
var isAlias = (function () { const j = Array.isArray; return function (x) { return j(x); }; })();
function same(a, b) { return Object.is(a, b); }
// The same calls inside loops keep their sequences in line.
function hotIsA(list) { var s = ''; for (var q = 0; q < list.length; q++) s += Array.isArray(list[q]) ? '1' : '0'; return s; }
function hotAlias(list) { const j = Array.isArray; var s = ''; for (var q = 0; q < list.length; q++) s += j(list[q]) ? '1' : '0'; return s; }
function hotSame(list) { var s = ''; for (var q = 0; q < list.length; q++) s += Object.is(list[q][0], list[q][1]) ? '1' : '0'; return s; }
var revocable = Proxy.revocable([], {}); revocable.revoke();
var values = [[], [1, 2], {}, { length: 0 }, null, undefined, 0, -0, 1, NaN, '', 'a', true, false,
  Symbol('s'), 10n, function () {}, new Proxy([], {}), new Proxy({}, {}), new Proxy(new Proxy([], {}), {}),
  Array.prototype, Object.create(Array.prototype), new Uint8Array(2), (function () { return arguments; })(),
  new (class extends Array {})(), new Map(), /x/, new Date(0), new String('s'), Object(1n)];
var cold = '';
for (var r = 0; r < 20; r++) {
  cold = '';
  for (var i = 0; i < values.length; i++) cold += (isA(values[i]) ? '1' : '0') + (isAlias(values[i]) ? '1' : '0');
}
rec(cold);
var hot = '';
for (var r2 = 0; r2 < 20; r2++) hot = hotIsA(values) + ':' + hotAlias(values);
rec(hot);
rec(err(function () { return isA(revocable.proxy); })); rec(err(function () { return isAlias(revocable.proxy); }));
rec(err(function () { return hotIsA([[], revocable.proxy]); })); rec(err(function () { return hotAlias([revocable.proxy]); }));
var nested = Proxy.revocable([], {}); var outer = new Proxy(nested.proxy, {}); nested.revoke();
rec(err(function () { return isA(outer); }));
// SameValue (spec.html:6008): NaN is NaN, +0 is not -0, Strings and BigInts by content.
var s1 = 'ab' + 'cd', s2 = 'abc' + 'd', w1 = 'éx'.repeat(3), w2 = 'éx'.repeat(3);
var big1 = 2n ** 64n, big2 = 2n ** 64n, o = {}, sym = Symbol('q');
var pairs = [[NaN, NaN], [0, -0], [-0, 0], [-0, -0], [0, 0], [1, 1], [1, 2], [NaN, 0], [s1, s2], [s1, 'abce'],
  [s1, 'abc'], ['', ''], [w1, w2], [w1, w1.slice(1)], [w1, s1 + 'xy'], [o, o], [o, {}], [sym, sym], [sym, Symbol('q')],
  [big1, big2], [1n, 2n], [1n, 1], [true, true], [true, false], [null, null], [undefined, undefined], [null, undefined],
  [0, '0'], [NaN, 'NaN'], [Infinity, Infinity], [-Infinity, Infinity], [1.5, 1.5], [0.1 + 0.2, 0.3], [isA, isA],
  [values[17], values[17]], [values[17], []], [-0, false], [0, null], ['1', 1]];
var coldSame = '';
for (var r3 = 0; r3 < 20; r3++) {
  coldSame = '';
  for (var p = 0; p < pairs.length; p++) coldSame += same(pairs[p][0], pairs[p][1]) ? '1' : '0';
}
rec(coldSame);
var hotSameResult = '';
for (var r4 = 0; r4 < 20; r4++) hotSameResult = hotSame(pairs);
rec(hotSameResult);
rec(same(1) + ':' + same() + ':' + same(undefined, void 0));
// Replacing a builtin changes the call cache's identity proof.
var savedIsArray = Array.isArray, savedIs = Object.is;
Array.isArray = function (x) { return 'patched' + typeof x; }; Object.is = function () { return 'patched'; };
rec(isA([])); rec(hotIsA([[]])); rec(same(1, 1)); rec(hotSame([[1, 1]]));
Array.isArray = savedIsArray; Object.is = savedIs;
rec(isA([])); rec(hotIsA([[]])); rec(same(1, 1)); rec(hotSame([[1, 1]])); rec(isAlias([]));
log.join(',');
"#;

/// Math.min and Math.max with three and four arguments over NaN, the signed zeros, the
/// infinities and the subnormal minimum, and with arguments that are not Numbers: ToNumber runs
/// on each in order (valueOf), a Symbol throws before the arguments after it convert.
const MIN_MAX: &str = r#"
var log = [];
function rec(x) { log[log.length] = (typeof x === 'number' && x !== x ? 'NaN' : Object.is(x, -0) ? '-0' : String(x)); }
function err(f) { try { return String(f()); } catch (e) { return e.constructor.name; } }
// Math.max and Math.min (spec.html:33847, :33869): NaN wins, -0 is below +0. A reference over
// comparisons alone checks every combination.
function refMax(list) {
  var highest = -Infinity;
  for (var i = 0; i < list.length; i++) {
    var n = list[i];
    if (n !== n) return NaN;
    if (n === 0 && highest === 0 && 1 / highest < 0 && 1 / n > 0) highest = n;
    if (n > highest) highest = n;
  }
  return highest;
}
function refMin(list) {
  var lowest = Infinity;
  for (var i = 0; i < list.length; i++) {
    var n = list[i];
    if (n !== n) return NaN;
    if (n === 0 && lowest === 0 && 1 / lowest > 0 && 1 / n < 0) lowest = n;
    if (n < lowest) lowest = n;
  }
  return lowest;
}
// One call site per form, outside any loop (the shared stubs); the callers loop.
function min3(a, b, c) { return Math.min(a, b, c); }
function max3(a, b, c) { return Math.max(a, b, c); }
function min4(a, b, c, d) { return Math.min(a, b, c, d); }
function max4(a, b, c, d) { return Math.max(a, b, c, d); }
var vals = [1, -1, 0, -0, NaN, Infinity, -Infinity, 2.5, -2.5, 1e308, 5e-324];
var bad = 0, calls = 0;
for (var i = 0; i < vals.length; i++) for (var j = 0; j < vals.length; j++) for (var k = 0; k < vals.length; k++) {
  var d = vals[(i * 7 + j * 3 + k) % vals.length], a = vals[i], b = vals[j], c = vals[k];
  if (!Object.is(min3(a, b, c), refMin([a, b, c]))) bad++;
  if (!Object.is(max3(a, b, c), refMax([a, b, c]))) bad++;
  if (!Object.is(min4(a, b, c, d), refMin([a, b, c, d]))) bad++;
  if (!Object.is(max4(a, b, c, d), refMax([a, b, c, d]))) bad++;
  calls += 4;
}
rec(bad + '/' + calls);
// The same calls inside a loop keep their sequences in line.
function hot(list) {
  var out = [];
  for (var q = 0; q + 3 < list.length; q++) {
    out.push(Math.min(list[q], list[q + 1], list[q + 2]), Math.max(list[q], list[q + 1], list[q + 2]),
      Math.min(list[q], list[q + 1], list[q + 2], list[q + 3]), Math.max(list[q], list[q + 1], list[q + 2], list[q + 3]));
  }
  return out;
}
var series = [3, -0, 0, NaN, 7, -Infinity, 0, -0, 2.5, 1e308, -1, 5e-324, Infinity, 4];
var hotBad = 0, hotOut;
for (var r = 0; r < 10; r++) {
  hotOut = hot(series);
  for (var q = 0; q + 3 < series.length; q++) {
    var w = series.slice(q, q + 4);
    if (!Object.is(hotOut[4 * q], refMin(w.slice(0, 3))) || !Object.is(hotOut[4 * q + 1], refMax(w.slice(0, 3))) ||
        !Object.is(hotOut[4 * q + 2], refMin(w)) || !Object.is(hotOut[4 * q + 3], refMax(w))) hotBad++;
  }
}
rec(hotBad); rec(hotOut.map(function (x) { return Object.is(x, -0) ? '-0' : String(x); }).join(' '));
// Arguments that are not Numbers: ToNumber runs on every argument, in order, before any compare.
var order = [];
function v(name, value) { return { valueOf: function () { order.push(name); return value; } }; }
rec(max3(1, v('a', 5), v('b', NaN))); rec(min4(v('c', 3), 2, v('d', -0), 7)); rec(min3(NaN, v('e', 1), v('f', 2)));
rec(max4(v('g', -0), 0, v('h', -0), -0)); rec(order.join(''));
rec(err(function () { return max3(1, Symbol(), v('i', 2)); })); rec(order.join(''));
rec(max3('3', 2, 1)); rec(min3(undefined, 1, 2)); rec(min3(null, 1, 2)); rec(max4(true, false, 0, -1));
rec(err(function () { return min3(1n, 1, 1); })); rec(min4('', ' ', '\n', 1)); rec(max3([5], [], 4));
function hotCoerce(n) { var t = []; for (var q = 0; q < n; q++) t.push(Math.max(q, v('h' + q, q * 2), -1), Math.min(-q, q, v('m' + q, -2 * q), 0)); return t.join(' '); }
order = [];
rec(hotCoerce(4)); rec(order.join(''));
// A replaced builtin and a receiver that is not Math.
var savedMax = Math.max; Math.max = function () { return 'patched'; };
rec(max3(1, 2, 3)); rec(max4(1, 2, 3, 4)); Math.max = savedMax; rec(max3(1, 2, 3));
var fake = { min: Math.min }; rec(fake.min(3, 1, 2)); rec(fake.min(3, 1, 2, -0));
log.join(',');
"#;

#[test]
fn is_array_and_object_is_templates_match_the_full_calls() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, IS_ARRAY_AND_OBJECT_IS),
            "111100000000000000000000000000000011001111000000110000000000,110000000000000001011000100000:110000000000000001011000100000,TypeError,TypeError,TypeError,TypeError,TypeError,100111001001100101010010110001010110000,100111001001100101010010110001010110000,false:true:true,patchedobject,1,patched,1,true,1,true,1,true",
            "{tier:?}"
        );
        engine.interp.gc_collect();
    }
}

#[test]
fn min_max_templates_match_the_full_calls() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, MIN_MAX),
            "0/5324,0,-0 3 NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN NaN -Infinity 7 -Infinity 7 -Infinity 0 -Infinity 2.5 -0 2.5 -0 1e+308 -0 1e+308 -1 1e+308 -1 1e+308 -1 1e+308 -1 1e+308 -1 Infinity -1 Infinity -1 Infinity,NaN,-0,NaN,0,abcdefgh,TypeError,abcdefgh,3,NaN,0,1,TypeError,0,5,0 0 2 -2 4 -4 6 -6,h0m0h1m1h2m2h3m3,patched,patched,3,1,-0",
            "{tier:?}"
        );
        engine.interp.gc_collect();
    }
}

/// In compiled code the templates finish warm calls without the call-hit helper: Array.isArray as a method and through a plain call, and Object.is, count their commits, and
/// Math.max and Math.min with three and four Numbers count as the other Math templates do,
/// both at sites outside loops (the shared stubs) and inside one (in line).
#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn call_templates_finish_compiled_calls() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        function isA(x) { return Array.isArray(x); }
        var isAlias = (function () { const j = Array.isArray; return function (x) { return j(x); }; })();
        function same(a, b) { return Object.is(a, b); }
        function min3(a, b, c) { return Math.min(a, b, c); }
        function max4(a, b, c, d) { return Math.max(a, b, c, d); }
        function hot(n) {
            var t = 0;
            for (var q = 0; q < n; q++) {
                t += (Array.isArray(t) ? 1 : 0) + (Object.is(q, 3) ? 1 : 0) + q + Math.max(q, 1, 2) +
                    Math.min(q, 1, 2, 3);
            }
            return t;
        }
        var arr = [1];
        for (var warm = 0; warm < 200; warm++) {
            isA(arr); isAlias(arr); same(1, 2); min3(1, 2, 3); max4(1, 2, 3, 4);
        }
        hot(50);
    "#,
    );
    let counts = || {
        (
            crate::jit::TEST_CALL_TEMPLATES.with(std::cell::Cell::get),
            crate::jit::TEST_FAST_MATH_INTRINSICS.with(std::cell::Cell::get),
        )
    };
    // CALL_IC_EPOCH is process-wide: a concurrently running test that recompiles a chunk or
    // reclaims native code invalidates every cached call site, and the next calls here
    // legitimately refill through the generic path. Measure a run that ended at the epoch the
    // previous run started at (see code_point_intrinsic_tests).
    let (mut served, mut epoch_before, mut epoch_after) = ((0, 0), 0, 1);
    let mut previous_start = None;
    for _attempt in 0..200 {
        let before = counts();
        epoch_before = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var ok = true;
            for (var n = 0; n < 40; n++) {
                ok = ok && isA(arr) && isAlias(arr) && !same(1, 2) && min3(3, 1, 2) === 1 &&
                    max4(1, 2, 3, 4) === 4;
            }
            ok && hot(30) === 903
        "#
            ),
            "true"
        );
        let after = counts();
        served = (after.0 - before.0, after.1 - before.1);
        epoch_after = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if previous_start == Some(epoch_after) {
            break;
        }
        previous_start = Some(epoch_before);
    }
    // Per run: the in-line templates, three 40 times and two 30 times; the Math calls, two 40 and 30 times.
    let (templates, math) = served;
    assert!(
        templates >= 180 && math >= 140,
        "templates {templates}, math {math}, epoch {epoch_before}->{epoch_after}"
    );
    assert!(engine.interp.fn_frames.is_empty());
}
