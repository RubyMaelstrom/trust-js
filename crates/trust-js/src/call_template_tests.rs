//! W3 item 2c's call templates for proven natives (ECMA-262 at e28783d5, spec.html): Array.isArray
//! (IsArray, :5878), Object.is (SameValue, :6008), Math.max and Math.min with three and four
//! arguments (:33847, :33869), String.prototype.slice with one argument (:36827) and
//! String.prototype.codePointAt (:36270, CodePointAt :16721). Each program runs in every tier,
//! its call sites both outside loops (the JIT's shared stubs) and inside them; the expected
//! values are Node v24.22.0's, and references written in other operations check every
//! combination of the listed operands.
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

/// codePointAt and slice (one and two arguments) on one- and two-byte strings, surrogate pairs,
/// lone surrogates, views of both widths, a view ending on a leading surrogate whose owner
/// continues with its trailing one (the code point is the lone lead), and the empty string, at
/// negative, fractional, infinite, NaN, out-of-range and u32-boundary indices; then operands
/// the templates do not take, whose conversions run in order, and replaced builtins.
const SLICE_AND_CODE_POINT: &str = r#"
var log = [];
function rec(x) { log[log.length] = (typeof x === 'number' && x !== x ? 'NaN' : String(x)); }
function err(f) { try { return String(f()); } catch (e) { return e.constructor.name; } }
// References over charCodeAt and element reads, which the templates do not use.
function toIndex(n) { n = Number(n); return n !== n ? 0 : n === Infinity || n === -Infinity ? n : Math.trunc(n); }
function refPoint(s, at) {
  var i = toIndex(at);
  if (i < 0 || i >= s.length) return undefined;
  var first = s.charCodeAt(i);
  if (first < 0xD800 || first > 0xDBFF || i + 1 === s.length) return first;
  var second = s.charCodeAt(i + 1);
  return second < 0xDC00 || second > 0xDFFF ? first : (first - 0xD800) * 0x400 + (second - 0xDC00) + 0x10000;
}
function refSlice(s, start, end) {
  var len = s.length, from = toIndex(start), to = end === undefined ? len : toIndex(end);
  from = from < 0 ? Math.max(len + from, 0) : Math.min(from, len);
  to = to < 0 ? Math.max(len + to, 0) : Math.min(to, len);
  return from < to ? s.substring(from, to) : '';
}
var ascii = 'hello world', two = 'héllo wörld →', astral = 'a😀b😀c',
  lone = '\ude00x\ud83dy\ude00z\ud83d', longTwo = 'é'.repeat(300) + '😀' + 'q'.repeat(10),
  viewTwo = longTwo.slice(1, 305), viewAscii = ('x'.repeat(600) + 'yz').slice(100, 602),
  // A view ending on a leading surrogate whose owner continues with its trailing one.
  viewLead = ('a' + '😀'.repeat(200)).slice(0, 300), empty = '';
var strings = [ascii, two, astral, lone, longTwo, viewTwo, viewAscii, viewLead, empty];
var indices = [-1.5, -1, -0.5, -0, 0, 0.5, 1, 1.5, 2, 3, 4, 7, 8, 9, 10, 11, 13, 14, 15, 298, 299, 300, 301, 303, 304, 305,
  310, 311, 312, 400, 501, 502, NaN, Infinity, -Infinity, 4294967295, 4294967296, -4294967296, 1e300];
// One call site per form, outside any loop (the shared stubs); the callers loop.
function point(s, at) { return s.codePointAt(at); }
function slice1(s, at) { return s.slice(at); }
function slice2(s, at, end) { return s.slice(at, end); }
var bad = 0, calls = 0;
for (var r = 0; r < 3; r++) for (var i = 0; i < strings.length; i++) for (var j = 0; j < indices.length; j++) {
  var s = strings[i], at = indices[j], end = indices[(j * 5 + 3) % indices.length];
  if (point(s, at) !== refPoint(s, at)) bad++;
  if (slice1(s, at) !== refSlice(s, at)) bad++;
  if (slice2(s, at, end) !== refSlice(s, at, end)) bad++;
  calls += 3;
}
rec(bad + '/' + calls);
// The same calls inside loops keep their sequences in line.
function hotPoints(s) { var t = 0; for (var q = -1; q <= s.length; q++) { var p = s.codePointAt(q); t = (t * 31 + (p === undefined ? -7 : p)) | 0; } return t; }
function pointsOf(s) { var t = []; for (var q = -1; q <= s.length; q++) t.push(s.codePointAt(q)); return t.join(' '); }
function hotSlices(s) { var t = 0; for (var q = -s.length - 1; q <= s.length + 1; q++) t += s.slice(q).length * (q + 3); return t; }
var hotBad = 0, expectedPoints = [], expectedSums = [];
for (var i2 = 0; i2 < strings.length; i2++) {
  var s2 = strings[i2], expected = 0, sum = 0;
  for (var q = -1; q <= s2.length; q++) { var p = refPoint(s2, q); expected = (expected * 31 + (p === undefined ? -7 : p)) | 0; }
  for (var q2 = -s2.length - 1; q2 <= s2.length + 1; q2++) sum += Math.max(s2.length - (q2 < 0 ? Math.max(s2.length + q2, 0) : Math.min(q2, s2.length)), 0) * (q2 + 3);
  expectedPoints.push(expected); expectedSums.push(sum);
}
for (var r2 = 0; r2 < 4; r2++) for (var i3 = 0; i3 < strings.length; i3++) {
  if (hotPoints(strings[i3]) !== expectedPoints[i3] || hotSlices(strings[i3]) !== expectedSums[i3]) hotBad++;
}
rec(hotBad);
rec(pointsOf(lone)); rec(pointsOf(astral)); rec(point(viewLead, 299)); rec(point(viewLead, 298)); rec(point(viewTwo, 299));
rec(point(viewTwo, 300)); rec(point(viewTwo, 303)); rec(point(viewAscii, 500)); rec(point(viewAscii, 501)); rec(point(viewAscii, 502));
function codes(s) { var t = []; for (var k = 0; k < s.length; k++) t.push(s.charCodeAt(k).toString(16)); return t.join('.'); }
rec(slice1(two, -3)); rec(codes(slice1(astral, 2))); rec(slice1(viewTwo, 298).length); rec(slice1(lone, -1).charCodeAt(0));
rec(slice2(two, 1, -1)); rec(slice2(astral, -4, 3)); rec(slice2(viewAscii, 499, 1e9));
// Operands the templates do not take: the full call, its conversions in order.
var order = [];
var at3 = { valueOf: function () { order.push('i'); return 3; } };
rec(point(two, at3)); rec(slice1(two, at3)); rec(slice2(two, at3, { valueOf: function () { order.push('e'); return 5; } }));
rec(point(two, '2')); rec(slice1(ascii, '6')); rec(point(ascii, null)); rec(slice1(ascii, undefined)); rec(slice1(ascii, true));
rec(order.join(''));
rec(point(new String('😀'), 0)); rec(slice1(new String('abc'), 1)); rec(err(function () { return point(null, 0); }));
rec(String.prototype.codePointAt.call(12345, 1)); rec(String.prototype.slice.call(12345, 2));
var savedPoint = String.prototype.codePointAt, savedSlice = String.prototype.slice;
String.prototype.codePointAt = function () { return 'patched'; }; String.prototype.slice = function () { return 'sliced'; };
rec(point(ascii, 0)); rec(slice1(ascii, 1)); rec(slice2(ascii, 1, 2)); rec(pointsOf('ab'));
String.prototype.codePointAt = savedPoint; String.prototype.slice = savedSlice;
rec(point(ascii, 0)); rec(slice1(ascii, 1)); rec(slice2(ascii, 1, 2)); rec(pointsOf('ab'));
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

#[test]
fn slice_and_code_point_templates_match_the_full_calls() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, SLICE_AND_CODE_POINT),
            "0/3159,0, 56832 120 55357 121 56832 122 55357 , 97 128512 56832 98 128512 56832 99 ,55357,56832,128512,56832,113,121,122,undefined,d →,de00.62.d83d.de00.63,6,55357,éllo wörld ,,xyz,108,lo wörld →,lo,108,world,104,hello world,ello world,iiie,128512,bc,TypeError,50,345,patched,sliced,sliced,patched patched patched patched,104,ello world,e, 97 98 ",
            "{tier:?}"
        );
        // The program's views are views: the templates' view paths ran.
        let env = engine.interp.global_env;
        for name in ["viewTwo", "viewAscii", "viewLead"] {
            let value = engine.interp.get_var(name, &env).ok().unwrap();
            let crate::value::ValueView::Str(s) = value.view() else {
                panic!("{name} is a String")
            };
            assert!(s.view_owner().is_some(), "{name} is a view");
        }
        engine.interp.gc_collect();
    }
}

/// In compiled code the templates finish warm calls without the call-hit helper: Array.isArray as a method and through a plain call, Object.is and codePointAt count their
/// commits, Math.max and Math.min with three and four Numbers count as the other Math templates
/// do, and the one-argument slice reaches the intrinsic helper's light path,
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
        function point(s, n) { return s.codePointAt(n); }
        function slice1(s, n) { return s.slice(n); }
        function hot(n) {
            var t = 0;
            for (var q = 0; q < n; q++) {
                t += (Array.isArray(t) ? 1 : 0) + (Object.is(q, 3) ? 1 : 0) + q + Math.max(q, 1, 2) +
                    Math.min(q, 1, 2, 3) + 'h\u00e9llo'.codePointAt(q % 5) +
                    'abcdef'.slice(q % 6).length;
            }
            return t;
        }
        var arr = [1], two = 'h\u00e9llo \ud83d\ude00';
        for (var warm = 0; warm < 200; warm++) {
            isA(arr); isAlias(arr); same(1, 2); min3(1, 2, 3); max4(1, 2, 3, 4); point(two, 6);
            slice1(two, 2);
        }
        hot(50);
    "#,
    );
    let counts = || {
        (
            crate::jit::TEST_CALL_TEMPLATES.with(std::cell::Cell::get),
            crate::jit::TEST_FAST_MATH_INTRINSICS.with(std::cell::Cell::get),
            crate::bytecode::TEST_JIT_SLICE1_INTRINSICS.with(std::cell::Cell::get),
        )
    };
    // CALL_IC_EPOCH is process-wide: a concurrently running test that recompiles a chunk or
    // reclaims native code invalidates every cached call site, and the next calls here
    // legitimately refill through the generic path. Measure a run that ended at the epoch the
    // previous run started at (see code_point_intrinsic_tests).
    let (mut served, mut epoch_before, mut epoch_after) = ((0, 0, 0), 0, 1);
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
                    max4(1, 2, 3, 4) === 4 && point(two, 6) === 0x1F600 &&
                    slice1(two, 2) === 'llo \ud83d\ude00';
            }
            ok && hot(30) === 4992
        "#
            ),
            "true"
        );
        let after = counts();
        served = (after.0 - before.0, after.1 - before.1, after.2 - before.2);
        epoch_after = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        if previous_start == Some(epoch_after) {
            break;
        }
        previous_start = Some(epoch_before);
    }
    // Per run: the in-line templates, four 40 times and three 30 times; the Math calls, two 40 and 30
    // times; the one-argument slice, 40 and 30 times.
    let (templates, math, slices) = served;
    assert!(
        templates >= 250 && math >= 140 && slices >= 70,
        "templates {templates}, math {math}, slices {slices}, epoch {epoch_before}->{epoch_after}"
    );
    assert!(engine.interp.fn_frames.is_empty());
}
