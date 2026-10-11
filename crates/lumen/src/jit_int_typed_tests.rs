//! Differential regression tests for the register region lowering of int32 operators and for the
//! native typed-array element templates (ARM64). ECMA-262 e28783d5fc9d (local snapshot):
//! ToInt32/ToUint32 (§7.1.6-7), NumberBitwiseOp and the bitwise/shift operators (§6.1.6.1.9-11,
//! §13.9-13.12), TypedArrayGetElement/TypedArraySetElement (§10.4.5.15-16), IsValidIntegerIndex,
//! CanonicalNumericIndexString and NumericToRawBytes (§25.1.3.17). Each script checks the three
//! tiers against an independent oracle written in plain arithmetic (no bitwise operator and no
//! typed array), and the tiers against each other.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn evaluate(source: &str, tier: Tier) -> String {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
    }
}

/// Run `source` (which defines `RESULT`) in every tier; all must agree.
fn in_every_tier(source: &str) -> String {
    let source = format!("{source}\n;RESULT");
    let reference = evaluate(&source, Tier::Interp);
    for tier in [Tier::Bytecode, Tier::Jit] {
        assert_eq!(
            evaluate(&source, tier),
            reference,
            "{tier:?} differs from the interpreter"
        );
    }
    reference
}

/// `RESULT` is `<mismatches>:<checks>:<first mismatches>`.
fn assert_clean(result: &str, checks: usize) {
    let mut parts = result.splitn(3, ':');
    assert_eq!(parts.next(), Some("0"), "mismatches: {result}");
    let counted: usize = parts.next().unwrap().parse().unwrap();
    assert!(
        counted >= checks,
        "fixture checked only {counted} of {checks}: {result}"
    );
}

#[test]
fn int32_operators_match_the_specification_in_every_tier() {
    let before = super::regions::executed_regions();
    assert_clean(&in_every_tier(INT32), 70_000);
    assert!(
        super::regions::executed_regions() > before,
        "the operator loops must execute as register regions"
    );
}

#[test]
fn typed_array_elements_match_the_specification_in_every_tier() {
    assert_clean(&in_every_tier(TYPED), 3_000);
}

const INT32: &str = r####"
function i32(x) {
  x = +x;
  if (x !== x || x === Infinity || x === -Infinity) return 0;
  x = x < 0 ? Math.ceil(x) : Math.floor(x);
  x = x % 4294967296;
  if (x < 0) x += 4294967296;
  return (x >= 2147483648 ? x - 4294967296 : x) + 0;
}
function u32(x) { var r = i32(x); return r < 0 ? r + 4294967296 : r; }
function bitop(a, b, op) {
  var x = u32(a), y = u32(b), r = 0, p = 1;
  for (var k = 0; k < 32; k++) {
    var xb = x % 2, yb = y % 2;
    x = (x - xb) / 2; y = (y - yb) / 2;
    var bit = op === 0 ? xb * yb : op === 1 ? (xb + yb > 0 ? 1 : 0) : (xb + yb) % 2;
    r += bit * p; p *= 2;
  }
  return i32(r);
}
// op: 0 &, 1 |, 2 ^, 3 <<, 4 >>, 5 >>>
function reference(op, a, b) {
  if (op < 3) return bitop(a, b, op);
  var count = Math.pow(2, u32(b) % 32);
  if (op === 3) return i32(u32(a) * count);
  if (op === 4) return Math.floor(i32(a) / count) + 0;
  return Math.floor(u32(a) / count);
}
var values = [0, -0, 1, -1, 2, 3, 7, 31, 32, 33, 63, 255, 256, 65535, 65536, 2147483647, 2147483648,
  2147483649, -2147483648, -2147483649, 4294967295, 4294967296, 4294967297, 1.5, -1.5, 0.5, -0.5, 2.9,
  -2.9, NaN, Infinity, -Infinity, 1e10, -1e10, 9007199254740993, 18446744073709552000, 1e300, -1e300,
  '7', true, false, null, undefined, '0x10', ' 12 ', {valueOf: function () { return 5.5; }}];
var bad = [], checked = 0;
function note(label, a, b, got, want) {
  checked++;
  if (got !== want) bad.push(label + '(' + String(a) + ',' + String(b) + ')=' + got + '!=' + want);
}
function pair(op, a, b) { return reference(op, +a, +b); }

// Variable operands, all six operators in a loop with a branch (the CFG region tier).
function variables(a, b) {
  var r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, n = 0;
  for (n = 0; n < 2; n = n + 1 | 0) {
    r0 = a & b; r1 = a | b; r2 = a ^ b; r3 = a << b; r4 = a >> b; r5 = a >>> b;
    if (n === 1) r0 = r0 + 0 | 0;
  }
  return [r0, r1, r2, r3, r4, r5];
}
for (var rep = 0; rep < 3; rep++) {
  for (var i = 0; i < values.length; i++) {
    for (var j = 0; j < values.length; j++) {
      var a = values[i], b = values[j];
      var got = variables(a, b);
      for (var op = 0; op < 6; op++) note('var' + op, a, b, got[op], pair(op, a, b));
    }
  }
}

// Constant operands on either side, for every operator.
var constants = [0, 1, -1, 2, 3, 7, 31, 32, 33, 255, 65535, 65536, 2147483647, 2147483648, -2147483648,
  4294967295, 4278255360, 1431655765, 1.5, -0.5, 4294967296, 2147483647.5, 4294967297];
var ops = ['&', '|', '^', '<<', '>>', '>>>'];
var source = '';
for (var op = 0; op < 6; op++) {
  for (var c = 0; c < constants.length; c++) {
    var lit = '(' + constants[c] + ')';
    source += 'function R' + op + '_' + c + '(a){var s=0,n=0;for(n=0;n<2;n=n+1|0){if(n===1)s=s+1|0;s=(s+(a ' + ops[op] + ' ' + lit + '))|0;}return s;}\n';
    source += 'function L' + op + '_' + c + '(a){var s=0,n=0;for(n=0;n<2;n=n+1|0){if(n===1)s=s+1|0;s=(s+(' + lit + ' ' + ops[op] + ' a))|0;}return s;}\n';
    source += 'function B' + op + '_' + c + '(){var s=0,n=0;for(n=0;n<2;n=n+1|0){if(n===1)s=s+1|0;s=(s+(' + lit + ' ' + ops[op] + ' ' + lit + '))|0;}return s;}\n';
  }
}
(0, eval)(source);
function accumulate(op, r) { var s = 0; for (var n = 0; n < 2; n++) { if (n === 1) s = i32(s + 1); s = i32(s + r); } return s; }
for (var rep = 0; rep < 3; rep++) {
  for (var op = 0; op < 6; op++) {
    for (var c = 0; c < constants.length; c++) {
      var k = constants[c];
      for (var i = 0; i < values.length; i++) {
        var a = values[i];
        note('R' + op + '_' + k, a, k, globalThis['R' + op + '_' + c](a), accumulate(op, pair(op, a, k)));
        note('L' + op + '_' + k, k, a, globalThis['L' + op + '_' + c](a), accumulate(op, pair(op, k, a)));
      }
      note('B' + op + '_' + k, k, k, globalThis['B' + op + '_' + c](), accumulate(op, pair(op, k, k)));
    }
  }
}

// ~x, unary minus and identity forms (x|0, x&-1, x^0, x<<0, x>>0, x>>>0).
function unary(a) {
  var s = 0, n = 0, t = 0;
  for (n = 0; n < 2; n = n + 1 | 0) {
    if (n === 1) s = s + 1 | 0;
    s = (s + ~a) | 0;
    t = (a | 0) + (a & -1) + (a ^ 0) + (a << 0) + (a >> 0);
    s = (s + t) | 0;
    s = (s + (a >>> 0)) | 0;
  }
  return s;
}
for (var rep = 0; rep < 3; rep++) {
  for (var i = 0; i < values.length; i++) {
    var a = values[i], x = +a, ia = i32(x), r = 0;
    for (var n = 0; n < 2; n++) {
      if (n === 1) r = i32(r + 1);
      r = i32(r + (-ia - 1));
      r = i32(r + 5 * ia);
      r = i32(r + u32(x));
    }
    note('unary', a, 0, unary(a), r);
  }
}

// Operand conversions run once per operand and in order (ECMA-262 ApplyStringOrNumericBinaryOperator).
var trace = [];
function loud(name, v) { return {valueOf: function () { trace.push(name); return v; }}; }
function ordered(a, b) { var r = 0, n = 0; for (n = 0; n < 2; n = n + 1 | 0) { r = (r + ((a & b) | 0)) | 0; } return r; }
for (var rep = 0; rep < 3; rep++) { trace.length = 0; ordered(loud('a', 6), loud('b', 3)); note('order', 0, 0, trace.join(''), 'abab'); }

var RESULT = bad.length + ':' + checked + ':' + bad.slice(0, 8).join(';');
"####;

const TYPED: &str = r####"
var bad = [], checked = 0;
function note(label, got, want) {
  checked++;
  if (!Object.is(got, want)) bad.push(label + ':' + String(got) + '!=' + String(want));
}
function trunc(x) { return x < 0 ? Math.ceil(x) : Math.floor(x); }
// ECMA-262 ToIntN/ToUintN/ToUint8Clamp/ToFloat32 (NumericToRawBytes) in plain arithmetic.
function wrap(v, bits, signed) {
  v = +v;
  if (v !== v || v === Infinity || v === -Infinity) return 0;
  var m = Math.pow(2, bits), r = trunc(v) % m;
  if (r < 0) r += m;
  if (signed && r >= m / 2) r -= m;
  return r + 0;
}
function clamp8(v) {
  v = +v;
  if (v !== v || v <= 0) return 0;
  if (v >= 255) return 255;
  var f = Math.floor(v);
  if (v - f < 0.5) return f;
  if (v - f > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}
var kinds = [
  ['Int8Array', Int8Array, function (v) { return wrap(v, 8, true); }],
  ['Uint8Array', Uint8Array, function (v) { return wrap(v, 8, false); }],
  ['Uint8ClampedArray', Uint8ClampedArray, clamp8],
  ['Int16Array', Int16Array, function (v) { return wrap(v, 16, true); }],
  ['Uint16Array', Uint16Array, function (v) { return wrap(v, 16, false); }],
  ['Int32Array', Int32Array, function (v) { return wrap(v, 32, true); }],
  ['Uint32Array', Uint32Array, function (v) { return wrap(v, 32, false); }],
  ['Float32Array', Float32Array, function (v) { return Math.fround(+v); }],
  ['Float64Array', Float64Array, function (v) { return +v; }]
];
var values = [0, -0, 1, -1, 0.5, -0.5, 1.5, 2.5, 3.5, 127, 128, 129, 254.5, 255, 255.5, 256, 257, -128, -129,
  32767, 32768, 65535, 65536, 2147483647, 2147483648, 2147483649, -2147483648, -2147483649, 4294967295,
  4294967296, 4294967297, 1e10, -1e10, 9007199254740993, 18446744073709552000, 1e300, -1e300, 1.1, 16777217,
  3.4028236e38, 1e-50, 5e-324, NaN, Infinity, -Infinity, '5', true, null, undefined, '0x7f', {valueOf: function () { return 258.5; }}];
var N = 16;

function build(K) {
  var arr = new K(N);
  function storeAll(vals, base) { for (var i = 0; i < vals.length; i = i + 1 | 0) arr[(i + base) & 15] = vals[i]; }
  function loadAll(out, base) { for (var i = 0; i < N; i = i + 1 | 0) out[i] = arr[(i + base) & 15]; }
  function storeParam(a, vals, base) { for (var i = 0; i < vals.length; i = i + 1 | 0) a[(i + base) & 15] = vals[i]; }
  function loadParam(a, out, base) { for (var i = 0; i < N; i = i + 1 | 0) out[i] = a[(i + base) & 15]; }
  return {arr: arr, storeAll: storeAll, loadAll: loadAll, storeParam: storeParam, loadParam: loadParam};
}

// Every element type, every value, closure and parameter receivers, interleaved with loads.
for (var k = 0; k < kinds.length; k++) {
  var name = kinds[k][0], K = kinds[k][1], expect = kinds[k][2];
  var t = build(K), other = new K(N), out = [], out2 = [];
  for (var rep = 0; rep < 3; rep++) {
    for (var v = 0; v < values.length; v += 16) {
      var chunk = values.slice(v, v + 16);
      t.storeAll(chunk, v);
      t.storeParam(other, chunk, v + 3);
      t.loadAll(out, v);
      t.loadParam(other, out2, v + 3);
      for (var i = 0; i < chunk.length; i++) {
        var want = expect(chunk[i]);
        note(name + '.closure[' + String(chunk[i]) + ']', out[i], want);
        note(name + '.param[' + String(chunk[i]) + ']', out2[i], want);
      }
    }
  }
}

// Mixed views over one buffer (little-endian host layout), Float32 rounding and NaN payloads.
(function () {
  var buf = new ArrayBuffer(64);
  var u8 = new Uint8Array(buf), i8 = new Int8Array(buf), u16 = new Uint16Array(buf), i16 = new Int16Array(buf),
      u32 = new Uint32Array(buf), i32 = new Int32Array(buf), f32 = new Float32Array(buf), f64 = new Float64Array(buf);
  function fill(n) { for (var i = 0; i < n; i = i + 1 | 0) i32[i & 15] = (i * 2654435761) | 0; }
  function sum(n) {
    var s = 0;
    for (var i = 0; i < n; i = i + 1 | 0) {
      s = (s + (u8[i & 63] | 0) + (i8[(i + 1) & 63] | 0) + (u16[i & 31] | 0) + (i16[(i + 2) & 31] | 0)) | 0;
      s = (s + (u32[i & 15] >>> 0)) | 0;
    }
    return s;
  }
  for (var rep = 0; rep < 3; rep++) {
    fill(16);
    var want = 0, bytes = new Uint8Array(64);
    for (var i = 0; i < 16; i++) { var w = (i * 2654435761) | 0; bytes[4 * i] = w & 255; bytes[4 * i + 1] = (w >> 8) & 255; bytes[4 * i + 2] = (w >> 16) & 255; bytes[4 * i + 3] = (w >> 24) & 255; }
    for (var i = 0; i < 64; i++) note('bytes' + i, u8[i], bytes[i]);
    var s = 0;
    for (var i = 0; i < 64; i++) {
      var b8 = bytes[i & 63], sb = bytes[(i + 1) & 63]; if (sb > 127) sb -= 256;
      var h = bytes[(2 * (i & 31))] + 256 * bytes[2 * (i & 31) + 1];
      var g = bytes[(2 * ((i + 2) & 31))] + 256 * bytes[2 * ((i + 2) & 31) + 1]; if (g > 32767) g -= 65536;
      var d = bytes[4 * (i & 15)] + 256 * bytes[4 * (i & 15) + 1] + 65536 * bytes[4 * (i & 15) + 2] + 16777216 * bytes[4 * (i & 15) + 3];
      s = (s + b8 + sb + h + g) | 0; s = (s + d) | 0;
    }
    note('sum', sum(64), s);
  }
  // Float32: round to nearest even, overflow to Infinity, underflow to zero, NaN stays NaN.
  function f32store(a, v) { a[0] = v; a[1] = v; return a[0]; }
  for (var rep = 0; rep < 3; rep++) {
    var f = new Float32Array(2);
    var cases = [16777217, 16777219, 1.1, 3.4028235677973366e38, 3.4028235677973367e38, 1e39, -1e39, 1e-46, 7e-46, NaN, -0, Infinity];
    for (var i = 0; i < cases.length; i++) note('f32', f32store(f, cases[i]), Math.fround(cases[i]));
  }
  // A NaN written through another view is read back as NaN and re-stored as NaN.
  u32[0] = 0x7fc00001; u32[1] = 0xffffffff; f32[2] = f32[0];
  note('nan', f32[0] !== f32[0] && f32[1] !== f32[1] && f32[2] !== f32[2], true);
  f64[4] = NaN; note('nan64', f64[4] !== f64[4], true);
  f64[5] = -0; note('negzero', 1 / f64[5], -Infinity);
})();

// Keys: invalid integer indices read undefined and write nothing, never consult the prototype chain.
(function () {
  Object.prototype[20] = 'proto';
  var u8 = new Uint8Array(16), keys = [0, 15, 16, 20, 100, -1, -0, 1.5, 4294967295, 4294967296, 4294967297, 9007199254740992,
    NaN, Infinity, -Infinity, '3', '-0', '1.5', '4294967296', 7.0, 1e21, 0.1 + 0.2];
  function read(a, k) { return a[k]; }
  function write(a, k, v) { a[k] = v; }
  function readInLoop(a, ks) { var out = []; for (var i = 0; i < ks.length; i = i + 1 | 0) out[i] = a[ks[i]]; return out; }
  function writeInLoop(a, ks) { for (var i = 0; i < ks.length; i = i + 1 | 0) a[ks[i]] = 77; }
  for (var rep = 0; rep < 3; rep++) {
    for (var i = 0; i < u8.length; i++) u8[i] = i + 1;
    var got = readInLoop(u8, keys);
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i], want;
      var n = typeof k === 'string' ? (String(+k) === k ? +k : NaN) : k;
      if (n === n && Number.isInteger(n) && !Object.is(n, -0) && n >= 0 && n < 16) want = n + 1;
      else if (Object.is(n, -0) && typeof k !== 'string') want = 1;
      else want = undefined;
      note('read[' + String(k) + ']', got[i], want);
      note('read1[' + String(k) + ']', read(u8, k), want);
    }
    for (var i = 0; i < u8.length; i++) u8[i] = 0;
    writeInLoop(u8, keys);
    var written = [];
    for (var i = 0; i < 16; i++) written.push(u8[i]);
    // exactly the valid keys wrote: 0, 15, -0 (element 0), '3' and 7.0
    var wantWritten = [];
    for (var i = 0; i < 16; i++) wantWritten.push(i === 0 || i === 15 || i === 3 || i === 7 ? 77 : 0);
    note('writes', written.join(), wantWritten.join());
    note('proto20', read(u8, 20), undefined);
    write(u8, 100, 5); write(u8, -1, 5); write(u8, 1.5, 5);
    note('nowrite', u8.length + ':' + Object.keys(u8).length, '16:16');
  }
  delete Object.prototype[20];
  // Non-numeric and non-canonical strings are ordinary property keys (CanonicalNumericIndexString).
  var plain = new Uint8Array(4);
  function setNamed(a, k, v) { a[k] = v; }
  function getNamed(a, k) { return a[k]; }
  for (var rep = 0; rep < 3; rep++) {
    setNamed(plain, 'x', 11); setNamed(plain, '03', 12); setNamed(plain, 1, 13);
    note('named', getNamed(plain, 'x') + ':' + getNamed(plain, '03') + ':' + getNamed(plain, 1), '11:12:13');
  }
  // Value conversion happens even when the index is invalid (TypedArraySetElement step 1/2).
  var calls = 0, boxed = {valueOf: function () { calls++; return 9; }};
  for (var rep = 0; rep < 3; rep++) { write(u8, 100, boxed); write(u8, 2, boxed); }
  note('valueOf', calls, 6);
  note('boxed', u8[2], 9);
})();

// Detached, resizable and length-tracking buffers, including changes in the middle of a loop.
(function () {
  function sumFirst(a, n) { var s = 0; for (var i = 0; i < n; i = i + 1 | 0) { var v = a[i]; s = (s + (v === undefined ? 1000 : v)) | 0; } return s; }
  function fillAll(a, n, v) { for (var i = 0; i < n; i = i + 1 | 0) a[i] = v; }
  for (var rep = 0; rep < 3; rep++) {
    var buffer = new ArrayBuffer(16);
    var view = new Uint8Array(buffer);
    fillAll(view, 16, 3);
    note('attached', sumFirst(view, 20), 3 * 16 + 4 * 1000);
    buffer.transfer();
    note('detached.read', sumFirst(view, 20), 20 * 1000);
    fillAll(view, 20, 9);
    note('detached.length', view.length + ':' + view.byteLength + ':' + view.byteOffset, '0:0:0');
    note('detached.flag', buffer.detached, true);

    // detach in the middle of a loop
    var b2 = new ArrayBuffer(32), v2 = new Int32Array(b2), total = 0;
    for (var i = 0; i < 8; i++) {
      if (i === 5) b2.transfer();
      var x = v2[i];
      total += x === undefined ? 100 : x;
      v2[i] = 7;
    }
    note('mid-detach', total, 100 * 3);

    // fixed-length view over a resizable buffer: out of bounds after a shrink, valid again after growth
    var rab = new ArrayBuffer(16, {maxByteLength: 64});
    var fixed = new Uint8Array(rab, 4, 8), track = new Uint8Array(rab, 4), whole = new Int32Array(rab);
    fillAll(fixed, 8, 5);
    note('rab.fixed', sumFirst(fixed, 8), 40);
    note('rab.track', sumFirst(track, 14), 40 + 4 * 0 + 1000 * 2);
    rab.resize(8);   // fixed [4,12) no longer fits
    note('rab.oob.fixed', sumFirst(fixed, 8) + ':' + fixed.length + ':' + fixed.byteLength, (8 * 1000) + ':0:0');
    note('rab.oob.track', sumFirst(track, 6) + ':' + track.length, (5 * 4 + 2 * 1000) + ':4');
    fillAll(fixed, 8, 9);
    note('rab.oob.write', whole[1], 0x05050505);
    rab.resize(32);
    note('rab.grown.fixed', fixed.length, 8);
    note('rab.grown.track', track.length, 28);
    note('rab.grown.whole', whole.length, 8);
    fillAll(track, 28, 1);
    note('rab.grown.sum', sumFirst(whole, 8), 7 * 0x01010101);
    // shrinking in the middle of a loop over a tracking view
    var rab2 = new ArrayBuffer(10, {maxByteLength: 16}), t32 = new Int16Array(rab2), seen = 0;
    for (var i = 0; i < 8; i++) {
      if (i === 3) rab2.resize(4);
      var y = t32[i];
      seen += y === undefined ? 100 : 1;
      t32[i] = 1;
    }
    note('mid-shrink', seen, 3 + 5 * 100);
  }
  // Views with a byteOffset and subarrays address their own element zero.
  var base = new Float64Array(32);
  for (var i = 0; i < 32; i++) base[i] = i + 0.5;
  var sub = base.subarray(5, 12), off = new Float64Array(base.buffer, 8 * 7, 4);
  function addAll(a, n) { var s = 0; for (var i = 0; i < n; i = i + 1 | 0) s = s + a[i]; return s; }
  for (var rep = 0; rep < 3; rep++) {
    note('subarray', addAll(sub, 7), 5.5 + 6.5 + 7.5 + 8.5 + 9.5 + 10.5 + 11.5);
    note('offset', addAll(off, 5), 7.5 + 8.5 + 9.5 + 10.5 + NaN);
  }
})();

// BigInt64Array and BigUint64Array keep BigInt semantics through the same sites.
(function () {
  var big = new BigInt64Array(4), ubig = new BigUint64Array(4);
  function put(a, i, v) { a[i] = v; }
  function get(a, i) { return a[i]; }
  for (var rep = 0; rep < 3; rep++) {
    put(big, 1, -5n); put(ubig, 1, -1n); put(big, 2, 2n ** 63n); put(ubig, 2, 2n ** 64n + 3n);
    note('bigint', [get(big, 1), get(ubig, 1), get(big, 2), get(ubig, 2), get(big, 9)].join(), '-5,18446744073709551615,-9223372036854775808,3,');
    var threw = false;
    try { put(big, 0, 1); } catch (e) { threw = e instanceof TypeError; }
    note('bigint.number', threw, true);
  }
})();

// Stores with a coercion that detaches the buffer while its value is converted.
(function () {
  for (var rep = 0; rep < 3; rep++) {
    var buffer = new ArrayBuffer(8), view = new Uint8Array(buffer);
    var evil = {valueOf: function () { buffer.transfer(); return 5; }};
    function put(a, i, v) { a[i] = v; }
    put(view, 0, evil);
    note('evil', view.length + ':' + view[0], '0:undefined');
  }
})();

var RESULT = bad.length + ':' + checked + ':' + bad.slice(0, 40).join(';');
"####;
