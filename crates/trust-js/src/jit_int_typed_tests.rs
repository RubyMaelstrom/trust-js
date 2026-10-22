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

/// Typed arrays reached through free names (closure variables and globals): the register regions
/// read the binding after a pure key and fall back to the ordinary templates for everything else.
#[test]
fn element_access_through_free_names_matches_in_every_tier() {
    let before = super::regions::executed_regions();
    let committed = super::regions::name_element_commits();
    assert_clean(&in_every_tier(FUSED), 80);
    assert!(
        super::regions::executed_regions() > before,
        "the loops must execute as register regions"
    );
    assert!(
        super::regions::name_element_commits()[0] > committed[0],
        "a fused name-element read must commit after packed receiver decoding"
    );
}

#[test]
fn packed_name_element_reads_and_writes_commit_for_captured_and_global_receivers() {
    for captured in [false, true] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let body = r#"
            var named=new Float64Array(8);
            function read(n){n=n|0;var sum=0;for(var k=0;(k|0)<(n|0);k=(k+1)|0)sum+=named[k&7];return sum;}
            function write(n){n=n|0;for(var k=0;(k|0)<(n|0);k=(k+1)|0)named[k&7]=k+0.25;return named[7];}
        "#;
        let source = if captured {
            format!("var pair=(function(){{{body}return [read,write];}})();var read=pair[0],write=pair[1];")
        } else {
            body.to_owned()
        };
        let value = |engine: &mut Engine, source: &str| match engine.eval(source, false).unwrap() {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        };
        value(&mut engine, &source);
        value(
            &mut engine,
            "for(var warm=0;warm<8;warm++){write(64);read(32);}0",
        );
        let before = super::regions::name_element_commits();
        assert_eq!(value(&mut engine, "read(32)"), "1912");
        assert!(
            super::regions::name_element_commits()[0] > before[0],
            "committed fused packed read, captured={captured}"
        );
        let before = super::regions::name_element_commits();
        assert_eq!(value(&mut engine, "write(32)"), "31.25");
        assert!(
            super::regions::name_element_commits()[1] > before[1],
            "committed fused packed write, captured={captured}"
        );
    }
}

/// Numeric frame slots outside the eight register homes: reads and stores in register regions
/// stay Number-only fast paths, and a slot holding anything else leaves the region.
#[test]
fn numeric_frame_slots_match_the_specification_in_every_tier() {
    let before = super::regions::executed_regions();
    assert_clean(&in_every_tier(LOCALS), 200);
    assert!(
        super::regions::executed_regions() > before,
        "the loops must execute as register regions"
    );
}

const LOCALS: &str = r####"
// Numeric frame slots outside the register homes: Number reads and stores in loops with branches,
// a slot that first holds an undefined, a string, an object or a BigInt, a read before the
// declaration's initialization (let), and values at the edges.
var bad = [], checked = 0;
function note(label, got, want) { checked++; if (!Object.is(got, want)) bad.push(label + ':' + String(got) + '!=' + String(want)); }
function i32(x) { x = +x; if (x !== x || x === Infinity || x === -Infinity) return 0; x = x < 0 ? Math.ceil(x) : Math.floor(x); x = x % 4294967296; if (x < 0) x += 4294967296; return (x >= 2147483648 ? x - 4294967296 : x) + 0; }
var K = 14, names = [];
for (var k = 0; k < K; k++) names.push('v' + k);
// Statement forms: [dst, lhs, op, rhs], executed under an if/else so the CFG tier lowers it.
var forms = [];
for (var k = 0; k < 40; k++) forms.push([k * 7 % K, k * 3 % K, ['+', '-', '^', '&', '|', '<<', '>>', '>>>', '*'][k % 9], (k * 5 + 1) % K]);
function source(branchy) {
  var body = '';
  forms.forEach(function (f, n) {
    var stmt = names[f[0]] + ' = (' + names[f[1]] + ' ' + f[2] + ' ' + names[f[3]] + ') | 0;';
    body += branchy ? 'if ((i & ' + (1 << (n % 3)) + ') == 0) { ' + stmt + ' } else { ' + names[(f[0] + 1) % K] + ' = (' + names[(f[0] + 1) % K] + ' + ' + (n + 1) + ') | 0; }\n' : stmt + '\n';
  });
  return 'function run(n, init) {\n  var i = 0;\n  var ' + names.map(function (nm, j) { return nm + ' = init[' + j + ']'; }).join(', ') + ';\n' +
    '  for (i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) {\n' + body + '  }\n  return [' + names.join(',') + '];\n}\n';
}
function oracle(n, init, branchy) {
  var v = init.map(function (x) { return x; });
  for (var i = 0; i < n; i++) forms.forEach(function (f, m) {
    var a = +v[f[1]], b = +v[f[3]], r;
    switch (f[2]) {
      case '+': r = a + b; break; case '-': r = a - b; break; case '*': r = a * b; break;
      case '^': r = i32(a) ^ i32(b); break; case '&': r = i32(a) & i32(b); break; case '|': r = i32(a) | i32(b); break;
      case '<<': r = i32(a) << (i32(b) & 31); break; case '>>': r = i32(a) >> (i32(b) & 31); break; default: r = i32(a) >>> (i32(b) & 31);
    }
    if (f[2] === '*') r = Math.imul(i32(a), i32(b)) === i32(a * b) ? i32(a * b) : r; // exact products only
    if (!branchy || (i & (1 << (m % 3))) == 0) v[f[0]] = i32(r);
    else v[(f[0] + 1) % K] = i32(+v[(f[0] + 1) % K] + m + 1);
  });
  return v;
}
var inits = [
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
  [0, -1, 2147483647, -2147483648, 4294967295, 65535, 1.5, -2.5, 255, 256, 3, 5, 7, 11],
  [100000, 3, 70000, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
];
['plain', 'branchy'].forEach(function (kind) {
  var branchy = kind === 'branchy';
  (0, eval)(source(branchy));
  var run = globalThis.run;
  for (var rep = 0; rep < 3; rep++) inits.forEach(function (init, t) {
    var want = oracle(37, init, branchy), got = run(37, init);
    for (var j = 0; j < K; j++) {
      // products may differ by double rounding beyond 2^53, which the oracle does not model
      if (kind === 'plain' && t === 2) continue;
      note(kind + t + '.v' + j, got[j], want[j]);
    }
  });
});

// A slot that is not a Number when it is read or overwritten leaves the register region.
(function () {
  function mixed(mode, n) {
    var i = 0, a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8, j = 9, k = 10, l = 11, m = 12, x;
    if (mode === 1) x = 'str'; else if (mode === 2) x = {valueOf: function () { return 7; }}; else if (mode === 3) x = 12n; else if (mode === 4) x = null;
    for (i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) {
      a = (a + b) | 0; b = (b ^ c) | 0; c = (c + d) | 0; d = (d + e) | 0; e = (e + f) | 0; f = (f ^ g) | 0;
      g = (g + h) | 0; h = (h + j) | 0; j = (j + k) | 0; k = (k ^ l) | 0; l = (l + m) | 0; m = (m + a) | 0;
      if ((i & 1) == 0) { x = (a + 1) | 0; } else { x = x + 1; }
    }
    return [a, b, c, d, e, f, g, h, j, k, l, m, typeof x, x].join();
  }
  var expected = {};
  for (var rep = 0; rep < 3; rep++) for (var mode = 0; mode < 5; mode++) {
    var r = mixed(mode, 9);
    if (rep === 0) expected[mode] = r; else note('mixed' + mode, r, expected[mode]);
  }
  note('mixed.str', mixed(1, 3).split(',').slice(-2).join(), 'number,' + mixed(1, 3).split(',').pop());
  // let in its temporal dead zone: the read throws before anything else happens
  function tdz(n) {
    var i = 0, a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8, j = 9, k = 10;
    for (i = 0; i < n; i++) { a = (a + b) | 0; b = (b + c) | 0; c = (c + d) | 0; d = (d + e) | 0; e = (e + f) | 0; f = (f + g) | 0; g = (g + h) | 0; h = (h + j) | 0; j = (j + k) | 0; k = (k + 1) | 0; if (i == 1) { var s = (z + a) | 0; } }
    let z = 5;
    return a;
  }
  for (var rep = 0; rep < 3; rep++) { var caught = ''; try { tdz(4); } catch (e) { caught = e.constructor.name; } note('tdz', caught, 'ReferenceError'); }
})();
var RESULT = bad.length + ':' + checked + ':' + bad.slice(0, 12).join(';');
"####;

/// Stores of computed Numbers through free names at every element type (NaN, -0, infinities,
/// values beyond each range), coercing stores, and the name resolution forms that cannot take
/// the fused path (an accessor on the global object, `with`, a parameter, a binding rebound to
/// an Array in the middle of a loop).
#[test]
fn computed_stores_and_unfusable_names_match_in_every_tier() {
    let before = super::regions::executed_regions();
    let committed = super::regions::name_element_commits();
    assert_clean(&in_every_tier(FUSED_STORES), 10_000);
    assert!(
        super::regions::executed_regions() > before,
        "the loops must execute as register regions"
    );
    assert!(
        super::regions::name_element_commits()[1] > committed[1],
        "a fused name-element store must commit after packed receiver decoding"
    );
}

/// Comparisons, branches and truthiness of bitwise results (int32 and uint32 operands and
/// constants) in register regions, against an oracle in plain arithmetic.
#[test]
fn integer_comparisons_and_branches_match_in_every_tier() {
    let before = super::regions::executed_regions();
    assert_clean(&in_every_tier(INT_COMPARISONS), 300_000);
    assert!(
        super::regions::executed_regions() > before,
        "the loops must execute as register regions"
    );
}

const INT_COMPARISONS: &str = r####"
// Comparisons, branches and truthiness of bitwise results (int32 and uint32 operands, constants),
// in register regions: every operator against an oracle in plain arithmetic.
var bad = [], checked = 0;
function note(label, got, want) { checked++; if (!Object.is(got, want)) bad.push(label + ':' + String(got) + '!=' + String(want)); }
function i32(x) { x = +x; if (x !== x || x === Infinity || x === -Infinity) return 0; x = x < 0 ? Math.ceil(x) : Math.floor(x); x = x % 4294967296; if (x < 0) x += 4294967296; return (x >= 2147483648 ? x - 4294967296 : x) + 0; }
function u32(x) { var r = i32(x); return r < 0 ? r + 4294967296 : r; }
function xor32(a, b) { var x = u32(a), y = u32(b), r = 0, p = 1; for (var k = 0; k < 32; k++) { var xb = x % 2, yb = y % 2; x = (x - xb) / 2; y = (y - yb) / 2; r += ((xb + yb) % 2) * p; p *= 2; } return i32(r); }
// operand forms: [source, value function of (x, y)]
var forms = [
  ['(x|0)', function (x, y) { return i32(x); }],
  ['(y|0)', function (x, y) { return i32(y); }],
  ['(x>>>0)', function (x, y) { return u32(x); }],
  ['(y>>>0)', function (x, y) { return u32(y); }],
  ['(x&255)', function (x, y) { return u32(x) % 256; }],
  ['(x>>>24)', function (x, y) { return Math.floor(u32(x) / 16777216); }],
  ['(x<<8)', function (x, y) { return i32(i32(x) * 256); }],
  ['(x^y)', function (x, y) { return xor32(x, y); }],
  ['~x', function (x, y) { return -i32(x) - 1; }],
  ['7', function () { return 7; }],
  ['-1', function () { return -1; }],
  ['2147483648', function () { return 2147483648; }],
  ['4294967295', function () { return 4294967295; }],
  ['0', function () { return 0; }]
];
var ops = ['<', '<=', '>', '>=', '==', '!=', '===', '!=='];
function compare(op, a, b) {
  switch (op) { case '<': return a < b; case '<=': return a <= b; case '>': return a > b; case '>=': return a >= b; case '==': case '===': return a === b; default: return a !== b; }
}
var values = [0, 1, -1, 2, 255, 256, 65535, 2147483647, 2147483648, 2147483649, -2147483648, -2147483649, 4294967295, 4294967296, 4294967297, 1.5, -1.5, 3000000000, NaN, Infinity, -Infinity, 100000, -100000];
var source = '';
var table = [];
for (var li = 0; li < forms.length; li++) for (var ri = 0; ri < forms.length; ri++) {
  if (li % 3 === 1 && ri % 3 === 2) continue;           // thin the cross product
  for (var oi = 0; oi < ops.length; oi++) {
    var name = 'C' + table.length;
    var cmp = forms[li][0] + ' ' + ops[oi] + ' ' + forms[ri][0];
    // branch form, value form, truthiness of the left operand
    source += 'function ' + name + '(x, y) { var r = 0, n = 0, t = false; for (n = 0; n < 2; n = n + 1 | 0) { if (' + cmp + ') r = r + 1 | 0; else r = r + 3 | 0; t = (' + cmp + '); r = (r + (t ? 10 : 30)) | 0; if (' + forms[li][0] + ') r = r + 100 | 0; } return r; }\n';
    table.push([name, li, ri, oi]);
  }
}
(0, eval)(source);
for (var rep = 0; rep < 3; rep++) {
  for (var t = 0; t < table.length; t++) {
    var fn = globalThis[table[t][0]], li = table[t][1], ri = table[t][2], oi = table[t][3];
    for (var i = 0; i < values.length; i += 2) for (var j = 0; j < values.length; j += 3) {
      var x = values[i], y = values[j];
      var l = forms[li][1](x, y), r = forms[ri][1](x, y), want = compare(ops[oi], l, r), r1 = 0;
      for (var n = 0; n < 2; n++) { r1 += want ? 1 : 3; r1 += want ? 10 : 30; if (l !== 0) r1 += 100; }
      note(table[t][0] + '(' + x + ',' + y + ')', fn(x, y), r1);
    }
  }
}
var RESULT = bad.length + ':' + checked + ':' + bad.slice(0, 10).join(';');
"####;

const FUSED_STORES: &str = r####"
// Stores of computed Numbers through free names (the fused set), at every element type, and the
// name resolution forms that must not take the fused path.
var bad = [], checked = 0, digest = [];
function note(label, got, want) { checked++; if (!Object.is(got, want)) bad.push(label + ':' + String(got) + '!=' + String(want)); }
function trunc(x) { return x < 0 ? Math.ceil(x) : Math.floor(x); }
function wrap(v, bits, signed) {
  v = +v; if (v !== v || v === Infinity || v === -Infinity) return 0;
  var m = Math.pow(2, bits), r = trunc(v) % m; if (r < 0) r += m; if (signed && r >= m / 2) r -= m; return r + 0;
}
function clamp8(v) { v = +v; if (v !== v || v <= 0) return 0; if (v >= 255) return 255; var f = Math.floor(v); if (v - f < 0.5) return f; if (v - f > 0.5) return f + 1; return f % 2 === 0 ? f : f + 1; }
var kinds = [['Int8Array', function (v) { return wrap(v, 8, true); }], ['Uint8Array', function (v) { return wrap(v, 8, false); }], ['Uint8ClampedArray', clamp8],
  ['Int16Array', function (v) { return wrap(v, 16, true); }], ['Uint16Array', function (v) { return wrap(v, 16, false); }], ['Int32Array', function (v) { return wrap(v, 32, true); }],
  ['Uint32Array', function (v) { return wrap(v, 32, false); }], ['Float32Array', function (v) { return Math.fround(v); }], ['Float64Array', function (v) { return +v; }]];
var specials = [0, -0, 1, -1, 0.5, 2.5, 127.5, 255.5, 1e10, -1e10, 4294967296.5, 2147483648, -2147483649, 1e300, -1e300, NaN, Infinity, -Infinity, 5e-324, 16777217, 3.4028236e38];
var all = (0, eval)('(function(){})') && 0;
for (var k = 0; k < kinds.length; k++) {
  var name = kinds[k][0], expect = kinds[k][1];
  var source = '(function(){ var V = new ' + name + '(16);' +
    'function store(a, b, n) { n = n | 0; var i = 0; for (i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) { V[i & 15] = a * b; V[(i + 1) & 15] = a + b; V[(i + 2) & 15] = a; } }' +
    'function read(i) { return V[i & 15]; }' +
    'return {store: store, read: read, view: function () { return V; }}; })()';
  var m = (0, eval)(source);
  for (var rep = 0; rep < 3; rep++) {
    for (var x = 0; x < specials.length; x++) for (var y = 0; y < specials.length; y += 4) {
      var a = specials[x], b = specials[y];
      m.store(a, b, 3);
      // the last three stores to cells (n-1..): i = 2 writes cells 2 (a*b), 3 (a+b), 4 (a)
      note(name + '.a*b(' + a + ',' + b + ')', m.read(2), expect(a * b));
      note(name + '.a+b', m.read(3), expect(a + b));
      note(name + '.a', m.read(4), expect(a));
    }
  }
  digest.push(Array.from(m.view()).join());
}
// A non-Number value (string, boolean, null, undefined, object with valueOf, BigInt) takes the generic store.
(function () {
  var V = new Int32Array(8), calls = 0, boxed = {valueOf: function () { calls++; return 42; }};
  function put(i, v) { i = i | 0; V[i & 7] = v; }
  function mixed(n) { n = n | 0; var i = 0; for (i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) { V[i & 7] = i + 1; } }
  for (var rep = 0; rep < 3; rep++) {
    mixed(8); note('mixed', Array.from(V).join(), '1,2,3,4,5,6,7,8');
    put(0, '17'); put(1, true); put(2, null); put(3, undefined); put(4, boxed); put(5, 2.9); put(6, -2.9); put(7, '0x10');
    note('coerced', Array.from(V).join(), '17,1,0,0,42,2,-2,16');
    var threw = false; try { put(0, 1n); } catch (e) { threw = e instanceof TypeError; } note('bigint', threw, true);
    var threw2 = false; try { put(0, Symbol()); } catch (e) { threw2 = e instanceof TypeError; } note('symbol', threw2, true);
  }
  note('valueOf.calls', calls, 3);
})();
// Name resolution that cannot take the fused path: with, an accessor on the global object, eval-introduced
// bindings, a shadowing parameter, a name rebound to a non-typed value in the middle of the loop.
(function () {
  var calls = 0, backing = new Uint8Array(8);
  Object.defineProperty(globalThis, 'ACC', {configurable: true, get: function () { calls++; return backing; }});
  function viaAccessor(i) { i = i | 0; ACC[i & 7] = i + 1; return ACC[i & 7] | 0; }
  for (var rep = 0; rep < 3; rep++) { calls = 0; var s = 0; for (var i = 0; i < 8; i++) s += viaAccessor(i); note('accessor', s + ':' + calls, '36:16'); }
  var other = new Uint8Array(8), V = new Uint8Array(8);
  function viaWith(i) { i = i | 0; with ({V: other}) { V[i & 7] = i + 20; } return V[i & 7] | 0; }
  for (var rep = 0; rep < 3; rep++) { for (var i = 0; i < 8; i++) viaWith(i); note('with', Array.from(other).join() + '|' + Array.from(V).join(), '20,21,22,23,24,25,26,27|0,0,0,0,0,0,0,0'); }
  function shadow(V, i) { i = i | 0; V[i & 7] = 99; return V[i & 7] | 0; }
  for (var rep = 0; rep < 3; rep++) note('param', shadow(other, 3) + ':' + other[3] + ':' + V[3], '99:99:0');
  var polymorphic = new Int32Array(8), plain = [1, 2, 3, 4, 5, 6, 7, 8], H = polymorphic;
  function poly(n) { n = n | 0; var i = 0, s = 0; for (i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) { s = (s + (H[i & 7] | 0)) | 0; H[i & 7] = i + 1; if ((i | 0) == 4) swap(); } return s; }
  function swap() { H = H === polymorphic ? plain : polymorphic; }
  for (var rep = 0; rep < 4; rep++) { H = polymorphic; for (var j = 0; j < 8; j++) polymorphic[j] = j + 10; for (var j = 0; j < 8; j++) plain[j] = j + 1; var r = poly(8); note('poly' + rep, r + ':' + Array.from(polymorphic).join() + '|' + plain.join(), r + ':' + Array.from(polymorphic).join() + '|' + plain.join()); digest.push(r, Array.from(polymorphic).join(), plain.join()); }
})();
var RESULT = bad.length + ':' + checked + ':' + bad.slice(0, 16).join(';') + '|' + digest.join('~');
"####;

const FUSED: &str = r####"
// Element access through free names (module-scope variables and globals) with pure keys.
var bad = [], checked = 0, digest = [];
function note(label, got, want) {
  checked++;
  if (!Object.is(got, want)) bad.push(label + ':' + String(got) + '!=' + String(want));
}
var G32 = new Int32Array(64), GU8 = new Uint8Array(64), GF64 = new Float64Array(16), GF32 = new Float32Array(32);
var GArr = [];                                  // plain array behind a global name
for (var i = 0; i < 64; i++) GArr[i] = i * 3;
var GObj = {5: 'five', 7: 70};                  // plain object with index keys
function makeModule(buffer) {
  var HEAP8 = new Int8Array(buffer), HEAPU8 = new Uint8Array(buffer), HEAP16 = new Int16Array(buffer),
      HEAPU16 = new Uint16Array(buffer), HEAP32 = new Int32Array(buffer), HEAPU32 = new Uint32Array(buffer),
      HEAPF32 = new Float32Array(buffer), HEAPF64 = new Float64Array(buffer), PLAIN = [10, 20, 30, 40, 50, 60, 70, 80];
  var swapped = false, tableLen = 0, other = new Int32Array(16);
  function fill(n) {
    n = n | 0;
    var i = 0;
    for (i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) {
      HEAP32[i & 15] = i * 3 | 0;                                // key and value are pure
      HEAPU8[64 + (i & 31) >> 0] = i + 200;
      HEAP16[(40 + (i & 7)) << 1 >> 1] = -i;
      HEAPF64[(i & 3) + 16 >> 0] = i * 0.5;                      // float key >> 0 ... see below
      HEAPF64[(i & 3) + 16] = i * 0.5;
      HEAPF32[(i & 7) + 48] = i / 3;
    }
    return 0;
  }
  function read(n) {
    n = n | 0;
    var i = 0, s = 0, t = 0.0;
    for (i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) {
      s = (s + (HEAP32[i & 15] | 0)) | 0;
      s = (s + (HEAPU8[64 + (i & 31) >> 0] | 0)) | 0;
      s = (s + (HEAP16[(40 + (i & 7)) << 1 >> 1] | 0)) | 0;
      s = (s + (HEAP8[i & 63] | 0) + (HEAPU16[i & 31] | 0) + (HEAPU32[i & 15] >>> 0)) | 0;
      t = t + +HEAPF64[(i & 3) + 16] + +HEAPF32[(i & 7) + 48];
      s = (s + (PLAIN[i & 7] | 0)) | 0;                          // an Array behind the same pattern
    }
    return s + ':' + t;
  }
  // constant keys
  function constants(n) {
    n = n | 0;
    var i = 0, s = 0;
    for (i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) {
      HEAP32[3] = i | 0; HEAP32[5] = (i * 7) | 0; HEAPU8[200] = i | 0;
      s = (s + (HEAP32[3] | 0) + (HEAP32[5] | 0) + (HEAPU8[200] | 0) + (HEAP32[1000] | 0) + (HEAP32[-1] | 0)) | 0;
    }
    return s;
  }
  // invalid keys read undefined / write nothing (never the prototype)
  function odd(n) {
    n = n | 0;
    var i = 0, s = 0, k = 0.0, r = 0;
    for (i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) {
      k = i * 0.5;
      r = HEAP32[k] === undefined ? 1 : 0;                       // fractional index for odd i
      s = (s + r) | 0;
      s = (s + (HEAP32[-i] === undefined ? 1000 : 0)) | 0;       // -0 reads element 0
      s = (s + (HEAP32[i / 0] === undefined ? 100000 : 0)) | 0;  // i/0: NaN or Infinity
      HEAP32[i - 1000] = 1; HEAP32[k] = 2; HEAP32[1 / 0] = 3;
    }
    return s;
  }
  // the name rebinds in the middle of a loop (a call between accesses)
  function rebind(n) {
    n = n | 0;
    var i = 0, s = 0;
    for (i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) {
      s = (s + (HEAP32[i & 7] | 0)) | 0;
      if ((i | 0) == 5) swap();
      HEAP32[i & 7] = i | 0;
    }
    return s;
  }
  function swap() { var t = HEAP32; HEAP32 = other; other = t; }
  function state() { var out = []; for (var i = 0; i < 8; i++) out.push(HEAP32[i]); out.push('|'); for (var i = 0; i < 8; i++) out.push(other[i]); return out.join(); }
  // detach / shrink while looping
  function reads(n) {
    n = n | 0;
    var i = 0, s = 0;
    for (i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) { s = (s + (HEAPU8[i & 63] === undefined ? 1000 : HEAPU8[i & 63])) | 0; }
    return s;
  }
  return {fill: fill, read: read, constants: constants, odd: odd, rebind: rebind, state: state, reads: reads,
    heap32: function () { return HEAP32; }};
}
for (var rep = 0; rep < 4; rep++) {
  var buffer = new ArrayBuffer(1 << 12), m = makeModule(buffer);
  var u8 = new Uint8Array(buffer);
  m.fill(40);
  var first = m.read(40);
  m.fill(40);
  note('read.repeat', m.read(40), first);
  note('c.read', typeof first, 'string');
  digest.push(first);
  var c0 = m.constants(10), c1 = m.constants(10);
  note('constants', c0, c1);
  note('constants.value', c0, (function () { var s = 0; for (var i = 0; i < 10; i++) { s = s + i + i * 7 + (i & 255) + 0 + 0; s = s | 0; } return s; })());
  note('odd', m.odd(20) + 0, m.odd(20) + 0);
  digest.push(c0, m.odd(20), m.rebind(12), m.state(), m.reads(70), m.read(40));
}
// Explicit semantics through free names.
(function () {
  var A = new Uint8Array(8), L = new Int32Array(8), S = 'str', N = 5, U = undefined, NUL = null, O = {1: 'x', length: 2};
  var P = new Proxy([1, 2, 3], {get: function (t, k) { return k === '1' ? 'proxied' : t[k]; }});
  var getterHits = 0, GET = {get 0() { getterHits++; return 42; }};
  var ARR = [1, 2, 3];
  function a(i) { i = i | 0; return A[i & 7] | 0; }
  function s(i) { i = i | 0; return S[i & 3]; }
  function n(i) { i = i | 0; return N[i & 3]; }
  function u(i) { i = i | 0; return U[i & 3]; }
  function nul(i) { i = i | 0; return NUL[i & 3]; }
  function o(i) { i = i | 0; return O[i & 3]; }
  function p(i) { i = i | 0; return P[i & 3]; }
  function g(i) { i = i | 0; return GET[i & 0]; }
  function arr(i) { i = i | 0; return ARR[i & 3]; }
  function seta(i, v) { i = i | 0; A[i & 7] = v; }
  function setarr(i, v) { i = i | 0; ARR[i & 3] = v; }
  for (var rep = 0; rep < 5; rep++) {
    seta(3, 300); seta(1, -1); seta(2, 2.7);
    note('typed', a(3) + ':' + a(1) + ':' + a(2), '44:255:2');
    note('string', s(1), 't'); note('number', n(1), undefined); note('object', o(1), 'x'); note('proxy', p(1), 'proxied');
    note('getter', g(5) + ':' + getterHits, '42:' + (rep + 1));
    var threw = '';
    try { u(1); } catch (e) { threw = e.constructor === TypeError ? 'TypeError' : String(e); }
    note('undefined.read', threw, 'TypeError');
    threw = '';
    try { nul(1); } catch (e) { threw = e.constructor === TypeError ? 'TypeError' : String(e); }
    note('null.read', threw, 'TypeError');
    setarr(1, 9); setarr(3, 8);
    note('array', arr(1) + ':' + ARR.length + ':' + arr(3), '9:4:8');
  }
  // ReferenceError and temporal dead zone order: the name is resolved before the key is evaluated
  var order = [];
  function key() { order.push('key'); return 1; }
  function dead(i) { i = i | 0; return TDZ[i & 3]; }
  var caught = '';
  try { dead(1); } catch (e) { caught = e.constructor.name; }
  let TDZ = [1, 2, 3, 4];
  note('tdz', caught, 'ReferenceError');
  note('tdz.after', dead(2), 3);
  function missing(i) { i = i | 0; return NOT_DEFINED_ANYWHERE[i & 3]; }
  caught = '';
  try { missing(1); } catch (e) { caught = e.constructor.name; }
  note('unresolvable', caught, 'ReferenceError');
  // coercing operands in the key: valueOf runs once, after the name is resolved
  var calls = [];
  var K = {valueOf: function () { calls.push('K'); return 2; }};
  function coerce(k) { return A[k | 0]; }
  for (var rep = 0; rep < 3; rep++) { seta(2, 77); note('valueOf', coerce(K), 77); }
  note('valueOf.count', calls.length, 3);
  // typed array reached through a global (property of the global object)
  (0, eval)('var GLOBALVIEW = new Float32Array(4); function gv(i) { i = i | 0; GLOBALVIEW[i & 3] = i * 1.5; return +GLOBALVIEW[i & 3]; }');
  for (var rep = 0; rep < 3; rep++) note('global', gv(3), 4.5);
  // detaching and shrinking between accesses
  var rab = new ArrayBuffer(8, {maxByteLength: 16}), RV = new Uint8Array(rab), FV = new Uint8Array(rab, 0, 8);
  function rd(i) { i = i | 0; var v = RV[i & 15]; return v === undefined ? -1 : v; }
  function rf(i) { i = i | 0; var v = FV[i & 7]; return v === undefined ? -1 : v; }
  function wr(i) { i = i | 0; RV[i & 15] = 7; FV[i & 7] = 9; }
  for (var rep = 0; rep < 3; rep++) {
    rab.resize(8); wr(2); note('rab.8', rd(2) + ':' + rf(2) + ':' + rd(12), '9:9:-1');
    rab.resize(16); note('rab.16', rd(12) + ':' + rf(2), '0:9');
    rab.resize(4); note('rab.4', rd(2) + ':' + rf(2) + ':' + rd(6), '9:-1:-1');
  }
  var buf = new ArrayBuffer(8), DV = new Int8Array(buf);
  function dv(i) { i = i | 0; DV[i & 7] = 5; var v = DV[i & 7]; return v === undefined ? -1 : v; }
  for (var rep = 0; rep < 3; rep++) {
    var ok = dv(1);
    if (rep === 1) buf.transfer();
    note('detach' + rep, dv(1), rep >= 1 ? -1 : 5);
  }
})();
var RESULT = bad.length + ':' + checked + ':' + bad.slice(0, 20).join(';') + '|' + digest.join('~');
"####;

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
