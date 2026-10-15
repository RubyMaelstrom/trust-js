//! Direct-call frame scans (ECMA-262 FunctionDeclarationInstantiation: every local starts
//! undefined; completion drops each owner once), function-table method reads
//! (EvaluatePropertyAccessWithExpressionKey / GetValue on dense array elements) and the
//! closure-variable (Declarative Environment Record) load and store fast paths must agree
//! across all three tiers. Expected values are V8's.

#![cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]

use crate::{bytecode::Tier, Completion, Engine};

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn all_tiers(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(evaluate(&mut engine, source), expected, "{tier:?}");
    }
}

#[test]
fn direct_call_frames_start_undefined_and_release_owners() {
    all_tiers(
        r####"// Direct-call frames: every local starts undefined, every owner is released at exit.
var token = { id: 1 };
var out = [];
function build(n) {
  // function with n locals; locals are filled with a mix of numbers, negatives, strings, objects
  var decls = [], body = [];
  for (var i = 0; i < n; i++) decls.push("v" + i);
  var src = "return function f" + n + "(a, b) {";
  if (n) src += "var " + decls.join(",") + ";";
  src += "var undef = 0; ";
  for (var i = 0; i < n; i++) src += "if (typeof v" + i + " !== 'undefined') undef++; ";
  for (var i = 0; i < n; i++) {
    switch (i % 7) {
      case 0: src += "v" + i + " = " + (i + 1) + "; "; break;
      case 1: src += "v" + i + " = -" + (i + 0.5) + "; "; break;
      case 2: src += "v" + i + " = a; "; break;
      case 3: src += "v" + i + " = 'str' + b; "; break;
      case 4: src += "v" + i + " = {k: " + i + "}; "; break;
      case 5: src += "v" + i + " = a === a; "; break;
      case 6: src += "v" + i + " = null; "; break;
    }
  }
  src += "return undef + (typeof a);}";
  return new Function(src)();
}
for (var n = 0; n <= 37; n++) {
  var f = build(n);
  var driver = new Function("f", "token", "var s = ''; for (var i = 0; i < 40; i++) { s = f(token, i); } return s;");
  out.push(n + ":" + driver(f, token));
}
// surplus and missing arguments, with owners among the surplus
function g(a, b) { var x = a; var y = b; return (typeof x) + (typeof y); }
var d2 = function () { var s = ''; for (var i = 0; i < 40; i++) { s = g(token, token, token, 'x', {}) + g() + g(1) + g(token); } return s; };
out.push(d2());
// recursion with locals holding objects
function rec(n, o) { var p = o, q = { n: n }, r = 'x' + n, t = -n; if (n === 0) return q.n + t; return rec(n - 1, p) + q.n; }
out.push(rec(200, token));
// exceptions thrown through frames full of owners
function thrower(n, o) { var a = o, b = { n: n }, c = 'c' + n, d = -n; if (n === 0) throw new Error('boom'); return thrower(n - 1, o); }
var caught = 0;
for (var i = 0; i < 40; i++) { try { thrower(8, token); } catch (e) { caught += e.message === 'boom' ? 1 : 0; } }
out.push(caught);
out.join('|');
"####,
        "0:0object|1:0object|2:0object|3:0object|4:0object|5:0object|6:0object|7:0object|8:0object|9:0object|10:0object|11:0object|12:0object|13:0object|14:0object|15:0object|16:0object|17:0object|18:0object|19:0object|20:0object|21:0object|22:0object|23:0object|24:0object|25:0object|26:0object|27:0object|28:0object|29:0object|30:0object|31:0object|32:0object|33:0object|34:0object|35:0object|36:0object|37:0object|objectobjectundefinedundefinednumberundefinedobjectundefined|20100|40",
    );
}

#[test]
fn function_table_method_reads_follow_element_semantics() {
    all_tiers(
        r####"var out = [];
function f0() { return 'f0:' + (this === T); }
function f1(a) { return 'f1:' + a; }
function f2(a, b) { return 'f2:' + a + b; }
var T = [f0, f1, f2, f0];
function callT(i) { return T[i & 3](i, 'x'); }
var s = '';
for (var i = 0; i < 40; i++) s += callT(i) + ',';
out.push(s.length + ':' + s.slice(0, 60));
// receiver is the array (this), arguments evaluated after the method read
var order = [];
var U = [function () { order.push('call:' + (this === U)); return 1; }];
function idx() { order.push('idx'); return 0; }
function arg() { order.push('arg'); return 5; }
for (var i = 0; i < 20; i++) U[idx()](arg());
out.push(order.slice(0, 6).join(',') + ',' + order.length);
// holes, prototype-supplied elements, accessors, absent
var P = []; P[3] = function () { return 'p3'; };
Array.prototype[1] = function () { return 'proto1'; };
var r = [];
for (var i = 0; i < 20; i++) { r.push(P[3]()); r.push(P[1]()); }
out.push(r.slice(0, 4).join(','));
delete Array.prototype[1];
var res = [];
for (var i = 0; i < 20; i++) { try { P[1](); } catch (e) { res.push(e.constructor.name); } }
out.push(res[0] + res.length);
var A = []; Object.defineProperty(A, 0, { get: function () { return function () { return 'getter' + (this === A); }; } });
var g = [];
for (var i = 0; i < 20; i++) g.push(A[0]());
out.push(g[0] + g.length);
// non-callable, nullish, primitive elements
var N = [1, null, undefined, 'x', {}];
var errs = [];
for (var k = 0; k < 5; k++) for (var i = 0; i < 12; i++) { try { N[k](); } catch (e) { if (i === 0) errs.push(e.constructor.name); } }
out.push(errs.join(','));
// keys: fractional, negative, huge, NaN, -0, strings, symbols
var K = [function () { return 'k0'; }, function () { return 'k1'; }];
K[-1] = function () { return 'neg'; }; K[1.5] = function () { return 'frac'; };
K['2'] = function () { return 'k2'; }; K[4294967295] = function () { return 'big'; };
var ks = [];
for (var i = 0; i < 12; i++) ks.push([K[0](), K[1](), K[-1](), K[1.5](), K['2'](), K[2](), K[-0](), K[4294967295]()].join(''));
out.push(ks[11]);
try { for (var i = 0; i < 12; i++) K[NaN](); } catch (e) { out.push(e.constructor.name); }
// reassignment mid-loop, length change, growth
var R = [function () { return 1; }, function () { return 2; }];
var sum = 0;
for (var i = 0; i < 40; i++) { sum += R[(i & 1) && R.length > 1 ? 1 : 0](); if (i === 20) R[1] = function () { return 20; }; if (i === 30) R.push(function () { return 5; }); if (i === 35) R.length = 1; }
out.push(sum);
// proxy, typed array, arguments, string receivers
var px = new Proxy([function () { return 'px'; }], {});
var ta = new Uint8Array(2);
var sres = [];
for (var i = 0; i < 12; i++) { sres.push(px[0]()); try { ta[0](); } catch (e) { sres.push(e.constructor.name); } }
out.push(sres.slice(0, 4).join(',') + sres.length);
function viaArgs() { return arguments[0](); }
var ar = [];
for (var i = 0; i < 12; i++) ar.push(viaArgs(function () { return 'arg' + i; }));
out.push(ar[11]);
var strs = [];
for (var i = 0; i < 12; i++) { try { 'abc'[0](); } catch (e) { strs.push(e.constructor.name); } }
out.push(strs[0] + strs.length);
// frozen arrays and array with non-writable / non-configurable elements
var Z = Object.freeze([function () { return 'z'; }]);
var zs = []; for (var i = 0; i < 12; i++) zs.push(Z[0]());
out.push(zs[11]);
// function table in closure with local index arithmetic (asm.js shape)
var m = (function (g, e, b) { "use asm"; var H = new g.Int32Array(b); var TT = [function (x) { x = x | 0; return x + 1 | 0; }, function (x) { x = x | 0; return x + 2 | 0; }, function (x) { x = x | 0; return x + 3 | 0; }, function (x) { x = x | 0; return x + 4 | 0; }];
  function run(n) { n = n | 0; var i = 0, s = 0; for (i = 0; (i | 0) < (n | 0); i = i + 1 | 0) s = s + (TT[i & 3](i) | 0) | 0; return s | 0; }
  return { run: run }; })(globalThis, {}, new ArrayBuffer(1 << 16));
out.push(m.run(1000));
out.join('|');
"####,
        "285:f0:true,f1:1,f2:2x,f0:true,f0:true,f1:5,f2:6x,f0:true,f0:tru|idx,arg,call:true,idx,arg,call:true,60|p3,proto1,p3,proto1|TypeError20|gettertrue20|TypeError,TypeError,TypeError,TypeError,TypeError|k0k1negfrack2k2k0big|TypeError|202|px,TypeError,px,TypeError24|arg11|TypeError12|z|502000",
    );
}

#[test]
fn closure_variables_load_and_store_every_value_kind() {
    all_tiers(
        r####"var out = [];
// closure variables of every kind through reads and writes
var mod = (function () {
  var num = 1, neg = -0, nan = NaN, str = 'a', obj = { v: 1 }, bool = true, un, nul = null, big = 10n, sym = Symbol('s');
  var inf = Infinity, i32 = 7;
  function read() { return [num, neg, nan, str, obj.v, bool, un, nul, big, typeof sym, inf, i32]; }
  function bump() { num = num + 1; i32 = i32 | 0; i32 = (i32 + 1) | 0; }
  function swap(k) {
    switch (k & 7) {
      case 0: num = 'now string'; break;
      case 1: num = 5; break;
      case 2: num = { v: 2 }; break;
      case 3: num = 6n; break;
      case 4: num = null; break;
      case 5: num = undefined; break;
      case 6: num = NaN; break;
      case 7: num = -0; break;
    }
    return num;
  }
  function loop(n) { var t = 0; for (var i = 0; i < n; i++) { bump(); t += i32 & 1; } return t; }
  return { read: read, bump: bump, swap: swap, loop: loop, get num() { return num; } };
})();
var r = [];
for (var i = 0; i < 30; i++) { r.push(String(mod.swap(i))); }
out.push(r.slice(0, 10).join(','));
out.push(JSON.stringify(mod.read().map(function (x) { return typeof x === 'bigint' ? x + 'n' : x; })).slice(0, 80));
out.push(mod.loop(100) + ':' + mod.num);
// stores of Number over each kind and back
var cell = 0;
function setCell(v) { cell = v; return cell; }
var seq = [1, 2.5, -3, 'x', 4, {}, 5, true, 6, null, 7, undefined, 8, 9n, 10, NaN, -0, Infinity, 11];
var o2 = [];
for (var k = 0; k < 3; k++) for (var i = 0; i < seq.length; i++) o2.push(typeof setCell(seq[i]) + (Object.is(cell, -0) ? 'm0' : ''));
out.push(o2.slice(0, 20).join(',') + o2.length);
// const and TDZ
function tdz() { try { return later; } catch (e) { return e.constructor.name; } }
var t1 = []; for (var i = 0; i < 20; i++) t1.push(tdz());
let later = 5;
var t2 = []; for (var i = 0; i < 20; i++) t2.push(tdz());
out.push(t1[19] + t2[19]);
const K1 = 3;
function setConst() { try { K1 = 4; } catch (e) { return e.constructor.name; } return K1; }
var t3 = []; for (var i = 0; i < 20; i++) t3.push(setConst());
out.push(t3[19]);
// reassigned from another closure mid-loop
var shared = 10;
function useShared() { return shared + 1; }
function reassign() { shared = 'now'; }
var t4 = []; for (var i = 0; i < 30; i++) { if (i === 15) reassign(); t4.push(useShared()); }
out.push(t4[0] + ',' + t4[14] + ',' + t4[15] + ',' + t4[29]);
// global object accessors and properties
Object.defineProperty(globalThis, 'accessorName', { get: function () { return 42; }, set: function (v) { out.push('set' + v); }, configurable: true });
function viaGlobal() { accessorName = 1; return accessorName; }
var t5 = []; for (var i = 0; i < 12; i++) t5.push(viaGlobal());
out.push(t5[11]);
// with and eval keep names resolvable
var wv = 'outer';
function withTest(o) { with (o) { return wv; } }
var t6 = []; for (var i = 0; i < 12; i++) t6.push(withTest(i & 1 ? { wv: 'inner' } : {}));
out.push(t6.slice(0, 4).join(','));
function evalTest() { var v = 1; eval('var added = 5'); return v + added; }
var t7 = []; for (var i = 0; i < 12; i++) t7.push(evalTest());
out.push(t7[11]);
// delete and re-add through eval in sloppy mode changes resolution
var dyn = 'global-dyn';
function dynRead() { return dyn; }
function shadow() { eval('var dyn = "local"'); return dyn + '/' + dynRead(); }
var t8 = []; for (var i = 0; i < 12; i++) t8.push(shadow());
out.push(t8[11]);
// function calls through closure variables holding different callees
var fn = function () { return 'A'; };
function callFn() { return fn(); }
var t9 = []; for (var i = 0; i < 20; i++) { if (i === 10) fn = function () { return 'B'; }; t9.push(callFn()); }
out.push(t9[9] + t9[10]);
// Math builtins through closure variables
var imul = Math.imul, floor = Math.floor, sqrt = Math.sqrt;
function mathCalls(a, b) { return imul(a, b) + floor(a / 3) + sqrt(a * a); }
var t10 = []; for (var i = 0; i < 20; i++) t10.push(mathCalls(i + 1.5, 65537 + i));
out.push(t10[19] + ',' + t10[0]);
imul = function () { return 'replaced'; };
out.push(mathCalls(3, 4));
// getters on the scope's object via Proxy global lookups are not involved; symbol/BigInt values
var big = 2n ** 64n; var symv = Symbol.iterator;
function readBig() { return big + 1n; }
var t11 = []; for (var i = 0; i < 12; i++) t11.push(readBig());
out.push(String(t11[11]) + (symv === Symbol.iterator));
out.join('|');
"####,
        "now string,5,[object Object],6,null,undefined,NaN,0,now string,5|[null,0,null,\"a\",1,true,null,null,\"10n\",\"symbol\",null,7]|50:NaN|number,number,number,string,number,object,number,boolean,number,object,number,undefined,number,bigint,number,number,numberm0,number,number,number57|ReferenceError5|TypeError|11,11,now1,now1|set1|set1|set1|set1|set1|set1|set1|set1|set1|set1|set1|set1|42|outer,inner,outer,inner|6|local/global-dyn|AB|1311146.5,65538.5|replaced13|18446744073709551617true",
    );
}

/// ECMA-262 §14.7.2 DoWhileStatement: a literal false test runs the body once; `continue`,
/// labelled `break`, `finally`, closures and generators behave as in any other loop.
#[test]
fn do_while_with_a_literal_false_test_runs_the_body_once() {
    all_tiers(
        r####"var out = [];
function a1(n) { var c = 0; do { c++; if (n > 2) continue; c += 10; } while (0); return c; }
function a2(n) { var c = 0; l: do { c++; if (n) break l; c += 100; } while (false); return c; }
function a3() { var c = 0; do { c++; do { c += 2; if (c > 0) continue; c += 1000; } while (null); } while (-0); return c; }
function a4() { var i = 0; do { i++; } while (i < 3); return i; }
function a5(n) { var c = 0, f; do { f = function () { return n + c; }; c = 5; } while (NaN); return f(); }
function a6() { var r = []; for (var i = 0; i < 3; i++) { do { r.push(i); if (i === 1) break; r.push('x'); } while (0); } return r.join(','); }
function a7() { var c = 0; do { try { c++; continue; } finally { c += 10; } } while (0); return c; }
function a8() { var c = 0; do { c++; { let z = c; do { z++; c += z; } while (0); } } while (0); return c; }
function a9(n) { var t = 0; do { switch (n) { case 1: t = 1; break; default: t = 2; } t *= 3; } while (0); return t; }
function a10() { var c = 0; do { c++; } while (0 + 0 && c++); return c; }
function a11() { var c = 0, side = 0; do { c++; } while ((side++, 0)); return c + ':' + side; }
function a12() { var c = 0; do { c++; if (c < 3) continue; } while (c < 3); return c; }
var r = [];
for (var i = 0; i < 40; i++) r = [a1(i & 7), a2(i & 1), a3(), a4(), a5(i), a6(), a7(), a8(), a9(i & 3), a10(), a11(), a12()];
out.push(r.join(','));
function labeled() { var s = ''; outer: for (var i = 0; i < 3; i++) { inner: do { s += i; if (i === 1) continue outer; s += 'a'; } while (0); s += 'b'; } return s; }
var l = []; for (var i = 0; i < 20; i++) l.push(labeled());
out.push(l[19]);
var gen = (function* () { do { yield 1; yield 2; } while (0); return 3; })();
out.push([gen.next().value, gen.next().value, gen.next().value, gen.next().done].join(','));
out.join('|');
"####,
        "1,1,3,3,44,0,x,1,2,x,11,3,6,1,1:1,3|0ab12ab|1,2,3,true",
    );
}

/// ECMA-262 §13.5.5 unary minus: literals are constants (`-0` is negative zero) and a Number
/// operand negates exactly in the native template; BigInt, strings, objects and Symbols
/// keep the checked ToNumeric semantics.
#[test]
fn unary_minus_negates_numbers_natively_and_coerces_everything_else() {
    all_tiers(
        r####"var out = [];
function neg(x) { return -x; }
function negPair(x, y) { var a = -x; var b = -y; return Object.is(a, -0) + ',' + a + ',' + b; }
function lit() { return [-0, -5, -1.5, -(2), - -3, -NaN, -Infinity, -1e300, -0x10, -2147483648, -4294967296].map(function (v) { return Object.is(v, -0) ? '-0' : String(v); }).join(';'); }
var vals = [0, -0, 1, -1, 2.5, NaN, Infinity, -Infinity, '5', '', 'abc', true, false, null, undefined, 7n, -7n, { valueOf: function () { return 3; } }, [4], [], { valueOf: function () { return 5n; } }];
var res = [];
for (var round = 0; round < 12; round++) {
  res = [];
  for (var i = 0; i < vals.length; i++) {
    var v;
    try { v = neg(vals[i]); res.push(Object.is(v, -0) ? '-0' : typeof v === 'bigint' ? v + 'n' : String(v)); } catch (e) { res.push(e.constructor.name); }
  }
}
out.push(res.join(','));
var l = []; for (var i = 0; i < 20; i++) l.push(lit()); out.push(l[19]);
var p = []; for (var i = 0; i < 20; i++) p.push(negPair(i - 3, 0)); out.push(p[0] + '|' + p[3] + '|' + p[19]);
try { for (var i = 0; i < 12; i++) neg(Symbol('s')); } catch (e) { out.push(e.constructor.name); }
function loopNeg(n) { var s = 0; for (var i = 0; i < n; i++) { s = -s + i; } return s; }
out.push(loopNeg(1000));
function negStr(s) { return -s; }
var ns = []; for (var i = 0; i < 12; i++) ns.push(negStr('12') + negStr(' 0x10 ') + negStr('1e3'));
out.push(ns[11]);
out.join('|');
"####,
        "-0,0,-1,1,-2.5,NaN,-Infinity,Infinity,-5,-0,NaN,-1,-0,-0,NaN,-7n,7n,-3,-4,-0,-5n|-0;-5;-1.5;-2;3;NaN;-Infinity;-1e+300;-16;-2147483648;-4294967296|false,3,0|true,0,0|false,-16,0|TypeError|500|-1028",
    );
}

/// A hot loop body of hundreds of operations reads closure variables of every kind in line; the
/// values, accumulations and a replaced callee agree with the interpreter.
#[test]
fn large_loop_bodies_read_closure_variables_of_every_kind() {
    all_tiers(
        r####"var out = [];
var n1 = 1.5, n2 = -2, s1 = 'a', o1 = { v: 3 }, b1 = true, u1, nl1 = null, big1 = 5n, sy1 = Symbol('q');
function fn1(x) { return x * n2; }
function run(rounds) {
  var acc = '';
  for (var i = 0; i < rounds; i++) {
    acc += n1 + n2 + 0; acc += s1; acc += o1.v + 0; acc += b1; acc += u1; acc += nl1; acc += big1; acc += typeof sy1; acc += (n1 > n2 ? 'g' : 'l'); acc += fn1(0);
    acc += n1 + n2 + 1; acc += s1; acc += o1.v + 1; acc += b1; acc += u1; acc += nl1; acc += big1; acc += typeof sy1; acc += (n1 > n2 ? 'g' : 'l'); acc += fn1(1);
    acc += n1 + n2 + 2; acc += s1; acc += o1.v + 2; acc += b1; acc += u1; acc += nl1; acc += big1; acc += typeof sy1; acc += (n1 > n2 ? 'g' : 'l'); acc += fn1(2);
    acc += n1 + n2 + 3; acc += s1; acc += o1.v + 3; acc += b1; acc += u1; acc += nl1; acc += big1; acc += typeof sy1; acc += (n1 > n2 ? 'g' : 'l'); acc += fn1(3);
    acc += n1 + n2 + 4; acc += s1; acc += o1.v + 4; acc += b1; acc += u1; acc += nl1; acc += big1; acc += typeof sy1; acc += (n1 > n2 ? 'g' : 'l'); acc += fn1(4);
    acc += n1 + n2 + 5; acc += s1; acc += o1.v + 5; acc += b1; acc += u1; acc += nl1; acc += big1; acc += typeof sy1; acc += (n1 > n2 ? 'g' : 'l'); acc += fn1(5);
    acc += n1 + n2 + 6; acc += s1; acc += o1.v + 6; acc += b1; acc += u1; acc += nl1; acc += big1; acc += typeof sy1; acc += (n1 > n2 ? 'g' : 'l'); acc += fn1(6);
    acc += n1 + n2 + 7; acc += s1; acc += o1.v + 7; acc += b1; acc += u1; acc += nl1; acc += big1; acc += typeof sy1; acc += (n1 > n2 ? 'g' : 'l'); acc += fn1(7);
    n1 = n1 + 1; s1 = s1 + i; n2 = i & 1 ? n2 : -n2;
    if (i === 10) { b1 = false; u1 = 7; o1 = { v: 9 }; big1 = 6n; }
    if (i === 20) { fn1 = function (x) { return x + 100; }; }
  }
  return acc.length + ':' + acc.slice(0, 50) + ':' + acc.slice(-50);
}
for (var r = 0; r < 4; r++) out.push(run(30));
out.join('|');
"####,
        "12001:-0.5a3trueundefinednull5symbolg00.5a4trueundefined:51617181920212223242526272816false7null6symbolg107|23690:33.5a012345678910111213141516171819202122232425262:51617181920212223242526272816false7null6symbolg107|35690:59.5a012345678910111213141516171819202122232425262:51617181920212223242526272816false7null6symbolg107|47886:93.5a012345678910111213141516171819202122232425262:51617181920212223242526272816false7null6symbolg107",
    );
}

#[cfg(target_arch = "aarch64")]
fn owners(engine: &mut Engine, name: &str) -> usize {
    use crate::value::Value;
    let env = engine.interp.global_env.clone();
    match engine.interp.get_var(name, &env).ok().expect("binding") {
        Value::Obj(object) => crate::value::Gc::strong_count(&object) - 1,
        Value::Str(text) => text.strong_count() - 1,
        _ => panic!("{name} is not an owner"),
    }
}

#[cfg(target_arch = "aarch64")]
fn settle(engine: &mut Engine, name: &str) {
    let env = engine.interp.global_env.clone();
    let function = engine.interp.get_var(name, &env).ok().expect("function");
    let object = function.as_obj().expect("object").borrow();
    let crate::value::Callable::User(user) = &object.call else {
        panic!("{name} is not a user function")
    };
    let chunk = user
        .func
        .code
        .get()
        .and_then(Option::as_ref)
        .expect("chunk");
    assert!(chunk.jit.get().flatten().is_some(), "{name} runs natively");
    chunk.inline_attempted.set(true);
    chunk.inline_retry_at.set(0);
}

/// The shared finish stub scans every callee slot: a slot count of any residue, with an owner
/// at any position (first or second word of a pair, the odd tail, beside negative Numbers
/// that take the slow pair path), releases that owner exactly once.
#[cfg(target_arch = "aarch64")]
#[test]
fn direct_calls_release_slot_owners_exactly_once() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let mut source = String::from(
        "var token={}, text='abc'+String(Math.random()), big=12345678901234567890n, sym=Symbol('s');\n",
    );
    let mut names = Vec::new();
    for n in 1..=13usize {
        for position in [0, n / 2, n - 1] {
            let name = format!("f{n}_{position}");
            let driver = format!("d{n}_{position}");
            let mut locals = String::new();
            let mut body = String::new();
            for k in 0..n {
                locals.push_str(&format!("l{k},"));
                let value = if k == position {
                    "a".to_string()
                } else {
                    ["-1.5", "2", "b", "big", "sym", "null", "true", "-7", "a"][(k * 5 + n) % 9]
                        .to_string()
                };
                body.push_str(&format!("l{k}={value};"));
            }
            source.push_str(&format!(
                "function {name}(a,b){{var {locals}z=0;{body}return z;}}\n\
                 function {driver}(n){{var s=0;for(var i=0;i<n;i++)s+={name}(token,text);return s;}}\n"
            ));
            names.push((name, driver));
        }
    }
    evaluate(&mut engine, &source);
    for (_, driver) in &names {
        evaluate(&mut engine, &format!("{driver}(2);"));
    }
    for (name, driver) in &names {
        settle(&mut engine, name);
        settle(&mut engine, driver);
    }
    let token = owners(&mut engine, "token");
    let text = owners(&mut engine, "text");
    // Another test may exhaust the process-wide call-cache epoch while this one runs, which
    // sends calls through the checked helper for a moment; retry until the whole batch has
    // gone through the direct sequence. Ownership must be exact after every round.
    let mut direct = false;
    for _ in 0..200 {
        super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
        for (_, driver) in &names {
            assert_eq!(evaluate(&mut engine, &format!("{driver}(40);")), "0");
        }
        assert_eq!(owners(&mut engine, "token"), token);
        assert_eq!(owners(&mut engine, "text"), text);
        if super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.get()) >= names.len() * 40 {
            direct = true;
            break;
        }
    }
    assert!(
        direct,
        "the callees must run through the direct-call sequence"
    );
}
