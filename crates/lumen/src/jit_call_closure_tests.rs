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

#[cfg(target_arch = "aarch64")]
fn owners(engine: &mut Engine, name: &str) -> usize {
    use crate::value::Value;
    let env = engine.interp.global_env.clone();
    match engine.interp.get_var(name, &env).ok().expect("binding") {
        Value::Obj(object) => std::rc::Rc::strong_count(&object) - 1,
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
