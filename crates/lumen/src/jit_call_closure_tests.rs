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
