//! TypeError messages that name the call or `new` target (see `crate::callee_name`).
//!
//! ECMA-262 Call step 2 and EvaluateNew step 7 require a TypeError; its message is
//! implementation-defined, and every tier must give the same one. Expected messages are V8's
//! (Node v24) wherever V8 names the target.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

const TIERS: [Tier; 3] = [Tier::Interp, Tier::Bytecode, Tier::Jit];

/// Evaluate `source` on every tier, both compiling functions on their first call and with the
/// default thresholds, and require the same completion value.
fn check(source: &str, expected: &str) {
    for tier in TIERS {
        for threshold in [Some(0), None] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            if let Some(threshold) = threshold {
                engine.set_tier_threshold(threshold);
            }
            match engine.eval(source, false) {
                Ok(Completion::Value(value)) => {
                    assert_eq!(value, expected, "{tier:?} threshold {threshold:?}")
                }
                Ok(Completion::Throw { name, message }) => {
                    panic!("{tier:?} threshold {threshold:?}: {name}: {message}")
                }
                Err(error) => panic!("parse error: {}", error.message),
            }
        }
    }
}

/// A harness that reports `Type: message` for each probe, run repeatedly so the probes' own
/// call sites warm up and compile.
const HARNESS: &str = r#"
function report(f) {
    let last;
    for (let round = 0; round < 40; round++) {
        try { f(); last = "no error"; }
        catch (e) { last = (e instanceof TypeError ? "TypeError" : e.name) + ": " + e.message; }
    }
    return last;
}
"#;

#[test]
fn calls_name_their_target_expression() {
    check(
        &format!(
            r#"{HARNESS}
            var o = {{ u: undefined, n: 5, s: "x", inner: {{}}, m() {{ return {{}}; }} }};
            var a = [], i = 0, key = "k", undefinedFn;
            [
                report(() => o.u()),
                report(() => o.n(1, 2)),
                report(() => undefinedFn()),
                report(() => o.m().then()),
                report(() => o.m(1, 2).then(3)),
                report(() => a[0]()),
                report(() => a[i]()),
                report(() => o[key]()),
                report(() => o["u"]()),
                report(() => o[Symbol.iterator]()),
                report(() => o.inner.deep()),
                report(() => (o.u)()),
                report(function () {{ return this.missing(); }}.bind(o)),
                report(() => o?.u()),
                report(() => o.n?.()),
                report(() => o.missing?.()),
                report(() => o.u(...[1, 2])),
                report(() => o.u`template`),
                report(() => (0, o.u)()),
                report(() => o.s.trim.missing()),
            ].join("\n")"#
        ),
        "TypeError: o.u is not a function\n\
         TypeError: o.n is not a function\n\
         TypeError: undefinedFn is not a function\n\
         TypeError: o.m(...).then is not a function\n\
         TypeError: o.m(...).then is not a function\n\
         TypeError: a[0] is not a function\n\
         TypeError: a[i] is not a function\n\
         TypeError: o[key] is not a function\n\
         TypeError: o.u is not a function\n\
         TypeError: o[Symbol.iterator] is not a function\n\
         TypeError: o.inner.deep is not a function\n\
         TypeError: o.u is not a function\n\
         TypeError: this.missing is not a function\n\
         TypeError: o?.u is not a function\n\
         TypeError: o.n is not a function\n\
         no error\n\
         TypeError: o.u is not a function\n\
         TypeError: o.u is not a function\n\
         TypeError: (intermediate value) is not a function\n\
         TypeError: o.s.trim.missing is not a function",
    );
}

#[test]
fn constructions_name_their_target_expression() {
    check(
        &format!(
            r#"{HARNESS}
            var o = {{ m() {{}}, arrow: () => 1, u: undefined, gen: function* () {{}} }};
            var bound = o.arrow.bind(null), C;
            [
                report(() => new o.m()),
                report(() => new o.arrow()),
                report(() => new o.u()),
                report(() => new C()),
                report(() => new C),
                report(() => new bound(1)),
                report(() => new o.gen()),
                report(() => new o.u(...[1])),
                report(() => new Math.max()),
                report(() => new (function () {{ return o; }})().u()),
            ].join("\n")"#
        ),
        "TypeError: o.m is not a constructor\n\
         TypeError: o.arrow is not a constructor\n\
         TypeError: o.u is not a constructor\n\
         TypeError: C is not a constructor\n\
         TypeError: C is not a constructor\n\
         TypeError: bound is not a constructor\n\
         TypeError: o.gen is not a constructor\n\
         TypeError: o.u is not a constructor\n\
         TypeError: Math.max is not a constructor\n\
         TypeError: (intermediate value).u is not a function",
    );
}

/// Only the target of the call that failed is named: an error thrown inside a callable target,
/// including by a built-in that calls a non-callable argument, keeps its own message.
#[test]
fn errors_from_inside_callable_targets_keep_their_messages() {
    check(
        &format!(
            r#"{HARNESS}
            var o = {{ f(x) {{ return x(); }}, g() {{ return o.f(undefined); }} }};
            var forward = Function.prototype.call;
            [
                report(() => o.f(5)),
                report(() => o.g()),
                report(() => forward.call(undefined)),
                report(() => [1].map(undefined)),
                report(() => {{ throw new TypeError("undefined is not a function"); }}),
            ].join("\n")"#
        ),
        "TypeError: x is not a function\n\
         TypeError: x is not a function\n\
         TypeError: undefined is not a function\n\
         TypeError: Array.prototype.map callback is not callable\n\
         TypeError: undefined is not a function",
    );
}

/// Warmed call sites whose cached target is replaced, inlined callers whose guard misses, and
/// proper tail calls still name the target.
#[test]
fn warmed_inlined_and_tail_call_sites_name_their_target() {
    check(
        r#"
        "use strict";
        var holder = { run: function (x) { return x + 1; } };
        function callRun(x) { const result = holder.run(x); return result; }
        function invoke(fn, x) { const result = fn(x); return result; }
        function tail(fn) { return fn(); }
        function make(K) { return new K(); }
        function K() { this.k = 1; }
        var total = 0;
        for (var n = 0; n < 3000; n++) {
            total += callRun(n) + invoke(holder.run, n) + make(K).k;
        }
        var out = [total > 0];
        holder.run = 7;
        try { callRun(1); } catch (e) { out.push(e.message); }
        try { invoke(undefined, 1); } catch (e) { out.push(e.message); }
        try { tail(null); } catch (e) { out.push(e.message); }
        try { make(holder.run); } catch (e) { out.push(e.message); }
        try { make(() => 1); } catch (e) { out.push(e.message); }
        out.join("|")
        "#,
        "true|holder.run is not a function|fn is not a function|fn is not a function\
         |K is not a constructor|K is not a constructor",
    );
}

/// A shadowed `eval` binding is an ordinary call target.
#[test]
fn a_non_callable_eval_binding_is_named() {
    check(
        r#"
        (function () {
            var eval = 5, out = [];
            for (var n = 0; n < 3; n++) {
                try { eval("1"); } catch (e) { out.push(e.constructor === TypeError, e.message); }
            }
            return out.join("|");
        })()
        "#,
        "true|eval is not a function|true|eval is not a function|true|eval is not a function",
    );
}
