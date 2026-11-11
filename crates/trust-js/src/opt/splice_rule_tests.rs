//! The exemplars' splice rule (P3 stage 2, steps 15 and 15b): with the optimizing tier on, the
//! inline planner splices no call site whose ways hold more than one distinct callee unless its
//! call is a named method call (Maglev's polymorphic method calls: `o.m(...)`, whose callee
//! `Op::GetMethod` reads, which the bytecode compiler marks), and none that met another closure
//! of a way's function, which the site records once, at its first code-keyed hit
//! (`CallSite::code_hint`); with the tier off it keeps P2's policy. Each record path in turn: the
//! machine code's full probe (then the hinted-way test and the call-hit helper), the Rust
//! code-sharing retry; a site of one closure splices as before; React's `map` callback; and a
//! late second closure, which the reoptimization meets with the guard's mismatch edge, not a new
//! plan. Then step 15b's: the shapes the compiler marks; a named method site over two classes
//! spliced, at depth 0 and inside a spliced callee; keyed, free-name and plain calls over two
//! callees not spliced, side by side with a named one; a named method site that met another
//! closure not spliced.

use super::ir::ExitReason;
use super::TestSwitches;
use crate::bytecode::{plan_inlines, Chunk, Op, Tier};
use crate::value::Value;
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

/// The tier off on this thread, compiling on the main thread: runs that fill call sites with no
/// tier compile among them.
fn tier_off() -> TestSwitches {
    TestSwitches::default()
}

/// The tier on and eager, compiling on the main thread, nothing forced.
fn tier_on() -> TestSwitches {
    TestSwitches {
        on: true,
        eager: true,
        ..Default::default()
    }
}

/// An engine at the JIT tier that compiles every function at its first call, `switches` set.
fn engine(switches: TestSwitches) -> Engine {
    switches.set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

fn global(engine: &mut Engine, name: &str) -> Value {
    let env = engine.interp.global_env;
    engine.interp.get_var(name, &env).ok().expect("bound name")
}

/// The global function `name`'s AST, [[Realm]] and first-stage chunk.
fn function(engine: &mut Engine, name: &str) -> (Rc<crate::ast::Function>, usize, Rc<Chunk>) {
    let value = global(engine, name);
    let object = value.as_obj().expect("a function");
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        panic!("{name} is an author function")
    };
    let first = user
        .func
        .runtime()
        .code
        .get()
        .and_then(Option::clone)
        .expect("compiled");
    (user.func.clone(), user.realm, first)
}

/// The global function `name`'s second-stage chunk, if any.
fn second_stage(engine: &mut Engine, name: &str) -> Option<Rc<Chunk>> {
    let value = global(engine, name);
    let object = value.as_obj()?;
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        return None;
    };
    user.func.runtime().code2.get().and_then(Option::clone)
}

/// The ways the inline planner splices at call site `site` of the global function `name`'s first
/// stage, with the tier on or off on this thread for the plan (0: the site is not spliced).
fn spliced(engine: &mut Engine, name: &str, site: u32, on: bool) -> usize {
    let (func, realm, first) = function(engine, name);
    TestSwitches {
        on,
        ..Default::default()
    }
    .set();
    let env = engine.interp.global_env;
    let plan = plan_inlines(&first, &func, realm, &env);
    plan.get(&site).map_or(0, |entry| entry.ways.len())
}

/// Whether `name`'s first-stage call site `site` met another closure of a way's function.
fn recorded(engine: &mut Engine, name: &str, site: u32) -> bool {
    let (_, _, first) = function(engine, name);
    first.test_call_site(site).met_another_closure()
}

/// The distinct callees in the ways of `name`'s first-stage call site `site`.
fn callees(engine: &mut Engine, name: &str, site: u32) -> usize {
    let (_, _, first) = function(engine, name);
    let mut callees: Vec<usize> = first
        .test_call_site(site)
        .entries
        .iter()
        .map(|way| way.get().callee)
        .filter(|&callee| callee != 0)
        .collect();
    callees.sort_unstable();
    callees.dedup();
    callees.len()
}

/// Whether a run that met closures of one function at `apply1`'s site 0 is one to judge: its
/// ways hold one callee, the first closure. A run whose call-cache epoch moved under it holds
/// more (a way of a moved epoch is refilled, not hit by code; the epoch is process-wide, and
/// other tests' discards move it), and is run again.
fn judged(engine: &mut Engine) -> bool {
    match callees(engine, "apply1", 0) {
        1 => true,
        n => {
            assert!(n > 1, "a filled way");
            false
        }
    }
}

fn exits() -> Vec<(u32, ExitReason)> {
    super::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()))
}

fn discarded() -> Vec<usize> {
    super::TEST_DISCARDED.with(|discarded| std::mem::take(&mut *discarded.borrow_mut()))
}

/// Whether `chunk`'s resident code is the tier's (it has an exit table).
fn is_tier_code(chunk: &Chunk) -> bool {
    chunk
        .jit
        .get()
        .flatten()
        .is_some_and(|code| super::exit::table_bytes(code.mem_ptr()) != 0)
}

fn guards(chunk: &Chunk) -> usize {
    chunk
        .jit_ops()
        .iter()
        .filter(|op| matches!(op, Op::InlineGuard(..)))
        .count()
}

const APPLY: &str = "function apply1(f, x) { return f(x); }\n\
                     function make(k) { return (v) => v + k; }\n";

/// Runs `test` with a fresh engine up to three times, while its run's call-cache epoch moved
/// under it (`test` answers whether its run was judged: see [`judged`]).
fn with_retries(mut test: impl FnMut() -> bool) {
    for _ in 0..3 {
        if test() {
            return;
        }
    }
    panic!("the call-cache epoch moved in each of three runs");
}

#[test]
fn a_site_with_two_callees_is_not_spliced_with_the_tier_on_and_is_with_it_off() {
    let program = "function inc(x) { return x + 1; }\n\
                   function dbl(x) { return x * 2; }\n\
                   function apply2(f, x) { return f(x); }\n\
                   let t = 0; for (let k = 0; k < 300; k++) t += apply2(k % 2 ? inc : dbl, k); t";
    let mut off = engine(tier_off());
    assert_eq!(eval(&mut off, program), "67350");
    assert!(!recorded(&mut off, "apply2", 0), "no code-keyed hit");
    assert_eq!(spliced(&mut off, "apply2", 0, true), 0, "the tier's rule");
    assert_eq!(spliced(&mut off, "apply2", 0, false), 2, "P2's policy");
    // End to end: the tier off's second stage splices both callees; with the tier on, the plan
    // splices nothing, so the tier compiles apply2's first stage in place (step 14), with no
    // second stage.
    assert!(second_stage(&mut off, "apply2").is_some_and(|chunk| guards(&chunk) == 2));
    let mut on = engine(tier_on());
    exits();
    assert_eq!(eval(&mut on, program), "67350");
    assert!(second_stage(&mut on, "apply2").is_none(), "nothing spliced");
    let (_, _, first) = function(&mut on, "apply2");
    assert!(
        super::in_place(&first),
        "the tier's code on the first stage"
    );
    assert!(is_tier_code(&first));
    assert!(exits().is_empty());
}

#[test]
fn a_site_of_one_closure_is_spliced_with_the_tier_on_as_before() {
    let program = format!(
        "{APPLY}const one = make(1);\n\
         let t = 0; for (let k = 0; k < 300; k++) t += apply1(one, k); t"
    );
    let mut off = engine(tier_off());
    assert_eq!(eval(&mut off, &program), "45150");
    assert!(!recorded(&mut off, "apply1", 0));
    assert_eq!(spliced(&mut off, "apply1", 0, true), 1);
    assert_eq!(spliced(&mut off, "apply1", 0, false), 1);
    let mut on = engine(tier_on());
    exits();
    assert_eq!(eval(&mut on, &program), "45150");
    let chunk = second_stage(&mut on, "apply1").expect("the tier's chunk");
    assert_eq!(guards(&chunk), 1, "the closure spliced");
    assert!(is_tier_code(&chunk));
    assert!(exits().is_empty());
}

/// The first code-keyed hit through the machine code's full probe, the direct sequence entering
/// the callee after it (its chunk's inline attempt made by 150 calls from elsewhere), so that no
/// Rust path runs: the probe's store of the hint is the only record.
#[test]
fn a_first_code_keyed_hit_through_the_full_probe_is_recorded_and_not_spliced() {
    with_retries(|| {
        let mut e = engine(tier_off());
        let program = format!(
            "{APPLY}const one = make(1);\n\
             let t = 0; for (let k = 0; k < 150; k++) t += one(k);\n\
             t += apply1(one, 1) + apply1(one, 2); t"
        );
        assert_eq!(eval(&mut e, &program), "11330");
        assert!(!recorded(&mut e, "apply1", 0));
        assert_eq!(spliced(&mut e, "apply1", 0, true), 1, "one closure so far");
        tier_off().set();
        assert_eq!(eval(&mut e, "apply1(make(2), 40)"), "42");
        if !judged(&mut e) {
            return false;
        }
        assert!(recorded(&mut e, "apply1", 0), "the full probe's record");
        let (_, _, first) = function(&mut e, "apply1");
        assert_eq!(first.test_call_site(0).code_hint.get(), 0, "way 0's hit");
        assert_eq!(spliced(&mut e, "apply1", 0, true), 0, "the tier's rule");
        assert_eq!(spliced(&mut e, "apply1", 0, false), 1, "P2's policy");
        true
    });
}

/// Later code-keyed hits through the hinted-way test, which writes nothing: the record stays.
#[test]
fn code_keyed_hits_through_the_hinted_way_test_keep_the_site_unspliced() {
    with_retries(|| {
        let mut e = engine(tier_off());
        let program =
            format!("{APPLY}let t = 0; for (let k = 0; k < 90; k++) t += apply1(make(k), k); t");
        assert_eq!(eval(&mut e, &program), "8010");
        if !judged(&mut e) {
            return false;
        }
        assert!(recorded(&mut e, "apply1", 0), "the record");
        assert_eq!(spliced(&mut e, "apply1", 0, true), 0, "the tier's rule");
        assert_eq!(spliced(&mut e, "apply1", 0, false), 1, "P2's policy");
        true
    });
}

/// A code-keyed hit through the call-hit helper: the callee's chunk has not made its inline
/// attempt (four calls), so the direct sequence declines to the helper, which runs the live
/// closure.
#[test]
fn a_code_keyed_hit_through_the_call_hit_helper_is_recorded_and_not_spliced() {
    with_retries(|| {
        let mut e = engine(tier_off());
        let program = format!(
            "{APPLY}const one = make(1);\n\
             apply1(one, 1) + apply1(one, 2) + apply1(one, 3) + apply1(make(5), 10)"
        );
        assert_eq!(eval(&mut e, &program), "24");
        if !judged(&mut e) {
            return false;
        }
        assert!(recorded(&mut e, "apply1", 0), "the record");
        assert_eq!(spliced(&mut e, "apply1", 0, true), 0, "the tier's rule");
        assert_eq!(spliced(&mut e, "apply1", 0, false), 1, "P2's policy");
        true
    });
}

/// A code-keyed hit of the Rust code-sharing retry (`Interp::resolve_cached_call_entry`, which
/// the generic call helper, the VM's cached calls and the prepared calls take, and every call
/// on x86-64): it records the hit as the machine code's full probe does.
#[test]
fn a_code_keyed_hit_of_the_rust_code_sharing_retry_is_recorded_and_not_spliced() {
    with_retries(|| {
        let mut e = engine(tier_off());
        let program = format!(
            "{APPLY}const one = make(1); const two = make(2);\n\
             apply1(one, 1) + apply1(one, 2)"
        );
        assert_eq!(eval(&mut e, &program), "5");
        assert!(!recorded(&mut e, "apply1", 0));
        let two = global(&mut e, "two");
        let (_, _, first) = function(&mut e, "apply1");
        let site = first.test_call_site(0);
        let args = [Value::Num(40.0)];
        let this = Value::Undefined;
        let result = unsafe {
            e.interp
                .call_jit_cached(site, &two, &this, args.as_ptr(), args.len())
        };
        let Some(result) = result else {
            // A moved epoch staled the way: no code-keyed hit (the retry finds no current way).
            return false;
        };
        let Ok(value) = result else {
            panic!("the call threw")
        };
        assert!(matches!(value.view(), crate::value::ValueView::Num(n) if n == 42.0));
        assert!(site.met_another_closure(), "the retry's record");
        assert_eq!(site.code_hint.get(), 0, "way 0's hit");
        assert_eq!(spliced(&mut e, "apply1", 0, true), 0, "the tier's rule");
        assert_eq!(spliced(&mut e, "apply1", 0, false), 1, "P2's policy");
        true
    });
}

/// React's reducer: `Array.prototype.map` (self-hosted) called with a fresh arrow each time. Its
/// callback site meets every arrow by code: with the tier on it is not spliced, so its code has
/// no splice guard to fail and nothing exits or is discarded; with the tier off it is spliced
/// (the program is strict, as React's bundle is and as the self-hosted `map` is, and keeps the
/// arrows alive, so that P2's planner finds its way's callee alive and of its strictness).
#[test]
fn a_self_hosted_maps_callback_site_of_fresh_arrows_is_not_spliced_with_the_tier_on() {
    let program = "'use strict';\n\
                   const kept = [];\n\
                   function bump(list, k) { const f = (x) => x + k; kept.push(f); \
                   return list.map(f); }\n\
                   const l = [1, 2, 3];\n\
                   let t = 0; for (let k = 0; k < 2000; k++) t += bump(l, k)[2]; t";
    let mut on = engine(tier_on());
    exits();
    discarded();
    assert_eq!(eval(&mut on, program), "2005000");
    let taken = exits();
    assert!(
        !taken.iter().any(|&(_, why)| why == ExitReason::WrongCallee),
        "{taken:?}"
    );
    assert!(discarded().is_empty());
    let mut off = engine(tier_off());
    assert_eq!(eval(&mut off, program), "2005000");
    // The callback site's record, and P2's splice of it in the tier off's plan.
    eval(&mut off, "var selfHostedMap = Array.prototype.map;");
    let (func, realm, first) = function(&mut off, "selfHostedMap");
    assert!(func.self_hosted);
    let site = first
        .jit_ops()
        .iter()
        .find_map(|op| match *op {
            Op::Call(_, c) | Op::CallWithThis(_, c)
                if first.test_call_site(c).met_another_closure() =>
            {
                Some(c)
            }
            _ => None,
        })
        .expect("the callback site met fresh arrows");
    let env = off.interp.global_env;
    TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    assert!(!plan_inlines(&first, &func, realm, &env).contains_key(&site));
    TestSwitches::default().set();
    assert!(plan_inlines(&first, &func, realm, &env).contains_key(&site));
}

// Step 15b: Maglev's polymorphic method calls.

/// Whether `name`'s first-stage call site `site` is marked a named method call.
fn marked(engine: &mut Engine, name: &str, site: u32) -> bool {
    let (_, _, first) = function(engine, name);
    first.test_call_site(site).is_named_method_call()
}

/// The ways the inline planner splices at call site `nested` of the callee spliced by the first
/// way of call site `site` of the global function `name`'s first stage (depth 1), with the tier
/// on or off for the plan (0: the outer site or the nested one is not spliced).
fn nested_spliced(engine: &mut Engine, name: &str, site: u32, nested: u32, on: bool) -> usize {
    let (func, realm, first) = function(engine, name);
    TestSwitches {
        on,
        ..Default::default()
    }
    .set();
    let env = engine.interp.global_env;
    let plan = plan_inlines(&first, &func, realm, &env);
    plan.get(&site)
        .and_then(|entry| entry.ways.first())
        .and_then(|way| way.nested.get(&nested))
        .map_or(0, |entry| entry.ways.len())
}

/// Two constructors in Octane's style (sloppy functions with prototype methods), whose `get`
/// methods are distinct functions.
const CLASSES: &str = "function A(v) { this.v = v; }\n\
                       A.prototype.get = function () { return this.v + 1; };\n\
                       function B(v) { this.v = v; }\n\
                       B.prototype.get = function () { return this.v * 2; };\n";

/// The bytecode compiler marks a call site a named method call where its callee is
/// `Op::GetMethod`'s value, whatever the receiver's expression or an optional link: `o.m()`,
/// `o.p.m()` (`this.task.run()`'s shape), `o?.m()`, `o.m?.()`, `(o.m)()`, a spread call's site
/// (which the planner never splices), and a chain's named links; not a keyed method call
/// (`Op::GetMethodElem`, a literal key's too), a local's or a free name's call, a comma
/// expression's, a tagged template's, a super or private method's.
#[test]
fn the_compiler_marks_the_calls_whose_callee_a_named_method_read() {
    let mut e = engine(tier_off());
    let program = "function shapes(o, k, f) {\n\
                     o.m(1); o[k](2); f(3); g(4); o?.m(5); o.m?.(6); o.p.m(7); (o.m)(8);\n\
                     (0, o.m)(9); o.m(...[10]); o.m`x`; o['m'](11);\n\
                   }\n\
                   function chain(o, k) { return o.a().b()[k]().c(); }\n\
                   class Base { m() { return 1; } }\n\
                   class Derived extends Base {\n\
                     #p() { return 2; }\n\
                     q() { return 3; }\n\
                     run() { return super.m() + this.#p() + this.q(); }\n\
                   }\n\
                   var run = Derived.prototype.run;\n\
                   function g(x) { return x; }\n\
                   const o = { m(x) { return x; }, p: { m(x) { return x; } } };\n\
                   o.a = function () { return { b() { return [function () { \
                   return { c() { return 4; } }; }]; } }; };\n\
                   shapes(o, 'm', g);\n\
                   chain(o, 0) + new Derived().run()";
    assert_eq!(eval(&mut e, program), "10");
    let expected = [
        ("o.m()", true),
        ("o[k]()", false),
        ("f() of a local", false),
        ("g() of a free name", false),
        ("o?.m()", true),
        ("o.m?.()", true),
        ("o.p.m()", true),
        ("(o.m)()", true),
        ("(0, o.m)()", false),
        ("o.m(...spread)", true),
        ("o.m`tagged`", false),
        ("o['m']()", false),
    ];
    for (site, (shape, named)) in expected.iter().enumerate() {
        assert_eq!(marked(&mut e, "shapes", site as u32), *named, "{shape}");
    }
    // o.a() before the chain's links .b(), [k]() and .c().
    for (site, named) in [true, true, false, true].into_iter().enumerate() {
        assert_eq!(marked(&mut e, "chain", site as u32), named, "{site}");
    }
    for (site, named) in [false, false, true].into_iter().enumerate() {
        assert_eq!(marked(&mut e, "run", site as u32), named, "run site {site}");
    }
}

/// End to end with the tier on, a second stage that splices a site's several callees, as before
/// step 15: its first tier compile takes no guard's mismatch edge, which the template code's
/// feedback has not seen, so the other callee's calls exit there (WrongCallee) up to the exit
/// limit, whose discard flags the guard; the reoptimization takes the mismatch edge, and the
/// second stage ends with the tier's code. One discard, every exit WrongCallee.
fn assert_settled_after_the_guards_warm_up(chunk: &Chunk) {
    let taken = exits();
    assert!(
        !taken.is_empty() && taken.iter().all(|&(_, why)| why == ExitReason::WrongCallee),
        "{taken:?}"
    );
    assert_eq!(discarded().len(), 1, "the warm-up's discard");
    assert!(is_tier_code(chunk), "the reoptimization's code");
}

/// A named method site over two classes: with the tier on its two callees splice, guarded by
/// callee, as before step 15 and as with the tier off; end to end, the tier's second stage holds
/// both guards and settles with the tier's code.
#[test]
fn a_named_method_site_over_two_classes_is_spliced_with_the_tier_on() {
    let program = format!(
        "{CLASSES}function call2(o) {{ return o.get(); }}\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += call2(k % 2 ? new A(k) : new B(k)); t"
    );
    let mut off = engine(tier_off());
    assert_eq!(eval(&mut off, &program), "749500");
    assert!(marked(&mut off, "call2", 0));
    assert_eq!(callees(&mut off, "call2", 0), 2);
    assert!(!recorded(&mut off, "call2", 0), "no code-keyed hit");
    assert_eq!(spliced(&mut off, "call2", 0, true), 2, "the tier's rule");
    assert_eq!(spliced(&mut off, "call2", 0, false), 2, "P2's policy");
    let mut on = engine(tier_on());
    exits();
    discarded();
    assert_eq!(eval(&mut on, &program), "749500");
    let chunk = second_stage(&mut on, "call2").expect("the tier's chunk");
    assert_eq!(guards(&chunk), 2, "both callees spliced");
    assert_settled_after_the_guards_warm_up(&chunk);
}

/// A keyed method site (`o[k]()`, `Op::GetMethodElem`) over the same two callees: not spliced
/// with the tier on (Maglev's continuation follows a named load alone), spliced with it off; the
/// tier compiles its first stage in place, with no second stage.
#[test]
fn a_keyed_method_site_over_two_callees_is_not_spliced_with_the_tier_on() {
    let program = format!(
        "{CLASSES}function callk(o, k) {{ return o[k](); }}\n\
         let t = 0; for (let k = 0; k < 300; k++) \
         t += callk(k % 2 ? new A(k) : new B(k), 'get'); t"
    );
    let mut off = engine(tier_off());
    assert_eq!(eval(&mut off, &program), "67350");
    assert!(!marked(&mut off, "callk", 0));
    assert_eq!(callees(&mut off, "callk", 0), 2);
    assert_eq!(spliced(&mut off, "callk", 0, true), 0, "the tier's rule");
    assert_eq!(spliced(&mut off, "callk", 0, false), 2, "P2's policy");
    let mut on = engine(tier_on());
    exits();
    assert_eq!(eval(&mut on, &program), "67350");
    assert!(second_stage(&mut on, "callk").is_none(), "nothing spliced");
    let (_, _, first) = function(&mut on, "callk");
    assert!(super::in_place(&first));
    assert!(is_tier_code(&first));
    assert!(exits().is_empty());
}

/// A free name's call over two functions: `Op::LoadNameForCall` then `Op::CallWithThis`, the
/// method call's operation, but its callee no named property's value: not spliced with the tier
/// on, spliced with it off.
#[test]
fn a_free_names_call_over_two_functions_is_not_spliced_with_the_tier_on() {
    let program = "function inc(x) { return x + 1; }\n\
                   function dbl(x) { return x * 2; }\n\
                   var g = inc;\n\
                   function callg(x) { return g(x); }\n\
                   let t = 0; for (let k = 0; k < 300; k++) { g = k % 2 ? inc : dbl; \
                   t += callg(k); } t";
    let mut off = engine(tier_off());
    assert_eq!(eval(&mut off, program), "67350");
    let (_, _, first) = function(&mut off, "callg");
    assert!(first
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::CallWithThis(_, 0))));
    assert!(!marked(&mut off, "callg", 0));
    assert_eq!(callees(&mut off, "callg", 0), 2);
    assert_eq!(spliced(&mut off, "callg", 0, true), 0, "the tier's rule");
    assert_eq!(spliced(&mut off, "callg", 0, false), 2, "P2's policy");
}

/// A plain call, a named method call and a keyed method call side by side (sites 0, 1 and 2),
/// each over two callees: with the tier on only the named one splices, so each site's own mark
/// decides (a mark read from a neighbouring site would splice the plain or the keyed call, or
/// not the named one); with the tier off all three splice. End to end, the tier's second stage
/// holds the named site's two guards alone and settles with the tier's code.
#[test]
fn only_the_named_method_site_among_its_neighbours_is_spliced_with_the_tier_on() {
    let program = format!(
        "{CLASSES}function inc(x) {{ return x + 1; }}\n\
         function dbl(x) {{ return x * 2; }}\n\
         function three(o, f, k) {{ return f(o.v) + o.get() + o[k](); }}\n\
         let t = 0; for (let k = 0; k < 1000; k++) \
         t += three(k % 2 ? new A(k) : new B(k), k % 2 ? inc : dbl, 'get'); t"
    );
    let mut off = engine(tier_off());
    assert_eq!(eval(&mut off, &program), "2248500");
    for (site, named) in [false, true, false].into_iter().enumerate() {
        let site = site as u32;
        assert_eq!(marked(&mut off, "three", site), named, "site {site}");
        assert_eq!(callees(&mut off, "three", site), 2, "site {site}");
        let ways = if named { 2 } else { 0 };
        assert_eq!(spliced(&mut off, "three", site, true), ways, "site {site}");
        assert_eq!(spliced(&mut off, "three", site, false), 2, "site {site}");
    }
    let mut on = engine(tier_on());
    exits();
    discarded();
    assert_eq!(eval(&mut on, &program), "2248500");
    let chunk = second_stage(&mut on, "three").expect("the tier's chunk");
    assert_eq!(guards(&chunk), 2, "the named site's two callees alone");
    assert_settled_after_the_guards_warm_up(&chunk);
}

/// Inside a spliced callee (depth 1) the rule reads the callee's own first-stage site, marked at
/// its own compile: a named method site over two classes splices there with the tier on, a keyed
/// one does not.
#[test]
fn a_spliced_callees_named_method_site_is_spliced_with_the_tier_on() {
    let program = format!(
        "{CLASSES}function inner(o) {{ return o.get(); }}\n\
         function innerk(o) {{ return o['get'](); }}\n\
         function outer(o) {{ return inner(o) + 1; }}\n\
         function outerk(o) {{ return innerk(o) + 1; }}\n\
         let t = 0; for (let k = 0; k < 300; k++) {{ const o = k % 2 ? new A(k) : new B(k); \
         t += outer(o) - outerk(o); }} t"
    );
    let mut off = engine(tier_off());
    assert_eq!(eval(&mut off, &program), "0");
    for (outer, ways) in [("outer", 2), ("outerk", 0)] {
        assert_eq!(spliced(&mut off, outer, 0, true), 1, "{outer}'s one callee");
        let tier = nested_spliced(&mut off, outer, 0, 0, true);
        let p2 = nested_spliced(&mut off, outer, 0, 0, false);
        assert_eq!((tier, p2), (ways, 2), "{outer}: the tier's, P2's");
    }
}

/// A named method site whose ways hold two distinct functions and which met another closure of
/// one of them (C's `get` is another closure of B's, a code-keyed hit at B's way): not spliced
/// with the tier on, as the rule's closure half declines any such site; spliced with it off.
#[test]
fn a_named_method_site_that_met_another_closure_is_not_spliced_with_the_tier_on() {
    with_retries(|| {
        let program = "function A(v) { this.v = v; }\n\
                       A.prototype.get = function () { return this.v + 1; };\n\
                       function make(n) { return function () { return this.v * n; }; }\n\
                       function B(v) { this.v = v; }\n\
                       B.prototype.get = make(2);\n\
                       function C(v) { this.v = v; }\n\
                       C.prototype.get = make(3);\n\
                       function call3(o) { return o.get(); }\n\
                       let t = 0; for (let k = 0; k < 300; k++) \
                       t += call3(k % 3 == 0 ? new A(k) : k % 3 == 1 ? new B(k) : new C(k)); t";
        let mut off = engine(tier_off());
        assert_eq!(eval(&mut off, program), "90000");
        // A run whose call-cache epoch moved under it refilled a way for C's closure rather than
        // hitting B's way by code (see `judged`): run again.
        match callees(&mut off, "call3", 0) {
            2 => {}
            n => {
                assert!(n > 2, "a filled way");
                return false;
            }
        }
        assert!(marked(&mut off, "call3", 0));
        assert!(recorded(&mut off, "call3", 0), "C's code-keyed hit");
        assert_eq!(spliced(&mut off, "call3", 0, true), 0, "the tier's rule");
        assert_eq!(spliced(&mut off, "call3", 0, false), 2, "P2's policy");
        // End to end with the tier on, the plan made after C's first call, which recorded its
        // code-keyed hit (later calls may refill a way for C: the in-place installs of the
        // callees' code stale the ways' entry words); a run that never recorded it had its epoch
        // moved before C's first call, and is run again.
        let mut on = engine(tier_on());
        exits();
        assert_eq!(eval(&mut on, program), "90000");
        if !recorded(&mut on, "call3", 0) {
            return false;
        }
        assert!(second_stage(&mut on, "call3").is_none(), "nothing spliced");
        assert!(!exits()
            .iter()
            .any(|&(_, why)| why == ExitReason::WrongCallee));
        true
    });
}
