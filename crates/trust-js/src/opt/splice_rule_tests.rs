//! The exemplars' splice rule (P3 stage 2, step 15): with the optimizing tier on, the inline
//! planner splices no call site whose ways hold more than one distinct callee, and none that met
//! another closure of a way's function, which the site records once, at its first code-keyed hit
//! (`CallSite::code_hint`); with the tier off it keeps P2's policy. Each record path in turn: the
//! machine code's full probe (then the hinted-way test and the call-hit helper), the Rust
//! code-sharing retry; a site of one closure splices as before; React's `map` callback; and a
//! late second closure, which the reoptimization meets with the guard's mismatch edge, not a new
//! plan.

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
