//! The optimizing tier end to end (P3 stage 1, milestone 2): programs whose hot functions reach
//! the second-stage checkpoints run with the tier on, eager, and with the forced-exit gate, and
//! their results equal the bytecode VM's; a callee-free loop function runs in the tier's code;
//! repeated exits move a function's new entries to the template JIT's code.

use super::TestSwitches;
use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

fn run(tier: Tier, source: &str) -> String {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    eval(&mut engine, source)
}

fn compiled() -> Vec<usize> {
    super::TEST_COMPILED.with(|compiled| std::mem::take(&mut *compiled.borrow_mut()))
}

fn exits() -> Vec<(u32, super::ir::ExitReason)> {
    super::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()))
}

/// Programs whose functions run past the second stage's checkpoints (100, 500, 2,100 and 8,500
/// native runs).
const PROGRAMS: &[&str] = &[
    // A callee-free arithmetic loop.
    "function sum(n) { let s = 0; for (let i = 0; i < n; i++) s = s + i; return s; }\n\
     let t = 0; for (let k = 0; k < 3000; k++) t += sum(k % 50); t",
    // int32 arithmetic that overflows inside its loop.
    "function mulsum(n) { let s = 1; for (let i = 1; i < n; i++) s = (s * 31 + i) | 0; \
     return s; }\n\
     let t = 0; for (let k = 0; k < 3000; k++) t = (t + mulsum(k % 40)) | 0; t",
    // Named reads and writes on constructed objects.
    "function P(x, y) { this.x = x; this.y = y; }\n\
     function len2(p) { return p.x * p.x + p.y * p.y; }\n\
     function bump(p) { p.x = p.x + 1; return p.x; }\n\
     let t = 0; for (let k = 0; k < 3000; k++) { const p = new P(k, k + 1); \
     t += len2(p) + bump(p); } t",
    // Calls, closures and strings.
    "function add(a, b) { return a + b; }\n\
     function twice(f, x) { return f(f(x)); }\n\
     let s = ''; for (let k = 0; k < 3000; k++) \
     s = add(s.length > 100 ? '' : s, String(twice(function (v) { return v + 1; }, k))); s",
    // Branches on values of more than one type: some checks fail.
    "function cls(v) { if (v < 10) return 'small'; if (v < 100) return 'mid'; return 'big'; }\n\
     let out = []; for (let k = 0; k < 3000; k++) out.push(cls(k % 7 === 0 ? String(k) : k));\n\
     out.join(',').length + ':' + out[7] + out[3]",
    // Truthiness of every kind of value.
    "function yes(v) { return v ? 1 : 0; }\n\
     const vals = [0, 1, '', 'a', null, undefined, {}, [], NaN, -0, 2n, 0n];\n\
     let t = 0; for (let k = 0; k < 3000; k++) t = t * 2 % 1000003 + yes(vals[k % vals.length]); t",
    // Throws inside a try.
    "function risky(k) { try { if (k % 13 === 0) throw new Error('e' + k); return k; } \
     catch (e) { return -1; } }\n\
     let t = 0; for (let k = 0; k < 3000; k++) t += risky(k); t",
    // for-of over an Array.
    "function total(a) { let s = 0; for (const v of a) s += v; return s; }\n\
     const arr = [1, 2, 3, 4, 5]; let t = 0; for (let k = 0; k < 3000; k++) t += total(arr); t",
    // A shape change after the feedback settles.
    "function getx(o) { return o.x; }\n\
     let t = 0; for (let k = 0; k < 3000; k++) t += getx(k < 2500 ? {x: k} : {y: 0, x: -k}); t",
    // A for-of whose body sometimes throws: the iterator's handler closes it on the way out.
    "function firstBig(a) { for (const v of a) { if (v > 1000) throw new RangeError('big ' + v); } \
     return a.length; }\n\
     function run(k) { try { return firstBig(k % 97 === 0 ? [1, 2, 5000] : [1, 2, 3]); } \
     catch (e) { return e.message; } }\n\
     let out = ''; for (let k = 0; k < 3000; k++) { const r = run(k); \
     out += typeof r === 'string' ? r[4] : r; }\n\
     out.length + out.slice(0, 40)",
];

fn with_switches(switches: TestSwitches, source: &str) -> String {
    switches.set();
    run(Tier::Jit, source)
}

#[test]
fn results_equal_the_vms_with_the_tier_on_eager_and_forcing_exits() {
    let configurations = [
        (
            "on",
            TestSwitches {
                on: true,
                eager: false,
                exit_every: 0,
            },
        ),
        (
            "eager",
            TestSwitches {
                on: true,
                eager: true,
                exit_every: 0,
            },
        ),
        (
            "forced exits",
            TestSwitches {
                on: true,
                eager: true,
                exit_every: 7,
            },
        ),
    ];
    let mut compiles = 0;
    for program in PROGRAMS {
        let expected = run(Tier::Bytecode, program);
        for (label, switches) in configurations {
            compiled();
            assert_eq!(
                with_switches(switches, program),
                expected,
                "{label}: {program}"
            );
            compiles += compiled().len();
        }
    }
    assert!(
        compiles > 0,
        "the tier compiled some of the programs' functions"
    );
}

fn second_stage(engine: &mut Engine, name: &str) -> Option<std::rc::Rc<crate::bytecode::Chunk>> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok()?;
    let object = value.as_obj()?;
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        return None;
    };
    user.func.runtime().code2.get().and_then(Option::clone)
}

#[test]
fn a_callee_free_loop_function_runs_in_the_tiers_code() {
    const LOOP: &str =
        "function sum(n) { let s = 0; for (let i = 0; i < n; i++) s = s + i; return s; }";
    // The tier off: a function with nothing to splice keeps its first stage.
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    TestSwitches::default().set();
    eval(&mut engine, LOOP);
    assert_eq!(
        eval(
            &mut engine,
            "let t = 0; for (let k = 0; k < 1000; k++) t += sum(10); t"
        ),
        "45000"
    );
    assert!(second_stage(&mut engine, "sum").is_none());
    // The tier on: the function's own chunk, compiled by the tier at a checkpoint.
    TestSwitches {
        on: true,
        eager: true,
        exit_every: 0,
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    compiled();
    eval(&mut engine, LOOP);
    assert_eq!(
        eval(
            &mut engine,
            "let t = 0; for (let k = 0; k < 1000; k++) t += sum(10); t"
        ),
        "45000"
    );
    let chunk = second_stage(&mut engine, "sum").expect("the tier's chunk was published");
    assert!(
        compiled().contains(&(std::rc::Rc::as_ptr(&chunk) as usize)),
        "the tier compiled it"
    );
    // Nothing spliced: the first stage's operations, its feedback forwarded by the record map.
    let first = {
        let env = engine.interp.global_env;
        let value = engine.interp.get_var("sum", &env).ok().unwrap();
        let object = value.as_obj().unwrap();
        let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
            panic!("an author function")
        };
        user.func
            .runtime()
            .code
            .get()
            .and_then(Option::clone)
            .unwrap()
    };
    assert_eq!(
        format!("{:?}", chunk.jit_ops()),
        format!("{:?}", first.jit_ops())
    );
    assert!(chunk.feedback_vector().record_map().is_some());
    let code = chunk.jit.get().flatten().expect("its code is resident");
    exits();
    assert_eq!(
        eval(
            &mut engine,
            "let u = 0; for (let k = 0; k < 1000; k++) u += sum(k % 20); u"
        ),
        "57000"
    );
    assert!(exits().is_empty(), "the loop runs without leaving the code");
    code.referenced.set(0);
    assert_eq!(eval(&mut engine, "sum(100000)"), "4999950000");
    assert!(code.referenced.get() != 0, "the tier's code ran");
}

#[test]
fn a_for_of_whose_binding_takes_its_handlers_operand_runs_in_the_tiers_code() {
    // The for-of pushes its iterator's handler with the step's value on the operand stack, and
    // the binding's store pops it: the frame states inside the loop are shallower than the
    // handler record, as the VM and the first stage allow.
    TestSwitches {
        on: true,
        eager: true,
        exit_every: 0,
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    compiled();
    eval(
        &mut engine,
        "function total(a) { let s = 0; for (const v of a) s += JSON.parse(v); return s; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += total(['1', '2', String(k)]); t",
    );
    let chunk = second_stage(&mut engine, "total").expect("the tier compiled total");
    assert!(compiled().contains(&(std::rc::Rc::as_ptr(&chunk) as usize)));
    let code = chunk.jit.get().flatten().expect("its code is resident");
    code.referenced.set(0);
    exits();
    assert_eq!(eval(&mut engine, "total(['4', '5', '6'])"), "15");
    assert!(code.referenced.get() != 0, "the tier's code ran");
    assert!(exits().is_empty());
    // A throw in the loop's call leaves with the throw pending, below the record's depth: the
    // handler closes the iterator and the caller's catch takes the error.
    assert_eq!(
        eval(
            &mut engine,
            "let r = 'none'; try { total(['1', 'x', '3']); } catch (e) { r = e.name; } r"
        ),
        "SyntaxError"
    );
    assert_eq!(
        exits()
            .into_iter()
            .map(|(_, reason)| reason)
            .collect::<Vec<_>>(),
        vec![super::ir::ExitReason::Throw]
    );
    // A failed check inside the loop: the VM resumes in the region with the handler rebuilt.
    assert_eq!(eval(&mut engine, "total(['1', '\"2\"', '3'])"), "123");
    assert_eq!(
        exits()
            .into_iter()
            .map(|(_, reason)| reason)
            .collect::<Vec<_>>(),
        vec![super::ir::ExitReason::NotNumber]
    );
}

#[test]
fn repeated_exits_send_new_entries_to_the_template_jits_code() {
    TestSwitches {
        on: true,
        eager: true,
        exit_every: 0,
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    compiled();
    eval(
        &mut engine,
        "function add(a, b) { return a + b; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += add(k, 1); t",
    );
    let chunk = second_stage(&mut engine, "add").expect("the tier compiled add");
    assert!(compiled().contains(&(std::rc::Rc::as_ptr(&chunk) as usize)));
    let optimized = chunk.jit.get().flatten().expect("resident");
    exits();
    super::TEST_FELL_BACK.with(|fell| fell.borrow_mut().clear());
    // Strings fail the Number checks: every call exits until the code falls back.
    assert_eq!(
        eval(
            &mut engine,
            "let s = ''; for (let k = 0; k < 300; k++) s = add('', k % 10); s"
        ),
        "9"
    );
    let taken = exits().len();
    assert_eq!(
        taken,
        super::EXIT_LIMIT as usize,
        "no exit after the fallback"
    );
    let fell = super::TEST_FELL_BACK.with(|fell| fell.borrow().clone());
    assert_eq!(fell, vec![optimized.mem_ptr() as usize]);
    let now = chunk.jit.get().flatten().expect("the template JIT's code");
    assert!(!std::rc::Rc::ptr_eq(&now, &optimized));
    assert_eq!(eval(&mut engine, "add(2, 3) + add('a', 'b')"), "5ab");
    assert!(exits().is_empty());
    drop(optimized);
    super::release_retired(&mut engine.interp);
    assert!(
        engine.interp.opt_retired.is_empty(),
        "no activation runs the retired code"
    );
}

#[test]
fn the_settled_feedback_rule_waits_for_an_unchanged_fingerprint() {
    TestSwitches {
        on: true,
        eager: false,
        exit_every: 0,
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "function getx(o) { return o.x; } getx({x: 1});",
    );
    let env = engine.interp.global_env;
    let value = engine.interp.get_var("getx", &env).ok().unwrap();
    let object = value.as_obj().unwrap();
    let chunk = {
        let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
            panic!("an author function")
        };
        user.func
            .runtime()
            .code
            .get()
            .and_then(Option::clone)
            .unwrap()
    };
    assert!(!super::settled(&chunk), "the first checkpoint records");
    assert!(super::settled(&chunk), "unchanged feedback has settled");
    eval(&mut engine, "getx({y: 1, x: 2});");
    assert!(!super::settled(&chunk), "a new receiver shape changes it");
    assert!(super::settled(&chunk));
    super::forget(&chunk);
    assert!(!super::settled(&chunk), "a forgotten chunk starts again");
    TestSwitches {
        on: true,
        eager: true,
        exit_every: 0,
    }
    .set();
    super::forget(&chunk);
    assert!(
        super::settled(&chunk),
        "eager compiles at the first checkpoint"
    );
    super::forget(&chunk);
}
