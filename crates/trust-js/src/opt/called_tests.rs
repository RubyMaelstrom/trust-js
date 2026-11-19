//! P1's tests (P3 stage 3 step 7: target/p3/stage3/s7-impl/design-note.txt, section 4): a call
//! the graph does not inline whose callee the compile knows is the known callee's direct call
//! (`oracle::called_ways`, `Builder::called_or_effect`, `jit::emit_known_sequence`): its results,
//! its exclusions, the stack and `caller` across it, a throw through it, the callee's code
//! changing between calls, and the teacher (the guard's WrongCallee exit, then the generic call).
use super::ir::ExitReason;
use super::TestSwitches;
use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

/// `source`'s completion with the tier off (the bytecode VM).
fn tier_off(source: &str) -> String {
    TestSwitches::default().set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, source)
}

/// `source`'s completion with the tier on and eager at TRUST_JS_OPT_INLINE=graph.
fn graph(source: &str) -> String {
    TestSwitches {
        on: true,
        eager: true,
        inline_graph: Some(true),
        ..TestSwitches::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source)
}

/// The calls built as known callees' direct calls since the last call.
fn known_calls() -> usize {
    super::build::TEST_KNOWN_CALLS.with(|count| count.replace(0))
}

/// The exits taken since the last call, by reason.
fn exits_for(reason: ExitReason) -> usize {
    super::exit::TEST_EXITS.with(|exits| {
        std::mem::take(&mut *exits.borrow_mut())
            .into_iter()
            .filter(|&(_, r)| r == reason)
            .count()
    })
}

/// A body the planner never inlines ("callee size/shape": over its 96 operations), which runs
/// `tail` (its parameters' names free to use) after a branch no call takes.
fn big(name: &str, params: &str, tail: &str) -> String {
    let dead: String = (0..60).map(|k| format!("t = t + {k}; ")).collect();
    format!("function {name}({params}) {{ var t = 0; if (t === 1) {{ {dead} }} {tail} }}\n")
}

/// `driver` after `defs`, warmed in a loop, its completion at both tiers equal; the known calls
/// the graph built.
fn agree(defs: &str, driver: &str) -> usize {
    let source = format!(
        "{defs}\nfunction run(n) {{ var r; for (var i = 0; i < n; i++) {{ r = {driver}; }} return r; }}\n\
         for (var w = 0; w < 80; w++) run(200);\nString(run(500));"
    );
    let off = tier_off(&source);
    known_calls();
    let on = graph(&source);
    assert_eq!(on, off, "the program:\n{source}");
    known_calls()
}

#[test]
fn a_known_callee_is_called_directly_with_the_tier_off_results() {
    // The argument count equal, fewer, more; `arguments`; a rest parameter; a strict callee's
    // `this`; a method call; an arrow.
    let cases: Vec<(String, &str)> = vec![
        (big("g", "a, b", "return a + b * 2;"), "g(i, 3)"),
        (
            big("g", "a, b, c", "return (c === undefined ? 10 : 20) + a;"),
            "g(i)",
        ),
        (big("g", "a", "return a;"), "g(i, 1, 2, 3)"),
        (big("g", "a", "return arguments.length + a;"), "g(i, 1, 2)"),
        (
            big("g", "a, ...r", "return r.length * 10 + a;"),
            "g(i, 1, 2)",
        ),
        (
            format!(
                "var o = {{ k: 5, m: {} }};",
                big("", "a", "return this.k + a;").trim_end()
            ),
            "o.m(i)",
        ),
        // A sloppy method on a primitive receiver at every call: the receiver's test.
        (
            format!(
                "var o = {{ m: {} }}; String.prototype.m = o.m;",
                big("", "a", "return typeof this + a;").trim_end()
            ),
            "'str'.m(i)",
        ),
        // An arrow reading its lexical new.target, made in a construction.
        (
            big("", "a", "return typeof new.target + a;")
                .replacen("function (a) {", "function K() { this.f = (a) => {", 1)
                .trim_end()
                .to_string()
                + "; }\nvar k = new K();",
            "k.f(i)",
        ),
        // An arrow reading its lexical new.target, called directly by an arrow that a construction
        // called (the direct sequence's construct-state path keeps an arrow's new.target).
        (
            big("", "x", "return typeof new.target + x;")
                .replacen(
                    "function (x) {",
                    "var A, B; function K(n) { if (!A) { B = (x) => {",
                    1,
                )
                .trim_end()
                .to_string()
                + "; A = (n) => { var s; for (var j = 0; j < n; j++) s = B(j); return s; }; } \
                   this.s = A(n); }",
            "new K(50).s",
        ),
        // A sloppy method on a primitive receiver after objects: the receiver's test.
        (
            format!(
                "var o = {{ k: 5, m: {} }}; String.prototype.m = o.m;",
                big("", "a", "return typeof this + a;").trim_end()
            ),
            "(i > 450 ? 'str' : o).m(i)",
        ),
        (
            format!(
                "var o = {{ k: 5, m: {} }};",
                big("", "a", "return this.k + a;").trim_end().replacen(
                    "{ var t",
                    "{ 'use strict'; var t",
                    1
                )
            ),
            "o.m(i)",
        ),
        (
            big("", "a", "return a + 1;")
                .replacen("function (a) {", "var g = (a) => {", 1)
                .trim_end()
                .to_string()
                + ";",
            "g(i)",
        ),
    ];
    for (defs, driver) in cases {
        assert!(
            agree(&defs, driver) > 0,
            "no known call for {driver} after {defs}"
        );
    }
}

#[test]
fn excluded_callees_keep_the_generic_call() {
    // A class constructor called without new, a bound function, a native, a generator.
    let cases: Vec<(String, &str)> = vec![
        (
            "class C { constructor(a) { this.a = a; } }\nfunction t(i) { try { return C(i); } catch (e) { return e.name; } }"
                .to_string(),
            "t(i)",
        ),
        (format!("{}var b = g.bind(null, 7);", big("g", "a, x", "return a + x;")), "b(i)"),
        (String::new(), "Math.abs(-i)"),
        ("function* gen(a) { yield a; }".to_string(), "gen(i).next().value"),
    ];
    for (k, (defs, driver)) in cases.into_iter().enumerate() {
        let known = agree(&defs, driver);
        // The class constructor's caller `t` (a try) may itself be a known callee of `run`.
        if k == 0 {
            continue;
        }
        assert_eq!(known, 0, "a known call for {driver}");
    }
}

#[test]
fn the_stack_caller_and_a_throw_across_known_calls_are_the_tier_offs() {
    // A throw from a nested known call caught two frames up, the stack's function names
    // printed; Function#caller read in the callee.
    let defs = format!(
        "{}{}{}",
        big(
            "inner",
            "a",
            "if (a === 400) throw new Error('at ' + a); return a;"
        ),
        big("mid", "a", "return inner(a) + 1;"),
        big("who", "a", "return who.caller === run2 ? a : -1;")
    );
    let source = format!(
        "{defs}\nfunction run2(n) {{ var r = 0; for (var i = 0; i < n; i++) {{ r += who(1); }} return r; }}\n\
         function run(n) {{ var r = 0; for (var i = 0; i < n; i++) {{ try {{ r += mid(i); }} catch (e) {{\n\
           r = e.message + '|' + e.stack.split('\\n').slice(1, 4).map(function (l) {{ return l.trim().split(' ')[1]; }}).join(','); break; }} }} return r; }}\n\
         for (var w = 0; w < 80; w++) {{ run(200); run2(50); }}\nString(run(500)) + ';' + run2(100);"
    );
    let off = tier_off(&source);
    known_calls();
    let on = graph(&source);
    assert_eq!(on, off);
    assert!(off.contains("inner,mid,run"), "the stack's names: {off}");
    assert!(known_calls() > 0);
}

#[test]
fn a_callee_replaced_after_the_compile_exits_once_then_calls_generically() {
    // The teacher: the guard's WrongCallee exit flags the site; the recompile builds the generic
    // call (JSC's BadConstantValue rule). The always-wrong form: a callee that changes at every
    // call after the compile.
    let defs = format!(
        "{}{}",
        big("g", "a", "return a + 1;"),
        big("h", "a", "return a + 2;")
    );
    let once = format!(
        "{defs}var o = {{ g: g }};\nfunction run(n) {{ var r = 0; for (var i = 0; i < n; i++) r += o.g(i); return r; }}\n\
         for (var w = 0; w < 80; w++) run(200);\nvar before = run(100); o.g = h; String(before) + ',' + run(5000);"
    );
    assert_eq!(graph(&once), tier_off(&once));
    let wrong = exits_for(ExitReason::WrongCallee);
    assert!((1..=3).contains(&wrong), "WrongCallee exits: {wrong}");
    let every = format!(
        "{defs}\nvar fs = [g, h];\nfunction run(n) {{ var r = 0; for (var i = 0; i < n; i++) r += fs[0](i); return r; }}\n\
         for (var w = 0; w < 80; w++) run(200);\nvar r = 0; for (var k = 0; k < 200; k++) {{ fs.reverse(); r += run(50); }} String(r);"
    );
    exits_for(ExitReason::WrongCallee);
    assert_eq!(graph(&every), tier_off(&every));
    let wrong = exits_for(ExitReason::WrongCallee);
    // One round of the exit limit's exits and its discard, as every guard's (the inlined ways'
    // WrongCallee), then the generic call.
    assert!(
        wrong <= 100,
        "an always-wrong site converges: {wrong} WrongCallee exits"
    );
}

/// The global function `name`'s current chunk.
fn chunk_of(engine: &mut Engine, name: &str) -> std::rc::Rc<crate::bytecode::Chunk> {
    let env = engine.interp.global_env;
    let value = engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("a bound name");
    let crate::value::ValueView::Obj(function) = value.view() else {
        panic!("{name} is not an object")
    };
    let crate::value::ObjectKind::User(user) = &function.borrow().kind else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    runtime
        .execution_code()
        .and_then(Option::as_ref)
        .cloned()
        .expect("compiled")
}

#[test]
fn a_known_callee_whose_code_changes_between_calls_is_entered_at_its_current_code() {
    // The callee's code discarded by its own exits (the tier's code replaced in place by the
    // template's), and taken out of its slot (the entry word zero: the template's call), between
    // two calls of a compiled caller that calls it directly.
    let defs = big("g", "a", "return a + 1;");
    let warm = format!(
        "{defs}\nfunction run(n, k) {{ var r = 0; for (var i = 0; i < n; i++) r = r + g(k === 1 ? 'x' + i : i); return r; }}\n\
         for (var w = 0; w < 80; w++) run(200, 0);\nString(run(100, 0));"
    );
    let steps = [
        "String(run(300, 1));",
        "String(run(300, 0));",
        "String(run(300, 0));",
    ];
    let off: Vec<String> = {
        TestSwitches::default().set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        std::iter::once(eval(&mut engine, &warm))
            .chain(steps.iter().map(|step| eval(&mut engine, step)))
            .collect()
    };
    TestSwitches {
        on: true,
        eager: true,
        inline_graph: Some(true),
        ..TestSwitches::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    known_calls();
    let mut on = vec![eval(&mut engine, &warm)];
    assert!(known_calls() > 0, "run calls g directly");
    // Strings: g's code exits and is discarded; the direct call enters the template's code.
    on.push(eval(&mut engine, steps[0]));
    // g's slot emptied: the entry word zero, the direct call's slow path.
    let g = chunk_of(&mut engine, "g");
    let taken = g.jit.discard();
    // As a discard's caller does: no call way enters the code it took out again.
    crate::bytecode::invalidate_call_caches();
    drop(taken);
    assert_eq!(g.jit.entry_word().map(|w| w.code.get()), Some(0));
    on.push(eval(&mut engine, steps[1]));
    on.push(eval(&mut engine, steps[2]));
    assert_eq!(on, off);
}

#[test]
fn a_site_whose_guard_exited_is_not_called_directly_again() {
    // JSC's exit-site rule: once the guard's WrongCallee flagged the site, its recompiles build
    // the generic call, whatever the record names then.
    let defs = format!(
        "{}{}",
        big("g", "a", "return a + 1;"),
        big("h", "a", "return a + 2;")
    );
    let warm = format!(
        "{defs}var o = {{ g: g }};\nfunction run(n) {{ var r = 0; for (var i = 0; i < n; i++) r += o.g(i); return r; }}\n\
         for (var w = 0; w < 80; w++) run(200);\nString(run(100));"
    );
    let after = "o.g = h; var s = 0; for (var w = 0; w < 80; w++) s += run(200); String(s);";
    let off = {
        TestSwitches::default().set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        [eval(&mut engine, &warm), eval(&mut engine, after)]
    };
    TestSwitches {
        on: true,
        eager: true,
        inline_graph: Some(true),
        ..TestSwitches::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    known_calls();
    let first = eval(&mut engine, &warm);
    assert!(known_calls() > 0);
    exits_for(ExitReason::WrongCallee);
    let second = eval(&mut engine, after);
    assert_eq!([first, second], off);
    assert!(exits_for(ExitReason::WrongCallee) > 0, "the guard exited");
    assert_eq!(
        known_calls(),
        0,
        "a flagged site's recompile calls generically"
    );
}

#[test]
fn a_folded_global_function_is_called_directly_with_no_guard() {
    // F4 part 2's fold of a function held in a global (b5b97df1) proves the callee's identity
    // and its Realm: the called way's guard is not built (`Builder::folded_callee`, the four
    // conditions: the folded word is the callee's, no caller environment to test, no receiver
    // to test, the fold's Realm the guard's).
    let defs = big("g", "a", "return a * 3;");
    let before = super::build::TEST_FOLDED_GUARDS.load(std::sync::atomic::Ordering::Relaxed);
    assert!(agree(&defs, "g(i)") > 0);
    let after = super::build::TEST_FOLDED_GUARDS.load(std::sync::atomic::Ordering::Relaxed);
    assert!(after > before, "the known call's guard was not folded away");
}
