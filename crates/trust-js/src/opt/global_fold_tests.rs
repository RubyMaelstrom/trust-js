//! Dependencies on global properties (F4 part 2, layer 2; `opt::global_fold`): a fuse's lists of
//! dependent codes and their firing, the code's validity word and its test, the Invalidated exit
//! and its discard, the install's check, `verify::folded_reads` and R-3's limit, driven by the
//! test-only fold of Number globals (`global_fold::TEST_FOLDS`). Every result is the VM's.

use super::build::build;
use super::ir::*;
use super::profile::ChunkProfile;
use super::snapshot::{Raw, Snapshot};
use super::TestSwitches;
use crate::bytecode::{Chunk, Tier};
use crate::object_fuse::{ObjectFuse, PropertyState, TEST_FIRED};
use crate::value::{ObjectKind, Value, ValueView};
use crate::{Completion, Engine};
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

fn vm(program: &str) -> String {
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, program)
}

fn folds(on: bool) {
    super::global_fold::TEST_FOLDS.with(|folds| folds.set(on));
}

fn invalidated_exits() -> usize {
    super::exit::TEST_EXITS.with(|exits| {
        std::mem::take(&mut *exits.borrow_mut())
            .iter()
            .filter(|&&(_, reason)| reason == ExitReason::Invalidated)
            .count()
    })
}

fn take_discarded() -> usize {
    super::TEST_DISCARDED.with(|discarded| std::mem::take(&mut *discarded.borrow_mut()).len())
}

fn take_compiled() -> usize {
    super::TEST_COMPILED.with(|compiled| std::mem::take(&mut *compiled.borrow_mut()).len())
}

fn take_fired() -> (u64, u64) {
    TEST_FIRED.with(|fired| fired.replace((0, 0)))
}

/// What a run of `source` with the tier on (not eager) at inlining mode `graph` counted, the fold
/// on: its result, Invalidated exits, discards, compiles, and fired lists and invalidated codes.
#[derive(Debug, PartialEq, Eq)]
struct Run {
    result: String,
    invalidated: usize,
    discards: usize,
    compiles: usize,
    fired: (u64, u64),
}

fn run(source: &str, graph: bool) -> Run {
    TestSwitches {
        on: true,
        inline_graph: Some(graph),
        ..Default::default()
    }
    .set();
    folds(true);
    invalidated_exits();
    take_discarded();
    take_compiled();
    take_fired();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    let result = eval(&mut engine, source);
    let run = Run {
        result,
        invalidated: invalidated_exits(),
        discards: take_discarded(),
        compiles: take_compiled(),
        fired: take_fired(),
    };
    super::release_retired(&mut engine.interp);
    folds(false);
    run
}

const WARM: &str = "var t = 0; for (var k = 0; k < 300; k++) t = (t + f(600, false)) | 0;";

/// A function folding `K` in a loop, warmed, then `K` changed once and the function run again.
fn changed_once() -> String {
    format!(
        "var K = 5;
         function f(n) {{ var s = 0; for (var i = 0; i < n; i++) s = (s + K) | 0; return s; }}
         {WARM}
         K = 6;
         for (var k = 0; k < 300; k++) t = (t + f(600)) | 0;
         t"
    )
}

#[test]
fn a_folded_global_changed_once_invalidates_its_code_once() {
    let source = changed_once();
    let expected = vm(&source);
    for graph in [false, true] {
        let run = run(&source, graph);
        assert_eq!(run.result, expected, "graph {graph}");
        // One list fired, one code on it; that code exits once at its test and is discarded
        // there; the recompile reads K NotConstant and does not fold it.
        assert_eq!(run.fired, (1, 1), "graph {graph}: {run:?}");
        assert_eq!(
            (run.invalidated, run.discards),
            (1, 1),
            "graph {graph}: {run:?}"
        );
        assert!(run.compiles >= 2, "graph {graph}: recompiled: {run:?}");
    }
}

#[test]
fn a_global_stored_every_iteration_is_never_folded() {
    // e1: K is NotConstant from its second write, before any compile reads it.
    let source = "var K = 0;
         function f(n) { var s = 0; for (var i = 0; i < n; i++) s = (s + K) | 0; return s; }
         var t = 0;
         for (var k = 0; k < 300; k++) { K = k; t = (t + f(600)) | 0; }
         t";
    let expected = vm(source);
    for graph in [false, true] {
        let run = run(source, graph);
        assert_eq!(run.result, expected, "graph {graph}");
        assert_eq!(
            (run.fired, run.invalidated, run.discards),
            ((0, 0), 0, 0),
            "graph {graph}: {run:?}"
        );
    }
}

/// `f` folds `K` and, on its last call, changes it in the middle of its loop through `how`.
fn changed_while_running(how: &str) -> String {
    let (setup, call) = match how {
        "call" => ("function g(c) { if (c) K = 7; }", "g(c)"),
        "getter" => (
            "var o = { c: false, get v() { if (this.c) K = 7; return 1; } };",
            "(o.c = c, o.v)",
        ),
        _ => ("", "(c ? (K = 7) : 0)"),
    };
    format!(
        "var K = 5;
         {setup}
         function f(n, c) {{
           var s = 0;
           for (var i = 0; i < n; i++) {{ s = (s + K) | 0; if (i === 300) {call}; }}
           return s;
         }}
         {WARM}
         t = (t + f(600, true)) | 0;
         t = (t + f(600, false)) | 0;
         t"
    )
}

#[test]
fn a_code_invalidated_while_it_runs_leaves_at_its_next_test() {
    // By a function it calls, by a getter it triggers, by its own store (an effect: a name
    // store to a global is never lowered): each run leaves at the next folded read's test,
    // once, and its code is discarded once.
    for how in ["call", "getter", "own"] {
        let source = changed_while_running(how);
        let expected = vm(&source);
        for graph in [false, true] {
            let run = run(&source, graph);
            assert_eq!(run.result, expected, "{how} graph {graph}");
            assert_eq!(run.fired.0, 1, "{how} graph {graph}: {run:?}");
            assert_eq!(
                (run.invalidated, run.discards),
                (1, 1),
                "{how} graph {graph}: {run:?}"
            );
        }
    }
}

#[test]
fn removing_shadowing_or_redefining_a_folded_global_invalidates_its_code() {
    // A removal (W6), a later script's lexical (W10), an accessor in its place (W4), and the
    // removal of an earlier global, after which K's list moved down with its state, then K's
    // change.
    for change in [
        "delete globalThis.K; globalThis.K = 9;",
        "Object.defineProperty(globalThis, 'K', { get() { return 9; }, configurable: true });",
        "delete globalThis.A; K = 9;",
    ] {
        let source = format!(
            "globalThis.A = 1; globalThis.K = 5;
             function f(n) {{ var s = 0; for (var i = 0; i < n; i++) s = (s + K) | 0; return s; }}
             {WARM}
             {change}
             for (var k = 0; k < 300; k++) t = (t + f(600)) | 0;
             t"
        );
        let expected = vm(&source);
        for graph in [false, true] {
            let run = run(&source, graph);
            assert_eq!(run.result, expected, "{change} graph {graph}");
            assert_eq!(run.fired.0, 1, "{change} graph {graph}: {run:?}");
            assert_eq!(run.invalidated, 1, "{change} graph {graph}: {run:?}");
        }
    }
    // The shadowing script runs as a script of its own.
    for graph in [false, true] {
        let first = format!(
            "globalThis.K = 5;
             function f(n) {{ var s = 0; for (var i = 0; i < n; i++) s = (s + K) | 0; return s; }}
             {WARM} t"
        );
        let second = "let K = 11; for (var k = 0; k < 300; k++) t = (t + f(600)) | 0; t";
        TestSwitches {
            on: false,
            ..Default::default()
        }
        .set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        eval(&mut engine, &first);
        let expected = eval(&mut engine, second);
        TestSwitches {
            on: true,
            inline_graph: Some(graph),
            ..Default::default()
        }
        .set();
        folds(true);
        invalidated_exits();
        take_fired();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        eval(&mut engine, &first);
        assert_eq!(
            eval(&mut engine, second),
            expected,
            "shadowed, graph {graph}"
        );
        assert_eq!(take_fired().0, 1, "shadowed, graph {graph}");
        assert_eq!(invalidated_exits(), 1, "shadowed, graph {graph}");
        folds(false);
    }
}

#[test]
fn a_function_invalidated_at_the_limit_folds_no_global() {
    // R-3: twelve globals, each changed once after the function recompiled with it folded; the
    // function is discarded at an Invalidated exit FOLD_LIMIT times, then folds nothing, and
    // the later changes cost it no exit.
    let names: Vec<String> = (0..12).map(|k| format!("G{k}")).collect();
    let sum = names.join(" + ");
    let defs: String = names
        .iter()
        .map(|name| format!("var {name} = 1; "))
        .collect();
    let rounds: String = names
        .iter()
        .map(|name| {
            format!("{name} = 2; for (var k = 0; k < 3000; k++) t = (t + f(20, false)) | 0; ")
        })
        .collect();
    let source = format!(
        "{defs}
         function f(n) {{ var s = 0; for (var i = 0; i < n; i++) s = (s + {sum}) | 0; return s; }}
         {WARM}
         {rounds}
         t"
    );
    let expected = vm(&source);
    for graph in [false, true] {
        let run = run(&source, graph);
        assert_eq!(run.result, expected, "graph {graph}");
        let limit = usize::from(super::global_fold::FOLD_LIMIT);
        assert_eq!(
            (run.invalidated, run.discards),
            (limit, limit),
            "graph {graph}: {run:?}"
        );
    }
}

#[test]
fn many_globals_each_changed_once_cost_at_most_the_limit() {
    // e4 (layer 2's counts, both forms of R-3's question): 40 globals, each folded by `f` and
    // changed once, one at a time with runs between. Counted: Invalidated exits, discards,
    // compiles; and the template's entries are the warm-ups, which R sets.
    let names: Vec<String> = (0..40).map(|k| format!("H{k}")).collect();
    let sum = names.join(" + ");
    let defs: String = names
        .iter()
        .map(|name| format!("var {name} = 1; "))
        .collect();
    let rounds: String = names
        .iter()
        .map(|name| format!("{name} = 2; for (var k = 0; k < 2000; k++) t = (t + f(8)) | 0; "))
        .collect();
    let source = format!(
        "{defs}
         function f(n) {{ var s = 0; for (var i = 0; i < n; i++) s = (s + {sum}) | 0; return s; }}
         {WARM}
         {rounds}
         t"
    );
    let expected = vm(&source);
    for counts_r in [true, false] {
        super::global_fold::TEST_INVALIDATION_COUNTS_R.with(|form| form.set(counts_r));
        for graph in [false, true] {
            let run = run(&source, graph);
            assert_eq!(run.result, expected, "graph {graph}");
            eprintln!("e4 R counted {counts_r} graph {graph}: {run:?}");
            assert!(
                run.invalidated <= usize::from(super::global_fold::FOLD_LIMIT),
                "{run:?}"
            );
        }
    }
    super::global_fold::TEST_INVALIDATION_COUNTS_R.with(|form| form.set(true));
}

fn chunk_of(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    let env = engine.interp.global_env;
    let value: Value = engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("a bound name");
    let ValueView::Obj(function) = value.view() else {
        panic!("{name} is not an object")
    };
    let ObjectKind::User(user) = &function.borrow().kind else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .cloned()
        .unwrap_or_else(|| panic!("{name} stayed interpreted"))
}

#[test]
fn the_verifier_refuses_a_folded_read_after_a_call_without_its_test() {
    // f reads K, calls h, reads K again: two tests of the code's word, one per effect epoch.
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(
        &mut engine,
        "var K = 5; function h() {} function f() { var a = K; h(); return a + K; }
         for (var k = 0; k < 50; k++) f();",
    );
    let chunk = chunk_of(&mut engine, "f");
    folds(true);
    let mut raw = Raw::take_in(&chunk, Some(&engine.interp));
    let made = super::global_fold::prepare(&engine.interp, &chunk, std::iter::once(&mut raw));
    folds(false);
    assert!(made.valid().is_some(), "f folds K");
    let snapshot = Snapshot::live(&chunk, &raw);
    let mut graph = build(&chunk, &snapshot).expect("f builds").graph;
    let tests: Vec<usize> = (0..graph.nodes.len())
        .filter(|&k| graph.nodes[k].op == NodeOp::CheckCodeValid)
        .collect();
    let folded = graph
        .nodes
        .iter()
        .filter(|node| matches!(node.op, NodeOp::FoldedGlobal { .. }))
        .count();
    assert_eq!((tests.len(), folded), (2, 2), "a test per effect epoch");
    // h's read, before the call, keeps its epoch check; K's two reads take none.
    assert_eq!(
        graph
            .nodes
            .iter()
            .filter(|node| matches!(node.op, NodeOp::CheckGlobalEpoch { .. }))
            .count(),
        1,
        "no epoch check for a folded read"
    );
    assert_eq!(super::verify::verify_folded_reads(&graph), Ok(()));
    // The test after the call taken out: refused.
    graph.nodes[tests[1]].op = NodeOp::Poll;
    assert!(super::verify::verify_folded_reads(&graph).is_err());
}

#[test]
fn a_global_changed_between_the_snapshot_and_the_install_drops_the_job() {
    TestSwitches {
        on: true,
        eager: true,
        asynchronous: true,
        ..Default::default()
    }
    .set();
    if !super::asynchronous() {
        eprintln!("one CPU: the tier compiles on the main thread, nothing asynchronous to test");
        return;
    }
    folds(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "var K = 5; function f(x) { var s = 0; for (var i = 0; i < 20; i++) s = s + x + K; return s; }",
    );
    super::TEST_DROPPED.with(|dropped| dropped.borrow_mut().clear());
    let hold = super::compiler::TEST_HOLD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut queued = false;
    for _ in 0..20_000 {
        eval(&mut engine, "f(1)");
        if engine.interp.opt_pending.len() != 0 {
            queued = true;
            break;
        }
    }
    assert!(queued, "a compile of f was queued");
    // K moves after the snapshot.
    eval(&mut engine, "K = 6");
    drop(hold);
    let start = std::time::Instant::now();
    while !engine.interp.poll.install_requested() {
        assert!(start.elapsed() < std::time::Duration::from_secs(60));
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
    assert_eq!(eval(&mut engine, "f(1)"), "140");
    let dropped = super::TEST_DROPPED.with(|dropped| std::mem::take(&mut *dropped.borrow_mut()));
    assert!(
        dropped
            .iter()
            .any(|&(_, reason)| reason == "a folded global changed since its snapshot"),
        "{dropped:?}"
    );
    folds(false);
}

#[test]
fn two_engines_on_one_thread_invalidate_only_their_own_code() {
    let source = format!(
        "var K = 5;
         function f(n) {{ var s = 0; for (var i = 0; i < n; i++) s = (s + K) | 0; return s; }}
         {WARM} t"
    );
    for graph in [false, true] {
        TestSwitches {
            on: true,
            inline_graph: Some(graph),
            ..Default::default()
        }
        .set();
        folds(true);
        let mut a = Engine::new();
        a.set_tier(Tier::Jit);
        let mut b = Engine::new();
        b.set_tier(Tier::Jit);
        eval(&mut a, &source);
        eval(&mut b, &source);
        invalidated_exits();
        take_fired();
        eval(&mut a, "K = 6");
        assert_eq!(take_fired(), (1, 1), "graph {graph}: A's list alone");
        assert_eq!(eval(&mut b, "f(600)"), "3000", "graph {graph}");
        assert_eq!(invalidated_exits(), 0, "graph {graph}: B's code runs on");
        assert_eq!(eval(&mut a, "f(600)"), "3600", "graph {graph}");
        assert_eq!(invalidated_exits(), 1, "graph {graph}: A's code exits");
        folds(false);
    }
}

/// A fuse of its own for the unit tests below, keyed by a fresh address.
fn fuse() -> (Arc<ObjectFuse>, Box<u64>) {
    let key = Box::new(0u64);
    (
        ObjectFuse::new_registered(&*key as *const u64 as usize),
        key,
    )
}

fn word() -> Arc<AtomicU32> {
    Arc::new(AtomicU32::new(0))
}

fn set(word: &Arc<AtomicU32>) -> bool {
    word.load(Ordering::Relaxed) != 0
}

#[test]
fn a_fuses_lists_fire_at_a_change_a_removal_a_shadowing_and_a_whole_change() {
    let (fuse, _key) = fuse();
    let generation = |slot| {
        fuse.try_optimize_constant(slot)
            .expect("Constant")
            .generation
    };
    // Only a Constant property under its generation takes a dependent.
    let early = word();
    assert!(!fuse.add_dependent(3, 0, &early), "Untracked");
    let g3 = generation(3);
    assert!(!fuse.add_dependent(3, g3 + 1, &early), "another generation");
    // A value change fires the list once and clears it.
    let (w1, w2) = (word(), word());
    assert!(fuse.add_dependent(3, g3, &w1) && fuse.add_dependent(3, g3, &w2));
    assert_eq!(fuse.dependents(3), 2);
    take_fired();
    fuse.handle_value_change(3);
    assert!(set(&w1) && set(&w2));
    assert_eq!((fuse.dependents(3), take_fired()), (0, (1, 2)));
    assert!(!fuse.add_dependent(3, g3, &word()), "NotConstant");
    // A removal fires the removed slot's list; a later slot's list moves down with its state.
    let g5 = generation(5);
    let g7 = generation(7);
    let (w5, w7) = (word(), word());
    assert!(fuse.add_dependent(5, g5, &w5) && fuse.add_dependent(7, g7, &w7));
    fuse.handle_remove(5);
    assert!(set(&w5) && !set(&w7));
    assert_eq!(fuse.state(6), PropertyState::Constant);
    assert_eq!(fuse.dependents(6), 1);
    fuse.handle_value_change(6);
    assert!(set(&w7));
    // A shadowing fires.
    let g9 = generation(9);
    let w9 = word();
    assert!(fuse.add_dependent(9, g9, &w9));
    fuse.handle_shadowed(9);
    assert!(set(&w9));
    // A whole change fires every list.
    let g10 = generation(10);
    let g11 = generation(11);
    let (w10, w11) = (word(), word());
    assert!(fuse.add_dependent(10, g10, &w10) && fuse.add_dependent(11, g11, &w11));
    fuse.handle_all_changed();
    assert!(set(&w10) && set(&w11));
}

#[test]
fn a_freed_codes_entry_is_pruned_at_the_lists_next_insertion() {
    let (fuse, _key) = fuse();
    let generation = fuse.try_optimize_constant(2).expect("Constant").generation;
    let freed = word();
    assert!(fuse.add_dependent(2, generation, &freed));
    drop(freed);
    assert_eq!(
        fuse.dependents(2),
        1,
        "a freed code's entry stays until the next insertion"
    );
    let live = word();
    assert!(fuse.add_dependent(2, generation, &live));
    assert_eq!(fuse.dependents(2), 1, "the insertion pruned it");
    take_fired();
    fuse.handle_value_change(2);
    assert_eq!(take_fired(), (1, 1), "only the live code counted");
}

#[test]
fn an_invalidated_exit_flags_no_site() {
    let source = changed_once();
    let run = run(&source, false);
    assert_eq!(run.invalidated, 1);
    // f's first stage keeps no flag for Invalidated at any pc.
    TestSwitches {
        on: true,
        inline_graph: Some(false),
        ..Default::default()
    }
    .set();
    folds(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    eval(&mut engine, &source);
    let chunk = chunk_of(&mut engine, "f");
    let first = super::profile::first_stage(&chunk);
    let mut flagged = Vec::new();
    if let Some(profile) = ChunkProfile::get(first) {
        for pc in 0..chunk.jit_ops().len() as u32 {
            profile.sites_at(pc, |reason, _, _| flagged.push((pc, reason)));
        }
        assert_eq!(profile.invalidations(), 1);
    }
    assert!(
        !flagged
            .iter()
            .any(|&(_, reason)| reason == ExitReason::Invalidated as u8),
        "{flagged:?}"
    );
    folds(false);
}

/// `name`'s graph after `source` ran in the VM, its Number globals folded.
fn folded_graph(source: &str, name: &str) -> Graph {
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, source);
    let chunk = chunk_of(&mut engine, name);
    folds(true);
    let mut raw = Raw::take_in(&chunk, Some(&engine.interp));
    let made = super::global_fold::prepare(&engine.interp, &chunk, std::iter::once(&mut raw));
    folds(false);
    assert!(made.valid().is_some(), "{name} folds");
    let snapshot = Snapshot::live(&chunk, &raw);
    build(&chunk, &snapshot).expect("it builds").graph
}

#[test]
fn the_valid_fact_meets_by_and_and_an_effectful_loops_header_has_none() {
    // A join of a path with a call and one without, either way round, and a loop whose body
    // calls: each folded read after them is tested again.
    for (name, body) in [
        ("f", "function f(c) { var a = K; if (c) h(); return a + K; }"),
        ("f", "function f(c) { var a = K; if (c) {} else h(); return a + K; }"),
        (
            "f",
            "function f(n) { var a = K; var s = 0; for (var i = 0; i < n; i++) { s = (s + K) | 0; h(); } return s + a; }",
        ),
    ] {
        let source = format!(
            "var K = 5; function h() {{}} {body}
             for (var k = 0; k < 60; k++) {{ f(k & 1); f(3); }}"
        );
        let graph = folded_graph(&source, name);
        assert_eq!(super::verify::verify_folded_reads(&graph), Ok(()), "{body}");
        let tests = graph
            .nodes
            .iter()
            .filter(|node| node.op == NodeOp::CheckCodeValid)
            .count();
        assert!(tests >= 2, "{body}: {tests} tests");
    }
}
