//! A check moved to a loop's entry edge teaches its own operation (P3 stage 3 step L1c): which
//! replaced checks run whenever their loop is entered (JSC's control equivalence on this CFG,
//! `phis::Selector::runs_whenever_entered`), and the discard's flags of a moved check's exit
//! (`driver::flag_moved`): the operations' own sites where every replaced check runs whenever the
//! loop is entered, HoistingFailed at the header (and those sites) where one is blind.

use super::build::build;
use super::ir::*;
use super::profile::ChunkProfile;
use super::snapshot::Snapshot;
use super::TestSwitches;
use crate::bytecode::{Chunk, Op, Tier};
use crate::value::{ObjectKind, Value, ValueView};
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

/// The global function `name`'s first-stage chunk.
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

/// The pc of `chunk`'s first operation `which` accepts.
fn pc_of(chunk: &Chunk, which: impl Fn(&Op) -> bool) -> u32 {
    chunk
        .jit_ops()
        .iter()
        .position(which)
        .unwrap_or_else(|| panic!("no such operation: {:?}", chunk.jit_ops())) as u32
}

/// The pc of `chunk`'s loop header: the target of its one backward jump.
fn header_pc(chunk: &Chunk) -> u32 {
    let ops = chunk.jit_ops();
    let targets: Vec<u32> = ops
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| match *op {
            Op::Jump(target) | Op::JumpIfFalse(target) if (target as usize) <= pc => Some(target),
            _ => None,
        })
        .collect();
    assert_eq!(targets.len(), 1, "one loop: {ops:?}");
    targets[0]
}

/// The exit profile's reasons at `pc` of `chunk`, with whether each is an inlined key.
fn flags(chunk: &Chunk, pc: u32) -> Vec<(u8, bool)> {
    let mut out = Vec::new();
    if let Some(profile) = ChunkProfile::get(chunk) {
        profile.sites_at(pc, |reason, inlined, _| out.push((reason, inlined)));
    }
    out
}

fn flagged(chunk: &Chunk, pc: u32, reason: ExitReason) -> bool {
    flags(chunk, pc)
        .iter()
        .any(|&(flag, _)| flag == reason as u8)
}

/// The VM's result of `program` (the tier off).
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

/// An engine at the JIT tier with the tier on (compiling on the main thread; `graph`: the
/// graph inliner), and `exit_every` forced exits.
fn tier_engine(graph: bool, exit_every: u32) -> Engine {
    TestSwitches {
        on: true,
        exit_every,
        inline_graph: Some(graph),
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine
}

/// The built graph of global function `name` (its snapshot as it stands), verified.
fn graph_of(engine: &mut Engine, name: &str) -> (Rc<Chunk>, Graph) {
    let chunk = chunk_of(engine, name);
    let snapshot = Snapshot::take(&chunk);
    let built = build(&chunk, &snapshot).unwrap_or_else(|refusal| panic!("{name}: {refusal:?}"));
    super::verify::verify(&built.graph, &chunk, &built.cfg)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    (chunk, built.graph)
}

/// The sites of `graph`'s moved checks of op `which`, each (pc, reason, anyway).
fn moved_sites(graph: &Graph, which: impl Fn(&NodeOp) -> bool) -> Vec<(u32, ExitReason, bool)> {
    graph
        .moved
        .iter()
        .filter(|check| which(&graph.node(check.node).op))
        .flat_map(|check| {
            check
                .sites
                .iter()
                .map(|site| (site.pc, site.reason, site.anyway))
        })
        .collect()
}

// ---- which replaced checks run whenever their loop is entered --------------------------------

const SHAPES: &str = "function g() {}
     function fo(a, n) { var s = 0; g(); for (var i = 0; i < n; i++) s = s + a * i; return s; }
     function dw(a, n) { var s = 0, i = 0; g(); do { s = s + a * i; i++; } while (i < n); return s; }
     function cd(a, n) { var s = 0; g(); for (var i = 0; i < n; i++) { if (i & 1) s = s + a; }
       return s; }
     function rt(a, n) { var s = 0; g(); for (var i = 0; ; i++) { if (i >= n) return s; s = s + a; } }
     function br(a, n) { var s = 0, i = 0; g(); while (true) { if (i >= n) break; s = s + a; i++; }
       return s; }
     for (var r = 0; r < 40; r++) { fo(1.5, 8); dw(1.5, 8); cd(1.5, 8); rt(1.5, 8); br(1.5, 8); }";

#[test]
fn a_replaced_check_runs_whenever_its_loop_is_entered_only_before_the_loops_first_exit() {
    // The loops are not rotated: the header holds the test, so `n`'s checks there run on every
    // entry, and `a`'s in the body behind it do not (the loop may run no iteration), nor behind
    // `if`, a return or a break; a do-while's body runs before its test.
    for graph_inline in [false, true] {
        let mut engine = tier_engine(graph_inline, 0);
        eval(&mut engine, SHAPES);
        for (name, a_anyway) in [
            ("fo", false),
            ("dw", true),
            ("cd", false),
            ("rt", false),
            ("br", false),
        ] {
            let (chunk, graph) = graph_of(&mut engine, name);
            let add_or_mul = |op: &Op| matches!(op, Op::Add | Op::Mul);
            let a_pc = if name == "fo" || name == "dw" {
                pc_of(&chunk, |op| matches!(op, Op::Mul))
            } else {
                pc_of(&chunk, add_or_mul)
            };
            let numbers = moved_sites(&graph, |op| *op == NodeOp::CheckNumber);
            let a_sites: Vec<_> = numbers.iter().filter(|site| site.0 == a_pc).collect();
            assert_eq!(a_sites.len(), 1, "{name}: a's check moved: {numbers:?}");
            assert_eq!(a_sites[0].2, a_anyway, "{name}: {numbers:?}");
            assert_eq!(a_sites[0].1, ExitReason::NotNumber, "{name}");
            // n's comparison: in the header for every loop but the do-while's, whose test comes
            // after the body: in every case on every path from the header before an exit.
            let compare = pc_of(&chunk, |op| matches!(op, Op::Lt | Op::Ge));
            let n_sites: Vec<_> = numbers.iter().filter(|site| site.0 == compare).collect();
            assert_eq!(n_sites.len(), 1, "{name}: n's check moved: {numbers:?}");
            assert!(n_sites[0].2, "{name}: {numbers:?}");
        }
    }
}

#[test]
fn a_typed_arrays_entry_checks_class_each_of_its_sites_by_its_own_block() {
    // Step 0c's checks of `x` stand for both of its reads: the condition's, which runs whenever
    // the loop is entered, and the body's, which does not (blind).
    for graph_inline in [false, true] {
        let mut engine = tier_engine(graph_inline, 0);
        eval(
            &mut engine,
            "var I = new Int32Array([4, 1, 2, 3, 4]);
             function tc(a, x) { var s = 0; for (var i = 0; i < x[0]; i++) s = s + a * x[i + 1];
               return s; }
             for (var r = 0; r < 40; r++) tc(1.5, I);",
        );
        let (chunk, graph) = graph_of(&mut engine, "tc");
        let reads = pcs_of(&chunk, |op| {
            matches!(op, Op::GetElem | Op::GetElemLocal(..))
        });
        assert_eq!(reads.len(), 2, "{:?}", chunk.jit_ops());
        let kinds = moved_sites(&graph, |op| matches!(op, NodeOp::CheckElements(_)));
        assert_eq!(
            kinds,
            [
                (reads[0], ExitReason::WrongElements, true),
                (reads[1], ExitReason::WrongElements, false)
            ],
            "graph {graph_inline}"
        );
    }
}

// ---- the discard's flags ---------------------------------------------------------------------

/// `call` in a loop of 14,000 rounds, `v` the warm-up's value for the first 1,500 and `then`'s
/// after: the invariant changes once the tier's code is in, as a stale record does, and the
/// change runs long enough for a second discard where the first one taught nothing.
fn phased(functions: &str, call: &str, warm: &str, then: &str) -> String {
    format!(
        "{functions}
         var t = 0;
         for (var r = 0; r < 14000; r++) {{
           var v = r < 1500 ? {warm} : {then};
           t = (t + {call}) % 1000003;
         }}
         t"
    )
}

/// `phased`'s program run with its first 300 rounds in the VM (whose records name int32s), the
/// rest of the warm-up with the tier on (which compiles the function on the warm-up's kinds), and
/// then the change: the engine, the exits taken after the warm-up (pc, reason) and the discards;
/// its result the VM's.
fn run_warmed(
    functions: &str,
    call: &str,
    warm: &str,
    then: &str,
    graph: bool,
) -> (Engine, Vec<(u32, ExitReason)>, usize) {
    let expected = vm(&phased(functions, call, warm, then));
    assert!(!expected.starts_with("throw"), "{expected}");
    let mut engine = tier_engine(graph, 0);
    engine.set_tier(Tier::Bytecode);
    eval(
        &mut engine,
        &format!(
            "{functions}
             var t = 0;
             for (var r = 0; r < 300; r++) {{ var v = {warm}; t = (t + {call}) % 1000003; }}"
        ),
    );
    engine.set_tier(Tier::Jit);
    super::TEST_COMPILED.with(|compiled| compiled.borrow_mut().clear());
    eval(
        &mut engine,
        &format!(
            "for (var r = 300; r < 1500; r++) {{ var v = {warm}; t = (t + {call}) % 1000003; }}"
        ),
    );
    assert!(
        super::TEST_COMPILED.with(|compiled| !compiled.borrow().is_empty()),
        "the tier compiled in the warm-up: {functions}"
    );
    super::TEST_DISCARDED.with(|discarded| discarded.borrow_mut().clear());
    super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
    let got = eval(
        &mut engine,
        &format!(
            "for (var r = 1500; r < 14000; r++) {{ var v = {then}; t = (t + {call}) % 1000003; }}
             t"
        ),
    );
    assert_eq!(got, expected, "{functions}");
    let exits = super::exit::TEST_EXITS.with(|exits| exits.borrow().clone());
    let discards = super::TEST_DISCARDED.with(|discarded| discarded.borrow().len());
    (engine, exits, discards)
}

/// The exits of `exits` at `pc` with `reason`.
fn count(exits: &[(u32, ExitReason)], pc: u32, reason: ExitReason) -> usize {
    exits
        .iter()
        .filter(|&&(at, r)| at == pc && r == reason)
        .count()
}

/// The pcs of `chunk`'s operations `which` accepts, in order.
fn pcs_of(chunk: &Chunk, which: impl Fn(&Op) -> bool) -> Vec<u32> {
    chunk
        .jit_ops()
        .iter()
        .enumerate()
        .filter(|(_, op)| which(op))
        .map(|(pc, _)| pc as u32)
        .collect()
}

type Case = (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    fn(&Op) -> bool,
    ExitReason,
);

/// Each kind the selector moves, and step 0c's typed array checks: an invariant the loop's
/// condition checks (its function `lb`, the warm-up's value and the one after, the operation
/// and the reason its check exits with).
const CONVERGE: [Case; 5] = [
    (
        "int32",
        "function lb(a, n) { var s = 0; for (var i = 0; i < n; i++) s = s + a * i; return s; }",
        "8",
        "8.5",
        |op| matches!(op, Op::Lt),
        ExitReason::NotInt32,
    ),
    (
        "number",
        "function lb(a, n) { var s = 0; for (var i = 0; i < n; i++) s = s + a * i; return s; }",
        "8",
        "'8'",
        |op| matches!(op, Op::Lt),
        ExitReason::NotNumber,
    ),
    (
        "object",
        "function lb(a, x) { var s = 0; for (var i = 0; i < x.length; i++) s = s + a * i;
           return s; }",
        "[1, 2, 3, 4, 5, 6, 7, 8]",
        "'abcdefgh'",
        |op| matches!(op, Op::GetProp(..) | Op::GetPropLocal(..)),
        ExitReason::NotObject,
    ),
    (
        "elements",
        "var A = [8, 1, 2], T = new Int32Array([8, 1, 2]);
         function lb(a, x) { var s = 0; for (var i = 0; i < x[0]; i++) s = s + a * i; return s; }",
        "A",
        "T",
        |op| matches!(op, Op::GetElem | Op::GetElemLocal(..)),
        ExitReason::WrongElements,
    ),
    (
        "typed",
        "var I = new Int32Array([8]), F = new Float64Array([8]);
         function lb(a, x) { var s = 0; for (var i = 0; i < x[0]; i++) s = s + a * i; return s; }",
        "I",
        "F",
        |op| matches!(op, Op::GetElem | Op::GetElemLocal(..)),
        ExitReason::WrongElements,
    ),
];

#[test]
fn a_moved_check_of_the_loops_condition_teaches_its_operation_in_one_discard() {
    // The invariant checked in the loop's condition changes kind after the warm-up; the moved
    // check fails at each entry, the discard flags the condition's operation with the check's own
    // reason and not the header, and the recompile takes the operation's other form: no exit at
    // the operation, one discard, and the loop's other invariant (`a`) still checked at the
    // entry. Before step L1c the discard flagged the header (HoistingFailed), the recompile
    // checked the operation in the loop, and its own exits brought a second discard.
    for graph_inline in [false, true] {
        for &(kind, functions, warm, then, operation, reason) in &CONVERGE {
            let (mut engine, exits, discards) =
                run_warmed(functions, "lb(1.5, v)", warm, then, graph_inline);
            let chunk = chunk_of(&mut engine, "lb");
            let header = header_pc(&chunk);
            let at = pc_of(&chunk, operation);
            let moved = count(&exits, header, ExitReason::HoistingFailed);
            let own = exits
                .iter()
                .filter(|&&(pc, r)| pc == at && r != ExitReason::HoistingFailed)
                .count();
            let text = format!(
                "{kind} (graph {graph_inline}): {discards} discards, {moved} moved-check exits, \
                 {own} at pc {at}, flags at the header {:?}, at pc {at} {:?}",
                flags(&chunk, header),
                flags(&chunk, at)
            );
            assert!(moved > 0, "{text}");
            assert_eq!(discards, 1, "{text}");
            assert_eq!(own, 0, "{text}");
            assert!(flagged(&chunk, at, reason), "{text}");
            assert!(
                !flagged(&chunk, header, ExitReason::HoistingFailed),
                "{text}"
            );
            // The hoist is not declined: `a`'s check still moves in the recompile's graph.
            let (_, graph) = graph_of(&mut engine, "lb");
            let mul = pc_of(&chunk, |op| matches!(op, Op::Mul));
            assert!(
                moved_sites(&graph, |op| *op == NodeOp::CheckNumber)
                    .iter()
                    .any(|site| site.0 == mul),
                "{text}"
            );
        }
    }
}

#[test]
fn a_blind_moved_check_keeps_hoisting_failed_and_flags_the_sites_that_run_anyway() {
    // `a` checked only in a conditional block (blind): HoistingFailed at the header, nothing at
    // the operation. In a do-while, `n` checked in a conditional block (blind) and after its join
    // (every iteration): the one moved check stands for both; HoistingFailed, and the join's
    // site, not the conditional one.
    let cases = [
        (
            "blind",
            "function bl(a, n) { var s = 0; for (var i = 0; i < n; i++) { if (i & 1) s = s + a; }
               return s; }",
            "(bl(v, 8) | 0)",
            "1.5",
            "'x'",
        ),
        (
            "mixed",
            "function bl(a, n) { var s = 0, t = 0, i = 0;
               do { if (i & 1) s = s + n; t = t + n; i++; } while (i < 8); return (s + t) | 0; }",
            "bl(0, v)",
            "1.5",
            "'8'",
        ),
    ];
    for graph_inline in [false, true] {
        for (label, functions, call, warm, then) in cases {
            let (mut engine, exits, discards) =
                run_warmed(functions, call, warm, then, graph_inline);
            let chunk = chunk_of(&mut engine, "bl");
            let header = header_pc(&chunk);
            let adds = pcs_of(&chunk, |op| matches!(op, Op::Add));
            let text = format!(
                "{label} (graph {graph_inline}): {discards} discards, exits {:?}, flags at the \
                 header {:?}, the adds {:?}",
                exits.iter().take(8).collect::<Vec<_>>(),
                flags(&chunk, header),
                adds.iter()
                    .map(|&pc| (pc, flags(&chunk, pc)))
                    .collect::<Vec<_>>()
            );
            assert!(
                count(&exits, header, ExitReason::HoistingFailed) > 0,
                "{text}"
            );
            assert!(
                flagged(&chunk, header, ExitReason::HoistingFailed),
                "{text}"
            );
            // The conditional block's add is the first; the join's the second.
            assert!(!flagged(&chunk, adds[0], ExitReason::NotNumber), "{text}");
            if label == "mixed" {
                assert!(flagged(&chunk, adds[1], ExitReason::NotNumber), "{text}");
            }
        }
    }
}

#[test]
fn a_moved_check_that_stands_for_two_sites_flags_both() {
    // Two checks of one value that both run whenever the loop is entered are one node (the first
    // dominates the second, whose operation takes its form), so step 0c's typed checks are where a
    // moved check stands for two such sites: two reads of `x` in the loop's condition.
    for graph_inline in [false, true] {
        let (mut engine, _, discards) = run_warmed(
            "var I = new Int32Array([4, 4]), F = new Float64Array([4, 4]);
             function tw(a, x) { var s = 0; for (var i = 0; i < x[0] + x[1]; i++) s = s + a * i;
               return s; }",
            "tw(1.5, v)",
            "I",
            "F",
            graph_inline,
        );
        let chunk = chunk_of(&mut engine, "tw");
        let reads = pcs_of(&chunk, |op| {
            matches!(op, Op::GetElem | Op::GetElemLocal(..))
        });
        let header = header_pc(&chunk);
        let text = format!(
            "graph {graph_inline}: {discards} discards, flags at the header {:?}, the reads {:?}",
            flags(&chunk, header),
            reads
                .iter()
                .map(|&pc| (pc, flags(&chunk, pc)))
                .collect::<Vec<_>>()
        );
        assert_eq!(reads.len(), 2, "{text}");
        for &read in &reads {
            assert!(flagged(&chunk, read, ExitReason::WrongElements), "{text}");
        }
        assert!(
            !flagged(&chunk, header, ExitReason::HoistingFailed),
            "{text}"
        );
    }
}

#[test]
fn an_inlined_callees_moved_check_flags_the_callees_chunk() {
    // `first`'s read of `x[0]` in `il`'s loop condition: its elements check moved to the loop's
    // entry, at splice in the spliced code (the flag at `first`'s baseline site, a spliced key)
    // and at graph in `first`'s inlined frame (the flag at `first`'s chunk, an inlined key).
    for graph_inline in [false, true] {
        let (mut engine, _, discards) = run_warmed(
            "var A = [8, 1, 2], T = new Int32Array([8, 1, 2]);
             function first(x) { return x[0]; }
             function il(a, x) { var s = 0; for (var i = 0; i < first(x); i++) s = s + a * i;
               return s; }",
            "il(1.5, v)",
            "A",
            "T",
            graph_inline,
        );
        let first = chunk_of(&mut engine, "first");
        let il = chunk_of(&mut engine, "il");
        let read = pc_of(&first, |op| {
            matches!(op, Op::GetElem | Op::GetElemLocal(..))
        });
        let header = header_pc(&il);
        let text = format!(
            "graph {graph_inline}: {discards} discards, first's read {:?}, il's header {:?}",
            flags(&first, read),
            flags(&il, header)
        );
        assert!(
            flags(&first, read)
                .iter()
                .any(|&(reason, inlined)| reason == ExitReason::WrongElements as u8 && inlined),
            "the flag at the callee's inlined (or spliced) key: {text}"
        );
        assert!(!flagged(&il, header, ExitReason::HoistingFailed), "{text}");
    }
}

#[test]
fn forced_exits_at_a_moved_check_flag_nothing() {
    // The forced-exit gate fails checks the moved ones among them; none counts, so no discard
    // flags anything at the header or at the condition's operation.
    let program = "function fe(a, n) { var s = 0; for (var i = 0; i < n; i++) s = s + a * i;
           return s; }
         var t = 0; for (var r = 0; r < 3000; r++) t = (t + fe(1.5, 8)) % 1000003; t";
    let expected = vm(program);
    for graph_inline in [false, true] {
        super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        let mut engine = tier_engine(graph_inline, 2);
        assert_eq!(eval(&mut engine, program), expected);
        let chunk = chunk_of(&mut engine, "fe");
        let header = header_pc(&chunk);
        let compare = pc_of(&chunk, |op| matches!(op, Op::Lt));
        let exits = super::exit::TEST_EXITS.with(|exits| exits.borrow().clone());
        assert!(
            count(&exits, header, ExitReason::Forced) > 0,
            "forced exits at the moved checks: {:?}",
            exits.iter().take(12).collect::<Vec<_>>()
        );
        assert_eq!(flags(&chunk, header), [], "graph {graph_inline}");
        assert_eq!(flags(&chunk, compare), [], "graph {graph_inline}");
    }
}

#[test]
fn a_moved_checks_flag_is_the_operations_own_exits_flag() {
    // The flag the discard sets for a site that runs anyway is the one an exit at that operation
    // sets: the same profile key, so the next snapshot's declined mask (step 0c's fingerprint
    // reads it) is what the operation's own exit leaves.
    let functions =
        "function lb(a, n) { var s = 0; for (var i = 0; i < n; i++) s = s + a * i; return s; }";
    let (mut engine, _, _) = run_warmed(functions, "lb(1.5, v)", "8", "8.5", false);
    let chunk = chunk_of(&mut engine, "lb");
    let compare = pc_of(&chunk, |op| matches!(op, Op::Lt));
    let moved = flags(&chunk, compare);
    // A fresh chunk of the same function, flagged as the compare's own exit flags it.
    let mut fresh = tier_engine(false, 0);
    fresh.set_tier_threshold(0);
    eval(&mut fresh, &format!("{functions} lb(1.5, 8);"));
    let other = chunk_of(&mut fresh, "lb");
    ChunkProfile::of(&other).add_site(compare, ExitReason::NotInt32 as u8, false, 0, false);
    assert_eq!(moved, flags(&other, compare));
    let (a, b) = (Snapshot::take(&chunk), Snapshot::take(&other));
    assert!(a.declines(compare as usize, ExitReason::NotInt32));
    assert_eq!(
        a.declines(compare as usize, ExitReason::NotInt32),
        b.declines(compare as usize, ExitReason::NotInt32)
    );
}
