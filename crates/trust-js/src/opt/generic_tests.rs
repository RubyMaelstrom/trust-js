//! The second-level Generic flag (P3 stage 3 step L5): an exit whose reason its code's own
//! snapshot already held flagged at its site, with the same data, means the form that flag left
//! there exits for its reason too; the discard then flags the site Generic, and the builder takes
//! the operation's generic form there before any lowering. TypeScript's fn(t1,t2,comboId,ret)
//! pc 72 is the case: a read past the used words of an ordinary object whose index keys are named
//! entries exited OutOfBounds at every recompile, its flag widening the read to the form that
//! exited.

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

/// The exit profile's reasons at `pc` of `chunk` (own and inlined keys alike).
fn flags(chunk: &Chunk, pc: u32) -> Vec<u8> {
    let mut out = Vec::new();
    if let Some(profile) = ChunkProfile::get(chunk) {
        profile.sites_at(pc, |reason, _, _| out.push(reason));
    }
    out
}

fn generic(chunk: &Chunk, pc: u32) -> bool {
    flags(chunk, pc).contains(&(ExitReason::Generic as u8))
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

/// The built graph of global function `name` (its snapshot as it stands), verified.
fn graph_of(engine: &mut Engine, name: &str) -> Graph {
    let chunk = chunk_of(engine, name);
    let snapshot = Snapshot::take(&chunk);
    let built = build(&chunk, &snapshot).unwrap_or_else(|refusal| panic!("{name}: {refusal:?}"));
    super::verify::verify(&built.graph, &chunk, &built.cfg)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    built.graph
}

fn count(graph: &Graph, which: impl Fn(&NodeOp) -> bool) -> usize {
    graph.nodes.iter().filter(|node| which(&node.op)).count()
}

fn generic_nodes_at(graph: &Graph, pc: u32) -> usize {
    graph
        .nodes
        .iter()
        .filter(|node| {
            node.pc == pc
                && node.frame == FrameIx::COMPILED
                && matches!(node.op, NodeOp::Generic { .. } | NodeOp::Effect { .. })
        })
        .count()
}

/// `functions` and a loop of `rounds` rounds calling `call` with `v`, `warm` for the first 1,500
/// (300 in the VM, whose records the sites' facts come from, then the tier's warm-up), `then`
/// after: the engine, the exits after the warm-up (pc, reason) and the discards since the tier
/// came on (the warm-up's among them); the result the VM's. `exit_every`: forced exits.
fn run_warmed(
    functions: &str,
    call: &str,
    warm: &str,
    then: &str,
    graph: bool,
    exit_every: u32,
) -> (Engine, Vec<(u32, ExitReason)>, usize) {
    let rounds = 20000;
    let whole = format!(
        "{functions}
         var t = 0;
         for (var r = 0; r < {rounds}; r++) {{
           var v = r < 1500 ? {warm} : {then};
           t = (t + {call}) % 1000003;
         }}
         t"
    );
    let expected = vm(&whole);
    assert!(!expected.starts_with("throw"), "{expected}");
    TestSwitches {
        on: true,
        exit_every,
        inline_graph: Some(graph),
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
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
    super::TEST_DISCARDED.with(|discarded| discarded.borrow_mut().clear());
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
    super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
    let got = eval(
        &mut engine,
        &format!(
            "for (var r = 1500; r < {rounds}; r++) {{ var v = {then}; t = (t + {call}) % 1000003; }}
             t"
        ),
    );
    assert_eq!(got, expected, "{functions}");
    let exits = super::exit::TEST_EXITS.with(|exits| exits.borrow().clone());
    let discards = super::TEST_DISCARDED.with(|discarded| discarded.borrow().len());
    (engine, exits, discards)
}

/// pc 72's shape: TypeScript's identicalCache, an ordinary object `{}` written at large integer
/// keys (named entries: HAS_FAR) and read by them, here after a warm-up on an empty object, whose
/// reads are past its used words.
const CACHE: &str = "var M = {}; M[734666623] = false; M[1263025917] = true; var E = {};
     function g(o, k) { var x = o[k]; return x === undefined ? 1 : x ? 2 : 3; }";

#[test]
fn a_site_whose_flagged_form_exits_again_goes_generic_after_one_more_discard() {
    // The first code reads in bounds: the empty object's reads exit OutOfBounds in the warm-up,
    // and the discard flags OutOfBounds, which widens the read past the used words (pc 72's
    // first flag). The far-keyed object's reads then exit OutOfBounds from that form too (its
    // index keys are named entries), with the flag the code's snapshot held: the second discard
    // flags the site Generic, and the next recompile reads it through the generic access: no
    // exit there after it.
    for graph in [false, true] {
        let (mut engine, exits, discards) =
            run_warmed(CACHE, "g(v, 734666623)", "E", "M", graph, 0);
        let chunk = chunk_of(&mut engine, "g");
        let at = pc_of(&chunk, |op| {
            matches!(op, Op::GetElem | Op::GetElemLocal(..))
        });
        let here = exits.iter().filter(|&&(pc, _)| pc == at).count();
        let text = format!(
            "graph {graph}: {discards} discards, {here} exits at pc {at}, flags {:?}",
            flags(&chunk, at)
        );
        assert_eq!(discards, 2, "{text}");
        assert!(generic(&chunk, at), "{text}");
        // The second code's exits, its limit (100 << 1).
        assert!(here <= 200, "{text}");
        let recompiled = graph_of(&mut engine, "g");
        assert_eq!(generic_nodes_at(&recompiled, at), 1, "{text}");
        assert_eq!(
            count(&recompiled, |op| *op == NodeOp::LoadElementOrUndefined),
            0
        );
        super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        eval(
            &mut engine,
            "for (var r = 0; r < 3000; r++) t = (t + g(M, 734666623) + g(E, 5)) % 1000003;",
        );
        let after = super::exit::TEST_EXITS.with(|exits| exits.borrow().clone());
        assert!(
            after.iter().all(|&(pc, _)| pc != at),
            "{text}: after the recompile {after:?}"
        );
    }
}

#[test]
fn a_flag_that_changes_the_build_is_left_as_it_is() {
    // An in-bounds read meets a shorter Array: OutOfBounds, whose flag widens the read past the
    // used words, which then reads undefined: one discard, no Generic flag, the widened read.
    for graph in [false, true] {
        let (mut engine, exits, discards) = run_warmed(
            "var L = [1, 2, 3, 4, 5, 6, 7, 8]; var S = [1, 2];
             function g(o, k) { var x = o[k]; return x === undefined ? 1 : x; }",
            "g(v, 5)",
            "L",
            "S",
            graph,
            0,
        );
        let chunk = chunk_of(&mut engine, "g");
        let at = pc_of(&chunk, |op| {
            matches!(op, Op::GetElem | Op::GetElemLocal(..))
        });
        let text = format!(
            "graph {graph}: {discards} discards, exits {exits:?}, flags {:?}",
            flags(&chunk, at)
        );
        assert_eq!(discards, 1, "{text}");
        assert!(!generic(&chunk, at), "{text}");
        assert!(
            flags(&chunk, at).contains(&(ExitReason::OutOfBounds as u8)),
            "{text}"
        );
        assert_eq!(
            count(&graph_of(&mut engine, "g"), |op| *op
                == NodeOp::LoadElementOrUndefined),
            1,
            "{text}"
        );
    }
}

#[test]
fn a_shape_flag_whose_ways_grew_changed_the_build() {
    // WrongShape carries the shapes its check expected: the site's first code checks A, its
    // exits at B flag WrongShape with one way, and the recompile checks A and B; C's exits then
    // carry two ways, more than the flag held, so that flag did change the build (one way, then
    // two): the second discard flags no Generic, and the third code checks all three.
    for graph in [false, true] {
        let (mut engine, _, discards) = run_warmed(
            "var A = { p: 1 }, B = { p: 2, q: 0 }, C = { p: 3, r: 0, s: 0 };
             function g(o, k) { return o.p + k; }",
            "g(v, 1)",
            "A",
            "(r < 8000 ? (r & 1 ? B : A) : r % 3 == 0 ? C : r & 1 ? B : A)",
            graph,
            0,
        );
        let chunk = chunk_of(&mut engine, "g");
        let at = pc_of(&chunk, |op| {
            matches!(op, Op::GetProp(..) | Op::GetPropLocal(..))
        });
        let text = format!(
            "graph {graph}: {discards} discards, flags {:?}",
            flags(&chunk, at)
        );
        assert_eq!(discards, 2, "{text}");
        assert!(
            flags(&chunk, at).contains(&(ExitReason::WrongShape as u8)),
            "{text}"
        );
        assert!(!generic(&chunk, at), "{text}");
        assert_eq!(
            generic_nodes_at(&graph_of(&mut engine, "g"), at),
            0,
            "{text}"
        );
    }
}

#[test]
fn another_reasons_exit_at_a_flagged_site_was_not_held() {
    // A read flagged Hole (a holey Array met where the code read in bounds) then meets a short
    // Array: its OutOfBounds exit is not the flag the code held there, which widens the read
    // past the used words: two discards, no Generic flag.
    for graph in [false, true] {
        let (mut engine, _, discards) = run_warmed(
            "var L = [1, 2, 3, 4, 5, 6, 7, 8], H = [1, 2, 3, 4, 5, , 7, 8], S = [1, 2];
             function g(o, k) { var x = o[k]; return x === undefined ? 1 : x; }",
            "g(v, 5)",
            "L",
            "(r < 8000 ? H : S)",
            graph,
            0,
        );
        let chunk = chunk_of(&mut engine, "g");
        let at = pc_of(&chunk, |op| {
            matches!(op, Op::GetElem | Op::GetElemLocal(..))
        });
        let text = format!(
            "graph {graph}: {discards} discards, flags {:?}",
            flags(&chunk, at)
        );
        assert_eq!(discards, 2, "{text}");
        for reason in [ExitReason::Hole, ExitReason::OutOfBounds] {
            assert!(flags(&chunk, at).contains(&(reason as u8)), "{text}");
        }
        assert!(!generic(&chunk, at), "{text}");
        assert_eq!(
            generic_nodes_at(&graph_of(&mut engine, "g"), at),
            0,
            "{text}"
        );
    }
}

/// `graph`'s code for `name`'s chunk, installed as its native code; the engine at the JIT tier.
fn install(engine: &mut Engine, name: &str, graph: &Graph) -> Rc<crate::jit::JitCode> {
    let chunk = chunk_of(engine, name);
    let alloc = super::regalloc::allocate(graph, &super::regalloc::ARM64).unwrap();
    super::regalloc::tests::assert_consistent(graph, &alloc);
    let cfg = crate::jit_ir::Cfg::build(&chunk).expect("the chunk's stack depths settle");
    let initialized = cfg.initialized_reads(chunk.jit_ops(), true);
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = super::a64::compile_with(
        &chunk,
        graph,
        &cfg,
        initialized,
        &layout,
        &ilayout,
        super::a64::Options::default(),
    )
    .unwrap_or_else(|refusal| panic!("the tier refused {name}: {refusal:?}"));
    chunk
        .jit
        .begin_compile()
        .expect("an untried code slot")
        .commit(Some(code.clone()));
    engine.set_tier(Tier::Jit);
    code
}

#[test]
fn a_moved_checks_held_site_goes_generic_at_its_discard() {
    // A moved check (P3 stage 3 step L1c) stands for its sites: its exit's discard flags each
    // site that runs whenever the loop is entered with the site's own reason, and Generic there
    // too where the code's snapshot held that reason at the site (MovedSite::held). No lowering
    // builds a movable check (CheckNumber, CheckInt32, CheckObject, CheckElements) against its
    // own reason's flag, so the held key is planted in the graph the builder made: `lb`'s
    // condition `i < n`, whose CheckInt32 of `n` moves to the loop's entry, `n` then 8.5. Planted,
    // the condition's site is flagged NotInt32 and Generic; not, NotInt32 alone.
    let functions =
        "function lb(a, n) { var s = 0; for (var i = 0; i < n; i++) s = s + a * i; return s; }";
    let expected = vm(&format!("{functions} lb(1.5, 8.5)"));
    for planted in [false, true] {
        TestSwitches {
            on: false,
            ..Default::default()
        }
        .set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        eval(
            &mut engine,
            &format!("{functions} for (var r = 0; r < 300; r++) lb(1.5, 8);"),
        );
        let chunk = chunk_of(&mut engine, "lb");
        let compare = pc_of(&chunk, |op| matches!(op, Op::Lt));
        let built = build(&chunk, &Snapshot::take(&chunk)).expect("lb builds");
        let mut graph = built.graph;
        let sites: Vec<_> = graph
            .moved
            .iter()
            .flat_map(|check| check.sites.iter().copied())
            .filter(|site| site.pc == compare && site.reason == ExitReason::NotInt32)
            .collect();
        assert_eq!(sites.len(), 1, "n's CheckInt32 moved: {:?}", graph.moved);
        assert!(sites[0].anyway);
        if planted {
            graph.held.insert(
                (FrameIx::COMPILED, compare),
                (1 << ExitReason::NotInt32 as u32, 0),
            );
        }
        TestSwitches {
            on: true,
            exit_limit: 1,
            ..Default::default()
        }
        .set();
        let code = install(&mut engine, "lb", &graph);
        assert_eq!(eval(&mut engine, "lb(1.5, 8.5)"), expected);
        let table = super::exit::table_of(&code).expect("the code is retired, not freed");
        assert!(
            table.discarded.get(),
            "the moved check's exit discarded the code"
        );
        let text = format!("planted {planted}: flags {:?}", flags(&chunk, compare));
        assert!(
            flags(&chunk, compare).contains(&(ExitReason::NotInt32 as u8)),
            "{text}"
        );
        assert_eq!(generic(&chunk, compare), planted, "{text}");
        super::pin_tier_off_for_test();
    }
}

#[test]
fn the_flags_data_decides_whether_it_was_held() {
    // ExitReason::data_unchanged: a shape check's ways, a free name's proof, a truncation's mark.
    assert!(ExitReason::WrongShape.data_unchanged(3, 2));
    assert!(ExitReason::WrongShape.data_unchanged(3, 3));
    assert!(!ExitReason::WrongShape.data_unchanged(2, 3));
    assert!(ExitReason::WrongName.data_unchanged(5, 5));
    assert!(!ExitReason::WrongName.data_unchanged(5, 6));
    assert!(ExitReason::NotNumber.data_unchanged(TRUNCATION_EXIT, TRUNCATION_EXIT));
    assert!(ExitReason::NotNumber.data_unchanged(TRUNCATION_EXIT, 0));
    assert!(!ExitReason::NotNumber.data_unchanged(0, TRUNCATION_EXIT));
    assert!(ExitReason::OutOfBounds.data_unchanged(0, 0));
    // The reasons that never make a site Generic.
    for reason in [
        ExitReason::Throw,
        ExitReason::Unsupported,
        ExitReason::Forced,
        ExitReason::NeverRan,
        ExitReason::HoistingFailed,
        ExitReason::PhiNotNumber,
        ExitReason::PhiNotInt32,
        ExitReason::ObservedBuffer,
        ExitReason::WrongCallee,
        ExitReason::Generic,
    ] {
        assert!(!reason.generic_eligible(), "{reason:?}");
    }
    for reason in [
        ExitReason::NotNumber,
        ExitReason::NotInt32,
        ExitReason::Overflow,
        ExitReason::NotObject,
        ExitReason::WrongShape,
        ExitReason::WrongElements,
        ExitReason::OutOfBounds,
        ExitReason::Hole,
        ExitReason::Detached,
        ExitReason::WrongName,
        ExitReason::NotString,
        ExitReason::NotObjectOrNullish,
    ] {
        assert!(reason.generic_eligible(), "{reason:?}");
    }
}

#[test]
fn an_inlined_callees_stale_site_goes_generic_in_the_callees_key() {
    // pc 72's shape in a callee the caller inlines: the flags land at the callee's chunk (its
    // inlined key at graph, its baseline site at the splice), and the caller's recompile reads
    // the callee's element through the generic access.
    for graph in [false, true] {
        let functions = format!("{CACHE} function f(o, k) {{ return g(o, k) + 1; }}");
        let (mut engine, _, discards) =
            run_warmed(&functions, "f(v, 734666623)", "E", "M", graph, 0);
        let callee = chunk_of(&mut engine, "g");
        let at = pc_of(&callee, |op| {
            matches!(op, Op::GetElem | Op::GetElemLocal(..))
        });
        let text = format!(
            "graph {graph}: {discards} discards, callee flags {:?}",
            flags(&callee, at)
        );
        assert!(generic(&callee, at), "{text}");
        super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        eval(
            &mut engine,
            "for (var r = 0; r < 3000; r++) t = (t + f(M, 734666623)) % 1000003;",
        );
        let after = super::exit::TEST_EXITS.with(|exits| exits.borrow().len());
        assert_eq!(after, 0, "{text}");
    }
}

#[test]
fn forced_exits_flag_no_site_generic() {
    // Forced exits (TRUST_JS_OPT_EXIT_EVERY) count toward no discard's flags: a function whose
    // only exits are forced keeps no flag at all.
    for graph in [false, true] {
        let (mut engine, _, _) = run_warmed(
            "var L = [1, 2, 3, 4, 5, 6, 7, 8];
             function g(o, k) { var x = o[k]; return x + 1; }",
            "g(v, 5)",
            "L",
            "L",
            graph,
            100,
        );
        let chunk = chunk_of(&mut engine, "g");
        let at = pc_of(&chunk, |op| {
            matches!(op, Op::GetElem | Op::GetElemLocal(..))
        });
        assert!(
            !generic(&chunk, at),
            "graph {graph}: {:?}",
            flags(&chunk, at)
        );
    }
}

#[test]
fn the_generic_flag_sends_every_operation_with_a_generic_form_there() {
    // One place honours the flag, before any lowering: an element read, a named read, an
    // arithmetic operation and an equality whose sites are flagged Generic are each built as
    // their generic operation, and the loop holding them is no longer free of effects.
    TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(
        &mut engine,
        "var A = [1, 2, 3, 4], O = { p: 5 };
         function h(a, o, n) { var s = 0; for (var i = 0; i < n; i++) {
           s = s + a[i & 3] + o.p; if (s === 7) s = 1; } return s; }
         for (var r = 0; r < 300; r++) h(A, O, 8);",
    );
    let chunk = chunk_of(&mut engine, "h");
    let elem = pc_of(&chunk, |op| {
        matches!(op, Op::GetElem | Op::GetElemLocal(..))
    });
    let prop = pc_of(&chunk, |op| {
        matches!(op, Op::GetProp(..) | Op::GetPropLocal(..))
    });
    let add = pc_of(&chunk, |op| matches!(op, Op::Add));
    let eq = pc_of(&chunk, |op| matches!(op, Op::StrictEq));
    let before = graph_of(&mut engine, "h");
    for pc in [elem, prop, add, eq] {
        assert_eq!(generic_nodes_at(&before, pc), 0, "pc {pc} lowered before");
        ChunkProfile::of(&chunk).add_site(pc, ExitReason::Generic as u8, false, 0, false);
    }
    let after = graph_of(&mut engine, "h");
    for pc in [elem, prop, add, eq] {
        assert_eq!(
            generic_nodes_at(&after, pc),
            1,
            "pc {pc} generic after its flag"
        );
    }
}
