//! A loop that keeps heap facts loads the globals it reads once, at its entry edge, and its header
//! keeps their values (F4 part 1; `Rules::global_source`, `Builder::loop_entry_checks`,
//! `loop_heap_facts`), as Ion's and the DFG's LICM move a global's guard and load out of a loop
//! that writes none (jit/LICM.cpp:154-172; DFGLICMPhase.cpp:388-413). A loop with a call keeps
//! its reads; every result is the VM's, a global reassigned between loops, one that is no typed
//! array at an entry, a script-level lexical and a field the loop stores among them. The entry's
//! checks of a read are the read's own [WrongName]: their exit teaches the read as its own check's
//! would, and the recompile after an epoch move loads the globals at the entry again; a typed
//! receiver's checks are the hoist's [HoistingFailed], whose flag leaves the global loads there.

use super::build::{build, Built};
use super::ir::*;
use super::profile::{self, ChunkProfile};
use super::snapshot::{NameView, Snapshot};
use super::TestSwitches;
use crate::bytecode::{Chunk, Tier};
use crate::value::{ObjectKind, Value, ValueView};
use crate::{Completion, Engine};
use std::rc::Rc;

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

fn agrees_at_every_switch(source: &str) {
    let expected = vm(source);
    for (exit_every, graph) in [(0, false), (0, true), (7, false), (7, true)] {
        TestSwitches {
            on: true,
            eager: exit_every != 0,
            exit_every,
            inline_graph: Some(graph),
            ..Default::default()
        }
        .set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        assert_eq!(
            eval(&mut engine, source),
            expected,
            "exits every {exit_every}, graph {graph}"
        );
    }
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

/// The blocks of `name`'s graph that hold a LoadField of the global object, and the first loop
/// header's block (blocks are in reverse postorder: a loop's entry edge before its header, its
/// body after).
fn global_loads(source: &str, name: &str) -> (Vec<u32>, u32) {
    let (_, built) = vm_graph(source, name);
    loads_and_header(&built.graph)
}

/// The blocks of `graph` that hold a LoadField of the global object, and its first loop header's
/// block.
fn loads_and_header(graph: &Graph) -> (Vec<u32>, u32) {
    let header = graph
        .blocks
        .iter()
        .position(|block| block.loop_header)
        .expect("a loop") as u32;
    let blocks = graph
        .nodes
        .iter()
        .filter(|node| {
            matches!(node.op, NodeOp::LoadField { .. })
                && matches!(graph.node(node.inputs[0]).op, NodeOp::GlobalObject)
        })
        .map(|node| node.block.0)
        .collect();
    (blocks, header)
}

/// `name`'s chunk after `source` ran in the bytecode VM, and its graph.
fn vm_graph(source: &str, name: &str) -> (Rc<Chunk>, Built) {
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, source);
    let chunk = chunk_of(&mut engine, name);
    let built = graph_now(&engine, &chunk);
    (chunk, built)
}

/// `chunk`'s graph built from a snapshot of it now.
fn graph_now(engine: &Engine, chunk: &Chunk) -> Built {
    let snapshot = Snapshot::take_in(chunk, Some(&engine.interp));
    build(chunk, &snapshot).expect("the function builds")
}

/// The first pc of `graph`'s first loop header.
fn header_pc(graph: &Graph) -> u32 {
    graph
        .blocks
        .iter()
        .find(|block| block.loop_header)
        .expect("a loop")
        .first_pc
}

/// The tier on and eager, nothing forced, its first compile at a function's first call.
fn eager_engine() -> Engine {
    TestSwitches {
        on: true,
        eager: true,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

fn exits() -> Vec<(u32, ExitReason)> {
    super::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()))
}

fn discarded() -> Vec<usize> {
    super::TEST_DISCARDED.with(|discarded| std::mem::take(&mut *discarded.borrow_mut()))
}

/// The chunk the tier's code of `name` runs (its second stage, or its first in place), with
/// that code.
fn tier_code(engine: &mut Engine, name: &str) -> Option<(Rc<Chunk>, Rc<crate::jit::JitCode>)> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok()?;
    let object = value.as_obj()?;
    let ObjectKind::User(user) = &object.borrow().kind else {
        return None;
    };
    let runtime = user.func.runtime();
    let chunk = runtime.code2.get().and_then(Option::clone).or_else(|| {
        runtime
            .code
            .get()
            .and_then(Option::clone)
            .filter(|first| super::in_place(first))
    })?;
    let code = chunk
        .jit
        .get()
        .flatten()
        .filter(|code| super::exit::table_bytes(code.mem_ptr()) != 0)?;
    Some((chunk, code))
}

/// The flags the exit profile keeps for `chunk`'s pc `pc` (where a discard of `chunk`'s code
/// puts them, `driver::flag_moved`): (reason, its byte) each.
fn flags(chunk: &Chunk, pc: u32) -> Vec<(u8, u8)> {
    let (at, pc) = match profile::baseline_site(chunk, pc as usize) {
        Some((baseline, pc, _)) => (baseline, pc as u32),
        None => (chunk, pc),
    };
    let mut flags = Vec::new();
    if let Some(profile) = ChunkProfile::get(at) {
        profile.sites_at(pc, |reason, _, byte| flags.push((reason, byte)));
    }
    flags
}

/// A hot loop reading two globals, and two epoch moves after it (a deletion each), then a read.
/// The epoch moves once before the compile too, so that the proof byte the reads' checks expect
/// is not 0 (the realm's environment address is aligned: its byte is the epoch's), which a flag
/// with no proof would also hold.
const EPOCH: [&str; 4] = [
    "globalThis.a = 1; globalThis.b = 2; globalThis.c = 3; globalThis.d = 4;
     globalThis.z = 0; delete globalThis.z;
     function sum(n) { var s = 0; for (var i = 0; i < n; i++) s = s + b + c; return s; }
     let t = 0; for (let k = 0; k < 1000; k++) t += sum(5); t",
    "delete globalThis.a; for (let k = 0; k < 2000; k++) t += sum(5); t",
    "delete globalThis.d; for (let k = 0; k < 2000; k++) t += sum(5); t",
    "t + sum(7) + b * 100 + c",
];

fn epoch_expected() -> Vec<String> {
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    EPOCH
        .iter()
        .map(|source| eval(&mut engine, source))
        .collect()
}

#[test]
fn a_recompile_after_an_epoch_move_loads_the_globals_at_the_entry_again() {
    // The unfolded read's checks (F4 part 1): the fold of its Number globals pinned off.
    super::global_fold::pin_folds_for_test(false);
    // Each deletion moves the global object's epoch: the code exits at its loop's entry, where its
    // reads' checks are, for the reads' reason (WrongName), until its discard; the recompile reads
    // the new epoch and checks it at the entry again, so the second deletion's exits are there too.
    let expected = epoch_expected();
    let mut engine = eager_engine();
    exits();
    discarded();
    assert_eq!(eval(&mut engine, EPOCH[0]), expected[0]);
    let (chunk, _) = tier_code(&mut engine, "sum").expect("the tier compiled sum");
    let header = header_pc(&graph_now(&engine, &chunk).graph);
    assert!(exits().is_empty());
    for round in 1..3 {
        let (_, code) = tier_code(&mut engine, "sum").expect("the tier's code is resident");
        assert_eq!(
            eval(&mut engine, EPOCH[round]),
            expected[round],
            "round {round}"
        );
        let taken = exits();
        assert!(!taken.is_empty(), "round {round}: the epoch moved");
        assert!(
            taken
                .iter()
                .all(|&exit| exit == (header, ExitReason::WrongName)),
            "round {round}: every exit the entry's, WrongName, at the header's pc {header}: \
             {taken:?}"
        );
        assert_eq!(discarded(), vec![code.mem_ptr() as usize], "round {round}");
    }
    tier_code(&mut engine, "sum").expect("the recompile is resident");
    assert_eq!(eval(&mut engine, EPOCH[3]), expected[3]);
    assert!(exits().is_empty(), "the recompile reads the new slots");
    super::release_retired(&mut engine.interp);
}

#[test]
fn an_entry_checks_exit_flags_its_reads_with_the_proof_and_not_the_header() {
    // The unfolded read's checks (F4 part 1): the fold of its Number globals pinned off.
    super::global_fold::pin_folds_for_test(false);
    // The entry's global-object checks prove both of the loop's reads, which run every
    // iteration: after the epoch's move the VM, resumed at the header, ran them (their caches
    // refilled), so the exit flags each with the proof its check expected, as its own exit would
    // (`Rules::stale_name` reads it), and the header not at all.
    let mut engine = eager_engine();
    exits();
    discarded();
    eval(&mut engine, EPOCH[0]);
    let (chunk, _) = tier_code(&mut engine, "sum").expect("the tier compiled sum");
    let snapshot = Snapshot::take_in(&chunk, Some(&engine.interp));
    let header = header_pc(&build(&chunk, &snapshot).expect("sum builds").graph);
    let reads: Vec<(u32, u8)> = (0..snapshot.len())
        .filter_map(|pc| match snapshot.name(pc) {
            view @ NameView::Global { .. } => Some((pc as u32, view.proof())),
            _ => None,
        })
        .collect();
    assert_eq!(reads.len(), 2, "b and c: {reads:?}");
    assert!(reads.iter().all(|&(_, proof)| proof != 0), "{reads:?}");
    eval(&mut engine, EPOCH[1]);
    assert!(
        !discarded().is_empty(),
        "the epoch's move discarded the code"
    );
    assert!(exits()
        .iter()
        .all(|&exit| exit == (header, ExitReason::WrongName)));
    for (k, &(read, proof)) in reads.iter().enumerate() {
        assert_eq!(
            flags(&chunk, read),
            vec![(ExitReason::WrongName as u8, proof)],
            "read {k}, flagged with the proof its check expected"
        );
    }
    assert_eq!(flags(&chunk, header), vec![], "nothing at the header");
    super::release_retired(&mut engine.interp);
}

#[test]
fn a_typed_arrays_decline_leaves_the_globals_at_the_entry() {
    // heap32 is an Array at entries of the loop that run no iteration: its entry checks exit
    // (HoistingFailed) until the discard, which flags the header (the element site is blind);
    // the recompile checks the typed array in the loop, where the site saw typed arrays alone,
    // and still loads the globals once, at the entry.
    let source = format!(
        "{HEAPS}
         function sum(n) {{
             var s = 0;
             for (var i = 0; i < n && i < limit; i++) s = s ^ heap32[i & 63];
             return s;
         }}
         var t = 0;
         for (var r = 0; r < 300; r++) t = (t * 31 + sum(100)) % 1000003;
         var keep = heap32; heap32 = [1, 2, 3];
         for (var r = 0; r < 300; r++) t = (t * 31 + sum(0)) % 1000003;
         heap32 = keep;
         for (var r = 0; r < 300; r++) t = (t * 31 + sum(100)) % 1000003;
         t"
    );
    let expected = vm(&source);
    let mut engine = eager_engine();
    exits();
    discarded();
    assert_eq!(eval(&mut engine, &source), expected);
    let taken = exits();
    assert!(
        taken
            .iter()
            .any(|&(_, why)| why == ExitReason::HoistingFailed),
        "the typed checks exited: {taken:?}"
    );
    assert!(!discarded().is_empty(), "and discarded the code");
    let (chunk, _) = tier_code(&mut engine, "sum").expect("the tier's code");
    let built = graph_now(&engine, &chunk);
    let graph = &built.graph;
    assert!(
        graph
            .declines
            .iter()
            .any(|decline| decline.reason == ExitReason::HoistingFailed),
        "the header's flag declines the typed checks: {:?}",
        graph.declines
    );
    let (loads, header) = loads_and_header(graph);
    assert!(
        !loads.is_empty() && loads.iter().all(|&block| block < header),
        "the global loads at the entry: {loads:?}, header {header}"
    );
    assert!(
        graph
            .nodes
            .iter()
            .any(|node| node.block.0 > header && matches!(node.op, NodeOp::CheckElements(_))),
        "the typed array checked in the loop"
    );
}

/// A loop reading two Number globals and no typed array: its entry holds the reads' checks alone.
const NUMBERS: &str = "var limit = 50, step = 3;
     function cnt(n) { var s = 0; for (var i = 0; i < n && i < limit; i++) s = (s + step) | 0; return s; }
     for (var r = 0; r < 50; r++) cnt(100);";

#[test]
fn the_verifier_refuses_a_misplaced_global_entry_check() {
    let (chunk, built) = vm_graph(NUMBERS, "cnt");
    let graph = &built.graph;
    let header = header_pc(graph);
    let epoch = graph
        .nodes
        .iter()
        .position(|node| {
            matches!(node.op, NodeOp::CheckGlobalEpoch { .. })
                && node.reason == Some(ExitReason::WrongName)
                && node.pc == header
        })
        .expect("the entry's epoch check");
    let rejects = |what: &str, broken: Graph, expected: &str| {
        let error = super::verify::verify(&broken, &chunk, &built.cfg).expect_err(what);
        assert!(error.contains(expected), "{what}: {error}");
    };
    // The hoist's reason: only a typed receiver's checks are the hoist's.
    let mut broken = graph.clone();
    broken.nodes[epoch].reason = Some(ExitReason::HoistingFailed);
    rejects(
        "the hoist's reason",
        broken,
        "speculation by CheckGlobalEpoch",
    );
    // Another pc than the header's.
    let mut broken = graph.clone();
    let state = broken.nodes[epoch].state.expect("a state").index();
    broken.states[state].pc = 0;
    rejects("another pc", broken, "not its header's");
    // A check of a value under the read's reason.
    let block = graph.nodes[epoch].block;
    let word = graph
        .nodes
        .iter()
        .position(|node| node.block == block && matches!(node.op, NodeOp::LoadField { .. }))
        .map(|at| NodeIx(at as u32))
        .expect("the entry's load");
    let mut broken = graph.clone();
    let check = broken.new_node(block, NodeOp::CheckObject, &[word], Rep::Cell, header);
    broken.nodes[check.index()].state = broken.nodes[epoch].state;
    broken.nodes[check.index()].reason = Some(ExitReason::WrongName);
    let body = &mut broken.blocks[block.index()].body;
    body.insert(body.len() - 1, check);
    rejects("another check", broken, "speculation by CheckObject");
    // A check that is no speculation after it in its block.
    let mut broken = graph.clone();
    let check = broken.new_node(block, NodeOp::CheckNumber, &[word], Rep::F64, 0);
    broken.nodes[check.index()].state = broken.nodes[epoch].state;
    broken.nodes[check.index()].reason = Some(ExitReason::NotNumber);
    let body = &mut broken.blocks[block.index()].body;
    body.insert(body.len() - 1, check);
    rejects("a check after it", broken, "before CheckNumber");
    super::verify::verify(graph, &chunk, &built.cfg).expect("the graph itself holds");
}

const HEAPS: &str = "
    var heap32 = new Int32Array(64), heap16 = new Int16Array(64), limit = 64;
    for (var k = 0; k < 64; k++) { heap32[k] = k * 7; heap16[k] = k & 15; }
    function f1(x) { return (x + 1) | 0; } function f2(x) { return (x ^ 3) | 0; }
    var table = [f1, f2];";

#[test]
fn an_effect_free_loop_loads_its_globals_at_its_entry() {
    let source = format!(
        "{HEAPS}
         function sum(n) {{
             var s = 0;
             for (var i = 0; i < n; i++) s = s ^ heap32[i & 63] ^ heap16[i & 63];
             return s;
         }}
         for (var r = 0; r < 50; r++) sum(200);"
    );
    let (blocks, header) = global_loads(&source, "sum");
    assert!(!blocks.is_empty(), "the loop's globals are read");
    assert!(
        blocks.iter().all(|&block| block < header),
        "every global load before the loop's header {header}: {blocks:?}"
    );
}

#[test]
fn a_loop_with_a_call_keeps_its_global_reads_in_its_body() {
    let source = format!(
        "{HEAPS}
         function sum(n) {{
             var s = 0;
             for (var i = 0; i < n; i++) {{
                 s = s ^ heap32[i & 63];
                 if ((i & 15) === 15) s = table[i & 1](s);
             }}
             return s;
         }}
         for (var r = 0; r < 50; r++) sum(200);"
    );
    let (blocks, header) = global_loads(&source, "sum");
    assert!(
        blocks.iter().any(|&block| block > header),
        "a global load in the loop's body: {blocks:?}, header {header}"
    );
}

#[test]
fn hoisted_globals_read_as_the_vm_does() {
    // A typed array and a Number bound read in an effect-free loop; the globals reassigned
    // between two runs of it (another array, another bound); a global that is no typed array at
    // one entry (its entry check fails: one exit, then the loop as before); nested loops.
    agrees_at_every_switch(&format!(
        "{HEAPS}
         function sum(n) {{
             var s = 0;
             for (var i = 0; i < n && i < limit; i++) s = s ^ heap32[i & 63] ^ heap16[i & 63];
             return s;
         }}
         function nest(n) {{
             var s = 0;
             for (var a = 0; a < n; a++) for (var b = 0; b < 8; b++) s = (s + heap32[(a + b) & 63]) | 0;
             return s;
         }}
         var t = 0;
         for (var r = 0; r < 300; r++) t = (t * 31 + sum(100) + nest(20)) % 1000003;
         heap32 = new Int32Array(64); for (var k = 0; k < 64; k++) heap32[k] = 1000 - k; limit = 40;
         for (var r = 0; r < 300; r++) t = (t * 31 + sum(100) + nest(20)) % 1000003;
         var keep = heap16; heap16 = [5, 6, 7];
         t = (t * 31 + sum(100)) % 1000003;
         heap16 = keep;
         for (var r = 0; r < 300; r++) t = (t * 31 + sum(100)) % 1000003;
         t"
    ));
}

#[test]
fn a_lexical_global_and_a_stored_field_read_as_the_vm_does() {
    // A script-level `let` typed array read in an effect-free loop; a loop that stores a global
    // property through `this` (sloppy, the global object) and reads it back as a free name.
    agrees_at_every_switch(
        "let lex = new Int32Array(32);
         for (let k = 0; k < 32; k++) lex[k] = k * 3;
         var g = 0;
         function lexsum(n) { var s = 0; for (var i = 0; i < n; i++) s = (s + lex[i & 31]) | 0; return s; }
         function store(n) { var s = 0; for (var i = 0; i < n; i++) { this.g = i; s = (s + g) | 0; } return s; }
         var t = 0;
         for (var r = 0; r < 300; r++) t = (t * 31 + lexsum(100) + store(50)) % 1000003;
         lex = new Int32Array(32);
         for (var r = 0; r < 300; r++) t = (t * 31 + lexsum(100) + store(50)) % 1000003;
         t",
    );
}

#[test]
fn a_global_changed_by_a_call_in_the_loop_reads_its_new_value() {
    // The global is read before the loop (its fact made there) and the loop's call reassigns it:
    // a loop with a call keeps no global's value through its header.
    agrees_at_every_switch(&format!(
        "{HEAPS}
         var other = new Int32Array(64); for (var k = 0; k < 64; k++) other[k] = 5000 + k;
         function swap() {{ var t = heap32; heap32 = other; other = t; return 0; }}
         function run(n) {{
             var s = heap32[1];
             for (var i = 0; i < n; i++) {{
                 s = (s + heap32[i & 63]) | 0;
                 if ((i & 31) === 31) s = (s + swap()) | 0;
             }}
             return s;
         }}
         var t = 0;
         for (var r = 0; r < 300; r++) t = (t * 31 + run(100)) % 1000003;
         t"
    ));
}

#[test]
fn a_loop_whose_back_edge_is_never_built_teaches_its_global_reads_too() {
    // zlib's shape (typed_loop_tests' NO_BACK_EDGE) over a global typed array: the only path to
    // the back edge runs `u[k]`, which the warm-up never reached and the builder cannot take, so
    // the loop's head is no loop header in the graph. The entry's checks of the global read still
    // stand for it (blind), so that their exit flags it, and the verifier admits them there.
    let (chunk, built) = vm_graph(
        "var T = new Int32Array(8);
         function zl(u, n) { var s = 0, i = 0, k = 'x';
           for (;;) { if (i >= n) break; s = (s + (T[i] | 0)) | 0; i = (i + 1) | 0;
             if (i >= n) break; s = (s + (u[k] | 0)) | 0; }
           return s | 0; }
         for (var r = 0; r < 20; r++) zl(T, 1);",
        "zl",
    );
    let graph = &built.graph;
    assert!(graph.blocks.iter().all(|block| !block.loop_header));
    let entry = graph
        .nodes
        .iter()
        .position(|node| {
            let block = &graph.blocks[node.block.index()];
            node.reason == Some(ExitReason::WrongName)
                && matches!(node.op, NodeOp::CheckGlobalEpoch { .. })
                && block.succs.len() == 1
                && graph.blocks[block.succs[0].index()].first_pc == node.pc
        })
        .map(|at| NodeIx(at as u32))
        .expect("the epoch check before the loop's head");
    let moved = graph
        .moved
        .iter()
        .find(|check| check.node == entry)
        .expect("a moved check");
    assert_eq!(moved.sites.len(), 1, "{:?}", moved.sites);
    assert!(moved.sites[0].pc > graph.nodes[entry.index()].pc && !moved.sites[0].anyway);
    super::verify::verify(graph, &chunk, &built.cfg).expect("the graph holds");
}

#[test]
fn a_quiet_loop_keeps_only_the_facts_of_the_globals_it_reads() {
    // `other` is read before and after an effect-free loop that reads `step` alone: the header
    // keeps `step`'s value, not `other`'s, so `other` is loaded again after the loop (the
    // builder's joins and prunes in the loop walk no fact its reads do not take).
    let (_, built) = vm_graph(
        "var other = 5, step = 3;
         function f(n) { var s = other; for (var i = 0; i < n; i++) s = (s + step) | 0; return (s + other) | 0; }
         for (var r = 0; r < 50; r++) f(100);",
        "f",
    );
    let graph = &built.graph;
    let header = graph
        .blocks
        .iter()
        .position(|block| block.loop_header)
        .expect("a loop") as u32;
    let mut loads: Vec<(u16, u32)> = graph
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                graph.node(*node.inputs.first().unwrap_or(&NodeIx(0))).op,
                NodeOp::GlobalObject
            )
        })
        .filter_map(|node| match node.op {
            NodeOp::LoadField { location, .. } => Some((location, node.block.0)),
            _ => None,
        })
        .collect();
    loads.sort_unstable();
    assert_eq!(
        loads.len(),
        3,
        "other before and after, step at the entry: {loads:?}"
    );
    let after: Vec<_> = loads.iter().filter(|&&(_, block)| block > header).collect();
    assert_eq!(
        after.len(),
        1,
        "one load after the loop's header: {loads:?}"
    );
    assert!(
        loads
            .iter()
            .any(|&(location, block)| location == after[0].0 && block < header),
        "the load after the loop is other's, loaded before it too: {loads:?}"
    );
}

/// The always-wrong programs (F4 part 1, condition c: f4-impl/wrong/*.js, smaller): a hot
/// effect-free loop reading `step`, the global object's epoch moved between its runs, ROUNDS
/// times: by a deletion (`del`), by an accessor and back (`acc`), and with the read on a path
/// that ran in the warm-up and never after (`blind`).
const ROUNDS: usize = 24;

fn always_wrong(kind: &str) -> String {
    let warm = "var t = 0; for (var k = 0; k < 300; k++) t = (t + hot(600)) | 0;";
    match kind {
        "del" => format!(
            "var step = 3;
             function hot(n) {{ var s = 0; for (var i = 0; i < n; i++) s = (s + step) | 0; return s; }}
             {warm}
             for (var round = 0; round < {ROUNDS}; round++) {{
               globalThis['p' + round] = round;
               for (var k = 0; k < 200; k++) t = (t + hot(50)) | 0;
               delete globalThis['p' + round];
             }}
             t"
        ),
        "acc" => format!(
            "var step = 3; globalThis.q = 1;
             function hot(n) {{ var s = 0; for (var i = 0; i < n; i++) s = (s + step) | 0; return s; }}
             {warm}
             for (var round = 0; round < {ROUNDS}; round++) {{
               Object.defineProperty(globalThis, 'q', {{ get() {{ return 7; }}, configurable: true }});
               for (var k = 0; k < 100; k++) t = (t + hot(50) + q) | 0;
               Object.defineProperty(globalThis, 'q', {{ value: round, writable: true, configurable: true, enumerable: true }});
               for (var k = 0; k < 100; k++) t = (t + hot(50) + q) | 0;
             }}
             t"
        ),
        _ => format!(
            "var step = 3;
             function hot(n) {{
               var s = 0;
               for (var i = 0; i < n; i++) {{ s = (s + i) | 0; if ((i & 255) === 255) s = (s + step) | 0; }}
               return s;
             }}
             {warm}
             for (var round = 0; round < {ROUNDS}; round++) {{
               globalThis['p' + round] = round;
               for (var k = 0; k < 200; k++) t = (t + hot(50)) | 0;
               delete globalThis['p' + round];
             }}
             t"
        ),
    }
}

/// `source`'s result and its exits, discards and compiles with the tier on (not eager) at the
/// inlining mode `graph`, with F4 part 1's hoist (`hoist`) or without it (the unhoisted form).
fn exits_discards_compiles(
    source: &str,
    graph: bool,
    hoist: bool,
) -> (String, usize, usize, usize) {
    TestSwitches {
        on: true,
        inline_graph: Some(graph),
        ..Default::default()
    }
    .set();
    super::build::set_no_global_hoist(!hoist);
    exits();
    discarded();
    super::TEST_COMPILED.with(|compiled| compiled.borrow_mut().clear());
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    let result = eval(&mut engine, source);
    let counts = (
        exits().len(),
        discarded().len(),
        super::TEST_COMPILED.with(|compiled| compiled.borrow().len()),
    );
    super::build::set_no_global_hoist(false);
    super::release_retired(&mut engine.interp);
    (result, counts.0, counts.1, counts.2)
}

#[test]
fn an_epoch_moved_under_a_read_that_runs_costs_what_the_unhoisted_read_costs() {
    // The unfolded read's checks (F4 part 1): the fold of its Number globals pinned off.
    super::global_fold::pin_folds_for_test(false);
    // del and acc: every move is followed by runs that read `step`, so the hoisted form exits,
    // discards and recompiles exactly as often as the unhoisted one, at the splice and at graph.
    for kind in ["del", "acc"] {
        let source = always_wrong(kind);
        let expected = vm(&source);
        for graph in [false, true] {
            let unhoisted = exits_discards_compiles(&source, graph, false);
            let hoisted = exits_discards_compiles(&source, graph, true);
            assert_eq!(
                unhoisted.0, expected,
                "{kind} graph {graph}: unhoisted result"
            );
            assert_eq!(hoisted.0, expected, "{kind} graph {graph}: hoisted result");
            assert!(
                unhoisted.1 > 0,
                "{kind} graph {graph}: the moves exit: {unhoisted:?}"
            );
            assert_eq!(
                (hoisted.1, hoisted.2, hoisted.3),
                (unhoisted.1, unhoisted.2, unhoisted.3),
                "{kind} graph {graph}: exits, discards, compiles"
            );
        }
    }
}

#[test]
fn an_epoch_moved_under_a_read_that_does_not_run_costs_one_cycle() {
    // The unfolded read's checks (F4 part 1): the fold of its Number globals pinned off.
    super::global_fold::pin_folds_for_test(false);
    // blind: the unhoisted read never runs after the warm-up and never exits; the hoisted one's
    // entry check fails at the first move, the read did not run since the compile, so the
    // discard flags a hoisting failure at the read: one exit-limit cycle and one discard, then
    // nothing over the further moves.
    let source = always_wrong("blind");
    let expected = vm(&source);
    for graph in [false, true] {
        let unhoisted = exits_discards_compiles(&source, graph, false);
        let hoisted = exits_discards_compiles(&source, graph, true);
        assert_eq!(unhoisted.0, expected, "graph {graph}: unhoisted result");
        assert_eq!(hoisted.0, expected, "graph {graph}: hoisted result");
        assert_eq!(
            (unhoisted.1, unhoisted.2),
            (0, 0),
            "graph {graph}: {unhoisted:?}"
        );
        assert_eq!(
            (hoisted.1, hoisted.2),
            (super::EXIT_LIMIT as usize, 1),
            "graph {graph}: one cycle, one discard: {hoisted:?}"
        );
    }
}

#[test]
fn each_read_an_entry_check_proves_is_judged_by_its_own_cache() {
    // The unfolded read's checks (F4 part 1): the fold of its Number globals pinned off.
    super::global_fold::pin_folds_for_test(false);
    // The loop reads `a` every iteration and `b` and `c` on a path that stops running; the
    // entry's epoch check proves all three. After a move the VM ran `a` (its cache refilled) and
    // not `b` or `c`: the discard flags `a` WrongName with the proof and `b` and `c` hoisting
    // failures, each judged by its own cache (judged by another's, a blind read would pass for
    // run whichever read's cache is taken), and the recompile loads `a` at the entry and `b` in
    // the loop.
    let source = "var a = 1, b = 2, c = 5;
         function two(n) {
           var s = 0;
           for (var i = 0; i < n; i++) { s = (s + a) | 0; if ((i & 255) === 255) s = (s + b + c) | 0; }
           return s;
         }
         var t = 0; for (var k = 0; k < 300; k++) t = (t + two(600)) | 0; t";
    let expected = vm(source);
    TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    exits();
    discarded();
    assert_eq!(eval(&mut engine, source), expected);
    let (chunk, _) = tier_code(&mut engine, "two").expect("the tier compiled two");
    let snapshot = Snapshot::take_in(&chunk, Some(&engine.interp));
    let reads: Vec<(u32, u8)> = (0..snapshot.len())
        .filter_map(|pc| match snapshot.name(pc) {
            view @ NameView::Global { .. } => Some((pc as u32, view.proof())),
            _ => None,
        })
        .collect();
    let [(a, proof), (b, _), (c, _)] = reads[..] else {
        panic!("a, b and c: {reads:?}")
    };
    let moves = "globalThis.x = 1; delete globalThis.x;
         for (var k = 0; k < 400; k++) t = (t + two(50)) | 0; t";
    eval(&mut engine, moves);
    assert!(
        !discarded().is_empty(),
        "the epoch's move discarded the code"
    );
    assert_eq!(
        flags(&chunk, a),
        vec![(ExitReason::WrongName as u8, proof)],
        "a ran"
    );
    for (read, name) in [(b, "b"), (c, "c")] {
        assert_eq!(
            flags(&chunk, read),
            vec![(ExitReason::HoistingFailed as u8, 0)],
            "{name} did not run"
        );
    }
    let graph = graph_now(&engine, &chunk).graph;
    let header = graph
        .blocks
        .iter()
        .position(|block| block.loop_header)
        .expect("a loop") as u32;
    let load_block = |location: u16| {
        graph
            .nodes
            .iter()
            .find(|node| {
                matches!(node.op, NodeOp::LoadField { location: l, .. } if l == location)
                    && matches!(graph.node(node.inputs[0]).op, NodeOp::GlobalObject)
            })
            .map(|node| node.block.0)
    };
    let location = |pc: u32| match snapshot.name(pc as usize) {
        NameView::Global { location, .. } => location,
        view => panic!("a global view: {view:?}"),
    };
    assert!(
        load_block(location(a)).is_some_and(|block| block < header),
        "a at the entry"
    );
    assert!(
        load_block(location(b)).is_some_and(|block| block > header),
        "b in the loop"
    );
    super::release_retired(&mut engine.interp);
}

#[test]
fn a_global_read_before_its_store_in_the_loop_reads_the_stored_value() {
    // The loop reads `g` and then stores it through `this` (sloppy: the global object), a lowered
    // store at `g`'s location: the header keeps no fact at that location, so each iteration's
    // read, before the store, takes the value the last iteration stored.
    agrees_at_every_switch(
        "var g = 0;
         function relay(n) { var s = 0; for (var i = 0; i < n; i++) { s = (s * 3 + g) | 0; this.g = i; } return s; }
         var t = 0;
         for (var r = 0; r < 300; r++) t = (t * 31 + relay(40)) % 1000003;
         t",
    );
}
