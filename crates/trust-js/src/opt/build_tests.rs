//! The front end's tests (P3 stage 1): small functions compiled and warmed as the engine runs
//! them, their snapshot taken, their graph built and verified, and its shape asserted.

use super::build::build;
use super::ir::*;
use super::snapshot::{fingerprint, Ran, Snapshot};
use crate::bytecode::{Chunk, Op, Tier};
use crate::feedback::SiteView;
use crate::value::{ObjectKind, Value, ValueView, PACK_EMPTY, PACK_NULL, PACK_UNDEFINED};
use crate::{Completion, Engine};
use std::fmt::Write as _;
use std::rc::Rc;

/// An engine at `tier` that compiles every function at its first call, with the tier's switches
/// off whatever the environment's: the graphs are built from the tier-off pipeline's chunks and
/// feedback, which forced exits would add the VM's records to.
fn engine(tier: Tier) -> Engine {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    super::TestSwitches {
        on: false,
        eager: false,
        exit_every: 0,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// The global function `name`'s runtime: its first stage and second stage.
fn runtime_chunk(engine: &mut Engine, name: &str, second: bool) -> Option<Rc<Chunk>> {
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
    let slot = if second {
        &runtime.code2
    } else {
        &runtime.code
    };
    slot.get().and_then(Option::as_ref).cloned()
}

/// The global function `name`'s compiled chunk.
fn chunk_of(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    runtime_chunk(engine, name, false).unwrap_or_else(|| panic!("{name} stayed interpreted"))
}

/// The global function `name`'s second-stage chunk, driving `drive` until it exists; None where
/// the second stage does not run at this threshold (as feedback_record_tests.rs reads it: the
/// recompile checkpoint reads call-cache feedback, which a concurrent test's invalidation can
/// empty for a while).
fn second_stage(engine: &mut Engine, name: &str, drive: &str) -> Option<Rc<Chunk>> {
    if crate::bytecode::inline_recompile_at() != 100 {
        return None;
    }
    for _ in 0..20 {
        if let Some(chunk) = runtime_chunk(engine, name, true) {
            return Some(chunk);
        }
        eval(engine, drive);
    }
    panic!("{name} never reached its second stage")
}

/// A readable listing of the chunk's operations and the graph, for failure messages.
fn dump(graph: &Graph, chunk: &Chunk) -> String {
    let mut out = String::new();
    for (pc, op) in chunk.jit_ops().iter().enumerate() {
        let _ = writeln!(out, "  {pc:>3}  {op:?}");
    }
    for (index, block) in graph.blocks.iter().enumerate() {
        let _ = writeln!(
            out,
            "b{index} (pc {}{}) preds {:?} succs {:?}",
            block.first_pc,
            if block.loop_header {
                ", loop header"
            } else {
                ""
            },
            block.preds.iter().map(|b| b.0).collect::<Vec<_>>(),
            block.succs.iter().map(|b| b.0).collect::<Vec<_>>(),
        );
        for &ix in block.phis.iter().chain(&block.body) {
            let node = graph.node(ix);
            let inputs: Vec<u32> = node.inputs.iter().map(|input| input.0).collect();
            let _ = write!(
                out,
                "  n{} = {:?} {inputs:?} {:?} @{}",
                ix.0, node.op, node.rep, node.pc
            );
            if let Some(state) = node.state {
                let frame = graph.state(state);
                let _ = write!(
                    out,
                    "  [exit {:?} to pc {} slots {:?} stack {:?} handlers {:?}]",
                    node.reason, frame.pc, frame.slots, frame.stack, frame.handlers
                );
            }
            out.push('\n');
        }
    }
    out
}

/// The verifier on `graph` with `chunk`'s own Cfg, as its build hands it over (a graph a test
/// built or broke by hand has none).
fn verify(graph: &Graph, chunk: &Chunk) -> Result<(), String> {
    let cfg = crate::jit_ir::Cfg::build(chunk).expect("the chunk's stack depths settle");
    super::verify::verify(graph, chunk, &cfg)
}

/// The verified graph of `chunk`.
fn graph_for(chunk: &Chunk, name: &str) -> Graph {
    let snapshot = Snapshot::take(chunk);
    let built = build(chunk, &snapshot)
        .unwrap_or_else(|refusal| panic!("{name}: {refusal:?}\n{:#?}", chunk.jit_ops()));
    let graph = built.graph;
    if let Err(error) = super::verify::verify(&graph, chunk, &built.cfg) {
        panic!("{name}: {error}\n{}", dump(&graph, chunk));
    }
    // Blocks in reverse postorder: the entry first, and only a loop's back edges go backward.
    assert_eq!(graph.entry, Some(BlockIx(0)), "{name}");
    for (index, block) in graph.blocks.iter().enumerate() {
        let backward = block
            .preds
            .iter()
            .filter(|pred| pred.index() >= index)
            .count();
        assert!(
            backward == 0 || block.loop_header,
            "{name}: block {index} has a backward edge\n{}",
            dump(&graph, chunk)
        );
    }
    graph
}

/// The verified graph of the global function `name`'s chunk.
fn graph_of(engine: &mut Engine, name: &str) -> (Rc<Chunk>, Graph) {
    let chunk = chunk_of(engine, name);
    let graph = graph_for(&chunk, name);
    (chunk, graph)
}

fn refusal_of(engine: &mut Engine, name: &str) -> Refusal {
    let chunk = chunk_of(engine, name);
    let snapshot = Snapshot::take(&chunk);
    match build(&chunk, &snapshot) {
        Ok(built) => panic!("{name} was built\n{}", dump(&built.graph, &chunk)),
        Err(refusal) => refusal,
    }
}

/// The nodes `which` accepts.
fn nodes<'g>(graph: &'g Graph, which: impl Fn(&NodeOp) -> bool + 'g) -> Vec<&'g Node> {
    graph.nodes.iter().filter(|node| which(&node.op)).collect()
}

fn count(graph: &Graph, which: impl Fn(&NodeOp) -> bool) -> usize {
    graph.nodes.iter().filter(|node| which(&node.op)).count()
}

fn is_effect(op: &NodeOp) -> bool {
    matches!(op, NodeOp::Effect { .. } | NodeOp::Generic { .. })
}

/// The pc of `name`'s one operation `which` accepts.
fn pc_of(chunk: &Chunk, which: impl Fn(&Op) -> bool) -> usize {
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| which(&chunk.jit_ops()[pc]))
        .collect();
    assert_eq!(pcs.len(), 1, "one such operation: {:?}", chunk.jit_ops());
    pcs[0]
}

/// A build hands over the analyses it made of the chunk (stage 2, step 10), which the verifier
/// and the back end take in place of making their own: the chunk's Cfg, and the local reads its
/// stack analysis proves initialized, as fresh ones say.
#[test]
fn a_build_hands_over_the_chunks_cfg_and_initialized_reads() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function tdz(n) { let s = 0; for (let i = 0; i < n; i++) { const d = i * 2; s = s + d; }
                           return s; }
         for (var k = 0; k < 20; k++) tdz(k);",
    );
    let chunk = chunk_of(&mut engine, "tdz");
    let built = build(&chunk, &Snapshot::take(&chunk)).expect("the tier builds tdz");
    let ops = chunk.jit_ops();
    let cfg = crate::jit_ir::Cfg::build(&chunk).expect("the chunk's stack depths settle");
    let reads = cfg.initialized_reads(ops, true);
    assert_eq!(built.cfg.jit_stack_capacity(), cfg.jit_stack_capacity());
    for pc in 0..ops.len() {
        assert_eq!(
            built.cfg.stack_depth_at(pc),
            cfg.stack_depth_at(pc),
            "pc {pc}"
        );
        assert_eq!(
            built.initialized.contains(pc),
            reads.contains(pc),
            "pc {pc}"
        );
    }
    assert!(
        (0..ops.len()).any(|pc| built.initialized.contains(pc)),
        "some read of a binding is proven initialized: {ops:#?}"
    );
    super::verify::verify(&built.graph, &chunk, &built.cfg).expect("the graph holds");
}

#[test]
fn number_arithmetic_is_f64_behind_number_checks() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function f(a, b) { return a + b * 2; } for (var i = 0; i < 20; i++) f(i, 0.5);",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Mul), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Add), 1, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::ConstF64(2f64.to_bits())),
        1,
        "the constant operand needs no check: {text}"
    );
    // Each check exits before its operation, with the operation's operands on the stack.
    let mul = pc_of(&chunk, |op| matches!(op, Op::Mul));
    let add = pc_of(&chunk, |op| matches!(op, Op::Add));
    let checks = nodes(&graph, |op| *op == NodeOp::CheckNumber);
    assert_eq!(checks.len(), 2, "{text}");
    for (check, (pc, depth)) in checks.iter().zip([(mul, 3), (add, 2)]) {
        let state = graph.state(check.state.expect("a check has a state"));
        assert_eq!(check.reason, Some(ExitReason::NotNumber), "{text}");
        assert_eq!(
            (state.pc as usize, state.stack.len()),
            (pc, depth),
            "{text}"
        );
    }
    assert_eq!(count(&graph, |op| *op == NodeOp::Return), 1, "{text}");
}

#[test]
fn int32_records_choose_int32_arithmetic_with_overflow_exits() {
    // The VM records every execution, so its int32-only records are rule U's int32 hint; the
    // first stage's in-line hits record nothing, which leaves f64.
    let source = "function mul(a, b) { return a * b; } function sub(a, b) { return a - b; }
                  for (var i = 0; i < 20; i++) { mul(i, 3); sub(i, 1); }";
    let mut vm = engine(Tier::Bytecode);
    eval(&mut vm, source);
    for (name, op, operation) in [
        ("mul", NodeOp::I32Mul, Op::Mul),
        ("sub", NodeOp::I32Sub, Op::Sub),
    ] {
        let (chunk, graph) = graph_of(&mut vm, name);
        let text = dump(&graph, &chunk);
        let int = nodes(&graph, |node| *node == op);
        assert_eq!(int.len(), 1, "{name}: {text}");
        assert_eq!(int[0].reason, Some(ExitReason::Overflow), "{name}: {text}");
        let state = graph.state(int[0].state.expect("an int32 operation exits"));
        let pc = pc_of(&chunk, |candidate| *candidate == operation);
        assert_eq!((state.pc as usize, state.stack.len()), (pc, 2), "{text}");
        assert_eq!(count(&graph, |op| *op == NodeOp::CheckInt32), 2, "{text}");
    }
    let mut jit = engine(Tier::Jit);
    eval(&mut jit, source);
    let (chunk, graph) = graph_of(&mut jit, "mul");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Mul), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Mul), 0, "{text}");
}

#[test]
fn bitwise_operators_truncate_their_operands_and_an_unsigned_shift_by_zero_is_an_f64() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function bits(a, b) { return (a | 0) + (b >>> 1) + (a >>> 0); }
         for (var i = 0; i < 20; i++) bits(i, i);",
    );
    let (chunk, graph) = graph_of(&mut engine, "bits");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Or), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32UShr), 1, "{text}");
    // Two proven int32s add as int32s, with an overflow exit.
    let add = nodes(&graph, |op| *op == NodeOp::I32Add);
    assert_eq!(add.len(), 1, "{text}");
    assert_eq!(add[0].reason, Some(ExitReason::Overflow), "{text}");
    // Each word is checked a Number and truncated once, `a`'s for its first use: ToInt32 needs
    // no exact int32.
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckNumber), 2, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::TruncateToInt32),
        2,
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckInt32), 0, "{text}");
    // `>>> 0` can leave the int32 range: its uint32 is an f64, which the last Add takes as one.
    let ushr = nodes(&graph, |op| *op == NodeOp::I32UShrToF64);
    assert_eq!(ushr.len(), 1, "{text}");
    assert_eq!(ushr[0].reason, None, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Add), 1, "{text}");
}

#[test]
fn bitwise_operations_whose_records_saw_oddballs_truncate_them_in_line() {
    // A holey array's read gives undefined (`a.shift() | 0` past a hole), a flag null or a
    // Boolean: as V8's kNumberOrOddball and JSC's NotCellNorBigIntUse, the operation stays
    // int32, each word truncated by TruncateNumberOrOddball, not the generic operation. The VM
    // records every execution (the first stage's in-line paths record nothing).
    let mut engine = engine(Tier::Bytecode);
    eval(
        &mut engine,
        "function or(a) { return a | 0; } function xor(a, b) { return a ^ b; }
         function not(a) { return ~a; } function str(a) { return a | 0; }
         function lit(a) { return (void 0 | a) + (true | 0) + (null ^ a); }
         function num(a) { return a & 7; }
         var odd = [undefined, null, true, false];
         for (var i = 0; i < 20; i++) {
             or(i); or(odd[i & 3]); xor(i, odd[i & 3]); not(i); not(odd[i & 3]);
             str(i); str('7'); lit(i); lit(odd[i & 3]); num(i);
         }",
    );
    for (name, op, words) in [
        ("or", NodeOp::I32Or, 1),
        ("xor", NodeOp::I32Xor, 2),
        ("not", NodeOp::I32Not, 1),
    ] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert_eq!(count(&graph, |node| *node == op), 1, "{name}: {text}");
        let truncations = nodes(&graph, |op| *op == NodeOp::TruncateNumberOrOddball);
        assert_eq!(truncations.len(), words, "{name}: {text}");
        for truncation in &truncations {
            assert_eq!(
                truncation.reason,
                Some(ExitReason::NotNumber),
                "{name}: {text}"
            );
            assert_eq!(truncation.rep, Rep::I32, "{name}: {text}");
        }
        assert_eq!(
            count(&graph, |op| *op == NodeOp::CheckNumber),
            0,
            "{name}: {text}"
        );
        assert_eq!(
            count(&graph, |op| *op == NodeOp::TruncateToInt32),
            0,
            "{name}: {text}"
        );
    }
    // A String is no oddball: the generic operation.
    let (chunk, graph) = graph_of(&mut engine, "str");
    let text = dump(&graph, &chunk);
    let effects = nodes(&graph, is_effect);
    assert_eq!(effects.len(), 1, "{text}");
    let or = pc_of(&chunk, |op| matches!(op, Op::BitOr));
    assert_eq!(effects[0].pc as usize, or, "{text}");
    // Oddball constants fold (`void 0 | a` is a's ToInt32, `true | 0` is 1), and `a` is
    // truncated once for both of its operations.
    let (chunk, graph) = graph_of(&mut engine, "lit");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::TruncateNumberOrOddball),
        1,
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::ConstI32(1)), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Or), 2, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Xor), 1, "{text}");
    // A Number check's NotNumber flag (byte 0) met oddballs, which the record holds: the
    // truncation is built again. Its own exit's (TRUNCATION_EXIT) declines it for the generic
    // operation.
    let (chunk, _) = graph_of(&mut engine, "or");
    let or = pc_of(&chunk, |op| matches!(op, Op::BitOr));
    flag(&chunk, or, ExitReason::NotNumber, false, 0);
    let graph = graph_for(&chunk, "or flagged by a Number check");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| *op == NodeOp::TruncateNumberOrOddball),
        1,
        "{text}"
    );
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert!(declines(&graph).is_empty(), "{text}");
    flag(&chunk, or, ExitReason::NotNumber, false, TRUNCATION_EXIT);
    let graph = graph_for(&chunk, "or flagged by its truncation");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| *op == NodeOp::TruncateNumberOrOddball),
        0,
        "{text}"
    );
    assert_eq!(nodes(&graph, is_effect)[0].pc as usize, or, "{text}");
    assert_eq!(declines(&graph), [(or, ExitReason::NotNumber)], "{text}");
    // A record of Numbers alone declines at a Number check's flag, as before.
    let (chunk, graph) = graph_of(&mut engine, "num");
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckNumber), 1);
    let num = pc_of(&chunk, |op| matches!(op, Op::BitAnd));
    flag(&chunk, num, ExitReason::NotNumber, false, 0);
    let graph = graph_for(&chunk, "num flagged");
    let text = dump(&graph, &chunk);
    assert_eq!(nodes(&graph, is_effect)[0].pc as usize, num, "{text}");
    assert_eq!(declines(&graph), [(num, ExitReason::NotNumber)], "{text}");
}

#[test]
fn an_f64_that_arithmetic_made_is_truncated_for_a_bitwise_operation() {
    // The first stage's in-line hits record no int32 hint, so the products are f64s; stage 1 sent
    // their `&` and `~` to the effect path.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function lo(a, b) { return (a * b) & 0xffff; }
         function inv(a) { return ~(a * 1.5); }
         for (var i = 0; i < 20; i++) { lo(i + 0.5, 3); inv(i); }",
    );
    for (name, op) in [("lo", NodeOp::I32And), ("inv", NodeOp::I32Not)] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        let truncations: Vec<NodeIx> = (0..graph.nodes.len() as u32)
            .map(NodeIx)
            .filter(|&ix| graph.node(ix).op == NodeOp::TruncateToInt32)
            .collect();
        assert_eq!(truncations.len(), 1, "{name}: {text}");
        let product = graph.node(graph.node(truncations[0]).inputs[0]);
        assert_eq!(product.op, NodeOp::F64Mul, "{name}: {text}");
        let result = nodes(&graph, |node| *node == op);
        assert_eq!(result.len(), 1, "{name}: {text}");
        assert!(
            result[0].inputs.contains(&truncations[0]),
            "{name}: the operation takes the truncation: {text}"
        );
    }
}

#[test]
fn a_loop_has_header_phis_with_back_edge_inputs() {
    const SUM: &str = "function sum(n) { var s = 0; for (var i = 0; i < n; i++) s += i; return s; }
         for (var k = 0; k < 20; k++) sum(10);";
    let header_of = |graph: &Graph| -> BlockData {
        let headers: Vec<&BlockData> = graph
            .blocks
            .iter()
            .filter(|block| block.loop_header)
            .collect();
        assert_eq!(headers.len(), 1);
        headers[0].clone()
    };
    // The first stage's in-line hits record nothing: no int32 hint, so `s` and `i` add as f64s,
    // and their phis are F64 (stage 2 step 6), from the constant 0 and the sums.
    let mut jit = engine(Tier::Jit);
    eval(&mut jit, SUM);
    let (chunk, graph) = graph_of(&mut jit, "sum");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let header = header_of(&graph);
    assert_eq!(header.preds.len(), 2, "an entry and a back edge: {text}");
    // The header's poll opens its body and exits, on a throw, with the header's state.
    let poll = graph.node(header.body[0]);
    assert_eq!(poll.op, NodeOp::Poll, "{text}");
    assert_eq!(poll.reason, Some(ExitReason::Throw), "{text}");
    let state = graph.state(poll.state.expect("a poll has a state"));
    assert_eq!(state.pc, header.first_pc, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::Poll), 1, "{text}");
    // `s` and `i` are assigned in the loop; `n` is not, and keeps its word through the header.
    assert_eq!(header.phis.len(), 2, "{text}");
    for &phi in &header.phis {
        let node = graph.node(phi);
        assert_eq!(node.rep, Rep::F64, "{text}");
        assert_eq!(node.inputs.len(), 2, "{text}");
        assert_eq!(graph.node(node.inputs[0]).op, NodeOp::ConstF64(0), "{text}");
        assert_eq!(graph.node(node.inputs[1]).op, NodeOp::F64Add, "{text}");
    }
    assert_eq!(
        count(&graph, |op| *op == NodeOp::F64Cmp(Cond::Lt)),
        1,
        "{text}"
    );
    // Only `n` is checked (in the loop: loop-invariant checks stay where they are); the phis
    // need no check and their sums no boxing, but where `s` is returned.
    let checks = nodes(&graph, |op| *op == NodeOp::CheckNumber);
    assert_eq!(checks.len(), 1, "{text}");
    assert_eq!(
        graph.node(checks[0].inputs[0]).op,
        NodeOp::EntrySlot(0),
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Add), 2, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64ToTagged), 1, "{text}");
    // With the VM's records, whose int32s are the hint, the phis are I32 and the adds int32s
    // with their overflow exits.
    let mut vm = engine(Tier::Bytecode);
    eval(&mut vm, SUM);
    let (chunk, graph) = graph_of(&mut vm, "sum");
    let text = dump(&graph, &chunk);
    let header = header_of(&graph);
    assert_eq!(header.phis.len(), 2, "{text}");
    for &phi in &header.phis {
        let node = graph.node(phi);
        assert_eq!(node.rep, Rep::I32, "{text}");
        assert_eq!(graph.node(node.inputs[0]).op, NodeOp::ConstI32(0), "{text}");
        assert_eq!(graph.node(node.inputs[1]).op, NodeOp::I32Add, "{text}");
    }
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Add), 2, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::I32Cmp(Cond::Lt)),
        1,
        "{text}"
    );
    // `n`'s checks; none of a phi.
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckNumber), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckInt32), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32ToTagged), 1, "{text}");
}

#[test]
fn values_cross_an_edge_into_a_block_of_one_predecessor_as_they_are() {
    // The product is an f64 in the entry block; the branch's arm, whose one predecessor that
    // is, adds to it as it is: no boxing before the branch and no check after it (stage 2 step
    // 6; stage 1 boxed every value at every block's end).
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function f(a, c) { var x = a * 2.5; if (c) { return x + 1; } return x; }
         for (var i = 0; i < 20; i++) { f(i, true); f(i, false); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    let text = dump(&graph, &chunk);
    let mul = graph
        .nodes
        .iter()
        .position(|node| node.op == NodeOp::F64Mul)
        .map(|index| NodeIx(index as u32))
        .unwrap_or_else(|| panic!("{text}"));
    let add = nodes(&graph, |op| *op == NodeOp::F64Add);
    assert_eq!(add.len(), 1, "{text}");
    assert_eq!(add[0].inputs[0], mul, "{text}");
    assert_ne!(add[0].block, graph.node(mul).block, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckNumber), 1, "{text}");
    // Each arm boxes its value where it returns it.
    assert_eq!(count(&graph, |op| *op == NodeOp::F64ToTagged), 2, "{text}");
    let entry = graph.block(graph.entry.expect("an entry"));
    assert!(
        entry
            .body
            .iter()
            .all(|&node| graph.node(node).op != NodeOp::F64ToTagged),
        "{text}"
    );
}

#[test]
fn a_loop_at_the_entry_gets_an_entry_block() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function down(n) { do { n--; } while (n > 0); return n; }
         for (var k = 0; k < 20; k++) down(5);",
    );
    let (chunk, graph) = graph_of(&mut engine, "down");
    let text = dump(&graph, &chunk);
    let entry = graph.block(graph.entry.expect("an entry"));
    assert!(entry.preds.is_empty(), "{text}");
    // The entry block holds the hoisted constants and the phis' entry inputs, the check of the
    // parameter `n` entering the loop's F64 phi (stage 2 step 6), which exits at the loop's pc
    // 0, and jumps.
    let exits: Vec<&Node> = entry
        .body
        .iter()
        .map(|&node| graph.node(node))
        .filter(|node| node.state.is_some())
        .collect();
    assert_eq!(exits.len(), 1, "{text}");
    assert_eq!(exits[0].op, NodeOp::CheckNumber, "{text}");
    assert_eq!(exits[0].reason, Some(ExitReason::PhiNotNumber), "{text}");
    assert_eq!(
        graph.state(exits[0].state.expect("a state")).pc,
        0,
        "{text}"
    );
    let last = *entry.body.last().expect("a terminator");
    let NodeOp::Jump(header) = graph.node(last).op else {
        panic!("the entry jumps to the loop: {text}")
    };
    assert!(graph.block(header).loop_header, "{text}");
    assert_eq!(graph.block(header).first_pc, 0, "{text}");
    assert_eq!(graph.block(header).preds.len(), 2, "{text}");
}

#[test]
fn monomorphic_loads_check_the_shape_and_load_the_field() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function getx(o) { return o.x; } var p = {x: 1};
         for (var i = 0; i < 20; i++) getx(p);",
    );
    let (chunk, graph) = graph_of(&mut engine, "getx");
    let text = dump(&graph, &chunk);
    let pc = pc_of(&chunk, |op| matches!(op, Op::GetPropLocal(..)));
    let snapshot = Snapshot::take(&chunk);
    let way = *snapshot
        .named(pc)
        .expect("a named view")
        .filled()
        .next()
        .expect("a filled way");
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let checks = nodes(&graph, |op| matches!(op, NodeOp::CheckShape(_)));
    assert_eq!(checks.len(), 1, "{text}");
    assert_eq!(
        checks[0].op,
        NodeOp::CheckShape(vec![way.receiver_shape]),
        "{text}"
    );
    assert_eq!(checks[0].reason, Some(ExitReason::WrongShape), "{text}");
    assert_eq!(
        count(&graph, |op| *op
            == NodeOp::LoadField {
                location: way.location,
                slot: way.slot
            }),
        1,
        "{text}"
    );
}

#[test]
fn polymorphic_loads_check_every_shape_that_shares_the_slot() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function same(o) { return o.x; } function other(o) { return o.x; }
         for (var i = 0; i < 20; i++) {
           same({x: 1}); same({x: 2, y: 3}); other({x: 1}); other({y: 1, x: 2});
         }",
    );
    let (chunk, graph) = graph_of(&mut engine, "same");
    let text = dump(&graph, &chunk);
    let checks = nodes(&graph, |op| matches!(op, NodeOp::CheckShape(_)));
    assert_eq!(checks.len(), 1, "{text}");
    assert!(
        matches!(&checks[0].op, NodeOp::CheckShape(shapes) if shapes.len() == 2),
        "{text}"
    );
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    // `x` in another slot of the second shape: a dispatch on the shape (P3 stage 3 step L2), one
    // check of both and a test of one.
    let (chunk, graph) = graph_of(&mut engine, "other");
    let text = dump(&graph, &chunk);
    assert!(
        matches!(&nodes(&graph, |op| matches!(op, NodeOp::CheckShape(_)))[..],
            [check] if matches!(&check.op, NodeOp::CheckShape(shapes) if shapes.len() == 2)),
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::ShapeIs(_))),
        1,
        "{text}"
    );
    assert_eq!(count(&graph, is_effect), 0, "{text}");
}

#[test]
fn stores_to_writable_own_data_store_the_field() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function setx(o, v) { o.x = v; } function keep(o, v) { return o.x = v; }
         function forward(o) { o.x = 1; return o.x; }
         var p = {x: 1};
         for (var i = 0; i < 20; i++) { setx(p, i); keep(p, i); forward(p); }",
    );
    for name in ["setx", "keep"] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert_eq!(
            count(&graph, |op| matches!(op, NodeOp::CheckShape(_))),
            1,
            "{text}"
        );
        assert_eq!(
            count(&graph, |op| matches!(op, NodeOp::StoreField { .. })),
            1,
            "{text}"
        );
    }
    // The stored word is the field's value for the rest of the epoch: no load, no second check.
    let (chunk, graph) = graph_of(&mut engine, "forward");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckShape(_))),
        1,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadField { .. })),
        0,
        "{text}"
    );
}

#[test]
fn proven_shapes_and_loaded_fields_last_until_an_effect() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function twice(o) { return o.x + o.x; }
         function around(o, g) { var a = o.x; g(); return a + o.x; }
         var p = {x: 1}; function nothing() {}
         for (var i = 0; i < 20; i++) { twice(p); around(p, nothing); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "twice");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckShape(_))),
        1,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadField { .. })),
        1,
        "{text}"
    );
    let (chunk, graph) = graph_of(&mut engine, "around");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 1, "the call: {text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckShape(_))),
        2,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadField { .. })),
        2,
        "{text}"
    );
}

#[test]
fn a_block_whose_sites_never_ran_is_an_exit() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function nr(o, c) { if (c) { return o.y + o.z(); } return 1; } var p = {y: 1};
         for (var i = 0; i < 20; i++) nr(p, false);",
    );
    let (chunk, graph) = graph_of(&mut engine, "nr");
    let text = dump(&graph, &chunk);
    let exits = nodes(&graph, |op| *op == NodeOp::Exit(ExitReason::NeverRan));
    assert_eq!(exits.len(), 1, "{text}");
    let first_read = (0..chunk.jit_ops().len())
        .find(|&pc| matches!(chunk.jit_ops()[pc], Op::GetPropLocal(..)))
        .expect("o.y");
    let state = graph.state(exits[0].state.expect("an exit has a state"));
    assert_eq!(
        state.pc as usize, first_read,
        "the block's first pc: {text}"
    );
    assert_eq!(graph.block(exits[0].block).body.len(), 1, "{text}");
    assert_eq!(
        count(&graph, is_effect),
        0,
        "nothing of the block is built: {text}"
    );
}

#[test]
fn effects_flush_dirty_slots_and_read_them_back_from_the_frame() {
    // A canonical-frame effect (the spread, whose helper runs the VM on a copy of the operand
    // stack) flushes the dirty `x` before it and reads it back after it, and `o` too is read back
    // and checked again.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function call(o, g) { var x = 1; x = x + 1; var t = [...g]; return x + o.x + t.length; }
         var p = {x: 1};
         for (var i = 0; i < 20; i++) call(p, []);",
    );
    let (chunk, graph) = graph_of(&mut engine, "call");
    let text = dump(&graph, &chunk);
    let block = graph.block(graph.entry.expect("an entry"));
    let x = slot_named(&chunk, "x");
    let position = |which: &dyn Fn(&NodeOp) -> bool| {
        block
            .body
            .iter()
            .position(|&node| which(&graph.node(node).op))
            .unwrap_or_else(|| panic!("{text}"))
    };
    let effect = position(&|op| matches!(op, NodeOp::Effect { .. }));
    let store = position(&|op| *op == NodeOp::StoreSlot(x));
    let reload = position(&|op| *op == NodeOp::FrameSlot(x));
    assert!(store < effect && effect < reload, "{text}");
    let effect = graph.node(block.body[effect]);
    assert_eq!(effect.reason, Some(ExitReason::Throw), "{text}");
    let state = graph.state(effect.state.expect("an effect has a state"));
    assert!(
        state
            .slots
            .iter()
            .all(|source| matches!(source, SlotSource::Frame | SlotSource::Dead)),
        "the flush left every live slot in the frame: {text}"
    );
    assert_eq!(state.pc, effect.pc, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::FrameSlot(0)), 1, "{text}");
}

#[test]
fn a_generic_node_neither_flushes_nor_reloads_the_slots() {
    // P3 stage 3 step 3a: the call is a generic node: nothing is flushed before it, the dirty `x`
    // read after it is the value from before it, and `o` is its entry value, never reloaded; the
    // node's state holds `x` as that value and `g`, read for the call, Dead.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function call(o, g) { var x = 1; x = x + 1; g(); return x + o.x; }
         var p = {x: 1}; function nothing() {}
         for (var i = 0; i < 20; i++) call(p, nothing);",
    );
    let (chunk, graph) = graph_of(&mut engine, "call");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| matches!(
            op,
            NodeOp::FrameSlot(_) | NodeOp::StoreSlot(_)
        )),
        0,
        "no flush and no reload: {text}"
    );
    let [generic] = nodes(&graph, |op| matches!(op, NodeOp::Generic { .. }))[..] else {
        panic!("one generic node: {text}")
    };
    assert_eq!(generic.reason, Some(ExitReason::Throw), "{text}");
    let state = graph.state(generic.state.expect("a generic node has a state"));
    let (x, g, o) = (
        slot_named(&chunk, "x") as usize,
        slot_named(&chunk, "g") as usize,
        slot_named(&chunk, "o") as usize,
    );
    assert_eq!(state.slots[g], SlotSource::Dead, "{text}");
    assert_eq!(state.slots[o], SlotSource::Frame, "{text}");
    let SlotSource::Value(held) = state.slots[x] else {
        panic!("x is a value at the call: {text}")
    };
    // The return's sum reads that value (x + o.x).
    let add = nodes(&graph, |op| matches!(op, NodeOp::I32Add | NodeOp::F64Add))
        .into_iter()
        .rev()
        .find(|node| node.pc as usize > generic.pc as usize)
        .unwrap_or_else(|| panic!("the return's sum: {text}"));
    let reads_held = add
        .inputs
        .iter()
        .any(|&input| input == held || graph.node(input).inputs.first() == Some(&held));
    assert!(
        reads_held,
        "the sum reads x's value from before the call: {text}"
    );
    entry_value(&graph, &chunk, "o");
}

#[test]
fn a_try_bodys_states_hold_its_handler_and_the_catch_is_not_built() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function h(o) { try { return o.f(); } catch (e) { return 0; } }
         var p = {f() { return 2; }};
         for (var i = 0; i < 20; i++) h(p);",
    );
    let (chunk, graph) = graph_of(&mut engine, "h");
    let text = dump(&graph, &chunk);
    let push = pc_of(&chunk, |op| matches!(op, Op::PushHandler(_)));
    let Op::PushHandler(catch) = chunk.jit_ops()[push] else {
        unreachable!()
    };
    let call = nodes(&graph, is_effect);
    assert_eq!(call.len(), 1, "{text}");
    let state = graph.state(call[0].state.expect("an effect has a state"));
    assert_eq!(&*state.handlers, &[(push as u32, 0)], "{text}");
    assert!(
        graph.nodes.iter().all(|node| node.pc < catch),
        "nothing at or after the catch pad: {text}"
    );
}

#[test]
fn a_for_of_body_holds_its_iterator_record_and_the_generic_step_exits() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function fo(a, g) { var t = 0; for (var x of a) { t = t + g(x); } return t; }
         function id(v) { return v; }
         for (var i = 0; i < 20; i++) fo([1, 2], id);",
    );
    let (chunk, graph) = graph_of(&mut engine, "fo");
    let text = dump(&graph, &chunk);
    // The body's call runs under the record its PushHandler pushed (the abort pad's).
    let push = pc_of(&chunk, |op| matches!(op, Op::PushHandler(_)));
    let call = (0..chunk.jit_ops().len())
        .filter(|&pc| matches!(chunk.jit_ops()[pc], Op::Call(..)))
        .max()
        .expect("g(x)");
    let effect = nodes(&graph, is_effect)
        .into_iter()
        .find(|node| node.pc as usize == call)
        .unwrap_or_else(|| panic!("{text}"));
    let state = graph.state(effect.state.expect("an effect has a state"));
    assert_eq!(state.handlers.len(), 1, "{text}");
    assert_eq!(state.handlers[0].0 as usize, push, "{text}");
    // The step's generic path (a `next` that is not built in) never ran: an exit.
    assert_eq!(
        count(&graph, |op| *op == NodeOp::Exit(ExitReason::NeverRan)),
        1,
        "{text}"
    );
    assert!(graph.blocks.iter().any(|block| block.loop_header), "{text}");
}

#[test]
fn equality_is_typed_from_its_operands() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function eq(a, b) { return a === undefined || b === 3 || a === b; }
         function numbers(a, b) { return (a + 0) !== (b + 0); }
         for (var i = 0; i < 20; i++) { eq(i, i); numbers(i, i); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "eq");
    let text = dump(&graph, &chunk);
    // Undefined and a Number other than 0 are equal only to their own word.
    assert_eq!(count(&graph, |op| *op == NodeOp::TaggedSame), 2, "{text}");
    // Two values nothing is known of: the generic operation.
    assert_eq!(count(&graph, is_effect), 1, "{text}");
    let (chunk, graph) = graph_of(&mut engine, "numbers");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| *op == NodeOp::F64Cmp(Cond::Ne)),
        1,
        "{text}"
    );
    assert_eq!(count(&graph, is_effect), 0, "{text}");
}

#[test]
fn nullish_coalescing_tests_both_words_and_joins_with_a_phi() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function nn(a, b) { return a ?? b; } for (var i = 0; i < 20; i++) nn(null, i);",
    );
    let (chunk, graph) = graph_of(&mut engine, "nn");
    let text = dump(&graph, &chunk);
    let same = nodes(&graph, |op| *op == NodeOp::TaggedSame);
    assert_eq!(same.len(), 2, "{text}");
    let words: Vec<NodeOp> = same
        .iter()
        .map(|node| graph.node(node.inputs[1]).op.clone())
        .collect();
    assert_eq!(
        words,
        [
            NodeOp::ConstTagged(PACK_UNDEFINED),
            NodeOp::ConstTagged(PACK_NULL)
        ],
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::Phi), 1, "{text}");
}

#[test]
fn branches_on_values_take_to_boolean_and_on_comparisons_take_the_comparison() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function t(o, a, b) { if (o) { return a < b; } return !o; }
         for (var i = 0; i < 20; i++) { t({}, i, 2); t(null, i, 2); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "t");
    let text = dump(&graph, &chunk);
    // One ToBoolean of `o`: the `!o` of the other arm reuses it.
    assert_eq!(count(&graph, |op| *op == NodeOp::ToBoolean), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::BoolNot), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckBoolean), 0, "{text}");
}

#[test]
fn a_slot_in_its_temporal_dead_zone_holds_the_marker_in_states_where_it_is_live() {
    // In `tdz`, `y` is in its temporal dead zone at `x + 1`, but the VM, resumed there, writes it
    // before any read: the state leaves it Dead (stage 2 step 6). In `tdz2` a read can find the
    // marker after the check of `c > 1000`: its state holds it.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function tdz(c) { let x = c; { let y = x + 1; x = y; } return x; }
         function tdz2(c) { let x = c; { if (c > 1000) x = y; let y = x + 1; x = y; } return x; }
         for (var i = 0; i < 20; i++) { tdz(i); tdz2(i); }",
    );
    let marked = |graph: &Graph, state: &FrameState| {
        state
            .slots
            .iter()
            .filter(|slot| {
                matches!(slot, SlotSource::Value(value)
                    if graph.node(*value).op == NodeOp::ConstTagged(PACK_EMPTY))
            })
            .count()
    };
    let (chunk, graph) = graph_of(&mut engine, "tdz");
    let text = dump(&graph, &chunk);
    let check = nodes(&graph, |op| *op == NodeOp::CheckNumber);
    assert_eq!(check.len(), 1, "{text}");
    let state = graph.state(check[0].state.expect("a check has a state"));
    assert_eq!(marked(&graph, state), 0, "{text}");
    assert_eq!(
        state.slots[slot_named(&chunk, "y") as usize],
        SlotSource::Dead,
        "{text}"
    );
    let (chunk, graph) = graph_of(&mut engine, "tdz2");
    let text = dump(&graph, &chunk);
    let gt = pc_of(&chunk, |op| matches!(op, Op::Gt));
    let check = nodes(&graph, |op| *op == NodeOp::CheckNumber)
        .into_iter()
        .find(|check| graph.state(check.state.expect("a state")).pc as usize == gt)
        .unwrap_or_else(|| panic!("the check of `c > 1000`: {text}"));
    let state = graph.state(check.state.expect("a check has a state"));
    assert_eq!(
        marked(&graph, state),
        1,
        "`y` before its declaration runs: {text}"
    );
}

#[test]
fn a_tail_call_that_runs_exits_to_the_vm() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function tail(n) { 'use strict'; if (n > 0) return tail(n - 1); return 0; }
         for (var i = 0; i < 20; i++) tail(2);",
    );
    let (chunk, graph) = graph_of(&mut engine, "tail");
    let text = dump(&graph, &chunk);
    let pc = pc_of(&chunk, |op| matches!(op, Op::TailCall(..)));
    let exits = nodes(&graph, |op| *op == NodeOp::Exit(ExitReason::Unsupported));
    assert_eq!(exits.len(), 1, "{text}");
    assert_eq!(exits[0].pc as usize, pc, "{text}");
    let state = graph.state(exits[0].state.expect("an exit has a state"));
    assert_eq!(
        state.stack.len(),
        3,
        "the call's receiver, callee and argument: {text}"
    );
}

#[test]
fn a_second_stage_guards_its_spliced_callee_and_reads_baseline_records() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function add(a, b) { return a + b; }
         function caller(x) { return add(x, 1); }
         function drive(n) { var t = 0; for (var i = 0; i < n; i++) t += caller(i); return t; }
         drive(150);",
    );
    let Some(transformed) = second_stage(&mut engine, "caller", "drive(150)") else {
        return;
    };
    let graph = graph_for(&transformed, "caller's second stage");
    let text = dump(&graph, &transformed);
    let ops = transformed.jit_ops();
    let guard = pc_of(&transformed, |op| matches!(op, Op::InlineGuard(..)));
    let Op::InlineGuard(target, mismatch) = ops[guard] else {
        unreachable!()
    };
    let checks = nodes(&graph, |op| *op == NodeOp::CheckCallee { target });
    assert_eq!(checks.len(), 1, "{text}");
    assert_eq!(checks[0].reason, Some(ExitReason::WrongCallee), "{text}");
    // The callee's word, and the receiver's for a guard of `this`.
    let check_this = transformed.jit_inline_target(target).check_this;
    assert_eq!(
        checks[0].inputs.len(),
        1 + usize::from(check_this),
        "{text}"
    );
    let state = graph.state(checks[0].state.expect("a guard has a state"));
    assert_eq!(state.pc as usize, guard, "{text}");
    let argc = transformed.jit_inline_target(target).argc as usize;
    assert!(state.stack.len() > argc, "the call's operands: {text}");
    // The guard's mismatch (the call it replaces) is not built; the splice's ResetSlots is no
    // effect.
    assert!(graph.nodes.iter().all(|node| node.pc != mismatch), "{text}");
    let reset = pc_of(&transformed, |op| matches!(op, Op::ResetSlots(..)));
    assert!(
        nodes(&graph, is_effect)
            .iter()
            .all(|node| node.pc as usize != reset),
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| *op == NodeOp::F64Add),
        1,
        "the spliced add: {text}"
    );
    // The spliced Add's view is its baseline site's record, through the record map.
    let spliced = pc_of(&transformed, |op| matches!(op, Op::Add));
    let snapshot = Snapshot::take(&transformed);
    let map = transformed
        .feedback_vector()
        .record_map()
        .expect("a second stage has a record map");
    if let Some((baseline, baseline_pc)) = map.target(spliced) {
        assert_eq!(
            snapshot.site(spliced),
            crate::feedback::read(baseline, baseline_pc),
            "{text}"
        );
    }
}

#[test]
fn a_spliced_method_that_reads_this_guards_its_receiver_too() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function get() { return this.v; }
         function caller(o) { return o.get(); }
         var p = {v: 1, get: get};
         function drive(n) { var t = 0; for (var i = 0; i < n; i++) t += caller(p); return t; }
         drive(150);",
    );
    let Some(transformed) = second_stage(&mut engine, "caller", "drive(150)") else {
        return;
    };
    let graph = graph_for(&transformed, "caller's second stage");
    let text = dump(&graph, &transformed);
    let guards: Vec<u32> = transformed
        .jit_ops()
        .iter()
        .filter_map(|op| match *op {
            Op::InlineGuard(target, _) => Some(target),
            _ => None,
        })
        .collect();
    let &target = guards
        .first()
        .unwrap_or_else(|| panic!("get is spliced: {text}"));
    assert!(transformed.jit_inline_target(target).check_this, "{text}");
    let checks = nodes(&graph, |op| *op == NodeOp::CheckCallee { target });
    assert_eq!(checks.len(), 1, "{text}");
    assert_eq!(
        checks[0].inputs.len(),
        2,
        "the callee and the receiver: {text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckObject), 0, "{text}");
}

#[test]
fn stage_1_refuses_what_it_does_not_compile() {
    let mut engine = engine(Tier::Jit);
    let mut big = String::from("function big() { var t = 0;");
    for k in 0..260 {
        let _ = write!(big, " var v{k} = {k}; t += v{k};");
    }
    big.push_str(" return t; }");
    eval(&mut engine, &big);
    eval(
        &mut engine,
        "function* gen() { yield 1; }
         function fin() { try { return 1; } finally { gen(); } }
         function ev(s) { return eval(s); }
         for (var i = 0; i < 20; i++) { gen().next(); fin(); ev('1'); big(); }",
    );
    assert_eq!(refusal_of(&mut engine, "gen").pc, None);
    assert!(refusal_of(&mut engine, "gen")
        .reason
        .starts_with("resumable"));
    let fin = refusal_of(&mut engine, "fin");
    assert!(fin.reason.starts_with("finally"), "{fin:?}");
    assert!(fin.pc.is_some(), "{fin:?}");
    assert!(refusal_of(&mut engine, "ev")
        .reason
        .starts_with("prepared entry"));
    assert!(refusal_of(&mut engine, "big").reason.contains("256 words"));
}

#[test]
fn the_snapshot_reads_evidence_from_named_and_call_sites() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function ev(s, o, g, c) { var n = s.length + o.x; if (c) { g(); o.y; } return n; }
         var p = {x: 1};
         for (var i = 0; i < 20; i++) ev('abc', p, null, false);",
    );
    let chunk = chunk_of(&mut engine, "ev");
    let snapshot = Snapshot::take(&chunk);
    let ops = chunk.jit_ops();
    let at = |which: &dyn Fn(&Op) -> bool| -> Vec<Ran> {
        (0..ops.len())
            .filter(|&pc| which(&ops[pc]))
            .map(|pc| snapshot.ran(pc))
            .collect()
    };
    let reads = at(&|op| matches!(op, Op::GetPropLocal(..) | Op::GetProp(..)));
    // `s.length` on a String fills no way: no evidence either way; `o.x` ran; `o.y` never did.
    assert_eq!(reads, [Ran::Unknown, Ran::Yes, Ran::No], "{ops:?}");
    assert_eq!(at(&|op| matches!(op, Op::Call(..))), [Ran::No], "{ops:?}");
    assert_eq!(at(&|op| matches!(op, Op::Add)), [Ran::Unknown], "{ops:?}");
    assert_eq!(snapshot.len(), ops.len());
    assert!(matches!(
        snapshot.site(pc_of(&chunk, |op| matches!(op, Op::Add))),
        SiteView::Values(_)
    ));
}

#[test]
fn the_fingerprint_holds_while_feedback_settles_and_moves_with_a_new_way() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function getx(o) { return o.x; } var p = {x: 1};
         for (var i = 0; i < 20; i++) getx(p);",
    );
    let chunk = chunk_of(&mut engine, "getx");
    let settled = fingerprint(&chunk);
    eval(&mut engine, "for (var i = 0; i < 20; i++) getx(p);");
    assert_eq!(fingerprint(&chunk), settled, "the same shape again");
    eval(&mut engine, "getx({y: 1, x: 2});");
    assert_ne!(fingerprint(&chunk), settled, "a new receiver shape");
}

#[test]
fn a_second_stages_fingerprint_moves_with_its_baseline_sites_records() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function add(a, b) { return a + b; }
         function caller(x) { return add(x, 1); }
         function drive(n) { var t = 0; for (var i = 0; i < n; i++) t += caller(i); return t; }
         drive(150);",
    );
    let Some(transformed) = second_stage(&mut engine, "caller", "drive(150)") else {
        return;
    };
    let settled = fingerprint(&transformed);
    eval(&mut engine, "drive(150);");
    assert_eq!(fingerprint(&transformed), settled, "the same values again");
    // The spliced Add's record is `add`'s first stage's, which a String miss fills; the second
    // stage's own record never changes.
    eval(&mut engine, "add('a', 'b');");
    assert_ne!(
        fingerprint(&transformed),
        settled,
        "a String at the spliced Add's baseline site"
    );
}

#[test]
fn the_verifier_rejects_broken_graphs() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function getx(o, c) { if (c) return o.x; return 0; } var p = {x: 1};
         for (var i = 0; i < 20; i++) { getx(p, true); getx(p, false); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "getx");
    let entry = graph.entry.expect("an entry");
    let load = graph
        .nodes
        .iter()
        .position(|node| matches!(node.op, NodeOp::LoadField { .. }))
        .expect("o.x");
    let rejects = |what: &str, broken: Graph, expected: &str| {
        let error = verify(&broken, &chunk).expect_err(what);
        assert!(error.contains(expected), "{what}: {error}");
    };
    // A block without its terminator.
    let mut broken = graph.clone();
    let last = *broken.blocks[entry.index()].body.last().expect("a body");
    broken.nodes[last.index()].op = NodeOp::ToBoolean;
    rejects("no terminator", broken, "terminator");
    // A node before its input.
    let mut broken = graph.clone();
    let block = broken.nodes[load].block;
    let body = &mut broken.blocks[block.index()].body;
    let at = body
        .iter()
        .position(|node| node.index() == load)
        .expect("listed");
    body.swap(at - 1, at);
    rejects("a use before its definition", broken, "dominate");
    // A representation the operation does not produce.
    let mut broken = graph.clone();
    broken.nodes[load].rep = Rep::F64;
    rejects("a wrong representation", broken, "->");
    // A check without its state.
    let mut broken = graph.clone();
    let check = broken
        .nodes
        .iter_mut()
        .find(|node| matches!(node.op, NodeOp::CheckShape(_)))
        .expect("the shape check");
    check.state = None;
    rejects("an exit without a state", broken, "frame state");
    // A state whose stack is not its pc's.
    let mut broken = graph.clone();
    let state = broken.states.first_mut().expect("a state");
    let mut stack = state.stack.to_vec();
    stack.push(NodeIx(0));
    state.stack = stack.into_boxed_slice();
    rejects("a state of another depth", broken, "depth");
    // A successor list that is not the terminator's.
    let mut broken = graph.clone();
    broken.blocks[entry.index()].succs.reverse();
    rejects("swapped successors", broken, "successors");
    // A slot marked Dead where the liveness says it is live (stage 2 step 6): the builder marks
    // every other slot so.
    let mut broken = graph.clone();
    let (state, slot) = broken
        .states
        .iter()
        .enumerate()
        .find_map(|(index, state)| {
            let slot = state
                .slots
                .iter()
                .position(|slot| *slot != SlotSource::Dead)?;
            Some((index, slot))
        })
        .expect("a live slot in a state");
    let mut slots = broken.states[state].slots.to_vec();
    slots[slot] = SlotSource::Dead;
    broken.states[state].slots = slots.into_boxed_slice();
    rejects("a live slot marked Dead", broken, "Dead where it is live");
    verify(&graph, &chunk).expect("the graph itself holds");
}

// ---- the exit profile's declines (stage 2, step 3) ----------------------------------------------

/// Flag `reason` at `pc` of `chunk`, as a discard keys an exit there (`opt::profile`): `inlined`
/// for a site a splice took, `ways` the shapes a WrongShape exit's check expected.
fn flag(chunk: &Chunk, pc: usize, reason: ExitReason, inlined: bool, ways: u8) {
    let latest = reason == ExitReason::WrongName;
    super::profile::ChunkProfile::of(chunk).add_site(
        pc as u32,
        reason as u8,
        inlined,
        ways,
        latest,
    );
}

fn declines(graph: &Graph) -> Vec<(usize, ExitReason)> {
    graph
        .declines
        .iter()
        .map(|decline| (decline.pc as usize, decline.reason))
        .collect()
}

#[test]
fn a_flagged_number_check_takes_the_generic_operation() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function f(a, b) { return a + b * 2; } for (var i = 0; i < 20; i++) f(i, 0.5);",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    assert_eq!(count(&graph, is_effect), 0);
    assert!(graph.declines.is_empty());
    let add = pc_of(&chunk, |op| matches!(op, Op::Add));
    flag(&chunk, add, ExitReason::NotNumber, false, 0);
    let graph = graph_for(&chunk, "f flagged");
    let text = dump(&graph, &chunk);
    // The Add, whose operand `a` needs its check, is the generic operation; the Mul keeps its.
    let effects = nodes(&graph, is_effect);
    assert_eq!(effects.len(), 1, "{text}");
    assert_eq!(effects[0].pc as usize, add, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Mul), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Add), 0, "{text}");
    assert_eq!(declines(&graph), [(add, ExitReason::NotNumber)], "{text}");
    assert_eq!(graph.declines[0].form, "the generic operation");
}

#[test]
fn flagged_int32_speculation_takes_f64_arithmetic() {
    // NotInt32: the hint's int32 path (CheckInt32 on each operand) is declined for f64.
    let mut vm = engine(Tier::Bytecode);
    eval(
        &mut vm,
        "function sub(a, b) { return a - b; } function lt(a, b) { return a < b; }
         function inc(a) { a++; return a; }
         for (var i = 0; i < 20; i++) { sub(i, 1); lt(i, 3); inc(i); }",
    );
    let (chunk, graph) = graph_of(&mut vm, "sub");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Sub), 1);
    let sub = pc_of(&chunk, |op| matches!(op, Op::Sub));
    flag(&chunk, sub, ExitReason::NotInt32, false, 0);
    let graph = graph_for(&chunk, "sub flagged");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Sub), 0, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Sub), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckInt32), 0, "{text}");
    assert_eq!(declines(&graph), [(sub, ExitReason::NotInt32)], "{text}");
    // A comparison too.
    let (chunk, graph) = graph_of(&mut vm, "lt");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Cmp(Cond::Lt)), 1);
    let lt = pc_of(&chunk, |op| matches!(op, Op::Lt));
    flag(&chunk, lt, ExitReason::NotInt32, false, 0);
    let graph = graph_for(&chunk, "lt flagged");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| *op == NodeOp::F64Cmp(Cond::Lt)),
        1,
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckInt32), 0, "{text}");
    // Overflow: `++` on its int32 takes f64 too.
    let (chunk, graph) = graph_of(&mut vm, "inc");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Add), 1);
    let inc = pc_of(&chunk, |op| matches!(op, Op::UpdateLocal(..)));
    flag(&chunk, inc, ExitReason::Overflow, false, 0);
    let graph = graph_for(&chunk, "inc flagged");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Add), 0, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Add), 1, "{text}");
    assert_eq!(declines(&graph), [(inc, ExitReason::Overflow)], "{text}");
}

#[test]
fn a_flagged_overflow_takes_f64_even_on_proven_int32s() {
    // JSC's makeSafe: proven int32s, whose add ignores the records, add as f64s.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function f(a, b) { return (a | 0) + (b | 0); } function half(a) { return a >>> 1; }
         for (var i = 0; i < 20; i++) { f(i, i); half(i); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Add), 1);
    let add = pc_of(&chunk, |op| matches!(op, Op::Add));
    flag(&chunk, add, ExitReason::Overflow, false, 0);
    let graph = graph_for(&chunk, "f flagged");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Add), 0, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Add), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32ToF64), 2, "{text}");
    assert_eq!(declines(&graph), [(add, ExitReason::Overflow)], "{text}");
    // `>>> 1`'s uint32 as an f64, without its exit.
    let (chunk, graph) = graph_of(&mut engine, "half");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32UShr), 1);
    let ushr = pc_of(&chunk, |op| matches!(op, Op::UShr));
    flag(&chunk, ushr, ExitReason::Overflow, false, 0);
    let graph = graph_for(&chunk, "half flagged");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, |op| *op == NodeOp::I32UShr), 0, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32UShrToF64), 1, "{text}");
    assert_eq!(declines(&graph), [(ushr, ExitReason::Overflow)], "{text}");
}

#[test]
fn unary_and_bitwise_operations_decline_their_number_checks() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function neg(a) { return -a; } function or(a) { return a | 1; }
         for (var i = 0; i < 20; i++) { neg(i + 0.5); or(i); }",
    );
    for (name, op) in [("neg", Op::Neg), ("or", Op::BitOr)] {
        let (chunk, graph) = graph_of(&mut engine, name);
        assert_eq!(count(&graph, is_effect), 0, "{name}");
        assert_eq!(count(&graph, |op| *op == NodeOp::CheckNumber), 1, "{name}");
        let pc = pc_of(&chunk, |candidate| *candidate == op);
        flag(&chunk, pc, ExitReason::NotNumber, false, 0);
        let graph = graph_for(&chunk, name);
        let text = dump(&graph, &chunk);
        assert_eq!(
            count(&graph, |op| *op == NodeOp::CheckNumber),
            0,
            "{name}: {text}"
        );
        assert_eq!(
            nodes(&graph, is_effect)[0].pc as usize,
            pc,
            "{name}: {text}"
        );
    }
}

#[test]
fn a_flagged_shape_check_is_declined_unless_its_ways_grew() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function getx(o) { return o.x; } var p = {x: 1};
         for (var i = 0; i < 20; i++) getx(p);",
    );
    let (chunk, graph) = graph_of(&mut engine, "getx");
    assert_eq!(count(&graph, |op| matches!(op, NodeOp::CheckShape(_))), 1);
    let get = pc_of(&chunk, |op| matches!(op, Op::GetPropLocal(..)));
    // The discarded check expected the one shape the site has: the generic access.
    flag(&chunk, get, ExitReason::WrongShape, false, 1);
    let graph = graph_for(&chunk, "getx flagged");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckShape(_))),
        0,
        "{text}"
    );
    assert_eq!(count(&graph, is_effect), 1, "{text}");
    assert_eq!(declines(&graph), [(get, ExitReason::WrongShape)], "{text}");
    // A second way since: both shapes are checked again.
    eval(&mut engine, "getx({x: 2, y: 3});");
    let graph = graph_for(&chunk, "getx with a new way");
    let text = dump(&graph, &chunk);
    let checks = nodes(&graph, |op| matches!(op, NodeOp::CheckShape(_)));
    assert_eq!(checks.len(), 1, "{text}");
    assert!(
        matches!(&checks[0].op, NodeOp::CheckShape(shapes) if shapes.len() == 2),
        "{text}"
    );
    assert!(graph.declines.is_empty(), "{text}");
}

#[test]
fn a_flagged_block_that_never_ran_is_built() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function nr(o, c) { if (c) { return o.y + o.z(); } return 1; } var p = {y: 1};
         for (var i = 0; i < 20; i++) nr(p, false);",
    );
    let (chunk, graph) = graph_of(&mut engine, "nr");
    let exits = nodes(&graph, |op| *op == NodeOp::Exit(ExitReason::NeverRan));
    assert_eq!(exits.len(), 1);
    let start = graph.state(exits[0].state.expect("a state")).pc as usize;
    flag(&chunk, start, ExitReason::NeverRan, false, 0);
    let graph = graph_for(&chunk, "nr flagged");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| *op == NodeOp::Exit(ExitReason::NeverRan)),
        0,
        "{text}"
    );
    // The block's sites hold no feedback: their generic operations.
    assert!(
        nodes(&graph, is_effect)
            .iter()
            .any(|node| node.pc as usize == start),
        "{text}"
    );
    assert_eq!(declines(&graph), [(start, ExitReason::NeverRan)], "{text}");
}

#[test]
fn a_flagged_guard_branches_to_its_mismatch_without_an_exit() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function add(a, b) { return a + b; }
         function caller(x) { return add(x, 1); }
         function drive(n) { var t = 0; for (var i = 0; i < n; i++) t += caller(i); return t; }
         drive(150);",
    );
    let Some(transformed) = second_stage(&mut engine, "caller", "drive(150)") else {
        return;
    };
    let ops = transformed.jit_ops();
    let guard = pc_of(&transformed, |op| matches!(op, Op::InlineGuard(..)));
    let Op::InlineGuard(target, mismatch) = ops[guard] else {
        unreachable!()
    };
    // The guard's pc is no site the record map names: its flag is the second stage's own.
    assert!(transformed
        .feedback_vector()
        .record_map()
        .is_some_and(|map| map.target(guard).is_none()));
    flag(&transformed, guard, ExitReason::WrongCallee, false, 0);
    let graph = graph_for(&transformed, "caller's flagged second stage");
    let text = dump(&graph, &transformed);
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckCallee { .. })),
        0,
        "{text}"
    );
    let tests = nodes(&graph, |op| *op == NodeOp::CalleeMatches { target });
    assert_eq!(tests.len(), 1, "{text}");
    assert_eq!(tests[0].rep, Rep::Bool, "{text}");
    assert!(tests[0].state.is_none(), "no exit: {text}");
    // The guard's block branches on it to the splice and to the mismatch, which is built.
    let block = graph.block(tests[0].block);
    let last = graph.node(*block.body.last().expect("a terminator"));
    let NodeOp::Branch {
        then_block,
        else_block,
    } = last.op
    else {
        panic!("the guard branches: {text}")
    };
    assert_eq!(
        graph.node(last.inputs[0]).op,
        NodeOp::CalleeMatches { target },
        "{text}"
    );
    assert_eq!(
        graph.block(then_block).first_pc as usize,
        guard + 1,
        "{text}"
    );
    assert_eq!(graph.block(else_block).first_pc, mismatch, "{text}");
    assert_eq!(
        declines(&graph),
        [(guard, ExitReason::WrongCallee)],
        "{text}"
    );
}

#[test]
fn exit_sites_key_where_every_compile_of_the_site_finds_them() {
    // `leaf` spliced into two callers. A discard of c1's second stage keys its exits at leaf's
    // first stage, marked spliced: c2's splice finds both; leaf's own second stage finds the
    // value site but not the named one, a site of another inline kind (JSC's split).
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function leaf(o, b) { return o.v + b; }
         function c1(o) { return leaf(o, 1); } function c2(o) { return leaf(o, 2); }
         var p = {v: 1};
         function drive(n) { var t = 0; for (var i = 0; i < n; i++) t += c1(p) + c2(p);
           return t; }
         drive(150);",
    );
    let (Some(first), Some(second)) = (
        second_stage(&mut engine, "c1", "drive(150)"),
        second_stage(&mut engine, "c2", "drive(150)"),
    ) else {
        return;
    };
    let leaf = chunk_of(&mut engine, "leaf");
    fn map(chunk: &Chunk) -> &crate::feedback::RecordMap {
        chunk.feedback_vector().record_map().expect("a map")
    }
    let site = |chunk: &Rc<Chunk>, which: &dyn Fn(&Op) -> bool| -> (usize, usize) {
        let pc = pc_of(chunk, which);
        let (baseline, baseline_pc, spliced) = map(chunk).site(pc).expect("a mapped site");
        assert!(std::ptr::eq(baseline, &*leaf) && spliced);
        (pc, baseline_pc)
    };
    let is_add = |op: &Op| matches!(op, Op::Add);
    let is_get = |op: &Op| matches!(op, Op::GetPropLocal(..) | Op::GetProp(..));
    let (_, add) = site(&first, &is_add);
    let (_, get) = site(&first, &is_get);
    flag(&leaf, add, ExitReason::NotNumber, true, 0);
    flag(&leaf, get, ExitReason::WrongShape, true, 1);
    // c1's guard: no site the map names, so its own.
    let guard = pc_of(&first, |op| matches!(op, Op::InlineGuard(..)));
    flag(&first, guard, ExitReason::WrongCallee, false, 0);
    // Another caller's splice: both of leaf's sites, none of c1's own.
    let (add2, _) = site(&second, &is_add);
    let (get2, _) = site(&second, &is_get);
    let snapshot = Snapshot::take(&second);
    assert!(snapshot.declines(add2, ExitReason::NotNumber));
    assert!(snapshot.declines(get2, ExitReason::WrongShape));
    assert_eq!(snapshot.discarded_ways(get2), 1);
    let guard2 = pc_of(&second, |op| matches!(op, Op::InlineGuard(..)));
    assert!(!snapshot.declines(guard2, ExitReason::WrongCallee));
    // c1's recompile: its own site too.
    let snapshot = Snapshot::take(&first);
    assert!(snapshot.declines(guard, ExitReason::WrongCallee));
    // leaf's own second stage (a recompile with nothing spliced): the value site only.
    let env = engine.interp.global_env;
    let value: Value = engine.interp.get_var("leaf", &env).ok().expect("leaf");
    let ValueView::Obj(function) = value.view() else {
        panic!("leaf is an object")
    };
    let ObjectKind::User(user) = &function.borrow().kind else {
        panic!("leaf is a user function")
    };
    let own = crate::bytecode::compile_with_inlines(&user.func, &Default::default(), &leaf)
        .expect("leaf's recompile");
    let own_add = pc_of(&own, is_add);
    let own_get = pc_of(&own, is_get);
    let (baseline, baseline_add, spliced) = map(&own).site(own_add).expect("mapped");
    assert!(std::ptr::eq(baseline, &*leaf) && baseline_add == add && !spliced);
    let snapshot = Snapshot::take(&own);
    assert!(snapshot.declines(own_add, ExitReason::NotNumber));
    assert!(!snapshot.declines(own_get, ExitReason::WrongShape));
    // And leaf's first stage itself, whose own sites are its spliced callers' context for the
    // named one.
    let snapshot = Snapshot::take(&leaf);
    assert!(snapshot.declines(add, ExitReason::NotNumber));
    assert!(!snapshot.declines(get, ExitReason::WrongShape));
}

#[test]
fn the_verifier_keeps_addresses_out_of_phis() {
    // An address dies with its effect epoch: a loop's back edge would carry it across the poll.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function sum(n) { var s = 0; for (var i = 0; i < n; i++) s += i; return s; }
         for (var k = 0; k < 20; k++) sum(10);",
    );
    let (chunk, graph) = graph_of(&mut engine, "sum");
    let phi = graph
        .blocks
        .iter()
        .flat_map(|block| block.phis.iter())
        .copied()
        .next()
        .expect("a loop phi");
    let mut broken = graph.clone();
    broken.nodes[phi.index()].rep = Rep::Ptr;
    let error = verify(&broken, &chunk).expect_err("a phi of addresses");
    assert!(error.contains("a phi of addresses"), "{error}");
}

#[test]
fn the_snapshot_reads_element_records() {
    // The VM records every element access before it runs it (feedback-schema.txt 5.2): the
    // receiver's family, the key's kind and an Array's facts.
    use crate::feedback::{ELEM_ARRAY, ELEM_HOLEY, ELEM_KEY_INDEX, ELEM_OUT_OF_BOUNDS};
    let mut engine = engine(Tier::Bytecode);
    eval(
        &mut engine,
        "function get(a, i) { return a[i]; } var holes = [1, , 3];
         for (var k = 0; k < 20; k++) { get([1, 2], 0); get(holes, 1); get([1], 5); }",
    );
    let chunk = chunk_of(&mut engine, "get");
    let snapshot = Snapshot::take(&chunk);
    let pc = pc_of(&chunk, |op| matches!(op, Op::GetElemLocal(_)));
    let view = snapshot.element(pc).expect("an element site's view");
    assert_eq!(view.recorded_receivers, ELEM_ARRAY);
    assert_eq!(view.recorded_keys, ELEM_KEY_INDEX);
    assert_eq!(
        view.facts & (ELEM_HOLEY | ELEM_OUT_OF_BOUNDS),
        ELEM_HOLEY | ELEM_OUT_OF_BOUNDS
    );
    assert!(!view.generic);
    assert!(snapshot.values(pc).is_none() && snapshot.named(pc).is_none());
    let ret = pc_of(&chunk, |op| matches!(op, Op::Return));
    assert!(snapshot.element(ret).is_none());
}

/// The VM's engine after `source`: its element records hold what every access saw.
fn vm_engine(source: &str) -> Engine {
    let mut engine = engine(Tier::Bytecode);
    eval(&mut engine, source);
    engine
}

#[test]
fn element_loads_check_the_receiver_and_the_bounds_and_load_the_word() {
    let mut engine = vm_engine(
        "function get(a, i) { return a[i]; }
         for (var k = 0; k < 20; k++) get([1, 2, 3], k % 3);",
    );
    let (chunk, graph) = graph_of(&mut engine, "get");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let pc = pc_of(&chunk, |op| matches!(op, Op::GetElemLocal(_)));
    for (op, reason) in [
        (NodeOp::CheckObject, ExitReason::NotObject),
        (
            NodeOp::CheckElements(ElemKind::Array),
            ExitReason::WrongElements,
        ),
        (NodeOp::CheckNumber, ExitReason::NotNumber),
        (NodeOp::CheckIndex, ExitReason::NotInt32),
        (NodeOp::CheckBounds, ExitReason::OutOfBounds),
        (NodeOp::LoadElement(Holes::Exit), ExitReason::Hole),
    ] {
        let found = nodes(&graph, |candidate| *candidate == op);
        assert_eq!(found.len(), 1, "{op:?}: {text}");
        assert_eq!(found[0].reason, Some(reason), "{text}");
        // Every check exits before the load, with its operand (the key) on the stack.
        let state = graph.state(found[0].state.expect("a check has a state"));
        assert_eq!((state.pc as usize, state.stack.len()), (pc, 1), "{text}");
    }
    assert_eq!(count(&graph, |op| *op == NodeOp::LoadElements), 1, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::LoadElementsUsed),
        1,
        "{text}"
    );
}

#[test]
fn an_element_records_facts_choose_what_holes_and_reads_past_used_do() {
    let mut engine = vm_engine(
        "function dense(a, i) { return a[i]; } function holey(a, i) { return a[i]; }
         function past(a, i) { return a[i]; } function mixed(o, i) { return o[i]; }
         var h = [1, , 3];
         for (var k = 0; k < 20; k++) {
           dense([1, 2], 1); holey(h, k % 3); past([1, 2], k % 4); mixed(k % 2 ? [1] : {0: 1}, 0);
         }",
    );
    let shapes = [
        ("dense", Some(NodeOp::LoadElement(Holes::Exit)), true),
        ("holey", Some(NodeOp::LoadElement(Holes::Undefined)), true),
        ("past", Some(NodeOp::LoadElementOrUndefined), false),
        ("mixed", Some(NodeOp::LoadElement(Holes::Exit)), true),
    ];
    for (name, load, bounds) in shapes {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        if let Some(load) = load {
            assert_eq!(count(&graph, |op| *op == load), 1, "{name}: {text}");
        }
        assert_eq!(
            count(&graph, |op| *op == NodeOp::CheckBounds),
            usize::from(bounds),
            "{name}: {text}"
        );
    }
    // Ordinary objects among the receivers: element words of either kind.
    let (chunk, graph) = graph_of(&mut engine, "mixed");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| *op == NodeOp::CheckElements(ElemKind::Words)),
        1,
        "{text}"
    );
    let past = nodes(&graph, |op| *op == NodeOp::LoadElementOrUndefined);
    assert!(past.is_empty(), "{text}");
}

#[test]
fn elements_and_reads_last_an_effect_epoch_and_kinds_outlive_it() {
    // `s + 'x'` is the generic Add (its record saw Strings), an effect with no cache, which the
    // VM's runs give the evidence a call site's would lack (the VM fills no call cache).
    let mut engine = vm_engine(
        "function twice(a, i) { return a[i] + a[i] + a[i + 1]; }
         function around(a, i, s) { var x = a[i]; s = s + 'x'; return x + a[i]; }
         for (var k = 0; k < 20; k++) { twice([1, 2, 3], 1); around([1, 2], 0, 's'); }",
    );
    // One read of a[i], one of a[i + 1]; the words and their used length once.
    let (chunk, graph) = graph_of(&mut engine, "twice");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let loads = count(&graph, |op| matches!(op, NodeOp::LoadElement(_)));
    assert_eq!(loads, 2, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::LoadElements), 1, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::LoadElementsUsed),
        1,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckElements(_))),
        1,
        "{text}"
    );
    // After the effect the words and the element are read again; `a` is reloaded from its slot,
    // so its checks run again too.
    let (chunk, graph) = graph_of(&mut engine, "around");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 1, "the Add: {text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::LoadElements), 2, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadElement(_))),
        2,
        "{text}"
    );
}

#[test]
fn element_keys_take_their_int32_form_or_a_checked_index() {
    let mut engine = vm_engine(
        "function at(a, i) { var j = i | 0; return a[j] + a[0]; }
         function half(a, x) { return a[x / 2]; }
         for (var k = 0; k < 20; k++) { at([1, 2], 1); half([1, 2], 2); }",
    );
    // `i | 0` is an int32 already, and the constant folds.
    let (chunk, graph) = graph_of(&mut engine, "at");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckIndex), 0, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadElement(_))),
        2,
        "{text}"
    );
    assert!(graph
        .nodes
        .iter()
        .any(|node| node.op == NodeOp::ConstI32(0)));
    // A quotient is an f64: the index is its CheckIndex.
    let (chunk, graph) = graph_of(&mut engine, "half");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let index = nodes(&graph, |op| *op == NodeOp::CheckIndex);
    assert_eq!(index.len(), 1, "{text}");
    assert_eq!(graph.node(index[0].inputs[0]).op, NodeOp::F64Div, "{text}");
}

#[test]
fn compound_element_assignment_takes_its_number_key_as_it_is() {
    let mut engine = vm_engine(
        "function add(a, i, v) { a[i] += v; return a; }
         function addo(o, i, v) { o.a[i] |= v; return o; }
         function named(o, k) { o[k] += 1; return o; }
         for (var k = 0; k < 20; k++) {
           add([1, 2], 1, 3); addo({a: [1, 2]}, 0, 4); named({x: 1}, 'x');
         }",
    );
    for name in ["add", "addo"] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        let to_key = pc_of(&chunk, |op| {
            matches!(op, Op::ToPropKey | Op::ToPropKeyLocal(_))
        });
        assert!(
            nodes(&graph, is_effect)
                .iter()
                .all(|node| node.pc as usize != to_key),
            "{name}: {text}"
        );
        // The base is checked an object at ToPropKey's pc, whose throw the VM then makes.
        let check = nodes(&graph, |op| *op == NodeOp::CheckObject);
        assert_eq!(check.len(), 1, "{name}: {text}");
        let state = graph.state(check[0].state.expect("a check has a state"));
        assert_eq!(state.pc as usize, to_key, "{name}: {text}");
        let load = pc_of(&chunk, |op| matches!(op, Op::GetElem | Op::GetElemLocal(_)));
        assert!(
            nodes(&graph, is_effect)
                .iter()
                .all(|node| node.pc as usize != load),
            "{name}: {text}"
        );
    }
    // A String key: the site is generic, and ToPropKey stays the operation.
    let (chunk, graph) = graph_of(&mut engine, "named");
    let text = dump(&graph, &chunk);
    let to_key = pc_of(&chunk, |op| {
        matches!(op, Op::ToPropKey | Op::ToPropKeyLocal(_))
    });
    assert!(
        nodes(&graph, is_effect)
            .iter()
            .any(|node| node.pc as usize == to_key),
        "{text}"
    );
}

/// Crypto's am3 (Octane crypto.js:108-122), which montReduce's second stage splices, on
/// BigInteger-like receivers, run by the VM so that its records hold what each access saw.
const AM3: &str = "function am3(i, x, w, j, c, n) {
       var this_array = this.array; var w_array = w.array;
       var xl = x & 0x3fff, xh = x >> 14;
       while (--n >= 0) {
         var l = this_array[i] & 0x3fff; var h = this_array[i++] >> 14;
         var m = xh * l + h * xl; l = xl * l + ((m & 0x3fff) << 14) + w_array[j] + c;
         c = (l >> 28) + (m >> 14) + xh * h; w_array[j++] = l & 0xfffffff;
       }
       return c;
     }
     function BI(n) { this.array = []; for (var k = 0; k < n; k++) this.array.push(k * 7919); }
     BI.prototype.am = am3;
     var x = new BI(8), w = new BI(16);
     for (var k = 0; k < 20; k++) x.am(0, 12345, w, k % 8, 0, 8);";

#[test]
fn am3_runs_its_loop_without_element_effects() {
    let mut engine = vm_engine(AM3);
    let (chunk, graph) = graph_of(&mut engine, "am3");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    // this_array[i] once for both of its reads (i, then i before i++), w_array[j] once, and
    // w_array[j++]'s write proven present by that read: no bounds or hole test.
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadElement(_))),
        2,
        "{text}"
    );
    let stores = nodes(&graph, |op| matches!(op, NodeOp::StoreElement(_)));
    assert_eq!(stores.len(), 1, "{text}");
    assert_eq!(stores[0].op, NodeOp::StoreElement(Store::Present), "{text}");
    // The digit is an int32: stored as its f64 bits, with no card.
    assert_eq!(graph.node(stores[0].inputs[3]).rep, Rep::I32, "{text}");
    // The loop's blocks hold its elements: the loads and the store are in a loop body.
    let header = graph
        .blocks
        .iter()
        .position(|block| block.loop_header)
        .expect("the loop");
    assert!(
        graph.nodes.iter().all(|node| {
            !matches!(node.op, NodeOp::StoreElement(_)) || node.block.index() > header
        }),
        "{text}"
    );
}

#[test]
fn a_loop_has_phis_only_for_the_slots_live_at_its_header() {
    // am3's `l`, `h` and `m`, declared in its body, are written there before any read: dead at
    // the header, where neither a phi nor the poll's state holds them (stage 2 step 6). `x`, read
    // only before the loop, is dead there too.
    let mut engine = vm_engine(AM3);
    let (chunk, graph) = graph_of(&mut engine, "am3");
    let text = dump(&graph, &chunk);
    let header = graph
        .blocks
        .iter()
        .find(|block| block.loop_header)
        .expect("the loop");
    let state = graph.state(graph.node(header.body[0]).state.expect("the poll's state"));
    let mut carried: Vec<&str> = (0..state.slots.len())
        .filter(|&slot| {
            matches!(state.slots[slot], SlotSource::Value(value) if header.phis.contains(&value))
        })
        .map(|slot| &*chunk.jit_slot_names()[slot])
        .collect();
    carried.sort_unstable();
    assert_eq!(carried, ["c", "i", "j", "n"], "{text}");
    assert_eq!(header.phis.len(), 4, "{text}");
    for name in ["l", "h", "m", "x"] {
        assert_eq!(
            state.slots[slot_named(&chunk, name) as usize],
            SlotSource::Dead,
            "{name}: {text}"
        );
    }
}

#[test]
fn element_stores_take_the_values_representation_and_end_the_epochs_reads() {
    let mut engine = vm_engine(
        "function seti(a, i, v) { a[i] = v | 0; } function setd(a, i, v) { a[i] = v * 0.5; }
         function setw(a, i, v) { a[i] = v; } function rmw(a, i) { a[i] = a[i] + 1; }
         function two(a, b, i) { a[i] = 1; return b[i] + a[i]; }
         for (var k = 0; k < 20; k++) {
           seti([1, 2], 1, k); setd([1, 2], 0, k); setw([1, 2], 1, 'v' + k); rmw([1, 2], 1);
           two([1, 2], [3, 4], 0);
         }",
    );
    // An Array site's store fills holes and appends too (Create), unless the element is proven.
    for (name, mode, rep) in [
        ("seti", Store::Create, Rep::I32),
        ("setd", Store::Create, Rep::F64),
        ("setw", Store::Create, Rep::Tagged),
        ("rmw", Store::Present, Rep::I32),
    ] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        let stores = nodes(&graph, |op| matches!(op, NodeOp::StoreElement(_)));
        assert_eq!(stores.len(), 1, "{name}: {text}");
        assert_eq!(stores[0].op, NodeOp::StoreElement(mode), "{name}: {text}");
        assert_eq!(graph.node(stores[0].inputs[3]).rep, rep, "{name}: {text}");
        assert_eq!(
            stores[0].reason,
            Some(ExitReason::WrongElements),
            "{name}: {text}"
        );
        // The store exits before its operation, with its operands on the stack.
        let state = graph.state(stores[0].state.expect("a store has a state"));
        let pc = pc_of(&chunk, |op| matches!(op, Op::SetElemLocalDrop(_)));
        assert_eq!((state.pc as usize, state.stack.len()), (pc, 2), "{text}");
    }
    // After a store, another word's words are loaded again (it may name the same object); the
    // stored element is the value written.
    let (chunk, graph) = graph_of(&mut engine, "two");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, |op| *op == NodeOp::LoadElements), 2, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadElement(_))),
        1,
        "b[i] alone: {text}"
    );
}

#[test]
fn array_stores_fill_holes_and_append_in_line() {
    // Appends and hole fills run in line in the first stage and record nothing, so a site that
    // saw Arrays always compiles them; a site of ordinary objects alone stores existing elements.
    let mut engine = vm_engine(
        "function put(a, i, v) { a[i] = v; } function puto(o, i, v) { o[i] = v; }
         for (var k = 0; k < 20; k++) { put([1, 2], 1, k); puto({0: 1, 1: 2}, 1, k); }",
    );
    for (name, mode) in [("put", Store::Create), ("puto", Store::Existing)] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        let stores = nodes(&graph, |op| matches!(op, NodeOp::StoreElement(_)));
        assert_eq!(stores.len(), 1, "{name}: {text}");
        assert_eq!(stores[0].op, NodeOp::StoreElement(mode), "{name}: {text}");
    }
}

#[test]
fn an_arrays_length_is_a_load_that_an_append_ends() {
    let mut engine = vm_engine(
        "function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
         function push(a, x) { var n = a.length; a[a.length] = x; return a.length - n; }
         for (var k = 0; k < 20; k++) { sum([1, 2, 3]); push([1, 2], k); }",
    );
    // `i < a.length` loops keep their bodies free of effects.
    let (chunk, graph) = graph_of(&mut engine, "sum");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::LoadArrayLength),
        1,
        "{text}"
    );
    // The loop's invariant `a` has its kind check moved to the loop's entry (P3 stage 3 step 3a:
    // its entry value is an invariant of the loop, which has no effect).
    assert!(
        nodes(&graph, |op| *op == NodeOp::CheckElements(ElemKind::Array))
            .iter()
            .all(|node| node.reason == Some(ExitReason::HoistingFailed)),
        "{text}"
    );
    // Read once for `n` and the key, and again after the append, which ends the first read.
    let (chunk, graph) = graph_of(&mut engine, "push");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::LoadArrayLength),
        2,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| *op == NodeOp::StoreElement(Store::Create)),
        1,
        "{text}"
    );
}

#[test]
fn an_element_update_reads_adds_and_writes_back() {
    let mut engine = vm_engine(
        "function inc(a, i) { a[i]++; return a; } function pre(a, i) { return ++a[i]; }
         function post(a, i) { return a[i]--; } function half(a, i) { return a[i]++; }
         for (var k = 0; k < 20; k++) {
           inc([1, 2], 1); pre([1, 2], 0); post([3, 4], 1); half([0.5, 1.5], 0);
         }",
    );
    // The VM's numeric update records nothing (its fast path), so no record hints int32: f64
    // arithmetic, whose store keeps INT32 for an exact int32.
    for (name, add, rep) in [
        ("inc", NodeOp::F64Add, Rep::F64),
        ("pre", NodeOp::F64Add, Rep::F64),
        ("post", NodeOp::F64Sub, Rep::F64),
        ("half", NodeOp::F64Add, Rep::F64),
    ] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert_eq!(count(&graph, |op| *op == add), 1, "{name}: {text}");
        // The read proved the element present: no second bounds or hole test.
        let stores = nodes(&graph, |op| matches!(op, NodeOp::StoreElement(_)));
        assert_eq!(stores.len(), 1, "{name}: {text}");
        assert_eq!(
            stores[0].op,
            NodeOp::StoreElement(Store::Present),
            "{name}: {text}"
        );
        assert_eq!(graph.node(stores[0].inputs[3]).rep, rep, "{name}: {text}");
        let pc = pc_of(&chunk, |op| matches!(op, Op::UpdateElem(_)));
        let state = graph.state(stores[0].state.expect("a store has a state"));
        assert_eq!((state.pc as usize, state.stack.len()), (pc, 2), "{text}");
    }
}

#[test]
fn element_sites_the_words_do_not_serve_stay_effects() {
    // Strings, arguments objects, a generic site, typed arrays whose stores take values that are
    // no Numbers (a typed array converts them first, which may run code), and typed arrays the
    // native code declines (BigInt elements), whose record holds no kind.
    let mut engine = vm_engine(
        "function str(s, i) { return s[i]; } function store(t, i) { t[i] = 's'; }
         function args(x, i) { return x[i]; } function generic(o, k) { return o[k]; }
         function big(t, i) { return t[i]; } function bigs(t, i) { return t[i]; }
         function argsof() { return arguments; } var u8 = new Uint8Array(4);
         var list = argsof(1, 2), b64 = new BigInt64Array(4);
         for (var k = 0; k < 20; k++) {
           str('abc', 1); store(u8, 1); args(list, 1); generic({a: 1}, 'a'); big(b64, 1);
           bigs(k % 2 ? b64 : [1, 2], 1);
         }",
    );
    for name in ["str", "store", "args", "generic", "big", "bigs"] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(
            count(&graph, |op| matches!(
                op,
                NodeOp::LoadElement(_)
                    | NodeOp::StoreTyped(_)
                    | NodeOp::TypedElement { .. }
                    | NodeOp::CheckElements(_)
            )),
            0,
            "{name}: {text}"
        );
        assert!(count(&graph, is_effect) >= 1, "{name}: {text}");
    }
}

#[test]
fn typed_array_sites_access_their_one_kind_in_line() {
    // Maglev's typed-array access (TryBuildElementAccessOnTypedArray, maglev-graph-builder.cc:
    // 5052-5211): the kind checked, the view and its length value facts, the bounds checked per
    // access and the data address per effect epoch, then the element by its kind; a store's
    // value converted first.
    let mut engine = vm_engine(
        "function sum(t) { var s = 0; for (var i = 0; i < 4; i++) s += t[i]; return s; }
         function sumU(t) { var s = 0; for (var i = 0; i < 4; i++) s += t[i]; return s; }
         function put(t, i, v) { t[i] = v; } function clamp(t, i, v) { t[i] = v; }
         function both(t, i) { t[i] = t[i] + 1; return t[i + 1]; }
         var f = new Float64Array(4), u = new Uint8Array(4), c = new Uint8ClampedArray(4);
         var w = new Int32Array(8), big = new Uint32Array(4);
         for (var k = 0; k < 20; k++) {
           sum(f); put(u, k % 4, k * 1.5); clamp(c, 1, k + 0.5); both(w, k % 4); sumU(big);
         }",
    );
    let typed = |kind| NodeOp::CheckElements(ElemKind::Typed(Some(kind)));
    // Uint32 elements pass the int32 range: an f64.
    for (name, kind, rep) in [
        ("sum", TypedKind::F64, Rep::F64),
        ("sumU", TypedKind::U32, Rep::F64),
    ] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        for op in [
            typed(kind),
            NodeOp::TypedView,
            NodeOp::TypedLength,
            NodeOp::CheckBounds,
            NodeOp::TypedData { write: false },
            NodeOp::LoadTyped(kind),
        ] {
            let found = nodes(&graph, |candidate| *candidate == op);
            assert_eq!(found.len(), 1, "{name} {op:?}: {text}");
            if op == NodeOp::LoadTyped(kind) {
                assert_eq!(found[0].rep, rep, "{name}: {text}");
            }
        }
        // The loop has no effect and reads its parameter's array first inside it: its entry
        // checks the kind and the data address (P3 stage 3 step 0c), whose exits there are
        // HoistingFailed's.
        let checks = nodes(&graph, |op| matches!(op, NodeOp::CheckElements(_)));
        assert_eq!(checks[0].reason, Some(ExitReason::HoistingFailed), "{text}");
        let data = nodes(&graph, |op| matches!(op, NodeOp::TypedData { .. }));
        assert_eq!(data[0].reason, Some(ExitReason::HoistingFailed), "{text}");
    }
    // The stores: an integer kind's ToInt32 truncation, Uint8Clamped's clamp.
    for (name, kind, convert) in [
        ("put", TypedKind::U8, NodeOp::TruncateToInt32),
        ("clamp", TypedKind::U8Clamped, NodeOp::ClampToUint8),
    ] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        let stores = nodes(&graph, |op| *op == NodeOp::StoreTyped(kind));
        assert_eq!(stores.len(), 1, "{name}: {text}");
        assert_eq!(
            graph.node(stores[0].inputs[4]).op,
            convert,
            "{name}: {text}"
        );
        assert_eq!(
            count(&graph, |op| *op == NodeOp::TypedData { write: true }),
            1,
            "{name}: {text}"
        );
    }
    // A read, a write and a read of one Int32Array: one view, and one data address, which the
    // first read checks for the store its block makes (P3 stage 3 step 0c) and the rest take.
    let (chunk, graph) = graph_of(&mut engine, "both");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(
        count(&graph, |op| *op == typed(TypedKind::I32)),
        1,
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::TypedView), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckBounds), 3, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::TypedData { write: true }),
        1,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| *op == NodeOp::TypedData { write: false }),
        0,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| *op == NodeOp::LoadTyped(TypedKind::I32)),
        2,
        "{text}"
    );
    let stores = nodes(&graph, |op| *op == NodeOp::StoreTyped(TypedKind::I32));
    assert_eq!(stores.len(), 1, "{text}");
    assert_eq!(graph.node(stores[0].inputs[4]).rep, Rep::I32, "{text}");
}

#[test]
fn typed_sites_of_more_than_one_kind_call_the_template_stub() {
    let mut engine = vm_engine(
        "function get(t, i) { return t[i]; } function put(t, i, v) { t[i] = v; }
         var f = new Float64Array(4), u = new Uint8Array(4), big = new BigInt64Array(4);
         for (var k = 0; k < 20; k++) { var t = k % 2 ? f : u; get(t, 1); put(t, 1, k); }
         get(big, 1);",
    );
    for (name, op) in [
        ("get", NodeOp::TypedElement { write: false }),
        ("put", NodeOp::TypedElement { write: true }),
    ] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        let calls = nodes(&graph, |candidate| *candidate == op);
        assert_eq!(calls.len(), 1, "{name}: {text}");
        assert_eq!(
            calls[0].reason,
            Some(ExitReason::OutOfBounds),
            "{name}: {text}"
        );
        assert_eq!(
            count(&graph, |op| *op
                == NodeOp::CheckElements(ElemKind::Typed(None))),
            1,
            "{name}: {text}"
        );
        assert_eq!(
            count(&graph, |op| *op == NodeOp::TypedView),
            0,
            "{name}: {text}"
        );
    }
}

#[test]
fn typed_arrays_among_other_receivers_branch_on_the_kind_tag() {
    // The template dispatches on the tag (emit_get_elem_inline): the stub on one side, the words
    // on the other, a phi of their words; an empty record (the first stage's in-line hits
    // record nothing) takes the same branch.
    let mut engine = vm_engine(
        "function get(o, i) { return o[i]; } function put(o, i, v) { o[i] = v; }
         var f = new Float64Array(4);
         for (var k = 0; k < 20; k++) { var o = k % 2 ? f : [1, 2]; get(o, 1); put(o, 1, k); }",
    );
    let shape = |graph: &Graph, write: bool| {
        count(graph, |op| *op == NodeOp::IsTypedArray) == 1
            && count(graph, |op| matches!(op, NodeOp::Branch { .. })) == 1
            && count(graph, |op| *op == NodeOp::TypedElement { write }) == 1
            && count(graph, |op| {
                matches!(op, NodeOp::CheckElements(ElemKind::Array))
            }) == 1
    };
    let (chunk, graph) = graph_of(&mut engine, "get");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert!(shape(&graph, false), "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadElement(_))),
        1,
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::Phi), 1, "{text}");
    let (chunk, graph) = graph_of(&mut engine, "put");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert!(shape(&graph, true), "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::StoreElement(_))),
        1,
        "{text}"
    );
    // Under ARM64's first stage alone, the site's record stays empty (x86-64's calls the
    // element helper, which records every access).
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    {
        let mut jit = self::engine(Tier::Jit);
        eval(
            &mut jit,
            "function get(o, i) { return o[i]; }
             for (var k = 0; k < 20; k++) get([1, 2, 3], k % 3);",
        );
        let (chunk, graph) = graph_of(&mut jit, "get");
        let pc = pc_of(&chunk, |op| matches!(op, Op::GetElemLocal(_)));
        let view = Snapshot::take(&chunk).element(pc).expect("an element site");
        assert_eq!(view.recorded_receivers, 0, "an empty record");
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{text}");
        assert_eq!(count(&graph, |op| *op == NodeOp::IsTypedArray), 1, "{text}");
        let words = |op: &NodeOp| matches!(op, NodeOp::CheckElements(ElemKind::Words));
        assert_eq!(count(&graph, words), 1, "{text}");
    }
}

#[test]
fn a_kind_template_code_saw_last_lowers_its_site_in_line_and_a_wider_record_stays_wide() {
    // P3 stage 3, rule U: a site the VM never ran (an empty record) whose last-seen byte holds a
    // kind, as template code's in-line hit leaves it, reads once merged as typed arrays of that
    // kind: per-kind loads and stores in line, with no stub and no kind branch. A record that
    // holds more keeps it: a Float64Array the VM saw with a Uint8Array last seen takes the stub
    // (two kinds), an Array the VM saw with an Int32Array last seen the kind branch.
    use super::profile::{ChunkProfile, UNSEEN};
    use crate::feedback::{elem_typed_kind, ELEM_KEY_INDEX, ELEM_TYPED_ARRAY};
    let mut engine = vm_engine(
        "function get(t, i, go) { if (go) return t[i]; return 0; }
         function put(t, i, v, go) { if (go) t[i] = v; return 0; }
         function wide(t, i) { return t[i]; } function mixed(o, i) { return o[i]; }
         var f = new Float64Array(4);
         for (var k = 0; k < 20; k++) {
           get(f, 1, false); put(f, 1, k, false); wide(f, 1); mixed([1, 2], 1);
         }",
    );
    let load = |op: &Op| matches!(op, Op::GetElemLocal(_));
    let store = |op: &Op| matches!(op, Op::SetElemLocalDrop(_));
    let typed = |kind| NodeOp::CheckElements(ElemKind::Typed(kind));
    for (name, site, kind, access) in [
        (
            "get",
            &load as &dyn Fn(&Op) -> bool,
            TypedKind::I32,
            NodeOp::LoadTyped(TypedKind::I32),
        ),
        (
            "put",
            &store,
            TypedKind::F32,
            NodeOp::StoreTyped(TypedKind::F32),
        ),
    ] {
        let chunk = chunk_of(&mut engine, name);
        let pc = pc_of(&chunk, site);
        assert_eq!(
            chunk.feedback_vector().record_word(pc),
            0,
            "{name}: never ran"
        );
        ChunkProfile::of(&chunk).see_element(chunk.jit_ops().len(), pc, kind as u8);
        let graph = graph_for(&chunk, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert_eq!(
            count(&graph, |op| *op == typed(Some(kind))),
            1,
            "{name}: {text}"
        );
        assert_eq!(count(&graph, |op| *op == access), 1, "{name}: {text}");
        assert_eq!(
            count(&graph, |op| matches!(
                op,
                NodeOp::TypedElement { .. } | NodeOp::IsTypedArray
            )),
            0,
            "{name}: {text}"
        );
        // Merged into the record, the byte emptied.
        assert_eq!(
            chunk.feedback_vector().record_word(pc),
            ELEM_TYPED_ARRAY | ELEM_KEY_INDEX | elem_typed_kind(kind as u8),
            "{name}"
        );
        assert_eq!(ChunkProfile::of(&chunk).element_seen(pc), UNSEEN, "{name}");
    }
    // The VM's Float64Array and a Uint8Array last seen: two kinds, the stub.
    let chunk = chunk_of(&mut engine, "wide");
    let pc = pc_of(&chunk, load);
    ChunkProfile::of(&chunk).see_element(chunk.jit_ops().len(), pc, TypedKind::U8 as u8);
    let graph = graph_for(&chunk, "wide");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, |op| *op == typed(None)), 1, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::TypedElement { write: false }),
        1,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadTyped(_))),
        0,
        "{text}"
    );
    // The VM's Array and an Int32Array last seen: the kind branch, the stub on its typed side.
    let chunk = chunk_of(&mut engine, "mixed");
    let pc = pc_of(&chunk, load);
    ChunkProfile::of(&chunk).see_element(chunk.jit_ops().len(), pc, TypedKind::I32 as u8);
    let graph = graph_for(&chunk, "mixed");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, |op| *op == NodeOp::IsTypedArray), 1, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::CheckElements(ElemKind::Array)),
        1,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadTyped(_))),
        0,
        "{text}"
    );
}

#[test]
fn a_family_template_code_saw_last_drops_the_kind_branch() {
    // P3 stage 3, rule U: an empty record whose family byte holds an Array's tag (template code's
    // words-path hit) reads once merged as Arrays alone: the element words with an Array's kind
    // check, no IsTypedArray, no stub; an ordinary object's tag, as words of either; a typed kind
    // and an Array's tag both seen, the kind branch.
    use super::profile::{ChunkProfile, UNSEEN};
    let mut engine = vm_engine(
        "function arr(a, i, go) { if (go) return a[i]; return 0; }
         function obj(o, i, go) { if (go) return o[i]; return 0; }
         function both(o, i, go) { if (go) return o[i]; return 0; }
         for (var k = 0; k < 20; k++) { arr(null, 1, false); obj(null, 1, false); both(null, 1, false); }",
    );
    let load = |op: &Op| matches!(op, Op::GetElemLocal(_));
    let array_tag = crate::value::KIND_ARRAY_TAG;
    for (name, tag, typed, kind) in [
        ("arr", array_tag, None, Some(ElemKind::Array)),
        ("obj", 0, None, Some(ElemKind::Words)),
        ("both", array_tag, Some(TypedKind::I32), None),
    ] {
        let chunk = chunk_of(&mut engine, name);
        let pc = pc_of(&chunk, load);
        assert_eq!(
            chunk.feedback_vector().record_word(pc),
            0,
            "{name}: never ran"
        );
        let profile = ChunkProfile::of(&chunk);
        profile.see_family(chunk.jit_ops().len(), pc, tag);
        if let Some(typed) = typed {
            profile.see_element(chunk.jit_ops().len(), pc, typed as u8);
        }
        let graph = graph_for(&chunk, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        let branch = count(&graph, |op| *op == NodeOp::IsTypedArray);
        match kind {
            Some(kind) => {
                assert_eq!(branch, 0, "{name}: {text}");
                assert_eq!(
                    count(&graph, |op| *op == NodeOp::CheckElements(kind)),
                    1,
                    "{name}: {text}"
                );
                assert_eq!(
                    count(&graph, |op| matches!(op, NodeOp::TypedElement { .. })),
                    0,
                    "{name}: {text}"
                );
            }
            None => assert_eq!(branch, 1, "{name}: {text}"),
        }
        assert_eq!(profile.family_seen(pc), UNSEEN, "{name}: emptied");
    }
}

#[test]
fn a_last_seen_kind_changes_the_fingerprint_once() {
    // The settled-feedback rule merges before it compares (JSC merges at every tier-up check):
    // a kind template code saw since the last checkpoint changes the fingerprint; seen again, it
    // changes nothing.
    use super::profile::ChunkProfile;
    let mut engine = vm_engine(
        "function get(t, i, go) { if (go) return t[i]; return 0; }
         for (var k = 0; k < 20; k++) get(null, 1, false);",
    );
    let chunk = chunk_of(&mut engine, "get");
    let pc = pc_of(&chunk, |op| matches!(op, Op::GetElemLocal(_)));
    let before = fingerprint(&chunk);
    assert_eq!(fingerprint(&chunk), before);
    ChunkProfile::of(&chunk).see_element(chunk.jit_ops().len(), pc, TypedKind::U16 as u8);
    let seen = fingerprint(&chunk);
    assert_ne!(seen, before, "a new kind is new feedback");
    ChunkProfile::of(&chunk).see_element(chunk.jit_ops().len(), pc, TypedKind::U16 as u8);
    assert_eq!(fingerprint(&chunk), seen, "the same kind again");
}

#[test]
fn flagged_typed_sites_read_past_the_length_or_take_the_generic_access() {
    let source = "function get(t, i) { return t[i]; } function put(t, i, v) { t[i] = v; }
         function any(t, i) { return t[i]; } function mixed(t, i) { return t[i]; }
         var f = new Float64Array(4), u = new Uint8Array(4);
         for (var k = 0; k < 20; k++) {
           get(f, 1); put(f, 1, k); any(k % 2 ? f : u, 1); mixed(k % 2 ? f : [1, 2], 1);
         }";
    let get = |op: &Op| matches!(op, Op::GetElemLocal(_));
    let put = |op: &Op| matches!(op, Op::SetElemLocalDrop(_));
    let loads = |graph: &Graph| count(graph, |op| matches!(op, NodeOp::LoadTyped(_))) == 1;
    let stores = |graph: &Graph| count(graph, |op| matches!(op, NodeOp::StoreTyped(_))) == 1;
    let stub = |graph: &Graph| count(graph, |op| matches!(op, NodeOp::TypedElement { .. })) == 1;
    // A load of one kind reads past the length: undefined there, with no bounds exit.
    let mut engine = vm_engine(source);
    let (chunk, pc, graph) =
        flagged_element_graph(&mut engine, "get", get, ExitReason::OutOfBounds, loads);
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(
        count(&graph, |op| *op
            == NodeOp::LoadTypedOrUndefined(TypedKind::F64)),
        1,
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckBounds), 0, "{text}");
    assert_eq!(declines(&graph), [(pc, ExitReason::OutOfBounds)], "{text}");
    assert_eq!(graph.declines[0].form, "the read past the view's length");
    // A buffer's flag, a store's bounds, and the bounds where the stub's status or the words'
    // reads share the flag: the generic access.
    let cases: [(
        &str,
        &dyn Fn(&Op) -> bool,
        ExitReason,
        &dyn Fn(&Graph) -> bool,
    ); 6] = [
        ("get", &get, ExitReason::Detached, &loads),
        ("put", &put, ExitReason::Detached, &stores),
        ("put", &put, ExitReason::OutOfBounds, &stores),
        ("any", &get, ExitReason::OutOfBounds, &stub),
        ("any", &get, ExitReason::Detached, &stub),
        ("mixed", &get, ExitReason::OutOfBounds, &stub),
    ];
    for (name, site, reason, before) in cases {
        let mut engine = vm_engine(source);
        let (chunk, pc, graph) = flagged_element_graph(&mut engine, name, site, reason, before);
        let text = dump(&graph, &chunk);
        assert!(
            nodes(&graph, is_effect)
                .iter()
                .any(|node| node.pc as usize == pc),
            "{name} {reason:?}: {text}"
        );
        assert_eq!(
            declines(&graph),
            [(pc, reason)],
            "{name} {reason:?}: {text}"
        );
        assert_eq!(graph.declines[0].form, "the generic access", "{name}");
    }
    // A Hole flag at a site of typed arrays and Arrays reads the words' holes.
    let mut engine = vm_engine(source);
    let (chunk, pc, graph) =
        flagged_element_graph(&mut engine, "mixed", get, ExitReason::Hole, stub);
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::LoadElement(Holes::Undefined)),
        1,
        "{text}"
    );
    assert_eq!(declines(&graph), [(pc, ExitReason::Hole)], "{text}");
}

/// The graph of `name` (an element site at its one `site` operation) with `reason` flagged
/// there, after its unflagged graph passed `before`.
fn flagged_element_graph(
    engine: &mut Engine,
    name: &str,
    site: impl Fn(&Op) -> bool,
    reason: ExitReason,
    before: impl Fn(&Graph) -> bool,
) -> (Rc<Chunk>, usize, Graph) {
    let (chunk, graph) = graph_of(engine, name);
    let pc = pc_of(&chunk, &site);
    assert!(before(&graph), "{name} unflagged: {}", dump(&graph, &chunk));
    assert!(graph.declines.is_empty(), "{name}");
    flag(&chunk, pc, reason, false, 0);
    let graph = graph_for(&chunk, name);
    (chunk, pc, graph)
}

#[test]
fn flagged_holes_and_reads_past_used_widen_an_element_load() {
    // JSC widens an array access's mode through the profile its exits update (getArrayMode,
    // DFGByteCodeParser.cpp:1346-1360): a Hole flag reads holes, an OutOfBounds flag past the
    // used words, each undefined under the prototype proof.
    let source = "function get(a, i) { return a[i]; } function past(a, i) { return a[i]; }
         for (var k = 0; k < 20; k++) { get([1, 2], 1); past([1, 2], 0); }";
    let is_get = |op: &Op| matches!(op, Op::GetElemLocal(_));
    let exits_on_holes = |graph: &Graph| {
        count(graph, |op| *op == NodeOp::LoadElement(Holes::Exit)) == 1
            && count(graph, |op| *op == NodeOp::CheckBounds) == 1
    };
    for (name, reason, form, load) in [
        (
            "get",
            ExitReason::Hole,
            "the hole read as undefined",
            NodeOp::LoadElement(Holes::Undefined),
        ),
        (
            "past",
            ExitReason::OutOfBounds,
            "the read past the used words",
            NodeOp::LoadElementOrUndefined,
        ),
    ] {
        let mut engine = vm_engine(source);
        let (chunk, pc, graph) =
            flagged_element_graph(&mut engine, name, is_get, reason, exits_on_holes);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert_eq!(count(&graph, |op| *op == load), 1, "{name}: {text}");
        assert_eq!(declines(&graph), [(pc, reason)], "{name}: {text}");
        assert_eq!(graph.declines[0].form, form, "{name}");
    }
}

#[test]
fn flagged_element_checks_take_the_generic_access() {
    // A receiver, key, bound or growth the site's checks refuted: the effect (JSC's parser gives
    // an array access up at its BadIndexingType and OutOfBounds exit sites).
    let source = "function get(a, i) { return a[i]; } function put(a, i, v) { a[i] = v; }
         function inc(a, i) { a[i]++; } function len(a) { return a.length; }
         function add(a, i, v) { a[i] += v; }
         for (var k = 0; k < 20; k++) {
           get([1, 2], 1); put([1, 2], 1, k); inc([1, 2], 0); len([1]); add([1, 2], 0, 1);
         }";
    let get = |op: &Op| matches!(op, Op::GetElemLocal(_));
    let put = |op: &Op| matches!(op, Op::SetElemLocalDrop(_));
    let inc = |op: &Op| matches!(op, Op::UpdateElem(_));
    let len = |op: &Op| matches!(op, Op::GetPropLocal(..));
    let to_key = |op: &Op| matches!(op, Op::ToPropKeyLocal(_));
    let loads = |graph: &Graph| count(graph, |op| matches!(op, NodeOp::LoadElement(_))) == 1;
    let stores = |graph: &Graph| count(graph, |op| matches!(op, NodeOp::StoreElement(_))) == 1;
    let lengths = |graph: &Graph| count(graph, |op| *op == NodeOp::LoadArrayLength) == 1;
    let cases: [(
        &str,
        &dyn Fn(&Op) -> bool,
        ExitReason,
        &dyn Fn(&Graph) -> bool,
    ); 14] = [
        ("get", &get, ExitReason::WrongElements, &loads),
        ("get", &get, ExitReason::NotObject, &loads),
        ("get", &get, ExitReason::NotNumber, &loads),
        ("get", &get, ExitReason::NotInt32, &loads),
        ("put", &put, ExitReason::WrongElements, &stores),
        ("put", &put, ExitReason::OutOfBounds, &stores),
        ("put", &put, ExitReason::Hole, &stores),
        // The last reason, whose bit a 16-bit mask lacked (it aliased NotNumber's).
        ("put", &put, ExitReason::CouldNotGrow, &stores),
        ("inc", &inc, ExitReason::Hole, &stores),
        ("inc", &inc, ExitReason::OutOfBounds, &stores),
        ("inc", &inc, ExitReason::WrongElements, &stores),
        ("len", &len, ExitReason::WrongElements, &lengths),
        ("len", &len, ExitReason::NotObject, &lengths),
        ("add", &to_key, ExitReason::NotObject, &loads),
    ];
    for (name, site, reason, before) in cases {
        let mut engine = vm_engine(source);
        let (chunk, pc, graph) = flagged_element_graph(&mut engine, name, site, reason, before);
        let text = dump(&graph, &chunk);
        assert!(
            nodes(&graph, is_effect)
                .iter()
                .any(|node| node.pc as usize == pc),
            "{name} {reason:?}: {text}"
        );
        assert_eq!(
            declines(&graph),
            [(pc, reason)],
            "{name} {reason:?}: {text}"
        );
        assert_eq!(graph.declines[0].form, "the generic access", "{name}");
        // The flag is no other reason's.
        let snapshot = Snapshot::take(&chunk);
        assert!(
            !snapshot.declines(pc, ExitReason::NotNumber) || reason == ExitReason::NotNumber,
            "{name} {reason:?}"
        );
    }
}

// ---- free names (stage 2, step 4) ---------------------------------------------------------------

/// The pcs of `chunk`'s free-name reads of `name` (LoadName and its call and splice forms).
fn name_reads(chunk: &Chunk, name: &str) -> Vec<usize> {
    (0..chunk.jit_ops().len())
        .filter(|&pc| match chunk.jit_ops()[pc] {
            Op::LoadName(n, _)
            | Op::LoadNameForCall(n, _)
            | Op::LoadNameIn(n, _, _)
            | Op::LoadNameForCallIn(n, _, _) => chunk.jit_name(n) == name,
            _ => false,
        })
        .collect()
}

/// The name cache of the free-name operation at `pc`.
fn name_cache(chunk: &Chunk, pc: usize) -> u32 {
    match chunk.jit_ops()[pc] {
        Op::LoadName(_, cache)
        | Op::LoadNameForCall(_, cache)
        | Op::LoadNameIn(_, cache, _)
        | Op::LoadNameForCallIn(_, cache, _)
        | Op::StoreNameCached(_, cache)
        | Op::UpdateNameCached(_, cache, _) => cache,
        ref op => panic!("no free-name operation at pc {pc}: {op:?}"),
    }
}

#[test]
fn free_name_sites_have_views_and_their_caches_are_evidence() {
    // `k` has a static address; `missing` never ran (its cache is empty); `toString` ran and its
    // fill was refused (the global object's prototype holds it); `Math` ran and filled.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function mk() { let k = 3; return function (c) { if (c) { return missing + k; }
           return Math === toString ? 0 : k; }; }
         var inner = mk();
         for (var i = 0; i < 20; i++) inner(false);",
    );
    let chunk = chunk_of(&mut engine, "inner");
    let snapshot = Snapshot::take(&chunk);
    let [k_read, k_read2] = name_reads(&chunk, "k")[..] else {
        panic!("two reads of k: {:?}", chunk.jit_ops())
    };
    for pc in [k_read, k_read2] {
        let super::snapshot::NameView::Static { address, base } = snapshot.name(pc) else {
            panic!("k's static address: {:?}", snapshot.name(pc))
        };
        // The closure makes no record of its own: `k`'s record is the running one.
        assert_eq!(
            (address.hops, address.lexical, address.mutable),
            (0, true, true)
        );
        assert_eq!(base, super::snapshot::Base::Running);
        assert_eq!(
            snapshot.ran(pc),
            Ran::Unknown,
            "a static read is no evidence"
        );
    }
    let missing = name_reads(&chunk, "missing")[0];
    assert_eq!(snapshot.name(missing), super::snapshot::NameView::Empty);
    assert_eq!(snapshot.ran(missing), Ran::No);
    let refused = name_reads(&chunk, "toString")[0];
    assert_eq!(snapshot.name(refused), super::snapshot::NameView::Refused);
    assert_eq!(snapshot.ran(refused), Ran::Yes);
    assert_eq!(
        chunk.jit_name_ic(name_cache(&chunk, refused)).env,
        crate::bytecode::REFUSED_NAME_IC
    );
    let math = name_reads(&chunk, "Math")[0];
    assert_eq!(snapshot.ran(math), Ran::Yes);
    assert!(!chunk.jit_name_ic(name_cache(&chunk, math)).unfilled());
    // The block whose only evidence is the empty cache never ran: an exit at its first pc.
    let graph = graph_for(&chunk, "inner");
    let text = dump(&graph, &chunk);
    let exits = nodes(&graph, |op| *op == NodeOp::Exit(ExitReason::NeverRan));
    assert_eq!(exits.len(), 1, "{text}");
    let block_pc = graph.state(exits[0].state.expect("a state")).pc as usize;
    assert!(block_pc <= missing, "{text}");
    assert!(
        graph
            .nodes
            .iter()
            .filter(|node| node.pc as usize == missing)
            .all(|node| matches!(node.op, NodeOp::Exit(_))),
        "nothing of the block is built: {text}"
    );
    // A fill moves the fingerprint: the site's class changed.
    let print = fingerprint(&chunk);
    eval(&mut engine, "var missing = 1; inner(true);");
    assert_ne!(fingerprint(&chunk), print);
    assert_eq!(Snapshot::take(&chunk).ran(missing), Ran::Yes);
}

#[test]
fn a_deep_cache_seeds_a_second_stage_with_the_refused_fill_mark() {
    // A deep chain's descriptor stays with its chunk: the recompile's site starts with the mark,
    // since it ran, and not empty, which would make its block an exit.
    let mut engine = engine(Tier::Bytecode);
    eval(&mut engine, "function readDeep() { return outer; }");
    let genv = engine.interp.global_env;
    let value = engine
        .interp
        .get_var("readDeep", &genv)
        .ok()
        .expect("readDeep");
    let ValueView::Obj(function) = value.view() else {
        panic!("a function")
    };
    let func = match &function.borrow().kind {
        ObjectKind::User(user) => user.func.clone(),
        _ => panic!("an author function"),
    };
    let chunk = crate::bytecode::compile(&func).expect("readDeep compiles");
    let pc = name_reads(&chunk, "outer")[0];
    let (name, cache) = match chunk.jit_ops()[pc] {
        Op::LoadName(name, cache) => (name, cache),
        ref op => panic!("{op:?}"),
    };
    // The records live only in this helper frame (the collector scans the stack).
    crate::gc_test::in_frame(|| {
        let scope = |parent, key: &str| {
            let env = crate::interpreter::new_scope(parent);
            env.borrow_mut().vars.insert(
                key,
                crate::interpreter::Binding::data(Value::Num(7.0), true, true),
            );
            env
        };
        let holder = scope(None, "outer");
        let middle = scope(Some(holder), "middle");
        let first = scope(Some(middle), "local");
        assert!(chunk
            .load_name_ic(&mut engine.interp, &first, name, cache)
            .is_ok());
        assert_eq!(
            chunk.jit_name_ic(cache).env,
            crate::bytecode::lexical_cache::DEEP_NAME_IC
        );
        let second = crate::bytecode::compile_with_inlines(&func, &Default::default(), &chunk)
            .expect("the recompile");
        let pc = name_reads(&second, "outer")[0];
        assert_eq!(
            second.jit_name_ic(name_cache(&second, pc)).env,
            crate::bytecode::REFUSED_NAME_IC
        );
        assert_eq!(Snapshot::take(&second).ran(pc), Ran::Yes);
    });
}

/// The static address of the free-name site at `pc`.
fn static_address(snapshot: &Snapshot, pc: usize) -> crate::bytecode::StaticName {
    match snapshot.name(pc) {
        super::snapshot::NameView::Static { address, .. } => address,
        other => panic!("no static address at pc {pc}: {other:?}"),
    }
}

/// The loads of `graph` from a record's slot `slot`.
fn binding_loads(graph: &Graph, slot: u16) -> usize {
    count(graph, |op| *op == NodeOp::LoadBinding { slot })
}

#[test]
fn static_reads_walk_their_hops_without_guards() {
    // `g` makes no record: x2 is in its running environment, x1 one record up, x0 two.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function f0() { var x0 = 1; function f1() { var x1 = 2; function f2() { var x2 = 3;
           return function () { return x0 + x1 + x2; }; } return f2(); } return f1(); }
         var g = f0();
         for (var i = 0; i < 20; i++) g();",
    );
    let (chunk, graph) = graph_of(&mut engine, "g");
    let text = dump(&graph, &chunk);
    let snapshot = Snapshot::take(&chunk);
    let hops: Vec<u16> = ["x0", "x1", "x2"]
        .iter()
        .map(|name| static_address(&snapshot, name_reads(&chunk, name)[0]).hops)
        .collect();
    assert_eq!(hops, [2, 1, 0], "{text}");
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    // One running environment at the entry, two parent links (x1's serves x0's walk), a load
    // a binding, and no guard: no check of an environment, no TDZ test of a `var`.
    assert_eq!(count(&graph, |op| *op == NodeOp::Env), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::EnvParent), 2, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadBinding { .. })),
        3,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| *op == NodeOp::CheckInitialized),
        0,
        "{text}"
    );
    let env = nodes(&graph, |op| *op == NodeOp::Env)[0];
    assert_eq!(env.block, graph.entry.expect("an entry"), "hoisted: {text}");
    // The words are checked Numbers for the adds, as any loaded word is.
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckNumber), 3, "{text}");
}

#[test]
fn a_lexical_read_is_checked_once_per_record() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function mk() { let x = 1; return function (c) { var t = x; if (c) { t = t + x; }
           return t + x; }; }
         var r = mk();
         for (var i = 0; i < 20; i++) { r(true); r(false); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "r");
    let text = dump(&graph, &chunk);
    let checks = nodes(&graph, |op| *op == NodeOp::CheckInitialized);
    assert_eq!(checks.len(), 1, "{text}");
    assert_eq!(checks[0].reason, Some(ExitReason::Tdz), "{text}");
    // The check exits before the read, with the read's operands (none) on the stack.
    let first = name_reads(&chunk, "x")[0];
    let state = graph.state(checks[0].state.expect("a state"));
    assert_eq!(state.pc as usize, first, "{text}");
    assert_eq!(
        state.stack.len(),
        crate::jit_ir::Cfg::build(&chunk)
            .unwrap()
            .stack_depth_at(first)
            .unwrap(),
        "{text}"
    );
    assert_eq!(count(&graph, is_effect), 0, "{text}");
}

#[test]
fn a_const_is_kept_across_a_call_and_a_var_reloaded_after_one() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function mk() { const k = 2; var v = 3; function noop() {}
           return function () { var a = k + v; noop(); return a + k + v; }; }
         var r = mk();
         for (var i = 0; i < 20; i++) r();",
    );
    let (chunk, graph) = graph_of(&mut engine, "r");
    let text = dump(&graph, &chunk);
    let snapshot = Snapshot::take(&chunk);
    let k = static_address(&snapshot, name_reads(&chunk, "k")[0]);
    let v = static_address(&snapshot, name_reads(&chunk, "v")[0]);
    assert!(!k.mutable && v.mutable, "{text}");
    assert_eq!(count(&graph, is_effect), 1, "the call: {text}");
    // `k`'s initialized value is a value fact; `v`'s a heap fact, which the call ends.
    assert_eq!(binding_loads(&graph, k.slot), 1, "{text}");
    assert_eq!(binding_loads(&graph, v.slot), 2, "{text}");
    // A `const`'s TDZ is tested once, a `var` has none.
    assert_eq!(
        count(&graph, |op| *op == NodeOp::CheckInitialized),
        1,
        "{text}"
    );
}

#[test]
fn a_free_name_called_has_an_undefined_receiver() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function mk() { var f = function () { return 1; }; return function () { return f(); }; }
         var r = mk();
         for (var i = 0; i < 20; i++) r();",
    );
    let (chunk, graph) = graph_of(&mut engine, "r");
    let text = dump(&graph, &chunk);
    let call = nodes(&graph, is_effect);
    assert_eq!(call.len(), 1, "{text}");
    let receiver = graph.node(call[0].inputs[0]);
    assert_eq!(receiver.op, NodeOp::ConstTagged(PACK_UNDEFINED), "{text}");
    let callee = graph.node(call[0].inputs[1]);
    assert!(matches!(callee.op, NodeOp::LoadBinding { .. }), "{text}");
}

#[test]
fn a_captured_binding_is_read_at_its_activations_slot() {
    // `c` lives in capt's own record (`get` captures it): LoadCap, at hops 0 from the running
    // environment, after the call (which ends the initialization's word as a fact), its TDZ
    // proven by the initialization before it.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function capt(n) { let c = n + 1; function get() { return c; } return get() + c * 2; }
         for (var i = 0; i < 20; i++) capt(i);",
    );
    let (chunk, graph) = graph_of(&mut engine, "capt");
    let text = dump(&graph, &chunk);
    if !chunk.jit_fixed_activation() {
        return; // activation plans off (TRUST_JS_NO_ACTIVATION_PLANS): LoadCap keeps its effect
    }
    let load = pc_of(&chunk, |op| matches!(op, Op::LoadCap(_)));
    let address = static_address(&Snapshot::take(&chunk), load);
    assert_eq!(address.hops, 0, "{text}");
    let loads = nodes(&graph, |op| {
        *op == NodeOp::LoadBinding { slot: address.slot }
    });
    assert_eq!(loads.len(), 1, "{text}");
    assert_eq!(graph.node(loads[0].inputs[0]).op, NodeOp::Env, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::CheckInitialized),
        0,
        "{text}"
    );
    assert!(
        nodes(&graph, is_effect)
            .iter()
            .all(|node| node.pc as usize != load),
        "{text}"
    );
}

#[test]
fn a_splices_static_reads_start_from_its_pinned_environments_holder() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function mk() { var base = 10; return function (v) { return v + base; }; }
         var add = mk();
         function caller(x) { return add(x); }
         function drive(n) { var t = 0; for (var i = 0; i < n; i++) t += caller(i); return t; }
         drive(150);",
    );
    let Some(transformed) = second_stage(&mut engine, "caller", "drive(150)") else {
        return;
    };
    let graph = graph_for(&transformed, "caller's second stage");
    let text = dump(&graph, &transformed);
    let Some(&read) = name_reads(&transformed, "base").first() else {
        panic!("add is spliced: {text}")
    };
    assert!(
        matches!(transformed.jit_ops()[read], Op::LoadNameIn(..)),
        "{text}"
    );
    // `add` makes no record: its [[Environment]], mk's activation, holds `base`.
    let value = engine
        .interp
        .get_var("add", &engine.interp.global_env.clone())
        .ok()
        .expect("add");
    let holder = match &value.as_obj().expect("a function").borrow().kind {
        ObjectKind::User(user) => crate::interpreter::Env::as_ptr(&user.env) as u64,
        _ => panic!("an author function"),
    };
    let snapshot = Snapshot::take(&transformed);
    assert_eq!(
        snapshot.name(read),
        super::snapshot::NameView::Static {
            address: static_address(&snapshot, read),
            base: super::snapshot::Base::Holder(holder as usize),
        },
        "{text}"
    );
    let loads = nodes(&graph, |op| matches!(op, NodeOp::LoadBinding { .. }));
    assert_eq!(loads.len(), 1, "{text}");
    assert_eq!(
        graph.node(loads[0].inputs[0]).op,
        NodeOp::ConstEnv(holder),
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::Env), 0, "{text}");
}

/// The free-name stores and updates of `name` in `chunk` (StoreNameCached, UpdateNameCached).
fn name_writes(chunk: &Chunk, name: &str) -> Vec<usize> {
    (0..chunk.jit_ops().len())
        .filter(|&pc| match chunk.jit_ops()[pc] {
            Op::StoreNameCached(n, _) | Op::UpdateNameCached(n, _, _) => chunk.jit_name(n) == name,
            _ => false,
        })
        .collect()
}

#[test]
fn a_static_store_writes_in_place_and_its_word_is_the_bindings_value() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function mk() { var count = 0; var last = null;
           return function (o) { count = count + 1; last = o; return count; }; }
         var r = mk();
         for (var i = 0; i < 20; i++) r({});",
    );
    let (chunk, graph) = graph_of(&mut engine, "r");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let snapshot = Snapshot::take(&chunk);
    let count_slot = static_address(&snapshot, name_writes(&chunk, "count")[0]).slot;
    let last_slot = static_address(&snapshot, name_writes(&chunk, "last")[0]).slot;
    let stores = nodes(&graph, |op| matches!(op, NodeOp::StoreBinding { .. }));
    assert_eq!(stores.len(), 2, "{text}");
    // `count`'s new value is the sum's word; the return reads it back from the fact, not the
    // record.
    assert_eq!(binding_loads(&graph, count_slot), 1, "{text}");
    assert_eq!(binding_loads(&graph, last_slot), 0, "{text}");
    let stored = stores
        .iter()
        .find(|node| node.op == NodeOp::StoreBinding { slot: count_slot })
        .expect("count's store");
    assert_eq!(
        graph.node(stored.inputs[1]).op,
        NodeOp::F64ToTagged,
        "{text}"
    );
    let ret = nodes(&graph, |op| *op == NodeOp::Return)[0];
    assert_eq!(ret.inputs[0], stored.inputs[1], "{text}");
}

#[test]
fn a_lexical_store_tests_the_old_words_tdz_once() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function mk() { let x = 0; return function (v) { x = v; x = v + 1; return x; }; }
         var r = mk();
         for (var i = 0; i < 20; i++) r(i);",
    );
    let (chunk, graph) = graph_of(&mut engine, "r");
    let text = dump(&graph, &chunk);
    let checks = nodes(&graph, |op| *op == NodeOp::CheckInitialized);
    assert_eq!(checks.len(), 1, "{text}");
    // It exits at the first store, its operand on the stack.
    let first = name_writes(&chunk, "x")[0];
    let state = graph.state(checks[0].state.expect("a state"));
    assert_eq!(state.pc as usize, first, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::StoreBinding { .. })),
        2,
        "{text}"
    );
    // The old word's load for the test, and none for the return.
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadBinding { .. })),
        1,
        "{text}"
    );
}

#[test]
fn static_updates_take_int32_or_f64_arithmetic_and_store_in_place() {
    // The VM's records say int32: int32 adds with overflow exits.
    let source = "function mk() { var n = 0; return function () { n++; return ++n; }; }
                  var r = mk();
                  for (var i = 0; i < 20; i++) r();";
    let mut vm = engine(Tier::Bytecode);
    eval(&mut vm, source);
    let (chunk, graph) = graph_of(&mut vm, "r");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let adds = nodes(&graph, |op| *op == NodeOp::I32Add);
    assert_eq!(adds.len(), 2, "{text}");
    assert!(
        adds.iter()
            .all(|add| add.reason == Some(ExitReason::Overflow)),
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::StoreBinding { .. })),
        2,
        "{text}"
    );
    // The second update reads the first's word: one load.
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadBinding { .. })),
        1,
        "{text}"
    );
    // The first stage's in-line updates record nothing: f64 arithmetic.
    let mut jit = engine(Tier::Jit);
    eval(&mut jit, source);
    let (chunk, graph) = graph_of(&mut jit, "r");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Add), 2, "{text}");
    assert_eq!(count(&graph, is_effect), 0, "{text}");
}

#[test]
fn captured_bindings_are_initialized_stored_and_updated_in_place() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function capt(n) { let c = n; function get() { return c; } c = c + 1; c++;
           return get() + c; }
         for (var i = 0; i < 20; i++) capt(i);",
    );
    let (chunk, graph) = graph_of(&mut engine, "capt");
    let text = dump(&graph, &chunk);
    if !chunk.jit_fixed_activation() {
        return; // activation plans off: the captured bindings keep their effects
    }
    let writes = |which: fn(&Op) -> bool| {
        (0..chunk.jit_ops().len())
            .filter(|&pc| which(&chunk.jit_ops()[pc]))
            .count()
    };
    assert_eq!(writes(|op| matches!(op, Op::StoreCapInit(_))), 1, "{text}");
    assert_eq!(writes(|op| matches!(op, Op::StoreCap(_))), 1, "{text}");
    assert_eq!(writes(|op| matches!(op, Op::UpdateCap(..))), 1, "{text}");
    // The initialization ends the TDZ: no test anywhere; the call reloads the binding.
    assert_eq!(
        count(&graph, |op| *op == NodeOp::CheckInitialized),
        0,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::StoreBinding { .. })),
        3,
        "{text}"
    );
    assert_eq!(
        count(&graph, is_effect),
        2,
        "the closure and the call: {text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadBinding { .. })),
        1,
        "the read after the call: {text}"
    );
}

#[test]
fn a_store_ends_its_slots_facts_on_every_record() {
    // `add`'s splices read `s` from their folded holder, mk's activation, which the caller's own
    // store reaches from its running environment: the second splice reads `s` again.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function mk() { var s = 0; function add(v) { return v + s; }
           return function (x) { var a = add(1); s = x; return a + add(2); }; }
         var caller = mk();
         function drive(n) { var t = 0; for (var i = 0; i < n; i++) t += caller(i); return t; }
         drive(150);",
    );
    let Some(transformed) = second_stage(&mut engine, "caller", "drive(150)") else {
        return;
    };
    let graph = graph_for(&transformed, "caller's second stage");
    let text = dump(&graph, &transformed);
    let reads = name_reads(&transformed, "s");
    if reads.len() != 2 {
        return; // add was not spliced twice
    }
    let snapshot = Snapshot::take(&transformed);
    let slot = static_address(&snapshot, reads[0]).slot;
    let from_holder = nodes(&graph, |op| *op == NodeOp::LoadBinding { slot })
        .into_iter()
        .filter(|node| matches!(graph.node(node.inputs[0]).op, NodeOp::ConstEnv(_)))
        .count();
    assert_eq!(from_holder, 2, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::StoreBinding { slot }),
        1,
        "{text}"
    );
}

/// The verified graph of `chunk` with its global names resolved in `engine`'s running realm, as
/// the driver's compiles take it.
fn graph_in(engine: &Engine, chunk: &Chunk, name: &str) -> Graph {
    let snapshot = Snapshot::take_in(chunk, Some(&engine.interp));
    let built = build(chunk, &snapshot)
        .unwrap_or_else(|refusal| panic!("{name}: {refusal:?}\n{:#?}", chunk.jit_ops()));
    let graph = built.graph;
    if let Err(error) = super::verify::verify(&graph, chunk, &built.cfg) {
        panic!("{name}: {error}\n{}", dump(&graph, chunk));
    }
    graph
}

fn is_global_check(op: &NodeOp) -> bool {
    matches!(
        op,
        NodeOp::CheckEnvIs { .. } | NodeOp::CheckCellIs { .. } | NodeOp::CheckGlobalEpoch { .. }
    )
}

#[test]
fn a_global_read_is_checked_once_per_effect_epoch_and_reloaded_after_one() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "var g = 5; function noop() {}
         function r() { var a = g + g; noop(); return a + g; }
         for (var i = 0; i < 20; i++) r();",
    );
    let chunk = chunk_of(&mut engine, "r");
    let graph = graph_in(&engine, &chunk, "r");
    let text = dump(&graph, &chunk);
    let snapshot = Snapshot::take_in(&chunk, Some(&engine.interp));
    let read = name_reads(&chunk, "g")[0];
    let super::snapshot::NameView::Global {
        location, epoch, ..
    } = snapshot.name(read)
    else {
        panic!("g's global view: {:?}", snapshot.name(read))
    };
    assert_eq!(snapshot.ran(read), Ran::Yes);
    // The realm's environment and object, checked once: value facts.
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckEnvIs { .. })),
        1,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckCellIs { .. })),
        1,
        "{text}"
    );
    // The epoch before the first read and after the call (a heap fact), each WrongName.
    let epochs = nodes(&graph, |op| matches!(op, NodeOp::CheckGlobalEpoch { .. }));
    assert_eq!(epochs.len(), 2, "{text}");
    assert!(
        epochs.iter().all(|check| check.op
            == NodeOp::CheckGlobalEpoch {
                epoch,
                proof: snapshot.name(read).proof()
            }
            && check.reason == Some(ExitReason::WrongName)),
        "{text}"
    );
    // `g`'s slot loaded before the call and after it: one load for `g + g`.
    let loads = nodes(
        &graph,
        |op| matches!(op, NodeOp::LoadField { location: at, .. } if *at == location),
    );
    assert_eq!(loads.len(), 2, "{text}");
    assert!(
        loads
            .iter()
            .all(|load| graph.node(load.inputs[0]).op == NodeOp::GlobalObject),
        "{text}"
    );
    assert_eq!(count(&graph, is_effect), 1, "the call: {text}");
    // Without the realm, the reads are the effects they were.
    let graph = graph_for(&chunk, "r without a realm");
    assert_eq!(count(&graph, is_global_check), 0);
    assert_eq!(count(&graph, is_effect), 5, "four reads and the call");
}

#[test]
fn a_global_called_has_an_undefined_receiver() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function one() { return 1; } function r() { return one(); }
         for (var i = 0; i < 20; i++) r();",
    );
    let chunk = chunk_of(&mut engine, "r");
    let graph = graph_in(&engine, &chunk, "r");
    let text = dump(&graph, &chunk);
    let call = nodes(&graph, is_effect);
    assert_eq!(call.len(), 1, "{text}");
    assert_eq!(
        graph.node(call[0].inputs[0]).op,
        NodeOp::ConstTagged(PACK_UNDEFINED),
        "{text}"
    );
    assert!(
        matches!(graph.node(call[0].inputs[1]).op, NodeOp::LoadField { .. }),
        "{text}"
    );
}

#[test]
fn a_global_the_compile_cannot_prove_stays_the_effect() {
    // An accessor (its fill refused), a property the global object's prototype holds (refused),
    // a global store, and a typeof: effects, with no check.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "Object.defineProperty(globalThis, 'acc', { get() { return 2; }, configurable: true });
         var plain = 1;
         function r() { plain = acc + valueOf.length; return typeof plain; }
         for (var i = 0; i < 20; i++) r();",
    );
    let chunk = chunk_of(&mut engine, "r");
    let graph = graph_in(&engine, &chunk, "r");
    let text = dump(&graph, &chunk);
    let snapshot = Snapshot::take_in(&chunk, Some(&engine.interp));
    for name in ["acc", "valueOf"] {
        assert_eq!(
            snapshot.name(name_reads(&chunk, name)[0]),
            super::snapshot::NameView::Refused,
            "{name}: {text}"
        );
    }
    assert_eq!(count(&graph, is_global_check), 0, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::GlobalObject), 0, "{text}");
}

#[test]
fn a_splices_global_read_has_its_guard_for_its_realm() {
    // `add`'s global read in a splice: the pinned [[Environment]] is the realm's global
    // environment, which the splice's guard proved, so no environment check; the object's and
    // the epoch's stay.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "var base = 10;
         function add(v) { return v + base; }
         function caller(x) { return add(x); }
         function drive(n) { var t = 0; for (var i = 0; i < n; i++) t += caller(i); return t; }
         drive(150);",
    );
    let Some(transformed) = second_stage(&mut engine, "caller", "drive(150)") else {
        return;
    };
    let graph = graph_in(&engine, &transformed, "caller's second stage");
    let text = dump(&graph, &transformed);
    let Some(&read) = name_reads(&transformed, "base").first() else {
        panic!("add is spliced: {text}")
    };
    let snapshot = Snapshot::take_in(&transformed, Some(&engine.interp));
    assert!(
        matches!(
            snapshot.name(read),
            super::snapshot::NameView::Global {
                search: super::snapshot::Search::Pinned,
                ..
            }
        ),
        "{:?}",
        snapshot.name(read)
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckEnvIs { .. })),
        // `add`'s own read, the call's callee, is the caller's static global.
        1,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckCellIs { .. })),
        1,
        "{text}"
    );
    assert!(
        nodes(&graph, |op| matches!(op, NodeOp::LoadField { .. }))
            .iter()
            .any(|load| load.pc as usize == read),
        "{text}"
    );
}

#[test]
fn the_global_property_epoch_moves_at_removals_kind_changes_and_shadowing_only() {
    let mut engine = engine(Tier::Bytecode);
    let epoch = |engine: &Engine| engine.interp.global.borrow().props.slot_epoch();
    let start = epoch(&engine);
    // Additions, value writes and attribute changes of a data property move nothing: a slot found
    // by its key stays that key's.
    eval(
        &mut engine,
        "globalThis.a = 1; var b = 2; globalThis.a = 3;
         Object.defineProperty(globalThis, 'a', { enumerable: false });
         for (var k = 0; k < 20; k++) globalThis['x' + k] = k;",
    );
    assert_eq!(epoch(&engine), start);
    // A removal closes its gap.
    eval(&mut engine, "delete globalThis.a;");
    assert_eq!(epoch(&engine), start + 1);
    // A var's removal fails (non-configurable): nothing moves.
    eval(&mut engine, "delete globalThis.b;");
    assert_eq!(epoch(&engine), start + 1);
    // A data property turned accessor, and back.
    eval(
        &mut engine,
        "Object.defineProperty(globalThis, 'c', { value: 1, configurable: true });
         Object.defineProperty(globalThis, 'c', { get() { return 1; }, configurable: true });",
    );
    assert_eq!(epoch(&engine), start + 2);
    eval(
        &mut engine,
        "Object.defineProperty(globalThis, 'c', { value: 2, configurable: true });",
    );
    assert_eq!(epoch(&engine), start + 3);
    // A later script's lexical that shadows a property; one that shadows none moves nothing.
    eval(&mut engine, "globalThis.d = 1;");
    eval(&mut engine, "let d = 2;");
    assert_eq!(epoch(&engine), start + 4);
    eval(&mut engine, "let e = 1;");
    assert_eq!(epoch(&engine), start + 4);
}

#[test]
fn a_script_level_lexical_is_read_under_its_caches_proof_without_a_tdz_test() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "let counter = 5; const fixed = 7;
         function r() { return counter + fixed + counter; }
         for (var i = 0; i < 20; i++) r();",
    );
    let chunk = chunk_of(&mut engine, "r");
    let graph = graph_in(&engine, &chunk, "r");
    let text = dump(&graph, &chunk);
    let snapshot = Snapshot::take_in(&chunk, Some(&engine.interp));
    for name in ["counter", "fixed"] {
        assert!(
            matches!(
                snapshot.name(name_reads(&chunk, name)[0]),
                super::snapshot::NameView::GlobalLexical { .. }
            ),
            "{name}: {:?}",
            snapshot.name(name_reads(&chunk, name)[0])
        );
    }
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckEnvIs { .. })),
        1,
        "{text}"
    );
    let generations = nodes(&graph, |op| matches!(op, NodeOp::CheckEnvGeneration { .. }));
    assert_eq!(generations.len(), 1, "{text}");
    assert_eq!(generations[0].reason, Some(ExitReason::WrongName), "{text}");
    // `counter`'s second read is the first's word.
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadGlobalLexical { .. })),
        2,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| *op == NodeOp::CheckInitialized),
        0,
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::GlobalObject), 0, "{text}");
}

/// The frame slot of `chunk`'s local `name`.
fn slot_named(chunk: &Chunk, name: &str) -> u16 {
    chunk
        .jit_slot_names()
        .iter()
        .position(|slot| &**slot == name)
        .unwrap_or_else(|| panic!("no slot {name}: {:?}", chunk.jit_slot_names())) as u16
}

/// The one loop header of `graph`, and the pc it starts at.
fn the_loop(graph: &Graph) -> u32 {
    let headers: Vec<&BlockData> = graph
        .blocks
        .iter()
        .filter(|block| block.loop_header)
        .collect();
    assert_eq!(headers.len(), 1, "one loop");
    headers[0].first_pc
}

#[test]
fn an_am3_loop_whose_names_elements_and_arithmetic_lower_keeps_its_invariant_slots() {
    // Crypto's am3 (its receivers read once before the loop here): with its element accesses,
    // its int32 arithmetic and its names lowered, the loop has no effect, which the prepass now
    // knows, so its invariant slots keep their values through the header: no store before it and
    // no reload in it, and each receiver's elements check stays where the read before the loop
    // made it.
    let mut engine = engine(Tier::Bytecode);
    eval(
        &mut engine,
        "function am3(a, i, x, w, j, c, n) {
           var first = a[i] | 0; var w0 = w[j] | 0;
           var xl = x & 0x3fff, xh = x >> 14;
           while (--n >= 0) {
             var l = a[i] & 0x3fff;
             var h = a[i++] >> 14;
             var m = xh * l + h * xl;
             l = xl * l + ((m & 0x3fff) << 14) + w[j] + c;
             c = (l >> 28) + (m >> 14) + xh * h;
             w[j++] = l & 0xfffffff;
           }
           return c + first + w0;
         }
         var A = [], W = [];
         for (var k = 0; k < 64; k++) { A.push(k * 977 & 0xfffffff); W.push(0); }
         for (var r = 0; r < 20; r++) am3(A, 0, 12345, W, 0, 0, 32);",
    );
    let (chunk, graph) = graph_of(&mut engine, "am3");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let header = the_loop(&graph);
    for name in ["a", "w", "xl", "xh"] {
        let slot = slot_named(&chunk, name);
        assert_eq!(
            count(&graph, |op| *op == NodeOp::StoreSlot(slot)),
            0,
            "{name} is stored: {text}"
        );
        assert_eq!(
            count(&graph, |op| *op == NodeOp::FrameSlot(slot)),
            0,
            "{name} is reloaded: {text}"
        );
    }
    let checks = nodes(&graph, |op| matches!(op, NodeOp::CheckElements(_)));
    assert_eq!(checks.len(), 2, "one a receiver: {text}");
    assert!(
        checks.iter().all(|check| check.pc < header),
        "before the loop: {text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckObject), 2, "{text}");
}

#[test]
fn a_loop_whose_only_names_are_lowered_has_no_effect() {
    let mut engine = engine(Tier::Bytecode);
    eval(
        &mut engine,
        "function mk() { const k = 3; var scale = 2;
           return function (n, bias) { var s = 0;
             for (var i = 0; i < n; i++) s = s + k * scale + bias + i;
             return s; }; }
         var f = mk();
         for (var r = 0; r < 20; r++) f(10, 1);",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    the_loop(&graph);
    let bias = slot_named(&chunk, "bias");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::FrameSlot(bias)),
        0,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| *op == NodeOp::StoreSlot(bias)),
        0,
        "{text}"
    );
}

#[test]
fn a_flagged_tdz_site_takes_the_effect() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function mk() { let x = 1; return function () { return x + 1; }; }
         var r = mk();
         for (var i = 0; i < 20; i++) r();",
    );
    let (chunk, graph) = graph_of(&mut engine, "r");
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckInitialized), 1);
    let read = name_reads(&chunk, "x")[0];
    flag(&chunk, read, ExitReason::Tdz, false, 0);
    let graph = graph_for(&chunk, "r flagged");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| *op == NodeOp::CheckInitialized),
        0,
        "{text}"
    );
    assert!(
        nodes(&graph, is_effect)
            .iter()
            .any(|node| node.pc as usize == read),
        "{text}"
    );
    assert_eq!(declines(&graph), [(read, ExitReason::Tdz)], "{text}");
}

#[test]
fn a_flagged_global_read_is_checked_again_where_its_proof_moved() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "var g = 5; function r() { return g + 1; } for (var i = 0; i < 20; i++) r();",
    );
    let chunk = chunk_of(&mut engine, "r");
    let read = name_reads(&chunk, "g")[0];
    let proof = Snapshot::take_in(&chunk, Some(&engine.interp))
        .name(read)
        .proof();
    // A discarded check of another proof (the epoch moved since): checked again, no decline.
    flag(
        &chunk,
        read,
        ExitReason::WrongName,
        false,
        proof.wrapping_add(1),
    );
    let graph = graph_in(&engine, &chunk, "r flagged, another proof");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckGlobalEpoch { .. })),
        1,
        "{text}"
    );
    assert!(graph.declines.is_empty(), "{text}");
    // Of this proof: the effect.
    flag(&chunk, read, ExitReason::WrongName, false, proof);
    let graph = graph_in(&engine, &chunk, "r flagged, its proof");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_global_check), 0, "{text}");
    assert_eq!(declines(&graph), [(read, ExitReason::WrongName)], "{text}");
}

#[test]
fn an_operand_a_lowered_form_cannot_take_exits_where_the_rules_lower_it() {
    // `'x' * 2` never ran: its record is empty, so the rules lower it (a loop around it would be
    // free of effects); its constant operand is no Number, so the builder exits there.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function f(c) { var t = 1; if (c) { t = 'x' * 2; } return t; }
         for (var i = 0; i < 20; i++) f(false);",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    let text = dump(&graph, &chunk);
    let mul = pc_of(&chunk, |op| matches!(op, Op::Mul));
    let exits = nodes(&graph, |op| *op == NodeOp::Exit(ExitReason::NotNumber));
    assert_eq!(exits.len(), 1, "{text}");
    assert_eq!(exits[0].pc as usize, mul, "{text}");
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    // Once its NotNumber is flagged, the operation is the effect again.
    flag(&chunk, mul, ExitReason::NotNumber, false, 0);
    let graph = graph_for(&chunk, "f flagged");
    let text = dump(&graph, &chunk);
    assert!(
        nodes(&graph, is_effect)
            .iter()
            .any(|node| node.pc as usize == mul),
        "{text}"
    );
}

// ---- untagged phis (stage 2, step 6) ------------------------------------------------------------

/// The phis of `graph`'s blocks, by block.
fn phis_of(graph: &Graph) -> Vec<&Node> {
    graph
        .blocks
        .iter()
        .flat_map(|block| block.phis.iter().map(|&phi| graph.node(phi)))
        .collect()
}

/// The one phi of `graph` that merges `chunk`'s local `name` at the block that starts at `pc`
/// (a phi a state at that pc names for the slot).
fn phi_of_slot(graph: &Graph, chunk: &Chunk, name: &str, pc: u32) -> NodeIx {
    let slot = slot_named(chunk, name) as usize;
    let block = graph
        .blocks
        .iter()
        .find(|block| block.first_pc == pc && !block.phis.is_empty())
        .unwrap_or_else(|| panic!("a join at pc {pc}"));
    graph
        .states
        .iter()
        .filter(|state| state.pc == pc)
        .find_map(|state| match state.slots[slot] {
            SlotSource::Value(value) if block.phis.contains(&value) => Some(value),
            _ => None,
        })
        .or_else(|| {
            // A join without a state at its pc: its phis' order follows the slots'.
            block.phis.iter().copied().find(|&phi| {
                graph
                    .nodes
                    .iter()
                    .any(|node| node.op == NodeOp::StoreSlot(slot as u16) && node.inputs[0] == phi)
            })
        })
        .unwrap_or_else(|| panic!("{name}'s phi at pc {pc}"))
}

#[test]
fn an_accumulator_of_halves_is_an_f64_phi_from_the_constant_zero() {
    // `x` starts at the int32 constant 0 and adds 0.5: an F64 phi whose entry input is the
    // constant as an f64 (inputs only widen). With the VM's records `i` is an I32 phi.
    let mut vm = vm_engine(
        "function acc(n) { var x = 0; for (var i = 0; i < n; i++) x += 0.5; return x; }
         for (var k = 0; k < 20; k++) acc(10);",
    );
    let (chunk, graph) = graph_of(&mut vm, "acc");
    let text = dump(&graph, &chunk);
    let header = the_loop(&graph);
    let x = graph.node(phi_of_slot(&graph, &chunk, "x", header));
    assert_eq!(x.rep, Rep::F64, "{text}");
    assert_eq!(graph.node(x.inputs[0]).op, NodeOp::ConstF64(0), "{text}");
    assert_eq!(graph.node(x.inputs[1]).op, NodeOp::F64Add, "{text}");
    let i = graph.node(phi_of_slot(&graph, &chunk, "i", header));
    assert_eq!(i.rep, Rep::I32, "{text}");
    assert_eq!(graph.node(i.inputs[0]).op, NodeOp::ConstI32(0), "{text}");
    // No phi is checked or boxed in the loop.
    assert_eq!(
        count(&graph, |op| *op == NodeOp::CheckNumber),
        1,
        "n's: {text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::I32ToTagged), 0, "{text}");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::F64ToTagged),
        1,
        "the return's: {text}"
    );
}

#[test]
fn a_join_of_an_int32_and_an_f64_is_an_f64_phi_the_int32_widened_in_its_predecessor() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function join(a, c) { var x; if (c) { x = a | 0; } else { x = a * 0.5; } return x + 1; }
         for (var i = 0; i < 20; i++) { join(i, true); join(i, false); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "join");
    let text = dump(&graph, &chunk);
    let phis = phis_of(&graph);
    assert_eq!(phis.len(), 1, "{text}");
    let phi = phis[0];
    assert_eq!(phi.rep, Rep::F64, "{text}");
    let preds = &graph.block(phi.block).preds;
    let widened: Vec<&Node> = phi
        .inputs
        .iter()
        .map(|&input| graph.node(input))
        .filter(|input| input.op == NodeOp::I32ToF64)
        .collect();
    assert_eq!(widened.len(), 1, "{text}");
    assert_eq!(graph.node(widened[0].inputs[0]).op, NodeOp::I32Or, "{text}");
    assert!(
        preds.contains(&widened[0].block),
        "in its predecessor: {text}"
    );
    assert!(
        phi.inputs
            .iter()
            .any(|&input| graph.node(input).op == NodeOp::F64Mul),
        "{text}"
    );
    // `x + 1` takes the phi as it is: no check (but `a`'s in each arm), no boxing of an input.
    let checks = nodes(&graph, |op| *op == NodeOp::CheckNumber);
    assert_eq!(checks.len(), 2, "{text}");
    assert!(
        checks
            .iter()
            .all(|check| graph.node(check.inputs[0]).op == NodeOp::EntrySlot(0)),
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::I32ToTagged), 0, "{text}");
}

#[test]
fn undefined_keeps_a_phi_tagged() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function u(c, a) { var x = void 0; if (c) { x = a * 2; } return x; }
         for (var i = 0; i < 20; i++) { u(true, i); u(false, i); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "u");
    let text = dump(&graph, &chunk);
    let phis = phis_of(&graph);
    assert_eq!(phis.len(), 1, "{text}");
    assert_eq!(phis[0].rep, Rep::Tagged, "{text}");
    assert!(
        phis[0]
            .inputs
            .iter()
            .any(|&input| graph.node(input).op == NodeOp::ConstTagged(PACK_UNDEFINED)),
        "{text}"
    );
    assert!(
        phis[0]
            .inputs
            .iter()
            .any(|&input| graph.node(input).op == NodeOp::F64ToTagged),
        "the f64 boxed in its predecessor: {text}"
    );
}

#[test]
fn a_back_edge_that_keeps_a_phis_value_does_not_vote() {
    // `x` reaches the header from the entry (1), from the `continue` (2.5) and from the body's
    // end, which keeps it: the phi is F64 and takes itself there.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function keep(n, c) { var x = 1, i = 0;
           while (i < n) { i = i + 1; if (c) { x = 2.5; continue; } } return x; }
         for (var k = 0; k < 20; k++) { keep(4, true); keep(4, false); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "keep");
    let text = dump(&graph, &chunk);
    let header = the_loop(&graph);
    let x = phi_of_slot(&graph, &chunk, "x", header);
    let node = graph.node(x);
    assert_eq!(node.rep, Rep::F64, "{text}");
    assert_eq!(node.inputs.len(), 3, "{text}");
    assert!(node.inputs.contains(&x), "its own back edge: {text}");
    let constants: Vec<NodeOp> = node
        .inputs
        .iter()
        .filter(|&&input| input != x)
        .map(|&input| graph.node(input).op.clone())
        .collect();
    assert_eq!(
        constants,
        [
            NodeOp::ConstF64(1f64.to_bits()),
            NodeOp::ConstF64(2.5f64.to_bits())
        ],
        "{text}"
    );
}

#[test]
fn an_inner_join_feeding_a_loop_phi_settles_with_it() {
    // The loop phi of `x` takes the inner join's phi on its back edge, and the join takes the
    // loop phi on one arm: undecided each until the other is, both untagged together.
    let source =
        "function inner(n) { var x = 0; for (var i = 0; i < n; i++) { if (i & 1) { x = x + 1; } }
           return x; }
         for (var k = 0; k < 20; k++) inner(9);";
    for (tier, rep) in [(Tier::Jit, Rep::F64), (Tier::Bytecode, Rep::I32)] {
        let mut engine = engine(tier);
        eval(&mut engine, source);
        let (chunk, graph) = graph_of(&mut engine, "inner");
        let text = dump(&graph, &chunk);
        let header = the_loop(&graph);
        let x = phi_of_slot(&graph, &chunk, "x", header);
        assert_eq!(graph.node(x).rep, rep, "{tier:?}: {text}");
        let join = graph
            .node(x)
            .inputs
            .iter()
            .copied()
            .find(|&input| graph.node(input).op == NodeOp::Phi)
            .unwrap_or_else(|| panic!("the inner join's phi: {text}"));
        assert_eq!(graph.node(join).rep, rep, "{tier:?}: {text}");
        assert!(graph.node(join).inputs.contains(&x), "{tier:?}: {text}");
        assert!(
            phis_of(&graph).iter().all(|phi| phi.rep != Rep::Tagged),
            "{tier:?}: {text}"
        );
    }
}

#[test]
fn slot_stores_and_effects_take_untagged_phis_and_a_call_keeps_them_in_callee_saved_homes() {
    // `s` (F64) and `i` (I32, by its `| 0`) are on the operand stack across the call, a generic
    // node: nothing is flushed before it (P3 stage 3 step 3a), its state holds `i` as its phi
    // (`s` is dead there), the call takes `i` as its operand as it is, and the allocator gives
    // each phi a callee-saved home (d8-d15, x23-x28).
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function withcall(n, g) { var s = 0.5, i = 0;
           for (var k = 0; k < n; k++) { s = s + (i + g(i)); i = (i + 1) | 0; } return s; }
         function id(v) { return v; }
         for (var r = 0; r < 20; r++) withcall(8, id);",
    );
    let (chunk, graph) = graph_of(&mut engine, "withcall");
    let text = dump(&graph, &chunk);
    let header = the_loop(&graph);
    let s = phi_of_slot(&graph, &chunk, "s", header);
    let i = phi_of_slot(&graph, &chunk, "i", header);
    assert_eq!(graph.node(s).rep, Rep::F64, "{text}");
    assert_eq!(graph.node(i).rep, Rep::I32, "{text}");
    let i_slot = slot_named(&chunk, "i");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::StoreSlot(_))),
        0,
        "{text}"
    );
    let call = nodes(&graph, is_effect);
    assert_eq!(call.len(), 1, "{text}");
    assert!(call[0].inputs.contains(&i), "{text}");
    let state = graph.state(call[0].state.expect("an effect's state"));
    assert_eq!(state.slots[i_slot as usize], SlotSource::Value(i), "{text}");
    assert_eq!(
        state.slots[slot_named(&chunk, "s") as usize],
        SlotSource::Dead,
        "{text}"
    );
    assert_eq!(count(&graph, |op| *op == NodeOp::I32ToTagged), 0, "{text}");
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    {
        use super::regalloc::{allocate, Home, ARM64};
        let alloc = allocate(&graph, &ARM64).expect("allocated");
        assert!(
            matches!(alloc.homes[s.index()], Home::Fp(r) if ARM64.fp_callee.contains(&r)),
            "{:?}: {text}",
            alloc.homes[s.index()]
        );
        assert!(
            matches!(alloc.homes[i.index()], Home::Gp(r) if ARM64.gp_callee.contains(&r)),
            "{:?}: {text}",
            alloc.homes[i.index()]
        );
    }
}

#[test]
fn a_phi_used_as_a_word_is_retagged_once_per_block() {
    // After the loop, `o.v = x` and `return x` take the F64 phi's word: one retagging.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function ret(o, n) { var x = 0; for (var i = 0; i < n; i++) x = x + 0.5; o.v = x;
           return x; }
         var p = {v: 0};
         for (var k = 0; k < 20; k++) ret(p, 4);",
    );
    let (chunk, graph) = graph_of(&mut engine, "ret");
    let text = dump(&graph, &chunk);
    let header = the_loop(&graph);
    let x = phi_of_slot(&graph, &chunk, "x", header);
    assert_eq!(graph.node(x).rep, Rep::F64, "{text}");
    let retags = nodes(&graph, |op| *op == NodeOp::F64ToTagged);
    assert_eq!(retags.len(), 1, "{text}");
    assert_eq!(retags[0].inputs, [x], "{text}");
    let retag = NodeIx(
        graph
            .nodes
            .iter()
            .position(|node| node.op == NodeOp::F64ToTagged)
            .expect("the retagging") as u32,
    );
    let store = nodes(&graph, |op| matches!(op, NodeOp::StoreField { .. }));
    assert_eq!(store.len(), 1, "{text}");
    assert_eq!(store[0].inputs[1], retag, "{text}");
    let ret = nodes(&graph, |op| *op == NodeOp::Return);
    assert_eq!(ret[0].inputs, [retag], "{text}");
}

#[test]
fn to_boolean_of_untagged_phis_tests_their_numbers() {
    // `a` (I32) and `b` (F64) after the loop: ToBoolean of each without a word.
    let mut vm = vm_engine(
        "function tb(n) { var a = 0, b = 0.5; for (var i = 0; i < n; i++) { a = a + 1; b = b * 2; }
           if (a) { if (b) { return 1; } return 2; } return 3; }
         for (var k = 0; k < 20; k++) { tb(3); tb(0); }",
    );
    let (chunk, graph) = graph_of(&mut vm, "tb");
    let text = dump(&graph, &chunk);
    let header = the_loop(&graph);
    let a = phi_of_slot(&graph, &chunk, "a", header);
    let b = phi_of_slot(&graph, &chunk, "b", header);
    assert_eq!(graph.node(a).rep, Rep::I32, "{text}");
    assert_eq!(graph.node(b).rep, Rep::F64, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::ToBoolean), 0, "{text}");
    assert!(
        graph
            .nodes
            .iter()
            .any(|node| node.op == NodeOp::I32Cmp(Cond::Ne) && node.inputs[0] == a),
        "{text}"
    );
    assert!(
        graph
            .nodes
            .iter()
            .any(|node| node.op == NodeOp::F64ToBoolean && node.inputs == [b]),
        "{text}"
    );
}

#[test]
fn the_verifier_refuses_a_phi_whose_inputs_are_not_its_representation() {
    let mut vm = vm_engine(
        "function acc(n) { var x = 0; for (var i = 0; i < n; i++) x += 0.5; return x; }
         for (var k = 0; k < 20; k++) acc(10);",
    );
    let (chunk, graph) = graph_of(&mut vm, "acc");
    let header = the_loop(&graph);
    let i = phi_of_slot(&graph, &chunk, "i", header);
    let x = phi_of_slot(&graph, &chunk, "x", header);
    // An I32 phi with an f64 input.
    let mut broken = graph.clone();
    broken.nodes[i.index()].inputs[0] = graph.node(x).inputs[0];
    let error = verify(&broken, &chunk).expect_err("mixed inputs");
    assert!(error.contains("for a I32 phi"), "{error}");
}

#[test]
fn am3s_counters_are_int32_phis_checked_once_on_the_loops_entry() {
    // n, i, j and c enter am3's loop from its parameters: each is an I32 phi whose entry input
    // is checked a Number and an int32 at the end of the entry predecessor, exiting at the
    // header's pc (PhiNotNumber, PhiNotInt32); in the loop no check reads a phi and nothing is
    // boxed (stage 2 step 6).
    let mut engine = vm_engine(AM3);
    let (chunk, graph) = graph_of(&mut engine, "am3");
    let text = dump(&graph, &chunk);
    let header_index = graph
        .blocks
        .iter()
        .position(|block| block.loop_header)
        .expect("the loop");
    let header = graph.block(BlockIx(header_index as u32));
    for name in ["n", "i", "j", "c"] {
        let phi = graph.node(phi_of_slot(&graph, &chunk, name, header.first_pc));
        assert_eq!(phi.rep, Rep::I32, "{name}: {text}");
        let (k, _) = header
            .preds
            .iter()
            .enumerate()
            .find(|(_, pred)| pred.index() < header_index)
            .expect("an entry edge");
        let int = graph.node(phi.inputs[k]);
        assert_eq!(int.op, NodeOp::CheckInt32, "{name}: {text}");
        assert_eq!(int.reason, Some(ExitReason::PhiNotInt32), "{name}: {text}");
        let number = graph.node(int.inputs[0]);
        assert_eq!(number.op, NodeOp::CheckNumber, "{name}: {text}");
        assert_eq!(
            number.reason,
            Some(ExitReason::PhiNotNumber),
            "{name}: {text}"
        );
        assert_eq!(
            graph.node(number.inputs[0]).op,
            NodeOp::EntrySlot(slot_named(&chunk, name)),
            "{name}: {text}"
        );
        for check in [int, number] {
            let state = graph.state(check.state.expect("a state"));
            assert_eq!(state.pc, header.first_pc, "{name}: {text}");
        }
    }
    let latch = header
        .preds
        .iter()
        .map(|pred| pred.index())
        .max()
        .expect("a back edge");
    let is_phi = |node: NodeIx| graph.node(node).op == NodeOp::Phi;
    for block in &graph.blocks[header_index..=latch] {
        for &node in &block.body {
            let node = graph.node(node);
            match node.op {
                NodeOp::CheckNumber
                | NodeOp::CheckInt32
                | NodeOp::CheckIndex
                | NodeOp::TruncateToInt32 => {
                    assert!(!is_phi(node.inputs[0]), "{node:?}: {text}");
                    let input = graph.node(node.inputs[0]);
                    assert!(
                        !matches!(input.op, NodeOp::I32ToF64) || !is_phi(input.inputs[0]),
                        "{node:?}: {text}"
                    );
                }
                NodeOp::I32ToTagged | NodeOp::F64ToTagged => panic!("a boxing in the loop: {text}"),
                _ => {}
            }
        }
    }
}

/// A loop whose phi `i` enters from the parameter `start`, its `+ 1` int32 by the VM's records.
const COUNT: &str = "function cnt(start, n) { var i = start;
       for (var k = 0; k < n; k++) i = i + 1; return i; }
     for (var r = 0; r < 20; r++) cnt(r, 5);";

#[test]
fn a_flag_at_a_loop_header_declines_its_entry_speculation() {
    let mut engine = vm_engine(COUNT);
    let (chunk, graph) = graph_of(&mut engine, "cnt");
    let text = dump(&graph, &chunk);
    let header = the_loop(&graph);
    let i = phi_of_slot(&graph, &chunk, "i", header);
    assert_eq!(graph.node(i).rep, Rep::I32, "{text}");
    assert!(graph.declines.is_empty(), "{text}");
    // PhiNotInt32 at the header: the parameter is checked a Number only, the phi is F64, and
    // the int32 add checks it at its use (an exit there, never a narrowing of the phi).
    flag(&chunk, header as usize, ExitReason::PhiNotInt32, false, 0);
    let graph = graph_for(&chunk, "cnt flagged PhiNotInt32");
    let text = dump(&graph, &chunk);
    let i = phi_of_slot(&graph, &chunk, "i", header);
    assert_eq!(graph.node(i).rep, Rep::F64, "{text}");
    let reasons = |graph: &Graph, reason| {
        graph
            .nodes
            .iter()
            .filter(|node| node.reason == Some(reason))
            .count()
    };
    assert_eq!(reasons(&graph, ExitReason::PhiNotNumber), 1, "{text}");
    assert_eq!(reasons(&graph, ExitReason::PhiNotInt32), 0, "{text}");
    assert!(
        nodes(&graph, |op| *op == NodeOp::CheckInt32)
            .iter()
            .any(|check| check.inputs == [i]),
        "the add's check of the phi: {text}"
    );
    assert_eq!(
        declines(&graph),
        [(header as usize, ExitReason::PhiNotInt32)],
        "{text}"
    );
    assert_eq!(graph.declines[0].form, "an f64 phi");
    // PhiNotNumber too: no speculation, the phi stays tagged and is checked at its uses.
    flag(&chunk, header as usize, ExitReason::PhiNotNumber, false, 0);
    let graph = graph_for(&chunk, "cnt flagged PhiNotNumber");
    let text = dump(&graph, &chunk);
    let i = phi_of_slot(&graph, &chunk, "i", header);
    assert_eq!(graph.node(i).rep, Rep::Tagged, "{text}");
    assert!(
        graph.nodes.iter().all(|node| !matches!(
            node.reason,
            Some(ExitReason::PhiNotNumber | ExitReason::PhiNotInt32)
        )),
        "{text}"
    );
    assert_eq!(
        declines(&graph),
        [(header as usize, ExitReason::PhiNotNumber)],
        "{text}"
    );
    assert_eq!(graph.declines[0].form, "a tagged phi");
}

#[test]
fn the_verifier_refuses_a_misplaced_speculation() {
    let mut engine = vm_engine(COUNT);
    let (chunk, graph) = graph_of(&mut engine, "cnt");
    let speculation = graph
        .nodes
        .iter()
        .position(|node| node.reason == Some(ExitReason::PhiNotNumber))
        .expect("the entry's check");
    let rejects = |what: &str, broken: Graph, expected: &str| {
        let error = verify(&broken, &chunk).expect_err(what);
        assert!(error.contains(expected), "{what}: {error}");
    };
    // Another pc than the header's.
    let mut broken = graph.clone();
    let state = broken.nodes[speculation].state.expect("a state").index();
    broken.states[state].pc = 0;
    rejects("another pc", broken, "not its header's");
    // The reason of the other check.
    let mut broken = graph.clone();
    broken.nodes[speculation].reason = Some(ExitReason::PhiNotInt32);
    rejects("the other reason", broken, "speculation by CheckNumber");
    // A check that is no speculation after it in its block.
    let mut broken = graph.clone();
    let block = broken.nodes[speculation].block;
    let word = broken.nodes[speculation].inputs[0];
    let check = broken.new_node(block, NodeOp::CheckNumber, &[word], Rep::F64, 0);
    broken.nodes[check.index()].state = broken.nodes[speculation].state;
    broken.nodes[check.index()].reason = Some(ExitReason::NotNumber);
    let body = &mut broken.blocks[block.index()].body;
    body.insert(body.len() - 1, check);
    rejects("a check after it", broken, "before CheckNumber");
    verify(&graph, &chunk).expect("the graph itself holds");
}

// ---- loop invariants (stage 3, step 0) ----------------------------------------------------------

/// An engine that ran `source` in the bytecode VM (its value records) and then `drive` at the
/// JIT tier (its call sites' evidence, which the VM's calls do not record).
fn warmed(source: &str, drive: &str) -> Engine {
    let mut engine = vm_engine(source);
    engine.set_tier(Tier::Jit);
    eval(&mut engine, drive);
    engine
}

/// The blocks of the natural loop whose header is block `header`: those that reach one of its back
/// edges (from a block numbered at or after it) without passing it.
fn loop_blocks(graph: &Graph, header: usize) -> Vec<bool> {
    let mut members = vec![false; graph.blocks.len()];
    members[header] = true;
    let mut work: Vec<usize> = graph.blocks[header]
        .preds
        .iter()
        .map(|pred| pred.index())
        .filter(|&pred| pred >= header)
        .collect();
    while let Some(block) = work.pop() {
        if std::mem::replace(&mut members[block], true) {
            continue;
        }
        work.extend(graph.blocks[block].preds.iter().map(|pred| pred.index()));
    }
    members
}

/// The loop whose header starts at `pc`: its header's block, its entry predecessors and its
/// blocks.
struct LoopAt {
    header: usize,
    entries: Vec<usize>,
    members: Vec<bool>,
}

fn loop_at(graph: &Graph, pc: u32) -> LoopAt {
    let header = graph
        .blocks
        .iter()
        .position(|block| block.loop_header && block.first_pc == pc)
        .unwrap_or_else(|| panic!("a loop header at pc {pc}"));
    let members = loop_blocks(graph, header);
    let entries = graph.blocks[header]
        .preds
        .iter()
        .map(|pred| pred.index())
        .filter(|&pred| !members[pred])
        .collect();
    LoopAt {
        header,
        entries,
        members,
    }
}

/// The loop headers' pcs, in block order.
fn loop_pcs(graph: &Graph) -> Vec<u32> {
    graph
        .blocks
        .iter()
        .filter(|block| block.loop_header)
        .map(|block| block.first_pc)
        .collect()
}

/// The FrameSlot loads of `chunk`'s local `name` in `graph`.
fn loads_of(graph: &Graph, chunk: &Chunk, name: &str) -> Vec<NodeIx> {
    let slot = slot_named(chunk, name);
    (0..graph.nodes.len() as u32)
        .map(NodeIx)
        .filter(|&node| graph.node(node).op == NodeOp::FrameSlot(slot))
        .collect()
}

/// The entry block's read of `chunk`'s local `name` (its EntrySlot): the value a loop reads where
/// its entry leaves the slot unwritten and no canonical-frame effect ran (P3 stage 3 step 3a: a
/// generic node leaves it in place, and no reload replaces it).
fn entry_value(graph: &Graph, chunk: &Chunk, name: &str) -> NodeIx {
    let slot = slot_named(chunk, name);
    (0..graph.nodes.len() as u32)
        .map(NodeIx)
        .find(|&node| graph.node(node).op == NodeOp::EntrySlot(slot))
        .unwrap_or_else(|| panic!("the entry value of {name}"))
}

/// The nodes of `graph` whose op `which` accepts and whose first input is `input`.
fn checks_of(graph: &Graph, input: NodeIx, which: impl Fn(&NodeOp) -> bool) -> Vec<NodeIx> {
    (0..graph.nodes.len() as u32)
        .map(NodeIx)
        .filter(|&node| {
            let node = graph.node(node);
            which(&node.op) && node.inputs.first() == Some(&input)
        })
        .collect()
}

/// The entry check `check` moved to its loop's entry: at the end of `entry`, HoistingFailed,
/// exiting at the header's pc with the slots `kept` in the frame's keeping.
fn assert_hoisted(
    graph: &Graph,
    check: NodeIx,
    entry: usize,
    header_pc: u32,
    kept: &[usize],
    text: &str,
) {
    let node = graph.node(check);
    assert_eq!(node.block.index(), entry, "{node:?}: {text}");
    assert_eq!(
        node.reason,
        Some(ExitReason::HoistingFailed),
        "{node:?}: {text}"
    );
    let state = graph.state(node.state.expect("a hoisted check's state"));
    assert_eq!(state.pc, header_pc, "{node:?}: {text}");
    for &slot in kept {
        assert_eq!(state.slots[slot], SlotSource::Frame, "{node:?}: {text}");
    }
}

/// An effect, then a loop without one reading the invariant locals `a` (in f64 arithmetic) and
/// `n` (in an int32 comparison) and writing `s` and `i`.
const INVARIANT: &str = "function inv(g, a, n) { var s = 0; g();
       for (var i = 0; i < n; i++) s = s + a * i; return s + a; }
     function nothing() {}
     for (var r = 0; r < 20; r++) inv(nothing, 1.5, 10);";
const INVARIANT_DRIVE: &str = "for (var r = 0; r < 20; r++) inv(nothing, 1.5, 10);";
/// INVARIANT with a canonical-frame effect in place of the call (an array spread, whose helper
/// runs the VM on a copy of the operand stack): every live slot is in the frame's keeping after
/// it (P3 stage 3 step 3a).
const INVARIANT_SPREAD: &str = "function inv(g, a, n) { var s = 0; var t = [...g];
       for (var i = 0; i < n; i++) s = s + a * i; return s + a + t.length; }
     for (var r = 0; r < 20; r++) inv([], 1.5, 10);";
const INVARIANT_SPREAD_DRIVE: &str = "for (var r = 0; r < 20; r++) inv([], 1.5, 10);";

#[test]
fn an_effect_free_loops_invariant_slots_are_loaded_once_on_its_entry() {
    // After the spread every slot is in the frame's keeping; the loop has no effect and does not
    // write `a` and `n`: each is loaded once, at the end of the loop's entry predecessor, and
    // held through the loop, where nothing reloads it and every state keeps it in the frame's
    // keeping. `s` and `i` take phis.
    let mut engine = warmed(INVARIANT_SPREAD, INVARIANT_SPREAD_DRIVE);
    let (chunk, graph) = graph_of(&mut engine, "inv");
    let text = dump(&graph, &chunk);
    let header_pc = the_loop(&graph);
    let lp = loop_at(&graph, header_pc);
    let [entry] = lp.entries[..] else {
        panic!("one entry edge: {text}")
    };
    let (a, n) = (
        slot_named(&chunk, "a") as usize,
        slot_named(&chunk, "n") as usize,
    );
    for node in &graph.nodes {
        if lp.members[node.block.index()] {
            assert!(!is_effect(&node.op), "an effect in the loop: {text}");
            assert!(
                !matches!(node.op, NodeOp::FrameSlot(_)),
                "a reload in the loop: {text}"
            );
            if let Some(state) = node.state {
                let state = graph.state(state);
                assert_eq!(state.slots[a], SlotSource::Frame, "{node:?}: {text}");
                assert_eq!(state.slots[n], SlotSource::Frame, "{node:?}: {text}");
            }
        }
    }
    for name in ["a", "n"] {
        let [load] = loads_of(&graph, &chunk, name)[..] else {
            panic!("one load of {name}: {text}")
        };
        assert_eq!(graph.node(load).block.index(), entry, "{text}");
    }
    for name in ["s", "i"] {
        phi_of_slot(&graph, &chunk, name, header_pc);
    }
}

#[test]
fn a_generic_node_before_a_loop_leaves_its_invariants_held_without_loads() {
    // P3 stage 3 step 3a: the call before the loop is a generic node, which leaves `a` and `n`
    // as they were, unwritten since the entry: the loop reads their entry values, nothing loads
    // them from the frame anywhere, and every state in the loop keeps them in the frame's
    // keeping. `s` and `i` take phis.
    let mut engine = warmed(INVARIANT, INVARIANT_DRIVE);
    let (chunk, graph) = graph_of(&mut engine, "inv");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 1, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::Generic { .. })),
        1,
        "{text}"
    );
    let header_pc = the_loop(&graph);
    let lp = loop_at(&graph, header_pc);
    let (a, n) = (
        slot_named(&chunk, "a") as usize,
        slot_named(&chunk, "n") as usize,
    );
    assert_eq!(
        count(&graph, |op| matches!(
            op,
            NodeOp::FrameSlot(_) | NodeOp::StoreSlot(_)
        )),
        0,
        "no load or flush: {text}"
    );
    for node in &graph.nodes {
        if lp.members[node.block.index()] {
            if let Some(state) = node.state {
                let state = graph.state(state);
                assert_eq!(state.slots[a], SlotSource::Frame, "{node:?}: {text}");
                assert_eq!(state.slots[n], SlotSource::Frame, "{node:?}: {text}");
            }
        }
    }
    for name in ["a", "n"] {
        entry_value(&graph, &chunk, name);
    }
    for name in ["s", "i"] {
        phi_of_slot(&graph, &chunk, name, header_pc);
    }
}

#[test]
fn a_slot_read_only_after_the_loop_stays_in_the_frame_through_it() {
    // `b` is live at the header but only the return reads it: it stays in the frame through the
    // loop (no load at the entry, none in the loop) and is read back after it; `a`, which the
    // loop reads, is loaded at the entry. A canonical-frame effect (the spread) leaves both in
    // the frame's keeping.
    let mut engine = warmed(
        "function after(g, a, b, n) { var s = 0; var t = [...g];
           for (var i = 0; i < n; i++) s = s + a; return s + b + t.length; }
         for (var r = 0; r < 20; r++) after([], 1.5, 2.5, 10);",
        "for (var r = 0; r < 20; r++) after([], 1.5, 2.5, 10);",
    );
    let (chunk, graph) = graph_of(&mut engine, "after");
    let text = dump(&graph, &chunk);
    let lp = loop_at(&graph, the_loop(&graph));
    let [entry] = lp.entries[..] else {
        panic!("one entry edge: {text}")
    };
    let [load_a] = loads_of(&graph, &chunk, "a")[..] else {
        panic!("one load of a: {text}")
    };
    assert_eq!(graph.node(load_a).block.index(), entry, "{text}");
    let [load_b] = loads_of(&graph, &chunk, "b")[..] else {
        panic!("one load of b: {text}")
    };
    let block = graph.node(load_b).block.index();
    assert!(block != entry && !lp.members[block], "{text}");
    let b = slot_named(&chunk, "b") as usize;
    let poll = graph.node(graph.blocks[lp.header].body[0]);
    assert_eq!(
        graph.state(poll.state.expect("a poll's state")).slots[b],
        SlotSource::Frame,
        "{text}"
    );
}

#[test]
fn an_effect_free_loops_invariant_checks_move_to_its_entry() {
    // The checks in the loop of the invariants `a` and `n`, held across the call (P3 stage 3
    // step 3a: their entry values), move to the end of the entry predecessor (CheckNumber of `a`,
    // CheckNumber and CheckInt32 of `n`; HoistingFailed, exiting at the header's pc with both in
    // the frame's keeping): nothing in the loop reads their words. The check of `a` after the
    // loop (the return's addition) stays there.
    let mut engine = warmed(INVARIANT, INVARIANT_DRIVE);
    let (chunk, graph) = graph_of(&mut engine, "inv");
    let text = dump(&graph, &chunk);
    let header_pc = the_loop(&graph);
    let lp = loop_at(&graph, header_pc);
    let [entry] = lp.entries[..] else {
        panic!("one entry edge: {text}")
    };
    let in_loop = |node: NodeIx| lp.members[graph.node(node).block.index()];
    let kept = [
        slot_named(&chunk, "a") as usize,
        slot_named(&chunk, "n") as usize,
    ];
    let value_a = entry_value(&graph, &chunk, "a");
    let value_n = entry_value(&graph, &chunk, "n");
    for value in [value_a, value_n] {
        assert!(
            (0..graph.nodes.len() as u32)
                .map(NodeIx)
                .filter(|&node| in_loop(node))
                .all(|node| !graph.node(node).inputs.contains(&value)),
            "a use of the word in the loop: {text}"
        );
    }
    let [number, after] = checks_of(&graph, value_a, |op| *op == NodeOp::CheckNumber)[..] else {
        panic!("two checks of a: {text}")
    };
    let (number, after) = if graph.node(number).block.index() == entry {
        (number, after)
    } else {
        (after, number)
    };
    assert_hoisted(&graph, number, entry, header_pc, &kept, &text);
    let after = graph.node(after);
    assert!(!lp.members[after.block.index()], "{text}");
    assert_ne!(after.block.index(), entry, "{text}");
    assert_eq!(after.reason, Some(ExitReason::NotNumber), "{text}");
    let [number] = checks_of(&graph, value_n, |op| *op == NodeOp::CheckNumber)[..] else {
        panic!("one check of n: {text}")
    };
    assert_hoisted(&graph, number, entry, header_pc, &kept, &text);
    let [int] = checks_of(&graph, number, |op| *op == NodeOp::CheckInt32)[..] else {
        panic!("one int32 check of n: {text}")
    };
    assert_hoisted(&graph, int, entry, header_pc, &kept, &text);
    assert!(graph.declines.is_empty(), "{text}");
}

#[test]
fn an_invariants_checks_move_to_its_loops_entry_in_the_renumbered_blocks() {
    // P3 stage 3 step 3a: the notes name an invariant's entry predecessor by its block, which the
    // builder numbers as it makes it and the reverse postorder renumbers. EarleyBoyer's conf-set
    // getter: its second stage splices `make` and, in it, `fill`'s loop, whose entry block the
    // builder made after the guards' fallback blocks; under the builder's number the array's
    // checks moved to the end of the loop's back edge, which dominates none of their uses (the
    // verifier refused the graph). They move to the entry block.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function fill(a, v) { for (var i = 0; i < a.length; i++) a[i] = v; }
         function make(size, v) { var a = new Array(size); if (v !== undefined) fill(a, v);
           return a; }
         function get(state, num, conf) { var set, found;
           return (found = state[conf + 1]) !== false ? found
             : (set = make(num + 6, false), set[1] = -3, state[conf + 1] = set, set); }
         function drive() { var s = 0;
           for (var k = 0; k < 40; k++) { var state = make(8, false);
             for (var j = 0; j < 10; j++) s += get(state, 4, j % 5).length; }
           return s; }
         for (var r = 0; r < 20; r++) drive();",
    );
    let Some(transformed) = second_stage(&mut engine, "get", "drive()") else {
        return;
    };
    let graph = graph_for(&transformed, "get's second stage");
    let text = dump(&graph, &transformed);
    let lp = loop_at(&graph, the_loop(&graph));
    let [entry] = lp.entries[..] else {
        panic!("one entry edge: {text}")
    };
    let hoisted = nodes(&graph, |_| true)
        .into_iter()
        .filter(|node| node.reason == Some(ExitReason::HoistingFailed))
        .collect::<Vec<_>>();
    assert!(
        hoisted.iter().any(|node| node.op == NodeOp::CheckObject),
        "the array's check: {text}"
    );
    for node in hoisted {
        assert_eq!(node.block.index(), entry, "{node:?}: {text}");
    }
}

#[test]
fn sibling_loops_holding_one_entry_value_each_check_it_at_their_entries() {
    // P3 stage 3 step 3a: an entry value two arms' loops read (NavierStokes' lin_solve: its
    // arrays, held across the branch's generic equality) is an invariant of each: the notes take
    // it at both headers and each loop's checks of it move to its own entry. Noted once, the
    // second arm's loop kept its checks (NavierStokes +2.3% instructions); each arm's nested loops
    // check it once, at the outer loop's entry.
    let mut engine = warmed(
        "function ls(b, x, x0, a, c, n, g) {
           if (a === 0 && c === 1) {
             for (var j = 1; j <= n; j++) { var r = j * 2;
               for (var i = 0; i < n; i++) { x[r] = x0[r]; ++r; } }
             g(b, x);
           } else {
             var inv = 1 / c;
             for (var k = 0; k < 2; k++) {
               for (var j = 1; j <= n; j++) { var r = j * 2; var last = x[r];
                 for (var i = 1; i <= n; i++) last = x[r] = (x0[r] + a * (last + x[++r])) * inv; }
             }
             g(b, x);
           }
         }
         function bnd(b, x) { if (b === 1) x[0] = 1; }
         var X = [], X0 = []; for (var q = 0; q < 40; q++) { X.push(0.5); X0.push(1.5); }
         for (var w = 0; w < 20; w++) { ls(0, X, X0, 0, 1, 8, bnd); ls(1, X, X0, 0.1, 1.4, 8, bnd); }",
        "for (var w = 0; w < 20; w++) { ls(0, X, X0, 0, 1, 8, bnd); ls(1, X, X0, 0.1, 1.4, 8, bnd); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "ls");
    let text = dump(&graph, &chunk);
    let headers: Vec<usize> = (0..graph.blocks.len())
        .filter(|&b| graph.blocks[b].loop_header)
        .collect();
    let in_any_loop = |block: usize| {
        headers
            .iter()
            .any(|&header| loop_blocks(&graph, header)[block])
    };
    for name in ["x", "x0"] {
        let value = entry_value(&graph, &chunk, name);
        let checks = checks_of(&graph, value, |op| *op == NodeOp::CheckObject);
        assert_eq!(
            checks.len(),
            2,
            "{name}: one at each arm's outer loop: {text}"
        );
        for check in checks {
            let node = graph.node(check);
            assert_eq!(
                node.reason,
                Some(ExitReason::HoistingFailed),
                "{name}: {text}"
            );
            assert!(!in_any_loop(node.block.index()), "{name}: {text}");
        }
    }
}

#[test]
fn a_loop_entered_from_a_branch_loads_there_and_checks_in_the_loop() {
    // The do-while's body is the branch's target: the invariant (`a`'s entry value, held across
    // the call) enters from a branch whose two successors leave no edge of the loop's alone to
    // move a check to (stage 2 step 6's speculation rule), so the check stays in the loop.
    let mut engine = warmed(
        "function br(g, a, n, c) { var s = 0; g();
           if (c) do { s = s + a; } while (--n > 0);
           return s; }
         function nothing() {}
         for (var r = 0; r < 20; r++) br(nothing, 1.5, 10, true);",
        "for (var r = 0; r < 20; r++) br(nothing, 1.5, 10, true);",
    );
    let (chunk, graph) = graph_of(&mut engine, "br");
    let text = dump(&graph, &chunk);
    let lp = loop_at(&graph, the_loop(&graph));
    let [entry] = lp.entries[..] else {
        panic!("one entry edge: {text}")
    };
    assert_eq!(graph.blocks[entry].succs.len(), 2, "{text}");
    assert!(loads_of(&graph, &chunk, "a").is_empty(), "{text}");
    let value = entry_value(&graph, &chunk, "a");
    let checks = checks_of(&graph, value, |op| *op == NodeOp::CheckNumber);
    assert_eq!(checks.len(), 1, "{text}");
    assert!(lp.members[graph.node(checks[0]).block.index()], "{text}");
    assert!(
        graph
            .nodes
            .iter()
            .all(|node| node.reason != Some(ExitReason::HoistingFailed)),
        "{text}"
    );
}

#[test]
fn a_loop_with_an_effect_reloads_and_checks_its_invariant_slots_as_before() {
    // A canonical-frame effect in the loop (a spread) leaves every slot in the frame's keeping:
    // `a` is reloaded after it and checked in the loop, as before P3 stage 3 step 3a.
    let mut engine = warmed(
        "function eff(g, a, n) { var s = 0; var t;
           for (var i = 0; i < n; i++) { s = s + a * i; t = [...g]; } return s + t.length; }
         for (var r = 0; r < 20; r++) eff([], 1.5, 10);",
        "for (var r = 0; r < 20; r++) eff([], 1.5, 10);",
    );
    let (chunk, graph) = graph_of(&mut engine, "eff");
    let text = dump(&graph, &chunk);
    let lp = loop_at(&graph, the_loop(&graph));
    let loads = loads_of(&graph, &chunk, "a");
    assert!(!loads.is_empty(), "{text}");
    for load in loads {
        assert!(lp.members[graph.node(load).block.index()], "{text}");
        let checks = checks_of(&graph, load, |op| *op == NodeOp::CheckNumber);
        assert_eq!(checks.len(), 1, "{text}");
        assert!(lp.members[graph.node(checks[0]).block.index()], "{text}");
    }
    assert!(
        graph
            .nodes
            .iter()
            .all(|node| node.reason != Some(ExitReason::HoistingFailed)),
        "{text}"
    );
}

#[test]
fn a_loop_with_a_generic_node_holds_its_invariant_and_checks_it_in_the_loop() {
    // P3 stage 3 step 3a: the call in the loop is a generic node, which leaves `a` alone: no
    // reload anywhere, the loop reads `a`'s entry value and checks it in the loop (step 0's rule
    // moves checks out of loops without effects or generic nodes only).
    let mut engine = warmed(
        "function eff(g, a, n) { var s = 0; g();
           for (var i = 0; i < n; i++) { s = s + a * i; g(); } return s; }
         function nothing() {}
         for (var r = 0; r < 20; r++) eff(nothing, 1.5, 10);",
        "for (var r = 0; r < 20; r++) eff(nothing, 1.5, 10);",
    );
    let (chunk, graph) = graph_of(&mut engine, "eff");
    let text = dump(&graph, &chunk);
    let lp = loop_at(&graph, the_loop(&graph));
    assert!(loads_of(&graph, &chunk, "a").is_empty(), "{text}");
    let value = entry_value(&graph, &chunk, "a");
    let checks = checks_of(&graph, value, |op| *op == NodeOp::CheckNumber);
    assert!(!checks.is_empty(), "{text}");
    assert!(
        checks
            .iter()
            .all(|&check| lp.members[graph.node(check).block.index()]),
        "{text}"
    );
    assert!(
        graph
            .nodes
            .iter()
            .all(|node| node.reason != Some(ExitReason::HoistingFailed)),
        "{text}"
    );
}

#[test]
fn a_loop_whose_equalities_compare_numbers_its_blocks_made_is_free_of_effects() {
    // `((i | 0) & 1) == 0`, `(s | 0) !== -1` and `i !== undefined` lower whatever the facts at
    // the loop's entry (Rules::equality_lowers: bitwise results and Number constants, a constant
    // no other value equals): the loop has no effect, and `a`, which it reads and does not write,
    // is never reloaded in it (P3 stage 3 step 0b); its entry value is held across the call
    // (step 3a).
    let mut engine = warmed(
        "function eqs(g, a, n) { var s = 0; g();
           for (var i = 0; i < n; i++) {
             if (((i | 0) & 1) == 0) s = s + a;
             if ((s | 0) !== -1 && i !== undefined) s = s + 1;
           }
           return s; }
         function nothing() {}
         for (var r = 0; r < 20; r++) eqs(nothing, 1.5, 10);",
        "for (var r = 0; r < 20; r++) eqs(nothing, 1.5, 10);",
    );
    let (chunk, graph) = graph_of(&mut engine, "eqs");
    let text = dump(&graph, &chunk);
    let lp = loop_at(&graph, the_loop(&graph));
    for node in &graph.nodes {
        if lp.members[node.block.index()] {
            assert!(!is_effect(&node.op), "an effect in the loop: {text}");
            assert!(
                !matches!(node.op, NodeOp::FrameSlot(_)),
                "a reload in the loop: {text}"
            );
        }
    }
    assert!(loads_of(&graph, &chunk, "a").is_empty(), "{text}");
    entry_value(&graph, &chunk, "a");
}

#[test]
fn a_loop_comparing_words_from_the_frame_counts_its_equality_an_effect() {
    // `a == b` on two locals the loop's block did not make, and `a == undefined`, whose constant
    // decides nothing under `==`: the rules cannot say they lower (the builder makes the generic
    // operation here), so the loop has a generic node; the function compiles, its prepass and its
    // builder agreeing, and the node leaves `a` alone (P3 stage 3 step 3a): no reload.
    for test in ["a == b", "a == undefined"] {
        let mut engine = warmed(
            &format!(
                "function eqf(g, a, b, n) {{ var s = 0; g();
                   for (var i = 0; i < n; i++) {{ if ({test}) s = s + a; }} return s; }}
                 function nothing() {{}}
                 for (var r = 0; r < 20; r++) eqf(nothing, 1.5, '1.5', 10);"
            ),
            "for (var r = 0; r < 20; r++) eqf(nothing, 1.5, '1.5', 10);",
        );
        let (chunk, graph) = graph_of(&mut engine, "eqf");
        let text = dump(&graph, &chunk);
        let lp = loop_at(&graph, the_loop(&graph));
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| lp.members[node.block.index()]
                    && matches!(node.op, NodeOp::Generic { .. })),
            "{test}: the equality's generic node in the loop: {text}"
        );
        assert!(
            loads_of(&graph, &chunk, "a").is_empty(),
            "{test}: no reload of a: {text}"
        );
    }
}

#[test]
fn a_slot_the_loop_writes_keeps_its_phi_and_its_entry_speculation() {
    // `s`, in the frame's keeping at the entry (after the spread, a canonical-frame effect) and
    // written in the loop, takes a phi whose entry input is its load (stage 2 step 6's
    // PhiNotNumber speculation), never the invariant rule.
    let mut engine = warmed(INVARIANT_SPREAD, INVARIANT_SPREAD_DRIVE);
    let (chunk, graph) = graph_of(&mut engine, "inv");
    let text = dump(&graph, &chunk);
    let header_pc = the_loop(&graph);
    let phi = phi_of_slot(&graph, &chunk, "s", header_pc);
    let [load] = loads_of(&graph, &chunk, "s")[..] else {
        panic!("one load of s: {text}")
    };
    let checks = checks_of(&graph, load, |_| true);
    assert_eq!(checks.len(), 1, "{text}");
    assert_eq!(
        graph.node(checks[0]).reason,
        Some(ExitReason::PhiNotNumber),
        "{text}"
    );
    assert_eq!(graph.node(phi).rep, Rep::F64, "{text}");
}

#[test]
fn an_inner_loop_without_effects_in_one_with_an_effect_loads_on_its_own_entry() {
    // The outer loop's call is a generic node; the inner loop, free of them, notes `a`'s entry
    // value as its invariant and checks it on its own entry edge (P3 stage 3 step 3a: held, not
    // loaded).
    let mut engine = warmed(
        "function nest(g, a, n) { var s = 0;
           for (var k = 0; k < 3; k++) { g(); for (var i = 0; i < n; i++) s = s + a * i; }
           return s; }
         function nothing() {}
         for (var r = 0; r < 20; r++) nest(nothing, 1.5, 10);",
        "for (var r = 0; r < 20; r++) nest(nothing, 1.5, 10);",
    );
    let (chunk, graph) = graph_of(&mut engine, "nest");
    let text = dump(&graph, &chunk);
    let [outer_pc, inner_pc] = loop_pcs(&graph)[..] else {
        panic!("two loops: {text}")
    };
    let (outer, inner) = (loop_at(&graph, outer_pc), loop_at(&graph, inner_pc));
    let [entry] = inner.entries[..] else {
        panic!("one entry edge: {text}")
    };
    assert!(outer.members[entry], "{text}");
    assert!(loads_of(&graph, &chunk, "a").is_empty(), "{text}");
    let value = entry_value(&graph, &chunk, "a");
    let [number] = checks_of(&graph, value, |op| *op == NodeOp::CheckNumber)[..] else {
        panic!("one check of a: {text}")
    };
    let a = slot_named(&chunk, "a") as usize;
    assert_hoisted(&graph, number, entry, inner_pc, &[a], &text);
    assert_ne!(outer.header, inner.header);
}

#[test]
fn invariant_checks_in_an_inner_loop_move_to_the_outermost_entry_without_effects() {
    let mut engine = warmed(
        "function nest2(g, a, n) { var s = 0; g();
           for (var k = 0; k < 3; k++) for (var i = 0; i < n; i++) s = s + a * i;
           return s; }
         function nothing() {}
         for (var r = 0; r < 20; r++) nest2(nothing, 1.5, 10);",
        "for (var r = 0; r < 20; r++) nest2(nothing, 1.5, 10);",
    );
    let (chunk, graph) = graph_of(&mut engine, "nest2");
    let text = dump(&graph, &chunk);
    let [outer_pc, inner_pc] = loop_pcs(&graph)[..] else {
        panic!("two loops: {text}")
    };
    let outer = loop_at(&graph, outer_pc);
    let [entry] = outer.entries[..] else {
        panic!("one entry edge: {text}")
    };
    assert!(loads_of(&graph, &chunk, "a").is_empty(), "{text}");
    let value = entry_value(&graph, &chunk, "a");
    let [number] = checks_of(&graph, value, |op| *op == NodeOp::CheckNumber)[..] else {
        panic!("one check of a: {text}")
    };
    let a = slot_named(&chunk, "a") as usize;
    assert_hoisted(&graph, number, entry, outer_pc, &[a], &text);
    assert!(inner_pc > outer_pc, "{text}");
}

#[test]
fn a_loop_with_two_entry_edges_merges_their_loads_and_checks_in_the_loop() {
    // `a` is in the frame's keeping on both entry edges (each after a spread, a canonical-frame
    // effect): each loads it, a phi of the loads is held through the loop (its back edges' input
    // itself), and with no one entry edge to move them to its checks stay in the loop.
    let mut engine = warmed(
        "function two(g, a, n, c) { var s = 0; var t = [...g]; var i = 0;
           if (c) { i = 1; } else { t = [...g]; }
           while (i < n) { s = s + a; i++; }
           return s + t.length; }
         for (var r = 0; r < 20; r++) two([], 1.5, 10, r % 2);",
        "for (var r = 0; r < 20; r++) two([], 1.5, 10, r % 2);",
    );
    let (chunk, graph) = graph_of(&mut engine, "two");
    let text = dump(&graph, &chunk);
    let header_pc = the_loop(&graph);
    let lp = loop_at(&graph, header_pc);
    assert_eq!(lp.entries.len(), 2, "{text}");
    let loads = loads_of(&graph, &chunk, "a");
    assert_eq!(loads.len(), 2, "{text}");
    let header = &graph.blocks[lp.header];
    let phi = header
        .phis
        .iter()
        .copied()
        .find(|&phi| {
            loads
                .iter()
                .all(|load| graph.node(phi).inputs.contains(load))
        })
        .unwrap_or_else(|| panic!("a phi of the loads: {text}"));
    for (k, pred) in header.preds.iter().enumerate() {
        let input = graph.node(phi).inputs[k];
        if lp.members[pred.index()] {
            assert_eq!(input, phi, "{text}");
        } else {
            assert_eq!(graph.node(input).block, *pred, "{text}");
        }
    }
    let checks = checks_of(&graph, phi, |op| *op == NodeOp::CheckNumber);
    assert_eq!(checks.len(), 1, "{text}");
    assert!(lp.members[graph.node(checks[0]).block.index()], "{text}");
    let a = slot_named(&chunk, "a") as usize;
    let poll = graph.node(header.body[0]);
    assert_eq!(
        graph.state(poll.state.expect("a poll's state")).slots[a],
        SlotSource::Frame,
        "{text}"
    );
    assert!(
        graph
            .nodes
            .iter()
            .all(|node| node.reason != Some(ExitReason::HoistingFailed)),
        "{text}"
    );
}

#[test]
fn a_loop_with_two_entry_edges_after_generic_nodes_reads_its_entry_value() {
    // P3 stage 3 step 3a: after generic nodes on both entry edges `a` is still its entry value,
    // unwritten: no load, no phi, and with two entry edges its check stays in the loop.
    let mut engine = warmed(
        "function two(g, a, n, c) { var s = 0; g(); var i = 0;
           if (c) { i = 1; } else { g(); }
           while (i < n) { s = s + a; i++; }
           return s; }
         function nothing() {}
         for (var r = 0; r < 20; r++) two(nothing, 1.5, 10, r % 2);",
        "for (var r = 0; r < 20; r++) two(nothing, 1.5, 10, r % 2);",
    );
    let (chunk, graph) = graph_of(&mut engine, "two");
    let text = dump(&graph, &chunk);
    let lp = loop_at(&graph, the_loop(&graph));
    assert_eq!(lp.entries.len(), 2, "{text}");
    assert!(loads_of(&graph, &chunk, "a").is_empty(), "{text}");
    let value = entry_value(&graph, &chunk, "a");
    let checks = checks_of(&graph, value, |op| *op == NodeOp::CheckNumber);
    assert_eq!(checks.len(), 1, "{text}");
    assert!(lp.members[graph.node(checks[0]).block.index()], "{text}");
    assert!(
        graph
            .nodes
            .iter()
            .all(|node| node.reason != Some(ExitReason::HoistingFailed)),
        "{text}"
    );
}

#[test]
fn a_flag_at_the_header_leaves_the_invariants_checks_in_the_loop() {
    let mut engine = warmed(INVARIANT, INVARIANT_DRIVE);
    let chunk = chunk_of(&mut engine, "inv");
    let graph = graph_for(&chunk, "inv");
    let header_pc = the_loop(&graph);
    flag(
        &chunk,
        header_pc as usize,
        ExitReason::HoistingFailed,
        false,
        0,
    );
    let graph = graph_for(&chunk, "inv flagged");
    let text = dump(&graph, &chunk);
    let lp = loop_at(&graph, header_pc);
    assert!(
        graph
            .nodes
            .iter()
            .all(|node| node.reason != Some(ExitReason::HoistingFailed)),
        "{text}"
    );
    // The invariants (their entry values, held across the call: P3 stage 3 step 3a) are checked
    // in the loop, not at the entry.
    let [entry] = lp.entries[..] else {
        panic!("one entry edge: {text}")
    };
    for name in ["a", "n"] {
        let value = entry_value(&graph, &chunk, name);
        let checks = checks_of(&graph, value, |op| *op == NodeOp::CheckNumber);
        assert!(
            checks
                .iter()
                .any(|&check| lp.members[graph.node(check).block.index()]),
            "{name}: {text}"
        );
        assert!(
            checks
                .iter()
                .all(|&check| graph.node(check).block.index() != entry),
            "{name}: {text}"
        );
    }
    assert_eq!(
        declines(&graph),
        [(header_pc as usize, ExitReason::HoistingFailed)],
        "{text}"
    );
    assert_eq!(graph.declines[0].form, "the checks in the loop");
}

#[test]
fn an_invariant_receivers_object_and_kind_checks_move_to_the_entry() {
    let mut engine = warmed(
        "function sumel(g, x, n) { var s = 0; g();
           for (var i = 0; i < n; i++) s = s + x[i]; return s; }
         function nothing() {}
         var X = [1.5, 2.5, 3.5, 4.5];
         for (var r = 0; r < 20; r++) sumel(nothing, X, 4);",
        "for (var r = 0; r < 20; r++) sumel(nothing, X, 4);",
    );
    let (chunk, graph) = graph_of(&mut engine, "sumel");
    let text = dump(&graph, &chunk);
    let header_pc = the_loop(&graph);
    let lp = loop_at(&graph, header_pc);
    let [entry] = lp.entries[..] else {
        panic!("one entry edge: {text}")
    };
    let value = entry_value(&graph, &chunk, "x");
    let x = slot_named(&chunk, "x") as usize;
    let [cell] = checks_of(&graph, value, |op| *op == NodeOp::CheckObject)[..] else {
        panic!("one object check of x: {text}")
    };
    assert_hoisted(&graph, cell, entry, header_pc, &[x], &text);
    let kinds = checks_of(&graph, cell, |op| matches!(op, NodeOp::CheckElements(_)));
    assert_eq!(kinds.len(), 1, "{text}");
    assert_hoisted(&graph, kinds[0], entry, header_pc, &[x], &text);
    assert!(
        graph
            .nodes
            .iter()
            .all(|node| !lp.members[node.block.index()]
                || !matches!(node.op, NodeOp::CheckObject | NodeOp::CheckElements(_))),
        "{text}"
    );
}

#[test]
fn a_kind_check_behind_a_typed_array_test_stays_in_the_loop() {
    // A site that met typed arrays and Arrays tests the cell's kind tag in the loop; the words
    // arm's kind check sits behind it and would fail at the entry for every typed array.
    let mut engine = warmed(
        "function mixed(g, x, n) { var s = 0; g();
           for (var i = 0; i < n; i++) s = s + x[i]; return s; }
         function nothing() {}
         var T = new Float64Array([1, 2, 3, 4]), X = [1.5, 2.5, 3.5, 4.5];
         for (var r = 0; r < 20; r++) mixed(nothing, r % 2 ? T : X, 4);",
        "for (var r = 0; r < 20; r++) mixed(nothing, r % 2 ? T : X, 4);",
    );
    let (chunk, graph) = graph_of(&mut engine, "mixed");
    let text = dump(&graph, &chunk);
    let header_pc = the_loop(&graph);
    let lp = loop_at(&graph, header_pc);
    let [entry] = lp.entries[..] else {
        panic!("one entry edge: {text}")
    };
    let value = entry_value(&graph, &chunk, "x");
    let [cell] = checks_of(&graph, value, |op| *op == NodeOp::CheckObject)[..] else {
        panic!("one object check of x: {text}")
    };
    let x = slot_named(&chunk, "x") as usize;
    assert_hoisted(&graph, cell, entry, header_pc, &[x], &text);
    let typed = checks_of(&graph, cell, |op| *op == NodeOp::IsTypedArray);
    assert_eq!(typed.len(), 1, "{text}");
    assert!(lp.members[graph.node(typed[0]).block.index()], "{text}");
    let kinds = nodes(&graph, |op| matches!(op, NodeOp::CheckElements(_)));
    assert!(!kinds.is_empty(), "{text}");
    assert!(
        kinds.iter().all(|kind| lp.members[kind.block.index()]
            && kind.reason != Some(ExitReason::HoistingFailed)),
        "{text}"
    );
}

#[test]
fn the_verifier_refuses_a_misplaced_invariant_check() {
    let mut engine = warmed(INVARIANT, INVARIANT_DRIVE);
    let (chunk, graph) = graph_of(&mut engine, "inv");
    let hoisted = graph
        .nodes
        .iter()
        .position(|node| {
            node.reason == Some(ExitReason::HoistingFailed) && node.op == NodeOp::CheckNumber
        })
        .expect("a hoisted check");
    let rejects = |what: &str, broken: Graph, expected: &str| {
        let error = verify(&broken, &chunk).expect_err(what);
        assert!(error.contains(expected), "{what}: {error}");
    };
    // Another pc than the header's.
    let mut broken = graph.clone();
    let state = broken.nodes[hoisted].state.expect("a state").index();
    broken.states[state].pc = 0;
    rejects("another pc", broken, "not its header's");
    // A check of another kind under the reason.
    let mut broken = graph.clone();
    let block = broken.nodes[hoisted].block;
    let word = broken.nodes[hoisted].inputs[0];
    let check = broken.new_node(block, NodeOp::CheckBoolean, &[word], Rep::Bool, 0);
    broken.nodes[check.index()].state = broken.nodes[hoisted].state;
    broken.nodes[check.index()].reason = Some(ExitReason::HoistingFailed);
    let body = &mut broken.blocks[block.index()].body;
    body.insert(body.len() - 1, check);
    rejects("another check", broken, "speculation by CheckBoolean");
    // A check that is no speculation after it in its block.
    let mut broken = graph.clone();
    let check = broken.new_node(block, NodeOp::CheckNumber, &[word], Rep::F64, 0);
    broken.nodes[check.index()].state = broken.nodes[hoisted].state;
    broken.nodes[check.index()].reason = Some(ExitReason::NotNumber);
    let body = &mut broken.blocks[block.index()].body;
    body.insert(body.len() - 1, check);
    rejects("a check after it", broken, "before CheckNumber");
    verify(&graph, &chunk).expect("the graph itself holds");
}

/// P3 stage 3 step 0c: a typed array's facts through block edges and effect-free loops.
#[path = "typed_loop_tests.rs"]
mod typed_loop;

/// P3 stage 3 step L2: named loads through the prototype chain.
#[path = "chain_tests.rs"]
mod chain;

/// An operation the tier does not lower is a generic node (P3 stage 3 step 3a); one that may read
/// any slot or whose helper runs the VM on a copy of the operand stack stays a canonical-frame
/// effect (`liveness::SlotUse::Any`, `bytecode::jit_runs_on_vm_copy`).
#[test]
fn operations_needing_the_whole_frame_stay_canonical_effects_and_others_are_generic() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function g(x) { return x; }
         function f(a) { var o = g(a); var s = [...a]; return o.length + s.length; }
         for (var i = 0; i < 20; i++) f([1, 2]);",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    let text = dump(&graph, &chunk);
    let call = pc_of(&chunk, |op| {
        matches!(op, Op::Call(..) | Op::CallWithThis(..))
    });
    let spread = pc_of(&chunk, |op| matches!(op, Op::ArraySpread));
    let at = |pc: usize| {
        nodes(&graph, is_effect)
            .into_iter()
            .find(|node| node.pc as usize == pc)
            .map(|node| node.op.clone())
    };
    assert!(
        matches!(at(call), Some(NodeOp::Generic { .. })),
        "the call is a generic node: {text}"
    );
    assert!(
        matches!(at(spread), Some(NodeOp::Effect { .. })),
        "the spread runs on a copy of the stack: {text}"
    );
}

/// Each generic node's inputs: its operands (the words its state's stack ends with), then the
/// values of the slots its operation names that its state holds as values, in their order (P3
/// stage 3 step 3a).
fn assert_generic_inputs(graph: &Graph, chunk: &Chunk, text: &str) {
    for node in &graph.nodes {
        let NodeOp::Generic { pc, pops, .. } = node.op else {
            continue;
        };
        let state = graph.state(node.state.expect("a generic node's state"));
        let pops = usize::from(pops);
        assert_eq!(
            node.inputs[..pops],
            state.stack[state.stack.len() - pops..],
            "pc {pc}'s operands: {text}"
        );
        let stale: Vec<NodeIx> =
            super::liveness::frame_slots(&chunk.jit_ops()[pc as usize], graph.n_slots)
                .iter()
                .filter_map(|&(slot, _)| match state.slots[usize::from(slot)] {
                    SlotSource::Value(value) => Some(value),
                    _ => None,
                })
                .collect();
        assert_eq!(node.inputs[pops..], stale[..], "pc {pc}'s slots: {text}");
    }
}

#[test]
fn a_slot_a_generic_node_writes_is_its_output() {
    // `s++` on a String is a generic node (UpdateLocal), which writes the slot: the slot after it
    // is its SlotResult, which the return reads, with no reload (P3 stage 3 step 3a).
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function upd(s) { s++; return s; } for (var k = 0; k < 20; k++) upd('1');",
    );
    let (chunk, graph) = graph_of(&mut engine, "upd");
    let text = dump(&graph, &chunk);
    let s = slot_named(&chunk, "s");
    let [generic] = nodes(&graph, |op| matches!(op, NodeOp::Generic { .. }))[..] else {
        panic!("one generic node: {text}")
    };
    assert!(
        matches!(chunk.jit_ops()[generic.pc as usize], Op::UpdateLocal(slot, _) if slot == s),
        "{text}"
    );
    let result = (0..graph.nodes.len() as u32)
        .map(NodeIx)
        .find(|&node| graph.node(node).op == NodeOp::SlotResult(s))
        .unwrap_or_else(|| panic!("the slot's result: {text}"));
    let generic_ix = graph.node(result).inputs[0];
    assert!(
        matches!(graph.node(generic_ix).op, NodeOp::Generic { .. }),
        "{text}"
    );
    let [ret] = nodes(&graph, |op| *op == NodeOp::Return)[..] else {
        panic!("one return: {text}")
    };
    assert_eq!(ret.inputs, [result], "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::FrameSlot(_))),
        0,
        "{text}"
    );
    assert_generic_inputs(&graph, &chunk, &text);
}

#[test]
fn a_join_of_slots_the_frames_held_takes_a_clean_phi() {
    // Each arm's `s++`/`s--` on a String is a generic node writing `s`: the join merges their
    // results by a phi (P3 stage 3 step 3a; the frame merged them before), Clean in the sense that
    // each arm's frame holds its input, which the return reads, with no reload.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function arms(s, c) { if (c) s++; else s--; return s; }
         for (var k = 0; k < 20; k++) arms('1', k & 1);",
    );
    let (chunk, graph) = graph_of(&mut engine, "arms");
    let text = dump(&graph, &chunk);
    let s = slot_named(&chunk, "s");
    let results: Vec<NodeIx> = (0..graph.nodes.len() as u32)
        .map(NodeIx)
        .filter(|&node| graph.node(node).op == NodeOp::SlotResult(s))
        .collect();
    assert_eq!(results.len(), 2, "{text}");
    let [ret] = nodes(&graph, |op| *op == NodeOp::Return)[..] else {
        panic!("one return: {text}")
    };
    let phi = graph.node(ret.inputs[0]);
    assert_eq!(phi.op, NodeOp::Phi, "{text}");
    let mut inputs = phi.inputs.clone();
    inputs.sort();
    assert_eq!(inputs, results, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::FrameSlot(_))),
        0,
        "{text}"
    );
}

#[test]
fn a_loop_header_takes_a_phi_for_the_slot_a_generic_node_writes() {
    // A `for…in` step is a generic node that writes its index slot: the loop's header takes a phi
    // for that slot (the prepass counts the node's writes, P3 stage 3 step 3a), whose back edge's
    // input is the step's SlotResult, and nothing in the loop reloads a slot.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function keys(o) { var n = 0; for (var k in o) n++; return n; }
         for (var r = 0; r < 20; r++) keys({a: 1, b: 2, c: 3});",
    );
    let (chunk, graph) = graph_of(&mut engine, "keys");
    let text = dump(&graph, &chunk);
    let step = pc_of(&chunk, |op| matches!(op, Op::ForInStepL(..)));
    let Op::ForInStepL(_, index, _) = chunk.jit_ops()[step] else {
        unreachable!()
    };
    let header = the_loop(&graph);
    let lp = loop_at(&graph, header);
    let result = (0..graph.nodes.len() as u32)
        .map(NodeIx)
        .find(|&node| graph.node(node).op == NodeOp::SlotResult(index))
        .unwrap_or_else(|| panic!("the step's result: {text}"));
    let phi = graph.blocks[lp.header]
        .phis
        .iter()
        .copied()
        .find(|&phi| graph.node(phi).inputs.contains(&result))
        .unwrap_or_else(|| panic!("a phi taking the step's result: {text}"));
    assert!(
        graph.node(phi).inputs.iter().any(|&input| input != result),
        "{text}"
    );
    for node in &graph.nodes {
        if lp.members[node.block.index()] {
            assert!(
                !matches!(node.op, NodeOp::FrameSlot(_) | NodeOp::StoreSlot(_)),
                "a reload or flush in the loop: {text}"
            );
        }
    }
    assert_generic_inputs(&graph, &chunk, &text);
}

#[test]
fn an_entry_value_read_on_one_arm_of_a_loop_merges_without_a_phi() {
    // `p`, an entry slot read on one arm of the loop's body only: the join takes the entry value
    // the arm read for the other arm too (the frame holds it there), so the back edge agrees with
    // the header (P3 stage 3 step 3a). The join made the slot Frame before, which the back edge
    // refused ("a loop's slot changes outside its phis").
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function arm(p, n) { var s = 0; for (var i = 0; i < n; i++) { if (i & 1) s += p; }
           return s; }
         for (var k = 0; k < 20; k++) arm(k, 10);",
    );
    let (chunk, graph) = graph_of(&mut engine, "arm");
    let text = dump(&graph, &chunk);
    let p = slot_named(&chunk, "p");
    assert_eq!(count(&graph, |op| *op == NodeOp::FrameSlot(p)), 0, "{text}");
    entry_value(&graph, &chunk, "p");
    // With two entry edges the header notes no invariant and holds `p` as the entry's: the join
    // in the body meets the arm's read (its entry value) and the other arm's Entry, whose input
    // is that same value, so that no phi and no load is made.
    eval(
        &mut engine,
        "function arm2(p, n, c) { var s = 0, i = 0; if (c) { i = 1; } else { i = 2; }
           while (i < n) { if (i & 1) s += p; i++; }
           return s; }
         for (var k = 0; k < 20; k++) arm2(k, 10, k & 1);",
    );
    let (chunk, graph) = graph_of(&mut engine, "arm2");
    let text = dump(&graph, &chunk);
    let lp = loop_at(&graph, the_loop(&graph));
    assert_eq!(lp.entries.len(), 2, "{text}");
    let p = slot_named(&chunk, "p");
    assert_eq!(count(&graph, |op| *op == NodeOp::FrameSlot(p)), 0, "{text}");
    entry_value(&graph, &chunk, "p");
}

#[test]
fn generic_nodes_take_their_operands_and_their_stale_named_slots() {
    // One function per family (P3 stage 3 step 3a): a call, a named load of an accessor (with
    // its receiver in a slot the frame holds, and in a dirty one; a data property's load through
    // the prototype chain is the tier's own since step L2, a getter's stays generic), an equality
    // of untyped words, a store that adds a property, an update of a property, a keyed method
    // load.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function call(f, x) { return f(x, 1); }
         function held(o) { return o.g; }
         function dirty(o) { var p = o.q; return p.g; }
         function eq(a, b) { return a == b; }
         function store(o, v) { var p = {}; p.z = v; o.w = p; }
         function update(o) { o.x++; }
         function keyed(o, k) { return o[k](); }
         function id(v) { return v; }
         var acc = {get g() { return 1; }}, holder = {q: acc};
         for (var k = 0; k < 20; k++) {
           call(id, k); held(acc); dirty(holder); eq(k, '1'); store({}, k);
           update({x: 'a'}); keyed([1, 2], 'pop');
         }",
    );
    for name in ["call", "held", "dirty", "eq", "store", "update", "keyed"] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert!(
            count(&graph, |op| matches!(op, NodeOp::Generic { .. })) > 0,
            "{name}: {text}"
        );
        assert_eq!(
            count(&graph, |op| matches!(
                op,
                NodeOp::StoreSlot(_) | NodeOp::FrameSlot(_)
            )),
            0,
            "{name}: no flush or reload: {text}"
        );
        assert_generic_inputs(&graph, &chunk, &text);
    }
    // The receiver's slot: none where the frame holds it (`held`), its value where it is dirty
    // (`dirty`).
    let local_load = |name: &str, receiver: &str, engine: &mut Engine| {
        let (chunk, graph) = graph_of(engine, name);
        let text = dump(&graph, &chunk);
        let slot = slot_named(&chunk, receiver);
        let pc = pc_of(
            &chunk,
            |op| matches!(op, Op::GetPropLocal(local, ..) if *local == slot),
        );
        let generic = nodes(&graph, |op| matches!(op, NodeOp::Generic { .. }))
            .into_iter()
            .find(|node| node.pc as usize == pc)
            .unwrap_or_else(|| panic!("{name}: the load's generic node: {text}"))
            .clone();
        (generic, text)
    };
    let (generic, text) = local_load("held", "o", &mut engine);
    assert!(generic.inputs.is_empty(), "held: {text}");
    let (generic, text) = local_load("dirty", "p", &mut engine);
    assert_eq!(generic.inputs.len(), 1, "dirty: {text}");
}

/// Sixteen locals made before a call and read after it (P3 stage 3 step 3a): the call, a generic
/// node, flushes none of them, so each crosses it in a home a call preserves, the callee-saved
/// registers (x23-x28, d8-d15) and, past them, spill slots, never a caller-saved register.
#[test]
fn values_across_a_generic_node_take_callee_saved_homes_and_spill_slots() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function many(a, g) { var x1 = a + 1, x2 = a * 2, x3 = a - 3, x4 = a / 4, x5 = a + 5.5,
           x6 = a * 6.5, x7 = a + 7, x8 = a + 8, x9 = a + 9, x10 = a + 10, x11 = a | 11,
           x12 = a ^ 12, x13 = a + 13, x14 = a & 14, x15 = a * 15, x16 = a + 16; g();
           return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15
             + x16; }
         function nothing() {}
         for (var k = 0; k < 20; k++) many(k + 0.5, nothing);",
    );
    let (chunk, graph) = graph_of(&mut engine, "many");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| matches!(
            op,
            NodeOp::StoreSlot(_) | NodeOp::FrameSlot(_)
        )),
        0,
        "{text}"
    );
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    {
        use super::regalloc::{allocate, Home, ARM64};
        let alloc = allocate(&graph, &ARM64).expect("allocated");
        super::regalloc::tests::assert_consistent(&graph, &alloc);
        let generic = (0..graph.nodes.len())
            .find(|&n| matches!(graph.nodes[n].op, NodeOp::Generic { .. }))
            .unwrap_or_else(|| panic!("the call: {text}"));
        let at = alloc.pos[generic];
        let across: Vec<Home> = (0..graph.nodes.len())
            .filter(|&v| {
                let (start, end) = alloc.ranges[v];
                start < at && at < end && !matches!(alloc.homes[v], Home::None | Home::Const)
            })
            .map(|v| alloc.homes[v])
            .collect();
        assert!(across.len() >= 16, "{across:?}: {text}");
        let callee = |home: &Home| match *home {
            Home::Gp(r) => ARM64.gp_callee.contains(&r),
            Home::Fp(r) => ARM64.fp_callee.contains(&r),
            _ => false,
        };
        assert!(
            across
                .iter()
                .all(|home| callee(home) || matches!(home, Home::Spill(_))),
            "{across:?}: {text}"
        );
        assert!(across.iter().any(callee), "{across:?}: {text}");
        assert!(
            across.iter().any(|home| matches!(home, Home::Spill(_))),
            "{across:?}: {text}"
        );
    }
}

#[test]
fn the_verifier_refuses_a_generic_nodes_missing_slot_input_and_a_stray_slot_result() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function dirty(o) { var p = o.q; return p.g; }
         function upd(s) { s++; return s; }
         var holder = {q: {get g() { return 1; }}};
         for (var k = 0; k < 20; k++) { dirty(holder); upd('1'); }",
    );
    let rejects = |chunk: &Chunk, what: &str, broken: Graph, expected: &str| {
        let error = verify(&broken, chunk).expect_err(what);
        assert!(error.contains(expected), "{what}: {error}");
    };
    // A generic node without the value of its dirty named slot.
    let (chunk, graph) = graph_of(&mut engine, "dirty");
    let generic = graph
        .nodes
        .iter()
        .position(|node| matches!(node.op, NodeOp::Generic { .. }) && !node.inputs.is_empty())
        .expect("the load with its slot input");
    let mut broken = graph.clone();
    broken.nodes[generic].inputs.pop();
    rejects(&chunk, "a missing slot input", broken, "stale slots");
    verify(&graph, &chunk).expect("the graph itself holds");
    // A slot result of a slot its node's operation does not write (the load reads `p`).
    let p = slot_named(&chunk, "p");
    let mut broken = graph.clone();
    let block = broken.nodes[generic].block;
    let stray = broken.new_node(
        block,
        NodeOp::SlotResult(p),
        &[NodeIx(generic as u32)],
        Rep::Tagged,
        broken.nodes[generic].pc,
    );
    let body = &mut broken.blocks[block.index()].body;
    let at = body
        .iter()
        .position(|&node| node.index() == generic)
        .expect("the load in its block");
    body.insert(at + 1, stray);
    rejects(&chunk, "another slot's result", broken, "does not write");
    // A slot result of no generic node.
    let (chunk, graph) = graph_of(&mut engine, "upd");
    let result = graph
        .nodes
        .iter()
        .position(|node| matches!(node.op, NodeOp::SlotResult(_)))
        .expect("the update's result");
    let mut broken = graph.clone();
    broken.nodes[result].inputs = vec![NodeIx(result as u32)];
    rejects(
        &chunk,
        "a result of another node",
        broken,
        "a slot's result of",
    );
    verify(&graph, &chunk).expect("the graph itself holds");
}

/// Heap facts end at a generic node, which may run any code (P3 stage 3 step 3a: as at an effect):
/// a field loaded before the call is loaded again after it, its object's shape checked again,
/// while the object's word itself, a value, is the same node.
#[test]
fn a_generic_node_ends_the_heap_facts_and_keeps_the_values() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function hf(o, g) { var a = o.x; g(o); return a + o.x; }
         function bump(o) { o.x = o.x + 1; }
         for (var k = 0; k < 20; k++) hf({x: k}, bump);",
    );
    let (chunk, graph) = graph_of(&mut engine, "hf");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::Generic { .. })),
        1,
        "{text}"
    );
    let loads = nodes(&graph, |op| matches!(op, NodeOp::LoadField { .. }));
    assert_eq!(
        loads.len(),
        2,
        "the field is loaded again after the call: {text}"
    );
    let shapes = nodes(&graph, |op| matches!(op, NodeOp::CheckShape(_)));
    assert_eq!(shapes.len(), 2, "its shape checked again: {text}");
    assert_eq!(shapes[0].inputs, shapes[1].inputs, "one word: {text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::FrameSlot(_))),
        0,
        "{text}"
    );
}

/// A throw from a generic node at a pc with a handler exits by recipe (P3 stage 3 step 3a, JSC's
/// exception exit): its recipe names the values the handler's frame needs, the slots its state
/// holds as values (`a`, live into the catch and after it; `c`, live into the catch alone, whose
/// last read is the call's own recipe) and the stack below the handler's depth (`fo`'s for-of,
/// whose handler, which closes the iterator, keeps the operand stack's first word: here the sum
/// the call's addition reads), and the allocator keeps each in a home the call preserves, which
/// the exit reads after the call.
#[test]
fn a_generic_nodes_throw_recipe_keeps_the_handlers_values_across_its_call() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function tr(o, g) { var a = o.x + 1, b = 0, c = o.x * 3;
           try { b = g(a); } catch (e) { return a * 10 + c + (e | 0); }
           return a + b; }
         function fo(d, g) { var s = 0; for (var x of d) { s += g(x); } return s; }
         function thrower(v) { if (v % 5 === 0) throw v + 1; return v * 2; }
         for (var k = 0; k < 20; k++) { tr({x: k}, thrower); fo([1, 2, 3], String); }",
    );
    for (name, slot_names, below) in [("tr", &["a", "c"][..], 0), ("fo", &[][..], 1)] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        let call = (0..graph.nodes.len() as u32)
            .map(NodeIx)
            .find(|&n| {
                let node = graph.node(n);
                matches!(node.op, NodeOp::Generic { .. })
                    && node
                        .state
                        .is_some_and(|state| !graph.state(state).handlers.is_empty())
            })
            .unwrap_or_else(|| panic!("{name}: the call in the try: {text}"));
        let (slots, stack) = graph.throw_recipe(call);
        for slot_name in slot_names {
            let slot = slot_named(&chunk, slot_name);
            assert!(
                slots.iter().any(|&(recipe, _)| recipe == slot),
                "{name}: {slot_name}: {slots:?}: {text}"
            );
        }
        let state = graph.state(graph.node(call).state.expect("a state"));
        assert_eq!(stack.len(), state.stack.len(), "{name}: {text}");
        let NodeOp::Generic { pops, .. } = graph.node(call).op else {
            unreachable!()
        };
        let (_, depth) = *state.handlers.last().expect("a handler");
        let kept = (state.stack.len() - usize::from(pops)).min(depth as usize);
        assert_eq!(
            kept, below,
            "{name}: the stack below the handler's depth: {text}"
        );
        for (k, value) in stack.iter().enumerate() {
            assert_eq!(value.is_some(), k < kept, "{name}: position {k}: {text}");
        }
        #[cfg(all(
            target_arch = "aarch64",
            any(target_os = "macos", target_os = "linux", target_os = "windows")
        ))]
        {
            use super::regalloc::{allocate, Home, ARM64};
            let alloc = allocate(&graph, &ARM64).expect("allocated");
            let preserved = |home: Home| match home {
                Home::Gp(r) => ARM64.gp_callee.contains(&r),
                Home::Fp(r) => ARM64.fp_callee.contains(&r),
                Home::Spill(_) | Home::Const => true,
                Home::None => false,
            };
            for value in slots
                .iter()
                .map(|&(_, value)| value)
                .chain(stack.iter().flatten().copied())
            {
                let home = alloc.homes[value.index()];
                assert!(preserved(home), "{name}: {value:?} in {home:?}: {text}");
                let (_, end) = alloc.ranges[value.index()];
                assert!(
                    end > alloc.pos[call.index()],
                    "{name}: {value:?} read after the call: {text}"
                );
            }
        }
    }
}
