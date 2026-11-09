//! The front end's tests (P3 stage 1): small functions compiled and warmed as the engine runs
//! them, their snapshot taken, their graph built and verified, and its shape asserted.

use super::build::build;
use super::ir::*;
use super::snapshot::{fingerprint, Ran, Snapshot};
use super::verify::verify;
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

/// The verified graph of `chunk`.
fn graph_for(chunk: &Chunk, name: &str) -> Graph {
    let snapshot = Snapshot::take(chunk);
    let graph = build(chunk, &snapshot)
        .unwrap_or_else(|refusal| panic!("{name}: {refusal:?}\n{:#?}", chunk.jit_ops()));
    if let Err(error) = verify(&graph, chunk) {
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
        Ok(graph) => panic!("{name} was built\n{}", dump(&graph, &chunk)),
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
    matches!(op, NodeOp::Effect { .. })
}

/// The pc of `name`'s one operation `which` accepts.
fn pc_of(chunk: &Chunk, which: impl Fn(&Op) -> bool) -> usize {
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| which(&chunk.jit_ops()[pc]))
        .collect();
    assert_eq!(pcs.len(), 1, "one such operation: {:?}", chunk.jit_ops());
    pcs[0]
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
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function sum(n) { var s = 0; for (var i = 0; i < n; i++) s += i; return s; }
         for (var k = 0; k < 20; k++) sum(10);",
    );
    let (chunk, graph) = graph_of(&mut engine, "sum");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let headers: Vec<&BlockData> = graph
        .blocks
        .iter()
        .filter(|block| block.loop_header)
        .collect();
    assert_eq!(headers.len(), 1, "{text}");
    let header = headers[0];
    assert_eq!(header.preds.len(), 2, "an entry and a back edge: {text}");
    // The header's poll opens its body and exits, on a throw, with the header's state.
    let poll = graph.node(header.body[0]);
    assert_eq!(poll.op, NodeOp::Poll, "{text}");
    assert_eq!(poll.reason, Some(ExitReason::Throw), "{text}");
    let state = graph.state(poll.state.expect("a poll has a state"));
    assert_eq!(state.pc, header.first_pc, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::Poll), 1, "{text}");
    // `s` and `i` are assigned in the loop; `n` is not, and the loop's arithmetic could have been
    // an effect, so it is read from the frame.
    assert_eq!(header.phis.len(), 2, "{text}");
    for &phi in &header.phis {
        assert_eq!(graph.node(phi).inputs.len(), 2, "{text}");
        assert_eq!(graph.node(phi).rep, Rep::Tagged, "{text}");
    }
    assert_eq!(
        count(&graph, |op| *op == NodeOp::F64Cmp(Cond::Lt)),
        1,
        "{text}"
    );
    // `i`'s check in the header serves `s += i` and `i++` too.
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckNumber), 3, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Add), 2, "{text}");
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
    // The entry block holds the hoisted constants and the phis' entry inputs, and jumps.
    assert!(
        entry
            .body
            .iter()
            .all(|&node| graph.node(node).state.is_none()),
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
    // `x` in another slot of the second shape: the generic access.
    let (chunk, graph) = graph_of(&mut engine, "other");
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckShape(_))),
        0,
        "{text}"
    );
    assert_eq!(count(&graph, is_effect), 1, "{text}");
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
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function call(o, g) { var x = 1; x = x + 1; g(); return x + o.x; }
         var p = {x: 1}; function nothing() {}
         for (var i = 0; i < 20; i++) call(p, nothing);",
    );
    let (chunk, graph) = graph_of(&mut engine, "call");
    let text = dump(&graph, &chunk);
    let block = graph.block(graph.entry.expect("an entry"));
    let position = |which: &dyn Fn(&NodeOp) -> bool| {
        block
            .body
            .iter()
            .position(|&node| which(&graph.node(node).op))
            .unwrap_or_else(|| panic!("{text}"))
    };
    let effect = position(&is_effect);
    let store = position(&|op| *op == NodeOp::StoreSlot(2));
    let reload = position(&|op| *op == NodeOp::FrameSlot(2));
    assert!(store < effect && effect < reload, "{text}");
    let effect = graph.node(block.body[effect]);
    assert_eq!(effect.reason, Some(ExitReason::Throw), "{text}");
    let state = graph.state(effect.state.expect("an effect has a state"));
    assert!(
        state.slots.iter().all(|slot| *slot == SlotSource::Frame),
        "the flush left every slot in the frame: {text}"
    );
    assert_eq!(state.pc, effect.pc, "{text}");
    // After the effect, `o` too is read back and checked again.
    assert_eq!(count(&graph, |op| *op == NodeOp::FrameSlot(0)), 1, "{text}");
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
fn a_slot_in_its_temporal_dead_zone_holds_the_marker_in_states() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function tdz(c) { let x = c; { let y = x + 1; x = y; } return x; }
         for (var i = 0; i < 20; i++) tdz(i);",
    );
    let (chunk, graph) = graph_of(&mut engine, "tdz");
    let text = dump(&graph, &chunk);
    let check = nodes(&graph, |op| *op == NodeOp::CheckNumber);
    assert_eq!(check.len(), 1, "{text}");
    let state = graph.state(check[0].state.expect("a check has a state"));
    let marked = state.slots.iter().filter(|slot| {
        matches!(slot, SlotSource::Value(value)
            if graph.node(*value).op == NodeOp::ConstTagged(PACK_EMPTY))
    });
    assert_eq!(marked.count(), 1, "`y` before its declaration runs: {text}");
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
    verify(&graph, &chunk).expect("the graph itself holds");
}

// ---- the exit profile's declines (stage 2, step 3) ----------------------------------------------

/// Flag `reason` at `pc` of `chunk`, as a discard keys an exit there (`opt::profile`): `inlined`
/// for a site a splice took, `ways` the shapes a WrongShape exit's check expected.
fn flag(chunk: &Chunk, pc: usize, reason: ExitReason, inlined: bool, ways: u8) {
    super::profile::ChunkProfile::of(chunk).add_site(pc as u32, reason as u8, inlined, ways);
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
    assert!(
        nodes(&graph, |op| *op == NodeOp::CheckElements(ElemKind::Array))
            .iter()
            .all(|node| node.reason == Some(ExitReason::WrongElements)),
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
    let mut engine = vm_engine(
        "function str(s, i) { return s[i]; } function typed(t, i) { return t[i]; }
         function args(x, i) { return x[i]; } function generic(o, k) { return o[k]; }
         function argsof() { return arguments; } var u8 = new Uint8Array(4);
         var list = argsof(1, 2);
         for (var k = 0; k < 20; k++) {
           str('abc', 1); typed(u8, 1); args(list, 1); generic({a: 1}, 'a');
         }",
    );
    for name in ["str", "typed", "args", "generic"] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(
            count(&graph, |op| matches!(op, NodeOp::LoadElement(_))),
            0,
            "{name}: {text}"
        );
        assert!(count(&graph, is_effect) >= 1, "{name}: {text}");
    }
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
