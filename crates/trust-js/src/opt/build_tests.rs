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

fn engine(tier: Tier) -> Engine {
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
fn bitwise_operators_are_int32_and_an_unsigned_shift_by_zero_stays_generic() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function bits(a, b) { return (a | 0) + (b >>> 1) + (a >>> 0); }
         for (var i = 0; i < 20; i++) bits(i, i);",
    );
    let (chunk, graph) = graph_of(&mut engine, "bits");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, |op| *op == NodeOp::I32Or), 1, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::I32UShr), 1, "{text}");
    // Two proven int32s add as int32s, with an overflow exit.
    let add = nodes(&graph, |op| *op == NodeOp::I32Add);
    assert_eq!(add.len(), 1, "{text}");
    assert_eq!(add[0].reason, Some(ExitReason::Overflow), "{text}");
    // `a`'s int32 check is made once, for its first use.
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckInt32), 2, "{text}");
    // `>>> 0` can leave the int32 range: the generic operation.
    let effects = nodes(&graph, is_effect);
    assert_eq!(effects.len(), 1, "{text}");
    assert!(
        matches!(chunk.jit_ops()[effects[0].pc as usize], Op::UShr),
        "{text}"
    );
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
