//! The graph's invariants (P3 stage 1), checked between the builder and the back end, which may
//! rely on each of them:
//! - every node sits in exactly one block, the one it names, phis in the block's phis and every
//!   other node in its body;
//! - every block ends in exactly one terminator and nothing else is one, its successors are the
//!   terminator's targets, predecessor lists mirror them, and the entry reaches every block;
//! - a loop header's body opens with its poll, and a poll opens nothing else; entry slots are
//!   read in the entry block;
//! - every input is available where it is used: defined earlier in the block or in a dominating
//!   block, a phi's k-th input at the end of the block's k-th predecessor;
//! - representations agree with each NodeOp's documented inputs and output (ir.rs), an effect's
//!   operand and result counts with its operation's stack effect;
//! - every node that can exit has a frame state and a reason (an Exit carries its own), and every
//!   state describes its pc's frame: one source per slot of the chunk, the stack analysis's
//!   depth at the pc, handler records the chunk pushes at their depth, and values of a
//!   representation an exit can box, available at the node that exits; a slot a state marks
//!   Dead is dead at its pc (`opt::liveness`, checked in tests and debug builds);
//! - an address (Rep::Ptr) is never a phi's or a frame state's value, so no block edge or exit
//!   recipe carries one; the signatures let only element nodes take one;
//! - a loop's entry speculation (stage 2 step 6), CheckNumber [PhiNotNumber] or CheckInt32
//!   [PhiNotInt32], sits at the end of a loop header's entry predecessor, where nothing after it
//!   exits or has an effect before the jump, and exits at the header's pc with its handlers.

use super::ir::{BlockIx, ExitReason, FrameState, Graph, Node, NodeIx, NodeOp, Rep, SlotSource};
use super::liveness::SlotLiveness;
use crate::bytecode::{Chunk, Op};
use crate::jit_ir::Cfg;

/// Whether `graph`, built from `chunk`, holds the invariants the back end relies on; the first
/// violation found otherwise.
pub(crate) fn verify(graph: &Graph, chunk: &Chunk) -> Result<(), String> {
    let ops = chunk.jit_ops();
    let cfg = Cfg::build(chunk).map_err(|error| format!("the chunk's stack analysis: {error}"))?;
    let (_, n_slots) = chunk.jit_frame();
    if graph.n_slots != n_slots {
        return Err(format!(
            "the graph has {} slots, the chunk {n_slots}",
            graph.n_slots
        ));
    }
    let entry = graph.entry.ok_or("the graph has no entry block")?;
    if entry.index() >= graph.blocks.len() {
        return Err(format!("the entry {entry:?} is not a block"));
    }
    if !graph.block(entry).preds.is_empty() {
        return Err(format!("the entry {entry:?} has predecessors"));
    }
    let place = placement(graph)?;
    edges(graph, ops.len())?;
    let dom = Dominators::new(graph, entry)?;
    let available = |value: NodeIx, (block, index): (BlockIx, usize)| -> bool {
        let Some(&Some((defined, at))) = place.get(value.index()) else {
            return false;
        };
        if defined == block {
            at < index
        } else {
            dom.dominates(defined, block)
        }
    };
    for (index, node) in graph.nodes.iter().enumerate() {
        let ix = NodeIx(index as u32);
        let at = place[index].expect("placement placed every node");
        let fail = |what: &str| {
            Err(format!(
                "node {index} ({:?} at pc {}): {what}",
                node.op, node.pc
            ))
        };
        if node.pc as usize >= ops.len() {
            return fail("a pc past the chunk");
        }
        if let Err(what) = representations(graph, node, ix, chunk, n_slots) {
            return fail(&what);
        }
        if node.op == NodeOp::Phi {
            let preds = &graph.block(node.block).preds;
            for (k, &input) in node.inputs.iter().enumerate() {
                let Some(&Some((defined, _))) = place.get(input.index()) else {
                    return fail("an input that is no node");
                };
                if defined != preds[k] && !dom.dominates(defined, preds[k]) {
                    return fail(&format!(
                        "input {k} is not available at the end of predecessor {:?}",
                        preds[k]
                    ));
                }
            }
        } else if let Some(input) = node.inputs.iter().find(|&&input| !available(input, at)) {
            return fail(&format!("input {input:?} does not dominate its use"));
        }
        if node.op.can_exit() {
            if node.state.is_none() {
                return fail("an exit without a frame state");
            }
            if node.reason.is_none() && !matches!(node.op, NodeOp::Exit(_)) {
                return fail("an exit without a reason");
            }
        }
        if matches!(
            node.reason,
            Some(ExitReason::PhiNotNumber | ExitReason::PhiNotInt32)
        ) {
            if let Err(what) = speculation(graph, node, &dom, at.1) {
                return fail(&what);
            }
        }
        if let Some(state) = node.state {
            let Some(frame) = graph.states.get(state.index()) else {
                return fail("a frame state that does not exist");
            };
            if let Some(value) = state_values(frame).find(|&value| !available(value, at)) {
                return fail(&format!(
                    "its frame state's {value:?} does not dominate the node"
                ));
            }
        }
    }
    // The builder's liveness again, where its cost is a test's or a debug build's.
    let liveness =
        cfg!(any(test, debug_assertions)).then(|| SlotLiveness::compute(ops, &cfg, n_slots));
    for (index, frame) in graph.states.iter().enumerate() {
        frame_state(graph, frame, ops, &cfg, n_slots, liveness.as_ref())
            .map_err(|what| format!("frame state {index} (pc {}): {what}", frame.pc))?;
    }
    Ok(())
}

/// Each node's block and position (phis first, then the body), checking that every node is
/// listed once, in the block it names and in the right list.
fn placement(graph: &Graph) -> Result<Vec<Option<(BlockIx, usize)>>, String> {
    let mut place = vec![None; graph.nodes.len()];
    for (index, block) in graph.blocks.iter().enumerate() {
        let ix = BlockIx(index as u32);
        for (position, &node) in block.phis.iter().chain(&block.body).enumerate() {
            let data = graph
                .nodes
                .get(node.index())
                .ok_or_else(|| format!("block {index} lists {node:?}, which is no node"))?;
            if data.block != ix {
                return Err(format!(
                    "block {index} lists {node:?}, which names {:?}",
                    data.block
                ));
            }
            if (position < block.phis.len()) != (data.op == NodeOp::Phi) {
                return Err(format!(
                    "block {index}: {node:?} ({:?}) in the wrong list",
                    data.op
                ));
            }
            if place[node.index()].replace((ix, position)).is_some() {
                return Err(format!("{node:?} is listed twice"));
            }
        }
    }
    if let Some(index) = place.iter().position(Option::is_none) {
        return Err(format!("node {index} sits in no block"));
    }
    Ok(place)
}

/// Terminators and edges: one terminator per block, at its end, naming the block's successors,
/// and predecessor lists that mirror the successor lists.
fn edges(graph: &Graph, n_ops: usize) -> Result<(), String> {
    let mut balance = crate::fasthash::FastMap::<(u32, u32), i64>::default();
    for (index, block) in graph.blocks.iter().enumerate() {
        if block.first_pc as usize >= n_ops {
            return Err(format!("block {index} starts past the chunk"));
        }
        let Some((&last, rest)) = block.body.split_last() else {
            return Err(format!("block {index} is empty"));
        };
        if !graph.node(last).op.is_terminator() {
            return Err(format!("block {index} ends without a terminator"));
        }
        if block.loop_header && graph.node(block.body[0]).op != NodeOp::Poll {
            return Err(format!("loop header {index} does not open with a poll"));
        }
        if let Some(node) = rest
            .iter()
            .find(|&&node| graph.node(node).op.is_terminator())
        {
            return Err(format!(
                "block {index} has a terminator, {node:?}, before its end"
            ));
        }
        let targets = match graph.node(last).op {
            NodeOp::Jump(target) => vec![target],
            NodeOp::Branch {
                then_block,
                else_block,
            } => {
                if then_block == else_block {
                    return Err(format!("block {index} branches to one block twice"));
                }
                vec![then_block, else_block]
            }
            _ => Vec::new(),
        };
        if block.succs != targets {
            return Err(format!(
                "block {index}'s successors {:?} are not its terminator's {targets:?}",
                block.succs
            ));
        }
        for succ in &block.succs {
            if succ.index() >= graph.blocks.len() {
                return Err(format!(
                    "block {index} has the successor {succ:?}, no block"
                ));
            }
            *balance.entry((index as u32, succ.0)).or_default() += 1;
        }
        for pred in &block.preds {
            *balance.entry((pred.0, index as u32)).or_default() -= 1;
        }
    }
    if let Some(((from, to), _)) = balance.iter().find(|(_, &count)| count != 0) {
        return Err(format!(
            "the edge {from} -> {to} is in one of its blocks' lists and not the other's"
        ));
    }
    Ok(())
}

/// The graph's dominator tree, by reverse postorder from the entry.
struct Dominators {
    /// Each block's index in reverse postorder.
    order: Vec<u32>,
    /// Each block's immediate dominator, by reverse-postorder index.
    idom: Vec<u32>,
}

impl Dominators {
    fn new(graph: &Graph, entry: BlockIx) -> Result<Self, String> {
        const UNSEEN: u32 = u32::MAX;
        let mut seen = vec![false; graph.blocks.len()];
        let mut post = Vec::with_capacity(graph.blocks.len());
        let mut work = vec![(entry, 0usize)];
        seen[entry.index()] = true;
        while let Some(top) = work.last_mut() {
            let (block, next) = *top;
            if let Some(&succ) = graph.block(block).succs.get(next) {
                top.1 += 1;
                if !std::mem::replace(&mut seen[succ.index()], true) {
                    work.push((succ, 0));
                }
            } else {
                post.push(block);
                work.pop();
            }
        }
        if let Some(index) = seen.iter().position(|seen| !seen) {
            return Err(format!("the entry does not reach block {index}"));
        }
        post.reverse();
        let mut order = vec![UNSEEN; graph.blocks.len()];
        for (index, block) in post.iter().enumerate() {
            order[block.index()] = index as u32;
        }
        let mut idom = vec![UNSEEN; post.len()];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for index in 1..post.len() {
                let mut new = UNSEEN;
                for pred in &graph.block(post[index]).preds {
                    let p = order[pred.index()];
                    if idom[p as usize] == UNSEEN {
                        continue;
                    }
                    new = if new == UNSEEN {
                        p
                    } else {
                        let (mut a, mut b) = (p, new);
                        while a != b {
                            while a > b {
                                a = idom[a as usize];
                            }
                            while b > a {
                                b = idom[b as usize];
                            }
                        }
                        a
                    };
                }
                if new != idom[index] {
                    idom[index] = new;
                    changed = true;
                }
            }
        }
        Ok(Dominators { order, idom })
    }

    fn dominates(&self, a: BlockIx, b: BlockIx) -> bool {
        let a = self.order[a.index()];
        let mut b = self.order[b.index()];
        while b > a {
            b = self.idom[b as usize];
        }
        a == b
    }
}

/// Whether a node is a loop's entry speculation (its reason says so).
fn speculative(node: &Node) -> bool {
    matches!(
        node.reason,
        Some(ExitReason::PhiNotNumber | ExitReason::PhiNotInt32)
    )
}

/// A loop's entry speculation `node`, at `position` among its block's phis and body: its check
/// (CheckNumber for PhiNotNumber, CheckInt32 for PhiNotInt32) at the end of a loop header's entry
/// predecessor, after which nothing but other speculations exits or has an effect before the jump
/// to the header, with a state at the header's pc and the header's handlers.
fn speculation(
    graph: &Graph,
    node: &Node,
    dom: &Dominators,
    position: usize,
) -> Result<(), String> {
    let check = if node.reason == Some(ExitReason::PhiNotNumber) {
        NodeOp::CheckNumber
    } else {
        NodeOp::CheckInt32
    };
    if node.op != check {
        return Err(format!("a phi's speculation by {:?}", node.op));
    }
    let block = graph.block(node.block);
    let [header] = block.succs[..] else {
        return Err("a phi's speculation on a block of other than one successor".into());
    };
    let target = graph.block(header);
    if !target.loop_header || dom.dominates(header, node.block) {
        return Err("a phi's speculation off a loop's entry edge".into());
    }
    for &later in &block.body[position - block.phis.len() + 1..] {
        let later = graph.node(later);
        if later.op.is_terminator() {
            break;
        }
        if !speculative(later) && (later.op.can_exit() || later.op.is_effectful()) {
            return Err(format!(
                "a phi's speculation before {:?} in its predecessor",
                later.op
            ));
        }
    }
    let state = graph.state(node.state.ok_or("a phi's speculation without a state")?);
    if state.pc != target.first_pc {
        return Err(format!(
            "a phi's speculation exiting at pc {}, not its header's {}",
            state.pc, target.first_pc
        ));
    }
    let poll = graph
        .node(target.body[0])
        .state
        .map(|poll| &graph.state(poll).handlers);
    if poll != Some(&state.handlers) {
        return Err("a phi's speculation with other handlers than its header's".into());
    }
    Ok(())
}

/// The input and output representations of `op` as ir.rs documents them; None for the nodes
/// whose arity varies (phis, effects and their projections).
fn signature(op: &NodeOp) -> Option<(&'static [Rep], Rep)> {
    const TAGGED: &[Rep] = &[Rep::Tagged];
    const F64: &[Rep] = &[Rep::F64];
    const I32: &[Rep] = &[Rep::I32];
    const BOOL: &[Rep] = &[Rep::Bool];
    const CELL: &[Rep] = &[Rep::Cell];
    const F64S: &[Rep] = &[Rep::F64, Rep::F64];
    const I32S: &[Rep] = &[Rep::I32, Rep::I32];
    Some(match op {
        NodeOp::ConstTagged(_) | NodeOp::EntrySlot(_) | NodeOp::FrameSlot(_) | NodeOp::This => {
            (&[], Rep::Tagged)
        }
        NodeOp::ConstF64(_) => (&[], Rep::F64),
        NodeOp::ConstI32(_) => (&[], Rep::I32),
        NodeOp::ConstBool(_) => (&[], Rep::Bool),
        NodeOp::CheckNumber => (TAGGED, Rep::F64),
        NodeOp::CheckInt32 => (F64, Rep::I32),
        NodeOp::CheckObject | NodeOp::CheckShape(_) => (TAGGED, Rep::Cell),
        NodeOp::CheckBoolean | NodeOp::ToBoolean => (TAGGED, Rep::Bool),
        NodeOp::F64ToBoolean => (F64, Rep::Bool),
        NodeOp::I32ToF64 => (I32, Rep::F64),
        NodeOp::TruncateToInt32 => (F64, Rep::I32),
        NodeOp::I32UShrToF64 => (I32S, Rep::F64),
        NodeOp::F64ToTagged => (F64, Rep::Tagged),
        NodeOp::I32ToTagged => (I32, Rep::Tagged),
        NodeOp::BoolToTagged => (BOOL, Rep::Tagged),
        NodeOp::I32Add
        | NodeOp::I32Sub
        | NodeOp::I32Mul
        | NodeOp::I32And
        | NodeOp::I32Or
        | NodeOp::I32Xor
        | NodeOp::I32Shl
        | NodeOp::I32Sar
        | NodeOp::I32UShr => (I32S, Rep::I32),
        NodeOp::I32Not => (I32, Rep::I32),
        NodeOp::F64Add | NodeOp::F64Sub | NodeOp::F64Mul | NodeOp::F64Div | NodeOp::F64Mod => {
            (F64S, Rep::F64)
        }
        NodeOp::F64Neg => (F64, Rep::F64),
        NodeOp::I32Cmp(_) => (I32S, Rep::Bool),
        NodeOp::F64Cmp(_) => (F64S, Rep::Bool),
        NodeOp::TaggedSame => (&[Rep::Tagged, Rep::Tagged], Rep::Bool),
        NodeOp::BoolNot => (BOOL, Rep::Bool),
        NodeOp::LoadField { .. } => (CELL, Rep::Tagged),
        NodeOp::StoreField { .. } => (&[Rep::Cell, Rep::Tagged], Rep::None),
        NodeOp::CheckElements(_) => (CELL, Rep::None),
        NodeOp::LoadElements => (CELL, Rep::Ptr),
        NodeOp::LoadElementsUsed => (&[Rep::Ptr], Rep::I32),
        NodeOp::CheckIndex => (F64, Rep::I32),
        NodeOp::CheckBounds => (I32S, Rep::None),
        // The cell first: it keeps the words alive while their address is used.
        NodeOp::LoadElement(_) => (&[Rep::Cell, Rep::Ptr, Rep::I32], Rep::Tagged),
        NodeOp::LoadElementOrUndefined => (&[Rep::Cell, Rep::Ptr, Rep::I32, Rep::I32], Rep::Tagged),
        // Its value takes one of three representations (`representations`).
        NodeOp::StoreElement(_) => return None,
        NodeOp::LoadArrayLength => (CELL, Rep::Tagged),
        NodeOp::TypedView => (CELL, Rep::Ptr),
        NodeOp::TypedLength => (&[Rep::Ptr], Rep::I32),
        NodeOp::TypedData { .. } => (&[Rep::Cell, Rep::Ptr], Rep::Ptr),
        NodeOp::LoadTyped(kind) => (&[Rep::Cell, Rep::Ptr, Rep::I32], kind.rep(false)),
        NodeOp::LoadTypedOrUndefined(_) => {
            (&[Rep::Cell, Rep::Ptr, Rep::I32, Rep::I32], Rep::Tagged)
        }
        NodeOp::StoreTyped(kind) => match kind.rep(true) {
            Rep::F64 => (
                &[Rep::Cell, Rep::Ptr, Rep::Ptr, Rep::I32, Rep::F64],
                Rep::None,
            ),
            _ => (
                &[Rep::Cell, Rep::Ptr, Rep::Ptr, Rep::I32, Rep::I32],
                Rep::None,
            ),
        },
        NodeOp::ClampToUint8 => (F64, Rep::I32),
        NodeOp::TypedElement { write: false } => (&[Rep::Cell, Rep::I32], Rep::Tagged),
        NodeOp::TypedElement { write: true } => (&[Rep::Cell, Rep::I32, Rep::F64], Rep::None),
        NodeOp::IsTypedArray => (CELL, Rep::Bool),
        // Its value takes any representation an exit can box (`representations`).
        NodeOp::StoreSlot(_) => return None,
        NodeOp::Env | NodeOp::ConstEnv(_) => (&[], Rep::Env),
        NodeOp::EnvParent => (&[Rep::Env], Rep::Env),
        NodeOp::LoadBinding { .. } => (&[Rep::Env], Rep::Tagged),
        NodeOp::CheckInitialized => (TAGGED, Rep::None),
        NodeOp::StoreBinding { .. } => (&[Rep::Env, Rep::Tagged], Rep::None),
        NodeOp::GlobalObject => (&[], Rep::Cell),
        NodeOp::CheckEnvIs { .. } => (&[Rep::Env], Rep::None),
        NodeOp::CheckCellIs { .. } | NodeOp::CheckGlobalEpoch { .. } => (CELL, Rep::None),
        NodeOp::CheckEnvGeneration { .. } => (&[Rep::Env], Rep::None),
        NodeOp::LoadGlobalLexical { .. } => (&[Rep::Env], Rep::Tagged),
        NodeOp::Poll | NodeOp::Jump(_) | NodeOp::Exit(_) => (&[], Rep::None),
        NodeOp::Branch { .. } => (BOOL, Rep::None),
        NodeOp::Return => (TAGGED, Rep::None),
        NodeOp::Phi
        | NodeOp::Projection(_)
        | NodeOp::Effect { .. }
        | NodeOp::CheckCallee { .. }
        | NodeOp::CalleeMatches { .. } => return None,
    })
}

/// `node`'s representations, and the operands its operation names.
fn representations(
    graph: &Graph,
    node: &Node,
    ix: NodeIx,
    chunk: &Chunk,
    n_slots: usize,
) -> Result<(), String> {
    let rep_of = |input: &NodeIx| graph.nodes.get(input.index()).map(|node| node.rep);
    if node.inputs.iter().any(|input| rep_of(input).is_none()) {
        return Err("an input that is no node".into());
    }
    let reps: Vec<Rep> = node.inputs.iter().filter_map(rep_of).collect();
    match &node.op {
        NodeOp::Phi => {
            if node.rep == Rep::None {
                return Err("a phi without a representation".into());
            }
            if node.rep == Rep::Ptr {
                // An address dies with its effect epoch: a back edge would carry it across its
                // loop header's poll, a safepoint (ir.rs, Rep::Ptr).
                return Err("a phi of addresses".into());
            }
            if node.inputs.len() != graph.block(node.block).preds.len() {
                return Err(format!(
                    "{} inputs for {} predecessors",
                    node.inputs.len(),
                    graph.block(node.block).preds.len()
                ));
            }
            if reps.iter().any(|&rep| rep != node.rep) {
                return Err(format!("inputs {reps:?} for a {:?} phi", node.rep));
            }
        }
        &NodeOp::Projection(result) => {
            let effect = match node.inputs[..] {
                [effect] => graph.node(effect),
                _ => return Err("a projection of other than one node".into()),
            };
            match effect.op {
                NodeOp::Effect { pushes, .. } if result < pushes => {}
                _ => return Err(format!("a projection of {:?}", effect.op)),
            }
            if node.rep != Rep::Tagged {
                return Err(format!("a {:?} projection", node.rep));
            }
        }
        &NodeOp::Effect { pc, pops, pushes } => {
            if pc != node.pc {
                return Err(format!("an effect of pc {pc} built at another pc"));
            }
            if chunk.jit_stack_effect(pc as usize) != Some((pops as usize, pushes as usize)) {
                return Err(format!(
                    "pops {pops} and pushes {pushes}, the operation {:?}",
                    chunk.jit_stack_effect(pc as usize)
                ));
            }
            // Each operand in a representation the code boxes into its word (stage 2 step 6:
            // an untagged phi's uses need no retagging).
            if node.inputs.len() != pops as usize || !reps.iter().all(|&rep| boxable(rep)) {
                return Err(format!("operands {reps:?} for {pops} words"));
            }
            if node.rep != Rep::None {
                return Err(format!("a {:?} effect", node.rep));
            }
        }
        &NodeOp::CheckCallee { target } | &NodeOp::CalleeMatches { target } => {
            // The guard's InlineGuard is at the node's pc; its inputs are the callee's word and,
            // for a guard of `this`, the receiver's.
            let Some(&Op::InlineGuard(guard, _)) = chunk.jit_ops().get(node.pc as usize) else {
                return Err(format!("no InlineGuard at its pc ({ix:?})"));
            };
            if guard != target {
                return Err(format!("the InlineGuard of target {guard}, not {target}"));
            }
            let inputs = 1 + usize::from(chunk.jit_inline_target(target).check_this);
            if reps.len() != inputs || reps.iter().any(|&rep| rep != Rep::Tagged) {
                return Err(format!("operands {reps:?} for {inputs} words"));
            }
            let check = matches!(node.op, NodeOp::CheckCallee { .. });
            if (check && !matches!(node.rep, Rep::None | Rep::Tagged))
                || (!check && node.rep != Rep::Bool)
            {
                return Err(format!(
                    "a {:?} callee {}",
                    node.rep,
                    if check { "check" } else { "test" }
                ));
            }
        }
        NodeOp::StoreSlot(_) => {
            // The frame's word of a value of any representation an exit can box: the store boxes
            // it (stage 2 step 6).
            if !matches!(reps[..], [rep] if boxable(rep)) || node.rep != Rep::None {
                return Err(format!("({reps:?}) -> {:?} for a slot's store", node.rep));
            }
        }
        NodeOp::StoreElement(_) => {
            // The owner's cell, its words, the index, then the value as an int32, an f64 or a
            // word; the words after the store out.
            let value = matches!(reps[..], [_, _, _, Rep::I32 | Rep::F64 | Rep::Tagged]);
            if !value || reps[..3] != [Rep::Cell, Rep::Ptr, Rep::I32] || node.rep != Rep::Ptr {
                return Err(format!("({reps:?}) -> {:?} for an element store", node.rep));
            }
        }
        op => {
            let (inputs, output) = signature(op).expect("every other node has a signature");
            if reps != inputs || node.rep != output {
                return Err(format!(
                    "({reps:?}) -> {:?}, not ({inputs:?}) -> {output:?}",
                    node.rep
                ));
            }
        }
    }
    match node.op {
        NodeOp::EntrySlot(slot) | NodeOp::FrameSlot(slot) | NodeOp::StoreSlot(slot)
            if slot as usize >= n_slots =>
        {
            Err(format!("slot {slot} of {n_slots}"))
        }
        NodeOp::CheckShape(ref shapes) if shapes.is_empty() || shapes.len() > 4 => {
            Err(format!("{} shapes", shapes.len()))
        }
        NodeOp::EntrySlot(_) if Some(node.block) != graph.entry => {
            Err("an entry slot outside the entry block".into())
        }
        NodeOp::Poll
            if !graph.block(node.block).loop_header
                || graph.block(node.block).body.first() != Some(&ix) =>
        {
            Err("a poll that does not open a loop header".into())
        }
        _ => Ok(()),
    }
}

/// Whether a value of `rep` has a word an exit, a slot's store or an effect's operand can box it
/// into: a Number of an f64 or an int32, a Boolean, or a word.
fn boxable(rep: Rep) -> bool {
    matches!(rep, Rep::Tagged | Rep::F64 | Rep::I32 | Rep::Bool)
}

/// The values a frame state names, slots then stack.
fn state_values(frame: &FrameState) -> impl Iterator<Item = NodeIx> + '_ {
    frame
        .slots
        .iter()
        .filter_map(|source| match source {
            SlotSource::Value(value) => Some(*value),
            SlotSource::Frame | SlotSource::Dead => None,
        })
        .chain(frame.stack.iter().copied())
}

/// A frame state against its pc: the slot count, the stack analysis's depth, the handler records
/// the chunk pushes at their depths, values an exit can box, and Dead slots dead at the pc where
/// `liveness` is given.
fn frame_state(
    graph: &Graph,
    frame: &FrameState,
    ops: &[Op],
    cfg: &Cfg,
    n_slots: usize,
    liveness: Option<&SlotLiveness>,
) -> Result<(), String> {
    let pc = frame.pc as usize;
    if pc >= ops.len() {
        return Err("a pc past the chunk".into());
    }
    if frame.slots.len() != n_slots {
        return Err(format!(
            "{} slot sources for {n_slots} slots",
            frame.slots.len()
        ));
    }
    if cfg.stack_depth_at(pc) != Some(frame.stack.len()) {
        return Err(format!(
            "{} stack values where the depth is {:?}",
            frame.stack.len(),
            cfg.stack_depth_at(pc)
        ));
    }
    for value in state_values(frame) {
        match graph.nodes.get(value.index()).map(|node| node.rep) {
            Some(rep) if boxable(rep) => {}
            rep => return Err(format!("{value:?} of representation {rep:?}")),
        }
    }
    if let Some(liveness) = liveness {
        if let Some(slot) = (0..n_slots)
            .find(|&slot| frame.slots[slot] == SlotSource::Dead && liveness.live(pc, slot))
        {
            return Err(format!("slot {slot} is Dead where it is live"));
        }
    }
    for &(push, depth) in frame.handlers.iter() {
        let push = push as usize;
        if !matches!(
            ops.get(push),
            Some(Op::PushHandler(_) | Op::PushFinally(..) | Op::PushIterator(..))
        ) {
            return Err(format!("a handler record of pc {push}, which pushes none"));
        }
        if cfg.stack_depth_at(push) != Some(depth as usize) {
            return Err(format!(
                "a handler record of depth {depth} pushed at depth {:?}",
                cfg.stack_depth_at(push)
            ));
        }
    }
    Ok(())
}
