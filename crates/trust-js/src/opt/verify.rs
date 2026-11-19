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
//! - a CheckShape that tests no tag (`object`), or a callee guard that does not test its receiver
//!   (`this_object`), takes a word that a CheckObject or a CheckShape of it proved an object's on
//!   every path to it (`proven_objects`; P3 stage 3, c-tag);
//! - an address (Rep::Ptr) is never a phi's or a frame state's value, so no block edge or exit
//!   recipe carries one; the signatures let only element nodes take one;
//! - a loop's entry speculation (stage 2 step 6), CheckNumber [PhiNotNumber] or CheckInt32
//!   [PhiNotInt32], or a loop invariant's check moved to the loop's entry (P3 stage 3 step 0),
//!   CheckNumber, CheckInt32, CheckObject or CheckElements [HoistingFailed], or a typed array's
//!   TypedData checked there (step 0c) [HoistingFailed], or the first CheckShape of an object a
//!   quiet loop takes made there (the quiet loop's facts, commit 2) [HoistingFailed], or a loop's
//!   global read's own CheckEnvIs,
//!   CheckCellIs, CheckGlobalEpoch or CheckEnvGeneration made there (F4 part 1) [WrongName: a
//!   WrongName check is one where it is built for the first pc of its block's one successor, from
//!   outside that successor's loop], sits at the end of a
//!   loop header's entry predecessor, where nothing after it exits or has an effect before the
//!   jump, and exits at the header's pc with its handlers. A HoistingFailed check (or a global
//!   read's WrongName check) may also stand
//!   before a block that heads a loop of its frame's bytecode (the frame's Cfg) and is no loop
//!   header in the graph, its one predecessor's only successor: the builder makes a loop's
//!   header state, and step 0c's entry checks of its typed arrays (`Builder::loop_entry_checks`),
//!   at the header's first entry, before any back edge, and a block becomes a loop header only at
//!   its first back edge (`Builder::back_edge`), which the build does not reach where every path
//!   to the latch ends in an exit (a block that never ran; an operation the rules lower whose
//!   operands the builder cannot take, `Builder::mismatch`, as zlib's). The exit is the same: the
//!   header's pc, the edge's state, whose flag leaves the checks at their sites. Such a block has
//!   no poll whose state names the header's handlers, so the rule compares none there: the
//!   state is the edge's, as the header's is (the builder refuses a block whose edges' handlers
//!   differ), and its records are checked as every state's. The phi selector's speculations and
//!   hoisted checks take only headers with a back edge (`phis.rs`);
//! - a loop whose header kept its entry's heap facts runs, in any of its frames, no effect, and no
//!   store of it writes a field location or binding slot whose value the header kept (the union
//!   rule, [`kept_heap`]).
//!
//! Inlined frames (P3 stage 3, step 1; `frames`, which the back end checks too):
//! - each frame's states and nodes are checked against that frame's chunk and Cfg: a node's pc
//!   and a block's first pc (its first node's frame) are pcs of their frame's chunk, and a node
//!   that exits rebuilds its own frame's state;
//! - an inlined frame's chunk is one of the compile's; its parent state is an earlier frame's,
//!   at its call's pc + 1, of the call's depth less its pops (the exit pushes the result), and
//!   no node's own state; the call is a Call or a CallWithThis; its callee and receiver are
//!   Tagged nodes, a Call's receiver undefined;
//! - an inlined frame's chunk is a callee a call enters as its rebuilt frame runs it (no
//!   activation, rest parameter, arguments value, prepared entry or coroutine), of the compiled
//!   function's strictness, whose frame fits a pooled activation record (the splice's callee
//!   set, the plan's question 6; step 9 widens it);
//! - an inlined frame's states have no `Frame` slot (the frame has none while the code runs: the
//!   owner's explicit operands) and no handler record (step 9's), and its nodes read no context of
//!   the compiled frame's (`NodeOp::reads_compiled_frame`);
//! - an exit's chain, each inlined frame's callee and receiver and its parent state's values, is
//!   available at the node that exits.

use super::ir::{
    BlockIx, ExitReason, FrameIx, FrameState, Graph, Node, NodeIx, NodeOp, Rep, SlotSource,
};
use super::liveness::SlotLiveness;
use super::view::FrozenChunk;
use crate::bytecode::Op;
use crate::jit_ir::Cfg;

/// A frame's chunk as the verifier and the back end read it (stage 3): its frozen view and Cfg.
pub(crate) type FrameChunk<'a> = (&'a FrozenChunk, &'a Cfg);

/// [`verify_view`] against a live chunk's frozen view (`Chunk::freeze`): the tests' path.
#[cfg(test)]
pub(crate) fn verify(
    graph: &Graph,
    chunk: &crate::bytecode::Chunk,
    cfg: &Cfg,
) -> Result<(), String> {
    verify_view(graph, &chunk.freeze(), cfg)
}

/// Whether `graph`, built from `chunk` (a chunk's frozen view), holds the invariants the back end
/// relies on; the first violation found otherwise. `cfg` is the chunk's, as its build made it
/// (`build::Built`). Reads nothing but its arguments, so that it runs on any thread. The tests';
/// a job verifies with its inlined frames' chunks (`verify_frames`).
#[cfg(test)]
pub(crate) fn verify_view(graph: &Graph, chunk: &FrozenChunk, cfg: &Cfg) -> Result<(), String> {
    verify_frames(graph, (chunk, cfg), &[])
}

/// [`verify_view`] of a graph with inlined frames (stage 3): `compiled` is the compiled
/// function's chunk, `inlined` the chunks its inlined frames name (`InlinedFrame::chunk`).
pub(crate) fn verify_frames(
    graph: &Graph,
    compiled: FrameChunk<'_>,
    inlined: &[FrameChunk<'_>],
) -> Result<(), String> {
    let (chunk, _) = compiled;
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
    frames(graph, compiled, inlined)?;
    // Each frame's chunk, by the frame's index.
    let chunks: Vec<FrameChunk<'_>> = std::iter::once(compiled)
        .chain(
            graph
                .frames
                .iter()
                .map(|frame| inlined[frame.chunk as usize]),
        )
        .collect();
    let chunk_of = |frame: FrameIx| chunks.get(frame.index()).copied();
    let place = placement(graph)?;
    edges(graph, |frame| {
        chunk_of(frame).map(|(view, _)| view.jit_ops().len())
    })?;
    let dom = Dominators::new(graph, entry)?;
    // The states that are a frame's parent: no node's own (`frames`).
    let mut parent = vec![false; graph.states.len()];
    for frame in &graph.frames {
        parent[frame.parent.index()] = true;
    }
    // Each frame's chain values, which an exit of a state of the frame reads too.
    let chain: Vec<Vec<NodeIx>> = (0..chunks.len())
        .map(|frame| graph.chain_values(FrameIx(frame as u32)))
        .collect();
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
        let Some((frame_chunk, _)) = chunk_of(node.frame) else {
            return fail("a frame that does not exist");
        };
        if node.pc as usize >= frame_chunk.jit_ops().len() {
            return fail("a pc past the chunk");
        }
        // A callee guard of a site nested in an inlined frame (P3 stage 3 step 3d) reads its
        // explicit inputs, its data in the compiled chunk's view (every way's) and the running
        // global environment, the same in every frame of the code (one Realm), but no caller
        // environment (a guard whose way requires one, `expected_env`, reads the compiled
        // frame's).
        let frame_free_guard = match node.op {
            NodeOp::CheckCallee { target, .. } | NodeOp::CalleeMatches { target, .. } => {
                chunk_of(FrameIx::COMPILED).is_some_and(|(compiled, _)| {
                    compiled.jit_inline_target(target).expected_env == 0
                })
            }
            _ => false,
        };
        if !node.frame.is_compiled() && node.op.reads_compiled_frame() && !frame_free_guard {
            return fail("a node of an inlined frame reads the compiled frame's context");
        }
        // Against its own frame's chunk (an inlined frame's generic node's operation and slots
        // are its callee's: P3 stage 3 step 3c).
        let frame_slots = if node.frame.is_compiled() {
            n_slots
        } else {
            frame_chunk.jit_frame().1
        };
        // Every way's guard data is the compiled chunk's view's (a nested site's too, step 3d).
        let compiled = chunk_of(FrameIx::COMPILED).map_or(frame_chunk, |(compiled, _)| compiled);
        if let Err(what) = representations(graph, node, ix, frame_chunk, compiled, frame_slots) {
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
        if speculative(graph, node, &dom) {
            let Some((_, frame_cfg)) = chunk_of(node.frame) else {
                return fail("a frame that does not exist");
            };
            if let Err(what) = speculation(graph, node, &dom, at.1, frame_cfg) {
                return fail(&what);
            }
        }
        if let Some(state) = node.state {
            let Some(frame) = graph.states.get(state.index()) else {
                return fail("a frame state that does not exist");
            };
            if frame.frame != node.frame {
                return fail(&format!(
                    "a state of frame {:?}, not its own {:?}",
                    frame.frame, node.frame
                ));
            }
            if parent[state.index()] {
                return fail("an inlined frame's parent state as its own");
            }
            if let Some(value) = state_values(frame).find(|&value| !available(value, at)) {
                return fail(&format!(
                    "its frame state's {value:?} does not dominate the node"
                ));
            }
            if let Some(value) = chain[node.frame.index()]
                .iter()
                .find(|&&value| !available(value, at))
            {
                return fail(&format!(
                    "its frame's chain's {value:?} does not dominate the node"
                ));
            }
        }
    }
    proven_objects(graph, &dom)?;
    kept_heap(graph, &dom)?;
    folded_reads(graph)?;
    // The builder's liveness again, where its cost is a test's or a debug build's: each frame's
    // chunk's.
    let liveness: Vec<Option<SlotLiveness>> = chunks
        .iter()
        .map(|&(view, cfg)| {
            cfg!(any(test, debug_assertions))
                .then(|| SlotLiveness::compute(view.jit_ops(), cfg, view.jit_frame().1))
        })
        .collect();
    for (index, frame) in graph.states.iter().enumerate() {
        let what = |what: String| format!("frame state {index} (pc {}): {what}", frame.pc);
        let (view, cfg) =
            chunk_of(frame.frame).ok_or_else(|| what("a frame that does not exist".into()))?;
        frame_state(
            graph,
            frame,
            (view, cfg),
            parent[index],
            liveness[frame.frame.index()].as_ref(),
        )
        .map_err(what)?;
    }
    Ok(())
}

/// The inlined frames' rules (see the module's doc), which the back end checks as well
/// (`a64::check_graph`), since its exits rebuild the frames they describe.
pub(crate) fn frames(
    graph: &Graph,
    compiled: FrameChunk<'_>,
    inlined: &[FrameChunk<'_>],
) -> Result<(), String> {
    for (k, frame) in graph.frames.iter().enumerate() {
        let ix = FrameIx(k as u32 + 1);
        let fail = |what: String| Err(format!("inlined frame {}: {what}", ix.0));
        let Some(&(callee, callee_cfg)) = inlined.get(frame.chunk as usize) else {
            return fail(format!("chunk {}, which the compile has not", frame.chunk));
        };
        let Some(parent) = graph.states.get(frame.parent.index()) else {
            return fail("a parent state that does not exist".into());
        };
        // Frames open outermost first: a parent's frame precedes its children.
        if parent.frame >= ix {
            return fail(format!("a parent state of frame {:?}", parent.frame));
        }
        let (caller, caller_cfg) = match graph.inlined(parent.frame) {
            None => compiled,
            Some(caller) => inlined[caller.chunk as usize],
        };
        let call = frame.call_pc as usize;
        let op = caller.jit_ops().get(call);
        if !matches!(op, Some(Op::Call(..) | Op::CallWithThis(..))) {
            return fail(format!("a call at pc {call} by {op:?}"));
        }
        if parent.pc as usize != call + 1 {
            return fail(format!(
                "a parent state at pc {}, not past its call at {call}",
                parent.pc
            ));
        }
        // Without the call's operands: the exit pushes the result.
        let depth = caller_cfg
            .stack_depth_at(call)
            .zip(caller.jit_stack_effect(call))
            .and_then(|(depth, (pops, _))| depth.checked_sub(pops));
        if depth != Some(parent.stack.len()) {
            return fail(format!(
                "a parent state of depth {}, where its call's less its operands is {depth:?}",
                parent.stack.len()
            ));
        }
        for (what, value) in [("callee", frame.callee), ("receiver", frame.receiver)] {
            if graph.nodes.get(value.index()).map(|node| node.rep) != Some(Rep::Tagged) {
                return fail(format!("a {what} that is no Tagged node"));
            }
        }
        if matches!(op, Some(Op::Call(..)))
            && graph.node(frame.receiver).op
                != NodeOp::ConstTagged(crate::value::Value::Undefined.bits())
        {
            return fail("a Call's receiver that is not undefined".into());
        }
        // A callee a call enters as the rebuilt frame runs it, on a pooled record.
        if callee.jit_is_resumable() || callee.prepared_entry || callee.frame_args_needed {
            return fail(
                "a callee whose call makes more than its frame (an activation, a rest parameter, \
                 an arguments value, a prepared entry or a coroutine)"
                    .into(),
            );
        }
        if callee.strict != compiled.0.strict {
            return fail("a callee of another strictness".into());
        }
        if callee.jit_frame().1 + callee_cfg.jit_stack_capacity() > crate::jit::FRAME_BUF {
            return fail("a callee's frame past a pooled activation record".into());
        }
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
/// and predecessor lists that mirror the successor lists. `n_ops` gives a frame's chunk's
/// operation count: a block's first pc is a pc of its first node's frame.
fn edges(graph: &Graph, n_ops: impl Fn(FrameIx) -> Option<usize>) -> Result<(), String> {
    let mut balance = crate::fasthash::FastMap::<(u32, u32), i64>::default();
    for (index, block) in graph.blocks.iter().enumerate() {
        let Some((&last, rest)) = block.body.split_last() else {
            return Err(format!("block {index} is empty"));
        };
        let first = block.phis.first().unwrap_or(&block.body[0]);
        if n_ops(graph.node(*first).frame).is_none_or(|n_ops| block.first_pc as usize >= n_ops) {
            return Err(format!("block {index} starts past the chunk"));
        }
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

/// The union rule (P3 stage 3 step 0c; F4 part 1, whose first form missed it: `build::Writes`):
/// a loop whose header kept its entry's heap facts (`Graph::kept`) runs, in any of its frames, no
/// effect and no generic node, none of its stores writes a field location or a binding slot whose
/// value the header kept, and where it kept a shape fact, no creation changes a shape (the quiet
/// loop's facts). Its loop is the natural loop of the header's back edges in the
/// graph, with the blocks of every frame inlined into it: the graph's statement of what the loop
/// prepass, which runs on bytecode before the build, summed over the frames from their chunks.
/// Maglev checks its loop's end state against an optimistic header's (TryMergeLoop and
/// IsCompatibleWithLoopHeader, maglev-interpreter-frame-state.cc:423-440,
/// maglev-known-node-aspects.cc:742-838), peeling where they disagree; here a disagreement is a
/// fault of the prepass.
fn kept_heap(graph: &Graph, dom: &Dominators) -> Result<(), String> {
    for kept in &graph.kept {
        // No loop header (the build reached no back edge): no loop.
        let Some(header) = kept_header(graph, kept) else {
            continue;
        };
        let latches: Vec<BlockIx> = graph
            .block(header)
            .preds
            .iter()
            .copied()
            .filter(|&pred| dom.dominates(header, pred))
            .collect();
        // No back edge (the build reached none): no loop.
        let mut member = vec![false; graph.blocks.len()];
        member[header.index()] = !latches.is_empty();
        let mut work = latches;
        while let Some(block) = work.pop() {
            if !std::mem::replace(&mut member[block.index()], true) {
                work.extend(graph.block(block).preds.iter().copied());
            }
        }
        for (index, block) in graph.blocks.iter().enumerate() {
            if !member[index] {
                continue;
            }
            for &node in block.phis.iter().chain(&block.body) {
                let data = graph.node(node);
                let what = match heap_write(&data.op) {
                    HeapWrite::Any => "an effect",
                    HeapWrite::Creation(_) if kept.shapes => {
                        "a creation (a shape transition), where it kept shape facts"
                    }
                    HeapWrite::Field(location) | HeapWrite::Creation(location)
                        if kept.fields.contains(&location) =>
                    {
                        "a store at the location of a field it kept"
                    }
                    HeapWrite::Binding(slot) if kept.slots.contains(&slot) => {
                        "a store at the slot of a binding it kept"
                    }
                    _ => continue,
                };
                return Err(format!(
                    "node {} ({:?} at pc {}, frame {:?}, block {index}): {what}, in the loop of \
                     header {header:?}, which kept its entry's heap facts",
                    node.index(),
                    data.op,
                    data.pc,
                    data.frame
                ));
            }
        }
    }
    Ok(())
}

/// Every folded global read is dominated within its effect epoch by a test of the code's
/// validity word (F4 part 2, layer 2): a forward pass whose bit a `CheckCodeValid` sets and an
/// effect or a generic node clears ([`heap_write`]'s `Any`: only these may run JavaScript, so
/// only these may change, remove or shadow a global property; a lowered store reaches only a
/// NotConstant property, which no compile folds), met by AND at a join, the entry's and an
/// unreached block's clear; each `FoldedGlobal` needs it set. A loop that keeps the bit through
/// its body needs no case of its own: its back edge brings the bit its entry had.
fn folded_reads(graph: &Graph) -> Result<(), String> {
    // A FoldedCell untags without a test: its input must be a folded object's word (layer 4a).
    for (index, node) in graph.nodes.iter().enumerate() {
        if node.op == NodeOp::FoldedCell {
            let input = node.inputs.first().map(|&input| &graph.node(input).op);
            let Some(&NodeOp::FoldedGlobal { word }) = input else {
                return Err(format!("node {index} (a FoldedCell) of no folded global"));
            };
            if word >> 48 != crate::value::PACK_OBJ >> 48 {
                return Err(format!(
                    "node {index} (a FoldedCell) of a folded word {word:#x}, no object's"
                ));
            }
        }
    }
    if !graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, NodeOp::FoldedGlobal { .. }))
    {
        return Ok(());
    }
    let entry = graph.entry.ok_or("the graph has no entry block")?;
    // The bit at each block's end, from "set" down (a must-analysis's top), until stable.
    let mut out = vec![true; graph.blocks.len()];
    let walk = |block: BlockIx, mut bit: bool, check: bool| -> Result<bool, String> {
        let data = graph.block(block);
        for &node in data.phis.iter().chain(&data.body) {
            let op = &graph.node(node).op;
            match op {
                NodeOp::CheckCodeValid => bit = true,
                NodeOp::FoldedGlobal { .. } if check && !bit => {
                    return Err(format!(
                        "node {} (a folded global read at pc {}, block {}) without a validity                          test in its effect epoch",
                        node.index(),
                        graph.node(node).pc,
                        block.index()
                    ));
                }
                _ if matches!(heap_write(op), HeapWrite::Any) => bit = false,
                _ => {}
            }
        }
        Ok(bit)
    };
    let in_bit = |block: BlockIx, out: &[bool]| -> bool {
        let preds = &graph.block(block).preds;
        block != entry && !preds.is_empty() && preds.iter().all(|pred| out[pred.index()])
    };
    loop {
        let mut changed = false;
        for index in 0..graph.blocks.len() {
            let block = BlockIx(index as u32);
            let bit = walk(block, in_bit(block, &out), false)?;
            if bit != out[index] {
                out[index] = bit;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for index in 0..graph.blocks.len() {
        let block = BlockIx(index as u32);
        walk(block, in_bit(block, &out), true)?;
    }
    Ok(())
}

/// Tests: the folded reads' rule alone ([`folded_reads`]) on `graph`.
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(crate) fn verify_folded_reads(graph: &Graph) -> Result<(), String> {
    folded_reads(graph)
}

/// Tests: the union rule alone ([`kept_heap`]) on `graph`, whatever else it breaks.
#[cfg(all(
    test,
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
pub(crate) fn verify_kept_heap(graph: &Graph) -> Result<(), String> {
    let entry = graph.entry.ok_or("the graph has no entry block")?;
    kept_heap(graph, &Dominators::new(graph, entry)?)
}

/// The loop header block of `kept`'s frame that starts at its pc, if a back edge made one.
pub(crate) fn kept_header(graph: &Graph, kept: &super::ir::KeptHeap) -> Option<BlockIx> {
    graph
        .blocks
        .iter()
        .position(|block| {
            block.loop_header
                && block.first_pc == kept.header_pc
                && block
                    .phis
                    .iter()
                    .chain(&block.body)
                    .next()
                    .is_some_and(|&first| graph.node(first).frame == kept.frame)
        })
        .map(|at| BlockIx(at as u32))
}

/// What a node may write that a loop header's kept heap facts rest on ([`kept_heap`]).
enum HeapWrite {
    Nothing,
    /// A field at this location, of whatever object.
    Field(u16),
    /// A field at this location created, its object's shape changed.
    Creation(u16),
    /// A binding at this slot, of whatever record.
    Binding(u16),
    /// Anything: it may run JavaScript.
    Any,
}

/// Each operation's [`HeapWrite`], with no arm for "the rest", so that an operation added to the
/// graph is placed here where it is added (a store a later piece lowers, a construct's).
fn heap_write(op: &NodeOp) -> HeapWrite {
    match *op {
        NodeOp::StoreField { location, .. } => HeapWrite::Field(location),
        NodeOp::StoreNew { location, .. } => HeapWrite::Creation(location),
        NodeOp::StoreBinding { slot } => HeapWrite::Binding(slot),
        NodeOp::Effect { .. } | NodeOp::Generic { .. } => HeapWrite::Any,
        // Element and typed stores write element words, which no kept fact is (a typed array's
        // data address and length stay); StoreSlot writes the frame, the frame-callee table's
        // nodes its entries; F64Mod calls fmod, which writes nothing; a poll's slow path runs no
        // JavaScript, and the collector moves nothing; an exit leaves the code.
        NodeOp::StoreElement { .. }
        | NodeOp::StoreTyped { .. }
        | NodeOp::TypedElement { .. }
        | NodeOp::StoreSlot { .. }
        | NodeOp::StoreFrameCallee { .. }
        | NodeOp::ClearFrameCallee { .. }
        | NodeOp::F64Mod
        | NodeOp::Poll
        | NodeOp::Exit { .. }
        // Constants, reads, checks, arithmetic and control flow write nothing.
        | NodeOp::ConstTagged { .. }
        | NodeOp::ConstF64 { .. }
        | NodeOp::ConstI32 { .. }
        | NodeOp::ConstBool { .. }
        | NodeOp::EntrySlot { .. }
        | NodeOp::FrameSlot { .. }
        | NodeOp::This
        | NodeOp::Phi
        | NodeOp::Projection { .. }
        | NodeOp::SlotResult { .. }
        | NodeOp::CheckNumber
        | NodeOp::CheckInt32
        | NodeOp::TruncateNumberOrOddball
        | NodeOp::CheckObject
        | NodeOp::CheckShape { .. }
        | NodeOp::CheckProto { .. }
        | NodeOp::CheckStringProto { .. }
        | NodeOp::CheckCreatable { .. }
        | NodeOp::CheckNoProto { .. }
        | NodeOp::CheckCtor
        | NodeOp::HasInPrototypeChain
        | NodeOp::CheckBoolean
        | NodeOp::ShapeIs { .. }
        | NodeOp::CheckCallee { .. }
        | NodeOp::CalleeMatches { .. }
        | NodeOp::I32ToF64
        | NodeOp::TruncateToInt32
        | NodeOp::F64ToTagged
        | NodeOp::I32ToTagged
        | NodeOp::BoolToTagged
        | NodeOp::ToBoolean
        | NodeOp::F64ToBoolean
        | NodeOp::I32Add
        | NodeOp::I32Sub
        | NodeOp::I32Mul
        | NodeOp::I32And
        | NodeOp::I32Or
        | NodeOp::I32Xor
        | NodeOp::I32Shl
        | NodeOp::I32Sar
        | NodeOp::I32UShr
        | NodeOp::I32UShrToF64
        | NodeOp::I32Not
        | NodeOp::F64Add
        | NodeOp::F64Sub
        | NodeOp::F64Mul
        | NodeOp::F64Div
        | NodeOp::F64Neg
        | NodeOp::I32Cmp { .. }
        | NodeOp::F64Cmp { .. }
        | NodeOp::TaggedSame
        | NodeOp::BoolNot
        | NodeOp::NullOrLikeUndefined
        | NodeOp::TypeofIs { .. }
        | NodeOp::CheckString
        | NodeOp::CheckObjectOrNullish
        | NodeOp::ObjectOrNullishEqual
        | NodeOp::StringEqual
        | NodeOp::WordAsF64
        | NodeOp::LoadField { .. }
        | NodeOp::CheckElements { .. }
        | NodeOp::LoadElements
        | NodeOp::LoadElementsUsed
        | NodeOp::CheckIndex
        | NodeOp::CheckBounds
        | NodeOp::LoadElement { .. }
        | NodeOp::LoadElementOrUndefined
        | NodeOp::StringLength
        | NodeOp::LoadArrayLength
        | NodeOp::TypedView
        | NodeOp::TypedLength
        | NodeOp::TypedData { .. }
        | NodeOp::LoadTyped { .. }
        | NodeOp::LoadTypedOrUndefined { .. }
        | NodeOp::ClampToUint8
        | NodeOp::IsTypedArray
        | NodeOp::Env
        | NodeOp::EnvParent
        | NodeOp::ConstEnv { .. }
        | NodeOp::FrameIndex { .. }
        | NodeOp::CheckFrameEnv { .. }
        | NodeOp::LoadBinding { .. }
        | NodeOp::CheckInitialized
        | NodeOp::GlobalObject
        | NodeOp::CheckEnvIs { .. }
        | NodeOp::CheckCellIs { .. }
        | NodeOp::CheckGlobalEpoch { .. }
        | NodeOp::CheckEnvGeneration { .. }
        | NodeOp::LoadGlobalLexical { .. }
        | NodeOp::CheckCodeValid
        | NodeOp::FoldedGlobal { .. }
        | NodeOp::FoldedCell
        | NodeOp::Jump { .. }
        | NodeOp::Branch { .. }
        | NodeOp::Return => HeapWrite::Nothing,
    }
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

/// The word a node proves an object's where it passes: a CheckObject's or a CheckShape's (each
/// tests the object tag, or takes a word proven so), as the builder records it (a callee guard
/// records nothing).
fn proved_object(node: &Node) -> Option<NodeIx> {
    match node.op {
        NodeOp::CheckObject | NodeOp::CheckShape { .. } => node.inputs.first().copied(),
        _ => None,
    }
}

/// The word a node takes as proven an object's without testing its tag: an `object` CheckShape's,
/// a `this_object` guard's receiver.
fn omitted_test(node: &Node) -> Option<NodeIx> {
    match node.op {
        NodeOp::CheckShape { object: true, .. } => node.inputs.first().copied(),
        NodeOp::CheckCallee {
            this_object: true, ..
        }
        | NodeOp::CalleeMatches {
            this_object: true, ..
        } => node.inputs.get(1).copied(),
        _ => None,
    }
}

/// The proven-object rule (P3 stage 3, c-tag): a CheckShape that tests no tag (`object`), or a
/// callee guard that does not test its receiver (`this_object`), takes a word that a check proved
/// an object's on every path to it (`proved_object`), earlier in its
/// block or on every path into the block, the builder's `Forms::object`: a must-dataflow over the
/// blocks in reverse postorder, a loop's back edge included, each block's facts at its entry the
/// intersection of its predecessors' at their ends (none at the entry block).
fn proven_objects(graph: &Graph, dom: &Dominators) -> Result<(), String> {
    if !graph.nodes.iter().any(|node| omitted_test(node).is_some()) {
        return Ok(());
    }
    // The words some check proves, numbered, and each block's proofs as bits of them.
    let mut number = crate::fasthash::FastMap::<NodeIx, usize>::default();
    for node in &graph.nodes {
        if let Some(word) = proved_object(node) {
            let next = number.len();
            number.entry(word).or_insert(next);
        }
    }
    let width = number.len().div_ceil(64);
    let bit = |word: NodeIx| number.get(&word).map(|&k| (k / 64, 1u64 << (k % 64)));
    let gen: Vec<Vec<u64>> = graph
        .blocks
        .iter()
        .map(|block| {
            let mut bits = vec![0u64; width];
            for &n in &block.body {
                if let Some((w, b)) = proved_object(graph.node(n)).and_then(bit) {
                    bits[w] |= b;
                }
            }
            bits
        })
        .collect();
    let entry = graph.entry.expect("checked");
    let mut rpo: Vec<BlockIx> = (0..graph.blocks.len() as u32).map(BlockIx).collect();
    rpo.sort_by_key(|b| dom.order[b.index()]);
    // At each block's end: optimistic (every word) before the pass reaches it.
    let mut out: Vec<Vec<u64>> = vec![vec![u64::MAX; width]; graph.blocks.len()];
    let in_of = |b: BlockIx, out: &[Vec<u64>]| -> Vec<u64> {
        if b == entry {
            return vec![0; width];
        }
        let mut bits = vec![u64::MAX; width];
        for pred in &graph.block(b).preds {
            for (x, y) in bits.iter_mut().zip(&out[pred.index()]) {
                *x &= y;
            }
        }
        bits
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &rpo {
            let mut bits = in_of(b, &out);
            for (x, y) in bits.iter_mut().zip(&gen[b.index()]) {
                *x |= y;
            }
            if bits != out[b.index()] {
                out[b.index()] = bits;
                changed = true;
            }
        }
    }
    for &b in &rpo {
        let mut bits = in_of(b, &out);
        for &n in &graph.block(b).body {
            let node = graph.node(n);
            if let Some(word) = omitted_test(node) {
                let proven = bit(word).is_some_and(|(w, b)| bits[w] & b != 0);
                if !proven {
                    return Err(format!(
                        "node {} ({:?} at pc {}): a check without its tag test on a word no check \
                         proved an object's on every path",
                        n.index(),
                        node.op,
                        node.pc
                    ));
                }
            }
            if let Some((w, b)) = proved_object(node).and_then(bit) {
                bits[w] |= b;
            }
        }
    }
    Ok(())
}

/// Whether a node is a loop's entry speculation: its reason says so, or it is a WrongName check
/// built for the first pc of its block's one successor, in that block's frame, from outside that
/// successor's loop: a loop's global read's own check made at its entry (F4 part 1,
/// `Builder::loop_entry_checks`, which builds the entry's nodes at the header's pc). A free name's
/// check made at its read is built for its read's pc, inside its own block.
fn speculative(graph: &Graph, node: &Node, dom: &Dominators) -> bool {
    match node.reason {
        Some(ExitReason::PhiNotNumber | ExitReason::PhiNotInt32 | ExitReason::HoistingFailed) => {
            true
        }
        // A loop's global read at its entry (F4 part 1), its validity test among its checks
        // where it folds (part 2).
        Some(ExitReason::WrongName | ExitReason::Invalidated) => {
            let [next] = graph.block(node.block).succs[..] else {
                return false;
            };
            let target = graph.block(next);
            let frame = target
                .phis
                .iter()
                .chain(&target.body)
                .next()
                .map(|&first| graph.node(first).frame);
            node.pc == target.first_pc
                && frame == Some(node.frame)
                && !dom.dominates(next, node.block)
        }
        _ => false,
    }
}

/// A loop's entry speculation `node`, at `position` among its block's phis and body: its check
/// (CheckNumber for PhiNotNumber, CheckInt32 for PhiNotInt32, any of CheckNumber, CheckInt32,
/// CheckObject, CheckElements, TypedData and CheckShape for HoistingFailed, and a loop's global read's own
/// CheckEnvIs, CheckCellIs, CheckGlobalEpoch or CheckEnvGeneration for WrongName) at the end of a
/// loop header's entry predecessor, after which nothing but other speculations exits or has an effect before the jump
/// to the header, with a state at the header's pc and the header's handlers; or a HoistingFailed
/// check before the head of a loop of its frame's bytecode (`cfg`, the frame's) that got no back
/// edge in the graph, its one predecessor's, with a state at its pc and no poll to compare the
/// handlers with (see the module's doc).
fn speculation(
    graph: &Graph,
    node: &Node,
    dom: &Dominators,
    position: usize,
    cfg: &Cfg,
) -> Result<(), String> {
    let check = match node.reason {
        Some(ExitReason::PhiNotNumber) => node.op == NodeOp::CheckNumber,
        Some(ExitReason::PhiNotInt32) => node.op == NodeOp::CheckInt32,
        // A loop's global, read at its entry (F4 part 1): the read's own checks.
        Some(ExitReason::WrongName) => matches!(
            node.op,
            NodeOp::CheckEnvIs { .. }
                | NodeOp::CheckCellIs { .. }
                | NodeOp::CheckGlobalEpoch { .. }
                | NodeOp::CheckEnvGeneration { .. }
        ),
        Some(ExitReason::Invalidated) => node.op == NodeOp::CheckCodeValid,
        _ => matches!(
            node.op,
            NodeOp::CheckNumber
                | NodeOp::CheckInt32
                | NodeOp::CheckObject
                | NodeOp::CheckElements(_)
                | NodeOp::TypedData { .. }
                | NodeOp::CheckShape { .. }
        ),
    };
    if !check {
        return Err(format!("a phi's speculation by {:?}", node.op));
    }
    let block = graph.block(node.block);
    let [header] = block.succs[..] else {
        return Err("a phi's speculation on a block of other than one successor".into());
    };
    let target = graph.block(header);
    if dom.dominates(header, node.block) {
        return Err("a phi's speculation off a loop's entry edge".into());
    }
    if !target.loop_header {
        // A loop of the bytecode whose back edge the build never reached (see the module's doc):
        // only a HoistingFailed check or a global read's own, before the loop's head, its one
        // entry.
        if !matches!(
            node.reason,
            Some(ExitReason::HoistingFailed | ExitReason::WrongName | ExitReason::Invalidated)
        ) || target.preds[..] != [node.block]
        {
            return Err("a phi's speculation off a loop's entry edge".into());
        }
        if cfg.loop_at_header(target.first_pc as usize).is_none() {
            return Err("an entry check before a block that heads no loop of its bytecode".into());
        }
    }
    for &later in &block.body[position - block.phis.len() + 1..] {
        let later = graph.node(later);
        if later.op.is_terminator() {
            break;
        }
        if !speculative(graph, later, dom) && (later.op.can_exit() || later.op.is_effectful()) {
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
    if target.loop_header {
        let poll = graph
            .node(target.body[0])
            .state
            .map(|poll| &graph.state(poll).handlers);
        if poll != Some(&state.handlers) {
            return Err("a phi's speculation with other handlers than its header's".into());
        }
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
        NodeOp::TruncateNumberOrOddball => (TAGGED, Rep::I32),
        NodeOp::CheckObject | NodeOp::CheckShape { .. } => (TAGGED, Rep::Cell),
        NodeOp::CheckProto { .. } => (CELL, Rep::Cell),
        NodeOp::CheckStringProto { .. } => (TAGGED, Rep::Cell),
        NodeOp::CheckCreatable { .. } | NodeOp::CheckNoProto { .. } => (CELL, Rep::None),
        NodeOp::CheckCtor => (CELL, Rep::None),
        NodeOp::HasInPrototypeChain => (&[Rep::Tagged, Rep::Cell], Rep::Bool),
        NodeOp::ShapeIs(_) => (CELL, Rep::Bool),
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
        NodeOp::NullOrLikeUndefined | NodeOp::TypeofIs(_) => (TAGGED, Rep::Bool),
        NodeOp::WordAsF64 => (TAGGED, Rep::F64),
        NodeOp::CheckString | NodeOp::CheckObjectOrNullish => (TAGGED, Rep::None),
        NodeOp::StringEqual | NodeOp::ObjectOrNullishEqual => {
            (&[Rep::Tagged, Rep::Tagged], Rep::Bool)
        }
        NodeOp::LoadField { .. } => (CELL, Rep::Tagged),
        NodeOp::StoreField { .. } | NodeOp::StoreNew { .. } => {
            (&[Rep::Cell, Rep::Tagged], Rep::None)
        }
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
        NodeOp::StringLength => (TAGGED, Rep::I32),
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
        NodeOp::FrameIndex { .. } => (&[], Rep::I32),
        NodeOp::StoreFrameCallee { .. } => (&[Rep::I32, Rep::Tagged], Rep::None),
        NodeOp::ClearFrameCallee { .. } => (&[Rep::I32], Rep::None),
        NodeOp::CheckFrameEnv { .. } => (&[Rep::Tagged, Rep::Env], Rep::None),
        NodeOp::LoadBinding { .. } => (&[Rep::Env], Rep::Tagged),
        NodeOp::CheckInitialized => (TAGGED, Rep::None),
        NodeOp::StoreBinding { .. } => (&[Rep::Env, Rep::Tagged], Rep::None),
        NodeOp::GlobalObject => (&[], Rep::Cell),
        NodeOp::CheckEnvIs { .. } => (&[Rep::Env], Rep::None),
        NodeOp::CheckCellIs { .. } | NodeOp::CheckGlobalEpoch { .. } => (CELL, Rep::None),
        NodeOp::CheckEnvGeneration { .. } => (&[Rep::Env], Rep::None),
        NodeOp::LoadGlobalLexical { .. } => (&[Rep::Env], Rep::Tagged),
        NodeOp::CheckCodeValid => (&[], Rep::None),
        NodeOp::FoldedGlobal { .. } => (&[], Rep::Tagged),
        NodeOp::FoldedCell => (TAGGED, Rep::Cell),
        NodeOp::Poll | NodeOp::Jump(_) | NodeOp::Exit(_) => (&[], Rep::None),
        NodeOp::Branch { .. } => (BOOL, Rep::None),
        NodeOp::Return => (TAGGED, Rep::None),
        NodeOp::Phi
        | NodeOp::Projection(_)
        | NodeOp::SlotResult(_)
        | NodeOp::Effect { .. }
        | NodeOp::Generic { .. }
        | NodeOp::CheckCallee { .. }
        | NodeOp::CalleeMatches { .. } => return None,
    })
}

/// `node`'s representations, and the operands its operation names.
fn representations(
    graph: &Graph,
    node: &Node,
    ix: NodeIx,
    chunk: &FrozenChunk,
    compiled: &FrozenChunk,
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
                NodeOp::Effect { pushes, .. } | NodeOp::Generic { pushes, .. }
                    if result < pushes => {}
                _ => return Err(format!("a projection of {:?}", effect.op)),
            }
            if node.rep != Rep::Tagged {
                return Err(format!("a {:?} projection", node.rep));
            }
        }
        &NodeOp::Effect { pc, pops, pushes } | &NodeOp::Generic { pc, pops, pushes } => {
            if pc != node.pc {
                return Err(format!("an effect of pc {pc} built at another pc"));
            }
            if chunk.jit_stack_effect(pc as usize) != Some((pops as usize, pushes as usize)) {
                return Err(format!(
                    "pops {pops} and pushes {pushes}, the operation {:?}",
                    chunk.jit_stack_effect(pc as usize)
                ));
            }
            // A generic node's slot inputs (P3 stage 3 step 3a): the slots its operation names
            // that its state holds as values, whose stale words it writes.
            let stale = match (&node.op, node.state) {
                (NodeOp::Generic { .. }, Some(state)) => {
                    let state = graph.state(state);
                    super::liveness::frame_slots(&chunk.jit_ops()[pc as usize], n_slots)
                        .iter()
                        .filter(|&&(slot, _)| {
                            matches!(
                                state.slots.get(usize::from(slot)),
                                Some(SlotSource::Value(_))
                            )
                        })
                        .count()
                }
                _ => 0,
            };
            // Each operand in a representation the code boxes into its word (stage 2 step 6:
            // an untagged phi's uses need no retagging).
            if node.inputs.len() != pops as usize + stale || !reps.iter().all(|&rep| boxable(rep)) {
                return Err(format!(
                    "operands {reps:?} for {pops} words and {stale} stale slots"
                ));
            }
            if node.rep != Rep::None {
                return Err(format!("a {:?} effect", node.rep));
            }
        }
        &NodeOp::SlotResult(slot) => {
            // A slot its generic node's operation writes, read in the node's block after it.
            let generic = match node.inputs[..] {
                [generic] => graph.node(generic),
                _ => return Err("a slot's result of other than one node".into()),
            };
            let NodeOp::Generic { pc, .. } = generic.op else {
                return Err(format!("a slot's result of {:?}", generic.op));
            };
            let writes = super::liveness::frame_slots(&chunk.jit_ops()[pc as usize], n_slots)
                .contains(&(slot, true));
            if !writes || generic.block != node.block || node.rep != Rep::Tagged {
                return Err(format!(
                    "a {:?} result of slot {slot}, which pc {pc}'s operation does not write \
                     or in another block",
                    node.rep
                ));
            }
        }
        &NodeOp::CheckCallee {
            target,
            this_object,
        }
        | &NodeOp::CalleeMatches {
            target,
            this_object,
        } => {
            // A splice's guard is at its InlineGuard; a graph-inlined call's (P3 stage 3, step 2)
            // at its call, of its target's argument count, a plain call's checking no `this`.
            // Its inputs are the callee's word and, for a guard of `this`, the receiver's.
            let Some(it) = compiled.inline_targets.get(target as usize) else {
                return Err(format!(
                    "the guard of target {target}, which the chunk has not"
                ));
            };
            match chunk.jit_ops().get(node.pc as usize) {
                Some(&Op::InlineGuard(guard, _)) if guard == target => {}
                Some(&Op::InlineGuard(guard, _)) => {
                    return Err(format!("the InlineGuard of target {guard}, not {target}"));
                }
                Some(&Op::Call(argc, _)) if argc == it.argc && !it.check_this => {}
                Some(&Op::CallWithThis(argc, _)) if argc == it.argc => {}
                op => return Err(format!("a guard of target {target} at {op:?} ({ix:?})")),
            }
            let inputs = 1 + usize::from(it.check_this);
            if reps.len() != inputs || reps.iter().any(|&rep| rep != Rep::Tagged) {
                return Err(format!("operands {reps:?} for {inputs} words"));
            }
            if this_object && !it.check_this {
                return Err("a guard that omits a receiver test it has not".into());
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
        NodeOp::CheckShape {
            ref shapes, ways, ..
        } if shapes.is_empty() || shapes.len() > usize::from(ways) || ways > 4 => {
            Err(format!("{} shapes of {ways} ways", shapes.len()))
        }
        NodeOp::ShapeIs(ref shapes) if shapes.is_empty() || shapes.len() > 4 => {
            Err(format!("{} shapes", shapes.len()))
        }
        NodeOp::CheckProto { ref shapes, ways } | NodeOp::CheckStringProto { ref shapes, ways }
            if shapes.is_empty() || shapes.len() > 4 || ways == 0 || ways > 4 =>
        {
            Err(format!(
                "a prototype check of {} shapes, {ways} ways",
                shapes.len()
            ))
        }
        NodeOp::CheckCreatable { ways } | NodeOp::CheckNoProto { ways }
            if ways == 0 || ways > 4 =>
        {
            Err(format!("a creation's check of {ways} ways"))
        }
        NodeOp::StoreNew { location, ways, .. } if location == 0 || ways == 0 || ways > 4 => {
            Err(format!("a creation at location {location}, {ways} ways"))
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

/// A frame state against its pc in its frame's chunk: the slot count, the stack analysis's depth
/// (one less for an inlined frame's `parent` state, whose call's result the exit pushes), the
/// handler records the chunk pushes at their depths, values an exit can box, Dead slots dead at
/// the pc where `liveness` is given, and for an inlined frame's state values or Dead slots only
/// and no handler record.
fn frame_state(
    graph: &Graph,
    frame: &FrameState,
    (chunk, cfg): FrameChunk<'_>,
    parent: bool,
    liveness: Option<&SlotLiveness>,
) -> Result<(), String> {
    let ops = chunk.jit_ops();
    let (_, n_slots) = chunk.jit_frame();
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
    let depth = cfg
        .stack_depth_at(pc)
        .and_then(|depth| depth.checked_sub(usize::from(parent)));
    if depth != Some(frame.stack.len()) {
        return Err(format!(
            "{} stack values where the depth is {depth:?}",
            frame.stack.len(),
        ));
    }
    if !frame.frame.is_compiled() {
        if frame.slots.contains(&SlotSource::Frame) {
            return Err("an inlined frame's slot read from a frame it has not".into());
        }
        if !frame.handlers.is_empty() {
            return Err("a handler record in an inlined frame".into());
        }
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
