//! The tier's graph builder (P3 stage 1; the plan's Design points 3 to 5 and 9 and its Stage 1
//! record): one forward pass in reverse postorder over the chunk's blocks (jit_ir's Cfg) that
//! interprets the canonical frame, its slots and operand stack, over SSA values, as
//! MaglevGraphBuilder interprets Ignition's registers and WarpBuilder the bytecode stack. It
//! reads only the chunk's operations and the snapshot.
//!
//! - Control flow: the blocks the entry reaches by normal edges. Handler pads and resumptions
//!   are other roots of the Cfg, which the graph never enters: a throw in optimized code exits to
//!   the VM at the throwing pc, where the VM runs the pad (the plan's point 11).
//! - Loops: a loop-assignment prepass gives each loop header a phi for every slot a native write
//!   in the loop assigns and for every operand-stack position. A slot the loop does not assign
//!   keeps its value through the header when nothing in the loop can be an effect, and is in the
//!   frame at the header otherwise (an effect leaves every slot in the frame). A header's poll
//!   opens its body, with the header's state, once a back edge reaches it.
//! - Joins: a phi where the predecessors' values differ and one of them is not in the frame.
//!   Values cross block boundaries as tagged words in stage 1 (untagging phis is stage 4's).
//! - Known facts per value (Maglev's KnownNodeAspects): its other representations (a word's
//!   Number as an f64, its exact int32, its Boolean, its object's cell, an unboxed value's word),
//!   which are properties of the immutable value; and the heap facts of the current effect epoch
//!   (an object's possible shapes, the fields loaded or stored), which an effect ends. Facts meet
//!   at joins, and a loop header keeps no heap fact. No check or load a fact proves is emitted.
//! - Feedback: a block whose named and call sites never ran is an exit at its first pc (Maglev,
//!   Warp and the DFG); arithmetic and comparisons take int32 or f64 from rule U's value classes;
//!   named loads and stores on own data ways check the shape and load or store the slot.
//! - Everything else that is not control flow is an effect: the operation's baseline template on
//!   the canonical frame, with the dirty slots flushed before it and every slot read back from the
//!   frame after it.
//! - Exits: a check before an operation exits at the operation's pc with its operands on the
//!   stack, and so does an operation's own exit (an int32 overflow); nothing exits after an
//!   operation has had an effect, so an exit never replays one (the regions' rule,
//!   jit_regions.rs:1-5). An effect's state is the frame state before it.
//! - Handlers: PushHandler, PushIterator and PopHandler emit nothing; the records they push are
//!   the states' handler lists. A return inside an iterator region exits to the VM, which routes
//!   it through the iterator's close.
//! - The graph's blocks are numbered in reverse postorder (the entry first, every block after its
//!   dominators), constants and `this` head the entry block, and values from the frame are read
//!   where a block first needs them.

use super::ir::{
    BlockData, BlockIx, Cond, ExitReason, FrameState, Graph, NodeIx, NodeOp, Refusal, Rep,
    SlotSource, StateIx,
};
use super::snapshot::{Ran, Snapshot};
use crate::bytecode::{Chunk, Op, UpdKind};
use crate::fasthash::FastMap;
use crate::feedback::{NamedWayKind, ValueClass, ValuesView};
use crate::jit_ir::{BlockId, BuildError, Cfg, InitializedReads};
use crate::value::{PACK_BOOL, PACK_CANON_NAN, PACK_EMPTY, PACK_NULL, PACK_UNDEFINED};

const NUMBER: u32 = ValueClass::NumberInt32.bit() | ValueClass::NumberDouble.bit();
const INT32: u32 = ValueClass::NumberInt32.bit();
/// The most operations a chunk may have: compile_entry's limit (an operation's index fits one
/// movz).
const MAX_OPS: usize = 0xffff;
/// A Cfg block the entry does not reach.
const UNREACHED: u32 = u32::MAX;
/// The slots a native field access reaches (the ARM64 back end admits fields below slot 4,096;
/// a location word that fits a cache's 15 bits names a lower slot anyway).
const MAX_FIELD_SLOT: u32 = 4096;

/// The graph of `chunk` from its operations and `snapshot` (its feedback), or why the tier
/// declines it.
pub(crate) fn build(chunk: &Chunk, snapshot: &Snapshot) -> Result<Graph, Refusal> {
    let ops = chunk.jit_ops();
    if chunk.jit_is_resumable() {
        return Err(Refusal::whole("resumable (a generator or an async body)"));
    }
    if chunk.prepared_entry {
        return Err(Refusal::whole(
            "prepared entry (mapped arguments, eval, or closures that need them)",
        ));
    }
    if ops.len() > MAX_OPS {
        return Err(Refusal::whole("over 65,535 operations"));
    }
    if chunk.jit_detailed_feedback_enabled() {
        // The diagnostic observes every operation through the templates; native lowerings
        // would record none of it (the regions decline it too, jit_regions.rs:634-636).
        return Err(Refusal::whole("the detailed feedback vector is on"));
    }
    if snapshot.len() != ops.len() {
        return Err(Refusal::whole("a snapshot of another chunk"));
    }
    if let Some((pc, reason)) = ops
        .iter()
        .enumerate()
        .find_map(|(pc, op)| refused(op).map(|reason| (pc, reason)))
    {
        return Err(Refusal::at(pc, reason));
    }
    let cfg = Cfg::build(chunk).map_err(stack_refusal)?;
    let (_, n_slots) = chunk.jit_frame();
    if n_slots + cfg.jit_stack_capacity() > crate::jit::FRAME_BUF {
        return Err(Refusal::whole("a frame over the record's 256 words"));
    }
    let initialized = cfg.initialized_reads(ops, true);
    let flow = Flow::new(chunk, &cfg, &initialized)?;
    Builder::new(chunk, snapshot, &cfg, &flow, &initialized).run()
}

/// Why stage 1 refuses a chunk holding `op` (the Stage 1 record's list).
fn refused(op: &Op) -> Option<&'static str> {
    Some(match op {
        Op::FragmentExit(_) => "a loop fragment",
        Op::PushFinally(..) | Op::AbruptJump(..) | Op::ResumeJump | Op::ResumeReturn => {
            "finally and its completions"
        }
        Op::PushDisposeFrame
        | Op::AddDisposable(_)
        | Op::DisposeNormal
        | Op::DisposeThrow
        | Op::DisposeReturn
        | Op::DisposeBareReturn
        | Op::DisposeResumeReturn
        | Op::DisposeJump => "resource disposal",
        Op::PushWith => "with",
        Op::EvalCallArgsArray | Op::TailEvalCallArgsArray => "direct eval",
        Op::SuperCallStart | Op::SuperCallArgsArray => "a derived constructor",
        Op::Await
        | Op::Yield
        | Op::YieldStar
        | Op::GetAsyncIter
        | Op::AsyncIterStepL(..)
        | Op::AsyncIterResumeL(..)
        | Op::AsyncIterCloseL(..) => "a suspension",
        _ => return None,
    })
}

fn stack_refusal(error: BuildError) -> Refusal {
    match error {
        BuildError::Empty => Refusal::whole("no operations"),
        BuildError::MissingStackEffect { pc } => {
            Refusal::at(pc, "an operation the native tiers do not compile")
        }
        BuildError::BadTarget { pc, .. }
        | BuildError::StackUnderflow { pc, .. }
        | BuildError::StackMismatch { pc, .. } => Refusal::at(pc, "the stack analysis fails"),
    }
}

fn ix(id: BlockId) -> usize {
    id.0 as usize
}

/// The constant word of chunk constant `k` (a scalar's, or a String's, which the chunk's pool
/// roots while code it owns can run), None for a constant generated code does not embed.
fn const_word(chunk: &Chunk, k: u32) -> Option<u64> {
    if chunk.jit_const_copyable(k) {
        chunk.jit_const_scalar_bits(k)
    } else {
        chunk.jit_const_str_word(k)
    }
}

/// Whether the operation at `pc` always lowers without an effect (the loop prepass's question).
fn never_effect(chunk: &Chunk, initialized: &InitializedReads, pc: usize) -> bool {
    match chunk.jit_ops()[pc] {
        Op::Const(k) => const_word(chunk, k).is_some(),
        Op::LoadLocal(_) => initialized.contains(pc),
        Op::Undef
        | Op::Dup
        | Op::Dup2
        | Op::Pop
        | Op::StoreLocal(_)
        | Op::Tdz(_)
        | Op::ResetSlots(..)
        | Op::LoadThis
        | Op::Void
        | Op::Not
        | Op::Jump(_)
        | Op::JumpIfFalse(_)
        | Op::JumpIfFalsePeek(_)
        | Op::JumpIfTruePeek(_)
        | Op::JumpIfNotNullishPeek(_)
        | Op::InlineGuard(..)
        | Op::PushHandler(_)
        | Op::PushIterator(..)
        | Op::PopHandler
        | Op::Return
        | Op::ReturnBare
        | Op::ReturnUndef
        | Op::Throw
        | Op::TailCall(..)
        | Op::IterAbortL(_)
        | Op::IterAbortIfNotDoneL(..) => true,
        _ => false,
    }
}

/// The f64 a tagged word holds, None for a word that is not a Number (value.rs: a Number is its
/// f64 bits, every other kind a tag above the canonical NaN).
fn number_word(word: u64) -> Option<f64> {
    (((word >> 48) as u16 & 0x7fff) < 0x7ff9).then(|| f64::from_bits(word))
}

/// `x` as an int32, when it is one exactly (not -0).
fn exact_i32(x: f64) -> Option<i32> {
    (x.fract() == 0.0
        && x >= f64::from(i32::MIN)
        && x <= f64::from(i32::MAX)
        && !(x == 0.0 && x.is_sign_negative()))
    .then_some(x as i32)
}

/// ToBoolean of a constant word, None where it takes the value's contents (a String).
fn const_truth(word: u64) -> Option<bool> {
    if word == PACK_UNDEFINED || word == PACK_NULL {
        return Some(false);
    }
    if word & !1 == PACK_BOOL {
        return Some(word & 1 != 0);
    }
    number_word(word).map(|x| x != 0.0 && !x.is_nan())
}

/// Whether `word === other` is `word`'s identity with `other`'s for every `other`: undefined,
/// null, a Boolean and a Number other than 0, -0 and NaN, each a word no other value has.
fn identity_constant(word: u64) -> bool {
    word == PACK_UNDEFINED
        || word == PACK_NULL
        || word & !1 == PACK_BOOL
        || number_word(word).is_some_and(|x| x != 0.0 && !x.is_nan())
}

/// Whether the record alone (a VM run's or a miss's) saw int32s only, in every role it holds:
/// rule U's int32/double split, a hint the int32 path's exits guard.
fn int32_hint(values: &ValuesView) -> bool {
    values.has_record() && values.recorded.iter().all(|&role| role & !INT32 == 0)
}

/// The function's control flow from its entry over the Cfg's blocks: the blocks the entry reaches
/// by normal edges, in reverse postorder, their dominators and natural loops.
struct Flow {
    rpo: Vec<BlockId>,
    /// Each Cfg block's index in `rpo` ([`UNREACHED`] for a block the entry does not reach).
    order: Vec<u32>,
    /// The loop each Cfg block heads, an index into `loops`.
    heads: Vec<Option<usize>>,
    loops: Vec<Loop>,
}

/// A natural loop's prepass facts.
struct Loop {
    /// The slots a native write in the loop assigns (StoreLocal, UpdateLocal, Tdz, ResetSlots).
    written: Vec<bool>,
    /// Whether an operation of the loop may lower to an effect, after which every slot is in the
    /// frame.
    effects: bool,
}

impl Flow {
    fn new(chunk: &Chunk, cfg: &Cfg, initialized: &InitializedReads) -> Result<Flow, Refusal> {
        let blocks = cfg.blocks();
        let ops = chunk.jit_ops();
        let (_, n_slots) = chunk.jit_frame();
        let entry = cfg
            .block_at(0)
            .ok_or_else(|| Refusal::whole("no entry block"))?;
        // Reverse postorder from the entry alone, successors in their order.
        let mut seen = vec![false; blocks.len()];
        let mut post = Vec::with_capacity(blocks.len());
        let mut work = vec![(entry, 0usize)];
        seen[ix(entry)] = true;
        while let Some(top) = work.last_mut() {
            let (id, next) = *top;
            if let Some(&succ) = blocks[ix(id)].successors.get(next) {
                top.1 += 1;
                if blocks[ix(succ)].stack_in.is_some() && !seen[ix(succ)] {
                    seen[ix(succ)] = true;
                    work.push((succ, 0));
                }
            } else {
                post.push(id);
                work.pop();
            }
        }
        post.reverse();
        let rpo = post;
        let mut order = vec![UNREACHED; blocks.len()];
        for (index, id) in rpo.iter().enumerate() {
            order[ix(*id)] = index as u32;
        }
        // Immediate dominators by rpo index (Cooper, Harvey and Kennedy's iteration), over the
        // predecessors the entry reaches.
        let mut idom = vec![UNREACHED; rpo.len()];
        idom[0] = 0;
        let mut changed = true;
        while changed {
            changed = false;
            for index in 1..rpo.len() {
                let mut new = UNREACHED;
                for pred in &blocks[ix(rpo[index])].predecessors {
                    let p = order[ix(*pred)];
                    if p == UNREACHED || idom[p as usize] == UNREACHED {
                        continue;
                    }
                    new = if new == UNREACHED {
                        p
                    } else {
                        intersect(&idom, p, new)
                    };
                }
                if new != idom[index] {
                    idom[index] = new;
                    changed = true;
                }
            }
        }
        // Back edges (to a block that dominates their source); any other retreating edge makes
        // the flow irreducible.
        let mut latches: Vec<Vec<usize>> = vec![Vec::new(); rpo.len()];
        for (index, id) in rpo.iter().enumerate() {
            for succ in &blocks[ix(*id)].successors {
                let s = order[ix(*succ)];
                if s == UNREACHED || s as usize > index {
                    continue;
                }
                if !dominates(&idom, s, index as u32) {
                    return Err(Refusal::at(
                        blocks[ix(*succ)].start,
                        "irreducible control flow",
                    ));
                }
                latches[s as usize].push(index);
            }
        }
        let mut heads = vec![None; blocks.len()];
        let mut loops = Vec::new();
        let mut member = vec![false; rpo.len()];
        for (header, latches) in latches.iter().enumerate() {
            if latches.is_empty() {
                continue;
            }
            member.fill(false);
            member[header] = true;
            let mut body = vec![header];
            let mut work = latches.clone();
            while let Some(index) = work.pop() {
                if std::mem::replace(&mut member[index], true) {
                    continue;
                }
                body.push(index);
                for pred in &blocks[ix(rpo[index])].predecessors {
                    let p = order[ix(*pred)];
                    if p != UNREACHED && !member[p as usize] {
                        work.push(p as usize);
                    }
                }
            }
            let mut found = Loop {
                written: vec![false; n_slots],
                effects: false,
            };
            for index in body {
                let block = &blocks[ix(rpo[index])];
                for pc in block.start..block.end {
                    let writes = match ops[pc] {
                        Op::StoreLocal(slot) | Op::UpdateLocal(slot, _) | Op::Tdz(slot) => {
                            slot as usize..slot as usize + 1
                        }
                        Op::ResetSlots(start, count) => {
                            start as usize..start as usize + count as usize
                        }
                        _ => 0..0,
                    };
                    for slot in writes {
                        if let Some(written) = found.written.get_mut(slot) {
                            *written = true;
                        }
                    }
                    if !never_effect(chunk, initialized, pc) {
                        found.effects = true;
                    }
                }
            }
            heads[ix(rpo[header])] = Some(loops.len());
            loops.push(found);
        }
        Ok(Flow {
            rpo,
            order,
            heads,
            loops,
        })
    }
}

/// `graph` with its blocks numbered in reverse postorder from the entry: the entry first, every
/// block after its dominators and forward edges forward, the order a one-pass back end walks.
fn in_reverse_postorder(mut graph: Graph) -> Graph {
    let entry = graph.entry.expect("the builder makes the entry");
    let count = graph.blocks.len();
    let mut seen = vec![false; count];
    let mut post = Vec::with_capacity(count);
    let mut work = vec![(entry, 0usize)];
    seen[entry.index()] = true;
    while let Some(top) = work.last_mut() {
        let (block, next) = *top;
        let succs = &graph.blocks[block.index()].succs;
        if next < succs.len() {
            // The last successor first, so that the first comes first in reverse postorder.
            let succ = succs[succs.len() - 1 - next];
            top.1 += 1;
            if !std::mem::replace(&mut seen[succ.index()], true) {
                work.push((succ, 0));
            }
        } else {
            post.push(block);
            work.pop();
        }
    }
    post.reverse();
    // The entry reaches every block the builder makes; any other would keep its place after.
    post.extend(
        (0..count)
            .filter(|&index| !seen[index])
            .map(|index| BlockIx(index as u32)),
    );
    let mut number = vec![BlockIx(0); count];
    for (position, block) in post.iter().enumerate() {
        number[block.index()] = BlockIx(position as u32);
    }
    let renumber = |block: BlockIx| number[block.index()];
    let mut old: Vec<Option<BlockData>> = graph.blocks.drain(..).map(Some).collect();
    graph.blocks = post
        .iter()
        .map(|block| old[block.index()].take().expect("each block once"))
        .collect();
    for block in &mut graph.blocks {
        for edge in block.preds.iter_mut().chain(block.succs.iter_mut()) {
            *edge = renumber(*edge);
        }
    }
    for node in &mut graph.nodes {
        node.block = renumber(node.block);
        match &mut node.op {
            NodeOp::Jump(target) => *target = renumber(*target),
            NodeOp::Branch {
                then_block,
                else_block,
            } => {
                *then_block = renumber(*then_block);
                *else_block = renumber(*else_block);
            }
            _ => {}
        }
    }
    graph.entry = Some(renumber(entry));
    graph
}

fn intersect(idom: &[u32], mut a: u32, mut b: u32) -> u32 {
    while a != b {
        while a > b {
            a = idom[a as usize];
        }
        while b > a {
            b = idom[b as usize];
        }
    }
    a
}

/// Whether rpo index `a` dominates rpo index `b`.
fn dominates(idom: &[u32], a: u32, mut b: u32) -> bool {
    while b > a {
        b = idom[b as usize];
    }
    a == b
}

/// Where a slot's value is during the pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    /// The frame holds the value the slot had at entry: nothing wrote it and no effect ran.
    Entry,
    /// The frame holds the value and no node does: an effect may have written it.
    Frame,
    /// The frame holds this node's value.
    Clean(NodeIx),
    /// This node holds the value; the frame's word is stale.
    Dirty(NodeIx),
}

/// A location of the canonical frame a phi merges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Loc {
    Slot(usize),
    Stack(usize),
}

/// The forms of one value: the nodes that hold it in each representation (a word, its Number
/// as an f64, its exact int32, its Boolean, its object's cell). They are properties of the
/// immutable value, true wherever the nodes are available.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Forms {
    tagged: Option<NodeIx>,
    f64: Option<NodeIx>,
    i32: Option<NodeIx>,
    boolean: Option<NodeIx>,
    cell: Option<NodeIx>,
}

impl Forms {
    fn or(self, other: Forms) -> Forms {
        Forms {
            tagged: self.tagged.or(other.tagged),
            f64: self.f64.or(other.f64),
            i32: self.i32.or(other.i32),
            boolean: self.boolean.or(other.boolean),
            cell: self.cell.or(other.cell),
        }
    }

    /// The forms both hold.
    fn meet(self, other: Forms) -> Forms {
        let same = |a: Option<NodeIx>, b: Option<NodeIx>| if a == b { a } else { None };
        Forms {
            tagged: same(self.tagged, other.tagged),
            f64: same(self.f64, other.f64),
            i32: same(self.i32, other.i32),
            boolean: same(self.boolean, other.boolean),
            cell: same(self.cell, other.cell),
        }
    }

    fn members(self) -> impl Iterator<Item = NodeIx> {
        [self.tagged, self.f64, self.i32, self.boolean, self.cell]
            .into_iter()
            .flatten()
    }
}

/// The abstract canonical frame at a point of the pass, with the facts known there.
#[derive(Clone, Default)]
struct State {
    slots: Vec<Slot>,
    stack: Vec<NodeIx>,
    /// The handler records active (push pc, operand depth at the push), innermost last.
    handlers: Vec<(u32, u32)>,
    /// Each value's other forms (see [`Forms`]).
    forms: FastMap<NodeIx, Forms>,
    /// ToBoolean of a value.
    truth: FastMap<NodeIx, NodeIx>,
    /// Heap facts of the current effect epoch: an object word's possible shapes...
    shapes: FastMap<NodeIx, Vec<u32>>,
    /// ...and the word at a location of an object word, loaded or stored.
    fields: FastMap<(NodeIx, u16), NodeIx>,
}

impl State {
    fn clear_heap(&mut self) {
        self.shapes.clear();
        self.fields.clear();
    }

    /// Whether a handler record of a `for…of` body is active (a return there is routed through
    /// the iterator's close).
    fn in_iterator(&self, ops: &[Op]) -> bool {
        self.handlers
            .iter()
            .any(|&(push, _)| matches!(ops[push as usize], Op::PushIterator(..)))
    }
}

/// The facts every predecessor's state holds: value facts present in all, the union of an
/// object's possible shapes where each knows some, fields loaded alike.
fn meet(preds: &[(BlockIx, State)]) -> State {
    let (first, rest) = (&preds[0].1, &preds[1..]);
    let forms = first
        .forms
        .iter()
        .filter_map(|(&value, &forms)| {
            let mut met = forms;
            for (_, state) in rest {
                met = met.meet(*state.forms.get(&value)?);
            }
            (met != Forms::default()).then_some((value, met))
        })
        .collect();
    let truth = first
        .truth
        .iter()
        .filter(|(value, truth)| {
            rest.iter()
                .all(|(_, state)| state.truth.get(value) == Some(truth))
        })
        .map(|(&value, &truth)| (value, truth))
        .collect();
    let shapes = first
        .shapes
        .iter()
        .filter_map(|(&word, shapes)| {
            let mut union = shapes.clone();
            for (_, state) in rest {
                for shape in state.shapes.get(&word)? {
                    if !union.contains(shape) {
                        union.push(*shape);
                    }
                }
            }
            Some((word, union))
        })
        .collect();
    let fields = first
        .fields
        .iter()
        .filter(|(field, value)| {
            rest.iter()
                .all(|(_, state)| state.fields.get(field) == Some(value))
        })
        .map(|(&field, &value)| (field, value))
        .collect();
    State {
        forms,
        truth,
        shapes,
        fields,
        ..State::default()
    }
}

/// A built loop header: its phis by the location each merges, the state it assumed for what the
/// phis do not merge (each back edge must agree with it), and its poll's frame state.
struct Header {
    block: BlockIx,
    phis: Vec<(NodeIx, Loc)>,
    state: State,
    /// The frame state at the header's pc, which its poll exits with; the poll opens the
    /// header's body once a back edge reaches it.
    poll: StateIx,
}

/// The receiver of a named access.
#[derive(Clone, Copy)]
enum Recv {
    /// Below the operands on the stack (the top of the stack for a load).
    Stack,
    /// The frame's `this`.
    This,
    /// A slot.
    Slot(usize),
}

/// A named site's own data ways: the receiver shapes and the one location and slot they share.
struct Ways {
    shapes: Vec<u32>,
    location: u16,
    slot: u32,
}

/// The values built once at the head of the entry block, where they dominate every use: the
/// constants, `this`, and the slots' entry values (read where the frame still holds them).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Hoisted {
    Tagged(u64),
    F64(u64),
    I32(i32),
    Bool(bool),
    This,
    Entry(u16),
}

/// The state an operation's exits share: the frame state before the operation (its operands on
/// the stack), made at the first check that needs it.
struct Before {
    pc: usize,
    state: Option<StateIx>,
}

impl Before {
    fn new(pc: usize) -> Self {
        Before { pc, state: None }
    }
}

struct Builder<'a> {
    chunk: &'a Chunk,
    ops: &'a [Op],
    snapshot: &'a Snapshot,
    cfg: &'a Cfg,
    flow: &'a Flow,
    initialized: &'a InitializedReads,
    n_slots: usize,
    graph: Graph,
    /// The block nodes go to.
    cur: BlockIx,
    /// Each Cfg block's graph block, made when the first edge reaches it.
    blocks: Vec<Option<BlockIx>>,
    /// The edges waiting at a Cfg block not built yet: their source and the state it leaves.
    incoming: Vec<Vec<(BlockIx, State)>>,
    built: Vec<bool>,
    headers: Vec<Option<Header>>,
    hoisted: FastMap<Hoisted, NodeIx>,
    /// The hoisted nodes in the order made, which head the entry block once it is built.
    hoisted_nodes: Vec<NodeIx>,
}

impl<'a> Builder<'a> {
    fn new(
        chunk: &'a Chunk,
        snapshot: &'a Snapshot,
        cfg: &'a Cfg,
        flow: &'a Flow,
        initialized: &'a InitializedReads,
    ) -> Self {
        let (_, n_slots) = chunk.jit_frame();
        let blocks = cfg.blocks().len();
        Builder {
            chunk,
            ops: chunk.jit_ops(),
            snapshot,
            cfg,
            flow,
            initialized,
            n_slots,
            graph: Graph::new(n_slots),
            cur: BlockIx(0),
            blocks: vec![None; blocks],
            incoming: (0..blocks).map(|_| Vec::new()).collect(),
            built: vec![false; blocks],
            headers: (0..blocks).map(|_| None).collect(),
            hoisted: FastMap::default(),
            hoisted_nodes: Vec::new(),
        }
    }

    fn run(mut self) -> Result<Graph, Refusal> {
        let flow = self.flow;
        let entry = flow.rpo[0];
        let initial = State {
            slots: vec![Slot::Entry; self.n_slots],
            ..State::default()
        };
        let first = self.graph.add_block(0, false);
        self.graph.entry = Some(first);
        self.cur = first;
        if flow.heads[ix(entry)].is_some() {
            // The first operation heads a loop: an empty entry block gives its phis their entry
            // inputs.
            let header = self.block_for(0)?;
            self.add(NodeOp::Jump(header), &[], Rep::None, 0);
            self.edge(first, 0, initial)?;
        } else {
            self.blocks[ix(entry)] = Some(first);
            self.incoming[ix(entry)].push((first, initial));
        }
        for &id in &flow.rpo {
            self.build_block(id)?;
        }
        let hoisted = std::mem::take(&mut self.hoisted_nodes);
        self.graph.blocks[first.index()].body.splice(0..0, hoisted);
        Ok(in_reverse_postorder(self.graph))
    }

    fn build_block(&mut self, id: BlockId) -> Result<(), Refusal> {
        let preds = std::mem::take(&mut self.incoming[ix(id)]);
        if preds.is_empty() {
            return Ok(()); // no edge of the graph reaches it
        }
        let cfg = self.cfg;
        let (start, end) = (cfg.blocks()[ix(id)].start, cfg.blocks()[ix(id)].end);
        let block = self.blocks[ix(id)].expect("an edge made the block");
        self.cur = block;
        self.built[ix(id)] = true;
        let never_ran = self.never_ran(start, end);
        let mut st = match self.flow.heads[ix(id)] {
            Some(found) if !never_ran => self.header(id, block, preds, found)?,
            _ => self.merge(block, preds, start, &mut Vec::new())?,
        };
        if never_ran {
            self.exit(start, ExitReason::NeverRan, st);
            return Ok(());
        }
        for pc in start..end {
            if cfg.stack_depth_at(pc) != Some(st.stack.len()) {
                return Err(Refusal::at(
                    pc,
                    "the stack depth disagrees with the stack analysis",
                ));
            }
            if !self.op(pc, &mut st)? {
                return Ok(());
            }
        }
        if end >= self.ops.len() {
            return Err(Refusal::at(end - 1, "control falls off the end"));
        }
        self.jump(end - 1, end, st)
    }

    /// Whether the block's named and call sites hold some and none of them ran.
    fn never_ran(&self, start: usize, end: usize) -> bool {
        let mut sites = false;
        for pc in start..end {
            match self.snapshot.ran(pc) {
                Ran::Yes => return false,
                Ran::No => sites = true,
                Ran::Unknown => {}
            }
        }
        sites
    }

    // ---- nodes and states ----------------------------------------------------------------------

    fn add(&mut self, op: NodeOp, inputs: &[NodeIx], rep: Rep, pc: usize) -> NodeIx {
        self.graph.add_node(self.cur, op, inputs, rep, pc as u32)
    }

    /// A node that can exit, with its state and reason.
    fn add_exit(
        &mut self,
        op: NodeOp,
        inputs: &[NodeIx],
        rep: Rep,
        pc: usize,
        state: StateIx,
        reason: ExitReason,
    ) -> NodeIx {
        let node = self.add(op, inputs, rep, pc);
        let data = self.graph.node_mut(node);
        data.state = Some(state);
        data.reason = Some(reason);
        node
    }

    /// A node placed before the terminator of `block` (built already).
    fn add_at_end(
        &mut self,
        block: BlockIx,
        op: NodeOp,
        inputs: &[NodeIx],
        rep: Rep,
        pc: usize,
    ) -> NodeIx {
        let node = self.graph.add_node(block, op, inputs, rep, pc as u32);
        let Graph { blocks, nodes, .. } = &mut self.graph;
        let body = &mut blocks[block.index()].body;
        body.pop();
        match body.last() {
            Some(last) if nodes[last.index()].op.is_terminator() => {
                body.insert(body.len() - 1, node)
            }
            _ => body.push(node),
        }
        node
    }

    /// A constant or `this`, made once at the head of the entry block.
    fn hoist(&mut self, key: Hoisted) -> NodeIx {
        if let Some(&node) = self.hoisted.get(&key) {
            return node;
        }
        let (op, rep) = match key {
            Hoisted::Tagged(word) => (NodeOp::ConstTagged(word), Rep::Tagged),
            Hoisted::F64(bits) => (NodeOp::ConstF64(bits), Rep::F64),
            Hoisted::I32(value) => (NodeOp::ConstI32(value), Rep::I32),
            Hoisted::Bool(value) => (NodeOp::ConstBool(value), Rep::Bool),
            Hoisted::This => (NodeOp::This, Rep::Tagged),
            Hoisted::Entry(slot) => (NodeOp::EntrySlot(slot), Rep::Tagged),
        };
        let entry = self.graph.entry.expect("the entry block is made first");
        let node = self.graph.add_node(entry, op, &[], rep, 0);
        // Set aside until the build ends, then placed at the entry block's head (`run`).
        self.graph.blocks[entry.index()].body.pop();
        self.hoisted_nodes.push(node);
        self.hoisted.insert(key, node);
        node
    }

    fn word(&mut self, word: u64) -> NodeIx {
        self.hoist(Hoisted::Tagged(word))
    }

    /// The frame state at `pc` as `st` describes it.
    fn frame_state(&mut self, pc: usize, st: &State) -> StateIx {
        let slots = st
            .slots
            .iter()
            .map(|slot| match *slot {
                Slot::Dirty(value) => SlotSource::Value(value),
                _ => SlotSource::Frame,
            })
            .collect();
        self.graph.add_state(FrameState {
            pc: pc as u32,
            slots,
            stack: st.stack.clone().into_boxed_slice(),
            handlers: st.handlers.clone().into_boxed_slice(),
        })
    }

    fn before(&mut self, cx: &mut Before, st: &State) -> StateIx {
        if let Some(state) = cx.state {
            return state;
        }
        let state = self.frame_state(cx.pc, st);
        cx.state = Some(state);
        state
    }

    /// An unconditional exit at `pc` with `st`'s frame; ends the block.
    fn exit(&mut self, pc: usize, reason: ExitReason, st: State) {
        let state = self.frame_state(pc, &st);
        let node = self.add(NodeOp::Exit(reason), &[], Rep::None, pc);
        self.graph.node_mut(node).state = Some(state);
    }

    // ---- values and their forms ----------------------------------------------------------------

    fn rep(&self, value: NodeIx) -> Rep {
        self.graph.node(value).rep
    }

    /// The constant `value` is, as its word.
    fn constant(&self, value: NodeIx) -> Option<u64> {
        match self.graph.node(value).op {
            NodeOp::ConstTagged(word) => Some(word),
            NodeOp::ConstF64(bits) if f64::from_bits(bits).is_nan() => Some(PACK_CANON_NAN),
            NodeOp::ConstF64(bits) => Some(bits),
            NodeOp::ConstI32(value) => Some(f64::from(value).to_bits()),
            NodeOp::ConstBool(value) => Some(PACK_BOOL | u64::from(value)),
            _ => None,
        }
    }

    fn forms(&self, st: &State, value: NodeIx) -> Forms {
        let mut forms = st.forms.get(&value).copied().unwrap_or_default();
        match self.rep(value) {
            Rep::Tagged => forms.tagged = Some(value),
            Rep::F64 => forms.f64 = Some(value),
            Rep::I32 => forms.i32 = Some(value),
            Rep::Bool => forms.boolean = Some(value),
            Rep::Cell => forms.cell = Some(value),
            Rep::None => {}
        }
        forms
    }

    /// Record that `a` and `b` hold one value.
    fn relate(&self, st: &mut State, a: NodeIx, b: NodeIx) {
        let forms = self.forms(st, a).or(self.forms(st, b));
        for member in forms.members() {
            st.forms.insert(member, forms);
        }
    }

    /// Whether `value` is proven an int32 (no check needed for its int32 form).
    fn known_int32(&self, st: &State, value: NodeIx) -> bool {
        self.forms(st, value).i32.is_some()
            || self
                .constant(value)
                .and_then(number_word)
                .and_then(exact_i32)
                .is_some()
    }

    /// Whether `value` is proven a Number.
    fn known_number(&self, st: &State, value: NodeIx) -> bool {
        let forms = self.forms(st, value);
        forms.f64.is_some()
            || forms.i32.is_some()
            || self.constant(value).and_then(number_word).is_some()
    }

    /// Whether a Number path can take `value`: a constant Number, an unboxed Number, or a word
    /// it checks (a Boolean or an object is never one).
    fn numeric(&self, value: NodeIx) -> bool {
        match self.constant(value) {
            Some(word) => number_word(word).is_some(),
            None => matches!(self.rep(value), Rep::Tagged | Rep::F64 | Rep::I32),
        }
    }

    /// Whether an int32 path can take `value` without an f64 that arithmetic made (whose int32
    /// test would fail where the arithmetic left the int32 range): a proven int32 or a word it
    /// checks.
    fn int32_operand(&self, st: &State, value: NodeIx) -> bool {
        self.known_int32(st, value)
            || (self.rep(value) == Rep::Tagged && self.constant(value).is_none())
    }

    /// `value`'s word, boxing an unboxed value.
    fn tagged(&mut self, st: &mut State, value: NodeIx, pc: usize) -> NodeIx {
        if let Some(word) = self.forms(st, value).tagged {
            return word;
        }
        if let Some(word) = self.constant(value) {
            return self.word(word);
        }
        let op = match self.rep(value) {
            Rep::F64 => NodeOp::F64ToTagged,
            Rep::I32 => NodeOp::I32ToTagged,
            Rep::Bool => NodeOp::BoolToTagged,
            // A cell always has its word as a form, and nothing else reaches here.
            _ => return value,
        };
        let word = self.add(op, &[value], Rep::Tagged, pc);
        self.relate(st, value, word);
        word
    }

    /// `value`'s Number as an f64, checking a word; None for a value that is never a Number.
    fn number(&mut self, st: &mut State, cx: &mut Before, value: NodeIx) -> Option<NodeIx> {
        if let Some(number) = self.forms(st, value).f64 {
            return Some(number);
        }
        if let Some(word) = self.constant(value) {
            let x = number_word(word)?;
            return Some(self.hoist(Hoisted::F64(x.to_bits())));
        }
        let pc = cx.pc;
        let number = match self.rep(value) {
            Rep::I32 => self.add(NodeOp::I32ToF64, &[value], Rep::F64, pc),
            Rep::Tagged => {
                let state = self.before(cx, st);
                self.add_exit(
                    NodeOp::CheckNumber,
                    &[value],
                    Rep::F64,
                    pc,
                    state,
                    ExitReason::NotNumber,
                )
            }
            _ => return None,
        };
        self.relate(st, value, number);
        Some(number)
    }

    /// `value` as an exact int32, checking a word or an f64; None for a value that is never one.
    fn int32(&mut self, st: &mut State, cx: &mut Before, value: NodeIx) -> Option<NodeIx> {
        if let Some(int) = self.forms(st, value).i32 {
            return Some(int);
        }
        if let Some(word) = self.constant(value) {
            let int = exact_i32(number_word(word)?)?;
            return Some(self.hoist(Hoisted::I32(int)));
        }
        let number = match self.rep(value) {
            Rep::F64 => value,
            Rep::Tagged => self.number(st, cx, value)?,
            _ => return None,
        };
        let state = self.before(cx, st);
        let int = self.add_exit(
            NodeOp::CheckInt32,
            &[number],
            Rep::I32,
            cx.pc,
            state,
            ExitReason::NotInt32,
        );
        self.relate(st, value, int);
        Some(int)
    }

    /// ToBoolean of `value`, as a Bool.
    fn truth(&mut self, st: &mut State, value: NodeIx, pc: usize) -> NodeIx {
        if let Some(boolean) = self.forms(st, value).boolean {
            return boolean;
        }
        if let Some(&truth) = st.truth.get(&value) {
            return truth;
        }
        if let Some(truth) = self.constant(value).and_then(const_truth) {
            return self.hoist(Hoisted::Bool(truth));
        }
        let truth = if self.rep(value) == Rep::I32 {
            let zero = self.hoist(Hoisted::I32(0));
            self.add(NodeOp::I32Cmp(Cond::Ne), &[value, zero], Rep::Bool, pc)
        } else {
            let word = self.tagged(st, value, pc);
            self.add(NodeOp::ToBoolean, &[word], Rep::Bool, pc)
        };
        st.truth.insert(value, truth);
        truth
    }

    /// The cell of an object word whose shape is one of `shapes`, checked unless the epoch's
    /// facts prove it.
    fn shaped(&mut self, st: &mut State, cx: &mut Before, word: NodeIx, shapes: &[u32]) -> NodeIx {
        if let (Some(known), Some(cell)) = (st.shapes.get(&word), self.forms(st, word).cell) {
            if known.iter().all(|shape| shapes.contains(shape)) {
                return cell;
            }
        }
        let state = self.before(cx, st);
        let cell = self.add_exit(
            NodeOp::CheckShape(shapes.to_vec()),
            &[word],
            Rep::Cell,
            cx.pc,
            state,
            ExitReason::WrongShape,
        );
        // The newest check's cell becomes the word's: a load's input is then the check that
        // proved its shape, whatever an earlier epoch checked.
        let mut forms = self.forms(st, word);
        forms.cell = Some(cell);
        for member in forms.members() {
            st.forms.insert(member, forms);
        }
        let proven = match st.shapes.get(&word) {
            Some(known) => {
                let both: Vec<u32> = known
                    .iter()
                    .copied()
                    .filter(|shape| shapes.contains(shape))
                    .collect();
                if both.is_empty() {
                    shapes.to_vec()
                } else {
                    both
                }
            }
            None => shapes.to_vec(),
        };
        st.shapes.insert(word, proven);
        cell
    }

    // ---- slots ---------------------------------------------------------------------------------

    fn slot(&self, pc: usize, slot: usize) -> Result<usize, Refusal> {
        if slot < self.n_slots {
            Ok(slot)
        } else {
            Err(Refusal::at(pc, "a slot past the frame"))
        }
    }

    fn read_slot(&mut self, st: &mut State, slot: usize, pc: usize) -> NodeIx {
        match st.slots[slot] {
            Slot::Clean(value) | Slot::Dirty(value) => value,
            // The frame holds the entry value: the entry block's read of it.
            Slot::Entry => {
                let value = self.hoist(Hoisted::Entry(slot as u16));
                st.slots[slot] = Slot::Clean(value);
                value
            }
            Slot::Frame => {
                let value = self.add(NodeOp::FrameSlot(slot as u16), &[], Rep::Tagged, pc);
                st.slots[slot] = Slot::Clean(value);
                value
            }
        }
    }

    /// The value at `loc` when `block` ends with `st`, read from the frame there if no node
    /// holds it.
    fn value_at_end(&mut self, block: BlockIx, st: &mut State, loc: Loc, pc: usize) -> NodeIx {
        let slot = match loc {
            Loc::Stack(position) => return st.stack[position],
            Loc::Slot(slot) => slot,
        };
        let value = match st.slots[slot] {
            Slot::Clean(value) | Slot::Dirty(value) => return value,
            Slot::Entry => self.hoist(Hoisted::Entry(slot as u16)),
            Slot::Frame => {
                self.add_at_end(block, NodeOp::FrameSlot(slot as u16), &[], Rep::Tagged, pc)
            }
        };
        st.slots[slot] = Slot::Clean(value);
        value
    }

    /// Write the dirty slots to the frame (before an effect).
    fn flush(&mut self, st: &mut State, pc: usize) {
        for slot in 0..self.n_slots {
            if let Slot::Dirty(value) = st.slots[slot] {
                let word = self.tagged(st, value, pc);
                self.add(NodeOp::StoreSlot(slot as u16), &[word], Rep::None, pc);
                st.slots[slot] = Slot::Clean(word);
            }
        }
    }

    /// Box every value the frame's locations hold (before a block's terminator: values cross
    /// block boundaries as words in stage 1).
    fn box_state(&mut self, st: &mut State, pc: usize) {
        for slot in 0..self.n_slots {
            if let Slot::Dirty(value) = st.slots[slot] {
                let word = self.tagged(st, value, pc);
                st.slots[slot] = Slot::Dirty(word);
            }
        }
        for position in 0..st.stack.len() {
            let value = st.stack[position];
            st.stack[position] = self.tagged(st, value, pc);
        }
        self.prune(st);
    }

    /// Drop the facts of values no later operation can reach: a value reaches an operation only
    /// through the frame (a slot or a stack position) or as `this`, so a fact about any other is
    /// dead weight on every state cloned at an edge.
    fn prune(&self, st: &mut State) {
        let mut live: crate::fasthash::FastSet<NodeIx> = st.stack.iter().copied().collect();
        live.extend(st.slots.iter().filter_map(|slot| match *slot {
            Slot::Clean(value) | Slot::Dirty(value) => Some(value),
            Slot::Entry | Slot::Frame => None,
        }));
        if let Some(&this) = self.hoisted.get(&Hoisted::This) {
            live.insert(this);
        }
        st.forms.retain(|value, forms| {
            live.contains(value) || forms.members().any(|m| live.contains(&m))
        });
        st.truth.retain(|value, _| live.contains(value));
        st.shapes.retain(|word, _| live.contains(word));
        st.fields.retain(|(word, _), _| live.contains(word));
    }

    // ---- control flow --------------------------------------------------------------------------

    /// The graph block of the Cfg block that starts at `pc`, made at its first edge.
    fn block_for(&mut self, pc: usize) -> Result<BlockIx, Refusal> {
        let id = self
            .cfg
            .block_at(pc)
            .filter(|id| self.cfg.blocks()[ix(*id)].start == pc)
            .ok_or_else(|| Refusal::at(pc, "a jump into the middle of a block"))?;
        if let Some(block) = self.blocks[ix(id)] {
            return Ok(block);
        }
        let block = self.graph.add_block(pc as u32, false);
        self.blocks[ix(id)] = Some(block);
        Ok(block)
    }

    /// The edge from `from` (whose terminator names it) to the block at `target`, leaving `st`.
    fn edge(&mut self, from: BlockIx, target: usize, st: State) -> Result<(), Refusal> {
        let id = self
            .cfg
            .block_at(target)
            .expect("block_for made the target");
        let to = self.blocks[ix(id)].expect("block_for made the target");
        self.graph.add_edge(from, to);
        if self.built[ix(id)] {
            self.back_edge(from, id, st)
        } else {
            self.incoming[ix(id)].push((from, st));
            Ok(())
        }
    }

    /// A back edge into a built loop header: its phis take their inputs, and what they do not
    /// merge must be as the header assumed.
    fn back_edge(&mut self, from: BlockIx, id: BlockId, mut st: State) -> Result<(), Refusal> {
        let pc = self.cfg.blocks()[ix(id)].start;
        let Some(header) = self.headers[ix(id)].take() else {
            return Err(Refusal::at(pc, "an edge into a block built already"));
        };
        if st.handlers != header.state.handlers || st.stack.len() != header.state.stack.len() {
            return Err(Refusal::at(pc, "a loop's back edge with another frame"));
        }
        if !self.graph.block(header.block).loop_header {
            // The first back edge makes the block a loop header, which its poll opens (where
            // the first stage polls at a backward branch's target); a throw from the poll exits
            // with the header's state.
            self.graph.blocks[header.block.index()].loop_header = true;
            let poll = self
                .graph
                .add_node(header.block, NodeOp::Poll, &[], Rep::None, pc as u32);
            let data = self.graph.node_mut(poll);
            data.state = Some(header.poll);
            data.reason = Some(ExitReason::Throw);
            let body = &mut self.graph.blocks[header.block.index()].body;
            body.pop();
            body.insert(0, poll);
        }
        for &(phi, loc) in &header.phis {
            let value = self.value_at_end(from, &mut st, loc, pc);
            self.graph.node_mut(phi).inputs.push(value);
        }
        for slot in 0..self.n_slots {
            if header.phis.iter().any(|&(_, loc)| loc == Loc::Slot(slot)) {
                continue;
            }
            let entry_value = matches!(st.slots[slot], Slot::Clean(value)
                if self.graph.node(value).op == NodeOp::EntrySlot(slot as u16));
            match (header.state.slots[slot], st.slots[slot]) {
                (Slot::Frame, Slot::Dirty(value)) => self.store_at_end(from, slot, value, pc),
                (Slot::Frame, _) | (Slot::Entry, Slot::Entry) => {}
                (Slot::Entry, Slot::Clean(_)) if entry_value => {}
                (Slot::Clean(assumed), Slot::Clean(value)) if assumed == value => {}
                (Slot::Clean(assumed), Slot::Dirty(value)) if assumed == value => {
                    self.store_at_end(from, slot, value, pc)
                }
                (Slot::Dirty(assumed), Slot::Clean(value) | Slot::Dirty(value))
                    if assumed == value => {}
                _ => return Err(Refusal::at(pc, "a loop's slot changes outside its phis")),
            }
        }
        self.headers[ix(id)] = Some(header);
        Ok(())
    }

    fn store_at_end(&mut self, block: BlockIx, slot: usize, word: NodeIx, pc: usize) {
        self.add_at_end(
            block,
            NodeOp::StoreSlot(slot as u16),
            &[word],
            Rep::None,
            pc,
        );
    }

    fn phi(&mut self, block: BlockIx, inputs: &[NodeIx], pc: usize) -> NodeIx {
        self.graph
            .add_node(block, NodeOp::Phi, inputs, Rep::Tagged, pc as u32)
    }

    /// The state at a forward join: a phi where the predecessors' values differ (`phis` gets
    /// them), the facts all of them hold.
    fn merge(
        &mut self,
        block: BlockIx,
        mut preds: Vec<(BlockIx, State)>,
        pc: usize,
        phis: &mut Vec<(NodeIx, Loc)>,
    ) -> Result<State, Refusal> {
        if preds.len() == 1 {
            return Ok(preds.pop().expect("one predecessor").1);
        }
        let handlers = preds[0].1.handlers.clone();
        if preds.iter().any(|(_, st)| st.handlers != handlers) {
            return Err(Refusal::at(pc, "the handler nesting differs at a join"));
        }
        let depth = preds[0].1.stack.len();
        let mut stack = Vec::with_capacity(depth);
        for position in 0..depth {
            let values: Vec<NodeIx> = preds.iter().map(|(_, st)| st.stack[position]).collect();
            if values.iter().all(|&value| value == values[0]) {
                stack.push(values[0]);
            } else {
                let phi = self.phi(block, &values, pc);
                phis.push((phi, Loc::Stack(position)));
                stack.push(phi);
            }
        }
        let mut slots = Vec::with_capacity(self.n_slots);
        for slot in 0..self.n_slots {
            let (merged, phi) = self.merge_slot(block, &mut preds, slot, pc);
            if let Some(phi) = phi {
                phis.push((phi, Loc::Slot(slot)));
            }
            slots.push(merged);
        }
        Ok(State {
            slots,
            stack,
            handlers,
            ..meet(&preds)
        })
    }

    fn merge_slot(
        &mut self,
        block: BlockIx,
        preds: &mut [(BlockIx, State)],
        slot: usize,
        pc: usize,
    ) -> (Slot, Option<NodeIx>) {
        let first = preds[0].1.slots[slot];
        if preds.iter().all(|(_, st)| st.slots[slot] == first) {
            return (first, None);
        }
        if preds
            .iter()
            .all(|(_, st)| !matches!(st.slots[slot], Slot::Dirty(_)))
        {
            return (Slot::Frame, None); // every frame holds its value
        }
        let mut values = Vec::with_capacity(preds.len());
        for (from, st) in preds.iter_mut() {
            values.push(self.value_at_end(*from, st, Loc::Slot(slot), pc));
        }
        if values.iter().all(|&value| value == values[0]) {
            return (Slot::Dirty(values[0]), None);
        }
        let phi = self.phi(block, &values, pc);
        (Slot::Dirty(phi), Some(phi))
    }

    /// The state at a loop header from its entry edges (see the module's doc).
    fn header(
        &mut self,
        id: BlockId,
        block: BlockIx,
        mut preds: Vec<(BlockIx, State)>,
        found: usize,
    ) -> Result<State, Refusal> {
        let pc = self.cfg.blocks()[ix(id)].start;
        let flow = self.flow;
        let lp = &flow.loops[found];
        let handlers = preds[0].1.handlers.clone();
        if preds.iter().any(|(_, st)| st.handlers != handlers) {
            return Err(Refusal::at(pc, "the handler nesting differs at a join"));
        }
        let mut phis = Vec::new();
        let depth = preds[0].1.stack.len();
        let mut stack = Vec::with_capacity(depth);
        for position in 0..depth {
            let values: Vec<NodeIx> = preds.iter().map(|(_, st)| st.stack[position]).collect();
            let phi = self.phi(block, &values, pc);
            phis.push((phi, Loc::Stack(position)));
            stack.push(phi);
        }
        let mut slots = Vec::with_capacity(self.n_slots);
        for slot in 0..self.n_slots {
            if lp.written[slot] {
                let mut values = Vec::with_capacity(preds.len());
                for (from, st) in preds.iter_mut() {
                    values.push(self.value_at_end(*from, st, Loc::Slot(slot), pc));
                }
                let phi = self.phi(block, &values, pc);
                phis.push((phi, Loc::Slot(slot)));
                slots.push(Slot::Dirty(phi));
            } else if lp.effects {
                for (from, st) in preds.iter_mut() {
                    if let Slot::Dirty(value) = st.slots[slot] {
                        self.store_at_end(*from, slot, value, pc);
                        st.slots[slot] = Slot::Clean(value);
                    }
                }
                slots.push(Slot::Frame);
            } else {
                let (merged, phi) = self.merge_slot(block, &mut preds, slot, pc);
                if let Some(phi) = phi {
                    phis.push((phi, Loc::Slot(slot)));
                }
                slots.push(merged);
            }
        }
        let met = meet(&preds);
        let st = State {
            slots,
            stack,
            handlers,
            forms: met.forms,
            truth: met.truth,
            ..State::default()
        };
        let poll = self.frame_state(pc, &st);
        self.headers[ix(id)] = Some(Header {
            block,
            phis,
            state: st.clone(),
            poll,
        });
        Ok(st)
    }

    fn jump(&mut self, pc: usize, target: usize, mut st: State) -> Result<(), Refusal> {
        self.box_state(&mut st, pc);
        let to = self.block_for(target)?;
        self.add(NodeOp::Jump(to), &[], Rep::None, pc);
        let from = self.cur;
        self.edge(from, target, st)
    }

    fn branch(
        &mut self,
        pc: usize,
        cond: NodeIx,
        then_pc: usize,
        else_pc: usize,
        mut st: State,
    ) -> Result<(), Refusal> {
        if self.cfg.block_at(then_pc) == self.cfg.block_at(else_pc) {
            return self.jump(pc, then_pc, st);
        }
        self.box_state(&mut st, pc);
        let then_block = self.block_for(then_pc)?;
        let else_block = self.block_for(else_pc)?;
        self.add(
            NodeOp::Branch {
                then_block,
                else_block,
            },
            &[cond],
            Rep::None,
            pc,
        );
        let from = self.cur;
        self.edge(from, then_pc, st.clone())?;
        self.edge(from, else_pc, st)
    }

    /// JumpIfNotNullishPeek: to `target` unless the top of the stack is undefined or null, two
    /// identity tests where no fact decides it.
    fn nullish(&mut self, pc: usize, target: usize, mut st: State) -> Result<(), Refusal> {
        let value = *st.stack.last().expect("the stack analysis saw the operand");
        let forms = self.forms(&st, value);
        let known = match self.constant(value) {
            Some(word) => Some(word == PACK_UNDEFINED || word == PACK_NULL),
            None if forms.f64.is_some()
                || forms.i32.is_some()
                || forms.boolean.is_some()
                || forms.cell.is_some() =>
            {
                Some(false)
            }
            None => None,
        };
        match known {
            Some(true) => return self.jump(pc, pc + 1, st),
            Some(false) => return self.jump(pc, target, st),
            None => {}
        }
        if self.cfg.block_at(pc + 1) == self.cfg.block_at(target) {
            return self.jump(pc, target, st);
        }
        self.box_state(&mut st, pc);
        let word = *st.stack.last().expect("the stack analysis saw the operand");
        let undefined = self.word(PACK_UNDEFINED);
        let null = self.word(PACK_NULL);
        let nullish = self.block_for(pc + 1)?;
        let taken = self.block_for(target)?;
        let first = self.cur;
        let is_undefined = self.add(NodeOp::TaggedSame, &[word, undefined], Rep::Bool, pc);
        let second = self.graph.add_block(pc as u32, false);
        self.add(
            NodeOp::Branch {
                then_block: nullish,
                else_block: second,
            },
            &[is_undefined],
            Rep::None,
            pc,
        );
        self.edge(first, pc + 1, st.clone())?;
        self.graph.add_edge(first, second);
        self.cur = second;
        let is_null = self.add(NodeOp::TaggedSame, &[word, null], Rep::Bool, pc);
        self.add(
            NodeOp::Branch {
                then_block: nullish,
                else_block: taken,
            },
            &[is_null],
            Rep::None,
            pc,
        );
        self.edge(second, pc + 1, st.clone())?;
        self.edge(second, target, st)
    }

    // ---- operations ----------------------------------------------------------------------------

    /// Lower the operation at `pc`; false when it ended the block.
    fn op(&mut self, pc: usize, st: &mut State) -> Result<bool, Refusal> {
        let op = self.ops[pc];
        match op {
            Op::Const(k) => match const_word(self.chunk, k) {
                Some(word) => {
                    let value = self.word(word);
                    st.stack.push(value);
                }
                None => self.effect(pc, st)?,
            },
            Op::Undef => {
                let value = self.word(PACK_UNDEFINED);
                st.stack.push(value);
            }
            Op::Dup => {
                let value = st.stack[st.stack.len() - 1];
                st.stack.push(value);
            }
            Op::Dup2 => {
                let at = st.stack.len() - 2;
                let pair = [st.stack[at], st.stack[at + 1]];
                st.stack.extend_from_slice(&pair);
            }
            Op::Pop => {
                st.stack.pop();
            }
            // A read the stack analysis proves initialized (Cfg::initialized_reads); any other
            // tests the slot for its TDZ marker in the baseline template.
            Op::LoadLocal(slot) if self.initialized.contains(pc) => {
                let slot = self.slot(pc, slot as usize)?;
                let value = self.read_slot(st, slot, pc);
                st.stack.push(value);
            }
            Op::StoreLocal(slot) => {
                let slot = self.slot(pc, slot as usize)?;
                let value = st.stack.pop().expect("the stack analysis saw the operand");
                st.slots[slot] = Slot::Dirty(value);
            }
            Op::Tdz(slot) => {
                let slot = self.slot(pc, slot as usize)?;
                let empty = self.word(PACK_EMPTY);
                st.slots[slot] = Slot::Dirty(empty);
            }
            Op::ResetSlots(start, count) => {
                let undefined = self.word(PACK_UNDEFINED);
                for slot in start as usize..start as usize + count as usize {
                    let slot = self.slot(pc, slot)?;
                    st.slots[slot] = Slot::Dirty(undefined);
                }
            }
            Op::LoadThis => {
                let this = self.hoist(Hoisted::This);
                st.stack.push(this);
            }
            Op::Void => {
                st.stack.pop();
                let undefined = self.word(PACK_UNDEFINED);
                st.stack.push(undefined);
            }
            Op::Not => {
                let value = st.stack.pop().expect("the stack analysis saw the operand");
                let truth = self.truth(st, value, pc);
                let not = self.add(NodeOp::BoolNot, &[truth], Rep::Bool, pc);
                st.stack.push(not);
            }
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Mod => self.arithmetic(pc, op, st)?,
            Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr | Op::UShr => {
                self.bitwise(pc, op, st)?
            }
            Op::Neg | Op::Plus | Op::BitNot => self.unary(pc, op, st)?,
            Op::Lt | Op::Gt | Op::Le | Op::Ge => self.relational(pc, op, st)?,
            Op::StrictEq | Op::StrictNotEq | Op::EqEq | Op::NotEq => self.equality(pc, op, st)?,
            Op::UpdateLocal(slot, kind) => {
                let slot = self.slot(pc, slot as usize)?;
                self.update_local(pc, slot, kind, st)?
            }
            Op::GetProp(name, _) => self.named_load(pc, name, Recv::Stack, false, st)?,
            Op::GetMethod(name, _) => self.named_load(pc, name, Recv::Stack, true, st)?,
            Op::GetPropThis(name, _) => self.named_load(pc, name, Recv::This, false, st)?,
            Op::GetPropLocal(slot, name, _) => {
                let slot = self.slot(pc, slot as usize)?;
                self.named_load(pc, name, Recv::Slot(slot), false, st)?
            }
            Op::SetProp(name, _) => self.named_store(pc, name, Recv::Stack, true, st)?,
            Op::SetPropDrop(name, _) => self.named_store(pc, name, Recv::Stack, false, st)?,
            Op::SetPropThisDrop(name, _) => self.named_store(pc, name, Recv::This, false, st)?,
            Op::SetPropLocalDrop(slot, name, _) => {
                let slot = self.slot(pc, slot as usize)?;
                self.named_store(pc, name, Recv::Slot(slot), false, st)?
            }
            Op::Jump(target) => {
                self.jump(pc, target as usize, std::mem::take(st))?;
                return Ok(false);
            }
            Op::JumpIfFalse(target) => {
                let value = st.stack.pop().expect("the stack analysis saw the operand");
                let cond = self.truth(st, value, pc);
                self.branch(pc, cond, pc + 1, target as usize, std::mem::take(st))?;
                return Ok(false);
            }
            Op::JumpIfFalsePeek(target) => {
                let value = st.stack[st.stack.len() - 1];
                let cond = self.truth(st, value, pc);
                self.branch(pc, cond, pc + 1, target as usize, std::mem::take(st))?;
                return Ok(false);
            }
            Op::JumpIfTruePeek(target) => {
                let value = st.stack[st.stack.len() - 1];
                let cond = self.truth(st, value, pc);
                self.branch(pc, cond, target as usize, pc + 1, std::mem::take(st))?;
                return Ok(false);
            }
            Op::JumpIfNotNullishPeek(target) => {
                self.nullish(pc, target as usize, std::mem::take(st))?;
                return Ok(false);
            }
            Op::InlineGuard(target, _) => {
                self.inline_guard(pc, target, st)?;
                self.jump(pc, pc + 1, std::mem::take(st))?;
                return Ok(false);
            }
            Op::PushHandler(_) | Op::PushIterator(..) => {
                st.handlers.push((pc as u32, st.stack.len() as u32));
            }
            Op::PopHandler => {
                if st.handlers.pop().is_none() {
                    return Err(Refusal::at(pc, "a handler popped outside every region"));
                }
            }
            Op::Return | Op::ReturnBare | Op::ReturnUndef => {
                if st.in_iterator(self.ops) {
                    self.exit(pc, ExitReason::Unsupported, std::mem::take(st));
                    return Ok(false);
                }
                let value = if op == Op::Return {
                    let value = st.stack.pop().expect("the stack analysis saw the operand");
                    self.tagged(st, value, pc)
                } else {
                    self.word(PACK_UNDEFINED)
                };
                self.add(NodeOp::Return, &[value], Rep::None, pc);
                return Ok(false);
            }
            // The VM performs these: a throw (which the VM routes to a handler or out), a proper
            // tail call that runs, an iterator's abort pad.
            Op::Throw | Op::TailCall(..) | Op::IterAbortL(_) | Op::IterAbortIfNotDoneL(..) => {
                self.exit(pc, ExitReason::Unsupported, std::mem::take(st));
                return Ok(false);
            }
            _ => self.effect(pc, st)?,
        }
        Ok(true)
    }

    /// The operation at `pc` as its baseline template on the canonical frame.
    fn effect(&mut self, pc: usize, st: &mut State) -> Result<(), Refusal> {
        let (pops, pushes) = self
            .chunk
            .jit_stack_effect(pc)
            .ok_or_else(|| Refusal::at(pc, "an operation without a stack effect"))?;
        let (Ok(pop_count), Ok(push_count)) = (u8::try_from(pops), u8::try_from(pushes)) else {
            return Err(Refusal::at(pc, "an operation over 255 operands"));
        };
        self.flush(st, pc);
        let len = st.stack.len();
        let mut inputs = Vec::with_capacity(pops);
        for position in len - pops..len {
            let value = st.stack[position];
            let word = self.tagged(st, value, pc);
            st.stack[position] = word;
            inputs.push(word);
        }
        let state = self.frame_state(pc, st);
        let effect = self.add_exit(
            NodeOp::Effect {
                pc: pc as u32,
                pops: pop_count,
                pushes: push_count,
            },
            &inputs,
            Rep::None,
            pc,
            state,
            ExitReason::Throw,
        );
        st.stack.truncate(len - pops);
        for result in 0..push_count {
            let value = self.add(NodeOp::Projection(result), &[effect], Rep::Tagged, pc);
            st.stack.push(value);
        }
        st.slots.fill(Slot::Frame);
        st.clear_heap();
        Ok(())
    }

    /// Add, Sub, Mul, Div and Mod: f64 arithmetic on Numbers, int32 with an overflow exit where
    /// rule U's hint or the operands' facts say int32 (Add and Sub on proven int32s; Mul only on
    /// the hint, its int32 products leaving the range too often); the generic operation where a
    /// role saw a value that is not a Number.
    fn arithmetic(&mut self, pc: usize, op: Op, st: &mut State) -> Result<(), Refusal> {
        let Some(values) = self.snapshot.values(pc) else {
            return self.effect(pc, st);
        };
        let len = st.stack.len();
        let (a, b) = (st.stack[len - 2], st.stack[len - 1]);
        if (values.operand0 | values.operand1) & !NUMBER != 0
            || !self.numeric(a)
            || !self.numeric(b)
        {
            return self.effect(pc, st);
        }
        let int = match op {
            Op::Add | Op::Sub => {
                (self.known_int32(st, a) && self.known_int32(st, b))
                    || (int32_hint(&values)
                        && self.int32_operand(st, a)
                        && self.int32_operand(st, b))
            }
            Op::Mul => {
                int32_hint(&values) && self.int32_operand(st, a) && self.int32_operand(st, b)
            }
            _ => false,
        };
        let mut cx = Before::new(pc);
        let result = if int {
            let (Some(x), Some(y)) = (self.int32(st, &mut cx, a), self.int32(st, &mut cx, b))
            else {
                return Err(Refusal::at(pc, "an int32 operand without an int32 form"));
            };
            let node = match op {
                Op::Add => NodeOp::I32Add,
                Op::Sub => NodeOp::I32Sub,
                _ => NodeOp::I32Mul,
            };
            let state = self.before(&mut cx, st);
            self.add_exit(node, &[x, y], Rep::I32, pc, state, ExitReason::Overflow)
        } else {
            let (Some(x), Some(y)) = (self.number(st, &mut cx, a), self.number(st, &mut cx, b))
            else {
                return Err(Refusal::at(pc, "a Number operand without an f64 form"));
            };
            let node = match op {
                Op::Add => NodeOp::F64Add,
                Op::Sub => NodeOp::F64Sub,
                Op::Mul => NodeOp::F64Mul,
                Op::Div => NodeOp::F64Div,
                _ => NodeOp::F64Mod,
            };
            let result = self.add(node, &[x, y], Rep::F64, pc);
            if op == Op::Mod {
                st.clear_heap(); // F64Mod calls out (NodeOp::is_effectful)
            }
            result
        };
        st.stack.truncate(len - 2);
        st.stack.push(result);
        Ok(())
    }

    /// The bitwise operators on int32s: a proven int32 or a word checked as one. An f64 that
    /// arithmetic made takes the generic operation (ToInt32 truncates it, which a check would
    /// fail), and so does `>>>` unless its count is a constant whose low five bits are not all
    /// zero (otherwise the result can leave the int32 range).
    fn bitwise(&mut self, pc: usize, op: Op, st: &mut State) -> Result<(), Refusal> {
        let Some(values) = self.snapshot.values(pc) else {
            return self.effect(pc, st);
        };
        let len = st.stack.len();
        let (a, b) = (st.stack[len - 2], st.stack[len - 1]);
        let fits = op != Op::UShr
            || self
                .constant(b)
                .and_then(number_word)
                .and_then(exact_i32)
                .is_some_and(|count| count & 31 != 0);
        if (values.operand0 | values.operand1) & !NUMBER != 0
            || !self.int32_operand(st, a)
            || !self.int32_operand(st, b)
            || !fits
        {
            return self.effect(pc, st);
        }
        let mut cx = Before::new(pc);
        let (Some(x), Some(y)) = (self.int32(st, &mut cx, a), self.int32(st, &mut cx, b)) else {
            return Err(Refusal::at(pc, "an int32 operand without an int32 form"));
        };
        let node = match op {
            Op::BitAnd => NodeOp::I32And,
            Op::BitOr => NodeOp::I32Or,
            Op::BitXor => NodeOp::I32Xor,
            Op::Shl => NodeOp::I32Shl,
            Op::Shr => NodeOp::I32Sar,
            _ => NodeOp::I32UShr,
        };
        let result = if node == NodeOp::I32UShr {
            let state = self.before(&mut cx, st);
            self.add_exit(node, &[x, y], Rep::I32, pc, state, ExitReason::Overflow)
        } else {
            self.add(node, &[x, y], Rep::I32, pc)
        };
        st.stack.truncate(len - 2);
        st.stack.push(result);
        Ok(())
    }

    /// Neg, Plus (ToNumber) and BitNot on Numbers.
    fn unary(&mut self, pc: usize, op: Op, st: &mut State) -> Result<(), Refusal> {
        let Some(values) = self.snapshot.values(pc) else {
            return self.effect(pc, st);
        };
        let len = st.stack.len();
        let value = st.stack[len - 1];
        let usable = match op {
            Op::BitNot => self.int32_operand(st, value),
            _ => self.numeric(value),
        };
        if values.operand0 & !NUMBER != 0 || !usable {
            return self.effect(pc, st);
        }
        let mut cx = Before::new(pc);
        let result = match op {
            Op::BitNot => {
                let Some(x) = self.int32(st, &mut cx, value) else {
                    return Err(Refusal::at(pc, "an int32 operand without an int32 form"));
                };
                self.add(NodeOp::I32Not, &[x], Rep::I32, pc)
            }
            // ToNumber of a Number is the Number: an int32 stays one.
            Op::Plus if self.rep(value) == Rep::I32 => value,
            _ => {
                let Some(x) = self.number(st, &mut cx, value) else {
                    return Err(Refusal::at(pc, "a Number operand without an f64 form"));
                };
                if op == Op::Neg {
                    self.add(NodeOp::F64Neg, &[x], Rep::F64, pc)
                } else {
                    x
                }
            }
        };
        st.stack.truncate(len - 1);
        st.stack.push(result);
        Ok(())
    }

    /// Lt, Gt, Le and Ge on Numbers (rule U's in-line set), int32 where both are proven or the
    /// hint says so.
    fn relational(&mut self, pc: usize, op: Op, st: &mut State) -> Result<(), Refusal> {
        let Some(values) = self.snapshot.values(pc) else {
            return self.effect(pc, st);
        };
        let len = st.stack.len();
        let (a, b) = (st.stack[len - 2], st.stack[len - 1]);
        if (values.operand0 | values.operand1) & !NUMBER != 0
            || !self.numeric(a)
            || !self.numeric(b)
        {
            return self.effect(pc, st);
        }
        let cond = match op {
            Op::Lt => Cond::Lt,
            Op::Gt => Cond::Gt,
            Op::Le => Cond::Le,
            _ => Cond::Ge,
        };
        let int = (self.known_int32(st, a) && self.known_int32(st, b))
            || (int32_hint(&values) && self.int32_operand(st, a) && self.int32_operand(st, b));
        let result = self.compare(pc, cond, a, b, int, st)?;
        st.stack.truncate(len - 2);
        st.stack.push(result);
        Ok(())
    }

    /// A Number comparison of `a` and `b`, int32 or f64.
    fn compare(
        &mut self,
        pc: usize,
        cond: Cond,
        a: NodeIx,
        b: NodeIx,
        int: bool,
        st: &mut State,
    ) -> Result<NodeIx, Refusal> {
        let mut cx = Before::new(pc);
        if int {
            if let (Some(x), Some(y)) = (self.int32(st, &mut cx, a), self.int32(st, &mut cx, b)) {
                return Ok(self.add(NodeOp::I32Cmp(cond), &[x, y], Rep::Bool, pc));
            }
        }
        let (Some(x), Some(y)) = (self.number(st, &mut cx, a), self.number(st, &mut cx, b)) else {
            return Err(Refusal::at(pc, "a Number operand without an f64 form"));
        };
        Ok(self.add(NodeOp::F64Cmp(cond), &[x, y], Rep::Bool, pc))
    }

    /// Equality, which records nothing (W4 decision 3), typed from its operands as JSC does: two
    /// proven Numbers compare as Numbers; words whose kinds make identity the answer (two proven
    /// Booleans or objects, or under `===` a constant no other value equals) compare as words;
    /// anything else is the generic operation.
    fn equality(&mut self, pc: usize, op: Op, st: &mut State) -> Result<(), Refusal> {
        let strict = matches!(op, Op::StrictEq | Op::StrictNotEq);
        let negate = matches!(op, Op::StrictNotEq | Op::NotEq);
        let len = st.stack.len();
        let (a, b) = (st.stack[len - 2], st.stack[len - 1]);
        let (fa, fb) = (self.forms(st, a), self.forms(st, b));
        let result = if self.known_number(st, a) && self.known_number(st, b) {
            let cond = if negate { Cond::Ne } else { Cond::Eq };
            let int = self.known_int32(st, a) && self.known_int32(st, b);
            self.compare(pc, cond, a, b, int, st)?
        } else if (fa.boolean.is_some() && fb.boolean.is_some())
            || (fa.cell.is_some() && fb.cell.is_some())
            || (strict
                && (self.constant(a).is_some_and(identity_constant)
                    || self.constant(b).is_some_and(identity_constant)))
        {
            let x = self.tagged(st, a, pc);
            let y = self.tagged(st, b, pc);
            let same = self.add(NodeOp::TaggedSame, &[x, y], Rep::Bool, pc);
            if negate {
                self.add(NodeOp::BoolNot, &[same], Rep::Bool, pc)
            } else {
                same
            }
        } else {
            return self.effect(pc, st);
        };
        st.stack.truncate(len - 2);
        st.stack.push(result);
        Ok(())
    }

    /// `++` and `--` on a slot holding a Number.
    fn update_local(
        &mut self,
        pc: usize,
        slot: usize,
        kind: UpdKind,
        st: &mut State,
    ) -> Result<(), Refusal> {
        let Some(values) = self.snapshot.values(pc) else {
            return self.effect(pc, st);
        };
        if values.operand0 & !NUMBER != 0 {
            return self.effect(pc, st);
        }
        let old = self.read_slot(st, slot, pc);
        if !self.numeric(old) {
            return self.effect(pc, st);
        }
        let increment = matches!(
            kind,
            UpdKind::PreInc | UpdKind::PostInc | UpdKind::IncDiscard
        );
        let int = self.known_int32(st, old) || (int32_hint(&values) && self.int32_operand(st, old));
        let mut cx = Before::new(pc);
        let (number, new) = match int.then(|| self.int32(st, &mut cx, old)).flatten() {
            Some(x) => {
                let one = self.hoist(Hoisted::I32(1));
                let node = if increment {
                    NodeOp::I32Add
                } else {
                    NodeOp::I32Sub
                };
                let state = self.before(&mut cx, st);
                let new = self.add_exit(node, &[x, one], Rep::I32, pc, state, ExitReason::Overflow);
                (x, new)
            }
            None => {
                let Some(x) = self.number(st, &mut cx, old) else {
                    return Err(Refusal::at(pc, "a Number operand without an f64 form"));
                };
                let one = self.hoist(Hoisted::F64(1f64.to_bits()));
                let node = if increment {
                    NodeOp::F64Add
                } else {
                    NodeOp::F64Sub
                };
                (x, self.add(node, &[x, one], Rep::F64, pc))
            }
        };
        st.slots[slot] = Slot::Dirty(new);
        match kind {
            UpdKind::PreInc | UpdKind::PreDec => st.stack.push(new),
            UpdKind::PostInc | UpdKind::PostDec => st.stack.push(number),
            UpdKind::IncDiscard | UpdKind::DecDiscard => {}
        }
        Ok(())
    }

    /// A named site's own data ways (one shape, or up to four sharing their location and slot),
    /// None where the site takes the generic operation: a generic mark, another kind of way, ways
    /// at different locations, no way, or a store site without its writability proof. A
    /// `prototype` property may hold a function's deferred prototype, which only the checked
    /// operation materializes.
    fn own_data_ways(&self, pc: usize, name: u32, store: bool) -> Option<Ways> {
        if self.chunk.jit_name(name) == "prototype" {
            return None;
        }
        let named = self.snapshot.named(pc)?;
        if named.generic || (store && !named.writable) {
            return None;
        }
        let mut shapes = Vec::new();
        let mut at = None;
        for way in named.filled() {
            let own = way.kind == NamedWayKind::Data
                && way.depth == 0
                && way.receiver_shape == way.holder_shape
                && crate::value::is_cacheable_shape(way.receiver_shape)
                && way.location != 0
                && way.slot < MAX_FIELD_SLOT;
            if !own || at.is_some_and(|at| at != (way.location, way.slot)) {
                return None;
            }
            at = Some((way.location, way.slot));
            if !shapes.contains(&way.receiver_shape) {
                shapes.push(way.receiver_shape);
            }
        }
        let (location, slot) = at?;
        Some(Ways {
            shapes,
            location,
            slot,
        })
    }

    fn receiver(&mut self, recv: Recv, st: &mut State, below: usize, pc: usize) -> NodeIx {
        match recv {
            Recv::Stack => st.stack[st.stack.len() - 1 - below],
            Recv::This => self.hoist(Hoisted::This),
            Recv::Slot(slot) => self.read_slot(st, slot, pc),
        }
    }

    /// GetProp, GetMethod (which keeps its receiver), GetPropThis and GetPropLocal on own data
    /// ways: the shape checked (unless proven) and the field loaded (unless loaded this epoch).
    fn named_load(
        &mut self,
        pc: usize,
        name: u32,
        recv: Recv,
        method: bool,
        st: &mut State,
    ) -> Result<(), Refusal> {
        let Some(ways) = self.own_data_ways(pc, name, false) else {
            return self.effect(pc, st);
        };
        let receiver = self.receiver(recv, st, 0, pc);
        let mut cx = Before::new(pc);
        let word = self.tagged(st, receiver, pc);
        let cell = self.shaped(st, &mut cx, word, &ways.shapes);
        let value = match st.fields.get(&(word, ways.location)) {
            Some(&value) => value,
            None => {
                let value = self.add(
                    NodeOp::LoadField {
                        location: ways.location,
                        slot: ways.slot,
                    },
                    &[cell],
                    Rep::Tagged,
                    pc,
                );
                st.fields.insert((word, ways.location), value);
                value
            }
        };
        if matches!(recv, Recv::Stack) && !method {
            st.stack.pop();
        }
        st.stack.push(value);
        Ok(())
    }

    /// SetProp (which keeps the value), SetPropDrop, SetPropThisDrop and SetPropLocalDrop on own
    /// writable data ways.
    fn named_store(
        &mut self,
        pc: usize,
        name: u32,
        recv: Recv,
        keep: bool,
        st: &mut State,
    ) -> Result<(), Refusal> {
        let Some(ways) = self.own_data_ways(pc, name, true) else {
            return self.effect(pc, st);
        };
        let len = st.stack.len();
        let value = st.stack[len - 1];
        let receiver = self.receiver(recv, st, 1, pc);
        let mut cx = Before::new(pc);
        let word = self.tagged(st, receiver, pc);
        let cell = self.shaped(st, &mut cx, word, &ways.shapes);
        let stored = self.tagged(st, value, pc);
        self.add(
            NodeOp::StoreField {
                location: ways.location,
                slot: ways.slot,
            },
            &[cell, stored],
            Rep::None,
            pc,
        );
        st.fields
            .retain(|&(_, location), _| location != ways.location);
        st.fields.insert((word, ways.location), stored);
        let pops = if matches!(recv, Recv::Stack) { 2 } else { 1 };
        st.stack.truncate(len - pops);
        if keep {
            st.stack.push(value);
        }
        Ok(())
    }

    /// InlineGuard (a spliced callee's): the callee and the Realm compared, and for a sloppy
    /// callee that reads `this` the receiver an object (CheckCallee, its inputs the callee's word
    /// and then the receiver's); a failure exits at the guard, whose mismatch the VM then takes,
    /// so the mismatch edge is not built.
    fn inline_guard(&mut self, pc: usize, target: u32, st: &mut State) -> Result<(), Refusal> {
        let guard = self.chunk.jit_inline_target(target);
        let callee_depth = guard.argc as usize + 1;
        let check_this = guard.check_this;
        let len = st.stack.len();
        if len < callee_depth + usize::from(check_this) {
            return Err(Refusal::at(pc, "an inline guard above too few operands"));
        }
        let mut cx = Before::new(pc);
        let callee = st.stack[len - callee_depth];
        let mut inputs = vec![self.tagged(st, callee, pc)];
        if check_this {
            let receiver = st.stack[len - callee_depth - 1];
            inputs.push(self.tagged(st, receiver, pc));
        }
        let state = self.before(&mut cx, st);
        self.add_exit(
            NodeOp::CheckCallee { target },
            &inputs,
            Rep::None,
            pc,
            state,
            ExitReason::WrongCallee,
        );
        Ok(())
    }
}
