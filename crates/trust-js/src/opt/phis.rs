//! The phi representation selector (P3 stage 2 step 6; the plan's "Step 6, untagged phis", its
//! input target/p3/stage2/phis-input.txt): Maglev's structure, a pass between the build and the
//! verifier over every phi, loop headers' and joins' alike, that rewrites their inputs and their
//! uses (maglev-phi-representation-selector.cc), with SpiderMonkey's rule from the inputs alone
//! (TypeAnalyzer::guessPhiType and propagateSpecialization, jit/TypeAnalysis.cpp).
//!
//! - The builder makes every phi Tagged, each input a word boxed at its predecessor's end with
//!   that input's untagged forms there noted ([`PhiNotes`]), and builds the phi's uses on the
//!   word: CheckNumber of it and the conversions of that, ToBoolean of it, slot stores, effects'
//!   operands, element stores' values, and the uses only a word serves.
//! - Each input votes: a boxing of an int32 or an f64 its operand's representation (Maglev's
//!   conversion inputs), a noted form its own (Maglev's alternatives), a Number constant I32
//!   where it is an exact int32 other than -0 and F64 otherwise, an untagged phi its
//!   representation, the phi itself nothing (a self back edge), a phi not yet decided nothing yet;
//!   any other input is Tagged.
//! - The phi: all I32, I32; I32 and F64, F64; anything else, Tagged. Phis start undecided and only
//!   widen, in block order until none changes, the undecided ones then Tagged and their users
//!   decided again (SpiderMonkey's optimistic specialization and its propagation to the phis
//!   that use a phi; Maglev retries a block's phis): a loop phi and an inner join's phi that feed
//!   each other settle together. Inputs only widen: an int32 input of an F64 phi is converted at
//!   its predecessor's end (Maglev's ChangeInt32ToFloat64, SpiderMonkey's MToDouble), and an f64
//!   is never narrowed, which could exit at every iteration (Maglev's deopt-loop rule). Mixed
//!   inputs make an F64 phi whatever its uses, where Maglev keeps such a phi tagged when a use
//!   wants an int32: its reasons, Smis and HeapNumbers, TRust lacks (Decision 5), and a TRust
//!   tagged phi is its costliest form (the plan's departure 1).
//! - A loop's entry edge speculates (stage 2 step 6 (e)): a word without forms (a parameter, a
//!   reload, a load) that enters a loop header's phi from a predecessor of that one successor,
//!   where the phi has a numeric use in its loop (CheckNumber of it: Maglev's same-loop use
//!   hints), votes nothing and takes the phi's representation: CheckNumber, and for an I32 phi
//!   CheckInt32, at that predecessor's end, exiting with PhiNotNumber or PhiNotInt32 at the
//!   header's pc with the state that edge leaves. That is Maglev's hoisted untagging of a loop
//!   phi's entry input (UntagInputWithHoistedUntagging), which Maglev keeps behind an
//!   experimental flag until it has feedback to back off with (maglev-phi-representation-
//!   selector.cc:267-286) and reaches by peeling its loops: TRust's exit profile is that back-off,
//!   a flag at the header declining the speculation there, PhiNotInt32 to an F64 phi and
//!   PhiNotNumber to a tagged one (the plan's departures 2 and 3: the check by default where V8
//!   peels; a flag per header, not per phi). No speculation on back edges or at forward joins
//!   (Maglev's default outside peeled loops; departure 4).
//! - Inputs of an untagged phi: a boxing's operand, a noted form, a constant of the phi's
//!   representation, an untagged phi as it is or widened, a speculation's check.
//! - Uses of an untagged phi: CheckNumber of an I32 phi becomes I32ToF64 of it, and CheckInt32,
//!   CheckIndex and TruncateToInt32 of that the phi itself; CheckNumber of an F64 phi is the phi;
//!   ToBoolean becomes I32Cmp(Ne, 0) or F64ToBoolean (Maglev's Int32ToBoolean and
//!   Float64ToBoolean); slot stores, effects' operands, element stores' values and frame states
//!   take it as it is (each boxes what it stores); a use only a word serves (a return, a field's
//!   or binding's store, an identity test, an object check, a tagged phi's input) takes a
//!   retagging, one per block before its first such use, or at a predecessor's end for a phi's
//!   input (Maglev's EnsurePhiTagged). The checks a rewrite replaces leave the graph, and so do
//!   the boxings and widenings no use is left of; the graph's nodes are numbered again, so that
//!   every node still sits in one block (the verifier's invariant).

use super::build::{Alternatives, PhiNotes};
use super::ir::{
    BlockIx, Cond, Decline, ExitReason, Graph, NodeIx, NodeOp, Rep, SlotSource, StateIx,
};
use super::snapshot::Snapshot;
use crate::fasthash::FastMap;

/// A phi's representation as the selection knows it (a lattice that only widens).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    /// No input has decided it yet.
    Undecided,
    I32,
    F64,
    Tagged,
}

impl Kind {
    fn of(rep: Rep) -> Kind {
        match rep {
            Rep::I32 => Kind::I32,
            Rep::F64 => Kind::F64,
            _ => Kind::Tagged,
        }
    }

    fn rep(self) -> Rep {
        match self {
            Kind::I32 => Rep::I32,
            Kind::F64 => Rep::F64,
            Kind::Undecided | Kind::Tagged => Rep::Tagged,
        }
    }
}

/// Where a phi input's untagged value would come from.
#[derive(Clone, Copy, Debug)]
enum Source {
    /// The phi itself, on a back edge that keeps its value: no vote.
    Own,
    /// A phi not decided yet: no vote yet.
    Pending,
    /// A node of `rep` (I32 or F64) available at the predecessor's end.
    Untagged(NodeIx, Rep),
    /// A Number constant.
    Number(f64),
    /// A word without forms on a loop's entry edge, checked there for the phi's representation:
    /// no vote (Undecided), or the vote a flag at the header leaves it (F64 for PhiNotInt32,
    /// Tagged for PhiNotNumber).
    Speculative(Kind),
    /// Any other word.
    Tagged,
}

impl Source {
    fn vote(self) -> Kind {
        match self {
            Source::Own | Source::Pending => Kind::Undecided,
            Source::Untagged(_, rep) => Kind::of(rep),
            Source::Number(x) => match exact_i32(x) {
                Some(_) => Kind::I32,
                None => Kind::F64,
            },
            Source::Speculative(kind) => kind,
            Source::Tagged => Kind::Tagged,
        }
    }
}

/// The f64 a tagged word holds, None for a word that is not a Number.
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

/// Pick every phi's representation in `graph`, as built from its chunk with `notes`, and rewrite
/// the phis, their inputs and their uses to match (see the module's doc).
pub(crate) fn select(graph: &mut Graph, notes: &PhiNotes, snapshot: &Snapshot) {
    let phis: Vec<NodeIx> = graph
        .blocks
        .iter()
        .flat_map(|block| block.phis.iter().copied())
        .collect();
    if phis.is_empty() {
        return;
    }
    let mut selector = Selector::new(graph, notes, snapshot);
    selector.decide(&phis);
    selector.declines(&phis);
    let untagged: Vec<NodeIx> = phis
        .iter()
        .copied()
        .filter(|&phi| matches!(selector.kind(phi), Kind::I32 | Kind::F64))
        .collect();
    if untagged.is_empty() {
        return;
    }
    selector.untag_inputs(&untagged);
    selector.rewrite_uses(&untagged);
    selector.finish();
}

struct Selector<'g> {
    graph: &'g mut Graph,
    notes: &'g PhiNotes,
    /// Each phi's kind, by node.
    kinds: FastMap<NodeIx, Kind>,
    /// Each placed node's users (nodes whose inputs name it), as built.
    users: Vec<Vec<NodeIx>>,
    /// Each placed node's position in its block's list, phis first.
    position: Vec<u32>,
    /// Nodes to place before a node of a block's body, in order.
    before: FastMap<NodeIx, Vec<NodeIx>>,
    /// Nodes to place before a block's terminator, in order, by block.
    at_end: Vec<Vec<NodeIx>>,
    /// Constants made for the rewrites, placed at the entry block's head.
    constants: Vec<NodeIx>,
    const_i32: FastMap<i32, NodeIx>,
    const_f64: FastMap<u64, NodeIx>,
    /// The nodes the rewrites replace by another (resolved transitively), and those that leave
    /// their blocks.
    replaced: FastMap<NodeIx, NodeIx>,
    removed: Vec<bool>,
    /// A retagging of an untagged phi in a block: (phi, block) -> its word.
    retags: FastMap<(NodeIx, BlockIx), NodeIx>,
    /// The nodes the rewrites made.
    made: Vec<NodeIx>,
    snapshot: &'g Snapshot,
    /// Each loop header's blocks (its natural loop), by header.
    loops: FastMap<BlockIx, Vec<bool>>,
    /// The phis' inputs as the builder made them, for the speculations' states.
    original: FastMap<NodeIx, Vec<NodeIx>>,
    /// The state each loop header's entry edge leaves, at the header's pc: (header, input).
    entry_states: FastMap<(BlockIx, usize), StateIx>,
    /// A speculation's checks at a predecessor's end: (block, word, representation).
    speculated: FastMap<(BlockIx, NodeIx, Rep), NodeIx>,
}

impl<'g> Selector<'g> {
    fn new(graph: &'g mut Graph, notes: &'g PhiNotes, snapshot: &'g Snapshot) -> Self {
        let n = graph.nodes.len();
        let mut users: Vec<Vec<NodeIx>> = vec![Vec::new(); n];
        let mut position = vec![u32::MAX; n];
        let mut const_i32 = FastMap::default();
        let mut const_f64 = FastMap::default();
        for block in &graph.blocks {
            for (at, &node) in block.phis.iter().chain(&block.body).enumerate() {
                position[node.index()] = at as u32;
                for &input in &graph.node(node).inputs {
                    let list = &mut users[input.index()];
                    if list.last() != Some(&node) {
                        list.push(node);
                    }
                }
                match graph.node(node).op {
                    NodeOp::ConstI32(x) if Some(graph.node(node).block) == graph.entry => {
                        const_i32.insert(x, node);
                    }
                    NodeOp::ConstF64(bits) if Some(graph.node(node).block) == graph.entry => {
                        const_f64.insert(bits, node);
                    }
                    _ => {}
                }
            }
        }
        let blocks = graph.blocks.len();
        // Each loop header's natural loop: the blocks that reach one of its back edges (from a
        // block numbered at or after it, the builder's reverse postorder) without passing it.
        let mut loops = FastMap::default();
        for (index, block) in graph.blocks.iter().enumerate() {
            if !block.loop_header {
                continue;
            }
            let mut members = vec![false; blocks];
            members[index] = true;
            let mut work: Vec<BlockIx> = block
                .preds
                .iter()
                .copied()
                .filter(|pred| pred.index() >= index)
                .collect();
            while let Some(member) = work.pop() {
                if std::mem::replace(&mut members[member.index()], true) {
                    continue;
                }
                work.extend(graph.block(member).preds.iter().copied());
            }
            loops.insert(BlockIx(index as u32), members);
        }
        Selector {
            graph,
            notes,
            kinds: FastMap::default(),
            users,
            position,
            before: FastMap::default(),
            at_end: vec![Vec::new(); blocks],
            constants: Vec::new(),
            const_i32,
            const_f64,
            replaced: FastMap::default(),
            removed: vec![false; n],
            retags: FastMap::default(),
            made: Vec::new(),
            snapshot,
            loops,
            original: FastMap::default(),
            entry_states: FastMap::default(),
            speculated: FastMap::default(),
        }
    }

    fn kind(&self, phi: NodeIx) -> Kind {
        self.kinds.get(&phi).copied().unwrap_or(Kind::Undecided)
    }

    /// Where input `k` of `phi`, the word `value`, would take its untagged value from, as the
    /// phis' kinds stand.
    fn source(&self, phi: NodeIx, k: usize, value: NodeIx) -> Source {
        if value == phi {
            return Source::Own;
        }
        let node = self.graph.node(value);
        if node.op == NodeOp::Phi {
            match self.kind(value) {
                Kind::Undecided => return Source::Pending,
                Kind::I32 => return Source::Untagged(value, Rep::I32),
                Kind::F64 => return Source::Untagged(value, Rep::F64),
                // A tagged phi's word may have a form at the predecessor's end, as any word.
                Kind::Tagged => {}
            }
        }
        match node.op {
            NodeOp::I32ToTagged => return Source::Untagged(node.inputs[0], Rep::I32),
            NodeOp::F64ToTagged => return Source::Untagged(node.inputs[0], Rep::F64),
            _ => {}
        }
        let alternatives = self.alternatives(phi, k);
        if let Some(int) = alternatives.i32 {
            return Source::Untagged(int, Rep::I32);
        }
        if let Some(number) = alternatives.f64 {
            return Source::Untagged(number, Rep::F64);
        }
        if let NodeOp::ConstTagged(word) = node.op {
            if let Some(x) = number_word(word) {
                return Source::Number(x);
            }
            return Source::Tagged;
        }
        match self.speculation(phi, k) {
            Some(kind) => Source::Speculative(kind),
            None => Source::Tagged,
        }
    }

    /// Whether input `k` of `phi`, a word without forms, may speculate: the edge enters a loop
    /// header from a predecessor outside the loop whose one successor the header is, and the phi
    /// has a numeric use in its loop (see the module's doc). The vote the header's flags leave
    /// it: none, F64 under PhiNotInt32, Tagged under PhiNotNumber.
    fn speculation(&self, phi: NodeIx, k: usize) -> Option<Kind> {
        let block = self.graph.node(phi).block;
        let members = self.loops.get(&block)?;
        let pred = self.graph.block(block).preds[k];
        if members[pred.index()] || self.graph.block(pred).succs.len() != 1 {
            return None;
        }
        let numeric = self.users[phi.index()].iter().any(|&user| {
            let user = self.graph.node(user);
            user.op == NodeOp::CheckNumber && members[user.block.index()]
        });
        if !numeric {
            return None;
        }
        let pc = self.graph.block(block).first_pc as usize;
        Some(if self.snapshot.declines(pc, ExitReason::PhiNotNumber) {
            Kind::Tagged
        } else if self.snapshot.declines(pc, ExitReason::PhiNotInt32) {
            Kind::F64
        } else {
            Kind::Undecided
        })
    }

    /// List each speculation a flag declined: a loop phi with a speculative input, whose other
    /// inputs would have made it I32 (PhiNotInt32: an f64 phi) or untagged (PhiNotNumber: a tagged
    /// phi), once per header and reason, as the builder lists its declines.
    fn declines(&mut self, phis: &[NodeIx]) {
        for &phi in phis {
            let inputs = self.graph.node(phi).inputs.clone();
            let mut flagged = None;
            let mut others = Kind::Undecided;
            for (k, &value) in inputs.iter().enumerate() {
                match self.source(phi, k, value) {
                    Source::Speculative(Kind::Tagged) => flagged = Some(ExitReason::PhiNotNumber),
                    Source::Speculative(Kind::F64) => {
                        flagged = flagged.or(Some(ExitReason::PhiNotInt32))
                    }
                    Source::Speculative(_) => {}
                    source => others = others.max(source.vote()),
                }
            }
            let (reason, form) = match (flagged, others) {
                (Some(ExitReason::PhiNotNumber), Kind::I32 | Kind::F64) => {
                    (ExitReason::PhiNotNumber, "a tagged phi")
                }
                (Some(ExitReason::PhiNotInt32), Kind::I32) => {
                    (ExitReason::PhiNotInt32, "an f64 phi")
                }
                _ => continue,
            };
            let pc = self.graph.block(self.graph.node(phi).block).first_pc;
            let decline = Decline { pc, reason, form };
            if !self.graph.declines.contains(&decline) {
                self.graph.declines.push(decline);
            }
        }
    }

    fn alternatives(&self, phi: NodeIx, k: usize) -> Alternatives {
        self.notes
            .alternatives
            .get(&(phi, k as u32))
            .copied()
            .unwrap_or_default()
    }

    /// Every phi's kind: their inputs' votes, widened until none changes (see the module's doc).
    fn decide(&mut self, phis: &[NodeIx]) {
        loop {
            let mut changed = false;
            for &phi in phis {
                let inputs = self.graph.node(phi).inputs.clone();
                let mut kind = self.kind(phi);
                for (k, &value) in inputs.iter().enumerate() {
                    kind = kind.max(self.source(phi, k, value).vote());
                }
                if kind != self.kind(phi) {
                    self.kinds.insert(phi, kind);
                    changed = true;
                }
            }
            if !changed {
                // Phis that only undecided phis and themselves feed carry no known number.
                let undecided: Vec<NodeIx> = phis
                    .iter()
                    .copied()
                    .filter(|&phi| self.kind(phi) == Kind::Undecided)
                    .collect();
                if undecided.is_empty() {
                    break;
                }
                for phi in undecided {
                    self.kinds.insert(phi, Kind::Tagged);
                }
            }
        }
    }

    /// A node of `op` made for a rewrite in `block` (placed at the end).
    fn make(&mut self, block: BlockIx, op: NodeOp, inputs: &[NodeIx], rep: Rep, pc: u32) -> NodeIx {
        let node = self.graph.new_node(block, op, inputs, rep, pc);
        self.removed.push(false);
        self.made.push(node);
        node
    }

    /// A node of `op` placed before `block`'s terminator.
    fn at_end(&mut self, block: BlockIx, op: NodeOp, inputs: &[NodeIx], rep: Rep) -> NodeIx {
        let pc = self.graph.block(block).first_pc;
        let node = self.make(block, op, inputs, rep, pc);
        self.at_end[block.index()].push(node);
        node
    }

    /// The int32 constant `x`, made at the entry block's head where it has none.
    fn int32(&mut self, x: i32) -> NodeIx {
        if let Some(&node) = self.const_i32.get(&x) {
            return node;
        }
        let entry = self.graph.entry.expect("a graph with an entry");
        let node = self.make(entry, NodeOp::ConstI32(x), &[], Rep::I32, 0);
        self.constants.push(node);
        self.const_i32.insert(x, node);
        node
    }

    /// The f64 constant `x`, made at the entry block's head where it has none.
    fn float64(&mut self, x: f64) -> NodeIx {
        let bits = x.to_bits();
        if let Some(&node) = self.const_f64.get(&bits) {
            return node;
        }
        let entry = self.graph.entry.expect("a graph with an entry");
        let node = self.make(entry, NodeOp::ConstF64(bits), &[], Rep::F64, 0);
        self.constants.push(node);
        self.const_f64.insert(bits, node);
        node
    }

    /// The inputs of each untagged phi in its representation (see the module's doc).
    fn untag_inputs(&mut self, untagged: &[NodeIx]) {
        // The inputs as built, which a speculation's state takes in place of the header's phis.
        for &phi in untagged {
            let block = self.graph.node(phi).block;
            for &other in &self.graph.block(block).phis {
                if !self.original.contains_key(&other) {
                    self.original
                        .insert(other, self.graph.node(other).inputs.clone());
                }
            }
        }
        for &phi in untagged {
            let kind = self.kind(phi);
            let rep = kind.rep();
            let block = self.graph.node(phi).block;
            let inputs = self.graph.node(phi).inputs.clone();
            let mut new = Vec::with_capacity(inputs.len());
            for (k, &value) in inputs.iter().enumerate() {
                let pred = self.graph.block(block).preds[k];
                let input = match self.source(phi, k, value) {
                    Source::Own => phi,
                    Source::Untagged(node, from) if from == rep => node,
                    Source::Untagged(node, Rep::I32) if rep == Rep::F64 => {
                        // An f64 form noted beside the int32 serves as it is, and a constant
                        // widens at the compile.
                        match (self.alternatives(phi, k).f64, &self.graph.node(node).op) {
                            (Some(number), _) => number,
                            (None, &NodeOp::ConstI32(x)) => self.float64(f64::from(x)),
                            (None, _) => self.at_end(pred, NodeOp::I32ToF64, &[node], Rep::F64),
                        }
                    }
                    Source::Number(x) => match (kind, exact_i32(x)) {
                        (Kind::I32, Some(int)) => self.int32(int),
                        _ => self.float64(x),
                    },
                    Source::Speculative(_) => self.speculate(block, k, value, rep),
                    source => unreachable!("a {kind:?} phi with the input {source:?}"),
                };
                new.push(input);
            }
            let node = self.graph.node_mut(phi);
            node.inputs = new;
            node.rep = rep;
        }
    }

    /// The checks of `word`, input `k` of a phi of loop header `header`, at the end of that
    /// input's predecessor, for a phi of `rep`: CheckNumber [PhiNotNumber], and for an I32 phi
    /// CheckInt32 [PhiNotInt32], exiting at the header's pc with the state the edge leaves.
    fn speculate(&mut self, header: BlockIx, k: usize, word: NodeIx, rep: Rep) -> NodeIx {
        let pred = self.graph.block(header).preds[k];
        let state = self.entry_state(header, k);
        let pc = self.graph.block(header).first_pc;
        let number = match self.speculated.get(&(pred, word, Rep::F64)) {
            Some(&number) => number,
            None => {
                let number = self.at_end(pred, NodeOp::CheckNumber, &[word], Rep::F64);
                let node = self.graph.node_mut(number);
                node.pc = pc;
                node.state = Some(state);
                node.reason = Some(ExitReason::PhiNotNumber);
                self.speculated.insert((pred, word, Rep::F64), number);
                number
            }
        };
        if rep == Rep::F64 {
            return number;
        }
        if let Some(&int) = self.speculated.get(&(pred, word, Rep::I32)) {
            return int;
        }
        let int = self.at_end(pred, NodeOp::CheckInt32, &[number], Rep::I32);
        let node = self.graph.node_mut(int);
        node.pc = pc;
        node.state = Some(state);
        node.reason = Some(ExitReason::PhiNotInt32);
        self.speculated.insert((pred, word, Rep::I32), int);
        int
    }

    /// The frame state loop header `header`'s entry edge `k` leaves, at the header's pc: its
    /// poll's state with each of its phis the input the builder gave it on that edge, a boxing's
    /// operand for a boxing (Maglev's CheckpointedJump frame for a hoisted untagging).
    fn entry_state(&mut self, header: BlockIx, k: usize) -> StateIx {
        if let Some(&state) = self.entry_states.get(&(header, k)) {
            return state;
        }
        let poll = self.graph.block(header).body[0];
        let at = self
            .graph
            .node(poll)
            .state
            .expect("a loop header's poll has a state");
        let mut frame = self.graph.state(at).clone();
        let phis = &self.graph.block(header).phis;
        let edge = |value: NodeIx| -> NodeIx {
            if !phis.contains(&value) {
                return value;
            }
            let input = self.original[&value][k];
            match self.graph.node(input).op {
                NodeOp::I32ToTagged | NodeOp::F64ToTagged => self.graph.node(input).inputs[0],
                _ => input,
            }
        };
        let slots: Vec<SlotSource> = frame
            .slots
            .iter()
            .map(|&source| match source {
                SlotSource::Value(value) => SlotSource::Value(edge(value)),
                other => other,
            })
            .collect();
        let stack: Vec<NodeIx> = frame.stack.iter().map(|&value| edge(value)).collect();
        frame.slots = slots.into_boxed_slice();
        frame.stack = stack.into_boxed_slice();
        let state = self.graph.add_state(frame);
        self.entry_states.insert((header, k), state);
        state
    }

    /// The uses of each untagged phi in its representation (see the module's doc).
    fn rewrite_uses(&mut self, untagged: &[NodeIx]) {
        for &phi in untagged {
            let rep = self.kind(phi).rep();
            let users = self.users[phi.index()].clone();
            // The uses only a word serves, by block, and tagged phis' inputs.
            let mut tagged: Vec<NodeIx> = Vec::new();
            let mut tagged_phis: Vec<NodeIx> = Vec::new();
            for user in users {
                match self.graph.node(user).op {
                    NodeOp::CheckNumber if rep == Rep::I32 => {
                        // The int32's f64, in the check's place; the int32 conversions of that
                        // are the int32 itself.
                        let node = self.graph.node_mut(user);
                        node.op = NodeOp::I32ToF64;
                        node.state = None;
                        node.reason = None;
                        for second in self.users[user.index()].clone() {
                            if matches!(
                                self.graph.node(second).op,
                                NodeOp::CheckInt32 | NodeOp::CheckIndex | NodeOp::TruncateToInt32
                            ) {
                                self.replace(second, phi);
                            }
                        }
                    }
                    NodeOp::CheckNumber => self.replace(user, phi),
                    NodeOp::ToBoolean if rep == Rep::I32 => {
                        let zero = self.int32(0);
                        let node = self.graph.node_mut(user);
                        node.op = NodeOp::I32Cmp(Cond::Ne);
                        node.inputs = vec![phi, zero];
                    }
                    NodeOp::ToBoolean => self.graph.node_mut(user).op = NodeOp::F64ToBoolean,
                    // Each boxes what it stores (a64's `boxed`).
                    NodeOp::StoreSlot(_) | NodeOp::Effect { .. } => {}
                    // An element store's value: its cell, words and index are no word.
                    NodeOp::StoreElement(_) => {}
                    NodeOp::Phi if self.kind(user) != Kind::Tagged => {}
                    NodeOp::Phi => tagged_phis.push(user),
                    _ => tagged.push(user),
                }
            }
            // A retagging per block, before its first use there.
            tagged.sort_by_key(|&user| (self.graph.node(user).block, self.position[user.index()]));
            let op = if rep == Rep::I32 {
                NodeOp::I32ToTagged
            } else {
                NodeOp::F64ToTagged
            };
            for user in tagged {
                let block = self.graph.node(user).block;
                let word = match self.retags.get(&(phi, block)) {
                    Some(&word) => word,
                    None => {
                        let pc = self.graph.node(user).pc;
                        let word = self.make(block, op.clone(), &[phi], Rep::Tagged, pc);
                        self.before.entry(user).or_default().push(word);
                        self.retags.insert((phi, block), word);
                        word
                    }
                };
                for input in &mut self.graph.node_mut(user).inputs {
                    if *input == phi {
                        *input = word;
                    }
                }
            }
            // A tagged phi's input: a retagging at its predecessor's end, or the one of that
            // block's uses, which precedes its end.
            for user in tagged_phis {
                let block = self.graph.node(user).block;
                for k in 0..self.graph.node(user).inputs.len() {
                    if self.graph.node(user).inputs[k] != phi {
                        continue;
                    }
                    let pred = self.graph.block(block).preds[k];
                    let word = match self.retags.get(&(phi, pred)) {
                        Some(&word) => word,
                        None => {
                            let word = self.at_end(pred, op.clone(), &[phi], Rep::Tagged);
                            self.retags.insert((phi, pred), word);
                            word
                        }
                    };
                    self.graph.node_mut(user).inputs[k] = word;
                }
            }
        }
    }

    /// `node` is replaced by `by` everywhere and leaves its block.
    fn replace(&mut self, node: NodeIx, by: NodeIx) {
        self.replaced.insert(node, by);
        self.removed[node.index()] = true;
    }

    /// What `node` is after the rewrites' replacements.
    fn resolve(&self, mut node: NodeIx) -> NodeIx {
        while let Some(&by) = self.replaced.get(&node) {
            node = by;
        }
        node
    }

    /// The replacements made in every input and frame state, the boxings and widenings nothing
    /// uses any more removed, and the blocks' lists rebuilt with the rewrites' nodes.
    fn finish(mut self) {
        if !self.replaced.is_empty() {
            for index in 0..self.graph.nodes.len() {
                for k in 0..self.graph.nodes[index].inputs.len() {
                    let input = self.graph.nodes[index].inputs[k];
                    self.graph.nodes[index].inputs[k] = self.resolve(input);
                }
            }
            for index in 0..self.graph.states.len() {
                let mut slots = self.graph.states[index].slots.to_vec();
                for slot in &mut slots {
                    if let SlotSource::Value(value) = *slot {
                        *slot = SlotSource::Value(self.resolve(value));
                    }
                }
                let stack: Vec<NodeIx> = self.graph.states[index]
                    .stack
                    .iter()
                    .map(|&value| self.resolve(value))
                    .collect();
                let state = &mut self.graph.states[index];
                state.slots = slots.into_boxed_slice();
                state.stack = stack.into_boxed_slice();
            }
        }
        self.remove_dead_conversions();
        let entry = self.graph.entry.expect("a graph with an entry");
        for b in 0..self.graph.blocks.len() {
            let old = std::mem::take(&mut self.graph.blocks[b].body);
            let mut body = Vec::with_capacity(old.len() + self.at_end[b].len());
            if b == entry.index() {
                body.extend(self.constants.iter().copied());
            }
            let last = old.len().saturating_sub(1);
            for (at, node) in old.into_iter().enumerate() {
                if let Some(nodes) = self.before.get(&node) {
                    body.extend(nodes.iter().copied().filter(|n| !self.removed[n.index()]));
                }
                if at == last {
                    body.extend(
                        self.at_end[b]
                            .iter()
                            .copied()
                            .filter(|n| !self.removed[n.index()]),
                    );
                }
                if !self.removed[node.index()] {
                    body.push(node);
                }
            }
            self.graph.blocks[b].body = body;
        }
        compact(self.graph);
    }

    /// Remove the boxings and widenings no node or frame state of a kept node uses: the
    /// builder's boxings of the inputs of phis now untagged, and the int32s' f64s whose every
    /// use was a conversion back.
    fn remove_dead_conversions(&mut self) {
        let n = self.graph.nodes.len();
        let mut kept = vec![false; n];
        for block in &self.graph.blocks {
            for &node in block.phis.iter().chain(&block.body) {
                kept[node.index()] = true;
            }
        }
        for &node in &self.made {
            kept[node.index()] = true;
        }
        let pure = |op: &NodeOp| {
            matches!(
                op,
                NodeOp::I32ToTagged | NodeOp::F64ToTagged | NodeOp::BoolToTagged | NodeOp::I32ToF64
            )
        };
        loop {
            let mut uses = vec![0u32; n];
            for index in 0..n {
                if !kept[index] || self.removed[index] {
                    continue;
                }
                let node = &self.graph.nodes[index];
                for &input in &node.inputs {
                    uses[input.index()] += 1;
                }
                if let Some(state) = node.state {
                    let state = self.graph.state(state);
                    for source in state.slots.iter() {
                        if let SlotSource::Value(value) = *source {
                            uses[value.index()] += 1;
                        }
                    }
                    for &value in state.stack.iter() {
                        uses[value.index()] += 1;
                    }
                }
            }
            let mut changed = false;
            for index in 0..n {
                if kept[index]
                    && !self.removed[index]
                    && uses[index] == 0
                    && pure(&self.graph.nodes[index].op)
                {
                    self.removed[index] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
}

/// `graph`'s nodes renumbered to those its blocks hold, and its states to those its nodes name:
/// the replaced and dead nodes leave the graph, which keeps every node in exactly one block.
fn compact(graph: &mut Graph) {
    let n = graph.nodes.len();
    let mut number = vec![u32::MAX; n];
    for block in &graph.blocks {
        for &node in block.phis.iter().chain(&block.body) {
            number[node.index()] = 0;
        }
    }
    let mut kept = 0u32;
    for slot in &mut number {
        if *slot == 0 {
            *slot = kept;
            kept += 1;
        }
    }
    if kept as usize == n {
        return;
    }
    // A removed node's index stays out of range, which the verifier refuses where it is named.
    let map = |node: NodeIx| NodeIx(number[node.index()]);
    let mut old_states: Vec<Option<super::ir::FrameState>> = std::mem::take(&mut graph.states)
        .into_iter()
        .map(Some)
        .collect();
    let mut state_number = vec![u32::MAX; old_states.len()];
    let mut states = Vec::new();
    let old_nodes = std::mem::take(&mut graph.nodes);
    let mut nodes = Vec::with_capacity(kept as usize);
    for (index, mut node) in old_nodes.into_iter().enumerate() {
        if number[index] == u32::MAX {
            continue;
        }
        for input in &mut node.inputs {
            *input = map(*input);
        }
        if let Some(state) = node.state {
            if state_number[state.index()] == u32::MAX {
                let mut frame = old_states[state.index()].take().expect("each state once");
                let slots: Vec<SlotSource> = frame
                    .slots
                    .iter()
                    .map(|&source| match source {
                        SlotSource::Value(value) => SlotSource::Value(map(value)),
                        other => other,
                    })
                    .collect();
                frame.slots = slots.into_boxed_slice();
                let stack: Vec<NodeIx> = frame.stack.iter().map(|&value| map(value)).collect();
                frame.stack = stack.into_boxed_slice();
                state_number[state.index()] = states.len() as u32;
                states.push(frame);
            }
            node.state = Some(StateIx(state_number[state.index()]));
        }
        nodes.push(node);
    }
    graph.nodes = nodes;
    graph.states = states;
    for block in &mut graph.blocks {
        for node in block.phis.iter_mut().chain(block.body.iter_mut()) {
            *node = map(*node);
        }
    }
}
