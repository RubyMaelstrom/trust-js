//! The optimizing tier's register allocator (P3 stage 1; the plan's Design point 10 and the
//! "Stage 1" record's back end): Maglev's single forward pass over the scheduled blocks.
//!
//! Live ranges and next uses come first: a dataflow liveness over the graph, frame states'
//! values counting as uses of the nodes that carry them, and each value's range the hull of the
//! positions where it is live in code order. One pass in code order then gives each value its
//! home at its definition: a free register of its class; a callee-saved one (x23-x28, d8-d15) or
//! a spill slot when the value lives across an effect or a call, which clobber every
//! caller-saved register (contracts 4.4: helpers and templates preserve x19-x28 and d8-d15, and
//! no baseline template touches x23-x28 or writes d8-d15); none of the registers a shared stub's
//! call clobbers when it lives across a node whose own path makes that call (`clobbers`: P3
//! stage 3 step 0b, the exemplars' clobber sets). When no register is free, the held
//! value whose next use is the farthest gives up its register and is spilled at its definition,
//! or the new value is when its own next use is farther (Maglev's choice). A home is the value's
//! for its whole range, so an exit's recipe names one place per value; registers and slots free
//! again where ranges end. Phis take their inputs by parallel moves on the edges into their blocks
//! (`a64`). Code order is the builder's, a loop's body right after its header and its exit code
//! after the body, and a loop phi's register is the hint of the values its back edges carry,
//! which take it where it is free and so need no move (stage 2 step 6; Maglev's bytecode order
//! and SetLoopPhiRegisterHint). Constants have no home: each use takes them as its instruction's
//! immediate where the encoding allows (stage 2 step 7) and rematerializes them otherwise, and a
//! recipe records them as constants.
//!
//! Raw cell pointers and tagged words in callee-saved registers and spill slots are roots: the
//! collector scans the native stack and the callee-saved registers conservatively and resolves
//! interior pointers (contracts 4.4; JSC's design point), so the homes need no maps.

use super::ir::{BlockIx, Graph, NodeIx, NodeOp, Refusal, Rep, StateIx};

/// Where a value lives for its whole range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Home {
    /// No value (`Rep::None`), or a value nothing reads.
    None,
    /// A constant node: rematerialized at each use.
    Const,
    /// An x register (Tagged, I32, Bool, Cell).
    Gp(u8),
    /// A d register (F64).
    Fp(u8),
    /// A spill slot: its byte offset from the frame area's base (`sp` between nodes).
    Spill(u32),
}

/// A register class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    Gp,
    Fp,
}

/// The class a value of `rep` lives in, if it has one.
pub(crate) fn class_of(rep: Rep) -> Option<Class> {
    match rep {
        Rep::Tagged | Rep::I32 | Rep::Bool | Rep::Cell => Some(Class::Gp),
        // An address (P3 stage 2 step 2): in an x register like a cell, and a root only through
        // the owner's cell, which the node that reads the address also takes (Rep::Ptr).
        Rep::Ptr => Some(Class::Gp),
        // An environment (stage 2 step 4) in an x register like a cell: a record the running
        // activation's chain keeps alive, which the collector does not move (Rep::Env).
        Rep::Env => Some(Class::Gp),
        Rep::F64 => Some(Class::Fp),
        Rep::None => None,
    }
}

/// The caller-saved registers that a node which is no effect clobbers by a call on its own path,
/// as masks of x and d registers (a shared stub's, whose contract names what it writes); none
/// for every other node. A value live across such a node takes no home among them, so that the
/// call keeps every value it does not clobber where it is, as Maglev frees a node's specific
/// temporaries before it (AssignFixedTemporaries, maglev/maglev-regalloc.cc:2017-2046, of
/// RequireSpecificTemporary, maglev-ir.h:2443-2452), Ion's fixed temps cover their instruction
/// (BacktrackingAllocator.cpp:1915-1928) and Air's patchpoint clobbers are defs at theirs
/// (Inst::forEachDefWithExtraClobberedRegs, b3/air/AirInstInlines.h:100-121). P3 stage 3 step
/// 0b: a TypedElement call that saved and restored every live caller-saved register was 13% of
/// zlib's instructions.
pub(crate) fn clobbers(op: &NodeOp) -> (u64, u64) {
    match op {
        // The TypedElement stub writes x9, x12-x17, d1, d2 and x30 (jit_typed_array.rs), and
        // its call's arguments x11 (the Object), w12 (the kind tag) and d1 (a store's value).
        // Of the registers the allocator hands out: x11-x15, d1 and d2.
        NodeOp::TypedElement { .. } => (TYPED_ELEMENT_GP, TYPED_ELEMENT_FP),
        _ => (0, 0),
    }
}

/// x11, x12, x13, x14 and x15.
const TYPED_ELEMENT_GP: u64 = (1 << 11) | (1 << 12) | (1 << 13) | (1 << 14) | (1 << 15);
/// d1 and d2.
const TYPED_ELEMENT_FP: u64 = (1 << 1) | (1 << 2);

/// Whether a node is a constant, which no register holds.
pub(crate) fn rematerializable(op: &NodeOp) -> bool {
    matches!(
        op,
        NodeOp::ConstTagged(_)
            | NodeOp::ConstF64(_)
            | NodeOp::ConstI32(_)
            | NodeOp::ConstBool(_)
            | NodeOp::ConstEnv(_)
    )
}

/// The registers the allocator hands out, by class and by whether calls preserve them.
pub(crate) struct RegisterFile {
    pub(crate) gp_caller: &'static [u8],
    pub(crate) gp_callee: &'static [u8],
    pub(crate) fp_caller: &'static [u8],
    pub(crate) fp_callee: &'static [u8],
}

/// ARM64 (the Stage 1 record's back end): x19 the context, x21 the helper table and x22 the
/// slots stay pinned, as effects need them; x20, the operand top, is set before each effect and
/// allocated to nothing; x9, x16, x17 and x30 are the code generator's scratch registers (x16 and
/// x17 the veneers' too), x18 is unused and x29 keeps the caller's frame pointer. d30 and d31 are
/// the code generator's FP scratch registers, which a spilled or constant F64 input needs.
pub(crate) const ARM64: RegisterFile = RegisterFile {
    gp_caller: &[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15],
    gp_callee: &[23, 24, 25, 26, 27, 28],
    fp_caller: &[
        0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    ],
    fp_callee: &[8, 9, 10, 11, 12, 13, 14, 15],
};

/// A graph's registers and code order.
#[derive(Debug)]
pub(crate) struct Allocation {
    /// The reachable blocks in code order: reverse postorder from the entry, so every block's
    /// dominators precede it.
    pub(crate) order: Vec<BlockIx>,
    /// Each node's home.
    pub(crate) homes: Vec<Home>,
    /// Each scheduled node's position: its inputs (and its state) are read at `pos`, its value
    /// is defined at `pos + 1`. A phi's position is its block's head.
    pub(crate) pos: Vec<u32>,
    /// Each value's live range `[start, end]`, positions in code order (`start` its definition).
    pub(crate) ranges: Vec<(u32, u32)>,
    /// The spill area's bytes.
    pub(crate) spill_bytes: u32,
    /// The callee-saved x registers some home uses, ascending (the frame saves them).
    pub(crate) callee_saved_gp: Vec<u8>,
    /// Whether a value living across an effect has a d8..d15 home (`a64` then checks that no
    /// shared stub writes those registers).
    pub(crate) fp_callee_across_effect: bool,
    /// The loop-level caches of spilled values (P3 stage 3 step R1, `loop_caches`).
    pub(crate) caches: Vec<Cache>,
    /// Each block's caches, indices into `caches` (empty, or one list per block of the graph).
    pub(crate) block_caches: Vec<Vec<u16>>,
    /// The read spans of spilled values (step R1, `read_spans`), by block and first read.
    pub(crate) spans: Vec<Span>,
    /// Each block's spans, indices into `spans` by first read (empty, or one list per block).
    pub(crate) block_spans: Vec<Vec<u16>>,
    /// `live_at`'s index of the ranges, made at its first query.
    live_index: std::cell::OnceCell<LiveIndex>,
}

/// A register that holds a copy of a spilled value through one loop (P3 stage 3 step R1,
/// `loop_caches`): loaded at the loop's header, read by the loop's blocks, and nothing else's
/// through the loop's hull. The value's spill slot stays its home.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cache {
    /// The value, whose home is a spill slot.
    pub(crate) value: NodeIx,
    /// The register: `Home::Gp` or `Home::Fp`.
    pub(crate) reg: Home,
    /// The loop's header, at whose start the register is loaded.
    pub(crate) header: BlockIx,
    /// The loop's hull: its header's head position to the last tail of its blocks.
    pub(crate) hull: (u32, u32),
}

/// A register that holds a copy of a spilled value over a run of its reads in one block with no
/// call between them (P3 stage 3 step R1, `read_spans`): loaded before its first reading node,
/// read by the nodes of the run, and nothing else's from its first read to its last. The value's
/// spill slot stays its home.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    /// The value, whose home is a spill slot.
    pub(crate) value: NodeIx,
    /// The register: `Home::Gp` or `Home::Fp`.
    pub(crate) reg: Home,
    /// The block of its reads.
    pub(crate) block: BlockIx,
    /// The positions of its first and its last reading node.
    pub(crate) from: u32,
    pub(crate) to: u32,
}

impl Allocation {
    /// The values live where `node` reads its inputs and its state and still needed there or
    /// after it, which `node` did not define: what a call made at `node` must keep. Ascending;
    /// each slow-path call's query reads the index of the ranges (`LiveIndex`), made at the
    /// first, in place of a scan of every value's.
    pub(crate) fn live_at(&self, node: NodeIx) -> Vec<NodeIx> {
        let at = self.pos[node.index()];
        self.live_index
            .get_or_init(|| LiveIndex::new(&self.homes, &self.ranges))
            .live_at(at)
    }

    /// The caches valid in `block` (step R1): its loops' caches.
    pub(crate) fn caches_in(&self, block: BlockIx) -> impl Iterator<Item = &Cache> + '_ {
        self.block_caches
            .get(block.index())
            .into_iter()
            .flatten()
            .map(|&k| &self.caches[k as usize])
    }

    /// The register that holds a copy of spilled value `v` in `block`, if a cache does.
    pub(crate) fn cache_of(&self, block: BlockIx, v: NodeIx) -> Option<Home> {
        self.caches_in(block).find(|c| c.value == v).map(|c| c.reg)
    }

    /// The read spans of `block` (step R1), by first read.
    pub(crate) fn spans_in(&self, block: BlockIx) -> impl Iterator<Item = &Span> + '_ {
        self.block_spans
            .get(block.index())
            .into_iter()
            .flatten()
            .map(|&k| &self.spans[k as usize])
    }

    /// Whether the edge from `from` into `to` is an edge of `to`'s loop, which enters the header
    /// after its caches' loads (step R1): `from` is one of the loop's blocks.
    pub(crate) fn enters_loaded(&self, from: BlockIx, to: BlockIx) -> bool {
        self.caches_in(from).any(|c| c.header == to)
    }

    /// [`Allocation::live_at`] by a scan of every value's range: the index's reference.
    #[cfg(test)]
    pub(crate) fn live_at_scan(&self, node: NodeIx) -> Vec<NodeIx> {
        let at = self.pos[node.index()];
        (0..self.homes.len())
            .filter(|&v| {
                let (start, end) = self.ranges[v];
                !matches!(self.homes[v], Home::None | Home::Const) && start < at && end >= at
            })
            .map(|v| NodeIx(v as u32))
            .collect()
    }
}

/// The ranges of the values that have a home, for [`Allocation::live_at`]: by start, with a
/// tree over their ends (each node the latest end of the ranges below it), so that a query
/// visits only the subtrees holding a range that reaches its position.
#[derive(Debug)]
pub(crate) struct LiveIndex {
    /// (start, end, value), by start.
    ranges: Vec<(u32, u32, u32)>,
    /// The tree: the leaves, `ranges`' ends, at `[leaves, 2 * leaves)`.
    ends: Vec<u32>,
    leaves: usize,
}

impl LiveIndex {
    fn new(homes: &[Home], ranges: &[(u32, u32)]) -> Self {
        let mut held: Vec<(u32, u32, u32)> = homes
            .iter()
            .enumerate()
            .filter(|(_, home)| !matches!(home, Home::None | Home::Const))
            .map(|(v, _)| (ranges[v].0, ranges[v].1, v as u32))
            .collect();
        held.sort_unstable();
        let leaves = held.len().next_power_of_two();
        let mut ends = vec![0; 2 * leaves];
        for (k, &(_, end, _)) in held.iter().enumerate() {
            ends[leaves + k] = end;
        }
        for k in (1..leaves).rev() {
            ends[k] = ends[2 * k].max(ends[2 * k + 1]);
        }
        LiveIndex {
            ranges: held,
            ends,
            leaves,
        }
    }

    /// The values whose range has start < `at` <= end, ascending.
    fn live_at(&self, at: u32) -> Vec<NodeIx> {
        // The ranges that start before `at` are a prefix; of those, the ones that end at or
        // after it.
        let before = self.ranges.partition_point(|&(start, _, _)| start < at);
        let mut live = Vec::new();
        let mut pending = vec![(1usize, 0usize, self.leaves)];
        while let Some((node, first, width)) = pending.pop() {
            if first >= before || self.ends[node] < at {
                continue;
            }
            if width == 1 {
                live.push(NodeIx(self.ranges[first].2));
                continue;
            }
            let half = width / 2;
            pending.push((2 * node + 1, first + half, half));
            pending.push((2 * node, first, half));
        }
        live.sort_unstable();
        live
    }
}

/// The values each node reads where it runs (stage 2, step 10): its inputs (a phi's are read on
/// its edges) and the values of its frame state, which its exit writes, but constants and nodes
/// of no representation, which no home holds. Each state's values are listed once, its operand
/// stack's and then its slots', not walked again at each of the allocator's passes; a state of an
/// inlined frame lists its chain's after them (stage 3, `Graph::chain_values`), which its exit
/// writes as well, so that every value of a parent's recipe keeps its one home at each exit
/// through the frame.
struct Reads<'g> {
    graph: &'g Graph,
    value: Vec<bool>,
    /// Each state's first entry in `state_values`, and one past the last state's.
    state_start: Vec<u32>,
    state_values: Vec<u32>,
}

impl<'g> Reads<'g> {
    fn new(graph: &'g Graph) -> Self {
        let value: Vec<bool> = graph
            .nodes
            .iter()
            .map(|node| node.rep != Rep::None && !rematerializable(&node.op))
            .collect();
        let mut state_start = Vec::with_capacity(graph.states.len() + 1);
        let mut state_values = Vec::new();
        state_start.push(0);
        for k in 0..graph.states.len() {
            graph.for_each_exit_value(StateIx(k as u32), |v| {
                if value[v.index()] {
                    state_values.push(v.0);
                }
            });
            state_start.push(state_values.len() as u32);
        }
        Reads {
            graph,
            value,
            state_start,
            state_values,
        }
    }

    /// Whether `v` is a value some home holds.
    fn is_value(&self, v: NodeIx) -> bool {
        self.value[v.index()]
    }

    /// Each value `node` reads.
    fn each(&self, node: NodeIx, mut f: impl FnMut(NodeIx)) {
        let n = self.graph.node(node);
        if n.op != NodeOp::Phi {
            for &input in &n.inputs {
                if self.value[input.index()] {
                    f(input);
                }
            }
        }
        if let Some(state) = n.state {
            let (from, to) = (
                self.state_start[state.index()] as usize,
                self.state_start[state.index() + 1] as usize,
            );
            for &v in &self.state_values[from..to] {
                f(NodeIx(v));
            }
        }
    }
}

/// The values each edge out of a reachable block carries into its successor's phis (values only),
/// in the successor's phi order, listed once with the edges' checks.
struct EdgeInputs {
    /// Each reachable block's first edge in `start`.
    first: Vec<u32>,
    /// Each edge's first entry in `values`, and one past the last edge's.
    start: Vec<u32>,
    values: Vec<u32>,
}

impl EdgeInputs {
    fn new(graph: &Graph, order: &[BlockIx], reads: &Reads) -> Result<Self, Refusal> {
        let mut edges = EdgeInputs {
            first: vec![0; graph.blocks.len()],
            start: Vec::new(),
            values: Vec::new(),
        };
        for &pred in order {
            edges.first[pred.index()] = edges.start.len() as u32;
            for &succ in &graph.block(pred).succs {
                edges.start.push(edges.values.len() as u32);
                let data = graph.block(succ);
                let mut found = data.preds.iter().enumerate().filter(|&(_, &p)| p == pred);
                let Some((index, _)) = found.next() else {
                    return Err(Refusal::whole("an edge its successor does not list"));
                };
                if found.next().is_some() && !data.phis.is_empty() {
                    return Err(Refusal::whole("two edges from one block into a merge"));
                }
                for &phi in &data.phis {
                    let Some(&input) = graph.node(phi).inputs.get(index) else {
                        return Err(Refusal::whole(
                            "a phi without an input for each predecessor",
                        ));
                    };
                    if reads.is_value(input) {
                        edges.values.push(input.0);
                    }
                }
            }
        }
        edges.start.push(edges.values.len() as u32);
        Ok(edges)
    }

    /// The values the edge from `pred` to its `k`th successor carries.
    fn of(&self, pred: BlockIx, k: usize) -> impl Iterator<Item = NodeIx> + '_ {
        let edge = self.first[pred.index()] as usize + k;
        self.values[self.start[edge] as usize..self.start[edge + 1] as usize]
            .iter()
            .map(|&v| NodeIx(v))
    }
}

/// Add node index `i` to a flat bit set's words.
fn insert(bits: &mut [u64], i: usize) {
    bits[i / 64] |= 1 << (i % 64);
}

/// Take node index `i` out of a flat bit set's words.
fn remove(bits: &mut [u64], i: usize) {
    bits[i / 64] &= !(1 << (i % 64));
}

/// Each index in `bits`, ascending.
fn for_each_bit(bits: &[u64], mut f: impl FnMut(usize)) {
    for (k, &word) in bits.iter().enumerate() {
        let mut rest = word;
        while rest != 0 {
            f(k * 64 + rest.trailing_zeros() as usize);
            rest &= rest - 1;
        }
    }
}

/// Reverse postorder from the entry over `succs`, the last successor first so that the first
/// comes first: the builder's block order (`build::in_reverse_postorder`), in which a loop's body
/// follows its header (its condition's then-block) and the loop's exit code follows the body, as
/// Maglev allocates in bytecode order (P3 stage 2 step 6). Visiting the first successor first
/// placed the body after the exit code, so that a loop phi's hull spanned that code, effects
/// and all.
fn reverse_postorder(graph: &Graph, entry: BlockIx) -> Vec<BlockIx> {
    let mut seen = vec![false; graph.blocks.len()];
    let mut post = Vec::with_capacity(graph.blocks.len());
    let mut stack = vec![(entry, 0usize)];
    seen[entry.index()] = true;
    while let Some(&mut (block, ref mut next)) = stack.last_mut() {
        let succs = &graph.block(block).succs;
        if *next < succs.len() {
            let succ = succs[succs.len() - 1 - *next];
            *next += 1;
            if !seen[succ.index()] {
                seen[succ.index()] = true;
                stack.push((succ, 0));
            }
        } else {
            post.push(block);
            stack.pop();
        }
    }
    post.reverse();
    post
}

/// The free registers of one class during the pass, as bit masks (Maglev's register state keeps
/// its free registers as a RegList, maglev/maglev-regalloc.h:57-94).
struct Pool {
    caller: u64,
    callee: u64,
}

impl Pool {
    fn of(caller: &[u8], callee: &[u8]) -> Self {
        Pool {
            caller: mask(caller),
            callee: mask(callee),
        }
    }

    /// The lowest-numbered free register of `free`, for stable code.
    fn take(free: &mut u64) -> Option<u8> {
        if *free == 0 {
            return None;
        }
        let r = free.trailing_zeros() as u8;
        *free &= !(1 << r);
        Some(r)
    }

    /// The lowest-numbered free register of `free` that `avoid` does not hold.
    fn take_but(free: &mut u64, avoid: u64) -> Option<u8> {
        let open = *free & !avoid;
        if open == 0 {
            return None;
        }
        let r = open.trailing_zeros() as u8;
        *free &= !(1 << r);
        Some(r)
    }

    /// Register `r` of `free`, where it is free.
    fn take_this(free: &mut u64, r: u8) -> Option<u8> {
        let bit = 1 << r;
        (*free & bit != 0).then(|| {
            *free &= !bit;
            r
        })
    }
}

/// The registers of `list` as a mask.
fn mask(list: &[u8]) -> u64 {
    list.iter().fold(0, |mask, &r| mask | (1 << r))
}

/// Allocate `graph`'s values to `file`'s registers and spill slots.
pub(crate) fn allocate(graph: &Graph, file: &RegisterFile) -> Result<Allocation, Refusal> {
    let entry = graph
        .entry
        .ok_or(Refusal::whole("a graph without an entry block"))?;
    let order = reverse_postorder(graph, entry);
    let n = graph.nodes.len();

    // Positions: a block's head (its phis' definitions at head + 1), then two per body node.
    let mut pos = vec![u32::MAX; n];
    let mut head = vec![u32::MAX; graph.blocks.len()];
    let mut tail = vec![u32::MAX; graph.blocks.len()];
    let mut next = 0u32;
    for &b in &order {
        head[b.index()] = next;
        for &phi in &graph.block(b).phis {
            pos[phi.index()] = next;
        }
        next += 2;
        for &node in &graph.block(b).body {
            pos[node.index()] = next;
            next += 2;
        }
        tail[b.index()] = next - 1;
    }

    // What each node and each edge reads, listed once for the passes below.
    let reads = Reads::new(graph);
    let edges = EdgeInputs::new(graph, &order, &reads)?;
    // The values a generic node's throw exit reads after its call, at its position + 1 (P3 stage
    // 3 step 3a, `Graph::throw_recipe`): each crosses the call, and so takes a home the call
    // preserves, as Maglev allocates a throwing call's lazy-deopt inputs, which its catch block
    // reuses, after the call cleared every register (maglev-regalloc.cc:767, 786-806).
    let mut after_call: Vec<(NodeIx, NodeIx)> = Vec::new();
    for &b in &order {
        for &node in &graph.block(b).body {
            if !matches!(graph.node(node).op, NodeOp::Generic { .. }) {
                continue;
            }
            let (slots, stack) = graph.throw_recipe(node);
            // An inlined frame's node whose chain has a handler (P3 stage 3 step 3c): its throw
            // exit rebuilds the frames above, from their callees, receivers and parent states.
            let data = graph.node(node);
            let chain = match data.state {
                Some(state) if !data.frame.is_compiled() && graph.chain_has_handlers(state) => {
                    graph.chain_values(data.frame)
                }
                _ => Vec::new(),
            };
            for v in slots
                .into_iter()
                .map(|(_, v)| v)
                .chain(stack.into_iter().flatten())
                .chain(chain)
            {
                if reads.is_value(v) && !after_call.contains(&(node, v)) {
                    after_call.push((node, v));
                }
            }
        }
    }
    // A by-literal frame's callee (P3 stage 3, piece B) has a home across every node of its frame
    // and of the frames nested in it, read again just after the last of them in code order: a
    // home is a value's for its whole range, so the closure is in a callee-saved register or a
    // spill slot across each of the frame's effectful nodes and among the values a slow path's
    // call saves at each other (`a64`'s `save_live_caller_saved`), where the collector scans,
    // while a helper the frame runs collects and then reads the stack, whose by-literal site
    // names the closure through the activation's frame-callee table (`Interp::frame_callee`).
    // The table is the way to find the closure, never what keeps it. JSC keeps a closure call's
    // callee in the inline frame's callee slot of the machine frame for the frame's extent
    // (DFGByteCodeParser.cpp:2213-2215, flushed at :934), and Maglev's lazy deopt frame state of
    // each call in an inlined frame holds the closure.
    let mut extents: Vec<(NodeIx, NodeIx)> = Vec::new();
    for &b in &order {
        for &node in &graph.block(b).body {
            let mut frame = graph.node(node).frame;
            while let Some(inlined) = graph.inlined(frame) {
                if inlined.by_literal && reads.is_value(inlined.callee) {
                    match extents
                        .iter_mut()
                        .find(|(callee, _)| *callee == inlined.callee)
                    {
                        Some(extent) => extent.1 = node,
                        None => extents.push((inlined.callee, node)),
                    }
                }
                frame = graph.state(inlined.parent).frame;
            }
        }
    }
    for (callee, last) in extents {
        if !after_call.contains(&(last, callee)) {
            after_call.push((last, callee));
        }
    }

    // Liveness over the blocks (values that are not constants), to a fixed point, in flat bit
    // sets of `words` words a block, the scratch sets made once.
    let words = n.div_ceil(64);
    // A block's set in `live_in` and `live_out`.
    let set = |b: BlockIx| b.index() * words..(b.index() + 1) * words;
    let mut live_in = vec![0u64; graph.blocks.len() * words];
    let mut live_out = vec![0u64; graph.blocks.len() * words];
    let mut live = vec![0u64; words];
    let mut from = vec![0u64; words];
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().rev() {
            let data = graph.block(b);
            live.fill(0);
            for (k, &s) in data.succs.iter().enumerate() {
                from.copy_from_slice(&live_in[set(s)]);
                for &phi in &graph.block(s).phis {
                    remove(&mut from, phi.index());
                }
                for (word, bits) in live.iter_mut().zip(&from) {
                    *word |= bits;
                }
                for input in edges.of(b, k) {
                    insert(&mut live, input.index());
                }
            }
            live_out[set(b)].copy_from_slice(&live);
            for &node in data.body.iter().rev() {
                remove(&mut live, node.index());
                reads.each(node, |read| insert(&mut live, read.index()));
            }
            for &phi in &data.phis {
                remove(&mut live, phi.index());
            }
            if live[..] != live_in[set(b)] {
                live_in[set(b)].copy_from_slice(&live);
                changed = true;
            }
        }
    }
    if live_in[set(entry)].iter().any(|&word| word != 0) {
        return Err(Refusal::whole("a value read before its definition"));
    }

    // Ranges: the hull of each value's definition, its reads and the blocks it lives out of.
    let mut ranges = vec![(u32::MAX, 0u32); n];
    for &b in &order {
        let data = graph.block(b);
        for &phi in &data.phis {
            ranges[phi.index()] = (head[b.index()] + 1, head[b.index()] + 1);
        }
        for &node in &data.body {
            let p = pos[node.index()];
            ranges[node.index()] = (p + 1, p + 1);
        }
    }
    for &b in &order {
        let data = graph.block(b);
        for &node in &data.body {
            let at = pos[node.index()];
            let mut unplaced = false;
            reads.each(node, |read| {
                unplaced |= pos[read.index()] == u32::MAX;
                let range = &mut ranges[read.index()];
                range.1 = range.1.max(at);
            });
            if unplaced {
                return Err(Refusal::whole("a value defined in no reachable block"));
            }
        }
        let end = tail[b.index()];
        for_each_bit(&live_out[set(b)], |v| {
            let range = &mut ranges[v];
            range.1 = range.1.max(end);
        });
    }
    for &(node, v) in &after_call {
        let range = &mut ranges[v.index()];
        range.1 = range.1.max(pos[node.index()] + 1);
    }

    // Next uses: each value's reads in code order (a phi's input is read at its edge's block end),
    // flat: counted, placed, then sorted and deduplicated in each value's run. Made at the forward
    // pass's first choice of a value to spill, their only reader: never in a compile whose values
    // all find a register (P3 stage 3 step R1, F5; about a fifth of the pass where they are made).
    let make_uses = || {
        let mut use_start = vec![0u32; n + 1];
        for &b in &order {
            let data = graph.block(b);
            for &node in &data.body {
                reads.each(node, |read| use_start[read.index() + 1] += 1);
            }
            for k in 0..data.succs.len() {
                for input in edges.of(b, k) {
                    use_start[input.index() + 1] += 1;
                }
            }
        }
        for &(_, v) in &after_call {
            use_start[v.index() + 1] += 1;
        }
        for v in 0..n {
            use_start[v + 1] += use_start[v];
        }
        let mut use_end: Vec<u32> = use_start[..n].to_vec();
        let mut uses = vec![0u32; use_start[n] as usize];
        for &b in &order {
            let data = graph.block(b);
            for &node in &data.body {
                let at = pos[node.index()];
                reads.each(node, |read| {
                    uses[use_end[read.index()] as usize] = at;
                    use_end[read.index()] += 1;
                });
            }
            for k in 0..data.succs.len() {
                for input in edges.of(b, k) {
                    uses[use_end[input.index()] as usize] = tail[b.index()];
                    use_end[input.index()] += 1;
                }
            }
        }
        for &(node, v) in &after_call {
            uses[use_end[v.index()] as usize] = pos[node.index()] + 1;
            use_end[v.index()] += 1;
        }
        for v in 0..n {
            let run = &mut uses[use_start[v] as usize..use_end[v] as usize];
            run.sort_unstable();
            let mut kept = 0;
            for k in 0..run.len() {
                if k == 0 || run[k] != run[kept - 1] {
                    run[kept] = run[k];
                    kept += 1;
                }
            }
            use_end[v] = use_start[v] + kept as u32;
        }
        (use_start, use_end, uses)
    };
    let next_uses = std::cell::OnceCell::new();
    // The first read of `v` after `at`, or the end of its range (a read in a loop's next turn).
    let next_use = |v: NodeIx, at: u32| {
        let (use_start, use_end, uses) = next_uses.get_or_init(make_uses);
        let list = &uses[use_start[v.index()] as usize..use_end[v.index()] as usize];
        list.get(list.partition_point(|&p| p <= at))
            .copied()
            .unwrap_or(ranges[v.index()].1)
    };

    // The positions of effects and calls, which clobber the caller-saved registers.
    let mut calls: Vec<u32> = order
        .iter()
        .flat_map(|&b| graph.block(b).body.iter())
        .filter(|&&node| graph.node(node).op.is_effectful())
        .map(|&node| pos[node.index()])
        .collect();
    calls.sort_unstable();
    let crosses_call = |(start, end): (u32, u32)| {
        // A call at `p` with start < p < end: the value is defined before it and read after it.
        let first = calls.partition_point(|&p| p <= start);
        calls.get(first).is_some_and(|&p| p < end)
    };
    // The positions of the nodes whose own calls clobber a few caller-saved registers
    // (`clobbers`), with those registers: a value live across one takes no home among them.
    let mut stub_calls: Vec<(u32, u64, u64)> = order
        .iter()
        .flat_map(|&b| graph.block(b).body.iter())
        .filter_map(|&node| {
            let (gp, fp) = clobbers(&graph.node(node).op);
            (gp | fp != 0).then(|| (pos[node.index()], gp, fp))
        })
        .collect();
    stub_calls.sort_unstable();
    let all_clobbered = stub_calls
        .iter()
        .fold((0, 0), |(gp, fp), &(_, g, f)| (gp | g, fp | f));
    let clobbered = |(start, end): (u32, u32), class: Class| {
        // The calls at `p` with start < p < end, as for `crosses_call`.
        let all = match class {
            Class::Gp => all_clobbered.0,
            Class::Fp => all_clobbered.1,
        };
        let first = stub_calls.partition_point(|&(p, _, _)| p <= start);
        let mut avoid = 0u64;
        for &(p, gp, fp) in &stub_calls[first..] {
            if p >= end || avoid == all {
                break;
            }
            avoid |= match class {
                Class::Gp => gp,
                Class::Fp => fp,
            };
        }
        avoid
    };

    // The pass: a home at each definition, freed where the range ends.
    let mut homes = vec![Home::None; n];
    let mut gp = Pool::of(file.gp_caller, file.gp_callee);
    let mut fp = Pool::of(file.fp_caller, file.fp_callee);
    let mut free_slots: Vec<u32> = Vec::new();
    let mut spill_bytes = 0u32;
    let mut used_callee_gp: Vec<u8> = Vec::new();
    let mut fp_callee_across_effect = false;
    // (range end, value) of the values that hold a register or a slot.
    let mut active: Vec<(u32, NodeIx)> = Vec::new();
    let is_callee = |file: &RegisterFile, class: Class, r: u8| match class {
        Class::Gp => file.gp_callee.contains(&r),
        Class::Fp => file.fp_callee.contains(&r),
    };
    let release = |home: Home, gp: &mut Pool, fp: &mut Pool, free_slots: &mut Vec<u32>| match home {
        Home::Gp(r) if file.gp_callee.contains(&r) => gp.callee |= 1 << r,
        Home::Gp(r) => gp.caller |= 1 << r,
        Home::Fp(r) if file.fp_callee.contains(&r) => fp.callee |= 1 << r,
        Home::Fp(r) => fp.caller |= 1 << r,
        Home::Spill(offset) => free_slots.push(offset),
        Home::None | Home::Const => {}
    };
    let expire = |active: &mut Vec<(u32, NodeIx)>,
                  upto: u32,
                  homes: &Vec<Home>,
                  gp: &mut Pool,
                  fp: &mut Pool,
                  free_slots: &mut Vec<u32>| {
        active.retain(|&(end, v)| {
            if end <= upto {
                release(homes[v.index()], gp, fp, free_slots);
                false
            } else {
                true
            }
        });
    };
    // A loop phi's register, hinted to its inputs defined after it (its back edges' values):
    // where the register is free when such a value is defined, the value takes it and its edge
    // moves nothing (Maglev's SetLoopPhiRegisterHint, maglev-regalloc.cc:1227-1240).
    let mut hints: Vec<Option<u8>> = vec![None; n];
    // The values the pass spills, as it spills them (a value's home never leaves its slot): the
    // copies' passes read them (`spill_copies`), and a pass that spills nothing gives them none.
    let mut spilled: Vec<NodeIx> = Vec::new();
    let mut assign = |v: NodeIx,
                      homes: &mut Vec<Home>,
                      active: &mut Vec<(u32, NodeIx)>,
                      gp: &mut Pool,
                      fp: &mut Pool,
                      free_slots: &mut Vec<u32>,
                      spill_bytes: &mut u32,
                      hints: &[Option<u8>]| {
        let node = graph.node(v);
        if rematerializable(&node.op) {
            homes[v.index()] = Home::Const;
            return;
        }
        let Some(class) = class_of(node.rep) else {
            return;
        };
        let range = ranges[v.index()];
        if range.1 == range.0 {
            return; // nothing reads it: the code writes it to a scratch register
        }
        let across = crosses_call(range);
        // A value across an effect takes a callee-saved home, which no call clobbers; across a
        // stub's call, none of the registers that call clobbers.
        let avoid = if across { 0 } else { clobbered(range, class) };
        let pool = match class {
            Class::Gp => &mut *gp,
            Class::Fp => &mut *fp,
        };
        let hinted = hints[v.index()]
            .filter(|&r| avoid & (1 << r) == 0)
            .and_then(|r| {
                if across {
                    Pool::take_this(&mut pool.callee, r)
                } else {
                    Pool::take_this(&mut pool.caller, r)
                        .or_else(|| Pool::take_this(&mut pool.callee, r))
                }
            });
        let mut register = hinted.or_else(|| {
            if across {
                Pool::take(&mut pool.callee)
            } else {
                Pool::take_but(&mut pool.caller, avoid).or_else(|| Pool::take(&mut pool.callee))
            }
        });
        if register.is_none() {
            // No register is free: as Maglev chooses, the held value whose next use is the
            // farthest gives up its register and is spilled at its definition, unless this
            // value's own next use is farther. Its range began before now, so its slot is a new
            // one, which nothing else has held.
            let def = range.0;
            let fits = |home: Home| match (home, class) {
                (Home::Gp(r), Class::Gp) => {
                    (!across || file.gp_callee.contains(&r)) && avoid & (1 << r) == 0
                }
                (Home::Fp(r), Class::Fp) => {
                    (!across || file.fp_callee.contains(&r)) && avoid & (1 << r) == 0
                }
                _ => false,
            };
            let victim = active
                .iter()
                .filter(|&&(_, u)| fits(homes[u.index()]))
                .map(|&(_, u)| (next_use(u, def), u))
                .max();
            if let Some((far, u)) = victim {
                if far > next_use(v, def) {
                    let (Home::Gp(r) | Home::Fp(r)) = homes[u.index()] else {
                        unreachable!("the victim holds a register")
                    };
                    homes[u.index()] = Home::Spill(*spill_bytes);
                    *spill_bytes += 8;
                    spilled.push(u);
                    register = Some(r);
                }
            }
        }
        let home = match (register, class) {
            (Some(r), Class::Gp) => Home::Gp(r),
            (Some(r), Class::Fp) => Home::Fp(r),
            (None, _) => {
                spilled.push(v);
                Home::Spill(free_slots.pop().unwrap_or_else(|| {
                    let offset = *spill_bytes;
                    *spill_bytes += 8;
                    offset
                }))
            }
        };
        if let (Some(r), Class::Gp) = (register, class) {
            if is_callee(file, class, r) && !used_callee_gp.contains(&r) {
                used_callee_gp.push(r);
            }
        }
        if let (Some(r), Class::Fp) = (register, class) {
            if across && is_callee(file, class, r) {
                fp_callee_across_effect = true;
            }
        }
        homes[v.index()] = home;
        active.push((range.1, v));
    };
    for &b in &order {
        let data = graph.block(b);
        expire(
            &mut active,
            head[b.index()],
            &homes,
            &mut gp,
            &mut fp,
            &mut free_slots,
        );
        for &phi in &data.phis {
            assign(
                phi,
                &mut homes,
                &mut active,
                &mut gp,
                &mut fp,
                &mut free_slots,
                &mut spill_bytes,
                &hints,
            );
            if let Home::Gp(r) | Home::Fp(r) = homes[phi.index()] {
                for &input in &graph.node(phi).inputs {
                    let at = pos[input.index()];
                    if at != u32::MAX && at > pos[phi.index()] {
                        hints[input.index()] = Some(r);
                    }
                }
            }
        }
        for &node in &data.body {
            let p = pos[node.index()];
            // A node that can exit takes its home while everything it reads is still held: its
            // exit may run after it wrote its value (an overflow check), and its recipe reads
            // the values it read.
            if graph.node(node).op.can_exit() {
                assign(
                    node,
                    &mut homes,
                    &mut active,
                    &mut gp,
                    &mut fp,
                    &mut free_slots,
                    &mut spill_bytes,
                    &hints,
                );
                expire(&mut active, p, &homes, &mut gp, &mut fp, &mut free_slots);
            } else {
                expire(&mut active, p, &homes, &mut gp, &mut fp, &mut free_slots);
                assign(
                    node,
                    &mut homes,
                    &mut active,
                    &mut gp,
                    &mut fp,
                    &mut free_slots,
                    &mut spill_bytes,
                    &hints,
                );
            }
        }
    }
    // Constants that nothing schedules still have no register.
    for (v, node) in graph.nodes.iter().enumerate() {
        if rematerializable(&node.op) {
            homes[v] = Home::Const;
        }
    }
    let copies = spill_copies(
        graph,
        file,
        &spilled,
        &Made {
            order: &order,
            head: &head,
            tail: &tail,
            homes: &homes,
            ranges: &ranges,
            edges: &edges,
            calls: &calls,
            stub_calls: &stub_calls,
            live_in: (&live_in, words),
        },
        &mut used_callee_gp,
        &mut fp_callee_across_effect,
    );
    used_callee_gp.sort_unstable();
    let alloc = Allocation {
        order,
        homes,
        pos,
        ranges,
        spill_bytes,
        callee_saved_gp: used_callee_gp,
        fp_callee_across_effect,
        caches: copies.caches,
        block_caches: copies.block_caches,
        spans: copies.spans,
        block_spans: copies.block_spans,
        live_index: std::cell::OnceCell::new(),
    };
    #[cfg(test)]
    {
        tests::assert_caches(graph, &alloc);
        tests::assert_reference(graph, file, &alloc);
        TEST_CACHES.with(|made| made.set(made.get() + alloc.caches.len()));
        TEST_SPANS.with(|made| made.set(made.get() + alloc.spans.len()));
        let callee = alloc
            .caches
            .iter()
            .filter(|c| matches!(c.reg, Home::Gp(r) if file.gp_callee.contains(&r)))
            .count();
        TEST_CALLEE_CACHES.with(|made| made.set(made.get() + callee));
    }
    Ok(alloc)
}

#[cfg(test)]
thread_local! {
    /// The loop caches the allocations on this thread made (step R1's end-to-end tests).
    pub(crate) static TEST_CACHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The read spans the allocations on this thread made.
    pub(crate) static TEST_SPANS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The loop caches in callee-saved x registers the allocations on this thread made.
    pub(crate) static TEST_CALLEE_CACHES: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// 10^k for the static execution estimate of a block at loop depth k (B3's,
/// b3/B3EstimateStaticExecutionCounts.cpp:42-43), up to depth 9.
const FREQUENCY: [u64; 10] = [
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
    1_000_000_000,
];

/// A natural loop of the code order (`loop_caches`).
struct Loop {
    header: BlockIx,
    /// Its blocks, the header first.
    blocks: Vec<BlockIx>,
    /// Its blocks as a bit set over block indices.
    member: Vec<u64>,
    /// The header's head position to the last tail of its blocks.
    hull: (u32, u32),
    /// The loops that hold its header, itself among them (`block_loops`).
    depth: u32,
    /// Whether one of its blocks holds an effect or a call (`NodeOp::is_effectful`).
    call: bool,
    /// The registers its blocks' stub calls clobber (`clobbers`), x and d.
    clobbered: (u64, u64),
}

/// Whether block index `i` is in the bit set `bits`.
fn contains(bits: &[u64], i: usize) -> bool {
    bits[i / 64] & (1 << (i % 64)) != 0
}

/// The natural loops of the code order: a block the builder marked a loop header with a
/// predecessor at or after it, and the blocks that reach those predecessors without passing it.
/// A walk that reaches the entry block found no natural loop there, and gives none. Whether a
/// loop calls and what its stub calls clobber come from the calls' positions (`Made`), a block's
/// by a search of its head to its tail, not from its nodes.
fn natural_loops(graph: &Graph, made: &Made) -> Vec<Loop> {
    let (order, head, tail) = (made.order, made.head, made.tail);
    // The code order's places are the heads' order; an unreachable block has none.
    let place = |b: BlockIx| head[b.index()];
    let reachable = |b: BlockIx| head[b.index()] != u32::MAX;
    let words = graph.blocks.len().div_ceil(64);
    let mut loops: Vec<Loop> = Vec::new();
    for &h in order {
        let data = graph.block(h);
        if !data.loop_header {
            continue;
        }
        let mut stack: Vec<BlockIx> = data
            .preds
            .iter()
            .copied()
            .filter(|&p| reachable(p) && place(p) >= place(h))
            .collect();
        if stack.is_empty() {
            continue;
        }
        let mut member = vec![0u64; words];
        insert(&mut member, h.index());
        let mut blocks = vec![h];
        let mut natural = true;
        while let Some(b) = stack.pop() {
            if contains(&member, b.index()) {
                continue;
            }
            if b == order[0] {
                natural = false;
                break;
            }
            insert(&mut member, b.index());
            blocks.push(b);
            stack.extend(
                graph
                    .block(b)
                    .preds
                    .iter()
                    .copied()
                    .filter(|&p| reachable(p) && !contains(&member, p.index())),
            );
        }
        if !natural {
            continue;
        }
        let mut hull = (head[h.index()], 0);
        let (mut call, mut clobbered) = (false, (0, 0));
        for &b in &blocks {
            let (from, to) = (head[b.index()], tail[b.index()]);
            hull.1 = hull.1.max(to);
            let first = made.calls.partition_point(|&p| p < from);
            call |= made.calls.get(first).is_some_and(|&p| p <= to);
            let first = made.stub_calls.partition_point(|&(p, _, _)| p < from);
            for &(p, gp, fp) in &made.stub_calls[first..] {
                if p > to {
                    break;
                }
                clobbered = (clobbered.0 | gp, clobbered.1 | fp);
            }
        }
        loops.push(Loop {
            header: h,
            blocks,
            member,
            hull,
            depth: 0,
            call,
            clobbered,
        });
    }
    loops
}

/// Each block's loops (`natural_loops`' indices, ascending) as one flat list, a block's run
/// `start[b]..start[b + 1]`: its loop depth is the run's length. Each loop's own depth is its
/// header's.
fn block_loops(loops: &mut [Loop], blocks: usize) -> (Vec<u32>, Vec<u32>) {
    let mut start = vec![0u32; blocks + 1];
    for l in loops.iter() {
        for &b in &l.blocks {
            start[b.index() + 1] += 1;
        }
    }
    for b in 0..blocks {
        start[b + 1] += start[b];
    }
    let mut fill = start.clone();
    let mut list = vec![0u32; start[blocks] as usize];
    for (k, l) in loops.iter().enumerate() {
        for &b in &l.blocks {
            list[fill[b.index()] as usize] = k as u32;
            fill[b.index()] += 1;
        }
    }
    for l in loops.iter_mut() {
        let h = l.header.index();
        l.depth = start[h + 1] - start[h];
    }
    (start, list)
}

/// Loop-level caches of spilled values (P3 stage 3 step R1). A value whose home is a spill slot,
/// defined outside a loop and read inside it, is loaded once at the loop's header into a register
/// that nothing else holds through the loop, and the loop's reads take the register; the slot
/// stays the value's one home, which every exit, throw and parent recipe names and the collector
/// scans. Maglev's reload hint (HoistLoopReloads, maglev/maglev-regalloc.cc:2240-2271: a node
/// defined before the loop and used in it without a call between, loaded into a free register on
/// the loop's forward edge and kept across its back edge); the choice among values by Air's
/// frequency-weighted use cost (b3/air/AirAllocateRegistersByGreedy.cpp:733-752, a block at depth
/// k counting 10^k), less the load at each entry (its loop split's fixup cost, :2715-2722), and
/// the outermost loop the register is free through first, as Air's loop split chooses
/// (:2645-2679). A loop with a call takes only a callee-saved register, which no call clobbers;
/// none takes a register one of its stub calls clobbers. A value defined in the loop is not
/// cached: that would be a second home over part of its range.
#[allow(clippy::too_many_arguments)]
fn loop_caches(
    graph: &Graph,
    file: &RegisterFile,
    loops: &[Loop],
    (start, list): (&[u32], &[u32]),
    reads: &[Read],
    held: &Held,
    used_callee_gp: &mut Vec<u8>,
    fp_callee_across_effect: &mut bool,
) -> (Vec<Cache>, Vec<Vec<u16>>) {
    // The candidates: (loop, value, weight), each register read of a spilled value (each read
    // once: `spilled_reads`) counting its block's frequency for each loop that holds the block
    // and not the value's definition.
    let mut weights: Vec<(u32, u32, u64)> = Vec::new();
    for r in reads {
        let b = r.block as usize;
        let loops_of = &list[start[b] as usize..start[b + 1] as usize];
        if loops_of.is_empty() {
            continue;
        }
        let weight = FREQUENCY[loops_of.len().min(9)];
        let def = graph.node(NodeIx(r.value)).block.index();
        for &k in loops_of {
            if !contains(&loops[k as usize].member, def) {
                weights.push((k, r.value, weight));
            }
        }
    }
    if weights.is_empty() {
        return (Vec::new(), Vec::new());
    }
    weights.sort_unstable();
    // (benefit, loop, value): the reads' weight less one load at each entry.
    let mut candidates: Vec<(u64, u32, u32)> = Vec::new();
    for run in weights.chunk_by(|a, b| a.0 == b.0 && a.1 == b.1) {
        let (k, v, _) = run[0];
        let weight: u64 = run.iter().map(|r| r.2).sum();
        let entry = FREQUENCY[(loops[k as usize].depth as usize - 1).min(9)];
        if weight > entry {
            candidates.push((weight - entry, k, v));
        }
    }
    candidates.sort_unstable_by(|a, b| {
        b.0.cmp(&a.0)
            .then(loops[b.1 as usize].depth.cmp(&loops[a.1 as usize].depth))
            .then(a.2.cmp(&b.2))
    });
    let mut caches: Vec<Cache> = Vec::new();
    let overlap = |a: (u32, u32), b: (u32, u32)| a.0 <= b.1 && b.0 <= a.1;
    for &(_, k, v) in &candidates {
        let l = &loops[k as usize];
        let v = NodeIx(v);
        if caches
            .iter()
            .any(|c| c.value == v && overlap(c.hull, l.hull))
        {
            continue;
        }
        let Some(class) = class_of(graph.node(v).rep) else {
            continue;
        };
        let (ci, caller, callee, avoid) = match class {
            Class::Gp => (0, file.gp_caller, file.gp_callee, l.clobbered.0),
            Class::Fp => (1, file.fp_caller, file.fp_callee, l.clobbered.1),
        };
        let free = |r: u8| {
            avoid & (1 << r) == 0
                && held.free(ci, r, l.hull)
                && !caches.iter().any(|c| {
                    matches!((c.reg, class), (Home::Gp(q), Class::Gp) | (Home::Fp(q), Class::Fp) if q == r)
                        && overlap(c.hull, l.hull)
                })
        };
        let choices: &[u8] = if l.call { &[] } else { caller };
        let Some(r) = choices.iter().chain(callee).copied().find(|&r| free(r)) else {
            continue;
        };
        let reg = match class {
            Class::Gp => Home::Gp(r),
            Class::Fp => Home::Fp(r),
        };
        if class == Class::Gp && callee.contains(&r) && !used_callee_gp.contains(&r) {
            used_callee_gp.push(r);
        }
        if class == Class::Fp && callee.contains(&r) && l.call {
            *fp_callee_across_effect = true;
        }
        caches.push(Cache {
            value: v,
            reg,
            header: l.header,
            hull: l.hull,
        });
    }
    if caches.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut block_caches: Vec<Vec<u16>> = vec![Vec::new(); graph.blocks.len()];
    for (k, c) in caches.iter().enumerate() {
        let l = loops
            .iter()
            .find(|l| l.header == c.header)
            .expect("a cache's loop");
        for &b in &l.blocks {
            block_caches[b.index()].push(k as u16);
        }
    }
    (caches, block_caches)
}

/// The register copies of spilled values (P3 stage 3 step R1): loop caches, then read spans.
#[derive(Default)]
struct Copies {
    caches: Vec<Cache>,
    block_caches: Vec<Vec<u16>>,
    spans: Vec<Span>,
    block_spans: Vec<Vec<u16>>,
}

/// Add `range` to `list`, a register's ranges by start, which it does not overlap.
fn hold(list: &mut Vec<(u32, u32)>, range: (u32, u32)) {
    let at = list.partition_point(|&(start, _)| start < range.0);
    list.insert(at, range);
}

/// Whether no range of `list` (a register's, by start, none overlapping another) overlaps
/// `range`.
fn free_over(list: &[(u32, u32)], range: (u32, u32)) -> bool {
    let first = list.partition_point(|&(_, end)| end < range.0);
    list.get(first).is_none_or(|&(start, _)| start > range.1)
}

/// Each register's ranges (P3 stage 3 step R1), by class (0 x, 1 d) and number, `32 * class +
/// number`: the homes' in one flat list, each register's run by start (they do not overlap: one
/// value a register at a time), and the copies' taken after them, each register's by start.
struct Held {
    start: Vec<u32>,
    homes: Vec<(u32, u32)>,
    taken: Vec<Vec<(u32, u32)>>,
}

impl Held {
    fn new(homes: &[Home], ranges: &[(u32, u32)]) -> Self {
        let index = |home: &Home| match *home {
            Home::Gp(r) => Some(r as usize),
            Home::Fp(r) => Some(32 + r as usize),
            _ => None,
        };
        let mut start = vec![0u32; 65];
        for home in homes {
            if let Some(i) = index(home) {
                start[i + 1] += 1;
            }
        }
        for i in 0..64 {
            start[i + 1] += start[i];
        }
        let mut fill = start.clone();
        let mut list = vec![(0, 0); start[64] as usize];
        for (v, home) in homes.iter().enumerate() {
            if let Some(i) = index(home) {
                list[fill[i] as usize] = ranges[v];
                fill[i] += 1;
            }
        }
        for i in 0..64 {
            list[start[i] as usize..start[i + 1] as usize].sort_unstable();
        }
        Held {
            start,
            homes: list,
            taken: vec![Vec::new(); 64],
        }
    }

    /// Whether register `r` of class `class` holds nothing over `range`.
    fn free(&self, class: usize, r: u8, range: (u32, u32)) -> bool {
        let i = 32 * class + r as usize;
        free_over(
            &self.homes[self.start[i] as usize..self.start[i + 1] as usize],
            range,
        ) && free_over(&self.taken[i], range)
    }

    /// Hold register `r` of class `class` over `range`, which it held nothing over.
    fn hold(&mut self, class: usize, r: u8, range: (u32, u32)) {
        hold(&mut self.taken[32 * class + r as usize], range);
    }
}

/// What the allocation's passes made that the copies' passes read (P3 stage 3 step R1): the code
/// order, the blocks' head and tail positions, the homes, ranges and edges, the positions of the
/// calls and of the stub calls with their clobbers, and each block's live-in set (flat, `words`
/// words a block).
struct Made<'a> {
    order: &'a [BlockIx],
    head: &'a [u32],
    tail: &'a [u32],
    homes: &'a [Home],
    ranges: &'a [(u32, u32)],
    edges: &'a EdgeInputs,
    calls: &'a [u32],
    stub_calls: &'a [(u32, u64, u64)],
    live_in: (&'a [u64], usize),
}

/// A register read of a spilled value (P3 stage 3 step R1, `spilled_reads`): one of the inputs of
/// the node at position `at` in `block`, after `calls` calls in the code order, or (`edge`) one
/// of the values its edges carry, at its tail.
#[derive(Clone, Copy, Debug)]
struct Read {
    value: u32,
    block: u32,
    at: u32,
    calls: u32,
    edge: bool,
}

/// The register reads of the spilled values (`spilled`, a bit set over values): one walk of the
/// blocks in code order, each block's reads once for both passes, a node's inputs at its
/// position and its block's edges at its tail, each read with the calls before it.
fn spilled_reads(graph: &Graph, made: &Made, spilled: &[u64]) -> Vec<Read> {
    let mut reads: Vec<Read> = Vec::new();
    let mut calls = 0usize;
    for &b in made.order {
        let data = graph.block(b);
        let mut at = made.head[b.index()];
        for &node in &data.body {
            at += 2;
            while made.calls.get(calls).is_some_and(|&p| p < at) {
                calls += 1;
            }
            for &input in &graph.node(node).inputs {
                if contains(spilled, input.index()) {
                    reads.push(Read {
                        value: input.0,
                        block: b.0,
                        at,
                        calls: calls as u32,
                        edge: false,
                    });
                }
            }
        }
        for s in 0..data.succs.len() {
            for input in made.edges.of(b, s) {
                if contains(spilled, input.index()) {
                    reads.push(Read {
                        value: input.0,
                        block: b.0,
                        at: made.tail[b.index()],
                        calls: calls as u32,
                        edge: true,
                    });
                }
            }
        }
    }
    reads
}

/// The register copies of spilled values (`Copies`), only where they can make one (P3 stage 3
/// step R1, F5): the values the forward pass spilled (`spilled_list`) and their register reads
/// (`spilled_reads`); none of the passes where no spilled value is read twice in a block with no
/// call between (a read span's run) and none is live into a loop header (a loop cache's value,
/// defined outside the loop); else the natural loops and each block's loops, each register's
/// ranges, then the loop caches (`loop_caches`) where a loop can have one, whose hulls then hold
/// their registers, then the read spans (`read_spans`) where a run can be made.
fn spill_copies(
    graph: &Graph,
    file: &RegisterFile,
    spilled_list: &[NodeIx],
    made: &Made,
    used_callee_gp: &mut Vec<u8>,
    fp_callee_across_effect: &mut bool,
) -> Copies {
    let mut copies = Copies::default();
    let homes = made.homes;
    if spilled_list.is_empty() {
        return copies;
    }
    let mut spilled = vec![0u64; homes.len().div_ceil(64)];
    for &v in spilled_list {
        insert(&mut spilled, v.index());
    }
    let reads = spilled_reads(graph, made, &spilled);
    // The node reads by (block, value, calls before, position): a run is two or more of one
    // block, value and call count (a call's own read ends a run), which a loop cache of the value
    // may then take from it (`read_spans`).
    let mut span_reads: Vec<(u32, u32, u32, u32)> = reads
        .iter()
        .filter(|r| !r.edge)
        .map(|r| (r.block, r.value, r.calls, r.at))
        .collect();
    span_reads.sort_unstable();
    let runs = span_reads
        .windows(2)
        .any(|w| (w[0].0, w[0].1, w[0].2) == (w[1].0, w[1].1, w[1].2));
    // A loop cache's value: a spilled value live into a natural loop's header, so defined outside
    // the loop. The headers: blocks the builder marked with a predecessor at or after them in the
    // code order.
    let (live_in, words) = made.live_in;
    let in_loop = made.order.iter().any(|&h| {
        let data = graph.block(h);
        let at = made.head[h.index()];
        data.loop_header
            && data
                .preds
                .iter()
                .any(|&p| made.head[p.index()] != u32::MAX && made.head[p.index()] >= at)
            && spilled_list
                .iter()
                .any(|&v| contains(&live_in[h.index() * words..][..words], v.index()))
    });
    if !runs && !in_loop {
        return copies;
    }
    let mut loops = natural_loops(graph, made);
    let (start, list) = block_loops(&mut loops, graph.blocks.len());
    let mut held = Held::new(homes, made.ranges);
    if in_loop && !loops.is_empty() {
        (copies.caches, copies.block_caches) = loop_caches(
            graph,
            file,
            &loops,
            (&start, &list),
            &reads,
            &held,
            used_callee_gp,
            fp_callee_across_effect,
        );
        for c in &copies.caches {
            match c.reg {
                Home::Gp(r) => held.hold(0, r, c.hull),
                Home::Fp(r) => held.hold(1, r, c.hull),
                _ => {}
            }
        }
    }
    if runs {
        (copies.spans, copies.block_spans) = read_spans(
            graph,
            file,
            made,
            span_reads,
            &start,
            (&copies.caches, &copies.block_caches),
            &mut held,
            used_callee_gp,
        );
    }
    copies
}

/// Read spans of spilled values (P3 stage 3 step R1): within a block, the reads of a spilled
/// value that no loop cache holds there, with no call between them (a call's own read ends a
/// run), read one register, loaded before the first, where a register of its class is free from
/// the first read to the last. Maglev's reload at a register use, kept while the register is free
/// and until the next call (AssignArbitraryRegisterInput, maglev/maglev-regalloc.cc:1423-1490;
/// SpillAndClearRegisters at a call, :1187, 1680-1683), within one-home allocation: the register
/// is a copy over the run, never the value's home. Runs by the weight of the loads they save
/// (Air's frequency-weighted cost); a run whose registers give out part way is cut there, its
/// rest a run of its own. A caller-saved register, or a callee-saved one the prologue saves
/// already (an x register a home or cache took): a span adds no save.
#[allow(clippy::too_many_arguments)]
fn read_spans(
    graph: &Graph,
    file: &RegisterFile,
    made: &Made,
    mut reads: Vec<(u32, u32, u32, u32)>,
    loops_start: &[u32],
    (caches, block_caches): (&[Cache], &[Vec<u16>]),
    held: &mut Held,
    used_callee_gp: &[u8],
) -> (Vec<Span>, Vec<Vec<u16>>) {
    // (block, value, the calls before it, position) of each node's read of a spilled value, by
    // that key (`spill_copies`), but those of a value a loop cache holds in the block: a run is
    // one block, value and call count.
    if !caches.is_empty() {
        reads.retain(|&(b, v, _, _)| {
            !block_caches
                .get(b as usize)
                .is_some_and(|ks| ks.iter().any(|&k| caches[k as usize].value.0 == v))
        });
    }
    if reads.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    // (weight, first, end) of each run of two reads or more, as indices into `reads`.
    let mut runs: Vec<(u64, u32, u32)> = Vec::new();
    let mut first = 0;
    while first < reads.len() {
        let key = |k: usize| (reads[k].0, reads[k].1, reads[k].2);
        let mut end = first + 1;
        while end < reads.len() && key(end) == key(first) {
            end += 1;
        }
        if end - first >= 2 {
            let b = reads[first].0 as usize;
            let depth = loops_start[b + 1] - loops_start[b];
            let frequency = FREQUENCY[(depth as usize).min(9)];
            runs.push((
                (end - first - 1) as u64 * frequency,
                first as u32,
                end as u32,
            ));
        }
        first = end;
    }
    runs.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    // The registers a span may take, by class: caller-saved, then the callee-saved x registers
    // the prologue saves already.
    let gp_regs: Vec<u8> = file
        .gp_caller
        .iter()
        .chain(file.gp_callee.iter().filter(|r| used_callee_gp.contains(r)))
        .copied()
        .collect();
    let mut spans: Vec<Span> = Vec::new();
    for &(_, first, end) in &runs {
        let run = &reads[first as usize..end as usize];
        let (block, v) = (BlockIx(run[0].0), NodeIx(run[0].1));
        let Some(class) = class_of(graph.node(v).rep) else {
            continue;
        };
        let (ci, regs): (usize, &[u8]) = match class {
            Class::Gp => (0, &gp_regs),
            Class::Fp => (1, file.fp_caller),
        };
        // The registers of the class free from position `p` to `q` in the block: no range in
        // them overlaps, and no stub call of the block's nodes there clobbers them.
        let free = |held: &Held, p: u32, q: u32| {
            let first = made.stub_calls.partition_point(|&(at, _, _)| at < p);
            let clobbered = made.stub_calls[first..]
                .iter()
                .take_while(|&&(at, _, _)| at <= q)
                .fold(0u64, |acc, &(_, gp, fp)| {
                    acc | if ci == 0 { gp } else { fp }
                });
            regs.iter().fold(0u64, |mask, &r| {
                if clobbered & (1 << r) == 0 && held.free(ci, r, (p, q)) {
                    mask | (1 << r)
                } else {
                    mask
                }
            })
        };
        // Segments of the run, each the longest from its first read through which a register
        // stays free, up to the definition of its last reading node's value (to + 1), so that the
        // span's register is no home its readers write: a node may write its output before it
        // reads all its inputs, and a cold path's restore would put the copy back over it.
        let mut take = |held: &mut Held, from: u32, to: u32, mask: u64| {
            let r = regs
                .iter()
                .copied()
                .find(|&r| mask & (1 << r) != 0)
                .expect("a free register");
            held.hold(ci, r, (from, to));
            spans.push(Span {
                value: v,
                reg: if ci == 0 { Home::Gp(r) } else { Home::Fp(r) },
                block,
                from,
                to,
            });
        };
        // A register free over the whole run: one span, as the cuts below would make it (their
        // steps' intervals cover the run's, so their registers are those free over it).
        let last = run[run.len() - 1].3;
        let whole = free(held, run[0].3, last + 1);
        if whole != 0 {
            take(held, run[0].3, last, whole);
            continue;
        }
        let mut start = 0;
        let mut mask = 0u64;
        for k in 1..run.len() {
            let step = free(held, run[k - 1].3, run[k].3 + 1);
            let next = if k - 1 == start { step } else { mask & step };
            if next != 0 {
                mask = next;
                continue;
            }
            if k - 1 > start {
                take(held, run[start].3, run[k - 1].3, mask);
            }
            start = k;
            mask = 0;
        }
        if run.len() - 1 > start {
            take(held, run[start].3, run[run.len() - 1].3, mask);
        }
    }
    if spans.is_empty() {
        return (Vec::new(), Vec::new());
    }
    spans.sort_unstable_by_key(|s| (s.block, s.from, s.value));
    let mut block_spans: Vec<Vec<u16>> = vec![Vec::new(); graph.blocks.len()];
    for (k, s) in spans.iter().enumerate() {
        block_spans[s.block.index()].push(k as u16);
    }
    (spans, block_spans)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::opt::ir::*;

    fn state(g: &mut Graph, pc: u32, stack: &[NodeIx]) -> StateIx {
        g.add_state(FrameState {
            pc,
            slots: vec![SlotSource::Frame; g.n_slots].into(),
            stack: stack.into(),
            handlers: Box::new([]),
            frame: FrameIx::COMPILED,
        })
    }

    /// No two values whose ranges overlap share a register or a slot, values living across a
    /// call have callee-saved homes or slots, every read value has a home, and `live_at`'s
    /// index agrees with a scan of every range at each scheduled node.
    pub(crate) fn assert_consistent(graph: &Graph, alloc: &Allocation) {
        assert_live_at_agrees(graph, alloc);
        assert_caches(graph, alloc);
        let calls: Vec<u32> = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(v, node)| node.op.is_effectful() && alloc.pos[*v] != u32::MAX)
            .map(|(v, _)| alloc.pos[v])
            .collect();
        let held: Vec<usize> = (0..graph.nodes.len())
            .filter(|&v| matches!(alloc.homes[v], Home::Gp(_) | Home::Fp(_) | Home::Spill(_)))
            .collect();
        for (k, &a) in held.iter().enumerate() {
            let (sa, ea) = alloc.ranges[a];
            if calls.iter().any(|&p| sa < p && p < ea) {
                assert!(
                    match alloc.homes[a] {
                        Home::Gp(r) => ARM64.gp_callee.contains(&r),
                        Home::Fp(r) => ARM64.fp_callee.contains(&r),
                        _ => true,
                    },
                    "value {a} lives across a call in {:?}",
                    alloc.homes[a]
                );
            }
            for (v, node) in graph.nodes.iter().enumerate() {
                let p = alloc.pos[v];
                if p == u32::MAX || !(sa < p && p < ea) {
                    continue;
                }
                let (gp, fp) = clobbers(&node.op);
                assert!(
                    match alloc.homes[a] {
                        Home::Gp(r) => gp & (1 << r) == 0,
                        Home::Fp(r) => fp & (1 << r) == 0,
                        _ => true,
                    },
                    "value {a} lives across node {v}'s stub call in {:?}, which it clobbers",
                    alloc.homes[a]
                );
            }
            for &b in &held[k + 1..] {
                let (sb, eb) = alloc.ranges[b];
                if sa <= eb && sb <= ea {
                    assert_ne!(
                        alloc.homes[a], alloc.homes[b],
                        "values {a} {:?} and {b} {:?} overlap",
                        alloc.ranges[a], alloc.ranges[b]
                    );
                }
            }
        }
        for (v, node) in graph.nodes.iter().enumerate() {
            let read = alloc.ranges[v].1 > alloc.ranges[v].0;
            if read && node.rep != Rep::None && alloc.pos[v] != u32::MAX {
                assert_ne!(alloc.homes[v], Home::None, "value {v} is read");
            }
        }
    }

    /// The loop caches' rules (step R1), checked after every allocation the tests make: each
    /// cache copies a spilled value of its register's class, defined outside the loop and live
    /// through the loop's hull; the cache's blocks are a loop whose header is its; no value homed
    /// in the register and no other cache in it overlaps the hull, and no node of the loop
    /// clobbers it (a stub call's clobbers; a call's every caller-saved register); a value has no
    /// two caches at a point; a callee-saved x register is saved by the prologue, and a d8-d15
    /// one through a loop with a call is flagged for the stubs' check.
    pub(crate) fn assert_caches(graph: &Graph, alloc: &Allocation) {
        let overlap = |a: (u32, u32), b: (u32, u32)| a.0 <= b.1 && b.0 <= a.1;
        for (k, c) in alloc.caches.iter().enumerate() {
            let v = c.value.index();
            assert!(
                matches!(alloc.homes[v], Home::Spill(_)),
                "cache {k} of value {v} in {:?}",
                alloc.homes[v]
            );
            let class = class_of(graph.nodes[v].rep);
            let (r, callee) = match (c.reg, class) {
                (Home::Gp(r), Some(Class::Gp)) => (r, ARM64.gp_callee.contains(&r)),
                (Home::Fp(r), Some(Class::Fp)) => (r, ARM64.fp_callee.contains(&r)),
                other => panic!("cache {k} of value {v}: {other:?}"),
            };
            let blocks: Vec<usize> = (0..alloc.block_caches.len())
                .filter(|&b| alloc.block_caches[b].contains(&(k as u16)))
                .collect();
            assert!(
                blocks.contains(&c.header.index()) && graph.blocks[c.header.index()].loop_header,
                "cache {k}'s header"
            );
            assert!(
                !blocks.contains(&graph.nodes[v].block.index()),
                "cache {k}: value {v} is defined in its loop"
            );
            let (start, end) = alloc.ranges[v];
            assert!(
                start < c.hull.0 && end >= c.hull.1,
                "cache {k}: value {v}'s range {:?} does not cover the loop's {:?}",
                (start, end),
                c.hull
            );
            for (u, &home) in alloc.homes.iter().enumerate() {
                if home == c.reg && alloc.pos[u] != u32::MAX {
                    assert!(
                        !overlap(alloc.ranges[u], c.hull),
                        "cache {k} in {:?} overlaps value {u} {:?} (loop {:?})",
                        c.reg,
                        alloc.ranges[u],
                        c.hull
                    );
                }
            }
            for (j, d) in alloc.caches.iter().enumerate() {
                if j != k && overlap(d.hull, c.hull) {
                    assert_ne!(d.reg, c.reg, "caches {k} and {j} share a register");
                    assert_ne!(d.value, c.value, "caches {k} and {j} copy one value");
                }
            }
            let mut call = false;
            for &b in &blocks {
                for &n in &graph.blocks[b].body {
                    let op = &graph.nodes[n.index()].op;
                    call |= op.is_effectful();
                    let (gp, fp) = clobbers(op);
                    let hit = match c.reg {
                        Home::Gp(_) => gp & (1 << r) != 0,
                        _ => fp & (1 << r) != 0,
                    };
                    assert!(!hit, "cache {k} in {:?}, which node {n:?} clobbers", c.reg);
                }
            }
            assert!(
                callee || !call,
                "cache {k} in caller-saved {:?} through a loop with a call",
                c.reg
            );
            if callee && matches!(c.reg, Home::Gp(_)) {
                assert!(
                    alloc.callee_saved_gp.contains(&r),
                    "cache {k}'s x{r} is not saved by the prologue"
                );
            }
            if callee && call && matches!(c.reg, Home::Fp(_)) {
                assert!(
                    alloc.fp_callee_across_effect,
                    "cache {k}'s d{r} across a call"
                );
            }
        }
        assert_spans(graph, alloc);
    }

    /// The read spans' rules (step R1): each span copies a spilled value of its register's class
    /// over reads of it in its block, from its first reading node to its last; no call runs
    /// before its last read; no node there clobbers its register; no value homed in the register
    /// overlaps the span or the definition after its last read, no cache and no other span in it
    /// overlaps the span; no loop cache of the value holds in
    /// the block and no other span of the value overlaps; a callee-saved x register is one the
    /// prologue saves.
    pub(crate) fn assert_spans(graph: &Graph, alloc: &Allocation) {
        let overlap = |a: (u32, u32), b: (u32, u32)| a.0 <= b.1 && b.0 <= a.1;
        for (k, s) in alloc.spans.iter().enumerate() {
            let v = s.value.index();
            assert!(
                matches!(alloc.homes[v], Home::Spill(_)),
                "span {k} of value {v} in {:?}",
                alloc.homes[v]
            );
            let r = match (s.reg, class_of(graph.nodes[v].rep)) {
                (Home::Gp(r), Some(Class::Gp)) | (Home::Fp(r), Some(Class::Fp)) => r,
                other => panic!("span {k} of value {v}: {other:?}"),
            };
            assert!(alloc.block_spans[s.block.index()].contains(&(k as u16)));
            let body = &graph.blocks[s.block.index()].body;
            let reads = |p: u32| {
                body.iter().any(|&n| {
                    alloc.pos[n.index()] == p && graph.nodes[n.index()].inputs.contains(&s.value)
                })
            };
            assert!(
                s.from <= s.to && reads(s.from) && reads(s.to),
                "span {k}: {s:?}"
            );
            for &n in body {
                let p = alloc.pos[n.index()];
                if p < s.from || p > s.to {
                    continue;
                }
                let op = &graph.nodes[n.index()].op;
                assert!(
                    !op.is_effectful() || p == s.to,
                    "span {k} across node {n:?}'s call"
                );
                let (gp, fp) = clobbers(op);
                let mask = if matches!(s.reg, Home::Gp(_)) { gp } else { fp };
                assert_eq!(
                    mask & (1 << r),
                    0,
                    "span {k} in {:?}, which {n:?} clobbers",
                    s.reg
                );
            }
            for (u, &home) in alloc.homes.iter().enumerate() {
                if home == s.reg && alloc.pos[u] != u32::MAX {
                    assert!(
                        !overlap(alloc.ranges[u], (s.from, s.to + 1)),
                        "span {k} in {:?} overlaps value {u} {:?}",
                        s.reg,
                        alloc.ranges[u]
                    );
                }
            }
            for c in &alloc.caches {
                assert!(
                    c.reg != s.reg || !overlap(c.hull, (s.from, s.to)),
                    "span {k} in a cache's register"
                );
            }
            assert!(
                alloc.cache_of(s.block, s.value).is_none(),
                "span {k} beside a loop cache of its value"
            );
            for (j, t) in alloc.spans.iter().enumerate() {
                if j != k && overlap((t.from, t.to), (s.from, s.to)) {
                    assert_ne!(t.reg, s.reg, "spans {k} and {j} share a register");
                    assert_ne!(t.value, s.value, "spans {k} and {j} copy one value");
                }
            }
            if matches!(s.reg, Home::Gp(_)) && ARM64.gp_callee.contains(&r) {
                assert!(
                    alloc.callee_saved_gp.contains(&r),
                    "span {k}'s x{r} is not saved by the prologue"
                );
            }
        }
    }

    // The copies' passes as step R1's B made them (1cff91e1: each loop's blocks scanned once per
    // loop that holds them, every block's reads for the spans, the held lists sorted, the
    // depths by a count over the loops): the reference that F5's cheaper passes must equal on
    // every test allocation (`assert_reference`).
    fn natural_loops_ref(
        graph: &Graph,
        order: &[BlockIx],
        (head, tail): (&[u32], &[u32]),
    ) -> Vec<Loop> {
        let mut place = vec![u32::MAX; graph.blocks.len()];
        for (k, &b) in order.iter().enumerate() {
            place[b.index()] = k as u32;
        }
        let reachable = |b: BlockIx| place[b.index()] != u32::MAX;
        let words = graph.blocks.len().div_ceil(64);
        let mut loops: Vec<Loop> = Vec::new();
        for &h in order {
            let data = graph.block(h);
            if !data.loop_header {
                continue;
            }
            let mut stack: Vec<BlockIx> = data
                .preds
                .iter()
                .copied()
                .filter(|&p| reachable(p) && place[p.index()] >= place[h.index()])
                .collect();
            if stack.is_empty() {
                continue;
            }
            let mut member = vec![0u64; words];
            insert(&mut member, h.index());
            let mut blocks = vec![h];
            let mut natural = true;
            while let Some(b) = stack.pop() {
                if contains(&member, b.index()) {
                    continue;
                }
                if b == order[0] {
                    natural = false;
                    break;
                }
                insert(&mut member, b.index());
                blocks.push(b);
                stack.extend(
                    graph
                        .block(b)
                        .preds
                        .iter()
                        .copied()
                        .filter(|&p| reachable(p) && !contains(&member, p.index())),
                );
            }
            if !natural {
                continue;
            }
            let mut hull = (head[h.index()], 0);
            let (mut call, mut clobbered) = (false, (0, 0));
            for &b in &blocks {
                hull.1 = hull.1.max(tail[b.index()]);
                for &node in &graph.block(b).body {
                    let op = &graph.node(node).op;
                    call |= op.is_effectful();
                    let (gp, fp) = clobbers(op);
                    clobbered = (clobbered.0 | gp, clobbered.1 | fp);
                }
            }
            loops.push(Loop {
                header: h,
                blocks,
                member,
                hull,
                depth: 0,
                call,
                clobbered,
            });
        }
        for k in 0..loops.len() {
            let h = loops[k].header.index();
            loops[k].depth = loops.iter().filter(|l| contains(&l.member, h)).count() as u32;
        }
        loops
    }

    #[allow(clippy::too_many_arguments)]
    fn loop_caches_ref(
        graph: &Graph,
        file: &RegisterFile,
        loops: &[Loop],
        depth: &[u32],
        homes: &[Home],
        edges: &EdgeInputs,
        held: &[Vec<Vec<(u32, u32)>>; 2],
        used_callee_gp: &mut Vec<u8>,
        fp_callee_across_effect: &mut bool,
    ) -> (Vec<Cache>, Vec<Vec<u16>>) {
        let spilled = |v: NodeIx| matches!(homes[v.index()], Home::Spill(_));
        // The candidates: (loop, value, weight), each read of a spilled value defined outside the
        // loop counting its block's frequency.
        let mut reads: Vec<(u32, u32, u64)> = Vec::new();
        for (k, l) in loops.iter().enumerate() {
            let outside =
                |v: NodeIx| spilled(v) && !contains(&l.member, graph.node(v).block.index());
            for &b in &l.blocks {
                let weight = FREQUENCY[(depth[b.index()] as usize).min(9)];
                let data = graph.block(b);
                for &node in &data.body {
                    for &input in &graph.node(node).inputs {
                        if outside(input) {
                            reads.push((k as u32, input.0, weight));
                        }
                    }
                }
                for s in 0..data.succs.len() {
                    for input in edges.of(b, s) {
                        if outside(input) {
                            reads.push((k as u32, input.0, weight));
                        }
                    }
                }
            }
        }
        if reads.is_empty() {
            return (Vec::new(), Vec::new());
        }
        reads.sort_unstable();
        // (benefit, loop, value): the reads' weight less one load at each entry.
        let mut candidates: Vec<(u64, u32, u32)> = Vec::new();
        for run in reads.chunk_by(|a, b| a.0 == b.0 && a.1 == b.1) {
            let (k, v, _) = run[0];
            let weight: u64 = run.iter().map(|r| r.2).sum();
            let entry = FREQUENCY[(loops[k as usize].depth as usize - 1).min(9)];
            if weight > entry {
                candidates.push((weight - entry, k, v));
            }
        }
        candidates.sort_unstable_by(|a, b| {
            b.0.cmp(&a.0)
                .then(loops[b.1 as usize].depth.cmp(&loops[a.1 as usize].depth))
                .then(a.2.cmp(&b.2))
        });
        let mut caches: Vec<Cache> = Vec::new();
        let overlap = |a: (u32, u32), b: (u32, u32)| a.0 <= b.1 && b.0 <= a.1;
        for &(_, k, v) in &candidates {
            let l = &loops[k as usize];
            let v = NodeIx(v);
            if caches
                .iter()
                .any(|c| c.value == v && overlap(c.hull, l.hull))
            {
                continue;
            }
            let Some(class) = class_of(graph.node(v).rep) else {
                continue;
            };
            let (ci, caller, callee, avoid) = match class {
                Class::Gp => (0, file.gp_caller, file.gp_callee, l.clobbered.0),
                Class::Fp => (1, file.fp_caller, file.fp_callee, l.clobbered.1),
            };
            let free = |r: u8| {
                let list = &held[ci][r as usize];
                let first = list.partition_point(|&(_, end)| end < l.hull.0);
                avoid & (1 << r) == 0
                    && list.get(first).is_none_or(|&(start, _)| start > l.hull.1)
                    && !caches.iter().any(|c| {
                        matches!((c.reg, class), (Home::Gp(q), Class::Gp) | (Home::Fp(q), Class::Fp) if q == r)
                            && overlap(c.hull, l.hull)
                    })
            };
            let choices: &[u8] = if l.call { &[] } else { caller };
            let Some(r) = choices.iter().chain(callee).copied().find(|&r| free(r)) else {
                continue;
            };
            let reg = match class {
                Class::Gp => Home::Gp(r),
                Class::Fp => Home::Fp(r),
            };
            if class == Class::Gp && callee.contains(&r) && !used_callee_gp.contains(&r) {
                used_callee_gp.push(r);
            }
            if class == Class::Fp && callee.contains(&r) && l.call {
                *fp_callee_across_effect = true;
            }
            caches.push(Cache {
                value: v,
                reg,
                header: l.header,
                hull: l.hull,
            });
        }
        if caches.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let mut block_caches: Vec<Vec<u16>> = vec![Vec::new(); graph.blocks.len()];
        for (k, c) in caches.iter().enumerate() {
            let l = loops
                .iter()
                .find(|l| l.header == c.header)
                .expect("a cache's loop");
            for &b in &l.blocks {
                block_caches[b.index()].push(k as u16);
            }
        }
        (caches, block_caches)
    }

    #[allow(clippy::too_many_arguments)]
    fn spill_copies_ref(
        graph: &Graph,
        file: &RegisterFile,
        (order, pos, head, tail): (&[BlockIx], &[u32], &[u32], &[u32]),
        homes: &[Home],
        ranges: &[(u32, u32)],
        edges: &EdgeInputs,
        used_callee_gp: &mut Vec<u8>,
        fp_callee_across_effect: &mut bool,
    ) -> Copies {
        let mut copies = Copies::default();
        if !homes.iter().any(|home| matches!(home, Home::Spill(_))) {
            return copies;
        }
        let loops = natural_loops_ref(graph, order, (head, tail));
        let mut depth = vec![0u32; graph.blocks.len()];
        for l in &loops {
            for &b in &l.blocks {
                depth[b.index()] += 1;
            }
        }
        // Each register's ranges, by start (they do not overlap: one value a register at a time).
        let mut held: [Vec<Vec<(u32, u32)>>; 2] = [vec![Vec::new(); 32], vec![Vec::new(); 32]];
        for (v, home) in homes.iter().enumerate() {
            match *home {
                Home::Gp(r) => held[0][r as usize].push(ranges[v]),
                Home::Fp(r) => held[1][r as usize].push(ranges[v]),
                _ => {}
            }
        }
        for list in held.iter_mut().flatten() {
            list.sort_unstable();
        }
        if !loops.is_empty() {
            (copies.caches, copies.block_caches) = loop_caches_ref(
                graph,
                file,
                &loops,
                &depth,
                homes,
                edges,
                &held,
                used_callee_gp,
                fp_callee_across_effect,
            );
            for c in &copies.caches {
                match c.reg {
                    Home::Gp(r) => hold(&mut held[0][r as usize], c.hull),
                    Home::Fp(r) => hold(&mut held[1][r as usize], c.hull),
                    _ => {}
                }
            }
        }
        (copies.spans, copies.block_spans) = read_spans_ref(
            graph,
            file,
            (order, pos, head),
            homes,
            &depth,
            (&copies.caches, &copies.block_caches),
            &mut held,
            used_callee_gp,
        );
        copies
    }

    #[allow(clippy::too_many_arguments)]
    fn read_spans_ref(
        graph: &Graph,
        file: &RegisterFile,
        (order, pos, head): (&[BlockIx], &[u32], &[u32]),
        homes: &[Home],
        depth: &[u32],
        (caches, block_caches): (&[Cache], &[Vec<u16>]),
        held: &mut [Vec<Vec<(u32, u32)>>; 2],
        used_callee_gp: &[u8],
    ) -> (Vec<Span>, Vec<Vec<u16>>) {
        // (block, value, the calls before it in the block, position) of each read of a spilled value
        // no loop cache holds there: a run is one block, value and call count.
        let mut reads: Vec<(u32, u32, u32, u32)> = Vec::new();
        for &b in order {
            let cached = |v: NodeIx| {
                block_caches
                    .get(b.index())
                    .is_some_and(|ks| ks.iter().any(|&k| caches[k as usize].value == v))
            };
            let mut calls = 0u32;
            for &node in &graph.block(b).body {
                let p = pos[node.index()];
                for &input in &graph.node(node).inputs {
                    if matches!(homes[input.index()], Home::Spill(_)) && !cached(input) {
                        reads.push((b.0, input.0, calls, p));
                    }
                }
                if graph.node(node).op.is_effectful() {
                    calls += 1;
                }
            }
        }
        if reads.len() < 2 {
            return (Vec::new(), Vec::new());
        }
        reads.sort_unstable();
        // (weight, first, end) of each run of two reads or more, as indices into `reads`.
        let mut runs: Vec<(u64, u32, u32)> = Vec::new();
        let mut first = 0;
        while first < reads.len() {
            let key = |k: usize| (reads[k].0, reads[k].1, reads[k].2);
            let mut end = first + 1;
            while end < reads.len() && key(end) == key(first) {
                end += 1;
            }
            if end - first >= 2 {
                let frequency = FREQUENCY[(depth[reads[first].0 as usize] as usize).min(9)];
                runs.push((
                    (end - first - 1) as u64 * frequency,
                    first as u32,
                    end as u32,
                ));
            }
            first = end;
        }
        runs.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        // The registers a span may take, by class: caller-saved, then the callee-saved x registers
        // the prologue saves already.
        let gp_regs: Vec<u8> = file
            .gp_caller
            .iter()
            .chain(file.gp_callee.iter().filter(|r| used_callee_gp.contains(r)))
            .copied()
            .collect();
        let mut spans: Vec<Span> = Vec::new();
        for &(_, first, end) in &runs {
            let run = &reads[first as usize..end as usize];
            let (block, v) = (BlockIx(run[0].0), NodeIx(run[0].1));
            let Some(class) = class_of(graph.node(v).rep) else {
                continue;
            };
            let (ci, regs): (usize, &[u8]) = match class {
                Class::Gp => (0, &gp_regs),
                Class::Fp => (1, file.fp_caller),
            };
            // The registers of the class free from position `p` to `q` in the block: no range in
            // them overlaps, and no stub call of the block's nodes there clobbers them.
            let body = &graph.block(block).body;
            let base = head[block.index()] + 2;
            let free = |held: &[Vec<Vec<(u32, u32)>>; 2], p: u32, q: u32| {
                let clobbered = body[((p - base) / 2) as usize..=((q - base) / 2) as usize]
                    .iter()
                    .fold(0u64, |acc, &n| {
                        let (gp, fp) = clobbers(&graph.node(n).op);
                        acc | if ci == 0 { gp } else { fp }
                    });
                regs.iter().fold(0u64, |mask, &r| {
                    if clobbered & (1 << r) == 0 && free_over(&held[ci][r as usize], (p, q)) {
                        mask | (1 << r)
                    } else {
                        mask
                    }
                })
            };
            // Segments of the run, each the longest from its first read through which a register
            // stays free, up to the definition of its last reading node's value (to + 1), so that the
            // span's register is no home its readers write: a node may write its output before it
            // reads all its inputs, and a cold path's restore would put the copy back over it.
            let mut take = |held: &mut [Vec<Vec<(u32, u32)>>; 2], from: u32, to: u32, mask: u64| {
                let r = regs
                    .iter()
                    .copied()
                    .find(|&r| mask & (1 << r) != 0)
                    .expect("a free register");
                hold(&mut held[ci][r as usize], (from, to));
                spans.push(Span {
                    value: v,
                    reg: if ci == 0 { Home::Gp(r) } else { Home::Fp(r) },
                    block,
                    from,
                    to,
                });
            };
            let mut start = 0;
            let mut mask = 0u64;
            for k in 1..run.len() {
                let step = free(held, run[k - 1].3, run[k].3 + 1);
                let next = if k - 1 == start { step } else { mask & step };
                if next != 0 {
                    mask = next;
                    continue;
                }
                if k - 1 > start {
                    take(held, run[start].3, run[k - 1].3, mask);
                }
                start = k;
                mask = 0;
            }
            if run.len() - 1 > start {
                take(held, run[start].3, run[run.len() - 1].3, mask);
            }
        }
        if spans.is_empty() {
            return (Vec::new(), Vec::new());
        }
        spans.sort_unstable_by_key(|s| (s.block, s.from, s.value));
        let mut block_spans: Vec<Vec<u16>> = vec![Vec::new(); graph.blocks.len()];
        for (k, s) in spans.iter().enumerate() {
            block_spans[s.block.index()].push(k as u16);
        }
        (spans, block_spans)
    }

    /// F5's copies equal the reference's (`spill_copies_ref`, B's passes) for this allocation:
    /// the same loop caches and read spans and the same callee-saved x registers.
    pub(crate) fn assert_reference(graph: &Graph, file: &RegisterFile, alloc: &Allocation) {
        let order = &alloc.order;
        let mut head = vec![u32::MAX; graph.blocks.len()];
        let mut tail = vec![u32::MAX; graph.blocks.len()];
        let mut next = 0u32;
        for &b in order {
            head[b.index()] = next;
            next += 2 + 2 * graph.block(b).body.len() as u32;
            tail[b.index()] = next - 1;
        }
        let reads = Reads::new(graph);
        let edges = EdgeInputs::new(graph, order, &reads).expect("the allocation's edges");
        let mut used: Vec<u8> = file
            .gp_callee
            .iter()
            .copied()
            .filter(|&r| alloc.homes.contains(&Home::Gp(r)))
            .collect();
        let mut fp = false;
        let copies = spill_copies_ref(
            graph,
            file,
            (order, &alloc.pos, &head, &tail),
            &alloc.homes,
            &alloc.ranges,
            &edges,
            &mut used,
            &mut fp,
        );
        used.sort_unstable();
        assert_eq!(
            copies.caches, alloc.caches,
            "the loop caches differ from B's"
        );
        assert_eq!(
            copies.block_caches, alloc.block_caches,
            "the blocks' caches differ"
        );
        assert_eq!(copies.spans, alloc.spans, "the read spans differ from B's");
        assert_eq!(
            copies.block_spans, alloc.block_spans,
            "the blocks' spans differ"
        );
        assert_eq!(
            used, alloc.callee_saved_gp,
            "the callee-saved x registers differ"
        );
    }

    /// `live_at` (its index) and `live_at_scan` at each node the allocation scheduled.
    pub(crate) fn assert_live_at_agrees(graph: &Graph, alloc: &Allocation) -> usize {
        let mut values = 0;
        for v in 0..graph.nodes.len() {
            if alloc.pos[v] == u32::MAX {
                continue;
            }
            let node = NodeIx(v as u32);
            let live = alloc.live_at(node);
            assert_eq!(live, alloc.live_at_scan(node), "live at node {v}");
            values += live.len();
        }
        values
    }

    #[test]
    fn values_get_registers_and_die_where_their_ranges_end() {
        // a + b, then the sum boxed: a and b die at the add, whose result may take a's register.
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        g.entry = Some(entry);
        let a = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let b = g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
        let s = state(&mut g, 2, &[a, b]);
        let fa = g.add_node(entry, NodeOp::CheckNumber, &[a], Rep::F64, 2);
        g.node_mut(fa).state = Some(s);
        let fb = g.add_node(entry, NodeOp::CheckNumber, &[b], Rep::F64, 2);
        g.node_mut(fb).state = Some(s);
        let sum = g.add_node(entry, NodeOp::F64Add, &[fa, fb], Rep::F64, 2);
        let boxed = g.add_node(entry, NodeOp::F64ToTagged, &[sum], Rep::Tagged, 2);
        g.add_node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert_eq!(alloc.homes[a.index()], Home::Gp(0));
        assert_eq!(alloc.homes[b.index()], Home::Gp(1));
        // The checks keep both tagged words held while they can exit.
        assert_eq!(alloc.homes[fa.index()], Home::Fp(0));
        assert_eq!(alloc.homes[fb.index()], Home::Fp(1));
        // fa and fb die at the add: its result reuses d0.
        assert_eq!(alloc.homes[sum.index()], Home::Fp(0));
        assert_eq!(alloc.homes[boxed.index()], Home::Gp(0));
        assert_eq!(alloc.spill_bytes, 0);
        assert!(alloc.callee_saved_gp.is_empty());
    }

    #[test]
    fn a_value_live_across_an_effect_gets_a_callee_saved_home() {
        let mut g = Graph::new(1);
        let entry = g.add_block(0, false);
        g.entry = Some(entry);
        let a = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let s = state(&mut g, 1, &[a]);
        let call = g.add_node(
            entry,
            NodeOp::Effect {
                pc: 1,
                pops: 0,
                pushes: 1,
            },
            &[],
            Rep::None,
            1,
        );
        g.node_mut(call).state = Some(s);
        let r = g.add_node(entry, NodeOp::Projection(0), &[call], Rep::Tagged, 1);
        let same = g.add_node(entry, NodeOp::TaggedSame, &[a, r], Rep::Bool, 2);
        let boxed = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 2);
        g.add_node(entry, NodeOp::Return, &[boxed], Rep::None, 3);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert_eq!(alloc.homes[a.index()], Home::Gp(23));
        assert_eq!(alloc.callee_saved_gp, vec![23]);
        // The projection is defined after the effect: a caller-saved register will do.
        assert_eq!(alloc.homes[r.index()], Home::Gp(0));
        assert_eq!(alloc.live_at(call), vec![a]);
    }

    #[test]
    fn values_live_across_a_typed_element_call_take_no_register_its_stub_clobbers() {
        // Fourteen words and three f64s live across a TypedElement call, whose stub clobbers
        // x11-x15, d1 and d2 (`clobbers`): taken lowest first, they would sit in x2-x8, x10-x15
        // and x23 and in d0-d2, and the call would have to save them.
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        g.entry = Some(entry);
        let a = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let i = g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
        let s = state(&mut g, 1, &[a, i]);
        let exiting = |g: &mut Graph, op: NodeOp, inputs: &[NodeIx], rep: Rep| {
            let n = g.add_node(entry, op, inputs, rep, 1);
            g.node_mut(n).state = Some(s);
            n
        };
        let words: Vec<NodeIx> = (0..14)
            .map(|_| g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0))
            .collect();
        let floats: Vec<NodeIx> = words[..3]
            .iter()
            .map(|&w| exiting(&mut g, NodeOp::CheckNumber, &[w], Rep::F64))
            .collect();
        let cell = exiting(&mut g, NodeOp::CheckObject, &[a], Rep::Cell);
        let number = exiting(&mut g, NodeOp::CheckNumber, &[i], Rep::F64);
        let index = exiting(&mut g, NodeOp::CheckIndex, &[number], Rep::I32);
        let element = exiting(
            &mut g,
            NodeOp::TypedElement { write: false },
            &[cell, index],
            Rep::Tagged,
        );
        let mut acc = element;
        for &w in &words {
            let same = g.add_node(entry, NodeOp::TaggedSame, &[acc, w], Rep::Bool, 2);
            acc = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 2);
        }
        let mut sum = floats[0];
        for &f in &floats[1..] {
            sum = g.add_node(entry, NodeOp::F64Add, &[sum, f], Rep::F64, 2);
        }
        let boxed = g.add_node(entry, NodeOp::F64ToTagged, &[sum], Rep::Tagged, 2);
        let same = g.add_node(entry, NodeOp::TaggedSame, &[acc, boxed], Rep::Bool, 2);
        let result = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 2);
        g.add_node(entry, NodeOp::Return, &[result], Rep::None, 3);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        let (gp, fp) = clobbers(&NodeOp::TypedElement { write: false });
        for &v in words.iter().chain(&floats) {
            match alloc.homes[v.index()] {
                Home::Gp(r) => assert_eq!(gp & (1 << r), 0, "word {v:?} in x{r}"),
                Home::Fp(r) => assert_eq!(fp & (1 << r), 0, "f64 {v:?} in d{r}"),
                home => panic!("value {v:?} in {home:?}"),
            }
        }
        // The rule binds only values across the call: its own inputs still take those registers.
        assert!(
            [cell, index]
                .iter()
                .any(|v| matches!(alloc.homes[v.index()], Home::Gp(r) if gp & (1 << r) != 0)),
            "{:?} {:?}",
            alloc.homes[cell.index()],
            alloc.homes[index.index()]
        );
        // Six words beyond x0-x8 and x10 (two hold `a` and `i` until the call): callee-saved.
        assert!(!alloc.callee_saved_gp.is_empty());
        assert_eq!(alloc.spill_bytes, 0);
    }

    #[test]
    fn a_back_edges_value_across_a_typed_element_call_skips_its_phis_clobbered_register() {
        // entry: eleven words read after the loop (x0-x8, x10, x23 through it), a's cell; jump
        // header.
        // header: p = phi(0, next); poll; branch p < 100. body: next = p | 1; TypedElement(cell,
        // next); jump header. exit: the words. `p`, read only before the call, takes x11 and
        // hints it to `next`, its back edge's value, which lives across the call: not there.
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        let header = g.add_block(1, true);
        let body = g.add_block(2, false);
        let exit = g.add_block(3, false);
        g.entry = Some(entry);
        g.add_edge(entry, header);
        g.add_edge(header, body);
        g.add_edge(header, exit);
        g.add_edge(body, header);
        let words: Vec<NodeIx> = (0..11)
            .map(|_| g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0))
            .collect();
        let a = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let s = state(&mut g, 0, &[]);
        let cell = g.add_node(entry, NodeOp::CheckObject, &[a], Rep::Cell, 0);
        g.node_mut(cell).state = Some(s);
        let zero = g.add_node(entry, NodeOp::ConstI32(0), &[], Rep::I32, 0);
        g.add_node(entry, NodeOp::Jump(header), &[], Rep::None, 0);
        let p = g.add_node(header, NodeOp::Phi, &[zero], Rep::I32, 1);
        let ps = state(&mut g, 1, &[]);
        let poll = g.add_node(header, NodeOp::Poll, &[], Rep::None, 1);
        g.node_mut(poll).state = Some(ps);
        let hundred = g.add_node(header, NodeOp::ConstI32(100), &[], Rep::I32, 1);
        let lt = g.add_node(
            header,
            NodeOp::I32Cmp(Cond::Lt),
            &[p, hundred],
            Rep::Bool,
            1,
        );
        g.add_node(
            header,
            NodeOp::Branch {
                then_block: body,
                else_block: exit,
            },
            &[lt],
            Rep::None,
            1,
        );
        let one = g.add_node(body, NodeOp::ConstI32(1), &[], Rep::I32, 2);
        let next = g.add_node(body, NodeOp::I32Or, &[p, one], Rep::I32, 2);
        let bs = state(&mut g, 2, &[]);
        let te = g.add_node(
            body,
            NodeOp::TypedElement { write: false },
            &[cell, next],
            Rep::Tagged,
            2,
        );
        g.node_mut(te).state = Some(bs);
        g.node_mut(p).inputs.push(next);
        g.add_node(body, NodeOp::Jump(header), &[], Rep::None, 2);
        let mut acc = words[0];
        for &w in &words[1..] {
            let same = g.add_node(exit, NodeOp::TaggedSame, &[acc, w], Rep::Bool, 3);
            acc = g.add_node(exit, NodeOp::BoolToTagged, &[same], Rep::Tagged, 3);
        }
        g.add_node(exit, NodeOp::Return, &[acc], Rep::None, 3);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        let (gp, _) = clobbers(&NodeOp::TypedElement { write: false });
        assert!(
            matches!(alloc.homes[p.index()], Home::Gp(r) if gp & (1 << r) != 0),
            "the phi in {:?}",
            alloc.homes[p.index()]
        );
        assert_ne!(alloc.homes[next.index()], alloc.homes[p.index()]);
    }

    #[test]
    fn a_value_across_a_typed_element_call_takes_no_victims_clobbered_register() {
        // Every x register holds a word when `v`, read after a TypedElement call, is defined:
        // ten read soon (x0-x8 and x10; x9 is the code generator's), five read late but before
        // the call (x11-x15), six read soon (x23-x28). The only words read later than `v` sit
        // in registers the call clobbers, so no victim gives `v` its register: `v` spills.
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        g.entry = Some(entry);
        let word = |g: &mut Graph| g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
        let soon: Vec<NodeIx> = (0..10).map(|_| word(&mut g)).collect();
        let late: Vec<NodeIx> = (0..5).map(|_| word(&mut g)).collect();
        let soon2: Vec<NodeIx> = (0..6).map(|_| word(&mut g)).collect();
        let v = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let chain = |g: &mut Graph, mut acc: NodeIx, words: &[NodeIx]| {
            for &w in words {
                let same = g.add_node(entry, NodeOp::TaggedSame, &[acc, w], Rep::Bool, 1);
                acc = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 1);
            }
            acc
        };
        let acc = chain(&mut g, soon[0], &soon[1..]);
        let acc = chain(&mut g, acc, &soon2);
        let acc = chain(&mut g, acc, &[v]);
        let acc = chain(&mut g, acc, &late);
        let s = state(&mut g, 1, &[]);
        let exiting = |g: &mut Graph, op: NodeOp, inputs: &[NodeIx], rep: Rep| {
            let n = g.add_node(entry, op, inputs, rep, 1);
            g.node_mut(n).state = Some(s);
            n
        };
        let cell = exiting(&mut g, NodeOp::CheckObject, &[acc], Rep::Cell);
        let number = exiting(&mut g, NodeOp::CheckNumber, &[acc], Rep::F64);
        let index = exiting(&mut g, NodeOp::CheckIndex, &[number], Rep::I32);
        let te = exiting(
            &mut g,
            NodeOp::TypedElement { write: false },
            &[cell, index],
            Rep::Tagged,
        );
        let last = chain(&mut g, te, &[v]);
        g.add_node(entry, NodeOp::Return, &[last], Rep::None, 2);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        let (gp, _) = clobbers(&NodeOp::TypedElement { write: false });
        for &w in &late {
            assert!(
                matches!(alloc.homes[w.index()], Home::Gp(r) if gp & (1 << r) != 0),
                "a late word in {:?}",
                alloc.homes[w.index()]
            );
        }
        assert!(
            matches!(alloc.homes[v.index()], Home::Spill(_)),
            "v in {:?}",
            alloc.homes[v.index()]
        );
    }

    #[test]
    fn values_spill_at_their_definition_when_no_register_is_free() {
        // Sixteen entry slots live to one return: fifteen caller-saved and six callee-saved x
        // registers hold twenty-one; the rest spill.
        let count = 24u16;
        let mut g = Graph::new(count as usize);
        let entry = g.add_block(0, false);
        g.entry = Some(entry);
        let slots: Vec<NodeIx> = (0..count)
            .map(|k| g.add_node(entry, NodeOp::EntrySlot(k), &[], Rep::Tagged, 0))
            .collect();
        let mut acc = slots[0];
        for &v in &slots[1..] {
            let same = g.add_node(entry, NodeOp::TaggedSame, &[acc, v], Rep::Bool, 1);
            acc = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 1);
        }
        // Keep every slot live to the end.
        let s = state(&mut g, 2, &slots);
        let exit = g.add_node(
            entry,
            NodeOp::Exit(ExitReason::Unsupported),
            &[],
            Rep::None,
            2,
        );
        g.node_mut(exit).state = Some(s);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        // The slots, read only at the end, give their registers up to the accumulator's
        // values, each read at once: no accumulator value spills.
        let spilled = slots
            .iter()
            .filter(|v| matches!(alloc.homes[v.index()], Home::Spill(_)))
            .count();
        assert!(spilled >= count as usize - 21, "{spilled} slots spilled");
        assert!(g.nodes.iter().enumerate().all(|(v, node)| {
            node.op != NodeOp::BoolToTagged || matches!(alloc.homes[v], Home::Gp(_) | Home::None)
        }));
        assert!(alloc.spill_bytes >= 8 * spilled as u32);
    }

    #[test]
    fn a_loop_keeps_values_read_inside_it_until_its_back_edge() {
        // entry: n = slot 0; jump header. header: i = phi(0, i'); poll; branch i < n.
        // body: i' = i + 1; jump header. exit: return.
        let mut g = Graph::new(1);
        let entry = g.add_block(0, false);
        let header = g.add_block(1, true);
        let body = g.add_block(2, false);
        let exit = g.add_block(3, false);
        g.entry = Some(entry);
        g.add_edge(entry, header);
        g.add_edge(header, body);
        g.add_edge(header, exit);
        g.add_edge(body, header);
        let n = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let s = state(&mut g, 0, &[]);
        let fn_ = g.add_node(entry, NodeOp::CheckNumber, &[n], Rep::F64, 0);
        g.node_mut(fn_).state = Some(s);
        let zero = g.add_node(entry, NodeOp::ConstF64(0), &[], Rep::F64, 0);
        g.add_node(entry, NodeOp::Jump(header), &[], Rep::None, 0);
        let i = g.add_node(header, NodeOp::Phi, &[zero], Rep::F64, 1);
        let ps = state(&mut g, 1, &[]);
        let poll = g.add_node(header, NodeOp::Poll, &[], Rep::None, 1);
        g.node_mut(poll).state = Some(ps);
        let lt = g.add_node(header, NodeOp::F64Cmp(Cond::Lt), &[i, fn_], Rep::Bool, 1);
        g.add_node(
            header,
            NodeOp::Branch {
                then_block: body,
                else_block: exit,
            },
            &[lt],
            Rep::None,
            1,
        );
        let one = g.add_node(body, NodeOp::ConstF64(1f64.to_bits()), &[], Rep::F64, 2);
        let next = g.add_node(body, NodeOp::F64Add, &[i, one], Rep::F64, 2);
        g.node_mut(i).inputs.push(next);
        g.add_node(body, NodeOp::Jump(header), &[], Rep::None, 2);
        let boxed = g.add_node(exit, NodeOp::F64ToTagged, &[i], Rep::Tagged, 3);
        g.add_node(exit, NodeOp::Return, &[boxed], Rep::None, 3);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        let (_, n_end) = alloc.ranges[fn_.index()];
        let body_end = alloc.pos[g.block(body).body.last().unwrap().index()] + 1;
        assert!(n_end >= body_end, "n lives through the back edge");
        assert_ne!(alloc.homes[fn_.index()], alloc.homes[i.index()]);
        assert_ne!(alloc.homes[fn_.index()], alloc.homes[next.index()]);
        assert_eq!(alloc.homes[zero.index()], Home::Const);
        assert_eq!(alloc.live_at(poll), vec![fn_, i]);
    }

    /// The index behind `live_at` (stage 2, step 10) answers as the scan it replaced: values in
    /// caller-saved and callee-saved registers and in spill slots, ranges that end at the node
    /// and ranges that start there, across an effect and through a loop.
    #[test]
    fn live_at_answers_as_a_scan_of_every_range() {
        let count = 24u16;
        let mut g = Graph::new(count as usize);
        let entry = g.add_block(0, false);
        let header = g.add_block(1, true);
        let body = g.add_block(2, false);
        let exit = g.add_block(3, false);
        g.entry = Some(entry);
        g.add_edge(entry, header);
        g.add_edge(header, body);
        g.add_edge(header, exit);
        g.add_edge(body, header);
        let slots: Vec<NodeIx> = (0..count)
            .map(|k| g.add_node(entry, NodeOp::EntrySlot(k), &[], Rep::Tagged, 0))
            .collect();
        let s = state(&mut g, 0, &[]);
        let limit = g.add_node(entry, NodeOp::CheckNumber, &[slots[0]], Rep::F64, 0);
        g.node_mut(limit).state = Some(s);
        let zero = g.add_node(entry, NodeOp::ConstF64(0), &[], Rep::F64, 0);
        g.add_node(entry, NodeOp::Jump(header), &[], Rep::None, 0);
        let i = g.add_node(header, NodeOp::Phi, &[zero], Rep::F64, 1);
        let ps = state(&mut g, 1, &[]);
        let poll = g.add_node(header, NodeOp::Poll, &[], Rep::None, 1);
        g.node_mut(poll).state = Some(ps);
        let lt = g.add_node(header, NodeOp::F64Cmp(Cond::Lt), &[i, limit], Rep::Bool, 1);
        g.add_node(
            header,
            NodeOp::Branch {
                then_block: body,
                else_block: exit,
            },
            &[lt],
            Rep::None,
            1,
        );
        // The body's effect keeps every slot across it.
        let es = state(&mut g, 2, &slots);
        let call = g.add_node(
            body,
            NodeOp::Effect {
                pc: 2,
                pops: 0,
                pushes: 0,
            },
            &[],
            Rep::None,
            2,
        );
        g.node_mut(call).state = Some(es);
        let one = g.add_node(body, NodeOp::ConstF64(1f64.to_bits()), &[], Rep::F64, 2);
        let next = g.add_node(body, NodeOp::F64Add, &[i, one], Rep::F64, 2);
        g.node_mut(i).inputs.push(next);
        g.add_node(body, NodeOp::Jump(header), &[], Rep::None, 2);
        let mut acc = slots[1];
        for &v in &slots[2..] {
            let same = g.add_node(exit, NodeOp::TaggedSame, &[acc, v], Rep::Bool, 3);
            acc = g.add_node(exit, NodeOp::BoolToTagged, &[same], Rep::Tagged, 3);
        }
        g.add_node(exit, NodeOp::Return, &[acc], Rep::None, 3);
        let alloc = allocate(&g, &ARM64).unwrap();
        let homes = |want: fn(Home) -> bool| alloc.homes.iter().any(|&home| want(home));
        assert!(
            homes(|home| matches!(home, Home::Spill(_))),
            "some value spills"
        );
        assert!(homes(
            |home| matches!(home, Home::Gp(r) if ARM64.gp_callee.contains(&r))
        ));
        assert!(homes(
            |home| matches!(home, Home::Gp(r) if ARM64.gp_caller.contains(&r))
        ));
        assert!(assert_live_at_agrees(&g, &alloc) > 0);
        assert_consistent(&g, &alloc);
        // The effect's state reads every slot.
        let mut live = alloc.live_at(call);
        live.retain(|v| slots.contains(v));
        assert_eq!(live, slots);
    }

    /// A loop over values spilled before it (step R1's tests): the entry defines 21 words and
    /// then `values`, which the loop reads; the words fill every x register and die in a chain
    /// before the loop, and each value's next use, in the loop, is farther than theirs, so each
    /// spills at its definition; then `keep` words read only after the loop, each holding a
    /// register through it. header: q = phi(acc, q'), i = phi(0, i'); poll; branch i < 100.
    /// body: `body` builds from (q, the values) the word the back edge gives q; i' = i | 1; jump
    /// header. exit: the kept words and return q.
    struct Looped {
        g: Graph,
        entry: BlockIx,
        header: BlockIx,
        body: BlockIx,
        values: Vec<NodeIx>,
    }

    fn looped(
        values: usize,
        keep: usize,
        body: impl FnOnce(&mut Graph, BlockIx, NodeIx, &[NodeIx]) -> NodeIx,
    ) -> Looped {
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        let header = g.add_block(1, true);
        let body_block = g.add_block(2, false);
        let exit = g.add_block(3, false);
        g.entry = Some(entry);
        g.add_edge(entry, header);
        g.add_edge(header, body_block);
        g.add_edge(header, exit);
        g.add_edge(body_block, header);
        let word = |g: &mut Graph| g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
        let words: Vec<NodeIx> = (0..21).map(|_| word(&mut g)).collect();
        let vs: Vec<NodeIx> = (0..values).map(|_| word(&mut g)).collect();
        let mut acc = words[0];
        for &w in &words[1..] {
            let same = g.add_node(entry, NodeOp::TaggedSame, &[acc, w], Rep::Bool, 0);
            acc = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 0);
        }
        // Defined once the chain freed the registers: each holds one through the loop.
        let kept: Vec<NodeIx> = (0..keep).map(|_| word(&mut g)).collect();
        let zero = g.add_node(entry, NodeOp::ConstI32(0), &[], Rep::I32, 0);
        g.add_node(entry, NodeOp::Jump(header), &[], Rep::None, 0);
        let q = g.add_node(header, NodeOp::Phi, &[acc], Rep::Tagged, 1);
        let i = g.add_node(header, NodeOp::Phi, &[zero], Rep::I32, 1);
        let ps = state(&mut g, 1, &[]);
        let poll = g.add_node(header, NodeOp::Poll, &[], Rep::None, 1);
        g.node_mut(poll).state = Some(ps);
        let hundred = g.add_node(header, NodeOp::ConstI32(100), &[], Rep::I32, 1);
        let lt = g.add_node(
            header,
            NodeOp::I32Cmp(Cond::Lt),
            &[i, hundred],
            Rep::Bool,
            1,
        );
        g.add_node(
            header,
            NodeOp::Branch {
                then_block: body_block,
                else_block: exit,
            },
            &[lt],
            Rep::None,
            1,
        );
        let q2 = body(&mut g, body_block, q, &vs);
        let one = g.add_node(body_block, NodeOp::ConstI32(1), &[], Rep::I32, 2);
        let i2 = g.add_node(body_block, NodeOp::I32Or, &[i, one], Rep::I32, 2);
        g.add_node(body_block, NodeOp::Jump(header), &[], Rep::None, 2);
        g.node_mut(q).inputs.push(q2);
        g.node_mut(i).inputs.push(i2);
        let mut out = q;
        for &k in &kept {
            let same = g.add_node(exit, NodeOp::TaggedSame, &[out, k], Rep::Bool, 3);
            out = g.add_node(exit, NodeOp::BoolToTagged, &[same], Rep::Tagged, 3);
        }
        g.add_node(exit, NodeOp::Return, &[out], Rep::None, 3);
        Looped {
            g,
            entry,
            header,
            body: body_block,
            values: vs,
        }
    }

    /// `acc` compared with each of `values` `times` times in `b`, as a chain of words.
    fn compare_each(
        g: &mut Graph,
        b: BlockIx,
        mut acc: NodeIx,
        values: &[NodeIx],
        times: usize,
    ) -> NodeIx {
        for &v in values {
            for _ in 0..times {
                let same = g.add_node(b, NodeOp::TaggedSame, &[acc, v], Rep::Bool, 2);
                acc = g.add_node(b, NodeOp::BoolToTagged, &[same], Rep::Tagged, 2);
            }
        }
        acc
    }

    #[test]
    fn a_spilled_invariant_read_in_a_call_free_loop_is_loaded_once_into_a_register() {
        // One value, spilled at its definition, read in the loop's body: a caller-saved cache
        // for the loop, valid in its header and body only, which its back edge enters after the
        // loads and its entry before them.
        let l = looped(1, 0, |g, b, q, vs| compare_each(g, b, q, vs, 1));
        let alloc = allocate(&l.g, &ARM64).unwrap();
        assert_consistent(&l.g, &alloc);
        let v = l.values[0];
        assert!(
            matches!(alloc.homes[v.index()], Home::Spill(_)),
            "{:?}",
            alloc.homes[v.index()]
        );
        assert_eq!(alloc.caches.len(), 1, "{:?}", alloc.caches);
        let cache = alloc.caches[0];
        assert_eq!((cache.value, cache.header), (v, l.header));
        let Home::Gp(r) = cache.reg else {
            panic!("{cache:?}")
        };
        assert!(ARM64.gp_caller.contains(&r), "x{r}");
        assert_eq!(alloc.cache_of(l.body, v), Some(cache.reg));
        assert_eq!(alloc.cache_of(l.header, v), Some(cache.reg));
        assert_eq!(alloc.cache_of(l.entry, v), None);
        assert!(alloc.enters_loaded(l.body, l.header));
        assert!(!alloc.enters_loaded(l.entry, l.header));
    }

    #[test]
    fn a_spilled_invariant_through_a_loop_with_a_call_takes_a_callee_saved_cache_or_none() {
        // The body calls (an effect) before it reads the value: only a callee-saved register
        // survives the call, and the prologue saves it.
        let with_call = |g: &mut Graph, b: BlockIx, q: NodeIx, vs: &[NodeIx]| {
            let s = state(g, 2, &[q]);
            let call = g.add_node(
                b,
                NodeOp::Effect {
                    pc: 2,
                    pops: 0,
                    pushes: 0,
                },
                &[],
                Rep::None,
                2,
            );
            g.node_mut(call).state = Some(s);
            compare_each(g, b, q, vs, 1)
        };
        let l = looped(1, 0, with_call);
        let alloc = allocate(&l.g, &ARM64).unwrap();
        assert_consistent(&l.g, &alloc);
        let v = l.values[0];
        assert!(matches!(alloc.homes[v.index()], Home::Spill(_)));
        let [cache] = alloc.caches[..] else {
            panic!("{:?}", alloc.caches)
        };
        let Home::Gp(r) = cache.reg else {
            panic!("{cache:?}")
        };
        assert!(ARM64.gp_callee.contains(&r), "x{r}");
        assert!(alloc.callee_saved_gp.contains(&r));
        // Six words read after the loop take every callee-saved register through it: no cache.
        let l = looped(1, 6, with_call);
        let alloc = allocate(&l.g, &ARM64).unwrap();
        assert_consistent(&l.g, &alloc);
        assert!(matches!(alloc.homes[l.values[0].index()], Home::Spill(_)));
        assert!(alloc.caches.is_empty(), "{:?}", alloc.caches);
    }

    /// Nested loops over a value spilled before them: entry (21 words in a chain, v, `keep` words
    /// read after the loops) -> H1 (q1, i; poll; i < 10) -> B1 (an effect where `call`) -> H2 (q2,
    /// j; poll; j < 10) -> B2 (q2 compared with v; j | 1) -> H2; H2 -> L1 (i | 1) -> H1; H1 -> X.
    fn nested(call: bool, keep: usize) -> (Graph, [BlockIx; 4], NodeIx) {
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        let h1 = g.add_block(1, true);
        let b1 = g.add_block(2, false);
        let h2 = g.add_block(3, true);
        let b2 = g.add_block(4, false);
        let l1 = g.add_block(5, false);
        let x = g.add_block(6, false);
        g.entry = Some(entry);
        for (a, b) in [
            (entry, h1),
            (h1, b1),
            (h1, x),
            (b1, h2),
            (h2, b2),
            (h2, l1),
            (b2, h2),
            (l1, h1),
        ] {
            g.add_edge(a, b);
        }
        let word = |g: &mut Graph| g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
        let words: Vec<NodeIx> = (0..21).map(|_| word(&mut g)).collect();
        let v = word(&mut g);
        let mut acc = words[0];
        for &w in &words[1..] {
            let same = g.add_node(entry, NodeOp::TaggedSame, &[acc, w], Rep::Bool, 0);
            acc = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 0);
        }
        let kept: Vec<NodeIx> = (0..keep).map(|_| word(&mut g)).collect();
        let zero = g.add_node(entry, NodeOp::ConstI32(0), &[], Rep::I32, 0);
        let one = g.add_node(entry, NodeOp::ConstI32(1), &[], Rep::I32, 0);
        let ten = g.add_node(entry, NodeOp::ConstI32(10), &[], Rep::I32, 0);
        g.add_node(entry, NodeOp::Jump(h1), &[], Rep::None, 0);
        let header = |g: &mut Graph, h: BlockIx, body: BlockIx, out: BlockIx, init: NodeIx| {
            let q = g.add_node(h, NodeOp::Phi, &[init], Rep::Tagged, 1);
            let i = g.add_node(h, NodeOp::Phi, &[zero], Rep::I32, 1);
            let ps = state(g, 1, &[]);
            let poll = g.add_node(h, NodeOp::Poll, &[], Rep::None, 1);
            g.node_mut(poll).state = Some(ps);
            let lt = g.add_node(h, NodeOp::I32Cmp(Cond::Lt), &[i, ten], Rep::Bool, 1);
            g.add_node(
                h,
                NodeOp::Branch {
                    then_block: body,
                    else_block: out,
                },
                &[lt],
                Rep::None,
                1,
            );
            (q, i)
        };
        let (q1, i) = header(&mut g, h1, b1, x, acc);
        if call {
            let s = state(&mut g, 2, &[q1]);
            let effect = g.add_node(
                b1,
                NodeOp::Effect {
                    pc: 2,
                    pops: 0,
                    pushes: 0,
                },
                &[],
                Rep::None,
                2,
            );
            g.node_mut(effect).state = Some(s);
        }
        g.add_node(b1, NodeOp::Jump(h2), &[], Rep::None, 2);
        let (q2, j) = header(&mut g, h2, b2, l1, q1);
        let q2b = compare_each(&mut g, b2, q2, &[v], 2);
        let j2 = g.add_node(b2, NodeOp::I32Or, &[j, one], Rep::I32, 4);
        g.add_node(b2, NodeOp::Jump(h2), &[], Rep::None, 4);
        g.node_mut(q2).inputs.push(q2b);
        g.node_mut(j).inputs.push(j2);
        let i2 = g.add_node(l1, NodeOp::I32Or, &[i, one], Rep::I32, 5);
        g.add_node(l1, NodeOp::Jump(h1), &[], Rep::None, 5);
        g.node_mut(q1).inputs.push(q2);
        g.node_mut(i).inputs.push(i2);
        let out = compare_each(&mut g, x, q1, &kept, 1);
        g.add_node(x, NodeOp::Return, &[out], Rep::None, 6);
        (g, [h1, b1, h2, b2], v)
    }

    #[test]
    fn nested_loops_cache_a_value_through_the_outermost_loop_a_register_is_free_through() {
        // No call: the outer loop's cache serves the inner loop's reads too, loaded at the outer
        // header (its weight counts the inner reads and its entries are rarer).
        let (g, [h1, b1, h2, b2], v) = nested(false, 0);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert!(matches!(alloc.homes[v.index()], Home::Spill(_)));
        let [cache] = alloc.caches[..] else {
            panic!("{:?}", alloc.caches)
        };
        assert_eq!(cache.header, h1);
        for b in [h1, b1, h2, b2] {
            assert_eq!(alloc.cache_of(b, v), Some(cache.reg), "{b:?}");
        }
        // The outer loop calls and six words read after the loops hold its callee-saved
        // registers: the inner, call-free loop takes the cache, a caller-saved register.
        let (g, [h1, b1, h2, b2], v) = nested(true, 6);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert!(matches!(alloc.homes[v.index()], Home::Spill(_)));
        let [cache] = alloc.caches[..] else {
            panic!("{:?}", alloc.caches)
        };
        assert_eq!(cache.header, h2);
        assert!(matches!(cache.reg, Home::Gp(r) if ARM64.gp_caller.contains(&r)));
        assert_eq!(alloc.cache_of(b2, v), Some(cache.reg));
        assert_eq!(alloc.cache_of(b1, v), None);
        assert_eq!(alloc.cache_of(h1, v), None);
        assert!(alloc.enters_loaded(b2, h2) && !alloc.enters_loaded(b1, h2));
    }

    #[test]
    fn a_loop_entered_from_two_blocks_loads_its_caches_after_both_entries() {
        // entry branches to A or B, both of which enter the header: neither is the loop's, so
        // both reach the loads; the body's back edge does not.
        let mut l = looped(1, 0, |g, b, q, vs| compare_each(g, b, q, vs, 1));
        let g = &mut l.g;
        let (a, b) = (g.add_block(5, false), g.add_block(6, false));
        let acc = g.node(g.block(l.header).phis[0]).inputs[0];
        let zero = g.node(g.block(l.header).phis[1]).inputs[0];
        // The entry's jump becomes a branch to A and B.
        g.blocks[l.entry.index()].body.pop();
        g.blocks[l.entry.index()].succs.clear();
        g.blocks[l.header.index()].preds.retain(|&p| p != l.entry);
        let c = g.add_node(l.entry, NodeOp::TaggedSame, &[acc, acc], Rep::Bool, 0);
        g.add_node(
            l.entry,
            NodeOp::Branch {
                then_block: a,
                else_block: b,
            },
            &[c],
            Rep::None,
            0,
        );
        g.add_edge(l.entry, a);
        g.add_edge(l.entry, b);
        for from in [a, b] {
            g.add_node(from, NodeOp::Jump(l.header), &[], Rep::None, 0);
            g.add_edge(from, l.header);
        }
        // The header's preds are now (body, a, b): each phi takes an input per pred, in order.
        for (k, &phi) in g.block(l.header).phis.clone().iter().enumerate() {
            let back = g.node(phi).inputs[1];
            let init = if k == 0 { acc } else { zero };
            g.node_mut(phi).inputs = vec![back, init, init];
        }
        let alloc = allocate(g, &ARM64).unwrap();
        assert_consistent(g, &alloc);
        let v = l.values[0];
        assert!(matches!(alloc.homes[v.index()], Home::Spill(_)));
        let [cache] = alloc.caches[..] else {
            panic!("{:?}", alloc.caches)
        };
        assert_eq!(cache.header, l.header);
        assert!(alloc.enters_loaded(l.body, l.header));
        assert!(!alloc.enters_loaded(a, l.header) && !alloc.enters_loaded(b, l.header));
        assert_eq!(alloc.cache_of(a, v), None);
    }

    #[test]
    fn more_candidates_than_free_registers_cache_the_most_read_first() {
        // Eight spilled values, the kth read k + 1 times an iteration, with fourteen words
        // holding registers through the loop: the cached ones are the most read.
        let l = looped(8, 14, |g, b, mut q, vs| {
            for (k, &v) in vs.iter().enumerate() {
                q = compare_each(g, b, q, &[v], k + 1);
            }
            q
        });
        let alloc = allocate(&l.g, &ARM64).unwrap();
        assert_consistent(&l.g, &alloc);
        for &v in &l.values {
            assert!(matches!(alloc.homes[v.index()], Home::Spill(_)), "{v:?}");
        }
        let cached: Vec<NodeIx> = alloc.caches.iter().map(|c| c.value).collect();
        let n = cached.len();
        assert!(n > 0 && n < l.values.len(), "{cached:?}");
        let mut most: Vec<NodeIx> = l.values[l.values.len() - n..].to_vec();
        most.reverse();
        assert_eq!(cached, most);
    }

    #[test]
    fn a_value_defined_in_the_loop_and_spilled_there_gets_no_cache() {
        // Twenty-four words made in the body and compared in reverse: some spill, all defined in
        // the loop; the invariant read beside them still may take a cache.
        let made = std::cell::RefCell::new(Vec::new());
        let l = looped(1, 0, |g, b, q, vs| {
            let mut words = Vec::new();
            for _ in 0..24 {
                let same = g.add_node(b, NodeOp::TaggedSame, &[q, q], Rep::Bool, 2);
                words.push(g.add_node(b, NodeOp::BoolToTagged, &[same], Rep::Tagged, 2));
            }
            let mut acc = compare_each(g, b, q, vs, 1);
            for &w in words.iter().rev() {
                acc = compare_each(g, b, acc, &[w], 1);
            }
            made.borrow_mut().extend(words);
            acc
        });
        let alloc = allocate(&l.g, &ARM64).unwrap();
        assert_consistent(&l.g, &alloc);
        let made = made.into_inner();
        assert!(
            made.iter()
                .any(|v| matches!(alloc.homes[v.index()], Home::Spill(_))),
            "some word made in the loop spills"
        );
        assert!(
            alloc.caches.iter().all(|c| !made.contains(&c.value)),
            "{:?}",
            alloc.caches
        );
    }

    #[test]
    fn a_value_made_in_its_loops_header_and_spilled_after_the_loop_gets_no_cache() {
        // vt is made in the header each iteration, read in the body and after the loop, where
        // twenty-two words made from q push it out (its next use is the farthest): its home is
        // a slot, and registers are free through the loop. A cache loaded at the header would
        // hold the previous iteration's vt.
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        let header = g.add_block(1, true);
        let body = g.add_block(2, false);
        let exit = g.add_block(3, false);
        g.entry = Some(entry);
        g.add_edge(entry, header);
        g.add_edge(header, body);
        g.add_edge(header, exit);
        g.add_edge(body, header);
        let a = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let zero = g.add_node(entry, NodeOp::ConstI32(0), &[], Rep::I32, 0);
        g.add_node(entry, NodeOp::Jump(header), &[], Rep::None, 0);
        let q = g.add_node(header, NodeOp::Phi, &[a], Rep::Tagged, 1);
        let i = g.add_node(header, NodeOp::Phi, &[zero], Rep::I32, 1);
        let ps = state(&mut g, 1, &[]);
        let poll = g.add_node(header, NodeOp::Poll, &[], Rep::None, 1);
        g.node_mut(poll).state = Some(ps);
        let same = g.add_node(header, NodeOp::TaggedSame, &[q, q], Rep::Bool, 1);
        let vt = g.add_node(header, NodeOp::BoolToTagged, &[same], Rep::Tagged, 1);
        let hundred = g.add_node(header, NodeOp::ConstI32(100), &[], Rep::I32, 1);
        let lt = g.add_node(
            header,
            NodeOp::I32Cmp(Cond::Lt),
            &[i, hundred],
            Rep::Bool,
            1,
        );
        g.add_node(
            header,
            NodeOp::Branch {
                then_block: body,
                else_block: exit,
            },
            &[lt],
            Rep::None,
            1,
        );
        let q2 = compare_each(&mut g, body, q, &[vt], 2);
        let one = g.add_node(body, NodeOp::ConstI32(1), &[], Rep::I32, 2);
        let i2 = g.add_node(body, NodeOp::I32Or, &[i, one], Rep::I32, 2);
        g.add_node(body, NodeOp::Jump(header), &[], Rep::None, 2);
        g.node_mut(q).inputs.push(q2);
        g.node_mut(i).inputs.push(i2);
        let words: Vec<NodeIx> = (0..22)
            .map(|_| {
                let same = g.add_node(exit, NodeOp::TaggedSame, &[q, q], Rep::Bool, 3);
                g.add_node(exit, NodeOp::BoolToTagged, &[same], Rep::Tagged, 3)
            })
            .collect();
        let acc = compare_each(&mut g, exit, words[0], &words[1..], 1);
        let out = compare_each(&mut g, exit, acc, &[vt], 1);
        g.add_node(exit, NodeOp::Return, &[out], Rep::None, 3);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert!(
            matches!(alloc.homes[vt.index()], Home::Spill(_)),
            "{:?}",
            alloc.homes[vt.index()]
        );
        assert!(
            alloc.caches.iter().all(|c| c.value != vt),
            "{:?}",
            alloc.caches
        );
    }

    #[test]
    fn a_block_outside_the_loop_inside_its_hull_reads_the_slot() {
        // entry (21 words, v spilled, the chain) -> H (q, i; poll; i < 100) -> A (q against v;
        // branch to X or B) ; B -> H (the latch); X returns q against v; H -> E returns q. In the
        // code order X lies between A and B, inside the loop's hull but none of its blocks: the
        // cache serves H, A and B, and X reads v's slot.
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        let h = g.add_block(1, true);
        let a = g.add_block(2, false);
        let b = g.add_block(3, false);
        let x = g.add_block(4, false);
        let e = g.add_block(5, false);
        g.entry = Some(entry);
        for (from, to) in [(entry, h), (h, a), (h, e), (a, x), (a, b), (b, h)] {
            g.add_edge(from, to);
        }
        let word = |g: &mut Graph| g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
        let words: Vec<NodeIx> = (0..21).map(|_| word(&mut g)).collect();
        let v = word(&mut g);
        let mut acc = words[0];
        for &w in &words[1..] {
            let same = g.add_node(entry, NodeOp::TaggedSame, &[acc, w], Rep::Bool, 0);
            acc = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 0);
        }
        let zero = g.add_node(entry, NodeOp::ConstI32(0), &[], Rep::I32, 0);
        g.add_node(entry, NodeOp::Jump(h), &[], Rep::None, 0);
        let q = g.add_node(h, NodeOp::Phi, &[acc], Rep::Tagged, 1);
        let i = g.add_node(h, NodeOp::Phi, &[zero], Rep::I32, 1);
        let ps = state(&mut g, 1, &[]);
        let poll = g.add_node(h, NodeOp::Poll, &[], Rep::None, 1);
        g.node_mut(poll).state = Some(ps);
        let hundred = g.add_node(h, NodeOp::ConstI32(100), &[], Rep::I32, 1);
        let lt = g.add_node(h, NodeOp::I32Cmp(Cond::Lt), &[i, hundred], Rep::Bool, 1);
        let branch = |then_block, else_block| NodeOp::Branch {
            then_block,
            else_block,
        };
        g.add_node(h, branch(a, e), &[lt], Rep::None, 1);
        let same = g.add_node(a, NodeOp::TaggedSame, &[q, v], Rep::Bool, 2);
        g.add_node(a, branch(x, b), &[same], Rep::None, 2);
        let one = g.add_node(b, NodeOp::ConstI32(1), &[], Rep::I32, 3);
        let i2 = g.add_node(b, NodeOp::I32Or, &[i, one], Rep::I32, 3);
        let q2 = compare_each(&mut g, b, q, &[v], 1);
        g.add_node(b, NodeOp::Jump(h), &[], Rep::None, 3);
        g.node_mut(q).inputs.push(q2);
        g.node_mut(i).inputs.push(i2);
        let out = compare_each(&mut g, x, q, &[v], 1);
        g.add_node(x, NodeOp::Return, &[out], Rep::None, 4);
        g.add_node(e, NodeOp::Return, &[q], Rep::None, 5);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert!(matches!(alloc.homes[v.index()], Home::Spill(_)));
        let [cache] = alloc.caches[..] else {
            panic!("{:?}", alloc.caches)
        };
        assert_eq!(cache.header, h);
        let place = |blk: BlockIx| alloc.order.iter().position(|&o| o == blk).unwrap();
        assert!(
            place(a) < place(x) && place(x) < place(b),
            "{:?}",
            alloc.order
        );
        let x_at = alloc.pos[g.block(x).body[0].index()];
        assert!(
            cache.hull.0 < x_at && x_at < cache.hull.1,
            "X inside the hull"
        );
        for blk in [h, a, b] {
            assert_eq!(alloc.cache_of(blk, v), Some(cache.reg), "{blk:?}");
        }
        assert_eq!(alloc.cache_of(x, v), None, "X reads the slot");
        assert!(!alloc.enters_loaded(x, h) && !alloc.enters_loaded(entry, h));
    }

    #[test]
    fn a_stub_call_in_the_loop_keeps_caches_out_of_the_registers_it_clobbers() {
        // A TypedElement call in the body: ten words read after the loop and live across it take
        // x0-x8 and x10 (not its clobbers), so the caller-saved registers free through the loop
        // are its clobbered x11-x15, which no cache may take.
        let l = looped(1, 10, |g, b, q, vs| {
            let s = state(g, 2, &[]);
            let cell = g.add_node(b, NodeOp::CheckObject, &[q], Rep::Cell, 2);
            g.node_mut(cell).state = Some(s);
            let zero = g.add_node(b, NodeOp::ConstI32(0), &[], Rep::I32, 2);
            let te = g.add_node(
                b,
                NodeOp::TypedElement { write: false },
                &[cell, zero],
                Rep::Tagged,
                2,
            );
            g.node_mut(te).state = Some(s);
            let acc = compare_each(g, b, te, vs, 1);
            compare_each(g, b, acc, &[q], 1)
        });
        let alloc = allocate(&l.g, &ARM64).unwrap();
        assert_consistent(&l.g, &alloc);
        assert!(matches!(alloc.homes[l.values[0].index()], Home::Spill(_)));
        let (gp, _) = clobbers(&NodeOp::TypedElement { write: false });
        let [cache] = alloc.caches[..] else {
            panic!("{:?}", alloc.caches)
        };
        assert!(
            matches!(cache.reg, Home::Gp(r) if gp & (1 << r) == 0),
            "{cache:?}"
        );
    }

    /// Straight-line reads of a spilled value (step R1's read spans): the entry defines 21 words
    /// and then v, spilled at its definition (its next use is the farthest); the words die in a
    /// chain, then `reads` builds on (the chain's word, v) and the result is returned.
    fn straight(
        reads: impl FnOnce(&mut Graph, BlockIx, NodeIx, NodeIx) -> NodeIx,
    ) -> (Graph, NodeIx) {
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        g.entry = Some(entry);
        let word = |g: &mut Graph| g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
        let words: Vec<NodeIx> = (0..21).map(|_| word(&mut g)).collect();
        let v = word(&mut g);
        let mut acc = words[0];
        for &w in &words[1..] {
            let same = g.add_node(entry, NodeOp::TaggedSame, &[acc, w], Rep::Bool, 0);
            acc = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 0);
        }
        let out = reads(&mut g, entry, acc, v);
        g.add_node(entry, NodeOp::Return, &[out], Rep::None, 1);
        (g, v)
    }

    fn effect(g: &mut Graph, b: BlockIx, keep: NodeIx) {
        let s = state(g, 1, &[keep]);
        let call = g.add_node(
            b,
            NodeOp::Effect {
                pc: 1,
                pops: 0,
                pushes: 0,
            },
            &[],
            Rep::None,
            1,
        );
        g.node_mut(call).state = Some(s);
    }

    #[test]
    fn a_spilled_values_reads_with_no_call_between_share_one_loaded_register() {
        // Three reads of v after the chain: one span from the first to the last, in a
        // caller-saved register.
        let (g, v) = straight(|g, b, acc, v| compare_each(g, b, acc, &[v], 3));
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert!(matches!(alloc.homes[v.index()], Home::Spill(_)));
        let readers: Vec<u32> = g
            .block(g.entry.unwrap())
            .body
            .iter()
            .filter(|&&n| g.node(n).inputs.contains(&v))
            .map(|&n| alloc.pos[n.index()])
            .collect();
        assert_eq!(readers.len(), 3);
        let [span] = alloc.spans[..] else {
            panic!("{:?}", alloc.spans)
        };
        assert_eq!(
            (span.value, span.from, span.to),
            (v, readers[0], readers[2])
        );
        assert!(
            matches!(span.reg, Home::Gp(r) if ARM64.gp_caller.contains(&r)),
            "{span:?}"
        );
    }

    #[test]
    fn the_skip_rules_keep_a_single_loop_read_two_block_reads_and_a_doubled_input() {
        // F5's skip rules (`spill_copies`) must not skip what one read or one pair can make.
        // A value spilled before a loop and read once in it, with no other spilled value read
        // twice in a block (no run): the loop rule alone admits it, and it takes a cache.
        let l = looped(1, 0, |g, b, q, vs| compare_each(g, b, q, vs, 1));
        let alloc = allocate(&l.g, &ARM64).unwrap();
        assert!(matches!(alloc.homes[l.values[0].index()], Home::Spill(_)));
        assert!(alloc.spans.is_empty(), "{:?}", alloc.spans);
        assert_eq!(alloc.caches.len(), 1, "{:?}", alloc.caches);
        assert_eq!(alloc.caches[0].value, l.values[0]);
        // Two reads of a spilled value in a block with no loop and no call between: the run rule
        // alone admits them, and they share a span.
        let (g, v) = straight(|g, b, acc, v| compare_each(g, b, acc, &[v], 2));
        let alloc = allocate(&g, &ARM64).unwrap();
        assert!(matches!(alloc.homes[v.index()], Home::Spill(_)));
        assert!(alloc.caches.is_empty());
        let [span] = alloc.spans[..] else {
            panic!("{:?}", alloc.spans)
        };
        assert!(span.value == v && span.from < span.to, "{span:?}");
        // One node that reads it twice (`v == v`): a run of one node, a span over it.
        let (g, v) = straight(|g, b, acc, v| {
            let same = g.add_node(b, NodeOp::TaggedSame, &[v, v], Rep::Bool, 2);
            let word = g.add_node(b, NodeOp::BoolToTagged, &[same], Rep::Tagged, 2);
            let same = g.add_node(b, NodeOp::TaggedSame, &[acc, word], Rep::Bool, 2);
            g.add_node(b, NodeOp::BoolToTagged, &[same], Rep::Tagged, 2)
        });
        let alloc = allocate(&g, &ARM64).unwrap();
        assert!(matches!(alloc.homes[v.index()], Home::Spill(_)));
        let [span] = alloc.spans[..] else {
            panic!("{:?}", alloc.spans)
        };
        assert!(span.value == v && span.from == span.to, "{span:?}");
    }

    #[test]
    fn a_one_block_loop_whose_header_is_its_own_latch_is_not_skipped() {
        // F5's loop rule takes a header's predecessor at or after it in the code order: the
        // header itself, where the loop is one block. The value spilled before it and read once
        // in it (no run: the rule for spans does not admit it) takes a cache.
        let mut g = Graph::new(2);
        let entry = g.add_block(0, false);
        let header = g.add_block(1, true);
        let exit = g.add_block(2, false);
        g.entry = Some(entry);
        g.add_edge(entry, header);
        g.add_edge(header, header);
        g.add_edge(header, exit);
        let word = |g: &mut Graph| g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
        let words: Vec<NodeIx> = (0..21).map(|_| word(&mut g)).collect();
        let v = word(&mut g);
        let mut acc = words[0];
        for &w in &words[1..] {
            let same = g.add_node(entry, NodeOp::TaggedSame, &[acc, w], Rep::Bool, 0);
            acc = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 0);
        }
        let zero = g.add_node(entry, NodeOp::ConstI32(0), &[], Rep::I32, 0);
        g.add_node(entry, NodeOp::Jump(header), &[], Rep::None, 0);
        let q = g.add_node(header, NodeOp::Phi, &[acc], Rep::Tagged, 1);
        let i = g.add_node(header, NodeOp::Phi, &[zero], Rep::I32, 1);
        let ps = state(&mut g, 1, &[]);
        let poll = g.add_node(header, NodeOp::Poll, &[], Rep::None, 1);
        g.node_mut(poll).state = Some(ps);
        let q2 = compare_each(&mut g, header, q, &[v], 1);
        let one = g.add_node(header, NodeOp::ConstI32(1), &[], Rep::I32, 1);
        let i2 = g.add_node(header, NodeOp::I32Or, &[i, one], Rep::I32, 1);
        let hundred = g.add_node(header, NodeOp::ConstI32(100), &[], Rep::I32, 1);
        let lt = g.add_node(
            header,
            NodeOp::I32Cmp(Cond::Lt),
            &[i2, hundred],
            Rep::Bool,
            1,
        );
        g.add_node(
            header,
            NodeOp::Branch {
                then_block: header,
                else_block: exit,
            },
            &[lt],
            Rep::None,
            1,
        );
        g.node_mut(q).inputs.push(q2);
        g.node_mut(i).inputs.push(i2);
        g.add_node(exit, NodeOp::Return, &[q2], Rep::None, 2);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert!(
            matches!(alloc.homes[v.index()], Home::Spill(_)),
            "{:?}",
            alloc.homes[v.index()]
        );
        assert!(alloc.spans.is_empty(), "{:?}", alloc.spans);
        let [cache] = alloc.caches[..] else {
            panic!("{:?}", alloc.caches)
        };
        assert_eq!((cache.value, cache.header), (v, header));
        assert!(alloc.enters_loaded(header, header));
        assert!(!alloc.enters_loaded(entry, header));
    }

    #[test]
    fn a_value_only_a_loops_back_edge_carries_takes_a_cache() {
        // The body reads nothing of v but gives it to the header's phi on its back edge: a read on
        // the edge, at the body's tail, which the copies' reads must hold (an edge move reads the
        // cache), as B's scan of each loop block's edges did.
        let l = looped(1, 0, |_, _, _, vs| vs[0]);
        let alloc = allocate(&l.g, &ARM64).unwrap();
        assert_consistent(&l.g, &alloc);
        let v = l.values[0];
        assert!(
            matches!(alloc.homes[v.index()], Home::Spill(_)),
            "{:?}",
            alloc.homes[v.index()]
        );
        let [cache] = alloc.caches[..] else {
            panic!("{:?}", alloc.caches)
        };
        assert_eq!((cache.value, cache.header), (v, l.header));
        assert_eq!(alloc.cache_of(l.body, v), Some(cache.reg));
    }

    #[test]
    fn the_held_lists_answer_by_start_whatever_order_the_values_come_in() {
        // F5's flat held lists (`Held`): a register's ranges by start, though the values that
        // held it come in another order (a builder's graph numbers nodes out of the code order).
        // x0 holds value 0 over [20, 30] and value 1 over [5, 8]; x1 holds nothing.
        let homes = [Home::Gp(0), Home::Gp(0), Home::Spill(0)];
        let ranges = [(20, 30), (5, 8), (1, 40)];
        let mut held = Held::new(&homes, &ranges);
        assert!(!held.free(0, 0, (6, 7)), "value 1's range");
        assert!(!held.free(0, 0, (25, 26)), "value 0's range");
        assert!(held.free(0, 0, (10, 18)), "between them");
        assert!(
            held.free(0, 1, (6, 7)) && held.free(1, 0, (6, 7)),
            "another register, class"
        );
        held.hold(0, 1, (10, 12));
        assert!(
            !held.free(0, 1, (11, 11)) && held.free(0, 1, (13, 14)),
            "a copy taken"
        );
    }

    #[test]
    fn a_call_between_a_spilled_values_reads_ends_their_span() {
        // Two reads, a call, two reads: a span on each side, none across the call.
        let (g, v) = straight(|g, b, acc, v| {
            let acc = compare_each(g, b, acc, &[v], 2);
            effect(g, b, acc);
            compare_each(g, b, acc, &[v], 2)
        });
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert!(matches!(alloc.homes[v.index()], Home::Spill(_)));
        let call = g
            .block(g.entry.unwrap())
            .body
            .iter()
            .find(|&&n| g.node(n).op.is_effectful())
            .map(|&n| alloc.pos[n.index()])
            .unwrap();
        assert_eq!(alloc.spans.len(), 2, "{:?}", alloc.spans);
        assert!(
            alloc.spans.iter().all(|s| s.to < call || s.from > call),
            "{:?}",
            alloc.spans
        );
        // One read on each side: no span.
        let (g, _) = straight(|g, b, acc, v| {
            let acc = compare_each(g, b, acc, &[v], 1);
            effect(g, b, acc);
            compare_each(g, b, acc, &[v], 1)
        });
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert!(alloc.spans.is_empty(), "{:?}", alloc.spans);
    }

    #[test]
    fn a_stub_call_between_a_spilled_values_reads_keeps_their_span_out_of_its_clobbers() {
        // A TypedElement call between the reads, with ten words read after it holding x0-x8 and
        // x10: the caller-saved registers left are its clobbers, which the span may not take.
        let (g, v) = straight(|g, b, acc, v| {
            let kept: Vec<NodeIx> = (0..10)
                .map(|_| g.add_node(b, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0))
                .collect();
            let acc = compare_each(g, b, acc, &[v], 2);
            let s = state(g, 1, &[]);
            let cell = g.add_node(b, NodeOp::CheckObject, &[acc], Rep::Cell, 1);
            g.node_mut(cell).state = Some(s);
            let zero = g.add_node(b, NodeOp::ConstI32(0), &[], Rep::I32, 1);
            let te = g.add_node(
                b,
                NodeOp::TypedElement { write: false },
                &[cell, zero],
                Rep::Tagged,
                1,
            );
            g.node_mut(te).state = Some(s);
            let acc = compare_each(g, b, te, &[v], 2);
            compare_each(g, b, acc, &kept, 1)
        });
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert!(matches!(alloc.homes[v.index()], Home::Spill(_)));
        let (gp, _) = clobbers(&NodeOp::TypedElement { write: false });
        assert!(!alloc.spans.is_empty());
        let te = g
            .block(g.entry.unwrap())
            .body
            .iter()
            .find(|&&n| matches!(g.node(n).op, NodeOp::TypedElement { .. }))
            .map(|&n| alloc.pos[n.index()])
            .unwrap();
        for s in &alloc.spans {
            if s.from <= te && te <= s.to {
                assert!(matches!(s.reg, Home::Gp(r) if gp & (1 << r) == 0), "{s:?}");
            }
        }
    }

    #[test]
    fn overlapping_runs_take_a_scarce_register_by_the_loads_they_save() {
        // w, then v, spilled; `keep` words read at the end hold registers over the reads, which
        // interleave w v w v v v: v's run saves three loads, w's one. With one register free over
        // both runs, v's takes it though w's comes first by value.
        for keep in [17usize, 18, 19] {
            let mut g = Graph::new(2);
            let entry = g.add_block(0, false);
            g.entry = Some(entry);
            let word = |g: &mut Graph| g.add_node(entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
            let words: Vec<NodeIx> = (0..21).map(|_| word(&mut g)).collect();
            let w = word(&mut g);
            let v = word(&mut g);
            let mut acc = words[0];
            for &x in &words[1..] {
                let same = g.add_node(entry, NodeOp::TaggedSame, &[acc, x], Rep::Bool, 0);
                acc = g.add_node(entry, NodeOp::BoolToTagged, &[same], Rep::Tagged, 0);
            }
            let kept: Vec<NodeIx> = (0..keep).map(|_| word(&mut g)).collect();
            let acc = compare_each(&mut g, entry, acc, &[w, v, w, v, v, v], 1);
            let out = compare_each(&mut g, entry, acc, &kept, 1);
            g.add_node(entry, NodeOp::Return, &[out], Rep::None, 1);
            let alloc = allocate(&g, &ARM64).unwrap();
            assert_consistent(&g, &alloc);
            assert!(matches!(alloc.homes[v.index()], Home::Spill(_)));
            assert!(matches!(alloc.homes[w.index()], Home::Spill(_)));
            if alloc.spans.len() == 1 {
                assert_eq!(alloc.spans[0].value, v, "keep {keep}: {:?}", alloc.spans);
                return;
            }
        }
        panic!("no keep gave one span");
    }

    #[test]
    fn a_loops_body_follows_its_header_and_its_back_edge_takes_the_phis_register() {
        // entry: n = slot 0; jump header. header: i = phi(0, i'); poll; branch i < n.
        // body: i' = i + 1; jump header. exit: return n. The body comes right after the header
        // (the builder's order), and i, dead at the add, gives its register to i', the value its
        // back edge carries (Maglev's loop-phi hint): the edge moves nothing.
        let mut g = Graph::new(1);
        let entry = g.add_block(0, false);
        let header = g.add_block(1, true);
        let body = g.add_block(2, false);
        let exit = g.add_block(3, false);
        g.entry = Some(entry);
        g.add_edge(entry, header);
        g.add_edge(header, body);
        g.add_edge(header, exit);
        g.add_edge(body, header);
        let n = g.add_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
        let s = state(&mut g, 0, &[]);
        let fn_ = g.add_node(entry, NodeOp::CheckNumber, &[n], Rep::F64, 0);
        g.node_mut(fn_).state = Some(s);
        let zero = g.add_node(entry, NodeOp::ConstF64(0), &[], Rep::F64, 0);
        g.add_node(entry, NodeOp::Jump(header), &[], Rep::None, 0);
        let i = g.add_node(header, NodeOp::Phi, &[zero], Rep::F64, 1);
        let ps = state(&mut g, 1, &[]);
        let poll = g.add_node(header, NodeOp::Poll, &[], Rep::None, 1);
        g.node_mut(poll).state = Some(ps);
        let lt = g.add_node(header, NodeOp::F64Cmp(Cond::Lt), &[i, fn_], Rep::Bool, 1);
        g.add_node(
            header,
            NodeOp::Branch {
                then_block: body,
                else_block: exit,
            },
            &[lt],
            Rep::None,
            1,
        );
        let one = g.add_node(body, NodeOp::ConstF64(1f64.to_bits()), &[], Rep::F64, 2);
        let next = g.add_node(body, NodeOp::F64Add, &[i, one], Rep::F64, 2);
        g.node_mut(i).inputs.push(next);
        g.add_node(body, NodeOp::Jump(header), &[], Rep::None, 2);
        g.add_node(exit, NodeOp::Return, &[n], Rep::None, 3);
        let alloc = allocate(&g, &ARM64).unwrap();
        assert_consistent(&g, &alloc);
        assert_eq!(alloc.order, vec![entry, header, body, exit]);
        assert!(matches!(alloc.homes[i.index()], Home::Fp(_)));
        assert_eq!(alloc.homes[next.index()], alloc.homes[i.index()]);
    }
}
