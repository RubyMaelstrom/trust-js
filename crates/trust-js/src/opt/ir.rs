//! The optimizing tier's IR (P3 stage 1; the plan's "Stage 1" record and Design point 4).
//!
//! A control-flow graph of basic blocks, each a list of SSA nodes ending in one terminator.
//! Values carry a machine representation; checks are explicit nodes, each with the frame state
//! its eager exit rebuilds; effectful nodes run an operation's baseline template on the canonical
//! frame. The builder (build.rs) produces a graph from a chunk and its snapshot, the verifier
//! (verify.rs) checks it, and the back end (regalloc.rs, a64.rs) lowers it. This module is the
//! contract between them and holds no policy.

/// A block's index in [`Graph::blocks`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct BlockIx(pub(crate) u32);

/// A node's index in [`Graph::nodes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct NodeIx(pub(crate) u32);

/// A frame state's index in [`Graph::states`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct StateIx(pub(crate) u32);

impl BlockIx {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

impl NodeIx {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

impl StateIx {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// A value's machine representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Rep {
    /// A NaN-boxed word (value.rs: a Number is its f64 bits, other kinds are tags in the NaN
    /// space), in an x register.
    Tagged,
    /// An unboxed double, in a d register.
    F64,
    /// An unboxed int32, in a w register.
    I32,
    /// 0 or 1, in a w register.
    Bool,
    /// An object's payload pointer, proven by a check, in an x register.
    Cell,
    /// No value: stores, flushes, terminators.
    None,
}

/// A comparison's condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Cond {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// Why a check fails or an exit is taken, for the log, the exit profile (stage 2) and tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ExitReason {
    NotNumber,
    NotInt32,
    Overflow,
    MinusZero,
    NotObject,
    WrongShape,
    NotBoolean,
    /// The inline guard of a spliced callee failed (the callee or the Realm changed).
    WrongCallee,
    /// A block whose named and call sites never ran.
    NeverRan,
    /// An operation the tier does not lower at this site (a TailCall that runs, a return inside an
    /// iterator region): the VM performs it.
    Unsupported,
    /// A throw at a pc with a handler: the VM runs the handler.
    Throw,
    /// TRUST_JS_OPT_EXIT_EVERY.
    Forced,
}

/// What a node does. Inputs are in [`Node::inputs`] in the order each variant names.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum NodeOp {
    // Values.
    /// A tagged constant word (a chunk constant, undefined, a Boolean).
    ConstTagged(u64),
    /// An f64 constant, by its bits.
    ConstF64(u64),
    ConstI32(i32),
    ConstBool(bool),
    /// The canonical frame's slot as the function was entered (Tagged).
    EntrySlot(u16),
    /// A slot reloaded from the frame after an effect (Tagged).
    FrameSlot(u16),
    /// The frame's `this` (ctx.this_word; Tagged).
    This,
    /// One input per predecessor of its block, in predecessor order.
    Phi,
    /// The i-th result of the effect that is this node's only input (Tagged).
    Projection(u8),

    // Checks: each node has a frame state (its eager exit's) and a reason.
    /// Tagged -> F64.
    CheckNumber,
    /// F64 -> I32: an exact int32 and not -0.
    CheckInt32,
    /// Tagged -> Cell.
    CheckObject,
    /// Tagged -> Cell: an object whose shape is one of these (at most four).
    CheckShape(Vec<u32>),
    /// Tagged -> Bool.
    CheckBoolean,
    /// A spliced callee's guard (InlineGuard): inputs the callee word (Tagged), then, when the
    /// InlineTarget's `check_this` is set, the receiver word (Tagged), which must be an object;
    /// the expected callee cell, its closure environment and the planning Realm's global
    /// environment are the chunk's InlineTarget.
    CheckCallee {
        target: u32,
    },

    // Conversions (no exit).
    I32ToF64,
    /// NaN canonicalized, as emit_exec_number_store does.
    F64ToTagged,
    I32ToTagged,
    BoolToTagged,
    /// Tagged -> Bool: ECMAScript ToBoolean (§7.1.2) of the word, false for undefined, null,
    /// false, +0, -0, NaN, the empty String, 0n and the [[IsHTMLDDA]] object and true for every
    /// other value. It runs no JavaScript, does not collect and cannot fail: the builder branches
    /// on it, and lowers `!` through it, where the value is not proven a Boolean (CheckBoolean
    /// would exit on every other value, as `if (object)` has one).
    ToBoolean,

    // Arithmetic. The I32 forms with a state exit on overflow (Mul also on a -0 result).
    I32Add,
    I32Sub,
    I32Mul,
    I32And,
    I32Or,
    I32Xor,
    I32Shl,
    I32Sar,
    /// `>>>`: exits when the result does not fit an int32.
    I32UShr,
    I32Not,
    F64Add,
    F64Sub,
    F64Mul,
    F64Div,
    /// A call to the runtime's fmod.
    F64Mod,
    F64Neg,
    /// -> Bool.
    I32Cmp(Cond),
    /// -> Bool; false for NaN operands except under Ne.
    F64Cmp(Cond),
    /// Tagged === Tagged on words whose kinds make identity the answer (both proven Booleans,
    /// both proven objects, a proven constant of undefined or null) -> Bool.
    TaggedSame,
    BoolNot,

    // Memory.
    /// Cell -> Tagged: the property at the way's location and slot (IcState::loc). The slot holds
    /// a data property's word, never a property-only one: a read keyed `prototype`, whose slot
    /// can hold a function's deferred prototype (value.rs's PACK_LAZY_PROTO), stays an effect.
    LoadField {
        location: u16,
        slot: u32,
    },
    /// (Cell, Tagged): a store to a writable data property's slot, with a card mark when the
    /// value may be a reference.
    StoreField {
        location: u16,
        slot: u32,
    },
    /// (Tagged): writes the slot to the frame (a flush the builder places before effects and
    /// before exits that need the frame current).
    StoreSlot(u16),

    // Safepoints.
    /// A loop header's interrupt and collection poll, the first node of every loop header's
    /// body (after its phis), where the first stage polls at a backward branch's target. Its
    /// slow path may collect (the back end keeps the live values where the collector scans
    /// them); a throw from it (the live-object limit's RangeError) exits with its state when
    /// the state has a handler, and otherwise leaves the function throwing. Not effectful: it
    /// writes nothing a later node reads.
    Poll,

    // Effects.
    /// The baseline template of the operation at `pc`, on the canonical frame: inputs are the
    /// operation's popped operands (Tagged), its results are the Projections that follow it, its
    /// state is the frame state before it (the operation's own pc; used when it throws at a pc
    /// with a handler, and by the forced-exit gate).
    Effect {
        pc: u32,
        pops: u8,
        pushes: u8,
    },

    // Terminators (the last node of every block, and only there).
    Jump(BlockIx),
    /// (Bool).
    Branch {
        then_block: BlockIx,
        else_block: BlockIx,
    },
    /// (Tagged).
    Return,
    /// An unconditional exit with this node's state.
    Exit(ExitReason),
}

impl NodeOp {
    /// Whether this node ends a block.
    pub(crate) fn is_terminator(&self) -> bool {
        matches!(
            self,
            NodeOp::Jump(_) | NodeOp::Branch { .. } | NodeOp::Return | NodeOp::Exit(_)
        )
    }

    /// Whether this node can exit (and so needs a frame state).
    pub(crate) fn can_exit(&self) -> bool {
        matches!(
            self,
            NodeOp::CheckNumber
                | NodeOp::CheckInt32
                | NodeOp::CheckObject
                | NodeOp::CheckShape(_)
                | NodeOp::CheckBoolean
                | NodeOp::CheckCallee { .. }
                | NodeOp::I32Add
                | NodeOp::I32Sub
                | NodeOp::I32Mul
                | NodeOp::I32UShr
                | NodeOp::Poll
                | NodeOp::Effect { .. }
                | NodeOp::Exit(_)
        )
    }

    /// Whether running this node may run JavaScript, collect, or write anything a later node reads
    /// (heap facts die across it; values live across it need a callee-saved home).
    pub(crate) fn is_effectful(&self) -> bool {
        matches!(self, NodeOp::Effect { .. } | NodeOp::F64Mod)
    }
}

/// One SSA node.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Node {
    pub(crate) op: NodeOp,
    pub(crate) inputs: Vec<NodeIx>,
    pub(crate) rep: Rep,
    /// The frame state an exit from this node rebuilds (checks, effects, exits).
    pub(crate) state: Option<StateIx>,
    /// The exit's reason, for nodes that can exit (Exit carries its own).
    pub(crate) reason: Option<ExitReason>,
    /// The bytecode pc this node was built for.
    pub(crate) pc: u32,
    pub(crate) block: BlockIx,
}

/// Where a slot's value is when an exit rebuilds the frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotSource {
    /// The frame already holds it (never written since entry or since the last reload, or
    /// flushed).
    Frame,
    /// A node holds it; the exit writes it.
    Value(NodeIx),
    /// The slot is dead at the pc: the exit leaves it as the frame has it.
    Dead,
}

/// The bytecode state at a pc, as an exit rebuilds it (one canonical frame; stage 3 adds the
/// parent of an inlined frame).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FrameState {
    /// The pc execution resumes at: the failing operation's own pc for a check before it, the next
    /// operation's after an effect.
    pub(crate) pc: u32,
    pub(crate) slots: Box<[SlotSource]>,
    /// Every live operand-stack position, bottom first (all Tagged-representable nodes).
    pub(crate) stack: Box<[NodeIx]>,
    /// Handler records active at the pc, innermost last: (push pc, operand depth at the push).
    pub(crate) handlers: Box<[(u32, u32)]>,
}

/// One basic block.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BlockData {
    /// The bytecode pc the block starts at.
    pub(crate) first_pc: u32,
    pub(crate) preds: Vec<BlockIx>,
    pub(crate) succs: Vec<BlockIx>,
    pub(crate) phis: Vec<NodeIx>,
    /// The block's nodes in order, its terminator last.
    pub(crate) body: Vec<NodeIx>,
    pub(crate) loop_header: bool,
}

/// Why the tier declines a chunk (the log names the reason and the pc).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Refusal {
    pub(crate) pc: Option<u32>,
    pub(crate) reason: &'static str,
}

impl Refusal {
    pub(crate) fn at(pc: usize, reason: &'static str) -> Self {
        Refusal {
            pc: Some(pc as u32),
            reason,
        }
    }

    pub(crate) fn whole(reason: &'static str) -> Self {
        Refusal { pc: None, reason }
    }
}

/// A function's graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Graph {
    pub(crate) blocks: Vec<BlockData>,
    pub(crate) nodes: Vec<Node>,
    pub(crate) states: Vec<FrameState>,
    pub(crate) n_slots: usize,
    /// The entry block (blocks[0] when built by the builder).
    pub(crate) entry: Option<BlockIx>,
}

impl Graph {
    pub(crate) fn new(n_slots: usize) -> Self {
        Graph {
            n_slots,
            ..Graph::default()
        }
    }

    pub(crate) fn add_block(&mut self, first_pc: u32, loop_header: bool) -> BlockIx {
        let ix = BlockIx(self.blocks.len() as u32);
        self.blocks.push(BlockData {
            first_pc,
            preds: Vec::new(),
            succs: Vec::new(),
            phis: Vec::new(),
            body: Vec::new(),
            loop_header,
        });
        ix
    }

    /// Appends a node to `block`'s body (a phi goes to its phis).
    pub(crate) fn add_node(
        &mut self,
        block: BlockIx,
        op: NodeOp,
        inputs: &[NodeIx],
        rep: Rep,
        pc: u32,
    ) -> NodeIx {
        let ix = NodeIx(self.nodes.len() as u32);
        let is_phi = op == NodeOp::Phi;
        self.nodes.push(Node {
            op,
            inputs: inputs.to_vec(),
            rep,
            state: None,
            reason: None,
            pc,
            block,
        });
        let data = &mut self.blocks[block.index()];
        if is_phi {
            data.phis.push(ix);
        } else {
            data.body.push(ix);
        }
        ix
    }

    pub(crate) fn add_state(&mut self, state: FrameState) -> StateIx {
        let ix = StateIx(self.states.len() as u32);
        self.states.push(state);
        ix
    }

    /// Records an edge (predecessor lists are in the order edges are added).
    pub(crate) fn add_edge(&mut self, from: BlockIx, to: BlockIx) {
        self.blocks[from.index()].succs.push(to);
        self.blocks[to.index()].preds.push(from);
    }

    pub(crate) fn node(&self, ix: NodeIx) -> &Node {
        &self.nodes[ix.index()]
    }

    pub(crate) fn node_mut(&mut self, ix: NodeIx) -> &mut Node {
        &mut self.nodes[ix.index()]
    }

    pub(crate) fn block(&self, ix: BlockIx) -> &BlockData {
        &self.blocks[ix.index()]
    }

    pub(crate) fn state(&self, ix: StateIx) -> &FrameState {
        &self.states[ix.index()]
    }
}
