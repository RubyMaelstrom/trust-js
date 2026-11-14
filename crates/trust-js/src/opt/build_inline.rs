//! The tier's graph inliner for callees without effects (P3 stage 3, step 2; the design draft's
//! "Step 2", target/p3/stage3/step2-impl/design-note.txt), under TRUST_JS_OPT_INLINE=graph: at a
//! call site the oracle inlines (`opt::oracle`, on the main thread at the job's preparation), each
//! way is its guard, as a splice's in the tier, and its callee built into the graph from the
//! callee's own first-stage chunk by a builder of its own, as Maglev's BuildEagerInlineCall
//! builds a callee with a second MaglevGraphBuilder over the same graph from the current block
//! (maglev-graph-builder.cc:7669-7744), each an inlined frame (step 1's) whose exits rebuild it
//! and its caller.
//!
//! - The tree ([`Tree`]), made before the caller's loop prepass: each callee chunk's analyses (its
//!   Cfg, initialized reads, liveness and loop prepass, from its own view and snapshot, the rules
//!   the compiled function's builder keeps, per frame) and E0's decision: a callee an operation of
//!   which, in a block its build reaches, the rules leave to the template (an effect: the owner's
//!   explicit operands are step 3's), or which the tier or an inlined frame does not take, is
//!   declined, the call staying the call it is. A site whose kept ways leave no generic call
//!   lowers for the caller's prepass (`Rules::lowers`): its callees run no effect and write no
//!   slot of the caller, as Maglev's inner builder adds its effects to its caller's loop's
//!   (maglev-graph-builder.cc:732-735, 14602-14611), none under E0.
//! - The guards: way k's is CalleeMatches and a branch where WrongCallee is flagged at the call
//!   with a guard byte above k (the furthest guard of the site's chain that exited, + 1: a
//!   splice's guards have pcs of their own, these share the call's), else CheckCallee, which ends
//!   the chain, as a splice's unflagged guard builds no mismatch edge; after a chain of branches
//!   only, the generic call.
//! - The callee's builder: its frame opened after its parent state (the caller's at the call's pc
//!   + 1 without its operands, one for the site's ways); its entry state the arguments in its
//!   formals, missing ones undefined and extras dropped (Maglev's SetArgument and undefined
//!   constant, maglev-graph-builder.cc:7419-7441), every other live slot undefined, as a fresh
//!   frame's, with the caller's value and heap facts, which it hands back with its returns (no
//!   effect runs: Maglev hands its KnownNodeAspects in and takes them back,
//!   maglev-graph-builder.cc:7696, 7726); `this` the receiver; its running environment its
//!   closure's [[Environment]], a constant while the guard holds, its free names' holders folded
//!   on the main thread as a splice's are. Each return jumps to the continuation at the call's
//!   pc + 1, which the caller's own merge joins with the generic call's edge; with no return and
//!   no generic call there is none, and the caller's block ends at the call (Maglev's
//!   DoneWithAbort, maglev-graph-builder.cc:7461-7468). Never-ran callee blocks exit with the
//!   chained state, a throw exits to the VM through both frames.
#![cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    allow(dead_code)
)]

use super::*;
use crate::opt::ir::InlinedFrame;

/// A call site the oracle inlines (`opt::oracle`): its ways, each guarded, in the planner's
/// order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InlineSite {
    /// The call's site: its call-cache index, Op::Call's and Op::CallWithThis's second operand.
    pub(crate) cache: u32,
    pub(crate) ways: Vec<InlineWay>,
}

/// A way of an inlined site, as the oracle read it on the main thread.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct InlineWay {
    /// Its guard's data: an index among the compiled chunk's view's inline targets, whose
    /// `guard` byte is the way's place in its site + 1.
    pub(crate) target: u32,
    /// The callee's first-stage chunk: its index among the compile's callee chunks.
    pub(crate) chunk: u32,
    /// The closure's [[Environment]] (`Env::as_ptr`), the callee's running environment.
    pub(crate) env: usize,
    /// The records `hops` parent links from `env` that the callee's free names count from, by
    /// `hops`, folded on the main thread as a splice's holders are (snapshot.rs's `holder`).
    pub(crate) holders: Box<[(u16, usize)]>,
}

/// A callee chunk as a compile reads it: its frozen view and its snapshot, whose exit profile is
/// read as an inlined frame's (`snapshot::Raw::take_inlined_into`).
pub(crate) type Callee<'v> = (&'v FrozenChunk, &'v Snapshot);

/// What the inliner did with a way of a site the build reached (the log's and the tests').
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InlineNote {
    /// The call's pc in the compiled chunk.
    pub(crate) pc: u32,
    /// The way's place in its site.
    pub(crate) way: u8,
    /// Its callee chunk: an index among the compile's callee chunks.
    pub(crate) chunk: u32,
    pub(crate) outcome: InlineOutcome,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum InlineOutcome {
    /// Built in the graph, an inlined frame.
    Inlined,
    /// Declined: why, and the callee's operation where one of them declined it (E0's).
    Declined {
        reason: &'static str,
        at: Option<(u32, Op)>,
    },
    /// After its site's unflagged guard, which passes no call on to it (as a splice's).
    Unreached,
}

/// Why the inliner declines a callee chunk, with its operation where one declined it.
type Refused = (&'static str, Option<(u32, Op)>);

/// The reason E0 declines a callee for.
pub(crate) const E0_EFFECT: &str = "an operation the tier leaves to its template (E0)";

/// A callee chunk's analyses (see the module's doc).
pub(super) struct Analysis {
    cfg: Cfg,
    initialized: InitializedReads,
    flow: Flow,
    liveness: SlotLiveness,
    /// Its index among the chunks the graph's inlined frames run (`InlinedFrame::chunk`).
    frame_chunk: u32,
}

/// The callees a compile inlines (see the module's doc).
pub(super) struct Tree<'a> {
    callees: &'a [Callee<'a>],
    /// Each callee chunk's analyses, by `InlineWay::chunk`, or why the inliner declines it.
    analyses: Vec<Result<Analysis, Refused>>,
    /// The sites, by call-cache index.
    sites: FastMap<u32, &'a InlineSite>,
}

impl<'a> Tree<'a> {
    /// The tree of `sites` in the compile of `compiled`, their callee chunks `callees`.
    pub(super) fn new(
        compiled: &FrozenChunk,
        sites: &'a [InlineSite],
        callees: &'a [Callee<'a>],
    ) -> Self {
        let mut analyses = Vec::with_capacity(callees.len());
        let mut kept = 0;
        for &(view, snapshot) in callees {
            let analysis = analyse(compiled, view, snapshot, kept);
            kept += u32::from(analysis.is_ok());
            analyses.push(analysis);
        }
        Tree {
            callees,
            analyses,
            sites: sites.iter().map(|site| (site.cache, site)).collect(),
        }
    }

    /// Whether the graph builds no generic call at site `cache`'s call at `pc` (the compiled
    /// chunk's, `snapshot` its feedback): one of its kept ways' guards is unflagged, a
    /// CheckCallee, which ends the site's chain.
    pub(super) fn effect_free(&self, cache: u32, pc: usize, snapshot: &Snapshot) -> bool {
        self.sites.get(&cache).is_some_and(|site| {
            site.ways.iter().enumerate().any(|(place, way)| {
                self.analyses[way.chunk as usize].is_ok() && !flagged(snapshot, pc, place)
            })
        })
    }

    /// The Cfgs of the chunks the graph's inlined frames may run, by `InlinedFrame::chunk`, each
    /// with its index among the compile's callee chunks.
    pub(super) fn into_cfgs(self) -> Vec<(u32, Cfg)> {
        self.analyses
            .into_iter()
            .enumerate()
            .filter_map(|(chunk, analysis)| Some((chunk as u32, analysis.ok()?.cfg)))
            .collect()
    }
}

/// Whether the guard of way `place` of the call at `pc` is flagged: WrongCallee at the call, the
/// furthest guard that exited (its byte) at or past this one.
fn flagged(snapshot: &Snapshot, pc: usize, place: usize) -> bool {
    snapshot.declines(pc, ExitReason::WrongCallee) && snapshot.discarded_ways(pc) > place
}

/// The analyses of `view`, a callee chunk inlined into `compiled`, its feedback `snapshot`, or why
/// the inliner declines it: the tier's refusals of a chunk (`admit`), an inlined frame's (the
/// verifier's `frames`: a call that makes more than its frame, another strictness), the loop
/// prepass's, and E0's (see the module's doc).
fn analyse(
    compiled: &FrozenChunk,
    view: &FrozenChunk,
    snapshot: &Snapshot,
    frame_chunk: u32,
) -> Result<Analysis, Refused> {
    let ops = view.jit_ops();
    let at = |refusal: Refusal| -> Refused {
        let op = refusal.pc.and_then(|pc| Some((pc, *ops.get(pc as usize)?)));
        (refusal.reason, op)
    };
    let cfg = admit(view, snapshot).map_err(at)?;
    if view.frame_args_needed {
        return Err((
            "a callee whose call makes more than its frame (an activation, a rest parameter or \
             an arguments value)",
            None,
        ));
    }
    if view.strict != compiled.strict {
        return Err(("a callee of another strictness", None));
    }
    let (_, n_slots) = view.jit_frame();
    let initialized = cfg.initialized_reads(ops, true);
    let rules = Rules {
        chunk: view,
        ops,
        snapshot,
        cfg: &cfg,
        initialized: &initialized,
        inlining: None,
    };
    let flow = Flow::new(view, &cfg, &rules).map_err(at)?;
    // E0: an operation of a block the build reaches (a never-ran block is an exit) that the rules
    // leave to the template.
    for &id in &flow.rpo {
        let block = &cfg.blocks()[ix(id)];
        if never_ran(snapshot, block.start, block.end)
            && !snapshot.declines(block.start, ExitReason::NeverRan)
        {
            continue;
        }
        if let Some(pc) = (block.start..block.end).find(|&pc| !rules.lowers(pc)) {
            return Err((E0_EFFECT, Some((pc as u32, ops[pc]))));
        }
    }
    let liveness = SlotLiveness::compute(ops, &cfg, n_slots);
    Ok(Analysis {
        cfg,
        initialized,
        flow,
        liveness,
        frame_chunk,
    })
}

/// Each frame's snapshot, by its index: `snapshot`, the compiled chunk's, then each inlined
/// frame's callee chunk's, whose loop headers' flags the phi selector reads (`phis::select`).
pub(super) fn frame_snapshots<'s>(
    graph: &Graph,
    snapshot: &'s Snapshot,
    tree: Option<&Tree<'s>>,
) -> Vec<&'s Snapshot> {
    let mut snapshots = vec![snapshot];
    if let Some(tree) = tree {
        let by_frame: Vec<&'s Snapshot> = tree
            .analyses
            .iter()
            .zip(tree.callees)
            .filter(|(analysis, _)| analysis.is_ok())
            .map(|(_, &(_, snapshot))| snapshot)
            .collect();
        snapshots.extend(
            graph
                .frames
                .iter()
                .map(|frame| by_frame[frame.chunk as usize]),
        );
    }
    snapshots
}

/// An inlined frame's builder's context (see the module's doc).
pub(super) struct Inlined<'a> {
    /// The receiver the call passes, the callee's `this`.
    receiver: NodeIx,
    /// The closure's [[Environment]], the callee's running environment.
    env: usize,
    holders: &'a [(u16, usize)],
    /// The values its callers' operations may reach after its return, whose facts its prunes
    /// keep.
    live: Vec<NodeIx>,
    /// The call's pc in the caller, and the caller's frame.
    call_pc: usize,
    caller: FrameIx,
    /// The edges into the call's continuation so far.
    exits: Exits,
}

impl Inlined<'_> {
    pub(super) fn receiver(&self) -> NodeIx {
        self.receiver
    }

    pub(super) fn env(&self) -> usize {
        self.env
    }

    /// The record `hops` parent links from the closure's environment, where the oracle folded
    /// it.
    pub(super) fn holder(&self, hops: u16) -> Option<usize> {
        self.holders
            .iter()
            .find(|&&(at, _)| at == hops)
            .map(|&(_, record)| record)
    }

    /// The values a prune of the frame's states keeps besides their own: the receiver and the
    /// callers'.
    pub(super) fn live(&self) -> impl Iterator<Item = NodeIx> + '_ {
        self.live
            .iter()
            .copied()
            .chain(std::iter::once(self.receiver))
    }
}

/// The edges into an inlined call's continuation, the block at the call's pc + 1 its first edge
/// makes.
#[derive(Default)]
struct Exits {
    continuation: Option<BlockIx>,
    preds: Vec<Pred>,
}

impl Exits {
    fn continuation(&mut self, graph: &mut Graph, call_pc: usize) -> BlockIx {
        *self
            .continuation
            .get_or_insert_with(|| graph.add_block(call_pc as u32 + 1, false))
    }
}

/// An edge into a continuation.
enum Pred {
    /// A callee's return of `value`, its state's facts `facts`.
    Return {
        block: BlockIx,
        value: NodeIx,
        facts: State,
    },
    /// The generic call's, its state after the call.
    Generic { block: BlockIx, state: State },
}

/// A call an inlined site's ways share.
struct Call {
    pc: usize,
    /// The frames' parent state.
    parent: StateIx,
    callee: NodeIx,
    receiver: NodeIx,
    args: Vec<NodeIx>,
    /// The values the callers' operations may reach after the call.
    live: Vec<NodeIx>,
}

impl<'a> Builder<'a> {
    /// Op::Call and Op::CallWithThis (`with_this`) of `argc` arguments at `pc`, call site
    /// `cache`: its ways built in the graph where the oracle inlines the site (see the module's
    /// doc), the call's template otherwise.
    pub(super) fn call(
        &mut self,
        pc: usize,
        argc: u16,
        cache: u32,
        with_this: bool,
        st: &mut State,
    ) -> Result<(), Refusal> {
        let Some((tree, site)) = self
            .inlining
            .and_then(|tree| Some((tree, *tree.sites.get(&cache)?)))
        else {
            return self.effect(pc, st);
        };
        let mut kept: Vec<(usize, &'a InlineWay, &'a Analysis)> = Vec::new();
        for (place, way) in site.ways.iter().enumerate() {
            match &tree.analyses[way.chunk as usize] {
                Ok(analysis) => kept.push((place, way, analysis)),
                Err((reason, at)) => self.inline_log.push(InlineNote {
                    pc: pc as u32,
                    way: place as u8,
                    chunk: way.chunk,
                    outcome: InlineOutcome::Declined { reason, at: *at },
                }),
            }
        }
        if kept.is_empty() {
            return self.effect(pc, st);
        }
        let argc = usize::from(argc);
        let pops = argc + 1 + usize::from(with_this);
        let len = st.stack.len();
        if len < pops {
            return Err(Refusal::at(pc, "a call above too few operands"));
        }
        // The callee's and the receiver's words (the guards' inputs), and the arguments.
        let callee = st.stack[len - argc - 1];
        let callee = self.tagged(st, callee, pc);
        let receiver = if with_this {
            let receiver = st.stack[len - pops];
            self.tagged(st, receiver, pc)
        } else {
            self.word(PACK_UNDEFINED)
        };
        let args = st.stack[len - argc..].to_vec();
        // The state the inlined frames' exits rebuild their caller from: at the call's pc + 1
        // without its operands, where an exit pushes the call's result.
        let caller = State {
            slots: st.slots.clone(),
            stack: st.stack[..len - pops].to_vec(),
            handlers: st.handlers.clone(),
            ..State::default()
        };
        let parent = self.frame_state(pc + 1, &caller);
        let mut live = caller.stack.clone();
        live.extend(st.slots.iter().filter_map(|slot| match *slot {
            Slot::Clean(value) | Slot::Dirty(value) => Some(value),
            Slot::Entry | Slot::Frame | Slot::Dead => None,
        }));
        if let Some(inlined) = &self.inlined {
            live.extend(inlined.live());
        }
        let call = Call {
            pc,
            parent,
            callee,
            receiver,
            args,
            live,
        };
        let mut exits = Exits::default();
        let mut generic = true;
        for (place, way, analysis) in kept {
            let note = |outcome| InlineNote {
                pc: pc as u32,
                way: place as u8,
                chunk: way.chunk,
                outcome,
            };
            if !generic {
                self.inline_log.push(note(InlineOutcome::Unreached));
                continue;
            }
            if flagged(self.snapshot, pc, place) {
                // The guard's test, the way on its match and the next way on its mismatch.
                let matches = self.callee_matches(pc, way.target, st)?;
                let then_block = self.graph.add_block(pc as u32, false);
                let else_block = self.graph.add_block(pc as u32, false);
                self.add(
                    NodeOp::Branch {
                        then_block,
                        else_block,
                    },
                    &[matches],
                    Rep::None,
                    pc,
                );
                let from = self.cur;
                self.graph.add_edge(from, then_block);
                self.graph.add_edge(from, else_block);
                self.cur = then_block;
                self.inline_way(&call, way, analysis, st, &mut exits)?;
                self.cur = else_block;
            } else {
                // The guard's exit, which ends the chain.
                self.inline_guard(pc, way.target, st)?;
                self.inline_way(&call, way, analysis, st, &mut exits)?;
                generic = false;
            }
            self.inline_log.push(note(InlineOutcome::Inlined));
        }
        if generic {
            // Every guard flagged: the call itself after the last mismatch.
            let mut after = st.clone();
            self.effect(pc, &mut after)?;
            let to = exits.continuation(&mut self.graph, pc);
            self.add(NodeOp::Jump(to), &[], Rep::None, pc);
            let from = self.cur;
            self.graph.add_edge(from, to);
            exits.preds.push(Pred::Generic {
                block: from,
                state: after,
            });
        }
        let Some(to) = exits.continuation else {
            // No return reaches the call's continuation, and no generic call: nothing after
            // the call in this block runs (Maglev's DoneWithAbort).
            self.ended = true;
            return Ok(());
        };
        // The returns and the generic call joined as any join of this frame, at the call's pc + 1:
        // the caller's frame as it was, each return's value pushed, the facts each edge holds.
        let preds = exits
            .preds
            .into_iter()
            .map(|pred| match pred {
                Pred::Return {
                    block,
                    value,
                    mut facts,
                } => {
                    facts.names.set_running(st.names.running());
                    let mut stack = caller.stack.clone();
                    stack.push(value);
                    let state = State {
                        slots: st.slots.clone(),
                        stack,
                        handlers: st.handlers.clone(),
                        ..facts
                    };
                    (block, state)
                }
                Pred::Generic { block, state } => (block, state),
            })
            .collect();
        self.cur = to;
        *st = self.merge(to, preds, pc + 1, &mut Vec::new())?;
        Ok(())
    }

    /// `way`'s callee built in the graph after its guard, in a frame of its own, by a builder over
    /// its chunk (see the module's doc); its returns' edges into `exits`.
    fn inline_way(
        &mut self,
        call: &Call,
        way: &'a InlineWay,
        analysis: &'a Analysis,
        st: &State,
        exits: &mut Exits,
    ) -> Result<(), Refusal> {
        let tree = self.inlining.expect("an inlined site's tree");
        let (view, snapshot) = tree.callees[way.chunk as usize];
        let frame = self.graph.open_frame(InlinedFrame {
            chunk: analysis.frame_chunk,
            parent: call.parent,
            call_pc: call.pc as u32,
            callee: call.callee,
            receiver: call.receiver,
        });
        let (n_params, n_slots) = view.jit_frame();
        let undefined = self.word(PACK_UNDEFINED);
        // A fresh frame's: formal k the argument k, or undefined where the call passes fewer;
        // extras dropped; every other slot undefined.
        let slots = (0..n_slots)
            .map(|slot| {
                if !analysis.liveness.live(0, slot) {
                    Slot::Dead
                } else if slot < n_params {
                    Slot::Dirty(call.args.get(slot).copied().unwrap_or(undefined))
                } else {
                    Slot::Dirty(undefined)
                }
            })
            .collect();
        let mut entry = State {
            slots,
            stack: Vec::new(),
            handlers: Vec::new(),
            ..st.clone()
        };
        entry.names.set_running(None);
        let blocks = analysis.cfg.blocks().len();
        let mut inner = Builder {
            chunk: view,
            ops: view.jit_ops(),
            snapshot,
            cfg: &analysis.cfg,
            flow: &analysis.flow,
            initialized: &analysis.initialized,
            liveness: &analysis.liveness,
            n_slots,
            graph: std::mem::take(&mut self.graph),
            cur: self.cur,
            blocks: vec![None; blocks],
            incoming: (0..blocks).map(|_| Vec::new()).collect(),
            built: vec![false; blocks],
            headers: (0..blocks).map(|_| None).collect(),
            hoisted: std::mem::take(&mut self.hoisted),
            hoisted_nodes: std::mem::take(&mut self.hoisted_nodes),
            moves_env: view.jit_ops().iter().any(names::moves_env),
            ended: false,
            notes: std::mem::take(&mut self.notes),
            frame,
            inlining: None,
            inlined: Some(Inlined {
                receiver: call.receiver,
                env: way.env,
                holders: &way.holders,
                live: call.live.clone(),
                call_pc: call.pc,
                caller: self.frame,
                exits: std::mem::take(exits),
            }),
            inline_log: Vec::new(),
        };
        let built = inner.run_inlined(entry);
        self.graph = std::mem::take(&mut inner.graph);
        self.hoisted = std::mem::take(&mut inner.hoisted);
        self.hoisted_nodes = std::mem::take(&mut inner.hoisted_nodes);
        self.notes = std::mem::take(&mut inner.notes);
        *exits = inner
            .inlined
            .take()
            .expect("an inlined frame's context")
            .exits;
        built
    }

    /// An inlined frame's build: its caller's block's jump into the callee's entry block (a loop
    /// header's where the callee's first operation heads one), `entry` the state the edge leaves,
    /// then the callee's blocks in reverse postorder.
    fn run_inlined(&mut self, entry: State) -> Result<(), Refusal> {
        let flow = self.flow;
        let first = flow.rpo[0];
        let to = self.block_for(0)?;
        let inlined = self.inlined.as_ref().expect("an inlined frame's builder");
        let (from, caller, call_pc) = (self.cur, inlined.caller, inlined.call_pc);
        self.graph.add_node_in(
            caller,
            from,
            NodeOp::Jump(to),
            &[],
            Rep::None,
            call_pc as u32,
        );
        self.graph.add_edge(from, to);
        self.incoming[ix(first)].push((from, entry));
        for &id in &flow.rpo {
            self.build_block(id)?;
        }
        Ok(())
    }

    /// An inlined callee's return of `value` at `pc`, `st` its state: a jump to the call's
    /// continuation, the edge's facts kept for the caller.
    pub(super) fn inline_return(&mut self, pc: usize, value: NodeIx, mut st: State) {
        // The value's facts, and those of the values the caller reaches.
        st.stack.push(value);
        self.prune(&mut st);
        st.stack.clear();
        let Builder { inlined, graph, .. } = self;
        let inlined = inlined.as_mut().expect("an inlined frame's builder");
        let to = inlined.exits.continuation(graph, inlined.call_pc);
        self.add(NodeOp::Jump(to), &[], Rep::None, pc);
        let from = self.cur;
        self.graph.add_edge(from, to);
        let inlined = self.inlined.as_mut().expect("an inlined frame's builder");
        inlined.exits.preds.push(Pred::Return {
            block: from,
            value,
            facts: st,
        });
    }
}
