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
//! - The guards (P3 stage 3 step 3f, piece D): each kept way's is CalleeMatches and a branch, the
//!   next way on its mismatch, but the last's, which is CheckCallee and ends the chain (as a
//!   splice's unflagged guard builds no mismatch edge) unless WrongCallee is flagged at the call
//!   with a guard byte above its place (the furthest guard of the site's chain that exited, + 1: a
//!   splice's guards have pcs of their own, these share the call's) or the site's cache is past
//!   the plan (`past_plan`), where it too is a test and the generic call follows its mismatch.
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
    /// The callee its guard expects, by its handle word (the target's `callee`), which the site's
    /// cache names its callees by (piece D's `past_plan`).
    pub(crate) callee: usize,
    /// The callee's first-stage chunk: its index among the compile's callee chunks.
    pub(crate) chunk: u32,
    /// The closure's [[Environment]] (`Env::as_ptr`), the callee's running environment.
    pub(crate) env: usize,
    /// The records `hops` parent links from `env` that the callee's free names count from, by
    /// `hops`, folded on the main thread as a splice's holders are (snapshot.rs's `holder`).
    pub(crate) holders: Box<[(u16, usize)]>,
    /// The way's inline site's position (`stack_trace::InlineSites`, registered by the oracle and
    /// held with the code: the callee, no position of its own, the call's position), which the
    /// frame's generic nodes publish while they run (P3 stage 3 step 3c); NO_POSITION for none.
    pub(crate) position: u32,
    /// The sites of the callee's own calls the plan inlines in turn (P3 stage 3 step 3d), by
    /// the callee chunk's call-cache index: frames nested in this way's frame, whose guards
    /// run in it and whose sites chain to its call's (`oracle::prepare`).
    pub(crate) nested: Vec<InlineSite>,
    /// The by-literal ways of this way's callee's calls (P3 stage 3, piece B): each a call whose
    /// callee is a formal the compiled frame binds to a closure it creates, taken where the
    /// call's callee node is that creation's (`Builder::closures`).
    pub(crate) literals: Vec<LiteralWay>,
}

/// A by-literal way (P3 stage 3, piece B; target/p3/stage3/step3f-impl/design-note-B.txt): the
/// closure literal a call of an inlined callee calls where its callee node is a closure the
/// compiled frame created from it, built with no guard, as Maglev builds a call of a new
/// closure (TryReduceCallForNewClosure, maglev-graph-builder.cc:11963-11980).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LiteralWay {
    /// The call's site in the way's callee chunk: its call-cache index.
    pub(crate) cache: u32,
    /// The literal: Op::MakeClosure's function index in the compiled chunk.
    pub(crate) literal: u32,
    /// Its first-stage chunk: its index among the compile's callee chunks.
    pub(crate) chunk: u32,
    /// Its frame's inline site's position (`stack_trace::InlineCallee::Frame`).
    pub(crate) position: u32,
    /// Whether its closures' [[Environment]] is a self-name record: a named function expression
    /// whose name may be observed (`ast::Function::needs_self_environment`, which the creation,
    /// `Interp::make_function_named`, and the scope plan's addresses both read), whose one
    /// binding is the closure and whose parent is the creation's running environment.
    pub(crate) self_name: bool,
    /// The interpreter the compile was made for and its frame-callee table
    /// (`stack_trace::frame_callee_table`), by address.
    pub(crate) interp: usize,
    pub(crate) table: usize,
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
    /// Built in the graph as a by-literal frame, with no guard (P3 stage 3, piece B).
    ByLiteral,
    /// Declined: why, and the callee's operation where one of them declined it (E0's).
    Declined {
        reason: &'static str,
        at: Option<(u32, Op)>,
    },
    /// After its site's unflagged guard, which passes no call on to it (as a splice's). It no
    /// longer arises (piece D): only a site's last kept way can be a check (`Builder::call`).
    Unreached,
}

/// Why the inliner declines a callee chunk, with its operation where one declined it.
type Refused = (&'static str, Option<(u32, Op)>);

/// The reason E0 declines a callee for: an operation the tier does not lower that needs the
/// whole canonical frame (P3 stage 3 step 3c: every other one is a generic node of the inlined
/// frame), or a call (calls from inlined frames are a step of their own: the oracle keeps leaves).
pub(crate) const E0_EFFECT: &str = "an operation the tier leaves to its template (E0)";

/// The reason the inliner declines a callee whose generic nodes' operand words and slot window
/// would put the compiled frame past a pooled activation record (P3 stage 3 step 3c).
pub(crate) const PAST_RECORD: &str = "an inlined frame past its record";
#[cfg(test)]
thread_local! {
    /// The frames the builders of this thread opened nested in an inlined frame (tests; P3
    /// stage 3 step 3d).
    pub(crate) static TEST_NESTED_FRAMES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// The by-literal frames the builders of this thread opened (tests; piece B).
    pub(crate) static TEST_LITERAL_FRAMES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// A callee chunk's analyses (see the module's doc).
pub(super) struct Analysis {
    cfg: Cfg,
    initialized: InitializedReads,
    flow: Flow,
    liveness: SlotLiveness,
    /// Its index among the chunks the graph's inlined frames run (`InlinedFrame::chunk`).
    frame_chunk: u32,
    /// Whether an operation of a block its build reaches is a generic node (P3 stage 3 step 3c):
    /// the callee may then run JavaScript and end the caller's heap facts, so that its call is no
    /// lowered operation for the caller's loop prepass (`Tree::effect_free`).
    generic: bool,
    /// What its operations may write in place, every block's (`Rules::chunk_writes`): its share of
    /// the union a caller's loop takes (`Tree::writes`).
    writes: Writes,
}

/// The callees a compile inlines (see the module's doc).
pub(super) struct Tree<'a> {
    /// The compiled chunk's view, whose inline targets hold every way's guard data, a nested
    /// frame's too (P3 stage 3 step 3d: `Builder::guard_inputs` of an inlined frame's builder).
    pub(super) compiled: &'a FrozenChunk,
    callees: &'a [Callee<'a>],
    /// Each callee chunk's analyses, by `InlineWay::chunk`, or why the inliner declines it.
    analyses: Vec<Result<Analysis, Refused>>,
    /// The sites, by call-cache index.
    sites: FastMap<u32, &'a InlineSite>,
    /// The closure literals some way's by-literal way names (piece B): the compiled frame's
    /// creations of them carry a fact (`Builder::closures`).
    pub(super) literals: Vec<u32>,
}

impl<'a> Tree<'a> {
    /// The tree of `sites` in the compile of `compiled`, their callee chunks `callees`.
    pub(super) fn new(
        compiled: &'a FrozenChunk,
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
        let mut literals: Vec<u32> = sites
            .iter()
            .flat_map(|site| &site.ways)
            .flat_map(|way| &way.literals)
            .map(|literal| literal.literal)
            .collect();
        literals.sort_unstable();
        literals.dedup();
        Tree {
            compiled,
            callees,
            analyses,
            sites: sites.iter().map(|site| (site.cache, site)).collect(),
            literals,
        }
    }

    /// Whether the graph builds no generic call at site `cache`'s call at `pc` (the compiled
    /// chunk's, `snapshot` its feedback): its last kept way's guard is neither flagged nor past
    /// the plan, a CheckCallee, which ends the site's chain (piece D: every way before it is a
    /// test; `Builder::call` keeps the same ways and asserts that the two agree).
    ///   Its built ways run no generic node either (P3 stage 3 step 3c: an inlined frame's
    ///   generic node may run JavaScript and ends the heap facts, as an effect does).
    pub(super) fn effect_free(&self, cache: u32, pc: usize, snapshot: &Snapshot) -> bool {
        let Some(site) = self.sites.get(&cache) else {
            return false;
        };
        // Every built way (a declined one is not: the next is its site's) and, after the last,
        // the generic call where its guard is flagged or the site is past the plan (piece D,
        // `Builder::call`).
        let mut last = None;
        for (place, way) in site.ways.iter().enumerate() {
            let Ok(analysis) = &self.analyses[way.chunk as usize] else {
                continue;
            };
            if analysis.generic {
                return false;
            }
            last = Some(place);
        }
        let kept: Vec<usize> = site
            .ways
            .iter()
            .filter(|way| self.analyses[way.chunk as usize].is_ok())
            .map(|way| way.callee)
            .collect();
        last.is_some_and(|place| !flagged(snapshot, pc, place) && !past_plan(snapshot, pc, &kept))
    }

    /// What every frame the build may inline at site `cache`'s call may write in place, for its
    /// caller's loop prepass (`Rules::writes_at`, the union rule): each analysed way's callee chunk
    /// (a declined way is no frame: the site's guard exits for its callee), and in it, at every
    /// depth, the frames the build may open: its nested sites' ways (step 3d) and its by-literal
    /// ways (piece B), which `Builder::call` takes from this tree alone, so that no frame the build
    /// makes is missing here. Each callee chunk's every operation counts, a block that never ran
    /// too (its build is an exit there). A construct's inlined initializer, when one is built,
    /// joins here the same way; the verifier checks every built loop against the rule
    /// (`verify::kept_heap`).
    pub(super) fn writes(&self, cache: u32) -> Writes {
        let mut writes = Writes::default();
        if let Some(site) = self.sites.get(&cache) {
            self.ways_writes(&site.ways, &mut writes);
        }
        writes
    }

    fn ways_writes(&self, ways: &[InlineWay], writes: &mut Writes) {
        for way in ways {
            let Ok(analysis) = &self.analyses[way.chunk as usize] else {
                continue;
            };
            writes.extend(&analysis.writes);
            for site in &way.nested {
                self.ways_writes(&site.ways, writes);
            }
            for literal in &way.literals {
                if let Ok(analysis) = &self.analyses[literal.chunk as usize] {
                    writes.extend(&analysis.writes);
                }
            }
        }
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

/// Whether the call site at `pc` of `snapshot`'s chunk is past its plan, whose built ways expect
/// the callees `kept` (handle words): its cache megamorphic, or naming a callee none of them is
/// (P3 stage 3 step 3f, piece D): its last way is then a test, the generic call after it.
fn past_plan(snapshot: &Snapshot, pc: usize, kept: &[usize]) -> bool {
    let crate::feedback::SiteView::Call(call) = snapshot.site(pc) else {
        return false;
    };
    call.state == crate::feedback::CallState::Megamorphic
        || call
            .ways
            .iter()
            .flatten()
            .any(|target| !kept.contains(&target.callee))
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
    // leave to the template and that needs the whole canonical frame, or a call other than a
    // plain or method call or a plain `new`; every other such operation is a generic node of the
    // inlined frame (P3 stage 3 step 3c), a plain or method call too (step 3c-2: the oracle took
    // the way for making only those, `InlineWay::graph_inlines`) and a plain `new` (step 3e: a
    // call node of its frame, as a call is).
    let mut generic = false;
    for &id in &flow.rpo {
        let block = &cfg.blocks()[ix(id)];
        if never_ran(snapshot, block.start, block.end)
            && !snapshot.declines(block.start, ExitReason::NeverRan)
        {
            continue;
        }
        for pc in (block.start..block.end).filter(|&pc| !rules.lowers(pc)) {
            let plain_call = matches!(ops[pc], Op::Call(..) | Op::CallWithThis(..) | Op::New(..));
            if canonical(&ops[pc]) || (makes_call(&ops[pc]) && !plain_call) {
                return Err((E0_EFFECT, Some((pc as u32, ops[pc]))));
            }
            generic = true;
        }
    }
    // A generic node's operand words lie at its own depth from the compiled frame's operand
    // base, its slot window above the callee's capacity (`a64`'s inlined generic nodes): the
    // compiled frame's record must hold them, as JSC's frame covers every inline call frame's
    // baseline frame (dfg/DFGGraph.cpp:1290-1300).
    if generic
        && compiled.jit_frame().1 + cfg.jit_stack_capacity() + n_slots > crate::jit::FRAME_BUF
    {
        return Err((PAST_RECORD, None));
    }
    let liveness = SlotLiveness::compute(ops, &cfg, n_slots);
    let writes = rules.chunk_writes();
    Ok(Analysis {
        cfg,
        initialized,
        flow,
        liveness,
        frame_chunk,
        generic,
        writes,
    })
}

/// Whether `op` calls or constructs a function as its operation, which an inlined frame makes as a
/// generic node only as a plain or method call (P3 stage 3 step 3c-2) or a plain `new` (step
/// 3e): spread, arguments-array, eval and tail calls are left to later pieces.
fn makes_call(op: &Op) -> bool {
    matches!(
        op,
        Op::Call(..)
            | Op::CallWithThis(..)
            | Op::CallSpread(..)
            | Op::CallSpreadThis(..)
            | Op::CallArgsArray
            | Op::CallArgsArrayThis
            | Op::New(..)
            | Op::NewArgsArray
            | Op::TailCall(..)
    )
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
    env: FrameEnv,
    holders: &'a [(u16, usize)],
    /// The values its callers' operations may reach after its return, whose facts its prunes
    /// keep.
    live: Vec<NodeIx>,
    /// The call's pc in the caller, and the caller's frame.
    call_pc: usize,
    caller: FrameIx,
    /// The edges into the call's continuation so far.
    exits: Exits,
    /// The compile's tree (its callee chunks and their analyses) and the sites of this frame's
    /// own calls the plan inlines in turn (P3 stage 3 step 3d): frames nested in this one.
    tree: &'a Tree<'a>,
    nested: &'a [InlineSite],
    /// The by-literal ways of this frame's calls (piece B).
    literals: &'a [LiteralWay],
    /// A by-literal frame's: its activation's frame-callee table entry and the table, which its
    /// returns clear.
    frame_entry: Option<(NodeIx, usize)>,
}

/// An inlined frame's running environment: a guarded way's closure's, a constant while its guard
/// holds; a by-literal frame's, its closure's [[Environment]] as two nodes (piece B).
#[derive(Clone, Copy)]
pub(super) enum FrameEnv {
    Const(usize),
    /// A by-literal frame's: `creation`, the running environment the creation captured, and,
    /// where the closure's [[Environment]] is a self-name record over it (`LiteralWay::
    /// self_name`), `self_name`, the frame's callee node, which is that record's one binding. The
    /// record itself is no node: a read of the name is the callee node, as V8's inlined frame
    /// reads its function_closure register (Maglev's BuildRegisterFrameInitialization,
    /// maglev-graph-builder.cc:830-831, from BuildInlineFunction, :7445), Warp's build_Callee its
    /// inlined call's callee (WarpBuilder.cpp:1969-1983) and JSC's DFG a closure call's callee
    /// slot (DFGByteCodeParser.cpp:2213-2215); every other name walks from `creation`, one record
    /// fewer (`Builder::running_walk`).
    Literal {
        creation: NodeIx,
        self_name: Option<NodeIx>,
    },
}

impl<'a> Inlined<'a> {
    /// The compiled chunk's view, which holds the guards' data of the sites nested in this frame.
    pub(super) fn compiled(&self) -> &'a FrozenChunk {
        self.tree.compiled
    }

    pub(super) fn receiver(&self) -> NodeIx {
        self.receiver
    }

    pub(super) fn env(&self) -> FrameEnv {
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

/// An inlined frame's way, as `Builder::open_inlined` builds it: a guarded way's or a by-literal
/// way's (piece B).
struct WayFrame<'a> {
    /// The callee chunk: its index among the compile's callee chunks.
    chunk: u32,
    position: u32,
    env: FrameEnv,
    holders: &'a [(u16, usize)],
    nested: &'a [InlineSite],
    literals: &'a [LiteralWay],
    /// A by-literal frame's table entry and table.
    frame_entry: Option<(NodeIx, usize)>,
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
        // A call whose callee node is a closure the compiled frame created from a literal this
        // frame's way lists a by-literal way for (piece B): that way, before any nested site's
        // guarded ways, as Maglev reduces a call by its target node's kind before a generic call
        // (ReduceCall, maglev-graph-builder.cc:11940-11983); the identity guards a nested site
        // would test are not built at it.
        let literal = self.literal_way(cache, argc, with_this, st);
        // The compiled frame's sites, or an inlined frame's own, nested in it (step 3d).
        let found = match &self.inlined {
            _ if literal.is_some() => None,
            Some(inlined) => {
                let tree = inlined.tree;
                inlined
                    .nested
                    .iter()
                    .find(|site| site.cache == cache)
                    .map(|site| (tree, site))
            }
            None => self
                .inlining
                .and_then(|tree| Some((tree, *tree.sites.get(&cache)?))),
        };
        if found.is_none() && literal.is_none() {
            return self.effect(pc, st);
        }
        let mut kept: Vec<(usize, &'a InlineWay, &'a Analysis)> = Vec::new();
        for (place, way) in found
            .iter()
            .flat_map(|(_, site)| site.ways.iter().enumerate())
        {
            let tree = found.expect("a site").0;
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
        if kept.is_empty() && literal.is_none() {
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
        if let Some((way, analysis, env)) = literal {
            // The activation's table entry (-1: none, the generic call), the closure stored in it
            // before the frame's first node, the frame on the entry's test.
            let index = self.hoist(Hoisted::FrameIndex(way.interp as u64, way.table as u64));
            let zero = self.hoist(Hoisted::I32(0));
            let stored = self.add(NodeOp::I32Cmp(Cond::Ge), &[index, zero], Rep::Bool, pc);
            let then_block = self.graph.add_block(pc as u32, false);
            let else_block = self.graph.add_block(pc as u32, false);
            self.add(
                NodeOp::Branch {
                    then_block,
                    else_block,
                },
                &[stored],
                Rep::None,
                pc,
            );
            let from = self.cur;
            self.graph.add_edge(from, then_block);
            self.graph.add_edge(from, else_block);
            self.cur = then_block;
            self.add(
                NodeOp::StoreFrameCallee {
                    table: way.table as u64,
                },
                &[index, call.callee],
                Rep::None,
                pc,
            );
            // Test builds: the frame's assumption about its closure's [[Environment]] checked
            // where it begins (see `NodeOp::CheckFrameEnv`).
            if cfg!(test) {
                self.add(
                    NodeOp::CheckFrameEnv {
                        self_name: way.self_name,
                    },
                    &[call.callee, env],
                    Rep::None,
                    pc,
                );
            }
            #[cfg(test)]
            TEST_LITERAL_FRAMES.with(|count| count.set(count.get() + 1));
            let frame = WayFrame {
                chunk: way.chunk,
                position: way.position,
                env: FrameEnv::Literal {
                    creation: env,
                    self_name: way.self_name.then_some(call.callee),
                },
                holders: &[],
                nested: &[],
                literals: &[],
                frame_entry: Some((index, way.table)),
            };
            self.open_inlined(&call, frame, analysis, st, &mut exits)?;
            self.cur = else_block;
            self.inline_log.push(InlineNote {
                pc: pc as u32,
                way: 0,
                chunk: way.chunk,
                outcome: InlineOutcome::ByLiteral,
            });
        }
        // Every planned way from the first compile (P3 stage 3 step 3f, piece D; the owner's D7):
        // each but the last a test and a branch, as Maglev's polymorphic call arms compare each
        // target (maglev-graph-builder.cc:6038-6125) and JSC's polymorphic call inlining switches
        // on the callee (dfg/DFGByteCodeParser.cpp:2726-2770); the last a check, as Maglev's last
        // arm checks (BuildCheckMaps), unless its guard is flagged (WrongCallee past it, as
        // before) or the site's cache is past the plan (megamorphic, or naming a callee no kept
        // way names), where it too is a test and the generic call follows.
        let expected: Vec<usize> = kept.iter().map(|&(_, way, _)| way.callee).collect();
        let past_plan = past_plan(self.snapshot, pc, &expected);
        let last = kept.len().saturating_sub(1);
        for (k, (place, way, analysis)) in kept.into_iter().enumerate() {
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
            if k < last || past_plan || flagged(self.snapshot, pc, place) {
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
        // The compiled frame's loop prepass read this site as `Tree::effect_free` does, from the
        // same ways (every way of an analysed callee: none is dropped while building, where a
        // refusal refuses the compile) and the same last guard: the call lowers there where no
        // generic call is built here and no built way runs a generic node.
        debug_assert!(
            self.inlined.is_some()
                || found.is_some_and(|(tree, site)| tree.effect_free(cache, pc, self.snapshot)
                    == (!generic
                        && !site.ways.iter().any(|way| {
                            matches!(&tree.analyses[way.chunk as usize], Ok(analysis) if analysis.generic)
                        }))),
            "the loop prepass and the build disagree on the call at pc {pc}"
        );
        if generic {
            // The last way a test (its guard flagged, or the site past the plan): the call itself
            // after its mismatch.
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
        let frame = WayFrame {
            chunk: way.chunk,
            position: way.position,
            env: FrameEnv::Const(way.env),
            holders: &way.holders,
            nested: &way.nested,
            literals: &way.literals,
            frame_entry: None,
        };
        self.open_inlined(call, frame, analysis, st, exits)
    }

    /// The by-literal way of call site `cache` in this inlined frame, whose operands are on
    /// `st`'s stack, where its callee node is a creation of the literal the way names (piece B):
    /// the way, its callee's analyses and the creation's running environment; None otherwise, or
    /// where the inliner declines the literal's chunk (logged).
    fn literal_way(
        &mut self,
        cache: u32,
        argc: u16,
        with_this: bool,
        st: &State,
    ) -> Option<(&'a LiteralWay, &'a Analysis, NodeIx)> {
        let inlined = self.inlined.as_ref()?;
        let (literals, tree): (&'a [LiteralWay], &'a Tree<'a>) = (inlined.literals, inlined.tree);
        if literals.is_empty() {
            return None;
        }
        let len = st.stack.len();
        if len < usize::from(argc) + 1 + usize::from(with_this) {
            return None;
        }
        let callee = st.stack[len - usize::from(argc) - 1];
        let &(literal, env) = self.closures.get(&callee)?;
        let way = literals
            .iter()
            .find(|way| way.cache == cache && way.literal == literal)?;
        match &tree.analyses[way.chunk as usize] {
            Ok(analysis) => Some((way, analysis, env)),
            Err((reason, at)) => {
                self.inline_log.push(InlineNote {
                    pc: 0,
                    way: 0,
                    chunk: way.chunk,
                    outcome: InlineOutcome::Declined { reason, at: *at },
                });
                None
            }
        }
    }

    /// A callee built in the graph in a frame of its own, `frame` its way's parts, by a builder
    /// over its chunk (see the module's doc); its returns' edges into `exits`.
    fn open_inlined(
        &mut self,
        call: &Call,
        frame: WayFrame<'a>,
        analysis: &'a Analysis,
        st: &State,
        exits: &mut Exits,
    ) -> Result<(), Refusal> {
        let way = &frame;
        let tree = self
            .inlining
            .or_else(|| self.inlined.as_ref().map(|inlined| inlined.tree))
            .expect("an inlined site's tree");
        #[cfg(test)]
        if self.inlined.is_some() {
            TEST_NESTED_FRAMES.with(|count| count.set(count.get() + 1));
        }
        let (view, snapshot) = tree.callees[way.chunk as usize];
        let frame_ix = self.graph.open_frame(InlinedFrame {
            chunk: analysis.frame_chunk,
            parent: call.parent,
            call_pc: call.pc as u32,
            callee: call.callee,
            receiver: call.receiver,
            position: way.position,
            env: match way.env {
                FrameEnv::Const(env) => env,
                FrameEnv::Literal { .. } => 0,
            },
            by_literal: way.frame_entry.is_some(),
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
            closures: std::mem::take(&mut self.closures),
            moves_env: view.jit_ops().iter().any(names::moves_env),
            checks_store: Vec::new(),
            ended: false,
            notes: std::mem::take(&mut self.notes),
            frame: frame_ix,
            inlining: None,
            inlined: Some(Inlined {
                receiver: call.receiver,
                env: way.env,
                holders: way.holders,
                live: call.live.clone(),
                call_pc: call.pc,
                caller: self.frame,
                exits: std::mem::take(exits),
                tree,
                nested: way.nested,
                literals: way.literals,
                frame_entry: way.frame_entry,
            }),
            inline_log: Vec::new(),
        };
        let built = inner.run_inlined(entry);
        // The nested sites' notes (step 3d), for the log and the tests.
        self.inline_log.append(&mut inner.inline_log);
        self.graph = std::mem::take(&mut inner.graph);
        self.hoisted = std::mem::take(&mut inner.hoisted);
        self.hoisted_nodes = std::mem::take(&mut inner.hoisted_nodes);
        self.closures = std::mem::take(&mut inner.closures);
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
        // A by-literal frame's entry cleared at its return (piece B: an entry is nonzero exactly
        // while the by-literal frame that stored it runs).
        if let Some((index, table)) = self
            .inlined
            .as_ref()
            .and_then(|inlined| inlined.frame_entry)
        {
            self.add(
                NodeOp::ClearFrameCallee {
                    table: table as u64,
                },
                &[index],
                Rep::None,
                pc,
            );
        }
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
