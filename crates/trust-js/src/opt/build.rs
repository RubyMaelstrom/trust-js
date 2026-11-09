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
//!   frame at the header otherwise (an effect leaves every slot in the frame); whether an
//!   operation can be one is the builder's own choice of its form ([`Rules`], stage 2 step 4),
//!   which the builder keeps to. A header's poll opens its body, with the header's state, once a
//!   back edge reaches it.
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
//! - The exit profile (stage 2): where the snapshot says a discarded code's exits flagged a site,
//!   the builder declines the speculation that exited, as JSC's parser consults its exit sites
//!   (dfg/DFGByteCodeParser.cpp): NotNumber takes the generic operation, NotInt32 the f64 path,
//!   Overflow f64 arithmetic even on proven int32s (JSC's makeSafe, DFGByteCodeParser.cpp:
//!   1371-1376), WrongShape the generic access unless the site's ways grew since, WrongCallee the
//!   guard's mismatch edge (CallLinkStatus's couldTakeSlowPath, CallLinkStatus.cpp:98-120), and
//!   NeverRan builds the block (the parser's InadequateCoverage backstop, :7105-7124). The graph
//!   lists each decline.
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
    BlockData, BlockIx, Cond, Decline, ExitReason, FrameState, Graph, NodeIx, NodeOp, Refusal, Rep,
    SlotSource, StateIx,
};
use super::ir::{ElemKind, Holes, Store, TypedKind};
use super::snapshot::{Ran, Snapshot};
use crate::bytecode::{Chunk, Op, UpdKind};
use crate::fasthash::FastMap;
use crate::feedback::{NamedWayKind, ValueClass, ValuesView};
use crate::jit_ir::{BlockId, BuildError, Cfg, InitializedReads};
use crate::value::{PACK_BOOL, PACK_CANON_NAN, PACK_EMPTY, PACK_NULL, PACK_UNDEFINED};

#[path = "build_names.rs"]
mod names;

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
    let rules = Rules {
        chunk,
        ops,
        snapshot,
        cfg: &cfg,
        initialized: &initialized,
    };
    let flow = Flow::new(chunk, &cfg, &rules)?;
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

/// The builder's choice between an operation's lowered form and the effect, made from the chunk
/// and the snapshot alone (P3 stage 2 step 4's (f)): the loop prepass (`Flow::new`) reads it
/// before any value exists, and the builder takes the effect only where it says so, which debug
/// builds check (`Builder::effect`). Where a value's facts rule out a lowered form the rules chose
/// (a constant, an unboxed value of another kind), the builder exits at the operation instead
/// (`Builder::mismatch`), whose exit flags the site for its recompile, so that a loop the prepass
/// found free of effects stays free of them, as Maglev's loop effects and JSC's clobberize rules
/// each come from one source. A release build refuses a disagreement at the loop's back edge.
#[derive(Clone, Copy)]
struct Rules<'a> {
    chunk: &'a Chunk,
    ops: &'a [Op],
    snapshot: &'a Snapshot,
    cfg: &'a Cfg,
    initialized: &'a InitializedReads,
}

impl Rules<'_> {
    /// A named site's own data ways (one shape, or up to four sharing their location and slot),
    /// None where the site takes the generic operation: a generic mark, another kind of way, ways
    /// at different locations, no way, a way of a primitive's (its receiver shape is a wrapper
    /// prototype's, which a CheckShape of the receiver word never passes for the primitive), or
    /// a store site without its writability proof. A `prototype` property may hold a function's
    /// deferred prototype, which only the checked operation materializes.
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
                && way.primitive == 0
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

    /// The element site at `pc` lowered on element words, or None for the generic access (an
    /// effect): a generic mark, a key that is no index, or receivers by rule R (the record's where
    /// it holds any, else the in-line set's) other than Arrays and ordinary objects.
    fn element_site(&self, pc: usize) -> Option<ElemSite> {
        use crate::feedback::{
            ELEM_ARRAY, ELEM_HOLEY, ELEM_KEY_INDEX, ELEM_ORDINARY, ELEM_OUT_OF_BOUNDS,
            ELEM_TYPED_ARRAY,
        };
        let view = self.snapshot.element(pc)?;
        if view.generic || view.recorded_keys & !ELEM_KEY_INDEX != 0 {
            return None;
        }
        let receivers = if view.recorded_receivers != 0 {
            view.recorded_receivers
        } else {
            ELEM_ARRAY | ELEM_TYPED_ARRAY | ELEM_ORDINARY
        };
        if receivers & !(ELEM_ARRAY | ELEM_TYPED_ARRAY | ELEM_ORDINARY) != 0 {
            return None;
        }
        let kind = if receivers & ELEM_ORDINARY == 0 {
            ElemKind::Array
        } else {
            ElemKind::Words
        };
        let typed = if receivers & ELEM_TYPED_ARRAY == 0 {
            Typed::None
        } else if view.recorded_receivers & ELEM_TYPED_ARRAY != 0 && view.typed_kinds == 0 {
            // Its typed arrays were views the native code declines (BigInt or Float16 elements,
            // a shared or too long buffer), which the stub would decline at every access: the
            // generic access, as Maglev leaves BigInt and Float16 elements to it
            // (TryBuildElementAccessOnTypedArray, maglev-graph-builder.cc:5059-5063).
            return None;
        } else if receivers == ELEM_TYPED_ARRAY {
            // The one kind the VM and the misses saw, or the stub for more than one.
            let mut kinds = TypedKind::ALL.into_iter().filter(|&kind| {
                view.typed_kinds & crate::feedback::elem_typed_kind(kind as u8) != 0
            });
            Typed::Only(kinds.next().filter(|_| kinds.next().is_none()))
        } else {
            Typed::Mixed
        };
        let reads = if view.facts & ELEM_OUT_OF_BOUNDS != 0 {
            Reads::PastUsed
        } else if view.facts & ELEM_HOLEY != 0 {
            Reads::Holes
        } else {
            Reads::InBounds
        };
        Some(ElemSite {
            kind,
            reads,
            arrays: receivers & ELEM_ARRAY != 0,
            typed,
        })
    }

    /// Whether the named load of `name` at `pc` reads Arrays' `length`: its ways all the Array's
    /// key-checked first slot, which own_data_ways refuses (an Array's `length` is its slot 0:
    /// value.rs length_property), and no generic mark.
    fn array_length_site(&self, pc: usize, name: u32) -> bool {
        if self.chunk.jit_name(name) != "length" {
            return false;
        }
        let Some(named) = self.snapshot.named(pc) else {
            return false;
        };
        !named.generic
            && named.filled().next().is_some()
            && named.filled().all(|way| {
                way.kind == NamedWayKind::KeyChecked
                    && way.depth == 0
                    && way.slot == 0
                    && way.primitive == 0
            })
    }

    /// Whether the first element load after `pc` in its block (a compound assignment's GetElem)
    /// is lowered on element words.
    fn lowered_load_follows(&self, pc: usize) -> bool {
        let Some(block) = self.cfg.block_at(pc) else {
            return false;
        };
        let end = self.cfg.blocks()[ix(block)].end;
        (pc + 1..end)
            .find(|&next| matches!(self.ops[next], Op::GetElem | Op::GetElemLocal(_)))
            .is_some_and(|next| self.element_site(next).is_some())
    }

    /// Whether the operation at `pc` lowers without an effect whatever its operands' facts (the
    /// loop prepass's question).
    fn lowers(&self, pc: usize) -> bool {
        let snapshot = self.snapshot;
        let flagged =
            |reasons: &[ExitReason]| reasons.iter().any(|&reason| snapshot.declines(pc, reason));
        match self.ops[pc] {
            Op::Const(k) => const_word(self.chunk, k).is_some(),
            Op::LoadLocal(_) => self.initialized.contains(pc),
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::Mod
            | Op::BitAnd
            | Op::BitOr
            | Op::BitXor
            | Op::Shl
            | Op::Shr
            | Op::UShr
            | Op::Lt
            | Op::Gt
            | Op::Le
            | Op::Ge => self.numbers(pc, true).is_some() && !flagged(&[ExitReason::NotNumber]),
            Op::Neg | Op::Plus | Op::BitNot | Op::UpdateLocal(..) => {
                self.numbers(pc, false).is_some() && !flagged(&[ExitReason::NotNumber])
            }
            Op::GetProp(name, _) | Op::GetPropThis(name, _) | Op::GetPropLocal(_, name, _)
                if self.array_length_site(pc, name) =>
            {
                !flagged(&[ExitReason::WrongElements, ExitReason::NotObject])
            }
            Op::GetProp(name, _)
            | Op::GetMethod(name, _)
            | Op::GetPropThis(name, _)
            | Op::GetPropLocal(_, name, _) => self
                .own_data_ways(pc, name, false)
                .is_some_and(|ways| !self.stale_ways(pc, &ways)),
            Op::SetProp(name, _)
            | Op::SetPropDrop(name, _)
            | Op::SetPropThisDrop(name, _)
            | Op::SetPropLocalDrop(_, name, _) => self
                .own_data_ways(pc, name, true)
                .is_some_and(|ways| !self.stale_ways(pc, &ways)),
            Op::GetElem | Op::GetElemLocal(_) => self.element_site(pc).is_some_and(|site| {
                !flagged(LOAD_DECLINES)
                    && match site.typed {
                        Typed::None => true,
                        Typed::Only(Some(_)) => !flagged(&[ExitReason::Detached]),
                        Typed::Only(None) | Typed::Mixed => !flagged(TYPED_DECLINES),
                    }
            }),
            Op::SetElem | Op::SetElemDrop | Op::SetElemLocal(_) | Op::SetElemLocalDrop(_) => {
                self.element_site(pc).is_some_and(|site| {
                    !flagged(STORE_DECLINES)
                        && (site.typed == Typed::None || !flagged(TYPED_DECLINES))
                        && (!matches!(site.typed, Typed::Only(_))
                            || self.store_value_may_be_number(pc))
                })
            }
            Op::UpdateElem(_) => self.numbers(pc, false).is_some() && !flagged(UPDATE_DECLINES),
            Op::ToPropKey | Op::ToPropKeyLocal(_) => {
                self.lowered_load_follows(pc)
                    && !flagged(&[ExitReason::NotObject, ExitReason::NotNumber])
            }
            Op::LoadName(..)
            | Op::LoadNameForCall(..)
            | Op::LoadNameIn(..)
            | Op::LoadNameForCallIn(..)
            | Op::LoadCap(_)
            | Op::StoreNameCached(..)
            | Op::StoreCap(_)
            | Op::StoreCapInit(_)
            | Op::UpdateNameCached(..)
            | Op::UpdateCap(..) => self.name_lowers(pc),
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

    /// The value record of an arithmetic, bitwise, relational, unary or update operation at
    /// `pc` (`two`: both operands' roles), where it saw Numbers only: the operation's Number path.
    fn numbers(&self, pc: usize, two: bool) -> Option<ValuesView> {
        let values = self.snapshot.values(pc)?;
        let roles = if two {
            values.operand0 | values.operand1
        } else {
            values.operand0
        };
        (roles & !NUMBER == 0).then_some(values)
    }

    /// Whether the value the element store at `pc` takes may be a Number: not a constant that
    /// is none, which the operation before the store pushes in its block (`t[i] = 's'`), where a
    /// typed array's store takes the effect, its conversion being the operation's.
    fn store_value_may_be_number(&self, pc: usize) -> bool {
        let starts_block = self
            .cfg
            .block_at(pc)
            .is_none_or(|block| self.cfg.blocks()[ix(block)].start == pc);
        if pc == 0 || starts_block {
            return true;
        }
        match self.ops[pc - 1] {
            Op::Const(k) => {
                const_word(self.chunk, k).is_none_or(|word| number_word(word).is_some())
            }
            Op::Undef => false,
            _ => true,
        }
    }

    /// WrongShape flagged at `pc` where the site's ways did not grow since a discarded check
    /// there (see `Builder::stale_ways`).
    fn stale_ways(&self, pc: usize, ways: &Ways) -> bool {
        self.snapshot.declines(pc, ExitReason::WrongShape)
            && ways.shapes.len() <= self.snapshot.discarded_ways(pc)
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
    fn new(chunk: &Chunk, cfg: &Cfg, rules: &Rules) -> Result<Flow, Refusal> {
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
        // Each Cfg block's index in `rpo` (UNREACHED for a block the entry does not reach).
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
                    if !rules.lowers(pc) {
                        found.effects = true;
                    }
                }
            }
            heads[ix(rpo[header])] = Some(loops.len());
            loops.push(found);
        }
        Ok(Flow { rpo, heads, loops })
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
    /// ToInt32 of an f64 value (its TruncateToInt32), which is no form of it: forms hold one
    /// value, and the truncation is another unless the f64 is an int32.
    truncated: FastMap<NodeIx, NodeIx>,
    /// Heap facts of the current effect epoch: an object word's possible shapes...
    shapes: FastMap<NodeIx, Vec<u32>>,
    /// ...and the word at a location of an object word, loaded or stored.
    fields: FastMap<(NodeIx, u16), NodeIx>,
    /// The elements kind an object word is proven to have (P3 stage 2 step 2): a value fact,
    /// since kind tags and IC_PLAIN never change once JavaScript sees an object (ElemKind).
    kinds: FastMap<NodeIx, ElemKind>,
    /// The element index of a Number value (its CheckIndex), which is no form of it: it takes
    /// -0 to index 0.
    indices: FastMap<NodeIx, NodeIx>,
    /// Element heap facts of the current effect epoch: an object word's element words...
    elements: FastMap<NodeIx, NodeIx>,
    /// ...their used length...
    used: FastMap<NodeIx, NodeIx>,
    /// ...the element read or written at an index of them...
    loaded: FastMap<(NodeIx, NodeIx), NodeIx>,
    /// ...and the indices below their used length whose word a load or store proved no hole.
    present: crate::fasthash::FastSet<(NodeIx, NodeIx)>,
    /// An Array word's `length`, loaded this effect epoch, which an append ends.
    lengths: FastMap<NodeIx, NodeIx>,
    /// A typed array word's view, and a view's element count: value facts, as the view is fixed
    /// at install (its address may cross a safepoint: its owner's cell, which the accesses take
    /// too, keeps the view's memory, and the collector moves nothing).
    views: FastMap<NodeIx, NodeIx>,
    typed_lengths: FastMap<NodeIx, NodeIx>,
    /// A view's elements' address, checked valid this effect epoch (for a store too).
    typed_data: FastMap<(NodeIx, bool), NodeIx>,
    /// The free names' facts (stage 2, step 4), value and heap facts (see `names`).
    names: names::Facts,
}

impl State {
    fn clear_heap(&mut self) {
        self.shapes.clear();
        self.fields.clear();
        self.clear_elements();
        self.names.clear_heap();
    }

    /// End the element heap facts (an effect, which may run any code).
    fn clear_elements(&mut self) {
        self.elements.clear();
        self.used.clear();
        self.loaded.clear();
        self.present.clear();
        self.lengths.clear();
        self.typed_data.clear();
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
    let truncated = first
        .truncated
        .iter()
        .filter(|(value, int)| {
            rest.iter()
                .all(|(_, state)| state.truncated.get(value) == Some(int))
        })
        .map(|(&value, &int)| (value, int))
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
        truncated,
        shapes,
        fields,
        names: names::Facts::meet(preds),
        ..meet_elements(first, rest)
    }
}

/// The element facts every predecessor's state holds (P3 stage 2 step 2): a kind each proves,
/// the weaker where they differ (an Array proves its words), and the indices and heap facts all
/// hold alike.
fn meet_elements(first: &State, rest: &[(BlockIx, State)]) -> State {
    fn alike<K: Copy + Eq + std::hash::Hash, V: Copy + Eq>(
        first: &FastMap<K, V>,
        rest: &[(BlockIx, State)],
        of: impl Fn(&State) -> &FastMap<K, V>,
    ) -> FastMap<K, V> {
        first
            .iter()
            .filter(|(key, value)| {
                rest.iter()
                    .all(|(_, state)| of(state).get(key) == Some(value))
            })
            .map(|(&key, &value)| (key, value))
            .collect()
    }
    let kinds = first
        .kinds
        .iter()
        .filter_map(|(&word, &kind)| {
            let mut met = kind;
            for (_, state) in rest {
                let other = *state.kinds.get(&word)?;
                if !other.proves(met) {
                    met = if met.proves(other) {
                        other
                    } else {
                        return None;
                    };
                }
            }
            Some((word, met))
        })
        .collect();
    State {
        kinds,
        indices: alike(&first.indices, rest, |state| &state.indices),
        elements: alike(&first.elements, rest, |state| &state.elements),
        used: alike(&first.used, rest, |state| &state.used),
        loaded: alike(&first.loaded, rest, |state| &state.loaded),
        present: first
            .present
            .iter()
            .filter(|site| rest.iter().all(|(_, state)| state.present.contains(site)))
            .copied()
            .collect(),
        lengths: alike(&first.lengths, rest, |state| &state.lengths),
        views: alike(&first.views, rest, |state| &state.views),
        typed_lengths: alike(&first.typed_lengths, rest, |state| &state.typed_lengths),
        typed_data: alike(&first.typed_data, rest, |state| &state.typed_data),
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
    /// The running environment, where no operation of the chunk moves it (stage 2, step 4).
    Env,
    /// A record known at the compile (a splice's folded holder).
    ConstEnv(u64),
    /// The running realm's global object, as a cell (stage 2, step 4).
    GlobalObject,
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

    /// The state an operation's exits share, made already (an arm of a branch the operation
    /// builds: P3 stage 2 step 2's kind-tag dispatch).
    fn with(pc: usize, state: StateIx) -> Self {
        Before {
            pc,
            state: Some(state),
        }
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
    /// Whether an operation of the chunk moves its running environment (`names::moves_env`).
    moves_env: bool,
    /// Set where an operation's lowering ended its block with an exit (`mismatch`).
    ended: bool,
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
            moves_env: chunk.jit_ops().iter().any(names::moves_env),
            ended: false,
        }
    }

    /// The rules the loop prepass read (see [`Rules`]).
    fn rules(&self) -> Rules<'a> {
        Rules {
            chunk: self.chunk,
            ops: self.ops,
            snapshot: self.snapshot,
            cfg: self.cfg,
            initialized: self.initialized,
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
        let never_ran = self.never_ran(start, end)
            && !self.declined(start, ExitReason::NeverRan, "the block built");
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

    /// Whether the exit profile flagged `reason` at `pc` (the snapshot's declined mask), listing
    /// the decline with `form`, what the builder builds instead: the caller asks only where it
    /// would otherwise speculate.
    fn declined(&mut self, pc: usize, reason: ExitReason, form: &'static str) -> bool {
        if !self.snapshot.declines(pc, reason) {
            return false;
        }
        self.graph.declines.push(Decline {
            pc: pc as u32,
            reason,
            form,
        });
        true
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
            Hoisted::Env => (NodeOp::Env, Rep::Env),
            Hoisted::ConstEnv(address) => (NodeOp::ConstEnv(address), Rep::Env),
            Hoisted::GlobalObject => (NodeOp::GlobalObject, Rep::Cell),
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
            // An address is no form of a value: it names storage the value's object owns.
            Rep::Ptr => {}
            // An environment is no language value (stage 2, step 4).
            Rep::Env | Rep::None => {}
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

    /// ToInt32 of `value` (ECMA-262 §7.1.6) as an int32, as Maglev's GetTruncatedInt32ForToNumber
    /// gives it: a proven int32 as it is, a constant folded, an f64 truncated, a word checked a
    /// Number and truncated; None for a value that is never a Number.
    fn truncated_int32(
        &mut self,
        st: &mut State,
        cx: &mut Before,
        value: NodeIx,
    ) -> Option<NodeIx> {
        if let Some(int) = self.forms(st, value).i32 {
            return Some(int);
        }
        if let Some(word) = self.constant(value) {
            let int = crate::eval::to_int32(number_word(word)?);
            return Some(self.hoist(Hoisted::I32(int)));
        }
        let number = match self.rep(value) {
            Rep::F64 => value,
            Rep::Tagged => self.number(st, cx, value)?,
            _ => return None,
        };
        if let Some(int) = self.forms(st, number).i32 {
            return Some(int);
        }
        if let Some(&int) = st.truncated.get(&number) {
            return Some(int);
        }
        let int = self.add(NodeOp::TruncateToInt32, &[number], Rep::I32, cx.pc);
        st.truncated.insert(number, int);
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
        // The global object every global read loads its properties from (stage 2, step 4).
        if let Some(&global) = self.hoisted.get(&Hoisted::GlobalObject) {
            live.insert(global);
        }
        st.forms.retain(|value, forms| {
            live.contains(value) || forms.members().any(|m| live.contains(&m))
        });
        st.truth.retain(|value, _| live.contains(value));
        // An f64 is reached through its value's forms (a slot holds the word).
        let forms = &st.forms;
        st.truncated
            .retain(|value, _| live.contains(value) || forms.contains_key(value));
        st.shapes.retain(|word, _| live.contains(word));
        st.fields.retain(|(word, _), _| live.contains(word));
        // Element facts (P3 stage 2 step 2): a kind or an element address is an object word's,
        // and an index a Number's (reached through its value's forms, as a truncation is); a
        // used length and the element facts are an address's.
        st.kinds.retain(|word, _| live.contains(word));
        let forms = &st.forms;
        st.indices
            .retain(|value, _| live.contains(value) || forms.contains_key(value));
        st.elements.retain(|word, _| live.contains(word));
        let addresses: crate::fasthash::FastSet<NodeIx> = st.elements.values().copied().collect();
        st.used.retain(|words, _| addresses.contains(words));
        st.loaded.retain(|(words, _), _| addresses.contains(words));
        st.present.retain(|(words, _)| addresses.contains(words));
        st.lengths.retain(|word, _| live.contains(word));
        st.views.retain(|word, _| live.contains(word));
        let views: crate::fasthash::FastSet<NodeIx> = st.views.values().copied().collect();
        st.typed_lengths.retain(|view, _| views.contains(view));
        st.typed_data.retain(|(view, _), _| views.contains(view));
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
            truncated: met.truncated,
            kinds: met.kinds,
            indices: met.indices,
            views: met.views,
            typed_lengths: met.typed_lengths,
            names: met.names.at_header(self.moves_env),
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
            Op::GetProp(name, _) if self.array_length_site(pc, name) => {
                self.array_length(pc, Recv::Stack, st)?
            }
            Op::GetPropThis(name, _) if self.array_length_site(pc, name) => {
                self.array_length(pc, Recv::This, st)?
            }
            Op::GetPropLocal(slot, name, _) if self.array_length_site(pc, name) => {
                let slot = self.slot(pc, slot as usize)?;
                self.array_length(pc, Recv::Slot(slot), st)?
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
            Op::GetElem => self.element_load(pc, Recv::Stack, st)?,
            Op::GetElemLocal(slot) => {
                let slot = self.slot(pc, slot as usize)?;
                self.element_load(pc, Recv::Slot(slot), st)?
            }
            Op::SetElem => self.element_store(pc, Recv::Stack, true, st)?,
            Op::SetElemDrop => self.element_store(pc, Recv::Stack, false, st)?,
            Op::SetElemLocal(slot) => {
                let slot = self.slot(pc, slot as usize)?;
                self.element_store(pc, Recv::Slot(slot), true, st)?
            }
            Op::SetElemLocalDrop(slot) => {
                let slot = self.slot(pc, slot as usize)?;
                self.element_store(pc, Recv::Slot(slot), false, st)?
            }
            Op::UpdateElem(kind) => self.update_element(pc, kind, st)?,
            Op::ToPropKey => self.to_prop_key(pc, Recv::Stack, st)?,
            Op::ToPropKeyLocal(slot) => {
                let slot = self.slot(pc, slot as usize)?;
                self.to_prop_key(pc, Recv::Slot(slot), st)?
            }
            Op::LoadName(..) | Op::LoadNameIn(..) | Op::LoadCap(_) => {
                self.name_load(pc, false, st)?
            }
            Op::LoadNameForCall(..) | Op::LoadNameForCallIn(..) => self.name_load(pc, true, st)?,
            Op::StoreNameCached(..) | Op::StoreCap(_) => self.name_store(pc, false, st)?,
            Op::StoreCapInit(_) => self.name_store(pc, true, st)?,
            Op::UpdateNameCached(_, _, kind) | Op::UpdateCap(_, kind) => {
                self.name_update(pc, kind, st)?
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
            Op::InlineGuard(target, mismatch) => {
                if self.declined(pc, ExitReason::WrongCallee, "the guard's mismatch edge") {
                    let matches = self.callee_matches(pc, target, st)?;
                    self.branch(pc, matches, pc + 1, mismatch as usize, std::mem::take(st))?;
                } else {
                    self.inline_guard(pc, target, st)?;
                    self.jump(pc, pc + 1, std::mem::take(st))?;
                }
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
        // An exit in place of the operation (`mismatch`) ended the block.
        Ok(!std::mem::take(&mut self.ended))
    }

    /// The operation at `pc`, whose lowered form the rules chose, cannot take it for its
    /// operands (a value whose facts rule it out: a constant, an unboxed value of another kind):
    /// an exit at the operation with `reason`, which flags the site for its recompile, where the
    /// rules say the operation lowers, so that the loop prepass's reading holds (see [`Rules`]);
    /// the effect where they do not.
    fn mismatch(&mut self, pc: usize, reason: ExitReason, st: &mut State) -> Result<(), Refusal> {
        if !self.rules().lowers(pc) {
            return self.effect(pc, st);
        }
        self.exit(pc, reason, std::mem::take(st));
        self.ended = true;
        Ok(())
    }

    /// The operation at `pc` as its baseline template on the canonical frame.
    fn effect(&mut self, pc: usize, st: &mut State) -> Result<(), Refusal> {
        // The loop prepass took no operation the rules lower for an effect (see `Rules`).
        debug_assert!(
            !self.rules().lowers(pc),
            "the rules lower pc {pc} ({:?}), which the builder made an effect",
            self.ops[pc]
        );
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
        if names::moves_env(&self.ops[pc]) {
            st.names.moved();
        }
        Ok(())
    }

    /// NotNumber flagged at `pc`, where an operand of `operands` needs its Number checked: the
    /// generic operation, as if the record had seen what is not a Number.
    fn declines_number(&mut self, pc: usize, st: &State, operands: &[NodeIx]) -> bool {
        !operands.iter().all(|&value| self.known_number(st, value))
            && self.declined(pc, ExitReason::NotNumber, "the generic operation")
    }

    /// Whether int32 arithmetic at `pc` is declined for f64: NotInt32 flagged where an operand
    /// needs its int32 checked (none does when `proven`), or Overflow flagged, which takes f64
    /// even on proven int32s (JSC's makeSafe).
    fn declines_int32(&mut self, pc: usize, proven: bool) -> bool {
        (!proven && self.declined(pc, ExitReason::NotInt32, "f64 arithmetic"))
            || self.declined(pc, ExitReason::Overflow, "f64 arithmetic")
    }

    /// Add, Sub, Mul, Div and Mod: f64 arithmetic on Numbers, int32 with an overflow exit where
    /// rule U's hint or the operands' facts say int32 (Add and Sub on proven int32s; Mul only on
    /// the hint, its int32 products leaving the range too often); the generic operation where a
    /// role saw a value that is not a Number.
    fn arithmetic(&mut self, pc: usize, op: Op, st: &mut State) -> Result<(), Refusal> {
        let Some(values) = self.rules().numbers(pc, true) else {
            return self.effect(pc, st);
        };
        let len = st.stack.len();
        let (a, b) = (st.stack[len - 2], st.stack[len - 1]);
        if !self.numeric(a) || !self.numeric(b) {
            return self.mismatch(pc, ExitReason::NotNumber, st);
        }
        if self.declines_number(pc, st, &[a, b]) {
            return self.effect(pc, st);
        }
        let proven = self.known_int32(st, a) && self.known_int32(st, b);
        let hinted = int32_hint(&values) && self.int32_operand(st, a) && self.int32_operand(st, b);
        let int = match op {
            Op::Add | Op::Sub => proven || hinted,
            Op::Mul => hinted,
            _ => false,
        } && !self.declines_int32(pc, proven);
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

    /// The bitwise operators on their operands' ToInt32 (ECMA-262 §7.1.6), each a Number
    /// truncated in line (Maglev's GetTruncatedInt32ForToNumber). `>>>` gives an int32 when its
    /// count is a constant whose low five bits are not all zero, and otherwise its uint32 result
    /// as an f64, as Maglev's Int32ShiftRightLogical gives a Uint32.
    fn bitwise(&mut self, pc: usize, op: Op, st: &mut State) -> Result<(), Refusal> {
        if self.rules().numbers(pc, true).is_none() {
            return self.effect(pc, st);
        }
        let len = st.stack.len();
        let (a, b) = (st.stack[len - 2], st.stack[len - 1]);
        if !self.numeric(a) || !self.numeric(b) {
            return self.mismatch(pc, ExitReason::NotNumber, st);
        }
        if self.declines_number(pc, st, &[a, b]) {
            return self.effect(pc, st);
        }
        let mut cx = Before::new(pc);
        let (Some(x), Some(y)) = (
            self.truncated_int32(st, &mut cx, a),
            self.truncated_int32(st, &mut cx, b),
        ) else {
            return Err(Refusal::at(pc, "a Number operand without its ToInt32"));
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
            let fits = self
                .constant(b)
                .and_then(number_word)
                .is_some_and(|count| crate::eval::to_int32(count) & 31 != 0)
                && !self.declined(pc, ExitReason::Overflow, "a uint32 as an f64");
            if fits {
                // The result is below 2^31: the exit never runs.
                let state = self.before(&mut cx, st);
                self.add_exit(node, &[x, y], Rep::I32, pc, state, ExitReason::Overflow)
            } else {
                self.add(NodeOp::I32UShrToF64, &[x, y], Rep::F64, pc)
            }
        } else {
            self.add(node, &[x, y], Rep::I32, pc)
        };
        st.stack.truncate(len - 2);
        st.stack.push(result);
        Ok(())
    }

    /// Neg, Plus (ToNumber) and BitNot on Numbers.
    fn unary(&mut self, pc: usize, op: Op, st: &mut State) -> Result<(), Refusal> {
        if self.rules().numbers(pc, false).is_none() {
            return self.effect(pc, st);
        }
        let len = st.stack.len();
        let value = st.stack[len - 1];
        if !self.numeric(value) {
            return self.mismatch(pc, ExitReason::NotNumber, st);
        }
        if self.declines_number(pc, st, &[value]) {
            return self.effect(pc, st);
        }
        let mut cx = Before::new(pc);
        let result = match op {
            Op::BitNot => {
                let Some(x) = self.truncated_int32(st, &mut cx, value) else {
                    return Err(Refusal::at(pc, "a Number operand without its ToInt32"));
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
        let Some(values) = self.rules().numbers(pc, true) else {
            return self.effect(pc, st);
        };
        let len = st.stack.len();
        let (a, b) = (st.stack[len - 2], st.stack[len - 1]);
        if !self.numeric(a) || !self.numeric(b) {
            return self.mismatch(pc, ExitReason::NotNumber, st);
        }
        if self.declines_number(pc, st, &[a, b]) {
            return self.effect(pc, st);
        }
        let cond = match op {
            Op::Lt => Cond::Lt,
            Op::Gt => Cond::Gt,
            Op::Le => Cond::Le,
            _ => Cond::Ge,
        };
        let proven = self.known_int32(st, a) && self.known_int32(st, b);
        let hinted = int32_hint(&values) && self.int32_operand(st, a) && self.int32_operand(st, b);
        let int =
            proven || (hinted && !self.declined(pc, ExitReason::NotInt32, "an f64 comparison"));
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
        let Some(values) = self.rules().numbers(pc, false) else {
            return self.effect(pc, st);
        };
        let old = self.read_slot(st, slot, pc);
        if !self.numeric(old) {
            return self.mismatch(pc, ExitReason::NotNumber, st);
        }
        if self.declines_number(pc, st, &[old]) {
            return self.effect(pc, st);
        }
        let increment = matches!(
            kind,
            UpdKind::PreInc | UpdKind::PostInc | UpdKind::IncDiscard
        );
        let proven = self.known_int32(st, old);
        let int = (proven || (int32_hint(&values) && self.int32_operand(st, old)))
            && !self.declines_int32(pc, proven);
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
    /// at different locations, no way, a way of a primitive's (its receiver shape is a wrapper
    /// prototype's, which a CheckShape of the receiver word never passes for the primitive), or
    /// a store site without its writability proof. A `prototype` property may hold a function's
    /// deferred prototype, which only the checked operation materializes.
    fn own_data_ways(&self, pc: usize, name: u32, store: bool) -> Option<Ways> {
        self.rules().own_data_ways(pc, name, store)
    }

    /// WrongShape flagged at `pc` where the site's ways did not grow since a discarded check
    /// there (a fifth shape, a receiver no way caches): the generic access, JSC's slow version
    /// being an IC in its optimized code. Ways that grew are checked again, as V8 and
    /// SpiderMonkey recompile on what their ICs learned.
    fn stale_ways(&mut self, pc: usize, ways: &Ways) -> bool {
        self.rules().stale_ways(pc, ways)
            && self.declined(pc, ExitReason::WrongShape, "the generic access")
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
        if self.stale_ways(pc, &ways) {
            return self.effect(pc, st);
        }
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
        if self.stale_ways(pc, &ways) {
            return self.effect(pc, st);
        }
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

    /// The words an InlineGuard tests: the callee's, and for a sloppy callee that reads `this`
    /// the receiver's.
    fn guard_inputs(
        &mut self,
        pc: usize,
        target: u32,
        st: &mut State,
    ) -> Result<Vec<NodeIx>, Refusal> {
        let guard = self.chunk.jit_inline_target(target);
        let callee_depth = guard.argc as usize + 1;
        let check_this = guard.check_this;
        let len = st.stack.len();
        if len < callee_depth + usize::from(check_this) {
            return Err(Refusal::at(pc, "an inline guard above too few operands"));
        }
        let callee = st.stack[len - callee_depth];
        let mut inputs = vec![self.tagged(st, callee, pc)];
        if check_this {
            let receiver = st.stack[len - callee_depth - 1];
            inputs.push(self.tagged(st, receiver, pc));
        }
        Ok(inputs)
    }

    /// InlineGuard (a spliced callee's): the callee and the Realm compared, and for a sloppy
    /// callee that reads `this` the receiver an object (CheckCallee, its inputs the callee's word
    /// and then the receiver's); a failure exits at the guard, whose mismatch the VM then takes,
    /// so the mismatch edge is not built.
    fn inline_guard(&mut self, pc: usize, target: u32, st: &mut State) -> Result<(), Refusal> {
        let inputs = self.guard_inputs(pc, target, st)?;
        let mut cx = Before::new(pc);
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

    /// An InlineGuard whose WrongCallee the exit profile flagged: its test as a Boolean
    /// (CalleeMatches), on which the caller branches to the splice or to the guard's mismatch,
    /// the generic call, so that another callee runs in place, as the template JIT's second stage
    /// runs it (JSC's couldTakeSlowPath turns the DFG's CheckBadValue into a slow-path call,
    /// dfg/DFGByteCodeParser.cpp:2867-2883; Maglev drops its CheckValue once the call's feedback
    /// is megamorphic, maglev-graph-builder.cc:11460-11485).
    fn callee_matches(
        &mut self,
        pc: usize,
        target: u32,
        st: &mut State,
    ) -> Result<NodeIx, Refusal> {
        let inputs = self.guard_inputs(pc, target, st)?;
        Ok(self.add(NodeOp::CalleeMatches { target }, &inputs, Rep::Bool, pc))
    }
}

/// What an element load does with a hole and with an index past the used words, by its record's
/// facts (P3 stage 2 step 2). Only the first stage's helpers take either there, and they record
/// it, so a fact's absence is evidence as of the snapshot (an Array's facts; an ordinary object's
/// record holds none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reads {
    /// Neither seen: a hole exits [Hole], an index past used [OutOfBounds].
    InBounds,
    /// Holes seen (HOLEY): a hole reads undefined under the prototype proof.
    Holes,
    /// Reads past the used words seen (OUT_OF_BOUNDS): those and holes read undefined under the
    /// proof.
    PastUsed,
}

/// An element site lowered on element words, as its record chooses (Maglev's
/// TryBuildElementAccess chooses from its feedback, maglev-graph-builder.cc:5496-5624): the
/// receivers' check, and what a load does with holes.
#[derive(Clone, Copy, Debug)]
struct ElemSite {
    /// Array where the receivers were Arrays only, Words where ordinary objects were among them.
    kind: ElemKind,
    reads: Reads,
    /// Whether Arrays are among the receivers: a store then fills holes and appends in line,
    /// whatever the facts, since the first stage does both in line and records neither.
    arrays: bool,
    /// Where typed arrays are among the receivers: the access's form for them.
    typed: Typed,
}

/// What an element site does with typed arrays (the design's choice from the receiver set).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Typed {
    /// None among the receivers: the element words alone.
    None,
    /// Typed arrays alone, of the one kind the record saw (per-kind loads and stores) or not
    /// (the template's TypedElement stub).
    Only(Option<TypedKind>),
    /// Typed arrays and other receivers, or no record at all: a branch on the kind tag between
    /// the stub and the element words, as the template dispatches.
    Mixed,
}

/// The exit reasons whose flag at an element load sends it to the generic access: its receiver's
/// object and elements checks and its key's Number and index checks. Hole and OutOfBounds widen
/// its reads instead (`flagged_reads`; a typed array site's, TYPED_DECLINES).
const LOAD_DECLINES: &[ExitReason] = &[
    ExitReason::WrongElements,
    ExitReason::NotObject,
    ExitReason::NotNumber,
    ExitReason::NotInt32,
];
/// A store's: a load's, and its writes past the used words, into holes, and growth.
const STORE_DECLINES: &[ExitReason] = &[
    ExitReason::WrongElements,
    ExitReason::NotObject,
    ExitReason::NotNumber,
    ExitReason::NotInt32,
    ExitReason::OutOfBounds,
    ExitReason::Hole,
    ExitReason::CouldNotGrow,
];
/// A typed array site's, beside a load's or a store's: a detached or shrunk buffer, and an index
/// past the view's length, whose flag at the stub (its one status) or at a site of other
/// receivers too (the words' reads) does not say what exited. A load of one kind reads past the
/// length instead, as Maglev's does under kHandleOOB (a Select on BranchIfTypedArrayBounds,
/// maglev-graph-builder.cc:5134-5143); a store's flag is STORE_DECLINES' own (Maglev skips such
/// a store in line, 5148-5200).
const TYPED_DECLINES: &[ExitReason] = &[ExitReason::OutOfBounds, ExitReason::Detached];
/// An update's: a load's, its read's hole and bounds (the write after it would exit at a hole
/// the read took as undefined); Overflow takes f64 arithmetic instead.
const UPDATE_DECLINES: &[ExitReason] = &[
    ExitReason::WrongElements,
    ExitReason::NotObject,
    ExitReason::NotNumber,
    ExitReason::NotInt32,
    ExitReason::OutOfBounds,
    ExitReason::Hole,
];

/// The element index a constant key names: an int32 Number, -0 being index 0.
fn const_index(word: u64) -> Option<i32> {
    let x = number_word(word)?;
    if x == 0.0 {
        Some(0)
    } else {
        exact_i32(x)
    }
}

/// Element accesses (P3 stage 2 step 2; the plan's Stage 2 record, "Step 2"): Maglev's element
/// access built from the snapshot on TRust's element words, which are SpiderMonkey's
/// ObjectElements, so with Warp's choices where Maglev's rest on V8's elements kinds.
impl Builder<'_> {
    /// The element site at `pc` lowered on element words, or None for the generic access (an
    /// effect): a generic mark, a key that is no index, or receivers by rule R (the record's where
    /// it holds any, else the in-line set's) other than Arrays and ordinary objects.
    fn element_site(&self, pc: usize) -> Option<ElemSite> {
        self.rules().element_site(pc)
    }

    /// Whether the exit profile flagged at the element site at `pc` one of `reasons`, whose
    /// check no in-line form of the site goes without (the receiver's, the key's, a write's
    /// bounds and growth): the generic access, as JSC's parser gives up an array access at its
    /// BadIndexingType and OutOfBounds exit sites (DFGByteCodeParser.cpp:3208-3211, 13830-13835).
    /// The first flagged reason is listed.
    fn declines_element(&mut self, pc: usize, reasons: &[ExitReason]) -> bool {
        reasons
            .iter()
            .any(|&reason| self.declined(pc, reason, "the generic access"))
    }

    /// A load's handling of holes and of indices past the used words, widened where the exit
    /// profile flagged them, as JSC widens an array access's mode through the ArrayProfile its
    /// exits update (getArrayMode, DFGByteCodeParser.cpp:1346-1360; ArrayMode::refine turns an
    /// in-bounds mode out-of-bounds at a hole-load exit site, DFGArrayMode.cpp:202-330): an
    /// OutOfBounds flag reads past the used words, a Hole flag reads holes, each undefined under
    /// the prototype proof, which any receiver of element words can meet.
    fn flagged_reads(&mut self, pc: usize, reads: Reads) -> Reads {
        let past = reads != Reads::PastUsed
            && self.declined(pc, ExitReason::OutOfBounds, "the read past the used words");
        let holes = reads == Reads::InBounds
            && self.declined(pc, ExitReason::Hole, "the hole read as undefined");
        if past {
            Reads::PastUsed
        } else if holes {
            Reads::Holes
        } else {
            reads
        }
    }

    /// Whether a lowered element access can take `receiver` and `key`: a word that may be an
    /// object (no constant, no unboxed value), and a key that may be an index (a Number value, or
    /// a constant one).
    fn element_operands(&self, receiver: NodeIx, key: NodeIx) -> bool {
        self.rep(receiver) == Rep::Tagged
            && self.constant(receiver).is_none()
            && match self.constant(key) {
                Some(word) => const_index(word).is_some(),
                None => matches!(self.rep(key), Rep::Tagged | Rep::F64 | Rep::I32),
            }
    }

    /// The cell of the object `word` names, with elements of `kind`: CheckObject [NotObject] and
    /// CheckElements [WrongElements] unless the value facts prove them (a CheckShape this epoch
    /// proves an ordinary kind with IC_PLAIN set, element words).
    fn elements_cell(
        &mut self,
        st: &mut State,
        cx: &mut Before,
        word: NodeIx,
        kind: ElemKind,
    ) -> NodeIx {
        let cell = self.object_cell(st, cx, word);
        self.elements_kind(st, cx, word, cell, kind);
        cell
    }

    /// The cell of the object `word` names: its cell form, or CheckObject [NotObject].
    fn object_cell(&mut self, st: &mut State, cx: &mut Before, word: NodeIx) -> NodeIx {
        if let Some(cell) = self.forms(st, word).cell {
            return cell;
        }
        let state = self.before(cx, st);
        let cell = self.add_exit(
            NodeOp::CheckObject,
            &[word],
            Rep::Cell,
            cx.pc,
            state,
            ExitReason::NotObject,
        );
        self.relate(st, word, cell);
        cell
    }

    /// CheckElements(`kind`) of the object `word` names (its cell `cell`) [WrongElements], unless
    /// the value facts prove it.
    fn elements_kind(
        &mut self,
        st: &mut State,
        cx: &mut Before,
        word: NodeIx,
        cell: NodeIx,
        kind: ElemKind,
    ) {
        let proven = st.kinds.get(&word).is_some_and(|known| known.proves(kind))
            || (kind == ElemKind::Words && st.shapes.contains_key(&word));
        if !proven {
            let state = self.before(cx, st);
            self.add_exit(
                NodeOp::CheckElements(kind),
                &[cell],
                Rep::None,
                cx.pc,
                state,
                ExitReason::WrongElements,
            );
            st.kinds.insert(word, kind);
        }
    }

    /// The element index `key` names, an int32: an int32 form as it is, a constant folded, an
    /// f64 or a word checked a Number converted by CheckIndex [NotInt32], which takes -0 to index
    /// 0 as Warp's GuardToInt32Index does (WarpCacheIRTranspiler.cpp:1572-1583); cached per value
    /// beside the forms, as truncations are. `key` passed `element_operands`.
    fn element_index(&mut self, st: &mut State, cx: &mut Before, key: NodeIx) -> NodeIx {
        if let Some(int) = self.forms(st, key).i32 {
            return int;
        }
        if let Some(index) = self.constant(key).and_then(const_index) {
            return self.hoist(Hoisted::I32(index));
        }
        let number = match self.rep(key) {
            Rep::Tagged => self
                .number(st, cx, key)
                .expect("a word that is no constant may be a Number"),
            _ => key,
        };
        if let Some(int) = self.forms(st, number).i32 {
            return int;
        }
        if let Some(&int) = st.indices.get(&number) {
            return int;
        }
        let state = self.before(cx, st);
        let int = self.add_exit(
            NodeOp::CheckIndex,
            &[number],
            Rep::I32,
            cx.pc,
            state,
            ExitReason::NotInt32,
        );
        st.indices.insert(number, int);
        int
    }

    /// The key an element index has in the epoch's element facts: a checked conversion's f64, so
    /// that CheckIndex and CheckInt32 of one Number name one element (`a[j]` read, then
    /// `a[j++] = v` written). They agree wherever both pass: CheckInt32 exits on -0, the one
    /// value they convert apart.
    fn index_key(&self, index: NodeIx) -> NodeIx {
        let node = self.graph.node(index);
        match node.op {
            NodeOp::CheckIndex | NodeOp::CheckInt32 => node.inputs[0],
            _ => index,
        }
    }

    /// The element words of the object `word` names (its cell `cell`), loaded once per effect
    /// epoch: Maglev's BuildLoadElements, a known property until a store kills it
    /// (maglev-reducer-inl.h:1251-1275).
    fn elements_of(&mut self, st: &mut State, word: NodeIx, cell: NodeIx, pc: usize) -> NodeIx {
        if let Some(&words) = st.elements.get(&word) {
            return words;
        }
        let words = self.add(NodeOp::LoadElements, &[cell], Rep::Ptr, pc);
        st.elements.insert(word, words);
        words
    }

    /// The used length of the element words at `words`, loaded once per effect epoch.
    fn used_of(&mut self, st: &mut State, words: NodeIx, pc: usize) -> NodeIx {
        if let Some(&used) = st.used.get(&words) {
            return used;
        }
        let used = self.add(NodeOp::LoadElementsUsed, &[words], Rep::I32, pc);
        st.used.insert(words, used);
        used
    }

    /// The element at `index` of the object `word` names (its cell `cell`, whose elements kind
    /// is proven), read once per effect epoch, with the bounds check and the hole handling that
    /// `reads` says.
    fn element_read(
        &mut self,
        st: &mut State,
        cx: &mut Before,
        word: NodeIx,
        cell: NodeIx,
        index: NodeIx,
        reads: Reads,
    ) -> NodeIx {
        let pc = cx.pc;
        let words = self.elements_of(st, word, cell, pc);
        let key = self.index_key(index);
        if let Some(&value) = st.loaded.get(&(words, key)) {
            return value;
        }
        let value = if st.present.contains(&(words, key)) {
            // In bounds and no hole, as a load or a store this epoch proved.
            let state = self.before(cx, st);
            self.add_exit(
                NodeOp::LoadElement(Holes::Exit),
                &[cell, words, index],
                Rep::Tagged,
                pc,
                state,
                ExitReason::Hole,
            )
        } else if reads == Reads::PastUsed {
            let used = self.used_of(st, words, pc);
            let state = self.before(cx, st);
            self.add_exit(
                NodeOp::LoadElementOrUndefined,
                &[cell, words, index, used],
                Rep::Tagged,
                pc,
                state,
                ExitReason::OutOfBounds,
            )
        } else {
            let used = self.used_of(st, words, pc);
            let state = self.before(cx, st);
            self.add_exit(
                NodeOp::CheckBounds,
                &[index, used],
                Rep::None,
                pc,
                state,
                ExitReason::OutOfBounds,
            );
            let holes = if reads == Reads::Holes {
                Holes::Undefined
            } else {
                Holes::Exit
            };
            let value = self.add_exit(
                NodeOp::LoadElement(holes),
                &[cell, words, index],
                Rep::Tagged,
                pc,
                state,
                ExitReason::Hole,
            );
            if holes == Holes::Exit {
                st.present.insert((words, key));
            }
            value
        };
        st.loaded.insert((words, key), value);
        value
    }

    /// GetElem and GetElemLocal on element words (Maglev's element load on a JSArray or JSObject,
    /// maglev-graph-builder.cc:5213-5290, as Warp loads dense elements, WarpCacheIRTranspiler.cpp:
    /// 2339-2381): the receiver's kind checked unless proven, its words and their used length
    /// loaded once per effect epoch, the key as an int32 index, an unsigned bounds check and the
    /// word with its hole test. The word is checked at each use, as Warp unboxes it per use.
    fn element_load(&mut self, pc: usize, recv: Recv, st: &mut State) -> Result<(), Refusal> {
        let Some(mut site) = self.element_site(pc) else {
            return self.effect(pc, st);
        };
        let len = st.stack.len();
        let key = st.stack[len - 1];
        let receiver = self.receiver(recv, st, 1, pc);
        if self.declines_element(pc, LOAD_DECLINES) {
            return self.effect(pc, st);
        }
        if !self.element_operands(receiver, key) {
            return self.mismatch(pc, ExitReason::WrongElements, st);
        }
        let mut past_length = false;
        match site.typed {
            Typed::None => site.reads = self.flagged_reads(pc, site.reads),
            Typed::Only(Some(_)) => {
                if self.declines_element(pc, &[ExitReason::Detached]) {
                    return self.effect(pc, st);
                }
                past_length = self.declined(
                    pc,
                    ExitReason::OutOfBounds,
                    "the read past the view's length",
                );
            }
            Typed::Only(None) | Typed::Mixed => {
                if self.declines_element(pc, TYPED_DECLINES) {
                    return self.effect(pc, st);
                }
                site.reads = self.flagged_reads(pc, site.reads);
            }
        }
        let mut cx = Before::new(pc);
        let value = match site.typed {
            Typed::None => {
                let cell = self.elements_cell(st, &mut cx, receiver, site.kind);
                let index = self.element_index(st, &mut cx, key);
                self.element_read(st, &mut cx, receiver, cell, index, site.reads)
            }
            Typed::Only(kind) => {
                let cell = self.elements_cell(st, &mut cx, receiver, ElemKind::Typed(kind));
                let index = self.element_index(st, &mut cx, key);
                self.typed_read(st, &mut cx, receiver, cell, index, kind, past_length)
            }
            Typed::Mixed => {
                let cell = self.object_cell(st, &mut cx, receiver);
                let index = self.element_index(st, &mut cx, key);
                let state = self.before(&mut cx, st);
                let typed = self.add(NodeOp::IsTypedArray, &[cell], Rep::Bool, pc);
                self.branch_on_typed(
                    st,
                    pc,
                    typed,
                    |this, st| {
                        let mut cx = Before::with(pc, state);
                        Some(this.typed_read(st, &mut cx, receiver, cell, index, None, false))
                    },
                    |this, st| {
                        let mut cx = Before::with(pc, state);
                        this.elements_kind(st, &mut cx, receiver, cell, site.kind);
                        let value =
                            this.element_read(st, &mut cx, receiver, cell, index, site.reads);
                        Some(this.tagged(st, value, pc))
                    },
                )
                .expect("both arms read a value")
            }
        };
        let pops = if matches!(recv, Recv::Stack) { 2 } else { 1 };
        st.stack.truncate(len - pops);
        st.stack.push(value);
        Ok(())
    }

    /// The representation an element store takes `value` in: its int32 form, its f64 form, a
    /// constant Number as either, else its word (a Boolean's boxed).
    fn element_value(&mut self, st: &mut State, value: NodeIx, pc: usize) -> NodeIx {
        let forms = self.forms(st, value);
        if let Some(int) = forms.i32 {
            return int;
        }
        if let Some(number) = forms.f64 {
            return number;
        }
        match self.constant(value).and_then(number_word) {
            Some(x) => match exact_i32(x) {
                Some(int) => self.hoist(Hoisted::I32(int)),
                None => self.hoist(Hoisted::F64(x.to_bits())),
            },
            None => self.tagged(st, value, pc),
        }
    }

    /// SetElem, SetElemDrop, SetElemLocal and SetElemLocalDrop to an existing element of element
    /// words (Maglev's element store on a JSArray or JSObject, maglev-graph-builder.cc:5346-5461,
    /// as Warp's StoreDenseElement, WarpCacheIRTranspiler.cpp:3167-3194): the receiver and the
    /// key as a load takes them, then StoreElement (`words_store`); a typed array's element by
    /// its kind or through the template's stub, its value a Number (`typed_write`); or, for
    /// typed arrays among other receivers, a branch on the kind tag between the two.
    fn element_store(
        &mut self,
        pc: usize,
        recv: Recv,
        keep: bool,
        st: &mut State,
    ) -> Result<(), Refusal> {
        let Some(mut site) = self.element_site(pc) else {
            return self.effect(pc, st);
        };
        let len = st.stack.len();
        let (key, value) = (st.stack[len - 2], st.stack[len - 1]);
        let receiver = self.receiver(recv, st, 2, pc);
        if self.declines_element(pc, STORE_DECLINES)
            || (site.typed != Typed::None && self.declines_element(pc, TYPED_DECLINES))
        {
            return self.effect(pc, st);
        }
        if !self.element_operands(receiver, key) {
            return self.mismatch(pc, ExitReason::WrongElements, st);
        }
        if !self.numeric(value) {
            // A typed array takes a Number in line (its store converts any other value first,
            // which may run code): the words alone, where the site saw others.
            match site.typed {
                Typed::Only(_) => return self.mismatch(pc, ExitReason::NotNumber, st),
                Typed::Mixed => site.typed = Typed::None,
                Typed::None => {}
            }
        }
        let mut cx = Before::new(pc);
        match site.typed {
            Typed::None => {
                let cell = self.elements_cell(st, &mut cx, receiver, site.kind);
                let index = self.element_index(st, &mut cx, key);
                self.words_store(st, &mut cx, receiver, cell, index, value, site.arrays);
            }
            Typed::Only(kind) => {
                let cell = self.elements_cell(st, &mut cx, receiver, ElemKind::Typed(kind));
                let index = self.element_index(st, &mut cx, key);
                self.typed_write(st, &mut cx, receiver, cell, index, value, kind);
            }
            Typed::Mixed => {
                let cell = self.object_cell(st, &mut cx, receiver);
                let index = self.element_index(st, &mut cx, key);
                let state = self.before(&mut cx, st);
                let typed = self.add(NodeOp::IsTypedArray, &[cell], Rep::Bool, pc);
                self.branch_on_typed(
                    st,
                    pc,
                    typed,
                    |this, st| {
                        let mut cx = Before::with(pc, state);
                        this.typed_write(st, &mut cx, receiver, cell, index, value, None);
                        None
                    },
                    |this, st| {
                        let mut cx = Before::with(pc, state);
                        this.elements_kind(st, &mut cx, receiver, cell, site.kind);
                        this.words_store(st, &mut cx, receiver, cell, index, value, site.arrays);
                        None
                    },
                );
            }
        }
        let pops = if matches!(recv, Recv::Stack) { 3 } else { 2 };
        st.stack.truncate(len - pops);
        if keep {
            st.stack.push(value);
        }
        Ok(())
    }

    /// StoreElement of `value` at `index` of the element words of the object `word` names (its
    /// cell `cell`, whose kind is proven): Present where the epoch proved the element, Create for
    /// an Array site, else Existing. A store may copy or grow a block, and another word may name
    /// its object, so every element address and read of the epoch ends with it but its own words,
    /// whose element it now knows; no store makes a hole, so the present elements stay.
    #[allow(clippy::too_many_arguments)]
    fn words_store(
        &mut self,
        st: &mut State,
        cx: &mut Before,
        word: NodeIx,
        cell: NodeIx,
        index: NodeIx,
        value: NodeIx,
        arrays: bool,
    ) {
        let pc = cx.pc;
        let words = self.elements_of(st, word, cell, pc);
        let stored = self.element_value(st, value, pc);
        let key = self.index_key(index);
        let mode = if st.present.contains(&(words, key)) {
            Store::Present
        } else if arrays {
            Store::Create
        } else {
            Store::Existing
        };
        let state = self.before(cx, st);
        let after = self.add_exit(
            NodeOp::StoreElement(mode),
            &[cell, words, index, stored],
            Rep::Ptr,
            pc,
            state,
            ExitReason::WrongElements,
        );
        st.elements.clear();
        st.used.clear();
        st.loaded.clear();
        if mode == Store::Create {
            // An append raises an Array's `length`, whichever word names the Array.
            st.lengths.clear();
        }
        st.elements.insert(word, after);
        st.present.insert((after, key));
        st.loaded.insert((after, key), value);
    }

    /// The view of the typed array `word` names (its cell `cell`) and the view's length, value
    /// facts.
    fn typed_view(
        &mut self,
        st: &mut State,
        word: NodeIx,
        cell: NodeIx,
        pc: usize,
    ) -> (NodeIx, NodeIx) {
        let view = match st.views.get(&word) {
            Some(&view) => view,
            None => {
                let view = self.add(NodeOp::TypedView, &[cell], Rep::Ptr, pc);
                st.views.insert(word, view);
                view
            }
        };
        let length = match st.typed_lengths.get(&view) {
            Some(&length) => length,
            None => {
                let length = self.add(NodeOp::TypedLength, &[view], Rep::I32, pc);
                st.typed_lengths.insert(view, length);
                length
            }
        };
        (view, length)
    }

    /// The elements' address of the typed array `cell` (its view `view`, its element kind
    /// proven), checked valid, for a store too where `write` [Detached]: a heap fact of the
    /// epoch, which a store's check serves a load too.
    fn typed_data(
        &mut self,
        st: &mut State,
        cx: &mut Before,
        cell: NodeIx,
        view: NodeIx,
        write: bool,
    ) -> NodeIx {
        let known = st
            .typed_data
            .get(&(view, true))
            .or_else(|| st.typed_data.get(&(view, false)).filter(|_| !write));
        if let Some(&data) = known {
            return data;
        }
        let state = self.before(cx, st);
        let data = self.add_exit(
            NodeOp::TypedData { write },
            &[cell, view],
            Rep::Ptr,
            cx.pc,
            state,
            ExitReason::Detached,
        );
        st.typed_data.insert((view, write), data);
        data
    }

    /// A typed array's view, the index's bounds check [OutOfBounds] and the elements' address
    /// (`typed_data`): the view and the address.
    fn typed_access(
        &mut self,
        st: &mut State,
        cx: &mut Before,
        word: NodeIx,
        cell: NodeIx,
        index: NodeIx,
        write: bool,
    ) -> (NodeIx, NodeIx) {
        let pc = cx.pc;
        let (view, length) = self.typed_view(st, word, cell, pc);
        let state = self.before(cx, st);
        self.add_exit(
            NodeOp::CheckBounds,
            &[index, length],
            Rep::None,
            pc,
            state,
            ExitReason::OutOfBounds,
        );
        let data = self.typed_data(st, cx, cell, view, write);
        (view, data)
    }

    /// The element of a typed array (its kind proven) at `index`: by its element kind where the
    /// record saw one (Maglev's typed-array loads), undefined past the length where `past_length`
    /// (the site exited there), else through the template's TypedElement stub [OutOfBounds],
    /// whose word it is.
    #[allow(clippy::too_many_arguments)]
    fn typed_read(
        &mut self,
        st: &mut State,
        cx: &mut Before,
        word: NodeIx,
        cell: NodeIx,
        index: NodeIx,
        kind: Option<TypedKind>,
        past_length: bool,
    ) -> NodeIx {
        let pc = cx.pc;
        let Some(kind) = kind else {
            let state = self.before(cx, st);
            return self.add_exit(
                NodeOp::TypedElement { write: false },
                &[cell, index],
                Rep::Tagged,
                pc,
                state,
                ExitReason::OutOfBounds,
            );
        };
        if past_length {
            let (view, length) = self.typed_view(st, word, cell, pc);
            let data = self.typed_data(st, cx, cell, view, false);
            return self.add(
                NodeOp::LoadTypedOrUndefined(kind),
                &[cell, data, index, length],
                Rep::Tagged,
                pc,
            );
        }
        let (_, data) = self.typed_access(st, cx, word, cell, index, false);
        self.add(
            NodeOp::LoadTyped(kind),
            &[cell, data, index],
            kind.rep(false),
            pc,
        )
    }

    /// A typed array's store of `value`, a Number (`numeric`), at `index`: converted first, as
    /// Maglev converts it (an integer kind's ToInt32 truncation, Uint8Clamped's clamp, a float's
    /// f64), then stored by its kind; or through the template's stub where the record saw no one
    /// kind [OutOfBounds].
    #[allow(clippy::too_many_arguments)]
    fn typed_write(
        &mut self,
        st: &mut State,
        cx: &mut Before,
        word: NodeIx,
        cell: NodeIx,
        index: NodeIx,
        value: NodeIx,
        kind: Option<TypedKind>,
    ) {
        let pc = cx.pc;
        let Some(kind) = kind else {
            let number = self.number(st, cx, value).expect("a Number value");
            let state = self.before(cx, st);
            self.add_exit(
                NodeOp::TypedElement { write: true },
                &[cell, index, number],
                Rep::None,
                pc,
                state,
                ExitReason::OutOfBounds,
            );
            return;
        };
        let stored = match kind {
            TypedKind::F32 | TypedKind::F64 => self.number(st, cx, value).expect("a Number value"),
            TypedKind::U8Clamped => {
                let number = self.number(st, cx, value).expect("a Number value");
                self.add(NodeOp::ClampToUint8, &[number], Rep::I32, pc)
            }
            _ => self.truncated_int32(st, cx, value).expect("a Number value"),
        };
        let (view, data) = self.typed_access(st, cx, word, cell, index, true);
        self.add(
            NodeOp::StoreTyped(kind),
            &[cell, view, data, index, stored],
            Rep::None,
            pc,
        );
    }

    /// Branch on `typed` (IsTypedArray) to `on_typed` and `on_words`, each built in a block of
    /// its own on a copy of `st`, then join them: the frame as it was (neither arm changes it),
    /// the facts both arms hold, and their values, where they give some, as a phi of words.
    fn branch_on_typed(
        &mut self,
        st: &mut State,
        pc: usize,
        typed: NodeIx,
        on_typed: impl FnOnce(&mut Self, &mut State) -> Option<NodeIx>,
        on_words: impl FnOnce(&mut Self, &mut State) -> Option<NodeIx>,
    ) -> Option<NodeIx> {
        let first = self.cur;
        let typed_block = self.graph.add_block(pc as u32, false);
        let words_block = self.graph.add_block(pc as u32, false);
        let join = self.graph.add_block(pc as u32, false);
        self.add(
            NodeOp::Branch {
                then_block: typed_block,
                else_block: words_block,
            },
            &[typed],
            Rep::None,
            pc,
        );
        self.graph.add_edge(first, typed_block);
        self.graph.add_edge(first, words_block);
        let mut typed_st = st.clone();
        self.cur = typed_block;
        let a = on_typed(self, &mut typed_st);
        self.add(NodeOp::Jump(join), &[], Rep::None, pc);
        self.graph.add_edge(typed_block, join);
        let mut words_st = st.clone();
        self.cur = words_block;
        let b = on_words(self, &mut words_st);
        self.add(NodeOp::Jump(join), &[], Rep::None, pc);
        self.graph.add_edge(words_block, join);
        self.cur = join;
        let met = meet(&[(typed_block, typed_st), (words_block, words_st)]);
        *st = State {
            slots: std::mem::take(&mut st.slots),
            stack: std::mem::take(&mut st.stack),
            handlers: std::mem::take(&mut st.handlers),
            ..met
        };
        let (a, b) = (a?, b?);
        Some(
            self.graph
                .add_node(join, NodeOp::Phi, &[a, b], Rep::Tagged, pc as u32),
        )
    }

    /// UpdateElem (`a[i]++`, `--a[i]`) where its value record says Number: the receiver's element
    /// words (UpdateElem's record is a value record, which names no receiver), the element read
    /// with a hole exiting, checked a Number (an int32 where the record's hint says so), one added
    /// or taken with an overflow exit, and the result written back to the element the read proved
    /// present. Every exit precedes the write, at the operation, which the VM then runs whole.
    fn update_element(&mut self, pc: usize, kind: UpdKind, st: &mut State) -> Result<(), Refusal> {
        let Some(values) = self.rules().numbers(pc, false) else {
            return self.effect(pc, st);
        };
        let len = st.stack.len();
        let (receiver, key) = (st.stack[len - 2], st.stack[len - 1]);
        if self.declines_element(pc, UPDATE_DECLINES) {
            return self.effect(pc, st);
        }
        if !self.element_operands(receiver, key) {
            return self.mismatch(pc, ExitReason::WrongElements, st);
        }
        let mut cx = Before::new(pc);
        let cell = self.elements_cell(st, &mut cx, receiver, ElemKind::Words);
        let index = self.element_index(st, &mut cx, key);
        let old = self.element_read(st, &mut cx, receiver, cell, index, Reads::InBounds);
        let increment = matches!(
            kind,
            UpdKind::PreInc | UpdKind::PostInc | UpdKind::IncDiscard
        );
        // Overflow flagged: f64 even on a proven int32 (step 3's makeSafe rule).
        let int = (self.known_int32(st, old)
            || (int32_hint(&values) && self.int32_operand(st, old)))
            && !self.declined(pc, ExitReason::Overflow, "f64 arithmetic");
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
                    return Err(Refusal::at(pc, "a Number element without an f64 form"));
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
        let words = self.elements_of(st, receiver, cell, pc);
        let key = self.index_key(index);
        let mode = if st.present.contains(&(words, key)) {
            Store::Present
        } else {
            Store::Existing
        };
        let state = self.before(&mut cx, st);
        let after = self.add_exit(
            NodeOp::StoreElement(mode),
            &[cell, words, index, new],
            Rep::Ptr,
            pc,
            state,
            ExitReason::WrongElements,
        );
        st.elements.clear();
        st.used.clear();
        st.loaded.clear();
        st.elements.insert(receiver, after);
        st.present.insert((after, key));
        st.loaded.insert((after, key), new);
        st.stack.truncate(len - 2);
        match kind {
            UpdKind::PreInc | UpdKind::PreDec => st.stack.push(new),
            UpdKind::PostInc | UpdKind::PostDec => st.stack.push(number),
            UpdKind::IncDiscard | UpdKind::DecDiscard => {}
        }
        Ok(())
    }

    /// Whether the named load of `name` at `pc` reads Arrays' `length`: its ways all the Array's
    /// key-checked first slot, which own_data_ways refuses (an Array's `length` is its slot 0:
    /// value.rs length_property), and no generic mark.
    fn array_length_site(&self, pc: usize, name: u32) -> bool {
        self.rules().array_length_site(pc, name)
    }

    /// GetProp, GetPropThis and GetPropLocal of an Array's `length` (`array_length_site`): the
    /// receiver checked an Array (CheckObject [NotObject], CheckElements [WrongElements]), then its
    /// first slot's word, a heap fact that an append and an effect end.
    fn array_length(&mut self, pc: usize, recv: Recv, st: &mut State) -> Result<(), Refusal> {
        let receiver = self.receiver(recv, st, 0, pc);
        if self.declines_element(pc, &[ExitReason::WrongElements, ExitReason::NotObject]) {
            return self.effect(pc, st);
        }
        if self.rep(receiver) != Rep::Tagged || self.constant(receiver).is_some() {
            return self.mismatch(pc, ExitReason::WrongElements, st);
        }
        let mut cx = Before::new(pc);
        let cell = self.elements_cell(st, &mut cx, receiver, ElemKind::Array);
        let length = match st.lengths.get(&receiver) {
            Some(&length) => length,
            None => {
                let length = self.add(NodeOp::LoadArrayLength, &[cell], Rep::Tagged, pc);
                st.lengths.insert(receiver, length);
                length
            }
        };
        if matches!(recv, Recv::Stack) {
            st.stack.pop();
        }
        st.stack.push(length);
        Ok(())
    }

    /// ToPropKey and ToPropKeyLocal (`a[k] op= v`): nothing for a Number key over an object, as
    /// the template passes a Number on (jit_operations.rs): the base checked an object
    /// [NotObject], since the operation throws for undefined and null, which the VM then does at
    /// this pc, and the key a proven Number, or a word checked one [NotNumber] where the element
    /// load that follows is lowered (its record saw index keys only). Anything else is the
    /// operation, an effect.
    fn to_prop_key(&mut self, pc: usize, recv: Recv, st: &mut State) -> Result<(), Refusal> {
        let len = st.stack.len();
        let key = st.stack[len - 1];
        let base = self.receiver(recv, st, 1, pc);
        let proven = self.known_number(st, key);
        let checked = !proven && self.rep(key) == Rep::Tagged && self.constant(key).is_none();
        if !self.lowered_load_follows(pc)
            || self.declines_element(pc, &[ExitReason::NotObject, ExitReason::NotNumber])
        {
            return self.effect(pc, st);
        }
        if self.rep(base) != Rep::Tagged || self.constant(base).is_some() {
            return self.mismatch(pc, ExitReason::NotObject, st);
        }
        if !(proven || checked) {
            return self.mismatch(pc, ExitReason::NotNumber, st);
        }
        let mut cx = Before::new(pc);
        if self.forms(st, base).cell.is_none() {
            let state = self.before(&mut cx, st);
            let cell = self.add_exit(
                NodeOp::CheckObject,
                &[base],
                Rep::Cell,
                pc,
                state,
                ExitReason::NotObject,
            );
            self.relate(st, base, cell);
        }
        if checked {
            self.number(st, &mut cx, key);
        }
        Ok(())
    }

    /// Whether the first element load after `pc` in its block (a compound assignment's GetElem)
    /// is lowered on element words.
    fn lowered_load_follows(&self, pc: usize) -> bool {
        self.rules().lowered_load_follows(pc)
    }
}
