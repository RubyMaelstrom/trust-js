//! The tier's summary of a chunk's feedback (the plan's Design point 3 and the Stage 1 record's
//! front end): one view of the read API (`crate::feedback::read`) per pc, copied once when a
//! compile starts, so that the builder reads only this snapshot and the chunk's immutable
//! operations, as V8's ProcessedFeedback, JSC's status objects and SpiderMonkey's WarpSnapshot
//! serve their builders. Taking it allocates the two per-pc arrays and nothing else; it interns
//! nothing (the views hold process shape ids and cell addresses).
//!
//! A transformed chunk (a second stage, its hot callees spliced in) keeps its own property and
//! call caches, which the second stage seeded from the first stage and goes on filling, but its
//! value and element records are its baseline sites' (W4 step 7: its own record words only
//! filter what it forwards): the snapshot reads those through the chunk's record map.
//!
//! The evidence that code ran comes from named and call sites only: their first execution fills a
//! way or sets the generic or megamorphic mark, and a second stage's seeded caches count. Value
//! and element records are no evidence, since in-line hits record nothing (rule U).

use crate::bytecode::{Chunk, Op};
use crate::feedback::{self, CallState, NamedView, RecordMap, SiteView, ValuesView};

/// Whether a pc's feedback shows that its operation ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ran {
    /// The feedback says nothing either way: an operation without a property or call cache, or a
    /// `length` read without a way (a String's length is read by its tag and fills none).
    Unknown,
    /// A named site with a filled way or its generic mark, or a call site that is not empty.
    Yes,
    /// A named or call site holding neither: it never ran.
    No,
}

/// The per-pc feedback one compile reads (see the module's doc).
pub(crate) struct Snapshot {
    views: Box<[SiteView]>,
    ran: Box<[Ran]>,
}

impl Snapshot {
    /// The feedback of every operation of `chunk` (its `jit_ops`), as it stands now.
    pub(crate) fn take(chunk: &Chunk) -> Snapshot {
        let ops = chunk.jit_ops();
        let map = chunk.feedback_vector().record_map();
        let mut views = Vec::with_capacity(ops.len());
        let mut ran = Vec::with_capacity(ops.len());
        for (pc, op) in ops.iter().enumerate() {
            let view = view(chunk, map, pc);
            ran.push(evidence(chunk, op, &view));
            views.push(view);
        }
        Snapshot {
            views: views.into_boxed_slice(),
            ran: ran.into_boxed_slice(),
        }
    }

    /// The operation count the snapshot covers.
    pub(crate) fn len(&self) -> usize {
        self.views.len()
    }

    /// The view of the operation at `pc` (None past the end).
    pub(crate) fn site(&self, pc: usize) -> SiteView {
        self.views.get(pc).copied().unwrap_or(SiteView::None)
    }

    /// Whether the operation at `pc` ran, by its feedback.
    pub(crate) fn ran(&self, pc: usize) -> Ran {
        self.ran.get(pc).copied().unwrap_or(Ran::Unknown)
    }

    /// The value classes of an arithmetic, unary, update or relational operation (an update of a
    /// named property's too).
    pub(crate) fn values(&self, pc: usize) -> Option<ValuesView> {
        match self.site(pc) {
            SiteView::Values(values) | SiteView::NamedUpdate(_, values) => Some(values),
            _ => None,
        }
    }

    /// The ways of a named load, store or update.
    pub(crate) fn named(&self, pc: usize) -> Option<NamedView> {
        match self.site(pc) {
            SiteView::Named(named) | SiteView::NamedUpdate(named, _) => Some(named),
            _ => None,
        }
    }
}

/// The view of `pc` the tier reads: the chunk's own, with a transformed chunk's value and element
/// records its baseline site's (`map` is the chunk's record map).
fn view(chunk: &Chunk, map: Option<&RecordMap>, pc: usize) -> SiteView {
    let own = feedback::read(chunk, pc);
    match map.and_then(|map| map.target(pc)) {
        Some((baseline, baseline_pc)) => {
            with_baseline_records(own, feedback::read(baseline, baseline_pc))
        }
        None => own,
    }
}

/// A transformed chunk's view of a site with its baseline site's value or element record in place
/// of its own (which holds only the in-line set: a transformed chunk reads no record of its own).
fn with_baseline_records(own: SiteView, baseline: SiteView) -> SiteView {
    match (own, baseline) {
        (SiteView::Values(_), SiteView::Values(values)) => SiteView::Values(values),
        (SiteView::Element(_), SiteView::Element(element)) => SiteView::Element(element),
        (SiteView::NamedUpdate(named, _), SiteView::NamedUpdate(_, values)) => {
            SiteView::NamedUpdate(named, values)
        }
        (own, _) => own,
    }
}

/// The evidence the view of `op` gives (see [`Ran`]).
fn evidence(chunk: &Chunk, op: &Op, view: &SiteView) -> Ran {
    match view {
        SiteView::Named(named) | SiteView::NamedUpdate(named, _) => {
            if named.generic || named.filled().next().is_some() {
                Ran::Yes
            } else if is_length_read(chunk, op) {
                Ran::Unknown
            } else {
                Ran::No
            }
        }
        SiteView::Call(call) if call.state == CallState::Empty => Ran::No,
        SiteView::Call(_) => Ran::Yes,
        _ => Ran::Unknown,
    }
}

/// Whether `op` reads a property named `length` (which a String receiver reads without a way).
fn is_length_read(chunk: &Chunk, op: &Op) -> bool {
    match *op {
        Op::GetProp(n, _)
        | Op::GetPropThis(n, _)
        | Op::GetPropLocal(_, n, _)
        | Op::GetMethod(n, _) => chunk.jit_name(n) == "length",
        _ => false,
    }
}

/// The feedback fingerprint the settled-feedback rule compares between a chunk's checkpoints (the
/// Stage 1 record's "When", V8's wait for a quiet IC): its named sites' ways and generic marks, its
/// call sites' ways and megamorphic states, and its value and element records, read as the
/// snapshot reads them (a transformed chunk's records are its baseline sites', which its misses
/// go on filling; its own never change). A call way's epoch (`CallTarget::current`) is left out:
/// an epoch moved by another function's code changes no feedback of this one.
pub(crate) fn fingerprint(chunk: &Chunk) -> u64 {
    use std::hash::Hasher;
    let mut hash = crate::fasthash::FxHasher::default();
    let map = chunk.feedback_vector().record_map();
    for pc in 0..chunk.jit_ops().len() {
        hash.write_usize(pc);
        match view(chunk, map, pc) {
            SiteView::Named(named) => hash_named(&mut hash, &named),
            SiteView::NamedUpdate(named, values) => {
                hash_named(&mut hash, &named);
                hash_values(&mut hash, &values);
            }
            SiteView::Values(values) => hash_values(&mut hash, &values),
            SiteView::Element(element) => {
                hash.write_u32(element.recorded_receivers);
                hash.write_u32(element.recorded_keys);
                hash.write_u32(element.facts);
                hash.write_u8(u8::from(element.generic));
            }
            SiteView::Call(call) => {
                hash.write_u8(call.state as u8);
                for target in call.ways.iter().flatten() {
                    hash.write_usize(target.callee);
                    hash.write_usize(target.function as usize);
                    hash.write_u32(u32::from(target.builtin));
                }
            }
            SiteView::Construct(construct) => {
                if let Some(target) = construct.target {
                    hash.write_usize(target.callee);
                }
                hash.write_u32(construct.prototype_shape);
            }
            SiteView::Allocation(_) | SiteView::None => {}
        }
    }
    hash.finish()
}

fn hash_named(hash: &mut impl std::hash::Hasher, named: &NamedView) {
    hash.write_u8(u8::from(named.generic));
    for way in named.filled() {
        hash.write_u32(way.receiver_shape);
        hash.write_u32(way.holder_shape);
        hash.write_u32(way.slot);
        hash.write_u8(way.depth);
        hash.write_u8(way.kind as u8);
    }
}

fn hash_values(hash: &mut impl std::hash::Hasher, values: &ValuesView) {
    for role in values.recorded {
        hash.write_u32(role);
    }
}
