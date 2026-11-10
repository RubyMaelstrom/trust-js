//! The read API (W4 step 6; feedback-schema.txt section 5.4): one typed view of a site's feedback,
//! for a chunk and a baseline pc, as the speculative tier (the plan's Phase 3) reads it. A view
//! allocates nothing and interns nothing: its shapes are the process's shape ids and its targets
//! the process's cells, which P3, running in the process that made them, compares directly. The
//! detailed vector (section 3) stays a diagnostic; the census (section 4) reads the same state.
//!
//! Which parts are exact and which are hints, and when a view goes stale:
//! - values and elements: rule U (the record together with the operation's in-line set) is
//!   complete; the record alone is exact for what it holds, and its int32/double split and an
//!   Array's facts are hints (the in-line paths record nothing);
//! - named ways: exact when read, each a proof generated code re-validates (shapes are never
//!   reused); a way may name a shape no live object has any more;
//! - call targets: exact for the call-cache epoch they were filled in (`CallTarget::current`); an
//!   older entry's chunk and code may have been replaced, though its callee and literal stay;
//! - allocation: whether the site's image and template exist, which only appear.
// The speculative tier is the API's consumer; until it exists, the tests and the census read it.
#![cfg_attr(not(test), allow(dead_code))]

use crate::bytecode::{Chunk, PROP_IC_WAYS};

/// A site's feedback, by its operation's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SiteView {
    /// Arithmetic, unary, update (UpdateElem's too) and relational operations: the value record.
    Values(ValuesView),
    /// Element loads and stores: the element record.
    Element(ElementView),
    /// Named loads and stores: the property cache.
    Named(NamedView),
    /// A named update (`o.x++`): the property cache and the value record.
    NamedUpdate(NamedView, ValuesView),
    /// Calls with a cache (spread and argument-array calls have none).
    Call(CallView),
    /// `new` with a cache.
    Construct(ConstructView),
    /// Allocation sites.
    Allocation(AllocationView),
    /// No feedback by design: equality (W4 decision 3), typeof and `in` (lowered generically),
    /// branches, names and everything else.
    None,
}

/// The feedback of the operation at baseline `pc` of `chunk`.
pub(crate) fn read(chunk: &Chunk, pc: usize) -> SiteView {
    chunk.feedback_view(pc)
}

/// [`read`], with the ways of a transformed chunk's copies of the site merged into a named view
/// (W4 step 7): the transformed chunk's caches are its own, its record map names their baseline
/// site. A union past the site's four ways is generic.
pub(crate) fn read_merged(chunk: &Chunk, pc: usize, transformed: &Chunk) -> SiteView {
    let view = read(chunk, pc);
    let Some(map) = transformed.feedback_vector().record_map() else {
        return view;
    };
    let merge = |mut named: NamedView| {
        for source in map.sources(chunk, pc) {
            let (SiteView::Named(copy) | SiteView::NamedUpdate(copy, _)) =
                read(transformed, source)
            else {
                continue;
            };
            named.generic |= copy.generic;
            for way in copy.filled() {
                if named.filled().any(|held| held == way) {
                    continue;
                }
                match named.ways.iter_mut().find(|slot| slot.is_none()) {
                    Some(slot) => *slot = Some(*way),
                    None => named.generic = true,
                }
            }
        }
        named
    };
    match view {
        SiteView::Named(named) => SiteView::Named(merge(named)),
        SiteView::NamedUpdate(named, values) => SiteView::NamedUpdate(merge(named), values),
        other => other,
    }
}

/// The value record's view (section 5.1): ValueClass sets, a class's bit `1 << (class - 1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ValuesView {
    /// The classes each role saw by rule U: the record's together with the operation's in-line
    /// set. A role the operation lacks (a unary's operand 1, a comparison's result) is 0.
    pub(crate) operand0: u32,
    pub(crate) operand1: u32,
    pub(crate) result: u32,
    /// The record alone (operand 0, operand 1, result): what the VM and the miss paths saw.
    pub(crate) recorded: [u32; 3],
}

impl ValuesView {
    /// Whether a VM execution or a miss wrote the record.
    pub(crate) fn has_record(&self) -> bool {
        self.recorded != [0; 3]
    }
}

/// The element record's view (section 5.2), in the record's bit positions (feedback.rs ELEM_*).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ElementView {
    /// The receivers and keys the access saw by rule U: the record's together with the in-line
    /// set's (an Array, typed-array or ordinary receiver; an index key).
    pub(crate) receivers: u32,
    pub(crate) keys: u32,
    /// The record alone: its receivers, keys and an Array's facts (hints).
    pub(crate) recorded_receivers: u32,
    pub(crate) recorded_keys: u32,
    pub(crate) facts: u32,
    /// The site is generic (a key that is not an index reached an object): it is lowered to the
    /// generic keyed access, and its record stopped growing at its helpers' misses.
    pub(crate) generic: bool,
    /// The record alone: the element kinds of the typed arrays it saw (feedback.rs
    /// ELEM_TYPED_KINDS), as the VM and the misses saw them.
    pub(crate) typed_kinds: u32,
}

impl ElementView {
    /// Whether a VM execution or a miss wrote the record.
    pub(crate) fn has_record(&self) -> bool {
        self.recorded_receivers | self.recorded_keys != 0
    }
}

/// A named site's view (section 5.3): its filled ways, most recent first, and its generic mark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NamedView {
    pub(crate) ways: [Option<NamedWay>; PROP_IC_WAYS],
    /// Megamorphic or uncacheable (a refused fill or a fifth receiver shape).
    pub(crate) generic: bool,
    /// A store's or an update's site, whose Data ways are writable data properties (W4 step 9,
    /// Pilot C): a matching shape proves the property writable.
    pub(crate) writable: bool,
}

impl NamedView {
    /// The filled ways.
    pub(crate) fn filled(&self) -> impl Iterator<Item = &NamedWay> {
        self.ways.iter().flatten()
    }
}

/// One way of a named site: a proof the site's generated code re-validates on each hit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NamedWay {
    /// The receiver's shape (a creation's: the shape before the insert).
    pub(crate) receiver_shape: u32,
    /// The holder's shape (a creation's: the shape the insert lands on; an absence's: the last
    /// level's).
    pub(crate) holder_shape: u32,
    /// The kind of primitive whose wrapper was the fill's receiver (`IC_PRIMITIVE_STRING` and
    /// the rest; 0: an object of no wrapper kind): the receiver shape is then its wrapper
    /// prototype's (or a wrapper object's), and a primitive's read hits the way.
    pub(crate) primitive: u8,
    /// Prototype hops from the receiver to the holder (0: own); an absence's levels.
    pub(crate) depth: u8,
    /// The holder's slot index (0 for an absence or a creation).
    pub(crate) slot: u32,
    /// Where the slot lives for generated code (`IcState::loc`; a creation's new slot).
    pub(crate) location: u16,
    pub(crate) kind: NamedWayKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NamedWayKind {
    /// A data property, shape-proven.
    Data,
    /// A data property of an Array holder, whose hits re-check the entry's key.
    KeyChecked,
    /// An accessor, whose getter or setter a hit calls.
    Accessor,
    /// The name is absent along the whole chain.
    Absent,
    /// A store that creates the property (a constructor's `this.x = v`).
    Creation,
}

/// A call site's view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CallView {
    pub(crate) state: CallState,
    pub(crate) ways: [Option<CallTarget>; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CallState {
    Empty,
    Monomorphic,
    Polymorphic,
    /// CallSite's megamorphic state (W3 item 7): its fills evicted four live ways in an epoch.
    Megamorphic,
}

/// One target of a call or construct site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CallTarget {
    /// The callee object's cell address (its handle word).
    pub(crate) callee: usize,
    /// The callee's function literal's address (closures of one literal share it, V8's
    /// FeedbackCell), a word to compare and never to read, so that a snapshot holding views
    /// crosses threads (P3 stage 2, step 12). 0 for a native target.
    pub(crate) function: usize,
    /// A native target's builtin id and entry (0 for a user function).
    pub(crate) builtin: u16,
    pub(crate) native: usize,
    /// A user function's [[Realm]] (0 for a native target).
    pub(crate) realm: usize,
    /// Filled in the current call-cache epoch (`bytecode::CALL_IC_EPOCH`).
    pub(crate) current: bool,
}

/// A construct site's view: its constructor and the instances' prototype's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConstructView {
    pub(crate) target: Option<CallTarget>,
    pub(crate) prototype_shape: u32,
}

/// An allocation site's view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AllocationView {
    pub(crate) kind: AllocationKind,
    /// The site's header image (W2e), which generated code copies: made by an execution.
    pub(crate) image: bool,
    /// An object literal's template (W2e), an array literal's shared block (W2d step 4), a RegExp
    /// literal's compiled pattern.
    pub(crate) template: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AllocationKind {
    /// An object literal (MakeObject) or `{}` (NewObject).
    Object,
    /// An array literal (MakeArray, MakeArrayLiteral) or `[]` (NewArray).
    Array,
    RegExp,
    /// A closure (MakeClosure), which needs no per-site state.
    Closure,
}
