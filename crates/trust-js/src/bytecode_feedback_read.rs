//! A chunk's side of the read API (W4 step 6; `crate::feedback::read`, feedback-schema.txt
//! section 5.4): the views of its records and caches by baseline pc.
//!
//! One decode makes a view ([`site_view`]) from the words a [`FeedbackWords`] source holds (P3
//! stage 2, step 16): the live chunk's cells, or the copy of them that the optimizing tier's
//! main thread makes for a compile ([`FeedbackCopy`]), which the compile thread decodes, as
//! SpiderMonkey's WarpOracle copies a Baseline IC's stub data on the main thread "to protect
//! against the stub being unlinked or mutated" (jit/WarpOracle.cpp:1333-1352) and WarpBuilder
//! interprets it off thread. The decode cannot tell the two apart: the copy holds, bitwise, every
//! word the decode reads of the chunk's live state (the value and element records, the property
//! caches' ways, the call and construct caches' entries' words and the call sites' megamorphic
//! marks, the allocation sites' readiness, the call caches' epoch, and a transformed chunk's
//! baseline operations and records through its record map), taken on the main thread with
//! nothing run between the reads, so that the copy is the live chunk's state at one instant;
//! the operations and names it reads are immutable. The tier's debug builds decode a compile's
//! views from both sources and compare them (`opt::snapshot`).
#![cfg_attr(not(test), allow(dead_code))]

use super::*;
use crate::feedback::{
    AllocationKind, AllocationView, CallState, CallTarget, CallView, ConstructView, ElementView,
    NamedView, NamedWay, NamedWayKind, SiteView, ValueClass, ValuesView, ELEM_ARRAY,
    ELEM_KEY_INDEX, ELEM_ORDINARY, ELEM_TYPED_ARRAY, RECORD_OPERAND1_SHIFT, RECORD_RESULT_SHIFT,
};

const NUMBER: u32 = ValueClass::NumberInt32.bit() | ValueClass::NumberDouble.bit();
const INT32: u32 = ValueClass::NumberInt32.bit();
/// The element record's receivers (bits 0-6), keys (8-12) and an Array's facts (16-20).
const ELEMENT_RECEIVERS: u32 = 0x7f;
const ELEMENT_KEYS: u32 = 0x1f << 8;
const ELEMENT_FACTS: u32 = 0x1f << 16;

/// The classes an operation's in-line set holds (operand 0, operand 1, result), which generated
/// code takes without a record (feedback-schema.txt section 5.1); None for an operation without
/// a value record.
pub(crate) fn value_in_line_set(op: &Op) -> Option<[u32; 3]> {
    Some(match op {
        Op::Mod => [INT32, INT32, INT32],
        Op::Lt | Op::Gt | Op::Le | Op::Ge => [NUMBER, NUMBER, 0],
        Op::Neg
        | Op::Plus
        | Op::BitNot
        | Op::UpdateLocal(..)
        | Op::UpdateCap(..)
        | Op::UpdateName(..)
        | Op::UpdateNameCached(..)
        | Op::UpdateConst(..)
        | Op::UpdatePrivate(..)
        | Op::SuperUpdate(..)
        | Op::UpdateProp(..)
        | Op::UpdateElem(..) => [NUMBER, 0, NUMBER],
        Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::UShr
        | Op::BitAnd
        | Op::BitOr
        | Op::BitXor
        | Op::Shl
        | Op::Shr
        | Op::GenBin(..) => [NUMBER, NUMBER, NUMBER],
        _ => return None,
    })
}

/// Whether `op` is an element load or store with the element record.
pub(crate) fn is_element_access(op: &Op) -> bool {
    matches!(
        op,
        Op::GetElem
            | Op::GetElemLocal(..)
            | Op::GetMethodElem
            | Op::SetElem
            | Op::SetElemDrop
            | Op::SetElemLocal(..)
            | Op::SetElemLocalDrop(..)
    )
}

/// The words of a call or construct cache entry the decode reads ([`CallIc`]'s): its callee,
/// function literal, builtin, native entry, realm and fill epoch (P3 stage 2, step 16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CallWords {
    pub(crate) callee: usize,
    pub(crate) func: usize,
    pub(crate) builtin: u16,
    pub(crate) native: usize,
    pub(crate) realm: usize,
    pub(crate) epoch: u32,
}

impl CallWords {
    #[inline]
    fn of(entry: &CallIc) -> Self {
        CallWords {
            callee: entry.callee,
            func: entry.func as usize,
            builtin: entry.builtin,
            native: entry.native,
            realm: entry.realm,
            epoch: entry.epoch,
        }
    }
}

/// A call site's words the decode reads: its ways' entries and its megamorphic mark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CallSiteWords {
    pub(crate) ways: [CallWords; CALL_IC_WAYS],
    pub(crate) megamorphic: bool,
}

impl CallSiteWords {
    #[inline]
    fn of(site: &CallSite) -> Self {
        CallSiteWords {
            ways: std::array::from_fn(|k| CallWords::of(&site.entries[k].get())),
            megamorphic: site.megamorphic(),
        }
    }
}

/// A construct site's words the decode reads: its entry's and its instances' prototype's shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConstructWords {
    pub(crate) call: CallWords,
    pub(crate) prototype_shape: u32,
}

impl ConstructWords {
    #[inline]
    fn of(site: &ConstructSite) -> Self {
        ConstructWords {
            call: CallWords::of(&site.call),
            prototype_shape: site.prototype_shape,
        }
    }
}

/// What the feedback decode ([`site_view`]) reads of a chunk (see the module's doc): its
/// operations, and the words of its live state, each by the index the decode has (a pc, a
/// cache's, a literal site's).
pub(crate) trait FeedbackWords {
    /// The operation at `pc` (None past the end).
    fn site_op(&self, pc: usize) -> Option<&Op>;
    /// The value or element record of the operation at `pc` (`FeedbackVector::record_word`: 0
    /// where none was made, and in a transformed chunk, whose records are its baseline sites').
    fn record_word(&self, pc: usize) -> u32;
    /// The ways of the property cache whose way 0 is `cache` (None past the caches).
    fn prop_ways(&self, cache: u32) -> [Option<IcState>; PROP_IC_WAYS];
    /// Call site `cache`'s words (None past the sites).
    fn call_site(&self, cache: u32) -> Option<CallSiteWords>;
    /// Construct site `cache`'s words (None past the sites).
    fn construct_site(&self, cache: u32) -> Option<ConstructWords>;
    /// Whether object-literal template `template`'s image is made (false past the templates).
    fn object_image_ready(&self, template: u32) -> bool;
    /// Whether the MakeArrays' image is made.
    fn empty_array_image_ready(&self) -> bool;
    /// Whether array-literal site `site`'s image is made (false past the sites).
    fn array_image_ready(&self, site: u32) -> bool;
    /// Whether constant array-literal site `site`'s shared block is made.
    fn array_literal_ready(&self, site: u32) -> bool;
    /// Whether the RegExp literal at `pc` has its compiled pattern.
    fn regexp_literal_ready(&self, pc: usize) -> bool;
    /// The call caches' epoch (`CALL_IC_EPOCH`) a target's `current` compares its fill with.
    fn call_epoch(&self) -> u32;
    /// A transformed chunk's baseline site of `pc` through its record map: the baseline
    /// operation (None past its chunk's end) and record (None where `pc` is unmapped).
    fn baseline_record(&self, pc: usize) -> Option<(Option<&Op>, u32)>;
}

impl FeedbackWords for Chunk {
    #[inline]
    fn site_op(&self, pc: usize) -> Option<&Op> {
        self.ops.get(pc)
    }
    #[inline]
    fn record_word(&self, pc: usize) -> u32 {
        self.feedback.record_word(pc)
    }
    #[inline]
    fn prop_ways(&self, cache: u32) -> [Option<IcState>; PROP_IC_WAYS] {
        std::array::from_fn(|way| {
            self.caches
                .get(cache as usize + way)
                .map(std::cell::Cell::get)
        })
    }
    #[inline]
    fn call_site(&self, cache: u32) -> Option<CallSiteWords> {
        self.call_caches.get(cache as usize).map(CallSiteWords::of)
    }
    #[inline]
    fn construct_site(&self, cache: u32) -> Option<ConstructWords> {
        self.construct_caches
            .get(cache as usize)
            .map(|site| ConstructWords::of(&site.get()))
    }
    #[inline]
    fn object_image_ready(&self, template: u32) -> bool {
        self.obj_images
            .get(template as usize)
            .is_some_and(|image| image.ready.get() != 0)
    }
    #[inline]
    fn empty_array_image_ready(&self) -> bool {
        self.empty_array_image.ready.get() != 0
    }
    #[inline]
    fn array_image_ready(&self, site: u32) -> bool {
        self.array_images
            .get(site as usize)
            .is_some_and(|image| image.ready.get() != 0)
    }
    #[inline]
    fn array_literal_ready(&self, site: u32) -> bool {
        self.array_literals
            .get(site as usize)
            .is_some_and(|literal| literal.block.get().is_some())
    }
    #[inline]
    fn regexp_literal_ready(&self, pc: usize) -> bool {
        self.regexp_literals
            .get(pc)
            .is_some_and(|literal| literal.get().is_some())
    }
    #[inline]
    fn call_epoch(&self) -> u32 {
        CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed)
    }
    #[inline]
    fn baseline_record(&self, pc: usize) -> Option<(Option<&Op>, u32)> {
        let (baseline, baseline_pc) = self.feedback.record_map()?.target(pc)?;
        Some((
            baseline.ops.get(baseline_pc),
            baseline.feedback.record_word(baseline_pc),
        ))
    }
}

impl Chunk {
    /// The feedback of the operation at `pc` (see `crate::feedback::read`).
    pub(crate) fn feedback_view(&self, pc: usize) -> SiteView {
        site_view(self, pc)
    }

    /// The copy of the words of this chunk's feedback that the decode reads, but the property
    /// caches' ways (the optimizing tier's frozen view copies those, once, for its templates and
    /// the decode alike), for a compile off the main thread (P3 stage 2, step 16; see the
    /// module's doc). Main thread only: it reads cells.
    pub(crate) fn copy_feedback(&self) -> FeedbackCopy {
        // A transformed chunk's value and element records are its baseline sites': the
        // baseline operation and record of each of its value and element sites, through the
        // map. Its own record words are never read (`FeedbackVector::record_word`).
        let mut baselines = Vec::new();
        if let Some(map) = self.feedback.record_map() {
            for (pc, op) in self.ops.iter().enumerate() {
                if value_in_line_set(op).is_none() && !is_element_access(op) {
                    continue;
                }
                if let Some((baseline, baseline_pc)) = map.target(pc) {
                    baselines.push((
                        pc as u32,
                        baseline.ops.get(baseline_pc).copied(),
                        baseline.feedback.record_word(baseline_pc),
                    ));
                }
            }
        }
        FeedbackCopy {
            records: self.feedback.own_record_words(),
            baselines: baselines.into_boxed_slice(),
            calls: self.call_caches.iter().map(CallSiteWords::of).collect(),
            constructs: self
                .construct_caches
                .iter()
                .map(|site| ConstructWords::of(&site.get()))
                .collect(),
            object_images: self
                .obj_images
                .iter()
                .map(|image| image.ready.get() != 0)
                .collect(),
            array_images: self
                .array_images
                .iter()
                .map(|image| image.ready.get() != 0)
                .collect(),
            array_literals: self
                .array_literals
                .iter()
                .map(|literal| literal.block.get().is_some())
                .collect(),
            empty_array_image: self.empty_array_image_ready(),
            regexps: self
                .regexp_literals
                .iter()
                .enumerate()
                .filter(|(_, literal)| literal.get().is_some())
                .map(|(pc, _)| pc as u32)
                .collect(),
            epoch: self.call_epoch(),
        }
    }
}

/// The words of a chunk's feedback copied on the main thread for a compile (P3 stage 2, step
/// 16; [`Chunk::copy_feedback`]), each by the index the decode reads it by, but the property
/// caches' ways, which the tier's frozen view holds ([`CopiedWords`] joins them).
#[derive(Debug)]
pub(crate) struct FeedbackCopy {
    /// The chunk's own records by pc (empty for a transformed chunk and one that never recorded).
    records: Box<[u32]>,
    /// A transformed chunk's value and element sites' baseline operations and records, by pc,
    /// ascending (the mapped ones).
    baselines: Box<[(u32, Option<Op>, u32)]>,
    /// The call sites' and construct sites' words, by cache.
    calls: Box<[CallSiteWords]>,
    constructs: Box<[ConstructWords]>,
    /// The literal sites' readiness: object-literal templates' and array-literal sites' images,
    /// constant array literals' blocks, by site; the MakeArrays' image; the pcs of the RegExp
    /// literals with a compiled pattern, ascending.
    object_images: Box<[bool]>,
    array_images: Box<[bool]>,
    array_literals: Box<[bool]>,
    empty_array_image: bool,
    regexps: Box<[u32]>,
    /// The call caches' epoch at the copy.
    epoch: u32,
}

impl FeedbackCopy {
    /// The call caches' epoch the copy read.
    pub(crate) fn epoch(&self) -> u32 {
        self.epoch
    }
}

/// The live chunk's words, the call caches' epoch a copy read in place of the process's now: the
/// other side of the optimizing tier's debug check of a copy (`opt::snapshot::check_decode`),
/// which must compare the two at one epoch, since engines on other threads move the process's
/// epoch meanwhile (P3 stage 2, step 16).
#[cfg(any(test, debug_assertions))]
pub(crate) struct LiveAt<'a> {
    pub(crate) chunk: &'a Chunk,
    pub(crate) epoch: u32,
}

#[cfg(any(test, debug_assertions))]
impl FeedbackWords for LiveAt<'_> {
    fn site_op(&self, pc: usize) -> Option<&Op> {
        self.chunk.site_op(pc)
    }
    fn record_word(&self, pc: usize) -> u32 {
        self.chunk.record_word(pc)
    }
    fn prop_ways(&self, cache: u32) -> [Option<IcState>; PROP_IC_WAYS] {
        self.chunk.prop_ways(cache)
    }
    fn call_site(&self, cache: u32) -> Option<CallSiteWords> {
        self.chunk.call_site(cache)
    }
    fn construct_site(&self, cache: u32) -> Option<ConstructWords> {
        self.chunk.construct_site(cache)
    }
    fn object_image_ready(&self, template: u32) -> bool {
        self.chunk.object_image_ready(template)
    }
    fn empty_array_image_ready(&self) -> bool {
        self.chunk.empty_array_image_ready()
    }
    fn array_image_ready(&self, site: u32) -> bool {
        self.chunk.array_image_ready(site)
    }
    fn array_literal_ready(&self, site: u32) -> bool {
        self.chunk.array_literal_ready(site)
    }
    fn regexp_literal_ready(&self, pc: usize) -> bool {
        self.chunk.regexp_literal_ready(pc)
    }
    fn call_epoch(&self) -> u32 {
        self.epoch
    }
    fn baseline_record(&self, pc: usize) -> Option<(Option<&Op>, u32)> {
        self.chunk.baseline_record(pc)
    }
}

/// A [`FeedbackCopy`] with the operations and the property caches' ways it was copied with (the
/// tier's frozen view's: `ways` by cache, ascending), as the decode reads them.
pub(crate) struct CopiedWords<'a> {
    pub(crate) copy: &'a FeedbackCopy,
    pub(crate) ops: &'a [Op],
    #[allow(clippy::type_complexity)]
    pub(crate) ways: &'a [(u32, Option<IcState>, [IcState; PROP_IC_WAYS])],
}

impl FeedbackWords for CopiedWords<'_> {
    #[inline]
    fn site_op(&self, pc: usize) -> Option<&Op> {
        self.ops.get(pc)
    }
    #[inline]
    fn record_word(&self, pc: usize) -> u32 {
        self.copy.records.get(pc).copied().unwrap_or(0)
    }
    #[inline]
    fn prop_ways(&self, cache: u32) -> [Option<IcState>; PROP_IC_WAYS] {
        match self
            .ways
            .binary_search_by_key(&cache, |&(cache, _, _)| cache)
        {
            Ok(at) => self.ways[at].2.map(Some),
            Err(_) => [None; PROP_IC_WAYS],
        }
    }
    #[inline]
    fn call_site(&self, cache: u32) -> Option<CallSiteWords> {
        self.copy.calls.get(cache as usize).copied()
    }
    #[inline]
    fn construct_site(&self, cache: u32) -> Option<ConstructWords> {
        self.copy.constructs.get(cache as usize).copied()
    }
    #[inline]
    fn object_image_ready(&self, template: u32) -> bool {
        self.copy
            .object_images
            .get(template as usize)
            .copied()
            .unwrap_or(false)
    }
    #[inline]
    fn empty_array_image_ready(&self) -> bool {
        self.copy.empty_array_image
    }
    #[inline]
    fn array_image_ready(&self, site: u32) -> bool {
        self.copy
            .array_images
            .get(site as usize)
            .copied()
            .unwrap_or(false)
    }
    #[inline]
    fn array_literal_ready(&self, site: u32) -> bool {
        self.copy
            .array_literals
            .get(site as usize)
            .copied()
            .unwrap_or(false)
    }
    #[inline]
    fn regexp_literal_ready(&self, pc: usize) -> bool {
        u32::try_from(pc).is_ok_and(|pc| self.copy.regexps.binary_search(&pc).is_ok())
    }
    #[inline]
    fn call_epoch(&self) -> u32 {
        self.copy.epoch
    }
    #[inline]
    fn baseline_record(&self, pc: usize) -> Option<(Option<&Op>, u32)> {
        let pc = u32::try_from(pc).ok()?;
        let at = self
            .copy
            .baselines
            .binary_search_by_key(&pc, |&(pc, _, _)| pc)
            .ok()?;
        let (_, op, word) = &self.copy.baselines[at];
        Some((op.as_ref(), *word))
    }
}

/// The feedback of the operation at `pc` of the chunk whose words `words` holds (see the
/// module's doc): the one decode, of the live chunk and of a copy alike.
pub(crate) fn site_view(words: &impl FeedbackWords, pc: usize) -> SiteView {
    let Some(op) = words.site_op(pc) else {
        return SiteView::None;
    };
    let word = words.record_word(pc);
    if let Some(cache) = named_site_cache(op) {
        let named = named_view(words, cache, writes_property(op));
        return match value_in_line_set(op) {
            Some(in_line) => SiteView::NamedUpdate(named, values_view(word, in_line)),
            None => SiteView::Named(named),
        };
    }
    if let Some(view) = record_view(op, word) {
        return view;
    }
    let allocation = |kind, image: bool, template: bool| {
        SiteView::Allocation(AllocationView {
            kind,
            image,
            template,
        })
    };
    match *op {
        Op::Call(_, cache)
        | Op::CallWithThis(_, cache)
        | Op::CallSpread(_, cache)
        | Op::CallSpreadThis(_, cache) => words.call_site(cache).map_or(SiteView::None, |site| {
            SiteView::Call(call_view(&site, words.call_epoch()))
        }),
        Op::New(_, cache) => words.construct_site(cache).map_or(SiteView::None, |site| {
            SiteView::Construct(ConstructView {
                target: call_target(&site.call, words.call_epoch()),
                prototype_shape: site.prototype_shape,
            })
        }),
        Op::MakeObject(_, _, template) => allocation(
            AllocationKind::Object,
            words.object_image_ready(template),
            template != u32::MAX,
        ),
        Op::NewObject => allocation(AllocationKind::Object, false, false),
        Op::MakeArray(_) => allocation(
            AllocationKind::Array,
            words.empty_array_image_ready(),
            false,
        ),
        Op::MakeArrayLiteral(site) => allocation(
            AllocationKind::Array,
            words.array_image_ready(site),
            words.array_literal_ready(site),
        ),
        Op::NewArray => allocation(AllocationKind::Array, false, false),
        Op::MakeRegExp(..) => allocation(
            AllocationKind::RegExp,
            false,
            words.regexp_literal_ready(pc),
        ),
        Op::MakeClosure(..) => allocation(AllocationKind::Closure, false, false),
        _ => SiteView::None,
    }
}

/// The view of a value site's or an element site's record `word` (`op` no named site); None for
/// another operation.
fn record_view(op: &Op, word: u32) -> Option<SiteView> {
    if let Some(in_line) = value_in_line_set(op) {
        return Some(SiteView::Values(values_view(word, in_line)));
    }
    if is_element_access(op) {
        let recorded_receivers = word & ELEMENT_RECEIVERS;
        let recorded_keys = word & ELEMENT_KEYS;
        return Some(SiteView::Element(ElementView {
            receivers: recorded_receivers | ELEM_ARRAY | ELEM_TYPED_ARRAY | ELEM_ORDINARY,
            keys: recorded_keys | ELEM_KEY_INDEX,
            recorded_receivers,
            recorded_keys,
            facts: word & ELEMENT_FACTS,
            generic: word & crate::feedback::ELEM_GENERIC != 0,
            typed_kinds: word & crate::feedback::ELEM_TYPED_KINDS,
        }));
    }
    None
}

/// The records of a baseline site whose operation is `op` (None past its chunk) and whose record
/// is `word`, as [`site_view`] decodes them: a value or element site's view, or a named update's
/// with its values (its ways empty: a transformed chunk's view keeps its own); None for another
/// operation. What a transformed chunk's view takes of its baseline site (`opt::snapshot`), so
/// that it reads no baseline cache.
pub(crate) fn baseline_records_view(op: Option<&Op>, word: u32) -> SiteView {
    let Some(op) = op else {
        return SiteView::None;
    };
    if named_site_cache(op).is_some() {
        return match value_in_line_set(op) {
            Some(in_line) => SiteView::NamedUpdate(
                NamedView {
                    ways: [None; PROP_IC_WAYS],
                    generic: false,
                    writable: writes_property(op),
                },
                values_view(word, in_line),
            ),
            None => SiteView::None,
        };
    }
    record_view(op, word).unwrap_or(SiteView::None)
}

/// The named view of the property cache whose way 0 is `cache`.
fn named_view(words: &impl FeedbackWords, cache: u32, writable: bool) -> NamedView {
    let ways = words.prop_ways(cache);
    NamedView {
        ways: ways.map(|way| way.and_then(named_way)),
        generic: ways[0].is_some_and(|state| state.mid_ok & IC_MID_GENERIC != 0),
        writable,
    }
}

/// A value record's view: rule U's sets and the record alone.
fn values_view(word: u32, in_line: [u32; 3]) -> ValuesView {
    let recorded = [
        word & 0x1ff,
        (word >> RECORD_OPERAND1_SHIFT) & 0x1ff,
        (word >> RECORD_RESULT_SHIFT) & 0x1ff,
    ];
    ValuesView {
        operand0: recorded[0] | in_line[0],
        operand1: recorded[1] | in_line[1],
        result: recorded[2] | in_line[2],
        recorded,
    }
}

/// One property cache way's view; None for an empty way.
fn named_way(state: IcState) -> Option<NamedWay> {
    let (kind, depth, slot) = match state.depth {
        IC_EMPTY => return None,
        IC_ABSENT => (NamedWayKind::Absent, state.slot as u8, 0),
        IC_CREATE => (NamedWayKind::Creation, 0, 0),
        // The hop count in the low bits, IC_ACCESSOR and IC_ARR_KEYCHK above them.
        depth if depth & IC_ACCESSOR != 0 => (NamedWayKind::Accessor, depth & 0x1f, state.slot),
        depth if depth & IC_ARR_KEYCHK != 0 => (NamedWayKind::KeyChecked, depth & 0x1f, state.slot),
        depth => (NamedWayKind::Data, depth, state.slot),
    };
    Some(NamedWay {
        receiver_shape: state.recv_shape,
        holder_shape: state.holder_shape,
        primitive: (state.mid_ok & IC_MID_PRIMITIVE) >> IC_MID_PRIMITIVE_SHIFT,
        depth,
        slot,
        location: state.loc,
        kind,
    })
}

/// A call site's view, its ways' fills current in call-cache epoch `epoch`.
fn call_view(site: &CallSiteWords, epoch: u32) -> CallView {
    let ways: [Option<CallTarget>; 4] = std::array::from_fn(|k| call_target(&site.ways[k], epoch));
    let filled = ways.iter().flatten().count();
    let state = if site.megamorphic {
        CallState::Megamorphic
    } else {
        match filled {
            0 => CallState::Empty,
            1 => CallState::Monomorphic,
            _ => CallState::Polymorphic,
        }
    };
    CallView { state, ways }
}

/// A call cache entry's target, current where its fill's epoch is `epoch`; None for an empty
/// entry.
fn call_target(entry: &CallWords, epoch: u32) -> Option<CallTarget> {
    (entry.callee != 0).then_some(CallTarget {
        callee: entry.callee,
        function: entry.func,
        builtin: entry.builtin,
        native: entry.native,
        realm: entry.realm,
        current: entry.epoch == epoch,
    })
}
