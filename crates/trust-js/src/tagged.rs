//! Engine-owned tagged execution words.
//!
//! This is the Phase 2 ABI nucleus. It is intentionally not connected to the current `Value`
//! enum or `PackedValue` storage yet: those still carry `Rc` ownership and need the central heap
//! before a heap reference can be resolved safely. Keeping this module independent lets bit-level
//! invariants and invalid-word handling be tested before any execution tier migrates.

#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;

const PAYLOAD_MASK: u64 = 0x0000_ffff_ffff_ffff;
const TAG_MASK: u64 = !PAYLOAD_MASK;
const TAG_CANON_NAN: u64 = 0x7ff8_0000_0000_0000;
const TAG_UNDEFINED: u64 = 0x7ff9_0000_0000_0000;
const TAG_EMPTY: u64 = 0x7ffa_0000_0000_0000;
const TAG_NULL: u64 = 0x7ffb_0000_0000_0000;
const TAG_BOOLEAN: u64 = 0x7ffc_0000_0000_0000;
const TAG_HEAP_REF: u64 = 0xfff9_0000_0000_0000;
const HEAP_REF_INDEX_BITS: u32 = 20;
const HEAP_REF_INDEX_MASK: u32 = (1 << HEAP_REF_INDEX_BITS) - 1;
const HEAP_REF_COOKIE_MASK: u32 = (1 << (32 - HEAP_REF_INDEX_BITS)) - 1;

/// A non-zero 32-bit handle-mode slot/cookie word owned by the active Agent's heap model. Cage
/// offsets use the same opaque type only after an explicit representation selection.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct HeapRef(u32);

impl HeapRef {
    /// Construct an encoded reference. The low 20 bits identify a slot and the high 12 bits are
    /// its generation cookie; slot zero is reserved as an invalid/null reference.
    pub(crate) const fn new(raw: u32) -> Option<Self> {
        if raw == 0 || raw & HEAP_REF_INDEX_MASK == 0 {
            None
        } else {
            Some(Self(raw))
        }
    }

    pub(crate) const fn from_parts(index: u32, cookie: u16) -> Option<Self> {
        if index == 0 || index > HEAP_REF_INDEX_MASK || cookie as u32 > HEAP_REF_COOKIE_MASK {
            return None;
        }
        Some(Self(((cookie as u32) << HEAP_REF_INDEX_BITS) | index))
    }

    pub(crate) const fn get(self) -> u32 {
        self.0
    }

    pub(crate) const fn index(self) -> u32 {
        self.0 & HEAP_REF_INDEX_MASK
    }

    pub(crate) const fn cookie(self) -> u16 {
        (self.0 >> HEAP_REF_INDEX_BITS) as u16
    }
}

/// One engine-owned 64-bit execution value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub(crate) struct TaggedValue(u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum InvalidTaggedValue {
    ReservedTag,
    InvalidBooleanPayload,
    NullHeapReference,
    WideHeapReference,
    NonCanonicalNan,
}

impl TaggedValue {
    pub(crate) const fn undefined() -> Self {
        Self(TAG_UNDEFINED)
    }

    pub(crate) const fn empty() -> Self {
        Self(TAG_EMPTY)
    }

    pub(crate) const fn null() -> Self {
        Self(TAG_NULL)
    }

    pub(crate) const fn boolean(value: bool) -> Self {
        Self(TAG_BOOLEAN | value as u64)
    }

    /// Preserve all non-NaN IEEE-754 bits, including `-0` and infinities. NaN payloads are
    /// canonicalized because arbitrary quiet-NaN payloads overlap the reserved tag range.
    pub(crate) fn number(value: f64) -> Self {
        Self(if value.is_nan() {
            TAG_CANON_NAN
        } else {
            value.to_bits()
        })
    }

    /// Construct a heap reference without embedding a native pointer. The upper payload bits stay
    /// zero so a future cage offset and the handle-table fallback share one word format.
    pub(crate) fn heap(reference: HeapRef) -> Self {
        Self(TAG_HEAP_REF | reference.get() as u64)
    }

    pub(crate) const fn raw(self) -> u64 {
        self.0
    }

    /// Construct an unchecked word for verifier and corruption tests. Production constructors
    /// remain typed; callers must invoke `validate` before treating this as a value.
    pub(crate) const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub(crate) fn validate(self) -> Result<(), InvalidTaggedValue> {
        let tag = self.0 & TAG_MASK;
        match tag {
            TAG_UNDEFINED | TAG_EMPTY | TAG_NULL => {
                if self.0 & PAYLOAD_MASK == 0 {
                    Ok(())
                } else {
                    Err(InvalidTaggedValue::ReservedTag)
                }
            }
            TAG_BOOLEAN => {
                if self.0 & PAYLOAD_MASK <= 1 {
                    Ok(())
                } else {
                    Err(InvalidTaggedValue::InvalidBooleanPayload)
                }
            }
            TAG_HEAP_REF => {
                if self.0 & 0x0000_ffff_0000_0000 != 0 {
                    return Err(InvalidTaggedValue::WideHeapReference);
                }
                if self.0 & 0xffff_ffff == 0 {
                    Err(InvalidTaggedValue::NullHeapReference)
                } else {
                    Ok(())
                }
            }
            TAG_CANON_NAN => {
                if self.0 == TAG_CANON_NAN {
                    Ok(())
                } else {
                    Err(InvalidTaggedValue::NonCanonicalNan)
                }
            }
            _ if f64::from_bits(self.0).is_nan() => Err(InvalidTaggedValue::NonCanonicalNan),
            // Every non-NaN IEEE-754 bit pattern is a Number, including infinities and
            // subnormals whose exponent bits do not match one of the reserved tagged prefixes.
            _ => Ok(()),
        }
    }

    pub(crate) fn as_boolean(self) -> Option<bool> {
        (self.0 & TAG_MASK == TAG_BOOLEAN)
            .then_some(self.0 & PAYLOAD_MASK)
            .and_then(|payload| match payload {
                0 => Some(false),
                1 => Some(true),
                _ => None,
            })
    }

    pub(crate) fn as_number(self) -> Option<f64> {
        (self.0 & TAG_MASK != TAG_UNDEFINED)
            .then_some(f64::from_bits(self.0))
            .filter(|value| !value.is_nan() || self.0 == TAG_CANON_NAN)
    }

    pub(crate) fn as_heap(self) -> Option<HeapRef> {
        if self.0 & TAG_MASK != TAG_HEAP_REF || self.0 & 0xffff_ffff == 0 {
            return None;
        }
        if self.0 & 0x0000_ffff_0000_0000 != 0 {
            return None;
        }
        HeapRef::new(self.0 as u32)
    }
}

/// Canonical bytecode identity used by feedback, safepoints, and deoptimization records. Native
/// instruction addresses are deliberately not part of this identity.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct BytecodePc(u32);

impl BytecodePc {
    pub(crate) const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub(crate) const fn get(self) -> u32 {
        self.0
    }
}

/// Stable index of one published safepoint or deoptimization record.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct SafepointId(u32);

impl SafepointId {
    pub(crate) const fn new(raw: u32) -> Self {
        Self(raw)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DeoptId(u32);

impl DeoptId {
    pub(crate) const fn new(raw: u32) -> Self {
        Self(raw)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FrameMapError {
    SlotBitmapTooLong,
    OverlappingRootClasses,
    SlotOutOfRange,
    OperandDepthOutOfRange,
    InvalidTaggedWord,
    FrameLengthMismatch,
}

/// Immutable root metadata for one exact execution point. Bitmaps address logical frame slots;
/// the generated tier may keep those slots in registers or spills as long as it publishes this
/// same logical map before a safepoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RootMap {
    pub(crate) safepoint: SafepointId,
    pub(crate) bytecode_pc: BytecodePc,
    pub(crate) slot_count: u16,
    pub(crate) operand_depth: u16,
    pub(crate) tagged_registers: u64,
    pub(crate) tagged_slots: Vec<u64>,
    pub(crate) handle_slots: Vec<u64>,
    pub(crate) environment_slots: Vec<u64>,
}

impl RootMap {
    fn bitmap_words(slot_count: u16) -> usize {
        usize::from(slot_count).div_ceil(64)
    }

    fn check_bitmap(bits: &[u64], slot_count: u16) -> Result<(), FrameMapError> {
        let expected = Self::bitmap_words(slot_count);
        if bits.len() != expected {
            return Err(FrameMapError::SlotBitmapTooLong);
        }
        let excess = expected
            .checked_mul(64)
            .and_then(|count| count.checked_sub(usize::from(slot_count)))
            .unwrap_or(0);
        if excess != 0 && bits.last().is_some_and(|word| word >> (64 - excess) != 0) {
            return Err(FrameMapError::SlotOutOfRange);
        }
        Ok(())
    }

    pub(crate) fn validate(&self) -> Result<(), FrameMapError> {
        if self.operand_depth > self.slot_count {
            return Err(FrameMapError::OperandDepthOutOfRange);
        }
        Self::check_bitmap(&self.tagged_slots, self.slot_count)?;
        Self::check_bitmap(&self.handle_slots, self.slot_count)?;
        Self::check_bitmap(&self.environment_slots, self.slot_count)?;
        for ((tagged, handles), environments) in self
            .tagged_slots
            .iter()
            .zip(&self.handle_slots)
            .zip(&self.environment_slots)
        {
            if tagged & handles != 0 || tagged & environments != 0 || handles & environments != 0 {
                return Err(FrameMapError::OverlappingRootClasses);
            }
        }
        Ok(())
    }

    /// Visit the union of all root classes in ascending logical-slot order. Callers validate the
    /// map first; the bitmap walk then skips dead frame slots without touching their payloads.
    fn try_for_each_root_slot<E>(
        &self,
        mut visit: impl FnMut(usize) -> Result<(), E>,
    ) -> Result<(), E> {
        for word in 0..self.tagged_slots.len() {
            let mut bits =
                self.tagged_slots[word] | self.handle_slots[word] | self.environment_slots[word];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                visit(word * 64 + bit)?;
                bits &= bits - 1;
            }
        }
        Ok(())
    }
}

/// A temporary tagged shadow frame used by the ABI verifier and migration tests. It has no
/// connection to the live `Value` frames until a complete relocation implementation exists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct TaggedFrame {
    slots: Vec<TaggedValue>,
}

impl TaggedFrame {
    pub(crate) fn new(slots: Vec<TaggedValue>) -> Self {
        Self { slots }
    }

    fn value_at(&self, slot: u16) -> Result<TaggedValue, FrameMapError> {
        let value = self
            .slots
            .get(usize::from(slot))
            .copied()
            .ok_or(FrameMapError::SlotOutOfRange)?;
        value
            .validate()
            .map_err(|_| FrameMapError::InvalidTaggedWord)?;
        Ok(value)
    }

    pub(crate) fn roots(&self, map: &RootMap) -> Result<Vec<TaggedValue>, FrameMapError> {
        map.validate()?;
        if self.slots.len() != usize::from(map.slot_count) {
            return Err(FrameMapError::FrameLengthMismatch);
        }
        let mut roots = Vec::new();
        map.try_for_each_root_slot(|slot| {
            let value = self
                .slots
                .get(slot)
                .copied()
                .ok_or(FrameMapError::SlotOutOfRange)?;
            value
                .validate()
                .map_err(|_| FrameMapError::InvalidTaggedWord)?;
            roots.push(value);
            Ok(())
        })?;
        Ok(roots)
    }

    /// Rewrite one forwarding edge in the slots named by an exact safepoint map. Dead slots are
    /// deliberately ignored: a poisoned non-root slot must never become an accidental strong edge.
    pub(crate) fn rewrite_heap_reference(
        &mut self,
        map: &RootMap,
        from: HeapRef,
        to: HeapRef,
    ) -> Result<usize, FrameMapError> {
        map.validate()?;
        if self.slots.len() != usize::from(map.slot_count) {
            return Err(FrameMapError::FrameLengthMismatch);
        }
        let replacement = TaggedValue::heap(to);
        let mut rewritten = 0;
        map.try_for_each_root_slot(|slot| {
            let value = self.value_at(u16::try_from(slot).unwrap_or(u16::MAX))?;
            if value.as_heap() == Some(from) {
                self.slots[slot] = replacement;
                rewritten += 1;
            }
            Ok(())
        })?;
        Ok(rewritten)
    }
}

/// A fixed, allocation-free tagged frame for the first execution migration slice.
///
/// The frame contains only Number operands, so it cannot cross a user-code, allocation, or
/// safepoint boundary. Any other ECMAScript value must remain on the canonical `Value` path,
/// where `+` can perform ToPrimitive/string concatenation and the other operators can perform
/// their complete ToNumeric algorithms. Keeping this frame on the stack makes the migration
/// auditable while avoiding a per-operation heap allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TaggedNumericFrame {
    slots: [TaggedValue; 2],
}

impl TaggedNumericFrame {
    #[inline(always)]
    pub(crate) fn new(left: f64, right: f64) -> Self {
        Self {
            slots: [TaggedValue::number(left), TaggedValue::number(right)],
        }
    }

    #[inline(always)]
    pub(crate) fn binary<F>(self, f: F) -> TaggedValue
    where
        F: FnOnce(f64, f64) -> f64,
    {
        // `new` constructs both slots through the typed Number constructor. Avoid re-validating
        // the tag on this no-safepoint path; the frame is private to the helper and cannot be
        // mutated between construction and this call.
        let left = f64::from_bits(self.slots[0].raw());
        let right = f64::from_bits(self.slots[1].raw());
        TaggedValue::number(f(left, right))
    }

    /// Apply an f64 predicate over the two slots and produce a Boolean immediate. The predicate
    /// is the exact float comparison the canonical `Value` path would run, so NaN (always false
    /// for the relational and equality operators), `-0 == +0`, infinities, and subnormals behave
    /// identically on this no-safepoint frame. Equality ops are routed here exactly like
    /// `Op::EqEq`/`Op::StrictEq` route through `bin_cmp`, whose f64 closure is the specified
    /// Number comparison after both operands are already primitive Numbers.
    #[inline(always)]
    pub(crate) fn cmp<F>(self, f: F) -> TaggedValue
    where
        F: FnOnce(f64, f64) -> bool,
    {
        let left = f64::from_bits(self.slots[0].raw());
        let right = f64::from_bits(self.slots[1].raw());
        TaggedValue::boolean(f(left, right))
    }

    /// Apply an f64 transcendental over the two slots and produce a Number immediate. For
    /// exponentiation the caller-level predicate is the complete Number::exponentiate distinction:
    /// an f64 `powf` alone would disagree with ECMA-262 §6.1.6.1.20 on `1 ** ±Infinity` (NaN, not
    /// 1.0) and on a NaN exponent with base 1, so this mirrors the exact guard the canonical
    /// `Interp::binary` fast path already applies before delegating to `powf`.
    #[inline(always)]
    pub(crate) fn pow(self) -> TaggedValue {
        let left = f64::from_bits(self.slots[0].raw());
        let right = f64::from_bits(self.slots[1].raw());
        TaggedValue::number(
            if right.is_nan() || (left.abs() == 1.0 && right.is_infinite()) {
                f64::NAN
            } else {
                left.powf(right)
            },
        )
    }

    /// Apply an i32 transform over the two slots and produce a Number immediate. The int32
    /// conversion (ECMA-262 §7.1.5) runs on the extracted bits exactly like `bin_i32`, so NaN,
    /// infinities, fractions, and values outside the int32 range wrap identically; the f64 result
    /// widening is exact for every int32.
    #[inline(always)]
    pub(crate) fn bitwise<F>(self, f: F) -> TaggedValue
    where
        F: FnOnce(i32, i32) -> i32,
    {
        let left = f64::from_bits(self.slots[0].raw());
        let right = f64::from_bits(self.slots[1].raw());
        TaggedValue::number(f(crate::eval::to_int32(left), crate::eval::to_int32(right)) as f64)
    }
}

/// A fixed, allocation-free one-slot tagged frame for the unary execution slice. It models a
/// single Number operand slot, the same shape a migrated local or VM operand will have; it cannot
/// cross a user-code, allocation, or safepoint boundary. Any non-Number value must remain on the
/// canonical `Value` path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TaggedNumberFrame {
    slot: TaggedValue,
}

impl TaggedNumberFrame {
    #[inline(always)]
    pub(crate) fn new(value: f64) -> Self {
        Self {
            slot: TaggedValue::number(value),
        }
    }

    /// Apply one f64 unary transform and produce a Number immediate. The canonical paths route
    /// here with the exact closure they would run on the `Value` fast path: negation (which flips
    /// signed zero), `+` identity, and bitwise NOT (which converts through ToInt32 first, then
    /// widens the i32 result exactly).
    #[inline(always)]
    pub(crate) fn unary<F>(self, f: F) -> TaggedValue
    where
        F: FnOnce(f64) -> f64,
    {
        TaggedValue::number(f(f64::from_bits(self.slot.raw())))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum MaterializationRecipe {
    CopySlot(u16),
    Constant(TaggedValue),
    BoxNumber(u16),
    Handle(u16),
    Duplicate(usize),
    VirtualObject { layout: u32, fields: Vec<usize> },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeoptRecordError {
    InvalidRootMap(FrameMapError),
    ValueCountOverflow,
    RecipeDependencyOutOfOrder,
    RecipeSlotOutOfRange,
    InvalidConstant,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum MaterializedValue {
    Word(TaggedValue),
    /// A virtual object remains an explicit recipe until a central heap can allocate its layout.
    /// Fields point at earlier entries in the same materialization vector, never native memory.
    VirtualObject {
        layout: u32,
        fields: Vec<usize>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeoptMaterializationError {
    InvalidRecord(DeoptRecordError),
    InvalidFrame(FrameMapError),
    SlotNotNumber(u16),
    SlotNotHandle(u16),
    MissingDependency(usize),
}

/// Deoptimization metadata is validated before publication. Recipes refer only to prior recipes
/// or logical frame slots, which makes reconstruction deterministic and prevents hidden native
/// pointers from entering a resumed frame.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeoptRecord {
    pub(crate) id: DeoptId,
    pub(crate) bytecode_pc: BytecodePc,
    pub(crate) root_map: RootMap,
    pub(crate) recipes: Vec<MaterializationRecipe>,
}

impl DeoptRecord {
    pub(crate) fn validate(&self) -> Result<(), DeoptRecordError> {
        self.root_map
            .validate()
            .map_err(DeoptRecordError::InvalidRootMap)?;
        if self.recipes.len() > usize::from(u16::MAX) {
            return Err(DeoptRecordError::ValueCountOverflow);
        }
        let slots = self.root_map.slot_count;
        for (index, recipe) in self.recipes.iter().enumerate() {
            match recipe {
                MaterializationRecipe::CopySlot(slot)
                | MaterializationRecipe::BoxNumber(slot)
                | MaterializationRecipe::Handle(slot)
                    if *slot >= slots =>
                {
                    return Err(DeoptRecordError::RecipeSlotOutOfRange);
                }
                MaterializationRecipe::Constant(value) => value
                    .validate()
                    .map_err(|_| DeoptRecordError::InvalidConstant)?,
                MaterializationRecipe::Duplicate(source) if *source >= index => {
                    return Err(DeoptRecordError::RecipeDependencyOutOfOrder);
                }
                MaterializationRecipe::VirtualObject { fields, .. }
                    if fields.iter().any(|source| *source >= index) =>
                {
                    return Err(DeoptRecordError::RecipeDependencyOutOfOrder);
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Reconstruct the logical values for a toy deoptimization handoff. This intentionally does
    /// not allocate a heap object for `VirtualObject`; the central heap will own that operation
    /// once a migrated object family exists. Every recipe is nevertheless checked and resolved in
    /// source order, so a future native tier cannot smuggle an unvalidated pointer into a frame.
    pub(crate) fn materialize(
        &self,
        frame: &TaggedFrame,
    ) -> Result<Vec<MaterializedValue>, DeoptMaterializationError> {
        self.validate()
            .map_err(DeoptMaterializationError::InvalidRecord)?;
        if frame.slots.len() != usize::from(self.root_map.slot_count) {
            return Err(DeoptMaterializationError::InvalidFrame(
                FrameMapError::FrameLengthMismatch,
            ));
        }

        let mut values = Vec::with_capacity(self.recipes.len());
        for recipe in &self.recipes {
            let value = match recipe {
                MaterializationRecipe::CopySlot(slot) => MaterializedValue::Word(
                    frame
                        .value_at(*slot)
                        .map_err(DeoptMaterializationError::InvalidFrame)?,
                ),
                MaterializationRecipe::Constant(value) => MaterializedValue::Word(*value),
                MaterializationRecipe::BoxNumber(slot) => {
                    let value = frame
                        .value_at(*slot)
                        .map_err(DeoptMaterializationError::InvalidFrame)?;
                    if value.as_number().is_none() {
                        return Err(DeoptMaterializationError::SlotNotNumber(*slot));
                    }
                    // Numbers are immediate in the canonical word format. A future heap-backed
                    // representation may replace this with a boxed Number allocation.
                    MaterializedValue::Word(value)
                }
                MaterializationRecipe::Handle(slot) => {
                    let value = frame
                        .value_at(*slot)
                        .map_err(DeoptMaterializationError::InvalidFrame)?;
                    if value.as_heap().is_none() {
                        return Err(DeoptMaterializationError::SlotNotHandle(*slot));
                    }
                    MaterializedValue::Word(value)
                }
                MaterializationRecipe::Duplicate(source) => values
                    .get(*source)
                    .cloned()
                    .ok_or(DeoptMaterializationError::MissingDependency(*source))?,
                MaterializationRecipe::VirtualObject { layout, fields } => {
                    if let Some(source) = fields.iter().find(|source| **source >= values.len()) {
                        return Err(DeoptMaterializationError::MissingDependency(*source));
                    }
                    MaterializedValue::VirtualObject {
                        layout: *layout,
                        fields: fields.clone(),
                    }
                }
            };
            values.push(value);
        }
        Ok(values)
    }
}

#[derive(Default)]
struct RootState {
    slots: Vec<Option<TaggedValue>>,
    free: Vec<usize>,
}

/// Agent-local strong roots for tagged migration tests. The registry is intentionally independent
/// of the current `Interp`: until the central heap exists, a root can validate a word but cannot
/// resolve or relocate its heap reference.
#[derive(Clone, Default)]
pub(crate) struct RootSet {
    state: Rc<RefCell<RootState>>,
}

/// Lexical strong handle. Dropping it releases its slot and makes that slot available for reuse;
/// no raw pointer or untyped frame address crosses the guard boundary.
pub(crate) struct RootedTagged {
    state: Rc<RefCell<RootState>>,
    index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NoGcError {
    AlreadyActive,
    SafepointForbidden,
}

/// Agent-local state for short raw-pointer regions. The scope is deliberately non-nestable:
/// requiring one lexical owner makes entry/exit ordering auditable and avoids a counter that could
/// be corrupted by dropping nested guards out of order.
#[derive(Default)]
pub(crate) struct NoGcState {
    active: Cell<bool>,
}

/// A lexical proof that the owning Agent is in a short no-safepoint region. The lifetime borrows
/// the owner, while the marker keeps the guard thread-local even if the owner is later changed to
/// use a synchronizable representation.
pub(crate) struct NoGcScope<'a> {
    state: &'a NoGcState,
    _not_send: PhantomData<Rc<()>>,
}

impl NoGcState {
    pub(crate) fn enter(&self) -> Result<NoGcScope<'_>, NoGcError> {
        if self.active.replace(true) {
            return Err(NoGcError::AlreadyActive);
        }
        Ok(NoGcScope {
            state: self,
            _not_send: PhantomData,
        })
    }

    pub(crate) fn is_active(&self) -> bool {
        self.active.get()
    }

    /// A collector, allocation slow path, host call, interrupt poll, or suspension must check
    /// this boundary before becoming a safepoint. Raw-pointer code cannot silently cross it.
    pub(crate) fn require_safepoint(&self) -> Result<(), NoGcError> {
        if self.is_active() {
            Err(NoGcError::SafepointForbidden)
        } else {
            Ok(())
        }
    }

    #[cfg(test)]
    pub(crate) fn set_active_for_test(&self, active: bool) {
        self.active.set(active);
    }
}

impl NoGcScope<'_> {
    pub(crate) fn is_active(&self) -> bool {
        self.state.is_active()
    }

    /// Debug-only assertion for operations whose signatures already require a `NoGcScope`.
    pub(crate) fn assert_active(&self) {
        debug_assert!(self.is_active());
    }
}

impl Drop for NoGcScope<'_> {
    fn drop(&mut self) {
        debug_assert!(self.state.is_active());
        self.state.active.set(false);
    }
}

impl RootSet {
    pub(crate) fn new() -> Self {
        Self {
            state: Rc::new(RefCell::new(RootState {
                slots: Vec::new(),
                free: Vec::new(),
            })),
        }
    }

    pub(crate) fn root(&self, value: TaggedValue) -> Result<RootedTagged, InvalidTaggedValue> {
        value.validate()?;
        let mut state = self.state.borrow_mut();
        let index = state.free.pop().unwrap_or_else(|| {
            state.slots.push(None);
            state.slots.len() - 1
        });
        state.slots[index] = Some(value);
        Ok(RootedTagged {
            state: Rc::clone(&self.state),
            index,
        })
    }

    pub(crate) fn active_len(&self) -> usize {
        self.state
            .borrow()
            .slots
            .iter()
            .filter(|slot| slot.is_some())
            .count()
    }

    /// Snapshot roots in stable slot order for a collector or verifier. Every slot is revalidated
    /// so corruption in a future bridge fails at the root boundary rather than becoming a hidden
    /// liveness error.
    pub(crate) fn snapshot(&self) -> Result<Vec<TaggedValue>, InvalidTaggedValue> {
        self.state
            .borrow()
            .slots
            .iter()
            .flatten()
            .copied()
            .map(|value| value.validate().map(|()| value))
            .collect()
    }

    /// Rewrite a relocated handle in every live root. The central heap performs the analogous
    /// walk over tagged object fields before its forwarding source is reclaimed.
    pub(crate) fn rewrite_heap_reference(&self, from: HeapRef, to: HeapRef) -> usize {
        let replacement = TaggedValue::heap(to);
        let mut state = self.state.borrow_mut();
        let mut rewritten = 0;
        for slot in state.slots.iter_mut().flatten() {
            if slot.as_heap() == Some(from) {
                *slot = replacement;
                rewritten += 1;
            }
        }
        rewritten
    }

    /// Rewrite all relocated handles in one root-table walk. A nursery collector can move many
    /// objects at one safepoint; batching avoids rescanning every root slot for each forwarding
    /// edge while preserving the same fail-closed tagged-word representation.
    pub(crate) fn rewrite_heap_references(
        &self,
        forwarding: &crate::fasthash::FastMap<HeapRef, HeapRef>,
    ) -> usize {
        let mut state = self.state.borrow_mut();
        let mut rewritten = 0;
        for slot in state.slots.iter_mut().flatten() {
            let Some(source) = slot.as_heap() else {
                continue;
            };
            let Some(target) = forwarding.get(&source) else {
                continue;
            };
            *slot = TaggedValue::heap(*target);
            rewritten += 1;
        }
        rewritten
    }
}

impl RootedTagged {
    pub(crate) fn value(&self) -> Option<TaggedValue> {
        self.state.borrow().slots.get(self.index).copied().flatten()
    }

    pub(crate) fn replace(&mut self, value: TaggedValue) -> Result<(), InvalidTaggedValue> {
        value.validate()?;
        let mut state = self.state.borrow_mut();
        let Some(slot) = state.slots.get_mut(self.index) else {
            return Err(InvalidTaggedValue::ReservedTag);
        };
        if slot.is_none() {
            return Err(InvalidTaggedValue::ReservedTag);
        }
        *slot = Some(value);
        Ok(())
    }
}

impl Drop for RootedTagged {
    fn drop(&mut self) {
        let mut state = self.state.borrow_mut();
        if let Some(slot) = state.slots.get_mut(self.index) {
            if slot.take().is_some() {
                state.free.push(self.index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tagged_word_is_one_word_and_preserves_immediates() {
        assert_eq!(std::mem::size_of::<TaggedValue>(), 8);
        assert_eq!(TaggedValue::undefined().validate(), Ok(()));
        assert_eq!(TaggedValue::empty().validate(), Ok(()));
        assert_eq!(TaggedValue::null().validate(), Ok(()));
        assert_eq!(TaggedValue::boolean(false).as_boolean(), Some(false));
        assert_eq!(TaggedValue::boolean(true).as_boolean(), Some(true));
    }

    #[test]
    fn numeric_shadow_frame_preserves_ieee_number_bits() {
        let frame = TaggedNumericFrame::new(-0.0, f64::INFINITY);
        let result = frame.binary(|left, right| left + right);
        assert_eq!(result.as_number(), Some(f64::INFINITY));

        let frame = TaggedNumericFrame::new(-0.0, 0.0);
        let result = frame.binary(|left, right| left + right);
        assert_eq!(result.as_number().unwrap().to_bits(), 0.0f64.to_bits());

        let frame = TaggedNumericFrame::new(f64::NAN, 1.0);
        let result = frame.binary(|left, right| left + right);
        assert!(result.as_number().is_some_and(f64::is_nan));
        assert_eq!(result.raw(), TAG_CANON_NAN);
    }

    #[test]
    fn numbers_preserve_signed_zero_and_infinities_and_canonicalize_nan() {
        for value in [0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, 42.5] {
            let tagged = TaggedValue::number(value);
            assert_eq!(tagged.validate(), Ok(()));
            assert_eq!(tagged.as_number().unwrap().to_bits(), value.to_bits());
        }
        let nan = TaggedValue::number(f64::from_bits(0x7ff8_0000_0000_0001));
        assert_eq!(nan.raw(), TAG_CANON_NAN);
        assert_eq!(nan.validate(), Ok(()));
        assert!(nan.as_number().unwrap().is_nan());
    }

    #[test]
    fn heap_refs_are_nonzero_32_bit_payloads() {
        assert_eq!(HeapRef::new(0), None);
        let reference = HeapRef::new(u32::MAX).unwrap();
        let tagged = TaggedValue::heap(reference);
        assert_eq!(tagged.validate(), Ok(()));
        assert_eq!(tagged.as_heap(), Some(reference));
    }

    #[test]
    fn heap_refs_round_trip_slot_and_generation_cookie() {
        let reference = HeapRef::from_parts(37, 0xabc).unwrap();
        assert_eq!(reference.index(), 37);
        assert_eq!(reference.cookie(), 0xabc);
        assert_eq!(HeapRef::new(reference.get()), Some(reference));
        assert_eq!(HeapRef::from_parts(0, 0), None);
        assert_eq!(HeapRef::from_parts(1 << 20, 0), None);
        assert_eq!(HeapRef::from_parts(1, 0x1000), None);
        assert_eq!(HeapRef::new(0xfff0_0000), None);
    }

    #[test]
    fn invalid_tagged_words_are_rejected() {
        assert_eq!(
            TaggedValue(TAG_BOOLEAN | 2).validate(),
            Err(InvalidTaggedValue::InvalidBooleanPayload)
        );
        assert_eq!(
            TaggedValue(TAG_HEAP_REF).validate(),
            Err(InvalidTaggedValue::NullHeapReference)
        );
        assert_eq!(
            TaggedValue(TAG_HEAP_REF | 0x1_0000_0000).validate(),
            Err(InvalidTaggedValue::WideHeapReference)
        );
        assert_eq!(
            TaggedValue(TAG_CANON_NAN | 1).validate(),
            Err(InvalidTaggedValue::NonCanonicalNan)
        );
    }

    #[test]
    fn root_map_extracts_only_declared_and_validated_slots() {
        let map = RootMap {
            safepoint: SafepointId::new(3),
            bytecode_pc: BytecodePc::new(17),
            slot_count: 4,
            operand_depth: 2,
            tagged_registers: 0b101,
            tagged_slots: vec![0b0001],
            handle_slots: vec![0b0010],
            environment_slots: vec![0b1000],
        };
        let frame = TaggedFrame::new(vec![
            TaggedValue::heap(HeapRef::new(1).unwrap()),
            TaggedValue::number(42.0),
            TaggedValue::boolean(true),
            TaggedValue::null(),
        ]);
        assert_eq!(
            frame.roots(&map).unwrap(),
            vec![
                TaggedValue::heap(HeapRef::new(1).unwrap()),
                TaggedValue::number(42.0),
                TaggedValue::null(),
            ]
        );
    }

    #[test]
    fn root_map_rejects_bad_bitmap_and_overlapping_classes() {
        let mut map = RootMap {
            safepoint: SafepointId::new(0),
            bytecode_pc: BytecodePc::new(0),
            slot_count: 1,
            operand_depth: 0,
            tagged_registers: 0,
            tagged_slots: vec![0b11],
            handle_slots: vec![0],
            environment_slots: vec![0],
        };
        assert_eq!(map.validate(), Err(FrameMapError::SlotOutOfRange));
        map.tagged_slots[0] = 1;
        map.handle_slots[0] = 1;
        assert_eq!(map.validate(), Err(FrameMapError::OverlappingRootClasses));
    }

    #[test]
    fn frame_root_map_rewrites_only_live_slots_at_a_safepoint() {
        let old = HeapRef::from_parts(7, 2).unwrap();
        let new = HeapRef::from_parts(7, 3).unwrap();
        let map = RootMap {
            safepoint: SafepointId::new(5),
            bytecode_pc: BytecodePc::new(19),
            slot_count: 3,
            operand_depth: 2,
            tagged_registers: 0,
            tagged_slots: vec![0b001],
            handle_slots: vec![0b010],
            environment_slots: vec![0],
        };
        let mut frame = TaggedFrame::new(vec![
            TaggedValue::heap(old),
            TaggedValue::number(3.0),
            // This is deliberately not named by the map; stress builds may poison it.
            TaggedValue::from_raw(TAG_BOOLEAN | 2),
        ]);
        assert_eq!(frame.rewrite_heap_reference(&map, old, new), Ok(1));
        assert_eq!(frame.slots[0].as_heap(), Some(new));
        assert_eq!(frame.slots[1].as_number(), Some(3.0));
        assert_eq!(frame.slots[2], TaggedValue::from_raw(TAG_BOOLEAN | 2));
        assert_eq!(frame.roots(&map).unwrap()[0].as_heap(), Some(new));
    }

    #[test]
    fn deopt_recipes_are_forward_only_and_validate_constants() {
        let root_map = RootMap {
            safepoint: SafepointId::new(1),
            bytecode_pc: BytecodePc::new(8),
            slot_count: 2,
            operand_depth: 1,
            tagged_registers: 0,
            tagged_slots: vec![0b01],
            handle_slots: vec![0b10],
            environment_slots: vec![0],
        };
        let valid = DeoptRecord {
            id: DeoptId::new(4),
            bytecode_pc: BytecodePc::new(8),
            root_map: root_map.clone(),
            recipes: vec![
                MaterializationRecipe::Constant(TaggedValue::number(-0.0)),
                MaterializationRecipe::Duplicate(0),
                MaterializationRecipe::VirtualObject {
                    layout: 7,
                    fields: vec![0, 1],
                },
            ],
        };
        assert_eq!(valid.validate(), Ok(()));

        let mut bad = valid.clone();
        bad.recipes[1] = MaterializationRecipe::Duplicate(2);
        assert_eq!(
            bad.validate(),
            Err(DeoptRecordError::RecipeDependencyOutOfOrder)
        );

        let mut bad_constant = valid;
        bad_constant.recipes[0] = MaterializationRecipe::Constant(TaggedValue(TAG_BOOLEAN | 2));
        assert_eq!(
            bad_constant.validate(),
            Err(DeoptRecordError::InvalidConstant)
        );
    }

    #[test]
    fn deopt_materialization_reconstructs_words_and_virtual_objects() {
        let handle = HeapRef::from_parts(9, 4).unwrap();
        let record = DeoptRecord {
            id: DeoptId::new(8),
            bytecode_pc: BytecodePc::new(23),
            root_map: RootMap {
                safepoint: SafepointId::new(2),
                bytecode_pc: BytecodePc::new(23),
                slot_count: 2,
                operand_depth: 1,
                tagged_registers: 0,
                tagged_slots: vec![0b01],
                handle_slots: vec![0b10],
                environment_slots: vec![0],
            },
            recipes: vec![
                MaterializationRecipe::CopySlot(0),
                MaterializationRecipe::BoxNumber(0),
                MaterializationRecipe::Handle(1),
                MaterializationRecipe::Duplicate(1),
                MaterializationRecipe::VirtualObject {
                    layout: 12,
                    fields: vec![0, 2, 3],
                },
            ],
        };
        let frame = TaggedFrame::new(vec![TaggedValue::number(-0.0), TaggedValue::heap(handle)]);
        assert_eq!(
            record.materialize(&frame).unwrap(),
            vec![
                MaterializedValue::Word(TaggedValue::number(-0.0)),
                MaterializedValue::Word(TaggedValue::number(-0.0)),
                MaterializedValue::Word(TaggedValue::heap(handle)),
                MaterializedValue::Word(TaggedValue::number(-0.0)),
                MaterializedValue::VirtualObject {
                    layout: 12,
                    fields: vec![0, 2, 3],
                },
            ]
        );
    }

    #[test]
    fn deopt_materialization_rejects_wrong_recipe_slot_kind() {
        let record = DeoptRecord {
            id: DeoptId::new(9),
            bytecode_pc: BytecodePc::new(24),
            root_map: RootMap {
                safepoint: SafepointId::new(3),
                bytecode_pc: BytecodePc::new(24),
                slot_count: 1,
                operand_depth: 0,
                tagged_registers: 0,
                tagged_slots: vec![1],
                handle_slots: vec![0],
                environment_slots: vec![0],
            },
            recipes: vec![MaterializationRecipe::Handle(0)],
        };
        let frame = TaggedFrame::new(vec![TaggedValue::number(1.0)]);
        assert_eq!(
            record.materialize(&frame),
            Err(DeoptMaterializationError::SlotNotHandle(0))
        );
    }

    #[test]
    fn rooted_tagged_handles_are_lexical_and_reuse_released_slots() {
        let roots = RootSet::new();
        let mut first = roots
            .root(TaggedValue::heap(HeapRef::new(9).unwrap()))
            .unwrap();
        let second = roots.root(TaggedValue::number(3.0)).unwrap();
        assert_eq!(roots.active_len(), 2);
        assert_eq!(
            first.value(),
            Some(TaggedValue::heap(HeapRef::new(9).unwrap()))
        );
        assert_eq!(roots.snapshot().unwrap().len(), 2);

        first.replace(TaggedValue::null()).unwrap();
        assert_eq!(first.value(), Some(TaggedValue::null()));
        assert_eq!(
            first.replace(TaggedValue(TAG_BOOLEAN | 2)),
            Err(InvalidTaggedValue::InvalidBooleanPayload)
        );
        assert_eq!(first.value(), Some(TaggedValue::null()));

        drop(first);
        assert_eq!(roots.active_len(), 1);
        let third = roots.root(TaggedValue::undefined()).unwrap();
        assert_eq!(roots.active_len(), 2);
        assert_eq!(second.value(), Some(TaggedValue::number(3.0)));
        assert_eq!(third.value(), Some(TaggedValue::undefined()));
    }

    #[test]
    fn no_gc_scope_is_lexical_non_nestable_and_blocks_safepoints() {
        let state = NoGcState::default();
        assert!(!state.is_active());
        assert_eq!(state.require_safepoint(), Ok(()));
        {
            let scope = state.enter().unwrap();
            assert!(scope.is_active());
            scope.assert_active();
            assert_eq!(
                state.require_safepoint(),
                Err(NoGcError::SafepointForbidden)
            );
            assert!(matches!(state.enter(), Err(NoGcError::AlreadyActive)));
        }
        assert!(!state.is_active());
        assert_eq!(state.require_safepoint(), Ok(()));
    }

    #[test]
    fn rooted_handles_reject_invalid_words_before_publishing_them() {
        let roots = RootSet::new();
        assert!(matches!(
            roots.root(TaggedValue(TAG_HEAP_REF)),
            Err(InvalidTaggedValue::NullHeapReference)
        ));
        assert_eq!(roots.active_len(), 0);
        assert!(roots.snapshot().unwrap().is_empty());
    }
}
