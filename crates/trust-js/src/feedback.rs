//! Versioned, representation-independent per-function feedback layout.
//!
//! See `FEEDBACK_SCHEMA.md`. This module deliberately contains no `Value` discriminants, shape
//! numbers, object addresses, or IC structs. Those belong to versioned runtime adapters.

use std::cell::{Cell, OnceCell};
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Numeric encodings below are part of the profile schema and may not be reordered in place.
pub(crate) const SCHEMA_VERSION: u16 = 1;

const PROFILE_MAGIC: &[u8; 8] = b"TRUST_JS";
const PROFILE_FORMAT_VERSION: u16 = 1;
const PROFILE_SCOPE_VECTOR_LOCAL: u8 = 1;
const PROFILE_HEADER_LEN: usize = 48;

static PROCESS_PROFILE_SESSION: OnceLock<u64> = OnceLock::new();
static NEXT_PROFILE_VECTOR_ID: AtomicU64 = AtomicU64::new(1);
static DETAILED_FEEDBACK_ENABLED: OnceLock<bool> = OnceLock::new();

fn process_profile_session() -> u64 {
    *PROCESS_PROFILE_SESSION.get_or_init(|| RandomState::new().build_hasher().finish().max(1))
}

fn detailed_feedback_enabled() -> bool {
    *DETAILED_FEEDBACK_ENABLED
        .get_or_init(|| std::env::var_os("TRUST_JS_FEEDBACK_PROFILE").is_some())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub(crate) struct SiteId(u32);

impl SiteId {
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// The source-level semantic operation represented by one feedback site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum OperationKind {
    NamedLoad = 1,
    NamedStore = 2,
    ElementLoad = 3,
    ElementStore = 4,
    Call = 5,
    Construct = 6,
    Arithmetic = 7,
    Branch = 8,
    Loop = 9,
    Allocation = 10,
}

/// Representation-independent meaning of one observation payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum ObservationKind {
    ValueClass = 1,
    ReceiverLayout = 2,
    HolderLayout = 3,
    ElementAccess = 4,
    CallTarget = 5,
    BranchCount = 6,
    Allocation = 7,
    PropertyAccess = 8,
}

/// Stable outcome class for a named-property access. The low nibble of an observation's flags
/// carries bounded prototype-depth/adapter details; the high nibble carries this encoding.
/// Values may only be appended, never reordered within a schema version.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum PropertyOutcome {
    Data = 1,
    Absent = 2,
    Created = 3,
    Accessor = 4,
    Exotic = 5,
    Rejected = 6,
}

/// Abstract receiver families for indexed/named element operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum ElementReceiverKind {
    Ordinary = 1,
    Array = 2,
    TypedArray = 3,
    String = 4,
    Primitive = 5,
    Exotic = 6,
}

/// Abstract post-coercion property-key categories. Exact indices and strings are intentionally
/// not profile data: they are workload facts, not portable optimizer identities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum ElementKeyKind {
    Index = 1,
    CanonicalNumeric = 2,
    String = 3,
    Symbol = 4,
}

/// Result family of an indexed operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum ElementOutcome {
    OwnData = 1,
    Hole = 2,
    Prototype = 3,
    Absent = 4,
    Accessor = 5,
    Exotic = 6,
    Created = 7,
    Rejected = 8,
}

/// Abstract callable target family. The target's current address or object identity is never
/// serialized; these families are stable enough for deciding which call path needs a guard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum CallTargetKind {
    UserFunction = 1,
    NativeFunction = 2,
    BoundFunction = 3,
    Proxy = 4,
    WrappedFunction = 5,
    CallableObject = 6,
    NonCallable = 7,
}

/// Bounded argument-count class for a completed call site. Spread/array calls use the actual
/// expanded list length, but retain only this portable bucket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum CallArityKind {
    Zero = 1,
    One = 2,
    Few = 3,
    Many = 4,
}

/// Whether a callable needs a retained activation environment or a dynamic forwarding path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum CallEnvironmentKind {
    None = 1,
    Captured = 2,
    Dynamic = 3,
    Unknown = 4,
}

const ELEMENT_RECEIVER_SHIFT: u32 = 0;
const ELEMENT_KEY_SHIFT: u32 = 8;
const ELEMENT_OUTCOME_SHIFT: u32 = 16;

const CALL_TARGET_SHIFT: u32 = 0;
const CALL_ARITY_SHIFT: u32 = 8;
const CALL_ENVIRONMENT_SHIFT: u32 = 16;

/// Abstract allocation families for explicit allocation bytecodes. These names describe the
/// ECMAScript result family, not the current Rust object representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum AllocationObjectKind {
    Function = 1,
    RegExp = 2,
    Array = 3,
    Object = 4,
}

/// Bounded requested-capacity class for an allocation site. The unit is operation-specific
/// (array elements, object properties, or source bytes for a RegExp), never allocator metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum AllocationSizeKind {
    Zero = 1,
    Small = 2,
    Medium = 3,
    Large = 4,
}

const ALLOCATION_OBJECT_SHIFT: u32 = 0;
const ALLOCATION_SIZE_SHIFT: u32 = 8;

pub(crate) const fn allocation_observation_payload(
    object: AllocationObjectKind,
    size: AllocationSizeKind,
) -> u32 {
    (1_u32 << (ALLOCATION_OBJECT_SHIFT + object as u32 - 1))
        | (1_u32 << (ALLOCATION_SIZE_SHIFT + size as u32 - 1))
}

fn allocation_group_counts(payload: u32) -> (u32, u32) {
    (
        ((payload >> ALLOCATION_OBJECT_SHIFT) & 0x0f).count_ones(),
        ((payload >> ALLOCATION_SIZE_SHIFT) & 0x0f).count_ones(),
    )
}

pub(crate) const fn allocation_size_kind(requested_units: usize) -> AllocationSizeKind {
    if requested_units == 0 {
        AllocationSizeKind::Zero
    } else if requested_units <= 4 {
        AllocationSizeKind::Small
    } else if requested_units <= 32 {
        AllocationSizeKind::Medium
    } else {
        AllocationSizeKind::Large
    }
}

// Branch counters are deliberately saturating. Conditional sites store taken/fallthrough
// counts as two u16 lanes; loop sites use the full u32 payload for back-edge transfers.
const BRANCH_TAKEN_SHIFT: u32 = 0;
const BRANCH_FALLTHROUGH_SHIFT: u32 = 16;
const BRANCH_COUNTER_MASK: u32 = 0xffff;

pub(crate) const fn element_observation_payload(
    receiver: ElementReceiverKind,
    key: ElementKeyKind,
    outcome: ElementOutcome,
) -> u32 {
    (1_u32 << (ELEMENT_RECEIVER_SHIFT + receiver as u32 - 1))
        | (1_u32 << (ELEMENT_KEY_SHIFT + key as u32 - 1))
        | (1_u32 << (ELEMENT_OUTCOME_SHIFT + outcome as u32 - 1))
}

fn element_group_counts(payload: u32) -> (u32, u32, u32) {
    (
        ((payload >> ELEMENT_RECEIVER_SHIFT) & 0x3f).count_ones(),
        ((payload >> ELEMENT_KEY_SHIFT) & 0x0f).count_ones(),
        ((payload >> ELEMENT_OUTCOME_SHIFT) & 0xff).count_ones(),
    )
}

pub(crate) const fn call_observation_payload(
    target: CallTargetKind,
    arity: CallArityKind,
    environment: CallEnvironmentKind,
) -> u32 {
    (1_u32 << (CALL_TARGET_SHIFT + target as u32 - 1))
        | (1_u32 << (CALL_ARITY_SHIFT + arity as u32 - 1))
        | (1_u32 << (CALL_ENVIRONMENT_SHIFT + environment as u32 - 1))
}

fn call_group_counts(payload: u32) -> (u32, u32, u32) {
    (
        ((payload >> CALL_TARGET_SHIFT) & 0x7f).count_ones(),
        ((payload >> CALL_ARITY_SHIFT) & 0x0f).count_ones(),
        ((payload >> CALL_ENVIRONMENT_SHIFT) & 0x0f).count_ones(),
    )
}

/// Runtime-only result of one named-property operation. Current `Props` shape numbers enter this
/// adapter record but are interned to vector-local layout tokens before any observation word is
/// written or serialized.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CurrentPropertyTrace {
    pub(crate) receiver_shape: Option<u32>,
    pub(crate) holder_shape: Option<u32>,
    pub(crate) depth: u8,
    pub(crate) field_slot: Option<u32>,
    pub(crate) outcome: Option<PropertyOutcome>,
    pub(crate) array_key_check: bool,
}

impl CurrentPropertyTrace {
    pub(crate) fn record(
        &mut self,
        outcome: PropertyOutcome,
        holder_shape: Option<u32>,
        depth: u8,
        field_slot: Option<u32>,
    ) {
        if self.outcome.is_some() {
            return;
        }
        self.outcome = Some(outcome);
        self.holder_shape = holder_shape;
        self.depth = depth;
        self.field_slot = field_slot;
    }
}

pub(crate) const PROPERTY_DEPTH_MASK: u8 = 0x07;
pub(crate) const PROPERTY_ARRAY_KEY_CHECK: u8 = 0x08;
const PROPERTY_OUTCOME_SHIFT: u8 = 4;

pub(crate) const fn property_access_flags(
    outcome: PropertyOutcome,
    depth: u8,
    array_key_check: bool,
) -> u8 {
    ((outcome as u8) << PROPERTY_OUTCOME_SHIFT)
        | (depth & PROPERTY_DEPTH_MASK)
        | if array_key_check {
            PROPERTY_ARRAY_KEY_CHECK
        } else {
            0
        }
}

fn property_outcome(flags: u8) -> Option<PropertyOutcome> {
    Some(match flags >> PROPERTY_OUTCOME_SHIFT {
        1 => PropertyOutcome::Data,
        2 => PropertyOutcome::Absent,
        3 => PropertyOutcome::Created,
        4 => PropertyOutcome::Accessor,
        5 => PropertyOutcome::Exotic,
        6 => PropertyOutcome::Rejected,
        _ => return None,
    })
}

/// Stable semantic classes for `ValueClass` observations.
///
/// ECMAScript has one binary64 Number type. `NumberInt32` is an optimizer refinement for values
/// that can use int32 arithmetic without losing -0 or overflowing; it is not a language type and
/// never changes the operation's specified Number semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum ValueClass {
    Undefined = 1,
    Null = 2,
    Boolean = 3,
    NumberInt32 = 4,
    NumberDouble = 5,
    String = 6,
    BigInt = 7,
    Symbol = 8,
    Object = 9,
}

impl ValueClass {
    const ALL_BITS: u32 = (1 << 9) - 1;

    pub(crate) const fn bit(self) -> u32 {
        1 << (self as u8 - 1)
    }
}

/// Bounded live entry observations for the optimizing tier. These use the same portable
/// classes as site feedback, but are indexed by formal argument position, not bytecode PC.
/// A native entry checkpoint supplies class bits without cloning/materializing Values.
/// They are samples, never proofs: publication requires guards on every invocation.
#[cfg(feature = "optimizing-jit")]
pub(crate) type StructuralObservation = (u32, Cell<Option<crate::bytecode::IcState>>, Cell<bool>);

#[cfg(feature = "optimizing-jit")]
pub(crate) struct EntrySamples {
    pub(crate) entries: Cell<u64>,
    pub(crate) interval: Cell<u32>,
    pub(crate) samples: Cell<u8>,
    pub(crate) classes: [Cell<u32>; 16],
    pub(crate) properties: Box<[(u32, Cell<u32>)]>,
    /// Repeated warmed data-IC snapshots; scalar observations, never live proofs.
    pub(crate) structural: Box<[StructuralObservation]>,
}

#[cfg(feature = "optimizing-jit")]
impl EntrySamples {
    pub(crate) fn new(interval: u32) -> Self {
        Self {
            entries: Cell::new(0),
            interval: Cell::new(interval),
            samples: Cell::new(0),
            classes: std::array::from_fn(|_| Cell::new(0)),
            properties: Box::default(),
            structural: Box::default(),
        }
    }
}

/// Semantic position of an observation within its site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum ObservationRole {
    Operand0 = 1,
    Operand1 = 2,
    Result = 3,
    Receiver = 4,
    Holder = 5,
    Access = 6,
    Target = 7,
    Outcome = 8,
}

/// Runtime-only bridge to the current baseline machinery. These indexes are never serialized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeBinding {
    Unbound,
    PropertyIc { first_way: u32, way_count: u8 },
}

/// Stable abstract state stored in an observation word. Numeric encodings are schema data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum ObservationState {
    Uninitialized = 0,
    Monomorphic = 1,
    Polymorphic = 2,
    Absent = 3,
    Generic = 4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub(crate) struct ObservationWord(u64);

impl ObservationWord {
    pub(crate) const UNINITIALIZED: Self = Self::new(ObservationState::Uninitialized, 0, 0);

    pub(crate) const fn new(state: ObservationState, payload: u32, flags: u8) -> Self {
        Self((state as u64) | ((flags as u64) << 8) | ((payload as u64) << 16))
    }

    fn decoded_state(self) -> Option<ObservationState> {
        Some(match self.0 as u8 {
            0 => ObservationState::Uninitialized,
            1 => ObservationState::Monomorphic,
            2 => ObservationState::Polymorphic,
            3 => ObservationState::Absent,
            4 => ObservationState::Generic,
            _ => return None,
        })
    }

    fn is_valid(self) -> bool {
        self.0 >> 48 == 0 && self.decoded_state().is_some()
    }

    fn is_valid_for(self, kind: ObservationKind) -> bool {
        if !self.is_valid() {
            return false;
        }
        if kind == ObservationKind::PropertyAccess {
            let flags = (self.0 >> 8) as u8;
            let payload = self.payload_bits();
            return match self.decoded_state().unwrap() {
                ObservationState::Uninitialized | ObservationState::Generic => {
                    flags == 0 && payload == 0
                }
                ObservationState::Polymorphic => flags == 0 && (2..=4).contains(&payload),
                ObservationState::Absent => {
                    property_outcome(flags) == Some(PropertyOutcome::Absent) && payload == 0
                }
                ObservationState::Monomorphic => match property_outcome(flags) {
                    Some(PropertyOutcome::Data) => payload != 0,
                    Some(
                        PropertyOutcome::Created
                        | PropertyOutcome::Accessor
                        | PropertyOutcome::Exotic
                        | PropertyOutcome::Rejected,
                    ) => payload == 0,
                    _ => false,
                },
            };
        }
        if kind == ObservationKind::ElementAccess {
            if ((self.0 >> 8) as u8) != 0 {
                return false;
            }
            let payload = self.payload_bits();
            let (receivers, keys, outcomes) = element_group_counts(payload);
            return match self.decoded_state().unwrap() {
                ObservationState::Uninitialized | ObservationState::Generic => payload == 0,
                ObservationState::Monomorphic => (receivers, keys, outcomes) == (1, 1, 1),
                ObservationState::Polymorphic => {
                    payload != 0
                        && (1..=4).contains(&receivers)
                        && (1..=4).contains(&keys)
                        && (1..=4).contains(&outcomes)
                }
                ObservationState::Absent => false,
            };
        }
        if kind == ObservationKind::CallTarget {
            if ((self.0 >> 8) as u8) != 0 {
                return false;
            }
            let payload = self.payload_bits();
            let (targets, arities, environments) = call_group_counts(payload);
            return match self.decoded_state().unwrap() {
                ObservationState::Uninitialized | ObservationState::Generic => payload == 0,
                ObservationState::Monomorphic => (targets, arities, environments) == (1, 1, 1),
                ObservationState::Polymorphic => {
                    payload != 0
                        && (1..=4).contains(&targets)
                        && (1..=4).contains(&arities)
                        && (1..=4).contains(&environments)
                }
                ObservationState::Absent => false,
            };
        }
        if kind == ObservationKind::BranchCount {
            if ((self.0 >> 8) as u8) != 0 {
                return false;
            }
            let payload = self.payload_bits();
            return match self.decoded_state().unwrap() {
                ObservationState::Uninitialized => payload == 0,
                ObservationState::Monomorphic => payload != 0,
                ObservationState::Polymorphic => {
                    payload & BRANCH_COUNTER_MASK != 0
                        && (payload >> BRANCH_FALLTHROUGH_SHIFT) & BRANCH_COUNTER_MASK != 0
                }
                ObservationState::Generic => payload == 0,
                ObservationState::Absent => false,
            };
        }
        if kind == ObservationKind::Allocation {
            if ((self.0 >> 8) as u8) != 0 {
                return false;
            }
            let payload = self.payload_bits();
            let (objects, sizes) = allocation_group_counts(payload);
            return match self.decoded_state().unwrap() {
                ObservationState::Uninitialized | ObservationState::Generic => payload == 0,
                ObservationState::Monomorphic => (objects, sizes) == (1, 1),
                ObservationState::Polymorphic => {
                    payload != 0 && (1..=4).contains(&objects) && (1..=4).contains(&sizes)
                }
                ObservationState::Absent => false,
            };
        }
        if kind != ObservationKind::ValueClass {
            return true;
        }
        if ((self.0 >> 8) as u8) != 0 {
            return false;
        }
        let bits = self.payload_bits();
        if bits & !ValueClass::ALL_BITS != 0 {
            return false;
        }
        match self.decoded_state().unwrap() {
            ObservationState::Uninitialized | ObservationState::Generic => bits == 0,
            ObservationState::Monomorphic => bits.count_ones() == 1,
            ObservationState::Polymorphic => (2..=4).contains(&bits.count_ones()),
            ObservationState::Absent => false,
        }
    }

    fn payload_bits(self) -> u32 {
        (self.0 >> 16) as u32
    }

    #[cfg(test)]
    pub(crate) fn state(self) -> ObservationState {
        self.decoded_state()
            .expect("invalid in-process observation state")
    }

    #[cfg(test)]
    pub(crate) fn payload(self) -> u32 {
        (self.0 >> 16) as u32
    }

    #[cfg(test)]
    pub(crate) fn flags(self) -> u8 {
        (self.0 >> 8) as u8
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub(crate) struct SlotDescriptor {
    pub(crate) kind: ObservationKind,
    pub(crate) role: ObservationRole,
}

/// Compact immutable description of one semantic site.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub(crate) struct SiteDescriptor {
    pub(crate) bytecode_pc: u32,
    pub(crate) first_slot: u32,
    pub(crate) operation: OperationKind,
    pub(crate) slot_count: u8,
    reserved: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FeedbackLayout {
    version: u16,
    sites: Box<[SiteDescriptor]>,
    slots: Box<[SlotDescriptor]>,
}

impl FeedbackLayout {
    #[cfg(test)]
    pub(crate) fn version(&self) -> u16 {
        self.version
    }

    pub(crate) fn len(&self) -> usize {
        self.sites.len()
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    pub(crate) fn slot_len(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn site(&self, id: SiteId) -> Option<&SiteDescriptor> {
        self.sites.get(id.index())
    }

    pub(crate) fn slots(&self, id: SiteId) -> Option<&[SlotDescriptor]> {
        let site = self.site(id)?;
        let start = site.first_slot as usize;
        let end = start.checked_add(site.slot_count as usize)?;
        self.slots.get(start..end)
    }

    pub(crate) fn site_ids(&self) -> impl ExactSizeIterator<Item = SiteId> + '_ {
        (0..self.sites.len()).map(|index| SiteId(index as u32))
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.sites
            .len()
            .saturating_mul(std::mem::size_of::<SiteDescriptor>())
            .saturating_add(
                self.slots
                    .len()
                    .saturating_mul(std::mem::size_of::<SlotDescriptor>()),
            )
    }

    /// Hash only the specified numeric schema fields, never Rust padding or discriminants.
    fn stable_hash(&self) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;

        fn add(hash: &mut u64, bytes: &[u8]) {
            for byte in bytes {
                *hash ^= u64::from(*byte);
                *hash = hash.wrapping_mul(FNV_PRIME);
            }
        }

        let mut hash = FNV_OFFSET;
        add(&mut hash, &self.version.to_le_bytes());
        add(&mut hash, &(self.sites.len() as u64).to_le_bytes());
        add(&mut hash, &(self.slots.len() as u64).to_le_bytes());
        for site in &self.sites {
            add(&mut hash, &site.bytecode_pc.to_le_bytes());
            add(&mut hash, &site.first_slot.to_le_bytes());
            add(&mut hash, &[site.operation as u8, site.slot_count]);
        }
        for slot in &self.slots {
            add(&mut hash, &[slot.kind as u8, slot.role as u8]);
        }
        hash
    }
}

/// Why a serialized profile was deliberately not consumed.
///
/// These distinctions are part of the fail-closed ingestion contract. Callers may report them,
/// but must not reinterpret or repair incompatible observation words themselves.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)] // consumed by the bounded profile-dump/ingestion slice later in Phase 1
pub(crate) enum ProfileDropReason {
    Truncated { minimum: usize, actual: usize },
    InvalidMagic,
    UnsupportedFormatVersion { found: u16 },
    UnsupportedSchemaVersion { found: u16 },
    UnsupportedScope { found: u8 },
    NonzeroReservedHeader,
    InvalidLength { expected: usize, actual: usize },
    DifferentProcessSession,
    DifferentFeedbackVector,
    LayoutMismatch,
    SiteCountMismatch { found: u32, expected: u32 },
    SlotCountMismatch { found: u32, expected: u32 },
    InvalidObservationWord { slot: u32 },
}

#[derive(Default)]
pub(crate) struct LayoutBuilder {
    sites: Vec<SiteDescriptor>,
    slots: Vec<SlotDescriptor>,
}

impl LayoutBuilder {
    pub(crate) fn add_site(
        &mut self,
        bytecode_pc: usize,
        operation: OperationKind,
        slots: &[SlotDescriptor],
    ) -> SiteId {
        assert!(
            !slots.is_empty(),
            "feedback sites require an observation slot"
        );
        let id = SiteId(
            u32::try_from(self.sites.len()).expect("feedback site identity space exhausted"),
        );
        let first_slot =
            u32::try_from(self.slots.len()).expect("feedback slot identity space exhausted");
        let slot_count =
            u8::try_from(slots.len()).expect("one feedback site has more than 255 slots");
        self.sites.push(SiteDescriptor {
            bytecode_pc: u32::try_from(bytecode_pc).expect("baseline bytecode PC exceeds u32"),
            first_slot,
            operation,
            slot_count,
            reserved: 0,
        });
        self.slots.extend_from_slice(slots);
        id
    }

    pub(crate) fn finish(self) -> FeedbackLayout {
        FeedbackLayout {
            version: SCHEMA_VERSION,
            sites: self.sites.into_boxed_slice(),
            slots: self.slots.into_boxed_slice(),
        }
    }
}

/// The payload is intentionally lazy. Schema construction is always on; detailed observation is
/// opt-in and does not allocate a word array until an adapter requests it.
pub(crate) struct FeedbackVector {
    layout: FeedbackLayout,
    bindings: Box<[RuntimeBinding]>,
    words: OnceCell<Box<[Cell<u64>]>>,
    /// Assigned only when this vector participates in profile serialization or ingestion.
    profile_vector_id: Cell<u64>,
    detailed_enabled: bool,
}

impl FeedbackVector {
    pub(crate) fn new(layout: FeedbackLayout, bindings: Box<[RuntimeBinding]>) -> Self {
        Self::new_with_enabled(layout, bindings, detailed_feedback_enabled())
    }

    pub(crate) fn new_with_enabled(
        layout: FeedbackLayout,
        bindings: Box<[RuntimeBinding]>,
        detailed_enabled: bool,
    ) -> Self {
        assert_eq!(layout.len(), bindings.len());
        Self {
            layout,
            bindings,
            words: OnceCell::new(),
            profile_vector_id: Cell::new(0),
            detailed_enabled,
        }
    }

    pub(crate) fn unbound(layout: FeedbackLayout) -> Self {
        let bindings = vec![RuntimeBinding::Unbound; layout.len()].into_boxed_slice();
        // Transformed/inlined chunks retain canonical schema identity but do not yet carry an
        // exact transformed-PC → baseline-site map. Disable detailed writes rather than letting
        // a coincidentally equal PC corrupt the baseline meaning.
        Self::new_with_enabled(layout, bindings, false)
    }

    pub(crate) fn layout(&self) -> &FeedbackLayout {
        &self.layout
    }

    #[inline(always)]
    pub(crate) fn detailed_enabled(&self) -> bool {
        self.detailed_enabled
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.layout
            .retained_bytes()
            .saturating_add(
                self.bindings
                    .len()
                    .saturating_mul(std::mem::size_of::<RuntimeBinding>()),
            )
            .saturating_add(self.words.get().map_or(0, |words| {
                words.len().saturating_mul(std::mem::size_of::<Cell<u64>>())
            }))
    }

    pub(crate) fn sites(&self) -> impl ExactSizeIterator<Item = (SiteId, RuntimeBinding)> + '_ {
        self.layout.site_ids().zip(self.bindings.iter().copied())
    }

    #[cfg(test)]
    pub(crate) fn write(
        &self,
        site: SiteId,
        kind: ObservationKind,
        role: ObservationRole,
        value: ObservationWord,
    ) {
        let Some(descriptors) = self.layout.slots(site) else {
            return;
        };
        let Some(offset) = descriptors
            .iter()
            .position(|slot| slot.kind == kind && slot.role == role)
        else {
            return;
        };
        let first = self.layout.site(site).unwrap().first_slot as usize;
        self.words()[first + offset].set(value.0);
    }

    pub(crate) fn merge_write(
        &self,
        site: SiteId,
        kind: ObservationKind,
        role: ObservationRole,
        incoming: ObservationWord,
    ) {
        let Some(descriptors) = self.layout.slots(site) else {
            return;
        };
        let Some(offset) = descriptors
            .iter()
            .position(|slot| slot.kind == kind && slot.role == role)
        else {
            return;
        };
        let first = self.layout.site(site).unwrap().first_slot as usize;
        let cell = &self.words()[first + offset];
        let current = ObservationWord(cell.get());
        let merged = if kind == ObservationKind::PropertyAccess {
            merge_property_access_words(current, incoming)
        } else if kind == ObservationKind::Allocation {
            merge_allocation_words(current, incoming)
        } else {
            merge_observation_words(current, incoming)
        };
        cell.set(merged.0);
    }

    /// Adapt one canonical runtime property result into the stable vector. This is used only for
    /// paths that do not populate the ordinary property IC (accessors and exotic/rejected
    /// operations); ordinary cached data/absence/creation paths are collected by the IC adapter.
    pub(crate) fn observe_current_property(
        &self,
        bytecode_pc: usize,
        shapes: &std::cell::RefCell<Vec<u32>>,
        trace: CurrentPropertyTrace,
    ) {
        if !self.detailed_enabled {
            return;
        }
        let Some(outcome) = trace.outcome else {
            return;
        };
        let Ok(bytecode_pc) = u32::try_from(bytecode_pc) else {
            return;
        };
        let Ok(index) = self
            .layout
            .sites
            .binary_search_by_key(&bytecode_pc, |site| site.bytecode_pc)
        else {
            return;
        };
        let site = SiteId(index as u32);
        if trace.depth > PROPERTY_DEPTH_MASK
            || trace
                .receiver_shape
                .is_some_and(|shape| !crate::value::is_cacheable_shape(shape))
            || trace
                .holder_shape
                .is_some_and(|shape| !crate::value::is_cacheable_shape(shape))
        {
            let generic = ObservationWord::new(ObservationState::Generic, 0, 0);
            self.merge_write(
                site,
                ObservationKind::ReceiverLayout,
                ObservationRole::Receiver,
                generic,
            );
            self.merge_write(
                site,
                ObservationKind::HolderLayout,
                ObservationRole::Holder,
                generic,
            );
            self.merge_write(
                site,
                ObservationKind::PropertyAccess,
                ObservationRole::Access,
                generic,
            );
            return;
        }

        let layout_flags = trace.depth | if trace.array_key_check { 0x80 } else { 0 };
        if let Some(shape) = trace.receiver_shape {
            self.merge_write(
                site,
                ObservationKind::ReceiverLayout,
                ObservationRole::Receiver,
                ObservationWord::new(
                    ObservationState::Monomorphic,
                    intern_current_shape(shapes, shape),
                    layout_flags,
                ),
            );
        }
        self.merge_write(
            site,
            ObservationKind::HolderLayout,
            ObservationRole::Holder,
            trace.holder_shape.map_or_else(
                || ObservationWord::new(ObservationState::Absent, 0, layout_flags),
                |shape| {
                    ObservationWord::new(
                        ObservationState::Monomorphic,
                        intern_current_shape(shapes, shape),
                        layout_flags,
                    )
                },
            ),
        );
        let payload = match (outcome, trace.field_slot) {
            (PropertyOutcome::Data, Some(slot)) => match slot.checked_add(1) {
                Some(payload) => payload,
                None => {
                    self.merge_write(
                        site,
                        ObservationKind::PropertyAccess,
                        ObservationRole::Access,
                        ObservationWord::new(ObservationState::Generic, 0, 0),
                    );
                    return;
                }
            },
            (PropertyOutcome::Data, None) => {
                self.merge_write(
                    site,
                    ObservationKind::PropertyAccess,
                    ObservationRole::Access,
                    ObservationWord::new(ObservationState::Generic, 0, 0),
                );
                return;
            }
            _ => 0,
        };
        let state = if outcome == PropertyOutcome::Absent {
            ObservationState::Absent
        } else {
            ObservationState::Monomorphic
        };
        self.merge_write(
            site,
            ObservationKind::PropertyAccess,
            ObservationRole::Access,
            ObservationWord::new(
                state,
                payload,
                property_access_flags(outcome, trace.depth, trace.array_key_check),
            ),
        );
    }

    /// Merge one current-representation element observation. Each semantic dimension is kept as
    /// a bounded one-hot set; when any dimension exceeds four alternatives the site becomes
    /// generic. This intentionally does not claim correlations between dimensions.
    pub(crate) fn observe_element(
        &self,
        bytecode_pc: usize,
        receiver: ElementReceiverKind,
        key: ElementKeyKind,
        outcome: ElementOutcome,
    ) {
        if !self.detailed_enabled {
            return;
        }
        let Ok(bytecode_pc) = u32::try_from(bytecode_pc) else {
            return;
        };
        let Ok(index) = self
            .layout
            .sites
            .binary_search_by_key(&bytecode_pc, |site| site.bytecode_pc)
        else {
            return;
        };
        let site = SiteId(index as u32);
        if self.layout.site(site).is_none_or(|site| {
            !matches!(
                site.operation,
                OperationKind::ElementLoad | OperationKind::ElementStore
            )
        }) {
            return;
        }
        let Some(descriptors) = self.layout.slots(site) else {
            return;
        };
        let Some(offset) = descriptors.iter().position(|slot| {
            slot.kind == ObservationKind::ElementAccess && slot.role == ObservationRole::Access
        }) else {
            return;
        };
        let first = self.layout.site(site).unwrap().first_slot as usize;
        let cell = &self.words()[first + offset];
        let current = ObservationWord(cell.get());
        let incoming_payload = element_observation_payload(receiver, key, outcome);
        if current.decoded_state() == Some(ObservationState::Generic) {
            return;
        }
        let merged_payload = if current.decoded_state() == Some(ObservationState::Uninitialized) {
            incoming_payload
        } else {
            current.payload_bits() | incoming_payload
        };
        let (receivers, keys, outcomes) = element_group_counts(merged_payload);
        let state = if receivers <= 1 && keys <= 1 && outcomes <= 1 {
            ObservationState::Monomorphic
        } else if (1..=4).contains(&receivers)
            && (1..=4).contains(&keys)
            && (1..=4).contains(&outcomes)
        {
            ObservationState::Polymorphic
        } else {
            ObservationState::Generic
        };
        cell.set(
            ObservationWord::new(
                state,
                if state == ObservationState::Generic {
                    0
                } else {
                    merged_payload
                },
                0,
            )
            .0,
        );
    }

    /// Merge one call/construct target observation. Target family, argument-count bucket, and
    /// activation-environment requirement are intentionally independent bounded dimensions; this
    /// avoids claiming a cross-product correlation that a future optimizer cannot guard cheaply.
    pub(crate) fn observe_call(
        &self,
        bytecode_pc: usize,
        target: CallTargetKind,
        arity: CallArityKind,
        environment: CallEnvironmentKind,
    ) {
        if !self.detailed_enabled {
            return;
        }
        let Ok(bytecode_pc) = u32::try_from(bytecode_pc) else {
            return;
        };
        let Ok(index) = self
            .layout
            .sites
            .binary_search_by_key(&bytecode_pc, |site| site.bytecode_pc)
        else {
            return;
        };
        let site = SiteId(index as u32);
        if self.layout.site(site).is_none_or(|site| {
            !matches!(
                site.operation,
                OperationKind::Call | OperationKind::Construct
            )
        }) {
            return;
        }
        let Some(descriptors) = self.layout.slots(site) else {
            return;
        };
        let Some(offset) = descriptors.iter().position(|slot| {
            slot.kind == ObservationKind::CallTarget && slot.role == ObservationRole::Target
        }) else {
            return;
        };
        let first = self.layout.site(site).unwrap().first_slot as usize;
        let cell = &self.words()[first + offset];
        let current = ObservationWord(cell.get());
        if current.decoded_state() == Some(ObservationState::Generic) {
            return;
        }
        let incoming_payload = call_observation_payload(target, arity, environment);
        let merged_payload = if current.decoded_state() == Some(ObservationState::Uninitialized) {
            incoming_payload
        } else {
            current.payload_bits() | incoming_payload
        };
        let (targets, arities, environments) = call_group_counts(merged_payload);
        let state = if targets <= 1 && arities <= 1 && environments <= 1 {
            ObservationState::Monomorphic
        } else if (1..=4).contains(&targets)
            && (1..=4).contains(&arities)
            && (1..=4).contains(&environments)
        {
            ObservationState::Polymorphic
        } else {
            ObservationState::Generic
        };
        cell.set(
            ObservationWord::new(
                state,
                if state == ObservationState::Generic {
                    0
                } else {
                    merged_payload
                },
                0,
            )
            .0,
        );
    }

    /// Record one completed conditional branch direction. The counters saturate instead of
    /// wrapping, so a hot loop can never appear cold (or more specialized) after overflow.
    pub(crate) fn observe_branch(&self, bytecode_pc: usize, taken: bool) {
        if !self.detailed_enabled {
            return;
        }
        let Ok(bytecode_pc) = u32::try_from(bytecode_pc) else {
            return;
        };
        let Ok(index) = self
            .layout
            .sites
            .binary_search_by_key(&bytecode_pc, |site| site.bytecode_pc)
        else {
            return;
        };
        let site = SiteId(index as u32);
        if self
            .layout
            .site(site)
            .is_none_or(|site| site.operation != OperationKind::Branch)
        {
            return;
        }
        let Some(descriptors) = self.layout.slots(site) else {
            return;
        };
        let Some(offset) = descriptors.iter().position(|slot| {
            slot.kind == ObservationKind::BranchCount && slot.role == ObservationRole::Outcome
        }) else {
            return;
        };
        let first = self.layout.site(site).unwrap().first_slot as usize;
        let cell = &self.words()[first + offset];
        let current = ObservationWord(cell.get());
        if current.decoded_state() == Some(ObservationState::Generic) {
            return;
        }
        let payload = current.payload_bits();
        let shift = if taken {
            BRANCH_TAKEN_SHIFT
        } else {
            BRANCH_FALLTHROUGH_SHIFT
        };
        let count = (payload >> shift) & BRANCH_COUNTER_MASK;
        let next = count.saturating_add(1).min(BRANCH_COUNTER_MASK);
        let updated = (payload & !(BRANCH_COUNTER_MASK << shift)) | (next << shift);
        let taken_count = (updated >> BRANCH_TAKEN_SHIFT) & BRANCH_COUNTER_MASK;
        let fallthrough_count = (updated >> BRANCH_FALLTHROUGH_SHIFT) & BRANCH_COUNTER_MASK;
        let state = if taken_count != 0 && fallthrough_count != 0 {
            ObservationState::Polymorphic
        } else {
            ObservationState::Monomorphic
        };
        cell.set(ObservationWord::new(state, updated, 0).0);
    }

    /// Record one completed unconditional loop back-edge transfer. The caller invokes this only
    /// after any host interruption poll succeeds, so the count describes actual control flow.
    pub(crate) fn observe_loop_backedge(&self, bytecode_pc: usize) {
        if !self.detailed_enabled {
            return;
        }
        let Ok(bytecode_pc) = u32::try_from(bytecode_pc) else {
            return;
        };
        let Ok(index) = self
            .layout
            .sites
            .binary_search_by_key(&bytecode_pc, |site| site.bytecode_pc)
        else {
            return;
        };
        let site = SiteId(index as u32);
        if self
            .layout
            .site(site)
            .is_none_or(|site| site.operation != OperationKind::Loop)
        {
            return;
        }
        let Some(descriptors) = self.layout.slots(site) else {
            return;
        };
        let Some(offset) = descriptors.iter().position(|slot| {
            slot.kind == ObservationKind::BranchCount && slot.role == ObservationRole::Outcome
        }) else {
            return;
        };
        let first = self.layout.site(site).unwrap().first_slot as usize;
        let cell = &self.words()[first + offset];
        let current = ObservationWord(cell.get());
        if current.decoded_state() == Some(ObservationState::Generic) {
            return;
        }
        let next = current.payload_bits().saturating_add(1);
        cell.set(ObservationWord::new(ObservationState::Monomorphic, next, 0).0);
    }

    /// Record an explicit allocation site's result family and requested capacity class. Lifetime
    /// and promotion are deliberately collected by the heap layer later; this site-local word
    /// never guesses a survival result at allocation time.
    pub(crate) fn observe_allocation(
        &self,
        bytecode_pc: usize,
        object: AllocationObjectKind,
        requested_units: usize,
    ) {
        if !self.detailed_enabled {
            return;
        }
        let Ok(bytecode_pc) = u32::try_from(bytecode_pc) else {
            return;
        };
        let Ok(index) = self
            .layout
            .sites
            .binary_search_by_key(&bytecode_pc, |site| site.bytecode_pc)
        else {
            return;
        };
        let site = SiteId(index as u32);
        if self
            .layout
            .site(site)
            .is_none_or(|site| site.operation != OperationKind::Allocation)
        {
            return;
        }
        let Some(descriptors) = self.layout.slots(site) else {
            return;
        };
        let Some(offset) = descriptors.iter().position(|slot| {
            slot.kind == ObservationKind::Allocation && slot.role == ObservationRole::Outcome
        }) else {
            return;
        };
        let first = self.layout.site(site).unwrap().first_slot as usize;
        let cell = &self.words()[first + offset];
        let current = ObservationWord(cell.get());
        let incoming = ObservationWord::new(
            ObservationState::Monomorphic,
            allocation_observation_payload(object, allocation_size_kind(requested_units)),
            0,
        );
        cell.set(merge_allocation_words(current, incoming).0);
    }

    /// Merge one stable semantic value class into the matching site's bitset.
    pub(crate) fn observe_value_class(
        &self,
        bytecode_pc: usize,
        role: ObservationRole,
        class: ValueClass,
    ) {
        let Ok(bytecode_pc) = u32::try_from(bytecode_pc) else {
            return;
        };
        let Ok(index) = self
            .layout
            .sites
            .binary_search_by_key(&bytecode_pc, |site| site.bytecode_pc)
        else {
            return;
        };
        let site = SiteId(index as u32);
        let Some(descriptors) = self.layout.slots(site) else {
            return;
        };
        let Some(offset) = descriptors
            .iter()
            .position(|slot| slot.kind == ObservationKind::ValueClass && slot.role == role)
        else {
            return;
        };
        let first = self.layout.site(site).unwrap().first_slot as usize;
        let cell = &self.words()[first + offset];
        let current = ObservationWord(cell.get());
        let incoming = ObservationWord::new(ObservationState::Monomorphic, class.bit(), 0);
        cell.set(merge_value_class_words(current, incoming).0);
    }

    fn words(&self) -> &[Cell<u64>] {
        self.words.get_or_init(|| {
            (0..self.layout.slot_len())
                .map(|_| Cell::new(ObservationWord::UNINITIALIZED.0))
                .collect::<Vec<_>>()
                .into_boxed_slice()
        })
    }

    fn profile_vector_id(&self) -> u64 {
        let current = self.profile_vector_id.get();
        if current != 0 {
            return current;
        }
        let assigned = crate::atomic_compat::try_update_u64(
            &NEXT_PROFILE_VECTOR_ID,
            Ordering::Relaxed,
            Ordering::Relaxed,
            |next| next.checked_add(1),
        )
        .expect("feedback profile vector identity space exhausted");
        self.profile_vector_id.set(assigned);
        assigned
    }

    /// Serialize a versioned compatibility envelope around this live vector's observations.
    ///
    /// Format version 1 is intentionally vector-local. The payload contains abstract observation
    /// words, but their current layout tokens are meaningful only to this exact vector.
    #[allow(dead_code)] // exposed to the bounded profile-dump slice later in Phase 1
    pub(crate) fn serialize_profile(&self) -> Vec<u8> {
        let site_count =
            u32::try_from(self.layout.len()).expect("feedback profile has more than u32 sites");
        let slot_count = u32::try_from(self.layout.slot_len())
            .expect("feedback profile has more than u32 slots");
        let mut bytes = Vec::with_capacity(
            PROFILE_HEADER_LEN.saturating_add(self.layout.slot_len().saturating_mul(8)),
        );
        bytes.extend_from_slice(PROFILE_MAGIC);
        bytes.extend_from_slice(&PROFILE_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&SCHEMA_VERSION.to_le_bytes());
        bytes.push(PROFILE_SCOPE_VECTOR_LOCAL);
        bytes.extend_from_slice(&[0; 3]);
        bytes.extend_from_slice(&process_profile_session().to_le_bytes());
        bytes.extend_from_slice(&self.profile_vector_id().to_le_bytes());
        bytes.extend_from_slice(&self.layout.stable_hash().to_le_bytes());
        bytes.extend_from_slice(&site_count.to_le_bytes());
        bytes.extend_from_slice(&slot_count.to_le_bytes());
        for word in self.words() {
            bytes.extend_from_slice(&word.get().to_le_bytes());
        }
        bytes
    }

    /// Validate and monotonically merge a profile produced by [`Self::serialize_profile`].
    ///
    /// No version-1 mismatch is upgraded heuristically. In particular, matching layout hashes do
    /// not permit data from another vector, because profile-local layout tokens could differ.
    #[allow(dead_code)] // exposed to the bounded profile-ingestion slice later in Phase 1
    pub(crate) fn merge_serialized_profile(&self, bytes: &[u8]) -> Result<(), ProfileDropReason> {
        if bytes.len() < PROFILE_HEADER_LEN {
            return Err(ProfileDropReason::Truncated {
                minimum: PROFILE_HEADER_LEN,
                actual: bytes.len(),
            });
        }
        if &bytes[0..8] != PROFILE_MAGIC {
            return Err(ProfileDropReason::InvalidMagic);
        }

        let format_version = u16::from_le_bytes([bytes[8], bytes[9]]);
        if format_version != PROFILE_FORMAT_VERSION {
            return Err(ProfileDropReason::UnsupportedFormatVersion {
                found: format_version,
            });
        }
        let schema_version = u16::from_le_bytes([bytes[10], bytes[11]]);
        if schema_version != SCHEMA_VERSION {
            return Err(ProfileDropReason::UnsupportedSchemaVersion {
                found: schema_version,
            });
        }
        if bytes[12] != PROFILE_SCOPE_VECTOR_LOCAL {
            return Err(ProfileDropReason::UnsupportedScope { found: bytes[12] });
        }
        if bytes[13..16] != [0; 3] {
            return Err(ProfileDropReason::NonzeroReservedHeader);
        }

        let session = read_u64(bytes, 16);
        if session != process_profile_session() {
            return Err(ProfileDropReason::DifferentProcessSession);
        }
        let vector_id = read_u64(bytes, 24);
        if vector_id != self.profile_vector_id() {
            return Err(ProfileDropReason::DifferentFeedbackVector);
        }
        if read_u64(bytes, 32) != self.layout.stable_hash() {
            return Err(ProfileDropReason::LayoutMismatch);
        }

        let site_count = read_u32(bytes, 40);
        let expected_sites =
            u32::try_from(self.layout.len()).expect("feedback profile has more than u32 sites");
        if site_count != expected_sites {
            return Err(ProfileDropReason::SiteCountMismatch {
                found: site_count,
                expected: expected_sites,
            });
        }
        let slot_count = read_u32(bytes, 44);
        let expected_slots = u32::try_from(self.layout.slot_len())
            .expect("feedback profile has more than u32 slots");
        if slot_count != expected_slots {
            return Err(ProfileDropReason::SlotCountMismatch {
                found: slot_count,
                expected: expected_slots,
            });
        }
        let payload_len = (slot_count as usize)
            .checked_mul(8)
            .expect("feedback profile payload length overflow");
        let expected_len = PROFILE_HEADER_LEN
            .checked_add(payload_len)
            .expect("feedback profile length overflow");
        if bytes.len() != expected_len {
            return Err(ProfileDropReason::InvalidLength {
                expected: expected_len,
                actual: bytes.len(),
            });
        }

        let (payload_words, remainder) = bytes[PROFILE_HEADER_LEN..].as_chunks::<8>();
        debug_assert!(remainder.is_empty());
        let incoming = payload_words
            .iter()
            .enumerate()
            .map(|(slot, bytes)| {
                let word = ObservationWord(u64::from_le_bytes(*bytes));
                if word.is_valid_for(self.layout.slots[slot].kind) {
                    Ok(word)
                } else {
                    Err(ProfileDropReason::InvalidObservationWord { slot: slot as u32 })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        for site_id in self.layout.site_ids() {
            let site = self.layout.site(site_id).unwrap();
            let first = site.first_slot as usize;
            let slots = self.layout.slots(site_id).unwrap();
            for (offset, descriptor) in slots.iter().enumerate() {
                let slot = first + offset;
                let current = ObservationWord(self.words()[slot].get());
                let merged = match descriptor.kind {
                    ObservationKind::ValueClass => merge_value_class_words(current, incoming[slot]),
                    ObservationKind::ElementAccess => {
                        merge_bounded_words(current, incoming[slot], element_group_counts)
                    }
                    ObservationKind::CallTarget => {
                        merge_bounded_words(current, incoming[slot], call_group_counts)
                    }
                    ObservationKind::BranchCount => {
                        merge_branch_count_words(current, incoming[slot], site.operation)
                    }
                    ObservationKind::Allocation => merge_allocation_words(current, incoming[slot]),
                    _ => merge_observation_words(current, incoming[slot]),
                };
                self.words()[slot].set(merged.0);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn read(
        &self,
        site: SiteId,
        kind: ObservationKind,
        role: ObservationRole,
    ) -> ObservationWord {
        let descriptors = self.layout.slots(site).unwrap();
        let offset = descriptors
            .iter()
            .position(|slot| slot.kind == kind && slot.role == role)
            .unwrap();
        let first = self.layout.site(site).unwrap().first_slot as usize;
        ObservationWord(self.words()[first + offset].get())
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

/// Merge the two independent allocation dimensions without preserving a false correlation
/// between object family and requested-capacity class. A site widens after four alternatives in
/// either dimension, matching the other bounded feedback families.
fn merge_allocation_words(current: ObservationWord, incoming: ObservationWord) -> ObservationWord {
    debug_assert!(current.is_valid_for(ObservationKind::Allocation));
    debug_assert!(incoming.is_valid_for(ObservationKind::Allocation));
    if current == incoming || incoming == ObservationWord::UNINITIALIZED {
        return current;
    }
    if current == ObservationWord::UNINITIALIZED {
        return incoming;
    }
    let current_state = current.decoded_state().unwrap();
    let incoming_state = incoming.decoded_state().unwrap();
    if current_state == ObservationState::Generic || incoming_state == ObservationState::Generic {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    }
    if current_state == ObservationState::Absent || incoming_state == ObservationState::Absent {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    }
    let payload = current.payload_bits() | incoming.payload_bits();
    let (objects, sizes) = allocation_group_counts(payload);
    let state = if objects == 1 && sizes == 1 {
        ObservationState::Monomorphic
    } else if (1..=4).contains(&objects) && (1..=4).contains(&sizes) {
        ObservationState::Polymorphic
    } else {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    };
    ObservationWord::new(state, payload, 0)
}

/// Merge bounded one-hot payloads from two diagnostic snapshots without inventing a correlation
/// between independent dimensions. This is the same monotone widening rule used by live
/// observation, applied to already-polymorphic incoming words as well.
fn merge_bounded_words(
    current: ObservationWord,
    incoming: ObservationWord,
    group_counts: fn(u32) -> (u32, u32, u32),
) -> ObservationWord {
    debug_assert!(current.is_valid());
    debug_assert!(incoming.is_valid());
    if current == incoming || incoming == ObservationWord::UNINITIALIZED {
        return current;
    }
    if current == ObservationWord::UNINITIALIZED {
        return incoming;
    }
    let current_state = current.decoded_state().unwrap();
    let incoming_state = incoming.decoded_state().unwrap();
    if current_state == ObservationState::Generic || incoming_state == ObservationState::Generic {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    }
    if current_state == ObservationState::Absent || incoming_state == ObservationState::Absent {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    }
    let payload = current.payload_bits() | incoming.payload_bits();
    let (first, second, third) = group_counts(payload);
    let state = if first == 1 && second == 1 && third == 1 {
        ObservationState::Monomorphic
    } else if (1..=4).contains(&first) && (1..=4).contains(&second) && (1..=4).contains(&third) {
        ObservationState::Polymorphic
    } else {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    };
    ObservationWord::new(state, payload, 0)
}

/// Merge saturating branch counters from separate profile snapshots. Loop sites use one full
/// payload lane; conditional sites use the two u16 direction lanes.
fn merge_branch_count_words(
    current: ObservationWord,
    incoming: ObservationWord,
    operation: OperationKind,
) -> ObservationWord {
    debug_assert!(current.is_valid());
    debug_assert!(incoming.is_valid());
    if current == incoming || incoming == ObservationWord::UNINITIALIZED {
        return current;
    }
    if current == ObservationWord::UNINITIALIZED {
        return incoming;
    }
    let current_state = current.decoded_state().unwrap();
    let incoming_state = incoming.decoded_state().unwrap();
    if current_state == ObservationState::Generic || incoming_state == ObservationState::Generic {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    }
    if current_state == ObservationState::Absent || incoming_state == ObservationState::Absent {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    }
    let payload = match operation {
        OperationKind::Loop => current
            .payload_bits()
            .saturating_add(incoming.payload_bits()),
        OperationKind::Branch => {
            let current_taken = current.payload_bits() & BRANCH_COUNTER_MASK;
            let incoming_taken = incoming.payload_bits() & BRANCH_COUNTER_MASK;
            let current_fallthrough =
                (current.payload_bits() >> BRANCH_FALLTHROUGH_SHIFT) & BRANCH_COUNTER_MASK;
            let incoming_fallthrough =
                (incoming.payload_bits() >> BRANCH_FALLTHROUGH_SHIFT) & BRANCH_COUNTER_MASK;
            current_taken
                .saturating_add(incoming_taken)
                .min(BRANCH_COUNTER_MASK)
                | (current_fallthrough
                    .saturating_add(incoming_fallthrough)
                    .min(BRANCH_COUNTER_MASK)
                    << BRANCH_FALLTHROUGH_SHIFT)
        }
        _ => return ObservationWord::new(ObservationState::Generic, 0, 0),
    };
    if payload == 0 {
        return ObservationWord::UNINITIALIZED;
    }
    let state = if operation == OperationKind::Branch
        && (payload & BRANCH_COUNTER_MASK) != 0
        && ((payload >> BRANCH_FALLTHROUGH_SHIFT) & BRANCH_COUNTER_MASK) != 0
    {
        ObservationState::Polymorphic
    } else {
        ObservationState::Monomorphic
    };
    ObservationWord::new(state, payload, 0)
}

fn merge_observation_words(current: ObservationWord, incoming: ObservationWord) -> ObservationWord {
    debug_assert!(current.is_valid());
    debug_assert!(incoming.is_valid());
    if current == incoming || incoming == ObservationWord::UNINITIALIZED {
        return current;
    }
    if current == ObservationWord::UNINITIALIZED {
        return incoming;
    }
    let current_state = current.decoded_state().unwrap();
    let incoming_state = incoming.decoded_state().unwrap();
    if current_state == ObservationState::Generic || incoming_state == ObservationState::Generic {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    }
    if incoming_state == ObservationState::Polymorphic {
        return incoming;
    }
    if current_state == ObservationState::Polymorphic {
        return current;
    }
    if current_state == ObservationState::Absent || incoming_state == ObservationState::Absent {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    }
    ObservationWord::new(ObservationState::Polymorphic, 2, 0)
}

fn merge_property_access_words(
    current: ObservationWord,
    incoming: ObservationWord,
) -> ObservationWord {
    debug_assert!(current.is_valid_for(ObservationKind::PropertyAccess));
    debug_assert!(incoming.is_valid_for(ObservationKind::PropertyAccess));
    if current == incoming || incoming == ObservationWord::UNINITIALIZED {
        return current;
    }
    if current == ObservationWord::UNINITIALIZED {
        return incoming;
    }
    if current.decoded_state() == Some(ObservationState::Generic)
        || incoming.decoded_state() == Some(ObservationState::Generic)
    {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    }
    if incoming.decoded_state() == Some(ObservationState::Polymorphic)
        && matches!(
            property_outcome((current.0 >> 8) as u8),
            Some(PropertyOutcome::Data | PropertyOutcome::Created)
        )
    {
        // The IC adapter has just summarized multiple ordinary data/create ways. Preserve its
        // bounded count when the earlier runtime word was itself an ordinary cacheable outcome.
        return incoming;
    }
    // Unlike layout-token observations, heterogeneous property outcomes cannot retain their
    // distinct flags/field payloads in one word. Widen conservatively instead of manufacturing a
    // partially exact polymorphic observation.
    ObservationWord::new(ObservationState::Generic, 0, 0)
}

fn intern_current_shape(shapes: &std::cell::RefCell<Vec<u32>>, shape: u32) -> u32 {
    debug_assert!(crate::value::is_cacheable_shape(shape));
    let mut shapes = shapes.borrow_mut();
    match shapes.iter().position(|existing| *existing == shape) {
        Some(index) => index as u32 + 1,
        None => {
            shapes.push(shape);
            shapes.len() as u32
        }
    }
}

fn merge_value_class_words(current: ObservationWord, incoming: ObservationWord) -> ObservationWord {
    debug_assert!(current.is_valid());
    debug_assert!(incoming.is_valid());
    let current_state = current.decoded_state().unwrap();
    let incoming_state = incoming.decoded_state().unwrap();
    if current_state == ObservationState::Generic || incoming_state == ObservationState::Generic {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    }
    if incoming_state == ObservationState::Uninitialized {
        return current;
    }
    if current_state == ObservationState::Uninitialized {
        return incoming;
    }
    if current_state == ObservationState::Absent || incoming_state == ObservationState::Absent {
        return ObservationWord::new(ObservationState::Generic, 0, 0);
    }
    let bits = (current.payload_bits() | incoming.payload_bits()) & ValueClass::ALL_BITS;
    match bits.count_ones() {
        0 => ObservationWord::new(ObservationState::Generic, 0, 0),
        1 => ObservationWord::new(ObservationState::Monomorphic, bits, 0),
        2..=4 => ObservationWord::new(ObservationState::Polymorphic, bits, 0),
        _ => ObservationWord::new(ObservationState::Generic, 0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RECEIVER: SlotDescriptor = SlotDescriptor {
        kind: ObservationKind::ReceiverLayout,
        role: ObservationRole::Receiver,
    };
    const HOLDER: SlotDescriptor = SlotDescriptor {
        kind: ObservationKind::HolderLayout,
        role: ObservationRole::Holder,
    };
    const ACCESS: SlotDescriptor = SlotDescriptor {
        kind: ObservationKind::PropertyAccess,
        role: ObservationRole::Access,
    };
    const ELEMENT_ACCESS: SlotDescriptor = SlotDescriptor {
        kind: ObservationKind::ElementAccess,
        role: ObservationRole::Access,
    };

    fn property_vector() -> FeedbackVector {
        let mut builder = LayoutBuilder::default();
        builder.add_site(4, OperationKind::NamedLoad, &[RECEIVER, HOLDER]);
        FeedbackVector::new(
            builder.finish(),
            vec![RuntimeBinding::Unbound].into_boxed_slice(),
        )
    }

    fn arithmetic_vector() -> FeedbackVector {
        let mut builder = LayoutBuilder::default();
        builder.add_site(
            4,
            OperationKind::Arithmetic,
            &[
                SlotDescriptor {
                    kind: ObservationKind::ValueClass,
                    role: ObservationRole::Operand0,
                },
                SlotDescriptor {
                    kind: ObservationKind::ValueClass,
                    role: ObservationRole::Result,
                },
            ],
        );
        FeedbackVector::new_with_enabled(
            builder.finish(),
            vec![RuntimeBinding::Unbound].into_boxed_slice(),
            true,
        )
    }

    fn element_vector() -> FeedbackVector {
        let mut builder = LayoutBuilder::default();
        builder.add_site(4, OperationKind::ElementLoad, &[ELEMENT_ACCESS]);
        FeedbackVector::new_with_enabled(
            builder.finish(),
            vec![RuntimeBinding::Unbound].into_boxed_slice(),
            true,
        )
    }

    #[test]
    fn layout_assigns_dense_stable_site_and_slot_ids() {
        let mut builder = LayoutBuilder::default();
        let first = builder.add_site(4, OperationKind::NamedLoad, &[RECEIVER, HOLDER]);
        let second = builder.add_site(
            9,
            OperationKind::Call,
            &[SlotDescriptor {
                kind: ObservationKind::CallTarget,
                role: ObservationRole::Target,
            }],
        );
        let layout = builder.finish();

        assert_eq!(layout.version(), 1);
        assert_eq!((first.index(), second.index()), (0, 1));
        assert_eq!(layout.site(first).unwrap().bytecode_pc, 4);
        assert_eq!(layout.slots(first).unwrap(), &[RECEIVER, HOLDER]);
        assert_eq!(layout.site(second).unwrap().first_slot, 2);
        assert_eq!(layout.site_ids().collect::<Vec<_>>(), vec![first, second]);
    }

    #[test]
    fn observation_words_are_lazy_and_layout_sized() {
        let vector = property_vector();
        let layout_bytes = vector.retained_bytes();

        vector.write(
            SiteId(0),
            ObservationKind::ReceiverLayout,
            ObservationRole::Receiver,
            ObservationWord::new(ObservationState::Monomorphic, 7, 0),
        );
        assert_eq!(vector.words().len(), 2);
        assert_eq!(
            vector
                .read(
                    SiteId(0),
                    ObservationKind::ReceiverLayout,
                    ObservationRole::Receiver,
                )
                .payload(),
            7
        );
        assert_eq!(
            vector.retained_bytes(),
            layout_bytes + 2 * std::mem::size_of::<Cell<u64>>()
        );
    }

    #[test]
    fn schema_numeric_encodings_are_frozen() {
        assert_eq!(OperationKind::NamedLoad as u8, 1);
        assert_eq!(OperationKind::Allocation as u8, 10);
        assert_eq!(ObservationKind::ValueClass as u8, 1);
        assert_eq!(ObservationKind::Allocation as u8, 7);
        assert_eq!(ObservationKind::PropertyAccess as u8, 8);
        assert_eq!(PropertyOutcome::Data as u8, 1);
        assert_eq!(PropertyOutcome::Rejected as u8, 6);
        assert_eq!(ElementReceiverKind::Ordinary as u8, 1);
        assert_eq!(ElementReceiverKind::Exotic as u8, 6);
        assert_eq!(ElementKeyKind::Index as u8, 1);
        assert_eq!(ElementKeyKind::Symbol as u8, 4);
        assert_eq!(ElementOutcome::OwnData as u8, 1);
        assert_eq!(ElementOutcome::Rejected as u8, 8);
        assert_eq!(CallTargetKind::UserFunction as u8, 1);
        assert_eq!(CallTargetKind::NonCallable as u8, 7);
        assert_eq!(CallArityKind::Zero as u8, 1);
        assert_eq!(CallArityKind::Many as u8, 4);
        assert_eq!(CallEnvironmentKind::None as u8, 1);
        assert_eq!(CallEnvironmentKind::Unknown as u8, 4);
        assert_eq!(AllocationObjectKind::Function as u8, 1);
        assert_eq!(AllocationObjectKind::Object as u8, 4);
        assert_eq!(AllocationSizeKind::Zero as u8, 1);
        assert_eq!(AllocationSizeKind::Large as u8, 4);
        assert_eq!(property_access_flags(PropertyOutcome::Data, 3, true), 0x1b);
        assert_eq!(ValueClass::Undefined as u8, 1);
        assert_eq!(ValueClass::Object as u8, 9);
        assert_eq!(ObservationRole::Operand0 as u8, 1);
        assert_eq!(ObservationRole::Outcome as u8, 8);
        assert_eq!(ObservationState::Uninitialized as u8, 0);
        assert_eq!(ObservationState::Generic as u8, 4);
        assert_eq!(std::mem::size_of::<SlotDescriptor>(), 2);
        assert_eq!(std::mem::size_of::<SiteDescriptor>(), 12);
    }

    #[test]
    fn element_feedback_keeps_bounded_semantic_dimensions() {
        let vector = element_vector();
        let payload = element_observation_payload(
            ElementReceiverKind::Array,
            ElementKeyKind::Index,
            ElementOutcome::OwnData,
        );
        assert_eq!(element_group_counts(payload), (1, 1, 1));
        assert!(
            ObservationWord::new(ObservationState::Monomorphic, payload, 0)
                .is_valid_for(ObservationKind::ElementAccess)
        );
        assert!(
            !ObservationWord::new(ObservationState::Monomorphic, payload, 1)
                .is_valid_for(ObservationKind::ElementAccess)
        );

        vector.observe_element(
            4,
            ElementReceiverKind::Array,
            ElementKeyKind::Index,
            ElementOutcome::OwnData,
        );
        vector.observe_element(
            4,
            ElementReceiverKind::Array,
            ElementKeyKind::Index,
            ElementOutcome::Hole,
        );
        let word = vector.read(
            SiteId(0),
            ObservationKind::ElementAccess,
            ObservationRole::Access,
        );
        assert_eq!(word.state(), ObservationState::Polymorphic);
        assert_eq!(element_group_counts(word.payload()), (1, 1, 2));

        for receiver in [
            ElementReceiverKind::Ordinary,
            ElementReceiverKind::TypedArray,
            ElementReceiverKind::String,
            ElementReceiverKind::Primitive,
        ] {
            vector.observe_element(4, receiver, ElementKeyKind::Index, ElementOutcome::OwnData);
        }
        assert_eq!(
            vector
                .read(
                    SiteId(0),
                    ObservationKind::ElementAccess,
                    ObservationRole::Access,
                )
                .state(),
            ObservationState::Generic
        );
    }

    #[test]
    fn call_feedback_keeps_target_arity_and_environment_bounded() {
        let mut builder = LayoutBuilder::default();
        builder.add_site(
            4,
            OperationKind::Call,
            &[SlotDescriptor {
                kind: ObservationKind::CallTarget,
                role: ObservationRole::Target,
            }],
        );
        let vector = FeedbackVector::new_with_enabled(
            builder.finish(),
            vec![RuntimeBinding::Unbound].into_boxed_slice(),
            true,
        );
        let payload = call_observation_payload(
            CallTargetKind::UserFunction,
            CallArityKind::Few,
            CallEnvironmentKind::Captured,
        );
        assert_eq!(call_group_counts(payload), (1, 1, 1));
        assert!(
            ObservationWord::new(ObservationState::Monomorphic, payload, 0)
                .is_valid_for(ObservationKind::CallTarget)
        );

        vector.observe_call(
            4,
            CallTargetKind::UserFunction,
            CallArityKind::Few,
            CallEnvironmentKind::Captured,
        );
        vector.observe_call(
            4,
            CallTargetKind::NativeFunction,
            CallArityKind::Many,
            CallEnvironmentKind::None,
        );
        let word = vector.read(
            SiteId(0),
            ObservationKind::CallTarget,
            ObservationRole::Target,
        );
        assert_eq!(word.state(), ObservationState::Polymorphic);
        assert_eq!(call_group_counts(word.payload()), (2, 2, 2));

        vector.observe_call(
            4,
            CallTargetKind::BoundFunction,
            CallArityKind::Zero,
            CallEnvironmentKind::Dynamic,
        );
        vector.observe_call(
            4,
            CallTargetKind::Proxy,
            CallArityKind::One,
            CallEnvironmentKind::Unknown,
        );
        vector.observe_call(
            4,
            CallTargetKind::WrappedFunction,
            CallArityKind::Few,
            CallEnvironmentKind::None,
        );
        assert_eq!(
            vector
                .read(
                    SiteId(0),
                    ObservationKind::CallTarget,
                    ObservationRole::Target,
                )
                .state(),
            ObservationState::Generic
        );
    }

    #[test]
    fn branch_feedback_counts_directions_and_saturates() {
        let mut builder = LayoutBuilder::default();
        builder.add_site(
            4,
            OperationKind::Branch,
            &[SlotDescriptor {
                kind: ObservationKind::BranchCount,
                role: ObservationRole::Outcome,
            }],
        );
        builder.add_site(
            8,
            OperationKind::Loop,
            &[SlotDescriptor {
                kind: ObservationKind::BranchCount,
                role: ObservationRole::Outcome,
            }],
        );
        let vector = FeedbackVector::new_with_enabled(
            builder.finish(),
            vec![RuntimeBinding::Unbound, RuntimeBinding::Unbound].into_boxed_slice(),
            true,
        );
        vector.observe_branch(4, true);
        vector.observe_branch(4, true);
        vector.observe_branch(4, false);
        let branch = vector.read(
            SiteId(0),
            ObservationKind::BranchCount,
            ObservationRole::Outcome,
        );
        assert_eq!(branch.payload() & BRANCH_COUNTER_MASK, 2);
        assert_eq!(branch.payload() >> BRANCH_FALLTHROUGH_SHIFT, 1);
        assert_eq!(branch.state(), ObservationState::Polymorphic);
        assert!(branch.is_valid_for(ObservationKind::BranchCount));

        for _ in 0..=u16::MAX {
            vector.observe_branch(4, true);
        }
        let branch = vector.read(
            SiteId(0),
            ObservationKind::BranchCount,
            ObservationRole::Outcome,
        );
        assert_eq!(branch.payload() & BRANCH_COUNTER_MASK, u32::from(u16::MAX));

        vector.observe_loop_backedge(8);
        vector.observe_loop_backedge(8);
        let loop_count = vector.read(
            SiteId(1),
            ObservationKind::BranchCount,
            ObservationRole::Outcome,
        );
        assert_eq!(loop_count.payload(), 2);
        assert_eq!(loop_count.state(), ObservationState::Monomorphic);
        assert!(loop_count.is_valid_for(ObservationKind::BranchCount));
    }

    #[test]
    fn allocation_feedback_keeps_kind_and_requested_size_bounded() {
        let mut builder = LayoutBuilder::default();
        builder.add_site(
            4,
            OperationKind::Allocation,
            &[SlotDescriptor {
                kind: ObservationKind::Allocation,
                role: ObservationRole::Outcome,
            }],
        );
        let vector = FeedbackVector::new_with_enabled(
            builder.finish(),
            vec![RuntimeBinding::Unbound].into_boxed_slice(),
            true,
        );

        vector.observe_allocation(4, AllocationObjectKind::Array, 2);
        let word = vector.read(
            SiteId(0),
            ObservationKind::Allocation,
            ObservationRole::Outcome,
        );
        assert_eq!(word.state(), ObservationState::Monomorphic);
        assert_eq!(allocation_group_counts(word.payload()), (1, 1));
        assert!(word.is_valid_for(ObservationKind::Allocation));

        vector.observe_allocation(4, AllocationObjectKind::Array, 64);
        let word = vector.read(
            SiteId(0),
            ObservationKind::Allocation,
            ObservationRole::Outcome,
        );
        assert_eq!(word.state(), ObservationState::Polymorphic);
        assert_eq!(allocation_group_counts(word.payload()), (1, 2));

        vector.observe_allocation(4, AllocationObjectKind::Object, 0);
        let word = vector.read(
            SiteId(0),
            ObservationKind::Allocation,
            ObservationRole::Outcome,
        );
        assert_eq!(word.state(), ObservationState::Polymorphic);
        assert_eq!(allocation_group_counts(word.payload()), (2, 3));
    }

    #[test]
    fn profile_merges_preserve_bounded_payloads_and_counts() {
        let element_a = ObservationWord::new(
            ObservationState::Monomorphic,
            element_observation_payload(
                ElementReceiverKind::Array,
                ElementKeyKind::Index,
                ElementOutcome::OwnData,
            ),
            0,
        );
        let element_b = ObservationWord::new(
            ObservationState::Monomorphic,
            element_observation_payload(
                ElementReceiverKind::TypedArray,
                ElementKeyKind::String,
                ElementOutcome::Hole,
            ),
            0,
        );
        let element = merge_bounded_words(element_a, element_b, element_group_counts);
        assert_eq!(element.state(), ObservationState::Polymorphic);
        assert_eq!(element_group_counts(element.payload()), (2, 2, 2));
        assert!(element.is_valid_for(ObservationKind::ElementAccess));

        let call_a = ObservationWord::new(
            ObservationState::Monomorphic,
            call_observation_payload(
                CallTargetKind::UserFunction,
                CallArityKind::One,
                CallEnvironmentKind::None,
            ),
            0,
        );
        let call_b = ObservationWord::new(
            ObservationState::Monomorphic,
            call_observation_payload(
                CallTargetKind::NativeFunction,
                CallArityKind::Many,
                CallEnvironmentKind::Captured,
            ),
            0,
        );
        let call = merge_bounded_words(call_a, call_b, call_group_counts);
        assert_eq!(call.state(), ObservationState::Polymorphic);
        assert_eq!(call_group_counts(call.payload()), (2, 2, 2));
        assert!(call.is_valid_for(ObservationKind::CallTarget));

        let branch_a = ObservationWord::new(ObservationState::Monomorphic, 3 | (2 << 16), 0);
        let branch_b = ObservationWord::new(ObservationState::Monomorphic, 4 | (5 << 16), 0);
        let branch = merge_branch_count_words(branch_a, branch_b, OperationKind::Branch);
        assert_eq!(branch.state(), ObservationState::Polymorphic);
        assert_eq!(branch.payload() & BRANCH_COUNTER_MASK, 7);
        assert_eq!(branch.payload() >> BRANCH_FALLTHROUGH_SHIFT, 7);
        assert!(branch.is_valid_for(ObservationKind::BranchCount));

        let loop_count = merge_branch_count_words(
            ObservationWord::new(ObservationState::Monomorphic, 10, 0),
            ObservationWord::new(ObservationState::Monomorphic, 20, 0),
            OperationKind::Loop,
        );
        assert_eq!(loop_count.state(), ObservationState::Monomorphic);
        assert_eq!(loop_count.payload(), 30);
        assert!(loop_count.is_valid_for(ObservationKind::BranchCount));
    }

    #[test]
    fn property_access_words_enforce_the_semantic_envelope() {
        let data = ObservationWord::new(
            ObservationState::Monomorphic,
            3,
            property_access_flags(PropertyOutcome::Data, 1, false),
        );
        assert!(data.is_valid_for(ObservationKind::PropertyAccess));
        assert_eq!(property_outcome(data.flags()), Some(PropertyOutcome::Data));

        let absent = ObservationWord::new(
            ObservationState::Absent,
            0,
            property_access_flags(PropertyOutcome::Absent, 0, false),
        );
        assert!(absent.is_valid_for(ObservationKind::PropertyAccess));

        let created = ObservationWord::new(
            ObservationState::Monomorphic,
            0,
            property_access_flags(PropertyOutcome::Created, 0, false),
        );
        assert!(created.is_valid_for(ObservationKind::PropertyAccess));

        assert!(!ObservationWord::new(
            ObservationState::Monomorphic,
            0,
            property_access_flags(PropertyOutcome::Data, 0, false),
        )
        .is_valid_for(ObservationKind::PropertyAccess));
        assert!(!ObservationWord::new(
            ObservationState::Monomorphic,
            1,
            property_access_flags(PropertyOutcome::Absent, 0, false),
        )
        .is_valid_for(ObservationKind::PropertyAccess));
        assert!(!ObservationWord::new(
            ObservationState::Absent,
            0,
            property_access_flags(PropertyOutcome::Data, 0, false),
        )
        .is_valid_for(ObservationKind::PropertyAccess));
    }

    #[test]
    fn current_property_adapter_interns_shapes_and_widens_mixed_outcomes() {
        let mut builder = LayoutBuilder::default();
        builder.add_site(4, OperationKind::NamedLoad, &[RECEIVER, HOLDER, ACCESS]);
        let vector = FeedbackVector::new_with_enabled(
            builder.finish(),
            vec![RuntimeBinding::Unbound].into_boxed_slice(),
            true,
        );
        let shapes = std::cell::RefCell::new(Vec::new());
        let trace = CurrentPropertyTrace {
            receiver_shape: Some(41),
            holder_shape: Some(73),
            depth: 1,
            field_slot: None,
            outcome: Some(PropertyOutcome::Accessor),
            array_key_check: false,
        };

        vector.observe_current_property(4, &shapes, trace);
        let site = vector.sites().next().unwrap().0;
        assert_eq!(
            vector
                .read(
                    site,
                    ObservationKind::ReceiverLayout,
                    ObservationRole::Receiver
                )
                .payload(),
            1
        );
        assert_eq!(
            vector
                .read(site, ObservationKind::HolderLayout, ObservationRole::Holder)
                .payload(),
            2
        );
        let access = vector.read(
            site,
            ObservationKind::PropertyAccess,
            ObservationRole::Access,
        );
        assert_eq!(access.state(), ObservationState::Monomorphic);
        assert_eq!(
            property_outcome(access.flags()),
            Some(PropertyOutcome::Accessor)
        );
        assert_eq!(&*shapes.borrow(), &[41, 73]);

        vector.observe_current_property(4, &shapes, trace);
        assert_eq!(
            vector
                .read(
                    site,
                    ObservationKind::PropertyAccess,
                    ObservationRole::Access
                )
                .state(),
            ObservationState::Monomorphic
        );

        vector.observe_current_property(
            4,
            &shapes,
            CurrentPropertyTrace {
                outcome: Some(PropertyOutcome::Exotic),
                ..trace
            },
        );
        assert_eq!(
            vector
                .read(
                    site,
                    ObservationKind::PropertyAccess,
                    ObservationRole::Access
                )
                .state(),
            ObservationState::Generic
        );
    }

    #[test]
    fn value_class_feedback_widens_without_losing_seen_categories() {
        let vector = arithmetic_vector();
        vector.observe_value_class(4, ObservationRole::Operand0, ValueClass::NumberInt32);
        vector.observe_value_class(4, ObservationRole::Operand0, ValueClass::NumberInt32);
        let word = vector.read(
            SiteId(0),
            ObservationKind::ValueClass,
            ObservationRole::Operand0,
        );
        assert_eq!(word.state(), ObservationState::Monomorphic);
        assert_eq!(word.payload(), ValueClass::NumberInt32.bit());

        for class in [
            ValueClass::NumberDouble,
            ValueClass::String,
            ValueClass::BigInt,
        ] {
            vector.observe_value_class(4, ObservationRole::Operand0, class);
        }
        let word = vector.read(
            SiteId(0),
            ObservationKind::ValueClass,
            ObservationRole::Operand0,
        );
        assert_eq!(word.state(), ObservationState::Polymorphic);
        assert_eq!(word.payload().count_ones(), 4);

        vector.observe_value_class(4, ObservationRole::Operand0, ValueClass::Object);
        assert_eq!(
            vector
                .read(
                    SiteId(0),
                    ObservationKind::ValueClass,
                    ObservationRole::Operand0,
                )
                .state(),
            ObservationState::Generic
        );
    }

    #[test]
    fn profile_envelope_encoding_is_frozen() {
        let vector = property_vector();
        let profile = vector.serialize_profile();

        assert_eq!(&profile[0..8], b"TRUST_JS");
        assert_eq!(u16::from_le_bytes([profile[8], profile[9]]), 1);
        assert_eq!(u16::from_le_bytes([profile[10], profile[11]]), 1);
        assert_eq!(profile[12], PROFILE_SCOPE_VECTOR_LOCAL);
        assert_eq!(&profile[13..16], &[0; 3]);
        assert_ne!(read_u64(&profile, 16), 0);
        assert_ne!(read_u64(&profile, 24), 0);
        assert_eq!(read_u64(&profile, 32), vector.layout.stable_hash());
        assert_eq!(vector.layout.stable_hash(), 0x2528_6bec_cecd_210a);
        assert_eq!(read_u32(&profile, 40), 1);
        assert_eq!(read_u32(&profile, 44), 2);
        assert_eq!(profile.len(), PROFILE_HEADER_LEN + 16);
    }

    #[test]
    fn profile_round_trip_merges_without_narrowing() {
        let vector = property_vector();
        vector.write(
            SiteId(0),
            ObservationKind::ReceiverLayout,
            ObservationRole::Receiver,
            ObservationWord::new(ObservationState::Monomorphic, 7, 0),
        );
        let profile = vector.serialize_profile();

        vector.write(
            SiteId(0),
            ObservationKind::ReceiverLayout,
            ObservationRole::Receiver,
            ObservationWord::UNINITIALIZED,
        );
        vector.merge_serialized_profile(&profile).unwrap();
        assert_eq!(
            vector
                .read(
                    SiteId(0),
                    ObservationKind::ReceiverLayout,
                    ObservationRole::Receiver,
                )
                .payload(),
            7
        );

        vector.write(
            SiteId(0),
            ObservationKind::ReceiverLayout,
            ObservationRole::Receiver,
            ObservationWord::new(ObservationState::Monomorphic, 8, 0),
        );
        vector.merge_serialized_profile(&profile).unwrap();
        assert_eq!(
            vector
                .read(
                    SiteId(0),
                    ObservationKind::ReceiverLayout,
                    ObservationRole::Receiver,
                )
                .state(),
            ObservationState::Polymorphic
        );

        vector.write(
            SiteId(0),
            ObservationKind::ReceiverLayout,
            ObservationRole::Receiver,
            ObservationWord::new(ObservationState::Generic, 0, 0),
        );
        vector.merge_serialized_profile(&profile).unwrap();
        assert_eq!(
            vector
                .read(
                    SiteId(0),
                    ObservationKind::ReceiverLayout,
                    ObservationRole::Receiver,
                )
                .state(),
            ObservationState::Generic
        );
    }

    #[test]
    fn profile_envelope_drops_every_incompatible_identity() {
        let vector = property_vector();
        let profile = vector.serialize_profile();

        let mut changed = profile.clone();
        changed[0] ^= 1;
        assert_eq!(
            vector.merge_serialized_profile(&changed),
            Err(ProfileDropReason::InvalidMagic)
        );

        let mut changed = profile.clone();
        changed[8..10].copy_from_slice(&2_u16.to_le_bytes());
        assert_eq!(
            vector.merge_serialized_profile(&changed),
            Err(ProfileDropReason::UnsupportedFormatVersion { found: 2 })
        );

        let mut changed = profile.clone();
        changed[10..12].copy_from_slice(&2_u16.to_le_bytes());
        assert_eq!(
            vector.merge_serialized_profile(&changed),
            Err(ProfileDropReason::UnsupportedSchemaVersion { found: 2 })
        );

        let mut changed = profile.clone();
        changed[12] = 2;
        assert_eq!(
            vector.merge_serialized_profile(&changed),
            Err(ProfileDropReason::UnsupportedScope { found: 2 })
        );

        let mut changed = profile.clone();
        changed[13] = 1;
        assert_eq!(
            vector.merge_serialized_profile(&changed),
            Err(ProfileDropReason::NonzeroReservedHeader)
        );

        let mut changed = profile.clone();
        changed[16..24].copy_from_slice(&read_u64(&profile, 16).wrapping_add(1).to_le_bytes());
        assert_eq!(
            vector.merge_serialized_profile(&changed),
            Err(ProfileDropReason::DifferentProcessSession)
        );

        assert_eq!(
            property_vector().merge_serialized_profile(&profile),
            Err(ProfileDropReason::DifferentFeedbackVector)
        );

        let mut changed = profile.clone();
        changed[32] ^= 1;
        assert_eq!(
            vector.merge_serialized_profile(&changed),
            Err(ProfileDropReason::LayoutMismatch)
        );

        let mut changed = profile.clone();
        changed[40..44].copy_from_slice(&2_u32.to_le_bytes());
        assert_eq!(
            vector.merge_serialized_profile(&changed),
            Err(ProfileDropReason::SiteCountMismatch {
                found: 2,
                expected: 1,
            })
        );

        let mut changed = profile;
        changed[44..48].copy_from_slice(&3_u32.to_le_bytes());
        assert_eq!(
            vector.merge_serialized_profile(&changed),
            Err(ProfileDropReason::SlotCountMismatch {
                found: 3,
                expected: 2,
            })
        );
    }

    #[test]
    fn profile_envelope_rejects_malformed_payload_before_mutating() {
        let vector = property_vector();
        let profile = vector.serialize_profile();

        assert_eq!(
            vector.merge_serialized_profile(&profile[..PROFILE_HEADER_LEN - 1]),
            Err(ProfileDropReason::Truncated {
                minimum: PROFILE_HEADER_LEN,
                actual: PROFILE_HEADER_LEN - 1,
            })
        );

        let mut changed = profile.clone();
        changed.push(0);
        assert_eq!(
            vector.merge_serialized_profile(&changed),
            Err(ProfileDropReason::InvalidLength {
                expected: profile.len(),
                actual: profile.len() + 1,
            })
        );

        let mut changed = profile;
        changed[PROFILE_HEADER_LEN] = 0xff;
        assert_eq!(
            vector.merge_serialized_profile(&changed),
            Err(ProfileDropReason::InvalidObservationWord { slot: 0 })
        );
        assert_eq!(
            vector
                .read(
                    SiteId(0),
                    ObservationKind::ReceiverLayout,
                    ObservationRole::Receiver,
                )
                .state(),
            ObservationState::Uninitialized
        );
    }
}
