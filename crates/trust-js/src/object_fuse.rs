//! The ObjectFuse (Djot's gap, candidate 4): which of an object's properties generated code may
//! treat as constant, after SpiderMonkey's (js/src at fa1e90ea: vm/ObjectFuse.h's SMDOC,
//! vm/ObjectFuse.cpp, vm/Watchtower.cpp). Each Realm's String.prototype has one, so that a site
//! reading `str.codePointAt` checks one word instead of the prototype's state.
//!
//! - Each property of the object's named entries has a state, two bits: Untracked, Constant
//!   (generated code may rely on its value) or NotConstant (written after it was Constant).
//!   Generated code marks a property Constant as it compiles a site that relies on it
//!   ([`ObjectFuse::try_optimize_constant`], SpiderMonkey's tryOptimizeConstantProperty).
//! - A value change ([`ObjectFuse::handle_value_change`], SpiderMonkey's
//!   handlePropertyValueChange) makes an Untracked property Constant and a Constant one
//!   NotConstant, setting the `invalidated` word. It is every write of the value, the same value
//!   included, and every defineProperty that is not redundant, attributes alone included
//!   (SpiderMonkey's define calls it after changing the flags, NativeObject.cpp:1449-1457). A
//!   redundant define changes nothing. Freezing and sealing change flags only and leave the fuse
//!   as it was (SpiderMonkey's freezeOrSealProperties): see [`with_flags_only`].
//! - Removing a tracked property bumps the generation and sets the word
//!   ([`ObjectFuse::handle_remove`], SpiderMonkey's handlePropertyRemove). This engine's maps close
//!   the gap a removal leaves (`Props::remove_key`), so the states after it move down with their
//!   properties, and a tracked property after the removed one changes its slot: removing any
//!   property before a tracked one bumps the generation too, where SpiderMonkey's slots stay put.
//!   The bumped generation fails every guard that named a slot.
//! - The guard a site checks ([`ObjectFuse::guard`], SpiderMonkey's GuardObjectFuseProperty):
//!   `invalidated` is 0, or the generation is the one the site recorded and the property's
//!   NotConstant bit is clear.
//! - Index keys are not tracked (SpiderMonkey's `tracksPropertyKey`).
//!
//! The objects' maps report their writes ([`note_value_change`], [`note_remove`], [`note_clear`])
//! for marked prototypes (`Props::proto_flag`): a fused object is marked as its fuse is made, and
//! a filter of the fused maps' addresses rejects nearly every other map with one load.
//!
//! String.prototype's [[SetPrototypeOf]] needs no hook. A primitive string's own properties are its
//! indices and `length` (String exotic objects' [[GetOwnProperty]]), so a method's name is never
//! one of them and the lookup's first hop is String.prototype, whose own property shadows
//! everything above it: changing String.prototype's [[Prototype]] cannot change what that property
//! holds. Removing the property is a removal, which the fuse sees.

// Generated code is the fuse's consumer (candidate 4's next step); until it lands, the tests read
// the guard's pieces.
#![cfg_attr(not(test), allow(dead_code))]

use std::cell::Cell;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// A property's state, two bits of [`ObjectFuse`]'s state words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum PropertyState {
    /// Initial: no generated code relies on the property.
    Untracked = 0,
    /// Generated code may rely on the property's value.
    Constant = 1,
    /// Written after it was Constant: no longer tracked. A guard tests its upper bit alone
    /// ([`GuardData::mask`]), which no other state has.
    NotConstant = 2,
}

const PROPS_PER_WORD: usize = 16;
const BITS_PER_PROP: usize = 2;
const STATE_MASK: u32 = (1 << BITS_PER_PROP) - 1;
/// The generation no guard accepts: a saturated counter stops the fuse for good (SpiderMonkey's
/// SaturatedGenerationCounter).
const GENERATION_INVALID: u32 = u32::MAX;

/// What a site relies on, recorded as it compiles (SpiderMonkey's GuardData): the fuse's generation
/// then, and the word and bit of the property's NotConstant state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GuardData {
    pub(crate) generation: u32,
    pub(crate) index: u32,
    pub(crate) mask: u32,
}

/// The constant properties of one object (see the module's documentation).
pub(crate) struct ObjectFuse {
    /// 1 once a Constant property became NotConstant or the generation was bumped (SpiderMonkey's
    /// invalidatedConstantProperty_): while it is 0, a guard needs nothing else. Generated code
    /// reads it ([`ObjectFuse::INVALIDATED_OFF`]).
    invalidated: AtomicU32,
    /// Bumped by the removal of a tracked property; [`GENERATION_INVALID`] once saturated.
    generation: AtomicU32,
    /// The state words (two bits a slot), written under the lock; their buffer's address is
    /// published in `words` for generated code, which reads a site's word there. The length never
    /// shrinks: a site's word, in range when it compiled, stays in range.
    states: Mutex<Vec<u32>>,
    words: AtomicPtr<u32>,
    /// The fused object's map, the registry's key.
    props: usize,
}

impl ObjectFuse {
    /// The offset of `invalidated`, the word generated code tests first.
    pub(crate) const INVALIDATED_OFF: usize = std::mem::offset_of!(ObjectFuse, invalidated);
    /// The offset of `generation`.
    pub(crate) const GENERATION_OFF: usize = std::mem::offset_of!(ObjectFuse, generation);
    /// The offset of `words`, the state words' published address.
    pub(crate) const WORDS_OFF: usize = std::mem::offset_of!(ObjectFuse, words);

    /// A fuse for the map at `props`, registered so that its writes reach it ([`note_value_change`]
    /// and the others). The caller marks the object a prototype (`Props::mark_proto`), which makes
    /// its map report them.
    pub(crate) fn new_registered(props: usize) -> Arc<Self> {
        let fuse = Arc::new(Self {
            invalidated: AtomicU32::new(0),
            generation: AtomicU32::new(0),
            states: Mutex::new(Vec::new()),
            words: AtomicPtr::new(std::ptr::null_mut()),
            props,
        });
        registry_insert(props, &fuse);
        fuse
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<u32>> {
        self.states.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn state_in(states: &[u32], slot: usize) -> PropertyState {
        let Some(word) = states.get(slot / PROPS_PER_WORD) else {
            return PropertyState::Untracked;
        };
        match (word >> shift(slot)) & STATE_MASK {
            0 => PropertyState::Untracked,
            1 => PropertyState::Constant,
            _ => PropertyState::NotConstant,
        }
    }

    fn set_state_in(&self, states: &mut Vec<u32>, slot: usize, state: PropertyState) {
        let index = slot / PROPS_PER_WORD;
        if index >= states.len() {
            if state == PropertyState::Untracked {
                return;
            }
            states.resize(index + 1, 0);
            self.words.store(states.as_mut_ptr(), Ordering::Release);
        }
        let shift = shift(slot);
        states[index] = (states[index] & !(STATE_MASK << shift)) | ((state as u32) << shift);
    }

    /// The state of the property at `slot`.
    pub(crate) fn state(&self, slot: usize) -> PropertyState {
        Self::state_in(&self.lock(), slot)
    }

    /// Whether a guard's fast path holds: no Constant property was invalidated and the generation
    /// never moved.
    pub(crate) fn intact(&self) -> bool {
        self.invalidated.load(Ordering::Relaxed) == 0
    }

    /// The current generation, [`GENERATION_INVALID`] once saturated.
    pub(crate) fn generation(&self) -> u32 {
        self.generation.load(Ordering::Relaxed)
    }

    fn bump_generation(&self) {
        self.invalidated.store(1, Ordering::Relaxed);
        let generation = self.generation.load(Ordering::Relaxed);
        if generation != GENERATION_INVALID {
            self.generation.store(generation + 1, Ordering::Relaxed);
        }
    }

    /// A site is about to rely on the property at `slot` (SpiderMonkey's
    /// tryOptimizeConstantProperty): an Untracked property becomes Constant, a Constant one stays;
    /// `None` for a NotConstant property or a saturated generation. The guard data is what the
    /// site checks.
    pub(crate) fn try_optimize_constant(&self, slot: usize) -> Option<GuardData> {
        let generation = self.generation();
        if generation == GENERATION_INVALID {
            return None;
        }
        let mut states = self.lock();
        match Self::state_in(&states, slot) {
            PropertyState::Untracked => {
                self.set_state_in(&mut states, slot, PropertyState::Constant)
            }
            PropertyState::Constant => {}
            PropertyState::NotConstant => return None,
        }
        Some(GuardData {
            generation,
            index: (slot / PROPS_PER_WORD) as u32,
            mask: 0b10 << shift(slot),
        })
    }

    /// The guard of a site that recorded `data` (SpiderMonkey's GuardObjectFuseProperty, which
    /// generated code repeats): the fast word, or the recorded generation with the property's
    /// NotConstant bit clear.
    pub(crate) fn guard(&self, data: GuardData) -> bool {
        if self.intact() {
            return true;
        }
        if self.generation() != data.generation || data.generation == GENERATION_INVALID {
            return false;
        }
        let states = self.lock();
        states
            .get(data.index as usize)
            .is_some_and(|word| word & data.mask == 0)
    }

    /// A write of the value at `slot`, or a define there that is not redundant (SpiderMonkey's
    /// handlePropertyValueChange).
    pub(crate) fn handle_value_change(&self, slot: usize) {
        let mut states = self.lock();
        match Self::state_in(&states, slot) {
            PropertyState::Untracked => {
                self.set_state_in(&mut states, slot, PropertyState::Constant)
            }
            PropertyState::Constant => {
                self.invalidated.store(1, Ordering::Relaxed);
                self.set_state_in(&mut states, slot, PropertyState::NotConstant);
            }
            PropertyState::NotConstant => {}
        }
    }

    /// The property at `slot` is removed, the map's later slots moving down one (SpiderMonkey's
    /// handlePropertyRemove and finishPropertyRemove, with this engine's compaction): removing a
    /// tracked property, or one before a tracked property, whose slot changes, bumps the
    /// generation.
    pub(crate) fn handle_remove(&self, slot: usize) {
        let mut states = self.lock();
        let len = states.len() * PROPS_PER_WORD;
        if (slot..len).any(|at| Self::state_in(&states, at) != PropertyState::Untracked) {
            self.bump_generation();
        }
        for at in slot..len.saturating_sub(1) {
            let next = Self::state_in(&states, at + 1);
            self.set_state_in(&mut states, at, next);
        }
        if len != 0 {
            self.set_state_in(&mut states, len - 1, PropertyState::Untracked);
        }
    }

    /// Every property is removed.
    pub(crate) fn handle_clear(&self) {
        let mut states = self.lock();
        if states.iter().any(|&word| word != 0) {
            self.bump_generation();
        }
        states.iter_mut().for_each(|word| *word = 0);
    }
}

impl Drop for ObjectFuse {
    fn drop(&mut self) {
        registry_remove(self.props, self);
    }
}

fn shift(slot: usize) -> u32 {
    ((slot % PROPS_PER_WORD) * BITS_PER_PROP) as u32
}

/// The fuses by their maps' addresses; a fuse removes itself as it drops.
fn registry() -> &'static Mutex<crate::fasthash::FastMap<usize, Weak<ObjectFuse>>> {
    static REGISTRY: OnceLock<Mutex<crate::fasthash::FastMap<usize, Weak<ObjectFuse>>>> =
        OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

/// A filter of the registered maps' addresses, a bit for each (4,096 bits): a map whose bit is
/// clear has no fuse, which one load tells, so that no other marked prototype's write takes the
/// registry's lock but about one in 4,096 for each fuse registered. Bits are added as fuses
/// register and recomputed from the live ones as a fuse leaves, both under the registry's lock: a
/// live fuse's bit is never cleared, whatever another Agent's thread reads meanwhile.
const FILTER_WORDS: usize = 64;
static FILTER: [AtomicU64; FILTER_WORDS] = [const { AtomicU64::new(0) }; FILTER_WORDS];

/// The filter's word and bit for the map at `props`.
fn filter_bit(props: usize) -> (usize, u64) {
    let hash = (props as u64 >> 4).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 52;
    ((hash >> 6) as usize % FILTER_WORDS, 1 << (hash & 63))
}

fn registry_insert(props: usize, fuse: &Arc<ObjectFuse>) {
    let mut map = registry().lock().unwrap_or_else(|e| e.into_inner());
    map.insert(props, Arc::downgrade(fuse));
    let (word, bit) = filter_bit(props);
    FILTER[word].fetch_or(bit, Ordering::Release);
}

fn registry_remove(props: usize, fuse: &ObjectFuse) {
    let mut map = registry().lock().unwrap_or_else(|e| e.into_inner());
    // A newer fuse may hold the address already (the old object died and a new map took its
    // place before this fuse dropped): leave it.
    if map
        .get(&props)
        .is_some_and(|weak| std::ptr::eq(weak.as_ptr(), fuse))
    {
        map.remove(&props);
    }
    let mut filter = [0u64; FILTER_WORDS];
    for &props in map.keys() {
        let (word, bit) = filter_bit(props);
        filter[word] |= bit;
    }
    for (word, bits) in FILTER.iter().zip(filter) {
        word.store(bits, Ordering::Release);
    }
}

fn fuse_of(props: usize) -> Option<Arc<ObjectFuse>> {
    registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&props)
        .and_then(Weak::upgrade)
}

/// Whether the map at `props` may have a fuse (its filter bit is set).
#[inline]
fn may_have_fuse(props: usize) -> bool {
    let (word, bit) = filter_bit(props);
    FILTER[word].load(Ordering::Acquire) & bit != 0
}

thread_local! {
    /// The map whose attributes alone [`with_flags_only`] is changing (0: none).
    static FLAGS_ONLY: Cell<usize> = const { Cell::new(0) };
}

/// Run `body`, in which defines on the map at `props` that keep a property's value (or accessors)
/// and change only its attributes are not value changes: freezing and sealing, which change flags
/// only (SpiderMonkey's freezeOrSealProperties leaves the fuse alone; a defineProperty that only
/// changes attributes does not). Only for an object whose defines run no author code.
pub(crate) fn with_flags_only<R>(props: usize, body: impl FnOnce() -> R) -> R {
    struct Restore(usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            FLAGS_ONLY.with(|c| c.set(self.0));
        }
    }
    let _restore = Restore(FLAGS_ONLY.with(|c| c.replace(props)));
    body()
}

/// A marked prototype's map at `props` wrote the value at `slot`: a value change, the same value
/// included (`Props::set_value`, `Props::set_value_at`).
#[inline]
pub(crate) fn note_value_change(props: usize, slot: usize) {
    if may_have_fuse(props) {
        value_change_slow(props, slot);
    }
}

#[cold]
#[inline(never)]
fn value_change_slow(props: usize, slot: usize) {
    if let Some(fuse) = fuse_of(props) {
        fuse.handle_value_change(slot);
    }
}

/// A marked prototype's map at `props` redefined the property at `slot`: `redundant` when nothing
/// changed (SpiderMonkey's DefinePropertyIsRedundant), `same_value` when the value (or the
/// accessors) stayed and only attributes changed.
#[inline]
pub(crate) fn note_redefine(props: usize, slot: usize, redundant: bool, same_value: bool) {
    if !may_have_fuse(props) || redundant {
        return;
    }
    if same_value && FLAGS_ONLY.with(Cell::get) == props {
        return;
    }
    value_change_slow(props, slot);
}

/// A marked prototype's map at `props` removed the property at `slot`, its later slots moving
/// down one.
#[inline]
pub(crate) fn note_remove(props: usize, slot: usize) {
    if may_have_fuse(props) {
        remove_slow(props, slot);
    }
}

#[cold]
#[inline(never)]
fn remove_slow(props: usize, slot: usize) {
    if let Some(fuse) = fuse_of(props) {
        fuse.handle_remove(slot);
    }
}

/// A marked prototype's map at `props` removed every property.
#[inline]
pub(crate) fn note_clear(props: usize) {
    if may_have_fuse(props) {
        if let Some(fuse) = fuse_of(props) {
            fuse.handle_clear();
        }
    }
}
