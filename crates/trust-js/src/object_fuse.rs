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

// Generated code is the fuse's consumer, the ARM64 JIT's fused String method calls
// (`jit::string_calls`); elsewhere only the tests read the guard's pieces.
#![cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    allow(dead_code)
)]

use std::cell::{Cell, RefCell};
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
///
/// A property's state is read without a lock ([`ObjectFuse::state`]): a couple of loads, which
/// the hot paths that ask "may this slot be stored in line" take (F4 part 2). The lock orders the
/// transitions (a state moving, the generation, a buffer growing). The read is sound because:
/// - every state word is an atomic, written only under the lock;
/// - a state buffer is published (`words`, Release) only after its words are written, and a
///   reader loads the address with Acquire and reads no word past the count the buffer's own
///   word 0 holds;
/// - no buffer the fuse published is freed while it lives (`buffers` keeps them), so an address
///   a reader loaded stays in bounds, whatever another thread does meanwhile.
///
/// A fuse's states are written on the thread of its object's Agent (the object's writes and the
/// fills of its caches run there), so a read there sees the current state. The optimizing tier's
/// compile thread reads no state: the snapshot and the install run on the engine's thread. A
/// reader on another thread would see a state that was current at some point: memory-safe, and,
/// since a state only moves Untracked -> Constant -> NotConstant while its property lives, never
/// ahead of the truth.
pub(crate) struct ObjectFuse {
    /// 1 once a Constant property became NotConstant or the generation was bumped (SpiderMonkey's
    /// invalidatedConstantProperty_): while it is 0, a guard needs nothing else. Generated code
    /// reads it ([`ObjectFuse::INVALIDATED_OFF`]).
    invalidated: AtomicU32,
    /// Bumped by the removal of a tracked property; [`GENERATION_INVALID`] once saturated.
    generation: AtomicU32,
    /// The lock of transitions, and what it guards ([`Locked`]): every state buffer the fuse
    /// published and the dependent codes of its Constant properties.
    locked: Mutex<Locked>,
    /// The current buffer's address ([`NO_STATES`] until a property is tracked), which generated
    /// code reads a site's word through (`jit::string_calls`, the word index its
    /// [`GuardData::index`] names). A buffer never shrinks: a site's word, in range when it
    /// compiled, stays in range.
    words: AtomicPtr<AtomicU32>,
    /// The fused object's map, the registry's key.
    props: usize,
}

/// What an [`ObjectFuse`]'s lock guards.
#[derive(Default)]
struct Locked {
    /// Every state buffer the fuse published, the current one last (see the type's
    /// documentation). A buffer's word 0 counts the state words after it, each holding sixteen
    /// properties' two bits; it grows by doubling, so the retired buffers hold fewer words than
    /// the current one.
    buffers: Vec<Box<[AtomicU32]>>,
    /// By slot, the validity words of the installed codes that folded the Constant property's
    /// value (F4 part 2, layer 2; SpiderMonkey's DepMap of DependentIonScriptSet,
    /// vm/ObjectFuse.h:121-125, V8's DependentCode list): the property's change, removal or
    /// shadowing sets each to 1 and clears the list ([`Locked::fire`]). A code holds its word
    /// for its life (its exit table's `valid`); an entry whose count is 1, its code freed, is
    /// pruned at the list's next insertion, so nothing dangles and nothing unregisters.
    dependents: crate::fasthash::FastMap<usize, Vec<Arc<AtomicU32>>>,
}

impl Locked {
    /// Invalidate every code on the list of the property at `slot` and clear the list: each
    /// live word set to 1, which the code tests before its folded reads (`CheckCodeValid`).
    /// Relaxed: the codes run on this fuse's Agent's thread, which writes the word.
    fn fire(&mut self, slot: usize) {
        if let Some(list) = self.dependents.remove(&slot) {
            fired(&list);
        }
    }

    /// [`Locked::fire`] for every property.
    fn fire_all(&mut self) {
        for (_, list) in self.dependents.drain() {
            fired(&list);
        }
    }
}

/// Set each live word of a fired list (see [`Locked::fire`]); the census counts the codes it
/// invalidated, not those already invalid (a discarded code stays on its other properties'
/// lists until it is freed).
fn fired(list: &[Arc<AtomicU32>]) {
    let mut codes = 0;
    for word in list {
        if Arc::strong_count(word) > 1 && word.swap(1, Ordering::Relaxed) == 0 {
            codes += 1;
        }
    }
    if codes > 0 {
        GLOBAL_INVALIDATIONS.fetch_add(1, Ordering::Relaxed);
        GLOBAL_INVALIDATED_CODES.fetch_add(codes, Ordering::Relaxed);
        #[cfg(test)]
        TEST_FIRED.with(|fired| fired.set((fired.get().0 + 1, fired.get().1 + codes)));
    }
}

/// Lists fired that held a live code, and the codes they invalidated (the census, F4 part 2).
static GLOBAL_INVALIDATIONS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_INVALIDATED_CODES: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    /// This thread's fired lists that held a live code, and the codes invalidated (tests).
    pub(crate) static TEST_FIRED: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

/// The state buffer of a fuse that tracks no property yet: a count of 0 words, never written.
static NO_STATES: AtomicU32 = AtomicU32::new(0);

impl ObjectFuse {
    /// The offset of `invalidated`, the word generated code tests first.
    pub(crate) const INVALIDATED_OFF: usize = std::mem::offset_of!(ObjectFuse, invalidated);
    /// The offset of `generation`.
    pub(crate) const GENERATION_OFF: usize = std::mem::offset_of!(ObjectFuse, generation);
    /// The map the fuse watches (the registry's key).
    pub(crate) fn props(&self) -> usize {
        self.props
    }

    /// The offset of `words`, the state words' published address.
    pub(crate) const WORDS_OFF: usize = std::mem::offset_of!(ObjectFuse, words);

    /// A fuse for the map at `props`, registered so that its writes reach it ([`note_value_change`]
    /// and the others). The caller marks the object a prototype (`Props::mark_proto`), which makes
    /// its map report them.
    pub(crate) fn new_registered(props: usize) -> Arc<Self> {
        let fuse = Arc::new(Self {
            invalidated: AtomicU32::new(0),
            generation: AtomicU32::new(0),
            locked: Mutex::new(Locked::default()),
            words: AtomicPtr::new(std::ptr::from_ref(&NO_STATES).cast_mut()),
            props,
        });
        registry_insert(props, &fuse);
        fuse
    }

    /// The lock of transitions (see the type's documentation).
    fn lock(&self) -> std::sync::MutexGuard<'_, Locked> {
        self.locked.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The state of the property at `slot`, read without the lock: the published buffer's
    /// address, its count and the slot's word (see the type's documentation).
    #[inline]
    pub(crate) fn state(&self, slot: usize) -> PropertyState {
        let words = self.words.load(Ordering::Acquire);
        let index = slot / PROPS_PER_WORD;
        // SAFETY: `words` is NO_STATES or a buffer this fuse published after writing it, which
        // stays allocated while the fuse lives; its word 0 counts the words after it.
        let count = unsafe { (*words).load(Ordering::Relaxed) } as usize;
        if index >= count {
            return PropertyState::Untracked;
        }
        // SAFETY: as above, and `1 + index` is within the buffer's `1 + count` words.
        let word = unsafe { (*words.add(1 + index)).load(Ordering::Relaxed) };
        match (word >> shift(slot)) & STATE_MASK {
            0 => PropertyState::Untracked,
            1 => PropertyState::Constant,
            _ => PropertyState::NotConstant,
        }
    }

    /// Set the state of the property at `slot`, under the lock (`buffers` is its guard's): a
    /// slot past the buffer's words takes a buffer twice as large, published once its words are
    /// written, the old one kept.
    fn set_state(&self, buffers: &mut Vec<Box<[AtomicU32]>>, slot: usize, state: PropertyState) {
        let index = slot / PROPS_PER_WORD;
        let mut words = self.words.load(Ordering::Relaxed);
        // SAFETY: as in `state`; under the lock no other buffer is published meanwhile.
        let count = unsafe { (*words).load(Ordering::Relaxed) } as usize;
        if index >= count {
            if state == PropertyState::Untracked {
                return;
            }
            let grown = (index + 1).max(2 * count).max(4);
            let old = words;
            buffers.push(
                (0..=grown)
                    .map(|at| {
                        AtomicU32::new(match at {
                            0 => grown as u32,
                            // SAFETY: the old buffer's words 1..=count.
                            at if at <= count => unsafe { (*old.add(at)).load(Ordering::Relaxed) },
                            _ => 0,
                        })
                    })
                    .collect(),
            );
            words = buffers
                .last()
                .expect("the buffer just made")
                .as_ptr()
                .cast_mut();
            self.words.store(words, Ordering::Release);
        }
        // SAFETY: `words` is this fuse's current buffer, `1 + index` within its words.
        let cell = unsafe { &*words.add(1 + index) };
        let shift = shift(slot);
        cell.store(
            (cell.load(Ordering::Relaxed) & !(STATE_MASK << shift)) | ((state as u32) << shift),
            Ordering::Relaxed,
        );
    }

    /// How many state buffers the fuse has published, every one kept (tests).
    #[cfg(test)]
    pub(crate) fn published_buffers(&self) -> usize {
        self.lock().buffers.len()
    }

    /// The number of slots the current buffer holds states for.
    fn tracked_len(&self) -> usize {
        // SAFETY: as in `state`.
        let count = unsafe { (*self.words.load(Ordering::Acquire)).load(Ordering::Relaxed) };
        count as usize * PROPS_PER_WORD
    }

    /// Whether a cache may let generated code or the VM store to the property at `slot` without
    /// reporting the write (SpiderMonkey's canOptimizeSetSlot, vm/ObjectFuse.h:222-224, which
    /// attaches SetProp stubs only for NotConstant properties): only a NotConstant property's
    /// value is relied on by no compile, now or later, since NotConstant is terminal while the
    /// property lives and a removal gives its map a fresh shape, which every cache proves. A
    /// refused store reports its write, which moves the property on (Untracked -> Constant ->
    /// NotConstant), so a property stored in a loop reaches the in-line store at its third
    /// store at the latest.
    #[inline]
    pub(crate) fn may_store_in_line(&self, slot: usize) -> bool {
        self.state(slot) == PropertyState::NotConstant
    }

    /// A script's lexical declaration shadows the property at `slot` (SpiderMonkey's
    /// handleShadowedGlobalProperty, vm/ObjectFuse.cpp:182-191): a read of the name no longer
    /// reaches the property, so a tracked property's guards fail. The state is kept.
    pub(crate) fn handle_shadowed(&self, slot: usize) {
        if self.state(slot) != PropertyState::Untracked {
            let mut locked = self.lock();
            self.bump_generation();
            locked.fire(slot);
        }
    }

    /// Every property may have changed (the map replaced whole, or the object no longer
    /// ordinary): a Constant property becomes NotConstant and the generation moves
    /// (SpiderMonkey's invalidateAllDependentIonScripts with a generation bump, as its
    /// handleGlobalLexicalMutation, vm/ObjectFuse.cpp:177-180).
    pub(crate) fn handle_all_changed(&self) {
        let mut locked = self.lock();
        for slot in 0..self.tracked_len() {
            if self.state(slot) == PropertyState::Constant {
                self.set_state(&mut locked.buffers, slot, PropertyState::NotConstant);
            }
        }
        self.bump_generation();
        locked.fire_all();
    }

    /// Whether a guard's fast path holds: no Constant property was invalidated and the generation
    /// never moved (tests; generated code reads the word at [`ObjectFuse::INVALIDATED_OFF`]).
    #[cfg(test)]
    pub(crate) fn intact(&self) -> bool {
        self.invalidated.load(Ordering::Relaxed) == 0
    }

    /// The current generation, [`GENERATION_INVALID`] once saturated.
    pub(crate) fn generation(&self) -> u32 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Under the lock.
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
        let mut locked = self.lock();
        match self.state(slot) {
            PropertyState::Untracked => {
                self.set_state(&mut locked.buffers, slot, PropertyState::Constant)
            }
            PropertyState::Constant => {}
            PropertyState::NotConstant => return None,
        }
        Some(GuardData {
            generation,
            // The word's index in the published buffer, past its count.
            index: (1 + slot / PROPS_PER_WORD) as u32,
            mask: 0b10 << shift(slot),
        })
    }

    /// The guard of a site that recorded `data` (SpiderMonkey's GuardObjectFuseProperty, which
    /// generated code repeats, `jit::string_calls`): the fast word, or the recorded generation with
    /// the property's NotConstant bit clear (tests).
    #[cfg(test)]
    pub(crate) fn guard(&self, data: GuardData) -> bool {
        if self.intact() {
            return true;
        }
        if self.generation() != data.generation || data.generation == GENERATION_INVALID {
            return false;
        }
        let words = self.words.load(Ordering::Acquire);
        // SAFETY: as in `state`; a site's index is within the buffer it was recorded under,
        // which never shrinks.
        unsafe { (*words.add(data.index as usize)).load(Ordering::Relaxed) & data.mask == 0 }
    }

    /// Whether the property at `slot` is Constant under `generation`, the one a compile's fold
    /// recorded ([`ObjectFuse::try_optimize_constant`]): the install's check (F4 part 2, layer 2;
    /// V8 drops a job whose dependencies fail, maglev/maglev-compiler.cc:359-368).
    pub(crate) fn still_constant(&self, slot: usize, generation: u32) -> bool {
        self.generation() == generation && self.state(slot) == PropertyState::Constant
    }

    /// Put `word`, an installed code's validity word, on the list of the property at `slot`,
    /// whose value the code folded under `generation` (SpiderMonkey's addDependency of an
    /// IonScript to its DepMap, V8's DependentCode::InstallDependency): false, and nothing put,
    /// where the property is no longer Constant under it, which the check and the insertion test
    /// under the lock together. The list's entries whose code was freed go first.
    pub(crate) fn add_dependent(
        &self,
        slot: usize,
        generation: u32,
        word: &Arc<AtomicU32>,
    ) -> bool {
        let mut locked = self.lock();
        if !self.still_constant(slot, generation) {
            return false;
        }
        let list = locked.dependents.entry(slot).or_default();
        list.retain(|word| Arc::strong_count(word) > 1);
        list.push(word.clone());
        true
    }

    /// The entries on the list of the property at `slot`, a freed code's among them until the
    /// list's next insertion prunes it (tests).
    #[cfg(test)]
    pub(crate) fn dependents(&self, slot: usize) -> usize {
        self.lock().dependents.get(&slot).map_or(0, Vec::len)
    }

    /// A write of the value at `slot`, or a define there that is not redundant (SpiderMonkey's
    /// handlePropertyValueChange). A NotConstant property, which no write moves, takes no lock.
    pub(crate) fn handle_value_change(&self, slot: usize) -> PropertyState {
        if self.state(slot) == PropertyState::NotConstant {
            return PropertyState::NotConstant;
        }
        let mut locked = self.lock();
        let state = self.state(slot);
        match state {
            PropertyState::Untracked => {
                self.set_state(&mut locked.buffers, slot, PropertyState::Constant)
            }
            PropertyState::Constant => {
                self.invalidated.store(1, Ordering::Relaxed);
                self.set_state(&mut locked.buffers, slot, PropertyState::NotConstant);
                locked.fire(slot);
            }
            PropertyState::NotConstant => {}
        }
        state
    }

    /// The property at `slot` is removed, the map's later slots moving down one (SpiderMonkey's
    /// handlePropertyRemove and finishPropertyRemove, with this engine's compaction): removing a
    /// tracked property, or one before a tracked property, whose slot changes, bumps the
    /// generation.
    pub(crate) fn handle_remove(&self, slot: usize) {
        let mut locked = self.lock();
        let len = self.tracked_len();
        if (slot..len).any(|at| self.state(at) != PropertyState::Untracked) {
            self.bump_generation();
        }
        for at in slot..len.saturating_sub(1) {
            let next = self.state(at + 1);
            self.set_state(&mut locked.buffers, at, next);
        }
        if len != 0 {
            self.set_state(&mut locked.buffers, len - 1, PropertyState::Untracked);
        }
        // The removed property's codes are invalidated; the later properties' lists move down
        // with their states (their values, which the codes folded, stay).
        locked.fire(slot);
        if locked.dependents.keys().any(|&at| at > slot) {
            locked.dependents = std::mem::take(&mut locked.dependents)
                .into_iter()
                .map(|(at, list)| (if at > slot { at - 1 } else { at }, list))
                .collect();
        }
    }

    /// Every property is removed (`Props::clear`, which only tests call).
    #[cfg(test)]
    pub(crate) fn handle_clear(&self) {
        let mut locked = self.lock();
        let len = self.tracked_len();
        if (0..len).any(|at| self.state(at) != PropertyState::Untracked) {
            self.bump_generation();
        }
        for at in 0..len {
            self.set_state(&mut locked.buffers, at, PropertyState::Untracked);
        }
        locked.fire_all();
    }
}

impl Drop for ObjectFuse {
    fn drop(&mut self) {
        // No Agent's heap names it any more: a heap that does holds an Arc of it
        // (`value::set_active_global`).
        registry_remove(self.props, self);
    }
}

#[cfg(test)]
thread_local! {
    /// This thread's watched-global value reports, by the path that found the fuse: the active
    /// realm's thread-local, or the registry (tests, R-5).
    pub(crate) static TEST_REPORT_PATHS: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

/// The map address of the active Agent's active global fuse, 0 for none (tests).
#[cfg(test)]
pub(crate) fn active_global_props() -> usize {
    crate::value::active_global_fuse().0
}

/// The fuse of the watched map at `props` when it is the active realm's global of the thread's
/// active Agent (F4 part 2, R-5: the Agent's heap holds it, `value::active_global_fuse`).
#[inline]
fn active_global_for(props: usize) -> Option<&'static ObjectFuse> {
    let (active, fuse) = crate::value::active_global_fuse();
    // SAFETY: the pointer names a fuse the active Agent's heap holds an Arc of
    // (`value::set_active_global`), which the report cannot drop.
    (active == props && !fuse.is_null()).then(|| unsafe { &*fuse })
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

/// A realm's fuses, held by the realm (`Interp::realm_fuses` and its record in `Interp::realms`):
/// String.prototype's and, once its built-ins are installed, the global object's (F4 part 2).
pub(crate) struct RealmFuses {
    pub(crate) string_proto: Option<Arc<ObjectFuse>>,
    pub(crate) global: Option<Arc<ObjectFuse>>,
}

/// The fuse registered for the map at `props`, if it lives (tests).
#[cfg(test)]
pub(crate) fn fuse_for(props: usize) -> Option<Arc<ObjectFuse>> {
    fuse_of(props)
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

/// The census of a watched global's reported writes (F4 part 2, R-4): value writes that reached
/// Rust (each a checked store: an in-line store reports nothing), those that made a Constant
/// property NotConstant, and of these the ones that stored the value it held (a2: the evidence
/// for V8's same-value rule); counted and printed at exit only under TRUST_JS_OPT_LOG
/// ([`census`], [`global_census_dump`]), so that a store's report counts nothing otherwise.
static GLOBAL_CHECKED_STORES: AtomicU64 = AtomicU64::new(0);
static GLOBAL_CHANGED_CONSTANTS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_SAME_VALUE_CHANGES: AtomicU64 = AtomicU64::new(0);
/// Fills of writing caches the rule refused (R2), and of these the Reference hints (a script's
/// top-level assignment of a global that is not yet NotConstant takes the checked path).
static GLOBAL_REFUSED_FILLS: AtomicU64 = AtomicU64::new(0);
static GLOBAL_REFUSED_HINTS: AtomicU64 = AtomicU64::new(0);

/// Whether the census counts: TRUST_JS_OPT_LOG, read once.
#[inline]
pub(crate) fn census() -> bool {
    env_flag!("TRUST_JS_OPT_LOG")
}

/// The census's count of checked stores, process-wide (tests).
#[cfg(test)]
pub(crate) fn census_checked_stores() -> u64 {
    GLOBAL_CHECKED_STORES.load(Ordering::Relaxed)
}

#[cfg(test)]
thread_local! {
    /// This thread's watched-global value writes that reached Rust (tests).
    pub(crate) static TEST_GLOBAL_CHECKED: Cell<u64> = const { Cell::new(0) };
}

/// A watched global's map at `props` wrote the value at `slot` (`Props::set_value`,
/// `Props::set_value_at`): a value change, as [`note_value_change`]'s; `same` when the value
/// stored is the one the slot held (computed only for the census).
#[inline]
pub(crate) fn note_global_value_change(props: usize, slot: usize, same: bool) {
    let census = census();
    if census {
        GLOBAL_CHECKED_STORES.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(test)]
    TEST_GLOBAL_CHECKED.with(|count| count.set(count.get() + 1));
    // The active realm's global: its fuse with no registry, and a NotConstant property with no
    // lock (`ObjectFuse::handle_value_change`).
    let state = match active_global_for(props) {
        Some(fuse) => {
            #[cfg(test)]
            TEST_REPORT_PATHS.with(|paths| paths.set((paths.get().0 + 1, paths.get().1)));
            fuse.handle_value_change(slot)
        }
        None => {
            #[cfg(test)]
            TEST_REPORT_PATHS.with(|paths| paths.set((paths.get().0, paths.get().1 + 1)));
            registered_value_change(props, slot)
        }
    };
    if census && state == PropertyState::Constant {
        GLOBAL_CHANGED_CONSTANTS.fetch_add(1, Ordering::Relaxed);
        if same {
            GLOBAL_SAME_VALUE_CHANGES.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cold]
#[inline(never)]
fn registered_value_change(props: usize, slot: usize) -> PropertyState {
    match may_have_fuse(props).then(|| fuse_of(props)).flatten() {
        Some(fuse) => fuse.handle_value_change(slot),
        None => PropertyState::Untracked,
    }
}

/// A watched global's map at `props` may have changed every property (its map replaced or
/// taken, or the object no longer ordinary).
pub(crate) fn note_all_changed(props: usize) {
    if may_have_fuse(props) {
        if let Some(fuse) = fuse_of(props) {
            fuse.handle_all_changed();
        }
    }
}

/// A script's lexical declaration shadows the watched global's property at `slot`
/// ([`ObjectFuse::handle_shadowed`]).
pub(crate) fn note_shadowed(props: usize, slot: usize) {
    if may_have_fuse(props) {
        if let Some(fuse) = fuse_of(props) {
            fuse.handle_shadowed(slot);
        }
    }
}

/// Whether a cache may store to the property at `slot` of the watched global whose map is at
/// `props` without reporting ([`ObjectFuse::may_store_in_line`]): through `active`, the active
/// realm's global fuse (`Interp::realm_fuses`), when it is that map's, with no registry and no
/// lock; else through the registry (another realm's global, which a store reaches rarely);
/// false where the map has no fuse, which a watched map always has while its realm lives.
#[inline]
pub(crate) fn global_may_store_in_line(
    active: Option<&ObjectFuse>,
    props: usize,
    slot: usize,
) -> bool {
    let may = match active {
        Some(fuse) if fuse.props == props => fuse.may_store_in_line(slot),
        _ => registered_may_store_in_line(props, slot),
    };
    if !may && census() {
        GLOBAL_REFUSED_FILLS.fetch_add(1, Ordering::Relaxed);
    }
    may
}

#[cold]
#[inline(never)]
fn registered_may_store_in_line(props: usize, slot: usize) -> bool {
    may_have_fuse(props) && fuse_of(props).is_some_and(|fuse| fuse.may_store_in_line(slot))
}

/// A Reference's hint refused by the rule (the census, [`global_census_dump`]).
pub(crate) fn note_refused_hint() {
    if census() {
        GLOBAL_REFUSED_HINTS.fetch_add(1, Ordering::Relaxed);
    }
}

/// The state of the watched global's property at `slot`, Untracked where the map has no fuse.
#[cfg(test)]
pub(crate) fn global_property_state(props: usize, slot: usize) -> PropertyState {
    fuse_of(props).map_or(PropertyState::Untracked, |fuse| fuse.state(slot))
}

/// Print the census of a watched global's writes under TRUST_JS_OPT_LOG (the shell's end of
/// run, `trust_js::unstable_feedback_census_dump`).
pub(crate) fn global_census_dump() {
    if census() {
        eprintln!(
            "[opt] global stores: checked {}, constants changed {}, same value {}, fills refused {}, reference hints refused {}; invalidations {} of {} codes",
            GLOBAL_CHECKED_STORES.load(Ordering::Relaxed),
            GLOBAL_CHANGED_CONSTANTS.load(Ordering::Relaxed),
            GLOBAL_SAME_VALUE_CHANGES.load(Ordering::Relaxed),
            GLOBAL_REFUSED_FILLS.load(Ordering::Relaxed),
            GLOBAL_REFUSED_HINTS.load(Ordering::Relaxed),
            GLOBAL_INVALIDATIONS.load(Ordering::Relaxed),
            GLOBAL_INVALIDATED_CODES.load(Ordering::Relaxed)
        );
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

/// A marked prototype's map at `props` removed every property (`Props::clear`, which only tests
/// call).
#[cfg(test)]
#[inline]
pub(crate) fn note_clear(props: usize) {
    if may_have_fuse(props) {
        if let Some(fuse) = fuse_of(props) {
            fuse.handle_clear();
        }
    }
}

/// A fused String method call's site (Djot's gap, candidate 4, step 2: `str.codePointAt(i)`
/// compiled as one step, `jit::string_calls`): the fuse it was validated against and the guard
/// data it relies on, which generated code reads at the offsets below (SpiderMonkey's
/// GuardObjectFuseProperty's operands, which its stub holds). Owned by its chunk
/// (`Chunk::jit_string_call_site`), as a call cache's way is. A chunk runs in whichever Realm is
/// active, so the step compares the active Realm's fuse with the site's, and a site validated in
/// another Realm, or never, takes the fill ([`StringCallSite::fill`]).
#[repr(C)]
pub(crate) struct StringCallSite {
    /// The address of the fuse the site was validated against, or that address | 1 when the
    /// method cannot be relied on there (the step then runs the unfused templates without asking
    /// again); 1 before the first fill, which no fuse's address matches.
    fuse: Cell<usize>,
    generation: Cell<u32>,
    index: Cell<u32>,
    mask: Cell<u32>,
    /// The method's call-cache class (`bytecode::INTRINSIC_CODE_POINT_AT`,
    /// `bytecode::INTRINSIC_CHAR_CODE_AT`).
    intrinsic: u8,
    /// Keeps the validated fuse's allocation, so that no later fuse takes its address while the
    /// site names it.
    pin: RefCell<Weak<ObjectFuse>>,
}

impl StringCallSite {
    /// The offsets generated code reads: the validated fuse's address, then the guard data.
    pub(crate) const FUSE_OFF: usize = std::mem::offset_of!(StringCallSite, fuse);
    pub(crate) const GENERATION_OFF: usize = std::mem::offset_of!(StringCallSite, generation);
    pub(crate) const INDEX_OFF: usize = std::mem::offset_of!(StringCallSite, index);
    pub(crate) const MASK_OFF: usize = std::mem::offset_of!(StringCallSite, mask);

    pub(crate) fn new(intrinsic: u8) -> Self {
        Self {
            fuse: Cell::new(1),
            generation: Cell::new(0),
            index: Cell::new(0),
            mask: Cell::new(0),
            intrinsic,
            pin: RefCell::new(Weak::new()),
        }
    }

    /// The method a site of call-cache class `intrinsic` calls.
    pub(crate) fn method_name(intrinsic: u8) -> Option<&'static str> {
        match intrinsic {
            crate::bytecode::INTRINSIC_CODE_POINT_AT => Some("codePointAt"),
            crate::bytecode::INTRINSIC_CHAR_CODE_AT => Some("charCodeAt"),
            _ => None,
        }
    }

    /// The step's fill, from its slow path: the site relies on the active Realm's
    /// String.prototype's own method when that is a data property holding a native of the
    /// site's class, which the property becoming Constant then keeps
    /// ([`ObjectFuse::try_optimize_constant`]); otherwise the site is refused in that Realm.
    /// The value is checked as the site fills: a method replaced while Untracked is Constant with
    /// its new value (SpiderMonkey's handlePropertyValueChange), and where a SpiderMonkey stub
    /// returns the value it recorded, the step runs the native's algorithm. Any Realm's native of
    /// the class will do: for a primitive String receiver and a Number index, the methods the
    /// classes name read nothing of their Realm. Whether the site now holds.
    pub(crate) fn fill(&self, interp: &crate::interpreter::Interp) -> bool {
        let Some(fuse) = interp.string_proto_fuse() else {
            return false;
        };
        let address = Arc::as_ptr(fuse) as usize;
        *self.pin.borrow_mut() = Arc::downgrade(fuse);
        match Self::relied_on(interp, fuse, self.intrinsic) {
            Some(data) => {
                self.generation.set(data.generation);
                self.index.set(data.index);
                self.mask.set(data.mask);
                self.fuse.set(address);
                true
            }
            None => {
                self.fuse.set(address | 1);
                false
            }
        }
    }

    fn relied_on(
        interp: &crate::interpreter::Interp,
        fuse: &ObjectFuse,
        intrinsic: u8,
    ) -> Option<GuardData> {
        let name = Self::method_name(intrinsic)?;
        let proto = interp.string_proto.borrow();
        let slot = proto.props.slot_of(name)?;
        let property = proto.props.property_at(slot)?;
        if property.accessor() {
            return None;
        }
        let crate::value::ValueView::Obj(function) = property.value().view() else {
            return None;
        };
        let native = match &function.borrow().kind {
            crate::value::ObjectKind::Native(native) => *native as usize,
            _ => return None,
        };
        if crate::builtins::builtin(native).1 != intrinsic {
            return None;
        }
        fuse.try_optimize_constant(slot)
    }
}

/// The fused String method step's fill (see [`StringCallSite::fill`]), called from its slow path
/// with the site's address; 1 when the site now holds, else 0. It runs no author code and
/// allocates nothing the collector sees.
///
/// # Safety
/// `ctx` is the running code's context and `site` a chunk's live site.
#[cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    allow(dead_code)
)]
pub(crate) unsafe extern "C" fn jit_string_call_fill(
    ctx: *mut crate::jit::JitCtx,
    site: *const StringCallSite,
) -> u32 {
    let interp = unsafe { &*(*ctx).interp };
    u32::from(unsafe { &*site }.fill(interp))
}

#[cfg(test)]
thread_local! {
    /// Fused calls x86-64's read finished (`jit_string_read`), for tests.
    pub(crate) static TEST_X64_FUSED_READS: Cell<usize> = const { Cell::new(0) };
}

/// x86-64's read of a fused String method call (`jit::string_calls`, step 4): with the call's
/// receiver and index on top of the operand stack, a primitive String and a Number that the
/// step's guards proved, the method's answer replaces them, String.prototype.codePointAt's or
/// charCodeAt's (`packed`, the call-cache class): ToIntegerOrInfinity of the Number, then the
/// code point (undefined past the end) or unit (NaN past the end). It runs no author code,
/// allocates nothing and cannot throw.
///
/// # Safety
/// `ctx` is the running code's context and `sp` its operand top, with the two operands below it.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) unsafe extern "C" fn jit_string_read(
    ctx: *mut crate::jit::JitCtx,
    packed: u32,
    sp: *mut crate::value::Value,
) -> crate::jit::SpFlag {
    use crate::value::{Value, ValueView};
    let interp = unsafe { &mut *(*ctx).interp };
    let base = unsafe { sp.sub(2) };
    let (receiver, index) = unsafe { (*base, *base.add(1)) };
    let (ValueView::Str(string), ValueView::Num(n)) = (receiver.view(), index.view()) else {
        unreachable!("the fused step's receiver and index guards")
    };
    let position = if n.is_nan() { 0.0 } else { n.trunc() };
    let at = (position >= 0.0 && position.is_finite()).then_some(position as usize);
    let value = match packed as u8 {
        crate::bytecode::INTRINSIC_CODE_POINT_AT => at
            .and_then(|at| interp.code_point_at(&string, at))
            .map_or(Value::Undefined, |point| Value::Num(f64::from(point))),
        crate::bytecode::INTRINSIC_CHAR_CODE_AT => Value::Num(
            at.and_then(|at| interp.unit_at(&string, at))
                .map_or(f64::NAN, f64::from),
        ),
        _ => unreachable!("a fused call's method has a read"),
    };
    #[cfg(test)]
    TEST_X64_FUSED_READS.with(|count| count.set(count.get() + 1));
    // The operands' words own nothing (the collector traces the stack): the answer replaces
    // them.
    unsafe { base.write(value) };
    crate::jit::SpFlag {
        sp: unsafe { base.add(1) },
        flag: 0,
    }
}
