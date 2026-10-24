//! P1 S2: the interpreter's half of a collection.
//!
//! Collections run only at interpreter safepoints — [`Interp::gc_check`], task boundaries,
//! buffer pressure and explicit requests — never on allocation. There the interpreter supplies
//! what the heap cannot see (Decision 2): its own fields, traced by an exhaustive
//! destructuring so that a new field does not compile until it is classified; the registered
//! VM frames and running coroutines (I1); owner-conditional side-table slots and Realm groups;
//! WeakMap ephemerons; and the graph its host reports. The native stack is scanned
//! conservatively (Decision 0).
//!
//! A ShadowRealm's interpreter shares its parent's heap: one heap per Agent, so no edge ever
//! crosses heaps (I6). Each interpreter can collect at its current safepoint, over its whole Agent, using scoped
//! current-borrow capabilities. The root's address is never saved between host entries. A realm's
//! roots belong to its ShadowRealm object, traced once that object is marked, so a realm kept
//! only by cycles through itself is collected; while code in a realm runs, its roots and its
//! ShadowRealm object are roots.
//!
//! Collections are minor or major (S6 step 10c, `gc_core::CollectionKind`). Allocation
//! safepoints and task boundaries collect minorly unless a major collection is due; explicit
//! requests (a host's, a test's, `$262.gc`, the idle and exit collections), ArrayBuffer
//! pressure and the live-object ceiling collect majorly. A minor collection differs in what the
//! heap traces of its old cells and which pages it sweeps, and the engine's tables lose only the
//! young dead. A major collection's client reports everything above in full; a minor one's
//! reports the roots, Realm groups and host graphs in full, but of the side-slot tables, the
//! WeakMap and WeakSet entries and the native functions' [[Realm]] edges only what was written
//! since the last collection (S6 step 10d, [`ClientRecords`]): an old entry that has not been
//! written since holds old values, which the last collection traced. The heap's rooted handles
//! (S6 step 11a, `handles::HandleTable`) follow the same rule: a major collection reports every
//! strong handle as a root and clears every weak handle whose target died, a minor one only
//! the handles written since the last collection; weak handles are cleared between the phases,
//! before the host sweeps and before any cell is freed.

use super::*;
use crate::fasthash::{FastMap, FastSet};
use crate::gc_core::{CellAge, CollectionKind};
use crate::gc_native::NativeGcGraph;
use crate::gc_trace::{GcSink, GcTrace};
use crate::host::{HostGcVisitor, NativeGcId};
use crate::value::GcState;
#[cfg(feature = "embed")]
use crate::value::ValueView;

/// The byte budget's floor. A major collection is due once the heap holds more than the budget
/// after a minor collection: its old cells, live or not, since minor collections free only young
/// ones. The budget is max(this floor, twice the bytes the last major collection left). It counts
/// every cell, environment records and strings as well as objects, and the live-object count
/// only enforces the engine's live-object limit (see `Interp::gc_next`). On the gate fixtures,
/// S2's tripwire measured peak RSS up to 6x higher at 64 MiB than at 8 MiB, with bytecode-tier
/// time unchanged (docs/v8-chase-2026-10-15.txt, S2 exit gate).
const GC_MIN_HEAP_BYTES: usize = 8 << 20;

/// The nursery budget: a minor collection is due once this many bytes of cells have been
/// allocated since the last collection (S6 step 10c).
const GC_NURSERY_BYTES: usize = 2 << 20;

/// Under GC stress, every this many collections the stress makes one major.
const GC_STRESS_MAJOR_EVERY: usize = 8;

/// Native stack zeroed below a collection's entry before marking (see `gc_core::scrub_stack`).
#[cfg(not(target_arch = "wasm32"))]
const GC_SCRUB_BYTES: usize = 256 * 1024;

/// Native stack zeroed before an explicit collection (see `Interp::gc_heap_collect_explicit`).
#[cfg(not(target_arch = "wasm32"))]
const EXPLICIT_SCRUB_BYTES: usize = 1 << 20;

/// `TRUST_JS_GC_STRESS=N`: collect at the first safepoint after every N cell allocations
/// (N = 1: at every safepoint that follows an allocation), and poll at every call.
pub(crate) fn gc_stress() -> usize {
    static STRESS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *STRESS.get_or_init(|| {
        std::env::var("TRUST_JS_GC_STRESS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    })
}

/// `TRUST_JS_GC_STRESS_MAJOR=M`: under GC stress, every Mth collection is major
/// ([`GC_STRESS_MAJOR_EVERY`] by default; 1 makes every collection major).
fn gc_stress_major() -> usize {
    static MAJOR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAJOR.get_or_init(|| {
        std::env::var("TRUST_JS_GC_STRESS_MAJOR")
            .ok()
            .and_then(|value| value.parse().ok())
            .filter(|&every| every > 0)
            .unwrap_or(GC_STRESS_MAJOR_EVERY)
    })
}

/// `TRUST_JS_GC_MIN_HEAP_BYTES=N` (diagnostic): the byte-budget floor instead of
/// [`GC_MIN_HEAP_BYTES`], for collector tuning.
fn gc_min_heap_bytes() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("TRUST_JS_GC_MIN_HEAP_BYTES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(GC_MIN_HEAP_BYTES)
    })
}

/// `TRUST_JS_GC_NURSERY_BYTES=N` (diagnostic): the nursery budget instead of
/// [`GC_NURSERY_BYTES`], for collector tuning; 0 turns minor collections off, so that every
/// collection is major and due once the allocated bytes pass the byte budget.
fn gc_nursery_bytes() -> Option<usize> {
    static NURSERY: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *NURSERY.get_or_init(|| {
        let bytes = std::env::var("TRUST_JS_GC_NURSERY_BYTES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(GC_NURSERY_BYTES);
        (bytes > 0).then_some(bytes)
    })
}

/// `TRUST_JS_GC_LOG=1` (diagnostic): log every collection, with its phases' times.
fn gc_log() -> bool {
    static LOG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LOG.get_or_init(|| std::env::var_os("TRUST_JS_GC_LOG").is_some())
}

/// The engine's own steps around the heap's phases, timed for `TRUST_JS_GC_LOG` (see
/// [`EngineStep`]).
const ENGINE_STEPS: usize = 6;

/// Steps of a collection the engine times for `TRUST_JS_GC_LOG`.
#[derive(Clone, Copy)]
enum EngineStep {
    /// Zeroing the native stack below the collection's frames.
    Scrub,
    /// Building the marking client (Realm members, hosts, templates).
    Client,
    /// Listing the dead objects for the tables' sweeps.
    Dead,
    /// Clearing weak handles and weak cells to the dead.
    Weak,
    /// Evicting the dead from the engine's tables.
    Tables,
    /// The Agent's bookkeeping after the sweep.
    After,
}

const ENGINE_STEP_NAMES: [&str; ENGINE_STEPS] =
    ["scrub", "client", "dead", "weak", "tables", "after"];

thread_local! {
    /// [`EngineStep`] times of the collection in progress, in nanoseconds.
    static ENGINE_STEP_NANOS: std::cell::Cell<[u64; ENGINE_STEPS]> =
        const { std::cell::Cell::new([0; ENGINE_STEPS]) };
}

/// Time `f` as `step` while `TRUST_JS_GC_LOG` is set.
#[inline]
fn timed<R>(step: EngineStep, f: impl FnOnce() -> R) -> R {
    if !gc_log() {
        return f();
    }
    let started = std::time::Instant::now();
    let result = f();
    let nanos = started.elapsed().as_nanos() as u64;
    ENGINE_STEP_NANOS.with(|steps| {
        let mut all = steps.get();
        all[step as usize] += nanos;
        steps.set(all);
    });
    result
}

/// Collection settings for a new Agent heap.
pub(crate) fn configure_heap(core: &mut crate::gc_core::Heap) {
    core.set_heap_growth(gc_min_heap_bytes(), 100);
    // `TRUST_JS_GC_LOG=1`: each collection's log line includes its phases' times.
    core.set_phase_timing(gc_log());
    core.set_nursery_bytes(gc_nursery_bytes());
    core.set_stress_every(gc_stress());
    core.set_stress_major_every(gc_stress_major());
    // `TRUST_JS_GC_WHY=1`: record retention paths, which only tests read (see `gc_test::why`).
    #[cfg(test)]
    if std::env::var_os("TRUST_JS_GC_WHY").is_some() {
        core.set_retention_tracing(true);
    }
}

struct WordSink<'a>(&'a mut dyn FnMut(u64));

impl GcSink for WordSink<'_> {
    #[inline]
    fn word(&mut self, word: u64) {
        (self.0)(word);
    }
}

fn value_word(value: &Value) -> Option<u64> {
    value.cell_word()
}

/// S6 step 10d: what the interpreters sharing a heap (a ShadowRealm tree) wrote since the last
/// collection's marking consumed it: the side-slot owners, WeakMap and WeakSet entries and
/// native functions' [[Realm]] registrations that a minor collection reports instead of whole
/// tables. The tables record into it as they are written ([`SideSlotTable`],
/// [`WeakCollectionData`], [`NativeFunctionRealms`]), each through the only methods that write
/// them. It lives in the heap's `GcState`, not beside each table, so that the tables keep the
/// size of the maps they wrap: generated code embeds `Interp`'s field offsets.
///
/// Each collection's marking consumes it once marked ([`ClientRecords::consume`]): what is
/// written from then on, between the phases and after the sweep included, is recorded for the
/// next collection. A deferred collection consumes nothing.
#[derive(Default)]
pub(crate) struct ClientRecords {
    /// The owners whose entries in a side-slot table were inserted or handed out mutably.
    side_slot_owners: FastSet<usize>,
    /// The WeakMap and WeakSet entries written, by (collection, key).
    weak_entries: FastSet<(usize, WeakKey)>,
    /// The native functions registered in `native_function_realms`, in order.
    native_functions: Vec<usize>,
    /// What `native_functions` held when the collection in progress consumed it: the functions
    /// its [[Realm]] sweep looks at in a minor collection (`Interp::gc_sweep_native_realms`).
    native_functions_swept: Vec<usize>,
}

impl ClientRecords {
    /// A newly installed native owner may already be old. Recording is unconditional, even
    /// during unpublished child bootstrap: publication of shadow_home happens before marking.
    pub(crate) fn remember_owner(&mut self, owner: usize) {
        self.side_slot_owners.insert(owner);
    }

    /// The collection in progress has marked: start recording for the next one, keeping the
    /// functions registered since the last collection for this one's [[Realm]] sweep.
    fn consume(&mut self) {
        fn clear<T: Eq + std::hash::Hash>(set: &mut FastSet<T>) {
            set.clear();
            if set.capacity() > 1024 {
                set.shrink_to(64);
            }
        }
        clear(&mut self.side_slot_owners);
        clear(&mut self.weak_entries);
        self.native_functions_swept.clear();
        std::mem::swap(&mut self.native_functions, &mut self.native_functions_swept);
    }

    /// Forget that `owner`'s side-slot entries were written (tests: a write the record missed).
    #[cfg(test)]
    pub(crate) fn forget_side_slot_owner(&mut self, owner: usize) {
        self.side_slot_owners.remove(&owner);
    }

    /// Forget that `owner`'s entry under `key` was written (tests).
    #[cfg(test)]
    pub(crate) fn forget_weak_entry(&mut self, owner: usize, key: WeakKey) {
        self.weak_entries.remove(&(owner, key));
    }

    /// Forget that `function` was registered since the last collection (tests).
    #[cfg(test)]
    pub(crate) fn forget_native_function(&mut self, function: usize) {
        self.native_functions
            .retain(|&registered| registered != function);
    }
}

/// An owner-keyed side-slot table (`gc_side_slot_tables!`, `Interp::generators`): an object's
/// address mapped to the strong edges it holds outside its cell, traced once the owner is
/// marked. Reads go through `Deref` to the map. Each method that inserts an entry or hands one
/// out mutably records its owner in the heap's [`ClientRecords`] (S6 step 10d), and nothing else
/// can write one; removing an entry adds no edge and records nothing. A minor collection
/// reports the recorded owners only: an old owner whose entries were not written holds old
/// values, which the last collection traced, and a young owner's entries were all inserted
/// since it was allocated. Debug builds check both as each collection starts
/// (`Interp::check_side_slot_records`), which also catches a write through a value's interior
/// mutability, out of the methods' sight.
#[repr(transparent)]
pub(crate) struct SideSlotTable<V, S = std::hash::BuildHasherDefault<crate::fasthash::FxHasher>>(
    std::collections::HashMap<usize, V, S>,
);

impl<V, S: Default> Default for SideSlotTable<V, S> {
    fn default() -> Self {
        Self(std::collections::HashMap::default())
    }
}

impl<V, S> std::ops::Deref for SideSlotTable<V, S> {
    type Target = std::collections::HashMap<usize, V, S>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V, S: std::hash::BuildHasher> SideSlotTable<V, S> {
    #[inline]
    pub(crate) fn insert(&mut self, heap: &GcState, owner: usize, value: V) -> Option<V> {
        heap.records.borrow_mut().side_slot_owners.insert(owner);
        self.0.insert(owner, value)
    }

    #[inline]
    pub(crate) fn get_mut(&mut self, heap: &GcState, owner: &usize) -> Option<&mut V> {
        let value = self.0.get_mut(owner)?;
        heap.records.borrow_mut().side_slot_owners.insert(*owner);
        Some(value)
    }

    #[inline]
    pub(crate) fn entry(
        &mut self,
        heap: &GcState,
        owner: usize,
    ) -> std::collections::hash_map::Entry<'_, usize, V> {
        heap.records.borrow_mut().side_slot_owners.insert(owner);
        self.0.entry(owner)
    }

    /// Every entry, mutably: every owner is recorded.
    pub(crate) fn values_mut(
        &mut self,
        heap: &GcState,
    ) -> std::collections::hash_map::ValuesMut<'_, usize, V> {
        heap.records
            .borrow_mut()
            .side_slot_owners
            .extend(self.0.keys().copied());
        self.0.values_mut()
    }

    #[inline]
    pub(crate) fn remove(&mut self, owner: &usize) -> Option<V> {
        self.0.remove(owner)
    }

    /// Evict the dead objects' entries (see [`crate::gc_sweep::map`]).
    pub(crate) fn evict_dead(&mut self, dead: &crate::gc_sweep::DeadObjects<'_>) {
        crate::gc_sweep::map(&mut self.0, dead);
    }

    /// Give back spare capacity (see [`crate::weak_metadata::shrink_map`]).
    pub(crate) fn shrink(&mut self) {
        crate::weak_metadata::shrink_map(&mut self.0);
    }
}

/// `Interp::weak_collection_data`: each WeakMap's and WeakSet's entries, (weakly held key,
/// value), by the collection's address. Reads go through `Deref` to the map. Each method that
/// writes an entry records it, by (collection, key), in the heap's [`ClientRecords`] (S6 step
/// 10d), and nothing else can write one; removing an entry records nothing. A minor collection
/// registers the recorded entries only as ephemerons: an entry not written since the last
/// collection whose key is old holds an old value (the last collection marked it, or found its
/// key or its collection dead and evicted the entry), and an entry whose key is young was
/// written since the key was allocated. Debug builds check both as each collection starts
/// (`Interp::check_weak_entry_records`).
#[repr(transparent)]
#[derive(Default)]
pub(crate) struct WeakCollectionData(FastMap<usize, Vec<(WeakTarget, Value)>>);

impl std::ops::Deref for WeakCollectionData {
    type Target = FastMap<usize, Vec<(WeakTarget, Value)>>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl WeakCollectionData {
    /// A collection's entries: a new collection's, or those moved to the instance a native base
    /// constructor grafts its slots onto. Each entry is recorded.
    pub(crate) fn insert(
        &mut self,
        heap: &GcState,
        owner: usize,
        entries: Vec<(WeakTarget, Value)>,
    ) -> Option<Vec<(WeakTarget, Value)>> {
        heap.records
            .borrow_mut()
            .weak_entries
            .extend(entries.iter().map(|(key, _)| (owner, key.key())));
        self.0.insert(owner, entries)
    }

    /// Replace the value of `owner`'s entry at `index`.
    pub(crate) fn set_value(&mut self, heap: &GcState, owner: usize, index: usize, value: Value) {
        let entry = &mut self
            .0
            .get_mut(&owner)
            .expect("a weak collection has backing data")[index];
        heap.records
            .borrow_mut()
            .weak_entries
            .insert((owner, entry.0.key()));
        entry.1 = value;
    }

    /// Append an entry to `owner`'s; returns its index.
    pub(crate) fn push(
        &mut self,
        heap: &GcState,
        owner: usize,
        key: WeakTarget,
        value: Value,
    ) -> usize {
        let entries = self
            .0
            .get_mut(&owner)
            .expect("a weak collection has backing data");
        heap.records
            .borrow_mut()
            .weak_entries
            .insert((owner, key.key()));
        entries.push((key, value));
        entries.len() - 1
    }

    /// Remove `owner`'s entry at `index`, the last entry taking its place (with its edges
    /// unchanged), and give back spare capacity.
    pub(crate) fn swap_remove(&mut self, owner: usize, index: usize) -> (WeakTarget, Value) {
        let entries = self
            .0
            .get_mut(&owner)
            .expect("weak collection backing data");
        let removed = entries.swap_remove(index);
        if entries.capacity() > entries.len().saturating_mul(4).max(32) {
            entries.shrink_to(entries.len().saturating_mul(2).max(16));
        }
        removed
    }

    pub(crate) fn remove(&mut self, owner: &usize) -> Option<Vec<(WeakTarget, Value)>> {
        self.0.remove(owner)
    }

    /// Give back spare capacity (see [`crate::weak_metadata::shrink_map`]).
    pub(crate) fn shrink(&mut self) {
        crate::weak_metadata::shrink_map(&mut self.0);
    }
}

/// `Interp::native_function_realms`: each built-in function's [[Realm]] (ECMA-262
/// CreateBuiltinFunction), by the function's address: a weak pin of the function and the address
/// of its Realm's global object. An entry never changes once registered
/// ([`NativeFunctionRealms::register`], which records the function in the heap's
/// [`ClientRecords`]; S6 step 10d). A minor collection reports the [[Realm]] edges of the
/// functions registered since the last collection only (an old function's Realm was marked
/// when the function was), and sweeps the table from them, the only functions it can find dead.
/// Debug builds check the first as each collection starts (`Interp::check_native_realm_records`).
#[repr(transparent)]
#[derive(Default)]
pub(crate) struct NativeFunctionRealms(FastMap<usize, (crate::value::WeakGc, usize)>);

impl std::ops::Deref for NativeFunctionRealms {
    type Target = FastMap<usize, (crate::value::WeakGc, usize)>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl NativeFunctionRealms {
    /// `function`'s [[Realm]] is the Realm whose global object is at `realm`.
    pub(crate) fn register(&mut self, heap: &GcState, function: &Gc, realm: usize) {
        let address = Gc::as_ptr(function) as usize;
        heap.records.borrow_mut().native_functions.push(address);
        self.0.insert(address, (Gc::downgrade(function), realm));
    }

    /// Keep the entries `keep` accepts (a major collection's sweep).
    fn retain(&mut self, mut keep: impl FnMut(&(crate::value::WeakGc, usize)) -> bool) {
        self.0.retain(|_, entry| keep(entry));
    }

    fn remove(&mut self, function: &usize) {
        self.0.remove(function);
    }

    /// Point `function`'s entry at another Realm without recording it (tests: a write the
    /// record missed).
    #[cfg(test)]
    pub(crate) fn set_realm_unrecorded(&mut self, function: usize, realm: usize) {
        if let Some(entry) = self.0.get_mut(&function) {
            entry.1 = realm;
        }
    }
}

/// What the client of a heap's last collection reported (test builds): the side-slot table
/// owners (`gc_side_slot_tables!` and `generators`; Realm members, ShadowRealm owners and host
/// owners apart), the WeakMap and WeakSet entries it registered as ephemerons, by (collection,
/// key's cell word), the native functions whose [[Realm]] edges it reported, and the entries of
/// the handle table whose values it reported as roots.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct ClientReport {
    pub(crate) kind: Option<CollectionKind>,
    pub(crate) side_slot_owners: Vec<usize>,
    pub(crate) weak_entries: Vec<(usize, u64)>,
    pub(crate) realm_edges: Vec<usize>,
    pub(crate) handle_slots: Vec<u32>,
}

/// One interpreter's host graph for this collection: what its registered host types report.
#[derive(Default)]
struct HostGraph {
    /// The collecting heap's identity: the weak handles a host reports are read for it alone.
    #[cfg_attr(not(feature = "embed"), allow(dead_code))]
    heap_id: u64,
    roots: Vec<u64>,
    /// Logical edges: each value is reachable while its owner is.
    edges: Vec<(usize, u64)>,
    native: NativeGcGraph,
}

impl HostGcVisitor for HostGraph {
    fn native_edge(&mut self, from: NativeGcId, to: NativeGcId) {
        self.native.native_edge(from, to);
    }

    #[cfg(feature = "embed")]
    fn js_to_native(&mut self, owner: &crate::handles::WeakHandle, to: NativeGcId) {
        if let Some(owner) = owner.reported(self.heap_id) {
            self.native.js_to_native(&owner, to);
        }
    }

    #[cfg(feature = "embed")]
    fn native_to_js(&mut self, from: NativeGcId, value: &crate::handles::WeakHandle) {
        if let Some(value) = value.reported(self.heap_id) {
            self.native.native_to_js(from, &value);
        }
    }

    fn native_root(&mut self, id: NativeGcId) {
        self.native.native_root(id);
    }

    #[cfg(feature = "embed")]
    fn edge(&mut self, owner: &crate::handles::WeakHandle, value: &crate::handles::WeakHandle) {
        if let (Some(ValueView::Obj(owner)), Some(value)) = (
            (owner.reported(self.heap_id)).map(|__value| (__value).view()),
            value.reported(self.heap_id).as_ref().and_then(value_word),
        ) {
            self.edges.push((Gc::as_ptr(&owner) as usize, value));
        }
    }

    #[cfg(feature = "embed")]
    fn root(&mut self, value: &crate::handles::WeakHandle) {
        if let Some(word) = value.reported(self.heap_id).as_ref().and_then(value_word) {
            self.roots.push(word);
        }
    }
}

/// The [`crate::gc_core::MarkClient`] of one collection over a ShadowRealm tree.
struct Marking<'a> {
    /// A minor collection's client reports what the tables recorded instead of the tables (see
    /// [`ClientRecords`]).
    kind: CollectionKind,
    /// Root first, then descendants in stable creation/topological order.
    interps: Vec<&'a Interp>,
    /// Parallel to `interps`.
    hosts: Vec<RefCell<HostGraph>>,
    /// ShadowRealm object -> the index of its realm's interpreter.
    shadow_owners: FastMap<usize, usize>,
    /// The realms with code running, as (ShadowRealm object, interpreter index): their roots and
    /// objects are roots. Listed here, before the stack below the collection is zeroed, so that
    /// [`MarkClient::roots`] handles no address that is not a root: the native stack scan that
    /// follows it reuses its frames, and an idle realm's object left in one would look like a
    /// root there.
    entered: Vec<(usize, usize)>,
    /// Realm-group members (objects and global environments) → (interpreter, Realm key). A
    /// Realm is not a root: marking any member activates the whole group, as in the old
    /// collector, so a retired Realm is reclaimed with everything it alone keeps alive.
    realm_members: FastMap<usize, Vec<(usize, usize)>>,
    /// GetTemplateObject's per-Realm [[TemplateMap]] values, by (interpreter, Realm key).
    templates: FastMap<(usize, usize), Vec<u64>>,
    activated_realms: RefCell<FastSet<(usize, usize)>>,
    activated_homes: RefCell<FastSet<usize>>,
}

impl<'a> Marking<'a> {
    /// Out of line: its frames must be dead (and scrubbed) before the native stack is scanned.
    #[inline(never)]
    fn new(root: &'a Interp, kind: CollectionKind) -> Self {
        let pointers = root
            .gc_heap
            .agent
            .access
            .shared_snapshot(root.realm_id, root);
        let mut interps = Vec::with_capacity(pointers.len());
        let mut shadow_owners = FastMap::default();
        let mut entered = Vec::new();
        for (id, pointer) in pointers {
            // SAFETY: the collection's current lease refreshes the requester; all other active
            // participants published their current borrow at suspension. Inactive allocations
            // have raw owners. No shared phase reference survives into sweep/retirement.
            let interp = unsafe { &*pointer };
            let index = interps.len();
            if let Some(home) = &interp.shadow_home {
                let owner = home
                    .owner
                    .upgrade()
                    .expect("a registered realm has a live owner cell");
                let owner = Gc::as_ptr(&owner) as usize;
                shadow_owners.insert(owner, index);
                if root.gc_heap.agent.access.active(id) {
                    entered.push((owner, index));
                }
            }
            interps.push(interp);
        }
        let mut hosts = Vec::with_capacity(interps.len());
        let mut realm_members: FastMap<usize, Vec<(usize, usize)>> = FastMap::default();
        let mut templates: FastMap<(usize, usize), Vec<u64>> = FastMap::default();
        for (index, interp) in interps.iter().enumerate() {
            let mut host = HostGraph {
                heap_id: crate::value::heap_id(&root.gc_heap),
                ..HostGraph::default()
            };
            interp.host_state.trace_gc(&mut host);
            hosts.push(RefCell::new(host));
            for (&key, realm) in &interp.realms {
                // A RealmState reports exactly its members.
                realm.trace_gc(&mut WordSink(&mut |member| {
                    realm_members
                        .entry(member as usize)
                        .or_default()
                        .push((index, key));
                }));
            }
            for (&(realm, _), value) in &interp.template_cache {
                if let Some(word) = value_word(value) {
                    templates.entry((index, realm)).or_default().push(word);
                }
            }
        }
        Marking {
            kind,
            interps,
            hosts,
            shadow_owners,
            entered,
            realm_members,
            templates,
            activated_realms: RefCell::default(),
            activated_homes: RefCell::default(),
        }
    }

    /// What the tree's tables recorded since the last collection (they share the root's heap).
    fn records(&self) -> std::cell::Ref<'_, ClientRecords> {
        self.interps[0].gc_heap.records.borrow()
    }

    /// Record part of what this collection's client reported (test builds).
    #[cfg(test)]
    fn report(&self, update: impl FnOnce(&mut ClientReport)) {
        let mut report = self.interps[0].gc_heap.client_report.borrow_mut();
        report.kind = Some(self.kind);
        update(&mut report);
    }

    /// Each interpreter's native graph, by interpreter address, for the host sweeps.
    fn into_native_graphs(self) -> FastMap<usize, NativeGcGraph> {
        self.interps
            .iter()
            .zip(self.hosts)
            .map(|(&interp, host)| {
                (
                    std::ptr::from_ref(interp) as usize,
                    host.into_inner().native,
                )
            })
            .collect()
    }
}

/// Reports a collection's ephemerons and side-table edges to the heap ([`Marking::ephemerons`]),
/// noting in test builds what it reported.
struct EdgeReporter<'s, 'e> {
    sink: &'s mut crate::gc_core::EdgeSink<'e>,
    #[cfg(test)]
    report: ClientReport,
}

impl EdgeReporter<'_, '_> {
    /// A WeakMap or WeakSet entry of `owner`: with an object or Symbol key, its value lives while
    /// both the collection and the key do.
    fn weak_entry(&mut self, owner: usize, key: &WeakTarget, value: &Value) {
        let Some(value) = value_word(value) else {
            return;
        };
        if let Some(key) = key.upgrade().as_ref().and_then(Value::cell_word) {
            self.sink.ephemeron(owner as *const u8, key, value);
            #[cfg(test)]
            self.report.weak_entries.push((owner, key));
        }
    }

    /// [[Realm]] is a logical edge: a live built-in keeps its Realm (keyed by the Realm's global
    /// object) alive, however its prototype has been rewired.
    fn realm(&mut self, function: usize, realm: usize) {
        self.sink.edge(function as *const u8, realm as u64);
        #[cfg(test)]
        self.report.realm_edges.push(function);
    }

    /// The logical edges a host graph reported.
    fn host(&mut self, host: &RefCell<HostGraph>) {
        for &(owner, value) in &host.borrow().edges {
            self.sink.edge(owner as *const u8, value);
        }
    }
}

impl Marking<'_> {
    fn trace_home(&self, index: usize, words: &mut dyn GcSink) {
        let Some(home) = &self.interps[index].shadow_home else {
            return;
        };
        let first = { self.activated_homes.borrow_mut().insert(index) };
        if first {
            home.route.trace_gc(words);
        }
    }

    /// The roots of interpreter `index` and of its host.
    fn trace_interp_roots(&self, index: usize, words: &mut dyn GcSink) {
        let interp = self.interps[index];
        interp.trace_gc_roots(words);
        interp
            .computed_reads
            .trace_pins(words, self.kind, interp.gc_heap.markings());
        let mut host = self.hosts[index].borrow_mut();
        for &root in &host.roots {
            words.word(root);
        }
        let mut reached = Vec::new();
        host.native.trace_roots(&mut reached);
        for target in reached {
            words.word(target as u64);
        }
    }
}

impl crate::gc_core::MarkClient for Marking<'_> {
    fn roots(&self, sink: &mut dyn FnMut(u64)) {
        let mut words = WordSink(sink);
        self.trace_interp_roots(0, &mut words);
        // A realm with code running: its roots, and its ShadowRealm object, which the sweep
        // must not take from under it. Other realms' roots are their objects' side slots.
        for &(owner, index) in &self.entered {
            words.word(owner as u64);
            self.trace_interp_roots(index, &mut words);
        }
        // The tree shares one heap; its interned strings live as long as it does.
        if let Some(root) = self.interps.first() {
            crate::value::trace_heap_roots_for(&root.gc_heap, &mut words, self.kind);
            // The Realms an embedder replaced by entering others, until it leaves them (S6 step
            // 11b, `GcState::entered_realms`).
            #[cfg(feature = "embed")]
            for entered in root.gc_heap.entered_realms.borrow().iter() {
                entered.saved.trace_gc(&mut words);
            }
        }
        crate::gc_trace::trace_thread_roots(&mut words);
        // The heap's strong handles (S6 step 11a): in a minor collection those written since
        // the last collection only; the others name old cells.
        #[cfg(test)]
        let mut reported = Vec::new();
        self.interps[0]
            .gc_heap
            .handles
            .trace_roots(self.kind, |_slot, value| {
                if let Some(word) = value_word(value) {
                    words.word(word);
                }
                #[cfg(test)]
                reported.push(_slot);
            });
        #[cfg(test)]
        self.report(|report| report.handle_slots = reported);
    }

    fn side_slot_owners(&self, owners: &mut dyn FnMut(usize)) {
        // Collected first: the heap may trace an already-marked owner's slots at once, which
        // re-enters this client.
        #[derive(Default)]
        struct Noted {
            seen: FastSet<usize>,
            list: Vec<usize>,
        }
        impl Noted {
            fn note(&mut self, owner: usize) {
                if self.seen.insert(owner) {
                    self.list.push(owner);
                }
            }
        }
        let mut noted = Noted::default();
        match self.kind {
            CollectionKind::Major => {
                for interp in &self.interps {
                    interp.for_each_side_slot_owner(&mut |owner| noted.note(owner));
                    // Owner pins are callback identities, not roots. Full collections census
                    // them; minors already receive every newly attached owner in ClientRecords.
                    for &owner in interp.gc_pins.keys() {
                        noted.note(owner);
                    }
                }
            }
            // The owners whose entries were written since the last collection, the young ones
            // among them (see `SideSlotTable`).
            CollectionKind::Minor => {
                for &owner in &self.records().side_slot_owners {
                    noted.note(owner);
                }
            }
        }
        #[cfg(test)]
        self.report(|report| report.side_slot_owners = noted.list.clone());
        for &owner in self.shadow_owners.keys() {
            noted.note(owner);
        }
        for &member in self.realm_members.keys() {
            noted.note(member);
        }
        for host in &self.hosts {
            for owner in host.borrow().native.js_owners() {
                noted.note(owner);
            }
        }
        for owner in noted.list {
            owners(owner);
        }
    }

    fn trace_side_slots(&self, owner: usize, sink: &mut dyn FnMut(u64)) {
        let mut words = WordSink(sink);
        if let Some(&index) = self.shadow_owners.get(&owner) {
            // Raw ownership is hierarchical: a live descendant implementation cannot outlive
            // the allocation containing its owning slot. This is a CONDITIONAL edge from a
            // reachable owner, never an unconditional root from the topology directory.
            self.trace_home(index, &mut words);
            // An entered realm's roots were traced as roots.
            if !self.interps[index].shadow_entered() {
                self.trace_interp_roots(index, &mut words);
            }
        }
        // A minor collection traces the side slots written since the last collection only: an
        // owner's others hold cells that collection marked (see `ClientRecords`).
        let slots =
            self.kind == CollectionKind::Major || self.records().side_slot_owners.contains(&owner);
        for (index, interp) in self.interps.iter().enumerate() {
            if interp.gc_pins.contains_key(&owner) {
                self.trace_home(index, &mut words);
            }
            if slots {
                interp.trace_owner_slots(owner, &mut words);
            }
        }
        for &(index, key) in self.realm_members.get(&owner).into_iter().flatten() {
            if !self.activated_realms.borrow_mut().insert((index, key)) {
                continue;
            }
            // A child global/intrinsic may be retained directly (AddToKeptObjects).
            self.trace_home(index, &mut words);
            if let Some(realm) = self.interps[index].realms.get(&key) {
                realm.trace_gc(&mut words);
            }
            for &template in self.templates.get(&(index, key)).into_iter().flatten() {
                words.word(template);
            }
        }
        let mut reached = Vec::new();
        for host in &self.hosts {
            host.borrow_mut().native.trace_js(owner, &mut reached);
            for target in reached.drain(..) {
                words.word(target as u64);
            }
        }
    }

    fn ephemerons(&self, sink: &mut crate::gc_core::EdgeSink<'_>) {
        let mut edges = EdgeReporter {
            sink,
            #[cfg(test)]
            report: ClientReport::default(),
        };
        match self.kind {
            CollectionKind::Major => {
                for (interp, host) in self.interps.iter().zip(&self.hosts) {
                    for (&owner, entries) in interp.weak_collection_data.iter() {
                        for (key, value) in entries {
                            edges.weak_entry(owner, key, value);
                        }
                    }
                    for (&function, &(_, realm)) in interp.native_function_realms.borrow().iter() {
                        edges.realm(function, realm);
                    }
                    edges.host(host);
                }
            }
            CollectionKind::Minor => {
                let records = self.records();
                // The entries written since the last collection, those of young keys among them
                // (see `WeakCollectionData`), each found in its collection's interpreter.
                for &(owner, key) in &records.weak_entries {
                    let entry = self.interps.iter().find_map(|interp| {
                        let entries = interp.weak_collection_data.get(&owner)?;
                        let index = interp.weak_collection_index.get(&owner)?.entries.get(&key);
                        Some(index.and_then(|&index| entries.get(index)))
                    });
                    if let Some((target, value)) = entry.flatten() {
                        edges.weak_entry(owner, target, value);
                    }
                }
                // The functions registered since the last collection: an old function's Realm
                // was marked when the function was (see `NativeFunctionRealms`).
                for &function in &records.native_functions {
                    let realm = self.interps.iter().find_map(|interp| {
                        let realms = interp.native_function_realms.borrow();
                        realms.get(&function).map(|&(_, realm)| realm)
                    });
                    if let Some(realm) = realm {
                        edges.realm(function, realm);
                    }
                }
                for host in &self.hosts {
                    edges.host(host);
                }
            }
        }
        #[cfg(test)]
        self.report(|report| {
            report.weak_entries = edges.report.weak_entries;
            report.realm_edges = edges.report.realm_edges;
        });
    }

    fn check_records(&self, heap: &crate::gc_core::Heap) {
        let records = self.records();
        let registered: FastSet<usize> = records.native_functions.iter().copied().collect();
        for interp in &self.interps {
            interp.check_side_slot_records(&records, heap);
            interp.check_weak_entry_records(&records, heap);
            interp.check_native_realm_records(&registered, heap);
        }
        self.interps[0].gc_heap.handles.check_records(heap);
        #[cfg(debug_assertions)]
        crate::value::check_layout_owners(&self.interps[0].gc_heap, heap, |sink| {
            for interp in &self.interps {
                // Verification only inspects layout-owning roots. NativeGcGraph traversal
                // consumes its worklist, which must remain intact for the actual marker.
                interp.trace_gc_roots(sink);
                for (_, layout) in interp.construct_capacity_hints.values() {
                    layout.trace_gc(sink);
                }
            }
            crate::gc_trace::trace_thread_roots(sink);
        });
    }
}

impl Interp {
    /// Whether this interpreter collects its heap. A ShadowRealm's interpreter shares its
    /// parent's heap, and only the root of that tree collects it.
    pub(crate) fn gc_heap_root(&self) -> bool {
        self.heap_release.owns_heap()
    }

    /// Whether a collection should run at this safepoint: the live-object count above its
    /// ceiling (`gc_next`), the heap's nursery or byte budget or stress interval, or a deferred
    /// request.
    pub(crate) fn gc_heap_due(&self) -> bool {
        crate::value::heap_live_objects(&self.gc_heap) > self.gc_heap.agent.next.get()
            || self.gc_heap.agent.pending.get().is_some()
            || self.gc_heap.core.borrow().collection_due()
    }

    /// An explicit collection of `kind` (a host or test request). It usually comes from just
    /// above a finished job, whose frames left stale cell addresses in the stack the
    /// collection's own frames are about to reuse. Zero that region first, from the caller's
    /// frame (inlined, see `Interp::gc_collect`).
    #[inline(always)]
    pub(crate) fn gc_heap_collect_explicit(
        &mut self,
        kind: CollectionKind,
    ) -> crate::agent_control::CollectionOutcome {
        #[cfg(not(target_arch = "wasm32"))]
        crate::gc_core::scrub_stack(EXPLICIT_SCRUB_BYTES);
        self.gc_heap_collect(crate::value::GcCause::Explicit, Some(kind))
    }

    /// Collect the whole Agent heap from this interpreter's current safepoint: a collection of `kind`,
    /// or with `None` of the kind the heap asks for (`gc_core::Heap::due_kind`: minor unless a
    /// major collection is due). Defers, leaving the request pending for the next safepoint,
    /// inside a no-GC region, while a cell borrow is outstanding or on a native stack segment
    /// the scanner cannot see; a deferred major request makes a major collection due.
    pub(crate) fn gc_heap_collect(
        &mut self,
        cause: crate::value::GcCause,
        kind: Option<CollectionKind>,
    ) -> crate::agent_control::CollectionOutcome {
        if self.gc_heap.agent.suppressed.get() != 0 || self.gc_heap.agent.collecting.get() {
            // An unpublished child must never publish its temporary identity as Agent root.
            self.gc_heap.agent.request(&self.gc_heap, kind);
            return crate::agent_control::CollectionOutcome::Deferred;
        }
        assert_ne!(
            self.realm_id,
            crate::agent_access::RealmId::UNPUBLISHED,
            "unpublished interpreter cannot collect"
        );
        self.with_access(|current| current.gc_heap_collect_active(cause, kind))
    }

    fn gc_heap_collect_active(
        &mut self,
        cause: crate::value::GcCause,
        kind: Option<CollectionKind>,
    ) -> crate::agent_control::CollectionOutcome {
        use crate::agent_control::CollectionOutcome;
        let heap = self.gc_heap.clone();
        let pending = heap.agent.pending.take();
        let requested =
            if kind == Some(CollectionKind::Major) || pending == Some(CollectionKind::Major) {
                Some(CollectionKind::Major)
            } else {
                kind.or(pending)
            };
        let kind = match heap.core.try_borrow_mut() {
            Ok(core) => requested.unwrap_or_else(|| core.due_kind()),
            Err(_) => {
                heap.agent.request(&heap, requested);
                return CollectionOutcome::Deferred;
            }
        };
        heap.agent.collecting.set(true);
        let _collecting = crate::agent_control::Collecting(heap.clone());
        let performance_started = crate::value::gc_performance_metrics_start();
        let log_started = gc_log().then(std::time::Instant::now);
        let objects_before = crate::value::heap_live_objects(&heap);
        // Frames that are live while the native stack is scanned — this one and those of
        // `gc_heap_mark` — must hold no stale cell addresses: a dead cell's address may be
        // reused by the time of the next collection. Sweeping, which handles every dead
        // address, runs in a sibling call, and the region both calls use is zeroed first.
        #[cfg(not(target_arch = "wasm32"))]
        timed(EngineStep::Scrub, || {
            crate::gc_core::scrub_stack(GC_SCRUB_BYTES)
        });
        let Some(native) = self.gc_heap_mark(&heap, kind) else {
            heap.agent.request(&heap, Some(kind));
            return CollectionOutcome::Deferred;
        };
        let log_marked = gc_log().then(std::time::Instant::now);
        let reclaimed = self.gc_heap_sweep(&heap, native, kind);
        let log_swept = gc_log().then(std::time::Instant::now);
        if crate::value::layout_census::enabled() {
            crate::value::layout_census::sample_live(&heap);
        }
        crate::value::gc_finish_generation(&heap);
        if let Some(pending) = heap.agent.pending.get() {
            // A request made while the core was borrowed could publish only Agent metadata.
            // Reattach it to the native due word AFTER finish_collection's budget reset.
            heap.core.borrow_mut().request_collection(pending);
        }
        let live = crate::value::heap_live_objects(&heap);
        // The ceiling is the live-object limit. Over the limit, keep the room gc_check left the
        // handler of the limit's RangeError: a stress or byte-budget collection must not take
        // it away.
        let limit = heap.agent.limit.get();
        self.set_agent_next(if live > limit {
            heap.agent.next.get().max(limit)
        } else {
            limit
        });
        if let (Some(started), Some(marked), Some(swept)) = (log_started, log_marked, log_swept) {
            let stats = heap.core.borrow().stats();
            let micros = |nanos: u64| nanos as f64 / 1000.0;
            let phases = crate::gc_core::Phase::NAMES
                .iter()
                .zip(stats.last_phase_nanos)
                .map(|(name, nanos)| format!(" {name}={:.0}", micros(nanos)))
                .collect::<String>();
            let core_nanos = stats.last_phase_nanos.iter().sum::<u64>();
            let names = heap.core.borrow().desc_names();
            let finalized = stats
                .last_finalized
                .iter()
                .enumerate()
                .filter(|(_, &count)| count != 0)
                .map(|(desc, count)| {
                    format!(
                        " fin:{}={count}",
                        names.get(desc).copied().unwrap_or("?").replace(' ', "_")
                    )
                })
                .collect::<String>();
            let steps = ENGINE_STEP_NANOS.with(|steps| steps.replace([0; ENGINE_STEPS]));
            let steps = ENGINE_STEP_NAMES
                .iter()
                .zip(steps)
                .map(|(name, nanos)| format!(" {name}={:.0}", micros(nanos)))
                .collect::<String>();
            let mark_nanos = (marked - started).as_nanos() as u64;
            let sweep_nanos = (swept - marked).as_nanos() as u64;
            let minor = match kind {
                CollectionKind::Minor => format!(
                    " young_pages={} card_cells={} promoted={}",
                    stats.last_pages_swept, stats.last_card_cells, stats.last_marked_bytes
                ),
                CollectionKind::Major => String::new(),
            };
            eprintln!(
                "[gc-heap] kind={} cause={cause:?} objects={objects_before}->{live} cells={} bytes={} pages={} spare={} collections={} deferred={}{minor} external={} us: mark={:.0} sweep={:.0} engine={:.0}{phases}{steps}{finalized}",
                match kind {
                    CollectionKind::Minor => "minor",
                    CollectionKind::Major => "major",
                },
                stats.live_cells,
                stats.live_bytes,
                stats.pages_live,
                stats.pages_spare,
                stats.collections,
                stats.deferred_collections,
                stats.external_bytes,
                micros(mark_nanos),
                micros(sweep_nanos),
                micros((mark_nanos + sweep_nanos).saturating_sub(core_nanos)),
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        if reclaimed >= 50_000 {
            crate::fastalloc::trim();
        }
        if let Some(started) = performance_started {
            crate::value::gc_performance_metrics_finish(
                started,
                objects_before.max(0) as usize,
                live,
                cause,
                kind,
            );
        }
        CollectionOutcome::Collected(kind)
    }

    /// Exercise the real root/side-slot/sweep client without conservative native roots. This
    /// makes packed-storage and native old-to-young edge tests independent of stale words in
    /// Rust frames.
    #[cfg(all(
        test,
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    pub(crate) fn gc_collect_precise_for_test(&mut self, kind: CollectionKind) {
        let heap = self.gc_heap.clone();
        let marking = Marking::new(self, kind);
        heap.handles.set_collecting(true);
        assert!(crate::value::mark_agent_heap(&heap, false, &marking, kind));
        heap.records.borrow_mut().consume();
        heap.handles.consume();
        let native = marking.into_native_graphs();
        self.gc_heap_sweep(&heap, native, kind);
        crate::value::gc_finish_generation(&heap);
    }

    /// Phase 1: mark from this tree's roots, side slots, ephemerons and the native stack (and,
    /// in a minor collection, from the heap's old cells whose cards are set). `None` when the
    /// heap deferred the collection.
    #[inline(never)]
    fn gc_heap_mark(
        &mut self,
        heap: &crate::value::GcHeap,
        kind: CollectionKind,
    ) -> Option<FastMap<usize, NativeGcGraph>> {
        let marking = timed(EngineStep::Client, || Marking::new(self, kind));
        // Building the client just enumerated every Realm member and host value, dead or
        // alive, in frames below this one; the marker's frames reuse that region.
        #[cfg(not(target_arch = "wasm32"))]
        timed(EngineStep::Scrub, || {
            crate::gc_core::scrub_stack(GC_SCRUB_BYTES)
        });
        // No handle may be made or made strong until the sweep has freed the dead.
        heap.handles.set_collecting(true);
        let started = crate::value::mark_agent_heap(heap, true, &marking, kind);
        if !started {
            heap.handles.set_collecting(false);
        }
        #[cfg(debug_assertions)]
        if started && kind == CollectionKind::Minor {
            let core = heap.core.borrow();
            crate::value::check_code_roots_marked(heap, &core);
            for interp in &marking.interps {
                interp.computed_reads.check_pins_marked(&core);
            }
        }
        started.then(|| {
            // This marking consumed what the tables and the handles recorded since the last
            // collection: what is written from now on, between the phases and after the sweep
            // included, is recorded for the next one.
            heap.records.borrow_mut().consume();
            heap.handles.consume();
            marking.into_native_graphs()
        })
    }

    /// Phase 2: evict the dead from the engine's tables, clear weak handles and free the dead
    /// cells (in a minor collection, the young ones only: the old are marked). Returns how many
    /// objects died.
    #[inline(never)]
    fn gc_heap_sweep(
        &mut self,
        heap: &crate::value::GcHeap,
        native: FastMap<usize, NativeGcGraph>,
        kind: CollectionKind,
    ) -> usize {
        // Between the phases every cell is intact and new cells are allocated marked: the
        // engine evicts the dead from its tables while their identities are still stable.
        let dead = timed(EngineStep::Dead, || crate::gc_sweep::DeadObjects::new(heap));
        // Weak handles to dead cells are cleared first (S6 step 11a): the host sweeps see them
        // cleared, and no cell they name is freed while they name it.
        timed(EngineStep::Weak, || {
            let core = heap.core.borrow();
            heap.handles
                .clear_dead_weak(kind, |word| core.is_dead_target(word as usize));
        });
        timed(EngineStep::Tables, || {
            crate::value::sweep_object_metadata(heap, &dead);
            heap.agent
                .metadata_owners
                .borrow_mut()
                .retain(|&identity, _| !dead.contains(identity));
            self.gc_heap_sweep_tree(&dead, &native, kind);
        });
        heap.records.borrow_mut().native_functions_swept.clear();
        drop(native);
        // Weak handles to dead cells are cleared; a cell a weak handle still names stays
        // reserved (finalized, never reused) until its last weak handle is dropped, which keeps
        // raw-address inline-cache comparisons ABA-safe.
        let pinned = timed(EngineStep::Weak, || {
            let core = heap.core.borrow();
            crate::gc_handle::clear_dead_weak_cells_in(kind == CollectionKind::Minor, |addr| {
                core.weak_target_state(addr)
            })
        });
        heap.core.borrow_mut().finish_collection(&pinned);
        if kind == CollectionKind::Major {
            crate::value::prune_function_runtimes_after_major(heap);
        }
        heap.handles.set_collecting(false);
        // The heap's bits no longer say what died: from here on only the list's length counts.
        let reclaimed = dead.len();
        drop(dead);
        timed(EngineStep::After, || self.gc_heap_after_sweep_agent(kind));
        reclaimed
    }

    /// Evict the dead from every interpreter's tables in this ShadowRealm tree. A dead
    /// ShadowRealm's interpreter is dropped here, while the cells it names are still intact.
    fn gc_heap_sweep_tree(
        &mut self,
        dead: &crate::gc_sweep::DeadObjects<'_>,
        native: &FastMap<usize, NativeGcGraph>,
        kind: CollectionKind,
    ) {
        let access = self.gc_heap.agent.access.clone();
        self.with_raw_access(|_| {
            let pointers = access.snapshot();
            let mut retired = Vec::new();
            for (_, pointer) in pointers {
                // SAFETY: marking references ended before this fresh mutable snapshot. Every
                // removed subtree remains in retired until ALL visitors have finished.
                let interp = unsafe { &mut *pointer.as_ptr() };
                let empty = NativeGcGraph::default();
                let graph = native.get(&(pointer.as_ptr() as usize)).unwrap_or(&empty);
                interp.gc_sweep_dead_objects(dead, graph, kind, &mut retired);
            }
            for realm in retired {
                drop_collected_realm(realm);
            }
        });
    }

    /// Evict the dead built-ins' [[Realm]] entries. A major collection walks the table. A minor
    /// collection finds only young objects dead, and every young built-in was registered since
    /// the last collection (S6 step 10d, see `NativeFunctionRealms`): it asks the dead list about
    /// the functions registered since, which its marking kept from the heap's records, or, when
    /// the dead list is the shorter, evicts each dead object's entry.
    pub(crate) fn gc_sweep_native_realms(
        &mut self,
        dead: &crate::gc_sweep::DeadObjects<'_>,
        kind: CollectionKind,
    ) {
        let realms = self.native_function_realms.get_mut();
        match kind {
            CollectionKind::Major => realms.retain(|(pin, _)| {
                pin.upgrade()
                    .is_some_and(|function| !dead.contains(Gc::as_ptr(&function) as usize))
            }),
            CollectionKind::Minor => {
                let records = self.gc_heap.records.borrow();
                let registered = &records.native_functions_swept;
                if registered.len() <= dead.len() {
                    for &function in registered {
                        if dead.contains(function) {
                            realms.remove(&function);
                        }
                    }
                } else {
                    for object in dead.iter() {
                        realms.remove(&object);
                    }
                }
            }
        }
    }

    /// Post-sweep bookkeeping for every interpreter in the tree: drop expired inline-cache
    /// pins, deliver destruction notifications and schedule FinalizationRegistry cleanup.
    fn gc_heap_after_sweep(&mut self) {
        self.gc_prune_dead_pins();
        self.gc_drain_weak_deaths();
        // The finalizers' death notices, just drained, can ready more registries, and every
        // registry ready here is live. Before the sweep, `gc_forget_weak_owners` removed each
        // dead registry from both `finalization_registries` and `ready_registries`; since then
        // only a registry still in `finalization_registries` can be readied
        // (`clear_weak_identity`), and no script has run to register one. So no dead check is
        // needed, and none could be made: the sweep has cleared the bits that said what died.
        self.gc_schedule_finalization_cleanup(None);
    }

    fn gc_heap_after_sweep_agent(&mut self, kind: CollectionKind) {
        self.gc_heap
            .agent
            .task_live
            .set(crate::value::heap_live_objects(&self.gc_heap));
        self.gc_heap.agent.task_pending.set(false);
        // ArrayBuffer data counts from the last major collection: a minor one leaves the old
        // buffers that died, and their data, for a major collection to free.
        if kind == CollectionKind::Major {
            let mut pressure = self.gc_heap.agent.buffers.get();
            pressure.since_gc = 0;
            self.gc_heap.agent.buffers.set(pressure);
        }
        let access = self.gc_heap.agent.access.clone();
        self.with_raw_access(|_| {
            for (_, pointer) in access.snapshot() {
                // SAFETY: exclusive post-sweep visits after the shared phase, resolving only
                // still-owned slots and the newly published requester capability.
                unsafe {
                    (&mut *pointer.as_ptr()).gc_heap_after_sweep();
                }
            }
        });
    }

    /// Owners of this interpreter's owner-keyed side-table slots (see `trace_owner_slots`).
    fn for_each_side_slot_owner(&self, owner: &mut dyn FnMut(usize)) {
        macro_rules! owners {
            ($($table:ident),+ $(,)?) => {
                $(for &key in self.$table.keys() {
                    owner(key);
                })+
            };
        }
        gc_side_slot_tables!(owners);
        for &key in self.generators.keys() {
            owner(key);
        }
    }

    /// The strong edges `owner` holds through this interpreter's owner-keyed side tables,
    /// traced once the owner is marked. A suspended generator's continuation belongs to its
    /// generator object the same way.
    fn trace_owner_slots(&self, owner: usize, sink: &mut dyn GcSink) {
        macro_rules! slots {
            ($($table:ident),+ $(,)?) => {
                $(if let Some(slots) = self.$table.get(&owner) {
                    slots.trace_gc(sink);
                })+
            };
        }
        gc_side_slot_tables!(slots);
        if let Some(coroutine) = self.generators.get(&owner) {
            GcTrace::trace_gc(coroutine, sink);
        }
    }

    /// S6 step 10d, debug builds, as each collection starts (`Marking::check_records`): a
    /// side-slot owner whose entries were not written since the last collection is old, and its
    /// side slots hold only cells the last collection marked, since a minor collection does not
    /// report it. Panics naming the table, the owner and the value.
    fn check_side_slot_records(&self, records: &ClientRecords, heap: &crate::gc_core::Heap) {
        let check = |table: &str, owner: usize, slots: &dyn GcTrace| {
            if records.side_slot_owners.contains(&owner) {
                return;
            }
            match heap.cell_age(owner as u64) {
                Some(CellAge::Young) => panic!(
                    "side-slot table {table}: young owner {owner:#x} was not recorded since the \
                     last collection"
                ),
                Some(CellAge::Old) => slots.trace_gc(&mut WordSink(&mut |value| {
                    if let Some(age @ (CellAge::Young | CellAge::Dead)) = heap.cell_age(value) {
                        panic!(
                            "side-slot table {table}: old owner {owner:#x}, not recorded since \
                             the last collection, holds an unmarked ({age:?}) value {value:#x}"
                        );
                    }
                })),
                // A zombie owner's entries were evicted as it died; an address that names no
                // allocated cell is not this heap's.
                _ => {}
            }
        };
        macro_rules! tables {
            ($($table:ident),+ $(,)?) => {
                $(for (&owner, slots) in self.$table.iter() {
                    check(stringify!($table), owner, slots);
                })+
            };
        }
        gc_side_slot_tables!(tables);
        if let Some(home) = &self.shadow_home {
            for &owner in self.gc_pins.keys() {
                check("native owner route", owner, &*home.route);
            }
        }
        for (&owner, coroutine) in self.generators.iter() {
            check("generators", owner, coroutine);
        }
    }

    /// S6 step 10d, debug builds, as each collection starts: a WeakMap or WeakSet entry not
    /// written since the last collection has an old key (or a dead one), and with an old key an
    /// old value, since a minor collection does not register it. Panics naming the collection,
    /// the key and the value.
    fn check_weak_entry_records(&self, records: &ClientRecords, heap: &crate::gc_core::Heap) {
        for (&owner, entries) in self.weak_collection_data.iter() {
            for (target, value) in entries {
                if records.weak_entries.contains(&(owner, target.key())) {
                    continue;
                }
                let Some(key) = target.upgrade().as_ref().and_then(Value::cell_word) else {
                    continue;
                };
                match heap.cell_age(key) {
                    Some(CellAge::Young) => panic!(
                        "weak_collection_data: the entry of {owner:#x} under young key {key:#x} \
                         was not recorded since the last collection"
                    ),
                    Some(CellAge::Old) => {
                        let age = value_word(value)
                            .and_then(|value| Some((value, heap.cell_age(value)?)));
                        if let Some((value, age @ (CellAge::Young | CellAge::Dead))) = age {
                            panic!(
                                "weak_collection_data: the entry of {owner:#x} under old key \
                                 {key:#x}, not recorded since the last collection, holds an \
                                 unmarked ({age:?}) value {value:#x}"
                            );
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    /// S6 step 10d, debug builds, as each collection starts: a built-in registered before the
    /// last collection (not in `registered`) is old, and its [[Realm]], the Realm's global
    /// object, was marked when it was, since a minor collection does not report the edge.
    /// Panics naming the function and the Realm.
    fn check_native_realm_records(&self, registered: &FastSet<usize>, heap: &crate::gc_core::Heap) {
        for (&function, &(_, realm)) in self.native_function_realms.borrow().iter() {
            if registered.contains(&function) {
                continue;
            }
            match heap.cell_age(function as u64) {
                Some(CellAge::Young) => panic!(
                    "native_function_realms: young function {function:#x} was not registered \
                     since the last collection"
                ),
                Some(CellAge::Old) => {
                    if let Some(age @ (CellAge::Young | CellAge::Dead)) =
                        heap.cell_age(realm as u64)
                    {
                        panic!(
                            "native_function_realms: old function {function:#x}, registered \
                             before the last collection, has an unmarked ({age:?}) [[Realm]] \
                             {realm:#x}"
                        );
                    }
                }
                _ => {}
            }
        }
    }

    /// Every strong root this interpreter holds itself. Owner-keyed side tables, Realm groups,
    /// weak tables and caches are not roots; each `_` below says why. ShadowRealm
    /// interpreters are traced as members of the collection's tree, not through this field.
    fn trace_gc_roots(&self, sink: &mut dyn GcSink) {
        let Interp {
            gc_heap: _,
            realm_id: _,
            symbol_agent,
            global,
            global_env,
            object_proto,
            function_proto,
            array_proto,
            array_ctor,
            string_proto,
            number_proto,
            boolean_proto,
            symbol_proto,
            error_protos,
            console: _,
            wall_clock: _,
            runtime_interrupt: _,
            interrupt_poll_tick: _,
            ast_tick: _,
            strict: _,
            tier: _,
            tier_threshold: _,
            jit_admit_runs: _,
            vm_pool,
            native_arg_pool,
            stub_cache: _,       // shapes and slots
            stub_cache_names: _, // property names
            arguments_shapes: _,
            descriptor_maps,
            iter_result_map,
            computed_reads: _, // keys pinned by collection kind (`Marking::trace_interp_roots`)
            fragment_cache: _, // weak Realm keys and compiled plans
            enumeration_keys,
            property_key_strings,
            frame_pool, // the active JIT records (Rust entries' contexts: thread roots)
            creation_pins: _, // weak pins
            global_env_pins: _, // weak pins
            call_site: _,
            code_frames,
            job_script_contexts,
            host_resolution_marker,
            jit_helpers: _,
            jit_layout: _,
            interp_layout: _,
            inline_ic_safe: _,
            str_units,
            re_texts,
            re_text_ascii_hot,
            regexp_dependency_cache: _,
            regexp_last,
            depth: _,
            direct_call_depth: _,
            proxy_forward_depth: _,
            class_info: _, // side slots
            eval_fn,
            eval_realm_fns: _,         // addresses
            native_function_realms: _, // weak pins; [[Realm]] edges are reported as ephemerons
            elems_protector: _,
            species_protector: _,
            ta_accessor_protector: _,     // weak pins
            array_iteration_protector: _, // weak pins
            construct_ics: _,             // weak pins
            call_overflow: _,             // weak pins
            native_callback_cache: _,     // weak pins
            proxy_trap_caches: _,         // weak pins
            self_hosted: _,
            shared_snapshots: _,         // parsed source
            construct_capacity_hints: _, // conditional side slots
            iterator_sym,
            wk_syms,
            short_circuit: _,
            import_meta,
            import_base: _,
            modules,
            module_recs,
            module_loader: _,
            dynamic_module_loader: _,
            routed_imports: _,       // conditional owner-keyed importValue bridge
            module_ns: _,            // side slots
            weak_collection_data: _, // ephemerons
            weak_collection_index: _,
            extra_protos,
            array_buffers: _,
            native_buffers: _,
            array_buffer_versions: _,
            array_buffer_dirty_ranges: _,
            immutable_buffers: _,
            host_keyed_buffers: _,
            can_block: _,
            agent: _,
            typed_arrays: _,
            async_gens: _,
            global_var_names: _,
            gc_pins: _,       // bookkeeping for side-table owners, not holders
            ta_buffer: _,     // side slots
            shadow_realms: _, // members of the collection's tree, traced as side slots
            shadow_home: _,   // a weak owner; its entry count is read by `Marking`
            data_views: _,
            regexps: _,
            regexp_programs: _,
            proxies: _,      // side slots
            host_indexed: _, // side slots
            new_target,
            pending_new_target,
            htmldda: _, // weak pins
            ctor_caller_realm,
            realms: _,   // Realm groups
            promises: _, // side slots
            unhandled_rejections,
            next_unhandled_rejection: _,
            handled_rejections,
            temporal: _,
            temporal_cal: _,
            host_job_context: _,
            script_entry: _,
            host_entry_realm: _,
            capture_job_script_caller: _,
            caller_origin, // actual conditional source-owner/Realm edges
            host_settings_states,
            retired_host_job_contexts: _,
            host_job_context_enter: _,
            host_job_context_leave: _,
            host_state: _, // reported through its HostGc visitors
            generators: _, // side slots
            gc_next: _,
            gc_tick: _,
            constructing: _,
            super_call_ok: _,
            fn_frames,
            module_async_seq: _,
            yield_raw_result: _,
            in_async_gen_body: _,
            pending_fn_name: _,
            pending_tail,
            tco_ok: _,
            template_cache: _, // Realm groups
            in_field_init_code: _,
            using_stack,
            accessor_seq: _,
            decorator_initializers,
            annexb_fn_sync: _, // parsed functions
            deferred_ns: _,
            deferred_ns_objs,
            promise_forward: _,  // side slots
            mapped_arguments: _, // side slots
            borrowed_parameters,
            module_source_objs,
            weak_refs: _,               // weak
            weak_metadata: _,           // weak
            finalization_registries: _, // side slots
            async_gen_busy: _,
            async_gen_queue: _, // side slots
            heap_release: _,
        } = self;
        for object in [
            global,
            object_proto,
            function_proto,
            array_proto,
            string_proto,
            number_proto,
            boolean_proto,
            symbol_proto,
            host_resolution_marker,
        ] {
            sink.object(object);
        }
        global_env.trace_gc(sink);
        array_ctor.trace_gc(sink);
        error_protos.trace_gc(sink);
        eval_fn.trace_gc(sink);
        extra_protos.trace_gc(sink);
        // Pooled buffers are empty; one in use is registered by its activation (`drive_vm`, or
        // a JIT entry's `gc_trace::JitFrameRoots`).
        for (slots, stack) in vm_pool {
            slots.trace_gc(sink);
            stack.trace_gc(sink);
        }
        frame_pool.trace_gc(sink);
        native_arg_pool.trace_gc(sink);
        for map in descriptor_maps.iter().chain([iter_result_map]) {
            if let Some(props) = map.get() {
                props.trace_gc(sink);
            }
        }
        for keys in enumeration_keys.values() {
            keys.trace_gc(sink);
        }
        for (_, symbol, _) in wk_syms {
            symbol.trace_gc(sink);
        }
        if let Some(symbol) = iterator_sym {
            sink.word(crate::value::symbol_identity(symbol) as u64);
        }
        // String caches keyed by a string's address pin it: a pin keeps the address from being
        // reused while an entry is keyed by it.
        for (_, string) in property_key_strings.values() {
            string.trace_gc(sink);
        }
        for (string, _, _) in str_units.values() {
            string.trace_gc(sink);
        }
        for (string, text) in re_texts.values() {
            string.trace_gc(sink);
            text.trace_gc(sink);
        }
        if let Some((string, _, text)) = re_text_ascii_hot {
            string.trace_gc(sink);
            text.trace_gc(sink);
        }
        if let Some(last) = regexp_last {
            last.input.trace_gc(sink);
            last.text.trace_gc(sink);
        }
        code_frames.trace_gc(sink);
        for (_, origin) in job_script_contexts {
            origin.trace_gc(sink);
        }
        if let Some(caller) = caller_origin {
            caller.origin.trace_gc(sink);
        }
        // The Agent's registered (Symbol.for) and well-known Symbols live as long as it does.
        {
            let agent = symbol_agent.borrow();
            for symbol in agent
                .global_by_key
                .values()
                .chain(agent.well_known.values())
            {
                sink.word(crate::value::symbol_identity(symbol) as u64);
            }
        }
        import_meta.trace_gc(sink);
        modules.trace_gc(sink);
        module_recs.trace_gc(sink);
        new_target.trace_gc(sink);
        pending_new_target.trace_gc(sink);
        ctor_caller_realm.trace_gc(sink);
        for (_, promise, reason, _) in unhandled_rejections.values() {
            promise.trace_gc(sink);
            reason.trace_gc(sink);
        }
        for (promise, reason, _) in handled_rejections {
            promise.trace_gc(sink);
            reason.trace_gc(sink);
        }
        host_settings_states.trace_gc(sink);
        fn_frames.trace_gc(sink);
        if let Some(tail) = pending_tail {
            let (callee, this, arguments) = &**tail;
            callee.trace_gc(sink);
            this.trace_gc(sink);
            arguments.trace_gc(sink);
        }
        using_stack.trace_gc(sink);
        decorator_initializers.trace_gc(sink);
        deferred_ns_objs.trace_gc(sink);
        borrowed_parameters.trace_gc(sink);
        module_source_objs.trace_gc(sink);
    }
}

#[cfg(test)]
mod tests {
    use crate::bytecode::Tier;
    use crate::value::Value;
    use crate::{Completion, Engine};

    fn eval(engine: &mut Engine, source: &str, expected: &str) {
        match engine.eval(source, false).expect("GC fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected),
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    #[test]
    fn collection_keeps_live_call_frames_scopes_weak_targets_and_side_tables_all_tiers() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            engine
                .interp
                .def_method(&engine.interp.global, "collect", 0, |interp, _, _| {
                    interp.gc_collect();
                    Ok(Value::Undefined)
                });
            engine.interp.gc_collect();
            eval(
                &mut engine,
                r#"
                function factory(n) {
                    let o = {n:n};
                    let map = new Map([[o, o]]), set = new Set([o]);
                    let weak = new WeakRef(o), weakMap = new WeakMap([[o, {owner:o}]]);
                    let bytes = new Uint8Array([n]);
                    function closure() {
                        collect();
                        return [o.n, map.get(o).n, set.has(o), weak.deref() === o,
                            weakMap.get(o).owner === o, bytes[0]].join(',');
                    }
                    collect();
                    return closure;
                }
                var closures = [];
                for (var i=0; i<20; i++) closures.push(factory(i));
                closures[7]() + ':' + closures[19]()
            "#,
                "7,7,true,true,true,7:19,19,true,true,true,19",
            );
            engine.interp.gc_collect();
            eval(&mut engine, "closures[3]()", "3,3,true,true,true,3");
        }
    }

    #[test]
    fn finalizer_targets_clear_together_and_ready_cells_are_removed() {
        let mut engine = Engine::new();
        engine.interp.gc_collect();
        eval(
            &mut engine,
            r#"
            var finalized = [], first, second;
            var registry = new FinalizationRegistry(function(value) { finalized.push(value); });
            (function() {
                var target = {}; target.self = target;
                first = new WeakRef(target); second = new WeakRef(target);
                registry.register(target, 91, target);
            })(); true
        "#,
            "true",
        );
        engine.interp.gc_collect();
        eval(&mut engine, "true", "true");
        eval(&mut engine, "[first.deref() === undefined, second.deref() === undefined, finalized.join(',')].join(':')", "true:true:91");
    }

    /// A dead-object view answers from the heap's bits between its collection's phases, and
    /// debug builds assert that no query comes later: once the sweep has cleared the marks
    /// and freed the dead, a query panics.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "a dead-object view queried outside its collection's phases")]
    fn dead_objects_view_queried_after_the_sweep_panics() {
        let mut engine = Engine::new();
        eval(
            &mut engine,
            "(() => { for (let i = 0; i < 100; i++) ({ i }); return 'done'; })()",
            "done",
        );
        let heap = engine.interp.gc_heap.clone();
        let major = crate::gc_core::CollectionKind::Major;
        let native = engine
            .interp
            .gc_heap_mark(&heap, major)
            .expect("not deferred");
        let dead = crate::gc_sweep::DeadObjects::new(&heap);
        for object in dead.iter() {
            assert!(dead.contains(object));
        }
        engine.interp.gc_heap_sweep(&heap, native, major);
        dead.contains(dead.iter().next().unwrap_or(0));
    }

    /// Owners of every internal-slot family keep their table edges across a collection, and
    /// cycles that pass only through those tables (or a suspended coroutine) are reclaimed: with
    /// few table entries, and beside 4,000 older Maps' entries, on every tier.
    #[test]
    fn side_slot_owners_keep_table_edges_with_sparse_and_dense_tables() {
        for dense_old_tables in [false, true] {
            for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
                let mut engine = Engine::new();
                engine.set_tier(tier);
                engine.set_tier_threshold(0);
                if dense_old_tables {
                    eval(
                        &mut engine,
                        "var oldMaps = []; for (var i = 0; i < 4000; i++) oldMaps.push(new Map([[i, {i}]])); true",
                        "true",
                    );
                }
                engine.interp.gc_collect();
                eval(
                    &mut engine,
                    r#"
                    var weak = [];
                    var keep = (function () {
                        var m = new Map(), key = {k: 1}; m.set(key, {v: 2});
                        var s = new Set([{s: 3}]);
                        var proxy = new Proxy({t: 5}, {});
                        function* gen() { var held = {g: 6}; yield held.g; yield held.g + 1; }
                        var it = gen(); it.next();
                        class C { x = {f: 7}; #p = {q: 9}; q() { return this.#p.q; } }
                        var args = (function (a) { return arguments; })({a: 8});
                        var pending = new Promise(function () {});
                        var reaction = {r: 4};
                        pending.then(function () { return reaction; });
                        var ta = new Uint8Array([10]);
                        return {m, key, s, proxy, it, C, args, ta};
                    })();
                    (function () {
                        var cyclic = new Map(); cyclic.set('self', cyclic);
                        var set = new Set(); set.add(set);
                        var it2 = (function* () { var me = it2; yield 1; })(); it2.next();
                        var target = {}; var p = new Proxy(target, {h: target}); target.p = p;
                        weak.push(new WeakRef(cyclic), new WeakRef(set), new WeakRef(it2),
                                  new WeakRef(target));
                    })();
                    true
                "#,
                    "true",
                );
                engine.interp.gc_collect();
                eval(&mut engine, "true", "true");
                eval(
                    &mut engine,
                    r#"[keep.m.get(keep.key).v, [...keep.s][0].s, keep.proxy.t,
                        keep.it.next().value, new keep.C().x.f, new keep.C().q(),
                        keep.args[0].a, keep.ta[0],
                        weak.map(function (w) { return w.deref() === undefined; }).join('/')
                       ].join(',')"#,
                    "2,3,5,7,7,9,8,10,true/true/true/true",
                );
            }
        }
    }
}
