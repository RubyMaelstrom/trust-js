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

/// Native stack zeroed below a collection's entry before marking, and again below
/// `Interp::gc_heap_mark` once its client is built (see `gc_core::scrub_stack`): the frames
/// live from there down to the stack scan's own, which take under 3 KiB in optimized builds
/// (2.4 KiB on ARM64 with fat LTO, 2.5 KiB on x86-64). Debug frames are several times larger.
/// Zeroing touches the stack's pages, so an optimized build zeroes no more than its frames need.
#[cfg(not(target_arch = "wasm32"))]
const GC_SCRUB_BYTES: usize = if cfg!(debug_assertions) {
    256 << 10
} else {
    16 << 10
};

/// Native stack zeroed before an explicit collection (see `Interp::gc_heap_collect_explicit`):
/// the collection's frames down to the stack scan's, which take under 4 KiB in optimized
/// builds (3.7 KiB on x86-64).
#[cfg(not(target_arch = "wasm32"))]
const EXPLICIT_SCRUB_BYTES: usize = if cfg!(debug_assertions) {
    1 << 20
} else {
    32 << 10
};

/// `TRUST_JS_GC_STRESS=N`: collect at the first safepoint after every N cell allocations
/// (N = 1: at every safepoint that follows an allocation), and poll at every call. While a
/// marking cycle runs, that collection is the cycle's final pause, not a slice
/// (`gc_core::Heap::due_work`), unless `TRUST_JS_GC_INCREMENTAL=force`.
pub(crate) fn gc_stress() -> usize {
    static STRESS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *STRESS.get_or_init(|| {
        std::env::var("TRUST_JS_GC_STRESS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    })
}

/// `TRUST_JS_GC_VERIFY=1`: verify an Agent's heap at every collection, in debug builds (see
/// `gc_core::Heap::set_heap_verification`). The engine's unit tests always verify; any other
/// debug build only when it is set to anything but `0`, since each check walks the whole heap.
fn gc_verify() -> bool {
    static VERIFY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    cfg!(test)
        || *VERIFY.get_or_init(|| {
            std::env::var_os("TRUST_JS_GC_VERIFY").is_some_and(|value| value != "0")
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

/// The heap's maximum, which with pacing bounds each major collection's growing factor
/// (`gc_pacing::max_growing_factor`) and its byte budget, at halfway from the bytes it kept
/// (`gc_pacing::bound_allocation_limit`): V8's, from the physical memory as V8 reads it
/// (`gc_pacing::max_heap_bytes`), read once. `TRUST_JS_GC_MAX_HEAP_BYTES=N` (diagnostic)
/// replaces it as V8's `--max-old-space-size` does; 0 keeps V8's.
fn gc_max_heap_bytes() -> usize {
    static MAX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX.get_or_init(|| {
        let configured = std::env::var("TRUST_JS_GC_MAX_HEAP_BYTES")
            .ok()
            .and_then(|value| value.parse().ok());
        let physical = crate::host_memory::amount_of_physical_memory().unwrap_or(0);
        crate::gc_pacing::max_heap_bytes(configured, physical)
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

/// Whether Agent heaps follow V8's sizing policy (see `gc_pacing`): outside the engine's unit
/// tests, unless `TRUST_JS_GC_PACING=0` (diagnostic) fixes the budgets; `TRUST_JS_GC_PACING=1`
/// turns it on in the unit tests too.
#[cfg(not(target_arch = "wasm32"))]
fn gc_pacing_enabled() -> bool {
    static PACING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PACING.get_or_init(|| match std::env::var_os("TRUST_JS_GC_PACING") {
        Some(value) => value != "0",
        None => !cfg!(test),
    })
}

/// `TRUST_JS_GC_INCREMENTAL` (W6): on by default, the heap marks the major collections it asks
/// for on its own in cycles of slices between which the program runs; `0` (or `off`) marks
/// every major collection in one pause; `force` makes the slices a few cells at every safepoint
/// (verification's mode). A wasm32 build keeps it off: a slice's time needs a clock, which
/// wasm32-unknown-unknown lacks (as pacing does, `gc_pacing_enabled`).
fn gc_incremental() -> crate::gc_core::IncrementalMode {
    static MODE: std::sync::OnceLock<crate::gc_core::IncrementalMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("TRUST_JS_GC_INCREMENTAL").as_deref() {
            Ok("0" | "off") => crate::gc_core::IncrementalMode::Off,
            Ok("force") => crate::gc_core::IncrementalMode::Force,
            _ if cfg!(target_arch = "wasm32") => crate::gc_core::IncrementalMode::Off,
            _ => crate::gc_core::IncrementalMode::On,
        },
    )
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
    core.set_max_heap_bytes(gc_max_heap_bytes());
    // `TRUST_JS_GC_LOG=1`: each collection's log line includes its phases' times.
    core.set_phase_timing(gc_log());
    core.set_nursery_bytes(gc_nursery_bytes());
    // V8's sizing policy and memory reducer (see `gc_pacing`), with a nursery that grows and
    // shrinks unless a diagnostic fixes it. The engine's unit tests keep the fixed policy, whose
    // budgets do not depend on time.
    #[cfg(not(target_arch = "wasm32"))]
    if gc_pacing_enabled() {
        core.enable_pacing(std::env::var_os("TRUST_JS_GC_NURSERY_BYTES").is_none());
    }
    core.set_stress_every(gc_stress());
    core.set_stress_major_every(gc_stress_major());
    core.set_heap_verification(gc_verify());
    core.set_incremental(gc_incremental());
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
/// `weak_metadata::WeakCollection`, [`NativeFunctionRealms`]), each through the only methods that
/// write them. It lives in the heap's `GcState`, not beside each table, so that the tables keep the
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
    /// A ShadowRealm interpreter's newly pinned owner (`Interp::gc_pin`), which may already be
    /// old. A child records each, even during its unpublished bootstrap: publication of
    /// shadow_home happens before marking.
    pub(crate) fn remember_owner(&mut self, owner: usize) {
        self.side_slot_owners.insert(owner);
    }

    /// The entry of the WeakMap or WeakSet at `owner` under `key` was written
    /// (`weak_metadata::WeakCollection`, whose writes are the only callers).
    #[inline]
    pub(crate) fn record_weak_entry(&mut self, owner: usize, key: WeakKey) {
        self.weak_entries.insert((owner, key));
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
    #[cfg(all(test, debug_assertions))]
    pub(crate) fn forget_side_slot_owner(&mut self, owner: usize) {
        self.side_slot_owners.remove(&owner);
    }

    /// Forget that `owner`'s entry under `key` was written (tests).
    #[cfg(all(test, debug_assertions))]
    pub(crate) fn forget_weak_entry(&mut self, owner: usize, key: WeakKey) {
        self.weak_entries.remove(&(owner, key));
    }

    /// Forget that `function` was registered since the last collection (tests).
    #[cfg(all(test, debug_assertions))]
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
/// since it was allocated. Heap verification checks both as each collection starts
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

/// `Interp::native_function_realms`: each built-in function's [[Realm]] (ECMA-262
/// CreateBuiltinFunction), by the function's address: a weak pin of the function and the address
/// of its Realm's global object. An entry never changes once registered
/// ([`NativeFunctionRealms::register`], which records the function in the heap's
/// [`ClientRecords`]; S6 step 10d). A minor collection reports the [[Realm]] edges of the
/// functions registered since the last collection only (an old function's Realm was marked
/// when the function was), and sweeps the table from them, the only functions it can find dead.
/// Heap verification checks the first as each collection starts
/// (`Interp::check_native_realm_records`).
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
    #[cfg(all(test, debug_assertions))]
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
    /// Whether the collection is minor ([`HostGcVisitor::is_minor`]).
    minor: bool,
    roots: Vec<u64>,
    /// Logical edges: each value is reachable while its owner is.
    edges: Vec<(usize, u64)>,
    native: NativeGcGraph,
}

impl HostGcVisitor for HostGraph {
    fn is_minor(&self) -> bool {
        self.minor
    }

    fn native_old_generation(
        &mut self,
        domain: &'static str,
        owner: usize,
        first_young_id: usize,
    ) -> bool {
        if self.minor {
            self.native
                .declare_old_generation(domain, owner, first_young_id);
        }
        self.minor
    }

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

/// Which marking a [`Marking`] reports for (W6 step 2b): a collection's in one pause, or a
/// marking cycle's start or final pause, whose kind is major.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Purpose {
    /// A collection's marking, of the client's kind: a major one reports every table in full, a
    /// minor one what the tables recorded since the last collection.
    Collection,
    /// A marking cycle's start: every table as a major collection's start, but for the weak
    /// collections, reported as side-slot owners whose entries the cycle discovers once it marks
    /// them ([`crate::gc_core::MarkClient::discovered_ephemerons`]).
    CycleStart,
    /// A marking cycle's final pause: the roots and the host graphs in full, of the tables what
    /// they recorded during the cycle (its start consumed the records before), and every Realm
    /// group and ShadowRealm again.
    FinalPause,
}

/// The [`crate::gc_core::MarkClient`] of one collection over a ShadowRealm tree.
struct Marking<'a> {
    /// A minor collection's client reports what the tables recorded instead of the tables (see
    /// [`ClientRecords`]).
    kind: CollectionKind,
    /// The marking it reports for: a collection's, or a cycle's start or final pause (W6).
    purpose: Purpose,
    /// The weak collections whose side slots a cycle's start or final pause traced, whose
    /// entries it reports next ([`crate::gc_core::MarkClient::discovered_ephemerons`]).
    discovered: RefCell<Vec<usize>>,
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
    /// Debug builds: the interpreters whose roots this marking traced, the ones heap verification
    /// examines after a minor marking (`Interp::gc_heap_mark`).
    #[cfg(debug_assertions)]
    traced: RefCell<FastSet<usize>>,
}

impl<'a> Marking<'a> {
    /// Out of line: its frames must be dead (and scrubbed) before the native stack is scanned.
    #[inline(never)]
    fn new(root: &'a Interp, kind: CollectionKind, purpose: Purpose) -> Self {
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
        let core = root.gc_heap.core.borrow();
        for (index, interp) in interps.iter().enumerate() {
            let mut host = HostGraph {
                heap_id: crate::value::heap_id(&root.gc_heap),
                minor: kind == CollectionKind::Minor,
                ..HostGraph::default()
            };
            interp.host_state.trace_gc(&mut host);
            hosts.push(RefCell::new(host));
            for (&key, realm) in &interp.realms {
                // A minor collection keeps every cell an earlier collection marked: a group
                // whose global object one marked is active from the start, and the global
                // object, marked already, stands for its members (it activates the group at
                // once). The members of the others are its members. So does a cycle's final
                // pause, whose marks are the cycle's: it traces each active group again.
                let global = Gc::as_ptr(&realm.global) as usize;
                if (kind == CollectionKind::Minor || purpose == Purpose::FinalPause)
                    && core.cell_age(global as u64) == Some(crate::gc_core::CellAge::Old)
                {
                    realm_members.entry(global).or_default().push((index, key));
                    continue;
                }
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
            purpose,
            discovered: RefCell::default(),
            interps,
            hosts,
            shadow_owners,
            entered,
            realm_members,
            templates,
            activated_realms: RefCell::default(),
            activated_homes: RefCell::default(),
            #[cfg(debug_assertions)]
            traced: RefCell::default(),
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

/// W6 step 2b: what a collection's marking of `kind` reports for: a major one while a cycle
/// marks is the cycle's final pause.
fn marking_purpose(heap: &crate::value::GcState, kind: CollectionKind) -> Purpose {
    if kind == CollectionKind::Major && heap.core.borrow().cycle_running() {
        Purpose::FinalPause
    } else {
        Purpose::Collection
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
        #[cfg(debug_assertions)]
        self.traced.borrow_mut().insert(index);
        let interp = self.interps[index];
        interp.trace_own_roots(self.kind, words);
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

    /// W6 step 2b: the Realm groups as a cycle's start found them, and those it activated, for
    /// its slices (`GcState::cycle_realms`).
    fn cycle_realms(&self) -> CycleRealms {
        let members = self
            .realm_members
            .iter()
            .map(|(&member, groups)| (member, groups.iter().map(|&(_, key)| key).collect()))
            .collect();
        let activated = self
            .activated_realms
            .borrow()
            .iter()
            .map(|&(_, key)| key)
            .collect();
        CycleRealms { members, activated }
    }

    /// W6 step 2b, heap verification once a cycle's final pause has marked
    /// ([`crate::gc_core::MarkClient::check_closure`]): beside each interpreter's tables (see
    /// `Interp::check_side_slots_closed`), a Realm group with a marked member is marked whole,
    /// with its template map, and a marked ShadowRealm's interpreter roots are marked.
    fn verify_closure(&self, heap: &crate::gc_core::Heap) {
        let check = |what: &str, word: u64| {
            if let Some(age @ (CellAge::Young | CellAge::Dead)) = heap.cell_age(word) {
                panic!(
                    "marking cycle: {what} holds an unmarked ({age:?}) cell {word:#x} after the \
                     final pause"
                );
            }
        };
        let marked = |word: u64| heap.cell_age(word) == Some(CellAge::Old);
        for (index, interp) in self.interps.iter().enumerate() {
            interp.check_side_slots_closed(heap);
            interp.check_weak_entries_closed(heap);
            interp.check_native_realms_closed(heap);
            for (&key, realm) in &interp.realms {
                let mut members = Vec::new();
                realm.trace_gc(&mut WordSink(&mut |word| members.push(word)));
                if !members.iter().any(|&word| marked(word)) {
                    continue;
                }
                for &word in &members {
                    check("a Realm group with a marked member", word);
                }
                for &word in self.templates.get(&(index, key)).into_iter().flatten() {
                    check(
                        "the template map of a Realm group with a marked member",
                        word,
                    );
                }
            }
        }
        for (&owner, &index) in &self.shadow_owners {
            if marked(owner as u64) {
                self.interps[index].trace_gc_roots(&mut WordSink(&mut |word| {
                    check("a marked ShadowRealm's interpreter", word)
                }));
            }
        }
    }
}

/// W6 step 2b: report the entries of the weak collections at `discovered`, those whose side slots
/// a cycle's client traced, as ephemerons ([`crate::gc_core::MarkClient::discovered_ephemerons`]),
/// skipping an address no interpreter registers as a collection.
fn report_discovered(
    interps: &[&Interp],
    discovered: Vec<usize>,
    sink: &mut crate::gc_core::EdgeSink<'_>,
) {
    let mut edges = EdgeReporter {
        sink,
        #[cfg(test)]
        report: ClientReport::default(),
    };
    for owner in discovered {
        if !interps
            .iter()
            .any(|interp| interp.weak_collections.contains(&owner))
        {
            continue;
        }
        // SAFETY: a registered collection's address names an allocated object, which nothing
        // writes while the collector marks.
        unsafe {
            crate::weak_metadata::with_weak_collection(owner, |data| {
                for (key, value) in data.iter() {
                    edges.weak_entry(owner, key, value);
                }
            })
        };
    }
}

/// W6 step 2b: what a marking cycle's clients share from its start to its final pause
/// (`GcState::cycle_realms`): the Realm groups' members as the start found them, each with its
/// group's Realm key (its global object's address, which names one Realm of the tree while the
/// cycle runs, since no cell is freed before its final pause), and the groups activated since.
/// A slice's client activates a group once in the cycle; the final pause traces every active
/// group again, from its own client.
#[derive(Default)]
pub(crate) struct CycleRealms {
    members: FastMap<usize, Vec<usize>>,
    activated: FastSet<usize>,
}

/// W6 step 2b: the client of a marking cycle's slice (`Interp::gc_heap_slice`), built for that
/// slice over the tree's interpreters as they are at its safepoint. The heap asks it only for the
/// side slots of the owners the slice marks and for the entries of the weak collections among
/// them. It traces an owner's side-slot tables, a ShadowRealm's home and interpreter roots and a
/// Realm group as a major collection's client does, and discovers a weak collection's entries.
/// It reports no root and no host graph (an embedder's), which the cycle's start and final pause
/// report in full, as V8 marks the embedder's roots again at its final pause.
struct SliceMarking<'a> {
    interps: Vec<&'a Interp>,
    /// ShadowRealm object -> the index of its realm's interpreter.
    shadow_owners: FastMap<usize, usize>,
    realms: &'a RefCell<CycleRealms>,
    activated_homes: RefCell<FastSet<usize>>,
    discovered: RefCell<Vec<usize>>,
}

impl<'a> SliceMarking<'a> {
    #[inline(never)]
    fn new(root: &'a Interp) -> Self {
        let pointers = root
            .gc_heap
            .agent
            .access
            .shared_snapshot(root.realm_id, root);
        let mut interps = Vec::with_capacity(pointers.len());
        let mut shadow_owners = FastMap::default();
        for (_, pointer) in pointers {
            // SAFETY: as in `Marking::new`: the slice's current lease refreshes the requester,
            // every other active participant published its current borrow at suspension, and
            // the references end with the slice.
            let interp = unsafe { &*pointer };
            if let Some(home) = &interp.shadow_home {
                let owner = home
                    .owner
                    .upgrade()
                    .expect("a registered realm has a live owner cell");
                shadow_owners.insert(Gc::as_ptr(&owner) as usize, interps.len());
            }
            interps.push(interp);
        }
        SliceMarking {
            interps,
            shadow_owners,
            realms: &root.gc_heap.cycle_realms,
            activated_homes: RefCell::default(),
            discovered: RefCell::default(),
        }
    }

    fn trace_home(&self, index: usize, words: &mut dyn GcSink) {
        let Some(home) = &self.interps[index].shadow_home else {
            return;
        };
        if self.activated_homes.borrow_mut().insert(index) {
            home.route.trace_gc(words);
        }
    }
}

impl crate::gc_core::MarkClient for SliceMarking<'_> {
    // A slice marks from its worklist alone: the heap asks for no root and no owner.
    fn roots(&self, _: &mut dyn FnMut(u64)) {}
    fn side_slot_owners(&self, _: &mut dyn FnMut(usize)) {}
    fn ephemerons(&self, _: &mut crate::gc_core::EdgeSink<'_>) {}

    fn trace_side_slots(&self, owner: usize, sink: &mut dyn FnMut(u64)) {
        let mut words = WordSink(sink);
        if let Some(&index) = self.shadow_owners.get(&owner) {
            self.trace_home(index, &mut words);
            // An entered realm's roots are roots, at the cycle's start and its final pause.
            let interp = self.interps[index];
            if !interp.shadow_entered() {
                interp.trace_own_roots(CollectionKind::Major, &mut words);
            }
        }
        for (index, interp) in self.interps.iter().enumerate() {
            if interp.gc_pins.contains_key(&owner) {
                self.trace_home(index, &mut words);
            }
            interp.trace_owner_slots_in_slice(owner, &mut words);
            if interp.weak_collections.contains(&owner) {
                self.discovered.borrow_mut().push(owner);
            }
        }
        // A Realm group the cycle's start found, once in the cycle (a Realm made since is the
        // final pause's).
        let keys: Vec<usize> = {
            let mut realms = self.realms.borrow_mut();
            let CycleRealms { members, activated } = &mut *realms;
            members
                .get(&owner)
                .into_iter()
                .flatten()
                .copied()
                .filter(|&key| activated.insert(key))
                .collect()
        };
        for key in keys {
            for (index, interp) in self.interps.iter().enumerate() {
                if let Some(realm) = interp.realms.get(&key) {
                    // A child global/intrinsic may be retained directly (AddToKeptObjects).
                    self.trace_home(index, &mut words);
                    realm.trace_gc(&mut words);
                    interp.trace_realm_templates(key, &mut words);
                }
            }
        }
    }

    fn discovered_ephemerons(&self, sink: &mut crate::gc_core::EdgeSink<'_>) {
        let discovered = std::mem::take(&mut *self.discovered.borrow_mut());
        report_discovered(&self.interps, discovered, sink);
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
        match (self.kind, self.purpose) {
            (CollectionKind::Major, Purpose::Collection | Purpose::CycleStart) => {
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
            // among them (see `SideSlotTable`); at a cycle's final pause, those written during
            // the cycle: its start registered the others, whose edges wait in the marker or
            // fired with the slots as they are.
            (CollectionKind::Minor, _) | (CollectionKind::Major, Purpose::FinalPause) => {
                for &owner in &self.records().side_slot_owners {
                    noted.note(owner);
                }
            }
        }
        #[cfg(test)]
        self.report(|report| report.side_slot_owners = noted.list.clone());
        // A cycle's start registers the weak collections as owners: marking one brings its
        // entries (`discovered_ephemerons`).
        if self.purpose == Purpose::CycleStart {
            for interp in &self.interps {
                for &owner in &interp.weak_collections {
                    noted.note(owner);
                }
            }
        }
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
        let cycle = self.purpose != Purpose::Collection;
        for (index, interp) in self.interps.iter().enumerate() {
            if interp.gc_pins.contains_key(&owner) {
                self.trace_home(index, &mut words);
            }
            if slots {
                interp.trace_owner_slots(owner, &mut words);
            }
            // A cycle discovers a weak collection's entries once it marks the collection.
            if cycle && interp.weak_collections.contains(&owner) {
                self.discovered.borrow_mut().push(owner);
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
        match (self.kind, self.purpose) {
            (CollectionKind::Major, Purpose::Collection | Purpose::CycleStart) => {
                for (interp, host) in self.interps.iter().zip(&self.hosts) {
                    // A cycle's start reports the weak collections as side-slot owners instead,
                    // and discovers their entries as it marks them.
                    if self.purpose == Purpose::Collection {
                        for &owner in &interp.weak_collections {
                            // SAFETY: a registered collection's address names an allocated
                            // object, which nothing writes while the collector marks.
                            unsafe {
                                crate::weak_metadata::with_weak_collection(owner, |data| {
                                    for (key, value) in data.iter() {
                                        edges.weak_entry(owner, key, value);
                                    }
                                })
                            };
                        }
                    }
                    for (&function, &(_, realm)) in interp.native_function_realms.borrow().iter() {
                        edges.realm(function, realm);
                    }
                    edges.host(host);
                }
            }
            // A minor collection, and a cycle's final pause, whose start registered every
            // collection, entry and function then: what was written since.
            (CollectionKind::Minor, _) | (CollectionKind::Major, Purpose::FinalPause) => {
                let records = self.records();
                // The entries written since the last collection, those of young keys among them
                // (see `weak_metadata::WeakCollection`), each in its collection, which an
                // interpreter registers while it is allocated.
                for &(owner, key) in &records.weak_entries {
                    if !self
                        .interps
                        .iter()
                        .any(|interp| interp.weak_collections.contains(&owner))
                    {
                        continue;
                    }
                    // SAFETY: a registered collection's address names an allocated object, which
                    // nothing writes while the collector marks.
                    unsafe {
                        crate::weak_metadata::with_weak_collection(owner, |data| {
                            if let Some((target, value)) = data.get(&key) {
                                edges.weak_entry(owner, target, value);
                            }
                        })
                    };
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

    fn discovered_ephemerons(&self, sink: &mut crate::gc_core::EdgeSink<'_>) {
        let discovered = std::mem::take(&mut *self.discovered.borrow_mut());
        report_discovered(&self.interps, discovered, sink);
    }

    fn check_closure(&self, heap: &crate::gc_core::Heap) {
        self.verify_closure(heap);
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
        // W6: an explicit major collection is precise. One that finishes a running cycle keeps
        // what the cycle marked before it died, so a second collection follows, once the first's
        // frames are gone and zeroed, and frees it (V8's PreciseCollectAllGarbage,
        // src/heap/heap.cc:1365-1370).
        let finishes_cycle = kind == CollectionKind::Major
            && self
                .gc_heap
                .core
                .try_borrow()
                .is_ok_and(|core| core.cycle_running());
        let outcome = self.gc_heap_collect(crate::value::GcCause::Explicit, Some(kind));
        if finishes_cycle && outcome == crate::agent_control::CollectionOutcome::Collected(kind) {
            #[cfg(not(target_arch = "wasm32"))]
            crate::gc_core::scrub_stack(EXPLICIT_SCRUB_BYTES);
            return self.gc_heap_collect(crate::value::GcCause::Explicit, Some(kind));
        }
        outcome
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
        // W6: a major collection the heap asks for on its own may start a marking cycle, and
        // while one runs the heap's own work is a slice; a requested major collection, or an
        // explicit minor one, is the cycle's final pause.
        let heap_choice = kind.is_none() && pending.is_none();
        let explicit_minor =
            matches!(cause, crate::value::GcCause::Explicit) && kind == Some(CollectionKind::Minor);
        let work = match heap.core.try_borrow_mut() {
            Ok(core) => core.due_work(requested, heap_choice, explicit_minor),
            Err(_) => {
                heap.agent.request(&heap, requested);
                return CollectionOutcome::Deferred;
            }
        };
        let kind = match work {
            crate::gc_core::DueWork::Collect(kind) => kind,
            crate::gc_core::DueWork::StartCycle => return self.gc_heap_start_cycle(&heap, cause),
            crate::gc_core::DueWork::Slice => return self.gc_heap_slice(&heap, cause),
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
        if crate::value::elem_census::enabled() {
            crate::value::elem_census::sample(
                &heap,
                match kind {
                    CollectionKind::Minor => "minor",
                    CollectionKind::Major => "major",
                },
            );
        }
        crate::value::gc_finish_generation(&heap);
        if let Some(pending) = heap.agent.pending.get() {
            // A request made while the core was borrowed could publish only Agent metadata.
            // Reattach it to the poll words' heap bit AFTER finish_collection's budget reset.
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
        #[cfg(not(target_arch = "wasm32"))]
        if reclaimed >= 50_000 {
            crate::fastalloc::trim();
        }
        if let (Some(started), Some(marked), Some(swept)) = (log_started, log_marked, log_swept) {
            // The pause: the whole collection, from the choice of its kind to here.
            let pause_nanos = started.elapsed().as_nanos() as u64;
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
                .zip(stats.last_finalize_nanos)
                .enumerate()
                .filter(|(_, (&count, _))| count != 0)
                .map(|(desc, (count, nanos))| {
                    let name = names.get(desc).copied().unwrap_or("?").replace(' ', "_");
                    format!(" fin:{name}={count} fin_us:{name}={:.0}", micros(nanos))
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
            let (kind_fields, cycle) = match kind {
                CollectionKind::Minor => (
                    format!(
                        " young_pages={} card_cells={} promoted={}",
                        stats.last_pages_swept, stats.last_card_cells, stats.last_marked_bytes
                    ),
                    String::new(),
                ),
                // A major collection is the final pause of its marking cycle (W6): the slices
                // that marked before it, the marked cells it rescanned for their set cards (none
                // in one pause), the bytes allocated since the cycle started, and the cycle's
                // whole marking time, its slices' and this pause's.
                CollectionKind::Major => (
                    format!(
                        " slices={} card_cells={} cycle_alloc={}",
                        stats.last_cycle_slices, stats.last_card_cells, stats.last_cycle_allocated
                    ),
                    format!(
                        " cycle_mark={:.0}",
                        micros(stats.last_cycle_slice_nanos + mark_nanos)
                    ),
                ),
            };
            eprintln!(
                "[gc-heap] kind={} cause={cause:?} objects={objects_before}->{live} cells={} bytes={} pages={} spare={} collections={} deferred={}{kind_fields} external={} budget={} atoms={} atom_bytes={} us: pause={:.0} mark={:.0} sweep={:.0} engine={:.0}{cycle}{phases}{steps}{finalized}",
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
                stats.trigger_bytes,
                heap.atoms.borrow().len(),
                heap.atoms.borrow().heap_bytes(),
                micros(pause_nanos),
                micros(mark_nanos),
                micros(sweep_nanos),
                micros((mark_nanos + sweep_nanos).saturating_sub(core_nanos)),
            );
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

    /// W6: start a marking cycle over the whole Agent heap from this interpreter's safepoint
    /// (design 6): the client, built as a major collection builds it, reports the roots and the
    /// side tables' conditional edges (step 2b: the weak collections as owners, whose entries
    /// the slices discover), and the heap marks the roots: no native stack and no slice, as V8's
    /// start (step 4). This marking consumes the tables' and the handles' records, as a
    /// collection's does: what is written from now on is recorded for the final pause. The
    /// Realm groups the client found stay for the slices (`GcState::cycle_realms`). Deferred as a
    /// collection is.
    #[cold]
    #[inline(never)]
    pub(super) fn gc_heap_start_cycle(
        &mut self,
        heap: &crate::value::GcHeap,
        cause: crate::value::GcCause,
    ) -> crate::agent_control::CollectionOutcome {
        use crate::agent_control::CollectionOutcome;
        heap.agent.collecting.set(true);
        let _collecting = crate::agent_control::Collecting(heap.clone());
        let performance_started = crate::value::gc_performance_metrics_start();
        let log_started = gc_log().then(std::time::Instant::now);
        // The start scans no native stack (V8's, step 4): no frame below is zeroed first.
        self.gc_evict_young_cache_keys();
        let marking = Marking::new(self, CollectionKind::Major, Purpose::CycleStart);
        heap.handles.set_collecting(true);
        let started = crate::value::start_agent_cycle(heap, &marking);
        heap.handles.set_collecting(false);
        if !started {
            heap.agent.request(heap, Some(CollectionKind::Major));
            return CollectionOutcome::Deferred;
        }
        heap.records.borrow_mut().consume();
        heap.handles.consume();
        *heap.cycle_realms.borrow_mut() = marking.cycle_realms();
        drop(marking);
        if let Some(started) = log_started {
            self.gc_log_slice(heap, cause, "start", 0, started);
        }
        if let Some(started) = performance_started {
            crate::value::gc_performance_metrics_slice(started);
        }
        CollectionOutcome::Marked
    }

    /// W6: a slice of the running cycle's marking (design 4), from this interpreter's
    /// safepoint: no root and no stack, the marker's worklist alone, with a client built for the
    /// slice that traces the side slots of the owners it marks (step 2b, `SliceMarking`).
    /// Deferred as a collection is.
    #[cold]
    #[inline(never)]
    fn gc_heap_slice(
        &mut self,
        heap: &crate::value::GcHeap,
        cause: crate::value::GcCause,
    ) -> crate::agent_control::CollectionOutcome {
        use crate::agent_control::CollectionOutcome;
        heap.agent.collecting.set(true);
        let _collecting = crate::agent_control::Collecting(heap.clone());
        let performance_started = crate::value::gc_performance_metrics_start();
        let log_started = gc_log().then(std::time::Instant::now);
        let client = SliceMarking::new(self);
        let Some(marked) = crate::value::slice_agent_heap(heap, &client) else {
            heap.agent.request(heap, None);
            return CollectionOutcome::Deferred;
        };
        if let Some(started) = log_started {
            self.gc_log_slice(heap, cause, "slice", marked, started);
        }
        if let Some(started) = performance_started {
            crate::value::gc_performance_metrics_slice(started);
        }
        CollectionOutcome::Marked
    }

    /// W6: the GC log's line for a cycle's start or slice (`TRUST_JS_GC_LOG`).
    #[cold]
    #[inline(never)]
    fn gc_log_slice(
        &self,
        heap: &crate::value::GcHeap,
        cause: crate::value::GcCause,
        phase: &str,
        marked: usize,
        started: std::time::Instant,
    ) {
        let pause = started.elapsed().as_nanos() as f64 / 1000.0;
        let core = heap.core.borrow();
        let Some(progress) = core.cycle_progress() else {
            return;
        };
        // The start's marking is the cycle's so far; its line adds the heap's phases (the rest of
        // its pause is the engine's: the client, the records).
        let (marked, phases) = if phase == "start" {
            let stats = core.stats();
            let phases = crate::gc_core::Phase::NAMES
                .iter()
                .zip(stats.last_phase_nanos)
                .filter(|(name, _)| matches!(**name, "begin" | "roots" | "side"))
                .map(|(name, nanos)| format!(" {name}={:.0}", nanos as f64 / 1000.0))
                .collect::<String>();
            (progress.marked, phases)
        } else {
            (marked, String::new())
        };
        eprintln!(
            "[gc-slice] phase={phase} cause={cause:?} n={} traced={} marked={marked} cycle_marked={} worklist={} cycle_alloc={} precleans={} precleaned={} complete={} us: pause={pause:.0}{phases}",
            progress.slices,
            progress.traced,
            progress.marked,
            progress.worklist,
            progress.allocated,
            progress.precleans,
            progress.last_precleaned,
            u8::from(progress.complete)
        );
    }

    /// W6 (tests): start a marking cycle now, whatever the heap asks for, as its own major
    /// collection would when incremental marking is on. False when no cycle could start (one runs
    /// already, or the start was deferred).
    #[cfg(test)]
    pub(crate) fn gc_start_cycle(&mut self) -> bool {
        if self.gc_heap.core.borrow().cycle_running() {
            return false;
        }
        let outcome = self.with_access(|current| {
            let heap = current.gc_heap.clone();
            current.gc_heap_start_cycle(&heap, crate::value::GcCause::Explicit)
        });
        outcome == crate::agent_control::CollectionOutcome::Marked
    }

    /// W6 (tests): slice the running cycle until its worklist runs out, so that its final pause
    /// is due (at the next safepoint, or an explicit collection's).
    #[cfg(test)]
    pub(crate) fn gc_slice_to_completion(&mut self) {
        let complete = |interp: &Self| {
            let progress = interp.gc_heap.core.borrow().cycle_progress();
            progress.expect("a running cycle").complete
        };
        for _ in 0..10_000_000 {
            if complete(self) {
                return;
            }
            assert!(self.gc_slice(), "a slice of the running cycle");
        }
        panic!("the cycle's worklist never ran out");
    }

    /// W6 (tests): run a slice of the running cycle now. False when none runs.
    #[cfg(test)]
    pub(crate) fn gc_slice(&mut self) -> bool {
        if !self.gc_heap.core.borrow().cycle_running() {
            return false;
        }
        let outcome = self.with_access(|current| {
            let heap = current.gc_heap.clone();
            current.gc_heap_slice(&heap, crate::value::GcCause::Explicit)
        });
        outcome == crate::agent_control::CollectionOutcome::Marked
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
        // A running marking cycle is finished first, whatever kind the caller asks for. For a
        // major one: the cycle's final pause keeps what the cycle marked before it died, and the
        // precise collection that follows marks afresh and frees it, as V8's
        // PreciseCollectAllGarbage finalizes incremental marking before its full collection
        // (src/heap/heap.cc:1365-1370) and as `gc_heap_collect_explicit` collects twice. For a
        // minor one: no minor collection runs while a cycle marks (`Heap::start_marking`).
        // Under TRUST_JS_GC_STRESS=1 a stress major's cycle can be running as a test asks.
        if self
            .gc_heap
            .core
            .try_borrow()
            .is_ok_and(|core| core.cycle_running())
        {
            self.gc_heap_collect(crate::value::GcCause::Explicit, Some(CollectionKind::Major));
        }
        let heap = self.gc_heap.clone();
        self.gc_evict_young_cache_keys();
        let marking = Marking::new(self, kind, marking_purpose(&heap, kind));
        heap.handles.set_collecting(true);
        assert!(crate::value::mark_agent_heap(&heap, false, &marking, kind));
        heap.records.borrow_mut().consume();
        heap.handles.consume();
        heap.cycle_realms.take();
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
        self.gc_evict_young_cache_keys();
        let purpose = marking_purpose(heap, kind);
        let marking = timed(EngineStep::Client, || Marking::new(self, kind, purpose));
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
        if started && kind == CollectionKind::Minor && heap.core.borrow().verifies() {
            let core = heap.core.borrow();
            crate::value::check_code_roots_marked(heap, &core);
            crate::value::check_shape_layouts_marked(heap, &core);
            heap.atoms.borrow().check_marked(&core);
            // An interpreter whose roots the marking did not trace is a ShadowRealm's that
            // died young: the sweep drops it, caches and all, before it frees a cell.
            let traced = marking.traced.borrow();
            for (index, interp) in marking.interps.iter().enumerate() {
                if traced.contains(&index) {
                    interp.computed_reads.check_pins_marked(&core);
                    interp.check_cache_roots_marked(&core);
                }
            }
        }
        started.then(|| {
            // This marking consumed what the tables and the handles recorded since the last
            // collection: what is written from now on, between the phases and after the sweep
            // included, is recorded for the next one.
            heap.records.borrow_mut().consume();
            heap.handles.consume();
            if purpose == Purpose::FinalPause {
                heap.cycle_realms.take();
            }
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

    /// As a collection starts, drop each interpreter's cache entries keyed by strings made since
    /// the last one (`ComputedReadCache::evict_young`), so that the collection need not pin them.
    fn gc_evict_young_cache_keys(&mut self) {
        let access = self.gc_heap.agent.access.clone();
        self.with_raw_access(|_| {
            for (_, pointer) in access.snapshot() {
                // SAFETY: as in `gc_heap_sweep_tree`: no marking reference exists yet, and this
                // mutable snapshot ends before the client takes its shared one.
                let interp = unsafe { &mut *pointer.as_ptr() };
                interp.computed_reads.evict_young();
            }
        });
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
        // buffers that died, and their data, for a major collection to free. The sweep has
        // finalized the dead buffers' payloads, which left the Agent's buffer ledger.
        if kind == CollectionKind::Major {
            self.reset_buffer_pressure();
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

    /// The roots this interpreter holds itself, as a marking of `kind` traces them; its host's
    /// are its host graph's (`Marking::trace_interp_roots`).
    fn trace_own_roots(&self, kind: CollectionKind, words: &mut dyn GcSink) {
        self.trace_gc_roots(words);
        self.computed_reads
            .trace_pins(words, kind, self.gc_heap.markings());
        self.trace_cache_roots(kind, words);
    }

    /// W6 step 2b: [`Interp::trace_owner_slots`] for a marking cycle's slices (`SliceMarking`), a
    /// copy of its own: with the slices as its second caller, LTO stopped inlining the walk into a
    /// collection's client (`Marking::trace_side_slots`), which takes it for every marked owner,
    /// and a collection's side phase grew by a third (W6 step 2b's record).
    fn trace_owner_slots_in_slice(&self, owner: usize, sink: &mut dyn GcSink) {
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

    /// W6 step 2b: the values of the Realm `realm`'s [[TemplateMap]] (GetTemplateObject's), as a
    /// slice activates the Realm's group (`SliceMarking::trace_side_slots`).
    fn trace_realm_templates(&self, realm: usize, words: &mut dyn GcSink) {
        for (&(of, _), value) in &self.template_cache {
            if of == realm {
                if let Some(word) = value_word(value) {
                    words.word(word);
                }
            }
        }
    }

    /// W6 step 2b, heap verification once a cycle's final pause has marked
    /// (`Marking::verify_closure`): every side slot of a marked owner, in the tables, the
    /// generators and, in a ShadowRealm's interpreter, the route its pins keep, names a marked
    /// cell. The final pause registered only the owners written during the cycle: this catches a
    /// write its records missed. Panics naming the table, the owner and the value.
    fn check_side_slots_closed(&self, heap: &crate::gc_core::Heap) {
        let check = |table: &str, owner: usize, slots: &dyn GcTrace| {
            if heap.cell_age(owner as u64) != Some(CellAge::Old) {
                return;
            }
            slots.trace_gc(&mut WordSink(&mut |value| {
                if let Some(age @ (CellAge::Young | CellAge::Dead)) = heap.cell_age(value) {
                    panic!(
                        "marking cycle: side-slot table {table}: marked owner {owner:#x} holds an \
                         unmarked ({age:?}) value {value:#x} after the final pause: a write its \
                         records missed"
                    );
                }
            }));
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

    /// W6 step 2b, heap verification once a cycle's final pause has marked: every entry of a
    /// marked WeakMap or WeakSet under a marked key holds a marked value. Panics naming the
    /// collection, the key and the value.
    fn check_weak_entries_closed(&self, heap: &crate::gc_core::Heap) {
        for &owner in &self.weak_collections {
            if heap.cell_age(owner as u64) != Some(CellAge::Old) {
                continue;
            }
            let check = |data: &crate::weak_metadata::WeakCollection| {
                for (target, value) in data.iter() {
                    let Some(key) = target.upgrade().as_ref().and_then(Value::cell_word) else {
                        continue;
                    };
                    if heap.cell_age(key) != Some(CellAge::Old) {
                        continue;
                    }
                    let age =
                        value_word(value).and_then(|value| Some((value, heap.cell_age(value)?)));
                    if let Some((value, age @ (CellAge::Young | CellAge::Dead))) = age {
                        panic!(
                            "marking cycle: weak collection {owner:#x}: the entry under marked \
                             key {key:#x} holds an unmarked ({age:?}) value {value:#x} after the \
                             final pause: a write its records missed"
                        );
                    }
                }
            };
            // SAFETY: a registered collection's address names an allocated object, which nothing
            // writes while the collection marks.
            unsafe { crate::weak_metadata::with_weak_collection(owner, check) };
        }
    }

    /// W6 step 2b, heap verification once a cycle's final pause has marked: a marked built-in's
    /// [[Realm]] is marked. Panics naming the function and the Realm.
    fn check_native_realms_closed(&self, heap: &crate::gc_core::Heap) {
        for (&function, &(_, realm)) in self.native_function_realms.borrow().iter() {
            if heap.cell_age(function as u64) != Some(CellAge::Old) {
                continue;
            }
            if let Some(age @ (CellAge::Young | CellAge::Dead)) = heap.cell_age(realm as u64) {
                panic!(
                    "marking cycle: native_function_realms: marked function {function:#x} has an \
                     unmarked ({age:?}) [[Realm]] {realm:#x} after the final pause: a \
                     registration its records missed"
                );
            }
        }
    }

    /// S6 step 10d, heap verification, as each collection starts (`Marking::check_records`): a
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

    /// S6 step 10d, heap verification, as each collection starts: a WeakMap or WeakSet entry not
    /// written since the last collection has an old key (or a dead one), and with an old key an
    /// old value, since a minor collection does not register it. Panics naming the collection,
    /// the key and the value.
    fn check_weak_entry_records(&self, records: &ClientRecords, heap: &crate::gc_core::Heap) {
        for &owner in &self.weak_collections {
            let check = |data: &crate::weak_metadata::WeakCollection| {
                for (target, value) in data.iter() {
                    if records.weak_entries.contains(&(owner, target.key())) {
                        continue;
                    }
                    let Some(key) = target.upgrade().as_ref().and_then(Value::cell_word) else {
                        continue;
                    };
                    match heap.cell_age(key) {
                        Some(CellAge::Young) => panic!(
                            "weak collection: the entry of {owner:#x} under young key {key:#x} \
                             was not recorded since the last collection"
                        ),
                        Some(CellAge::Old) => {
                            let age = value_word(value)
                                .and_then(|value| Some((value, heap.cell_age(value)?)));
                            if let Some((value, age @ (CellAge::Young | CellAge::Dead))) = age {
                                panic!(
                                    "weak collection: the entry of {owner:#x} under old key \
                                     {key:#x}, not recorded since the last collection, holds an \
                                     unmarked ({age:?}) value {value:#x}"
                                );
                            }
                        }
                        _ => {}
                    }
                }
            };
            // SAFETY: a registered collection's address names an allocated object, which nothing
            // writes while the collection starts.
            unsafe { crate::weak_metadata::with_weak_collection(owner, check) };
        }
    }

    /// S6 step 10d, heap verification, as each collection starts: a built-in registered before the
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
    /// The string caches keyed by a string's address, which pin it (a pin keeps the address
    /// from being reused while an entry is keyed by it), as a marking of `kind` traces them: a
    /// minor collection traces the entries made since the last one (`ByteLru::trace_roots`).
    fn trace_cache_roots(&self, kind: CollectionKind, sink: &mut dyn GcSink) {
        self.enumeration_keys
            .trace_roots(kind, |keys| keys.trace_gc(sink));
        self.property_key_strings
            .trace_roots(kind, |string| string.trace_gc(sink));
    }

    /// Heap verification, after a minor collection's marking: every cell the string caches pin is
    /// marked, so the minor collection, which traced only the entries made since the last one,
    /// missed none.
    #[cfg(debug_assertions)]
    pub(crate) fn check_cache_roots_marked(&self, core: &crate::gc_core::Heap) {
        struct Marked<'a>(&'a crate::gc_core::Heap);
        impl GcSink for Marked<'_> {
            fn word(&mut self, word: u64) {
                assert!(
                    self.0.cell_age(word) != Some(crate::gc_core::CellAge::Young),
                    "a string cache pins the young cell {word:#x} after a minor collection's \
                     marking"
                );
            }
        }
        let sink = &mut Marked(core);
        for keys in self.enumeration_keys.values() {
            keys.trace_gc(sink);
        }
        for string in self.property_key_strings.values() {
            string.trace_gc(sink);
        }
    }

    fn trace_gc_roots(&self, sink: &mut dyn GcSink) {
        let Interp {
            gc_heap: _,
            gc_core_raw: _,
            realm_id: _,
            symbol_agent,
            global,
            global_env,
            object_proto,
            function_proto,
            array_proto,
            array_ctor,
            string_proto,
            string_proto_fuse: _,
            string_proto_fuse_ptr: _,
            number_proto,
            boolean_proto,
            symbol_proto,
            error_protos,
            console: _,
            wall_clock: _,
            runtime_interrupt: _,
            poll: _,
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
            regexp_array_maps,
            computed_reads: _, // keys pinned by collection kind (`Marking::trace_interp_roots`)
            fragment_cache: _, // weak Realm keys and compiled plans
            enumeration_keys: _, // by collection kind (`Marking::trace_interp_roots`)
            property_key_strings: _, // by collection kind
            frame_pool,        // the active JIT records (Rust entries' contexts: thread roots)
            creation_pins: _,  // weak pins
            global_env_pins: _, // weak pins
            call_site: _,
            code_frames,
            job_script_contexts,
            host_resolution_marker,
            jit_helpers: _,
            jit_layout: _,
            interp_layout: _,
            regexp_protector: _, // weak pins
            regexp_instance_shape: _,
            regexp_statics,
            depth: _,
            direct_call_depth: _,
            proxy_forward_depth: _,
            class_info: _, // side slots
            eval_fn,
            eval_realm_fns: _,         // addresses
            native_function_realms: _, // weak pins; [[Realm]] edges are reported as ephemerons
            elems_protector: _,
            ic_store_fill: _,
            species_protector: _,
            ta_accessor_protector: _,     // weak pins
            array_iteration_protector: _, // weak pins
            construct_ics: _,             // weak pins
            call_overflow: _,             // weak pins
            native_callback_cache: _,     // weak pins
            proxy_trap_caches: _,         // weak pins
            trap_calls: _,                // weak pins
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
            routed_imports: _,   // conditional owner-keyed importValue bridge
            module_ns: _,        // side slots
            weak_collections: _, // the collections, whose payloads' entries are ephemerons
            extra_protos,
            can_block: _,
            agent: _,
            async_gens: _,
            global_var_names: _,
            gc_pins: _,       // bookkeeping for side-table owners, not holders
            shadow_realms: _, // members of the collection's tree, traced as side slots
            shadow_home: _,   // a weak owner; its entry count is read by `Marking`
            regexp_programs: _,
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
            promise_forward: _, // side slots
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
        for map in descriptor_maps
            .iter()
            .chain([iter_result_map])
            .chain(regexp_array_maps)
        {
            if let Some(props) = map.get() {
                props.trace_gc(sink);
            }
        }
        for (_, symbol, _) in wk_syms {
            symbol.trace_gc(sink);
        }
        if let Some(symbol) = iterator_sym {
            sink.word(crate::value::symbol_identity(symbol) as u64);
        }
        // The active Realm's pending legacy-statics match (each saved Realm's travels in its
        // RealmState, traced with it).
        if let Some(last) = regexp_statics.borrow().as_ref() {
            last.input.trace_gc(sink);
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
