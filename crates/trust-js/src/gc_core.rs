//! P1-S1 heap core: pointer-based mark-sweep heap for the GC-managed
//! object model (docs/v8-chase-p1-plan-2026-10-15.txt, Decisions 0–4).
//! Feature `gc-heap`, default off; nothing executes on this heap yet.
//!
//! Design summary:
//! - Cells live in 64 KiB pages segregated by (size class, descriptor);
//!   page metadata is a side table found by page number, so interior
//!   pointers resolve by masking to the page base. Large cells (> 4 KiB)
//!   get dedicated allocations, found by address range.
//! - References are direct pointers (never handle-table indices); a
//!   tagged execution word masks to the same pointer (payload mask).
//! - Tracing is precise inside the heap (per-descriptor trace functions)
//!   and for registered Rust-heap roots, owner-conditional for side tables
//!   (edge sources), and conservative only for the native stack and
//!   registers, per the Decision 0 scanning contract (see `native`).
//! - Weak slots and ephemerons are rediscovered every collection from
//!   reachable holders. Nothing weak is registered globally, so a reused
//!   cell address can neither revive a dead entry nor be clobbered by one.
//! - Collection honors I9: it never runs inside a no-GC region or while a
//!   GcCell borrow is outstanding (deferred via a pending flag; the borrow
//!   case asserts in debug builds), and cells are allocated zeroed.
//! - Every registration is an RAII handle. Dropping the heap finalizes
//!   the remaining cells and frees all heap memory (I5 teardown).
//! - Debug builds poison freed cells and verify the marked graph.

// Until S2 wires execution to this heap, most of the surface is used
// only from tests (same posture as heap.rs/tagged.rs).
#![allow(dead_code)]

use crate::fasthash::FastMap;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;

pub(crate) const PAGE_SIZE: usize = 64 * 1024;
const PAGE_SHIFT: u32 = PAGE_SIZE.trailing_zeros();
const PAYLOAD_MASK: u64 = 0x0000_ffff_ffff_ffff;
/// Cells larger than the last class are large objects.
const SIZE_CLASSES: [usize; 16] = [
    16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
];
const DEFAULT_MIN_TRIGGER: usize = 1024 * 1024;
const DEFAULT_GROWTH_PERCENT: usize = 100;
#[cfg(debug_assertions)]
const POISON: u64 = 0xdddd_dddd_dddd_dddd;

fn size_class(size: usize) -> Option<usize> {
    SIZE_CLASSES.iter().position(|&c| size <= c)
}

fn bit(bits: &[u64], index: usize) -> bool {
    bits[index / 64] >> (index % 64) & 1 == 1
}

fn set_bit(bits: &mut [u64], index: usize, on: bool) {
    let mask = 1u64 << (index % 64);
    if on {
        bits[index / 64] |= mask;
    } else {
        bits[index / 64] &= !mask;
    }
}

fn page_layout() -> Layout {
    Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).expect("GC page layout")
}

fn large_layout(size: usize) -> Layout {
    Layout::from_size_align(size, 16).expect("large GC cell layout")
}

/// Trace callback: report every strong heap edge of the cell at `cell` by
/// calling `visit(edge)` with a cell pointer or tagged word. Must tolerate
/// every state the cell can be in at an allocation point (I9): allocation
/// returns zeroed memory, and a cell held only in a native local is traced
/// (conservative roots) before it is published anywhere, so initialization
/// must finish before the next allocation or write fields in an order in
/// which each intermediate state traces safely.
pub(crate) type TraceFn = fn(*const u8, &mut dyn FnMut(*const u8));
/// Weak callback: report the cell's weak slots and ephemeron entries
/// through the [`WeakSink`]. Called once per collection for each reachable
/// cell of the descriptor; same state tolerance as [`TraceFn`].
pub(crate) type WeakFn = fn(*mut u8, &mut WeakSink<'_>);
/// Finalize callback for a dead cell, run during sweep or heap teardown.
/// Must not allocate from this heap, resurrect the cell, or read other
/// cells (they may already be finalized); same state tolerance as
/// [`TraceFn`].
pub(crate) type FinalizeFn = fn(*mut u8);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DescId(pub(crate) u32);

pub(crate) struct LayoutDesc {
    pub(crate) name: &'static str,
    pub(crate) trace: TraceFn,
    pub(crate) weak: Option<WeakFn>,
    pub(crate) finalize: Option<FinalizeFn>,
}

/// Receives one reachable cell's weak data during marking.
pub(crate) struct WeakSink<'a> {
    weak_slots: &'a mut Vec<*mut u64>,
    ephemerons: &'a mut Vec<(*mut u64, *mut u64)>,
}

impl WeakSink<'_> {
    /// `slot`, a word inside the cell, holds a weak reference: it is zeroed
    /// when its target does not survive the collection.
    pub(crate) fn weak_slot(&mut self, slot: *mut u64) {
        self.weak_slots.push(slot);
    }

    /// An ephemeron entry inside the cell: the value word is reachable only
    /// while the key word's target is. When the key dies both words are
    /// zeroed, so the holder reads the entry as empty. A zero key is an
    /// empty entry; a key that is not a heap reference never dies.
    pub(crate) fn ephemeron(&mut self, key: *mut u64, value: *mut u64) {
        self.ephemerons.push((key, value));
    }
}

/// Receives the edges an edge source keeps off-heap.
pub(crate) struct EdgeSink<'a> {
    edges: &'a mut Vec<SourceEdge>,
}

#[derive(Clone, Copy)]
struct SourceEdge {
    owner: u64,
    key: Option<u64>,
    value: u64,
}

impl EdgeSink<'_> {
    /// `value` is reachable while `owner` is: an ordinary strong edge whose
    /// storage lives in a side table. A null owner makes it a root.
    pub(crate) fn edge(&mut self, owner: *const u8, value: u64) {
        self.edges.push(SourceEdge {
            owner: owner as u64,
            key: None,
            value,
        });
    }

    /// `value` is reachable only while both `owner` and `key` are (a WeakMap
    /// entry). A zero key is an empty entry and is skipped. Pruning entries
    /// whose key or owner died is the table's sweep hook's job.
    pub(crate) fn ephemeron(&mut self, owner: *const u8, key: u64, value: u64) {
        if key != 0 {
            self.edges.push(SourceEdge {
                owner: owner as u64,
                key: Some(key),
                value,
            });
        }
    }
}

/// Registration table shared by the heap and the RAII handles that own its
/// entries, so a handle can be dropped while the heap is borrowed elsewhere.
struct Registry<T> {
    entries: RefCell<Vec<Option<T>>>,
    free: RefCell<Vec<usize>>,
    /// Removals requested while the collector was iterating `entries`.
    deferred: RefCell<Vec<usize>>,
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Self {
            entries: RefCell::new(Vec::new()),
            free: RefCell::new(Vec::new()),
            deferred: RefCell::new(Vec::new()),
        }
    }
}

impl<T> Registry<T> {
    fn insert(self: &Rc<Self>, entry: T) -> Registration<T> {
        let mut entries = self
            .entries
            .try_borrow_mut()
            .expect("GC registration added during collection");
        let index = match self.free.borrow_mut().pop() {
            Some(index) => {
                entries[index] = Some(entry);
                index
            }
            None => {
                entries.push(Some(entry));
                entries.len() - 1
            }
        };
        Registration {
            registry: Rc::clone(self),
            index,
        }
    }

    fn remove(&self, index: usize) {
        if let Ok(mut entries) = self.entries.try_borrow_mut() {
            entries[index] = None;
            self.free.borrow_mut().push(index);
        } else {
            self.deferred.borrow_mut().push(index);
        }
    }

    /// Visit live entries. An entry whose handle is dropped meanwhile (from
    /// inside a callback) is skipped for the rest of the pass and removed
    /// after it.
    fn for_each(&self, mut f: impl FnMut(&mut T)) {
        {
            let mut entries = self.entries.borrow_mut();
            for index in 0..entries.len() {
                if self.deferred.borrow().contains(&index) {
                    continue;
                }
                if let Some(entry) = entries[index].as_mut() {
                    f(entry);
                }
            }
        }
        let deferred = std::mem::take(&mut *self.deferred.borrow_mut());
        for index in deferred {
            self.remove(index);
        }
    }
}

/// Owns one registration; dropping it unregisters.
pub(crate) struct Registration<T> {
    registry: Rc<Registry<T>>,
    index: usize,
}

impl<T> Drop for Registration<T> {
    fn drop(&mut self) {
        self.registry.remove(self.index);
    }
}

pub(crate) struct RootRangeEntry {
    name: &'static str,
    lo: *const u64,
    hi: *const u64,
}

pub(crate) struct TraceRootEntry {
    name: &'static str,
    trace: Box<dyn FnMut(&mut dyn FnMut(*const u8))>,
}

pub(crate) struct EdgeSourceEntry {
    name: &'static str,
    report: Box<dyn FnMut(&mut EdgeSink<'_>)>,
}

pub(crate) struct SweepHookEntry {
    name: &'static str,
    hook: Box<dyn FnMut(&dyn Fn(*const u8) -> bool)>,
}

pub(crate) type RootRange = Registration<RootRangeEntry>;
pub(crate) type TraceRoot = Registration<TraceRootEntry>;
pub(crate) type EdgeSource = Registration<EdgeSourceEntry>;
pub(crate) type SweepHook = Registration<SweepHookEntry>;

impl Registration<RootRangeEntry> {
    /// Re-point the range after its container reallocated.
    ///
    /// # Safety
    /// As for [`Heap::add_root_range`].
    pub(crate) unsafe fn set(&self, lo: *const u64, hi: *const u64) {
        let mut entries = self
            .registry
            .entries
            .try_borrow_mut()
            .expect("root range re-pointed during collection");
        let entry = entries[self.index].as_mut().expect("registered root range");
        entry.lo = lo;
        entry.hi = hi;
    }
}

struct PageMeta {
    base: usize,
    cell_size: usize,
    cells: usize,
    desc: DescId,
    class: usize,
    marks: Vec<u64>,
    /// Allocated (handed-out) cells. Conservative marking must ignore
    /// free cells: a stale stack word resolving into a free cell would
    /// otherwise "mark" free-list linkage and trace it as payload.
    allocated: Vec<u64>,
    /// Cells a conditional edge is waiting on during this collection.
    waiters: Vec<u64>,
    /// Dead cells already finalized but still pinned by weak handles: like the allocation
    /// behind `std::rc::Weak`, the address is not reused while a weak handle may compare it.
    /// Never marked or traced; freed once no weak handle pins it.
    zombies: Vec<u64>,
    /// Index of first free cell, or -1. Free cells thread their next
    /// index+1 through their first word (0 = end).
    free_head: i32,
    live: usize,
}

impl PageMeta {
    fn cell_ptr(&self, index: usize) -> usize {
        self.base + index * self.cell_size
    }

    fn take_free(&mut self) -> Option<*mut u8> {
        if self.free_head < 0 {
            return None;
        }
        let index = self.free_head as usize;
        let ptr = self.cell_ptr(index) as *mut u8;
        // SAFETY: free cells hold their free-list link in word 0.
        let next = unsafe { std::ptr::read(ptr as *const u64) };
        debug_assert!(next as usize <= self.cells, "GC free list corrupted");
        self.free_head = if next == 0 { -1 } else { (next - 1) as i32 };
        set_bit(&mut self.allocated, index, true);
        self.live += 1;
        // SAFETY: the cell lies inside this page and is now owned by the caller.
        unsafe { std::ptr::write_bytes(ptr, 0, self.cell_size) };
        Some(ptr)
    }
}

/// Allocation state for one (descriptor, size class).
#[derive(Default)]
struct ClassPages {
    current: Option<u32>,
    /// Pages with free cells after the last sweep, other than `current`.
    avail: Vec<u32>,
}

struct LargeObj {
    size: usize,
    desc: DescId,
    marked: bool,
    waiters: bool,
    /// See `PageMeta::zombies`.
    zombie: bool,
}

#[derive(Clone, Copy)]
enum Loc {
    Small { page: u32, index: u32 },
    Large,
}

/// A resolved cell: start address, descriptor, and where its bits live.
#[derive(Clone, Copy)]
struct CellRef {
    start: usize,
    desc: DescId,
    loc: Loc,
}

/// An edge whose value is reachable once its conditions are marked.
struct CondEdge {
    owner: Option<CellRef>,
    key: Option<CellRef>,
    value: EdgeValue,
    fired: bool,
    /// In-cell entry words to zero if the key dies (from a [`WeakFn`]).
    slots: Option<(*mut u64, *mut u64)>,
}

#[derive(Clone, Copy)]
enum EdgeValue {
    /// A cell address or tagged word.
    Word(u64),
    /// Ask the [`MarkClient`] for the side-table edges of this owner.
    SideSlots(usize),
}

/// The embedder's part of marking (S2: the interpreter, whose state the heap cannot see).
/// Called while the heap is mutably borrowed: implementations report words and must neither
/// allocate on this heap nor mutate engine state.
pub(crate) trait MarkClient {
    /// Strong roots: engine state, active VM frames, host roots.
    fn roots(&self, sink: &mut dyn FnMut(u64));
    /// Owners whose side-table edges are traced once (and only if) the owner is marked —
    /// owner-keyed tables would otherwise keep every cycle through them alive.
    fn side_slot_owners(&self, owners: &mut dyn FnMut(usize));
    /// The side-table edges of `owner`, which has just been marked.
    fn trace_side_slots(&self, owner: usize, sink: &mut dyn FnMut(u64));
    /// Ephemeron entries kept in side tables (WeakMap/WeakSet data).
    fn ephemerons(&self, sink: &mut EdgeSink<'_>);
}

/// The client of a collection driven by the heap itself (registered roots only).
struct NoClient;

impl MarkClient for NoClient {
    fn roots(&self, _: &mut dyn FnMut(u64)) {}
    fn side_slot_owners(&self, _: &mut dyn FnMut(usize)) {}
    fn trace_side_slots(&self, _: usize, _: &mut dyn FnMut(u64)) {}
    fn ephemerons(&self, _: &mut EdgeSink<'_>) {}
}

/// Why a cell was marked: retention diagnostics (see [`Heap::set_retention_tracing`]).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum MarkSource {
    /// A registered root range or trace root, or the embedder's roots.
    #[default]
    Root,
    /// The native stack word (or spilled register) at this address.
    Stack(usize),
    /// An edge of the cell starting at this address.
    Cell(usize),
    /// A side-table slot of this owner.
    SideSlots(usize),
    /// A conditional edge (ephemeron, side-table edge source) and its conditions.
    Conditional { owner: usize, key: usize },
}

/// Per-collection marking state. Built fresh every cycle, so nothing in it
/// can outlive the cells it describes.
#[derive(Default)]
struct Marker {
    /// What is marking now, and (when retention tracing) each cell's first marker.
    source: MarkSource,
    parents: Option<FastMap<usize, MarkSource>>,
    worklist: Vec<(usize, DescId)>,
    edges: Vec<CondEdge>,
    /// Waiting edges keyed by cell start >> 4 (cells are 16-aligned).
    waiting: FastMap<usize, Vec<u32>>,
    ready: Vec<u32>,
    weak_slots: Vec<*mut u64>,
    trace_scratch: Vec<*const u8>,
    slot_scratch: Vec<*mut u64>,
    pair_scratch: Vec<(*mut u64, *mut u64)>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct GcStats {
    pub(crate) allocations: u64,
    pub(crate) collections: u64,
    pub(crate) cells_marked: u64,
    pub(crate) cells_swept: u64,
    pub(crate) pages_live: usize,
    /// Released pages kept for reuse (see `Heap::spare_pages`): memory the heap holds beyond
    /// its live pages.
    pub(crate) pages_spare: usize,
    pub(crate) live_cells: usize,
    pub(crate) live_bytes: usize,
    /// Collections that were due but postponed, counted once per pending
    /// request (not per allocation that found it still pending).
    pub(crate) deferred_collections: u64,
    /// Subset of `deferred_collections` postponed because the native stack
    /// could not be scanned soundly (see [`Heap::collect`]).
    pub(crate) deferred_native_stack: u64,
    /// Allocated bytes at which the next collection becomes due.
    pub(crate) trigger_bytes: usize,
}

pub(crate) struct Heap {
    descs: Vec<LayoutDesc>,
    /// Indexed by descriptor, then size class.
    classes: Vec<[ClassPages; SIZE_CLASSES.len()]>,
    /// Slots are reused after release, so page indices stay stable.
    pages: Vec<Option<PageMeta>>,
    free_page_slots: Vec<u32>,
    /// P1 S4: the memory of released pages, kept for reuse. Freeing each empty page to the
    /// allocator and taking a fresh 64 KiB-aligned block for the next fragments the allocator's
    /// arena; peak memory grew where pages empty out every cycle. Trimmed to `spare_page_cap`
    /// at the end of each sweep.
    spare_pages: Vec<usize>,
    /// Page number (address >> PAGE_SHIFT; page addresses would leave
    /// FxHash's low bits zero) to index in `pages`.
    page_map: FastMap<usize, u32>,
    larges: BTreeMap<usize, LargeObj>,
    /// Address span of all large objects: cheap rejection of stack words.
    large_span: (usize, usize),
    root_ranges: Rc<Registry<RootRangeEntry>>,
    trace_roots: Rc<Registry<TraceRootEntry>>,
    edge_sources: Rc<Registry<EdgeSourceEntry>>,
    sweep_hooks: Rc<Registry<SweepHookEntry>>,
    /// Shared with NoGcScope/BorrowScope guards so a guard can be alive
    /// while `alloc(&mut self)` runs.
    control: Rc<Control>,
    allocated_bytes: usize,
    trigger_bytes: usize,
    min_trigger_bytes: usize,
    growth_percent: usize,
    stress_every: usize,
    allocs_since_gc: usize,
    /// When false, allocation never collects: the embedder polls
    /// [`Heap::collection_due`] at its own safepoints, where it can supply
    /// roots the heap cannot see (the interpreter's state).
    collect_on_alloc: bool,
    /// The embedder's copy of [`Heap::collection_due`], as a word that code polls without
    /// calling into the heap (see [`Heap::attach_due_word`]).
    due_word: Option<NonNull<Cell<u64>>>,
    /// Allocated bytes past which [`Heap::alloc`] recomputes the due word: the byte trigger;
    /// under a stress interval the current total (the allocation count decides, so every
    /// allocation recomputes); `usize::MAX` once due (no allocation can make it false; marking,
    /// sweeping and the settings recompute it).
    due_check_bytes: usize,
    stats: GcStats,
    retention_tracing: bool,
    /// The last traced collection's mark sources, by cell start.
    last_parents: FastMap<usize, MarkSource>,
    /// The last traced collection's frame-pointer chain, (frame record, return address), from
    /// the collector's frame outwards (aarch64 retention diagnostics).
    last_frames: Vec<(usize, usize)>,
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}

impl Heap {
    pub(crate) fn new() -> Self {
        #[cfg(all(test, not(target_arch = "wasm32")))]
        crate::gc_test::scrub_stack_at_thread_exit();
        Self {
            descs: Vec::new(),
            classes: Vec::new(),
            pages: Vec::new(),
            free_page_slots: Vec::new(),
            spare_pages: Vec::new(),
            page_map: FastMap::default(),
            larges: BTreeMap::new(),
            large_span: (usize::MAX, 0),
            root_ranges: Rc::default(),
            trace_roots: Rc::default(),
            edge_sources: Rc::default(),
            sweep_hooks: Rc::default(),
            control: Rc::default(),
            allocated_bytes: 0,
            trigger_bytes: DEFAULT_MIN_TRIGGER,
            min_trigger_bytes: DEFAULT_MIN_TRIGGER,
            growth_percent: DEFAULT_GROWTH_PERCENT,
            stress_every: 0,
            allocs_since_gc: 0,
            collect_on_alloc: true,
            due_word: None,
            due_check_bytes: DEFAULT_MIN_TRIGGER,
            stats: GcStats::default(),
            retention_tracing: false,
            last_parents: FastMap::default(),
            last_frames: Vec::new(),
        }
    }

    pub(crate) fn register_desc(&mut self, desc: LayoutDesc) -> DescId {
        let id = DescId(self.descs.len() as u32);
        self.descs.push(desc);
        self.classes
            .push(std::array::from_fn(|_| ClassPages::default()));
        id
    }

    /// Register a Rust-heap root range of tagged words (VM operand stacks,
    /// frame slot arrays). Words are recognized like stack words: anything
    /// whose payload addresses a cell, interior addresses included.
    ///
    /// # Safety
    /// `[lo, hi)` must stay readable, u64-aligned memory until the handle is
    /// dropped or re-pointed with `RootRange::set`.
    pub(crate) unsafe fn add_root_range(
        &self,
        name: &'static str,
        lo: *const u64,
        hi: *const u64,
    ) -> RootRange {
        self.root_ranges.insert(RootRangeEntry { name, lo, hi })
    }

    /// Register a traced Rust-heap container root (Decision 2 registry).
    /// Its edges are unconditional; data owned by a cell belongs in an
    /// edge source instead, or it would outlive (and leak) its owner.
    pub(crate) fn add_trace_root(
        &self,
        name: &'static str,
        trace: impl FnMut(&mut dyn FnMut(*const u8)) + 'static,
    ) -> TraceRoot {
        self.trace_roots.insert(TraceRootEntry {
            name,
            trace: Box::new(trace),
        })
    }

    /// Register a side table whose edges are conditional on cells (Decision
    /// 2): Map/Set data keyed by owner, WeakMap entries, FinalizationRegistry
    /// held values. Reported once per collection, at the start of marking.
    pub(crate) fn add_edge_source(
        &self,
        name: &'static str,
        report: impl FnMut(&mut EdgeSink<'_>) + 'static,
    ) -> EdgeSource {
        self.edge_sources.insert(EdgeSourceEntry {
            name,
            report: Box::new(report),
        })
    }

    /// Register a pre-sweep hook for address-keyed side tables (I8). The
    /// hook receives `is_live(cell_start)`, valid for this collection.
    pub(crate) fn add_sweep_hook(
        &self,
        name: &'static str,
        hook: impl FnMut(&dyn Fn(*const u8) -> bool) + 'static,
    ) -> SweepHook {
        self.sweep_hooks.insert(SweepHookEntry {
            name,
            hook: Box::new(hook),
        })
    }

    /// Record why each cell is marked, for [`Heap::retention_path`] (diagnostics: costs a map
    /// entry per marked cell).
    pub(crate) fn set_retention_tracing(&mut self, on: bool) {
        self.retention_tracing = on;
        self.last_parents = FastMap::default();
    }

    /// For the last collection with retention tracing on: what marked the cell containing
    /// `addr`, then what marked that, back to a root or the native stack. Empty when the cell
    /// was not marked.
    pub(crate) fn retention_path(&self, addr: usize) -> Vec<(usize, &'static str, MarkSource)> {
        let mut path = Vec::new();
        let mut next = addr;
        while path.len() < 10_000 {
            let Some(cell) = self.locate(next as u64) else {
                break;
            };
            let Some(&source) = self.last_parents.get(&cell.start) else {
                break;
            };
            path.push((cell.start, self.descs[cell.desc.0 as usize].name, source));
            next = match source {
                MarkSource::Cell(parent) | MarkSource::SideSlots(parent) => parent,
                MarkSource::Conditional { owner, .. } if owner != 0 => owner,
                _ => break,
            };
        }
        path
    }

    /// The function owning the native-stack word at `addr` in the last traced collection: the
    /// return address its callee saved (symbolize with addr2line against the binary). `None`
    /// when no frame record covers `addr`.
    /// The return address into the function whose frame holds stack word `addr` in the last
    /// traced collection, and how far below that frame's record the word lies.
    pub(crate) fn frame_return_address(&self, addr: usize) -> Option<(usize, usize)> {
        // A frame record sits at the top of its frame: the owner of `addr` has the lowest
        // record at or above it; the record below that one holds the return into the owner.
        let index = self.last_frames.iter().position(|&(fp, _)| fp >= addr)?;
        let owner = self.last_frames[index].0;
        index
            .checked_sub(1)
            .map(|below| (self.last_frames[below].1, owner - addr))
    }

    pub(crate) fn set_stress_every(&mut self, allocs: usize) {
        self.stress_every = allocs;
        self.refresh_due();
    }

    /// See the `collect_on_alloc` field.
    pub(crate) fn set_collect_on_alloc(&mut self, collect: bool) {
        self.collect_on_alloc = collect;
    }

    /// Whether a collection is due: the allocation budget is spent, the
    /// stress interval elapsed, or an earlier request was deferred.
    pub(crate) fn collection_due(&self) -> bool {
        self.control.pending_collect.get()
            || self.allocated_bytes > self.trigger_bytes
            || (self.stress_every > 0 && self.allocs_since_gc >= self.stress_every)
    }

    /// Keep `word` equal to [`Heap::collection_due`] (1 or 0) from now on, for code that polls
    /// for a collection without calling into the heap (the native JIT's polls; see
    /// `value::LiveCount`). The heap is the word's only writer. An allocation that spends the
    /// byte budget or reaches the stress interval sets it ([`Heap::alloc`]), and so does a
    /// deferred request; marking (which takes the pending request), the end of every sweep (a
    /// new budget and interval, and any request that arrived meanwhile) and the settings
    /// recompute it. Nothing else changes what `collection_due` reads.
    ///
    /// # Safety
    /// `word` must stay valid at its address, and written by no one else, for as long as the
    /// heap lives.
    pub(crate) unsafe fn attach_due_word(&mut self, word: NonNull<Cell<u64>>) {
        self.due_word = Some(word);
        self.refresh_due();
    }

    /// Collections run so far (without the page walk of [`Heap::stats`]).
    #[cfg(test)]
    pub(crate) fn collections(&self) -> u64 {
        self.stats.collections
    }

    /// Bytes allocated past the trigger: how far allocation has run beyond the budget.
    #[cfg(test)]
    pub(crate) fn budget_overshoot(&self) -> usize {
        self.allocated_bytes.saturating_sub(self.trigger_bytes)
    }

    /// Whether the attached due word (if any) agrees with [`Heap::collection_due`]: the
    /// invariant [`Heap::attach_due_word`] promises, for debug assertions.
    pub(crate) fn due_word_agrees(&self) -> bool {
        // SAFETY: `attach_due_word`'s contract.
        self.due_word
            .is_none_or(|word| (unsafe { word.as_ref() }.get() != 0) == self.collection_due())
    }

    /// Recompute the due word and the allocation check after an input of
    /// [`Heap::collection_due`] changed.
    #[cold]
    fn refresh_due(&mut self) {
        let due = self.collection_due();
        self.due_check_bytes = if due {
            usize::MAX
        } else if self.stress_every > 0 {
            self.allocated_bytes
        } else {
            self.trigger_bytes
        };
        if let Some(word) = self.due_word {
            // SAFETY: `attach_due_word`'s contract.
            unsafe { word.as_ref() }.set(u64::from(due));
        }
    }

    /// The next collection is due once allocated bytes exceed
    /// max(`min_bytes`, live × (1 + `growth_percent`/100)), so marking work
    /// per allocated byte stays bounded as the live heap grows.
    pub(crate) fn set_heap_growth(&mut self, min_bytes: usize, growth_percent: usize) {
        self.min_trigger_bytes = min_bytes;
        self.growth_percent = growth_percent;
        self.trigger_bytes = min_bytes;
        self.refresh_due();
    }

    pub(crate) fn stats(&self) -> GcStats {
        let mut s = self.stats;
        let pages = self.pages.iter().flatten();
        s.pages_live = pages.clone().count();
        s.pages_spare = self.spare_pages.len();
        s.live_cells = pages.clone().map(|p| p.live).sum::<usize>() + self.larges.len();
        s.live_bytes = pages.map(|p| p.live * p.cell_size).sum::<usize>()
            + self.larges.values().map(|l| l.size).sum::<usize>();
        s.trigger_bytes = self.trigger_bytes;
        s
    }

    /// Enter a bounded no-GC region (I9). Allocation proceeds inside;
    /// collection requests are deferred, and a deferred collection runs at
    /// the next allocation or collection request after the outermost
    /// region exits, never at region exit itself (exit is not a safepoint).
    pub(crate) fn nogc(&self) -> NoGcScope {
        self.control
            .nogc_depth
            .set(self.control.nogc_depth.get() + 1);
        NoGcScope {
            control: Rc::clone(&self.control),
        }
    }

    /// I9 hook for S2's GcCell: hold one scope per outstanding Ref/RefMut.
    /// A collection requested meanwhile asserts in debug builds and defers
    /// in release builds.
    pub(crate) fn borrow_scope(&self) -> BorrowScope {
        self.control.borrows.set(self.control.borrows.get() + 1);
        BorrowScope {
            control: Rc::clone(&self.control),
        }
    }

    /// Debug check for task boundaries and teardown: a leaked guard would
    /// otherwise defer collection forever.
    pub(crate) fn debug_assert_quiescent(&self) {
        debug_assert_eq!(
            self.control.nogc_depth.get(),
            0,
            "I9: no-GC region still open"
        );
        debug_assert_eq!(
            self.control.borrows.get(),
            0,
            "I9: GcCell borrow still outstanding"
        );
        debug_assert!(!self.control.collecting.get(), "GC collection in progress");
    }

    /// A collection was requested but deferred; it runs at the next
    /// allocation or explicit request (wasm32: the next VM safepoint).
    pub(crate) fn collection_pending(&self) -> bool {
        self.control.pending_collect.get()
    }

    /// Allocate a zeroed cell for `desc`. May first collect, conservatively
    /// (see [`Heap::collect`]). The caller initializes, then publishes (I9).
    ///
    /// Between [`Heap::start_marking`] and [`Heap::finish_collection`] the
    /// cell is allocated marked (black): marking is over, so the cell survives
    /// this sweep untraced. Code running in that window must not store a
    /// reference to an unmarked cell into a new or surviving one.
    pub(crate) fn alloc(&mut self, desc: DescId, size: usize) -> *mut u8 {
        debug_assert!(
            (desc.0 as usize) < self.descs.len(),
            "unregistered GC descriptor"
        );
        let class = size_class(size);
        let cost = class.map_or(size, |class| SIZE_CLASSES[class]);
        let collecting = self.control.collecting.get();
        if !collecting {
            self.maybe_collect_for_alloc(cost);
        }
        let ptr = match class {
            Some(class) => self.alloc_small(desc, class),
            None => self.alloc_large(desc, size),
        };
        if collecting {
            let cell = self
                .locate(ptr as u64)
                .expect("a fresh cell resolves to itself");
            self.mark(&cell);
        }
        self.stats.allocations += 1;
        self.allocated_bytes = self.allocated_bytes.saturating_add(cost);
        self.allocs_since_gc += 1;
        // The due word's one compare: an allocation past the budget (any allocation under a
        // stress interval) recomputes it, out of line.
        if self.allocated_bytes > self.due_check_bytes {
            self.refresh_due();
        }
        ptr
    }

    fn maybe_collect_for_alloc(&mut self, upcoming: usize) {
        // Stress collects *before* an allocation: collecting after handing
        // out the cell would free it before the caller can root it.
        let due = self.control.pending_collect.get()
            || self.allocated_bytes.saturating_add(upcoming) > self.trigger_bytes
            || (self.stress_every > 0 && self.allocs_since_gc >= self.stress_every);
        if due && self.collect_on_alloc {
            self.collect();
        }
    }

    fn alloc_small(&mut self, desc: DescId, class: usize) -> *mut u8 {
        loop {
            let pages = &mut self.classes[desc.0 as usize][class];
            if pages.current.is_none() {
                pages.current = pages.avail.pop();
            }
            let Some(index) = pages.current else {
                break;
            };
            if let Some(ptr) = self.pages[index as usize]
                .as_mut()
                .and_then(PageMeta::take_free)
            {
                return ptr;
            }
            self.classes[desc.0 as usize][class].current = None;
        }
        let index = self.new_page(desc, class);
        self.classes[desc.0 as usize][class].current = Some(index);
        self.pages[index as usize]
            .as_mut()
            .and_then(PageMeta::take_free)
            .expect("fresh page has free cells")
    }

    fn new_page(&mut self, desc: DescId, class: usize) -> u32 {
        let cell_size = SIZE_CLASSES[class];
        let base = match self.spare_pages.pop() {
            Some(base) => {
                let base = base as *mut u8;
                // SAFETY: a spare page is a `page_layout()` block no cell occupies; zeroed like a
                // fresh one.
                unsafe { std::ptr::write_bytes(base, 0, PAGE_SIZE) };
                base
            }
            None => {
                // SAFETY: non-zero size; null is checked below.
                let base = unsafe { alloc_zeroed(page_layout()) };
                assert!(!base.is_null(), "GC page allocation failed");
                base
            }
        };
        let cells = PAGE_SIZE / cell_size;
        // Thread the free list: cell i -> i+1 (stored as i+2), last -> 0.
        for i in 0..cells {
            let next = if i + 1 < cells { (i + 2) as u64 } else { 0 };
            // SAFETY: the cell lies inside the fresh page.
            unsafe { std::ptr::write(base.add(i * cell_size) as *mut u64, next) };
        }
        let words = cells.div_ceil(64);
        let meta = PageMeta {
            base: base as usize,
            cell_size,
            cells,
            desc,
            class,
            marks: vec![0; words],
            allocated: vec![0; words],
            waiters: vec![0; words],
            zombies: vec![0; words],
            free_head: 0,
            live: 0,
        };
        let index = match self.free_page_slots.pop() {
            Some(index) => {
                self.pages[index as usize] = Some(meta);
                index
            }
            None => {
                self.pages.push(Some(meta));
                (self.pages.len() - 1) as u32
            }
        };
        self.page_map.insert(base as usize >> PAGE_SHIFT, index);
        index
    }

    fn alloc_large(&mut self, desc: DescId, size: usize) -> *mut u8 {
        // SAFETY: size > 4 KiB; null is checked below.
        let ptr = unsafe { alloc_zeroed(large_layout(size)) };
        assert!(!ptr.is_null(), "GC large allocation failed");
        let start = ptr as usize;
        self.larges.insert(
            start,
            LargeObj {
                size,
                desc,
                marked: false,
                waiters: false,
                zombie: false,
            },
        );
        self.large_span = (
            self.large_span.0.min(start),
            self.large_span.1.max(start + size),
        );
        ptr
    }

    /// Resolve a candidate word (raw pointer or tagged word, interior
    /// addresses included) to its cell. `None` for non-heap words.
    fn locate(&self, word: u64) -> Option<CellRef> {
        let addr = (word & PAYLOAD_MASK) as usize;
        if addr < 4096 {
            return None;
        }
        if let Some(&index) = self.page_map.get(&(addr >> PAGE_SHIFT)) {
            let page = self.pages[index as usize].as_ref()?;
            let cell = (addr - page.base) / page.cell_size;
            return (cell < page.cells).then(|| CellRef {
                start: page.cell_ptr(cell),
                desc: page.desc,
                loc: Loc::Small {
                    page: index,
                    index: cell as u32,
                },
            });
        }
        if !(self.large_span.0..self.large_span.1).contains(&addr) {
            return None;
        }
        let (&start, large) = self.larges.range(..=addr).next_back()?;
        (addr < start + large.size).then_some(CellRef {
            start,
            desc: large.desc,
            loc: Loc::Large,
        })
    }

    /// Resolve `ptr` to the start of its cell, accepting interior addresses.
    pub(crate) fn resolve(&self, word: u64) -> Option<*const u8> {
        self.locate(word).map(|cell| cell.start as *const u8)
    }

    fn is_marked(&self, cell: &CellRef) -> bool {
        match cell.loc {
            Loc::Small { page, index } => self.pages[page as usize]
                .as_ref()
                .is_some_and(|p| bit(&p.marks, index as usize)),
            Loc::Large => self.larges.get(&cell.start).is_some_and(|l| l.marked),
        }
    }

    /// Set the mark of an allocated cell. `Some(has_waiters)` when newly
    /// marked; `None` when already marked or free (conservative or stale
    /// hits on free cells are ignored).
    fn mark(&mut self, cell: &CellRef) -> Option<bool> {
        match cell.loc {
            Loc::Small { page, index } => {
                let page = self.pages[page as usize].as_mut()?;
                let index = index as usize;
                if !bit(&page.allocated, index)
                    || bit(&page.zombies, index)
                    || bit(&page.marks, index)
                {
                    return None;
                }
                set_bit(&mut page.marks, index, true);
                Some(bit(&page.waiters, index))
            }
            Loc::Large => {
                let large = self.larges.get_mut(&cell.start)?;
                if large.marked || large.zombie {
                    return None;
                }
                large.marked = true;
                Some(large.waiters)
            }
        }
    }

    fn mark_cell(&mut self, cell: CellRef, m: &mut Marker) {
        if let Some(has_waiters) = self.mark(&cell) {
            if let Some(parents) = &mut m.parents {
                parents.insert(cell.start, m.source);
            }
            self.stats.cells_marked += 1;
            m.worklist.push((cell.start, cell.desc));
            if has_waiters {
                if let Some(edges) = m.waiting.remove(&(cell.start >> 4)) {
                    m.ready.extend(edges);
                }
            }
        }
    }

    fn mark_word(&mut self, word: u64, m: &mut Marker) {
        if let Some(cell) = self.locate(word) {
            self.mark_cell(cell, m);
        }
    }

    /// Request a collection with conservative native roots (Decision 0):
    /// registered roots plus this thread's registered native stack chain
    /// and a register snapshot. Returns false when the request is deferred
    /// (pending until the next allocation or request): inside a no-GC
    /// region; while a GcCell borrow is outstanding (debug builds assert,
    /// I9); or when the native stack cannot be scanned soundly — no stack
    /// registered on this thread (see [`enter_native_stack`]), SP outside
    /// every registered region (e.g. an unregistered `stacker::grow`
    /// segment), or no scanner for this target (wasm32).
    pub(crate) fn collect(&mut self) -> bool {
        if self.blocked() {
            return self.defer(false);
        }
        if !native::scannable(native::current_sp()) {
            return self.defer(true);
        }
        self.mark_phase(true, &NoClient);
        self.sweep_phase(&|_| false);
        true
    }

    /// Collect at a precise safepoint: registered roots only, no native
    /// stack or registers. The caller guarantees no heap reference is held
    /// solely in native frames or registers (wasm32 VM safepoints per the
    /// S0 decision, task boundaries, tests). Deferred like [`Heap::collect`]
    /// inside no-GC regions and while borrows are outstanding.
    pub(crate) fn collect_at_safepoint(&mut self) -> bool {
        if self.blocked() {
            return self.defer(false);
        }
        self.mark_phase(false, &NoClient);
        self.sweep_phase(&|_| false);
        true
    }

    /// Phase 1 of an embedder-driven collection: mark from the registered roots, `client`
    /// and, when `native_roots`, this thread's native stack (deferred as for
    /// [`Heap::collect`] when it cannot be scanned soundly). Marks stay set until
    /// [`Heap::finish_collection`]; meanwhile the embedder sweeps its own tables using
    /// [`Heap::dead_cells`] and [`Heap::is_dead_target`]. Returns false when deferred, in
    /// which case `finish_collection` must not be called.
    pub(crate) fn start_marking(&mut self, native_roots: bool, client: &dyn MarkClient) -> bool {
        if self.blocked() {
            return self.defer(false);
        }
        if native_roots && !native::scannable(native::current_sp()) {
            return self.defer(true);
        }
        self.mark_phase(native_roots, client);
        true
    }

    /// Phase 2: run sweep hooks, finalize the newly dead cells and free them, except those
    /// whose address `pinned` reports still held by a weak handle (they become zombies until
    /// a later collection finds them unpinned).
    pub(crate) fn finish_collection(&mut self, pinned: &dyn Fn(usize) -> bool) {
        assert!(
            self.control.collecting.get(),
            "finish_collection without start_marking"
        );
        self.sweep_phase(pinned);
    }

    /// The allocated cells of `desc` the current marking left unmarked (between
    /// [`Heap::start_marking`] and [`Heap::finish_collection`]).
    pub(crate) fn dead_cells(&self, desc: DescId) -> Vec<usize> {
        let mut dead = Vec::new();
        for page in self.pages.iter().flatten().filter(|page| page.desc == desc) {
            for i in 0..page.cells {
                if bit(&page.allocated, i) && !bit(&page.marks, i) && !bit(&page.zombies, i) {
                    dead.push(page.cell_ptr(i));
                }
            }
        }
        for (&start, large) in &self.larges {
            if large.desc == desc && !large.marked && !large.zombie {
                dead.push(start);
            }
        }
        dead
    }

    /// The allocated, non-zombie cells of `desc`: after a collection, the live ones.
    /// How many cells [`Heap::live_cells`] would list.
    pub(crate) fn live_count(&self, desc: DescId) -> usize {
        let paged: usize = self
            .pages
            .iter()
            .flatten()
            .filter(|page| page.desc == desc)
            .map(|page| {
                (0..page.cells)
                    .filter(|&i| bit(&page.allocated, i) && !bit(&page.zombies, i))
                    .count()
            })
            .sum();
        paged
            + self
                .larges
                .values()
                .filter(|large| large.desc == desc && !large.zombie)
                .count()
    }

    pub(crate) fn live_cells(&self, desc: DescId) -> Vec<usize> {
        let mut live = Vec::new();
        for page in self.pages.iter().flatten().filter(|page| page.desc == desc) {
            for i in 0..page.cells {
                if bit(&page.allocated, i) && !bit(&page.zombies, i) {
                    live.push(page.cell_ptr(i));
                }
            }
        }
        for (&start, large) in &self.larges {
            if large.desc == desc && !large.zombie {
                live.push(start);
            }
        }
        live
    }

    /// Whether `addr` (interior addresses included) lies in a cell of this heap that the
    /// current marking left unmarked: a weak reference to it must be cleared.
    pub(crate) fn is_dead_target(&self, addr: usize) -> bool {
        self.locate(addr as u64)
            .is_some_and(|cell| !self.is_marked(&cell))
    }

    fn blocked(&self) -> bool {
        if self.control.nogc_depth.get() > 0 {
            return true;
        }
        let borrows = self.control.borrows.get();
        if borrows > 0 && cfg!(debug_assertions) {
            panic!(
                "I9 violation: collection requested with {borrows} outstanding GcCell \
                 borrow(s); allocate inside a no-GC region or after the borrow ends"
            );
        }
        borrows > 0
    }

    fn defer(&mut self, native_stack: bool) -> bool {
        if !self.control.pending_collect.replace(true) {
            self.stats.deferred_collections += 1;
            if native_stack {
                self.stats.deferred_native_stack += 1;
            }
            self.refresh_due();
        }
        false
    }

    fn mark_phase(&mut self, native_roots: bool, client: &dyn MarkClient) {
        assert!(
            !self.control.collecting.replace(true),
            "re-entrant GC collection"
        );
        self.control.pending_collect.set(false);
        self.refresh_due();
        self.stats.collections += 1;
        let mut m = Marker {
            parents: self.retention_tracing.then(FastMap::default),
            ..Marker::default()
        };
        self.mark_root_ranges(&mut m);
        self.mark_trace_roots(&mut m);
        client.roots(&mut |word| self.mark_word(word, &mut m));
        if native_roots {
            self.mark_native_stack(&mut m);
        }
        m.source = MarkSource::Root;
        self.add_source_edges(&mut m, client);
        client.side_slot_owners(&mut |owner| {
            // An owner outside this heap is not a condition: its slots are traced now.
            let condition = self.locate(owner as u64);
            self.add_edge(
                CondEdge {
                    owner: condition,
                    key: None,
                    value: EdgeValue::SideSlots(owner),
                    fired: false,
                    slots: None,
                },
                &mut m,
                client,
            );
        });
        self.drain(&mut m, client);
        self.clear_dead_weak(&m);
        #[cfg(debug_assertions)]
        self.verify(&m);
        if let Some(parents) = m.parents.take() {
            self.last_parents = parents;
        }
    }

    fn sweep_phase(&mut self, pinned: &dyn Fn(usize) -> bool) {
        self.run_sweep_hooks();
        let live_bytes = self.sweep(pinned);
        self.allocated_bytes = live_bytes;
        let budget =
            live_bytes.saturating_add(live_bytes.saturating_mul(self.growth_percent) / 100);
        self.trigger_bytes = self.min_trigger_bytes.max(budget);
        self.trim_spare_pages();
        self.allocs_since_gc = 0;
        self.control.collecting.set(false);
        // A new budget and interval; a request deferred meanwhile keeps the word set.
        self.refresh_due();
    }

    fn mark_root_ranges(&mut self, m: &mut Marker) {
        Rc::clone(&self.root_ranges).for_each(|range| {
            let mut p = range.lo;
            while p < range.hi {
                // SAFETY: registered ranges are readable (add_root_range contract).
                let word = unsafe { std::ptr::read(p) };
                self.mark_word(word, m);
                p = p.wrapping_add(1);
            }
        });
    }

    fn mark_trace_roots(&mut self, m: &mut Marker) {
        let mut edges: Vec<*const u8> = Vec::new();
        Rc::clone(&self.trace_roots).for_each(|root| (root.trace)(&mut |cell| edges.push(cell)));
        for cell in edges {
            self.mark_word(cell as u64, m);
        }
    }

    /// Decision 0 spill protocol. Never inlined: callee-saved registers are
    /// spilled into this frame, then the scan starts at this frame's SP, so
    /// every value a caller keeps in a callee-saved register is in `regs`,
    /// in a prologue save area above SP, or in a caller frame. Caller-saved
    /// registers are dead or spilled by the caller at this call.
    #[inline(never)]
    fn mark_native_stack(&mut self, m: &mut Marker) {
        let mut regs = [0u64; native::SPILL_WORDS];
        native::spill_registers(&mut regs);
        let sp = native::current_sp();
        #[cfg(target_arch = "aarch64")]
        if self.retention_tracing {
            self.last_frames.clear();
            let mut fp: usize;
            // SAFETY: reads x29 only.
            unsafe {
                std::arch::asm!("mov {}, x29", out(reg) fp, options(nomem, nostack, preserves_flags))
            };
            while fp > sp && fp.is_multiple_of(8) && self.last_frames.len() < 4096 {
                // SAFETY: a frame record of this thread's stack: [saved x29, saved x30].
                let (next, lr) = unsafe {
                    (
                        std::ptr::read(fp as *const usize),
                        std::ptr::read((fp + 8) as *const usize),
                    )
                };
                self.last_frames.push((fp, lr));
                if next <= fp {
                    break;
                }
                fp = next;
            }
        }
        let ranges = native::scan_ranges(sp).expect("collect() checked the native stack");
        for (lo, hi) in ranges {
            let mut p = lo;
            while p + 8 <= hi {
                // SAFETY: registered stack regions are mapped (enter_native_stack
                // and grow contracts); volatile keeps the compiler from reasoning
                // about reads of other frames.
                let word = unsafe { std::ptr::read_volatile(p as *const u64) };
                m.source = MarkSource::Stack(p);
                self.mark_word(word, m);
                p += 8;
            }
        }
        std::hint::black_box(&regs);
    }

    fn add_source_edges(&mut self, m: &mut Marker, client: &dyn MarkClient) {
        let mut reported = Vec::new();
        Rc::clone(&self.edge_sources).for_each(|source| {
            (source.report)(&mut EdgeSink {
                edges: &mut reported,
            })
        });
        client.ephemerons(&mut EdgeSink {
            edges: &mut reported,
        });
        for edge in reported {
            // An owner or key outside this heap is not a condition.
            let owner = self.locate(edge.owner);
            let key = edge.key.and_then(|key| self.locate(key));
            self.add_edge(
                CondEdge {
                    owner,
                    key,
                    value: EdgeValue::Word(edge.value),
                    fired: false,
                    slots: None,
                },
                m,
                client,
            );
        }
    }

    fn drain(&mut self, m: &mut Marker, client: &dyn MarkClient) {
        loop {
            while let Some((cell, desc)) = m.worklist.pop() {
                m.source = MarkSource::Cell(cell);
                let desc = &self.descs[desc.0 as usize];
                let (trace, weak) = (desc.trace, desc.weak);
                let mut edges = std::mem::take(&mut m.trace_scratch);
                edges.clear();
                trace(cell as *const u8, &mut |edge| edges.push(edge));
                for &edge in &edges {
                    self.mark_word(edge as u64, m);
                }
                m.trace_scratch = edges;
                if let Some(weak) = weak {
                    self.visit_weak(cell, weak, m, client);
                }
            }
            let Some(edge) = m.ready.pop() else {
                break;
            };
            self.try_fire(edge, m, client);
        }
    }

    fn visit_weak(&mut self, cell: usize, weak: WeakFn, m: &mut Marker, client: &dyn MarkClient) {
        let mut slots = std::mem::take(&mut m.slot_scratch);
        let mut pairs = std::mem::take(&mut m.pair_scratch);
        slots.clear();
        pairs.clear();
        weak(
            cell as *mut u8,
            &mut WeakSink {
                weak_slots: &mut slots,
                ephemerons: &mut pairs,
            },
        );
        debug_assert!(
            slots
                .iter()
                .chain(pairs.iter().flat_map(|(key, value)| [key, value]))
                .all(|&slot| self.locate(slot as u64).is_some_and(|c| c.start == cell)),
            "WeakFn reported a slot outside its cell"
        );
        m.weak_slots.extend_from_slice(&slots);
        for &(key_slot, value_slot) in &pairs {
            // SAFETY: the descriptor reported words inside this marked cell.
            let (key, value) = unsafe { (std::ptr::read(key_slot), std::ptr::read(value_slot)) };
            if key == 0 {
                continue;
            }
            let key = self.locate(key);
            self.add_edge(
                CondEdge {
                    owner: None,
                    key,
                    value: EdgeValue::Word(value),
                    fired: false,
                    slots: Some((key_slot, value_slot)),
                },
                m,
                client,
            );
        }
        m.slot_scratch = slots;
        m.pair_scratch = pairs;
    }

    fn add_edge(&mut self, edge: CondEdge, m: &mut Marker, client: &dyn MarkClient) {
        let index = m.edges.len() as u32;
        m.edges.push(edge);
        self.try_fire(index, m, client);
    }

    /// Mark the edge's value if all its conditions are marked; otherwise
    /// wait on the first unmarked one. Each edge waits at most twice and
    /// fires at most once, so conditional marking is linear — no fixed-point
    /// rescans, and a condition that can never be marked (a free cell) just
    /// leaves its edge unfired.
    fn try_fire(&mut self, index: u32, m: &mut Marker, client: &dyn MarkClient) {
        let edge = &m.edges[index as usize];
        if edge.fired {
            return;
        }
        let (conditions, value) = ([edge.owner, edge.key], edge.value);
        for condition in conditions.into_iter().flatten() {
            if !self.is_marked(&condition) {
                self.wait_on(&condition, index, m);
                return;
            }
        }
        m.edges[index as usize].fired = true;
        let start = |cell: Option<CellRef>| cell.map_or(0, |cell| cell.start);
        let saved = m.source;
        match value {
            EdgeValue::Word(word) => {
                m.source = MarkSource::Conditional {
                    owner: start(conditions[0]),
                    key: start(conditions[1]),
                };
                self.mark_word(word, m);
            }
            EdgeValue::SideSlots(owner) => {
                m.source = MarkSource::SideSlots(owner);
                client.trace_side_slots(owner, &mut |word| self.mark_word(word, m));
            }
        }
        m.source = saved;
    }

    fn wait_on(&mut self, cell: &CellRef, edge: u32, m: &mut Marker) {
        match cell.loc {
            Loc::Small { page, index } => {
                if let Some(page) = self.pages[page as usize].as_mut() {
                    set_bit(&mut page.waiters, index as usize, true);
                }
            }
            Loc::Large => {
                if let Some(large) = self.larges.get_mut(&cell.start) {
                    large.waiters = true;
                }
            }
        }
        m.waiting.entry(cell.start >> 4).or_default().push(edge);
    }

    /// After marking: zero weak slots whose targets died and in-cell
    /// ephemeron entries whose keys died. Holders are marked, so the words
    /// written belong to surviving cells, reported this cycle.
    fn clear_dead_weak(&self, m: &Marker) {
        for &slot in &m.weak_slots {
            // SAFETY: reported by the WeakFn of a marked cell.
            let word = unsafe { std::ptr::read(slot) };
            if self.locate(word).is_some_and(|cell| !self.is_marked(&cell)) {
                // SAFETY: as above.
                unsafe { std::ptr::write(slot, 0) };
            }
        }
        for edge in &m.edges {
            if let (false, Some((key, value))) = (edge.fired, edge.slots) {
                // SAFETY: as above.
                unsafe {
                    std::ptr::write(key, 0);
                    std::ptr::write(value, 0);
                }
            }
        }
    }

    /// Structural check: every strong edge of every marked cell, and the
    /// value of every fired conditional edge, resolves to a marked cell.
    /// Cannot detect descriptor *omissions* (that proof is the registry
    /// audit vs the S0 inventory — Decision 2).
    fn verify(&self, m: &Marker) {
        let mut edges: Vec<*const u8> = Vec::new();
        let mut check = |from: usize, desc: DescId| {
            let desc = &self.descs[desc.0 as usize];
            edges.clear();
            (desc.trace)(from as *const u8, &mut |edge| edges.push(edge));
            for &edge in &edges {
                if edge.is_null() {
                    continue;
                }
                match self.locate(edge as u64) {
                    Some(cell) => assert!(
                        self.is_marked(&cell),
                        "heap verifier: marked cell {from:#x} ({}) references unmarked {:#x}",
                        desc.name,
                        cell.start
                    ),
                    None => panic!(
                        "heap verifier: marked cell {from:#x} ({}) references non-heap {edge:p}",
                        desc.name
                    ),
                }
            }
        };
        for page in self.pages.iter().flatten() {
            for i in 0..page.cells {
                if bit(&page.marks, i) {
                    check(page.cell_ptr(i), page.desc);
                }
            }
        }
        for (&start, large) in &self.larges {
            if large.marked {
                check(start, large.desc);
            }
        }
        for edge in m.edges.iter().filter(|edge| edge.fired) {
            let EdgeValue::Word(value) = edge.value else {
                continue;
            };
            if let Some(cell) = self.locate(value) {
                assert!(
                    self.is_marked(&cell),
                    "heap verifier: conditional edge references unmarked {:#x}",
                    cell.start
                );
            }
        }
    }

    /// Exact liveness for side-table keys: `addr` must be a cell start.
    fn is_live(&self, addr: usize) -> bool {
        self.locate(addr as u64)
            .is_some_and(|cell| cell.start == addr && self.is_marked(&cell))
    }

    fn run_sweep_hooks(&self) {
        Rc::clone(&self.sweep_hooks)
            .for_each(|entry| (entry.hook)(&|cell| self.is_live(cell as usize)));
    }

    /// Finalize, poison (debug) and free unmarked cells, release empty
    /// pages, clear marks. Dead cells `pinned` by weak handles stay allocated
    /// as zombies. Returns the live bytes that remain.
    fn sweep(&mut self, pinned: &dyn Fn(usize) -> bool) -> usize {
        let mut live_bytes = 0usize;
        let mut swept = 0u64;
        for index in 0..self.pages.len() {
            let Some(page) = self.pages[index].as_mut() else {
                continue;
            };
            let finalize = self.descs[page.desc.0 as usize].finalize;
            let mut head: i32 = -1;
            let mut live = 0usize;
            for i in 0..page.cells {
                if bit(&page.marks, i) {
                    live += 1;
                    continue;
                }
                let cell = page.cell_ptr(i) as *mut u64;
                if bit(&page.allocated, i) {
                    let pin = pinned(cell as usize);
                    if !bit(&page.zombies, i) {
                        swept += 1;
                        if let Some(finalize) = finalize {
                            finalize(cell as *mut u8);
                        }
                        #[cfg(debug_assertions)]
                        for w in 1..page.cell_size / 8 {
                            // SAFETY: the dead cell lies inside this page.
                            unsafe { std::ptr::write(cell.add(w), POISON) };
                        }
                    }
                    if pin {
                        // Finalized, but its address stays reserved for weak handles.
                        set_bit(&mut page.zombies, i, true);
                        live += 1;
                        continue;
                    }
                    set_bit(&mut page.zombies, i, false);
                    set_bit(&mut page.allocated, i, false);
                }
                let link = if head < 0 { 0 } else { head as u64 + 1 };
                // SAFETY: as above; word 0 of a free cell is its free-list link.
                unsafe { std::ptr::write(cell, link) };
                head = i as i32;
            }
            page.free_head = head;
            page.live = live;
            page.marks.fill(0);
            page.waiters.fill(0);
            let cell_size = page.cell_size;
            if live == 0 {
                self.release_page(index);
            } else {
                live_bytes += live * cell_size;
            }
        }
        for class in self.classes.iter_mut().flatten() {
            class.avail.clear();
        }
        for (index, page) in self.pages.iter().enumerate() {
            let Some(page) = page else {
                continue;
            };
            let class = &mut self.classes[page.desc.0 as usize][page.class];
            if page.free_head >= 0 && class.current != Some(index as u32) {
                class.avail.push(index as u32);
            }
        }
        let descs = &self.descs;
        self.larges.retain(|&start, large| {
            if large.marked {
                large.marked = false;
                large.waiters = false;
                live_bytes += large.size;
                return true;
            }
            large.waiters = false;
            if !large.zombie {
                swept += 1;
                if let Some(finalize) = descs[large.desc.0 as usize].finalize {
                    finalize(start as *mut u8);
                }
                #[cfg(debug_assertions)]
                // SAFETY: the dead object spans `size` bytes from `start`.
                unsafe {
                    std::ptr::write_bytes(start as *mut u8, 0xdd, large.size)
                };
            }
            if pinned(start) {
                large.zombie = true;
                return true;
            }
            // SAFETY: allocated in alloc_large with this layout.
            unsafe { dealloc(start as *mut u8, large_layout(large.size)) };
            false
        });
        self.large_span = match (self.larges.first_key_value(), self.larges.last_key_value()) {
            (Some((&lo, _)), Some((&last, large))) => (lo, last + large.size),
            _ => (usize::MAX, 0),
        };
        self.stats.cells_swept += swept;
        live_bytes
    }

    fn release_page(&mut self, index: usize) {
        let page = self.pages[index].take().expect("live page slot");
        self.page_map.remove(&(page.base >> PAGE_SHIFT));
        let class = &mut self.classes[page.desc.0 as usize][page.class];
        if class.current == Some(index as u32) {
            class.current = None;
        }
        self.free_page_slots.push(index as u32);
        // Kept for now; the sweep trims the pool once it has set the next cycle's budget.
        self.spare_pages.push(page.base);
    }

    /// How many released pages to keep: as many as the allocation budget until the next
    /// collection can fill, and at least 64. A heap that empties and refills its pages each
    /// cycle takes them back from the pool, and one that shrank returns what its smaller budget
    /// will not use.
    fn spare_page_cap(&self) -> usize {
        let budget = self.trigger_bytes.saturating_sub(self.allocated_bytes);
        budget.div_ceil(PAGE_SIZE).max(64)
    }

    /// Free the spare pages beyond the cap, once the sweep has released its empty pages and set
    /// the next budget. (A cap checked as each page went would shrink with the pages still to
    /// go, and keep about half of a large heap that emptied at once.)
    fn trim_spare_pages(&mut self) {
        let cap = self.spare_page_cap();
        while self.spare_pages.len() > cap {
            let base = self
                .spare_pages
                .pop()
                .expect("more spare pages than the cap");
            // SAFETY: a spare page is a `page_layout()` block from new_page; no cell remains.
            unsafe { dealloc(base as *mut u8, page_layout()) };
        }
    }

    #[cfg(test)]
    pub(crate) fn is_allocated_word(&self, word: u64) -> bool {
        self.locate(word).is_some_and(|cell| match cell.loc {
            Loc::Small { page, index } => self.pages[page as usize]
                .as_ref()
                .is_some_and(|p| bit(&p.allocated, index as usize)),
            Loc::Large => true,
        })
    }
}

impl Heap {
    /// I5 teardown: finalize every remaining cell and free all heap memory,
    /// leaving an empty heap. Finalizers run in no particular order (see
    /// FinalizeFn). Every handle into this heap dangles afterwards.
    pub(crate) fn release_all(&mut self) {
        for page in std::mem::take(&mut self.pages).into_iter().flatten() {
            if let Some(finalize) = self.descs[page.desc.0 as usize].finalize {
                for i in 0..page.cells {
                    if bit(&page.allocated, i) && !bit(&page.zombies, i) {
                        finalize(page.cell_ptr(i) as *mut u8);
                    }
                }
            }
            // SAFETY: allocated in new_page with this layout.
            unsafe { dealloc(page.base as *mut u8, page_layout()) };
        }
        for (start, large) in std::mem::take(&mut self.larges) {
            if let Some(finalize) = self.descs[large.desc.0 as usize].finalize {
                if !large.zombie {
                    finalize(start as *mut u8);
                }
            }
            // SAFETY: allocated in alloc_large with this layout.
            unsafe { dealloc(start as *mut u8, large_layout(large.size)) };
        }
        for base in std::mem::take(&mut self.spare_pages) {
            // SAFETY: a spare page is a `page_layout()` block from new_page.
            unsafe { dealloc(base as *mut u8, page_layout()) };
        }
        self.free_page_slots.clear();
        self.page_map.clear();
        for class in self.classes.iter_mut().flatten() {
            *class = ClassPages::default();
        }
        self.large_span = (usize::MAX, 0);
        self.allocated_bytes = 0;
        self.refresh_due();
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.debug_assert_quiescent();
        }
        // The due word belongs to the heap's owner, which may be dropping it too.
        self.due_word = None;
        self.release_all();
    }
}

#[derive(Default)]
pub(crate) struct Control {
    nogc_depth: Cell<u32>,
    pending_collect: Cell<bool>,
    borrows: Cell<u32>,
    collecting: Cell<bool>,
}

/// RAII no-GC region (I9). Collection requests inside are deferred.
/// Holds only the shared control block, so `alloc(&mut Heap)` remains
/// callable while the guard is alive.
pub(crate) struct NoGcScope {
    control: Rc<Control>,
}

impl Drop for NoGcScope {
    fn drop(&mut self) {
        self.control
            .nogc_depth
            .set(self.control.nogc_depth.get() - 1);
    }
}

/// An outstanding GcCell borrow (I9); see [`Heap::borrow_scope`].
pub(crate) struct BorrowScope {
    control: Rc<Control>,
}

impl Drop for BorrowScope {
    fn drop(&mut self) {
        self.control.borrows.set(self.control.borrows.get() - 1);
    }
}

/// Declares this thread's native stack, from the current frame up to a
/// top, as holding heap references (Decision 0). Conservative collections
/// on this thread scan it, plus segments entered through [`grow`]; without
/// a declaration they defer. Nested declarations are no-ops.
pub(crate) struct NativeStackScope {
    registered: bool,
    _thread: PhantomData<*const ()>,
}

/// # Safety
/// `top` must lie in the current thread's stack mapping, at or above every
/// frame that holds heap references across collections while the scope is
/// alive, and the scope must be dropped on this thread outside any segment
/// entered through [`grow`].
pub(crate) unsafe fn enter_native_stack(top: *const u8) -> NativeStackScope {
    NativeStackScope {
        registered: native::enter(top as usize),
        _thread: PhantomData,
    }
}

impl Drop for NativeStackScope {
    fn drop(&mut self) {
        if self.registered {
            native::exit();
        }
    }
}

/// Register this thread's whole native stack, from the current frame up to the top of its
/// mapping, as a permanent entry region (idempotent). Interpreters call this at creation so
/// every engine frame on the thread lies below the scanned top. Returns false where the
/// stack bounds cannot be read; conservative collection then defers on this thread.
pub(crate) fn register_thread_native_stack() -> bool {
    native::enter_thread()
}

/// Native stack an embedder entry (a script evaluation, a microtask checkpoint) zeroes before
/// running JavaScript, so the frames it builds hold no stale cell addresses from earlier jobs.
/// Debug frames are several times larger than optimized ones.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const ENTRY_SCRUB_BYTES: usize = if cfg!(debug_assertions) {
    1 << 20
} else {
    128 << 10
};

/// Zero up to `bytes` of native stack below the caller's stack pointer, within the current stack
/// or segment. Frames built there afterwards (a collection's own, or the next job's) then start
/// without stale cell addresses, which the conservative scan would otherwise take as roots:
/// false retention, never unsoundness. Inlined, so the zeroing leaf runs straight from the
/// caller's frame: a scrubbing frame of its own would leave its unwritten slots stale for the
/// caller's next callee.
#[cfg(not(target_arch = "wasm32"))]
#[inline(always)]
pub(crate) fn scrub_stack(bytes: usize) {
    let available = stacker::remaining_stack()
        .unwrap_or(0)
        .saturating_sub(128 * 1024);
    let bytes = bytes.min(available) & !15;
    if bytes > 0 {
        // SAFETY: the zeroed range lies below the stack pointer, above this stack's limit
        // with 128 KiB to spare, and holds no live data.
        unsafe { zero_below_sp(bytes) };
    }
}

/// Zero `bytes` (a multiple of 16) below the caller's stack pointer. A leaf with no frame, so
/// nothing of its own is left in the zeroed range.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn zero_below_sp(bytes: usize) {
    std::arch::naked_asm!(
        "mov x9, sp",
        "sub x10, x9, x0",
        "2:",
        "cmp x10, x9",
        "b.hs 3f",
        "stp xzr, xzr, [x10], #16",
        "b 2b",
        "3:",
        "ret",
    )
}

/// Zero `bytes` (a multiple of 16) below the caller's stack pointer, sparing the return address
/// the call pushed. A leaf with no frame, so nothing of its own is left in the zeroed range.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "C" fn zero_below_sp(bytes: usize) {
    std::arch::naked_asm!(
        "lea rcx, [rsp + 8]",
        "sub rcx, rdi",
        "2:",
        "cmp rcx, rsp",
        "jae 3f",
        "mov qword ptr [rcx], 0",
        "add rcx, 8",
        "jmp 2b",
        "3:",
        "ret",
    )
}

/// Portable fallback: zero from frames of its own (whose own slots may stay stale).
#[cfg(not(any(
    target_arch = "aarch64",
    target_arch = "x86_64",
    target_arch = "wasm32"
)))]
unsafe fn zero_below_sp(bytes: usize) {
    const CHUNK: usize = 4096;
    #[inline(never)]
    fn zero(chunks: usize) {
        let mut chunk = [0u8; CHUNK];
        std::hint::black_box(&mut chunk);
        if chunks > 1 {
            zero(chunks - 1);
        }
        // Used after the call, so the frame cannot be reused for it.
        std::hint::black_box(&chunk);
    }
    zero(bytes / CHUNK);
}

/// `stacker::grow` whose new segment is registered, with exact bounds, for
/// conservative scanning on this thread (contract option (a)). From S2,
/// every grow path that can allocate on this heap must use it; collections
/// requested on an unregistered segment defer.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn grow<R, F: FnOnce() -> R>(stack_size: usize, f: F) -> R {
    native::grow(stack_size, f)
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod native {
    //! Conservative native-stack scanning: the Decision 0 contract, encoded.
    //! - Only this thread's registered regions are scanned: the entry stack
    //!   up to its declared top, plus stacker segments entered through
    //!   `grow`. With SP outside the innermost region (an unregistered
    //!   `stacker::grow` segment, a foreign stack) the heap defers instead
    //!   of scanning blindly.
    //! - A region below the innermost is scanned from the SP recorded when
    //!   its successor was entered, after the same register spill as a
    //!   collection: stacker's own frames below that SP are not scanned, and
    //!   they may save and reuse callee-saved registers.
    use std::cell::RefCell;

    pub(super) const SPILL_WORDS: usize = 24;

    #[derive(Clone, Copy)]
    struct Region {
        /// Lowest address SP can take in the region (membership test only).
        lo: usize,
        /// Exclusive scan bound: the region's top.
        hi: usize,
        /// SP in the previous region when this one was entered (0 for the
        /// entry stack); the previous region is scanned from there.
        resume_sp: usize,
    }

    thread_local! {
        static REGIONS: RefCell<Vec<Region>> = const { RefCell::new(Vec::new()) };
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub(super) fn spill_registers(buf: &mut [u64; SPILL_WORDS]) {
        // SAFETY: stores x19–x30 and d8–d15 into the first 20 words of `buf`.
        unsafe {
            std::arch::asm!(
                "stp x19, x20, [{buf}]",
                "stp x21, x22, [{buf}, #16]",
                "stp x23, x24, [{buf}, #32]",
                "stp x25, x26, [{buf}, #48]",
                "stp x27, x28, [{buf}, #64]",
                "stp x29, x30, [{buf}, #80]",
                "stp d8, d9, [{buf}, #96]",
                "stp d10, d11, [{buf}, #112]",
                "stp d12, d13, [{buf}, #128]",
                "stp d14, d15, [{buf}, #144]",
                buf = in(reg) buf.as_mut_ptr(),
                options(nostack, preserves_flags),
            );
        }
    }

    /// System V callee-saved registers. Not compiled or run on the aarch64
    /// development host (its LLVM has no x86 backend): validate on x64
    /// before this heap is authoritative there (S6).
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub(super) fn spill_registers(buf: &mut [u64; SPILL_WORDS]) {
        // SAFETY: stores rbx, rbp, r12–r15 into the first 6 words of `buf`.
        unsafe {
            std::arch::asm!(
                "mov [{buf}], rbx",
                "mov [{buf} + 8], rbp",
                "mov [{buf} + 16], r12",
                "mov [{buf} + 24], r13",
                "mov [{buf} + 32], r14",
                "mov [{buf} + 40], r15",
                buf = in(reg) buf.as_mut_ptr(),
                options(nostack, preserves_flags),
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    pub(super) fn current_sp() -> usize {
        let sp: usize;
        // SAFETY: reads SP only.
        unsafe {
            std::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags))
        };
        sp
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub(super) fn current_sp() -> usize {
        let sp: usize;
        // SAFETY: reads RSP only.
        unsafe {
            std::arch::asm!("mov {}, rsp", out(reg) sp, options(nomem, nostack, preserves_flags))
        };
        sp
    }

    fn page_size() -> usize {
        static PAGE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *PAGE.get_or_init(|| {
            unsafe extern "C" {
                fn sysconf(name: std::ffi::c_int) -> std::ffi::c_long;
            }
            #[cfg(target_os = "linux")]
            const SC_PAGESIZE: std::ffi::c_int = 30;
            #[cfg(target_os = "macos")]
            const SC_PAGESIZE: std::ffi::c_int = 29;
            // SAFETY: sysconf has no preconditions.
            let size = unsafe { sysconf(SC_PAGESIZE) };
            usize::try_from(size)
                .ok()
                .filter(|size| size.is_power_of_two())
                .expect("sysconf(_SC_PAGESIZE)")
        })
    }

    pub(super) fn scannable(sp: usize) -> bool {
        REGIONS.with(|regions| {
            regions
                .borrow()
                .last()
                .is_some_and(|region| (region.lo..region.hi).contains(&sp))
        })
    }

    /// Ranges to scan for a collection whose deepest frame has stack
    /// pointer `sp`, innermost first; `None` when `sp` is outside the
    /// innermost registered region.
    pub(super) fn scan_ranges(sp: usize) -> Option<Vec<(usize, usize)>> {
        REGIONS.with(|regions| {
            let regions = regions.borrow();
            let current = regions
                .last()
                .filter(|region| (region.lo..region.hi).contains(&sp))?;
            let mut ranges = vec![(sp, current.hi)];
            for pair in regions.windows(2).rev() {
                ranges.push((pair[1].resume_sp, pair[0].hi));
            }
            Some(ranges)
        })
    }

    /// Register the entry stack; `false` (a no-op) when already registered.
    pub(super) fn enter(top: usize) -> bool {
        REGIONS.with(|regions| {
            let mut regions = regions.borrow_mut();
            if !regions.is_empty() {
                return false;
            }
            let sp = current_sp();
            debug_assert!(sp < top, "native stack top below the current frame");
            // stacker knows this thread's stack limit. Without it no SP can be
            // shown to lie on this stack, so conservative collection defers.
            let lo = stacker::remaining_stack()
                .map_or(usize::MAX, |remaining| sp.saturating_sub(remaining));
            regions.push(Region {
                lo,
                hi: top,
                resume_sp: 0,
            });
            true
        })
    }

    pub(super) fn exit() {
        REGIONS.with(|regions| {
            let mut regions = regions.borrow_mut();
            debug_assert_eq!(
                regions.len(),
                1,
                "native stack scope dropped inside a grown segment"
            );
            regions.pop();
        });
    }

    /// Permanently register the current thread's stack up to its mapping's top.
    pub(super) fn enter_thread() -> bool {
        if REGIONS.with(|regions| !regions.borrow().is_empty()) {
            return true;
        }
        match thread_stack_top() {
            // Never popped: the region lives as long as the thread.
            Some(top) => enter(top) || REGIONS.with(|regions| !regions.borrow().is_empty()),
            None => false,
        }
    }

    #[cfg(target_os = "linux")]
    fn thread_stack_top() -> Option<usize> {
        // Opaque pthread_attr_t: 56 bytes (glibc x86_64) / 64 bytes (aarch64); 128 covers both.
        #[repr(C, align(16))]
        struct Attr([u8; 128]);
        unsafe extern "C" {
            fn pthread_self() -> usize;
            fn pthread_getattr_np(thread: usize, attr: *mut Attr) -> std::ffi::c_int;
            fn pthread_attr_getstack(
                attr: *const Attr,
                addr: *mut *mut std::ffi::c_void,
                size: *mut usize,
            ) -> std::ffi::c_int;
            fn pthread_attr_destroy(attr: *mut Attr) -> std::ffi::c_int;
        }
        let mut attr = Attr([0; 128]);
        let mut addr = std::ptr::null_mut();
        let mut size = 0usize;
        // SAFETY: `attr` outlives every call; getattr initializes it and destroy releases it.
        unsafe {
            if pthread_getattr_np(pthread_self(), &mut attr) != 0 {
                return None;
            }
            let status = pthread_attr_getstack(&attr, &mut addr, &mut size);
            pthread_attr_destroy(&mut attr);
            (status == 0 && size != 0).then(|| addr as usize + size)
        }
    }

    #[cfg(target_os = "macos")]
    fn thread_stack_top() -> Option<usize> {
        unsafe extern "C" {
            fn pthread_self() -> usize;
            fn pthread_get_stackaddr_np(thread: usize) -> *mut std::ffi::c_void;
        }
        // SAFETY: no preconditions; the result is the top (highest address) of the stack.
        let top = unsafe { pthread_get_stackaddr_np(pthread_self()) } as usize;
        (top != 0).then_some(top)
    }

    struct SegmentScope;

    impl Drop for SegmentScope {
        fn drop(&mut self) {
            REGIONS.with(|regions| regions.borrow_mut().pop());
        }
    }

    #[inline(never)]
    pub(super) fn grow<R, F: FnOnce() -> R>(stack_size: usize, f: F) -> R {
        if REGIONS.with(|regions| regions.borrow().is_empty()) {
            // Nothing on this thread is scanned: a plain segment is equivalent.
            return stacker::grow(stack_size, f);
        }
        let mut regs = [0u64; SPILL_WORDS];
        spill_registers(&mut regs);
        let resume_sp = current_sp();
        let out = stacker::grow(stack_size, move || {
            let region = segment_region(stack_size, resume_sp);
            REGIONS.with(|regions| regions.borrow_mut().push(region));
            let _scope = SegmentScope;
            f()
        });
        std::hint::black_box(&regs);
        out
    }

    /// Bounds of the stacker segment we are running on. stacker sets its
    /// limit to the page-aligned segment base before switching, and the
    /// usable area is max(1, ceil(size / page)) pages (stacker 0.1.25; the
    /// bounds-precision test checks this against the mapping). A mismatch
    /// yields an empty region: collections inside then defer.
    fn segment_region(stack_size: usize, resume_sp: usize) -> Region {
        let sp = current_sp();
        let page = page_size();
        let region = stacker::remaining_stack().map(|remaining| {
            // remaining_stack measured SP a few frames below ours.
            let lo = (sp.wrapping_sub(remaining) + page / 2) & !(page - 1);
            Region {
                lo,
                hi: lo + stack_size.div_ceil(page).max(1) * page,
                resume_sp,
            }
        });
        match region {
            Some(region) if (region.lo..region.hi).contains(&sp) => region,
            _ => {
                if cfg!(debug_assertions) {
                    panic!("could not derive stacker segment bounds");
                }
                Region {
                    lo: usize::MAX,
                    hi: 0,
                    resume_sp,
                }
            }
        }
    }

    #[cfg(test)]
    pub(super) fn innermost_region() -> Option<(usize, usize)> {
        REGIONS.with(|regions| regions.borrow().last().map(|r| (r.lo, r.hi)))
    }
}

#[cfg(not(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
mod native {
    //! No conservative scanner for this target (wasm32 cannot scan its
    //! stack): `Heap::collect` always defers, and collections run only at
    //! precise safepoints (`Heap::collect_at_safepoint`), per the S0 wasm32
    //! decision.
    pub(super) const SPILL_WORDS: usize = 1;

    pub(super) fn spill_registers(_: &mut [u64; SPILL_WORDS]) {}

    pub(super) fn current_sp() -> usize {
        0
    }

    pub(super) fn scannable(_: usize) -> bool {
        false
    }

    pub(super) fn scan_ranges(_: usize) -> Option<Vec<(usize, usize)>> {
        None
    }

    pub(super) fn enter(_: usize) -> bool {
        false
    }

    pub(super) fn exit() {}

    pub(super) fn enter_thread() -> bool {
        false
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn grow<R, F: FnOnce() -> R>(stack_size: usize, f: F) -> R {
        stacker::grow(stack_size, f)
    }

    #[cfg(test)]
    pub(super) fn innermost_region() -> Option<(usize, usize)> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::Instant;

    const TAG: u64 = 0xfff9_0000_0000_0000;

    fn tagged(p: *const u8) -> u64 {
        TAG | (p as u64 & PAYLOAD_MASK)
    }

    fn trace_node(cell: *const u8, visit: &mut dyn FnMut(*const u8)) {
        let next = unsafe { std::ptr::read(cell as *const u64) };
        if next != 0 {
            visit((next & PAYLOAD_MASK) as *const u8);
        }
    }

    fn trace_none(_: *const u8, _: &mut dyn FnMut(*const u8)) {}

    /// WeakRef-like cell: word 0 is a weak target slot.
    fn weak_target(cell: *mut u8, sink: &mut WeakSink<'_>) {
        sink.weak_slot(cell as *mut u64);
    }

    /// WeakMap-like cell: four (key, value) ephemeron entries.
    fn weak_entries(cell: *mut u8, sink: &mut WeakSink<'_>) {
        let words = cell as *mut u64;
        for entry in 0..4 {
            unsafe { sink.ephemeron(words.add(2 * entry), words.add(2 * entry + 1)) };
        }
    }

    thread_local! {
        static FIN_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    fn fin_count(_: *mut u8) {
        FIN_COUNT.with(|c| c.set(c.get() + 1));
    }

    struct Fixture {
        heap: Heap,
        node: DescId,
        leaf: DescId,
        weakref: DescId,
        table: DescId,
        _roots: RootRange,
        roots: Box<[u64; 64]>,
    }

    impl Fixture {
        fn new() -> Self {
            let mut heap = Heap::new();
            let node = heap.register_desc(LayoutDesc {
                name: "node",
                trace: trace_node,
                weak: None,
                finalize: None,
            });
            let leaf = heap.register_desc(LayoutDesc {
                name: "leaf",
                trace: trace_none,
                weak: None,
                finalize: Some(fin_count),
            });
            let weakref = heap.register_desc(LayoutDesc {
                name: "weakref",
                trace: trace_none,
                weak: Some(weak_target),
                finalize: None,
            });
            let table = heap.register_desc(LayoutDesc {
                name: "table",
                trace: trace_none,
                weak: Some(weak_entries),
                finalize: None,
            });
            let roots = Box::new([0u64; 64]);
            // SAFETY: `roots` is declared after `_roots`, so it outlives it.
            let _roots = unsafe {
                heap.add_root_range("test-roots", roots.as_ptr(), roots.as_ptr().add(64))
            };
            Self {
                heap,
                node,
                leaf,
                weakref,
                table,
                _roots,
                roots,
            }
        }

        fn alloc_node(&mut self, next: u64, val: u64) -> *mut u8 {
            let p = self.heap.alloc(self.node, 16);
            unsafe {
                std::ptr::write(p as *mut u64, next);
                std::ptr::write((p as *mut u64).add(1), val);
            }
            p
        }

        /// Chain of `n` nodes, head rooted in `roots[0]`.
        fn build_chain(&mut self, n: u64) -> *mut u8 {
            let mut head: *mut u8 = std::ptr::null_mut();
            for i in 0..n {
                let next = if head.is_null() { 0 } else { tagged(head) };
                head = self.alloc_node(next, i);
                self.roots[0] = tagged(head);
            }
            head
        }

        fn val(&self, p: *const u8) -> u64 {
            word(p, 1)
        }

        /// Precise collection: native frames are not roots, so death
        /// assertions do not depend on what registers happen to retain.
        fn collect(&mut self) {
            assert!(self.heap.collect_at_safepoint());
        }

        fn live(&self) -> usize {
            self.heap.stats().live_cells
        }
    }

    fn word(p: *const u8, index: usize) -> u64 {
        unsafe { std::ptr::read((p as *const u64).add(index)) }
    }

    fn set_word(p: *mut u8, index: usize, value: u64) {
        unsafe { std::ptr::write((p as *mut u64).add(index), value) };
    }

    /// A stack top above the calling test's frame.
    fn stack_top(marker: &u64) -> *const u8 {
        (marker as *const u64).wrapping_add(256) as *const u8
    }

    /// Allocate where the caller never holds the plain pointer.
    #[inline(never)]
    fn alloc_hidden(heap: &mut Heap, desc: DescId, size: usize) -> u64 {
        !(heap.alloc(desc, size) as u64)
    }

    /// Store a hidden pointer, unhidden, into a stack slot of the caller.
    #[inline(never)]
    fn stash(slot: &mut [u64; 2], hidden: u64) {
        unsafe { std::ptr::write_volatile(&mut slot[0], !hidden) };
    }

    #[test]
    fn unreachable_cells_are_swept_and_chains_survive() {
        let mut fx = Fixture::new();
        let c = fx.alloc_node(0, 3);
        let b = fx.alloc_node(tagged(c), 2);
        let a = fx.alloc_node(tagged(b), 1);
        let _garbage = fx.alloc_node(0, 99);
        fx.roots[0] = tagged(a);
        fx.collect();
        assert_eq!(fx.live(), 3);
        assert_eq!((fx.val(a), fx.val(b), fx.val(c)), (1, 2, 3));
        assert_eq!(fx.heap.stats().cells_swept, 1);
    }

    /// P1 S4: a collection's emptied pages are kept and reused before any new page is allocated.
    #[test]
    fn emptied_pages_are_reused() {
        let mut fx = Fixture::new();
        let per_page = PAGE_SIZE / 16;
        let first: std::collections::HashSet<usize> = (0..per_page * 4)
            .map(|i| fx.alloc_node(0, i as u64) as usize >> PAGE_SHIFT)
            .collect();
        assert_eq!(first.len(), 4);
        fx.collect();
        assert_eq!((fx.live(), fx.heap.stats().pages_live), (0, 0));
        assert_eq!(fx.heap.spare_pages.len(), 4);
        let second: std::collections::HashSet<usize> = (0..per_page * 4)
            .map(|i| fx.alloc_node(0, i as u64) as usize >> PAGE_SHIFT)
            .collect();
        assert_eq!(second, first, "the spare pages are reused");
        assert!(fx.heap.spare_pages.is_empty());
    }

    /// P1 S4 review: the pool keeps at most the pages the next cycle's allocation budget can
    /// fill, however many pages one sweep released, and shrinks with the budget when the heap
    /// does.
    #[test]
    fn spare_pages_follow_the_next_cycles_budget() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        // The engine's 8 MiB floor: 128 pages until the live heap grows past it.
        fx.heap.set_heap_growth(8 << 20, 100);
        let per_page = PAGE_SIZE / 16;
        let pages = |fx: &Fixture| {
            let stats = fx.heap.stats();
            (stats.pages_live, stats.pages_spare)
        };
        // A burst of 300 pages, all garbage.
        for i in 0..per_page * 300 {
            fx.alloc_node(0, i as u64);
        }
        fx.collect();
        assert_eq!(pages(&fx), (0, 128));
        fx.collect();
        assert_eq!(pages(&fx), (0, 128), "a later collection keeps the cap");
        // 300 full live pages raise the budget to 300 pages, so a 200-page burst of garbage on
        // top of them stays pooled.
        fx.build_chain((per_page * 300) as u64);
        for i in 0..per_page * 200 {
            fx.alloc_node(0, i as u64);
        }
        fx.collect();
        assert_eq!(pages(&fx), (300, 200));
        // When the live heap dies, the budget and the pool return to the floor's 128 pages.
        fx.roots[0] = 0;
        fx.collect();
        assert_eq!(pages(&fx), (0, 128));
    }

    #[test]
    fn interior_and_tagged_roots_resolve() {
        let mut fx = Fixture::new();
        let a = fx.alloc_node(0, 7);
        // Root holds a tagged interior pointer (second word of the cell).
        fx.roots[0] = TAG | ((a as u64 + 8) & PAYLOAD_MASK);
        fx.collect();
        assert_eq!(fx.live(), 1);
        assert_eq!(fx.val(a), 7);
    }

    #[test]
    fn in_cell_ephemerons_reach_a_fixed_point_and_clear_dead_entries() {
        let mut fx = Fixture::new();
        let table = fx.heap.alloc(fx.table, 64);
        let [k1, v1, v2, k3, v3] = [0; 5].map(|_| fx.heap.alloc(fx.leaf, 16));
        // (k1 -> v1), (v1 -> v2): v1 is itself a key, so marking v2 needs
        // the fixed point. (k3 -> v3) dies with k3.
        for (entry, (key, value)) in [(k1, v1), (v1, v2), (k3, v3)].into_iter().enumerate() {
            set_word(table, 2 * entry, tagged(key));
            set_word(table, 2 * entry + 1, tagged(value));
        }
        fx.roots[0] = tagged(table);
        fx.roots[1] = tagged(k1);
        fx.collect();
        assert_eq!(fx.live(), 4, "table, k1, v1, v2");
        assert_eq!(word(table, 3), tagged(v2));
        assert_eq!(
            (word(table, 4), word(table, 5)),
            (0, 0),
            "dead-key entry cleared"
        );
    }

    #[test]
    fn unreachable_table_keeps_no_values_alive() {
        let mut fx = Fixture::new();
        let table = fx.heap.alloc(fx.table, 64);
        let key = fx.heap.alloc(fx.leaf, 16);
        let value = fx.heap.alloc(fx.leaf, 16);
        set_word(table, 0, tagged(key));
        set_word(table, 1, tagged(value));
        fx.roots[0] = tagged(key);
        fx.collect();
        assert_eq!(fx.live(), 1, "only the rooted key survives a dead table");
    }

    #[test]
    fn weak_slots_clear_dead_targets_and_keep_the_rest() {
        let mut fx = Fixture::new();
        let [dead_ref, live_ref, number_ref] = [0; 3].map(|_| fx.heap.alloc(fx.weakref, 16));
        let dead = fx.heap.alloc(fx.leaf, 16);
        let live = fx.heap.alloc(fx.leaf, 16);
        let number = std::f64::consts::PI.to_bits();
        set_word(dead_ref, 0, tagged(dead));
        set_word(live_ref, 0, tagged(live));
        set_word(number_ref, 0, number);
        for (slot, cell) in [dead_ref, live_ref, number_ref, live]
            .into_iter()
            .enumerate()
        {
            fx.roots[slot] = tagged(cell);
        }
        fx.collect();
        assert_eq!(word(dead_ref, 0), 0, "dead target cleared");
        assert_eq!(word(live_ref, 0), tagged(live), "live target kept");
        assert_eq!(word(number_ref, 0), number, "non-reference word untouched");
    }

    /// Fill the 4 KiB class (16 cells per page) until `target` is handed
    /// out again; returns it rooted in `roots[slot]`.
    fn reuse_cell(fx: &mut Fixture, desc: DescId, target: u64, slot: usize) -> *mut u8 {
        for _ in 0..64 {
            let p = fx.heap.alloc(desc, 4096);
            if p as u64 == target {
                fx.roots[slot] = tagged(p);
                return p;
            }
        }
        panic!("cell was not reused");
    }

    #[test]
    fn reused_cells_are_unaffected_by_dead_weak_holders() {
        // Formerly weak slots and ephemerons were registered globally and
        // outlived their holders: a reused weak-holder cell had an ordinary
        // word zeroed, and a reused key cell next to a free value cell made
        // the ephemeron loop spin forever.
        let mut fx = Fixture::new();
        // Anchors keep both pages alive, so freed cells are reused in place.
        let anchor = fx.heap.alloc(fx.leaf, 4096);
        let anchor_ref = fx.heap.alloc(fx.weakref, 4096);
        fx.roots[0] = tagged(anchor);
        fx.roots[3] = tagged(anchor_ref);
        let table = fx.heap.alloc(fx.table, 4096);
        let value = fx.heap.alloc(fx.leaf, 4096);
        let key = fx.heap.alloc(fx.leaf, 4096);
        set_word(table, 0, tagged(key));
        set_word(table, 1, tagged(value));
        let holder = fx.heap.alloc(fx.weakref, 4096);
        set_word(holder, 0, tagged(key));
        let (key, holder) = (key as u64, holder as u64);
        fx.collect();
        assert_eq!(fx.live(), 2, "only the anchors survive");
        let (leaf, weakref) = (fx.leaf, fx.weakref);
        let new_key = reuse_cell(&mut fx, leaf, key, 1);
        let new_holder = reuse_cell(&mut fx, weakref, holder, 2);
        let number = std::f64::consts::PI.to_bits();
        set_word(new_holder, 0, number);
        fx.collect();
        assert!(fx.heap.is_allocated_word(new_key as u64));
        assert_eq!(word(new_holder, 0), number);
    }

    #[test]
    fn edge_sources_trace_side_tables_through_their_owners() {
        let mut fx = Fixture::new();
        // Map-like side table: (owner, value), pruned by its sweep hook.
        let table: Rc<RefCell<Vec<(usize, u64)>>> = Rc::default();
        let entries = Rc::clone(&table);
        let _source = fx.heap.add_edge_source("map-data", move |sink| {
            for &(owner, value) in entries.borrow().iter() {
                sink.edge(owner as *const u8, value);
            }
        });
        let entries = Rc::clone(&table);
        let _hook = fx.heap.add_sweep_hook("map-data", move |is_live| {
            entries
                .borrow_mut()
                .retain(|&(owner, _)| is_live(owner as *const u8));
        });
        let owner = fx.alloc_node(0, 1);
        // The value points back at its owner: a cycle through the table.
        let value = fx.alloc_node(tagged(owner), 2);
        table.borrow_mut().push((owner as usize, tagged(value)));
        fx.roots[0] = tagged(owner);
        fx.collect();
        assert_eq!(fx.live(), 2);
        fx.roots[0] = 0;
        fx.collect();
        assert_eq!(fx.live(), 0, "a cycle through a side table is collected");
        assert!(table.borrow().is_empty());
    }

    #[test]
    fn source_ephemerons_need_both_owner_and_key() {
        let mut fx = Fixture::new();
        let table: Rc<RefCell<Vec<(usize, u64, u64)>>> = Rc::default();
        let entries = Rc::clone(&table);
        let _source = fx.heap.add_edge_source("weak-collection", move |sink| {
            for &(owner, key, value) in entries.borrow().iter() {
                sink.ephemeron(owner as *const u8, key, value);
            }
        });
        let [map, key, value, dead_map, key2, value2] = [0; 6].map(|_| fx.heap.alloc(fx.leaf, 16));
        table.borrow_mut().extend([
            (map as usize, tagged(key), tagged(value)),
            (dead_map as usize, tagged(key2), tagged(value2)),
        ]);
        for (slot, cell) in [map, key, key2].into_iter().enumerate() {
            fx.roots[slot] = tagged(cell);
        }
        fx.collect();
        assert!(fx.heap.is_allocated_word(value as u64), "live map and key");
        assert!(!fx.heap.is_allocated_word(value2 as u64), "dead map");
        fx.roots[1] = 0;
        table.borrow_mut().truncate(1);
        fx.collect();
        assert!(!fx.heap.is_allocated_word(value as u64), "dead key");
    }

    /// A side table still holding a freed cell (a source bug) must not hang
    /// marking, as the old ephemeron loop did; debug verification reports it.
    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "conditional edge references unmarked")
    )]
    fn source_value_in_a_free_cell_terminates() {
        let mut fx = Fixture::new();
        let anchor = fx.heap.alloc(fx.leaf, 16);
        let freed = fx.heap.alloc(fx.leaf, 16) as u64;
        fx.roots[0] = tagged(anchor);
        fx.collect();
        let _source = fx.heap.add_edge_source("stale", move |sink| {
            sink.ephemeron(std::ptr::null(), tagged(anchor), freed);
        });
        fx.collect();
        assert_eq!(fx.live(), 1);
    }

    /// An embedder (S2: the interpreter) supplying roots and owner-keyed side tables.
    struct TableClient {
        roots: Vec<u64>,
        side_slots: Vec<(usize, u64)>,
    }

    impl MarkClient for TableClient {
        fn roots(&self, sink: &mut dyn FnMut(u64)) {
            for &root in &self.roots {
                sink(root);
            }
        }

        fn side_slot_owners(&self, owners: &mut dyn FnMut(usize)) {
            for &(owner, _) in &self.side_slots {
                owners(owner);
            }
        }

        fn trace_side_slots(&self, owner: usize, sink: &mut dyn FnMut(u64)) {
            for &(entry_owner, value) in &self.side_slots {
                if entry_owner == owner {
                    sink(value);
                }
            }
        }

        fn ephemerons(&self, _: &mut EdgeSink<'_>) {}
    }

    #[test]
    fn embedder_marking_traces_side_slots_of_marked_owners_only() {
        let mut fx = Fixture::new();
        let [owner, value, dead_owner, dead_value] =
            [0; 4].map(|_| fx.heap.alloc(fx.leaf, 16) as usize);
        let client = TableClient {
            roots: vec![tagged(owner as *const u8)],
            // The dead owner's table entry points back at it: a cycle through the table.
            side_slots: vec![
                (owner, value as u64),
                (dead_owner, dead_value as u64),
                (dead_value, dead_owner as u64),
            ],
        };
        assert!(fx.heap.start_marking(false, &client));
        let dead = fx.heap.dead_cells(fx.leaf);
        assert_eq!(dead.len(), 2);
        assert!(dead.contains(&dead_owner) && dead.contains(&dead_value));
        assert!(
            fx.heap.is_dead_target(dead_value + 8),
            "interior addresses resolve"
        );
        assert!(!fx.heap.is_dead_target(value));
        fx.heap.finish_collection(&|_| false);
        assert_eq!(fx.live(), 2);
    }

    #[test]
    fn cells_allocated_between_the_phases_survive_that_sweep() {
        // The embedder sweeps its own tables between the phases and may allocate there.
        let mut fx = Fixture::new();
        let dead = fx.heap.alloc(fx.leaf, 16) as usize;
        let client = TableClient {
            roots: Vec::new(),
            side_slots: Vec::new(),
        };
        assert!(fx.heap.start_marking(false, &client));
        let small = fx.heap.alloc(fx.leaf, 16) as usize;
        let large = fx.heap.alloc(fx.leaf, 8192) as usize;
        assert_eq!(fx.heap.dead_cells(fx.leaf), vec![dead]);
        fx.heap.finish_collection(&|_| false);
        assert!(!fx.heap.is_allocated_word(dead as u64));
        assert!(fx.heap.is_allocated_word(small as u64), "allocated black");
        assert!(fx.heap.is_allocated_word(large as u64), "allocated black");
        assert_eq!(fx.live(), 2);
    }

    #[inline(never)]
    fn leave_word_below(word: u64) -> usize {
        let mut slot = [0u64; 64];
        slot[17] = word;
        std::hint::black_box(&mut slot);
        std::ptr::from_ref(&slot[17]) as usize
    }

    #[test]
    fn scrub_stack_clears_stale_words_below_the_caller() {
        let marker = 0x5afe_c0de_dead_beef;
        let addr = leave_word_below(marker);
        let here = 0u64;
        assert!(
            addr < std::ptr::from_ref(&here) as usize,
            "the stale word is below"
        );
        // SAFETY: a dead frame's slot in this thread's mapped stack, read before any call
        // reuses it (and again after the scrub).
        let before = unsafe { std::ptr::read_volatile(addr as *const u64) };
        assert_eq!(before, marker, "the dead frame still holds the word");
        super::scrub_stack(256 * 1024);
        // SAFETY: as above.
        let after = unsafe { std::ptr::read_volatile(addr as *const u64) };
        assert_ne!(after, marker);
    }

    /// Fill `bytes` (a multiple of 16) below the caller's stack pointer with `marker`.
    #[cfg(target_arch = "aarch64")]
    #[unsafe(naked)]
    unsafe extern "C" fn fill_below_sp(bytes: usize, marker: u64) {
        std::arch::naked_asm!(
            "mov x9, sp",
            "sub x10, x9, x0",
            "2:",
            "cmp x10, x9",
            "b.hs 3f",
            "stp x1, x1, [x10], #16",
            "b 2b",
            "3:",
            "ret",
        )
    }

    /// The nonzero words among the `bytes` below the caller's stack pointer.
    #[cfg(target_arch = "aarch64")]
    #[unsafe(naked)]
    unsafe extern "C" fn nonzero_below_sp(bytes: usize) -> usize {
        std::arch::naked_asm!(
            "mov x9, sp",
            "sub x10, x9, x0",
            "mov x0, xzr",
            "2:",
            "cmp x10, x9",
            "b.hs 3f",
            "ldr x11, [x10], #8",
            "cmp x11, #0",
            "cinc x0, x0, ne",
            "b 2b",
            "3:",
            "ret",
        )
    }

    /// Fill `bytes` (a multiple of 16) below the caller's stack pointer with `marker`, sparing
    /// the return address.
    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "C" fn fill_below_sp(bytes: usize, marker: u64) {
        std::arch::naked_asm!(
            "lea rcx, [rsp + 8]",
            "sub rcx, rdi",
            "2:",
            "cmp rcx, rsp",
            "jae 3f",
            "mov qword ptr [rcx], rsi",
            "add rcx, 8",
            "jmp 2b",
            "3:",
            "ret",
        )
    }

    /// The nonzero words among the `bytes` below the caller's stack pointer, but for the
    /// return address of this call.
    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "C" fn nonzero_below_sp(bytes: usize) -> usize {
        std::arch::naked_asm!(
            "lea rcx, [rsp + 8]",
            "sub rcx, rdi",
            "xor eax, eax",
            "2:",
            "cmp rcx, rsp",
            "jae 3f",
            "cmp qword ptr [rcx], 0",
            "je 4f",
            "inc rax",
            "4:",
            "add rcx, 8",
            "jmp 2b",
            "3:",
            "ret",
        )
    }

    /// The scrub leaves the range entirely zero, right below the caller included: a scrubber
    /// with a frame of its own left that frame's unwritten slots stale, and the collector's
    /// next frame inherited a dead cell's address from one of them.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[test]
    fn scrub_stack_leaves_no_word_right_below_the_caller() {
        const BYTES: usize = 16 * 1024;
        // SAFETY: these touch dead stack below the stack pointer, far above this stack's limit.
        unsafe { fill_below_sp(BYTES, 0x5afe_c0de_0bad_f00d) };
        super::scrub_stack(64 * 1024);
        let left = unsafe { nonzero_below_sp(BYTES) };
        assert_eq!(left, 0, "nonzero words below the caller after the scrub");
    }

    #[test]
    fn weakly_pinned_dead_cells_stay_reserved_until_unpinned() {
        // Like the allocation behind `std::rc::Weak`: a dead cell a weak handle may still
        // compare by address is finalized once and not reused until that handle is gone.
        FIN_COUNT.with(|c| c.set(0));
        let mut fx = Fixture::new();
        let anchor = fx.heap.alloc(fx.leaf, 4096);
        fx.roots[0] = tagged(anchor);
        let pinned = fx.heap.alloc(fx.leaf, 4096) as usize;
        let client = TableClient {
            roots: Vec::new(),
            side_slots: Vec::new(),
        };
        assert!(fx.heap.start_marking(false, &client));
        assert_eq!(fx.heap.dead_cells(fx.leaf), vec![pinned]);
        fx.heap.finish_collection(&|addr| addr == pinned);
        assert_eq!(FIN_COUNT.with(|c| c.get()), 1, "finalized when it died");
        assert!(
            fx.heap.is_allocated_word(pinned as u64),
            "address still reserved"
        );
        for _ in 0..14 {
            assert_ne!(
                fx.heap.alloc(fx.leaf, 4096) as usize,
                pinned,
                "no reuse while pinned"
            );
        }
        assert!(fx.heap.start_marking(false, &client));
        assert!(
            fx.heap.dead_cells(fx.leaf).len() >= 14,
            "zombies are not reported as newly dead"
        );
        assert!(!fx.heap.dead_cells(fx.leaf).contains(&pinned));
        assert!(fx.heap.is_dead_target(pinned));
        fx.heap.finish_collection(&|_| false);
        assert!(
            !fx.heap.is_allocated_word(pinned as u64),
            "freed once unpinned"
        );
        assert_eq!(
            FIN_COUNT.with(|c| c.get()),
            15,
            "a zombie is never finalized twice"
        );
    }

    #[test]
    fn nogc_region_defers_collection() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        fx.heap.set_stress_every(1);
        let before = fx.heap.stats().collections;
        {
            let _scope = fx.heap.nogc();
            for i in 0..50 {
                let p = fx.heap.alloc(fx.leaf, 16);
                black_box((p, i));
            }
            assert_eq!(fx.heap.stats().collections, before);
            assert_eq!(fx.heap.stats().deferred_collections, 1);
            assert!(fx.heap.collection_pending());
        }
        let _p = fx.heap.alloc(fx.leaf, 16);
        assert!(fx.heap.stats().collections > before, "deferred GC must run");
        black_box(&top);
    }

    /// The attached due word follows `collection_due` through each of its inputs: the byte
    /// budget (set by the allocation that spends it, not before), the stress interval, a
    /// deferred request (also one deferred while a collection runs) and the collections that
    /// reset them.
    #[test]
    fn due_word_tracks_collection_due() {
        // Declared before the fixture: the word outlives the heap.
        let word = Box::new(Cell::new(7u64));
        let mut fx = Fixture::new();
        // As in an engine heap: allocation never collects.
        fx.heap.set_collect_on_alloc(false);
        // SAFETY: `word` outlives the heap and nothing else writes it.
        unsafe { fx.heap.attach_due_word(NonNull::from(&*word)) };
        let due = |fx: &Fixture| {
            assert!(fx.heap.due_word_agrees());
            word.get() == 1
        };
        assert!(!due(&fx), "attaching writes the current state");

        fx.heap.set_heap_growth(4096, 100);
        let mut allocated = 0;
        while !fx.heap.collection_due() {
            assert!(!due(&fx));
            black_box(fx.alloc_node(0, 0));
            allocated += 16;
        }
        assert!(
            due(&fx),
            "the allocation that spends the budget sets the word"
        );
        assert!(allocated > 4096 && allocated <= 4096 + 16, "{allocated}");
        fx.collect();
        assert!(!due(&fx), "a collection resets the budget");

        fx.heap.set_stress_every(3);
        for _ in 0..2 {
            black_box(fx.alloc_node(0, 0));
            assert!(!due(&fx));
        }
        black_box(fx.alloc_node(0, 0));
        assert!(
            due(&fx),
            "the stress interval's last allocation sets the word"
        );
        fx.collect();
        assert!(!due(&fx));
        fx.heap.set_stress_every(1);
        black_box(fx.alloc_node(0, 0));
        assert!(due(&fx));
        fx.heap.set_stress_every(0);
        assert!(!due(&fx), "a setting recomputes the word");

        {
            let _scope = fx.heap.nogc();
            assert!(!fx.heap.collect_at_safepoint());
            assert!(due(&fx), "a deferred request sets the word");
        }
        assert!(due(&fx), "until a collection takes the request");
        fx.collect();
        assert!(!due(&fx));

        // A request deferred while a collection runs survives its sweep.
        assert!(fx.heap.start_marking(false, &NoClient));
        {
            let _scope = fx.heap.nogc();
            assert!(!fx.heap.start_marking(false, &NoClient));
        }
        fx.heap.finish_collection(&|_| false);
        assert!(due(&fx) && fx.heap.collection_pending());
        fx.collect();
        assert!(!due(&fx));
    }

    #[test]
    fn finalizers_run_for_dead_cells() {
        FIN_COUNT.with(|c| c.set(0));
        let mut fx = Fixture::new();
        let _a = fx.heap.alloc(fx.leaf, 16);
        let _b = fx.heap.alloc(fx.leaf, 16);
        fx.collect();
        assert_eq!(FIN_COUNT.with(|c| c.get()), 2);
    }

    #[test]
    fn large_objects_allocate_and_collect() {
        let mut fx = Fixture::new();
        let big = fx.heap.alloc(fx.leaf, 8192);
        set_word(big, 0, 42);
        fx.roots[0] = tagged(big);
        fx.collect();
        assert_eq!(word(big, 0), 42);
        fx.roots[0] = 0;
        fx.collect();
        assert_eq!(fx.live(), 0);
    }

    #[test]
    fn large_objects_resolve_interior_but_not_past_the_end() {
        let mut fx = Fixture::new();
        let big = fx.heap.alloc(fx.leaf, 64 * 1024);
        let other = fx.heap.alloc(fx.leaf, 8192);
        fx.roots[0] = big as u64 + 40_000;
        fx.roots[1] = other as u64 + 8192;
        fx.collect();
        assert!(fx.heap.is_allocated_word(big as u64));
        assert!(!fx.heap.is_allocated_word(other as u64));
    }

    #[test]
    fn sweep_hook_sees_liveness_for_side_tables() {
        let mut fx = Fixture::new();
        let live = fx.alloc_node(0, 1);
        let dead = fx.alloc_node(0, 2);
        let table: Rc<RefCell<FastMap<usize, u64>>> = Rc::default();
        table.borrow_mut().insert(live as usize, 10);
        table.borrow_mut().insert(dead as usize, 20);
        let t = Rc::clone(&table);
        let _hook = fx.heap.add_sweep_hook("test-table", move |is_live| {
            t.borrow_mut().retain(|&addr, _| is_live(addr as *const u8));
        });
        fx.roots[0] = tagged(live);
        fx.collect();
        let t = table.borrow();
        assert_eq!(t.len(), 1);
        assert_eq!(t[&(live as usize)], 10);
    }

    #[test]
    fn stress_graph_survives_collect_every_alloc() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        fx.heap.set_stress_every(1);
        // Build a 200-node chain under stress, rooting the head and keeping
        // every intermediate reachable through the chain.
        let mut head: *mut u8 = fx.alloc_node(0, 0);
        fx.roots[0] = tagged(head);
        for i in 1..200u64 {
            head = fx.alloc_node(tagged(head), i);
            fx.roots[0] = tagged(head);
            let _garbage = fx.heap.alloc(fx.leaf, 16);
        }
        assert!(fx.heap.stats().collections >= 398);
        fx.collect();
        assert_eq!(fx.live(), 200);
        let mut p = head as *const u8;
        let mut expect = 199u64;
        loop {
            assert_eq!(fx.val(p), expect);
            let next = word(p, 0);
            if next == 0 {
                break;
            }
            expect -= 1;
            p = (next & PAYLOAD_MASK) as *const u8;
        }
        assert_eq!(expect, 0);
        black_box(&top);
    }

    #[test]
    fn conservative_collection_requires_a_registered_stack() {
        let mut fx = Fixture::new();
        assert!(!fx.heap.collect());
        assert!(fx.heap.collection_pending());
        assert_eq!(fx.heap.stats().deferred_native_stack, 1);
        assert!(fx.heap.collect_at_safepoint());
        assert!(!fx.heap.collection_pending());
    }

    #[inline(never)]
    fn deep_collect(heap: &mut Heap, held: *const u8) -> bool {
        // Spill the held reference to this frame, then collect: the cell
        // is reachable *only* from here.
        let mut spill = [0u64; 4];
        unsafe { std::ptr::write_volatile(&mut spill[0], held as u64) };
        black_box(&spill);
        let ran = heap.collect();
        black_box(&spill);
        ran
    }

    #[test]
    fn thread_stack_registration_covers_every_frame() {
        std::thread::spawn(|| {
            assert!(register_thread_native_stack());
            assert!(register_thread_native_stack(), "idempotent");
            let mut fx = Fixture::new();
            let a = fx.alloc_node(0, 9);
            assert!(deep_collect(&mut fx.heap, a));
            assert!(fx.heap.is_allocated_word(a as u64));
            assert_eq!(fx.val(a), 9);
        })
        .join()
        .expect("collector thread");
    }

    #[test]
    fn stack_resident_reference_roots_cell() {
        let mut fx = Fixture::new();
        let a = fx.alloc_node(0, 5);
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        assert!(deep_collect(&mut fx.heap, a));
        assert_eq!(fx.live(), 1);
        assert_eq!(fx.val(a), 5);
        // Precise collections ignore native frames: deterministic death.
        fx.collect();
        assert_eq!(fx.live(), 0);
        black_box(&top);
    }

    struct Holder {
        field: *mut u8,
    }

    /// Stress collects before every allocation; each earlier cell is held
    /// only by this frame, in one recognized form (Decision 0 (b)).
    #[inline(never)]
    fn hold_recognized_forms(fx: &mut Fixture) -> usize {
        let raw = fx.heap.alloc(fx.node, 16);
        let tag = tagged(fx.heap.alloc(fx.node, 16));
        let interior = unsafe { fx.heap.alloc(fx.node, 32).add(24) };
        let holder = Holder {
            field: fx.heap.alloc(fx.node, 16),
        };
        let last = fx.heap.alloc(fx.node, 16);
        let words = [
            raw as u64,
            tag,
            interior as u64,
            holder.field as u64,
            last as u64,
        ];
        let survived = words
            .iter()
            .filter(|&&w| fx.heap.is_allocated_word(w))
            .count();
        black_box((raw, tag, interior, &holder, last));
        survived
    }

    #[test]
    fn recognized_forms_survive_allocation_triggered_collections() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        fx.heap.set_stress_every(1);
        let before = fx.heap.stats().collections;
        assert_eq!(hold_recognized_forms(&mut fx), 5);
        assert!(fx.heap.stats().collections >= before + 4);
        black_box(&top);
    }

    /// Ten references live across nine allocation-triggered collections;
    /// optimized builds keep them in callee-saved registers.
    #[inline(never)]
    fn hold_ten_references(heap: &mut Heap, desc: DescId) -> usize {
        let a = black_box(heap.alloc(desc, 16));
        let b = black_box(heap.alloc(desc, 16));
        let c = black_box(heap.alloc(desc, 16));
        let d = black_box(heap.alloc(desc, 16));
        let e = black_box(heap.alloc(desc, 16));
        let f = black_box(heap.alloc(desc, 16));
        let g = black_box(heap.alloc(desc, 16));
        let h = black_box(heap.alloc(desc, 16));
        let i = black_box(heap.alloc(desc, 16));
        let j = black_box(heap.alloc(desc, 16));
        [a, b, c, d, e, f, g, h, i, j]
            .iter()
            .filter(|&&p| heap.is_allocated_word(p as u64))
            .count()
    }

    #[test]
    fn register_held_references_survive_allocation_triggered_collections() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        fx.heap.set_stress_every(1);
        assert_eq!(hold_ten_references(&mut fx.heap, fx.leaf), 10);
        black_box(&top);
    }

    #[cfg(target_arch = "aarch64")]
    extern "C" fn collect_trampoline(heap: *mut std::ffi::c_void) {
        let heap = unsafe { &mut *(heap as *mut Heap) };
        assert!(heap.collect());
    }

    /// Collect while the only reference sits in d8, a callee-saved register
    /// no Rust frame on the collection path saves: only the register spill
    /// can find it. Returns d8 after the call.
    #[cfg(target_arch = "aarch64")]
    #[inline(never)]
    fn collect_with_reference_only_in_d8(heap: &mut Heap, hidden: u64) -> u64 {
        let found: u64;
        unsafe {
            std::arch::asm!(
                "mvn x9, {hidden}",
                "fmov d8, x9",
                "bl {trampoline}",
                "fmov x9, d8",
                hidden = in(reg) hidden,
                trampoline = sym collect_trampoline,
                in("x0") heap as *mut Heap as *mut std::ffi::c_void,
                lateout("x9") found,
                out("d8") _,
                clobber_abi("C"),
            );
        }
        found
    }

    /// Overwrite stale stack below the caller so an old copy of a pointer
    /// cannot retain it by accident.
    #[inline(never)]
    fn scrub_stack() {
        let buf = [0u8; 16 * 1024];
        black_box(&buf);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn reference_only_in_a_callee_saved_register_is_found() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        let hidden = alloc_hidden(&mut fx.heap, fx.leaf, 16);
        scrub_stack();
        let found = collect_with_reference_only_in_d8(&mut fx.heap, hidden);
        assert_eq!(found, !hidden, "d8 is callee-saved");
        assert_eq!(fx.heap.stats().collections, 1);
        assert!(fx.heap.is_allocated_word(found), "register spill missed d8");
        black_box(&top);
    }

    #[test]
    fn collecting_on_registered_segments_scans_every_stack_below() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        let mut entry_slot = [0u64; 2];
        stash(&mut entry_slot, alloc_hidden(&mut fx.heap, fx.leaf, 16));
        let live = grow(256 * 1024, || {
            let mut segment_slot = [0u64; 2];
            stash(&mut segment_slot, alloc_hidden(&mut fx.heap, fx.leaf, 16));
            let live = grow(256 * 1024, || {
                assert!(fx.heap.collect());
                fx.heap.stats().live_cells
            });
            black_box(&segment_slot);
            live
        });
        black_box((&entry_slot, &top));
        assert_eq!(
            live, 2,
            "references held only on the entry stack and the outer segment"
        );
    }

    #[test]
    fn collecting_on_an_unregistered_segment_defers() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        let ran = stacker::grow(256 * 1024, || fx.heap.collect());
        assert!(
            !ran,
            "an unregistered segment must defer, not be scanned blindly"
        );
        assert_eq!(fx.heap.stats().deferred_native_stack, 1);
        assert!(
            fx.heap.collect(),
            "back on the entry stack the collection runs"
        );
        black_box(&top);
    }

    #[cfg(target_os = "linux")]
    fn mapping_containing(addr: usize) -> Option<(usize, usize)> {
        let maps = std::fs::read_to_string("/proc/self/maps").ok()?;
        maps.lines().find_map(|line| {
            let (lo, hi) = line.split_whitespace().next()?.split_once('-')?;
            let lo = usize::from_str_radix(lo, 16).ok()?;
            let hi = usize::from_str_radix(hi, 16).ok()?;
            (lo..hi).contains(&addr).then_some((lo, hi))
        })
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn registered_segment_bounds_match_the_mapping() {
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        for size in [64 * 1024, 100_000, 1 << 20] {
            let (region, mapping) = grow(size, || {
                let sp = native::current_sp();
                (native::innermost_region(), mapping_containing(sp))
            });
            assert!(region.is_some());
            assert_eq!(region, mapping, "segment of {size} bytes");
        }
        black_box(&top);
    }

    #[test]
    fn dropping_or_repointing_a_root_range_takes_effect() {
        let mut fx = Fixture::new();
        let a = fx.heap.alloc(fx.leaf, 16);
        let first = Box::new([a as u64]);
        let range = unsafe {
            fx.heap
                .add_root_range("r", first.as_ptr(), first.as_ptr().add(1))
        };
        fx.collect();
        assert!(fx.heap.is_allocated_word(a as u64));
        let b = fx.heap.alloc(fx.leaf, 16);
        let second = Box::new([b as u64]);
        unsafe { range.set(second.as_ptr(), second.as_ptr().add(1)) };
        fx.collect();
        assert!(!fx.heap.is_allocated_word(a as u64));
        assert!(fx.heap.is_allocated_word(b as u64));
        drop(range);
        fx.collect();
        assert_eq!(fx.live(), 0);
    }

    #[test]
    fn dropped_trace_roots_and_sources_stop_rooting() {
        let mut fx = Fixture::new();
        let a = fx.heap.alloc(fx.leaf, 16) as u64;
        let b = fx.heap.alloc(fx.leaf, 16) as u64;
        let root = fx
            .heap
            .add_trace_root("a", move |visit| visit(a as *const u8));
        let source = fx
            .heap
            .add_edge_source("b", move |sink| sink.edge(std::ptr::null(), b));
        fx.collect();
        assert_eq!(fx.live(), 2);
        drop((root, source));
        fx.collect();
        assert_eq!(fx.live(), 0);
    }

    #[test]
    fn a_hook_dropping_another_hook_mid_pass_is_deferred() {
        let mut fx = Fixture::new();
        let calls = Rc::new(Cell::new(0));
        let victim: Rc<RefCell<Option<SweepHook>>> = Rc::default();
        let slot = Rc::clone(&victim);
        let _killer = fx.heap.add_sweep_hook("killer", move |_| {
            slot.borrow_mut().take();
        });
        let counter = Rc::clone(&calls);
        *victim.borrow_mut() = Some(fx.heap.add_sweep_hook("victim", move |_| {
            counter.set(counter.get() + 1);
        }));
        fx.collect();
        fx.collect();
        assert_eq!(calls.get(), 0);
        assert!(victim.borrow().is_none());
    }

    #[test]
    fn dropping_the_heap_finalizes_every_remaining_cell() {
        FIN_COUNT.with(|c| c.set(0));
        {
            let mut fx = Fixture::new();
            let a = fx.heap.alloc(fx.leaf, 16);
            fx.roots[0] = tagged(a);
            fx.heap.alloc(fx.leaf, 16);
            fx.heap.alloc(fx.leaf, 8192);
        }
        assert_eq!(FIN_COUNT.with(|c| c.get()), 3);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "I9 violation")]
    fn collecting_with_an_outstanding_borrow_asserts_in_debug() {
        let mut heap = Heap::new();
        let _borrow = heap.borrow_scope();
        heap.collect_at_safepoint();
    }

    #[test]
    #[cfg(not(debug_assertions))]
    fn collecting_with_an_outstanding_borrow_defers_in_release() {
        let mut heap = Heap::new();
        let borrow = heap.borrow_scope();
        assert!(!heap.collect_at_safepoint());
        drop(borrow);
        assert!(heap.collect_at_safepoint());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "no-GC region still open")]
    fn leaked_nogc_region_is_caught_at_quiescence() {
        let heap = Heap::new();
        std::mem::forget(heap.nogc());
        heap.debug_assert_quiescent();
    }

    #[test]
    fn collection_budget_grows_with_the_live_heap() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        fx.build_chain(1 << 18); // 4 MiB live
        let before = fx.heap.stats().collections;
        for _ in 0..1 << 19 {
            black_box(fx.heap.alloc(fx.leaf, 16)); // 8 MiB of garbage
        }
        let during = fx.heap.stats().collections - before;
        assert!(
            during <= 3,
            "8 MiB churn over a 4 MiB live heap took {during} collections"
        );
        black_box(&top);
    }

    #[test]
    fn retained_large_allocations_do_not_collect_every_time() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        for slot in 0..16 {
            let p = fx.heap.alloc(fx.leaf, 2 << 20);
            fx.roots[slot] = tagged(p);
        }
        let collections = fx.heap.stats().collections;
        assert!(
            collections <= 6,
            "16 retained 2 MiB allocations took {collections} collections"
        );
        black_box(&top);
    }

    #[test]
    fn freed_cells_are_reused_before_new_pages_and_empty_pages_are_released() {
        let mut fx = Fixture::new();
        for i in 0..48 {
            let p = fx.heap.alloc(fx.leaf, 4096);
            if i % 2 == 0 {
                fx.roots[i / 2] = tagged(p);
            }
        }
        fx.collect();
        assert_eq!((fx.heap.stats().pages_live, fx.live()), (3, 24));
        for _ in 0..24 {
            fx.heap.alloc(fx.leaf, 4096);
        }
        assert_eq!(fx.heap.stats().pages_live, 3, "freed cells reused first");
        fx.heap.alloc(fx.leaf, 4096);
        assert_eq!(fx.heap.stats().pages_live, 4);
        fx.roots.fill(0);
        fx.collect();
        assert_eq!(fx.heap.stats().pages_live, 0, "empty pages released");
        let p = fx.heap.alloc(fx.leaf, 4096);
        assert!(fx.heap.is_allocated_word(p as u64));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "heap verifier")]
    fn verifier_rejects_dangling_edge_in_debug() {
        let mut fx = Fixture::new();
        let a = fx.alloc_node(0x0000_0000_00de_ad00, 1);
        fx.roots[0] = tagged(a);
        fx.collect();
    }

    // Timing probes behind the S1 review numbers, run with
    // `cargo test --release -p trust-js --lib --features embed,gc-heap
    //  gc_core::tests::perf_ -- --ignored --nocapture --test-threads=1`.

    fn perf_heap() -> (Heap, DescId, DescId) {
        let mut heap = Heap::new();
        let node = heap.register_desc(LayoutDesc {
            name: "node",
            trace: trace_node,
            weak: None,
            finalize: None,
        });
        let blob = heap.register_desc(LayoutDesc {
            name: "blob",
            trace: trace_none,
            weak: None,
            finalize: None,
        });
        (heap, node, blob)
    }

    #[test]
    #[ignore]
    fn perf_churn_over_a_live_heap() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        let start = Instant::now();
        fx.build_chain(1 << 20);
        let build = start.elapsed();
        let before = fx.heap.stats().collections;
        let start = Instant::now();
        for _ in 0..1 << 20 {
            black_box(fx.heap.alloc(fx.node, 16));
        }
        eprintln!(
            "perf churn: 1M-node live chain built in {build:?}; 1M x 16 B churn {:?}, {} collections",
            start.elapsed(),
            fx.heap.stats().collections - before
        );
        black_box(&top);
    }

    #[test]
    #[ignore]
    fn perf_collect_with_sweep_hook() {
        let mut fx = Fixture::new();
        {
            let _scope = fx.heap.nogc();
            fx.build_chain(1 << 20);
        }
        fx.collect();
        let start = Instant::now();
        fx.collect();
        let bare = start.elapsed();
        let _hook = fx.heap.add_sweep_hook("noop", |is_live| {
            black_box(is_live(std::ptr::null()));
        });
        let start = Instant::now();
        fx.collect();
        eprintln!(
            "perf collect, 1M live cells: no hooks {bare:?}, one no-op hook {:?}",
            start.elapsed()
        );
    }

    #[test]
    #[ignore]
    fn perf_large_objects_and_scanned_words() {
        for (label, size) in [("large 8 KiB", 8192usize), ("small 4 KiB", 4096)] {
            for count in [1000usize, 2000, 4000] {
                let (mut heap, _, blob) = perf_heap();
                let objects: Vec<u64> = {
                    let _scope = heap.nogc();
                    (0..count).map(|_| heap.alloc(blob, size) as u64).collect()
                };
                let noise: Vec<u64> = (0..32768u64).map(|i| i * 0x9e37_79b9).collect();
                let _objects = unsafe {
                    heap.add_root_range("objects", objects.as_ptr(), objects.as_ptr().add(count))
                };
                let _noise = unsafe {
                    heap.add_root_range("noise", noise.as_ptr(), noise.as_ptr().add(noise.len()))
                };
                heap.collect_at_safepoint();
                let start = Instant::now();
                heap.collect_at_safepoint();
                eprintln!(
                    "perf {label} x {count} + 32k words: collect {:?}",
                    start.elapsed()
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn perf_page_release() {
        for pages in [1000usize, 2000, 4000] {
            let (mut heap, _, blob) = perf_heap();
            {
                let _scope = heap.nogc();
                for _ in 0..pages * 16 {
                    black_box(heap.alloc(blob, 4096));
                }
            }
            let start = Instant::now();
            heap.collect_at_safepoint();
            eprintln!(
                "perf release {pages} empty pages: collect {:?}",
                start.elapsed()
            );
        }
    }

    #[test]
    #[ignore]
    fn perf_alloc_fast_path() {
        const N: usize = 1 << 20;
        let (mut heap, _, blob) = perf_heap();
        let _scope = heap.nogc();
        let start = Instant::now();
        for _ in 0..N {
            black_box(heap.alloc(blob, 32));
        }
        eprintln!(
            "perf alloc: {:.1} ns per 32 B cell",
            start.elapsed().as_nanos() as f64 / N as f64
        );
    }
}
