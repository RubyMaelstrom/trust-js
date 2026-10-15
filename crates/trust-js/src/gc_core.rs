//! P1-S1 heap core: pointer-based mark-sweep heap for the GC-managed
//! object model (docs/v8-chase-p1-plan-2026-10-15.txt, Decisions 0–4).
//! Feature `gc-heap`, default off; nothing executes on this heap yet.
//!
//! Design summary:
//! - Cells live in 64 KiB pages segregated by (size class, descriptor);
//!   page metadata is side table, so interior pointers resolve by masking
//!   to the page base. Large cells (> 4 KiB) get dedicated allocations.
//! - References are direct pointers (never handle-table indices); a
//!   tagged execution word masks to the same pointer (payload mask).
//! - Tracing is precise inside the heap (per-descriptor trace functions)
//!   and conservative at the roots (native stack + register snapshot +
//!   registered root ranges), per the Decision 0 scanning contract.
//! - Collection honors I9: it never runs inside a no-GC region
//!   (deferred via pending flag) and cells are allocated zeroed so a
//!   descriptor can trace a not-yet-initialized cell safely.
//! - Debug builds poison freed cells and can verify the marked graph.

// Until S2 wires execution to this heap, most of the surface is used
// only from tests (same posture as heap.rs/tagged.rs).
#![allow(dead_code)]

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::Cell;
use std::collections::HashMap;

pub(crate) const PAGE_SIZE: usize = 64 * 1024;
const PAGE_MASK: usize = !(PAGE_SIZE - 1);
const PAYLOAD_MASK: u64 = 0x0000_ffff_ffff_ffff;
const LARGE_THRESHOLD: usize = 4096;
const SIZE_CLASSES: [usize; 16] = [
    16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
];
#[cfg(debug_assertions)]
const POISON: u64 = 0xdddd_dddd_dddd_dddd;

fn size_class(size: usize) -> Option<usize> {
    SIZE_CLASSES.iter().position(|&c| size <= c)
}

/// Trace callback: report every heap edge of the cell at `cell` by
/// calling `visit(edge_target)`. Must tolerate a zeroed cell (I9:
/// allocation returns zeroed memory; publication follows init).
pub(crate) type TraceFn = fn(*const u8, &mut dyn FnMut(*const u8));
/// Finalize callback for a dead cell, run during sweep. Must not
/// allocate from this heap or resurrect the cell.
pub(crate) type FinalizeFn = fn(*mut u8);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DescId(pub(crate) u32);

pub(crate) struct LayoutDesc {
    pub(crate) name: &'static str,
    pub(crate) trace: TraceFn,
    pub(crate) finalize: Option<FinalizeFn>,
}

struct PageMeta {
    base: usize,
    cell_size: usize,
    cells: usize,
    desc: DescId,
    marks: Vec<u64>,
    /// Allocated (handed-out) cells. Conservative marking must ignore
    /// free cells: a stale stack word resolving into a free cell would
    /// otherwise "mark" free-list linkage and trace it as payload.
    allocated: Vec<u64>,
    /// Index of first free cell, or -1. Free cells thread their next
    /// index+1 through their first word (0 = end).
    free_head: i32,
    live: usize,
}

impl PageMeta {
    fn cell_ptr(&self, index: usize) -> *mut u8 {
        (self.base + index * self.cell_size) as *mut u8
    }
    fn marked(&self, index: usize) -> bool {
        self.marks[index / 64] >> (index % 64) & 1 == 1
    }
    fn set_mark(&mut self, index: usize) {
        self.marks[index / 64] |= 1u64 << (index % 64);
    }
    fn is_allocated(&self, index: usize) -> bool {
        self.allocated[index / 64] >> (index % 64) & 1 == 1
    }
    fn set_allocated(&mut self, index: usize, yes: bool) {
        if yes {
            self.allocated[index / 64] |= 1u64 << (index % 64);
        } else {
            self.allocated[index / 64] &= !(1u64 << (index % 64));
        }
    }
}

struct LargeObj {
    ptr: *mut u8,
    size: usize,
    desc: DescId,
    marked: bool,
}

type TraceRootFn = Box<dyn FnMut(&mut dyn FnMut(*const u8))>;
type SweepHookFn = Box<dyn FnMut(&dyn Fn(*const u8) -> bool)>;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct GcStats {
    pub(crate) allocations: u64,
    pub(crate) collections: u64,
    pub(crate) cells_marked: u64,
    pub(crate) cells_swept: u64,
    pub(crate) pages_live: usize,
    pub(crate) live_cells: usize,
    pub(crate) live_bytes: usize,
    pub(crate) deferred_collections: u64,
}

pub(crate) struct Heap {
    descs: Vec<LayoutDesc>,
    pages: Vec<PageMeta>,
    page_map: HashMap<usize, usize>,
    current: HashMap<(usize, DescId), usize>,
    larges: Vec<LargeObj>,
    root_ranges: Vec<(String, *const u64, *const u64)>,
    trace_roots: Vec<(String, TraceRootFn)>,
    sweep_hooks: Vec<(String, SweepHookFn)>,
    ephemerons: Vec<(*const u8, *const u8)>,
    weak_slots: Vec<*mut u64>,
    stack_top: Cell<usize>,
    extra_stack_ranges: Vec<(usize, usize)>,
    /// Shared with NoGcScope guards so a guard can be alive while
    /// `alloc(&mut self)` runs inside the region.
    control: std::rc::Rc<Control>,
    bytes_since_gc: usize,
    gc_threshold: usize,
    stress_every: usize,
    allocs_since_gc: usize,
    stats: GcStats,
}

impl Default for Heap {
    fn default() -> Self {
        Self::new()
    }
}

impl Heap {
    pub(crate) fn new() -> Self {
        Self {
            descs: Vec::new(),
            pages: Vec::new(),
            page_map: HashMap::new(),
            current: HashMap::new(),
            larges: Vec::new(),
            root_ranges: Vec::new(),
            trace_roots: Vec::new(),
            sweep_hooks: Vec::new(),
            ephemerons: Vec::new(),
            weak_slots: Vec::new(),
            stack_top: Cell::new(0),
            extra_stack_ranges: Vec::new(),
            control: std::rc::Rc::new(Control::default()),
            bytes_since_gc: 0,
            gc_threshold: 1024 * 1024,
            stress_every: 0,
            allocs_since_gc: 0,
            stats: GcStats::default(),
        }
    }

    pub(crate) fn register_desc(&mut self, desc: LayoutDesc) -> DescId {
        let id = DescId(self.descs.len() as u32);
        self.descs.push(desc);
        id
    }

    pub(crate) fn set_stack_top(&self, top: *const u64) {
        self.stack_top.set(top as usize);
    }

    /// Register an extra scannable stack range (e.g. a grown segment
    /// with exact bounds from the segment registry). Bounds are the
    /// caller's contract responsibility; never guess them.
    pub(crate) fn register_stack_range(&mut self, lo: *const u64, hi: *const u64) {
        self.extra_stack_ranges.push((lo as usize, hi as usize));
    }

    /// Register a Rust-heap root range of tagged words (VM operand
    /// stacks, frame slot arrays). Scanned precisely-as-tagged.
    pub(crate) fn register_root_range(&mut self, name: &str, lo: *const u64, hi: *const u64) {
        self.root_ranges.push((name.to_string(), lo, hi));
    }

    /// Register a traced Rust-heap container root (Decision 2 registry).
    pub(crate) fn register_trace_root(&mut self, name: &str, f: TraceRootFn) {
        self.trace_roots.push((name.to_string(), f));
    }

    /// Register a pre-sweep hook for address-keyed side tables (I8).
    /// The hook receives `is_live(cell_ptr)` valid for this cycle.
    pub(crate) fn register_sweep_hook(&mut self, name: &str, f: SweepHookFn) {
        self.sweep_hooks.push((name.to_string(), f));
    }

    pub(crate) fn add_ephemeron(&mut self, key: *const u8, value: *const u8) {
        self.ephemerons.push((key, value));
    }

    /// Register a weak slot: a word holding a tagged reference that the
    /// collector zeroes when its target is unmarked.
    pub(crate) fn add_weak_slot(&mut self, slot: *mut u64) {
        self.weak_slots.push(slot);
    }

    pub(crate) fn set_stress_every(&mut self, allocs: usize) {
        self.stress_every = allocs;
    }

    pub(crate) fn stats(&self) -> GcStats {
        let mut s = self.stats;
        s.pages_live = self.pages.len();
        s.live_cells = self.pages.iter().map(|p| p.live).sum::<usize>() + self.larges.len();
        s.live_bytes = self
            .pages
            .iter()
            .map(|p| p.live * p.cell_size)
            .sum::<usize>()
            + self.larges.iter().map(|l| l.size).sum::<usize>();
        s
    }

    /// Enter a bounded no-GC region (I9). Allocation proceeds inside;
    /// collection is deferred until the outermost region exits *and* a
    /// later allocation or explicit collect observes the pending flag.
    pub(crate) fn nogc(&self) -> NoGcScope {
        self.control
            .nogc_depth
            .set(self.control.nogc_depth.get() + 1);
        NoGcScope {
            control: self.control.clone(),
        }
    }

    /// Allocate a zeroed cell for `desc`. May collect first (unless
    /// inside a no-GC region, where the request is deferred).
    /// The caller initializes, then publishes (I9).
    pub(crate) fn alloc(&mut self, desc: DescId, size: usize) -> *mut u8 {
        self.maybe_collect_for_alloc(size);
        let ptr = if size > LARGE_THRESHOLD {
            self.alloc_large(desc, size)
        } else {
            self.alloc_small(desc, size)
        };
        self.stats.allocations += 1;
        self.bytes_since_gc += size;
        self.allocs_since_gc += 1;
        if self.stress_every > 0 && self.allocs_since_gc >= self.stress_every {
            // Stress: collect *after* handing out the cell would free it
            // before the caller can root it; collect is armed for the
            // next allocation instead (see maybe_collect_for_alloc).
            self.control.pending_collect.set(true);
        }
        ptr
    }

    fn maybe_collect_for_alloc(&mut self, upcoming: usize) {
        let due = self.control.pending_collect.get()
            || self.bytes_since_gc + upcoming > self.gc_threshold
            || (self.stress_every > 0 && self.allocs_since_gc >= self.stress_every);
        if !due {
            return;
        }
        if self.control.nogc_depth.get() > 0 {
            self.control.pending_collect.set(true);
            self.stats.deferred_collections += 1;
            return;
        }
        self.collect();
    }

    fn alloc_small(&mut self, desc: DescId, size: usize) -> *mut u8 {
        let class = size_class(size).expect("small size has a class");
        let cell_size = SIZE_CLASSES[class];
        let key = (class, desc);
        if let Some(&meta_idx) = self.current.get(&key) {
            if let Some(ptr) = self.take_free(meta_idx) {
                return ptr;
            }
        }
        // Reuse any non-current page of this (class, desc) with space,
        // e.g. after a sweep freed cells in older pages.
        for idx in 0..self.pages.len() {
            if self.pages[idx].cell_size == cell_size && self.pages[idx].desc == desc {
                if let Some(ptr) = self.take_free(idx) {
                    self.current.insert(key, idx);
                    return ptr;
                }
            }
        }
        let idx = self.new_page(cell_size, desc);
        self.current.insert(key, idx);
        self.take_free(idx).expect("fresh page has free cells")
    }

    fn take_free(&mut self, meta_idx: usize) -> Option<*mut u8> {
        let page = &mut self.pages[meta_idx];
        if page.free_head < 0 {
            return None;
        }
        let index = page.free_head as usize;
        let ptr = page.cell_ptr(index);
        let next = unsafe { std::ptr::read(ptr as *const u64) };
        page.free_head = if next == 0 { -1 } else { (next - 1) as i32 };
        page.set_allocated(index, true);
        page.live += 1;
        unsafe {
            std::ptr::write_bytes(ptr, 0, page.cell_size);
        }
        Some(ptr)
    }

    fn new_page(&mut self, cell_size: usize, desc: DescId) -> usize {
        let layout = Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).unwrap();
        let base = unsafe { alloc_zeroed(layout) };
        assert!(!base.is_null(), "GC page allocation failed");
        let cells = PAGE_SIZE / cell_size;
        // Thread the free list: cell i -> i+1 (stored as i+2), last -> 0.
        for i in 0..cells {
            let p = unsafe { base.add(i * cell_size) } as *mut u64;
            let next = if i + 1 < cells { (i + 2) as u64 } else { 0 };
            unsafe { std::ptr::write(p, next) };
        }
        let idx = self.pages.len();
        self.pages.push(PageMeta {
            base: base as usize,
            cell_size,
            cells,
            desc,
            marks: vec![0; cells.div_ceil(64)],
            allocated: vec![0; cells.div_ceil(64)],
            free_head: 0,
            live: 0,
        });
        self.page_map.insert(base as usize, idx);
        idx
    }

    fn alloc_large(&mut self, desc: DescId, size: usize) -> *mut u8 {
        let layout = Layout::from_size_align(size, 16).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "GC large allocation failed");
        self.larges.push(LargeObj {
            ptr,
            size,
            desc,
            marked: false,
        });
        ptr
    }

    /// Resolve a candidate word (raw pointer or tagged word) to the
    /// start of its cell, accepting interior addresses. `None` for
    /// non-heap words.
    pub(crate) fn resolve(&self, word: u64) -> Option<*const u8> {
        let addr = (word & PAYLOAD_MASK) as usize;
        if addr < 4096 {
            return None;
        }
        if let Some(&idx) = self.page_map.get(&(addr & PAGE_MASK)) {
            let page = &self.pages[idx];
            let offset = addr - page.base;
            if offset < page.cells * page.cell_size {
                return Some(page.cell_ptr(offset / page.cell_size) as *const u8);
            }
            return None;
        }
        for large in &self.larges {
            let lo = large.ptr as usize;
            if addr >= lo && addr < lo + large.size {
                return Some(large.ptr as *const u8);
            }
        }
        None
    }

    fn is_marked(&self, cell: *const u8) -> bool {
        let addr = cell as usize;
        if let Some(&idx) = self.page_map.get(&(addr & PAGE_MASK)) {
            let page = &self.pages[idx];
            return page.marked((addr - page.base) / page.cell_size);
        }
        self.larges
            .iter()
            .any(|l| l.ptr as usize == addr && l.marked)
    }

    fn mark_cell(&mut self, cell: *const u8, worklist: &mut Vec<*const u8>) {
        let addr = cell as usize;
        if let Some(&idx) = self.page_map.get(&(addr & PAGE_MASK)) {
            let page = &mut self.pages[idx];
            let index = (addr - page.base) / page.cell_size;
            if !page.is_allocated(index) {
                return; // stale/conservative hit on a free cell: ignore
            }
            if !page.marked(index) {
                page.set_mark(index);
                self.stats.cells_marked += 1;
                worklist.push(page.cell_ptr(index) as *const u8);
            }
            return;
        }
        for large in &mut self.larges {
            if large.ptr as usize == addr && !large.marked {
                large.marked = true;
                self.stats.cells_marked += 1;
                worklist.push(cell);
                return;
            }
        }
    }

    fn desc_of(&self, cell: *const u8) -> Option<DescId> {
        let addr = cell as usize;
        if let Some(&idx) = self.page_map.get(&(addr & PAGE_MASK)) {
            return Some(self.pages[idx].desc);
        }
        self.larges
            .iter()
            .find(|l| l.ptr as usize == addr)
            .map(|l| l.desc)
    }
}

#[derive(Default)]
pub(crate) struct Control {
    nogc_depth: Cell<u32>,
    pending_collect: Cell<bool>,
}

/// RAII no-GC region (I9). Collection requests inside are deferred.
/// Holds only the shared control block, so `alloc(&mut Heap)` remains
/// callable while the guard is alive.
pub(crate) struct NoGcScope {
    control: std::rc::Rc<Control>,
}

impl Drop for NoGcScope {
    fn drop(&mut self) {
        self.control
            .nogc_depth
            .set(self.control.nogc_depth.get() - 1);
    }
}

/// Capture callee-saved registers into `buf` so the stack scan sees
/// references the compiler kept in registers at the collection entry
/// (Decision 0 spill protocol). Caller-saved registers need no capture:
/// at the allocating call boundary they are dead or already spilled.
#[cfg(target_arch = "aarch64")]
fn spill_registers(buf: &mut [u64; 24]) {
    unsafe {
        std::arch::asm!(
            "stp x19, x20, [x0]",
            "stp x21, x22, [x0, #16]",
            "stp x23, x24, [x0, #32]",
            "stp x25, x26, [x0, #48]",
            "stp x27, x28, [x0, #64]",
            "stp x29, x30, [x0, #80]",
            "stp d8, d9, [x0, #96]",
            "stp d10, d11, [x0, #112]",
            "stp d12, d13, [x0, #128]",
            "stp d14, d15, [x0, #144]",
            in("x0") buf.as_mut_ptr(),
            options(nostack),
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn spill_registers(buf: &mut [u64; 24]) {
    // No register capture on this target yet: stack scan only. The
    // contract requires a per-target spill before this heap may become
    // authoritative there (x64 is re-validated at S6 on zombie).
    buf.fill(0);
    std::hint::black_box(buf);
}

impl Heap {
    /// Run a full collection now. No-op (deferred) inside a no-GC region.
    pub(crate) fn collect(&mut self) {
        if self.control.nogc_depth.get() > 0 {
            self.control.pending_collect.set(true);
            self.stats.deferred_collections += 1;
            return;
        }
        self.control.pending_collect.set(false);
        self.stats.collections += 1;
        let mut worklist: Vec<*const u8> = Vec::new();
        self.mark_roots(&mut worklist);
        self.drain(&mut worklist);
        self.mark_ephemerons(&mut worklist);
        self.clear_weak_slots();
        #[cfg(debug_assertions)]
        self.verify();
        self.run_sweep_hooks();
        self.sweep();
        self.bytes_since_gc = 0;
        self.allocs_since_gc = 0;
    }

    fn mark_word(&mut self, word: u64, worklist: &mut Vec<*const u8>) {
        if let Some(cell) = self.resolve(word) {
            self.mark_cell(cell, worklist);
        }
    }

    fn mark_roots(&mut self, worklist: &mut Vec<*const u8>) {
        // Register snapshot first (Decision 0 spill protocol).
        let mut regs = [0u64; 24];
        spill_registers(&mut regs);
        // Registered tagged root ranges.
        let ranges = self.root_ranges.clone();
        for (_, lo, hi) in &ranges {
            let mut p = *lo;
            while p < *hi {
                let word = unsafe { std::ptr::read(p) };
                self.mark_word(word, worklist);
                p = unsafe { p.add(1) };
            }
        }
        // Traced container roots. The callbacks borrow disjoint root
        // storage; collect their edges via a temporary buffer to avoid
        // holding &mut self across the callback.
        let mut edges: Vec<*const u8> = Vec::new();
        let mut roots = std::mem::take(&mut self.trace_roots);
        for (_, f) in &mut roots {
            f(&mut |cell| edges.push(cell));
        }
        self.trace_roots = roots;
        for cell in edges {
            if let Some(start) = self.resolve(cell as u64) {
                self.mark_cell(start, worklist);
            }
        }
        // Conservative stack scan: [current SP approx, stack_top) plus
        // registered extra ranges (grown segments with exact bounds).
        let top = self.stack_top.get();
        if top != 0 {
            let marker = 0u64;
            let bottom = &marker as *const u64 as usize;
            self.scan_range(bottom, top, worklist);
            let extra = self.extra_stack_ranges.clone();
            for (lo, hi) in extra {
                self.scan_range(lo, hi, worklist);
            }
        }
        for word in regs {
            self.mark_word(word, worklist);
        }
        std::hint::black_box(&regs);
    }

    fn scan_range(&mut self, lo: usize, hi: usize, worklist: &mut Vec<*const u8>) {
        let mut p = lo as *const u64;
        let end = hi as *const u64;
        while p < end {
            let word = unsafe { std::ptr::read(p) };
            self.mark_word(word, worklist);
            p = unsafe { p.add(1) };
        }
    }

    fn drain(&mut self, worklist: &mut Vec<*const u8>) {
        let mut edges: Vec<*const u8> = Vec::new();
        while let Some(cell) = worklist.pop() {
            let Some(desc) = self.desc_of(cell) else {
                continue;
            };
            edges.clear();
            (self.descs[desc.0 as usize].trace)(cell, &mut |edge| edges.push(edge));
            for i in 0..edges.len() {
                let edge = edges[i];
                if let Some(start) = self.resolve(edge as u64) {
                    self.mark_cell(start, worklist);
                }
            }
        }
    }

    fn mark_ephemerons(&mut self, worklist: &mut Vec<*const u8>) {
        // Fixed point: a value becomes live only if its key is live for
        // reasons other than this ephemeron itself.
        loop {
            let mut progress = false;
            for i in 0..self.ephemerons.len() {
                let (key, value) = self.ephemerons[i];
                let key_cell = self.resolve(key as u64);
                let value_cell = self.resolve(value as u64);
                if let (Some(k), Some(v)) = (key_cell, value_cell) {
                    if self.is_marked(k) && !self.is_marked(v) {
                        self.mark_cell(v, worklist);
                        progress = true;
                    }
                }
            }
            if !progress {
                break;
            }
            self.drain(worklist);
        }
    }

    fn clear_weak_slots(&mut self) {
        for i in 0..self.weak_slots.len() {
            let slot = self.weak_slots[i];
            let word = unsafe { std::ptr::read(slot) };
            if word != 0 {
                let live = self.resolve(word).is_some_and(|cell| self.is_marked(cell));
                if !live {
                    unsafe { std::ptr::write(slot, 0) };
                }
            }
        }
    }

    /// Structural check: every edge from every marked cell resolves to
    /// a marked cell. Cannot detect descriptor *omissions* (that proof
    /// is the registry audit vs the S0 inventory — Decision 2).
    pub(crate) fn verify(&self) {
        let mut edges: Vec<*const u8> = Vec::new();
        let check = |heap: &Heap, from: *const u8, desc: DescId, edges: &mut Vec<*const u8>| {
            edges.clear();
            (heap.descs[desc.0 as usize].trace)(from, &mut |e| edges.push(e));
            for &edge in edges.iter() {
                if edge.is_null() {
                    continue;
                }
                match heap.resolve(edge as u64) {
                    Some(cell) => assert!(
                        heap.is_marked(cell),
                        "heap verifier: marked cell {from:p} ({}) references unmarked {cell:p}",
                        heap.descs[desc.0 as usize].name
                    ),
                    None => panic!(
                        "heap verifier: marked cell {from:p} ({}) references non-heap {edge:p}",
                        heap.descs[desc.0 as usize].name
                    ),
                }
            }
        };
        for page in &self.pages {
            for i in 0..page.cells {
                if page.marked(i) {
                    check(self, page.cell_ptr(i) as *const u8, page.desc, &mut edges);
                }
            }
        }
        for large in &self.larges {
            if large.marked {
                check(self, large.ptr as *const u8, large.desc, &mut edges);
            }
        }
    }

    fn run_sweep_hooks(&mut self) {
        if self.sweep_hooks.is_empty() {
            return;
        }
        // Hooks receive a liveness predicate over resolved cells.
        let live_set: std::collections::HashSet<usize> = {
            let mut set = std::collections::HashSet::new();
            for page in &self.pages {
                for i in 0..page.cells {
                    if page.marked(i) {
                        set.insert(page.cell_ptr(i) as usize);
                    }
                }
            }
            for large in &self.larges {
                if large.marked {
                    set.insert(large.ptr as usize);
                }
            }
            set
        };
        let mut hooks = std::mem::take(&mut self.sweep_hooks);
        for (_, hook) in &mut hooks {
            hook(&|cell| live_set.contains(&(cell as usize)));
        }
        self.sweep_hooks = hooks;
    }

    fn sweep(&mut self) {
        let layout = Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).unwrap();
        let mut release: Vec<usize> = Vec::new();
        let mut swept: u64 = 0;
        for (idx, page) in self.pages.iter_mut().enumerate() {
            let finalize = self.descs[page.desc.0 as usize].finalize;
            let mut head: i32 = -1;
            let mut live = 0usize;
            for i in 0..page.cells {
                if page.marked(i) {
                    live += 1;
                }
            }
            // Dead allocated cells get finalized, poisoned (debug),
            // and threaded onto the rebuilt free list.
            for i in 0..page.cells {
                if !page.marked(i) {
                    if page.is_allocated(i) {
                        swept += 1;
                        if let Some(fin) = finalize {
                            fin(page.cell_ptr(i));
                        }
                    }
                    #[cfg(debug_assertions)]
                    unsafe {
                        let words = page.cell_ptr(i) as *mut u64;
                        for w in 1..page.cell_size / 8 {
                            std::ptr::write(words.add(w), POISON);
                        }
                    }
                    page.set_allocated(i, false);
                    let p = page.cell_ptr(i) as *mut u64;
                    unsafe {
                        std::ptr::write(p, if head < 0 { 0 } else { (head + 1) as u64 });
                    }
                    head = i as i32;
                }
            }
            page.free_head = head;
            page.live = live;
            page.marks.iter_mut().for_each(|m| *m = 0);
            if live == 0 {
                release.push(idx);
            }
        }
        // Release fully-empty pages (keep at most one spare per class
        // would be a tuning choice; S1 returns all to the OS allocator).
        release.sort_unstable_by(|a, b| b.cmp(a));
        for idx in release {
            let page = self.pages.remove(idx);
            self.page_map.remove(&page.base);
            self.current.retain(|_, v| *v != idx);
            for v in self.current.values_mut() {
                if *v > idx {
                    *v -= 1;
                }
            }
            // Rebuild map indices after removal shifts.
            for (i, p) in self.pages.iter().enumerate() {
                self.page_map.insert(p.base, i);
            }
            unsafe { dealloc(page.base as *mut u8, layout) };
        }
        self.stats.cells_swept += swept;
        // Large objects.
        let mut i = 0;
        while i < self.larges.len() {
            if self.larges[i].marked {
                self.larges[i].marked = false;
                i += 1;
            } else {
                let large = self.larges.remove(i);
                self.stats.cells_swept += 1;
                if let Some(fin) = self.descs[large.desc.0 as usize].finalize {
                    fin(large.ptr);
                }
                let layout = Layout::from_size_align(large.size, 16).unwrap();
                unsafe { dealloc(large.ptr, layout) };
            }
        }
        // Dead ephemerons/weak slots stay registered; their words were
        // cleared or resolve to None in later cycles.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;

    const TAG: u64 = 0xfff9_0000_0000_0000;

    fn trace_node(cell: *const u8, visit: &mut dyn FnMut(*const u8)) {
        let next = unsafe { std::ptr::read(cell as *const u64) };
        if next != 0 {
            visit((next & PAYLOAD_MASK) as *const u8);
        }
    }
    fn trace_leaf(_: *const u8, _: &mut dyn FnMut(*const u8)) {}
    thread_local! {
        static FIN_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }
    fn fin_count(_: *mut u8) {
        FIN_COUNT.with(|c| c.set(c.get() + 1));
    }

    struct Fixture {
        heap: Heap,
        node: DescId,
        leaf: DescId,
        roots: Box<[u64; 64]>,
    }

    impl Fixture {
        fn new() -> Self {
            let mut heap = Heap::new();
            let node = heap.register_desc(LayoutDesc {
                name: "node",
                trace: trace_node,
                finalize: None,
            });
            let leaf = heap.register_desc(LayoutDesc {
                name: "leaf",
                trace: trace_leaf,
                finalize: Some(fin_count),
            });
            let roots = Box::new([0u64; 64]);
            heap.register_root_range("test-roots", roots.as_ptr(), unsafe {
                roots.as_ptr().add(64)
            });
            Self {
                heap,
                node,
                leaf,
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
        fn val(&self, p: *const u8) -> u64 {
            unsafe { std::ptr::read((p as *const u64).add(1)) }
        }
        fn root(&mut self, slot: usize, p: *const u8) {
            self.roots[slot] = TAG | (p as u64 & PAYLOAD_MASK);
        }
        /// Collect on a fresh thread: death assertions are meaningless
        /// while this frame's registers/stack still hold the pointers
        /// (conservative retention is correct behavior there).
        fn collect_clean(&mut self) {
            // SAFETY (test only): one thread at a time; joined before reuse.
            let addr = &mut self.heap as *mut Heap as usize;
            std::thread::spawn(move || {
                let heap = unsafe { &mut *(addr as *mut Heap) };
                heap.collect();
            })
            .join()
            .expect("collector thread");
        }
    }

    #[test]
    fn unreachable_cells_are_swept_and_chains_survive() {
        let mut fx = Fixture::new();
        let c = fx.alloc_node(0, 3);
        let b = fx.alloc_node(TAG | (c as u64 & PAYLOAD_MASK), 2);
        let a = fx.alloc_node(TAG | (b as u64 & PAYLOAD_MASK), 1);
        let _garbage = fx.alloc_node(0, 99);
        fx.root(0, a as *const u8);
        fx.heap.collect();
        assert_eq!(fx.heap.stats().live_cells, 3);
        assert_eq!(fx.val(a as *const u8), 1);
        assert_eq!(fx.val(b as *const u8), 2);
        assert_eq!(fx.val(c as *const u8), 3);
        assert!(fx.heap.stats().cells_swept >= 1);
    }

    #[test]
    fn interior_and_tagged_roots_resolve() {
        let mut fx = Fixture::new();
        let a = fx.alloc_node(0, 7);
        // Root holds a tagged interior pointer (second word of the cell).
        fx.roots[0] = TAG | ((a as u64 + 8) & PAYLOAD_MASK);
        fx.heap.collect();
        assert_eq!(fx.heap.stats().live_cells, 1);
        assert_eq!(fx.val(a as *const u8), 7);
    }

    #[test]
    fn ephemeron_value_lives_only_with_key() {
        let mut fx = Fixture::new();
        let key = fx.alloc_node(0, 1);
        let val = fx.alloc_node(0, 2);
        let key2 = fx.alloc_node(0, 3);
        let val2 = fx.alloc_node(0, 4);
        fx.heap.add_ephemeron(key as *const u8, val as *const u8);
        // Chain: val is itself the key for val2 (fixed point required).
        fx.heap.add_ephemeron(val as *const u8, val2 as *const u8);
        let val3 = fx.alloc_node(0, 5);
        fx.heap.add_ephemeron(key2 as *const u8, val3 as *const u8);
        fx.root(0, key as *const u8);
        fx.collect_clean();
        // key, val, val2 survive; key2 and its value do not.
        assert_eq!(fx.heap.stats().live_cells, 3);
        assert_eq!(fx.val(val2 as *const u8), 4);
    }

    #[test]
    fn weak_slots_clear_on_death_and_keep_on_life() {
        let mut fx = Fixture::new();
        let live = fx.alloc_node(0, 1);
        let dead = fx.alloc_node(0, 2);
        fx.roots[0] = live as u64;
        fx.roots[1] = dead as u64;
        fx.heap.add_weak_slot(&mut fx.roots[0] as *mut u64);
        fx.heap.add_weak_slot(&mut fx.roots[1] as *mut u64);
        // roots[0] also a strong root via the range; make live reachable
        // only strongly, dead only weakly: move dead out of the range.
        let mut weak_dead = dead as u64;
        fx.heap.add_weak_slot(&mut weak_dead as *mut u64);
        fx.roots[1] = 0;
        fx.collect_clean();
        assert_ne!(fx.roots[0], 0, "live weak slot must keep its word");
        let cleared = unsafe { std::ptr::read_volatile(&weak_dead) };
        assert_eq!(cleared, 0, "dead weak slot must be cleared");
        black_box(&fx.roots);
    }

    #[test]
    fn nogc_region_defers_collection() {
        let mut fx = Fixture::new();
        fx.heap.set_stress_every(1);
        let before = fx.heap.stats().collections;
        {
            let _scope = fx.heap.nogc();
            for i in 0..50 {
                let p = fx.heap.alloc(fx.leaf, 16);
                black_box((p, i));
            }
            assert_eq!(fx.heap.stats().collections, before);
            assert!(fx.heap.stats().deferred_collections > 0);
        }
        let _p = fx.heap.alloc(fx.leaf, 16);
        assert!(fx.heap.stats().collections > before, "deferred GC must run");
    }

    #[test]
    fn finalizers_run_for_dead_cells() {
        FIN_COUNT.with(|c| c.set(0));
        let mut fx = Fixture::new();
        let _a = fx.heap.alloc(fx.leaf, 16);
        let _b = fx.heap.alloc(fx.leaf, 16);
        fx.heap.collect();
        assert_eq!(FIN_COUNT.with(|c| c.get()), 2);
    }

    #[test]
    fn large_objects_allocate_and_collect() {
        let mut fx = Fixture::new();
        let big = fx.heap.alloc(fx.leaf, 8192);
        unsafe { std::ptr::write(big as *mut u64, 42) };
        fx.root(0, big as *const u8);
        fx.heap.collect();
        assert_eq!(unsafe { std::ptr::read(big as *const u64) }, 42);
        fx.roots[0] = 0;
        fx.heap.collect();
        assert_eq!(fx.heap.stats().live_cells, 0);
    }

    #[test]
    fn sweep_hook_sees_liveness_for_side_tables() {
        use std::cell::RefCell;
        use std::collections::HashMap;
        use std::rc::Rc;
        let mut fx = Fixture::new();
        let live = fx.alloc_node(0, 1);
        let dead = fx.alloc_node(0, 2);
        let table: Rc<RefCell<HashMap<usize, u64>>> = Rc::new(RefCell::new(HashMap::new()));
        table.borrow_mut().insert(live as usize, 10);
        table.borrow_mut().insert(dead as usize, 20);
        let t = table.clone();
        fx.heap.register_sweep_hook(
            "test-table",
            Box::new(move |is_live| {
                t.borrow_mut().retain(|addr, _| is_live(*addr as *const u8));
            }),
        );
        fx.root(0, live as *const u8);
        fx.heap.collect();
        let t = table.borrow();
        assert_eq!(t.len(), 1);
        assert_eq!(t[&(live as usize)], 10);
    }

    #[test]
    fn stress_graph_survives_collect_every_alloc() {
        let mut fx = Fixture::new();
        fx.heap.set_stress_every(1);
        // Build a 200-node chain under stress, rooting the head and
        // keeping every intermediate reachable through the chain.
        let mut head: *mut u8 = fx.alloc_node(0, 0);
        fx.root(0, head as *const u8);
        for i in 1..200u64 {
            let n = fx.alloc_node(TAG | (head as u64 & PAYLOAD_MASK), i);
            head = n;
            fx.root(0, head as *const u8);
            let _garbage = fx.heap.alloc(fx.leaf, 16);
        }
        fx.heap.collect();
        assert_eq!(fx.heap.stats().live_cells, 200);
        let mut p = head as *const u8;
        let mut expect = 199u64;
        loop {
            assert_eq!(fx.val(p), expect);
            let next = unsafe { std::ptr::read(p as *const u64) };
            if next == 0 {
                break;
            }
            expect -= 1;
            p = (next & PAYLOAD_MASK) as *const u8;
        }
        assert_eq!(expect, 0);
    }

    #[inline(never)]
    fn deep_collect(heap: &mut Heap, held: *const u8) {
        // Spill the held reference as the safepoint protocol requires,
        // then collect. The cell is reachable *only* from this frame.
        let mut spill = [0u64; 4];
        unsafe { std::ptr::write_volatile(&mut spill[0], held as u64) };
        black_box(&spill);
        heap.collect();
        black_box(&spill);
    }

    #[test]
    fn stack_resident_reference_roots_cell() {
        let mut fx = Fixture::new();
        let a = fx.alloc_node(0, 5);
        let top_marker = 0u64;
        fx.heap
            .set_stack_top(unsafe { (&top_marker as *const u64).add(256) });
        deep_collect(&mut fx.heap, a as *const u8);
        black_box(&top_marker);
        assert_eq!(fx.heap.stats().live_cells, 1);
        assert_eq!(fx.val(a as *const u8), 5);
        // Death proof needs a stack that never held the pointer:
        // conservative scanning legitimately retains via stale words
        // and registers on this thread. Collect on a fresh thread.
        // SAFETY (test only): the heap is used from one thread at a
        // time; the spawned thread is joined before reuse.
        let heap_addr = &mut fx.heap as *mut Heap as usize;
        let live = std::thread::spawn(move || {
            let heap = unsafe { &mut *(heap_addr as *mut Heap) };
            let marker = 0u64;
            heap.set_stack_top(unsafe { (&marker as *const u64).add(256) });
            heap.collect();
            black_box(&marker);
            heap.stats().live_cells
        })
        .join()
        .expect("collector thread");
        assert_eq!(live, 0);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "heap verifier")]
    fn verifier_rejects_dangling_edge_in_debug() {
        let mut fx = Fixture::new();
        let a = fx.alloc_node(0x0000_0000_00de_ad00, 1);
        fx.root(0, a as *const u8);
        fx.heap.collect();
    }
}
