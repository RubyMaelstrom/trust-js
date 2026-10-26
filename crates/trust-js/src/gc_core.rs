//! P1-S1 heap core: pointer-based mark-sweep heap for the GC-managed
//! object model (docs/v8-chase-p1-plan-2026-10-15.txt, Decisions 0–4).
//! The engine allocates its objects, environment records, strings and
//! payload cells here.
//!
//! Design summary:
//! - Cells live in 64 KiB pages segregated by (size class, descriptor),
//!   memory mapped from the OS (see `page_source`); page metadata is a side
//!   table found by page number through a directly indexed page table, so
//!   interior pointers resolve by masking to the page base. Large cells
//!   (> 4 KiB) get dedicated allocations from the allocator, found by address
//!   range.
//! - References are direct pointers (never handle-table indices); a
//!   tagged execution word masks to the same pointer (payload mask).
//! - Tracing is precise inside the heap (per-descriptor trace functions)
//!   and for the roots the embedder reports through its [`MarkClient`],
//!   owner-conditional for its side tables, and conservative only for the
//!   native stack and registers, per the Decision 0 scanning contract (see
//!   `native`).
//! - Ephemerons (and the other conditional edges of the embedder's side
//!   tables) are reported afresh by the [`MarkClient`] at every collection;
//!   cells hold no weak slots of their own.
//! - Collection honors I9: it never runs inside a no-GC region or while a
//!   GcCell borrow is outstanding (deferred via a pending flag; the borrow
//!   case asserts in debug builds), and cells are allocated zeroed.
//! - Dropping the heap finalizes the remaining cells and frees all heap
//!   memory (I5 teardown).
//! - Marks persist from one collection to the next (S6 step 10b): between
//!   collections a marked cell is old and an allocated unmarked one young. The
//!   pages of kinds whose cells gain references after initialization end in a
//!   card table, where the write barrier records each written cell
//!   ([`card_mark`]).
//! - Collections are minor or major (S6 step 10c, [`CollectionKind`]). A major
//!   collection clears every mark and card, marks everything reachable and
//!   sweeps every page. A minor one keeps the marks: it traces from the roots
//!   and from the old cells whose cards are set, and no other old cell, and
//!   sweeps only the pages that received cells since the last collection, and
//!   the large cells. A cell is old once any collection has marked it, so an
//!   old cell that died waits for a major collection.
//! - Debug builds poison freed cells. Heap verification ([`Heap::set_heap_verification`];
//!   debug builds, opt-in but for the engine's unit tests) checks the write barrier and the
//!   young pages' bookkeeping as each collection starts and verifies the marked graph.

use crate::fasthash::FastMap;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::rc::Rc;

pub(crate) const PAGE_SIZE: usize = 64 * 1024;
pub(crate) const PAGE_SHIFT: u32 = PAGE_SIZE.trailing_zeros();
const PAYLOAD_MASK: u64 = 0x0000_ffff_ffff_ffff;
/// No cell lies in the first page of the address space.
const MIN_CELL_ADDRESS: usize = 4096;
/// Cells larger than the last class are large objects.
const SIZE_CLASSES: [usize; 16] = [
    16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
];
/// The largest small cell, which lives in a page: a larger one is a large object.
pub(crate) const LARGEST_SMALL_CELL: usize = SIZE_CLASSES[SIZE_CLASSES.len() - 1];
const DEFAULT_MIN_TRIGGER: usize = 1024 * 1024;
const DEFAULT_GROWTH_PERCENT: usize = 100;
/// Under a stress interval with a nursery, every this many collections the stress makes one
/// major (see [`Heap::set_stress_major_every`]).
const DEFAULT_STRESS_MAJOR_EVERY: usize = 8;
/// The fewest released pages the spare pool keeps with a nursery (see `Heap::spare_page_cap`).
const MIN_SPARE_PAGES_WITH_NURSERY: usize = 16;
/// The fewest released pages the spare pool keeps without a nursery.
const MIN_SPARE_PAGES: usize = 64;
#[cfg(debug_assertions)]
const POISON: u64 = 0xdddd_dddd_dddd_dddd;

/// S6 step 10b: a page of a kind whose cells gain references after their initialization
/// ([`LayoutDesc::carded`]) ends in a card table, a byte per card of `1 << CARD_SHIFT` bytes of
/// the page. The write barrier sets the card of a written cell's start ([`card_mark`]); a
/// collection checks the cards (heap verification) and clears them when it starts.
pub(crate) const CARD_SHIFT: u32 = 7;
/// The bytes of a carded page's card table, its last: one per card of the page.
pub(crate) const CARD_TABLE_BYTES: usize = PAGE_SIZE >> CARD_SHIFT;
/// Where a carded page's card table starts in the page (65,024): no cell reaches it.
pub(crate) const CARD_TABLE_OFFSET: usize = PAGE_SIZE - CARD_TABLE_BYTES;
const _: () = assert!(CARD_TABLE_BYTES == 512 && CARD_TABLE_OFFSET == 65_024);
/// The page's card summary: the last byte of a carded page, which is the card of the table's
/// own last 128 bytes, so no cell's start maps to it. The barrier sets it with every card, and
/// a minor collection skips the table of a page whose summary is clear, reading one byte
/// instead of the table (see `PageMeta::take_card_cells`).
pub(crate) const CARD_SUMMARY: usize = PAGE_SIZE - 1;
const _: () = assert!((CARD_SUMMARY - CARD_TABLE_OFFSET) == CARD_TABLE_BYTES - 1);

/// Whether `word` (a raw pointer or a packed word, its tag masked) can name a cell at all.
/// Zero and small integers, the most common words in frames and records, cannot: conservative
/// scans drop them before anything else.
#[inline(always)]
pub(crate) fn may_name_cell(word: u64) -> bool {
    (word & PAYLOAD_MASK) as usize >= MIN_CELL_ADDRESS
}

/// The write barrier: record that the cell starting at `cell` was written, by setting its card
/// (a nonzero byte at its page's base + `CARD_TABLE_OFFSET` + its start's offset in the page >>
/// `CARD_SHIFT`) and its page's summary ([`CARD_SUMMARY`]), from the address alone: no lookup,
/// no load, no branch. A card stays set until the next collection starts. Generated code
/// stores the same bytes (ARM64's `emit_card_mark`, x64's `card_mark`); an Object's address
/// (its cell's start + 8) lies in its cell's card.
///
/// # Safety
/// `cell` is the start of a live small cell of a carded kind ([`LayoutDesc::carded`]), whose
/// page therefore ends in its card table.
#[inline]
pub(crate) unsafe fn card_mark(cell: *const u8) {
    let addr = cell as usize;
    let card = (addr >> CARD_SHIFT) & (CARD_TABLE_BYTES - 1);
    let base = addr & !(PAGE_SIZE - 1);
    // SAFETY: the caller's contract: the cell's page is live and carded, and no cell overlaps
    // its card table, of which the summary is the last byte.
    unsafe {
        ((base + CARD_TABLE_OFFSET + card) as *mut u8).write(1);
        ((base + CARD_SUMMARY) as *mut u8).write(1);
    }
}

/// The size class of each 16-byte multiple up to the largest small cell: `CLASS_OF[(size +
/// 15) / 16]` is the first class whose cells hold `size` bytes.
const CLASS_OF: [u8; LARGEST_SMALL_CELL / 16 + 1] = {
    let mut table = [0u8; LARGEST_SMALL_CELL / 16 + 1];
    let mut class = 0;
    let mut units = 0;
    while units < table.len() {
        while SIZE_CLASSES[class] < units * 16 {
            class += 1;
        }
        table[units] = class as u8;
        units += 1;
    }
    table
};

#[inline]
fn size_class(size: usize) -> Option<usize> {
    CLASS_OF.get(size.div_ceil(16)).map(|&class| class as usize)
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

fn large_layout(size: usize) -> Layout {
    Layout::from_size_align(size, 16).expect("large GC cell layout")
}

/// The bits of a page number each level of the [`PageTable`] decodes: a 48-bit payload holds a
/// 32-bit page number above `PAGE_SHIFT`.
const LEAF_BITS: u32 = 16;
const LEAF_LEN: usize = 1 << LEAF_BITS;
const _: () = assert!(PAYLOAD_MASK.count_ones() == PAGE_SHIFT + 2 * LEAF_BITS);

/// The slot in `Heap::pages` of the page at each page number (an address's 48-bit payload >>
/// `PAGE_SHIFT`), directly indexed: the page number's high 16 bits select a leaf of 65,536
/// entries (4 GiB of addresses), its low 16 bits the entry, which holds the slot plus one, or 0.
/// Both levels are zeroed memory mapped from the OS ([`TableLevel`]), so only the OS pages
/// holding entries in use are ever written: with 4 KiB pages, of the top level (512 KiB) those
/// recording leaves, and of a leaf (256 KiB, mapped when a page first falls in its range) one
/// per 64 MiB of addresses holding pages. Leaves are unmapped by [`PageTable::clear`] or with
/// the table.
struct PageTable {
    /// Each leaf's entries, by the page number's high 16 bits (null: no leaf).
    top: TableLevel<*mut u32>,
    /// The leaves, which `top` points into, with their indices there.
    leaves: Vec<(u16, TableLevel<u32>)>,
    /// Entries in use.
    len: usize,
}

impl PageTable {
    fn new() -> Self {
        Self {
            top: TableLevel::new(),
            leaves: Vec::new(),
            len: 0,
        }
    }

    /// The entry of the page at `page`, a page number, if its leaf is mapped.
    #[inline]
    fn entry(&self, page: usize) -> Option<*mut u32> {
        let page = u32::try_from(page).ok()?;
        let leaf = self.top[(page >> LEAF_BITS) as usize];
        if leaf.is_null() {
            return None;
        }
        // SAFETY: a non-null pointer in `top` addresses the LEAF_LEN entries of a leaf in
        // `leaves`.
        Some(unsafe { leaf.add(page as usize & (LEAF_LEN - 1)) })
    }

    /// The slot of the page at `page`, a page number, if one is mapped there.
    #[inline]
    fn get(&self, page: usize) -> Option<u32> {
        // SAFETY: an entry of a mapped leaf, which no reference covers.
        unsafe { self.entry(page)?.read() }.checked_sub(1)
    }

    /// Map the page at `page` to `slot`. Returns the slot it replaces, if any.
    fn insert(&mut self, page: usize, slot: u32) -> Option<u32> {
        let index = u32::try_from(page).expect("a GC page above the 48-bit payload") >> LEAF_BITS;
        if self.top[index as usize].is_null() {
            let leaf = TableLevel::new();
            self.top[index as usize] = leaf.as_ptr();
            self.leaves.push((index as u16, leaf));
        }
        let entry = self.entry(page).expect("a mapped leaf");
        let new = slot.checked_add(1).expect("a page slot below u32::MAX");
        // SAFETY: as in `get`.
        let old = unsafe { entry.replace(new) };
        if old == 0 {
            self.len += 1;
        }
        old.checked_sub(1)
    }

    /// Unmap the page at `page`. Returns its slot, if one was mapped.
    fn remove(&mut self, page: usize) -> Option<u32> {
        let entry = self.entry(page)?;
        // SAFETY: as in `get`.
        let slot = unsafe { entry.replace(0) }.checked_sub(1)?;
        self.len -= 1;
        Some(slot)
    }

    /// Unmap every page, and the leaves.
    fn clear(&mut self) {
        for &(index, _) in &self.leaves {
            self.top[index as usize] = std::ptr::null_mut();
        }
        self.leaves.clear();
        self.len = 0;
    }

    /// The pages mapped.
    fn len(&self) -> usize {
        self.len
    }
}

/// One level of a [`PageTable`]: LEAF_LEN `T`s, zeroed, in memory mapped from the OS for it and
/// unmapped with it, whose pages cost no memory until written. (A zeroed allocation does not
/// promise that: glibc serves a 512 KiB `calloc` from its heap, writing every page, once its
/// dynamic mmap threshold has risen past that size, as freeing a larger mapped block makes it.)
struct TableLevel<T: ZeroValid> {
    entries: NonNull<[T; LEAF_LEN]>,
}

/// A table entry type, valid as all-zero bytes, without drop glue.
///
/// # Safety
/// All-zero bytes must be a valid `Self`.
unsafe trait ZeroValid: Copy {}

// SAFETY: all-zero bytes are the u32 0.
unsafe impl ZeroValid for u32 {}
// SAFETY: all-zero bytes are the null pointer.
unsafe impl ZeroValid for *mut u32 {}

impl<T: ZeroValid> TableLevel<T> {
    const BYTES: usize = std::mem::size_of::<[T; LEAF_LEN]>();

    fn new() -> Self {
        let entries = NonNull::new(os_pages::map(Self::BYTES).cast())
            .expect("GC page table allocation failed");
        Self { entries }
    }

    /// The first entry: the level's entries by address.
    fn as_ptr(&self) -> *mut T {
        self.entries.as_ptr().cast()
    }
}

impl<T: ZeroValid> std::ops::Deref for TableLevel<T> {
    type Target = [T; LEAF_LEN];

    fn deref(&self) -> &[T; LEAF_LEN] {
        // SAFETY: mapped zeroed, which is a valid [T; LEAF_LEN], and owned by the level.
        unsafe { self.entries.as_ref() }
    }
}

impl<T: ZeroValid> std::ops::DerefMut for TableLevel<T> {
    fn deref_mut(&mut self) -> &mut [T; LEAF_LEN] {
        // SAFETY: as in `deref`.
        unsafe { self.entries.as_mut() }
    }
}

impl<T: ZeroValid> Drop for TableLevel<T> {
    fn drop(&mut self) {
        // SAFETY: mapped in `new` with this size; the entries need no drop.
        unsafe { os_pages::unmap(self.as_ptr().cast(), Self::BYTES) };
    }
}

/// Zeroed memory mapped from the OS (see [`TableLevel`] and `page_source`).
#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod os_pages {
    unsafe extern "C" {
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
        fn munmap(addr: *mut u8, len: usize) -> i32;
        #[cfg(target_os = "linux")]
        fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    }
    const PROT_READ_WRITE: i32 = 0x1 | 0x2;
    #[cfg(target_os = "linux")]
    const MAP_PRIVATE_ANONYMOUS: i32 = 0x02 | 0x20;
    #[cfg(target_os = "macos")]
    const MAP_PRIVATE_ANONYMOUS: i32 = 0x02 | 0x1000;
    #[cfg(target_os = "linux")]
    const MADV_DONTNEED: i32 = 4;
    #[cfg(target_os = "linux")]
    const MADV_HUGEPAGE: i32 = 14;

    /// `len` zeroed bytes at a page boundary; null when the system refuses.
    pub(super) fn map(len: usize) -> *mut u8 {
        // SAFETY: a new private anonymous mapping, where the system chooses.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ_WRITE,
                MAP_PRIVATE_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr as isize == -1 {
            std::ptr::null_mut()
        } else {
            ptr
        }
    }

    /// `len` zeroed bytes at an `align` boundary, `align` a power of two and both multiples of the
    /// system's page size: `len + align` bytes mapped, and the ends outside the range unmapped.
    /// Null when the system refuses.
    pub(super) fn map_aligned(len: usize, align: usize) -> *mut u8 {
        let raw = map(len + align);
        if raw.is_null() {
            return raw;
        }
        let head = (raw as usize).next_multiple_of(align) - raw as usize;
        // SAFETY: both ends lie in the fresh mapping, at page boundaries, outside the range kept
        // (the tail is `align - head` bytes, never empty).
        unsafe {
            if head > 0 {
                munmap(raw, head);
            }
            munmap(raw.add(head + len), align - head);
            raw.add(head)
        }
    }

    /// # Safety
    /// `ptr` and `len` are a `map` or `map_aligned` call's, and nothing uses the memory again.
    pub(super) unsafe fn unmap(ptr: *mut u8, len: usize) {
        // SAFETY: the caller's contract.
        unsafe { munmap(ptr, len) };
    }

    /// Ask the system to back `ptr..ptr + len`, mapped here, with transparent huge pages. A system
    /// without them refuses, which changes nothing.
    #[cfg(target_os = "linux")]
    pub(super) fn advise_huge_pages(ptr: *mut u8, len: usize) {
        // SAFETY: advice only: no contents change.
        unsafe { madvise(ptr, len, MADV_HUGEPAGE) };
    }

    /// Free the memory behind `ptr..ptr + len`, mapped here, at page boundaries, and leave the
    /// range mapped: it reads as zero when next touched. False when the system refuses (a locked
    /// mapping), which leaves the contents as they were.
    ///
    /// # Safety
    /// Nothing holds a reference into the range or needs its contents.
    #[cfg(target_os = "linux")]
    pub(super) unsafe fn discard(ptr: *mut u8, len: usize) -> bool {
        // SAFETY: the caller's contract.
        unsafe { madvise(ptr, len, MADV_DONTNEED) == 0 }
    }
}

/// Zeroed memory mapped from the OS (see [`TableLevel`] and `page_source`).
#[cfg(all(
    target_os = "windows",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod os_pages {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualAlloc(addr: *mut u8, len: usize, kind: u32, protect: u32) -> *mut u8;
        fn VirtualFree(addr: *mut u8, len: usize, kind: u32) -> i32;
    }
    const MEM_COMMIT_RESERVE: u32 = 0x1000 | 0x2000;
    const MEM_RELEASE: u32 = 0x8000;
    const PAGE_READWRITE: u32 = 0x04;

    /// `len` zeroed bytes at a page boundary; null when the system refuses.
    pub(super) fn map(len: usize) -> *mut u8 {
        // SAFETY: a new committed region, where the system chooses; its pages read as zero.
        unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                len,
                MEM_COMMIT_RESERVE,
                PAGE_READWRITE,
            )
        }
    }

    /// # Safety
    /// `ptr` is a `map` call's, and nothing uses the memory again.
    pub(super) unsafe fn unmap(ptr: *mut u8, _len: usize) {
        // SAFETY: the caller's contract.
        unsafe { VirtualFree(ptr, 0, MEM_RELEASE) };
    }
}

/// Elsewhere, a zeroed allocation, which may write its pages.
#[cfg(not(all(
    any(target_os = "linux", target_os = "macos", target_os = "windows"),
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
mod os_pages {
    use std::alloc::{alloc_zeroed, dealloc, Layout};

    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len, 4096).expect("GC page table layout")
    }

    /// `len` zeroed bytes at a page boundary; null when the allocator refuses.
    pub(super) fn map(len: usize) -> *mut u8 {
        // SAFETY: a nonzero size.
        unsafe { alloc_zeroed(layout(len)) }
    }

    /// # Safety
    /// `ptr` and `len` are a `map` call's, and nothing uses the memory again.
    pub(super) unsafe fn unmap(ptr: *mut u8, len: usize) {
        // SAFETY: the caller's contract.
        unsafe { dealloc(ptr, layout(len)) };
    }
}

/// Where the heap's pages come from, and where they go back: memory mapped from the OS, outside
/// the allocator. (glibc served each page as a 64 KiB-aligned block of its arena: the alignment
/// slack beside it went to its free lists, where other allocations spread into it, and freed pages
/// stayed in the arena.) `PageSource::take` gives a zeroed page at a `PAGE_SIZE` boundary. The
/// heap keeps the pages it releases in its spare pool and gives back those the pool trims; its
/// release returns every page.
///
/// On Linux the source maps chunks of 2 MiB at 2 MiB boundaries, advised for transparent huge
/// pages once the heap has a few of them, and carves its pages from them. A page given back is
/// discarded (its memory freed, the range left mapped to read as zero) and taken again once the
/// current chunk is carved, before a new chunk is mapped. Chunks are unmapped only with the heap.
#[cfg(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod page_source {
    use super::{os_pages, PAGE_SIZE};

    /// The size and alignment of a chunk: one transparent huge page, where base pages are 4 KiB.
    pub(super) const CHUNK_SIZE: usize = 2 << 20;
    const PAGES_PER_CHUNK: usize = CHUNK_SIZE / PAGE_SIZE;
    /// The chunks a heap maps before it asks for huge pages. A huge page makes its whole chunk
    /// resident, so a small heap's partly carved chunks keep base pages, which only the pages
    /// taken make resident; a heap that reaches this many chunks advises them all.
    pub(super) const HUGE_PAGE_MIN_CHUNKS: usize = 4;

    pub(super) struct PageSource {
        /// The chunks mapped, by address.
        chunks: Vec<usize>,
        /// The pages of the last chunk not taken yet: `next..end`.
        next: usize,
        end: usize,
        /// The pages given back, discarded, to be taken again before a new chunk is mapped.
        free: Vec<usize>,
    }

    impl PageSource {
        pub(super) fn new() -> Self {
            Self {
                chunks: Vec::new(),
                next: 0,
                end: 0,
                free: Vec::new(),
            }
        }

        /// A zeroed page at a `PAGE_SIZE` boundary: the current chunk's next page, else the page
        /// given back last, else a new chunk's first. (A chunk a huge page backs is resident
        /// whole, so its pages not taken yet cost no memory, where a page given back would.)
        pub(super) fn take(&mut self) -> *mut u8 {
            if self.next == self.end {
                if let Some(base) = self.free.pop() {
                    return base as *mut u8;
                }
                self.map_chunk();
            }
            let base = self.next;
            self.next += PAGE_SIZE;
            base as *mut u8
        }

        #[cold]
        fn map_chunk(&mut self) {
            let base = os_pages::map_aligned(CHUNK_SIZE, CHUNK_SIZE);
            assert!(!base.is_null(), "GC page allocation failed");
            self.chunks.push(base as usize);
            match self.chunks.len() {
                n if n < HUGE_PAGE_MIN_CHUNKS => {}
                // The earlier chunks too: the heap is now large enough to use them whole.
                HUGE_PAGE_MIN_CHUNKS => {
                    for &chunk in &self.chunks {
                        os_pages::advise_huge_pages(chunk as *mut u8, CHUNK_SIZE);
                    }
                }
                _ => os_pages::advise_huge_pages(base, CHUNK_SIZE),
            }
            (self.next, self.end) = (base as usize, base as usize + CHUNK_SIZE);
        }

        /// Whether the chunks mapped so far are advised for huge pages.
        #[cfg(test)]
        pub(super) fn huge_pages_advised(&self) -> bool {
            self.chunks.len() >= HUGE_PAGE_MIN_CHUNKS
        }

        /// Return the memory of the page at `base` to the OS. The page stays in its chunk, to be
        /// taken again.
        ///
        /// # Safety
        /// `base` is a page `take` gave and not given back since, which nothing uses any more.
        pub(super) unsafe fn give_back(&mut self, base: *mut u8) {
            // SAFETY: the caller's contract.
            if !unsafe { os_pages::discard(base, PAGE_SIZE) } {
                // The system keeps the memory (a locked mapping), so the page is zeroed here.
                // SAFETY: as above.
                unsafe { std::ptr::write_bytes(base, 0, PAGE_SIZE) };
            }
            self.free.push(base as usize);
        }

        /// Return every page to the OS, as the heap is released: `pages` are those `take` gave
        /// and `give_back` did not take back (debug builds count them). A page taken afterwards
        /// comes from a new chunk.
        ///
        /// # Safety
        /// Nothing uses a page the source gave any more.
        pub(super) unsafe fn release(&mut self, pages: impl IntoIterator<Item = usize>) {
            if cfg!(debug_assertions) {
                let untaken = (self.end - self.next) / PAGE_SIZE;
                assert_eq!(
                    pages.into_iter().count() + self.free.len() + untaken,
                    self.chunks.len() * PAGES_PER_CHUNK,
                    "GC page source: pages unaccounted for at the heap's release"
                );
            }
            self.unmap_chunks();
        }

        fn unmap_chunks(&mut self) {
            for base in self.chunks.drain(..) {
                // SAFETY: mapped by `map_chunk` with this size; no page of it is used any more
                // (`release`'s contract, or the heap is gone).
                unsafe { os_pages::unmap(base as *mut u8, CHUNK_SIZE) };
            }
            self.free.clear();
            (self.next, self.end) = (0, 0);
        }

        /// The chunks mapped.
        #[cfg(test)]
        pub(super) fn chunks(&self) -> usize {
            self.chunks.len()
        }

        /// The pages given back and not taken again.
        #[cfg(test)]
        pub(super) fn free_pages(&self) -> usize {
            self.free.len()
        }

        /// The bytes mapped for pages.
        #[cfg(test)]
        pub(super) fn mapped_bytes(&self) -> usize {
            self.chunks.len() * CHUNK_SIZE
        }
    }

    impl Drop for PageSource {
        /// The heap's release has unmapped the chunks, unless a finalizer panicked during it.
        fn drop(&mut self) {
            self.unmap_chunks();
        }
    }
}

/// Where the heap's pages come from (see the Linux `page_source`), elsewhere a page at a time,
/// returned when given back: on macOS a mapping (twice a page mapped, the ends unmapped); on
/// Windows and other targets the allocator's zeroed 64 KiB-aligned block. (Windows mapped each
/// page with VirtualAlloc from S6 step 10a: a released heap's addresses then went to the next
/// heap in the same order, and under Wine stale stack words kept dead Realms alive in tests.)
#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
mod page_source {
    use super::PAGE_SIZE;

    pub(super) struct PageSource {
        /// The bytes mapped for pages.
        #[cfg(test)]
        mapped: usize,
    }

    impl PageSource {
        pub(super) fn new() -> Self {
            Self {
                #[cfg(test)]
                mapped: 0,
            }
        }

        /// A zeroed page at a `PAGE_SIZE` boundary.
        pub(super) fn take(&mut self) -> *mut u8 {
            let base = map_page();
            assert!(!base.is_null(), "GC page allocation failed");
            #[cfg(test)]
            {
                self.mapped += PAGE_SIZE;
            }
            base
        }

        /// Return the page at `base` to the OS.
        ///
        /// # Safety
        /// `base` is a page `take` gave and not given back since, which nothing uses any more.
        pub(super) unsafe fn give_back(&mut self, base: *mut u8) {
            // SAFETY: the caller's contract.
            unsafe { unmap_page(base) };
            #[cfg(test)]
            {
                self.mapped -= PAGE_SIZE;
            }
        }

        /// Return every page to the OS, as the heap is released: `pages` are those `take` gave
        /// and `give_back` did not take back.
        ///
        /// # Safety
        /// Nothing uses a page the source gave any more.
        pub(super) unsafe fn release(&mut self, pages: impl IntoIterator<Item = usize>) {
            for base in pages {
                // SAFETY: the caller's contract.
                unsafe { self.give_back(base as *mut u8) };
            }
        }

        /// The bytes mapped for pages.
        #[cfg(test)]
        pub(super) fn mapped_bytes(&self) -> usize {
            self.mapped
        }
    }

    #[cfg(all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    fn map_page() -> *mut u8 {
        super::os_pages::map_aligned(PAGE_SIZE, PAGE_SIZE)
    }

    /// # Safety
    /// `base` is a `map_page` page, which nothing uses any more.
    #[cfg(all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    unsafe fn unmap_page(base: *mut u8) {
        // SAFETY: the caller's contract.
        unsafe { super::os_pages::unmap(base, PAGE_SIZE) };
    }

    #[cfg(not(all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )))]
    fn layout() -> std::alloc::Layout {
        std::alloc::Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).expect("GC page layout")
    }

    #[cfg(not(all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )))]
    fn map_page() -> *mut u8 {
        // SAFETY: a nonzero size.
        unsafe { std::alloc::alloc_zeroed(layout()) }
    }

    /// # Safety
    /// `base` is a `map_page` page, which nothing uses any more.
    #[cfg(not(all(
        target_os = "macos",
        any(target_arch = "aarch64", target_arch = "x86_64")
    )))]
    unsafe fn unmap_page(base: *mut u8) {
        // SAFETY: the caller's contract.
        unsafe { std::alloc::dealloc(base, layout()) };
    }
}

/// Trace callback: report every strong heap edge of the cell at `cell` by
/// calling `visit(edge)` with a cell pointer or tagged word. Must tolerate
/// every state the cell can be in at an allocation point (I9): allocation
/// returns zeroed memory, and a cell held only in a native local is traced
/// (conservative roots) before it is published anywhere, so initialization
/// must finish before the next allocation or write fields in an order in
/// which each intermediate state traces safely.
pub(crate) type TraceFn = fn(*const u8, &mut dyn FnMut(*const u8));
/// Finalize callback for a dead cell, run during sweep or heap teardown.
/// Must not allocate from this heap, resurrect the cell, or read other
/// cells (they may already be finalized); same state tolerance as
/// [`TraceFn`].
pub(crate) type FinalizeFn = fn(*mut u8);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct DescId(pub(crate) u32);

/// Actual marker work; ordinary trace callbacks/verifiers never consume remembered state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TracePurpose {
    Full,
    Remembered,
}

pub(crate) type MarkTraceFn = fn(*const u8, TracePurpose, &mut MarkSink<'_>);

/// What the marker hands a [`LayoutDesc::mark_trace`]: each edge reported to it is marked at
/// once ([`MarkSink::mark`]), through the path every word the marker resolves takes, with no
/// buffer and no dynamic call between the trace and the mark.
pub(crate) struct MarkSink<'a> {
    heap: &'a mut Heap,
    m: &'a mut Marker,
}

impl MarkSink<'_> {
    /// Mark the cell `word` names, if it names one of this heap's (see [`Heap::mark_word`]).
    #[inline(always)]
    pub(crate) fn mark(&mut self, word: u64) {
        self.heap.mark_word(word, self.m);
    }
}

#[derive(Clone, Copy)]
struct TraceWork {
    cell: usize,
    desc: DescId,
    purpose: TracePurpose,
}

#[cfg(target_pointer_width = "64")]
const _: () = {
    assert!(std::mem::size_of::<TraceWork>() == 16);
    assert!(std::mem::size_of::<TraceWork>() == std::mem::size_of::<(usize, DescId)>());
};

pub(crate) struct LayoutDesc {
    pub(crate) name: &'static str,
    pub(crate) trace: TraceFn,
    /// Used only by actual marker drain, after first-mark bookkeeping. The ordinary trace
    /// remains complete and nonconsuming for verification and root/diagnostic visitors.
    pub(crate) mark_trace: Option<MarkTraceFn>,
    pub(crate) finalize: Option<FinalizeFn>,
    /// The kind's cells gain references after their initialization (the engine's objects,
    /// environments and deferred prototypes), through writes the barrier records
    /// ([`card_mark`]): its pages end in a card table, so a page holds `CARD_TABLE_OFFSET /
    /// cell size` cells. Cells of other kinds never get a reference after initialization (a
    /// closure's environment, a string view's owner), and their pages are cells throughout.
    pub(crate) carded: bool,
}

/// What a collection traces and sweeps (S6 step 10c).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CollectionKind {
    /// Keeps the marks, so that every old cell (one an earlier collection marked) stays marked:
    /// traces from the roots, side slots, ephemerons and native stack, and from the old cells
    /// whose cards are set, and from no other old cell. Sweeps the pages that received cells
    /// since the last collection, and the large cells. An old cell that died stays allocated
    /// until a major collection.
    Minor = 0,
    /// Clears every mark and card, marks everything reachable and sweeps every page.
    Major = 1,
}

/// Receives the edges a [`MarkClient`] keeps off-heap, in its side tables.
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
    /// whose key or owner died is the embedder's job: it sweeps its tables
    /// between the phases (see [`Heap::start_marking`]).
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

/// A page's bitmap, held as one raw allocation: marking ([`MarkPage::marks`]) and allocation
/// ([`Cursor::bits`]) reach it through copies of its own pointer, which every other access
/// derives its short-lived reference from. (A `Box` would invalidate the copies at each access
/// through it under Rust's aliasing models.)
struct PageBits {
    ptr: NonNull<u64>,
    len: usize,
}

impl PageBits {
    fn zeroed(len: usize) -> Self {
        let bits = Box::into_raw(vec![0u64; len].into_boxed_slice());
        Self {
            ptr: NonNull::new(bits.cast::<u64>()).expect("a boxed slice's pointer"),
            len,
        }
    }

    /// The bitmap's own pointer, for a copy that reads it while it lives.
    fn as_ptr(&self) -> *const u64 {
        self.ptr.as_ptr()
    }

    /// The bitmap's own pointer, for a copy that writes it while it lives.
    fn as_mut_ptr(&mut self) -> *mut u64 {
        self.ptr.as_ptr()
    }
}

impl std::ops::Deref for PageBits {
    type Target = [u64];

    fn deref(&self) -> &[u64] {
        // SAFETY: `len` initialized words, allocated while `self` lives.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl std::ops::DerefMut for PageBits {
    fn deref_mut(&mut self) -> &mut [u64] {
        // SAFETY: as in `deref`; no copy of the pointer is used while the reference lives.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for PageBits {
    fn drop(&mut self) {
        let bits = std::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len);
        // SAFETY: the allocation `zeroed` made, of `len` words, freed once.
        drop(unsafe { Box::from_raw(bits) });
    }
}

/// A page's metadata. Marking tests a cell's mark through the page's [`MarkPage`]; a first
/// mark ([`Heap::mark_unmarked`]) reads the fields before `cells`, in the struct's first cache
/// line.
#[repr(C, align(64))]
struct PageMeta {
    base: usize,
    /// Set by the marking that found a cell reachable, and kept by the sweep: between
    /// collections a marked cell is old (a collection kept it), an allocated unmarked one young
    /// (allocated since). A major collection clears them before it marks; a minor one keeps
    /// them.
    marks: PageBits,
    /// Allocated (handed-out) cells. Conservative marking must ignore
    /// free cells: a stale stack word resolving into a free cell would
    /// otherwise "mark" free-list linkage and trace it as payload.
    allocated: PageBits,
    cell_size: usize,
    desc: DescId,
    /// `ceil(2^32 / cell_size)`, by which [`PageMeta::cell_index`] divides.
    reciprocal: u32,
    /// The page's kind holds no edges ([`Heap::set_leaf`]): marking one of its cells queues no
    /// trace.
    leaf: bool,
    /// Some cell of the page is a zombie (see `zombies`): only then does marking read them.
    has_zombies: bool,
    /// A conditional edge waits on some cell of the page in the marking in progress (see
    /// `waiters`): only then does a first mark read them.
    has_waiters: bool,
    /// The page ends in a card table ([`LayoutDesc::carded`]).
    carded: bool,
    cells: usize,
    class: usize,
    /// The page received a cell since the last collection (it was made, or a cell was taken
    /// from its reclaimed list or virgin tail), and is listed in `Heap::young_pages`: only
    /// such a page can hold a young cell, so only such pages are swept by a minor collection.
    /// Cleared by every sweep.
    young: bool,
    /// Cells a conditional edge is waiting on during this collection.
    waiters: Vec<u64>,
    /// Dead cells already finalized but still pinned by weak handles: like the allocation
    /// behind `std::rc::Weak`, the address is not reused while a weak handle may compare it.
    /// Never marked or traced; freed once no weak handle pins it.
    zombies: Vec<u64>,
    /// Cells a weak handle still names, recorded by exact start for the sweep of the collection
    /// being finished ([`Heap::finish_collection`]): the sweep keeps those it finds unmarked as
    /// zombies, and clears every bit.
    pins: Vec<u64>,
    /// The first never-handed-out cell. Only the prefix below this cursor can have allocation
    /// bits; the virgin tail's payload is never read or written before it is handed out.
    /// Monotonic until this page is emptied and its storage reused with fresh metadata.
    next_unused: usize,
    /// Where the page's [`Cursor`] resumes: the next word of `allocated` to load is
    /// `next_word`, and `free` the free cells of word `next_word - 1` while the cursor has not
    /// taken them. Allocation takes the lowest free cell, so it reuses reclaimed holes in
    /// address order and reaches the virgin tail last, and it never reads a free cell: the
    /// sweep frees by clearing bits, without writing to dead cells. Reset by every sweep.
    free: u64,
    next_word: u32,
    live: usize,
}

// A first mark's fields end within the struct's first cache line.
const _: () = assert!(std::mem::offset_of!(PageMeta, cells) <= 64);

/// What marking reads of a page to test an edge's mark ([`Heap::mark_small`]), by page slot
/// beside `Heap::pages`, 16 bytes a page: the heap's are few enough to stay in the nearest
/// cache, where its [`PageMeta`]s are not. A page's address is its edges' with the low
/// `PAGE_SHIFT` bits clear. A free slot's has no cells.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct MarkPage {
    /// The page's `PageMeta::marks`, whose bits stay where they are while it lives.
    marks: *const u64,
    /// As `PageMeta::reciprocal` and `PageMeta::cells`.
    reciprocal: u32,
    cells: u32,
}

impl MarkPage {
    const FREE: Self = Self {
        marks: std::ptr::null(),
        reciprocal: 0,
        cells: 0,
    };

    fn of(page: &PageMeta) -> Self {
        Self {
            marks: page.marks.as_ptr(),
            reciprocal: page.reciprocal,
            cells: u32::try_from(page.cells).expect("a page's cells"),
        }
    }

    /// The index of the cell the address `addr`, in this page, falls in, as
    /// [`PageMeta::cell_index`] finds it.
    #[inline(always)]
    fn cell_index(&self, addr: usize) -> usize {
        (((addr & (PAGE_SIZE - 1)) as u64 * u64::from(self.reciprocal)) >> 32) as usize
    }
}

impl PageMeta {
    fn cell_ptr(&self, index: usize) -> usize {
        self.base + index * self.cell_size
    }

    /// The index of the cell the address `addr`, in this page, falls in: `cells` or more for
    /// the page's tail past its last cell and its card table. `offset / cell_size`, by a
    /// multiplication. Exact: the reciprocal exceeds 2^32 / cell_size by less than one and the
    /// offset is below 2^16, so the product exceeds the quotient by less than 2^-16, while a
    /// non-integral quotient is at least 1 / cell_size >= 2^-12 short of the next integer.
    #[inline(always)]
    fn cell_index(&self, addr: usize) -> usize {
        debug_assert!(addr.wrapping_sub(self.base) < PAGE_SIZE);
        (((addr - self.base) as u64 * u64::from(self.reciprocal)) >> 32) as usize
    }

    /// Whether the card of the cell at `index` is set: always false on a page without a card
    /// table.
    fn card_set(&self, index: usize) -> bool {
        if !self.carded {
            return false;
        }
        let card = (index * self.cell_size) >> CARD_SHIFT;
        // SAFETY: a carded page ends in its card table, which no cell overlaps; the page is
        // live while its metadata is.
        unsafe { ((self.base + CARD_TABLE_OFFSET + card) as *const u8).read() != 0 }
    }

    /// Clear the page's cards, if it has a card table.
    fn clear_cards(&mut self) {
        if self.carded {
            // SAFETY: as in `card_set`.
            unsafe {
                std::ptr::write_bytes(
                    (self.base + CARD_TABLE_OFFSET) as *mut u8,
                    0,
                    CARD_TABLE_BYTES,
                )
            };
        }
    }

    /// For a minor collection as it starts: list in `cells` every marked (old) cell that starts
    /// in a set card of this page, and clear the cards. A card covers `1 << CARD_SHIFT` bytes,
    /// so one card can cover the starts of several small cells; each marked one is listed. An
    /// unmarked cell is young (or free, or a zombie) and is traced, if at all, as it is reached.
    ///
    /// The barrier sets the card of every cell written, young or old, so a page that received
    /// cells since the last collection has most of its cards set for young cells, which this
    /// collection does not need: such a page is walked by its old cells, listing those whose
    /// card is set. An old page has few set cards and only old cells: it is walked by its cards.
    fn take_card_cells(&mut self, cells: &mut Vec<TraceWork>) {
        // SAFETY: a carded page ends in its card table, the summary its last byte; the page is
        // live while its metadata is.
        if !self.carded || unsafe { ((self.base + CARD_SUMMARY) as *const u8).read() } == 0 {
            return;
        }
        let old_word = |word: usize| self.marks[word] & !self.zombies[word];
        let work = |index: usize| TraceWork {
            cell: self.cell_ptr(index),
            desc: self.desc,
            purpose: TracePurpose::Remembered,
        };
        if self.young {
            for word in 0..self.next_unused.div_ceil(64) {
                let mut old = old_word(word);
                while old != 0 {
                    let index = word * 64 + old.trailing_zeros() as usize;
                    old &= old - 1;
                    if self.card_set(index) {
                        cells.push(work(index));
                    }
                }
            }
        } else {
            const WORDS: usize = CARD_TABLE_BYTES / 8;
            const LOW7: u64 = 0x7f7f_7f7f_7f7f_7f7f;
            const HIGH: u64 = 0x8080_8080_8080_8080;
            let table = (self.base + CARD_TABLE_OFFSET) as *const u64;
            // The first cell that starts at or after byte `offset` of the page:
            // `ceil(offset / cell_size)`, by a multiplication. Exact: the reciprocal exceeds
            // 2^32 / cell_size by less than one and `offset + cell_size - 1` is below 2^17, so
            // the product exceeds the quotient by less than 2^-15, while a non-integral quotient
            // is at least 1 / cell_size >= 2^-12 short of the next integer.
            let size = self.cell_size as u64;
            let reciprocal = u64::from(self.reciprocal);
            let first_at =
                |offset: usize| (((offset as u64 + size - 1) * reciprocal) >> 32) as usize;
            for word in 0..WORDS {
                // SAFETY: a carded page ends in its card table, 8-aligned (the page is 64 KiB-
                // aligned and CARD_TABLE_OFFSET a multiple of 8), which no cell overlaps; the
                // page is live while its metadata is, and nothing else touches it during a
                // collection. Little-endian: bit `8 * k` begins card `8 * word + k`'s byte.
                let bytes = u64::from_le(unsafe { table.add(word).read() });
                // The high bit of each byte that is not zero.
                let mut set = (((bytes & LOW7) + LOW7) | bytes) & HIGH;
                while set != 0 {
                    let card = word * 8 + set.trailing_zeros() as usize / 8;
                    set &= set - 1;
                    let first = first_at(card << CARD_SHIFT);
                    let end = first_at((card + 1) << CARD_SHIFT).min(self.cells);
                    for index in first..end {
                        if old_word(index / 64) & (1 << (index % 64)) != 0 {
                            cells.push(work(index));
                        }
                    }
                }
            }
        }
        self.clear_cards();
    }

    fn has_room(&self) -> bool {
        // `live` counts every allocated cell, zombies included.
        self.live < self.cells
    }

    /// Restart the allocation cursor after `allocated` changed other than through it.
    fn reset_cursor(&mut self) {
        self.free = 0;
        self.next_word = 0;
    }

    /// Load the cursor with the next word of `allocated` that has a free cell; `None` once the
    /// page has none left.
    #[cold]
    fn next_free_word(&mut self) -> Option<()> {
        let words = self.cells.div_ceil(64);
        while (self.next_word as usize) < words {
            let word = self.next_word as usize;
            self.next_word += 1;
            let cells = (self.cells - word * 64).min(64);
            let valid = u64::MAX >> (64 - cells);
            self.free = !self.allocated[word] & valid;
            if self.free != 0 {
                return Some(());
            }
        }
        None
    }
}

/// The allocation cursor of one (descriptor, size class): the page it hands cells out from and
/// the free cells of that page's current bitmap word (Go's mcache with its allocCache,
/// JavaScriptCore's local allocators). The cursors of all classes sit in one dense array, so
/// allocation touches the cursor's line and the bitmap word, and reads the page's metadata only
/// when the word runs out ([`Heap::refill`]). The page's `live` and `next_unused` catch up when
/// the cursor leaves the word, as a collection starts and before a sweep ([`Heap::flush_cursors`]).
#[derive(Clone, Copy)]
struct Cursor {
    /// The free cells of bitmap word `word` not yet handed out, as set bits.
    free: u64,
    /// The page's allocation bitmap (its `allocated` buffer, which never moves).
    bits: *mut u64,
    /// The page's first cell.
    base: usize,
    cell_size: usize,
    word: u32,
    /// The page's slot, or [`NO_PAGE`].
    page: u32,
    /// Cells handed out since the page's `live` last caught up.
    taken: u32,
    /// One past the highest cell index handed out, for the page's `next_unused`.
    high: u32,
}

/// [`Cursor::page`] of a cursor with no page.
const NO_PAGE: u32 = u32::MAX;

impl Cursor {
    /// Hand out the lowest free cell of the loaded word, which has one (`free` is not zero).
    /// Cells leave a page's cursor in increasing order, so the last is the highest.
    #[inline(always)]
    fn take(&mut self) -> *mut u8 {
        debug_assert_ne!(self.free, 0, "a cursor takes from a loaded word");
        let bit = self.free.trailing_zeros();
        self.free &= self.free - 1;
        let index = self.word * 64 + bit;
        // SAFETY: the cursor's page is live, `bits` is its allocation bitmap and `word` one of
        // its words ([`Heap::refill`]); the cell was free.
        unsafe {
            let word = self.bits.add(self.word as usize);
            debug_assert_eq!(*word & (1 << bit), 0, "a cursor's free cell is allocated");
            *word |= 1 << bit;
        }
        self.taken += 1;
        debug_assert!(
            index >= self.high,
            "a cursor hands out cells in increasing order"
        );
        self.high = index + 1;
        (self.base + index as usize * self.cell_size) as *mut u8
    }

    const EMPTY: Cursor = Cursor {
        free: 0,
        bits: std::ptr::null_mut(),
        base: 0,
        cell_size: 0,
        word: 0,
        page: NO_PAGE,
        taken: 0,
        high: 0,
    };
}

/// Allocation state for one (descriptor, size class).
#[derive(Default)]
struct ClassPages {
    current: Option<u32>,
    /// Pages with reclaimed cells or virgin capacity after the last sweep, other than `current`.
    avail: Vec<u32>,
}

struct LargeObj {
    size: usize,
    desc: DescId,
    /// Allocated since the last collection, and listed in `Heap::young_larges`. Cleared by the
    /// sweep that meets it.
    young: bool,
    /// See `PageMeta::marks`. A large cell has no card: the write barrier never records one.
    marked: bool,
    waiters: bool,
    /// See `PageMeta::zombies`.
    zombie: bool,
    /// See `PageMeta::pins`.
    pinned: bool,
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

/// A resolved address is not necessarily allocated: conservative words can name free or
/// weakly pinned dead cells. Only the two live outcomes can seed a positive-word cache.
enum MarkOutcome {
    Rejected,
    AlreadyMarked,
    NewlyMarked { has_waiters: bool },
}

/// An edge whose value is reachable once its conditions are marked.
struct CondEdge {
    owner: Option<CellRef>,
    key: Option<CellRef>,
    value: EdgeValue,
    fired: bool,
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
    /// Heap verification, as each collection starts (beside the write barrier's check, before
    /// any mark is cleared): check what the embedder recorded since the last collection against
    /// that collection's marks, read with [`Heap::cell_age`], and panic on a violation. A minor
    /// collection's client may report only what changed since the last collection (S6 step
    /// 10d): an old owner whose entries were not written holds old values, which the last
    /// collection traced.
    fn check_records(&self, _heap: &Heap) {}
}

/// What the marks say of an allocated cell between collections, and as one starts before its
/// marks are cleared: the last collection's verdict ([`Heap::cell_age`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CellAge {
    /// A collection marked it: it survived the last collection, or was allocated during one.
    Old,
    /// Allocated since the last collection, and unmarked.
    Young,
    /// Dead and finalized, its address kept for weak handles (a zombie): no edge may name it.
    Dead,
}

/// The client of a collection driven by the heap itself. It reports nothing: such a collection
/// has no roots but the native stack, when it scans that.
struct NoClient;

impl MarkClient for NoClient {
    fn roots(&self, _: &mut dyn FnMut(u64)) {}
    fn side_slot_owners(&self, _: &mut dyn FnMut(usize)) {}
    fn trace_side_slots(&self, _: usize, _: &mut dyn FnMut(u64)) {}
    fn ephemerons(&self, _: &mut EdgeSink<'_>) {}
}

/// Why a cell was marked: retention diagnostics (see [`Heap::set_retention_tracing`]). Only
/// tests read them, so only test builds record them.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) enum MarkSource {
    /// The embedder's roots ([`MarkClient::roots`]).
    #[default]
    Root,
    /// The native stack word (or spilled register) at this address.
    Stack(usize),
    /// An edge of the cell starting at this address.
    Cell(usize),
    /// A side-table slot of this owner.
    SideSlots(usize),
    /// A conditional edge (an ephemeron, or a side-table edge from [`EdgeSink`]) and its
    /// owner (0 for none).
    Conditional { owner: usize },
}

/// Per-collection marking state. Built fresh every cycle, so nothing in it
/// can outlive the cells it describes.
#[derive(Default)]
struct Marker {
    /// Exact words proven to name allocated, non-zombie marked cells. This cache is born
    /// before native-stack scanning and dies with this marker, before sweeping or any
    /// post-marking allocation. Any conservative observation names an already-marked cell, adding no new
    /// lifetime or rooting; card-owner worklist tracing is never suppressed.
    positive_words: PositiveWords,
    #[cfg(test)]
    word_lookups: usize,
    #[cfg(test)]
    word_cache_hits: usize,
    /// What is marking now, and (when retention tracing) each cell's first marker.
    #[cfg(test)]
    source: MarkSource,
    #[cfg(test)]
    parents: Option<FastMap<usize, MarkSource>>,
    worklist: Vec<TraceWork>,
    edges: Vec<CondEdge>,
    /// Waiting edges keyed by cell start >> 4 (cells are 16-aligned).
    waiting: FastMap<usize, Vec<u32>>,
    ready: Vec<u32>,
}

/// A small direct-mapped table of exact positive words. Cell and page bits both
/// participate so aligned shared prototype/layout addresses do not systematically collide.
const MARK_CACHE_WORDS: usize = 32;
const _: () = assert!(MARK_CACHE_WORDS.is_power_of_two());
/// How many cells [`Heap::drain`] prefetches ahead of tracing them.
const MARK_PREFETCH: usize = 8;

/// Ask the CPU to bring the first two cache lines of the cell at `addr` closer ahead of its
/// trace. A hint: it neither faults nor changes memory.
#[inline(always)]
fn prefetch_cell(addr: usize) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: PRFM only hints; it never faults, whatever the address.
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{0}]",
            "prfm pldl1keep, [{0}, #64]",
            in(reg) addr,
            options(nostack, readonly, preserves_flags)
        );
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: PREFETCHT0 only hints; it never faults, whatever the address.
    unsafe {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch::<_MM_HINT_T0>(addr as *const i8);
        _mm_prefetch::<_MM_HINT_T0>(addr.wrapping_add(64) as *const i8);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = addr;
}

#[derive(Default)]
struct PositiveWords {
    words: [u64; MARK_CACHE_WORDS],
    #[cfg(test)]
    probes: Cell<usize>,
}

impl PositiveWords {
    #[inline]
    fn index(word: u64) -> usize {
        ((word >> 4) ^ (word >> 16)) as usize & (MARK_CACHE_WORDS - 1)
    }

    #[inline]
    fn contains(&self, word: u64) -> bool {
        #[cfg(test)]
        self.probes.set(self.probes.get() + 1);
        // Zero is an empty entry, never a proven allocated cell. Hash collisions only
        // replace an entry: a hit still requires equality of the complete original word.
        word != 0 && self.words[Self::index(word)] == word
    }

    #[inline]
    fn insert(&mut self, word: u64) {
        self.words[Self::index(word)] = word;
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct GcStats {
    pub(crate) allocations: u64,
    pub(crate) collections: u64,
    /// The minor collections among `collections`.
    pub(crate) minor_collections: u64,
    /// The pages the last collection swept: a minor collection's young pages, or every page.
    pub(crate) last_pages_swept: usize,
    /// The old cells the last collection traced because their cards were set (a minor
    /// collection's; a major one traces every reachable cell anyway).
    pub(crate) last_card_cells: usize,
    /// The bytes of the cells the last collection marked. A minor collection marks young cells
    /// only (old ones are marked already), so for one these are the bytes it promoted.
    pub(crate) last_marked_bytes: usize,
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
    /// could not be scanned soundly (see [`Heap::collect_kind`]).
    pub(crate) deferred_native_stack: u64,
    /// The byte budget: without a nursery, the allocated bytes past which a collection is due;
    /// with one, the heap's bytes after a minor collection past which a major one is.
    pub(crate) trigger_bytes: usize,
    /// The bytes the cells own outside the heap, charged and not released ([`ExternalBytes`]).
    pub(crate) external_bytes: usize,
    /// The last collection's phases, in nanoseconds, while [`Heap::set_phase_timing`] is on:
    /// see [`Phase`].
    pub(crate) last_phase_nanos: [u64; PHASES],
    /// The cells the last collection finalized, by descriptor (the first [`COUNTED_DESCS`]),
    /// while phases are timed.
    pub(crate) last_finalized: [u32; COUNTED_DESCS],
}

/// How many descriptors [`GcStats::last_finalized`] counts.
pub(crate) const COUNTED_DESCS: usize = 32;

/// The phases of a collection the heap times (diagnostics; [`Heap::set_phase_timing`]).
#[derive(Clone, Copy)]
pub(crate) enum Phase {
    /// Checking (heap verification), clearing or gathering the cards and marks.
    Begin,
    /// The client's roots.
    Roots,
    /// The native stack and registers.
    Stack,
    /// Side-table edges and side-slot owners.
    Side,
    /// Tracing everything reached.
    Drain,
    /// The sweep's finalizers.
    Finalize,
    /// The sweep's freeing of cells and pages.
    Free,
}

/// The number of [`Phase`]s.
pub(crate) const PHASES: usize = 7;

impl Phase {
    pub(crate) const NAMES: [&str; PHASES] = [
        "begin", "roots", "stack", "side", "drain", "finalize", "free",
    ];
}

pub(crate) struct Heap {
    descs: Vec<LayoutDesc>,
    /// By descriptor: its cells hold no edges ([`Heap::set_leaf`]), so marking one queues no
    /// trace.
    leaf: Vec<bool>,
    /// Indexed by descriptor, then size class.
    classes: Vec<[ClassPages; SIZE_CLASSES.len()]>,
    /// The allocation cursors, by descriptor × size classes + class ([`Cursor`]).
    cursors: Vec<Cursor>,
    /// Slots are reused after release, so page indices stay stable.
    pages: Vec<Option<PageMeta>>,
    /// By page slot, what marking reads first ([`MarkPage`]).
    mark_pages: Vec<MarkPage>,
    free_page_slots: Vec<u32>,
    /// P1 S4: the memory of released pages, kept (resident) for reuse, so that pages that empty
    /// out every cycle do not go back to the page source and come again, faulting anew. Trimmed
    /// to `spare_page_cap` at the end of each sweep; the pages trimmed go back to the source.
    spare_pages: Vec<usize>,
    /// Where the pages come from (see `page_source`).
    page_source: page_source::PageSource,
    /// Page number (address >> PAGE_SHIFT) to index in `pages`, for the live pages: where every
    /// word the collector resolves finds its page.
    page_table: PageTable,
    larges: BTreeMap<usize, LargeObj>,
    /// The large cells allocated since the last collection ([`LargeObj::young`]), by start: a
    /// minor collection sweeps these and leaves the old ones, as it leaves the old pages.
    young_larges: Vec<usize>,
    /// Address span of all large objects: cheap rejection of stack words.
    large_span: (usize, usize),
    /// Shared with NoGcScope/BorrowScope guards so a guard can be alive
    /// while `alloc(&mut self)` runs.
    control: Rc<Control>,
    /// The bytes of the allocated cells: those the last sweep left (`swept_bytes`) and those
    /// allocated since.
    allocated_bytes: usize,
    /// The bytes of the allocated cells the last sweep left: the live ones, zombies and, after a
    /// minor collection, the old cells that died since a collection last marked them.
    swept_bytes: usize,
    /// The byte budget (see `GcStats::trigger_bytes`): max(`min_trigger_bytes`, the bytes left
    /// by the last major collection's sweep grown by `growth_percent`).
    trigger_bytes: usize,
    min_trigger_bytes: usize,
    growth_percent: usize,
    /// The nursery budget (S6 step 10c): a minor collection is due once more than this many
    /// bytes have been allocated since the last collection. `None`, the default: every
    /// collection is major, due once the allocated bytes pass the byte budget.
    nursery_bytes: Option<usize>,
    /// The sizing policy's measurements ([`crate::gc_pacing::Pacing`], after V8's): with them a
    /// major collection's budget grows by V8's dynamic factor instead of `growth_percent`, the
    /// nursery's budget follows V8's young generation sizing, and the memory reducer runs.
    /// `None`, the default, keeps the fixed policy.
    pacing: Option<Box<crate::gc_pacing::Pacing>>,
    /// The memory the heap held from the system as the major collection in progress started.
    committed_at_mark: usize,
    /// The memory the heap held from the system once the engine finished bootstrapping
    /// ([`Heap::bootstrapped`]), until the memory reducer was told the heap first grew past it.
    bootstrap_committed: Option<usize>,
    /// A major collection is due (with a nursery): the heap held more than the byte budget after
    /// a minor collection's sweep, or a major collection was requested and deferred. Cleared as
    /// a major collection starts.
    major_due: bool,
    /// The next collection is to be major, whatever asks for it, though none is due
    /// ([`Heap::request_major`]). Cleared as a major collection starts.
    major_requested: bool,
    stress_every: usize,
    /// Heap verification at every collection (debug builds; [`Heap::set_heap_verification`]).
    verify_heap: bool,
    /// With a nursery and a stress interval, every this many collections one is major.
    stress_major_every: usize,
    /// Collections started under a stress interval.
    stress_collections: u64,
    allocs_since_gc: usize,
    /// The kind of the collection in progress, while one is (`Control::collecting`).
    collecting_kind: CollectionKind,
    /// The pages that received cells since the last collection ([`PageMeta::young`]), by slot.
    young_pages: Vec<u32>,
    /// A minor collection's old cells whose cards were set, gathered as it starts
    /// (`Heap::begin_marking`) and traced as its marking starts.
    card_cells: Vec<TraceWork>,
    /// The bytes of the cells marked by the collection in progress.
    marked_bytes: usize,
    /// The external bytes charged as the collection in progress started marking, and whether
    /// they had made it due (see `ExternalBytes` and `Heap::sweep_phase`).
    external_at_mark: usize,
    external_due_at_mark: bool,
    /// When false, allocation never collects: the embedder polls
    /// [`Heap::collection_due`] at its own safepoints, where it can supply
    /// roots the heap cannot see (the interpreter's state).
    collect_on_alloc: bool,
    /// Allocation may take its fast path ([`Heap::alloc_cell`]), which charges each cell's
    /// bytes at once but counts the cells (statistics, a stress interval's count, the page's)
    /// only as their cursor is flushed (at a refill, and as a collection starts and as it
    /// sweeps): not when allocation may collect, under a stress interval or during a
    /// collection, which count each cell as it is taken ([`Heap::update_fast_alloc`]).
    fast_alloc: bool,
    /// The embedder's copy of [`Heap::collection_due`], as a word that code polls without
    /// calling into the heap (see [`Heap::attach_due_word`]).
    due_word: Option<NonNull<Cell<u64>>>,
    /// The embedder's live-cell count for one descriptor, stored at a stable address beside
    /// its due word. Allocation increments it; each swept page subtracts its newly dead cells.
    live_count: Option<(DescId, NonNull<Cell<i64>>)>,
    /// Allocated bytes past which [`Heap::alloc`] recomputes the due word: the byte trigger;
    /// under a stress interval the current total (the allocation count decides, so every
    /// allocation recomputes); `usize::MAX` once due (no allocation can make it false; marking,
    /// sweeping and the settings recompute it).
    due_check_bytes: usize,
    stats: GcStats,
    /// Time the phases of each collection into `stats.last_phase_nanos` (diagnostics).
    phase_timing: bool,
    #[cfg(test)]
    retention_tracing: bool,
    /// The last traced collection's mark sources, by cell start.
    #[cfg(test)]
    last_parents: FastMap<usize, MarkSource>,
    /// The last traced collection's frame-pointer chain, (frame record, return address), from
    /// the collector's frame outwards (aarch64 retention diagnostics).
    #[cfg(test)]
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
            leaf: Vec::new(),
            classes: Vec::new(),
            cursors: Vec::new(),
            pages: Vec::new(),
            mark_pages: Vec::new(),
            free_page_slots: Vec::new(),
            spare_pages: Vec::new(),
            page_source: page_source::PageSource::new(),
            page_table: PageTable::new(),
            larges: BTreeMap::new(),
            young_larges: Vec::new(),
            large_span: (usize::MAX, 0),
            control: Rc::default(),
            allocated_bytes: 0,
            swept_bytes: 0,
            trigger_bytes: DEFAULT_MIN_TRIGGER,
            min_trigger_bytes: DEFAULT_MIN_TRIGGER,
            growth_percent: DEFAULT_GROWTH_PERCENT,
            nursery_bytes: None,
            pacing: None,
            committed_at_mark: 0,
            bootstrap_committed: None,
            major_due: false,
            major_requested: false,
            stress_every: 0,
            verify_heap: cfg!(test),
            stress_major_every: DEFAULT_STRESS_MAJOR_EVERY,
            stress_collections: 0,
            allocs_since_gc: 0,
            collecting_kind: CollectionKind::Major,
            young_pages: Vec::new(),
            card_cells: Vec::new(),
            marked_bytes: 0,
            external_at_mark: 0,
            external_due_at_mark: false,
            collect_on_alloc: true,
            fast_alloc: false,
            due_word: None,
            live_count: None,
            due_check_bytes: DEFAULT_MIN_TRIGGER,
            stats: GcStats::default(),
            phase_timing: false,
            #[cfg(test)]
            retention_tracing: false,
            #[cfg(test)]
            last_parents: FastMap::default(),
            #[cfg(test)]
            last_frames: Vec::new(),
        }
    }

    pub(crate) fn register_desc(&mut self, desc: LayoutDesc) -> DescId {
        let id = DescId(self.descs.len() as u32);
        self.descs.push(desc);
        self.leaf.push(false);
        self.classes
            .push(std::array::from_fn(|_| ClassPages::default()));
        self.cursors
            .extend(std::iter::repeat_n(Cursor::EMPTY, SIZE_CLASSES.len()));
        id
    }

    /// Time each collection's phases into [`GcStats::last_phase_nanos`] (diagnostics: a clock
    /// read between phases).
    pub(crate) fn set_phase_timing(&mut self, on: bool) {
        self.phase_timing = on;
    }

    /// The clock for [`Heap::phase_done`], when phases are timed.
    #[inline]
    fn phase_clock(&self) -> Option<std::time::Instant> {
        self.phase_timing.then(std::time::Instant::now)
    }

    /// Record `phase` as having run since `started` (from [`Heap::phase_clock`]), and restart
    /// the clock for the next phase.
    #[inline]
    fn phase_done(&mut self, phase: Phase, started: &mut Option<std::time::Instant>) {
        if let Some(start) = started {
            let now = std::time::Instant::now();
            self.stats.last_phase_nanos[phase as usize] += (now - *start).as_nanos() as u64;
            *start = now;
        }
    }

    /// Declare that `desc`'s cells hold no edges (its trace reports none): marking one sets its
    /// mark and queues no trace.
    pub(crate) fn set_leaf(&mut self, desc: DescId) {
        self.leaf[desc.0 as usize] = true;
        for page in self.pages.iter_mut().flatten() {
            if page.desc == desc {
                page.leaf = true;
            }
        }
    }

    /// Record why each cell is marked, for [`Heap::retention_path`] (diagnostics: costs a map
    /// entry per marked cell).
    #[cfg(test)]
    pub(crate) fn set_retention_tracing(&mut self, on: bool) {
        self.retention_tracing = on;
        self.last_parents = FastMap::default();
    }

    /// For the last collection with retention tracing on: what marked the cell containing
    /// `addr`, then what marked that, back to a root or the native stack. Empty when the cell
    /// was not marked.
    #[cfg(test)]
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
                MarkSource::Conditional { owner } if owner != 0 => owner,
                _ => break,
            };
        }
        path
    }

    /// The return address into the function whose frame holds stack word `addr` in the last
    /// traced collection (symbolize it with addr2line against the binary), and how far below
    /// that frame's record the word lies. `None` when no frame record covers `addr`.
    #[cfg(test)]
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
        // Cells taken so far count before the interval changes.
        self.flush_cursors();
        self.stress_every = allocs;
        self.update_fast_alloc();
        self.refresh_due();
    }

    /// Heap verification, in debug builds: as every collection starts, check the write barrier's
    /// invariant, the young pages' bookkeeping and the client's records ([`Heap::begin_marking`]);
    /// after its marking, check every marked cell's edges ([`Heap::verify`]); after its sweep,
    /// the live-cell accounting. Each walks the whole heap, which makes a minor collection cost
    /// what a major one does, so it is off but in the engine's unit tests (`TRUST_JS_GC_VERIFY`
    /// turns it on for an Agent's heap; see `gc_collect::gc_verify`). Release builds never
    /// verify. Checks proportional to a collection's own work run in every debug build.
    pub(crate) fn set_heap_verification(&mut self, on: bool) {
        self.verify_heap = on;
    }

    /// Whether collections verify the heap ([`Heap::set_heap_verification`]): never in release
    /// builds.
    pub(crate) fn verifies(&self) -> bool {
        cfg!(debug_assertions) && self.verify_heap
    }

    /// With a nursery and a stress interval, make every `collections`-th collection that the heap
    /// asks for major ([`Heap::due_kind`]); 1 makes every one major.
    pub(crate) fn set_stress_major_every(&mut self, collections: usize) {
        self.stress_major_every = collections.max(1);
    }

    /// The nursery budget (see the `nursery_bytes` field); `None` makes every collection major.
    pub(crate) fn set_nursery_bytes(&mut self, bytes: Option<usize>) {
        self.nursery_bytes = bytes;
        if let Some(pacing) = &mut self.pacing {
            pacing.fix_nursery(bytes);
        }
        self.reset_external_budget();
        self.refresh_due();
    }

    /// Follow V8's sizing policy from now on (see [`crate::gc_pacing`] and the `pacing` field),
    /// with the nursery budget set: when `dynamic_nursery`, that budget is the policy's first
    /// and smallest, and it grows and shrinks with the measurements.
    pub(crate) fn enable_pacing(&mut self, dynamic_nursery: bool) {
        self.pacing = Some(Box::new(crate::gc_pacing::Pacing::new(
            self.nursery_bytes,
            dynamic_nursery,
        )));
    }

    /// The memory the heap holds from the system: its pages, spare ones included, and its large
    /// cells.
    fn committed_bytes(&self) -> usize {
        let pages = self.pages.len() - self.free_page_slots.len() + self.spare_pages.len();
        pages * PAGE_SIZE + self.larges.values().map(|large| large.size).sum::<usize>()
    }

    /// The engine finished bootstrapping its first Realm (V8's `NotifyBootstrapComplete`): the
    /// first time a minor collection finds the heap holding 1 MB more than now, with no major
    /// collection yet, the memory reducer is told of possible garbage.
    pub(crate) fn bootstrapped(&mut self) {
        if self.pacing.is_some() {
            self.bootstrap_committed = Some(self.committed_bytes());
        }
    }

    /// The nursery's budget (see the `nursery_bytes` field).
    #[cfg(test)]
    pub(crate) fn nursery_budget(&self) -> Option<usize> {
        self.nursery_bytes
    }

    /// The memory the heap holds from the system (see `committed_bytes`).
    #[cfg(test)]
    pub(crate) fn committed(&self) -> usize {
        self.committed_bytes()
    }

    /// The memory reducer's state.
    #[cfg(test)]
    pub(crate) fn memory_reducer(&self) -> Option<crate::gc_pacing::MemoryReducer> {
        self.pacing.as_ref().map(|pacing| pacing.reducer)
    }

    /// When the memory reducer's timer should next run ([`Heap::memory_reducer_timer`]), if it
    /// waits to.
    pub(crate) fn memory_reducer_deadline(&self) -> Option<std::time::Instant> {
        self.pacing
            .as_ref()
            .and_then(|pacing| pacing.reducer.deadline())
    }

    /// Run the memory reducer's timer at `now` ([`crate::gc_pacing::MemoryReducer`]);
    /// `can_start` is whether a collection could run now. True when the caller is to collect
    /// majorly now to reduce memory: that collection then grows the budget minimally, returns
    /// the nursery to its smallest and keeps no spare pages beyond the minimum.
    pub(crate) fn memory_reducer_timer(
        &mut self,
        now: std::time::Instant,
        can_start: bool,
    ) -> bool {
        let allocated = self.allocated_bytes.saturating_sub(self.swept_bytes);
        let committed = self.committed_bytes();
        let Some(pacing) = &mut self.pacing else {
            return false;
        };
        // V8's timer samples allocation, then asks whether its rate is low.
        pacing.sample(now, allocated);
        let low = pacing.low_allocation_rate(now, allocated);
        let start = pacing.reducer.timer(now, committed, low, can_start);
        if start {
            pacing.reduce_memory_next();
        }
        start
    }

    /// Tell the memory reducer, at `now`, that much of the heap may just have died (the host
    /// let a document go, say): it collects after its start delay if the program is quiet.
    pub(crate) fn notify_possible_garbage(&mut self, now: std::time::Instant) {
        if let Some(pacing) = &mut self.pacing {
            pacing.reducer.possible_garbage(now);
        }
    }

    /// The external budget a new setting starts from: the nursery's, or without a nursery the
    /// byte budget (see `ExternalBytes`; each sweep recomputes it).
    fn reset_external_budget(&mut self) {
        let base = self.nursery_bytes.unwrap_or(self.trigger_bytes).max(1);
        self.control.external.budget.set(base);
    }

    /// Make the next collection major, whatever asks for it, without making one due: for an
    /// embedder that released a graph the heap holds as old cells (a retired document's Realm),
    /// which only a major collection reclaims, and that collects at a boundary of its own.
    pub(crate) fn request_major(&mut self) {
        self.major_requested = true;
    }

    /// See the `collect_on_alloc` field.
    pub(crate) fn set_collect_on_alloc(&mut self, collect: bool) {
        // Cells taken so far count before allocation may collect on its counts.
        self.flush_cursors();
        self.collect_on_alloc = collect;
        self.update_fast_alloc();
    }

    /// Whether a collection is due: with a nursery, its budget is spent or a major collection is
    /// due; without one, the byte budget is spent; or the stress interval elapsed, or an earlier
    /// request was deferred, or the external bytes charged since the last collection passed
    /// their budget ([`ExternalBytes`]). Which kind runs is the collector's choice
    /// ([`Heap::due_kind`]).
    pub(crate) fn collection_due(&self) -> bool {
        self.control.pending_collect.get()
            || (self.stress_every > 0 && self.allocs_since_gc >= self.stress_every)
            || self.bytes_due(0)
            || self.control.external.due.get()
    }

    /// The heap's control block, shared with the containers that charge external bytes to it
    /// (see [`Control::charge_external`]).
    pub(crate) fn control(&self) -> Rc<Control> {
        self.control.clone()
    }

    /// Whether the allocated bytes make a collection due, once `upcoming` more are allocated.
    fn bytes_due(&self, upcoming: usize) -> bool {
        let allocated = self.allocated_bytes.saturating_add(upcoming);
        match self.nursery_bytes {
            Some(nursery) => self.major_due || allocated.saturating_sub(self.swept_bytes) > nursery,
            None => allocated > self.trigger_bytes,
        }
    }

    /// The kind of collection the heap asks for when it collects for its own reasons (a budget,
    /// the stress interval, a deferred request): major without a nursery, when a major
    /// collection is due or requested, or for every `stress_major_every`-th collection under a
    /// stress interval; minor otherwise.
    pub(crate) fn due_kind(&self) -> CollectionKind {
        let stress_major = self.stress_every > 0
            && (self.stress_collections + 1).is_multiple_of(self.stress_major_every as u64);
        if self.nursery_bytes.is_none() || self.major_due || self.major_requested || stress_major {
            CollectionKind::Major
        } else {
            CollectionKind::Minor
        }
    }

    /// Keep `word` equal to [`Heap::collection_due`] (1 or 0) from now on, for code that polls
    /// for a collection without calling into the heap (the native JIT's polls; see
    /// `value::LiveCount`). The heap is the word's only writer, with its control block's external
    /// ledger. An allocation that spends a budget or reaches the stress interval sets it
    /// ([`Heap::alloc`]), and so do a deferred request and an external charge that spends the
    /// external budget ([`Control::charge_external`]); marking (which takes the pending request
    /// and a due major collection), the end of every sweep (new budgets and interval, a major
    /// collection a minor one made due, and any request that arrived meanwhile) and the settings
    /// recompute it. Nothing else changes what `collection_due` reads.
    ///
    /// # Safety
    /// `word` must stay valid at its address, and written by no one else, for as long as the
    /// heap lives.
    pub(crate) unsafe fn attach_due_word(&mut self, word: NonNull<Cell<u64>>) {
        self.due_word = Some(word);
        self.control.external.due_word.set(Some(word));
        self.refresh_due();
    }

    /// Publish the number of allocated, non-zombie cells of `desc` through `count`. Only the
    /// heap writes it, including on teardown; its address stays usable by generated polls.
    ///
    /// # Safety
    /// `count` must remain valid at its address, and written by no one else, until the heap
    /// has been dropped (including its final `release_all`).
    pub(crate) unsafe fn attach_live_count(&mut self, desc: DescId, count: NonNull<Cell<i64>>) {
        // SAFETY: the caller keeps the count valid until the heap is dropped.
        unsafe { count.as_ref() }.set(self.live_count(desc) as i64);
        self.live_count = Some((desc, count));
    }

    /// Whether `addr` is the start of an initialized, non-zombie cell of `desc`. This resolves
    /// ownership without reading the cell, including when another heap is active.
    pub(crate) fn owns_cell(&self, addr: usize, desc: DescId) -> bool {
        self.locate(addr as u64).is_some_and(|cell| {
            cell.start == addr
                && cell.desc == desc
                && matches!(
                    self.cell_age(addr as u64),
                    Some(CellAge::Old | CellAge::Young)
                )
        })
    }

    /// Collections run so far (without the page walk of [`Heap::stats`]).
    #[cfg(test)]
    pub(crate) fn collections(&self) -> u64 {
        self.stats.collections
    }

    /// Minor collections run so far.
    #[cfg(test)]
    pub(crate) fn minor_collections(&self) -> u64 {
        self.stats.minor_collections
    }

    /// The pages that received cells since the last collection.
    #[cfg(test)]
    pub(crate) fn young_page_count(&self) -> usize {
        self.young_pages.len()
    }

    /// The large cells allocated: live, or dead and weakly pinned.
    #[cfg(test)]
    pub(crate) fn large_cells(&self) -> usize {
        self.larges.len()
    }

    /// Bytes allocated past the budget that makes a collection due: the nursery's with a
    /// nursery (bytes allocated since the last collection), the byte budget without.
    #[cfg(test)]
    pub(crate) fn budget_overshoot(&self) -> usize {
        match self.nursery_bytes {
            Some(nursery) => self
                .allocated_bytes
                .saturating_sub(self.swept_bytes)
                .saturating_sub(nursery),
            None => self.allocated_bytes.saturating_sub(self.trigger_bytes),
        }
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
            match self.nursery_bytes {
                Some(nursery) => self.swept_bytes.saturating_add(nursery),
                None => self.trigger_bytes,
            }
        };
        if let Some(word) = self.due_word {
            // SAFETY: `attach_due_word`'s contract.
            unsafe { word.as_ref() }.set(u64::from(due));
        }
    }

    /// The byte budget: max(`min_bytes`, the bytes a major collection's sweep leaves, with the
    /// external bytes still charged ([`ExternalBytes`]), × (1 + `growth_percent`/100)), so that
    /// marking work per allocated byte stays bounded as the live heap grows. Without a nursery a
    /// collection is due once the allocated bytes pass it; with one, a major collection is due
    /// once the heap's bytes and the external bytes after a minor collection do. Until the next
    /// major collection, the budget is `min_bytes`.
    pub(crate) fn set_heap_growth(&mut self, min_bytes: usize, growth_percent: usize) {
        if let Some(pacing) = &mut self.pacing {
            pacing.fix_growth();
        }
        self.min_trigger_bytes = min_bytes;
        self.growth_percent = growth_percent;
        self.trigger_bytes = min_bytes;
        self.reset_external_budget();
        self.refresh_due();
    }

    pub(crate) fn stats(&self) -> GcStats {
        let mut s = self.stats;
        // The cells the cursors have taken since they were last flushed.
        s.allocations += self.cursors.iter().map(|c| u64::from(c.taken)).sum::<u64>();
        let pages = self.pages.iter().flatten();
        s.pages_live = pages.clone().count();
        s.pages_spare = self.spare_pages.len();
        // From the bits, which the cursors keep exact, unlike the pages' counts.
        let live = |p: &PageMeta| -> usize {
            p.allocated
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum()
        };
        s.live_cells = pages.clone().map(live).sum::<usize>() + self.larges.len();
        s.live_bytes = pages.map(|p| live(p) * p.cell_size).sum::<usize>()
            + self.larges.values().map(|l| l.size).sum::<usize>();
        s.trigger_bytes = self.trigger_bytes;
        s.external_bytes = self.control.external.live.get();
        s
    }

    /// Enter a bounded no-GC region (I9). Allocation proceeds inside;
    /// collection requests are deferred, and a deferred collection runs at
    /// the next allocation or collection request after the outermost
    /// region exits, never at region exit itself (exit is not a safepoint).
    #[cfg(test)]
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
    #[cfg(test)]
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
    #[cfg(test)]
    pub(crate) fn collection_pending(&self) -> bool {
        self.control.pending_collect.get()
    }

    /// Allocate a zeroed cell for `desc`. May first collect, conservatively
    /// (see [`Heap::collect_kind`]). The caller initializes, then publishes (I9).
    ///
    /// Between [`Heap::start_marking`] and [`Heap::finish_collection`] the
    /// cell is allocated marked (black): marking is over, so the cell survives
    /// this sweep untraced. Code running in that window must not store a
    /// reference to an unmarked cell into a new or surviving one.
    #[inline]
    pub(crate) fn alloc(&mut self, desc: DescId, size: usize) -> *mut u8 {
        self.alloc_cell(desc, size, true)
            .expect("GC large allocation failed")
    }

    /// [`Heap::alloc`] without zeroing the cell, or `None` when the memory of a large cell
    /// cannot be had: for a leaf kind ([`Heap::set_leaf`]) whose owner writes every byte it later
    /// reads, and which neither the collector nor anyone else reads before then.
    #[inline]
    pub(crate) fn try_alloc_uninit(&mut self, desc: DescId, size: usize) -> Option<*mut u8> {
        debug_assert!(self.leaf[desc.0 as usize], "uninitialized cells are leaves");
        self.alloc_cell(desc, size, false)
    }

    /// The allocation path: a small cell from its class's cursor, which holds the free cells of
    /// one word of a page's bitmap, unless the heap counts each allocation as it happens
    /// ([`Heap::fast_alloc`]); everything else ([`Heap::alloc_cell_slow`]) out of line. Inlined
    /// into each caller, where the size is usually a constant, and with it the class and the
    /// length the cell is zeroed for.
    #[inline(always)]
    fn alloc_cell(&mut self, desc: DescId, size: usize, zero: bool) -> Option<*mut u8> {
        if let Some(class) = size_class(size) {
            if self.fast_alloc {
                let slot = desc.0 as usize * SIZE_CLASSES.len() + class;
                if let Some(cursor) = self.cursors.get_mut(slot) {
                    if cursor.free != 0 {
                        let ptr = cursor.take();
                        if zero {
                            // SAFETY: the cell lies inside its page, `SIZE_CLASSES[class]` bytes
                            // (the cursor's cell size), and now belongs to the caller.
                            unsafe { std::ptr::write_bytes(ptr, 0, SIZE_CLASSES[class]) };
                        }
                        self.count_live(desc);
                        self.charge_bytes(SIZE_CLASSES[class]);
                        return Some(ptr);
                    }
                }
            }
        }
        self.alloc_cell_slow(desc, size, zero)
    }

    /// Every allocation the fast path does not take: a cursor to load, a large cell, an
    /// allocation that may collect first, under a stress interval or during a collection. The
    /// fast path counts its cells as their cursor is flushed; this one, when the heap counts
    /// each allocation, flushes the cursor at once. Both charge the bytes at once.
    #[inline(never)]
    fn alloc_cell_slow(&mut self, desc: DescId, size: usize, zero: bool) -> Option<*mut u8> {
        debug_assert!(
            (desc.0 as usize) < self.descs.len(),
            "unregistered GC descriptor"
        );
        let class = size_class(size);
        let collecting = self.control.collecting.get();
        if !collecting {
            self.maybe_collect_for_alloc(class.map_or(size, |class| SIZE_CLASSES[class]));
        }
        let ptr = match class {
            Some(class) => {
                let slot = desc.0 as usize * SIZE_CLASSES.len() + class;
                if self.cursors[slot].free == 0 {
                    self.refill(slot, desc, class);
                }
                let ptr = self.cursors[slot].take();
                if zero {
                    // SAFETY: as in `alloc_cell`.
                    unsafe { std::ptr::write_bytes(ptr, 0, SIZE_CLASSES[class]) };
                }
                self.charge_bytes(SIZE_CLASSES[class]);
                if !self.fast_alloc {
                    self.flush_cursor(slot);
                }
                ptr
            }
            None => {
                let ptr = self.alloc_large(desc, size, zero)?;
                self.charge_bytes(size);
                self.count_cells(1);
                ptr
            }
        };
        if collecting {
            let cell = self
                .locate(ptr as u64)
                .expect("a fresh cell resolves to itself");
            self.mark(&cell);
        }
        self.count_live(desc);
        Some(ptr)
    }

    /// Count a new cell of `desc` in the embedder's live count, if it keeps one for `desc`.
    #[inline(always)]
    fn count_live(&self, desc: DescId) {
        if let Some((counted, count)) = self.live_count {
            if desc == counted {
                // SAFETY: `attach_live_count` keeps the count valid until this heap is dropped.
                let count = unsafe { count.as_ref() };
                count.set(count.get() + 1);
            }
        }
    }

    /// Charge a new cell's `bytes` to the budgets: exactly, so the allocation that spends a
    /// budget is the one that makes the collection due.
    #[inline(always)]
    fn charge_bytes(&mut self, bytes: usize) {
        self.allocated_bytes = self.allocated_bytes.saturating_add(bytes);
        // The due word's one compare: an allocation past the budget (any allocation under a
        // stress interval) recomputes it, out of line.
        if self.allocated_bytes > self.due_check_bytes {
            self.refresh_due_cold();
        }
    }

    #[cold]
    #[inline(never)]
    fn refresh_due_cold(&mut self) {
        self.refresh_due();
    }

    /// Count `cells` new cells in the statistics and toward a stress interval, as their cursor
    /// is flushed (see [`Heap::fast_alloc`]).
    fn count_cells(&mut self, cells: usize) {
        self.stats.allocations += cells as u64;
        self.allocs_since_gc += cells;
        if self.stress_every > 0 {
            // The interval counts cells, which `charge_bytes` did not see yet.
            self.refresh_due();
        }
    }

    /// Whether allocation may take the fast path ([`Heap::fast_alloc`]): the heap does not
    /// collect on allocation, has no stress interval and is not collecting.
    fn update_fast_alloc(&mut self) {
        self.fast_alloc =
            !self.collect_on_alloc && self.stress_every == 0 && !self.control.collecting.get();
    }

    fn maybe_collect_for_alloc(&mut self, upcoming: usize) {
        if !self.collect_on_alloc {
            // The embedder collects at its own safepoints (the engine's heaps).
            return;
        }
        // Stress collects *before* an allocation: collecting after handing
        // out the cell would free it before the caller can root it.
        let due = self.control.pending_collect.get()
            || self.bytes_due(upcoming)
            || (self.stress_every > 0 && self.allocs_since_gc >= self.stress_every);
        if due {
            self.collect_kind(self.due_kind());
        }
    }

    /// Load the cursor in `slot` with free cells: the next word of its page that has some, or
    /// the first of the next page with room, or of a new page. A page that takes cells is young
    /// until the next sweep (the next minor collection sweeps it).
    #[cold]
    #[inline(never)]
    fn refill(&mut self, slot: usize, desc: DescId, class: usize) {
        self.flush_cursor(slot);
        loop {
            let page = self.cursors[slot].page;
            if page != NO_PAGE {
                let meta = self.pages[page as usize]
                    .as_mut()
                    .expect("a cursor's page is live");
                if meta.next_free_word().is_some() {
                    let cursor = &mut self.cursors[slot];
                    cursor.free = std::mem::take(&mut meta.free);
                    cursor.word = meta.next_word - 1;
                    return;
                }
                // Full: on no list until a sweep frees one of its cells.
                self.cursors[slot] = Cursor::EMPTY;
                self.classes[desc.0 as usize][class].current = None;
            }
            let index = match self.classes[desc.0 as usize][class].avail.pop() {
                Some(index) => {
                    let meta = self.pages[index as usize]
                        .as_mut()
                        .expect("an available page is live");
                    if !meta.young {
                        meta.young = true;
                        self.young_pages.push(index);
                    }
                    meta.reset_cursor();
                    index
                }
                // A new page is young from the start.
                None => self.new_page(desc, class),
            };
            self.classes[desc.0 as usize][class].current = Some(index);
            let meta = self.pages[index as usize]
                .as_mut()
                .expect("a page just taken is live");
            self.cursors[slot] = Cursor {
                free: 0,
                bits: meta.allocated.as_mut_ptr(),
                base: meta.base,
                cell_size: meta.cell_size,
                word: 0,
                page: index,
                taken: 0,
                high: 0,
            };
        }
    }

    /// Bring the cursor in `slot`'s page's `live` and `next_unused` up to date with the cells it
    /// handed out.
    fn flush_cursor(&mut self, slot: usize) {
        let cursor = &mut self.cursors[slot];
        if cursor.page != NO_PAGE && cursor.taken != 0 {
            let taken = std::mem::take(&mut cursor.taken) as usize;
            let meta = self.pages[cursor.page as usize]
                .as_mut()
                .expect("a cursor's page is live");
            meta.live += taken;
            meta.next_unused = meta.next_unused.max(cursor.high as usize);
            self.count_cells(taken);
        }
    }

    /// The page in `slot`, its counts brought up to date with its cursor (tests).
    #[cfg(test)]
    fn page_view(&mut self, slot: usize) -> &PageMeta {
        self.flush_cursors();
        self.pages[slot].as_ref().expect("a live page")
    }

    /// [`Heap::flush_cursor`] every cursor: as a collection starts, and before its sweep reads
    /// the pages' counts.
    fn flush_cursors(&mut self) {
        for slot in 0..self.cursors.len() {
            self.flush_cursor(slot);
        }
    }

    /// Forget every cursor and current page, flushed already: after a sweep, whose bits they
    /// cached. The next allocation of each class takes a page again, from its list.
    fn reset_cursors(&mut self) {
        self.cursors.fill(Cursor::EMPTY);
        for class in self.classes.iter_mut().flatten() {
            class.current = None;
        }
    }

    fn new_page(&mut self, desc: DescId, class: usize) -> u32 {
        let cell_size = SIZE_CLASSES[class];
        let base = match self.spare_pages.pop() {
            // A spare has no remaining cell or handle. Its payload may be dirty: allocation
            // zeros each whole cell before handing it out, without touching the virgin tail.
            Some(base) => base as *mut u8,
            None => self.page_source.take(),
        };
        debug_assert!(
            (base as usize).is_multiple_of(PAGE_SIZE),
            "GC page off its alignment"
        );
        // A carded kind's page ends in its card table. Reused storage can have held an
        // uncarded kind whose payload reached that table, so it is explicitly cleared below.
        let carded = self.descs[desc.0 as usize].carded;
        let cells = if carded {
            CARD_TABLE_OFFSET / cell_size
        } else {
            PAGE_SIZE / cell_size
        };
        debug_assert!(!carded || cells * cell_size <= CARD_TABLE_OFFSET);
        let words = cells.div_ceil(64);
        let mut meta = PageMeta {
            base: base as usize,
            cell_size,
            reciprocal: u32::try_from((1u64 << 32).div_ceil(cell_size as u64))
                .expect("a cell of at least 2 bytes"),
            cells,
            desc,
            class,
            carded,
            leaf: self.leaf[desc.0 as usize],
            has_zombies: false,
            has_waiters: false,
            young: true,
            marks: PageBits::zeroed(words),
            allocated: PageBits::zeroed(words),
            waiters: vec![0; words],
            zombies: vec![0; words],
            pins: vec![0; words],
            next_unused: 0,
            free: 0,
            next_word: 0,
            live: 0,
        };
        meta.clear_cards();
        let mark_page = MarkPage::of(&meta);
        let index = match self.free_page_slots.pop() {
            Some(index) => {
                self.pages[index as usize] = Some(meta);
                self.mark_pages[index as usize] = mark_page;
                index
            }
            None => {
                self.pages.push(Some(meta));
                self.mark_pages.push(mark_page);
                (self.pages.len() - 1) as u32
            }
        };
        self.page_table.insert(base as usize >> PAGE_SHIFT, index);
        // Made for a cell about to be allocated: young until the next sweep.
        self.young_pages.push(index);
        index
    }

    /// A large cell of `size` bytes, or `None` when its memory cannot be had.
    fn alloc_large(&mut self, desc: DescId, size: usize, zero: bool) -> Option<*mut u8> {
        let layout = Layout::from_size_align(size, 16).ok()?;
        // SAFETY: size > 4 KiB; null is checked below.
        let ptr = unsafe {
            if zero {
                alloc_zeroed(layout)
            } else {
                std::alloc::alloc(layout)
            }
        };
        if ptr.is_null() {
            return None;
        }
        let start = ptr as usize;
        self.larges.insert(
            start,
            LargeObj {
                size,
                desc,
                young: true,
                marked: false,
                waiters: false,
                zombie: false,
                pinned: false,
            },
        );
        self.young_larges.push(start);
        self.large_span = (
            self.large_span.0.min(start),
            self.large_span.1.max(start + size),
        );
        Some(ptr)
    }

    /// Resolve a candidate word (raw pointer or tagged word, interior
    /// addresses included) to its cell. `None` for non-heap words, among them an address in a
    /// page's tail past its last cell or in its card table: the bound on the cell's index
    /// rejects both.
    fn locate(&self, word: u64) -> Option<CellRef> {
        if !may_name_cell(word) {
            return None;
        }
        let addr = (word & PAYLOAD_MASK) as usize;
        if let Some(index) = self.page_table.get(addr >> PAGE_SHIFT) {
            let page = self.pages[index as usize].as_ref()?;
            let cell = page.cell_index(addr);
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

    /// The names of the descriptors, by id (diagnostics).
    pub(crate) fn desc_names(&self) -> Vec<&'static str> {
        self.descs.iter().map(|desc| desc.name).collect()
    }

    /// The name of the kind of the cell `word` resolves to (diagnostics).
    #[cfg(debug_assertions)]
    pub(crate) fn kind_name(&self, word: u64) -> Option<&'static str> {
        self.locate(word)
            .map(|cell| self.descs[cell.desc.0 as usize].name)
    }

    /// Resolve `ptr` to the start of its cell, accepting interior addresses.
    pub(crate) fn resolve(&self, word: u64) -> Option<*const u8> {
        self.locate(word).map(|cell| cell.start as *const u8)
    }

    /// The age of the allocated cell that `word` resolves to, as marking resolves it (interior
    /// and packed words included); `None` when it names no allocated cell of this heap. Between
    /// collections, and as one starts before its marks are cleared
    /// ([`MarkClient::check_records`]), the marks are the last collection's.
    pub(crate) fn cell_age(&self, word: u64) -> Option<CellAge> {
        let cell = self.locate(word)?;
        let (allocated, zombie, marked) = match cell.loc {
            Loc::Small { page, index } => {
                let page = self.pages[page as usize].as_ref()?;
                let index = index as usize;
                (
                    bit(&page.allocated, index),
                    bit(&page.zombies, index),
                    bit(&page.marks, index),
                )
            }
            Loc::Large => {
                let large = self.larges.get(&cell.start)?;
                (true, large.zombie, large.marked)
            }
        };
        allocated.then_some(if zombie {
            CellAge::Dead
        } else if marked {
            CellAge::Old
        } else {
            CellAge::Young
        })
    }

    fn is_marked(&self, cell: &CellRef) -> bool {
        match cell.loc {
            Loc::Small { page, index } => self.pages[page as usize]
                .as_ref()
                .is_some_and(|p| bit(&p.marks, index as usize)),
            Loc::Large => self.larges.get(&cell.start).is_some_and(|l| l.marked),
        }
    }

    /// Set the mark of an allocated live cell, distinguishing an existing mark from a
    /// rejected free/zombie address without resolving the candidate a second time.
    #[inline]
    fn mark(&mut self, cell: &CellRef) -> MarkOutcome {
        match cell.loc {
            Loc::Small { page, index } => {
                let Some(page) = self.pages[page as usize].as_mut() else {
                    return MarkOutcome::Rejected;
                };
                let index = index as usize;
                if !bit(&page.allocated, index) || bit(&page.zombies, index) {
                    return MarkOutcome::Rejected;
                }
                if bit(&page.marks, index) {
                    return MarkOutcome::AlreadyMarked;
                }
                set_bit(&mut page.marks, index, true);
                self.marked_bytes += page.cell_size;
                MarkOutcome::NewlyMarked {
                    has_waiters: bit(&page.waiters, index),
                }
            }
            Loc::Large => {
                let Some(large) = self.larges.get_mut(&cell.start) else {
                    return MarkOutcome::Rejected;
                };
                if large.zombie {
                    return MarkOutcome::Rejected;
                }
                if large.marked {
                    return MarkOutcome::AlreadyMarked;
                }
                large.marked = true;
                self.marked_bytes += large.size;
                MarkOutcome::NewlyMarked {
                    has_waiters: large.waiters,
                }
            }
        }
    }

    /// Return true only after this cell is proven live and all first-mark bookkeeping,
    /// including waiter notifications, is complete. A caller may then cache its exact word.
    #[inline]
    fn mark_cell(&mut self, cell: CellRef, m: &mut Marker) -> bool {
        match self.mark(&cell) {
            MarkOutcome::Rejected => false,
            MarkOutcome::AlreadyMarked => true,
            MarkOutcome::NewlyMarked { has_waiters } => {
                #[cfg(test)]
                if let Some(parents) = &mut m.parents {
                    parents.insert(cell.start, m.source);
                }
                self.stats.cells_marked += 1;
                if !self.leaf[cell.desc.0 as usize] {
                    m.worklist.push(TraceWork {
                        cell: cell.start,
                        desc: cell.desc,
                        purpose: TracePurpose::Full,
                    });
                }
                if has_waiters {
                    Self::wake(cell.start, m);
                }
                true
            }
        }
    }

    /// Ready the conditional edges waiting on the cell at `start`, which was just marked.
    #[cold]
    fn wake(start: usize, m: &mut Marker) {
        if let Some(edges) = m.waiting.remove(&(start >> 4)) {
            m.ready.extend(edges);
        }
    }

    /// Mark the cell `word` resolves to, as [`Heap::locate`] resolves it (interior and tagged
    /// words included, others ignored), and queue its trace if this marks it first. A small
    /// cell takes [`Heap::mark_small`]; an address in the large cells' span the general path.
    #[inline(always)]
    fn mark_word(&mut self, word: u64, m: &mut Marker) {
        if m.positive_words.contains(word) {
            #[cfg(test)]
            {
                m.word_cache_hits += 1;
            }
            return;
        }
        #[cfg(test)]
        {
            m.word_lookups += 1;
        }
        if !may_name_cell(word) {
            return;
        }
        let addr = (word & PAYLOAD_MASK) as usize;
        let live = match self.page_table.get(addr >> PAGE_SHIFT) {
            Some(slot) => self.mark_small(slot, addr, m),
            None if (self.large_span.0..self.large_span.1).contains(&addr) => {
                self.mark_large_word(word, m)
            }
            None => false,
        };
        if live {
            m.positive_words.insert(word);
        }
    }

    /// [`Heap::mark_word`] for an address in the page in slot `slot`: [`Heap::locate`] and
    /// [`Heap::mark_cell`] in one look at the page's metadata. True when the address names a
    /// live cell, marked now or before.
    #[inline(always)]
    fn mark_small(&mut self, slot: u32, addr: usize, m: &mut Marker) -> bool {
        debug_assert!(self.pages.get(slot as usize).is_some_and(Option::is_some));
        // SAFETY: the page table maps the live pages' numbers to their slots and nothing else
        // (`Heap::new_page` and `Heap::release_page` keep the two in step, and debug builds
        // check it after every sweep), and `mark_pages` has an entry for every slot.
        let page = unsafe { *self.mark_pages.get_unchecked(slot as usize) };
        let index = page.cell_index(addr);
        if index >= page.cells as usize {
            return false;
        }
        // Marking marks allocated cells that are no zombies, and nothing else is marked: a
        // sweep leaves its free cells and zombies unmarked.
        // SAFETY: the page's bits cover its `cells`; its `PageMeta` owns them, live (above).
        if unsafe { *page.marks.add(index / 64) } & (1 << (index % 64)) != 0 {
            debug_assert!(self.pages[slot as usize].as_ref().is_some_and(|page| {
                page.allocated[index / 64] & !page.zombies[index / 64] & (1 << (index % 64)) != 0
            }));
            return true;
        }
        self.mark_unmarked(slot, index, m)
    }

    /// The rest of [`Heap::mark_small`], out of line, for the unmarked cell at `index` of the
    /// page in slot `slot`: mark it if it is allocated and no zombie, and queue its trace.
    #[inline(never)]
    fn mark_unmarked(&mut self, slot: u32, index: usize, m: &mut Marker) -> bool {
        debug_assert!(self.pages.get(slot as usize).is_some_and(Option::is_some));
        // SAFETY: as in `mark_small`, which found the slot through the page table.
        let page = unsafe {
            self.pages
                .get_unchecked_mut(slot as usize)
                .as_mut()
                .unwrap_unchecked()
        };
        let (word, bit) = (index / 64, 1u64 << (index % 64));
        if page.allocated[word] & bit == 0 || (page.has_zombies && page.zombies[word] & bit != 0) {
            return false;
        }
        page.marks[word] |= bit;
        self.marked_bytes += page.cell_size;
        self.stats.cells_marked += 1;
        let start = page.cell_ptr(index);
        #[cfg(test)]
        if let Some(parents) = &mut m.parents {
            parents.insert(start, m.source);
        }
        if !page.leaf {
            m.worklist.push(TraceWork {
                cell: start,
                desc: page.desc,
                purpose: TracePurpose::Full,
            });
        }
        if page.has_waiters && page.waiters[word] & bit != 0 {
            Self::wake(start, m);
        }
        true
    }

    /// [`Heap::mark_word`] for an address in the large cells' span.
    #[inline(never)]
    fn mark_large_word(&mut self, word: u64, m: &mut Marker) -> bool {
        self.locate(word)
            .is_some_and(|cell| self.mark_cell(cell, m))
    }

    /// Request a major collection, as [`Heap::collect_kind`] does.
    #[cfg(test)]
    pub(crate) fn collect(&mut self) -> bool {
        self.collect_kind(CollectionKind::Major)
    }

    /// Request a collection of `kind` with conservative native roots (Decision 0):
    /// this thread's registered native stack chain and a register snapshot,
    /// its only roots. Returns false when the request is deferred
    /// (pending until the next allocation or request): inside a no-GC
    /// region; while a GcCell borrow is outstanding (debug builds assert,
    /// I9); or when the native stack cannot be scanned soundly — no stack
    /// registered on this thread (see [`register_thread_native_stack`]), SP
    /// outside every registered region (e.g. an unregistered `stacker::grow`
    /// segment), or no scanner for this target (wasm32).
    fn collect_kind(&mut self, kind: CollectionKind) -> bool {
        if self.blocked() {
            return self.defer(false, kind);
        }
        if !native::scannable(native::current_sp()) {
            return self.defer(true, kind);
        }
        self.mark_phase(true, &NoClient, kind);
        self.sweep_phase();
        true
    }

    /// Collect with no roots at all (no native stack or registers, and no
    /// embedder to report any), so every cell dies: for tests of what a
    /// collection does rather than what it keeps. A major collection, deferred like
    /// [`Heap::collect`] inside no-GC regions and while borrows are outstanding.
    #[cfg(test)]
    pub(crate) fn collect_at_safepoint(&mut self) -> bool {
        if self.blocked() {
            return self.defer(false, CollectionKind::Major);
        }
        self.mark_phase(false, &NoClient, CollectionKind::Major);
        self.sweep_phase();
        true
    }

    /// Phase 1 of an embedder-driven collection of `kind`: mark from `client`'s roots, side
    /// slots and ephemerons and, when `native_roots`, this thread's native stack (deferred as
    /// for [`Heap::collect_kind`] when it cannot be scanned soundly), and in a minor collection
    /// from the old cells whose cards are set. Marks stay set until [`Heap::finish_collection`];
    /// meanwhile the embedder sweeps its own tables using [`Heap::dead_cells`],
    /// [`Heap::is_dead_small_cell`] and [`Heap::is_dead_target`], which in a minor collection
    /// find only young cells dead. Returns false when deferred, in which case
    /// `finish_collection` must not be called; a deferred major request makes a major
    /// collection due.
    pub(crate) fn start_marking(
        &mut self,
        native_roots: bool,
        client: &dyn MarkClient,
        kind: CollectionKind,
    ) -> bool {
        if self.blocked() {
            return self.defer(false, kind);
        }
        if native_roots && !native::scannable(native::current_sp()) {
            return self.defer(true, kind);
        }
        self.mark_phase(native_roots, client, kind);
        true
    }

    /// Phase 2: finalize the newly dead cells and free them, except those that an address in
    /// `pinned` starts, which a weak handle still names: they become zombies until a later
    /// collection finds them unpinned. Each pin is recorded in its cell's page (a flag on a
    /// large cell) for the sweep, which reads it where it meets the cell and clears it. An
    /// address that does not start an allocated cell (one inside a cell, a free cell's, or one
    /// outside the heap) records nothing.
    pub(crate) fn finish_collection(&mut self, pinned: &[usize]) {
        assert!(
            self.control.collecting.get(),
            "finish_collection without start_marking"
        );
        for &addr in pinned {
            self.record_pin(addr);
        }
        self.sweep_phase();
    }

    /// Pin the allocated cell that starts at `addr`, if one does, for this collection's sweep
    /// (see [`Heap::finish_collection`]). A minor collection's sweep leaves the pages that
    /// received no cell since the last collection alone, and their cells with them (a dead one
    /// there is a zombie already), and the old large cells likewise, so it records no pin
    /// there: none would be cleared.
    fn record_pin(&mut self, addr: usize) {
        let Some(cell) = self.locate(addr as u64).filter(|cell| cell.start == addr) else {
            return;
        };
        let minor = self.collecting_kind == CollectionKind::Minor;
        match cell.loc {
            Loc::Small { page, index } => {
                let page = self.pages[page as usize]
                    .as_mut()
                    .expect("a located page is live");
                if bit(&page.allocated, index as usize) && (page.young || !minor) {
                    set_bit(&mut page.pins, index as usize, true);
                }
            }
            Loc::Large => {
                let large = self
                    .larges
                    .get_mut(&addr)
                    .expect("a located large cell is allocated");
                if large.young || !minor {
                    large.pinned = true;
                }
            }
        }
    }

    /// Debug builds: a reader of the current marking's verdict is between a collection's phases.
    /// Marks persist between collections, where they say which cells are old, not which are
    /// dead.
    fn debug_assert_marking_current(&self, reader: &str) {
        debug_assert!(
            self.control.collecting.get(),
            "{reader} read the marks outside a collection, where they say which cells are old"
        );
    }

    /// Call `f` with each page the collection in progress sweeps: a minor collection's young
    /// pages, or every page.
    fn for_each_swept_page(&self, mut f: impl FnMut(&PageMeta)) {
        match self.collecting_kind {
            CollectionKind::Minor => {
                for &index in &self.young_pages {
                    if let Some(page) = self.pages[index as usize].as_ref() {
                        f(page);
                    }
                }
            }
            CollectionKind::Major => self.pages.iter().flatten().for_each(f),
        }
    }

    /// The allocated cells of `desc` the current marking left unmarked, zombies apart (between
    /// [`Heap::start_marking`] and [`Heap::finish_collection`]). A minor collection lists the
    /// young pages' only (and the large cells'): every other page's allocated cells are old,
    /// marked, or zombies.
    pub(crate) fn dead_cells(&self, desc: DescId) -> Vec<usize> {
        self.debug_assert_marking_current("dead_cells");
        let mut dead = Vec::new();
        self.for_each_dead(desc, |page, mut bits, word| {
            while bits != 0 {
                dead.push(page.cell_ptr(word * 64 + bits.trailing_zeros() as usize));
                bits &= bits - 1;
            }
        });
        self.for_each_dead_large(desc, |start| dead.push(start));
        dead
    }

    /// How many cells [`Heap::dead_cells`] would list, from the bits alone.
    pub(crate) fn dead_count(&self, desc: DescId) -> usize {
        self.debug_assert_marking_current("dead_count");
        let mut count = 0;
        self.for_each_dead(desc, |_, bits, _| count += bits.count_ones() as usize);
        self.for_each_dead_large(desc, |_| count += 1);
        count
    }

    /// Call `f` with each word of dead-cell bits (allocated, unmarked, not zombies) of `desc`'s
    /// pages the collection in progress sweeps, and the word's index in its page.
    fn for_each_dead(&self, desc: DescId, mut f: impl FnMut(&PageMeta, u64, usize)) {
        self.for_each_swept_page(|page| {
            if page.desc != desc {
                return;
            }
            // Only the prefix below `next_unused` holds allocated cells.
            for word in 0..page.next_unused.div_ceil(64) {
                let bits = page.allocated[word] & !page.marks[word] & !page.zombies[word];
                if bits != 0 {
                    f(page, bits, word);
                }
            }
        });
    }

    /// Call `f` with each dead large cell of `desc` among those the collection in progress
    /// sweeps: a minor collection's young ones, or all.
    fn for_each_dead_large(&self, desc: DescId, mut f: impl FnMut(usize)) {
        let mut visit = |start: usize, large: &LargeObj| {
            if large.desc == desc && !large.marked && !large.zombie {
                f(start);
            }
        };
        match self.collecting_kind {
            CollectionKind::Minor => {
                for start in &self.young_larges {
                    if let Some(large) = self.larges.get(start) {
                        visit(*start, large);
                    }
                }
            }
            CollectionKind::Major => {
                for (&start, large) in &self.larges {
                    visit(start, large);
                }
            }
        }
    }

    /// Whether `addr` is the start of a small cell of `desc` that the current marking left
    /// unmarked, zombies apart: between [`Heap::start_marking`] and [`Heap::finish_collection`],
    /// whether [`Heap::dead_cells`] lists `addr` for a descriptor whose cells are all small.
    /// The page table and the page's bits answer, without hashing; large cells are not
    /// consulted, so a dead large cell's start answers false. An address in a page's tail or
    /// card table starts no cell: the bound on the index rejects it.
    pub(crate) fn is_dead_small_cell(&self, addr: usize, desc: DescId) -> bool {
        self.debug_assert_marking_current("is_dead_small_cell");
        let Some(slot) = self.page_table.get(addr >> PAGE_SHIFT) else {
            return false;
        };
        let Some(page) = self.pages[slot as usize].as_ref() else {
            return false;
        };
        let index = (addr - page.base) / page.cell_size;
        page.desc == desc
            && page.cell_ptr(index) == addr
            && index < page.cells
            && bit(&page.allocated, index)
            && !bit(&page.marks, index)
            && !bit(&page.zombies, index)
    }

    /// The number of the collection in progress (the first is 1), while one is: for an
    /// embedder's, from [`Heap::start_marking`] until [`Heap::finish_collection`] has swept.
    #[cfg(debug_assertions)]
    pub(crate) fn collection_in_progress(&self) -> Option<u64> {
        self.control
            .collecting
            .get()
            .then_some(self.stats.collections)
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

    /// What the current marking says of the cell `addr` lies in, for a weak reference to it:
    /// marked (live), unmarked (dead: clear the reference), or no cell of this heap. Between
    /// the phases of a collection.
    pub(crate) fn weak_target_state(&self, addr: usize) -> crate::gc_handle::TargetState {
        use crate::gc_handle::TargetState;
        self.debug_assert_marking_current("weak_target_state");
        match self.locate(addr as u64) {
            None => TargetState::Elsewhere,
            Some(cell) if self.is_marked(&cell) => TargetState::Live,
            Some(_) => TargetState::Dead,
        }
    }

    /// Whether `addr` (interior addresses included) lies in a cell of this heap that the
    /// current marking left unmarked: a weak reference to it must be cleared. Between the phases
    /// of a collection.
    pub(crate) fn is_dead_target(&self, addr: usize) -> bool {
        self.debug_assert_marking_current("is_dead_target");
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

    /// Defer a requested collection of `kind`: it is pending until the next safepoint, and a
    /// major request makes a major collection due.
    /// Publish an Agent-level request without collecting at a non-safepoint. In particular a
    /// suppressed major must remain due after the suppression/byte loan ends.
    pub(crate) fn request_collection(&mut self, kind: CollectionKind) {
        self.defer(false, kind);
    }

    fn defer(&mut self, native_stack: bool, kind: CollectionKind) -> bool {
        // With a request pending the due word is set already.
        self.major_due |= kind == CollectionKind::Major;
        if !self.control.pending_collect.replace(true) {
            self.stats.deferred_collections += 1;
            if native_stack {
                self.stats.deferred_native_stack += 1;
            }
            self.refresh_due();
        }
        false
    }

    fn mark_phase(&mut self, native_roots: bool, client: &dyn MarkClient, kind: CollectionKind) {
        assert!(
            !self.control.collecting.replace(true),
            "re-entrant GC collection"
        );
        self.control.markings.set(self.control.markings.get() + 1);
        self.update_fast_alloc();
        self.control.pending_collect.set(false);
        self.collecting_kind = kind;
        if kind == CollectionKind::Major {
            self.major_due = false;
            self.major_requested = false;
        } else {
            self.stats.minor_collections += 1;
        }
        if self.stress_every > 0 {
            self.stress_collections += 1;
        }
        self.marked_bytes = 0;
        self.external_at_mark = self.control.external.live.get();
        self.external_due_at_mark = self.control.external.due.get();
        self.refresh_due();
        self.stats.collections += 1;
        self.stats.last_phase_nanos = [0; PHASES];
        self.stats.last_finalized = [0; COUNTED_DESCS];
        let mut clock = self.phase_clock();
        self.flush_cursors();
        if self.pacing.is_some() {
            if kind == CollectionKind::Major {
                self.committed_at_mark = self.committed_bytes();
            }
            let allocated = self.allocated_bytes.saturating_sub(self.swept_bytes);
            if let Some(pacing) = &mut self.pacing {
                pacing.start(std::time::Instant::now(), allocated);
            }
        }
        self.begin_marking(kind, client);
        // `begin_marking` handled cells' and pages' addresses in frames below this one, which the
        // native stack scan's frames reuse: zero that region first, as the engine zeroes its own
        // before a collection.
        #[cfg(not(target_arch = "wasm32"))]
        if native_roots {
            scrub_stack(BEGIN_MARKING_SCRUB_BYTES);
        }
        let mut m = Marker::default();
        #[cfg(test)]
        {
            m.parents = self.retention_tracing.then(FastMap::default);
        }
        // A minor collection traces the old cells whose cards were set, although they are
        // marked: they may hold the only edges to young cells.
        m.worklist.append(&mut self.card_cells);
        self.phase_done(Phase::Begin, &mut clock);
        client.roots(&mut |word| self.mark_word(word, &mut m));
        self.phase_done(Phase::Roots, &mut clock);
        if native_roots {
            // The roots' marking handled the heap's metadata (the large cells' span starts at a
            // cell's address, which may be a dead one's) in frames below this one, which the
            // stack scan's frame reuses: zero them first, as after `begin_marking`.
            #[cfg(not(target_arch = "wasm32"))]
            scrub_stack(BEGIN_MARKING_SCRUB_BYTES);
            self.mark_native_stack(&mut m);
        }
        self.phase_done(Phase::Stack, &mut clock);
        #[cfg(test)]
        {
            m.source = MarkSource::Root;
        }
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
                },
                &mut m,
                client,
            );
        });
        self.phase_done(Phase::Side, &mut clock);
        self.drain(&mut m, client);
        self.phase_done(Phase::Drain, &mut clock);
        if self.verifies() {
            self.verify(&m);
        }
        #[cfg(test)]
        if let Some(parents) = m.parents.take() {
            self.last_parents = parents;
        }
    }

    /// As a collection starts: with heap verification, check the write barrier's invariant
    /// against the last collection's marks and the cards set since
    /// ([`Heap::check_write_barrier`]), the young pages' bookkeeping
    /// ([`Heap::check_young_pages`]) and the client's records of what changed since
    /// ([`MarkClient::check_records`]). Then a major collection clears every mark
    /// and every card, so that marking starts from nothing; a minor one gathers the old cells
    /// whose cards are set into `card_cells` (S6 step 10c) and clears those cards. Either way
    /// the next collection's cards record only what is written from now on. Out of line, so
    /// that `mark_phase` can zero the frames it used (see there).
    #[inline(never)]
    fn begin_marking(&mut self, kind: CollectionKind, client: &dyn MarkClient) {
        if self.verifies() {
            self.check_write_barrier();
            self.check_young_pages();
            client.check_records(self);
        }
        match kind {
            CollectionKind::Major => {
                for page in self.pages.iter_mut().flatten() {
                    page.marks.fill(0);
                    page.clear_cards();
                }
                for large in self.larges.values_mut() {
                    large.marked = false;
                }
                self.stats.last_card_cells = 0;
            }
            CollectionKind::Minor => {
                let mut cells = std::mem::take(&mut self.card_cells);
                cells.clear();
                for page in self.pages.iter_mut().flatten() {
                    page.take_card_cells(&mut cells);
                }
                self.stats.last_card_cells = cells.len();
                self.card_cells = cells;
            }
        }
    }

    /// The young pages' bookkeeping, checked by heap verification as each collection starts:
    /// the list names exactly the pages flagged young, each once, and every other page's
    /// allocated cells are old (marked) or zombies. A minor collection sweeps the young pages
    /// alone, and lists the dead there, so it relies on the second.
    fn check_young_pages(&self) {
        let flagged = self
            .pages
            .iter()
            .flatten()
            .filter(|page| page.young)
            .count();
        let listed: crate::fasthash::FastSet<u32> = self.young_pages.iter().copied().collect();
        assert!(
            listed.len() == self.young_pages.len() && listed.len() == flagged,
            "young pages: the list ({} entries, {} distinct) and the pages' flags ({flagged}) \
             disagree",
            self.young_pages.len(),
            listed.len()
        );
        for &index in &self.young_pages {
            assert!(
                self.pages[index as usize]
                    .as_ref()
                    .is_some_and(|page| page.young),
                "young pages: slot {index} is listed but not a young page"
            );
        }
        for page in self.pages.iter().flatten().filter(|page| !page.young) {
            let young = page
                .allocated
                .iter()
                .zip(page.marks.iter())
                .zip(&page.zombies)
                .any(|((&allocated, &marks), &zombies)| allocated & !marks & !zombies != 0);
            assert!(
                !young,
                "young pages: the page at {:#x} holds a young cell but is not listed",
                page.base
            );
        }
    }

    /// The write barrier's invariant, checked by heap verification as each collection starts,
    /// before it clears any mark or card: an old cell (one a collection marked) whose card is
    /// clean holds no traced edge to an allocated, unmarked cell of this heap (one allocated since
    /// the last collection). A cell of a kind without cards, or a large cell, has no card: it
    /// never holds such an edge. Every verifying collection runs it, minor and major, so GC
    /// stress with verification runs it at every safepoint. Minor collections, which trace an
    /// old cell only through its set card, rely on it. Panics naming the holder's descriptor and
    /// address and the target's.
    fn check_write_barrier(&self) {
        let mut edges: Vec<*const u8> = Vec::new();
        let mut check = |holder: usize, desc: DescId| {
            let desc = &self.descs[desc.0 as usize];
            edges.clear();
            (desc.trace)(holder as *const u8, &mut |edge| edges.push(edge));
            for &edge in &edges {
                let Some(target) = self.locate(edge as u64) else {
                    continue;
                };
                let unmarked = match target.loc {
                    Loc::Small { page, index } => {
                        self.pages[page as usize].as_ref().is_some_and(|p| {
                            bit(&p.allocated, index as usize) && !bit(&p.marks, index as usize)
                        })
                    }
                    Loc::Large => self.larges.get(&target.start).is_some_and(|l| !l.marked),
                };
                assert!(
                    !unmarked,
                    "write barrier: old cell {holder:#x} ({}) with a clean card holds an edge to \
                     unmarked cell {:#x} ({}), allocated since the last collection",
                    desc.name, target.start, self.descs[target.desc.0 as usize].name
                );
            }
        };
        for page in self.pages.iter().flatten() {
            for i in 0..page.cells {
                if bit(&page.marks, i) && !page.card_set(i) {
                    check(page.cell_ptr(i), page.desc);
                }
            }
            // A minor collection reads a page's cards only when its summary is set.
            if page.carded && (0..page.cells).any(|i| page.card_set(i)) {
                // SAFETY: a carded page ends in its card table, the summary its last byte.
                let summary = unsafe { ((page.base + CARD_SUMMARY) as *const u8).read() };
                assert!(
                    summary != 0,
                    "write barrier: the page at {:#x} has a set card but a clear summary",
                    page.base
                );
            }
        }
        for (&start, large) in &self.larges {
            if large.marked {
                check(start, large.desc);
            }
        }
    }

    fn sweep_phase(&mut self) {
        let kind = self.collecting_kind;
        let freed = self.sweep(kind);
        if self.verifies() {
            if let Some((desc, count)) = self.live_count {
                // SAFETY: `attach_live_count`'s contract.
                assert_eq!(
                    unsafe { count.as_ref() }.get(),
                    self.live_count(desc) as i64,
                    "GC live-cell accounting"
                );
            }
        }
        // What the heap holds now: its live cells, zombies and, after a minor collection, the old
        // cells that died since a collection last marked them.
        let held = self.allocated_bytes.saturating_sub(freed);
        if cfg!(debug_assertions) {
            assert_eq!(held, self.held_bytes(), "GC byte accounting");
        }
        self.allocated_bytes = held;
        self.swept_bytes = held;
        self.stats.last_marked_bytes = self.marked_bytes;
        // The cells' payloads outside the heap count with the cells (see `ExternalBytes`).
        let external = &self.control.external;
        let external_live = external.live.get();
        let size = held.saturating_add(external_live);
        // With pacing: V8's growing factor for a major collection's budget, and its nursery.
        let ended = self.pacing.is_some().then(std::time::Instant::now);
        let reducing = self
            .pacing
            .as_ref()
            .is_some_and(|pacing| pacing.reducing_memory());
        let factor = match (&mut self.pacing, ended) {
            (Some(pacing), Some(now)) => {
                let factor = pacing.end(now, kind == CollectionKind::Major, self.marked_bytes);
                self.nursery_bytes = pacing.nursery();
                factor
            }
            _ => None,
        };
        match kind {
            CollectionKind::Major => {
                // V8's limit adds the young generation's capacity, which the next minor
                // collection may promote.
                let budget = match factor {
                    Some(factor) => ((size as f64 * factor) as usize)
                        .max(size.saturating_add(crate::gc_pacing::MIN_GROWING_STEP))
                        .saturating_add(self.nursery_bytes.unwrap_or(0)),
                    None => size.saturating_add(size.saturating_mul(self.growth_percent) / 100),
                };
                self.trigger_bytes = self.min_trigger_bytes.max(budget);
            }
            // The old cells, live or not, and the payloads still charged have passed the byte
            // budget: a major collection is due, at the next safepoint.
            CollectionKind::Minor => self.major_due |= size > self.trigger_bytes,
        }
        // The next external budget. A collection the external charges made due, whose sweep
        // released less than a quarter of what was charged since the last one, found those bytes
        // live (a growing collection, say): collecting again at the same rate would find them
        // live again, so the budget doubles, up to the byte budget, past which the bytes count
        // toward a major collection instead. Otherwise it is the nursery's (or, without a
        // nursery, the byte budget).
        let charged = external.since.replace(0);
        let released = self.external_at_mark.saturating_sub(external_live);
        external.due.set(false);
        let base = self.nursery_bytes.unwrap_or(self.trigger_bytes).max(1);
        let budget = if self.external_due_at_mark && released.saturating_mul(4) < charged {
            external
                .budget
                .get()
                .saturating_mul(2)
                .min(self.trigger_bytes)
                .max(base)
        } else {
            base
        };
        external.budget.set(budget);
        self.trim_spare_pages(reducing);
        if let Some(now) = ended {
            match kind {
                CollectionKind::Major => {
                    let (before, used) = (self.committed_at_mark, self.swept_bytes);
                    let committed = self.committed_bytes();
                    self.bootstrap_committed = None;
                    if let Some(pacing) = &mut self.pacing {
                        pacing.reducer.mark_compact(now, before, committed, used);
                    }
                }
                // V8's `NotifyOldGenerationExpansion`: the heap grew 1 MB past its bootstrap with
                // no major collection yet (a major one ends this check).
                CollectionKind::Minor => {
                    if let Some(bootstrap) = self.bootstrap_committed {
                        if self.committed_bytes() >= bootstrap + (1 << 20) {
                            self.bootstrap_committed = None;
                            if let Some(pacing) = &mut self.pacing {
                                pacing.reducer.possible_garbage(now);
                            }
                        }
                    }
                }
            }
        }
        self.allocs_since_gc = 0;
        self.control.collecting.set(false);
        self.update_fast_alloc();
        // New budgets and interval; a request deferred meanwhile keeps the word set.
        self.refresh_due();
    }

    /// The bytes of every allocated cell, live or dead, zombies included: what `allocated_bytes`
    /// counts once a sweep has set it (debug builds check the accounting after every sweep).
    fn held_bytes(&self) -> usize {
        self.pages
            .iter()
            .flatten()
            .map(|page| page.live * page.cell_size)
            .sum::<usize>()
            + self.larges.values().map(|large| large.size).sum::<usize>()
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
        #[cfg(all(test, target_arch = "aarch64"))]
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
            self.mark_stack_range(lo, hi, m);
        }
        std::hint::black_box(&regs);
    }

    /// Mark the cells the words in `lo..hi`, native stack at and above the scan's own frame,
    /// name. Out of line, so that what marking derives from each word and from the heap's
    /// metadata (the large cells' span, which starts at a cell's address, among them) lives in
    /// this frame, below the words it reads, and never among them.
    #[inline(never)]
    fn mark_stack_range(&mut self, lo: usize, hi: usize, m: &mut Marker) {
        let mut p = lo;
        while p + 8 <= hi {
            // SAFETY: registered stack regions are mapped (enter_native_stack
            // and grow contracts); volatile keeps the compiler from reasoning
            // about reads of other frames.
            let word = unsafe { std::ptr::read_volatile(p as *const u64) };
            #[cfg(test)]
            {
                m.source = MarkSource::Stack(p);
            }
            self.mark_word(word, m);
            p += 8;
        }
    }

    fn add_source_edges(&mut self, m: &mut Marker, client: &dyn MarkClient) {
        let mut reported = Vec::new();
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
                },
                m,
                client,
            );
        }
    }

    fn drain(&mut self, m: &mut Marker, client: &dyn MarkClient) {
        let mut ahead = [TraceWork {
            cell: 0,
            desc: DescId(0),
            purpose: TracePurpose::Full,
        }; MARK_PREFETCH];
        let (mut head, mut queued) = (0usize, 0usize);
        loop {
            loop {
                while queued < MARK_PREFETCH {
                    let Some(work) = m.worklist.pop() else {
                        break;
                    };
                    prefetch_cell(work.cell);
                    ahead[(head + queued) % MARK_PREFETCH] = work;
                    queued += 1;
                }
                if queued == 0 {
                    break;
                }
                let TraceWork {
                    cell,
                    desc,
                    purpose,
                } = ahead[head];
                head = (head + 1) % MARK_PREFETCH;
                queued -= 1;
                #[cfg(test)]
                {
                    m.source = MarkSource::Cell(cell);
                }
                let descriptor = &self.descs[desc.0 as usize];
                let (trace, mark_trace) = (descriptor.trace, descriptor.mark_trace);
                // Each edge is marked as the trace reports it.
                match mark_trace {
                    Some(mark_trace) => {
                        mark_trace(cell as *const u8, purpose, &mut MarkSink { heap: self, m })
                    }
                    None => trace(cell as *const u8, &mut |edge| {
                        self.mark_word(edge as u64, m)
                    }),
                }
            }
            let Some(edge) = m.ready.pop() else {
                break;
            };
            self.try_fire(edge, m, client);
        }
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
        #[cfg(test)]
        let saved = m.source;
        match value {
            EdgeValue::Word(word) => {
                #[cfg(test)]
                {
                    m.source = MarkSource::Conditional {
                        owner: conditions[0].map_or(0, |cell| cell.start),
                    };
                }
                self.mark_word(word, m);
            }
            EdgeValue::SideSlots(owner) => {
                #[cfg(test)]
                {
                    m.source = MarkSource::SideSlots(owner);
                }
                client.trace_side_slots(owner, &mut |word| self.mark_word(word, m));
            }
        }
        #[cfg(test)]
        {
            m.source = saved;
        }
    }

    fn wait_on(&mut self, cell: &CellRef, edge: u32, m: &mut Marker) {
        match cell.loc {
            Loc::Small { page, index } => {
                if let Some(page) = self.pages[page as usize].as_mut() {
                    set_bit(&mut page.waiters, index as usize, true);
                    page.has_waiters = true;
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

    /// Finalize, poison (debug) and free the unmarked cells of the pages a collection of `kind`
    /// sweeps (a minor collection's young pages, or every page) and of the large cells, release
    /// emptied pages, clear waiters and pins, and clear the young pages' flags and list. Dead
    /// cells pinned for weak handles (see `PageMeta::pins`) stay allocated as zombies. The marks
    /// stay set: until the next collection starts they tell old cells from young ones (see
    /// `PageMeta::marks`). Returns the bytes freed.
    ///
    /// Every finalizer runs before any dead cell is poisoned, linked or released: a finalizer
    /// may read leaf cells that die with its cell, such as the storage blocks of an object's
    /// payloads (`gc_storage`), whatever order the pages are swept in.
    fn sweep(&mut self, kind: CollectionKind) -> usize {
        let mut clock = self.phase_clock();
        // Allocation between the phases moved the cursors on.
        self.flush_cursors();
        let young = std::mem::take(&mut self.young_pages);
        match kind {
            CollectionKind::Minor => {
                for &index in &young {
                    self.finalize_page(index as usize);
                }
            }
            CollectionKind::Major => {
                for index in 0..self.pages.len() {
                    self.finalize_page(index);
                }
            }
        }
        self.finalize_larges(kind);
        self.phase_done(Phase::Finalize, &mut clock);
        let mut freed = 0usize;
        let mut swept_pages = 0usize;
        let mut sweep = |heap: &mut Self, index: usize| {
            if heap.pages[index].is_some() {
                swept_pages += 1;
                freed += heap.sweep_page(index);
            }
        };
        match kind {
            CollectionKind::Minor => {
                for &index in &young {
                    sweep(self, index as usize);
                }
            }
            CollectionKind::Major => {
                for index in 0..self.pages.len() {
                    sweep(self, index);
                }
            }
        }
        self.stats.last_pages_swept = swept_pages;
        // The cursors cached the swept pages' bits; their pages go back to the lists below.
        self.reset_cursors();
        // Every page is old until it receives a cell again (pages released above are gone).
        for &index in &young {
            if let Some(page) = self.pages[index as usize].as_mut() {
                page.young = false;
            }
        }
        match kind {
            // The old pages were left alone: those with free cells are on their classes' lists
            // still (or current). A young page was taken from its list (or made) since, so it
            // is on none: put back those that have free cells now.
            CollectionKind::Minor => {
                for &index in &young {
                    let Some(page) = self.pages[index as usize].as_ref() else {
                        continue;
                    };
                    let class = &mut self.classes[page.desc.0 as usize][page.class];
                    if page.has_room() && class.current != Some(index) {
                        debug_assert!(!class.avail.contains(&index), "a young page on its list");
                        class.avail.push(index);
                    }
                }
            }
            CollectionKind::Major => {
                for class in self.classes.iter_mut().flatten() {
                    class.avail.clear();
                }
                for (index, page) in self.pages.iter().enumerate() {
                    let Some(page) = page else {
                        continue;
                    };
                    let class = &mut self.classes[page.desc.0 as usize][page.class];
                    if page.has_room() && class.current != Some(index as u32) {
                        class.avail.push(index as u32);
                    }
                }
            }
        }
        // The list's buffer is kept for the next cycle's young pages.
        self.young_pages = young;
        self.young_pages.clear();
        let mut swept = 0u64;
        let live_count = self.live_count;
        let mut counted_dead = 0i64;
        // Sweep one large cell; false once its memory is freed.
        let mut sweep_large = |start: usize, large: &mut LargeObj| -> bool {
            large.young = false;
            let pinned = std::mem::take(&mut large.pinned);
            if large.marked {
                large.waiters = false;
                return true;
            }
            large.waiters = false;
            if !large.zombie {
                // Finalized already (`Heap::finalize_larges`).
                swept += 1;
                if live_count.is_some_and(|(desc, _)| desc == large.desc) {
                    counted_dead += 1;
                }
                #[cfg(debug_assertions)]
                // SAFETY: the dead object spans `size` bytes from `start`.
                unsafe {
                    std::ptr::write_bytes(start as *mut u8, 0xdd, large.size)
                };
            }
            if pinned {
                large.zombie = true;
                return true;
            }
            freed += large.size;
            // SAFETY: allocated in alloc_large with this layout.
            unsafe { dealloc(start as *mut u8, large_layout(large.size)) };
            false
        };
        let young_larges = std::mem::take(&mut self.young_larges);
        match kind {
            // The old large cells stay marked, live or not, until a major collection, as the
            // old pages' cells do; a dead one there is a zombie already.
            CollectionKind::Minor => {
                for &start in &young_larges {
                    let Some(large) = self.larges.get_mut(&start) else {
                        continue;
                    };
                    if !sweep_large(start, large) {
                        self.larges.remove(&start);
                    }
                }
            }
            CollectionKind::Major => self
                .larges
                .retain(|&start, large| sweep_large(start, large)),
        }
        // The list's buffer is kept for the next cycle's large cells.
        self.young_larges = young_larges;
        self.young_larges.clear();
        if let Some((_, count)) = live_count {
            // SAFETY: `attach_live_count`'s contract.
            let count = unsafe { count.as_ref() };
            count.set(count.get() - counted_dead);
        }
        self.large_span = match (self.larges.first_key_value(), self.larges.last_key_value()) {
            (Some((&lo, _)), Some((&last, large))) => (lo, last + large.size),
            _ => (usize::MAX, 0),
        };
        self.stats.cells_swept += swept;
        if cfg!(debug_assertions) {
            self.check_page_table();
        }
        self.phase_done(Phase::Free, &mut clock);
        freed
    }

    /// Run the finalizer of every newly dead cell of the page in slot `index`: the unmarked
    /// allocated cells that are not zombies (finalized before). The sweep's first pass; nothing
    /// is freed yet (see [`Heap::sweep`]).
    fn finalize_page(&mut self, index: usize) {
        let Some(page) = self.pages[index].as_ref() else {
            return;
        };
        let Some(finalize) = self.descs[page.desc.0 as usize].finalize else {
            return;
        };
        let mut finalized = 0u32;
        for word in 0..page.next_unused.div_ceil(64) {
            let used = (page.next_unused - word * 64).min(64);
            let mut candidates = page.allocated[word] & !page.marks[word] & !page.zombies[word];
            candidates &= u64::MAX >> (64 - used);
            finalized += candidates.count_ones();
            while candidates != 0 {
                let i = word * 64 + candidates.trailing_zeros() as usize;
                candidates &= candidates - 1;
                finalize(page.cell_ptr(i) as *mut u8);
            }
        }
        let desc = page.desc.0 as usize;
        if self.phase_timing && desc < COUNTED_DESCS {
            self.stats.last_finalized[desc] += finalized;
        }
    }

    /// The first pass's large cells: finalize the newly dead ones among those the collection of
    /// `kind` sweeps (a minor collection's young ones, or all).
    fn finalize_larges(&self, kind: CollectionKind) {
        let finalize = |start: usize, large: &LargeObj| {
            if !large.marked && !large.zombie {
                if let Some(finalize) = self.descs[large.desc.0 as usize].finalize {
                    finalize(start as *mut u8);
                }
            }
        };
        match kind {
            CollectionKind::Minor => {
                for start in &self.young_larges {
                    if let Some(large) = self.larges.get(start) {
                        finalize(*start, large);
                    }
                }
            }
            CollectionKind::Major => {
                for (&start, large) in &self.larges {
                    finalize(start, large);
                }
            }
        }
    }

    /// Sweep the page in slot `index` (see [`Heap::sweep`]), releasing it if no cell is left.
    /// Returns the bytes freed.
    fn sweep_page(&mut self, index: usize) -> usize {
        let Some(page) = self.pages[index].as_mut() else {
            return 0;
        };
        let mut freed = 0usize;
        let mut swept = 0u64;
        debug_assert!(page.next_unused <= page.cells);
        for word in 0..page.next_unused.div_ceil(64) {
            // Include unmarked zombies: an unpinned zombie frees storage without another
            // finalization/death. Existing holes and the untouched virgin tail are excluded.
            let used = (page.next_unused - word * 64).min(64);
            let dead = page.allocated[word] & !page.marks[word] & (u64::MAX >> (64 - used));
            if dead == 0 {
                continue;
            }
            // Finalized already (`Heap::finalize_page`); unmarked zombies died before.
            let newly_dead = dead & !page.zombies[word];
            swept += u64::from(newly_dead.count_ones());
            #[cfg(debug_assertions)]
            {
                let mut poison = newly_dead;
                while poison != 0 {
                    let i = word * 64 + poison.trailing_zeros() as usize;
                    poison &= poison - 1;
                    let cell = page.cell_ptr(i) as *mut u64;
                    for w in 0..page.cell_size / 8 {
                        // SAFETY: the dead cell lies inside this page.
                        unsafe { std::ptr::write(cell.add(w), POISON) };
                    }
                }
            }
            // A dead cell a weak handle pins stays allocated as a zombie: finalized, its address
            // reserved. The others are free: a clear bit, which allocation finds without reading
            // the cell (see `PageMeta::free`).
            let pinned = dead & page.pins[word];
            let reclaimed = dead & !pinned;
            page.zombies[word] = (page.zombies[word] & !dead) | pinned;
            page.allocated[word] &= !reclaimed;
            let count = reclaimed.count_ones() as usize;
            page.live -= count;
            freed += count * page.cell_size;
        }
        page.reset_cursor();
        page.has_zombies = page.zombies.iter().any(|&word| word != 0);
        // These cover the full bitmap, including live cells and zero-death sweeps: a pin
        // observed for a live cell must not keep that cell pinned when it later dies.
        page.waiters.fill(0);
        page.has_waiters = false;
        page.pins.fill(0);
        self.stats.cells_swept += swept;
        if let Some((desc, count)) = self.live_count {
            if page.desc == desc {
                // Zombies were counted only on their first sweep, when finalized.
                // SAFETY: `attach_live_count`'s contract.
                let count = unsafe { count.as_ref() };
                count.set(count.get() - swept as i64);
            }
        }
        if page.live == 0 {
            self.release_page(index);
        }
        freed
    }

    /// Debug check, after the pages change in bulk: the page table maps every live page's
    /// number to its slot and holds no other entry.
    fn check_page_table(&self) {
        let mut live = 0;
        assert_eq!(self.mark_pages.len(), self.pages.len(), "GC mark pages");
        for (slot, page) in self.pages.iter().enumerate() {
            let mark_page = self.mark_pages[slot];
            let Some(page) = page else {
                assert_eq!(mark_page.cells, 0, "GC mark pages: a free slot's");
                continue;
            };
            live += 1;
            assert_eq!(
                self.page_table.get(page.base >> PAGE_SHIFT),
                Some(slot as u32),
                "GC page table: the page at {:#x} does not map to its slot",
                page.base
            );
            let expected = MarkPage::of(page);
            assert!(
                mark_page.marks == expected.marks
                    && mark_page.reciprocal == expected.reciprocal
                    && mark_page.cells == expected.cells,
                "GC mark pages: the page at {:#x}'s",
                page.base
            );
        }
        assert_eq!(
            self.page_table.len(),
            live,
            "GC page table: entries for pages that are not live"
        );
    }

    fn release_page(&mut self, index: usize) {
        let page = self.pages[index].take().expect("live page slot");
        self.mark_pages[index] = MarkPage::FREE;
        self.page_table.remove(page.base >> PAGE_SHIFT);
        let class = &mut self.classes[page.desc.0 as usize][page.class];
        if class.current == Some(index as u32) {
            class.current = None;
        }
        self.free_page_slots.push(index as u32);
        // Kept for now; the sweep trims the pool once it has set the next cycle's budget.
        self.spare_pages.push(page.base);
    }

    /// How many released pages to keep: as many as allocation until the next collection can
    /// fill. With a nursery that is its budget, and at least 16 pages; without one, the byte
    /// budget's room, and at least 64. A heap that empties and refills its pages each cycle
    /// takes them back from the pool, and one that shrank returns what its smaller budget will
    /// not use.
    fn spare_page_cap(&self) -> usize {
        match self.nursery_bytes {
            Some(nursery) => nursery
                .div_ceil(PAGE_SIZE)
                .max(MIN_SPARE_PAGES_WITH_NURSERY),
            None => {
                let budget = self.trigger_bytes.saturating_sub(self.allocated_bytes);
                budget.div_ceil(PAGE_SIZE).max(MIN_SPARE_PAGES)
            }
        }
    }

    /// Give the spare pages beyond the cap back to the page source, once the sweep has released
    /// its empty pages and set the next budget. (A cap checked as each page went would shrink
    /// with the pages still to go, and keep about half of a large heap that emptied at once.)
    /// A collection that reduces memory keeps only the minimum.
    fn trim_spare_pages(&mut self, reducing: bool) {
        let cap = if reducing {
            MIN_SPARE_PAGES_WITH_NURSERY
        } else {
            self.spare_page_cap()
        };
        while self.spare_pages.len() > cap {
            let base = self
                .spare_pages
                .pop()
                .expect("more spare pages than the cap");
            // SAFETY: a spare page is a page of the source, from new_page; no cell remains.
            unsafe { self.page_source.give_back(base as *mut u8) };
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
        let pages = std::mem::take(&mut self.pages);
        self.mark_pages.clear();
        for page in pages.iter().flatten() {
            if let Some(finalize) = self.descs[page.desc.0 as usize].finalize {
                for i in 0..page.cells {
                    if bit(&page.allocated, i) && !bit(&page.zombies, i) {
                        finalize(page.cell_ptr(i) as *mut u8);
                    }
                }
            }
        }
        let larges = std::mem::take(&mut self.larges);
        // Every cell is finalized before any is freed (see `Heap::sweep`).
        for (&start, large) in &larges {
            if let Some(finalize) = self.descs[large.desc.0 as usize].finalize {
                if !large.zombie {
                    finalize(start as *mut u8);
                }
            }
        }
        for (start, large) in larges {
            // SAFETY: allocated in alloc_large with this layout.
            unsafe { dealloc(start as *mut u8, large_layout(large.size)) };
        }
        self.young_larges.clear();
        let spare = std::mem::take(&mut self.spare_pages);
        // SAFETY: the live pages and the spare ones are all the pages the source gave and has not
        // taken back; their cells are finalized, and nothing uses them any more.
        unsafe {
            self.page_source
                .release(pages.iter().flatten().map(|page| page.base).chain(spare))
        };
        self.free_page_slots.clear();
        self.page_table.clear();
        self.young_pages.clear();
        self.cursors.fill(Cursor::EMPTY);
        self.card_cells.clear();
        for class in self.classes.iter_mut().flatten() {
            *class = ClassPages::default();
        }
        self.large_span = (usize::MAX, 0);
        self.allocated_bytes = 0;
        if let Some((_, count)) = self.live_count {
            // SAFETY: `attach_live_count`'s contract.
            unsafe { count.as_ref() }.set(0);
        }
        self.swept_bytes = 0;
        self.major_due = false;
        self.major_requested = false;
        self.refresh_due();
        if cfg!(debug_assertions) {
            self.check_page_table();
        }
    }
}

impl Drop for Heap {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            self.debug_assert_quiescent();
        }
        // The due word belongs to the heap's owner, which may be dropping it too, and a charge
        // that outlives the heap keeps its ledger (`value::ExternalCharge`): neither may write it.
        self.due_word = None;
        self.control.external.due_word.set(None);
        self.release_all();
    }
}

#[derive(Default)]
pub(crate) struct Control {
    nogc_depth: Cell<u32>,
    pending_collect: Cell<bool>,
    borrows: Cell<u32>,
    collecting: Cell<bool>,
    /// How many markings of this heap have begun.
    markings: Cell<u64>,
    /// The bytes this heap's cells own outside it (see [`ExternalBytes`]).
    external: ExternalBytes,
}

/// The ledger of the bytes the heap's cells own outside the heap: Rust-heap payloads (ordered
/// collection tables, hash indexes, binding storage, bound arguments, stack traces), which their
/// containers report as they acquire and release them ([`Control::charge_external`],
/// [`Control::release_external`]). Only a collection frees the payload of a dead cell, so these
/// bytes count toward the collection triggers as cells do (SpiderMonkey's malloc-heap accounting,
/// JavaScriptCore's extra memory): a collection is due once more than `budget` bytes were charged
/// since the last one, and the bytes still charged count with the cells against the byte budget
/// ([`Heap::set_heap_growth`]). Buffer data has its own pressure (`interpreter::BufferPressure`).
#[derive(Default)]
struct ExternalBytes {
    /// Charged and not released.
    live: Cell<usize>,
    /// Charged since the last collection's sweep.
    since: Cell<usize>,
    /// `since` past which a collection is due; 0 while the heap has set none.
    budget: Cell<usize>,
    /// `since` passed `budget`: a collection is due. Cleared by every sweep.
    due: Cell<bool>,
    /// The heap's due word (see [`Heap::attach_due_word`]), set to 1 with `due`.
    due_word: Cell<Option<NonNull<Cell<u64>>>>,
}

impl Control {
    /// Whether a collection of this heap is in progress (from its marking to the end of its
    /// sweep).
    pub(crate) fn collecting(&self) -> bool {
        self.collecting.get()
    }

    /// How many markings of this heap have begun: a cell made since the count was read is
    /// young at the next one.
    pub(crate) fn markings(&self) -> u64 {
        self.markings.get()
    }

    /// Record `bytes` of Rust-heap payload acquired by a cell of this heap, or by a container a
    /// cell will own. A collection becomes due once the bytes charged since the last one pass the
    /// external budget; the due word says so at once, so the next safepoint collects.
    pub(crate) fn charge_external(&self, bytes: usize) {
        let external = &self.external;
        external.live.set(external.live.get().saturating_add(bytes));
        let since = external.since.get().saturating_add(bytes);
        external.since.set(since);
        let budget = external.budget.get();
        if budget != 0 && since > budget && !external.due.replace(true) {
            if let Some(word) = external.due_word.get() {
                // SAFETY: `Heap::attach_due_word`'s contract: the word stays valid while the heap,
                // whose control block this is, lives, and only the heap and this ledger write it.
                unsafe { word.as_ref() }.set(1);
            }
        }
    }

    /// Record that `bytes` charged by [`Control::charge_external`] were released.
    pub(crate) fn release_external(&self, bytes: usize) {
        let external = &self.external;
        external.live.set(external.live.get().saturating_sub(bytes));
    }
}

/// RAII no-GC region (I9). Collection requests inside are deferred.
/// Holds only the shared control block, so `alloc(&mut Heap)` remains
/// callable while the guard is alive.
#[cfg(test)]
pub(crate) struct NoGcScope {
    control: Rc<Control>,
}

#[cfg(test)]
impl Drop for NoGcScope {
    fn drop(&mut self) {
        self.control
            .nogc_depth
            .set(self.control.nogc_depth.get() - 1);
    }
}

/// An outstanding GcCell borrow (I9); see [`Heap::borrow_scope`].
#[cfg(test)]
pub(crate) struct BorrowScope {
    control: Rc<Control>,
}

#[cfg(test)]
impl Drop for BorrowScope {
    fn drop(&mut self) {
        self.control.borrows.set(self.control.borrows.get() - 1);
    }
}

/// Declares this thread's native stack, from the current frame up to a
/// top, as holding heap references (Decision 0). Conservative collections
/// on this thread scan it, plus segments entered through [`grow`]; without
/// a declaration they defer. Nested declarations are no-ops.
#[cfg(test)]
pub(crate) struct NativeStackScope {
    registered: bool,
    _thread: std::marker::PhantomData<*const ()>,
}

/// # Safety
/// `top` must lie in the current thread's stack mapping, at or above every
/// frame that holds heap references across collections while the scope is
/// alive, and the scope must be dropped on this thread outside any segment
/// entered through [`grow`].
#[cfg(test)]
pub(crate) unsafe fn enter_native_stack(top: *const u8) -> NativeStackScope {
    NativeStackScope {
        registered: native::enter(top as usize),
        _thread: std::marker::PhantomData,
    }
}

#[cfg(test)]
impl Drop for NativeStackScope {
    fn drop(&mut self) {
        if self.registered {
            native::exit();
        }
    }
}

/// Register this thread's whole native stack, from the current frame up to its outermost frame,
/// as a permanent entry region (idempotent). Interpreters call this at creation so every engine
/// frame on the thread lies below the scanned top. The top is the stack mapping's, or on a
/// spawned Linux thread the start of the thread descriptor and static TLS that glibc and musl
/// put at the top of the mapping: thread-locals are not roots. Returns false where the stack
/// bounds cannot be read; conservative collection then defers on this thread.
pub(crate) fn register_thread_native_stack() -> bool {
    native::enter_thread()
}

/// Native stack a collection zeroes below its frame once `Heap::begin_marking` has run, before it
/// scans the stack: the frames of heap verification's barrier check, which traces every old
/// cell, and of the clearing of marks and cards, which reads page addresses.
#[cfg(not(target_arch = "wasm32"))]
const BEGIN_MARKING_SCRUB_BYTES: usize = if cfg!(debug_assertions) {
    256 << 10
} else {
    16 << 10
};

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
/// nothing of its own is left in the zeroed range. Whole blocks are zeroed with `dc zva`, which
/// writes a block (DCZID_EL0's size, usually 64 bytes) without reading it, as memset does;
/// pairs of zero registers cover the unaligned head and the tail, and everything when the
/// instruction is prohibited.
#[cfg(target_arch = "aarch64")]
#[unsafe(naked)]
unsafe extern "C" fn zero_below_sp(bytes: usize) {
    std::arch::naked_asm!(
        "mov x9, sp",
        "sub x10, x9, x0",
        // The block size, 4 << DCZID_EL0.BS bytes, unless DZP prohibits the instruction.
        "mrs x11, dczid_el0",
        "tbnz x11, #4, 2f",
        "and x11, x11, #15",
        "mov x12, #4",
        "lsl x12, x12, x11",
        "sub x13, x12, #1",
        // The head, up to a block boundary.
        "4:",
        "tst x10, x13",
        "b.eq 5f",
        "cmp x10, x9",
        "b.hs 3f",
        "stp xzr, xzr, [x10], #16",
        "b 4b",
        // Whole blocks.
        "5:",
        "add x14, x10, x12",
        "cmp x14, x9",
        "b.hi 2f",
        "dc zva, x10",
        "mov x10, x14",
        "b 5b",
        // The tail, or everything without `dc zva`.
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
/// the call pushed. A leaf with no frame, so nothing of its own is left in the zeroed range. The
/// System V convention on every OS, Windows included: the argument arrives in rdi, and rcx is free.
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "sysv64" fn zero_below_sp(bytes: usize) {
    // `rep stosq` from the range's start up to the return address: (bytes - 8) / 8 words. The
    // direction flag is clear on entry (the calling convention).
    std::arch::naked_asm!(
        "lea rcx, [rsp + 8]",
        "sub rcx, rdi",
        "xchg rdi, rcx",
        "sub rcx, 8",
        "shr rcx, 3",
        "xor eax, eax",
        "rep stosq",
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
    any(target_os = "linux", target_os = "macos", target_os = "windows"),
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
    //! - Windows: stacker runs a segment on a fiber, and the system reports a
    //!   fiber's (or the thread's) stack bounds exactly.
    use std::cell::RefCell;

    /// Words for the callee-saved registers: AAPCS64 x19–x30 and d8–d15 (20), System V rbx,
    /// rbp and r12–r15 (6), Win64 rbx, rbp, rdi, rsi, r12–r15 and xmm6–xmm15 (28).
    #[cfg(not(all(target_arch = "x86_64", target_os = "windows")))]
    pub(super) const SPILL_WORDS: usize = 24;
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    pub(super) const SPILL_WORDS: usize = 28;

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

    /// System V callee-saved registers.
    #[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
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

    /// Win64 callee-saved registers, xmm6–xmm15 included: the compiler may keep a 16-byte
    /// value, a cell address in its low half, in one across a call.
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    #[inline(always)]
    pub(super) fn spill_registers(buf: &mut [u64; SPILL_WORDS]) {
        // SAFETY: stores rbx, rbp, rdi, rsi and r12–r15 into the first 8 words of `buf`, and
        // xmm6–xmm15 into the 20 after them.
        unsafe {
            std::arch::asm!(
                "mov [{buf}], rbx",
                "mov [{buf} + 8], rbp",
                "mov [{buf} + 16], rdi",
                "mov [{buf} + 24], rsi",
                "mov [{buf} + 32], r12",
                "mov [{buf} + 40], r13",
                "mov [{buf} + 48], r14",
                "mov [{buf} + 56], r15",
                "movdqu [{buf} + 64], xmm6",
                "movdqu [{buf} + 80], xmm7",
                "movdqu [{buf} + 96], xmm8",
                "movdqu [{buf} + 112], xmm9",
                "movdqu [{buf} + 128], xmm10",
                "movdqu [{buf} + 144], xmm11",
                "movdqu [{buf} + 160], xmm12",
                "movdqu [{buf} + 176], xmm13",
                "movdqu [{buf} + 192], xmm14",
                "movdqu [{buf} + 208], xmm15",
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

    #[cfg(not(target_os = "windows"))]
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

    #[cfg(test)]
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

    /// The entry region's exclusive top, when one is registered.
    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn entry_top() -> Option<usize> {
        REGIONS.with(|regions| regions.borrow().first().map(|region| region.hi))
    }

    /// Permanently register the current thread's stack up to its outermost frame.
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
    unsafe extern "C" {
        fn pthread_self() -> usize;
    }

    /// The top of the stack mapping, lowered to [`tls_floor`].
    #[cfg(target_os = "linux")]
    fn thread_stack_top() -> Option<usize> {
        // Opaque pthread_attr_t: 56 bytes (glibc x86_64) / 64 bytes (aarch64); 128 covers both.
        #[repr(C, align(16))]
        struct Attr([u8; 128]);
        unsafe extern "C" {
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
        let status = unsafe {
            if pthread_getattr_np(pthread_self(), &mut attr) != 0 {
                return None;
            }
            let status = pthread_attr_getstack(&attr, &mut addr, &mut size);
            pthread_attr_destroy(&mut attr);
            status
        };
        (status == 0 && size != 0).then(|| tls_floor(current_sp(), addr as usize + size))
    }

    /// The lowest of `top` and, between `sp` and `top`, this thread's descriptor (`pthread_self`
    /// is its address on glibc and musl) and its instances of the loaded modules' TLS blocks
    /// (`dlpi_tls_data`). A spawned thread has both at the top of its stack mapping, above its
    /// outermost frame, where pthread_getattr_np's bounds include them; the main thread's lie
    /// elsewhere. A lazily built thread-local can hold a stale cell address in bytes it never
    /// wrote (an enum's unused payload), which would otherwise be a root.
    #[cfg(target_os = "linux")]
    fn tls_floor(sp: usize, top: usize) -> usize {
        /// `struct dl_phdr_info`, glibc's and musl's layout.
        #[repr(C)]
        struct PhdrInfo {
            addr: usize,
            name: *const std::ffi::c_char,
            phdr: *const std::ffi::c_void,
            phnum: u16,
            adds: u64,
            subs: u64,
            tls_modid: usize,
            tls_data: *mut std::ffi::c_void,
        }
        type Visit =
            unsafe extern "C" fn(*mut PhdrInfo, usize, *mut std::ffi::c_void) -> std::ffi::c_int;
        unsafe extern "C" {
            fn dl_iterate_phdr(visit: Visit, data: *mut std::ffi::c_void) -> std::ffi::c_int;
        }
        struct Window {
            sp: usize,
            floor: usize,
        }
        unsafe extern "C" fn lower_to_block(
            info: *mut PhdrInfo,
            size: usize,
            data: *mut std::ffi::c_void,
        ) -> std::ffi::c_int {
            // SAFETY: dl_iterate_phdr passes an `info` of `size` bytes and our `data`.
            let (info, window) = unsafe { (&*info, &mut *data.cast::<Window>()) };
            let block = if size >= size_of::<PhdrInfo>() {
                info.tls_data as usize
            } else {
                0
            };
            if window.sp < block && block < window.floor {
                window.floor = block;
            }
            0
        }
        let mut window = Window { sp, floor: top };
        // SAFETY: no preconditions.
        let descriptor = unsafe { pthread_self() };
        if sp < descriptor && descriptor < top {
            window.floor = descriptor;
        }
        // SAFETY: the visitor reads only `info` and writes only `window`, which outlives the call.
        unsafe { dl_iterate_phdr(lower_to_block, (&raw mut window).cast()) };
        window.floor
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

    /// The bounds of the stack in use: the thread's, or on a stacker segment its fiber's
    /// (SwitchToFiber swaps the bounds the system reports).
    #[cfg(target_os = "windows")]
    fn stack_limits() -> (usize, usize) {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThreadStackLimits(low: *mut usize, high: *mut usize);
        }
        let (mut low, mut high) = (0, 0);
        // SAFETY: writes the current stack's bounds into the two locals.
        unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
        (low, high)
    }

    #[cfg(target_os = "windows")]
    fn thread_stack_top() -> Option<usize> {
        let (_, high) = stack_limits();
        (high != 0).then_some(high)
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
    #[cfg(not(target_os = "windows"))]
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

    /// Bounds of the stacker segment we are running on: on Windows stacker runs it on a fiber,
    /// whose stack the system reports exactly (the bounds-precision test checks them against
    /// the allocation). A mismatch yields an empty region: collections inside then defer.
    #[cfg(target_os = "windows")]
    fn segment_region(_stack_size: usize, resume_sp: usize) -> Region {
        let sp = current_sp();
        let (lo, hi) = stack_limits();
        if (lo..hi).contains(&sp) {
            return Region { lo, hi, resume_sp };
        }
        if cfg!(debug_assertions) {
            panic!("could not derive stacker segment bounds");
        }
        Region {
            lo: usize::MAX,
            hi: 0,
            resume_sp,
        }
    }

    #[cfg(test)]
    pub(super) fn innermost_region() -> Option<(usize, usize)> {
        REGIONS.with(|regions| regions.borrow().last().map(|r| (r.lo, r.hi)))
    }
}

#[cfg(not(all(
    any(target_os = "linux", target_os = "macos", target_os = "windows"),
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
mod native {
    //! No conservative scanner for this target (wasm32 cannot scan its
    //! stack): every collection that would scan the native stack defers.
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

    #[cfg(test)]
    pub(super) fn enter(_: usize) -> bool {
        false
    }

    #[cfg(test)]
    pub(super) fn exit() {}

    #[cfg(all(test, target_os = "linux"))]
    pub(super) fn entry_top() -> Option<usize> {
        None
    }

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
    use std::cell::RefCell;
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
        /// A node whose cells gain edges after initialization: its pages end in a card table.
        carded: DescId,
        /// The words the Fixture's client reports as roots. Boxed, off the native stack: a
        /// collection the heap drives itself, which has no client, does not see them.
        roots: Box<[u64; 64]>,
    }

    impl Fixture {
        fn new() -> Self {
            let mut heap = Heap::new();
            let node = heap.register_desc(LayoutDesc {
                name: "node",
                trace: trace_node,
                mark_trace: None,
                finalize: None,
                carded: false,
            });
            let leaf = heap.register_desc(LayoutDesc {
                name: "leaf",
                trace: trace_none,
                mark_trace: None,
                finalize: Some(fin_count),
                carded: false,
            });
            let carded = heap.register_desc(LayoutDesc {
                name: "carded node",
                trace: trace_node,
                mark_trace: None,
                finalize: None,
                carded: true,
            });
            Self {
                heap,
                node,
                leaf,
                carded,
                roots: Box::new([0; 64]),
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

        /// Precise major collection through the Fixture's client: native frames are
        /// not roots, so death assertions do not depend on what registers
        /// happen to retain.
        fn collect(&mut self) {
            self.collect_with(&no_edges);
        }

        /// [`Fixture::collect`], minor.
        fn collect_minor(&mut self) {
            assert!(self.start_marking_kind(&no_edges, CollectionKind::Minor));
            self.heap.finish_collection(&[]);
        }

        /// [`Fixture::collect`] whose client also reports side-table edges
        /// and ephemerons through `edges`.
        fn collect_with(&mut self, edges: &dyn Fn(&mut EdgeSink<'_>)) {
            assert!(self.start_marking(edges));
            self.heap.finish_collection(&[]);
        }

        /// Phase 1 of [`Fixture::collect_with`], without native roots.
        fn start_marking(&mut self, edges: &dyn Fn(&mut EdgeSink<'_>)) -> bool {
            self.start_marking_kind(edges, CollectionKind::Major)
        }

        /// [`Fixture::start_marking`] of a collection of `kind`.
        fn start_marking_kind(
            &mut self,
            edges: &dyn Fn(&mut EdgeSink<'_>),
            kind: CollectionKind,
        ) -> bool {
            let client = FixtureClient {
                roots: &self.roots[..],
                edges,
            };
            self.heap.start_marking(false, &client, kind)
        }

        fn live(&self) -> usize {
            self.heap.stats().live_cells
        }
    }

    /// The Fixture's embedder, as the interpreter is the engine's: it reports
    /// the Fixture's root words, and through `edges` the edges and ephemerons
    /// of side tables, which reach the heap through the EdgeSink the engine's
    /// client uses.
    struct FixtureClient<'a> {
        roots: &'a [u64],
        edges: &'a dyn Fn(&mut EdgeSink<'_>),
    }

    impl MarkClient for FixtureClient<'_> {
        fn roots(&self, sink: &mut dyn FnMut(u64)) {
            for &root in self.roots {
                sink(root);
            }
        }

        fn side_slot_owners(&self, _: &mut dyn FnMut(usize)) {}

        fn trace_side_slots(&self, _: usize, _: &mut dyn FnMut(u64)) {}

        fn ephemerons(&self, sink: &mut EdgeSink<'_>) {
            (self.edges)(sink);
        }
    }

    /// The side tables of a client that has none.
    fn no_edges(_: &mut EdgeSink<'_>) {}

    thread_local! {
        static WORK_PURPOSES: RefCell<Vec<(usize, TracePurpose)>> = const { RefCell::new(Vec::new()) };
    }

    fn remember_work(cell: *const u8, purpose: TracePurpose, sink: &mut MarkSink<'_>) {
        WORK_PURPOSES.with(|work| work.borrow_mut().push((cell as usize, purpose)));
        trace_node(cell, &mut |edge| sink.mark(edge as u64));
    }

    #[test]
    fn remembered_work_items_keep_full_children_and_ephemeron_ready_work_distinct() {
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            std::mem::size_of::<TraceWork>(),
            std::mem::size_of::<(usize, DescId)>()
        );
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        fx.heap.descs[fx.node.0 as usize].mark_trace = Some(remember_work);
        fx.heap.descs[fx.carded.0 as usize].mark_trace = Some(remember_work);
        let old = fx.heap.alloc(fx.carded, 16);
        fx.roots[0] = tagged(old);
        fx.collect();
        WORK_PURPOSES.with(|work| work.borrow_mut().clear());
        let direct = fx.alloc_node(0, 1);
        let child = fx.alloc_node(0, 2);
        let held = fx.alloc_node(0, 3);
        fx.roots[1] = tagged(direct);
        fx.roots[2] = tagged(old); // A repeated positive old root must not suppress card work.
        set_word(old, 0, tagged(child));
        // SAFETY: old is a live carded cell of this heap.
        unsafe { card_mark(old) };
        let before = fx.heap.stats().cells_marked;
        assert!(fx.start_marking_kind(
            &|sink| sink.ephemeron(old, tagged(child), tagged(held)),
            CollectionKind::Minor
        ));
        fx.heap.finish_collection(&[]);
        assert_eq!(fx.heap.stats().cells_marked - before, 3);
        assert_eq!(fx.heap.stats().last_card_cells, 1);
        WORK_PURPOSES.with(|work| {
            let work = work.borrow();
            assert_eq!(work.len(), 4);
            assert!(work.contains(&(old as usize, TracePurpose::Remembered)));
            for cell in [direct, child, held] {
                assert!(work.contains(&(cell as usize, TracePurpose::Full)));
            }
        });
        assert!(fx.heap.owns_cell(held as usize, fx.node));
    }

    fn word(p: *const u8, index: usize) -> u64 {
        unsafe { std::ptr::read((p as *const u64).add(index)) }
    }

    fn set_word(p: *mut u8, index: usize, value: u64) {
        unsafe { std::ptr::write((p as *mut u64).add(index), value) };
    }

    fn assert_zeroed_cell(cell: *const u8, bytes: usize) {
        // SAFETY: tests call this only for a complete cell just handed out by Heap::alloc.
        assert!(unsafe { std::slice::from_raw_parts(cell, bytes) }
            .iter()
            .all(|&byte| byte == 0));
    }

    fn page_slot(heap: &Heap, cell: usize) -> usize {
        heap.page_table
            .get(cell >> PAGE_SHIFT)
            .expect("a live small cell's page") as usize
    }

    /// The page's reclaimed holes, in address order (the order allocation takes them): the
    /// cells below `next_unused` whose allocation bit is clear. Checks that no free or virgin
    /// cell is marked or a zombie, that `live` counts every allocated cell (zombies included)
    /// and that `has_room` agrees. The holes are never read: the sweep writes no link into
    /// them, so the virgin tail and the holes may hold any bytes.
    fn checked_free_chain(page: &PageMeta) -> Vec<usize> {
        assert!(page.next_unused <= page.cells);
        let mut holes = Vec::new();
        let mut allocated = 0;
        for index in 0..page.cells {
            let used = index < page.next_unused;
            let occupied = bit(&page.allocated, index);
            if occupied {
                assert!(used, "an allocated cell in the virgin tail");
                allocated += 1;
            } else {
                assert!(!bit(&page.marks, index), "a marked free cell");
                assert!(!bit(&page.zombies, index), "a free zombie");
                if used {
                    holes.push(index);
                }
            }
        }
        assert_eq!(
            page.live, allocated,
            "page.live includes all allocated zombies"
        );
        assert_eq!(
            page.has_room(),
            !holes.is_empty() || page.next_unused < page.cells
        );
        holes
    }

    fn check_free_chains(heap: &mut Heap) {
        heap.flush_cursors();
        for page in heap.pages.iter().flatten() {
            checked_free_chain(page);
        }
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

    /// Pages are 64 KiB-aligned and every whole cell is zero before handout: new, reused from
    /// the spare pool, and given back to the source when the pool is trimmed and taken again.
    /// On Linux the pages given back are taken again before a new chunk is mapped (after the
    /// current chunk's last pages), so they come from the same chunks. The heap's release returns
    /// A small heap's chunks keep base pages, which only the pages taken make resident; the
    /// chunk that makes the heap large enough advises every chunk mapped for huge pages, as the
    /// kernel's own mapping flags show.
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "aarch64", target_arch = "x86_64")
    ))]
    #[test]
    fn small_heaps_keep_base_pages_until_enough_chunks_are_mapped() {
        use super::page_source::{PageSource, CHUNK_SIZE, HUGE_PAGE_MIN_CHUNKS};
        // The VmFlags of the mapping holding `address`, from /proc/self/smaps.
        fn advised(address: usize) -> bool {
            let smaps = std::fs::read_to_string("/proc/self/smaps").expect("smaps");
            let mut inside = false;
            for line in smaps.lines() {
                if let Some((range, _)) = line.split_once(' ') {
                    if let Some((start, end)) = range.split_once('-') {
                        if let (Ok(start), Ok(end)) = (
                            usize::from_str_radix(start, 16),
                            usize::from_str_radix(end, 16),
                        ) {
                            inside = (start..end).contains(&address);
                            continue;
                        }
                    }
                }
                if inside {
                    if let Some(flags) = line.strip_prefix("VmFlags:") {
                        return flags.split_whitespace().any(|flag| flag == "hg");
                    }
                }
            }
            panic!("no mapping holds {address:#x}");
        }
        let mut source = PageSource::new();
        let mut chunks = Vec::new();
        let mut taken = Vec::new();
        for chunk in 1..=HUGE_PAGE_MIN_CHUNKS {
            let first = source.take() as usize;
            assert_eq!(first % CHUNK_SIZE, 0, "a new chunk's first page");
            taken.push(first);
            for _ in 1..CHUNK_SIZE / PAGE_SIZE {
                taken.push(source.take() as usize);
            }
            chunks.push(first);
            assert_eq!(source.chunks(), chunk);
            assert_eq!(source.huge_pages_advised(), chunk == HUGE_PAGE_MIN_CHUNKS);
            for &base in &chunks {
                assert_eq!(
                    advised(base),
                    chunk == HUGE_PAGE_MIN_CHUNKS,
                    "chunk {chunk}"
                );
            }
        }
        // SAFETY: every page this source gave is listed and no longer used.
        unsafe { source.release(taken) };
    }

    /// every page: live, spare and given back.
    #[test]
    fn pages_are_aligned_and_zeroed_when_new_reused_or_taken_again() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let per_page = PAGE_SIZE / 16;
        // Fill pages with nodes whose value words are all ones. Check both words of every
        // handout, including recycled storage: untouched spare payload need not be zero yet.
        let fill = |fx: &mut Fixture, pages: usize| -> Vec<usize> {
            (0..pages)
                .map(|_| {
                    let first = fx.heap.alloc(fx.node, 16);
                    let base = first as usize;
                    assert_eq!(base % PAGE_SIZE, 0, "a new page's first cell starts it");
                    assert_zeroed_cell(first, 16);
                    set_word(first, 1, u64::MAX);
                    for index in 1..per_page {
                        let cell = fx.heap.alloc(fx.node, 16);
                        assert_eq!(cell as usize, base + 16 * index);
                        assert_zeroed_cell(cell, 16);
                        set_word(cell, 1, u64::MAX);
                    }
                    base
                })
                .collect()
        };
        // On Linux, the source's chunks and the pages given back to it; elsewhere, the pages it
        // has mapped are the heap's, live and spare.
        let source = |fx: &Fixture, chunks: usize, free: usize| {
            #[cfg(all(
                target_os = "linux",
                any(target_arch = "aarch64", target_arch = "x86_64")
            ))]
            assert_eq!(
                (
                    fx.heap.page_source.chunks(),
                    fx.heap.page_source.free_pages()
                ),
                (chunks, free)
            );
            #[cfg(not(all(
                target_os = "linux",
                any(target_arch = "aarch64", target_arch = "x86_64")
            )))]
            {
                let _ = (chunks, free);
                let stats = fx.heap.stats();
                assert_eq!(
                    fx.heap.page_source.mapped_bytes(),
                    (stats.pages_live + stats.pages_spare) * PAGE_SIZE
                );
            }
        };
        // On Linux, 32 pages a chunk: 28 of the fourth chunk's are not taken yet.
        let first = fill(&mut fx, 100);
        source(&fx, 4, 0);
        // Nothing is kept: the pool keeps 64 pages and gives the other 36 back.
        fx.collect();
        assert_eq!(fx.heap.stats().pages_live, 0);
        assert_eq!(fx.heap.spare_pages.len(), 64);
        source(&fx, 4, 36);
        // The pool's 64 pages come first; on Linux then the fourth chunk's last 28 and 8 of the
        // pages given back, then the other 28 of those, and only then a new chunk.
        let mut taken = fill(&mut fx, 100);
        assert!(taken[..64].iter().all(|base| first.contains(base)));
        source(&fx, 4, 28);
        taken.extend(fill(&mut fx, 28));
        source(&fx, 4, 0);
        let last = fill(&mut fx, 1)[0];
        source(&fx, 5, 0);
        #[cfg(all(
            target_os = "linux",
            any(target_arch = "aarch64", target_arch = "x86_64")
        ))]
        {
            let chunks = |pages: &[usize]| -> std::collections::BTreeSet<usize> {
                pages
                    .iter()
                    .map(|base| base & !(page_source::CHUNK_SIZE - 1))
                    .collect()
            };
            let distinct: std::collections::HashSet<&usize> = taken.iter().collect();
            assert_eq!(distinct.len(), 128);
            assert_eq!(chunks(&taken), chunks(&first), "the same four chunks");
            assert!(chunks(&first).is_disjoint(&chunks(&[last])));
        }
        // Two pages keep a cell; the pool keeps 64 of the other 127 and gives 63 back.
        fx.roots[0] = tagged(taken[0] as *const u8);
        fx.roots[1] = tagged(last as *const u8);
        fx.collect();
        assert_eq!(fx.heap.stats().pages_live, 2);
        assert_eq!(fx.heap.spare_pages.len(), 64);
        source(&fx, 5, 63);
        fx.heap.release_all();
        assert_eq!(fx.heap.page_source.mapped_bytes(), 0);
        source(&fx, 0, 0);
    }

    #[test]
    fn virgin_slots_are_not_read_or_linked_by_allocation_or_sweep() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let anchor = fx.alloc_node(0, 7);
        fx.roots[0] = tagged(anchor);
        let child = fx.heap.alloc(fx.leaf, 16) as usize;
        let slot = page_slot(&fx.heap, anchor as usize);
        let page = fx.heap.page_view(slot);
        assert_eq!((page.next_unused, checked_free_chain(page).len()), (1, 0));
        let sentinels: Vec<(usize, [u64; 2])> = [1, 63, 64, 65, page.cells - 1]
            .into_iter()
            .map(|index| {
                let ptr = page.cell_ptr(index) as *mut u8;
                let words = if index == 65 {
                    [tagged(child as *const u8), 0xa5a5_a5a5_a5a5_a5a5]
                } else {
                    [0xa5a5_a5a5_a5a5_a5a5; 2]
                };
                // Deliberately dirty reserved, unallocated storage: GC must neither read it
                // as payload nor rewrite it as a free-list link.
                set_word(ptr, 0, words[0]);
                set_word(ptr, 1, words[1]);
                (ptr as usize, words)
            })
            .collect();
        let virgin = sentinels[3].0;
        fx.roots[1] = tagged((virgin + 8) as *const u8);
        assert_eq!(
            fx.heap.resolve(virgin as u64 + 8),
            Some(virgin as *const u8)
        );
        assert_eq!(fx.heap.cell_age(virgin as u64), None);
        assert!(!fx.heap.owns_cell(virgin, fx.node));
        assert!(fx.start_marking_kind(&no_edges, CollectionKind::Minor));
        fx.heap.finish_collection(&[virgin, sentinels[0].0]);
        assert!(
            !fx.heap.is_allocated_word(child as u64),
            "virgin payload is not traced"
        );
        for kind in [CollectionKind::Major, CollectionKind::Minor] {
            assert!(fx.start_marking_kind(&no_edges, kind));
            fx.heap.finish_collection(&[virgin]);
            let page = fx.heap.page_view(slot);
            assert_eq!((page.next_unused, page.live), (1, 1));
            assert!(checked_free_chain(page).is_empty());
            for &(ptr, expected) in &sentinels {
                assert_eq!(
                    [word(ptr as *const u8, 0), word(ptr as *const u8, 1)],
                    expected
                );
            }
        }
        let next = fx.heap.alloc(fx.node, 16);
        assert_eq!(next as usize, sentinels[0].0);
        assert_zeroed_cell(next, 16);
        fx.roots[2] = tagged(next);
        fx.collect_minor();
        for &(ptr, expected) in &sentinels[1..] {
            assert_eq!(
                [word(ptr as *const u8, 0), word(ptr as *const u8, 1)],
                expected
            );
        }
        check_free_chains(&mut fx.heap);
    }

    #[test]
    fn bitmap_sweep_preserves_holes_across_word_and_capacity_boundaries() {
        // 768-byte cells have 85 slots: a partial last bitmap word. The other used-prefix
        // lengths put both death and survival immediately around the 63/64/65 boundary.
        for used in [63usize, 64, 65, 85] {
            FIN_COUNT.with(|count| count.set(0));
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            let size = 768;
            let cells: Vec<usize> = (0..used)
                .map(|_| {
                    let cell = fx.heap.alloc(fx.leaf, size);
                    assert_zeroed_cell(cell, size);
                    // Leaf finalization never reads payload, so every byte may be dirty.
                    unsafe { std::ptr::write_bytes(cell, 0xa5, size) };
                    cell as usize
                })
                .collect();
            let slot = page_slot(&fx.heap, cells[0]);
            let capacity = fx.heap.page_view(slot).cells;
            assert_eq!(capacity, 85);
            let kept = [0, used / 2, used - 1];
            for (root, index) in kept.into_iter().enumerate() {
                fx.roots[root] = tagged(cells[index] as *const u8);
            }
            fx.collect_minor();
            let first_deaths = used - kept.len();
            assert_eq!(FIN_COUNT.with(Cell::get), first_deaths);
            let page = fx.heap.page_view(slot);
            assert_eq!((page.next_unused, page.live), (used, kept.len()));
            let holes = checked_free_chain(page);
            assert_eq!(holes.len(), first_deaths);
            fx.collect();
            fx.collect_minor();
            assert_eq!(checked_free_chain(fx.heap.page_view(slot)), holes);
            assert_eq!(FIN_COUNT.with(Cell::get), first_deaths);

            let mut handed_out: std::collections::HashSet<usize> =
                kept.into_iter().map(|index| cells[index]).collect();
            for _ in 0..holes.len() {
                let cell = fx.heap.alloc(fx.leaf, size);
                assert_zeroed_cell(cell, size);
                assert!(
                    cells.contains(&(cell as usize)),
                    "reclaimed holes precede virgin slots"
                );
                assert!(
                    handed_out.insert(cell as usize),
                    "a cell was handed out twice"
                );
            }
            assert_eq!(fx.heap.page_view(slot).next_unused, used);
            for index in used..capacity {
                let cell = fx.heap.alloc(fx.leaf, size);
                assert_eq!(cell as usize, cells[0] + index * size);
                assert_zeroed_cell(cell, size);
                assert!(handed_out.insert(cell as usize));
            }
            assert_eq!(handed_out.len(), capacity);
            let page = fx.heap.page_view(slot);
            assert_eq!(page.next_unused, capacity);
            assert!(!page.has_room());
            assert!(checked_free_chain(page).is_empty());
            let extra = fx.heap.alloc(fx.leaf, size);
            assert_zeroed_cell(extra, size);
            assert_ne!(extra as usize >> PAGE_SHIFT, cells[0] >> PAGE_SHIFT);
            assert!(handed_out.insert(extra as usize));
            fx.collect_minor();
            assert_eq!(fx.live(), kept.len());
            assert_eq!(
                FIN_COUNT.with(Cell::get),
                first_deaths + capacity - kept.len() + 1
            );
            let holes = checked_free_chain(fx.heap.page_view(slot));
            assert_eq!(holes.len(), capacity - kept.len());
            fx.collect();
            assert_eq!(checked_free_chain(fx.heap.page_view(slot)), holes);
            check_free_chains(&mut fx.heap);
        }
    }

    #[test]
    fn virgin_capacity_is_available_after_minor_and_major_sweeps() {
        for kind in [CollectionKind::Minor, CollectionKind::Major] {
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            let anchor = fx.alloc_node(0, 1);
            fx.roots[0] = tagged(anchor);
            let slot = page_slot(&fx.heap, anchor as usize);
            let class = size_class(16).unwrap();
            // Park a valid partially used page off current so both availability rebuilds must
            // recognize virgin capacity even though no reclaimed hole exists.
            fx.heap.classes[fx.node.0 as usize][class].current = None;
            assert!(fx.start_marking_kind(&no_edges, kind));
            fx.heap.finish_collection(&[]);
            assert_eq!(
                fx.heap.classes[fx.node.0 as usize][class].avail,
                [slot as u32]
            );
            let page = fx.heap.page_view(slot);
            assert_eq!((page.next_unused, checked_free_chain(page).len()), (1, 0));
            let next = fx.heap.alloc(fx.node, 16);
            assert_eq!(next as usize, anchor as usize + 16);
            assert_zeroed_cell(next, 16);
            assert_eq!(fx.heap.stats().pages_live, 1);
            assert_eq!(fx.heap.young_page_count(), 1);
            check_free_chains(&mut fx.heap);
        }
    }

    #[test]
    fn black_allocations_use_holes_then_virgin_slots_and_survive() {
        FIN_COUNT.with(|count| count.set(0));
        let count = Box::new(Cell::new(-1));
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        // SAFETY: count outlives the heap and is written only by it.
        unsafe { fx.heap.attach_live_count(fx.leaf, NonNull::from(&*count)) };
        let anchor = fx.heap.alloc(fx.leaf, 16);
        let dead = fx.heap.alloc(fx.leaf, 16);
        fx.roots[0] = tagged(anchor);
        fx.collect();
        let slot = page_slot(&fx.heap, anchor as usize);
        assert_eq!(checked_free_chain(fx.heap.page_view(slot)), [1]);
        assert_eq!((count.get(), FIN_COUNT.with(Cell::get)), (1, 1));
        assert!(fx.start_marking_kind(&no_edges, CollectionKind::Minor));
        let hole = fx.heap.alloc(fx.leaf, 16);
        let virgin = fx.heap.alloc(fx.leaf, 16);
        assert_eq!(hole, dead);
        assert_eq!(virgin as usize, anchor as usize + 32);
        for cell in [hole, virgin] {
            assert_zeroed_cell(cell, 16);
            assert!(
                is_marked_at(&fx.heap, cell as usize),
                "allocation between phases is black"
            );
        }
        assert_eq!(count.get(), 3);
        fx.heap.finish_collection(&[]);
        assert_eq!(
            (fx.live(), count.get(), FIN_COUNT.with(Cell::get)),
            (3, 3, 1)
        );
        check_free_chains(&mut fx.heap);
        fx.collect_minor();
        assert_eq!(count.get(), 3, "black survivors are now old");
        fx.collect();
        assert_eq!(
            (fx.live(), count.get(), FIN_COUNT.with(Cell::get)),
            (1, 1, 3)
        );
        // The holes, in the address order allocation takes them.
        assert_eq!(checked_free_chain(fx.heap.page_view(slot)), [1, 2]);
    }

    #[test]
    fn black_allocation_on_a_new_page_survives_both_collection_kinds() {
        for kind in [CollectionKind::Minor, CollectionKind::Major] {
            FIN_COUNT.with(|count| count.set(0));
            let count = Box::new(Cell::new(-1));
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            // SAFETY: count outlives the heap and is written only by it.
            unsafe { fx.heap.attach_live_count(fx.leaf, NonNull::from(&*count)) };
            let size = 4096;
            let capacity = PAGE_SIZE / size;
            let cells: Vec<usize> = (0..capacity)
                .map(|_| fx.heap.alloc(fx.leaf, size) as usize)
                .collect();
            fx.roots[0] = tagged(cells[0] as *const u8);
            assert_eq!(fx.heap.stats().pages_live, 1);
            assert!(fx.start_marking_kind(&no_edges, kind));
            // The first page is still full until sweep. Allocating between phases must
            // create, register and blacken a new page's first cell without a traced root.
            let black = fx.heap.alloc(fx.leaf, size);
            assert_ne!(black as usize >> PAGE_SHIFT, cells[0] >> PAGE_SHIFT);
            assert_zeroed_cell(black, size);
            assert!(is_marked_at(&fx.heap, black as usize));
            assert_eq!(fx.heap.young_page_count(), 2);
            fx.heap.finish_collection(&[]);
            assert_eq!(fx.heap.stats().last_pages_swept, 2);
            assert_eq!(fx.heap.young_page_count(), 0);
            assert_eq!((fx.live(), count.get()), (2, 2));
            assert_eq!(FIN_COUNT.with(Cell::get), capacity - 1);
            assert!(fx.heap.owns_cell(black as usize, fx.leaf));
            check_free_chains(&mut fx.heap);
            fx.collect_minor();
            assert_eq!(count.get(), 2, "the new page's survivor is now old");
            fx.collect();
            assert_eq!(count.get(), 1);
            assert!(!fx.heap.is_allocated_word(black as u64));
            fx.roots.fill(0);
            fx.collect();
            assert_eq!((fx.live(), count.get()), (0, 0));
            assert_eq!(FIN_COUNT.with(Cell::get), capacity + 1);
            assert_eq!(fx.heap.stats().cells_swept, (capacity + 1) as u64);
            assert_eq!(fx.heap.stats().pages_live, 0);
        }
    }

    #[test]
    fn reused_pages_zero_each_handout_and_clear_new_card_tables() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let leaf = fx.leaf;
        let old = fill_page(&mut fx, leaf, 64);
        let base = old[0];
        fx.collect();
        assert_eq!(fx.heap.spare_pages, [base]);
        // SAFETY: this spare mapping has no remaining live cell or weak pin. Poison the full
        // old payload, including the future card table, to distinguish lazy per-cell clearing.
        unsafe { std::ptr::write_bytes(base as *mut u8, 0xa5, PAGE_SIZE) };
        let first = fx.heap.alloc(fx.carded, 192);
        assert_eq!(first as usize, base);
        assert_zeroed_cell(first, 192);
        fx.roots[0] = tagged(first);
        let slot = page_slot(&fx.heap, base);
        let page = fx.heap.page_view(slot);
        assert_eq!((page.next_unused, checked_free_chain(page).len()), (1, 0));
        assert!((0..page.cells).all(|index| !page.card_set(index)));
        assert_eq!(word((base + 384) as *const u8, 1), 0xa5a5_a5a5_a5a5_a5a5);
        let second = fx.heap.alloc(fx.carded, 192);
        assert_zeroed_cell(second, 192);
        fx.collect_minor();
        assert_eq!(checked_free_chain(fx.heap.page_view(slot)), [1]);
        assert_eq!(word((base + 384) as *const u8, 1), 0xa5a5_a5a5_a5a5_a5a5);
        assert_eq!(fx.heap.alloc(fx.carded, 192), second);
        assert_zeroed_cell(second, 192);
        let third = fx.heap.alloc(fx.carded, 192);
        assert_eq!(third as usize, base + 384);
        assert_zeroed_cell(third, 192);
        check_free_chains(&mut fx.heap);
    }

    #[test]
    fn zero_death_sweeps_clear_live_pins_and_all_waiters() {
        for kind in [CollectionKind::Minor, CollectionKind::Major] {
            FIN_COUNT.with(|count| count.set(0));
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            let anchor = fx.heap.alloc(fx.leaf, 16);
            fx.roots[0] = tagged(anchor);
            let slot = page_slot(&fx.heap, anchor as usize);
            assert!(fx.start_marking_kind(&no_edges, kind));
            let page = fx.heap.pages[slot].as_mut().unwrap();
            page.waiters.fill(u64::MAX);
            page.pins.fill(u64::MAX);
            fx.heap.finish_collection(&[anchor as usize]);
            let page = fx.heap.page_view(slot);
            assert!(page.pins.iter().all(|&word| word == 0));
            assert!(page.waiters.iter().all(|&word| word == 0));
            assert_eq!(FIN_COUNT.with(Cell::get), 0);
            assert!(checked_free_chain(page).is_empty());
            fx.roots[0] = 0;
            fx.collect();
            assert!(
                !fx.heap.is_allocated_word(anchor as u64),
                "no stale live-cell pin"
            );
            assert_eq!(FIN_COUNT.with(Cell::get), 1);
            assert_eq!(fx.heap.stats().pages_live, 0);
        }
    }

    #[test]
    fn zombies_are_not_recycled_until_unpinned_or_counted_twice() {
        FIN_COUNT.with(|count| count.set(0));
        let count = Box::new(Cell::new(-1));
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        // SAFETY: count outlives the heap and is written only by it.
        unsafe { fx.heap.attach_live_count(fx.leaf, NonNull::from(&*count)) };
        let anchor = fx.heap.alloc(fx.leaf, 16);
        let hole = fx.heap.alloc(fx.leaf, 16);
        let zombie = fx.heap.alloc(fx.leaf, 16);
        fx.roots[0] = tagged(anchor);
        assert!(fx.start_marking(&no_edges));
        fx.heap.finish_collection(&[zombie as usize]);
        let slot = page_slot(&fx.heap, anchor as usize);
        assert_eq!(checked_free_chain(fx.heap.page_view(slot)), [1]);
        assert_eq!(fx.heap.page_view(slot).live, 2);
        assert_eq!((count.get(), FIN_COUNT.with(Cell::get)), (1, 2));
        let reused = fx.heap.alloc(fx.leaf, 16);
        let virgin = fx.heap.alloc(fx.leaf, 16);
        assert_eq!(reused, hole);
        assert_ne!(reused, zombie);
        assert_eq!(virgin as usize, anchor as usize + 48);
        assert_zeroed_cell(reused, 16);
        assert_zeroed_cell(virgin, 16);
        fx.roots[1] = tagged(reused);
        fx.roots[2] = tagged(virgin);
        for _ in 0..2 {
            assert!(fx.start_marking(&no_edges));
            fx.heap.finish_collection(&[zombie as usize]);
            assert_eq!((count.get(), FIN_COUNT.with(Cell::get)), (3, 2));
            let page = fx.heap.page_view(slot);
            assert_eq!(page.live, 4);
            assert!(checked_free_chain(page).is_empty());
        }
        fx.collect();
        assert!(!fx.heap.is_allocated_word(zombie as u64));
        assert_eq!(checked_free_chain(fx.heap.page_view(slot)), [2]);
        assert_eq!((count.get(), FIN_COUNT.with(Cell::get)), (3, 2));
        assert_eq!(fx.heap.page_view(slot).live, 3);
        fx.roots.fill(0);
        fx.collect();
        assert_eq!((count.get(), FIN_COUNT.with(Cell::get)), (0, 5));
        assert_eq!(fx.heap.stats().pages_live, 0);
    }

    #[test]
    fn old_to_young_cards_cover_recycled_and_virgin_allocations() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let holder = fx.heap.alloc(fx.carded, 16);
        let dead = fx.heap.alloc(fx.carded, 16);
        fx.roots[0] = tagged(holder);
        fx.collect();
        let recycled = fx.heap.alloc(fx.carded, 16);
        assert_eq!(recycled, dead);
        let first_leaf = fx.heap.alloc(fx.leaf, 16);
        set_word(recycled, 0, tagged(first_leaf));
        set_word(holder, 0, tagged(recycled));
        // SAFETY: holder is an old, live cell of a carded descriptor.
        unsafe { card_mark(holder) };
        assert_eq!(fx.heap.cell_age(recycled as u64), Some(CellAge::Young));
        fx.collect_minor();
        assert!(fx.heap.stats().last_card_cells > 0);
        assert!(fx.heap.owns_cell(recycled as usize, fx.carded));
        assert!(fx.heap.owns_cell(first_leaf as usize, fx.leaf));
        let virgin = fx.heap.alloc(fx.carded, 16);
        assert_eq!(virgin as usize, holder as usize + 32);
        let second_leaf = fx.heap.alloc(fx.leaf, 16);
        set_word(virgin, 0, tagged(second_leaf));
        set_word(recycled, 0, tagged(virgin));
        // SAFETY: recycled survived the previous collection and is now old.
        unsafe { card_mark(recycled) };
        assert_eq!(fx.heap.cell_age(virgin as u64), Some(CellAge::Young));
        fx.collect_minor();
        assert!(fx.heap.stats().last_card_cells > 0);
        assert!(fx.heap.owns_cell(virgin as usize, fx.carded));
        assert!(fx.heap.owns_cell(second_leaf as usize, fx.leaf));
        check_free_chains(&mut fx.heap);
        fx.collect();
        assert!(!fx.heap.is_allocated_word(first_leaf as u64));
        assert!(fx.heap.owns_cell(second_leaf as usize, fx.leaf));
        check_free_chains(&mut fx.heap);
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
    fn edge_sources_trace_side_tables_through_their_owners() {
        let mut fx = Fixture::new();
        // Map-like side table: (owner, value), reported through the client's
        // EdgeSink and pruned between the phases, as the engine prunes its own.
        let table: RefCell<Vec<(usize, u64)>> = RefCell::default();
        let edges = |sink: &mut EdgeSink<'_>| {
            for &(owner, value) in table.borrow().iter() {
                sink.edge(owner as *const u8, value);
            }
        };
        let collect = |fx: &mut Fixture| {
            assert!(fx.start_marking(&edges));
            table
                .borrow_mut()
                .retain(|&(owner, _)| !fx.heap.is_dead_target(owner));
            fx.heap.finish_collection(&[]);
        };
        let owner = fx.alloc_node(0, 1);
        // The value points back at its owner: a cycle through the table.
        let value = fx.alloc_node(tagged(owner), 2);
        table.borrow_mut().push((owner as usize, tagged(value)));
        fx.roots[0] = tagged(owner);
        collect(&mut fx);
        assert_eq!(fx.live(), 2);
        fx.roots[0] = 0;
        collect(&mut fx);
        assert_eq!(fx.live(), 0, "a cycle through a side table is collected");
        assert!(table.borrow().is_empty());
    }

    #[test]
    fn source_ephemerons_need_both_owner_and_key() {
        let mut fx = Fixture::new();
        // WeakMap-like side table: (owner, key, value), reported through the
        // client's EdgeSink.
        let table: RefCell<Vec<(usize, u64, u64)>> = RefCell::default();
        let ephemerons = |sink: &mut EdgeSink<'_>| {
            for &(owner, key, value) in table.borrow().iter() {
                sink.ephemeron(owner as *const u8, key, value);
            }
        };
        let [map, key, value, dead_map, key2, value2] = [0; 6].map(|_| fx.heap.alloc(fx.leaf, 16));
        table.borrow_mut().extend([
            (map as usize, tagged(key), tagged(value)),
            (dead_map as usize, tagged(key2), tagged(value2)),
        ]);
        for (slot, cell) in [map, key, key2].into_iter().enumerate() {
            fx.roots[slot] = tagged(cell);
        }
        fx.collect_with(&ephemerons);
        assert!(fx.heap.is_allocated_word(value as u64), "live map and key");
        assert!(!fx.heap.is_allocated_word(value2 as u64), "dead map");
        fx.roots[1] = 0;
        table.borrow_mut().truncate(1);
        fx.collect_with(&ephemerons);
        assert!(!fx.heap.is_allocated_word(value as u64), "dead key");
    }

    /// A side table still holding a freed cell (a client bug) must not hang
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
        fx.collect_with(&|sink| sink.ephemeron(std::ptr::null(), tagged(anchor), freed));
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
        assert!(fx.heap.start_marking(false, &client, CollectionKind::Major));
        let dead = fx.heap.dead_cells(fx.leaf);
        assert_eq!(dead.len(), 2);
        assert!(dead.contains(&dead_owner) && dead.contains(&dead_value));
        assert!(
            fx.heap.is_dead_target(dead_value + 8),
            "interior addresses resolve"
        );
        assert!(!fx.heap.is_dead_target(value));
        fx.heap.finish_collection(&[]);
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
        assert!(fx.heap.start_marking(false, &client, CollectionKind::Major));
        let small = fx.heap.alloc(fx.leaf, 16) as usize;
        let large = fx.heap.alloc(fx.leaf, 8192) as usize;
        assert_eq!(fx.heap.dead_cells(fx.leaf), vec![dead]);
        fx.heap.finish_collection(&[]);
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
    /// the return address. System V convention on every OS, like `zero_below_sp`.
    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "sysv64" fn fill_below_sp(bytes: usize, marker: u64) {
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
    /// return address of this call. System V convention on every OS.
    #[cfg(target_arch = "x86_64")]
    #[unsafe(naked)]
    unsafe extern "sysv64" fn nonzero_below_sp(bytes: usize) -> usize {
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
        assert!(fx.start_marking(&no_edges));
        assert_eq!(fx.heap.dead_cells(fx.leaf), vec![pinned]);
        fx.heap.finish_collection(&[pinned]);
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
        assert!(fx.start_marking(&no_edges));
        assert!(
            fx.heap.dead_cells(fx.leaf).len() >= 14,
            "zombies are not reported as newly dead"
        );
        assert!(!fx.heap.dead_cells(fx.leaf).contains(&pinned));
        assert!(fx.heap.is_dead_target(pinned));
        fx.heap.finish_collection(&[]);
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

    /// Pins are recorded by exact start in the pinned cells' pages (a flag on a large cell)
    /// and cleared by the sweep: a pinned dead small cell and a pinned dead large cell stay
    /// reserved as zombies, finalized once, through every collection that pins them, and the
    /// first collection after their pins go frees them. An address inside a dead cell, a dead
    /// cell's tagged word, a free cell's start and a live cell's start pin nothing.
    #[test]
    fn pins_keep_small_and_large_cells_reserved_by_exact_start() {
        FIN_COUNT.with(|c| c.set(0));
        let mut fx = Fixture::new();
        let live = fx.heap.alloc(fx.leaf, 16) as usize;
        let small = fx.heap.alloc(fx.leaf, 16) as usize;
        let inside_small = fx.heap.alloc(fx.leaf, 16) as usize;
        let tagged_small = fx.heap.alloc(fx.leaf, 16) as usize;
        let large = fx.heap.alloc(fx.leaf, 8192) as usize;
        let inside_large = fx.heap.alloc(fx.leaf, 8192) as usize;
        // A fresh page hands out its cells in order: one 64 cells on is free.
        let free = small + 64 * 16;
        assert!(!fx.heap.is_allocated_word(free as u64));
        fx.roots[0] = tagged(live as *const u8);
        let pins = [
            small,
            large,
            inside_small + 8,
            inside_large + 8,
            tagged(tagged_small as *const u8) as usize,
            free,
            live,
        ];
        assert!(fx.start_marking(&no_edges));
        fx.heap.finish_collection(&pins);
        assert_eq!(FIN_COUNT.with(|c| c.get()), 5, "every dead cell finalized");
        assert!(fx.heap.is_allocated_word(small as u64), "pinned small cell");
        assert!(fx.heap.is_allocated_word(large as u64), "pinned large cell");
        for (addr, what) in [
            (inside_small, "a small cell pinned inside"),
            (inside_large, "a large cell pinned inside"),
            (tagged_small, "a cell pinned by its tagged word"),
            (free, "a free cell"),
        ] {
            assert!(!fx.heap.is_allocated_word(addr as u64), "{what} is freed");
        }
        assert_eq!(fx.heap.large_cells(), 1);
        assert_eq!(fx.live(), 3, "the live cell and the two zombies");

        // Pinned again, the zombies stay, and nothing is finalized twice.
        assert!(fx.start_marking(&no_edges));
        assert!(
            fx.heap.dead_cells(fx.leaf).is_empty(),
            "zombies are not newly dead"
        );
        fx.heap.finish_collection(&[small, large]);
        assert!(fx.heap.is_allocated_word(small as u64));
        assert!(fx.heap.is_allocated_word(large as u64));
        assert_eq!(FIN_COUNT.with(|c| c.get()), 5);

        // Once their pins go, the next collection frees them; the live cell, unrooted, dies
        // with no pin left over from the first collection.
        fx.roots[0] = 0;
        assert!(fx.start_marking(&no_edges));
        fx.heap.finish_collection(&[]);
        for addr in [small, large, live] {
            assert!(!fx.heap.is_allocated_word(addr as u64), "{addr:#x} freed");
        }
        assert_eq!(FIN_COUNT.with(|c| c.get()), 6, "only the live cell is new");
        assert_eq!(fx.heap.large_cells(), 0);
        assert_eq!(fx.live(), 0);
    }

    /// Between the phases, `is_dead_small_cell` answers for every address what `dead_cells`
    /// lists, for a descriptor whose cells are all small: over every byte of the heap's pages
    /// (live, dead, zombie, free and black-allocated cells of two size classes, one of which
    /// leaves a tail in its page, and cells of another descriptor), every byte around a dead and
    /// a live large cell of the other descriptor, the dead cells' tagged words and addresses
    /// outside the heap.
    #[test]
    fn dead_small_cells_answer_as_the_dead_cells_list_for_every_address() {
        let mut fx = Fixture::new();
        // 16-byte cells, and 48-byte ones, whose page ends in a 16-byte tail.
        let small: Vec<usize> = (0..12)
            .map(|_| fx.heap.alloc(fx.leaf, 16) as usize)
            .collect();
        let mid: Vec<usize> = (0..12)
            .map(|_| fx.heap.alloc(fx.leaf, 48) as usize)
            .collect();
        let others: Vec<usize> = (0..6).map(|i| fx.alloc_node(0, i) as usize).collect();
        let large_dead = fx.heap.alloc(fx.node, 8192) as usize;
        let large_live = fx.heap.alloc(fx.node, 8192) as usize;
        let root = |fx: &mut Fixture, cells: &[usize]| {
            fx.roots.fill(0);
            for (slot, &cell) in cells.iter().enumerate() {
                fx.roots[slot] = tagged(cell as *const u8);
            }
        };

        // The first collection frees two cells and keeps a pinned one as a zombie.
        let mut kept: Vec<usize> = small[..10].to_vec();
        kept.extend(&mid[..11]);
        kept.extend(&others);
        kept.extend([large_dead, large_live]);
        root(&mut fx, &kept);
        assert!(fx.start_marking(&no_edges));
        fx.heap.finish_collection(&[mid[11]]);
        let (free, zombie) = ([small[10], small[11]], mid[11]);

        // The second leaves half of each kind dead, and allocates a cell black meanwhile.
        let mut kept: Vec<usize> = small[..5].to_vec();
        kept.extend(&mid[..5]);
        kept.extend(&others[..3]);
        kept.push(large_live);
        root(&mut fx, &kept);
        assert!(fx.start_marking(&no_edges));
        let black = fx.heap.alloc(fx.leaf, 16) as usize;
        let listed: crate::fasthash::FastSet<usize> =
            fx.heap.dead_cells(fx.leaf).into_iter().collect();
        let mut expected: Vec<usize> = small[5..10].to_vec();
        expected.extend(&mid[5..11]);
        let expected: crate::fasthash::FastSet<usize> = expected.into_iter().collect();
        assert_eq!(listed, expected);
        let agrees = |addr: usize| {
            assert_eq!(
                fx.heap.is_dead_small_cell(addr, fx.leaf),
                listed.contains(&addr),
                "{addr:#x}"
            );
        };
        let bases: Vec<usize> = fx.heap.pages.iter().flatten().map(|p| p.base).collect();
        assert_eq!(bases.len(), 3, "a page of each kind of small cell");
        for base in bases {
            (base..base + PAGE_SIZE).for_each(&agrees);
        }
        for large in [large_dead, large_live] {
            (large - 64..large + 8192 + 64).for_each(&agrees);
        }
        for &cell in listed.iter().chain(&[free[0], zombie, black, others[3]]) {
            agrees(tagged(cell as *const u8) as usize);
        }
        let top = 0u64;
        for addr in [0, 8, 4096, usize::MAX, std::ptr::from_ref(&top) as usize] {
            agrees(addr);
        }
        // The answers above are not vacuous: each kind of cell answers as it should.
        let dead = |addr: usize| fx.heap.is_dead_small_cell(addr, fx.leaf);
        assert!(dead(small[5]) && dead(mid[10]), "dead");
        assert!(!dead(small[0]) && !dead(mid[0]), "live");
        assert!(!dead(free[0]) && !dead(free[1]), "free");
        assert!(!dead(zombie), "a zombie");
        assert!(!dead(black), "allocated black");
        assert!(
            !dead(small[5] + 8) && !dead(mid[5] + 16),
            "inside a dead cell"
        );
        assert!(!dead(others[3]), "another descriptor's dead cell");
        assert!(fx.heap.is_dead_small_cell(others[3], fx.node));
        // A large cell is never consulted: a dead one's start answers false although its
        // descriptor's list holds it.
        assert!(fx.heap.dead_cells(fx.node).contains(&large_dead));
        assert!(!fx.heap.is_dead_small_cell(large_dead, fx.node));
        fx.heap.finish_collection(&[]);
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
        assert!(fx
            .heap
            .start_marking(false, &NoClient, CollectionKind::Major));
        {
            let _scope = fx.heap.nogc();
            assert!(!fx
                .heap
                .start_marking(false, &NoClient, CollectionKind::Major));
        }
        fx.heap.finish_collection(&[]);
        assert!(due(&fx) && fx.heap.collection_pending());
        fx.collect();
        assert!(!due(&fx));
    }

    /// Descriptor accounting follows allocations and newly finalized cells, independently of
    /// whether their storage is retained for weak pins or whether the collection is minor.
    #[test]
    fn live_cell_count_tracks_pages_large_cells_zombies_and_teardown() {
        // The heap's raw counter pointer stays valid through its Drop, as in GcState.
        let count = Box::new(Cell::new(-1));
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        fx.heap.set_nursery_bytes(Some(1024 * 1024));
        // SAFETY: `count` outlives the heap and no other code writes it.
        unsafe { fx.heap.attach_live_count(fx.leaf, NonNull::from(&*count)) };
        let small = fx.heap.alloc(fx.leaf, 16);
        let large = fx.heap.alloc(fx.leaf, LARGEST_SMALL_CELL + 16);
        fx.alloc_node(0, 7); // another descriptor never contributes
        for _ in 0..4100 {
            fx.heap.alloc(fx.leaf, 16);
        }
        assert_eq!(count.get(), 4102);
        fx.roots[0] = small as u64;
        fx.roots[1] = large as u64;
        fx.collect();
        assert_eq!(count.get(), 2);

        // An old object without a root stays counted through a minor collection; both kinds
        // of new dead cell are decremented once even when their memory becomes a zombie.
        fx.roots.fill(0);
        let young_small = fx.heap.alloc(fx.leaf, 16);
        let young_large = fx.heap.alloc(fx.leaf, LARGEST_SMALL_CELL + 32);
        assert_eq!(count.get(), 4);
        assert!(fx.start_marking_kind(&no_edges, CollectionKind::Minor));
        // A cell allocated between phases is black and counted immediately.
        let black = fx.heap.alloc(fx.leaf, 16);
        assert_eq!(count.get(), 5);
        fx.heap
            .finish_collection(&[young_small as usize, young_large as usize]);
        assert_eq!(count.get(), 3);
        assert_eq!(fx.heap.cell_age(young_small as u64), Some(CellAge::Dead));
        assert_eq!(fx.heap.cell_age(young_large as u64), Some(CellAge::Dead));
        assert_eq!(fx.heap.live_count(fx.leaf), 3);

        // Unpinning the zombies frees storage without another death. Old live cells survive
        // the minor collection; the following major collection drops their count to zero.
        fx.collect_minor();
        assert_eq!(count.get(), 3);
        // A zombie on a page with no new allocations waits for a major sweep, and so does an
        // old large zombie: a minor collection sweeps only the large cells allocated since the
        // last one. Neither kind is included in the published live count.
        assert!(fx.heap.is_allocated_word(young_small as u64));
        assert!(fx.heap.is_allocated_word(young_large as u64));
        assert!(fx.heap.is_allocated_word(black as u64));
        fx.collect();
        assert_eq!(count.get(), 0);
        fx.heap.alloc(fx.leaf, 16);
        fx.heap.alloc(fx.leaf, LARGEST_SMALL_CELL + 64);
        assert_eq!(count.get(), 2);
        fx.heap.release_all();
        assert_eq!(count.get(), 0);
        fx.heap.release_all();
        assert_eq!(count.get(), 0);
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

    /// A precise edge from a small cell to a large one (in the engine, an object property's long
    /// string), which resolves outside the pages: the large cell, and the small cell it
    /// references in turn, live while the edge does and die when it goes.
    #[test]
    fn a_large_cell_lives_and_dies_with_its_edge_from_a_small_cell() {
        let mut fx = Fixture::new();
        let small = fx.alloc_node(0, 1);
        let large = fx.heap.alloc(fx.node, 8192);
        let leaf = fx.heap.alloc(fx.leaf, 16);
        set_word(large, 0, tagged(leaf));
        set_word(large, 1, 42);
        set_word(small, 0, tagged(large));
        fx.roots[0] = tagged(small);
        fx.collect();
        assert_eq!(fx.live(), 3);
        assert!(fx.heap.is_allocated_word(large as u64));
        assert!(fx.heap.is_allocated_word(leaf as u64));
        assert_eq!((word(large, 0), word(large, 1)), (tagged(leaf), 42));
        set_word(small, 0, 0);
        fx.collect();
        assert_eq!(fx.live(), 1, "only the rooted small cell");
        assert!(!fx.heap.is_allocated_word(large as u64));
        assert!(!fx.heap.is_allocated_word(leaf as u64));
        assert_eq!(fx.heap.stats().cells_swept, 2);
    }

    #[test]
    fn stress_graph_survives_collect_every_alloc() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        fx.heap.set_stress_every(1);
        // Build a 200-node chain under stress, keeping every intermediate
        // reachable through the chain. The allocation-triggered collections
        // find the head in this frame; the final precise one in roots[0].
        let mut head: *mut u8 = fx.alloc_node(0, 0);
        for i in 1..200u64 {
            head = fx.alloc_node(tagged(head), i);
            let _garbage = fx.heap.alloc(fx.leaf, 16);
        }
        assert!(fx.heap.stats().collections >= 398);
        fx.roots[0] = tagged(head);
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

    #[cfg(target_os = "linux")]
    thread_local! {
        static HELD: Cell<u64> = const { Cell::new(0) };
    }

    /// Store a hidden pointer, unhidden, in this thread's `HELD`.
    #[cfg(target_os = "linux")]
    #[inline(never)]
    fn hold_in_thread_local(hidden: u64) {
        HELD.with(|held| unsafe { std::ptr::write_volatile(held.as_ptr(), !hidden) });
    }

    /// A spawned thread's descriptor and static TLS sit at the top of its stack mapping (glibc,
    /// musl), above its outermost frame: the registered stack ends below them, so a cell that
    /// only a thread-local holds dies at a conservative collection.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_thread_local_is_not_a_root() {
        std::thread::spawn(|| {
            let outer = 0u64;
            assert!(register_thread_native_stack());
            let top = native::entry_top().expect("registered");
            let held = HELD.with(|held| held.as_ptr() as usize);
            assert!(
                (&raw const outer as usize) < top && top <= held,
                "the stack ends between the thread's frames and its thread-locals"
            );
            let mut fx = Fixture::new();
            let hidden = alloc_hidden(&mut fx.heap, fx.leaf, 16);
            hold_in_thread_local(hidden);
            scrub_stack();
            assert!(fx.heap.collect());
            assert!(
                !fx.heap.is_allocated_word(HELD.with(Cell::get)),
                "a thread-local retained its cell"
            );
            black_box(&outer);
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

    /// Collect conservatively with `words` spilled to this frame.
    #[inline(never)]
    fn collect_with_words_in_frame(heap: &mut Heap, words: [u64; 4]) -> bool {
        let mut spill = [0u64; 4];
        for (slot, word) in spill.iter_mut().zip(words) {
            unsafe { std::ptr::write_volatile(slot, word) };
        }
        black_box(&spill);
        let ran = heap.collect();
        black_box(&spill);
        ran
    }

    /// The stack scan reads no word its own frame derived from the heap: the large cells'
    /// span starts at the lowest large cell's address, and a scan whose frame held the span
    /// among the words it reads (marking inlined into the scan, in debug builds) kept that
    /// cell alive, dead or not; so did a frame the roots' marking left where the scan's frame
    /// went. A stale word elsewhere on the stack (another test's, on a reused thread stack)
    /// may still keep it: only a source in the scan's own frame is the scan's.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn the_stack_scan_reads_no_word_its_own_frame_derived() {
        // Boxed, as an engine's heap is: a heap on the stack would hold the span itself.
        let mut fx = Box::new(Fixture::new());
        fx.heap.set_collect_on_alloc(false);
        fx.heap.set_retention_tracing(true);
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        // Hidden cells: only their addresses' complements are on the stack. The lowest, where
        // the large cells' span starts, is dead; another is rooted.
        let hidden: Vec<u64> = (0..3)
            .map(|_| alloc_hidden(&mut fx.heap, fx.leaf, 8192))
            .collect();
        let lowest = *hidden.iter().max().expect("cells");
        let rooted = *hidden.iter().find(|&&cell| cell != lowest).expect("cells");
        fx.roots[0] = tagged(!rooted as *mut u8);
        crate::gc_core::scrub_stack(64 * 1024);
        assert!(fx.heap.collect());
        // The scan's frame lies below its frame record, the first it walked.
        let (record, _) = fx.heap.last_frames[0];
        let path = fx.heap.retention_path((!lowest) as usize);
        if let Some(&(_, _, MarkSource::Stack(slot))) = path.first() {
            assert!(
                slot >= record,
                "the scan's own frame kept the dead large cell: word {slot:#x} below {record:#x}, \
                 {} bytes below it; frames {:x?}",
                record - slot,
                &fx.heap.last_frames[..fx.heap.last_frames.len().min(6)]
            );
        }
        assert!(fx.heap.owns_cell(!rooted as usize, fx.leaf));
        black_box(&top);
    }

    /// Conservative words resolve through the page table as they did through the hash map: a
    /// tagged interior address of a small cell finds the cell, while an address in a released
    /// (spare) page, one between the heap's pages and one in a 4 GiB range holding no page
    /// find nothing, whether held on the native stack or reported as roots. Only the cell is
    /// marked, and the spare page stays spare.
    #[test]
    fn conservative_words_find_cells_in_live_pages_only() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        let per_page = PAGE_SIZE / 16;
        let cells: Vec<*mut u8> = (0..per_page * 2)
            .map(|i| fx.alloc_node(0, i as u64))
            .collect();
        let (kept, released) = (cells[3], cells[per_page + 5]);
        let page_of = |p: usize| p & !(PAGE_SIZE - 1);
        let (live_base, spare_base) = (page_of(kept as usize), page_of(released as usize));
        fx.roots[0] = tagged(kept);
        fx.collect();
        assert_eq!(fx.heap.spare_pages, [spare_base]);
        // A 64 KiB range beside the live page that no page occupies, in the leaf holding it.
        let between = [PAGE_SIZE, 2 * PAGE_SIZE]
            .into_iter()
            .flat_map(|offset| [live_base + offset, live_base.wrapping_sub(offset)])
            .find(|&base| base != spare_base && base >> 32 == live_base >> 32)
            .expect("a free range beside the live page");
        let words = [
            tagged(kept) + 8,
            released as u64 + 8,
            between as u64 + 0x88,
            (live_base ^ (1 << 40)) as u64 + 0x88,
        ];
        assert_eq!(
            words.map(|word| fx.heap.resolve(word)),
            [Some(kept as *const u8), None, None, None]
        );
        let check = |fx: &Fixture, marked_before: u64| {
            let stats = fx.heap.stats();
            assert_eq!(
                (stats.cells_marked - marked_before, stats.live_cells),
                (1, 1)
            );
            assert_eq!(fx.heap.spare_pages, [spare_base]);
            assert_eq!(fx.val(kept), 3);
        };
        let marked = fx.heap.stats().cells_marked;
        assert!(collect_with_words_in_frame(&mut fx.heap, words));
        check(&fx, marked);
        fx.roots[..4].copy_from_slice(&words);
        let marked = fx.heap.stats().cells_marked;
        fx.collect();
        check(&fx, marked);
        black_box((&top, &cells));
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

    #[cfg(target_arch = "x86_64")]
    extern "C" fn collect_trampoline(heap: *mut std::ffi::c_void) {
        let heap = unsafe { &mut *(heap as *mut Heap) };
        assert!(heap.collect());
    }

    /// Collect while the only reference sits in xmm6, a register Win64 makes callee-saved and
    /// no Rust frame on the collection path saves: only the register spill can find it.
    /// Returns xmm6's low half after the call.
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    #[inline(never)]
    fn collect_with_reference_only_in_xmm6(heap: &mut Heap, hidden: u64) -> u64 {
        let found: u64;
        unsafe {
            std::arch::asm!(
                "mov rax, {hidden}",
                "not rax",
                "movq xmm6, rax",
                "sub rsp, 32", // the callee's shadow space
                "call {trampoline}",
                "add rsp, 32",
                "movq rax, xmm6",
                hidden = in(reg) hidden,
                trampoline = sym collect_trampoline,
                in("rcx") heap as *mut Heap as *mut std::ffi::c_void,
                lateout("rax") found,
                out("xmm6") _,
                clobber_abi("C"),
            );
        }
        found
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    fn reference_only_in_xmm6_is_found() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        let hidden = alloc_hidden(&mut fx.heap, fx.leaf, 16);
        scrub_stack();
        let found = collect_with_reference_only_in_xmm6(&mut fx.heap, hidden);
        assert_eq!(found, !hidden, "xmm6 is callee-saved");
        assert_eq!(fx.heap.stats().collections, 1);
        assert!(
            fx.heap.is_allocated_word(found),
            "register spill missed xmm6"
        );
        black_box(&top);
    }

    /// Collect while the only reference sits in r15, which System V makes callee-saved. A frame
    /// on the collection path that uses r15 saves it where the stack scan finds it; where none
    /// does, as in debug builds, only the register spill can. Returns r15 after the call.
    #[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
    #[inline(never)]
    fn collect_with_reference_only_in_r15(heap: &mut Heap, hidden: u64) -> u64 {
        let found: u64;
        unsafe {
            std::arch::asm!(
                "mov rax, {hidden}",
                "not rax",
                "mov r15, rax",
                "call {trampoline}",
                "mov rax, r15",
                hidden = in(reg) hidden,
                trampoline = sym collect_trampoline,
                in("rdi") heap as *mut Heap as *mut std::ffi::c_void,
                lateout("rax") found,
                out("r15") _,
                clobber_abi("C"),
            );
        }
        found
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", not(target_os = "windows")))]
    fn reference_only_in_r15_is_found() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        let hidden = alloc_hidden(&mut fx.heap, fx.leaf, 16);
        scrub_stack();
        let found = collect_with_reference_only_in_r15(&mut fx.heap, hidden);
        assert_eq!(found, !hidden, "r15 is callee-saved");
        assert_eq!(fx.heap.stats().collections, 1);
        assert!(
            fx.heap.is_allocated_word(found),
            "register spill missed r15"
        );
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

    /// The reservation holding `addr`: its allocation base, and the end of the last region
    /// with that base (VirtualQuery, independent of the stack bounds the scanner uses).
    #[cfg(target_os = "windows")]
    fn allocation_containing(addr: usize) -> Option<(usize, usize)> {
        #[repr(C)]
        struct MemoryBasicInformation {
            base_address: usize,
            allocation_base: usize,
            allocation_protect: u32,
            partition_id: u16,
            region_size: usize,
            state: u32,
            protect: u32,
            kind: u32,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn VirtualQuery(
                address: *const std::ffi::c_void,
                buffer: *mut MemoryBasicInformation,
                length: usize,
            ) -> usize;
        }
        let query = |at: usize| {
            let mut info = std::mem::MaybeUninit::<MemoryBasicInformation>::uninit();
            let size = std::mem::size_of::<MemoryBasicInformation>();
            // SAFETY: VirtualQuery writes at most `size` bytes into `info`.
            let written = unsafe { VirtualQuery(at as *const _, info.as_mut_ptr(), size) };
            // SAFETY: a nonzero return means `info` was filled.
            (written != 0).then(|| unsafe { info.assume_init() })
        };
        let base = query(addr)?.allocation_base;
        let mut end = base;
        while let Some(info) = query(end).filter(|info| info.allocation_base == base) {
            end = info.base_address + info.region_size;
        }
        Some((base, end))
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn registered_segment_bounds_match_the_allocation() {
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        for size in [64 * 1024, 100_000, 1 << 20] {
            let (region, allocation) = grow(size, || {
                let sp = native::current_sp();
                (native::innermost_region(), allocation_containing(sp))
            });
            assert!(region.is_some());
            assert_eq!(region, allocation, "segment of {size} bytes");
        }
        black_box(&top);
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
        // The allocation-triggered collections find the chain's head in this frame.
        let head = fx.build_chain(1 << 18); // 4 MiB live
        let before = fx.heap.stats().collections;
        for _ in 0..1 << 19 {
            black_box(fx.heap.alloc(fx.leaf, 16)); // 8 MiB of garbage
        }
        let during = fx.heap.stats().collections - before;
        assert!(
            during <= 3,
            "8 MiB churn over a 4 MiB live heap took {during} collections"
        );
        black_box((&top, &head));
    }

    #[test]
    fn retained_large_allocations_do_not_collect_every_time() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        // Retained in this frame, where the allocation-triggered collections find them.
        let mut held = [0u64; 16];
        for slot in &mut held {
            *slot = tagged(fx.heap.alloc(fx.leaf, 2 << 20));
        }
        let collections = fx.heap.stats().collections;
        assert!(
            collections <= 6,
            "16 retained 2 MiB allocations took {collections} collections"
        );
        black_box((&top, &held));
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

    /// The page table maps page numbers in the first, a middle and the last leaf, at either end
    /// of a leaf; a removed or cleared entry is gone, a page number can be mapped to another
    /// slot (a spare page reused) and a slot to another page number (a slot reused).
    #[test]
    fn page_table_maps_inserts_until_removed_or_cleared() {
        let mut table = PageTable::new();
        // Leaves 0, 1, 0x1234 and 0xffff; slot i for pages[i].
        let pages = [
            0,
            1,
            0xffff,
            0x1_0000,
            0x1234_5678,
            0xffff_0000,
            0xffff_ffff,
        ];
        for page in pages {
            assert_eq!(table.get(page), None, "{page:#x} before");
        }
        for (slot, &page) in pages.iter().enumerate() {
            assert_eq!(table.insert(page, slot as u32), None, "{page:#x}");
        }
        assert_eq!((table.len(), table.leaves.len()), (pages.len(), 4));
        for (slot, &page) in pages.iter().enumerate() {
            assert_eq!(table.get(page), Some(slot as u32), "{page:#x}");
        }
        // Neighbors in allocated leaves, pages of unallocated leaves and page numbers above
        // the payload's 32 bits map nothing.
        for page in [
            2,
            0xfffe,
            0x1_0001,
            0x1234_5677,
            0x1234_5679,
            0x2_0000,
            0x7fff_5678,
            0xffff_0001,
            0xffff_fffe,
            1 << 32,
            (1 << 32) | 0x1234_5678,
            usize::MAX,
        ] {
            assert_eq!(table.get(page), None, "{page:#x}");
        }
        assert_eq!(table.insert(0xffff, 9), Some(2), "a page mapped again");
        assert_eq!(table.get(0xffff), Some(9));
        assert_eq!(table.len(), pages.len());
        assert_eq!(table.remove(0x1234_5678), Some(4));
        assert_eq!(table.remove(0x1234_5678), None, "removed already");
        assert_eq!(table.remove(0x5555_5555), None, "an unallocated leaf");
        assert_eq!(table.get(0x1234_5678), None);
        assert_eq!(table.len(), pages.len() - 1);
        // A released page's slot taken by another page, in the same leaf and in another.
        assert_eq!(table.insert(0x1234_9abc, 4), None);
        assert_eq!(table.remove(0), Some(0));
        assert_eq!(table.insert(0x8000_0000, 0), None);
        assert_eq!(
            [0x1234_5678, 0x1234_9abc, 0, 0x8000_0000].map(|page| table.get(page)),
            [None, Some(4), None, Some(0)]
        );
        assert_eq!((table.len(), table.leaves.len()), (pages.len(), 5));
        table.clear();
        assert_eq!((table.len(), table.leaves.len()), (0, 0));
        assert!(
            table.top.iter().all(|leaf| leaf.is_null()),
            "leaves unmapped"
        );
        for page in pages.into_iter().chain([0x1234_9abc, 0x8000_0000]) {
            assert_eq!(table.get(page), None, "{page:#x} after clear");
        }
        assert_eq!(table.insert(0xffff_ffff, 3), None, "usable after clear");
        assert_eq!(table.get(0xffff_ffff), Some(3));
        assert_eq!((table.len(), table.leaves.len()), (1, 1));
    }

    /// The heap's page table follows its pages: a new page is mapped to its slot, a released
    /// one (kept as a spare) is not, a reused spare page is mapped to the slot it gets, and the
    /// heap's release unmaps every page and frees the leaves.
    #[test]
    fn page_table_follows_pages_through_release_and_reuse() {
        let mut fx = Fixture::new();
        let per_page = PAGE_SIZE / 16;
        let page_of = |p: *mut u8| p as usize & !(PAGE_SIZE - 1);
        let cells: Vec<*mut u8> = (0..per_page * 3)
            .map(|i| fx.alloc_node(0, i as u64))
            .collect();
        let bases: [usize; 3] = std::array::from_fn(|i| page_of(cells[i * per_page]));
        let mapped = |fx: &Fixture| bases.map(|base| fx.heap.page_table.get(base >> PAGE_SHIFT));
        assert_eq!(mapped(&fx), [Some(0), Some(1), Some(2)]);
        fx.heap.check_page_table();
        // The middle page keeps a cell; the others are released to the spare pool.
        fx.roots[0] = tagged(cells[per_page]);
        fx.collect();
        fx.heap.check_page_table();
        assert_eq!(fx.heap.page_table.len(), 1);
        assert_eq!(fx.heap.spare_pages, [bases[0], bases[2]]);
        assert_eq!(mapped(&fx), [None, Some(1), None]);
        // A page of another descriptor and class takes the last released slot and spare page.
        let reused = fx.heap.alloc(fx.leaf, 32);
        assert_eq!(page_of(reused), bases[2]);
        assert_eq!(mapped(&fx), [None, Some(1), Some(2)]);
        assert_eq!(
            fx.heap.resolve(reused as u64 + 24),
            Some(reused as *const u8)
        );
        fx.heap.check_page_table();
        fx.heap.release_all();
        fx.heap.check_page_table();
        assert_eq!(
            (fx.heap.page_table.len(), fx.heap.page_table.leaves.len()),
            (0, 0)
        );
        assert_eq!(mapped(&fx), [None, None, None]);
    }

    /// The cell of the heap starting at or containing `addr`: whether it is marked.
    fn is_marked_at(heap: &Heap, addr: usize) -> bool {
        heap.locate(addr as u64)
            .is_some_and(|cell| heap.is_marked(&cell))
    }

    /// Fill a fresh page of `desc` with cells of `size` bytes: the cells' starts, the page's
    /// first.
    fn fill_page(fx: &mut Fixture, desc: DescId, size: usize) -> Vec<usize> {
        let first = fx.heap.alloc(desc, size) as usize;
        assert_eq!(first % PAGE_SIZE, 0, "a new page's first cell starts it");
        let cells = fx
            .heap
            .pages
            .iter()
            .flatten()
            .find(|p| p.base == first)
            .unwrap()
            .cells;
        let mut all = vec![first];
        all.extend((1..cells).map(|_| fx.heap.alloc(desc, size) as usize));
        assert!(all.iter().all(|&cell| cell & !(PAGE_SIZE - 1) == first));
        all
    }

    /// A page's bytes.
    fn page_bytes(base: usize) -> Vec<u8> {
        // SAFETY: a live page of the heap, which nothing writes meanwhile.
        unsafe { std::slice::from_raw_parts(base as *const u8, PAGE_SIZE) }.to_vec()
    }

    /// S6 step 10b: a carded kind's pages end in a 512-byte card table, so they hold
    /// `(PAGE_SIZE - 512) / size` cells (338, 508 and 1,354 of 192, 128 and 48 bytes) where an
    /// uncarded kind's hold `PAGE_SIZE / size` (341, 512 and 1,365): the next cell starts a new
    /// page. A fresh page's table is clean.
    #[test]
    fn carded_pages_end_in_a_card_table() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        for (size, carded, full) in [(192, 338, 341), (128, 508, 512), (48, 1354, 1365)] {
            for (desc, cells) in [(fx.carded, carded), (fx.node, full)] {
                let all = fill_page(&mut fx, desc, size);
                let base = all[0];
                assert_eq!(all.len(), cells, "{size}-byte cells");
                let page = fx.heap.pages.iter().flatten().find(|p| p.base == base);
                let page = page.unwrap();
                assert_eq!(page.carded, desc == fx.carded);
                assert!(!page.carded || base + cells * size <= base + CARD_TABLE_OFFSET);
                assert!(
                    (0..cells).all(|i| !page.card_set(i)),
                    "a fresh table is clean"
                );
                let next = fx.heap.alloc(desc, size) as usize;
                assert_eq!(
                    next % PAGE_SIZE,
                    0,
                    "the next {size}-byte cell starts a new page"
                );
                assert_ne!(next, base);
            }
        }
    }

    /// S6 step 10b: no address in a carded page's tail or card table resolves to a cell, whether
    /// asked directly (`resolve`, `is_dead_small_cell`), held on the native stack or reported as
    /// a root: such words mark nothing, while a word in the page's last cell marks that cell.
    #[test]
    fn no_cell_resolves_in_a_card_table() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        // Three carded pages, every cell's card set, each page's last cell rooted.
        let pages: Vec<(usize, usize, usize)> = [192, 128, 48]
            .into_iter()
            .map(|size| {
                let carded = fx.carded;
                let all = fill_page(&mut fx, carded, size);
                for &cell in &all {
                    // SAFETY: a live carded cell.
                    unsafe { card_mark(cell as *const u8) };
                }
                (all[0], size, all.len())
            })
            .collect();
        let last = |page: usize| pages[page].0 + (pages[page].2 - 1) * pages[page].1;
        for page in 0..3 {
            fx.roots[page] = tagged(last(page) as *const u8);
        }
        for &(base, size, cells) in &pages {
            let end = base + cells * size;
            for addr in end..base + PAGE_SIZE {
                assert_eq!(fx.heap.resolve(addr as u64), None, "{addr:#x}");
                assert_eq!(
                    fx.heap.resolve(tagged(addr as *const u8)),
                    None,
                    "{addr:#x}"
                );
            }
            assert_eq!(
                fx.heap.resolve(end as u64 - 1),
                Some((end - size) as *const u8)
            );
        }
        // Between the phases, a dead cell's start answers, but no address of a tail or a table.
        assert!(fx.start_marking(&no_edges));
        for &(base, size, cells) in &pages {
            assert!(fx.heap.is_dead_small_cell(base, fx.carded));
            assert!(!fx
                .heap
                .is_dead_small_cell(base + (cells - 1) * size, fx.carded));
            for addr in base + cells * size..base + PAGE_SIZE {
                assert!(!fx.heap.is_dead_small_cell(addr, fx.carded), "{addr:#x}");
            }
        }
        fx.heap.finish_collection(&[]);
        assert_eq!(fx.live(), 3);
        // Words in each live page's table, and one inside page 0's last cell, as roots of a
        // precise collection beside the other two last cells: only the three cells are marked.
        let table = |page: usize, card: usize| (pages[page].0 + CARD_TABLE_OFFSET + card) as u64;
        let words = [
            table(0, 0),
            table(1, 300),
            tagged(table(2, 511) as *const u8),
            last(0) as u64 + 8,
        ];
        fx.roots[..4].copy_from_slice(&words);
        fx.roots[4] = tagged(last(1) as *const u8);
        fx.roots[5] = tagged(last(2) as *const u8);
        let marked = fx.heap.stats().cells_marked;
        fx.collect();
        assert_eq!(
            fx.heap.stats().cells_marked - marked,
            3,
            "the three cells only"
        );
        assert_eq!(fx.live(), 3);
        // Page 0's table and last cell as stack words, once page 0's last cell is the only cell
        // left: only it is marked.
        fx.roots.fill(0);
        fx.roots[0] = tagged(last(0) as *const u8);
        fx.collect();
        assert_eq!(fx.live(), 1);
        fx.roots.fill(0);
        let words = [
            table(0, 0),
            table(0, 300),
            tagged(table(0, 511) as *const u8),
            last(0) as u64 + 8,
        ];
        let marked = fx.heap.stats().cells_marked;
        assert!(collect_with_words_in_frame(&mut fx.heap, words));
        assert_eq!(
            fx.heap.stats().cells_marked - marked,
            1,
            "only the last cell"
        );
        assert!(fx.heap.is_allocated_word(last(0) as u64));
        black_box(&top);
    }

    /// S6 step 10b: `card_mark` sets the card its cell starts in, by the cell's address alone,
    /// and changes no other byte of the page; the page's metadata then reads the card as set.
    /// A 192- or 128-byte cell has a card of its own; 48-byte cells share theirs.
    #[test]
    fn card_mark_sets_exactly_its_cells_card() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        for size in [192, 128, 48] {
            let carded = fx.carded;
            let all = fill_page(&mut fx, carded, size);
            let base = all[0];
            for (index, &cell) in all.iter().enumerate() {
                let card = CARD_TABLE_OFFSET + ((cell - base) >> CARD_SHIFT);
                let before = page_bytes(base);
                // SAFETY: a live carded cell.
                unsafe { card_mark(cell as *const u8) };
                let after = page_bytes(base);
                assert_ne!(after[card], 0, "{size}-byte cell {index}'s card");
                assert_ne!(
                    after[CARD_SUMMARY], 0,
                    "{size}-byte cell {index}'s page summary"
                );
                assert!(
                    (0..PAGE_SIZE).all(|i| i == card || i == CARD_SUMMARY || before[i] == after[i]),
                    "{size}-byte cell {index} changed another byte"
                );
                let page = fx.heap.pages.iter().flatten().find(|p| p.base == base);
                assert!(page.unwrap().card_set(index));
            }
            // Exactly the cards cells start in are set, and the page's summary, the card of the
            // table's own last bytes, which no cell starts in.
            let bytes = page_bytes(base);
            for card in 0..CARD_TABLE_BYTES {
                let starts = all.iter().any(|&cell| (cell - base) >> CARD_SHIFT == card);
                let summary = CARD_TABLE_OFFSET + card == CARD_SUMMARY;
                assert!(!(starts && summary), "a cell starts in the summary's card");
                assert_eq!(
                    bytes[CARD_TABLE_OFFSET + card] != 0,
                    starts || summary,
                    "card {card}"
                );
            }
            let distinct: std::collections::HashSet<usize> = all
                .iter()
                .map(|&cell| (cell - base) >> CARD_SHIFT)
                .collect();
            assert_eq!(distinct.len() == all.len(), size >= 128, "{size}");
        }
    }

    /// S6 step 10b: a collection clears every card when it starts: between its phases and after
    /// its sweep, every carded page's table is clean, the survivors' included.
    #[test]
    fn cards_clear_when_a_collection_starts() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let cells: Vec<usize> = (0..64)
            .map(|_| fx.heap.alloc(fx.carded, 128) as usize)
            .collect();
        for (slot, &cell) in cells.iter().step_by(2).enumerate() {
            fx.roots[slot] = tagged(cell as *const u8);
        }
        fx.collect();
        for &cell in &cells[..32] {
            // SAFETY: a live carded cell (the odd ones are free now: not marked).
            if fx.heap.is_allocated_word(cell as u64) {
                unsafe { card_mark(cell as *const u8) };
            }
        }
        let base = cells[0] & !(PAGE_SIZE - 1);
        let table_clean = || {
            page_bytes(base)[CARD_TABLE_OFFSET..]
                .iter()
                .all(|&byte| byte == 0)
        };
        assert!(!table_clean());
        assert!(fx.start_marking(&no_edges));
        assert!(table_clean(), "between the phases");
        fx.heap.finish_collection(&[]);
        assert!(table_clean(), "after the sweep");
        assert_eq!(fx.live(), 32);
    }

    /// A page that held a carded kind, its cards set, and is reused for an uncarded kind carries
    /// no table: every handed-out cell is zeroed, reaches the page's end and resolves to itself.
    /// Reused again for a carded kind, its table is clean before its first handout.
    #[test]
    fn a_spare_page_reused_for_another_kind_carries_no_table() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let (carded_desc, leaf) = (fx.carded, fx.leaf);
        let carded = fill_page(&mut fx, carded_desc, 192);
        let base = carded[0];
        for &cell in &carded {
            // SAFETY: a live carded cell.
            unsafe { card_mark(cell as *const u8) };
        }
        fx.collect();
        assert_eq!(fx.heap.spare_pages, [base]);
        let leaves = fill_page(&mut fx, leaf, 64);
        assert_eq!(leaves[0], base, "the spare page is reused");
        assert_eq!(leaves.len(), PAGE_SIZE / 64);
        let page = fx
            .heap
            .pages
            .iter()
            .flatten()
            .find(|p| p.base == base)
            .unwrap();
        assert!(!page.carded);
        assert!(page_bytes(base).iter().all(|&byte| byte == 0), "zeroed");
        for &leaf in leaves
            .iter()
            .filter(|&&leaf| leaf >= base + CARD_TABLE_OFFSET)
        {
            assert_eq!(fx.heap.resolve(leaf as u64 + 8), Some(leaf as *const u8));
        }
        // Empty again, then reused for 128-byte carded cells: a clean table.
        fx.collect();
        assert_eq!(fx.heap.spare_pages, [base]);
        let again = fill_page(&mut fx, carded_desc, 128);
        assert_eq!(again[0], base);
        let page = fx
            .heap
            .pages
            .iter()
            .flatten()
            .find(|p| p.base == base)
            .unwrap();
        assert!(page.carded && (0..page.cells).all(|i| !page.card_set(i)));
    }

    /// S6 step 10b: marks persist. After a collection its survivors stay marked (old) and cells
    /// allocated since are unmarked (young) until the next collection starts, which clears them
    /// before it marks: between its phases the marks are its own, so `dead_cells`,
    /// `is_dead_small_cell` and black allocation answer as before, and after it the marked
    /// cells are its survivors. Large cells keep their mark the same way.
    #[test]
    fn survivors_stay_marked_until_the_next_collection_starts() {
        let mut fx = Fixture::new();
        let [a, b, c] = [1, 2, 3].map(|val| fx.alloc_node(0, val) as usize);
        let large = fx.heap.alloc(fx.node, 8192) as usize;
        for (slot, cell) in [a, b, large].into_iter().enumerate() {
            fx.roots[slot] = tagged(cell as *const u8);
        }
        fx.collect();
        assert!(!fx.heap.is_allocated_word(c as u64));
        let young = fx.alloc_node(0, 4) as usize;
        let marked = |fx: &Fixture, cell: usize| is_marked_at(&fx.heap, cell);
        assert!(
            marked(&fx, a) && marked(&fx, b) && marked(&fx, large),
            "old"
        );
        assert!(!marked(&fx, young), "young");
        // The next collection keeps `a` and `young` and lets `b` and the large cell die.
        fx.roots[1] = 0;
        fx.roots[2] = tagged(young as *const u8);
        assert!(fx.start_marking(&no_edges));
        let black = fx.alloc_node(0, 5) as usize;
        assert!(marked(&fx, a) && marked(&fx, young) && marked(&fx, black));
        assert!(!marked(&fx, b) && !marked(&fx, large));
        assert_eq!(fx.heap.dead_cells(fx.node), vec![b, large]);
        for (cell, dead) in [(a, false), (b, true), (young, false), (black, false)] {
            assert_eq!(fx.heap.is_dead_small_cell(cell, fx.node), dead, "{cell:#x}");
        }
        fx.heap.finish_collection(&[]);
        assert!(
            marked(&fx, a) && marked(&fx, young) && marked(&fx, black),
            "old"
        );
        assert!(!fx.heap.is_allocated_word(b as u64));
        assert_eq!(fx.heap.large_cells(), 0);
        assert_eq!(fx.live(), 3);
    }

    /// In debug builds a reader of the current marking's verdict panics outside a collection,
    /// where the marks say which cells are old.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "dead_cells read the marks outside a collection")]
    fn dead_cells_outside_a_collection_panics() {
        let mut fx = Fixture::new();
        fx.alloc_node(0, 1);
        fx.collect();
        fx.heap.dead_cells(fx.node);
    }

    /// S6 step 10b: the write barrier's check, as a collection starts (debug builds): an old
    /// carded cell written with a young cell's address and its card set passes, the young cell
    /// survives through it, and a later write with no card set panics, naming both cells: the
    /// card set before the first collection does not outlive its start.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "(carded node) with a clean card holds an edge to unmarked cell")]
    fn write_barrier_check_panics_on_an_old_cell_written_without_its_card() {
        let mut fx = Fixture::new();
        let old = fx.heap.alloc(fx.carded, 16);
        fx.roots[0] = tagged(old);
        fx.collect();
        let young = fx.alloc_node(0, 7);
        // SAFETY: a live carded cell.
        unsafe { card_mark(old) };
        set_word(old, 0, tagged(young));
        fx.collect();
        assert!(
            fx.heap.is_allocated_word(young as u64),
            "kept through `old`"
        );
        let second = fx.alloc_node(0, 8);
        set_word(old, 0, tagged(second));
        fx.collect();
    }

    /// The check's passing side: old cells written with their cards set (a carded small cell)
    /// and old cells whose edges only move to old cells or go (an uncarded cell, a large one)
    /// let every collection run, and the young cells written survive through their holders.
    #[test]
    fn write_barrier_check_passes_with_the_cards_set() {
        let mut fx = Fixture::new();
        let holders = [fx.heap.alloc(fx.carded, 16), fx.heap.alloc(fx.carded, 192)];
        let plain = fx.alloc_node(0, 1);
        let large = fx.heap.alloc(fx.node, 8192);
        for (slot, cell) in holders.iter().chain(&[plain, large]).enumerate() {
            fx.roots[slot] = tagged(*cell);
        }
        fx.collect();
        for round in 0..3 {
            let young: Vec<*mut u8> = holders.iter().map(|_| fx.alloc_node(0, round)).collect();
            for (&holder, &target) in holders.iter().zip(&young) {
                // SAFETY: a live carded cell.
                unsafe { card_mark(holder) };
                set_word(holder, 0, tagged(target));
            }
            // An old cell may point at another old cell, or at nothing, without a card.
            set_word(plain, 0, tagged(holders[round as usize % 2]));
            set_word(large, 0, 0);
            fx.collect();
            assert!(young
                .iter()
                .all(|&cell| fx.heap.is_allocated_word(cell as u64)));
        }
        assert_eq!(fx.live(), 2 + 2 + 2);
    }

    /// Whether the cell at `cell` is allocated (a word naming it, for the fixtures below).
    fn allocated(fx: &Fixture, cell: usize) -> bool {
        fx.heap.is_allocated_word(cell as u64)
    }

    /// S6 step 10c: a minor collection frees the young cells nothing reaches, small and large,
    /// keeps the young cells the roots reach, and traces no old cell whose card is clean: an old
    /// node whose only holder died, the holder and an old large cell, all unrooted, stay
    /// allocated through it (only the kept young cell is marked) and die at the next major
    /// collection. The old cells on the page that took the young ones are kept by its sweep.
    #[test]
    fn a_minor_collection_frees_young_garbage_and_keeps_old_cells_it_never_traces() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let held = fx.alloc_node(0, 1) as usize;
        let holder = fx.alloc_node(tagged(held as *const u8), 2) as usize;
        let old_large = fx.heap.alloc(fx.leaf, 8192) as usize;
        fx.roots[0] = tagged(holder as *const u8);
        fx.roots[1] = tagged(old_large as *const u8);
        fx.collect();
        fx.roots[0] = 0;
        fx.roots[1] = 0;
        let kept = fx.alloc_node(0, 3) as usize;
        let garbage = fx.alloc_node(0, 4) as usize;
        let young_large = fx.heap.alloc(fx.leaf, 8192) as usize;
        fx.roots[2] = tagged(kept as *const u8);
        assert_eq!(kept & !(PAGE_SIZE - 1), held & !(PAGE_SIZE - 1), "one page");
        let marked = fx.heap.stats().cells_marked;
        fx.collect_minor();
        assert_eq!(
            fx.heap.stats().cells_marked - marked,
            1,
            "only the young kept cell is marked"
        );
        for (cell, live, what) in [
            (held, true, "the old held node"),
            (holder, true, "its old holder"),
            (old_large, true, "the old large cell"),
            (kept, true, "the kept young node"),
            (garbage, false, "the young garbage node"),
            (young_large, false, "the young large cell"),
        ] {
            assert_eq!(allocated(&fx, cell), live, "{what}");
        }
        assert_eq!(fx.val(kept as *const u8), 3);
        fx.collect();
        for cell in [held, holder, old_large] {
            assert!(
                !allocated(&fx, cell),
                "{cell:#x} died at the major collection"
            );
        }
        assert!(allocated(&fx, kept));
        assert_eq!(fx.heap.minor_collections(), 1);
    }

    /// A minor collection traces the old cells whose cards are set, though they are marked, and
    /// every marked cell that starts in such a card (three 48-byte cells start in one): young
    /// cells only they hold survive it, an unrooted old holder's included (it is traced all the
    /// same, so its young cell lives until a major collection finds the holder dead). The cards
    /// are clean afterwards, and the young cells are old.
    #[test]
    fn young_cells_held_by_old_cells_with_set_cards_survive_a_minor_collection() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let carded = fx.carded;
        let cells = fill_page(&mut fx, carded, 48);
        let (a, b, unrooted) = (cells[0], cells[2], cells[3]);
        assert_eq!(
            (a - cells[0]) >> CARD_SHIFT,
            (b - cells[0]) >> CARD_SHIFT,
            "one card"
        );
        for (slot, cell) in [a, b, unrooted].into_iter().enumerate() {
            fx.roots[slot] = tagged(cell as *const u8);
        }
        fx.collect();
        fx.roots[2] = 0;
        let [ya, yb, yu] = [1, 2, 3].map(|val| fx.alloc_node(0, val) as usize);
        // `a`'s write sets the card `b` starts in too; `unrooted`'s sets its own.
        // SAFETY: live carded cells.
        unsafe { card_mark(a as *const u8) };
        set_word(a as *mut u8, 0, tagged(ya as *const u8));
        set_word(b as *mut u8, 0, tagged(yb as *const u8));
        // SAFETY: as above.
        unsafe { card_mark(unrooted as *const u8) };
        set_word(unrooted as *mut u8, 0, tagged(yu as *const u8));
        fx.collect_minor();
        assert_eq!(fx.heap.stats().last_card_cells, 3, "a, b and unrooted");
        for cell in [ya, yb, yu] {
            assert!(
                allocated(&fx, cell) && is_marked_at(&fx.heap, cell),
                "{cell:#x}"
            );
        }
        assert!(
            page_bytes(cells[0])[CARD_TABLE_OFFSET..]
                .iter()
                .all(|&card| card == 0),
            "the cards are clean"
        );
        fx.collect();
        assert!(!allocated(&fx, unrooted) && !allocated(&fx, yu));
        assert!([a, b, ya, yb].iter().all(|&cell| allocated(&fx, cell)));
    }

    /// The barrier's check runs as a minor collection starts (debug builds): an old cell
    /// written with a young cell's address and no card set panics there, before the minor
    /// collection, which would not trace the old cell, frees the young one.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "(carded node) with a clean card holds an edge to unmarked cell")]
    fn a_minor_collection_checks_the_barrier_before_it_trusts_the_cards() {
        let mut fx = Fixture::new();
        let old = fx.heap.alloc(fx.carded, 16);
        fx.roots[0] = tagged(old);
        fx.collect();
        let young = fx.alloc_node(0, 7);
        set_word(old, 0, tagged(young));
        fx.collect_minor();
    }

    /// A minor collection sweeps the pages that received cells since the last collection and
    /// no other: four full pages of old cells are left alone while two new pages, and an old
    /// page that gives a freed cell again, are swept. A major collection sweeps every page.
    #[test]
    fn a_minor_collection_sweeps_only_the_young_pages() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let per_page = PAGE_SIZE / 16;
        // Four full pages of a live chain, and a fifth whose cells but one die.
        fx.build_chain((per_page * 4) as u64);
        let survivor = fx.heap.alloc(fx.leaf, 16);
        for _ in 1..per_page {
            fx.heap.alloc(fx.leaf, 16);
        }
        fx.roots[1] = tagged(survivor);
        fx.collect();
        assert_eq!(fx.heap.stats().last_pages_swept, 5);
        assert_eq!(fx.heap.young_page_count(), 0);
        // Young leaves on the fifth page (it has free cells), a young node on a new page and a
        // young large cell (swept with every collection, but not a page).
        let young: Vec<*mut u8> = (0..10).map(|_| fx.heap.alloc(fx.leaf, 16)).collect();
        assert!(young
            .iter()
            .all(|&leaf| leaf as usize & !(PAGE_SIZE - 1) == survivor as usize & !(PAGE_SIZE - 1)));
        fx.alloc_node(0, 1);
        fx.heap.alloc(fx.leaf, 8192);
        assert_eq!(fx.heap.young_page_count(), 2);
        let swept = fx.heap.stats().cells_swept;
        fx.collect_minor();
        let stats = fx.heap.stats();
        assert_eq!(stats.last_pages_swept, 2);
        assert_eq!(
            stats.cells_swept - swept,
            12,
            "ten leaves, a node, a large cell"
        );
        assert_eq!(fx.heap.young_page_count(), 0);
        assert_eq!(stats.pages_live, 5, "the node's page emptied");
        fx.collect();
        assert_eq!(fx.heap.stats().last_pages_swept, 5);
    }

    /// Between a minor collection's phases the dead are the young cells nothing reached: an old
    /// cell that died since the last collection is marked, so `dead_cells`, `is_dead_small_cell`
    /// and `is_dead_target` call it live, and a cell allocated meanwhile (black) is live too. The
    /// next major collection lists the old one.
    #[test]
    fn a_minor_collection_finds_only_young_cells_dead() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let old_dead = fx.alloc_node(0, 1) as usize;
        let old_live = fx.alloc_node(0, 2) as usize;
        fx.roots[0] = tagged(old_dead as *const u8);
        fx.roots[1] = tagged(old_live as *const u8);
        fx.collect();
        fx.roots[0] = 0;
        let young_dead = fx.alloc_node(0, 3) as usize;
        let young_live = fx.alloc_node(0, 4) as usize;
        fx.roots[2] = tagged(young_live as *const u8);
        assert!(fx.start_marking_kind(&no_edges, CollectionKind::Minor));
        let black = fx.alloc_node(0, 5) as usize;
        assert_eq!(fx.heap.dead_cells(fx.node), vec![young_dead]);
        for (cell, dead) in [
            (old_dead, false),
            (old_live, false),
            (young_dead, true),
            (young_live, false),
            (black, false),
        ] {
            assert_eq!(fx.heap.is_dead_small_cell(cell, fx.node), dead, "{cell:#x}");
            assert_eq!(fx.heap.is_dead_target(cell), dead, "{cell:#x}");
        }
        fx.heap.finish_collection(&[]);
        assert!(!allocated(&fx, young_dead));
        assert!(allocated(&fx, old_dead) && allocated(&fx, black));
        assert!(fx.start_marking(&no_edges));
        let mut dead = fx.heap.dead_cells(fx.node);
        dead.sort_unstable();
        let mut expected = vec![old_dead, black];
        expected.sort_unstable();
        assert_eq!(dead, expected);
        fx.heap.finish_collection(&[]);
    }

    /// Promotion at first survival: a cell a minor collection keeps is old for the next, and
    /// the bytes it marked are its promoted bytes. Written with a young cell's address and its
    /// card set, the promoted cell keeps that cell through the next minor collection; unrooted,
    /// it stays through a minor collection and dies at a major one.
    #[test]
    fn a_cell_that_survives_a_minor_collection_is_old_for_the_next() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let cell = fx.heap.alloc(fx.carded, 16);
        fx.roots[0] = tagged(cell);
        fx.collect_minor();
        assert!(is_marked_at(&fx.heap, cell as usize), "promoted");
        assert_eq!(fx.heap.stats().last_marked_bytes, 16);
        let young = fx.alloc_node(0, 9);
        // SAFETY: a live carded cell.
        unsafe { card_mark(cell) };
        set_word(cell, 0, tagged(young));
        fx.collect_minor();
        assert!(allocated(&fx, young as usize) && is_marked_at(&fx.heap, young as usize));
        assert_eq!(fx.heap.stats().last_marked_bytes, 16, "the young node");
        fx.roots[0] = 0;
        fx.collect_minor();
        assert!(allocated(&fx, cell as usize) && allocated(&fx, young as usize));
        assert_eq!(fx.heap.stats().last_marked_bytes, 0);
        fx.collect();
        assert!(!allocated(&fx, cell as usize) && !allocated(&fx, young as usize));
        assert_eq!(fx.heap.minor_collections(), 3);
    }

    /// The due logic with a nursery, and the due word with it: a minor collection is due once
    /// the bytes allocated since the last collection pass the nursery budget; a major one once
    /// a minor collection leaves the heap holding more than the byte budget (live or not), at
    /// once and until a major collection runs, which sets the next budget; and a major request
    /// that is deferred makes one due. Without a nursery the heap asks for major collections.
    #[test]
    fn minor_and_major_collections_fall_due_by_their_budgets() {
        // Declared before the fixture: the word outlives the heap.
        let word = Box::new(Cell::new(7u64));
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        // SAFETY: `word` outlives the heap and nothing else writes it.
        unsafe { fx.heap.attach_due_word(NonNull::from(&*word)) };
        let due = |fx: &Fixture| {
            assert!(fx.heap.due_word_agrees());
            word.get() == 1
        };
        assert_eq!(fx.heap.due_kind(), CollectionKind::Major, "no nursery");
        // A nursery of 256 16-byte cells, and a byte budget of 16 KiB.
        fx.heap.set_heap_growth(16 << 10, 100);
        fx.heap.set_nursery_bytes(Some(4 << 10));
        assert_eq!(fx.heap.due_kind(), CollectionKind::Minor);
        let mut head: *mut u8 = std::ptr::null_mut();
        // Grow a live chain until a collection is due: the 257th cell passes the nursery.
        let mut grow = |fx: &mut Fixture| {
            let mut cells = 0;
            while !fx.heap.collection_due() {
                let next = if head.is_null() { 0 } else { tagged(head) };
                head = fx.alloc_node(next, cells);
                fx.roots[0] = tagged(head);
                cells += 1;
            }
            cells
        };
        for round in 1..=4u64 {
            assert!(!due(&fx));
            assert_eq!(grow(&mut fx), 257, "round {round}");
            assert!(due(&fx));
            assert_eq!(fx.heap.due_kind(), CollectionKind::Minor, "round {round}");
            fx.collect_minor();
            // The heap holds round × 4,112 bytes: past 16 KiB after the fourth round.
            assert_eq!(
                (due(&fx), fx.heap.due_kind() == CollectionKind::Major),
                (round == 4, round == 4),
                "round {round}"
            );
        }
        fx.collect();
        assert!(
            !due(&fx),
            "a major collection sets the next budget, 2 × 16,448 bytes"
        );
        assert_eq!(fx.heap.stats().trigger_bytes, 2 * 4 * 257 * 16);
        assert_eq!(fx.heap.due_kind(), CollectionKind::Minor);
        assert_eq!(grow(&mut fx), 257, "the nursery after a major collection");
        fx.collect_minor();
        assert!(!due(&fx));
        // A major request deferred in a no-GC region makes a major collection due.
        {
            let _scope = fx.heap.nogc();
            assert!(!fx.heap.collect_at_safepoint());
        }
        assert!(due(&fx));
        assert_eq!(fx.heap.due_kind(), CollectionKind::Major);
        fx.collect();
        assert!(!due(&fx));
        assert_eq!(fx.heap.due_kind(), CollectionKind::Minor);
        fx.heap.set_nursery_bytes(None);
        assert_eq!(fx.heap.due_kind(), CollectionKind::Major, "no nursery");
    }

    /// A requested major collection makes none due, and makes the next collection the heap asks
    /// for major, whatever its reason: a minor collection keeps the request, a major one clears
    /// it.
    #[test]
    fn a_requested_major_makes_the_next_collection_major_without_making_one_due() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        fx.heap.set_nursery_bytes(Some(4 << 10));
        assert_eq!(fx.heap.due_kind(), CollectionKind::Minor);
        fx.heap.request_major();
        assert!(!fx.heap.collection_due());
        assert_eq!(fx.heap.due_kind(), CollectionKind::Major);
        fx.collect_minor();
        assert_eq!(fx.heap.due_kind(), CollectionKind::Major, "kept by a minor");
        fx.collect();
        assert_eq!(fx.heap.due_kind(), CollectionKind::Minor);
        assert!(!fx.heap.collection_due());
    }

    /// Under a stress interval with a nursery, the collections the heap runs itself are minor
    /// but for every Mth, which is major; with M = 1 every one is major.
    #[test]
    fn stress_collections_are_minor_but_every_mth_is_major() {
        let mut fx = Fixture::new();
        let top = 0u64;
        let _stack = unsafe { enter_native_stack(stack_top(&top)) };
        fx.heap.set_nursery_bytes(Some(1 << 20));
        fx.heap.set_stress_every(1);
        // The kinds of the collections `allocations` allocations run: m minor, M major.
        let kinds = |fx: &mut Fixture, allocations: usize| {
            let mut seen = String::new();
            for _ in 0..allocations {
                let (all, minor) = (fx.heap.collections(), fx.heap.minor_collections());
                black_box(fx.heap.alloc(fx.leaf, 16));
                if fx.heap.collections() > all {
                    seen.push(if fx.heap.minor_collections() > minor {
                        'm'
                    } else {
                        'M'
                    });
                }
            }
            seen
        };
        fx.heap.set_stress_major_every(3);
        assert_eq!(kinds(&mut fx, 10), "mmMmmMmmM");
        fx.heap.set_stress_major_every(1);
        assert_eq!(kinds(&mut fx, 4), "MMMM");
        black_box(&top);
    }

    /// With a nursery the spare pool keeps what the nursery can fill (its budget's pages, at
    /// least 16), whatever the byte budget.
    #[test]
    fn spare_pages_follow_the_nursery_budget() {
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        fx.heap.set_heap_growth(8 << 20, 100);
        fx.heap.set_nursery_bytes(Some(2 << 20));
        let per_page = PAGE_SIZE / 16;
        let pages = |fx: &Fixture| {
            let stats = fx.heap.stats();
            (stats.pages_live, stats.pages_spare)
        };
        for i in 0..per_page * 100 {
            fx.alloc_node(0, i as u64);
        }
        fx.collect_minor();
        assert_eq!(pages(&fx), (0, 32), "2 MiB: 32 pages");
        fx.heap.set_nursery_bytes(Some(64 << 10));
        fx.collect();
        assert_eq!(pages(&fx), (0, 16), "the floor");
    }

    /// Pins in minor collections: a pinned young cell that dies becomes a zombie; once its page
    /// receives no cell, minor collections leave it alone, pinned or not (a pin recorded there
    /// would outlast the collection); a minor collection that sweeps its page again frees it
    /// once unpinned, without finalizing it twice.
    #[test]
    fn a_minor_collection_keeps_and_frees_zombies_on_the_pages_it_sweeps() {
        FIN_COUNT.with(|c| c.set(0));
        let mut fx = Fixture::new();
        fx.heap.set_collect_on_alloc(false);
        let anchor = fx.heap.alloc(fx.leaf, 4096);
        fx.roots[0] = tagged(anchor);
        fx.collect();
        let pinned = fx.heap.alloc(fx.leaf, 4096) as usize;
        assert!(fx.start_marking_kind(&no_edges, CollectionKind::Minor));
        assert_eq!(fx.heap.dead_cells(fx.leaf), vec![pinned]);
        fx.heap.finish_collection(&[pinned]);
        assert!(allocated(&fx, pinned), "a zombie");
        assert_eq!(FIN_COUNT.with(|c| c.get()), 1);
        // Its page received no cell since: pinned again, the pin is not recorded there.
        assert_eq!(fx.heap.young_page_count(), 0);
        assert!(fx.start_marking_kind(&no_edges, CollectionKind::Minor));
        assert!(fx.heap.dead_cells(fx.leaf).is_empty());
        fx.heap.finish_collection(&[pinned]);
        assert!(allocated(&fx, pinned));
        // A cell taken from the page makes it young: the next minor collection sweeps it, and
        // frees the zombie, unpinned now (no pin left over from the last collection keeps it).
        let other = fx.heap.alloc(fx.leaf, 4096) as usize;
        assert_eq!(other & !(PAGE_SIZE - 1), pinned & !(PAGE_SIZE - 1));
        fx.collect_minor();
        assert!(!allocated(&fx, pinned) && !allocated(&fx, other));
        assert_eq!(
            FIN_COUNT.with(|c| c.get()),
            2,
            "the zombie is not finalized twice"
        );
        assert!(allocated(&fx, anchor as usize));
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

    #[test]
    fn cell_index_divides_exactly_for_every_class_and_offset() {
        for cell_size in SIZE_CLASSES {
            let page = MarkPage {
                marks: std::ptr::null(),
                reciprocal: u32::try_from((1u64 << 32).div_ceil(cell_size as u64)).unwrap(),
                cells: 0,
            };
            let base = 7 * PAGE_SIZE;
            for offset in 0..PAGE_SIZE {
                assert_eq!(
                    page.cell_index(base + offset),
                    offset / cell_size,
                    "offset {offset} in a page of {cell_size}-byte cells"
                );
            }
        }
    }

    mod positive_word_cache {
        use super::*;

        fn trace_object(cell: *const u8, sink: &mut dyn FnMut(*const u8)) {
            // A synthetic object's prototype, shared layout and independent child.
            for index in 0..3 {
                let edge = word(cell, index);
                if edge != 0 {
                    sink((edge & PAYLOAD_MASK) as *const u8);
                }
            }
        }

        fn trace_batch(cell: *const u8, sink: &mut dyn FnMut(*const u8)) {
            for index in 0..word(cell, 0) as usize {
                sink((word(cell, index + 1) & PAYLOAD_MASK) as *const u8);
            }
        }

        fn alloc_batch(fx: &mut Fixture, edges: &[*mut u8]) -> (*mut u8, DescId) {
            let desc = fx.heap.register_desc(LayoutDesc {
                name: "edge batch",
                trace: trace_batch,
                mark_trace: None,
                finalize: None,
                carded: false,
            });
            let cell = fx.heap.alloc(desc, (edges.len() + 1) * 8);
            set_word(cell, 0, edges.len() as u64);
            for (index, &edge) in edges.iter().enumerate() {
                set_word(cell, index + 1, tagged(edge));
            }
            (cell, desc)
        }

        /// A trace longer than the 64 edges the marker once cached (it marks every edge as
        /// the trace reports it now, through the cache, whatever the trace's length).
        const BATCH: usize = 65;

        #[test]
        fn drain_probes_the_cache_for_every_edge_of_any_trace() {
            for count in [1, 64, BATCH, 1000] {
                let mut fx = Fixture::new();
                fx.heap.set_collect_on_alloc(false);
                let child = fx.heap.alloc(fx.leaf, 16);
                let target = fx.alloc_node(tagged(child), 1);
                let (batch, _) = alloc_batch(&mut fx, &vec![target; count]);
                let mut marker = Marker::default();
                let before = fx.heap.stats().cells_marked;
                fx.heap.mark_word(tagged(batch), &mut marker);
                fx.heap.drain(&mut marker, &NoClient);
                fx.heap.verify(&marker);
                assert_eq!(fx.heap.stats().cells_marked - before, 3);
                assert!(is_marked_at(&fx.heap, child as usize));
                // The batch, the first of the repeated edges and the child resolve; every
                // repetition hits.
                assert_eq!(marker.positive_words.probes.get(), count + 2);
                assert_eq!(marker.word_cache_hits, count - 1, "repeated edges hit");
                assert_eq!(marker.word_lookups, 3);
                assert!(marker.positive_words.contains(target as u64));
            }
        }

        #[test]
        fn large_batch_retains_sole_targets_descendants_and_ephemeron_wakeups() {
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            let count = BATCH;
            let children: Vec<_> = (0..count).map(|_| fx.heap.alloc(fx.leaf, 16)).collect();
            let targets: Vec<_> = children
                .iter()
                .map(|&child| fx.alloc_node(tagged(child), 1))
                .collect();
            let mut edges = targets.clone();
            edges.push(*targets.last().unwrap()); // An adjacent repeat, which the cache answers.
            let (batch, batch_desc) = alloc_batch(&mut fx, &edges);
            let conditional_child = fx.heap.alloc(fx.leaf, 16);
            let conditional = fx.alloc_node(tagged(conditional_child), 2);
            let mut marker = Marker {
                parents: Some(FastMap::default()),
                ..Marker::default()
            };
            fx.heap.mark_word(tagged(batch), &mut marker);
            fx.heap.drain(&mut marker, &NoClient);
            fx.heap.verify(&marker);
            assert_eq!(marker.word_cache_hits, 1, "the adjacent repeat hits");
            assert_eq!(marker.word_lookups, 1 + (edges.len() - 1) + count);
            assert_eq!(marker.parents.as_ref().unwrap().len(), 1 + 2 * count);
            assert!(targets.iter().all(|target| matches!(
                marker.parents.as_ref().unwrap().get(&(*target as usize)),
                Some(MarkSource::Cell(owner)) if *owner == batch as usize
            )));

            // Only the batch is a precise root. Rust vectors/closure captures are excluded;
            // the first target must wake the conditional edge when the drain marks it.
            fx.roots[0] = tagged(batch);
            let ephemeron = |sink: &mut EdgeSink<'_>| {
                sink.ephemeron(batch, tagged(targets[0]), tagged(conditional));
            };
            let before = fx.heap.stats().cells_marked;
            fx.collect_with(&ephemeron);
            assert_eq!(
                fx.heap.stats().cells_marked - before,
                (1 + 2 * count + 2) as u64
            );
            let assert_live = |fx: &Fixture| {
                assert!(fx.heap.owns_cell(batch as usize, batch_desc));
                for (&target, &child) in targets.iter().zip(&children) {
                    assert!(fx.heap.owns_cell(target as usize, fx.node), "sole target");
                    assert!(
                        fx.heap.owns_cell(child as usize, fx.leaf),
                        "sole descendant"
                    );
                }
                assert!(
                    fx.heap.owns_cell(conditional as usize, fx.node),
                    "waiter value"
                );
                assert!(
                    fx.heap.owns_cell(conditional_child as usize, fx.leaf),
                    "waiter child"
                );
            };
            assert_live(&fx); // Before any further allocation could reuse a missing edge's cell.
            assert!(fx.start_marking_kind(&ephemeron, CollectionKind::Minor));
            fx.heap.finish_collection(&[]);
            assert_live(&fx);
            fx.roots[0] = 0;
            fx.collect_with(&ephemeron);
            assert!(!allocated(&fx, batch as usize));
            for (&target, &child) in targets.iter().zip(&children) {
                assert!(!allocated(&fx, target as usize));
                assert!(!allocated(&fx, child as usize));
            }
            assert!(!allocated(&fx, conditional as usize));
            assert!(!allocated(&fx, conditional_child as usize));
        }

        struct BatchTrees {
            targets: Vec<*mut u8>,
            children: Vec<*mut u8>,
            conditional: *mut u8,
            conditional_child: *mut u8,
        }

        impl BatchTrees {
            fn new(fx: &mut Fixture) -> Self {
                let children: Vec<_> = (0..BATCH).map(|_| fx.heap.alloc(fx.leaf, 16)).collect();
                let targets = children
                    .iter()
                    .map(|&child| fx.alloc_node(tagged(child), 1))
                    .collect();
                let conditional_child = fx.heap.alloc(fx.leaf, 16);
                let conditional = fx.alloc_node(tagged(conditional_child), 2);
                Self {
                    targets,
                    children,
                    conditional,
                    conditional_child,
                }
            }

            fn edges(&self) -> Vec<*mut u8> {
                let mut edges = self.targets.clone();
                edges.push(*self.targets.last().unwrap());
                assert!(edges.len() > BATCH);
                edges
            }

            fn cells(&self) -> impl Iterator<Item = *mut u8> + '_ {
                self.targets
                    .iter()
                    .chain(&self.children)
                    .copied()
                    .chain([self.conditional, self.conditional_child])
            }

            fn assert_age(&self, fx: &Fixture, expected: CellAge) {
                for cell in self.cells() {
                    assert_eq!(fx.heap.cell_age(cell as u64), Some(expected), "{cell:p}");
                }
            }

            fn assert_live(&self, fx: &Fixture) {
                for (&target, &child) in self.targets.iter().zip(&self.children) {
                    assert!(fx.heap.owns_cell(target as usize, fx.node), "sole target");
                    assert!(
                        fx.heap.owns_cell(child as usize, fx.leaf),
                        "sole descendant"
                    );
                }
                assert!(
                    fx.heap.owns_cell(self.conditional as usize, fx.node),
                    "waiter value"
                );
                assert!(
                    fx.heap.owns_cell(self.conditional_child as usize, fx.leaf),
                    "waiter child"
                );
            }

            fn assert_dead(&self, fx: &Fixture) {
                for cell in self.cells() {
                    assert!(!allocated(fx, cell as usize), "{cell:p} must be reclaimed");
                }
            }
        }

        #[test]
        fn first_minor_large_young_batch_retains_sole_trees_and_wakes_waiters() {
            let mut fx = Fixture::new();
            // Explicit precise collections preserve the young precondition even under STRESS=1.
            fx.heap.set_collect_on_alloc(false);
            let trees = BatchTrees::new(&mut fx);
            let (batch, desc) = alloc_batch(&mut fx, &trees.edges());
            fx.roots[0] = batch as u64;
            let ephemeron = |sink: &mut EdgeSink<'_>| {
                sink.ephemeron(batch, tagged(trees.targets[0]), tagged(trees.conditional));
            };
            // No preliminary mark/major may promote the batch and make this minor vacuous.
            assert_eq!(fx.heap.collections(), 0);
            assert_eq!(fx.heap.cell_age(batch as u64), Some(CellAge::Young));
            trees.assert_age(&fx, CellAge::Young);
            let marked = fx.heap.stats().cells_marked;
            // Only the batch is rooted. Native scanning excludes these vectors and captures.
            assert!(fx.start_marking_kind(&ephemeron, CollectionKind::Minor));
            fx.heap.finish_collection(&[]);
            assert_eq!(fx.heap.collections(), 1);
            assert_eq!(fx.heap.minor_collections(), 1);
            assert_eq!(fx.heap.stats().last_card_cells, 0);
            assert_eq!(fx.heap.stats().cells_marked - marked, 133);
            assert!(fx.heap.owns_cell(batch as usize, desc));
            trees.assert_live(&fx); // Before another allocation can reuse an omitted edge.
            assert_eq!(fx.heap.cell_age(batch as u64), Some(CellAge::Old));
            trees.assert_age(&fx, CellAge::Old);
            fx.collect_with(&ephemeron);
            assert!(fx.heap.owns_cell(batch as usize, desc));
            trees.assert_live(&fx);
            fx.roots[0] = 0;
            fx.collect_with(&ephemeron);
            assert_eq!(fx.live(), 0);
            assert!(!allocated(&fx, batch as usize));
            trees.assert_dead(&fx);
        }

        #[test]
        fn old_carded_large_batch_traces_sole_young_trees_and_wakes_waiters() {
            let mut fx = Fixture::new();
            // Keep the newly published targets young until this test's explicit minor.
            fx.heap.set_collect_on_alloc(false);
            let desc = fx.heap.register_desc(LayoutDesc {
                name: "carded edge batch",
                trace: trace_batch,
                mark_trace: None,
                finalize: None,
                carded: true,
            });
            let edge_count = BATCH + 1;
            let batch = fx.heap.alloc(desc, (edge_count + 1) * 8);
            set_word(batch, 0, edge_count as u64); // Other words start zeroed.
            fx.roots[0] = batch as u64; // Exactly the raw word also queued by the dirty card.
            fx.collect();
            assert_eq!(fx.live(), 1);
            assert_eq!(fx.heap.cell_age(batch as u64), Some(CellAge::Old));
            let trees = BatchTrees::new(&mut fx);
            let edges = trees.edges();
            assert_eq!(edges.len(), edge_count);
            // SAFETY: an allocated small carded cell; mark before publishing its young edges.
            unsafe { card_mark(batch) };
            for (index, edge) in edges.into_iter().enumerate() {
                set_word(batch, index + 1, tagged(edge));
            }
            let ephemeron = |sink: &mut EdgeSink<'_>| {
                sink.ephemeron(batch, tagged(trees.targets[0]), tagged(trees.conditional));
            };
            assert_eq!(fx.heap.collections(), 1);
            assert_eq!(fx.heap.cell_age(batch as u64), Some(CellAge::Old));
            trees.assert_age(&fx, CellAge::Young);
            let marked = fx.heap.stats().cells_marked;
            // The raw root becomes a positive before its independent card trace drains.
            // Its >64 young edges must still mark their descendants and wake the waiter.
            assert!(fx.start_marking_kind(&ephemeron, CollectionKind::Minor));
            fx.heap.finish_collection(&[]);
            assert_eq!(fx.heap.collections(), 2);
            assert_eq!(fx.heap.minor_collections(), 1);
            assert_eq!(fx.heap.stats().last_card_cells, 1);
            assert_eq!(
                fx.heap.stats().cells_marked - marked,
                132,
                "only young targets mark anew"
            );
            assert!(fx.heap.owns_cell(batch as usize, desc));
            trees.assert_live(&fx);
            trees.assert_age(&fx, CellAge::Old);
            fx.collect_with(&ephemeron);
            assert!(fx.heap.owns_cell(batch as usize, desc));
            trees.assert_live(&fx);
            fx.roots[0] = 0;
            fx.collect_with(&ephemeron);
            assert_eq!(fx.live(), 0);
            assert!(!allocated(&fx, batch as usize));
            trees.assert_dead(&fx);
        }

        #[test]
        fn small_and_large_first_marks_queue_one_trace_each_and_then_hit() {
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            let child = fx.heap.alloc(fx.leaf, 16);
            let other_child = fx.heap.alloc(fx.leaf, 16);
            let large = fx.heap.alloc(fx.node, 8192);
            set_word(large, 0, tagged(other_child));
            // The large cell's cache entry follows its mapping's address, and a small cell
            // sharing it would take it over (a collision, which
            // `direct_collisions_keep_both_rooted_trees_and_their_distinct_descendants` covers):
            // take a small cell whose entry is another, as the next one's is.
            let small = std::iter::repeat_with(|| fx.alloc_node(tagged(child), 1))
                .find(|&small| {
                    PositiveWords::index(small as u64) != PositiveWords::index(large as u64)
                })
                .expect("a small cell outside the large cell's entry");
            let mut marker = Marker::default();
            let before = fx.heap.stats().cells_marked;
            for candidate in [small as u64, large as u64, small as u64, large as u64] {
                fx.heap.mark_word(candidate, &mut marker);
            }
            assert_eq!(marker.word_lookups, 2, "each cell resolves once");
            assert_eq!(marker.word_cache_hits, 2, "and then hits");
            assert_eq!(fx.heap.stats().cells_marked - before, 2);
            assert_eq!(marker.worklist.len(), 2, "one trace queued for each");
            fx.heap.drain(&mut marker, &NoClient);
            fx.heap.verify(&marker);
            assert_eq!(fx.heap.stats().cells_marked - before, 4);
            assert!(is_marked_at(&fx.heap, child as usize));
            assert!(is_marked_at(&fx.heap, other_child as usize));
        }

        #[test]
        fn shared_prototype_layout_fan_in_hits_without_losing_first_mark_work() {
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            let object = fx.heap.register_desc(LayoutDesc {
                name: "prototype/layout fan-in",
                trace: trace_object,
                mark_trace: None,
                finalize: None,
                carded: false,
            });
            let [prototype_child, layout_child] = [0; 2].map(|_| fx.heap.alloc(fx.leaf, 16));
            let prototype = fx.alloc_node(tagged(prototype_child), 1);
            let layout = fx.alloc_node(tagged(layout_child), 2);
            let objects: Vec<*mut u8> = (0..48)
                .map(|_| {
                    let child = fx.heap.alloc(fx.leaf, 16);
                    let cell = fx.heap.alloc(object, 32);
                    for (index, edge) in [prototype, layout, child].into_iter().enumerate() {
                        set_word(cell, index, tagged(edge));
                    }
                    cell
                })
                .collect();
            let mut marker = Marker {
                parents: Some(FastMap::default()),
                ..Marker::default()
            };
            let before = fx.heap.stats().cells_marked;
            for &cell in &objects {
                fx.heap.mark_word(tagged(cell), &mut marker);
            }
            fx.heap.drain(&mut marker, &NoClient);
            fx.heap.verify(&marker); // Independent validation never reads the cache.
            assert_eq!(fx.heap.stats().cells_marked - before, 100);
            assert_eq!(marker.parents.as_ref().unwrap().len(), 100);
            assert!(matches!(
                marker.parents.as_ref().unwrap().get(&(prototype as usize)),
                Some(MarkSource::Cell(owner)) if *owner == *objects.last().unwrap() as usize
            ));
            assert_eq!(marker.word_lookups + marker.word_cache_hits, 194);
            assert!(
                marker.word_cache_hits >= 24,
                "shared edges must bypass repeated locate/mark checks: {} hits / {} lookups",
                marker.word_cache_hits,
                marker.word_lookups
            );
            for &cell in &[prototype_child, layout_child] {
                assert!(is_marked_at(&fx.heap, cell as usize));
            }
        }

        #[test]
        fn tagged_raw_and_interior_words_cache_only_exact_words_for_small_and_large_cells() {
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            let mut marker = Marker::default();
            for size in [16, 8192] {
                let child = fx.heap.alloc(fx.leaf, 16);
                let parent = fx.heap.alloc(fx.node, size);
                set_word(parent, 0, tagged(child));
                let raw = parent as u64;
                for candidate in [
                    raw,
                    TAG | raw,
                    0x7ffe_0000_0000_0000 | raw,
                    raw + 8,
                    TAG | (raw + 8),
                ] {
                    let lookups = marker.word_lookups;
                    let hits = marker.word_cache_hits;
                    fx.heap.mark_word(candidate, &mut marker);
                    assert_eq!(
                        marker.word_lookups,
                        lookups + 1,
                        "distinct exact word {candidate:#x}"
                    );
                    fx.heap.mark_word(candidate, &mut marker);
                    assert_eq!(marker.word_cache_hits, hits + 1, "repeat {candidate:#x}");
                }
                assert_eq!(marker.worklist.len(), 1);
                // The first variant marks the parent; subsequent variants are already-live
                // positives. Neither may duplicate its worklist entry or lose its child.
                fx.heap.drain(&mut marker, &NoClient);
                assert!(is_marked_at(&fx.heap, child as usize));
            }
            assert_eq!(fx.heap.stats().cells_marked, 4);
            fx.heap.verify(&marker);
        }

        #[test]
        fn direct_collisions_keep_both_rooted_trees_and_their_distinct_descendants() {
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            let parents: Vec<*mut u8> = (0..MARK_CACHE_WORDS + 1)
                .map(|_| fx.alloc_node(0, 1))
                .collect();
            let (first, second) = (parents[0], parents[MARK_CACHE_WORDS]);
            let bucket = PositiveWords::index(tagged(first));
            assert_ne!(first, second);
            assert_eq!(
                PositiveWords::index(tagged(second)),
                bucket,
                "deliberate real-cell collision"
            );
            let children: Vec<*mut u8> = (0..2)
                .map(|_| {
                    (0..MARK_CACHE_WORDS)
                        .map(|_| fx.heap.alloc(fx.leaf, 16))
                        .find(|&child| PositiveWords::index(child as u64) != bucket)
                        .expect("a descendant in another bucket")
                })
                .collect();
            set_word(first, 0, tagged(children[0]));
            set_word(second, 0, tagged(children[1]));
            fx.roots[0] = tagged(first);
            fx.roots[1] = tagged(second);
            fx.collect(); // Precise: the two colliding roots are the only holders.
            assert!(
                fx.heap.owns_cell(second as usize, fx.node),
                "collision must keep the second rooted tree"
            );
            assert!(
                fx.heap.owns_cell(children[1] as usize, fx.leaf),
                "collision must keep its descendant"
            );
            assert!(fx.heap.owns_cell(first as usize, fx.node));
            assert!(fx.heap.owns_cell(children[0] as usize, fx.leaf));
            assert_eq!(fx.live(), 4);
            assert_eq!(fx.heap.stats().cells_marked, 4);
        }

        #[test]
        fn fresh_major_and_minor_markers_do_not_retain_removed_roots_or_hide_reused_addresses() {
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            let anchor = fx.alloc_node(0, 1);
            let old_child = fx.alloc_node(0, 2);
            let old = fx.alloc_node(tagged(old_child), 3);
            fx.roots[0] = tagged(anchor);
            fx.roots[1] = tagged(old);
            fx.collect();
            fx.collect_minor();
            assert_eq!(fx.live(), 3);
            fx.roots[1] = 0;
            fx.collect();
            assert!(!allocated(&fx, old as usize));
            assert!(!allocated(&fx, old_child as usize));
            let child = fx.heap.alloc(fx.leaf, 16);
            let reused = (0..PAGE_SIZE / 16)
                .map(|_| fx.alloc_node(0, 4))
                .find(|&candidate| candidate == old)
                .expect("the anchored node page reuses the actual old address");
            assert_eq!(reused, old);
            set_word(reused, 0, tagged(child));
            fx.roots[1] = tagged(reused); // Exactly the old word, now with a new descendant.
            fx.collect();
            assert_eq!(fx.live(), 3);
            assert!(fx.heap.owns_cell(child as usize, fx.leaf));
            assert!(fx.heap.owns_cell(reused as usize, fx.node));
            fx.collect_minor();
            assert!(fx.heap.owns_cell(child as usize, fx.leaf));
            fx.roots[1] = 0;
            fx.collect();
            assert_eq!(fx.live(), 1);
            assert!(!allocated(&fx, reused as usize));
            assert!(!allocated(&fx, child as usize));
        }

        #[test]
        fn a_cached_raw_old_root_still_traces_its_card_and_sole_young_tree() {
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            let old = fx.heap.alloc(fx.carded, 16);
            fx.roots[0] = old as u64; // Exactly the raw word queued by the card table.
            fx.collect();
            let child = fx.heap.alloc(fx.leaf, 16);
            let young = fx.alloc_node(tagged(child), 7);
            // SAFETY: an allocated small carded cell. These are the young tree's only roots.
            unsafe { card_mark(old) };
            set_word(old, 0, tagged(young));
            assert_eq!(fx.heap.cell_age(young as u64), Some(CellAge::Young));
            assert_eq!(fx.heap.cell_age(child as u64), Some(CellAge::Young));
            {
                let mut probe = Marker::default();
                fx.heap.mark_word(old as u64, &mut probe);
                assert!(
                    probe.positive_words.contains(old as u64),
                    "old allocated marks are positives"
                );
                assert!(
                    probe.worklist.is_empty(),
                    "the existing mark does not queue another trace"
                );
            }
            // Actual minor marking queues the dirty card first, then processes this same
            // raw root. No other positive root intervenes before draining its queued trace.
            // Cache membership must never deduplicate that independent card-owner worklist.
            let marked = fx.heap.stats().cells_marked;
            fx.collect_minor(); // Precise roots: Rust locals cannot keep the young tree alive.
            assert_eq!(fx.heap.stats().last_card_cells, 1);
            assert_eq!(fx.heap.stats().cells_marked - marked, 2);
            assert!(fx.heap.owns_cell(young as usize, fx.node));
            assert!(fx.heap.owns_cell(child as usize, fx.leaf));
            assert_eq!(fx.heap.cell_age(child as u64), Some(CellAge::Old));
        }

        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        #[test]
        fn native_major_cache_words_do_not_retain_a_previous_cycles_only_root() {
            #[inline(never)]
            fn warm_from_hidden(heap: &mut Heap, hidden: u64) -> bool {
                // Only this popped frame decodes the address. Four adjacent volatile roots
                // exercise the real conservative scanner and its repeated positive words.
                collect_with_words_in_frame(heap, [!hidden; 4])
            }
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            FIN_COUNT.with(|count| count.set(0));
            let top = 0u64;
            let _stack = unsafe { enter_native_stack(stack_top(&top)) };
            for round in 0..2 {
                let hidden = alloc_hidden(&mut fx.heap, fx.leaf, 16);
                crate::gc_core::scrub_stack(64 * 1024);
                let marked = fx.heap.stats().cells_marked;
                assert!(warm_from_hidden(&mut fx.heap, hidden));
                assert_eq!(fx.live(), 1, "native words kept round {round}'s cell");
                assert_eq!(fx.heap.stats().cells_marked - marked, 1);
                // This removes dead helper/collector frames while retaining only the
                // complemented address in this frame; it does not decode or root the cell.
                crate::gc_core::scrub_stack(64 * 1024);
                assert!(fx.heap.collect());
                assert_eq!(
                    fx.live(),
                    0,
                    "a fresh native major must not see cached old words"
                );
                assert_eq!(FIN_COUNT.with(Cell::get), round + 1);
                assert!(fx.heap.collect());
                assert_eq!(fx.live(), 0);
                assert_eq!(FIN_COUNT.with(Cell::get), round + 1, "finalization is once");
                black_box(hidden);
            }
            black_box(&top);
        }

        #[test]
        fn repeated_free_zombie_foreign_and_zero_words_never_become_positives() {
            let mut fx = Fixture::new();
            let mut foreign = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            foreign.heap.set_collect_on_alloc(false);
            FIN_COUNT.with(|count| count.set(0));
            let anchor = fx.heap.alloc(fx.leaf, 16);
            let free = fx.heap.alloc(fx.leaf, 16);
            let zombie = fx.heap.alloc(fx.leaf, 16);
            let large_free = fx.heap.alloc(fx.leaf, 8192);
            let large_zombie = fx.heap.alloc(fx.leaf, 8192);
            let other = foreign.alloc_node(0, 5);
            fx.roots[0] = tagged(anchor);
            assert!(fx.start_marking(&no_edges));
            fx.heap
                .finish_collection(&[zombie as usize, large_zombie as usize]);
            assert_eq!(FIN_COUNT.with(Cell::get), 4);
            assert_eq!(fx.heap.cell_age(zombie as u64), Some(CellAge::Dead));
            assert_eq!(fx.heap.cell_age(large_zombie as u64), Some(CellAge::Dead));
            let rejected = [
                0,
                123,
                free as u64,
                tagged(free),
                free as u64 + 8,
                zombie as u64,
                tagged(zombie),
                large_free as u64,
                large_zombie as u64,
                tagged(large_zombie),
                other as u64,
                tagged(other),
            ];
            let mut marker = Marker::default();
            let marked = fx.heap.stats().cells_marked;
            for _ in 0..3 {
                for &candidate in &rejected {
                    fx.heap.mark_word(candidate, &mut marker);
                    assert!(!marker.positive_words.contains(candidate), "{candidate:#x}");
                }
            }
            assert_eq!(marker.word_lookups, rejected.len() * 3);
            assert_eq!(marker.word_cache_hits, 0);
            assert_eq!(fx.heap.stats().cells_marked, marked);
            assert!(marker.worklist.is_empty());
            assert_eq!(foreign.heap.cell_age(other as u64), Some(CellAge::Young));
            assert!(fx.start_marking(&no_edges));
            fx.heap
                .finish_collection(&[zombie as usize, large_zombie as usize]);
            assert_eq!(FIN_COUNT.with(Cell::get), 4, "zombies finalize once");
            fx.collect(); // Unpinning releases storage, without another finalization.
            assert_eq!(FIN_COUNT.with(Cell::get), 4);
            assert!(!allocated(&fx, zombie as usize));
            assert!(!allocated(&fx, large_zombie as usize));
        }

        #[test]
        fn reverse_ephemeron_wakeups_and_conditional_side_slots_reach_their_descendants() {
            struct ChainClient {
                roots: [u64; 2],
                map: usize,
                keys: Vec<usize>,
                tail: usize,
                dead_owner: usize,
                dead_value: usize,
            }
            impl MarkClient for ChainClient {
                fn roots(&self, sink: &mut dyn FnMut(u64)) {
                    for &root in &self.roots {
                        sink(root);
                    }
                }
                fn side_slot_owners(&self, sink: &mut dyn FnMut(usize)) {
                    sink(*self.keys.last().unwrap());
                    sink(self.dead_owner);
                }
                fn trace_side_slots(&self, owner: usize, sink: &mut dyn FnMut(u64)) {
                    if owner == *self.keys.last().unwrap() {
                        sink(tagged(self.tail as *const u8));
                        sink(tagged(self.tail as *const u8)); // Repeated positive, after wakeups.
                    } else if owner == self.dead_owner {
                        sink(self.dead_value as u64);
                    }
                }
                fn ephemerons(&self, sink: &mut EdgeSink<'_>) {
                    for pair in self.keys.windows(2).rev() {
                        for _ in 0..2 {
                            sink.ephemeron(
                                self.map as *const u8,
                                tagged(pair[0] as *const u8),
                                tagged(pair[1] as *const u8),
                            );
                        }
                    }
                }
            }
            let mut fx = Fixture::new();
            fx.heap.set_collect_on_alloc(false);
            let map = fx.heap.alloc(fx.leaf, 8192);
            let keys: Vec<usize> = (0..9)
                .map(|index| {
                    fx.heap
                        .alloc(fx.leaf, if index % 3 == 2 { 8192 } else { 16 })
                        as usize
                })
                .collect();
            let bridge = fx.alloc_node(tagged(keys[0] as *const u8), 1);
            let child = fx.heap.alloc(fx.leaf, 16);
            let tail = fx.alloc_node(tagged(child), 2);
            let dead_owner = fx.heap.alloc(fx.leaf, 16);
            let dead_value = fx.alloc_node(tagged(dead_owner), 3);
            let mut client = ChainClient {
                roots: [tagged(map), tagged(bridge)],
                map: map as usize,
                keys,
                tail: tail as usize,
                dead_owner: dead_owner as usize,
                dead_value: dead_value as usize,
            };
            let before = fx.heap.stats().cells_marked;
            assert!(fx.heap.start_marking(false, &client, CollectionKind::Major));
            fx.heap.finish_collection(&[]);
            assert_eq!(
                fx.heap.stats().cells_marked - before,
                (client.keys.len() + 4) as u64
            );
            assert!(fx.heap.owns_cell(child as usize, fx.leaf));
            assert!(client
                .keys
                .iter()
                .all(|&cell| fx.heap.owns_cell(cell, fx.leaf)));
            assert!(!allocated(&fx, dead_owner as usize));
            assert!(!allocated(&fx, dead_value as usize));
            client.roots[1] = 0;
            assert!(fx.heap.start_marking(false, &client, CollectionKind::Major));
            fx.heap.finish_collection(&[]);
            assert_eq!(
                fx.live(),
                1,
                "only the map remains when the first key loses its root"
            );
            assert!(!allocated(&fx, child as usize));
            assert!(!allocated(&fx, tail as usize));
        }
    }

    // Timing probes behind the S1 review numbers, run with
    // `cargo test --release -p trust-js --lib --features embed
    //  gc_core::tests::perf_ -- --ignored --nocapture --test-threads=1`.

    fn perf_heap() -> (Heap, DescId, DescId) {
        let mut heap = Heap::new();
        let node = heap.register_desc(LayoutDesc {
            name: "node",
            trace: trace_node,
            mark_trace: None,
            finalize: None,
            carded: false,
        });
        let blob = heap.register_desc(LayoutDesc {
            name: "blob",
            trace: trace_none,
            mark_trace: None,
            finalize: None,
            carded: false,
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
        // The allocation-triggered collections find the chain's head in this frame.
        let head = fx.build_chain(1 << 20);
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
        black_box((&top, &head));
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
                let noise = (0..32768u64).map(|i| i * 0x9e37_79b9);
                // Rooted like the Fixture: the objects and the noise are its client's roots.
                let roots: Vec<u64> = objects.into_iter().chain(noise).collect();
                let client = FixtureClient {
                    roots: &roots,
                    edges: &no_edges,
                };
                let mut collect = || {
                    assert!(heap.start_marking(false, &client, CollectionKind::Major));
                    heap.finish_collection(&[]);
                };
                collect();
                let start = Instant::now();
                collect();
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
