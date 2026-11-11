//! The engine shell's allocator: per-thread size-class caches over spans of the allocator's own
//! pages (std-only).
//!
//! The engine's workloads are allocation-bound in exactly the way general-purpose system
//! allocators are slowest: millions of short-lived, same-sized blocks (spilled `Props` and
//! binding storage, property keys, argument lists, the parser's trees, boxed internal state;
//! objects, environments and strings are GC cells in pages of their own).
//!
//! Small blocks (≤ `MAX_CLASS` bytes, alignment ≤ 16) are rounded up to a 16-byte size class and
//! served from a per-thread INTRUSIVE free list: a freed block's first word points at the next
//! free block, so an allocation the list serves, or a free it takes, locks nothing. Large or
//! over-aligned requests pass straight through to [`std::alloc::System`].
//!
//! Class blocks come from spans: 16 KiB of one class at a boundary of their size (a span is never
//! smaller than the system's page: `span_for_page`), carved from `SEGMENT`s mapped from the OS,
//! never from the system allocator. (Each miss and each overflow of a thread's list was one call
//! of the system allocator a block, and once a process has a second thread glibc's malloc takes
//! its locking path on every such call: the tier's compile thread cost CodeLoad's parse bursts
//! about 10%.) The shape is JSC's libpas's: a thread allocates from a span it holds and frees
//! into a thread-local log flushed in batches (`pas_local_allocator`, `pas_thread_local_cache`):
//! - a thread's miss takes the next block of its bump region of the class, a span's tail it
//!   holds, with nothing to lock. With the region spent it refills under the class's lock: the
//!   blocks returned to the class's spans with blocks to hand out become its list (64 KiB's
//!   worth, from 16 spans at most), and such a span's tail its region; else an empty span,
//!   from the pool or a new segment, becomes its region;
//! - a free that finds the thread's list at its cap returns the colder half of the list to the
//!   blocks' spans (each found by masking the block's address), a few dozen runs of one span's
//!   blocks at a time under the class's lock;
//! - a span whose blocks are all back goes to the pool of empty spans, for any class. A segment
//!   whose spans are all in the pool is free: past `POOL_COMMITTED_BYTES` of empty spans the
//!   oldest free segments go back to the OS whole, and at [`trim`] (after a major collection)
//!   all of them. Where a span's memory can go back alone (below), the pool's oldest spans past
//!   the bound, and all of them at a trim, then have their memory returned to the OS and their
//!   ranges kept for reuse.
//!
//! A segment's first span holds the segment's header and its spans' headers (64 bytes a span), so
//! a span's memory can go back to the OS alone: a span is a whole number of the system's pages
//! (on a Linux kernel of 64 KiB pages spans are 64 KiB), so that none of its neighbours' memory
//! goes with it.
//!
//! On Linux each segment is advised for transparent huge pages (as the collector advises its
//! chunks once its heap is large enough): size classes put the blocks a program makes together
//! (a compiled chunk's `Vec`s of ops, names and caches, adjacent in glibc's heap) on as many
//! pages as they have classes, and with base pages the L1 data TLB missed twice as often as
//! with glibc on Vue's passes (+3% cycles on an A725 core, Djot's +1.5%); one huge page a
//! segment maps all its spans with one TLB entry. A huge page makes its whole segment resident
//! (about 2 MiB more RSS for a small program), which spans of 16 KiB keep small: a class with a
//! few blocks holds one 16 KiB span, not 64 KiB. A huge page's memory goes back to the OS only
//! whole: a span's discard inside it splits the page's mapping and frees nothing until the
//! kernel splits the page itself (under memory pressure), which costs the segment its one TLB
//! entry, and khugepaged may collapse the range into a huge page again. So where the system
//! grants huge pages of a segment's size (`os_pages::huge_page_size`), no span's memory goes back
//! alone: segments go back whole, once free.
//!
//! Threads: a block freed on another thread than the one it was allocated on joins the freeing
//! thread's list, and goes back to its span at that thread's overflow, trim or exit; spans are
//! global, under their class's lock. A thread's exit (`ExitDrain`) returns its lists and regions
//! to their spans; after that its frees go back to their spans at once and its allocations take
//! one block at a time from them. (Windows ends a process by ending its other threads wherever
//! they are, a lock they held held for good, and then runs the exiting thread's destructors: see
//! `ending`.)
//!
//! The allocator never allocates: its lists and span headers are intrusive, its memory is mapped
//! directly, and its locks never allocate (std's mutex where it is a futex or an SRW lock;
//! `os_unfair_lock` on Apple's systems, where std's mutex boxes a pthread mutex at its first
//! lock; elsewhere a spin lock that yields).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, UnsafeCell};
use std::mem::size_of;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU32, Ordering};

/// Largest cached block size, in bytes.
const MAX_CLASS: usize = 1024;
/// 16-byte class granularity (also the maximum supported alignment for cached blocks).
const STEP: usize = 16;
const NUM_CLASSES: usize = MAX_CLASS / STEP;
/// Maximum cached storage for one size class. Keeping the bound in bytes rather than items is
/// important: a 65,536-item limit allowed the 64 classes to retain just over 2 GiB per thread.
/// This bounds the complete thread-local cache to 16 MiB while preserving a deeper free list for
/// the small, hot allocation classes.
const BYTES_PER_CLASS: usize = 256 * 1024;

const fn class_caps() -> [usize; NUM_CLASSES] {
    let mut caps = [0; NUM_CLASSES];
    let mut class = 0;
    while class < NUM_CLASSES {
        caps[class] = BYTES_PER_CLASS / ((class + 1) * STEP);
        class += 1;
    }
    caps
}

const CLASS_CAPS: [usize; NUM_CLASSES] = class_caps();

/// The smallest and the largest span (see [`span_for_page`]).
const MIN_SPAN: usize = 16 * 1024;
const MAX_SPAN: usize = 64 * 1024;
/// The size and alignment of a segment, the spans mapped from the OS together (one huge page
/// where base pages are 4 KiB). Its first span holds the headers of the segment and its spans.
const SEGMENT: usize = 2 * 1024 * 1024;
/// A refill gathers the blocks returned to its class's spans until it holds this many bytes'
/// worth, from `REFILL_SPANS` spans at most.
const REFILL_BYTES: usize = 64 * 1024;
const REFILL_SPANS: usize = 16;
/// The empty spans' bytes the pool keeps committed between trims (4 MiB): past them its oldest
/// free segments go back to the OS, and where a span's memory goes back alone, its oldest spans'.
const POOL_COMMITTED_BYTES: usize = 4 << 20;
/// The runs of one span's blocks gathered outside the class's lock and spliced under it at a
/// time, when blocks go back to their spans.
const RUNS: usize = 32;

/// The span size for the system's page size `page`, and whether a span's memory can go back to
/// the OS alone: `MIN_SPAN`, or the page where that is larger, up to `MAX_SPAN`. A discard is
/// whole pages (a 16 KiB span's on a kernel of 64 KiB pages would discard three neighbours'
/// blocks with it, or be refused), so with larger pages (or a size not a power of two) spans of
/// `MIN_SPAN` keep their memory until their segment goes back.
const fn span_for_page(page: usize) -> (usize, bool) {
    if !page.is_power_of_two() || page > MAX_SPAN {
        (MIN_SPAN, false)
    } else if page < MIN_SPAN {
        (MIN_SPAN, true)
    } else {
        (page, true)
    }
}

/// How spans are laid out and returned, set at the first use from the system's answers (0 until
/// then): the span size's log2 in the low byte, and `ALONE` where a span's memory goes back to
/// the OS alone (`span_for_page`, and no huge page backs a segment).
static SPAN_BITS: AtomicU32 = AtomicU32::new(0);
const SHIFT_MASK: u32 = 0xff;
const ALONE: u32 = 1 << 8;

#[inline]
fn span_bits() -> u32 {
    let bits = SPAN_BITS.load(Ordering::Relaxed);
    if bits != 0 {
        bits
    } else {
        init_span_bits()
    }
}

/// The system's answers are the same at every call: threads that race store one value.
#[cold]
#[inline(never)]
fn init_span_bits() -> u32 {
    let (span, alone) = span_for_page(segments::page_size());
    let bits = span.trailing_zeros() | if alone && !segments::huge() { ALONE } else { 0 };
    SPAN_BITS.store(bits, Ordering::Relaxed);
    bits
}

/// The log2 of the size and alignment of a span, the blocks of one class.
#[inline]
fn span_shift() -> u32 {
    span_bits() & SHIFT_MASK
}

/// The size and alignment of a span.
#[inline]
fn span_size() -> usize {
    1 << span_shift()
}

/// Whether a span's memory goes back to the OS alone (see the module docs).
#[inline]
fn alone() -> bool {
    span_bits() & ALONE != 0
}

/// Where a span is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Place {
    /// Every block handed out: in use, or in a thread's list or region. (A zeroed header's.)
    Out = 0,
    /// On its class's list of spans with blocks to hand out.
    Partial,
    /// In the pool, empty, its memory committed.
    Committed,
    /// In the pool, empty, its memory returned to the OS.
    Decommitted,
}

/// A span's header. Each segment's first span holds the headers of its spans: the `k`th span from
/// the segment's start is described by the header at `k * 64` (the first slot holds the
/// segment's own header, a [`Segment`]). A header is guarded by its class's lock while its span
/// holds blocks of the class, by the pool's lock while the span is in the pool, and by nothing
/// while a thread sets up a span taken from the pool (no other thread can reach it then).
#[repr(C, align(64))]
struct Span {
    /// The links of its class's list of spans with blocks to hand out, or of the pool's lists.
    next: *mut Span,
    prev: *mut Span,
    /// The blocks returned to the span: a list through each block's first word, and its last.
    returned: *mut u8,
    returned_last: *mut u8,
    returned_count: u32,
    /// The blocks handed out: in use, or in a thread's list or region.
    out: u32,
    /// The offset of the span's first block never handed out, which starts its tail, and the end
    /// of its last block.
    cursor: u32,
    end: u32,
    class: u8,
    place: Place,
    /// The tail's blocks read as zero (the span came fresh from the OS, or back from a decommit
    /// that zeroes).
    tail_zero: bool,
}

/// A segment's header, in the first slot of its first span (its spans' headers follow), guarded
/// by the pool's lock.
#[repr(C, align(64))]
struct Segment {
    /// The links of the pool's list of the segments mapped.
    next: *mut Segment,
    prev: *mut Segment,
    /// The links of the pool's list of free segments, while every span carved from the segment
    /// is in the pool.
    next_free: *mut Segment,
    prev_free: *mut Segment,
    /// The spans carved from the segment (those after its first, in order), and how many of them
    /// are in the pool's lists.
    carved: u32,
    pooled: u32,
    /// On the pool's list of free segments.
    free: bool,
}

// The headers fit in a segment's first span, at any span size; a segment holds spans of the
// largest.
const _: () = assert!(
    size_of::<Span>() == 64
        && size_of::<Segment>() == 64
        && SEGMENT / MIN_SPAN * 64 <= MIN_SPAN
        && 4 * MAX_SPAN <= SEGMENT
);

/// The header of the span holding the address `at`.
#[inline]
fn span_of(at: usize) -> *mut Span {
    let segment = at & !(SEGMENT - 1);
    (segment + ((at - segment) >> span_shift()) * size_of::<Span>()) as *mut Span
}

/// The start of the span `span` describes.
#[inline]
fn base_of(span: *mut Span) -> usize {
    let at = span as usize;
    let segment = at & !(SEGMENT - 1);
    segment + (((at - segment) / size_of::<Span>()) << span_shift())
}

/// The header of the segment holding `span`.
#[inline]
fn segment_of(span: *mut Span) -> *mut Segment {
    (span as usize & !(SEGMENT - 1)) as *mut Segment
}

#[inline]
fn class_size(class: usize) -> usize {
    (class + 1) * STEP
}

/// The end of a span's last block of the class.
#[inline]
fn class_end(class: usize) -> u32 {
    (span_size() / class_size(class) * class_size(class)) as u32
}

/// A lock that never allocates (see the module docs).
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "windows",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "dragonfly",
    target_os = "fuchsia"
))]
mod lock {
    use std::sync::{Mutex, MutexGuard, PoisonError};

    /// std's mutex, a futex here (or an SRW lock), which never allocates.
    pub(super) struct Lock(Mutex<()>);

    impl Lock {
        pub(super) const fn new() -> Self {
            Self(Mutex::new(()))
        }

        pub(super) fn lock(&self) -> MutexGuard<'_, ()> {
            // Nothing panics under the lock but a test's assertion; the state stays consistent.
            self.0.lock().unwrap_or_else(PoisonError::into_inner)
        }
    }
}

/// A lock that never allocates (see the module docs).
#[cfg(target_vendor = "apple")]
mod lock {
    use std::cell::UnsafeCell;

    unsafe extern "C" {
        fn os_unfair_lock_lock(lock: *mut u32);
        fn os_unfair_lock_unlock(lock: *mut u32);
    }

    /// An `os_unfair_lock`: std's mutex is a pthread mutex boxed at its first lock here, which
    /// would allocate inside the allocator.
    pub(super) struct Lock(UnsafeCell<u32>);

    // SAFETY: the system's lock functions are the word's only users.
    unsafe impl Sync for Lock {}

    pub(super) struct Guard<'a>(&'a Lock);

    impl Lock {
        pub(super) const fn new() -> Self {
            // OS_UNFAIR_LOCK_INIT.
            Self(UnsafeCell::new(0))
        }

        pub(super) fn lock(&self) -> Guard<'_> {
            // SAFETY: an initialized lock word, which the guard unlocks on this thread.
            unsafe { os_unfair_lock_lock(self.0.get()) };
            Guard(self)
        }
    }

    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            // SAFETY: locked by this thread in `lock`.
            unsafe { os_unfair_lock_unlock(self.0 .0.get()) };
        }
    }
}

/// A lock that never allocates (see the module docs).
#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "windows",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "dragonfly",
    target_os = "fuchsia",
    target_vendor = "apple"
)))]
mod lock {
    pub(super) use super::spin::SpinLock as Lock;
}

/// A spin lock that yields the processor while it waits: the lock of the systems whose std
/// mutex allocates (and tested everywhere).
#[cfg(any(
    test,
    not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "dragonfly",
        target_os = "fuchsia",
        target_vendor = "apple"
    ))
))]
mod spin {
    use std::sync::atomic::{AtomicBool, Ordering};

    pub(super) struct SpinLock(AtomicBool);

    pub(super) struct SpinGuard<'a>(&'a SpinLock);

    impl SpinLock {
        pub(super) const fn new() -> Self {
            Self(AtomicBool::new(false))
        }

        pub(super) fn lock(&self) -> SpinGuard<'_> {
            let mut spins = 0;
            while self
                .0
                .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                while self.0.load(Ordering::Relaxed) {
                    if spins < 64 {
                        spins += 1;
                        std::hint::spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
            }
            SpinGuard(self)
        }
    }

    impl Drop for SpinGuard<'_> {
        fn drop(&mut self) {
            self.0 .0.store(false, Ordering::Release);
        }
    }
}

use lock::Lock;

/// Where segments come from: memory mapped from the OS (the collector's `os_pages`).
#[cfg(all(
    any(target_os = "linux", target_os = "macos", target_os = "windows"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod segments {
    use super::SEGMENT;
    use crate::gc_core::os_pages;
    pub(super) use os_pages::{decommit, page_size, recommit, DECOMMIT_ZEROES};

    /// Whether the system backs a segment with one huge page (see the module docs).
    #[cfg(target_os = "linux")]
    pub(super) fn huge() -> bool {
        os_pages::huge_page_size() == Some(SEGMENT)
    }

    #[cfg(not(target_os = "linux"))]
    pub(super) fn huge() -> bool {
        false
    }

    /// Return a segment `map` gave to the OS.
    ///
    /// # Safety
    /// Nothing uses the segment's memory any more.
    pub(super) unsafe fn unmap(segment: *mut u8) {
        // SAFETY: the caller's contract; mapped with this size.
        unsafe { os_pages::unmap(segment, SEGMENT) };
    }

    /// A new segment: `SEGMENT` zeroed bytes at a `SEGMENT` boundary, on Linux advised for
    /// transparent huge pages (see the module docs); null when the system refuses.
    pub(super) fn map() -> *mut u8 {
        let ptr = map_aligned();
        #[cfg(target_os = "linux")]
        if !ptr.is_null() {
            os_pages::advise_huge_pages(ptr, SEGMENT);
        }
        ptr
    }

    /// Linux places a mapping of a huge page's size at a huge page's boundary, so the plain
    /// mapping usually serves; else the aligned mapping maps twice the size for a moment.
    fn map_aligned() -> *mut u8 {
        let ptr = os_pages::map(SEGMENT);
        if ptr.is_null() || (ptr as usize).is_multiple_of(SEGMENT) {
            return ptr;
        }
        // SAFETY: just mapped, with this size, and unused.
        unsafe { os_pages::unmap(ptr, SEGMENT) };
        os_pages::map_aligned(SEGMENT, SEGMENT)
    }
}

/// Where segments come from elsewhere: the system allocator, which keeps their memory.
#[cfg(not(all(
    any(target_os = "linux", target_os = "macos", target_os = "windows"),
    any(target_arch = "aarch64", target_arch = "x86_64")
)))]
mod segments {
    use super::SEGMENT;
    use std::alloc::{GlobalAlloc, Layout, System};

    pub(super) const DECOMMIT_ZEROES: bool = false;

    pub(super) fn page_size() -> usize {
        4096
    }

    pub(super) fn huge() -> bool {
        false
    }

    pub(super) fn map() -> *mut u8 {
        // SAFETY: a nonzero size and a power-of-two alignment.
        unsafe { System.alloc_zeroed(Layout::from_size_align_unchecked(SEGMENT, SEGMENT)) }
    }

    pub(super) unsafe fn unmap(segment: *mut u8) {
        // SAFETY: allocated by `map` with this layout, and unused.
        unsafe { System.dealloc(segment, Layout::from_size_align_unchecked(SEGMENT, SEGMENT)) }
    }

    pub(super) unsafe fn decommit(_ptr: *mut u8, _len: usize) -> bool {
        false
    }

    pub(super) unsafe fn recommit(_ptr: *mut u8, _len: usize) -> bool {
        true
    }
}

/// A class's spans with blocks to hand out (returned blocks or a tail), under the class's lock:
/// a list through the headers' `next` and `prev`, the span blocks came back to last first.
#[repr(align(64))]
struct ClassSpans {
    lock: Lock,
    first: UnsafeCell<*mut Span>,
}

// SAFETY: `first` and the headers on its list are touched only under `lock`.
unsafe impl Sync for ClassSpans {}

static CLASSES: [ClassSpans; NUM_CLASSES] = [const {
    ClassSpans {
        lock: Lock::new(),
        first: UnsafeCell::new(null_mut()),
    }
}; NUM_CLASSES];

/// The empty spans and the segments, under the pool's lock.
struct Pool {
    lock: Lock,
    state: UnsafeCell<PoolState>,
}

struct PoolState {
    /// Empty spans whose memory is committed: a list through `next` and `prev`, the most recently
    /// emptied first, and its last.
    committed: *mut Span,
    committed_last: *mut Span,
    committed_len: usize,
    /// Empty spans whose memory went back to the OS: a list through `next` and `prev`.
    decommitted: *mut Span,
    decommitted_len: usize,
    /// The newest segment's spans never taken: `carve..carve_end`.
    carve: usize,
    carve_end: usize,
    /// The segments mapped: a list through their headers' `next` and `prev`, the newest first.
    segments: *mut Segment,
    /// The free segments: a list through `next_free` and `prev_free`, the most recently freed
    /// first, and its last.
    free: *mut Segment,
    free_last: *mut Segment,
}

// SAFETY: `state`, and the headers of the spans on its lists, are touched only under `lock`.
unsafe impl Sync for Pool {}

static POOL: Pool = Pool {
    lock: Lock::new(),
    state: UnsafeCell::new(PoolState {
        committed: null_mut(),
        committed_last: null_mut(),
        committed_len: 0,
        decommitted: null_mut(),
        decommitted_len: 0,
        carve: 0,
        carve_end: 0,
        segments: null_mut(),
        free: null_mut(),
        free_last: null_mut(),
    }),
};

impl PoolState {
    /// Put `span`, empty and off every list, first in the committed list.
    ///
    /// # Safety
    /// The pool's lock is held, and no thread reaches `span` but through it.
    unsafe fn push_committed(&mut self, span: *mut Span) {
        // SAFETY: the caller's contract.
        unsafe {
            (*span).place = Place::Committed;
            (*span).prev = null_mut();
            (*span).next = self.committed;
            if self.committed.is_null() {
                self.committed_last = span;
            } else {
                (*self.committed).prev = span;
            }
            self.committed = span;
            self.committed_len += 1;
            self.count_in(span);
        }
    }

    /// Put `span`, empty and off every list, its memory returned, first in the decommitted list.
    ///
    /// # Safety
    /// As `push_committed`.
    unsafe fn push_decommitted(&mut self, span: *mut Span) {
        // SAFETY: the caller's contract.
        unsafe {
            (*span).place = Place::Decommitted;
            (*span).prev = null_mut();
            (*span).next = self.decommitted;
            if !self.decommitted.is_null() {
                (*self.decommitted).prev = span;
            }
            self.decommitted = span;
            self.decommitted_len += 1;
            self.count_in(span);
        }
    }

    /// Take `span` off the pool's list it is on, for a thread.
    ///
    /// # Safety
    /// The pool's lock is held, and `span` is on one of its lists.
    unsafe fn remove(&mut self, span: *mut Span) {
        // SAFETY: the caller's contract.
        unsafe {
            self.unlist(span);
            let segment = segment_of(span);
            if (*segment).free {
                self.unlink_free(segment);
            }
            (*segment).pooled -= 1;
        }
    }

    /// Take `span` off the pool's list it is on, its segment's count left as it was.
    ///
    /// # Safety
    /// As `remove`.
    unsafe fn unlist(&mut self, span: *mut Span) {
        // SAFETY: the caller's contract.
        unsafe {
            let (prev, next) = ((*span).prev, (*span).next);
            if (*span).place == Place::Committed {
                if prev.is_null() {
                    self.committed = next;
                } else {
                    (*prev).next = next;
                }
                if next.is_null() {
                    self.committed_last = prev;
                } else {
                    (*next).prev = prev;
                }
                self.committed_len -= 1;
            } else {
                debug_assert_eq!((*span).place, Place::Decommitted);
                if prev.is_null() {
                    self.decommitted = next;
                } else {
                    (*prev).next = next;
                }
                if !next.is_null() {
                    (*next).prev = prev;
                }
                self.decommitted_len -= 1;
            }
            (*span).place = Place::Out;
        }
    }

    /// One more span of `span`'s segment is in the pool: once every span carved is, the segment
    /// goes first on the list of free segments.
    ///
    /// # Safety
    /// The pool's lock is held.
    unsafe fn count_in(&mut self, span: *mut Span) {
        // SAFETY: the caller's contract; the segment's header is the pool's.
        unsafe {
            let segment = segment_of(span);
            (*segment).pooled += 1;
            debug_assert!((*segment).pooled <= (*segment).carved);
            if (*segment).pooled == (*segment).carved {
                (*segment).free = true;
                (*segment).prev_free = null_mut();
                (*segment).next_free = self.free;
                if self.free.is_null() {
                    self.free_last = segment;
                } else {
                    (*self.free).prev_free = segment;
                }
                self.free = segment;
            }
        }
    }

    /// Take `segment` off the list of free segments.
    ///
    /// # Safety
    /// The pool's lock is held, and the segment is on the list.
    unsafe fn unlink_free(&mut self, segment: *mut Segment) {
        // SAFETY: the caller's contract.
        unsafe {
            let (prev, next) = ((*segment).prev_free, (*segment).next_free);
            if prev.is_null() {
                self.free = next;
            } else {
                (*prev).next_free = next;
            }
            if next.is_null() {
                self.free_last = prev;
            } else {
                (*next).prev_free = prev;
            }
            (*segment).next_free = null_mut();
            (*segment).prev_free = null_mut();
            (*segment).free = false;
        }
    }

    /// The oldest free segment, taken off every list and its spans off the pool's, for the
    /// caller to return to the OS (linked by `next` with the others it collects); null when no
    /// segment is free.
    ///
    /// # Safety
    /// The pool's lock is held.
    unsafe fn take_free(&mut self) -> *mut Segment {
        // SAFETY: the caller's contract: the segment's spans are all on the pool's lists.
        unsafe {
            let segment = self.free_last;
            if segment.is_null() {
                return segment;
            }
            self.unlink_free(segment);
            let first = segment as usize;
            for k in 1..=(*segment).carved as usize {
                self.unlist(span_of(first + (k << span_shift())));
            }
            let (prev, next) = ((*segment).prev, (*segment).next);
            if prev.is_null() {
                self.segments = next;
            } else {
                (*prev).next = next;
            }
            if !next.is_null() {
                (*next).prev = prev;
            }
            if self.carve_end == first + SEGMENT {
                // The newest segment: the next span is a new segment's.
                self.carve = 0;
                self.carve_end = 0;
            }
            segment
        }
    }

    /// Take the free segments (see `take_free`), linked by `next`: all of them, or as many as
    /// bring the committed spans down to `bound`.
    ///
    /// # Safety
    /// The pool's lock is held.
    unsafe fn take_free_segments(&mut self, bound: Option<usize>) -> *mut Segment {
        let mut taken = null_mut::<Segment>();
        while bound.is_none_or(|bound| self.committed_len > bound) {
            // SAFETY: the caller's contract.
            let segment = unsafe { self.take_free() };
            if segment.is_null() {
                break;
            }
            // SAFETY: taken off every list: the caller's.
            unsafe { (*segment).next = taken };
            taken = segment;
        }
        taken
    }
}

/// The per-thread lists and regions. The type has no destructor, so the const-initialized
/// thread-local is a plain TLS access on every allocation (a destructor would add a lazy
/// registration-state check to each one). Teardown is delegated to [`ExitDrain`].
struct Cache {
    heads: [Cell<*mut u8>; NUM_CLASSES],
    counts: [Cell<usize>; NUM_CLASSES],
    /// Each class's bump region, the tail of a span the thread holds: the blocks from `bump` to
    /// `bump_end`.
    bump: [Cell<usize>; NUM_CLASSES],
    bump_end: [Cell<usize>; NUM_CLASSES],
    /// Bit `class` set: that class's region reads as zero.
    bump_zero: Cell<u64>,
    /// `ExitDrain` has been registered for this thread.
    registered: Cell<bool>,
    /// This thread's `ExitDrain` has run: frees go straight back to their spans from now on.
    exited: Cell<bool>,
}

thread_local! {
    static CACHE: Cache = const { Cache {
        heads: [const { Cell::new(null_mut()) }; NUM_CLASSES],
        counts: [const { Cell::new(0) }; NUM_CLASSES],
        bump: [const { Cell::new(0) }; NUM_CLASSES],
        bump_end: [const { Cell::new(0) }; NUM_CLASSES],
        bump_zero: Cell::new(0),
        registered: Cell::new(false),
        exited: Cell::new(false),
    } };
    static EXIT_DRAIN: ExitDrain = const { ExitDrain };
}

/// Returns a thread's lists and regions to their spans when the thread exits. Registered lazily,
/// at the thread's first free or refill.
struct ExitDrain;

impl Drop for ExitDrain {
    fn drop(&mut self) {
        // The process is ending (see `ending`): the thread's lists and regions stay as they are and
        // serve it until the end.
        #[cfg(windows)]
        if ending::process_is_shutting_down() {
            return;
        }
        let _ = CACHE.try_with(|cache| {
            cache.exited.set(true);
            drain(cache);
        });
    }
}

/// Windows ends a process (ExitProcess) by ending its other threads wherever they are, one
/// perhaps holding a class's lock or the pool's (`take_span` holds it across VirtualAlloc), and
/// then runs the exiting thread's thread-local destructors (std's FLS callback), where a drain
/// would wait for that lock for good. They run in the loader's shutdown, which ExitProcess begins
/// once the other threads are ended (LdrShutdownProcess: `RtlDllShutdownInProgress` is TRUE from
/// then on), on both ways out of a process: the return from `main` (the CRT's `exit`, then
/// ExitProcess) and `std::process::exit` (ExitProcess at once). So a thread's exit drain returns
/// at once while the loader says the process is shutting down, as libpas's thread-local cache's
/// destructor does (pas_thread_local_cache.c, `destructor`; pas_process.c,
/// `pas_process_is_shutting_down`), and the thread's lists and regions keep serving it until the
/// end. Elsewhere a process's end leaves its threads running until it is over, and their locks
/// are released as ever.
#[cfg(windows)]
mod ending {
    use std::ffi::c_void;
    use std::ptr::null_mut;
    use std::sync::atomic::{AtomicPtr, Ordering};

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetModuleHandleW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
    }

    const fn wide<const N: usize>(text: &[u8; N]) -> [u16; N] {
        let mut wide = [0; N];
        let mut k = 0;
        while k < N {
            wide[k] = text[k] as u16;
            k += 1;
        }
        wide
    }

    const NTDLL: [u16; 10] = wide(b"ntdll.dll\0");

    /// ntdll's `RtlDllShutdownInProgress`, resolved once: null until then, `ABSENT` where ntdll
    /// does not export it (the process's end is then not told, as in libpas).
    static SHUTDOWN_IN_PROGRESS: AtomicPtr<c_void> = AtomicPtr::new(null_mut());
    pub(super) const ABSENT: *mut c_void = std::ptr::without_provenance_mut(1);

    /// Resolve `RtlDllShutdownInProgress`: at the first thread's registration, so that the loader
    /// is not asked while the process ends.
    pub(super) fn resolve() -> *mut c_void {
        let resolved = SHUTDOWN_IN_PROGRESS.load(Ordering::Relaxed);
        if !resolved.is_null() {
            return resolved;
        }
        // SAFETY: a NUL-terminated module name, and a NUL-terminated export name of a module
        // that stays loaded.
        let function = unsafe {
            let ntdll = GetModuleHandleW(NTDLL.as_ptr());
            if ntdll.is_null() {
                null_mut()
            } else {
                GetProcAddress(ntdll, c"RtlDllShutdownInProgress".as_ptr().cast())
            }
        };
        let resolved = if function.is_null() { ABSENT } else { function };
        SHUTDOWN_IN_PROGRESS.store(resolved, Ordering::Relaxed);
        resolved
    }

    /// Whether the process is ending: ExitProcess's loader shutdown has begun.
    pub(super) fn process_is_shutting_down() -> bool {
        let resolved = resolve();
        if resolved == ABSENT {
            return false;
        }
        // SAFETY: ntdll's `BOOLEAN NTAPI RtlDllShutdownInProgress(void)`.
        unsafe {
            let function: unsafe extern "system" fn() -> u8 = std::mem::transmute(resolved);
            function() != 0
        }
    }
}

/// Register the thread's [`ExitDrain`], which may allocate. False when the thread's destructors
/// already destroyed it: the thread caches nothing from then on.
#[cold]
#[inline(never)]
fn register(cache: &Cache) -> bool {
    cache.registered.set(true);
    #[cfg(windows)]
    ending::resolve();
    if EXIT_DRAIN.try_with(|_| ()).is_ok() {
        return true;
    }
    cache.exited.set(true);
    false
}

#[inline]
fn class_of(size: usize, align: usize) -> Option<usize> {
    if align <= STEP && size <= MAX_CLASS && size > 0 {
        Some(size.div_ceil(STEP) - 1)
    } else {
        None
    }
}

/// A class allocation the thread's list could not serve: the next block of the thread's bump
/// region of the class, else a refill. The block, and whether it reads as zero; null when the OS
/// refuses memory.
#[inline(never)]
fn alloc_miss(class: usize) -> (*mut u8, bool) {
    CACHE.with(|cache| loop {
        // The list first (a refill fills it): the region's tail may be pages never touched.
        let p = cache.heads[class].get();
        if !p.is_null() {
            // SAFETY: a block of the thread's list, whose first word links the next.
            cache.heads[class].set(unsafe { *(p as *mut *mut u8) });
            cache.counts[class].set(cache.counts[class].get() - 1);
            return (p, false);
        }
        let next = cache.bump[class].get();
        if next < cache.bump_end[class].get() {
            cache.bump[class].set(next + class_size(class));
            return (next as *mut u8, cache.bump_zero.get() & 1 << class != 0);
        }
        if cache.exited.get() {
            return alloc_unowned(class);
        }
        if !cache.registered.get() {
            // Registering may allocate, and refill this class: look again.
            register(cache);
            continue;
        }
        if !refill(cache, class) {
            return (null_mut(), false);
        }
    })
}

/// Refill the thread's empty list and spent region of `class`, under the class's lock: the blocks
/// returned to the class's spans with blocks to hand out, `REFILL_BYTES`' worth from
/// `REFILL_SPANS` spans at most, become the list, and one such span's tail the region; with none,
/// an empty span (the pool's or a new segment's) becomes the region whole. False when the OS
/// refuses memory.
fn refill(cache: &Cache, class: usize) -> bool {
    let size = class_size(class);
    let spans = &CLASSES[class];
    let (mut first, mut last, mut count) = (null_mut::<u8>(), null_mut::<u8>(), 0);
    let mut region = None;
    {
        let _guard = spans.lock.lock();
        // SAFETY: the class's list and its spans' headers, under the class's lock.
        unsafe {
            let mut taken = 0;
            while count * size < REFILL_BYTES && taken < REFILL_SPANS {
                let span = *spans.first.get();
                if span.is_null() {
                    break;
                }
                let tail = (*span).cursor < (*span).end;
                if tail && region.is_some() {
                    break;
                }
                if (*span).returned_count != 0 {
                    if first.is_null() {
                        first = (*span).returned;
                    } else {
                        *(last as *mut *mut u8) = (*span).returned;
                    }
                    last = (*span).returned_last;
                    count += (*span).returned_count as usize;
                    (*span).out += (*span).returned_count;
                    (*span).returned = null_mut();
                    (*span).returned_last = null_mut();
                    (*span).returned_count = 0;
                }
                if tail {
                    let base = base_of(span);
                    region = Some((
                        base + (*span).cursor as usize,
                        base + (*span).end as usize,
                        (*span).tail_zero,
                    ));
                    (*span).out += ((*span).end - (*span).cursor) / size as u32;
                    (*span).cursor = (*span).end;
                }
                unlink(class, span);
                taken += 1;
            }
        }
    }
    if first.is_null() && region.is_none() {
        let (span, zero) = take_span();
        if span.is_null() {
            return false;
        }
        let end = class_end(class);
        // SAFETY: an empty span taken from the pool, which no other thread reaches.
        unsafe {
            setup(span, class);
            (*span).cursor = end;
            (*span).out = end / size as u32;
        }
        let base = base_of(span);
        region = Some((base, base + end as usize, zero));
    }
    cache.heads[class].set(first);
    cache.counts[class].set(count);
    if let Some((from, to, zero)) = region {
        cache.bump[class].set(from);
        cache.bump_end[class].set(to);
        let bit = 1 << class;
        cache.bump_zero.set(if zero {
            cache.bump_zero.get() | bit
        } else {
            cache.bump_zero.get() & !bit
        });
    }
    true
}

/// Set up the header of an empty span for `class`, nothing handed out yet (the caller hands
/// out its tail).
///
/// # Safety
/// `span` is an empty span taken from the pool, which no other thread reaches.
unsafe fn setup(span: *mut Span, class: usize) {
    // SAFETY: the caller's contract.
    unsafe {
        (*span).next = null_mut();
        (*span).prev = null_mut();
        (*span).returned = null_mut();
        (*span).returned_last = null_mut();
        (*span).returned_count = 0;
        (*span).out = 0;
        (*span).cursor = 0;
        (*span).end = class_end(class);
        (*span).class = class as u8;
        (*span).place = Place::Out;
        (*span).tail_zero = false;
    }
}

/// A class block for a thread past its exit drain, which keeps no list or region: one block,
/// under the class's lock, from a span with blocks to hand out, else from an empty span set up
/// for the class.
fn alloc_unowned(class: usize) -> (*mut u8, bool) {
    let size = class_size(class);
    {
        let spans = &CLASSES[class];
        let _guard = spans.lock.lock();
        // SAFETY: the class's list and its spans' headers, under the class's lock.
        unsafe {
            let span = *spans.first.get();
            if !span.is_null() {
                let block = if (*span).returned_count != 0 {
                    let p = (*span).returned;
                    (*span).returned = *(p as *mut *mut u8);
                    (*span).returned_count -= 1;
                    if (*span).returned_count == 0 {
                        (*span).returned_last = null_mut();
                    }
                    (p, false)
                } else {
                    let p = base_of(span) + (*span).cursor as usize;
                    (*span).cursor += size as u32;
                    (p as *mut u8, (*span).tail_zero)
                };
                (*span).out += 1;
                if (*span).returned_count == 0 && (*span).cursor == (*span).end {
                    unlink(class, span);
                }
                return block;
            }
        }
    }
    let (span, zero) = take_span();
    if span.is_null() {
        return (null_mut(), false);
    }
    // SAFETY: an empty span taken from the pool, which no other thread reaches until it is on
    // the class's list.
    unsafe {
        setup(span, class);
        (*span).cursor = size as u32;
        (*span).out = 1;
        (*span).tail_zero = zero;
        let _guard = CLASSES[class].lock.lock();
        link(class, span);
    }
    (base_of(span) as *mut u8, zero)
}

/// A class free the thread's list could not take: past the thread's exit drain the block goes
/// back to its span at once; else the list is at its cap, and its colder half goes back to the
/// spans (the half freed longest ago: the list is last freed first).
///
/// # Safety
/// `ptr` is a block of `class` this allocator gave, which nothing uses any more.
#[inline(never)]
unsafe fn dealloc_miss(ptr: *mut u8, class: usize) {
    CACHE.with(|cache| {
        if cache.exited.get() {
            // SAFETY: the caller's block, a list of one.
            unsafe {
                *(ptr as *mut *mut u8) = null_mut();
                give_back(class, ptr);
            }
            return;
        }
        let keep = CLASS_CAPS[class] / 2;
        debug_assert!(cache.counts[class].get() >= CLASS_CAPS[class]);
        let head = cache.heads[class].get();
        // SAFETY: the thread's list holds at least `keep` blocks, each linking the next; the
        // colder part cut off it is the thread's to give back.
        unsafe {
            let mut last_kept = head;
            for _ in 1..keep {
                last_kept = *(last_kept as *mut *mut u8);
            }
            let colder = std::mem::replace(&mut *(last_kept as *mut *mut u8), null_mut());
            cache.counts[class].set(keep);
            give_back(class, colder);
            *(ptr as *mut *mut u8) = head;
        }
        cache.heads[class].set(ptr);
        cache.counts[class].set(keep + 1);
    })
}

/// A run of blocks of one span, consecutive on a list being given back.
#[derive(Clone, Copy)]
struct Run {
    span: *mut Span,
    first: *mut u8,
    last: *mut u8,
    count: u32,
}

/// Give a list of blocks of `class` (linked through their first words, null-terminated) back to
/// their spans: its runs of one span's blocks are gathered outside the class's lock and spliced
/// onto their spans' returned lists under it, `RUNS` at a time; the spans emptied go to the pool.
///
/// # Safety
/// The list's blocks are blocks of `class` this allocator gave, which nothing uses any more.
unsafe fn give_back(class: usize, mut p: *mut u8) {
    let mut runs = [Run {
        span: null_mut(),
        first: null_mut(),
        last: null_mut(),
        count: 0,
    }; RUNS];
    let mut emptied = null_mut();
    while !p.is_null() {
        let mut n = 0;
        while n < RUNS && !p.is_null() {
            let span = span_of(p as usize);
            let (first, mut last, mut count) = (p, p, 1);
            loop {
                // SAFETY: a block of the list, whose first word links the next.
                let next = unsafe { *(last as *mut *mut u8) };
                if next.is_null() || span_of(next as usize) != span {
                    p = next;
                    break;
                }
                last = next;
                count += 1;
            }
            runs[n] = Run {
                span,
                first,
                last,
                count,
            };
            n += 1;
        }
        let _guard = CLASSES[class].lock.lock();
        for run in &runs[..n] {
            // SAFETY: the runs' spans hold blocks of the class: their headers are the lock's.
            unsafe { splice(class, run, &mut emptied) };
        }
    }
    if !emptied.is_null() {
        // SAFETY: the spans emptied, which no thread reaches any more.
        unsafe { pool_put(emptied) };
    }
}

/// Under the class's lock: a run's blocks join their span's returned list.
///
/// # Safety
/// The class's lock is held; the run's blocks are out of its span and nothing uses them.
unsafe fn splice(class: usize, run: &Run, emptied: &mut *mut Span) {
    let span = run.span;
    // SAFETY: the caller's contract.
    unsafe {
        debug_assert_eq!((*span).class as usize, class);
        debug_assert!((*span).out >= run.count);
        *(run.last as *mut *mut u8) = (*span).returned;
        if (*span).returned.is_null() {
            (*span).returned_last = run.last;
        }
        (*span).returned = run.first;
        (*span).returned_count += run.count;
        (*span).out -= run.count;
        settle(class, span, emptied);
    }
}

/// Under the class's lock, after blocks or a tail came back to `span`: a span emptied leaves its
/// class for `emptied` (for the pool, once the lock is released); a span that had nothing to
/// hand out joins its class's list.
///
/// # Safety
/// The class's lock is held, and `span` holds blocks of the class.
unsafe fn settle(class: usize, span: *mut Span, emptied: &mut *mut Span) {
    // SAFETY: the caller's contract.
    unsafe {
        if (*span).out == 0 {
            if (*span).place == Place::Partial {
                unlink(class, span);
            }
            (*span).next = *emptied;
            *emptied = span;
        } else if (*span).place == Place::Out {
            link(class, span);
        }
    }
}

/// Put `span` first on its class's list.
///
/// # Safety
/// The class's lock is held (or `span` is unreachable and the list is the caller's), and `span`
/// is off every list.
unsafe fn link(class: usize, span: *mut Span) {
    // SAFETY: the caller's contract.
    unsafe {
        let first = CLASSES[class].first.get();
        (*span).prev = null_mut();
        (*span).next = *first;
        if !(*first).is_null() {
            (**first).prev = span;
        }
        *first = span;
        (*span).place = Place::Partial;
    }
}

/// Take `span` off its class's list.
///
/// # Safety
/// The class's lock is held, and `span` is on the class's list.
unsafe fn unlink(class: usize, span: *mut Span) {
    // SAFETY: the caller's contract.
    unsafe {
        let (prev, next) = ((*span).prev, (*span).next);
        if prev.is_null() {
            *CLASSES[class].first.get() = next;
        } else {
            (*prev).next = next;
        }
        if !next.is_null() {
            (*next).prev = prev;
        }
        (*span).place = Place::Out;
    }
}

/// Give the thread's region `from..to` of `class`, the tail of its span, back to the span.
///
/// # Safety
/// `from..to` is a region this allocator gave the thread, which hands out none of it any more.
unsafe fn give_back_region(class: usize, from: usize, to: usize, zero: bool) {
    let span = span_of(from);
    let mut emptied = null_mut();
    {
        let _guard = CLASSES[class].lock.lock();
        // SAFETY: the caller's contract; the span holds blocks of the class.
        unsafe {
            debug_assert_eq!(base_of(span) + (*span).end as usize, to);
            debug_assert_eq!((*span).cursor, (*span).end);
            (*span).cursor = (from - base_of(span)) as u32;
            (*span).tail_zero = zero;
            (*span).out -= ((to - from) / class_size(class)) as u32;
            settle(class, span, &mut emptied);
        }
    }
    if !emptied.is_null() {
        // SAFETY: the span emptied, which no thread reaches any more.
        unsafe { pool_put(emptied) };
    }
}

/// Return the thread's lists and regions to their spans (its exit, and [`trim`]).
fn drain(cache: &Cache) {
    for class in 0..NUM_CLASSES {
        let list = cache.heads[class].replace(null_mut());
        cache.counts[class].set(0);
        if !list.is_null() {
            // SAFETY: the thread's list, which it no longer holds.
            unsafe { give_back(class, list) };
        }
        let from = cache.bump[class].replace(0);
        let to = cache.bump_end[class].replace(0);
        if from < to {
            // SAFETY: the thread's region, which it no longer holds.
            unsafe { give_back_region(class, from, to, cache.bump_zero.get() & 1 << class != 0) };
        }
    }
    cache.bump_zero.set(0);
}

/// An empty span: the pool's most recently emptied committed span, else one whose memory went
/// back to the OS (recommitted), else the newest segment's next span, else a new segment's first.
/// Its header, and whether its memory reads as zero; null when the OS refuses memory.
fn take_span() -> (*mut Span, bool) {
    let guard = POOL.lock.lock();
    // SAFETY: the pool's state and its spans' headers, under the pool's lock.
    unsafe {
        let pool = &mut *POOL.state.get();
        let span = pool.committed;
        if !span.is_null() {
            pool.remove(span);
            return (span, false);
        }
        let span = pool.decommitted;
        if !span.is_null() {
            pool.remove(span);
            drop(guard);
            if segments::recommit(base_of(span) as *mut u8, span_size()) {
                return (span, (*span).tail_zero);
            }
            // The system's commit limit: the span waits in the pool.
            let _guard = POOL.lock.lock();
            let pool = &mut *POOL.state.get();
            pool.push_decommitted(span);
            return (null_mut(), false);
        }
        if pool.carve == pool.carve_end {
            let first = segments::map();
            if first.is_null() {
                return (null_mut(), false);
            }
            // A new segment's header reads as zero: no span carved, none pooled, not free.
            let segment = first as *mut Segment;
            (*segment).next = pool.segments;
            if !pool.segments.is_null() {
                (*pool.segments).prev = segment;
            }
            pool.segments = segment;
            pool.carve = first as usize + span_size();
            pool.carve_end = first as usize + SEGMENT;
        }
        let base = pool.carve;
        pool.carve += span_size();
        let span = span_of(base);
        let segment = segment_of(span);
        (*segment).carved += 1;
        if (*segment).free {
            pool.unlink_free(segment);
        }
        (span, true)
    }
}

/// Put the spans emptied, linked by `next`, first in the pool's committed list. Past
/// `POOL_COMMITTED_BYTES` of committed spans, the oldest free segments go back to the OS, and
/// where a span's memory goes back alone, the oldest spans' memory after them.
///
/// # Safety
/// The spans are empty and off every list, and no thread reaches them.
unsafe fn pool_put(mut list: *mut Span) {
    let mut excess = null_mut::<Span>();
    let freed;
    {
        let _guard = POOL.lock.lock();
        // SAFETY: the pool's state and the spans' headers, under the pool's lock.
        unsafe {
            let pool = &mut *POOL.state.get();
            while !list.is_null() {
                let span = list;
                list = (*span).next;
                pool.push_committed(span);
            }
            let bound = POOL_COMMITTED_BYTES >> span_shift();
            freed = if releases() {
                pool.take_free_segments(Some(bound))
            } else {
                null_mut()
            };
            if alone() {
                while pool.committed_len > bound {
                    let span = pool.committed_last;
                    pool.remove(span);
                    (*span).next = excess;
                    excess = span;
                }
            }
        }
    }
    // SAFETY: taken off the pool's lists: no thread reaches them.
    unsafe {
        unmap_segments(freed);
        if !excess.is_null() {
            decommit_spans(excess);
        }
    }
}

/// Return the memory of the spans of `list`, linked by `next`, to the OS where a span's memory
/// goes back alone (a system call each, outside the pool's lock), and put them first in the
/// pool's decommitted list, in order.
///
/// # Safety
/// The spans are empty and off every list, and no thread reaches them.
unsafe fn decommit_spans(list: *mut Span) {
    // SAFETY: the caller's contract.
    unsafe {
        let (mut span, mut last) = (list, null_mut::<Span>());
        while !span.is_null() {
            let discarded = alone() && segments::decommit(base_of(span) as *mut u8, span_size());
            (*span).tail_zero = discarded && segments::DECOMMIT_ZEROES;
            last = span;
            span = (*span).next;
        }
        let _guard = POOL.lock.lock();
        let pool = &mut *POOL.state.get();
        let (mut span, mut prev) = (list, null_mut::<Span>());
        while !span.is_null() {
            (*span).place = Place::Decommitted;
            (*span).prev = prev;
            pool.decommitted_len += 1;
            pool.count_in(span);
            prev = span;
            span = (*span).next;
        }
        (*last).next = pool.decommitted;
        if !pool.decommitted.is_null() {
            (*pool.decommitted).prev = last;
        }
        pool.decommitted = list;
    }
}

/// Return the segments taken off the pool's lists, linked by `next`, to the OS.
///
/// # Safety
/// No thread reaches them.
unsafe fn unmap_segments(mut segment: *mut Segment) {
    while !segment.is_null() {
        // SAFETY: the caller's contract: the header is read before the segment goes.
        unsafe {
            let next = (*segment).next;
            segments::unmap(segment as *mut u8);
            segment = next;
        }
        #[cfg(test)]
        tests::RELEASED.fetch_add(1, Ordering::Relaxed);
    }
}

/// Whether free segments go back to the OS: always.
#[cfg(not(test))]
#[inline]
fn releases() -> bool {
    true
}

/// But in the tests that keep them (`tests::KEEP_SEGMENTS`), which read spans after they empty.
#[cfg(test)]
fn releases() -> bool {
    !tests::KEEP_SEGMENTS.load(Ordering::Relaxed)
}

/// Return the thread's lists and regions to their spans, and the pool's free segments to the OS
/// (and where a span's memory goes back alone, its empty spans' memory). Called only after a
/// major GC: doing this on every collection would throw away the hot allocation cache and turn
/// steady-state churn into refills and page faults.
pub(crate) fn trim() {
    // The library's test binary does not register the allocator: only its own tests use it, and
    // they trim when they mean to (`trim_now`), so the other tests' collections leave it alone.
    if cfg!(test) {
        return;
    }
    trim_now();
}

fn trim_now() {
    let _ = CACHE.try_with(drain);
    let (freed, list) = {
        let _guard = POOL.lock.lock();
        // SAFETY: the pool's state and its spans' headers, under the pool's lock.
        unsafe {
            let pool = &mut *POOL.state.get();
            let freed = if releases() {
                pool.take_free_segments(None)
            } else {
                null_mut()
            };
            let mut list = null_mut::<Span>();
            if alone() {
                while !pool.committed_last.is_null() {
                    let span = pool.committed_last;
                    pool.remove(span);
                    (*span).next = list;
                    list = span;
                }
            }
            (freed, list)
        }
    };
    // SAFETY: taken off the pool's lists: no thread reaches them.
    unsafe {
        unmap_segments(freed);
        if !list.is_null() {
            decommit_spans(list);
        }
    }
    #[cfg(target_os = "macos")]
    unsafe {
        unsafe extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        // A null zone requests pressure relief from every registered malloc zone (including the
        // nano/tiny zones that may own the large blocks returned by `System`).
        let _ = malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
}

/// See the module docs.
pub struct ClassAlloc;

// The fast paths belong in the allocator shims (`__rust_alloc`, `__rust_dealloc`) the compiler
// generates for the registered allocator: out of line they cost every allocation and free a jump
// and the front end a code location of their own (Vue's passes +3% cycles at equal instructions,
// more L1I refills and mispredicted branches, when fat LTO left them out of line).
unsafe impl GlobalAlloc for ClassAlloc {
    #[inline(always)]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if let Some(class) = class_of(layout.size(), layout.align()) {
            let cached = CACHE.with(|c| {
                let p = c.heads[class].get();
                if !p.is_null() {
                    c.heads[class].set(unsafe { *(p as *mut *mut u8) });
                    c.counts[class].set(c.counts[class].get() - 1);
                }
                p
            });
            if !cached.is_null() {
                return cached;
            }
            return alloc_miss(class).0;
        }
        unsafe { System.alloc(layout) }
    }

    #[inline(always)]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(class) = class_of(layout.size(), layout.align()) {
            let cached = CACHE.with(|c| {
                if c.exited.get() {
                    return false;
                }
                if !c.registered.get() && !register(c) {
                    return false;
                }
                if c.counts[class].get() < CLASS_CAPS[class] {
                    unsafe { *(ptr as *mut *mut u8) = c.heads[class].get() };
                    c.heads[class].set(ptr);
                    c.counts[class].set(c.counts[class].get() + 1);
                    true
                } else {
                    false
                }
            });
            if !cached {
                unsafe { dealloc_miss(ptr, class) };
            }
            return;
        }
        unsafe { System.dealloc(ptr, layout) }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Blocks too large for a class (ArrayBuffer data) come from the system already zeroed:
        // calloc maps fresh pages, which cost nothing until written. A class block from a
        // thread's list is reused dirty, so it is cleared here; one from a region that reads as
        // zero (a span fresh from the OS or back from a decommit that zeroes) is not.
        let Some(class) = class_of(layout.size(), layout.align()) else {
            return unsafe { System.alloc_zeroed(layout) };
        };
        let cached = CACHE.with(|c| {
            let p = c.heads[class].get();
            if !p.is_null() {
                c.heads[class].set(unsafe { *(p as *mut *mut u8) });
                c.counts[class].set(c.counts[class].get() - 1);
            }
            p
        });
        let (ptr, zero) = if cached.is_null() {
            alloc_miss(class)
        } else {
            (cached, false)
        };
        if !ptr.is_null() && !zero {
            unsafe { std::ptr::write_bytes(ptr, 0, layout.size()) };
        }
        ptr
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Within one size class a grow/shrink is free; otherwise allocate-copy-free through
        // the same class discipline.
        if let (Some(a), Some(b)) = (
            class_of(layout.size(), layout.align()),
            class_of(new_size, layout.align()),
        ) {
            if a == b {
                return ptr;
            }
        }
        let new_layout = unsafe { Layout::from_size_align_unchecked(new_size, layout.align()) };
        let new_ptr = unsafe { self.alloc(new_layout) };
        if !new_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(ptr, new_ptr, layout.size().min(new_size));
                self.dealloc(ptr, layout);
            }
        }
        new_ptr
    }
}

#[cfg(test)]
#[path = "fastalloc_tests.rs"]
mod tests;
