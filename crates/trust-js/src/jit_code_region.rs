//! Executable memory from one reserved region (P3 stage 2 step 9,
//! docs/v8-chase-p3-plan-2026-11-08.txt).
//!
//! On Linux and Windows the process's code goes in one region of addresses reserved at the first
//! allocation, as V8's code range, SpiderMonkey's process executable memory and JSC's fixed pool
//! hold theirs: a bitmap of the region's pages and a cursor, under a lock (the process's threads
//! all compile). An allocation takes the first run of free pages at or after the cursor, wrapping
//! at the region's end, commits it writable, copies the code, flushes and protects it executable,
//! and moves the cursor past it; a free decommits the pages, whose addresses stay reserved, and
//! clears their bits. Each code object has pages of its own, so its W^X protection stays its own.
//!
//! The cursor never moves back. SpiderMonkey's moves back to a freed page, to reuse pages before
//! it fragments the region (ProcessExecutableMemory::deallocate, jit/ProcessExecutableMemory.cpp:
//! 881-885), but its frees come from the GC's finalization, well after an invalidation. Here a
//! discard frees the tier's code before its recompile, and on the Cortex-X925 new code at the
//! addresses of code just freed mispredicts its branches at every execution: so a freed page is
//! taken again only once the cursor has passed the rest of the region, which stands in for the
//! exemplars' wait for the GC.
//!
//! Where the region cannot be reserved (an address-space limit) or has no free run long enough,
//! an allocation maps its own memory, as every allocation did before the region
//! (`sys::alloc_exec`). The budget's accounting is the same either way.

use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use super::sys;

/// The process region's size for a code budget of `budget` bytes: 1 GiB of addresses, or twice a
/// budget above 512 MiB (TRUST_JS_JIT_CODE_BUDGET_MB), so that live code fills at most half.
fn region_bytes(budget: usize) -> usize {
    (1usize << 30).max(budget.saturating_mul(2))
}

/// The process's code region, reserved at the first call (the first executable allocation);
/// `None` where the system refused the addresses.
pub(crate) fn process_region() -> Option<&'static CodeRegion> {
    static REGION: OnceLock<Option<CodeRegion>> = OnceLock::new();
    REGION
        .get_or_init(|| CodeRegion::reserve(region_bytes(super::executable_code_budget().limit)))
        .as_ref()
}

/// Reserved addresses whose pages hold code: whole pages for each code object.
pub(crate) struct CodeRegion {
    /// The first page's address.
    base: usize,
    /// The system's page size, a power of two.
    page_size: usize,
    /// The region's size in pages.
    page_count: usize,
    map: Mutex<PageMap>,
}

/// Which of a region's pages are taken, and where the next allocation looks first.
struct PageMap {
    /// Bit `p % 64` of word `p / 64` is set while code holds page `p`, or for good where the
    /// system left the page's state unknown.
    taken: Box<[u64]>,
    /// The page the next allocation's search starts at.
    cursor: usize,
}

impl CodeRegion {
    /// `bytes` of addresses, rounded up to whole pages, reserved with no access; `None` when the
    /// system refuses them.
    pub(crate) fn reserve(bytes: usize) -> Option<CodeRegion> {
        let page_size = sys::page_size();
        let page_count = bytes.div_ceil(page_size);
        if page_count == 0 {
            return None;
        }
        // SAFETY: a new mapping, where the system chooses.
        let base = unsafe { sys::reserve(page_count.checked_mul(page_size)?) };
        if base.is_null() {
            return None;
        }
        Some(CodeRegion {
            base: base as usize,
            page_size,
            page_count,
            map: Mutex::new(PageMap {
                taken: vec![0; page_count.div_ceil(64)].into_boxed_slice(),
                cursor: 0,
            }),
        })
    }

    /// `code` in pages of the region, published executable: its address, or `None` when the
    /// region has no free run of pages long enough or the system refuses the memory.
    pub(crate) fn allocate(&self, code: &[u8]) -> Option<*mut u8> {
        if code.is_empty() {
            return None;
        }
        let pages = code.len().div_ceil(self.page_size);
        let first = self.map().take(pages, self.page_count)?;
        let mem = (self.base + first * self.page_size) as *mut u8;
        let len = pages * self.page_size;
        // SAFETY: the pages are the region's, reserved, and taken now: no other allocation
        // commits them, and nothing runs them before this returns.
        unsafe {
            if !sys::commit(mem, len) {
                // The range's state is unknown, so its pages stay taken: no commit of a later
                // allocation can land on a mapping that took their place.
                return None;
            }
            if sys::publish(mem, code.as_ptr(), code.len()) {
                return Some(mem);
            }
            if sys::decommit(mem, len) {
                self.map().release(first, pages);
            }
        }
        None
    }

    /// Give the pages of the `len` bytes of code at `mem`, from [`Self::allocate`], back to the
    /// system: their memory freed and their addresses reserved again, free for the region's
    /// allocations once the cursor comes back to them.
    ///
    /// # Safety
    /// Nothing runs or reads the code any more.
    pub(crate) unsafe fn free(&self, mem: *mut u8, len: usize) {
        let offset = mem as usize - self.base;
        debug_assert!(
            offset.is_multiple_of(self.page_size) && offset / self.page_size < self.page_count
        );
        let pages = len.div_ceil(self.page_size);
        // Decommitted before their bits clear, so that no allocation commits them first; pages
        // the system does not decommit stay taken.
        // SAFETY: the pages hold only this code (the caller's contract).
        if unsafe { sys::decommit(mem, pages * self.page_size) } {
            self.map().release(offset / self.page_size, pages);
        }
    }

    fn map(&self) -> MutexGuard<'_, PageMap> {
        self.map.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl PageMap {
    /// Take the first run of `n` free pages, `n` positive, at or after the cursor, else (wrapping)
    /// from the region's start, and move the cursor past it. A run never crosses the region's
    /// end, which is `page_count` pages.
    fn take(&mut self, n: usize, page_count: usize) -> Option<usize> {
        let first = free_run(&self.taken, self.cursor, page_count, n)
            .or_else(|| free_run(&self.taken, 0, (self.cursor + n - 1).min(page_count), n))?;
        set_bits(&mut self.taken, first..first + n, true);
        self.cursor = if first + n == page_count {
            0
        } else {
            first + n
        };
        Some(first)
    }

    /// Free the `n` pages from `first`. The cursor stays where it is.
    fn release(&mut self, first: usize, n: usize) {
        set_bits(&mut self.taken, first..first + n, false);
    }
}

/// The first run of `n` free pages that lies within pages `from..to`.
fn free_run(taken: &[u64], from: usize, to: usize, n: usize) -> Option<usize> {
    let mut start = from;
    while start + n <= to {
        let free = next_page(taken, start, to, false)?;
        if free + n > to {
            return None;
        }
        match next_page(taken, free, free + n, true) {
            None => return Some(free),
            Some(held) => start = held + 1,
        }
    }
    None
}

/// The first page in `from..to` that is taken (`taken`) or free (not).
fn next_page(map: &[u64], from: usize, to: usize, taken: bool) -> Option<usize> {
    let mut page = from;
    while page < to {
        let word = map[page / 64];
        let bits = (if taken { word } else { !word }) >> (page % 64);
        if bits != 0 {
            let found = page + bits.trailing_zeros() as usize;
            return (found < to).then_some(found);
        }
        page = (page / 64 + 1) * 64;
    }
    None
}

fn set_bits(map: &mut [u64], pages: std::ops::Range<usize>, taken: bool) {
    for page in pages {
        let (word, bit) = (page / 64, 1u64 << (page % 64));
        debug_assert_eq!(
            map[word] & bit != 0,
            !taken,
            "code region page {page} taken or freed twice"
        );
        if taken {
            map[word] |= bit;
        } else {
            map[word] &= !bit;
        }
    }
}

#[cfg(test)]
#[path = "jit_code_region_tests.rs"]
mod tests;
