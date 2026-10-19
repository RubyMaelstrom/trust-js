//! A thread-local size-class caching allocator (std-only).
//!
//! The engine's workloads are allocation-bound in exactly the way general-purpose system
//! allocators are slowest: millions of short-lived, same-sized blocks (spilled `Props` and
//! binding storage, property keys, argument lists, boxed internal state; objects, environments
//! and strings are GC cells in pages of their own). On macOS in particular, `malloc`/`free`
//! pairs dominate parser-shaped profiles.
//!
//! This allocator sits in front of [`std::alloc::System`]: small blocks (≤ [`MAX_CLASS`] bytes,
//! alignment ≤ 16) are rounded up to a 16-byte size class and served from a per-thread
//! INTRUSIVE free list — a freed block's first word points at the next free block, so the
//! allocator itself never allocates (re-entrancy is what a `Vec`-backed cache dies on).
//! Every cacheable request is allocated from the system with its CLASS layout, never the
//! caller's exact layout, so a block can migrate between call sites of the same class and the
//! system layout contract still holds. Large or over-aligned requests pass straight through.
//!
//! Threads (coroutine parking) are handled by construction: each thread caches its own frees,
//! and a block freed on a different thread than it was allocated on simply joins that thread's
//! cache — the backing system allocation is thread-agnostic. Thread teardown drains the lists
//! back to the system (`ExitDrain`); frees after that fall through to the system.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

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

/// The per-thread free lists. The type has no destructor, so the const-initialized
/// thread-local is a plain TLS access on every allocation (a destructor would add a lazy
/// registration-state check to each one). Teardown is delegated to [`ExitDrain`].
struct Cache {
    heads: [Cell<*mut u8>; NUM_CLASSES],
    counts: [Cell<usize>; NUM_CLASSES],
    /// `ExitDrain` has been registered for this thread.
    registered: Cell<bool>,
    /// This thread's `ExitDrain` has run: frees go straight to the system from now on.
    exited: Cell<bool>,
}

thread_local! {
    static CACHE: Cache = const { Cache {
        heads: [const { Cell::new(std::ptr::null_mut()) }; NUM_CLASSES],
        counts: [const { Cell::new(0) }; NUM_CLASSES],
        registered: Cell::new(false),
        exited: Cell::new(false),
    } };
    static EXIT_DRAIN: ExitDrain = const { ExitDrain };
}

/// Returns a thread's cached blocks to the system when the thread exits. Registered lazily,
/// when the thread first caches a freed block.
struct ExitDrain;

impl Drop for ExitDrain {
    fn drop(&mut self) {
        let _ = CACHE.try_with(|cache| {
            cache.exited.set(true);
            drain(cache);
        });
    }
}

#[inline]
fn class_of(size: usize, align: usize) -> Option<usize> {
    if align <= STEP && size <= MAX_CLASS && size > 0 {
        Some(size.div_ceil(STEP) - 1)
    } else {
        None
    }
}

#[inline]
fn class_layout(class: usize) -> Layout {
    // Size is a non-zero multiple of STEP with STEP alignment: always valid.
    unsafe { Layout::from_size_align_unchecked((class + 1) * STEP, STEP) }
}

fn drain(cache: &Cache) {
    for (k, head) in cache.heads.iter().enumerate() {
        let layout = class_layout(k);
        let mut p = head.replace(std::ptr::null_mut());
        cache.counts[k].set(0);
        while !p.is_null() {
            let next = unsafe { *(p as *mut *mut u8) };
            unsafe { System.dealloc(p, layout) };
            p = next;
        }
    }
}

/// Return cached small blocks to the system and ask the platform allocator to release unused
/// pages. Called only after a major GC: doing this on every collection would throw away the hot
/// allocation cache and turn steady-state churn back into malloc/free traffic.
pub(crate) fn trim() {
    let _ = CACHE.try_with(drain);
    #[cfg(target_os = "macos")]
    unsafe {
        unsafe extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        // A null zone requests pressure relief from every registered malloc zone (including the
        // nano/tiny zones that may own allocations returned by `System`).
        let _ = malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
    }
}

/// See the module docs.
pub struct ClassAlloc;

unsafe impl GlobalAlloc for ClassAlloc {
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
            return System.alloc(class_layout(class));
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if let Some(class) = class_of(layout.size(), layout.align()) {
            let cached = CACHE.with(|c| {
                if c.exited.get() {
                    return false;
                }
                if !c.registered.get() {
                    c.registered.set(true);
                    // Touching the key registers its destructor; during teardown it may
                    // already be gone, in which case the blocks simply stay with the thread.
                    let _ = EXIT_DRAIN.try_with(|_| ());
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
                System.dealloc(ptr, class_layout(class));
            }
            return;
        }
        System.dealloc(ptr, layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // Blocks too large for a class (ArrayBuffer data) come from the system already zeroed:
        // calloc maps fresh pages, which cost nothing until written. Cached class blocks are
        // reused dirty, so they are cleared here.
        if class_of(layout.size(), layout.align()).is_none() {
            return unsafe { System.alloc_zeroed(layout) };
        }
        let ptr = unsafe { self.alloc(layout) };
        if !ptr.is_null() {
            unsafe { std::ptr::write_bytes(ptr, 0, layout.size()) };
        }
        ptr
    }

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
