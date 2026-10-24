//! P1 S6: storage blocks, the variable-size payloads of the engine's cells kept in the Agent
//! heap's own pages beside the cells (JavaScriptCore's auxiliary space): an object's spilled
//! named properties, its element sidecar, packed elements, numeric mirror and dense slot map.
//!
//! A block is a leaf cell of the heap's storage kind ([`layout`]): it has no edges and no
//! finalizer of its own, and nothing zeroes it. Its owner writes every byte it reads, traces
//! the values the block holds as part of its own edges, and reports the block's address as an
//! edge too ([`trace_block`]), which is what keeps the block alive: a block lives exactly as long
//! as some traced owner (an object, a template's property map) or a native frame names it, and
//! the sweep that finds it unmarked frees it. Nothing frees a block explicitly, so an owner that
//! replaces its block (growth) just drops the old address.
//!
//! What the move buys: the blocks count toward the collection triggers as the cells do, an
//! object whose payloads are all blocks needs no finalizer (its death is a bit in a page), and
//! allocation is the heap's, with no Rust-heap traffic. The rules it imposes:
//!
//! - Every owner traces its blocks, in every trace (marking, verification, the write barrier's
//!   check). An owner held in a Rust-heap container across a safepoint must be traced there.
//! - An old owner that takes a new block records the write (its card, through the handle's
//!   mutable borrow), as for any new edge: a minor collection finds the block through it.
//! - What a block holds is traced by the owner only: values, and the blocks a block names (the
//!   element sidecar's buffers). A block that only a native frame names keeps its memory, not
//!   the blocks or the targets it names, so a detached property map must not cross a safepoint
//!   unless something traces it, as any Rust-heap value must not.

use crate::gc_trace::GcSink;
use std::mem::{align_of, size_of};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// The descriptor of storage blocks: no edges (the owner traces what a block holds), no
/// finalizer (a block's contents own nothing outside the heap), no cards.
pub(crate) fn layout() -> crate::gc_core::LayoutDesc {
    crate::gc_core::LayoutDesc {
        name: "storage",
        trace: crate::gc_box::trace_none,
        mark_trace: None,
        finalize: None,
        carded: false,
    }
}

/// The largest alignment a block guarantees: every cell is 16-byte aligned.
pub(crate) const BLOCK_ALIGN: usize = 16;

/// A block of at least `bytes` (non-zero) bytes in the active Agent heap, 16-byte aligned and
/// uninitialized; `None` when the memory of a large block cannot be had.
#[inline]
pub(crate) fn try_alloc_block(bytes: usize) -> Option<NonNull<u8>> {
    debug_assert!(bytes != 0, "empty storage block");
    crate::value::alloc_storage_block(bytes).and_then(NonNull::new)
}

/// [`try_alloc_block`], aborting when the memory cannot be had.
#[inline]
pub(crate) fn alloc_block(bytes: usize) -> NonNull<u8> {
    try_alloc_block(bytes).unwrap_or_else(|| {
        std::alloc::handle_alloc_error(
            std::alloc::Layout::from_size_align(bytes, BLOCK_ALIGN)
                .unwrap_or(std::alloc::Layout::new::<u8>()),
        )
    })
}

/// Report the block at `ptr` as an edge of its owner (see the module comment).
#[inline]
pub(crate) fn trace_block(ptr: *const u8, sink: &mut dyn GcSink) {
    sink.word(ptr as u64);
}

/// One value of type `T` in a storage block: `Box<T>` in the heap's pages, for plain data (no
/// drop glue, which `new` checks), so dropping the box reads nothing: the block may already
/// have gone with its heap, and the collector frees it. The owner traces it with
/// [`StorageBox::trace`]. A nullable `Option<StorageBox<T>>` is one pointer word, as
/// `Option<Box<T>>` is.
#[repr(transparent)]
pub(crate) struct StorageBox<T>(NonNull<T>);

impl<T> StorageBox<T> {
    pub(crate) fn new(value: T) -> Self {
        const { assert!(align_of::<T>() <= BLOCK_ALIGN) };
        const {
            assert!(
                !std::mem::needs_drop::<T>(),
                "storage boxes hold plain data"
            )
        };
        let ptr = alloc_block(size_of::<T>().max(1)).cast::<T>();
        // SAFETY: the block is fresh, large enough and aligned for a `T`.
        unsafe { ptr.as_ptr().write(value) };
        Self(ptr)
    }

    /// Report the block to `sink` (see [`trace_block`]).
    #[inline]
    pub(crate) fn trace(this: &Self, sink: &mut dyn GcSink) {
        trace_block(this.0.as_ptr().cast(), sink);
    }
}

impl<T> Deref for StorageBox<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: the box owns an initialized `T` in a block its owner keeps alive.
        unsafe { self.0.as_ref() }
    }
}

impl<T> DerefMut for StorageBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as in `deref`; `&mut self` makes the access unique.
        unsafe { self.0.as_mut() }
    }
}

impl<T: Clone> Clone for StorageBox<T> {
    fn clone(&self) -> Self {
        Self::new((**self).clone())
    }
}

impl<T: Default> Default for StorageBox<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

/// [`crate::inline_vec::InlineVec`] spills into storage blocks: for vectors that live in a cell
/// (or in a traced template) and whose owner traces [`crate::inline_vec::InlineVec::spill_ptr`].
pub(crate) struct GcSpill;

impl crate::inline_vec::Spill for GcSpill {
    #[inline]
    fn alloc(layout: std::alloc::Layout) -> *mut u8 {
        debug_assert!(layout.align() <= BLOCK_ALIGN);
        alloc_block(layout.size()).as_ptr()
    }

    #[inline]
    unsafe fn free(_: *mut u8, _: std::alloc::Layout) {
        // The collector frees the block once no owner reports it.
    }
}
