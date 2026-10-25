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
pub(crate) fn trace_block<S: GcSink + ?Sized>(ptr: *const u8, sink: &mut S) {
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
    pub(crate) fn trace<S: GcSink + ?Sized>(this: &Self, sink: &mut S) {
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

/// A kind of value an engine cell keeps beside itself in a payload cell ([`PayloadBox`]).
pub(crate) trait PayloadKind: Sized + 'static {
    /// The descriptor of this kind's payload cells in `heap`.
    fn desc(heap: &crate::value::GcState) -> crate::gc_core::DescId;
}

/// The descriptor of `T`'s payload cells: no edges of their own (the owner traces what the
/// value holds, as it reports the cell), uncarded (the value changes only through its owner,
/// whose card the change sets), and a finalizer that drops the value when the cell dies, if
/// `T` has anything to drop.
pub(crate) fn payload_layout<T: PayloadKind>(name: &'static str) -> crate::gc_core::LayoutDesc {
    crate::gc_core::LayoutDesc {
        name,
        trace: crate::gc_box::trace_none,
        mark_trace: None,
        finalize: std::mem::needs_drop::<T>().then_some(finalize_payload::<T> as _),
        carded: false,
    }
}

/// Drop the value of a dead payload cell of `T`: its descriptor's finalizer.
fn finalize_payload<T: PayloadKind>(cell: *mut u8) {
    // SAFETY: payload cells of `T` hold a value written by `PayloadBox::new` before any
    // collection, which only this finalizer drops, once.
    unsafe { std::ptr::drop_in_place(cell.cast::<T>()) };
}

/// One value of type `T` in a payload cell of its own: a `Box<T>` whose value the collector
/// drops. An object keeps its Rust-heap payloads (a Map's tables, a bound function's arguments,
/// an error's stack trace...) in these, so the object itself holds nothing to drop and dies
/// without a finalizer (`value::Object`); the payload cell's finalizer drops the value when the
/// cell dies, with its owner or once the owner let it go. The owner reports the cell with
/// [`PayloadBox::trace`] and traces the value's edges itself, in every trace; it changes the
/// value only through itself, so the owner's card records a change as for any of its fields.
/// Dropping the box drops nothing: the value is the cell's. A clone is a new cell.
#[repr(transparent)]
pub(crate) struct PayloadBox<T: PayloadKind>(NonNull<T>, std::marker::PhantomData<T>);

impl<T: PayloadKind> PayloadBox<T> {
    pub(crate) fn new(value: T) -> Self {
        const { assert!(align_of::<T>() <= BLOCK_ALIGN) };
        let cell = crate::value::alloc_payload_cell(T::desc, size_of::<T>().max(1)).cast::<T>();
        // SAFETY: a fresh cell sized and aligned for `T`. Allocation never collects, so the
        // value is written before any collection can finalize the cell.
        unsafe {
            cell.write(value);
            Self(NonNull::new_unchecked(cell), std::marker::PhantomData)
        }
    }

    /// Report the cell to `sink`.
    #[inline]
    pub(crate) fn trace<S: GcSink + ?Sized>(this: &Self, sink: &mut S) {
        trace_block(this.0.as_ptr().cast(), sink);
    }
}

impl<T: PayloadKind> Deref for PayloadBox<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: the box names a live payload cell its owner reports.
        unsafe { self.0.as_ref() }
    }
}

impl<T: PayloadKind> DerefMut for PayloadBox<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as in `deref`; the box is the cell's one owner, borrowed mutably.
        unsafe { self.0.as_mut() }
    }
}

impl<T: PayloadKind + Default> Default for PayloadBox<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: PayloadKind> AsRef<T> for PayloadBox<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

/// The cell, and the value's own edges.
impl<T: PayloadKind + crate::gc_trace::GcTrace> crate::gc_trace::GcTrace for PayloadBox<T> {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        Self::trace(self, sink);
        (**self).trace_gc(sink);
    }
}

impl<T: PayloadKind + Clone> Clone for PayloadBox<T> {
    fn clone(&self) -> Self {
        Self::new((**self).clone())
    }
}

impl<T: PayloadKind + std::fmt::Debug> std::fmt::Debug for PayloadBox<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

/// An immutable value of type `T` in a payload cell that copies share: a function object's
/// [[Call]] payload (a native function's data, a bound function's target and arguments...),
/// which every call copies out of the object before it runs. A copy names the same cell, as a
/// [`crate::value::Gc`] copy names the same object, and follows the same rule: the cell lives
/// while a traced owner reports it ([`SharedPayload::trace`], with the value's edges) or a
/// native frame holds a copy; one kept anywhere else across a safepoint must be traced there.
/// The cell's finalizer drops the value, once, when the last of them lets it go. A frame that
/// keeps a pointer into the value itself across a call (such as an `Rc` field's target) holds
/// that value's own handle, not only the copy.
#[repr(transparent)]
pub(crate) struct SharedPayload<T: PayloadKind>(NonNull<T>, std::marker::PhantomData<T>);

impl<T: PayloadKind> SharedPayload<T> {
    pub(crate) fn new(value: T) -> Self {
        const { assert!(align_of::<T>() <= BLOCK_ALIGN) };
        let cell = crate::value::alloc_payload_cell(T::desc, size_of::<T>().max(1)).cast::<T>();
        // SAFETY: as in `PayloadBox::new`.
        unsafe {
            cell.write(value);
            Self(NonNull::new_unchecked(cell), std::marker::PhantomData)
        }
    }

    /// Report the cell to `sink`.
    #[inline]
    pub(crate) fn trace<S: GcSink + ?Sized>(this: &Self, sink: &mut S) {
        trace_block(this.0.as_ptr().cast(), sink);
    }
}

impl<T: PayloadKind> Clone for SharedPayload<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: PayloadKind> Copy for SharedPayload<T> {}

impl<T: PayloadKind> Deref for SharedPayload<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: the copy names a live payload cell (see the type's comment), whose value no
        // one changes.
        unsafe { self.0.as_ref() }
    }
}

impl<T: PayloadKind> AsRef<T> for SharedPayload<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

/// The cell, and the value's own edges.
impl<T: PayloadKind + crate::gc_trace::GcTrace> crate::gc_trace::GcTrace for SharedPayload<T> {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        Self::trace(self, sink);
        (**self).trace_gc(sink);
    }
}

impl<T: PayloadKind + std::fmt::Debug> std::fmt::Debug for SharedPayload<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}
