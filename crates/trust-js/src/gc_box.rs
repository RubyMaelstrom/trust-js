//! P1 S2: handles to immutable payload cells (feature `gc-heap`), such as a BigInt's digits.
//!
//! A handle is its cell's address: `Copy`, with no reference count. The cell lives in the
//! Agent heap that was active when the handle was made, until a collection finds it
//! unreachable; its descriptor's finalizer then drops the payload. Payload cells have no
//! outgoing edges.

use std::ptr::NonNull;

/// A payload type kept in cells of its own kind.
pub(crate) trait Payload: Sized + 'static {
    /// The descriptor of this payload's cells in `heap`.
    fn desc(heap: &crate::value::GcState) -> crate::gc_core::DescId;
}

/// A handle to one payload cell.
#[repr(transparent)]
pub(crate) struct GcBox<T: Payload>(NonNull<T>);

impl<T: Payload> GcBox<T> {
    /// Allocate `value` in the active Agent's heap.
    #[inline]
    pub(crate) fn new(value: T) -> Self {
        let cell = crate::value::alloc_payload_cell(T::desc, std::mem::size_of::<T>()).cast::<T>();
        // SAFETY: a fresh cell sized and aligned for `T`. Allocation never collects, so the
        // cell is initialized before any collection can trace or finalize it.
        unsafe {
            cell.write(value);
            GcBox(NonNull::new_unchecked(cell))
        }
    }

    /// The cell's address: the handle's identity and its GC edge.
    #[inline]
    pub(crate) fn as_ptr(this: &Self) -> *const T {
        this.0.as_ptr()
    }
}

impl<T: Payload> Clone for GcBox<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: Payload> Copy for GcBox<T> {}

impl<T: Payload> std::ops::Deref for GcBox<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: a handle reachable by the program keeps its cell alive (a GC root or a
        // traced edge), and payload cells are never mutated after initialization.
        unsafe { self.0.as_ref() }
    }
}

impl<T: Payload + std::fmt::Debug> std::fmt::Debug for GcBox<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (**self).fmt(f)
    }
}

/// Drop the payload of a dead `T` cell: its descriptor's finalizer.
pub(crate) fn finalize<T: Payload>(cell: *mut u8) {
    // SAFETY: cells of `T`'s descriptor are initialized by `GcBox::new` before any
    // collection or teardown, and finalized once.
    unsafe { std::ptr::drop_in_place(cell.cast::<T>()) };
}

/// The trace of a payload cell: it has no outgoing edges.
pub(crate) fn trace_none(_: *const u8, _: &mut dyn FnMut(*const u8)) {}
