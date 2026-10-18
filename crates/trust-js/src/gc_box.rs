//! P1 S2: handles to payload cells (feature `gc-heap`): a BigInt's digits, a Symbol's identity,
//! (P1 S4) a closure's `UserCallable` and (P1 S5) a function's deferred `prototype`.
//!
//! A handle is its cell's address: `Copy`, with no reference count. The cell lives in the
//! Agent heap that was active when the handle was made, until a collection finds it
//! unreachable; its descriptor's finalizer then drops the payload. Payload cells have no
//! outgoing edges, except a closure cell's to its environment (see `value::UserFn`) and a
//! deferred prototype's to its function and to the prototype's parent or, once materialized,
//! the prototype (see `value::LazyProto`).

use std::ptr::NonNull;

/// A payload type kept in cells of its own kind.
pub(crate) trait Payload: Sized + 'static {
    /// The descriptor of this payload's cells in `heap`.
    fn desc(heap: &crate::value::GcState) -> crate::gc_core::DescId;
}

/// A handle to one payload cell. Public because `Value::Sym` holds one; only the engine makes
/// them.
#[repr(transparent)]
pub struct GcBox<T>(NonNull<T>);

/// Allocate `value` in the active Agent's heap.
#[inline]
pub(crate) fn new<T: Payload>(value: T) -> GcBox<T> {
    let cell = crate::value::alloc_payload_cell(T::desc, std::mem::size_of::<T>()).cast::<T>();
    // SAFETY: a fresh cell sized and aligned for `T`. Allocation never collects, so the cell
    // is initialized before any collection can trace or finalize it.
    unsafe {
        cell.write(value);
        GcBox(NonNull::new_unchecked(cell))
    }
}

impl<T> GcBox<T> {
    /// The cell's address: the handle's identity and its GC edge.
    #[inline]
    pub(crate) fn as_ptr(this: &Self) -> *const T {
        this.0.as_ptr()
    }

    /// A handle for the cell at `ptr` ([`GcBox::as_ptr`] output).
    ///
    /// # Safety
    /// `ptr` must address a live cell of `T`'s kind.
    #[inline]
    pub(crate) unsafe fn from_raw(ptr: *const T) -> Self {
        GcBox(unsafe { NonNull::new_unchecked(ptr.cast_mut()) })
    }

    #[inline]
    pub(crate) fn ptr_eq(a: &Self, b: &Self) -> bool {
        a.0 == b.0
    }
}

impl<T> Clone for GcBox<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for GcBox<T> {}

impl<T> std::ops::Deref for GcBox<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: a handle reachable by the program keeps its cell alive (a GC root or a
        // traced edge); a payload changes only through its own interior mutability.
        unsafe { self.0.as_ref() }
    }
}

impl<T> AsRef<T> for GcBox<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for GcBox<T> {
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
