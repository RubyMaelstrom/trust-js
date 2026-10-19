//! The engine's environment-record handle.
//!
//! [`Env`] is the `Copy` address of a GC cell holding the `RefCell<Scope>` (P1 S2), traced by
//! the scope descriptor (see `value::trace_scope_cell`). It dereferences to the
//! `RefCell<Scope>`, so `.borrow()`/`.borrow_mut()` work on a handle. Identity goes through
//! associated functions — [`Env::as_ptr`], [`Env::ptr_eq`], [`Env::downgrade`]. Rust frames
//! hold environments in locals everywhere, where the conservative stack scan finds them.

#[cfg(feature = "gc-heap")]
pub use cell::{Env, WeakEnv};

#[cfg(feature = "gc-heap")]
mod cell {
    use crate::gc_handle::WeakRef;
    use crate::interpreter::Scope;
    use std::cell::RefCell;
    use std::ptr::NonNull;

    /// A handle to an environment record: the address of its cell.
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct Env(NonNull<RefCell<Scope>>);

    impl Env {
        /// Allocate `scope` in the active Agent's heap.
        #[inline]
        pub(crate) fn new(scope: Scope) -> Env {
            let cell = crate::value::alloc_scope_cell();
            // SAFETY: a fresh cell sized and aligned for `RefCell<Scope>`. Allocation never
            // collects, so the cell is initialized before any collection can trace it.
            unsafe {
                std::ptr::write(cell, RefCell::new(scope));
                Env(NonNull::new_unchecked(cell))
            }
        }

        #[inline]
        pub fn as_ptr(this: &Env) -> *const RefCell<Scope> {
            this.0.as_ptr()
        }

        #[inline]
        pub fn ptr_eq(a: &Env, b: &Env) -> bool {
            a.0 == b.0
        }

        #[inline]
        pub fn downgrade(this: &Env) -> WeakEnv {
            WeakEnv(WeakRef::to(this.0.as_ptr() as usize))
        }

        /// # Safety
        /// `ptr` must address a live environment cell.
        #[inline]
        pub unsafe fn from_raw(ptr: *const RefCell<Scope>) -> Env {
            Env(unsafe { NonNull::new_unchecked(ptr.cast_mut()) })
        }

        #[inline]
        pub fn into_raw(this: Env) -> *const RefCell<Scope> {
            this.0.as_ptr()
        }
    }

    impl std::ops::Deref for Env {
        type Target = RefCell<Scope>;
        #[inline]
        fn deref(&self) -> &RefCell<Scope> {
            // SAFETY: a reachable handle names a live cell (Decision 0 / I1).
            unsafe { self.0.as_ref() }
        }
    }

    /// A weak environment handle; cleared when a collection finds the record dead.
    #[derive(Clone, Default)]
    pub struct WeakEnv(WeakRef);

    impl WeakEnv {
        #[inline]
        pub fn new() -> WeakEnv {
            WeakEnv(WeakRef::default())
        }

        #[inline]
        pub fn upgrade(&self) -> Option<Env> {
            // SAFETY: the alive flag is cleared before the record's cell is freed.
            self.0
                .live_ptr()
                .map(|ptr| unsafe { Env::from_raw(ptr as *const RefCell<Scope>) })
        }

        #[inline]
        pub fn as_ptr(&self) -> *const RefCell<Scope> {
            self.0.ptr() as *const RefCell<Scope>
        }

        #[inline]
        pub fn ptr_eq(&self, other: &WeakEnv) -> bool {
            self.as_ptr() == other.as_ptr()
        }

        #[inline]
        pub fn strong_count(&self) -> usize {
            usize::from(self.0.live_ptr().is_some())
        }
    }
}
