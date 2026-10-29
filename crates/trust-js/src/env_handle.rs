//! The engine's environment-record handle.
//!
//! [`Env`] is the `Copy` address of a GC cell holding the `RefCell<Scope>` (P1 S2), traced by
//! the scope descriptor. Its inherent borrow API keeps the RefCell private; both mutable borrows
//! record the environment's card first. Identity uses the associated `Env::as_ptr` (the cell
//! address), while `scope_ptr` explicitly names its payload. Conservative stack scanning finds
//! the environment handles in Rust frames.

pub use cell::{Env, WeakEnv};

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
        /// Allocate `scope`, a record's header, in the active Agent's heap, in a cell of `size`
        /// bytes: its in-place binding slots follow the header, zeroed (see
        /// [`crate::interpreter::VarMap`], whose size class the header records).
        #[inline]
        pub(crate) fn new(scope: Scope, size: usize) -> Env {
            let cell = crate::value::alloc_scope_cell(size);
            // SAFETY: a fresh zeroed cell of at least `RefCell<Scope>`'s size, aligned for it.
            // Allocation never collects, so the header is written before any collection can
            // trace it, and zeroed slots trace as numbers.
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

        /// Borrow the live cell's payload. The RefCell itself never leaves this handle API.
        #[inline]
        pub fn borrow(&self) -> std::cell::Ref<'_, Scope> {
            // SAFETY: a reachable handle names a live cell (Decision 0 / I1).
            unsafe { self.0.as_ref() }.borrow()
        }

        // Kept as part of the explicit handle API after removing RefCell dereferencing.
        #[allow(dead_code)]
        #[inline]
        pub fn try_borrow(&self) -> Result<std::cell::Ref<'_, Scope>, std::cell::BorrowError> {
            // SAFETY: as in `borrow`.
            unsafe { self.0.as_ref() }.try_borrow()
        }

        /// The payload's address, distinct from `Env::as_ptr`'s RefCell address. Reading or
        /// writing through it requires the usual raw-pointer safety proof; writes must also
        /// record the owning cell's card before publishing any new edge.
        // Kept as part of the explicit handle API after removing RefCell dereferencing.
        #[allow(dead_code)]
        #[inline]
        pub(crate) fn scope_ptr(&self) -> *mut Scope {
            // SAFETY: a reachable handle names a live cell (Decision 0 / I1).
            unsafe { self.0.as_ref() }.as_ptr()
        }

        /// Mutably borrow the record, as `RefCell::borrow_mut`, first recording the write in the
        /// record's card (the write barrier, `gc_core::card_mark`; see `Gc::borrow_mut`).
        #[inline]
        pub fn borrow_mut(&self) -> std::cell::RefMut<'_, Scope> {
            // SAFETY: a reachable handle names a live environment cell (Decision 0 / I1): small
            // (it fills the 128-byte class) and of the carded environment kind.
            unsafe { crate::gc_core::card_mark(self.0.as_ptr().cast()) };
            unsafe { self.0.as_ref() }.borrow_mut()
        }

        /// `RefCell::try_borrow_mut` with the write barrier, as [`Env::borrow_mut`].
        // Kept as part of the explicit handle API after removing RefCell dereferencing.
        #[allow(dead_code)]
        #[inline]
        pub fn try_borrow_mut(
            &self,
        ) -> Result<std::cell::RefMut<'_, Scope>, std::cell::BorrowMutError> {
            // SAFETY: as in `borrow_mut`.
            unsafe { crate::gc_core::card_mark(self.0.as_ptr().cast()) };
            unsafe { self.0.as_ref() }.try_borrow_mut()
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

        /// The record while it lives (tests).
        #[cfg(test)]
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
        pub fn strong_count(&self) -> usize {
            usize::from(self.0.live_ptr().is_some())
        }
    }
}
