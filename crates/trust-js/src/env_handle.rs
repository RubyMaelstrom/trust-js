//! The engine's environment-record handle.
//!
//! Default build: [`Env`] is `repr(transparent)` over `Rc<RefCell<Scope>>`, with exactly the
//! bits, raw-pointer conversions and reference counting of that `Rc`. It dereferences to the
//! `RefCell<Scope>`, so `.borrow()`/`.borrow_mut()` work unchanged. Identity and ownership go
//! through associated functions — [`Env::as_ptr`], [`Env::ptr_eq`], [`Env::downgrade`],
//! [`Env::strong_count`] — never through `Rc` directly, so the P1 GC heap (feature `gc-heap`)
//! can replace the `Rc` behind this seam: Rust frames hold environments in locals everywhere,
//! and the conservative stack scan can only discover handles that point into the GC heap.
//!
//! Feature `gc-heap` (P1 S2): [`Env`] is the `Copy` address of a GC cell holding the
//! `RefCell<Scope>`, traced by the scope descriptor (see `value::trace_scope_cell`).

#[cfg(feature = "gc-heap")]
pub use cell::{Env, WeakEnv};
#[cfg(not(feature = "gc-heap"))]
pub use rc::{Env, WeakEnv};

#[cfg(not(feature = "gc-heap"))]
mod rc {
    use crate::interpreter::Scope;
    use std::cell::RefCell;
    use std::rc::Rc;

    /// A strong handle to an environment record; see the module documentation.
    #[repr(transparent)]
    #[derive(Clone)]
    pub struct Env(Rc<RefCell<Scope>>);

    impl Env {
        /// Allocate `scope` in its agent heap (`scope.gc_heap`).
        #[inline]
        pub(crate) fn new(scope: Scope) -> Env {
            Env(Rc::new(RefCell::new(scope)))
        }

        /// Adopt a registry snapshot's `Rc` (the refcount collector's scope snapshots).
        #[inline]
        pub(crate) fn from_rc(rc: Rc<RefCell<Scope>>) -> Env {
            Env(rc)
        }

        /// The record's address: its identity for address-keyed tables and raw-pointer round
        /// trips. An associated function (like `Rc::as_ptr`), so `env.as_ptr()` still names
        /// `RefCell::as_ptr`.
        #[inline]
        pub fn as_ptr(this: &Env) -> *const RefCell<Scope> {
            Rc::as_ptr(&this.0)
        }

        #[inline]
        pub fn ptr_eq(a: &Env, b: &Env) -> bool {
            Rc::ptr_eq(&a.0, &b.0)
        }

        #[inline]
        pub fn downgrade(this: &Env) -> WeakEnv {
            WeakEnv(Rc::downgrade(&this.0))
        }

        /// The number of strong handles. Production code asks [`Env::outlives_release`]
        /// instead: a tracing heap has no such count.
        #[inline]
        pub fn strong_count(this: &Env) -> usize {
            Rc::strong_count(&this.0)
        }

        #[inline]
        pub fn weak_count(this: &Env) -> usize {
            Rc::weak_count(&this.0)
        }

        /// Whether the record survives the release of `released` of its handles (always true
        /// on a tracing heap, where releasing a handle never destroys anything).
        #[inline]
        pub fn outlives_release(this: &Env, released: usize) -> bool {
            Rc::strong_count(&this.0) > released
        }

        /// Reconstruct a handle from [`Env::into_raw`] output.
        ///
        /// # Safety
        /// As for `Rc::from_raw`: `ptr` must carry one strong reference that this handle owns.
        #[inline]
        pub unsafe fn from_raw(ptr: *const RefCell<Scope>) -> Env {
            Env(unsafe { Rc::from_raw(ptr) })
        }

        /// Give up this handle's reference as a raw pointer.
        #[inline]
        pub fn into_raw(this: Env) -> *const RefCell<Scope> {
            Rc::into_raw(this.0)
        }

        /// A new owner of the record at `ptr`, which another handle keeps alive.
        ///
        /// # Safety
        /// `ptr` must be [`Env::as_ptr`] of a live record.
        #[inline]
        pub unsafe fn clone_from_raw(ptr: *const RefCell<Scope>) -> Env {
            unsafe {
                Rc::increment_strong_count(ptr);
                Env::from_raw(ptr)
            }
        }
    }

    impl std::ops::Deref for Env {
        type Target = RefCell<Scope>;
        #[inline]
        fn deref(&self) -> &RefCell<Scope> {
            &self.0
        }
    }

    /// A weak environment handle. [`WeakEnv::upgrade`] yields an [`Env`] while it is alive.
    #[derive(Clone, Default)]
    pub struct WeakEnv(std::rc::Weak<RefCell<Scope>>);

    impl WeakEnv {
        #[inline]
        pub fn new() -> WeakEnv {
            WeakEnv(std::rc::Weak::new())
        }

        #[inline]
        pub fn upgrade(&self) -> Option<Env> {
            self.0.upgrade().map(Env)
        }

        /// The record's address ([`Env::as_ptr`]), valid as an identity even after it died.
        #[inline]
        pub fn as_ptr(&self) -> *const RefCell<Scope> {
            self.0.as_ptr()
        }

        #[inline]
        pub fn ptr_eq(&self, other: &WeakEnv) -> bool {
            self.0.ptr_eq(&other.0)
        }

        #[inline]
        pub fn strong_count(&self) -> usize {
            self.0.strong_count()
        }
    }
}

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
        /// Allocate `scope` in its agent heap (`scope.gc_heap`).
        #[inline]
        pub(crate) fn new(scope: Scope) -> Env {
            let cell = crate::value::alloc_scope_cell(&scope.gc_heap);
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

        /// The GC heap has no reference counts: only the disabled cycle collector and
        /// refcount-observing tests ask, so reaching this is a bug.
        #[track_caller]
        pub fn strong_count(_: &Env) -> usize {
            panic!("Env::strong_count: the gc-heap object model has no reference counts")
        }

        #[track_caller]
        pub fn weak_count(_: &Env) -> usize {
            panic!("Env::weak_count: the gc-heap object model has no reference counts")
        }

        #[inline]
        pub fn outlives_release(_: &Env, _: usize) -> bool {
            true
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

        /// # Safety
        /// `ptr` must address a live environment cell.
        #[inline]
        pub unsafe fn clone_from_raw(ptr: *const RefCell<Scope>) -> Env {
            unsafe { Env::from_raw(ptr) }
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
