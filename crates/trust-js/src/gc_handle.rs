//! The engine's strong object handle.
//!
//! [`Gc`] is the `Copy` address of a GC cell holding the `RefCell<Object>` (P1 S2; see
//! `gc_core`). It dereferences to the `RefCell<Object>`, so `.borrow()`/`.borrow_mut()` work on
//! a handle. Identity goes through associated functions — [`Gc::as_ptr`], [`Gc::ptr_eq`],
//! [`Gc::downgrade`] — so `gc.as_ptr()` names `RefCell::as_ptr` through the deref. Copying a
//! handle is free and dropping one releases nothing; the collector reclaims unreachable cells.

#[cfg(feature = "gc-heap")]
pub(crate) use cell::{clear_dead_weak_cells, WeakRef};
#[cfg(feature = "gc-heap")]
pub use cell::{Gc, WeakGc};

#[cfg(feature = "gc-heap")]
mod cell {
    use crate::value::Object;
    use std::cell::{Cell, RefCell};
    use std::ptr::NonNull;
    use std::rc::Rc;

    /// A handle to a heap object: the address of its cell; see the module documentation.
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    pub struct Gc(NonNull<RefCell<Object>>);

    impl Gc {
        /// Allocate `object` in `heap`, the agent heap that owns it.
        #[inline]
        pub(crate) fn new_in(heap: &crate::value::GcHeap, object: RefCell<Object>) -> Gc {
            let cell = crate::value::alloc_object_cell(heap);
            // SAFETY: a fresh cell sized and aligned for `RefCell<Object>`. Allocation never
            // collects, so the cell is initialized before any collection can trace it.
            unsafe {
                std::ptr::write(cell, object);
                Gc(NonNull::new_unchecked(cell))
            }
        }

        /// A handle for the object at `ptr` ([`Gc::as_ptr`] / [`Gc::into_raw`] output).
        ///
        /// # Safety
        /// `ptr` must address a live object cell.
        #[inline]
        pub unsafe fn from_raw(ptr: *const RefCell<Object>) -> Gc {
            Gc(unsafe { NonNull::new_unchecked(ptr.cast_mut()) })
        }

        #[inline]
        pub fn into_raw(this: Gc) -> *const RefCell<Object> {
            this.0.as_ptr()
        }

        #[inline]
        pub fn as_ptr(this: &Gc) -> *const RefCell<Object> {
            this.0.as_ptr()
        }

        #[inline]
        pub fn ptr_eq(a: &Gc, b: &Gc) -> bool {
            a.0 == b.0
        }

        #[inline]
        pub fn downgrade(this: &Gc) -> WeakGc {
            WeakGc::of(this)
        }
    }

    impl std::ops::Deref for Gc {
        type Target = RefCell<Object>;
        #[inline]
        fn deref(&self) -> &RefCell<Object> {
            // SAFETY: a reachable handle names a live cell (Decision 0 / I1).
            unsafe { self.0.as_ref() }
        }
    }

    impl std::fmt::Pointer for Gc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            std::fmt::Pointer::fmt(&self.0, f)
        }
    }

    /// A weak object handle. [`WeakGc::upgrade`] yields a [`Gc`] until a collection (or the
    /// heap's teardown) finds the target dead and clears the shared cell.
    #[derive(Clone, Default)]
    pub struct WeakGc(WeakRef);

    /// The untyped weak reference behind [`WeakGc`] and `WeakEnv`: a target address and a flag
    /// the owning heap clears when the target dies. Never cleared means never upgraded twice
    /// across a reused address: the flag is cleared before the target's cell is freed.
    #[derive(Clone, Default)]
    pub(crate) struct WeakRef(Option<Rc<WeakCell>>);

    pub(crate) struct WeakCell {
        ptr: usize,
        alive: Cell<bool>,
    }

    impl WeakRef {
        pub(crate) fn to(ptr: usize) -> WeakRef {
            let cell = Rc::new(WeakCell {
                ptr,
                alive: Cell::new(true),
            });
            WEAK_CELLS.with(|cells| {
                let mut cells = cells.borrow_mut();
                if cells.cells.len() >= cells.prune_at {
                    cells.cells.retain(|cell| cell.strong_count() > 0);
                    cells.prune_at = (cells.cells.len() * 2).max(1024);
                }
                cells.cells.push(Rc::downgrade(&cell));
            });
            WeakRef(Some(cell))
        }

        /// The target address while it is alive.
        #[inline]
        pub(crate) fn live_ptr(&self) -> Option<usize> {
            let cell = self.0.as_ref()?;
            cell.alive.get().then_some(cell.ptr)
        }

        /// The target address, alive or not (null for a never-upgrading handle).
        #[inline]
        pub(crate) fn ptr(&self) -> usize {
            self.0.as_ref().map_or(0, |cell| cell.ptr)
        }

        #[inline]
        pub(crate) fn handles(&self) -> usize {
            self.0.as_ref().map_or(0, Rc::strong_count)
        }
    }

    struct WeakCells {
        cells: Vec<std::rc::Weak<WeakCell>>,
        prune_at: usize,
    }

    thread_local! {
        /// Every weak cell created on this thread; a heap clears the cells of its dead targets.
        static WEAK_CELLS: RefCell<WeakCells> = const {
            RefCell::new(WeakCells { cells: Vec::new(), prune_at: 1024 })
        };
    }

    impl WeakGc {
        fn of(target: &Gc) -> WeakGc {
            WeakGc(WeakRef::to(Gc::as_ptr(target) as usize))
        }

        /// A handle that never upgrades.
        #[inline]
        pub fn new() -> WeakGc {
            WeakGc(WeakRef::default())
        }

        #[inline]
        pub fn upgrade(&self) -> Option<Gc> {
            // SAFETY: the alive flag is cleared before the target's cell is freed.
            self.0
                .live_ptr()
                .map(|ptr| unsafe { Gc::from_raw(ptr as *const RefCell<Object>) })
        }

        /// The object's address ([`Gc::as_ptr`]), valid as an identity even after it died.
        #[inline]
        pub fn as_ptr(&self) -> *const RefCell<Object> {
            self.0.ptr() as *const RefCell<Object>
        }

        #[inline]
        pub fn ptr_eq(&self, other: &WeakGc) -> bool {
            self.as_ptr() == other.as_ptr()
        }

        #[inline]
        pub fn strong_count(&self) -> usize {
            usize::from(self.0.live_ptr().is_some())
        }

        #[inline]
        pub fn weak_count(&self) -> usize {
            self.0.handles()
        }
    }

    /// Clear the weak cells whose target `is_dead` (as decided by the owning heap's collection
    /// or teardown) and return those dead targets' addresses: while a weak handle exists, its
    /// target's cell must stay reserved (the heap keeps it as a zombie), exactly as
    /// `std::rc::Weak` keeps its allocation. Abandoned cells stop being tracked.
    pub(crate) fn clear_dead_weak_cells(
        mut is_dead: impl FnMut(usize) -> bool,
    ) -> crate::fasthash::FastSet<usize> {
        let mut pinned = crate::fasthash::FastSet::default();
        WEAK_CELLS.with(|cells| {
            cells.borrow_mut().cells.retain(|cell| {
                let Some(cell) = cell.upgrade() else {
                    return false;
                };
                if is_dead(cell.ptr) {
                    cell.alive.set(false);
                    pinned.insert(cell.ptr);
                }
                true
            });
        });
        pinned
    }
}
