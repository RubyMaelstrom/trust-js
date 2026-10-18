//! The engine's strong object handle.
//!
//! Default build: [`Gc`] is `repr(transparent)` over `Rc<RefCell<Object>>`: its bits, the packed-value payload,
//! every JIT layout offset and every raw-pointer conversion are exactly those of the `Rc`. It
//! dereferences to the `RefCell<Object>`, so `.borrow()`/`.borrow_mut()` keep working unchanged.
//! Identity and ownership go through associated functions — [`Gc::as_ptr`], [`Gc::ptr_eq`],
//! [`Gc::downgrade`], [`Gc::strong_count`] — never through `Rc` directly, so the P1 GC heap
//! (feature `gc-heap`) can replace the `Rc` behind this seam. What the wrapper adds is an
//! engine-owned release path: the collector and memory-management policy can observe the release
//! of a final reference without changing any of the thousands of handle users.
//!
//! Releasing the final reference destroys the object and, through ordinary field drop glue,
//! releases its children. Chains (linked lists, closure chains, nested arrays) therefore used to
//! recurse once per link, and a 50,000-node list overflowed the native stack. A final release
//! now counts destruction depth; past [`DEPTH_LIMIT`] the handle is queued, and the outermost
//! release drains the queue iteratively. The count is a destructor-free thread local (a single
//! TLS-relative load) and non-final releases never touch it.

//!
//! Feature `gc-heap` (P1 S2): [`Gc`] is instead the `Copy` address of a GC cell holding the
//! `RefCell<Object>` (see `gc_core`). Copying a handle is free and dropping one releases
//! nothing; the collector reclaims unreachable cells. Both builds expose the same API.

#[cfg(feature = "gc-heap")]
pub(crate) use cell::{clear_dead_weak_cells, WeakRef};
#[cfg(feature = "gc-heap")]
pub use cell::{Gc, WeakGc};
#[cfg(not(feature = "gc-heap"))]
pub use rc::{Gc, WeakGc};

#[cfg(not(feature = "gc-heap"))]
mod rc {
    use crate::value::Object;
    use std::cell::{Cell, RefCell};
    use std::mem::ManuallyDrop;
    use std::rc::Rc;

    /// A strong handle to a heap object; see the module documentation.
    #[repr(transparent)]
    pub struct Gc(ManuallyDrop<Rc<RefCell<Object>>>);

    impl Gc {
        /// Adopt an `Rc` produced by a raw-pointer round trip or a `Weak` upgrade.
        #[inline]
        pub fn from_rc(rc: Rc<RefCell<Object>>) -> Gc {
            Gc(ManuallyDrop::new(rc))
        }

        /// Allocate `object` in `heap`, the agent heap that owns it.
        #[inline]
        pub(crate) fn new_in(heap: &crate::value::GcHeap, object: RefCell<Object>) -> Gc {
            let _ = heap;
            Gc::from_rc(Rc::new(object))
        }

        /// Reconstruct a handle from [`Rc::into_raw`]/[`Rc::as_ptr`] output.
        ///
        /// # Safety
        /// As for [`Rc::from_raw`]: `ptr` must come from an `Rc<RefCell<Object>>` and carry one strong
        /// reference that this handle now owns.
        #[inline]
        pub unsafe fn from_raw(ptr: *const RefCell<Object>) -> Gc {
            Gc::from_rc(unsafe { Rc::from_raw(ptr) })
        }

        /// Give up this handle's reference as a raw pointer (see [`Rc::into_raw`]).
        #[inline]
        pub fn into_raw(this: Gc) -> *const RefCell<Object> {
            let this = ManuallyDrop::new(this);
            Rc::as_ptr(&this.0)
        }

        /// The underlying `Rc`, consuming this handle without releasing its reference.
        #[inline]
        pub fn into_rc(this: Gc) -> Rc<RefCell<Object>> {
            let mut this = ManuallyDrop::new(this);
            // SAFETY: `this` is never used or dropped again, so the Rc is moved out exactly once.
            unsafe { ManuallyDrop::take(&mut this.0) }
        }

        /// The object's address: its identity for address-keyed tables and raw-pointer round trips.
        /// An associated function (like `Rc::as_ptr`), so `gc.as_ptr()` still names
        /// `RefCell::as_ptr`.
        #[inline]
        pub fn as_ptr(this: &Gc) -> *const RefCell<Object> {
            Rc::as_ptr(&this.0)
        }

        /// Whether two handles name the same object.
        #[inline]
        pub fn ptr_eq(a: &Gc, b: &Gc) -> bool {
            Rc::ptr_eq(&a.0, &b.0)
        }

        /// A weak handle (see [`WeakGc`]).
        #[inline]
        pub fn downgrade(this: &Gc) -> WeakGc {
            WeakGc(Rc::downgrade(&this.0))
        }

        /// The number of strong handles. Production code asks [`Gc::outlives_release`] instead: a
        /// tracing heap has no such count.
        #[inline]
        pub fn strong_count(this: &Gc) -> usize {
            Rc::strong_count(&this.0)
        }

        /// Whether the object survives the release of `released` of its handles: releasing them
        /// cannot run its destructor (or weak observers) synchronously. Always true on a tracing
        /// heap, where releasing a handle never destroys anything.
        #[inline]
        pub fn outlives_release(this: &Gc, released: usize) -> bool {
            Rc::strong_count(&this.0) > released
        }

        /// The `Rc` behind this handle, for the few paths that genuinely depend on `Rc` itself
        /// (generated-code layouts, `Rc`-typed host interfaces). Each use is a P1 migration site.
        #[inline]
        pub fn as_rc(this: &Gc) -> &Rc<RefCell<Object>> {
            &this.0
        }

        /// A new owner of the object at `ptr`, which another handle keeps alive.
        ///
        /// # Safety
        /// `ptr` must be [`Gc::as_ptr`] of a live object.
        #[inline]
        pub unsafe fn clone_from_raw(ptr: *const RefCell<Object>) -> Gc {
            unsafe {
                Rc::increment_strong_count(ptr);
                Gc::from_raw(ptr)
            }
        }

        /// Release a handle, skipping the final-release bookkeeping when another owner remains.
        /// Equivalent to `drop(this)`; hot VM paths use it for consumed callee slots.
        #[inline]
        pub fn release(this: Gc) {
            if Rc::strong_count(&this.0) > 1 {
                // SAFETY: `into_raw` gives up this handle's reference, which is released here.
                unsafe { Rc::decrement_strong_count(Gc::into_raw(this)) };
            } else {
                drop(this);
            }
        }
    }

    /// A weak object handle. [`WeakGc::upgrade`] yields a [`Gc`] while the object is alive.
    #[derive(Clone, Default)]
    pub struct WeakGc(std::rc::Weak<RefCell<Object>>);

    impl WeakGc {
        /// A handle that never upgrades.
        #[inline]
        pub fn new() -> WeakGc {
            WeakGc(std::rc::Weak::new())
        }

        #[inline]
        pub fn upgrade(&self) -> Option<Gc> {
            self.0.upgrade().map(Gc::from_rc)
        }

        /// The object's address ([`Gc::as_ptr`]), valid as an identity even after it died.
        #[inline]
        pub fn as_ptr(&self) -> *const RefCell<Object> {
            self.0.as_ptr()
        }

        #[inline]
        pub fn ptr_eq(&self, other: &WeakGc) -> bool {
            self.0.ptr_eq(&other.0)
        }

        #[inline]
        pub fn strong_count(&self) -> usize {
            self.0.strong_count()
        }

        #[inline]
        pub fn weak_count(&self) -> usize {
            self.0.weak_count()
        }
    }

    impl std::ops::Deref for Gc {
        type Target = RefCell<Object>;
        #[inline]
        fn deref(&self) -> &RefCell<Object> {
            &self.0
        }
    }

    impl Clone for Gc {
        #[inline]
        fn clone(&self) -> Gc {
            Gc::from_rc(Rc::clone(&self.0))
        }
    }

    impl From<Rc<RefCell<Object>>> for Gc {
        #[inline]
        fn from(rc: Rc<RefCell<Object>>) -> Gc {
            Gc::from_rc(rc)
        }
    }

    impl std::fmt::Pointer for Gc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            std::fmt::Pointer::fmt(&*self.0, f)
        }
    }

    /// Nested final releases before further ones are queued. Each level costs a few hundred bytes of
    /// native stack in the drop glue, so the bound keeps destruction within a small constant stack.
    const DEPTH_LIMIT: u32 = 128;

    thread_local! {
        static DEPTH: Cell<u32> = const { Cell::new(0) };
        static PENDING: Cell<bool> = const { Cell::new(false) };
        static QUEUE: RefCell<Vec<Rc<RefCell<Object>>>> = const { RefCell::new(Vec::new()) };
    }

    impl Drop for Gc {
        #[inline]
        fn drop(&mut self) {
            if Rc::strong_count(&self.0) != 1 {
                // Not the final reference: an ordinary decrement.
                // SAFETY: `self.0` is dropped exactly once, here.
                unsafe { ManuallyDrop::drop(&mut self.0) };
                return;
            }
            // SAFETY: `self.0` is moved out exactly once and never used again.
            release_final(unsafe { ManuallyDrop::take(&mut self.0) });
        }
    }

    #[inline(never)]
    fn release_final(rc: Rc<RefCell<Object>>) {
        let Ok(depth) = DEPTH.try_with(Cell::get) else {
            // Thread-local teardown: ordinary (recursive) destruction.
            drop(rc);
            return;
        };
        if depth >= DEPTH_LIMIT {
            // Queue the handle; the outermost release destroys it. If the queue itself is already
            // torn down, the closure drops `rc` in place.
            if QUEUE
                .try_with(move |queue| queue.borrow_mut().push(rc))
                .is_ok()
            {
                PENDING.with(|pending| pending.set(true));
            }
            return;
        }
        // `with` on these const, destructor-free keys is a plain TLS access; `LocalKey::set` goes
        // through the lazy-initialization path.
        DEPTH.with(|cell| cell.set(depth + 1));
        drop(rc);
        if depth == 0 && PENDING.with(Cell::get) {
            // Each queued release runs at depth one and may queue deeper links again.
            while let Some(next) = QUEUE
                .try_with(|queue| queue.borrow_mut().pop())
                .ok()
                .flatten()
            {
                drop(next);
            }
            PENDING.with(|pending| pending.set(false));
        }
        DEPTH.with(|cell| cell.set(depth));
    }
}

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

        /// Releasing a handle never destroys anything on a tracing heap.
        #[inline]
        pub fn outlives_release(_: &Gc, _: usize) -> bool {
            true
        }

        /// # Safety
        /// `ptr` must address a live object cell.
        #[inline]
        pub unsafe fn clone_from_raw(ptr: *const RefCell<Object>) -> Gc {
            unsafe { Gc::from_raw(ptr) }
        }

        #[inline]
        pub fn release(_: Gc) {}
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
