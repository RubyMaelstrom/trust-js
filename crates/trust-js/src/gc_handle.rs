//! The engine's strong object handle.
//!
//! [`Gc`] is `repr(transparent)` over `Rc<RefCell<Object>>`: its bits, the packed-value payload,
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
