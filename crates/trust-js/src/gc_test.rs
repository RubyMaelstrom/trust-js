//! Test support for observing collection under the conservative stack scan (feature
//! `gc-heap`, Decision 0): a strong handle left in a test function's frame — even after `drop`,
//! since handles are `Copy`, and even as an `upgrade()` temporary — keeps its object alive.
//! Tests that expect an object to be reclaimed hold it only inside [`in_frame`] and ask
//! [`alive`] instead of upgrading in their own frame.

/// Run `f` in its own frame, which is popped (and then scrubbed by the next explicit
/// collection) before the caller continues.
#[inline(never)]
pub(crate) fn in_frame<R>(f: impl FnOnce() -> R) -> R {
    let result = f();
    std::hint::black_box(());
    result
}

/// A weak handle to a cell: an object or an environment record.
pub(crate) trait WeakCell {
    /// The referent's address while it is alive.
    fn live_address(&self) -> Option<usize>;
    /// The referent's heap while it is alive (retention paths).
    fn heap(&self) -> Option<crate::value::GcHeap>;
}

impl WeakCell for crate::value::WeakGc {
    fn live_address(&self) -> Option<usize> {
        self.upgrade()
            .map(|object| crate::value::Gc::as_ptr(&object) as usize)
    }

    fn heap(&self) -> Option<crate::value::GcHeap> {
        self.upgrade().map(|object| object.borrow().gc_heap.clone())
    }
}

impl WeakCell for crate::interpreter::WeakEnv {
    fn live_address(&self) -> Option<usize> {
        self.upgrade()
            .map(|env| crate::interpreter::Env::as_ptr(&env) as usize)
    }

    /// The active heap: an environment cell records no heap of its own.
    fn heap(&self) -> Option<crate::value::GcHeap> {
        self.upgrade().map(|_| crate::value::active_gc_heap())
    }
}

impl WeakCell for crate::interpreter::WeakValue {
    fn live_address(&self) -> Option<usize> {
        match self.upgrade() {
            Some(crate::value::Value::Obj(object)) => {
                Some(crate::value::Gc::as_ptr(&object) as usize)
            }
            _ => None,
        }
    }

    fn heap(&self) -> Option<crate::value::GcHeap> {
        match self.upgrade() {
            Some(crate::value::Value::Obj(object)) => Some(object.borrow().gc_heap.clone()),
            _ => None,
        }
    }
}

/// Reference-counted Rust allocations (code, layouts, heaps): not cells, but asked the same way.
impl<T> WeakCell for std::rc::Weak<T> {
    fn live_address(&self) -> Option<usize> {
        self.upgrade()
            .map(|strong| std::rc::Rc::as_ptr(&strong) as *const u8 as usize)
    }

    fn heap(&self) -> Option<crate::value::GcHeap> {
        None
    }
}

/// A Symbol cell (gc-heap).
impl WeakCell for crate::value::WeakSymbol {
    fn live_address(&self) -> Option<usize> {
        self.upgrade()
            .map(|symbol| crate::value::symbol_identity(&symbol))
    }

    fn heap(&self) -> Option<crate::value::GcHeap> {
        None
    }
}

impl WeakCell for crate::interpreter::WeakTarget {
    fn live_address(&self) -> Option<usize> {
        match self {
            crate::interpreter::WeakTarget::Object(object) => object.live_address(),
            crate::interpreter::WeakTarget::Symbol(symbol, _) => symbol.live_address(),
        }
    }

    fn heap(&self) -> Option<crate::value::GcHeap> {
        match self {
            crate::interpreter::WeakTarget::Object(object) => object.heap(),
            crate::interpreter::WeakTarget::Symbol(..) => None,
        }
    }
}

impl<T: WeakCell> WeakCell for &T {
    fn live_address(&self) -> Option<usize> {
        (**self).live_address()
    }

    fn heap(&self) -> Option<crate::value::GcHeap> {
        (**self).heap()
    }
}

/// Whether `weak`'s referent is alive, asked in a frame of its own: `weak.upgrade()` in the
/// caller would leave a strong handle in a caller temporary for the rest of the function.
#[inline(never)]
pub(crate) fn alive(weak: &impl WeakCell) -> bool {
    let alive = weak.live_address().is_some();
    if alive && std::env::var_os("TRUST_JS_GC_WHY").is_some() {
        eprintln!("[gc-why] alive:\n{}", why(weak));
    }
    alive
}

/// The retention path of `weak`'s referent in the last collection of its heap (run with
/// `TRUST_JS_GC_WHY=1`).
#[inline(never)]
pub(crate) fn why(weak: &impl WeakCell) -> String {
    let Some(address) = weak.live_address() else {
        return "dead".into();
    };
    let Some(heap) = weak.heap() else {
        return format!("  {address:#x}: not a GC cell");
    };
    why_address(&heap, address)
}

/// The retention path of the cell at `address` in `heap`'s last collection.
#[inline(never)]
pub(crate) fn why_address(heap: &crate::value::GcHeap, address: usize) -> String {
    let core = heap.core.borrow();
    let anchor = why as fn(&crate::value::WeakGc) -> String as usize;
    core.retention_path(address)
        .iter()
        .map(|(cell, kind, source)| match source {
            crate::gc_core::MarkSource::Stack(word) => match core.frame_return_address(*word) {
                Some((ret, depth)) => format!(
                    "  {cell:#x} {kind} <- {source:x?} {depth} bytes below the record of the frame returning to Some({ret:x}) (why() at {anchor:#x})"
                ),
                None => format!("  {cell:#x} {kind} <- {source:x?} outside the traced frames"),
            },
            _ => format!("  {cell:#x} {kind} <- {source:x?}"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `TRUST_JS_SEGV_BACKTRACE=1`: print a backtrace when a test crashes on a fault (debugging
/// use-after-free under GC stress). Not async-signal-safe; good enough to locate a crash.
#[cfg(target_os = "linux")]
pub(crate) fn install_crash_backtrace() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    if std::env::var_os("TRUST_JS_SEGV_BACKTRACE").is_none() {
        return;
    }
    extern "C" fn on_fault(signal: i32) {
        eprintln!(
            "fatal signal {signal}:\n{}",
            std::backtrace::Backtrace::force_capture()
        );
        std::process::abort();
    }
    unsafe extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }
    ONCE.call_once(|| {
        // SAFETY: installs a handler for SIGSEGV (11) and SIGBUS (7).
        unsafe {
            signal(11, on_fault as extern "C" fn(i32) as usize);
            signal(7, on_fault as extern "C" fn(i32) as usize);
        }
    });
}

/// Zero this thread's stack when it exits (gc-heap test builds). libtest runs each test on a
/// new thread, and glibc gives a new thread a cached stack with its old contents: slots a test
/// function has not written yet then hold addresses an earlier test's frames left there, and a
/// heap whose pages land at that test's addresses takes them for roots.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn scrub_stack_at_thread_exit() {
    struct Scrub;
    impl Drop for Scrub {
        fn drop(&mut self) {
            crate::gc_core::scrub_stack(usize::MAX);
        }
    }
    thread_local! {
        static SCRUB: Scrub = const { Scrub };
    }
    SCRUB.with(|_| {});
}

/// A cell address a test keeps without retaining the cell: the conservative stack scan takes a
/// plain integer equal to a cell's address for a reference. Stored complemented (never a valid
/// address); computed and compared only in helper frames.
#[derive(Clone, Copy)]
pub(crate) struct HiddenAddress(usize);

impl HiddenAddress {
    /// The address of the object `value` holds.
    #[inline(never)]
    pub(crate) fn of(value: &crate::value::Value) -> Self {
        let object = value.as_obj().expect("an object");
        Self(!(crate::value::Gc::as_ptr(object) as usize))
    }

    /// The address `address()` computes, in a frame of its own.
    #[inline(never)]
    pub(crate) fn from_fn(address: impl FnOnce() -> usize) -> Self {
        Self(!address())
    }

    /// Whether the address-keyed `map` has an entry for this address.
    #[inline(never)]
    pub(crate) fn in_map<V, S: std::hash::BuildHasher>(
        self,
        map: &std::collections::HashMap<usize, V, S>,
    ) -> bool {
        map.contains_key(&!self.0)
    }

    /// The address itself, for a check in a helper frame (its callers are `embed` tests).
    #[inline(never)]
    #[cfg_attr(not(feature = "embed"), allow(dead_code))]
    pub(crate) fn get(self) -> usize {
        !self.0
    }
}

/// Whether `TRUST_JS_GC_STRESS` is driving collections (gc-heap). Tests that assert when
/// collections or cleanup jobs happen skip their checks then: stress collects at every
/// safepoint by design.
pub(crate) fn stress_active() -> bool {
    crate::interpreter::gc_collect::gc_stress() != 0
}

/// The JavaScript values a test's native closure captures, reported for it
/// (`NativeCaptures::Reported`).
pub(crate) struct CapturedValues(pub(crate) Vec<crate::value::Value>);

impl crate::value::NativeCallableRetained for CapturedValues {
    fn scan_retained_memory(&self, visitor: &mut dyn crate::value::NativeRetainedMemoryVisitor) {
        for value in &self.0 {
            visitor.value(value);
        }
    }
}
