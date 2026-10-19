//! P1 S2: precise tracing of the engine's Rust-side GC edges.
//!
//! [`GcSink`] receives strong edges as cell words; [`GcTrace`] is implemented by every engine
//! type that can hold a value, an object handle or an environment. Implementations for engine
//! structs destructure their type exhaustively (no `..`), so adding a field does not compile
//! until it is classified as traced or not: the Decision 2 registry, enforced by the type
//! system rather than by review. A verifier that walks these same implementations cannot detect
//! an omitted field; the exhaustive patterns are what close that gap.

use crate::interpreter::Env;
use crate::value::{Gc, PackedValue, Value};

/// Receives the strong edges found while tracing roots or cell payloads.
pub(crate) trait GcSink {
    /// A cell address or tagged word. The heap resolves it; words that address no cell of the
    /// collecting heap (numbers, other Agents' cells) are ignored.
    fn word(&mut self, word: u64);

    #[inline]
    fn object(&mut self, object: &Gc) {
        self.word(Gc::as_ptr(object) as u64);
    }

    #[inline]
    fn env(&mut self, env: &Env) {
        self.word(Env::as_ptr(env) as u64);
    }

    #[inline]
    fn value(&mut self, value: &Value) {
        if let Some(word) = value.cell_word() {
            self.word(word);
        }
    }
}

/// Report every strong GC edge this value holds.
pub(crate) trait GcTrace {
    fn trace_gc(&self, sink: &mut dyn GcSink);
}

impl GcTrace for Value {
    #[inline]
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        sink.value(self);
    }
}

impl GcTrace for Gc {
    #[inline]
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        sink.object(self);
    }
}

impl GcTrace for crate::lstr::LStr {
    #[inline]
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        sink.word(self.cell_word());
    }
}

/// Rust-heap string slices are not GC cells.
impl GcTrace for str {
    #[inline]
    fn trace_gc(&self, _: &mut dyn GcSink) {}
}

impl GcTrace for Env {
    #[inline]
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        sink.env(self);
    }
}

impl GcTrace for PackedValue {
    #[inline]
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        self.trace_gc_edges(sink);
    }
}

impl<T: GcTrace> GcTrace for Option<T> {
    #[inline]
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        if let Some(inner) = self {
            inner.trace_gc(sink);
        }
    }
}

impl<T: GcTrace + ?Sized> GcTrace for Box<T> {
    #[inline]
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        (**self).trace_gc(sink);
    }
}

/// Shared payloads are traced through every holder; a payload reached twice is harmless.
impl<T: GcTrace + ?Sized> GcTrace for std::rc::Rc<T> {
    #[inline]
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        (**self).trace_gc(sink);
    }
}

impl<T: GcTrace> GcTrace for [T] {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        for item in self {
            item.trace_gc(sink);
        }
    }
}

impl<T: GcTrace> GcTrace for Vec<T> {
    #[inline]
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        self.as_slice().trace_gc(sink);
    }
}

impl<T: GcTrace> GcTrace for std::collections::VecDeque<T> {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        for item in self {
            item.trace_gc(sink);
        }
    }
}

impl<T: GcTrace> GcTrace for std::cell::RefCell<T> {
    /// Root and payload tracing run at safepoints, where no engine borrow is outstanding (I9);
    /// a held mutable borrow here is a bug the panic locates.
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        self.try_borrow()
            .expect("GC traced a container mutably borrowed at a safepoint (I9)")
            .trace_gc(sink);
    }
}

impl<A: GcTrace, B: GcTrace> GcTrace for (A, B) {
    #[inline]
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        self.0.trace_gc(sink);
        self.1.trace_gc(sink);
    }
}

/// Rust strings are not GC cells (side-table names and labels).
impl GcTrace for String {
    #[inline]
    fn trace_gc(&self, _: &mut dyn GcSink) {}
}

/// Values of a map; keys are traced by the map's owner when they are values.
impl<K, V: GcTrace, S> GcTrace for std::collections::HashMap<K, V, S> {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        for value in self.values() {
            value.trace_gc(sink);
        }
    }
}

/// A [`GcSink`] over a `gc_core` trace visitor (cell descriptors' trace callbacks).
pub(crate) struct VisitSink<'a>(pub(crate) &'a mut dyn FnMut(*const u8));

impl GcSink for VisitSink<'_> {
    #[inline]
    fn word(&mut self, word: u64) {
        (self.0)(word as *const u8);
    }
}

impl GcTrace for crate::interpreter::Scope {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::Scope {
            vars,
            parent,
            with_obj,
            under_with: _,
            var_boundary: _,
            catch_param: _,
            lexical_names: _,
        } = self;
        for binding in vars.values() {
            binding.trace_gc(sink);
        }
        parent.trace_gc(sink);
        with_obj.trace_gc(sink);
    }
}

impl GcTrace for crate::interpreter::Binding {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::Binding {
            value,
            mutable: _,
            initialized: _,
            import_ref,
            imported: _,
            deletable: _,
            strict_immutable: _,
        } = self;
        value.trace_gc(sink);
        if let Some(import) = import_ref {
            sink.env(&import.0);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// State Rust frames keep on the Rust heap while they run (I1): the stack scan sees the frames'
// pointers to it, never its contents, so the owners register it for the duration.

/// One active `drive_vm` activation's frame state (slots, operand stack, references, class
/// evaluation and disposal state); the buffers live on the Rust heap.
pub(crate) struct VmFrameRoots {
    pub(crate) env: *const Env,
    pub(crate) cap_env: *const Env,
    pub(crate) slots: *const [PackedValue],
    pub(crate) stack: *const crate::execution_storage::ValueStack<PackedValue>,
    pub(crate) references: *const [crate::eval::PreparedReferenceSlot],
    pub(crate) this_val: *const Value,
    pub(crate) disposal_frames: *const Vec<Vec<crate::interpreter::Disposable>>,
    pub(crate) class_states: *const [Option<crate::eval::PreparedClassEvaluation>],
}

/// One running JIT activation whose context lives in a Rust entry's frame (`jit::run`,
/// `jit::run_moved_oversized` and `jit::run_borrowed_frame`; the other activations run on pooled
/// records, see `jit::JitFrame`). The stack scan sees the context's own words, but not the
/// Rust-heap buffers its machine code keeps slots and operands in, nor what its fields own.
/// The entry registers both for the whole native run ([`enter_jit_frame`]); traced as the
/// context's values precisely (`GcTrace for JitCtx`) and each word range conservatively
/// ([`scan_words`]).
pub(crate) struct JitFrameRoots {
    pub(crate) ctx: *const crate::jit::JitCtx,
    /// Word ranges `(start, words)`: the entry's slots (none for a borrowed VM frame, whose
    /// slots `drive_vm` registers) and its operand buffer. Every word in them has been written
    /// before the run starts (each entry says how); stale words above the live operand top can
    /// only retain.
    pub(crate) words: [(*const u64, usize); 2],
}

thread_local! {
    static VM_FRAMES: std::cell::RefCell<Vec<VmFrameRoots>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static JIT_FRAMES: std::cell::RefCell<Vec<JitFrameRoots>> =
        const { std::cell::RefCell::new(Vec::new()) };
    /// Coroutines removed from their owner's table while they run.
    static RUNNING_COROUTINES: std::cell::RefCell<Vec<*const crate::coroutine::Coroutine>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Unregisters a [`VmFrameRoots`] (also on unwind).
#[must_use]
pub(crate) struct VmFrameGuard;

impl Drop for VmFrameGuard {
    fn drop(&mut self) {
        VM_FRAMES.with(|frames| frames.borrow_mut().pop());
    }
}

/// Register a running VM activation's frame state. The referents must outlive the guard.
pub(crate) fn enter_vm_frame(roots: VmFrameRoots) -> VmFrameGuard {
    VM_FRAMES.with(|frames| frames.borrow_mut().push(roots));
    VmFrameGuard
}

/// Unregisters a running coroutine (also on unwind).
#[must_use]
pub(crate) struct CoroutineGuard;

impl Drop for CoroutineGuard {
    fn drop(&mut self) {
        RUNNING_COROUTINES.with(|running| running.borrow_mut().pop());
    }
}

/// Register a coroutine for the duration of its resumption; it must outlive the guard.
pub(crate) fn enter_coroutine(coroutine: *const crate::coroutine::Coroutine) -> CoroutineGuard {
    RUNNING_COROUTINES.with(|running| running.borrow_mut().push(coroutine));
    CoroutineGuard
}

/// Unregisters a [`JitFrameRoots`] (also on unwind).
#[must_use]
pub(crate) struct JitFrameGuard;

impl Drop for JitFrameGuard {
    fn drop(&mut self) {
        JIT_FRAMES.with(|frames| frames.borrow_mut().pop());
    }
}

/// Register a Rust entry's running JIT activation. The context and the word ranges must stay
/// in place, and the ranges written, until the guard drops.
pub(crate) fn enter_jit_frame(roots: JitFrameRoots) -> JitFrameGuard {
    JIT_FRAMES.with(|frames| frames.borrow_mut().push(roots));
    JitFrameGuard
}

/// The contexts of the Rust entries' JIT activations registered on this thread (tests).
#[cfg(test)]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) fn jit_frame_contexts() -> Vec<*const crate::jit::JitCtx> {
    JIT_FRAMES.with(|frames| frames.borrow().iter().map(|frame| frame.ctx).collect())
}

/// The word-range lengths of each Rust entry's JIT activation registered on this thread
/// (tests).
#[cfg(test)]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) fn jit_frames_registered() -> Vec<[usize; 2]> {
    JIT_FRAMES.with(|frames| {
        frames
            .borrow()
            .iter()
            .map(|frame| [frame.words[0].1, frame.words[1].1])
            .collect()
    })
}

/// Report each of the `len` words at `start` as a candidate cell word: a conservative root
/// range. The heap resolves the words that address an allocated cell (masking a packed word's
/// tag) and ignores the rest, so a stale or numeric word can only retain.
///
/// # Safety
/// `start..start + len` is readable, and every word in it has been written.
pub(crate) unsafe fn scan_words(start: *const u64, len: usize, sink: &mut dyn GcSink) {
    for k in 0..len {
        // SAFETY: the caller's contract.
        sink.word(unsafe { start.add(k).read() });
    }
}

/// Trace this thread's registered VM frames, Rust entries' JIT activations, rooted value
/// buffers and running coroutines.
pub(crate) fn trace_thread_roots(sink: &mut dyn GcSink) {
    VM_FRAMES.with(|frames| {
        for frame in frames.borrow().iter() {
            // SAFETY: a frame is registered only while drive_vm holds its referents.
            unsafe {
                (*frame.env).trace_gc(sink);
                (*frame.cap_env).trace_gc(sink);
                (*frame.slots).trace_gc(sink);
                (*frame.stack).trace_gc(sink);
                (*frame.references).trace_gc(sink);
                (*frame.this_val).trace_gc(sink);
                (*frame.disposal_frames).trace_gc(sink);
                (*frame.class_states).trace_gc(sink);
            }
        }
    });
    JIT_FRAMES.with(|frames| {
        for frame in frames.borrow().iter() {
            // SAFETY: registered only while the entry's native run holds the context and its
            // written word ranges in place.
            unsafe {
                (*frame.ctx).trace_gc(sink);
                for &(start, len) in &frame.words {
                    scan_words(start, len, sink);
                }
            }
        }
    });
    crate::rooting::trace_rooted(sink);
    RUNNING_COROUTINES.with(|running| {
        for &coroutine in running.borrow().iter() {
            // SAFETY: registered only while `Coroutine::resume` borrows it.
            unsafe { GcTrace::trace_gc(&*coroutine, sink) };
        }
    });
}

impl GcTrace for crate::execution_storage::ValueStack<PackedValue> {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        self.raw_slice().trace_gc(sink);
    }
}

// ---------------------------------------------------------------------------------------------
// Engine types with public fields. Each destructures exhaustively.

impl GcTrace for crate::interpreter::RealmState {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::RealmState {
            global,
            global_env,
            global_var_names: _,
            host_job_context: _,
            object_proto,
            function_proto,
            array_proto,
            array_ctor,
            string_proto,
            number_proto,
            boolean_proto,
            symbol_proto,
            error_protos,
            eval_fn,
            extra_protos,
        } = self;
        for object in [
            global,
            object_proto,
            function_proto,
            array_proto,
            string_proto,
            number_proto,
            boolean_proto,
            symbol_proto,
        ] {
            sink.object(object);
        }
        global_env.trace_gc(sink);
        array_ctor.trace_gc(sink);
        error_protos.trace_gc(sink);
        eval_fn.trace_gc(sink);
        extra_protos.trace_gc(sink);
    }
}

impl GcTrace for crate::interpreter::PromiseState {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::PromiseState {
            status: _,
            value,
            reactions,
            handled: _,
            outstanding: _,
        } = self;
        value.trace_gc(sink);
        for (fulfilled, rejected, result, _) in reactions {
            fulfilled.trace_gc(sink);
            rejected.trace_gc(sink);
            result.trace_gc(sink);
        }
    }
}

impl GcTrace for crate::interpreter::JobCallback {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::JobCallback {
            callback,
            script_caller,
            active_script: _,
            internal: _, // realm identities and flags only
        } = self;
        callback.trace_gc(sink);
        script_caller.trace_gc(sink);
    }
}

impl GcTrace for crate::interpreter::Job {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::Job {
            handler,
            result,
            value,
            fulfilled: _,
            realm: _,
            host_context: _,
        } = self;
        handler.trace_gc(sink);
        result.trace_gc(sink);
        value.trace_gc(sink);
    }
}

impl GcTrace for crate::interpreter::FnFrame {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::FnFrame {
            fn_ptr,
            call_site: _,
            strict: _,
            construct: _,
            extra,
        } = self;
        // The callee's address: FnFrame::callee() turns it back into a handle.
        sink.word(*fn_ptr as u64);
        if let Some(extra) = extra {
            let crate::interpreter::FrameExtra { args_obj, lazy } = &**extra;
            args_obj.trace_gc(sink);
            if let Some((_, arguments, env)) = lazy {
                arguments.trace_gc(sink);
                env.trace_gc(sink);
            }
        }
    }
}

impl GcTrace for crate::interpreter::Disposable {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::Disposable {
            value,
            method,
            kind_is_async: _,
            method_is_async: _,
        } = self;
        value.trace_gc(sink);
        method.trace_gc(sink);
    }
}

impl GcTrace for crate::interpreter::Abrupt {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        match self {
            crate::interpreter::Abrupt::Throw(value)
            | crate::interpreter::Abrupt::Return(value)
            | crate::interpreter::Abrupt::Break(_, value)
            | crate::interpreter::Abrupt::Continue(_, value) => value.trace_gc(sink),
            crate::interpreter::Abrupt::Interrupt(_) => {}
        }
    }
}

impl GcTrace for crate::interpreter::AsyncWait {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::AsyncWait {
            promise,
            token: _,
            id: _,
            block: _,
            index: _,
            deadline: _,
        } = self;
        promise.trace_gc(sink);
    }
}

impl GcTrace for crate::interpreter::HostSettingsState {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::HostSettingsState {
            global_env,
            global_var_names: _,
        } = self;
        global_env.trace_gc(sink);
    }
}

impl GcTrace for crate::interpreter::ClassInfo {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::ClassInfo {
            fields,
            field_env,
            derived: _,
            instance_initializers,
            private_members,
        } = self;
        for field in fields.iter() {
            let crate::interpreter::FieldInit {
                key, // a computed key may be a Symbol, which is a cell
                init: _,
                transforms,
            } = field;
            key.trace_gc(sink);
            transforms.trace_gc(sink);
        }
        field_env.trace_gc(sink);
        instance_initializers.trace_gc(sink);
        for (_, property) in private_members {
            property.trace_gc(sink);
        }
    }
}

impl GcTrace for crate::interpreter::HostIndexedProperties {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::HostIndexedProperties {
            length: _,
            getter,
            live,
        } = self;
        getter.trace_gc(sink);
        if let Some(live) = live {
            let crate::interpreter::HostIndexedLive {
                state,
                length: _,
                names: _,
                named_getter,
            } = &**live;
            state.trace_gc(sink);
            named_getter.trace_gc(sink);
        }
    }
}

impl GcTrace for crate::interpreter::FinalizationState {
    /// The held values are the registry's strong edges; targets and tokens are weak.
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::interpreter::FinalizationState {
            cleanup_callback,
            cells,
            cleanup_scheduled: _,
        } = self;
        cleanup_callback.trace_gc(sink);
        for cell in cells.iter() {
            cell.held_value.trace_gc(sink);
        }
    }
}

impl GcTrace for crate::modules::NsBinding {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        match self {
            crate::modules::NsBinding::Live(env, _) => env.trace_gc(sink),
            crate::modules::NsBinding::Static(value) => value.trace_gc(sink),
        }
    }
}

impl GcTrace for crate::ordered_collection::CollectionIterator {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let crate::ordered_collection::CollectionIterator {
            target,
            cursor: _,
            kind: _,
            brand: _,
        } = self;
        target.trace_gc(sink);
    }
}

impl GcTrace for crate::coroutine::Resume {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        match self {
            crate::coroutine::Resume::Next(value)
            | crate::coroutine::Resume::Return(value)
            | crate::coroutine::Resume::Throw(value) => value.trace_gc(sink),
            crate::coroutine::Resume::Terminate => {}
        }
    }
}

impl GcTrace for crate::coroutine::Coroutine {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        match self {
            crate::coroutine::Coroutine::Vm(coroutine) => GcTrace::trace_gc(&**coroutine, sink),
            crate::coroutine::Coroutine::Module(coroutine) => GcTrace::trace_gc(&**coroutine, sink),
            crate::coroutine::Coroutine::FromAsync(coroutine) => {
                GcTrace::trace_gc(&**coroutine, sink)
            }
            crate::coroutine::Coroutine::Unavailable(coroutine) => {
                GcTrace::trace_gc(coroutine, sink)
            }
        }
    }
}
