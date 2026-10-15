//! P1 S2: precise tracing of the engine's Rust-side GC edges (feature `gc-heap`).
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
        if let Value::Obj(object) = value {
            self.object(object);
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
            gc_slot: _,
            gc_heap: _,
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
