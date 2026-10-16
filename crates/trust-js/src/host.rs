//! Embedder host state: typed per-subsystem storage plus an integer-keyed table of open
//! handles, reachable from native functions through their `&mut Interp` argument. This is how
//! a runtime layer (event loop, fs, timers) keeps Rust state despite `NativeFn` being a bare
//! `fn` pointer that cannot capture.
//!
//! Modeled on deno_core's `OpState` + `ResourceTable`.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::rc::Rc;

use crate::value::{RetainedManagedAllocation, Value};

/// Reports heap storage retained by an embedder-owned value.
///
/// The returned byte count is the requested size of allocations owned below `Self`; it excludes
/// the inline `size_of::<Self>()`, which TRust JS adds itself. Implementations should use container
/// capacities rather than lengths, must avoid crediting shared storage from more than one live
/// host entry, and must exclude backing separately emitted through [`RetainedExternalMemory`].
pub trait RetainedBytes {
    fn retained_bytes(&self) -> usize;
}

type RetainedReporter = fn(&dyn Any) -> usize;

/// Canonical sinks available to an embedder-owned value's retained-memory reporter.
pub trait HostRetainedMemoryVisitor {
    /// Report one identity-bearing allocation retained below the registered host value.
    fn allocation(&mut self, allocation: RetainedManagedAllocation);
    /// Report a JavaScript value retained by the host so its engine-owned graph is visited once.
    fn value(&mut self, value: &Value);
    /// Mark requested bytes as a documented lower bound because a standard-library container or
    /// channel does not expose its complete allocation layout.
    fn opaque_storage(&mut self);
    /// Mark one reachable owner as unavailable until the embedder can enumerate it. This keeps a
    /// partially migrated reporter from making the Agent snapshot look complete.
    fn unavailable(&mut self);
}

/// Identity-aware retained-memory reporting for embedder values that own shared allocations or
/// JavaScript roots. The registered value's inline payload is counted by [`OpState`]; the reporter
/// enumerates allocations and values below that payload.
pub trait RetainedMemory {
    fn scan_retained_memory(&self, visitor: &mut dyn HostRetainedMemoryVisitor);
}

/// Native caches can own JavaScript references without making them permanent roots.
/// Implementations must enumerate each owned strong handle exactly once with `internal`,
/// then describe the logical edges that keep it alive. Omitted handles remain ordinary
/// conservative roots, which is the safe fallback when a native store is currently borrowed.
/// These callbacks must not execute JavaScript, allocate JS objects, or re-enter collection.
pub trait HostGc {
    fn trace_gc(&self, visitor: &mut dyn HostGcVisitor);
    /// Release temporary native roots at the same synchronous-job boundary as ClearKeptObjects.
    /// This may update native bookkeeping only: no JavaScript, JS allocation or collection.
    /// Persistent async work must own explicit roots rather than relying on a job lease.
    fn end_job(&mut self) {}
    /// Remove every cache handle that `is_live` rejects before the JS heap is swept.
    /// Do not retain new handles or change the graph during this phase.
    fn sweep_gc(&mut self, is_live: &dyn Fn(&Value) -> bool);
    /// Joint native/JavaScript sweep. Native identities were traced in this collection, so only
    /// identities rejected by this callback may be reclaimed. Existing JS-only hosts need no
    /// changes; native graph users override this and leave the legacy sweep conservative.
    fn sweep_gc_with_native(
        &mut self,
        is_live: &dyn Fn(&Value) -> bool,
        _native_is_live: &dyn Fn(NativeGcId) -> bool,
    ) {
        self.sweep_gc(is_live);
    }
}

/// A native graph node, independent of JavaScript object allocation. The host owns the identity
/// domain and stable owner token; id must distinguish live resources and must not alias a retired
/// resource while references to its old identity remain possible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NativeGcId {
    pub domain: &'static str,
    pub owner: usize,
    pub id: usize,
}

pub trait HostGcVisitor {
    /// Nursery collections conservatively retain old JavaScript and native owners. A host
    /// may enumerate only its young and remembered state when it can prove all omitted
    /// handles/edges are old. The default keeps existing embedders on the full graph path.
    fn is_minor(&self) -> bool {
        false
    }
    /// Declare the old prefix of a host's monotonic native identity domain. During a minor
    /// collection every ID below `first_young_id` is conservatively live, including sources
    /// later reported by another host. Returning false requires the ordinary full graph.
    /// Do not clear journals or promote young state until the corresponding sweep succeeds.
    fn native_old_generation(
        &mut self,
        _domain: &'static str,
        _owner: usize,
        _first_young_id: usize,
    ) -> bool {
        false
    }
    /// Discount exactly one native-owned strong handle during root classification.
    fn internal(&mut self, value: &Value);
    /// A logical (non-Rc-owning) edge from a live owner to a retained value.
    fn edge(&mut self, owner: &Value, value: &Value);
    /// A value reachable from a native root not represented by a JS owner.
    fn root(&mut self, value: &Value);
    /// Old/third-party visitors can conservatively retain their native graph. A host must not
    /// discount the handles owned by that graph unless this visitor supports tracing it.
    fn supports_native(&self) -> bool {
        false
    }
    fn native_edge(&mut self, _from: NativeGcId, _to: NativeGcId) {}
    fn js_to_native(&mut self, _owner: &Value, _to: NativeGcId) {}
    fn native_to_js(&mut self, _from: NativeGcId, _value: &Value) {}
    fn native_root(&mut self, _id: NativeGcId) {}
}

#[derive(Clone, Copy)]
struct GcReporter {
    trace: fn(&dyn Any, &mut dyn HostGcVisitor),
    end_job: fn(&mut dyn Any),
    sweep_native: fn(&mut dyn Any, &dyn Fn(&Value) -> bool, &dyn Fn(NativeGcId) -> bool),
}

#[derive(Clone, Copy)]
struct ManagedReporter {
    scan: fn(&dyn Any, &mut dyn HostRetainedMemoryVisitor),
    inline_bytes: usize,
}

fn report_managed<T: Any + RetainedMemory>(
    value: &dyn Any,
    visitor: &mut dyn HostRetainedMemoryVisitor,
) {
    let value = value
        .downcast_ref::<T>()
        .expect("managed-memory reporter must match its registered host value type");
    value.scan_retained_memory(visitor);
}

fn managed_reporter<T: Any + RetainedMemory>() -> ManagedReporter {
    ManagedReporter {
        scan: report_managed::<T>,
        inline_bytes: std::mem::size_of::<T>(),
    }
}

/// An embedder allocation whose bytes belong in a managed external-memory category rather than
/// the host metadata total.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RetainedExternalAllocation {
    pub(crate) identity: RetainedExternalIdentity,
    pub(crate) bytes: usize,
    pub(crate) kind: RetainedExternalKind,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RetainedExternalIdentity {
    ArrayBufferBytes(usize),
    Embedder(&'static str, usize, u64),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum RetainedExternalKind {
    WasmMemory,
}

impl RetainedExternalAllocation {
    /// Report Wasm memory that is identified with TRust JS's `ArrayBufferBytes` storage.
    ///
    /// The allocation identity lets the post-GC visitor avoid crediting an exposed
    /// `WebAssembly.Memory.buffer` a second time as an ordinary ArrayBuffer backing store.
    pub fn wasm_array_buffer(storage: &crate::interpreter::ArrayBufferBytes) -> Self {
        Self {
            identity: RetainedExternalIdentity::ArrayBufferBytes(Rc::as_ptr(storage) as usize),
            bytes: storage.borrow().capacity(),
            kind: RetainedExternalKind::WasmMemory,
        }
    }

    /// Report Agent-local Wasm memory owned by another engine or allocator.
    ///
    /// `identity_domain` must be a stable, namespaced identifier for the embedder allocation
    /// family. `owner_identity` identifies the live store/allocator instance and `identity` must
    /// uniquely identify one live allocation within that owner.
    pub fn wasm_memory(
        identity_domain: &'static str,
        owner_identity: usize,
        identity: u64,
        requested_bytes: usize,
    ) -> Self {
        Self {
            identity: RetainedExternalIdentity::Embedder(identity_domain, owner_identity, identity),
            bytes: requested_bytes,
            kind: RetainedExternalKind::WasmMemory,
        }
    }
}

/// Reports external backing allocations retained by an embedder-owned value.
pub trait RetainedExternalMemory {
    fn retained_external_memory(&self, visit: &mut dyn FnMut(RetainedExternalAllocation));
}

type ExternalReporter = fn(&dyn Any, &mut dyn FnMut(RetainedExternalAllocation));

fn report_retained<T: Any + RetainedBytes>(value: &dyn Any) -> usize {
    let value = value
        .downcast_ref::<T>()
        .expect("retained-size reporter must match its registered host value type");
    std::mem::size_of::<T>().saturating_add(value.retained_bytes())
}

fn report_external<T: Any + RetainedExternalMemory>(
    value: &dyn Any,
    visit: &mut dyn FnMut(RetainedExternalAllocation),
) {
    let value = value
        .downcast_ref::<T>()
        .expect("external-memory reporter must match its registered host value type");
    value.retained_external_memory(visit);
}

#[derive(Clone, Debug, Default)]
pub(crate) struct HostRetainedMemory {
    pub(crate) reported_bytes: usize,
    pub(crate) unavailable_entries: usize,
    pub(crate) unclassified_external_entries: usize,
    pub(crate) identity_conflict: bool,
    pub(crate) opaque_storage: bool,
    pub(crate) external_allocations: Vec<RetainedExternalAllocation>,
}

impl HostRetainedMemory {
    pub(crate) fn add(&mut self, other: Self) {
        self.reported_bytes = self.reported_bytes.saturating_add(other.reported_bytes);
        self.unavailable_entries = self
            .unavailable_entries
            .saturating_add(other.unavailable_entries);
        self.unclassified_external_entries = self
            .unclassified_external_entries
            .saturating_add(other.unclassified_external_entries);
        self.identity_conflict |= other.identity_conflict;
        self.opaque_storage |= other.opaque_storage;
        self.external_allocations.extend(other.external_allocations);
    }
}

/// Typed host state: at most one value per Rust type, plus the [`ResourceTable`]. Op crates
/// each keep their state (timer heap, fd table, ...) under their own type.
#[derive(Default)]
pub struct OpState {
    /// Looked up on every host call that reaches typed state; `TypeId` keys are fixed program
    /// constants, so the default DoS-resistant SipHash only adds latency.
    map: crate::fasthash::FastMap<TypeId, Box<dyn Any>>,
    retained_reporters: HashMap<TypeId, RetainedReporter>,
    managed_reporters: HashMap<TypeId, ManagedReporter>,
    external_reporters: HashMap<TypeId, ExternalReporter>,
    gc_reporters: HashMap<TypeId, GcReporter>,
    pub resources: ResourceTable,
}

impl OpState {
    /// Install (or replace) the `T` slot.
    pub fn put<T: Any>(&mut self, value: T) {
        let type_id = TypeId::of::<T>();
        self.map.insert(type_id, Box::new(value));
        self.retained_reporters.remove(&type_id);
        self.managed_reporters.remove(&type_id);
        self.external_reporters.remove(&type_id);
    }
    /// Install a slot whose owned heap storage participates in retained-memory diagnostics and
    /// which retains no separately allocated external backing. Use
    /// [`Self::put_retained_with_external_memory`] when both kinds are present.
    pub fn put_retained<T: Any + RetainedBytes>(&mut self, value: T) {
        let type_id = TypeId::of::<T>();
        self.map.insert(type_id, Box::new(value));
        self.retained_reporters
            .insert(type_id, report_retained::<T>);
        self.managed_reporters.remove(&type_id);
        self.external_reporters.remove(&type_id);
    }
    /// Install a slot whose shared allocations and JavaScript roots participate in the
    /// identity-aware Agent-wide retained-memory visit and which owns no external backing.
    pub fn put_retained_memory<T: Any + RetainedMemory>(&mut self, value: T) {
        let type_id = TypeId::of::<T>();
        self.map.insert(type_id, Box::new(value));
        self.retained_reporters.remove(&type_id);
        self.managed_reporters
            .insert(type_id, managed_reporter::<T>());
        self.external_reporters.remove(&type_id);
    }
    /// Install a slot that reports external backing while leaving its host metadata unavailable.
    pub fn put_external_memory<T: Any + RetainedExternalMemory>(&mut self, value: T) {
        let type_id = TypeId::of::<T>();
        self.map.insert(type_id, Box::new(value));
        self.retained_reporters.remove(&type_id);
        self.managed_reporters.remove(&type_id);
        self.external_reporters
            .insert(type_id, report_external::<T>);
    }
    /// Install a slot that reports both host metadata and external backing.
    pub fn put_retained_with_external_memory<T: Any + RetainedBytes + RetainedExternalMemory>(
        &mut self,
        value: T,
    ) {
        let type_id = TypeId::of::<T>();
        self.map.insert(type_id, Box::new(value));
        self.retained_reporters
            .insert(type_id, report_retained::<T>);
        self.managed_reporters.remove(&type_id);
        self.external_reporters
            .insert(type_id, report_external::<T>);
    }
    /// Install a slot with identity-aware managed-memory and external-backing reporters.
    pub fn put_retained_memory_with_external_memory<
        T: Any + RetainedMemory + RetainedExternalMemory,
    >(
        &mut self,
        value: T,
    ) {
        let type_id = TypeId::of::<T>();
        self.map.insert(type_id, Box::new(value));
        self.retained_reporters.remove(&type_id);
        self.managed_reporters
            .insert(type_id, managed_reporter::<T>());
        self.external_reporters
            .insert(type_id, report_external::<T>);
    }
    pub fn get<T: Any>(&self) -> Option<&T> {
        self.map.get(&TypeId::of::<T>())?.downcast_ref()
    }
    pub fn get_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.map.get_mut(&TypeId::of::<T>())?.downcast_mut()
    }
    pub fn take<T: Any>(&mut self) -> Option<T> {
        let type_id = TypeId::of::<T>();
        let value = self.map.remove(&type_id)?;
        self.retained_reporters.remove(&type_id);
        self.managed_reporters.remove(&type_id);
        self.external_reporters.remove(&type_id);
        Some(*value.downcast().ok()?)
    }
    pub fn has<T: Any>(&self) -> bool {
        self.map.contains_key(&TypeId::of::<T>())
    }

    /// Opt an installed host type into cycle collection. Replacing a slot of the same
    /// Rust type preserves this registration; an absent slot is simply not visited.
    pub fn register_gc<T: Any + HostGc>(&mut self) {
        self.gc_reporters.insert(
            TypeId::of::<T>(),
            GcReporter {
                trace: |value, visitor| value.downcast_ref::<T>().unwrap().trace_gc(visitor),
                end_job: |value| value.downcast_mut::<T>().unwrap().end_job(),
                sweep_native: |value, is_live, native_is_live| {
                    value
                        .downcast_mut::<T>()
                        .unwrap()
                        .sweep_gc_with_native(is_live, native_is_live)
                },
            },
        );
    }

    pub(crate) fn trace_gc(&self, visitor: &mut dyn HostGcVisitor) {
        for (key, reporter) in &self.gc_reporters {
            if let Some(value) = self.map.get(key) {
                (reporter.trace)(value.as_ref(), visitor);
            }
        }
    }

    pub(crate) fn end_job(&mut self) {
        for (key, reporter) in &self.gc_reporters {
            if let Some(value) = self.map.get_mut(key) {
                (reporter.end_job)(value.as_mut());
            }
        }
    }

    pub(crate) fn sweep_gc_with_native(
        &mut self,
        is_live: &dyn Fn(&Value) -> bool,
        native_is_live: &dyn Fn(NativeGcId) -> bool,
    ) {
        for (key, reporter) in &self.gc_reporters {
            if let Some(value) = self.map.get_mut(key) {
                (reporter.sweep_native)(value.as_mut(), is_live, native_is_live);
            }
        }
    }

    pub(crate) fn retained_memory(&self) -> HostRetainedMemory {
        let mut memory = HostRetainedMemory {
            reported_bytes: self
                .map
                .len()
                .saturating_mul(std::mem::size_of::<(TypeId, Box<dyn Any>)>())
                .saturating_add(
                    self.retained_reporters
                        .len()
                        .saturating_mul(std::mem::size_of::<(TypeId, RetainedReporter)>()),
                )
                .saturating_add(
                    self.managed_reporters
                        .len()
                        .saturating_mul(std::mem::size_of::<(TypeId, ManagedReporter)>()),
                )
                .saturating_add(
                    self.external_reporters
                        .len()
                        .saturating_mul(std::mem::size_of::<(TypeId, ExternalReporter)>()),
                )
                .saturating_add(
                    self.gc_reporters
                        .len()
                        .saturating_mul(std::mem::size_of::<(TypeId, GcReporter)>()),
                ),
            unavailable_entries: self
                .map
                .len()
                .saturating_sub(self.retained_reporters.len() + self.managed_reporters.len()),
            unclassified_external_entries: self
                .map
                .keys()
                .filter(|type_id| {
                    !self.retained_reporters.contains_key(type_id)
                        && !self.managed_reporters.contains_key(type_id)
                        && !self.external_reporters.contains_key(type_id)
                })
                .count(),
            identity_conflict: false,
            opaque_storage: !self.map.is_empty()
                || !self.retained_reporters.is_empty()
                || !self.managed_reporters.is_empty()
                || !self.external_reporters.is_empty()
                || !self.gc_reporters.is_empty(),
            external_allocations: Vec::new(),
        };
        for (type_id, reporter) in &self.retained_reporters {
            if let Some(value) = self.map.get(type_id) {
                memory.reported_bytes = memory
                    .reported_bytes
                    .saturating_add(reporter(value.as_ref()));
            }
        }
        for reporter in self.managed_reporters.values() {
            memory.reported_bytes = memory.reported_bytes.saturating_add(reporter.inline_bytes);
        }
        for (type_id, reporter) in &self.external_reporters {
            if let Some(value) = self.map.get(type_id) {
                reporter(value.as_ref(), &mut |allocation| {
                    memory.external_allocations.push(allocation);
                });
            }
        }
        memory.add(self.resources.retained_memory());
        memory
    }

    pub(crate) fn scan_retained_memory(&self, visitor: &mut dyn HostRetainedMemoryVisitor) {
        for (type_id, reporter) in &self.managed_reporters {
            if let Some(value) = self.map.get(type_id) {
                (reporter.scan)(value.as_ref(), visitor);
            }
        }
        self.resources.scan_retained_memory(visitor);
    }
}

/// A resource id, the JS-visible handle to an entry in the [`ResourceTable`] (an fd number,
/// in effect).
pub type ResourceId = u32;

/// Open handles (files, sockets, streams): `Rc<dyn Any>` keyed by a small integer that JS code
/// holds. Ids are never reused within a table's lifetime, so a stale id after `close` is a
/// lookup miss, not a use-after-free of a recycled slot.
#[derive(Default)]
pub struct ResourceTable {
    next: ResourceId,
    map: HashMap<ResourceId, Rc<dyn Any>>,
    retained_reporters: HashMap<ResourceId, RetainedReporter>,
    managed_reporters: HashMap<ResourceId, ManagedReporter>,
    external_reporters: HashMap<ResourceId, ExternalReporter>,
}

impl ResourceTable {
    /// Store `resource` under a fresh id. Ids are never reused, so once ids 0..u32::MAX have been
    /// handed out the id space is exhausted and this panics rather than recycle an id a script
    /// may still hold.
    pub fn add<T: Any>(&mut self, resource: T) -> ResourceId {
        let rid = self.next;
        self.next = rid
            .checked_add(1)
            .expect("ResourceTable id space exhausted; ids are never reused");
        self.map.insert(rid, Rc::new(resource));
        rid
    }
    /// Add a resource whose owned heap storage participates in retained-memory diagnostics and
    /// which retains no separately allocated external backing. Use
    /// [`Self::add_retained_with_external_memory`] when both kinds are present.
    pub fn add_retained<T: Any + RetainedBytes>(&mut self, resource: T) -> ResourceId {
        let rid = self.add(resource);
        self.retained_reporters.insert(rid, report_retained::<T>);
        rid
    }
    /// Add a resource whose shared allocations and JavaScript roots participate in the
    /// identity-aware Agent-wide retained-memory visit.
    pub fn add_retained_memory<T: Any + RetainedMemory>(&mut self, resource: T) -> ResourceId {
        let rid = self.add(resource);
        self.managed_reporters.insert(rid, managed_reporter::<T>());
        rid
    }
    /// Add a resource that reports external backing while leaving its host metadata unavailable.
    pub fn add_external_memory<T: Any + RetainedExternalMemory>(
        &mut self,
        resource: T,
    ) -> ResourceId {
        let rid = self.add(resource);
        self.external_reporters.insert(rid, report_external::<T>);
        rid
    }
    /// Add a resource that reports both host metadata and external backing.
    pub fn add_retained_with_external_memory<T: Any + RetainedBytes + RetainedExternalMemory>(
        &mut self,
        resource: T,
    ) -> ResourceId {
        let rid = self.add_retained(resource);
        self.external_reporters.insert(rid, report_external::<T>);
        rid
    }
    /// Add a resource with identity-aware managed-memory and external-backing reporters.
    pub fn add_retained_memory_with_external_memory<
        T: Any + RetainedMemory + RetainedExternalMemory,
    >(
        &mut self,
        resource: T,
    ) -> ResourceId {
        let rid = self.add_retained_memory(resource);
        self.external_reporters.insert(rid, report_external::<T>);
        rid
    }
    pub fn get<T: Any>(&self, rid: ResourceId) -> Option<Rc<T>> {
        Rc::downcast(self.map.get(&rid)?.clone()).ok()
    }
    pub fn has(&self, rid: ResourceId) -> bool {
        self.map.contains_key(&rid)
    }
    /// Remove and return the handle; the resource drops (and e.g. the file closes) when the
    /// last `Rc` clone does.
    pub fn close(&mut self, rid: ResourceId) -> Option<Rc<dyn Any>> {
        let value = self.map.remove(&rid);
        self.retained_reporters.remove(&rid);
        self.managed_reporters.remove(&rid);
        self.external_reporters.remove(&rid);
        value
    }
    /// Live handle count — the event loop stays alive while this is non-zero.
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn retained_memory(&self) -> HostRetainedMemory {
        let mut memory = HostRetainedMemory {
            reported_bytes: self
                .map
                .len()
                .saturating_mul(std::mem::size_of::<(ResourceId, Rc<dyn Any>)>())
                .saturating_add(
                    self.retained_reporters
                        .len()
                        .saturating_mul(std::mem::size_of::<(ResourceId, RetainedReporter)>()),
                )
                .saturating_add(
                    self.managed_reporters
                        .len()
                        .saturating_mul(std::mem::size_of::<(ResourceId, ManagedReporter)>()),
                )
                .saturating_add(
                    self.external_reporters
                        .len()
                        .saturating_mul(std::mem::size_of::<(ResourceId, ExternalReporter)>()),
                ),
            unavailable_entries: self
                .map
                .len()
                .saturating_sub(self.retained_reporters.len() + self.managed_reporters.len()),
            unclassified_external_entries: self
                .map
                .keys()
                .filter(|rid| {
                    !self.retained_reporters.contains_key(rid)
                        && !self.managed_reporters.contains_key(rid)
                        && !self.external_reporters.contains_key(rid)
                })
                .count(),
            identity_conflict: false,
            opaque_storage: !self.map.is_empty()
                || !self.retained_reporters.is_empty()
                || !self.managed_reporters.is_empty()
                || !self.external_reporters.is_empty(),
            external_allocations: Vec::new(),
        };
        for (rid, reporter) in &self.retained_reporters {
            if let Some(value) = self.map.get(rid) {
                memory.reported_bytes = memory
                    .reported_bytes
                    .saturating_add(reporter(value.as_ref()));
            }
        }
        for reporter in self.managed_reporters.values() {
            memory.reported_bytes = memory.reported_bytes.saturating_add(reporter.inline_bytes);
        }
        for (rid, reporter) in &self.external_reporters {
            if let Some(value) = self.map.get(rid) {
                reporter(value.as_ref(), &mut |allocation| {
                    memory.external_allocations.push(allocation);
                });
            }
        }
        memory
    }

    fn scan_retained_memory(&self, visitor: &mut dyn HostRetainedMemoryVisitor) {
        for (rid, reporter) in &self.managed_reporters {
            if let Some(value) = self.map.get(rid) {
                (reporter.scan)(value.as_ref(), visitor);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct NativeCache {
        values: Vec<Value>,
        edges: Vec<(usize, usize)>,
        roots: Vec<usize>,
        busy: bool,
    }

    impl HostGc for NativeCache {
        fn trace_gc(&self, visitor: &mut dyn HostGcVisitor) {
            if self.busy {
                return;
            }
            for value in &self.values {
                visitor.internal(value);
            }
            for &(owner, target) in &self.edges {
                visitor.edge(&self.values[owner], &self.values[target]);
            }
            for &root in &self.roots {
                visitor.root(&self.values[root]);
            }
        }
        fn sweep_gc(&mut self, is_live: &dyn Fn(&Value) -> bool) {
            // Keep indices stable in this synthetic cache.
            for value in &mut self.values {
                if !is_live(value) {
                    *value = Value::Undefined;
                }
            }
        }
    }

    #[test]
    fn host_gc_edges_preserve_ephemerons_and_release_unrooted_cycles() {
        let mut engine = crate::Engine::new();
        engine
            .eval(
                r#"
            globalThis.owner = {};
            globalThis.target = {};
            globalThis.payload = {marker: 42};
            const map = new WeakMap([[target, payload]]);
            payload.back = target;
        "#,
                false,
            )
            .unwrap();
        // Strong handles live only in this helper frame (gc-heap scans the stack).
        let (target_weak, payload_weak) = crate::gc_test::in_frame(|| {
            let global = Value::Obj(engine.interp.global.clone());
            let owner = engine
                .interp
                .member_get(&global, "owner")
                .unwrap_or_else(|_| panic!("owner"));
            let target = engine
                .interp
                .member_get(&global, "target")
                .unwrap_or_else(|_| panic!("target"));
            let payload = engine
                .interp
                .member_get(&global, "payload")
                .unwrap_or_else(|_| panic!("payload"));
            let target_weak = engine.interp.downgrade_object_value(&target).unwrap();
            let payload_weak = engine.interp.downgrade_object_value(&payload).unwrap();
            engine.interp.op_state().put(NativeCache {
                // Two cache handles to one target must both be discounted, but logical edges
                // must not be counted as additional strong owners.
                values: vec![owner, target.clone(), target],
                edges: vec![(0, 1), (1, 0)],
                ..Default::default()
            });
            (target_weak, payload_weak)
        });
        engine.interp.op_state().register_gc::<NativeCache>();
        engine.eval("target = payload = null", false).unwrap();
        engine.interp.gc_collect();
        assert!(crate::gc_test::alive(&target_weak));
        assert!(
            crate::gc_test::alive(&payload_weak),
            "native edge activates WeakMap value"
        );

        engine.eval("owner = null", false).unwrap();
        engine
            .interp
            .op_state()
            .get_mut::<NativeCache>()
            .unwrap()
            .roots
            .push(1);
        engine.interp.gc_collect();
        assert!(
            crate::gc_test::alive(&payload_weak),
            "native root activates entire component"
        );
        engine
            .interp
            .op_state()
            .get_mut::<NativeCache>()
            .unwrap()
            .roots
            .clear();
        engine.interp.gc_collect();
        assert!(!crate::gc_test::alive(&target_weak));
        assert!(!crate::gc_test::alive(&payload_weak));
        assert!(engine
            .interp
            .op_state()
            .get::<NativeCache>()
            .unwrap()
            .values
            .iter()
            .all(|value| matches!(value, Value::Undefined)));
    }

    #[test]
    #[cfg_attr(
        feature = "gc-heap",
        ignore = "P1 S2: a host omitting its handles keeps no roots under gc-heap (mandatory enumeration)"
    )]
    fn host_gc_busy_fallback_and_replaced_slots_remain_safe() {
        let mut engine = crate::Engine::new();
        engine.interp.op_state().register_gc::<NativeCache>();
        // A registration is attached to a Rust type, not to the old slot's address.
        engine.interp.op_state().put(NativeCache::default());
        engine.interp.op_state().take::<NativeCache>();
        engine.interp.gc_collect();
        let value = Value::Obj(engine.interp.new_object());
        let weak = engine.interp.downgrade_object_value(&value).unwrap();
        engine.interp.op_state().put(NativeCache {
            values: vec![value],
            busy: true,
            ..Default::default()
        });
        engine.interp.gc_collect();
        assert!(
            crate::gc_test::alive(&weak),
            "undiscounted handle remains a root"
        );
        engine
            .interp
            .op_state()
            .get_mut::<NativeCache>()
            .unwrap()
            .busy = false;
        engine.interp.gc_collect();
        assert!(
            !crate::gc_test::alive(&weak),
            "idle, unrooted cache entry is released"
        );
    }

    #[test]
    #[cfg_attr(
        feature = "gc-heap",
        ignore = "P1 S2: a host omitting its handles keeps no roots under gc-heap (mandatory enumeration)"
    )]
    fn host_gc_and_continuations_share_an_owner_without_losing_either_edge() {
        let mut engine = crate::Engine::new();
        engine
            .eval(
                r#"
            var mixedOwner = (function*() {
                const payload = {n:41};
                yield 0; yield payload.n;
            })();
            mixedOwner.next();
        "#,
                false,
            )
            .unwrap();
        let global = Value::Obj(engine.interp.global.clone());
        let owner = engine
            .interp
            .member_get(&global, "mixedOwner")
            .unwrap_or_else(|_| panic!("generator owner"));
        let owner_weak = engine.interp.downgrade_object_value(&owner).unwrap();
        let native_target = Value::Obj(engine.interp.new_object());
        let native_weak = engine
            .interp
            .downgrade_object_value(&native_target)
            .unwrap();
        engine.interp.op_state().put(NativeCache {
            values: vec![owner, native_target],
            edges: vec![(0, 1)],
            ..Default::default()
        });
        engine.interp.op_state().register_gc::<NativeCache>();
        engine.interp.gc_collect();
        assert!(
            crate::gc_test::alive(&native_weak),
            "the native edge shares the coroutine owner"
        );
        match engine.eval("mixedOwner.next().value", false).unwrap() {
            crate::Completion::Value(value) => assert_eq!(value, "41"),
            crate::Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
        engine.eval("mixedOwner = null", false).unwrap();
        engine
            .interp
            .op_state()
            .get_mut::<NativeCache>()
            .unwrap()
            .busy = true;
        engine.interp.gc_collect();
        assert!(crate::gc_test::alive(&owner_weak));
        assert!(crate::gc_test::alive(&native_weak));
        engine
            .interp
            .op_state()
            .get_mut::<NativeCache>()
            .unwrap()
            .busy = false;
        engine.interp.gc_collect();
        assert!(!crate::gc_test::alive(&owner_weak));
        assert!(!crate::gc_test::alive(&native_weak));
        assert!(engine.interp.generators.is_empty());
    }

    struct Reported(Vec<u8>);

    struct External(crate::interpreter::ArrayBufferBytes);

    impl RetainedBytes for Reported {
        fn retained_bytes(&self) -> usize {
            self.0.capacity()
        }
    }

    impl RetainedExternalMemory for External {
        fn retained_external_memory(&self, visit: &mut dyn FnMut(RetainedExternalAllocation)) {
            visit(RetainedExternalAllocation::wasm_array_buffer(&self.0));
        }
    }

    #[test]
    fn retained_reporters_follow_state_and_resource_lifetimes() {
        let mut state = OpState::default();
        state.put_retained(Reported(Vec::with_capacity(41)));
        let rid = state
            .resources
            .add_retained(Reported(Vec::with_capacity(73)));

        let reported = state.retained_memory();
        assert_eq!(reported.unavailable_entries, 0);
        assert_eq!(reported.unclassified_external_entries, 0);
        assert!(reported.reported_bytes >= 41 + 73 + 2 * std::mem::size_of::<Reported>());
        assert!(reported.opaque_storage);

        state.put(String::from("not reported"));
        assert_eq!(state.retained_memory().unavailable_entries, 1);
        assert_eq!(state.retained_memory().unclassified_external_entries, 1);
        state.take::<String>();
        state.resources.close(rid);
        assert_eq!(state.retained_memory().unavailable_entries, 0);
    }

    #[test]
    fn replacing_reported_state_with_legacy_state_removes_the_reporter() {
        let mut state = OpState::default();
        state.put_retained(Reported(Vec::with_capacity(19)));
        assert_eq!(state.retained_memory().unavailable_entries, 0);

        state.put(Reported(Vec::new()));
        assert_eq!(state.retained_memory().unavailable_entries, 1);
    }

    #[test]
    fn external_reporters_are_independent_from_host_metadata_completeness() {
        let storage = Rc::new(std::cell::RefCell::new(Vec::with_capacity(61)));
        let mut state = OpState::default();
        state.put_external_memory(External(storage.clone()));
        let rid = state.resources.add_external_memory(External(storage));

        let memory = state.retained_memory();
        assert_eq!(memory.unavailable_entries, 2);
        assert_eq!(memory.unclassified_external_entries, 0);
        assert_eq!(memory.external_allocations.len(), 2);
        assert_eq!(
            memory.external_allocations[0],
            memory.external_allocations[1]
        );

        state.take::<External>();
        state.resources.close(rid);
        assert!(state.retained_memory().external_allocations.is_empty());
    }
}

#[cfg(test)]
mod resource_id_tests {
    use super::ResourceTable;

    #[test]
    #[should_panic(expected = "ids are never reused")]
    fn exhausted_resource_ids_are_not_recycled() {
        let mut table = ResourceTable {
            next: u32::MAX - 1,
            ..ResourceTable::default()
        };
        assert_eq!(table.add(()), u32::MAX - 1);
        // Ids 0..=u32::MAX - 1 are all spent: the next addition refuses to wrap.
        table.add(());
    }
}
