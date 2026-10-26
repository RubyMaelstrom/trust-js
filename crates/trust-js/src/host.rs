//! Embedder host state: typed per-subsystem storage plus an integer-keyed table of open
//! handles, reachable from native functions through their context (`embed::Ctx::op_state`).
//! This is how a runtime layer (event loop, fs, timers) keeps Rust state despite `NativeFn`
//! being a bare `fn` pointer that cannot capture.
//!
//! Modeled on deno_core's `OpState` + `ResourceTable`.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::rc::Rc;

#[cfg(feature = "embed")]
use crate::handles::WeakHandle;
use crate::value::RetainedManagedAllocation;

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

/// Canonical sinks available to an embedder-owned value's retained-memory reporter. The
/// JavaScript values a host keeps are its handles, which the memory snapshot counts itself.
pub trait HostRetainedMemoryVisitor {
    /// Report one identity-bearing allocation retained below the registered host value.
    fn allocation(&mut self, allocation: RetainedManagedAllocation);
    /// Mark requested bytes as a documented lower bound because a standard-library container or
    /// channel does not expose its complete allocation layout.
    fn opaque_storage(&mut self);
    /// Mark one reachable owner as unavailable until the embedder can enumerate it. This keeps a
    /// partially migrated reporter from making the Agent snapshot look complete.
    fn unavailable(&mut self);
}

/// Identity-aware retained-memory reporting for embedder values that own shared allocations. The
/// registered value's inline payload is counted by [`OpState`]; the reporter enumerates the
/// allocations below that payload.
pub trait RetainedMemory {
    fn scan_retained_memory(&self, visitor: &mut dyn HostRetainedMemoryVisitor);
}

/// A host value that reports a graph of JavaScript values to the collector (P1 S2, Decision 2),
/// for hosts whose values must live exactly as long as their JavaScript owners do: a value
/// reported with `edge` stays alive while its owner does, and a native graph ties JavaScript
/// values to native identities. The host names each value by a
/// [`WeakHandle`](crate::embed::WeakHandle), which keeps nothing by itself: what keeps the
/// value is the report, made again at every collection. A value the host simply keeps needs
/// none of this: it keeps the value in a [`Handle`](crate::embed::Handle) (S6 step 11a),
/// wherever it is stored. A type is reported at every collection, minor or major, once it is
/// registered with [`OpState::register_trace`] or [`ResourceTable::register_trace`] (or, as a
/// [`HostGc`], with [`OpState::register_gc`]). A minor collection's report may leave out what
/// is old ([`HostGcVisitor::is_minor`]). These callbacks must not execute JavaScript,
/// allocate JS objects, or re-enter collection.
pub trait HostTrace {
    fn trace_gc(&self, visitor: &mut dyn HostGcVisitor);
}

/// A host graph with native caches: beside its [`HostTrace`] report, it releases job-scoped
/// state at each job's end and sweeps its caches once a collection has decided what died.
/// Register with [`OpState::register_gc`], which also registers the type's [`HostTrace`].
/// These callbacks must not execute JavaScript, allocate JS objects, or re-enter collection.
pub trait HostGc: HostTrace {
    /// Release temporary native roots at the end of the outer host checkpoint, alongside
    /// ClearKeptObjects. A successful partial one-job pump retains these leases until its
    /// empty result or a later full checkpoint; nested same-Agent requests do not end it.
    /// This may update native bookkeeping only: no JavaScript, JS allocation or collection.
    /// Persistent async work must own explicit roots rather than relying on a job lease.
    fn end_job(&mut self) {}
    /// Sweep the host's caches between a collection's phases, after the weak handles whose
    /// values died have been cleared ([`WeakHandle::is_cleared`](crate::embed::WeakHandle::is_cleared))
    /// and before any cell is freed: drop what names a dead value. Native identities were traced
    /// in this collection, so only identities `native_is_live` rejects may be reclaimed. Do not
    /// make handles or change the graph during this phase.
    fn sweep_gc_with_native(&mut self, native_is_live: &dyn Fn(NativeGcId) -> bool);
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

/// The collector's side of a [`HostTrace`] report. Its JavaScript ends are weak handles of the
/// collecting Engine (another Engine's, or a cleared one, report nothing). A major collection
/// takes the host's whole graph; a minor one may take only its young part (S6 step 11c).
pub trait HostGcVisitor {
    /// Whether this collection is minor: it frees only values allocated since the last
    /// collection, and no native identity the host declares old
    /// ([`HostGcVisitor::native_old_generation`]). A host may then report only what it began to
    /// hold since the last collection, as long as every value it leaves out is one an earlier
    /// collection kept, which a minor collection keeps whatever the report says. A host that
    /// reports its whole graph is right in every collection.
    fn is_minor(&self) -> bool {
        false
    }
    /// In a minor collection, declare the identities of `domain` and `owner` below
    /// `first_young_id` old: the host numbers an owner's identities in allocation order and
    /// counts them all old once a collection's sweep has seen them. The collection keeps the old
    /// identities live, reported or not, and traces the edges reported from them, so the host
    /// reports, of that owner's graph, its young identities' edges and roots and the edges of
    /// each old identity whose edges changed since the last collection. Returns false in a major
    /// collection, which takes the whole graph.
    fn native_old_generation(
        &mut self,
        _domain: &'static str,
        _owner: usize,
        _first_young_id: usize,
    ) -> bool {
        false
    }
    /// A logical edge from a live owner to a retained value: `value` stays alive while `owner`
    /// does.
    #[cfg(feature = "embed")]
    fn edge(&mut self, owner: &WeakHandle, value: &WeakHandle);
    /// A value reachable from a native root not represented by a JS owner.
    #[cfg(feature = "embed")]
    fn root(&mut self, value: &WeakHandle);
    /// An edge between two native identities.
    fn native_edge(&mut self, from: NativeGcId, to: NativeGcId);
    /// An edge from a JavaScript owner (an object) to a native identity.
    #[cfg(feature = "embed")]
    fn js_to_native(&mut self, owner: &WeakHandle, to: NativeGcId);
    /// An edge from a native identity to a JavaScript value.
    #[cfg(feature = "embed")]
    fn native_to_js(&mut self, from: NativeGcId, value: &WeakHandle);
    /// A native identity that is a root.
    fn native_root(&mut self, id: NativeGcId);
}

/// A registered host type's [`HostTrace`].
type HostTracer = fn(&dyn Any, &mut dyn HostGcVisitor);

fn host_tracer<T: Any + HostTrace>(value: &dyn Any, visitor: &mut dyn HostGcVisitor) {
    value
        .downcast_ref::<T>()
        .expect("host tracer must match its stored value type")
        .trace_gc(visitor);
}

#[derive(Clone, Copy)]
struct GcReporter {
    end_job: fn(&mut dyn Any),
    sweep_native: fn(&mut dyn Any, &dyn Fn(NativeGcId) -> bool),
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
    /// The [`HostTrace`] of each type registered for collections' reports
    /// ([`OpState::register_trace`], [`OpState::register_gc`]).
    tracers: crate::fasthash::FastMap<TypeId, HostTracer>,
    pub resources: ResourceTable,
}

impl OpState {
    /// Store `value` in the `T` slot.
    fn insert<T: Any>(&mut self, value: T) -> TypeId {
        let type_id = TypeId::of::<T>();
        self.map.insert(type_id, Box::new(value));
        type_id
    }
    /// Install (or replace) the `T` slot. Its JavaScript values, if any, are kept in
    /// [`Handle`](crate::embed::Handle)s, or reported as a graph by a type registered with
    /// [`OpState::register_trace`].
    pub fn put<T: Any>(&mut self, value: T) {
        let type_id = self.insert(value);
        self.retained_reporters.remove(&type_id);
        self.managed_reporters.remove(&type_id);
        self.external_reporters.remove(&type_id);
    }
    /// Install a slot whose owned heap storage participates in retained-memory diagnostics and
    /// which retains no separately allocated external backing. Use
    /// [`Self::put_retained_with_external_memory`] when both kinds are present.
    pub fn put_retained<T: Any + RetainedBytes>(&mut self, value: T) {
        let type_id = self.insert(value);
        self.retained_reporters
            .insert(type_id, report_retained::<T>);
        self.managed_reporters.remove(&type_id);
        self.external_reporters.remove(&type_id);
    }
    /// Install a slot whose shared allocations and JavaScript roots participate in the
    /// identity-aware Agent-wide retained-memory visit and which owns no external backing.
    pub fn put_retained_memory<T: Any + RetainedMemory>(&mut self, value: T) {
        let type_id = self.insert(value);
        self.retained_reporters.remove(&type_id);
        self.managed_reporters
            .insert(type_id, managed_reporter::<T>());
        self.external_reporters.remove(&type_id);
    }
    /// Install a slot that reports external backing while leaving its host metadata unavailable.
    pub fn put_external_memory<T: Any + RetainedExternalMemory>(&mut self, value: T) {
        let type_id = self.insert(value);
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
        let type_id = self.insert(value);
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
        let type_id = self.insert(value);
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

    /// Opt a host type into the collector's [`HostTrace`] reports: from now on every collection
    /// asks the `T` slot, while there is one, for its graph. The registration belongs to the Rust
    /// type, not to a slot: replacing or taking the slot keeps it, and an absent slot is simply
    /// not visited.
    pub fn register_trace<T: Any + HostTrace>(&mut self) {
        self.tracers.insert(TypeId::of::<T>(), host_tracer::<T>);
    }

    /// Opt a host type into the collector's [`HostGc`] callbacks (job end and sweep) and, as a
    /// [`HostTrace`], its reports ([`OpState::register_trace`]). Replacing a slot of the same Rust
    /// type preserves this registration; an absent slot is simply not visited.
    pub fn register_gc<T: Any + HostGc>(&mut self) {
        self.register_trace::<T>();
        self.gc_reporters.insert(
            TypeId::of::<T>(),
            GcReporter {
                end_job: |value| value.downcast_mut::<T>().unwrap().end_job(),
                sweep_native: |value, native_is_live| {
                    value
                        .downcast_mut::<T>()
                        .unwrap()
                        .sweep_gc_with_native(native_is_live)
                },
            },
        );
    }

    /// Report the host's JavaScript graphs: the [`HostTrace`] of every slot and resource whose
    /// type is registered.
    pub(crate) fn trace_gc(&self, visitor: &mut dyn HostGcVisitor) {
        for (key, tracer) in &self.tracers {
            if let Some(value) = self.map.get(key) {
                tracer(value.as_ref(), visitor);
            }
        }
        self.resources.trace_gc(visitor);
    }

    pub(crate) fn end_job(&mut self) {
        for (key, reporter) in &self.gc_reporters {
            if let Some(value) = self.map.get_mut(key) {
                (reporter.end_job)(value.as_mut());
            }
        }
    }

    pub(crate) fn sweep_gc_with_native(&mut self, native_is_live: &dyn Fn(NativeGcId) -> bool) {
        for (key, reporter) in &self.gc_reporters {
            if let Some(value) = self.map.get_mut(key) {
                (reporter.sweep_native)(value.as_mut(), native_is_live);
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
    /// The [`HostTrace`] of each resource type registered for collections' reports
    /// ([`ResourceTable::register_trace`]).
    tracers: HashMap<TypeId, HostTracer>,
}

impl ResourceTable {
    /// Store `resource` under a fresh id. Ids are never reused, so once ids 0..u32::MAX have been
    /// handed out the id space is exhausted and this panics rather than recycle an id a script
    /// may still hold. Its JavaScript values, if any, are kept in
    /// [`Handle`](crate::embed::Handle)s, or reported as a graph by a type registered with
    /// [`ResourceTable::register_trace`].
    pub fn add<T: Any>(&mut self, resource: T) -> ResourceId {
        let rid = self.next;
        self.next = rid
            .checked_add(1)
            .expect("ResourceTable id space exhausted; ids are never reused");
        self.map.insert(rid, Rc::new(resource));
        rid
    }
    /// Opt a resource type into the collector's [`HostTrace`] reports: every collection asks
    /// each open resource of type `T` for its graph.
    pub fn register_trace<T: Any + HostTrace>(&mut self) {
        self.tracers.insert(TypeId::of::<T>(), host_tracer::<T>);
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
    /// Report the JavaScript graph of every open resource whose type is registered.
    fn trace_gc(&self, visitor: &mut dyn HostGcVisitor) {
        if self.tracers.is_empty() {
            return;
        }
        for resource in self.map.values() {
            if let Some(tracer) = self.tracers.get(&(**resource).type_id()) {
                tracer(resource.as_ref(), visitor);
            }
        }
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
    #[cfg(feature = "embed")]
    use crate::handles::WeakHandle;
    #[cfg(feature = "embed")]
    use crate::scope::Value;

    /// A host graph reported through weak handles (S6 step 11b, `embed`): the report, made at
    /// every collection, keeps the values; the weak handles keep nothing and are cleared once
    /// their values die.
    #[cfg(feature = "embed")]
    #[derive(Default)]
    struct NativeCache {
        values: Vec<WeakHandle>,
        edges: Vec<(usize, usize)>,
        roots: Vec<usize>,
        /// What the last sweep saw cleared, per value.
        swept: Vec<bool>,
    }

    #[cfg(feature = "embed")]
    impl HostTrace for NativeCache {
        fn trace_gc(&self, visitor: &mut dyn HostGcVisitor) {
            for &(owner, target) in &self.edges {
                visitor.edge(&self.values[owner], &self.values[target]);
            }
            for &root in &self.roots {
                visitor.root(&self.values[root]);
            }
        }
    }

    #[cfg(feature = "embed")]
    impl HostGc for NativeCache {
        fn sweep_gc_with_native(&mut self, _: &dyn Fn(NativeGcId) -> bool) {
            // The weak handles of the dead are cleared before the sweep.
            self.swept = self.values.iter().map(WeakHandle::is_cleared).collect();
        }
    }

    /// Weak handles to `names`, globals of `engine`, and internal weak references for watching
    /// them die, made in a frame of their own.
    #[cfg(feature = "embed")]
    fn weak_globals(
        engine: &mut crate::Engine,
        names: &[&str],
    ) -> (Vec<WeakHandle>, Vec<crate::interpreter::WeakValue>) {
        crate::gc_test::in_frame(|| {
            engine.scope(|cx| {
                let global = cx.global_this();
                names
                    .iter()
                    .map(|name| {
                        let value = cx
                            .member_get(&global, name)
                            .unwrap_or_else(|_| panic!("global {name}"));
                        (
                            WeakHandle::new(cx, &value).expect("a weak handle"),
                            cx.interp()
                                .downgrade_object_value(&value.into_internal())
                                .expect("an object"),
                        )
                    })
                    .unzip()
            })
        })
    }

    #[test]
    #[cfg(feature = "embed")]
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
        let (handles, weak) = weak_globals(&mut engine, &["owner", "target", "payload"]);
        let [_, target_weak, payload_weak] = &weak[..] else {
            unreachable!()
        };
        let [owner, target, _] = &handles[..] else {
            unreachable!()
        };
        engine.op_state().put(NativeCache {
            // Two handles to one target, and edges in both directions.
            values: vec![owner.clone(), target.clone(), target.clone()],
            edges: vec![(0, 1), (1, 0)],
            ..Default::default()
        });
        engine.op_state().register_gc::<NativeCache>();
        engine.eval("target = payload = null", false).unwrap();
        engine.interp.gc_collect();
        assert!(crate::gc_test::alive(target_weak));
        assert!(
            crate::gc_test::alive(payload_weak),
            "native edge activates WeakMap value"
        );

        engine.eval("owner = null", false).unwrap();
        engine.host_mut::<NativeCache>().unwrap().roots.push(1);
        engine.interp.gc_collect();
        assert!(
            crate::gc_test::alive(payload_weak),
            "native root activates entire component"
        );
        engine.host_mut::<NativeCache>().unwrap().roots.clear();
        engine.interp.gc_collect();
        assert!(!crate::gc_test::alive(target_weak));
        assert!(!crate::gc_test::alive(payload_weak));
        let cache = engine.host::<NativeCache>().unwrap();
        assert!(cache.values.iter().all(WeakHandle::is_cleared));
        assert_eq!(cache.swept, [true; 3], "the sweep saw them cleared");
    }

    /// A GC registration belongs to its Rust type, not to the slot it was made beside: a
    /// collection skips the taken slot and sweeps a later slot of the type.
    #[test]
    #[cfg(feature = "embed")]
    fn host_gc_registration_outlives_a_taken_slot() {
        let mut engine = crate::Engine::new();
        engine.op_state().register_gc::<NativeCache>();
        engine.op_state().put(NativeCache::default());
        engine.op_state().take::<NativeCache>();
        engine.interp.gc_collect();
        // The value lives only in this helper frame (the collector scans the stack).
        crate::gc_test::in_frame(|| {
            let value = engine.scope(|cx| {
                WeakHandle::new(cx, &Value::Obj(cx.new_object())).expect("a weak handle")
            });
            engine.op_state().put(NativeCache {
                values: vec![value],
                ..Default::default()
            });
        });
        engine.interp.gc_collect();
        assert_eq!(
            engine.host::<NativeCache>().unwrap().swept,
            [true],
            "the later slot's unreported value is swept"
        );
    }

    /// A host edge from a suspended generator shares its owner with the generator's
    /// continuation: while the generator lives, both its captured payload and the host's target
    /// do, and once it is unreachable a collection frees both and its continuation.
    #[test]
    #[cfg(feature = "embed")]
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
            var nativeTarget = {};
        "#,
                false,
            )
            .unwrap();
        let (handles, weak) = weak_globals(&mut engine, &["mixedOwner", "nativeTarget"]);
        engine.eval("nativeTarget = null", false).unwrap();
        let [owner_weak, native_weak] = &weak[..] else {
            unreachable!()
        };
        engine.op_state().put(NativeCache {
            values: handles,
            edges: vec![(0, 1)],
            ..Default::default()
        });
        engine.op_state().register_gc::<NativeCache>();
        engine.interp.gc_collect();
        assert!(
            crate::gc_test::alive(native_weak),
            "the native edge shares the coroutine owner"
        );
        match engine.eval("mixedOwner.next().value", false).unwrap() {
            crate::Completion::Value(value) => assert_eq!(value, "41"),
            crate::Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
        engine.eval("mixedOwner = null", false).unwrap();
        engine.interp.gc_collect();
        assert!(!crate::gc_test::alive(owner_weak));
        assert!(!crate::gc_test::alive(native_weak));
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

/// P1 S2: the JavaScript values of a host type registered for tracing live exactly as its
/// [`HostTrace`] reports, through weak handles (S6 step 11b).
#[cfg(all(test, feature = "embed"))]
mod host_trace_tests {
    use super::*;
    use crate::handles::WeakHandle;
    use crate::scope::{Ctx, Value};

    /// Reports `root` as a root and `target` as an edge from `owner`.
    struct Slots {
        root: WeakHandle,
        owner: WeakHandle,
        target: WeakHandle,
    }

    impl HostTrace for Slots {
        fn trace_gc(&self, visitor: &mut dyn HostGcVisitor) {
            visitor.root(&self.root);
            visitor.edge(&self.owner, &self.target);
        }
    }

    struct Resource(WeakHandle);

    impl HostTrace for Resource {
        fn trace_gc(&self, visitor: &mut dyn HostGcVisitor) {
            visitor.root(&self.0);
        }
    }

    /// A fresh object `{n: value}`, weakly.
    fn marker(cx: &mut Ctx<'_>, value: f64) -> WeakHandle {
        let object = Value::Obj(cx.new_object());
        cx.member_set(&object, "n", Value::Num(value))
            .unwrap_or_else(|_| panic!("set n"));
        WeakHandle::new(cx, &object).expect("a weak handle")
    }

    /// The `n` of a live marker.
    fn n(engine: &mut crate::Engine, marker: &WeakHandle) -> f64 {
        engine.scope(|cx| {
            let object = marker
                .upgrade(cx)
                .expect("the engine's handle")
                .expect("a live marker");
            cx.member_get(&object, "n")
                .ok()
                .and_then(|n| n.as_num_opt())
                .expect("marker number")
        })
    }

    #[test]
    fn host_slots_and_resources_keep_what_they_report() {
        let mut engine = crate::Engine::new();
        engine.op_state().register_trace::<Slots>();
        engine.op_state().resources.register_trace::<Resource>();
        // The markers live in host state only (and in this helper's frame while it runs).
        let rid = crate::gc_test::in_frame(|| {
            let (slots, resource) = engine.scope(|cx| {
                (
                    Slots {
                        root: marker(cx, 1.0),
                        owner: marker(cx, 2.0),
                        target: marker(cx, 3.0),
                    },
                    Resource(marker(cx, 4.0)),
                )
            });
            engine.op_state().put(slots);
            engine.op_state().resources.add(resource)
        });
        engine.interp.gc_collect();
        engine.interp.gc_collect();
        let root = engine.host::<Slots>().expect("slots").root.clone();
        assert_eq!(n(&mut engine, &root), 1.0, "a root survives");
        let resource = engine
            .op_state()
            .resources
            .get::<Resource>(rid)
            .expect("resource");
        assert_eq!(
            n(&mut engine, &resource.0),
            4.0,
            "a resource's root survives"
        );
        // Nothing keeps the edge's owner alive, so neither it nor its target survives.
        let slots = engine.host::<Slots>().expect("slots");
        assert!(slots.owner.is_cleared(), "an unrooted owner dies");
        assert!(slots.target.is_cleared(), "its edge target dies with it");
    }

    /// Tracing is opt-in (S6 step 11a): a type stored without registering its [`HostTrace`]
    /// reports nothing, and what only it holds dies.
    #[test]
    fn an_unregistered_host_type_reports_nothing() {
        let mut engine = crate::Engine::new();
        crate::gc_test::in_frame(|| {
            let slots = engine.scope(|cx| Slots {
                root: marker(cx, 1.0),
                owner: marker(cx, 2.0),
                target: marker(cx, 3.0),
            });
            engine.op_state().put(slots);
        });
        engine.interp.gc_collect();
        assert!(
            engine.host::<Slots>().expect("slots").root.is_cleared(),
            "an unregistered type keeps nothing alive"
        );
    }

    #[test]
    fn host_edges_keep_targets_while_their_owner_lives() {
        let mut engine = crate::Engine::new();
        engine.op_state().register_trace::<Slots>();
        crate::gc_test::in_frame(|| {
            let slots = engine.scope(|cx| {
                let slots = Slots {
                    root: marker(cx, 1.0),
                    owner: marker(cx, 2.0),
                    target: marker(cx, 3.0),
                };
                // The owner is reachable from the global object.
                let owner = slots.owner.upgrade(cx).unwrap().unwrap();
                let global = cx.global_this();
                cx.member_set(&global, "owner", owner)
                    .unwrap_or_else(|_| panic!("set owner"));
                slots
            });
            engine.op_state().put(slots);
        });
        engine.interp.gc_collect();
        let target = engine.host::<Slots>().expect("slots").target.clone();
        assert!(!target.is_cleared(), "a live owner keeps its target");
        assert_eq!(n(&mut engine, &target), 3.0);
    }

    /// A native closure keeps the JavaScript values it captures as engine captures, which live
    /// as long as the function (S6 step 11b; before it, a reporter declared them).
    #[test]
    fn reported_native_closure_captures_survive() {
        let mut engine = crate::Engine::new();
        crate::gc_test::in_frame(|| {
            engine.scope(|cx| {
                let captured = marker(cx, 5.0).upgrade(cx).unwrap().unwrap();
                let read = cx.new_native_closure_with_captures(
                    "readCaptured",
                    0,
                    |cx, _, _, captures| cx.member_get(&captures[0], "n"),
                    &[captured],
                    crate::value::NativeCaptures::None,
                );
                let global = cx.global_this();
                cx.member_set(&global, "readCaptured", read)
                    .unwrap_or_else(|_| panic!("install"));
            });
        });
        engine.interp.gc_collect();
        engine.interp.gc_collect();
        match engine.eval("readCaptured()", false).expect("parses") {
            crate::Completion::Value(value) => assert_eq!(value, "5"),
            crate::Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }
}
