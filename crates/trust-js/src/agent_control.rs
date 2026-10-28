//! Agent state shared by all interpreter implementations, independent of native-thread TLS.
use crate::agent_access::{Access, RealmId};
use crate::gc_trace::{GcSink, GcTrace};
use crate::interpreter::Interp;
use crate::rooting::Rooted;
use crate::value::Value;
use crate::value::{Gc, GcHeap};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

/// Host-defined authorization identity, independent of a callable's execution Realm/settings.
/// Historical code carries only identity: it must not root an otherwise unreachable origin.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LoadAuthority {
    pub(crate) agent: u64,
    pub(crate) owner: RealmId,
    pub(crate) context: u64,
}

/// The identity is this stable Rc allocation, never a URL or an authority tuple. Derived
/// functions/eval retain the SAME record. Neutral parsed AST/Chunk caches never own one.
pub(crate) type RuntimeOrigin = Rc<ReferrerRecord>;

pub(crate) enum ReferrerKind {
    Script,
    Module(String),
    Realm,
    Closed,
}

#[derive(Clone, Eq, PartialEq, Hash)]
pub(crate) struct LoadedRequest {
    pub(crate) specifier: String,
    pub(crate) attr_type: Option<String>,
    pub(crate) defer: bool,
}

pub(crate) struct ReferrerRecord {
    pub(crate) source: Option<Rc<crate::stack_trace::ScriptSource>>,
    pub(crate) referrer: Option<Rc<str>>,
    pub(crate) authority: LoadAuthority,
    /// A source's actual owner and Realm are independent of both its execution caller and
    /// authorization owner. None is the deliberately closed missing-metadata sentinel.
    pub(crate) owner: Option<Rc<Route>>,
    pub(crate) realm: Option<Gc>,
    pub(crate) host_context: u64,
    pub(crate) kind: ReferrerKind,
    /// Plain owner-local module keys add no mutable GC links to old uncarded closures.
    pub(crate) loaded: RefCell<crate::fasthash::FastMap<LoadedRequest, String>>,
}

impl GcTrace for ReferrerRecord {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        self.trace_edges(sink);
    }
}

impl ReferrerRecord {
    /// [`GcTrace`]'s edges, for any sink (a closure's trace reports its origin's).
    #[inline]
    pub(crate) fn trace_edges<S: crate::gc_trace::TraceSink + ?Sized>(&self, sink: &mut S) {
        let Self {
            source: _,
            referrer: _,
            authority: _,
            owner,
            realm,
            host_context: _,
            kind: _,
            loaded: _,
        } = self;
        owner.trace_gc(sink.as_dyn());
        if let Some(realm) = realm {
            sink.object(realm);
        }
    }
}

/// Only contexts entered ABOVE this floor precede the real cross-interpreter caller. The
/// target's suspended older local stack must not override a child caller's null-source up-call.
pub(crate) struct CallerOrigin {
    pub(crate) floor: u32,
    pub(crate) job_floor: usize,
    pub(crate) origin: Option<RuntimeOrigin>,
}

impl GcTrace for CallerOrigin {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let Self {
            floor: _,
            job_floor: _,
            origin,
        } = self;
        origin.trace_gc(sink);
    }
}

/// Immutable retained path. The topology directory does not trace this cache; active entries
/// and queued/pending work explicitly trace it. Thus registration does not root an idle realm.
pub(crate) struct Route {
    pub(crate) agent: u64,
    pub(crate) target: RealmId,
    pub(crate) owners: Box<[Gc]>,
}

impl GcTrace for Route {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let Self {
            agent: _,
            target: _,
            owners,
        } = self;
        owners.trace_gc(sink);
    }
}

pub(crate) enum MetadataOwners {
    One(RealmId),
    Many(Box<[RealmId]>),
}
impl MetadataOwners {
    pub(crate) fn ids(&self) -> &[RealmId] {
        match self {
            Self::One(id) => std::slice::from_ref(id),
            Self::Many(ids) => ids,
        }
    }
    fn include(&mut self, id: RealmId) {
        if self.ids().contains(&id) {
            return;
        }
        let mut ids = self.ids().to_vec();
        ids.push(id);
        *self = Self::Many(ids.into_boxed_slice());
    }
}

pub(crate) struct AgentControl {
    pub(crate) access: Rc<Access<Interp>>,
    pub(crate) root_route: Rc<Route>,
    /// Non-owning native-metadata index. Addresses are protected by the owning table's pin;
    /// sweep removes them before cell reuse. The directory/index never traces a Realm. It is
    /// kept only while `children` is nonzero: a lone interpreter's own tables are authoritative.
    pub(crate) metadata_owners: RefCell<crate::fasthash::FastMap<usize, MetadataOwners>>,
    /// Live ShadowRealm interpreters (every descendant of the root). Zero means no metadata,
    /// callable or Realm can belong to another interpreter, so lookups stay local.
    pub(crate) children: Cell<u32>,
    pub(crate) suppressed: Cell<u32>,
    pub(crate) collecting: Cell<bool>,
    pub(crate) pending: Cell<Option<crate::gc_core::CollectionKind>>,
    pub(crate) kept_alive: RefCell<Vec<Value>>,
    pub(crate) work: RefCell<Work>,
    pub(crate) execution_depth: Cell<u32>,
    pub(crate) checkpoint: Cell<bool>,
    pub(crate) checkpoint_requested: Cell<bool>,
    pub(crate) limit: Cell<i64>,
    pub(crate) next: Cell<i64>,
    pub(crate) task_allocated: Cell<u64>,
    pub(crate) task_live: Cell<i64>,
    pub(crate) task_pending: Cell<bool>,
    pub(crate) task_deferred: Cell<bool>,
    /// ArrayBuffer data every member allocated since the last major collection, and the amount
    /// that triggers the next one (see [`Interp::note_buffer_allocation`]).
    pub(crate) buffers: Cell<crate::interpreter::BufferPressure>,
}

impl AgentControl {
    pub(crate) fn new(agent: u64) -> Self {
        let limit = crate::interpreter::default_live_object_limit();
        Self {
            access: Rc::default(),
            root_route: Rc::new(Route {
                agent,
                target: RealmId::ROOT,
                owners: Box::new([]),
            }),
            metadata_owners: RefCell::default(),
            children: Cell::new(0),
            suppressed: Cell::new(0),
            collecting: Cell::new(false),
            pending: Cell::new(None),
            kept_alive: RefCell::default(),
            work: RefCell::default(),
            execution_depth: Cell::new(0),
            checkpoint: Cell::new(false),
            checkpoint_requested: Cell::new(false),
            limit: Cell::new(limit),
            next: Cell::new(limit),
            task_allocated: Cell::new(0),
            task_live: Cell::new(0),
            task_pending: Cell::new(false),
            task_deferred: Cell::new(false),
            buffers: Cell::new(crate::interpreter::BufferPressure::default()),
        }
    }
}

pub(crate) struct Suppression {
    heap: GcHeap,
}

impl Suppression {
    pub(crate) fn new(heap: &GcHeap) -> Self {
        let depth = heap.agent.suppressed.get();
        heap.agent
            .suppressed
            .set(depth.checked_add(1).expect("GC suppression depth overflow"));
        Self { heap: heap.clone() }
    }
}

impl Drop for Suppression {
    fn drop(&mut self) {
        self.heap
            .agent
            .suppressed
            .set(self.heap.agent.suppressed.get() - 1);
        // A pending request is serviced at a later safepoint, never from Drop.
    }
}

impl Interp {
    /// Agent policy is canonical; every native cache mirrors the same allowance before a Rust
    /// poll returns. An idle child's next entry must not inherit an obsolete ceiling. The heap
    /// keeps the poll words' heap bit set while the live-object count is above it.
    pub(crate) fn set_agent_next(&mut self, next: i64) {
        // All writers use this method, and new_in_heap initializes every member from this
        // same canonical Cell before adoption (bootstrap cannot run host callbacks). Thus
        // unchanged policy needs no full-directory walk at each ordinary collection.
        if self.gc_heap.agent.next.get() == next {
            return;
        }
        self.gc_heap.agent.next.set(next);
        self.gc_heap.core.borrow_mut().set_live_limit(next);
        let access = self.gc_heap.agent.access.clone();
        self.with_raw_access(|_| {
            for (_, pointer) in access.snapshot() {
                unsafe {
                    (*pointer.as_ptr()).gc_next = next;
                }
            }
        });
    }

    pub(crate) fn set_agent_limit(&mut self, limit: i64) {
        self.gc_heap.agent.limit.set(limit);
        self.set_agent_next(limit);
    }

    /// Whether this interpreter is its Agent's only one (no live ShadowRealm interpreter).
    #[inline(always)]
    pub(crate) fn single_interpreter(&self) -> bool {
        self.gc_heap.agent.children.get() == 0
    }

    pub(crate) fn remember_metadata_owner(&self, identity: usize) {
        // Bootstrap records are published together after the real child id is assigned. A
        // lone interpreter needs no index: the first child's adoption publishes its owners.
        if self.realm_id == RealmId::UNPUBLISHED || self.single_interpreter() {
            return;
        }
        let mut index = self.gc_heap.agent.metadata_owners.borrow_mut();
        index
            .entry(identity)
            .and_modify(|owners| owners.include(self.realm_id))
            .or_insert(MetadataOwners::One(self.realm_id));
    }

    /// Pure authoritative lookup. Same-owner is one ordinary table probe; only a foreign
    /// identity consults the non-owning index. `has` may neither run JS/GC nor return a loan.
    pub(crate) fn metadata_owner(
        &self,
        identity: usize,
        has: fn(&Interp, usize) -> bool,
    ) -> Option<RealmId> {
        if has(self, identity) {
            return Some(self.realm_id);
        }
        self.foreign_metadata_owner(identity, has)
    }

    /// The other interpreter owning `identity`, after the caller's own table missed.
    #[inline(always)]
    pub(crate) fn foreign_metadata_owner(
        &self,
        identity: usize,
        has: fn(&Interp, usize) -> bool,
    ) -> Option<RealmId> {
        if self.single_interpreter() {
            return None;
        }
        self.indexed_metadata_owner(identity, has)
    }

    #[inline(never)]
    fn indexed_metadata_owner(
        &self,
        identity: usize,
        has: fn(&Interp, usize) -> bool,
    ) -> Option<RealmId> {
        let index = self.gc_heap.agent.metadata_owners.borrow();
        let owners = index.get(&identity)?;
        owners.ids().iter().copied().find(|&id| {
            id != self.realm_id
                && self.gc_heap.agent.access.contains(id)
                && self.with_realm_view(id, |owner| has(owner, identity))
        })
    }

    pub(crate) fn realm_owner(&self, global: usize) -> Option<RealmId> {
        self.metadata_owner(global, |interp, global| interp.realms.contains_key(&global))
    }

    pub(crate) fn realm_host_context(&self, global: usize, fallback: u64) -> u64 {
        self.realm_owner(global).map_or(fallback, |owner| {
            self.with_realm_view(owner, |interp| interp.realms[&global].host_job_context)
        })
    }

    /// A callback-free metadata loan, not a language Realm switch. The closure receives ONLY
    /// the selected table, and must finish before author calls/collection. Return owned data,
    /// never a reference into a table. Thus borrowed builtin algorithms retain their caller's
    /// Realm for errors, species construction and observable callbacks.
    pub(crate) fn with_metadata_table<T, R>(
        &mut self,
        owner: RealmId,
        table: fn(&mut Interp) -> &mut T,
        operation: impl FnOnce(&mut T) -> R,
    ) -> R {
        if owner == self.realm_id {
            return operation(table(self));
        }
        let access = self.gc_heap.agent.access.clone();
        self.with_raw_access(|_| {
            let pointer = access
                .resolve(owner)
                .expect("authoritative metadata owner is live");
            // SAFETY: requester is suspended; the newest owner lineage is used only during
            // this short, callback-free loan. No table reference escapes the closure.
            operation(table(unsafe { &mut *pointer.as_ptr() }))
        })
    }

    pub(crate) fn realm_route(&self) -> Rc<Route> {
        self.shadow_home.as_ref().map_or_else(
            || self.gc_heap.agent.root_route.clone(),
            |home| home.route.clone(),
        )
    }

    /// Prepare metadata before publishing, then suspend only through the current capability.
    /// No caller may use its older self reference until this closure has returned.
    pub(crate) fn with_raw_access<R>(&mut self, operation: impl FnOnce(NonNull<Interp>) -> R) -> R {
        let _agent = self.enter_agent();
        let route = self.realm_route();
        debug_assert_eq!(route.target, self.realm_id);
        debug_assert_eq!(route.agent, crate::value::heap_id(&self.gc_heap));
        let _route = (!route.owners.is_empty()).then(|| Rooted::new(route));
        let access = self.gc_heap.agent.access.clone();
        let id = self.realm_id;
        access.suspend(id, self, operation)
    }

    pub(crate) fn with_access<R>(&mut self, operation: impl FnOnce(&mut Interp) -> R) -> R {
        self.with_raw_access(|pointer| {
            // SAFETY: this is the published current lineage. Deeper suspension refreshes it
            // from its own current reference; it never revives an older owner pointer.
            operation(unsafe { &mut *pointer.as_ptr() })
        })
    }

    pub(crate) fn with_route<R>(
        &mut self,
        target: RealmId,
        operation: impl FnOnce(&mut Interp) -> R,
    ) -> R {
        let access = self.gc_heap.agent.access.clone();
        let requester = self.realm_id;
        let caller_origin = self.active_runtime_origin();
        self.with_raw_access(|_| {
            let pointer = access
                .resolve(target)
                .expect("routed interpreter is still owned");
            // SAFETY: active target is the latest suspended lineage; inactive target is the
            // raw-owning capability. No reference into the requester survives this call.
            let target_interp = unsafe { &mut *pointer.as_ptr() };
            target_interp.with_access(|target_interp| {
                if target == requester {
                    operation(target_interp)
                } else {
                    target_interp.with_runtime_origin(caller_origin, |target_interp| {
                        target_interp.with_routed_execution(operation)
                    })
                }
            })
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CollectionOutcome {
    Collected(crate::gc_core::CollectionKind),
    Deferred,
}

pub(crate) struct Collecting(pub(crate) GcHeap);
impl Drop for Collecting {
    fn drop(&mut self) {
        self.0.agent.collecting.set(false);
    }
}

struct OriginRestore {
    pointer: NonNull<Interp>,
    saved: Rooted<Option<CallerOrigin>>,
}
impl Drop for OriginRestore {
    fn drop(&mut self) {
        // SAFETY: same local capability after operation references end; neutral metadata only.
        unsafe {
            (*self.pointer.as_ptr()).caller_origin = self.saved.replace(None);
        }
    }
}
impl Interp {
    pub(crate) fn default_load_authority(&self) -> LoadAuthority {
        LoadAuthority {
            agent: crate::value::heap_id(&self.gc_heap),
            owner: self.realm_id,
            context: self.host_job_context,
        }
    }
    pub(crate) fn default_runtime_origin(
        &self,
        source: Option<Rc<crate::stack_trace::ScriptSource>>,
    ) -> RuntimeOrigin {
        let referrer = match source
            .as_deref()
            .map(crate::stack_trace::ScriptSource::kind)
        {
            None | Some(crate::stack_trace::SourceKind::Script) => {
                Some(Rc::from(self.import_base.as_str()))
            }
            _ => source
                .as_ref()
                .and_then(|source| source.referrer().cloned()),
        };
        let kind = if source.is_some() {
            ReferrerKind::Script
        } else {
            ReferrerKind::Realm
        };
        self.new_referrer_record(source, referrer, self.default_load_authority(), kind)
    }

    pub(crate) fn new_referrer_record(
        &self,
        source: Option<Rc<crate::stack_trace::ScriptSource>>,
        referrer: Option<Rc<str>>,
        authority: LoadAuthority,
        kind: ReferrerKind,
    ) -> RuntimeOrigin {
        Rc::new(ReferrerRecord {
            source,
            referrer,
            authority,
            owner: Some(self.realm_route()),
            realm: Some(self.global),
            host_context: self.host_job_context,
            kind,
            loaded: RefCell::default(),
        })
    }
    pub(crate) fn load_authority(&self) -> LoadAuthority {
        self.active_runtime_origin()
            .map_or_else(|| self.default_load_authority(), |origin| origin.authority)
    }
    pub(crate) fn with_runtime_origin<R>(
        &mut self,
        origin: Option<RuntimeOrigin>,
        operation: impl FnOnce(&mut Interp) -> R,
    ) -> R {
        let floor = self.context_depth();
        let job_floor = self.job_script_contexts.len();
        crate::agent_access::suspend_local(self, |pointer| {
            let saved = unsafe {
                (*pointer.as_ptr()).caller_origin.replace(CallerOrigin {
                    floor,
                    job_floor,
                    origin,
                })
            };
            let _restore = OriginRestore {
                pointer,
                saved: Rooted::new(saved),
            };
            operation(unsafe { &mut *pointer.as_ptr() })
        })
    }
    pub(crate) fn authority_is_live(&self, authority: LoadAuthority) -> bool {
        authority.agent == crate::value::heap_id(&self.gc_heap)
            && self.gc_heap.agent.access.contains(authority.owner)
            && self.with_realm_view(authority.owner, |owner| {
                !owner.retired_host_job_contexts.contains(&authority.context)
            })
    }
    pub(crate) fn authority_route(&self, authority: LoadAuthority) -> Option<Rc<Route>> {
        self.authority_is_live(authority)
            .then(|| self.with_realm_view(authority.owner, Interp::realm_route))
    }
    pub(crate) fn cancel_load_authority(&mut self, authority: LoadAuthority) {
        let heap = self.gc_heap.clone();
        heap.agent
            .work
            .borrow_mut()
            .imports
            .retain(|_, request| request.payload.authority != authority);
        let access = heap.agent.access.clone();
        self.with_raw_access(|_| {
            for (_, pointer) in access.snapshot() {
                let target = unsafe { &mut *pointer.as_ptr() };
                let id = target.realm_id;
                target
                    .modules
                    .retain(|key, _| !crate::modules::module_key_has_authority(key, id, authority));
                target
                    .module_recs
                    .retain(|key, _| !crate::modules::module_key_has_authority(key, id, authority));
                let cancelled: Vec<_> = target
                    .routed_imports
                    .iter()
                    .filter(|(_, state)| state.authority == authority)
                    .map(|(&owner, _)| owner)
                    .collect();
                for owner in cancelled {
                    target.routed_imports.remove(&owner);
                }
            }
        });
    }
}

/// Saved caller state stays precisely rooted across all reentry. Its mutable borrow exists
/// only for the short swap itself, never across author code or host callbacks.
struct RealmRestore {
    pointer: NonNull<Interp>,
    key: usize,
    saved: Rooted<RefCell<crate::interpreter::RealmState>>,
}
impl Drop for RealmRestore {
    fn drop(&mut self) {
        // SAFETY: callback references descended from this capability and have ended. This
        // restoration performs bookkeeping only, with no JavaScript or collection.
        let interp = unsafe { &mut *self.pointer.as_ptr() };
        {
            let mut saved = self.saved.borrow_mut();
            interp.swap_realm(&mut saved);
        }
        interp
            .realms
            .insert(self.key, self.saved.borrow().snapshot_clone());
    }
}
struct ConstructorRestore {
    pointer: NonNull<Interp>,
    saved: Rooted<RefCell<Option<crate::interpreter::RealmState>>>,
}
impl Drop for ConstructorRestore {
    fn drop(&mut self) {
        // SAFETY: same scoped lineage after callback references ended, including on unwind.
        unsafe {
            (*self.pointer.as_ptr()).ctor_caller_realm = self.saved.borrow_mut().take();
        }
    }
}
impl Interp {
    pub(crate) fn with_realm_snapshot<R>(
        &mut self,
        key: usize,
        target: crate::interpreter::RealmState,
        operation: impl FnOnce(&mut Interp) -> R,
    ) -> R {
        self.with_raw_access(|pointer| {
            let restore = RealmRestore {
                pointer,
                key,
                saved: Rooted::new(RefCell::new(target)),
            };
            unsafe {
                (&mut *pointer.as_ptr()).swap_realm(&mut restore.saved.borrow_mut());
            }
            let result = operation(unsafe { &mut *pointer.as_ptr() });
            drop(restore);
            result
        })
    }
    pub(crate) fn with_constructor_caller<R>(
        &mut self,
        caller: Option<crate::interpreter::RealmState>,
        operation: impl FnOnce(&mut Interp) -> R,
    ) -> R {
        if caller.is_none() && self.ctor_caller_realm.is_none() {
            return operation(self);
        }
        self.with_raw_access(|pointer| {
            let saved = Rooted::new(RefCell::new(unsafe {
                std::mem::replace(&mut (*pointer.as_ptr()).ctor_caller_realm, caller)
            }));
            let restore = ConstructorRestore { pointer, saved };
            let result = operation(unsafe { &mut *pointer.as_ptr() });
            drop(restore);
            result
        })
    }
}

struct SettingsRestore {
    pointer: NonNull<Interp>,
    saved: Rooted<crate::interpreter::RealmState>,
}
impl Drop for SettingsRestore {
    fn drop(&mut self) {
        // SAFETY: original scoped capability after callback refs end. Save only bookkeeping;
        // no host hook, JS allocation or collection is invoked from unwinding restoration.
        let interp = unsafe { &mut *self.pointer.as_ptr() };
        if !interp
            .retired_host_job_contexts
            .contains(&interp.host_job_context)
        {
            interp.host_settings_states.insert(
                (Gc::as_ptr(&interp.global) as usize, interp.host_job_context),
                crate::interpreter::HostSettingsState {
                    global_env: interp.global_env,
                    global_var_names: interp.global_var_names.clone(),
                },
            );
        }
        interp.restore_realm(&self.saved);
    }
}
impl Interp {
    pub(crate) fn with_settings_context<R: GcTrace + 'static>(
        &mut self,
        context: u64,
        operation: impl FnOnce(&mut Interp) -> R,
    ) -> R {
        if context == self.host_job_context {
            return operation(self);
        }
        self.with_raw_access(|pointer| {
            let saved = Rooted::new(unsafe { (&*pointer.as_ptr()).snapshot_realm() });
            let restore = SettingsRestore { pointer, saved };
            if let Some(enter) = unsafe { (*pointer.as_ptr()).host_job_context_enter } {
                unsafe {
                    (&mut *pointer.as_ptr()).with_host_callback(|interp| enter(interp, context));
                }
            }
            unsafe {
                (&mut *pointer.as_ptr()).switch_host_job_context(context);
            }
            let result = Rooted::new(RefCell::new(Some(operation(unsafe {
                &mut *pointer.as_ptr()
            }))));
            if let Some(leave) = unsafe { (*pointer.as_ptr()).host_job_context_leave } {
                unsafe {
                    (&mut *pointer.as_ptr()).with_host_callback(leave);
                }
            }
            drop(restore);
            let output = result
                .borrow_mut()
                .take()
                .expect("settings callback returned");
            output
        })
    }
    pub(crate) fn with_realm_view<R>(
        &self,
        target: RealmId,
        operation: impl FnOnce(&Interp) -> R,
    ) -> R {
        if target == self.realm_id {
            return operation(self);
        }
        let pointer = self
            .gc_heap
            .agent
            .access
            .resolve(target)
            .expect("viewed realm is owned");
        // SAFETY: active targets published their newest lineage at suspension; inactive ones
        // have raw owners. The synchronous shared visitor cannot retain a reference or run JS.
        operation(unsafe { &*pointer.as_ptr() })
    }
}

impl AgentControl {
    pub(crate) fn request(&self, heap: &GcHeap, kind: Option<crate::gc_core::CollectionKind>) {
        use crate::gc_core::CollectionKind;
        let mut core = heap.core.try_borrow_mut().ok();
        let requested = kind.unwrap_or_else(|| {
            core.as_ref()
                .map_or(CollectionKind::Major, |core| core.due_kind())
        });
        let requested = if self.pending.get() == Some(CollectionKind::Major) {
            CollectionKind::Major
        } else {
            requested
        };
        self.pending.set(Some(requested));
        if let Some(core) = &mut core {
            core.request_collection(requested);
        }
    }
}

struct EntryRestore {
    pointer: NonNull<Interp>,
    script: Option<(usize, usize)>,
    host: Option<usize>,
}
impl Drop for EntryRestore {
    fn drop(&mut self) {
        // SAFETY: callback borrowed through this capability and has ended. Numeric execution
        // bookkeeping restoration cannot execute JS or collect.
        let interp = unsafe { &mut *self.pointer.as_ptr() };
        interp.script_entry = self.script;
        interp.host_entry_realm = self.host;
    }
}
impl Interp {
    pub(crate) fn with_entry_context<R>(
        &mut self,
        script: Option<(usize, usize)>,
        host: Option<usize>,
        operation: impl FnOnce(&mut Interp) -> R,
    ) -> R {
        self.with_raw_access(|pointer| {
            let interp = unsafe { &mut *pointer.as_ptr() };
            let restore = EntryRestore {
                pointer,
                script: interp.script_entry,
                host: interp.host_entry_realm,
            };
            interp.script_entry = script;
            interp.host_entry_realm = host;
            // The setup borrow is not used again; execution descends from its raw capability.
            let result = operation(unsafe { &mut *pointer.as_ptr() });
            drop(restore);
            result
        })
    }
}

/// Every work item owns its dispatch path separately from the handler's language Realm.
pub(crate) struct Routed<T> {
    pub(crate) route: Rc<Route>,
    pub(crate) payload: T,
}
impl<T: GcTrace> GcTrace for Routed<T> {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let Self { route, payload } = self;
        route.trace_gc(sink);
        payload.trace_gc(sink);
    }
}

/// An intrinsic continuation is an edge FROM its outer promise, not an unconditional Agent
/// root. Discarding an unresolved intrinsic await therefore permits the whole realm to retire.
pub(crate) struct RoutedImport {
    pub(crate) requester: Rc<Route>,
    pub(crate) requester_realm: Gc,
    pub(crate) context: u64,
    pub(crate) authority: LoadAuthority,
    pub(crate) source: Rc<Route>,
    pub(crate) request: Value,
    pub(crate) mode: ImportMode,
    pub(crate) origin: Option<RuntimeOrigin>,
}

pub(crate) enum ImportMode {
    Namespace,
    Shadow { shadow: Gc, name: String },
}
impl GcTrace for ImportMode {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        match self {
            Self::Namespace => {}
            Self::Shadow { shadow, name: _ } => shadow.trace_gc(sink),
        }
    }
}
impl GcTrace for RoutedImport {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let Self {
            requester,
            requester_realm,
            context: _,
            authority: _,
            source,
            request,
            mode,
            origin,
        } = self;
        requester.trace_gc(sink);
        requester_realm.trace_gc(sink);
        source.trace_gc(sink);
        request.trace_gc(sink);
        mode.trace_gc(sink);
        origin.trace_gc(sink);
    }
}

pub(crate) struct Cleanup {
    pub(crate) registry: Value,
    pub(crate) registry_realm: Gc,
    pub(crate) registry_context: u64,
    pub(crate) callback_realm: Option<usize>,
    pub(crate) callback_context: u64,
}
impl GcTrace for Cleanup {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let Self {
            registry,
            registry_realm,
            registry_context: _,
            callback_realm,
            callback_context: _,
        } = self;
        registry.trace_gc(sink);
        registry_realm.trace_gc(sink);
        if let Some(global) = callback_realm {
            sink.word(*global as u64);
        }
    }
}

pub(crate) struct Timer {
    pub(crate) callback: Value,
    pub(crate) deadline: std::time::Instant,
    pub(crate) order: u64,
}
impl GcTrace for Timer {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let Self {
            callback,
            deadline: _,
            order: _,
        } = self;
        callback.trace_gc(sink);
    }
}

pub(crate) struct Work {
    pub(crate) promises: VecDeque<Routed<crate::interpreter::Job>>,
    pub(crate) cleanup: VecDeque<Routed<Cleanup>>,
    pub(crate) waits: Vec<Routed<crate::interpreter::AsyncWait>>,
    pub(crate) inbox: Option<Arc<crate::interpreter::AsyncWaitInbox>>,
    pub(crate) timers: Vec<Routed<Timer>>,
    pub(crate) next_timer: u64,
    pub(crate) imports: crate::fasthash::FastMap<u64, Routed<crate::modules::PendingDynamicImport>>,
    pub(crate) next_import: u64,
}
impl Default for Work {
    fn default() -> Self {
        Self {
            promises: VecDeque::new(),
            cleanup: VecDeque::new(),
            waits: Vec::new(),
            inbox: None,
            timers: Vec::new(),
            next_timer: 0,
            imports: Default::default(),
            next_import: 1,
        }
    }
}
impl GcTrace for Work {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let Self {
            promises,
            cleanup,
            waits,
            inbox: _,
            timers,
            next_timer: _,
            imports,
            next_import: _,
        } = self;
        promises.trace_gc(sink);
        cleanup.trace_gc(sink);
        waits.trace_gc(sink);
        timers.trace_gc(sink);
        imports.trace_gc(sink);
    }
}
impl Drop for Work {
    fn drop(&mut self) {
        // Accepted host wait ownership ends with the Agent, not with an individual realm.
        // Remove the futex records before releasing their Shared Data Blocks/inbox.
        for wait in &self.waits {
            crate::interpreter::futex_remove_async(
                wait.payload.id,
                wait.payload.index,
                wait.payload.token,
            );
        }
    }
}
impl GcTrace for AgentControl {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        let Self {
            access: _,
            root_route: _,
            metadata_owners: _, // non-owning identities, never roots
            children: _,
            suppressed: _,
            collecting: _,
            pending: _,
            kept_alive,
            work,
            execution_depth: _,
            checkpoint: _,
            checkpoint_requested: _,
            limit: _,
            next: _,
            task_allocated: _,
            task_live: _,
            task_pending: _,
            task_deferred: _,
            buffers: _,
        } = self;
        kept_alive.trace_gc(sink);
        work.trace_gc(sink);
    }
}
impl Interp {
    #[cfg(test)]
    pub(crate) fn enqueue_test_cleanup(&mut self, registry: Value) {
        let ptr = registry
            .as_obj()
            .map(|object| Gc::as_ptr(&object) as usize)
            .expect("registry object");
        let state = self
            .finalization_registries
            .get(&ptr)
            .expect("registry state");
        let payload = Cleanup {
            registry,
            registry_realm: state.registry_realm,
            registry_context: state.registry_context,
            callback_realm: state.callback_realm,
            callback_context: state.callback_context,
        };
        let route = self.realm_route();
        self.gc_heap
            .agent
            .work
            .borrow_mut()
            .cleanup
            .push_back(Routed { route, payload });
    }

    /// Ready Agent work only. Future IO/waits/timers are not a reason to hold a checkpoint open.
    pub(crate) fn has_pending_jobs(&self) -> bool {
        let ready = {
            let work = self.gc_heap.agent.work.borrow();
            !work.promises.is_empty() || !work.cleanup.is_empty()
        };
        ready || self.async_waits_ready()
    }

    pub(crate) fn add_kept_object(&self, value: Value) {
        self.gc_heap.agent.kept_alive.borrow_mut().push(value);
    }
    pub(crate) fn enqueue_promise_job(&mut self, payload: crate::interpreter::Job) {
        let route = if let Some((requester, _, _, _)) = payload.handler.internal.routed_import() {
            self.with_realm_view(requester, |interp| interp.realm_route())
        } else if self.single_interpreter() {
            self.realm_route()
        } else if let Some(owner) = payload.realm.and_then(|realm| self.realm_owner(realm)) {
            self.with_realm_view(owner, Interp::realm_route)
        } else {
            self.realm_route()
        };
        self.gc_heap
            .agent
            .work
            .borrow_mut()
            .promises
            .push_back(Routed { route, payload });
    }
    pub(crate) fn agent_async_inbox(&self) -> Arc<crate::interpreter::AsyncWaitInbox> {
        self.gc_heap
            .agent
            .work
            .borrow_mut()
            .inbox
            .get_or_insert_with(Default::default)
            .clone()
    }
    pub(crate) fn enqueue_async_wait(&mut self, payload: crate::interpreter::AsyncWait) {
        let route = self.realm_route();
        self.gc_heap
            .agent
            .work
            .borrow_mut()
            .waits
            .push(Routed { route, payload });
    }
    pub(crate) fn enqueue_agent_timer(&mut self, callback: Value, deadline: std::time::Instant) {
        let route = self.realm_route();
        let mut work = self.gc_heap.agent.work.borrow_mut();
        let order = work.next_timer;
        work.next_timer = order
            .checked_add(1)
            .expect("Agent timer sequence exhausted");
        work.timers.push(Routed {
            route,
            payload: Timer {
                callback,
                deadline,
                order,
            },
        });
    }
}

pub(crate) enum WorkItem {
    Promise(Routed<crate::interpreter::Job>),
    Cleanup(Routed<Cleanup>),
}
impl WorkItem {
    fn route(&self) -> &Rc<Route> {
        match self {
            Self::Promise(item) => &item.route,
            Self::Cleanup(item) => &item.route,
        }
    }
}
impl GcTrace for WorkItem {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        match self {
            Self::Promise(item) => item.trace_gc(sink),
            Self::Cleanup(item) => item.trace_gc(sink),
        }
    }
}

/// Entry bookkeeping, not a new ECMAScript job for internal calls. No queue/kept clearing is
/// performed from Drop; a Rust unwind must not manufacture a host checkpoint.
struct Execution {
    heap: GcHeap,
    previous_depth: u32,
    pointer: NonNull<Interp>,
    frames: usize,
    code: usize,
    job_contexts: usize,
    using: usize,
    depth: u32,
    strict: bool,
    constructing: bool,
    new_target: Value,
    pending_new_target: Value,
}
impl Drop for Execution {
    fn drop(&mut self) {
        if std::thread::panicking() {
            // SAFETY: the entry descends from this local capability; all callback borrows end
            // before Drop. Discarding abandoned frames/resources is memory teardown, no author
            // finally/disposal/resumption runs here.
            let interp = unsafe { &mut *self.pointer.as_ptr() };
            interp.fn_frames.truncate(self.frames);
            interp.code_frames.truncate(self.code);
            interp.job_script_contexts.truncate(self.job_contexts);
            interp.using_stack.truncate(self.using);
            interp.pending_tail = None;
            interp.depth = self.depth;
            interp.strict = self.strict;
            interp.constructing = self.constructing;
            interp.new_target = self.new_target;
            interp.pending_new_target = self.pending_new_target;
        }
        self.heap.agent.execution_depth.set(self.previous_depth);
    }
}

struct Checkpoint(GcHeap);
impl Drop for Checkpoint {
    fn drop(&mut self) {
        self.0.agent.checkpoint.set(false);
    }
}

impl Interp {
    /// Host calls that can invoke author code use this boundary. Internal calls inherit the
    /// outer span, and a Ctx borrow scope itself does not begin one.
    pub(crate) fn with_author_entry<R>(&mut self, operation: impl FnOnce(&mut Interp) -> R) -> R {
        self.activate_gc_heap();
        // A host callback may catch a panic from a nested public call and continue the outer
        // invocation. Each public span therefore owns its own floor even while the Agent is
        // already executing. Ordinary internal/native calls do not pass through this API.
        self.with_routed_execution(operation)
    }

    /// A real cross-interpreter transfer records that target's stack floors too. A Rust
    /// unwind can abandon more than the outer root's frames; these guards restore in LIFO
    /// order without author disposal/resumption or inventing a checkpoint.
    fn with_routed_execution<R>(&mut self, operation: impl FnOnce(&mut Interp) -> R) -> R {
        let heap = self.gc_heap.clone();
        let previous_depth = heap.agent.execution_depth.replace(1);
        crate::agent_access::suspend_local(self, |pointer| {
            let interp = unsafe { &*pointer.as_ptr() };
            let _execution = Execution {
                heap,
                previous_depth,
                pointer,
                frames: interp.fn_frames.len(),
                code: interp.code_frames.len(),
                job_contexts: interp.job_script_contexts.len(),
                using: interp.using_stack.len(),
                depth: interp.depth,
                strict: interp.strict,
                constructing: interp.constructing,
                new_target: interp.new_target,
                pending_new_target: interp.pending_new_target,
            };
            operation(unsafe { &mut *pointer.as_ptr() })
        })
    }

    pub(crate) fn agent_checkpoint(
        &mut self,
        limit: Option<usize>,
    ) -> Result<usize, crate::InterruptReason> {
        let heap = self.gc_heap.clone();
        if self.realm_id != RealmId::ROOT
            || heap.agent.execution_depth.get() != 0
            || heap.agent.checkpoint.get()
        {
            heap.agent.checkpoint_requested.set(true);
            return Ok(0);
        }
        heap.agent.checkpoint.set(true);
        heap.agent.checkpoint_requested.set(false);
        let _checkpoint = Checkpoint(heap.clone());
        let result = self.pump_agent_work(limit, true);
        if result.is_err() {
            // Existing host interruption policy discards Promise continuations Agent-wide.
            // The distinct finalization lane/remaining held values are not cancelled here.
            heap.agent.work.borrow_mut().promises.clear();
        }
        if !matches!(&result, Ok((_, false))) {
            self.finish_agent_checkpoint();
        }
        result.map(|(completed, _)| completed)
    }

    /// A host's explicit synchronous drain from inside author code (Bun's `drainMicrotasks`):
    /// run the queued Promise jobs now. Unlike a checkpoint it does not end the outer job:
    /// KeptAlive, host `end_job`, cleanup tasks and the task-boundary collection stay with the
    /// outer checkpoint. A drain reached from a job of a running drain still defers.
    pub(crate) fn drain_promise_jobs_for_host(&mut self) {
        let heap = self.gc_heap.clone();
        if heap.agent.execution_depth.get() == 0 {
            self.drain_microtasks();
            return;
        }
        if heap.agent.checkpoint.get() {
            heap.agent.checkpoint_requested.set(true);
            return;
        }
        heap.agent.checkpoint.set(true);
        let _checkpoint = Checkpoint(heap.clone());
        if self.pump_agent_work(None, false).is_err() {
            heap.agent.work.borrow_mut().promises.clear();
        }
    }

    /// Run ready Agent work in FIFO order (Promise jobs, then cleanup tasks when `cleanup`),
    /// each through its own route. Returns the jobs completed and whether the pump may close
    /// its checkpoint.
    fn pump_agent_work(
        &mut self,
        limit: Option<usize>,
        cleanup: bool,
    ) -> Result<(usize, bool), crate::InterruptReason> {
        let heap = self.gc_heap.clone();
        self.with_access(|current| {
            let mut completed = 0;
            let mut settled = false;
            loop {
                if limit.is_some_and(|limit| completed >= limit) {
                    return Ok((completed, false));
                }
                settled |= current.settle_async_waits();
                let next = {
                    let mut work = heap.agent.work.borrow_mut();
                    work.promises
                        .pop_front()
                        .map(WorkItem::Promise)
                        .or_else(|| {
                            cleanup
                                .then(|| work.cleanup.pop_front().map(WorkItem::Cleanup))
                                .flatten()
                        })
                };
                let Some(next) = next else {
                    // A successful partial pump does not end the checkpoint. A subsequent
                    // empty pump (or an explicit full checkpoint) owns that boundary. Future
                    // waits/timers/imports are not ready work and do not hold it open.
                    let close = limit.is_none() || !settled;
                    return Ok((completed.max(usize::from(settled)), close));
                };
                // The queue borrow ended. Payload/path are roots before any poll or host hook.
                let next = Rooted::new(next);
                if let Err(crate::interpreter::Abrupt::Interrupt(reason)) =
                    current.interrupt_poll_force()
                {
                    return Err(reason);
                }
                let route = next.route().clone();
                assert_eq!(
                    route.agent,
                    crate::value::heap_id(&heap),
                    "job belongs to this Agent"
                );
                current.with_route(route.target, |target| {
                    target.with_author_entry(|target| match &*next {
                        WorkItem::Promise(item) => target.run_job_interruptible(&item.payload),
                        WorkItem::Cleanup(item) => target.run_prepared_cleanup(&item.payload),
                    })
                })?;
                completed += 1;
            }
        })
    }

    fn run_prepared_cleanup(&mut self, cleanup: &Cleanup) -> Result<(), crate::InterruptReason> {
        // Existing host task policy: release_host_job_context destroys tasks for that
        // registry's settings object. Merely inactive registry/callback/incumbent contexts
        // remain runnable. Callback preparation below is a separate identity (HTML e5071a20,
        // HostEnqueueFinalizationRegistryCleanupJob); the bare engine has no exception reporter.
        if self
            .retired_host_job_contexts
            .contains(&cleanup.registry_context)
        {
            return Ok(());
        }
        assert!(
            self.realms
                .contains_key(&(Gc::as_ptr(&cleanup.registry_realm) as usize)),
            "a retained registry task retains its creation Realm"
        );
        let body = |interp: &mut Interp| {
            interp.with_settings_context(cleanup.callback_context, |interp| {
                interp.run_finalization_cleanup_job(cleanup.registry)
            })
        };
        if let Some(key) = cleanup
            .callback_realm
            .filter(|key| *key != Gc::as_ptr(&self.global) as usize)
        {
            if let Some(target) = self
                .realms
                .get(&key)
                .map(crate::interpreter::RealmState::snapshot_clone)
            {
                return self.with_realm_snapshot(key, target, body);
            }
        }
        body(self)
    }

    /// The selected host policy is ClearKeptObjects once at the END of the outer checkpoint.
    /// HostGc::end_job is a no-JavaScript/no-collection hook. Its loans end before optional GC.
    fn finish_agent_checkpoint(&mut self) {
        let heap = self.gc_heap.clone();
        heap.agent.kept_alive.borrow_mut().clear();
        let access = heap.agent.access.clone();
        self.with_raw_access(|_| {
            for (_, pointer) in access.snapshot() {
                unsafe {
                    (&mut *pointer.as_ptr()).host_state.end_job();
                }
            }
        });
        self.gc_task_boundary();
    }

    pub(crate) fn cancel_agent_checkpoint(&mut self) {
        let heap = self.gc_heap.clone();
        if heap.agent.execution_depth.get() != 0 || heap.agent.checkpoint.get() {
            heap.agent.checkpoint_requested.set(true);
            return;
        }
        heap.agent.checkpoint.set(true);
        let _checkpoint = Checkpoint(heap.clone());
        heap.agent.work.borrow_mut().promises.clear();
        self.finish_agent_checkpoint();
    }
}
