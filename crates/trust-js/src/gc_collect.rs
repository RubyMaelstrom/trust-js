//! P1 S2: the interpreter's half of a gc-heap collection (feature `gc-heap`).
//!
//! Collections run only at interpreter safepoints — [`Interp::gc_check`], task boundaries,
//! buffer pressure and explicit requests — never on allocation. There the interpreter supplies
//! what the heap cannot see (Decision 2): its own fields, traced by an exhaustive
//! destructuring so that a new field does not compile until it is classified; the registered
//! VM frames and running coroutines (I1); owner-conditional side-table slots and Realm groups;
//! WeakMap ephemerons; and the graph its host reports. The native stack is scanned
//! conservatively (Decision 0).
//!
//! A ShadowRealm's interpreter shares its parent's heap: one heap per Agent, so no edge ever
//! crosses heaps (I6). Only the root interpreter collects, over its whole ShadowRealm tree; a
//! collection requested inside a ShadowRealm runs at the root's next safepoint. A realm's
//! roots belong to its ShadowRealm object, traced once that object is marked, so a realm kept
//! only by cycles through itself is collected; while code in a realm runs, its roots and its
//! ShadowRealm object are roots.

use super::*;
use crate::fasthash::{FastMap, FastSet};
use crate::gc_native::NativeGcGraph;
use crate::gc_trace::{GcSink, GcTrace};
use crate::host::{HostGcVisitor, NativeGcId};

/// The byte budget's floor: below half of it live, a collection is due once this much has been
/// allocated. Object counts also drive collection (see `gc_check`), but environment records
/// are cells, not objects, so this budget alone bounds scope churn. On the gate fixtures, S2's
/// tripwire measured peak RSS up to 6x higher at 64 MiB than at 8 MiB, with bytecode-tier time
/// unchanged (docs/v8-chase-2026-10-15.txt, S2 exit gate).
const GC_MIN_HEAP_BYTES: usize = 8 << 20;

/// Native stack zeroed below a collection's entry before marking (see `gc_core::scrub_stack`).
#[cfg(not(target_arch = "wasm32"))]
const GC_SCRUB_BYTES: usize = 256 * 1024;

/// Native stack zeroed before an explicit collection (see `Interp::gc_heap_collect_explicit`).
#[cfg(not(target_arch = "wasm32"))]
const EXPLICIT_SCRUB_BYTES: usize = 1 << 20;

/// `TRUST_JS_GC_STRESS=N`: collect at the first safepoint after every N cell allocations
/// (N = 1: at every safepoint that follows an allocation), and poll at every call.
pub(crate) fn gc_stress() -> usize {
    static STRESS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *STRESS.get_or_init(|| {
        std::env::var("TRUST_JS_GC_STRESS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0)
    })
}

/// `TRUST_JS_GC_MIN_HEAP_BYTES=N` (diagnostic): the byte-budget floor instead of
/// [`GC_MIN_HEAP_BYTES`], for collector tuning.
fn gc_min_heap_bytes() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("TRUST_JS_GC_MIN_HEAP_BYTES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(GC_MIN_HEAP_BYTES)
    })
}

/// Collection settings for a new Agent heap.
pub(crate) fn configure_heap(core: &mut crate::gc_core::Heap) {
    core.set_heap_growth(gc_min_heap_bytes(), 100);
    core.set_stress_every(gc_stress());
    // `TRUST_JS_GC_WHY=1`: record retention paths (see `gc_test::why`).
    if std::env::var_os("TRUST_JS_GC_WHY").is_some() {
        core.set_retention_tracing(true);
    }
}

struct WordSink<'a>(&'a mut dyn FnMut(u64));

impl GcSink for WordSink<'_> {
    #[inline]
    fn word(&mut self, word: u64) {
        (self.0)(word);
    }
}

fn value_word(value: &Value) -> Option<u64> {
    value.cell_word()
}

/// One interpreter's host graph for this collection: what its `HostGc` slots report.
#[derive(Default)]
struct HostGraph {
    roots: Vec<u64>,
    /// Logical edges: each value is reachable while its owner is.
    edges: Vec<(usize, u64)>,
    native: NativeGcGraph,
}

impl HostGcVisitor for HostGraph {
    fn supports_native(&self) -> bool {
        true
    }

    fn native_edge(&mut self, from: NativeGcId, to: NativeGcId) {
        self.native.native_edge(from, to);
    }

    fn js_to_native(&mut self, owner: &Value, to: NativeGcId) {
        self.native.js_to_native(owner, to);
    }

    fn native_to_js(&mut self, from: NativeGcId, value: &Value) {
        self.native.native_to_js(from, value);
    }

    fn native_root(&mut self, id: NativeGcId) {
        self.native.native_root(id);
    }

    /// A cache handle lives only through the edges and roots reported for it; the host's
    /// sweep drops it once its value dies.
    fn internal(&mut self, _: &Value) {}

    fn edge(&mut self, owner: &Value, value: &Value) {
        if let (Value::Obj(owner), Some(value)) = (owner, value_word(value)) {
            self.edges.push((Gc::as_ptr(owner) as usize, value));
        }
    }

    fn root(&mut self, value: &Value) {
        if let Some(word) = value_word(value) {
            self.roots.push(word);
        }
    }
}

/// The [`crate::gc_core::MarkClient`] of one collection over a ShadowRealm tree.
struct Marking<'a> {
    /// The root interpreter, then its ShadowRealm descendants breadth-first.
    interps: Vec<&'a Interp>,
    /// Parallel to `interps`.
    hosts: Vec<RefCell<HostGraph>>,
    /// ShadowRealm object -> the index of its realm's interpreter.
    shadow_owners: FastMap<usize, usize>,
    /// Realm-group members (objects and global environments) → (interpreter, Realm key). A
    /// Realm is not a root: marking any member activates the whole group, as in the old
    /// collector, so a retired Realm is reclaimed with everything it alone keeps alive.
    realm_members: FastMap<usize, Vec<(usize, usize)>>,
    /// GetTemplateObject's per-Realm [[TemplateMap]] values, by (interpreter, Realm key).
    templates: FastMap<(usize, usize), Vec<u64>>,
    activated_realms: RefCell<FastSet<(usize, usize)>>,
}

impl<'a> Marking<'a> {
    /// Out of line: its frames must be dead (and scrubbed) before the native stack is scanned.
    #[inline(never)]
    fn new(root: &'a Interp) -> Self {
        let mut interps = vec![root];
        let mut shadow_owners = FastMap::default();
        let mut next = 0;
        while let Some(&interp) = interps.get(next) {
            for (&owner, child) in &interp.shadow_realms {
                shadow_owners.insert(owner, interps.len());
                interps.push(&**child);
            }
            next += 1;
        }
        let mut hosts = Vec::with_capacity(interps.len());
        let mut realm_members: FastMap<usize, Vec<(usize, usize)>> = FastMap::default();
        let mut templates: FastMap<(usize, usize), Vec<u64>> = FastMap::default();
        for (index, interp) in interps.iter().enumerate() {
            let mut host = HostGraph::default();
            interp.host_state.trace_gc(&mut host);
            hosts.push(RefCell::new(host));
            for (&key, realm) in &interp.realms {
                // A RealmState reports exactly its members.
                realm.trace_gc(&mut WordSink(&mut |member| {
                    realm_members
                        .entry(member as usize)
                        .or_default()
                        .push((index, key));
                }));
            }
            for (&(realm, _), value) in &interp.template_cache {
                if let Some(word) = value_word(value) {
                    templates.entry((index, realm)).or_default().push(word);
                }
            }
        }
        Marking {
            interps,
            hosts,
            shadow_owners,
            realm_members,
            templates,
            activated_realms: RefCell::default(),
        }
    }

    /// Each interpreter's native graph, by interpreter address, for the host sweeps.
    fn into_native_graphs(self) -> FastMap<usize, NativeGcGraph> {
        self.interps
            .iter()
            .zip(self.hosts)
            .map(|(&interp, host)| {
                (
                    std::ptr::from_ref(interp) as usize,
                    host.into_inner().native,
                )
            })
            .collect()
    }
}

impl Marking<'_> {
    /// The roots of interpreter `index` and of its host.
    fn trace_interp_roots(&self, index: usize, words: &mut WordSink<'_>) {
        self.interps[index].trace_gc_roots(words);
        let mut host = self.hosts[index].borrow_mut();
        for &root in &host.roots {
            words.word(root);
        }
        let mut reached = Vec::new();
        host.native.trace_roots(&mut reached);
        for target in reached {
            words.word(target as u64);
        }
    }
}

impl crate::gc_core::MarkClient for Marking<'_> {
    fn roots(&self, sink: &mut dyn FnMut(u64)) {
        let mut words = WordSink(sink);
        self.trace_interp_roots(0, &mut words);
        // A realm with code running: its roots, and its ShadowRealm object, which the sweep
        // must not take from under it. Other realms' roots are their objects' side slots.
        for (&owner, &index) in &self.shadow_owners {
            if self.interps[index].shadow_entered() {
                words.word(owner as u64);
                self.trace_interp_roots(index, &mut words);
            }
        }
        // The tree shares one heap; its interned strings live as long as it does.
        if let Some(root) = self.interps.first() {
            crate::value::trace_interned_strings(&root.gc_heap, &mut words);
        }
        crate::gc_trace::trace_thread_roots(&mut words);
    }

    fn side_slot_owners(&self, owners: &mut dyn FnMut(usize)) {
        // Collected first: the heap may trace an already-marked owner's slots at once, which
        // re-enters this client.
        let mut seen = FastSet::default();
        let mut list = Vec::new();
        let mut note = |owner: usize| {
            if seen.insert(owner) {
                list.push(owner);
            }
        };
        for interp in &self.interps {
            interp.for_each_side_slot_owner(&mut note);
        }
        for &owner in self.shadow_owners.keys() {
            note(owner);
        }
        for &member in self.realm_members.keys() {
            note(member);
        }
        for host in &self.hosts {
            for owner in host.borrow().native.js_owners() {
                note(owner);
            }
        }
        for owner in list {
            owners(owner);
        }
    }

    fn trace_side_slots(&self, owner: usize, sink: &mut dyn FnMut(u64)) {
        let mut words = WordSink(sink);
        if let Some(&index) = self.shadow_owners.get(&owner) {
            // An entered realm's roots were traced as roots.
            if !self.interps[index].shadow_entered() {
                self.trace_interp_roots(index, &mut words);
            }
        }
        for interp in &self.interps {
            interp.trace_owner_slots(owner, &mut words);
        }
        for &(index, key) in self.realm_members.get(&owner).into_iter().flatten() {
            if !self.activated_realms.borrow_mut().insert((index, key)) {
                continue;
            }
            if let Some(realm) = self.interps[index].realms.get(&key) {
                realm.trace_gc(&mut words);
            }
            for &template in self.templates.get(&(index, key)).into_iter().flatten() {
                words.word(template);
            }
        }
        let mut reached = Vec::new();
        for host in &self.hosts {
            host.borrow_mut().native.trace_js(owner, &mut reached);
            for target in reached.drain(..) {
                words.word(target as u64);
            }
        }
    }

    fn ephemerons(&self, sink: &mut crate::gc_core::EdgeSink<'_>) {
        for (interp, host) in self.interps.iter().zip(&self.hosts) {
            for (&owner, entries) in &interp.weak_collection_data {
                for (key, value) in entries {
                    let Some(value) = value_word(value) else {
                        continue;
                    };
                    // An object or Symbol key: the value lives while both the collection and
                    // the key do.
                    if let Some(key) = key.upgrade().as_ref().and_then(Value::cell_word) {
                        sink.ephemeron(owner as *const u8, key, value);
                    }
                }
            }
            // [[Realm]] is a logical edge: a live built-in keeps its Realm (keyed by the
            // Realm's global object) alive, however its prototype has been rewired.
            for (&function, &(_, realm)) in interp.native_function_realms.borrow().iter() {
                sink.edge(function as *const u8, realm as u64);
            }
            for &(owner, value) in &host.borrow().edges {
                sink.edge(owner as *const u8, value);
            }
        }
    }
}

impl Interp {
    /// Whether this interpreter collects its heap. A ShadowRealm's interpreter shares its
    /// parent's heap, and only the root of that tree collects it.
    pub(crate) fn gc_heap_root(&self) -> bool {
        self.heap_release.owns_heap()
    }

    /// Whether a collection should run at this safepoint: object pressure (the old policy's
    /// threshold, `gc_next`), the heap's byte budget or stress interval, or a deferred request.
    pub(crate) fn gc_heap_due(&self) -> bool {
        crate::value::heap_live_objects(&self.gc_heap) > self.gc_next
            || self.gc_heap.core.borrow().collection_due()
    }

    /// An explicit collection (a host or test request). It usually comes from just above a
    /// finished job, whose frames left stale cell addresses in the stack the collection's own
    /// frames are about to reuse. Zero that region first, from the caller's frame (inlined,
    /// see `Interp::gc_collect`).
    #[inline(always)]
    pub(crate) fn gc_heap_collect_explicit(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        crate::gc_core::scrub_stack(EXPLICIT_SCRUB_BYTES);
        self.gc_heap_collect(crate::value::GcCause::Explicit);
    }

    /// Collect the Agent heap (a no-op in a ShadowRealm's interpreter). Defers, leaving the
    /// request pending for the next safepoint, inside a no-GC region, while a cell borrow is
    /// outstanding or on a native stack segment the scanner cannot see.
    pub(crate) fn gc_heap_collect(&mut self, cause: crate::value::GcCause) {
        if !self.gc_heap_root() {
            return;
        }
        let heap = self.gc_heap.clone();
        if heap.core.try_borrow_mut().is_err() {
            // Requested from inside this collection (a host sweep or Drop): it is running.
            return;
        }
        let performance_started = crate::value::gc_performance_metrics_start();
        let objects_before = crate::value::heap_live_objects(&heap);
        // Frames that are live while the native stack is scanned — this one and those of
        // `gc_heap_mark` — must hold no stale cell addresses: a dead cell's address may be
        // reused by the time of the next collection. Sweeping, which handles every dead
        // address, runs in a sibling call, and the region both calls use is zeroed first.
        #[cfg(not(target_arch = "wasm32"))]
        crate::gc_core::scrub_stack(GC_SCRUB_BYTES);
        let Some(native) = self.gc_heap_mark(&heap) else {
            return;
        };
        let reclaimed = self.gc_heap_sweep(&heap, native);
        if crate::value::layout_census::enabled() {
            crate::value::layout_census::sample_live(&heap);
        }
        crate::value::gc_finish_generation(&heap);
        let live = crate::value::heap_live_objects(&heap);
        let next = next_threshold(live, self.live_object_limit);
        // Over the live-object limit, keep the room gc_check re-armed for the handler of the
        // limit's RangeError: a stress or byte-budget collection must not take it away.
        self.gc_next = if live > self.live_object_limit {
            self.gc_next.max(next)
        } else {
            next
        };
        if std::env::var_os("TRUST_JS_GC_LOG").is_some() {
            let stats = heap.core.borrow().stats();
            eprintln!(
                "[gc-heap] cause={cause:?} objects={objects_before}->{live} cells={} bytes={} pages={} spare={} collections={} deferred={}",
                stats.live_cells,
                stats.live_bytes,
                stats.pages_live,
                stats.pages_spare,
                stats.collections,
                stats.deferred_collections
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        if reclaimed >= 50_000 {
            crate::fastalloc::trim();
        }
        if let Some(started) = performance_started {
            crate::value::gc_performance_metrics_finish(
                started,
                objects_before.max(0) as usize,
                live,
                cause,
            );
        }
    }

    /// Phase 1: mark from this tree's roots, side slots, ephemerons and the native stack.
    /// `None` when the heap deferred the collection.
    #[inline(never)]
    fn gc_heap_mark(
        &mut self,
        heap: &crate::value::GcHeap,
    ) -> Option<FastMap<usize, NativeGcGraph>> {
        let marking = Marking::new(self);
        // Building the client just enumerated every Realm member and host value, dead or
        // alive, in frames below this one; the marker's frames reuse that region.
        #[cfg(not(target_arch = "wasm32"))]
        crate::gc_core::scrub_stack(GC_SCRUB_BYTES);
        let started = heap.core.borrow_mut().start_marking(true, &marking);
        started.then(|| marking.into_native_graphs())
    }

    /// Phase 2: evict the dead from the engine's tables, clear weak handles and free the dead
    /// cells. Returns how many objects died.
    #[inline(never)]
    fn gc_heap_sweep(
        &mut self,
        heap: &crate::value::GcHeap,
        native: FastMap<usize, NativeGcGraph>,
    ) -> usize {
        // Between the phases every cell is intact and new cells are allocated marked: the
        // engine evicts the dead from its tables while their identities are still stable.
        let garbage: FastSet<usize> = heap
            .core
            .borrow()
            .dead_cells(heap.object_desc)
            .into_iter()
            .collect();
        self.gc_heap_sweep_tree(&garbage, &native);
        drop(native);
        // Weak handles to dead cells are cleared; a cell a weak handle still names stays
        // reserved (finalized, never reused) until its last weak handle is dropped, which keeps
        // raw-address inline-cache comparisons ABA-safe.
        let pinned = {
            let core = heap.core.borrow();
            crate::gc_handle::clear_dead_weak_cells(|addr| core.is_dead_target(addr))
        };
        heap.core
            .borrow_mut()
            .finish_collection(&|addr| pinned.contains(&addr));
        self.gc_heap_after_sweep(&garbage);
        garbage.len()
    }

    /// Evict the dead from every interpreter's tables in this ShadowRealm tree. A dead
    /// ShadowRealm's interpreter is dropped here, while the cells it names are still intact.
    fn gc_heap_sweep_tree(
        &mut self,
        garbage: &FastSet<usize>,
        native: &FastMap<usize, NativeGcGraph>,
    ) {
        let empty = NativeGcGraph::default();
        let graph = native
            .get(&(std::ptr::from_ref(self) as usize))
            .unwrap_or(&empty);
        self.gc_sweep_dead_objects(garbage, graph);
        for child in self.shadow_realms.values_mut() {
            child.gc_heap_sweep_tree(garbage, native);
        }
    }

    /// Post-sweep bookkeeping for every interpreter in the tree: drop expired inline-cache
    /// pins, deliver destruction notifications and schedule FinalizationRegistry cleanup.
    fn gc_heap_after_sweep(&mut self, garbage: &FastSet<usize>) {
        self.gc_prune_dead_pins();
        self.gc_drain_weak_deaths();
        self.gc_schedule_finalization_cleanup(garbage);
        self.gc_task_live = crate::value::heap_live_objects(&self.gc_heap);
        self.gc_task_pending = false;
        self.buffer_gc.since_gc = 0;
        for child in self.shadow_realms.values_mut() {
            child.gc_heap_after_sweep(garbage);
        }
    }

    /// Owners of this interpreter's owner-keyed side-table slots (see `trace_owner_slots`).
    fn for_each_side_slot_owner(&self, owner: &mut dyn FnMut(usize)) {
        macro_rules! owners {
            ($($table:ident),+ $(,)?) => {
                $(for &key in self.$table.keys() {
                    owner(key);
                })+
            };
        }
        gc_side_slot_tables!(owners);
        for &key in self.generators.keys() {
            owner(key);
        }
    }

    /// The strong edges `owner` holds through this interpreter's owner-keyed side tables,
    /// traced once the owner is marked. A suspended generator's continuation belongs to its
    /// generator object the same way.
    fn trace_owner_slots(&self, owner: usize, sink: &mut dyn GcSink) {
        macro_rules! slots {
            ($($table:ident),+ $(,)?) => {
                $(if let Some(slots) = self.$table.get(&owner) {
                    slots.trace_gc(sink);
                })+
            };
        }
        gc_side_slot_tables!(slots);
        if let Some(coroutine) = self.generators.get(&owner) {
            GcTrace::trace_gc(coroutine, sink);
        }
    }

    /// Every strong root this interpreter holds itself. Owner-keyed side tables, Realm groups,
    /// weak tables and caches are not roots; each `_` below says why. ShadowRealm
    /// interpreters are traced as members of the collection's tree, not through this field.
    fn trace_gc_roots(&self, sink: &mut dyn GcSink) {
        let Interp {
            gc_heap: _,
            symbol_agent,
            global,
            global_env,
            object_proto,
            function_proto,
            array_proto,
            array_ctor,
            string_proto,
            number_proto,
            boolean_proto,
            symbol_proto,
            error_protos,
            console: _,
            wall_clock: _,
            runtime_interrupt: _,
            interrupt_poll_tick: _,
            ast_tick: _,
            strict: _,
            tier: _,
            tier_threshold: _,
            jit_admit_runs: _,
            tagged_arithmetic: _,
            vm_pool,
            native_arg_pool,
            stub_cache: _,       // shapes and slots
            stub_cache_names: _, // property names
            arguments_shapes: _,
            descriptor_maps,
            iter_result_map,
            computed_reads,
            fragment_cache: _, // weak Realm keys and compiled plans
            enumeration_keys,
            property_key_strings,
            frame_pool, // the active JIT records (Rust entries' contexts: thread roots)
            creation_pins: _, // weak pins
            global_env_pins: _, // weak pins
            call_site: _,
            code_frames,
            job_script_contexts: _,
            host_resolution_marker,
            jit_helpers: _,
            jit_layout: _,
            interp_layout: _,
            inline_ic_safe: _,
            str_units,
            re_texts,
            re_text_ascii_hot,
            regexp_dependency_cache: _,
            regexp_last,
            depth: _,
            direct_call_depth: _,
            proxy_forward_depth: _,
            class_info: _, // side slots
            eval_fn,
            eval_realm_fns: _,         // addresses
            native_function_realms: _, // weak pins; [[Realm]] edges are reported as ephemerons
            elems_protector: _,
            species_protector: _,
            ta_accessor_protector: _,     // weak pins
            array_iteration_protector: _, // weak pins
            construct_ics: _,             // weak pins
            call_overflow: _,             // weak pins
            native_callback_cache: _,     // weak pins
            proxy_trap_caches: _,         // weak pins
            self_hosted: _,
            shared_snapshots: _,         // parsed source
            construct_capacity_hints: _, // weak pins
            iterator_sym,
            wk_syms,
            short_circuit: _,
            import_meta,
            import_base: _,
            modules,
            module_recs,
            module_loader: _,
            dynamic_module_loader: _,
            pending_dynamic_imports,
            next_dynamic_import_id: _,
            module_ns: _,            // side slots
            collection_iterators: _, // side slots
            weak_collection_data: _, // ephemerons
            weak_collection_index: _,
            extra_protos,
            array_buffers: _,
            native_buffers: _,
            array_buffer_versions: _,
            array_buffer_dirty_ranges: _,
            immutable_buffers: _,
            host_keyed_buffers: _,
            can_block: _,
            pending_async_waits,
            async_wait_inbox: _,
            pending_timers,
            agent: _,
            typed_arrays: _,
            async_gens: _,
            global_var_names: _,
            gc_pins: _,       // bookkeeping for side-table owners, not holders
            ta_buffer: _,     // side slots
            shadow_realms: _, // members of the collection's tree, traced as side slots
            shadow_home: _,   // a weak owner; its entry count is read by `Marking`
            data_views: _,
            regexps: _,
            regexp_programs: _,
            proxies: _,      // side slots
            host_indexed: _, // side slots
            new_target,
            pending_new_target,
            htmldda: _, // weak pins
            ctor_caller_realm,
            realms: _,   // Realm groups
            promises: _, // side slots
            unhandled_rejections,
            next_unhandled_rejection: _,
            handled_rejections,
            temporal: _,
            temporal_cal: _,
            microtasks,
            host_job_context: _,
            script_entry: _,
            host_entry_realm: _,
            capture_job_script_caller: _,
            host_settings_states,
            retired_host_job_contexts: _,
            host_job_context_enter: _,
            host_job_context_leave: _,
            kept_alive,
            host_state: _, // reported through its HostGc visitors
            generators: _, // side slots
            gc_next: _,
            live_object_limit: _,
            gc_suppressed: _,
            buffer_gc: _,
            gc_tick: _,
            gc_task_allocated: _,
            gc_task_live: _,
            gc_task_deferred: _,
            gc_task_pending: _,
            gc_task_major: _,
            scope_gc_next: _,
            constructing: _,
            super_call_ok: _,
            fn_frames,
            module_async_seq: _,
            yield_raw_result: _,
            in_async_gen_body: _,
            pending_fn_name: _,
            pending_tail,
            tco_ok: _,
            template_cache: _, // Realm groups
            in_field_init_code: _,
            using_stack,
            accessor_seq: _,
            decorator_initializers,
            annexb_fn_sync: _, // parsed functions
            deferred_ns: _,
            deferred_ns_objs,
            promise_forward: _,  // side slots
            mapped_arguments: _, // side slots
            borrowed_parameters,
            module_source_objs,
            weak_refs: _,               // weak
            weak_metadata: _,           // weak
            finalization_registries: _, // side slots
            pending_finalization_cleanup,
            async_gen_busy: _,
            async_gen_queue: _, // side slots
            heap_release: _,
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
            host_resolution_marker,
        ] {
            sink.object(object);
        }
        global_env.trace_gc(sink);
        array_ctor.trace_gc(sink);
        error_protos.trace_gc(sink);
        eval_fn.trace_gc(sink);
        extra_protos.trace_gc(sink);
        // Pooled buffers are empty; one in use is registered by its activation (`drive_vm`, or
        // a JIT entry's `gc_trace::JitFrameRoots`).
        for (slots, stack) in vm_pool {
            slots.trace_gc(sink);
            stack.trace_gc(sink);
        }
        frame_pool.trace_gc(sink);
        native_arg_pool.trace_gc(sink);
        for map in descriptor_maps.iter().chain([iter_result_map]) {
            if let Some(props) = map.get() {
                props.trace_gc(sink);
            }
        }
        for keys in enumeration_keys.values() {
            keys.trace_gc(sink);
        }
        for (_, symbol, _) in wk_syms {
            symbol.trace_gc(sink);
        }
        if let Some(symbol) = iterator_sym {
            sink.word(crate::value::symbol_identity(symbol) as u64);
        }
        // String caches keyed by a string's address pin it: a pin keeps the address from being
        // reused while an entry is keyed by it.
        computed_reads.trace_pins(sink);
        for (_, string) in property_key_strings.values() {
            string.trace_gc(sink);
        }
        for (string, _, _) in str_units.values() {
            string.trace_gc(sink);
        }
        for (string, text) in re_texts.values() {
            string.trace_gc(sink);
            text.trace_gc(sink);
        }
        if let Some((string, _, text)) = re_text_ascii_hot {
            string.trace_gc(sink);
            text.trace_gc(sink);
        }
        if let Some(last) = regexp_last {
            last.input.trace_gc(sink);
            last.text.trace_gc(sink);
        }
        code_frames.trace_gc(sink);
        // The Agent's registered (Symbol.for) and well-known Symbols live as long as it does.
        {
            let agent = symbol_agent.borrow();
            for symbol in agent
                .global_by_key
                .values()
                .chain(agent.well_known.values())
            {
                sink.word(crate::value::symbol_identity(symbol) as u64);
            }
        }
        import_meta.trace_gc(sink);
        modules.trace_gc(sink);
        module_recs.trace_gc(sink);
        pending_dynamic_imports.trace_gc(sink);
        pending_async_waits.trace_gc(sink);
        for (callback, _) in pending_timers {
            callback.trace_gc(sink);
        }
        new_target.trace_gc(sink);
        pending_new_target.trace_gc(sink);
        ctor_caller_realm.trace_gc(sink);
        for (_, promise, reason, _) in unhandled_rejections.values() {
            promise.trace_gc(sink);
            reason.trace_gc(sink);
        }
        for (promise, reason, _) in handled_rejections {
            promise.trace_gc(sink);
            reason.trace_gc(sink);
        }
        microtasks.trace_gc(sink);
        host_settings_states.trace_gc(sink);
        kept_alive.trace_gc(sink);
        fn_frames.trace_gc(sink);
        if let Some(tail) = pending_tail {
            let (callee, this, arguments) = &**tail;
            callee.trace_gc(sink);
            this.trace_gc(sink);
            arguments.trace_gc(sink);
        }
        using_stack.trace_gc(sink);
        decorator_initializers.trace_gc(sink);
        deferred_ns_objs.trace_gc(sink);
        borrowed_parameters.trace_gc(sink);
        module_source_objs.trace_gc(sink);
        pending_finalization_cleanup.trace_gc(sink);
    }
}

#[cfg(test)]
mod tests {
    use crate::bytecode::Tier;
    use crate::value::Value;
    use crate::{Completion, Engine};

    fn eval(engine: &mut Engine, source: &str, expected: &str) {
        match engine.eval(source, false).expect("GC fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected),
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    #[test]
    fn collection_keeps_live_call_frames_scopes_weak_targets_and_side_tables_all_tiers() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            engine
                .interp
                .def_method(&engine.interp.global, "collect", 0, |interp, _, _| {
                    interp.gc_collect();
                    Ok(Value::Undefined)
                });
            engine.interp.gc_collect();
            eval(
                &mut engine,
                r#"
                function factory(n) {
                    let o = {n:n};
                    let map = new Map([[o, o]]), set = new Set([o]);
                    let weak = new WeakRef(o), weakMap = new WeakMap([[o, {owner:o}]]);
                    let bytes = new Uint8Array([n]);
                    function closure() {
                        collect();
                        return [o.n, map.get(o).n, set.has(o), weak.deref() === o,
                            weakMap.get(o).owner === o, bytes[0]].join(',');
                    }
                    collect();
                    return closure;
                }
                var closures = [];
                for (var i=0; i<20; i++) closures.push(factory(i));
                closures[7]() + ':' + closures[19]()
            "#,
                "7,7,true,true,true,7:19,19,true,true,true,19",
            );
            engine.interp.gc_collect();
            eval(&mut engine, "closures[3]()", "3,3,true,true,true,3");
        }
    }

    #[test]
    fn finalizer_targets_clear_together_and_ready_cells_are_removed() {
        let mut engine = Engine::new();
        engine.interp.gc_collect();
        eval(
            &mut engine,
            r#"
            var finalized = [], first, second;
            var registry = new FinalizationRegistry(function(value) { finalized.push(value); });
            (function() {
                var target = {}; target.self = target;
                first = new WeakRef(target); second = new WeakRef(target);
                registry.register(target, 91, target);
            })(); true
        "#,
            "true",
        );
        engine.interp.gc_collect();
        eval(&mut engine, "true", "true");
        eval(&mut engine, "[first.deref() === undefined, second.deref() === undefined, finalized.join(',')].join(':')", "true:true:91");
    }

    /// Owners of every internal-slot family keep their table edges across a collection, and
    /// cycles that pass only through those tables (or a suspended coroutine) are reclaimed: with
    /// few table entries, and beside 4,000 older Maps' entries, on every tier.
    #[test]
    fn side_slot_owners_keep_table_edges_with_sparse_and_dense_tables() {
        for dense_old_tables in [false, true] {
            for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
                let mut engine = Engine::new();
                engine.set_tier(tier);
                engine.set_tier_threshold(0);
                if dense_old_tables {
                    eval(
                        &mut engine,
                        "var oldMaps = []; for (var i = 0; i < 4000; i++) oldMaps.push(new Map([[i, {i}]])); true",
                        "true",
                    );
                }
                engine.interp.gc_collect();
                eval(
                    &mut engine,
                    r#"
                    var weak = [];
                    var keep = (function () {
                        var m = new Map(), key = {k: 1}; m.set(key, {v: 2});
                        var s = new Set([{s: 3}]);
                        var proxy = new Proxy({t: 5}, {});
                        function* gen() { var held = {g: 6}; yield held.g; yield held.g + 1; }
                        var it = gen(); it.next();
                        class C { x = {f: 7}; #p = {q: 9}; q() { return this.#p.q; } }
                        var args = (function (a) { return arguments; })({a: 8});
                        var pending = new Promise(function () {});
                        var reaction = {r: 4};
                        pending.then(function () { return reaction; });
                        var ta = new Uint8Array([10]);
                        return {m, key, s, proxy, it, C, args, ta};
                    })();
                    (function () {
                        var cyclic = new Map(); cyclic.set('self', cyclic);
                        var set = new Set(); set.add(set);
                        var it2 = (function* () { var me = it2; yield 1; })(); it2.next();
                        var target = {}; var p = new Proxy(target, {h: target}); target.p = p;
                        weak.push(new WeakRef(cyclic), new WeakRef(set), new WeakRef(it2),
                                  new WeakRef(target));
                    })();
                    true
                "#,
                    "true",
                );
                engine.interp.gc_collect();
                eval(&mut engine, "true", "true");
                eval(
                    &mut engine,
                    r#"[keep.m.get(keep.key).v, [...keep.s][0].s, keep.proxy.t,
                        keep.it.next().value, new keep.C().x.f, new keep.C().q(),
                        keep.args[0].a, keep.ta[0],
                        weak.map(function (w) { return w.deref() === undefined; }).join('/')
                       ].join(',')"#,
                    "2,3,5,7,7,9,8,10,true/true/true/true",
                );
            }
        }
    }
}
