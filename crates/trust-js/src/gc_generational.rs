//! Non-moving nursery collection for the production object/environment heap.
//!
//! The existing Rc ownership boundary is also a complete, conservative remembered set:
//! references from old objects, native stacks and embedding code exceed a young node's
//! nursery-internal reference count. This collects young cycles without traversing the old
//! object/property graph, and cannot miss an old-to-young write made outside generated code.
//! Native logical edges still participate through HostGc. Old ephemeron/Realm tables are
//! conservatively retained until a bounded periodic full collection; explicit and host idle
//! collections always use the full ephemeron fixed point.
//!
//! ECMA-262 §9.9.2–3 (#sec-liveness, #sec-weakref-execution), snapshot e28783d5:
//! choosing a non-maximal non-live set is permitted, but its weak edges must be cleared
//! atomically. Both collectors use the same weak clearing and internal-slot sweep.

use crate::fasthash::{FastMap, FastSet};
use crate::gc_edges::GcEdgeSink;
use crate::host::HostGcVisitor;
use crate::interpreter::{Env, Interp};
use crate::value::{Callable, Exotic, Gc, Value};

pub(crate) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("TRUST_JS_GC_GENERATIONAL").as_deref() != Ok("0"))
}

pub(crate) fn next_threshold(live: i64, limit: i64) -> i64 {
    let next = if enabled() {
        live.saturating_add((live / 8).max(crate::interpreter::GC_TRIGGER))
    } else {
        live.saturating_mul(2)
    };
    next.clamp(
        crate::interpreter::GC_TRIGGER,
        limit.max(crate::interpreter::GC_TRIGGER),
    )
}

#[derive(Clone, Copy)]
enum Node {
    Object(usize),
    Scope(usize),
}

/// Marks an environment-record target in [`Nursery::edges`]; object targets are plain indices.
const SCOPE_EDGE: u32 = 1 << 31;

struct Nursery {
    objects: Vec<Gc>,
    scopes: Vec<Env>,
    object_index: FastMap<usize, usize>,
    scope_index: FastMap<usize, usize>,
    object_internal: Vec<usize>,
    scope_internal: Vec<usize>,
    object_mark: Vec<bool>,
    scope_mark: Vec<bool>,
    pending: Vec<Node>,
    host_edges: FastMap<usize, Vec<usize>>,
    native: crate::gc_native::NativeGcGraph,
    /// Young-to-young strong edges found while counting internal references, replayed while
    /// marking. No graph mutation or JS runs in between, so both passes see the same graph;
    /// only identities are stored (never extra Rc owners). Node `k` (objects first, then
    /// scopes) owns `edges[edge_starts[k]..edge_starts[k + 1]]`.
    edge_starts: Vec<usize>,
    edges: Vec<u32>,
}

impl Nursery {
    fn new(objects: Vec<Gc>, scopes: Vec<Env>) -> Self {
        Self {
            object_index: objects
                .iter()
                .enumerate()
                .map(|(i, o)| (Gc::as_ptr(o) as usize, i))
                .collect(),
            scope_index: scopes
                .iter()
                .enumerate()
                .map(|(i, s)| (Env::as_ptr(s) as usize, i))
                .collect(),
            object_internal: vec![0; objects.len()],
            scope_internal: vec![0; scopes.len()],
            object_mark: vec![false; objects.len()],
            scope_mark: vec![false; scopes.len()],
            pending: Vec::new(),
            host_edges: FastMap::default(),
            native: crate::gc_native::NativeGcGraph::default(),
            edge_starts: Vec::with_capacity(objects.len() + scopes.len() + 1),
            edges: Vec::new(),
            objects,
            scopes,
        }
    }

    fn mark(&mut self, node: Node) {
        let marked = match node {
            Node::Object(index) => &mut self.object_mark[index],
            Node::Scope(index) => &mut self.scope_mark[index],
        };
        if !*marked {
            *marked = true;
            self.pending.push(node);
        }
    }

    /// Count and record young node `index`'s edges (objects first, then scopes). Edges to old
    /// or foreign nodes are irrelevant to a minor collection: those nodes are not candidates.
    fn count_object(
        &mut self,
        interp: &Interp,
        index: usize,
        side_slots: bool,
        captures: &mut crate::native_captures::NativeCaptureSnapshot,
        scratch: (&mut Vec<Gc>, &mut Vec<Env>),
    ) {
        self.edge_starts.push(self.edges.len());
        let object = &self.objects[index];
        captures.observe(object, &object.borrow());
        let mut recorder = YoungEdges {
            object_index: &self.object_index,
            scope_index: &self.scope_index,
            object_internal: &mut self.object_internal,
            scope_internal: &mut self.scope_internal,
            edges: &mut self.edges,
        };
        interp.visit_object_edges(object, side_slots, &mut recorder);
        if !side_slots {
            return;
        }
        if let Some(coroutine) = interp.generators.get(&(Gc::as_ptr(object) as usize)) {
            let (objects, scopes) = scratch;
            coroutine.trace_gc(&mut crate::gc_edges::DirectGcEdges { objects, scopes });
            for object in objects.drain(..) {
                recorder.object(&object);
            }
            for scope in scopes.drain(..) {
                recorder.scope(&scope);
            }
        }
    }

    fn count_scope(&mut self, index: usize) {
        self.edge_starts.push(self.edges.len());
        let mut recorder = YoungEdges {
            object_index: &self.object_index,
            scope_index: &self.scope_index,
            object_internal: &mut self.object_internal,
            scope_internal: &mut self.scope_internal,
            edges: &mut self.edges,
        };
        visit_scope_edges(&self.scopes[index], &mut recorder);
    }

    /// Mark the young targets recorded for node `k`.
    fn mark_recorded(&mut self, k: usize) {
        let (start, end) = (self.edge_starts[k], self.edge_starts[k + 1]);
        for slot in start..end {
            let target = self.edges[slot];
            self.mark(if target & SCOPE_EDGE != 0 {
                Node::Scope((target & !SCOPE_EDGE) as usize)
            } else {
                Node::Object(target as usize)
            });
        }
    }
}

/// Counts internal references to young nodes and records the edge for marking.
struct YoungEdges<'a> {
    object_index: &'a FastMap<usize, usize>,
    scope_index: &'a FastMap<usize, usize>,
    object_internal: &'a mut [usize],
    scope_internal: &'a mut [usize],
    edges: &'a mut Vec<u32>,
}

impl GcEdgeSink for YoungEdges<'_> {
    #[inline]
    fn object(&mut self, object: &Gc) {
        if let Some(&index) = self.object_index.get(&(Gc::as_ptr(object) as usize)) {
            self.object_internal[index] += 1;
            self.edges.push(index as u32);
        }
    }

    #[inline]
    fn scope(&mut self, scope: &Env) {
        if let Some(&index) = self.scope_index.get(&(Env::as_ptr(scope) as usize)) {
            self.scope_internal[index] += 1;
            self.edges.push(index as u32 | SCOPE_EDGE);
        }
    }
}

impl HostGcVisitor for Nursery {
    fn is_minor(&self) -> bool {
        true
    }
    fn native_old_generation(
        &mut self,
        domain: &'static str,
        owner: usize,
        first_young_id: usize,
    ) -> bool {
        self.native
            .declare_old_generation(domain, owner, first_young_id);
        true
    }
    fn supports_native(&self) -> bool {
        true
    }
    fn native_edge(&mut self, from: crate::host::NativeGcId, to: crate::host::NativeGcId) {
        self.native.native_edge(from, to);
    }
    fn js_to_native(&mut self, owner: &Value, to: crate::host::NativeGcId) {
        self.native.js_to_native(owner, to);
    }
    fn native_to_js(&mut self, from: crate::host::NativeGcId, value: &Value) {
        self.native.native_to_js(from, value);
    }
    fn native_root(&mut self, id: crate::host::NativeGcId) {
        self.native.native_root(id);
    }
    fn internal(&mut self, value: &Value) {
        if let Value::Obj(object) = value {
            if let Some(&index) = self.object_index.get(&(Gc::as_ptr(object) as usize)) {
                self.object_internal[index] += 1;
            }
        }
    }

    fn edge(&mut self, owner: &Value, value: &Value) {
        let (Value::Obj(owner), Value::Obj(value)) = (owner, value) else {
            return;
        };
        let Some(&target) = self.object_index.get(&(Gc::as_ptr(value) as usize)) else {
            return;
        };
        if let Some(&source) = self.object_index.get(&(Gc::as_ptr(owner) as usize)) {
            self.host_edges.entry(source).or_default().push(target);
        } else {
            // Every old/foreign owner is live for a minor collection. Its logical edges count
            // even when there is no Rust strong reference connecting the two objects.
            self.mark(Node::Object(target));
        }
    }

    fn root(&mut self, value: &Value) {
        if let Value::Obj(object) = value {
            if let Some(&index) = self.object_index.get(&(Gc::as_ptr(object) as usize)) {
                self.mark(Node::Object(index));
            }
        }
    }
}

/// Every strong edge an environment record owns: its parent, a `with` object, binding values
/// and live module-import environments.
fn visit_scope_edges(scope: &Env, sink: &mut impl GcEdgeSink) {
    let scope = scope.borrow();
    if let Some(parent) = &scope.parent {
        sink.scope(parent);
    }
    if let Some(with) = &scope.with_obj {
        sink.value(with);
    }
    for binding in scope.vars.values() {
        sink.value(&binding.value);
        if let Some((scope, _)) = binding.import_ref.as_deref() {
            sink.scope(scope);
        }
    }
}

impl Interp {
    pub(crate) fn gc_collect_young(&mut self, cause: crate::value::GcCause) {
        // P1 S2: the refcount nursery must never see cell handles; gc-heap collects in full.
        // A thin entry (the nursery's frame is large): see `Interp::gc_heap_collect_explicit`.
        #[cfg(feature = "gc-heap")]
        return self.gc_collect_with_cause(cause);
        #[cfg(not(feature = "gc-heap"))]
        self.gc_collect_young_rc(cause)
    }

    #[cfg_attr(feature = "gc-heap", allow(dead_code))]
    fn gc_collect_young_rc(&mut self, cause: crate::value::GcCause) {
        let (objects, scopes) = crate::value::heap_young_snapshot(&self.gc_heap);
        if objects.len() >= SCOPE_EDGE as usize || scopes.len() >= SCOPE_EDGE as usize {
            // Edge identities are 31-bit; such a nursery is far beyond any live-object limit.
            drop((objects, scopes));
            self.gc_collect_with_edge_budget(cause, crate::gc_edges::EDGE_CACHE_BYTES);
            return;
        }
        let started = crate::value::gc_performance_metrics_start();
        let total_objects_before = crate::value::heap_live_objects(&self.gc_heap).max(0) as usize;
        let total_scopes_before =
            started.map(|_| crate::value::gc_scope_registry_prune(&self.gc_heap));
        let mut nursery = Nursery::new(objects, scopes);
        let (object_count, scope_count) = (nursery.objects.len(), nursery.scopes.len());
        // Most young objects key no internal-slot table (or coroutine); skip those probes.
        let side_slots = self.side_slot_owners(&nursery.object_index, object_count, true);
        let mut native_captures = crate::native_captures::NativeCaptureSnapshot::empty();
        let mut object_edges = Vec::new();
        let mut scope_refs = Vec::new();
        for index in 0..object_count {
            let side = side_slots.as_ref().is_none_or(|owners| owners[index]);
            nursery.count_object(
                self,
                index,
                side,
                &mut native_captures,
                (&mut object_edges, &mut scope_refs),
            );
        }
        for index in 0..scope_count {
            nursery.count_scope(index);
        }
        nursery.edge_starts.push(nursery.edges.len());
        // Pins represent side-table bookkeeping, not externally reachable JS handles. Actual
        // outgoing strong side-table edges were counted alongside their owning young objects.
        // Test only young identities, from whichever side is smaller: walking every old pinned
        // side-table owner would make each nursery collection scale with the stable platform
        // heap.
        if self.gc_pins.len() < object_count {
            for owner in self.gc_pins.keys() {
                if let Some(&index) = nursery.object_index.get(owner) {
                    nursery.object_internal[index] += 1;
                }
            }
        } else {
            for (index, object) in nursery.objects.iter().enumerate() {
                if self.gc_pins.contains_key(&(Gc::as_ptr(object) as usize)) {
                    nursery.object_internal[index] += 1;
                }
            }
        }
        self.host_state.trace_gc(&mut nursery);
        native_captures.trace(&mut nursery);
        let mut native_js = Vec::new();
        nursery.native.trace_roots(&mut native_js);
        let mut native_owners = vec![false; object_count];
        let mut old_owners = Vec::new();
        for owner in nursery.native.js_owners() {
            match nursery.object_index.get(&owner) {
                Some(&index) => native_owners[index] = true,
                None => old_owners.push(owner),
            }
        }
        for owner in old_owners {
            nursery.native.trace_js(owner, &mut native_js);
        }
        for target in native_js.drain(..) {
            if let Some(&index) = nursery.object_index.get(&target) {
                nursery.mark(Node::Object(index));
            }
        }
        for index in 0..object_count {
            if Gc::strong_count(&nursery.objects[index]) > nursery.object_internal[index] + 1 {
                nursery.mark(Node::Object(index));
            }
        }
        for index in 0..scope_count {
            if Env::strong_count(&nursery.scopes[index]) > nursery.scope_internal[index] + 1 {
                nursery.mark(Node::Scope(index));
            }
        }
        while let Some(node) = nursery.pending.pop() {
            match node {
                Node::Object(index) => {
                    if native_owners[index] {
                        nursery
                            .native
                            .trace_js(Gc::as_ptr(&nursery.objects[index]) as usize, &mut native_js);
                        for target in native_js.drain(..) {
                            if let Some(&index) = nursery.object_index.get(&target) {
                                nursery.mark(Node::Object(index));
                            }
                        }
                    }
                    nursery.mark_recorded(index);
                    if !nursery.host_edges.is_empty() {
                        if let Some(targets) = nursery.host_edges.remove(&index) {
                            for target in targets {
                                nursery.mark(Node::Object(target));
                            }
                        }
                    }
                }
                Node::Scope(index) => nursery.mark_recorded(object_count + index),
            }
        }
        let dead: FastSet<_> = nursery
            .objects
            .iter()
            .enumerate()
            .filter(|(index, _)| !nursery.object_mark[*index])
            .map(|(_, object)| Gc::as_ptr(object) as usize)
            .collect();
        self.gc_sweep_dead_objects(&dead, &nursery.native);
        for (index, object) in nursery.objects.iter().enumerate() {
            if !nursery.object_mark[index] {
                let detached = {
                    let mut object = object.borrow_mut();
                    (
                        std::mem::take(&mut object.props),
                        object.proto.take(),
                        std::mem::replace(&mut object.call, Callable::None),
                        std::mem::replace(&mut object.exotic, Exotic::None),
                    )
                };
                drop(detached);
            }
        }
        for (index, scope) in nursery.scopes.iter().enumerate() {
            if !nursery.scope_mark[index] {
                let detached = {
                    let mut scope = scope.borrow_mut();
                    (
                        std::mem::take(&mut scope.vars),
                        scope.parent.take(),
                        scope.with_obj.take(),
                    )
                };
                drop(detached);
            }
        }
        let scanned = (object_count, scope_count);
        drop(nursery);
        self.gc_drain_weak_deaths();
        self.gc_schedule_finalization_cleanup(&dead);
        crate::value::gc_finish_generation(&self.gc_heap, false);
        self.gc_task_live = crate::value::heap_live_objects(&self.gc_heap);
        self.gc_task_pending = false;
        if let Some(started) = started {
            crate::value::gc_performance_metrics_finish(
                started,
                total_objects_before,
                self.gc_task_live,
                total_scopes_before.unwrap_or(0),
                crate::value::gc_scope_registry_prune(&self.gc_heap),
                cause,
                Some(scanned),
            );
            let objects = crate::value::heap_gc_snapshot(&self.gc_heap);
            let scopes = crate::value::gc_scope_snapshot(&self.gc_heap);
            crate::memory::record_post_gc(self, &objects, &scopes);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Tier;
    use crate::value::{GcCause, Object, Property};
    use crate::{Completion, Engine};

    fn eval(engine: &mut Engine, source: &str, expected: &str) {
        match engine.eval(source, false).expect("nursery fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected),
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    fn young(interp: &mut Interp) {
        interp.gc_collect_young(GcCause::Explicit);
    }

    #[cfg_attr(
        feature = "gc-heap",
        ignore = "P1 S2: tests the refcount cycle collector, disabled under gc-heap"
    )]
    #[test]
    fn nursery_collects_new_cycles_without_snapshotting_retained_old_objects() {
        let mut engine = Engine::new();
        eval(
            &mut engine,
            "var old = []; for (var i=0;i<20000;i++) old.push({id:i}); true",
            "true",
        );
        engine.interp.gc_collect();
        assert!(crate::value::heap_young_snapshot(&engine.interp.gc_heap)
            .0
            .is_empty());
        let mut dead = Vec::new();
        for _ in 0..2000 {
            let object = Object::new(None);
            object
                .borrow_mut()
                .props
                .insert("self", Property::builtin(Value::Obj(object.clone())));
            dead.push(Gc::downgrade(&object));
        }
        let (objects, scopes) = crate::value::heap_young_snapshot(&engine.interp.gc_heap);
        assert_eq!(objects.len(), 2000);
        assert!(scopes.is_empty());
        drop((objects, scopes));
        young(&mut engine.interp);
        assert!(dead.iter().all(|object| object.upgrade().is_none()));
        assert!(crate::value::heap_young_snapshot(&engine.interp.gc_heap)
            .0
            .is_empty());
        eval(
            &mut engine,
            "[old.length, old[0].id, old[19999].id].join(',')",
            "20000,0,19999",
        );
    }

    #[cfg_attr(
        feature = "gc-heap",
        ignore = "P1 S2: tests the refcount cycle collector, disabled under gc-heap"
    )]
    #[test]
    fn nursery_preserves_old_to_young_edges_and_major_collects_cross_generation_cycles() {
        let mut engine = Engine::new();
        let old = Object::new(None);
        engine.interp.gc_collect();
        let child = Object::new(None);
        old.borrow_mut()
            .props
            .insert("child", Property::builtin(Value::Obj(child.clone())));
        child
            .borrow_mut()
            .props
            .insert("parent", Property::builtin(Value::Obj(old.clone())));
        let old_weak = Gc::downgrade(&old);
        let child_weak = Gc::downgrade(&child);
        drop(child);
        young(&mut engine.interp);
        assert!(child_weak.upgrade().is_some());
        drop(old);
        young(&mut engine.interp);
        assert!(
            old_weak.upgrade().is_some(),
            "minor must not sweep its old generation"
        );
        engine.interp.gc_collect();
        assert!(old_weak.upgrade().is_none());
        assert!(child_weak.upgrade().is_none());
    }

    #[cfg_attr(
        feature = "gc-heap",
        ignore = "P1 S2: tests the refcount cycle collector, disabled under gc-heap"
    )]
    #[test]
    fn nursery_acyclic_churn_does_not_accumulate_weak_allocation_headers() {
        let mut engine = Engine::new();
        engine.interp.gc_collect();
        for _ in 0..50000 {
            let object = Object::new(None);
            drop(object);
        }
        let (objects, scopes) = crate::value::heap_young_snapshot(&engine.interp.gc_heap);
        assert!(objects.is_empty());
        assert!(scopes.is_empty());
    }

    #[test]
    fn nursery_live_call_frames_scopes_weak_targets_and_side_tables_all_tiers() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            engine
                .interp
                .def_method(&engine.interp.global, "minorCollect", 0, |interp, _, _| {
                    young(interp);
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
                        minorCollect();
                        return [o.n, map.get(o).n, set.has(o), weak.deref() === o,
                            weakMap.get(o).owner === o, bytes[0]].join(',');
                    }
                    minorCollect();
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
    fn nursery_finalizer_targets_clear_together_and_ready_cells_are_removed() {
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
        young(&mut engine.interp);
        eval(&mut engine, "true", "true");
        eval(&mut engine, "[first.deref() === undefined, second.deref() === undefined, finalized.join(',')].join(':')", "true:true:91");
    }

    /// Young owners of every internal-slot family keep their table edges across a nursery
    /// collection, and cycles that pass only through those tables (or a suspended coroutine)
    /// are reclaimed. Covers both the owner-flag path (few table entries) and the conservative
    /// probe-everything path (old table entries outnumber the nursery), on every tier.
    #[test]
    fn nursery_side_slot_owners_keep_table_edges_with_and_without_owner_flags() {
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
                young(&mut engine.interp);
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
