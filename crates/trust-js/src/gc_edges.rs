//! Bounded, collection-local strong-edge snapshots. No graph mutation or JS runs between
//! reference counting and marking. Reusing those edges avoids decoding every live object's
//! properties and interpreter side tables twice. The cache records identities only, never
//! extra Rc owners, so it cannot distort the strong counts that classify roots.

use crate::fasthash::FastMap;
use crate::interpreter::Env;
use crate::value::{Gc, Value};
use std::mem::size_of;

pub(crate) const EDGE_CACHE_BYTES: usize = 32 * 1024 * 1024;

/// Collection-local absence proof for strong interpreter side slots. This is
/// not an owner/edge index and holds no Rc or raw object pointer. Dense heaps
/// retain the original cached walk; sparse heaps use at most 256 KiB of bits.
/// Test-only: the runtime experiment was withdrawn after startup/resource
/// regressions. Keep its ownership and budget controls for future designs.
#[cfg(test)]
pub(crate) struct GcSlotOwnerMask {
    words: Vec<u64>,
    object_count: usize,
}

#[cfg(test)]
impl GcSlotOwnerMask {
    pub(crate) fn new(object_count: usize, owner_entries: usize) -> Self {
        let words = object_count.div_ceil(64);
        let enabled = object_count >= 4096
            && owner_entries <= object_count / 4
            && words <= (256 * 1024) / size_of::<u64>();
        Self {
            words: if enabled { vec![0; words] } else { Vec::new() },
            object_count,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        !self.words.is_empty()
    }

    pub(crate) fn note_owner(&mut self, index: usize) {
        assert!(index < self.object_count);
        if let Some(word) = self.words.get_mut(index / 64) {
            *word |= 1u64 << (index % 64);
        }
    }

    #[inline]
    pub(crate) fn may_own_slots(&self, index: Option<usize>) -> bool {
        // Foreign objects and the disabled/dense path must keep the full trace.
        let Some(index) = index.filter(|&index| index < self.object_count) else {
            return true;
        };
        self.words
            .get(index / 64)
            .is_none_or(|word| word & (1u64 << (index % 64)) != 0)
    }
}

/// Receives the strong graph edges of one heap node, borrowed from their owner. Each stored
/// handle is reported once per occurrence (do not deduplicate). The owner stays borrowed while
/// the sink runs, so a sink may read collector scratch cells but must never mutably borrow an
/// object or environment, allocate JavaScript values or run author code.
pub(crate) trait GcEdgeSink {
    fn object(&mut self, object: &Gc);
    fn scope(&mut self, scope: &Env);
    #[inline]
    fn value(&mut self, value: &Value) {
        if let Value::Obj(object) = value {
            self.object(object);
        }
    }
}

/// Direct, owning edges of one internal-slot payload, appended to the collector's
/// existing scratch handles. Do not deduplicate: two stored Rc handles are two
/// internal owners even when both point to the same object or environment.
pub(crate) struct DirectGcEdges<'a> {
    pub objects: &'a mut Vec<Gc>,
    pub scopes: &'a mut Vec<Env>,
}

impl DirectGcEdges<'_> {
    pub fn value(&mut self, value: &Value) {
        if let Value::Obj(object) = value {
            self.object(object);
        }
    }

    pub fn object(&mut self, object: &Gc) {
        self.objects.push(object.clone());
    }

    pub fn scope(&mut self, scope: &Env) {
        self.scopes.push(scope.clone());
    }
}

impl GcEdgeSink for DirectGcEdges<'_> {
    #[inline]
    fn object(&mut self, object: &Gc) {
        self.objects.push(object.clone());
    }

    #[inline]
    fn scope(&mut self, scope: &Env) {
        self.scopes.push(scope.clone());
    }
}

/// Identity of a counted object edge. It owns nothing: it stays valid only while the
/// collection that recorded it holds its snapshot, during which no graph mutation or JS runs,
/// so every recorded target is still owned by the snapshot object that held the edge.
pub(crate) type ObjectEdge = *const std::cell::RefCell<crate::value::Object>;

/// The full collector's counting sink: adds each edge to its target's internal-reference count
/// and records it (an object identity, or a snapshot environment index) for marking. Foreign
/// environments are not candidates and were never counted.
pub(crate) struct CountedEdges<'a> {
    pub objects: &'a mut Vec<ObjectEdge>,
    pub scopes: &'a mut Vec<u32>,
    pub scope_index: &'a FastMap<usize, usize>,
    pub scope_internal: &'a mut [u32],
}

impl GcEdgeSink for CountedEdges<'_> {
    #[inline]
    fn object(&mut self, object: &Gc) {
        // SAFETY: as in the collector's `cells`: only a scratch Cell is touched, and no mutable
        // object borrow exists while the collector runs (an outstanding one panics).
        let target =
            unsafe { object.try_borrow_unguarded() }.expect("object mutably borrowed during GC");
        target.gc_scratch.set(target.gc_scratch.get() + 1);
        self.objects.push(Gc::as_ptr(object));
    }

    #[inline]
    fn scope(&mut self, scope: &Env) {
        if let Some(&index) = self.scope_index.get(&(Env::as_ptr(scope) as usize)) {
            self.scope_internal[index] += 1;
            self.scopes.push(index as u32);
        }
    }
}

#[derive(Clone, Copy)]
struct Edges {
    object_start: u32,
    object_len: u32,
    scope_start: u32,
    scope_len: u32,
}

impl Edges {
    const UNCACHED: Self = Self {
        object_start: 0,
        object_len: u32::MAX,
        scope_start: 0,
        scope_len: 0,
    };
}

pub(crate) struct GcEdgeCache {
    entries: Vec<Edges>,
    objects: Vec<ObjectEdge>,
    scopes: Vec<u32>,
    budget: usize,
    enabled: bool,
}

impl GcEdgeCache {
    pub(crate) fn new(object_count: usize, budget: usize) -> Self {
        let enabled = object_count < u32::MAX as usize
            && object_count.saturating_mul(size_of::<Edges>()) <= budget
            && budget != 0;
        Self {
            entries: Vec::with_capacity(if enabled { object_count } else { 0 }),
            objects: Vec::new(),
            scopes: Vec::new(),
            budget,
            enabled,
        }
    }

    /// Reserve geometrically, but cap requested backing capacities (not just populated bytes).
    /// A large graph can partly use the cache; uncached owners keep the original tracing path.
    pub(crate) fn prepare(&mut self, objects: usize, scopes: usize) -> bool {
        if !self.enabled {
            return false;
        }
        let capacity = |len: usize, cap: usize, extra: usize| {
            let need = len.saturating_add(extra);
            if need > cap {
                need.max(cap.saturating_mul(2)).max(4)
            } else {
                cap
            }
        };
        let object_cap = capacity(self.objects.len(), self.objects.capacity(), objects);
        let scope_cap = capacity(self.scopes.len(), self.scopes.capacity(), scopes);
        let bytes = self
            .entries
            .capacity()
            .saturating_mul(size_of::<Edges>())
            .saturating_add(object_cap.saturating_mul(size_of::<ObjectEdge>()))
            .saturating_add(scope_cap.saturating_mul(size_of::<u32>()));
        if bytes > self.budget || object_cap >= u32::MAX as usize || scope_cap >= u32::MAX as usize
        {
            return false;
        }
        if object_cap > self.objects.capacity() {
            self.objects.reserve_exact(object_cap - self.objects.len());
        }
        if scope_cap > self.scopes.capacity() {
            self.scopes.reserve_exact(scope_cap - self.scopes.len());
        }
        true
    }

    /// Keep one owner's counted edges for marking, or mark it uncached (re-traced while
    /// marking). The scratch vectors are emptied either way.
    pub(crate) fn record(
        &mut self,
        cached: bool,
        objects: &mut Vec<ObjectEdge>,
        scopes: &mut Vec<u32>,
    ) {
        if cached {
            let object_start = self.objects.len();
            let scope_start = self.scopes.len();
            self.objects.append(objects);
            self.scopes.append(scopes);
            self.entries.push(Edges {
                object_start: object_start as u32,
                object_len: (self.objects.len() - object_start) as u32,
                scope_start: scope_start as u32,
                scope_len: (self.scopes.len() - scope_start) as u32,
            });
        } else {
            objects.clear();
            scopes.clear();
            if self.enabled {
                self.entries.push(Edges::UNCACHED);
            }
        }
    }

    pub(crate) fn get(&self, index: usize) -> Option<(&[ObjectEdge], &[u32])> {
        let entry = self.entries.get(index)?;
        if entry.object_len == u32::MAX {
            return None;
        }
        let object_start = entry.object_start as usize;
        let scope_start = entry.scope_start as usize;
        Some((
            &self.objects[object_start..object_start + entry.object_len as usize],
            &self.scopes[scope_start..scope_start + entry.scope_len as usize],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Tier;
    use crate::value::Object;
    use crate::{Completion, Engine};
    use std::rc::Rc;

    fn value(engine: &mut Engine, source: &str) -> String {
        match engine.eval(source, false).expect("GC fixture parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    fn sparse_heap_padding(engine: &Engine) -> Vec<Gc> {
        (0..4500)
            .map(|_| Object::new(Some(engine.interp.object_proto.clone())))
            .collect()
    }

    #[test]
    fn gc_slot_owner_mask_is_bounded_and_keeps_conservative_fallbacks() {
        for (objects, owners) in [(4095, 0), (4096, 1025), (2_097_153, 0)] {
            let mask = GcSlotOwnerMask::new(objects, owners);
            assert!(!mask.enabled());
            assert!(mask.may_own_slots(Some(0)));
            assert!(mask.may_own_slots(None));
            assert_eq!(mask.words.capacity(), 0);
        }
        let mut mask = GcSlotOwnerMask::new(4097, 2);
        assert!(mask.enabled());
        assert_eq!(mask.words.capacity() * size_of::<u64>(), 520);
        mask.note_owner(63);
        mask.note_owner(63);
        mask.note_owner(4096);
        for index in 0..4097 {
            assert_eq!(
                mask.may_own_slots(Some(index)),
                index == 63 || index == 4096
            );
        }
        assert!(mask.may_own_slots(Some(4097)));
        assert!(mask.may_own_slots(Some(usize::MAX)));
        assert!(mask.may_own_slots(None));
    }

    #[test]
    fn gc_slot_owner_mask_inventories_every_strong_table_family() {
        use crate::interpreter::{
            ClassInfo, FinalizationState, HostIndexedProperties, PromiseState,
        };
        let mut engine = Engine::new();
        let ordinary = sparse_heap_padding(&engine);
        let index = ordinary
            .iter()
            .enumerate()
            .map(|(n, object)| (Gc::as_ptr(object) as usize, n))
            .collect();
        let key = |n: usize| Gc::as_ptr(&ordinary[n]) as usize;
        engine.interp.class_info.insert(
            key(0),
            ClassInfo {
                fields: Rc::from([]),
                field_env: engine.interp.global_env.clone(),
                derived: false,
                instance_initializers: Vec::new(),
                private_members: Vec::new(),
            },
        );
        engine.interp.module_ns.insert(key(1), Default::default());
        // Family 2: [[MapData]]'s table; gc-heap keeps a Map's data in its cell, and its table in
        // the list is the iterator table.
        let insert_family_2 = |engine: &mut Engine| {
            #[cfg(not(feature = "gc-heap"))]
            engine.interp.map_data.insert(key(2), Vec::new().into());
            #[cfg(feature = "gc-heap")]
            engine.interp.collection_iterators.insert(
                key(2),
                crate::ordered_collection::CollectionIterator {
                    target: None,
                    cursor: None,
                    kind: 0,
                    brand: crate::ordered_collection::CollectionKind::Map,
                },
            );
        };
        let remove_family_2 = |engine: &mut Engine| {
            #[cfg(not(feature = "gc-heap"))]
            engine.interp.map_data.remove(&key(2));
            #[cfg(feature = "gc-heap")]
            engine.interp.collection_iterators.remove(&key(2));
        };
        insert_family_2(&mut engine);
        engine.interp.ta_buffer.insert(key(3), Value::Undefined);
        engine
            .interp
            .proxies
            .insert(key(4), (Value::Undefined, Value::Undefined));
        engine.interp.host_indexed.insert(
            key(5),
            HostIndexedProperties {
                length: 0,
                getter: Value::Undefined,
                live: None,
            },
        );
        engine
            .interp
            .promises
            .insert(key(6), PromiseState::default());
        engine
            .interp
            .promise_forward
            .insert(key(7), Value::Undefined);
        engine
            .interp
            .async_gen_queue
            .insert(key(8), Default::default());
        engine.interp.finalization_registries.insert(
            key(9),
            FinalizationState {
                cleanup_callback: crate::interpreter::JobCallback::plain(Value::Undefined),
                cells: Default::default(),
                cleanup_scheduled: false,
            },
        );
        engine
            .interp
            .mapped_arguments
            .insert(key(10), (engine.interp.global_env.clone(), Vec::new()));
        // Ephemeron payloads must not turn into ordinary strong edges.
        engine
            .interp
            .weak_collection_data
            .insert(key(11), Vec::new());
        let mask = engine.interp.gc_slot_owner_mask(&index);
        assert!(mask.enabled());
        for n in 0..ordinary.len() {
            assert_eq!(mask.may_own_slots(Some(n)), n < 11, "table owner {n}");
        }
        // Inventory is rebuilt from actual entries each collection, not inferred
        // from gc_pins and not invalidated by a stale persistent owner flag.
        remove_family_2(&mut engine);
        assert!(!engine
            .interp
            .gc_slot_owner_mask(&index)
            .may_own_slots(Some(2)));

        // The retained experimental pinned-owner proof must verify its premise,
        // not assume every synthetic/embed owner was pinned by a constructor.
        let count = ordinary.len();
        assert!(!engine.interp.gc_can_skip_unpinned_slots(count));
        for object in ordinary.iter().take(11) {
            engine.interp.gc_pin(object);
        }
        assert!(engine.interp.gc_can_skip_unpinned_slots(count));
        for n in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10] {
            let owner = engine.interp.gc_pins.remove(&key(n)).unwrap();
            assert!(
                !engine.interp.gc_can_skip_unpinned_slots(count),
                "unpinned family {n}"
            );
            engine.interp.gc_pin(&owner);
            assert!(engine.interp.gc_can_skip_unpinned_slots(count));
        }
        insert_family_2(&mut engine);
        let owner = engine.interp.gc_pins.remove(&key(2)).unwrap();
        assert!(!engine.interp.gc_can_skip_unpinned_slots(count));
        engine.interp.gc_pin(&owner);
        assert!(engine.interp.gc_can_skip_unpinned_slots(count));
        assert!(!engine.interp.gc_can_skip_unpinned_slots(4095));
        for object in ordinary.iter().take(count / 4 + 1) {
            engine.interp.gc_pin(object);
        }
        assert!(!engine.interp.gc_can_skip_unpinned_slots(count));
    }

    #[test]
    fn gc_releases_abandoned_continuations_without_running_author_cleanup() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            for budget in [0, EDGE_CACHE_BYTES] {
                let mut engine = Engine::new();
                engine.set_tier(tier);
                engine.set_tier_threshold(0);
                value(
                    &mut engine,
                    r#"
                    var targets = [], cleaned = 0;
                    function make(mode) {
                        const target = {mode}; target.self = target;
                        function* sequence() {
                            try { yield target; return 1; }
                            finally { cleaned++; }
                        }
                        const generator = sequence();
                        // Keep an actual cycle in every tier, even when compiled capture
                        // analysis omits the otherwise unused local generator binding.
                        target.generator = generator;
                        if (mode !== 0) generator.next();
                        if (mode === 2) generator.return();
                        if (mode === 3) generator.next();
                        targets.push(new WeakRef(target));
                    }
                    for (var mode = 0; mode < 4; mode++) make(mode);
                    'ready'
                "#,
                );
                for _ in 0..3 {
                    engine
                        .interp
                        .gc_collect_with_edge_budget(crate::value::GcCause::Explicit, budget);
                    assert_eq!(value(&mut engine,
                        "targets.map(ref => ref.deref() === undefined).join(',') + ':' + cleaned"),
                        "true,true,true,true:2", "{tier:?}, budget {budget}");
                }
                assert!(
                    engine.interp.generators.is_empty(),
                    "dead continuation metadata must be evicted"
                );
            }
        }
    }

    #[test]
    fn gc_preserves_live_suspended_and_executing_continuations() {
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            for budget in [0, EDGE_CACHE_BYTES] {
                let mut engine = Engine::new();
                engine.set_tier(tier);
                engine.set_tier_threshold(0);
                value(
                    &mut engine,
                    r#"
                    var live, weak;
                    (() => {
                        const target = {n: 39}; weak = new WeakRef(target);
                        live = (function*(input) {
                            const values = [input];
                            yield values[0].n;
                            // While running, this state has left the side table. Its native
                            // stack-owned handles must NOT be discounted as internal owners.
                            $262.gc();
                            yield input.n + 1;
                            return values[0];
                        })(target);
                    })();
                    'ready'
                "#,
                );
                for expected in ["39", "40", "[object Object]"] {
                    engine
                        .interp
                        .gc_collect_with_edge_budget(crate::value::GcCause::Explicit, budget);
                    assert_eq!(value(&mut engine, "weak.deref().n"), "39");
                    assert_eq!(
                        value(&mut engine, "live.next().value"),
                        expected,
                        "{tier:?}"
                    );
                }
                value(&mut engine, "live = null; 'released'");
                engine
                    .interp
                    .gc_collect_with_edge_budget(crate::value::GcCause::Explicit, budget);
                assert_eq!(value(&mut engine, "weak.deref() === undefined"), "true");
            }
        }
    }

    #[test]
    fn gc_coroutine_edges_preserve_external_promise_resumption_and_weak_ephemerons() {
        for budget in [0, EDGE_CACHE_BYTES] {
            let mut engine = Engine::new();
            value(
                &mut engine,
                r#"
                var resolve, done, weak, abandoned, wm = new WeakMap();
                (() => {
                    const target = {n: 41}; weak = new WeakRef(target);
                    wm.set(target, {n: 42});
                    (async function(input) {
                        await new Promise(r => resolve = r);
                        $262.gc();
                        done = wm.get(input).n;
                    })(target);
                })();
                (() => {
                    const target = {}; target.self = target;
                    abandoned = new WeakRef(target);
                    (async function(input) {
                        // No resolver escapes, so this await cannot ever resume.
                        await new Promise(() => {});
                        globalThis.unreachable = input;
                    })(target);
                })();
                'ready'
            "#,
            );
            engine
                .interp
                .gc_collect_with_edge_budget(crate::value::GcCause::Explicit, budget);
            assert_eq!(
                value(
                    &mut engine,
                    "[weak.deref().n, abandoned.deref() === undefined, done].join(':')"
                ),
                "41:true:"
            );
            value(&mut engine, "resolve(); resolve = null; 'queued'");
            assert_eq!(value(&mut engine, "done"), "42");
            engine
                .interp
                .gc_collect_with_edge_budget(crate::value::GcCause::Explicit, budget);
            assert_eq!(value(&mut engine, "weak.deref() === undefined"), "true");
        }
    }

    #[test]
    fn gc_async_generator_reactions_own_the_generator_across_await() {
        for budget in [0, EDGE_CACHE_BYTES] {
            for return_mode in [false, true] {
                for reject in [false, true] {
                    let mut engine = Engine::new();
                    value(
                        &mut engine,
                        r#"
                        var settle, result = 'pending', weak;
                        function fulfilled(value) { result = 'ok:' + value.value; }
                        function rejected(error) { result = 'error:' + error; }
                        function start(returnMode, reject) {
                            const gate = new Promise((resolve, fail) => settle = reject ? fail : resolve);
                            const generator = (async function*() { yield await gate; })();
                            weak = new WeakRef(generator);
                            // The caller retains only the external resolver, not the iterator.
                            const request = returnMode ? generator.return(gate) : generator.next();
                            request.then(fulfilled, rejected);
                        }
                    "#,
                    );
                    value(
                        &mut engine,
                        &format!("start({return_mode}, {reject}); 'ready'"),
                    );
                    engine
                        .interp
                        .gc_collect_with_edge_budget(crate::value::GcCause::Explicit, budget);
                    assert_eq!(value(&mut engine, "weak.deref() !== undefined"), "true");
                    value(&mut engine, "settle(42); settle = null; 'queued'");
                    assert_eq!(
                        value(&mut engine, "result"),
                        if reject { "error:42" } else { "ok:42" },
                        "return={return_mode}, reject={reject}, budget={budget}"
                    );
                    engine
                        .interp
                        .gc_collect_with_edge_budget(crate::value::GcCause::Explicit, budget);
                    assert_eq!(value(&mut engine, "weak.deref() === undefined"), "true");
                }
            }
        }
    }

    #[test]
    fn gc_continuation_reference_class_delegate_and_disposal_state_survives() {
        for budget in [0, EDGE_CACHE_BYTES] {
            let mut engine = Engine::new();
            value(
                &mut engine,
                r#"
                var refWeak, classWeak, iterWeak, resourceWeak, disposed = 0;
                function referenceTarget() {
                    const object = {x:1}; refWeak = new WeakRef(object); return object;
                }
                function parentClass() {
                    const parent = class {base(){return 40}};
                    classWeak = new WeakRef(parent); return parent;
                }
                function iterator() {
                    const iter = {n:0, next(){return {value:++this.n, done:this.n>2}},
                                  [Symbol.iterator](){return this}};
                    iterWeak = new WeakRef(iter); return iter;
                }
                function resource() {
                    const object = {[Symbol.dispose](){disposed++}};
                    resourceWeak = new WeakRef(object); return object;
                }
                var referenceGen = (function*(){referenceTarget().x += yield 'rhs';return 7})();
                var classGen = (function*(){return class extends (yield 'parent') {
                    [yield 'name'](){return 2}
                }})();
                var delegateGen = (function*(){return yield* iterator()})();
                var resourceGen = (function*(){using item = resource();yield 1;return 9})();
                referenceGen.next(); classGen.next(); classGen.next(parentClass());
                delegateGen.next(); resourceGen.next();
                'ready'
            "#,
            );
            engine
                .interp
                .gc_collect_with_edge_budget(crate::value::GcCause::Explicit, budget);
            assert_eq!(value(&mut engine,
                "[refWeak,classWeak,iterWeak,resourceWeak].every(ref => ref.deref() !== undefined)"), "true");
            assert_eq!(
                value(
                    &mut engine,
                    r#"
                var refTarget = refWeak.deref();
                var a = referenceGen.next(5).value;
                var sum = refTarget.x;
                var C = classGen.next('extra').value;
                var c = new C();
                [a, sum, c.base()+c.extra(), delegateGen.next().value,
                 resourceGen.next().value, disposed].join(':')
            "#
                ),
                "7:6:42:2:9:1"
            );
            value(
                &mut engine,
                "referenceGen=classGen=delegateGen=resourceGen=C=c=refTarget=null; 'released'",
            );
            engine
                .interp
                .gc_collect_with_edge_budget(crate::value::GcCause::Explicit, budget);
            assert_eq!(value(&mut engine,
                "[refWeak,classWeak,iterWeak,resourceWeak].every(ref => ref.deref() === undefined)"), "true");
            assert_eq!(value(&mut engine, "disposed"), "1");
        }
    }

    #[test]
    fn gc_from_async_retains_its_iterator_until_external_resumption() {
        for budget in [0, EDGE_CACHE_BYTES] {
            let mut engine = Engine::new();
            value(
                &mut engine,
                r#"
                var settle, result, weak;
                function done(values) { result = values.join(','); }
                (() => {
                    const iter = {
                        index:0,
                        next() { return this.index++ ? {done:true} : new Promise(r => settle=r); },
                        [Symbol.asyncIterator]() { return this; }
                    };
                    weak = new WeakRef(iter);
                    Array.fromAsync(iter).then(done);
                })();
                'ready'
            "#,
            );
            engine
                .interp
                .gc_collect_with_edge_budget(crate::value::GcCause::Explicit, budget);
            assert_eq!(value(&mut engine, "weak.deref() !== undefined"), "true");
            value(
                &mut engine,
                "settle({done:false,value:42}); settle = null; 'queued'",
            );
            assert_eq!(value(&mut engine, "result"), "42");
            engine
                .interp
                .gc_collect_with_edge_budget(crate::value::GcCause::Explicit, budget);
            assert_eq!(value(&mut engine, "weak.deref() === undefined"), "true");
        }
    }
}
