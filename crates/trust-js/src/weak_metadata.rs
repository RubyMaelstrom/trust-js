//! Target-directed weak clearing metadata (ECMA-262 #sec-weakref-execution).
//!
//! Index only observed identities, retaining allocation headers but never strong JS edges.
//! Object/Symbol destruction sends pinned identities to every observing interpreter, including
//! other heaps sharing the Agent. Queue and index storage shrink geometrically under churn.

use crate::fasthash::{FastMap, FastSet};
use crate::interpreter::Interp;
use crate::interpreter::{WeakKey, WeakTarget};
use crate::value::Value;
use std::cell::{Cell, RefCell};
use std::mem::size_of;
use std::rc::{Rc, Weak};

/// Exact deduplicated pending count, readable without borrowing the journal at every loop/job
/// safepoint. All mutations publish the count while owning the entries borrow. The queue pins
/// allocation headers only; it never owns a strong JS edge or invokes author code.
#[derive(Default)]
pub(crate) struct DeathQueue {
    pending: Cell<usize>,
    entries: RefCell<FastMap<WeakKey, WeakTarget>>,
}

impl DeathQueue {
    #[inline]
    fn pending_len(&self) -> usize {
        self.pending.get()
    }

    pub(crate) fn enqueue(&self, target: WeakTarget) {
        let mut entries = self.entries.borrow_mut();
        debug_assert_eq!(self.pending.get(), entries.len());
        entries.insert(target.key(), target);
        self.pending.set(entries.len());
    }

    fn remove(&self, key: WeakKey) {
        if self.pending_len() == 0 {
            return;
        }
        let mut entries = self.entries.borrow_mut();
        debug_assert_eq!(self.pending.get(), entries.len());
        entries.remove(&key);
        self.pending.set(entries.len());
        shrink_map(&mut entries);
    }

    fn take_all(&self) -> Vec<WeakTarget> {
        if self.pending_len() == 0 {
            return Vec::new();
        }
        let mut entries = self.entries.borrow_mut();
        debug_assert_eq!(self.pending.get(), entries.len());
        let result = entries.drain().map(|(_, target)| target).collect();
        self.pending.set(0);
        shrink_map(&mut entries);
        // Releasing subscribers/payloads can enqueue further deaths, so callers process this
        // batch only after the RefMut has gone away, then take further batches to a fixed point.
        result
    }

    fn allocated_bytes(&self) -> usize {
        size_of::<Self>() + self.entries.borrow().capacity() * size_of::<(WeakKey, WeakTarget)>()
    }
}
const DEATH_BATCH_LIMIT: usize = 256;

/// [[WeakMapData]] and [[WeakSetData]] are distinct internal slots. Brand identity travels
/// with the index during derived-constructor transplantation, never in an authored property.
pub(crate) struct WeakCollectionIndex {
    pub(crate) kind: crate::ordered_collection::CollectionKind,
    pub(crate) entries: FastMap<WeakKey, usize>,
}

impl WeakCollectionIndex {
    pub(crate) fn new(kind: crate::ordered_collection::CollectionKind) -> Self {
        Self {
            kind,
            entries: FastMap::default(),
        }
    }
}

pub(crate) struct DeathObservers {
    target: WeakTarget,
    queues: FastMap<usize, Weak<DeathQueue>>,
}

impl DeathObservers {
    pub(crate) fn new(target: WeakTarget) -> Self {
        Self {
            target,
            queues: FastMap::default(),
        }
    }

    pub(crate) fn subscribe(&mut self, queue: &Rc<DeathQueue>) {
        self.queues
            .insert(Rc::as_ptr(queue) as usize, Rc::downgrade(queue));
    }

    pub(crate) fn unsubscribe(&mut self, queue: &Rc<DeathQueue>) -> bool {
        self.queues.remove(&(Rc::as_ptr(queue) as usize));
        shrink_map(&mut self.queues);
        self.queues.is_empty()
    }

    pub(crate) fn notify(self) {
        // Nothing invokes script or drops a strong managed value under the queue borrow.
        // A WeakTarget pins the allocation header until the receiving index has consumed it.
        for queue in self.queues.values().filter_map(Weak::upgrade) {
            queue.enqueue(self.target.clone());
        }
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.queues.capacity() * size_of::<(usize, Weak<DeathQueue>)>()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum Subscriber {
    WeakRef(usize),
    Collection(usize),
    Registry(usize),
}

// A single WeakRef/WeakMap is common: do not allocate a hash table for every target.
enum Subscribers {
    One(Subscriber),
    Many(FastSet<Subscriber>),
}

impl Subscribers {
    fn insert(&mut self, subscriber: Subscriber) {
        match self {
            Self::One(old) if *old == subscriber => {}
            Self::One(old) => *self = Self::Many([*old, subscriber].into_iter().collect()),
            Self::Many(subscribers) => {
                subscribers.insert(subscriber);
            }
        }
    }

    fn remove(&mut self, subscriber: Subscriber) -> bool {
        match self {
            Self::One(old) => *old == subscriber,
            Self::Many(subscribers) => {
                subscribers.remove(&subscriber);
                if subscribers.len() == 1 {
                    *self = Self::One(*subscribers.iter().next().unwrap());
                } else if subscribers.capacity() > subscribers.len().saturating_mul(4).max(32) {
                    subscribers.shrink_to(subscribers.len().saturating_mul(2).max(16));
                }
                false
            }
        }
    }
}

struct TargetEntry {
    target: WeakTarget,
    subscribers: Subscribers,
}

#[derive(Default)]
pub(crate) struct WeakMetadata {
    targets: FastMap<WeakKey, TargetEntry>,
    deaths: Rc<DeathQueue>,
    pub(crate) ready_registries: FastSet<usize>,
    #[cfg(test)]
    pub(crate) cleared_subscribers: usize,
}

impl WeakMetadata {
    pub(crate) fn subscribe(&mut self, target: WeakTarget, subscriber: Subscriber) {
        use std::collections::hash_map::Entry;
        match self.targets.entry(target.key()) {
            Entry::Occupied(mut entry) => entry.get_mut().subscribers.insert(subscriber),
            Entry::Vacant(entry) => {
                crate::value::observe_weak_target(&target, &self.deaths);
                entry.insert(TargetEntry {
                    target,
                    subscribers: Subscribers::One(subscriber),
                });
            }
        }
    }

    pub(crate) fn unsubscribe(&mut self, key: WeakKey, subscriber: Subscriber) {
        if self
            .targets
            .get_mut(&key)
            .is_some_and(|entry| entry.subscribers.remove(subscriber))
        {
            let entry = self.targets.remove(&key).unwrap();
            crate::value::unobserve_weak_target(&entry.target, &self.deaths);
            // Unregister/delete can follow an acyclic death before any GC, including symbol-only
            // churn. Remove that now-unobserved queued header instead of accumulating history.
            self.deaths.remove(key);
            shrink_map(&mut self.targets);
        }
    }

    pub(crate) fn take_subscribers(&mut self, key: WeakKey) -> Vec<Subscriber> {
        let Some(entry) = self.targets.remove(&key) else {
            return Vec::new();
        };
        crate::value::unobserve_weak_target(&entry.target, &self.deaths);
        self.deaths.remove(key);
        let subscribers: Vec<_> = match entry.subscribers {
            Subscribers::One(subscriber) => vec![subscriber],
            Subscribers::Many(subscribers) => subscribers.into_iter().collect(),
        };
        #[cfg(test)]
        {
            self.cleared_subscribers += subscribers.len();
        }
        shrink_map(&mut self.targets);
        subscribers
    }

    pub(crate) fn take_deaths(&self) -> Vec<WeakTarget> {
        self.deaths.take_all()
    }

    pub(crate) fn allocated_bytes(&self) -> usize {
        self.targets.capacity() * size_of::<(WeakKey, TargetEntry)>()
            + self
                .targets
                .values()
                .map(|entry| match &entry.subscribers {
                    Subscribers::One(_) => 0,
                    Subscribers::Many(subscribers) => {
                        subscribers.capacity() * size_of::<Subscriber>()
                    }
                })
                .sum::<usize>()
            + 2 * size_of::<usize>()
            + self.deaths.allocated_bytes()
            + self.ready_registries.capacity() * size_of::<usize>()
    }
}

impl Drop for WeakMetadata {
    fn drop(&mut self) {
        // An observer heap may die before a foreign target. Remove its sparse subscription now,
        // without keeping either heap alive until that target happens to be destroyed.
        for entry in self.targets.values() {
            crate::value::unobserve_weak_target(&entry.target, &self.deaths);
        }
    }
}

pub(crate) fn shrink_map<K, V, S>(map: &mut std::collections::HashMap<K, V, S>)
where
    K: Eq + std::hash::Hash,
    S: std::hash::BuildHasher,
{
    if map.capacity() > map.len().saturating_mul(4).max(32) {
        map.shrink_to(map.len().saturating_mul(2).max(16));
    }
}

impl Interp {
    /// Symbol-only churn may never cross the object allocation threshold. Consume already
    /// proven acyclic deaths in bounded batches at ordinary safepoints and weak insertions.
    /// This does not select additional dead objects or invoke cleanup callbacks. Ready cells
    /// merely request a task-boundary collection, which validates registry owners before jobs
    /// are scheduled. ClearKeptObjects still belongs exclusively to the host job boundary.
    #[inline]
    pub(crate) fn gc_weak_metadata_safepoint(&mut self, force: bool) {
        if self.gc_suppressed != 0 {
            return;
        }
        let pending = self.weak_metadata.deaths.pending_len();
        if !force && pending < DEATH_BATCH_LIMIT {
            return;
        }
        if pending != 0 {
            self.gc_drain_weak_deaths();
        }
        // A prior drain can have readied cells without queued deaths remaining. Forced job
        // checkpoints must still request owner validation/scheduling in that case. Neither
        // ClearKeptObjects nor host end_job is gated here (ECMA-262 #sec-weakref-invariants).
        self.gc_task_pending |= !self.weak_metadata.ready_registries.is_empty();
    }

    /// Delete one weak entry and its identity index together. Weak collection order is not
    /// observable; relocate the last entry and repair its index before releasing any payload.
    pub(crate) fn weak_collection_remove(&mut self, owner: usize, key: WeakKey) -> bool {
        let Some(index) = self
            .weak_collection_index
            .get_mut(&owner)
            .and_then(|index| index.entries.remove(&key))
        else {
            return false;
        };
        let entries = self
            .weak_collection_data
            .get_mut(&owner)
            .expect("weak collection backing data");
        let removed = entries.swap_remove(index);
        let lookup = &mut self
            .weak_collection_index
            .get_mut(&owner)
            .expect("weak collection index")
            .entries;
        if let Some((moved, _)) = entries.get(index) {
            lookup.insert(moved.key(), index);
        }
        if entries.capacity() > entries.len().saturating_mul(4).max(32) {
            entries.shrink_to(entries.len().saturating_mul(2).max(16));
        }
        shrink_map(lookup);
        self.weak_metadata
            .unsubscribe(key, Subscriber::Collection(owner));
        // Dropping an ephemeron payload can enqueue another observed identity. The collector
        // drains that cascade to a fixed point before resuming script.
        drop(removed);
        true
    }

    fn clear_weak_identity(&mut self, key: WeakKey) {
        for subscriber in self.weak_metadata.take_subscribers(key) {
            match subscriber {
                Subscriber::WeakRef(owner) => {
                    if let Some(target) = self.weak_refs.get_mut(&owner) {
                        if target.as_ref().is_some_and(|target| target.key() == key) {
                            *target = None;
                        }
                    }
                }
                Subscriber::Collection(owner) => {
                    self.weak_collection_remove(owner, key);
                }
                Subscriber::Registry(owner) => {
                    if let Some(registry) = self.finalization_registries.get_mut(&owner) {
                        if registry.cells.clear_target(key) != 0 {
                            self.weak_metadata.ready_registries.insert(owner);
                        }
                    }
                }
            }
        }
    }

    /// ECMA-262's atomic weak-clearing phase. Collector-proven cycles and acyclic destructor
    /// notifications share this path. Never invoke author code here or while a queue is borrowed.
    pub(crate) fn gc_clear_weak_targets(&mut self, dead: &FastSet<usize>) {
        #[cfg(test)]
        {
            self.weak_metadata.cleared_subscribers = 0;
        }
        for &object in dead {
            self.clear_weak_identity(WeakKey::Object(object));
        }
        self.gc_drain_weak_deaths();
    }

    /// Also called after snapshots/strong edges are released: symbols and acyclic descendants
    /// can die during that phase, so a single pre-sweep drain is insufficient.
    #[inline(never)]
    pub(crate) fn gc_drain_weak_deaths(&mut self) {
        loop {
            let deaths = self.weak_metadata.take_deaths();
            if deaths.is_empty() {
                break;
            }
            for target in deaths {
                debug_assert!(
                    target.upgrade().is_none(),
                    "destruction notification has a live referent"
                );
                self.clear_weak_identity(target.key());
            }
        }
    }

    pub(crate) fn gc_schedule_finalization_cleanup(&mut self, dead: &FastSet<usize>) {
        let ready: Vec<_> = self.weak_metadata.ready_registries.drain().collect();
        for owner in ready {
            if dead.contains(&owner) {
                continue;
            }
            let Some(registry) = self.finalization_registries.get_mut(&owner) else {
                continue;
            };
            if !registry.cleanup_scheduled && registry.cells.has_ready() {
                if let Some(object) = self.gc_pins.get(&owner) {
                    registry.cleanup_scheduled = true;
                    self.pending_finalization_cleanup
                        .push_back(Value::Obj(object.clone()));
                }
            }
        }
        let ready = &mut self.weak_metadata.ready_registries;
        if ready.capacity() > 64 {
            ready.shrink_to(16);
        }
    }

    pub(crate) fn gc_forget_weak_owners(&mut self, dead: &FastSet<usize>) {
        fn owners<V>(table: &FastMap<usize, V>, dead: &FastSet<usize>) -> Vec<usize> {
            if table.len() < dead.len() {
                table
                    .keys()
                    .filter(|owner| dead.contains(owner))
                    .copied()
                    .collect()
            } else {
                dead.iter()
                    .filter(|owner| table.contains_key(owner))
                    .copied()
                    .collect()
            }
        }
        for owner in owners(&self.weak_refs, dead) {
            if let Some(Some(target)) = self.weak_refs.remove(&owner) {
                self.weak_metadata
                    .unsubscribe(target.key(), Subscriber::WeakRef(owner));
            }
        }
        for owner in owners(&self.weak_collection_data, dead) {
            let entries = self.weak_collection_data.remove(&owner).unwrap();
            self.weak_collection_index.remove(&owner);
            for (target, _) in &entries {
                self.weak_metadata
                    .unsubscribe(target.key(), Subscriber::Collection(owner));
            }
            drop(entries);
        }
        for owner in owners(&self.finalization_registries, dead) {
            let registry = self.finalization_registries.remove(&owner).unwrap();
            self.weak_metadata.ready_registries.remove(&owner);
            for target in registry.cells.targets() {
                self.weak_metadata
                    .unsubscribe(target.key(), Subscriber::Registry(owner));
            }
            drop(registry);
        }
        shrink_map(&mut self.weak_refs);
        shrink_map(&mut self.weak_collection_data);
        shrink_map(&mut self.weak_collection_index);
        shrink_map(&mut self.finalization_registries);
    }

    /// Native base constructors graft their internal slots onto a derived instance. Subscription
    /// owner identities must migrate with those slots, including WeakRef/FinalizationRegistry.
    pub(crate) fn move_weak_slots(&mut self, source: usize, destination: usize) -> bool {
        let mut moved = false;
        if let Some(target) = self.weak_refs.remove(&source) {
            if let Some(target) = &target {
                self.weak_metadata
                    .subscribe(target.clone(), Subscriber::WeakRef(destination));
                self.weak_metadata
                    .unsubscribe(target.key(), Subscriber::WeakRef(source));
            }
            self.weak_refs.insert(destination, target);
            moved = true;
        }
        if let Some(entries) = self.weak_collection_data.remove(&source) {
            for (target, _) in &entries {
                self.weak_metadata
                    .subscribe(target.clone(), Subscriber::Collection(destination));
                self.weak_metadata
                    .unsubscribe(target.key(), Subscriber::Collection(source));
            }
            self.weak_collection_data.insert(destination, entries);
            if let Some(index) = self.weak_collection_index.remove(&source) {
                self.weak_collection_index.insert(destination, index);
            }
            moved = true;
        }
        if let Some(registry) = self.finalization_registries.remove(&source) {
            for target in registry.cells.targets() {
                self.weak_metadata
                    .subscribe(target.clone(), Subscriber::Registry(destination));
                self.weak_metadata
                    .unsubscribe(target.key(), Subscriber::Registry(source));
            }
            self.finalization_registries.insert(destination, registry);
            if self.weak_metadata.ready_registries.remove(&source) {
                self.weak_metadata.ready_registries.insert(destination);
            }
            moved = true;
        }
        moved
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{Gc, GcCause, Object};
    use crate::{Completion, Engine};

    fn eval(engine: &mut Engine, source: &str, expected: &str) {
        match engine
            .eval(source, false)
            .expect("weak metadata fixture parses")
        {
            Completion::Value(value) => assert_eq!(value, expected),
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    fn reference(interp: &mut Interp, target: &WeakTarget) -> Gc {
        let owner = Object::new(None);
        let identity = Rc::as_ptr(&owner) as usize;
        interp.gc_pin(&owner);
        interp.weak_refs.insert(identity, Some(target.clone()));
        interp
            .weak_metadata
            .subscribe(target.clone(), Subscriber::WeakRef(identity));
        owner
    }

    #[test]
    fn weak_metadata_pending_count_is_exact_deduplicated_and_accounted() {
        let mut engine = Engine::new();
        let queue = DeathQueue::default();
        assert_eq!(queue.allocated_bytes(), size_of::<DeathQueue>());
        assert!(
            size_of::<DeathQueue>()
                >= size_of::<Cell<usize>>() + size_of::<RefCell<FastMap<WeakKey, WeakTarget>>>()
        );
        let object_value = Value::Obj(Object::new(None));
        let symbol_value = engine.interp.new_symbol(None);
        let object = WeakTarget::of(&object_value).unwrap();
        let symbol = WeakTarget::of(&symbol_value).unwrap();
        drop(object_value);
        drop(symbol_value);
        let check = |expected| {
            assert_eq!(queue.pending_len(), expected);
            assert_eq!(queue.entries.borrow().len(), expected);
            assert_eq!(
                queue.allocated_bytes(),
                size_of::<DeathQueue>()
                    + queue.entries.borrow().capacity() * size_of::<(WeakKey, WeakTarget)>()
            );
        };
        for _ in 0..3 {
            queue.enqueue(object.clone());
            check(1);
        }
        queue.remove(symbol.key()); // an absent identity never decrements the count
        check(1);
        queue.enqueue(symbol.clone());
        check(2);
        queue.remove(object.key());
        check(1);
        queue.remove(object.key());
        check(1);
        let batch = queue.take_all();
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].key(), symbol.key());
        check(0);
        // Empty take/remove cannot borrow or scan even if retained capacity exists.
        let borrowed = queue.entries.borrow_mut();
        assert!(queue.take_all().is_empty());
        queue.remove(symbol.key());
        assert_eq!(queue.pending_len(), 0);
        drop(borrowed);
        for _ in 0..3 {
            queue.enqueue(object.clone());
            check(1);
            assert_eq!(queue.take_all().len(), 1);
            check(0);
        }
    }

    #[test]
    fn weak_metadata_already_dead_subscription_counts_each_queue_once() {
        for symbol in [false, true] {
            let mut engine = Engine::new();
            let value = if symbol {
                engine.interp.new_symbol(None)
            } else {
                Value::Obj(Object::new(None))
            };
            let target = WeakTarget::of(&value).unwrap();
            drop(value);
            assert!(target.upgrade().is_none());
            let mut first = WeakMetadata::default();
            let mut second = WeakMetadata::default();
            first.subscribe(target.clone(), Subscriber::WeakRef(1));
            first.subscribe(target.clone(), Subscriber::WeakRef(2));
            second.subscribe(target.clone(), Subscriber::WeakRef(3));
            crate::value::observe_weak_target(&target, &second.deaths);
            assert_eq!(first.deaths.pending_len(), 1);
            assert_eq!(second.deaths.pending_len(), 1);
            first.unsubscribe(target.key(), Subscriber::WeakRef(1));
            assert_eq!(first.deaths.pending_len(), 1);
            first.unsubscribe(target.key(), Subscriber::WeakRef(2));
            assert_eq!(first.deaths.pending_len(), 0);
            assert_eq!(first.deaths.entries.borrow().len(), 0);
            assert_eq!(second.take_deaths().len(), 1);
            assert_eq!(second.deaths.pending_len(), 0);
            assert_eq!(
                second.take_subscribers(target.key()),
                vec![Subscriber::WeakRef(3)]
            );
            assert_eq!(
                second.deaths.pending_len(),
                0,
                "batch consumption must not decrement twice"
            );
            assert!(second.take_deaths().is_empty());
        }
    }

    #[test]
    fn weak_metadata_empty_safepoints_do_not_borrow_and_preserve_ready_cleanup() {
        let mut engine = Engine::new();
        let queue = engine.interp.weak_metadata.deaths.clone();
        {
            let _borrowed = queue.entries.borrow_mut();
            engine.interp.gc_weak_metadata_safepoint(false);
            engine.interp.gc_weak_metadata_safepoint(true);
            engine.interp.gc_drain_weak_deaths();
            assert_eq!(queue.pending_len(), 0);
        }
        // A previous drain can leave ready registries while the death journal is empty.
        // These cells request scheduling only at a task boundary; no callback runs here.
        engine.interp.gc_suppressed += 1;
        eval(&mut engine,
            "var calls=0, registry=new FinalizationRegistry(()=>calls++);registry.register(Symbol('ready'),7);calls", "0");
        engine.interp.gc_suppressed -= 1;
        engine.interp.gc_drain_weak_deaths();
        assert_eq!(queue.pending_len(), 0);
        assert!(!engine.interp.weak_metadata.ready_registries.is_empty());
        assert!(engine.interp.pending_finalization_cleanup.is_empty());
        engine.interp.gc_task_pending = false;
        let _borrowed = queue.entries.borrow_mut();
        engine.interp.gc_suppressed += 1;
        engine.interp.gc_weak_metadata_safepoint(true);
        assert!(
            !engine.interp.gc_task_pending,
            "suppression preserves the scheduling boundary"
        );
        engine.interp.gc_suppressed -= 1;
        engine.interp.gc_weak_metadata_safepoint(false);
        assert!(
            !engine.interp.gc_task_pending,
            "nonforced empty gate retains batching policy"
        );
        engine.interp.gc_weak_metadata_safepoint(true);
        assert!(engine.interp.gc_task_pending);
        assert!(engine.interp.pending_finalization_cleanup.is_empty());
    }

    #[test]
    fn weak_metadata_empty_queue_does_not_skip_kept_objects_or_host_job_hooks() {
        struct JobRoots {
            calls: Rc<Cell<usize>>,
            held: Option<Value>,
        }
        impl crate::host::HostGc for JobRoots {
            fn trace_gc(&self, _: &mut dyn crate::host::HostGcVisitor) {}
            fn sweep_gc(&mut self, _: &dyn Fn(&Value) -> bool) {}
            fn end_job(&mut self) {
                self.calls.set(self.calls.get() + 1);
                self.held = None;
            }
        }
        let mut engine = Engine::new();
        let calls = Rc::new(Cell::new(0));
        let value = Value::Obj(Object::new(None));
        let target = WeakTarget::of(&value).unwrap();
        let owner = reference(&mut engine.interp, &target);
        engine.interp.kept_alive.push(value.clone());
        engine.interp.host_state.put(JobRoots {
            calls: calls.clone(),
            held: Some(value),
        });
        engine.interp.host_state.register_gc::<JobRoots>();
        let queue = engine.interp.weak_metadata.deaths.clone();
        engine.interp.gc_weak_metadata_safepoint(true);
        assert_eq!(calls.get(), 0);
        assert!(
            target.upgrade().is_some(),
            "safepoints do not end synchronous work"
        );
        assert_eq!(engine.interp.kept_alive.len(), 1);
        engine.interp.gc_task_deferred = true;
        engine.interp.gc_task_boundary();
        assert_eq!(calls.get(), 1);
        assert!(engine.interp.kept_alive.is_empty());
        assert!(target.upgrade().is_none());
        assert!(engine.interp.weak_refs[&(Rc::as_ptr(&owner) as usize)].is_none());
        assert_eq!(queue.pending_len(), 0);
        // Even an entirely empty subsequent checkpoint must notify the host exactly once.
        let _borrowed = queue.entries.borrow_mut();
        engine.interp.gc_task_boundary();
        assert_eq!(calls.get(), 2);
        assert_eq!(queue.pending_len(), 0);
    }

    #[test]
    fn weak_metadata_minor_work_depends_on_dead_subscribers_not_stable_old_targets() {
        for tier in [
            crate::bytecode::Tier::Interp,
            crate::bytecode::Tier::Bytecode,
            crate::bytecode::Tier::Jit,
        ] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            eval(
                &mut engine,
                r#"
            var stableTargets=[], references=[], map=new WeakMap(), set=new WeakSet();
            var cleaned=[], registry=new FinalizationRegistry(value=>cleaned.push(value));
            for(let i=0;i<5000;i++) {
                let target={id:i}; stableTargets.push(target);
                references.push(new WeakRef(target)); map.set(target,i); set.add(target);
                registry.register(target,i);
            }
            true
        "#,
                "true",
            );
            engine.interp.gc_collect();
            let stable_count = engine.interp.weak_metadata.targets.len();
            eval(
                &mut engine,
                r#"
            (()=>{let target={};target.self=target;
                references.push(new WeakRef(target),new WeakRef(target));
                map.set(target,{});set.add(target);registry.register(target,'young');})();true
        "#,
                "true",
            );
            assert!(engine.interp.kept_alive.is_empty());
            engine.interp.gc_collect_young(GcCause::Explicit);
            assert_eq!(engine.interp.weak_metadata.cleared_subscribers, 5);
            assert_eq!(engine.interp.weak_metadata.targets.len(), stable_count);
            assert_eq!(
                engine
                    .interp
                    .weak_collection_data
                    .values()
                    .map(Vec::len)
                    .sum::<usize>(),
                10000
            );
            eval(
                &mut engine,
                "[references[5000].deref(),references[5001].deref(),map.get(stableTargets[4999])].join(':')",
                "::4999",
            );
            eval(&mut engine, "cleaned.join(',')", "young");
        }
    }

    #[test]
    fn weak_metadata_weakset_has_no_strong_dummy_target_for_add_or_constructor() {
        for symbol in [false, true] {
            for constructor in [false, true] {
                let mut engine = Engine::new();
                let source = format!(
                    r#"
                    var reference, set;
                    (()=>{{let target={};reference=new WeakRef(target);
                        set={};
                        if(!set.has(target)||set.add(target)!==set)throw Error('membership');
                    }})();true
                "#,
                    if symbol { "Symbol('set')" } else { "{}" },
                    if constructor {
                        "new WeakSet([target])"
                    } else {
                        "new WeakSet().add(target)"
                    }
                );
                eval(&mut engine, &source, "true");
                assert!(
                    engine
                        .interp
                        .weak_refs
                        .values()
                        .flatten()
                        .all(|target| target.upgrade().is_none()),
                    "WeakSet retained its acyclic target"
                );
                assert!(engine
                    .interp
                    .weak_collection_data
                    .values()
                    .flatten()
                    .all(|(_, value)| matches!(value, Value::Undefined)));
                engine.interp.gc_collect_young(GcCause::Explicit);
                assert!(engine
                    .interp
                    .weak_collection_data
                    .values()
                    .all(Vec::is_empty));
                eval(&mut engine, "reference.deref()===undefined", "true");
            }
        }
    }

    #[test]
    fn weak_metadata_collection_brand_is_internal_exact_and_survives_subclass_transplant() {
        for tier in [
            crate::bytecode::Tier::Interp,
            crate::bytecode::Tier::Bytecode,
            crate::bytecode::Tier::Jit,
        ] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            eval(
                &mut engine,
                r#"
                class M extends WeakMap {} class S extends WeakSet {}
                const key={},m=new M([[key,7]]),s=new S([key]);
                if(Reflect.ownKeys(m).length||Reflect.ownKeys(s).length)throw Error('internal slot exposed');
                m.__ck='WeakSet';s.__ck='WeakMap';
                if(m.get(key)!==7||!s.has(key))throw Error('property changed brand');
                delete m.__ck;delete s.__ck;
                Object.defineProperty(m,'__ck',{get(){throw Error('brand getter called')}});
                Object.defineProperty(s,'__ck',{get(){throw Error('brand getter called')}});
                function rejects(method,receiver) {
                    try { method.call(receiver,key,()=>1); }
                    catch(error) { if(error instanceof TypeError)return;throw error; }
                    throw Error('accepted wrong brand');
                }
                const mapMethods=['get','set','has','delete','getOrInsert','getOrInsertComputed'];
                const setMethods=['add','has','delete'];
                const fake={__ck:'WeakMap'}, trap={get(){throw Error('proxy trap called')}};
                for(const method of mapMethods) {
                    rejects(WeakMap.prototype[method],s);
                    rejects(WeakMap.prototype[method],new Proxy(m,trap));
                    rejects(WeakMap.prototype[method],fake);
                    rejects(WeakMap.prototype[method],null);
                }
                for(const method of setMethods) {
                    rejects(WeakSet.prototype[method],m);
                    rejects(WeakSet.prototype[method],new Proxy(s,trap));
                    rejects(WeakSet.prototype[method],fake);
                    rejects(WeakSet.prototype[method],null);
                }
                if(!WeakMap.prototype.has.call(m,key)||!WeakSet.prototype.has.call(s,key))throw Error('borrowed has');
                if(!WeakMap.prototype.delete.call(m,key)||!WeakSet.prototype.delete.call(s,key))throw Error('borrowed delete');
                if(m.has(key)||s.has(key))throw Error('delete did not remove');
                true
            "#,
                "true",
            );
        }
    }

    #[test]
    fn weak_metadata_acyclic_ephemeron_value_deaths_drain_to_fixed_point() {
        for symbol in [false, true] {
            let mut engine = Engine::new();
            let source = format!(
                r#"
                var map=new WeakMap(), refs=[], cleaned=[];
                var registry=new FinalizationRegistry(x=>cleaned.push(x));
                (()=>{{let a={{}},b={};
                    map.set(a,b);refs.push(new WeakRef(a),new WeakRef(b));
                    registry.register(a,'a');registry.register(b,'b');}})();true
            "#,
                if symbol { "Symbol('cascade')" } else { "{}" }
            );
            // Isolate the collector's fixed point from normal job-boundary queue draining.
            engine.interp.gc_suppressed += 1;
            eval(&mut engine, &source, "true");
            engine.interp.gc_suppressed -= 1;
            engine.interp.clear_kept_objects();
            // a died at the job boundary, before any nursery snapshot; b is held only by its
            // ephemeron payload. Processing a must release b and consume b's new notification.
            assert_ne!(engine.interp.weak_metadata.deaths.pending_len(), 0);
            engine.interp.gc_collect_young(GcCause::Explicit);
            assert!(engine
                .interp
                .weak_collection_data
                .values()
                .all(Vec::is_empty));
            assert!(engine.interp.weak_refs.values().all(Option::is_none));
            assert!(engine.interp.weak_metadata.targets.is_empty());
            assert_eq!(engine.interp.weak_metadata.deaths.pending_len(), 0);
            eval(&mut engine, "true", "true");
            eval(&mut engine, "cleaned.sort().join(',')", "a,b");
        }
    }

    #[test]
    fn weak_metadata_symbol_only_churn_is_bounded_inside_one_job() {
        let mut engine = Engine::new();
        let global = engine.interp.global.clone();
        engine
            .interp
            .def_method(&global, "checkWeakChurn", 0, |interp, _, _| {
                assert!(interp.weak_metadata.deaths.pending_len() <= DEATH_BATCH_LIMIT);
                assert!(interp.weak_metadata.targets.len() <= DEATH_BATCH_LIMIT + 1);
                for entries in interp.weak_collection_data.values() {
                    assert!(entries.len() <= DEATH_BATCH_LIMIT + 1);
                    assert!(entries.capacity() <= DEATH_BATCH_LIMIT * 4);
                }
                Ok(Value::Undefined)
            });
        let allocated = crate::value::heap_allocated_objects(&engine.interp.gc_heap);
        eval(
            &mut engine,
            r#"
            var map=new WeakMap(),set=new WeakSet();
            for(let i=0;i<30000;i++) {
                let target=Symbol('churn');map.set(target,i);set.add(target);
                if((i&1023)===0)checkWeakChurn();
            }
            true
        "#,
            "true",
        );
        assert!(
            crate::value::heap_allocated_objects(&engine.interp.gc_heap) - allocated < 1000,
            "fixture accidentally relied on heavy object allocation GC"
        );
        assert!(engine.interp.weak_metadata.targets.is_empty());
        assert_eq!(engine.interp.weak_metadata.deaths.pending_len(), 0);
        assert!(engine.interp.weak_metadata.targets.capacity() <= 64);
        assert!(
            engine
                .interp
                .weak_metadata
                .deaths
                .entries
                .borrow()
                .capacity()
                <= 64
        );
        assert!(engine
            .interp
            .weak_collection_data
            .values()
            .all(|entries| entries.is_empty() && entries.capacity() <= 64));
    }

    #[test]
    fn weak_metadata_safepoint_honors_suppression_and_validates_cleanup_owners() {
        let mut engine = Engine::new();
        let value = engine.interp.new_symbol(None);
        let target = WeakTarget::of(&value).unwrap();
        let reference = reference(&mut engine.interp, &target);
        drop(value);
        engine.interp.gc_suppressed += 1;
        engine.interp.gc_weak_metadata_safepoint(true);
        assert_eq!(engine.interp.weak_metadata.deaths.pending_len(), 1);
        engine.interp.gc_suppressed -= 1;
        engine.interp.gc_weak_metadata_safepoint(true);
        assert!(engine.interp.weak_refs[&(Rc::as_ptr(&reference) as usize)].is_none());
        for keep_owner in [false, true] {
            let mut engine = Engine::new();
            let source = format!(
                r#"
                var results=[],registry;
                (()=>{{let owner=new FinalizationRegistry(value=>results.push(value));
                    for(let i=0;i<600;i++)owner.register(Symbol('ready'),i);
                    {}
                }})();results.length
            "#,
                if keep_owner { "registry=owner;" } else { "" }
            );
            eval(&mut engine, &source, "0");
            assert_eq!(
                !engine.interp.pending_finalization_cleanup.is_empty(),
                keep_owner
            );
            eval(&mut engine, "true", "true");
            eval(
                &mut engine,
                "results.length",
                if keep_owner { "600" } else { "0" },
            );
        }
    }

    #[test]
    fn weak_metadata_object_and_symbol_notify_separate_observer_heaps() {
        for symbol in [false, true] {
            let mut first = Engine::new();
            let target = if symbol {
                first.interp.new_symbol(None)
            } else {
                Value::Obj(Object::new(None))
            };
            let weak = WeakTarget::of(&target).unwrap();
            let first_ref = reference(&mut first.interp, &weak);
            let first_ref_again = reference(&mut first.interp, &weak);
            let mut second = Interp::new_with_symbol_agent(first.interp.symbol_agent.clone());
            let second_ref = reference(&mut second, &weak);
            drop(target); // currently active Agent/heap is deliberately the *other* one
            assert_eq!(first.interp.weak_metadata.deaths.pending_len(), 1);
            assert_eq!(second.weak_metadata.deaths.pending_len(), 1);
            first.interp.gc_clear_weak_targets(&FastSet::default());
            second.gc_clear_weak_targets(&FastSet::default());
            assert!(first.interp.weak_refs[&(Rc::as_ptr(&first_ref) as usize)].is_none());
            assert!(first.interp.weak_refs[&(Rc::as_ptr(&first_ref_again) as usize)].is_none());
            assert!(second.weak_refs[&(Rc::as_ptr(&second_ref) as usize)].is_none());
            assert_eq!(first.interp.weak_metadata.deaths.pending_len(), 0);
            assert_eq!(second.weak_metadata.deaths.pending_len(), 0);
        }
    }

    #[test]
    fn weak_metadata_dead_observer_heap_does_not_retain_target_or_other_queues() {
        let mut first = Engine::new();
        let target = Value::Obj(Object::new(None));
        let weak = WeakTarget::of(&target).unwrap();
        let owner = reference(&mut first.interp, &weak);
        let queue = Rc::downgrade(&first.interp.weak_metadata.deaths);
        let mut second = Engine::new();
        let other_owner = reference(&mut second.interp, &weak);
        drop(owner);
        drop(first);
        assert!(
            queue.upgrade().is_none(),
            "target retained its observer heap queue"
        );
        drop(target);
        second.interp.gc_clear_weak_targets(&FastSet::default());
        assert!(second.interp.weak_refs[&(Rc::as_ptr(&other_owner) as usize)].is_none());
    }

    #[test]
    fn weak_metadata_kept_target_and_owner_retirement_preserve_job_semantics() {
        let mut engine = Engine::new();
        let target = Value::Obj(Object::new(None));
        let weak = WeakTarget::of(&target).unwrap();
        let owner = reference(&mut engine.interp, &weak);
        engine.interp.kept_alive.push(target.clone());
        drop(target);
        engine.interp.gc_collect_young(GcCause::Explicit);
        assert!(
            weak.upgrade().is_some(),
            "same-job KeptAlive target was cleared"
        );
        engine.interp.clear_kept_objects();
        engine.interp.gc_collect_young(GcCause::Explicit);
        assert!(engine.interp.weak_refs[&(Rc::as_ptr(&owner) as usize)].is_none());
        let other = Value::Obj(Object::new(None));
        let other_weak = WeakTarget::of(&other).unwrap();
        let ephemeral_owner = reference(&mut engine.interp, &other_weak);
        drop(ephemeral_owner);
        engine.interp.gc_collect_young(GcCause::Explicit);
        assert!(!engine
            .interp
            .weak_metadata
            .targets
            .contains_key(&other_weak.key()));
        assert!(other_weak.upgrade().is_some());
    }

    #[test]
    fn weak_metadata_unregister_and_delete_bound_symbol_only_churn() {
        let mut engine = Engine::new();
        eval(
            &mut engine,
            r#"
            var token={},map=new WeakMap(),registry=new FinalizationRegistry(()=>{});
            for(let i=0;i<10000;i++) {
                let key=Symbol('retired'); map.set(key,i); registry.register(key,i,token);
                map.delete(key); key=null; registry.unregister(token);
            }
            true
        "#,
            "true",
        );
        let metadata = &engine.interp.weak_metadata;
        assert!(metadata.targets.is_empty());
        assert_eq!(metadata.deaths.pending_len(), 0);
        assert!(metadata.targets.capacity() <= 64);
        assert!(metadata.deaths.entries.borrow().capacity() <= 64);
        assert!(engine
            .interp
            .weak_collection_data
            .values()
            .all(|entries| entries.capacity() <= 64));
    }

    #[test]
    fn weak_metadata_subclass_slots_move_with_reverse_subscriptions_all_tiers() {
        for tier in [
            crate::bytecode::Tier::Interp,
            crate::bytecode::Tier::Bytecode,
            crate::bytecode::Tier::Jit,
        ] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            eval(
                &mut engine,
                r#"
                class R extends WeakRef {} class M extends WeakMap {} class S extends WeakSet {}
                class F extends FinalizationRegistry {}
                var refs=[],map=new M(),set=new S(),cleaned=[],registry=new F(x=>cleaned.push(x));
                (()=>{let target={};target.self=target;refs.push(new R(target));
                    map.set(target,1);set.add(target);registry.register(target,'done');})();true
            "#,
                "true",
            );
            engine.interp.gc_collect();
            assert!(engine.interp.weak_refs.values().all(Option::is_none));
            assert!(engine
                .interp
                .weak_collection_data
                .values()
                .all(Vec::is_empty));
            eval(&mut engine, "true", "true");
            eval(&mut engine, "cleaned.join(',')", "done");
        }
    }

    #[test]
    fn weak_metadata_queued_headers_prevent_address_aba() {
        let mut engine = Engine::new();
        let target = Value::Obj(Object::new(None));
        let weak = WeakTarget::of(&target).unwrap();
        let key = weak.key();
        engine
            .interp
            .weak_metadata
            .subscribe(weak, Subscriber::WeakRef(1));
        drop(target);
        for _ in 0..1000 {
            let value = Value::Obj(Object::new(None));
            assert_ne!(WeakKey::of(&value).unwrap(), key);
        }
        engine
            .interp
            .weak_metadata
            .unsubscribe(key, Subscriber::WeakRef(1));
        assert_eq!(engine.interp.weak_metadata.deaths.pending_len(), 0);
        for _ in 0..1000 {
            let value = Value::Obj(Object::new(None));
            let weak = WeakTarget::of(&value).unwrap();
            engine
                .interp
                .weak_metadata
                .subscribe(weak.clone(), Subscriber::WeakRef(2));
            engine
                .interp
                .weak_metadata
                .unsubscribe(weak.key(), Subscriber::WeakRef(2));
        }
        assert!(engine.interp.weak_metadata.targets.is_empty());
    }

    #[test]
    fn weak_metadata_symbol_destruction_does_not_reborrow_its_agent() {
        let mut engine = Engine::new();
        let Value::Sym(symbol) = engine.interp.new_symbol(None) else {
            unreachable!()
        };
        let target = WeakTarget::Symbol(Rc::downgrade(&symbol), symbol.id);
        let owner = reference(&mut engine.interp, &target);
        let agent = engine.interp.symbol_agent.clone();
        let borrowed = agent.borrow_mut();
        drop(symbol);
        drop(borrowed);
        engine.interp.gc_drain_weak_deaths();
        assert!(engine.interp.weak_refs[&(Rc::as_ptr(&owner) as usize)].is_none());
    }

    #[test]
    fn weak_metadata_symbol_property_keys_are_strong_and_registered_symbols_stay_ineligible() {
        let mut engine = Engine::new();
        eval(
            &mut engine,
            r#"
            var key=Symbol('property'), holder={[key]:1}, reference=new WeakRef(key);
            key=null;
            var refused=0;
            try { new WeakRef(Symbol.for('registered')); } catch(e) { if(e instanceof TypeError)refused++; }
            try { new WeakMap().set(Symbol.for('registered'),1); } catch(e) { if(e instanceof TypeError)refused++; }
            refused
        "#,
            "2",
        );
        engine.interp.gc_collect();
        eval(&mut engine, "typeof reference.deref()", "symbol");
        eval(&mut engine, "holder=null;true", "true");
        engine.interp.gc_collect_young(GcCause::Explicit);
        eval(&mut engine, "String(reference.deref()===undefined)", "true");
    }
}
