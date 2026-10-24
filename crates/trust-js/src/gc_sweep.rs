//! Remove the already-classified dead identities from internal-slot tables. Root discovery,
//! ephemerons and atomic weak-target clearing happen before this step; no liveness decision
//! is made here. Choose key removal or a table scan according to the smaller working set.

use crate::gc_core::{self, DescId};
use crate::value::Object;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

/// The objects a collection found dead, which the engine evicts from its tables between the
/// collection's phases ([`gc_core::Heap::start_marking`], [`gc_core::Heap::finish_collection`]):
/// the cells of the object descriptor that marking left unmarked, zombies apart. Walks
/// ([`DeadObjects::iter`]) use the list that [`gc_core::Heap::dead_cells`] made as the view was
/// built; membership ([`DeadObjects::contains`]) reads the heap's bits, without hashing. Between
/// the phases nothing changes a dead cell's bits (a cell allocated there is allocated marked),
/// so the two agree. After the sweep the bits no longer say what died (marks are cleared, dead
/// cells freed or kept as zombies, pages released) and only the list stays meaningful: debug
/// builds assert that every query comes between the phases of the view's collection, and that
/// every answer is the list's.
pub(crate) struct DeadObjects<'h> {
    /// The dead cells' addresses, listed on the first walk ([`DeadObjects::iter`]): a table
    /// smaller than the dead count is swept by membership alone and needs no list.
    list: std::cell::OnceCell<Vec<usize>>,
    count: usize,
    membership: Membership<'h>,
}

/// What answers [`DeadObjects::contains`].
enum Membership<'h> {
    /// The heap's bits for its object descriptor ([`gc_core::Heap::is_dead_small_cell`]).
    Bits {
        core: &'h RefCell<gc_core::Heap>,
        desc: DescId,
        /// The collection that listed the dead: every query comes between its phases.
        #[cfg(debug_assertions)]
        collection: u64,
        /// The list as a set: every answer must be its.
        #[cfg(debug_assertions)]
        listed: crate::fasthash::FastSet<usize>,
    },
    /// A test view's list, as a set.
    #[cfg(test)]
    List(crate::fasthash::FastSet<usize>),
}

// Objects are never large cells (value.rs keeps an object's cell within the 192-byte class), so
// membership reads the pages alone and never the heap's map of large cells.
const _: () = assert!(std::mem::size_of::<RefCell<Object>>() <= gc_core::LARGEST_SMALL_CELL);

impl<'h> DeadObjects<'h> {
    /// The objects of `heap` found dead by the collection between whose phases the heap is.
    pub(crate) fn new(heap: &'h crate::value::GcState) -> Self {
        let core = heap.core.borrow();
        let count = core.dead_count(heap.object_desc);
        #[cfg(debug_assertions)]
        let collection = core
            .collection_in_progress()
            .expect("dead objects are listed between a collection's phases");
        #[cfg(debug_assertions)]
        let listed: crate::fasthash::FastSet<usize> =
            core.dead_cells(heap.object_desc).into_iter().collect();
        drop(core);
        Self {
            membership: Membership::Bits {
                core: &heap.core,
                desc: heap.object_desc,
                #[cfg(debug_assertions)]
                collection,
                #[cfg(debug_assertions)]
                listed,
            },
            list: std::cell::OnceCell::new(),
            count,
        }
    }

    /// A view of `list` with no heap behind it, whose membership is the list's.
    #[cfg(test)]
    pub(crate) fn from_list(list: Vec<usize>) -> Self {
        Self {
            membership: Membership::List(list.iter().copied().collect()),
            count: list.len(),
            list: list.into(),
        }
    }

    /// Whether the object whose cell starts at `addr` is dead. Only between the phases of the
    /// view's collection (see [`DeadObjects`]).
    pub(crate) fn contains(&self, addr: usize) -> bool {
        match &self.membership {
            Membership::Bits {
                core,
                desc,
                #[cfg(debug_assertions)]
                collection,
                #[cfg(debug_assertions)]
                listed,
            } => {
                // Borrowed per query: the window between the phases may allocate.
                let core = core.borrow();
                #[cfg(debug_assertions)]
                assert_eq!(
                    core.collection_in_progress(),
                    Some(*collection),
                    "a dead-object view queried outside its collection's phases"
                );
                let dead = core.is_dead_small_cell(addr, *desc);
                #[cfg(debug_assertions)]
                assert_eq!(
                    dead,
                    listed.contains(&addr),
                    "the heap's bits and the list of dead objects disagree on {addr:#x}"
                );
                dead
            }
            #[cfg(test)]
            Membership::List(listed) => listed.contains(&addr),
        }
    }

    /// How many objects died.
    pub(crate) fn len(&self) -> usize {
        self.count
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The dead objects' addresses, in the list's order. The first walk lists them from the
    /// heap's bits, so it must come between the phases of the view's collection.
    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.list
            .get_or_init(|| match &self.membership {
                Membership::Bits {
                    core,
                    desc,
                    #[cfg(debug_assertions)]
                    collection,
                    #[cfg(debug_assertions)]
                        listed: _,
                } => {
                    let core = core.borrow();
                    #[cfg(debug_assertions)]
                    assert_eq!(
                        core.collection_in_progress(),
                        Some(*collection),
                        "dead objects listed outside their collection's phases"
                    );
                    core.dead_cells(*desc)
                }
                #[cfg(test)]
                Membership::List(_) => unreachable!("a test view's list is given"),
            })
            .iter()
            .copied()
    }
}

thread_local! {
    static EVICTING: Cell<bool> = const { Cell::new(false) };
}

/// Marks a collector's eviction step on this thread while it lives (see
/// `Interp::gc_evict_dead_owners`).
pub(crate) struct Evicting(bool);

impl Evicting {
    pub(crate) fn enter() -> Self {
        Self(EVICTING.replace(true))
    }
}

impl Drop for Evicting {
    fn drop(&mut self) {
        EVICTING.set(self.0);
    }
}

/// Whether this thread is in a collector's eviction step, which runs no VM code: what it drops
/// ends as memory (I5).
pub(crate) fn evicting() -> bool {
    EVICTING.get()
}

/// Remove the dead objects' entries from `table`, walking the smaller side: each dead object's
/// key is removed, which hashes it, or, when the dead outnumber the table's capacity (closure
/// churn), the table is scanned once, each key asking the view, which tests the heap's bits.
pub(crate) fn map<T, S: BuildHasher>(table: &mut HashMap<usize, T, S>, dead: &DeadObjects<'_>) {
    if table.is_empty() || dead.is_empty() {
        return;
    }
    // HashMap::retain scans capacity, not just length. A sparse, large table must not
    // be rescanned for a handful of dead objects; conversely, closure churn must not
    // probe every internal-slot table once for every ordinary function/prototype.
    if dead.len() < table.capacity() {
        for key in dead.iter() {
            table.remove(&key);
        }
    } else {
        table.retain(|&key, _| !dead.contains(key));
    }
}

/// [`map`] for a set of addresses.
pub(crate) fn set<S: BuildHasher>(table: &mut HashSet<usize, S>, dead: &DeadObjects<'_>) {
    if table.is_empty() || dead.is_empty() {
        return;
    }
    if dead.len() < table.capacity() {
        for key in dead.iter() {
            table.remove(&key);
        }
    } else {
        table.retain(|&key| !dead.contains(key));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fasthash::FastMap;
    use std::rc::Rc;

    #[test]
    fn adaptive_table_sweeps_match_individual_removal_and_release_only_dead_owners() {
        for capacity in [0, 16, 4096] {
            for dead_count in [0, 1, 8, 8192] {
                let values: Vec<_> = (0..16).map(Rc::new).collect();
                let mut table = FastMap::with_capacity_and_hasher(capacity, Default::default());
                let mut set_table = HashSet::with_capacity(capacity);
                for (key, value) in values.iter().enumerate() {
                    table.insert(key, value.clone());
                    set_table.insert(key);
                }
                let dead = DeadObjects::from_list((0..dead_count).map(|key| key * 2).collect());
                map(&mut table, &dead);
                set(&mut set_table, &dead);
                for (key, value) in values.iter().enumerate() {
                    let live = !dead.contains(key);
                    assert_eq!(table.contains_key(&key), live);
                    assert_eq!(set_table.contains(&key), live);
                    assert_eq!(Rc::strong_count(value), 1 + usize::from(live));
                }
                // Repeated cleanup and absent identities must not disturb survivors.
                map(&mut table, &dead);
                set(&mut set_table, &dead);
                assert_eq!(table.len(), set_table.len());
            }
        }
    }

    #[test]
    fn gc_table_sweep_preserves_live_internal_slots_and_releases_dead_peers() {
        use crate::bytecode::Tier;
        use crate::{Completion, Engine};

        fn eval(engine: &mut Engine, source: &str) -> String {
            match engine.eval(source, false).expect("GC fixture parses") {
                Completion::Value(value) => value,
                Completion::Throw { name, message } => panic!("{name}: {message}"),
            }
        }

        // Weak targets are cleared as one chosen set before internal slots are evicted
        // (ECMA-262 #sec-weakref-execution). Retained peers, including a foreign function's
        // logical [[Realm]] and suspended generator state, must remain fully usable.
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            eval(
                &mut engine,
                r#"
                var keep, dead, resolved;
                function fixture(n, suspend) {
                    const buffer = new ArrayBuffer(8);
                    const typed = new Uint8Array(buffer); typed[0] = n;
                    const view = new DataView(buffer);
                    const map = new Map([[buffer, typed]]);
                    const set = new Set([view]);
                    const regex = /a+/g;
                    const proxy = new Proxy({n}, {});
                    const promise = Promise.resolve(n);
                    const foreign = $262.createRealm().evalScript(
                        '(function(n) { return new Uint8Array([n])[0]; })');
                    function* sequence() { yield n; yield typed[0] + 1; return n + 2; }
                    const result = {buffer, typed, view, map, set, regex, proxy,
                                    promise, foreign};
                    if (suspend) {
                        result.generator = sequence(); result.generator.next();
                    }
                    result.self = result;
                    return result;
                }
                keep = fixture(17, true);
                (() => {
                    const gone = fixture(23);
                    // All these weak handles must empty together, not keep each other live.
                    dead = [gone, gone.buffer, gone.typed, gone.view, gone.map, gone.set,
                            gone.regex, gone.proxy, gone.promise, gone.foreign]
                           .map(value => new WeakRef(value));
                })();
                'ready'
                "#,
            );
            for _ in 0..3 {
                engine.interp.gc_collect();
                assert_eq!(
                    eval(
                        &mut engine,
                        r#"
                        keep.regex.lastIndex = 0;
                        [keep.typed[0], keep.view.getUint8(0),
                         keep.map.get(keep.buffer) === keep.typed, keep.set.has(keep.view),
                         keep.regex.exec('baa')[0], keep.proxy.n, keep.foreign(41)].join(':')
                        "#,
                    ),
                    "17:17:true:true:aa:17:41",
                    "{tier:?}"
                );
                assert_eq!(
                    eval(
                        &mut engine,
                        "dead.map(ref => ref.deref() === undefined).join(',')"
                    ),
                    "true,true,true,true,true,true,true,true,true,true",
                    "{tier:?} dead internal-slot owners"
                );
            }
            assert_eq!(eval(&mut engine, "keep.generator.next().value"), "18");
            eval(
                &mut engine,
                "keep.promise.then(n => resolved = n); 'queued'",
            );
            assert_eq!(eval(&mut engine, "resolved"), "17");
            // Release the live graph, then repeatedly allocate different metadata-bearing
            // objects so allocator address reuse cannot inherit stale internal slots.
            eval(
                &mut engine,
                "keep.generator.return(); keep = null; 'released'",
            );
            for _ in 0..3 {
                engine.interp.gc_collect();
                assert_eq!(
                    eval(
                        &mut engine,
                        "var next = fixture(29);\
                         var check = next.view.getUint8(0) + next.foreign(3);\
                         next = null; check",
                    ),
                    "32",
                    "{tier:?}"
                );
            }
        }
    }
}
