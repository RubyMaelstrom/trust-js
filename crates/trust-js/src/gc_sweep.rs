//! Remove the already-classified dead identities from internal-slot tables. Root discovery,
//! ephemerons and atomic weak-target clearing happen before this step; no liveness decision
//! is made here. Choose key removal or a table scan according to the smaller working set.

use crate::fasthash::FastSet;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

pub(crate) fn map<T, S: BuildHasher>(table: &mut HashMap<usize, T, S>, dead: &FastSet<usize>) {
    if table.is_empty() || dead.is_empty() {
        return;
    }
    // HashMap::retain scans capacity, not just length. A sparse, large table must not
    // be rescanned for a handful of dead objects; conversely, closure churn must not
    // probe every internal-slot table once for every ordinary function/prototype.
    if dead.len() < table.capacity() {
        for key in dead {
            table.remove(key);
        }
    } else {
        table.retain(|key, _| !dead.contains(key));
    }
}

pub(crate) fn set<S: BuildHasher>(table: &mut HashSet<usize, S>, dead: &FastSet<usize>) {
    if table.is_empty() || dead.is_empty() {
        return;
    }
    if dead.len() < table.capacity() {
        for key in dead {
            table.remove(key);
        }
    } else {
        table.retain(|key| !dead.contains(key));
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
                let dead: FastSet<_> = (0..dead_count).map(|key| key * 2).collect();
                map(&mut table, &dead);
                set(&mut set_table, &dead);
                for (key, value) in values.iter().enumerate() {
                    let live = !dead.contains(&key);
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
