//! ECMA-262 e28783d5: Map/Set live iteration, Set composition, IteratorClose, GeneratorValidate,
//! CanonicalizeKeyedCollectionKey, and #sec-liveness. All language fixtures run in all three tiers.

use crate::bytecode::Tier;
use crate::value::Value;
use crate::{Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine
            .eval(source, false)
            .expect("collection fixture parses")
        {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        engine.interp.gc_collect();
    }
}

#[test]
fn ordered_collection_exact_size_with_colliding_hashes() {
    check(
        r#"
        var v = new DataView(new ArrayBuffer(8));
        v.setUint32(0, 0x91054a88, true); v.setUint32(4, 0xbe60db93, true);
        var k = v.getFloat64(0, true), m = new Map([[undefined, 'u'], [k, 'n']]);
        var s = new Set([undefined, k]), out = [m.size, m.get(undefined), m.get(k), s.size];
        m.delete(undefined); m.set(undefined, 'again');
        out.push(m.size, Array.from(m.values()).join(','));
        out.push(s.difference(new Set([k])).size, s.intersection(new Set([k])).size);
        m.set(-0, 1); m.set(0, 2); m.set(NaN, 3); m.set(Number('bad'), 4);
        out.push(m.size, m.get(NaN), 1/Array.from(m.keys())[2]);
        out.join('|')
    "#,
        "2|u|n|2|2|n,again|1|1|4|4|Infinity",
    );
}

#[test]
fn ordered_collection_compaction_preserves_distinct_live_cursors() {
    check(r#"
        var m = new Map(); for (var n = 0; n < 128; n++) m.set(n, n);
        var a = m.keys(), b = m.entries(), c = m.keys();
        a.next(); for (var n = 0; n < 50; n++) b.next();
        for (var n = 0; n < 127; n++) c.next();
        for (var n = 0; n < 96; n++) m.delete(n);
        m.set(96, 'updated');
        var out = [a.next().value, b.next().value.join(':'), c.next().value];
        m.delete(96); m.set(96, 'new');
        out.push(Array.from(a).join(','), c.next().value, c.next().done);
        out.join('|')
    "#, "96|96:updated|127|97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,96|96|true");
}

#[test]
fn ordered_collection_clear_preserves_live_not_completed_iterators() {
    check(
        r#"
        var m = new Map([[1, 'a'], [2, 'b']]);
        var active = m.keys(), dormant = m.values(), done = m.keys();
        active.next(); done.next(); done.next(); done.next();
        m.clear(); m.set(3, 'c');
        var out = [active.next().value, dormant.next().value, done.next().done];
        m.clear(); m.clear(); m.set(4, 'd');
        out.push(active.next().value, dormant.next().value, done.next().done);
        var empty = new Set(), waiting = empty.values();
        empty.clear(); empty.add(-0);
        out.push(1/waiting.next().value, waiting.next().done);
        empty.add(5); out.push(waiting.next().done);
        out.join('|')
    "#,
        "3|c|true|4|d|true|Infinity|true|true",
    );
}

#[test]
fn ordered_collection_foreach_reentrancy_clear_compaction_and_arguments() {
    check(
        r#"
        var m = new Map(); for (var n = 0; n < 128; n++) m.set(n, n);
        var out = [], receiver = {};
        m.forEach(function(value, key, owner) {
            if (this !== receiver || owner !== m || value !== key) throw Error('arguments');
            out.push(key);
            if (key === 0) for (var n = 1; n < 96; n++) m.delete(n);
            if (key === 96) m.forEach(function(v) {
                out.push('nested' + v);
                m.clear(); if (v === 0) m.set(200, 200);
            });
        }, receiver);
        var s = new Set([1, 2]); var seen = [];
        s.forEach(function(value, key, owner) {
            if (value !== key || owner !== s) throw Error('set arguments');
            seen.push(value); if (value === 1) { s.clear(); s.add(3); }
        });
        out.join(',') + '|' + seen.join(',')
    "#,
        "0,96,nested0,nested200|1,3",
    );
}

#[test]
fn ordered_collection_native_brands_cannot_be_forged_or_mutated() {
    check(
        r#"
        var m = new Map([[1, 2]]), s = new Set([1]);
        m.__ck = 'Set'; s.__ck = 'Map';
        var mi = m.entries(), si = s.values(), errors = 0;
        for (var test of [
            function() { Map.prototype.get.call(s, 1); },
            function() { Set.prototype.add.call(m, 2); },
            function() { mi.next.call(si); },
            function() { si.next.call(mi); },
            function() { mi.next.call({__ci_coll:m,__ci_index:0,__ci_kind:2}); },
            function() { mi.next.call(new Proxy(mi, {})); }
        ]) { try { test(); } catch (e) { if (!(e instanceof TypeError)) throw e; errors++; } }
        mi.__ci_done = true; mi.__ci_index = 100; mi.__ci_coll = null;
        [m.get(1), s.has(1), errors, mi.next().value.join(','),
         Reflect.ownKeys(s.values()).length].join('|')
    "#,
        "2|true|6|1,2|0",
    );
}

#[test]
fn ordered_collection_live_set_algebra_callbacks_survive_compaction() {
    check(
        r#"
        function exercise(method, answer) {
            var s = new Set(); for (var n=0; n<128; n++) s.add(n);
            var seen = [];
            var result = s[method]({size:1000, has:function(k) {
                seen.push(k);
                if (k === 0) for (var n=1; n<96; n++) s.delete(n);
                if (k === 96) { s.clear(); s.add(200); }
                return answer;
            }, keys:function() { throw Error('wrong branch'); }});
            return seen.join(',') + ':' + (result instanceof Set ? Array.from(result).join(',') : result);
        }
        [exercise('intersection', true), exercise('isSubsetOf', true),
         exercise('isDisjointFrom', false)].join('|')
    "#,
        "0,96,200:0,96,200|0,96,200:true|0,96,200:true",
    );
}

#[test]
fn ordered_collection_difference_index_preserves_snapshot_and_order() {
    check(
        r#"
        var s = new Set([0,1,2,3,4,5]), events = [];
        var other = {size:2, has:function() { throw Error('wrong branch'); }, keys:function() {
            events.push('keys'); s.clear(); s.add(100);
            var values = [2, 2, -0, 4], n = 0;
            return {next:function() { events.push(n); return n < values.length ?
                {value:values[n++], done:false} : {done:true}; }};
        }};
        var result = s.difference(other);
        var small = new Set([1,2,3]);
        var snapshot = small.difference({size:100, has:function(k) {
            small.clear(); small.add(99); return k === 2;
        }, keys:function() { throw Error('wrong branch'); }});
        [Array.from(result).join(','), Array.from(snapshot).join(','), events.join(',')].join('|')
    "#,
        "1,3,5|1,3|keys,0,1,2,3,4",
    );
}

#[test]
fn ordered_collection_set_like_iterator_close_normal_completion() {
    check(
        r#"
        var out = [];
        for (var method of ['isSupersetOf','isDisjointFrom']) {
            for (var mode of ['get','call','primitive','noncallable','null']) {
                var s = new Set([1,2]);
                var other = {size:1, has:function(){}, keys:function() {
                    var iterator = {next:function() { return {value:method === 'isSupersetOf' ? 3 : 1}; }};
                    Object.defineProperty(iterator, 'return', {get:function() {
                        if (mode === 'get') throw 'getter';
                        if (mode === 'null') return null;
                        if (mode === 'noncallable') return 1;
                        return function() { if (mode === 'call') throw 'call'; return 1; };
                    }}); return iterator;
                }};
                try { out.push(s[method](other)); }
                catch(e) { out.push(e instanceof TypeError ? 'TypeError' : e); }
            }
        }
        var touched = false;
        Object.defineProperty(Number.prototype, 'next', {get:function(){ touched=true; throw 42; }, configurable:true});
        try { new Set().union({size:0,has:function(){},keys:function(){return 1;}}); }
        catch(e) { out.push(e instanceof TypeError, touched); }
        out.join('|')
    "#,
        "getter|call|TypeError|TypeError|false|getter|call|TypeError|TypeError|false|true|false",
    );
}

#[test]
fn ordered_collection_algebra_uses_intrinsic_set_prototype() {
    check(
        r#"
        var intrinsic = Set.prototype, methods = ['union','intersection','difference','symmetricDifference'];
        class Derived extends Set {}
        Object.defineProperty(Derived, Symbol.species, {get:function(){throw Error('species');}});
        function Unrelated() {
            var s = new Derived([1,2]);
            return methods.every(function(method) {
                return Object.getPrototypeOf(s[method](new Set([2]))) === intrinsic;
            });
        }
        var called = false;
        function Construct() { called = Unrelated(); }
        new Construct(); called
    "#,
        "true",
    );
}

#[test]
fn ordered_collection_iterator_gc_edges_and_exhaustion_release() {
    let mut engine = Engine::new();
    engine
        .eval(
            "var collection = new Map([[{}, {}]]); var iterator = collection.entries();",
            false,
        )
        .unwrap();
    let global = Value::Obj(engine.interp.global.clone());
    let value = engine
        .interp
        .member_get(&global, "collection")
        .unwrap_or_else(|_| panic!("collection lookup"));
    let weak = engine.interp.downgrade_object_value(&value).unwrap();
    drop(value);
    engine.eval("collection = null", false).unwrap();
    engine.interp.gc_collect();
    assert!(weak.upgrade().is_some(), "live iterator retains collection");
    engine
        .eval("iterator.next(); iterator.next();", false)
        .unwrap();
    engine.interp.gc_collect();
    assert!(
        weak.upgrade().is_none(),
        "completed iterator releases collection"
    );
    assert_eq!(engine.interp.collection_iterators.len(), 1);
    engine.eval("iterator = null", false).unwrap();
    engine.interp.gc_collect();
    assert!(engine.interp.collection_iterators.is_empty());
}

#[test]
fn ordered_collection_iterator_cycle_is_collectable() {
    let mut engine = Engine::new();
    engine.eval("var collection = new Map(); var iterator = collection.values(); collection.set(iterator, iterator);", false).unwrap();
    let global = Value::Obj(engine.interp.global.clone());
    let value = engine
        .interp
        .member_get(&global, "collection")
        .unwrap_or_else(|_| panic!("collection lookup"));
    let weak = engine.interp.downgrade_object_value(&value).unwrap();
    drop(value);
    engine.eval("collection = iterator = null", false).unwrap();
    engine.interp.gc_collect();
    assert!(weak.upgrade().is_none());
    assert!(engine.interp.map_data.is_empty());
    assert!(engine.interp.collection_iterators.is_empty());
}
