//! P1 S2: payload cells (BigInt digits, Symbols, strings) under the gc-heap collector.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn live_cells(engine: &Engine) -> usize {
    engine.interp.gc_heap.core.borrow().stats().live_cells
}

/// BigInts keep their digits across collections wherever they are held: array elements, Map
/// keys, wrapper objects, typed arrays, closure bindings, literals in compiled code.
#[test]
fn bigints_survive_collections_in_every_holder() {
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let result = eval(
            &mut engine,
            r#"
            var big = 123456789012345678901234567890n;
            function make(i) { return BigInt(i) ** 20n + big; }
            var kept = [], wrapped = new Map(), typed = new BigInt64Array(8);
            var closures = [];
            for (let i = 0; i < 200; i++) {
                let value = make(i);
                kept.push(value);
                wrapped.set(value, Object(value));
                typed[i % 8] = BigInt.asIntN(64, value);
                closures.push(() => value * 2n);
                if (i % 40 === 0) $262.gc();
            }
            $262.gc();
            [
                kept.every((value, i) => value === make(i)),
                [...wrapped].every(([key, box]) => typeof box === 'object' && box.valueOf() === key),
                typed[3] === BigInt.asIntN(64, make(195)),
                closures.every((f, i) => f() === make(i) * 2n),
                0x1234_5678_9abc_def0_1234n.toString(16),
            ].join()
            "#,
        );
        assert_eq!(
            result, "true,true,true,true,123456789abcdef01234",
            "{tier:?}"
        );
    }
}

/// BigInts are cells, and the ones nothing holds are reclaimed by the next collection.
#[test]
fn dead_bigints_are_reclaimed() {
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, "true");
        engine.interp.gc_collect();
        let before = live_cells(&engine);
        eval(
            &mut engine,
            "let sum = 0n; for (let i = 0; i < 4000; i++) sum += BigInt(i) * 3n; sum === 23994000n",
        );
        // Three cells per iteration, unless GC stress collected them along the way.
        let allocated = live_cells(&engine);
        assert!(
            crate::gc_test::stress_active() || allocated > before + 10_000,
            "{tier:?}: {before} live cells before, {allocated} after the loop"
        );
        engine.interp.gc_collect();
        let after = live_cells(&engine);
        assert!(
            after < before + 1_000,
            "{tier:?}: {before} live cells before, {after} after collecting"
        );
    }
}

/// A BigInt kept only by a WeakMap entry with a live key, or as a FinalizationRegistry cell's
/// held value, survives collections (ephemeron values and side-table values are edges).
#[test]
fn bigints_held_by_weak_collections_survive() {
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            var key = {}, weak = new WeakMap(), held = [];
            var registry = new FinalizationRegistry(value => held.push(value));
            (function () {
                weak.set(key, 2n ** 100n + 7n);
                registry.register({}, 3n ** 50n);
            })();
            true
            "#,
        );
        engine.interp.gc_collect();
        eval(&mut engine, "true");
        engine.interp.gc_collect();
        let result = eval(
            &mut engine,
            "[weak.get(key) === 2n ** 100n + 7n, held.length, held[0] === 3n ** 50n].join()",
        );
        assert_eq!(result, "true,1,true", "{tier:?}");
    }
}

/// Symbols keep their identity across collections wherever they are held: values, property
/// keys (named only by id in the key text), wrapper objects, Map keys, WeakMap keys reachable
/// elsewhere, closures, the Symbol.for registry and the well-known Symbols.
#[test]
fn symbols_survive_collections_in_every_holder() {
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let result = eval(
            &mut engine,
            r#"
            var symbols = [], keyed = {}, wrappers = [], map = new Map(), weak = new WeakMap();
            var closures = [];
            for (let i = 0; i < 100; i++) {
                let symbol = Symbol('s' + i);
                symbols.push(symbol);
                keyed[Symbol('key' + i)] = i;
                wrappers.push(Object(symbol));
                map.set(symbol, i);
                weak.set(symbol, { i });
                closures.push(() => symbol);
                if (i % 25 === 0) $262.gc();
            }
            (function () { Symbol.for('registered-only'); })();
            $262.gc();
            const keys = Object.getOwnPropertySymbols(keyed);
            [
                symbols.every((s, i) => s.description === 's' + i),
                keys.length === 100 && keys.every((k, i) => k.description === 'key' + i && keyed[k] === i),
                wrappers.every((w, i) => w.valueOf() === symbols[i]),
                symbols.every((s, i) => map.get(s) === i && weak.get(s).i === i),
                closures.every((f, i) => f() === symbols[i]),
                Symbol.keyFor(Symbol.for('registered-only')),
                typeof [][Symbol.iterator],
            ].join()
            "#,
        );
        assert_eq!(
            result, "true,true,true,true,true,registered-only,function",
            "{tier:?}"
        );
    }
}

/// Symbols nothing holds are reclaimed; a WeakRef to one clears, a WeakMap entry keyed by one
/// goes (its value with it), and a FinalizationRegistry cell targeting one is cleaned up.
#[test]
fn dead_symbols_are_reclaimed_and_their_weak_holders_cleared() {
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, "true");
        engine.interp.gc_collect();
        let before = live_cells(&engine);
        eval(
            &mut engine,
            "for (let i = 0; i < 4000; i++) Symbol('transient'); true",
        );
        // The first collection finds the Symbols dead; their cells stay reserved while the
        // Agent's identity index still names them weakly, and the index drops those entries
        // right after it. The next collection frees the cells.
        engine.interp.gc_collect();
        engine.interp.gc_collect();
        let after = live_cells(&engine);
        assert!(
            after < before + 1_000,
            "{tier:?}: {before} live cells before, {after} after collecting"
        );
        eval(
            &mut engine,
            r#"
            var cleaned = [];
            var registry = new FinalizationRegistry(held => cleaned.push(held));
            var weakMap = new WeakMap(), ref, valueRef;
            (function () {
                const symbol = Symbol('weakly held');
                const value = {};
                ref = new WeakRef(symbol);
                valueRef = new WeakRef(value);
                weakMap.set(symbol, value);
                registry.register(symbol, 'held');
            })();
            true
            "#,
        );
        engine.interp.gc_collect();
        eval(&mut engine, "true");
        let result = eval(
            &mut engine,
            "[ref.deref(), valueRef.deref(), cleaned.join()].join()",
        );
        assert_eq!(result, ",,held", "{tier:?}");
    }
}

/// Strings keep their contents across collections wherever they are held: values, Map keys
/// and values, String wrappers and iterators, closures, error stacks (frame names), RegExp
/// legacy statics, literals in compiled code, function names, private names and the heap's
/// interned one-unit and index strings.
#[test]
fn strings_survive_collections_in_every_holder() {
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let result = eval(
            &mut engine,
            r#"
            function named() { throw new Error('boom'); }
            var error;
            try { named(); } catch (e) { error = e; }
            var words = [], map = new Map(), wrapped = [], iterators = [], closures = [];
            for (let i = 0; i < 100; i++) {
                const word = 'w' + i + '-' + 'x'.repeat(i % 7);
                words.push(word);
                map.set(word, word.toUpperCase());
                wrapped.push(new String(word));
                const iterator = word[Symbol.iterator]();
                iterator.next();
                iterators.push(iterator);
                closures.push(() => word);
                if (i % 25 === 0) $262.gc();
            }
            /(\w+)-(\w*)/.exec('match-' + 'me');
            class Box { #secret = 'hid' + 'den'; get() { return this.#secret; } }
            const box = new Box();
            $262.gc();
            [
                words.every((w, i) => w === 'w' + i + '-' + 'x'.repeat(i % 7)),
                [...map].every(([key, value]) => value === key.toUpperCase()),
                wrapped.every((w, i) => w.valueOf() === words[i] && w.length === words[i].length),
                iterators.every((it, i) => it.next().value === words[i][1]),
                closures.every((f, i) => f() === words[i]),
                error.stack.includes('named'),
                RegExp.$1 + RegExp.$2,
                box.get(),
                named.name,
                'a literal string',
                'abc'.charAt(1) + [10, 20][1] + String(5),
            ].join()
            "#,
        );
        assert_eq!(
            result, "true,true,true,true,true,true,matchme,hidden,named,a literal string,b205",
            "{tier:?}"
        );
    }
}

/// Frontier append: accumulation extends a buffer in place, yet every alias keeps its own
/// contents, and an alias's own later append copies rather than overwriting.
#[test]
fn string_accumulation_extends_in_place_and_keeps_aliases() {
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let result = eval(
            &mut engine,
            r#"
            var s = '', aliases = [], o = { s: '' };
            for (let i = 0; i < 2000; i++) {
                s += 'ab';
                o.s += 'cd';
                if (i % 500 === 0) aliases.push(s, o.s);
                if (i % 700 === 0) $262.gc();
            }
            var t = s;
            s += 'X';
            t += 'Y';
            var u = t;
            u += u;
            [
                aliases.map(a => a.length).join('|'),
                s.length, t.length, s.endsWith('X'), t.endsWith('Y'),
                s.slice(0, -1) === t.slice(0, -1),
                u.length, u === t + t,
                o.s.length, o.s.startsWith('cdcd'),
            ].join()
            "#,
        );
        assert_eq!(
            result,
            "2|2|1002|1002|2002|2002|3002|3002,4001,4001,true,true,true,8002,true,4000,true",
            "{tier:?}"
        );
    }
}

/// Strings nothing holds are reclaimed by the next collection.
#[test]
fn dead_strings_are_reclaimed() {
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, "true");
        engine.interp.gc_collect();
        let before = live_cells(&engine);
        eval(
            &mut engine,
            "let n = 0; for (let i = 0; i < 4000; i++) n += ('s' + i).length; n > 0",
        );
        let allocated = live_cells(&engine);
        assert!(
            crate::gc_test::stress_active() || allocated > before + 3_000,
            "{tier:?}: {before} live cells before, {allocated} after the loop"
        );
        engine.interp.gc_collect();
        let after = live_cells(&engine);
        assert!(
            after < before + 1_000,
            "{tier:?}: {before} live cells before, {after} after collecting"
        );
    }
}

/// The heap's interned one-unit and index strings stay valid while nothing references them:
/// a later lookup returns the same interned cell.
#[test]
fn interned_strings_outlive_their_references() {
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            'xyz'.charAt(1);
            var seen = [];
            new Proxy([], { get(target, key) { seen.push(key); } })[7];
            seen = null;
            true
            "#,
        );
        engine.interp.gc_collect();
        eval(&mut engine, "true");
        engine.interp.gc_collect();
        let result = eval(
            &mut engine,
            r#"
            var keys = [];
            new Proxy([], { get(target, key) { keys.push(key); } })[7];
            ['xyz'.charAt(1), keys[0], typeof keys[0]].join()
            "#,
        );
        assert_eq!(result, "y,7,string", "{tier:?}");
    }
}
