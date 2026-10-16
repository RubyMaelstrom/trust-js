//! P1 S2: payload cells (BigInt digits) under the gc-heap collector.

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
