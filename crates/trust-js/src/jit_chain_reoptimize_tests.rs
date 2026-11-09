//! Operator chains whose baked property states go stale (P2's C1, after the tier-up design
//! note): a chain's bail replays its remaining operations through the generic helper, which
//! counts a replayed property operation whose site's cache holds the receiver's shape
//! (`Chunk::note_chain_replay`). Past `STALE_CHAIN_REPLAYS` the function recompiles once
//! without property chains, which later calls run: into `FunctionRuntime::code3` after a
//! second stage, else into `code2`. A chain whose state holds never asks. A warm engine runs
//! loop-free functions on the bytecode VM first, so their first native code bakes the shapes
//! those runs saw, as a second stage's seeded caches do.

use crate::bytecode::{Chunk, Tier, STALE_CHAIN_REPLAYS};
use crate::{Completion, Engine};

/// The bytecode VM runs a warm engine's loop-free functions make before native code.
const WARM_RUNS: u32 = 20;

fn engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(1);
    engine.interp.jit_admit_runs = WARM_RUNS;
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// The global function `name`'s chunks: its first-stage chunk, its second stage (an inline
/// recompile, or the recompile without property chains of a function without one) and its
/// recompile without property chains after a second stage.
fn with_chunks<R>(
    engine: &mut Engine,
    name: &str,
    f: impl FnOnce(&Chunk, Option<&Chunk>, Option<&Chunk>) -> R,
) -> R {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let object_handle = value.as_obj().expect("a function object");
    let object = object_handle.borrow();
    let crate::value::ObjectKind::User(user) = &object.kind else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    let first = runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap_or_else(|| panic!("{name} has no chunk"));
    let second = runtime.code2.get().and_then(Option::as_ref);
    let third = runtime.code3.get().and_then(Option::as_ref);
    f(
        first,
        second.map(|chunk| &**chunk),
        third.map(|chunk| &**chunk),
    )
}

/// A recompile without property chains: native, with none, replaying nothing.
fn assert_recompiled(name: &str, chunk: Option<&Chunk>) {
    let chunk = chunk.unwrap_or_else(|| panic!("{name} recompiled"));
    assert!(
        chunk.no_property_chains.get(),
        "{name}'s recompile builds no chains"
    );
    assert!(
        chunk.jit.get().flatten().is_some(),
        "{name}'s recompile runs natively"
    );
    assert!(
        !chunk.property_chains.get(),
        "{name}'s recompile has no property chain"
    );
    assert_eq!(
        chunk.stale_chain_replays(),
        0,
        "{name}'s recompile replays nothing"
    );
}

/// Chains build a property operation only where they compute with it (`jit::build_chain`): an
/// ordered comparison, a bitwise operator, a store of a bitwise result. `staged` calls a helper
/// its inline checkpoint splices, so it reaches its second stage on the first class.
const PROGRAM: &str = r#"
function A(x) { this.x = x; this.n = x; }
function B(x) { this.y = 0; this.x = x; this.n = x; }
function local(o) { if (o.x < 50) return 1; return 2; }
function self() { return this.x & 7; }
function store(o, v) { o.n = v & 255; return o.n; }
function held(o) { return o.x & 3; }
function one() { return 1; }
function staged(o) { var k = one(); if (o.x < 50) return k; return 0; }
var a = new A(1), b = new B(2), t = 0;
for (var i = 0; i < 30; i++) t += local(a) + self.call(a) + store(a, i) + held(a);
for (var i = 0; i < 300; i++) t += staged(a);
t;
"#;

#[test]
fn a_stale_chain_recompiles_once_without_property_chains() {
    // Property chains are the template JIT's code; the tier's second stages build none.
    crate::opt::pin_tier_off_for_test();
    let mut engine = engine();
    eval(&mut engine, PROGRAM);
    for name in ["local", "self", "store", "held"] {
        with_chunks(&mut engine, name, |first, second, third| {
            assert!(first.jit.get().flatten().is_some(), "{name} is native");
            assert!(first.property_chains.get(), "{name} baked a property chain");
            assert!(
                second.is_none() && third.is_none(),
                "{name} has not recompiled"
            );
        });
    }
    with_chunks(&mut engine, "staged", |_, second, third| {
        let second = second.expect("staged reached its second stage");
        assert!(second.jit_second_stage() && second.property_chains.get());
        assert!(third.is_none(), "staged has not recompiled");
    });
    // A second receiver class: its reads and stores miss every chain's baked state.
    let rounds = STALE_CHAIN_REPLAYS as usize + 100;
    let total = eval(
        &mut engine,
        &format!(
            "var s = 0; for (var i = 0; i < {rounds}; i++) s += local(b) + self.call(b) + store(b, i) + staged(b); s"
        ),
    );
    let expected: usize = (0..rounds).map(|i| 1 + 2 + (i & 255) + 1).sum();
    assert_eq!(total, expected.to_string());
    for name in ["local", "self", "store"] {
        with_chunks(&mut engine, name, |first, second, third| {
            assert!(
                first.stale_chain_replays() >= STALE_CHAIN_REPLAYS,
                "{name}: {} replays",
                first.stale_chain_replays()
            );
            // No second stage before it: the recompile takes the second slot.
            assert_recompiled(name, second);
            assert!(third.is_none());
        });
    }
    with_chunks(&mut engine, "staged", |_, second, third| {
        let second = second.unwrap();
        assert!(second.stale_chain_replays() >= STALE_CHAIN_REPLAYS);
        assert!(!second.no_property_chains.get());
        assert_recompiled("staged", third);
    });
    // A chain whose state holds never asks.
    eval(&mut engine, "for (var i = 0; i < 300; i++) t += held(a); t");
    with_chunks(&mut engine, "held", |first, second, third| {
        assert_eq!(first.stale_chain_replays(), 0);
        assert!(!first.reoptimize.get());
        assert!(second.is_none() && third.is_none(), "held keeps its chain");
    });
    // The recompiles keep computing what the chains did, for both classes.
    assert_eq!(
        eval(
            &mut engine,
            "local(a) + ',' + local({ x: 99 }) + ',' + self.call(b) + ',' + store(a, 300) + ',' + staged(a) + staged({ x: 99 })"
        ),
        "1,2,2,44,10"
    );
}
