//! Tier 1's step (a): a function's first native compile has no acyclic register region; its
//! second stage, the inline recompile of a function that has run, has its spans.
use crate::{bytecode::Tier, Completion, Engine};

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("stage fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// Acyclic regions entered so far on this thread (each one's entry records guarded event 0).
fn acyclic_entries() -> u64 {
    super::guarded_regions()[0]
}

#[test]
fn first_stages_leave_acyclic_spans_to_second_stages() {
    // `pick` is one acyclic span from pc 0 in both stages (its first stage's 25 operations, its
    // second stage's 35 with `leaf` spliced); `run` calls it from a loop.
    let setup = r#"
        function leaf(x) { return x * 3; }
        function pick(o, k) {
            var t = o.a * 2 + o.b;
            if (o.id === k) { t = t + leaf(o.a); } else { t = t - o.b * 4; }
            return t;
        }
        var objs = [];
        for (var i = 0; i < 8; i++) objs.push({ id: i & 1, a: i, b: i + 1 });
        function run(n) { var s = 0; for (var r = 0; r < n; r++) s += pick(objs[r & 7], r & 1); return s; }
        "ready"
    "#;
    let mut reference = Engine::new();
    reference.set_tier(Tier::Interp);
    evaluate(&mut reference, setup);
    let few = evaluate(&mut reference, "run(20)");
    let many = evaluate(&mut reference, "run(400)");

    let mut jit = Engine::new();
    jit.set_tier(Tier::Jit);
    jit.set_tier_threshold(0);
    evaluate(&mut jit, setup);
    let before = acyclic_entries();
    // Twenty calls run `pick`'s first stage, short of its first inline checkpoint.
    assert_eq!(evaluate(&mut jit, "run(20)"), few);
    if std::env::var_os("TRUST_JS_JIT_FIRST_STAGE_SPANS").is_none() {
        assert_eq!(
            acyclic_entries(),
            before,
            "a first stage runs no acyclic span"
        );
    }
    // Past the checkpoint, its second stage runs, span included.
    assert_eq!(evaluate(&mut jit, "run(400)"), many);
    assert!(
        acyclic_entries() > before,
        "the second stage runs pick's acyclic span"
    );
}

/// Register regions entered so far on this thread that were loops (every region's entry records
/// one execution, an acyclic one's also guarded event 0).
fn loop_entries() -> u64 {
    super::executed_regions() - super::guarded_regions()[0]
}

#[test]
fn second_stages_leave_out_regions_whose_code_has_not_run() {
    // `g`'s loop runs only for a large `x`: none had come when its second stage was compiled
    // (`leaf` spliced, at the first inline checkpoint), so that stage has no region for the loop,
    // whose `o.b` read never filled its cache. Its first stage has the loop region (tier 1 keeps
    // first stages' loops), entered once a large `x` comes before the recompile.
    let setup = r#"
        function leaf(x) { return x * 3; }
        function g(o, x) {
            var s = o.a;
            if (x > 1000) { for (var i = 0; i < 4; i++) { s = s + o.b * i; } }
            return s + leaf(o.c);
        }
        var o = { a: 1, b: 2, c: 3 };
        function run(n, x) { var t = 0; for (var r = 0; r < n; r++) t += g(o, x); return t; }
        "ready"
    "#;
    let mut reference = Engine::new();
    reference.set_tier(Tier::Interp);
    evaluate(&mut reference, setup);
    let small = evaluate(&mut reference, "run(300, 1)");
    let large = evaluate(&mut reference, "run(20, 5000)");

    let mut jit = Engine::new();
    jit.set_tier(Tier::Jit);
    jit.set_tier_threshold(0);
    evaluate(&mut jit, setup);
    // Past the first inline checkpoint with a small `x`: `g`'s second stage is compiled.
    assert_eq!(evaluate(&mut jit, "run(300, 1)"), small);
    let before = loop_entries();
    assert_eq!(evaluate(&mut jit, "run(20, 5000)"), large);
    let entered = loop_entries() - before;
    // `run`'s own loop is entered once a call: anything beyond it would be `g`'s loop.
    if std::env::var_os("TRUST_JS_JIT_UNRUN_REGIONS").is_none() {
        assert!(entered <= 1, "g's loop ran as a region {entered} times");
    }
}
