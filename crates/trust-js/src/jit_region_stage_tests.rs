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
