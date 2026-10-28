//! Host interruption reaches ShadowRealm code (Decision 19), as V8's `TerminateExecution`
//! reaches a ShadowRealm, which runs in its creator's isolate: a ShadowRealm's interpreter
//! answers to its creator's interrupt handle, also across `Engine::set_interrupt_handle`, and an
//! interruption passes `ShadowRealm.prototype.evaluate` and wrapped functions' calls in both
//! directions, where the proposal copies a throw completion into a TypeError of the caller's
//! realm (PerformShadowRealmEval, OrdinaryWrappedFunctionCall; proposal 9ff2a01f). HTML's "abort
//! a running script" is no completion: execution ceases without author `catch` or `finally`.

use crate::bytecode::Tier;
use crate::{Completion, Engine, ExecutionOutcome, InterruptReason, RuntimeInterrupt};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

fn value(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("threw {name}: {message}"),
    }
}

/// Run `source`, which only an interruption ends, until a cancellation from another thread or a
/// deadline stops it, and return the reason the evaluation reports.
fn interrupted(engine: &mut Engine, source: &str, deadline: bool) -> InterruptReason {
    let interrupt = engine.interrupt_handle();
    let canceller = (!deadline).then(|| {
        let interrupt = interrupt.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            interrupt.cancel();
        })
    });
    if deadline {
        interrupt.set_deadline(Some(Instant::now() + Duration::from_millis(20)));
    }
    let started = Instant::now();
    let outcome = engine.eval_interruptible(source, false).expect("parses");
    if let Some(canceller) = canceller {
        canceller.join().expect("canceller");
    }
    assert!(started.elapsed() < Duration::from_secs(20), "{source}");
    match outcome {
        ExecutionOutcome::Interrupted { reason } => reason,
        ExecutionOutcome::Value(value) => panic!("{source} returned {value}"),
        ExecutionOutcome::Throw { name, message } => {
            panic!("{source} threw {name}: {message}")
        }
    }
}

const SETUP: &str = r#"
    var realm = new ShadowRealm(), caught = 0;
    var spin = realm.evaluate('(function () { for (;;) {} })');
    var call = realm.evaluate('(function (callback) { return callback(); })');
    function hostSpin() { for (;;) {} }
"#;

/// A cancellation and a deadline each stop a loop inside a ShadowRealm (`evaluate`'s own code,
/// a wrapped function's) and a loop in the host realm called back from one, on every tier, and
/// the caller sees the interruption: no TypeError, and no author `catch` runs.
#[test]
fn host_interruptions_stop_shadow_realm_code_and_pass_its_boundary() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for (case, source) in [
            (
                "evaluate",
                "try { realm.evaluate('for (;;) {}'); } catch (e) { caught = 1; }",
            ),
            ("wrapped", "try { spin(); } catch (e) { caught = 1; }"),
            (
                "host callback",
                "try { call(hostSpin); } catch (e) { caught = 1; }",
            ),
        ] {
            for deadline in [false, true] {
                let mut engine = engine(tier);
                value(&mut engine, SETUP);
                let expected = if deadline {
                    InterruptReason::DeadlineExceeded
                } else {
                    InterruptReason::Cancelled
                };
                assert_eq!(
                    interrupted(&mut engine, source, deadline),
                    expected,
                    "{tier:?}, {case}"
                );
                if deadline {
                    engine.interrupt_handle().set_deadline(None);
                    assert_eq!(
                        value(&mut engine, "caught"),
                        "0",
                        "{tier:?}, {case}: an author catch ran"
                    );
                }
            }
        }
    }
}

/// Ordinary throws still cross every boundary as TypeErrors of the caller's realm.
#[test]
fn ordinary_throws_still_cross_shadow_realm_boundaries_as_type_errors() {
    let mut engine = engine(Tier::Jit);
    value(&mut engine, SETUP);
    let source = r#"
        var kinds = [];
        try { realm.evaluate('throw new RangeError("inside")'); }
        catch (e) { kinds.push(e instanceof TypeError); }
        var thrower = realm.evaluate('(function () { throw 1; })');
        try { thrower(); } catch (e) { kinds.push(e instanceof TypeError); }
        try { call(function () { throw new Error("host"); }); }
        catch (e) { kinds.push(e instanceof TypeError); }
        kinds.join()
    "#;
    assert_eq!(value(&mut engine, source), "true,true,true");
}

/// A handle swap moves the ShadowRealm's interpreter with the root's: the old handle reaches
/// neither, the new one stops ShadowRealm code, and a ShadowRealm made later shares it too.
#[test]
fn a_handle_swap_moves_shadow_realm_interpreters_too() {
    let mut engine = engine(Tier::Jit);
    value(&mut engine, SETUP);
    let old = engine.interrupt_handle();
    assert_eq!(
        old.registered_words(),
        2,
        "the root's and the ShadowRealm's"
    );
    let new = Arc::new(RuntimeInterrupt::default());
    engine.set_interrupt_handle(new.clone());
    assert_eq!(old.registered_words(), 0);
    assert_eq!(new.registered_words(), 2);
    old.cancel();
    assert_eq!(value(&mut engine, "realm.evaluate('1 + 1')"), "2");
    assert_eq!(
        interrupted(&mut engine, "realm.evaluate('for (;;) {}')", true),
        InterruptReason::DeadlineExceeded
    );
    new.set_deadline(None);
    value(&mut engine, "var later = new ShadowRealm();");
    assert_eq!(new.registered_words(), 3);
}
