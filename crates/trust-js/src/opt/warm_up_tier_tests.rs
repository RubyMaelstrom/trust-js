//! A discard's warm-up and the call sites' feedback (P3 stage 3, step 4p), end to end: the
//! warm-up's settled test reads a call site's state and not its ways (`snapshot::
//! warm_up_fingerprint`, as V8 waits after a deoptimization for no call feedback), while a
//! function's first compiles still wait for its call sites' ways (stage 2, step 11).

use super::{address, compiled, discarded, discards, eval, exits, is_tier_code, jit_engine};
use super::{tier_chunk, waits_of};
use crate::opt::TestSwitches;

/// g calls whichever of f1 .. f4 it is given: its call site's ways are the callees it met. Each
/// callee runs once first, so that it has native code when g's site meets it (a site fills a way
/// for a callee with native code).
const SOURCE: &str = "function f1(x) { return x + 1; }\n\
                      function f2(x) { return x + 2; }\n\
                      function f3(x) { return x + 3; }\n\
                      function f4(x) { return x + 4; }\n\
                      function g(f, x) { return f(x) * 2 + x; }\n\
                      f1(0); f2(0); f3(0); f4(0);";

/// The tier on, its settled-feedback rule waiting (not eager), nothing forced; g compiled on its
/// first stage's checkpoints from calls with `warm` (callees chosen by k), then discarded by 100
/// calls whose string argument fails a Number check once each. The chunk whose warm-up is now
/// armed; the waits recorded so far cleared.
fn discarded_g(warm: &str) -> (crate::Engine, std::rc::Rc<crate::bytecode::Chunk>) {
    TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    let mut engine = jit_engine();
    eval(&mut engine, SOURCE);
    compiled();
    exits();
    discarded();
    eval(
        &mut engine,
        &format!("var t = 0; for (var k = 0; k < 3000; k++) t += g({warm}, k); t"),
    );
    let chunk = tier_chunk(&mut engine, "g").expect("g's tier chunk");
    let optimized = chunk.jit.get().flatten().expect("resident");
    assert!(is_tier_code(&optimized), "g runs the tier's code");
    assert!(exits().is_empty(), "no exit while warm");
    eval(
        &mut engine,
        "var u = ''; for (var k = 0; k < 100; k++) u = g(f1, 'x'); u",
    );
    // f1's own code, compiled from g's calls, exits on the strings too.
    assert!(
        discarded().contains(&(optimized.mem_ptr() as usize)),
        "the 100th exit discards g's code"
    );
    assert_eq!(discards(&chunk), 1, "R");
    assert!(crate::opt::reoptimizing(&chunk), "the warm-up is armed");
    super::super::TEST_WAITS.with(|waits| waits.borrow_mut().clear());
    compiled();
    drop(optimized);
    (engine, chunk)
}

#[test]
fn a_warm_ups_checkpoint_does_not_wait_for_a_call_site_whose_ways_grew() {
    // g's call site met f1 and f2 before the discard; during the warm-up it meets f3 and f4 as
    // well: polymorphic before and after, its ways changed. The warm-up's first checkpoint, 200
    // invocations after the discard at R = 1, finds the feedback settled and recompiles.
    let (mut engine, chunk) = discarded_g("k % 2 ? f1 : f2");
    let site_state = |chunk: &crate::bytecode::Chunk| {
        let snapshot = super::super::snapshot::Snapshot::take(chunk);
        (0..snapshot.len())
            .find_map(|pc| match snapshot.site(pc) {
                crate::feedback::SiteView::Call(call) => Some(call.state),
                _ => None,
            })
            .expect("g's call site")
    };
    assert_eq!(site_state(&chunk), crate::feedback::CallState::Polymorphic);
    let before = super::super::snapshot::fingerprint(&chunk);
    let before_warm = super::super::snapshot::warm_up_fingerprint(&chunk);
    eval(&mut engine, "g(f3, 1); g(f4, 1);");
    assert_eq!(
        site_state(&chunk),
        crate::feedback::CallState::Polymorphic,
        "still polymorphic"
    );
    assert_ne!(
        super::super::snapshot::fingerprint(&chunk),
        before,
        "a first compile's fingerprint sees the new ways"
    );
    assert_eq!(
        super::super::snapshot::warm_up_fingerprint(&chunk),
        before_warm,
        "the warm-up's does not"
    );
    eval(
        &mut engine,
        "var t = 0; for (var k = 0; k < 300; k++) t += g([f1, f2, f3, f4][k % 4], k); t",
    );
    assert_eq!(
        waits_of(&chunk),
        Vec::<u8>::new(),
        "no warm-up checkpoint waited"
    );
    assert!(
        compiled().contains(&address(&chunk)),
        "the recompile at the first checkpoint"
    );
    assert!(!crate::opt::reoptimizing(&chunk), "the warm-up ended");
    let now = chunk.jit.get().flatten().expect("resident");
    assert!(is_tier_code(&now), "the recompile is resident");
    assert_eq!(
        eval(&mut engine, "g(f3, 2) + g(f1, 'y')"),
        "12NaNy",
        "results as the VM's"
    );
    drop(now);
    crate::opt::release_retired(&mut engine.interp);
}

#[test]
fn a_warm_ups_checkpoint_waits_for_a_call_sites_transition() {
    // g's call site met f1 alone before the discard; during the warm-up it meets f2: monomorphic
    // to polymorphic, a transition, which the warm-up's first checkpoint waits for; its second,
    // the feedback unchanged since, recompiles.
    let (mut engine, chunk) = discarded_g("f1");
    eval(&mut engine, "g(f2, 1);");
    eval(
        &mut engine,
        "var t = 0; for (var k = 0; k < 300; k++) t += g(k % 2 ? f1 : f2, k); t",
    );
    assert_eq!(waits_of(&chunk), vec![1], "the first checkpoint waited");
    assert!(crate::opt::reoptimizing(&chunk), "the warm-up goes on");
    assert!(compiled().is_empty(), "no recompile yet");
    // The second checkpoint, 400 invocations x K << 1 native runs after the first.
    eval(
        &mut engine,
        "var t = 0; for (var k = 0; k < 900; k++) t += g(k % 2 ? f1 : f2, k); t",
    );
    assert_eq!(waits_of(&chunk), vec![1], "the second did not wait");
    assert!(
        compiled().contains(&address(&chunk)),
        "the recompile at the second checkpoint"
    );
    assert!(!crate::opt::reoptimizing(&chunk), "the warm-up ended");
    crate::opt::release_retired(&mut engine.interp);
}

#[test]
fn a_first_compile_still_waits_for_a_call_site_whose_ways_grew() {
    // Before any compile, g's first stage meets f1 and f2 by its first checkpoint (100
    // invocations), f3 and f4 between it and its second (500): the second waits, as step 11's
    // fingerprint keeps call ways for first compiles; the third (2,100), unchanged, compiles.
    TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    let mut engine = jit_engine();
    eval(&mut engine, SOURCE);
    compiled();
    eval(
        &mut engine,
        "var t = 0; for (var k = 0; k < 150; k++) t += g(k % 2 ? f1 : f2, k); t",
    );
    let first = super::first_stage(&mut engine, "g");
    super::super::TEST_WAITS.with(|waits| waits.borrow_mut().clear());
    eval(&mut engine, "g(f3, 1); g(f4, 1);");
    eval(
        &mut engine,
        "var t = 0; for (var k = 0; k < 400; k++) t += g(k % 2 ? f1 : f2, k); t",
    );
    assert!(
        waits_of(&first).contains(&2),
        "the second checkpoint waited: {:?}",
        waits_of(&first)
    );
    assert!(
        tier_chunk(&mut engine, "g").is_none(),
        "no tier code before the third checkpoint"
    );
    eval(
        &mut engine,
        "var t = 0; for (var k = 0; k < 2000; k++) t += g(k % 2 ? f1 : f2, k); t",
    );
    let chunk = tier_chunk(&mut engine, "g").expect("compiled at the third checkpoint");
    assert!(compiled().contains(&address(&chunk)));
    crate::opt::release_retired(&mut engine.interp);
}
