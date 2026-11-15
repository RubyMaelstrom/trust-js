//! A reoptimization re-reads its call sites' ways (P3 stage 3, step 4p, rule 5;
//! `opt::reread_call_ways`), end to end, at the splice and at graph: a site whose guard did not
//! exit but which met another callee since the code was compiled is built as a test with its
//! mismatch edge, as a flagged guard is; a site that met no other callee keeps its check.

use super::{discards, eval, exits, jit_engine, tier_chunk};
use crate::bytecode::Tier;
use crate::opt::ir::ExitReason;

/// h calls four callees: f at site A, g at B, k at C (one callee each when warm), and o's method
/// at D, a named method call of two callees when warm (the splice rule admits several callees at
/// a named method call alone). Each callee runs once first, so that it has native code when h's
/// sites meet it.
const SOURCE: &str = "function fa(x) { return x + 1; }\n\
                      function fb(x) { return x + 2; }\n\
                      function ga(x) { return x * 2; }\n\
                      function gb(x) { return x * 3; }\n\
                      function ka(x) { return x - 1; }\n\
                      function da(x) { return x + 10; }\n\
                      function db(x) { return x + 20; }\n\
                      function dc(x) { return x + 30; }\n\
                      function h(f, g, k, o, x) { return f(x) + g(x) + k(x) + o.m(x); }\n\
                      var oa = {m: da}, ob = {m: db}, oc = {m: dc};\n\
                      fa(0); fb(0); ga(0); gb(0); ka(0); da(0); db(0); dc(0);";
/// D's first guard, a check while unflagged, fails for its second callee every other call: those
/// exits discard the code, whose reoptimization builds D's first guard as a test.
const WARM: &str =
    "var t = 0; for (var i = 0; i < 3000; i++) t += h(fa, ga, ka, i % 2 ? oa : ob, i); t";
/// A's callee changes: its guard fails first in each call, the rest of which the VM runs; 200
/// exits at R = 1 discard the code again.
const EXIT: &str = "var u = 0; for (var i = 0; i < 200; i++) u += h(fb, gb, ka, oa, i); u";
/// The warm-up's template code (its first checkpoint 400 invocations away at R = 2) meets gb at B
/// and dc at D, which no guard holds; the reoptimization re-reads them.
const WARM_UP: &str = "var v = 0; for (var i = 0; i < 450; i++) v += h(fb, gb, ka, oc, i); v";
const AFTER: &str = "var w = 0; for (var i = 0; i < 50; i++) w += h(fb, gb, ka, oc, i); w";
/// C meets a callee its guard does not hold.
const C_CHANGES: &str = "h(fb, gb, fa, oc, 7)";

#[test]
fn a_reoptimization_turns_the_guards_of_sites_that_met_other_callees_into_tests() {
    let expected: Vec<String> = {
        let mut vm = crate::Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        [SOURCE, WARM, EXIT, WARM_UP, AFTER, C_CHANGES]
            .iter()
            .map(|source| eval(&mut vm, source))
            .collect()
    };
    for graph in [false, true] {
        let mode = if graph { "graph" } else { "splice" };
        crate::opt::TestSwitches {
            inline_graph: Some(graph),
            ..super::eager()
        }
        .set();
        let mut engine = jit_engine();
        assert_eq!(eval(&mut engine, SOURCE), expected[0]);
        assert_eq!(eval(&mut engine, WARM), expected[1], "{mode}: warm");
        let chunk = tier_chunk(&mut engine, "h").expect("h's tier chunk");
        assert_eq!(super::super::in_place(&chunk), graph, "{mode}: placement");
        assert_eq!(discards(&chunk), 1, "{mode}: D's discard while warm");
        assert!(
            !super::super::reoptimizing(&chunk),
            "{mode}: reoptimized while warm"
        );
        exits();
        super::super::TEST_REREAD.with(|reread| reread.borrow_mut().clear());
        assert_eq!(eval(&mut engine, EXIT), expected[2], "{mode}: exits");
        let taken = exits();
        assert_eq!(
            taken.len(),
            2 * super::super::EXIT_LIMIT as usize,
            "{mode}: A's guard until the discard at R = 1: {taken:?}"
        );
        assert!(
            taken.iter().all(|&(_, why)| why == ExitReason::WrongCallee),
            "{mode}: {taken:?}"
        );
        assert_eq!(discards(&chunk), 2, "{mode}: R");
        assert_eq!(eval(&mut engine, WARM_UP), expected[3], "{mode}: warm-up");
        assert!(!super::super::reoptimizing(&chunk), "{mode}: reoptimized");
        let reread: Vec<usize> = super::super::TEST_REREAD.with(|reread| {
            reread
                .borrow()
                .iter()
                .filter(|&&(at, _)| at == std::rc::Rc::as_ptr(&chunk) as usize)
                .map(|&(_, pc)| pc)
                .collect()
        });
        // B's guard and D's second (its first flagged by D's exits) at the splice; B's call and
        // D's, its guard byte raised from its first way to both, at graph.
        assert_eq!(reread.len(), 2, "{mode}: the re-read's flags: {reread:?}");
        assert_eq!(eval(&mut engine, AFTER), expected[4], "{mode}: after");
        assert!(
            exits().is_empty(),
            "{mode}: B and D take their mismatch edges, no exit"
        );
        assert_eq!(discards(&chunk), 2, "{mode}: no third discard");
        assert_eq!(
            eval(&mut engine, C_CHANGES),
            expected[5],
            "{mode}: C changes"
        );
        let taken = exits();
        assert_eq!(
            taken.iter().map(|&(_, why)| why).collect::<Vec<_>>(),
            vec![ExitReason::WrongCallee],
            "{mode}: C met no other callee and kept its check, which exits once"
        );
        crate::opt::release_retired(&mut engine.interp);
    }
}
