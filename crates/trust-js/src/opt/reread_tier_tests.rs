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
/// At the splice D's first guard, a check while unflagged, fails for its second callee every other
/// call: those exits discard the code, whose reoptimization builds D's first guard as a test. At
/// graph every planned way is built from the first compile (P3 stage 3 step 3f, piece D): D's first
/// way is a test, its second the check, and nothing exits, so R stays 0.
const WARM: &str =
    "var t = 0; for (var i = 0; i < 3000; i++) t += h(fa, ga, ka, i % 2 ? oa : ob, i); t";
/// A's callee changes: its guard fails first in each call, the rest of which the VM runs; the exit
/// limit's exits at R (200 at R = 1 at the splice, 100 at R = 0 at graph, whose other 100 calls
/// run the template code) discard the code again.
const EXIT: &str = "var u = 0; for (var i = 0; i < 200; i++) u += h(fb, gb, ka, oa, i); u";
/// The warm-up's template code (its first checkpoint 400 invocations from the discard at R = 2 at
/// the splice, 200 at R = 1 at graph) meets gb at B and dc at D, which no guard holds; the
/// reoptimization re-reads them.
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
        // R after the warm-up: D's discard at the splice; none at graph (piece D).
        let r = u8::from(!graph);
        assert_eq!(discards(&chunk), r, "{mode}: D's discard while warm");
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
            (super::super::EXIT_LIMIT as usize) << r,
            "{mode}: A's guard until the discard at R = {r}: {taken:?}"
        );
        assert!(
            taken.iter().all(|&(_, why)| why == ExitReason::WrongCallee),
            "{mode}: {taken:?}"
        );
        assert_eq!(discards(&chunk), r + 1, "{mode}: R");
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
        // D's at graph.
        assert_eq!(reread.len(), 2, "{mode}: the re-read's flags: {reread:?}");
        if graph {
            // The flags' data: at B's call its one way, at D's its guard byte at both ways (no
            // exit flagged D at graph, whose first compile built both). At graph the recompile's
            // last ways are tests either way, since the snapshot's caches name the callees met
            // (piece D's past-plan rule): the flags are this test's subject there.
            let ops = chunk.jit_ops();
            let calls: Vec<usize> = (0..ops.len())
                .filter(|&pc| matches!(ops[pc], crate::bytecode::Op::Call(..)))
                .collect();
            let method = (0..ops.len())
                .find(|&pc| matches!(ops[pc], crate::bytecode::Op::CallWithThis(..)))
                .expect("D's call");
            let (b, d) = (calls[1], method);
            let mut sorted = reread.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, vec![b, d], "graph: B's and D's calls re-read");
            let flag = |pc: usize| {
                let mut ways = None;
                super::super::profile::ChunkProfile::of(&chunk).sites_at(
                    pc as u32,
                    |reason, inlined, n| {
                        if reason == ExitReason::WrongCallee as u8 && !inlined {
                            ways = Some(n);
                        }
                    },
                );
                ways
            };
            assert_eq!(flag(b), Some(1), "graph: B's flag covers its way");
            assert_eq!(flag(d), Some(2), "graph: D's flag covers both ways");
        }
        assert_eq!(eval(&mut engine, AFTER), expected[4], "{mode}: after");
        assert!(
            exits().is_empty(),
            "{mode}: B and D take their mismatch edges, no exit"
        );
        assert_eq!(discards(&chunk), r + 1, "{mode}: no further discard");
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

/// Rule 5 at a site nested in an inlined frame (P3 stage 3 step 3d): `hn` compiles in place at
/// graph with `md` inlined and `md`'s call of `f` (`la` when warm) a frame nested in `md`'s. A
/// string `n` discards `hn`'s code (its add's exits), a string `x` `md`'s own (its multiply's), so
/// that `md` runs its template code too; in `hn`'s warm-up the template code calls `md` with `lb`,
/// which `md`'s call cache meets and no guard holds; the reoptimization re-reads the
/// nested site from that cache and flags `md`'s first stage at its call with the inlined key (the
/// key the nested guard's exit sets and an inlined frame reads), so that the recompile builds the
/// nested guard as a test whose mismatch edge is the generic call: no exit and no discard after.
const NESTED: &str = "function la(x) { return x + 1; }\n\
                      function lb(x) { return x + 2; }\n\
                      function md(f, x) { return f(x) * 2; }\n\
                      function hn(f, n, x) { return md(f, x) + n; }\n\
                      la(0); lb(0);\n\
                      var t = 0; for (var i = 0; i < 3000; i++) t += hn(la, i, i); t";
/// The exit limit's exits at R = 0 (`EXIT_LIMIT`) and no call after the discard, so that the
/// warm-up's template code runs in `NESTED_WARM_UP` alone.
const NESTED_EXIT: &str = "var u = ''; for (var i = 0; i < 100; i++) u = hn(la, 's' + i, i); u";
const NESTED_MD_EXIT: &str = "var m = 0; for (var i = 0; i < 100; i++) m = md(la, 's' + i); m";
const NESTED_WARM_UP: &str =
    "var v = 0; for (var i = 0; i < 3000; i++) v += hn(i % 2 ? la : lb, i, i); v";
const NESTED_AFTER: &str =
    "var w = 0; for (var i = 0; i < 50; i++) w += hn(i % 2 ? la : lb, i, i); w";

#[test]
fn a_reoptimization_re_reads_the_ways_of_a_site_nested_in_an_inlined_frame() {
    let expected: Vec<String> = {
        let mut vm = crate::Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        [
            NESTED,
            NESTED_EXIT,
            NESTED_MD_EXIT,
            NESTED_WARM_UP,
            NESTED_AFTER,
        ]
        .iter()
        .map(|source| eval(&mut vm, source))
        .collect()
    };
    crate::opt::TestSwitches {
        inline_graph: Some(true),
        ..super::eager()
    }
    .set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, NESTED), expected[0], "warm");
    let chunk = tier_chunk(&mut engine, "hn").expect("hn's tier chunk");
    assert!(super::super::in_place(&chunk), "hn compiles in place");
    let md = super::first_stage(&mut engine, "md");
    let md_call = md
        .jit_ops()
        .iter()
        .position(|op| matches!(op, crate::bytecode::Op::Call(..)))
        .expect("md's call");
    let before = discards(&chunk);
    assert_eq!(super::super::EXIT_LIMIT, 100);
    assert_eq!(eval(&mut engine, NESTED_EXIT), expected[1], "exits");
    assert_eq!(discards(&chunk), before + 1, "the add's exits discard");
    let md_before = discards(&md);
    assert_eq!(eval(&mut engine, NESTED_MD_EXIT), expected[2], "md's exits");
    assert_eq!(
        discards(&md),
        md_before + 1,
        "md's multiply's exits discard"
    );
    super::super::TEST_REREAD.with(|reread| reread.borrow_mut().clear());
    exits();
    let warm_before = discards(&chunk);
    assert_eq!(eval(&mut engine, NESTED_WARM_UP), expected[3], "warm-up");
    assert!(!super::super::reoptimizing(&chunk), "reoptimized");
    assert!(super::super::in_place(&chunk), "hn compiles in place again");
    // The recompile read the flag before any nested guard exited: no WrongCallee exit in the
    // warm-up's rest and no discard (a nested guard left a check would exit for every `lb` to
    // the limit and discard again; one flagged with the own key too, which an inlined frame does
    // not read for a kind-bound reason).
    assert!(
        !exits()
            .iter()
            .any(|&(_, why)| why == ExitReason::WrongCallee),
        "no WrongCallee exit after the re-read"
    );
    assert_eq!(discards(&chunk), warm_before, "no discard in the warm-up");
    let reread: Vec<(usize, usize)> =
        super::super::TEST_REREAD.with(|reread| reread.borrow().clone());
    assert!(
        reread.contains(&(std::rc::Rc::as_ptr(&md) as usize, md_call)),
        "md's call, the nested site, re-read: {reread:?}"
    );
    let mut inlined_key = false;
    super::super::profile::ChunkProfile::of(&md).sites_at(md_call as u32, |reason, inlined, _| {
        inlined_key |= reason == ExitReason::WrongCallee as u8 && inlined;
    });
    assert!(inlined_key, "the flag's key is the inlined one");
    exits();
    let after = discards(&chunk);
    assert_eq!(eval(&mut engine, NESTED_AFTER), expected[4], "after");
    assert!(
        !exits()
            .iter()
            .any(|&(_, why)| why == ExitReason::WrongCallee),
        "the nested site takes its mismatch edge"
    );
    assert_eq!(discards(&chunk), after, "no discard after");
    crate::opt::release_retired(&mut engine.interp);
}
