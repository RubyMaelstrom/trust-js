//! A stale-chain request of a first stage compiled in place (P3 stage 3, step 4p, rules 2a and
//! 2b; `opt::chain_request_in_place`), end to end: under graph inlining it makes the tier's
//! warm-up checkpoint due and no bytecode recompile, and the chunk gives its property chains up:
//! its resident template code is compiled again without them, its kept fallback and its
//! borrowed-frame code are discarded, so that the next discard and the next loop transfer compile
//! theirs without them, and the request is not made again; at the splice the recompile without
//! property chains is made as before.

use super::super::profile::ChunkProfile;
use super::super::ChainsDropped;
use super::{address, compiled, discards, eval, exits, is_tier_code};
use crate::bytecode::{Chunk, Tier, STALE_CHAIN_REPLAYS};
use std::rc::Rc;

/// `local`'s template code bakes A's shape into a property chain (its first native code after
/// the VM's runs, a warm engine's); B's receivers discard its tier code at its WrongShape check,
/// and then miss the kept template code's chain, whose replays reach the request. `local`'s
/// second argument, a function, is called before the read: an activation that calls back into
/// `local` runs on in its code while the inner calls make the request.
const PROGRAM: &str = "function A(x) { this.x = x; }\n\
                       function B(x) { this.y = 0; this.x = x; }\n\
                       function local(o, f) { if (f) f(); if (o.x < 50) return 1; return 2; }\n\
                       var a = new A(1), b = new B(2);\n\
                       function inner() { var s = 0; for (var k = 0; k < 150; k++) s += local(b, null); return s; }\n\
                       var t = 0; for (var i = 0; i < 300; i++) t += local(a, null); t";

fn engine() -> crate::Engine {
    let mut engine = crate::Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(1);
    engine.interp.jit_admit_runs = 20;
    engine
}

/// Each test runs as it is, with forced exits (every 100th check passed fails, the forced-exit
/// gate's TRUST_JS_OPT_EXIT_EVERY) and under the poison (TRUST_JS_OPT_POISON).
#[derive(Clone, Copy, Debug)]
enum Variant {
    Plain,
    Forced,
    Poison,
}

const VARIANTS: [Variant; 3] = [Variant::Plain, Variant::Forced, Variant::Poison];

fn switches(graph: bool, variant: Variant) {
    crate::opt::TestSwitches {
        inline_graph: Some(graph),
        exit_every: if matches!(variant, Variant::Forced) {
            100
        } else {
            0
        },
        poison: matches!(variant, Variant::Poison),
        ..super::eager()
    }
    .set();
}

/// `test` under each variant.
fn each(test: fn(Variant)) {
    for variant in VARIANTS {
        eprintln!("variant {variant:?}");
        test(variant);
    }
}

/// `name`'s function runtime's chunks: its first stage and whether a second slot is set.
fn chunks(engine: &mut crate::Engine, name: &str) -> (Rc<Chunk>, bool) {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("a global");
    let object = value.as_obj().expect("a function");
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        panic!("a user function")
    };
    let runtime = user.func.runtime();
    let first = runtime
        .code
        .get()
        .and_then(Option::clone)
        .expect("its chunk");
    let second = runtime.code2.get().and_then(Option::as_ref).is_some();
    (first, second)
}

fn base(code: &crate::jit::JitCode) -> usize {
    code.mem_ptr() as usize
}

fn resident(chunk: &Chunk) -> Rc<crate::jit::JitCode> {
    chunk.jit.get().flatten().expect("resident code")
}

fn kept(chunk: &Chunk) -> Option<usize> {
    ChunkProfile::get(chunk)
        .and_then(ChunkProfile::kept_code)
        .map(|code| base(&code))
}

fn retired(engine: &crate::Engine, at: usize) -> bool {
    engine
        .interp
        .opt_retired
        .iter()
        .any(|code| base(code) == at)
}

fn requests() -> Vec<(usize, Option<u8>)> {
    super::super::TEST_CHAIN_REQUESTS.with(|requests| requests.take())
}

fn drops() -> Vec<(usize, ChainsDropped)> {
    super::super::TEST_CHAINS_DROPPED.with(|drops| drops.take())
}

fn epoch() -> u32 {
    crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed)
}

/// The warm program at `graph`: `local` in place, the tier's code resident, its template code
/// kept with A's chain.
fn warm(graph: bool, variant: Variant) -> (crate::Engine, Rc<Chunk>) {
    switches(graph, variant);
    let mut engine = engine();
    assert_eq!(eval(&mut engine, PROGRAM), "300", "warm");
    let (first, second) = chunks(&mut engine, "local");
    assert!(super::super::in_place(&first) && !second, "in place");
    assert!(
        first.property_chains.get(),
        "its template code baked a chain"
    );
    assert!(is_tier_code(&resident(&first)), "the tier's code");
    assert!(kept(&first).is_some(), "its template code kept");
    compiled();
    exits();
    requests();
    drops();
    (engine, first)
}

/// B's receivers: 100 exits discard the tier's code, and then `replays` calls miss the kept
/// template code's chain.
fn discard_then_miss(engine: &mut crate::Engine, replays: usize) {
    let rounds = 100 + replays;
    assert_eq!(
        eval(
            engine,
            &format!("var s = 0; for (var i = 0; i < {rounds}; i++) s += local(b, null); s")
        ),
        rounds.to_string(),
        "results"
    );
}

/// The chunk builds no chain and counts no replay any more (rule 2b's fourth part).
fn chains_given_up(first: &Chunk) {
    assert!(first.no_property_chains.get(), "it builds no chain");
    assert!(
        !first.property_chains.get(),
        "no code compiled since holds a chain"
    );
    assert_eq!(first.stale_chain_replays(), 0, "its replays' count reset");
}

#[test]
fn a_stale_chain_request_of_an_in_place_function_at_graph_makes_its_warm_up_due() {
    each(a_stale_chain_request_of_an_in_place_function_at_graph_makes_its_warm_up_due_in);
}

fn a_stale_chain_request_of_an_in_place_function_at_graph_makes_its_warm_up_due_in(
    variant: Variant,
) {
    for graph in [true, false] {
        let mode = if graph { "graph" } else { "splice" };
        let (mut engine, first) = warm(graph, variant);
        // The template code the discard puts back: the kept one, with A's chain.
        let chained = kept(&first).expect("kept");
        discard_then_miss(&mut engine, STALE_CHAIN_REPLAYS as usize + 20);
        assert_eq!(discards(&first), 1, "{mode}: one discard");
        let (first_now, second) = chunks(&mut engine, "local");
        assert!(Rc::ptr_eq(&first, &first_now));
        let requests = requests();
        let drops = drops();
        if graph {
            assert_eq!(
                requests,
                vec![(address(&first), Some(1))],
                "{mode}: the warm-up's first checkpoint made due"
            );
            assert!(!second, "{mode}: no recompile of the bytecode");
            // Rule 2b: the resident template code compiled again without chains.
            let [(key, dropped)] = drops[..] else {
                panic!("{mode}: one drop: {drops:?}")
            };
            assert_eq!(key, address(&first));
            let (old, new) = dropped.resident.expect("the resident template code");
            assert_eq!(old, chained, "{mode}: the discard's template code replaced");
            assert!(new != 0 && new != old, "{mode}: compiled again");
            assert_eq!(dropped.kept, None, "{mode}: the discard took the kept code");
            assert_eq!(
                dropped.borrowed, None,
                "{mode}: no loop, no borrowed-frame code"
            );
            chains_given_up(&first);
            assert!(
                compiled().contains(&address(&first)),
                "{mode}: the tier compiled the first stage again"
            );
            assert!(
                is_tier_code(&resident(&first)),
                "{mode}: the tier's code in place again"
            );
            assert_eq!(
                kept(&first),
                Some(new),
                "{mode}: its install kept the template code without chains"
            );
            assert!(
                !super::super::reoptimizing(&first),
                "{mode}: the warm-up ended"
            );
        } else {
            assert!(
                requests.is_empty() && drops.is_empty(),
                "{mode}: the splice takes no such request"
            );
            assert!(
                first.stale_chain_replays() >= STALE_CHAIN_REPLAYS,
                "{mode}: the request was made: {} replays",
                first.stale_chain_replays()
            );
            assert!(
                second,
                "{mode}: the recompile without property chains, as before"
            );
        }
        crate::opt::release_retired(&mut engine.interp);
    }
}

/// Rule 2b's first part, by itself: in the discard's warm-up the request compiles the slot's
/// template code again without chains, moves the chunk's entry word and the call caches' epoch,
/// and retires the old code; later calls run the new code, whose misses count nothing, until the
/// checkpoint the request made due installs the tier's code over it.
#[test]
fn the_requests_template_code_is_compiled_again_without_chains_and_its_entry_word_moves() {
    each(the_requests_template_code_is_compiled_again_without_chains_and_its_entry_word_moves_in);
}

fn the_requests_template_code_is_compiled_again_without_chains_and_its_entry_word_moves_in(
    variant: Variant,
) {
    let (mut engine, first) = warm(true, variant);
    discard_then_miss(&mut engine, 10);
    assert_eq!(discards(&first), 1);
    assert!(super::super::reoptimizing(&first), "the warm-up armed");
    let old = resident(&first);
    assert!(!is_tier_code(&old), "the discard's template code");
    let old = base(&old);
    let word = first.entry_word();
    let moved = epoch();
    super::super::chain_request_in_place(&mut engine.interp, &first);
    let now = resident(&first);
    assert!(!is_tier_code(&now), "template code");
    assert_ne!(base(&now), old, "compiled again");
    assert_eq!(
        first.entry_word(),
        word.wrapping_add(1),
        "its entry word moved"
    );
    assert!(epoch() != moved, "the call caches' epoch moved");
    assert!(retired(&engine, old), "the old code retired");
    let [(_, dropped)] = drops()[..] else {
        panic!("one drop")
    };
    assert_eq!(dropped.resident, Some((old, base(&now))));
    chains_given_up(&first);
    assert_eq!(requests().len(), 1, "the request itself");
    // Later calls: the new code, whose misses count nothing; the due checkpoint compiles.
    assert_eq!(
        eval(
            &mut engine,
            "var s = 0; for (var i = 0; i < 200; i++) s += local(b, null); s"
        ),
        "200"
    );
    chains_given_up(&first);
    assert!(requests().is_empty(), "not made again");
    assert!(is_tier_code(&resident(&first)), "the tier's code again");
    crate::opt::release_retired(&mut engine.interp);
    assert!(!retired(&engine, old), "released once nothing runs it");
}

/// An activation that runs the old template code when the request comes finishes in it: `local`
/// (b, inner) runs the discard's template code, and the 150 calls `inner` makes reach the request
/// at its 101st replay; the outer activation then reads `o.x` in the old code, whose bail counts
/// nothing, and the old code stays among the retired codes, which a release at the request's
/// checkpoint kept because the activation ran it, until nothing runs it.
#[test]
fn an_activation_in_the_old_template_code_finishes_in_it_and_its_code_retires_after() {
    each(an_activation_in_the_old_template_code_finishes_in_it_and_its_code_retires_after_in);
}

fn an_activation_in_the_old_template_code_finishes_in_it_and_its_code_retires_after_in(
    variant: Variant,
) {
    let (mut engine, first) = warm(true, variant);
    discard_then_miss(&mut engine, 50);
    let old = base(&resident(&first));
    assert!(!is_tier_code(&resident(&first)));
    assert_eq!(eval(&mut engine, "local(b, inner)"), "1");
    let requests = requests();
    assert_eq!(requests.len(), 1, "the request: {requests:?}");
    let [(_, dropped)] = drops()[..] else {
        panic!("one drop")
    };
    assert_eq!(dropped.resident.map(|(from, _)| from), Some(old));
    chains_given_up(&first);
    assert!(
        retired(&engine, old),
        "kept while the outer activation ran it"
    );
    crate::opt::release_retired(&mut engine.interp);
    assert!(!retired(&engine, old), "released once nothing runs it");
    assert_eq!(
        eval(
            &mut engine,
            "var s = 0; for (var i = 0; i < 200; i++) s += local(b, null) + local(a, null); s"
        ),
        "400"
    );
    assert!(super::super::TEST_CHAIN_REQUESTS.with(|r| r.borrow().is_empty()));
}

/// Rule 2b's second part: a request that comes while the tier's code is resident (no warm-up
/// armed; here made as `Chunk::note_chain_replay` makes it) leaves that code and discards the
/// kept fallback, so that the next discard compiles the template code without chains.
#[test]
fn a_request_over_the_tiers_code_discards_the_kept_fallback_and_the_next_discard_compiles_one_without_chains(
) {
    each(a_request_over_the_tiers_code_discards_the_kept_fallback_and_the_next_discard_compiles_one_without_chains_in);
}

fn a_request_over_the_tiers_code_discards_the_kept_fallback_and_the_next_discard_compiles_one_without_chains_in(
    variant: Variant,
) {
    let (mut engine, first) = warm(true, variant);
    let tier = resident(&first);
    let chained = kept(&first).expect("kept");
    let moved = epoch();
    first.reoptimize.set(true);
    first.set_inline_retry_at(first.jit_runs.get().saturating_add(1).max(1));
    assert_eq!(eval(&mut engine, "local(a, null)"), "1");
    assert_eq!(
        requests(),
        vec![(address(&first), None)],
        "no warm-up armed"
    );
    let [(_, dropped)] = drops()[..] else {
        panic!("one drop")
    };
    assert_eq!(
        dropped,
        ChainsDropped {
            resident: None,
            kept: Some(chained),
            borrowed: None
        }
    );
    assert!(
        Rc::ptr_eq(&resident(&first), &tier),
        "the tier's code stays"
    );
    assert_eq!(kept(&first), None, "the kept fallback discarded");
    assert!(retired(&engine, chained));
    assert!(epoch() != moved, "the call caches' epoch moved");
    chains_given_up(&first);
    // The next discard compiles its fallback, which builds no chain; its misses count nothing.
    discard_then_miss(&mut engine, STALE_CHAIN_REPLAYS as usize + 20);
    assert_eq!(discards(&first), 1);
    let fallback = base(&resident(&first));
    assert_ne!(fallback, chained, "a fallback compiled at the discard");
    chains_given_up(&first);
    assert!(requests().is_empty(), "not made again");
    crate::opt::release_retired(&mut engine.interp);
}

/// Rule 2b's third part: the borrowed-frame code a loop transfer of an exited activation compiled
/// is discarded at the request, and the next exited activation's loop transfer compiles it again,
/// without chains.
#[test]
fn a_request_discards_the_borrowed_frame_code_and_the_next_loop_transfer_compiles_one_without_chains(
) {
    each(a_request_discards_the_borrowed_frame_code_and_the_next_loop_transfer_compiles_one_without_chains_in);
}

fn a_request_discards_the_borrowed_frame_code_and_the_next_loop_transfer_compiles_one_without_chains_in(
    variant: Variant,
) {
    switches(true, variant);
    let mut engine = engine();
    eval(
        &mut engine,
        "function lp(a, o) { var t = 0; if (o.k < 50) t = 1; for (var i = 0; i < a.length; i++) t = t + a[i]; return t; }\n\
         var o = { k: 1 }, a = [1, 2, 3, 4]; for (var i = 0; i < 400; i++) lp(a, o);",
    );
    let (lp, _) = chunks(&mut engine, "lp");
    assert!(super::super::in_place(&lp), "in place");
    assert!(is_tier_code(&resident(&lp)));
    // A string element: the add exits in the loop, and the rebuilt frame runs the rest of the
    // loop in the VM, which transfers to its borrowed-frame code at the back edge. Its first
    // borrowed-frame code came with its first native code, before any run filled `o.k`'s cache:
    // the slot is emptied first, so that the transfer compiles one that bakes `o`'s shape.
    let state = lp.osr.get().expect("its first loop transferred");
    let _ = state.discard();
    assert!(!lp.property_chains.get());
    let strings = "var t = ''; for (var i = 0; i < 3; i++) t = lp([1, 'x', 2, 3, 4, 5], o); t";
    assert_eq!(eval(&mut engine, strings), "2x2345");
    let borrowed = state.resident().expect("borrowed-frame code");
    assert!(
        lp.property_chains.get(),
        "its borrowed-frame code baked a chain"
    );
    requests();
    drops();
    lp.reoptimize.set(true);
    lp.set_inline_retry_at(lp.jit_runs.get().saturating_add(1).max(1));
    assert_eq!(eval(&mut engine, "lp(a, o)"), "11");
    let [(_, dropped)] = drops()[..] else {
        panic!("one drop")
    };
    assert_eq!(
        dropped.borrowed,
        Some(Rc::as_ptr(&borrowed) as usize),
        "{dropped:?}"
    );
    assert!(state.resident().is_none(), "the slot cold");
    chains_given_up(&lp);
    // The next exited activation's loop transfer compiles it again, without chains.
    assert_eq!(eval(&mut engine, strings), "2x2345");
    let again = state.resident().expect("borrowed-frame code again");
    assert!(!Rc::ptr_eq(&again, &borrowed), "compiled again");
    chains_given_up(&lp);
    crate::opt::release_retired(&mut engine.interp);
}

/// An exited activation that runs the borrowed-frame code when the request comes finishes in it:
/// `lp`'s outer call exits at its string element, its VM run transfers to the borrowed-frame code
/// at the back edge, and its loop calls back into `lp`, whose entry makes the request; the slot is
/// emptied under the running activation, which holds its own lease, and the code is released once
/// that activation leaves it.
#[test]
fn an_exited_activation_in_the_borrowed_frame_code_finishes_in_it_and_the_code_is_released_after() {
    each(an_exited_activation_in_the_borrowed_frame_code_finishes_in_it_and_the_code_is_released_after_in);
}

fn an_exited_activation_in_the_borrowed_frame_code_finishes_in_it_and_the_code_is_released_after_in(
    variant: Variant,
) {
    switches(true, variant);
    let mut engine = engine();
    eval(
        &mut engine,
        "function lp(a, o, f) { var t = 0; if (o.k < 50) t = 1; \
         for (var i = 0; i < a.length; i++) { t = t + a[i]; if (f && i == 2) t = t + f(); } return t; }\n\
         function inner() { return lp([1, 2], o, null); }\n\
         var o = { k: 1 }, a = [1, 2, 3, 4]; for (var i = 0; i < 400; i++) lp(a, o, null);",
    );
    let (lp, _) = chunks(&mut engine, "lp");
    assert!(super::super::in_place(&lp), "in place");
    assert!(is_tier_code(&resident(&lp)));
    // As above: the borrowed-frame code compiled once the template code filled `o.k`'s cache.
    let state = lp.osr.get().expect("its first loop transferred");
    let _ = state.discard();
    assert_eq!(eval(&mut engine, "lp([1, 'x', 2, 3], o, null)"), "2x23");
    let borrowed = Rc::downgrade(&state.resident().expect("borrowed-frame code"));
    assert!(lp.property_chains.get(), "it baked a chain");
    requests();
    drops();
    // The request at the inner call's entry: the outer call's entry comes first.
    lp.reoptimize.set(true);
    lp.set_inline_retry_at(lp.jit_runs.get().saturating_add(2));
    // The outer call exits at 'x', runs the rest of its loop in the borrowed-frame code, and at
    // i == 2 calls inner, whose `lp` entry makes the request; it then finishes in that code.
    assert_eq!(eval(&mut engine, "lp([1, 'x', 2, 3], o, inner)"), "2x243");
    let [(_, dropped)] = drops()[..] else {
        panic!("one drop")
    };
    assert_eq!(
        dropped.borrowed,
        Some(borrowed.as_ptr() as usize),
        "the request came while the outer call ran it: {dropped:?}"
    );
    assert!(state
        .resident()
        .is_none_or(|code| !std::ptr::eq(Rc::as_ptr(&code), borrowed.as_ptr())));
    chains_given_up(&lp);
    assert!(
        borrowed.upgrade().is_none(),
        "released once its activation left it"
    );
    crate::opt::release_retired(&mut engine.interp);
}
