//! A stale-chain request of a first stage compiled in place (P3 stage 3, step 4p, rule 2a;
//! `opt::chain_request_in_place`), end to end: under graph inlining it makes the tier's warm-up
//! checkpoint due and no bytecode recompile; at the splice the recompile without property chains
//! is made as before.

use super::{address, compiled, discards, eval, exits, is_tier_code};
use crate::bytecode::{Tier, STALE_CHAIN_REPLAYS};

/// `local`'s template code bakes A's shape into a property chain (its first native code after
/// the VM's runs, a warm engine's); B's receivers discard its tier code at its WrongShape check,
/// and then miss the kept template code's chain, whose replays reach the request.
const PROGRAM: &str = "function A(x) { this.x = x; }\n\
                       function B(x) { this.y = 0; this.x = x; }\n\
                       function local(o) { if (o.x < 50) return 1; return 2; }\n\
                       var a = new A(1), b = new B(2);\n\
                       var t = 0; for (var i = 0; i < 300; i++) t += local(a); t";

fn engine() -> crate::Engine {
    let mut engine = crate::Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(1);
    engine.interp.jit_admit_runs = 20;
    engine
}

/// local's function runtime's chunks: its first stage and its second slot.
fn chunks(engine: &mut crate::Engine) -> (std::rc::Rc<crate::bytecode::Chunk>, bool) {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var("local", &env).ok().expect("local");
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

#[test]
fn a_stale_chain_request_of_an_in_place_function_at_graph_makes_its_warm_up_due() {
    for graph in [true, false] {
        let mode = if graph { "graph" } else { "splice" };
        crate::opt::TestSwitches {
            inline_graph: Some(graph),
            ..super::eager()
        }
        .set();
        let mut engine = engine();
        assert_eq!(eval(&mut engine, PROGRAM), "300", "{mode}: warm");
        let (first, second) = chunks(&mut engine);
        assert!(
            super::super::in_place(&first) && !second,
            "{mode}: in place"
        );
        assert!(
            first.property_chains.get(),
            "{mode}: its template code baked a chain"
        );
        let optimized = first.jit.get().flatten().expect("resident");
        assert!(is_tier_code(&optimized), "{mode}: the tier's code");
        compiled();
        exits();
        super::super::TEST_CHAIN_REQUESTS.with(|requests| requests.borrow_mut().clear());
        // B's receivers: 100 exits discard the code; the kept template code's chain then bails
        // at each call, and its replays ask for a recompile without chains.
        let rounds = 100 + STALE_CHAIN_REPLAYS as usize + 20;
        assert_eq!(
            eval(
                &mut engine,
                &format!("var s = 0; for (var i = 0; i < {rounds}; i++) s += local(b); s")
            ),
            rounds.to_string(),
            "{mode}: results"
        );
        assert_eq!(discards(&first), 1, "{mode}: one discard");
        let (first_now, second) = chunks(&mut engine);
        assert!(std::rc::Rc::ptr_eq(&first, &first_now));
        let requests = super::super::TEST_CHAIN_REQUESTS.with(|requests| requests.take());
        if graph {
            assert!(
                first.stale_chain_replays() >= STALE_CHAIN_REPLAYS,
                "{mode}: the request was made: {} replays",
                first.stale_chain_replays()
            );
            assert_eq!(
                requests,
                vec![(address(&first), Some(1))],
                "{mode}: the warm-up's first checkpoint made due"
            );
            assert!(!second, "{mode}: no recompile of the bytecode");
            assert!(
                compiled().contains(&address(&first)),
                "{mode}: the tier compiled the first stage again"
            );
            let now = first.jit.get().flatten().expect("resident");
            assert!(is_tier_code(&now), "{mode}: the tier's code in place again");
            assert!(
                !super::super::reoptimizing(&first),
                "{mode}: the warm-up ended"
            );
        } else {
            assert!(
                requests.is_empty(),
                "{mode}: the splice takes no such request"
            );
            assert!(
                second,
                "{mode}: the recompile without property chains, as before"
            );
        }
        crate::opt::release_retired(&mut engine.interp);
    }
}
