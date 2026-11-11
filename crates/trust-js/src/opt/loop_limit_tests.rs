//! An exited frame that loops (P3 stage 2, step 13; JSC's form with the template JIT as the
//! baseline): from the loop limit, 5 << R exits, each exit of the tier's code that does not
//! discard it arms the loop limit's warm-up on its chunk's loop word, which only template code's
//! loop iterations count toward; a loop whose count reaches the word asks the next native entry,
//! whose check discards the code as at its exit limit. The rule runs on either chunk the tier's
//! code goes in (a second stage, or a first stage that carries it, step 14, whose discard returns
//! to the kept template code); entries, loops shorter than the warm-up and exits that no loop
//! follows discard nothing sooner; the words hold their states through the arming, the crossing
//! (after which loops count nothing) and the check.

use super::ir::ExitReason;
use super::profile::{self, ChunkProfile};
use super::TestSwitches;
use crate::bytecode::{Chunk, Tier};
use crate::jit::JitCode;
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

/// An engine at the JIT tier that compiles every function at its first call.
fn jit_engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

/// The tier on and eager, compiling on the main thread, nothing forced.
fn eager() -> TestSwitches {
    TestSwitches {
        on: true,
        eager: true,
        ..Default::default()
    }
}

fn compiled() -> Vec<usize> {
    super::TEST_COMPILED.with(|compiled| std::mem::take(&mut *compiled.borrow_mut()))
}

fn exits() -> Vec<(u32, ExitReason)> {
    super::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()))
}

/// The codes discards retired since the last call.
fn discarded() -> Vec<usize> {
    super::TEST_DISCARDED.with(|discarded| std::mem::take(&mut *discarded.borrow_mut()))
}

/// The limits the discards since the last call reached, with the exits each came after.
fn limits() -> Vec<(super::Limit, u32)> {
    super::TEST_LIMITS.with(|limits| std::mem::take(&mut *limits.borrow_mut()))
}

/// The entries into borrowed-frame code (the loop transfers) since the last call.
fn osr_entries() -> usize {
    crate::jit::TEST_OSR_ENTRIES.with(|count| count.replace(0))
}

fn address(chunk: &Rc<Chunk>) -> usize {
    Rc::as_ptr(chunk) as usize
}

/// Whether `code` is the tier's (it has an exit table), not the template JIT's.
fn is_tier_code(code: &JitCode) -> bool {
    super::exit::table_bytes(code.mem_ptr()) != 0
}

/// R: the discards of the function whose tier chunk `chunk` is.
fn discards(chunk: &Chunk) -> u8 {
    ChunkProfile::get(profile::first_stage(chunk)).map_or(0, ChunkProfile::discards)
}

/// The code in `chunk`'s slot.
fn resident(chunk: &Chunk) -> Rc<JitCode> {
    chunk.jit.get().flatten().expect("resident code")
}

/// The template JIT's code an install of the tier's kept for `chunk` (step 14).
fn kept(chunk: &Chunk) -> Option<Rc<JitCode>> {
    ChunkProfile::get(chunk).and_then(ChunkProfile::kept_code)
}

/// Where the tier's code goes (P3 stage 2, step 14).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    /// A second stage: the function's plan splices a callee.
    Spliced,
    /// The first stage itself, over its template code, which is kept: the plan splices nothing.
    InPlace,
}

impl Placement {
    /// `f` adds 1 to `a | 0` first (an int32 add), which overflows when `a` is 2^31 - 1, then
    /// loops `n` times: its tier code exits early in such a call (Overflow), and the exited frame
    /// runs the loop in the VM to its first back edge and then in the chunk's borrowed-frame
    /// template code. Spliced, `f` calls `cb` at its loop's first iteration, which its plan
    /// splices (`noop`); in place it calls nothing. The warm-up compiles `f` with the tier.
    fn source(self) -> &'static str {
        match self {
            Placement::Spliced => {
                "function noop() {}\n\
                 function f(a, n, cb) { let s = (a | 0) + 1; \
                 for (let i = 0; i < n; i++) { if (i === 0) cb(); s = s + (i & 3); } return s; }\n\
                 let t = 0; for (let k = 0; k < 1000; k++) t = (t + f(k, 4, noop)) | 0; t"
            }
            Placement::InPlace => {
                "function noop() {}\n\
                 function f(a, n, cb) { let s = (a | 0) + 1; \
                 for (let i = 0; i < n; i++) s = s + (i & 3); return s; }\n\
                 let t = 0; for (let k = 0; k < 1000; k++) t = (t + f(k, 4, noop)) | 0; t"
            }
        }
    }
}

/// The chunk the tier compiled for the global function `name`, and where it is: its second
/// stage, else its first stage once the tier's code was installed in it.
fn tier_chunk(engine: &mut Engine, name: &str) -> Option<(Rc<Chunk>, Placement)> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok()?;
    let object = value.as_obj()?;
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        return None;
    };
    let runtime = user.func.runtime();
    if let Some(second) = runtime.code2.get().and_then(Option::clone) {
        return Some((second, Placement::Spliced));
    }
    runtime
        .code
        .get()
        .and_then(Option::clone)
        .filter(|first| super::in_place(first))
        .map(|first| (first, Placement::InPlace))
}

/// The first-stage chunk of the global function `name`.
fn first_stage(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().unwrap();
    let object = value.as_obj().unwrap();
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        panic!("an author function")
    };
    user.func
        .runtime()
        .code
        .get()
        .and_then(Option::clone)
        .unwrap()
}

/// The VM's result of each source after `placement`'s program.
fn in_the_vm(placement: Placement, sources: &[&str]) -> Vec<String> {
    let mut vm = Engine::new();
    vm.set_tier(Tier::Bytecode);
    vm.set_tier_threshold(0);
    eval(&mut vm, placement.source());
    sources.iter().map(|source| eval(&mut vm, source)).collect()
}

/// An engine that ran `placement`'s program under `switches`, `f`'s tier chunk, and its tier
/// code, with the tests' records cleared. The chunk's words idle once the tier's code is
/// installed (a second stage's settled at its making, a first stage's checkpoints ended at the
/// install), so that an exit can arm the loop limit.
fn exit_then_loop(
    switches: TestSwitches,
    placement: Placement,
) -> (Engine, Rc<Chunk>, Rc<JitCode>) {
    switches.set();
    let mut engine = jit_engine();
    compiled();
    eval(&mut engine, placement.source());
    let (chunk, at) = tier_chunk(&mut engine, "f").expect("f's tier chunk");
    assert_eq!(at, placement, "where the tier's code went");
    assert!(
        compiled().contains(&address(&chunk)),
        "the tier compiled it"
    );
    let code = resident(&chunk);
    assert!(is_tier_code(&code));
    assert_eq!(
        (chunk.inline_retry_at.get(), chunk.loop_word()),
        (0, 0),
        "{placement:?}: the words idle"
    );
    assert_eq!(
        kept(&chunk).is_some(),
        placement == Placement::InPlace,
        "the template code kept in place"
    );
    exits();
    discarded();
    limits();
    osr_entries();
    (engine, chunk, code)
}

/// Whether the tier's loop limit is armed on `chunk` (its warm-up on the loop word, the
/// checkpoint word idle), and whether a loop's crossing asked for the next entry's check (the
/// checkpoint word holds the request, the loop word is cleared).
fn loop_limit_state(chunk: &Chunk) -> (bool, bool) {
    let armed = ChunkProfile::get(chunk).is_some_and(ChunkProfile::loop_armed);
    let asked = armed && chunk.inline_retry_at.get() != 0;
    if asked {
        assert_eq!(chunk.loop_word(), 0, "the crossing cleared the loop word");
    } else if armed {
        assert_ne!(chunk.loop_word(), 0, "the warm-up on the loop word");
    }
    (armed, asked)
}

/// Call `source` until `chunk`'s code `code` is discarded, at most `most` times, each call
/// checked against the VM's `expected`: the calls it took, and whether each call's exited frame
/// moved into borrowed-frame code (an OSR entry).
fn calls_until_discarded(
    engine: &mut Engine,
    source: &str,
    expected: &str,
    code: &Rc<JitCode>,
    most: usize,
) -> (Option<usize>, Vec<bool>) {
    let mut transfers = Vec::new();
    for call in 1..=most {
        assert_eq!(eval(engine, source), expected, "call {call}");
        transfers.push(osr_entries() != 0);
        let gone = discarded();
        if !gone.is_empty() {
            assert_eq!(gone, vec![code.mem_ptr() as usize], "call {call}");
            return (Some(call), transfers);
        }
    }
    (None, transfers)
}

/// `count` calls of `source`, each checked against the VM's `expected`, with nothing discarded.
fn calls_discarding_nothing(engine: &mut Engine, source: &str, expected: &str, count: usize) {
    for call in 1..=count {
        assert_eq!(eval(engine, source), expected, "call {call}");
        assert!(discarded().is_empty(), "call {call}");
    }
}

/// The one loop's weight in `chunk`, max(1, round(K x B / L)) (see `jit::loop_count_tests`).
fn the_loop_weight(chunk: &Chunk) -> u32 {
    use crate::bytecode::Op;
    let ops = chunk.jit_ops();
    let spans: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| match *op {
            Op::Jump(target)
            | Op::AbruptJump(target, _)
            | Op::JumpIfFalse(target)
            | Op::JumpIfFalsePeek(target)
            | Op::JumpIfTruePeek(target)
            | Op::JumpIfNotNullishPeek(target)
                if target as usize <= pc =>
            {
                Some(pc - target as usize + 1)
            }
            _ => None,
        })
        .collect();
    assert_eq!(spans.len(), 1, "one back edge");
    let (k, len) = (u64::from(super::ENTRY_WEIGHT), ops.len() as u64);
    ((k * spans[0] as u64 + len / 2) / len).max(1) as u32
}

const LONG: &str = "f(2147483647, 2000, noop)";

/// Without the exit profile the recompile overflows again (step 3's backoff test), so each code
/// exits at its start in every call with a = 2^31 - 1, and the call's frame loops 2,000 times.
fn long_loops_discard_at_the_loop_limit(placement: Placement) {
    let expected = in_the_vm(placement, &[LONG]);
    let (mut engine, chunk, code) = exit_then_loop(
        TestSwitches {
            no_exit_profile: true,
            ..eager()
        },
        placement,
    );
    let template = kept(&chunk);
    // Exits 1-4 are below the loop limit and arm nothing; every frame moves into the
    // borrowed-frame code at its first back edge, its loop counting nothing (the words idle).
    calls_discarding_nothing(&mut engine, LONG, &expected[0], 4);
    assert_eq!(
        loop_limit_state(&chunk),
        (false, false),
        "{placement:?}: below the loop limit"
    );
    assert_eq!(osr_entries(), 4, "each frame's loop moved");
    // The fifth exit arms the warm-up, which its frame's loop crosses: the checkpoint word takes
    // the request, the loop word is cleared, the count stops one below the request.
    let runs = chunk.jit_runs.get();
    let warm_up = super::loop_warm_up(&chunk);
    calls_discarding_nothing(&mut engine, LONG, &expected[0], 1);
    assert_eq!(osr_entries(), 1);
    assert_eq!(
        loop_limit_state(&chunk),
        (true, true),
        "{placement:?}: the crossing asked"
    );
    let request = chunk.inline_retry_at.get();
    assert!(
        request == runs + warm_up || request == runs + warm_up + super::ENTRY_WEIGHT,
        "the warm-up from the exit (the call's entry counted or not): {request} {runs} {warm_up}"
    );
    assert_eq!(chunk.jit_runs.get(), request - 1);
    // The sixth call's entry discards the code at its loop limit, after 5 exits.
    let (at, _) = calls_until_discarded(&mut engine, LONG, &expected[0], &code, 1);
    assert_eq!(at, Some(1), "{placement:?}: discarded at the next entry");
    assert_eq!(limits(), vec![(super::Limit::Loop, 5)]);
    assert_eq!(exits().len(), 6, "{placement:?}");
    assert_eq!(discards(&chunk), 1, "R");
    assert!(super::reoptimizing(&chunk), "the warm-up is armed");
    assert_eq!(loop_limit_state(&chunk), (false, false));
    // The warm-up's first checkpoint on the checkpoint word, which the sixth call's own loop,
    // run after its exit of the discarded code, crossed: the loop word cleared, the count one
    // below the checkpoint.
    assert_ne!(chunk.inline_retry_at.get(), 0);
    assert_eq!(
        (chunk.loop_word(), chunk.jit_runs.get()),
        (0, chunk.inline_retry_at.get() - 1),
        "the warm-up's checkpoint crossed"
    );
    let fallback = resident(&chunk);
    assert!(!is_tier_code(&fallback));
    if placement == Placement::InPlace {
        assert!(
            Rc::ptr_eq(&fallback, template.as_ref().unwrap()),
            "the kept template code again, not a compile"
        );
    }
    // The warm-up's recompile (eager: at its first checkpoint), whose loop limit is 5 << 1.
    const WARM: &str = "let u = 0; for (let k = 0; k < 1000; k++) u = (u + f(k, 4, noop)) | 0; u";
    assert_eq!(eval(&mut engine, WARM), in_the_vm(placement, &[WARM])[0]);
    let recompiled = resident(&chunk);
    assert!(is_tier_code(&recompiled) && !Rc::ptr_eq(&recompiled, &code));
    assert_eq!(
        compiled(),
        vec![address(&chunk)],
        "{placement:?}: one recompile"
    );
    exits();
    calls_discarding_nothing(&mut engine, LONG, &expected[0], 9);
    assert_eq!(loop_limit_state(&chunk), (false, false), "below 5 << 1");
    calls_discarding_nothing(&mut engine, LONG, &expected[0], 1);
    assert_eq!(loop_limit_state(&chunk), (true, true));
    let (at, _) = calls_until_discarded(&mut engine, LONG, &expected[0], &recompiled, 1);
    assert_eq!(at, Some(1));
    assert_eq!(limits(), vec![(super::Limit::Loop, 10)]);
    assert_eq!(exits().len(), 11);
    assert_eq!(discards(&chunk), 2, "R");
    if placement == Placement::InPlace {
        assert!(Rc::ptr_eq(&resident(&chunk), template.as_ref().unwrap()));
    }
    drop((code, recompiled));
    super::release_retired(&mut engine.interp);
}

#[test]
fn an_exited_frame_that_loops_long_discards_a_second_stages_code_at_5_exits_shifted_by_r() {
    long_loops_discard_at_the_loop_limit(Placement::Spliced);
}

#[test]
fn an_exited_frame_that_loops_long_discards_a_first_stages_code_at_5_exits_shifted_by_r() {
    long_loops_discard_at_the_loop_limit(Placement::InPlace);
}

#[test]
fn after_the_crossing_loops_count_nothing_until_the_next_entry_takes_the_check() {
    // `d` calls itself once at its loop's 100th iteration when `depth` is 1. The outer call is
    // the fourth exit (below the loop limit): its frame moves into the borrowed-frame code at its
    // first back edge, and at its 100th iteration the inner call exits (the fifth), arms the
    // warm-up and crosses it in its own loop; the outer loop's 1,900 iterations after it count
    // nothing, the loop word cleared, until the next entry discards the code.
    const SOURCE: &str = "function d(a, n, depth) { let s = (a | 0) + 1; \
         for (let i = 0; i < n; i++) { if (i === 100 && depth === 1) s = s + d(a, n, 0); \
         s = s + (i & 3); } return s; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = (t + d(k, 4, 0)) | 0; t";
    const ONE: &str = "d(2147483647, 2000, 0)";
    const NESTED: &str = "d(2147483647, 2000, 1)";
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        eval(&mut vm, SOURCE);
        [
            eval(&mut vm, ONE),
            eval(&mut vm, NESTED),
            eval(&mut vm, ONE),
        ]
    };
    TestSwitches {
        no_exit_profile: true,
        ..eager()
    }
    .set();
    let mut engine = jit_engine();
    eval(&mut engine, SOURCE);
    let (chunk, _) = tier_chunk(&mut engine, "d").expect("d's tier chunk");
    let code = resident(&chunk);
    assert!(is_tier_code(&code));
    exits();
    discarded();
    limits();
    calls_discarding_nothing(&mut engine, ONE, &expected[0], 3);
    assert_eq!(loop_limit_state(&chunk), (false, false));
    osr_entries();
    let runs = chunk.jit_runs.get();
    calls_discarding_nothing(&mut engine, NESTED, &expected[1], 1);
    assert_eq!(exits().len(), 5);
    assert_eq!(osr_entries(), 2, "both frames' loops moved");
    assert_eq!(
        loop_limit_state(&chunk),
        (true, true),
        "the inner loop crossed"
    );
    let request = chunk.inline_retry_at.get();
    assert!(request > runs + super::loop_warm_up(&chunk) - 1);
    // The count stopped one below the request: the outer loop's later iterations count nothing.
    assert_eq!(chunk.jit_runs.get(), request - 1);
    assert_eq!(chunk.loop_word(), 0);
    // The next entry takes the check: the code at its loop limit is discarded.
    assert_eq!(eval(&mut engine, ONE), expected[2]);
    assert_eq!(discarded(), vec![code.mem_ptr() as usize]);
    assert_eq!(limits(), vec![(super::Limit::Loop, 5)]);
    assert!(super::reoptimizing(&chunk));
    assert_eq!(loop_limit_state(&chunk), (false, false));
    drop(code);
    super::release_retired(&mut engine.interp);
}

#[test]
fn a_loop_limit_discard_that_arms_no_warm_up_leaves_the_words_idle() {
    // Under TRUST_JS_OPT_NO_REOPT a discard sends new entries to the template JIT's code for
    // good: the loop limit ends with the crossing's request, the words idle, and later entries
    // take no checkpoint and compile nothing.
    for placement in [Placement::Spliced, Placement::InPlace] {
        let expected = in_the_vm(placement, &[LONG]);
        let (mut engine, chunk, code) = exit_then_loop(
            TestSwitches {
                no_reopt: true,
                ..eager()
            },
            placement,
        );
        let (at, _) = calls_until_discarded(&mut engine, LONG, &expected[0], &code, 6);
        assert_eq!(
            at,
            Some(6),
            "{placement:?}: at the entry after the fifth exit"
        );
        assert_eq!(limits(), vec![(super::Limit::Loop, 5)]);
        assert!(!super::reoptimizing(&chunk));
        assert_eq!(loop_limit_state(&chunk), (false, false));
        assert_eq!(
            (chunk.inline_retry_at.get(), chunk.loop_word()),
            (0, 0),
            "{placement:?}: the words idle"
        );
        compiled();
        calls_discarding_nothing(&mut engine, LONG, &expected[0], 3);
        assert!(compiled().is_empty(), "no checkpoint compiles");
        assert!(!is_tier_code(&resident(&chunk)));
        drop(code);
        super::release_retired(&mut engine.interp);
    }
}

#[test]
fn loops_shorter_than_the_warm_up_after_each_exit_discard_nothing_and_exits_count_to_the_limit() {
    // Two iterations after each exit, each frame moving into the borrowed-frame code at its first
    // back edge as on main: past the loop limit each exit arms the warm-up again, which no
    // frame's loop crosses, and the exits count to 100 << R, the 100th (R = 0) and then the 200th
    // (R = 1) discarding the code at its exit.
    const SHORT: &str = "f(2147483647, 2, noop)";
    for placement in [Placement::Spliced, Placement::InPlace] {
        let expected = in_the_vm(placement, &[SHORT]);
        let (mut engine, chunk, code) = exit_then_loop(
            TestSwitches {
                no_exit_profile: true,
                ..eager()
            },
            placement,
        );
        calls_discarding_nothing(&mut engine, SHORT, &expected[0], 50);
        assert_eq!(
            loop_limit_state(&chunk),
            (true, false),
            "{placement:?}: armed, never crossed"
        );
        let (at, transfers) = calls_until_discarded(&mut engine, SHORT, &expected[0], &code, 300);
        assert_eq!(at, Some(50), "{placement:?}: the 100th exit");
        assert!(transfers.iter().all(|&moved| moved), "every frame moved");
        assert_eq!(exits().len(), super::EXIT_LIMIT as usize);
        assert_eq!(limits(), vec![(super::Limit::Exits, 100)]);
        assert_eq!(discards(&chunk), 1);
        assert_eq!(
            loop_limit_state(&chunk),
            (false, false),
            "the discard ends it"
        );
        eval(&mut engine, "for (let k = 0; k < 1000; k++) f(k, 4, noop);");
        let recompiled = resident(&chunk);
        assert!(is_tier_code(&recompiled) && !Rc::ptr_eq(&recompiled, &code));
        exits();
        osr_entries();
        let (at, transfers) =
            calls_until_discarded(&mut engine, SHORT, &expected[0], &recompiled, 300);
        assert_eq!(
            at,
            Some(2 * super::EXIT_LIMIT as usize),
            "{placement:?}: 100 << 1"
        );
        assert!(transfers.iter().all(|&moved| moved));
        assert_eq!(limits(), vec![(super::Limit::Exits, 200)]);
        assert_eq!(discards(&chunk), 2);
        drop((code, recompiled));
        super::release_retired(&mut engine.interp);
    }
}

#[test]
fn each_exit_restarts_the_loop_limits_warm_up() {
    // Each frame's loop counts about half the warm-up: as each exit restarts the count, no loop
    // crosses it (two frames' together would).
    let (mut engine, chunk, code) = exit_then_loop(
        TestSwitches {
            no_exit_profile: true,
            ..eager()
        },
        Placement::Spliced,
    );
    let w = the_loop_weight(&chunk);
    let warm_up = super::loop_warm_up(&chunk);
    assert_eq!(
        warm_up,
        100 * super::ENTRY_WEIGHT,
        "a discard warm-up's first distance"
    );
    let n = warm_up / w / 2 + 2;
    assert!(
        n * w < warm_up && 2 * (n - 1) * w >= warm_up,
        "n {n}, w {w}"
    );
    let source = format!("f(2147483647, {n}, noop)");
    let expected = in_the_vm(Placement::Spliced, &[&source]);
    calls_discarding_nothing(&mut engine, &source, &expected[0], 30);
    assert_eq!(exits().len(), 30);
    assert!(limits().is_empty());
    assert_eq!(loop_limit_state(&chunk), (true, false));
    // The warm-up runs from the last exit: the word is ahead of the runs by the warm-up less the
    // last frame's loop.
    let ahead = chunk.loop_word() - chunk.jit_runs.get();
    assert!(ahead > warm_up / 4 && ahead < warm_up, "{ahead}");
    assert!(Rc::ptr_eq(&resident(&chunk), &code));
}

#[test]
fn entries_into_the_tiers_code_count_nothing_toward_the_loop_limit() {
    // g exits at its start for 2^31 - 1 and has no loop (its tier code is in its first stage):
    // ten exits arm the loop limit, then thousands of entries (from the VM, one by one, and from
    // a template loop's call site) leave the chunk's runs and words as they were, and nothing is
    // discarded.
    eager().set();
    let mut engine = jit_engine();
    eval(
        &mut engine,
        "function g(a) { return (a | 0) + 1; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = (t + g(k)) | 0; t",
    );
    let (chunk, placement) = tier_chunk(&mut engine, "g").expect("g's tier chunk");
    assert_eq!(placement, Placement::InPlace);
    let code = resident(&chunk);
    assert!(is_tier_code(&code));
    exits();
    discarded();
    for _ in 0..10 {
        assert_eq!(eval(&mut engine, "g(2147483647)"), "2147483648");
    }
    assert_eq!(exits().len(), 10);
    assert_eq!(loop_limit_state(&chunk), (true, false));
    let (runs, word) = (chunk.jit_runs.get(), chunk.loop_word());
    for _ in 0..200 {
        assert_eq!(eval(&mut engine, "g(1)"), "2");
    }
    assert_eq!(
        eval(
            &mut engine,
            "let u = 0; for (let k = 0; k < 20000; k++) u = (u + g(k)) | 0; u"
        ),
        "200010000"
    );
    assert!(exits().is_empty() && discarded().is_empty());
    assert_eq!(
        (chunk.jit_runs.get(), chunk.loop_word()),
        (runs, word),
        "entries counted nothing"
    );
    assert_eq!(chunk.inline_retry_at.get(), 0, "the checkpoint word idles");
    assert_eq!(loop_limit_state(&chunk), (true, false));
}

#[test]
fn exits_that_no_loop_transfer_follows_count_to_the_exit_limit() {
    // The loop comes before the exit: each frame exits at its return's overflowing int32 add
    // and finishes in the VM without a back edge, so no loop counts toward the warm-up.
    const SOURCE: &str = "function h(a, n) { let s = 0; for (let i = 0; i < n; i++) s = s + i; \
                          return (s | 0) + (a | 0); }\n\
                          let t = 0; for (let k = 0; k < 1000; k++) t = (t + h(k, 3)) | 0; t";
    TestSwitches {
        no_exit_profile: true,
        ..eager()
    }
    .set();
    let mut engine = jit_engine();
    eval(&mut engine, SOURCE);
    let (chunk, _) = tier_chunk(&mut engine, "h").expect("h's tier chunk");
    let code = resident(&chunk);
    assert!(is_tier_code(&code));
    exits();
    discarded();
    limits();
    osr_entries();
    let (at, transfers) =
        calls_until_discarded(&mut engine, "h(2147483647, 3)", "2147483650", &code, 300);
    assert_eq!(at, Some(super::EXIT_LIMIT as usize));
    assert!(transfers.iter().all(|&moved| !moved), "no transfer");
    assert_eq!(limits(), vec![(super::Limit::Exits, 100)]);
    assert_eq!(discards(&chunk), 1);
    eval(&mut engine, "for (let k = 0; k < 1000; k++) h(k, 3);");
    let recompiled = resident(&chunk);
    assert!(is_tier_code(&recompiled) && !Rc::ptr_eq(&recompiled, &code));
    exits();
    let (at, _) = calls_until_discarded(
        &mut engine,
        "h(2147483647, 3)",
        "2147483650",
        &recompiled,
        300,
    );
    assert_eq!(at, Some(2 * super::EXIT_LIMIT as usize), "100 << 1");
    assert_eq!(limits(), vec![(super::Limit::Exits, 200)]);
    drop((code, recompiled));
    super::release_retired(&mut engine.interp);
}

#[test]
fn the_loop_limit_is_5_shifted_by_r_and_never_above_the_exit_limit() {
    eager().set();
    let mut engine = jit_engine();
    eval(&mut engine, "function g(a) { return a; } g(1);");
    let chunk = first_stage(&mut engine, "g");
    assert_eq!(super::LOOP_EXIT_LIMIT, 5);
    assert_eq!(super::loop_exit_limit(&chunk), 5);
    assert_eq!(super::loop_warm_up(&chunk), 100 * super::ENTRY_WEIGHT);
    let profile = ChunkProfile::of(&chunk);
    profile.add_discard();
    assert_eq!(super::loop_exit_limit(&chunk), 10);
    assert_eq!(super::loop_warm_up(&chunk), 200 * super::ENTRY_WEIGHT);
    for _ in 0..30 {
        profile.add_discard();
    }
    assert_eq!(super::loop_exit_limit(&chunk), 5 << profile::MAX_DISCARDS);
    assert_eq!(
        super::loop_warm_up(&chunk),
        (100 * super::ENTRY_WEIGHT) << profile::MAX_DISCARDS
    );
    // Never above the exit limit: an exit limit of 1 discards at the first exit as before.
    TestSwitches {
        exit_limit: 1,
        ..eager()
    }
    .set();
    assert_eq!(super::loop_exit_limit(&chunk), 1 << profile::MAX_DISCARDS);
    TestSwitches {
        exit_limit: 3,
        ..eager()
    }
    .set();
    assert_eq!(super::loop_exit_limit(&chunk), 3 << profile::MAX_DISCARDS);
    // Its base is a switch (TRUST_JS_OPT_LOOP_EXIT_LIMIT), under the exit limit too.
    TestSwitches {
        loop_exit_limit: 7,
        ..eager()
    }
    .set();
    assert_eq!(super::loop_exit_limit(&chunk), 7 << profile::MAX_DISCARDS);
    TestSwitches {
        loop_exit_limit: 300,
        ..eager()
    }
    .set();
    assert_eq!(
        super::loop_exit_limit(&chunk),
        super::exit_limit(&chunk),
        "100 << R"
    );
}

#[test]
fn an_exit_limit_of_one_discards_at_the_first_exit_whatever_follows() {
    for placement in [Placement::Spliced, Placement::InPlace] {
        let expected = in_the_vm(placement, &[LONG]);
        let (mut engine, chunk, code) = exit_then_loop(
            TestSwitches {
                exit_limit: 1,
                ..eager()
            },
            placement,
        );
        let table = super::exit::table_of(&code).expect("its table");
        assert_eq!((table.limit, table.loop_limit), (1, 1));
        let (at, transfers) = calls_until_discarded(&mut engine, LONG, &expected[0], &code, 10);
        assert_eq!(at, Some(1), "{placement:?}");
        assert_eq!(
            transfers,
            vec![true],
            "the discarded code's frame moves at once"
        );
        assert_eq!(exits().len(), 1);
        assert_eq!(limits(), vec![(super::Limit::Exits, 1)]);
        assert_eq!(discards(&chunk), 1);
        assert!(!loop_limit_state(&chunk).0, "a discarded code arms nothing");
        drop(code);
        super::release_retired(&mut engine.interp);
    }
}

#[test]
fn a_frame_in_the_vm_before_the_codes_install_discards_it_at_its_loop_limit_too() {
    // JSC's check reads the replacement of the frame's code block, whichever frame loops: here
    // a frame that resumed in the VM from the first code (its first exit, below the loop limit)
    // calls `drive` at its first iteration, in which 99 more exits discard that code at its exit
    // limit, the warm-up installs the recompile and 10 exits of one-iteration loops bring the
    // recompile to its loop limit (5 << 1), arming its warm-up; the frame's loop, moved into the
    // borrowed-frame code at its first back edge after `drive` returns, crosses the warm-up, and
    // the next entry discards the recompile, installed after the frame started in the VM (the
    // spliced program, whose `f` calls `cb`).
    const DRIVE: &str = "function drive() { \
         for (let k = 0; k < 99; k++) f(2147483647, 1, noop); \
         for (let k = 0; k < 1000; k++) f(k, 4, noop); \
         for (let k = 0; k < 10; k++) f(2147483647, 1, noop); }";
    const CALL: &str = "f(2147483647, 2000, drive)";
    let placement = Placement::Spliced;
    let expected = in_the_vm(placement, &[DRIVE, CALL, "f(1, 0, noop)"]);
    let (mut engine, chunk, code) = exit_then_loop(
        TestSwitches {
            no_exit_profile: true,
            ..eager()
        },
        placement,
    );
    eval(&mut engine, DRIVE);
    assert_eq!(eval(&mut engine, CALL), expected[1]);
    assert_eq!(exits().len(), 1 + 99 + 10);
    assert_eq!(
        loop_limit_state(&chunk),
        (true, true),
        "the frame's loop crossed"
    );
    assert_eq!(eval(&mut engine, "f(1, 0, noop)"), expected[2]);
    let gone = discarded();
    assert_eq!(gone.len(), 2, "{gone:?}");
    assert_eq!(gone[0], code.mem_ptr() as usize, "the first code first");
    assert_eq!(
        limits(),
        vec![(super::Limit::Exits, 100), (super::Limit::Loop, 10)],
        "the recompile at its loop limit"
    );
    assert_eq!(discards(&chunk), 2);
    drop(code);
    super::release_retired(&mut engine.interp);
}

#[test]
fn with_the_tier_off_an_exited_frames_loop_is_unchanged() {
    // No tier code, so no exit, no table and no loop limit; the template JIT's code counts no
    // loop, and the words agree.
    TestSwitches::default().set();
    let expected = in_the_vm(Placement::Spliced, &[LONG]);
    let mut engine = jit_engine();
    compiled();
    exits();
    discarded();
    limits();
    eval(&mut engine, Placement::Spliced.source());
    let (chunk, placement) = tier_chunk(&mut engine, "f").expect("f's second stage");
    assert_eq!(
        placement,
        Placement::Spliced,
        "the template JIT's second stage"
    );
    let code = resident(&chunk);
    assert!(
        super::exit::table_of(&code).is_none(),
        "the template JIT's code"
    );
    for _ in 0..10 {
        assert_eq!(eval(&mut engine, LONG), expected[0]);
    }
    assert!(compiled().is_empty() && exits().is_empty() && discarded().is_empty());
    assert!(limits().is_empty());
    assert_eq!(loop_limit_state(&chunk), (false, false));
    assert_eq!(
        (chunk.loop_word(), chunk.inline_retry_at.get()),
        (0, 0),
        "the words agree"
    );
    assert!(Rc::ptr_eq(&resident(&chunk), &code));
}

#[test]
fn a_loop_limit_discard_logs_its_cause_at_the_end_of_the_discard_line() {
    // TRUST_JS_OPT_LOG is read once a process: the log is a child process's.
    const CHILD: &str = "TRUST_JS_TEST_OPT_LOOP_LIMIT_LOG_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "opt::loop_limit_tests::a_loop_limit_discard_logs_its_cause_at_the_end_of_the_discard_line",
                "--test-threads=1",
                // The harness captures a test's stderr: the log goes to the child's own.
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("TRUST_JS_OPT_LOG", "1")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "{}\n{stderr}",
            String::from_utf8_lossy(&output.stdout)
        );
        let lines: Vec<&str> = stderr
            .lines()
            .filter(|line| line.starts_with("[opt] discarded fn("))
            .collect();
        assert_eq!(lines.len(), 3, "{lines:#?}");
        // The loop limit's: the line's fields as before, its cause at the end; the second stage's
        // fallback compiled, the first stage's kept.
        for (line, now) in [
            (lines[0], "the template JIT's code"),
            (lines[1], "the template JIT's kept code"),
        ] {
            let (head, cause) = line
                .split_once("; ")
                .unwrap_or_else(|| panic!("a cause: {line}"));
            assert!(
                head.contains(&format!(
                    " after 5 exits (limit 100, R 0 -> 1) for {now}: pc "
                )) && head.ends_with(" Overflow x5"),
                "{line}"
            );
            assert_eq!(cause, "at its loop limit: 5 exits >= 5");
        }
        // The exit limit's, unchanged.
        assert!(
            lines[2].contains(
                " after 100 exits (limit 100, R 0 -> 1) for the template JIT's code: pc "
            ) && lines[2].ends_with(" Overflow x100"),
            "{}",
            lines[2]
        );
        return;
    }
    // A long loop after each exit: the loop limit's discard at the entry after the fifth call,
    // of a second stage's code and of a first stage's.
    for placement in [Placement::Spliced, Placement::InPlace] {
        let (mut engine, _, code) = exit_then_loop(eager(), placement);
        for _ in 0..6 {
            eval(&mut engine, LONG);
        }
        assert_eq!(discarded(), vec![code.mem_ptr() as usize]);
    }
    // A short one: the exit limit's at the 100th.
    let (mut engine, _, code) = exit_then_loop(eager(), Placement::Spliced);
    for _ in 0..100 {
        eval(&mut engine, "f(2147483647, 2, noop)");
    }
    assert_eq!(discarded(), vec![code.mem_ptr() as usize]);
}
