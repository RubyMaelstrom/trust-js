//! The tier's code on the first stage (P3 stage 2, step 14; the plan's "Step 14, the tier's code
//! on the first stage"): a function whose plan splices nothing gets the tier's code in its own
//! chunk's slot, over the template JIT's code, which stays as the fallback, and no second chunk.
//! A call way filled before the install enters the tier's code at its next call, through the
//! chunk's entry word and each path's refill, with no move of the call caches' epoch; a discard
//! returns to the kept code without a compile, and compiles one where the kept code is gone; the
//! warm-up and the reoptimization run on the first stage; no later checkpoint builds a second
//! stage over it, but the chain recompile supersedes it; its exit profile, keyed on the first
//! stage, is read by a caller's splice.

use super::ir::ExitReason;
use super::profile::ChunkProfile;
use super::TestSwitches;
use crate::bytecode::{CallIc, Chunk, Op, Tier, CALL_IC_EPOCH};
use crate::value::{ObjectKind, Value, ValueView};
use crate::{Completion, Engine};
use std::rc::Rc;
use std::sync::atomic::Ordering::Relaxed;

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

/// The tier on and eager, compiling on the main thread: an install happens in the call whose
/// checkpoint asks for it, which these tests read at once.
fn synchronous() -> TestSwitches {
    TestSwitches {
        on: true,
        eager: true,
        asynchronous: false,
        ..Default::default()
    }
}

fn value(engine: &mut Engine, name: &str) -> Value {
    let env = engine.interp.global_env;
    engine.interp.get_var(name, &env).ok().expect("a global")
}

/// The global user function `name`: its identity (what the ways record) and its function.
fn function(engine: &mut Engine, name: &str) -> (usize, Rc<crate::ast::Function>) {
    let value = value(engine, name);
    let ValueView::Obj(object) = value.view() else {
        panic!("{name} is not an object")
    };
    let key = crate::value::Gc::as_ptr(&object) as usize;
    let ObjectKind::User(user) = &object.borrow().kind else {
        panic!("{name} is not a user function")
    };
    (key, Rc::clone(&user.func))
}

fn first_stage(func: &crate::ast::Function) -> Rc<Chunk> {
    func.runtime()
        .code
        .get()
        .and_then(Option::clone)
        .expect("a compiled function")
}

fn second_stage(func: &crate::ast::Function) -> Option<Rc<Chunk>> {
    func.runtime().code2.get().and_then(Option::clone)
}

/// Whether `code` is the tier's (it has an exit table), not the template JIT's.
fn is_tier_code(code: &crate::jit::JitCode) -> bool {
    super::exit::table_bytes(code.mem_ptr()) != 0
}

/// The code in `chunk`'s slot.
fn resident(chunk: &Chunk) -> Rc<crate::jit::JitCode> {
    chunk.jit.get().flatten().expect("resident code")
}

/// The template JIT's code an install kept for `chunk`.
fn kept(chunk: &Chunk) -> Option<Rc<crate::jit::JitCode>> {
    ChunkProfile::get(chunk).and_then(ChunkProfile::kept_code)
}

fn compiled() -> Vec<usize> {
    super::TEST_COMPILED.with(|compiled| std::mem::take(&mut *compiled.borrow_mut()))
}

fn discarded() -> Vec<usize> {
    super::TEST_DISCARDED.with(|discarded| std::mem::take(&mut *discarded.borrow_mut()))
}

fn installed_in_place() -> Vec<usize> {
    super::TEST_IN_PLACE.with(|in_place| std::mem::take(&mut *in_place.borrow_mut()))
}

fn refills() -> usize {
    crate::bytecode::TEST_STALE_REFILLS.with(std::cell::Cell::get)
}

/// The code the last call or construction through a cache entry entered from Rust.
fn entered() -> usize {
    crate::interpreter::TEST_ENTERED_CODE.with(std::cell::Cell::get)
}

/// Call `name` from Rust with `args`.
fn call(engine: &mut Engine, name: &str, args: &[Value]) -> Value {
    let callee = value(engine, name);
    engine
        .interp
        .call(callee, Value::Undefined, args)
        .ok()
        .expect("the call returns")
}

/// Call `name` from Rust with the numbers `k, 1` until its first stage carries the tier's code,
/// installed at its checkpoint (Rust's calls are native entries, which count toward it): the
/// calls made.
fn until_in_place(engine: &mut Engine, name: &str) -> usize {
    let (_, func) = function(engine, name);
    for calls in 1..=2_000 {
        call(engine, name, &[Value::Num(calls as f64), Value::Num(1.0)]);
        if super::in_place(&first_stage(&func)) {
            return calls;
        }
    }
    panic!("{name}'s first stage never got the tier's code");
}

/// Run the tests here that read the call caches' process-wide epoch one at a time: each makes and
/// drops engines, and a dropped function runtime moves the epoch, which stales another such test's
/// run (they retry, and two that retried together kept staling each other's runs).
fn epoch_quiet() -> std::sync::MutexGuard<'static, ()> {
    static EPOCH_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    EPOCH_TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The way of `chunk`'s call sites that holds the callee `key`.
fn way(chunk: &Chunk, key: usize) -> Option<CallIc> {
    chunk.jit_ops().iter().find_map(|op| {
        let (Op::Call(_, cache) | Op::CallWithThis(_, cache)) = op else {
            return None;
        };
        // A site's ways are contiguous cells from its first (`Chunk::jit_call_cache_ptr`).
        let ways = chunk.jit_call_cache_ptr(*cache) as *const std::cell::Cell<CallIc>;
        (0..crate::bytecode::CALL_IC_WAYS)
            .map(|k| unsafe { (*ways.add(k)).get() })
            .find(|ic| ic.callee == key)
    })
}

#[test]
fn an_empty_plans_function_runs_the_tiers_code_on_its_first_stage_with_no_second_chunk() {
    synchronous().set();
    let mut engine = jit_engine();
    compiled();
    installed_in_place();
    eval(
        &mut engine,
        "function sq(x) { return x * x + 1; }\n\
         function seven() { var t = 0; for (var k = 0; k < 1000; k++) t += sq(k % 7); return t; }",
    );
    let (_, sq) = function(&mut engine, "sq");
    call(&mut engine, "sq", &[Value::Num(2.0)]);
    let first = first_stage(&sq);
    let template = resident(&first);
    assert!(!is_tier_code(&template), "the template JIT's code first");
    assert_eq!(first.entry_word(), 0);
    let calls = until_in_place(&mut engine, "sq");
    assert!(
        calls <= 200,
        "the first checkpoint compiles it ({calls} calls)"
    );
    assert!(second_stage(&sq).is_none(), "no second chunk");
    assert_eq!(compiled(), vec![Rc::as_ptr(&first) as usize]);
    assert_eq!(installed_in_place(), vec![Rc::as_ptr(&first) as usize]);
    let code = resident(&first);
    assert!(
        is_tier_code(&code),
        "the tier's code in the first stage's own slot"
    );
    assert!(
        Rc::ptr_eq(&kept(&first).expect("the template code kept"), &template),
        "the template JIT's code stays as the fallback"
    );
    assert_eq!(first.entry_word(), 1, "one change of its code in place");
    assert_eq!(first.inline_retry_at.get(), 0, "its checkpoints ended");
    assert!(first.feedback_vector().record_map().is_none());
    code.referenced.set(0);
    assert_eq!(eval(&mut engine, "seven()"), "13977");
    assert!(code.referenced.get() != 0, "the tier's code ran");
    for _ in 0..20 {
        assert_eq!(eval(&mut engine, "seven()"), "13977");
    }
    assert!(second_stage(&sq).is_none(), "still none");
    assert!(
        !compiled().contains(&(Rc::as_ptr(&first) as usize)),
        "no compile of it after it"
    );
    assert!(Rc::ptr_eq(&resident(&first), &code));
}

/// One path's redirect test: `source` defines `callee`, an empty plan's function (`callee(k, 1)`
/// a number), and code that reaches it through the path; `run` runs that code, twice to fill
/// the path's way and once more after the callee's first stage got the tier's code, each time
/// with the result `expected`. `refill`: the way refills through `refill_stale` (a call way's
/// probe), not through its fill (a construct site's, an overflow entry's). `entry`: the way's
/// entry, where the test can read it.
struct Path<'a> {
    source: &'a str,
    callee: &'a str,
    run: &'a str,
    expected: &'a str,
    refill: bool,
    entry: Option<&'a dyn Fn(&mut Engine) -> Option<CallIc>>,
}

/// Run `path`'s redirect: the way filled before the install held the template code, and after
/// `run` holds the tier's code with the epoch it had, the call caches' epoch unmoved; the run
/// entered the tier's code from Rust (the refill's call), and not the kept code. Retries where a
/// test beside this one moved the process-wide epoch.
fn redirect(path: Path<'_>) {
    let _quiet = epoch_quiet();
    let callee = path.callee;
    for attempt in 0..100 {
        synchronous().set();
        let start = CALL_IC_EPOCH.load(Relaxed);
        let mut engine = jit_engine();
        eval(&mut engine, path.source);
        assert_eq!(eval(&mut engine, path.run), path.expected);
        assert_eq!(eval(&mut engine, path.run), path.expected);
        let (_, func) = function(&mut engine, callee);
        let first = first_stage(&func);
        assert!(!super::in_place(&first), "{callee}: not yet");
        let template = resident(&first);
        let before = path.entry.map(|entry| {
            let before = entry(&mut engine).unwrap_or_else(|| panic!("{callee}: a way"));
            assert_eq!(
                before.code,
                Rc::as_ptr(&template),
                "{callee}: the template code"
            );
            assert_eq!(before.entry, 0);
            before
        });
        if before.is_none() {
            assert_eq!(
                entered(),
                template.mem_ptr() as usize,
                "{callee}: the template code"
            );
        }
        until_in_place(&mut engine, callee);
        let code = resident(&first);
        assert!(is_tier_code(&code));
        assert!(Rc::ptr_eq(&kept(&first).expect("kept"), &template));
        if let (Some(entry), Some(before)) = (path.entry, before) {
            assert_eq!(
                entry(&mut engine).map(|ic| ic.code),
                Some(before.code),
                "{callee}: the way holds the template code until its next call"
            );
        }
        code.referenced.set(0);
        template.referenced.set(0);
        let refilled = refills();
        crate::interpreter::TEST_ENTERED_CODE.with(|entered| entered.set(0));
        assert_eq!(eval(&mut engine, path.run), path.expected);
        if CALL_IC_EPOCH.load(Relaxed) != start && attempt < 99 {
            continue;
        }
        assert_eq!(
            CALL_IC_EPOCH.load(Relaxed),
            start,
            "{callee}: no epoch move"
        );
        if let (Some(entry), Some(before)) = (path.entry, before) {
            let after = entry(&mut engine).unwrap_or_else(|| panic!("{callee}: the way"));
            assert_eq!(after.code, Rc::as_ptr(&code), "{callee}: the tier's code");
            assert_eq!(after.code_mem, code.mem_ptr());
            assert_eq!(after.entry, u32::from(first.entry_word()));
            assert_eq!(after.epoch, before.epoch, "{callee}: refilled in its epoch");
        }
        if path.refill {
            assert!(refills() > refilled, "{callee}: refill_stale refilled it");
        }
        assert_eq!(
            entered(),
            code.mem_ptr() as usize,
            "{callee}: entered from Rust"
        );
        assert!(code.referenced.get() != 0, "{callee}: the tier's code ran");
        assert_eq!(
            template.referenced.get(),
            0,
            "{callee}: the kept code did not"
        );
        assert!(second_stage(&func).is_none());
        break;
    }
}

fn call_way(engine: &mut Engine) -> Option<CallIc> {
    let (key, _) = function(engine, "add");
    let (_, caller) = function(engine, "caller");
    way(&first_stage(&caller), key)
}

#[test]
fn a_way_filled_before_the_install_enters_the_tiers_code_through_the_direct_sequence() {
    redirect(Path {
        source: "function add(x, y) { return x * x + y; }\n\
                 function caller(n) { var t = 0; for (var k = 0; k < n; k++) t += add(k, 1); \
                 return t; }",
        callee: "add",
        run: "caller(5)",
        expected: "35",
        refill: true,
        entry: Some(&call_way),
    });
}

#[test]
fn a_way_filled_before_the_install_enters_the_tiers_code_through_the_call_helper() {
    // A sloppy callee that reads `this`, called without a receiver, fails the direct
    // sequence's static gates: every call takes the call helper (`jit_call_hit`).
    redirect(Path {
        source: "function add(x, y) { var self = this; return x * x + y; }\n\
                 function caller(n) { var t = 0; for (var k = 0; k < n; k++) t += add(k, 1); \
                 return t; }",
        callee: "add",
        run: "caller(5)",
        expected: "35",
        refill: true,
        entry: Some(&call_way),
    });
}

#[test]
fn a_way_filled_before_the_install_enters_the_tiers_code_through_a_rust_caller() {
    // A Proxy's `get` trap runs through the read's call cache from Rust
    // (`Interp::call_jit_cached`, `resolve_cached_call`).
    redirect(Path {
        source: "function add(t, key) { return key.length * 2; }\n\
                 var p = new Proxy({}, { get: add });\n\
                 function reads(n) { var t = 0; for (var k = 0; k < n; k++) t += p.abc; \
                 return t; }",
        callee: "add",
        run: "reads(5)",
        expected: "30",
        refill: true,
        entry: None,
    });
}

#[test]
fn a_prepared_call_enters_the_tiers_code_after_an_install_mid_iteration() {
    // A Map's forEach calls its callback through a prepared entry (`Interp::prepared_call_current`):
    // the callback's checkpoint installs the tier's code a hundred calls in, and the rest of the
    // iteration's calls enter it, the last of them from Rust, with no epoch move.
    let _quiet = epoch_quiet();
    for attempt in 0..100 {
        synchronous().set();
        let start = CALL_IC_EPOCH.load(Relaxed);
        let mut engine = jit_engine();
        installed_in_place();
        eval(
            &mut engine,
            "var acc = 0; function cb(v) { acc += v * v; }\n\
             var m = new Map(); for (var k = 0; k < 400; k++) m.set(k, k);\n\
             function each() { acc = 0; m.forEach(cb); return acc; }",
        );
        let (_, cb) = function(&mut engine, "cb");
        let sum: u64 = (0..400u64).map(|k| k * k).sum();
        assert_eq!(eval(&mut engine, "each()"), sum.to_string());
        let first = first_stage(&cb);
        assert!(super::in_place(&first), "the checkpoint came mid-iteration");
        assert_eq!(installed_in_place(), vec![Rc::as_ptr(&first) as usize]);
        if CALL_IC_EPOCH.load(Relaxed) != start && attempt < 99 {
            continue;
        }
        assert_eq!(CALL_IC_EPOCH.load(Relaxed), start, "no epoch move");
        let code = resident(&first);
        assert!(is_tier_code(&code));
        assert_eq!(
            entered(),
            code.mem_ptr() as usize,
            "the iteration's last call entered the tier's code"
        );
        assert!(second_stage(&cb).is_none());
        break;
    }
}

#[test]
fn a_construct_site_filled_before_the_install_enters_the_tiers_code() {
    let construct_way = |engine: &mut Engine| {
        let (key, _) = function(engine, "P");
        let (_, make) = function(engine, "make");
        let chunk = first_stage(&make);
        let cache = chunk
            .jit_ops()
            .iter()
            .find_map(|op| match op {
                Op::New(_, cache) => Some(*cache),
                _ => None,
            })
            .expect("make's `new` site");
        Some(chunk.construct_cache(cache).call).filter(|ic| ic.callee == key)
    };
    redirect(Path {
        source: "function P(x, y) { this.v = x * x + y; }\n\
                 function make(n) { var t = 0; for (var k = 0; k < n; k++) t += new P(k, 1).v; \
                 return t; }",
        callee: "P",
        run: "make(5)",
        expected: "35",
        refill: false,
        entry: Some(&construct_way),
    });
}

#[test]
fn an_overflow_entry_filled_before_the_install_enters_the_tiers_code() {
    // Six callees through four ways: two live in the overflow table, whose entries the probe in
    // line reads before the direct sequence, and Rust's resolution after the ways.
    let _quiet = epoch_quiet();
    for attempt in 0..100 {
        synchronous().set();
        let start = CALL_IC_EPOCH.load(Relaxed);
        let mut engine = jit_engine();
        eval(
            &mut engine,
            "var fs = []; for (var q = 0; q < 6; q++) fs.push(new Function('x', 'y', \
             'return x * x + y + ' + q + ';'));\n\
             function wide(n) { var s = 0; for (var k = 0; k < n; k++) s += fs[k % 6](k, 1); \
             return s; }",
        );
        assert_eq!(eval(&mut engine, "wide(12)"), "548");
        assert_eq!(eval(&mut engine, "wide(12)"), "548");
        let genv = crate::interpreter::Env::as_ptr(&engine.interp.global_env) as usize;
        let epoch = CALL_IC_EPOCH.load(Relaxed);
        let (_, wide) = function(&mut engine, "wide");
        let wide = first_stage(&wide);
        // A callee the table holds and no way of the site does.
        let found = (0..6).find_map(|q| {
            eval(&mut engine, &format!("var probed = fs[{q}];"));
            let (key, func) = function(&mut engine, "probed");
            if way(&wide, key).is_some() {
                return None;
            }
            let entry = engine.interp.call_overflow.lookup(key, genv, epoch)?;
            Some((key, func, entry))
        });
        // Entries of another epoch miss the lookup: a test beside this one moved it.
        let Some((key, func, before)) = found.or_else(|| {
            assert!(
                CALL_IC_EPOCH.load(Relaxed) != start && attempt < 99,
                "a callee in the overflow table"
            );
            None
        }) else {
            continue;
        };
        let first = first_stage(&func);
        let template = resident(&first);
        assert_eq!(before.code, Rc::as_ptr(&template));
        until_in_place(&mut engine, "probed");
        let code = resident(&first);
        assert!(is_tier_code(&code));
        code.referenced.set(0);
        template.referenced.set(0);
        assert_eq!(eval(&mut engine, "wide(12)"), "548");
        if CALL_IC_EPOCH.load(Relaxed) != start && attempt < 99 {
            continue;
        }
        assert_eq!(CALL_IC_EPOCH.load(Relaxed), start, "no epoch move");
        let after = engine
            .interp
            .call_overflow
            .lookup(key, genv, epoch)
            .expect("the overflow entry");
        assert_eq!(after.code, Rc::as_ptr(&code), "the tier's code");
        assert_eq!(after.entry, u32::from(first.entry_word()));
        // The site's other callees enter from Rust too (their checkpoints): the code that ran is
        // told by the reference words.
        assert!(code.referenced.get() != 0, "the tier's code ran");
        assert_eq!(template.referenced.get(), 0, "the kept code did not");
        break;
    }
}

/// A function whose tier code is in place, with an exit limit of 1: `add(k, 1)` numbers, a
/// string exits.
fn add_in_place(switches: TestSwitches) -> (Engine, Rc<Chunk>) {
    switches.set();
    let mut engine = jit_engine();
    eval(&mut engine, "function add(a, b) { return a + b; }");
    until_in_place(&mut engine, "add");
    let (_, add) = function(&mut engine, "add");
    let first = first_stage(&add);
    assert!(is_tier_code(&resident(&first)));
    (engine, first)
}

fn exit_limit_one() -> TestSwitches {
    TestSwitches {
        exit_limit: 1,
        ..synchronous()
    }
}

#[test]
fn a_discard_returns_to_the_kept_code_without_a_compile() {
    let (mut engine, first) = add_in_place(exit_limit_one());
    let tier = resident(&first);
    let template = kept(&first).expect("the template code kept");
    discarded();
    compiled();
    // A string fails the Number checks: at the limit of 1, the discard.
    assert_eq!(eval(&mut engine, "add('a', 'b')"), "ab");
    assert_eq!(discarded(), vec![tier.mem_ptr() as usize]);
    assert!(
        Rc::ptr_eq(&resident(&first), &template),
        "the kept code again, not a compile"
    );
    assert!(kept(&first).is_none(), "taken back from its slot");
    assert_eq!(first.entry_word(), 2);
    // R on the first stage, its warm-up armed on its own word.
    assert_eq!(
        ChunkProfile::get(&first).map(ChunkProfile::discards),
        Some(1)
    );
    assert!(super::reoptimizing(&first));
    assert_ne!(first.inline_retry_at.get(), 0);
    assert!(compiled().is_empty());
    let (_, add) = function(&mut engine, "add");
    assert!(second_stage(&add).is_none());
    assert_eq!(eval(&mut engine, "add(2, 3) + add('c', 'd')"), "5cd");
}

#[test]
fn a_discard_whose_kept_code_was_evicted_compiles_the_template_code() {
    let (mut engine, first) = add_in_place(exit_limit_one());
    let tier = resident(&first);
    // CLOCK evicts the kept code as it evicts any: out of its slot, the epoch moved.
    let evicted = ChunkProfile::get(&first)
        .expect("a profile")
        .kept()
        .discard()
        .expect("the kept code");
    crate::bytecode::invalidate_call_caches();
    discarded();
    assert_eq!(eval(&mut engine, "add('a', 'b')"), "ab");
    assert_eq!(discarded(), vec![tier.mem_ptr() as usize]);
    let now = resident(&first);
    assert!(!is_tier_code(&now), "the template JIT's code");
    assert!(!Rc::ptr_eq(&now, &evicted), "compiled now");
    assert_ne!(now.mem_ptr(), evicted.mem_ptr());
    assert_eq!(first.entry_word(), 2);
    assert!(super::reoptimizing(&first));
}

#[test]
fn the_warm_up_and_the_reoptimization_run_on_the_first_stage() {
    let (mut engine, first) = add_in_place(exit_limit_one());
    assert_eq!(eval(&mut engine, "add('a', 'b')"), "ab");
    let template = resident(&first);
    assert!(!is_tier_code(&template));
    assert!(super::reoptimizing(&first));
    compiled();
    // Numbers until the warm-up's checkpoint recompiles the first stage, past 100 << R entries.
    let mut calls = 0;
    while !is_tier_code(&resident(&first)) {
        call(
            &mut engine,
            "add",
            &[Value::Num(f64::from(calls)), Value::Num(1.0)],
        );
        calls += 1;
        assert!(calls < 5_000, "the warm-up reoptimizes");
    }
    assert!(calls >= 150, "the warm-up waits 100 << R entries ({calls})");
    assert_eq!(compiled(), vec![Rc::as_ptr(&first) as usize]);
    assert!(
        Rc::ptr_eq(&kept(&first).expect("kept again"), &template),
        "the reoptimization's install keeps the template code again"
    );
    assert_eq!(first.entry_word(), 3);
    assert!(!super::reoptimizing(&first));
    assert_eq!(first.inline_retry_at.get(), 0);
    let (_, add) = function(&mut engine, "add");
    assert!(second_stage(&add).is_none());
    // The recompile declined the flagged site: strings run in it without an exit.
    let code = resident(&first);
    assert_eq!(eval(&mut engine, "add('x', 'y') + add(1, 2)"), "xy3");
    assert!(Rc::ptr_eq(&resident(&first), &code), "no discard");
}

#[test]
fn no_later_checkpoint_builds_a_second_stage_over_the_tiers_code() {
    let (mut engine, first) = add_in_place(synchronous());
    assert_eq!(first.inline_retry_at.get(), 0);
    let code = resident(&first);
    compiled();
    // A checkpoint armed again, as a superseded chunk current again in another Realm, or a
    // stale chain's request the chain recompile declined, arms it.
    first.set_inline_retry_at(first.jit_runs.get().saturating_add(1));
    call(&mut engine, "add", &[Value::Num(1.0), Value::Num(2.0)]);
    let (_, add) = function(&mut engine, "add");
    assert!(second_stage(&add).is_none(), "no second stage on top");
    assert!(compiled().is_empty(), "no compile of the first stage again");
    assert_eq!(first.inline_retry_at.get(), 0, "its checkpoints end again");
    assert!(Rc::ptr_eq(&resident(&first), &code));
}

#[test]
fn the_chain_recompile_supersedes_a_first_stage_that_carries_the_tiers_code() {
    let (mut engine, first) = add_in_place(synchronous());
    eval(
        &mut engine,
        "function caller(n) { var t = 0; for (var k = 0; k < n; k++) t += add(k, 1); return t; }",
    );
    assert_eq!(eval(&mut engine, "caller(5)"), "15");
    compiled();
    // A stale chain's request, as `Chunk::note_chain_replay` makes it.
    first.reoptimize.set(true);
    first.set_inline_retry_at(first.jit_runs.get().saturating_add(1).max(1));
    call(&mut engine, "add", &[Value::Num(1.0), Value::Num(2.0)]);
    let (add_key, add) = function(&mut engine, "add");
    let code2 = second_stage(&add).expect("the recompile without property chains");
    assert!(code2.no_property_chains.get());
    assert!(
        compiled().contains(&(Rc::as_ptr(&code2) as usize)),
        "the tier compiled it"
    );
    assert!(
        first.superseded(),
        "it supersedes the first stage, the tier's code and all"
    );
    assert!(is_tier_code(&resident(&code2)));
    // The caller's way takes it at its next call.
    assert_eq!(eval(&mut engine, "caller(5)"), "15");
    let (_, caller) = function(&mut engine, "caller");
    let picked = way(&first_stage(&caller), add_key).expect("caller's way");
    assert!(std::ptr::eq(picked.chunk_raw, Rc::as_ptr(&code2)));
}

#[test]
fn the_exit_profile_on_the_first_stage_is_read_by_a_callers_splice() {
    // `leaf`'s own code exits at its add, keyed on its first stage (it has no other chunk); a
    // caller's second stage that splices it declines the site.
    exit_limit_one().set();
    let mut engine = jit_engine();
    eval(
        &mut engine,
        "function leaf(a, b) { return a + b; }\n\
         function c(n) { var t = 0; for (var k = 0; k < n; k++) t += leaf(k, 1); return t; }",
    );
    let (_, c) = function(&mut engine, "c");
    let mut runs = 0;
    while second_stage(&c).is_none() {
        assert_eq!(eval(&mut engine, "c(10)"), "55");
        runs += 1;
        assert!(runs < 200, "c's second stage splices leaf");
    }
    let spliced = second_stage(&c).expect("c's second stage");
    until_in_place(&mut engine, "leaf");
    let (_, leaf) = function(&mut engine, "leaf");
    let first = first_stage(&leaf);
    // A string exits leaf's code at its add: the discard keys the site on the first stage.
    assert_eq!(eval(&mut engine, "leaf('a', 'b')"), "ab");
    let add = first
        .jit_ops()
        .iter()
        .position(|op| matches!(op, Op::Add))
        .expect("leaf's add");
    let mut keys = Vec::new();
    ChunkProfile::get(&first)
        .expect("leaf's profile")
        .sites_at(add as u32, |reason, inlined, _| {
            keys.push((reason, inlined))
        });
    assert_eq!(
        keys,
        vec![(ExitReason::NotNumber as u8, false)],
        "an own key"
    );
    assert!(second_stage(&leaf).is_none());
    // c's splice of leaf finds it at the spliced add.
    let map = spliced.feedback_vector().record_map().expect("a map");
    let pc = (0..spliced.jit_ops().len())
        .find(|&pc| {
            map.site(pc).is_some_and(|(baseline, at, spliced)| {
                std::ptr::eq(baseline, &*first) && at == add && spliced
            })
        })
        .expect("the spliced add");
    let snapshot = super::snapshot::Snapshot::take(&spliced);
    assert!(snapshot.declines(pc, ExitReason::NotNumber));
}
