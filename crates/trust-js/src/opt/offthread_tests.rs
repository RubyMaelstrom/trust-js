//! The tier compiling off the main thread (P3 stage 2, step 12; the plan's "Step 12, compiling
//! off the main thread"): a function's checkpoint queues its compile and runs the code it has
//! until the poll after the compile ends installs the tier's code; a result whose chunk was
//! superseded, whose sites a discard flagged, or whose function was collected meanwhile is
//! dropped, and the function tiers up again; an engine dropped with compiles pending cancels
//! them; a process confined to one CPU compiles on the main thread.

use super::ir::ExitReason;
use super::profile::ChunkProfile;
use super::TestSwitches;
use crate::bytecode::{Chunk, Tier};
use crate::value::Value;
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

/// The tier on, eager, compiling off the main thread.
fn asynchronous() -> TestSwitches {
    TestSwitches {
        on: true,
        eager: true,
        asynchronous: true,
        ..Default::default()
    }
}

/// Set `asynchronous()`'s switches: whether this process compiles off the main thread with
/// them, which takes a second CPU (`opt::asynchronous`): a test of the asynchronous path run on
/// one CPU has nothing to test, and returns (`one_cpu_compiles_on_the_main_thread` tests that).
fn asynchronous_here() -> bool {
    asynchronous().set();
    let here = super::asynchronous();
    if !here {
        eprintln!("one CPU: the tier compiles on the main thread, nothing asynchronous to test");
    }
    here
}

fn jit_engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

fn compiled() -> Vec<usize> {
    super::TEST_COMPILED.with(|compiled| std::mem::take(&mut *compiled.borrow_mut()))
}

fn dropped() -> Vec<(usize, &'static str)> {
    super::TEST_DROPPED.with(|dropped| std::mem::take(&mut *dropped.borrow_mut()))
}

fn function(engine: &mut Engine, name: &str) -> Value {
    let env = engine.interp.global_env;
    engine.interp.get_var(name, &env).ok().expect("a global")
}

fn runtime_slot(
    engine: &mut Engine,
    name: &str,
    slot: fn(&crate::function_runtime::FunctionRuntime) -> Option<Rc<Chunk>>,
) -> Option<Rc<Chunk>> {
    let value = function(engine, name);
    let object = value.as_obj()?;
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        return None;
    };
    slot(&user.func.runtime())
}

fn first_stage(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    runtime_slot(engine, name, |runtime| {
        runtime.code.get().and_then(Option::clone)
    })
    .expect("its first stage")
}

fn second_stage(engine: &mut Engine, name: &str) -> Option<Rc<Chunk>> {
    runtime_slot(engine, name, |runtime| {
        runtime.code2.get().and_then(Option::clone)
    })
}

/// Call `name` with `x` from Rust: a native entry, whose prologue polls.
fn call(engine: &mut Engine, name: &str, x: f64) -> f64 {
    let callee = function(engine, name);
    match engine
        .interp
        .call(callee, Value::Undefined, &[Value::Num(x)])
        .ok()
        .expect("the call returns")
        .view()
    {
        crate::value::ValueView::Num(n) => n,
        _ => panic!("a Number"),
    }
}

/// Wait until the compile thread finished a compile of this engine (its install bit set).
fn wait_finished(engine: &Engine) {
    let start = std::time::Instant::now();
    while !engine.interp.poll.install_requested() {
        assert!(
            start.elapsed() < std::time::Duration::from_secs(60),
            "the compile thread finishes the compile"
        );
        std::thread::sleep(std::time::Duration::from_micros(200));
    }
}

/// Call `name` until its checkpoint queues its compile: the calls' count. The compile thread is
/// held meanwhile (`compiler::TEST_HOLD`), so that the compile stays pending until the caller
/// lets it go (the guard returned) however the threads are scheduled.
fn until_pending(engine: &mut Engine, name: &str) -> (usize, std::sync::MutexGuard<'static, ()>) {
    let hold = super::compiler::TEST_HOLD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for calls in 1..=20_000 {
        call(engine, name, calls as f64);
        if engine.interp.opt_pending.len() != 0 {
            return (calls, hold);
        }
    }
    panic!("no compile of {name} was queued");
}

const SQ: &str = "function sq(x) { return x * x + 1; }";

#[test]
fn a_function_gets_the_tiers_code_at_the_poll_after_its_compile_ends() {
    if !asynchronous_here() {
        return;
    }
    let mut engine = jit_engine();
    compiled();
    eval(&mut engine, SQ);
    let (calls, hold) = until_pending(&mut engine, "sq");
    assert!(calls <= 200, "the first checkpoint asks ({calls} calls)");
    let first = first_stage(&mut engine, "sq");
    // Pending: the chunk is not published, the first stage runs its template code, marked
    // compiling with no checkpoint armed.
    assert!(second_stage(&mut engine, "sq").is_none());
    assert!(ChunkProfile::get(&first).is_some_and(ChunkProfile::compiling));
    assert_eq!(first.inline_retry_at.get(), 0);
    assert!(compiled().is_empty(), "nothing installed before its poll");
    // Another call runs the template code meanwhile.
    assert_eq!(call(&mut engine, "sq", 4.0), 17.0);
    assert!(second_stage(&mut engine, "sq").is_none());
    drop(hold);
    wait_finished(&engine);
    // Finished, but no poll yet: still not installed.
    assert!(second_stage(&mut engine, "sq").is_none());
    assert_eq!(engine.interp.opt_pending.len(), 1);
    // The next entry's poll installs it.
    assert_eq!(call(&mut engine, "sq", 3.0), 10.0);
    assert_eq!(engine.interp.opt_pending.len(), 0);
    let chunk = second_stage(&mut engine, "sq").expect("published at the install");
    assert_eq!(compiled(), vec![Rc::as_ptr(&chunk) as usize]);
    assert!(!ChunkProfile::get(&first).is_some_and(ChunkProfile::compiling));
    let code = chunk.jit.get().flatten().expect("its code is resident");
    assert!(
        super::exit::table_bytes(code.mem_ptr()) != 0,
        "the tier's code, not the template JIT's"
    );
    // It runs, and results hold.
    code.referenced.set(0);
    assert_eq!(
        eval(
            &mut engine,
            "let t = 0; for (let k = 0; k < 1000; k++) t += sq(k % 7); t"
        ),
        "13977"
    );
    assert!(code.referenced.get() != 0, "the tier's code ran");
}

#[test]
fn a_result_whose_sites_a_discard_flagged_meanwhile_is_dropped_and_the_function_tiers_up_again() {
    if !asynchronous_here() {
        return;
    }
    let mut engine = jit_engine();
    compiled();
    dropped();
    eval(&mut engine, SQ);
    let (_, hold) = until_pending(&mut engine, "sq");
    let first = first_stage(&mut engine, "sq");
    // A discard flags a site of the first stage after the snapshot.
    ChunkProfile::of(&first).add_site(0, ExitReason::NotNumber as u8, false, 0, false);
    drop(hold);
    wait_finished(&engine);
    call(&mut engine, "sq", 2.0);
    assert_eq!(engine.interp.opt_pending.len(), 0);
    let drops = dropped();
    assert_eq!(drops.len(), 1, "{drops:?}");
    assert_eq!(drops[0].1, "a discard flagged its sites since its snapshot");
    assert!(compiled().is_empty());
    assert!(second_stage(&mut engine, "sq").is_none());
    // It may tier up again: its checkpoint armed again, a later one compiles it.
    assert!(!ChunkProfile::get(&first).is_some_and(ChunkProfile::compiling));
    assert_ne!(first.inline_retry_at.get(), 0, "a checkpoint armed again");
    drop(until_pending(&mut engine, "sq"));
    wait_finished(&engine);
    call(&mut engine, "sq", 2.0);
    let chunk = second_stage(&mut engine, "sq").expect("the second compile installs");
    assert_eq!(compiled(), vec![Rc::as_ptr(&chunk) as usize]);
    assert!(dropped().is_empty());
}

#[test]
fn a_result_after_a_recompile_was_published_meanwhile_is_dropped() {
    if !asynchronous_here() {
        return;
    }
    let mut engine = jit_engine();
    compiled();
    dropped();
    eval(&mut engine, SQ);
    let (_, hold) = until_pending(&mut engine, "sq");
    let first = first_stage(&mut engine, "sq");
    // Another recompile takes the function's second stage meanwhile (the first stage itself, as
    // a stand-in): the pending chunk is no longer the one to install.
    {
        let value = function(&mut engine, "sq");
        let object = value.as_obj().unwrap();
        let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
            unreachable!()
        };
        let runtime = user.func.runtime();
        let genv = crate::interpreter::Env::as_ptr(&engine.interp.global_env) as usize;
        assert!(runtime.publish(&runtime.code2, first.clone(), genv));
    }
    drop(hold);
    wait_finished(&engine);
    call(&mut engine, "sq", 2.0);
    let drops = dropped();
    assert_eq!(drops.len(), 1, "{drops:?}");
    assert_eq!(drops[0].1, "a recompile was published meanwhile");
    assert!(compiled().is_empty());
    assert!(Rc::ptr_eq(
        &second_stage(&mut engine, "sq").unwrap(),
        &first
    ));
}

#[test]
fn a_result_whose_chunk_was_superseded_meanwhile_is_dropped() {
    if !asynchronous_here() {
        return;
    }
    let mut engine = jit_engine();
    compiled();
    dropped();
    eval(&mut engine, SQ);
    let (_, hold) = until_pending(&mut engine, "sq");
    let first = first_stage(&mut engine, "sq");
    // Another compile of the function supersedes the first stage the compile started from.
    {
        let value = function(&mut engine, "sq");
        let object = value.as_obj().unwrap();
        let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
            unreachable!()
        };
        let other = crate::bytecode::compile(&user.func).expect("a chunk");
        let runtime = user.func.runtime();
        let genv = crate::interpreter::Env::as_ptr(&engine.interp.global_env) as usize;
        assert!(runtime.publish(&runtime.code2, other, genv));
    }
    assert!(first.superseded());
    drop(hold);
    wait_finished(&engine);
    call(&mut engine, "sq", 2.0);
    let drops = dropped();
    assert_eq!(drops.len(), 1, "{drops:?}");
    assert_eq!(drops[0].1, "its chunk was superseded");
    assert!(compiled().is_empty());
}

#[test]
fn a_result_whose_function_was_collected_meanwhile_is_dropped() {
    if !asynchronous_here() {
        return;
    }
    let mut engine = jit_engine();
    compiled();
    dropped();
    eval(
        &mut engine,
        "var fresh = new Function('x', 'return x * x + 2;');",
    );
    let (_, hold) = until_pending(&mut engine, "fresh");
    // The function's only closure goes, and a collection frees its function.
    eval(&mut engine, "fresh = undefined; 0");
    engine.interp.gc_collect();
    engine.interp.gc_collect();
    drop(hold);
    wait_finished(&engine);
    // Any poll installs: a loop's back edges.
    eval(
        &mut engine,
        "let t = 0; for (let k = 0; k < 100; k++) t += k; t",
    );
    assert_eq!(engine.interp.opt_pending.len(), 0);
    let drops = dropped();
    assert_eq!(drops.len(), 1, "{drops:?}");
    assert_eq!(drops[0].1, "its function was collected");
    assert!(compiled().is_empty());
}

#[test]
fn an_engine_dropped_with_compiles_pending_cancels_them() {
    use std::sync::atomic::Ordering;
    if !asynchronous_here() {
        return;
    }
    let cancelled = super::compiler::TEST_CANCELLED.load(Ordering::Relaxed);
    for _ in 0..4 {
        let mut engine = jit_engine();
        eval(&mut engine, SQ);
        let (_, hold) = until_pending(&mut engine, "sq");
        drop(engine);
        drop(hold);
    }
    // Each is skipped, or its result dropped once compiled.
    let start = std::time::Instant::now();
    while super::compiler::TEST_CANCELLED.load(Ordering::Relaxed) < cancelled + 4 {
        assert!(
            start.elapsed() < std::time::Duration::from_secs(60),
            "the compile thread cancels the dropped engines' jobs"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    // The compile thread goes on serving the engines after them.
    let mut engine = jit_engine();
    compiled();
    eval(&mut engine, SQ);
    drop(until_pending(&mut engine, "sq"));
    wait_finished(&engine);
    call(&mut engine, "sq", 2.0);
    let chunk = second_stage(&mut engine, "sq").expect("installed");
    assert_eq!(compiled(), vec![Rc::as_ptr(&chunk) as usize]);
}

#[test]
fn results_hold_with_the_tier_compiling_off_the_main_thread() {
    // Programs whose functions tier up while their callers run, and are installed at the polls
    // of the code that runs meanwhile, with forced exits too.
    for exit_every in [0, 7] {
        let mut switches = asynchronous();
        switches.exit_every = exit_every;
        switches.set();
        let mut engine = jit_engine();
        let result = eval(
            &mut engine,
            "function P(x, y) { this.x = x; this.y = y; }\n\
             function len2(p) { return p.x * p.x + p.y * p.y; }\n\
             function sum(n) { let s = 0; for (let i = 0; i < n; i++) s = s + i; return s; }\n\
             let t = 0; for (let k = 0; k < 20000; k++) { t = (t + len2(new P(k % 13, k % 7)) \
             + sum(k % 30)) % 1000003; } t",
        );
        // The VM's answer.
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        let expected = eval(
            &mut vm,
            "function P(x, y) { this.x = x; this.y = y; }\n\
             function len2(p) { return p.x * p.x + p.y * p.y; }\n\
             function sum(n) { let s = 0; for (let i = 0; i < n; i++) s = s + i; return s; }\n\
             let t = 0; for (let k = 0; k < 20000; k++) { t = (t + len2(new P(k % 13, k % 7)) \
             + sum(k % 30)) % 1000003; } t",
        );
        assert_eq!(result, expected, "exits every {exit_every}");
    }
}

#[test]
#[cfg(target_os = "linux")]
fn one_cpu_compiles_on_the_main_thread() {
    // A child process confined to one CPU before it asks how many it has.
    const CHILD: &str = "TRUST_JS_OPT_ONE_CPU_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "opt::offthread_tests::one_cpu_compiles_on_the_main_thread",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    extern "C" {
        fn sched_getcpu() -> i32;
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
    }
    let mut mask = [0u64; 16];
    let cpu = unsafe { sched_getcpu() }.max(0) as usize;
    mask[cpu / 64] |= 1 << (cpu % 64);
    assert_eq!(
        unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) },
        0
    );
    assert_eq!(
        std::thread::available_parallelism().unwrap().get(),
        1,
        "the affinity mask holds one CPU"
    );
    asynchronous().set();
    assert!(
        !super::asynchronous(),
        "one CPU compiles on the main thread"
    );
    let mut engine = jit_engine();
    compiled();
    eval(&mut engine, SQ);
    for k in 0..200 {
        call(&mut engine, "sq", f64::from(k));
    }
    assert_eq!(engine.interp.opt_pending.len(), 0, "nothing ran off it");
    let chunk = second_stage(&mut engine, "sq").expect("compiled and published at once");
    assert_eq!(compiled(), vec![Rc::as_ptr(&chunk) as usize]);
}
