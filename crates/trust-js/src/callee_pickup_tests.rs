//! A function's new code picked up per callee: a publication supersedes the chunk it replaces
//! (`Chunk::supersede`), whose callers refill their ways at their next entry, while every other
//! call site keeps its ways and the call caches' epoch stays where it is (design note, plan
//! "### A function's new code picked up per callee, not through the global call-cache epoch").

use crate::bytecode::*;
use crate::interpreter::{Env, Interp};
use crate::value::{ObjectKind, Property, Value, ValueView};
use crate::{Completion, Engine};
use std::rc::Rc;
use std::sync::atomic::Ordering::Relaxed;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// The global user function `name` of the active Realm: its identity (what the ways record)
/// and its function.
fn function(i: &mut Interp, name: &str) -> (usize, Rc<crate::ast::Function>) {
    let env = i.global_env;
    let value: Value = i.get_var(name, &env).ok().expect("bound name");
    let ValueView::Obj(object) = value.view() else {
        panic!("{name} is not an object")
    };
    let key = crate::value::Gc::as_ptr(&object) as usize;
    let ObjectKind::User(user) = &object.borrow().kind else {
        panic!("{name} is not a user function")
    };
    (key, Rc::clone(&user.func))
}

/// `func`'s first-stage chunk and, once published, its second stage.
fn stages(func: &crate::ast::Function) -> (Rc<Chunk>, Option<Rc<Chunk>>) {
    let runtime = func.runtime();
    let first = runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .cloned()
        .expect("a compiled function");
    let second = runtime.code2.get().and_then(Option::as_ref).cloned();
    (first, second)
}

/// The way of `chunk`'s call sites that holds the callee `key`.
fn way(chunk: &Chunk, key: usize) -> Option<CallIc> {
    chunk
        .call_caches
        .iter()
        .flat_map(|site| site.entries.iter())
        .map(std::cell::Cell::get)
        .find(|ic| ic.callee == key)
}

fn engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

/// Σ k² + 1 for k < n, what `caller(n)` returns below.
fn squares(n: u64) -> String {
    (0..n).map(|k| k * k + 1).sum::<u64>().to_string()
}

const LEAF: &str = r#"
    function sq(x) { return x * x; }
    function leaf(x) { return sq(x) + 1; }
    function caller(n) { var t = 0; for (var k = 0; k < n; k++) t += leaf(k); return t; }
    function other(x) { return x * 2; }
    function bystander(n) { var t = 0; for (var k = 0; k < n; k++) t += other(k); return t; }
    bystander(40) + caller(40)"#;

/// A caller of a recompiled function enters its new code through the way it had, at the next
/// call after the publication; the way of a site that does not call the function, and the call
/// caches' epoch, stay as they were.
#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_callers_way_picks_up_its_callees_new_code_and_no_other_site_refills() {
    // The callee's new code is published at its checkpoint, which this test reads at once.
    crate::opt::pin_synchronous_for_test();
    // CALL_IC_EPOCH is process-wide: a test beside this one that drops a function's runtime or
    // reclaims code moves it, and every way refills. Check a run that ended at its start's epoch,
    // read before the setup fills the bystander's way.
    for attempt in 0..200 {
        let start = CALL_IC_EPOCH.load(Relaxed);
        let mut engine = engine();
        eval(&mut engine, LEAF);
        let i = &mut *engine.interp;
        let (leaf_key, leaf) = function(i, "leaf");
        let (other_key, _) = function(i, "other");
        let (_, caller) = function(i, "caller");
        let (_, bystander) = function(i, "bystander");
        let (first, second) = stages(&leaf);
        assert!(second.is_none(), "leaf's first checkpoint is later");
        let caller_chunk = stages(&caller).0;
        let bystander_chunk = stages(&bystander).0;
        let before = way(&bystander_chunk, other_key).expect("bystander's way");
        // leaf's checkpoint (`inline_recompile_at` runs) publishes its second stage, which
        // splices sq; the caller's later calls take it.
        let mut runs = 0;
        while stages(&leaf).1.is_none() && runs < 100 {
            assert_eq!(eval(&mut engine, "caller(50)"), squares(50));
            runs += 1;
        }
        let second = stages(&leaf).1.expect("leaf's second stage");
        assert!(first.superseded(), "the replaced chunk is superseded");
        assert!(!second.superseded());
        for _ in 0..4 {
            assert_eq!(eval(&mut engine, "caller(50)"), squares(50));
        }
        let picked = way(&caller_chunk, leaf_key).expect("caller's way");
        assert!(
            std::ptr::eq(picked.chunk_raw, Rc::as_ptr(&second)),
            "the caller's way holds the new code"
        );
        assert!(second.jit_runs.get() > 0, "the new code ran natively");
        assert_eq!(eval(&mut engine, "bystander(40)"), "1560");
        let after = way(&bystander_chunk, other_key).expect("bystander's way");
        if CALL_IC_EPOCH.load(Relaxed) != start && attempt < 199 {
            continue;
        }
        assert_eq!(CALL_IC_EPOCH.load(Relaxed), start, "no epoch move");
        assert_eq!(picked.epoch, start, "the refill kept the epoch");
        assert_eq!(
            (after.epoch, after.chunk_raw, after.code, after.code_mem),
            (before.epoch, before.chunk_raw, before.code, before.code_mem),
            "the bystander's way did not refill"
        );
        break;
    }
}

/// A `new` site picks up its constructor's new code: the construct caches miss on a superseded
/// chunk, and the fill gives them the current one.
#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_construct_site_picks_up_its_constructors_new_code() {
    // The constructor's new code is published at its checkpoint, and the test reads `make`'s first
    // stage's site: off the main thread `make`'s own second stage could land first, and its site,
    // not this one, would take the new code.
    crate::opt::pin_synchronous_for_test();
    let mut engine = engine();
    let sum = |n: u64| (0..n).map(|k| k * k).sum::<u64>().to_string();
    assert_eq!(
        eval(
            &mut engine,
            "function sq(x) { return x * x; }
             function P(x) { this.v = sq(x); }
             function make(n) { var t = 0; for (var k = 0; k < n; k++) t += new P(k).v; return t; }
             make(40)"
        ),
        sum(40)
    );
    let i = &mut *engine.interp;
    let (p_key, p) = function(i, "P");
    let (_, make) = function(i, "make");
    let first = stages(&p).0;
    // With the optimizing tier on, the second stage waits for settled feedback, at a later
    // checkpoint (nothing forces it: P3 stage 2, step 11).
    let mut runs = 0;
    while stages(&p).1.is_none() && runs < 200 {
        assert_eq!(eval(&mut engine, "make(50)"), sum(50));
        runs += 1;
    }
    let second = stages(&p).1.expect("P's second stage");
    assert!(first.superseded());
    for _ in 0..4 {
        assert_eq!(eval(&mut engine, "make(50)"), sum(50));
    }
    let make_chunk = stages(&make).0;
    let cache = make_chunk
        .jit_ops()
        .iter()
        .find_map(|op| match op {
            Op::New(_, cache) => Some(*cache),
            _ => None,
        })
        .expect("make's `new` site");
    let entry = make_chunk.construct_cache(cache).call;
    assert_eq!(entry.callee, p_key);
    assert!(
        std::ptr::eq(entry.chunk_raw, Rc::as_ptr(&second)),
        "the construct site's entry holds the new code"
    );
}

/// A frame running the replaced chunk finishes in it, and its caller's next call enters the
/// new chunk.
#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_frame_running_the_replaced_chunk_finishes_in_it() {
    // It publishes `long`'s second stage itself: with the tier on, its warm-up's loop iterations
    // reach the checkpoint that builds one first (P3 stage 2, step 8).
    crate::opt::pin_tier_off_for_test();
    /// Publish a fresh compile of `long` as its new code, as a recompile would.
    fn republish(i: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
        let (_, long) = function(i, "long");
        let runtime = long.runtime();
        let fresh = crate::bytecode::compile(&long).expect("recompile");
        let genv = Env::as_ptr(&i.global_env) as usize;
        assert!(runtime.publish(&runtime.code2, fresh, genv));
        Ok(Value::Undefined)
    }
    let mut engine = engine();
    let hook = engine.interp.make_native("republish", 0, republish);
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("republish", Property::plain(Value::Obj(hook)));
    let sum = |n: u64| (0..n).map(|k| k * k).sum::<u64>().to_string();
    eval(
        &mut engine,
        "function sq(x) { return x * x; }
         var once = false;
         function long(n) {
             var t = 0;
             for (var k = 0; k < n; k++) { if (once && k === 10) { once = false; republish(); } t += sq(k); }
             return t;
         }
         function caller(n) { return long(n); }
         for (var w = 0; w < 20; w++) caller(30);",
    );
    let (_, long) = function(&mut engine.interp, "long");
    let first = stages(&long).0;
    assert!(first.jit.get().flatten().is_some(), "long runs natively");
    // The publication lands while a frame runs `first`, ten iterations in.
    assert_eq!(eval(&mut engine, "once = true; caller(400)"), sum(400));
    let second = stages(&long).1.expect("the published chunk");
    assert!(first.superseded());
    assert_eq!(second.jit_runs.get(), 0);
    for _ in 0..8 {
        assert_eq!(eval(&mut engine, "caller(30)"), sum(30));
    }
    assert!(
        second.jit.get().flatten().is_some() && second.jit_runs.get() > 0,
        "the caller's later calls enter the new chunk"
    );
    let (_, caller) = function(&mut engine.interp, "caller");
    let (long_key, _) = function(&mut engine.interp, "long");
    let picked = way(&stages(&caller).0, long_key).expect("caller's way");
    assert!(std::ptr::eq(picked.chunk_raw, Rc::as_ptr(&second)));
}

/// Rust's callers pick up new code too: a Map's forEach calls its callback through a prepared
/// entry, which a superseded chunk makes stale mid-iteration, and a Proxy's trap through its
/// call cache.
#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn rust_callers_pick_up_new_code() {
    let mut engine = engine();
    eval(
        &mut engine,
        "function sq(x) { return x * x; }
         var acc = 0;
         function cb(v) { acc += sq(v); }
         var m = new Map(); for (var k = 0; k < 400; k++) m.set(k, k);
         function each() { acc = 0; m.forEach(cb); return acc; }
         var handler = { get: function (t, key) { return sq(key.length); } };
         var p = new Proxy({}, handler);
         function reads(n) { var t = 0; for (var k = 0; k < n; k++) t += p.abc; return t; }",
    );
    let sum: u64 = (0..400u64).map(|k| k * k).sum();
    // One forEach: cb's checkpoint publishes its second stage after its hundredth call, and the
    // rest of the iteration's calls enter it. With the optimizing tier on, the second stage
    // waits for settled feedback (a later checkpoint, nothing forced), which a later forEach
    // reaches as far into its iteration.
    let (_, cb) = function(&mut engine.interp, "cb");
    for _ in 0..25 {
        assert_eq!(eval(&mut engine, "each()"), sum.to_string());
        if !crate::opt::enabled() || stages(&cb).1.is_some() {
            break;
        }
    }
    let (first, second) = stages(&cb);
    let second = second.expect("cb's second stage");
    assert!(first.superseded());
    assert!(
        second.jit_runs.get() >= 100,
        "the forEach's later calls entered the new chunk ({} runs)",
        second.jit_runs.get()
    );
    // The trap: its calls through the read's cache, from machine code or from Rust, in as many
    // rounds as its second stage waits for.
    let env = engine.interp.global_env;
    let handler = engine
        .interp
        .get_var("handler", &env)
        .ok()
        .expect("handler");
    let trap = engine
        .interp
        .get_member_key(&handler, "get")
        .ok()
        .expect("the trap");
    let ValueView::Obj(trap) = trap.view() else {
        panic!("the trap is an object")
    };
    let trap = {
        let ObjectKind::User(user) = &trap.borrow().kind else {
            panic!("the trap is a user function")
        };
        Rc::clone(&user.func)
    };
    for _ in 0..25 {
        assert_eq!(eval(&mut engine, "reads(400)"), "3600");
        if !crate::opt::enabled() || stages(&trap).1.is_some() {
            break;
        }
    }
    let (first, second) = stages(&trap);
    let second = second.expect("the trap's second stage");
    assert!(first.superseded());
    let runs = entries(&second);
    assert_eq!(eval(&mut engine, "reads(50)"), "450");
    assert!(
        entries(&second) > runs,
        "the trap's later calls entered the new chunk"
    );
}

/// The entries into `chunk` its counters see: through Rust and the call helpers (`jit_runs`)
/// and, on ARM64, through the direct sequence, its direct returns and throws. A second stage is
/// built settled, so the direct sequence enters it from its first call and counts no run of it.
fn entries(chunk: &Chunk) -> usize {
    #[cfg(target_arch = "aarch64")]
    let direct = chunk.test_direct_returns.get() + chunk.test_direct_throws.get();
    #[cfg(not(target_arch = "aarch64"))]
    let direct = 0;
    chunk.jit_runs.get() as usize + direct
}

/// A megamorphic site's overflow entries stay through its callees' publications: each callee's
/// entry refills in place, without an admission, and holds its new code.
#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_megamorphic_site_keeps_its_overflow_entries() {
    // The callees publish their second stages while `wide` runs its first stage, whose site this
    // reads: `wide`'s loop counts nothing toward its own checkpoints (P3 stage 2, step 8).
    crate::opt::pin_loop_count_off_for_test();
    let mut engine = engine();
    eval(
        &mut engine,
        "function sq(x) { return x * x; }
         var twelve = []; for (var q = 0; q < 12; q++) twelve.push(new Function('a', 'return sq(a) - ' + q));
         function wide(n) { var s = 0; for (var k = 0; k < n; k++) s += twelve[k % 12](k); return s; }
         wide(240)",
    );
    let callees: Vec<(usize, Rc<crate::ast::Function>)> = (0..12)
        .map(|q| {
            eval(&mut engine, &format!("var probedCallee = twelve[{q}];"));
            function(&mut engine.interp, "probedCallee")
        })
        .collect();
    // Each callee's checkpoint publishes its second stage after its hundredth call.
    let mut runs = 0;
    while callees.iter().any(|(_, f)| stages(f).1.is_none()) && runs < 100 {
        eval(&mut engine, "wide(240)");
        runs += 1;
    }
    // CALL_IC_EPOCH is process-wide: a test running beside this one that recompiles a chunk or
    // drops its runtime moves it, which stales every way and overflow entry until calls refill
    // them. The entries are checked as runs that ended at the epoch they started at left them.
    let mut attempts = 0;
    let epoch = loop {
        let epoch = CALL_IC_EPOCH.load(Relaxed);
        for _ in 0..4 {
            eval(&mut engine, "wide(240)");
        }
        attempts += 1;
        if CALL_IC_EPOCH.load(Relaxed) == epoch || attempts == 200 {
            break epoch;
        }
    };
    let (_, wide) = function(&mut engine.interp, "wide");
    let chunk = stages(&wide).0;
    let site = (0..chunk.call_caches.len())
        .find(|&site| chunk.call_caches[site].megamorphic())
        .expect("wide's megamorphic site");
    let genv = Env::as_ptr(&engine.interp.global_env) as usize;
    let mut in_overflow = 0;
    for (q, (key, func)) in callees.iter().enumerate() {
        let (first, second) = stages(func);
        let second = second.unwrap_or_else(|| panic!("twelve[{q}]'s second stage"));
        assert!(first.superseded());
        let entry = chunk.call_caches[site]
            .entries
            .iter()
            .map(std::cell::Cell::get)
            .find(|way| way.callee == *key && way.epoch == epoch)
            .or_else(|| {
                in_overflow += 1;
                engine.interp.call_overflow.lookup(*key, genv, epoch)
            })
            .unwrap_or_else(|| panic!("twelve[{q}] is not cached"));
        assert!(
            std::ptr::eq(entry.chunk_raw, Rc::as_ptr(&second)),
            "twelve[{q}]'s entry holds its new code"
        );
    }
    assert!(in_overflow >= 8, "the ways hold four callees at most");
    assert!(
        engine.interp.call_overflow.occupied() <= 12,
        "the refills replaced their entries ({} occupied)",
        engine.interp.call_overflow.occupied()
    );
}

/// A polymorphic site's callees get new code: a callee in the site's ways refills its way, and
/// one in the overflow table refills its entry there, evicting no way (a callee's new code is
/// not a new callee). The site stays polymorphic with its ways, and the caller's own second
/// stage splices the callees they hold, with the optimizing tier on as with it off.
#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_polymorphic_sites_callees_get_new_code_and_it_keeps_its_ways() {
    // Six callees through four ways: two live in the overflow table. Each calls a helper, so its
    // checkpoint publishes a second stage (with the tier on, past its settled-feedback wait).
    let configurations = [
        ("tier off", false, false),
        #[cfg(target_arch = "aarch64")]
        ("tier on", true, false),
        #[cfg(target_arch = "aarch64")]
        ("tier eager", true, true),
    ];
    let mut splices = None;
    for (label, on, eager) in configurations {
        // The callees' checkpoints come first: `stable`'s loop counts nothing toward its own. With
        // the counting (P3 stage 2, step 8) it gets six runs for each of a callee's, and its
        // second stage, built first, splices the four in its ways, which then count no entries.
        #[cfg(target_arch = "aarch64")]
        crate::opt::TestSwitches {
            on,
            eager,
            exit_every: 0,
            no_loop_count: true,
            ..Default::default()
        }
        .set();
        #[cfg(not(target_arch = "aarch64"))]
        let _ = (on, eager);
        let mut engine = engine();
        eval(
            &mut engine,
            "function h(a) { return a; }
             var six = []; for (var q = 0; q < 6; q++) six.push(new Function('a', 'return h(a) * ' + q));
             function stable(n) { var s = 0; for (var k = 0; k < n; k++) s += six[k % 6](k); return s; }",
        );
        let callees: Vec<(usize, Rc<crate::ast::Function>)> = (0..6)
            .map(|q| {
                eval(&mut engine, &format!("var probedCallee = six[{q}];"));
                function(&mut engine.interp, "probedCallee")
            })
            .collect();
        // The callees past the checkpoint that publishes their second stages: the first, or with
        // the tier on the first whose feedback has settled (nothing forced).
        for _ in 0..80 {
            assert_eq!(eval(&mut engine, "stable(400)"), "199070", "{label}");
            if callees.iter().all(|(_, func)| stages(func).1.is_some()) {
                break;
            }
        }
        for (q, (_, func)) in callees.iter().enumerate() {
            let (first, second) = stages(func);
            assert!(second.is_some(), "{label}: six[{q}]'s second stage");
            assert!(first.superseded(), "{label}");
        }
        let (_, stable) = function(&mut engine.interp, "stable");
        let chunk = stages(&stable).0;
        let site = chunk
            .call_caches
            .iter()
            .find(|site| site.entries.iter().any(|way| way.get().callee != 0))
            .expect("stable's call site");
        assert!(
            !site.megamorphic(),
            "{label}: a callee's new code is no new callee"
        );
        let mut held: Vec<usize> = site.entries.iter().map(|way| way.get().callee).collect();
        held.sort_unstable();
        held.dedup();
        assert_eq!(held.len(), 4, "{label}: four callees in the ways");
        for (q, (key, _)) in callees.iter().enumerate() {
            assert!(
                held.contains(key) || engine.interp.call_overflow.holds(*key),
                "{label}: six[{q}] is cached"
            );
        }
        // `stable` past its own checkpoint: its second stage splices its ways' callees.
        for _ in 0..40 {
            eval(&mut engine, "for (var r = 0; r < 250; r++) stable(6);");
            if stages(&stable).1.is_some() {
                break;
            }
        }
        let second = stages(&stable)
            .1
            .unwrap_or_else(|| panic!("{label}: stable's second stage"));
        let guards = second
            .jit_ops()
            .iter()
            .filter(|op| matches!(op, Op::InlineGuard(..)))
            .count();
        assert!(guards >= 4, "{label}: {guards} splices");
        assert_eq!(*splices.get_or_insert(guards), guards, "{label}");
    }
}

/// A megamorphic site's admissions to the overflow table renew after its refusals, and at an
/// epoch move.
#[test]
fn a_megamorphic_sites_admissions_renew_after_its_refusals() {
    let site = CallSite::empty();
    for _ in 0..CALL_SITE_OVERFLOW_ADMISSIONS {
        assert!(site.admits_overflow(3));
    }
    for _ in 1..CALL_SITE_OVERFLOW_RENEWAL {
        assert!(!site.admits_overflow(3));
    }
    assert!(site.admits_overflow(3), "a new window");
    for _ in 1..CALL_SITE_OVERFLOW_ADMISSIONS {
        assert!(site.admits_overflow(3));
    }
    assert!(!site.admits_overflow(3));
    assert!(site.admits_overflow(4), "an epoch move renews the window");
}

/// A caller in another Realm that still selects the replaced chunk (a second stage planned in
/// the publishing Realm, `Chunk::inline_realm_compatible`) makes it current again once, with an
/// epoch move, and then calls it directly; the publishing Realm's callers keep the new code.
#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_caller_in_another_realm_keeps_the_replaced_chunk_after_one_fallback() {
    use crate::realm_inline_guard_tests::{eval_string, in_realm};
    const SOURCE: &str = r#"(function(){
        globalThis.leaf=function leaf(x){return x+3;};
        globalThis.subject=function subject(v){return leaf(v)+1;};
        globalThis.drive=function drive(n){var t=0;for(var k=0;k<n;k++)t+=subject(k);return t;};
    })();"#;
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot: &'static [u8] = SNAPSHOT.get_or_init(|| crate::compile_snapshot(SOURCE).unwrap());
    fn run(i: &mut Interp, snapshot: &'static [u8]) {
        let body = i
            .shared_snapshot_program(snapshot)
            .expect("snapshot decodes");
        assert!(i.run_classic_program(&body, None).is_ok());
    }
    // Σ k + 4 for k < 20.
    const DRIVE: &str = "270";
    let mut engine = engine();
    let i = &mut *engine.interp;
    run(i, snapshot);
    assert_eq!(eval_string(i, "drive(20)"), DRIVE);
    let (main_key, subject) = function(i, "subject");
    let (_, drive) = function(i, "drive");
    let realm = {
        let env = i.global_env;
        let value: Value = i.get_var("subject", &env).ok().expect("subject");
        let object = value.as_obj().expect("an object");
        let ObjectKind::User(user) = &object.borrow().kind else {
            panic!("a user function")
        };
        user.realm
    };
    let primary = stages(&subject).0;
    let genv = Env::as_ptr(&i.global_env) as usize;
    let plan = plan_inlines(&primary, &subject, realm, &i.global_env);
    assert!(!plan.is_empty(), "subject splices leaf");
    let inlined = compile_with_inlines(&subject, &plan, &primary).expect("subject's second stage");
    assert!(inlined.inline_realm_compatible(genv));
    let child = i.create_realm();
    let (child_key, child_genv) = in_realm(i, &child, |i| {
        run(i, snapshot);
        assert_eq!(eval_string(i, "drive(20)"), DRIVE);
        let (key, shared) = function(i, "subject");
        assert!(Rc::ptr_eq(&shared, &subject), "one AST, one runtime");
        (key, Env::as_ptr(&i.global_env) as usize)
    });
    assert!(!inlined.inline_realm_compatible(child_genv));
    let drive_chunk = stages(&drive).0;
    let child_way = way(&drive_chunk, child_key).expect("the child's way");
    assert!(std::ptr::eq(child_way.chunk_raw, Rc::as_ptr(&primary)));
    // The main Realm publishes: its callers move to the new chunk.
    let runtime = subject.runtime();
    assert!(runtime.publish(&runtime.code2, inlined.clone(), genv));
    assert!(primary.superseded());
    for _ in 0..4 {
        assert_eq!(eval_string(i, "drive(20)"), DRIVE);
    }
    let main_way = way(&drive_chunk, main_key).expect("the main Realm's way");
    assert!(std::ptr::eq(main_way.chunk_raw, Rc::as_ptr(&inlined)));
    // The child still selects the first stage: its next fill makes it current again.
    let before = CALL_IC_EPOCH.load(Relaxed);
    assert_eq!(in_realm(i, &child, |i| eval_string(i, "drive(20)")), DRIVE);
    assert!(!primary.superseded(), "current again for the child");
    assert_ne!(
        CALL_IC_EPOCH.load(Relaxed),
        before,
        "the fallback moved the epoch"
    );
    for _ in 0..4 {
        assert_eq!(in_realm(i, &child, |i| eval_string(i, "drive(20)")), DRIVE);
        assert_eq!(eval_string(i, "drive(20)"), DRIVE);
    }
    assert!(!primary.superseded(), "no second fallback");
    let child_way = way(&drive_chunk, child_key).expect("the child's way");
    assert!(std::ptr::eq(child_way.chunk_raw, Rc::as_ptr(&primary)));
    let main_way = way(&drive_chunk, main_key).expect("the main Realm's way");
    assert!(std::ptr::eq(main_way.chunk_raw, Rc::as_ptr(&inlined)));
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    {
        // The child's calls enter the first stage directly again.
        let direct = primary.test_direct_returns.get();
        assert_eq!(in_realm(i, &child, |i| eval_string(i, "drive(20)")), DRIVE);
        assert!(primary.test_direct_returns.get() > direct, "direct entries");
    }
}
