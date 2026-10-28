//! Native residency preserves ECMA-262 §9.4 and [[Call]] context/return state.
//! Local official snapshot e28783d5fc9dc12b3de905961e2c71410b38a202.

use super::*;
use crate::bytecode::Tier;
use crate::value::Callable;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("cache fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn chunk(i: &mut Interp, name: &str) -> Rc<Chunk> {
    let env = i.global_env;
    let value = i.get_var(name, &env).ok().expect("fixture function");
    let object_handle = value.as_obj().expect("function object");
    let object = object_handle.borrow();
    let Callable::User(user) = &object.call else {
        panic!("user function")
    };
    user.func
        .runtime()
        .execution_code()
        .and_then(Option::as_ref)
        .expect("fixture bytecode")
        .clone()
}

fn engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

#[test]
fn inactive_code_reclaims_but_rust_leases_and_native_marks_do_not() {
    let mut e = engine();
    eval(&mut e, "function cold(x){return x+2;}cold(1);");
    let c = chunk(&mut e.interp, "cold");
    let lease = c.jit.get().flatten().expect("native cold function");
    let weak = Rc::downgrade(&lease);
    cache::reclaim(usize::MAX);
    assert!(
        c.jit.get().flatten().is_some(),
        "owned Rust lease protects code"
    );
    // A live activation record running the code (as a direct call's does, holding no lease)
    // protects it.
    let record = crate::jit::JitFrame::alloc(&mut e.interp);
    unsafe { (*record.as_ptr()).ctx.code_base = lease.mem };
    drop(lease);
    cache::reclaim(usize::MAX);
    assert!(
        c.jit.get().flatten().is_some(),
        "native return PC protects code"
    );
    // SAFETY: the record came from this interpreter's pool and runs nothing.
    unsafe { crate::jit::JitFrame::release(&mut e.interp, record) };
    let before = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
    cache::reclaim(usize::MAX);
    assert!(
        c.jit.get().is_none(),
        "inactive slot is retryable, not unavailable"
    );
    assert!(
        weak.upgrade().is_none(),
        "the actual mapping owner was released"
    );
    assert_ne!(
        crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
        before
    );
    assert_eq!(
        eval(&mut e, "var out;for(var n=0;n<40;n++)out=cold(n);out;"),
        "41"
    );
    assert!(
        c.jit.get().flatten().is_some(),
        "later real use recompiles code"
    );
}

fn reclaim_from_native(i: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    for name in ["activeParent", "activeLeaf"] {
        let c = chunk(i, name);
        let code = c
            .jit
            .get()
            .flatten()
            .expect("active function stays resident");
        assert!(
            code.test_residency_marks(false).0 > 0,
            "actual native entry for {name}"
        );
    }
    // No test Rc leases are held here. In particular a direct callee is protected
    // by its activation record, not by an accidental test-owned strong handle.
    cache::reclaim(usize::MAX);
    i.gc_collect();
    Ok(Value::Undefined)
}

#[test]
fn direct_return_and_throw_survive_reentrant_reclamation() {
    const CHILD: &str = "TRUST_JS_TEST_DIRECT_RECLAIM_CHILD";
    let Ok(mode) = std::env::var(CHILD) else {
        // CALL_IC_EPOCH is process-wide: unrelated parallel tests can invalidate
        // the sole normal-return probe between warming and entry. Isolate its
        // coverage assertion, and separately require the SAME observable result
        // when an explicit invalidation forces the checked-call fallback.
        for mode in ["stable", "stale"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "jit::cache_tests::direct_return_and_throw_survive_reentrant_reclamation",
                    "--nocapture",
                ])
                .env(CHILD, mode)
                .env("TRUST_JS_INLINE_AT", "100")
                .env_remove("TRUST_JS_JIT_NO_DIRECT_CALLS")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{mode}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return;
    };
    assert!(matches!(mode.as_str(), "stable" | "stale"));
    let mut e = engine();
    let global = e.interp.global;
    e.interp
        .def_method(&global, "reclaimNow", 0, reclaim_from_native);
    eval(
        &mut e,
        r#"
        function activeLeaf(x,force){if(force)reclaimNow();if(x<0)throw x;return {x:x+1};}
        function activeParent(x,force){try{return activeLeaf(x,force).x+1;}
            catch(e){return 'caught:'+e;}finally{ticks++;}}
        function victim(x){return x*2;}
        var ticks=0;for(var k=0;k<20;k++)activeParent(k,false);victim(3);
    "#,
    );
    for name in ["activeParent", "activeLeaf"] {
        let c = chunk(&mut e.interp, name);
        c.inline_attempted.set(true);
        c.inline_retry_at.set(0);
    }
    let victim = chunk(&mut e.interp, "victim");
    assert!(victim.jit.get().flatten().is_some());
    let before = cache::eviction_count();
    #[cfg(target_arch = "aarch64")]
    TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
    if mode == "stale" {
        crate::bytecode::invalidate_call_caches();
    }
    assert_eq!(
        eval(
            &mut e,
            "activeParent(40,true)+':'+activeParent(-3,true)+':'+ticks;"
        ),
        "42:caught:-3:22"
    );
    assert!(cache::eviction_count() > before);
    assert!(victim.jit.get().is_none());
    for name in ["activeParent", "activeLeaf"] {
        assert_eq!(
            chunk(&mut e.interp, name)
                .jit
                .get()
                .flatten()
                .unwrap()
                .test_residency_marks(false)
                .0,
            0
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        let returns = TEST_DIRECT_PACKED_RETURNS.with(|count| count.get());
        if mode == "stable" {
            assert!(
                returns > 0,
                "must exercise a real direct native return, not only Rust-leased entries"
            );
        } else {
            assert_eq!(returns, 0, "stale call caches must use the checked path");
        }
    }
}

#[test]
fn suspended_continuation_resumes_after_its_mapping_is_reclaimed() {
    let mut e = engine();
    assert_eq!(
        eval(
            &mut e,
            r#"
        var log=[];function* suspended(x){try{log.push('a');x+=yield x;
            log.push('b');return x;}finally{log.push('f');}}
        var it=suspended(3);it.next().value;
    "#
        ),
        "3"
    );
    let c = chunk(&mut e.interp, "suspended");
    let code = c.jit.get().flatten().expect("first slice native");
    assert_eq!(code.test_residency_marks(false).0, 0);
    let weak = Rc::downgrade(&code);
    drop(code);
    cache::reclaim(usize::MAX);
    assert!(
        weak.upgrade().is_none(),
        "suspension retains PCs, not a native lease"
    );
    assert_eq!(
        eval(
            &mut e,
            "var resumed=it.next(8);[resumed.value,resumed.done,log.join('')].join(':');"
        ),
        "11:true:abf"
    );
}

#[test]
fn pending_await_reclaims_code_without_losing_rejection_or_finally_state() {
    // ECMA-262 #await / #sec-asyncblockstart: both reactions resume the saved
    // context, and an awaiting finally retains the preceding completion.
    let mut e = engine();
    eval(
        &mut e,
        r#"
        var token={},sentinel={},log=[],done=[];
        var resolveGood,rejectBad,resolveCleanup;
        var good=new Promise(r=>resolveGood=r);
        var bad=new Promise((r,j)=>rejectBad=j);
        var cleanup=new Promise(r=>resolveCleanup=r);
        async function pending(input,gate){
            var saved=input;log.push('enter');
            try{var value=await gate;log.push('ok');return [saved===token,value];}
            catch(error){log.push(error===sentinel?'throw':'wrong');throw error;}
            finally{log.push('finally');await cleanup;log.push('clean');}
        }
        pending(token,good).then(value=>done.push('ok:'+value.join(':')));
        pending(token,bad).catch(error=>done.push('bad:'+(error===sentinel)));
    "#,
    );
    let c = chunk(&mut e.interp, "pending");
    let code = c
        .jit
        .get()
        .flatten()
        .expect("awaiting body entered native code");
    assert_eq!(code.test_residency_marks(false).0, 0);
    let weak = Rc::downgrade(&code);
    drop(code);
    cache::reclaim(usize::MAX);
    assert!(
        weak.upgrade().is_none(),
        "pending reactions do not pin code"
    );
    e.interp.gc_collect();
    eval(&mut e, "resolveGood(7);rejectBad(sentinel);");
    assert_eq!(
        eval(&mut e, "log.join(',')+'|'+done.length;"),
        "enter,enter,ok,finally,throw,finally|0"
    );
    cache::reclaim(usize::MAX);
    e.interp.gc_collect();
    eval(&mut e, "resolveCleanup();");
    assert_eq!(
        eval(&mut e, "log.join(',')+'|'+done.join(',');"),
        "enter,enter,ok,finally,throw,finally,clean,clean|ok:true:7,bad:true"
    );
}

#[test]
fn call_epoch_exhaustion_permanently_refuses_cached_raw_addresses() {
    const CHILD: &str = "TRUST_JS_TEST_CODE_EPOCH_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "jit::cache_tests::call_epoch_exhaustion_permanently_refuses_cached_raw_addresses",
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
    use std::sync::atomic::Ordering::Relaxed;
    let mut e = engine();
    eval(
        &mut e,
        "function leaf(x){return x+1;}function Box(x){this.x=x;}function drive(x){return leaf(x)+(new Box(x)).x;}drive(2);",
    );
    crate::bytecode::CALL_IC_EPOCH.store(u32::MAX - 1, Relaxed);
    assert_eq!(eval(&mut e, "drive(3);"), "7");
    cache::reclaim(usize::MAX);
    assert_eq!(crate::bytecode::CALL_IC_EPOCH.load(Relaxed), u32::MAX);
    crate::bytecode::invalidate_call_caches();
    assert_eq!(crate::bytecode::CALL_IC_EPOCH.load(Relaxed), u32::MAX);
    assert_eq!(
        eval(&mut e, "var out;for(var k=0;k<80;k++)out=drive(k);out;"),
        "159"
    );
    let c = chunk(&mut e.interp, "drive");
    assert!(
        c.jit.get().flatten().is_some(),
        "native execution survives cache exhaustion"
    );
}

#[test]
fn tiny_budget_reclaims_cold_code_and_compiles_later_functions() {
    const CHILD: &str = "TRUST_JS_TEST_CODE_RECLAIM_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "jit::cache_tests::tiny_budget_reclaims_cold_code_and_compiles_later_functions",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .env("TRUST_JS_JIT_CODE_BUDGET_MB", "1")
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
    use std::sync::atomic::Ordering::Relaxed;
    let mut e = engine();
    let budget = executable_code_budget();
    // Leave only 64 KiB of actual mapping capacity. The reservation is not in
    // the eviction registry; reclaiming it would violate the allocator contract.
    let held = ExecutableCodeReservation::try_new(
        budget,
        budget.remaining.load(Relaxed).saturating_sub(64 << 10),
    )
    .unwrap();
    let before = cache::eviction_count();
    assert_eq!(
        eval(
            &mut e,
            r#"
        var keep=[],sum=0;
        for(var k=0;k<600;k++){
            var f=Function('x','return x+'+k);keep.push(f);sum+=f(1);
        }
        var firstHot=keep[0],lastHot=keep[599];sum;
    "#
        ),
        "180300"
    );
    assert!(
        cache::eviction_count() > before,
        "must reclaim actual mapping owners"
    );
    assert!(
        chunk(&mut e.interp, "lastHot")
            .jit
            .get()
            .flatten()
            .is_some(),
        "late workload must really compile despite retained early functions"
    );
    let first = chunk(&mut e.interp, "firstHot");
    assert!(
        first.jit.get().is_none(),
        "old cold function was actually evicted"
    );
    assert_eq!(
        eval(&mut e, "var out;for(var n=0;n<40;n++)out=firstHot(n);out;"),
        "39"
    );
    assert!(
        first.jit.get().flatten().is_some(),
        "eviction is not permanent tier refusal"
    );
    assert!(executable_code_stats().0 <= budget.limit);
    drop(held);
}

#[test]
fn retained_callers_reject_evicted_call_and_constructor_addresses() {
    let mut e = engine();
    eval(
        &mut e,
        r#"
        function sum(a,b){return a+b;}
        function Box(x){this.x=x;if(x<0)throw 'bad';}
        function callSum(x){return sum(x,2,99);}
        function makeBox(x){return new Box(x);}
        for(var k=0;k<20;k++){callSum(k);makeBox(k);}
    "#,
    );
    let caller = chunk(&mut e.interp, "callSum");
    let constructor_caller = chunk(&mut e.interp, "makeBox");
    let caller_lease = caller.jit.get().flatten().unwrap();
    let constructor_lease = constructor_caller.jit.get().flatten().unwrap();
    let sum = chunk(&mut e.interp, "sum");
    let constructor = chunk(&mut e.interp, "Box");
    let old_sum = Rc::downgrade(&sum.jit.get().flatten().unwrap());
    let old_constructor = Rc::downgrade(&constructor.jit.get().flatten().unwrap());
    cache::reclaim(usize::MAX);
    assert!(old_sum.upgrade().is_none());
    assert!(old_constructor.upgrade().is_none());
    assert_eq!(
        eval(
            &mut e,
            r#"
        Box.prototype={changed:true};var result='',caught='';
        for(var k=0;k<40;k++)result=callSum(k)+':'+makeBox(k).x;
        try{makeBox(-1);}catch(e){caught=e;}
        result+':'+makeBox(2).changed+':'+caught;
    "#
        ),
        "41:39:true:bad"
    );
    assert!(sum.jit.get().flatten().is_some());
    assert!(constructor.jit.get().flatten().is_some());
    assert_eq!(caller_lease.test_residency_marks(false).0, 0);
    assert_eq!(constructor_lease.test_residency_marks(false).0, 0);
}

#[test]
fn recompiled_functions_keep_their_original_realm() {
    let mut e = engine();
    let child = e.interp.create_realm();
    e.interp
        .global
        .borrow_mut()
        .props
        .insert("other", crate::value::Property::plain(child));
    e.interp
        .eval_in_realm(
            &child,
            "var marker=73;function foreign(x){return [marker+x,Array,globalThis];}foreign(1);",
        )
        .ok()
        .expect("foreign setup");
    eval(
        &mut e,
        "var marker=9;function invokeOther(x){return other.foreign(x);}invokeOther(1);",
    );
    let caller = chunk(&mut e.interp, "invokeOther");
    let lease = caller
        .jit
        .get()
        .flatten()
        .expect("native cross-realm caller");
    cache::reclaim(usize::MAX);
    assert_eq!(
        eval(
            &mut e,
            r#"
        var good=true;
        for(var k=0;k<40;k++){
            var result=invokeOther(k);
            good=good&&result[0]===73+k&&result[1]===other.Array&&result[2]===other;
        }
        good;
    "#
        ),
        "true"
    );
    assert_eq!(lease.test_residency_marks(false).0, 0);
}

/// Code owners' deaths and the cache states they leave, observed through collections with
/// every strong handle and cell address the test uses kept in helper frames (the collector
/// scans the stack, and a cached entry's raw callee address would retain its callee). A cache's
/// object pin is a weak handle, so the dead callee's address stays reserved while the cache
/// names it.
#[test]
fn dead_code_owners_leave_unmatchable_call_ics_and_live_siblings_keep_sharing() {
    // Epoch is process-global. Isolate this proof from other tests deliberately
    // reclaiming mappings; it must observe OWNER death, not explicit eviction.
    const CHILD: &str = "TRUST_JS_TEST_CACHE_OWNER_DEATH_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "jit::cache_tests::dead_code_owners_leave_unmatchable_call_ics_and_live_siblings_keep_sharing",
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
    use crate::gc_test::{alive, in_frame, HiddenAddress};
    use crate::value::{Gc, Value};
    use std::sync::atomic::Ordering::Relaxed;
    fn value(e: &mut Engine, name: &str) -> Value {
        let env = e.interp.global_env;
        e.interp.get_var(name, &env).ok().expect("fixture binding")
    }
    fn function_pin(value: &Value) -> std::rc::Weak<crate::ast::Function> {
        let object_handle = value.as_obj().unwrap();
        let object = object_handle.borrow();
        let Callable::User(user) = &object.call else {
            panic!("ordinary function")
        };
        Rc::downgrade(&user.func)
    }
    fn warm(e: &mut Engine) {
        assert_eq!(
            eval(
                e,
                "target(1);invokeTarget(target,2);invokeTarget(target,3);invokeTarget(target,4);"
            ),
            "5"
        );
    }
    /// The address-free part of `caller`'s cached entry for the callee at `key`: its code and
    /// function.
    #[inline(never)]
    fn cached(caller: &Chunk, key: HiddenAddress) -> Option<(usize, usize)> {
        caller
            .cached_call_for(key.get())
            .map(|ic| (ic.code as usize, ic.func as usize))
    }
    /// The native code of the global function `name`, weakly.
    fn native_code(e: &mut Engine, name: &str) -> std::rc::Weak<JitCode> {
        let callee = chunk(&mut e.interp, name);
        let code = callee.jit.get().flatten().expect("native callee");
        Rc::downgrade(&code)
    }
    /// Whether the cell at `key` is still allocated (live, or dead and weakly pinned).
    #[inline(never)]
    fn reserved(e: &Engine, key: HiddenAddress) -> bool {
        e.interp
            .gc_heap
            .core
            .borrow()
            .is_allocated_word(key.get() as u64)
    }
    // ECMA-262 OrdinaryFunctionCreate / PrepareForOrdinaryCall (local e28783d5):
    // a new function retains its own identity/context even when code storage
    // from an unreachable earlier function has been released and reused.
    {
        let mut e = engine();
        eval(
            &mut e,
            "var target=new Function('x','return x+1');function invokeTarget(f,x){return f(x,99);}",
        );
        let (object_pin, function_pin, key) = in_frame(|| {
            let target = value(&mut e, "target");
            (
                Gc::downgrade(&target.as_obj().unwrap()),
                function_pin(&target),
                HiddenAddress::of(&target),
            )
        });
        let function_weak_before = function_pin.weak_count();
        warm(&mut e);
        let caller = chunk(&mut e.interp, "invokeTarget");
        assert!(caller.jit.get().flatten().is_some(), "real compiled caller");
        let ic = cached(&caller, key).expect("actual populated primary IC");
        assert_eq!(ic.1, function_pin.as_ptr() as usize);
        assert!(
            function_pin.weak_count() > function_weak_before,
            "code sharing separately pins AST allocation"
        );
        let code = in_frame(|| native_code(&mut e, "target"));
        // Interp assignment avoids compiling unrelated Script/inline code in
        // the observed interval. Collection does not mean cache eviction.
        e.set_tier(Tier::Interp);
        eval(&mut e, "target=null;");
        let epoch = crate::bytecode::CALL_IC_EPOCH.load(Relaxed);
        e.interp.gc_collect();
        assert!(!alive(&object_pin));
        assert!(function_pin.upgrade().is_none());
        assert!(
            code.upgrade().is_none(),
            "mapping owner died while caller cache survived"
        );
        assert_ne!(crate::bytecode::CALL_IC_EPOCH.load(Relaxed), epoch);
        let stale = cached(&caller, key).expect("stale identity intentionally remains weak-pinned");
        assert_eq!(stale, ic);
        // Without the test's own pin, the cache's keeps the dead callee's address reserved.
        drop(object_pin);
        e.interp.gc_collect();
        assert!(reserved(&e, key), "cache separately pins object allocation");
        e.set_tier(Tier::Jit);
        assert_eq!(eval(&mut e, "target=new Function('x','return x+70');target(1);invokeTarget(target,2);invokeTarget(target,3);"), "73");
        assert!(in_frame(|| {
            let target = value(&mut e, "target");
            Gc::as_ptr(&target.as_obj().unwrap()) as usize != key.get()
        }));
    }
    // The first closure may die while a sibling with the SAME immutable AST
    // remains. A code-sharing hit must use the live sibling's environment, not
    // the dead cached closure's environment or identity.
    {
        let mut e = engine();
        eval(&mut e, "var factory=new Function('offset','return function(x){return offset+x}');var first=factory(17),sibling=factory(37);function invokeShared(f,x){return f(x,99);}");
        let (first_pin, first_key, ast, sibling_key, factory_pin) = in_frame(|| {
            let first = value(&mut e, "first");
            let sibling = value(&mut e, "sibling");
            let factory = value(&mut e, "factory");
            let ast = function_pin(&first);
            assert_eq!(ast.as_ptr(), function_pin(&sibling).as_ptr());
            (
                Gc::downgrade(&first.as_obj().unwrap()),
                HiddenAddress::of(&first),
                ast,
                HiddenAddress::of(&sibling),
                Gc::downgrade(&factory.as_obj().unwrap()),
            )
        });
        // Retire the factory's separate runtime before warming the sharing proof. After the
        // first closure dies below, only the sibling retains the inner AST and runtime.
        e.set_tier(Tier::Interp);
        eval(&mut e, "factory=null;");
        e.interp.gc_collect();
        assert!(
            !alive(&factory_pin),
            "factory no longer retains the inner AST"
        );
        e.set_tier(Tier::Jit);
        assert_eq!(
            eval(
                &mut e,
                "first(1);invokeShared(first,2);invokeShared(first,3);"
            ),
            "20"
        );
        let caller = chunk(&mut e.interp, "invokeShared");
        assert!(caller.jit.get().flatten().is_some());
        assert!(cached(&caller, first_key).is_some());
        assert!(cached(&caller, sibling_key).is_none());
        let code = in_frame(|| native_code(&mut e, "first"));
        e.set_tier(Tier::Interp);
        eval(&mut e, "first=null;");
        let epoch = crate::bytecode::CALL_IC_EPOCH.load(Relaxed);
        e.interp.gc_collect();
        assert!(!alive(&first_pin));
        assert!(ast.upgrade().is_some());
        assert!(
            code.upgrade().is_some(),
            "live sibling still owns the immutable code slot"
        );
        assert_eq!(crate::bytecode::CALL_IC_EPOCH.load(Relaxed), epoch);
        e.set_tier(Tier::Jit);
        assert_eq!(eval(&mut e, "invokeShared(sibling,4);"), "41");
        assert!(
            cached(&caller, sibling_key).is_none(),
            "Function-only hit did not refill identity"
        );
        assert!(cached(&caller, first_key).is_some());
    }
    // Overflow owns an independent object pin after its original caller Chunk
    // disappears. First prove its current-epoch hit through a fresh empty callback cache;
    // caller retirement must then fence that entry, while a later callback recovers normally.
    {
        let mut e = engine();
        eval(
            &mut e,
            "var target=new Function('x','return x+1');function invokeTarget(f,x){return f(x,99);}",
        );
        warm(&mut e);
        let (pin, key) = in_frame(|| {
            let target = value(&mut e, "target");
            (
                Gc::downgrade(&target.as_obj().unwrap()),
                HiddenAddress::of(&target),
            )
        });
        let caller_code = in_frame(|| {
            let caller = chunk(&mut e.interp, "invokeTarget");
            let ic = caller
                .cached_call_for(key.get())
                .expect("native primary entry");
            e.interp.call_overflow.insert(ic, pin.clone());
            Rc::downgrade(&caller.jit.get().flatten().unwrap())
        });
        let callee_code = in_frame(|| native_code(&mut e, "target"));
        /// Whether the overflow table holds an entry for the callee at `key` in this epoch.
        #[inline(never)]
        fn overflow_hit(e: &Engine, key: HiddenAddress) -> bool {
            let epoch = crate::bytecode::CALL_IC_EPOCH.load(Relaxed);
            let genv = crate::interpreter::Env::as_ptr(&e.interp.global_env) as usize;
            e.interp
                .call_overflow
                .lookup(key.get(), genv, epoch)
                .is_some()
        }
        assert!(
            overflow_hit(&e, key),
            "a current-epoch overflow entry is published"
        );
        assert!(in_frame(|| {
            let target = value(&mut e, "target");
            // This callback cache has no primary entry: the published overflow is the only
            // cached route to the live target's native code on its first call.
            let callback = crate::callback::CallbackCache::new();
            matches!(
                (callback.call(&mut e.interp, &target, Value::Undefined, [Value::Num(8.0)]))
                    .map(|__value| (__value).view()),
                Ok(crate::value::ValueView::Num(9.0))
            )
        }));
        assert!(overflow_hit(&e, key));
        e.set_tier(Tier::Interp);
        eval(&mut e, "invokeTarget=null;");
        let epoch = crate::bytecode::CALL_IC_EPOCH.load(Relaxed);
        e.interp.gc_collect();
        assert!(
            caller_code.upgrade().is_none(),
            "overflow does not retain original caller Chunk"
        );
        assert_ne!(crate::bytecode::CALL_IC_EPOCH.load(Relaxed), epoch);
        assert!(
            !overflow_hit(&e, key),
            "caller runtime retirement fences old raw overflow addresses"
        );
        assert!(
            pin.weak_count() >= 2,
            "overflow retains its own pin besides the observer"
        );
        e.set_tier(Tier::Jit);
        assert!(in_frame(|| {
            let target = value(&mut e, "target");
            let callback = crate::callback::CallbackCache::new();
            matches!(
                (callback.call(&mut e.interp, &target, Value::Undefined, [Value::Num(9.0)]))
                    .map(|__value| (__value).view()),
                Ok(crate::value::ValueView::Num(10.0))
            )
        }));
        e.set_tier(Tier::Interp);
        eval(&mut e, "target=null;");
        let epoch = crate::bytecode::CALL_IC_EPOCH.load(Relaxed);
        e.interp.gc_collect();
        assert!(!alive(&pin));
        assert!(callee_code.upgrade().is_none());
        assert_ne!(crate::bytecode::CALL_IC_EPOCH.load(Relaxed), epoch);
        assert!(
            !overflow_hit(&e, key),
            "GC retires dead overflow entry and its pin together"
        );
    }
}
