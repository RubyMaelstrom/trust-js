//! Native entry, guarded splice hits and exact-PC recovery for flattened inline graphs.
//! ECMA-262 e28783d5 [[Call]], OrdinaryCallBindThis, closure environments, Completion
//! Records and WeakRef liveness. Expected JavaScript results are also checked with Node.

use super::compile_with_deopt;
use crate::bytecode::{self, native_deopt::ENTRIES, Op, Tier};
use crate::value::{Callable, Value};
use crate::{Completion, Engine};
use std::collections::HashSet;
use std::mem::{discriminant, Discriminant};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("inline fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn user(engine: &mut Engine, name: &str) -> Rc<crate::value::UserCallable> {
    let env = engine.interp.global_env.clone();
    let value = engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("declared function");
    let object = value.as_obj().expect("function object").borrow();
    let Callable::User(user) = &object.call else {
        panic!("user function")
    };
    user.clone()
}

/// Warm real calls, then run the established planner. Disabling its automatic retry for
/// this one primary makes the test independent of admission-distance environment settings.
fn flattened(
    engine: &mut Engine,
    source: &str,
    warm: &str,
) -> (Rc<crate::ast::Function>, Rc<bytecode::Chunk>) {
    eval(engine, source);
    let user = user(engine, "subject");
    let primary = bytecode::compile(&user.func).expect("primary caller");
    primary.inline_attempted.set(true);
    primary.inline_retry_at.set(0);
    assert!(user.func.code.set(Some(primary.clone())).is_ok());
    // The first call compiles a cold callee; cache publication can happen only on a later
    // call. Exercise every supplied warm path enough to seed all polymorphic/nested ways.
    for _ in 0..8 {
        eval(engine, warm);
    }
    let plan = bytecode::plan_inlines(&primary, &user.func, user.realm, &engine.interp.global_env);
    assert!(
        !plan.is_empty(),
        "fixture must produce a real guarded splice"
    );
    let chunk = bytecode::compile_with_inlines(&user.func, &plan, &primary)
        .expect("flattened canonical caller");
    assert!(chunk
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::InlineGuard(..))));
    assert!(bytecode::native_deopt::supported(&chunk));
    assert!(!bytecode::native_deopt::supported_loop_continuation(&chunk));
    (user.func.clone(), chunk)
}

fn install(
    engine: &mut Engine,
    function: &crate::ast::Function,
    chunk: &Rc<bytecode::Chunk>,
    pc: Option<usize>,
) -> Rc<crate::jit::JitCode> {
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = Rc::new(
        compile_with_deopt(chunk, &values, &layout, pc)
            .unwrap_or_else(|error| panic!("inline native code at {pc:?}: {error}")),
    );
    assert!(code.pc_offsets.is_empty(), "actual optimizing backend");
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.code2.set(Some(chunk.clone())).is_ok());
    code
}

fn assert_finished(engine: &Engine, code: &crate::jit::JitCode) {
    assert_eq!(
        code.residency.referenced.get(),
        1,
        "actual optimizing native entry"
    );
    assert_eq!(code.residency.active.get(), 0, "no stranded native lease");
    assert_eq!(engine.interp.depth, 0);
    assert!(engine.interp.fn_frames.is_empty());
}

fn clear_callee_entry_marks(chunk: &bytecode::Chunk) -> Vec<Rc<crate::jit::JitCode>> {
    let mut codes = Vec::new();
    for op in chunk.jit_ops() {
        let Op::InlineGuard(target, _) = *op else {
            continue;
        };
        let object = chunk
            .jit_inline_target(target)
            .pin
            .upgrade()
            .expect("live callee");
        let object = object.borrow();
        let Callable::User(user) = &object.call else {
            panic!("planned user callee")
        };
        let body = user
            .func
            .execution_code()
            .and_then(Option::as_ref)
            .expect("warm callee");
        let code = body.jit.get().flatten().expect("native warm callee");
        code.residency.referenced.set(0);
        codes.push(code);
    }
    codes
}

const NESTED: &str = r#"
    var calls=0;
    function add(value) {globalThis.calls++;return value+2;}
    function twice(value) {return add(value)*2;}
    function other(value) {return value+20;}
    var holder={bias:3,f:twice};
    function subject(value) {return holder.f(value)+holder.bias;}
"#;

#[test]
fn semantic_integer_mirrors_cross_inline_formals_and_keep_fallback_results_dynamic() {
    const SOURCE: &str = r#"
        var calls=0,items=[0,0,0,0,0,0,0,0];
        function choose(input){return(input^3)&7;}
        function replacement(input){calls++;return 1.5;}
        var target=choose;
        function subject(input){var x=input>>>0;var key=target(x);items[key]=x;
            return(x|0)+':'+(key|0)+':'+items[key];}
    "#;
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, SOURCE);
    assert_eq!(eval(&mut oracle, "subject(-1)"), "-1:4:4294967295");
    assert_eq!(
        eval(&mut oracle, "target=replacement;subject(-1)+'|'+calls"),
        "-1:1:4294967295|1"
    );
    for recover in [false, true] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let (function, chunk) = flattened(&mut engine, SOURCE, "subject(-1)");
        let pc = recover.then(|| {
            chunk
                .jit_ops()
                .iter()
                .position(|op| matches!(op, Op::BitXor))
                .expect("inlined bitwise body")
        });
        let callees = clear_callee_entry_marks(&chunk);
        super::integers::TEST_MIRRORS.with(|counts| counts.set([0; 4]));
        let code = install(&mut engine, &function, &chunk, pc);
        let counts = super::integers::TEST_MIRRORS.with(|counts| counts.get());
        if std::env::var("TRUST_JS_OPT_JIT_INT_SSA").as_deref() != Ok("0")
            && std::env::var("TRUST_JS_OPT_JIT_RANGES").as_deref() != Ok("0")
        {
            assert!(
                counts[0] > 0 && counts[2] > 0,
                "real inline mirror consumers {counts:?}"
            );
        }
        code.residency.referenced.set(0);
        assert_eq!(eval(&mut engine, "subject(-1)"), "-1:4:4294967295");
        assert_finished(&engine, &code);
        assert!(
            callees
                .iter()
                .all(|code| code.residency.referenced.get() == 0),
            "guard hit executes the native splice"
        );
        assert_eq!(
            eval(&mut engine, "target=replacement;subject(-1)+'|'+calls"),
            "-1:1:4294967295|1"
        );
        assert_eq!(eval(&mut engine,"target=function(){calls++;return 1n;};try{subject(-1)}catch(e){e.name+':'+items[1]+':'+calls}"),"TypeError:4294967295:2");
        assert_eq!(code.residency.active.get(), 0);
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn optimizing_inline_executes_nested_polymorphic_hits_and_identity_misses() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let (function, chunk) = flattened(
        &mut engine,
        NESTED,
        "subject(4);holder.f=other;subject(4);holder.f=twice;calls=0;",
    );
    assert!(
        chunk
            .jit_ops()
            .iter()
            .filter(|op| matches!(op, Op::InlineGuard(..)))
            .count()
            >= 3,
        "both polymorphic ways and their nested callee"
    );
    let code = install(&mut engine, &function, &chunk, None);
    let callees = clear_callee_entry_marks(&chunk);
    code.residency.referenced.set(0);
    assert_eq!(eval(&mut engine, "subject(4)+':'+calls"), "15:1");
    assert_finished(&engine, &code);
    assert!(
        callees
            .iter()
            .all(|callee| callee.residency.referenced.get() == 0),
        "a guard hit must execute the splice, not silently perform the original Call"
    );
    assert_eq!(
        eval(&mut engine, "holder.f=other;subject(4)+':'+calls"),
        "27:1"
    );
    assert!(callees
        .iter()
        .all(|callee| callee.residency.referenced.get() == 0));
    assert_eq!(
        eval(
            &mut engine,
            "holder.f=function(v){return v+40};subject(4)+':'+calls"
        ),
        "47:1"
    );
    assert_eq!(eval(&mut engine,
        "holder.f=new Proxy(twice,{apply(t,r,a){return Reflect.apply(t,r,a)+5}});subject(4)+':'+calls"),
        "20:2");
    assert_eq!(
        eval(
            &mut engine,
            "holder.f=twice.bind(null,8);subject(4)+':'+calls"
        ),
        "23:3"
    );
    assert_eq!(
        eval(
            &mut engine,
            "holder.f=Proxy.revocable(twice,{});var rev=holder.f;holder.f=rev.proxy;rev.revoke();"
        ),
        "undefined"
    );
    assert_eq!(
        eval(
            &mut engine,
            "try{subject(4)}catch(e){e instanceof TypeError}"
        ),
        "true"
    );
    assert_finished(&engine, &code);
}

#[test]
fn optimizing_inline_binds_missing_and_surplus_arguments_and_boxes_sloppy_receivers() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let (function, chunk) = flattened(
        &mut engine,
        r#"
        var trace=[];
        function leaf(a,b) {return a+':'+b+':'+this.marker;}
        var holder={marker:'object',f:leaf};
        Number.prototype.f=leaf;Number.prototype.marker='number';
        function subject(receiver,wide) {
            return wide?receiver.f((trace.push('a'),1),(trace.push('b'),2),(trace.push('extra'),3)):
                        receiver.f((trace.push('only'),4));
        }
    "#,
        "subject(holder,true);subject(holder,false);trace=[];",
    );
    let code = install(&mut engine, &function, &chunk, None);
    let callees = clear_callee_entry_marks(&chunk);
    assert_eq!(
        eval(
            &mut engine,
            "subject(holder,true)+'|'+subject(holder,false)+'|'+trace.join(',')"
        ),
        "1:2:object|4:undefined:object|a,b,extra,only"
    );
    assert!(
        callees
            .iter()
            .all(|code| code.residency.referenced.get() == 0),
        "both arities really run their mapped inline bodies"
    );
    assert_eq!(eval(&mut engine, "subject(3,false)"), "4:undefined:number");
    assert!(
        callees
            .iter()
            .any(|code| code.residency.referenced.get() == 1),
        "sloppy primitive receiver needs ordinary this boxing, so its guard must miss"
    );
    assert_finished(&engine, &code);
}

/// Each requested boundary is a separate real invocation. Observe only actual terminal
/// exits; unreachable branches are deliberately counted separately from exercised PCs.
fn sweep(
    source: &str,
    warm: &str,
    setup: &str,
    observation: &str,
    expected: &str,
) -> HashSet<Discriminant<Op>> {
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, source);
    eval(&mut oracle, warm);
    eval(&mut oracle, setup);
    assert_eq!(
        eval(&mut oracle, observation),
        expected,
        "independent interpreter oracle"
    );
    let mut probe = Engine::new();
    probe.set_tier(Tier::Jit);
    probe.set_tier_threshold(0);
    let (_, probe_chunk) = flattened(&mut probe, source, warm);
    let count = probe_chunk.jit_ops().len();
    let mut kinds = HashSet::new();
    let mut reached = 0;
    for pc in 0..=count {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let (function, chunk) = flattened(&mut engine, source, warm);
        assert_eq!(chunk.jit_ops().len(), count);
        let code = install(&mut engine, &function, &chunk, Some(pc));
        eval(&mut engine, setup);
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        code.residency.referenced.set(0);
        assert_eq!(eval(&mut engine, observation), expected, "boundary {pc}");
        assert_finished(&engine, &code);
        ENTRIES.with(|entries| {
            let entries = entries.borrow();
            assert!(entries.len() <= 1, "a VM transfer is terminal");
            if let Some(&(actual, _, floor)) = entries.first() {
                assert_eq!(actual, pc);
                assert_eq!(floor, 0);
                reached += 1;
                if let Some(op) = chunk.jit_ops().get(pc) {
                    kinds.insert(discriminant(op));
                }
            }
        });
        engine.interp.gc_collect();
    }
    // A generic-call miss skips every spliced way, so its reachable set is much smaller
    // than a hit's. Sweep every static boundary without counting skipped ways as failures.
    assert!(
        reached >= 8,
        "must exercise a real guard/bind/body continuation path"
    );
    eprintln!(
        "[inline-bailout-sweep] {} boundaries, {reached} reached; {expected}",
        count + 1
    );
    kinds
}

#[test]
fn optimizing_inline_recovery_sweeps_nested_polymorphic_hits_and_fallbacks() {
    let warm = "subject(4);holder.f=other;subject(4);holder.f=twice;calls=0;";
    let mut kinds = sweep(
        NESTED,
        warm,
        "calls=0;holder.f=twice;",
        "subject(4)+':'+calls",
        "15:1",
    );
    kinds.extend(sweep(
        NESTED,
        warm,
        "calls=0;holder.f=other;",
        "subject(4)+':'+calls",
        "27:0",
    ));
    kinds.extend(sweep(
        NESTED,
        warm,
        "calls=0;holder.f=function(v){return v+40};",
        "subject(4)+':'+calls",
        "47:0",
    ));
    for op in [
        Op::InlineGuard(0, 0),
        Op::StoreLocal(0),
        Op::LoadNameIn(0, 0, 0),
        Op::Add,
    ] {
        assert!(
            kinds.contains(&discriminant(&op)),
            "actually recover at {op:?}"
        );
    }
}

const ACTIVATION: &str = r#"
    function make(closed) {return function leaf(value) {
        this.count++;
        var y=this.bias+closed;
        return y+value;
    };}
    var holder={count:0,bias:2,f:make(4)},trace=[];
    function subject(value) {
        var read=()=>value;
        var closed=99;
        var arg={valueOf(){trace.push('coerce');$262.gc();return 3;}};
        try {
            var r=holder.f(arg);
            arguments[0]=value+1;
            eval('value += 2');
            return r+':'+read()+':'+arguments[0]+':'+closed;
        } finally {trace.push('finally');$262.gc();}
    }
"#;

#[test]
fn optimizing_inline_recovery_keeps_caller_captures_arguments_eval_and_cleanup() {
    let kinds = sweep(
        ACTIVATION,
        "subject(5);",
        "holder.count=0;trace=[];",
        "subject(5)+'|'+holder.count+'|'+trace.join(',')",
        "9:8:8:99|1|coerce,finally",
    );
    for op in [
        Op::InlineGuard(0, 0),
        Op::StoreLocal(0),
        Op::LoadNameIn(0, 0, 0),
        Op::Add,
    ] {
        assert!(
            kinds.contains(&discriminant(&op)),
            "actually recover at {op:?}"
        );
    }
}

const THROW: &str = r#"
    var trace=[],fail=false;
    function leaf(value) {trace.push('leaf');if(fail)throw new Error('broken');return value+2;}
    function middle(value) {return leaf(value)*2;}
    function subject(value) {
        try {return middle(value);}
        catch(e) {trace.push(e.message);return e.stack.includes('leaf')+':'+e.stack.includes('middle');}
        finally {trace.push('finally');}
    }
"#;

#[test]
fn optimizing_inline_recovery_preserves_nested_throw_positions_and_caller_handlers() {
    let kinds = sweep(
        THROW,
        "subject(3);",
        "trace=[];fail=true;",
        "subject(3)+'|'+trace.join(',')",
        "true:true|leaf,broken,finally",
    );
    assert!(
        kinds.contains(&discriminant(&Op::Throw)),
        "recover before the inlined throw"
    );
}

#[test]
fn optimizing_inline_hidden_callee_owner_keeps_closure_live_and_is_released() {
    for deopt in [false, true] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        let (function, chunk) = flattened(
            &mut engine,
            r#"
            function make(secret) {return function leaf(n) {
                if(n){holder.f=null;$262.gc();}
                return secret.value+3;
            };}
            var sentinel={value:7}, holder={f:make(sentinel)};
            function subject(n) {return holder.f(n);}
        "#,
            "subject(0);",
        );
        let sentinel = {
            let env = engine.interp.global_env.clone();
            let value = engine.interp.get_var("sentinel", &env).ok().unwrap();
            Rc::downgrade(value.as_obj().unwrap())
        };
        let callee = chunk
            .jit_ops()
            .iter()
            .find_map(|op| match *op {
                Op::InlineGuard(target, _) => Some(chunk.jit_inline_target(target).pin.clone()),
                _ => None,
            })
            .unwrap();
        let pc = deopt.then(|| {
            chunk
                .jit_ops()
                .iter()
                .position(|op| matches!(op, Op::LoadNameIn(..)))
                .expect("closure read in splice")
        });
        let code = install(&mut engine, &function, &chunk, pc);
        eval(&mut engine, "sentinel=null;");
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        assert_eq!(eval(&mut engine, "subject(1)"), "10");
        assert_finished(&engine, &code);
        ENTRIES.with(|entries| assert_eq!(entries.borrow().len(), usize::from(deopt)));
        engine.interp.gc_collect();
        assert!(
            callee.upgrade().is_none(),
            "hidden function slot released after splice exit"
        );
        assert!(
            sentinel.upgrade().is_none(),
            "callee destructor released captured environment owners"
        );
    }
}

#[test]
fn optimizing_inline_shared_graph_guards_running_realm_and_preserves_foreign_errors() {
    use crate::realm_inline_guard_tests::{eval_string, in_realm, set_global};
    const SOURCE: &str = r#"(function () {
        'use strict';
        globalThis.callee=function leaf(boom){if(boom)return missingInlineName;return [7];};
        globalThis.subject=function subject(f,boom){var result=f(boom);return result;};
    })();"#;
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot: &'static [u8] =
        SNAPSHOT.get_or_init(|| crate::compile_snapshot(SOURCE).expect("Realm fixture snapshot"));
    let run = |it: &mut crate::interpreter::Interp| {
        let body = it
            .shared_snapshot_program(snapshot)
            .expect("snapshot decodes");
        assert!(it.run_classic_program(&body, None).is_ok());
    };
    for deopt in [false, true] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        run(&mut engine.interp);
        let (function, chunk) = flattened(&mut engine, "", "subject(callee,false);");
        let primary = function
            .code
            .get()
            .and_then(Option::as_ref)
            .unwrap()
            .clone();
        let primary_code = primary.jit.get().flatten().expect("warmed primary body");
        let pc = deopt.then(|| {
            chunk
                .jit_ops()
                .iter()
                .position(|op| matches!(op, Op::InlineGuard(..)))
                .expect("planned Realm guard")
        });
        let code = install(&mut engine, &function, &chunk, pc);
        let callees = clear_callee_entry_marks(&chunk);
        assert_eq!(
            eval(
                &mut engine,
                "Object.getPrototypeOf(subject(callee,false))===Array.prototype"
            ),
            "true"
        );
        assert!(
            callees
                .iter()
                .all(|code| code.residency.referenced.get() == 0),
            "own-Realm guard hits the splice even after VM recovery"
        );
        assert_finished(&engine, &code);
        let main = Value::Obj(engine.interp.global.clone());
        let child = engine.interp.create_realm();
        in_realm(&mut engine.interp, &child, run);
        in_realm(&mut engine.interp, &child, |it| {
            set_global(it, "mainRealm", main.clone())
        });
        let child_function = in_realm(&mut engine.interp, &child, |it| {
            let env = it.global_env.clone();
            let value = it.get_var("subject", &env).ok().unwrap();
            let object = value.as_obj().unwrap().borrow();
            let Callable::User(user) = &object.call else {
                panic!("shared subject")
            };
            user.func.clone()
        });
        assert!(
            Rc::ptr_eq(&function, &child_function),
            "the shared AST retains both canonical bodies for Realm-aware invocation selection"
        );
        code.residency.referenced.set(0);
        primary_code.residency.referenced.set(0);
        let result = in_realm(&mut engine.interp, &child, |it| {
            eval_string(
                it,
                r#"
            var fromMain=subject(mainRealm.callee,false),own=subject(callee,false),err;
            try{subject(mainRealm.callee,true)}catch(e){err=e}
            [Object.getPrototypeOf(fromMain)===mainRealm.Array.prototype,
             Object.getPrototypeOf(own)===Array.prototype,
             err instanceof mainRealm.ReferenceError,
             !(err instanceof ReferenceError),
             err.stack.includes('leaf'),err.stack.includes('subject')].join(':')
        "#,
            )
        });
        assert_eq!(result, "true:true:true:true:true:true");
        assert_eq!(
            code.residency.referenced.get(),
            0,
            "foreign Realm selects the warmed primary instead of guaranteed-missing guards"
        );
        assert_finished(&engine, &primary_code);
        assert!(
            callees
                .iter()
                .any(|code| code.residency.referenced.get() == 1),
            "matching identity in the wrong Realm must take the original Call"
        );
        engine.interp.gc_collect();
    }
}

#[test]
fn realm_selection_skips_foreign_optimized_inlines_and_keeps_retired_baseline_preference() {
    use crate::realm_inline_guard_tests::{eval_string, in_realm};
    const SOURCE:&str="globalThis.leaf=function leaf(n){return n+3;};globalThis.subject=function subject(n){return leaf(n)*2;};";
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot: &'static [u8] = SNAPSHOT.get_or_init(|| crate::compile_snapshot(SOURCE).unwrap());
    let run = |i: &mut crate::interpreter::Interp| {
        let body = i.shared_snapshot_program(snapshot).unwrap();
        assert!(i.run_classic_program(&body, None).is_ok());
    };
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    run(&mut engine.interp);
    let (function, inlined) = flattened(&mut engine, "", "subject(5);");
    let inline_code = install(&mut engine, &function, &inlined, None);
    let primary = function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    let primary_code = primary.jit.get().flatten().unwrap();
    let genv = Rc::as_ptr(&engine.interp.global_env) as usize;
    let mut optimized = bytecode::compile_for_optimizer(&function, &inlined, &[], &[]).unwrap();
    // This tests explicit whole-body compilation and dispatch, independently of hot
    // specialization admission. The copied canonical graph and Realm constraint stay exact.
    Rc::get_mut(&mut optimized).unwrap().optimizing_candidate = false;
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = Rc::new(compile_with_deopt(&optimized, &values, &layout, None).unwrap());
    optimized
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function
        .optimizing_code
        .set(Some(optimized.clone()))
        .is_ok());
    bytecode::invalidate_call_caches();
    assert!(optimized.inline_realm_compatible(genv));
    assert_eq!(eval(&mut engine, "subject(5)"), "16");
    assert_finished(&engine, &code);
    let child = engine.interp.create_realm();
    code.residency.referenced.set(0);
    inline_code.residency.referenced.set(0);
    primary_code.residency.referenced.set(0);
    let child_genv = in_realm(&mut engine.interp, &child, |i| {
        run(i);
        let genv = Rc::as_ptr(&i.global_env) as usize;
        assert!(!optimized.inline_realm_compatible(genv));
        assert!(Rc::ptr_eq(
            function
                .execution_code_for(genv)
                .and_then(Option::as_ref)
                .unwrap(),
            &primary
        ));
        assert_eq!(eval_string(i, "subject(5)"), "16");
        genv
    });
    assert_eq!(code.residency.referenced.get(), 0);
    assert_eq!(inline_code.residency.referenced.get(), 0);
    assert_finished(&engine, &primary_code);
    // Selection after retirement must retain the compatible warmed baseline in each Realm.
    // Actual specialization-bailout/reclamation is covered separately by retirement tests.
    optimized
        .optimizing_misses
        .set(bytecode::SPECIALIZATION_MISS_LIMIT);
    bytecode::invalidate_call_caches();
    assert!(Rc::ptr_eq(
        function
            .execution_code_for(genv)
            .and_then(Option::as_ref)
            .unwrap(),
        &inlined
    ));
    assert!(Rc::ptr_eq(
        function
            .execution_code_for(child_genv)
            .and_then(Option::as_ref)
            .unwrap(),
        &primary
    ));
    assert_eq!(eval(&mut engine, "subject(5)"), "16");
    assert_finished(&engine, &inline_code);
    assert_eq!(
        code.residency.referenced.get(),
        0,
        "retired native body stays unentered"
    );
    engine.interp.gc_collect();
}
