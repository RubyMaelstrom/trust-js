use super::*;
use crate::{bytecode::Tier, value::Callable, Completion, Engine};
use std::rc::Rc;

struct Enabled(Option<bool>, Option<bool>);
impl Enabled {
    fn new() -> Self {
        Self(
            generic_own::TEST_ENABLED.with(|c| c.replace(Some(true))),
            object_facts::TEST_ENABLED.with(|c| c.replace(Some(true))),
        )
    }
}
impl Drop for Enabled {
    fn drop(&mut self) {
        generic_own::TEST_ENABLED.with(|c| c.set(self.0));
        object_facts::TEST_ENABLED.with(|c| c.set(self.1));
    }
}
fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("Object fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}
fn author_function(engine: &mut Engine) -> Rc<crate::ast::Function> {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("author function")
    };
    user.func.clone()
}
fn warm(source: &str, run: &str) -> (Engine, Rc<crate::ast::Function>, Rc<Chunk>) {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    eval(&mut engine, run);
    let function = author_function(&mut engine);
    let chunk = function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    (engine, function, chunk)
}
fn plans(
    chunk: &Chunk,
) -> (
    Vec<(u32, bytecode::IcState)>,
    Vec<(u32, bytecode::IcState)>,
    Option<object_facts::Plan>,
) {
    let cfg = Cfg::build(chunk).unwrap();
    let stack = stack::StackPlan::build(chunk, &cfg).unwrap();
    let selected = generic_admission::selected(&generic_admission::observations(chunk));
    let objects: Vec<_> = selected.iter().filter(|&&(pc, _)| matches!(chunk.jit_ops()[pc as usize], Op::GetPropLocal(_, name, _) | Op::GetPropThis(name, _) if chunk.jit_name(name) == "alternate")).copied().collect();
    let plan = object_facts::Plan::build(chunk, &cfg, &stack, &objects, &selected, &[], &[]);
    (selected, objects, plan)
}

const SOURCE: &str = "function cold(){throw 'cold'}
function finishCopy(o){return [o.source,o.x,o.a,o.b,o.c,o.d,o.e].join('|')}
function subject(o){'use strict';var n=o.alternate;if(null===n)return cold();o.x=o.source;o.a=o.x;o.b=o.a;o.c=o.b;o.d=o.c;o.e=o.d;return finishCopy(o)}
var receiver={alternate:{alive:1},source:'value',x:0,a:1,b:2,c:3,d:4,e:5};";

#[test]
fn object_guard_certificates_remove_only_exact_null_strict_branch() {
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    for comparison in [
        "null===n",
        "n===null",
        "null!==n",
        "n!==null",
        "n==null",
        "!n",
        "typeof n==='object'",
    ] {
        let source = format!("function subject(o){{var n=o.alternate;if({comparison})return 1;return 0}}var receiver={{alternate:{{}}}};");
        let (_, _, chunk) = warm(&source, "subject(receiver)");
        let (_, _, plan) = plans(&chunk);
        let plan = plan.unwrap();
        let folded = chunk
            .jit_ops()
            .iter()
            .enumerate()
            .filter(|&(pc, _)| plan.branch(pc).is_some())
            .count();
        assert_eq!(
            folded,
            usize::from(comparison.contains("===") || comparison.contains("!=="))
                * usize::from(!comparison.starts_with("typeof"))
        );
    }
}

#[test]
fn feasible_structural_minimum_debits_guard_and_preserves_abrupt_prefixes() {
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    let (_, _, chunk) = warm(SOURCE, "subject(receiver)");
    let (selected, _, object) = plans(&chunk);
    let object = object.unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    assert!(generic_admission::Plan::build(&chunk, &cfg, &stack, &selected).is_none());
    let improved =
        generic_admission::Plan::with_objects(&chunk, &cfg, &stack, &selected, Some(&object))
            .unwrap();
    assert_eq!(improved.minimum_tests, 53);
    assert!(object.reachable.iter().any(|&yes| !yes));
    for prefix in [
        "callback();",
        "if(flag)throw 1;",
        "var z=+unknown;",
        "var z=o.getter;",
    ] {
        let source = SOURCE
            .replace("var n=o.alternate;", &format!("{prefix}var n=o.alternate;"))
            .replace(
                "var receiver=",
                "function callback(){}var flag=false,unknown={valueOf(){return 1}};var receiver=",
            )
            .replace(
                "alternate:{alive:1}",
                "get getter(){return 1},alternate:{alive:1}",
            );
        let (_, _, chunk) = warm(&source, "subject(receiver)");
        let (selected, _, object) = plans(&chunk);
        let object = object.unwrap();
        let cfg = Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        assert!(
            generic_admission::Plan::with_objects(&chunk, &cfg, &stack, &selected, Some(&object))
                .is_none(),
            "{prefix}"
        );
    }
}

#[test]
fn mixed_phi_tdz_observers_and_repeated_guards_cannot_invent_feasible_edges() {
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    for source in [
        "function subject(o,b){var n=o.alternate;if(b)n=null;if(null===n)return 1;return 0}",
        "function subject(o){var n=o.alternate;function mutate(){n=null}mutate();if(null===n)return 1;return 0}",
        "function subject(o){var n=o.alternate;eval('n=null');if(null===n)return 1;return 0}",
    ] {
        let source=format!("{source};var receiver={{alternate:{{}}}};");let(_,_,chunk)=warm(&source,"subject(receiver,false)");let(_,_,plan)=plans(&chunk);
        if let Some(plan)=plan { assert!(chunk.jit_ops().iter().enumerate().all(|(pc,_)|plan.branch(pc).is_none())); }
    }
    let(_,_,chunk)=warm("function subject(o,b){while(b){var n=o.alternate;b=false;}return n}var receiver={alternate:{}};","subject(receiver,true)");
    assert!(
        plans(&chunk).2.is_none(),
        "repeated result guard has no bounded per-entry debit"
    );
    let source="function subject(o,b){if(b)return n;let n=o.alternate;if(null===n)return 1;o.x=o.x;return o.x}var receiver={alternate:{},x:'v'};";
    let (mut engine, _, chunk) = warm(source, "subject(receiver,false)");
    let (selected, _, object) = plans(&chunk);
    let object = object.unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    assert!(
        generic_admission::Plan::with_objects(&chunk, &cfg, &stack, &selected, Some(&object))
            .is_none()
    );
    assert_eq!(
        eval(&mut engine, "try{subject(receiver,true)}catch(e){e.name}"),
        "ReferenceError"
    );
}

#[test]
fn rare_tdz_prefix_before_later_copies_has_zero_credit() {
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    let source = SOURCE
        .replace("function subject(o){", "function subject(o,b){")
        .replace("var n=o.alternate;", "if(b)n;let n=o.alternate;");
    let (mut engine, _, chunk) = warm(&source, "subject(receiver,false)");
    let (selected, _, object) = plans(&chunk);
    let object = object.unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    assert!(
        generic_admission::Plan::with_objects(&chunk, &cfg, &stack, &selected, Some(&object))
            .is_none()
    );
    assert!(chunk.jit_ops().iter().enumerate().any(
        |(pc, op)| matches!(op,Op::LoadLocal(slot) if usize::from(*slot)>=chunk.jit_frame().0)
            && !object.values.at[pc].local.initialized()
    ));
    assert_eq!(
        eval(&mut engine, "try{subject(receiver,true)}catch(e){e.name}"),
        "ReferenceError"
    );
}

fn child(name: &str, default_policy: bool) -> bool {
    const MARKER: &str = "TRUST_JS_OBJECT_FIELDS_CHILD";
    if std::env::var(MARKER).as_deref() == Ok(name) {
        return false;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            &format!("jit::optimizing::object_fact_tests::{name}"),
            "--nocapture",
        ])
        .env(MARKER, name)
        .env("TRUST_JS_OPT_JIT", "hot")
        .env("TRUST_JS_OPT_JIT_DIAGNOSTICS", "compile")
        .env(
            "TRUST_JS_OPT_JIT_GENERIC_OWN",
            std::env::var("TRUST_JS_OPT_JIT_GENERIC_OWN").unwrap_or_else(|_| "1".into()),
        )
        .env(
            "TRUST_JS_OPT_JIT_OBJECT_FIELDS",
            std::env::var("TRUST_JS_OPT_JIT_OBJECT_FIELDS").unwrap_or_else(|_| "1".into()),
        )
        .env("TRUST_JS_INLINE_AT", "0");
    if default_policy {
        command.env_remove("TRUST_JS_OPT_JIT_HOT_AT");
    } else {
        command.env("TRUST_JS_OPT_JIT_HOT_AT", "8");
    }
    if name == "normal_hot_policy_snapshots_actual_baseline_inline_publication" {
        command.env_remove("TRUST_JS_INLINE_AT");
    }
    let output = command.output().unwrap();
    eprintln!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    true
}

fn candidate(
    engine: &mut Engine,
    function: &Rc<crate::ast::Function>,
    baseline: &Rc<Chunk>,
    deopt: Option<usize>,
) -> (Rc<Chunk>, Rc<JitCode>) {
    let (selected, objects, proof) = plans(baseline);
    assert!(proof.is_some());
    let next = bytecode::compile_for_optimizer_with_objects(
        function,
        baseline,
        &[],
        &[],
        &selected,
        &objects,
    )
    .unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    object_facts::TEST_EMITTED.with(|cell| cell.set([0; 3]));
    let code = Rc::new(
        compile_with_deopt(&next, &values, &layout, deopt)
            .expect("Object candidate stays inside unchanged IR budget"),
    );
    if deopt.is_none() {
        assert!(object_facts::TEST_EMITTED.with(|cell| cell.get().iter().all(|&count| count > 0)));
    }
    next.jit.begin_compile().unwrap().commit(Some(code.clone()));
    assert!(function.optimizing_code.set(Some(next.clone())).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Jit);
    (next, code)
}

fn collect(
    i: &mut crate::interpreter::Interp,
    _: crate::Value,
    _: &[crate::Value],
) -> Result<crate::Value, crate::Value> {
    i.gc_collect();
    Ok(crate::Value::Undefined)
}

fn assert_before_read(chunk: &Chunk) {
    let pc = chunk.optimizing_objects[0].0 as usize;
    let cfg = Cfg::build(chunk).unwrap();
    let stack = stack::StackPlan::build(chunk, &cfg).unwrap();
    bytecode::native_deopt::ENTRIES.with(|cell| {
        assert_eq!(
            cell.borrow()
                .iter()
                .map(|&(at, depth, _)| (at, depth))
                .collect::<Vec<_>>(),
            [(pc, stack.at[pc].depth)]
        )
    });
}

#[test]
fn callable_proxy_and_revoked_proxy_values_are_objects_without_observation() {
    if child(
        "callable_proxy_and_revoked_proxy_values_are_objects_without_observation",
        false,
    ) {
        return;
    }
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    for setup in [
        "var traps=0;receiver.alternate=function(){traps++;throw 'called'};",
        "var traps=0;receiver.alternate=new Proxy({}, {get(){traps++;throw 'get'},getPrototypeOf(){traps++;throw 'proto'}});",
        "var traps=0;var record=Proxy.revocable({}, {get(){traps++;throw 'get'},getPrototypeOf(){traps++;throw 'proto'}});record.revoke();receiver.alternate=record.proxy;",
    ] {
        let(mut engine,function,baseline)=warm(SOURCE,"subject(receiver)");let(next,code)=candidate(&mut engine,&function,&baseline,None);
        eval(&mut engine,&format!("{setup}void 0"));bytecode::native_deopt::ENTRIES.with(|cell|cell.borrow_mut().clear());code.residency.referenced.set(0);
        assert_eq!(eval(&mut engine,"subject(receiver)+'|'+traps"),"value|value|value|value|value|value|value|0");
        assert!(code.residency.referenced.get()>0);assert_eq!(next.optimizing_misses.get(),0);bytecode::native_deopt::ENTRIES.with(|cell|assert!(cell.borrow().is_empty()));
    }
}

#[test]
fn native_strict_null_orders_and_not_equal_keep_html_dda_semantics() {
    if child(
        "native_strict_null_orders_and_not_equal_keep_html_dda_semantics",
        false,
    ) {
        return;
    }
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    for head in [
        "if(null===n)return cold();",
        "if(n===null)return cold();",
        "if(null!==n){}else return cold();",
        "if(n!==null){}else return cold();",
    ] {
        let source = SOURCE.replace("if(null===n)return cold();", head);
        let (mut engine, function, baseline) = warm(&source, "subject(receiver)");
        let (next, code) = candidate(&mut engine, &function, &baseline, None);
        let dda = engine.interp.make_html_dda();
        engine
            .interp
            .global
            .borrow_mut()
            .props
            .insert("dda", crate::value::Property::plain(dda));
        eval(&mut engine, "receiver.alternate=dda");
        bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
        code.residency.referenced.set(0);
        assert_eq!(
            eval(
                &mut engine,
                "subject(receiver)+'|'+(dda==null)+'|'+Boolean(dda)+'|'+(typeof dda)"
            ),
            "value|value|value|value|value|value|value|true|false|undefined"
        );
        assert!(code.residency.referenced.get() > 0);
        assert_eq!(next.optimizing_misses.get(), 0);
        bytecode::native_deopt::ENTRIES.with(|cell| assert!(cell.borrow().is_empty()));
        assert_eq!(
            eval(
                &mut engine,
                "receiver.alternate=null;try{subject(receiver)}catch(e){String(e)}"
            ),
            "cold"
        );
        assert_eq!(next.optimizing_misses.get(), 1);
    }
}

#[test]
fn disabled_controls_keep_old_policy_and_reject_credited_object_snapshot() {
    const MARKER: &str = "TRUST_JS_OBJECT_DISABLED_CHILD";
    if std::env::var_os(MARKER).is_none() {
        for flag in [
            "TRUST_JS_OPT_JIT_OBJECT_FIELDS",
            "TRUST_JS_OPT_JIT_VALUE_FACTS",
            "TRUST_JS_OPT_JIT_HEAP_OPS",
            "TRUST_JS_OPT_JIT_COMPACT_STORES",
        ] {
            let output=std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact","jit::optimizing::object_fact_tests::disabled_controls_keep_old_policy_and_reject_credited_object_snapshot","--nocapture"])
                .env(MARKER,flag).env("TRUST_JS_OPT_JIT","hot").env("TRUST_JS_OPT_JIT_GENERIC_OWN","1").env("TRUST_JS_OPT_JIT_OBJECT_FIELDS","1")
                .env(flag,"0").env_remove("TRUST_JS_OPT_JIT_HOT_AT").env("TRUST_JS_INLINE_AT","0").output().unwrap();
            assert!(
                output.status.success(),
                "{flag}:{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        }
        return;
    }
    assert!(!object_facts::enabled());
    let (mut engine, function, baseline) = warm(SOURCE, "subject(receiver)");
    let objects: Vec<_> = baseline
        .jit_ops()
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| {
            let Op::GetPropLocal(_, name, cache) = *op else {
                return None;
            };
            (baseline.jit_name(name) == "alternate")
                .then(|| {
                    baseline
                        .jit_cache_preferred(cache)
                        .map(|state| (pc as u32, state))
                })
                .flatten()
        })
        .collect();
    assert_eq!(objects.len(), 1);
    let selected = objects.clone();
    let next = bytecode::compile_for_optimizer_with_objects(
        &function,
        &baseline,
        &[],
        &[],
        &selected,
        &objects,
    )
    .unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    assert!(matches!(
        compile_with_deopt(&next, &values, &layout, None),
        Err(CompileError::Unsupported)
    ));
    engine.set_tier(Tier::Jit);
    assert_eq!(
        eval(
            &mut engine,
            "for(var k=0;k<50000;k++)subject(receiver);subject(receiver)"
        ),
        "value|value|value|value|value|value|value"
    );
    assert!(function.optimizing_code.get().is_none());
    assert_eq!(
        eval(
            &mut engine,
            "receiver.alternate=null;try{subject(receiver)}catch(e){String(e)}"
        ),
        "cold"
    );
}

#[test]
fn object_read_invalidators_resume_before_exact_access_and_throw_once() {
    if child(
        "object_read_invalidators_resume_before_exact_access_and_throw_once",
        false,
    ) {
        return;
    }
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    for expression in ["undefined", "1", "'text'", "Symbol('s')", "3n", "false"] {
        let (mut engine, function, baseline) = warm(SOURCE, "subject(receiver)");
        let (next, code) = candidate(&mut engine, &function, &baseline, None);
        let pc = next.optimizing_objects[0].0 as usize;
        eval(&mut engine, &format!("receiver.alternate={expression}"));
        bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
        code.residency.referenced.set(0);
        assert_eq!(
            eval(&mut engine, "subject(receiver)"),
            "value|value|value|value|value|value|value"
        );
        assert_eq!(next.optimizing_misses.get(), 1);
        assert!(code.residency.referenced.get() > 0);
        let cfg = Cfg::build(&next).unwrap();
        let stack = stack::StackPlan::build(&next, &cfg).unwrap();
        bytecode::native_deopt::ENTRIES.with(|cell| {
            assert_eq!(
                cell.borrow()
                    .iter()
                    .map(|&(at, depth, _)| (at, depth))
                    .collect::<Vec<_>>(),
                [(pc, stack.at[pc].depth)]
            )
        });
    }
    let (mut engine, function, baseline) = warm(SOURCE, "subject(receiver)");
    let (next, code) = candidate(&mut engine, &function, &baseline, None);
    let global = engine.interp.global.clone();
    engine.interp.def_method(&global, "observeGC", 0, collect);
    bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
    assert_eq!(eval(&mut engine,"var gets=0;Object.defineProperty(receiver,'alternate',{get(){gets++;observeGC();subject({alternate:{},source:'other',x:0,a:1,b:2,c:3,d:4,e:5});throw 'getter'},configurable:true});try{subject(receiver)}catch(e){e+'|'+gets}"),"getter|1");
    assert_eq!(next.optimizing_misses.get(), 1);
    assert_before_read(&next);
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
    let (mut engine, function, baseline) = warm(SOURCE, "subject(receiver)");
    let (next, _) = candidate(&mut engine, &function, &baseline, None);
    bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
    assert_eq!(eval(&mut engine,"var gets=0;var proxy=new Proxy(receiver,{get(o,k,r){if(k==='alternate')gets++;return Reflect.get(o,k,r)}});subject(proxy)+'|'+gets"),"value|value|value|value|value|value|value|1");
    assert_eq!(next.optimizing_misses.get(), 1);
    assert_before_read(&next);
    let (mut engine, function, baseline) = warm(SOURCE, "subject(receiver)");
    let (next, _) = candidate(&mut engine, &function, &baseline, None);
    bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
    assert_eq!(
        eval(&mut engine, "delete receiver.alternate;subject(receiver)"),
        "value|value|value|value|value|value|value"
    );
    assert_eq!(next.optimizing_misses.get(), 1);
    assert_before_read(&next);
}

#[test]
fn copied_object_word_survives_heap_replacement_callback_and_collection() {
    if child(
        "copied_object_word_survives_heap_replacement_callback_and_collection",
        false,
    ) {
        return;
    }
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    let source = SOURCE.replace(
        "if(null===n)",
        "receiver.alternate=null;observeGC();if(null===n)",
    );
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    let global = engine.interp.global.clone();
    engine.interp.def_method(&global, "observeGC", 0, collect);
    eval(&mut engine, &source);
    eval(&mut engine,"subject(receiver);receiver.alternate={alive:1};subject(receiver);receiver.alternate={alive:2}");
    let function = author_function(&mut engine);
    let baseline = function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    // The callback can throw BEFORE later savings, so a body with only that late
    // work correctly fails admission. Test its copied-word certificate independently.
    let (_, _, proof) = plans(&baseline);
    let proof = proof.unwrap();
    assert!(baseline
        .jit_ops()
        .iter()
        .enumerate()
        .any(|(pc, _)| proof.branch(pc).is_some()));
    let cfg = Cfg::build(&baseline).unwrap();
    let stack = stack::StackPlan::build(&baseline, &cfg).unwrap();
    let (selected, _, _) = plans(&baseline);
    assert!(generic_admission::Plan::with_objects(
        &baseline,
        &cfg,
        &stack,
        &selected,
        Some(&proof)
    )
    .is_none());
    let source = SOURCE.replace(
        "return finishCopy(o)",
        "receiver.alternate=null;observeGC();return n.alive+'|'+finishCopy(o)",
    );
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    let global = engine.interp.global.clone();
    engine.interp.def_method(&global, "observeGC", 0, collect);
    eval(&mut engine, &source);
    eval(
        &mut engine,
        "subject(receiver);receiver.alternate={alive:1}",
    );
    let function = author_function(&mut engine);
    let baseline = function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    let (next, code) = candidate(&mut engine, &function, &baseline, None);
    bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
    code.residency.referenced.set(0);
    assert_eq!(
        eval(&mut engine, "subject(receiver)"),
        "1|value|value|value|value|value|value|value"
    );
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(next.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|cell| assert!(cell.borrow().is_empty()));
    assert_eq!(eval(&mut engine, "receiver.alternate===null"), "true");
}

#[test]
fn late_object_miss_preserves_completed_prefix_and_exact_canonical_pc() {
    if child(
        "late_object_miss_preserves_completed_prefix_and_exact_canonical_pc",
        false,
    ) {
        return;
    }
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    let source = SOURCE
        .replace("var n=o.alternate;if(null===n)return cold();", "")
        .replace(
            "return finishCopy(o)",
            "prefix++;if(null===o.alternate)return cold();return finishCopy(o)",
        )
        + "var prefix=0;";
    for getter in [false, true] {
        let (mut engine, function, baseline) = warm(&source, "subject(receiver)");
        let (next, code) = candidate(&mut engine, &function, &baseline, None);
        let global = engine.interp.global.clone();
        engine.interp.def_method(&global, "observeGC", 0, collect);
        if getter {
            eval(&mut engine,"var gets=0;Object.defineProperty(receiver,'alternate',{get(){gets++;observeGC();return null},configurable:true});prefix=0");
        } else {
            eval(&mut engine, "receiver.alternate=null;prefix=0");
        }
        bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
        assert_eq!(
            eval(&mut engine, "try{subject(receiver)}catch(e){String(e)}"),
            "cold"
        );
        assert_eq!(eval(&mut engine, "prefix"), "1");
        if getter {
            assert_eq!(eval(&mut engine, "gets"), "1");
        }
        let pc = next.optimizing_objects[0].0 as usize;
        let cfg = Cfg::build(&next).unwrap();
        let stack = stack::StackPlan::build(&next, &cfg).unwrap();
        bytecode::native_deopt::ENTRIES.with(|cell| {
            assert_eq!(
                cell.borrow()
                    .iter()
                    .map(|&(at, depth, _)| (at, depth))
                    .collect::<Vec<_>>(),
                [(pc, stack.at[pc].depth)]
            )
        });
        assert_eq!(next.optimizing_misses.get(), 1);
        assert_eq!(code.residency.active.get(), 0);
    }
}

#[test]
fn prepared_inline_object_snapshot_keeps_canonical_pcs_and_terminal_recovery() {
    if child(
        "prepared_inline_object_snapshot_keeps_canonical_pcs_and_terminal_recovery",
        true,
    ) {
        return;
    }
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let source=SOURCE.replace("function subject(o){'use strict';var n=o.alternate;if(null===n)return cold();", "function subject(){if(null===this.alternate)return cold();")
        .replace("o.x=o.source;o.a=o.x;o.b=o.a;o.c=o.b;o.d=o.c;o.e=o.d;return finishCopy(o)","this.x=this.source;this.a=this.x;this.b=this.a;this.c=this.b;this.d=this.c;this.e=this.d;eval('0');return leaf(3)+finishCopy(this)")
        +"function leaf(n){return n*2+1}";
    eval(&mut engine, &source);
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let user = {
        let object = value.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("author")
        };
        user.clone()
    };
    let function = user.func.clone();
    let primary = bytecode::compile(&function).unwrap();
    assert!(primary.prepared_entry);
    assert_eq!(primary.jit_frame().0, 0);
    primary.inline_attempted.set(true);
    primary.inline_retry_at.set(0);
    assert!(function.code.set(Some(primary.clone())).is_ok());
    for _ in 0..8 {
        eval(&mut engine, "subject.call(receiver)");
    }
    let inline = bytecode::plan_inlines(&primary, &function, user.realm, &engine.interp.global_env);
    assert!(!inline.is_empty());
    let flattened = bytecode::compile_with_inlines(&function, &inline, &primary).unwrap();
    assert!(flattened.prepared_entry);
    assert!(flattened
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::InlineGuard(..))));
    assert!(function.code2.set(Some(flattened.clone())).is_ok());
    bytecode::invalidate_call_caches();
    for _ in 0..8 {
        eval(&mut engine, "subject.call(receiver)");
    }
    let (next, code) = candidate(&mut engine, &function, &flattened, None);
    assert_eq!(next.jit_ops(), flattened.jit_ops());
    assert_eq!(next.jit_frame(), flattened.jit_frame());
    for &(pc, state) in &next.optimizing_objects {
        let Op::GetPropThis(name, cache) = next.jit_ops()[pc as usize] else {
            panic!("prepared This guard")
        };
        assert_eq!(next.jit_name(name), flattened.jit_name(name));
        assert!(generic_admission::same(
            state,
            flattened.jit_cache_preferred(cache).unwrap()
        ));
    }
    bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
    code.residency.referenced.set(0);
    assert_eq!(
        eval(&mut engine, "subject.call(receiver)"),
        "7value|value|value|value|value|value|value"
    );
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(next.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|cell| assert!(cell.borrow().is_empty()));
    eval(&mut engine,"var gets=0;Object.defineProperty(receiver,'alternate',{get(){gets++;return null},configurable:true})");
    bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
    assert_eq!(
        eval(
            &mut engine,
            "try{subject.call(receiver)}catch(e){e+'|'+gets}"
        ),
        "cold|1"
    );
    let pc = next.optimizing_objects[0].0 as usize;
    let cfg = Cfg::build(&next).unwrap();
    let stack = stack::StackPlan::build(&next, &cfg).unwrap();
    bytecode::native_deopt::ENTRIES.with(|cell| {
        assert_eq!(
            cell.borrow()
                .iter()
                .map(|&(at, depth, _)| (at, depth))
                .collect::<Vec<_>>(),
            [(pc, stack.at[pc].depth)]
        )
    });
    assert_eq!(next.optimizing_misses.get(), 1);
    assert_eq!(code.residency.active.get(), 0);
}

#[test]
fn forced_boundaries_keep_canonical_owners_and_never_enter_pruned_arm() {
    if child(
        "forced_boundaries_keep_canonical_owners_and_never_enter_pruned_arm",
        false,
    ) {
        return;
    }
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    let (_, _, baseline) = warm(SOURCE, "subject(receiver)");
    let (_, objects, proof) = plans(&baseline);
    let proof = proof.unwrap();
    let pcs: Vec<_> = baseline
        .jit_ops()
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| {
            (pc == 0
                || pc == objects[0].0 as usize
                || matches!(op, Op::StrictEq | Op::JumpIfFalse(_) | Op::Return)
                || !proof.reachable[pc] && matches!(op, Op::Call(..)))
            .then_some(pc)
        })
        .collect();
    for pc in pcs {
        let (mut engine, function, baseline) = warm(SOURCE, "subject(receiver)");
        let (next, code) = candidate(&mut engine, &function, &baseline, Some(pc));
        bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
        assert_eq!(
            eval(&mut engine, "subject(receiver)"),
            "value|value|value|value|value|value|value"
        );
        let cfg = Cfg::build(&next).unwrap();
        let stack = stack::StackPlan::build(&next, &cfg).unwrap();
        bytecode::native_deopt::ENTRIES.with(|cell| {
            if proof.reachable[pc] {
                assert_eq!(
                    cell.borrow()
                        .iter()
                        .map(|&(at, depth, _)| (at, depth))
                        .collect::<Vec<_>>(),
                    [(pc, stack.at[pc].depth)]
                )
            } else {
                assert!(cell.borrow().is_empty())
            }
        });
        assert_eq!(next.optimizing_misses.get(), 0);
        assert_eq!(code.residency.active.get(), 0);
        engine.interp.gc_collect();
    }
}

#[test]
fn last_owned_comparison_runs_destructor_after_gc_with_canonical_stack_owner() {
    if child(
        "last_owned_comparison_runs_destructor_after_gc_with_canonical_stack_owner",
        false,
    ) {
        return;
    }
    let _enabled = Enabled::new();
    if !object_facts::enabled() {
        return;
    }
    let source = SOURCE
        .replace("var n=o.alternate;if(null===n)return cold();", "")
        .replace(
            "return finishCopy(o)",
            "if(o.alternate===(clear(o),null))return cold();return finishCopy(o)",
        )
        + "var armed=false;function clear(o){if(armed){o.alternate=null;observeGC()}}";
    let (mut engine, function, baseline) = warm(&source, "subject(receiver)");
    let (next, code) = candidate(&mut engine, &function, &baseline, None);
    struct Probe(Rc<std::cell::Cell<usize>>);
    impl Drop for Probe {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let destroyed = Rc::new(std::cell::Cell::new(0));
    let probe = Probe(destroyed.clone());
    let sentinel = engine.interp.make_native_closure(
        "sentinel",
        0,
        Rc::new(move |_, _, _| {
            let _ = &probe;
            Ok(crate::Value::Undefined)
        }),
    );
    let weak_sentinel = Rc::downgrade(&sentinel);
    let weak_payload = {
        let env = engine.interp.global_env.clone();
        let receiver = engine.interp.get_var("receiver", &env).ok().unwrap();
        let payload = receiver
            .as_obj()
            .unwrap()
            .borrow()
            .props
            .get("alternate")
            .unwrap()
            .value();
        let object = payload.as_obj().unwrap();
        object.borrow_mut().props.insert(
            "sentinel",
            crate::value::Property::data(crate::Value::Obj(sentinel), true, true, true),
        );
        Rc::downgrade(object)
    };
    assert_eq!(weak_payload.strong_count(), 1);
    assert_eq!(weak_sentinel.strong_count(), 1);
    let observed = Rc::new(std::cell::Cell::new(0));
    let count = observed.clone();
    let live_payload = weak_payload.clone();
    let released = destroyed.clone();
    let callback = engine.interp.make_native_closure(
        "observeGC",
        0,
        Rc::new(move |i, _, _| {
            // Field owner was removed; the only surviving payload owner is the
            // published operand prefix crossing clear()/the nested collection.
            assert_eq!(live_payload.strong_count(), 1);
            i.gc_collect();
            assert_eq!(live_payload.strong_count(), 1);
            assert_eq!(released.get(), 0);
            count.set(count.get() + 1);
            Ok(crate::Value::Undefined)
        }),
    );
    engine.interp.global.borrow_mut().props.insert(
        "observeGC",
        crate::value::Property::data(crate::Value::Obj(callback), true, true, true),
    );
    eval(&mut engine, "armed=true");
    bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
    code.residency.referenced.set(0);
    assert_eq!(
        eval(&mut engine, "subject(receiver)"),
        "value|value|value|value|value|value|value"
    );
    assert_eq!(observed.get(), 1);
    assert_eq!(destroyed.get(), 1);
    assert_eq!(weak_sentinel.strong_count(), 0);
    assert_eq!(weak_payload.strong_count(), 0);
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(next.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|cell| assert!(cell.borrow().is_empty()));
}

#[test]
fn actual_default_hot_object_branch_executes_and_null_getter_misses_retire() {
    if child(
        "actual_default_hot_object_branch_executes_and_null_getter_misses_retire",
        true,
    ) {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, SOURCE);
    assert_eq!(
        eval(
            &mut engine,
            "for(var k=0;k<50000;k++)subject(receiver);subject(receiver)"
        ),
        "value|value|value|value|value|value|value"
    );
    let function = author_function(&mut engine);
    if !object_facts::enabled() {
        assert!(function.optimizing_code.get().is_none());
        return;
    }
    let candidate = function
        .optimizing_code
        .get()
        .and_then(Option::as_ref)
        .expect("actual normal-policy Object hot publication");
    assert_eq!(candidate.optimizing_objects.len(), 1);
    let code = candidate.jit.get().flatten().unwrap();
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    let limit =
        (candidate.jit_ops().len() * 40 + names::instruction_allowance(candidate)).min(16384);
    assert!(record.ir_instructions.get() <= limit);
    eprintln!(
        "object default-hot ops={} IR={} limit={} samples={}",
        candidate.jit_ops().len(),
        record.ir_instructions.get(),
        limit,
        function
            .code
            .get()
            .and_then(Option::as_ref)
            .unwrap()
            .optimizing_samples
            .get()
            .unwrap()
            .entries
            .get()
    );
    bytecode::native_deopt::ENTRIES.with(|v| v.borrow_mut().clear());
    code.residency.referenced.set(0);
    assert_eq!(
        eval(&mut engine, "subject(receiver)"),
        "value|value|value|value|value|value|value"
    );
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(candidate.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|v| assert!(v.borrow().is_empty()));
    assert_eq!(
        eval(
            &mut engine,
            "receiver.alternate=null;try{subject(receiver)}catch(e){String(e)}"
        ),
        "cold"
    );
    assert_eq!(candidate.optimizing_misses.get(), 1);
    assert_before_read(candidate);
    assert_eq!(eval(&mut engine,"var gets=0;Object.defineProperty(receiver,'alternate',{get(){gets++;return {}},configurable:true});for(var k=0;k<8;k++)subject(receiver);gets"),"8");
    assert_eq!(candidate.optimizing_misses.get(), SPECIALIZATION_MISS_LIMIT);
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
}

#[test]
fn normal_hot_policy_snapshots_actual_baseline_inline_publication() {
    if child(
        "normal_hot_policy_snapshots_actual_baseline_inline_publication",
        true,
    ) {
        return;
    }
    if !object_facts::enabled() {
        return;
    }
    let source = SOURCE.replace("return finishCopy(o)", "return leaf(3)+'|'+finishCopy(o)")
        + "function leaf(x){'use strict';return x*2+1}";
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, &source);
    assert_eq!(
        eval(
            &mut engine,
            "for(var k=0;k<200000;k++)subject(receiver);subject(receiver)"
        ),
        "7|value|value|value|value|value|value|value"
    );
    let function = author_function(&mut engine);
    let flattened = function
        .code2
        .get()
        .and_then(Option::as_ref)
        .expect("actual default baseline inline publication");
    assert!(flattened
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::InlineGuard(..))));
    let next = function
        .optimizing_code
        .get()
        .and_then(Option::as_ref)
        .expect("actual normal hot publication after real inliner");
    assert_eq!(next.jit_ops(), flattened.jit_ops());
    assert_eq!(next.jit_frame(), flattened.jit_frame());
    assert!(next
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::InlineGuard(..))));
    let code = next.jit.get().flatten().unwrap();
    let diagnostic = code.optimizing_diagnostics.as_ref().unwrap();
    let limit = (next.jit_ops().len() * 40 + names::instruction_allowance(next)).min(16384);
    assert!(diagnostic.ir_instructions.get() <= limit);
    eprintln!(
        "normal after-inline ops={} IR={} cap={} samples={} guards={}",
        next.jit_ops().len(),
        diagnostic.ir_instructions.get(),
        limit,
        flattened.optimizing_samples.get().unwrap().entries.get(),
        next.jit_ops()
            .iter()
            .filter(|op| matches!(op, Op::InlineGuard(..)))
            .count()
    );
    code.residency.referenced.set(0);
    bytecode::native_deopt::ENTRIES.with(|cell| cell.borrow_mut().clear());
    assert_eq!(
        eval(&mut engine, "subject(receiver)"),
        "7|value|value|value|value|value|value|value"
    );
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(next.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|cell| assert!(cell.borrow().is_empty()));
    assert_eq!(eval(&mut engine,"var gets=0;Object.defineProperty(receiver,'alternate',{get(){gets++;return null},configurable:true});try{subject(receiver)}catch(e){e+'|'+gets}"),"cold|1");
    assert_eq!(next.optimizing_misses.get(), 1);
    assert_before_read(next);
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
}
