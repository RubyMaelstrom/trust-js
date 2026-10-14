//! Forced-only ordinary own-data proofs, ECMA-262 e28783d5 OrdinaryGet,
//! OrdinarySetWithOwnDescriptor, Completion Records and WeakRef liveness.
//! Read and write remain distinct owner transactions and exact recovery PCs.

use super::*;
use crate::{bytecode::Tier, value::Callable, Completion, Engine};
use std::rc::Rc;

struct Enabled(Option<bool>);

impl Enabled {
    fn new(enabled: bool) -> Self {
        Self(generic_own::TEST_ENABLED.with(|cell| cell.replace(Some(enabled))))
    }
}

impl Drop for Enabled {
    fn drop(&mut self) {
        generic_own::TEST_ENABLED.with(|cell| cell.set(self.0));
    }
}

fn active() -> bool {
    [
        "TRUST_JS_OPT_JIT_HEAP_OPS",
        "TRUST_JS_OPT_JIT_COMPACT_STORES",
    ]
    .iter()
    .all(|key| std::env::var(key).as_deref() != Ok("0"))
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("generic own fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn function(engine: &mut Engine) -> Rc<crate::value::UserCallable> {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("author function")
    };
    user.clone()
}

fn warmed(source: &str, warm: &str) -> (Engine, Rc<crate::ast::Function>, Rc<Chunk>) {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    eval(&mut engine, warm);
    let function = function(&mut engine).func.clone();
    let chunk = function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    (engine, function, chunk)
}

fn compile_fixture(
    engine: &mut Engine,
    chunk: &Chunk,
    enabled: bool,
    deopt: Option<usize>,
) -> JitCode {
    let _enabled = Enabled::new(enabled);
    generic_own::TEST_SITES.with(|cell| cell.set([0; 2]));
    generic_own::TEST_ELISIONS.with(|cell| cell.set([0; 8]));
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    compile_with_deopt(chunk, &values, &layout, deopt).expect("forced generic optimizer")
}

fn install(engine: &mut Engine, chunk: &Rc<Chunk>, code: JitCode) -> Rc<JitCode> {
    let code = Rc::new(code);
    assert!(code.pc_offsets.is_empty(), "actual optimizing code");
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    engine.set_tier(Tier::Jit);
    code.residency.referenced.set(0);
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    code
}

fn finished(engine: &mut Engine, code: &JitCode) {
    assert!(
        code.residency.referenced.get() > 0,
        "actual native subject entry"
    );
    assert_eq!(code.residency.active.get(), 0);
    assert_eq!(engine.interp.depth, 0);
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
}

fn fixture(
    source: &str,
    warm: &str,
    run: &str,
    expected: &str,
    misses: bool,
) -> ([usize; 2], [usize; 8]) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, source);
        eval(&mut engine, warm);
        assert_eq!(eval(&mut engine, run), expected, "{tier:?}");
        engine.interp.gc_collect();
    }
    let (mut engine, _function, chunk) = warmed(source, warm);
    let compiled = compile_fixture(&mut engine, &chunk, true, None);
    let sites = generic_own::TEST_SITES.with(std::cell::Cell::get);
    let removed = generic_own::TEST_ELISIONS.with(std::cell::Cell::get);
    let code = install(&mut engine, &chunk, compiled);
    engine.interp.gc_collect();
    assert_eq!(eval(&mut engine, run), expected);
    if !misses {
        assert_eq!(
            chunk.optimizing_misses.get(),
            0,
            "positive accesses executed natively"
        );
        bytecode::native_deopt::ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    } else if active() {
        assert!(
            chunk.optimizing_misses.get() > 0,
            "changed guards terminate native execution"
        );
    }
    finished(&mut engine, &code);
    (sites, removed)
}

#[test]
fn read_store_read_reuses_structural_guards_without_skipping_writability() {
    let source = "function subject(o){'use strict';o.x=o.x;return o.x;}var receiver={x:'value'};";
    let (mut engine, _function, chunk) = warmed(source, "subject(receiver);subject(receiver)");
    let baseline = compile_fixture(&mut engine, &chunk, false, None);
    let baseline_size = stores::TEST_BODY_SIZE.with(std::cell::Cell::get);
    let compiled = compile_fixture(&mut engine, &chunk, true, None);
    let size = stores::TEST_BODY_SIZE.with(std::cell::Cell::get);
    let sites = generic_own::TEST_SITES.with(std::cell::Cell::get);
    let removed = generic_own::TEST_ELISIONS.with(std::cell::Cell::get);
    if active() {
        assert_eq!(sites, [2, 1]);
        assert_eq!(removed, [1, 1, 1, 1, 1, 1, 1, 0]);
        assert!(size.0 < baseline_size.0, "actual generated IR shrinks");
        eprintln!(
            "generic-own IR={} blocks={} bytes={}; baseline IR={} blocks={} bytes={}",
            size.0, size.1, compiled.len, baseline_size.0, baseline_size.1, baseline.len
        );
    } else {
        assert_eq!(sites, [0; 2]);
        assert_eq!(removed, [0; 8]);
    }
    let code = install(&mut engine, &chunk, compiled);
    assert_eq!(eval(&mut engine, "subject(receiver)"), "value");
    assert_eq!(chunk.optimizing_misses.get(), 0);
    finished(&mut engine, &code);
}

#[test]
fn separate_read_store_transactions_preserve_all_packed_categories_and_aliases() {
    for expression in [
        "undefined",
        "null",
        "false",
        "-0",
        "NaN",
        "Infinity",
        "'shared'",
        "Symbol('value')",
        "12345678901234567890n",
        "receiver",
        "({tag:1})",
        "[1,2]",
        "Math.max",
    ] {
        for body in ["o.x=o.x;return o.x", "this.x=this.x;return this.x"] {
            let source = format!("function subject(o){{'use strict';{body};}}var receiver={{x:0}},held={expression};receiver.x=held;");
            let call = if body.starts_with("this") {
                "subject.call(receiver,receiver)"
            } else {
                "subject(receiver)"
            };
            let warm = format!("{call};{call};");
            let run =
                format!("var result={call};Object.is(result,held)+'|'+Object.is(receiver.x,held)");
            let (sites, removed) = fixture(&source, &warm, &run, "true|true", false);
            if active() {
                assert_eq!(sites, [2, 1], "{body}, {expression}");
                assert_eq!(removed, [1, 1, 1, 1, 1, 1, 1, 0]);
            } else {
                assert_eq!(sites, [0; 2]);
            }
        }
    }
}

#[test]
fn different_fields_and_receivers_never_cache_contents_or_cancel_owners() {
    for (body, run) in [
        (
            "o.x=o.y;return o.x",
            "Object.is(subject(receiver),held)+'|'+Object.is(receiver.y,held)",
        ),
        (
            "target.x=o.x;return target.x",
            "Object.is(subject(receiver,target),held)+'|'+Object.is(receiver.x,held)",
        ),
    ] {
        let source = format!("function subject(o,target){{'use strict';{body};}}var held={{keep:1}},receiver={{x:held,y:held}},target={{x:held}};");
        let (sites, _) = fixture(
            &source,
            "subject(receiver,target);subject(receiver,target)",
            run,
            "true|true",
            false,
        );
        assert_eq!(sites, if active() { [2, 1] } else { [0; 2] });
    }
}

#[test]
fn last_owned_old_field_recovers_before_store_with_the_completed_read_owner() {
    let source = "function subject(source,target){'use strict';target.x=source.x;return 42;}var held={keep:1},source={x:held},target={x:held};";
    let (mut engine, _function, chunk) =
        warmed(source, "subject(source,target);subject(source,target)");
    let compiled = compile_fixture(&mut engine, &chunk, true, None);
    let store_pc = chunk
        .jit_ops()
        .iter()
        .position(|op| matches!(op, Op::SetPropLocalDrop(..)))
        .unwrap();
    let code = install(&mut engine, &chunk, compiled);
    eval(&mut engine, "target.x={onlyOldOwner:true}");
    let old = {
        let env = engine.interp.global_env.clone();
        let target = engine.interp.get_var("target", &env).ok().unwrap();
        let target = target.as_obj().unwrap().borrow();
        let value = target.props.property_at(0).unwrap().value();
        Rc::downgrade(value.as_obj().unwrap())
    };
    assert_eq!(old.strong_count(), 1);
    assert_eq!(eval(&mut engine, "subject(source,target)"), "42");
    assert!(
        old.upgrade().is_none(),
        "old sole owner is released exactly once"
    );
    assert_eq!(eval(&mut engine, "Object.is(target.x,held)"), "true");
    if active() {
        bytecode::native_deopt::ENTRIES.with(|entries| {
            assert_eq!(
                entries
                    .borrow()
                    .iter()
                    .map(|row| (row.0, row.1))
                    .collect::<Vec<_>>(),
                [(store_pc, 1)]
            );
        });
        assert_eq!(chunk.optimizing_misses.get(), 1);
    }
    finished(&mut engine, &code);
}

#[test]
fn last_owned_host_callable_payload_is_destroyed_on_the_vm_store_once() {
    struct Probe(Rc<std::cell::Cell<usize>>);
    impl Drop for Probe {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1)
        }
    }
    let (mut engine,_function,chunk)=warmed(
        "function subject(source,target){'use strict';target.x=source.x;return 42;}var held={keep:1},source={x:held},target={x:held};",
        "subject(source,target);subject(source,target)");
    let compiled = compile_fixture(&mut engine, &chunk, true, None);
    let pc = chunk
        .jit_ops()
        .iter()
        .position(|op| matches!(op, Op::SetPropLocalDrop(..)))
        .unwrap();
    let code = install(&mut engine, &chunk, compiled);
    let destroyed = Rc::new(std::cell::Cell::new(0));
    let probe = Probe(destroyed.clone());
    let callback = engine.interp.make_native_closure(
        "ownedHostPayload",
        0,
        Rc::new(move |_, _, _| {
            let _ = &probe;
            Ok(crate::Value::Undefined)
        }),
    );
    let weak = Rc::downgrade(&callback);
    let env = engine.interp.global_env.clone();
    let target = engine.interp.get_var("target", &env).ok().unwrap();
    target.as_obj().unwrap().borrow_mut().props.insert(
        "x",
        crate::value::Property::data(crate::Value::Obj(callback), true, true, true),
    );
    assert_eq!(weak.strong_count(), 1);
    assert_eq!(eval(&mut engine, "subject(source,target)"), "42");
    assert_eq!(destroyed.get(), 1);
    assert!(weak.upgrade().is_none());
    assert_eq!(eval(&mut engine, "Object.is(target.x,held)"), "true");
    if active() {
        bytecode::native_deopt::ENTRIES.with(|entries| {
            assert_eq!(
                entries
                    .borrow()
                    .iter()
                    .map(|row| (row.0, row.1))
                    .collect::<Vec<_>>(),
                [(pc, 1)]
            )
        });
        assert_eq!(chunk.optimizing_misses.get(), 1);
    }
    finished(&mut engine, &code);
    assert_eq!(destroyed.get(), 1);
}

#[test]
fn current_storage_is_reloaded_after_same_shape_backing_replacement() {
    let (mut engine, _function, chunk) = warmed(
        "function subject(o){'use strict';o.x=o.x;return o.x;}var receiver={x:'live'};",
        "subject(receiver);subject(receiver)",
    );
    let compiled = compile_fixture(&mut engine, &chunk, true, None);
    let code = install(&mut engine, &chunk, compiled);
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("receiver", &env).ok().unwrap();
    let receiver = value.as_obj().unwrap();
    let shape = receiver.borrow().props.shape();
    let replacement = {
        let object = receiver.borrow();
        let mut replacement = crate::value::Props::with_capacity(256);
        assert!(replacement.try_copy_named_data_from(&object.props));
        assert_eq!(replacement.shape(), shape);
        assert_ne!(
            replacement.property_at(0).unwrap() as *const _,
            object.props.property_at(0).unwrap() as *const _
        );
        replacement
    };
    receiver.borrow_mut().replace_properties(replacement);
    engine.interp.gc_collect();
    property::TEST_READ_HELPERS.with(|cell| cell.set(0));
    elements::TEST_HEAP_HELPERS.with(|cell| cell.set([0; 3]));
    assert_eq!(eval(&mut engine, "subject(receiver)"), "live");
    assert_eq!(chunk.optimizing_misses.get(), 0);
    assert_eq!(property::TEST_READ_HELPERS.with(std::cell::Cell::get), 0);
    assert_eq!(
        elements::TEST_HEAP_HELPERS.with(std::cell::Cell::get),
        [
            usize::from(std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() == Ok("0")),
            0,
            0
        ]
    );
    finished(&mut engine, &code);
}

#[test]
fn getter_proxy_and_readonly_misses_execute_complete_semantics_once() {
    let (sites, _) = fixture(
        "function subject(o){'use strict';o.x=o.x;return o.x;}var receiver={x:1},trace=[];",
        "subject(receiver);subject(receiver)",
        r#"Object.defineProperty(receiver,'x',{get(){trace.push('get');return 7},set(v){trace.push('set:'+v)},configurable:true});
        var first=subject(receiver);
        receiver=new Proxy({x:3},{get(t,k,r){trace.push('proxy-get');return Reflect.get(t,k,r)},set(t,k,v,r){trace.push('proxy-set');return Reflect.set(t,k,v,r)}});
        var second=subject(receiver);
        receiver={x:4};Object.defineProperty(receiver,'x',{writable:false});var third;
        try{subject(receiver)}catch(e){third=e.name}
        [first,second,third,trace.join(',')].join('|')"#,
        "7|3|TypeError|get,set:7,get,proxy-get,proxy-set,proxy-get",
        true,
    );
    assert_eq!(sites, if active() { [2, 1] } else { [0; 2] });
}

#[test]
fn readonly_guard_and_observed_rhs_recover_at_the_store_without_replaying_read_or_call() {
    for (body,warm,change,run) in [
        ("o.x=o.x;return o.x","subject(receiver);subject(receiver)",
         "Object.defineProperty(receiver,'x',{writable:false})","subject(receiver)"),
        ("o.x=rhs();return o.x","subject(receiver,rhs);subject(receiver,rhs)",
         "rhs=function(){calls++;Object.defineProperty(receiver,'x',{writable:false});return held}","subject(receiver,rhs)"),
    ] {
        let source=format!("function subject(o,rhs){{'use strict';{body};}}var held={{tag:1}},receiver={{x:held}},calls=0;function rhs(){{return held}}");
        let (mut engine,_function,chunk)=warmed(&source,warm);
        let compiled=compile_fixture(&mut engine,&chunk,true,None);
        let pc=chunk.jit_ops().iter().position(|op|matches!(op,Op::SetPropLocalDrop(..))).unwrap();
        let code=install(&mut engine,&chunk,compiled);
        eval(&mut engine,change);
        let run=format!("var error;try{{{run}}}catch(e){{error=e.name}}error+'|'+Object.is(receiver.x,held)+'|'+calls");
        assert_eq!(eval(&mut engine,&run),if body.contains("rhs()") {"TypeError|true|1"}else{"TypeError|true|0"});
        if active(){
            bytecode::native_deopt::ENTRIES.with(|entries|assert_eq!(entries.borrow().iter().map(|row|(row.0,row.1)).collect::<Vec<_>>(),[(pc,1)]));
            assert_eq!(chunk.optimizing_misses.get(),1);
        }
        finished(&mut engine,&code);
    }
}

#[test]
fn shape_prototype_deletion_primitive_and_exotic_changes_keep_full_get_set_semantics() {
    let (sites,_) = fixture(
        "function subject(o){'use strict';o.x=o.x;return o.x;}var receiver={x:3},trace=[];",
        "subject(receiver);subject(receiver)",
        r#"receiver.extra=1;var values=[subject(receiver)];
        Object.setPrototypeOf(receiver,{get x(){trace.push('get');return 11},set x(v){trace.push('set:'+v)}});
        delete receiver.x;values.push(subject(receiver));values.push(Object.hasOwn(receiver,'x'));
        var array=[];array.x=9;values.push(subject(array));
        for(var item of [null,undefined,1,true,'text',Symbol('p'),1n]){try{subject(item)}catch(e){values.push(e.name)}}
        values.join('|')+'|'+trace.join(',')"#,
        "3|11|false|9|TypeError|TypeError|TypeError|TypeError|TypeError|TypeError|TypeError|get,set:11,get",true);
    assert_eq!(sites, if active() { [2, 1] } else { [0; 2] });
}

#[test]
fn observer_rejoin_rebind_and_branch_roots_do_not_import_metadata() {
    let (sites, removed) = fixture(
        r#"function subject(o,callback,flag){'use strict';o.x=o.x;callback();
        if(flag)o=other;else o=receiver;o.x=o.x;return o.x;}
        var receiver={x:2},other={x:9},trace=[];function nothing(){}"#,
        "subject(receiver,nothing,true);subject(receiver,nothing,false)",
        r#"function mutate(){trace.push('callback');Object.defineProperty(receiver,'x',{
        get(){trace.push('get');return 5},set(v){trace.push('set:'+v)},configurable:true});}
        var result=subject(receiver,mutate,false);[result,trace.join(',')].join('|')"#,
        "5|callback,get,set:5,get",
        true,
    );
    assert_eq!(sites, if active() { [3, 2] } else { [0; 2] });
    if active() {
        assert_eq!(removed, [1, 1, 1, 1, 2, 2, 2, 0]);
    }
}

#[test]
fn internal_before_store_stress_boundary_materializes_the_read_rhs() {
    let (mut engine, _function, chunk) = warmed(
        "function subject(o){'use strict';o.x=o.x;return o.x;}var held={tag:1},receiver={x:held};",
        "subject(receiver);subject(receiver)",
    );
    let pc = chunk
        .jit_ops()
        .iter()
        .position(|op| matches!(op, Op::SetPropLocalDrop(..)))
        .unwrap();
    let compiled = compile_fixture(&mut engine, &chunk, true, Some(pc));
    let code = install(&mut engine, &chunk, compiled);
    assert_eq!(
        eval(
            &mut engine,
            "Object.is(subject(receiver),held)+'|'+Object.is(receiver.x,held)"
        ),
        "true|true"
    );
    bytecode::native_deopt::ENTRIES.with(|entries| {
        assert_eq!(
            entries
                .borrow()
                .iter()
                .map(|row| (row.0, row.1))
                .collect::<Vec<_>>(),
            [(pc, 1)]
        )
    });
    assert_eq!(
        chunk.optimizing_misses.get(),
        0,
        "stress transfer is not a specialization miss"
    );
    finished(&mut engine, &code);
}

#[test]
fn getter_throw_recovers_exact_cleanup_and_stack_frames() {
    let (sites, _) = fixture(
        r#"function subject(o){'use strict';try{o.x=o.x;return o.x}finally{trace.push('finally')}}
        var receiver={x:3},trace=[];"#,
        "subject(receiver);subject(receiver);trace=[]",
        r#"Object.defineProperty(receiver,'x',{get(){trace.push('get');throw 'stop'},configurable:true});
        var result;try{subject(receiver)}catch(e){result=e}
        result+'|'+trace.join(',')"#,
        "stop|get,finally",
        true,
    );
    assert_eq!(sites, if active() { [2, 1] } else { [0; 2] });
}

#[test]
fn unselected_getter_rejoin_and_reentrant_collection_kill_metadata() {
    fn collect(
        i: &mut crate::interpreter::Interp,
        _: crate::Value,
        _: &[crate::Value],
    ) -> Result<crate::Value, crate::Value> {
        i.gc_collect();
        Ok(crate::Value::Undefined)
    }
    const SOURCE: &str = r#"var armed=false,inside=false,trace=[],other={x:9,middle:0};
        function subject(o){'use strict';o.x=o.x;var ignored=o.middle;o.x=o.x;return o.x;}
        var receiver={x:2,get middle(){if(armed&&!inside){trace.push('middle');inside=true;
        subject(other);inside=false;observeGC();Object.defineProperty(receiver,'x',{
        get(){trace.push('get');return 5},set(v){trace.push('set:'+v)},configurable:true});}return 0}};"#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(if tier == Tier::Jit {
            Tier::Bytecode
        } else {
            tier
        });
        engine.set_tier_threshold(0);
        let global = engine.interp.global.clone();
        engine.interp.def_method(&global, "observeGC", 0, collect);
        eval(&mut engine, SOURCE);
        eval(&mut engine, "subject(receiver);subject(receiver)");
        let native = if tier == Tier::Jit {
            // Warm the forced body with ordinary VM calls, independently of the
            // platform's template admission and inline retry thresholds.
            engine.set_tier(Tier::Bytecode);
            let user = function(&mut engine);
            let chunk = user
                .func
                .execution_code()
                .and_then(Option::as_ref)
                .unwrap()
                .clone();
            let compiled = compile_fixture(&mut engine, &chunk, true, None);
            Some((chunk.clone(), install(&mut engine, &chunk, compiled)))
        } else {
            None
        };
        assert_eq!(
            eval(
                &mut engine,
                "armed=true;subject(receiver)+'|'+trace.join(',')"
            ),
            "5|middle,get,set:5,get"
        );
        if let Some((chunk, code)) = native {
            if active() {
                let pc = chunk
                    .jit_ops()
                    .iter()
                    .enumerate()
                    .filter_map(|(pc, op)| {
                        matches!(op,Op::GetPropLocal(_,name,_) if chunk.jit_name(*name)=="x")
                            .then_some(pc)
                    })
                    .nth(1)
                    .unwrap();
                bytecode::native_deopt::ENTRIES.with(|entries| {
                    assert_eq!(
                        entries.borrow().iter().map(|row| row.0).collect::<Vec<_>>(),
                        [pc]
                    )
                });
                assert_eq!(chunk.optimizing_misses.get(), 1);
            }
            finished(&mut engine, &code);
        }
    }
}

#[test]
fn flattened_inline_getter_throw_preserves_completed_reads_and_outer_finally() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"var trace=[],receiver={x:3};
        function copy(o){'use strict';o.x=o.x;return o.x}
        function subject(o){'use strict';try{return copy(o)}finally{trace.push('finally')}}"#,
    );
    let user = function(&mut engine);
    let primary = bytecode::compile(&user.func).unwrap();
    primary.inline_attempted.set(true);
    primary.inline_retry_at.set(0);
    assert!(user.func.code.set(Some(primary.clone())).is_ok());
    // Populate the real native call-site cache that the inline planner reads.
    // VM-only calls deliberately do not publish that dispatch cache.
    engine.set_tier(Tier::Jit);
    for _ in 0..8 {
        eval(&mut engine, "subject(receiver)");
    }
    let plan = bytecode::plan_inlines(&primary, &user.func, user.realm, &engine.interp.global_env);
    assert!(!plan.is_empty(), "actual inline planner selected copy");
    let chunk = bytecode::compile_with_inlines(&user.func, &plan, &primary).unwrap();
    assert!(chunk
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::InlineGuard(..))));
    // A newly flattened graph has fresh property PCs: seed its real caches on
    // the VM before lowering, rather than inventing synthetic descriptors.
    assert!(user.func.code2.set(Some(chunk.clone())).is_ok());
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, "subject(receiver);subject(receiver);trace=[]");
    let compiled = compile_fixture(&mut engine, &chunk, true, None);
    let sites = generic_own::TEST_SITES.with(std::cell::Cell::get);
    assert_eq!(sites, if active() { [2, 1] } else { [0; 2] });
    let code = install(&mut engine, &chunk, compiled);
    let getter_pc = chunk
        .jit_ops()
        .iter()
        .position(|op| matches!(op,Op::GetPropLocal(_,name,_) if chunk.jit_name(*name)=="x"))
        .unwrap();
    assert_eq!(
        eval(
            &mut engine,
            r#"Object.defineProperty(receiver,'x',{get(){trace.push('get');throw new Error('stop')},configurable:true});
        var answer;try{subject(receiver)}catch(e){answer=e.message+'|'+e.stack.includes('at copy (')+'|'+e.stack.includes('at subject (')}
        answer+'|'+trace.join(',')"#
        ),
        "stop|true|true|get,finally"
    );
    if active() {
        bytecode::native_deopt::ENTRIES.with(|entries| {
            assert_eq!(
                entries.borrow().iter().map(|row| row.0).collect::<Vec<_>>(),
                [getter_pc]
            )
        });
        assert_eq!(chunk.optimizing_misses.get(), 1);
    }
    finished(&mut engine, &code);
}

#[test]
fn eight_before_access_misses_retire_this_version_without_stranding_a_lease() {
    let (mut engine, _function, chunk) = warmed(
        "function subject(o){'use strict';o.x=o.x;return o.x;}var receiver={x:3},gets=0,sets=0;",
        "subject(receiver);subject(receiver)",
    );
    let compiled = compile_fixture(&mut engine, &chunk, true, None);
    let code = install(&mut engine, &chunk, compiled);
    eval(&mut engine,"Object.defineProperty(receiver,'x',{get(){gets++;return 7},set(v){sets++},configurable:true})");
    for _ in 0..8 {
        assert_eq!(eval(&mut engine, "subject(receiver)"), "7");
    }
    assert_eq!(eval(&mut engine, "gets+'|'+sets"), "16|8");
    if active() {
        assert_eq!(chunk.optimizing_misses.get(), SPECIALIZATION_MISS_LIMIT);
        bytecode::native_deopt::ENTRIES.with(|entries| assert_eq!(entries.borrow().len(), 8));
        let calls = code.residency.referenced.get();
        assert_eq!(eval(&mut engine, "subject(receiver)"), "7");
        assert_eq!(
            code.residency.referenced.get(),
            calls,
            "retired native version is no longer entered"
        );
        assert_eq!(chunk.optimizing_misses.get(), SPECIALIZATION_MISS_LIMIT);
    }
    finished(&mut engine, &code);
}

#[test]
fn explicit_disable_retains_generic_helper_rejoins_and_hot_candidates_stay_separate() {
    let (mut engine, function, chunk) = warmed(
        "function subject(o){'use strict';o.x=o.x;return o.x;}var receiver={x:3};",
        "subject(receiver);subject(receiver)",
    );
    let compiled = compile_fixture(&mut engine, &chunk, false, None);
    assert_eq!(generic_own::TEST_SITES.with(std::cell::Cell::get), [0; 2]);
    let code = install(&mut engine, &chunk, compiled);
    assert_eq!(eval(&mut engine,"Object.defineProperty(receiver,'x',{get(){return 7},set(v){},configurable:true});subject(receiver)"),"7");
    assert_eq!(chunk.optimizing_misses.get(), 0);
    finished(&mut engine, &code);
    let candidate = bytecode::compile_for_optimizer(&function, &chunk, &[], &[]).unwrap();
    let _enabled = Enabled::new(true);
    let cfg = Cfg::build(&candidate).unwrap();
    let stack = stack::StackPlan::build(&candidate, &cfg).unwrap();
    let frame = frame::FramePlan::build(&candidate, &cfg, true).unwrap();
    assert!(generic_own::Plan::build(&candidate, &cfg, &frame, &stack).is_none());
}

#[test]
fn experimental_environment_flag_requires_explicit_one() {
    const CHILD: &str = "TRUST_JS_TEST_GENERIC_OWN_FLAG_CHILD";
    if std::env::var_os(CHILD).is_none() {
        for value in ["0", "1"] {
            let output=std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact","jit::optimizing::generic_own_tests::experimental_environment_flag_requires_explicit_one","--nocapture"])
                .env(CHILD,"1")
                .env("TRUST_JS_OPT_JIT_GENERIC_OWN",value)
                .output().unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        }
        return;
    }
    assert_eq!(generic_own::TEST_ENABLED.with(std::cell::Cell::get), None);
    let (mut engine, _function, chunk) = warmed(
        "function subject(o){'use strict';o.x=o.x;return o.x;}var receiver={x:'flag'};",
        "subject(receiver);subject(receiver)",
    );
    generic_own::TEST_SITES.with(|cell| cell.set([0; 2]));
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let compiled = compile_with_deopt(&chunk, &values, &layout, None).unwrap();
    let enabled = std::env::var("TRUST_JS_OPT_JIT_GENERIC_OWN").as_deref() == Ok("1") && active();
    assert_eq!(
        generic_own::TEST_SITES.with(std::cell::Cell::get),
        if enabled { [2, 1] } else { [0; 2] }
    );
    let code = install(&mut engine, &chunk, compiled);
    assert_eq!(eval(&mut engine, "subject(receiver)"), "flag");
    assert_eq!(chunk.optimizing_misses.get(), 0);
    finished(&mut engine, &code);
}

#[test]
fn hot_selection_is_exact_bounded_and_rejects_every_invalid_site() {
    if !active() {
        return;
    }
    let (mut engine, function, primary) = warmed(
        "function subject(o){'use strict';o.x=o.x;return o.x;}var receiver={x:'selected'};",
        "subject(receiver);subject(receiver)",
    );
    let _enabled = Enabled::new(true);
    let selected: Vec<_> = primary
        .jit_ops()
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| {
            let cache = match *op {
                Op::GetPropLocal(_, _, cache) | Op::SetPropLocalDrop(_, _, cache) => cache,
                _ => return None,
            };
            Some((pc as u32, primary.jit_cache_preferred(cache).unwrap()))
        })
        .collect();
    assert_eq!(selected.len(), 3);
    let cfg = Cfg::build(&primary).unwrap();
    let stack = stack::StackPlan::build(&primary, &cfg).unwrap();
    let frame = frame::FramePlan::build(&primary, &cfg, true).unwrap();
    let build = |sites: &[(u32, crate::bytecode::IcState)]| {
        generic_own::Plan::build_hot_selection(&primary, &cfg, &frame, &stack, sites)
    };
    assert!(build(&selected).is_some());
    assert!(
        generic_own::Plan::build_hot(&primary, &cfg, &frame, &stack, &selected).is_none(),
        "lowering requires an actual candidate"
    );
    assert!(build(&[]).is_none());
    assert!(build(&vec![selected[0]; 33]).is_none());
    assert!(build(&[selected[0], selected[0]]).is_none());
    assert!(build(&[selected[1], selected[0]]).is_none());
    let mut invalid = selected.clone();
    invalid[0].0 = u32::MAX;
    assert!(build(&invalid).is_none());
    let nonproperty = primary
        .jit_ops()
        .iter()
        .position(|op| !matches!(op, Op::GetPropLocal(..) | Op::SetPropLocalDrop(..)))
        .unwrap();
    assert!(build(&[(nonproperty as u32, selected[0].1)]).is_none());
    for field in 0..9 {
        let mut stale = selected.clone();
        let state = &mut stale[0].1;
        match field {
            0 => state.recv_shape = state.recv_shape.wrapping_add(1),
            1 => state.holder_shape = state.holder_shape.wrapping_add(1),
            2 => state.slot = state.slot.wrapping_add(1),
            3 => state.depth = 1,
            4 => state.mid_ok ^= 1,
            5 => state.mid_shape = state.mid_shape.wrapping_add(1),
            6 => state.mid2_shape = state.mid2_shape.wrapping_add(1),
            7 => state.mid3_shape = state.mid3_shape.wrapping_add(1),
            8 => state.mid4_shape = state.mid4_shape.wrapping_add(1),
            _ => unreachable!(),
        }
        assert!(
            build(&stale).is_none(),
            "stale scalar field{field} cannot silently lose credit"
        );
    }
    let mut invalid = selected.clone();
    invalid[0].1.recv_shape = u32::MAX;
    assert!(build(&invalid).is_none());
    let candidate = bytecode::compile_for_optimizer(&function, &primary, &[], &[]).unwrap();
    let candidate_cfg = Cfg::build(&candidate).unwrap();
    let candidate_stack = stack::StackPlan::build(&candidate, &candidate_cfg).unwrap();
    let candidate_frame = frame::FramePlan::build(&candidate, &candidate_cfg, true).unwrap();
    assert!(generic_own::Plan::build_hot(
        &candidate,
        &candidate_cfg,
        &candidate_frame,
        &candidate_stack,
        &selected
    )
    .is_some());
    // The middle write was not selected and retains its generic rejoining
    // helper. Even its normal completion cannot establish a metadata fact.
    let subset = build(&[selected[0], selected[2]]).unwrap();
    let final_read = subset.read(selected[2].0 as usize).unwrap().removed;
    assert!(!final_read.ordinary && !final_read.shape && !final_read.bounds && !final_read.data);
    assert!(subset.write(selected[1].0 as usize).is_none());
    let subset = build(&[selected[1], selected[2]]).unwrap();
    let final_read = subset.read(selected[2].0 as usize).unwrap().removed;
    assert!(
        final_read.ordinary && final_read.shape && final_read.bounds && final_read.data,
        "successful writable data proves readable data"
    );
    engine.interp.gc_collect();
}

#[test]
fn hot_selection_rejects_indexed_or_environment_homed_receivers() {
    if !active() {
        return;
    }
    for (source,warm) in [
        ("function subject(o){'use strict';o['0']=o['0'];return o.x}var receiver={'0':3,x:4}","subject(receiver);subject(receiver)"),
        ("function subject(o){'use strict';function capture(){return o}o.x=o.x;return o.x}var receiver={x:3}","subject(receiver);subject(receiver)"),
    ] {
        let (_engine,_function,chunk)=warmed(source,warm);
        let _enabled=Enabled::new(true);
        let cfg=Cfg::build(&chunk).unwrap();
        let stack=stack::StackPlan::build(&chunk,&cfg).unwrap();
        let frame=frame::FramePlan::build(&chunk,&cfg,true).unwrap();
        let site=chunk.jit_ops().iter().enumerate().find_map(|(pc,op)|{
            let cache=match *op{Op::GetPropLocal(_,_,cache)|Op::SetPropLocalDrop(_,_,cache)|Op::GetProp(_,cache)|Op::SetProp(_,cache)|Op::SetPropDrop(_,cache)=>cache,_=>return None};
            Some((pc as u32,chunk.jit_cache_preferred(cache).unwrap()))
        }).unwrap();
        let rejected_pc=if source.contains("o['0']") {
            // Indexed syntax uses the element opcode family without a named
            // IC. A warmed x IC supplies valid scalar state, but cannot make
            // the selected element operation a named-property proof site.
            chunk.jit_ops().iter().position(|op|matches!(op,Op::GetElem|Op::GetElemLocal(_))).unwrap() as u32
        }else{site.0};
        assert!(generic_own::Plan::build_hot_selection(&chunk,&cfg,&frame,&stack,&[(rejected_pc,site.1)]).is_none());
    }
}

#[test]
fn pinned_react_qu_copy_body_has_real_structural_elisions_in_forced_native_code() {
    pinned_react_qu_copy_body(false);
}

#[test]
fn pinned_react_qu_copy_body_inherits_the_real_bundle_strict_context() {
    pinned_react_qu_copy_body(true);
}

fn pinned_react_qu_copy_body(strict: bool) {
    // Exact original Qu bytes107479..108041 from the pinned official React
    // complex-DOM bundle (SHA bfaa94e2...): synthetic fibers validate this
    // emission's eligibility, not actual application entry/admission or speed.
    // $u/Wu/Qu are from react-dom.production.min.js, React v17.0.2.
    // Upstream license: https://github.com/facebook/react/blob/v17.0.2/LICENSE
    // MIT License
    // Copyright (c) Facebook, Inc. and its affiliates.
    //
    // Permission is hereby granted, free of charge, to any person obtaining a copy
    // of this software and associated documentation files (the "Software"), to deal
    // in the Software without restriction, including without limitation the rights
    // to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    // copies of the Software, and to permit persons to whom the Software is
    // furnished to do so, subject to the following conditions:
    //
    // The above copyright notice and this permission notice shall be included in all
    // copies or substantial portions of the Software.
    //
    // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    // AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    // OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    // SOFTWARE.
    const SOURCE: &str = r#"
function $u(e,t,n,r){this.tag=e,this.key=n,this.sibling=this.child=this.return=this.stateNode=this.type=this.elementType=null,this.index=0,this.ref=null,this.pendingProps=t,this.dependencies=this.memoizedState=this.updateQueue=this.memoizedProps=null,this.mode=r,this.flags=0,this.lastEffect=this.firstEffect=this.nextEffect=null,this.childLanes=this.lanes=0,this.alternate=null}
function Wu(e,t,n,r){return new $u(e,t,n,r)}
function Qu(e,t){var n=e.alternate;return null===n?((n=Wu(e.tag,t,e.key,e.mode)).elementType=e.elementType,n.type=e.type,n.stateNode=e.stateNode,n.alternate=e,e.alternate=n):(n.pendingProps=t,n.type=e.type,n.flags=0,n.nextEffect=null,n.firstEffect=null,n.lastEffect=null),n.childLanes=e.childLanes,n.lanes=e.lanes,n.child=e.child,n.memoizedProps=e.memoizedProps,n.memoizedState=e.memoizedState,n.updateQueue=e.updateQueue,t=e.dependencies,n.dependencies=null===t?null:{lanes:t.lanes,firstContext:t.firstContext},n.sibling=e.sibling,n.index=e.index,n.ref=e.ref,n}
var subject=Qu,held={tag:1},fiber=Wu(0,null,null,0);
fiber.child=held;fiber.memoizedProps=held;fiber.memoizedState=held;fiber.updateQueue=held;
fiber.sibling=held;fiber.ref=held;fiber.childLanes=3;fiber.lanes=5;fiber.index=7;
"#;
    let source = if strict {
        format!("'use strict';{SOURCE}")
    } else {
        SOURCE.to_owned()
    };
    let (mut engine, function, chunk) = warmed(&source, "subject(fiber,held);subject(fiber,held)");
    assert_eq!(
        function.is_strict, strict,
        "lexical strictness belongs to the original module context"
    );
    let name_allowance = names::instruction_allowance(&chunk);
    let candidate_limit = chunk
        .jit_ops()
        .len()
        .saturating_mul(40)
        .saturating_add(name_allowance)
        .min(16_384);
    eprintln!("Qu strict={strict} candidate IR formula ops*40 +name_allowance({name_allowance}) capped16384 = {candidate_limit}");
    let baseline = compile_fixture(&mut engine, &chunk, false, None);
    let baseline_size = stores::TEST_BODY_SIZE.with(std::cell::Cell::get);
    ownership::TEST_OWNER_IR.with(|counts| counts.set([0; 9]));
    let compiled = compile_fixture(&mut engine, &chunk, true, None);
    let size = stores::TEST_BODY_SIZE.with(std::cell::Cell::get);
    let sites = generic_own::TEST_SITES.with(std::cell::Cell::get);
    let elisions = generic_own::TEST_ELISIONS.with(std::cell::Cell::get);
    eprintln!(
        "Qu strict={strict} owner builder-IR(retain/release/multi calls,instructions,blocks)={:?}",
        ownership::TEST_OWNER_IR.with(std::cell::Cell::get)
    );
    if active() {
        assert!(sites[0] > 0 && sites[1] > 0);
        assert!(elisions[0] > 0 && elisions[1] > 0);
        assert!(elisions[4] > 0 && elisions[5] > 0);
        assert!(size.0 < baseline_size.0);
        eprintln!("synthetic pinned Qu strict={strict} ops={} sites={sites:?} elisions={elisions:?} IR={} blocks={} bytes={}; baseline IR={} blocks={} bytes={}",
            chunk.jit_ops().len(),size.0,size.1,compiled.len,baseline_size.0,baseline_size.1,baseline.len);
    } else {
        assert_eq!(sites, [0; 2]);
    }
    let code = install(&mut engine, &chunk, compiled);
    assert_eq!(
        eval(
            &mut engine,
            r#"var result=subject(fiber,held);
        [result===fiber.alternate,result.child===held,result.memoizedProps===held,
        result.memoizedState===held,result.updateQueue===held,result.ref===held,
        result.pendingProps===held,result.childLanes,result.lanes,result.index].join('|')"#
        ),
        "true|true|true|true|true|true|true|3|5|7"
    );
    assert_eq!(chunk.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    finished(&mut engine, &code);
}
