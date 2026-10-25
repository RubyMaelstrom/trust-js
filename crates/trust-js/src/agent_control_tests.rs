//! Agent-wide scheduling/access integration. Precise collections occur only after JavaScript
//! has retired; in-loop probes observe ordinary production safepoints instead.
#![cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]

use crate::bytecode::Tier;
use crate::gc_core::CollectionKind;
use crate::interpreter::Interp;
use crate::value::{Gc, NativeCaptures, Property, Value, ValueView, WeakGc};
use crate::{Completion, Engine};
use std::cell::{Cell, RefCell};
use std::rc::Rc;

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("Agent fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn without_checkpoint(engine: &mut Engine, source: &str) -> Value {
    let body = crate::parser::parse_script(source, false)
        .unwrap_or_else(|error| panic!("Agent fixture parse: {}", error.message));
    match engine
        .interp
        .with_author_entry(|interp| interp.run_program(&body))
    {
        Ok(value) => value,
        Err(_) => panic!("Agent fixture evaluation failed"),
    }
}

fn global(interp: &mut Interp, name: &str) -> Value {
    let env = interp.global_env;
    interp
        .get_var(name, &env)
        .unwrap_or_else(|_| panic!("fixture global {name}"))
}

fn owner(engine: &mut Engine, name: &str) -> Box<WeakGc> {
    crate::gc_test::in_frame(|| {
        let object = global(&mut engine.interp, name)
            .as_obj()
            .expect("ShadowRealm object");
        Box::new(Gc::downgrade(&object))
    })
}

fn precise(engine: &mut Engine) {
    assert!(
        engine.interp.fn_frames.is_empty(),
        "no live JS frame at precise collection"
    );
    assert!(
        engine.interp.code_frames.is_empty(),
        "no live VM/JIT frame at precise collection"
    );
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
}

struct EndPolls(bool);
impl EndPolls {
    fn disabled() -> Self {
        Self(crate::jit::TEST_NO_END_POLLS.with(|flag| flag.replace(true)))
    }
}
impl Drop for EndPolls {
    fn drop(&mut self) {
        crate::jit::TEST_NO_END_POLLS.with(|flag| flag.set(self.0));
    }
}

#[derive(Default)]
struct LoopObservation {
    first: Option<u64>,
    native_before: bool,
    native_after_collection: bool,
}

fn active_code(interp: &Interp, name: &str) -> Option<Rc<crate::jit::JitCode>> {
    for frame in interp.fn_frames.iter().rev() {
        let callee = frame.callee();
        let object = callee.borrow();
        if let crate::value::Callable::User(user) = &object.call {
            if user.func.name.as_deref() == Some(name) {
                return user
                    .func
                    .runtime()
                    .execution_code()?
                    .as_ref()?
                    .jit
                    .get()?
                    .as_ref()
                    .cloned();
            }
        }
    }
    None
}

fn configure_tree(interp: &mut Interp, tier: Tier) {
    configure_observed_tree(
        interp,
        tier,
        Rc::new(RefCell::new(LoopObservation::default())),
    );
}

fn configure_observed_tree(
    interp: &mut Interp,
    tier: Tier,
    observation: Rc<RefCell<LoopObservation>>,
) {
    interp.tier = tier;
    interp.tier_threshold = 0;
    let observed = observation.clone();
    let probe = interp.make_native_closure(
        "collectionCount",
        0,
        Rc::new(move |current, _, _| {
            let collections = current.gc_heap.core.borrow().collections();
            let active = active_code(current, "churn")
                .is_some_and(|code| code.test_residency_marks(false).0 > 0);
            let mut observed = observed.borrow_mut();
            let first = *observed.first.get_or_insert(collections);
            observed.native_before |= active && collections == first;
            observed.native_after_collection |= active && collections > first;
            Ok(Value::Num(collections as f64))
        }),
        NativeCaptures::None,
    );
    interp
        .global
        .borrow_mut()
        .props
        .insert("collectionCount", Property::plain(Value::Obj(probe)));
    let ids: Vec<_> = interp
        .shadow_realms
        .values()
        .map(crate::agent_access::Owned::id)
        .collect();
    for id in ids {
        interp.with_route(id, |child| {
            configure_observed_tree(child, tier, observation.clone())
        });
    }
}

#[test]
fn agent_collections_finish_inside_child_and_grandchild_loops_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for grandchild in [false, true] {
            let _polls = EndPolls::disabled();
            let mut engine = Engine::new();
            engine.interp.gc_heap.agent.task_deferred.set(true);
            let body = "(function churn(n){var start=collectionCount(),during=0,last;\
                for(var j=0;j<n;j++){last={j:j};\
                  if(j%1000===0 && collectionCount()>start)during=1;}\
                return during*100000+last.j;})";
            let source = if grandchild {
                format!(
                    "var realm=new ShadowRealm;var run=realm.evaluate(\
                    'var nested=new ShadowRealm;var inner=nested.evaluate({body:?});\
                     (function(n){{return inner(n);}})');"
                )
            } else {
                format!("var realm=new ShadowRealm;var run=realm.evaluate({body:?});")
            };
            // JSON/Rust debug quoting contains only plain ASCII here.
            without_checkpoint(&mut engine, &source);
            let observed = Rc::new(RefCell::new(LoopObservation::default()));
            configure_observed_tree(&mut engine.interp, tier, observed.clone());
            {
                let mut heap = engine.interp.gc_heap.core.borrow_mut();
                heap.set_nursery_bytes(Some(256 << 10));
                heap.set_heap_growth(1 << 40, 100);
            }
            crate::jit::take_poll_census();
            assert_eq!(
                evaluate(
                    &mut engine,
                    "var parentKept={v:42};var answer=run(30000);answer+parentKept.v"
                ),
                "130041",
                "{tier:?}, grandchild={grandchild}"
            );
            let polls = crate::jit::take_poll_census();
            assert!(engine.interp.gc_heap.core.borrow().collections() > 0);
            if tier == Tier::Jit {
                let observed = observed.borrow();
                assert!(
                    observed.native_before && observed.native_after_collection,
                    "exact churn mapping must be active on both sides of the in-loop collection"
                );
                assert!(
                    polls.collections[crate::jit::POLL_LOOP as usize] > 0,
                    "the only source loop must collect through its machine loop poll"
                );
            }
        }
    }
}

#[test]
fn agent_promise_fifo_spans_parent_children_and_nested_children() {
    let mut engine = Engine::new();
    without_checkpoint(
        &mut engine,
        r#"
        var order=[]; function note(x){order.push(x);}
        var first=new ShadowRealm,second=new ShadowRealm;
        var schedule=first.evaluate(`
            var grand=new ShadowRealm;
            var nested=grand.evaluate('(function(note){Promise.resolve().then(()=>note("grand"));})');
            (function(note){Promise.resolve().then(()=>{note("first");
                Promise.resolve().then(()=>note("first-next"));});nested(note);})`);
        schedule(note);
        Promise.resolve().then(()=>{note('parent');Promise.resolve().then(()=>note('parent-next'));});
        second.evaluate('(function(note){Promise.resolve().then(()=>note("second"));})')(note);
        order.length
    "#,
    );
    assert_eq!(engine.interp.gc_heap.agent.work.borrow().promises.len(), 4);
    assert_eq!(
        global(&mut engine.interp, "order")
            .as_obj()
            .expect("order")
            .borrow()
            .props
            .get("length")
            .expect("array length")
            .value()
            .as_num_opt(),
        Some(0.0)
    );
    engine.interp.drain_microtasks();
    assert_eq!(
        evaluate(&mut engine, "order.join(',')"),
        "first,grand,parent,second,first-next,parent-next"
    );
    assert!(!engine.interp.has_pending_jobs());
}

#[test]
fn agent_queued_child_route_survives_then_retires_after_checkpoint() {
    let mut engine = Engine::new();
    without_checkpoint(
        &mut engine,
        r#"
        var ran=0,realm=new ShadowRealm;
        realm.evaluate('(function(done){Promise.resolve().then(()=>done(42));})')(x=>ran=x);
    "#,
    );
    let weak = owner(&mut engine, "realm");
    without_checkpoint(&mut engine, "realm=null;0");
    assert_eq!(engine.interp.gc_heap.agent.work.borrow().promises.len(), 1);
    precise(&mut engine);
    assert!(
        crate::gc_test::alive(weak.as_ref()),
        "runnable child work retains complete path"
    );
    engine.interp.drain_microtasks();
    assert_eq!(global(&mut engine.interp, "ran").as_num_opt(), Some(42.0));
    precise(&mut engine);
    assert!(
        !crate::gc_test::alive(weak.as_ref()),
        "finished route releases child owner"
    );
    assert!(engine.interp.shadow_realms.is_empty());
}

#[test]
fn agent_unresolved_intrinsic_import_bridge_does_not_root_discarded_child() {
    let mut engine = Engine::new();
    engine.set_module_loader(|_, _| {
        Some((
            "intrinsic-never".into(),
            "await new Promise(()=>{}); export const answer=42;".into(),
        ))
    });
    without_checkpoint(
        &mut engine,
        "var realm=new ShadowRealm;realm.importValue('never','answer');0",
    );
    let weak = owner(&mut engine, "realm");
    without_checkpoint(&mut engine, "realm=null;0");
    engine.interp.drain_microtasks();
    assert!(
        engine.interp.gc_heap.agent.work.borrow().imports.is_empty(),
        "no accepted external host work"
    );
    precise(&mut engine);
    assert!(
        !crate::gc_test::alive(weak.as_ref()),
        "intrinsic promise cycle is conditional"
    );
    assert!(engine.interp.shadow_realms.is_empty());
    assert!(engine.interp.routed_imports.is_empty());
}

#[test]
fn agent_external_import_routes_unique_completions_and_releases_child() {
    let mut engine = Engine::new();
    let requests = Rc::new(RefCell::new(Vec::new()));
    let captured = requests.clone();
    engine.set_async_dynamic_module_loader(move |id, specifier, _, _| {
        captured.borrow_mut().push((id, specifier.to_string()));
        true
    });
    without_checkpoint(
        &mut engine,
        r#"
        var result=false,realm=new ShadowRealm;
        realm.importValue('external','answer').then(value=>result=value===undefined);
    "#,
    );
    let weak = owner(&mut engine, "realm");
    without_checkpoint(&mut engine, "realm=null;0");
    assert_eq!(requests.borrow().len(), 1);
    assert_eq!(engine.interp.gc_heap.agent.work.borrow().imports.len(), 1);
    precise(&mut engine);
    assert!(
        crate::gc_test::alive(weak.as_ref()),
        "accepted request retains routed child"
    );
    let id = requests.borrow()[0].0;
    assert!(engine.finish_dynamic_module_load(
        id,
        Some((
            "external-module".into(),
            "export const answer=undefined;".into()
        ))
    ));
    assert!(
        !engine.finish_dynamic_module_load(id, None),
        "completed id is stale"
    );
    engine.interp.drain_microtasks();
    assert!(matches!(
        global(&mut engine.interp, "result").view(),
        ValueView::Bool(true)
    ));
    precise(&mut engine);
    assert!(
        !crate::gc_test::alive(weak.as_ref()),
        "fulfilled primitive import releases path"
    );
}

#[test]
fn agent_import_value_waits_for_tla_and_distinguishes_missing_undefined_and_rejection() {
    let mut engine = Engine::new();
    engine.set_module_loader(|specifier, _| {
        Some((
            specifier.into(),
            match specifier {
                "good" => "await Promise.resolve(); export const answer=undefined;",
                "bad" => "await Promise.reject({message:'child'}); export const answer=42;",
                _ => "export const other=1;",
            }
            .into(),
        ))
    });
    assert_eq!(
        evaluate(
            &mut engine,
            r#"
        var r=new ShadowRealm,results=[];
        r.importValue('good','answer').then(x=>results.push(x===undefined));
        r.importValue('missing','answer').catch(e=>results.push(e instanceof TypeError));
        r.importValue('bad','answer').catch(e=>results.push(e instanceof TypeError));'started'
    "#
        ),
        "started"
    );
    assert_eq!(
        evaluate(&mut engine, "results.length+':'+results.every(x=>x)"),
        "3:true"
    );
    assert!(engine.interp.routed_imports.is_empty());
}

#[cfg(feature = "embed")]
#[test]
fn agent_partial_job_pumps_keep_roots_until_empty_or_full_checkpoint() {
    struct JobRoots(Rc<Cell<usize>>);
    impl crate::host::HostTrace for JobRoots {
        fn trace_gc(&self, _: &mut dyn crate::host::HostGcVisitor) {}
    }
    impl crate::host::HostGc for JobRoots {
        fn end_job(&mut self) {
            self.0.set(self.0.get() + 1);
        }
        fn sweep_gc_with_native(&mut self, _: &dyn Fn(crate::host::NativeGcId) -> bool) {}
    }
    let mut engine = Engine::new();
    let ended = Rc::new(Cell::new(0));
    engine.interp.host_state.put(JobRoots(ended.clone()));
    engine.interp.host_state.register_gc::<JobRoots>();
    let weak = crate::gc_test::in_frame(|| {
        let symbol = engine
            .interp
            .new_symbol(Some(Rc::from("partial checkpoint")));
        let weak = crate::interpreter::WeakTarget::of(&symbol).expect("Symbol target");
        engine.interp.add_kept_object(symbol);
        Box::new(weak)
    });
    without_checkpoint(
        &mut engine,
        "var ran=0;Promise.resolve().then(()=>ran++);Promise.resolve().then(()=>ran++);0",
    );
    for expected in [1.0, 2.0] {
        assert!(engine.run_one_job_interruptible().expect("one job"));
        assert_eq!(
            global(&mut engine.interp, "ran").as_num_opt(),
            Some(expected)
        );
        assert_eq!(ended.get(), 0, "success is a partial pump");
        precise(&mut engine);
        assert!(crate::gc_test::alive(weak.as_ref()));
    }
    assert!(!engine.has_pending_jobs());
    assert!(
        !engine.interp.gc_heap.agent.kept_alive.borrow().is_empty(),
        "predicate is read-only"
    );
    assert!(!engine.run_one_job_interruptible().expect("empty pump"));
    assert_eq!(ended.get(), 1);
    assert!(engine.interp.gc_heap.agent.kept_alive.borrow().is_empty());
    precise(&mut engine);
    assert!(!crate::gc_test::alive(weak.as_ref()));
    engine.run_microtasks();
    assert_eq!(ended.get(), 2, "empty explicit checkpoint closes once too");
}

#[test]
fn agent_nested_checkpoints_defer_while_independent_engine_progresses() {
    let mut outer = Engine::new();
    let inner = Rc::new(RefCell::new(Engine::new()));
    // Creating `inner` made its Agent current; direct allocation below belongs to `outer`.
    outer.interp.activate_gc_heap();
    let captured = inner.clone();
    let called = Rc::new(Cell::new(0));
    let seen = called.clone();
    let native = outer.interp.make_native_closure(
        "nestedCheckpoint",
        0,
        Rc::new(move |interp, _, _| {
            assert_eq!(interp.gc_heap.agent.execution_depth.get(), 1);
            assert!(!interp.gc_heap.agent.kept_alive.borrow().is_empty());
            let queued = interp.gc_heap.agent.work.borrow().promises.len();
            interp.drain_microtasks();
            assert_eq!(
                interp.gc_heap.agent.work.borrow().promises.len(),
                queued,
                "same Agent deferred"
            );
            assert!(
                !interp.gc_heap.agent.kept_alive.borrow().is_empty(),
                "a deferred nested checkpoint keeps the outer job's KeptAlive epoch"
            );
            assert_eq!(
                evaluate(
                    &mut captured.borrow_mut(),
                    "var ran=0;Promise.resolve().then(()=>ran=42);0"
                ),
                "0"
            );
            assert_eq!(
                global(&mut captured.borrow_mut().interp, "ran").as_num_opt(),
                Some(42.0)
            );
            seen.set(seen.get() + 1);
            Ok(Value::Undefined)
        }),
        NativeCaptures::None,
    );
    outer
        .interp
        .global
        .borrow_mut()
        .props
        .insert("nestedCheckpoint", Property::plain(Value::Obj(native)));
    without_checkpoint(
        &mut outer,
        r#"
        var ran=0,realm=new ShadowRealm,weak=new WeakRef({});weak.deref();
        Promise.resolve().then(()=>ran=42);
        realm.evaluate('(function(callback){callback();})')(nestedCheckpoint);0
    "#,
    );
    assert_eq!(called.get(), 1);
    assert_eq!(global(&mut outer.interp, "ran").as_num_opt(), Some(0.0));
    outer.interp.drain_microtasks();
    assert_eq!(global(&mut outer.interp, "ran").as_num_opt(), Some(42.0));
}

#[test]
fn agent_cleanup_keeps_registry_realm_with_changed_prototype_and_revoked_callback() {
    let mut engine = Engine::new();
    without_checkpoint(
        &mut engine,
        r#"
        var realmB=$262.createRealm().global,realmC=$262.createRealm().global;
        var Registry=realmB.FinalizationRegistry,callback=realmC.Function('held','return held');
        var registry=new Registry(callback);Object.setPrototypeOf(registry,null);
    "#,
    );
    let b = crate::gc_test::in_frame(|| {
        Box::new(Gc::downgrade(
            &global(&mut engine.interp, "realmB")
                .as_obj()
                .expect("B global"),
        ))
    });
    let c = crate::gc_test::in_frame(|| {
        Box::new(Gc::downgrade(
            &global(&mut engine.interp, "realmC")
                .as_obj()
                .expect("C global"),
        ))
    });
    without_checkpoint(&mut engine, "realmB=realmC=Registry=callback=null;0");
    precise(&mut engine);
    assert!(
        crate::gc_test::alive(b.as_ref()),
        "registry creation Realm is an internal slot"
    );
    assert!(
        crate::gc_test::alive(c.as_ref()),
        "callback entry Realm remains retained"
    );
    without_checkpoint(&mut engine, "registry=null;0");
    engine.interp.drain_microtasks();
    precise(&mut engine);
    assert!(!crate::gc_test::alive(b.as_ref()));
    assert!(!crate::gc_test::alive(c.as_ref()));
    assert_eq!(evaluate(&mut engine,"var p=Proxy.revocable(function(){},{});p.revoke();new FinalizationRegistry(p.proxy);true"),"true");
}

#[test]
fn agent_route_and_entry_guards_restore_after_host_unwind() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Interp);
    let native = engine
        .interp
        .make_native("panicInside", 0, |_, _, _| -> Result<Value, Value> {
            panic!("host fixture unwind")
        });
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("panicInside", Property::plain(Value::Obj(native)));
    without_checkpoint(&mut engine,"var realm=new ShadowRealm;var enter=realm.evaluate('(function(f){function deep(){f();}deep();})');0");
    configure_tree(&mut engine.interp, Tier::Interp);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        without_checkpoint(&mut engine, "enter(panicInside)");
    }));
    assert!(result.is_err());
    assert_eq!(engine.interp.gc_heap.agent.execution_depth.get(), 0);
    assert!(!engine.interp.gc_heap.agent.checkpoint.get());
    assert!(engine.interp.fn_frames.is_empty());
    assert!(engine.interp.code_frames.is_empty());
    let child = global(&mut engine.interp, "realm").as_obj().expect("realm");
    engine
        .interp
        .with_shadow(&child, |sub| {
            assert!(sub.fn_frames.is_empty());
            assert!(sub.code_frames.is_empty());
            assert_eq!(sub.depth, 0);
        })
        .expect("owned child");
    assert_eq!(evaluate(&mut engine, "realm.evaluate('6*7')"), "42");
}

#[test]
fn agent_render_and_error_coercions_defer_reentrant_checkpoints() {
    let mut engine = Engine::new();
    let probe = engine
        .interp
        .make_native("tryCheckpoint", 0, |interp, _, _| {
            assert_eq!(interp.gc_heap.agent.execution_depth.get(), 1);
            let before = interp.gc_heap.agent.work.borrow().promises.len();
            interp.drain_microtasks();
            assert_eq!(interp.gc_heap.agent.work.borrow().promises.len(), before);
            Ok(Value::Undefined)
        });
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("tryCheckpoint", Property::plain(Value::Obj(probe)));
    assert_eq!(
        evaluate(
            &mut engine,
            r#"var ran=0;({toString(){
        Promise.resolve().then(()=>ran=42);tryCheckpoint();return 'formatted';}})"#
        ),
        "formatted"
    );
    assert_eq!(global(&mut engine.interp, "ran").as_num_opt(), Some(0.0));
    assert_eq!(engine.interp.gc_heap.agent.work.borrow().promises.len(), 1);
    engine.interp.drain_microtasks();
    assert_eq!(global(&mut engine.interp, "ran").as_num_opt(), Some(42.0));
    match engine
        .eval(
            r#"throw {get name(){Promise.resolve().then(()=>ran=84);
        tryCheckpoint();return 'Custom';},message:'failure'}"#,
            false,
        )
        .expect("throw fixture")
    {
        Completion::Throw { name, message } => {
            assert_eq!((name.as_str(), message.as_str()), ("Custom", "failure"))
        }
        _ => panic!("throw completion"),
    }
    assert_eq!(global(&mut engine.interp, "ran").as_num_opt(), Some(42.0));
    engine.interp.drain_microtasks();
    assert_eq!(global(&mut engine.interp, "ran").as_num_opt(), Some(84.0));
}

#[test]
fn agent_normal_and_throw_completions_remain_roots_across_checkpoint_gc() {
    let mut engine = Engine::new();
    let collect = engine.interp.make_native("collectNow", 0, |interp, _, _| {
        interp.gc_collect();
        Ok(Value::Undefined)
    });
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("collectNow", Property::plain(Value::Obj(collect)));
    assert_eq!(
        evaluate(
            &mut engine,
            r#"var value={a:42,toString(){return this.a;}};
        Promise.resolve().then(()=>{value=null;collectNow();});value"#
        ),
        "42"
    );
    match engine
        .eval(
            r#"var error={name:'RootedError',message:'retained'};
        Promise.resolve().then(()=>{error=null;collectNow();});throw error"#,
            false,
        )
        .expect("throw fixture")
    {
        Completion::Throw { name, message } => assert_eq!(
            (name.as_str(), message.as_str()),
            ("RootedError", "retained")
        ),
        _ => panic!("rooted throw completion"),
    }
}

#[cfg(feature = "embed")]
#[test]
fn agent_nested_public_entry_restores_floors_when_host_catches_unwind() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Interp);
    let panic = engine
        .interp
        .make_native("nestedPanic", 0, |_, _, _| -> Result<Value, Value> {
            panic!("nested author entry host unwind")
        });
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("nestedPanic", Property::plain(Value::Obj(panic)));
    let recovered = Rc::new(Cell::new(0));
    let observed = recovered.clone();
    let catch = engine.interp.make_native_closure(
        "catchInner",
        0,
        Rc::new(move |interp, _, _| {
            let floors = (
                interp.fn_frames.len(),
                interp.code_frames.len(),
                interp.using_stack.len(),
                interp.depth,
            );
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                // SAFETY: same active native scope. No value or reference escapes this nested entry.
                let cx = unsafe { crate::scope::Ctx::new(interp) };
                let _ = cx.eval_value("function nested(){nestedPanic();} nested();");
            }));
            assert!(result.is_err());
            assert_eq!(
                (
                    interp.fn_frames.len(),
                    interp.code_frames.len(),
                    interp.using_stack.len(),
                    interp.depth
                ),
                floors
            );
            assert_eq!(
                interp.gc_heap.agent.execution_depth.get(),
                1,
                "outer author span survives"
            );
            interp.gc_collect();
            // A second public call through the same current borrow must still work after the catch.
            let result = unsafe { crate::scope::Ctx::new(interp) }.eval_value("6*7");
            assert!(matches!(result, Ok(Ok(value)) if value.as_num_opt() == Some(42.0)));
            observed.set(observed.get() + 1);
            Ok(Value::Num(42.0))
        }),
        NativeCaptures::None,
    );
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("catchInner", Property::plain(Value::Obj(catch)));
    assert_eq!(
        evaluate(
            &mut engine,
            "function outer(){return catchInner();} outer()"
        ),
        "42"
    );
    assert_eq!(recovered.get(), 1);
    assert_eq!(engine.interp.gc_heap.agent.execution_depth.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
}

#[test]
fn agent_limit_and_next_threshold_changes_reach_same_warmed_native_parent_poll() {
    let _polls = EndPolls::disabled();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine.interp.gc_heap.agent.task_deferred.set(true);
    let selected: Rc<RefCell<Option<Rc<crate::jit::JitCode>>>> = Rc::new(RefCell::new(None));
    let phase = Rc::new(Cell::new(0u8));
    let before_collection = Rc::new(Cell::new(0));
    let observed = Rc::new(Cell::new((0, 0)));
    let code = selected.clone();
    let current_phase = phase.clone();
    let before = before_collection.clone();
    let seen = observed.clone();
    let change = engine.interp.make_native_closure(
        "changeAgentPolicy",
        0,
        Rc::new(move |interp, _, _| {
            let phase = current_phase.get();
            if phase == 0 {
                return Ok(Value::Undefined);
            }
            assert!(
                code.borrow()
                    .as_ref()
                    .expect("selected parent mapping")
                    .test_residency_marks(false)
                    .0
                    > 0,
                "the exact warmed parent is suspended natively while the child calls up"
            );
            seen.set((seen.get().0 + 1, seen.get().1));
            match phase {
                1 => interp.set_agent_limit(100_000),
                2 => interp.set_agent_limit(200_000),
                3 => interp.set_agent_next(0), // Force the existing-poll collection/rearm path.
                _ => unreachable!(),
            }
            let expected = interp.gc_heap.agent.next.get();
            let access = interp.gc_heap.agent.access.clone();
            interp.with_raw_access(|_| {
                for (_, pointer) in access.snapshot() {
                    assert_eq!(
                        unsafe { (*pointer.as_ptr()).gc_next },
                        expected,
                        "all paused/inactive native caches mirror policy"
                    );
                }
            });
            before.set(interp.gc_heap.core.borrow().collections());
            Ok(Value::Undefined)
        }),
        NativeCaptures::None,
    );
    let code = selected.clone();
    let current_phase = phase.clone();
    let before = before_collection.clone();
    let seen = observed.clone();
    let verify = engine.interp.make_native_closure(
        "verifyAgentPolicy",
        0,
        Rc::new(move |interp, _, _| {
            if current_phase.get() == 0 {
                return Ok(Value::Undefined);
            }
            assert!(
                code.borrow()
                    .as_ref()
                    .expect("selected parent mapping")
                    .test_residency_marks(false)
                    .0
                    > 0,
                "the same parent mapping resumed and reached the post-loop call"
            );
            seen.set((seen.get().0, seen.get().1 + 1));
            if current_phase.get() == 3 {
                assert!(
                    interp.gc_heap.core.borrow().collections() > before.get(),
                    "existing loop poll serviced the forced next-threshold request"
                );
                assert_eq!(
                    interp.gc_heap.agent.next.get(),
                    interp.gc_heap.agent.limit.get()
                );
            }
            Ok(Value::Undefined)
        }),
        NativeCaptures::None,
    );
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("changeAgentPolicy", Property::plain(Value::Obj(change)));
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("verifyAgentPolicy", Property::plain(Value::Obj(verify)));
    without_checkpoint(
        &mut engine,
        r#"
        var realm=new ShadowRealm;
        var childHook=realm.evaluate('(function(callback){return function(){callback();};})')(changeAgentPolicy);
        function sameNativeLoop(n){var sum=0;for(var j=0;j<n;j++){
            sum+=j&7;if(j===1000)childHook();}verifyAgentPolicy();return sum;}
        sameNativeLoop(40000)
    "#,
    );
    let selected_code = crate::gc_test::in_frame(|| {
        let function = global(&mut engine.interp, "sameNativeLoop")
            .as_obj()
            .expect("parent function");
        let object = function.borrow();
        let crate::value::Callable::User(user) = &object.call else {
            panic!("user parent")
        };
        user.func
            .runtime()
            .execution_code()
            .and_then(Option::as_ref)
            .and_then(|chunk| chunk.jit.get().flatten())
            .expect("warmed native mapping")
    });
    *selected.borrow_mut() = Some(selected_code);
    for p in [1, 2, 3] {
        phase.set(p);
        crate::jit::take_poll_census();
        assert_eq!(evaluate(&mut engine, "sameNativeLoop(40000)"), "140000");
        let census = crate::jit::take_poll_census();
        if p == 3 {
            assert!(
                census.collections[crate::jit::POLL_LOOP as usize] > 0,
                "wrappers have no loops; the selected parent loop must perform the collection"
            );
        }
    }
    assert_eq!(observed.get(), (3, 3));
    assert_eq!(engine.interp.gc_heap.agent.limit.get(), 200_000);
    assert_eq!(engine.interp.gc_next, 200_000);
}

#[cfg(feature = "embed")]
#[test]
fn agent_cleanup_task_eligibility_uses_registry_context_and_whole_loop_callback_settings() {
    #[derive(Default)]
    struct Hooks {
        entered: Vec<u64>,
        left: Vec<u64>,
        ended: usize,
    }
    impl crate::host::HostTrace for Hooks {
        fn trace_gc(&self, _: &mut dyn crate::host::HostGcVisitor) {}
    }
    impl crate::host::HostGc for Hooks {
        fn end_job(&mut self) {
            self.ended += 1;
        }
        fn sweep_gc_with_native(&mut self, _: &dyn Fn(crate::host::NativeGcId) -> bool) {}
    }
    fn enter(cx: &mut crate::scope::Ctx<'_>, context: u64) {
        cx.host_mut::<Hooks>().expect("hooks").entered.push(context);
    }
    fn leave(cx: &mut crate::scope::Ctx<'_>) {
        let context = cx.host_job_context();
        cx.host_mut::<Hooks>().expect("hooks").left.push(context);
    }
    for cancel_registry in [false, true] {
        let mut engine = Engine::new();
        let heap = engine.interp.gc_heap.clone();
        let no_gc = { heap.core.borrow().nogc() };
        engine.set_job_callback_script_caller_capture(true);
        let a = engine.interp.global;
        let b = engine.interp.create_embed_realm();
        let c = engine.interp.create_embed_realm();
        let realms = crate::rooting::Rooted::new((Value::Obj(a), b, c));
        let ctor = engine
            .interp
            .with_embed_realm(&realms.1, |sub| {
                sub.set_host_job_context(22);
                global(sub, "FinalizationRegistry")
            })
            .unwrap_or_else(|_| panic!("registry Realm B"));
        let callbacks = Rc::new(Cell::new(0));
        let observed = callbacks.clone();
        let callback=engine.interp.with_embed_realm(&realms.2,|sub| {
            sub.set_host_job_context(23);
            let probe=sub.make_native_closure("checkCleanupBoundary",0,Rc::new(move|interp,_,_| {
                assert_eq!(interp.host_job_context,23,"callback settings C prepared once around loop");
                assert_eq!(interp.host::<Hooks>().expect("hooks").ended,0,"no boundary between cleanup cells");
                assert!(!interp.gc_heap.agent.kept_alive.borrow().is_empty(),"both callbacks share the kept epoch");
                let queued=interp.gc_heap.agent.work.borrow().promises.len();
                interp.drain_microtasks();
                assert_eq!(interp.gc_heap.agent.work.borrow().promises.len(),queued,"cleanup cannot nest a checkpoint");
                observed.set(observed.get()+1);Ok(Value::Undefined)
            }),NativeCaptures::None);
            sub.global.borrow_mut().props.insert("checkCleanupBoundary",Property::plain(Value::Obj(probe)));
            let body=crate::parser::parse_script(r#"
                var events=[],count=0,same=false,weak;
                function cleanup(held){count++;if(count===1)weak=new WeakRef(Symbol('cleanup epoch'));
                    else same=weak.deref()!==undefined;checkCleanupBoundary();events.push('callback');
                    Promise.resolve().then(()=>events.push('promise'));}
            "#,false).unwrap_or_else(|e|panic!("callback parse {}",e.message));
            match sub.with_author_entry(|sub|sub.run_program(&body)) {Ok(_)=>{},Err(_)=>panic!("callback setup")}
            global(sub,"cleanup")
        }).unwrap_or_else(|_|panic!("callback Realm C"));
        engine
            .interp
            .global
            .borrow_mut()
            .props
            .insert("OtherRegistry", Property::plain(ctor));
        engine
            .interp
            .global
            .borrow_mut()
            .props
            .insert("otherCleanup", Property::plain(callback));
        engine.interp.set_host_job_context(11);
        without_checkpoint(
            &mut engine,
            r#"
            function registerInA(){var registry=new OtherRegistry(otherCleanup);
                registry.register({},1);registry.register({},2);globalThis.keptRegistry=registry;}
            registerInA();0
        "#,
        );
        let registry = engine
            .interp
            .get_member_key(&realms.0, "keptRegistry")
            .unwrap_or_else(|_| panic!("registry"));
        let ptr = registry
            .as_obj()
            .map(|object| Gc::as_ptr(&object) as usize)
            .expect("registry object");
        {
            let state = engine
                .interp
                .finalization_registries
                .get(&ptr)
                .expect("registry slots");
            assert!(Gc::ptr_eq(
                &state.registry_realm,
                &realms.1.as_obj().expect("B")
            ));
            assert_eq!(
                state.registry_context, 22,
                "creation Realm context B, not registration A"
            );
            assert_eq!(
                state.callback_realm,
                realms.2.as_obj().map(|object| Gc::as_ptr(&object) as usize)
            );
            assert_eq!(state.callback_context, 23);
            assert!(
                state
                    .cleanup_callback
                    .script_caller
                    .is_some_and(|global| Gc::ptr_eq(&global, &a)),
                "incumbent A captured separately"
            );
        }
        engine.interp.set_host_job_context(0);
        engine.interp.host_state.put(Hooks::default());
        engine.interp.host_state.register_gc::<Hooks>();
        engine.set_host_job_context_hooks(enter, leave);
        assert!(
            engine.interp.gc_heap.agent.work.borrow().cleanup.is_empty(),
            "fixture setup cannot schedule early"
        );
        if cancel_registry {
            assert!(engine.release_host_job_context(22));
        } else {
            assert!(
                engine.release_host_job_context(11),
                "retire only registration/incumbent A"
            );
        }
        drop(no_gc);
        precise(&mut engine);
        if cancel_registry {
            assert!(
                engine.interp.gc_heap.agent.work.borrow().cleanup.is_empty(),
                "a GC must not enqueue a new task for the destroyed registry context"
            );
            precise(&mut engine);
            assert!(engine.interp.gc_heap.agent.work.borrow().cleanup.is_empty());
            // Independently exercise dispatch eligibility for a stale task record, while the
            // distinct live callback context C remains available.
            engine.interp.enqueue_test_cleanup(registry);
        } else {
            assert_eq!(
                engine.interp.gc_heap.agent.work.borrow().cleanup.len(),
                1,
                "one whole-loop task survives the distinct retired incumbent A"
            );
        }
        engine.interp.drain_microtasks();
        assert_eq!(callbacks.get(), if cancel_registry { 0 } else { 2 });
        let hooks = engine.interp.host::<Hooks>().expect("hooks");
        if cancel_registry {
            assert!(hooks.entered.is_empty());
        } else {
            // Entering callback Realm C already installs its settings context 23; the existing
            // contract (tests.rs cross_realm_promise_jobs_prepare_the_handler_realm_before_host_settings)
            // runs no legacy enter/leave switch in that case.
            assert!(hooks.entered.is_empty());
            assert!(hooks.left.is_empty());
            let result = engine
                .interp
                .with_embed_realm(&realms.2, |sub| {
                    let body = crate::parser::parse_script("events.join(',')+':'+same", false)
                        .unwrap_or_else(|e| panic!("result parse {}", e.message));
                    sub.with_author_entry(|sub| sub.run_program(&body))
                })
                .unwrap_or_else(|_| panic!("callback Realm remains available"));
            let Ok(result) = result else {
                panic!("callback result")
            };
            assert_eq!(
                result.as_engine_text(),
                Some("callback,callback,promise,promise:true")
            );
        }
        assert_eq!(
            engine.interp.host::<Hooks>().expect("hooks").ended,
            1,
            "one outer checkpoint boundary"
        );
    }
}

#[test]
fn agent_busy_core_deferral_preserves_pending_major_over_explicit_minor() {
    use crate::agent_control::CollectionOutcome;
    let mut engine = Engine::new();
    let heap = engine.interp.gc_heap.clone();
    heap.agent.pending.set(Some(CollectionKind::Major));
    let (before_all, before_minor) = {
        let core = heap.core.borrow();
        (core.collections(), core.minor_collections())
    };
    let busy = heap.core.borrow_mut();
    assert_eq!(
        engine
            .interp
            .gc_heap_collect(crate::value::GcCause::Explicit, Some(CollectionKind::Minor)),
        CollectionOutcome::Deferred
    );
    assert_eq!(heap.agent.pending.get(), Some(CollectionKind::Major));
    drop(busy);
    assert_eq!(
        engine
            .interp
            .gc_heap_collect(crate::value::GcCause::Explicit, Some(CollectionKind::Minor)),
        CollectionOutcome::Collected(CollectionKind::Major)
    );
    let core = heap.core.borrow();
    assert_eq!(core.collections(), before_all + 1);
    assert_eq!(core.minor_collections(), before_minor);
}

#[test]
fn agent_newly_adopted_children_inherit_canonical_next_threshold() {
    let mut engine = Engine::new();
    engine.interp.set_agent_limit(200_000);
    let heap = engine.interp.gc_heap.clone();
    let suppressed = crate::agent_control::Suppression::new(&heap);
    engine.interp.set_agent_next(250_000);
    without_checkpoint(&mut engine, "var inherited=new ShadowRealm;0");
    let realm = global(&mut engine.interp, "inherited")
        .as_obj()
        .expect("child");
    engine
        .interp
        .with_shadow(&realm, |child| {
            assert_eq!(child.gc_next, 250_000);
            assert_eq!(child.gc_heap.agent.next.get(), 250_000);
            // The unchanged-value fast path must be valid even from this just-adopted member.
            child.set_agent_next(250_000);
            assert_eq!(child.gc_next, 250_000);
        })
        .expect("adopted child");
    assert_eq!(engine.interp.gc_next, 250_000);
    drop(suppressed);
}

#[cfg(feature = "embed")]
#[test]
fn agent_import_authority_survives_tla_later_calls_and_cancels_without_child_settings_copy() {
    let mut engine = Engine::new();
    engine.set_host_job_context(17);
    engine.set_import_base("https://authority17.example/host.js");
    struct ParentOnly;
    engine.interp.host_state.put(ParentOnly);
    let requests = Rc::new(RefCell::new(Vec::new()));
    let captured = requests.clone();
    engine.set_async_dynamic_module_loader_with_context(
        move |id, context, specifier, referrer, _| {
            captured
                .borrow_mut()
                .push((id, context, specifier.to_string(), referrer.to_string()));
            true
        },
    );
    let static_referrers = Rc::new(RefCell::new(Vec::new()));
    let captured = static_referrers.clone();
    engine.set_module_loader(move |specifier, referrer| {
        captured
            .borrow_mut()
            .push((specifier.to_string(), referrer.to_string()));
        (specifier == "static").then(|| {
            (
                "https://authority17.example/static.js".into(),
                "export const staticValue=1;".into(),
            )
        })
    });
    without_checkpoint(
        &mut engine,
        "globalThis.realm=new ShadowRealm;globalThis.later=null;0",
    );
    let weak = owner(&mut engine, "realm");
    let environment = crate::gc_test::HiddenAddress::from_fn(|| {
        let object = global(&mut engine.interp, "realm").as_obj().expect("realm");
        engine
            .interp
            .with_shadow(&object, |child| {
                assert_eq!(child.host_job_context, 0);
                assert!(child.host::<ParentOnly>().is_none());
                assert!(child.host_job_context_enter.is_none());
                crate::interpreter::Env::as_ptr(&child.global_env) as usize
            })
            .expect("owned child")
    });
    without_checkpoint(
        &mut engine,
        "realm.importValue('entry','later').then(fn=>globalThis.later=fn);0",
    );
    assert_eq!(requests.borrow().len(), 1);
    assert_eq!(
        requests.borrow()[0].1,
        17,
        "loader authority is requester token, not child zero"
    );
    let entry = requests.borrow()[0].0;
    // Move the public Engine while its symbolic root route owns accepted external work.
    let mut engine = Box::new(engine);
    assert!(engine.finish_dynamic_module_load(entry,Some(("https://authority17.example/entry.js".into(),r#"
        import {staticValue} from 'static';
        await import('awaited');
        export function later(specifier='late'){import(specifier).then(ns=>globalThis.lateValue=ns.x);return 'requested';}
    "#.into()))));
    engine.interp.drain_microtasks();
    assert!(
        matches!(global(&mut engine.interp, "later").view(), ValueView::Null),
        "TLA has not fulfilled importValue"
    );
    assert_eq!(
        static_referrers.borrow().as_slice(),
        &[(
            "static".to_string(),
            "https://authority17.example/entry.js".to_string()
        )]
    );
    assert_eq!(requests.borrow()[1].1, 17);
    assert_eq!(requests.borrow()[1].2, "awaited");
    assert_eq!(
        requests.borrow()[1].3,
        "https://authority17.example/entry.js"
    );
    let awaited = requests.borrow()[1].0;
    assert!(engine.finish_dynamic_module_load(
        awaited,
        Some((
            "https://authority17.example/awaited.js".into(),
            "export const x=42;".into()
        ))
    ));
    engine.interp.drain_microtasks();
    engine.set_host_job_context(23);
    engine.set_import_base("https://authority23.example/unrelated.js");
    assert_eq!(evaluate(&mut engine, "later()"), "requested");
    assert_eq!(
        requests.borrow()[2].1,
        17,
        "later exported call keeps actual originating code authority"
    );
    assert_eq!(requests.borrow()[2].2, "late");
    assert_eq!(
        requests.borrow()[2].3,
        "https://authority17.example/entry.js"
    );
    let late = requests.borrow()[2].0;
    assert!(engine.finish_dynamic_module_load(
        late,
        Some((
            "https://authority17.example/late.js".into(),
            "export const x=84;".into()
        ))
    ));
    engine.interp.drain_microtasks();
    let object = global(&mut engine.interp, "realm").as_obj().expect("realm");
    engine
        .interp
        .with_shadow(&object, |child| {
            assert_eq!(
                child.host_job_context, 0,
                "load authorization never becomes execution settings"
            );
            assert_eq!(
                crate::interpreter::Env::as_ptr(&child.global_env) as usize,
                environment.get()
            );
            assert!(child.host::<ParentOnly>().is_none());
            assert_eq!(global(child, "lateValue").as_num_opt(), Some(84.0));
        })
        .expect("child");
    assert_eq!(
        evaluate(&mut engine, "realm.evaluate(\"import('ordinary');0\")"),
        "0"
    );
    assert_eq!(
        requests.borrow()[3].1,
        0,
        "delegation cannot leak into unrelated child evaluate"
    );
    let ordinary = requests.borrow()[3].0;
    assert!(engine.finish_dynamic_module_load(
        ordinary,
        Some((
            "https://child.example/ordinary.js".into(),
            "export const value=0;".into()
        ))
    ));
    engine.interp.drain_microtasks();
    // A repeated request from the same Module Record reuses its loaded module, so each
    // remaining check uses a fresh specifier that must reach the loader.
    assert_eq!(
        evaluate(&mut engine, "later('late-cancelled')"),
        "requested"
    );
    assert_eq!(requests.borrow()[4].2, "late-cancelled");
    let cancelled = requests.borrow()[4].0;
    assert!(engine.release_host_job_context(17));
    assert!(
        !engine.finish_dynamic_module_load(cancelled, None),
        "origin retirement cancels child-dispatched host request"
    );
    let count = requests.borrow().len();
    assert_eq!(evaluate(&mut engine, "later('late-retired')"), "requested");
    assert_eq!(
        requests.borrow().len(),
        count,
        "retired source authority never falls back to current context23/zero"
    );
    without_checkpoint(&mut engine, "globalThis.realm=null;globalThis.later=null;0");
    engine.interp.drain_microtasks();
    precise(&mut engine);
    assert!(
        !crate::gc_test::alive(weak.as_ref()),
        "cancellation releases both routes and intrinsic graph"
    );
}

#[cfg(feature = "embed")]
#[test]
fn agent_concurrent_import_authorities_keep_same_url_graphs_distinct() {
    let mut engine = Engine::new();
    let requests = Rc::new(RefCell::new(Vec::new()));
    let captured = requests.clone();
    engine.set_async_dynamic_module_loader_with_context(move |id, context, specifier, _, _| {
        captured
            .borrow_mut()
            .push((id, context, specifier.to_string()));
        true
    });
    without_checkpoint(
        &mut engine,
        "globalThis.realm=new ShadowRealm;globalThis.answers=[];0",
    );
    for context in [17, 23] {
        engine.set_host_job_context(context);
        without_checkpoint(
            &mut engine,
            &format!("realm.importValue('same','answer').then(x=>answers.push(x));{context}"),
        );
    }
    assert_eq!(
        requests
            .borrow()
            .iter()
            .map(|request| request.1)
            .collect::<Vec<_>>(),
        vec![17, 23]
    );
    let pending = requests.borrow().clone();
    for (id, context, _) in pending.iter().rev() {
        assert!(engine.finish_dynamic_module_load(
            *id,
            Some((
                "https://shared.example/same.js".into(),
                format!("export const answer={context};")
            ))
        ));
    }
    engine.interp.drain_microtasks();
    assert_eq!(
        evaluate(&mut engine, "answers.sort((a,b)=>a-b).join(',')"),
        "17,23"
    );
    let object = global(&mut engine.interp, "realm").as_obj().expect("realm");
    engine
        .interp
        .with_shadow(&object, |child| {
            assert_eq!(child.host_job_context, 0);
            for context in [17, 23] {
                assert!(child
                    .module_recs
                    .keys()
                    .any(|key| crate::modules::module_map_context(key) == context));
            }
        })
        .expect("child");
    assert!(engine.release_host_job_context(17));
    engine
        .interp
        .with_shadow(&object, |child| {
            assert!(!child
                .module_recs
                .keys()
                .any(|key| crate::modules::module_map_context(key) == 17));
            assert!(child
                .module_recs
                .keys()
                .any(|key| crate::modules::module_map_context(key) == 23));
        })
        .expect("child");
}

#[cfg(feature = "embed")]
#[test]
fn agent_upward_null_source_eval_and_function_use_actual_child_module_authority() {
    let mut engine = Engine::new();
    let requests = Rc::new(RefCell::new(Vec::new()));
    let captured = requests.clone();
    engine.set_async_dynamic_module_loader_with_context(
        move |id, context, specifier, referrer, _| {
            captured
                .borrow_mut()
                .push((id, context, specifier.to_string(), referrer.to_string()));
            true
        },
    );
    let foreign_requests = Rc::new(RefCell::new(Vec::new()));
    let foreign = Rc::new(RefCell::new(Engine::new()));
    foreign.borrow_mut().set_host_job_context(77);
    let captured_foreign = foreign_requests.clone();
    foreign
        .borrow_mut()
        .set_async_dynamic_module_loader_with_context(move |id, context, specifier, _, _| {
            captured_foreign
                .borrow_mut()
                .push((id, context, specifier.to_string()));
            true
        });
    let other = foreign.clone();
    // Creating `foreign` made its Agent current; direct allocation below belongs to `engine`.
    engine.interp.activate_gc_heap();
    let independent = engine.interp.make_native_closure(
        "independent",
        0,
        Rc::new(move |interp, _, _| {
            let expected = interp.load_authority();
            without_checkpoint(&mut other.borrow_mut(), "import('independent-agent');0");
            other.borrow_mut().interp.drain_microtasks();
            assert_eq!(
                interp.load_authority(),
                expected,
                "independent Agent cannot replace caller authority"
            );
            Ok(Value::Undefined)
        }),
        NativeCaptures::None,
    );
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("independent", Property::plain(Value::Obj(independent)));
    engine.set_host_job_context(17);
    engine.set_import_base("https://origin17.example/host.js");
    without_checkpoint(&mut engine,"globalThis.realm=new ShadowRealm;globalThis.invokeDerived=null;realm.importValue('entry','derive').then(fn=>globalThis.invokeDerived=fn);0");
    let entry = requests.borrow()[0].0;
    assert!(engine.finish_dynamic_module_load(
        entry,
        Some((
            "https://origin17.example/module.js".into(),
            r#"
        export function derive(parentFunction,parentEval,independent){
            independent();
            const generated=parentFunction("import('from-constructor');return 1;");
            // A wrapped call may return only primitives or callables; complete with 0, not the Promise.
            parentEval("import('from-eval');0");generated();return 'derived';
        }
    "#
            .into()
        ))
    ));
    engine.interp.drain_microtasks();
    engine.set_host_job_context(23);
    engine.set_import_base("https://ambient23.example/older-parent.js");
    assert_eq!(
        evaluate(&mut engine, "invokeDerived(Function,eval,independent)"),
        "derived"
    );
    assert_eq!(
        foreign_requests
            .borrow()
            .iter()
            .map(|record| record.1)
            .collect::<Vec<_>>(),
        vec![77]
    );
    let captured = requests.borrow();
    assert_eq!(captured.len(), 3);
    for request in &captured[1..] {
        assert_eq!(
            request.1, 17,
            "suspended older parent source must not win null-source up-call"
        );
        assert_eq!(request.3, "https://origin17.example/module.js");
    }
    assert_eq!(captured[1].2, "from-eval");
    assert_eq!(captured[2].2, "from-constructor");
    drop(captured);
    without_checkpoint(&mut engine, "import('ambient-after-return');0");
    let last = requests.borrow().last().cloned().expect("ambient request");
    assert_eq!(
        (last.1, last.2.as_str(), last.3.as_str()),
        (
            23,
            "ambient-after-return",
            "https://ambient23.example/older-parent.js"
        )
    );
}

#[test]
fn agent_kept_descendant_owner_and_realm_member_conditionally_keep_ancestors() {
    for nested_owner in [false, true] {
        let mut engine = Engine::new();
        without_checkpoint(&mut engine, "globalThis.realm=new ShadowRealm;0");
        let outer = owner(&mut engine, "realm");
        without_checkpoint(
            &mut engine,
            if nested_owner {
                "realm.evaluate('new WeakRef(new ShadowRealm);0');0"
            } else {
                "realm.evaluate('new WeakRef(globalThis);0');0"
            },
        );
        let nested = if nested_owner {
            Some(crate::gc_test::in_frame(|| {
                let object = global(&mut engine.interp, "realm").as_obj().expect("outer");
                engine
                    .interp
                    .with_shadow(&object, |child| {
                        let id = child
                            .shadow_realms
                            .values()
                            .next()
                            .expect("nested slot")
                            .id();
                        child.with_realm_view(id, |nested| {
                            Box::new(
                                nested
                                    .shadow_home
                                    .as_ref()
                                    .expect("nested home")
                                    .owner
                                    .clone(),
                            )
                        })
                    })
                    .expect("outer slot")
            }))
        } else {
            None
        };
        without_checkpoint(&mut engine, "globalThis.realm=null;0");
        precise(&mut engine);
        assert!(
            crate::gc_test::alive(outer.as_ref()),
            "reachable descendant/member requires its owning ancestor"
        );
        if let Some(nested) = &nested {
            assert!(crate::gc_test::alive(nested.as_ref()));
        }
        engine.interp.drain_microtasks();
        precise(&mut engine);
        assert!(
            !crate::gc_test::alive(outer.as_ref()),
            "directory itself is not an unconditional root"
        );
        if let Some(nested) = &nested {
            assert!(!crate::gc_test::alive(nested.as_ref()));
        }
        assert!(engine.interp.shadow_realms.is_empty());
    }
}

#[test]
fn agent_kept_symbol_can_outlive_its_former_child_realm_without_retaining_it() {
    let mut engine = Engine::new();
    without_checkpoint(&mut engine,"globalThis.realm=new ShadowRealm;realm.evaluate(\"new WeakRef(Symbol('agent kept'));0\");0");
    let outer = owner(&mut engine, "realm");
    let symbol = crate::gc_test::in_frame(|| {
        let kept = engine.interp.gc_heap.agent.kept_alive.borrow();
        Box::new(
            kept.iter()
                .find_map(|value| match value.view() {
                    ValueView::Sym(_) => crate::interpreter::WeakTarget::of(value),
                    _ => None,
                })
                .expect("kept Symbol"),
        )
    });
    without_checkpoint(&mut engine, "globalThis.realm=null;0");
    precise(&mut engine);
    assert!(
        !crate::gc_test::alive(outer.as_ref()),
        "a scalar Agent root does not retain a realm"
    );
    assert!(
        crate::gc_test::alive(symbol.as_ref()),
        "shared kept epoch survives realm retirement"
    );
    engine.interp.drain_microtasks();
    precise(&mut engine);
    assert!(
        !crate::gc_test::alive(symbol.as_ref()),
        "actual outer checkpoint releases Symbol"
    );
}

#[test]
fn agent_kept_prototype_null_native_owners_preserve_metadata_and_then_retire() {
    for (constructor, kind) in [
        ("new ArrayBuffer(16)", 0),
        ("/native-metadata/u", 1),
        ("new WeakMap", 2),
        ("new WeakSet", 2),
    ] {
        let mut engine = Engine::new();
        without_checkpoint(&mut engine, "globalThis.realm=new ShadowRealm;0");
        let outer = owner(&mut engine, "realm");
        let child_source = format!(
            "let value={constructor};Object.setPrototypeOf(value,null);new WeakRef(value);0"
        );
        without_checkpoint(&mut engine, &format!("realm.evaluate({child_source:?});0"));
        let native = crate::gc_test::in_frame(|| {
            let kept = engine.interp.gc_heap.agent.kept_alive.borrow();
            let value = kept
                .iter()
                .find(|value| matches!(value.view(), ValueView::Obj(_)))
                .expect("native kept owner");
            crate::gc_test::HiddenAddress::of(value)
        });
        without_checkpoint(&mut engine, "globalThis.realm=null;0");
        precise(&mut engine);
        assert!(
            crate::gc_test::alive(outer.as_ref()),
            "{constructor}: native owner requires its implementation before metadata use"
        );
        let id = engine
            .interp
            .shadow_realms
            .values()
            .next()
            .expect("retained owner slot")
            .id();
        engine.interp.with_realm_view(id, |child| match kind {
            0 => {
                assert!(native.in_map(&child.array_buffers));
                assert_eq!(
                    child
                        .array_buffers
                        .get(&native.get())
                        .and_then(|data| data.byte_len()),
                    Some(16)
                );
            }
            1 => assert!(native.in_map(&child.regexps)),
            2 => assert!(child.weak_collection_data.contains_key(&native.get())),
            _ => unreachable!(),
        });
        engine.interp.drain_microtasks();
        precise(&mut engine);
        assert!(
            !crate::gc_test::alive(outer.as_ref()),
            "{constructor}: native pin directory must not become an unconditional root"
        );
        assert!(engine.interp.shadow_realms.is_empty());
    }
}

/// Attach only scalar native data, whose map does NOT itself record SideSlotTable writes.
/// The old language object is the sole conditional edge into a newly created child.
fn old_native_owner_fixture() -> (Engine, crate::rooting::Rooted<Value>, Box<WeakGc>) {
    let mut engine = Engine::new();
    let object = crate::rooting::Rooted::new(Value::Obj(crate::value::Object::new(None)));
    precise(&mut engine);
    assert_eq!(
        engine
            .interp
            .gc_heap
            .core
            .borrow()
            .cell_age(object.cell_word().expect("object")),
        Some(crate::gc_core::CellAge::Old)
    );
    let weak = crate::gc_test::in_frame(|| {
        let shadow = crate::value::Object::new(None);
        engine.interp.gc_pin(&shadow);
        let child = Interp::new_shadow(&engine.interp, &shadow);
        engine
            .interp
            .shadow_realms
            .insert(Gc::as_ptr(&shadow) as usize, child);
        engine
            .interp
            .with_shadow(&shadow, |child| {
                let owner = object.as_obj().expect("old native owner");
                child.gc_pin(&owner);
                child.regexps.insert(
                    Gc::as_ptr(&owner) as usize,
                    Rc::new(
                        crate::regex::Regex::new("x", "")
                            .unwrap_or_else(|_| panic!("static regex")),
                    ),
                );
            })
            .expect("new child");
        Box::new(Gc::downgrade(&shadow))
    });
    (engine, object, weak)
}

#[test]
fn agent_new_native_ownership_on_old_object_is_remembered_before_minor() {
    let (mut engine, object, weak) = old_native_owner_fixture();
    let identity = object
        .as_obj()
        .map(|owner| Gc::as_ptr(&owner) as usize)
        .expect("owner");
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert!(
        crate::gc_test::alive(weak.as_ref()),
        "recorded old owner must keep young home path"
    );
    assert!(
        engine
            .interp
            .gc_heap
            .client_report
            .borrow()
            .side_slot_owners
            .contains(&identity),
        "minor must consume the actual newly pinned owner"
    );
    drop(object);
    precise(&mut engine);
    assert!(
        !crate::gc_test::alive(weak.as_ref()),
        "owner path is conditional, not an Agent root"
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "native owner route")]
fn agent_unrecorded_old_native_owner_with_young_home_is_detected() {
    let (mut engine, object, _weak) = old_native_owner_fixture();
    let identity = object
        .as_obj()
        .map(|owner| Gc::as_ptr(&owner) as usize)
        .expect("owner");
    engine
        .interp
        .gc_heap
        .records
        .borrow_mut()
        .forget_side_slot_owner(identity);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
}

#[test]
fn agent_derived_function_retains_actual_source_record_and_module_namespace_identity() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let loads = Rc::new(RefCell::new(Vec::new()));
        let noted = loads.clone();
        engine.set_module_loader(move |specifier, referrer| {
            noted.borrow_mut().push((specifier.to_string(), referrer.to_string()));
            match specifier {
                "source-main" => Some(("source-main".into(),
                    "import * as dep from 'source-dep'; export function make(F) { dep.bump(); return F(\"return import('source-dep')\"); }".into())),
                "source-dep" => Some(("source-dep".into(),
                    "export let count=0; export function bump(){ return ++count; }".into())),
                _ => None,
            }
        });
        without_checkpoint(&mut engine, "var realm=new ShadowRealm; var derivedWrapper; realm.importValue('source-main','make').then(make=>{derivedWrapper=make(Function)});0");
        engine.interp.drain_microtasks();
        let weak = owner(&mut engine, "realm");
        crate::gc_test::in_frame(|| {
            let wrapper = global(&mut engine.interp, "derivedWrapper")
                .as_obj()
                .expect("wrapped derived function");
            let cross = match &wrapper.borrow().call {
                crate::value::Callable::WrappedShadow(wrapped) => *wrapped.target,
                _ => panic!("explicit ShadowRealm wrapper"),
            };
            // Wrapping never unwraps: the child's own wrapper of the parent Function is inside.
            let actual = match &cross.as_obj().expect("child wrapper").borrow().call {
                crate::value::Callable::WrappedCross(wrapped) => wrapped.target,
                _ => panic!("child-side wrapper of the parent Function"),
            };
            let derived = actual.as_obj().expect("actual parent Function");
            match &derived.borrow().call {
                crate::value::Callable::User(user) => {
                    assert_eq!(
                        user.realm,
                        Gc::as_ptr(&engine.interp.global) as usize,
                        "Function executes in its actual parent Realm"
                    );
                    let record = user
                        .runtime_origin(
                            engine.interp.realm_id,
                            crate::value::heap_id(&engine.interp.gc_heap),
                        )
                        .expect("actual source record");
                    assert_ne!(
                        record.owner.as_ref().expect("source home").target,
                        engine.interp.realm_id,
                        "source owner differs from execution Realm"
                    );
                }
                _ => panic!("ordinary derived Function"),
            }
            engine
                .interp
                .global
                .borrow_mut()
                .props
                .insert("derived", Property::data(actual, true, true, true));
        });
        without_checkpoint(&mut engine, "realm=null; derivedWrapper=null;0");
        engine.interp.drain_microtasks();
        precise(&mut engine);
        assert!(
            crate::gc_test::alive(weak.as_ref()),
            "only immutable source ownership retains child"
        );
        assert_eq!(evaluate(&mut engine,
            "derived().then(ns=>{globalThis.foreignNamespace=ns;globalThis.firstCounter=ns.bump()});0"), "0");
        assert_eq!(
            evaluate(
                &mut engine,
                "[firstCounter,foreignNamespace.count].join(',')"
            ),
            "2,2"
        );
        assert_eq!(evaluate(&mut engine,
            "derived().then(ns=>{globalThis.sameNamespace=ns===foreignNamespace;globalThis.secondCounter=ns.bump()});0"), "0");
        assert_eq!(
            evaluate(
                &mut engine,
                "[sameNamespace,secondCounter,foreignNamespace.count].join(',')"
            ),
            "true,3,3"
        );
        assert_eq!(
            loads
                .borrow()
                .iter()
                .filter(|(specifier, _)| specifier == "source-dep")
                .count(),
            1,
            "derived parent import must reuse child's actual loaded Module Record"
        );
        without_checkpoint(&mut engine, "derived=null;foreignNamespace=null;0");
        engine.interp.drain_microtasks();
        precise(&mut engine);
        assert!(
            !crate::gc_test::alive(weak.as_ref()),
            "records and namespace owners retire conditionally"
        );
    }
}

fn major_collections(engine: &Engine) -> u64 {
    let stats = engine.interp.gc_heap.core.borrow().stats();
    stats.collections - stats.minor_collections
}

#[test]
fn agent_buffer_pressure_counts_parent_and_child_data_together() {
    // The expected collections assume no stress collections between allocations.
    if crate::gc_test::stress_active() {
        return;
    }
    let mut engine = Engine::new();
    without_checkpoint(
        &mut engine,
        "var realm=new ShadowRealm;\
         var childAlloc=realm.evaluate('(function(n){return new ArrayBuffer(n).byteLength;})');0",
    );
    precise(&mut engine);
    let before = major_collections(&engine);
    // Five rounds of 12 MiB on each side: 60 MiB per interpreter stays under the 64 MiB
    // trigger, while the Agent's 120 MiB passes it.
    without_checkpoint(
        &mut engine,
        "for(var k=0;k<5;k++){new ArrayBuffer(12<<20);childAlloc(12<<20);}0",
    );
    assert!(
        major_collections(&engine) > before,
        "the Agent's combined buffer data collects"
    );
    assert!(engine.interp.gc_heap.agent.buffers.get().since_gc < 64 << 20);
}

#[test]
fn agent_buffer_growth_counts_like_allocation() {
    // The expected collections assume no stress collections between allocations.
    if crate::gc_test::stress_active() {
        return;
    }
    for (label, source) in [
        (
            "ArrayBuffer resize",
            "for(var k=0;k<6;k++){var b=new ArrayBuffer(0,{maxByteLength:16<<20});b.resize(16<<20);}0",
        ),
        (
            "SharedArrayBuffer grow",
            "for(var k=0;k<6;k++){var s=new SharedArrayBuffer(0,{maxByteLength:16<<20});s.grow(16<<20);}0",
        ),
    ] {
        let mut engine = Engine::new();
        precise(&mut engine);
        let before = major_collections(&engine);
        // 96 MiB of growth from empty buffers, discarded as they go.
        without_checkpoint(&mut engine, source);
        assert!(
            major_collections(&engine) > before,
            "{label}: growth counts toward buffer pressure"
        );
    }
}

#[test]
fn agent_owner_index_follows_the_first_and_last_child_interpreter() {
    let mut engine = Engine::new();
    // Metadata a lone root interpreter created before any child existed.
    without_checkpoint(&mut engine, "globalThis.p = new Proxy({}, {}); 0");
    let proxy = crate::gc_test::in_frame(|| {
        Gc::as_ptr(&global(&mut engine.interp, "p").as_obj().expect("proxy")) as usize
    });
    assert!(engine
        .interp
        .gc_heap
        .agent
        .metadata_owners
        .borrow()
        .is_empty());
    for round in 0..2 {
        without_checkpoint(&mut engine, "globalThis.realm = new ShadowRealm; 0");
        assert_eq!(
            engine.interp.gc_heap.agent.children.get(),
            1,
            "round {round}"
        );
        let shadow = crate::gc_test::in_frame(|| {
            Box::new(global(&mut engine.interp, "realm").as_obj().expect("realm"))
        });
        engine
            .interp
            .with_shadow(&shadow, |child| {
                assert!(
                    child.proxy_at(proxy).is_some(),
                    "round {round}: the child finds the root's earlier record"
                );
            })
            .expect("owned child");
        drop(shadow);
        without_checkpoint(&mut engine, "globalThis.realm = null; 0");
        engine.interp.drain_microtasks();
        precise(&mut engine);
        assert_eq!(
            engine.interp.gc_heap.agent.children.get(),
            0,
            "round {round}"
        );
        assert!(
            engine
                .interp
                .gc_heap
                .agent
                .metadata_owners
                .borrow()
                .is_empty(),
            "round {round}: a lone root keeps no owner index"
        );
    }
    assert!(engine.interp.proxy_at(proxy).is_some());
}

#[test]
fn agent_closures_of_one_parsed_function_keep_each_script_record() {
    let mut engine = Engine::new();
    let requests = Rc::new(RefCell::new(Vec::new()));
    let captured = requests.clone();
    engine.set_async_dynamic_module_loader_with_context(move |_, _, specifier, referrer, _| {
        captured
            .borrow_mut()
            .push((specifier.to_string(), referrer.to_string()));
        true
    });
    // One parsed body run as two scripts: each run is its own Script Record, so closures of
    // the same function keep different records (the first one's is shared, the second boxed).
    let body = crate::parser::parse_script(
        "(globalThis.made ||= []).push(function (specifier) { import(specifier); }); 0",
        false,
    )
    .unwrap_or_else(|error| panic!("parse: {}", error.message));
    for base in ["https://a.example/one.js", "https://b.example/two.js"] {
        engine.set_import_base(base);
        if engine
            .interp
            .with_author_entry(|interp| interp.run_program(&body))
            .is_err()
        {
            panic!("script run");
        }
    }
    engine.set_import_base("https://c.example/later.js");
    without_checkpoint(
        &mut engine,
        "made[0]('x0'); made[1]('x1'); made[0]('x2'); 0",
    );
    assert_eq!(
        requests.borrow().as_slice(),
        &[
            ("x0".to_string(), "https://a.example/one.js".to_string()),
            ("x1".to_string(), "https://b.example/two.js".to_string()),
            ("x2".to_string(), "https://a.example/one.js".to_string()),
        ]
    );
}

#[test]
fn agent_suppressed_requests_defer_and_keep_a_pending_major_over_a_later_minor() {
    use crate::agent_control::{CollectionOutcome, Suppression};
    use crate::value::GcCause;
    let mut engine = Engine::new();
    let heap = engine.interp.gc_heap.clone();
    let before = heap.core.borrow().stats();
    {
        let _suppressed = Suppression::new(&heap);
        assert_eq!(
            engine
                .interp
                .gc_heap_collect(GcCause::Explicit, Some(CollectionKind::Major)),
            CollectionOutcome::Deferred
        );
        assert_eq!(
            engine
                .interp
                .gc_heap_collect(GcCause::Explicit, Some(CollectionKind::Minor)),
            CollectionOutcome::Deferred
        );
        assert_eq!(
            heap.agent.pending.get(),
            Some(CollectionKind::Major),
            "a later minor request cannot replace the pending major"
        );
        assert_eq!(heap.core.borrow().stats().collections, before.collections);
    }
    // The next safepoint performs the pending major collection, not a minor one.
    assert_eq!(
        engine
            .interp
            .gc_heap_collect(GcCause::AllocationThreshold, None),
        CollectionOutcome::Collected(CollectionKind::Major)
    );
    let after = heap.core.borrow().stats();
    assert_eq!(after.collections, before.collections + 1);
    assert_eq!(after.minor_collections, before.minor_collections);
    assert_eq!(heap.agent.pending.get(), None);
}

#[test]
fn agent_child_await_resumes_in_its_own_interpreter_when_the_parent_settles() {
    let mut engine = Engine::new();
    engine.set_module_loader(|specifier, _| match specifier {
        "await-main" => Some((
            "await-main".into(),
            "export function make(F) { return F(\"return import('await-dep')\"); }".into(),
        )),
        "await-dep" => Some((
            "await-dep".into(),
            "export let resumed = 0;\
             export async function waitFor(p) { const v = await p; resumed++; return v; }"
                .into(),
        )),
        _ => None,
    });
    without_checkpoint(
        &mut engine,
        "var realm=new ShadowRealm; var derivedWrapper;\
         realm.importValue('await-main','make').then(make=>{derivedWrapper=make(Function)});0",
    );
    engine.interp.drain_microtasks();
    // The actual parent Function inside the parent's wrapper of the child's wrapper. Its
    // ordinary import() yields the child module's own namespace, unwrapped.
    crate::gc_test::in_frame(|| {
        let wrapper = global(&mut engine.interp, "derivedWrapper")
            .as_obj()
            .expect("wrapped derived function");
        let cross = match &wrapper.borrow().call {
            crate::value::Callable::WrappedShadow(wrapped) => *wrapped.target,
            _ => panic!("explicit ShadowRealm wrapper"),
        };
        let actual = match &cross.as_obj().expect("child wrapper").borrow().call {
            crate::value::Callable::WrappedCross(wrapped) => wrapped.target,
            _ => panic!("child-side wrapper of the parent Function"),
        };
        engine
            .interp
            .global
            .borrow_mut()
            .props
            .insert("derived", Property::data(actual, true, true, true));
    });
    assert_eq!(
        evaluate(&mut engine, "derived().then(ns=>{globalThis.dep=ns});0"),
        "0"
    );
    // The parent settles a promise the child's async function awaits: the resumption job is
    // enqueued by the parent but belongs to (and must run in) the child's interpreter.
    assert_eq!(
        evaluate(
            &mut engine,
            "var waited, settle, p = new Promise(r => settle = r);\
             dep.waitFor(p).then(v => waited = v); settle(7); 0"
        ),
        "0"
    );
    assert_eq!(evaluate(&mut engine, "waited + ':' + dep.resumed"), "7:1");
}

#[test]
fn agent_host_requested_drain_runs_queued_jobs_inside_the_outer_job() {
    let mut engine = Engine::new();
    let drain = engine.interp.make_native_closure(
        "hostDrain",
        0,
        Rc::new(|interp, _, _| {
            assert!(!interp.gc_heap.agent.kept_alive.borrow().is_empty());
            interp.drain_microtasks_for_host();
            assert!(
                !interp.gc_heap.agent.kept_alive.borrow().is_empty(),
                "a host drain does not end the outer job's KeptAlive epoch"
            );
            Ok(Value::Undefined)
        }),
        NativeCaptures::None,
    );
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("hostDrain", Property::plain(Value::Obj(drain)));
    // The drain runs the queued job now; that job's own drain request defers to the running
    // drain, which then also runs the job queued meanwhile, before the script continues.
    assert_eq!(
        evaluate(
            &mut engine,
            "var order = [], weak = new WeakRef({}); weak.deref();\
             Promise.resolve().then(() => {\
               order.push('a'); Promise.resolve().then(() => order.push('c'));\
               hostDrain(); order.push('b');\
             });\
             hostDrain(); order.push('after'); order.join()"
        ),
        "a,b,c,after"
    );
    assert!(engine.interp.gc_heap.agent.kept_alive.borrow().is_empty());
}
