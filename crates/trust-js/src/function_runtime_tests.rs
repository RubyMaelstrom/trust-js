//! Shared parsed code retains heap-local runtime state (ECMA-262 e28783d5,
//! OrdinaryFunctionCreate / PrepareForOrdinaryCall / RunJobs).
use crate::bytecode::Tier;
use crate::gc_trace::{GcSink, GcTrace};
use crate::value::{Callable, Gc, PropertyLayout};
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).ok().unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn install(engine: &mut Engine, body: &[crate::ast::Stmt]) {
    engine.interp.activate_gc_heap();
    assert!(engine.interp.run_program(body).is_ok());
}

fn function(engine: &mut Engine, name: &str) -> Rc<crate::ast::Function> {
    engine.interp.activate_gc_heap();
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().unwrap();
    let object_handle = value.as_obj().unwrap();
    let object = object_handle.borrow();
    let Callable::User(user) = &object.call else {
        panic!("user function")
    };
    user.func.clone()
}

fn assert_local_code(engine: &mut Engine, name: &str) {
    let function = function(engine, name);
    let runtime = function.runtime();
    assert_eq!(
        runtime.heap.id,
        crate::value::heap_id(&engine.interp.gc_heap)
    );
    struct Local<'a>(&'a crate::gc_core::Heap);
    impl GcSink for Local<'_> {
        fn word(&mut self, word: u64) {
            assert!(
                self.0.is_allocated_word(word),
                "foreign code cell {word:#x}"
            );
        }
        fn layout(&mut self, layout: &PropertyLayout) {
            self.word(layout.cell_word());
        }
    }
    let core = engine.interp.gc_heap.core.borrow();
    let mut sink = Local(&core);
    if let Some(Some(chunk)) = runtime.code.get() {
        assert!(chunk.belongs_to_heap(runtime.heap.id));
        assert_eq!(chunk.heap_id(), runtime.heap.id);
        chunk.trace_gc(&mut sink);
    }
    if let Some(Some(chunk)) = runtime.code2.get() {
        assert!(chunk.belongs_to_heap(runtime.heap.id));
        chunk.trace_gc(&mut sink);
    }
    if let Some(maps) = runtime.fn_maps.get() {
        maps.trace_gc(&mut sink);
    }
}

fn assert_raw_call_ic(engine: &mut Engine) {
    let caller = function(engine, "caller");
    let runtime = caller.runtime();
    let chunk = runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .expect("caller compiled");
    assert!(chunk.jit.get().flatten().is_some());
    assert!(chunk.jit_runs.get() > 0);
    let epoch = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        chunk.jit_ops().iter().any(|op| {
            let (crate::bytecode::Op::Call(_, site) | crate::bytecode::Op::CallWithThis(_, site)) =
                op
            else {
                return false;
            };
            let entries =
                chunk.jit_call_cache_ptr(*site) as *const std::cell::Cell<crate::bytecode::CallIc>;
            (0..crate::bytecode::CALL_IC_WAYS).any(|way| {
                // The chunk owns this fixed cache array for the complete assertion.
                let entry = unsafe { (*entries.add(way)).get() };
                entry.callee != 0
                    && !entry.chunk.is_null()
                    && !entry.code.is_null()
                    && !entry.code_mem.is_null()
                    && entry.epoch == epoch
                    && entry.direct & 1 != 0
            })
        }),
        "a live raw user-call IC must be filled before runtime retirement"
    );
}

const TEMPLATE_SOURCE: &str = r#"
function C(x){this.first=x;this.second='constructor string';}
function literal(x){return {first:x,second:'literal string',third:123456789012345678901234567890n};}
function named(){return ()=> 'named string';}
function* generate(){yield literal(9);}
async function asynchronous(){return literal(10);}
class PrivateBox{#value='private string';read(){return this.#value;}}
function privateRead(){return new PrivateBox().read();}
function drive(){var result;for(var k=0;k<150;k++)result=literal(k);return result;}
"#;

fn exercise(engine: &mut Engine) {
    assert_eq!(eval(engine, "var kept=drive();var instance=new C(7);var arrow=named();[kept.first,kept.second,String(kept.third),instance.first,instance.second,arrow(),generate().next().value.first,privateRead()].join('|')"),
        "149|literal string|123456789012345678901234567890|7|constructor string|named string|9|private string");
    eval(
        engine,
        "var asyncValue;asynchronous().then(value=>{asyncValue=value;});0",
    );
    engine.interp.activate_gc_heap();
    engine.interp.drain_microtasks();
    engine.interp.gc_task_boundary();
    assert_eq!(
        eval(engine, "asyncValue.first+':'+asyncValue.second"),
        "10:literal string"
    );
    engine.interp.gc_collect_minor();
    engine.interp.gc_collect();
    assert_eq!(
        eval(
            engine,
            "[kept.second,instance.second,arrow(),String(literal(3).third)].join('|')"
        ),
        "literal string|constructor string|named string|123456789012345678901234567890"
    );
    for name in [
        "C",
        "literal",
        "named",
        "generate",
        "asynchronous",
        "drive",
        "privateRead",
    ] {
        assert_local_code(engine, name);
    }
}

#[test]
fn parsed_templates_and_whole_chunks_are_heap_local_after_switches_and_release_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let body = crate::parser::parse_script(TEMPLATE_SOURCE, false)
            .ok()
            .unwrap();
        let mut first = Engine::new();
        first.set_tier(tier);
        first.set_tier_threshold(0);
        install(&mut first, &body);
        exercise(&mut first);
        let mut second = Engine::new();
        second.set_tier(tier);
        second.set_tier_threshold(0);
        install(&mut second, &body);
        exercise(&mut second);
        exercise(&mut first);
        exercise(&mut second);
        drop(first);
        exercise(&mut second);
        drop(second);
        let mut later = Engine::new();
        later.set_tier(tier);
        later.set_tier_threshold(0);
        install(&mut later, &body);
        exercise(&mut later);
    }
}

#[test]
fn constructor_hints_trace_layouts_conditionally_and_record_old_to_young_writes() {
    let mut engine = Engine::new();
    eval(&mut engine, "function hinted(){};0");
    let insert = |engine: &mut Engine, key: &str| {
        crate::gc_test::in_frame(|| {
            let env = engine.interp.global_env;
            let value = engine.interp.get_var("hinted", &env).ok().unwrap();
            let constructor = value.as_obj().unwrap();
            let layout = crate::value::new_property_layout(vec![Rc::from(key)]);
            let weak = crate::gc_handle::WeakRef::to(layout.as_ptr() as usize);
            engine.interp.construct_capacity_hints.insert(
                &engine.interp.gc_heap,
                Gc::as_ptr(&constructor) as usize,
                (Gc::downgrade(&constructor), layout),
            );
            weak
        })
    };
    let first = insert(&mut engine, "first hint");
    engine.interp.gc_collect();
    assert!(crate::gc_test::alive(&first));
    let young = insert(&mut engine, "young hint");
    engine.interp.gc_collect_minor();
    assert!(crate::gc_test::alive(&young));
    eval(&mut engine, "hinted=undefined;0");
    engine.interp.gc_collect();
    assert!(engine.interp.construct_capacity_hints.is_empty());
    assert!(!crate::gc_test::alive(&young));
}

#[test]
fn initialized_function_clones_start_with_neutral_runtime_state() {
    let body = crate::parser::parse_script("var f=function source(x){return x+1;};", false)
        .ok()
        .unwrap();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    install(&mut engine, &body);
    assert_eq!(eval(&mut engine, "f(1)"), "2");
    let original = function(&mut engine, "f");
    assert!(original.runtime().code.get().is_some());
    let mut cloned = (*original).clone();
    assert!(cloned.runtime.get().is_none());
    assert!(cloned.env_layouts.get().is_none());
    assert!(cloned.hoist.get().is_none());
    assert_eq!(cloned.scan.get(), 0);
    cloned.is_strict = !cloned.is_strict;
    let runtime = cloned.runtime();
    assert!(runtime.code.get().is_none());
    assert!(runtime.code2.get().is_none());
    assert!(runtime.fn_maps.get().is_none());
    assert_eq!(runtime.calls.get(), 0);
    assert!(original.runtime().code.get().is_some());
}

#[test]
fn initialized_function_can_be_cloned_without_an_agent_after_heap_release() {
    let mut first = Engine::new();
    first.set_tier(Tier::Jit);
    first.set_tier_threshold(0);
    assert_eq!(
        eval(&mut first, "function source(x){return x+1;}source(4)"),
        "5"
    );
    let original = function(&mut first, "source");
    assert!(original.runtime().code.get().is_some());
    drop(first);
    let cloned = (*original).clone();
    assert!(cloned.runtime.get().is_none());
    assert!(cloned.env_layouts.get().is_none());
    assert!(cloned.hoist.get().is_none());
    assert_eq!(cloned.scan.get(), 0);
    let mut later = Engine::new();
    later.set_tier(Tier::Jit);
    later.set_tier_threshold(0);
    let closure = later
        .interp
        .make_function(Rc::new(cloned), later.interp.global_env);
    later
        .interp
        .global
        .borrow_mut()
        .props
        .insert("copy", crate::value::Property::plain(closure));
    assert_eq!(eval(&mut later, "copy(8)"), "9");
    assert_local_code(&mut later, "copy");
}

#[test]
fn retiring_an_evicted_foreign_runtime_fences_warmed_same_ast_call_ics() {
    const CHILD: &str = "TRUST_JS_TEST_FOREIGN_RUNTIME_RETIREMENT_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "function_runtime_tests::retiring_an_evicted_foreign_runtime_fences_warmed_same_ast_call_ics", "--nocapture"])
            .env(CHILD, "1").env("TRUST_JS_INLINE_AT", "4294967295").output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    for tier in [Tier::Bytecode, Tier::Jit] {
        let body = crate::parser::parse_script("var f=function shared(x){return x+1;};", false)
            .ok()
            .unwrap();
        let mut primary = Engine::new();
        primary.set_tier(tier);
        primary.set_tier_threshold(0);
        let mut outer = Engine::new();
        outer.set_tier(tier);
        outer.set_tier_threshold(0);
        let mut foreign = Vec::new();
        for _ in 0..6 {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            foreign.push(engine);
        }
        install(&mut primary, &body);
        install(&mut outer, &body);
        outer.interp.gc_collect();
        assert_eq!(eval(&mut outer, "function caller(){return f(6)+1;}var sum=0;for(var k=0;k<200;k++)sum+=caller();sum"), "1600");
        if tier == Tier::Jit {
            assert_raw_call_ic(&mut outer);
        }
        let (closure, runtime, code_lease) = crate::gc_test::in_frame(|| {
            let env = outer.interp.global_env;
            let value = outer.interp.get_var("f", &env).ok().unwrap();
            let object = value.as_obj().unwrap();
            let borrowed = object.borrow();
            let Callable::User(user) = &borrowed.call else {
                panic!("user function")
            };
            let sidecar = user.cache.take().expect("foreign sidecar");
            let runtime = sidecar.runtime.as_ref().unwrap();
            let code_lease = runtime.code.get().and_then(Option::as_ref).unwrap().clone();
            let runtime = Rc::downgrade(runtime);
            user.cache.set(Some(sidecar));
            (Gc::downgrade(&object), runtime, code_lease)
        });
        for engine in &mut foreign {
            install(engine, &body);
        }
        assert_eq!(
            runtime.strong_count(),
            1,
            "the live closure pins its evicted runtime"
        );
        eval(&mut outer, "f=undefined;0");
        if tier == Tier::Jit {
            assert_raw_call_ic(&mut outer);
        }
        let epoch = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        outer.interp.gc_collect();
        outer.interp.gc_collect();
        assert!(!crate::gc_test::alive(&closure));
        assert_eq!(
            runtime.strong_count(),
            0,
            "last runtime lease retires with the old closure"
        );
        assert_ne!(
            epoch,
            crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
            "raw slot addresses must be fenced before retirement"
        );
        // Keep the old chunk/native image alive while the runtime's OnceCell slot dies:
        // Chunk/JitCode retirement cannot supply the epoch change asserted above.
        assert!(code_lease.belongs_to_heap(crate::value::heap_id(&outer.interp.gc_heap)));
        install(&mut outer, &body);
        assert_eq!(
            eval(
                &mut outer,
                "var sum=0;for(var k=0;k<200;k++)sum+=caller();sum"
            ),
            "1600"
        );
        assert_local_code(&mut outer, "f");
        drop(code_lease);
    }
}

#[test]
fn an_evicted_foreign_runtime_is_recovered_while_its_closure_lives() {
    let body = crate::parser::parse_script("var f=function shared(x){return x+1;};", false)
        .ok()
        .unwrap();
    let mut primary = Engine::new();
    install(&mut primary, &body);
    let mut outer = Engine::new();
    outer.set_tier(Tier::Jit);
    outer.set_tier_threshold(0);
    install(&mut outer, &body);
    assert_eq!(eval(&mut outer, "f(1)"), "2");
    let original = function(&mut outer, "f");
    let runtime = match original.runtime() {
        crate::function_runtime::RuntimeLease::Foreign(runtime) => Rc::downgrade(&runtime),
        _ => panic!("foreign runtime"),
    };
    let cloned = (*original).clone();
    assert!(cloned.runtime().code.get().is_none());
    let mut foreign = Vec::new();
    for _ in 0..6 {
        let mut engine = Engine::new();
        install(&mut engine, &body);
        foreign.push(engine);
    }
    assert_eq!(
        runtime.strong_count(),
        1,
        "closure is the only remaining runtime lease"
    );
    outer.interp.activate_gc_heap();
    let recovered = original.runtime();
    assert_eq!(std::ptr::from_ref(&*recovered), runtime.as_ptr());
    assert!(
        recovered.code.get().is_some_and(Option::is_some),
        "recover the compiled entry rather than recompiling"
    );
    assert_eq!(eval(&mut outer, "f(9)"), "10");
    assert!(
        cloned.runtime().code.get().is_none(),
        "cloned AST stays independent after foreign cache churn"
    );
}

#[test]
fn foreign_getter_fill_stale_reset_and_tdz_preserve_the_runtime_lease() {
    let body = crate::parser::parse_script("var getter=(function(){let held=7;return function sharedGetter(){return held;};})();var box={};Object.defineProperty(box,'value',{get:getter});", false).ok().unwrap();
    let mut primary = Engine::new();
    install(&mut primary, &body);
    let mut outer = Engine::new();
    install(&mut outer, &body);
    let env = outer.interp.global_env;
    let getter = outer.interp.get_var("getter", &env).ok().unwrap();
    let borrowed_handle = getter.as_obj().unwrap();
    let borrowed = borrowed_handle.borrow();
    let Callable::User(user) = &borrowed.call else {
        panic!("getter")
    };
    let captured = user.env;
    let user = *user;
    drop(borrowed);
    let inspect = |expected: bool| {
        let sidecar = user
            .cache
            .take()
            .expect("foreign sidecar remains allocated");
        assert_eq!(sidecar.binding_getter.is_some(), expected);
        let runtime = Rc::as_ptr(
            sidecar
                .runtime
                .as_ref()
                .expect("runtime lease survives getter changes"),
        );
        user.cache.set(Some(sidecar));
        runtime
    };
    let identity = inspect(false);
    assert_eq!(eval(&mut outer, "box.value"), "7");
    assert_eq!(inspect(true), identity);
    captured.borrow_mut().vars.insert(
        "unrelated",
        crate::interpreter::Binding::data(crate::value::Value::Num(0.0), true, true),
    );
    assert_eq!(eval(&mut outer, "box.value"), "7");
    assert_eq!(
        inspect(true),
        identity,
        "stale proof is re-resolved without losing the runtime"
    );
    let tdz_held = captured
        .borrow_mut()
        .vars
        .get_mut("held")
        .unwrap()
        .enter_tdz_for_test();
    assert_eq!(
        eval(&mut outer, "try{box.value}catch(error){error.name}"),
        "ReferenceError"
    );
    assert_eq!(inspect(true), identity);
    captured
        .borrow_mut()
        .vars
        .get_mut("held")
        .unwrap()
        .initialize(tdz_held);
    captured.borrow_mut().vars.remove("held");
    assert_eq!(
        eval(&mut outer, "try{box.value}catch(error){error.name}"),
        "ReferenceError"
    );
    assert_eq!(
        inspect(false),
        identity,
        "failed re-resolution clears only the getter proof"
    );
    captured.borrow_mut().vars.insert(
        "held",
        crate::interpreter::Binding::data(crate::value::Value::Num(13.0), true, true),
    );
    assert_eq!(eval(&mut outer, "box.value"), "13");
    assert_eq!(inspect(true), identity);
}

#[test]
fn major_gc_prunes_dead_runtime_index_storage_and_recovers_its_live_entry() {
    let body = crate::parser::parse_script("function indexed(x){return x+1;}", false)
        .ok()
        .unwrap();
    let crate::ast::Stmt::FuncDecl(template) = &body[0] else {
        panic!("function")
    };
    let primary = Engine::new();
    let mut owner = Engine::new();
    let baseline = crate::value::function_runtime_index_stats(&owner.interp.gc_heap).0;
    let (kept, runtime) = crate::gc_test::in_frame(|| {
        let mut survivor = None;
        for n in 0..512 {
            primary.interp.activate_gc_heap();
            let function = Rc::new((**template).clone());
            assert!(matches!(
                function.runtime(),
                crate::function_runtime::RuntimeLease::Primary(_)
            ));
            owner.interp.activate_gc_heap();
            let closure = owner
                .interp
                .make_function(function.clone(), owner.interp.global_env);
            if n == 0 {
                owner
                    .interp
                    .global
                    .borrow_mut()
                    .props
                    .insert("kept", crate::value::Property::plain(closure));
                let selected = function.runtime();
                let crate::function_runtime::RuntimeLease::Foreign(runtime) = selected else {
                    panic!("foreign entry")
                };
                survivor = Some((function, Rc::downgrade(&runtime)));
            }
            // All other closures become garbage; their cells own the AST/runtime until GC.
        }
        survivor.unwrap()
    });
    let mut foreign = Vec::new();
    for _ in 0..6 {
        let engine = Engine::new();
        assert!(matches!(
            kept.runtime(),
            crate::function_runtime::RuntimeLease::Foreign(_)
        ));
        foreign.push(engine);
    }
    assert_eq!(
        runtime.strong_count(),
        1,
        "only the surviving closure pins its evicted runtime"
    );
    let before = crate::value::function_runtime_index_stats(&owner.interp.gc_heap);
    assert!(before.0 >= baseline + 512);
    owner.interp.activate_gc_heap();
    owner.interp.gc_collect();
    let after = crate::value::function_runtime_index_stats(&owner.interp.gc_heap);
    assert_eq!(
        after.2, 0,
        "major maintenance removes every dead weak entry after finalization"
    );
    assert!(
        after.0 <= baseline + 1,
        "garbage closure finalizers release the other runtimes"
    );
    assert!(
        before.1 > after.0 * 4 && before.1 > 128,
        "the fixture must substantially underfill a large index"
    );
    assert!(
        after.1 < before.1,
        "major maintenance shrinks high-peak buckets"
    );
    assert!(after.1 >= 64 && after.1 >= after.0);
    assert_eq!(runtime.strong_count(), 1);
    let recovered = kept.runtime();
    assert_eq!(
        std::ptr::from_ref(&*recovered),
        runtime.as_ptr(),
        "pruning preserves recovery of the live, evicted entry"
    );
}
