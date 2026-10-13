use super::*;
use crate::{bytecode::Tier, feedback::ValueClass, value::Callable, Completion, Engine};
use std::rc::Rc;

thread_local! {
    static RETIRING_VERSION: std::cell::RefCell<Option<std::rc::Weak<JitCode>>> = const {
        std::cell::RefCell::new(None)
    };
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn install(engine: &mut Engine, classes: &[u32]) -> (Rc<Chunk>, Rc<JitCode>) {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let function = {
        let object = value.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("user function")
        };
        user.func.clone()
    };
    let mut chunk = bytecode::compile(&function).unwrap();
    Rc::get_mut(&mut chunk).unwrap().optimizing_inputs = classes
        .iter()
        .enumerate()
        .map(|(slot, &bits)| (slot as u16, bits))
        .collect();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let plan = specialization::Plan::build(&chunk, &cfg, &stack, &chunk.optimizing_inputs, &[])
        .expect("fixture has a guarded semantic simplification");
    assert!(plan.saved_tests > 0);
    assert!(
        plan.opportunities.iter().any(|op| op.site.is_some()),
        "stable semantic sites reach the optimizer"
    );
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = Rc::new(compile_with_deopt(&chunk, &values, &layout, None).unwrap());
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.code.set(Some(chunk.clone())).is_ok());
    (chunk, code)
}

#[test]
fn retirement_reuses_primary_while_existing_native_frames_finish() {
    const CHILD: &str = "TRUST_JS_TEST_RETIRED_PRIMARY_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "jit::optimizing::specialization_tests::retirement_reuses_primary_while_existing_native_frames_finish",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("TRUST_JS_OPT_JIT", "hot")
            .env("TRUST_JS_OPT_JIT_HOT_AT", "8")
            .env("TRUST_JS_INLINE_AT", "0")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        return;
    }
    fn observe(
        i: &mut crate::interpreter::Interp,
        _: crate::Value,
        _: &[crate::Value],
    ) -> Result<crate::Value, crate::Value> {
        let weak = RETIRING_VERSION.with(|state| state.borrow().as_ref().unwrap().clone());
        {
            let code = weak.upgrade().unwrap();
            assert!(code.residency.active.get() > 0);
        }
        // No Rust lease remains here: the active native return PC alone protects
        // this version even after a recursive call has retired future dispatch.
        crate::jit::cache::reclaim(usize::MAX);
        assert!(weak.upgrade().is_some(), "active native frame stays mapped");
        i.gc_collect();
        Ok(crate::Value::Undefined)
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let global = engine.interp.global.clone();
    engine
        .interp
        .def_method(&global, "observeRetirement", 0, observe);
    assert_eq!(
        eval(
            &mut engine,
            r#"
        function subject(x,callback){'use strict';
            let answer=x*2+1;
            if(typeof callback==='function')callback();
            return answer;
        }
        var coercions=0;
        var mixed={valueOf(){coercions++;return 3;}};
        function reenter(){for(var k=0;k<8;k++)subject(mixed);observeRetirement();}
        var total=0;for(var k=0;k<80;k++)total+=subject(k);total;
    "#
        ),
        "6400"
    );
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let function = {
        let object = value.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("user function")
        };
        user.func.clone()
    };
    let primary = function.code.get().and_then(Option::as_ref).unwrap();
    let original_native = primary.jit.get().flatten().unwrap();
    let upgraded = function.code2.get().and_then(Option::as_ref).unwrap();
    assert!(upgraded.optimizing_candidate);
    let old = Rc::downgrade(&upgraded.jit.get().flatten().unwrap());
    RETIRING_VERSION.with(|state| *state.borrow_mut() = Some(old.clone()));
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    assert_eq!(eval(&mut engine, "subject(4,reenter)"), "9");
    assert_eq!(eval(&mut engine,
        "[subject(3),[1,2].map(subject).join(','),new subject(mixed) instanceof subject,coercions].join('|')"),
        "7|3,5|true|9");
    assert_eq!(upgraded.optimizing_misses.get(), SPECIALIZATION_MISS_LIMIT);
    let selected = function.execution_code().and_then(Option::as_ref).unwrap();
    assert!(
        Rc::ptr_eq(selected, primary),
        "new entries reuse the canonical template chunk"
    );
    assert!(
        Rc::ptr_eq(&primary.jit.get().flatten().unwrap(), &original_native),
        "retirement must not recompile an already resident template"
    );
    let lease = old.upgrade().expect(
        "ordinary dispatch leaves retired code in the bounded cache until pressure or destruction",
    );
    assert_eq!(lease.residency.active.get(), 0);
    let retired_bytes = lease.len;
    crate::jit::cache::reclaim(usize::MAX);
    assert!(old.upgrade().is_some(), "a Rust lease still protects code");
    let available = crate::jit::executable_code_budget()
        .remaining
        .load(std::sync::atomic::Ordering::Relaxed);
    drop(lease);
    crate::jit::cache::reclaim(usize::MAX);
    assert!(
        upgraded.jit.get().is_none(),
        "pressure clears the retired slot"
    );
    assert!(
        old.upgrade().is_none(),
        "pressure reclaims the actual inactive, unleased mapping"
    );
    assert!(
        crate::jit::executable_code_budget()
            .remaining
            .load(std::sync::atomic::Ordering::Relaxed)
            >= available + retired_bytes,
        "retired executable bytes stayed charged until actual reclamation"
    );
    bytecode::native_deopt::ENTRIES.with(|entries| assert_eq!(entries.borrow().len(), 8));
    assert_eq!(original_native.residency.active.get(), 0);
    assert_eq!(engine.interp.depth, 0);
    assert!(engine.interp.fn_frames.is_empty());
    RETIRING_VERSION.with(|state| *state.borrow_mut() = None);
}

#[test]
fn guarded_number_inputs_preserve_ieee_values_and_resume_coercion_once() {
    let source = r#"
        function subject(a,b) { 'use strict';
            let total=a+b;
            for(let k=0;k<3;k++)total=total*1.25-b;
            return total;
        }
        var trace=[];
    "#;
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    let (_, code) = install(&mut engine, &[ValueClass::NumberInt32.bit(); 2]);
    let observation = r#"
        var finite=subject(2,1);
        var minusZero=Object.is(subject(-0,-0),-0);
        var nan=Number.isNaN(subject(NaN,1));
        var inf=subject(Infinity,1)===Infinity;
        var obj={valueOf(){trace.push('coerce');return 2;}};
        var slow=subject(obj,1);
        var big;try{subject(1n,2n);}catch(e){big=e instanceof TypeError;}
        [finite,minusZero,nan,inf,slow,big,trace.join(',')].join('|');
    "#;
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    let expected = "2.046875|false|true|true|2.046875|true|coerce";
    assert_eq!(eval(&mut engine, observation), expected);
    bytecode::native_deopt::ENTRIES.with(|entries| {
        assert_eq!(
            entries.borrow().iter().map(|row| row.0).collect::<Vec<_>>(),
            [0, 0]
        );
    });
    assert_eq!(code.residency.active.get(), 0);
    engine.interp.gc_collect();
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, source);
    assert_eq!(eval(&mut oracle, observation), expected);
}

#[test]
fn number_intermediates_preserve_ieee_order_across_joins_and_observers() {
    let source = r#"
        function subject(a,b,choice) { 'use strict';
            if(choice===0)return [(a/b)+(a-b),(a+1)-a,(a*b)+b,(a/b)*(a-b)];
            if(choice===1)return (choice?a/b:a-b)+observe();
            try { return (a/b)+fail(); }
            catch(e) { return (a-b)+e; }
        }
        var trace=[];
        function observe(){trace.push('observe');return 1;}
        function fail(){trace.push('throw');throw 7;}
        function describe(n){return Number.isNaN(n)?'NaN':Object.is(n,-0)?'-0':String(n);}
    "#;
    let observation = r#"
        var result=[];
        for(var pair of [[0,0],[-0,1],[1,0],[Infinity,Infinity],
                         [9007199254740992,1],[5e-324,2],[1e308,1e308]]) {
            result.push(subject(pair[0],pair[1],0).map(describe).join(','));
            result.push(describe(subject(pair[0],pair[1],1)));
            result.push(describe(subject(pair[0],pair[1],2)));
        }
        var changing={valueOf(){trace.push('coerce');return 2;}};
        result.push(subject(changing,1,0).map(describe).join(','));
        result.join('|')+'|'+trace.join(',');
    "#;
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, source);
    let expected = eval(&mut oracle, observation);
    // Independent checks constrain the oracle at the NaN and rounding boundaries.
    assert!(expected.starts_with("NaN,1,0,NaN|NaN|7|"));
    assert!(expected.contains("18014398509481984,0,9007199254740992,"));
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    let (_, code) = install(&mut engine, &[ValueClass::NumberDouble.bit(); 2]);
    engine.interp.gc_next = -1;
    assert_eq!(eval(&mut engine, observation), expected);
    assert_eq!(code.residency.active.get(), 0);
    engine.interp.gc_collect();
}

#[test]
fn guarded_private_facts_survive_getters_but_not_completion_landings_or_heap_mutation() {
    let source = r#"
        function subject(n) { 'use strict';
            let result=n+1;
            try { result+=receiver.value; }
            catch(e) { result=e; }
            finally { trace.push('finally'); }
            return result;
        }
        var trace=[],receiver={get value(){trace.push('get');return 3;}};
    "#;
    let observation = r#"
        var first=subject(4);
        Object.defineProperty(receiver,'value',{get(){trace.push('throw');throw 'caught';}});
        var second=subject(4);
        [first,second,trace.join(',')].join('|');
    "#;
    let expected = "8|caught|get,finally,throw,finally";
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    let (_, code) = install(&mut engine, &[ValueClass::NumberDouble.bit()]);
    engine.interp.gc_next = -1;
    assert_eq!(eval(&mut engine, observation), expected);
    assert_eq!(code.residency.active.get(), 0);
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, source);
    assert_eq!(eval(&mut oracle, observation), expected);
}

#[test]
fn entry_feedback_never_specializes_environment_homed_parameters_or_generic_bodies() {
    for source in [
        "function subject(x){arguments[0]='changed';return x+1;}",
        "function subject(x){eval('x=\"changed\"');return x+1;}",
        "function subject(x){return x.value;}",
    ] {
        let statements = crate::parser::parse_script(source, false).ok().unwrap();
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("function")
        };
        let chunk = bytecode::compile(function).unwrap();
        let cfg = Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        let inputs = [(0, ValueClass::NumberInt32.bit())];
        assert!(
            specialization::Plan::build(&chunk, &cfg, &stack, &inputs, &[]).is_none(),
            "{source}"
        );
    }
}

#[test]
fn boolean_entry_guard_keeps_toboolean_without_coercing_objects() {
    let source = r#"
        function subject(flag) { 'use strict';
            let sum=0;for(let k=0;k<3;k++){if(flag)sum++;}return sum;
        }
        var calls=0;
    "#;
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    install(&mut engine, &[ValueClass::Boolean.bit()]);
    let observation = r#"
        var obj={valueOf(){calls++;return 0;}};
        [subject(true),subject(false),subject(0),subject(-0),subject(NaN),
         subject(obj),subject(Symbol('truthy')),calls].join('|');
    "#;
    let expected = "3|0|0|0|0|3|3|0";
    assert_eq!(eval(&mut engine, observation), expected);
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, source);
    assert_eq!(eval(&mut oracle, observation), expected);
}

fn install_warmed_fields(engine: &mut Engine) -> (Rc<Chunk>, Rc<JitCode>) {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let function = {
        let object = value.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("user function")
        };
        user.func.clone()
    };
    let primary = function.code.get().and_then(Option::as_ref).unwrap();
    let properties: Vec<_> = primary
        .jit_ops()
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| {
            let (Op::GetPropLocal(_, _, cache) | Op::GetPropThis(_, cache)) = op else {
                return None;
            };
            primary
                .jit_cache_preferred(*cache)
                .filter(|state| state.depth == 0)
                .map(|_| (pc as u32, ValueClass::NumberDouble.bit()))
        })
        .collect();
    assert!(properties.len() >= 2, "two actual warmed own-field sites");
    let chunk = bytecode::compile_for_optimizer(&function, primary, &[], &properties).unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = Rc::new(compile_with_deopt(&chunk, &values, &layout, None).unwrap());
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.code2.set(Some(chunk.clone())).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Jit);
    (chunk, code)
}

#[test]
fn numeric_field_feedback_keeps_descriptor_prototype_and_value_changes_observable() {
    let source = r#"
        function subject(o) { 'use strict';
            let first=2+o.x;let prefix=first*3;return prefix+o.y;
        }
        var receiver={x:3,y:5},trace=[];
    "#;
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    assert_eq!(eval(&mut engine, "subject(receiver)"), "20");
    let (chunk, code) = install_warmed_fields(&mut engine);
    let observe = r#"
        var normal=subject(receiver);
        receiver.y='s';var string=subject(receiver);
        Object.defineProperty(receiver,'y',{get(){trace.push('get');return 't';},configurable:true});
        var getter=subject(receiver);
        delete receiver.x;Object.setPrototypeOf(receiver,{get x(){trace.push('proto');return 4;}});
        var prototype=subject(receiver);
        Object.defineProperty(receiver,'y',{get(){trace.push('throw');throw 'boom';}});
        var thrown;try{subject(receiver);}catch(e){thrown=e;}
        [normal,string,getter,prototype,thrown,trace.join(',')].join('|');
    "#;
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    let expected = "20|15s|15t|18t|boom|get,proto,get,proto,throw";
    assert_eq!(eval(&mut engine, observe), expected);
    bytecode::native_deopt::ENTRIES.with(|entries| {
        let entries = entries.borrow();
        assert_eq!(entries.len(), 4);
        assert!(entries
            .iter()
            .all(|&(pc, _, _)| matches!(chunk.jit_ops()[pc], Op::GetPropLocal(..))));
    });
    assert_eq!(code.residency.active.get(), 0);
    engine.interp.gc_collect();
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, source);
    assert_eq!(eval(&mut oracle, observe), expected);
}

#[test]
fn numeric_field_guards_recheck_after_reentrant_mutation_and_gc() {
    let source = r#"
        function subject(o,change) { 'use strict';
            let first=o.x;change(o);let sum=first+o.x;
            for(let k=0;k<3;k++)sum+=o.y;
            return sum;
        }
        var receiver={x:2,y:4},trace=[];
        function untouched(){}
    "#;
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    assert_eq!(eval(&mut engine, "subject(receiver,untouched)"), "16");
    let (_, code) = install_warmed_fields(&mut engine);
    engine.interp.gc_next = -1;
    let observe = r#"
        var normal=subject(receiver,untouched);
        var changed=subject(receiver,function(o){trace.push('change');Object.defineProperty(o,'x',{
            get(){trace.push('get');return {valueOf(){trace.push('coerce');return 7;}};},configurable:true});});
        [normal,changed,trace.join(',')].join('|');
    "#;
    let expected = "16|21|change,get,coerce";
    assert_eq!(eval(&mut engine, observe), expected);
    assert_eq!(code.residency.active.get(), 0);
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, source);
    assert_eq!(eval(&mut oracle, observe), expected);
}

#[test]
fn unstable_specialization_retires_without_unmapping_an_active_caller() {
    fn observe(
        i: &mut crate::interpreter::Interp,
        _: crate::Value,
        _: &[crate::Value],
    ) -> Result<crate::Value, crate::Value> {
        RETIRING_VERSION.with(|state| {
            let code = state
                .borrow()
                .as_ref()
                .unwrap()
                .upgrade()
                .expect("active caller mapping");
            assert!(code.residency.active.get() > 0);
        });
        // No test-owned strong native-code lease spans this real collection.
        i.gc_collect();
        RETIRING_VERSION.with(|state| {
            assert!(state.borrow().as_ref().unwrap().upgrade().is_some());
        });
        Ok(crate::Value::Undefined)
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let global = engine.interp.global.clone();
    engine
        .interp
        .def_method(&global, "observeRetirement", 0, observe);
    eval(
        &mut engine,
        r#"
        var trace=[],coercions=0;
        function subject(n,callback) { 'use strict';
            let value=n+1;
            try { callback(); return value*2; }
            finally { trace.push('finally'); }
        }
        function untouched() {}
        function changed(){coercions++;return 2;}
        function reenter(){
            for(let k=0;k<8;k++)subject({valueOf:changed},untouched);
            observeRetirement();
        }
    "#,
    );
    let (chunk, code) = install(&mut engine, &[ValueClass::NumberInt32.bit()]);
    let old = Rc::downgrade(&code);
    RETIRING_VERSION.with(|state| *state.borrow_mut() = Some(old.clone()));
    drop(code);
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    assert_eq!(
        eval(
            &mut engine,
            "[subject(4,reenter),coercions,trace.length].join('|')"
        ),
        "10|8|9"
    );
    assert_eq!(chunk.optimizing_misses.get(), SPECIALIZATION_MISS_LIMIT);
    assert!(!selected(&chunk));
    assert_eq!(
        eval(
            &mut engine,
            r#"
        let last;for(let k=0;k<12;k++)last=subject({valueOf:changed},untouched);
        [last,subject(3,untouched),coercions].join('|');
    "#
        ),
        "6|8|20"
    );
    bytecode::native_deopt::ENTRIES.with(|entries| assert_eq!(entries.borrow().len(), 8));
    assert!(
        chunk.jit.get().flatten().is_some(),
        "later calls use resident template code"
    );
    assert!(
        old.upgrade().is_none(),
        "old mapping released after its final return"
    );
    RETIRING_VERSION.with(|state| *state.borrow_mut() = None);
}
