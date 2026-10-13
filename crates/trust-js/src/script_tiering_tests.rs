//! ScriptEvaluation / UpdateEmpty / ForBodyEvaluation / TryStatement Evaluation.
//! ECMA-262 local official snapshot e28783d5fc9dc12b3de905961e2c71410b38a202.
//! OSR changes execution machinery only: declarations, side effects, identity and
//! completion records must not be recreated or replayed at a hot header.
use crate::{bytecode::Tier, Completion, Engine};

fn value(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("tiering fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn reset_counts() {
    crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.set((0, 0)));
    crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.set(0));
    crate::jit::TEST_OSR_ENTRIES.with(|count| count.set(0));
}

fn assert_osr(tier: Tier) {
    let entries = crate::jit::TEST_OSR_ENTRIES.with(|count| count.get());
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "linux", target_os = "macos", target_os = "windows")
    ))]
    if matches!(tier, Tier::Jit) {
        assert!(entries > 0, "must enter actual borrowed native code");
    }
    if !matches!(tier, Tier::Jit) {
        assert_eq!(entries, 0);
    }
}

fn check_hot(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        reset_counts();
        assert_eq!(value(&mut engine, source), expected, "{tier:?}");
        if !matches!(tier, Tier::Interp) {
            assert!(
                crate::bytecode::loop_fragment::TEST_FRAGMENT_ENTRIES.with(|count| count.get()) > 0,
                "hot Script must first transfer its live AST loop to baseline VM: {source}"
            );
        }
        assert_osr(tier);
    }
}

#[test]
fn cold_scripts_and_short_loops_do_not_compile_native_entries() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        for (source, expected, baseline) in [
            ("", "undefined", false),
            ("1+2", "3", false),
            ("function unused(){while(true){}}; 4", "4", false),
            ("var arrow=()=>{for(;;){}}; 5", "5", false),
            ("class C{static{for(let k=0;k<3;k++){}}};6", "6", false),
            ("var n=0;for(var k=0;k<100;k++)n+=k;n", "4950", false),
            ("if(false){for(;;){}};7", "7", false),
        ] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(32);
            reset_counts();
            assert_eq!(value(&mut engine, source), expected);
            assert_eq!(crate::jit::TEST_OSR_ENTRIES.with(|count| count.get()), 0);
            assert_eq!(
                crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.get()),
                (usize::from(baseline), 0),
                "{tier:?}: {source}"
            );
        }
    }
}

#[test]
fn own_loop_scan_borrows_controls_but_not_nested_function_activations() {
    for source in [
        "if(false){while(false){}}",
        "try{}catch(e){for(;;){}}",
        "try{}finally{do{}while(false)}",
        "switch(0){case 1:for(;;){}}",
        "label:with({}){for(var x in {}){}}",
        "for(const x of []){}",
    ] {
        let body = crate::parser::parse_script(source, false)
            .ok()
            .expect("scanner source");
        assert!(crate::ast::statement_list_has_own_loop(&body), "{source}");
    }
    for source in [
        "function f(){for(;;){}}",
        "(()=>{for(;;){}})",
        "class C{m(){for(;;){}} static{for(;;){}}}",
        "if(false){function f(){for(;;){}}}",
    ] {
        let body = crate::parser::parse_script(source, false)
            .ok()
            .expect("scanner source");
        assert!(!crate::ast::statement_list_has_own_loop(&body), "{source}");
    }
}

#[test]
fn hot_script_preserves_gdi_side_effects_and_escaped_iteration_identity() {
    check_hot(
        r#"
        var calls=0, before=hoisted, retained=[], n=0;
        function hoisted(){return 7;}
        var initial=++calls;
        for(let k=0;k<2000;k++){
            let object={k}; n+=k;
            if(k===0||k===127||k===1999) retained.push(()=>[k,object.k]);
        }
        [calls,initial,before===hoisted,n,retained.map(f=>f().join(':')).join(',')].join('|');
    "#,
        "1|1|true|1999000|0:0,127:127,1999:1999",
    );
}

#[test]
fn hot_script_preserves_completion_values_across_empty_and_abrupt_finalizers() {
    check_hot("var k=0;for(;k<2000;k++){k;} ;{}", "1999");
    check_hot(
        r#"
        var cleaned=0, caught=0;
        try{for(var k=0;k<2000;k++){
            try{if(k===1777)throw 42; if(k%3===0)continue;}
            finally{cleaned++;}
        }}catch(e){caught=e;}
        [k,cleaned,caught].join('|');
    "#,
        "1777|1778|42",
    );
    check_hot(
        r#"
        var count=0;
        outer:for(var k=0;k<2000;k++){
            try{if(k===1700)break outer;continue;}
            finally{count++;}
        }
        [k,count].join('|');
    "#,
        "1700|1701",
    );
}

#[test]
fn hot_script_conditional_backedges_and_nested_loops_settle_once() {
    check_hot(
        "var k=0,n=0;do{n+=k++;}while(k<2000);[k,n].join('|')",
        "2000|1999000",
    );
    check_hot(
        r#"
        var n=0,outer=0;
        for(var a=0;a<4;a++){
            outer++;
            for(var b=0;b<1000;b++){if((b&1)===0)n++;else n+=2;}
        }
        [outer,n,a,b].join('|');
    "#,
        "4|6000|4|1000",
    );
}

#[test]
fn hot_script_keeps_iterator_and_disposal_state_at_transfer() {
    check_hot(
        r#"
        var next=0,closed=0,disposed=0,total=0;
        var source={[Symbol.iterator](){return {
            next(){return {value:++next,done:next>2000}},
            return(){closed++;return {}}
        }}};
        for(const item of source){
            using resource={[Symbol.dispose](){disposed++;}};
            total+=item;
            if(item===1700)break;
        }
        [next,closed,disposed,total].join('|');
    "#,
        "1700|1|1700|1445850",
    );
    check_hot(
        r#"
        var disposed=0,caught=0,total=0;
        try{
            using resource={[Symbol.dispose](){disposed++;throw 42;}};
            for(var k=0;k<2000;k++)total+=k;
        }catch(e){caught=e;}
        [disposed,caught,total].join('|');
    "#,
        "1|42|1999000",
    );
}

#[test]
fn hot_script_keeps_dynamic_scope_and_catch_binding_resolution() {
    check_hot(
        r#"
        var x=1, object={x:2}, total=0;
        with(object){for(var k=0;k<2000;k++){
            try{throw k;}catch(x){var x=x+1;total+=x;}
            x++;
        }}
        [x,object.x,total].join('|');
    "#,
        "1|2002|2001000",
    );
}

#[test]
fn hot_script_borrowed_this_is_the_global_environment_this_value() {
    use crate::value::{Object, Property, Value};
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        let wrapper = Object::new(Some(engine.interp.object_proto.clone()));
        wrapper
            .borrow_mut()
            .props
            .insert("marker", Property::data(Value::Num(9.0), true, true, true));
        engine
            .interp
            .global_env
            .borrow_mut()
            .vars
            .get_mut("this")
            .unwrap()
            .value = Value::Obj(wrapper);
        reset_counts();
        assert_eq!(
            value(
                &mut engine,
                r#"
            var total=0;
            for(var k=0;k<2000;k++)total+=this.marker;
            [total,this!==globalThis,globalThis.total].join('|');
        "#
            ),
            "18000|true|18000"
        );
        assert_osr(tier);
    }
}

#[test]
fn hot_script_completion_owner_survives_gc_at_borrowed_entry_and_finalizer() {
    use crate::value::{Property, Value};
    use std::rc::Rc;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        let collect = engine.interp.make_native("collect", 0, |ctx, _, _| {
            ctx.gc_collect();
            Ok(Value::Undefined)
        });
        engine.interp.global.borrow_mut().props.insert(
            "collect",
            Property::data(Value::Obj(collect), true, true, true),
        );
        let body = crate::parser::parse_script(
            r#"
            try{for(let k=0;k<2000;k++){
                let result={k};result.self=result;
                if(k%127===0)collect();result;
            }}finally{collect();}
        "#,
            false,
        )
        .ok()
        .expect("GC fixture parses");
        reset_counts();
        let result = engine
            .interp
            .run_program(&body)
            .ok()
            .expect("GC fixture completes");
        assert_osr(tier);
        let Value::Obj(object) = &result else {
            panic!("completion is owned object");
        };
        assert!(matches!(
            object.borrow().props.get("k").unwrap().value(),
            Value::Num(1999.0)
        ));
        let weak = Rc::downgrade(object);
        drop(result);
        drop(body);
        engine.interp.gc_collect();
        assert!(weak.upgrade().is_none(), "completion leaked after {tier:?}");
    }
}

#[test]
fn hot_script_interrupt_restores_borrowed_owners_without_author_cleanup() {
    use crate::value::{Property, Value};
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        let handle = engine.interrupt_handle();
        let expire = engine.interp.make_native("expire", 0, |ctx, _, _| {
            ctx.runtime_interrupt
                .set_deadline(Some(std::time::Instant::now()));
            Ok(Value::Undefined)
        });
        engine.interp.global.borrow_mut().props.insert(
            "expire",
            Property::data(Value::Obj(expire), true, true, true),
        );
        reset_counts();
        let outcome = engine
            .eval_interruptible(
                r#"
            var caught=0,finalized=0,n=0;
            try{while(true){if(++n===1500)expire();}}
            catch(e){caught++;}finally{finalized++;}
        "#,
                false,
            )
            .expect("interrupt fixture parses");
        assert!(matches!(
            outcome,
            crate::ExecutionOutcome::Interrupted {
                reason: crate::InterruptReason::DeadlineExceeded
            }
        ));
        assert_osr(tier);
        handle.set_deadline(None);
        assert_eq!(value(&mut engine, "caught+'|'+finalized"), "0|0");
        engine.interp.gc_collect();
    }
}

#[cfg(feature = "embed")]
#[test]
fn hot_script_native_callback_keeps_script_caller_and_entry_identity() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        engine.define_global("caller", 0, |ctx, _, _| Ok(ctx.script_caller_global()));
        engine.define_global("entry", 0, |ctx, _, _| Ok(ctx.script_entry_global()));
        reset_counts();
        assert_eq!(
            value(
                &mut engine,
                r#"
            var okay=true;
            for(var k=0;k<2000;k++)okay=okay&&caller()===globalThis&&entry()===globalThis;
            okay;
        "#
            ),
            "true"
        );
        assert_osr(tier);
    }
}
