//! ECMA-262 ScriptEvaluation, GlobalDeclarationInstantiation, StatementList/UpdateEmpty and
//! TryStatement Evaluation. Local official snapshot e28783d5fc9dc12b3de905961e2c71410b38a202.

use crate::{bytecode::Tier, Completion, Engine};

fn value(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("script fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.set((0, 0)));
        assert_eq!(value(&mut engine, source), expected, "{tier:?}: {source}");
        let (vm, native) = crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.get());
        if !matches!(tier, Tier::Interp) {
            assert!(
                vm + native > 0,
                "must execute compiled script, not AST fallback: {source}"
            );
        }
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        if matches!(tier, Tier::Jit) {
            assert!(native > 0, "must enter native Script code: {source}");
        }
    }
}

#[test]
fn compiled_script_statement_completions_keep_empty_distinct_from_undefined() {
    for (source, expected) in [
        ("", "undefined"),
        ("1;;;;{};var x;", "1"),
        ("1;let x;const y=2;class C{}", "1"),
        ("1;if(false)2;", "undefined"),
        ("1;if(true){}else 3;", "undefined"),
        ("1;label:{break label;}", "1"),
        ("1;label:{2;{break label;}}", "2"),
        ("1;label:{2;if(true)break label;}", "undefined"),
        ("1;while(false)2;", "undefined"),
        ("for(var i=0;i<3;i++){i;}", "2"),
        ("for(var i=0;i<3;i++){i;continue;}", "2"),
        ("for(var i=0;i<3;i++){i;break;}", "0"),
        ("for(var i=0;i<3;i++){i;if(i===2)break;}", "undefined"),
        ("var i=0;do{i++;}while(i<3);", "2"),
        ("1;L:for(var x of [2,3]){x;break L;}", "2"),
        ("1;for(var key in null)2;", "undefined"),
        ("1;switch(5){}", "undefined"),
        ("switch(1){case 1:3;case 2:break;default:4;}", "3"),
        ("switch(9){case 1:3;break;default:4;case 2:5;}", "5"),
        ("1;with({}){}", "undefined"),
    ] {
        check(source, expected);
    }
}

#[test]
fn compiled_script_finalizers_preserve_or_replace_exact_completion_values() {
    for (source, expected) in [
        ("try{4;}finally{9;}", "4"),
        ("1;try{}finally{2;}", "undefined"),
        ("try{5;throw 6;}catch(e){}", "undefined"),
        ("try{5;throw 6;}catch(e){e+1;}finally{9;}", "7"),
        ("L:{try{7;break L;}finally{8;}}", "7"),
        ("L:{try{7;}finally{8;break L;}}", "8"),
        ("L:{try{7;}finally{break L;}}", "undefined"),
        ("L:{try{try{3;break L;}finally{4;}}finally{5;}}", "3"),
        ("L:{try{throw 9;}finally{6;break L;}}", "6"),
        (
            "try{L:{try{7;break L;}finally{throw 8;}}}catch(e){e+1;}",
            "9",
        ),
        (
            "for(var i=0;i<2;i++){try{i+10;continue;}finally{99;}}",
            "11",
        ),
        (
            "for(var i=0;i<2;i++){try{i+10;}finally{i+20;continue;}}",
            "21",
        ),
    ] {
        check(source, expected);
    }
}

#[test]
fn compiled_script_globals_share_properties_cells_and_hoisted_function_identity() {
    check(
        r#"
        var seen=hoisted, n=2;
        function hoisted(){return lexical+n;}
        let lexical=3; const fixed=7; class C {value(){return fixed;}}
        function capture(){return [lexical,n,hoisted===seen,new C().value()];}
        globalThis.n=5; lexical=11;
        var property=Object.getOwnPropertyDescriptor(globalThis,'n');
        [capture().join(','),hoisted(),property.writable,property.enumerable,
         property.configurable,Object.hasOwn(globalThis,'lexical'),
         Object.hasOwn(globalThis,'fixed'),Object.hasOwn(globalThis,'C')].join('|');
    "#,
        "11,5,true,7|16|true|true|false|false|false|false",
    );
}

#[test]
fn compiled_script_root_lexicals_keep_tdz_const_and_destructuring_order() {
    check(
        r#"
        var trace=[];
        try{later;}catch(e){trace.push(e.name);}
        let later=3; const fixed=4;
        try{fixed=5;}catch(e){trace.push(e.name);}
        var step=0;
        var iterable={[Symbol.iterator](){return {next(){
            step++;trace.push(step===1?'first':String(first));
            return {value:step*10,done:false};
        },return(){trace.push('close');return {};}};}};
        let [first,second]=iterable;
        trace.push(later,fixed,first,second);trace.join('|');
    "#,
        "ReferenceError|TypeError|first|10|close|3|4|10|20",
    );
}

#[test]
fn compiled_script_var_references_precede_rhs_and_with_base_changes() {
    check(
        r#"
        var x=1, o={x:2}, trace=[];
        with(o){var x=(delete o.x,5);}
        trace.push(x,o.x);
        with(o){for(var x=(delete o.x,6);false;){};}
        trace.push(x,o.x);
        o[Symbol.unscopables]={x:true};
        with(o){var x=(o[Symbol.unscopables].x=false,7);}
        trace.push(x,o.x);
        var a=0,b=0;
        with({a:3}){var [a,b]=[8,9];}
        trace.push(a,b);trace.join('|');
    "#,
        "1|5|1|6|7|6|0|9",
    );
}

#[test]
fn compiled_script_captures_keep_fresh_block_and_iteration_environments() {
    check(
        r#"
        var callbacks=[];
        for(let n=0;n<3;n++){let local=n+10;callbacks.push(()=>[n,local]);}
        {let local=77;callbacks.push(()=>local);}
        var [a,b]=[1,2], {x:c=3,...rest}={y:4};
        for(var [a,b] of [[5,6],[7,8]]){}
        [callbacks.map(f=>String(f())).join(';'),a,b,c,rest.y].join('|');
    "#,
        "0,10;1,11;2,12;77|7|8|3|4",
    );
}

#[test]
fn compiled_script_annex_b_writeback_targets_global_not_block_or_with() {
    check(
        r#"
        var before=typeof promoted;
        {function promoted(){return 7;} var inside=promoted;}
        var object={promoted:'with'};
        with(object){{function promoted(){return 8;} var next=promoted;}}
        if(false){function skipped(){return 9;}}
        [before,inside(),promoted(),promoted===next,object.promoted,
         typeof skipped,Object.getOwnPropertyDescriptor(globalThis,'promoted').configurable].join('|');
    "#,
        "undefined|7|8|true|with|undefined|false",
    );
    check(
        "'use strict'; {function hidden(){return 1;}} typeof hidden;",
        "undefined",
    );
}

#[test]
fn compiled_script_eval_this_arguments_and_strictness_stay_script_scoped() {
    check(
        r#"
        var ordinaryThis=this;
        let outer=2; {let inner=3;var seen=eval('outer+inner');}
        eval('var introduced=9');
        var arrows=()=>()=>this;
        [seen,introduced,arrows()()===globalThis,ordinaryThis===globalThis,
         typeof arguments,eval('typeof inner')].join('|');
    "#,
        "5|9|true|true|undefined|undefined",
    );
    check(
        r#"'use strict'; eval('var hidden=1');
        [this===globalThis,typeof hidden,typeof arguments].join('|');"#,
        "true|undefined|undefined",
    );
}

#[test]
fn compiled_script_successive_declarations_validate_before_body_effects() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        assert_eq!(
            value(&mut engine, "var effect=0;let lexical=3;effect;"),
            "0"
        );
        assert!(
            matches!(engine.eval("effect=1;var lexical;", false).unwrap(),
            Completion::Throw { ref name, .. } if name == "SyntaxError")
        );
        assert_eq!(value(&mut engine, "effect+'|'+lexical;"), "0|3");
        assert!(
            matches!(engine.eval("effect=2;let lexical;", false).unwrap(),
            Completion::Throw { ref name, .. } if name == "SyntaxError")
        );
        assert_eq!(value(&mut engine, "effect;"), "0");
        assert_eq!(
            value(&mut engine, "Object.preventExtensions(globalThis);0;"),
            "0"
        );
        assert!(
            matches!(engine.eval("effect=3;var absent;", false).unwrap(),
            Completion::Throw { ref name, .. } if name == "TypeError")
        );
        assert_eq!(value(&mut engine, "effect;"), "0");
    }
}

#[test]
fn compiled_script_uses_global_environment_this_value_not_property_object() {
    use crate::value::{Object, Property, Value};
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        let wrapper = Object::new(Some(engine.interp.object_proto.clone()));
        wrapper
            .borrow_mut()
            .props
            .insert("marker", Property::data(Value::Num(41.0), true, true, true));
        engine
            .interp
            .global_env
            .borrow_mut()
            .vars
            .get_mut("this")
            .unwrap()
            .value = Value::Obj(wrapper);
        assert_eq!(value(&mut engine,
            "var actual=3;let get=()=>this;[this.marker,get()===this,this!==globalThis,globalThis.actual].join('|');"),
            "41|true|true|3");
    }
}

#[test]
fn compiled_script_completion_owners_survive_finalizer_gc_and_are_released() {
    use crate::value::{Property, Value};
    use std::rc::Rc;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let collect = engine.interp.make_native("collect", 0, |ctx, _, _| {
            ctx.gc_collect();
            Ok(Value::Undefined)
        });
        engine.interp.global.borrow_mut().props.insert(
            "collect",
            Property::data(Value::Obj(collect), true, true, true),
        );
        let body = crate::parser::parse_script(
            "try {(function(){var result={n:42};result.self=result;return result;})();} finally {collect();}",
            false).ok().expect("GC script parses");
        let result = engine
            .interp
            .run_program(&body)
            .ok()
            .expect("GC script completes");
        let Value::Obj(object) = &result else {
            panic!("completion lost on {tier:?}");
        };
        assert!(matches!(
            object.borrow().props.get("n").unwrap().value(),
            Value::Num(42.0)
        ));
        let weak = Rc::downgrade(object);
        drop(result);
        drop(body);
        engine.interp.gc_collect();
        assert!(
            weak.upgrade().is_none(),
            "hidden Script completion retained on {tier:?}"
        );
    }
}

#[test]
fn compiled_script_top_level_deadline_bypasses_author_handlers_and_recovers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        let interrupt = engine.interrupt_handle();
        interrupt.set_deadline(Some(
            std::time::Instant::now() + std::time::Duration::from_millis(30),
        ));
        let result = engine.eval_interruptible(
            "var caught=0,finalized=0;try{while(true){}}catch(e){caught++;}finally{finalized++;}", false).unwrap();
        assert!(
            matches!(
                result,
                crate::ExecutionOutcome::Interrupted {
                    reason: crate::InterruptReason::DeadlineExceeded
                }
            ),
            "deadline escaped as language completion on {tier:?}"
        );
        interrupt.set_deadline(None);
        assert_eq!(value(&mut engine, "caught+'|'+finalized"), "0|0");
    }
}

#[cfg(feature = "embed")]
#[test]
fn compiled_script_nested_host_entry_masks_outer_function_caller_and_restores_it() {
    use crate::Value;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.define_global("caller", 0, |ctx, _, _| Ok(ctx.script_caller_global()));
        engine.define_global("runChildScript", 1, |ctx, _, args| {
            let result = ctx.with_embed_realm(&args[0], |inner| {
                inner.eval_classic_script_interruptible(
                    "globalThis.callback=()=>parent.caller();parent.caller()===globalThis;",
                )
            })?;
            match result {
                Ok(Ok(value)) => Ok(value),
                Ok(Err(crate::embed::EvalError::Throw(error))) => Err(error),
                _ => Err(Value::lstr("nested Script failed".to_owned())),
            }
        });
        let root = engine.global_this();
        let child = engine.ctx().create_embed_realm();
        engine
            .ctx()
            .member_set(&root, "child", child.clone())
            .ok()
            .unwrap();
        engine
            .ctx()
            .member_set(&child, "parent", root)
            .ok()
            .unwrap();
        assert_eq!(
            value(
                &mut engine,
                r#"
            function invoke(){return runChildScript(child);}
            var first;for(var n=0;n<12;n++)first=invoke();
            [first,child.callback()===child,caller()===globalThis].join('|');
        "#
            ),
            "true|true|true"
        );
    }
}
