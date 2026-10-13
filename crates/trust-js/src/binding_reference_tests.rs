//! ECMA-262 IteratorBindingInitialization, KeyedBindingInitialization,
//! RestBindingInitialization and IteratorClose. Local official snapshot e28783d5fc9dc12b3.
//! Var identifiers retain a Reference before reading a source; nested patterns do not.

use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.set((0, 0)));
        match engine.eval(source, false).expect("binding fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}: {source}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}\n{source}"),
        }
        let (vm, native) = crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.get());
        if tier != Tier::Interp {
            assert!(vm + native > 0, "binding Script must not fall back to AST");
        }
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        if tier == Tier::Jit {
            assert!(native > 0, "binding Script must enter native code");
        }
    }
}

#[test]
fn binding_reference_keyed_before_get_and_default_retains_with_base() {
    for property in ["p", "[key]"] {
        let source = format!(
            r#"
            var trace=[], box={{x:0}}, x='outer';
            var scope=new Proxy(box,{{has(t,k){{
                if(k==='x')trace.push('has');return Reflect.has(t,k);
            }}}});
            var key={{[Symbol.toPrimitive](hint){{trace.push(hint);return 'p'}}}};
            var source={{get p(){{trace.push('get');delete box.x;$262.gc();return undefined}}}};
            with(scope){{var {{{property}:x=(trace.push('default'),17)}}=source;}}
            [trace.join(','),x,box.x].join('|');
            "#
        );
        let expected = if property == "p" {
            "has,get,default,has|outer|17"
        } else {
            "string,has,get,default,has|outer|17"
        };
        check(&source, expected);
    }
    // The plain-property/no-default emitter has a separate compact GetProp path.
    check(
        r#"
        var trace=[],box={x:0},x='outer';
        var scope=new Proxy(box,{has(t,k){if(k==='x')trace.push('has');return Reflect.has(t,k)}});
        with(scope){var {p:x}={get p(){trace.push('get');delete box.x;return 19}};}
        [trace.join(','),x,box.x].join('|');
        "#,
        "has,get,has|outer|19",
    );
}

#[test]
fn binding_reference_array_before_step_default_and_close() {
    check(
        r#"
        var trace=[],box={x:0},x='outer';
        var scope=new Proxy(box,{has(t,k){if(k==='x')trace.push('has');return Reflect.has(t,k)}});
        var source={[Symbol.iterator](){trace.push('iter');return {
            next(){trace.push('next');delete box.x;$262.gc();return {done:false}},
            return(){trace.push('close');return {}}
        }}};
        with(scope){var [x=(trace.push('default'),23)]=source;}
        [trace.join(','),x,box.x].join('|');
        "#,
        "iter,has,next,default,has,close|outer|23",
    );
}

#[test]
fn binding_reference_array_rest_resolves_before_draining() {
    check(
        r#"
        var trace=[],box={x:0},x='outer',n=0;
        var scope=new Proxy(box,{has(t,k){if(k==='x')trace.push('has');return Reflect.has(t,k)}});
        var source={[Symbol.iterator](){trace.push('iter');return {
            next(){trace.push('next');delete box.x;$262.gc();return {value:++n,done:n===3}},
            return(){trace.push('unexpected-close');return {}}
        }}};
        with(scope){var [...x]=source;}
        [trace.join(','),x,box.x.join(',')].join('|');
        "#,
        "iter,has,next,next,next,has|outer|1,2",
    );
}

#[test]
fn binding_reference_object_rest_resolves_before_proxy_copy() {
    check(
        r#"
        var trace=[],box={x:0},x='outer';
        var scope=new Proxy(box,{has(t,k){if(k==='x')trace.push('has');return Reflect.has(t,k)}});
        var source=new Proxy({p:29},{
            ownKeys(t){trace.push('keys');delete box.x;$262.gc();return Reflect.ownKeys(t)},
            getOwnPropertyDescriptor(t,k){trace.push('desc:'+k);return Reflect.getOwnPropertyDescriptor(t,k)},
            get(t,k){trace.push('get:'+k);return t[k]}
        });
        with(scope){var {...x}=source;}
        [trace.join(','),x,box.x.p].join('|');
        "#,
        "has,keys,desc:p,get:p,has|outer|29",
    );
}

#[test]
fn binding_reference_nested_patterns_resolve_after_outer_read() {
    check(
        r#"
        var trace=[],box={x:0},x='outer';
        var scope=new Proxy(box,{has(t,k){if(k==='x')trace.push('has');return Reflect.has(t,k)}});
        var source={[Symbol.iterator](){trace.push('iter');return {
            next(){trace.push('next');return {value:{get p(){trace.push('get');delete box.x;return 31}},done:false}},
            return(){trace.push('close');return {}}
        }}};
        with(scope){var [{p:x}]=source;}
        [trace.join(','),x,box.x].join('|');
        "#,
        "iter,next,has,get,has,close|outer|31",
    );
    check(
        r#"
        var trace=[],box={x:0},x='outer',n=0;
        var scope=new Proxy(box,{has(t,k){if(k==='x')trace.push('has');return Reflect.has(t,k)}});
        var source={[Symbol.iterator](){trace.push('iter');return {
            next(){trace.push('next');return {value:37,done:++n===2}},
            return(){trace.push('unexpected-close');return {}}
        }}};
        with(scope){var [...[x]]=source;}
        [trace.join(','),x,box.x].join('|');
        "#,
        "iter,next,next,has,has|outer|37",
    );
    check(
        r#"
        var trace=[],box={x:0},x='outer';
        var scope=new Proxy(box,{has(t,k){if(k==='x')trace.push('has');return Reflect.has(t,k)}});
        with(scope){var {p:{q:x}}={get p(){trace.push('outer');return {get q(){trace.push('inner');return 41}}}};}
        [trace.join(','),x,box.x].join('|');
        "#,
        "outer,has,inner,has|outer|41",
    );
}

#[test]
fn binding_reference_failure_closes_without_stepping_and_keeps_original_throw() {
    for pattern in ["[x]", "[...x]"] {
        check(
            &format!(
                r#"
                var trace=[],sentinel={{}},x;
                var scope=new Proxy({{x:0}},{{has(t,k){{if(k==='x'){{trace.push('has');throw sentinel}}return Reflect.has(t,k)}}}});
                var source={{[Symbol.iterator](){{trace.push('iter');return {{
                    next(){{trace.push('unexpected-next');return {{done:true}}}},
                    return(){{trace.push('close');throw 'replacement'}}
                }}}}}};
                var preserved=false;
                try{{with(scope){{var {pattern}=source;}}}}catch(e){{preserved=e===sentinel;}}
                [preserved,trace.join(',')].join('|');
                "#
            ),
            "true|iter,has,close",
        );
    }
}

#[test]
fn binding_reference_iterator_errors_do_not_close_but_default_errors_do() {
    for (step, expected) in [
        ("throw sentinel", "iter,has,next"),
        (
            "return {get done(){trace.push('done');throw sentinel}}",
            "iter,has,next,done",
        ),
        (
            "return {done:false,get value(){trace.push('value');throw sentinel}}",
            "iter,has,next,value",
        ),
        ("return {done:false}", "iter,has,next,default,close"),
    ] {
        check(
            &format!(
                r#"
                var trace=[],sentinel={{}},x;
                var scope=new Proxy({{x:0}},{{has(t,k){{if(k==='x')trace.push('has');return Reflect.has(t,k)}}}});
                var source={{[Symbol.iterator](){{trace.push('iter');return {{
                    next(){{trace.push('next');{step}}},
                    return(){{trace.push('close');throw 'replacement'}}
                }}}}}};
                function fail(){{trace.push('default');throw sentinel;}}
                var preserved=false;
                try{{with(scope){{var [x=fail()]=source;}}}}catch(e){{preserved=e===sentinel;}}
                [preserved,trace.join(',')].join('|');
                "#
            ),
            &format!("true|{expected}"),
        );
    }
}

#[test]
fn binding_reference_unscopables_changes_do_not_redirect_and_defaults_keep_names() {
    check(
        r#"
        var x='outer',box={x:0,[Symbol.unscopables]:{x:false}},trace=[];
        var scope=new Proxy(box,{has(t,k){if(k==='x')trace.push('has');return Reflect.has(t,k)}});
        var source={get p(){box[Symbol.unscopables].x=true;trace.push('get');return undefined}};
        with(scope){var {p:x=function(){}}=source;}
        [trace.join(','),x,box.x.name].join('|');
        "#,
        "has,get,has|outer|x",
    );
    check(
        r#"
        var x='outer',box={x:0,[Symbol.unscopables]:{x:true}},trace=[];
        var scope=new Proxy(box,{has(t,k){if(k==='x')trace.push('has');return Reflect.has(t,k)}});
        var source={[Symbol.iterator](){return {
            next(){box[Symbol.unscopables].x=false;trace.push('next');return {done:false}},
            return(){trace.push('close');return {}}
        }}};
        with(scope){var [x=function(){}]=source;}
        [trace.join(','),x.name,box.x].join('|');
        "#,
        "has,next,close|x|0",
    );
}

#[cfg(feature = "embed")]
#[test]
fn binding_reference_host_interrupt_bypasses_cleanup_and_releases_activation() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for phase in ["resolve", "next", "default"] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            // Arm only once execution reaches the requested semantic boundary. This avoids
            // relying on compile duration, scheduling or a wall-clock sleep in the fixture.
            engine.define_global("interruptNow", 0, |ctx, _, _| {
                ctx.runtime_interrupt
                    .set_deadline(Some(std::time::Instant::now()));
                Ok(crate::Value::Undefined)
            });
            let source = format!(
                r#"
                var caught=0,finalized=0,closed=0,phase='{phase}',x='outer',box={{x:0}};
                function stop(at){{if(at===phase){{interruptNow();while(true){{}}}}}}
                var scope=new Proxy(box,{{has(t,k){{if(k==='x')stop('resolve');return Reflect.has(t,k)}}}});
                var source={{[Symbol.iterator](){{return {{
                    next(){{stop('next');return {{done:false}}}},
                    return(){{closed++;return {{}}}}
                }}}}}};
                try{{with(scope){{var [x=stop('default')]=source;}}}}
                catch(e){{caught++;}}finally{{finalized++;}}
                "#
            );
            let result = engine
                .eval_interruptible(&source, false)
                .expect("interrupt fixture parses");
            assert!(
                matches!(
                    result,
                    crate::ExecutionOutcome::Interrupted {
                        reason: crate::InterruptReason::DeadlineExceeded
                    }
                ),
                "{tier:?}/{phase}: host interrupt must not become a JS throw"
            );
            engine.interrupt_handle().set_deadline(None);
            match engine
                .eval("[caught,finalized,closed,x,box.x].join('|')", false)
                .unwrap()
            {
                Completion::Value(value) => assert_eq!(value, "0|0|0|outer|0", "{tier:?}/{phase}"),
                Completion::Throw { name, message } => {
                    panic!("{tier:?}/{phase}: {name}: {message}")
                }
            }
        }
    }
}
