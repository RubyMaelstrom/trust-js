//! Force a before-effect bailout at each static PC, including completion landing pads.
//! This stress mode checks reconstruction; it is not a speed benchmark or a type policy.

use super::compile_with_deopt;
use crate::bytecode::{self, native_deopt::ENTRIES, Op, Tier};
use crate::interpreter::Interp;
use crate::value::{Callable, Value};
use crate::{Completion, Engine};
use std::collections::HashSet;
use std::mem::{discriminant, Discriminant};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("deoptimization fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn sweep(
    source: &str,
    arguments: &[f64],
    state: &str,
    expected: &str,
    install: Option<fn(&mut Engine)>,
    after: Option<fn(&mut Engine)>,
) -> HashSet<Discriminant<Op>> {
    let statements = crate::parser::parse_script(source, false).ok().unwrap();
    let crate::ast::Stmt::FuncDecl(parsed_function) = &statements[0] else {
        panic!("first statement declares subject")
    };
    let argument_source = arguments
        .iter()
        .map(f64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let observation = format!(
        "var observed;try{{observed='return:'+String(subject({argument_source}));}}\
         catch(error){{observed='throw:'+String(error);}}observed+'|'+({state});"
    );
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    if let Some(install) = install {
        install(&mut oracle);
    }
    assert_eq!(
        eval(&mut oracle, &format!("{source}\n{observation}")),
        expected,
        "independent normative expected result / interpreter oracle"
    );
    if let Some(after) = after {
        after(&mut oracle);
    }
    let mut kinds = HashSet::new();
    let count = bytecode::compile(parsed_function).unwrap().jit_ops().len();
    let mut reached = 0;
    for pc in 0..=count {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        if let Some(install) = install {
            install(&mut engine);
        }
        eval(&mut engine, source);
        let function = function(&mut engine, "subject");
        let chunk = bytecode::compile(&function).expect("stress bytecode");
        let layout = crate::interpreter::interp_layout(&mut engine.interp);
        let values = crate::value::jit_layout(&engine.interp.object_proto);
        let code = Rc::new(
            compile_with_deopt(&chunk, &values, &layout, Some(pc))
                .unwrap_or_else(|error| panic!("PC {pc}: {error}")),
        );
        chunk
            .jit
            .begin_compile()
            .unwrap()
            .commit(Some(code.clone()));
        assert!(function.code.set(Some(chunk.clone())).is_ok());
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        // Use the real Call entry, including FunctionDeclarationInstantiation, mapped
        // arguments, prepared environments, FnFrame identity and proper-tail-call teardown.
        // Calling jit::run directly cannot synthesize those general-entry contracts.
        assert_eq!(eval(&mut engine, &observation), expected, "PC {pc}");
        assert_eq!(code.residency.active.get(), 0);
        ENTRIES.with(|entries| {
            let entries = entries.borrow();
            assert!(entries.len() <= 1, "a bailout is terminal, not a reentry");
            if let Some(&(actual_pc, _, floor)) = entries.first() {
                assert_eq!(actual_pc, pc);
                assert_eq!(floor, 0);
                reached += 1;
                if let Some(op) = chunk.jit_ops().get(pc) {
                    kinds.insert(discriminant(op));
                }
            }
        });
        if let Some(after) = after {
            after(&mut engine);
        }
    }
    assert!(
        reached > count / 3,
        "stress fixture must execute substantial PC coverage"
    );
    eprintln!(
        "[bailout-sweep] {} compiled boundaries, {reached} actually reached; expected {expected}",
        count + 1
    );
    kinds
}

#[test]
fn bailout_sweep_keeps_expression_owners_effect_order_and_catch_state() {
    let kinds = sweep(
        r#"
        function subject() {
            var box={v:3}, n=4;
            function number(tag,value) {return {valueOf(){trace.push(tag);return value;}};}
            var left=number('L',2), right=number('R',5);
            box.v += (trace.push('rhs'),left+right);
            var result=box.v*(n+1);
            try {if(result===50)throw {label:'boom'};}
            catch(e) {trace.push(e.label);result+=1;}
            finally {trace.push('finally');result+=2;}
            return result+'|'+box.v;
        }
        var trace=[];
        "#,
        &[],
        "trace.join(',')",
        "return:53|10|rhs,L,R,boom,finally",
        None,
        None,
    );
    for op in [Op::Add, Op::Throw, Op::Return] {
        assert!(
            kinds.contains(&discriminant(&op)),
            "bailout actually reached {op:?}"
        );
    }
}

#[test]
fn bailout_sweep_preserves_resolved_reference_after_rhs_changes_resolution() {
    // Resolve the global Reference before eval introduces a new function-local binding of
    // the same name. Re-resolving at StoreRef would silently write 6 into the wrong scope.
    let kinds = sweep(
        r#"
        function subject() {
            item += (eval('var item=40'),trace.push('rhs'),2);
            return item+':'+globalThis.item;
        }
        var item=4,trace=[];
        "#,
        &[],
        "trace.join(',')",
        "return:40:6|rhs",
        None,
        None,
    );
    for op in [Op::LoadRef(0), Op::StoreRef(0)] {
        assert!(
            kinds.contains(&discriminant(&op)),
            "must retain prepared Reference at {op:?}"
        );
    }
}

#[test]
fn bailout_sweep_keeps_iterator_close_precedence_after_consumed_operands() {
    sweep(
        r#"
        function subject() {
            var source={ [Symbol.iterator](){return {
                next(){trace.push('next');return {done:false,value:undefined};},
                return(){trace.push('close');throw 'closing';}
            };}};
            function fail(){trace.push('default');throw 'original';}
            try {let [value=fail()]=source;trace.push('unreachable');}
            catch(error){trace.push(error);}
            return trace.length;
        }
        var trace=[];
        "#,
        &[],
        "trace.join(',')",
        "return:4|next,default,close,original",
        None,
        None,
    );
}

#[test]
fn bailout_sweep_keeps_mapped_arguments_eval_captures_and_tdz() {
    sweep(
        r#"
        function subject(value) {
            var read=()=>value;
            arguments[0]=5;
            {let saved=value+1;published=()=>saved;eval('value=7');}
            try {trace.push(before);let before=1;}
            catch(error){trace.push(error.name);}
            return [value,read(),published(),arguments[0]].join(':');
        }
        var trace=[],published;
        "#,
        &[1.0],
        "trace.join(',')",
        "return:7:7:6:7|ReferenceError",
        None,
        None,
    );
}

#[test]
fn bailout_sweep_does_not_repeat_parameter_initialization_or_tail_call() {
    sweep(
        r#"
        function subject(value=(trace.push('default'),2)) {
            trace.push('body');return value+1;
        }
        var trace=[];
        "#,
        &[],
        "trace.join(',')",
        "return:3|default,body",
        None,
        None,
    );
    let kinds = sweep(
        r#"
        function subject(value) {
            'use strict';trace.push('subject');return target(value+1);
        }
        function target(value) {'use strict';trace.push('target');return value*2;}
        var trace=[];
        "#,
        &[2.0],
        "trace.join(',')",
        "return:6|subject,target",
        None,
        None,
    );
    assert!(
        kinds.contains(&discriminant(&Op::Return)),
        "must exit after staging the tail call"
    );
}

#[test]
fn bailout_sweep_preserves_pending_return_throw_and_labelled_completion() {
    let source = r#"
        function subject(mode) {
            outer:for(var k=0;k<4;k++) {
                try {
                    try {
                        trace.push(k);
                        if(mode===0)return 7;
                        if(k===1)continue outer;
                        if(k===2)break outer;
                    } finally {trace.push('i'+k);}
                } finally {
                    trace.push('o'+k);
                    if(mode===2)throw 'override';
                }
            }
            return 9;
        }
        var trace=[];
    "#;
    for (mode, expected) in [
        (0.0, "return:7|0,i0,o0"),
        (1.0, "return:9|0,i0,o0,1,i1,o1,2,i2,o2"),
        (2.0, "throw:override|0,i0,o0"),
    ] {
        sweep(source, &[mode], "trace.join(',')", expected, None, None);
    }
}

thread_local! {
    static OWNER: std::cell::RefCell<Option<std::rc::Weak<std::cell::RefCell<crate::value::Object>>>> = const { std::cell::RefCell::new(None) };
}

fn make_owner(i: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    let object = i.new_object();
    object
        .borrow_mut()
        .props
        .insert("value", crate::value::Property::plain(Value::Num(7.0)));
    object.borrow_mut().props.insert(
        "self",
        crate::value::Property::plain(Value::Obj(object.clone())),
    );
    OWNER.with(|owner| *owner.borrow_mut() = Some(Rc::downgrade(&object)));
    Ok(Value::Obj(object))
}

fn probe_owner(i: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    i.gc_collect();
    OWNER.with(|owner| assert!(owner.borrow().as_ref().unwrap().upgrade().is_some()));
    Ok(Value::Undefined)
}

#[test]
fn bailout_sweep_roots_cyclic_owners_and_releases_them_on_exit() {
    for (mode, expected) in [(0.0, "return:9|complete"), (1.0, "throw:abrupt|complete")] {
        sweep(
            r#"
        function subject(mode) {
            var a=makeOwner(), token=Symbol('s'), big=123456789012345678901234567890n;
            var pair=[a,(probeOwner(),2),token,big,'dynamic:'+big];
            var value=(probeOwner(),a.value+pair[1]);
            if(pair[2]!==token || pair[3]!==big || pair[4]!=='dynamic:'+big)throw 'owner lost';
            if(mode)throw 'abrupt';
            a=null;pair=null;return value;
        }
        "#,
            &[mode],
            "'complete'",
            expected,
            Some(|engine| {
                let global = engine.interp.global.clone();
                engine
                    .interp
                    .def_method(&global, "makeOwner", 0, make_owner);
                engine
                    .interp
                    .def_method(&global, "probeOwner", 0, probe_owner);
            }),
            Some(|engine| {
                engine.interp.gc_collect();
                OWNER.with(|owner| {
                    assert!(
                        owner.borrow().as_ref().unwrap().upgrade().is_none(),
                        "neither native nor VM continuation may retain the dead cycle"
                    )
                });
            }),
        );
    }
}

fn function(engine: &mut Engine, name: &str) -> Rc<crate::ast::Function> {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var(name, &env).ok().unwrap();
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("fixture user function")
    };
    user.func.clone()
}

#[test]
fn bailout_sweep_keeps_native_field_element_and_string_key_transfers_exact() {
    let kinds = sweep(
        r#"
        function subject() {
            var row={value:root},a=[root],key='0',sum=0;
            for(var k=0;k<3;k++) {
                row.value=a[key];a[key]=row.value;
                var kept=((0,a)[key]=row.value);
                if(kept===root)sum++;
            }
            try {row.value=root;Object.defineProperty(a,'0',{set(v){trace.push(v===root);throw 'stop';}});a[key]=root;}
            catch(e){trace.push(e);}
            finally {trace.push(row.value===root);}
            return sum;
        }
        var root={},trace=[];
    "#,
        &[],
        "trace.join(',')",
        "return:3|true,stop,true",
        None,
        None,
    );
    for op in [
        Op::GetElemLocal(0),
        Op::SetElemLocalDrop(0),
        Op::SetElem,
        Op::SetPropLocalDrop(0, 0, 0),
    ] {
        assert!(
            kinds.contains(&discriminant(&op)),
            "real native-store bailout boundary: {op:?}"
        );
    }
}

#[test]
fn bailout_callee_cannot_consume_native_callers_handlers_or_sidecar() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        function callee(x) {
            try {var a=x+1;if(x===9)throw a;return a*2;}
            finally {trace.push('callee'+x);}
        }
        function caller() {
            var sum=0;
            for(var x=0;x<12;x++) {
                try {sum+=callee(x);}
                catch(e) {var read=()=>e;trace.push('catch'+read());sum+=read();}
                finally {trace.push('caller'+x);}
            }
            return sum;
        }
        var trace=[];
    "#,
    );
    let mut leases = Vec::new();
    for name in ["callee", "caller"] {
        let function = function(&mut engine, name);
        let chunk = bytecode::compile(&function).unwrap();
        let layout = crate::interpreter::interp_layout(&mut engine.interp);
        let values = crate::value::jit_layout(&engine.interp.object_proto);
        let code = if name == "callee" {
            assert!(!chunk.jit_needs_activation_state());
            let pc = chunk
                .jit_ops()
                .iter()
                .position(|op| matches!(op, Op::Add))
                .unwrap();
            compile_with_deopt(&chunk, &values, &layout, Some(pc)).unwrap()
        } else {
            // Explicitly select a real template caller, independent of the ambient tier mode.
            crate::jit::compile(&chunk, &values, &layout).expect("template caller")
        };
        let code = Rc::new(code);
        chunk
            .jit
            .begin_compile()
            .unwrap()
            .commit(Some(code.clone()));
        // This is an ABI test, not an admission/feedback test. Keep these exact code owners
        // resident and eligible for the shared-context sequence, without an inline upgrade.
        chunk.inline_attempted.set(true);
        chunk.inline_retry_at.set(0);
        assert!(function.code.set(Some(chunk)).is_ok());
        leases.push(code);
    }
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    bytecode::native_deopt::FOREIGN_SIDECAR_ENTRIES.with(|count| count.set(0));
    assert_eq!(eval(&mut engine, "caller();"), "146");
    let trace = eval(&mut engine, "trace.join(',');");
    let expected: Vec<_> = (0..12)
        .flat_map(|x| {
            let mut items = vec![format!("callee{x}")];
            if x == 9 {
                items.push("catch10".into());
            }
            items.push(format!("caller{x}"));
            items
        })
        .collect();
    assert_eq!(trace, expected.join(","));
    ENTRIES.with(|entries| {
        let entries = entries.borrow();
        assert_eq!(entries.len(), 12);
        // Every compiled call, direct or helper-mediated, runs the callee on a frame record
        // of its own: the callee's continuation sees none of its caller's handlers.
        assert!(
            entries.iter().all(|&(_, _, floor)| floor == 0),
            "a callee context must not contain its caller's handlers"
        );
    });
    assert_eq!(
        bytecode::native_deopt::FOREIGN_SIDECAR_ENTRIES.with(|count| count.get()),
        0,
        "a callee context must not carry its caller's catch-scope sidecar"
    );
    for code in leases {
        assert_eq!(code.residency.active.get(), 0);
    }
    assert_eq!(engine.interp.depth, 0);
    assert!(engine.interp.fn_frames.is_empty());
}

fn fresh_fields(interp: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    let object = interp.new_object();
    object.borrow_mut().props = crate::value::Props::with_layout(
        3,
        Some(crate::value::new_property_layout(vec![
            Rc::from("first"),
            Rc::from("second"),
            Rc::from("third"),
        ])),
    );
    Ok(Value::Obj(object))
}

#[test]
fn bailout_sweep_keeps_created_fields_shapes_and_alias_owners_before_rejection() {
    // Repeated fresh before-shapes warm the creation IC inside this one function.
    // The predicted layout owns keys but does not create the uninitialized third
    // property. Self-alias owners and the new shape must survive a later bailout.
    let kinds = sweep(
        r#"
        function subject() {
            'use strict';var row,result;
            for(var k=0;k<3;k++){
                row=freshFields();row.first=held;result=((0,row).second=row);
            }
            trace.push(Object.keys(row).join(','));
            try{Object.preventExtensions(row);row.third=1;}
            catch(e){trace.push(e.name);}
            return result===row&&row.first===held;
        }
        var held={},trace=[];
        "#,
        &[],
        "trace.join('|')",
        "return:true|first,second|TypeError",
        Some(|engine| {
            let global = engine.interp.global.clone();
            engine
                .interp
                .def_method(&global, "freshFields", 0, fresh_fields);
        }),
        Some(|engine| {
            engine.interp.gc_collect();
            assert!(engine.interp.fn_frames.is_empty());
        }),
    );
    for op in [Op::SetPropLocalDrop(0, 0, 0), Op::SetProp(0, 0)] {
        assert!(
            kinds.contains(&discriminant(&op)),
            "creation bailout actually reaches {op:?}"
        );
    }
}
