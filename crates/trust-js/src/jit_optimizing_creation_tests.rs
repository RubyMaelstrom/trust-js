//! Creation coverage and normative descriptor/prototype/ordering/owner contracts.
//! ECMA-262 e28783d5 OrdinarySetWithOwnDescriptor, CreateDataProperty,
//! OrdinaryOwnPropertyKeys and WeakRef liveness. Correct fallback results alone
//! do not satisfy the native-path assertions.

use super::*;
use crate::bytecode::Tier;
use crate::value::{Gc, Property, Props, Value as JsValue};
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("creation fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn object(engine: &mut Engine, name: &str) -> Gc {
    let env = engine.interp.global_env.clone();
    engine
        .interp
        .get_var(name, &env)
        .ok()
        .unwrap()
        .as_obj()
        .unwrap()
        .clone()
}

fn setup(source: &str) -> (Engine, Rc<JitCode>) {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    let code = call_tests::optimized(&mut engine, "subject", None);
    (engine, code)
}

/// Opcode-level coverage for arbitrary constant names. The current frontend leaves
/// bracketed String keys as SetElem: do NOT pretend that path has creation ICs.
/// Translate just this fixture's literal reference to the semantically identical
/// static Member before bytecode compilation, retaining the real function call.
fn static_name_fixture(name: &str) -> (Engine, Rc<JitCode>) {
    use crate::ast::{Expr, Stmt};
    use crate::value::Callable;
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        &format!(
            "function subject(){{target['{name}']=held;return true;}}var target={{}},held={{}};"
        ),
    );
    let owner = object(&mut engine, "subject");
    let function = match &owner.borrow().call {
        Callable::User(user) => user.func.clone(),
        _ => panic!("user function"),
    };
    let mut syntax = (*function).clone();
    let Stmt::Expr(Expr::Assign { target, .. }) = &mut syntax.body[0] else {
        panic!("assignment")
    };
    let Expr::Index {
        obj,
        index,
        optional: false,
        pos,
    } = &**target
    else {
        panic!("literal reference")
    };
    assert!(matches!(&**index, Expr::Str(key) if &**key == name));
    **target = Expr::Member {
        obj: obj.clone(),
        prop: name.to_owned(),
        optional: false,
        pos: *pos,
    };
    let chunk = bytecode::compile(&syntax).expect("static-name bytecode");
    assert!(chunk
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::SetPropDrop(..))));
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let interp = crate::interpreter::interp_layout(&mut engine.interp);
    let code = Rc::new(
        compile_with_deopt(&chunk, &values, &interp, None).expect("static-name native code"),
    );
    assert!(code.pc_offsets.is_empty());
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.code.set(Some(chunk)).is_ok());
    (engine, code)
}

fn disabled() -> bool {
    ["TRUST_JS_OPT_JIT_HEAP_OPS", "TRUST_JS_OPT_JIT_CREATION"]
        .iter()
        .any(|key| std::env::var(key).as_deref() == Ok("0"))
}

// Other engine tests can invalidate the process-global prototype epoch. Do not
// mistake such a real guard miss for absent native coverage: warm and measure
// within a stable epoch, without changing/locking/resetting the global protocol.
fn measured(
    engine: &mut Engine,
    code: &JitCode,
    invocation: &str,
    mut fresh: impl FnMut(&mut Engine),
) -> (String, [usize; 3]) {
    for _ in 0..256 {
        let epoch = crate::value::proto_epoch();
        fresh(engine);
        eval(engine, invocation);
        fresh(engine);
        elements::TEST_HEAP_HELPERS.with(|c| c.set([0; 3]));
        code.residency.referenced.set(0);
        let result = eval(engine, invocation);
        let helpers = elements::TEST_HEAP_HELPERS.with(|c| c.get());
        if crate::value::proto_epoch() == epoch {
            assert!(
                code.residency.referenced.get() > 0,
                "optimized subject actually entered"
            );
            assert_eq!(code.residency.active.get(), 0);
            assert!(engine.interp.fn_frames.is_empty());
            return (result, helpers);
        }
        std::thread::yield_now();
    }
    panic!("no stable creation-epoch trial in 256 attempts");
}

fn native(counts: [usize; 3], stores: usize) {
    assert_eq!(counts, [if disabled() { stores } else { 0 }, 0, 0]);
}

#[test]
fn native_creation_covers_all_receivers_results_and_value_owners() {
    for held in [
        "undefined",
        "null",
        "false",
        "-0",
        "NaN",
        "Infinity",
        "'held string'",
        "Symbol('held')",
        "123456789012345678901234n",
        "({})",
        "target",
    ] {
        let source = format!(
            r#"
            function subject() {{
                var row=target;
                row.alpha=held;
                (0,row).beta=held;
                var result=(row.gamma=held);
                this.delta=held;
                return Object.is(result,held);
            }}
            var target={{}},held={held};
        "#
        );
        let (mut engine, code) = setup(&source);
        let target = object(&mut engine, "target");
        let layout = crate::value::new_shared_indexed_property_layout(
            ["alpha", "beta", "gamma", "delta"]
                .map(Rc::<str>::from)
                .to_vec(),
        );
        let (result, counts) = measured(&mut engine, &code, "subject.call(target)", |_| {
            target.borrow_mut().props = Props::with_layout(4, Some(layout.clone()));
        });
        assert_eq!(result, "true", "{held}");
        native(counts, 4);
        assert!(Rc::ptr_eq(
            target.borrow().props.shared_layout().unwrap(),
            &layout
        ));
        assert_eq!(
            eval(&mut engine, "Object.keys(target).join(',')"),
            "alpha,beta,gamma,delta"
        );
        assert_eq!(eval(&mut engine, "Object.keys(target).every(k=>{var d=Object.getOwnPropertyDescriptor(target,k);return d.writable&&d.enumerable&&d.configurable&&Object.is(d.value,held)})"), "true");
        if held == "({})" {
            assert_eq!(
                Rc::strong_count(&object(&mut engine, "held")),
                6,
                "four field edges + binding + test pin"
            );
        } else if held == "target" {
            assert_eq!(
                Rc::strong_count(&target),
                7,
                "four self edges + two bindings + test pin"
            );
        }
        engine.interp.gc_collect();
        assert_eq!(eval(&mut engine, "Object.is(target.gamma,held)"), "true");
    }
}

#[test]
fn creation_predictions_compare_equal_unicode_empty_long_and_special_names() {
    for name in [
        "".to_owned(),
        "é漢😀".to_owned(),
        "length".to_owned(),
        "prototype".to_owned(),
        "abcdefgh".repeat(129),
        "abcdefgh".repeat(129) + "尾",
    ] {
        let (mut engine, code) = static_name_fixture(&name);
        let target = object(&mut engine, "target");
        let layout = crate::value::new_property_layout(vec![Rc::<str>::from(name.as_str())]);
        let (result, counts) = measured(&mut engine, &code, "subject()", |_| {
            target.borrow_mut().props = Props::with_layout(1, Some(layout.clone()));
        });
        assert_eq!(result, "true", "{name}");
        native(counts, 1);
        assert!(Rc::ptr_eq(
            target.borrow().props.shared_layout().unwrap(),
            &layout
        ));
        assert_eq!(eval(&mut engine, "Reflect.ownKeys(target).length"), "1");
        assert_eq!(
            eval(&mut engine, &format!("target['{name}']===held")),
            "true"
        );
    }
}

#[test]
fn creation_adopts_shared_layouts_but_destroys_private_last_owners_in_rust() {
    let (mut engine, code) = setup("function subject(){target.alpha=held;target.beta=held;return target.beta===held;}var target={},held={};");
    let target = object(&mut engine, "target");
    for shared in [false, true] {
        let previous =
            crate::value::new_property_layout(vec![Rc::from("wrong"), Rc::from("unused")]);
        let (result, counts) = measured(&mut engine, &code, "subject()", |_| {
            target.borrow_mut().props = if shared {
                Props::with_layout(3, Some(previous.clone()))
            } else {
                Props::with_capacity(3)
            };
        });
        assert_eq!(result, "true");
        native(counts, 2);
        assert_eq!(Rc::strong_count(&previous), 1);
    }
    let (result, counts) = measured(&mut engine, &code, "subject()", |_| {
        target.borrow_mut().props = Props::with_layout(
            3,
            Some(crate::value::new_property_layout(vec![Rc::from("private")])),
        );
    });
    assert_eq!(result, "true");
    assert_eq!(counts, [if disabled() { 2 } else { 1 }, 0, 0]);
    assert_eq!(
        eval(&mut engine, "Object.keys(target).join(',')"),
        "alpha,beta"
    );
}

#[test]
fn native_creation_probes_all_four_polymorphic_before_shapes() {
    let (mut engine, code) = setup("function subject(row){row.created=held;return row.created===held;}var held={},a={},b={},c={},d={};");
    let rows = ["a", "b", "c", "d"].map(|name| object(&mut engine, name));
    let layouts: Vec<_> = (0..4)
        .map(|length| {
            let mut keys: Vec<Rc<str>> = (0..length)
                .map(|n| Rc::from(format!("prefix{n}")))
                .collect();
            keys.push(Rc::from("created"));
            crate::value::new_shared_indexed_property_layout(keys)
        })
        .collect();
    let (result, counts) = measured(
        &mut engine,
        &code,
        "subject(a)&&subject(b)&&subject(c)&&subject(d)",
        |_| {
            for (i, row) in rows.iter().enumerate() {
                let mut props = Props::with_layout(4, Some(layouts[i].clone()));
                for n in 0..i {
                    props.insert(
                        layouts[i][n].clone(),
                        Property::plain(JsValue::Num(n as f64)),
                    );
                }
                row.borrow_mut().props = props;
            }
        },
    );
    assert_eq!(result, "true");
    native(counts, 4);
}

#[test]
fn native_creation_preserves_the_small_map_boundary_and_reserved_capacity() {
    let (mut engine, code) = setup("function subject(){target.created=held;return target.created===held;}var target={},held={};");
    let target = object(&mut engine, "target");
    for length in [0, 7, 8, 12] {
        let mut keys: Vec<Rc<str>> = (0..length)
            .map(|i| Rc::from(format!("prefix{i}")))
            .collect();
        keys.push(Rc::from("created"));
        let layout = crate::value::new_shared_indexed_property_layout(keys);
        for reserved in [false, true] {
            let (result, counts) = measured(&mut engine, &code, "subject()", |_| {
                let mut props =
                    Props::with_layout(length + usize::from(reserved), Some(layout.clone()));
                for n in 0..length {
                    props.insert(layout[n].clone(), Property::plain(JsValue::Num(n as f64)));
                }
                target.borrow_mut().props = props;
            });
            assert_eq!(result, "true");
            assert_eq!(
                counts,
                [
                    usize::from(disabled() || !reserved || length >= crate::value::INDEX_THRESHOLD),
                    0,
                    0
                ]
            );
            assert_eq!(
                eval(&mut engine, "Object.keys(target).length"),
                (length + 1).to_string()
            );
        }
    }
}

#[test]
fn warmed_creation_rechecks_prototype_descriptors_identity_and_extensibility() {
    let (mut engine, code) = setup("function subject(){'use strict';return target.field=held;}var proto={field:1},target=Object.create(proto),held={},trace=[];");
    let target = object(&mut engine, "target");
    let layout = crate::value::new_property_layout(vec![Rc::from("field")]);
    let fresh = || Props::with_layout(1, Some(layout.clone()));
    let (_, counts) = measured(&mut engine, &code, "subject()===held", |_| {
        target.borrow_mut().props = fresh()
    });
    native(counts, 1);
    for (change, expected, own) in [
        ("Object.defineProperty(proto,'field',{set(v){trace.push(v===held?'set':'bad')},configurable:true})", "true", false),
        ("Object.defineProperty(proto,'field',{value:2,writable:false,configurable:true})", "TypeError", false),
        ("Object.defineProperty(proto,'field',{value:2,writable:true,configurable:true})", "true", true),
        ("Object.setPrototypeOf(target,{set field(v){trace.push('new')}})", "true", false),
        ("Object.setPrototypeOf(target,null)", "true", true),
        ("Object.preventExtensions(target)", "TypeError", false),
    ] {
        target.borrow_mut().props = fresh();
        eval(&mut engine, change);
        assert_eq!(eval(&mut engine, "try{subject()===held}catch(e){e.name}"), expected, "{change}");
        assert_eq!(eval(&mut engine, "Object.hasOwn(target,'field')"), own.to_string(), "{change}");
        assert!(engine.interp.fn_frames.is_empty());
    }
    assert_eq!(eval(&mut engine, "trace.join(',')"), "set,new");
}

#[test]
fn creation_on_a_prototype_invalidates_descendant_proofs_and_preserves_order() {
    let (mut engine, code) = setup("function subject(){target.field=held;return target.field===held;}var target={},held={},child=Object.create(target);");
    let target = object(&mut engine, "target");
    let layout = crate::value::new_property_layout(vec![Rc::from("field")]);
    // First prove that this exact function can take its native path.
    let (result, counts) = measured(&mut engine, &code, "subject()", |_| {
        target.borrow_mut().props = Props::with_layout(2, Some(layout.clone()));
    });
    assert_eq!(result, "true");
    native(counts, 1);
    target.borrow_mut().props = Props::with_layout(2, Some(layout));
    target.borrow().props.mark_proto();
    let before = crate::value::proto_epoch();
    elements::TEST_HEAP_HELPERS.with(|c| c.set([0; 3]));
    assert_eq!(eval(&mut engine, "subject()"), "true");
    assert_eq!(elements::TEST_HEAP_HELPERS.with(|c| c.get()), [1, 0, 0]);
    assert_ne!(crate::value::proto_epoch(), before);
    assert_eq!(eval(&mut engine, "child.field===held"), "true");
    assert_eq!(
        eval(
            &mut engine,
            "target.other=2;delete target.field;subject();Object.keys(target).join(',')"
        ),
        "other,field"
    );
}

#[test]
fn creation_keeps_rhs_effects_and_reentrant_collection_before_setter_or_rejection() {
    let source = r#"
        function subject(mode) {
            'use strict';var row=target, saved=1;
            try {var result=(row.field=(saved=7,rhs(mode)));trace.push(result===held);}
            catch(e){trace.push(saved,e.name);}
            return saved;
        }
        var target={},held={},trace=[];
        function rhs(mode) {
            trace.push('rhs');
            if(mode===1)Object.defineProperty(Object.getPrototypeOf(target),'field',{set(v){trace.push(v===held?'set':'bad')},configurable:true});
            if(mode===2)Object.preventExtensions(target);
            collect();return held;
        }
    "#;
    let (mut engine, code) = setup(source);
    let global = engine.interp.global.clone();
    engine
        .interp
        .def_method(&global, "collect", 0, |interp, _, _| {
            interp.gc_collect();
            Ok(JsValue::Undefined)
        });
    let target = object(&mut engine, "target");
    let layout = crate::value::new_property_layout(vec![Rc::from("field")]);
    let (_, counts) = measured(&mut engine, &code, "subject(0)", |_| {
        target.borrow_mut().props = Props::with_layout(1, Some(layout.clone()));
    });
    native(counts, 1);
    eval(&mut engine, "trace=[]");
    target.borrow_mut().props = Props::with_layout(1, Some(layout.clone()));
    assert_eq!(
        eval(
            &mut engine,
            "subject(1)+'|'+Object.hasOwn(target,'field')+'|'+trace.join(',')"
        ),
        "7|false|rhs,set,true"
    );
    eval(&mut engine, "delete Object.prototype.field;trace=[]");
    target.borrow_mut().props = Props::with_layout(1, Some(layout));
    assert_eq!(
        eval(
            &mut engine,
            "subject(2)+'|'+Object.hasOwn(target,'field')+'|'+trace.join(',')"
        ),
        "7|false|rhs,7,TypeError"
    );
    assert!(engine.interp.fn_frames.is_empty());
}

#[test]
fn warmed_creation_does_not_bypass_proxy_traps_or_primitive_rejection() {
    let (mut engine, code) = setup("function subject(){'use strict';target.field=held;return true;}var target={},held={},trace=[];");
    let target = object(&mut engine, "target");
    let layout = crate::value::new_property_layout(vec![Rc::from("field")]);
    let (_, counts) = measured(&mut engine, &code, "subject()", |_| {
        target.borrow_mut().props = Props::with_layout(1, Some(layout.clone()));
    });
    native(counts, 1);
    target.borrow_mut().props = Props::with_layout(1, Some(layout));
    eval(&mut engine, "var original=target, revocable=Proxy.revocable(original,{set(t,k,v,r){trace.push('set:'+k);return Reflect.set(t,k,v,r)},defineProperty(t,k,d){trace.push('define:'+k);return Reflect.defineProperty(t,k,d)}});target=revocable.proxy");
    assert_eq!(
        eval(
            &mut engine,
            "subject()+'|'+trace.join(',')+'|'+(original.field===held)"
        ),
        "true|set:field,define:field|true"
    );
    assert_eq!(
        eval(
            &mut engine,
            "revocable.revoke();try{subject()}catch(e){e.name}"
        ),
        "TypeError"
    );
    for value in ["null", "undefined", "3", "'boxed'"] {
        assert_eq!(
            eval(
                &mut engine,
                &format!("target={value};try{{subject()}}catch(e){{e.name}}")
            ),
            "TypeError"
        );
    }
    assert!(engine.interp.fn_frames.is_empty());
}

#[test]
fn creation_layout_abi_guards_and_long_key_lowering_are_bounded() {
    let engine = Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    assert!(creation::supported(&layout, "field"));
    assert!(!creation::supported(&layout, "0"));
    assert!(!creation::supported(&layout, "\0symbol"));
    let mut changed = layout;
    changed.key_probe_ok = false;
    assert!(!creation::supported(&changed, "field"));
    let mut changed = layout;
    changed.heap_layouts = i32::MAX as usize + 1;
    assert!(!creation::supported(&changed, "field"));
    let mut changed = layout;
    changed.vec_cap_off = i32::MAX as usize + 1;
    assert!(!creation::supported(&changed, "field"));
    let (_, short) = static_name_fixture(&"x".repeat(64));
    let (_, long) = static_name_fixture(&"x".repeat(16384));
    assert!(
        long.len <= short.len + 1024,
        "property-name length must not unroll proportional IR/machine code: {} versus {}",
        long.len,
        short.len
    );
}
