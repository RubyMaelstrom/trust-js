//! Own-data shortcuts for captured Object Environment Records. ECMA-262 snapshot
//! e28783d5fc9d: HasBinding, GetBindingValue, SetMutableBinding, OrdinaryHasProperty/Get/Set.
use super::*;
use crate::{bytecode::Tier, Completion, Engine};

fn run(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("own Reference fixture parses")
    {
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

fn reset_paths() {
    TEST_REFERENCE_OWN_PATHS.with(|counts| counts.set([0; 5]));
}

fn paths() -> [usize; 5] {
    TEST_REFERENCE_OWN_PATHS.with(|counts| counts.get())
}

fn collect(interp: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    interp.gc_collect();
    Ok(Value::Undefined)
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for threshold in [8, 0] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(threshold);
            engine
                .interp
                .def_method(&engine.interp.global, "collectReferenceOwn", 0, collect);
            assert_eq!(
                run(&mut engine, source),
                expected,
                "{tier:?}, threshold {threshold}"
            );
        }
    }
}

#[test]
fn prepared_reference_own_cold_global_updates_take_actual_hint_paths_in_every_tier() {
    // Below the unchanged128-backedge admission threshold: every tier exercises the AST
    // path behind the public API regression, not an already-optimized VM/native substitute.
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        reset_paths();
        assert_eq!(
            run(
                &mut engine,
                "var total=0;for(var index=0;index<100;index++)total+=index;total;"
            ),
            "4950"
        );
        let [captures, refreshed, reads, writes, _] = paths();
        assert!(
            captures >= 200 && refreshed >= 200,
            "{tier:?}: {:?}",
            paths()
        );
        assert!(reads >= 200 && writes >= 200, "{tier:?}: {:?}", paths());
    }
}

#[test]
fn prepared_reference_own_refresh_keeps_saved_owner_and_live_descriptor_strictness() {
    let mut engine = Engine::new();
    engine.interp.activate_gc_heap();
    let owner = Object::new(None);
    owner
        .borrow_mut()
        .props
        .insert("first", Property::plain(Value::Num(2.0)));
    owner
        .borrow_mut()
        .props
        .insert("value", Property::plain(Value::Num(3.0)));
    let mut reference = PreparedReference::object(Rc::from("value"), owner.clone(), false);
    reset_paths();
    assert!(matches!(
        engine.interp.read_prepared_reference(&mut reference),
        Ok(Value::Num(3.0))
    ));
    let prior_shape = reference.shape;
    owner.borrow_mut().props.remove("first");
    owner
        .borrow_mut()
        .props
        .insert("later", Property::plain(Value::Num(4.0)));
    owner
        .borrow_mut()
        .props
        .get_mut("value")
        .unwrap()
        .set_value(Value::Num(7.0));
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("value", Property::plain(Value::Num(99.0)));
    assert_ne!(owner.borrow().props.shape(), prior_shape);
    assert!(matches!(
        engine.interp.read_prepared_reference(&mut reference),
        Ok(Value::Num(7.0))
    ));
    engine.interp.strict = true; // must not override the captured sloppy Reference.
    owner
        .borrow_mut()
        .props
        .get_mut("value")
        .unwrap()
        .set_writable(false);
    assert!(engine
        .interp
        .write_prepared_reference(&mut reference, Value::Num(11.0))
        .is_ok());
    assert!(matches!(
        owner.borrow().props.get("value").unwrap().value(),
        Value::Num(7.0)
    ));
    reference.strict = 1;
    engine.interp.strict = false;
    assert!(engine
        .interp
        .write_prepared_reference(&mut reference, Value::Num(13.0))
        .is_err());
    assert!(
        !engine.interp.strict,
        "checked setter restored the ambient strictness"
    );
    assert!(
        paths()[1] >= 2 && paths()[2] == 2 && paths()[3] == 0 && paths()[4] == 2,
        "{:?}",
        paths()
    );
}

#[test]
fn prepared_reference_own_with_unscopables_and_rhs_descriptor_effects_stay_ordered() {
    check(
        r#"
        var trace=[], box={x:1};
        Object.defineProperty(box,Symbol.unscopables,{get(){
            trace.push('unscopables');
            Object.defineProperty(box,'x',{configurable:true,
                get(){trace.push('get');return 3},
                set(v){trace.push('set:'+v)}});
            return null;
        }});
        with(box){x += (trace.push('rhs'),2)}
        trace.join('|');
    "#,
        "unscopables|get|rhs|set:5",
    );
    check(
        r#"
        var trace=[], box={x:2}, result;
        with(box){result=(x += (Object.defineProperty(box,'x',{configurable:true,
            set(v){trace.push('set:'+v)}}),collectReferenceOwn(),3))}
        [result,trace.join(',')].join('|');
    "#,
        "5|set:5",
    );
    check(
        r#"
        var trace=[], box={};
        Object.defineProperty(box,'x',{get(){trace.push('bad-get');return 2},
            set(v){trace.push('set:'+v)},configurable:true});
        with(box){x=(trace.push('rhs'),4)}
        trace.join('|');
    "#,
        "rhs|set:4",
    );
}

#[test]
fn prepared_reference_own_proxy_has_get_set_and_abrupt_coercions_remain_checked() {
    check(
        r#"
        var trace=[], target={x:2}, proxy=new Proxy(target,{
            has(t,k){if(k==='x')trace.push('has');return Reflect.has(t,k)},
            get(t,k,r){if(k===Symbol.unscopables)trace.push('unscopables');
                if(k==='x')trace.push('get');return Reflect.get(t,k,r)},
            set(t,k,v,r){if(k==='x')trace.push('set');return Reflect.set(t,k,v,r)}
        });
        with(proxy){x += (trace.push('rhs'),3)}
        [target.x,trace.join(',')].join('|');
    "#,
        "5|has,unscopables,has,get,rhs,has,set",
    );
    check(
        r#"
        var box={x:1}, trace=[];
        try { with(box){x += {valueOf(){trace.push('coerce');delete box.x;
            collectReferenceOwn();throw 17}}} } catch(e){trace.push(e)}
        [trace.join(','),Object.hasOwn(box,'x')].join('|');
    "#,
        "coerce,17|false",
    );
    check(
        r#"
        var trace=[], target={x:1}, revoke;
        var pair=Proxy.revocable(target,{});revoke=pair.revoke;
        try {with(pair.proxy){x += (revoke(),2)}}catch(e){trace.push(e.name)}
        [target.x,trace.join(',')].join('|');
    "#,
        "1|TypeError",
    );
}

#[test]
fn prepared_reference_own_numeric_internal_names_never_bypass_mapped_arguments() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Interp);
    run(
        &mut engine,
        "var observe;var owner=(function(a){observe=()=>a;return arguments})(3);",
    );
    let owner = object(&mut engine, "owner");
    let pointer = Rc::as_ptr(&owner) as usize;
    assert!(engine.interp.mapped_arguments.contains_key(&pointer));
    let mut reference = PreparedReference::object(Rc::from("0"), owner.clone(), false);
    // Simulate a previously published ordinary shape hint: owner kind must still be checked.
    reference.shape = owner.borrow().props.shape();
    reference.slot = owner.borrow().props.slot_of("0").unwrap_or(usize::MAX);
    reset_paths();
    assert!(matches!(
        engine.interp.read_prepared_reference(&mut reference),
        Ok(Value::Num(3.0))
    ));
    assert_eq!(reference.shape, 0);
    assert!(engine
        .interp
        .write_prepared_reference(&mut reference, Value::Num(7.0))
        .is_ok());
    assert!(matches!(
        engine.interp.mapped_arg_value(pointer, "0"),
        Some(Value::Num(7.0))
    ));
    assert_eq!(run(&mut engine, "observe()"), "7");
    assert_eq!(paths()[2], 0);
    assert_eq!(paths()[3], 0);
    assert_eq!(paths()[4], 2);
}

#[test]
fn prepared_reference_own_exotic_host_and_exhausted_shapes_withdraw_hints() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Interp);
    run(
        &mut engine,
        "var array=[1], string=new String('x'), typed=new Uint8Array([2]);",
    );
    for name in ["array", "string", "typed"] {
        let owner = object(&mut engine, name);
        let mut reference = PreparedReference::object(Rc::from("length"), owner, false);
        assert!(
            !engine
                .interp
                .refresh_reference_own_data_hint(&mut reference),
            "{name}"
        );
        assert_eq!(reference.shape, 0);
    }
    engine.interp.activate_gc_heap();
    let owner = Object::new(None);
    owner
        .borrow_mut()
        .props
        .insert("value", Property::plain(Value::Num(9.0)));
    let mut reference = PreparedReference::object(Rc::from("value"), owner.clone(), false);
    assert!(engine
        .interp
        .refresh_reference_own_data_hint(&mut reference));
    let pointer = Rc::as_ptr(&owner) as usize;
    // Registration marks the object as side-table exotic, as install_indexed_properties does.
    owner.borrow().ic_plain.set(false);
    engine.interp.host_indexed.insert(
        pointer,
        crate::interpreter::HostIndexedProperties {
            length: 0,
            getter: Value::Undefined,
            live: None,
        },
    );
    assert!(!engine
        .interp
        .refresh_reference_own_data_hint(&mut reference));
    engine.interp.host_indexed.remove(&pointer);
    assert!(engine
        .interp
        .refresh_reference_own_data_hint(&mut reference));
    owner.borrow_mut().props.force_uncacheable_shape_for_test();
    reset_paths();
    assert!(matches!(
        engine.interp.read_prepared_reference(&mut reference),
        Ok(Value::Num(9.0))
    ));
    assert!(engine
        .interp
        .write_prepared_reference(&mut reference, Value::Num(11.0))
        .is_ok());
    assert_eq!(reference.shape, 0);
    assert_eq!(paths()[2], 0);
    assert_eq!(paths()[3], 0);
    assert_eq!(paths()[4], 2);
}

#[test]
fn prepared_reference_own_values_and_last_owner_survive_gc_without_reboxing_changes() {
    let mut engine = Engine::new();
    engine.interp.activate_gc_heap();
    let values = [
        Value::Undefined,
        Value::Null,
        Value::Bool(false),
        Value::Bool(true),
        Value::Num(0.0),
        Value::Num(-0.0),
        Value::Num(f64::NAN),
        Value::Num(f64::INFINITY),
        Value::Num(f64::NEG_INFINITY),
        Value::lstr("owned reference λ"),
        Value::BigInt(crate::bigint::JsBigInt::from_i128(i128::MAX)),
        engine.interp.new_symbol(Some(Rc::from("reference-symbol"))),
        Value::Obj(Object::new(None)),
    ];
    for value in values {
        let owner = Object::new(None);
        owner
            .borrow_mut()
            .props
            .insert("value", Property::plain(value.clone()));
        let mut reference = PreparedReference::object(Rc::from("value"), owner, false);
        reset_paths();
        let actual = engine
            .interp
            .read_prepared_reference(&mut reference)
            .ok()
            .unwrap();
        assert!(crate::builtins::same_value_pub(&actual, &value));
        assert!(engine
            .interp
            .write_prepared_reference(&mut reference, value.clone())
            .is_ok());
        engine.interp.gc_collect();
        assert!(crate::builtins::same_value_pub(
            &engine
                .interp
                .read_prepared_reference(&mut reference)
                .ok()
                .unwrap(),
            &value
        ));
        assert_eq!(paths()[2], 2);
        assert_eq!(paths()[3], 1);
    }
    let child = Object::new(None);
    child
        .borrow_mut()
        .props
        .insert("value", Property::plain(Value::Num(29.0)));
    let weak_child = Rc::downgrade(&child);
    let owner = Object::new(None);
    owner
        .borrow_mut()
        .props
        .insert("value", Property::plain(Value::Obj(child)));
    let weak_owner = Rc::downgrade(&owner);
    let mut reference = PreparedReference::object(Rc::from("value"), owner, false);
    let returned = engine
        .interp
        .read_prepared_reference(&mut reference)
        .ok()
        .unwrap();
    engine.interp.gc_collect();
    assert!(
        weak_owner.upgrade().is_some(),
        "saved base is a strong external owner"
    );
    drop(reference);
    engine.interp.gc_collect();
    assert!(weak_owner.upgrade().is_none());
    assert!(
        weak_child.upgrade().is_some(),
        "returned Value owns the last child"
    );
    drop(returned);
    engine.interp.gc_collect();
    assert!(weak_child.upgrade().is_none());
}

#[test]
fn prepared_reference_own_saved_reference_survives_suspension_and_shape_changes() {
    check(
        r#"
        var box={x:2}, trace=[], saved=box;
        function* body(){with(box){return x += yield 7}}
        var iterator=body(), first=iterator.next();
        Object.defineProperty(saved,'x',{configurable:true,
            set(v){trace.push('set:'+v)}});
        box=null;collectReferenceOwn();
        var next=iterator.next(3);
        [first.value,first.done,next.value,next.done,trace.join(',')].join('|');
    "#,
        "7|false|5|true|set:5",
    );
    check(
        r#"
        var box={x:4}, trace=[], outer=99;
        with(box){x += {valueOf(){delete box.x;
            Object.setPrototypeOf(box,{set x(v){trace.push('inherited:'+v)}});
            collectReferenceOwn();return 6}}}
        [Object.hasOwn(box,'x'),trace.join(',')].join('|');
    "#,
        "false|inherited:10",
    );
}
