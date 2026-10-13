//! OrdinaryGet/GetOwnProperty and Proxy [[Get]], ECMA-262 snapshot e28783d5fc9d.
//! These call get_from_chain directly so property ICs cannot bypass the tested path.
use super::*;
use crate::bytecode::Tier;
use crate::feedback::{CurrentPropertyTrace, PropertyOutcome};
use crate::{Completion, Engine};

fn run(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("read fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn value(engine: &mut Engine, name: &str) -> Value {
    let env = engine.interp.global_env.clone();
    engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("fixture value")
}

fn object(engine: &mut Engine, name: &str) -> Gc {
    value(engine, name)
        .as_obj()
        .expect("fixture object")
        .clone()
}

fn read(
    interp: &mut Interp,
    start: &Gc,
    key: &str,
    receiver: &Value,
    traced: bool,
) -> (Value, CurrentPropertyTrace) {
    let mut trace = CurrentPropertyTrace::default();
    let value = interp
        .get_from_chain(start, key, receiver, traced.then_some(&mut trace))
        .ok()
        .expect("ordinary read succeeds");
    (value, trace)
}

fn collect(interp: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    interp.gc_collect();
    Ok(Value::Undefined)
}

#[test]
fn ordinary_data_read_all_value_kinds_owned_and_traced() {
    let mut engine = Engine::new();
    engine.interp.activate_gc_heap();
    let values = [
        Value::Undefined,
        Value::Null,
        Value::Bool(false),
        Value::Bool(true),
        Value::Num(0.0),
        Value::Num(-0.0),
        Value::Num(19.25),
        Value::Num(-19.25),
        Value::Num(f64::NAN),
        Value::Num(f64::INFINITY),
        Value::Num(f64::NEG_INFINITY),
        Value::lstr("owned λ string"),
        Value::BigInt(crate::bigint::JsBigInt::from_i128(i128::MAX)),
        engine.interp.new_symbol(Some(Rc::from("read-symbol"))),
        Value::Obj(Object::new(None)),
    ];
    for expected in values {
        for key in ["data", "0"] {
            let holder = Object::new(None);
            holder
                .borrow_mut()
                .props
                .insert(key, Property::plain(expected.clone()));
            let shape = holder.borrow().props.shape();
            let slot = holder
                .borrow()
                .props
                .get_with_slot(key)
                .unwrap()
                .1
                .map(|n| n as u32);
            for inherited in [false, true] {
                let receiver = if inherited {
                    Object::new(Some(holder.clone()))
                } else {
                    holder.clone()
                };
                for traced in [false, true] {
                    let (actual, trace) = read(
                        &mut engine.interp,
                        &receiver,
                        key,
                        &Value::Obj(receiver.clone()),
                        traced,
                    );
                    assert!(
                        crate::builtins::same_value_pub(&actual, &expected),
                        "{key}, inherited={inherited}, traced={traced}"
                    );
                    if traced {
                        assert_eq!(trace.outcome, Some(PropertyOutcome::Data));
                        assert_eq!(trace.holder_shape, Some(shape));
                        assert_eq!(trace.depth, u8::from(inherited));
                        assert_eq!(trace.field_slot, slot);
                    } else {
                        assert_eq!(trace, CurrentPropertyTrace::default());
                    }
                }
            }
        }
    }
}

#[test]
fn ordinary_data_read_returned_last_owner_survives_removal_and_gc() {
    let mut engine = Engine::new();
    engine.interp.activate_gc_heap();
    for traced in [false, true] {
        let holder = Object::new(None);
        let payload = Object::new(None);
        payload
            .borrow_mut()
            .props
            .insert("alive", Property::plain(Value::Num(37.0)));
        let weak = Rc::downgrade(&payload);
        holder
            .borrow_mut()
            .props
            .insert("data", Property::plain(Value::Obj(payload)));
        let (result, _) = read(
            &mut engine.interp,
            &holder,
            "data",
            &Value::Obj(holder.clone()),
            traced,
        );
        assert_eq!(
            weak.strong_count(),
            2,
            "one property owner and one returned owner"
        );
        assert!(holder.borrow_mut().props.remove("data"));
        drop(holder);
        engine.interp.gc_collect();
        assert_eq!(
            weak.strong_count(),
            1,
            "only the returned Value roots its payload"
        );
        let marker = result
            .as_obj()
            .unwrap()
            .borrow()
            .props
            .get("alive")
            .unwrap()
            .value();
        assert!(matches!(marker, Value::Num(37.0)));
        drop(result);
        engine.interp.gc_collect();
        assert!(weak.upgrade().is_none());

        let symbol = engine.interp.new_symbol(Some(Rc::from("sole symbol")));
        let Value::Sym(symbol_ref) = &symbol else {
            unreachable!()
        };
        let weak_symbol = Rc::downgrade(symbol_ref);
        let id = symbol_ref.id;
        let holder = Object::new(None);
        holder
            .borrow_mut()
            .props
            .insert("data", Property::plain(symbol));
        let (result, _) = read(
            &mut engine.interp,
            &holder,
            "data",
            &Value::Obj(holder.clone()),
            traced,
        );
        assert!(holder.borrow_mut().props.remove("data"));
        drop(holder);
        engine.interp.gc_collect();
        assert!(matches!(&result, Value::Sym(s) if s.id == id));
        assert_eq!(weak_symbol.strong_count(), 1);
        drop(result);
        assert!(weak_symbol.upgrade().is_none());

        for payload in [
            Value::lstr("last string λ"),
            Value::bigint_from_i64(-123456789),
        ] {
            let holder = Object::new(None);
            holder
                .borrow_mut()
                .props
                .insert("data", Property::plain(payload));
            let (result, _) = read(
                &mut engine.interp,
                &holder,
                "data",
                &Value::Obj(holder.clone()),
                traced,
            );
            assert!(holder.borrow_mut().props.remove("data"));
            drop(holder);
            engine.interp.gc_collect();
            match result {
                Value::Str(text) => assert_eq!(text.as_str(), "last string λ"),
                Value::BigInt(n) => assert_eq!(n, crate::bigint::JsBigInt::from(-123456789i64)),
                _ => panic!("lost heap-valued data"),
            }
        }
    }
}

#[test]
fn ordinary_data_read_getter_reentry_mutation_gc_and_throw_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for traced in [false, true] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            engine
                .interp
                .def_method(&engine.interp.global, "collectReadTest", 0, collect);
            run(
                &mut engine,
                r#"
                var calls=[], token={answer:91}, holder={};
                var receiver=Object.create(holder); receiver.marker='receiver';
                Object.defineProperty(holder,'data',{configurable:true,get:function(){
                    calls.push(this.marker); delete holder.data;
                    Object.setPrototypeOf(holder,{data:33}); collectReadTest();
                    calls.push(this.data); return token;
                }});
            "#,
            );
            let receiver = object(&mut engine, "receiver");
            let holder = object(&mut engine, "holder");
            let before = holder.borrow().props.shape();
            let token = value(&mut engine, "token");
            let (result, trace) = read(
                &mut engine.interp,
                &receiver,
                "data",
                &Value::Obj(receiver.clone()),
                traced,
            );
            assert!(crate::builtins::same_value_pub(&result, &token));
            assert_eq!(
                run(&mut engine, "calls.join('|')"),
                "receiver|33",
                "{tier:?}"
            );
            if traced {
                assert_eq!(trace.outcome, Some(PropertyOutcome::Accessor));
                assert_eq!(trace.holder_shape, Some(before));
                assert_eq!(trace.depth, 1);
                assert_eq!(trace.field_slot, None);
            }
            run(&mut engine, "Object.defineProperty(holder,'bad',{get:function(){collectReadTest();throw token;}});Object.defineProperty(holder,'missingGetter',{get:undefined});");
            let mut thrown_trace = CurrentPropertyTrace::default();
            let result = engine.interp.get_from_chain(
                &receiver,
                "bad",
                &Value::Obj(receiver.clone()),
                traced.then_some(&mut thrown_trace),
            );
            assert!(
                matches!(result, Err(Abrupt::Throw(ref thrown)) if crate::builtins::same_value_pub(thrown,&token)),
                "{tier:?}"
            );
            if traced {
                assert_eq!(thrown_trace.outcome, Some(PropertyOutcome::Accessor));
            }
            let (missing, trace) = read(
                &mut engine.interp,
                &receiver,
                "missingGetter",
                &Value::Obj(receiver.clone()),
                traced,
            );
            assert!(matches!(missing, Value::Undefined));
            if traced {
                assert_eq!(trace.outcome, Some(PropertyOutcome::Accessor));
            }
        }
    }
}

#[test]
fn ordinary_data_read_dense_holes_and_present_undefined_are_distinct() {
    let mut engine = Engine::new();
    run(&mut engine, "var root=[];Object.setPrototypeOf(root,{0:44,1:55});root.length=3;root[1]=undefined;Object.defineProperty(root,'2',{get:function(){return this[0]+1;},configurable:true});");
    let root = object(&mut engine, "root");
    for traced in [false, true] {
        for (key, expected, outcome, depth) in [
            ("0", Value::Num(44.0), PropertyOutcome::Data, 1),
            ("1", Value::Undefined, PropertyOutcome::Data, 0),
            ("2", Value::Num(45.0), PropertyOutcome::Accessor, 0),
            ("absent", Value::Undefined, PropertyOutcome::Absent, 0),
        ] {
            let (result, trace) = read(
                &mut engine.interp,
                &root,
                key,
                &Value::Obj(root.clone()),
                traced,
            );
            assert!(crate::builtins::same_value_pub(&result, &expected), "{key}");
            if traced {
                assert_eq!(trace.outcome, Some(outcome));
                if outcome != PropertyOutcome::Absent {
                    assert_eq!(trace.depth, depth);
                }
                assert_eq!(trace.field_slot, None);
            }
        }
    }
}

#[test]
fn ordinary_data_read_lazy_prototype_identity_and_returned_ownership() {
    for traced in [false, true] {
        let mut engine = Engine::new();
        run(&mut engine, "var F=function(){};");
        let function = object(&mut engine, "F");
        let weak_function = Rc::downgrade(&function);
        let holder_shape = function.borrow().props.shape();
        let before = crate::value::heap_live_objects(&engine.interp.gc_heap);
        let (first, trace) = read(
            &mut engine.interp,
            &function,
            "prototype",
            &Value::Obj(function.clone()),
            traced,
        );
        assert_eq!(
            crate::value::heap_live_objects(&engine.interp.gc_heap),
            before + 1
        );
        let (second, _) = read(
            &mut engine.interp,
            &function,
            "prototype",
            &Value::Obj(function.clone()),
            traced,
        );
        assert!(crate::builtins::same_value_pub(&first, &second));
        assert_eq!(
            crate::value::heap_live_objects(&engine.interp.gc_heap),
            before + 1
        );
        if traced {
            assert_eq!(trace.outcome, Some(PropertyOutcome::Data));
            assert_eq!(trace.holder_shape, Some(holder_shape));
        }
        run(&mut engine, "F=null;");
        drop(function);
        drop(second);
        engine.interp.gc_collect();
        let constructor = first
            .as_obj()
            .unwrap()
            .borrow()
            .props
            .get("constructor")
            .unwrap()
            .value();
        assert!(Rc::ptr_eq(
            constructor.as_obj().unwrap(),
            &weak_function.upgrade().unwrap()
        ));
        drop(constructor);
        drop(first);
        engine.interp.gc_collect();
        assert!(weak_function.upgrade().is_none());
    }
}

#[test]
fn ordinary_data_read_proxy_dispatch_receivers_and_invariants_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        run(
            &mut engine,
            r#"
            var receiver={marker:73}, calls=[];
            var proxy=new Proxy({x:1},{get:function(t,k,r){calls.push(k);return r.marker;}});
            var forwarding=new Proxy({get x(){return this.marker;}},{});
            var frozenData=new Proxy(Object.freeze({x:1}),{get:function(){return 2;}});
            var frozenGetter=new Proxy(Object.defineProperty({},'x',{get:undefined}),{get:function(){return 2;}});
            var revokedPair=Proxy.revocable({},{});revokedPair.revoke();var revoked=revokedPair.proxy;
        "#,
        );
        let receiver = value(&mut engine, "receiver");
        for traced in [false, true] {
            for name in ["proxy", "forwarding"] {
                let start = object(&mut engine, name);
                let (result, trace) = read(&mut engine.interp, &start, "x", &receiver, traced);
                assert!(matches!(result, Value::Num(73.0)), "{tier:?}, {name}");
                if traced {
                    assert_eq!(trace.outcome, Some(PropertyOutcome::Exotic));
                }
            }
            for name in ["frozenData", "frozenGetter", "revoked"] {
                let start = object(&mut engine, name);
                let mut trace = CurrentPropertyTrace::default();
                let result = engine.interp.get_from_chain(
                    &start,
                    "x",
                    &receiver,
                    traced.then_some(&mut trace),
                );
                let Err(Abrupt::Throw(error)) = result else {
                    panic!("{tier:?}, {name}: expected TypeError");
                };
                let name = engine
                    .interp
                    .get_member(&error, "name")
                    .ok()
                    .expect("error name");
                assert!(matches!(name, Value::Str(s) if s.as_str()=="TypeError"));
                if traced {
                    assert_eq!(trace.outcome, Some(PropertyOutcome::Exotic));
                }
            }
        }
        assert_eq!(run(&mut engine, "calls.join('|')"), "x|x");
    }
}

#[test]
fn ordinary_data_read_host_indexed_dispatch_precedes_ordinary_lookup() {
    let mut engine = Engine::new();
    engine.interp.activate_gc_heap();
    let start = Object::new(None);
    let target = Value::Obj(start.clone());
    let getter = Value::Obj(
        engine
            .interp
            .make_native("indexedRead", 1, |interp, this, args| {
                assert!(this.as_obj().is_some());
                interp.gc_collect();
                Ok(Value::Num(match args.first() {
                    Some(Value::Num(n)) => n + 21.0,
                    _ => panic!("index"),
                }))
            }),
    );
    engine
        .interp
        .install_readonly_indexed_properties(&target, 2, getter)
        .ok()
        .expect("host indexed install");
    start
        .borrow_mut()
        .props
        .insert("named", Property::plain(Value::Num(39.0)));
    for traced in [false, true] {
        let (result, trace) = read(&mut engine.interp, &start, "1", &target, traced);
        assert!(matches!(result, Value::Num(22.0)));
        if traced {
            assert_eq!(trace.outcome, Some(PropertyOutcome::Exotic));
        }
        let (result, trace) = read(&mut engine.interp, &start, "named", &target, traced);
        assert!(matches!(result, Value::Num(39.0)));
        if traced {
            assert_eq!(trace.outcome, Some(PropertyOutcome::Data));
        }
    }
}
