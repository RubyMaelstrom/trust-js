//! A live user callable retains its creation Realm after host global-Env replacement.
//! ECMA-262 e28783d5 GetFunctionRealm, PrepareForOrdinaryCall and #sec-liveness.

use crate::interpreter::Abrupt;
use crate::realm_inline_guard_tests::{eval_string, in_realm, set_global, TIERS};
use crate::value::Value;
use crate::Engine;

fn prototype_identity(value: &Value) -> usize {
    let object = value.as_obj().expect("fixture object");
    let object = object.borrow();
    crate::value::Gc::as_ptr(object.proto.as_ref().expect("intrinsic prototype")) as usize
}

fn thrown(result: Result<Value, Abrupt>) -> Value {
    match result {
        Err(Abrupt::Throw(error)) => error,
        _ => panic!("fixture must throw its creation Realm's TypeError"),
    }
}

#[test]
fn old_global_environment_keeps_detached_callables_creation_realm_live() {
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let realm = engine.interp.create_realm();
        let global = realm.as_obj().expect("Realm global");
        let realm_key = crate::value::Gc::as_ptr(global) as usize;
        let weak_global = crate::value::Gc::downgrade(global);
        let (closure, constructor, old_environment, array_proto, object_proto, error_proto) =
            in_realm(&mut engine.interp, &realm, |i| {
                i.switch_host_job_context(41);
                eval_string(
                    i,
                    r#"
                    var oldClosure = mode => {
                        if (mode) throw new TypeError('birth closure');
                        return [];
                    };
                    function oldConstructor(mode) {
                        'use strict';
                        if (mode) throw new TypeError('birth constructor');
                        this.marker = 7;
                    }
                    Object.setPrototypeOf(oldClosure, null);
                    Object.setPrototypeOf(oldConstructor, null);
                    oldConstructor.prototype = null;
                    'ready'
                    "#,
                );
                let old = i.global_env.clone();
                {
                    let record = old.borrow();
                    let binding = record.vars.get("this").expect("global this binding");
                    assert!(!binding.mutable && !binding.deletable);
                    assert!(crate::value::Gc::ptr_eq(
                        binding.value.as_obj().expect("original global"),
                        &i.global
                    ));
                }
                let closure = i.get_var("oldClosure", &old).ok().expect("closure");
                let constructor = i.get_var("oldConstructor", &old).ok().expect("constructor");
                assert!(closure.as_obj().unwrap().borrow().proto.is_none());
                assert!(constructor.as_obj().unwrap().borrow().proto.is_none());
                // Exclude an accidentally retained intrinsic as an alternate Realm root.
                for binding in old.borrow().vars.values() {
                    if let Value::Obj(object) = &binding.value {
                        assert!(
                            crate::value::Gc::ptr_eq(object, &i.global)
                                || crate::value::Gc::ptr_eq(object, closure.as_obj().unwrap())
                                || crate::value::Gc::ptr_eq(object, constructor.as_obj().unwrap())
                        );
                    }
                }
                let array_proto = crate::value::Gc::downgrade(&i.array_proto);
                let object_proto = crate::value::Gc::downgrade(&i.object_proto);
                let error_proto =
                    crate::value::Gc::downgrade(i.error_protos.get("TypeError").unwrap());
                i.switch_host_job_context(42);
                assert!(!crate::interpreter::Env::ptr_eq(&old, &i.global_env));
                {
                    let record = i.global_env.borrow();
                    let binding = record.vars.get("this").expect("replacement this binding");
                    assert!(!binding.mutable && !binding.deletable);
                    assert!(crate::value::Gc::ptr_eq(
                        binding.value.as_obj().expect("same Realm global"),
                        &i.global
                    ));
                }
                (
                    closure,
                    constructor,
                    crate::interpreter::Env::downgrade(&old),
                    array_proto,
                    object_proto,
                    error_proto,
                )
            });
        assert!(engine.interp.release_host_job_context(41));
        assert!(engine.interp.release_host_job_context(42));
        set_global(&mut engine.interp, "heldOldClosure", closure);
        set_global(&mut engine.interp, "heldOldConstructor", constructor);
        drop(realm);

        // No external strong Realm, Env or intrinsic handle survives collection. Only
        // the live callables' old Env chains provide their original global this edge.
        for _ in 0..3 {
            engine.interp.gc_collect();
            assert!(engine.interp.realms.contains_key(&realm_key), "{tier:?}");
            assert!(weak_global.strong_count() != 0, "{tier:?}");
            assert!(old_environment.strong_count() != 0, "{tier:?}");
            let active = engine.interp.global_env.clone();
            let closure = engine
                .interp
                .get_var("heldOldClosure", &active)
                .ok()
                .expect("retained closure");
            let constructor = engine
                .interp
                .get_var("heldOldConstructor", &active)
                .ok()
                .expect("retained constructor");
            assert_eq!(
                engine.interp.callee_realm_global(closure.as_obj().unwrap()),
                Some(realm_key),
                "{tier:?}"
            );
            let array = engine
                .interp
                .call(closure.clone(), Value::Undefined, &[Value::Num(0.0)])
                .ok()
                .expect("old closure returns array");
            assert_eq!(prototype_identity(&array), array_proto.as_ptr() as usize);
            let instance = engine
                .interp
                .construct(constructor.clone(), &[Value::Num(0.0)])
                .ok()
                .expect("old constructor builds instance");
            assert_eq!(
                prototype_identity(&instance),
                object_proto.as_ptr() as usize
            );
            let closure_error = thrown(engine.interp.call(
                closure,
                Value::Undefined,
                &[Value::Num(1.0)],
            ));
            let constructor_error =
                thrown(engine.interp.construct(constructor, &[Value::Num(1.0)]));
            assert_eq!(
                prototype_identity(&closure_error),
                error_proto.as_ptr() as usize
            );
            assert_eq!(
                prototype_identity(&constructor_error),
                error_proto.as_ptr() as usize
            );
            assert!(engine.interp.fn_frames.is_empty(), "{tier:?}");
            assert_eq!(engine.interp.host_job_context, 0, "{tier:?}");
        }

        // The proof must not turn the registry into an immortal Realm root.
        match engine
            .eval(
                "heldOldClosure = null; heldOldConstructor = null; 'released'",
                false,
            )
            .expect("release parses")
        {
            crate::Completion::Value(value) => assert_eq!(value, "released"),
            crate::Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
        engine.interp.gc_collect();
        assert_eq!(weak_global.strong_count(), 0, "{tier:?}");
        assert_eq!(old_environment.strong_count(), 0, "{tier:?}");
        assert!(!engine.interp.realms.contains_key(&realm_key), "{tier:?}");
    }
}
