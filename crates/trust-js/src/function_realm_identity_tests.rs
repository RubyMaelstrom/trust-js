//! ECMAScript functions keep their creation [[Realm]] independently of [[Environment]]
//! and mutable [[Prototype]]. ECMA-262 e28783d5 GetFunctionRealm and
//! PrepareForOrdinaryCall; host settings can replace the active global environment.

use crate::realm_inline_guard_tests::{set_global, TIERS};
use crate::value::Value;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("Realm fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

#[test]
fn function_realm_identity_survives_host_environment_and_prototype_changes() {
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.interp.switch_host_job_context(41);
        eval(
            &mut engine,
            r#"
                globalThis.OwnerCtor = function OwnerCtor() { return {owner: true}; };
                globalThis.ownerObjectPrototype = Object.prototype;
                globalThis.ownerArrayPrototype = Array.prototype;
            "#,
        );
        let foreign = engine.interp.create_realm();
        set_global(&mut engine.interp, "foreign", foreign);
        let previous_environment = engine.interp.global_env.clone();
        engine.interp.switch_host_job_context(42);
        assert!(!crate::interpreter::Env::ptr_eq(
            &previous_environment,
            &engine.interp.global_env
        ));
        drop(previous_environment);
        engine.interp.def_method(
            &engine.interp.global,
            "functionRealmForTest",
            1,
            |i, _, args| i.callable_realm_global(args.first().unwrap_or(&Value::Undefined)),
        );
        eval(
            &mut engine,
            r#"
                Object.setPrototypeOf(OwnerCtor, foreign.Function.prototype);
                OwnerCtor.prototype = null;
                globalThis.boundOwner = OwnerCtor.bind(null);
                globalThis.proxyOwner = new Proxy(boundOwner, {});
                Object.setPrototypeOf(boundOwner, null);
                $262.gc();
            "#,
        );
        let env = engine.interp.global_env.clone();
        let owner = engine
            .interp
            .get_var("OwnerCtor", &env)
            .ok()
            .expect("retained owner constructor");
        assert_eq!(
            engine.interp.callee_realm_global(owner.as_obj().unwrap()),
            None,
            "the changed environment/prototype cannot move [[Realm]]: {tier:?}"
        );
        assert_eq!(
            eval(
                &mut engine,
                r#"
                (function () {
                    function check(target) {
                        return functionRealmForTest(target) === globalThis &&
                            Object.getPrototypeOf(Reflect.construct(Object, [], target)) === ownerObjectPrototype &&
                            Object.getPrototypeOf(Reflect.construct(Array, [1, 2], target)) === ownerArrayPrototype;
                    }
                    for (let n=0; n<160; n++) {
                        if (!check(OwnerCtor) || !check(boundOwner) || !check(proxyOwner)) return 'wrong';
                    }
                    const revoked = Proxy.revocable(OwnerCtor, {});
                    revoked.revoke();
                    try { functionRealmForTest(revoked.proxy); return 'missing revocation'; }
                    catch (error) { if (!(error instanceof TypeError)) return 'wrong error Realm'; }
                    return 'ok';
                })()
            "#,
            ),
            "ok",
            "{tier:?}"
        );
        assert_eq!(engine.interp.host_job_context, 42);
        engine.interp.gc_collect();
    }
}
