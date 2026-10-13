//! Host constructors receive the same NewTarget/GetFunctionRealm semantics as
//! built-ins: ECMA-262 #sec-builtincallorconstruct / #sec-getfunctionrealm,
//! local snapshot e28783d5fc9dc12b3de905961e2c71410b38a202.

use crate::{bytecode::Tier, value::Value, Completion, Engine};

#[test]
fn native_constructor_context_survives_reentry_and_reports_function_realms() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let ctor = engine.interp.new_native_fn_with_captures(
            "HostConstructor",
            1,
            |ctx, _, args, _| {
                let target = ctx.constructor_new_target();
                if !ctx.is_constructing() {
                    return Ok(target);
                }
                if let Some(callback) = args.first() {
                    ctx.invoke(callback.clone(), Value::Undefined, &[])?;
                }
                assert!(ctx.values_strict_equal(&target, &ctx.constructor_new_target()));
                Ok(target)
            },
            vec![],
        );
        let prototype = Value::Obj(engine.interp.new_object());
        engine.interp.set_constructor_prototype(&ctor, &prototype);
        let global = engine.interp.global_this();
        engine
            .interp
            .member_set(&global, "HostConstructor", ctor)
            .unwrap_or_else(|_| panic!("install host constructor"));
        engine.define_global("functionRealm", 1, |ctx, _, args| {
            ctx.callable_realm_global(args.first().unwrap_or(&Value::Undefined))
        });
        let foreign = engine.interp.create_embed_realm();
        engine
            .interp
            .member_set(&global, "foreign", foreign)
            .unwrap_or_else(|_| panic!("install foreign Realm"));
        let source = r#"
            (() => {
                function check(v, m) { if (!v) throw Error(m); }
                function Target() {}
                check(HostConstructor() === undefined, 'ordinary call NewTarget');
                check(new HostConstructor() === HostConstructor, 'native new target');
                const callback = () => {
                    check(HostConstructor() === undefined, 'nested call clears NewTarget');
                    check(new HostConstructor() === HostConstructor, 'nested construction');
                };
                check(Reflect.construct(HostConstructor, [callback], Target) === Target, 'explicit NewTarget and restoration');
                let target = foreign.Function('');
                for (let i = 0; i < 192; i++)
                    target = i % 2 ? target.bind(null) : new Proxy(target, {});
                check(functionRealm(target) === foreign, 'unbounded bound/proxy realm traversal');
                const proxy = Proxy.revocable(target, {}); proxy.revoke();
                try { functionRealm(proxy.proxy); throw Error('missing revocation check'); }
                catch (e) { check(e instanceof TypeError, 'revocation error realm'); }
                check(functionRealm(HostConstructor) === globalThis, 'native function realm');
                return 'ok';
            })()
        "#;
        match engine.eval(source, false).unwrap() {
            Completion::Value(value) => assert_eq!(value, "ok", "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}
