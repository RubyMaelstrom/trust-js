//! Host constructors receive the same NewTarget/GetFunctionRealm semantics as
//! built-ins: ECMA-262 #sec-builtincallorconstruct / #sec-getfunctionrealm,
//! local snapshot e28783d5fc9dc12b3de905961e2c71410b38a202.

use crate::{bytecode::Tier, embed::Value, Completion, Engine};

#[test]
fn native_constructor_context_survives_reentry_and_reports_function_realms() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.scope(|cx| {
            let ctor = cx.new_native_fn_with_captures(
                "HostConstructor",
                1,
                |ctx, _, args, _| {
                    let target = ctx.constructor_new_target();
                    if !ctx.is_constructing() {
                        return Ok(target);
                    }
                    if let Some(callback) = args.first() {
                        ctx.invoke(*callback, Value::Undefined, &[])?;
                    }
                    assert!(ctx.values_strict_equal(&target, &ctx.constructor_new_target()));
                    Ok(target)
                },
                &[],
            );
            let prototype = Value::Obj(cx.new_object());
            cx.set_constructor_prototype(&ctor, &prototype);
            let global = cx.global_this();
            cx.member_set(&global, "HostConstructor", ctor)
                .unwrap_or_else(|_| panic!("install host constructor"));
            let foreign = cx.create_embed_realm();
            cx.member_set(&global, "foreign", foreign)
                .unwrap_or_else(|_| panic!("install foreign Realm"));
        });
        engine.define_global("functionRealm", 1, |ctx, _, args| {
            ctx.callable_realm_global(args.first().unwrap_or(&Value::Undefined))
        });
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

/// Actual bare scoped callbacks use Callable tag 13, including native constructor/super paths
/// which the captured-native fixture above cannot cover.
#[test]
fn bare_scoped_native_constructors_cover_new_target_bound_proxy_reflect_and_super() {
    #[inline(never)]
    fn constructor<'s>(
        cx: &mut crate::embed::Ctx<'s>,
        _: Value<'s>,
        args: &[Value<'s>],
    ) -> Result<Value<'s>, Value<'s>> {
        let target = cx.constructor_new_target();
        if !cx.is_constructing() {
            return Ok(target);
        }
        if matches!(
            args.first().map(Value::kind),
            Some(crate::scope::ValueKind::Bool(true))
        ) {
            return Err(args.get(1).copied().unwrap_or_default());
        }
        if let Some(callback) = args.first().filter(|value| value.is_callable()) {
            cx.invoke(*callback, Value::Undefined, &[])?;
            assert!(cx.values_strict_equal(&target, &cx.constructor_new_target()));
        }
        let prototype = cx.member_get(&target, "prototype")?;
        let instance = cx.new_object_with_proto(&prototype);
        cx.member_set(&instance, "target", target)?;
        Ok(instance)
    }
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.scope(|cx| {
            let ctor = cx.make_native(
                "BareHost",
                2,
                std::hint::black_box(constructor as crate::embed::NativeFn),
            );
            assert!(matches!(
                &ctor.gc().borrow().call,
                crate::value::Callable::ScopedNative(_)
            ));
            let ctor = Value::Obj(ctor);
            cx.set_constructor_prototype(&ctor, &Value::Obj(cx.new_object()));
            let global = cx.global_this();
            cx.member_set(&global, "BareHost", ctor).unwrap();
        });
        let source = r#"
            function check(v,m){if(!v)throw Error(m)}
            function Target(){}
            function callback(){check(BareHost()===undefined,'nested ordinary');check(new BareHost().target===BareHost,'nested new')}
            class Derived extends BareHost {constructor(){super(callback);this.derived=true}}
            var bound=BareHost.bind(null), proxy=new Proxy(BareHost, {}), error={nativeThrow:true};
            for(var n=0;n<40;n++) {
                var ordinary=new BareHost(callback), boundValue=new bound(), proxyValue=new proxy(), derived=new Derived();
                check(ordinary instanceof BareHost && ordinary.target===BareHost,'ordinary');
                check(boundValue instanceof BareHost && boundValue.target===BareHost,'bound');
                check(proxyValue instanceof BareHost && proxyValue.target===proxy,'proxy');
                var reflected=Reflect.construct(BareHost,[callback],Target);
                check(reflected instanceof Target && reflected.target===Target,'Reflect.construct');
                check(derived instanceof Derived && derived.derived && derived.target===Derived,'super');
                try {new BareHost(true,error);throw 'missing throw'}catch(e){check(e===error,'typed native throw')}
                check(BareHost()===undefined,'restored after throw');
            }
            'ok'
        "#;
        match engine.eval(source, false).unwrap() {
            Completion::Value(value) => assert_eq!(value, "ok", "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}
