//! ECMA-262 #sec-set-immutable-prototype and WHATWG HTML
//! #windowproxy-{setprototypeof,isextensible,preventextensions}: the essential internal methods
//! a host selects with `Ctx::set_host_object_invariants`.

use crate::embed::Value;
use crate::realm_inline_guard_tests::TIERS;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("host object fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

#[test]
fn host_object_semantics_window_style_invariants_and_proxy_checks() {
    for tier in TIERS {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert!(engine.scope(|cx| {
            let window = cx.global_this();
            cx.set_host_object_invariants(&window, true, true)
        }));
        assert_eq!(
            eval(
                &mut engine,
                r#"(() => {
            const check=(ok,label)=>{if(!ok)throw Error(label)};
            const object=globalThis, prototype=Object.getPrototypeOf(object);
            check(Reflect.setPrototypeOf(object,prototype)&&!Reflect.setPrototypeOf(object,{}),'immutable');
            let error='';try{Object.setPrototypeOf(object,{})}catch(e){error=e.name}
            check(error==='TypeError'&&Object.getPrototypeOf(object)===prototype,'setPrototypeOf');
            error='';try{object.__proto__={}}catch(e){error=e.name}
            check(error==='TypeError'&&Object.getPrototypeOf(object)===prototype,'__proto__');
            check(Reflect.isExtensible(object)&&!Reflect.preventExtensions(object),'extensible');
            for(const operation of [Object.preventExtensions,Object.seal,Object.freeze]) {
                error='';try{operation(object)}catch(e){error=e.name}
                check(error==='TypeError'&&Reflect.isExtensible(object),'integrity level');
            }
            object.expando=42;check(object.expando===42,'expando');
            const wrapped=new Proxy(object,{preventExtensions(){return true}});
            error='';try{Reflect.preventExtensions(wrapped)}catch(e){error=e.name}
            check(error==='TypeError','Proxy invariant');
            const forwarded=new Proxy(object,{});
            check(!Reflect.preventExtensions(forwarded)&&!Reflect.setPrototypeOf(forwarded,{}),'Proxy forwarding');
            const copy=Object.assign({},object);
            check(Reflect.setPrototypeOf(copy,{})&&Reflect.preventExtensions(copy),'copy has no host rules');
            object['#\0hostAlwaysExtensible']=false;delete object['#\0hostImmutablePrototype'];
            check(!Reflect.preventExtensions(object)&&!Reflect.setPrototypeOf(object,{}),'author properties cannot replace internal slots');
            return 'host-object-ok';
        })()"#
            ),
            "host-object-ok",
            "{tier:?}"
        );
    }
}

#[test]
fn host_object_semantics_configuration_refuses_proxies_and_existing_integrity_levels() {
    let mut engine = Engine::new();
    engine.scope(|cx| {
        let object = cx.new_object_with_proto(&Value::Null);
        assert!(cx.set_host_object_invariants(&object, true, false));
        let global = cx.global_this();
        cx.member_set(&global, "hostObject", object)
            .unwrap_or_else(|_| panic!("publish test object"));
    });
    // Only the prototype rule was selected: the object can still become non-extensible.
    assert_eq!(
        eval(&mut engine, "Reflect.preventExtensions(hostObject)"),
        "true"
    );
    engine.scope(|cx| {
        let global = cx.global_this();
        let object = cx
            .member_get(&global, "hostObject")
            .unwrap_or_else(|_| panic!("read test object"));
        assert!(!cx.set_host_object_invariants(&object, false, true));
        let proxy = cx
            .eval_value("new Proxy({},{})")
            .expect("parse")
            .unwrap_or_else(|_| panic!("proxy"));
        assert!(!cx.set_host_object_invariants(&proxy, true, true));
        assert!(!cx.set_host_object_invariants(&Value::Num(1.0), true, true));
    });
}
