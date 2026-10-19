//! ECMA-262 e28783d5 (local official snapshot, 2026-09-06): PutValue,
//! EvaluatePropertyAccessWithExpressionKey, ToPropertyKey, PropertyDefinitionEvaluation,
//! OrdinaryOwnPropertyKeys. A Reference owns its Symbol key until the heap takes ownership.

use crate::{bytecode::Tier, value::Callable, Completion, Engine};

fn check_all_tiers(source: &str, native_functions: &[&str]) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine
            .eval(source, false)
            .expect("reference fixture parses")
        {
            Completion::Value(value) => assert_eq!(value, "ok", "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        if tier == Tier::Jit
            && cfg!(all(
                any(target_arch = "aarch64", target_arch = "x86_64"),
                any(
                    target_os = "linux",
                    target_os = "macos",
                    target_os = "windows"
                )
            ))
        {
            for name in native_functions {
                let env = engine.interp.global_env.clone();
                let function = engine
                    .interp
                    .get_var(name, &env)
                    .unwrap_or_else(|_| panic!("{name}"));
                let object = function.as_obj().expect("fixture function").borrow();
                let Callable::User(user) = &object.call else {
                    panic!("fixture user function")
                };
                let chunk = user
                    .func
                    .code
                    .get()
                    .and_then(Option::as_ref)
                    .expect("compiled function");
                assert!(
                    chunk.jit.get().flatten().is_some(),
                    "{name} has native code"
                );
            }
        }
    }
}

#[test]
fn computed_reference_put_checks_base_after_rhs_before_key() {
    check_all_tiers(
        r#"
        function check(ok) { if (!ok) throw Error('computed reference order'); }
        function kept(base, key, rhs) { return base[key] = rhs(); }
        function dropped(base, key, rhs) { base[key] = rhs(); }
        function general(base, key, rhs) { return base()[key()] = rhs(); }
        for (var warm=0; warm<150; ++warm) {
            for (var put of [kept,dropped]) {
                var events=[], key={toString(){events.push('key');throw Error('key must not run')}};
                for (var base of [null,undefined]) {
                    events=[];
                    try {put(base,key,()=>{events.push('rhs');return 1});check(false)}
                    catch(e){check(e instanceof TypeError && events.join(',')==='rhs')}
                }
                var rhsError={};
                try {put(null,key,()=>{throw rhsError});check(false)} catch(e){check(e===rhsError)}
            }
            var order=[], object={}, k={ [Symbol.toPrimitive](hint){
                check(hint==='string');order.push('coerce');return 'x';
            }};
            Object.defineProperty(object,'x',{set(v){order.push('set:'+v)}});
            check(general(()=>{order.push('base');return object},()=>{
                order.push('key-expr');return k;
            },()=>{order.push('rhs');return 9})===9);
            check(order.join(',')==='base,key-expr,rhs,coerce,set:9');
            var trap=0, proxy=new Proxy({}, {set(t,k,v,r){check(k==='x'&&v===3&&r===proxy);trap++;return true}});
            check(kept(proxy,'x',()=>3)===3 && trap===1);
        }
        'ok'
    "#,
        &["kept", "dropped", "general"],
    );
}

#[test]
fn computed_reference_symbol_literals_own_keys_across_values_and_gc() {
    check_all_tiers(
        r#"
        function check(ok) { if (!ok) throw Error('computed symbol ownership'); }
        function build() {
            return {
                [Symbol('value')]: ($262.gc(),17),
                [{[Symbol.toPrimitive](){return Symbol('anonymous')}}]: function(){},
                [Symbol('method')](){return 23},
                get [Symbol('getter')](){return 29},
                set [Symbol('setter')](v){this.seen=v}
            };
        }
        for(var n=0;n<150;n++) {
            var o=build();$262.gc();var keys=Object.getOwnPropertySymbols(o);
            check(keys.length===5 && keys.map(k=>k.description).join(',')==='value,anonymous,method,getter,setter');
            check(o[keys[0]]===17 && o[keys[1]].name==='[anonymous]' && o[keys[2]]()===23 && o[keys[3]]===29);
            o[keys[4]]=31;check(o.seen===31);
            check(Object.getOwnPropertyDescriptor(o,keys[3]).get.name==='get [getter]');
            check(Object.getOwnPropertyDescriptor(o,keys[4]).set.name==='set [setter]');
            var spread={...o};check(Object.getOwnPropertySymbols(spread).length===5);
        }
        'ok'
    "#,
        &["build"],
    );
}

#[test]
fn computed_reference_compound_and_super_retain_coerced_symbol_once() {
    check_all_tiers(
        r#"
        function check(ok) { if (!ok) throw Error('compound symbol reference'); }
        function compound(object,key){return object[key] += ($262.gc(),2)}
        class Base {}
        class Derived extends Base {
            compound(){return super[{[Symbol.toPrimitive](){return Symbol('super-compound')}}] += ($262.gc(),2)}
            update(){return super[{[Symbol.toPrimitive](){return Symbol('super-update')}}]++}
        }
        Object.setPrototypeOf(Derived.prototype,new Proxy({}, {
            get(t,k){check(typeof k==='symbol');return 4}
        }));
        for(var n=0;n<150;n++) {
            var count=0, object=new Proxy({}, {get(t,k){check(typeof k==='symbol');return 3},
                set(t,k,v){check(typeof k==='symbol');t[k]=v;return true}});
            check(compound(object,{[Symbol.toPrimitive](){count++;return Symbol('compound')}})===5);
            var keys=Object.getOwnPropertySymbols(object);
            check(count===1 && keys.length===1 && keys[0].description==='compound');
            var receiver=new Derived();check(receiver.compound()===6 && receiver.update()===4);
            keys=Object.getOwnPropertySymbols(receiver);
            check(keys.length===2 && keys[0].description==='super-compound' && keys[1].description==='super-update');
            check(receiver[keys[0]]===6 && receiver[keys[1]]===5);
        }
        'ok'
    "#,
        &["compound"],
    );
}
