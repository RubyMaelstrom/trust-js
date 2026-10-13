//! Packed constructor-to-property ownership transfers. Local ECMA-262 e28783d5fc9d:
//! [[Construct]] (§10.2.2), OrdinarySetWithOwnDescriptor (§10.1.9.2), ToBoolean (§7.1.2).
//! Guards still perform the normative operations; only the intermediate representation changes.
use super::*;
use crate::bytecode::Tier;
use crate::{Completion, Engine};
use std::cell::Cell;

thread_local! {
    static SIMPLE: Cell<usize> = const { Cell::new(0) };
    static FORWARDED: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn record_simple() {
    SIMPLE.with(|count| count.set(count.get() + 1));
}

pub(super) fn record_forwarded() {
    FORWARDED.with(|count| count.set(count.get() + 1));
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("constructor fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn setup(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        function Simple(x,y){this.value=x;this.other=y;}
        function Wrapper(){this.initialize.apply(this,arguments);}
        Wrapper.prototype.initialize=function(x,y){
            if(!x)x=19;if(!y)y=23;this.value=x;this.other=y;
        };
        function simple(x,y){return new Simple(x,y);}
        function forwarded(x,y){return new Wrapper(x,y);}
        for(var warm=0;warm<600;warm++){simple(warm,1);forwarded(warm,2);}
    "#,
    );
    engine
}

fn assert_native_hit(tier: Tier, forwarded: bool, before: usize) {
    if tier == Tier::Jit
        && cfg!(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(
                target_os = "macos",
                target_os = "linux",
                target_os = "windows"
            )
        ))
    {
        let after = if forwarded {
            FORWARDED.with(Cell::get)
        } else {
            SIMPLE.with(Cell::get)
        };
        assert!(
            after > before,
            "must exercise committed packed transfer, not just its fallback"
        );
    }
}

#[test]
fn constructor_transfer_every_value_kind_and_descriptor_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let before = SIMPLE.with(Cell::get);
        let mut engine = setup(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var values=[undefined,null,false,true,0,-0,NaN,Infinity,-Infinity,
                3.25,-4.5,0n,-98765432109876543210n,'','λ string',Symbol('s'),{key:7}];
            var okay=true;
            for(var pass=0;pass<40;pass++)for(var k=0;k<values.length;k++){
                var result=simple(values[k],values[(k+1)%values.length]);
                var desc=Object.getOwnPropertyDescriptor(result,'value');
                okay=okay&&Object.is(result.value,values[k])&&
                    Object.is(result.other,values[(k+1)%values.length])&&
                    desc.writable&&desc.enumerable&&desc.configurable&&
                    Object.keys(result).join(',')==='value,other';
            }
            okay;
        "#
            ),
            "true",
            "{tier:?}"
        );
        assert_native_hit(tier, false, before);
    }
}

#[test]
fn constructor_transfer_forwarded_defaults_keep_truthiness_and_identity() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let before = FORWARDED.with(Cell::get);
        let mut engine = setup(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var falseValues=[undefined,null,false,0,-0,NaN,0n,'',$262.IsHTMLDDA];
            var calls=0, coercion={valueOf:function(){calls++;throw 42;},toString:function(){calls++;throw 43;}};
            var trueValues=[true,3.25,-4.5,Infinity,-Infinity,1n,-1n,'0','λ',Symbol('s'),coercion];
            var okay=true;
            for(var pass=0;pass<40;pass++){
                for(var k=0;k<falseValues.length;k++){
                    var result=forwarded(falseValues[k],falseValues[k]);
                    okay=okay&&result.value===19&&result.other===23;
                }
                for(var k=0;k<trueValues.length;k++){
                    var result=forwarded(trueValues[k],trueValues[k]);
                    okay=okay&&Object.is(result.value,trueValues[k])&&Object.is(result.other,trueValues[k]);
                }
            }
            var missing=new Wrapper();
            okay&&calls===0&&missing.value===19&&missing.other===23;
        "#
            ),
            "true",
            "{tier:?}"
        );
        assert_native_hit(tier, true, before);
    }
}

#[test]
fn constructor_transfer_sole_object_owner_survives_gc_then_releases() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for maker in ["simple", "forwarded"] {
            let mut engine = setup(tier);
            eval(
                &mut engine,
                &format!("var retained={maker}({{marker:71}}, Symbol('retained'));"),
            );
            let env = engine.interp.global_env.clone();
            let retained = engine.interp.get_var("retained", &env).ok().unwrap();
            let object = retained.as_obj().unwrap();
            let payload = object.borrow().props.get("value").unwrap().value();
            let weak = Rc::downgrade(payload.as_obj().unwrap());
            drop(payload);
            engine.interp.gc_collect();
            assert_eq!(
                eval(
                    &mut engine,
                    "retained.value.marker+'|'+String(retained.other)"
                ),
                "71|Symbol(retained)"
            );
            eval(&mut engine, "retained=null;");
            drop(retained);
            engine.interp.gc_collect();
            assert!(
                weak.upgrade().is_none(),
                "{tier:?}, {maker}: no stale argument owner"
            );
        }
    }
}

#[test]
fn constructor_transfer_surplus_evaluation_duplicates_and_fallback_order() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = setup(tier);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var log=[], token={id:1};
            function argument(n){log.push(n);return token;}
            var result=new Simple(argument(1),argument(2),argument(3));
            function Duplicate(x){this.a=x;this.b=x;}
            var dup;
            for(var k=0;k<200;k++)dup=new Duplicate(token);
            Object.defineProperty(Simple.prototype,'value',{
                configurable:true,set:function(v){log.push(v===token?'setter':'bad');this.saved=v;}
            });
            var fallback=new Simple(argument(4),argument(5),argument(6));
            var beforeThrow={};
            Object.defineProperty(Simple.prototype,'value',{
                configurable:true,set:function(v){log.push('throw');throw beforeThrow;}
            });
            var thrown=false;try{new Simple(argument(7),argument(8),argument(9));}catch(e){thrown=e===beforeThrow;}
            [result.value===token,result.other===token,dup.a===token,dup.b===token,
                fallback.saved===token,!Object.hasOwn(fallback,'value'),fallback.other===token,
                thrown,log.join(',')].join('|');
        "#
            ),
            "true|true|true|true|true|true|true|true|1,2,3,4,5,6,setter,7,8,9,throw",
            "{tier:?}"
        );
    }
}
