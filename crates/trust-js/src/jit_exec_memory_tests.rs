//! Compact native memory boundaries, checked against every execution tier.
//! ECMA-262 snapshot e28783d5: Get/Set, IsStrictlyEqual, ToBoolean and Number
//! semantics remain unchanged by the owned-word representation.

use crate::{bytecode::Tier, Completion, Engine};

#[test]
fn compiled_numeric_array_initialization_retains_coherent_mirrors() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for source in [
            "var numeric=[];for(let index=0;index<128;index++)numeric[index]=index+0.25;",
            "var numeric=[],other=[];for(let index=0;index<128;index++)numeric[index]=other[index]=index+0.25;",
            "function fill(){let array=[];for(let index=0;index<128;index++)array[index]=index+0.25;return array;}var numeric=fill();",
        ] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            assert!(matches!(engine.eval(source, false).unwrap(), Completion::Value(_)));
            let global = crate::value::Value::Obj(engine.interp.global.clone());
            let value = engine.interp.get_member(&global, "numeric")
                .unwrap_or_else(|_| panic!("numeric fixture binding"));
            let object = value.as_obj().expect("numeric array").borrow();
            for index in 0..128 {
                assert_eq!(object.props.mirror_get(index), Some(index as f64 + 0.25),
                    "{tier:?}: index {index}, {source}");
            }
            drop(object);
            #[cfg(target_arch = "aarch64")]
            super::TEST_NUMERIC_REGION_ENTRIES.with(|count| count.set(0));
            assert!(matches!(engine.eval(
                "function scan(array){let sum=0;for(let index=0;index<128;index++)sum+=array[index];return sum;}scan(numeric)",
                false).unwrap(), Completion::Value(value) if value == "8160"));
            #[cfg(target_arch = "aarch64")]
            if tier == Tier::Jit {
                assert!(super::TEST_NUMERIC_REGION_ENTRIES.with(|count| count.get()) > 0,
                    "numeric receiver guards never admitted the initialized array: {source}");
            }
        }
    }
}

fn assert_all_tiers(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "tier {tier:?}"),
            Completion::Throw { name, message } => panic!("tier {tier:?}: {name}: {message}"),
        }
    }
}

#[test]
fn compact_native_properties_elements_and_name_owners() {
    assert_all_tiers(
        "var shared, captured;
         function named(v) { shared=v; return shared; }
         function exercise(v) {
             let box={value:v}, array=[v,0], copy;
             function scope(value) { captured=value; return captured; }
             for (let k=0;k<50;k++) {
                 copy=box.value;
                 if (!Object.is(copy,v)) throw 'property read';
                 box.value=v;
                 if (!Object.is(array[0],v)) throw 'element read';
                 array[1]=k+0.25;
                 if (array[1]!==k+0.25) throw 'element write';
                 if (!Object.is(named(v),v) || !Object.is(scope(v),v)) throw 'name';
             }
             return Object.is(copy,v);
         }
         var values=[undefined,null,false,true,-0,NaN,Infinity,-Infinity,42.5,
                     'text',Symbol('same'),{x:1},12345678901234567890n];
         values.map(exercise).every(function(x){return x;})",
        "true",
    );
}

#[test]
fn compact_native_comparisons_and_truthiness_preserve_identity() {
    assert_all_tiers(
        "function compare(a,b) {
             return [a===b,a!==b,a==b,a!=b,!a,!!b,a??'nullish',b?1:0];
         }
         function check(a,b) {
             var first=compare(a,b).map(String).join('|');
             for(var k=0;k<80;k++) {
                 if(compare(a,b).map(String).join('|')!==first) throw 'unstable';
             }
             return first;
         }
         var o={},s=Symbol('x');
         [check(-0,0),check(NaN,NaN),check(null,undefined),check(o,o),check(s,s),
          check('same','same'),check('',false)].join(';')",
        "true|false|true|false|true|false|0|0;false|true|false|true|true|false|NaN|0;false|true|true|false|true|false|nullish|0;true|false|true|false|false|true|[object Object]|1;true|false|true|false|false|true|Symbol(x)|1;true|false|true|false|false|true|same|1;false|true|true|false|true|false||0",
    );
}

#[test]
fn compact_native_numeric_element_fusions_keep_nan_zero_and_updates() {
    assert_all_tiers(
        "function scan(a) {
             let k=-1,sum=0;
             for(let i=0;i<4;i++) sum+=a[++k];
             a[0]=-0; a[1]=0/0; a[2]=Infinity; a[3]=-Infinity;
             return [sum,1/a[0],Number.isNaN(a[1]),a[2],a[3]].join(',');
         }
         function increment(o) { let before=o.n++; return before+','+(++o.n); }
         var result;
         for(let j=0;j<70;j++) {
             result=scan([1,2,3,4]);
             if(increment({n:0})!=='0,2') throw 'update';
         }
         result",
        "10,-Infinity,true,Infinity,-Infinity",
    );
}

#[test]
fn compact_native_scalar_equality_keeps_nan_nullish_and_coercion_distinct() {
    assert_all_tiers(r#"
        function flags(a,b){return [a==b,a!=b,a===b,a!==b].join(':');}
        var symbol=Symbol('same'),object={},dda=$262.IsHTMLDDA,trace=[];
        var coercing={valueOf(){trace.push('coerce');return 7;}};
        function run(){return [flags(-0,0),flags(NaN,NaN),flags(undefined,null),
            flags(undefined,undefined),flags(false,false),flags(false,true),flags(0,false),
            flags(1n,1),flags(1n,1n),flags('same',('s'+'ame').slice(0)),
            flags(symbol,symbol),flags(Symbol('x'),Symbol('x')),
            flags(object,object),flags({},{}),flags(dda,null),flags(null,dda),
            flags(coercing,7)].join('|');}
        var result;for(var k=0;k<150;k++)result=run();
        result+'|'+trace.length;
    "#, "true:false:true:false|false:true:false:true|true:false:false:true|true:false:true:false|true:false:true:false|false:true:false:true|true:false:false:true|true:false:false:true|true:false:true:false|true:false:true:false|true:false:true:false|false:true:false:true|true:false:true:false|false:true:false:true|true:false:false:true|true:false:false:true|true:false:false:true|300");
}

#[test]
fn compact_native_truthy_peeks_preserve_owners_and_htmldda() {
    assert_all_tiers(r#"
        function observe(value){
            var branch;if(value)branch=1;else branch=0;
            return [!value,!!value,branch,(value&&'taken')==='taken',
                (value||'fallback')==='fallback',(value??'nullish')==='nullish',
                Object.is(value??'nullish',value)].join(':');
        }
        var values=[undefined,null,false,true,0,-0,NaN,Infinity,-Infinity,42.25,
            '', 'text', Symbol('s'), {}, 0n, 1n, $262.IsHTMLDDA];
        function run(){var out=[];for(var k=0;k<values.length;k++)out.push(observe(values[k]));return out.join('|');}
        var result;for(var k=0;k<120;k++)result=run();result;
    "#, "true:false:0:false:true:true:false|true:false:0:false:true:true:false|true:false:0:false:true:false:true|false:true:1:true:false:false:true|true:false:0:false:true:false:true|true:false:0:false:true:false:true|true:false:0:false:true:false:true|false:true:1:true:false:false:true|false:true:1:true:false:false:true|false:true:1:true:false:false:true|true:false:0:false:true:false:true|false:true:1:true:false:false:true|false:true:1:true:false:false:true|false:true:1:true:false:false:true|true:false:0:false:true:false:true|false:true:1:true:false:false:true|true:false:0:false:true:false:true");
}
