//! CopyDataProperties / CreateDataPropertyOrThrow, ECMA-262 snapshot e28783d5:
//! spec.html:6740, 6290, 40915, 41269. Closed data paths must preserve live descriptor
//! checks, ordered effects, independent owners, and ArraySpeciesCreate's observable targets.

use crate::bytecode::Tier;
use crate::value::{Property, Props, Value};
use crate::{Completion, Engine};
use std::rc::Rc;

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine
            .eval(source, false)
            .expect("data copy fixture parses")
        {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        engine.interp.gc_collect();
        assert!(matches!(engine.eval("1+2", false), Ok(Completion::Value(v)) if v == "3"));
    }
}

#[test]
fn spread_copies_frozen_data_into_independent_default_descriptors() {
    check(
        r#"
        function run() {
            var nested={v:7}, source=Object.freeze({a:1,nested:nested,length:3,prototype:4});
            var copy={...source};
            copy.a=2;delete copy.length;copy.extra=5;
            var d=Object.getOwnPropertyDescriptor(copy,'nested');
            var merged={prefix:0,a:9,...source,a:6,...{last:8}};
            var {a,...rest}=source;
            return [source.a,copy.a,copy.nested===nested,d.writable,d.enumerable,d.configurable,
                Object.keys(copy).join(','),Object.keys(merged).join(','),merged.a,
                Object.keys(rest).join(',')].join('|');
        }
        run()
        "#,
        "1|2|true|true|true|true|a,nested,prototype,extra|prefix,a,nested,length,prototype,last|6|nested,length,prototype",
    );
}

#[test]
fn spread_getters_keep_key_snapshot_and_read_later_descriptors_live() {
    check(
        r#"
        function run(){
            var log=[], source={a:1,get b(){log.push('b');delete this.c;this.d=4;return 2},c:3};
            var copy={...source};
            var sym=Symbol('s'), numbered={z:9,2:2,1:1,[sym]:7};
            Object.defineProperty(numbered,'hidden',{value:8});
            var other={...numbered};
            var proxy=new Proxy({a:5,b:6},{ownKeys(t){log.push('keys');return ['b','a']},
                getOwnPropertyDescriptor(t,k){log.push('desc'+k);return Object.getOwnPropertyDescriptor(t,k)},
                get(t,k){log.push('get'+k);return t[k]}});
            var proxied={...proxy};
            return [Object.keys(copy).join(','),copy.b,Object.keys(other).join(','),other[sym],
                Object.keys(proxied).join(','),log.join(',')].join('|');
        }run()
        "#,
        "a,b|2|1,2,z|7|b,a|b,keys,descb,getb,desca,geta",
    );
}

#[test]
fn spread_shared_layout_preserves_mutated_descriptors_and_symbol_owners() {
    check(
        r#"
        function copy(o){return {...o}}
        var o={a:1,b:2}, out=[];
        for(var k=0;k<140;k++)out.push(copy(o).b);
        Object.defineProperty(o,'b',{get(){return 9},enumerable:true,configurable:true});
        out.push(copy(o).b);
        Object.defineProperty(o,'a',{value:1,enumerable:false});
        var s=Symbol('s');o[s]={n:7};
        var c=copy(o);delete o[s];
        [out[0],out[139],out[140],Object.keys(c).join(','),c[s].n].join('|')
        "#,
        "2|2|9|b|7",
    );
}

#[test]
fn array_data_creation_rechecks_species_result_after_callbacks() {
    check(
        r#"
        function run(){
            var result,trace=[],a=[1,2,3];
            a.constructor={[Symbol.species]:function(n){result=new Array(n);return result}};
            var mapped=a.map((v,k)=>{trace.push(k);if(k===0){result.length=0;
                Object.defineProperty(result,'0',{get(){throw 99},configurable:true})}return v+10});
            var flags=Object.getOwnPropertyDescriptor(mapped,'0');
            var holey=[1,,3].map(v=>v+1),filtered=[1,,3].filter(v=>true);
            var calls=0;
            try{a.map((v,k)=>{calls++;if(k===0)Object.preventExtensions(result);return v})}
            catch(e){trace.push(e instanceof TypeError)}
            try{a.map((v,k)=>{if(k===0){result.length=0;Object.defineProperty(result,'length',{writable:false})}return v})}
            catch(e){trace.push(e instanceof TypeError)}
            return [mapped.join(','),mapped.length,flags.writable,flags.enumerable,flags.configurable,
                holey.join(','),holey.length,1 in holey,filtered.join(','),calls,trace.join(',')].join('|');
        }run()
        "#,
        "11,12,13|3|true|true|true|2,,4|3|false|1,3|1|0,1,2,true,true",
    );
}

#[test]
fn array_creation_preserves_proxy_defines_and_non_index_boundaries() {
    check(
        r#"
        function run(){
            var trace=[],a=[1,,3];
            a.constructor={[Symbol.species]:function(n){trace.push('species'+n);return new Proxy({}, {
                defineProperty(t,k,d){trace.push(k+':'+d.value+':'+d.writable+':'+d.enumerable+':'+d.configurable);
                    return Reflect.defineProperty(t,k,d)}})}};
            var b=a.map(v=>v*2);
            var boundary={4294967294:'a',4294967295:'b',length:4294967296};
            boundary.constructor={[Symbol.species]:function(){return {}}};
            var result=Array.prototype.slice.call(boundary,4294967294);
            return [b[0],b[2],trace.join(','),result.join(',')].join('|');
        }run()
        "#,
        "2|6|species3,0:2:true:true:true,2:6:true:true:true|a,b",
    );
}

#[test]
fn named_copy_shares_only_keys_and_normalizes_flags() {
    let mut source = Props::new();
    source.insert(
        "a",
        Property::data(Value::lstr("owned"), false, true, false),
    );
    source.insert("length", Property::plain(Value::Num(3.0)));
    let mut target = Props::new();
    assert!(target.try_copy_named_data_from(&source));
    assert_eq!(source.shape(), target.shape());
    assert!(Rc::ptr_eq(
        source.shared_layout().unwrap(),
        target.shared_layout().unwrap()
    ));
    let property = target.get("a").unwrap();
    assert!(property.writable() && property.enumerable() && property.configurable());
    target.insert("a", Property::plain(Value::lstr("changed")));
    target.insert("tail", Property::plain(Value::Num(7.0)));
    assert!(matches!(source.get("a").unwrap().value(),Value::Str(s) if &*s == "owned"));
    assert!(!source.contains("tail"));
    assert!(matches!(
        target.length_property().unwrap().value(),
        Value::Num(3.0)
    ));
}

#[test]
fn named_copy_rebuilds_wide_lookup_and_keeps_indexed_sources_checked() {
    let mut source = Props::new();
    for n in 0..80 {
        source.insert(format!("field{n}"), Property::plain(Value::Num(n as f64)));
    }
    let mut target = Props::new();
    assert!(target.try_copy_named_data_from(&source));
    for n in 0..80 {
        assert!(
            matches!(target.get(&format!("field{n}")).unwrap().value(),Value::Num(value) if value==n as f64)
        );
    }
    assert_eq!(source.shape(), target.shape());
    target.remove("field2");
    assert!(source.contains("field2"));
    source.insert("4294967294", Property::plain(Value::Num(9.0)));
    let mut refused = Props::new();
    assert!(!refused.try_copy_named_data_from(&source));
    assert_eq!(refused.iter().count(), 0, "a guard miss writes no prefix");
}

#[test]
fn assign_preserves_setters_readonly_descriptors_proxies_and_proto_set() {
    check(
        r#"
        function run(){
            var source=Object.freeze({a:1,b:2}),copy=Object.assign({},source),trace=[];
            copy.a=3;
            var proto={set a(v){trace.push('set'+v)}};
            var target=Object.create(proto);Object.assign(target,source);
            var blocked=Object.create({});
            Object.defineProperty(Object.getPrototypeOf(blocked),'b',{value:9});
            try{Object.assign(blocked,source)}catch(e){trace.push(e instanceof TypeError)}
            var attrs={};Object.defineProperty(attrs,'a',{value:0,writable:true});
            Object.assign(attrs,{a:4});
            var p={marker:7}, withProto=Object.assign({}, {['__proto__']:p});
            var proxy=new Proxy({}, {set(t,k,v){trace.push(k+v);t[k]=v;return true}});
            Object.assign(proxy,source);
            return [copy.a,target.hasOwnProperty('a'),target.b,blocked.a,attrs.a,
                Object.getOwnPropertyDescriptor(attrs,'a').enumerable,withProto.marker,
                trace.join(',')].join('|');
        }run()
        "#,
        "3|false|2|1|4|false|7|set1,true,a1,b2",
    );
}
