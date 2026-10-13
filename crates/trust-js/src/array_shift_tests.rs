//! ECMA-262 e28783d5: Array.prototype.shift, OrdinarySetWithOwnDescriptor,
//! DeletePropertyOrThrow, LengthOfArrayLike and ArraySetLength.
use crate::{bytecode::Tier, value::Value, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for threshold in [0, 32] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(threshold);
            engine
                .interp
                .def_method(&engine.interp.global, "collectShiftTest", 0, |i, _, _| {
                    i.gc_collect();
                    Ok(Value::Undefined)
                });
            engine.interp.def_method(
                &engine.interp.global,
                "collectShiftYoungTest",
                0,
                |i, _, _| {
                    i.gc_collect_young(crate::value::GcCause::Explicit);
                    Ok(Value::Undefined)
                },
            );
            match engine.eval(source, false).expect("shift fixture parses") {
                Completion::Value(actual) => assert_eq!(actual, expected, "{tier:?}/{threshold}"),
                Completion::Throw { name, message } => {
                    panic!("{tier:?}/{threshold}: {name}: {message}")
                }
            }
            engine.interp.gc_collect();
            assert!(matches!(engine.eval("21+21", false), Ok(Completion::Value(v)) if v == "42"));
        }
    }
}

#[test]
fn array_shift_mixed_values_preserve_identity_keys_and_own_descriptor_attributes() {
    check(
        r#"
        var shift=Array.prototype.shift,sym=Symbol('value'),key=Symbol('key'),object={},fn=function(){};
        var values=[undefined,null,true,false,-0,NaN,Infinity,-Infinity,1.25,
            12345678901234567890n,'text',sym,object,fn];
        var a=values.slice(),count=0;
        a.named=object;a[key]=fn;a['01']='one';a['4294967295']='large';
        for(var n=0;n<values.length;n++){
            var first=shift.call(a);
            if(!Object.is(first,values[n])||a.length!==values.length-n-1)throw 'value/length';
            if(a.named!==object||a[key]!==fn||a['01']!=='one'||a['4294967295']!=='large')throw 'named';
            if(Object.hasOwn(a,a.length))throw 'tail not deleted';
            collectShiftTest();count++;
        }
        if(shift.call(a)!==undefined||a.length!==0)throw 'empty';
        var b=[10,20,30];
        Object.defineProperty(b,'0',{enumerable:false,configurable:false});
        Object.defineProperty(b,'1',{configurable:false});
        Object.defineProperty(b,'2',{writable:false});
        Object.preventExtensions(b);
        if(shift.call(b)!==10||b.length!==2||b[0]!==20||b[1]!==30)throw 'attributes result';
        var d0=Object.getOwnPropertyDescriptor(b,'0'),d1=Object.getOwnPropertyDescriptor(b,'1');
        if(d0.enumerable||d0.configurable||!d0.writable||!d1.enumerable||d1.configurable||!d1.writable)
            throw 'attributes moved';
        if(Object.isExtensible(b)||Reflect.ownKeys(b).join(',')!=='0,1,length')throw 'keys';
        var c=[{id:1},{id:2}],traps=0;
        Object.setPrototypeOf(c,new Proxy({}, {
            get(){traps++;throw 'prototype get'},has(){traps++;throw 'prototype has'},
            set(){traps++;throw 'prototype set'},getOwnPropertyDescriptor(){traps++;throw 'prototype own'}
        }));
        Object.preventExtensions(c);
        if(shift.call(c).id!==1||c[0].id!==2||c.length!==1||traps!==0)throw 'prototype observed';
        var single=[9];Object.defineProperty(single,'0',{writable:false});
        if(shift.call(single)!==9||single.length!==0)throw 'readonly last source';
        count
    "#,
        "14",
    );
}

#[test]
fn array_shift_failures_preserve_ordered_partial_mutations() {
    check(
        r#"
        function state(a){var out=[];for(var k=0;k<a.length;k++)out.push(Object.hasOwn(a,k)?a[k]:'hole');return out.join(',');}
        function fail(a,expected){var error='';try{Array.prototype.shift.call(a)}catch(e){error=e.name}
            if(error!=='TypeError'||state(a)!==expected)throw 'wrong failure '+error+':'+state(a);}
        var a=[1,2,3];Object.defineProperty(a,'0',{writable:false});fail(a,'1,2,3');
        a=[1,2,3];Object.defineProperty(a,'1',{writable:false});fail(a,'2,2,3');
        a=[1,2,3];Object.defineProperty(a,'2',{configurable:false});fail(a,'2,3,3');
        a=[1,2,3];Object.defineProperty(a,'length',{writable:false});fail(a,'2,3,hole');
        a=Object.freeze([1,2,3]);fail(a,'1,2,3');
        a=Object.seal([1,2,3]);fail(a,'2,3,3');
        a=Object.preventExtensions([,2,3]);fail(a,'hole,2,3');
        a=[];Object.defineProperty(a,'length',{writable:false});fail(a,'');
        fail(Object.freeze([]),'');
        var b=Object.preventExtensions([1,2,3]);
        if(Array.prototype.shift.call(b)!==1||state(b)!=='2,3'||b.length!==2)throw 'nonextensible dense';
        'ok'
    "#,
        "ok",
    );
}

#[test]
fn array_shift_proxy_traps_and_abrupt_reads_keep_specification_order() {
    check(
        r#"
        var trace=[],target=[1,,3];
        var p=new Proxy(target,{
            get(t,k,r){trace.push('g:'+k);return Reflect.get(t,k,r)},
            has(t,k){trace.push('h:'+k);return Reflect.has(t,k)},
            set(t,k,v,r){trace.push('s:'+k+'='+v);return Reflect.set(t,k,v,r)},
            deleteProperty(t,k){trace.push('d:'+k);return Reflect.deleteProperty(t,k)},
            getOwnPropertyDescriptor(t,k){trace.push('o:'+k);return Reflect.getOwnPropertyDescriptor(t,k)},
            defineProperty(t,k,d){trace.push('D:'+k);return Reflect.defineProperty(t,k,d)}
        });
        if(Array.prototype.shift.call(p)!==1)throw 'proxy return';
        if(trace.join('|')!=='g:length|g:0|h:1|d:0|h:2|g:2|s:1=3|o:1|D:1|d:2|s:length=2|o:length|D:length')
            throw 'proxy order '+trace.join('|');
        if(target.length!==2||Object.hasOwn(target,0)||target[1]!==3)throw 'proxy state';
        trace=[];target=[1,2];var token={};
        p=new Proxy(target,{
            get(t,k,r){trace.push('g:'+k);if(k==='1')throw token;return Reflect.get(t,k,r)},
            has(t,k){trace.push('h:'+k);return Reflect.has(t,k)},
            set(){throw 'unexpected write'}
        });
        var caught=false;try{Array.prototype.shift.call(p)}catch(e){caught=e===token}
        if(!caught||trace.join('|')!=='g:length|g:0|h:1|g:1'||target.join(',')!=='1,2')throw 'abrupt proxy';
        var rev=Proxy.revocable([1,2],{});rev.revoke();caught=false;
        try{Array.prototype.shift.call(rev.proxy)}catch(e){caught=e instanceof TypeError}
        if(!caught)throw 'revoked proxy';
        'ok'
    "#,
        "ok",
    );
}

#[test]
fn array_shift_getters_gc_and_inherited_holes_keep_order_and_receiver() {
    check(
        r#"
        var shift=Array.prototype.shift,trace=[],token={},stored,a=[1,2,3];
        Object.defineProperty(a,'0',{
            get(){trace.push('get0');collectShiftTest();a[2]=9;return token},
            set(v){trace.push('set0:'+v);stored=v;collectShiftTest()},configurable:true
        });
        if(shift.call(a)!==token||stored!==2||a[1]!==9||a.length!==2||Object.hasOwn(a,2))throw 'getter result';
        if(trace.join('|')!=='get0|set0:2')throw 'getter order';
        trace=[];a=[1,,3];var proto=Object.create(Array.prototype);
        Object.defineProperty(proto,'1',{
            get(){if(this!==a)throw 'get receiver';trace.push('get1');collectShiftTest();return 7},
            set(v){if(this!==a)throw 'set receiver';trace.push('set1:'+v);collectShiftTest()},configurable:true
        });
        Object.setPrototypeOf(a,proto);
        if(shift.call(a)!==1||a[0]!==7||a.length!==2||Object.hasOwn(a,1)||Object.hasOwn(a,2))throw 'inherited result';
        if(trace.join('|')!=='get1|set1:3')throw 'inherited order';
        a=[1,2,3];trace=[];
        Object.defineProperty(a,'0',{get(){trace.push('shrink');a.length=1;collectShiftTest();return token},configurable:true});
        if(shift.call(a)!==token||a.length!==2||Object.hasOwn(a,0)||Object.hasOwn(a,1))throw 'captured length';
        if(trace.join('|')!=='shrink')throw 'repeat first getter';
        a=[1,2,3];trace=[];
        Object.defineProperty(a,'1',{get(){trace.push('throw1');collectShiftTest();throw token},configurable:true});
        var caught=false;try{shift.call(a)}catch(e){caught=e===token}
        if(!caught||a[0]!==1||a[2]!==3||a.length!==3||trace.join('|')!=='throw1')throw 'getter abrupt';
        'ok'
    "#,
        "ok",
    );
}

#[test]
fn array_shift_generic_length_conversion_and_string_receivers_are_unchanged() {
    check(
        r#"
        var trace=[],obj={0:'first',1:'second',2:'third'};
        Object.defineProperty(obj,'length',{
            get(){trace.push('length');return {valueOf(){trace.push('number');collectShiftTest();obj[1]='changed';return 2.9}}},
            set(v){trace.push('setlength:'+v)},configurable:true
        });
        if(Array.prototype.shift.call(obj)!=='first'||obj[0]!=='changed'||Object.hasOwn(obj,1)||obj[2]!=='third')throw 'generic';
        if(trace.join('|')!=='length|number|setlength:1')throw 'length ordering';
        var errors=0;
        for(var value of [null,undefined,'ab',Object('ab')]){
            try{Array.prototype.shift.call(value)}catch(e){if(e instanceof TypeError)errors++}
        }
        if(errors!==4)throw 'ToObject or String exotic';
        var empty={length:-1},seen;
        Object.defineProperty(empty,'length',{get(){return -1},set(v){seen=v}});
        if(Array.prototype.shift.call(empty)!==undefined||!Object.is(seen,0))throw 'zero length set';
        'ok'
    "#,
        "ok",
    );
}

#[test]
fn array_shift_warm_calls_keep_named_symbols_owners_and_old_to_young_edges() {
    check(
        r#"
        function consume(a){return a.shift();}
        function scan(a){var sum=0;for(var k=0;k<a.length;k++)sum+=a[k].id;return sum;}
        var key=Symbol('owner'),a=[],total=0;
        a.named={id:91};a[key]={id:92};
        for(var k=0;k<16;k++)a.push({id:k});
        collectShiftTest();
        for(var round=0;round<100;round++){
            var item=consume(a);
            if(item.id!==round)throw 'queue order';
            total+=item.id;a.push({id:round+16});
            if(scan(a)!==16*round+136)throw 'shift/append values';
            if(round%3===0)collectShiftYoungTest();
            if(round%7===0)collectShiftTest();
            if(a.named.id!==91||a[key].id!==92||a.length!==16)throw 'owners';
        }
        collectShiftTest();
        total
    "#,
        "4950",
    );
}
