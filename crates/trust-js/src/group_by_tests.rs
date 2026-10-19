//! ECMA-262 e28783d5 (local official snapshot, 2026-09-06): GroupBy,
//! AddValueToKeyedGroup, IteratorStepValue, IteratorClose and ToPropertyKey.
//! Expected results verified with Node v24.21.0; exercised in all three TRust JS tiers.

use crate::{bytecode::Tier, value::Value, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine
            .interp
            .def_method(&engine.interp.global, "collectGroupByTest", 0, |i, _, _| {
                i.gc_collect();
                Ok(Value::Undefined)
            });
        match engine.eval(source, false).expect("groupBy fixture parses") {
            Completion::Value(actual) => assert_eq!(actual, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        engine.interp.gc_collect();
    }
}

#[test]
fn group_by_interleaves_iterator_callback_and_key_conversion() {
    check(
        r#"
        var out=[];
        for(var owner of [Object,Map]) {
            var events=[],index=0;
            var iterable={ [Symbol.iterator]() {
                events.push('open');
                return {get next() {
                    events.push('next-get');
                    return function() {
                        events.push('next:'+index);
                        if(index===3)return {get done(){events.push('done');return true},get value(){throw 'done value'}};
                        var value=index++;
                        return {get done(){events.push('done');return false},get value(){events.push('value:'+value);return value}};
                    };
                }};
            }};
            var result=owner.groupBy(iterable,function(value,k) {
                'use strict';
                if(this!==undefined||value!==k)throw 'callback arguments';
                events.push('callback:'+k);
                if(owner===Map)return k%2;
                return {[Symbol.toPrimitive](hint){events.push('key:'+k+':'+hint);return k%2}};
            });
            out.push(events.join(','),owner===Map?Array.from(result.values()).map(a=>a.join(':')).join(';'):Object.values(result).map(a=>a.join(':')).join(';'));
        }
        out.join('|')
        "#,
        "open,next-get,next:0,done,value:0,callback:0,key:0:string,next:1,done,value:1,callback:1,key:1:string,next:2,done,value:2,callback:2,key:2:string,next:3,done|0:2;1|open,next-get,next:0,done,value:0,callback:0,next:1,done,value:1,callback:1,next:2,done,value:2,callback:2,next:3,done|0:2;1",
    );
}

#[test]
fn group_by_observes_live_array_changes_and_cached_next() {
    check(
        r#"
        var out=[];
        for(var owner of [Object,Map]) {
            var source=[1,2],seen=[];
            owner.groupBy(source,function(value,k){seen.push(value);if(k===0){source[1]=20;source.push(30)}return 'x'});
            out.push(seen.join(','));
            source=[1,2,3];seen=[];
            owner.groupBy(source,function(value,k){seen.push(value);if(k===0)source.length=1;return 'x'});
            out.push(seen.join(','));
            source=[1,,3];seen=[];
            owner.groupBy(source,function(value,k){seen.push(value);if(k===0)Object.defineProperty(source,1,{get(){seen.push('get');return 9}});return 'x'});
            out.push(seen.join(','));
            var position=0,iter={next(){return position<2?{value:position++,done:false}:{done:true}}};
            seen=[];
            owner.groupBy({[Symbol.iterator](){return iter}},function(value,k){seen.push(value);iter.next=function(){throw 'changed next'};return 'x'});
            out.push(seen.join(','));
        }
        out.join('|')
        "#,
        "1,20,30|1|1,get,9,3|0,1|1,20,30|1|1,get,9,3|0,1",
    );
}

#[test]
fn group_by_closes_callback_and_key_conversion_throws_only() {
    check(
        r#"
        var out=[];
        for(var owner of [Object,Map]) {
            for(var mode of ['get','call','primitive','noncallable','null']) {
                var events=[],failure={},iter={next(){events.push('next');return {value:1,done:false}}};
                Object.defineProperty(iter,'return',{get(){events.push('return-get');if(mode==='get')throw 'close';if(mode==='null')return null;if(mode==='noncallable')return 1;return function(){events.push('return-call');if(mode==='call')throw 'close';return mode==='primitive'?1:{}}}});
                try{owner.groupBy({[Symbol.iterator](){return iter}},()=>{events.push('callback');throw failure})}catch(e){out.push(e===failure,events.join(','))}
            }
        }
        var events=[],failure={},iter={next(){events.push('next');return {value:1,done:false}},return(){events.push('return');throw 'close'}};
        try{Object.groupBy({[Symbol.iterator](){return iter}},()=>({[Symbol.toPrimitive](){events.push('key');throw failure}}))}catch(e){out.push(e===failure,events.join(','))}
        out.join('|')
        "#,
        "true|next,callback,return-get|true|next,callback,return-get,return-call|true|next,callback,return-get,return-call|true|next,callback,return-get|true|next,callback,return-get|true|next,callback,return-get|true|next,callback,return-get,return-call|true|next,callback,return-get,return-call|true|next,callback,return-get|true|next,callback,return-get|true|next,key,return",
    );
}

#[test]
fn group_by_does_not_close_iterator_step_or_open_throws() {
    check(
        r#"
        var out=[];
        for(var owner of [Object,Map])for(var mode of ['next','done','value','primitive','noncallable','next-get','iterator']) {
            var failure={},closed=0,callbacks=0,iter={return(){closed++;return {}}};
            Object.defineProperty(iter,'next',{get(){if(mode==='next-get')throw failure;if(mode==='noncallable')return 1;return function(){if(mode==='next')throw failure;if(mode==='primitive')return 1;return {get done(){if(mode==='done')throw failure;return false},get value(){if(mode==='value')throw failure;return 1}}}}});
            var items={[Symbol.iterator](){if(mode==='iterator')return 1;return iter}};
            try{owner.groupBy(items,()=>{callbacks++;throw 'unexpected'})}catch(e){out.push(mode+':'+(e===failure?'same':e instanceof TypeError)+':'+closed+':'+callbacks)}
        }
        var opened=0,items={[Symbol.iterator](){opened++;throw 'unexpected'}};
        for(var owner of [Object,Map])try{owner.groupBy(items,null)}catch(e){out.push(e instanceof TypeError,opened)}
        out.join('|')
        "#,
        "next:same:0:0|done:same:0:0|value:same:0:0|primitive:true:0:0|noncallable:true:0:0|next-get:same:0:0|iterator:true:0:0|next:same:0:0|done:same:0:0|value:same:0:0|primitive:true:0:0|noncallable:true:0:0|next-get:same:0:0|iterator:true:0:0|true|0|true|0",
    );
}

#[test]
fn group_by_preserves_property_keys_map_identity_and_group_order() {
    check(
        r#"
        var sym=Symbol('key'),keys=['__proto__','constructor','2','1',sym,'\0symbol:1','__proto__',sym];
        var result=Object.groupBy(keys,(x,k)=>({[Symbol.toPrimitive](){return x}}));
        var out=[Object.getPrototypeOf(result)===null,Reflect.ownKeys(result).map(x=>typeof x==='symbol'?'symbol':x).join(','),result.__proto__.length,result[sym].length];
        var desc=Object.getOwnPropertyDescriptor(result,'__proto__');out.push(desc.writable,desc.enumerable,desc.configurable);
        var one={},two={},mapkeys=[one,two,one,NaN,Number('bad'),-0,0,1n,1,'1'];
        var map=Map.groupBy(mapkeys,x=>x);
        out.push(map.size,map.get(one).length,map.get(two).length,map.get(NaN).length,map.get(0).length,1/Array.from(map.keys())[3],map.get(1n).length,map.get(1).length,map.get('1').length);
        var order=Map.groupBy(['b','a','b'],x=>x);out.push(Array.from(order.keys()).join(','),order.get('b').join(','));
        out.join('|')
        "#,
        "true|1,2,__proto__,constructor,\0symbol:1,symbol|2|2|true|true|true|7|2|1|2|2|Infinity|1|1|1|b,a|b,b",
    );
}

#[test]
fn group_by_map_hash_collisions_preserve_distinct_groups_and_order() {
    // This Number and undefined have the same actual digest. Prove the fixture still
    // exercises a collision if the engine's hash implementation changes.
    let collision = f64::from_bits(0xbe60_db93_9105_4a88);
    assert_eq!(
        crate::ordered_collection::key_hash(&Value::Undefined),
        crate::ordered_collection::key_hash(&Value::Num(collision)),
    );
    check(
        r#"
        var bits=new DataView(new ArrayBuffer(8));
        bits.setUint32(0,0x91054a88,true);bits.setUint32(4,0xbe60db93,true);
        var k=bits.getFloat64(0,true);
        var items=[
            {key:undefined,label:'u0'},{key:k,label:'n0'},
            {key:NaN,label:'nan0'},{key:-0,label:'z0'},
            {key:k,label:'n1'},{key:undefined,label:'u1'},
            {key:0,label:'z1'},{key:Number('bad'),label:'nan1'},
            {key:k,label:'n2'},{key:null,label:'null'}
        ];
        var groups=Map.groupBy(items,function(value){collectGroupByTest();return value.key});
        var keys=Array.from(groups.keys());
        [groups.size,keys[0]===undefined,Object.is(keys[1],k),Number.isNaN(keys[2]),
         1/keys[3],keys[4]===null,Array.from(groups.values()).map(a=>a.map(v=>v.label).join(',')).join(';')].join('|')
        "#,
        "5|true|true|true|Infinity|true|u0,u1;n0,n1,n2;nan0,nan1;z0,z1;null",
    );
}

#[test]
fn group_by_retains_temporary_objects_and_symbol_keys_across_gc() {
    check(
        r#"
        var objectGroups=Object.groupBy({*[Symbol.iterator](){for(var k=0;k<8;k++)yield {value:k}}},function(value,k){collectGroupByTest();return Symbol('key'+k)});
        collectGroupByTest();
        var symbols=Object.getOwnPropertySymbols(objectGroups);
        var out=[symbols.map(s=>s.description).join(','),symbols.map(s=>objectGroups[s][0].value).join(',')];
        var map=Map.groupBy({*[Symbol.iterator](){for(var k=0;k<8;k++)yield {value:k}}},function(value){collectGroupByTest();return {key:value.value}});
        collectGroupByTest();
        out.push(Array.from(map.entries()).map(pair=>pair[0].key+':'+pair[1][0].value).join(','));
        out.join('|')
        "#,
        "key0,key1,key2,key3,key4,key5,key6,key7|0,1,2,3,4,5,6,7|0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7",
    );
}

#[test]
fn group_by_cached_callbacks_keep_bound_and_live_proxy_calls() {
    check(
        r#"
        var out=[];
        for(var owner of [Object,Map]) {
            var receiver={},calls=0,source=[];
            for(var n=0;n<128;n++)source.push(n);
            function callback(prefix,value,index) {
                'use strict';
                if(this!==receiver||arguments.length!==3||value!==index)throw 'bound arguments';
                calls++;return prefix+(value%2);
            }
            var bound=owner.groupBy(source,callback.bind(receiver,'key'));
            out.push(calls,(owner===Map?bound.get('key0'):bound.key0).length);
            var position=0,nextCalls=0,proxyCalls=0,iter={};
            function next(){'use strict';if(this!==iter)throw 'next receiver';nextCalls++;return position<128?{value:position++,done:false}:{done:true}}
            iter.next=new Proxy(next,{apply(t,s,a){return Reflect.apply(t,s,a)}}).bind(iter);
            var handler={apply(t,s,a){
                if(s!==undefined||a.length!==2)throw 'proxy arguments';
                proxyCalls++;
                var result=Reflect.apply(t,s,a);
                if(a[1]===63)handler.apply=function(){proxyCalls++;return 'later'};
                return result;
            }};
            var proxy=new Proxy(function(value){return 'key'+(value%2)},handler);
            var grouped=owner.groupBy({[Symbol.iterator](){return iter}},proxy);
            out.push(nextCalls,proxyCalls,(owner===Map?grouped.get('key0'):grouped.key0).length,(owner===Map?grouped.get('key1'):grouped.key1).length,(owner===Map?grouped.get('later'):grouped.later).length);
            position=0;nextCalls=0;proxyCalls=0;var closed=0;
            iter.next=next;iter.return=function(){closed++;throw 'close'};
            var rev=Proxy.revocable(function(value,k){proxyCalls++;if(k===64)rev.revoke();return 'x'},{});
            try{owner.groupBy({[Symbol.iterator](){return iter}},rev.proxy)}catch(e){out.push(e instanceof TypeError,proxyCalls,nextCalls,closed)}
        }
        out.join('|')
        "#,
        "128|64|129|128|32|32|64|true|65|66|1|128|64|129|128|32|32|64|true|65|66|1",
    );
}

#[test]
fn group_by_cached_callbacks_and_next_keep_their_running_realms() {
    check(
        r#"
        var other=$262.createRealm().global,out=[];
        other.eval('var index=0,closed=0;var iterator={next:function(){if(this!==iterator)throw Error("next receiver");return index<128?{value:{n:index++},done:false}:{done:true}},return:function(){closed++;throw "close"}};var items={[Symbol.iterator]:function(){return iterator}}');
        var failing=other.eval('(function(value,k){if(this!==globalThis||arguments.length!==2)throw Error("callback realm");if(k===96)throw new TypeError("foreign");return value.n%2})');
        var succeeding=other.eval('(function(value,k){if(this!==globalThis||arguments.length!==2||value.n!==k)throw Error("callback realm");return value.n%2})');
        for(var owner of [Object,Map]) {
            other.eval('index=0;closed=0');
            try{owner.groupBy(other.items,failing)}catch(e){out.push(e instanceof other.TypeError,e instanceof TypeError,other.closed)}
            other.eval('index=0');
            var groups=owner.groupBy(other.items,succeeding),first=owner===Map?groups.get(0):groups[0];
            out.push(first instanceof Array,first instanceof other.Array,first.length,first[0] instanceof other.Object,first[0] instanceof Object,first[63].n);
            if(owner===Map)out.push(groups instanceof Map,groups instanceof other.Map);
            else out.push(Object.getPrototypeOf(groups)===null);
        }
        out.join('|')
        "#,
        "true|false|1|true|false|64|true|false|126|true|true|false|1|true|false|64|true|false|126|true|false",
    );
}

#[test]
fn group_by_cached_callees_keep_then_release_their_final_owners() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for owner in ["Object", "Map"] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            engine
                .interp
                .def_method(&engine.interp.global, "collectGroupByTest", 0, |i, _, _| {
                    i.gc_collect();
                    Ok(Value::Undefined)
                });
            let setup = r#"
                var release=false,position=0,source=[];
                for(var k=0;k<128;k++)source.push({n:k});
                var callback=function(value,k){
                    if(release){callback=null;collectGroupByTest()}
                    return value.n%2;
                };
                var next=function(){
                    if(release){next=null;iterator.next=null;collectGroupByTest()}
                    return position<source.length?{value:source[position++],done:false}:{done:true};
                };
                var iterator={next:next},items={[Symbol.iterator](){return iterator}};
                for(var warm=0;warm<150;warm++){callback(source[0],0);next()}
                position=0;'ready'
            "#;
            assert!(matches!(engine.eval(setup, false), Ok(Completion::Value(v)) if v == "ready"));
            // Strong handles live only in this helper frame (the collector scans the stack).
            let (weak_callees, weak_fields) = crate::gc_test::in_frame(|| {
                let mut weak_callees = Vec::new();
                let mut weak_fields = Vec::new();
                for name in ["callback", "next"] {
                    let env = engine.interp.global_env.clone();
                    let callee = engine
                        .interp
                        .get_var(name, &env)
                        .unwrap_or_else(|_| panic!("fixture callee {name}"));
                    let object = callee.as_obj().unwrap();
                    weak_callees.push(crate::value::Gc::downgrade(object));
                    let sentinel = engine.interp.new_object();
                    weak_fields.push(crate::value::Gc::downgrade(&sentinel));
                    object.borrow_mut().props.insert(
                        "sentinel",
                        crate::value::Property::plain(Value::Obj(sentinel)),
                    );
                }
                (weak_callees, weak_fields)
            });
            let run = format!(
                "release=true;var result={owner}.groupBy(items,callback);var first={};\
                [first.length,first[63].n,callback===null,next===null].join('|')",
                if owner == "Map" {
                    "result.get(0)"
                } else {
                    "result[0]"
                }
            );
            match engine.eval(&run, false).unwrap() {
                Completion::Value(value) => {
                    assert_eq!(value, "64|126|true|true", "{tier:?}/{owner}")
                }
                Completion::Throw { name, message } => {
                    panic!("{tier:?}/{owner}: {name}: {message}")
                }
            }
            engine.interp.gc_collect();
            for weak in weak_callees {
                assert_eq!(weak.strong_count(), 0, "{tier:?}/{owner} callee retained");
            }
            // Each field was owned only by its callee. Its independent Weak proves the
            // callee destructor ran, rather than just an incorrect zero owner count.
            for weak in weak_fields {
                assert!(
                    !crate::gc_test::alive(&weak),
                    "{tier:?}/{owner} field retained"
                );
            }
            assert!(engine.interp.fn_frames.is_empty());
        }
    }
}
