//! ECMA-262 e28783d5: IteratorStepValue, IteratorClose, CreateIteratorResultObject,
//! %ArrayIteratorPrototype%.next and String/Map/Set iterator state transitions.
use crate::{bytecode::Tier, value::Value, Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("iterator fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn global(engine: &mut Engine, name: &str) -> Value {
    let global = Value::Obj(engine.interp.global.clone());
    engine
        .interp
        .get_member_key(&global, name)
        .unwrap_or_else(|_| panic!("global {name}"))
}

#[test]
fn iterator_results_remove_wrapper_allocations_but_public_next_keeps_identity() {
    for expression in [
        "[1,2,3].values()",
        "'abc'[Symbol.iterator]()",
        "new Map([[1,10],[2,20],[3,30]]).values()",
        "new Set([1,2,3]).values()",
    ] {
        let mut engine = Engine::new();
        eval(&mut engine, &format!("var iterator={expression};"));
        let iterator = global(&mut engine, "iterator");
        let next = engine
            .interp
            .get_member_key(&iterator, "next")
            .unwrap_or_else(|_| panic!("next"));
        let before = crate::value::heap_allocated_objects(&engine.interp.gc_heap);
        for _ in 0..3 {
            assert!(engine
                .interp
                .iterator_step(&iterator, &next)
                .unwrap_or_else(|_| panic!("iterator step"))
                .is_some());
        }
        assert!(engine
            .interp
            .iterator_step(&iterator, &next)
            .unwrap_or_else(|_| panic!("iterator exhaustion"))
            .is_none());
        let allocated = crate::value::heap_allocated_objects(&engine.interp.gc_heap) - before;
        assert_eq!(
            allocated,
            if std::env::var("TRUST_JS_ITERATOR_RESULTS").as_deref() == Ok("0") {
                4
            } else {
                0
            },
            "{expression}"
        );
        // Exhausted calls must still return fresh, ordinary observable objects.
        let first = engine
            .interp
            .call(next.clone(), iterator.clone(), &[])
            .unwrap_or_else(|_| panic!("public next"));
        let second = engine
            .interp
            .call(next, iterator, &[])
            .unwrap_or_else(|_| panic!("public next"));
        assert!(!crate::value::Gc::ptr_eq(
            first.as_obj().unwrap(),
            second.as_obj().unwrap()
        ));
        assert_eq!(
            crate::value::heap_allocated_objects(&engine.interp.gc_heap) - before,
            allocated + 2
        );
    }
}

#[test]
fn iterator_results_preserve_reentrancy_captured_next_close_and_custom_results() {
    let source = r#"
    (function(){
        var trace=[],a=[1,2,3],it=a.values();
        Object.defineProperty(a,'0',{get:function(){trace.push(it.next().value);return 10}});
        var iterable={[Symbol.iterator]:function(){return it}};
        var proto=Object.getPrototypeOf(it), saved=proto.next;
        proto.return=function(){trace.push(this===it);return {done:true}};
        for(var value of iterable){
            trace.push(value);
            proto.next=function(){throw 'must use captured next'};
            break;
        }
        proto.next=saved;delete proto.return;
        trace.push(it.next().value,it.next().done);
        var custom={ [Symbol.iterator]:function(){return this}, next:function(){
            return {get done(){trace.push('done');return false},get value(){trace.push('value');return 7}};
        },return:function(){trace.push('closed');return {}}};
        for(var x of custom){trace.push(x);break}
        var hooked=new Proxy(saved,{apply:function(target,receiver,args){trace.push('apply');return {value:5,done:true}}});
        var obj={[Symbol.iterator]:function(){return {next:hooked}}};
        for(var ignored of obj){throw 'done was ignored'}
        var keys=[0];Object.defineProperty(keys,'0',{get:function(){throw 'keys read element'}});
        trace.push(keys.keys().next().value);
        return trace.join('|');
    })()
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, source),
            "2|10|true|3|true|done|value|7|closed|apply|0",
            "{tier:?}"
        );
    }
}

#[test]
fn iterator_results_keep_unicode_live_collections_and_growth() {
    let source = r#"
    (function(){
        var chars=[]; for(var ch of 'A\uD83D\uDE00\uD800Z')chars.push(ch.length);
        var a=[1],seen=[];for(var x of a){seen.push(x);if(x===1)a.push(2)}
        var m=new Map([[1,10],[2,20]]),pairs=[];
        for(var pair of m){pairs.push(pair.join(':'));if(pair[0]===1){m.delete(2);m.set(3,30)}}
        var s=new Set([1,2]),set=[];for(var x of s){set.push(x);if(x===1){s.delete(2);s.add(3)}}
        var it=a.values();it.next();it.next();var done=it.next().done;a.push(4);
        return [chars.join(','),seen.join(','),pairs.join(','),set.join(','),done,it.next().done].join('|');
    })()
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, source),
            "1,2,1,1|1,2|1:10,3:30|1,3|true|true",
            "{tier:?}"
        );
    }
}

#[test]
fn iterator_results_builtin_consumers_preserve_mapping_close_and_getter_order() {
    let source = r#"
    (function(){
        var trace=[],it=[1,2,3].values();
        it.return=function(){trace.push(this===it?'close':'wrong receiver');return {}};
        try { Array.from(it,function(v,k){
            trace.push(v+':'+k);
            it.next=function(){throw 'next was re-read'};
            if(k===1)throw 19;
            return v;
        }); } catch(e) { trace.push(e); }
        var s=new Set([2,4,6]);
        trace.push(Array.from(s.values(),function(v){return v+1}).join(','));
        trace.push(s.values().reduce(function(a,v){return a+v},0));
        var helper=[7,8].values();
        helper.return=function(){trace.push('helper close');return {}};
        trace.push(helper.some(function(v){return v===7}));
        var calls=0,custom={ [Symbol.iterator]:function(){return this},get next(){
            trace.push('next');return function(){
                var done=calls++>0;
                return {get done(){trace.push('done'+done);return done},
                    get value(){if(done)throw 'exhausted value read';trace.push('value');return 11}};
            };
        }};
        trace.push(Array.from(custom).join(','));
        return trace.join('|');
    })()
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, source),
            "1:0|2:1|close|19|3,5,7|12|helper close|true|next|donefalse|value|donetrue|11",
            "{tier:?}"
        );
    }
}

#[test]
fn iterator_results_keep_realm_errors_and_new_target_after_nested_calls() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
        (function(){
            var other=$262.createRealm().global;
            var next=other.Array.prototype.values.call([]).next;
            var iterable={[Symbol.iterator]:function(){return {next:next}}};
            var realm=false;try{for(var x of iterable){}}catch(e){realm=e instanceof other.TypeError}
            function C(){
                var a=[1],seen=false;
                Object.defineProperty(a,'0',{get:function(){seen=new.target===undefined;return 9}});
                for(var x of a){this.v=x}
                this.valid=seen && new.target===C;
            }
            var c=new C;return [realm,c.v,c.valid].join('|');
        })()
        "#
            ),
            "true|9|true",
            "{tier:?}"
        );
    }
}

#[test]
fn iterator_results_private_slots_survive_freeze_and_reject_forged_brands() {
    let source = r#"
    (function(){
        var trace=[],a=[3,5].values(),s='A\uD83D\uDE00'[Symbol.iterator]();
        trace.push(Reflect.ownKeys(a).length,Reflect.ownKeys(s).length);
        a.__ai_index=100;a.__ai_target=[];a.__ai_kind=1;
        s.__si_index=100;s.__si_str='wrong';
        Object.freeze(a);Object.freeze(s);
        trace.push(a.next().value,a.next().value,a.next().done);
        trace.push(s.next().value,s.next().value.length,s.next().done);
        var anext=[].values().next,snext=''[Symbol.iterator]().next;
        for(var pair of [
            [anext,{__ai_target:[],__ai_index:0,__ai_kind:0}],
            [snext,{__si_str:'forged',__si_index:0}],
            [anext,new Proxy([].values(),{})],
            [snext,new Proxy(''[Symbol.iterator](),{})],
            [snext,TypeError.prototype]
        ]) {try{pair[0].call(pair[1]);trace.push(false)}catch(e){trace.push(e instanceof TypeError)}}
        return trace.join('|');
    })()
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, source),
            "0|0|3|5|true|A|2|true|true|true|true|true|true",
            "{tier:?}"
        );
    }
}

#[test]
fn iterator_results_private_target_is_traced_and_collectable() {
    let mut engine = Engine::new();
    eval(
        &mut engine,
        "var target=[{answer:42}];target.push(target);var iterator=target.values();target=null;",
    );
    // Strong handles live only in this helper frame (gc-heap scans the stack).
    let target = crate::gc_test::in_frame(|| {
        let iterator = global(&mut engine, "iterator");
        let object = iterator.as_obj().unwrap().borrow();
        let crate::value::Exotic::ArrayIterator(state) = &object.exotic else {
            panic!("private iterator slots")
        };
        crate::value::Gc::downgrade(state.target.as_obj().unwrap())
    });
    engine.interp.gc_collect();
    assert!(crate::gc_test::alive(&target));
    assert_eq!(eval(&mut engine, "iterator.next().value.answer"), "42");
    eval(&mut engine, "iterator=null;");
    engine.interp.gc_collect();
    assert!(
        !crate::gc_test::alive(&target),
        "unreachable cycle retained"
    );
}

#[test]
fn array_for_of_steps_keep_holes_getters_growth_and_close() {
    // Expected values from Node v24.
    let source = r#"(function(){
        var log=[],r=[];
        function each(it){var o=[];for(const x of it)o.push(x);return o.join(',')}
        function pairs(it){var o=[];for(const [k,v] of it)o.push(k+':'+v);return o.join(',')}
        var AIP=Object.getPrototypeOf([].values());
        r.push(each([5,,7]),each([5,,7].keys()),pairs(['a','b'].entries()),pairs(new Map([[1,2],[3,4]])),pairs(new Set(['x','y']).entries()),each(new Set([1,2,1])),each('hé😀'));
        Array.prototype[1]='H';r.push(each([5,,7]),pairs([5,,7].entries()));delete Array.prototype[1];
        var a=[1,2,3];Object.defineProperty(a,1,{get(){log.push('g');return 'G'}});r.push(each(a),pairs(a.entries()),log.splice(0).join());
        a=[1,2];r.push((function(){var o=[];for(const x of a){o.push(x);if(a.length<5)a.push(x*10)}return o.join(',')})());
        a=[1,2,3,4];r.push((function(){var o=[];for(const x of a){o.push(x);a.length=2}return o.join(',')})());
        var m=new Map([[1,1]]);r.push((function(){var o=[];for(const [k] of m){o.push(k);if(k<4)m.set(k+1,1)}return o.join(',')})());
        AIP.return=function(){log.push('closed');return {}};
        r.push((function(){for(const x of [1,2,3])if(x===2)break;try{for(const x of [1,2])throw 0}catch(e){}return log.splice(0).join()})());
        delete AIP.return;
        var it=[1,2,3][Symbol.iterator]();it.return=function(){log.push('own');return {}};
        var iterable={[Symbol.iterator](){return it}};for(const x of iterable)break;r.push(log.splice(0).join());
        var t=Array.prototype.values.call({length:2,0:'x',1:'y'});r.push(each({[Symbol.iterator]:()=>t}));
        var u=new Uint8Array([1,2]);r.push(each(u),pairs(u.entries()));
        var next=AIP.next;AIP.next=function(){var x=next.call(this);if(!x.done)x.value='n'+x.value;return x};r.push(each([1,2]));AIP.next=next;
        r.push([...[1,,3]].join(),Array.from(new Map([[1,2]])).join(';'),[...'ab'].join());
        return r.join('|');
    })()"#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, source),
            "5,,7|0,1,2|0:a,1:b|1:2,3:4|x:x,y:y|1,2|h,\u{e9},\u{1F600}|5,H,7|0:5,1:H,2:7|1,G,3|\
             0:1,1:G,2:3|g,g|1,2,10,20,100|1,2|1,2,3,4|closed,closed|own|x,y|1,2|0:1,1:2|n1,n2|\
             1,,3|1,2|a,b",
            "{tier:?}"
        );
    }
}
