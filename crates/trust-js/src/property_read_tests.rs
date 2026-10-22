//! ECMA-262 e28783d5: OrdinaryGet, GetValue, Call, OrdinaryCallBindThis,
//! ValidateAndApplyPropertyDescriptor, Proxy [[Get]], and Forbidden Extensions.
use super::*;
use crate::bytecode::{
    IcState, Tier, IC_ABSENT, IC_ACCESSOR, IC_ARR_KEYCHK, IC_EMPTY, PROP_IC_WAYS,
};
use crate::value::ValueView;
use crate::{Completion, Engine};
use std::cell::Cell;
use std::rc::Rc;

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("getter cache fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for threshold in [0, 32] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(threshold);
            engine
                .interp
                .def_method(&engine.interp.global, "collectGetterTest", 0, |i, _, _| {
                    i.gc_collect();
                    Ok(Value::Undefined)
                });
            assert_eq!(
                evaluate(&mut engine, source),
                expected,
                "{tier:?}/{threshold}"
            );
            assert!(engine.interp.fn_frames.is_empty());
            engine.interp.gc_collect();
            assert_eq!(evaluate(&mut engine, "1+2"), "3");
        }
    }
}

fn global_user_callable(engine: &mut Engine, name: &str) -> crate::value::UserFn {
    let env = engine.interp.global_env.clone();
    let value = engine
        .interp
        .get_var(name, &env)
        .unwrap_or_else(|_| panic!("missing global {name}"));
    let object = value
        .as_obj()
        .unwrap_or_else(|| panic!("{name} is not an object"));
    let borrowed = object.borrow();
    let crate::value::Callable::User(user) = &borrowed.call else {
        panic!("{name} is not an ordinary function")
    };
    user.clone()
}

fn has_binding_getter_cache(user: &crate::value::UserCallable) -> bool {
    let cache = user.take_binding_getter_cache();
    let cached = cache.is_some();
    user.put_binding_getter_cache(cache);
    cached
}

#[test]
fn binding_getter_cache_is_per_closure_and_reads_live_values() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
                function makeReader(value) {
                    return { get value() { return value; }, set value(next) { value = next; } };
                }
                var left = makeReader(1), right = makeReader(2);
                var leftGetter = Object.getOwnPropertyDescriptor(left, 'value').get;
                var rightGetter = Object.getOwnPropertyDescriptor(right, 'value').get;
                for (var n = 0; n < 4; n++) { left.value; right.value; }
                [left.value, right.value].join(',')
            "#
            ),
            "1,2",
            "{tier:?}"
        );
        let left = global_user_callable(&mut engine, "leftGetter");
        let right = global_user_callable(&mut engine, "rightGetter");
        assert!(
            has_binding_getter_cache(&left),
            "{tier:?}: left cache filled"
        );
        assert!(
            has_binding_getter_cache(&right),
            "{tier:?}: right cache filled"
        );
        assert!(Rc::ptr_eq(&left.func, &right.func), "{tier:?}: shared AST");
        assert!(
            !std::ptr::eq(
                crate::value::user_fn_ptr(&left),
                crate::value::user_fn_ptr(&right)
            ),
            "{tier:?}: closure state is separate"
        );
        assert_eq!(
            evaluate(
                &mut engine,
                "left.value = 7; right.value = 8; [left.value, right.value].join(',')"
            ),
            "7,8",
            "{tier:?}: in-place binding writes stay live"
        );
    }
}

#[test]
fn binding_getter_cache_re_resolves_after_eval_insert_and_delete() {
    check(
        r#"
        function changeBinding() {
            let delayed = 1;
            function make() {
                var object = { get value() { return delayed; } };
                var before = object.value;
                eval('var delayed = 2');
                var shadow = object.value;
                for (var n = 0; n < 48; n++) eval('var filler' + n + ' = ' + n);
                var afterGrowth = object.value;
                var removed = eval('delete delayed');
                var afterDelete = object.value;
                return [before, shadow, afterGrowth, removed, afterDelete].join('|');
            }
            return make();
        }
        changeBinding()
    "#,
        "1|2|2|true|1",
    );
}

#[test]
fn record_getter_specialization_returns_fresh_live_records() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let before = property_read::TEST_RECORD_GETTERS.with(Cell::get);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            function position(){return {index:this.index,line:this.line,column:this.column}}
            function read(s){return s.state}
            var source={index:4,line:8,column:-0},ok=true;
            Object.defineProperty(source,'state',{get:position,configurable:true});
            for(var n=0;n<160;n++){var r=read(source);ok=ok&&r.index===4&&r.line===8&&Object.is(r.column,-0)}
            var a=read(source),b=read(source);a.index=99;source.index=7;
            ok&&a!==b&&b.index===4&&read(source).index===7&&Object.getPrototypeOf(a)===Object.prototype
        "#
            ),
            "true",
            "{tier:?}"
        );
        let entries = property_read::TEST_RECORD_GETTERS.with(Cell::get) - before;
        if tier == Tier::Jit {
            assert!(
                entries > 100,
                "warm closed getters execute their structural plan"
            );
        } else {
            assert_eq!(entries, 0);
        }
        engine.interp.gc_collect();
        assert_eq!(evaluate(&mut engine, "read(source).index"), "7");
        if tier == Tier::Jit {
            let env = engine.interp.global_env.clone();
            let source = engine
                .interp
                .get_var("source", &env)
                .unwrap_or_else(|_| panic!("source binding"));
            let before = property_read::TEST_RECORD_GETTERS.with(Cell::get);
            engine.interp.gc_tick = GC_CALL_POLL_MASK;
            // A property IC site is PROP_IC_WAYS adjacent cells; a fill rotates older ways.
            let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
            let value = engine
                .interp
                .get_prop_ic(&source, "state", &cache[0])
                .unwrap_or_else(|_| panic!("maintenance getter"));
            assert!(matches!(
                (engine.interp.get_member_key(&value, "index")).map(|__value| (__value).view()),
                Ok(ValueView::Num(7.))
            ));
            assert_eq!(
                property_read::TEST_RECORD_GETTERS.with(Cell::get),
                before,
                "a due collection/metadata poll retains the ordinary Call boundary"
            );
        }
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn record_getter_guard_misses_preserve_observer_frames_and_changes() {
    check(
        r#"
        function position(){return {index:this.index,line:this.line,column:this.column}}
        function read(s){return s.state}
        var source={index:4,line:8,column:9},trace='',caller=false,frames=false;
        Object.defineProperty(source,'state',{get:position,configurable:true});
        for(var warm=0;warm<160;warm++)read(source);
        source.line='wide';var nonNumber=read(source).line;
        Object.defineProperty(source,'line',{get:function lineObserver(){
            trace+='l';caller=position.caller===read;
            var stack=new Error().stack;frames=stack.includes('position')&&stack.includes('read')&&stack.includes('lineObserver');
            return 21;
        },configurable:true});
        var observed=read(source).line;
        Object.defineProperty(source,'column',{get(){trace+='c';throw new Error('bad')},configurable:true});
        var thrown='';try{read(source)}catch(e){thrown=e.message}
        Object.defineProperty(source,'state',{get(){trace+='r';return {index:42}}});
        var replaced=read(source).index;
        [nonNumber,observed,caller,frames,trace,thrown,replaced].join('|')
    "#,
        "wide|21|true|true|llcr|bad|42",
    );
}

#[test]
fn record_getter_specialization_retains_foreign_async_class_and_proxy_calls() {
    check(
        r#"
        function position(){return {index:this.index}}
        function read(s){return s.state}
        var source={index:12};Object.defineProperty(source,'state',{get:position,configurable:true});
        for(var warm=0;warm<160;warm++)read(source);
        var trace='',proxy=new Proxy(source,{get(t,k,r){trace+=k+',';return Reflect.get(t,k,r)}});
        var proxied=read(proxy).index;
        var other=$262.createRealm(),foreign=other.evalScript('(function(){return {index:this.index}})');
        Object.defineProperty(source,'state',{get:foreign,configurable:true});
        for(var warm=0;warm<160;warm++)read(source);
        var result=read(source),foreignProto=Object.getPrototypeOf(result)===other.global.Object.prototype;
        Object.defineProperty(source,'state',{get:async function(){return {index:this.index}},configurable:true});
        var promised=read(source) instanceof Promise;
        Object.defineProperty(source,'state',{get:class Invalid{constructor(){return {index:this.index}}}});
        var classError=false;try{read(source)}catch(e){classError=e instanceof TypeError}
        [proxied,trace,result.index,foreignProto,promised,classError].join('|')
    "#,
        "12|state,index,|12|true|true|true",
    );
}

/// A cached read's owned snapshot is two words, as is its Option, and finishing it yields the
/// snapshotted value after its property is gone. That the snapshot alone keeps its value alive
/// across a collection is checked by
/// `getter_cache_packed_data_snapshot_materializes_lazy_prototype_before_owner_drops`.
#[test]
fn getter_cache_owned_snapshot_stays_two_words() {
    assert_eq!(std::mem::size_of::<CachedGet>(), 16);
    assert_eq!(std::mem::size_of::<Option<CachedGet>>(), 16);
    let mut engine = Engine::new();
    let object = Object::new(None);
    let property = Property::plain(Value::Obj(object.clone()));
    let read = engine
        .interp
        .snapshot_cached_get(&property, "value", 0, 0, false)
        .unwrap();
    drop(property);
    let value = engine
        .interp
        .finish_cached_get(read, &Value::Undefined, None)
        .ok()
        .unwrap();
    assert!(matches!(&(value).view(), ValueView::Obj(actual) if Gc::ptr_eq(actual, &object)));
}

#[test]
fn getter_cache_packed_data_snapshot_materializes_lazy_prototype_before_owner_drops() {
    let mut engine = Engine::new();
    // Strong handles live only in helper frames (the collector scans the stack): the outer one
    // keeps the snapshot, the inner one the function.
    let weak = crate::gc_test::in_frame(|| {
        let (read, weak) = crate::gc_test::in_frame(|| {
            let function = engine
                .interp
                .eval_in_realm(&engine.interp.global_this(), "(function Snapshot(){})")
                .ok()
                .unwrap();
            let read = {
                let object_handle = function.as_obj().unwrap();
                let object = object_handle.borrow();
                engine
                    .interp
                    .snapshot_cached_get(
                        object.props.get("prototype").unwrap(),
                        "prototype",
                        0,
                        0,
                        false,
                    )
                    .unwrap()
            };
            (read, Gc::downgrade(&function.as_obj().unwrap()))
        });
        engine.interp.gc_collect();
        let prototype = read.data().unwrap();
        let constructor = engine
            .interp
            .get_member_key(&prototype, "constructor")
            .ok()
            .unwrap();
        assert!(Gc::ptr_eq(
            &constructor.as_obj().unwrap(),
            &weak.upgrade().unwrap()
        ));
        weak
    });
    engine.interp.gc_collect();
    assert!(!crate::gc_test::alive(&weak));
}

#[test]
fn getter_cache_records_all_live_chain_levels_and_releases_getter_owners() {
    let mut engine = Engine::new();
    for depth in 0..=6 {
        let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
        let holder = Object::new(None);
        // The getter lives only in this helper frame (the collector scans the stack).
        let weak = crate::gc_test::in_frame(|| {
            let getter = engine.interp.new_native_fn(
                "get cached",
                0,
                Rc::new(|_, this, args| {
                    assert!(args.is_empty());
                    let object_handle = this.as_obj().unwrap();
                    let object = object_handle.borrow();
                    Ok(object.props.get("id").unwrap().value())
                }),
                crate::value::NativeCaptures::None,
            );
            let weak = Gc::downgrade(&getter.as_obj().unwrap());
            holder.borrow_mut().props.insert(
                "cached",
                Property::accessor_prop(Some(getter), None, true, true),
            );
            weak
        });
        let mut receiver = holder.clone();
        for _ in 0..depth {
            receiver = Object::new(Some(receiver));
        }
        receiver
            .borrow_mut()
            .props
            .insert("id", Property::plain(Value::Num(19.0)));
        let base = Value::Obj(receiver.clone());
        for _ in 0..3 {
            let value = engine
                .interp
                .get_prop_ic(&base, "cached", &cache[0])
                .ok()
                .unwrap();
            assert!(matches!((value).view(), ValueView::Num(19.0)));
            assert_eq!(
                cache[0].get().depth,
                if depth <= 5 {
                    IC_ACCESSOR | depth
                } else {
                    IC_EMPTY
                }
            );
            assert_eq!(
                weak.strong_count(),
                1,
                "the property is the only getter owner"
            );
        }
        holder
            .borrow_mut()
            .props
            .insert("cached", Property::plain(Value::Num(23.0)));
        engine.interp.gc_collect();
        assert!(
            !crate::gc_test::alive(&weak),
            "layout caches must not pin getter values"
        );
        assert!(matches!(
            (engine.interp.get_prop_ic(&base, "cached", &cache[0])).map(|__value| (__value).view()),
            Ok(ValueView::Num(23.0))
        ));
    }
}

#[test]
fn getter_cache_same_shape_replacement_and_kind_changes_are_live() {
    let mut engine = Engine::new();
    let object = Object::new(None);
    let base = Value::Obj(object.clone());
    let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
    let calls = Rc::new(Cell::new(0));
    let mut shape = None;
    for expected in [2.0, 9.0] {
        let calls = calls.clone();
        let getter = engine.interp.new_native_fn(
            "get cached",
            0,
            Rc::new(move |_, _, _| {
                calls.set(calls.get() + 1);
                Ok(Value::Num(expected))
            }),
            crate::value::NativeCaptures::None,
        );
        object.borrow_mut().props.insert(
            "cached",
            Property::accessor_prop(Some(getter), None, false, true),
        );
        if let Some(shape) = shape {
            assert_eq!(object.borrow().props.shape(), shape);
        } else {
            shape = Some(object.borrow().props.shape());
        }
        for _ in 0..3 {
            assert!(
                matches!((engine.interp.get_prop_ic(&base, "cached", &cache[0])).map(|__value| (__value).view()), Ok(ValueView::Num(n)) if n==expected)
            );
            assert_eq!(cache[0].get().depth, IC_ACCESSOR);
        }
    }
    assert_eq!(calls.get(), 6);
    let shape = object.borrow().props.shape();
    // Clearing the getter through the insert funnel is a class-preserving
    // (accessor -> accessor) replacement, so the shape is retained and the cached
    // accessor state must still observe the live (now missing) getter.
    object
        .borrow_mut()
        .props
        .insert("cached", Property::accessor_prop(None, None, false, true));
    assert_eq!(object.borrow().props.shape(), shape);
    assert!(matches!(
        (engine.interp.get_prop_ic(&base, "cached", &cache[0])).map(|__value| (__value).view()),
        Ok(ValueView::Undefined)
    ));
    assert_eq!(calls.get(), 6);
    object
        .borrow_mut()
        .props
        .insert("cached", Property::plain(Value::Num(31.0)));
    assert!(matches!(
        (engine.interp.get_prop_ic(&base, "cached", &cache[0])).map(|__value| (__value).view()),
        Ok(ValueView::Num(31.0))
    ));
    assert_eq!(cache[0].get().depth, 0);
}

#[test]
fn getter_cache_array_holder_uses_live_key_checks_and_computed_keys() {
    let mut engine = Engine::new();
    evaluate(
        &mut engine,
        r#"
        var arrays=[];
        for(var n=0;n<24;n++){var a=[];for(var k=0;k<n;k++)a.push(k);
            Object.defineProperty(a,'cached',{get:function(){return this.length+1},configurable:true});
            arrays.push(a);}
    "#,
    );
    let global = Value::Obj(engine.interp.global.clone());
    let arrays = engine
        .interp
        .get_member_key(&global, "arrays")
        .ok()
        .unwrap();
    let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
    let key = crate::lstr::LStr::from("cached");
    for n in (0..24).chain((0..24).rev()) {
        let array = engine
            .interp
            .get_member_key(&arrays, &n.to_string())
            .ok()
            .unwrap();
        assert!(
            matches!((engine.interp.get_prop_ic(&array, "cached", &cache[0])).map(|__value| (__value).view()), Ok(ValueView::Num(v)) if v==n as f64+1.0)
        );
        assert_eq!(cache[0].get().depth, IC_ACCESSOR | IC_ARR_KEYCHK);
        assert!(
            matches!((engine.interp.get_computed_property(&array, &key)).map(|__value| (__value).view()), Ok(ValueView::Num(v)) if v==n as f64+1.0)
        );
        let shape = array.as_obj().unwrap().borrow().props.shape();
        assert_eq!(
            engine
                .interp
                .computed_reads
                .lookup(key.as_ptr() as usize, shape)
                .unwrap()
                .depth,
            IC_ACCESSOR | IC_ARR_KEYCHK,
            "computed getters publish a real checked resolution"
        );
        let mut trace = crate::feedback::CurrentPropertyTrace::default();
        assert!(engine
            .interp
            .get_prop_ic_profiled(&array, "cached", &cache[0], &mut trace)
            .is_ok());
        assert_eq!(
            trace.outcome,
            Some(crate::feedback::PropertyOutcome::Accessor)
        );
        assert!(trace.array_key_check);
    }
}

/// A getter whose own computed reads grow the cache that holds its resolution, and which then
/// replaces itself with a data property and collects, still returns its result; the next read
/// finds the data property without calling the getter, and a collection frees the getter.
#[test]
fn getter_cache_reentrant_computed_growth_keeps_owned_result_and_live_replacement() {
    let mut engine = Engine::new();
    let object = Object::new(None);
    let key = crate::lstr::LStr::from("cached");
    let called = Rc::new(Cell::new(0));
    // The getter lives only in this helper frame (the collector scans the stack).
    let weak = crate::gc_test::in_frame(|| {
        let counter = called.clone();
        let getter = engine.interp.new_native_fn(
            "get cached",
            0,
            Rc::new(move |interp, this, _| {
                counter.set(counter.get() + 1);
                // The outer resolution has already been cached. Recursive reads fill
                // and resize that table before this getter returns its owned result.
                let other = Value::Obj(Object::new(None));
                for k in 0..5000 {
                    let transient = crate::lstr::LStr::from(format!("absent{k}"));
                    assert!(matches!(
                        (interp.get_computed_property(&other, &transient))
                            .map(|__value| (__value).view()),
                        Ok(ValueView::Undefined)
                    ));
                }
                assert_eq!(
                    interp.computed_reads.mask as usize + 1,
                    crate::bytecode::ComputedReadCache::MAX_SETS
                );
                this.as_obj()
                    .unwrap()
                    .borrow_mut()
                    .props
                    .insert("cached", Property::plain(Value::Num(47.0)));
                interp.gc_collect();
                Ok(this)
            }),
            crate::value::NativeCaptures::None,
        );
        let weak = Gc::downgrade(&getter.as_obj().unwrap());
        object.borrow_mut().props.insert(
            "cached",
            Property::accessor_prop(Some(getter), None, true, true),
        );
        weak
    });
    let base = Value::Obj(object.clone());
    let result = engine
        .interp
        .get_computed_property(&base, &key)
        .ok()
        .unwrap();
    assert!(Gc::ptr_eq(&result.as_obj().unwrap(), &object));
    assert_eq!(called.get(), 1);
    engine.interp.gc_collect();
    assert!(
        !crate::gc_test::alive(&weak),
        "the replaced getter is released"
    );
    assert!(matches!(
        (engine.interp.get_computed_property(&base, &key)).map(|__value| (__value).view()),
        Ok(ValueView::Num(47.0))
    ));
    assert_eq!(called.get(), 1);
}

#[test]
fn getter_cache_callable_proxy_and_revocation_obey_call_semantics() {
    check(
        r#"
        var o={id:7},calls=0,applies=0,correct=true;
        var target=function(){'use strict';calls++;return this.id};
        var handler={apply(t,receiver,args){applies++;collectGetterTest();
            correct=correct&&t===target&&receiver===o&&args.length===0;
            return Reflect.apply(t,receiver,args);}};
        var revocable=Proxy.revocable(target,handler);
        Object.defineProperty(o,'value',{get:revocable.proxy,configurable:true});
        function named(x){return x.value}function keyed(x,k){return x[k]}
        var sum=0;for(var k=0;k<100;k++)sum+=named(o)+keyed(o,'value');
        revocable.revoke();var throws=0;
        try{named(o)}catch(e){if(e instanceof TypeError)throws++}
        try{keyed(o,'value')}catch(e){if(e instanceof TypeError)throws++}
        [sum,calls,applies,correct,throws].join('|');
    "#,
        "1400|200|200|true|2",
    );
}

#[test]
fn getter_cache_warm_named_computed_and_methods_preserve_receiver_and_effects() {
    check(
        r#"
        var gets=0,calls=0,token={id:7},s=Symbol('cached'),o={id:7};
        Object.defineProperty(o,'value',{get(){gets++;return this===o?token:null},configurable:true});
        Object.defineProperty(o,'method',{get(){gets++;return function(v){'use strict';calls++;return this===o&&v===token}},configurable:true});
        Object.defineProperty(o,s,{get(){gets++;return this===o},configurable:true});
        function named(x){return x.value===token&&x.method(token)}
        function computed(x,k,m){return x[k]===token&&x[m](token)}
        var good=true;
        for(var i=0;i<100;i++){good=named(o)&&computed(o,'value','method')&&o[s]&&good;}
        [good,gets,calls].join('|');
    "#,
        "true|500|200",
    );
}

#[test]
fn getter_cache_reentrancy_replacement_throw_and_gc_keep_exactly_one_call() {
    check(
        r#"
        var reads=0,nested=0,mode=0,token={},o={};
        function named(x){return x.value}
        function keyed(x,k){return x[k]}
        function get(){reads++;collectGetterTest();
            if(mode===1){mode=0;nested+=named(o);return 3;}
            if(mode===2){delete o.value;collectGetterTest();
                Object.defineProperty(o,'value',{get(){reads++;return 9},configurable:true});return token;}
            if(mode===3){collectGetterTest();throw token;}
            return 2;}
        Object.defineProperty(o,'value',{get:get,configurable:true});
        for(var i=0;i<100;i++){named(o);keyed(o,'value');}
        reads=0;mode=1;var a=named(o);mode=3;var caught=false;
        try{keyed(o,'value')}catch(e){caught=e===token}
        mode=2;var b=named(o)===token,c=keyed(o,'value');
        [a,nested,caught,b,c,reads].join('|');
    "#,
        "3|2|true|true|9|5",
    );
}

#[test]
fn getter_cache_live_prototype_swap_proxy_and_inherited_shadowing() {
    check(
        r#"
        var calls=0,proto={get value(){calls++;return this.id+1}},o=Object.create(proto);o.id=3;
        function read(x){return x.value}
        for(var i=0;i<100;i++)read(o);
        calls=0;var a=read(o);Object.setPrototypeOf(o,{get value(){calls++;return this.id+10}});
        var b=read(o);Object.defineProperty(o,'value',{value:90,configurable:true});var c=read(o);delete o.value;
        var trace=[],target={get value(){trace.push('getter');return this.id+20}};
        Object.setPrototypeOf(o,new Proxy(target,{get(t,k,r){trace.push(k+':'+(r===o));return Reflect.get(t,k,r)}}));
        var d=read(o),e=read(o);Object.setPrototypeOf(o,null);var missing=read(o)===undefined;
        [a,b,c,d,e,missing,calls,trace.join(',')].join('|');
    "#,
        "4|13|90|23|23|true|2|value:true,getter,value:true,getter",
    );
}

#[test]
fn getter_cache_primitive_strict_and_sloppy_receivers_remain_distinct() {
    check(
        r#"
        var trace=[],protos=[Number.prototype,String.prototype,Boolean.prototype];
        for(var p of protos){Object.defineProperty(p,'strictCached',{configurable:true,get:function(){'use strict';return typeof this+':'+String(this)}});
            Object.defineProperty(p,'sloppyCached',{configurable:true,get:function(){return typeof this+':'+String(this)}});}
        function read(x){return x.strictCached+'|'+x.sloppyCached}
        var good=true;
        for(var i=0;i<100;i++){good=read(0)==='number:0|object:0'&&read('')==='string:|object:'&&read(false)==='boolean:false|object:false'&&good;}
        good;
    "#,
        "true",
    );
}

#[test]
fn getter_cache_transient_keys_conversion_and_foreign_getter_realm() {
    check(
        r#"
        var calls=0,keys=0,o={get first(){calls++;return 2},get second(){calls++;return 3}};
        function read(x,k){return x[k]}
        var sum=0;
        for(var i=0;i<100;i++){var key={toString(){keys++;collectGetterTest();return keys%2?'first':'second'}};sum+=read(o,key);}
        var other=$262.createRealm(),foreign=other.evalScript('(function(){return this instanceof Object})');
        Object.defineProperty(Number.prototype,'foreignCached',{get:foreign,configurable:true});
        function primitive(x){return x.foreignCached}
        var good=true;for(var i=0;i<100;i++)good=primitive(7)&&good;
        [sum,calls,keys,good].join('|');
    "#,
        "250|100|100|true",
    );
}

#[test]
fn getter_cache_legacy_poison_stays_generic_and_getter_exceptions_propagate() {
    let mut engine = Engine::new();
    evaluate(
        &mut engine,
        "function sloppy(){}; var strict=function(){'use strict';};",
    );
    let global = Value::Obj(engine.interp.global.clone());
    let sloppy = engine
        .interp
        .get_member_key(&global, "sloppy")
        .ok()
        .unwrap();
    let strict = engine
        .interp
        .get_member_key(&global, "strict")
        .ok()
        .unwrap();
    for key in ["arguments", "caller"] {
        let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
        for _ in 0..3 {
            assert!(matches!(
                (engine.interp.get_prop_ic(&sloppy, key, &cache[0]))
                    .map(|__value| (__value).view()),
                Ok(ValueView::Null)
            ));
            assert!(cache.iter().all(|c| c.get().depth == IC_EMPTY));
            assert!(engine.interp.get_prop_ic(&strict, key, &cache[0]).is_err());
        }
    }
}

#[test]
fn primitive_wrapper_proto_data_reads_are_tier_identical() {
    check(
        r#"
        Number.prototype.__flag = 11;
        Boolean.prototype.__flag = 22;
        String.prototype.__flag = 33;
        function viaLocal(v) { return v.__flag; }
        function viaAbsent(v) { return v.__no_such; }
        function viaThis() { 'use strict'; return this.__flag; }
        function fixed(v) { return v.toFixed(1); }
        function boolStr(v) { return v.toString(); }
        function strLen(v) { return v.length; }
        function nullishThrew(v) { try { v.__flag; return false; } catch (e) { return e instanceof TypeError; } }
        function missingThrew(v) { try { v.__nope(); return false; } catch (e) { return e instanceof TypeError; } }
        var other = $262.createRealm();
        var foreignRead = other.evalScript('(function(v) { return v.__flag; })');
        other.evalScript('Number.prototype.__flag = 111');
        var ok = true;
        for (var i = 0; i < 120; i++) {
            ok = viaLocal(1) === 11 && viaLocal(true) === 22 && viaLocal('s') === 33 && ok;
            ok = viaLocal(NaN) === 11 && viaLocal(-0) === 11 && viaLocal(Infinity) === 11
                && viaLocal(false) === 22 && ok;
            ok = viaAbsent(1) === undefined && viaAbsent(false) === undefined
                && viaAbsent('s') === undefined && ok;
            ok = viaThis.call(2) === 11 && viaThis.call(false) === 22
                && viaThis.call('x') === 33 && ok;
            ok = fixed(1.25) === '1.3' && boolStr(true) === 'true' && strLen('abcd') === 4 && ok;
            ok = foreignRead(1) === 111 && viaLocal(1) === 11 && ok;
            ok = nullishThrew(null) && nullishThrew(undefined)
                && missingThrew(1) && missingThrew(true) && ok;
            ok = viaLocal(Symbol('x')) === undefined && viaLocal(10n) === undefined && ok;
        }
        var wn = Object(7), wb = Object(true), ws = Object('q');
        var wrapped = [viaLocal(wn), viaLocal(wb), viaLocal(ws)].join(',');
        wn.__flag = 99;
        [ok, wrapped, viaLocal(wn), viaLocal(Object(3)), (5).__flag,
            strLen(5) === undefined].join('|')
    "#,
        "true|11,22,33|99|11|11|true",
    );
}

#[test]
fn primitive_wrapper_proto_getters_run_and_mutations_are_live() {
    check(
        r#"
        var calls = 0;
        function readG(v) { return v.__g; }
        Object.defineProperty(Number.prototype, '__g', { configurable: true,
            get: function() { calls++; return 100 + Number(this); } });
        var sum = 0;
        for (var i = 0; i < 60; i++) sum += readG(1);
        var afterGetter = [sum, calls].join(',');
        Object.defineProperty(Number.prototype, '__g', { configurable: true, value: 7 });
        var asData = readG(1) + ',' + calls;
        for (var i = 0; i < 60; i++) readG(1);
        Object.defineProperty(Number.prototype, '__g', { configurable: true,
            get: function() { calls++; return 9; } });
        var asGetterAgain = readG(1) + ',' + calls;
        Number.prototype.__m = 1;
        function readM(v) { return v.__m; }
        for (var i = 0; i < 60; i++) readM(0);
        var m1 = readM(0);
        Number.prototype.__m = 2;
        var m2 = readM(0);
        delete Number.prototype.__m;
        var m3 = readM(0) === undefined;
        function readLate(v) { return v.__late; }
        for (var i = 0; i < 60; i++) readLate(1);
        var lateBefore = readLate(1) === undefined;
        Number.prototype.__late = 5;
        var lateAfter = readLate(1);
        function readLateB(v) { return v.__lateB; }
        for (var i = 0; i < 60; i++) readLateB(true);
        var lateBBefore = readLateB(true) === undefined;
        Object.prototype.__lateB = 6;
        var lateBAfter = readLateB(true) + ',' + readLateB(0);
        delete Object.prototype.__lateB;
        var w = Object(true);
        function readW(v) { return v.__flagW; }
        Boolean.prototype.__flagW = 1;
        for (var i = 0; i < 60; i++) readW(w);
        var wBefore = readW(w);
        Object.setPrototypeOf(w, { __flagW: 77 });
        var wAfter = readW(w);
        var ocalls = 0;
        function readO(v) { return v.__og; }
        for (var i = 0; i < 60; i++) readO(1);
        Object.defineProperty(Object.prototype, '__og', { configurable: true,
            get: function() { ocalls++; return 55; } });
        var og = readO(1) + ',' + readO(1) + ',' + ocalls;
        delete Object.prototype.__og;
        var ogGone = readO(1) === undefined;
        var wg = Object(5), wcalls = 0;
        Object.defineProperty(wg, '__wg', { configurable: true,
            get: function() { wcalls++; return Number(this) + 1; } });
        function readWG(v) { return v.__wg; }
        var wgsum = 0;
        for (var i = 0; i < 60; i++) wgsum += readWG(wg);
        [afterGetter, asData, asGetterAgain, m1, m2, m3, lateBefore, lateAfter,
            lateBBefore, lateBAfter, wBefore, wAfter, og, ogGone, wgsum, wcalls].join('|')
    "#,
        "6060,60|7,60|9,61|1|2|true|true|5|true|6,6|1|77|55,55,2|true|360|60",
    );
}

#[test]
fn primitive_wrapper_prototypes_fill_the_get_ic() {
    let mut engine = Engine::new();
    evaluate(
        &mut engine,
        r#"
        Number.prototype.__pilot = 41;
        Boolean.prototype.__pilot = 42;
        var pilotGetterCalls = 0;
        Object.defineProperty(Boolean.prototype, '__pilot_g', { configurable: true,
            get: function() { pilotGetterCalls++; return 43; } });
        var pilotWrapper = Object(3);
        "#,
    );
    for (label, base, expected) in [
        ("number", Value::Num(1.0), 41.0),
        ("boolean", Value::Bool(true), 42.0),
    ] {
        let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
        for _ in 0..2 {
            assert!(
                matches!((engine.interp.get_prop_ic(&base, "__pilot", &cache[0])).map(|__value| (__value).view()), Ok(ValueView::Num(n)) if n == expected),
                "{label}"
            );
            assert_eq!(
                cache[0].get().depth,
                0,
                "{label}: own holder on the wrapper prototype"
            );
        }
        let absent = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
        for _ in 0..2 {
            assert!(matches!(
                (engine
                    .interp
                    .get_prop_ic(&base, "__pilot_absent", &absent[0]))
                .map(|__value| (__value).view()),
                Ok(ValueView::Undefined)
            ));
            assert_eq!(absent[0].get().depth, IC_ABSENT, "{label}");
        }
    }
    // A cached absence must observe a late addition on the wrapper prototype.
    let late = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
    assert!(matches!(
        (engine
            .interp
            .get_prop_ic(&Value::Num(1.0), "__pilot_late", &late[0]))
        .map(|__value| (__value).view()),
        Ok(ValueView::Undefined)
    ));
    evaluate(&mut engine, "Number.prototype.__pilot_late = 44;");
    assert!(matches!(
        (engine
            .interp
            .get_prop_ic(&Value::Num(1.0), "__pilot_late", &late[0]))
        .map(|__value| (__value).view()),
        Ok(ValueView::Num(44.0))
    ));
    // Accessors cache as accessor states and the live getter runs on every read.
    let accessor = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
    for _ in 0..2 {
        assert!(matches!(
            (engine
                .interp
                .get_prop_ic(&Value::Bool(false), "__pilot_g", &accessor[0]))
            .map(|__value| (__value).view()),
            Ok(ValueView::Num(43.0))
        ));
        assert_eq!(accessor[0].get().depth, IC_ACCESSOR);
    }
    let global = Value::Obj(engine.interp.global.clone());
    assert!(matches!(
        (engine.interp.get_member_key(&global, "pilotGetterCalls")).map(|__value| (__value).view()),
        Ok(ValueView::Num(2.0))
    ));
    // Wrapper objects resolve through their wrapper prototype at depth 1.
    let wrapper = engine
        .interp
        .get_member_key(&global, "pilotWrapper")
        .ok()
        .unwrap();
    let wrap_cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
    for _ in 0..2 {
        assert!(matches!(
            (engine
                .interp
                .get_prop_ic(&wrapper, "__pilot", &wrap_cache[0]))
            .map(|__value| (__value).view()),
            Ok(ValueView::Num(41.0))
        ));
        assert_eq!(wrap_cache[0].get().depth, 1);
    }
}

#[test]
fn getter_cache_data_only_forwarding_does_not_call_a_speculative_getter() {
    check(
        r#"
        var gets=0,calls=0,token={};
        function Initializer(x){calls++;this.value=x;}
        function Forward(x){this.initialize.apply(this,arguments)}
        Forward.prototype.initialize=Initializer;
        for(var i=0;i<100;i++)new Forward(i);
        gets=0;calls=0;
        Object.defineProperty(Forward.prototype,'initialize',{get(){gets++;collectGetterTest();return Initializer},configurable:true});
        var a=new Forward(token),b=new Forward(7);
        [a.value===token,b.value,gets,calls].join('|');
    "#,
        "true|7|2|2",
    );
}

#[test]
fn propstat_classify_buckets_match_template_guards_without_running_getters() {
    let mut engine = Engine::new();
    assert_eq!(
        evaluate(
            &mut engine,
            "var plain={a:1},acc={},gets=0;\
             Object.defineProperty(acc,'x',{get(){gets++;return 1},configurable:true});\
             var px=new Proxy(plain,{});'ok'"
        ),
        "ok"
    );
    let env = engine.interp.global_env.clone();
    let var = |engine: &mut Engine, name: &str| {
        engine
            .interp
            .get_var(name, &env)
            .ok()
            .unwrap_or_else(|| panic!("missing global {name}"))
    };
    let plain = var(&mut engine, "plain");
    let acc = var(&mut engine, "acc");
    let px = var(&mut engine, "px");
    let cache = [const { Cell::new(IcState::EMPTY) }; PROP_IC_WAYS];
    let classify = |engine: &Engine, base: &Value, name: &str| {
        engine.interp.propstat_classify(base, name, &cache[0])
    };
    assert_eq!(
        classify(&engine, &plain, "a"),
        "data uncached (shape churn)"
    );
    assert_eq!(classify(&engine, &acc, "x"), "accessor descriptor");
    assert_eq!(classify(&engine, &px, "a"), "receiver: proxy");
    assert_eq!(classify(&engine, &plain, "missing"), "absent uncached");
    assert_eq!(
        classify(&engine, &Value::Num(1.0), "x"),
        "receiver: primitive"
    );
    assert!(cache.iter().all(|c| c.get().depth == IC_EMPTY));
    assert_eq!(evaluate(&mut engine, "String(gets)"), "0");
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn propstat_diagnostic_tallies_helper_reasons_at_exit() {
    if std::env::var_os("TRUST_JS_PROPSTAT_CHILD").is_none() {
        let name =
            "interpreter::property_read_tests::propstat_diagnostic_tallies_helper_reasons_at_exit";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture"])
            .env("TRUST_JS_PROPSTAT_CHILD", "1")
            .env("TRUST_JS_JIT_PROPSTAT", "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "exact child must execute: {stdout}\n{stderr}"
        );
        assert!(
            stderr.contains("[jit-propstat]") && stderr.contains("accessor descriptor"),
            "accessor helper reads remain attributed: {stderr}"
        );
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(
        evaluate(
            &mut engine,
            "var acc={},sum=0;\
             Object.defineProperty(acc,'x',{get(){return 2},configurable:true});\
             function read(o){return o.x}\
             for(var k=0;k<80;k++)sum+=read(acc);\
             String(sum)"
        ),
        "160"
    );
}

#[test]
fn validated_accessor_reads_observe_live_mutations_in_every_tier() {
    // Warmed accessor sites (the JIT template's validated accessor landing) must observe
    // every mutation between fill and hit: binding updates, getter replacement,
    // accessor<->data conversion, delete, prototype swaps, missing getters and throws.
    // Expectations are Node-verified.
    check(
        r#"
        function f1(){var gets=0,binding={tag:'b1'};var o={};Object.defineProperty(o,'x',{get(){gets++;return binding},configurable:true});function read(v){return v.x}var out=[];for(var k=0;k<90;k++)out.push(read(o).tag);binding={tag:'b2'};out.push(read(o).tag);Object.defineProperty(o,'x',{get(){gets++;return {tag:'g2'}},configurable:true});out.push(read(o).tag);Object.defineProperty(o,'x',{value:{tag:'data'},configurable:true});out.push(read(o).tag);for(var k=0;k<40;k++)read(o);Object.defineProperty(o,'x',{get(){gets++;return {tag:'g3'}},configurable:true});out.push(read(o).tag);delete o.x;out.push(String(read(o)));return [out[0],out[89],out[90],out[91],out[92],out[93],out[94],gets].join('|')}
        function f2(){var gets=0;function C(){}Object.defineProperty(C.prototype,'y',{get(){gets++;return this.mark},configurable:true});var a=new C();a.mark='A';var b=new C();b.mark='B';function read(v){return v.y}var s='';for(var k=0;k<60;k++)s+=read(k%2?a:b);Object.setPrototypeOf(a,{y:'patched'});var after=read(a)+read(b);Object.defineProperty(C.prototype,'y',{value:'proto-data',configurable:true});return [s.length,s.slice(0,4),after,read(b),gets].join('|')}
        function f3(){var exp={},current={v:1};Object.defineProperty(exp,'kJ',{enumerable:true,get:function(){return current}});function use(){return exp.kJ.v}var s=0;for(var k=0;k<80;k++)s+=use();current={v:5};s+=use();return [s,exp.kJ===current].join('|')}
        function f4(){var warm={};Object.defineProperty(warm,'x',{get(){return 1},configurable:true});var o={n:0};Object.defineProperty(o,'x',{get(){this.n++;if(this.n===3)throw new Error('boom');return this.n},configurable:true});function read(v){return v.x}for(var k=0;k<60;k++)read(warm);var out=[read(o),read(o)];var err='';try{read(o)}catch(e){err=e.message}Object.defineProperty(o,'x',{get:undefined,set(v){},configurable:true});return [out.join(','),err,String(read(o)),o.n].join('|')}
        function f5(){var o={tag:'T'};Object.defineProperty(o,'m',{get(){return function(){return this.tag}},configurable:true});function call(v){return v.m()}var s='';for(var k=0;k<60;k++)s+=call(o);return [s.length,s.slice(0,3),call(o)].join('|')}
        [f1(),f2(),f3(),f4(),f5()].join('\n')
    "#,
        "b1|b1|b2|g2|data|g3|undefined|93\n60|BABA|patchedB|proto-data|61\n85|true\n1,2|boom|undefined|3\n60|TTT|T",
    );
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn validated_accessor_landing_is_exercised_by_jit_reads() {
    let before = property_read::TEST_VALIDATED_ACCESSORS.with(Cell::get);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(
        evaluate(
            &mut engine,
            "var acc={},sum=0;\
             Object.defineProperty(acc,'x',{get(){return 2},configurable:true});\
             function read(o){return o.x}\
             for(var k=0;k<80;k++)sum+=read(acc);\
             String(sum)"
        ),
        "160"
    );
    assert!(
        property_read::TEST_VALIDATED_ACCESSORS.with(Cell::get) > before,
        "warmed JIT accessor reads must take the validated landing"
    );
}
