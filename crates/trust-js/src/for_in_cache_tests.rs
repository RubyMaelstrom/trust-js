//! EnumerateObjectProperties / CreateForInIterator, official ECMA-262 local snapshot
//! e28783d5fc9dc12b3de905961e2c71410b38a202. Candidate storage is unobservable; live
//! deletion, key order, shadowing and each activation's independent cursor are not.
use crate::value::{heap_allocated_objects, Object, Property, Value};
use crate::{bytecode::Tier, Completion, Engine};
use std::rc::Rc;

fn run(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("enumeration fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn backing(engine: &mut Engine, source: &Value) -> Value {
    engine.interp.activate_gc_heap();
    engine.interp.for_in_keys(source).ok().unwrap().into_value()
}

#[cfg(feature = "embed")]
fn owned_value(engine: &mut Engine, source: &str) -> Value {
    engine
        .eval_value(source)
        .expect("parse")
        .ok()
        .expect("evaluate")
}

#[test]
#[cfg(feature = "embed")]
fn for_in_cache_distinguishes_foreign_heaps_and_mixed_prototypes() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut first = Engine::new();
        let mut second = Engine::new();
        let a = owned_value(&mut first, "({alpha:1})");
        let b = owned_value(&mut second, "({beta:2})");
        assert_ne!(
            a.as_obj().unwrap().borrow().props.shape(),
            b.as_obj().unwrap().borrow().props.shape(),
            "foreign key sequences must have distinct global identities"
        );
        let mut reader = Engine::new();
        reader.set_tier(tier);
        reader.set_tier_threshold(0);
        let global = reader.global_this();
        assert!(reader.ctx().set_member(&global, "first", a).is_ok());
        assert!(reader.ctx().set_member(&global, "second", b).is_ok());
        assert_eq!(run(&mut reader, "function keys(o){var out='';for(var k in o)out+=k+',';return out;}keys(first)+'|'+keys(second)+'|'+keys(first)"),
            "alpha,|beta,|alpha,", "{tier:?}");
        assert_eq!(run(&mut reader, "var a=Object.create(first),b=Object.create(second);a.own=1;b.own=2;keys(a)+'|'+keys(b)+'|'+keys(a)"),
            "own,alpha,|own,beta,|own,alpha,", "{tier:?}");
    }
}

#[test]
#[cfg(feature = "embed")]
fn for_in_cache_ignores_shape_ids_assigned_under_foreign_mutation() {
    let mut owner = Engine::new();
    let original = owned_value(&mut owner, "({alpha:1})");
    let same_heap = owned_value(&mut owner, "({alpha:1,gamma:2})");
    let mut foreign = Engine::new();
    let _foreign_shape = owned_value(&mut foreign, "({beta:1})");
    assert!(foreign
        .ctx()
        .set_member(&original, "delta", Value::Num(3.0))
        .is_ok());
    assert_ne!(
        original.as_obj().unwrap().borrow().props.shape(),
        same_heap.as_obj().unwrap().borrow().props.shape(),
        "foreign mutation must not collide with a local transition"
    );
    let first = backing(&mut owner, &same_heap);
    let second = backing(&mut owner, &original);
    assert_eq!(
        (key_at(&first, 0), key_at(&first, 1)),
        ("alpha".into(), "gamma".into())
    );
    assert_eq!(
        (key_at(&second, 0), key_at(&second, 1)),
        ("alpha".into(), "delta".into())
    );
    foreign.ctx();
    original
        .as_obj()
        .unwrap()
        .borrow_mut()
        .props
        .remove("alpha");
    let third = backing(&mut owner, &original);
    assert_eq!(key_at(&third, 0), "delta");
    assert_eq!(
        key_at(&first, 0),
        "alpha",
        "an old cursor retains its own immutable candidates"
    );
}

fn predicted_object(layout: &crate::value::PropertyLayout, count: usize) -> Value {
    let mut props = crate::value::Props::with_layout(count, Some(layout.clone()));
    for key in layout.iter().take(count) {
        props.append_initialized_field(key, Property::plain(Value::Num(1.0)));
    }
    Value::Obj(Object::new_with_parts(
        None,
        props,
        crate::value::Exotic::None,
    ))
}

#[test]
fn for_in_cache_pins_layout_not_sources_and_releases_evicted_layouts() {
    let mut engine = Engine::new();
    let layout = crate::value::new_property_layout(vec![Rc::from("a"), Rc::from("b")]);
    let weak_layout = Rc::downgrade(&layout);
    let a = predicted_object(&layout, 1);
    let b = predicted_object(&layout, 2);
    let weak_a = Rc::downgrade(a.as_obj().unwrap());
    let first = backing(&mut engine, &a);
    let second = backing(&mut engine, &b);
    assert_eq!(key_at(&first, 0), "a");
    assert_eq!(key_at(&second, 1), "b");
    assert!(
        !Rc::ptr_eq(first.as_obj().unwrap(), second.as_obj().unwrap()),
        "live prefix participates in proof"
    );
    let references = Rc::strong_count(&layout);
    for _ in 0..10 {
        backing(&mut engine, &a);
        backing(&mut engine, &b); // leave stale recency entries for both pinned layouts.
    }
    assert_eq!(
        Rc::strong_count(&layout),
        references,
        "recency keys must not retain layout pins"
    );
    drop(a);
    drop(b);
    drop(layout);
    assert!(weak_a.upgrade().is_none(), "cache does not root receiver");
    assert!(
        weak_layout.upgrade().is_some(),
        "live cache entries own layout proof"
    );
    for index in 0..400 {
        let layout = crate::value::new_property_layout(vec![Rc::from(format!("fresh{index}"))]);
        backing(&mut engine, &predicted_object(&layout, 1));
    }
    assert!(
        weak_layout.upgrade().is_none(),
        "stale recency records must not own evicted layouts"
    );
    engine.interp.gc_collect();
    assert_eq!(key_at(&first, 0), "a");
    assert_eq!(
        key_at(&second, 1),
        "b",
        "active candidates survive origin/layout eviction"
    );
}

#[test]
fn for_in_cache_accounts_unused_predictions_and_detaches_on_mutation() {
    let mut engine = Engine::new();
    let layout =
        crate::value::new_property_layout(vec![Rc::from("a"), Rc::from("x".repeat((1 << 20) + 1))]);
    let source = predicted_object(&layout, 1);
    let before = engine.interp.enumeration_keys.stats();
    assert_eq!(key_at(&backing(&mut engine, &source), 0), "a");
    assert_eq!(
        engine.interp.enumeration_keys.stats(),
        before,
        "budget must include pinned unused suffix"
    );
    let empty = predicted_object(&layout, 0);
    backing(&mut engine, &empty);
    assert_eq!(
        engine.interp.enumeration_keys.stats().0,
        before.0 + 1,
        "empty prefix does not need to pin an unused layout"
    );
    let layout = crate::value::new_property_layout(vec![Rc::from("a"), Rc::from("b")]);
    let source = predicted_object(&layout, 2);
    let original = backing(&mut engine, &source);
    let old_layout = Rc::downgrade(&layout);
    drop(layout);
    source.as_obj().unwrap().borrow_mut().props.remove("a");
    assert!(
        old_layout.upgrade().is_some(),
        "cache pins immutable pre-mutation allocation"
    );
    assert_eq!(key_at(&original, 0), "a");
    assert_eq!(key_at(&backing(&mut engine, &source), 0), "b");
}

fn key_at(backing: &Value, index: u32) -> String {
    match backing
        .as_obj()
        .unwrap()
        .borrow()
        .props
        .get_index(index)
        .unwrap()
        .value()
    {
        Value::Str(key) => key.to_string(),
        _ => panic!("candidate must be a string"),
    }
}

#[test]
fn for_in_backing_hits_share_identity_without_retaining_author_objects() {
    let mut engine = Engine::new();
    engine.interp.activate_gc_heap();
    let prototype = Object::new(None);
    prototype
        .borrow_mut()
        .props
        .insert("inherited", Property::plain(Value::Num(1.0)));
    let object = Object::new(Some(prototype.clone()));
    object
        .borrow_mut()
        .props
        .insert("own", Property::plain(Value::Num(2.0)));
    let weak_object = Rc::downgrade(&object);
    let weak_proto = Rc::downgrade(&prototype);
    let source = Value::Obj(object);
    let keys = backing(&mut engine, &source);
    let allocated = heap_allocated_objects(&engine.interp.gc_heap);
    for _ in 0..1000 {
        let next = backing(&mut engine, &source);
        assert!(Rc::ptr_eq(keys.as_obj().unwrap(), next.as_obj().unwrap()));
    }
    assert_eq!(heap_allocated_objects(&engine.interp.gc_heap), allocated);
    {
        let object = keys.as_obj().unwrap().borrow();
        assert!(object.proto.is_none());
        assert!(!object.extensible);
        for index in 0..2 {
            let property = object.props.get_index(index).unwrap();
            assert!(!property.writable() && !property.configurable());
        }
    }
    drop(source);
    drop(prototype);
    assert!(weak_object.upgrade().is_none());
    assert!(weak_proto.upgrade().is_none());
    engine.interp.gc_collect();
    assert_eq!(key_at(&keys, 0), "own");
}

#[test]
fn for_in_backing_is_bounded_and_eviction_does_not_invalidate_live_candidates() {
    let mut engine = Engine::new();
    engine.interp.activate_gc_heap();
    let first = Object::new(None);
    first
        .borrow_mut()
        .props
        .insert("first", Property::plain(Value::Num(1.0)));
    let pinned = backing(&mut engine, &Value::Obj(first));
    for index in 0..400 {
        let object = Object::new(None);
        object.borrow_mut().props.insert(
            format!("key{index}").as_str(),
            Property::plain(Value::Num(1.0)),
        );
        backing(&mut engine, &Value::Obj(object));
    }
    let (count, bytes) = engine.interp.enumeration_keys.stats();
    assert_eq!(count, 256);
    assert!(bytes <= 1 << 20);
    let huge = Object::new(None);
    huge.borrow_mut().props.insert(
        "x".repeat((1 << 20) + 1).as_str(),
        Property::plain(Value::Num(1.0)),
    );
    let huge = backing(&mut engine, &Value::Obj(huge));
    assert_eq!(engine.interp.enumeration_keys.stats(), (count, bytes));
    engine.interp.gc_collect();
    assert_eq!(key_at(&pinned, 0), "first");
    assert_eq!(key_at(&huge, 0).len(), (1 << 20) + 1);
}

#[test]
fn for_in_hot_compiled_and_ast_loops_do_not_allocate_per_enumeration() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(run(&mut engine, "var input={a:1,b:2,c:3};function count(n){var total=0;for(var j=0;j<n;j++)for(var k in input)total++;return total;}count(100);"), "300");
        let before = heap_allocated_objects(&engine.interp.gc_heap);
        assert_eq!(run(&mut engine, "count(1000)"), "3000");
        let allocated = heap_allocated_objects(&engine.interp.gc_heap) - before;
        assert!(
            allocated < 8,
            "{tier:?}: {allocated} new objects for cached enumerations"
        );
    }
}

#[test]
fn for_in_nested_cursors_mutations_and_string_keys_are_independent() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            run(
                &mut engine,
                r#"
            function check(ok){if(!ok)throw Error('candidate backing');}
            function keys(o){var s='';for(var k in o)s+=k+',';return s;}
            var a={a:1,b:2,c:3},b={a:4,b:5,c:6};
            check(keys(a)==='a,b,c,' && keys(b)==='a,b,c,');
            var result='';
            for(var k in a){result+=k+'['+keys(b)+']';if(k==='a')delete a.b;}
            check(result==='a[a,b,c,]c[a,b,c,]');
            check(keys(b)==='a,b,c,');
            var p={hidden:1,z:2},q=Object.create(p);
            Object.defineProperty(q,'hidden',{value:3,enumerable:false,configurable:true});
            q.a=1;check(keys(q)==='a,z,');
            Object.defineProperty(q,'hidden',{enumerable:true});check(keys(q)==='hidden,a,z,');
            Object.setPrototypeOf(q,{newProto:1});check(keys(q)==='hidden,a,newProto,');
            var names={'😀':1};names[String.fromCharCode(0xD800)]=2;names[Symbol('skip')]=3;
            var stable=keys(names);for(var j=0;j<100;j++)check(keys(names)===stable);
            var key;for(key in b){key+='changed';}check(keys(b)==='a,b,c,');
            'ok'
        "#
            ),
            "ok",
            "{tier:?}"
        );
    }
}

#[test]
fn for_in_suspended_activations_pin_backing_after_cache_eviction_and_gc() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            run(
                &mut engine,
                r#"
            function* enumerate(o){for(var key in o)yield key;}
            var a={a:1,b:2,c:3},b={a:4,b:5,c:6},left=enumerate(a),right=enumerate(b);
            left.next().value+right.next().value
        "#
            ),
            "aa"
        );
        assert_eq!(
            run(
                &mut engine,
                r#"
            for(var n=0;n<400;n++){var fresh={};fresh['other'+n]=1;for(var key in fresh){}}
            delete a.b;'evicted'
        "#
            ),
            "evicted"
        );
        engine.interp.gc_collect();
        assert_eq!(run(&mut engine, "[left.next().value,right.next().value,right.next().value,left.next().done,right.next().done].join('|')"), "c|b|c|true|true", "{tier:?}");
    }
}

#[test]
fn for_in_exotics_keep_their_observable_internal_methods_on_repeated_runs() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            run(
                &mut engine,
                r#"
            var own=0,descriptors=0,gets=0;
            var proxy=new Proxy({p:1},{ownKeys(t){own++;return Reflect.ownKeys(t)},
                getOwnPropertyDescriptor(t,k){descriptors++;return Reflect.getOwnPropertyDescriptor(t,k)},
                get(t,k){gets++;return Reflect.get(t,k)}});
            var child=Object.create(proxy);child.a=1;
            function keys(o){var result='';for(var k in o)result+=k;return result;}
            var first=keys(child),previous=own,previousDescriptors=descriptors;
            var second=keys(child);
            [first,second,own===previous+1,descriptors>previousDescriptors,gets,
             keys(new Uint8Array(2)),keys('xy')].join('|')
        "#
            ),
            "ap|ap|true|true|0|01|01",
            "{tier:?}"
        );
    }
}
