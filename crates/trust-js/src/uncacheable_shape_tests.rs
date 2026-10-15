//! Exercise the exhausted-ID state without resetting the shared shape allocator.
//! OrdinaryGet/Set, GetBindingValue, GetPrototypeFromConstructor and OrdinaryHasInstance:
//! ECMA-262 local official snapshot e28783d5fc9d. Uncacheable means use the ordinary
//! algorithm, never reject a valid program or reuse an unrelated descriptor location.
use super::*;
use crate::bytecode::{IcState, Tier, IC_EMPTY, PROP_IC_WAYS};
use crate::{Completion, Engine};
use std::cell::Cell;

fn run(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("shape fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn object(engine: &mut Engine, name: &str) -> Gc {
    let env = engine.interp.global_env.clone();
    let value = engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("fixture object");
    value.as_obj().expect("object").clone()
}

fn unknown(object: &Gc) {
    object.borrow_mut().props.force_uncacheable_shape_for_test();
}

fn cells() -> [Cell<IcState>; PROP_IC_WAYS] {
    std::array::from_fn(|_| Cell::new(IcState::EMPTY))
}

#[test]
fn uncacheable_shapes_skip_property_computed_stub_and_creation_publication() {
    let mut engine = Engine::new();
    engine.interp.activate_gc_heap();
    let first = Object::new(None);
    first
        .borrow_mut()
        .props
        .insert("alpha", Property::plain(Value::Num(7.0)));
    unknown(&first);
    let second = Object::new(None);
    second
        .borrow_mut()
        .props
        .insert("beta", Property::plain(Value::Num(9.0)));
    unknown(&second);
    let cache = cells();
    let key = Rc::from("alpha");
    let computed = crate::lstr::LStr::from("alpha");
    let i = &mut engine.interp;
    for _ in 0..4 {
        assert!(matches!(
            i.get_prop_keyed(&Value::Obj(first.clone()), "alpha", &cache[0]),
            Ok(Value::Num(7.0))
        ));
        assert!(matches!(
            i.get_prop_keyed(&Value::Obj(second.clone()), "alpha", &cache[0]),
            Ok(Value::Undefined)
        ));
        assert!(matches!(
            i.get_computed_property(&Value::Obj(first.clone()), &computed),
            Ok(Value::Num(7.0))
        ));
        assert!(matches!(
            i.get_computed_property(&Value::Obj(second.clone()), &computed),
            Ok(Value::Undefined)
        ));
    }
    assert!(cache.iter().all(|way| way.get().depth == IC_EMPTY));
    assert!(i.stub_cache_get(SHAPE_UNCACHEABLE, "alpha").is_none());
    assert!(i
        .computed_reads
        .lookup(computed.as_ptr() as usize, SHAPE_UNCACHEABLE)
        .is_none());
    assert!(i.try_ic_set(&first, &key, &Value::Num(11.0), &cache[0]));
    assert!(i.try_ic_set(&second, &key, &Value::Num(13.0), &cache[0]));
    assert!(cache.iter().all(|way| way.get().depth == IC_EMPTY));
    assert!(matches!(
        first.borrow().props.get("alpha").unwrap().value(),
        Value::Num(11.0)
    ));
    assert!(matches!(
        second.borrow().props.get("alpha").unwrap().value(),
        Value::Num(13.0)
    ));
    assert!(matches!(
        second.borrow().props.get("beta").unwrap().value(),
        Value::Num(9.0)
    ));
}

#[test]
fn uncacheable_shapes_preserve_live_reads_writes_and_prototype_mutation_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        run(&mut engine, "var a={alpha:1}, b={beta:2};var p={alpha:3}, q={beta:4};var c=Object.create(p),d=Object.create(q);function read(o){return o.alpha;}function dynamic(o,k){return o[k];}function write(o,v){o.alpha=v;return o.alpha;}function absent(o){return o.missing;}");
        for name in ["a", "b", "p", "q", "c", "d"] {
            unknown(&object(&mut engine, name));
        }
        assert_eq!(run(&mut engine, "var out='';for(var j=0;j<160;j++){out=read(a)+'|'+read(b)+'|'+dynamic(c,'alpha')+'|'+dynamic(d,'alpha')+'|'+absent(c);}out"), "1|undefined|3|undefined|undefined", "{tier:?}");
        assert_eq!(
            run(
                &mut engine,
                "write(a,5)+'|'+write(b,6)+'|'+b.beta+'|'+Object.keys(b).join(',')"
            ),
            "5|6|2|beta,alpha",
            "{tier:?}"
        );
        assert_eq!(
            run(
                &mut engine,
                "q.missing=12;p.missing=11;absent(c)+'|'+absent(d)"
            ),
            "11|12",
            "{tier:?}"
        );
        run(&mut engine, "delete p.alpha;Object.defineProperty(q,'alpha',{get:function(){return this.marker+10;},configurable:true});d.marker=7;");
        unknown(&object(&mut engine, "p"));
        unknown(&object(&mut engine, "q"));
        assert_eq!(
            run(&mut engine, "read(c)+'|'+dynamic(d,'alpha')"),
            "undefined|17",
            "{tier:?}"
        );
        run(&mut engine, "var base={alpha:3},midA=Object.create(base),midB=Object.create(base);midB.alpha=8;var left=Object.create(midA),right=Object.create(midB);");
        unknown(&object(&mut engine, "midA"));
        unknown(&object(&mut engine, "midB"));
        assert_eq!(
            run(&mut engine, "read(left)+'|'+read(right)+'|'+read(left)"),
            "3|8|3",
            "a sentinel intermediate hop cannot prove absence: {tier:?}"
        );
    }
}

#[test]
fn uncacheable_shapes_preserve_direct_and_deep_global_names_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        run(&mut engine, "function deep(){let captured=0;return function(){return sentinelValue+captured;};}var load=deep();");
        let global = engine.interp.global.clone();
        global
            .borrow_mut()
            .props
            .insert("sentinelBefore", Property::plain(Value::Num(1.0)));
        global
            .borrow_mut()
            .props
            .insert("sentinelValue", Property::plain(Value::Num(23.0)));
        unknown(&global);
        assert_eq!(
            run(
                &mut engine,
                "var sum=0;for(var j=0;j<160;j++)sum=load();sum"
            ),
            "23"
        );
        assert_eq!(run(&mut engine, "sentinelValue"), "23");
        global.borrow_mut().props.remove("sentinelBefore");
        global
            .borrow_mut()
            .props
            .insert("sentinelAfter", Property::plain(Value::Num(97.0)));
        unknown(&global);
        assert_eq!(
            run(&mut engine, "load()+'|'+sentinelValue"),
            "23|23",
            "{tier:?}"
        );
        run(&mut engine, "Object.defineProperty(globalThis,'sentinelValue',{get:function(){return 31;},configurable:true});");
        unknown(&global);
        assert_eq!(
            run(&mut engine, "load()+'|'+sentinelValue"),
            "31|31",
            "{tier:?}"
        );
    }
}

#[test]
fn uncacheable_shapes_preserve_constructor_and_instanceof_semantics_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        run(&mut engine, "function C(x){this.x=x;}function make(x){return new C(x);}var old=make(1);for(var j=0;j<160;j++)make(j);");
        let ctor = object(&mut engine, "C");
        unknown(&ctor);
        engine
            .interp
            .construct_ics
            .remove(&(Gc::as_ptr(&ctor) as usize));
        let epoch = crate::bytecode::CALL_IC_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        let global_env = Rc::as_ptr(&engine.interp.global_env) as usize;
        assert!(engine
            .interp
            .construct_ic_fill(&ctor, Gc::as_ptr(&ctor) as usize, epoch, global_env)
            .is_none());
        let cache = Cell::new(IcState::EMPTY);
        let old = object(&mut engine, "old");
        assert!(matches!(
            engine
                .interp
                .instanceof_ic(&Value::Obj(old), &Value::Obj(ctor.clone()), &cache),
            Ok(Value::Bool(true))
        ));
        assert_eq!(cache.get().depth, IC_EMPTY);
        assert_eq!(
            run(
                &mut engine,
                "var fresh=make(5);fresh.x+'|'+(fresh instanceof C)+'|'+(old instanceof C)"
            ),
            "5|true|true",
            "{tier:?}"
        );
        assert!(!engine
            .interp
            .construct_ics
            .contains_key(&(Gc::as_ptr(&ctor) as usize)));
        run(
            &mut engine,
            "function makeAfter(x){return new C(x);}for(var j=0;j<160;j++)makeAfter(j);",
        );
        let decoy = Object::new(None);
        decoy
            .borrow_mut()
            .props
            .insert("tag", Property::plain(Value::Num(99.0)));
        ctor.borrow_mut().props.remove("name");
        ctor.borrow_mut()
            .props
            .insert("decoy", Property::plain(Value::Obj(decoy)));
        unknown(&ctor);
        assert_eq!(
            run(
                &mut engine,
                "var moved=makeAfter(8);moved.x+'|'+(Object.getPrototypeOf(moved)===C.prototype)"
            ),
            "8|true",
            "moved prototype slot: {tier:?}"
        );
        let fresh = object(&mut engine, "fresh");
        assert!(matches!(
            engine
                .interp
                .instanceof_ic(&Value::Obj(fresh), &Value::Obj(ctor.clone()), &cache),
            Ok(Value::Bool(true))
        ));
        assert_eq!(run(&mut engine, "C.prototype={tag:7};var fresh=make(6);fresh.x+'|'+fresh.tag+'|'+(fresh instanceof C)+'|'+(old instanceof C)"), "6|7|true|false", "{tier:?}");
    }
}

#[test]
fn uncacheable_shapes_keep_regexp_dependencies_live_without_publishing_slots() {
    let mut engine = Engine::new();
    assert!(crate::builtins::regexp_dependencies_canonical(
        &engine.interp
    ));
    let before = engine.interp.regexp_dependency_cache.get().shape;
    let proto = engine.interp.extra_protos.get("RegExp").unwrap().clone();
    unknown(&proto);
    assert!(crate::builtins::regexp_dependencies_canonical(
        &engine.interp
    ));
    assert_eq!(engine.interp.regexp_dependency_cache.get().shape, before);
    run(&mut engine, "Object.defineProperty(RegExp.prototype,'global',{get:function(){return true;},configurable:true});");
    unknown(&proto);
    assert!(!crate::builtins::regexp_dependencies_canonical(
        &engine.interp
    ));
    assert_eq!(run(&mut engine, "(/a/).flags"), "g");
    assert_ne!(
        engine.interp.regexp_dependency_cache.get().shape,
        SHAPE_UNCACHEABLE
    );
}
