//! Shared embed objects obey ordinary [[Get]]/[[Set]] across Engine boundaries.
//! Official local ECMA-262 snapshot e28783d5, OrdinaryGet, OrdinarySetWithOwnDescriptor,
//! OrdinaryGetOwnProperty. Cache identities are not part of the observable semantics.

use super::*;
use crate::{bytecode::Tier, Completion, Engine};

fn value(engine: &mut Engine, source: &str) -> Value {
    engine
        .eval_value(source)
        .expect("parse")
        .ok()
        .expect("evaluate")
}

fn run(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("parse") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn pair(reader: &mut Engine) -> (Engine, Engine) {
    let mut first = Engine::new();
    let mut second = Engine::new();
    let a = value(&mut first, "({alpha:1})");
    let b = value(&mut second, "({beta:2})");
    assert_ne!(
        a.as_obj().unwrap().borrow().props.shape(),
        b.as_obj().unwrap().borrow().props.shape()
    );
    let global = reader.global_this();
    assert!(reader.ctx().set_member(&global, "first", a).is_ok());
    assert!(reader.ctx().set_member(&global, "second", b).is_ok());
    // Keep the producing Engines alive: their explicit shutdown clears their own
    // heaps. The active TLS heap is the reader, not either producer.
    (first, second)
}

#[test]
fn property_shapes_foreign_own_absent_computed_and_prototype_reads() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for (source, expected) in [
            ("function read(o){return o.alpha;}for(var n=0;n<200;n++)read(first);read(first)+'|'+read(second)+'|'+read(first)", "1|undefined|1"),
            ("function read(o){return o.beta;}for(var n=0;n<200;n++)read(first);read(first)+'|'+read(second)+'|'+read(first)", "undefined|2|undefined"),
            ("function read(o,k){return o[k];}for(var n=0;n<200;n++)read(first,'alpha');read(first,'alpha')+'|'+read(second,'alpha')+'|'+read(second,'beta')", "1|undefined|2"),
            ("var a=Object.create(first),b=Object.create(second);function read(o){return o.alpha;}for(var n=0;n<200;n++)read(a);read(a)+'|'+read(b)+'|'+read(a)", "1|undefined|1"),
        ] {
            let mut reader = Engine::new();
            reader.set_tier(tier);
            reader.set_tier_threshold(0);
            let _owners = pair(&mut reader);
            assert_eq!(run(&mut reader, source), expected, "{tier:?}");
        }
    }
}

#[test]
fn property_shapes_foreign_writes_preserve_key_and_descriptor_selection() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut reader = Engine::new();
        reader.set_tier(tier);
        reader.set_tier_threshold(0);
        let _owners = pair(&mut reader);
        assert_eq!(
            run(
                &mut reader,
                r#"
            function write(o,v){o.alpha=v;}
            for(var n=0;n<200;n++)write(first,1);
            write(second,9);
            var out=Object.keys(second).join(',')+'|'+second.beta+'|'+second.alpha;
            var seen=0; Object.defineProperty(second,'alpha',{set(v){seen=v},configurable:true});
            write(second,13);out+='|'+seen+'|'+second.beta;
            Object.defineProperty(second,'alpha',{value:17,writable:false,configurable:true});
            write(second,99);out+'|'+second.alpha;
        "#
            ),
            "beta,alpha|2|9|13|2|17",
            "{tier:?}"
        );
    }
}

#[test]
fn property_shapes_foreign_mutation_cannot_alias_local_transition() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut owner = Engine::new();
        let original = value(&mut owner, "({alpha:1})");
        let local = value(&mut owner, "({alpha:1,gamma:2})");
        let mut foreign = Engine::new();
        let _different_prefix = value(&mut foreign, "({beta:1})");
        assert!(foreign
            .ctx()
            .set_member(&original, "delta", Value::Num(3.0))
            .is_ok());
        assert_ne!(
            original.as_obj().unwrap().borrow().props.shape(),
            local.as_obj().unwrap().borrow().props.shape()
        );
        owner.set_tier(tier);
        owner.set_tier_threshold(0);
        let global = owner.global_this();
        owner.ctx().set_member(&global, "a", local).ok().unwrap();
        owner.ctx().set_member(&global, "b", original).ok().unwrap();
        assert_eq!(run(&mut owner, "function read(o){return o.gamma;}for(var n=0;n<200;n++)read(a);read(a)+'|'+read(b)+'|'+b.delta"), "2|undefined|3", "{tier:?}");
    }
}

#[test]
fn property_shapes_identity_survives_agent_destruction_without_reuse() {
    let mut old = Vec::new();
    for n in 0..12 {
        let mut engine = Engine::new();
        let object = value(&mut engine, &format!("({{field{n}:1}})"));
        let shape = object.as_obj().unwrap().borrow().props.shape();
        assert!(is_cacheable_shape(shape));
        assert!(!old.contains(&shape));
        old.push(shape);
    }
}

#[test]
fn property_shapes_uncacheable_prefix_never_enters_transition_or_layout_cache() {
    let engine = Engine::new();
    engine.interp.activate_gc_heap();
    let mut first = Props::new();
    let mut second = Props::new();
    first.insert("first", Property::plain(Value::Num(1.0)));
    second.insert("second", Property::plain(Value::Num(2.0)));
    first.shape = SHAPE_UNCACHEABLE;
    second.shape = SHAPE_UNCACHEABLE;
    for map in [&mut first, &mut second] {
        map.insert("tail", Property::plain(Value::Num(3.0)));
        assert_eq!(map.shape(), SHAPE_UNCACHEABLE);
    }
    assert!(first.contains("first") && !first.contains("second"));
    assert!(second.contains("second") && !second.contains("first"));
    let shapes = engine.interp.gc_heap.shapes.borrow();
    assert!(!shapes
        .transitions
        .keys()
        .any(|(parent, _)| *parent == SHAPE_UNCACHEABLE));
    assert!(shapes.layouts.get(SHAPE_UNCACHEABLE).is_none());
}

#[test]
fn property_shapes_exhausted_array_length_hint_does_not_hide_named_properties() {
    let engine = Engine::new();
    engine.interp.activate_gc_heap();
    let mut props = Props::new();
    props.elem_mode.set(true);
    props.insert("length", Property::plain(Value::Num(12.0)));
    for index in 0..12 {
        // Keep the original classic-storage regression covered when the normal
        // array builder now selects keyless elements.
        let slot = props.entries.len();
        props
            .entries
            .push((index_key(index), Property::plain(Value::Num(index as f64))));
        props.note_inserted(slot);
    }
    props.insert("named", Property::plain(Value::Num(37.0)));
    assert!(props.entries.len() > INDEX_THRESHOLD);
    props.force_uncacheable_shape_for_test();
    // Only this disposable test Agent's hint is exhausted, never the global ID allocator.
    engine
        .interp
        .gc_heap
        .array_length_shape
        .set(SHAPE_UNCACHEABLE);
    assert!(matches!(
        props.get("named").unwrap().value(),
        Value::Num(37.0)
    ));
    assert!(props.get("missing").is_none());
    assert!(matches!(
        props.get("length").unwrap().value(),
        Value::Num(12.0)
    ));
    assert!(matches!(props.get("11").unwrap().value(), Value::Num(11.0)));
}
