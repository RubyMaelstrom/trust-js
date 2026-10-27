//! Property shapes stay distinct across Engines' transition caches, while each Engine's
//! objects and layouts remain on its own heap. Official local ECMA-262 snapshot e28783d5,
//! OrdinaryGet, OrdinarySetWithOwnDescriptor, OrdinaryGetOwnProperty. Cache identities are not
//! part of the observable semantics. Scoped values and checked handles prevent an embedder
//! from creating a strong object edge between these separate Agents (I6).

use super::*;
use crate::value::ValueView;
use crate::{bytecode::Tier, Completion, Engine};

/// The engine value of `source` (unbranded: the crate's own form).
fn value(engine: &mut Engine, source: &str) -> Value {
    engine.scope(|cx| {
        cx.eval_value(source)
            .expect("parse")
            .unwrap_or_else(|_| panic!("evaluate"))
            .into_internal()
    })
}

fn run(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("parse") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

#[test]
fn property_shapes_foreign_mutation_cannot_alias_local_transition() {
    use crate::gc_core::CollectionKind;

    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut owner = Engine::new();
        let original = value(&mut owner, "({alpha:1})");
        let local = value(&mut owner, "({alpha:1,gamma:2})");
        let parent_shape = original.as_obj().unwrap().borrow().props.shape();
        let local_shape = local.as_obj().unwrap().borrow().props.shape();
        let global = owner.interp.global_this();
        owner
            .interp
            .set_member_key(&global, "a", local)
            .ok()
            .unwrap();
        owner
            .interp
            .set_member_key(&global, "b", original)
            .ok()
            .unwrap();

        let mut foreign = Engine::new();
        let _different_prefix = value(&mut foreign, "({beta:1})");
        foreign.interp.activate_gc_heap();
        // Reproduce a foreign cache learning a transition from an unknown parent using
        // only neutral keys and an integer identity. Heap-local shape counters would give
        // this alpha/delta child the same ID as the owner's alpha/gamma child. The layout
        // allocated here belongs to foreign; publishing it in original would violate I6.
        let prefix = [(PropKey::from_text("alpha"), PropClass::DataWritable)];
        let (foreign_shape, foreign_layout) = shape_transition(
            parent_shape,
            PropKey::from_text("delta"),
            PropClass::DataWritable,
            Some(&prefix),
        );
        assert_ne!(
            foreign_shape, local_shape,
            "foreign child identity, {tier:?}"
        );
        let foreign_layout = foreign_layout.expect("foreign transition layout");
        assert_eq!(&*foreign_layout[0].text(), "alpha");
        assert_eq!(&*foreign_layout[1].text(), "delta");
        let foreign_word = foreign_layout.cell_word() as usize;
        assert_eq!(
            foreign
                .interp
                .gc_heap
                .shapes
                .borrow()
                .layouts
                .get(foreign_shape)
                .unwrap()
                .cell_word() as usize,
            foreign_word
        );

        owner.interp.activate_gc_heap();
        assert!(owner
            .interp
            .set_member_key(&original, "delta", Value::Num(3.0))
            .is_ok());
        assert_ne!(
            original.as_obj().unwrap().borrow().props.shape(),
            local.as_obj().unwrap().borrow().props.shape()
        );
        owner.set_tier(tier);
        owner.set_tier_threshold(0);
        let owner_words = [&original, &local].map(|value| {
            value
                .as_obj()
                .unwrap()
                .borrow()
                .props
                .shared_layout()
                .expect("owner transition layout")
                .cell_word() as usize
        });
        assert_eq!(run(&mut owner, "function read(o){return o.gamma;}for(var n=0;n<200;n++)read(a);read(a)+'|'+read(b)+'|'+b.delta"), "2|undefined|3", "{tier:?}");
        for kind in [CollectionKind::Minor, CollectionKind::Major] {
            owner.interp.activate_gc_heap();
            owner.interp.gc_heap_collect_explicit(kind);
            foreign.interp.activate_gc_heap();
            foreign.interp.gc_heap_collect_explicit(kind);
            let owner_core = owner.interp.gc_heap.core.borrow();
            let foreign_core = foreign.interp.gc_heap.core.borrow();
            for word in owner_words {
                assert!(owner_core.owns_cell(word, owner.interp.gc_heap.property_layout_desc));
                assert!(!foreign_core.owns_cell(word, foreign.interp.gc_heap.property_layout_desc));
            }
            assert!(
                foreign_core.owns_cell(foreign_word, foreign.interp.gc_heap.property_layout_desc)
            );
            assert!(!owner_core.owns_cell(foreign_word, owner.interp.gc_heap.property_layout_desc));
            drop(foreign_core);
            drop(owner_core);
            assert_eq!(
                run(&mut owner, "read(a)+'|'+read(b)+'|'+b.delta"),
                "2|undefined|3",
                "{tier:?}, {kind:?}"
            );
        }
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
        .any(|(parent, ..)| *parent == SHAPE_UNCACHEABLE));
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
        (props.get("named").unwrap().value()).view(),
        ValueView::Num(37.0)
    ));
    assert!(props.get("missing").is_none());
    assert!(matches!(
        (props.get("length").unwrap().value()).view(),
        ValueView::Num(12.0)
    ));
    assert!(matches!(
        (props.get("11").unwrap().value()).view(),
        ValueView::Num(11.0)
    ));
}

#[test]
fn shape_transitions_distinguish_descriptor_classes() {
    let mut engine = Engine::new();
    let shape_of = |engine: &mut Engine, source: &str| {
        value(engine, source)
            .as_obj()
            .unwrap()
            .borrow()
            .props
            .shape()
    };
    let writable = shape_of(&mut engine, "({pc_x: 1})");
    let writable_again = shape_of(&mut engine, "({pc_x: 2})");
    let nonwritable = shape_of(
        &mut engine,
        "Object.defineProperty({}, 'pc_x', {value: 1, writable: false})",
    );
    let accessor = shape_of(
        &mut engine,
        "Object.defineProperty({}, 'pc_x', {get() { return 1; }})",
    );
    assert_eq!(writable, writable_again, "same key + class reconverges");
    assert_ne!(writable, nonwritable);
    assert_ne!(writable, accessor);
    assert_ne!(nonwritable, accessor);
}

#[test]
fn class_changing_insert_demotes_but_class_preserving_insert_keeps_shape() {
    let engine = Engine::new();
    engine.interp.activate_gc_heap();
    let mut props = Props::new();
    props.insert("pc_k", Property::plain(Value::Num(1.0)));
    let shared = props.shape();
    // Same-class replacement (enumerable-only difference included) keeps the shape.
    props.insert("pc_k", Property::data(Value::Num(2.0), true, false, true));
    assert_eq!(props.shape(), shared);
    // Writable -> non-writable is a class change: demote to a fresh unique shape.
    props.insert("pc_k", Property::data(Value::Num(3.0), false, false, true));
    assert_ne!(props.shape(), shared);
    let demoted = props.shape();
    // Data -> accessor likewise demotes again rather than reusing any shared shape.
    props.insert(
        "pc_k",
        Property::accessor_prop(Some(Value::Num(0.0)), None, false, true),
    );
    assert_ne!(props.shape(), demoted);
    assert_ne!(props.shape(), shared);
}

#[test]
fn value_only_writes_keep_shape_but_mutable_escape_demotes() {
    let engine = Engine::new();
    engine.interp.activate_gc_heap();
    let mut props = Props::new();
    props.insert("pc_k", Property::plain(Value::Num(1.0)));
    let shared = props.shape();
    assert!(props.set_value("pc_k", Value::Num(2.0)));
    assert_eq!(props.shape(), shared, "value-only write by key");
    assert!(props.set_value_at(0, Value::Num(3.0)));
    assert_eq!(props.shape(), shared, "value-only write by slot");
    assert!(
        matches!((props.take_value("pc_k")).map(|__value| (__value).view()), Some(ValueView::Num(n)) if n == 3.0)
    );
    assert_eq!(props.shape(), shared, "value-only take");
    assert!(props.set_value("pc_k", Value::Num(4.0)));
    // Acquiring &mut Property demotes even before any descriptor is touched: the
    // caller *could* flip the class behind the shape's back.
    let _ = props.get_mut("pc_k").unwrap();
    assert_ne!(props.shape(), shared, "mutable escape acquisition demotes");

    // Freeze demotes exactly when a writable data entry flips class.
    let mut frozen = Props::new();
    frozen.insert("pc_k", Property::plain(Value::Num(1.0)));
    let frozen_shape = frozen.shape();
    frozen.freeze_descriptors();
    assert_ne!(
        frozen.shape(),
        frozen_shape,
        "freeze flips writable -> demote"
    );
    assert!(!frozen.get("pc_k").unwrap().writable());
    let mut accessor_only = Props::new();
    accessor_only.insert("pc_k", Property::accessor_prop(None, None, false, true));
    let accessor_shape = accessor_only.shape();
    accessor_only.freeze_descriptors();
    assert_eq!(
        accessor_only.shape(),
        accessor_shape,
        "configurable-only freeze keeps the shape"
    );
    assert!(!accessor_only.get("pc_k").unwrap().configurable());
}

#[test]
fn replace_properties_preserves_live_prototype_flag() {
    let mut engine = Engine::new();
    let obj = value(&mut engine, "({keep: 1})");
    let o = obj.as_obj().unwrap();
    o.borrow().props.mark_proto();
    o.borrow_mut().replace_properties(Props::with_capacity(0));
    let epoch = proto_epoch();
    o.borrow_mut()
        .props
        .insert("added", Property::plain(Value::Num(1.0)));
    assert_ne!(
        proto_epoch(),
        epoch,
        "structural changes after a whole-map swap still invalidate creation ICs"
    );
}

#[test]
fn shape_transition_census_counts_shared_edges_once() {
    let mut engine = Engine::new();
    run(&mut engine, "var warm = ({census_a: 1}); 'ok'");
    let before = active_shape_transition_count();
    run(
        &mut engine,
        "var x1 = ({census_p: 1, census_q: 2}), x2 = ({census_r: 3}); 'ok'",
    );
    let after_first = active_shape_transition_count();
    assert!(
        after_first >= before + 3,
        "three new (parent, key) edges are memoized: {before} -> {after_first}"
    );
    // Reassignment (not fresh `var`s, whose global keys would add their own edges):
    // structurally identical literals must reuse every memoized transition.
    run(
        &mut engine,
        "x1 = ({census_p: 4, census_q: 5}); x2 = ({census_r: 6}); 'ok'",
    );
    assert_eq!(
        active_shape_transition_count(),
        after_first,
        "structurally identical objects reuse the memoized transitions"
    );
}
