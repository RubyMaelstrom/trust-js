//! P1 S5 step 4: [`JitLayout`] v2 describes the build truthfully. Every exported offset is read
//! on a live structure through the raw offset and compared with the safe accessor: the
//! traced handles, in-cell named fields (in place and spilled), narrow slot memos, in-cell first
//! bindings and closure cells. Nothing here runs generated code.

use super::*;
use crate::interpreter::{new_scope, Binding};
use crate::value::ValueView;

/// The `T` at byte offset `off` from `base`.
fn read<T: Copy>(base: usize, off: usize) -> T {
    // SAFETY: callers pass a live structure and an offset `jit_layout` measured within it.
    unsafe { ((base + off) as *const T).read_unaligned() }
}

/// A handle's stored word: what a `Value::Obj` payload and an `Option<Gc>` field hold.
fn stored(object: &Gc) -> usize {
    read(object as *const Gc as usize, 0)
}

/// An environment handle's stored word: what an `Option<Env>` parent field holds.
fn env_stored(env: &Env) -> usize {
    read(env as *const Env as usize, 0)
}

/// The `Object` behind a handle, as generated code finds it.
fn body(layout: &JitLayout, object: &Gc) -> usize {
    stored(object) + layout.obj_from_cell
}

fn probe(engine: &crate::Engine) -> JitLayout {
    jit_layout(&engine.interp.object_proto)
}

/// A sequence's data address, length and capacity word, read through `seq` at `at`.
fn sequence(seq: JitVecLayout, at: usize) -> (usize, usize, usize) {
    match seq {
        JitVecLayout::Vec { ptr, len, cap } => (read(at, ptr), read(at, len), read(at, cap)),
        JitVecLayout::Inline {
            len,
            cap,
            inline,
            heap,
            ..
        } => {
            let cap = read::<u32>(at, cap) as usize;
            let data = if cap == 0 {
                at + inline
            } else {
                read(at, heap)
            };
            (data, read::<u32>(at, len) as usize, cap)
        }
    }
}

/// How the layout's binding-map entries hold their elements.
fn entries_of(layout: &JitLayout) -> JitVecLayout {
    layout
        .repr
        .scope_small
        .expect("small binding storage")
        .storage
}

fn global(engine: &mut crate::Engine, source: &str, name: &str) -> Gc {
    engine.eval(source, false).expect("fixture parses");
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let ValueView::Obj(object) = (value).view() else {
        panic!("{name} is an object")
    };
    object
}

#[test]
fn layout_is_valid_with_the_build_ownership() {
    let engine = crate::Engine::new();
    let layout = probe(&engine);
    assert!(layout.valid && layout.key_probe_ok && layout.call_probe_valid);
    assert!(layout.scope_parent_valid && layout.scope_small_valid && layout.scope_with_valid);
    assert!(layout.repr.exotic_ordinary_contiguous);
    let object = Object::new(None);
    assert_eq!(
        body(&layout, &object),
        &*object.borrow() as *const Object as usize
    );
    assert_eq!(
        stored(&object),
        Gc::as_ptr(&object) as usize,
        "the stored word is the cell address, the `Gc::as_ptr` identity"
    );
    assert!(matches!(
        layout.repr.props_fields,
        JitVecLayout::Inline {
            inline_capacity: INLINE_NAMED_SLOTS,
            ..
        }
    ));
    assert!(matches!(
        entries_of(&layout),
        JitVecLayout::Inline {
            inline_capacity: 1,
            ..
        }
    ));
    assert_eq!(
        (layout.repr.slot_memo_width, layout.repr.slot_memo_none),
        (2, u32::from(u16::MAX))
    );
    assert!(layout.repr.exotic_collection_tag.is_some());
}

#[test]
fn object_offsets_read_proto_shape_and_exotic_tags() {
    let mut engine = crate::Engine::new();
    let layout = probe(&engine);
    let proto = Object::new(None);
    let object = Object::new(Some(proto.clone()));
    object
        .borrow_mut()
        .props
        .insert("x", Property::plain(Value::Num(1.0)));
    let base = body(&layout, &object);
    assert_eq!(read::<usize>(base, layout.obj_proto), stored(&proto));
    assert_eq!(
        read::<usize>(base, layout.obj_proto),
        Gc::as_ptr(&proto) as usize
    );
    assert_eq!(
        read::<usize>(body(&layout, &proto), layout.obj_proto),
        0,
        "a null prototype is the handle's niche"
    );
    assert_eq!(
        read::<u32>(base, layout.obj_props + layout.props_shape),
        object.borrow().props.shape()
    );
    assert_eq!(read::<u8>(base, layout.obj_ic_plain), 1);
    assert_eq!(read::<u8>(base, layout.obj_extensible), 1);
    let tag = |object: &Gc| read::<u8>(body(&layout, object), layout.obj_exotic);
    assert_eq!(tag(&object), layout.exotic_none_tag);
    for (exotic, expected) in [
        (Exotic::Array, layout.exotic_array_tag),
        (Exotic::str_wrap("s".into()), layout.exotic_strwrap_tag),
        (Exotic::NumWrap(2.5), layout.exotic_numwrap_tag),
        (Exotic::BoolWrap(true), layout.exotic_boolwrap_tag),
    ] {
        let wrapper = Object::new_with_parts(None, Props::new(), exotic);
        assert_eq!(tag(&wrapper), expected);
    }
    // Maps and Sets keep their data in the cell (`Exotic::Collection`, declared right after
    // `None`).
    let map = global(&mut engine, "var probeMap = new Map([[1, 2]]);", "probeMap");
    let set = global(&mut engine, "var probeSet = new Set([3]);", "probeSet");
    for collection in [&map, &set] {
        let found = tag(collection);
        assert_eq!(
            collection.borrow().exotic.is_collection(),
            layout.repr.exotic_collection_tag.is_some()
        );
        assert_eq!(
            found,
            layout
                .repr
                .exotic_collection_tag
                .unwrap_or(layout.exotic_none_tag)
        );
        let last = layout
            .repr
            .exotic_collection_tag
            .unwrap_or(layout.exotic_none_tag);
        assert!((layout.exotic_none_tag..=last).contains(&found));
        assert_eq!(last - layout.exotic_none_tag, 1);
    }
}

#[test]
fn named_fields_read_through_the_fields_description_in_place_and_spilled() {
    let engine = crate::Engine::new();
    let layout = probe(&engine);
    for count in [1usize, 2, 4, 5, 9, 40] {
        let object = Object::new(None);
        for slot in 0..count {
            object.borrow_mut().props.insert(
                format!("f{slot}").as_str(),
                Property::plain(Value::Num(slot as f64)),
            );
        }
        let at = body(&layout, &object) + layout.obj_props + layout.props_entries;
        let (data, len, cap) = sequence(layout.repr.props_fields, at);
        let borrowed = object.borrow();
        let props = &borrowed.props;
        let values: &[Property] = &props.entries.fields;
        assert_eq!(len, values.len(), "{count} fields");
        assert_eq!(data, values.as_ptr() as usize, "{count} fields");
        assert_eq!(cap, props.entries.heap_capacity(), "{count} fields");
        assert_eq!(
            cap == 0,
            count <= INLINE_NAMED_SLOTS,
            "in place exactly while the capacity word is 0"
        );
        for (slot, value) in values.iter().enumerate() {
            let entry = data + slot * layout.entry_size;
            assert_eq!(read::<u64>(entry, layout.entry_value), value.packed.bits());
            assert_eq!(
                props
                    .property_at(slot)
                    .map(|p| p as *const Property as usize),
                Some(entry)
            );
        }
    }
}

#[test]
fn slot_memos_read_their_width_and_none() {
    let engine = crate::Engine::new();
    let layout = probe(&engine);
    let memo = |object: &Gc| {
        let at = body(&layout, object) + layout.obj_props + layout.props_len_slot;
        match layout.repr.slot_memo_width {
            2 => u32::from(read::<u16>(at, 0)),
            4 => read::<u32>(at, 0),
            width => panic!("no {width}-byte slot memo"),
        }
    };
    let plain = Object::new(None);
    plain
        .borrow_mut()
        .props
        .insert("x", Property::plain(Value::Num(1.0)));
    assert_eq!(plain.borrow().props.len_slot.get().slot(), None);
    assert_eq!(memo(&plain), layout.repr.slot_memo_none);
    // A slot past one byte tells the width apart.
    for before in [0usize, 2, 300] {
        let object = Object::new(None);
        for slot in 0..before {
            object.borrow_mut().props.insert(
                format!("k{slot}").as_str(),
                Property::plain(Value::Num(0.0)),
            );
        }
        object
            .borrow_mut()
            .props
            .insert("length", Property::plain(Value::Num(0.0)));
        let slot = object.borrow().props.len_slot.get().slot();
        assert_eq!(slot, Some(before));
        assert_eq!(memo(&object) as usize, before);
    }
}

#[test]
fn scope_offsets_read_generation_parent_and_bindings() {
    let engine = crate::Engine::new();
    let layout = probe(&engine);
    let entries = |env: &Env| {
        let base = Env::as_ptr(env) as usize;
        let tag = read::<u8>(base, layout.scope_small_tag);
        let (data, len, _) = sequence(entries_of(&layout), base + layout.scope_small_vec);
        (tag, data, len)
    };
    let check_bindings = |env: &Env, data: usize, count: usize| {
        let scope = env.borrow();
        for slot in 0..count {
            let binding = data + slot * layout.scope_binding_stride + layout.scope_binding_offset;
            let live = scope.vars.fixed_binding_at(slot).expect("fixed slot");
            assert_eq!(binding, live as *const Binding as usize, "slot {slot}");
            assert_eq!(read::<u8>(binding, layout.binding_init), 1);
            assert_eq!(read::<u8>(binding, layout.binding_mutable), 1);
            assert_eq!(read::<u8>(binding, layout.binding_import), 0);
            assert_eq!(layout.binding_value, Binding::value_offset());
            assert!((binding + layout.binding_value).is_multiple_of(8));
            assert!(layout.binding_value + 8 <= std::mem::size_of::<Binding>());
            assert_eq!(
                read::<u64>(binding, layout.binding_value),
                live.word().bits()
            );
            for flag in [
                layout.binding_init,
                layout.binding_mutable,
                layout.binding_import,
            ] {
                assert!(
                    !(layout.binding_value..layout.binding_value + 8).contains(&flag),
                    "the execution word must not overlap a binding flag"
                );
            }
        }
    };
    let parent = new_scope(None);
    assert_eq!(
        read::<usize>(Env::as_ptr(&parent) as usize, layout.scope_parent),
        0,
        "a root scope's parent is the handle's niche"
    );
    // One binding stays in the environment cell; more spill; past eight the map is Large.
    for count in [1usize, 2, 5, 9] {
        let env = new_scope(Some(parent.clone()));
        for slot in 0..count {
            env.borrow_mut().vars.insert(
                format!("b{slot}"),
                Binding::data(Value::Num(slot as f64), true, true),
            );
        }
        let base = Env::as_ptr(&env) as usize;
        {
            let scope = env.borrow();
            assert_eq!(read::<u32>(base, layout.scope_gen), scope.vars.generation());
            assert_eq!(
                read::<u32>(base, layout.scope_layout),
                scope.vars.layout_id()
            );
            assert!((base + layout.scope_with).is_multiple_of(8));
            assert!(
                layout.scope_with + 8
                    <= std::mem::size_of::<std::cell::RefCell<crate::interpreter::Scope>>()
            );
            assert_eq!(read::<u64>(base, layout.scope_with), layout.scope_with_none);
            assert_eq!(layout.scope_with_none, crate::value::PACK_EMPTY);
            assert_eq!(read::<u8>(base, layout.scope_under_with), 0);
        }
        let parent_word = read::<usize>(base, layout.scope_parent);
        assert_eq!(parent_word, env_stored(&parent));
        assert_eq!(parent_word, Env::as_ptr(&parent) as usize);
        let (tag, data, len) = entries(&env);
        if count > 8 {
            assert_eq!(tag, 1, "a Large map");
            continue;
        }
        assert_eq!((tag, len), (0, count), "a Small map");
        check_bindings(&env, data, count);
    }
    // An Indexed map shares the Small arm's entries.
    let env = new_scope(None);
    {
        let mut scope = env.borrow_mut();
        for name in ["x", "y", "z"] {
            scope
                .vars
                .insert(name, Binding::data(Value::Num(1.0), true, true));
        }
        scope.vars.share_layout(&std::cell::OnceCell::new());
    }
    let (tag, data, len) = entries(&env);
    assert_eq!((tag, len), (2, 3), "an Indexed map");
    check_bindings(&env, data, 3);
    assert_eq!(
        read::<u32>(Env::as_ptr(&env) as usize, layout.scope_layout),
        env.borrow().vars.layout_id()
    );
    // Presence is a complete-word test, including non-object and falsey synthetic records.
    let base = Env::as_ptr(&env) as usize;
    for value in [
        Value::Null,
        Value::Undefined,
        Value::Bool(false),
        Value::Num(-0.0),
    ] {
        env.borrow_mut().set_with_object(Some(value));
        assert!(env.borrow().has_with_object());
        assert_eq!(
            read::<u64>(base, layout.scope_with),
            crate::value::PackedValue::pack(value).bits()
        );
        assert_ne!(read::<u64>(base, layout.scope_with), layout.scope_with_none);
    }
    env.borrow_mut().set_with_object(None);
    assert!(!env.borrow().has_with_object());
    assert_eq!(
        read::<u64>(base, layout.scope_with),
        crate::value::PACK_EMPTY
    );
}

#[test]
fn property_layout_words_address_the_live_cell_payload() {
    let engine = crate::Engine::new();
    let layout = probe(&engine);
    let object = Object::new(None);
    object
        .borrow_mut()
        .props
        .insert("a", Property::plain(Value::Num(1.0)));
    let word = read::<usize>(
        body(&layout, &object),
        layout.obj_props + layout.props_layout,
    );
    let shared = object
        .borrow()
        .props
        .entries
        .layout
        .clone()
        .expect("a named map has a key layout");
    assert_eq!(word, read(&shared as *const PropertyLayout as usize, 0));
    assert_eq!(word, shared.as_ptr() as usize);
    assert!(shared.is_shared());
    let keys = word + layout.layout_data_off;
    assert_eq!(keys, &shared.keys as *const Vec<Rc<str>> as usize);
}

#[test]
fn callable_fields_read_a_live_closure() {
    let mut engine = crate::Engine::new();
    let layout = probe(&engine);
    let function = global(
        &mut engine,
        "function probe(a){return a;} var probeClosure = (function(k){return function(){return k;};})(1);",
        "probeClosure",
    );
    let call = body(&layout, &function) + layout.obj_call;
    assert_eq!(read::<u8>(call, 0), CALLABLE_USER_TAG);
    let payload = read::<usize>(call, 8);
    let borrowed = function.borrow();
    let Callable::User(user) = &borrowed.call else {
        panic!("a closure")
    };
    assert_eq!(payload, read(user as *const UserFn as usize, 0));
    assert_eq!(
        read::<usize>(payload, layout.user_func) + layout.func_data_off,
        Rc::as_ptr(&user.func) as usize
    );
    assert_eq!(
        read::<usize>(payload, layout.user_env),
        Env::as_ptr(&user.env) as usize
    );
    assert_eq!(read::<usize>(payload, layout.user_realm), user.realm);
    assert_eq!(
        read::<u8>(Env::as_ptr(&user.env) as usize, layout.scope_under_with),
        0
    );
}
