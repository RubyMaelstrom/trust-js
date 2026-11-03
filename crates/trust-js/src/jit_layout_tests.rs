//! P1 S5 step 4: [`JitLayout`] v2 describes the build truthfully. Every exported offset is read
//! on a live structure through the raw offset and compared with the safe accessor: the
//! traced handles, in-cell named fields (in place and spilled), narrow slot memos, in-cell first
//! bindings and closure cells. Nothing here runs generated code.

use super::*;
use crate::interpreter::{new_scope, Binding, BindingSlot};
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
    let env = engine.interp.global_env;
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
    // Named slots (W2d step 2): seven words in the cell, then the record.
    assert_eq!(INLINE_NAMED_SLOTS, 7);
    assert_eq!(
        layout.props_slots + 8 * INLINE_NAMED_SLOTS,
        std::mem::size_of::<crate::value::Props>(),
        "the cell's slots end the map, and the 128-byte cell"
    );
    // A record's cell holds at least two slots in place (more in a larger cell).
    assert!(matches!(
        entries_of(&layout),
        JitVecLayout::Inline {
            inline_capacity: 2,
            ..
        }
    ));
}

#[test]
fn object_offsets_read_proto_shape_and_exotic_tags() {
    let mut engine = crate::Engine::new();
    let layout = probe(&engine);
    let proto = Object::new(None);
    let object = Object::new(Some(proto));
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
    // One flag byte: ordinary and extensible, nothing else.
    assert_eq!(
        read::<u8>(base, layout.obj_flags),
        crate::value::FLAG_IC_PLAIN | crate::value::FLAG_EXTENSIBLE
    );
    let tag = |object: &Gc| read::<u8>(body(&layout, object), layout.obj_kind);
    assert_eq!(tag(&object), 0, "a plain object's kind");
    for (exotic, expected) in [
        (ObjectKind::Array, layout.exotic_array_tag),
        (ObjectKind::str_wrap("s".into()), layout.exotic_strwrap_tag),
        (ObjectKind::NumWrap(2.5), layout.exotic_numwrap_tag),
        (ObjectKind::BoolWrap(true), layout.exotic_boolwrap_tag),
    ] {
        let wrapper = Object::new_with_parts(None, Props::new(), exotic);
        assert_eq!(tag(&wrapper), expected);
    }
    // Maps and Sets keep their data in the cell (`ObjectKind::Collection`), whose tag is the last
    // the ordinary-object guards admit: plain objects, typed arrays' views and the callables come
    // before it, every exotic kind after it.
    let map = global(&mut engine, "var probeMap = new Map([[1, 2]]);", "probeMap");
    let set = global(&mut engine, "var probeSet = new Set([3]);", "probeSet");
    let function = global(&mut engine, "var probeFn = function() {};", "probeFn");
    for collection in [&map, &set] {
        assert!(collection.borrow().kind.is_collection());
        assert_eq!(tag(collection), crate::value::KIND_ORDINARY_LAST);
    }
    let function_tag = tag(&function);
    assert_eq!(function_tag, crate::value::CALLABLE_USER_TAG);
    assert!(
        (crate::value::KIND_CALLABLE_FIRST..=crate::value::KIND_EXOTIC_NONE_LAST)
            .contains(&function_tag)
    );
    assert!(layout.exotic_array_tag > crate::value::KIND_ORDINARY_LAST);
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
        // W2d step 2: slots below INLINE_NAMED_SLOTS sit in the cell, from `props_entries`; the
        // rest in the record's block that `props_elems` points to, from RECORD_SLOTS_OFF.
        let base = body(&layout, &object) + layout.obj_props;
        let borrowed = object.borrow();
        let props = &borrowed.props;
        let record = read::<usize>(base, layout.props_record);
        assert_eq!(
            record != 0,
            count > INLINE_NAMED_SLOTS,
            "{count} fields: a record exactly past the cell's slots"
        );
        for slot in 0..count {
            let expected = if slot < INLINE_NAMED_SLOTS {
                base + layout.props_slots + slot * 8
            } else {
                record + crate::value::RECORD_SLOTS_OFF + (slot - INLINE_NAMED_SLOTS) * 8
            };
            assert_eq!(
                props.slot_address(slot),
                expected,
                "{count} fields: slot {slot}"
            );
            assert_eq!(
                read::<u64>(expected, 0),
                Value::Num(slot as f64).bits(),
                "{count} fields: slot {slot}"
            );
        }
    }
}

#[test]
fn scope_offsets_read_generation_parent_and_bindings() {
    let engine = crate::Engine::new();
    let layout = probe(&engine);
    let entries = |env: &Env| {
        let base = Env::as_ptr(env) as usize;
        let (data, len, _) = sequence(entries_of(&layout), base + layout.scope_small_vec);
        (data, len)
    };
    let check_bindings = |env: &Env, data: usize, count: usize| {
        let scope = env.borrow();
        for slot in 0..count {
            let binding = data + slot * layout.scope_binding_stride + layout.scope_binding_offset;
            let live = scope.vars.fixed_binding_at(slot).expect("fixed slot");
            assert_eq!(binding, live.as_ptr() as usize, "slot {slot}");
            // A slot is one word: its value, or the empty word in TDZ.
            assert_eq!(layout.binding_value, 0);
            assert_eq!(layout.scope_binding_stride, 8);
            assert_eq!(std::mem::size_of::<BindingSlot>(), 8);
            assert!((binding + layout.binding_value).is_multiple_of(8));
            assert_eq!(
                read::<u64>(binding, layout.binding_value),
                live.value().bits()
            );
            assert_ne!(live.value().bits(), crate::value::PACK_EMPTY);
        }
    };
    let parent = new_scope(None);
    assert_eq!(
        read::<usize>(Env::as_ptr(&parent) as usize, layout.scope_parent),
        0,
        "a root scope's parent is the handle's niche"
    );
    // A record's cell holds two slots in place; more spill; past eight the names are indexed,
    // and the values stay where generated code reads them.
    for count in [1usize, 2, 3, 5, 9] {
        let env = new_scope(Some(parent));
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
            // The names word: a record built name by name shares no descriptor.
            assert_eq!(scope.vars.layout_id(), 0);
            assert_ne!(read::<usize>(base, layout.scope_layout), 0);
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
        let (data, len) = entries(&env);
        assert_eq!(len, count);
        check_bindings(&env, data, count);
    }
    // A record that shares a descriptor keeps its values where generated code reads them.
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
    let (data, len) = entries(&env);
    assert_eq!(len, 3);
    assert!(env.borrow().vars.shared_desc().is_some());
    check_bindings(&env, data, 3);
    assert_eq!(
        read::<usize>(Env::as_ptr(&env) as usize, layout.scope_layout),
        env.borrow().vars.layout_id()
    );
    assert_ne!(env.borrow().vars.layout_id(), 0);
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
        assert_eq!(read::<u64>(base, layout.scope_with), (value).bits());
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
    assert_eq!(
        read::<u32>(word, layout.layout_keys_len) as usize,
        shared.len()
    );
    assert_eq!(
        read::<usize>(word, layout.layout_keys_ptr),
        <[crate::atom::PropKey]>::as_ptr(&shared) as usize
    );
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
    let call = body(&layout, &function) + layout.obj_kind;
    assert_eq!(read::<u8>(call, 0), CALLABLE_USER_TAG);
    let payload = read::<usize>(call, 8);
    let borrowed = function.borrow();
    let ObjectKind::User(user) = &borrowed.kind else {
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
