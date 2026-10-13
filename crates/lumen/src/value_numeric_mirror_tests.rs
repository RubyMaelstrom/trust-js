use super::*;

fn packed(length: usize) -> Props {
    let initial = length.min(32);
    let mut words: Vec<_> = (0..initial)
        .map(|index| PackedValue::pack(Value::Num(index as f64)))
        .collect();
    let mut properties = unsafe { Props::packed_array_from_raw(words.as_mut_ptr(), initial) };
    unsafe {
        words.set_len(0);
    }
    for index in initial..length {
        properties.push_dense(Property::plain(Value::Num(index as f64)));
    }
    if length != initial {
        properties
            .get_mut("length")
            .unwrap()
            .set_value(Value::Num(length as f64));
    }
    properties
}

#[test]
fn packed_mirror_storage_only_adds_numeric_words_and_preserves_canonical_addresses() {
    for length in [3, 16] {
        let mut props = packed(length);
        props.insert("named", Property::plain(Value::lstr("kept")));
        let before = props.retained_requested_storage_bytes().0;
        let pointer = props.elems.packed_ref().unwrap().as_ptr();
        let shape = props.shape();
        let length_slot = props.slot_of("length");
        let keys = props.ordered_keys();
        assert!(props.prepare_packed_numeric_mirror());
        assert_eq!(props.elems.packed_ref().unwrap().as_ptr(), pointer);
        assert_eq!(
            (props.shape(), props.slot_of("length")),
            (shape, length_slot)
        );
        assert_eq!(props.ordered_keys(), keys);
        assert_eq!(
            props.retained_requested_storage_bytes().0 - before,
            length * 8
        );
        assert_eq!(
            props.mirror_flags,
            MIRROR_OK | MIRROR_PACKED | MIRROR_NO_HOLES | MIRROR_ALL_I32
        );
        for index in 0..length {
            assert_eq!(props.mirror_get(index as u32), Some(index as f64));
        }
        let mirror_pointer = props.elems.mirror.as_ptr();
        assert!(props.prepare_packed_numeric_mirror());
        assert_eq!(props.elems.mirror.as_ptr(), mirror_pointer);
    }
}

#[test]
fn packed_mirror_mutable_escapes_and_structural_edits_invalidate_the_view() {
    let edits: [(&str, fn(&mut Props)); 8] = [
        ("get_mut", |p| {
            p.get_mut("0").unwrap().set_value(Value::Num(8.0));
        }),
        ("insert", |p| {
            p.insert("0", Property::plain(Value::Num(8.0)))
        }),
        ("append", |p| {
            assert!(p
                .try_append_element(3, Property::plain(Value::Num(8.0)))
                .is_ok())
        }),
        ("push", |p| p.push_dense(Property::plain(Value::Num(8.0)))),
        ("delete", |p| assert!(p.remove("0"))),
        ("pop", |p| {
            assert!(p.pop_last_element(2).is_some());
        }),
        ("truncate", |p| p.remove_indices_from(1)),
        ("non-number", |p| {
            assert!(p.set_index_value(0, Value::lstr("changed")).is_ok())
        }),
    ];
    for (name, edit) in edits {
        let mut props = packed(3);
        assert!(props.prepare_packed_numeric_mirror());
        edit(&mut props);
        if dense_elements_enabled() && matches!(name, "append" | "push") {
            assert_eq!(props.mirror_get(0), Some(0.0), "{name}");
            assert_eq!(props.mirror_get(3), Some(8.0), "{name}");
            assert_eq!(props.get_index(3).unwrap().number_value(), Some(8.0));
            continue;
        }
        assert_eq!(props.mirror_flags & MIRROR_OK, 0, "{name}");
        assert!(props.mirror_get(0).is_none(), "{name}");
    }
}

#[test]
fn packed_mirror_numbers_preserve_bits_and_refused_states_recover_after_mutation() {
    let mut props = packed(3);
    assert!(props.prepare_packed_numeric_mirror());
    for number in [0.5, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(props.set_index_value(0, Value::Num(number)).is_ok());
        assert_eq!(props.mirror_get(0).unwrap().to_bits(), number.to_bits());
        assert_eq!(props.mirror_flags & MIRROR_ALL_I32, 0);
    }
    props
        .get_mut("0")
        .unwrap()
        .set_value(Value::lstr("not numeric"));
    let before = props.retained_requested_storage_bytes().0;
    assert!(!props.prepare_packed_numeric_mirror());
    assert_eq!(props.mirror_flags, MIRROR_PACKED_FAILED);
    assert!(!props.prepare_packed_numeric_mirror());
    assert_eq!(props.retained_requested_storage_bytes().0, before);
    assert!(props.set_index_value(0, Value::Num(7.0)).is_ok());
    assert!(props.prepare_packed_numeric_mirror());
    assert_eq!(props.mirror_get(0), Some(7.0));
    props.get_mut("0").unwrap().set_writable(false);
    assert!(!props.prepare_packed_numeric_mirror());
    props.get_mut("0").unwrap().set_writable(true);
    assert!(props.prepare_packed_numeric_mirror());
    props.remove("1");
    assert!(!props.prepare_packed_numeric_mirror());
    props.insert("1", Property::plain(Value::Num(1.0)));
    assert!(props.prepare_packed_numeric_mirror());
    props.clear();
    assert!(!props.prepare_packed_numeric_mirror());
    assert!(props.values().next().is_none());
}

#[test]
fn packed_mirror_clone_is_independent_and_preparation_work_is_bounded() {
    let mut original = packed(3);
    original.insert("named", Property::plain(Value::lstr("kept")));
    assert!(original.prepare_packed_numeric_mirror());
    let mut copy = original.clone();
    assert_ne!(original.elems.mirror.as_ptr(), copy.elems.mirror.as_ptr());
    assert!(copy.set_index_value(0, Value::Num(0.5)).is_ok());
    assert_eq!(original.mirror_get(0), Some(0.0));
    assert_eq!(copy.mirror_get(0), Some(0.5));
    original
        .get_mut("1")
        .unwrap()
        .set_value(Value::lstr("changed"));
    drop(original);
    assert_eq!(copy.mirror_get(1), Some(1.0));
    assert!(matches!(copy.get("named").unwrap().value(), Value::Str(value) if &*value == "kept"));

    let mut maximum = packed(4096);
    assert!(maximum.prepare_packed_numeric_mirror());
    assert_eq!(maximum.elems.mirror.len(), 4096);
    let mut oversized = packed(4097);
    let before = oversized.retained_requested_storage_bytes().0;
    let pointer = oversized.elems.packed_ref().unwrap().as_ptr();
    for _ in 0..3 {
        assert!(!oversized.prepare_packed_numeric_mirror());
        assert_eq!(oversized.elems.packed_ref().unwrap().as_ptr(), pointer);
        assert_eq!(oversized.retained_requested_storage_bytes().0, before);
        assert_eq!(oversized.mirror_flags, MIRROR_PACKED_FAILED);
    }
}
