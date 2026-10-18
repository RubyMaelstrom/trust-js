//! Storage proof for ECMA-262 Array.prototype.shift, snapshot e28783d5.
use super::*;

fn array(values: Vec<Value>, packed: bool, reverse: bool, named_tail: bool) -> Props {
    let len = values.len();
    let mut properties = if packed {
        let mut words: Vec<_> = values.into_iter().map(PackedValue::pack).collect();
        let props = unsafe { Props::packed_array_from_raw(words.as_mut_ptr(), len) };
        unsafe { words.set_len(0) };
        props
    } else {
        let mut props = Props::new();
        props.mark_array();
        props.insert(
            "length",
            Property::data(Value::Num(len as f64), true, false, false),
        );
        let mut entries: Vec<_> = values.into_iter().enumerate().collect();
        if reverse {
            entries.reverse();
        }
        for (index, value) in entries {
            // Exercise the retained classic fallback explicitly: ordinary array
            // insertion now chooses contiguous elements for this fixture too.
            let slot = props.entries.len();
            props
                .entries
                .push((index_key(index), Property::plain(value)));
            props.note_inserted(slot);
        }
        props
    };
    if named_tail {
        properties.insert("named", Property::plain(Value::lstr("preserved")));
        properties.insert("01", Property::plain(Value::Num(71.0)));
        properties.insert("4294967295", Property::plain(Value::Num(72.0)));
    }
    properties
}

fn numbers(len: usize, packed: bool, reverse: bool, named_tail: bool) -> Props {
    array(
        (0..len).map(|n| Value::Num(n as f64)).collect(),
        packed,
        reverse,
        named_tail,
    )
}

fn number(value: Value) -> f64 {
    match value {
        Value::Num(value) => value,
        _ => panic!("expected a Number"),
    }
}

#[test]
fn dense_shift_storage_drains_inline_packed_and_classic_without_losing_views() {
    for packed in [false, true] {
        for named in [false, true] {
            for initial in [1, 3, 10, 16, 32] {
                let mut props = numbers(initial, packed, false, named);
                if packed {
                    assert!(props.prepare_packed_numeric_mirror());
                }
                let shape = props.shape();
                let inline = packed && initial <= INLINE_PACKED_CAPACITY;
                let pointer = props.elems.packed_ref().map(|p| p.as_ptr());
                let mirror_pointer = props.elems.mirror.as_ptr();
                for removed in 0..initial {
                    let len = initial - removed;
                    assert_eq!(
                        number(props.shift_dense_array(len).unwrap()),
                        removed as f64
                    );
                    assert_eq!(
                        number(props.length_property().unwrap().value()),
                        (len - 1) as f64
                    );
                    assert_eq!(props.shape(), shape);
                    assert!(props.get_index((len - 1) as u32).is_none());
                    assert_eq!(props.elems.mirror_len(), len - 1);
                    let mirror_front = if packed { removed + 1 } else { 0 };
                    assert_eq!(props.elems.mirror.as_ptr(), unsafe {
                        mirror_pointer.add(mirror_front)
                    });
                    assert_eq!(props.mirror_holes, 0);
                    assert_ne!(props.mirror_flags & MIRROR_NO_HOLES, 0);
                    if packed && len > 1 {
                        let expected = pointer.map(|p| {
                            if inline {
                                p
                            } else {
                                unsafe { p.add(removed + 1) }
                            }
                        });
                        assert_eq!(props.elems.packed_ref().map(|p| p.as_ptr()), expected);
                    }
                    if inline {
                        assert!(
                            props.elems.packed.is_none(),
                            "shift promoted inline storage"
                        );
                    }
                    for index in 0..len - 1 {
                        let expected = (removed + index + 1) as f64;
                        assert_eq!(
                            props.get_index(index as u32).unwrap().number_value(),
                            Some(expected)
                        );
                        assert_eq!(props.mirror_get(index as u32), Some(expected));
                    }
                    if named {
                        assert!(
                            matches!(props.get("named").unwrap().value(), Value::Str(s) if &*s == "preserved")
                        );
                        assert_eq!(props.get("01").unwrap().number_value(), Some(71.0));
                        assert_eq!(props.get("4294967295").unwrap().number_value(), Some(72.0));
                    }
                }
                assert!(props.shift_dense_array(0).is_none());
                assert!(props
                    .try_append_element(0, Property::plain(Value::Num(42.0)))
                    .is_ok());
                props.get_mut("length").unwrap().set_value(Value::Num(1.0));
                assert_eq!(number(props.shift_dense_array(1).unwrap()), 42.0);
            }
        }
    }
}

#[test]
fn dense_shift_storage_preserves_out_of_order_slots_and_destination_attributes() {
    for packed in [false, true] {
        let mut props = numbers(3, packed, !packed, true);
        props.get_mut("0").unwrap().set_configurable(false);
        props.get_mut("0").unwrap().set_enumerable(false);
        props.get_mut("1").unwrap().set_configurable(false);
        props.get_mut("2").unwrap().set_writable(false);
        let zero_flags = props.get("0").unwrap().meta;
        let one_flags = props.get("1").unwrap().meta;
        let shape = props.shape();
        assert_eq!(number(props.shift_dense_array(3).unwrap()), 0.0);
        assert_eq!(props.get("0").unwrap().meta, zero_flags);
        assert_eq!(props.get("1").unwrap().meta, one_flags);
        assert_eq!(props.get("0").unwrap().number_value(), Some(1.0));
        assert_eq!(props.get("1").unwrap().number_value(), Some(2.0));
        assert!(props.get("2").is_none());
        assert_eq!(props.shape(), shape);
        assert_eq!(
            props
                .ordered_keys()
                .iter()
                .map(|k| &**k)
                .collect::<Vec<_>>(),
            ["0", "1", "length", "named", "01", "4294967295"]
        );
        assert!(
            props.shift_dense_array(2).is_none(),
            "non-configurable tail must reject"
        );
    }
}

#[test]
fn dense_shift_storage_rejections_are_atomic_and_do_not_invalidate_mirrors() {
    let edits: [fn(&mut Props); 10] = [
        |p| {
            p.get_mut("0").unwrap().set_writable(false);
        },
        |p| {
            p.get_mut("2").unwrap().set_configurable(false);
        },
        |p| {
            p.remove("1");
        },
        |p| {
            p.insert("1", Property::accessor_prop(None, None, true, true));
        },
        |p| {
            p.insert("1", Property::plain(Value::Empty));
        },
        |p| {
            p.get_mut("length").unwrap().set_writable(false);
        },
        |p| {
            p.get_mut("length").unwrap().set_value(Value::Num(4.0));
        },
        |p| {
            p.has_far.set(true);
        },
        |p| {
            p.elem_mode.set(false);
        },
        |p| {
            p.insert("length", Property::accessor_prop(None, None, false, false));
        },
    ];
    for packed in [false, true] {
        for edit in edits {
            let mut props = numbers(3, packed, false, true);
            if packed {
                assert!(props.prepare_packed_numeric_mirror());
            }
            edit(&mut props);
            let before: Vec<_> = props
                .ordered_keys()
                .into_iter()
                .map(|key| {
                    let p = props.get(&key).unwrap();
                    (key, p.packed.0.get(), p.meta)
                })
                .collect();
            let mirror: Vec<_> = props.elems.mirror.iter().map(|f| f.to_bits()).collect();
            let flags = props.mirror_flags;
            let pointer = props.elems.mirror.as_ptr();
            let shape = props.shape();
            assert!(props.shift_dense_array(3).is_none());
            for (key, bits, meta) in before {
                let p = props.get(&key).unwrap();
                assert_eq!((p.packed.0.get(), p.meta), (bits, meta));
            }
            assert_eq!(
                props
                    .elems
                    .mirror
                    .iter()
                    .map(|f| f.to_bits())
                    .collect::<Vec<_>>(),
                mirror
            );
            assert_eq!(
                (
                    props.mirror_flags,
                    props.elems.mirror.as_ptr(),
                    props.shape()
                ),
                (flags, pointer, shape)
            );
        }
    }
}

#[test]
fn dense_shift_storage_preserves_numeric_bits_and_invalidates_prototype_creation_proofs() {
    for packed in [false, true] {
        let values = vec![
            Value::Num(-0.0),
            Value::Num(f64::NAN),
            Value::Num(f64::INFINITY),
            Value::Num(1.5),
        ];
        let mut props = array(values, packed, false, true);
        if packed {
            assert!(props.prepare_packed_numeric_mirror());
        }
        let expected: Vec<_> = (0..4)
            .map(|n| {
                props
                    .get_index(n)
                    .unwrap()
                    .number_value()
                    .unwrap()
                    .to_bits()
            })
            .collect();
        props.mark_proto();
        for (removed, &bits) in expected.iter().enumerate() {
            let epoch = proto_epoch();
            let first = props.shift_dense_array(4 - removed).unwrap();
            assert!(proto_epoch() > epoch || epoch == u32::MAX);
            assert_eq!(number(first).to_bits(), bits);
            for n in 0..3 - removed {
                assert_eq!(
                    props.mirror_get(n as u32).unwrap().to_bits(),
                    expected[removed + n + 1]
                );
            }
        }
    }
}

#[test]
fn packed_creation_and_prepend_do_not_certify_holes_or_nonplain_descriptors() {
    let mut properties = Props::packed_array_properties(
        [
            Property::plain(Value::Num(1.0)),
            Property::plain(Value::Empty),
            Property::plain(Value::Num(3.0)),
        ]
        .into_iter(),
    );
    assert!(!properties.elems.inline_packed.packed_plain.get());
    assert!(properties.shift_dense_array(3).is_none());
    assert!(!properties.unshift_dense_array(3, &[Value::Num(0.0)]));
    properties.insert("1", Property::plain(Value::Num(2.0)));
    assert!(properties.unshift_dense_array(3, &[Value::Empty]));
    assert!(!properties.elems.inline_packed.packed_plain.get());
    assert!(properties.shift_dense_array(4).is_none());
    let mut descriptor = Props::packed_array_properties(
        [
            Property::plain(Value::Num(1.0)),
            Property::data(Value::Num(2.0), false, true, true),
            Property::plain(Value::Num(3.0)),
        ]
        .into_iter(),
    );
    assert!(!descriptor.elems.inline_packed.packed_plain.get());
    assert!(descriptor.shift_dense_array(3).is_none());
}

/// Prepending one element at a time keeps a packed array inline up to its inline capacity and
/// promotes it once past it; every element keeps its identity and order through the prepends
/// and the shifts that drain them.
#[test]
fn tiny_prepends_stay_inline_before_promoting_once() {
    let owners: Vec<_> = (0..12).map(|_| Object::new(None)).collect();
    let mut properties = Props::packed_array_from_values(Vec::new());
    for (index, owner) in owners.iter().enumerate() {
        assert!(properties.unshift_dense_array(index, &[Value::Obj(owner.clone())]));
        assert_eq!(
            properties.elems.packed.is_none(),
            index < INLINE_PACKED_CAPACITY
        );
        for n in 0..=index {
            assert!(
                matches!(properties.get_index(n as u32).unwrap().value(), Value::Obj(value)
                if Gc::ptr_eq(&value, &owners[index-n]))
            );
        }
    }
    for index in (0..12).rev() {
        let first = properties.shift_dense_array(index + 1).unwrap();
        assert!(matches!(&first, Value::Obj(value) if Gc::ptr_eq(value, &owners[index])));
    }
}

#[test]
fn packed_plain_proof_uses_existing_inline_padding_and_clones_independently() {
    #[allow(dead_code)]
    struct PreviousInlinePacked {
        len: u8,
        slots: [std::mem::MaybeUninit<Property>; INLINE_PACKED_CAPACITY],
    }
    assert_eq!(
        std::mem::size_of::<InlinePacked>(),
        std::mem::size_of::<PreviousInlinePacked>()
    );
    assert_eq!(
        std::mem::offset_of!(InlinePacked, len),
        std::mem::offset_of!(PreviousInlinePacked, len)
    );
    assert_eq!(
        std::mem::offset_of!(InlinePacked, slots),
        std::mem::offset_of!(PreviousInlinePacked, slots)
    );
    for length in [3, 16] {
        let original = numbers(length, true, false, false);
        let mut clone = original.clone();
        assert!(original.elems.inline_packed.packed_plain.get());
        assert!(clone.elems.inline_packed.packed_plain.get());
        clone.elems.packed_slice_mut().unwrap()[0] =
            Property::data(Value::Num(0.0), false, true, true);
        assert!(!clone.elems.inline_packed.packed_plain.get());
        assert!(clone.shift_dense_array(length).is_none());
        assert!(original.elems.inline_packed.packed_plain.get());
        assert!(original.get_index(0).unwrap().writable());
    }
}
