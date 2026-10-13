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
                    assert_eq!(props.elems.mirror.as_ptr(), mirror_pointer);
                    assert_eq!(props.mirror_holes, 0);
                    assert_ne!(props.mirror_flags & MIRROR_NO_HOLES, 0);
                    if packed && len > 1 {
                        assert_eq!(props.elems.packed_ref().map(|p| p.as_ptr()), pointer);
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
fn dense_shift_storage_moves_reference_owners_without_retain_or_loss() {
    for (packed, reverse, named, len) in [
        (false, false, false, 32),
        (false, true, true, 16),
        (true, false, false, 8),
        (true, false, true, 16),
    ] {
        let objects: Vec<_> = (0..len).map(|_| Object::new(None)).collect();
        let mut props = array(
            objects.iter().cloned().map(Value::Obj).collect(),
            packed,
            reverse,
            named,
        );
        assert!(objects.iter().all(|o| Rc::strong_count(o) == 2));
        for removed in 0..len {
            let first = props.shift_dense_array(len - removed).unwrap();
            assert!(matches!(&first, Value::Obj(o) if Rc::ptr_eq(o, &objects[removed])));
            for (index, object) in objects.iter().enumerate() {
                assert_eq!(
                    Rc::strong_count(object),
                    if index < removed { 1 } else { 2 }
                );
            }
            drop(first);
            assert_eq!(Rc::strong_count(&objects[removed]), 1);
        }
        drop(props);
        assert!(objects.iter().all(|o| Rc::strong_count(o) == 1));
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
