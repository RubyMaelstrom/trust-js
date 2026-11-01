//! Storage proof for ECMA-262 Array.prototype.shift, snapshot e28783d5.
use super::*;
use crate::value::ValueView;

fn array(values: Vec<Value>, packed: bool, reverse: bool, named_tail: bool) -> Props {
    let len = values.len();
    let mut properties = if packed {
        let mut words: Vec<_> = values.into_iter().collect();
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
    match (value).view() {
        ValueView::Num(value) => value,
        _ => panic!("expected a Number"),
    }
}

#[test]
fn dense_shift_storage_drains_words_and_classic_without_losing_views() {
    for packed in [false, true] {
        for named in [false, true] {
            for initial in [1, 3, 10, 16, 32] {
                let mut props = numbers(initial, packed, false, named);
                let shape = props.shape();
                // A shift moves the words' header one word on, never the words.
                let pointer = props.elems.words().map(|words| words.as_slice().as_ptr());
                for removed in 0..initial {
                    let len = initial - removed;
                    assert_eq!(
                        number(props.shift_dense_array(len, false).unwrap()),
                        removed as f64
                    );
                    assert_eq!(
                        number(props.length_property().unwrap().value()),
                        (len - 1) as f64
                    );
                    assert_eq!(props.shape(), shape);
                    assert!(props.get_index((len - 1) as u32).is_none());
                    if packed {
                        let words = props.elems.words().unwrap();
                        assert_eq!(
                            Some(words.as_slice().as_ptr()),
                            pointer.map(|p| unsafe { p.add(removed + 1) })
                        );
                        assert_eq!(words.front_room(), removed + 1);
                    }
                    for index in 0..len - 1 {
                        let expected = (removed + index + 1) as f64;
                        assert_eq!(
                            props.get_index(index as u32).unwrap().number_value(),
                            Some(expected)
                        );
                    }
                    if named {
                        assert!(
                            matches!((props.get("named").unwrap().value()).view(), ValueView::Str(s) if s.text() == "preserved")
                        );
                        assert_eq!(props.get("01").unwrap().number_value(), Some(71.0));
                        assert_eq!(props.get("4294967295").unwrap().number_value(), Some(72.0));
                    }
                }
                assert!(props.shift_dense_array(0, false).is_none());
                assert!(props
                    .try_append_element(0, Property::plain(Value::Num(42.0)))
                    .is_ok());
                props.get_mut("length").unwrap().set_value(Value::Num(1.0));
                assert_eq!(number(props.shift_dense_array(1, false).unwrap()), 42.0);
            }
        }
    }
}

#[test]
fn dense_shift_storage_preserves_out_of_order_slots_and_destination_attributes() {
    // Element words hold plain elements only: an attribute change moves the element to the
    // named entries, and the storage path then refuses, leaving the generic algorithm to run.
    let mut words = numbers(3, true, false, true);
    words.get_mut("0").unwrap().set_configurable(false);
    assert!(words.has_far.get());
    assert!(words.shift_dense_array(3, false).is_none());
    assert_eq!(words.get("0").unwrap().number_value(), Some(0.0));
    assert!(!words.get("0").unwrap().configurable());
    // The classic storage keeps each index's attributes in its entry.
    {
        let mut props = numbers(3, false, true, true);
        props.get_mut("0").unwrap().set_configurable(false);
        props.get_mut("0").unwrap().set_enumerable(false);
        props.get_mut("1").unwrap().set_configurable(false);
        props.get_mut("2").unwrap().set_writable(false);
        let zero_flags = props.get("0").unwrap().as_entry().meta;
        let one_flags = props.get("1").unwrap().as_entry().meta;
        let shape = props.shape();
        assert_eq!(number(props.shift_dense_array(3, false).unwrap()), 0.0);
        assert_eq!(props.get("0").unwrap().as_entry().meta, zero_flags);
        assert_eq!(props.get("1").unwrap().as_entry().meta, one_flags);
        assert_eq!(props.get("0").unwrap().number_value(), Some(1.0));
        assert_eq!(props.get("1").unwrap().number_value(), Some(2.0));
        assert!(props.get("2").is_none());
        assert_eq!(props.shape(), shape);
        assert_eq!(
            props
                .ordered_keys()
                .iter()
                .map(|k| k.text().to_string())
                .collect::<Vec<_>>(),
            ["0", "1", "length", "named", "01", "4294967295"]
        );
        assert!(
            props.shift_dense_array(2, false).is_none(),
            "non-configurable tail must reject"
        );
    }
}

#[test]
fn dense_shift_storage_rejections_are_atomic() {
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
    // Each own property's value word and attributes, as the element view reports them.
    let snapshot = |props: &Props| -> Vec<_> {
        props
            .ordered_keys()
            .into_iter()
            .map(|key| {
                let p = props.get_key(key).unwrap();
                let word = if p.accessor() { 0 } else { p.value().bits() };
                let attributes = [p.accessor(), p.writable(), p.enumerable(), p.configurable()];
                (key, word, attributes)
            })
            .collect()
    };
    for packed in [false, true] {
        for edit in edits {
            let mut props = numbers(3, packed, false, true);
            edit(&mut props);
            let before = snapshot(&props);
            let shape = props.shape();
            assert!(props.shift_dense_array(3, false).is_none());
            assert_eq!(snapshot(&props), before);
            assert_eq!(props.shape(), shape);
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
            let first = props.shift_dense_array(4 - removed, false).unwrap();
            assert!(proto_epoch() > epoch || epoch == u32::MAX);
            assert_eq!(number(first).to_bits(), bits);
            for n in 0..3 - removed {
                let element = props.get_index(n as u32).unwrap();
                assert_eq!(
                    element.number_value().unwrap().to_bits(),
                    expected[removed + n + 1]
                );
            }
        }
    }
}

#[test]
fn holes_and_nonplain_descriptors_keep_words_out_of_the_storage_paths() {
    let mut properties =
        Props::packed_array([Value::Num(1.0), Value::Empty, Value::Num(3.0)].into_iter());
    let words = properties.elems.words().unwrap();
    assert_eq!(words.kind() & word_block::KIND_NO_HOLES, 0);
    assert!(properties.shift_dense_array(3, false).is_none());
    assert!(!properties.unshift_dense_array(3, &[Value::Num(0.0)]));
    // Filling the hole leaves the kind bit clear; the storage paths prove it again.
    properties.insert("1", Property::plain(Value::Num(2.0)));
    assert!(properties.unshift_dense_array(3, &[Value::Empty]));
    assert!(properties.get_index(0).is_none());
    assert!(properties.shift_dense_array(4, false).is_none());
    // A non-plain element leaves the words for the named entries.
    let mut descriptor =
        Props::packed_array([Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)].into_iter());
    descriptor.insert("1", Property::data(Value::Num(2.0), false, true, true));
    assert!(descriptor.has_far.get());
    assert!(descriptor.elems.words().unwrap().get(1).is_none());
    let element = descriptor.get_index(1).expect("the named element");
    assert!(matches!(element, Element::Entry(_)) && !element.writable());
    assert!(descriptor.shift_dense_array(3, false).is_none());
}

/// Prepending one element at a time keeps front room ahead of the words, so the block moves
/// once per doubling; every element keeps its identity and order through the prepends and the
/// shifts that drain them.
#[test]
fn repeated_prepends_move_the_words_once_per_doubling() {
    let owners: Vec<_> = (0..100).map(|_| Object::new(None)).collect();
    let mut properties = Props::packed_array(std::iter::empty());
    let mut blocks = Vec::new();
    for (index, owner) in owners.iter().enumerate() {
        assert!(properties.unshift_dense_array(index, &[Value::Obj(*owner)]));
        let block = properties.elems.words().unwrap().block_start();
        if blocks.last() != Some(&block) {
            blocks.push(block);
        }
        for n in [0, index / 2, index] {
            assert!(
                matches!((properties.get_index(n as u32).unwrap().value()).view(), ValueView::Obj(value)
                if Gc::ptr_eq(&value, &owners[index-n]))
            );
        }
    }
    assert!(
        blocks.len() <= 6,
        "{} blocks for 100 prepends",
        blocks.len()
    );
    for index in (0..100).rev() {
        let first = properties.shift_dense_array(index + 1, false).unwrap();
        assert!(
            matches!(first.view(), ValueView::Obj(value) if Gc::ptr_eq(&value, &owners[index]))
        );
    }
}

#[test]
fn holes_shift_in_words_once_the_caller_proves_them_absent() {
    // Short (3) and longer (16) word arrays with a leading and an inner hole: without the
    // caller's proof the storage path refuses; with it each element and each hole moves down
    // one index, and the leading hole reads as undefined.
    for len in [3usize, 16] {
        let mut elements: Vec<Value> = (0..len).map(|n| Value::Num(n as f64)).collect();
        elements[0] = Value::Empty;
        elements[len / 2] = Value::Empty;
        let mut properties = Props::packed_array(elements.into_iter());
        assert!(properties.shift_dense_array(len, false).is_none());
        let first = properties.shift_dense_array(len, true).unwrap();
        assert!(matches!(first.view(), ValueView::Undefined));
        assert_eq!(
            properties.length_property().unwrap().number_value(),
            Some((len - 1) as f64)
        );
        for n in 0..len - 1 {
            let element = properties.get_index(n as u32);
            if n + 1 == len / 2 {
                assert!(element.is_none());
            } else {
                let element = element.unwrap();
                assert!(matches!(element, Element::Word(_)));
                assert_eq!(number(element.value()), (n + 1) as f64);
            }
        }
        assert!(properties.get_index((len - 1) as u32).is_none());
    }
    // An element whose attributes are not a plain element's still refuses.
    let mut descriptor = Props::packed_array([Value::Empty, Value::Num(2.0)].into_iter());
    descriptor.insert("1", Property::data(Value::Num(2.0), false, true, true));
    assert!(descriptor.shift_dense_array(2, true).is_none());
}

#[test]
fn cloned_words_change_independently() {
    for length in [3, 16, 100] {
        let original = numbers(length, true, false, false);
        let mut clone = original.clone();
        assert_ne!(
            original.elems.words().unwrap().block_start(),
            clone.elems.words().unwrap().block_start()
        );
        clone.insert("0", Property::data(Value::Num(0.0), false, true, true));
        assert!(clone.shift_dense_array(length, false).is_none());
        assert!(!original.has_far.get());
        assert!(original.get_index(0).unwrap().writable());
        assert!(matches!(original.get_index(0).unwrap(), Element::Word(_)));
    }
}
