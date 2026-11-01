//! Number arrays in element words: the words are the f64 view, and their kind bits stay exact
//! through every edit.
use super::*;
use crate::value::ValueView;

fn packed(length: usize) -> Props {
    let initial = length.min(32);
    let mut words: Vec<_> = (0..initial).map(|index| Value::Num(index as f64)).collect();
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

/// The kind bits a scan of the words proves, which the header's bits never exceed.
fn exact_kind(props: &Props) -> u32 {
    let words = props.elems.words().expect("element words");
    assert_eq!(
        words.kind() & !words.computed_kind(),
        0,
        "kind bits beyond the words"
    );
    words.kind()
}

#[test]
fn number_words_keep_their_kind_bits() {
    for length in [3, 16, 4097] {
        let mut props = packed(length);
        props.insert("named", Property::plain(Value::lstr("kept")));
        let before = props.retained_requested_storage_bytes().0;
        let pointer = props.elems.words().unwrap().as_slice().as_ptr();
        let shape = props.shape();
        assert_eq!(props.elems.words().unwrap().as_slice().as_ptr(), pointer);
        assert_eq!(props.shape(), shape);
        assert_eq!(props.retained_requested_storage_bytes().0, before);
        assert_eq!(
            exact_kind(&props),
            word_block::KIND_NO_HOLES | word_block::KIND_NUMBERS | word_block::KIND_INT32
        );
        for index in [0, length / 2, length - 1] {
            assert_eq!(
                props.get_index(index as u32).unwrap().number_value(),
                Some(index as f64)
            );
        }
    }
}

#[test]
fn word_edits_keep_the_kind_bits_exact() {
    let edits: [(&str, fn(&mut Props)); 9] = [
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
        ("push", |p| p.push_dense(Property::plain(Value::Num(8.5)))),
        ("delete", |p| assert!(p.remove("0"))),
        ("pop", |p| {
            assert!(p.pop_last_element(2).is_some());
        }),
        ("truncate", |p| p.remove_indices_from(1)),
        ("non-number", |p| {
            assert!(p.set_index_value(0, Value::lstr("changed")).is_ok())
        }),
        ("hole", |p| p.insert("1", Property::plain(Value::Empty))),
    ];
    for (name, edit) in edits {
        let mut props = packed(3);
        edit(&mut props);
        let kind = exact_kind(&props);
        let expected = match name {
            "push" => word_block::KIND_NO_HOLES | word_block::KIND_NUMBERS,
            "non-number" => word_block::KIND_NO_HOLES,
            "delete" | "hole" => word_block::KIND_NUMBERS | word_block::KIND_INT32,
            // The escape moved element 0 to the named entries, its word a hole.
            "get_mut" => word_block::KIND_NUMBERS | word_block::KIND_INT32,
            _ => word_block::KIND_ALL,
        };
        assert_eq!(kind, expected, "{name}");
    }
}

#[test]
fn number_words_preserve_bits_and_their_kind_comes_back_only_when_proven() {
    let mut props = packed(3);
    for number in [0.5, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(props.set_index_value(0, Value::Num(number)).is_ok());
        assert_eq!(
            props
                .get_index(0)
                .unwrap()
                .number_value()
                .unwrap()
                .to_bits(),
            number.to_bits()
        );
        assert_eq!(exact_kind(&props) & word_block::KIND_INT32, 0);
    }
    assert!(props.set_index_value(0, Value::lstr("not numeric")).is_ok());
    assert_eq!(exact_kind(&props), word_block::KIND_NO_HOLES);
    // The bits are clear-only: a Number written back leaves them clear until a proof.
    assert!(props.set_index_value(0, Value::Num(7.0)).is_ok());
    assert_eq!(exact_kind(&props), word_block::KIND_NO_HOLES);
    let words = props.elems.words_mut().unwrap();
    assert_eq!(words.reprove(), word_block::KIND_ALL);
    assert_eq!(exact_kind(&props), word_block::KIND_ALL);
    props.clear();
    assert!(props.values().next().is_none());
}

#[test]
fn cloned_number_words_are_independent() {
    let mut original = packed(3);
    original.insert("named", Property::plain(Value::lstr("kept")));
    let mut copy = original.clone();
    assert_ne!(
        original.elems.words().unwrap().as_slice().as_ptr(),
        copy.elems.words().unwrap().as_slice().as_ptr()
    );
    assert!(copy.set_index_value(0, Value::Num(0.5)).is_ok());
    assert_eq!(original.get_index(0).unwrap().number_value(), Some(0.0));
    assert_eq!(copy.get_index(0).unwrap().number_value(), Some(0.5));
    assert_eq!(exact_kind(&original), word_block::KIND_ALL);
    assert!(original.set_index_value(1, Value::lstr("changed")).is_ok());
    drop(original);
    assert_eq!(copy.get_index(1).unwrap().number_value(), Some(1.0));
    assert!(
        matches!((copy.get("named").unwrap().value()).view(), ValueView::Str(value) if value.text() == "kept")
    );
}
