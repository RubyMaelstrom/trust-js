//! Collector-only remembering of element words (the word block's record); ordinary language
//! semantics are unchanged.
use super::word_block::Words;
use super::*;
use crate::gc_core::{CellAge, CollectionKind, TracePurpose};
use crate::gc_trace::{GcSink, GcTrace};
use crate::rooting::Rooted;
use crate::{Completion, Engine};

struct Ignore;
impl GcSink for Ignore {
    fn word(&mut self, _: u64) {}
}

fn visits() -> [usize; 2] {
    word_block::MARK_VISITS.with(Cell::get)
}

fn selected(words: &Words, purpose: TracePurpose) -> usize {
    let before = visits();
    words.trace_values(Some(purpose), &mut Ignore);
    let after = visits();
    after[0] + after[1] - before[0] - before[1]
}

/// `len` Null words in a block with room for `capacity`.
fn nulls(len: usize, capacity: usize) -> Words {
    let mut words = Words::with_capacity(capacity);
    for _ in 0..len {
        words.push(Value::Null);
    }
    words
}

fn clean(words: &Words) {
    assert_eq!(selected(words, TracePurpose::Full), words.len());
    words.finish_mark_trace();
}

fn elements(props: &Props) -> &Words {
    props.elems.words().expect("element words")
}

fn remembered(array: &Gc) -> Option<(Option<usize>, Vec<usize>)> {
    elements(&array.borrow().props).remembered()
}

fn object(value: Value) -> Gc {
    value.as_obj().expect("fixture object")
}

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

#[test]
fn storage_carriers_are_plain_data_with_the_native_header_first() {
    // The key index and the element words live in storage blocks (`gc_storage`): plain data, so
    // that dropping a map never reads a block, which may have gone with its heap.
    assert!(!std::mem::needs_drop::<KeyIndex>());
    assert!(!std::mem::needs_drop::<Words>());
    // The map keeps one word for its element words (W2d step 2: the key index and the Rust-heap
    // extras moved to the named entries' out-of-line record, a nullable pointer).
    assert_eq!(
        std::mem::size_of::<DenseStorage>(),
        std::mem::size_of::<usize>()
    );
    assert_eq!(std::mem::size_of::<Record>(), std::mem::size_of::<usize>());
    // The header just before the first word, where generated code reads `used` at -16,
    // `capacity` at -12 and `flags` at -4.
    assert_eq!(std::mem::size_of::<word_block::WordHeader>(), 16);
    assert_eq!(std::mem::offset_of!(word_block::WordHeader, used), 0);
    assert_eq!(std::mem::offset_of!(word_block::WordHeader, capacity), 4);
    assert_eq!(std::mem::offset_of!(word_block::WordHeader, flags), 12);
}

#[test]
fn distant_dirty_blocks_and_partial_prefix_tail_are_bounded_without_duplicates() {
    // Room for the appends below: a grown block starts full.
    let mut words = nulls(200_003, 200_008);
    clean(&words);
    assert_eq!(words.remembered(), Some((Some(200_003), Vec::new())));
    // A Number or a hole is never an edge: writing one marks nothing.
    words.set(1000, Value::Num(1000.0));
    words.set(1001, Value::Empty);
    assert_eq!(words.remembered(), Some((Some(200_003), Vec::new())));
    for index in [63, 64, 65, 2000, 200_002] {
        words.set(index, Value::Bool(true));
    }
    for _ in 0..5 {
        words.push(Value::Null);
    }
    // Three complete blocks, the three-word partial block, and the five new tail words.
    assert_eq!(selected(&words, TracePurpose::Remembered), 3 * 64 + 3 + 5);
    assert_eq!(
        words.remembered(),
        Some((Some(200_003), vec![0, 1, 31, 3125])),
        "reporting alone does not consume"
    );
    words.finish_mark_trace();
    assert_eq!(selected(&words, TracePurpose::Remembered), 0);
    assert_eq!(words.remembered(), Some((Some(200_008), Vec::new())));
}

#[test]
fn shrink_same_length_regrowth_preserves_old_dirty_blocks_and_new_tail() {
    let mut words = nulls(130, 130);
    clean(&words);
    words.set(1, Value::Bool(true));
    assert_eq!(words.pop(), Some(Value::Null.bits()));
    assert_eq!(words.remembered(), Some((Some(129), vec![0])));
    words.push(Value::Bool(false));
    // Block 0 and the word past the lowered traced length.
    assert_eq!(selected(&words, TracePurpose::Remembered), 65);
    words.finish_mark_trace();
    words.set(1, Value::Bool(false));
    words.set(100, Value::Bool(false));
    words.truncate(40);
    assert_eq!(words.remembered(), Some((Some(40), vec![0, 1])));
    words.push(Value::Null);
    // Block 0 up to the lowered traced length and the new word past it, one run; block 1 is
    // past the length.
    assert_eq!(selected(&words, TracePurpose::Remembered), 41);
}

#[test]
fn batch_edits_stay_bounded_but_reorders_copies_and_map_moves_start_full() {
    let mut properties = Props::packed_array(std::iter::repeat_n(Value::Null, 130));
    clean(elements(&properties));
    // A copy is a fresh block, whose record starts full.
    let clone = properties.clone();
    assert_eq!(elements(&clone).remembered(), Some((None, Vec::new())));
    // A batch edit writes through the words: a set marks its block, and a splice lowers the
    // traced length to its start.
    properties
        .plain_elements_mut(130)
        .unwrap()
        .set(1, Value::Bool(true));
    assert_eq!(
        selected(elements(&properties), TracePurpose::Remembered),
        64
    );
    clean(elements(&properties));
    properties
        .plain_elements_mut(130)
        .unwrap()
        .splice(1..2, &[Value::Bool(false)]);
    assert_eq!(
        selected(elements(&properties), TracePurpose::Remembered),
        129
    );
    clean(elements(&properties));
    properties.plain_elements_mut(130).unwrap().reverse();
    assert_eq!(
        selected(elements(&properties), TracePurpose::Remembered),
        130
    );
    clean(elements(&properties));
    assert!(properties.shift_dense_array(130, false).is_some());
    assert_eq!(
        selected(elements(&properties), TracePurpose::Remembered),
        129
    );
    clean(elements(&properties));
    assert!(properties.unshift_dense_array(129, &[Value::Null]));
    assert_eq!(
        selected(elements(&properties), TracePurpose::Remembered),
        130
    );
    clean(elements(&properties));
    // The record of a map moved to another object described its old owner's marks.
    let array = Object::new_with_parts(None, properties, ObjectKind::Array);
    assert_eq!(remembered(&array), Some((None, Vec::new())));
}

#[test]
fn indexed_setters_hole_fill_and_gap_appends_are_bounded() {
    let mut properties = Props::packed_array(std::iter::repeat_n(Value::Null, 2000));
    for operation in 0..3 {
        clean(elements(&properties));
        match operation {
            0 => assert!(properties.set_index_value(65, Value::Bool(true)).is_ok()),
            1 => assert!(properties.set_value("65", Value::Bool(false))),
            _ => properties.insert("65", Property::plain(Value::Undefined)),
        }
        assert_eq!(
            selected(elements(&properties), TracePurpose::Remembered),
            64
        );
    }
    // A deletion writes a hole, which is no edge.
    clean(elements(&properties));
    assert!(properties.remove("65"));
    assert_eq!(selected(elements(&properties), TracePurpose::Remembered), 0);
    // Filling the hole writes its word.
    assert!(properties
        .try_define_dense_element(65, Property::plain(Value::Null))
        .is_ok());
    assert_eq!(
        selected(elements(&properties), TracePurpose::Remembered),
        64
    );
    // A gap append past the capacity grows the block, which keeps its record: the holes and
    // the word past the traced length.
    clean(elements(&properties));
    assert!(properties
        .try_define_dense_element(2010, Property::plain(Value::Null))
        .is_ok());
    assert_eq!(
        selected(elements(&properties), TracePurpose::Remembered),
        11
    );
    // One within the capacity writes its holes and its word past the traced length.
    clean(elements(&properties));
    assert!(properties
        .try_define_dense_element(2020, Property::plain(Value::Null))
        .is_ok());
    assert_eq!(
        selected(elements(&properties), TracePurpose::Remembered),
        10
    );
}

#[test]
fn precise_descriptor_replacement_still_invalidates_the_plain_element_proof() {
    let mut properties = Props::packed_array(std::iter::repeat_n(Value::Null, 128));
    clean(elements(&properties));
    assert!(properties.element_words().is_some());
    properties.insert("65", Property::accessor_prop(None, None, true, true));
    // The accessor leaves the words for the named entries, its word a hole.
    assert!(properties.has_far() && properties.element_words().is_none());
    assert!(elements(&properties).get(65).is_none());
    assert!(properties.get("65").is_some_and(Element::accessor));
    assert!(properties.shift_dense_array(128, false).is_none());
    assert!(!properties.unshift_dense_array(128, &[Value::Null]));
    // The hole marks nothing; the named entries are traced in full.
    assert_eq!(selected(elements(&properties), TracePurpose::Remembered), 0);
}

fn witness(value: Value) -> u64 {
    value.cell_word().expect("GC value")
}

fn empty_native(_: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    Ok(Value::Undefined)
}

fn assert_live(engine: &Engine, words: &[u64], age: CellAge) {
    let core = engine.interp.gc_heap.core.borrow();
    for &word in words {
        assert_eq!(core.cell_age(word), Some(age), "cell {word:#x}");
    }
}

fn assert_labeled_live(engine: &Engine, words: &[(&str, u64)], age: CellAge) {
    let core = engine.interp.gc_heap.core.borrow();
    for &(label, word) in words {
        assert_eq!(core.cell_age(word), Some(age), "{label}: {word:#x}");
    }
}

fn assert_symbol_pin_live(pin: &WeakSymbol, word: u64) {
    assert_eq!(
        pin.upgrade().map(|symbol| witness(Value::Sym(symbol))),
        Some(word)
    );
}

fn assert_dead_then_drop_symbol_pin_and_reclaim(
    engine: &mut Engine,
    words: &[(&str, u64)],
    pin: WeakSymbol,
    symbol_word: u64,
) {
    assert_symbol_pin_live(&pin, symbol_word);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    {
        let core = engine.interp.gc_heap.core.borrow();
        for &(label, word) in words {
            assert!(
                matches!(core.cell_age(word), None | Some(CellAge::Dead)),
                "first major retained a live {label}: {word:#x}, age={:?}",
                core.cell_age(word)
            );
        }
        assert_eq!(
            core.cell_age(symbol_word),
            Some(CellAge::Dead),
            "the explicitly owned weak pin must reserve the dead Symbol"
        );
    }
    assert!(
        pin.upgrade().is_none(),
        "first major must clear the weak handle"
    );
    assert_eq!(pin.strong_count(), 0);
    // The Agent's identity index may already be pruned by post-sweep bookkeeping. This
    // independently captured handle deliberately controls the reservation lifetime instead.
    drop(pin);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    let core = engine.interp.gc_heap.core.borrow();
    for &(label, word) in words {
        assert_eq!(
            core.cell_age(word),
            None,
            "storage remained after pin release: {label} {word:#x}"
        );
    }
}

#[test]
fn actual_marker_keeps_sole_young_trees_in_distant_blocks_and_appended_tail() {
    let mut engine = Engine::new();
    let _agent = engine.interp.enter_agent();
    let value = engine.interp.make_array(vec![Value::Null; 200_003]);
    let array = object(value);
    let owner = Rooted::new(value);
    // Room for the appends below, which then keep the block and its record.
    array
        .borrow_mut()
        .props
        .elems
        .words_mut()
        .unwrap()
        .reserve_exact(5);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    assert_eq!(remembered(&array), Some((Some(200_003), Vec::new())));
    let mut words = Vec::new();
    for index in [63, 64, 65, 2000, 200_002] {
        let child = Object::new(Some(engine.interp.object_proto));
        let grandchild = Value::lstr(format!("not-interned-remembered-{index}-{}", index * 317));
        child
            .borrow_mut()
            .props
            .insert("child", Property::plain(grandchild));
        words.extend([witness(Value::Obj(child)), witness(grandchild)]);
        assert!(array
            .borrow_mut()
            .props
            .set_index_value(index, Value::Obj(child))
            .is_ok());
    }
    for index in 0..5 {
        let child = Object::new(None);
        words.push(witness(Value::Obj(child)));
        assert!(array
            .borrow_mut()
            .props
            .try_append_element(200_003 + index, Property::plain(Value::Obj(child)))
            .is_ok());
    }
    array.borrow_mut().props.set_array_length_value(200_008.0);
    assert_live(&engine, &words, CellAge::Young);
    let dirty = remembered(&array);
    assert_eq!(dirty, Some((Some(200_003), vec![0, 1, 31, 3125])));
    {
        let heap = engine.interp.gc_heap.clone();
        let _guard = heap.core.borrow().nogc();
        assert!(!crate::value::mark_agent_heap(
            &heap,
            false,
            &ArrayRoot(witness(value)),
            CollectionKind::Minor
        ));
    }
    array.borrow().trace_gc(&mut Ignore);
    array.borrow().props.trace_gc(&mut Ignore);
    assert_eq!(remembered(&array), dirty);
    let before = visits();
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_live(&engine, &words, CellAge::Old);
    assert_eq!(visits()[1] - before[1], 200);
    assert_eq!(remembered(&array), Some((Some(200_008), Vec::new())));
    let before = visits();
    array.borrow_mut().props.set_array_length_value(200_008.0); // unrelated coarse card
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_eq!(visits()[1] - before[1], 0);
    assert_live(&engine, &words, CellAge::Old);
    drop(owner);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    for word in words {
        assert_eq!(engine.interp.gc_heap.core.borrow().cell_age(word), None);
    }
}

#[test]
fn remembered_slots_keep_string_bigint_symbol_closure_and_accessor_edges() {
    let mut engine = Engine::new();
    let _agent = engine.interp.enter_agent();
    let value = engine.interp.make_array(vec![Value::Null; 256]);
    let array = object(value);
    let owner = Rooted::new(value);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    let heap = engine.interp.gc_heap.clone();
    let guard = heap.core.borrow().nogc();
    let string = Value::lstr("remembered-only-long-fresh-string-012345678901234567890".repeat(3));
    let bigint = Value::bigint_from_i128(1234567890123456789012345);
    let symbol = engine
        .interp
        .new_symbol(Some(Rc::from("remembered-only-symbol")));
    let ValueView::Sym(symbol_ref) = symbol.view() else {
        panic!("Symbol")
    };
    let symbol_pin = symbol_downgrade(&symbol_ref);
    let symbol_word = witness(symbol);
    evaluate(&mut engine, "var rememberedClosure=(function(){var leaf={retained:731};return function(){return leaf;};})();0");
    let env = engine.interp.global_env;
    let closure = engine
        .interp
        .get_var("rememberedClosure", &env)
        .unwrap_or_else(|_| panic!("fixture closure binding exists"));
    let closure_object = object(closure);
    let mut words = vec![
        witness(string),
        witness(bigint),
        witness(symbol),
        witness(closure),
    ];
    {
        let body = closure_object.borrow();
        let ObjectKind::User(user) = &body.kind else {
            panic!("user closure")
        };
        words.push(crate::gc_box::GcBox::as_ptr(user) as u64);
        words.push(crate::interpreter::Env::as_ptr(&user.env) as u64);
        let leaf = engine
            .interp
            .get_var("leaf", &user.env)
            .unwrap_or_else(|_| panic!("captured leaf binding exists"));
        words.push(witness(leaf));
    }
    evaluate(&mut engine, "rememberedClosure=undefined;0");
    for (index, value) in [string, bigint, symbol, closure].into_iter().enumerate() {
        assert!(array
            .borrow_mut()
            .props
            .set_index_value((index * 64) as u32, value)
            .is_ok());
    }
    let getter = Value::Obj(
        engine
            .interp
            .make_native("rememberedGetter", 0, empty_native),
    );
    let setter = Value::Obj(
        engine
            .interp
            .make_native("rememberedSetter", 1, empty_native),
    );
    words.extend([witness(getter), witness(setter)]);
    array.borrow_mut().props.insert(
        "65",
        Property::accessor_prop(Some(getter), Some(setter), true, true),
    );
    // The accessor is a named entry, its word a hole: the entries are traced in full.
    assert!(array.borrow().props.has_far());
    assert!(elements(&array.borrow().props).get(65).is_none());
    let labels = [
        "string",
        "BigInt",
        "Symbol",
        "closure object",
        "closure payload",
        "captured environment",
        "captured leaf",
        "accessor getter",
        "accessor setter",
    ];
    assert_eq!(labels.len(), words.len());
    let labeled: Vec<_> = labels.into_iter().zip(words).collect();
    assert_labeled_live(&engine, &labeled, CellAge::Young);
    assert_symbol_pin_live(&symbol_pin, symbol_word);
    drop(guard);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_labeled_live(&engine, &labeled, CellAge::Old);
    assert_symbol_pin_live(&symbol_pin, symbol_word);
    drop(owner);
    assert_dead_then_drop_symbol_pin_and_reclaim(&mut engine, &labeled, symbol_pin, symbol_word);
}

#[test]
fn remembered_object_trace_preserves_every_non_element_edge_family() {
    let mut engine = Engine::new();
    let _agent = engine.interp.enter_agent();
    let value = engine.interp.make_array(vec![Value::Null; 128]);
    let array = object(value);
    let owner = Rooted::new(value);
    // An object holds one kind (W2d step 1): the iterator's edge is a second remembered array's.
    let iterator_value = engine.interp.make_array(vec![Value::Null; 128]);
    let iterator_array = object(iterator_value);
    let iterator_owner = Rooted::new(iterator_value);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    let values: Vec<Value> = (0..7)
        .map(|index| {
            if index == 4 {
                Value::Obj(
                    engine
                        .interp
                        .make_native("rememberedBoundTarget", 0, empty_native),
                )
            } else {
                Value::Obj(Object::new(None))
            }
        })
        .collect();
    let symbol = engine
        .interp
        .new_symbol(Some(Rc::from("remembered-key-only")));
    let ValueView::Sym(symbol_ref) = symbol.view() else {
        panic!("Symbol")
    };
    let symbol_pin = symbol_downgrade(&symbol_ref);
    let symbol_word = witness(symbol);
    let key = PropertyKey::symbol(symbol_ref);
    // No element changed: the remembered element range should remain empty while every
    // ordinary property, internal/exotic/callable edge still receives complete tracing.
    {
        let mut body = array.borrow_mut();
        body.proto = Some(object(values[0]));
        body.props.insert("named", Property::plain(values[1]));
        body.internal_slots_mut()
            .insert("internal", Property::plain(values[2]));
        body.props
            .insert(key.as_str(), Property::plain(Value::Null));
        body.kind = ObjectKind::bound(object(values[4]), values[5], vec![values[6]]);
    }
    iterator_array.borrow_mut().kind =
        ObjectKind::ArrayIterator(crate::gc_storage::PayloadBox::new(ArrayIteratorState {
            target: values[3],
            index: 0,
            kind: 0,
        }));
    let words: Vec<u64> = values.into_iter().chain([symbol]).map(witness).collect();
    let labels = [
        "prototype",
        "named property",
        "internal slot",
        "iterator target",
        "bound target",
        "bound receiver",
        "bound argument",
        "Symbol key",
    ];
    assert_eq!(labels.len(), words.len());
    let labeled: Vec<_> = labels.into_iter().zip(words).collect();
    assert_labeled_live(&engine, &labeled, CellAge::Young);
    assert_symbol_pin_live(&symbol_pin, symbol_word);
    let before = visits();
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_eq!(visits()[1] - before[1], 0);
    assert_labeled_live(&engine, &labeled, CellAge::Old);
    assert_symbol_pin_live(&symbol_pin, symbol_word);
    drop((owner, iterator_owner));
    assert_dead_then_drop_symbol_pin_and_reclaim(&mut engine, &labeled, symbol_pin, symbol_word);
}

#[test]
fn unwinding_after_elements_before_callable_edges_does_not_consume_dirtiness() {
    let mut engine = Engine::new();
    let _agent = engine.interp.enter_agent();
    let value = engine.interp.make_array(vec![Value::Null; 128]);
    let array = object(value);
    let _owner = Rooted::new(value);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    let child = Object::new(None);
    let target = Object::new(None);
    {
        let mut body = array.borrow_mut();
        assert!(body.props.set_index_value(65, Value::Obj(child)).is_ok());
        body.kind = ObjectKind::bound(target, Value::Null, Vec::new());
    }
    let dirty = remembered(&array);
    assert_eq!(dirty, Some((Some(128), vec![1])));
    struct PanicAt(u64);
    impl GcSink for PanicAt {
        fn word(&mut self, word: u64) {
            if word == self.0 {
                panic!("after element words")
            }
        }
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mark_object(
            &array.borrow(),
            TracePurpose::Remembered,
            &mut PanicAt(Gc::as_ptr(&target) as u64),
        );
    }));
    assert!(result.is_err());
    assert_eq!(remembered(&array), dirty);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_live(
        &engine,
        &[witness(Value::Obj(child)), witness(Value::Obj(target))],
        CellAge::Old,
    );
}

#[test]
fn small_blocks_reallocation_transplant_and_empty_storage_start_safely() {
    let append = |properties: &mut Props, index: usize| {
        assert!(properties
            .try_append_element(index as u32, Property::plain(Value::Null))
            .is_ok());
    };
    // A block with room for at most 64 words keeps no record: every trace reports all of it.
    let mut properties = Props::packed_array(std::iter::repeat_n(Value::Null, 10));
    assert_eq!(elements(&properties).remembered(), None);
    clean(elements(&properties));
    assert_eq!(
        selected(elements(&properties), TracePurpose::Remembered),
        10
    );
    // Growth past 64 words gives a fresh block, whose record starts full.
    for index in 10..100 {
        append(&mut properties, index);
    }
    assert_eq!(elements(&properties).remembered(), Some((None, Vec::new())));
    clean(elements(&properties));
    // Appends within the capacity are words past the traced length.
    let capacity = elements(&properties).capacity();
    assert!(capacity > 100);
    for index in 100..capacity {
        append(&mut properties, index);
    }
    assert_eq!(
        selected(elements(&properties), TracePurpose::Remembered),
        capacity - 100
    );
    clean(elements(&properties));
    // One more grows the block again, keeping its record: the new word is past the traced
    // length.
    append(&mut properties, capacity);
    assert_eq!(selected(elements(&properties), TracePurpose::Remembered), 1);
    clean(elements(&properties));
    let receiver = Object::new(None);
    receiver.borrow_mut().replace_properties(properties);
    assert_eq!(remembered(&receiver), Some((None, Vec::new())));
    // An empty block with a record: its first append is past the traced length 0.
    let mut properties = Props::packed_array(std::iter::empty());
    assert!(!properties.elems.has_words());
    properties.reserve_dense_exact(word_block::REMEMBER_MIN + 1);
    clean(elements(&properties));
    assert_eq!(
        elements(&properties).remembered(),
        Some((Some(0), Vec::new()))
    );
    append(&mut properties, 0);
    assert_eq!(selected(elements(&properties), TracePurpose::Remembered), 1);
}

struct ArrayRoot(u64);
impl crate::gc_core::MarkClient for ArrayRoot {
    fn roots(&self, sink: &mut dyn FnMut(u64)) {
        sink(self.0);
    }
    fn side_slot_owners(&self, _: &mut dyn FnMut(usize)) {}
    fn trace_side_slots(&self, _: usize, _: &mut dyn FnMut(u64)) {}
    fn ephemerons(&self, _: &mut crate::gc_core::EdgeSink<'_>) {}
}

#[test]
fn between_phase_mutation_belongs_to_the_following_collection() {
    let mut engine = Engine::new();
    let _agent = engine.interp.enter_agent();
    let value = engine.interp.make_array(vec![Value::Null; 128]);
    let array = object(value);
    let _owner = Rooted::new(value);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    assert!(array
        .borrow_mut()
        .props
        .set_index_value(65, Value::Null)
        .is_ok());
    assert_eq!(remembered(&array), Some((Some(128), vec![1])));
    let heap = engine.interp.gc_heap.clone();
    // Every other engine cell is already old from the precise major. This minor has no new
    // side-table state and reports this array explicitly, without any conservative roots.
    assert!(crate::value::mark_agent_heap(
        &heap,
        false,
        &ArrayRoot(witness(value)),
        CollectionKind::Minor
    ));
    assert_eq!(remembered(&array), Some((Some(128), Vec::new())));
    let black = Object::new(None);
    assert!(array
        .borrow_mut()
        .props
        .set_index_value(65, Value::Obj(black))
        .is_ok());
    heap.core.borrow_mut().finish_collection(&[]);
    assert_eq!(remembered(&array), Some((Some(128), vec![1])));
    let before = visits();
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_eq!(visits()[1] - before[1], 64);
    assert_live(&engine, &[witness(Value::Obj(black))], CellAge::Old);
}

#[test]
fn reentrant_getter_mutation_remembers_the_actual_old_destination() {
    let mut engine = Engine::new();
    evaluate(&mut engine, "var rememberedDst=new Array(128).fill(null);var rememberedReceiver={get trigger(){rememberedDst[65]={nested:{value:912}};return 1}};function rememberedCallback(){return rememberedReceiver.trigger}0");
    let env = engine.interp.global_env;
    let array = object(
        engine
            .interp
            .get_var("rememberedDst", &env)
            .unwrap_or_else(|_| panic!("fixture destination binding exists")),
    );
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    let heap = engine.interp.gc_heap.clone();
    let guard = heap.core.borrow().nogc();
    evaluate(&mut engine, "[1].map(rememberedCallback);0");
    let parent = array.borrow().props.get_index(65).unwrap().value();
    let child = object(parent).borrow().props.get("nested").unwrap().value();
    let words = [witness(parent), witness(child)];
    assert_live(&engine, &words, CellAge::Young);
    drop(guard);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_live(&engine, &words, CellAge::Old);
    evaluate(
        &mut engine,
        "rememberedDst=undefined;rememberedReceiver=undefined;rememberedCallback=undefined;0",
    );
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    for word in words {
        assert_eq!(heap.core.borrow().cell_age(word), None);
    }
}

#[test]
fn named_and_sparse_element_edges_remain_complete_outside_the_packed_blocks() {
    let mut engine = Engine::new();
    let _agent = engine.interp.enter_agent();
    // Element 0 in the named entries (its descriptor non-plain), then made plain again in
    // place: the named element's edge stays complete outside the words.
    let mut props = Props::new();
    props.insert("0", Property::data(Value::Null, true, true, false));
    props.insert(
        "length",
        Property::data(Value::Num(1.0), true, false, false),
    );
    props.mark_array();
    assert!(!props.elems.has_words());
    assert!(props.has_far());
    let array = Object::new_with_parts(None, props, ObjectKind::Array);
    let _owner = Rooted::new(Value::Obj(array));
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    let first = Value::Obj(Object::new(None));
    // A word write leaves a named element to the generic path; its value-only write is this.
    assert!(array.borrow_mut().props.set_index_value(0, first).is_err());
    assert!(array.borrow_mut().props.set_value("0", first));
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_live(&engine, &[witness(first)], CellAge::Old);
    let props = Props::packed_array(std::iter::repeat_n(Value::Null, 128));
    array.borrow_mut().replace_properties(props);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    let far = Value::Obj(Object::new(None));
    array
        .borrow_mut()
        .props
        .insert("4000000000", Property::plain(far));
    assert!(array.borrow().props.has_far());
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_live(&engine, &[witness(far)], CellAge::Old);
    array.borrow_mut().props.remove("65");
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    let hole = Value::Obj(Object::new(None));
    // With a far key, an index no named entry holds still fills its word, remembered.
    array.borrow_mut().props.insert("65", Property::plain(hole));
    assert_eq!(remembered(&array), Some((Some(128), vec![1])));
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_live(&engine, &[witness(hole)], CellAge::Old);
}

#[cfg(target_arch = "aarch64")]
thread_local! {
    static NATIVE_FACTORY_ARMED: Cell<bool> = const { Cell::new(false) };
    static NATIVE_FACTORY_CALLS: Cell<usize> = const { Cell::new(0) };
    static NATIVE_FACTORY_WORDS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

#[cfg(target_arch = "aarch64")]
#[derive(Debug)]
struct NativeAppendGuards {
    array: bool,
    plain: bool,
    extensible: bool,
    prototype: bool,
    element_mode: bool,
    far: bool,
    length_memo: Option<usize>,
    memo_is_length: bool,
    length_descriptor: Option<(bool, bool, Option<f64>)>,
    words: Option<(usize, usize)>,
    remembered: Option<(Option<usize>, Vec<usize>)>,
}

#[cfg(target_arch = "aarch64")]
fn native_append_guards(array: &Gc) -> NativeAppendGuards {
    let body = array.borrow();
    let length_memo = body.props.len_slot.get().slot();
    let snapshot = NativeAppendGuards {
        array: matches!(body.kind, ObjectKind::Array),
        plain: body.ic_plain(),
        extensible: body.extensible(),
        prototype: body.props.proto_flag(),
        element_mode: body.props.elem_mode(),
        far: body.props.has_far(),
        length_memo,
        memo_is_length: length_memo.is_some_and(|slot| {
            body.props
                .entries
                .keys()
                .get(slot)
                .is_some_and(|key| key.is("length"))
        }),
        // Read the stored descriptor directly: snapshots must not prime the length memo.
        length_descriptor: body
            .props
            .entries
            .iter()
            .find(|(key, _)| key.is("length"))
            .map(|(_, property)| {
                (
                    property.accessor(),
                    property.writable(),
                    property.number_value(),
                )
            }),
        words: body
            .props
            .elems
            .words()
            .map(|words| (words.len(), words.capacity())),
        remembered: body.props.elems.words().and_then(Words::remembered),
    };
    snapshot
}

#[cfg(target_arch = "aarch64")]
fn native_fresh_tree(it: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    if !NATIVE_FACTORY_ARMED.with(Cell::get) {
        return Ok(Value::Null);
    }
    let serial = NATIVE_FACTORY_CALLS.with(|count| {
        count.set(count.get() + 1);
        count.get()
    });
    let leaf = Value::lstr(format!(
        "sole-native-append-child-{serial}-012345678901234567890"
    ));
    let parent = Object::new(None);
    parent
        .borrow_mut()
        .props
        .insert("child", Property::plain(leaf));
    NATIVE_FACTORY_WORDS
        .with(|words| *words.borrow_mut() = vec![witness(Value::Obj(parent)), witness(leaf)]);
    let env = it.global_env;
    let destination = object(
        it.get_var("dst", &env)
            .unwrap_or_else(|_| panic!("native destination exists")),
    );
    eprintln!(
        "[append-guards] callback serial={serial} state={:?}",
        native_append_guards(&destination)
    );
    Ok(Value::Obj(parent))
}

#[cfg(target_arch = "aarch64")]
fn append_counts() -> (usize, usize, usize) {
    (
        crate::jit::TEST_PACKED_APPEND_COMMITS.with(Cell::get),
        crate::bytecode::TEST_JIT_DEFINE_ELEMENT_HELPERS.with(Cell::get),
        NATIVE_FACTORY_CALLS.with(Cell::get),
    )
}

#[test]
#[cfg(target_arch = "aarch64")]
fn native_define_element_remembers_old_heap_append_and_same_length_regrowth() {
    const CHILD: &str = "TRUST_JS_REMEMBERED_NATIVE_APPEND_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "value::remembered_elements_tests::native_define_element_remembers_old_heap_append_and_same_length_regrowth", "--nocapture"])
            .env(CHILD, "1").output().unwrap();
        assert!(
            output.status.success(),
            "native append child exited {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    // Only this isolated child opts in, before any warmup/native compilation.
    crate::jit::TEST_PACKED_APPEND_ENABLED.with(|enabled| enabled.set(true));
    let mut engine = Engine::new();
    engine.set_tier(crate::bytecode::Tier::Jit);
    engine.set_tier_threshold(0);
    let fresh = engine
        .interp
        .make_native("freshYoung", 0, native_fresh_tree);
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert("freshYoung", Property::plain(Value::Obj(fresh)));
    // The destination's block has room for more than 64 words, so it keeps a record.
    evaluate(&mut engine, "var dst=new Array(128).fill(null); dst.length=40; var src=new Array(41);src[40]=1;src.constructor={ [Symbol.species]:function(){return dst;} };for(var w=0;w<64;w++){dst.length=40;src.map(freshYoung);}dst.length=40;0");
    let env = engine.interp.global_env;
    let array = object(
        engine
            .interp
            .get_var("dst", &env)
            .unwrap_or_else(|_| panic!("native destination binding exists")),
    );
    let map = engine
        .interp
        .array_proto
        .borrow()
        .props
        .get("map")
        .unwrap()
        .value();
    let map = object(map);
    let chunk = {
        let body = map.borrow();
        let ObjectKind::User(user) = &body.kind else {
            panic!("self-hosted map")
        };
        user.func
            .runtime()
            .execution_code()
            .and_then(Option::as_ref)
            .expect("compiled map")
            .clone()
    };
    assert!(chunk.jit.get().is_some_and(|code| code.is_some()));
    assert!(chunk.jit_ops().iter().any(|op| matches!(
        op,
        crate::bytecode::Op::Abstract(crate::bytecode::AbstractOp::CreateDataPropertyOrThrow)
    )));
    {
        let body = array.borrow();
        let words = elements(&body.props);
        assert_eq!(words.len(), 40);
        assert!(words.capacity() > 40 && words.remembered().is_some());
    }
    // The define element template appends the word itself where generated code reads words.
    let native = usize::from(jit_layout(&engine.interp.object_proto).words_valid);
    assert_eq!(native, 1, "this target's element templates read words");
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    NATIVE_FACTORY_ARMED.with(|armed| armed.set(true));
    crate::jit::TEST_PACKED_APPEND_RECEIVER.with(|cell| cell.set(Gc::as_ptr(&array) as usize));
    crate::jit::TEST_PACKED_APPEND_LENGTH.with(|cell| cell.set(41));
    let heap = engine.interp.gc_heap.clone();
    for (trial, preparation) in ["0", "dst.pop();0", "dst.length=40;0"]
        .into_iter()
        .enumerate()
    {
        let guard = heap.core.borrow().nogc();
        evaluate(&mut engine, preparation);
        let unprimed = native_append_guards(&array);
        eprintln!("[append-guards] trial={trial} before-prime={unprimed:?}");
        // Establish the existing native length-slot precondition through its read-only API.
        // Preparation and append keep the SAME no-GC guard.
        {
            let body = array.borrow();
            let length = body.props.length_property().expect("own Array length");
            assert_eq!(length.number_value(), Some(40.0));
        }
        let primed = native_append_guards(&array);
        assert_eq!(
            unprimed.remembered, primed.remembered,
            "priming changed the remembered record"
        );
        assert!(
            primed.array
                && primed.plain
                && primed.extensible
                && !primed.prototype
                && primed.element_mode
                && !primed.far
                && primed.length_memo.is_some()
                && primed.memo_is_length,
            "trial={trial} invalid native receiver guards: {primed:?}"
        );
        assert_eq!(
            primed.length_descriptor,
            Some((false, true, Some(40.0))),
            "trial={trial} {primed:?}"
        );
        assert!(
            matches!(primed.words, Some((40, capacity)) if capacity > 40),
            "trial={trial} {primed:?}"
        );
        // Traced up to 40: by the major, or lowered there by the pop or the shorter length.
        assert_eq!(
            primed.remembered,
            Some((Some(40), Vec::new())),
            "trial={trial} {primed:?}"
        );
        let before = append_counts();
        evaluate(&mut engine, "src.map(freshYoung);0");
        let after = append_counts();
        assert_eq!(
            (after.0 - before.0, after.1 - before.1, after.2 - before.2),
            (native, 1 - native, 1),
            "trial={trial} unprimed={unprimed:?} primed={primed:?} after={:?}",
            native_append_guards(&array),
        );
        let words = NATIVE_FACTORY_WORDS.with(|words| words.borrow().clone());
        {
            let body = array.borrow();
            assert_eq!(elements(&body.props).len(), 41);
            assert_eq!(witness(body.props.get_index(40).unwrap().value()), words[0]);
        }
        assert_live(&engine, &words, CellAge::Young);
        drop(guard);
        let before_mark = visits();
        engine
            .interp
            .gc_collect_precise_for_test(CollectionKind::Minor);
        // Only the appended word, past the traced length.
        assert_eq!(visits()[1] - before_mark[1], 1, "trial={trial}");
        assert_live(&engine, &words, CellAge::Old);
        assert_eq!(remembered(&array), Some((Some(41), Vec::new())));
    }
    // Same arming and source call, but the extensibility gate must reject before commit.
    let guard = heap.core.borrow().nogc();
    evaluate(&mut engine, "dst.length=40;Object.preventExtensions(dst);0");
    let before = append_counts();
    assert_eq!(evaluate(&mut engine, "var appendRejected=false;try{src.map(freshYoung)}catch(e){appendRejected=e instanceof TypeError}appendRejected"), "true");
    let after = append_counts();
    assert_eq!(
        (after.0 - before.0, after.1 - before.1, after.2 - before.2),
        (0, 1, 1)
    );
    let words = NATIVE_FACTORY_WORDS.with(|words| words.borrow().clone());
    assert_eq!(elements(&array.borrow().props).len(), 40);
    assert_live(&engine, &words, CellAge::Young);
    drop(guard);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    for word in words {
        assert_eq!(heap.core.borrow().cell_age(word), None);
    }
    crate::jit::TEST_PACKED_APPEND_RECEIVER.with(|cell| cell.set(0));
    NATIVE_FACTORY_ARMED.with(|armed| armed.set(false));
}
