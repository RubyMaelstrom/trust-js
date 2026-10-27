//! Collector-only packed-element remembering; ordinary language semantics are unchanged.
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
    PACKED_MARK_VISITS.with(Cell::get)
}

fn selected(buffer: &PackedBuffer, purpose: TracePurpose) -> usize {
    let before = visits();
    buffer.trace_elements(purpose, &mut Ignore);
    let after = visits();
    after[0] + after[1] - before[0] - before[1]
}

fn packed(len: usize) -> PackedBuffer {
    (0..len).map(|_| Property::plain(Value::Null)).collect()
}

fn clean(buffer: &PackedBuffer) {
    assert_eq!(selected(buffer, TracePurpose::Full), buffer.len());
    buffer.finish_mark_trace();
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
    // The sidecar and the packed buffer live in storage blocks (`gc_storage`): plain data, so
    // that dropping a map never reads a block, which may have gone with its heap.
    assert!(!std::mem::needs_drop::<DenseBuffers>());
    assert!(!std::mem::needs_drop::<PackedBuffer>());
    // The map's storage word the JIT follows comes first; the Rust-heap extras follow it.
    assert_eq!(
        std::mem::size_of::<DenseStorage>(),
        2 * std::mem::size_of::<usize>()
    );
    assert_eq!(std::mem::offset_of!(PackedBuffer, values), 0);
    let sample = Object::new(None);
    let layout = jit_layout(&sample);
    assert!(layout.packed_elems_valid);
}

#[test]
fn distant_dirty_blocks_and_partial_prefix_tail_are_bounded_without_duplicates() {
    let mut buffer = packed(200_003);
    clean(&buffer);
    assert!(buffer.dirty.borrow().is_empty());
    for index in [63, 64, 65, 2000, 200_002] {
        buffer
            .index_mut(index)
            .unwrap()
            .set_value(Value::Num(index as f64));
    }
    for _ in 0..5 {
        buffer.push(Property::plain(Value::Null));
    }
    // Three complete blocks, the three-element partial block, and the five new tail slots.
    assert_eq!(selected(&buffer, TracePurpose::Remembered), 3 * 64 + 3 + 5);
    assert_eq!(
        buffer.traced_len.get(),
        200_003,
        "reporting alone does not consume"
    );
    buffer.finish_mark_trace();
    assert_eq!(selected(&buffer, TracePurpose::Remembered), 0);
    assert!(buffer.dirty.borrow().is_empty());
}

#[test]
fn shrink_same_length_regrowth_preserves_old_dirty_blocks_and_new_tail() {
    let mut buffer = packed(130);
    clean(&buffer);
    buffer.index_mut(1).unwrap().set_value(Value::Bool(true));
    buffer.pop().unwrap();
    assert_eq!(buffer.traced_len.get(), 129);
    buffer.push(Property::plain(Value::Bool(false)));
    assert_eq!(selected(&buffer, TracePurpose::Remembered), 65);
    buffer.finish_mark_trace();
    buffer.full_mut().truncate(40);
    buffer.push(Property::plain(Value::Null));
    assert_eq!(selected(&buffer, TracePurpose::Remembered), 41);
}

#[test]
fn mutable_escapes_reorders_clone_and_whole_replacement_start_full() {
    let mut properties = Props::packed_array(std::iter::repeat_n(Value::Null, 130));
    clean(properties.elems.packed.as_deref().unwrap());
    let clone = properties.clone();
    assert_eq!(
        clone.elems.packed.as_deref().unwrap().traced_len.get(),
        PACKED_TRACE_FULL
    );
    properties.elems.packed_slice_mut().unwrap()[1].set_value(Value::Num(1.0));
    assert_eq!(
        selected(
            properties.elems.packed.as_deref().unwrap(),
            TracePurpose::Remembered
        ),
        130
    );
    clean(properties.elems.packed.as_deref().unwrap());
    properties.plain_elements_mut(130).unwrap().reverse();
    assert_eq!(
        selected(
            properties.elems.packed.as_deref().unwrap(),
            TracePurpose::Remembered
        ),
        130
    );
    clean(properties.elems.packed.as_deref().unwrap());
    assert!(properties.shift_dense_array(130, false).is_some());
    assert_eq!(
        selected(
            properties.elems.packed.as_deref().unwrap(),
            TracePurpose::Remembered
        ),
        129
    );
    clean(properties.elems.packed.as_deref().unwrap());
    assert!(properties.unshift_dense_array(129, &[Value::Null]));
    assert_eq!(
        selected(
            properties.elems.packed.as_deref().unwrap(),
            TracePurpose::Remembered
        ),
        130
    );
    clean(properties.elems.packed.as_deref().unwrap());
    let new_buffer = crate::gc_storage::StorageBox::new(packed(32));
    clean(&new_buffer);
    properties.elems.set_packed(Some(new_buffer));
    assert_eq!(
        properties.elems.packed.as_deref().unwrap().traced_len.get(),
        PACKED_TRACE_FULL
    );
}

#[test]
fn bitmap_growth_failure_falls_back_full_without_losing_the_write() {
    let mut properties = Props::packed_array(std::iter::repeat_n(Value::Null, 10_000));
    clean(properties.elems.packed.as_deref().unwrap());
    PACKED_BITMAP_FAIL_GROWTH.with(|flag| flag.set(true));
    let result = properties.set_index_value(9000, Value::Bool(true));
    PACKED_BITMAP_FAIL_GROWTH.with(|flag| flag.set(false));
    assert!(result.is_ok());
    assert!(matches!(
        properties.get_index(9000).unwrap().value().view(),
        ValueView::Bool(true)
    ));
    assert_eq!(
        selected(
            properties.elems.packed.as_deref().unwrap(),
            TracePurpose::Remembered
        ),
        10_000
    );
}

#[test]
fn indexed_setters_and_hole_fill_record_blocks_but_generic_escapes_remain_full() {
    let mut properties = Props::packed_array(std::iter::repeat_n(Value::Null, 2000));
    for operation in 0..3 {
        clean(properties.elems.packed.as_deref().unwrap());
        match operation {
            0 => {
                assert!(properties.set_index_value(65, Value::Bool(true)).is_ok());
            }
            1 => {
                assert!(properties.set_value("65", Value::Bool(false)));
            }
            _ => properties.insert("65", Property::plain(Value::Num(3.0))),
        }
        assert_eq!(
            selected(
                properties.elems.packed.as_deref().unwrap(),
                TracePurpose::Remembered
            ),
            64
        );
    }
    properties.remove("65");
    clean(properties.elems.packed.as_deref().unwrap());
    assert!(properties
        .try_define_dense_element(65, Property::plain(Value::Null))
        .is_ok());
    // The generic bounded-gap/deque path intentionally abandons precision.
    assert_eq!(
        selected(
            properties.elems.packed.as_deref().unwrap(),
            TracePurpose::Remembered
        ),
        2000
    );
}

#[test]
fn precise_descriptor_replacement_still_invalidates_the_plain_element_proof() {
    let mut properties = Props::packed_array(std::iter::repeat_n(Value::Null, 128));
    clean(properties.elems.packed.as_deref().unwrap());
    assert!(properties.elems.packed_all_plain());
    properties.insert("65", Property::accessor_prop(None, None, true, true));
    assert!(!properties.elems.packed_all_plain());
    assert!(properties.shift_dense_array(128, false).is_none());
    assert!(!properties.unshift_dense_array(128, &[Value::Null]));
    assert_eq!(
        selected(
            properties.elems.packed.as_deref().unwrap(),
            TracePurpose::Remembered
        ),
        64
    );
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
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    assert_eq!(
        array
            .borrow()
            .props
            .elems
            .packed
            .as_deref()
            .unwrap()
            .traced_len
            .get(),
        200_003
    );
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
    let dirty = array
        .borrow()
        .props
        .elems
        .packed
        .as_deref()
        .unwrap()
        .dirty
        .borrow()
        .to_vec();
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
    assert_eq!(
        array
            .borrow()
            .props
            .elems
            .packed
            .as_deref()
            .unwrap()
            .dirty
            .borrow()
            .to_vec(),
        dirty
    );
    let before = visits();
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_live(&engine, &words, CellAge::Old);
    assert_eq!(visits()[1] - before[1], 200);
    assert_eq!(
        array
            .borrow()
            .props
            .elems
            .packed
            .as_deref()
            .unwrap()
            .traced_len
            .get(),
        200_008
    );
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
        let Callable::User(user) = &body.call else {
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
        body.exotic =
            Exotic::ArrayIterator(crate::gc_storage::PayloadBox::new(ArrayIteratorState {
                target: values[3],
                index: 0,
                kind: 0,
            }));
        body.call = Callable::bound(object(values[4]), values[5], vec![values[6]]);
    }
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
    drop(owner);
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
        body.call = Callable::bound(target, Value::Null, Vec::new());
    }
    let dirty = array
        .borrow()
        .props
        .elems
        .packed
        .as_deref()
        .unwrap()
        .dirty
        .borrow()
        .to_vec();
    struct PanicAt(u64);
    impl GcSink for PanicAt {
        fn word(&mut self, word: u64) {
            if word == self.0 {
                panic!("after packed elements")
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
    assert_eq!(
        array
            .borrow()
            .props
            .elems
            .packed
            .as_deref()
            .unwrap()
            .dirty
            .borrow()
            .to_vec(),
        dirty
    );
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
fn inline_promotion_reallocation_transplant_and_empty_heap_storage_start_safely() {
    let mut properties = Props::packed_array(std::iter::repeat_n(Value::Null, 10));
    assert!(properties.elems.packed.is_none());
    assert!(properties
        .try_append_element(10, Property::plain(Value::Null))
        .is_ok());
    assert_eq!(
        properties.elems.packed.as_deref().unwrap().traced_len.get(),
        PACKED_TRACE_FULL
    );
    clean(properties.elems.packed.as_deref().unwrap());
    for index in 11..1000 {
        assert!(properties
            .try_append_element(index, Property::plain(Value::Null))
            .is_ok());
    }
    assert_eq!(
        selected(
            properties.elems.packed.as_deref().unwrap(),
            TracePurpose::Remembered
        ),
        989
    );
    clean(properties.elems.packed.as_deref().unwrap());
    let receiver = Object::new(None);
    receiver.borrow_mut().replace_properties(properties);
    assert_eq!(
        receiver
            .borrow()
            .props
            .elems
            .packed
            .as_deref()
            .unwrap()
            .traced_len
            .get(),
        PACKED_TRACE_FULL
    );
    let mut properties = Props::packed_array(std::iter::empty());
    properties.reserve_dense_exact(64, false);
    clean(properties.elems.packed.as_deref().unwrap());
    assert!(properties
        .try_append_element(0, Property::plain(Value::Null))
        .is_ok());
    assert_eq!(
        selected(
            properties.elems.packed.as_deref().unwrap(),
            TracePurpose::Remembered
        ),
        1
    );
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
    let heap = engine.interp.gc_heap.clone();
    // Every other engine cell is already old from the precise major. This minor has no new
    // side-table state and reports this array explicitly, without any conservative roots.
    assert!(crate::value::mark_agent_heap(
        &heap,
        false,
        &ArrayRoot(witness(value)),
        CollectionKind::Minor
    ));
    assert!(array
        .borrow()
        .props
        .elems
        .packed
        .as_deref()
        .unwrap()
        .dirty
        .borrow()
        .is_empty());
    let black = Object::new(None);
    assert!(array
        .borrow_mut()
        .props
        .set_index_value(65, Value::Obj(black))
        .is_ok());
    heap.core.borrow_mut().finish_collection(&[]);
    assert!(!array
        .borrow()
        .props
        .elems
        .packed
        .as_deref()
        .unwrap()
        .dirty
        .borrow()
        .is_empty());
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
fn classic_and_sparse_element_edges_remain_complete_outside_the_packed_blocks() {
    let mut engine = Engine::new();
    let _agent = engine.interp.enter_agent();
    let mut props = Props::new();
    props.insert("0", Property::plain(Value::Null));
    props.insert(
        "length",
        Property::data(Value::Num(1.0), true, false, false),
    );
    props.mark_array();
    assert!(props.elems.packed.is_none());
    let array = Object::new_with_parts(None, props, Exotic::Array);
    let _owner = Rooted::new(Value::Obj(array));
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    let first = Value::Obj(Object::new(None));
    assert!(array.borrow_mut().props.set_index_value(0, first).is_ok());
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
    assert!(array.borrow().props.has_far.get());
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_live(&engine, &[witness(far)], CellAge::Old);
    array.borrow_mut().props.remove("65");
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    let hole = Value::Obj(Object::new(None));
    // A far key declines direct dense definition; ordinary insert still records the hole.
    array.borrow_mut().props.insert("65", Property::plain(hole));
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
    packed: Option<(usize, usize)>,
    mirror_flags: u8,
    remembered: Option<(usize, Vec<u64>)>,
}

#[cfg(target_arch = "aarch64")]
fn native_append_guards(array: &Gc) -> NativeAppendGuards {
    let body = array.borrow();
    let length_memo = body.props.len_slot.get().slot();
    let snapshot = NativeAppendGuards {
        array: matches!(body.exotic, Exotic::Array),
        plain: body.ic_plain.get(),
        extensible: body.extensible,
        prototype: body.props.proto_flag.get(),
        element_mode: body.props.elem_mode.get(),
        far: body.props.has_far.get(),
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
        packed: body
            .props
            .elems
            .packed
            .as_deref()
            .map(|packed| (packed.len(), packed.cap)),
        mirror_flags: body.props.mirror_flags,
        remembered: body
            .props
            .elems
            .packed
            .as_deref()
            .map(|packed| (packed.traced_len.get(), packed.dirty.borrow().to_vec())),
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
fn append_counts() -> (usize, usize, usize, usize) {
    (
        crate::jit::TEST_PACKED_APPEND_COMMITS.with(Cell::get),
        crate::jit::TEST_PACKED_APPEND_HEAP_COMMITS.with(Cell::get),
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
    evaluate(&mut engine, "var dst=new Array(64).fill(null); dst.length=40; var src=new Array(41);src[40]=1;src.constructor={ [Symbol.species]:function(){return dst;} };for(var w=0;w<64;w++){dst.length=40;src.map(freshYoung);}dst.length=40;0");
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
        let Callable::User(user) = &body.call else {
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
        let packed = body
            .props
            .elems
            .packed
            .as_deref()
            .expect("heap-packed destination");
        assert_eq!(packed.len(), 40);
        assert!(packed.cap > 40);
        assert_eq!(body.props.mirror_flags, 0);
    }
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
        // Preparation and append keep the SAME no-GC guard; no shrink/regrow promotion.
        {
            let body = array.borrow();
            let length = body.props.length_property().expect("own Array length");
            assert_eq!(length.number_value(), Some(40.0));
        }
        let primed = native_append_guards(&array);
        assert_eq!(
            unprimed.remembered, primed.remembered,
            "priming changed packed metadata"
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
            matches!(primed.packed, Some((40, cap)) if cap > 40),
            "trial={trial} {primed:?}"
        );
        assert_eq!(primed.mirror_flags, 0, "trial={trial} {primed:?}");
        let before = append_counts();
        evaluate(&mut engine, "src.map(freshYoung);0");
        let after = append_counts();
        assert_eq!(
            (
                after.0 - before.0,
                after.1 - before.1,
                after.2 - before.2,
                after.3 - before.3
            ),
            (1, 1, 0, 1),
            "trial={trial} unprimed={unprimed:?} primed={primed:?} after={:?}",
            native_append_guards(&array),
        );
        let words = NATIVE_FACTORY_WORDS.with(|words| words.borrow().clone());
        {
            let body = array.borrow();
            assert_eq!(body.props.elems.packed_ref().unwrap().len(), 41);
            assert_eq!(witness(body.props.get_index(40).unwrap().value()), words[0]);
        }
        assert_live(&engine, &words, CellAge::Young);
        drop(guard);
        let before_mark = visits();
        engine
            .interp
            .gc_collect_precise_for_test(CollectionKind::Minor);
        assert_eq!(visits()[1] - before_mark[1], if trial < 2 { 1 } else { 41 });
        assert_live(&engine, &words, CellAge::Old);
        assert_eq!(
            array
                .borrow()
                .props
                .elems
                .packed
                .as_deref()
                .unwrap()
                .traced_len
                .get(),
            41
        );
    }
    // Same arming and source call, but the extensibility gate must reject before commit.
    let guard = heap.core.borrow().nogc();
    evaluate(&mut engine, "dst.length=40;Object.preventExtensions(dst);0");
    let before = append_counts();
    assert_eq!(evaluate(&mut engine, "var appendRejected=false;try{src.map(freshYoung)}catch(e){appendRejected=e instanceof TypeError}appendRejected"), "true");
    let after = append_counts();
    assert_eq!(
        (
            after.0 - before.0,
            after.1 - before.1,
            after.2 - before.2,
            after.3 - before.3
        ),
        (0, 0, 1, 1)
    );
    let words = NATIVE_FACTORY_WORDS.with(|words| words.borrow().clone());
    assert_eq!(array.borrow().props.elems.packed_ref().unwrap().len(), 40);
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
