//! Layout lifetime and COW invariants, independent of conservative stack retention.
use super::*;
use crate::gc_core::{CollectionKind, EdgeSink, MarkClient};
use crate::gc_trace::{GcSink, GcTrace, VisitSink};

pub(crate) struct Words(pub(crate) Vec<u64>);
impl GcTrace for Words {
    fn trace_gc(&self, sink: &mut dyn GcSink) {
        for &word in &self.0 {
            sink.word(word);
        }
    }
}

pub(crate) fn collect(heap: &GcHeap, roots: &dyn GcTrace, kind: CollectionKind) {
    struct Roots<'a> {
        heap: &'a GcHeap,
        roots: &'a dyn GcTrace,
    }
    impl MarkClient for Roots<'_> {
        fn roots(&self, sink: &mut dyn FnMut(u64)) {
            let mut words = VisitSink(&mut |cell| sink(cell as u64));
            self.roots.trace_gc(&mut words);
            trace_heap_roots(self.heap, &mut words);
            crate::gc_trace::trace_thread_roots(&mut words);
        }
        fn side_slot_owners(&self, _: &mut dyn FnMut(usize)) {}
        fn trace_side_slots(&self, _: usize, _: &mut dyn FnMut(u64)) {}
        fn ephemerons(&self, _: &mut EdgeSink<'_>) {}
    }
    let client = Roots { heap, roots };
    assert!(heap.core.borrow_mut().start_marking(false, &client, kind));
    let dead = crate::gc_sweep::DeadObjects::new(heap);
    sweep_object_metadata(heap, &dead);
    let pins = {
        let core = heap.core.borrow();
        crate::gc_handle::clear_dead_weak_cells(|cell| core.is_dead_target(cell))
    };
    heap.core.borrow_mut().finish_collection(&pins);
}

pub(crate) fn isolated() -> (GcHeap, HeapRelease, AgentActivationGuard) {
    let heap = new_gc_heap();
    let symbols = new_symbol_agent();
    let active = enter_agent(&heap, &symbols);
    let release = HeapRelease::new(&heap);
    (heap, release, active)
}

fn predicted(key: &str) -> Props {
    Props::with_layout(1, Some(new_property_layout(vec![Rc::from(key)])))
}

#[test]
fn private_mutation_reuses_its_cell_and_clone_sharing_is_sticky() {
    let (_heap, _release, _active) = isolated();
    let mut props = predicted("unused");
    // Keep this test on the private COW path rather than adopting a shape-cache layout.
    props.shape = SHAPE_UNCACHEABLE;
    let private = props.shared_layout().unwrap().as_ptr();
    assert!(!props.shared_layout().unwrap().is_shared());
    props.insert("first", Property::plain(Value::Num(1.0)));
    assert_eq!(props.shared_layout().unwrap().as_ptr(), private);
    let copy = props.clone();
    assert!(props.shared_layout().unwrap().is_shared());
    drop(copy);
    assert!(props.shared_layout().unwrap().is_shared());
    props.insert("next", Property::data(Value::Num(2.0), false, true, false));
    assert_ne!(props.shared_layout().unwrap().as_ptr(), private);
    assert!(!props.shared_layout().unwrap().is_shared());
    let replacement = props.shared_layout().unwrap().as_ptr();
    props.remove("first");
    assert_eq!(props.shared_layout().unwrap().as_ptr(), replacement);
    assert_eq!(
        props
            .ordered_keys()
            .iter()
            .map(|key| &**key)
            .collect::<Vec<_>>(),
        ["next"]
    );
    let next = props.get("next").unwrap();
    assert!(!next.writable());
    assert!(!next.configurable());
}

#[test]
fn object_and_standalone_props_trace_unused_predictions_in_minor_and_major() {
    let (heap, _release, _active) = isolated();
    let object = Object::new(None);
    collect(&heap, &object, CollectionKind::Major);
    object.borrow_mut().props = predicted("object prediction");
    let object_layout = object.borrow().props.shared_layout().unwrap().as_ptr() as usize;
    let standalone = crate::rooting::Rooted::new(predicted("standalone prediction"));
    let standalone_layout = standalone.shared_layout().unwrap().as_ptr() as usize;
    for kind in [CollectionKind::Minor, CollectionKind::Major] {
        collect(&heap, &object, kind);
        let core = heap.core.borrow();
        assert!(core.owns_cell(object_layout, heap.property_layout_desc));
        assert!(core.owns_cell(standalone_layout, heap.property_layout_desc));
    }
    assert_eq!(
        standalone.entries.len(),
        0,
        "unused predictions remain unobservable"
    );
}

#[test]
fn array_length_and_shape_cache_are_layout_roots_in_minor_and_major() {
    let (heap, _release, _active) = isolated();
    collect(&heap, &Words(Vec::new()), CollectionKind::Major);
    let array = array_length_layout();
    let array_word = array.cell_word();
    assert!(array.is_shared());
    drop(array);
    let shape = heap.shapes.borrow_mut().fresh();
    let layout = new_property_layout(vec![Rc::from("shape only")]);
    let shape_word = layout.cell_word();
    heap.shapes.borrow_mut().layouts.insert(shape, layout);
    for kind in [CollectionKind::Minor, CollectionKind::Major] {
        collect(&heap, &Words(Vec::new()), kind);
        assert!(heap
            .core
            .borrow()
            .owns_cell(array_word as usize, heap.property_layout_desc));
        assert!(heap
            .core
            .borrow()
            .owns_cell(shape_word as usize, heap.property_layout_desc));
    }
}

#[test]
fn deferred_prototype_retains_its_unused_layout_as_sole_holder() {
    let (heap, _release, _active) = isolated();
    let owner = Object::new(None);
    let parent = Object::new(None);
    let template = predicted("deferred prediction");
    let word = template.shared_layout().unwrap().cell_word();
    let lazy = crate::gc_box::new(LazyFunctionPrototype::new(&owner, parent, &template));
    drop(template);
    let root = Words(vec![crate::gc_box::GcBox::as_ptr(&lazy) as u64]);
    for kind in [CollectionKind::Major, CollectionKind::Minor] {
        collect(&heap, &root, kind);
        assert!(heap
            .core
            .borrow()
            .owns_cell(word as usize, heap.property_layout_desc));
    }
}

#[test]
fn actual_function_map_roots_cover_function_prototype_and_late_named_maps() {
    let (heap, _release, _active) = isolated();
    let maps = Rc::new(crate::ast::FunctionMaps {
        function: predicted("function map"),
        prototype: Some(predicted("prototype map")),
        named: RefCell::new(None),
        heap: heap_id(&heap),
    });
    register_code_roots(&maps);
    let function = maps.function.shared_layout().unwrap().cell_word();
    let prototype = maps
        .prototype
        .as_ref()
        .unwrap()
        .shared_layout()
        .unwrap()
        .cell_word();
    let weak_function = crate::gc_handle::WeakRef::to(function as usize);
    let weak_prototype = crate::gc_handle::WeakRef::to(prototype as usize);
    collect(&heap, &Words(Vec::new()), CollectionKind::Major);
    for word in [function, prototype] {
        assert!(heap
            .core
            .borrow()
            .owns_cell(word as usize, heap.property_layout_desc));
    }
    assert!(weak_function.live_ptr().is_some());
    assert!(weak_prototype.live_ptr().is_some());
    *maps.named.borrow_mut() = Some((Rc::from("inferred"), predicted("named map")));
    let words = [
        function,
        prototype,
        maps.named
            .borrow()
            .as_ref()
            .unwrap()
            .1
            .shared_layout()
            .unwrap()
            .cell_word(),
    ];
    for kind in [CollectionKind::Minor, CollectionKind::Major] {
        collect(&heap, &Words(Vec::new()), kind);
        for &word in &words {
            assert!(heap
                .core
                .borrow()
                .owns_cell(word as usize, heap.property_layout_desc));
        }
    }
    drop(maps);
    collect(&heap, &Words(Vec::new()), CollectionKind::Major);
    assert!(weak_function.live_ptr().is_none());
    assert!(weak_prototype.live_ptr().is_none());
    for word in words {
        assert!(!heap
            .core
            .borrow()
            .owns_cell(word as usize, heap.property_layout_desc));
    }
}

#[test]
fn layout_finalization_releases_keys_once_including_weakly_pinned_cells() {
    let (heap, _release, _active) = isolated();
    let key: Rc<str> = Rc::from("finalized key");
    let weak_key = Rc::downgrade(&key);
    let layout = new_property_layout(vec![key]);
    let weak_layout = crate::gc_handle::WeakRef::to(layout.as_ptr() as usize);
    drop(layout);
    collect(&heap, &Words(Vec::new()), CollectionKind::Minor);
    assert!(weak_layout.live_ptr().is_none());
    assert!(weak_key.upgrade().is_none());
    collect(&heap, &Words(Vec::new()), CollectionKind::Major);
    assert!(weak_key.upgrade().is_none());
}

#[test]
#[cfg(debug_assertions)]
fn private_layout_verification_deduplicates_visits_to_one_slot() {
    let (heap, _release, _active) = isolated();
    let props = predicted("one slot");
    check_layout_owners(&heap, &heap.core.borrow(), |sink| {
        props.trace_gc(sink);
        props.trace_gc(sink);
    });
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "has two owning slots")]
fn private_layout_verification_rejects_distinct_owning_slots() {
    let (heap, _release, _active) = isolated();
    let props = predicted("invalid duplicate");
    let layout = props.shared_layout().unwrap();
    // Deliberately model a missed shared-bit write; both handles remain live for the check.
    let duplicate = PropertyLayout(unsafe { crate::gc_box::GcBox::from_raw(layout.as_ptr()) });
    let other = Props::with_layout(1, Some(duplicate));
    check_layout_owners(&heap, &heap.core.borrow(), |sink| {
        props.trace_gc(sink);
        other.trace_gc(sink);
    });
}

#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn native_creation_records_an_old_receivers_only_edge_to_a_young_layout() {
    use crate::bytecode::Tier;
    use crate::{Completion, Engine};
    const CHILD: &str = "TRUST_JS_TEST_NATIVE_LAYOUT_CARD_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "value::property_layout_cell_tests::native_creation_records_an_old_receivers_only_edge_to_a_young_layout",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    // This fixture controls promotion explicitly; automatic polls must not age the layout
    // while the directory still holds it, including in an external GC_STRESS=1 run.
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
    engine.interp.gc_task_deferred = true;
    let eval = |engine: &mut Engine, source: &str| {
        assert!(matches!(
            engine.eval(source, false).unwrap(),
            Completion::Value(_)
        ));
    };
    eval(&mut engine, "var target={},prefix={alpha:0};function write(){target.alpha=7;target.beta=8;}for(var k=0;k<100;k++){target={};write();}0");
    let env = engine.interp.global_env;
    let target_value = engine.interp.get_var("target", &env).ok().unwrap();
    let target = target_value.as_obj().unwrap();
    let prefix_value = engine.interp.get_var("prefix", &env).ok().unwrap();
    let prefix = prefix_value.as_obj().unwrap();
    let shapes = [prefix.borrow().props.shape(), target.borrow().props.shape()];
    let heap = engine.interp.gc_heap.clone();
    for _ in 0..256 {
        let epoch = proto_epoch();
        target.borrow_mut().props = Props::with_capacity(3);
        eval(&mut engine, "write();0");
        target.borrow_mut().props = Props::with_capacity(3);
        engine
            .interp
            .gc_collect_precise_for_test(CollectionKind::Major);
        assert_eq!(
            heap.core.borrow().cell_age(Gc::as_ptr(&target) as u64),
            Some(crate::gc_core::CellAge::Old)
        );
        let layout = new_shared_indexed_property_layout(vec![Rc::from("alpha"), Rc::from("beta")]);
        let word = layout.cell_word();
        let weak = crate::gc_handle::WeakRef::to(word as usize);
        assert_eq!(
            heap.core.borrow().cell_age(word),
            Some(crate::gc_core::CellAge::Young)
        );
        {
            let mut table = heap.shapes.borrow_mut();
            for shape in shapes {
                table.layouts.insert(shape, layout.clone());
            }
        }
        drop(layout);
        let before = (
            crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(|count| count.get()),
            crate::bytecode::TEST_JIT_EXEC_PROPERTY_HELPERS.with(|count| count.get()),
        );
        eval(&mut engine, "write();0");
        if proto_epoch() != epoch {
            continue;
        }
        assert_eq!(
            before,
            (
                crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(|count| count.get()),
                crate::bytecode::TEST_JIT_EXEC_PROPERTY_HELPERS.with(|count| count.get()),
            ),
            "both creation stores must commit without property helpers"
        );
        assert_eq!(
            target.borrow().props.shared_layout().unwrap().cell_word(),
            word
        );
        {
            let mut table = heap.shapes.borrow_mut();
            for shape in shapes {
                let index = shape as usize & (SHAPE_LAYOUT_CACHE_LIMIT - 1);
                let page = table.layouts.pages[index >> property_shapes::SHAPE_LAYOUT_PAGE_BITS]
                    .as_mut()
                    .unwrap();
                let entry = &mut page[index & (SHAPE_LAYOUT_PAGE_SIZE - 1)];
                assert_eq!(entry.shape, shape);
                entry.keys = None;
            }
        }
        struct NoOtherRoot(u64);
        impl GcSink for NoOtherRoot {
            fn word(&mut self, word: u64) {
                assert_ne!(
                    word, self.0,
                    "all directory/template holders must be evicted"
                );
            }
        }
        trace_heap_roots(&heap, &mut NoOtherRoot(word));
        crate::gc_trace::trace_thread_roots(&mut NoOtherRoot(word));
        assert_eq!(
            heap.core.borrow().cell_age(word),
            Some(crate::gc_core::CellAge::Young),
            "the final minor must still require the old receiver's new edge"
        );
        // This complete interpreter collection ignores even the live local `word` above.
        engine
            .interp
            .gc_collect_precise_for_test(CollectionKind::Minor);
        assert!(weak.live_ptr().is_some());
        assert!(heap
            .core
            .borrow()
            .owns_cell(word as usize, heap.property_layout_desc));
        assert_eq!(
            heap.core.borrow().cell_age(word),
            Some(crate::gc_core::CellAge::Old)
        );
        return;
    }
    panic!("prototype epoch changed during every native creation trial");
}
