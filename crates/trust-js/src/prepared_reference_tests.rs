use super::*;
use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine
            .eval(source, false)
            .expect("reference fixture parses")
        {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}: {source}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

#[test]
fn prepared_reference_recreates_deleted_binding_in_captured_record() {
    // The example from DeclarativeEnvironmentRecord.SetMutableBinding, extended with an
    // outer binding to distinguish recreating this record from repeating name resolution.
    check(
        r#"
        var x='outer';
        function simple(){return eval("var x=1; x=(delete x,7); x;")}
        function compound(){return eval("var x=3; x+=(delete x,4); x;")}
        function update(){return eval("var x={valueOf(){delete x;return 8}}; x++; x;")}
        [simple(),compound(),update(),x].join('|');
    "#,
        "7|7|9|outer",
    );
}

#[test]
fn prepared_reference_update_keeps_with_base_after_numeric_coercion() {
    check(
        r#"
        var x='outer', trace=[], box;
        box={x:{valueOf(){trace.push('number');delete box.x;return 8}}};
        var proxy=new Proxy(box,{has(t,k){
            if(k==='x')trace.push('has');return Reflect.has(t,k);
        }});
        with(proxy){x++}
        [box.x,x,trace.join(',')].join('|');
    "#,
        "9|outer|has,has,number,has",
    );
    check(
        r#"
        var x='outer', box={x:1}, old;
        with(box){old=(x += (delete box.x, Object.defineProperty(box,Symbol.unscopables,
            {value:{x:true}}), 4));}
        [old,box.x,x].join('|');
    "#,
        "5|5|outer",
    );
}

#[test]
fn prepared_reference_unresolvable_base_survives_rhs_binding_creation() {
    check(
        r#"
        var box={};
        with(box){unresolved=(box.unresolved='inner',17);}
        [unresolved,box.unresolved].join('|');
    "#,
        "17|inner",
    );
}

#[test]
fn prepared_reference_does_not_read_or_reject_binding_during_capture() {
    check(
        r#"
        var trace=[];
        try { x=(trace.push('rhs'),1); let x; }
        catch(e){trace.push(e.name)}
        const y=1;
        try { y=(trace.push('const-rhs'),2); }
        catch(e){trace.push(e.name)}
        trace.join('|');
    "#,
        "rhs|ReferenceError|const-rhs|TypeError",
    );
}

#[test]
fn prepared_reference_captures_strictness_and_survives_generation_exhaustion() {
    let mut engine = Engine::new();
    let scope = new_scope(Some(engine.interp.global_env.clone()));
    scope
        .borrow_mut()
        .vars
        .insert("value", Binding::data(Value::Num(1.0), true, true));
    scope
        .borrow_mut()
        .vars
        .set_generation_for_test(u32::MAX - 1);
    engine.interp.strict = false;
    let name: Rc<str> = Rc::from("value");
    let mut reference = engine
        .interp
        .prepare_shared_name_reference(name.clone(), &scope)
        .ok()
        .unwrap();
    assert!(
        Rc::ptr_eq(&reference.name, &name),
        "capture shares the interned name"
    );
    scope.borrow_mut().vars.remove("value");
    assert_eq!(scope.borrow().vars.generation(), u32::MAX);
    engine.interp.strict = true;
    assert!(engine
        .interp
        .write_prepared_reference(&mut reference, Value::Num(7.0))
        .is_ok());
    assert_eq!(scope.borrow().vars.generation(), u32::MAX);
    assert!(!scope.borrow().vars.matches_generation(u32::MAX));
    // Move/replace the map repeatedly after saturation; no address may be reused as proof.
    for index in 0..64 {
        scope.borrow_mut().vars.insert(
            format!("other{index}"),
            Binding::data(Value::Undefined, true, true),
        );
    }
    assert!(matches!(
        engine.interp.read_prepared_reference(&mut reference),
        Ok(Value::Num(7.0))
    ));
    let mut strict = engine
        .interp
        .prepare_shared_name_reference(name, &scope)
        .ok()
        .unwrap();
    scope.borrow_mut().vars.remove("value");
    engine.interp.strict = false;
    assert!(engine
        .interp
        .write_prepared_reference(&mut strict, Value::Num(9.0))
        .is_err());
    assert!(!scope.borrow().vars.contains_key("value"));
}

#[test]
fn prepared_reference_imports_are_live_and_never_mutable_address_hints() {
    let mut engine = Engine::new();
    let exporter = new_scope(None);
    exporter
        .borrow_mut()
        .vars
        .insert("source", Binding::data(Value::Num(7.0), true, true));
    let importer = new_scope(None);
    let mut binding = Binding::data(Value::Undefined, false, true);
    binding.set_import_reference(Some((exporter.clone(), "source".into())));
    importer.borrow_mut().vars.insert("alias", binding);
    let mut reference = engine
        .interp
        .prepare_name_reference("alias", &importer)
        .ok()
        .unwrap();
    assert!(matches!(
        engine.interp.read_prepared_reference(&mut reference),
        Ok(Value::Num(7.0))
    ));
    exporter.borrow_mut().vars.get_mut("source").unwrap().value = Value::Num(19.0);
    assert!(matches!(
        engine.interp.read_prepared_reference(&mut reference),
        Ok(Value::Num(19.0))
    ));
    assert!(engine
        .interp
        .write_prepared_reference(&mut reference, Value::Num(31.0))
        .is_err());
    assert!(matches!(
        exporter.borrow().vars.get("source").unwrap().value,
        Value::Num(19.0)
    ));
}

#[test]
fn prepared_reference_object_record_get_and_set_repeat_has_not_unscopables() {
    check(
        r#"
        var trace=[], box={x:1};
        var proxy=new Proxy(box,{
            has(t,k){if(k==='x')trace.push('has');return Reflect.has(t,k)},
            get(t,k,r){if(k===Symbol.unscopables)trace.push('unscopables');
                if(k==='x')trace.push('get');return Reflect.get(t,k,r)},
            set(t,k,v,r){if(k==='x')trace.push('set');return Reflect.set(t,k,v,r)}
        });
        with(proxy){x += (trace.push('rhs'),2);}
        [box.x,trace.join(',')].join('|');
    "#,
        "3|has,unscopables,has,get,rhs,has,set",
    );
}

/// gc-heap: a slot reports its Reference's environment to the collector, and replacing the
/// Reference drops the old one in place: once its handles are gone a collection finds the old
/// environment dead, while the slot keeps the new one alive.
#[cfg(feature = "gc-heap")]
#[test]
fn prepared_reference_slot_reports_its_reference_and_releases_a_replaced_one() {
    struct Words(Vec<u64>);
    impl crate::gc_trace::GcSink for Words {
        fn word(&mut self, word: u64) {
            self.0.push(word);
        }
    }
    let mut engine = Engine::new();
    let mut slot = PreparedReferenceSlot::default();
    assert!(slot.as_ref().is_none());
    // Each environment's handles live only in a helper frame (gc-heap scans the stack).
    let weak = crate::gc_test::in_frame(|| {
        let scope = new_scope(None);
        scope
            .borrow_mut()
            .vars
            .insert("x", Binding::data(Value::Num(1.0), true, true));
        slot.set(PreparedReference::scope(
            Rc::from("x"),
            scope.clone(),
            false,
        ));
        Env::downgrade(&scope)
    });
    let next_weak = crate::gc_test::in_frame(|| {
        let next = new_scope(None);
        next.borrow_mut()
            .vars
            .insert("x", Binding::data(Value::Num(2.0), true, true));
        slot.set(PreparedReference::scope(Rc::from("x"), next.clone(), true));
        Env::downgrade(&next)
    });
    engine.interp.gc_collect();
    assert!(
        !crate::gc_test::alive(&weak),
        "the replaced Reference released its environment"
    );
    assert!(
        crate::gc_test::alive(&next_weak),
        "the slot keeps its live Reference"
    );
    let reported = crate::gc_test::in_frame(|| {
        let mut words = Words(Vec::new());
        crate::gc_trace::GcTrace::trace_gc(&slot, &mut words);
        words.0 == [Env::as_ptr(&next_weak.upgrade().unwrap()) as u64]
    });
    assert!(
        reported,
        "the slot reports exactly its Reference's environment"
    );
}

#[test]
fn prepared_reference_native_kind_flags_use_compact_binding_size_and_live_imports() {
    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        std::mem::size_of::<Binding>(),
        32,
        "kind bit uses flag padding in the compact binding"
    );
    let target = new_scope(None);
    let mut binding = Binding::data(Value::Undefined, false, true);
    assert!(!binding.imported);
    binding.set_import_reference(Some((target, "exported".into())));
    assert!(binding.imported);
    binding.set_import_reference(None);
    assert!(!binding.imported);
    // The entries validate: an InlineVec keeping a map's first binding in place.
    assert!(
        VarMap::jit_small_storage_layout().is_some(),
        "current Rust small map layout validates"
    );
}
