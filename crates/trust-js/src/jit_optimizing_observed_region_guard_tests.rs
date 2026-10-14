//! Integrated guard costs must describe checks emitted on the retained native path.
//! These manually installed versions prove selection/recovery, not hot publication.

use super::*;

struct Enabled(Option<bool>, Option<bool>);

impl Enabled {
    fn new() -> Self {
        Self(
            generic_own::TEST_ENABLED.with(|value| value.replace(Some(true))),
            object_facts::TEST_ENABLED.with(|value| value.replace(Some(true))),
        )
    }
}

impl Drop for Enabled {
    fn drop(&mut self) {
        generic_own::TEST_ENABLED.with(|value| value.set(self.0));
        object_facts::TEST_ENABLED.with(|value| value.set(self.1));
    }
}

fn own_read(chunk: &Chunk, name: &str) -> usize {
    chunk
        .jit_ops()
        .iter()
        .position(|op| {
            matches!(*op, Op::GetPropLocal(_, key, _) | Op::GetPropThis(key, _)
                if chunk.jit_name(key) == name)
        })
        .unwrap()
}

fn exact_transfer(pc: usize, depth: usize) {
    ENTRIES.with(|entries| assert_eq!(*entries.borrow(), [(pc, depth, 0)]));
}

#[test]
fn numeric_entry_and_object_result_guards_are_each_charged_once_before_a_cut() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let _enabled = Enabled::new();
    assert!(object_facts::enabled());
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        var receiver={alternate:{alive:1},source:'copied',x:0,a:0,b:0,c:0,d:0,e:0};
        function subject(o,n,cold){'use strict';
            var alternate=o.alternate;
            if(alternate===null)return -1;
            o.x=o.source;o.a=o.x;o.b=o.a;o.c=o.b;o.d=o.c;o.e=o.d;
            var total=n+n;
            if(cold){for(var k=0;k<1;k++){}return total+100;}
            return total;
        }
        subject(receiver,7,false);subject(receiver,7,false);
        "#,
    );
    let function = lookup_function(&mut engine, "subject");
    let primary = baseline(&function);
    let frozen = primary.execution_coverage.freeze();
    let cfg = Cfg::build(&primary).unwrap();
    let stack = stack::StackPlan::build(&primary, &cfg).unwrap();
    let regions = observed_regions::Plan::build(&primary, &cfg, &stack, &frozen).unwrap();
    let sites = generic_admission::selected(&generic_admission::observations(&primary))
        .into_iter()
        .filter(|&(pc, _)| regions.emission(pc as usize) == observed_regions::Emission::Native)
        .collect::<Vec<_>>();
    let object_pc = own_read(&primary, "alternate");
    let property_samples = [(object_pc as u32, crate::feedback::ValueClass::Object.bit())];
    let objects = object_facts::selected(&property_samples, &sites);
    assert_eq!(objects.len(), 1);
    let inputs = [(1, crate::feedback::ValueClass::NumberInt32.bit())];
    let numeric = specialization::Plan::build(&primary, &cfg, &stack, &inputs, &[]).unwrap();
    let generic = value_facts::ValuePlan::build(&primary, &cfg, &stack).unwrap();
    assert_eq!(numeric.inputs.len(), 1, "one live Number entry guard");
    assert!(numeric.properties.is_empty());
    let object = object_facts::Plan::build(
        &primary,
        &cfg,
        &stack,
        &objects,
        &sites,
        &numeric.inputs,
        &numeric.properties,
    )
    .unwrap();
    assert_eq!(object.guard_cost, 1, "one live PACK_OBJ result guard");
    let structural = generic_admission::Plan::with_objects_and_regions(
        &primary,
        &cfg,
        &stack,
        &sites,
        Some(&object),
        Some(&regions),
    )
    .unwrap();

    // The two owning formal loads precede the first potentially abrupt Add.
    // The first generic read admits owners and Empty, so lowering emits both
    // owner and initialized-value checks. Its normal successor removes Empty
    // from the local; the second read retains only the owner check. The live
    // Number guard removes all three checks. Later Add checks and the VM-only
    // arithmetic suffix cannot pay an early completion.
    let first_add = primary
        .jit_ops()
        .iter()
        .position(|op| matches!(op, Op::Add))
        .unwrap();
    let number_loads = primary.jit_ops()[..first_add]
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| matches!(op, Op::LoadLocal(1)).then_some(pc))
        .collect::<Vec<_>>();
    assert_eq!(number_loads.len(), 2);
    assert_eq!(number_loads[1], number_loads[0] + 1);
    for (index, &pc) in number_loads.iter().enumerate() {
        assert_eq!(regions.emission(pc), observed_regions::Emission::Native);
        if index == 0 {
            assert_eq!(
                generic.at[pc].local,
                value_facts::Types::ANY,
                "load={index} pc={pc}"
            );
        } else {
            assert_ne!(
                generic.at[pc].local,
                value_facts::Types::ANY,
                "load={index} pc={pc}"
            );
            assert_eq!(
                generic.at[pc].local, generic.at[pc].top,
                "load={index} pc={pc}: normal prior read refined both its copied result and local"
            );
        }
        assert!(!generic.at[pc].local.is_copyable(), "load={index} pc={pc}");
        assert_eq!(
            generic.at[pc].local.initialized(),
            index != 0,
            "load={index} pc={pc}"
        );
        assert!(numeric.values.at[pc].local.is_number());
        assert!(numeric.values.at[pc].local.is_copyable());
        assert!(numeric.values.at[pc].local.initialized());
        assert_eq!(object.values.at[pc].local, numeric.values.at[pc].local);
        assert_eq!(
            numeric
                .opportunities
                .iter()
                .find(|opportunity| opportunity.pc == pc)
                .unwrap()
                .removed_tests,
            [2, 1][index],
            "load={index} pc={pc}: the first read proves initialization for the second"
        );
    }
    let structural_only = regions
        .minimum_saved_tests(&primary, &cfg, None, Some(&structural), Some(&object))
        .unwrap();
    assert_eq!(structural_only, structural.minimum_tests);
    let combined = regions
        .minimum_saved_tests(
            &primary,
            &cfg,
            Some(&numeric),
            Some(&structural),
            Some(&object),
        )
        .unwrap();
    assert_eq!(
        combined,
        structural_only + 3 - 1,
        "Object guard is already debited once; three removed Number load checks pay one entry guard"
    );
    assert!(
        numeric.opportunities.iter().any(|opportunity| {
            regions.emission(opportunity.pc) != observed_regions::Emission::Native
        }),
        "there is real uncredited numeric work in the VM suffix"
    );

    let mut candidate = bytecode::compile_for_optimizer_with_objects_and_coverage(
        &function,
        &primary,
        &inputs,
        &property_samples,
        &structural.selected,
        &objects,
        &frozen,
    )
    .unwrap();
    Rc::get_mut(&mut candidate).unwrap().optimizing_candidate = false;
    object_facts::TEST_EMITTED.with(|counts| counts.set([0; 3]));
    TEST_VALUE_PROOFS.with(|counts| counts.set([0; 4]));
    let code = compile(&mut engine, &candidate, true);
    assert_eq!(object_facts::TEST_EMITTED.with(|counts| counts.get()[0]), 1);
    assert!(TEST_VALUE_PROOFS.with(|counts| counts.get()[2] >= 2));
    let lowered = observed_regions::TEST_LOWERED.with(|pcs| pcs.borrow().clone());
    for pc in number_loads {
        assert!(lowered.contains(&pc));
    }
    for opportunity in &numeric.opportunities {
        if regions.emission(opportunity.pc) != observed_regions::Emission::Native {
            assert!(!lowered.contains(&opportunity.pc));
        }
    }
    install(&mut engine, &function, &candidate, &code);
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    clear_entries();
    assert_eq!(
        eval(&mut engine, "subject(receiver,7,false)+'|'+receiver.e"),
        "14|copied"
    );
    assert_eq!(record.counters[0].get(), 1, "actual native entry");
    ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));

    clear_entries();
    assert_eq!(
        eval(&mut engine, "subject(receiver,7,true)+'|'+receiver.e"),
        "114|copied"
    );
    let cut = assert_one_cut(&engine, &candidate, &code, &regions);
    exact_transfer(cut, stack.at[cut].depth);
    assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 1);
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        0
    );

    clear_entries();
    assert_eq!(
        eval(&mut engine, "subject(receiver,'7',false)+'|'+receiver.e"),
        "77|copied"
    );
    exact_transfer(0, stack.at[0].depth);
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        1
    );
    assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 1);

    clear_entries();
    assert_eq!(
        eval(
            &mut engine,
            "receiver.alternate=null;subject(receiver,7,false)"
        ),
        "-1"
    );
    exact_transfer(object_pc, stack.at[object_pc].depth);
    assert_eq!(record.counters[0].get(), 4);
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        2
    );
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 3);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY_NORMAL].get(), 3);
    assert_eq!(candidate.optimizing_misses.get(), 2);
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
    eprintln!("combined region guards: structural={structural_only} numeric_prefix=3 number_guards=1 object_guards=1 saved={combined}");
}

#[test]
fn omitted_warmed_object_read_cannot_seed_native_facts_or_guard_cost() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let _enabled = Enabled::new();
    assert!(object_facts::enabled());
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        var receiver={alternate:{alive:1},source:'copied',x:0,a:0,b:0,c:0,d:0,e:0};
        function subject(o,cold){'use strict';
            o.x=o.source;o.a=o.x;o.b=o.a;o.c=o.b;o.d=o.c;o.e=o.d;
            if(cold){o.e=42;var n=o.alternate;if(n===null)return 'cold:null';return 'cold:object';}
            return 'hot';
        }
        subject(receiver,false);
        "#,
    );
    let function = lookup_function(&mut engine, "subject");
    let env = engine.interp.global_env.clone();
    let receiver = engine.interp.get_var("receiver", &env).ok().unwrap();
    let mut primary = bytecode::compile(&function).unwrap();
    let run = |engine: &mut Engine, chunk: &Chunk, cold| {
        bytecode::run(
            &mut engine.interp,
            chunk,
            &env,
            JsValue::Undefined,
            &[receiver.clone(), JsValue::Bool(cold)],
        )
        .ok()
        .unwrap()
    };
    assert_eq!(
        run(&mut engine, &primary, true).as_engine_text(),
        Some("cold:object")
    );
    assert_eq!(
        run(&mut engine, &primary, false).as_engine_text(),
        Some("hot")
    );
    // Preserve genuinely warmed scalar ICs, but start a new test-only coverage
    // window on the identical immutable graph, as in the VM coverage fixtures.
    // This models pre-window cache state; it introduces no production reset API.
    {
        let chunk = Rc::get_mut(&mut primary).unwrap();
        chunk.execution_coverage = crate::execution_coverage::Handle::new(chunk);
    }
    assert_eq!(
        run(&mut engine, &primary, false).as_engine_text(),
        Some("hot")
    );
    assert_eq!(
        run(&mut engine, &primary, false).as_engine_text(),
        Some("hot")
    );
    let frozen = primary.execution_coverage.freeze();
    let cfg = Cfg::build(&primary).unwrap();
    let stack = stack::StackPlan::build(&primary, &cfg).unwrap();
    let regions = observed_regions::Plan::build(&primary, &cfg, &stack, &frozen).unwrap();
    let object_pc = own_read(&primary, "alternate");
    assert_eq!(
        regions.emission(object_pc),
        observed_regions::Emission::OmittedInterior
    );
    let all_sites = generic_admission::selected(&generic_admission::observations(&primary));
    let properties = [(object_pc as u32, crate::feedback::ValueClass::Object.bit())];
    assert_eq!(
        object_facts::selected(&properties, &all_sites).len(),
        1,
        "the warmed own-data IC and sampled live Object would otherwise be selectable"
    );
    let native_sites = all_sites
        .iter()
        .copied()
        .filter(|&(pc, _)| regions.emission(pc as usize) == observed_regions::Emission::Native)
        .collect::<Vec<_>>();
    let objects = object_facts::selected(&properties, &native_sites);
    assert!(
        objects.is_empty(),
        "an un-emitted read supplies no guarded native category"
    );
    assert!(
        object_facts::Plan::build(&primary, &cfg, &stack, &objects, &native_sites, &[], &[])
            .is_none()
    );
    let structural = generic_admission::Plan::with_objects_and_regions(
        &primary,
        &cfg,
        &stack,
        &native_sites,
        None,
        Some(&regions),
    )
    .unwrap();
    assert!(structural.per_pc[object_pc].is_none());
    assert_eq!(
        regions.minimum_saved_tests(&primary, &cfg, None, Some(&structural), None),
        Some(structural.minimum_tests),
        "no nonexistent Object guard is charged"
    );

    let mut candidate = bytecode::compile_for_optimizer_with_objects_and_coverage(
        &function,
        &primary,
        &[],
        &properties,
        &structural.selected,
        &objects,
        &frozen,
    )
    .unwrap();
    Rc::get_mut(&mut candidate).unwrap().optimizing_candidate = false;
    assert!(candidate.optimizing_objects.is_empty());
    object_facts::TEST_EMITTED.with(|counts| counts.set([0; 3]));
    let code = compile(&mut engine, &candidate, true);
    assert_eq!(object_facts::TEST_EMITTED.with(Cell::get), [0; 3]);
    let lowered = observed_regions::TEST_LOWERED.with(|pcs| pcs.borrow().clone());
    assert!(!lowered.contains(&object_pc));
    assert!(structural
        .selected
        .iter()
        .all(|&(pc, _)| lowered.contains(&(pc as usize))));
    install(&mut engine, &function, &candidate, &code);
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    clear_entries();
    assert_eq!(
        eval(&mut engine, "subject(receiver,false)+'|'+receiver.e"),
        "hot|copied"
    );
    assert_eq!(record.counters[0].get(), 1);
    ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));

    clear_entries();
    assert_eq!(
        eval(&mut engine, "subject(receiver,true)+'|'+receiver.e"),
        "cold:object|42"
    );
    let cut = assert_one_cut(&engine, &candidate, &code, &regions);
    exact_transfer(cut, stack.at[cut].depth);
    assert!(
        cut < object_pc,
        "one before-effect rescue reaches the later live field read"
    );
    clear_entries();
    assert_eq!(
        eval(
            &mut engine,
            "receiver.alternate=null;subject(receiver,true)+'|'+receiver.e"
        ),
        "cold:null|42"
    );
    exact_transfer(cut, stack.at[cut].depth);
    assert_eq!(record.counters[0].get(), 3);
    assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 2);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 2);
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        0
    );
    assert_eq!(candidate.optimizing_misses.get(), 0);
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
}
