use super::*;
use crate::{bytecode::Tier, feedback::ValueClass, value::Callable, Completion, Engine};
use std::rc::Rc;

struct Enabled(Option<bool>);
impl Enabled {
    fn new(value: bool) -> Self {
        Self(generic_own::TEST_ENABLED.with(|cell| cell.replace(Some(value))))
    }
}
impl Drop for Enabled {
    fn drop(&mut self) {
        generic_own::TEST_ENABLED.with(|cell| cell.set(self.0));
    }
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("admission fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn function(engine: &mut Engine) -> Rc<crate::ast::Function> {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("author function")
    };
    user.func.clone()
}

fn warm(source: &str, run: &str) -> (Engine, Rc<crate::ast::Function>, Rc<Chunk>) {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    eval(&mut engine, run);
    let function = function(&mut engine);
    let chunk = function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    (engine, function, chunk)
}

fn selection(chunk: &Chunk) -> Vec<(u32, bytecode::IcState)> {
    let observed = generic_admission::observations(chunk);
    for _ in 0..4 {
        generic_admission::observe(chunk, &observed);
    }
    generic_admission::selected(&observed)
}

fn plan(chunk: &Chunk, selected: &[(u32, bytecode::IcState)]) -> Option<generic_admission::Plan> {
    let cfg = Cfg::build(chunk).unwrap();
    let stack = stack::StackPlan::build(chunk, &cfg).unwrap();
    generic_admission::Plan::build(chunk, &cfg, &stack, selected)
}

#[test]
fn structural_minimum_rejects_optional_work_and_counts_both_branches_conservatively() {
    let _enabled = Enabled::new(true);
    for (source, warmup, positive) in [
        ("function subject(o,b){'use strict';if(b){o.x=o.x;return o.x;}return 0;}var receiver={x:'v'};",
            "subject(receiver,true);subject(receiver,false)", false),
        ("function subject(o,b){'use strict';if(b){o.x=o.x;return o.x;}o.y=o.y;return o.y;}var receiver={x:'v',y:'w'};",
            "subject(receiver,true);subject(receiver,false)", true),
        ("function subject(o){return o.x;}var receiver={x:'v'};", "subject(receiver)", false),
        ("function subject(o,b){'use strict';while(b){o.x=o.x;b=false;}return 0;}var receiver={x:'v'};",
            "subject(receiver,true);subject(receiver,false)", false),
        ("function subject(o){'use strict';try{o.x=o.x;return o.x;}catch(e){return 0;}}var receiver={x:'v'};",
            "subject(receiver)", false),
        ("function subject(o,b){'use strict';if(b)throw 0;o.x=o.x;return o.x;}var receiver={x:'v'};",
            "subject(receiver,false);try{subject(receiver,true)}catch(e){}", false),
        ("function subject(o,b){'use strict';o.x=o.x;var value=o.x;if(b)throw value;return value;}var receiver={x:'v'};",
            "subject(receiver,false);try{subject(receiver,true)}catch(e){}", true),
        ("function subject(o,callback){'use strict';callback();o.x=o.x;return o.x;}var receiver={x:'v'};function callback(){}",
            "subject(receiver,callback)", false),
        ("function subject(o,callback){'use strict';o.x=o.x;var value=o.x;callback();return value;}var receiver={x:'v'};function callback(){}",
            "subject(receiver,callback)", true),
        ("function subject(o,unknown){'use strict';var ignored=+unknown;o.x=o.x;return o.x;}var receiver={x:'v'},unknown={valueOf(){return 1}};",
            "subject(receiver,unknown)", false),
        ("function subject(o){'use strict';var ignored=o.observed;o.x=o.x;return o.x;}var receiver={x:'v',get observed(){return 1}};",
            "subject(receiver)", false),
        ("function subject(o,callback){'use strict';o.x=o.x;var value=o.x;callback();o.y=o.y;return o.y;}var receiver={x:'v',y:'w'};function callback(){}",
            "subject(receiver,callback)", true),
    ] {
        let (_, _, chunk) = warm(source, warmup);
        let selected = selection(&chunk);
        let result = plan(&chunk, &selected);
        if generic_admission::enabled() && positive {
            assert_eq!(result.unwrap().minimum_tests, 11, "one branch, not the sum of both");
        } else {
            assert!(result.is_none());
        }
    }
}

#[test]
fn structural_snapshots_are_bounded_permanently_stable_and_revalidated_before_emission() {
    let _enabled = Enabled::new(true);
    let (mut engine, function, chunk) = warm(
        "function subject(o){'use strict';o.x=o.x;return o.x;}var receiver={x:'value'};",
        "subject(receiver)",
    );
    let observed = generic_admission::observations(&chunk);
    let selected = generic_admission::selected(&observed);
    if !generic_admission::enabled() {
        assert!(selected.is_empty());
        return;
    }
    assert!(plan(&chunk, &selected).is_some());
    let mut stale = selected.clone();
    stale[0].1.recv_shape = stale[0].1.recv_shape.wrapping_add(1);
    assert!(plan(&chunk, &stale).is_none());
    let mut duplicate = selected.clone();
    duplicate.insert(0, duplicate[0]);
    assert!(plan(&chunk, &duplicate).is_none());
    assert!(plan(&chunk, &vec![selected[0]; 33]).is_none());
    let candidate =
        bytecode::compile_for_optimizer_with_structural(&function, &chunk, &[], &[], &selected)
            .unwrap();
    assert!(candidate.optimizing_candidate);
    assert_eq!(candidate.optimizing_structural.len(), selected.len());
    for (&(pc, state), &(other_pc, other)) in selected.iter().zip(&*candidate.optimizing_structural)
    {
        assert_eq!(pc, other_pc);
        assert!(generic_admission::same(state, other));
    }
    eval(&mut engine, "subject({extra:1,x:'other'})");
    generic_admission::observe(&chunk, &observed);
    assert!(
        generic_admission::selected(&observed).is_empty(),
        "a second warm shape stops selection"
    );
    eval(&mut engine, "subject(receiver)");
    generic_admission::observe(&chunk, &observed);
    assert!(
        generic_admission::selected(&observed).is_empty(),
        "rejection is permanent"
    );
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    // The candidate owns its copied scalar ICs, not the mutated source cache.
    assert!(
        plan(&candidate, &candidate.optimizing_structural).is_some(),
        "copied IC selection remains valid"
    );
    // Code size may improve independently; the invariant is the unchanged cap,
    // not a historical expectation that this particular small body must fail.
    stores::TEST_BODY_SIZE.with(|size| size.set((0, 0)));
    match compile_with_deopt(&candidate, &values, &layout, None) {
        Ok(code) => {
            assert!(code.pc_offsets.is_empty(), "actual optimizing code");
            let (instructions, _) = stores::TEST_BODY_SIZE.with(|size| size.get());
            let limit = candidate
                .jit_ops()
                .len()
                .saturating_mul(40)
                .saturating_add(names::instruction_allowance(&candidate))
                .min(16_384);
            assert!(
                instructions > 0 && instructions <= limit,
                "finalized IR {instructions} exceeds actual candidate limit {limit}"
            );
        }
        Err(CompileError::LoweringBudget) => {}
        Err(other) => panic!("unexpected compile failure: {other}"),
    }

    let _disabled = Enabled::new(false);
    assert!(
        compile_with_deopt(&candidate, &values, &layout, None).is_err(),
        "selection cannot silently lose its credited mechanism"
    );
}

#[test]
fn structural_and_numeric_benefits_compose_without_changing_numeric_only_distance() {
    let _enabled = Enabled::new(true);
    let (_, _, chunk) = warm(
        "function subject(o,x){'use strict';o.x=o.x;return o.x+(x*2+1);}var receiver={x:'value'};",
        "subject(receiver,3)",
    );
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let numeric = specialization::Plan::build(
        &chunk,
        &cfg,
        &stack,
        &[(1, ValueClass::NumberInt32.bit())],
        &[],
    )
    .unwrap();
    assert_eq!(
        generic_admission::admission_entries(&chunk, numeric.saved_tests, 0),
        Some(numeric.admission_entries(&chunk))
    );
    if generic_admission::enabled() {
        let structural = plan(&chunk, &selection(&chunk)).unwrap();
        assert!(
            generic_admission::admission_entries(
                &chunk,
                numeric.saved_tests,
                structural.minimum_tests
            )
            .unwrap()
                < numeric.admission_entries(&chunk)
        );
    }
    assert_eq!(generic_admission::admission_entries(&chunk, 0, 0), None);
}

fn child(name: &str) -> bool {
    const CHILD: &str = "TRUST_JS_TEST_STRUCTURAL_ADMISSION_CHILD";
    if std::env::var(CHILD).as_deref() == Ok(name) {
        return false;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            &format!("jit::optimizing::generic_admission_tests::{name}"),
            "--nocapture",
        ])
        .env(CHILD, name)
        .env("TRUST_JS_OPT_JIT", "hot")
        .env("TRUST_JS_OPT_JIT_DIAGNOSTICS", "compile")
        .env("TRUST_JS_OPT_JIT_HOT_AT", "8")
        .env("TRUST_JS_INLINE_AT", "0");
    if matches!(
        name,
        "generic_only_hot_admission_executes_then_recovers_and_retires_exactly"
            | "prepared_inline_snapshot_preserves_selected_schema_and_executes_native_guards"
    ) {
        command.env_remove("TRUST_JS_OPT_JIT_HOT_AT");
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    true
}

#[test]
fn prepared_inline_snapshot_preserves_selected_schema_and_executes_native_guards() {
    if child("prepared_inline_snapshot_preserves_selected_schema_and_executes_native_guards") {
        return;
    }
    let _enabled = Enabled::new(true);
    if !generic_admission::enabled() {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, "function leaf(n){return n*2+1}
        function finishCopy(o){return [o.source,o.x,o.a,o.b,o.c,o.d,o.e].join('|')}
        function subject(){this.x=this.source;this.a=this.x;this.b=this.a;this.c=this.b;this.d=this.c;this.e=this.d;eval('0');return leaf(3)+finishCopy(this)}
        var receiver={source:'value',x:0,a:1,b:2,c:3,d:4,e:5};");
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let user = {
        let object = value.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("author function")
        };
        user.clone()
    };
    let function = user.func.clone();
    let primary = bytecode::compile(&function).unwrap();
    assert!(
        primary.prepared_entry,
        "direct eval retains a real prepared activation"
    );
    primary.inline_attempted.set(true);
    primary.inline_retry_at.set(0);
    assert!(function.code.set(Some(primary.clone())).is_ok());
    for _ in 0..8 {
        eval(&mut engine, "subject.call(receiver)");
    }
    let inline = bytecode::plan_inlines(&primary, &function, user.realm, &engine.interp.global_env);
    assert!(!inline.is_empty(), "actual warmed inline call feedback");
    let flattened = bytecode::compile_with_inlines(&function, &inline, &primary).unwrap();
    assert!(flattened.prepared_entry);
    assert!(flattened
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::InlineGuard(..))));
    assert!(function.code2.set(Some(flattened.clone())).is_ok());
    bytecode::invalidate_call_caches();
    for _ in 0..8 {
        eval(&mut engine, "subject.call(receiver)");
    }
    let selected = selection(&flattened);
    assert!(plan(&flattened, &selected).is_some());
    let candidate =
        bytecode::compile_for_optimizer_with_structural(&function, &flattened, &[], &[], &selected)
            .unwrap();
    assert!(candidate.prepared_entry);
    assert_eq!(candidate.jit_ops(), flattened.jit_ops());
    assert_eq!(candidate.jit_frame(), flattened.jit_frame());
    let cfg = Cfg::build(&candidate).unwrap();
    let stack = stack::StackPlan::build(&candidate, &cfg).unwrap();
    assert!(
        specialization::Plan::build(
            &candidate,
            &cfg,
            &stack,
            &candidate.optimizing_inputs,
            &candidate.optimizing_properties
        )
        .is_none(),
        "structural candidate does not borrow a numeric admission gate"
    );
    for &(pc, state) in &selected {
        let op = &flattened.jit_ops()[pc as usize];
        assert_eq!(
            candidate.jit_ops()[pc as usize],
            *op,
            "opcode, receiver slot and cache index"
        );
        let name = match *op {
            Op::GetPropLocal(_, name, _)
            | Op::SetPropLocalDrop(_, name, _)
            | Op::GetPropThis(name, _)
            | Op::SetPropThisDrop(name, _) => name,
            _ => panic!("selected own-data access"),
        };
        assert_eq!(
            candidate.jit_name(name),
            flattened.jit_name(name),
            "exact key schema"
        );
        assert!(candidate
            .optimizing_structural
            .iter()
            .any(|&(other_pc, other)| other_pc == pc && generic_admission::same(state, other)));
    }
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = Rc::new(
        compile_with_deopt(&candidate, &values, &layout, None)
            .expect("prepared inline structural candidate fits unchanged IR budget"),
    );
    assert!(code.pc_offsets.is_empty());
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    let limit =
        (candidate.jit_ops().len() * 40 + names::instruction_allowance(&candidate)).min(16_384);
    assert!(
        !record.runtime_counters
            && record.ir_instructions.get() > 0
            && record.ir_instructions.get() <= limit
    );
    eprintln!(
        "manual prepared-inline schema ops={} IR={} limit={} minimum_tests={} selected_sites={}",
        candidate.jit_ops().len(),
        record.ir_instructions.get(),
        limit,
        plan(&candidate, &candidate.optimizing_structural)
            .unwrap()
            .minimum_tests,
        candidate.optimizing_structural.len()
    );
    candidate
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function
        .optimizing_code
        .set(Some(candidate.clone()))
        .is_ok());
    bytecode::invalidate_call_caches();
    code.residency.referenced.set(0);
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    assert_eq!(
        eval(&mut engine, "subject.call(receiver)"),
        "7value|value|value|value|value|value|value"
    );
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(candidate.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    assert_eq!(eval(&mut engine, "var gets=0;Object.defineProperty(receiver,'source',{get(){gets++;return 'changed'},configurable:true});subject.call(receiver)+'|'+gets"),
        "7changed|changed|changed|changed|changed|changed|changed|2");
    assert_eq!(candidate.optimizing_misses.get(), 1);
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
}

#[test]
fn generic_only_hot_admission_executes_then_recovers_and_retires_exactly() {
    if child("generic_only_hot_admission_executes_then_recovers_and_retires_exactly") {
        return;
    }
    let _enabled = Enabled::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(eval(&mut engine,
        "function finishCopy(o){return [o.source,o.x,o.a,o.b,o.c,o.d,o.e].join('|')}
         function subject(o){'use strict';o.x=o.source;o.a=o.x;o.b=o.a;o.c=o.b;o.d=o.c;o.e=o.d;return finishCopy(o)}
         var receiver={source:'value',x:0,a:1,b:2,c:3,d:4,e:5};for(var k=0;k<32000;k++)subject(receiver);subject(receiver)"),
         "value|value|value|value|value|value|value");
    let function = function(&mut engine);
    if !generic_admission::enabled() {
        assert!(function.optimizing_code.get().is_none());
        return;
    }
    let primary = function.code.get().and_then(Option::as_ref).unwrap();
    let candidate = function
        .optimizing_code
        .get()
        .and_then(Option::as_ref)
        .expect("generic-only hot publication");
    assert!(candidate.optimizing_candidate);
    assert!(!candidate.optimizing_structural.is_empty());
    let cfg = Cfg::build(candidate).unwrap();
    let stack = stack::StackPlan::build(candidate, &cfg).unwrap();
    assert!(specialization::Plan::build(
        candidate,
        &cfg,
        &stack,
        &candidate.optimizing_inputs,
        &candidate.optimizing_properties
    )
    .is_none());
    let code = candidate.jit.get().flatten().unwrap();
    assert!(code.pc_offsets.is_empty());
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert!(!record.runtime_counters);
    let limit =
        (candidate.jit_ops().len() * 40 + names::instruction_allowance(candidate)).min(16_384);
    assert!(record.ir_instructions.get() > 0 && record.ir_instructions.get() <= limit);
    let benefit = plan(candidate, &candidate.optimizing_structural).unwrap();
    eprintln!(
        "normal-policy generic-only ops={} IR={} limit={} minimum_tests={} baseline_entries={}",
        candidate.jit_ops().len(),
        record.ir_instructions.get(),
        limit,
        benefit.minimum_tests,
        primary.optimizing_samples.get().unwrap().entries.get()
    );
    code.residency.referenced.set(0);
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    assert_eq!(
        eval(&mut engine, "subject(receiver)"),
        "value|value|value|value|value|value|value"
    );
    assert!(
        code.residency.referenced.get() > 0,
        "actual optimizing entry"
    );
    assert_eq!(candidate.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    assert_eq!(eval(&mut engine,
        "var sources=0,gets=0,sets=0;Object.defineProperty(receiver,'source',{get(){sources++;return 'changed'},configurable:true});
         Object.defineProperty(receiver,'x',{get(){gets++;return 'changed'},set(v){sets++;},configurable:true});
         for(var k=0;k<9;k++)subject(receiver);[sources,gets,sets,receiver.a,receiver.b,receiver.c,receiver.d,receiver.e].join('|')"),
         "18|18|9|changed|changed|changed|changed|changed");
    assert_eq!(candidate.optimizing_misses.get(), SPECIALIZATION_MISS_LIMIT);
    assert!(Rc::ptr_eq(
        function.execution_code().and_then(Option::as_ref).unwrap(),
        primary
    ));
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
}

#[test]
fn mixed_hot_admission_emits_both_plans_and_disabled_policy_stays_numeric_only() {
    if child("mixed_hot_admission_emits_both_plans_and_disabled_policy_stays_numeric_only") {
        return;
    }
    for enabled in [true, false] {
        let _enabled = Enabled::new(enabled);
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine,
            "function subject(o,x){'use strict';o.x=o.x;return o.x+(x*2+1);}var receiver={x:'value'};for(var k=0;k<32;k++)subject(receiver,k);subject(receiver,3)"), "value7");
        let function = function(&mut engine);
        if !generic_admission::enabled()
            && std::env::var("TRUST_JS_OPT_JIT_VALUE_FACTS").as_deref() == Ok("0")
        {
            assert!(function.optimizing_code.get().is_none());
            continue;
        }
        if std::env::var("TRUST_JS_OPT_JIT_COMPACT_STORES").as_deref() == Ok("0") {
            assert!(function.optimizing_code.get().is_none());
            let primary = function.code.get().and_then(Option::as_ref).unwrap();
            let samples = primary.optimizing_samples.get().unwrap();
            let inputs: Vec<_> = samples.classes[..primary.jit_frame().0]
                .iter()
                .enumerate()
                .map(|(slot, bits)| (slot as u16, bits.get()))
                .collect();
            let properties: Vec<_> = samples
                .properties
                .iter()
                .map(|(pc, bits)| (*pc, bits.get()))
                .collect();
            let candidate =
                bytecode::compile_for_optimizer(&function, primary, &inputs, &properties).unwrap();
            assert!(
                candidate.optimizing_structural.is_empty(),
                "disabled structure keeps old numeric-only snapshot"
            );
            let values = crate::value::jit_layout(&engine.interp.object_proto);
            let layout = crate::interpreter::interp_layout(&mut engine.interp);
            assert!(
                matches!(
                    compile_with_deopt(&candidate, &values, &layout, None),
                    Err(CompileError::LoweringBudget)
                ),
                "original polymorphic store emission keeps the unchanged candidate budget"
            );
            continue;
        }
        let candidate = function
            .optimizing_code
            .get()
            .and_then(Option::as_ref)
            .expect("numeric hot publication preserved");
        let cfg = Cfg::build(candidate).unwrap();
        let stack = stack::StackPlan::build(candidate, &cfg).unwrap();
        assert!(specialization::Plan::build(
            candidate,
            &cfg,
            &stack,
            &candidate.optimizing_inputs,
            &candidate.optimizing_properties
        )
        .is_some());
        assert_eq!(
            !candidate.optimizing_structural.is_empty(),
            enabled && generic_admission::enabled()
        );
        let code = candidate.jit.get().flatten().unwrap();
        code.residency.referenced.set(0);
        assert_eq!(eval(&mut engine, "subject(receiver,3)"), "value7");
        assert!(code.residency.referenced.get() > 0);
        assert_eq!(candidate.optimizing_misses.get(), 0);
        engine.interp.gc_collect();
    }
    let _disabled = Enabled::new(false);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, "function subject(o){'use strict';o.x=o.x;return o.x;}var receiver={x:'value'};for(var k=0;k<32;k++)subject(receiver);");
    assert!(
        function(&mut engine).optimizing_code.get().is_none(),
        "generic-only remains established policy when disabled"
    );
}

#[test]
fn mixed_admission_distance_credits_only_enabled_numeric_lowering() {
    const CHILD: &str = "TRUST_JS_TEST_STRUCTURAL_DISTANCE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        for value_facts in ["0", "1"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "jit::optimizing::generic_admission_tests::mixed_admission_distance_credits_only_enabled_numeric_lowering", "--nocapture"])
                .env(CHILD, "1").env("TRUST_JS_OPT_JIT", "hot")
                .env_remove("TRUST_JS_OPT_JIT_HOT_AT").env("TRUST_JS_INLINE_AT", "0")
                .env("TRUST_JS_OPT_JIT_VALUE_FACTS", value_facts).output().unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        }
        return;
    }
    let _enabled = Enabled::new(true);
    if !generic_admission::enabled() {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, "function subject(o,x){'use strict';o.x=o.x;return o.x+(x*2+1);}var receiver={x:'value'};for(var k=0;k<300;k++)subject(receiver,k);");
    let function = function(&mut engine);
    assert!(
        function.optimizing_code.get().is_none(),
        "past entries have not paid even the unchanged floor"
    );
    let chunk = function.code.get().and_then(Option::as_ref).unwrap();
    let samples = chunk.optimizing_samples.get().unwrap();
    assert_eq!(samples.samples.get(), 4);
    let cfg = Cfg::build(chunk).unwrap();
    let stack = stack::StackPlan::build(chunk, &cfg).unwrap();
    let selected = generic_admission::selected(&samples.structural);
    let structural = generic_admission::Plan::build(chunk, &cfg, &stack, &selected).unwrap();
    let inputs: Vec<_> = samples
        .classes
        .iter()
        .enumerate()
        .map(|(slot, bits)| (slot as u16, bits.get()))
        .collect();
    let properties: Vec<_> = samples
        .properties
        .iter()
        .map(|(pc, bits)| (*pc, bits.get()))
        .collect();
    let numeric = specialization::Plan::build(chunk, &cfg, &stack, &inputs, &properties).unwrap();
    let enabled = std::env::var("TRUST_JS_OPT_JIT_VALUE_FACTS").as_deref() != Ok("0");
    let target = generic_admission::admission_entries(
        chunk,
        if enabled { numeric.saved_tests } else { 0 },
        structural.minimum_tests,
    )
    .unwrap()
    .max(4096);
    assert_eq!(
        samples.interval.get() as u64,
        target - samples.entries.get(),
        "actual rearm matches emitted mechanisms"
    );
    if !enabled {
        assert!(
            target
                > generic_admission::admission_entries(
                    chunk,
                    numeric.saved_tests,
                    structural.minimum_tests
                )
                .unwrap()
                .max(4096),
            "disabled numeric checks receive no credit"
        );
    }
}
