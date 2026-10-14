//! Composed recovery, execution observations, structural admission and worker publication.
//! ECMA-262 e28783d5 OrdinaryGet/OrdinarySetWithOwnDescriptor, Completion Records,
//! Execution Contexts and WeakRef liveness: completed prefix effects and owning RHS
//! words survive the exact before-effect transfer, and compiler jobs own no JS payloads.

use super::*;
use crate::value::Callable;
use crate::{bytecode::Tier, Completion, Engine};
use std::rc::Rc;
use std::time::{Duration, Instant};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("integration fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn function(engine: &mut Engine) -> Rc<crate::ast::Function> {
    let env = engine.interp.global_env.clone();
    let callable = engine.interp.get_var("subject", &env).ok().unwrap();
    let object = callable.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("ordinary User")
    };
    user.func.clone()
}

fn weak_payload(engine: &mut Engine) -> std::rc::Weak<std::cell::RefCell<crate::value::Object>> {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("payload", &env).ok().unwrap();
    Rc::downgrade(value.as_obj().unwrap())
}

/// Process isolation preserves the production OnceLocks and engine-thread TLS. The
/// environment is specified here instead of inheriting a caller's diagnostic ablations.
fn matrix(test: &str, hot: bool, variants: &[(bool, bool)]) -> bool {
    const CASE: &str = "TRUST_JS_TEST_COMPOSED_INTEGRATION_CHILD";
    if std::env::var(CASE).as_deref() == Ok(test) {
        return false;
    }
    for &(shared, owners) in variants {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("TRUST_JS") {
                command.env_remove(key);
            }
        }
        command
            .args([
                "--exact",
                &format!("jit::optimizing::integration_tests::{test}"),
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CASE, test)
            .env("TRUST_JS_OPT_JIT", if hot { "hot" } else { "0" })
            .env("TRUST_JS_OPT_JIT_BACKGROUND", if hot { "1" } else { "0" })
            .env(
                "TRUST_JS_OPT_JIT_DIAGNOSTICS",
                if hot { "compile" } else { "execution" },
            )
            .env("TRUST_JS_OPT_JIT_GENERIC_OWN", "1")
            .env(
                "TRUST_JS_OPT_JIT_SHARED_RECOVERY",
                if shared { "1" } else { "0" },
            )
            .env(
                "TRUST_JS_OPT_JIT_SHARED_OWNERS",
                if owners { "1" } else { "0" },
            )
            .env("TRUST_JS_OPT_JIT_HEAP_OPS", "1")
            .env("TRUST_JS_OPT_JIT_COMPACT_STORES", "1")
            .env("TRUST_JS_OPT_JIT_VALUE_FACTS", "1")
            .env("TRUST_JS_OPT_JIT_LOCAL_EFFECTS", "1")
            .env("TRUST_JS_OPT_JIT_SPARSE_RELOADS", "1")
            .env("TRUST_JS_OPT_JIT_RANGES", "1")
            .env("TRUST_JS_OPT_JIT_INT_SSA", "1")
            .env("TRUST_JS_OPT_JIT_NAMES", "1")
            .env("TRUST_JS_OPT_JIT_OSR", "0")
            .env("TRUST_JS_INLINE_AT", "0")
            .env("TRUST_JS_PERF_METRICS", "1");
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "shared={shared}, owners={owners}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "exact child ran"
        );
    }
    true
}

fn counters(record: &diagnostics::Code) -> [u64; 7] {
    [
        0,
        diagnostics::NORMAL_EXIT,
        diagnostics::ERROR_EXIT,
        diagnostics::SPECIALIZATION_BAILOUT,
        diagnostics::VM_RECOVERY,
        diagnostics::VM_RECOVERY_NORMAL,
        diagnostics::RETIREMENT_REQUEST,
    ]
    .map(|index| record.counters[index].get())
}

#[test]
fn shared_generic_overwrite_recovery_counts_once_and_keeps_rhs_through_setter_gc() {
    if matrix(
        "shared_generic_overwrite_recovery_counts_once_and_keeps_rhs_through_setter_gc",
        false,
        &[(false, false), (false, true), (true, false), (true, true)],
    ) {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        var trace=[],prefixes=0,sets=0,gets=0,finishes=0,throwing=false;
        var payload={tag:'alive'},source={value:payload};
        var target={x:0,a:1,b:2,c:3,d:4,e:5,saved:payload};
        function finishCopy(o){finishes++;trace.push('F');
            return [o.x.tag,o.a.tag,o.b.tag,o.c.tag,o.d.tag,o.e.tag].join('|');}
        function subject(s,t){'use strict';prefixes++;trace.push('P');
            t.x=s.value;t.a=t.x;t.b=t.a;t.c=t.b;t.d=t.c;t.e=t.d;
            return finishCopy(t);}
        subject(source,target);subject(source,target);
        trace=[];prefixes=sets=gets=finishes=0;
    "#,
    );
    let payload = weak_payload(&mut engine);
    let function = function(&mut engine);
    let primary = function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    // Forced generic coverage intentionally has an observable prefix before its
    // credited accesses. It is not evidence that this body passes hot admission.
    let mut candidate = bytecode::compile_for_optimizer(&function, &primary, &[], &[]).unwrap();
    Rc::get_mut(&mut candidate).unwrap().optimizing_candidate = false;
    let write_pc = candidate
        .jit_ops()
        .iter()
        .position(|op| match *op {
            Op::SetPropLocalDrop(_, name, _) => candidate.jit_name(name) == "x",
            _ => false,
        })
        .expect("selected target overwrite");
    generic_own::TEST_SITES.with(|sites| sites.set([0; 2]));
    recovery::TEST_COUNTS.with(|counts| counts.set([0; 4]));
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = Rc::new(compile_with_deopt(&candidate, &values, &layout, None).unwrap());
    let selected = generic_own::TEST_SITES.with(std::cell::Cell::get);
    assert!(
        selected[0] >= 2 && selected[1] >= 2,
        "real selected read and overwrite guards"
    );
    let recovery_shapes = recovery::TEST_COUNTS.with(std::cell::Cell::get);
    if std::env::var("TRUST_JS_OPT_JIT_SHARED_RECOVERY").as_deref() == Ok("1") {
        assert!(
            recovery_shapes[1] < recovery_shapes[0],
            "actual terminal sites share helpers"
        );
    } else {
        assert_eq!(recovery_shapes[0], recovery_shapes[1]);
    }
    assert!(code.pc_offsets.is_empty(), "actual optimizing machine body");
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
    engine.set_tier(Tier::Jit);
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert!(record.execution_counters && !record.runtime_counters);
    assert!(!record.counts(diagnostics::OWNER_LEAF));
    assert!(!record.counts(diagnostics::OWNER_LEAF_GUARDS));
    owner_leaf::TEST_CALLS.with(|calls| calls.set([0; 2]));
    assert_eq!(
        eval(&mut engine, "subject(source,target)+'|'+trace.join(',')"),
        "alive|alive|alive|alive|alive|alive|P,F"
    );
    assert_eq!(counters(record), [1, 1, 0, 0, 0, 0, 0]);
    let leaf_calls = owner_leaf::TEST_CALLS.with(std::cell::Cell::get);
    if owner_leaf::requested() {
        assert!(
            leaf_calls[0] > 0 && record.emitted_owner_leaf_calls.get() > 0,
            "enabled heap retain leaf actually executed"
        );
    } else {
        assert_eq!(leaf_calls, [0; 2]);
    }
    eval(
        &mut engine,
        r#"
        Object.defineProperty(target,'x',{configurable:true,
            get(){gets++;trace.push('G');return this.saved;},
            set(v){sets++;trace.push('S');$262.gc();
                if(v!==payload || v.tag!=='alive')throw 'lost RHS owner';
                if(throwing)throw 'boom';this.saved=v;}});
    "#,
    );
    for attempt in 0..8 {
        let throws = attempt % 2 == 1;
        eval(&mut engine, &format!("throwing={throws};trace=[];"));
        bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
        let observed = eval(
            &mut engine,
            "var outcome;try{outcome='return:'+subject(source,target);}
            catch(e){outcome='throw:'+e;}outcome+'|'+trace.join(',')",
        );
        assert_eq!(
            observed,
            if throws {
                "throw:boom|P,S"
            } else {
                "return:alive|alive|alive|alive|alive|alive|P,S,G,F,G"
            }
        );
        bytecode::native_deopt::ENTRIES.with(|entries| {
            let entries = entries.borrow();
            assert_eq!(
                entries.len(),
                1,
                "one terminal transfer per actual invocation"
            );
            assert_eq!(
                entries[0].0, write_pc,
                "already-retained source RHS precedes exact overwrite PC"
            );
        });
        let completed = attempt + 1;
        let normal = (completed + 1) / 2;
        assert_eq!(candidate.optimizing_misses.get(), completed as u8);
        assert_eq!(
            counters(record),
            [
                1 + completed as u64,
                1 + normal as u64,
                (completed - normal) as u64,
                completed as u64,
                completed as u64,
                normal as u64,
                u64::from(completed == 8)
            ]
        );
        assert_eq!(code.residency.active.get(), 0);
        assert!(engine.interp.fn_frames.is_empty());
    }
    assert_eq!(candidate.optimizing_misses.get(), SPECIALIZATION_MISS_LIMIT);
    assert!(
        Rc::ptr_eq(
            function
                .execution_code_for(Rc::as_ptr(&engine.interp.global_env) as usize)
                .and_then(Option::as_ref)
                .unwrap(),
            &primary
        ),
        "eight misses restore baseline selection"
    );
    let retired = counters(record);
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    assert_eq!(
        eval(
            &mut engine,
            "throwing=false;trace=[];subject(source,target)"
        ),
        "alive|alive|alive|alive|alive|alive"
    );
    assert_eq!(
        counters(record),
        retired,
        "retired native counters receive no baseline work"
    );
    bytecode::native_deopt::ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    assert_eq!(
        eval(&mut engine, "[prefixes,sets,gets,finishes].join(':')"),
        "10:9:10:6"
    );
    eval(&mut engine, "payload=source=target=null;trace=[];");
    engine.interp.gc_collect();
    assert_eq!(
        payload.strong_count(),
        0,
        "retired code and diagnostics keep no RHS payload owner"
    );
    assert_eq!(record.counters[diagnostics::OWNER_LEAF].get(), 0);
    assert_eq!(record.counters[diagnostics::OWNER_LEAF_GUARDS].get(), 0);
    eprintln!("composed generic recovery shapes={recovery_shapes:?}, counters={retired:?}, leaf_calls={leaf_calls:?}");
}

#[test]
fn background_structural_hot_publication_keeps_exact_budget_guards_and_weak_heap_ownership() {
    if matrix(
        "background_structural_hot_publication_keeps_exact_budget_guards_and_weak_heap_ownership",
        true,
        &[(true, false), (true, true)],
    ) {
        return;
    }
    assert!(background::enabled() && generic_admission::enabled());
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        var payload={tag:'alive'},finishes=0;
        var receiver={source:payload,x:0,a:1,b:2,c:3,d:4,e:5};
        function finishCopy(o){finishes++;return [o.x.tag,o.a.tag,o.b.tag,o.c.tag,o.d.tag,o.e.tag].join('|');}
        function subject(o){'use strict';o.x=o.source;o.a=o.x;o.b=o.a;o.c=o.b;o.d=o.c;o.e=o.d;return finishCopy(o);}
        subject(receiver);subject(receiver);
    "#,
    );
    let payload = weak_payload(&mut engine);
    let function = function(&mut engine);
    let baseline = function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    let observations = generic_admission::observations(&baseline);
    let selected = generic_admission::selected(&observations);
    let cfg = Cfg::build(&baseline).unwrap();
    let stack = stack::StackPlan::build(&baseline, &cfg).unwrap();
    let benefit = generic_admission::Plan::build(&baseline, &cfg, &stack, &selected)
        .expect("positive mandatory structural work before finish callback");
    let distance = generic_admission::admission_entries(&baseline, 0, benefit.minimum_tests)
        .unwrap()
        .max(4096);
    assert!(distance <= 200_000, "bounded real-policy fixture warmup");
    assert!(std::env::var_os("TRUST_JS_OPT_JIT_HOT_AT").is_none());
    engine.set_tier(Tier::Jit);
    eval(
        &mut engine,
        &format!(
            "for(var k=0;k<{};k++)subject(receiver);'warm'",
            distance + 512
        ),
    );
    // Work submission comes from real template entry checkpoints. Only the owner
    // thread polls publication; no JS/Env/Rc is sent to the worker by this test.
    let deadline = Instant::now() + Duration::from_secs(15);
    while function.optimizing_code.get().is_none()
        && engine
            .interp
            .optimizing_background
            .as_ref()
            .is_some_and(|state| state.contains(&baseline))
    {
        engine.interp.optimizing_background.as_mut().unwrap().poll();
        if function.optimizing_code.get().is_some() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "real worker must finish or explicitly fail, never hang"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let limit =
        (baseline.jit_ops().len() * 40 + names::instruction_allowance(&baseline)).min(16_384);
    if function.optimizing_code.get().is_none() {
        let samples = baseline
            .optimizing_samples
            .get()
            .expect("actual hot samples collected");
        let selected = generic_admission::selected(&samples.structural);
        assert!(!selected.is_empty());
        let inputs: Vec<_> = samples
            .classes
            .iter()
            .enumerate()
            .take(baseline.jit_frame().0)
            .map(|(slot, bits)| (slot as u16, bits.get()))
            .collect();
        let properties: Vec<_> = samples
            .properties
            .iter()
            .map(|(pc, bits)| (*pc, bits.get()))
            .collect();
        let candidate = bytecode::compile_for_optimizer_with_structural(
            &function,
            &baseline,
            &inputs,
            &properties,
            &selected,
        )
        .unwrap();
        let values = crate::value::jit_layout(&engine.interp.object_proto);
        let layout = crate::interpreter::interp_layout(&mut engine.interp);
        assert!(
            matches!(
                prepare_entry_with_limits(&candidate, &values, &layout, None, limit, 65_536, None),
                Err(CompileError::LoweringBudget)
            ),
            "an absent version must prove the exact unchanged IR budget decline"
        );
        assert_eq!(baseline.optimizing_remaining.get(), 0);
        assert!(engine
            .interp
            .optimizing_background
            .as_ref()
            .is_none_or(|state| !state.contains(&baseline)));
        assert_eq!(
            eval(&mut engine, "subject(receiver)"),
            "alive|alive|alive|alive|alive|alive"
        );
        eprintln!("composed background structural decline: ops={} exact_cap={limit} distance={distance} minimum_tests={}",baseline.jit_ops().len(),benefit.minimum_tests);
        eval(&mut engine, "payload=receiver=null;");
        engine.interp.gc_collect();
        drop(engine);
        assert_eq!(
            payload.strong_count(),
            0,
            "declined worker preparation adds no JS owner"
        );
        return;
    }
    let candidate = function
        .optimizing_code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    assert!(!candidate.optimizing_structural.is_empty());
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
        "publication is generic structural, with no numeric admission credit"
    );
    let code = candidate.jit.get().flatten().unwrap();
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert!(
        !record.runtime_counters && !record.execution_counters,
        "compile mode preserves ordinary native graph and budget"
    );
    assert!(record.ir_instructions.get() > 0 && record.ir_instructions.get() <= limit);
    assert!(
        code.pc_offsets.is_empty(),
        "real completed worker native body"
    );
    code.residency.referenced.set(0);
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    assert_eq!(
        eval(&mut engine, "subject(receiver)"),
        "alive|alive|alive|alive|alive|alive"
    );
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(candidate.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    let before = eval(&mut engine, "finishes").parse::<u64>().unwrap();
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    assert_eq!(
        eval(
            &mut engine,
            "var getterCalls=0;Object.defineProperty(receiver,'source',{
        configurable:true,get(){getterCalls++;$262.gc();return payload;}});
        subject(receiver)+'|'+getterCalls"
        ),
        "alive|alive|alive|alive|alive|alive|1"
    );
    assert_eq!(
        eval(&mut engine, "finishes").parse::<u64>().unwrap(),
        before + 1
    );
    assert_eq!(candidate.optimizing_misses.get(), 1);
    bytecode::native_deopt::ENTRIES.with(|entries| {
        let entries = entries.borrow();
        assert_eq!(entries.len(), 1);
        assert!(
            candidate
                .optimizing_structural
                .iter()
                .any(|(pc, _)| *pc as usize == entries[0].0),
            "live selected descriptor guard actually recovers"
        );
    });
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
    eprintln!("composed background structural native: ops={} IR={} exact_cap={limit} distance={distance} minimum_tests={} selected={}",
        candidate.jit_ops().len(),record.ir_instructions.get(),benefit.minimum_tests,candidate.optimizing_structural.len());
    eval(&mut engine, "payload=receiver=null;");
    engine.interp.gc_collect();
    assert_eq!(
        payload.strong_count(),
        0,
        "published worker/code/diagnostics retain no JS payload"
    );
    drop(engine);
    assert_eq!(
        payload.strong_count(),
        0,
        "engine shutdown preserves no worker payload owner"
    );
    assert!(
        record.counters.iter().all(|counter| counter.get() == 0),
        "compile-only native graph has no execution probes"
    );
}
