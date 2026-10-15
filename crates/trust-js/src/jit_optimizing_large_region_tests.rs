//! Large-source admission/recovery contracts. Synthetic fixtures prove the
//! mechanism; actual pinned React publication/repayment is separate evidence.
use super::*;
use crate::{bytecode::Tier, value::Callable, Completion, Engine};
use std::rc::Rc;

fn child(name: &str, enabled: bool) -> bool {
    const MARKER: &str = "TRUST_JS_LARGE_REGION_TEST_CHILD";
    if std::env::var(MARKER).as_deref() == Ok(name) {
        return false;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    for (key, _) in std::env::vars().filter(|(key, _)| key.starts_with("TRUST_JS_")) {
        command.env_remove(key);
    }
    let output = command
        .args([
            "--exact",
            &format!("jit::optimizing::large_region_tests::{name}"),
            "--nocapture",
            "--test-threads=1",
        ])
        .env(MARKER, name)
        .env("TRUST_JS_OPT_JIT", "hot")
        .env("TRUST_JS_OPT_JIT_COVERAGE", "1")
        .env("TRUST_JS_OPT_JIT_OBSERVED_REGIONS", "1")
        .env(
            "TRUST_JS_OPT_JIT_LARGE_REGIONS",
            if enabled { "1" } else { "0" },
        )
        .env("TRUST_JS_OPT_JIT_GENERIC_OWN", "1")
        .env("TRUST_JS_OPT_JIT_SHARED_OWNERS", "1")
        .env("TRUST_JS_OPT_JIT_SHARED_READ_GUARDS", "1")
        .env("TRUST_JS_OPT_JIT_DIAGNOSTICS", "execution")
        .env("TRUST_JS_OPT_JIT_BACKGROUND", "0")
        .env("TRUST_JS_OPT_JIT_OSR", "0")
        .env("TRUST_JS_INLINE_AT", "0")
        .output()
        .unwrap();
    eprintln!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
    true
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("large fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}
fn lookup_function(engine: &mut Engine, name: &str) -> Rc<crate::ast::Function> {
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var(name, &env).ok().unwrap();
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("User")
    };
    user.func.clone()
}
fn primary(function: &crate::ast::Function) -> Rc<Chunk> {
    function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone()
}
fn selected(chunk: &Chunk) -> Vec<(u32, bytecode::IcState)> {
    generic_admission::selected(&generic_admission::observations(chunk))
}
fn prepare(engine: &mut Engine, chunk: &Chunk) -> Result<JitCode, CompileError> {
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    diagnostics::with_test_mode("execution", || {
        compile_with_deopt(chunk, &values, &layout, None)
    })
}
fn source(cold: &str) -> String {
    format!("var receiver={{source:'live',x:0,a:0,b:0,c:0,d:0,e:0}},effects=[];function subject(o,cold){{o.x=o.source;o.a=o.x;o.b=o.a;o.c=o.b;o.d=o.c;o.e=o.d;if(cold){{{cold}}}return o.e;}}")
}

#[test]
fn independent_large_heap_proofs_share_aggregate_limits_and_never_cross_blocks() {
    use heap_proofs::{Block, Budget, Event, Graph, HeapEffect, Metadata, Proof, Receiver};
    let guard = Event::Guard(Proof {
        receiver: Receiver(0),
        metadata: Metadata::NamedShape(3),
    });
    let graph = Graph {
        blocks: (0..300)
            .map(|_| Block {
                events: vec![guard, guard, Event::Effect(HeapEffect::Observer), guard],
                successors: vec![],
            })
            .collect(),
        roots: (0..300).collect(),
    };
    let budget = Budget {
        blocks: large_regions::MAX_BLOCKS,
        ..Budget::default()
    };
    let proof = heap_proofs::analyze_independent(&graph, budget).unwrap();
    for block in 0..300 {
        assert!(!proof.redundant(block, 0));
        assert!(proof.redundant(block, 1));
        assert!(!proof.redundant(block, 3));
    }
    assert!(heap_proofs::analyze_independent(
        &graph,
        Budget {
            events: 1199,
            ..budget
        }
    )
    .is_none());
    assert!(heap_proofs::analyze_independent(&graph, Budget { work: 0, ..budget }).is_none());
    assert!(
        heap_proofs::analyze_independent(
            &graph,
            Budget {
                work: 100,
                ..budget
            }
        )
        .is_none(),
        "nonzero work is shared across blocks, never reset per block"
    );
    assert!(heap_proofs::analyze_independent(
        &graph,
        Budget {
            blocks: 299,
            ..budget
        }
    )
    .is_none());
}

#[test]
fn missing_unknown_no_cut_and_oversized_sources_cannot_publish_large_native_code() {
    if child(
        "missing_unknown_no_cut_and_oversized_sources_cannot_publish_large_native_code",
        true,
    ) {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    let cold = (0..160).map(|_| "o.x=o.source;").collect::<String>();
    eval(
        &mut engine,
        &format!("{}subject(receiver,false);", source(&cold)),
    );
    let function = lookup_function(&mut engine, "subject");
    let hot = primary(&function);
    assert!((257..=large_regions::MAX_OPS).contains(&hot.jit_ops().len()));
    assert!(
        hot_counter(&hot).is_some(),
        "bounded Source can seed four observations"
    );
    let sites = selected(&hot);
    let mut missing =
        bytecode::compile_for_optimizer_with_structural(&function, &hot, &[], &[], &sites).unwrap();
    let fresh = crate::execution_coverage::Handle::new(&missing).freeze();
    Rc::get_mut(&mut missing).unwrap().execution_coverage = fresh;
    assert!(matches!(
        prepare(&mut engine, &missing),
        Err(CompileError::Unsupported)
    ));
    let mut unavailable =
        bytecode::compile_for_optimizer_with_structural(&function, &hot, &[], &[], &sites).unwrap();
    Rc::get_mut(&mut unavailable).unwrap().execution_coverage =
        crate::execution_coverage::Handle::Unavailable(crate::execution_coverage::Reason::Quota);
    assert!(
        matches!(
            prepare(&mut engine, &unavailable),
            Err(CompileError::Unsupported)
        ),
        "quota-unavailable provenance cannot become a no-cut full-body fallback"
    );
    hot.execution_coverage
        .invalidate(crate::execution_coverage::Reason::AstBody);
    let unknown =
        bytecode::compile_for_optimizer_with_structural(&function, &hot, &[], &[], &sites).unwrap();
    assert!(matches!(
        prepare(&mut engine, &unknown),
        Err(CompileError::Unsupported)
    ));
    eval(
        &mut engine,
        &format!("function full(o){{{cold}return o.source;}}full(receiver);"),
    );
    let full_function = lookup_function(&mut engine, "full");
    let full = primary(&full_function);
    let no_cut = bytecode::compile_for_optimizer_with_structural(
        &full_function,
        &full,
        &[],
        &[],
        &selected(&full),
    )
    .unwrap();
    assert!(matches!(
        prepare(&mut engine, &no_cut),
        Err(CompileError::Unsupported)
    ));
    let oversized = (0..2100).map(|_| "o.x=o.source;").collect::<String>();
    eval(&mut engine,&format!("function oversized(o,cold){{if(cold){{{oversized}}}return o.source;}}oversized(receiver,false);"));
    let oversized = primary(&lookup_function(&mut engine, "oversized"));
    assert!(oversized.jit_ops().len() > large_regions::MAX_OPS);
    assert_eq!(hot_threshold(oversized.jit_ops().len()), 0);
    assert!(hot_counter(&oversized).is_none());
    assert!(function.optimizing_code.get().is_none());
}

#[test]
fn cold_disposal_remains_a_full_body_large_policy_exclusion() {
    if child(
        "cold_disposal_remains_a_full_body_large_policy_exclusion",
        true,
    ) {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    let cold = (0..160).map(|_| "o.x=o.source;").collect::<String>();
    eval(
        &mut engine,
        &format!(
            "{}subject(receiver,false);",
            source(&format!("using resource={{[Symbol.dispose](){{}}}};{cold}"))
        ),
    );
    let function = lookup_function(&mut engine, "subject");
    let hot = primary(&function);
    assert!(hot.jit_ops().len() > 256);
    assert!(large_regions::unsupported(&hot));
    assert!(hot_counter(&hot).is_none());
    let candidate =
        bytecode::compile_for_optimizer_with_structural(&function, &hot, &[], &[], &selected(&hot))
            .unwrap();
    assert!(matches!(
        prepare(&mut engine, &candidate),
        Err(CompileError::Unsupported)
    ));
    assert!(function.optimizing_code.get().is_none());
}

#[test]
fn original_frame_and_labels_and_every_cut_pay_work_without_vm_tail_credit() {
    if child(
        "original_frame_and_labels_and_every_cut_pay_work_without_vm_tail_credit",
        true,
    ) {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    let cold = (0..160).map(|_| "o.x=o.source;").collect::<String>();
    eval(
        &mut engine,
        &format!("{}subject(receiver,false);", source(&cold)),
    );
    let function = lookup_function(&mut engine, "subject");
    let hot = primary(&function);
    let cfg = Cfg::build(&hot).unwrap();
    let stack = stack::StackPlan::build(&hot, &cfg).unwrap();
    let frozen = hot.execution_coverage.freeze();
    let regions = observed_regions::Plan::build(&hot, &cfg, &stack, &frozen).unwrap();
    let structural = generic_admission::Plan::with_objects_and_regions(
        &hot,
        &cfg,
        &stack,
        &selected(&hot),
        None,
        Some(&regions),
    )
    .unwrap();
    assert_eq!(
        structural.minimum_tests, 0,
        "large opportunity is never labelled a guaranteed minimum"
    );
    let frame = frame::FramePlan::build(&hot, &cfg, true).unwrap();
    let work = large_regions::work(&hot, &cfg, &stack, &frame, &regions).unwrap();
    assert_eq!(
        work.original_labels,
        (hot.jit_ops().len() as u64 + 6) * 2
            + native_pcs::analysis_work(hot.jit_ops().len()).unwrap(),
        "retain full original label work plus selected-graph certificate analysis"
    );
    assert!(work.original_frame > 0);
    assert!(work.cold_recovery >= regions.cut_blocks as u64 * 16);
    let opportunity =
        large_regions::opportunity_tests(&hot, &cfg, &regions, None, Some(&structural)).unwrap();
    assert!(work.admission_entries(opportunity).unwrap() >= 4096);
    assert!(work.admission_entries(0).is_none());
    for pc in 0..hot.jit_ops().len() {
        if regions.emission(pc) != observed_regions::Emission::Native {
            assert_eq!(structural.per_pc[pc], None);
        }
    }
    let numeric = specialization::Plan::build(
        &hot,
        &cfg,
        &stack,
        &[(1, crate::feedback::ValueClass::Boolean.bit())],
        &[],
    )
    .unwrap();
    let mut raw = vec![0u32; hot.jit_ops().len()];
    for item in &numeric.opportunities {
        raw[item.pc] += item.removed_tests;
    }
    let with_numeric =
        large_regions::opportunity_tests(&hot, &cfg, &regions, Some(&numeric), Some(&structural))
            .unwrap();
    let extra = raw
        .iter()
        .enumerate()
        .filter(|&(pc, _)| regions.emission(pc) == observed_regions::Emission::Native)
        .map(|(_, credit)| *credit)
        .sum::<u32>();
    assert_eq!(
        with_numeric,
        opportunity + extra - (numeric.inputs.len() + numeric.properties.len()) as u32,
        "all emitted numeric guards are charged once"
    );
}

#[test]
fn normal_hot_large_publication_runs_native_and_cold_same_invocation_without_overrides() {
    if child(
        "normal_hot_large_publication_runs_native_and_cold_same_invocation_without_overrides",
        true,
    ) {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let cold = (0..160).map(|_| "o.x=o.source;").collect::<String>();
    eval(
        &mut engine,
        &source(&format!(
            "{cold}effects.push('C');$262.gc();return 'cold:'+o.e;"
        )),
    );
    assert_eq!(
        eval(
            &mut engine,
            "for(var i=0;i<30000;i++)subject(receiver,false);subject(receiver,false)"
        ),
        "live"
    );
    let function = lookup_function(&mut engine, "subject");
    let hot = primary(&function);
    assert!(hot.jit_ops().len() > 256);
    let next = function
        .optimizing_code
        .get()
        .and_then(Option::as_ref)
        .expect("normal large-policy publication without HOT_AT");
    let code = next.jit.get().flatten().unwrap();
    assert!(code.pc_offsets.is_empty());
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert!(record.ir_instructions.get() <= 16384);
    let json = record.json();
    assert!(json.contains("observed_native_opportunity_heuristic"));
    assert!(json.contains("\"frequency_or_minimum_claim\":false"));
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    code.residency.referenced.set(0);
    let entries_before = record.counters[0].get();
    assert_eq!(eval(&mut engine, "subject(receiver,false)"), "live");
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(
        record.counters[0].get(),
        entries_before + 1,
        "normal call physically enters optimizing-native code"
    );
    assert_eq!(
        eval(&mut engine, "subject(receiver,true)+'|'+effects.join(',')"),
        "cold:live|C"
    );
    bytecode::native_deopt::ENTRIES.with(|entries| assert_eq!(entries.borrow().len(), 1));
    assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 1);
    assert_eq!(
        record.counters[0].get(),
        entries_before + 2,
        "cold call physically enters optimizing-native code once"
    );
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 1);
    assert_eq!(next.optimizing_misses.get(), 0);
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
}

#[test]
fn positive_large_certificate_still_refuses_actual_ir_above_default_hard_cap() {
    if child(
        "positive_large_certificate_still_refuses_actual_ir_above_default_hard_cap",
        true,
    ) {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    let native = (0..400).map(|_| "o.x=o.source;").collect::<String>();
    eval(&mut engine,&format!("var receiver={{source:'live',x:0}};function subject(o,cold){{{native}if(cold){{for(var i=0;i<1;i++){{}}}}return o.source;}}subject(receiver,false);"));
    let function = lookup_function(&mut engine, "subject");
    let hot = primary(&function);
    let cfg = Cfg::build(&hot).unwrap();
    let stack = stack::StackPlan::build(&hot, &cfg).unwrap();
    let frozen = hot.execution_coverage.freeze();
    let regions = observed_regions::Plan::build(&hot, &cfg, &stack, &frozen).unwrap();
    let structural = generic_admission::Plan::with_objects_and_regions(
        &hot,
        &cfg,
        &stack,
        &selected(&hot),
        None,
        Some(&regions),
    )
    .unwrap();
    assert!(
        large_regions::opportunity_tests(&hot, &cfg, &regions, None, Some(&structural)).is_some()
    );
    let candidate = bytecode::compile_for_optimizer_with_objects_and_coverage(
        &function,
        &hot,
        &[],
        &[],
        &structural.selected,
        &[],
        &frozen,
    )
    .unwrap();
    assert!(
        matches!(
            prepare(&mut engine, &candidate),
            Err(CompileError::LoweringBudget)
        ),
        "positive workload/proof certificates never waive actual default16K expanded-IR cap"
    );
    assert!(candidate.jit.get().is_none());
    assert!(function.optimizing_code.get().is_none());
    assert_eq!(
        eval(&mut engine, "subject(receiver,true)+'|'+receiver.x"),
        "live|live"
    );
    assert!(engine.interp.fn_frames.is_empty());
}

#[test]
fn large_work_arithmetic_overflow_is_declined_without_wrapping_or_dividing_zero() {
    let work = large_regions::Work {
        native_expansion: u64::MAX,
        original_labels: 1,
        original_frame: 0,
        cold_recovery: 0,
        call_imports: 0,
        literal_analysis: 0,
        literal_dispatch_lowering: 0,
        borrowed_operands: 0,
    };
    assert!(work.total().is_none());
    assert!(work.admission_entries(1).is_none());
    assert!(work.admission_entries(0).is_none());
    let work = large_regions::Work {
        native_expansion: u64::MAX,
        original_labels: 0,
        original_frame: 0,
        cold_recovery: 0,
        call_imports: 0,
        literal_analysis: 0,
        literal_dispatch_lowering: 0,
        borrowed_operands: 0,
    };
    assert_eq!(work.total(), Some(u64::MAX));
    assert!(work.admission_entries(1).is_none());
}

#[test]
fn large_policy_disabled_retains_the_existing_whole_body_size_gate() {
    if child(
        "large_policy_disabled_retains_the_existing_whole_body_size_gate",
        false,
    ) {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let cold = (0..160).map(|_| "o.x=o.source;").collect::<String>();
    eval(
        &mut engine,
        &format!(
            "{}for(var i=0;i<1000;i++)subject(receiver,false);",
            source(&cold)
        ),
    );
    let function = lookup_function(&mut engine, "subject");
    let hot = primary(&function);
    assert!(hot.jit_ops().len() > 256);
    assert!(hot_counter(&hot).is_none());
    assert_eq!(hot_threshold(hot.jit_ops().len()), 0);
    assert!(function.optimizing_code.get().is_none());
}
