//! Physical label reduction keeps native execution, cold recovery and full-PC stress.

use super::super::*;
use super::{Plan, TestControl, TEST_COUNTS};
use crate::bytecode::{native_deopt::ENTRIES, Tier};
use crate::value::Callable;
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn candidate(engine: &mut Engine) -> (Rc<crate::ast::Function>, Rc<Chunk>) {
    let env = engine.interp.global_env.clone();
    let callable = engine
        .interp
        .get_var("subject", &env)
        .unwrap_or_else(|_| panic!("bound subject"));
    let object = callable.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("ordinary function")
    };
    let function = user.func.clone();
    let primary = function.execution_code().and_then(Option::as_ref).unwrap();
    let mut candidate = bytecode::compile_for_optimizer(&function, primary, &[], &[]).unwrap();
    Rc::get_mut(&mut candidate).unwrap().optimizing_candidate = false;
    (function, candidate)
}

fn compile(engine: &mut Engine, chunk: &Chunk, deopt: Option<usize>) -> Rc<JitCode> {
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    diagnostics::with_test_mode("execution", || {
        Rc::new(compile_with_deopt(chunk, &values, &layout, deopt).unwrap())
    })
}

fn install(
    engine: &mut Engine,
    function: &crate::ast::Function,
    chunk: &Rc<Chunk>,
    code: &Rc<JitCode>,
) {
    assert!(code.pc_offsets.is_empty());
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.optimizing_code.set(Some(chunk.clone())).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Jit);
}

fn region(chunk: &Chunk) -> (Cfg, stack::StackPlan, observed_regions::Plan) {
    let cfg = Cfg::build(chunk).unwrap();
    let stack = stack::StackPlan::build(chunk, &cfg).unwrap();
    let region =
        observed_regions::Plan::build(chunk, &cfg, &stack, &chunk.execution_coverage).unwrap();
    (cfg, stack, region)
}

#[test]
fn sparse_labels_reject_missing_native_branch_handler_entry_and_end_targets() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let _sparse = TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, "function subject(cold){var n=1;try{if(cold){n+=2;n+=3;n+=4;}return n;}catch(e){n+=5;return n;}finally{n+=6;}}subject(false);subject(false);");
    let (_, chunk) = candidate(&mut engine);
    let (cfg, stack, regions) = region(&chunk);
    let plan = Plan::build(
        chunk.jit_ops(),
        &cfg,
        &stack,
        0,
        Some(&regions),
        None,
        super::Selected::default(),
    )
    .unwrap();
    assert!(plan.labels().iter().any(|&label| !label));
    let mut mandatory = vec![0, chunk.jit_ops().len()];
    mandatory.extend(
        plan.reachable
            .iter()
            .enumerate()
            .filter_map(|(pc, &live)| live.then_some(pc)),
    );
    for pc in super::dispatch_destinations(chunk.jit_ops(), &cfg) {
        assert!(
            plan.labels()[pc],
            "every dynamic destination remains mapped"
        );
    }
    for missing in mandatory {
        let mut broken = plan.labels().to_vec();
        assert!(broken[missing]);
        broken[missing] = false;
        assert!(
            matches!(
                Plan::validate(&broken, &plan.reachable, &vec![false; broken.len()]),
                Err(CompileError::Unsupported)
            ),
            "missing original target {missing} must decline before lowering"
        );
    }
}

#[test]
fn sparse_native_labels_preserve_getter_throw_cold_branch_and_finally_once() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let setup = r#"
        var trace=[],bad=false,override=false,gets=0,prefixes=0;
        var receiver={get x(){gets++;trace.push('G');$262.gc();if(bad)throw 'bad';return 5;}};
        function subject(o,cold){prefixes++;trace.push('P');var total=2;
            try {total+=o.x;if(cold){trace.push('C');total+=10;total+=o.x;}return total;}
            catch(error){trace.push('E'+error);total+=20;return total;}
            finally{trace.push('F');if(override)return 'override';}}
        subject(receiver,false);subject(receiver,false);
    "#;
    let cases = [
        (false, false, false, "7|P,G,F|1:1", false),
        (true, false, false, "22|P,G,C,G,F|1:2", true),
        (false, true, false, "22|P,G,Ebad,F|1:1", true),
        (false, false, true, "override|P,G,F|1:1", true),
    ];
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, setup);
        for (cold, bad, override_, expected, _) in cases {
            let text = format!("trace=[];gets=prefixes=0;bad={bad};override={override_};subject(receiver,{cold})+'|'+trace.join(',')+'|'+prefixes+':'+gets");
            assert_eq!(eval(&mut engine, &text), expected);
        }
    }
    let mut sizes = Vec::new();
    for sparse in [false, true] {
        let _sparse = TestControl::new(sparse);
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        engine.set_tier_threshold(0);
        eval(&mut engine, setup);
        let (function, chunk) = candidate(&mut engine);
        let (cfg, _, regions) = region(&chunk);
        let omitted = (0..chunk.jit_ops().len())
            .filter(|&pc| regions.emission(pc) == observed_regions::Emission::OmittedInterior)
            .count();
        assert!(omitted > 0);
        let code = compile(&mut engine, &chunk, None);
        let [allocated, absent] = TEST_COUNTS.with(std::cell::Cell::get);
        assert_eq!(allocated + absent, chunk.jit_ops().len() + 1);
        if sparse {
            assert!(absent >= omitted);
        } else {
            assert_eq!(absent, 0);
        }
        let record = code.optimizing_diagnostics.as_ref().unwrap();
        sizes.push((record.ir_instructions.get(), record.ir_blocks.get()));
        install(&mut engine, &function, &chunk, &code);
        for (call, (cold, bad, override_, expected, cut)) in cases.into_iter().enumerate() {
            ENTRIES.with(|entries| entries.borrow_mut().clear());
            let text = format!("trace=[];gets=prefixes=0;bad={bad};override={override_};subject(receiver,{cold})+'|'+trace.join(',')+'|'+prefixes+':'+gets");
            assert_eq!(eval(&mut engine, &text), expected);
            ENTRIES.with(|entries| {
                let entries = entries.borrow();
                assert_eq!(entries.len(), usize::from(cut));
                if let Some((pc, ..)) = entries.first() {
                    assert_eq!(regions.emission(*pc), observed_regions::Emission::Cut);
                    assert_eq!(
                        cfg.blocks()[cfg.block_at(*pc).unwrap().0 as usize].start,
                        *pc
                    );
                }
            });
            assert_eq!(record.counters[0].get(), (call + 1) as u64);
            assert_eq!(
                record.counters[diagnostics::NORMAL_EXIT].get(),
                (call + 1) as u64
            );
            assert_eq!(chunk.optimizing_misses.get(), 0);
            assert!(code.residency.referenced.get() > 0);
            assert_eq!(code.residency.active.get(), 0);
            assert!(engine.interp.fn_frames.is_empty());
        }
    }
    assert!(
        sizes[1].0 < sizes[0].0 && sizes[1].1 < sizes[0].1,
        "physical sparse labels reduce real sealed IR: {sizes:?}"
    );
}

#[test]
fn full_body_and_inside_cold_region_stress_preserve_actual_original_entry() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let _sparse = TestControl::new(true);
    for stress in [false, true] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        engine.set_tier_threshold(0);
        eval(&mut engine,"var prefixes=0;function subject(cold){prefixes++;var n=1;if(cold){n+=2;n+=3;n+=4;}return n;}subject(false);subject(false);");
        let (function, chunk) = candidate(&mut engine);
        let (_, _, regions) = region(&chunk);
        let interior = (0..chunk.jit_ops().len())
            .find(|&pc| regions.emission(pc) == observed_regions::Emission::OmittedInterior)
            .unwrap();
        let code = if stress {
            compile(&mut engine, &chunk, Some(interior))
        } else {
            let _regions_off = observed_regions::TestControl::new(false);
            compile(&mut engine, &chunk, None)
        };
        let [allocated, absent] = TEST_COUNTS.with(std::cell::Cell::get);
        assert_eq!(allocated + absent, chunk.jit_ops().len() + 1);
        assert!(allocated > 0);
        install(&mut engine, &function, &chunk, &code);
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        assert_eq!(
            eval(&mut engine, "prefixes=0;subject(true)+'|'+prefixes"),
            "10|1"
        );
        ENTRIES.with(|entries| {
            let entries = entries.borrow();
            assert_eq!(entries.len(), usize::from(stress));
            if stress {
                assert_eq!(entries[0].0, interior);
            }
        });
        assert!(code.residency.referenced.get() > 0);
        assert_eq!(code.residency.active.get(), 0);
        assert!(engine.interp.fn_frames.is_empty());
    }
}

fn selected_child(name: &str) -> bool {
    const CONTROL: &str = "TRUST_JS_TEST_SELECTED_NATIVE_PCS_CHILD";
    if std::env::var(CONTROL).as_deref() == Ok(name) {
        return false;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command.args([
        "--exact",
        &format!("jit::optimizing::native_pcs::tests::{name}"),
        "--nocapture",
        "--test-threads=1",
    ]);
    for (key, _) in std::env::vars_os() {
        if key.as_encoded_bytes().starts_with(b"TRUST_JS_")
            || key.as_encoded_bytes().starts_with(b"LUMEN_JS_")
        {
            command.env_remove(key);
        }
    }
    command
        .env(CONTROL, name)
        .env("TRUST_JS_OPT_JIT", "0")
        .env("TRUST_JS_INLINE_AT", "0")
        .env("TRUST_JS_OPT_JIT_BORROWED_OPERANDS", "1")
        .env("TRUST_JS_OPT_JIT_VALUE_FACTS", "0")
        .env_remove("LD_PRELOAD");
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("test result: ok. 1 passed; 0 failed;")
    );
    true
}

fn selections(
    chunk: &Chunk,
    cfg: &Cfg,
    stack: &stack::StackPlan,
    regions: Option<&observed_regions::Plan>,
    deopt: Option<usize>,
) -> (
    Option<borrowed_operands::Plan>,
    Option<string_literals::Plan>,
    Option<string_dispatch::Plan>,
) {
    let borrowed = borrowed_operands::Plan::build(
        chunk,
        cfg,
        stack,
        regions,
        None,
        deopt,
        NativeEntryKind::FreshFrame,
        None,
    );
    let literals = string_literals::Plan::build(chunk, cfg, stack, regions, deopt);
    let dispatch = literals
        .as_ref()
        .and_then(|l| string_dispatch::Plan::build(chunk, cfg, stack, regions, None, deopt, l));
    (borrowed, literals, dispatch)
}

#[test]
fn selected_labels_remove_only_consumed_fusion_entries_and_price_full_source_work() {
    const NAME: &str =
        "selected_labels_remove_only_consumed_fusion_entries_and_price_full_source_work";
    if selected_child(NAME) {
        return;
    }
    let _sparse = TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, "function finish(a,b){return a.v+b.v;}function subject(a,b,key){var home=a;home=b;var total=finish(home,a);switch(key){case 'a':total+=1;break;case 'b':total+=2;break;case 'c':total+=3;break;default:total+=4;}if(key==='last')total+=8;return total;}subject({v:2},{v:3},'a');");
    let (_, chunk) = candidate(&mut engine);
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let (borrowed, literals, dispatch) = selections(&chunk, &cfg, &stack, None, None);
    assert!(borrowed.is_some() && literals.is_some() && dispatch.is_some());
    let selected = super::Selected {
        borrowed: borrowed.as_ref(),
        dispatch: dispatch.as_ref(),
        literals: literals.as_ref(),
    };
    let plan = Plan::build(chunk.jit_ops(), &cfg, &stack, 0, None, None, selected).unwrap();
    for kind in [
        super::Kind::Borrowed,
        super::Kind::Dispatch,
        super::Kind::Literal,
    ] {
        assert!(plan.kinds.contains(&kind), "actual selected {kind:?}");
    }
    for (pc, &kind) in plan.kinds.iter().enumerate() {
        if kind == super::Kind::Interior {
            assert!(!plan.labels()[pc]);
        }
        if plan.reachable[pc] {
            Plan::successors(chunk.jit_ops(), pc, kind, selected, |target| {
                assert!(
                    plan.labels()[target],
                    "exact normal successor {pc}->{target}"
                );
                Ok(())
            })
            .unwrap();
        }
    }
    let end = chunk.jit_ops().len();
    assert!(plan.labels()[end]);
    assert!(
        !plan.emits_end(),
        "unused end mapping has no fabricated SSA reads"
    );
    let mut corrupt = plan.labels().to_vec();
    for pc in 0..=end {
        if plan.reachable[pc] || pc == end {
            corrupt[pc] = false;
            assert!(
                Plan::validate(&corrupt, &plan.reachable, &vec![false; end + 1]).is_err(),
                "missing required {pc}"
            );
            corrupt[pc] = true;
        }
    }
    let dispatch_pc = plan
        .kinds
        .iter()
        .enumerate()
        .find_map(|(pc, &kind)| (kind == super::Kind::Dispatch).then_some(pc))
        .unwrap();
    // Falsify a formerly external branch AFTER obtaining the immutable plans.
    // Preflight must independently reject entry into a grouped comparison.
    let mut ops = chunk.jit_ops().to_vec();
    let end = dispatch.as_ref().unwrap().site(dispatch_pc).unwrap().end();
    let outside = ops
        .iter()
        .enumerate()
        .find_map(|(pc, op)| (pc >= end && matches!(op, Op::JumpIfFalse(_))).then_some(pc))
        .unwrap();
    ops[outside] = Op::JumpIfFalse((dispatch_pc + 5) as u32);
    let changed = Cfg::test_build_with(&ops, |pc| chunk.jit_stack_effect(pc)).unwrap();
    assert!(Plan::build(&ops, &changed, &stack, 0, None, None, selected).is_err());
    // A different target INSIDE the same group is not the certified branch.
    // Co-owned labels cannot excuse a changed source-order comparison edge.
    let mut internal = chunk.jit_ops().to_vec();
    internal[dispatch_pc + 3] = Op::JumpIfFalse((dispatch_pc + 10) as u32);
    let changed = Cfg::test_build_with(&internal, |pc| chunk.jit_stack_effect(pc)).unwrap();
    assert!(Plan::build(&internal, &changed, &stack, 0, None, None, selected).is_err());
    // Syntactic operand and comparison identities are also exact certificate fields.
    for offset in [0, 1, 2] {
        let mut altered = chunk.jit_ops().to_vec();
        altered[dispatch_pc + offset] = match altered[dispatch_pc + offset] {
            Op::LoadLocal(slot) => Op::LoadLocal(slot + 1),
            Op::Const(_) => Op::Undef,
            Op::StrictEq => Op::StrictNotEq,
            _ => panic!("literal fixture operands"),
        };
        assert!(Plan::build(&altered, &cfg, &stack, 0, None, None, selected).is_err());
    }
    let loan = (0..chunk.jit_ops().len())
        .find(|&pc| borrowed.as_ref().unwrap().site(pc).is_some())
        .unwrap();
    let mut changed_loan = chunk.jit_ops().to_vec();
    let Op::LoadLocal(slot) = changed_loan[loan] else {
        unreachable!()
    };
    changed_loan[loan] = Op::LoadLocal(slot + 1);
    assert!(Plan::build(&changed_loan, &cfg, &stack, 0, None, None, selected).is_err());
    let single = plan
        .kinds
        .iter()
        .position(|&kind| kind == super::Kind::Literal)
        .unwrap();
    let mut changed_single = chunk.jit_ops().to_vec();
    changed_single[single + 3] = Op::JumpIfFalse((single + 1) as u32);
    assert!(Plan::build(&changed_single, &cfg, &stack, 0, None, None, selected).is_err());
    for target in 1..4 {
        assert!(Plan::build(
            chunk.jit_ops(),
            &cfg,
            &stack,
            dispatch_pc + target,
            None,
            None,
            selected
        )
        .is_err());
        assert!(Plan::build(
            chunk.jit_ops(),
            &cfg,
            &stack,
            0,
            None,
            Some(dispatch_pc + target),
            selected
        )
        .is_err());
    }
    assert_eq!(super::analysis_work(0), Some(64));
    assert_eq!(super::analysis_work(100), Some(6464));
    assert!(super::analysis_work(usize::MAX).is_none());
}

#[test]
fn terminal_vm_cut_omits_later_cold_loop_islands_but_executes_original_vm_once() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let setup = "var trace=[];function subject(cold){trace.push('P');var n=0;if(cold){trace.push('C');while(n<3){trace.push(n);n++;}trace.push('D');}return n;}subject(false);subject(false);";
    for sparse in [false, true] {
        let _sparse = TestControl::new(sparse);
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        engine.set_tier_threshold(0);
        eval(&mut engine, setup);
        let (function, chunk) = candidate(&mut engine);
        let (cfg, stack, regions) = region(&chunk);
        let plan = Plan::build(
            chunk.jit_ops(),
            &cfg,
            &stack,
            0,
            Some(&regions),
            None,
            super::Selected::default(),
        )
        .unwrap();
        let cut_islands = (0..chunk.jit_ops().len())
            .filter(|&pc| {
                regions.emission(pc) == observed_regions::Emission::Cut && !plan.labels()[pc]
            })
            .count();
        assert_eq!(
            cut_islands > 0,
            sparse,
            "a cut is terminal, not native flow to later cold leaders"
        );
        if sparse {
            let loop_header = cfg
                .loops()
                .first()
                .map(|l| cfg.blocks()[l.header.0 as usize].start)
                .unwrap();
            assert!(
                !plan.labels()[loop_header],
                "a cut-only backedge invents no native root"
            );
        }
        let code = compile(&mut engine, &chunk, None);
        install(&mut engine, &function, &chunk, &code);
        ENTRIES.with(|entries| entries.borrow_mut().clear());
        assert_eq!(
            eval(&mut engine, "trace=[];subject(true)+'|'+trace.join(',')"),
            "3|P,C,0,1,2,D"
        );
        assert_eq!(
            code.optimizing_diagnostics.as_ref().unwrap().counters[0].get(),
            1
        );
        ENTRIES.with(|entries| {
            let entries = entries.borrow();
            assert_eq!(entries.len(), 1);
            assert_eq!(
                regions.emission(entries[0].0),
                observed_regions::Emission::Cut
            );
            assert!(plan.labels()[entries[0].0]);
        });
        assert_eq!(chunk.optimizing_misses.get(), 0);
    }
}

const FUSED_LOOP: &str = r#"
    var trace=[],bad=false,left={v:3},right={v:7};
    function observe(a,b){trace.push('O');$262.gc();if(bad)throw 'bad';return a.v+b.v;}
    function subject(a,b,key){var home=a,total=0;try{for(var i=0;i<3;i++){
        home=b;total+=observe(home,a);
        switch(key){case 'a':total+=1;break;case 'b':total+=2;break;case 'c':total+=3;break;default:total+=4;}
    }if(key==='last')total+=8;return total+':'+home.v;}catch(e){return e+':'+home.v;}finally{trace.push('F');}}
    subject(left,right,'a');subject(left,right,'a');
"#;

#[test]
fn fused_native_loop_calls_gc_and_throw_keep_real_endpoint_and_handler_roots() {
    const NAME: &str = "fused_native_loop_calls_gc_and_throw_keep_real_endpoint_and_handler_roots";
    if selected_child(NAME) {
        return;
    }
    let _regions = observed_regions::TestControl::new(false);
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, FUSED_LOOP);
        assert_eq!(
            eval(
                &mut engine,
                "trace=[];subject(left,right,'a')+'|'+trace.join(',')"
            ),
            "33:7|O,O,O,F"
        );
        assert_eq!(
            eval(
                &mut engine,
                "trace=[];bad=true;subject(left,right,'b')+'|'+trace.join(',')"
            ),
            "bad:7|O,F"
        );
    }
    let mut sizes = Vec::new();
    for sparse in [false, true] {
        let _sparse = TestControl::new(sparse);
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        engine.set_tier_threshold(0);
        eval(&mut engine, FUSED_LOOP);
        let (function, chunk) = candidate(&mut engine);
        let cfg = Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        let (borrowed, literals, dispatch) = selections(&chunk, &cfg, &stack, None, None);
        assert!(borrowed.is_some() && dispatch.is_some());
        let plan = Plan::build(
            chunk.jit_ops(),
            &cfg,
            &stack,
            0,
            None,
            None,
            super::Selected {
                borrowed: borrowed.as_ref(),
                dispatch: dispatch.as_ref(),
                literals: literals.as_ref(),
            },
        )
        .unwrap();
        for root in cfg.handler_roots() {
            assert!(plan.labels()[cfg.blocks()[root.target.0 as usize].start]);
        }
        for header in cfg.loops() {
            assert!(plan.labels()[cfg.blocks()[header.header.0 as usize].start]);
        }
        borrowed_operands::TEST_SELECTED.with(|sites| sites.borrow_mut().clear());
        string_dispatch::TEST_SELECTED.with(|sites| sites.borrow_mut().clear());
        let code = compile(&mut engine, &chunk, None);
        assert!(!borrowed_operands::TEST_SELECTED.with(|sites| sites.borrow().is_empty()));
        assert!(!string_dispatch::TEST_SELECTED.with(|sites| sites.borrow().is_empty()));
        let record = code.optimizing_diagnostics.as_ref().unwrap();
        sizes.push((record.ir_instructions.get(), record.ir_blocks.get()));
        install(&mut engine, &function, &chunk, &code);
        for (source, expected) in [
            (
                "trace=[];subject(left,right,'a')+'|'+trace.join(',')",
                "33:7|O,O,O,F",
            ),
            (
                "trace=[];bad=true;subject(left,right,'b')+'|'+trace.join(',')",
                "bad:7|O,F",
            ),
        ] {
            ENTRIES.with(|entries| entries.borrow_mut().clear());
            assert_eq!(eval(&mut engine, source), expected);
            assert_eq!(ENTRIES.with(|e| e.borrow().len()), 0);
        }
        assert_eq!(record.counters[0].get(), 2);
        assert_eq!(record.counters[diagnostics::NORMAL_EXIT].get(), 2);
        assert_eq!(code.residency.active.get(), 0);
        assert!(engine.interp.fn_frames.is_empty());
    }
    assert!(
        sizes[1].0 < sizes[0].0 && sizes[1].1 < sizes[0].1,
        "{sizes:?}"
    );
}

#[test]
fn every_consumed_kind_stress_and_after_call_recovery_keep_exact_original_pc() {
    const NAME: &str = "every_consumed_kind_stress_and_after_call_recovery_keep_exact_original_pc";
    if selected_child(NAME) {
        return;
    }
    let _regions = observed_regions::TestControl::new(false);
    let mut reference = Engine::new();
    reference.set_tier(Tier::Bytecode);
    reference.set_tier_threshold(0);
    eval(&mut reference, FUSED_LOOP);
    let (_, chunk) = candidate(&mut reference);
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let (borrowed, literals, dispatch) = selections(&chunk, &cfg, &stack, None, None);
    let borrowed = borrowed.unwrap();
    let dispatch = dispatch.unwrap();
    let mut targets = std::collections::BTreeSet::new();
    for call in [false, true] {
        let head = (0..chunk.jit_ops().len())
            .find(|&pc| {
                borrowed.site(pc).is_some_and(|s| {
                    matches!(chunk.jit_ops()[s.native_end()], Op::CallWithThis(..)) == call
                })
            })
            .unwrap();
        let site = borrowed.site(head).unwrap();
        for target in head..=site.native_end() + 1 {
            targets.insert((target, "a"));
        }
    }
    let head = (0..chunk.jit_ops().len())
        .find(|&pc| dispatch.site(pc).is_some())
        .unwrap();
    let group = dispatch.site(head).unwrap();
    assert_eq!(group.count(), 3, "actual a/b/c source-order group");
    for (index, key) in ["a", "b", "c"].into_iter().enumerate() {
        // Every original comparison boundary and each retained case Jump must
        // survive stress, not only the first comparison in the group.
        for target in head + index * 5..head + index * 5 + 5 {
            targets.insert((target, key));
        }
    }
    targets.insert((group.end(), "none"));
    let literals = literals.unwrap();
    let selected = super::Selected {
        borrowed: Some(&borrowed),
        dispatch: Some(&dispatch),
        literals: Some(&literals),
    };
    let graph = Plan::build(chunk.jit_ops(), &cfg, &stack, 0, None, None, selected).unwrap();
    let literal = graph
        .kinds
        .iter()
        .enumerate()
        .find_map(|(pc, &kind)| (kind == super::Kind::Literal).then_some(pc))
        .expect("actual independent literal comparison");
    for target in literal..literal + 4 {
        targets.insert((target, "a"));
    }
    targets.insert((literal + 4, "last"));
    targets.insert((literals.site(literal).unwrap().identity().3, "a"));
    for (target, key) in targets {
        let _sparse = TestControl::new(true);
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        engine.set_tier_threshold(0);
        eval(&mut engine, FUSED_LOOP);
        let (function, current) = candidate(&mut engine);
        assert_eq!(
            format!("{:?}", current.jit_ops()),
            format!("{:?}", chunk.jit_ops())
        );
        let code = compile(&mut engine, &current, Some(target));
        install(&mut engine, &function, &current, &code);
        ENTRIES.with(|e| e.borrow_mut().clear());
        let expected = match key {
            "a" => "33:7|O,O,O,F",
            "b" => "36:7|O,O,O,F",
            "c" => "39:7|O,O,O,F",
            "none" => "42:7|O,O,O,F",
            "last" => "50:7|O,O,O,F",
            _ => unreachable!(),
        };
        assert_eq!(
            eval(
                &mut engine,
                &format!("trace=[];subject(left,right,'{key}')+'|'+trace.join(',')")
            ),
            expected
        );
        ENTRIES.with(|entries| {
            assert_eq!(
                &*entries.borrow(),
                &[(target, stack.at[target].depth, 0)],
                "exact before-operation state at {target}"
            );
        });
        assert_eq!(
            code.optimizing_diagnostics.as_ref().unwrap().counters[0].get(),
            1
        );
        assert_eq!(code.residency.active.get(), 0);
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn equal_depth_handler_root_cannot_enter_a_consumed_dispatch_comparison() {
    const NAME: &str = "equal_depth_handler_root_cannot_enter_a_consumed_dispatch_comparison";
    if selected_child(NAME) {
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, "function subject(x){var keep={};try{try{keep=null;throw 1;}catch(e){switch(x){case'a':return'A';case'b':return'B';case'z':return'Z';default:return'?';}}}catch(e){return'error';}}subject('a');");
    let (function, _) = candidate(&mut engine);
    let (chunk, _, _) = bytecode::compile_test_parked_handler_owner(&function, "keep", 2).unwrap();
    let cfg = Cfg::build(&chunk).unwrap();
    let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
    let (borrowed, literals, dispatch) = selections(&chunk, &cfg, &stack, None, None);
    let dispatch = dispatch.unwrap();
    let head = (0..chunk.jit_ops().len())
        .find(|&pc| dispatch.site(pc).is_some())
        .unwrap();
    assert_eq!(stack.at[head + 5].depth, 1);
    let selected = super::Selected {
        borrowed: borrowed.as_ref(),
        literals: literals.as_ref(),
        dispatch: Some(&dispatch),
    };
    assert!(Plan::build(chunk.jit_ops(), &cfg, &stack, 0, None, None, selected).is_ok());
    let mut ops = chunk.jit_ops().to_vec();
    let handler = ops
        .iter()
        .position(|op| matches!(op, Op::PushHandler(_)))
        .unwrap();
    ops[handler] = Op::PushHandler((head + 5) as u32);
    let changed = Cfg::test_build_with(&ops, |pc| chunk.jit_stack_effect(pc)).unwrap();
    assert!(
        Plan::build(&ops, &changed, &stack, 0, None, None, selected).is_err(),
        "valid equal-depth full CFG still cannot waive an interior root"
    );
}

#[test]
fn sparse_dynamic_completion_roots_preserve_finally_iterator_close_and_overrides() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let setup = r#"
        var trace=[],closeFail=false;
        var iterable={ [Symbol.iterator](){var n=0;return{
            next(){n++;return {value:n,done:n>2};},
            return(){trace.push('R');$262.gc();if(closeFail)throw 'close';return {};}
        };}};
        function subject(mode){try{outer:for(const value of iterable){try{
            if(mode==='continue'&&value===1)continue outer;
            if(mode==='break')break outer;
            if(mode==='return')return 'r';
            if(mode==='throw')throw 'boom';
            trace.push(value);
        }finally{trace.push('F'+value);}}return 'done';}
        catch(error){trace.push('E'+error);return 'caught';}finally{trace.push('O');}}
        subject('normal');subject('normal');
    "#;
    let cases = [
        ("normal", false, "done|1,F1,2,F2,O"),
        ("continue", false, "done|F1,2,F2,O"),
        ("break", false, "done|F1,R,O"),
        ("return", false, "r|F1,R,O"),
        ("throw", false, "caught|F1,R,Eboom,O"),
        ("break", true, "caught|F1,R,Eclose,O"),
        ("throw", true, "caught|F1,R,Eboom,O"),
    ];
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, setup);
        for (mode, fails, expected) in cases {
            assert_eq!(
                eval(
                    &mut engine,
                    &format!("trace=[];closeFail={fails};subject('{mode}')+'|'+trace.join(',')")
                ),
                expected,
                "{tier:?}:{mode}:{fails}"
            );
        }
    }
    for sparse in [false, true] {
        let _sparse = TestControl::new(sparse);
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        engine.set_tier_threshold(0);
        eval(&mut engine, setup);
        let (function, chunk) = candidate(&mut engine);
        let (cfg, stack, regions) = region(&chunk);
        let plan = Plan::build(
            chunk.jit_ops(),
            &cfg,
            &stack,
            0,
            Some(&regions),
            None,
            super::Selected::default(),
        )
        .unwrap();
        let mut modes = 0;
        for op in chunk.jit_ops() {
            if let Op::PushFinally(t, r, b, s, j) = *op {
                for target in [t, r, b, s, j] {
                    assert!(plan.labels()[target as usize]);
                    modes += 1;
                }
            }
            if let Op::PushIterator(t, r, b, s) = *op {
                for target in [t, r, b, s] {
                    assert!(plan.labels()[target as usize]);
                    modes += 1;
                }
            }
        }
        assert!(modes >= 5, "real full-mode finally metadata");
        let code = compile(&mut engine, &chunk, None);
        install(&mut engine, &function, &chunk, &code);
        for (mode, fails, expected) in cases {
            assert_eq!(
                eval(
                    &mut engine,
                    &format!("trace=[];closeFail={fails};subject('{mode}')+'|'+trace.join(',')")
                ),
                expected,
                "native:{mode}:{fails}"
            );
        }
        assert_eq!(
            code.optimizing_diagnostics.as_ref().unwrap().counters[0].get(),
            cases.len() as u64
        );
        assert_eq!(code.residency.active.get(), 0);
        assert!(engine.interp.fn_frames.is_empty());
    }
}
