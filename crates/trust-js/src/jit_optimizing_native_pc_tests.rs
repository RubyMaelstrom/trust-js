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
    let (cfg, _, regions) = region(&chunk);
    let plan = Plan::build(chunk.jit_ops(), &cfg, 0, Some(&regions)).unwrap();
    assert!(plan.labels().iter().any(|&label| !label));
    let mut mandatory = vec![0, chunk.jit_ops().len()];
    mandatory.extend(cfg.blocks().iter().map(|block| block.start));
    mandatory.extend(
        cfg.handler_roots()
            .iter()
            .map(|root| cfg.blocks()[root.target.0 as usize].start),
    );
    mandatory.extend(
        chunk
            .jit_ops()
            .iter()
            .filter_map(crate::jit_ir::jump_target),
    );
    mandatory.push(
        (0..chunk.jit_ops().len())
            .find(|&pc| {
                regions.emission(pc) == observed_regions::Emission::Native
                    && !cfg.blocks().iter().any(|block| block.start == pc)
            })
            .unwrap(),
    );
    for missing in mandatory {
        let mut broken = plan.labels().to_vec();
        assert!(broken[missing]);
        broken[missing] = false;
        assert!(
            matches!(
                Plan::validate(&broken, chunk.jit_ops(), &cfg, 0, Some(&regions)),
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
        assert_eq!(
            TEST_COUNTS.with(std::cell::Cell::get),
            if sparse {
                [chunk.jit_ops().len() + 1 - omitted, omitted]
            } else {
                [chunk.jit_ops().len() + 1, 0]
            }
        );
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
fn full_body_and_inside_cold_region_stress_keep_every_original_pc() {
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
        assert_eq!(
            TEST_COUNTS.with(std::cell::Cell::get),
            [chunk.jit_ops().len() + 1, 0]
        );
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
