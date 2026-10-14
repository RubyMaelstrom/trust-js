//! Native observed regions finish the original invocation through exact VM cuts.
//! These forced coverage fixtures prove emission/recovery, not large-body admission
//! or application performance. Hot publication has a separate real-policy fixture.

use super::*;
use crate::bytecode::{native_deopt::ENTRIES, Tier};
use crate::execution_coverage::{Reason, State};
use crate::interpreter::Interp;
use crate::value::{Callable, Object, Property, Value as JsValue};
use crate::{Completion, Engine};
use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

#[path = "jit_optimizing_observed_region_guard_tests.rs"]
mod integration_guard_tests;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("region fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn lookup_function(engine: &mut Engine, name: &str) -> Rc<crate::ast::Function> {
    let env = engine.interp.global_env.clone();
    let callable = engine.interp.get_var(name, &env).ok().unwrap();
    let object = callable.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("ordinary User")
    };
    user.func.clone()
}

fn baseline(function: &crate::ast::Function) -> Rc<Chunk> {
    function
        .execution_code()
        .and_then(Option::as_ref)
        .unwrap()
        .clone()
}

fn make_candidate(function: &crate::ast::Function, primary: &Chunk) -> Rc<Chunk> {
    let mut candidate = bytecode::compile_for_optimizer(function, primary, &[], &[]).unwrap();
    // Forced generic emission is not a claim that this semantic fixture earns hot credit.
    Rc::get_mut(&mut candidate).unwrap().optimizing_candidate = false;
    candidate
}

fn compile(engine: &mut Engine, chunk: &Chunk, regions: bool) -> Rc<JitCode> {
    let _control = observed_regions::TestControl::new(regions);
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    diagnostics::with_test_mode("execution", || {
        Rc::new(compile_with_deopt(chunk, &values, &layout, None).unwrap())
    })
}

fn install(
    engine: &mut Engine,
    function: &crate::ast::Function,
    chunk: &Rc<Chunk>,
    code: &Rc<JitCode>,
) {
    assert!(code.pc_offsets.is_empty(), "actual optimizing-native body");
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(function.optimizing_code.set(Some(chunk.clone())).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Jit);
}

fn selection(chunk: &Chunk) -> observed_regions::Plan {
    let _control = observed_regions::TestControl::new(true);
    let cfg = Cfg::build(chunk).unwrap();
    let stack = stack::StackPlan::build(chunk, &cfg).unwrap();
    observed_regions::Plan::build(chunk, &cfg, &stack, &chunk.execution_coverage)
        .expect("actual entered baseline entry and unseen block(s)")
}

fn clear_entries() {
    ENTRIES.with(|entries| entries.borrow_mut().clear());
}

fn assert_one_cut(
    engine: &Engine,
    chunk: &Chunk,
    code: &JitCode,
    plan: &observed_regions::Plan,
) -> usize {
    let pc = ENTRIES.with(|entries| {
        let entries = entries.borrow();
        assert_eq!(entries.len(), 1, "one terminal transfer, no native rejoin");
        assert_eq!(plan.emission(entries[0].0), observed_regions::Emission::Cut);
        entries[0].0
    });
    assert!(code.residency.referenced.get() > 0, "real native entry");
    assert_eq!(code.residency.active.get(), 0);
    assert_eq!(
        chunk.optimizing_misses.get(),
        0,
        "ordinary cold cuts never retire specialization"
    );
    assert!(engine.interp.fn_frames.is_empty());
    pc
}

#[test]
fn frozen_selection_omits_real_interior_lowering_keeps_unknown_and_shrinks_ir() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    let body = (0..24)
        .map(|n| format!("total+=o.k{n};"))
        .collect::<String>();
    let fields = (0..24)
        .map(|n| format!("k{n}:{n}"))
        .collect::<Vec<_>>()
        .join(",");
    eval(&mut engine, &format!("var receiver={{{fields}}},prefixes=0;function subject(o,cold){{prefixes++;var total=1;if(cold){{{body}}}return total;}}subject(receiver,false);subject(receiver,false);"));
    let function = lookup_function(&mut engine, "subject");
    let primary = baseline(&function);
    let candidate = make_candidate(&function, &primary);
    let plan = selection(&candidate);
    assert!(plan.native_ops < candidate.jit_ops().len() / 2);
    let full = compile(&mut engine, &candidate, false);
    let full_ir = full
        .optimizing_diagnostics
        .as_ref()
        .unwrap()
        .ir_instructions
        .get();
    let partial = compile(&mut engine, &candidate, true);
    let partial_ir = partial
        .optimizing_diagnostics
        .as_ref()
        .unwrap()
        .ir_instructions
        .get();
    let lowered = observed_regions::TEST_LOWERED.with(|pcs| pcs.borrow().clone());
    for pc in 0..candidate.jit_ops().len() {
        if plan.emission(pc) != observed_regions::Emission::Native {
            assert!(
                !lowered.contains(&pc),
                "no unselected opcode reaches native lowering"
            );
        }
    }
    assert!(partial_ir < full_ir);
    assert!(partial.len < full.len);
    // Later source execution cannot change the frozen candidate selection.
    assert_eq!(eval(&mut engine, "subject(receiver,true)"), "277");
    let snapshot = candidate.execution_coverage.frozen_snapshot().unwrap();
    let cut = (0..candidate.jit_ops().len())
        .find(|&pc| plan.emission(pc) == observed_regions::Emission::Cut)
        .unwrap();
    assert_eq!(snapshot.state_for(&candidate, cut), State::UnseenInWindow);
    assert_eq!(
        primary
            .execution_coverage
            .freeze()
            .frozen_snapshot()
            .unwrap()
            .state_for(&primary, cut),
        State::Entered
    );
    install(&mut engine, &function, &candidate, &partial);
    clear_entries();
    assert_eq!(
        eval(
            &mut engine,
            "prefixes=0;subject(receiver,true)+'|'+prefixes"
        ),
        "277|1"
    );
    assert_one_cut(&engine, &candidate, &partial, &plan);
    let record = partial.optimizing_diagnostics.as_ref().unwrap();
    assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 1);
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        0
    );
    eprintln!("observed-region size fullIR={full_ir} partialIR={partial_ir} fullBytes={} partialBytes={} ops={} selected={} cuts={}", full.len,partial.len,candidate.jit_ops().len(),plan.native_ops,plan.cut_blocks);

    // Sticky Unknown retains the entire original body. Negative feedback from
    // a different immutable graph also cannot omit any native operation.
    primary.execution_coverage.invalidate(Reason::AstBody);
    let unknown = make_candidate(&function, &primary);
    let _control = observed_regions::TestControl::new(true);
    let unknown_cfg = Cfg::build(&unknown).unwrap();
    let unknown_stack = stack::StackPlan::build(&unknown, &unknown_cfg).unwrap();
    assert!(observed_regions::Plan::build(
        &unknown,
        &unknown_cfg,
        &unknown_stack,
        &unknown.execution_coverage
    )
    .is_none());
    eval(
        &mut engine,
        "function unrelated(o,cold){return cold?o.k0:o.k1;}unrelated(receiver,false);",
    );
    let unrelated = baseline(&lookup_function(&mut engine, "unrelated"));
    let unrelated_cfg = Cfg::build(&unrelated).unwrap();
    let unrelated_stack = stack::StackPlan::build(&unrelated, &unrelated_cfg).unwrap();
    assert!(observed_regions::Plan::build(
        &unrelated,
        &unrelated_cfg,
        &unrelated_stack,
        &candidate.execution_coverage
    )
    .is_none());
}

#[test]
fn dead_unseen_tail_cannot_create_an_observed_region_certificate() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "function subject(value){return value+1;}subject(3);subject(3);",
    );
    let function = lookup_function(&mut engine, "subject");
    let candidate = make_candidate(&function, &baseline(&function));
    let cfg = Cfg::build(&candidate).unwrap();
    let stack = stack::StackPlan::build(&candidate, &cfg).unwrap();
    let snapshot = candidate.execution_coverage.frozen_snapshot().unwrap();
    let unseen = cfg
        .blocks()
        .iter()
        .filter(|block| snapshot.state_for(&candidate, block.start) == State::UnseenInWindow)
        .collect::<Vec<_>>();
    assert!(
        !unseen.is_empty(),
        "real compiler appends a dead ReturnUndef tail"
    );
    assert!(unseen.iter().all(|block| !stack.reachable[block.start]));
    assert!(
        observed_regions::Plan::build(&candidate, &cfg, &stack, &candidate.execution_coverage)
            .is_none(),
        "only reachable original-CFG cut boundaries can authorize emission selection"
    );
    let code = compile(&mut engine, &candidate, true);
    install(&mut engine, &function, &candidate, &code);
    clear_entries();
    assert_eq!(eval(&mut engine, "subject(9)"), "10");
    ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(code.residency.active.get(), 0);
    assert_eq!(
        code.optimizing_diagnostics.as_ref().unwrap().counters[diagnostics::OBSERVED_REGION_EXIT]
            .get(),
        0
    );
}

thread_local! {
    static PAYLOADS: RefCell<Vec<Weak<RefCell<Object>>>> = const { RefCell::new(Vec::new()) };
    static COERCIONS: Cell<usize> = const { Cell::new(0) };
    static LIVE_AT_COERCION: Cell<bool> = const { Cell::new(false) };
}

fn coerce_payload(i: &mut Interp, _: JsValue, _: &[JsValue]) -> Result<JsValue, JsValue> {
    i.gc_collect();
    COERCIONS.with(|count| count.set(count.get() + 1));
    LIVE_AT_COERCION.with(|live| {
        live.set(PAYLOADS.with(|payloads| payloads.borrow().last().unwrap().strong_count() != 0))
    });
    Ok(JsValue::Num(7.0))
}

fn make_payload(i: &mut Interp, _: JsValue, _: &[JsValue]) -> Result<JsValue, JsValue> {
    let object = Object::new(Some(i.object_proto.clone()));
    object.borrow_mut().props.insert(
        "tag",
        Property::data(JsValue::str("alive"), true, true, true),
    );
    i.def_method(&object, "valueOf", 0, coerce_payload);
    PAYLOADS.with(|payloads| payloads.borrow_mut().push(Rc::downgrade(&object)));
    Ok(JsValue::Obj(object))
}

#[test]
fn owning_operand_survives_cold_call_gc_and_normal_or_throw_completion_once() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    for throwing in [false, true] {
        PAYLOADS.with(|payloads| payloads.borrow_mut().clear());
        COERCIONS.with(|count| count.set(0));
        LIVE_AT_COERCION.with(|live| live.set(false));
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        engine.set_tier_threshold(0);
        let global = engine.interp.global.clone();
        engine
            .interp
            .def_method(&global, "makePayload", 0, make_payload);
        eval(
            &mut engine,
            r#"
            var trace=[],gets=0,prefixes=0,finishes=0,throwing=false;
            var source={get value(){gets++;trace.push('G');return makePayload();}};
            function finish(){finishes++;trace.push('F');$262.gc();if(throwing)throw 'boom';return 2;}
            function subject(cold){prefixes++;trace.push('P');return source.value+(cold?finish():1);}
            subject(false);subject(false);trace=[];gets=prefixes=finishes=0;
        "#,
        );
        let function = lookup_function(&mut engine, "subject");
        let candidate = make_candidate(&function, &baseline(&function));
        let plan = selection(&candidate);
        let code = compile(&mut engine, &candidate, true);
        install(&mut engine, &function, &candidate, &code);
        COERCIONS.with(|count| count.set(0));
        clear_entries();
        let result=eval(&mut engine,&format!("throwing={throwing};var result;try{{result='return:'+subject(true);}}catch(e){{result='throw:'+e;}}result+'|'+trace.join(',')+'|'+[prefixes,gets,finishes].join(':')"));
        assert_eq!(
            result,
            if throwing {
                "throw:boom|P,G,F|1:1:1"
            } else {
                "return:9|P,G,F|1:1:1"
            }
        );
        let pc = assert_one_cut(&engine, &candidate, &code, &plan);
        let cfg = Cfg::build(&candidate).unwrap();
        let stack = stack::StackPlan::build(&candidate, &cfg).unwrap();
        assert!(
            stack.at[pc].known() > 0,
            "cut has a real produced owning operand"
        );
        assert_eq!(COERCIONS.with(Cell::get), usize::from(!throwing));
        if !throwing {
            assert!(LIVE_AT_COERCION.with(Cell::get));
        }
        let record = code.optimizing_diagnostics.as_ref().unwrap();
        assert_eq!(record.counters[0].get(), 1);
        assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 1);
        assert_eq!(
            record.counters[diagnostics::VM_RECOVERY_NORMAL].get(),
            u64::from(!throwing)
        );
        assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 1);
        assert_eq!(
            record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
            0
        );
        engine.interp.gc_collect();
        PAYLOADS.with(|payloads| {
            assert!(
                payloads
                    .borrow()
                    .iter()
                    .all(|payload| payload.strong_count() == 0),
                "same-call normal/throw cleanup releases all real operands"
            )
        });
    }
}

#[test]
fn native_throw_routes_cold_iterator_and_finalizer_landings_with_original_precedence() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        var trace=[],bad=false,mode=0;
        var iterable={ [Symbol.iterator](){return {
            next(){trace.push('N');return {done:false,value:undefined};},
            return(){trace.push('R');$262.gc();if(bad)throw 'closing';return {};}
        };}};
        function fail(){trace.push('V');if(bad)throw 'original';return 4;}
        function subject(){
            try {
                try {let [value=fail()]=iterable;return 'return:'+value;}
                finally {trace.push('I');$262.gc();if(mode===1)return 'replacement';if(mode===2)throw 'finally';}
            } catch(error){trace.push('C:'+error);return 'caught:'+error;}
            finally {trace.push('O');$262.gc();if(mode===3)return 'outer';}
        }
        subject();subject();trace=[];
    "#,
    );
    let function = lookup_function(&mut engine, "subject");
    let candidate = make_candidate(&function, &baseline(&function));
    let plan = selection(&candidate);
    let code = compile(&mut engine, &candidate, true);
    install(&mut engine, &function, &candidate, &code);
    let cfg = Cfg::build(&candidate).unwrap();
    for (mode, expected) in [
        (0, "caught:original|N,V,R,I,C:original,O"),
        (1, "replacement|N,V,R,I,O"),
        (2, "caught:finally|N,V,R,I,C:finally,O"),
        (3, "outer|N,V,R,I,C:original,O"),
    ] {
        clear_entries();
        assert_eq!(
            eval(
                &mut engine,
                &format!("bad=true;mode={mode};trace=[];subject()+'|'+trace.join(',')")
            ),
            expected
        );
        let pc = assert_one_cut(&engine, &candidate, &code, &plan);
        assert!(
            cfg.handler_roots()
                .iter()
                .any(|root| cfg.blocks()[root.target.0 as usize].start == pc),
            "native helper consumed operands then unwound to exact cold cleanup root"
        );
    }
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert_eq!(record.counters[0].get(), 4);
    assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 4);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 4);
    assert_eq!(record.counters[diagnostics::NORMAL_EXIT].get(), 4);
    assert_eq!(record.counters[diagnostics::ERROR_EXIT].get(), 0);
}

#[test]
fn cold_eval_preserves_prepared_reference_mapped_arguments_and_captured_scope() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        var item=1,effects=0,prefixes=0;
        function subject(cold,value){prefixes++;var read=()=>value;
            arguments[1]=5;
            item=(cold?eval('effects++;var item=99;value=7;6'):6);
            return [item,globalThis.item,value,read(),arguments[1]].join(':');
        }
        subject(false,2);subject(false,2);effects=prefixes=0;
    "#,
    );
    let function = lookup_function(&mut engine, "subject");
    let candidate = make_candidate(&function, &baseline(&function));
    assert!(candidate
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::ResolveNameRef(..))));
    assert!(candidate
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::StoreRef(_))));
    let plan = selection(&candidate);
    let code = compile(&mut engine, &candidate, true);
    install(&mut engine, &function, &candidate, &code);
    clear_entries();
    assert_eq!(
        eval(
            &mut engine,
            "subject(true,2)+'|'+[effects,prefixes].join(':')"
        ),
        "99:6:7:7:7|1:1"
    );
    assert_one_cut(&engine, &candidate, &code, &plan);
}

#[test]
fn loop_cut_keeps_same_iteration_environments_parameters_and_prefix_effects() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        var defaults=0,prefixes=0,trace=[];
        function subject(cutAt,value=(defaults++,3)) {prefixes++;var closures=[];
            try {for(let n=0;n<4;n++){closures.push(()=>n);
                if(n===cutAt){trace.push('cut');$262.gc();}
            }}finally{trace.push('finally');}
            return value+':'+closures.map(f=>f()).join(',');
        }
        subject(-1);subject(-1);defaults=prefixes=0;trace=[];
    "#,
    );
    let function = lookup_function(&mut engine, "subject");
    let candidate = make_candidate(&function, &baseline(&function));
    let plan = selection(&candidate);
    let code = compile(&mut engine, &candidate, true);
    install(&mut engine, &function, &candidate, &code);
    clear_entries();
    assert_eq!(
        eval(
            &mut engine,
            "subject(2)+'|'+trace.join(',')+'|'+[defaults,prefixes].join(':')"
        ),
        "3:0,1,2,3|cut,finally|1:1"
    );
    assert_one_cut(&engine, &candidate, &code, &plan);
}

#[test]
fn cold_loop_abrupt_completions_keep_all_original_labels_and_lexical_cleanup() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        var trace=[];
        function subject(action){
            try {outer:for(let n=0;n<3;n++){
                try {if(n===1&&action){
                    $262.gc();if(action===1)continue outer;if(action===2)break outer;
                    if(action===3)return 'R';throw 'T';
                }trace.push('B'+n);}
                finally {trace.push('I'+n);}
            }return 'E';}finally{trace.push('O');}
        }
        subject(0);subject(0);trace=[];
    "#,
    );
    let function = lookup_function(&mut engine, "subject");
    let candidate = make_candidate(&function, &baseline(&function));
    let plan = selection(&candidate);
    let cfg = Cfg::build(&candidate).unwrap();
    for root in cfg.handler_roots() {
        let pc = cfg.blocks()[root.target.0 as usize].start;
        assert_ne!(
            plan.emission(pc),
            observed_regions::Emission::OmittedInterior,
            "every dynamic handler pad remains native or an exact-PC cut"
        );
    }
    let abrupt: Vec<_> = candidate
        .jit_ops()
        .iter()
        .filter_map(|op| match *op {
            Op::AbruptJump(target, _) => Some(target as usize),
            _ => None,
        })
        .collect();
    assert!(!abrupt.is_empty(), "real labelled completion routing");
    for pc in abrupt {
        assert_ne!(
            plan.emission(pc),
            observed_regions::Emission::OmittedInterior
        );
    }
    let code = compile(&mut engine, &candidate, true);
    install(&mut engine, &function, &candidate, &code);
    for (action, expected) in [
        (1, "return:E|B0,I0,I1,B2,I2,O"),
        (2, "return:E|B0,I0,I1,O"),
        (3, "return:R|B0,I0,I1,O"),
        (4, "throw:T|B0,I0,I1,O"),
    ] {
        clear_entries();
        assert_eq!(eval(&mut engine,&format!("trace=[];var result;try{{result='return:'+subject({action});}}catch(e){{result='throw:'+e;}}result+'|'+trace.join(',')")),expected);
        assert_one_cut(&engine, &candidate, &code, &plan);
    }
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 4);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 4);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY_NORMAL].get(), 3);
    assert_eq!(record.counters[diagnostics::ERROR_EXIT].get(), 1);
}

#[test]
fn globally_unsupported_cold_disposal_suspension_and_fragment_tails_still_decline() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    for (source,warm,kind) in [
        ("function subject(o,cold){o.x=o.a;o.y=o.x;o.z=o.y;if(cold){using resource=null;}return o.z;}var o={a:1,x:0,y:0,z:0};", "subject(o,false);subject(o,false);", "disposal"),
        ("async function subject(cold){if(cold)await 1;return 3;}", "subject(false);subject(false);", "await"),
        ("function* subject(cold){if(cold)yield 1;return 3;}", "subject(false).next();subject(false).next();", "yield"),
    ] {
        let mut engine=Engine::new();engine.set_tier(Tier::Bytecode);engine.set_tier_threshold(0);
        eval(&mut engine,&format!("{source}{warm}"));
        let function=lookup_function(&mut engine,"subject");let primary=baseline(&function);
        let forbidden=primary.jit_ops().iter().enumerate().filter(|(_,op)|match kind {
            "disposal"=>matches!(op,Op::DisposeNormal|Op::DisposeThrow|Op::DisposeReturn|Op::DisposeBareReturn|Op::DisposeResumeReturn|Op::DisposeJump),
            "await"=>matches!(op,Op::Await),
            _=>matches!(op,Op::Yield),
        }).map(|(pc,_)|pc).collect::<Vec<_>>();
        assert!(!forbidden.is_empty(),"real {kind} opcode in cold original tail");
        let frozen=primary.execution_coverage.freeze();let snapshot=frozen.frozen_snapshot().unwrap();
        assert_eq!(snapshot.state_for(&primary,0),State::Entered);
        assert!(forbidden.iter().all(|&pc|snapshot.state_for(&primary,pc)==State::UnseenInWindow));
        let candidate=make_candidate(&function,&primary);
        let values=crate::value::jit_layout(&engine.interp.object_proto);let layout=crate::interpreter::interp_layout(&mut engine.interp);
        assert!(matches!(compile_with_deopt(&candidate,&values,&layout,None),Err(CompileError::Unsupported)),"{kind} cannot extend ordinary terminal recovery ABI");
        assert!(function.optimizing_code.get().is_none());
    }
    let mut engine = Engine::new();
    let parsed = crate::parser::parse_script("while(false){var n=1;} ", false)
        .ok()
        .unwrap();
    let crate::ast::Stmt::While { test, body } = &parsed[0] else {
        panic!("loop fixture")
    };
    let fragment = bytecode::loop_fragment::compile_for_region_exclusion_test(
        bytecode::loop_fragment::Source::While(test, &body.statement),
        &engine.interp,
    )
    .unwrap();
    assert!(fragment
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::FragmentExit(_))));
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    assert!(
        matches!(
            compile_with_deopt(&fragment, &values, &layout, None),
            Err(CompileError::Unsupported)
        ),
        "real borrowed fragment is never an ordinary region entry"
    );
}

#[test]
fn shared_exact_version_runs_cold_foreign_realm_and_preserves_caller_handler_floor() {
    use crate::realm_inline_guard_tests::{eval_string, in_realm, set_global};
    let _coverage = crate::execution_coverage::TestControl::new(true);
    const SOURCE: &str = r#"(function(){
        globalThis.trace=[];
        globalThis.subject=function subject(cold){trace.push('P');if(cold){$262.gc();return callback();}return [3];};
        globalThis.driver=function driver(cold){try{return subject(cold);}catch(e){trace.push('C');return e;}finally{trace.push('D');}};
    })();"#;
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot: &'static [u8] = SNAPSHOT.get_or_init(|| crate::compile_snapshot(SOURCE).unwrap());
    let run = |it: &mut Interp| {
        let program = it.shared_snapshot_program(snapshot).unwrap();
        assert!(it.run_classic_program(&program, None).is_ok());
    };
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    run(&mut engine.interp);
    eval(&mut engine, "driver(false);driver(false);trace=[];");
    let caller = lookup_function(&mut engine, "driver");
    let function = lookup_function(&mut engine, "subject");
    let primary = baseline(&function);
    let candidate = make_candidate(&function, &primary);
    let plan = selection(&candidate);
    let code = compile(&mut engine, &candidate, true);
    install(&mut engine, &function, &candidate, &code);
    let child = engine.interp.create_realm();
    let third = engine.interp.create_realm();
    in_realm(&mut engine.interp, &third, |it| {
        eval_string(
            it,
            "globalThis.foreign=function foreign(){ $262.gc();return null.missing; }; ",
        );
    });
    let callback = in_realm(&mut engine.interp, &third, |it| {
        let env = it.global_env.clone();
        it.get_var("foreign", &env).ok().unwrap()
    });
    in_realm(&mut engine.interp, &child, run);
    in_realm(&mut engine.interp, &child, |it| {
        set_global(it, "callback", callback);
        set_global(it, "foreignRealm", third.clone());
    });
    let child_function = in_realm(&mut engine.interp, &child, |it| {
        let env = it.global_env.clone();
        let callable = it.get_var("subject", &env).ok().unwrap();
        let object = callable.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("shared User")
        };
        user.func.clone()
    });
    assert!(Rc::ptr_eq(&function, &child_function));
    assert_eq!(
        candidate.execution_coverage.source_version(),
        primary.execution_coverage.source_version()
    );
    clear_entries();
    bytecode::native_deopt::FOREIGN_SIDECAR_ENTRIES.with(|count| count.set(0));
    let result = in_realm(&mut engine.interp, &child, |it| {
        eval_string(it,"var result=driver(true);[result instanceof foreignRealm.TypeError,!(result instanceof TypeError),trace.join(','),Object.getPrototypeOf(subject(false))===Array.prototype].join('|')")
    });
    assert_eq!(result, "true|true|P,C,D|true");
    let pc = assert_one_cut(&engine, &candidate, &code, &plan);
    ENTRIES.with(|entries| assert_eq!(entries.borrow()[0].2, 0,
        "every direct/checked call owns a separate frame record; caller handlers never enter the callee context"));
    let caller_code = baseline(&caller)
        .jit
        .get()
        .flatten()
        .expect("actual compiled template driver");
    assert!(!caller_code.pc_offsets.is_empty());
    assert!(caller_code.residency.referenced.get() > 0);
    assert_eq!(caller_code.residency.active.get(), 0);
    assert_eq!(
        bytecode::native_deopt::FOREIGN_SIDECAR_ENTRIES.with(Cell::get),
        0,
        "callee continuation cannot borrow its caller's sidecar"
    );
    assert!(candidate.jit_ops().get(pc).is_some());
}

thread_local! {
    static RELEASED:RefCell<Vec<Weak<RefCell<Object>>>>=const { RefCell::new(Vec::new()) };
    static RELEASE_COUNTS:RefCell<Vec<usize>>=const { RefCell::new(Vec::new()) };
}

fn check_released(i: &mut Interp, _: JsValue, _: &[JsValue]) -> Result<JsValue, JsValue> {
    i.gc_collect();
    RELEASE_COUNTS.with(|counts| {
        *counts.borrow_mut() =
            RELEASED.with(|owners| owners.borrow().iter().map(Weak::strong_count).collect())
    });
    Ok(JsValue::Undefined)
}

#[test]
fn inline_reset_clean_stale_word_is_not_revived_at_later_cold_cut() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let global = engine.interp.global.clone();
    engine
        .interp
        .def_method(&global, "checkReleased", 0, check_released);
    eval(
        &mut engine,
        r#"
        function make(secret){return function leaf(drop){if(drop){holder.f=null;return secret.tag.length+1;}return 2;};}
        var sentinel={tag:'x'},holder={f:make(sentinel)},receiver={a:3,b:5};
        function subject(o,drop,cold){var total=holder.f(drop);checkReleased();return total+(cold?o.a:o.b);}
    "#,
    );
    let function = lookup_function(&mut engine, "subject");
    let primary = bytecode::compile(&function).unwrap();
    primary.inline_attempted.set(true);
    primary.inline_retry_at.set(0);
    assert!(function.code.set(Some(primary.clone())).is_ok());
    for _ in 0..8 {
        eval(&mut engine, "subject(receiver,false,false)");
    }
    let callable = {
        let env = engine.interp.global_env.clone();
        engine.interp.get_var("subject", &env).ok().unwrap()
    };
    let realm = {
        let object = callable.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("User")
        };
        user.realm
    };
    drop(callable);
    let inline_plan = bytecode::plan_inlines(&primary, &function, realm, &engine.interp.global_env);
    assert!(!inline_plan.is_empty());
    let flattened = bytecode::compile_with_inlines(&function, &inline_plan, &primary).unwrap();
    drop(inline_plan);
    let callee = flattened
        .jit_ops()
        .iter()
        .find_map(|op| match *op {
            Op::InlineGuard(target, _) => Some(flattened.jit_inline_target(target).pin.clone()),
            _ => None,
        })
        .unwrap();
    let callee_code = {
        let callee = callee.upgrade().unwrap();
        let object = callee.borrow();
        let Callable::User(user) = &object.call else {
            panic!("leaf")
        };
        baseline(&user.func).jit.get().flatten().unwrap()
    };
    let payload = {
        let env = engine.interp.global_env.clone();
        let value = engine.interp.get_var("sentinel", &env).ok().unwrap();
        Rc::downgrade(value.as_obj().unwrap())
    };
    assert!(function.code2.set(Some(flattened.clone())).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Bytecode);
    // Observe both inline paths and the real ResetSlots without losing the live
    // callable used by its Weak identity guards. Release this temporary JS lease
    // before the tested invocation; it is not a native-code root.
    eval(&mut engine,"var saved=holder.f;subject(receiver,false,false);subject(receiver,true,false);holder.f=saved;saved=null;");
    let candidate = make_candidate(&function, &flattened);
    let plan = selection(&candidate);
    let cut = (0..candidate.jit_ops().len())
        .find(|&pc| {
            plan.emission(pc) == observed_regions::Emission::Cut
                && candidate.jit_ops()[pc..]
                    .iter()
                    .take(4)
                    .any(|op| match *op {
                        Op::GetPropLocal(_, name, _) | Op::GetProp(name, _) => {
                            candidate.jit_name(name) == "a"
                        }
                        _ => false,
                    })
        })
        .unwrap();
    let cfg = Cfg::build(&candidate).unwrap();
    let frame = frame::FramePlan::build(&candidate, &cfg, true).unwrap();
    assert!(candidate.jit_ops().iter().enumerate().any(|(reset,op)|matches!(*op,Op::ResetSlots(slot,1) if frame.tracked.contains(&slot)&&!frame.live_after[reset].contains(&slot)&&!frame.dirty_at[cut].contains(&slot))),"real stale clean inline register copy at cut");
    let code = compile(&mut engine, &candidate, true);
    install(&mut engine, &function, &candidate, &code);
    RELEASED.with(|owners| *owners.borrow_mut() = vec![callee.clone(), payload.clone()]);
    RELEASE_COUNTS.with(|counts| counts.borrow_mut().clear());
    callee_code.residency.referenced.set(0);
    eval(&mut engine, "sentinel=null;");
    clear_entries();
    assert_eq!(eval(&mut engine, "subject(receiver,true,true)"), "5");
    assert_eq!(assert_one_cut(&engine, &candidate, &code, &plan), cut);
    RELEASE_COUNTS.with(|counts| {
        assert_eq!(
            *counts.borrow(),
            vec![0, 0],
            "actual destruction before cold recovery, not eventual cleanup"
        )
    });
    assert_eq!(
        callee_code.residency.referenced.get(),
        0,
        "real flattened hit, not generic leaf fallback"
    );
    engine.interp.gc_collect();
    assert_eq!(callee.strong_count(), 0);
    assert_eq!(payload.strong_count(), 0);
}

fn hot_child(test: &str) -> bool {
    const FLAG: &str = "TRUST_JS_TEST_OBSERVED_REGION_HOT_CHILD";
    if std::env::var(FLAG).as_deref() == Ok(test) {
        return false;
    }
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("TRUST_JS") {
            command.env_remove(key);
        }
    }
    let output = command
        .args([
            "--exact",
            &format!("jit::optimizing::observed_region_tests::{test}"),
            "--nocapture",
            "--test-threads=1",
        ])
        .env(FLAG, test)
        .env("TRUST_JS_OPT_JIT", "hot")
        .env("TRUST_JS_OPT_JIT_BACKGROUND", "1")
        .env("TRUST_JS_OPT_JIT_COVERAGE", "1")
        .env("TRUST_JS_OPT_JIT_OBSERVED_REGIONS", "1")
        .env("TRUST_JS_OPT_JIT_GENERIC_OWN", "1")
        .env("TRUST_JS_OPT_JIT_SHARED_OWNERS", "1")
        .env("TRUST_JS_OPT_JIT_DIAGNOSTICS", "execution")
        .env("TRUST_JS_OPT_JIT_OSR", "0")
        .env("TRUST_JS_INLINE_AT", "0")
        .env("TRUST_JS_PERF_METRICS", "1")
        .output()
        .unwrap();
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
fn early_cold_cut_cannot_claim_structural_savings_in_the_vm_remainder() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let old = generic_own::TEST_ENABLED.with(|enabled| enabled.replace(Some(true)));
    struct Reset(Option<bool>);
    impl Drop for Reset {
        fn drop(&mut self) {
            generic_own::TEST_ENABLED.with(|enabled| enabled.set(self.0));
        }
    }
    let _reset = Reset(old);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        var o={source:1,x:0,a:0,b:0,c:0,d:0,e:0};
        function subject(o,cold){if(cold){var harmless=1;}
            o.x=o.source;o.a=o.x;o.b=o.a;o.c=o.b;o.d=o.c;o.e=o.d;return o.e;}
        subject(o,false);subject(o,false);
    "#,
    );
    let function = lookup_function(&mut engine, "subject");
    let primary = baseline(&function);
    let frozen = primary.execution_coverage.freeze();
    let cfg = Cfg::build(&primary).unwrap();
    let stack = stack::StackPlan::build(&primary, &cfg).unwrap();
    let observations = generic_admission::observations(&primary);
    let sites = generic_admission::selected(&observations);
    let full = generic_admission::Plan::build(&primary, &cfg, &stack, &sites)
        .expect("real structural savings in whole native body");
    assert!(full.minimum_tests > 0);
    let regions = observed_regions::Plan::build(&primary, &cfg, &stack, &frozen).unwrap();
    assert!(
        generic_admission::Plan::build_with_regions(&primary, &cfg, &stack, &sites, Some(&regions))
            .is_none(),
        "earliest VM cut saves zero of the later checks"
    );
    assert!(regions
        .minimum_saved_tests(&primary, &cfg, None, Some(&full), None)
        .is_none());
    let candidate = bytecode::compile_for_optimizer_with_structural_and_coverage(
        &function,
        &primary,
        &[],
        &[],
        &sites,
        &frozen,
    )
    .unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    assert!(
        matches!(
            compile_with_deopt(&candidate, &values, &layout, None),
            Err(CompileError::Unsupported)
        ),
        "candidate must recheck positive cut-aware benefit before publication"
    );
    assert!(function.optimizing_code.get().is_none());
    // This is the exact immutable checkpoint snapshot copied through the actual
    // production API, not an independently recaptured candidate observation.
    assert!(std::ptr::eq(
        frozen.frozen_snapshot().unwrap(),
        candidate.execution_coverage.frozen_snapshot().unwrap()
    ));
}

#[test]
fn numeric_only_early_cut_cannot_charge_later_arithmetic_as_native_benefit() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine,"function subject(value,cold){if(cold){var harmless=1;}return value+value+value+value+value+value;}subject(3,null);subject(3,null);");
    let function = lookup_function(&mut engine, "subject");
    let primary = baseline(&function);
    let frozen = primary.execution_coverage.freeze();
    let cfg = Cfg::build(&primary).unwrap();
    let stack = stack::StackPlan::build(&primary, &cfg).unwrap();
    // Null control is unclassified; only arithmetic gets a speculative fact.
    // A Boolean control would genuinely save checks before the cut.
    let inputs = [(0, crate::feedback::ValueClass::NumberInt32.bit())];
    let numeric = specialization::Plan::build(&primary, &cfg, &stack, &inputs, &[]).unwrap();
    assert!(
        numeric.saved_tests > 2,
        "whole-body static numeric benefit exists"
    );
    let regions = observed_regions::Plan::build(&primary, &cfg, &stack, &frozen).unwrap();
    assert!(
        regions
            .minimum_saved_tests(&primary, &cfg, Some(&numeric), None, None)
            .is_none(),
        "formal guards cannot be paid by arithmetic in the VM remainder"
    );
    let candidate = bytecode::compile_for_optimizer_with_structural_and_coverage(
        &function,
        &primary,
        &inputs,
        &[],
        &[],
        &frozen,
    )
    .unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    assert!(matches!(
        compile_with_deopt(&candidate, &values, &layout, None),
        Err(CompileError::Unsupported)
    ));
    assert!(function.optimizing_code.get().is_none());
}

#[test]
fn cold_inline_body_keeps_logical_error_frames_and_utf16_source_columns() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    const SOURCE:&str="function leaf(cold){var prefix='😀';if(cold){$262.gc();throw new Error('boom');}return prefix.length;}\nfunction subject(cold){return leaf(cold);}";
    const OBSERVATION: &str = "var stack;try{subject(true)}catch(e){stack=e.stack}stack";
    let origin = crate::SourceOrigin::new("observed-inline.js");
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    assert!(matches!(
        oracle
            .eval_interruptible_with_origin(SOURCE, false, Some(&origin))
            .unwrap(),
        crate::ExecutionOutcome::Value(_)
    ));
    let expected = eval(&mut oracle, OBSERVATION);
    let column = SOURCE[..SOURCE.find("new Error").unwrap()]
        .encode_utf16()
        .count()
        + 1;
    assert!(
        expected.contains(&format!("leaf (observed-inline.js:1:{column})")),
        "oracle matches project UTF16 position contract: {expected}"
    );
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert!(matches!(
        engine
            .eval_interruptible_with_origin(SOURCE, false, Some(&origin))
            .unwrap(),
        crate::ExecutionOutcome::Value(_)
    ));
    let function = lookup_function(&mut engine, "subject");
    let primary = bytecode::compile(&function).unwrap();
    primary.inline_attempted.set(true);
    primary.inline_retry_at.set(0);
    assert!(function.code.set(Some(primary.clone())).is_ok());
    for _ in 0..8 {
        assert_eq!(eval(&mut engine, "subject(false)"), "2");
    }
    let realm = {
        let env = engine.interp.global_env.clone();
        let value = engine.interp.get_var("subject", &env).ok().unwrap();
        let object = value.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("User")
        };
        user.realm
    };
    let inline_plan = bytecode::plan_inlines(&primary, &function, realm, &engine.interp.global_env);
    assert!(!inline_plan.is_empty());
    let flattened = bytecode::compile_with_inlines(&function, &inline_plan, &primary).unwrap();
    drop(inline_plan);
    let leaf_code = {
        let env = engine.interp.global_env.clone();
        let value = engine.interp.get_var("leaf", &env).ok().unwrap();
        let object = value.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("leaf")
        };
        baseline(&user.func).jit.get().flatten().unwrap()
    };
    assert!(function.code2.set(Some(flattened.clone())).is_ok());
    bytecode::invalidate_call_caches();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, "subject(false);subject(false);");
    let candidate = make_candidate(&function, &flattened);
    let plan = selection(&candidate);
    let code = compile(&mut engine, &candidate, true);
    install(&mut engine, &function, &candidate, &code);
    leaf_code.residency.referenced.set(0);
    clear_entries();
    assert_eq!(
        eval(&mut engine, OBSERVATION),
        expected,
        "VM continuation restores original inline marker/cleanup and exact source positions"
    );
    let pc = assert_one_cut(&engine, &candidate, &code, &plan);
    assert!(
        crate::stack_trace::is_inline_position(candidate.position(pc)),
        "actual cut in the logical callee body"
    );
    assert_eq!(
        leaf_code.residency.referenced.get(),
        0,
        "native guarded splice hit before VM cut, not an ordinary leaf call"
    );
}

#[test]
fn real_hot_background_publication_uses_frozen_regions_and_keeps_the_256_op_gate() {
    if hot_child("real_hot_background_publication_uses_frozen_regions_and_keeps_the_256_op_gate") {
        return;
    }
    assert!(mode() == Mode::Hot && background::enabled() && observed_regions::enabled());
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        r#"
        var payload={tag:'alive'},receiver={source:payload,x:0,a:0,b:0,c:0,d:0,e:0},finishes=0;
        function finish(o){finishes++;return o.e.tag;}
        function coldFinish(o){finishes++;$262.gc();return 'cold:'+o.e.tag;}
        function subject(o,cold){o.x=o.source;o.a=o.x;o.b=o.a;o.c=o.b;o.d=o.c;o.e=o.d;
            if(cold){for(var j=0;j<3;j++)o.e=o.d;return coldFinish(o);}return finish(o);}
        subject(receiver,false);subject(receiver,false);
    "#,
    );
    let payload = {
        let env = engine.interp.global_env.clone();
        let value = engine.interp.get_var("payload", &env).ok().unwrap();
        Rc::downgrade(value.as_obj().unwrap())
    };
    let function = lookup_function(&mut engine, "subject");
    let primary = baseline(&function);
    let cfg = Cfg::build(&primary).unwrap();
    let stack = stack::StackPlan::build(&primary, &cfg).unwrap();
    let frozen = primary.execution_coverage.freeze();
    let regions = observed_regions::Plan::build(&primary, &cfg, &stack, &frozen).unwrap();
    let sites = generic_admission::selected(&generic_admission::observations(&primary));
    let structural =
        generic_admission::Plan::build_with_regions(&primary, &cfg, &stack, &sites, Some(&regions))
            .unwrap();
    let boolean = crate::feedback::ValueClass::Boolean.bit();
    let numeric = specialization::Plan::build(&primary, &cfg, &stack, &[(1, boolean)], &[]);
    let saved = regions
        .minimum_saved_tests(&primary, &cfg, numeric.as_ref(), Some(&structural), None)
        .unwrap();
    let distance = generic_admission::admission_entries(&primary, saved, 0)
        .unwrap()
        .max(4096);
    assert!(distance < 200_000 && std::env::var_os("TRUST_JS_OPT_JIT_HOT_AT").is_none());
    engine.set_tier(Tier::Jit);
    eval(
        &mut engine,
        &format!(
            "for(var n=0;n<{};n++)subject(receiver,false);",
            distance + 1024
        ),
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while function.optimizing_code.get().is_none()
        && engine
            .interp
            .optimizing_background
            .as_ref()
            .is_some_and(|state| state.contains(&primary))
    {
        engine.interp.optimizing_background.as_mut().unwrap().poll();
        assert!(
            std::time::Instant::now() < deadline,
            "worker finishes or explicitly declines, never hangs"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let candidate=function.optimizing_code.get().and_then(Option::as_ref).expect("actual checkpoint submission and successful background native publication within unchanged cap").clone();
    let plan = selection(&candidate);
    let code = candidate.jit.get().flatten().unwrap();
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    let limit =
        (candidate.jit_ops().len() * 40 + names::instruction_allowance(&candidate)).min(16_384);
    assert!(record.ir_instructions.get() > 0 && record.ir_instructions.get() <= limit);
    assert!(code.pc_offsets.is_empty() && record.execution_counters && !record.runtime_counters);
    code.residency.referenced.set(0);
    clear_entries();
    assert_eq!(eval(&mut engine, "subject(receiver,true)"), "cold:alive");
    assert_one_cut(&engine, &candidate, &code, &plan);
    assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 1);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 1);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY_NORMAL].get(), 1);
    assert!(!candidate.optimizing_structural.is_empty());
    assert_eq!(
        candidate.execution_coverage.source_version(),
        primary.execution_coverage.source_version()
    );
    let misses = candidate.optimizing_misses.get();
    for _ in 0..12 {
        assert_eq!(eval(&mut engine, "subject(receiver,true)"), "cold:alive");
    }
    assert_eq!(
        candidate.optimizing_misses.get(),
        misses,
        "more than eight ordinary cold exits never retire code"
    );
    let first_read = candidate
        .jit_ops()
        .iter()
        .position(|op| match *op {
            Op::GetPropLocal(_, name, _) => candidate.jit_name(name) == "source",
            _ => false,
        })
        .unwrap();
    assert_eq!(
        plan.emission(first_read),
        observed_regions::Emission::Native
    );
    eval(&mut engine,"var getterCalls=0;Object.defineProperty(receiver,'source',{configurable:true,get(){getterCalls++;$262.gc();return payload;}});");
    for attempt in 1..=8 {
        clear_entries();
        assert_eq!(eval(&mut engine, "subject(receiver,true)"), "cold:alive");
        ENTRIES.with(|entries| {
            assert_eq!(entries.borrow().len(),1);
            assert_eq!(entries.borrow()[0].0,first_read,"live selected guard misses before the ordinary region cut; VM remainder executes once");
        });
        assert_eq!(candidate.optimizing_misses.get(), attempt);
    }
    code.residency.referenced.set(0);
    assert_eq!(
        eval(&mut engine, "subject(receiver,true)+'|'+getterCalls"),
        "cold:alive|9"
    );
    assert_eq!(
        code.residency.referenced.get(),
        0,
        "the ninth guarded miss uses the established tier after retirement"
    );
    eval(&mut engine, "payload=receiver=null;");
    engine.interp.gc_collect();
    assert_eq!(
        payload.strong_count(),
        0,
        "worker/native code/frozen coverage own no JS payload"
    );
    let cold_body = (0..160).map(|_| "o.x=o.source;").collect::<String>();
    eval(&mut engine,&format!("function large(o,cold){{if(cold){{{cold_body}}}return o.source;}}var largeReceiver={{x:0,source:1}};large(largeReceiver,false);"));
    let large = baseline(&lookup_function(&mut engine, "large"));
    assert!(large.jit_ops().len() > 256);
    assert!(
        hot_counter(&large).is_none(),
        "observed-region stage does not change whole-body hot admission policy"
    );
    eprintln!("observed-region actual hot: ops={} selected={} cuts={} IR={} cap={limit} distance={distance} saved={saved}",candidate.jit_ops().len(),plan.native_ops,plan.cut_blocks,record.ir_instructions.get());
}

#[test]
fn object_branch_and_cold_cut_compose_credit_guards_and_same_call_effects() {
    let _coverage = crate::execution_coverage::TestControl::new(true);
    let _regions = observed_regions::TestControl::new(true);
    struct Enabled(Option<bool>, Option<bool>);
    impl Drop for Enabled {
        fn drop(&mut self) {
            generic_own::TEST_ENABLED.with(|enabled| enabled.set(self.0));
            object_facts::TEST_ENABLED.with(|enabled| enabled.set(self.1));
        }
    }
    let _enabled = Enabled(
        generic_own::TEST_ENABLED.with(|enabled| enabled.replace(Some(true))),
        object_facts::TEST_ENABLED.with(|enabled| enabled.replace(Some(true))),
    );
    assert!(
        object_facts::enabled(),
        "fixture requires live Object guards"
    );
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    let collections = Rc::new(Cell::new(0usize));
    let counted = collections.clone();
    let collector = engine.interp.make_native_closure(
        "collectCold",
        0,
        Rc::new(move |interp, _, _| {
            interp.gc_collect();
            counted.set(counted.get() + 1);
            Ok(JsValue::Undefined)
        }),
    );
    engine.interp.global.borrow_mut().props.insert(
        "collectCold",
        Property::data(JsValue::Obj(collector), true, true, true),
    );
    eval(
        &mut engine,
        r#"
        var effects=[],receiver={alternate:{alive:1},source:'live',x:0,a:0,b:0,c:0,d:0,e:0};
        function subject(o,cold){
            var n=o.alternate;
            if(null===n){effects.push('N');return 'null';}
            o.x=o.source;o.a=o.x;o.b=o.a;o.c=o.b;o.d=o.c;o.e=o.d;
            if(cold){for(var k=0;k<1;k++){}effects.push('C');collectCold();return 'cold:'+o.e;}
            return o.e;
        }
        subject(receiver,false);subject(receiver,false);
    "#,
    );
    let function = lookup_function(&mut engine, "subject");
    let primary = baseline(&function);
    let frozen = primary.execution_coverage.freeze();
    let cfg = Cfg::build(&primary).unwrap();
    let stack = stack::StackPlan::build(&primary, &cfg).unwrap();
    let regions = observed_regions::Plan::build(&primary, &cfg, &stack, &frozen)
        .expect("actual unseen cold loop and null arm");
    let sites = generic_admission::selected(&generic_admission::observations(&primary))
        .into_iter()
        .filter(|&(pc, _)| regions.emission(pc as usize) == observed_regions::Emission::Native)
        .collect::<Vec<_>>();
    let objects = sites
        .iter()
        .filter(|&&(pc, _)| {
            matches!(primary.jit_ops()[pc as usize], Op::GetPropLocal(_, name, _)
                if primary.jit_name(name) == "alternate")
        })
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(objects.len(), 1);
    let object = object_facts::Plan::build(&primary, &cfg, &stack, &objects, &sites, &[], &[])
        .expect("own-data Object result proves only strict-null edge");
    assert_eq!(object.guard_cost, 1);
    assert!(
        generic_admission::Plan::build_with_regions(&primary, &cfg, &stack, &sites, Some(&regions))
            .is_none(),
        "the unpruned null cut has no prefix credit"
    );
    let structural = generic_admission::Plan::with_objects_and_regions(
        &primary,
        &cfg,
        &stack,
        &sites,
        Some(&object),
        Some(&regions),
    )
    .expect("Object pruning preserves profitable copies before the later cut");
    let mut weights = vec![0u32; cfg.blocks().len()];
    for (id, block) in cfg.blocks().iter().enumerate() {
        for pc in block.start..block.end {
            if regions.emission(pc) == observed_regions::Emission::Native {
                weights[id] += structural.per_pc[pc].unwrap_or(0);
            } else {
                assert_eq!(structural.per_pc[pc], None, "no credit for VM tail");
            }
        }
    }
    let raw = generic_admission::minimum_path(
        &cfg,
        &primary,
        &weights,
        &structural.per_pc,
        &[],
        Some(&object),
        Some(&regions),
    )
    .unwrap();
    let saved = regions
        .minimum_saved_tests(&primary, &cfg, None, Some(&structural), Some(&object))
        .unwrap();
    assert_eq!(saved, raw.checked_sub(1).unwrap(), "one live result guard");
    assert_eq!(saved, structural.minimum_tests, "no second debit");
    assert!(saved > 0);

    let mut candidate = bytecode::compile_for_optimizer_with_objects_and_coverage(
        &function,
        &primary,
        &[],
        &[],
        &structural.selected,
        &objects,
        &frozen,
    )
    .unwrap();
    // Prove combined emission/recovery without asserting that this forced
    // fixture meets the independent normal hot policy's compilation estimate.
    Rc::get_mut(&mut candidate).unwrap().optimizing_candidate = false;
    assert!(std::ptr::eq(
        frozen.frozen_snapshot().unwrap(),
        candidate.execution_coverage.frozen_snapshot().unwrap()
    ));
    object_facts::TEST_EMITTED.with(|emitted| emitted.set([0; 3]));
    let code = compile(&mut engine, &candidate, true);
    assert!(object_facts::TEST_EMITTED.with(|emitted| emitted.get().iter().all(|&count| count > 0)));
    let plan = selection(&candidate);
    install(&mut engine, &function, &candidate, &code);
    clear_entries();
    code.residency.referenced.set(0);
    assert_eq!(
        eval(&mut engine, "subject(receiver,false)+'|'+effects.join(',')"),
        "live|"
    );
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(collections.get(), 0);
    ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    clear_entries();
    code.residency.referenced.set(0);
    assert_eq!(
        eval(&mut engine, "subject(receiver,true)+'|'+effects.join(',')"),
        "cold:live|C"
    );
    assert_eq!(
        collections.get(),
        1,
        "the cold VM effect and real GC run once"
    );
    assert_one_cut(&engine, &candidate, &code, &plan);
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 1);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 1);
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        0
    );

    clear_entries();
    code.residency.referenced.set(0);
    assert_eq!(
        eval(
            &mut engine,
            "effects=[];receiver.alternate=null;subject(receiver,false)+'|'+effects.join(',')"
        ),
        "null|N"
    );
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(collections.get(), 1);
    assert_eq!(candidate.optimizing_misses.get(), 1);
    ENTRIES.with(|entries| {
        let entries = entries.borrow();
        assert_eq!(entries.len(), 1);
        assert_eq!(
            entries[0].0, objects[0].0 as usize,
            "rescue BEFORE the Object field read"
        );
        assert_eq!(entries[0].1, stack.at[objects[0].0 as usize].depth);
    });
    assert_eq!(record.counters[diagnostics::OBSERVED_REGION_EXIT].get(), 1);
    assert_eq!(record.counters[diagnostics::VM_RECOVERY].get(), 2);
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        1
    );
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
}
