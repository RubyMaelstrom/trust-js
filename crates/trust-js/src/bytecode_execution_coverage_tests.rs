//! Actual before-operation and middle-PC entries use canonical VM-owned state.
use super::*;
use crate::execution_coverage::{Handle, State, TestControl};
use crate::{Engine, InterruptReason};
use std::cell::Cell;

thread_local! {
    static CALLBACKS: Cell<usize> = const { Cell::new(0) };
    static BEFORE_CALLBACK: std::cell::RefCell<Option<(std::rc::Weak<Chunk>, usize)>> = const { std::cell::RefCell::new(None) };
}

fn function(source: &str) -> Rc<Function> {
    let statements = crate::parser::parse_script(source, false).ok().unwrap();
    let Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function fixture")
    };
    function.clone()
}

fn status(chunk: &Chunk, pc: usize) -> State {
    let frozen = chunk.execution_coverage.freeze();
    frozen.frozen_snapshot().unwrap().state_for(chunk, pc)
}

fn prefix(i: &mut Interp, state: &mut VmCoro, stop: usize) {
    assert!(matches!(
        run_vm(
            i,
            &state.chunk,
            &mut state.env,
            &state.cap_env,
            &mut state.references,
            &state.slots,
            &mut state.stack,
            &mut state.pc,
            &state.this_val,
            &mut state.handlers,
            &mut state.disposal_frames,
            &mut state.class_states,
            Some(stop),
            None,
        ),
        Ok(VmRunExit::Step(VmStep::Done(Value::Undefined)))
    ));
    assert_eq!(state.pc, stop);
}

#[test]
fn vm_stop_before_operation_does_not_record_that_entry() {
    let _control = TestControl::new(true);
    let mut engine = Engine::new();
    let function = function("function* subject(a){if(a)yield 7;else yield 11;}");
    let chunk = compile(&function).unwrap();
    let env = engine.interp.global_env.clone();
    let mut state = VmCoro::new_generator(
        &mut engine.interp,
        chunk,
        env,
        Value::Undefined,
        &[Value::Bool(true)],
        &[],
    );
    prefix(&mut engine.interp, &mut state, 0);
    let coverage = state.chunk.execution_coverage.source().unwrap().clone();
    assert_eq!(status(&state.chunk, 0), State::Unknown);
    assert_eq!(coverage.markers(&[(0, 1)]).unwrap().len(), 1);
    let branch = state
        .chunk
        .ops
        .iter()
        .position(|op| matches!(op, Op::JumpIfFalse(_)))
        .unwrap();
    let target = branch + 1;
    prefix(&mut engine.interp, &mut state, target);
    let frozen = state.chunk.execution_coverage.freeze();
    let frozen = frozen.frozen_snapshot().unwrap();
    assert_eq!(frozen.state_for(&state.chunk, 0), State::Entered);
    assert_eq!(
        frozen.state_for(&state.chunk, target),
        State::UnseenInWindow,
        "the stop check precedes its new block's marker"
    );
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn borrowed_native_nonleader_entry_marks_before_effect_and_preentry_interrupt_does_not() {
    let _control = TestControl::new(true);
    let mut engine = Engine::new();
    let initial_calls = CALLBACKS.with(Cell::get);
    engine.interp.def_method(
        &engine.interp.global,
        "coverageCallback",
        1,
        |_, _, args| {
            BEFORE_CALLBACK.with(|probe| {
                let probe = probe.borrow();
                let (chunk, pc) = probe.as_ref().expect("native callback probe installed");
                let chunk = chunk.upgrade().expect("live continuation Chunk");
                assert_eq!(
                    status(&chunk, *pc),
                    State::Entered,
                    "native resume marker must precede the actual callback effect"
                );
                assert_eq!(
                    crate::execution_coverage::test_vm_marks(&chunk),
                    0,
                    "the later Yield helper cannot manufacture this positive entry"
                );
            });
            CALLBACKS.with(|counter| counter.set(counter.get() + 1));
            Ok(Value::Num(args[0].as_num_opt().unwrap() + 1.0))
        },
    );
    let env = engine.interp.global_env.clone();
    let callback = engine
        .interp
        .get_var("coverageCallback", &env)
        .ok()
        .unwrap();
    // Prepare authoritative owners with the VM outside the fresh collection window. No graph
    // mutation occurs: the only test-only replacement is a new profile on this exact version.
    let mut state = {
        let _off = TestControl::new(false);
        let function =
            function("function* subject(callback){var x=7;var y=callback(x);yield y;return y;}");
        let chunk = compile(&function).unwrap();
        let mut state = VmCoro::new_generator(
            &mut engine.interp,
            chunk,
            env,
            Value::Undefined,
            &[callback],
            &[],
        );
        let pc = state
            .chunk
            .ops
            .iter()
            .position(|op| matches!(op, Op::Call(..) | Op::CallWithThis(..)))
            .unwrap();
        assert!(
            !crate::jit_ir::semantic_block_starts(&state.chunk.ops)
                .unwrap()
                .contains(&pc),
            "fixture must resume at a genuine nonleader, rather than Yield+1"
        );
        prefix(&mut engine.interp, &mut state, pc);
        state
    };
    {
        let chunk = Rc::get_mut(&mut state.chunk).expect("only continuation owns test version");
        chunk.execution_coverage = Handle::new(chunk);
    }
    let chunk = state.chunk.clone();
    let middle = state.pc;
    BEFORE_CALLBACK.with(|probe| *probe.borrow_mut() = Some((Rc::downgrade(&chunk), middle)));
    assert_eq!(status(&chunk, middle), State::Unknown);
    let code =
        crate::jit::continuation_code(&mut engine.interp, &chunk).expect("borrowed native fixture");
    assert_eq!(
        status(&chunk, middle),
        State::Unknown,
        "compilation cannot mark entry"
    );
    let mut native = NativeContinuation {
        chunk: &*chunk,
        env: &mut state.env,
        cap_env: &state.cap_env,
        references: &mut state.references[..],
        stack: &mut state.stack,
        disposal_frames: &mut state.disposal_frames,
        class_states: &mut state.class_states[..],
    };
    let before = crate::jit::TEST_NATIVE_SLICES.with(Cell::get);
    engine.interrupt_handle().request_user_navigation();
    let interrupted = unsafe {
        crate::jit::run_continuation(
            &mut engine.interp,
            &chunk,
            &code,
            &mut native,
            &state.slots,
            &mut state.pc,
            &state.this_val,
            &mut state.handlers,
        )
    };
    assert!(matches!(
        interrupted,
        Err(Abrupt::Interrupt(InterruptReason::UserNavigation))
    ));
    assert_eq!(CALLBACKS.with(Cell::get), initial_calls);
    assert_eq!(crate::jit::TEST_NATIVE_SLICES.with(Cell::get), before);
    assert_eq!(status(&chunk, middle), State::Unknown);
    engine.interrupt_handle().begin_user_interaction();
    let result = unsafe {
        crate::jit::run_continuation(
            &mut engine.interp,
            &chunk,
            &code,
            &mut native,
            &state.slots,
            &mut state.pc,
            &state.this_val,
            &mut state.handlers,
        )
    };
    assert!(matches!(result, Ok(VmStep::Yield(Value::Num(8.0)))));
    assert_eq!(
        CALLBACKS.with(Cell::get),
        initial_calls + 1,
        "prepared prefix is not replayed"
    );
    assert_eq!(crate::jit::TEST_NATIVE_SLICES.with(Cell::get), before + 1);
    assert_eq!(status(&chunk, middle), State::Entered);
    assert_eq!(code.test_residency_marks(false), (0, 1));
    BEFORE_CALLBACK.with(|probe| probe.borrow_mut().take());
}

#[test]
fn exact_operation_and_block_layout_limits_fail_closed() {
    let _control = TestControl::new(true);
    let function = function("function subject(){return;}");
    let mut chunk = compile(&function).unwrap();
    let chunk = Rc::get_mut(&mut chunk).unwrap();
    // Full valid stack-neutral graph, finalized before its new source identity is allocated.
    chunk.ops = (0..32_767)
        .flat_map(|_| [Op::Undef, Op::Pop])
        .chain([Op::Undef, Op::Return])
        .collect();
    chunk.execution_coverage = Handle::new(chunk);
    assert_eq!(chunk.ops.len(), 65_536);
    assert!(chunk.execution_coverage.source().is_some());
    chunk.ops.insert(0, Op::Undef);
    chunk.execution_coverage = Handle::new(chunk);
    assert!(matches!(
        chunk.execution_coverage,
        Handle::Unavailable(crate::execution_coverage::Reason::Layout)
    ));
    // Reachable jump-only blocks, with an ordinary terminal return.
    chunk.ops = (1..8_192).map(Op::Jump).chain([Op::ReturnUndef]).collect();
    chunk.execution_coverage = Handle::new(chunk);
    assert!(
        chunk.execution_coverage.source().is_some(),
        "exactly 8192 blocks are representable"
    );
    chunk.ops = (1..8_193).map(Op::Jump).chain([Op::ReturnUndef]).collect();
    chunk.execution_coverage = Handle::new(chunk);
    assert!(matches!(
        chunk.execution_coverage,
        Handle::Unavailable(crate::execution_coverage::Reason::Layout)
    ));
}
