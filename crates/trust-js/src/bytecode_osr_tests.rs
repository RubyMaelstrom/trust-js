use super::*;

fn operand_owner_loop(i: &mut Interp) -> (Rc<Chunk>, Value) {
    let body =
        crate::parser::parse_script("let n=0; for(let k=0;k<2000;k++){n+=k;n+=1;} n;", false)
            .ok()
            .expect("metadata fixture parses");
    let mut chunk = compile_script(&body, false).expect("metadata fixture compiles");
    let owner = Value::Obj(crate::value::Object::new(Some(i.object_proto.clone())));
    let raw = Rc::get_mut(&mut chunk).unwrap();
    // Keep one owned operand below the loop condition. Every map must retain it;
    // the conditional edge has already popped its boolean before OSR admission.
    let base = raw.consts.len() as u32;
    raw.consts.extend([
        owner.clone(),
        Value::Num(0.0),
        Value::Num(2000.0),
        Value::Num(1.0),
    ]);
    assert!(raw.ops.len() >= 13); // existing feedback/caches cover these PCs
    raw.ops = vec![
        Op::Const(base),
        Op::Const(base + 1),
        Op::StoreLocal(0),
        Op::LoadLocal(0),
        Op::Const(base + 2),
        Op::Lt,
        Op::JumpIfFalse(12),
        Op::LoadLocal(0),
        Op::Const(base + 3),
        Op::Add,
        Op::StoreLocal(0),
        Op::Jump(3),
        Op::Return,
    ];
    // This fixture replaces the completed operation stream rather than using the compiler's
    // finalization path. Keep its derived immutable-body facts consistent with that replacement.
    raw.jit_needs_activation_state = raw.ops.iter().any(jit_bridge_op);
    raw.has_tail_calls = false;
    (chunk, owner)
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn osr_preserves_nonempty_operand_prefix_and_rejects_wrong_initial_depth() {
    let mut engine = crate::Engine::new();
    engine.set_tier(Tier::Jit);
    let (chunk, owner) = operand_owner_loop(&mut engine.interp);
    assert!(script_osr_code(&mut engine.interp, &chunk, 3, 2, None)
        .ok()
        .unwrap()
        .is_none());
    let code = script_osr_code(&mut engine.interp, &chunk, 3, 1, None)
        .ok()
        .unwrap()
        .unwrap();
    assert_eq!(code.osr_entry_depth(3), Some(1));
    assert!(
        chunk.jit.get().is_none(),
        "ordinary CallIc cache remains separate"
    );
    let env = engine.interp.global_env.clone();
    crate::jit::TEST_OSR_ENTRIES.with(|count| count.set(0));
    let result = run_tiered_script(&mut engine.interp, &chunk, &env, Value::Undefined)
        .ok()
        .expect("borrowed operand loop completes");
    let (Value::Obj(actual), Value::Obj(expected)) = (result, owner) else {
        panic!("owned objects");
    };
    assert!(Rc::ptr_eq(&actual, &expected));
    assert!(crate::jit::TEST_OSR_ENTRIES.with(|count| count.get()) > 0);
}

#[test]
fn osr_compile_unavailable_keeps_exact_baseline_state_without_replay() {
    let mut engine = crate::Engine::new();
    engine.set_tier(Tier::Jit);
    let (chunk, owner) = operand_owner_loop(&mut engine.interp);
    // A language-resumable chunk is deliberately ineligible for Script OSR. This
    // fixture only uses ordinary ops, so the VM still completes the same loop.
    let mut chunk = chunk;
    Rc::get_mut(&mut chunk).unwrap().resumable = true;
    let state = chunk
        .osr
        .get_or_init(|| Box::new(crate::tiering::OsrCodeState::default()));
    assert!(state.code(&mut engine.interp, &chunk).is_none());
    assert!(state.unavailable());
    Rc::get_mut(&mut chunk).unwrap().resumable = false;
    let env = engine.interp.global_env.clone();
    crate::jit::TEST_OSR_ENTRIES.with(|count| count.set(0));
    let result = run_tiered_script(&mut engine.interp, &chunk, &env, Value::Undefined)
        .ok()
        .expect("unavailable native code stays bytecode");
    let (Value::Obj(actual), Value::Obj(expected)) = (result, owner) else {
        panic!("owned objects");
    };
    assert!(Rc::ptr_eq(&actual, &expected));
    assert_eq!(crate::jit::TEST_OSR_ENTRIES.with(|count| count.get()), 0);
}
