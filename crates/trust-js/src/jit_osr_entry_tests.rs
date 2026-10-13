//! Borrowed native entry is an implementation tier transition, not a fresh ECMAScript call.
//! ECMA-262 Execution Contexts / ScriptEvaluation, local snapshot e28783d5 (2026-09-06).

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn native_osr_entry_kind_maps_and_language_resumability_are_separate() {
    use super::{JitCompileOutcome, NativeEntryKind};
    let mut engine = crate::Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let interpreter_layout = crate::interpreter::interp_layout(&mut engine.interp);
    let source = "function run(n){let sum=0;for(let k=0;k<n;k++)sum+=k;return sum;}";
    let statements = crate::parser::parse_script(source, false).ok().unwrap();
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function")
    };
    let chunk = crate::bytecode::compile(function).expect("ordinary loop compiles");
    let ordinary =
        super::compile(&chunk, &layout, &interpreter_layout).expect("ordinary native code");
    let JitCompileOutcome::Compiled(borrowed) =
        super::compile_borrowed_profiled(&chunk, &layout, &interpreter_layout)
    else {
        panic!("borrowed native loop code")
    };
    assert_eq!(ordinary.entry_kind, NativeEntryKind::FreshFrame);
    assert_eq!(borrowed.entry_kind, NativeEntryKind::BorrowedFrame);
    assert!(ordinary.osr_entry_depths.is_empty());
    assert!(
        !chunk.jit_is_resumable(),
        "OSR never changes language suspension eligibility"
    );
    let cfg = crate::jit_ir::Cfg::build_osr(&chunk).unwrap();
    let mut headers = 0;
    for pc in 0..chunk.jit_ops().len() {
        assert_eq!(borrowed.osr_entry_depth(pc), cfg.osr_entry_depth(pc));
        if borrowed.osr_entry_depth(pc).is_some() {
            headers += 1;
            assert!(borrowed.pc_offsets[pc] < borrowed.len as u32);
            assert_eq!(borrowed.resume_depths[pc], borrowed.osr_entry_depth(pc));
        }
    }
    assert!(headers > 0);
    assert_eq!(borrowed.osr_entry_depth(chunk.jit_ops().len()), None);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| borrowed.mem_ptr())).is_err(),
        "borrowed code must not enter an ordinary inline call cache"
    );

    let statements = crate::parser::parse_script("function* body(){yield 1;}", false)
        .ok()
        .unwrap();
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("generator")
    };
    let chunk = crate::bytecode::compile(function).expect("generator compiles");
    assert!(chunk.jit_is_resumable());
    assert!(matches!(
        super::compile_borrowed_profiled(&chunk, &layout, &interpreter_layout),
        JitCompileOutcome::Unavailable
    ));
    let coroutine =
        super::compile(&chunk, &layout, &interpreter_layout).expect("coroutine native code");
    assert_eq!(coroutine.entry_kind, NativeEntryKind::BorrowedFrame);
    assert!(coroutine.osr_entry_depths.is_empty());
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn native_osr_entry_kind_admits_disposal_only_with_canonical_completion_state() {
    // ECMA-262 DisposeResources / ForIn/OfBodyEvaluation: an active resource stack and the
    // pending completion survive a tier transition; cleanup is never restarted or skipped.
    let mut engine = crate::Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let interpreter_layout = crate::interpreter::interp_layout(&mut engine.interp);
    let statements = crate::parser::parse_script(
        "function run(){for(let k=0;k<2000;k++){using value={[Symbol.dispose](){}};if(k===1700)break;}}",
        false,
    ).ok().unwrap();
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function")
    };
    let chunk = crate::bytecode::compile(function).expect("synchronous disposal bytecode");
    assert!(!chunk.jit_is_resumable());
    let pc = chunk
        .jit_ops()
        .iter()
        .position(|op| matches!(op, crate::bytecode::Op::DisposeNormal))
        .expect("normal disposal boundary");
    assert_eq!(chunk.jit_stack_effect(pc), None);
    assert_eq!(chunk.jit_borrowed_stack_effect(pc), Some((0, 0)));
    assert!(super::compile(&chunk, &layout, &interpreter_layout).is_none());
    let super::JitCompileOutcome::Compiled(code) =
        super::compile_borrowed_profiled(&chunk, &layout, &interpreter_layout)
    else {
        panic!("borrowed disposal code")
    };
    assert_eq!(code.entry_kind, super::NativeEntryKind::BorrowedFrame);
    assert!(code.osr_entry_depths.iter().any(Option::is_some));
}
