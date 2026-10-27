use super::*;
use crate::value::ValueView;

fn operand_owner_loop(i: &mut Interp) -> (Rc<Chunk>, Value) {
    let body =
        crate::parser::parse_script("let n=0; for(let k=0;k<2000;k++){n+=k;n+=1;} n;", false)
            .ok()
            .expect("metadata fixture parses");
    let mut chunk = compile_script(&body, false).expect("metadata fixture compiles");
    let owner = Value::Obj(crate::value::Object::new(Some(i.object_proto)));
    chunk = Chunk::rewrite_owned_for_test(chunk, |raw| {
        // Keep one owned operand below the loop condition. Every map must retain it;
        // the conditional edge has already popped its boolean before OSR admission.
        let base = raw.consts.len() as u32;
        raw.consts
            .extend([owner, Value::Num(0.0), Value::Num(2000.0), Value::Num(1.0)]);
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
    });
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
    let env = engine.interp.global_env;
    crate::jit::TEST_OSR_ENTRIES.with(|count| count.set(0));
    let result = run_tiered_script(&mut engine.interp, &chunk, &env, Value::Undefined)
        .ok()
        .expect("borrowed operand loop completes");
    let (ValueView::Obj(actual), ValueView::Obj(expected)) = (result.view(), owner.view()) else {
        panic!("owned objects");
    };
    assert!(crate::value::Gc::ptr_eq(&actual, &expected));
    assert!(crate::jit::TEST_OSR_ENTRIES.with(|count| count.get()) > 0);
}

/// Rooting harness (`jit::rooting_tests`): the operand prefix a hot Script loop
/// lends its native code at OSR entry is a root for that native run. A fresh object the VM
/// tracks below a loop entered at depth 1 waits in the borrowed frame's operand words, which the
/// entry registers; when the native code calls `collect()` after the loop, they mark it. The
/// run returns it intact, and once the run ends nothing keeps it.
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn osr_entry_operand_prefix_is_a_root() {
    use crate::jit::rooting_tests::{assert_tracked_collectable, jit_engine, last_census};
    let mut engine = jit_engine(0);
    let body = crate::parser::parse_script(
        "track({tag: 'O'}); for (let k = 0; k < 2000; k++) {} osrMarks = collect();",
        false,
    )
    .ok()
    .expect("prefix fixture parses");
    let mut chunk = compile_script(&body, false).expect("prefix fixture compiles");
    chunk = Chunk::rewrite_owned_for_test(chunk, |raw| {
        // The compiled call, literal and reference sites (with their caches) are reused in another
        // order, so that `track`'s result stays on the stack below the loop instead of becoming the
        // completion value.
        assert_eq!(
            raw.ops[..4],
            [
                Op::LoadNameForCall(0, 0),
                Op::Const(0),
                Op::MakeObject(1, 1, 0),
                Op::CallWithThis(1, 0)
            ]
        );
        assert_eq!(
            raw.ops[16..19],
            [
                Op::ResolveNameRef(2, 0),
                Op::LoadNameForCall(3, 1),
                Op::CallWithThis(0, 1)
            ]
        );
        assert!(matches!((raw.consts[1]).view(), ValueView::Num(n) if n == 0.0));
        assert!(matches!((raw.consts[2]).view(), ValueView::Num(n) if n == 2000.0));
        raw.ops = vec![
            // track({tag: 'O'}): the prefix.
            Op::LoadNameForCall(0, 0),
            Op::Const(0),
            Op::MakeObject(1, 1, 0),
            Op::CallWithThis(1, 0),
            // for (k = 0; k < 2000; k++) {}, its header at depth 1.
            Op::Const(1),
            Op::StoreLocal(1),
            Op::LoadLocal(1),
            Op::Const(2),
            Op::Lt,
            Op::JumpIfFalse(12),
            Op::UpdateLocal(1, UpdKind::IncDiscard),
            Op::Jump(6),
            // osrMarks = collect(), in the same native run.
            Op::ResolveNameRef(2, 0),
            Op::LoadNameForCall(3, 1),
            Op::CallWithThis(0, 1),
            Op::StoreRef(0),
            Op::Return,
        ];
        raw.jit_needs_activation_state = raw.ops.iter().any(jit_bridge_op);
        raw.has_tail_calls = false;
    });
    let env = engine.interp.global_env;
    crate::jit::TEST_OSR_ENTRIES.with(|count| count.set(0));
    // The result's handle lives only in this helper frame (the collector scans the stack).
    let tag = crate::gc_test::in_frame(|| {
        let result = run_tiered_script(&mut engine.interp, &chunk, &env, Value::Undefined)
            .ok()
            .expect("the prefix loop completes");
        let marks = engine.interp.get_var("osrMarks", &env).ok().expect("marks");
        assert!(
            matches!(&(marks).view(), ValueView::Str(marks) if marks.text() == "R"),
            "the prefix was marked by a root"
        );
        let tag = result
            .as_obj()
            .expect("the prefix")
            .borrow()
            .props
            .get("tag")
            .map(crate::value::Property::value);
        tag
    });
    assert!(
        matches!((tag).map(|__value| (__value).view()), Some(ValueView::Str(tag)) if tag.text() == "O")
    );
    assert!(crate::jit::TEST_OSR_ENTRIES.with(|count| count.get()) > 0);
    let census = last_census();
    assert!(
        census
            .entries
            .iter()
            .any(|&[slots, stack]| slots == 0 && stack > 0),
        "collect() ran in the loop's borrowed frame: {census:?}"
    );
    assert_tracked_collectable(&mut engine);
}

#[test]
fn osr_compile_unavailable_keeps_exact_baseline_state_without_replay() {
    let mut engine = crate::Engine::new();
    engine.set_tier(Tier::Jit);
    let (chunk, owner) = operand_owner_loop(&mut engine.interp);
    // A language-resumable chunk is deliberately ineligible for Script OSR. This
    // fixture only uses ordinary ops, so the VM still completes the same loop.
    let mut chunk = chunk;
    chunk = Chunk::rewrite_owned_for_test(chunk, |chunk| chunk.resumable = true);
    let state = chunk
        .osr
        .get_or_init(|| Box::new(crate::tiering::OsrCodeState::default()));
    assert!(state.code(&mut engine.interp, &chunk).is_none());
    assert!(state.unavailable());
    chunk = Chunk::rewrite_owned_for_test(chunk, |chunk| chunk.resumable = false);
    let env = engine.interp.global_env;
    crate::jit::TEST_OSR_ENTRIES.with(|count| count.set(0));
    let result = run_tiered_script(&mut engine.interp, &chunk, &env, Value::Undefined)
        .ok()
        .expect("unavailable native code stays bytecode");
    let (ValueView::Obj(actual), ValueView::Obj(expected)) = (result.view(), owner.view()) else {
        panic!("owned objects");
    };
    assert!(crate::value::Gc::ptr_eq(&actual, &expected));
    assert_eq!(crate::jit::TEST_OSR_ENTRIES.with(|count| count.get()), 0);
}
