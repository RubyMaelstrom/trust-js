//! Packed activation roots: independent initialization flags and suspended receiver words.
//! ECMA-262 e28783d5 BindThisValue, GeneratorYield, Await and GetThisEnvironment.

use super::*;

#[test]
fn initialized_and_tdz_binding_flags_are_independent_of_word_contents() {
    use crate::interpreter::{Binding, VarMap};
    let mut vars = VarMap::default();
    vars.insert("initialized", Binding::data(Value::Empty, true, true));
    vars.insert("tdz", Binding::data(Value::Undefined, false, false));
    assert!(vars.get("initialized").unwrap().initialized());
    assert_eq!(
        vars.get("initialized").unwrap().value().bits(),
        crate::value::PACK_EMPTY
    );
    assert!(!vars.get("tdz").unwrap().initialized());
    assert_eq!(
        vars.get("tdz").unwrap().value().bits(),
        crate::value::PACK_UNDEFINED
    );
    vars.get_mut("initialized")
        .unwrap()
        .set_value(Value::Undefined);
    vars.get_mut("tdz").unwrap().set_value(Value::Null);
    let initialized = vars.get("initialized").unwrap();
    let tdz = vars.get("tdz").unwrap();
    assert!(initialized.initialized() && initialized.mutable());
    assert!(!tdz.initialized() && !tdz.mutable());
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn old_with_record_keeps_its_sole_young_object_through_the_packed_word() {
    use crate::gc_core::{CellAge, CollectionKind};
    let mut engine = crate::Engine::new();
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
    engine.interp.gc_heap.agent.task_deferred.set(true);
    let scope = crate::rooting::Rooted::new(crate::interpreter::new_scope(None));
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    let heap = engine.interp.gc_heap.clone();
    assert_eq!(
        heap.core.borrow().cell_age(Env::as_ptr(&scope) as u64),
        Some(CellAge::Old)
    );
    let object = engine.interp.new_object();
    let cell = crate::value::Gc::as_ptr(&object) as usize;
    let weak = crate::gc_handle::WeakRef::to(cell);
    scope.borrow_mut().set_with_object(Some(Value::Obj(object)));
    assert_eq!(
        heap.core.borrow().cell_age(cell as u64),
        Some(CellAge::Young)
    );
    for kind in [CollectionKind::Minor, CollectionKind::Major] {
        engine.interp.gc_collect_precise_for_test(kind);
        assert_eq!(weak.live_ptr(), Some(cell));
        assert!(heap.core.borrow().owns_cell(cell, heap.object_desc));
    }
    scope.borrow_mut().set_with_object(None);
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    assert_eq!(weak.live_ptr(), None);
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn suspended_this_word_is_the_sole_receiver_root_across_yield_await_and_completion() {
    use crate::gc_core::{CellAge, CollectionKind};
    use crate::{Completion, Engine};
    let evaluate = |engine: &mut Engine, source: &str| match engine.eval(source, false).unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    };
    for tier in [Tier::Bytecode, Tier::Jit] {
        for asynchronous in [false, true] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            // This fixture proves an actual young receiver survives its first precise minor.
            // Automatic stress/task polls must not promote it before that controlled collection.
            engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
            engine.interp.gc_heap.agent.task_deferred.set(true);
            evaluate(
                &mut engine,
                if asynchronous {
                    "var release,result='pending',gate=new Promise(resolve=>release=resolve);\
                    async function paused(){await gate;return this;}\
                 var receiver={marker:41},pending=paused.call(receiver);0"
                } else {
                    "function* paused(){yield 0;return this;}\
                 var receiver={marker:41},pending=paused.call(receiver);pending.next();0"
                },
            );
            let globals = engine.interp.global_env;
            let receiver = engine.interp.get_var("receiver", &globals).ok().unwrap();
            let word = (receiver).bits();
            let cell = receiver.cell_word().unwrap() as usize;
            let weak = crate::gc_handle::WeakRef::to(cell);
            evaluate(&mut engine, "receiver=null;0");
            let paused_chunk = {
                let continuations: Vec<_> = engine
                    .interp
                    .generators
                    .values()
                    .filter_map(|state| {
                        if let crate::coroutine::Coroutine::Vm(coro) = state {
                            Some(coro)
                        } else {
                            None
                        }
                    })
                    .collect();
                assert_eq!(continuations.len(), 1, "one parked source continuation");
                let coro = continuations[0];
                assert!(coro.started && !coro.done);
                assert!(!coro.chunk.env_this && !coro.chunk.lexical_this);
                assert_eq!(coro.this_val.bits(), word);
                assert!(coro.slots.iter().all(|slot| slot.get().bits() != word));
                assert!(coro
                    .stack
                    .raw_slice()
                    .iter()
                    .all(|value| value.bits() != word));
                assert!(coro.references.iter().all(|slot| slot.as_ref().is_none()));
                assert!(Env::ptr_eq(&coro.cap_env, &coro.env));
                // Ordinary FunctionDeclarationInstantiation can leave a redundant `this`
                // binding in the original environment. This compiled body has no env/lexical
                // this dependency (proved above): remove that bookkeeping copy to isolate the
                // continuation word, without changing any operation the body will execute.
                let mut env = Some(coro.env);
                while let Some(current) = env {
                    let mut scope = current.borrow_mut();
                    if let Some(mut binding) = scope.vars.get_mut("this") {
                        if binding.value().bits() == word {
                            binding.set_value(Value::Undefined);
                        }
                    }
                    assert!(scope
                        .vars
                        .values()
                        .all(|binding| binding.value().bits() != word));
                    assert!(!scope
                        .with_object()
                        .is_some_and(|value| (value).bits() == word));
                    env = scope.parent;
                }
                coro.chunk.clone()
            };
            let heap = engine.interp.gc_heap.clone();
            assert_eq!(
                heap.core.borrow().cell_age(cell as u64),
                Some(CellAge::Young)
            );
            for kind in [CollectionKind::Minor, CollectionKind::Major] {
                // Native scanning is disabled: the stale receiver/word Rust locals above
                // are deliberately excluded; only the actual parked continuation retains it.
                engine.interp.gc_collect_precise_for_test(kind);
                assert_eq!(weak.live_ptr(), Some(cell));
                assert!(heap.core.borrow().owns_cell(cell, heap.object_desc));
            }
            // The initial native slice only reached yield/await, before any receiver read.
            // Observe this exact paused Chunk around its targeted resume, not aggregate setup.
            let native_before = (
                paused_chunk.jit_runs.get(),
                paused_chunk.test_this_commits.get(),
            );
            if asynchronous {
                evaluate(
                    &mut engine,
                    "pending.then(value=>result=value.marker);release();0",
                );
                assert_eq!(evaluate(&mut engine, "result"), "41");
            } else {
                assert_eq!(evaluate(&mut engine, "pending.next().value.marker"), "41");
            }
            if tier == Tier::Jit {
                assert!(
                    paused_chunk.jit_runs.get() > native_before.0,
                    "the exact paused continuation must enter native code on receiver resume"
                );
                assert!(
                    paused_chunk.test_this_commits.get() > native_before.1,
                    "the exact paused Chunk must commit its receiver word after that resume"
                );
            }
            evaluate(&mut engine, "pending=null;0");
            engine
                .interp
                .gc_collect_precise_for_test(CollectionKind::Major);
            assert_eq!(
                weak.live_ptr(),
                None,
                "a completed continuation must release its receiver"
            );
            assert!(!heap.core.borrow().owns_cell(cell, heap.object_desc));
        }
    }
}
