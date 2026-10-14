//! Actual native leaf calls, live SSA values and unchanged deoptimization ABI.
//! ECMA-262 e28783d5 OrdinaryGet/Set, Number values and WeakRef liveness.

use super::*;
use crate::{bytecode::Tier, value::Callable, Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn warmed(source: &str, warm: &str) -> (Engine, Rc<Chunk>) {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    eval(&mut engine, warm);
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("author callable")
    };
    let chunk = user
        .func
        .execution_code()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    drop(object);
    drop(value);
    (engine, chunk)
}

fn install(
    engine: &mut Engine,
    chunk: &Rc<Chunk>,
    values: &crate::value::JitLayout,
) -> Rc<JitCode> {
    generic_own::TEST_ENABLED.with(|cell| cell.set(Some(true)));
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = Rc::new(compile_with_deopt(chunk, values, &layout, None).unwrap());
    generic_own::TEST_ENABLED.with(|cell| cell.set(None));
    assert!(code.pc_offsets.is_empty());
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    engine.set_tier(Tier::Jit);
    code.residency.referenced.set(0);
    bytecode::native_deopt::ENTRIES.with(|entries| entries.borrow_mut().clear());
    owner_leaf::TEST_CALLS.with(|cell| cell.set([0; 2]));
    code
}

fn finished(engine: &mut Engine, code: &JitCode) {
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(code.residency.active.get(), 0);
    assert_eq!(engine.interp.depth, 0);
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
}

#[test]
fn actual_native_heap_calls_preserve_required_transfers_and_immediate_bypass() {
    let (mut engine, chunk) = warmed(
        "'use strict';function subject(o){o.x=o.x;return o.x;}var receiver={x:'shared'};",
        "subject(receiver);subject(receiver)",
    );
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let code = install(&mut engine, &chunk, &values);
    assert_eq!(eval(&mut engine, "subject(receiver)"), "shared");
    let calls = owner_leaf::TEST_CALLS.with(std::cell::Cell::get);
    assert_eq!(
        calls,
        if owner_leaf::requested() {
            [
                2,
                usize::from(std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() != Ok("0")),
            ]
        } else {
            [0; 2]
        }
    );
    assert_eq!(chunk.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    eval(&mut engine, "receiver.x=3");
    owner_leaf::TEST_CALLS.with(|cell| cell.set([0; 2]));
    assert_eq!(eval(&mut engine, "subject(receiver)"), "3");
    assert_eq!(owner_leaf::TEST_CALLS.with(std::cell::Cell::get), [0; 2]);
    finished(&mut engine, &code);
}

#[test]
fn caller_live_numeric_and_packed_ssa_values_survive_the_scalar_abi() {
    let (mut engine,chunk)=warmed(
        "function subject(o,v,a,b,c){'use strict';var left=a*1.25+b,right=c*2;o.x=o.x;return left+right+a+b+c+(o.x===v?1:0)}var held={tag:1},receiver={x:held};",
        "subject(receiver,held,.75,2.5,-3);subject(receiver,held,.75,2.5,-3)");
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let code = install(&mut engine, &chunk, &values);
    assert_eq!(
        eval(&mut engine, "subject(receiver,held,.75,2.5,-3)"),
        "-1.3125"
    );
    assert_eq!(chunk.optimizing_misses.get(), 0);
    if owner_leaf::requested() {
        let calls = owner_leaf::TEST_CALLS.with(std::cell::Cell::get);
        assert!(calls[0] >= 2);
        if std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() != Ok("0") {
            assert!(calls[1] >= 1);
        }
    }
    finished(&mut engine, &code);
}

#[test]
fn unboxed_arithmetic_prefix_survives_heap_owner_calls_without_observer_reload() {
    let (mut engine, chunk) = warmed(
        "'use strict';function subject(o,a,b,c){return(a*1.25+b*1)+(o,o,c*2)}var receiver={};",
        "subject(receiver,.75,2.5,-3);subject(receiver,.75,2.5,-3)",
    );
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let code = install(&mut engine, &chunk, &values);
    assert_eq!(eval(&mut engine, "subject(receiver,.75,2.5,-3)"), "-2.5625");
    let calls = owner_leaf::TEST_CALLS.with(std::cell::Cell::get);
    assert_eq!(
        calls,
        if owner_leaf::requested() {
            [2, 2]
        } else {
            [0; 2]
        }
    );
    assert_eq!(chunk.optimizing_misses.get(), 0);
    bytecode::native_deopt::ENTRIES.with(|entries| assert!(entries.borrow().is_empty()));
    finished(&mut engine, &code);
}

#[test]
fn unsupported_runtime_header_layout_emits_no_leaf_and_recovers_ownership() {
    let (mut engine, chunk) = warmed(
        "function subject(o){'use strict';o.x=o.x;return o.x;}var receiver={x:'kept'};",
        "subject(receiver);subject(receiver)",
    );
    let mut values = crate::value::jit_layout(&engine.interp.object_proto);
    values.rc_strong_off = 8;
    assert!(!crate::value::jit_packed_owners_supported(&values));
    let code = install(&mut engine, &chunk, &values);
    assert_eq!(eval(&mut engine, "subject(receiver)"), "kept");
    assert_eq!(owner_leaf::TEST_CALLS.with(std::cell::Cell::get), [0; 2]);
    // A packed heap owner is never treated as copyable under a bad layout.
    if std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() != Ok("0")
        && std::env::var("TRUST_JS_OPT_JIT_COMPACT_STORES").as_deref() != Ok("0")
    {
        assert_eq!(chunk.optimizing_misses.get(), 1);
    }
    finished(&mut engine, &code);
}

#[test]
fn full_diagnostics_count_real_heap_only_calls_guards_and_transfers_once() {
    const CHILD: &str = "TRUST_JS_TEST_OWNER_LEAF_DIAGNOSTICS_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","jit::optimizing::owner_leaf_tests::full_diagnostics_count_real_heap_only_calls_guards_and_transfers_once","--nocapture"])
            .env(CHILD,"1").env("TRUST_JS_OPT_JIT_SHARED_OWNERS","1")
            .env("TRUST_JS_OPT_JIT_DIAGNOSTICS","1")
            .env_remove("TRUST_JS_OPT_JIT_HEAP_OPS").env_remove("TRUST_JS_OPT_JIT_COMPACT_STORES")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let (mut engine, chunk) = warmed(
        "'use strict';function subject(o){o.x=o.x;return o.x;}var receiver={x:'shared'};",
        "subject(receiver);subject(receiver)",
    );
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let code = install(&mut engine, &chunk, &values);
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    assert!(record.runtime_counters);
    assert_eq!(record.emitted_owner_leaf_calls.get(), 3);
    for counter in &record.counters {
        counter.set(0)
    }
    assert_eq!(eval(&mut engine, "subject(receiver)"), "shared");
    assert_eq!(record.counters[diagnostics::OWNER_LEAF].get(), 3);
    assert_eq!(record.counters[diagnostics::OWNER_LEAF_GUARDS].get(), 9);
    assert_eq!(record.counters[diagnostics::RETAIN].get(), 2);
    assert_eq!(record.counters[diagnostics::RELEASE].get(), 1);
    for index in [
        diagnostics::OWNER_LEAF,
        diagnostics::OWNER_LEAF_GUARDS,
        diagnostics::RETAIN,
        diagnostics::RELEASE,
    ] {
        record.counters[index].set(u64::MAX - 1);
    }
    assert_eq!(eval(&mut engine, "subject(receiver)"), "shared");
    for index in [
        diagnostics::OWNER_LEAF,
        diagnostics::OWNER_LEAF_GUARDS,
        diagnostics::RETAIN,
        diagnostics::RELEASE,
    ] {
        assert_eq!(
            record.counters[index].get(),
            u64::MAX,
            "shared leaf dynamic counter saturates index{index}"
        );
    }
    eval(&mut engine, "receiver.x=3");
    for counter in &record.counters {
        counter.set(0)
    }
    assert_eq!(eval(&mut engine, "subject(receiver)"), "3");
    for index in [
        diagnostics::OWNER_LEAF,
        diagnostics::OWNER_LEAF_GUARDS,
        diagnostics::RETAIN,
        diagnostics::RELEASE,
    ] {
        assert_eq!(
            record.counters[index].get(),
            0,
            "immediate bypass index{index}"
        );
    }
    finished(&mut engine, &code);
    // The RHS is already materialized. A sole old-field owner rejects the
    // write leaf: count exactly its failed check, with no release.
    let (mut engine, chunk) = warmed(
        "'use strict';function subject(o,v){o.x=v;return 42}var held={tag:1},receiver={x:held};",
        "subject(receiver,held);subject(receiver,held)",
    );
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let code = install(&mut engine, &chunk, &values);
    let record = code.optimizing_diagnostics.as_ref().unwrap();
    eval(&mut engine, "receiver.x={sole:true}");
    for counter in &record.counters {
        counter.set(0)
    }
    assert_eq!(eval(&mut engine, "subject(receiver,7)"), "42");
    assert_eq!(record.counters[diagnostics::OWNER_LEAF].get(), 1);
    assert_eq!(record.counters[diagnostics::OWNER_LEAF_GUARDS].get(), 3);
    assert_eq!(record.counters[diagnostics::RETAIN].get(), 0);
    assert_eq!(record.counters[diagnostics::RELEASE].get(), 0);
    assert_eq!(
        record.counters[diagnostics::SPECIALIZATION_BAILOUT].get(),
        1
    );
    finished(&mut engine, &code);
}
