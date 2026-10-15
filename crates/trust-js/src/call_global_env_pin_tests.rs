//! Raw CallIc/ConstructIc Realm proofs must not survive recycling their genv identity.
//! ECMA-262 e28783d5 GetFunctionRealm / BuiltinCallOrConstruct: a native function
//! entered from another Realm still allocates and observes globals in its own Realm.

use crate::bytecode::{CallIc, Op, Tier, CALL_IC_EPOCH, CALL_IC_WAYS};
use crate::realm_inline_guard_tests::{eval_string, in_realm, set_global};
use crate::value::Callable;
use crate::{Completion, Engine};
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::Ordering::Relaxed;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("call pin fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

#[test]
fn expired_call_global_env_pin_fences_live_native_cache_before_identity_reuse() {
    const CHILD: &str = "TRUST_JS_TEST_GENV_PIN_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "jit::call_global_env_pin_tests::expired_call_global_env_pin_fences_live_native_cache_before_identity_reuse",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("TRUST_JS_INLINE_AT", "0")
            .output()
            .expect("isolated global epoch test");
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine.interp.switch_host_job_context(41);
    let main = engine.interp.global_this();
    engine
        .interp
        .def_method(&engine.interp.global, "nativeRealmProbe", 0, |i, _, _| {
            Ok(i.global_this())
        });
    let native = engine
        .interp
        .member_get(&main, "nativeRealmProbe")
        .ok()
        .expect("installed native probe");
    let native_id = Rc::as_ptr(native.as_obj().unwrap()) as usize;
    let native_pin = Rc::downgrade(native.as_obj().unwrap());
    let bytes =
        crate::compile_snapshot("globalThis.subject = function(native) { return native(); };")
            .expect("shared caller snapshot");
    // The program's function AST, unlike its per-Realm callable, survives the settings change.
    let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
    let program = engine.interp.shared_snapshot_program(bytes).unwrap();
    assert!(engine.interp.run_classic_program(&program, None).is_ok());
    for _ in 0..8 {
        assert_eq!(
            eval(&mut engine, "subject(nativeRealmProbe) === globalThis"),
            "true"
        );
    }
    let subject = engine
        .interp
        .member_get(&main, "subject")
        .ok()
        .expect("shared caller callable");
    let function = match &subject.as_obj().unwrap().borrow().call {
        Callable::User(user) => user.func.clone(),
        _ => panic!("shared caller function"),
    };
    let chunk = function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    let code = chunk.jit.get().flatten().expect("real template caller");
    assert!(!code.pc_offsets.is_empty());
    let cache = chunk
        .jit_ops()
        .iter()
        .find_map(|op| match *op {
            Op::Call(0, site) => Some(site),
            _ => None,
        })
        .expect("native call site");
    let cells = unsafe {
        std::slice::from_raw_parts(
            chunk.jit_call_cache_ptr(cache) as *const Cell<CallIc>,
            CALL_IC_WAYS,
        )
    };
    let original = cells
        .iter()
        .map(Cell::get)
        .find(|entry| entry.callee == native_id)
        .expect("warmed live native entry");
    assert_ne!(original.native, 0);
    assert_eq!(
        original.global_env,
        Rc::as_ptr(&engine.interp.global_env) as usize
    );
    let retired_env = Rc::downgrade(&engine.interp.global_env);
    assert!(engine
        .interp
        .global_env_pins
        .iter()
        .any(|pin| pin.as_ptr() == retired_env.as_ptr()));

    // A collection with only live environment pins does not invalidate unrelated call sites.
    engine.interp.gc_collect();
    let unchanged = CALL_IC_EPOCH.load(Relaxed);
    engine.interp.gc_collect();
    assert_eq!(CALL_IC_EPOCH.load(Relaxed), unchanged);
    assert_eq!(
        cells
            .iter()
            .map(Cell::get)
            .find(|entry| entry.callee == native_id)
            .unwrap()
            .epoch,
        unchanged
    );

    eval(&mut engine, "delete globalThis.subject;");
    drop(subject);
    assert!(engine.interp.release_host_job_context(41));
    engine.interp.switch_host_job_context(42);
    assert_eq!(
        retired_env.strong_count(),
        0,
        "callee/code do not retain the old settings environment"
    );
    let before = CALL_IC_EPOCH.load(Relaxed);
    engine.interp.gc_collect();
    assert_eq!(
        CALL_IC_EPOCH.load(Relaxed),
        before + 1,
        "one fence before removing expired genv pins"
    );
    assert!(!engine
        .interp
        .global_env_pins
        .iter()
        .any(|pin| pin.as_ptr() == retired_env.as_ptr()));
    assert!(
        native_pin.upgrade().is_some(),
        "the native cache's callee remains live"
    );

    let child = engine.interp.create_realm();
    in_realm(&mut engine.interp, &child, |i| {
        assert!(i.run_classic_program(&program, None).is_ok());
        set_global(i, "mainRealm", main.clone());
        set_global(i, "mainNative", native.clone());
        let child_subject = i
            .member_get(&i.global_this(), "subject")
            .ok()
            .expect("child caller callable");
        let body = child_subject.as_obj().unwrap().borrow();
        let Callable::User(user) = &body.call else {
            panic!("child caller");
        };
        assert!(
            Rc::ptr_eq(&function, &user.func),
            "same AST/code/IC in another Realm"
        );
        drop(body);
        // Model allocator reuse deterministically: all cached fields except the retired
        // genv address remain the genuine live entry. The old epoch must reject it before
        // direct native dispatch, which would return the child global instead of main.
        for cell in cells {
            let entry = cell.get();
            if entry.callee == native_id {
                cell.set(CallIc {
                    global_env: Rc::as_ptr(&i.global_env) as usize,
                    ..entry
                });
            }
        }
        code.residency.referenced.set(0);
        assert_eq!(eval_string(i, "subject(mainNative) === mainRealm"), "true");
        assert_eq!(
            code.residency.referenced.get(),
            1,
            "actual cached caller body executes"
        );
    });
    assert!(engine.interp.fn_frames.is_empty());
}
