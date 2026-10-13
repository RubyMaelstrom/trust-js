//! Executed native counters and lifetime/profiler isolation, including observable fallback.
use super::*;
use crate::{bytecode::Tier, Completion, Engine};
use std::{any::Any, cell::Cell, rc::Rc};

#[test]
fn diagnostic_code_lifetime_and_profiler_isolation() {
    const CHILD: &str = "TRUST_JS_TEST_OPT_DIAGNOSTICS_CHILD";
    if std::env::var_os(CHILD).is_none() {
        for mode in ["1", "compile"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "jit::optimizing::diagnostic_tests::diagnostic_code_lifetime_and_profiler_isolation", "--nocapture"])
            .env(CHILD, "1")
            .env("TRUST_JS_OPT_JIT_DIAGNOSTICS", mode)
            .env("TRUST_JS_OPT_JIT", "0")
            .env_remove("TRUST_JS_OPT_JIT_DEOPT_AT")
            .output().unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        return;
    }
    fn eval(engine: &mut Engine, source: &str) -> String {
        match engine.eval(source, false).unwrap() {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }
    struct HostProfiler(Rc<Cell<usize>>);
    impl cranelift_codegen::timing::Profiler for HostProfiler {
        fn start_pass(&self, _: cranelift_codegen::timing::Pass) -> Box<dyn Any> {
            self.0.set(self.0.get() + 1);
            Box::new(())
        }
    }
    let host_calls = Rc::new(Cell::new(0));
    let previous =
        cranelift_codegen::timing::set_thread_profiler(Box::new(HostProfiler(host_calls.clone())));
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "var source={x:3},reads=0;function subject(o){var local=o;return local.x+1;}",
    );
    let code = call_tests::optimized(&mut engine, "subject", None);
    let record = code
        .optimizing_diagnostics
        .as_ref()
        .expect("diagnostic owner");
    assert_eq!(record.counters[0].get(), 0);
    assert_eq!(host_calls.get(), 0, "compiler passes isolated from host");
    drop(cranelift_codegen::timing::verifier());
    assert_eq!(host_calls.get(), 1, "host profiler restored");
    let _ = cranelift_codegen::timing::set_thread_profiler(previous);
    assert_eq!(eval(&mut engine, "subject(source)"), "4");
    assert_eq!(eval(&mut engine, "subject(source)"), "4");
    assert_eq!(eval(&mut engine, "Object.defineProperty(source,'x',{get(){reads++;throw 'read';}});try{subject(source);}catch(e){e+':'+reads;}"), "read:1");
    if record.runtime_counters {
        assert_eq!(
            record.counters[0].get(),
            3,
            "actual native entries including throw"
        );
        assert!(
            record.counters[2].get() > 0,
            "live getter used checked read"
        );
        assert!(record.counters[diagnostics::PUBLICATION].get() > 0);
    } else {
        assert!(
            record.counters.iter().all(|counter| counter.get() == 0),
            "compiler-only observation emits no counter writes"
        );
    }
    assert!(record.codegen_ns.get() > 0);
    assert!(record.native_bytes.get() > 0);
    let weak = Rc::downgrade(record);
    drop(engine);
    assert!(
        weak.upgrade().is_some(),
        "retained native code pins its counter addresses"
    );
    drop(code);
    assert!(
        weak.upgrade().is_none(),
        "diagnostic directory cannot retain code metadata"
    );
    assert!(!diagnostics::snapshot_json().contains("\"ops\":"));
}
