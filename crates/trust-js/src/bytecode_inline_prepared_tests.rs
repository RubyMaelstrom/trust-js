//! Frame-independent splices retain the prepared caller's normative activation.
//! ECMA-262 e28783d5 FunctionDeclarationInstantiation, mapped arguments, eval and closures.

use crate::bytecode::{self as bytecode, Op, Tier};
use crate::value::Callable;
use crate::{Completion, Engine};

thread_local! {
    static LEAF_FUNCTION:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};
    static SAW_REAL_LEAF_FRAME:std::cell::Cell<bool>=const{std::cell::Cell::new(false)};
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("prepared inline fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

#[test]
fn prepared_baseline_inlines_retain_mapped_arguments_eval_and_captured_bindings() {
    const SOURCE: &str = r#"
        function make(closed) {return function leaf(arg) {
            this.count++;return this.bias+closed+arg;
        };}
        var holder={count:0,bias:2,f:make(4)},trace=[];
        function subject(value) {
            var read=()=>value,closed=99;
            var arg={valueOf(){trace.push('coerce');collectPreparedInlineTest();return 3;}};
            try {
                var r=holder.f(arg);
                arguments[0]=value+1;
                eval('value += 2');
                return r+':'+read()+':'+arguments[0]+':'+closed;
            } finally {trace.push('finally');collectPreparedInlineTest();}
        }
    "#;
    const OBSERVE: &str = "holder.count=0;trace=[];subject(5)+'|'+holder.count+'|'+trace.join(',')";
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        LEAF_FUNCTION.with(|function| function.set(0));
        SAW_REAL_LEAF_FRAME.with(|saw| saw.set(false));
        engine.interp.def_method(
            &engine.interp.global,
            "collectPreparedInlineTest",
            0,
            |i, _, _| {
                let target = LEAF_FUNCTION.with(std::cell::Cell::get);
                if target != 0 && i.fn_frames.iter().any(|frame| frame.fn_ptr == target) {
                    SAW_REAL_LEAF_FRAME.with(|saw| saw.set(true));
                }
                i.gc_collect();
                Ok(crate::value::Value::Undefined)
            },
        );
        eval(&mut engine, SOURCE);
        let env = engine.interp.global_env.clone();
        let value = engine.interp.get_var("subject", &env).ok().unwrap();
        let object_handle = value.as_obj().unwrap();
        let object = object_handle.borrow();
        let Callable::User(user) = &object.call else {
            panic!("prepared subject")
        };
        let user = user.clone();
        drop(object);
        if tier == Tier::Interp {
            assert_eq!(eval(&mut engine, OBSERVE), "9:8:8:99|1|coerce,finally");
            continue;
        }
        let primary = bytecode::compile(&user.func).expect("prepared primary");
        assert!(primary.prepared_entry);
        primary.inline_attempted.set(true);
        primary.inline_retry_at.set(0);
        assert!(user.func.runtime().code.set(Some(primary.clone())).is_ok());
        // Inline planning consumes native call-cache feedback; the transformed continuation
        // is then exercised separately on both the bytecode and template native tiers.
        engine.set_tier(Tier::Jit);
        for _ in 0..8 {
            eval(&mut engine, "subject(5);");
        }
        let plan =
            bytecode::plan_inlines(&primary, &user.func, user.realm, &engine.interp.global_env);
        assert!(!plan.is_empty());
        let chunk =
            bytecode::compile_with_inlines(&user.func, &plan, &primary).expect("prepared splice");
        assert!(
            chunk.prepared_entry,
            "the transformed caller must reuse the same prepared activation"
        );
        assert_eq!(chunk.jit_frame().0, primary.jit_frame().0);
        assert!(chunk
            .jit_ops()
            .iter()
            .any(|op| matches!(op, Op::InlineGuard(..))));
        let callee_code = chunk.jit_ops().iter().find_map(|op| {
            let Op::InlineGuard(target, _) = *op else {
                return None;
            };
            let callee = chunk.jit_inline_target(target).pin.upgrade().unwrap();
            let object = callee.borrow();
            let Callable::User(user) = &object.call else {
                panic!("planned user callee")
            };
            // FnFrame tracks the callable object identity, rather than the shared AST.
            LEAF_FUNCTION.with(|function| function.set(crate::value::Gc::as_ptr(&callee) as usize));
            let runtime = user.func.runtime();
            let body = runtime.execution_code().and_then(Option::as_ref).unwrap();
            body.jit.get().flatten()
        });
        // Positive control: the original Call really exposes the physical leaf frame at
        // the coercion observer. A stale/wrong identity must not make the hit proof vacuous.
        SAW_REAL_LEAF_FRAME.with(|saw| saw.set(false));
        eval(&mut engine, "subject(5);");
        assert!(SAW_REAL_LEAF_FRAME.with(std::cell::Cell::get));
        if let Some(code) = &callee_code {
            code.residency.referenced.set(0);
        }
        assert!(user.func.runtime().code2.set(Some(chunk.clone())).is_ok());
        engine.set_tier(tier);
        SAW_REAL_LEAF_FRAME.with(|saw| saw.set(false));
        assert_eq!(
            eval(&mut engine, OBSERVE),
            "9:8:8:99|1|coerce,finally",
            "{tier:?}"
        );
        if tier == Tier::Jit {
            let code = chunk
                .jit
                .get()
                .flatten()
                .expect("actual template native entry");
            assert!(!code.pc_offsets.is_empty(), "default template backend");
            assert_eq!(code.residency.referenced.get(), 1);
            assert_eq!(code.residency.active.get(), 0);
            assert_eq!(
                callee_code.unwrap().residency.referenced.get(),
                0,
                "real inline hit"
            );
        }
        assert!(
            !SAW_REAL_LEAF_FRAME.with(std::cell::Cell::get),
            "both VM and template tiers must execute the splice instead of the original Call"
        );
        assert_eq!(engine.interp.depth, 0);
        assert!(engine.interp.fn_frames.is_empty());
        engine.interp.gc_collect();
    }
}

#[test]
fn prepared_native_hot_calls_publish_and_execute_baseline_inline_bodies() {
    const CHILD: &str = "TRUST_JS_TEST_PREPARED_INLINE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output=std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact","jit::inline_prepared_tests::prepared_native_hot_calls_publish_and_execute_baseline_inline_bodies","--nocapture"])
            .env(CHILD,"1").env("TRUST_JS_INLINE_AT","8").output().unwrap();
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
    assert_eq!(
        eval(
            &mut engine,
            r#"
        function leaf(x){return x+3;}
        function subject(x){
            var read=()=>x;
            arguments[0]=x+1;
            var out=leaf(x);
            eval('x += 2');
            return out+read();
        }
        var sum=0;for(var k=0;k<200;k++)sum+=subject(k);sum;
    "#
        ),
        "41200"
    );
    let env = engine.interp.global_env.clone();
    let value = engine.interp.get_var("subject", &env).ok().unwrap();
    let object_handle = value.as_obj().unwrap();
    let object = object_handle.borrow();
    let Callable::User(user) = &object.call else {
        panic!("prepared subject")
    };
    let function = user.func.clone();
    drop(object);
    let runtime = function.runtime();
    let primary = runtime.code.get().and_then(Option::as_ref).unwrap();
    assert!(primary.prepared_entry);
    let runtime = function.runtime();
    let chunk = runtime
        .code2
        .get()
        .and_then(Option::as_ref)
        .expect("hot prepared caller is inlined automatically");
    assert!(chunk.prepared_entry);
    assert!(chunk
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::InlineGuard(..))));
    let code = chunk
        .jit
        .get()
        .flatten()
        .expect("published body actually executed");
    assert!(!code.pc_offsets.is_empty());
    code.residency.referenced.set(0);
    assert_eq!(eval(&mut engine, "subject(5)"), "17");
    assert_eq!(code.residency.referenced.get(), 1);
    assert_eq!(code.residency.active.get(), 0);
    assert_eq!(engine.interp.depth, 0);
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
}
