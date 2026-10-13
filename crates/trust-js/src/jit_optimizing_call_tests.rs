//! Real-call integration and actual shared-stub coverage. ECMA-262 e28783d5 [[Call]],
//! OrdinaryCallBindThis, execution contexts, completion routing and WeakRef liveness.

use super::compile_with_deopt;
use crate::bytecode::{self, Tier};
use crate::value::Callable;
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("call fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

pub(super) fn optimized(
    engine: &mut Engine,
    name: &str,
    deopt: Option<usize>,
) -> Rc<crate::jit::JitCode> {
    let env = engine.interp.global_env.clone();
    let value = engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("declared function");
    let object = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &object.call else {
        panic!("user function")
    };
    let function = user.func.clone();
    drop(object);
    let chunk = bytecode::compile(&function).expect("call fixture compiles");
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let interp = crate::interpreter::interp_layout(&mut engine.interp);
    let code =
        Rc::new(compile_with_deopt(&chunk, &values, &interp, deopt).expect("optimizing body"));
    assert!(
        code.pc_offsets.is_empty(),
        "must not silently use the template backend"
    );
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    assert!(
        function.code.set(Some(chunk)).is_ok(),
        "install before first call"
    );
    code
}

fn fixture(source: &str, expected: &str, callees: &[(&str, Option<usize>)], minimum_direct: usize) {
    fixture_with_stub(source, expected, callees, minimum_direct, true);
}

fn fixture_with_stub(
    source: &str,
    expected: &str,
    callees: &[(&str, Option<usize>)],
    minimum_direct: usize,
    expects_stub: bool,
) {
    let mut oracle = Engine::new();
    oracle.set_tier(Tier::Interp);
    eval(&mut oracle, source);
    assert_eq!(
        eval(&mut oracle, "subject()"),
        expected,
        "independent interpreter / expected effects"
    );

    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    let caller = optimized(&mut engine, "subject", None);
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        !caller.call_stubs.is_empty(),
        expects_stub,
        "actual outlined-call ownership"
    );
    #[cfg(not(target_arch = "aarch64"))]
    let _ = expects_stub;
    let callees: Vec<_> = callees
        .iter()
        .map(|&(name, pc)| optimized(&mut engine, name, pc))
        .collect();
    #[cfg(target_arch = "aarch64")]
    crate::jit::TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
    assert_eq!(eval(&mut engine, "subject()"), expected);
    #[cfg(target_arch = "aarch64")]
    assert!(
        crate::jit::TEST_DIRECT_PACKED_RETURNS.with(|count| count.get()) >= minimum_direct,
        "enough real shared-context calls must bypass the ordinary helper"
    );
    #[cfg(not(target_arch = "aarch64"))]
    let _ = minimum_direct;
    assert_eq!(engine.interp.depth, 0);
    assert!(engine.interp.fn_frames.is_empty());
    assert_eq!(caller.residency.active.get(), 0);
    for callee in callees {
        assert_eq!(callee.residency.active.get(), 0);
    }
    engine.interp.gc_collect();
}

#[test]
fn shared_call_stubs_cover_arity_boundaries_and_missing_parameters() {
    const CHILD: &str = "TRUST_JS_TEST_SHARED_CALL_ARITY_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "jit::optimizing::call_tests::shared_call_stubs_cover_arity_boundaries_and_missing_parameters", "--nocapture"])
            .env(CHILD, "1").env("TRUST_JS_OPT_JIT", "0").env("TRUST_JS_INLINE_AT", "0")
            .env_remove("TRUST_JS_JIT_NO_DIRECT_CALLS").env_remove("TRUST_JS_OPT_JIT_CALL_STUBS")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    // Zero, register-sized, wide stack-block and maximum supported arities, with
    // both receiver conventions. One extra formal verifies undefined initialization.
    // The adjacent unsupported arity must preserve the complete checked-call path.
    for argc in [0, 1, 8, 16, 31, 32, 63, 64, 65] {
        let parameters = (0..argc).map(|n| format!("p{n},")).collect::<String>();
        let args = (0..argc)
            .map(|n| format!("n+{n}"))
            .collect::<Vec<_>>()
            .join(",");
        let sum = (0..argc).map(|n| format!("+p{n}")).collect::<String>();
        for method in [false, true] {
            let (target, receiver, extra) = if method {
                ("object.leaf", "this.offset", 7)
            } else {
                ("leaf", "1", 1)
            };
            let source = format!(
                r#"
                function leaf({parameters}missing) {{
                    'use strict';
                    if (missing !== undefined) throw 'missing argument corrupted';
                    return {receiver}{sum};
                }}
                var object={{offset:7,leaf:leaf}};
                function subject() {{
                    var total=0;
                    for(var n=0;n<200;n++)total+={target}({args});
                    return total;
                }}
            "#
            );
            let expected = (argc * 19900 + 200 * (argc * (argc - 1) / 2 + extra)).to_string();
            fixture_with_stub(
                &source,
                &expected,
                &[("leaf", None)],
                if argc <= 64 { 170 } else { 0 },
                argc <= 64,
            );
        }
    }
}

#[test]
fn shared_call_stubs_preserve_real_calls_and_mixed_tier_completions() {
    const CHILD: &str = "TRUST_JS_TEST_SHARED_CALL_STUB_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "jit::optimizing::call_tests::shared_call_stubs_preserve_real_calls_and_mixed_tier_completions", "--nocapture"])
            .env(CHILD, "1").env("TRUST_JS_OPT_JIT", "0").env("TRUST_JS_INLINE_AT", "0")
            .env_remove("TRUST_JS_JIT_NO_DIRECT_CALLS").env_remove("TRUST_JS_OPT_JIT_CALL_STUBS")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let numeric = r#"
        function subject(){var sum=0;for(var n=0;n<400;n++)sum+=leaf(n);return sum;}
        function leaf(n){return n+2;}
    "#;
    fixture(numeric, "80600", &[], 350);
    fixture(numeric, "80600", &[("leaf", None)], 350);
    fixture(numeric, "80600", &[("leaf", Some(2))], 350);

    let completions = r#"
        function subject(){
            var sum=0;
            for(var n=0;n<200;n++){
                try{sum+=leaf(n);}catch(e){sum+=e;}finally{sum+=1;}
            }
            return sum;
        }
        function leaf(n){try{if(n%4===0)throw 3;return 2;}finally{if(n===17)return 7;}}
    "#;
    // The coverage counter records packed normal returns; fifty of these calls throw.
    fixture(completions, "655", &[], 130);
    fixture(completions, "655", &[("leaf", None)], 130);
    fixture(completions, "655", &[("leaf", Some(7))], 130);

    fixture(
        r#"
        function subject(){
            var total=0;for(var n=0;n<200;n++)total+=leaf(n).value;
            return total+'|'+typeof leaf;
        }
        function leaf(n){if(n===199)leaf=undefined;return {value:n};}
    "#,
        "19900|undefined",
        &[],
        170,
    );
    fixture(
        r#"
        function subject(){
            var total=0;for(var n=0;n<200;n++){
                try{total+=leaf(n);}catch(e){total+=e.value;}
            }
            return total+'|'+typeof leaf;
        }
        function leaf(n){if(n===199){leaf=undefined;throw {value:n};}return n;}
    "#,
        "19900|undefined",
        &[],
        170,
    );

    fixture(
        r#"
        function subject(){
            var total=0;
            for(var n=0;n<400;n++){var f=choices[n%choices.length];total+=f(n);}
            return total;
        }
        function a(n){return n+1;} function b(n){return n+2;}
        function c(n){return n+3;} function d(n){return n+4;}
        function e(n){return n+5;} function f(n){return n+6;}
        function g(n){return n+7;} function h(n){return n+8;}
        var choices=[a,b,c,d,e,f,g,h];
    "#,
        "81600",
        &[],
        350,
    );
}

#[test]
fn shared_call_stubs_keep_receivers_and_checked_exotic_call_paths() {
    const CHILD: &str = "TRUST_JS_TEST_SHARED_CALL_RECEIVER_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "jit::optimizing::call_tests::shared_call_stubs_keep_receivers_and_checked_exotic_call_paths", "--nocapture"])
            .env(CHILD, "1").env("TRUST_JS_OPT_JIT", "0").env("TRUST_JS_INLINE_AT", "0")
            .env_remove("TRUST_JS_JIT_NO_DIRECT_CALLS").env_remove("TRUST_JS_OPT_JIT_CALL_STUBS")
            .output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    fixture(
        r#"
        function subject(){var sum=0;for(var n=0;n<300;n++)sum+=object.read(n);return sum;}
        function read(n){'use strict';return this.value+n;}
        var object={value:7,read:read};
    "#,
        "46950",
        &[],
        250,
    );
    fixture(
        r#"
        function subject(){var sum=0;for(var n=0;n<300;n++)sum+=(3).read(n);return sum;}
        function read(n){'use strict';if(typeof this!=='number')throw 'boxed';return this+n;}
        Number.prototype.read=read;
    "#,
        "45750",
        &[],
        250,
    );
    fixture(
        r#"
        function subject(){var sum=0;for(var n=0;n<20;n++)sum+=(3).read(n);return sum;}
        function read(n){if(typeof this!=='object')throw 'unboxed';return this.valueOf()+n;}
        Number.prototype.read=read;
    "#,
        "250",
        &[],
        0,
    );
    fixture(
        r#"
        function subject(){
            var total=0;
            for(var n=0;n<20;n++){
                var f=choices[n%4];total+=f(n,10,20);
            }
            return total+'|'+trace.join(',');
        }
        function ordinary(n){return n+1;}
        function make(value){return n=>value+n;}
        function rest(n,...xs){return n+xs.length;}
        var trace=[];
        var proxy=new Proxy(ordinary,{apply(t,r,a){trace.push(a[0]);return Reflect.apply(t,r,a)+3;}});
        var choices=[ordinary.bind(null),make(5),rest,proxy];
    "#,
        "250|3,7,11,15,19",
        &[],
        0,
    );
}
