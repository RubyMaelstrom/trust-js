//! ECMA-262 e28783d5 Assignment Evaluation / ApplyStringOrNumericBinaryOperator /
//! GetValue / PutValue. Actual native transactions, original partial/resume state and
//! observer/coercion fallbacks are checked independently of helper presence.
use super::*;
use crate::ast::{Function, Stmt};
use crate::bytecode::{self, Tier};
use crate::interpreter::{new_scope, Binding};
use crate::value::ValueView;
use crate::{Completion, Engine};
use std::cell::Cell;
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("accumulation fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn function(source: &str) -> Rc<Function> {
    let statements = crate::parser::parse_script(source, false).ok().unwrap();
    let Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function fixture")
    };
    function.clone()
}

fn chunk() -> Rc<Chunk> {
    bytecode::compile(&function("function subject(part){return result+=part;}"))
        .expect("ordinary primitive accumulation")
}

fn first_plan(chunk: &Chunk) -> Plan {
    (0..chunk.ops.len())
        .find_map(|pc| Plan::at(chunk, pc))
        .expect("complete original region")
}

fn binding(value: Value) -> Binding {
    Binding::data(value, true, true)
}

fn scope(engine: &Engine, value: Value) -> Env {
    let env = new_scope(Some(engine.interp.global_env));
    env.borrow_mut()
        .vars
        .insert("result".to_owned(), binding(value));
    env
}

fn result(env: &Env) -> Value {
    env.borrow().vars.get("result").unwrap().value()
}

#[test]
fn populated_holder_transaction_preserves_aliases_views_and_surrogate_boundary() {
    let mut engine = Engine::new();
    let chunk = chunk();
    let plan = first_plan(&chunk);
    let root = crate::lstr::LStr::from("x".repeat(1024));
    let view = root.slice_units(0, 900);
    let alias = view;
    let env = scope(&engine, Value::Str(view));
    let stored = try_store(&chunk, &mut engine.interp, &env, plan, Value::lstr("!"))
        .expect("ordinary view-backed holder");
    assert_eq!(
        stored.result.unwrap().as_engine_text().unwrap(),
        format!("{}!", "x".repeat(900))
    );
    assert_eq!(alias.text(), "x".repeat(900));
    assert_eq!(root.text(), "x".repeat(1024));
    let old = result(&env);
    let twice = try_store(&chunk, &mut engine.interp, &env, plan, old).unwrap();
    assert_eq!(
        twice.result.unwrap().as_engine_text().unwrap(),
        format!("{0}!{0}!", "x".repeat(900))
    );
    assert_eq!(
        old.as_engine_text().unwrap(),
        format!("{}!", "x".repeat(900))
    );
    env.borrow_mut()
        .vars
        .get_mut("result")
        .unwrap()
        .set_value(Value::from_utf16(&[0xd834]));
    let joined = try_store(
        &chunk,
        &mut engine.interp,
        &env,
        plan,
        Value::from_utf16(&[0xdf06]),
    )
    .unwrap()
    .result
    .unwrap();
    assert_eq!(joined.as_engine_text().unwrap(), "𝌆");
    assert_eq!(result(&env).as_engine_text().unwrap(), "𝌆");
}

#[test]
fn proof_denials_leave_the_original_holder_and_stack_untouched() {
    let mut engine = Engine::new();
    let chunk = chunk();
    let plan = first_plan(&chunk);
    for kind in 0..7 {
        let env = scope(&engine, Value::lstr("prefix"));
        match kind {
            0 => env.borrow_mut().vars.get_mut("result").unwrap().initialized = false,
            1 => env.borrow_mut().vars.get_mut("result").unwrap().mutable = false,
            2 => env.borrow_mut().vars.get_mut("result").unwrap().imported = true,
            3 => env
                .borrow_mut()
                .vars
                .get_mut("result")
                .unwrap()
                .set_import_reference(Some((engine.interp.global_env, "exported".into()))),
            4 => env.borrow_mut().set_with_object(Some(Value::Null)),
            5 => {
                env.borrow_mut().vars.remove("result");
            }
            6 => env.borrow_mut().parent = None,
            _ => unreachable!(),
        }
        let old = env.borrow().vars.get("result").map(|b| b.value());
        assert!(
            try_store(
                &chunk,
                &mut engine.interp,
                &env,
                plan,
                Value::lstr("suffix"),
            )
            .is_none(),
            "denial {kind}"
        );
        if let Some(old) = old {
            assert_eq!(old.as_engine_text(), result(&env).as_engine_text());
        }
    }
    let env = scope(&engine, Value::lstr("prefix"));
    for right in [
        Value::Empty,
        Value::Undefined,
        Value::Null,
        Value::Num(2.0),
        Value::Bool(true),
    ] {
        assert!(try_store(&chunk, &mut engine.interp, &env, plan, right).is_none());
        assert_eq!(result(&env).as_engine_text().unwrap(), "prefix");
    }
    let mut deep = env;
    for _ in 0..MAX_SCOPES {
        deep = new_scope(Some(deep));
    }
    assert!(try_store(&chunk, &mut engine.interp, &deep, plan, Value::lstr("!")).is_none());
    assert_eq!(result(&env).as_engine_text().unwrap(), "prefix");
}

#[test]
fn original_reference_and_coercion_fallbacks_agree_across_all_tiers() {
    let source = r#"
        (function(){
            let result='A', calls=0;
            function add(part){return result+=part;}
            let snapshot=result, first=add('B'), second=add(result);
            let observer={ [Symbol.toPrimitive](){calls++;result='changed';return 'C';} };
            let converted=add(observer);
            let error;try{add(Symbol('s'));}catch(e){error=e.name;}
            let unchanged=result;
            let n=2;function number(x){return n+=x;}
            let numeric=number(3), mixed=number('!');
            let fixed='K', errors=[];
            function constant(x){const target=fixed;return target+=x;}
            try{constant({toString(){calls++;return '!';}});}catch(e){errors.push(e.name);}
            function tdz(x){return hidden+=x;let hidden;}
            try{tdz({toString(){calls+=100;return '!';}});}catch(e){errors.push(e.name);}
            let outer='O';function dynamic(x){return outer+=(eval('var outer="I"'),x);}
            let dynamicValue=dynamic('R');
            return JSON.stringify([snapshot,first,second,converted,calls,error,unchanged,
                numeric,mixed,errors,dynamicValue,outer]);
        })();
    "#;
    let expected = r#"["A","AB","ABAB","ABABC",2,"TypeError","ABABC",5,"5!",["TypeError","ReferenceError"],"OR","OR"]"#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine, source), expected, "{tier:?}");
    }
}

#[test]
fn global_accessors_and_with_resolution_keep_their_observations_all_tiers() {
    let source = r#"
        var log=[],saved='A';
        Object.defineProperty(globalThis,'accumulator',{configurable:true,
            get(){log.push('get');return saved;},set(value){log.push('set:'+value);saved=value;}});
        function globalAdd(part){return accumulator+=part;}
        var first=globalAdd('B');
        var object={result:'W'};
        function inWith(part){with(object){return result+=part;}}
        var second=inWith('!');
        var answer=JSON.stringify([first,saved,second,object.result,log]);
        delete globalThis.accumulator;answer;
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, source),
            r#"["AB","AB","W!","W!",["get","set:AB"]]"#,
            "{tier:?}"
        );
    }
}

#[test]
fn accumulator_surrogate_boundaries_and_literals_agree_all_tiers() {
    let source = r#"
        (function(){let result='';function add(part){return result+=part;}
            let first=add('\ud834'),joined=add('\udf06'),last=add('\ud800');
            return JSON.stringify([first.length,first.charCodeAt(0),joined.length,
                joined.codePointAt(0),last.length,last.charCodeAt(2)]);})();
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(&mut engine, source),
            "[1,55348,2,119558,3,55296]",
            "{tier:?}"
        );
    }
}

const REACT_CALLBACK: &str = r#"
    function create(){let result='';function push(chunk){null!==chunk&&(result+=chunk);return !0;}
        return [push,function(){return result;}];}
    var holder=create(),push=holder[0];
"#;

fn prepare_callback(engine: &mut Engine) -> Rc<Chunk> {
    eval(engine, REACT_CALLBACK);
    let env = engine.interp.global_env;
    let value = engine.interp.get_var("push", &env).ok().unwrap();
    let function = {
        let object_handle = value.as_obj().unwrap();
        let object = object_handle.borrow();
        let crate::value::Callable::User(user) = &object.call else {
            panic!("callback")
        };
        user.func.clone()
    };
    let chunk = bytecode::compile(&function).unwrap();
    let plan = first_plan(&chunk);
    assert!(plan.keep);
    assert_eq!(chunk.n_slots, 1);
    assert_eq!(
        plan.pc, 5,
        "same original operation sequence as React's push callback"
    );
    assert_eq!(chunk.ops.len(), 15);
    chunk.inline_attempted.set(true);
    chunk.inline_retry_at.set(0);
    assert!(function.runtime().code.set(Some(chunk.clone())).is_ok());
    chunk
}

#[test]
fn actual_react_push_callback_commits_vm_and_native_transactions() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let chunk = prepare_callback(&mut engine);
        let vm = TEST_VM_HITS.with(Cell::get);
        let native = TEST_NATIVE_HITS.with(Cell::get);
        assert_eq!(eval(&mut engine,
            "var okay=true;for(var k=0;k<40;k++)okay=okay&&push('x')===true;okay&&push(null)===true&&holder[1]()==='x'.repeat(40);"),"true");
        if tier == Tier::Bytecode {
            assert!(TEST_VM_HITS.with(Cell::get) > vm);
        }
        if tier == Tier::Jit
            && cfg!(all(
                any(target_arch = "aarch64", target_arch = "x86_64"),
                any(
                    target_os = "linux",
                    target_os = "macos",
                    target_os = "windows"
                )
            ))
        {
            assert!(
                TEST_NATIVE_HITS.with(Cell::get) > native,
                "actual callback must commit natively"
            );
            assert!(
                chunk.jit_runs.get() > 0,
                "actual callback body entered native code"
            );
        }
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn opstat_diagnostic_retains_original_checked_operation_observations() {
    if std::env::var_os("TRUST_JS_STRING_ACCUMULATION_OPSTAT_CHILD").is_none() {
        let name="bytecode::string_accumulation::tests::opstat_diagnostic_retains_original_checked_operation_observations";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture"])
            .env("TRUST_JS_STRING_ACCUMULATION_OPSTAT_CHILD", "1")
            .env("TRUST_JS_JIT_OPSTAT", "2")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "exact child must execute: {stdout}\n{stderr}"
        );
        assert!(
            stderr.contains("[jit-opstat]") && stderr.contains("Add @ fn(chunk) pc7"),
            "original checked Add remains attributed: {stderr}"
        );
        return;
    }
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let chunk = prepare_callback(&mut engine);
    let hits = TEST_NATIVE_HITS.with(Cell::get);
    assert_eq!(
        eval(
            &mut engine,
            "for(var k=0;k<40;k++)push('x');holder[1]()==='x'.repeat(40);"
        ),
        "true"
    );
    assert!(chunk.jit_runs.get() > 0);
    assert_eq!(
        TEST_NATIVE_HITS.with(Cell::get),
        hits,
        "profiler keeps original checked operations rather than counting fabricated calls"
    );
}

#[test]
fn committed_assignment_results_survive_later_appends_and_gc_all_tiers() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var make=function(){let result='A';return [function(part){return result+=part;},
                function(){return result;}];};
            var pair=make(),first=pair[0]('B'),second=pair[0]('C');
            JSON.stringify([first,second,pair[1]()]);
        "#
            ),
            r#"["AB","ABC","ABC"]"#
        );
        engine.interp.gc_collect();
        assert_eq!(
            eval(
                &mut engine,
                r#"pair[0]('D');JSON.stringify([first,second,pair[1]()]);"#
            ),
            r#"["AB","ABC","ABCD"]"#,
            "{tier:?}"
        );
    }
}

#[test]
fn accumulation_success_and_observer_fallback_preserve_live_numeric_locals() {
    let source = r#"
        function create(){let result='A';return function append(part,n){
            var left=n+1,right=n*3;result+=part;return [result,left,right];};}
        var append=create(),okay=true;
        for(var k=0;k<40;k++){
            var row=append('x',k);okay=okay&&row[1]===k+1&&row[2]===k*3;
        }
        var calls=0,observer={toString(){calls++;return '!';}},row=append(observer,17);
        JSON.stringify([okay,row[0],row[1],row[2],calls]);
    "#;
    let expected = format!("[true,\"A{}!\",18,51,1]", "x".repeat(40));
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let hits = TEST_NATIVE_HITS.with(Cell::get);
        assert_eq!(eval(&mut engine, source), expected, "{tier:?}");
        if tier == Tier::Jit
            && cfg!(all(
                any(target_arch = "aarch64", target_arch = "x86_64"),
                any(
                    target_os = "linux",
                    target_os = "macos",
                    target_os = "windows"
                )
            ))
        {
            assert!(
                TEST_NATIVE_HITS.with(Cell::get) > hits,
                "actual native success with live locals"
            );
        }
    }
}

#[test]
fn exact_shared_chunk_never_reuses_another_closures_holder() {
    let mut engine = Engine::new();
    let chunk = chunk();
    let plan = first_plan(&chunk);
    let left = scope(&engine, Value::lstr("left"));
    let right = scope(&engine, Value::lstr("right"));
    for _ in 0..8 {
        try_store(&chunk, &mut engine.interp, &left, plan, Value::lstr("L")).unwrap();
        try_store(&chunk, &mut engine.interp, &right, plan, Value::lstr("R")).unwrap();
    }
    assert_eq!(result(&left).as_engine_text().unwrap(), "leftLLLLLLLL");
    assert_eq!(result(&right).as_engine_text().unwrap(), "rightRRRRRRRR");
}

#[test]
fn shared_snapshot_native_code_uses_each_realms_live_declarative_holder() {
    use crate::realm_inline_guard_tests::{eval_string, in_realm};
    static SNAPSHOT: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let snapshot = SNAPSHOT.get_or_init(|| {
        crate::compile_host_snapshot(
            r#"
        (function(){let result=realmTag;
            globalThis.appendPart=function(part){return result+=part;};
            globalThis.readResult=function(){return result;};})();
    "#,
        )
        .unwrap()
    });
    fn append_function(i: &mut Interp) -> Rc<Function> {
        let env = i.global_env;
        let value = i.get_var("appendPart", &env).ok().unwrap();
        let object_handle = value.as_obj().unwrap();
        let object = object_handle.borrow();
        let crate::value::Callable::User(user) = &object.call else {
            panic!("shared function")
        };
        user.func.clone()
    }
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let i = &mut *engine.interp;
        let child = i.create_realm();
        let body = i.shared_snapshot_program(snapshot).unwrap();
        eval_string(i, "var realmTag='M';");
        assert!(i.run_classic_program(&body, None).is_ok());
        assert_eq!(
            eval_string(i, "for(var k=0;k<40;k++)appendPart('x');readResult();"),
            format!("M{}", "x".repeat(40))
        );
        let main = append_function(i);
        in_realm(i, &child, |i| {
            eval_string(i, "var realmTag='C';");
            assert!(i.run_classic_program(&body, None).is_ok());
            assert!(
                Rc::ptr_eq(&main, &append_function(i)),
                "shared immutable AST/code"
            );
            assert_eq!(
                eval_string(i, "for(var k=0;k<40;k++)appendPart('y');readResult();"),
                format!("C{}", "y".repeat(40))
            );
        });
        assert_eq!(
            eval_string(i, "appendPart('!');readResult();"),
            format!("M{}!", "x".repeat(40))
        );
    }
}

#[test]
fn length_denial_preserves_binding_and_throws_at_original_add() {
    let mut engine = Engine::new();
    let chunk = chunk();
    let plan = first_plan(&chunk);
    let holder = scope(
        &engine,
        Value::from_engine_text("x".repeat(MAX_STR_LEN - 1)),
    );
    assert!(try_store(&chunk, &mut engine.interp, &holder, plan, Value::lstr("xx")).is_none());
    assert_eq!(
        result(&holder).as_engine_text().unwrap().len(),
        MAX_STR_LEN - 1
    );
    let mut state = bytecode::VmCoro::new(
        &mut engine.interp,
        chunk.clone(),
        holder,
        Value::Undefined,
        &[Value::lstr("xx")],
        &[],
    );
    let failure = bytecode::run_vm(
        &mut engine.interp,
        &chunk,
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
        None,
        None,
    );
    assert!(matches!(failure, Err(crate::interpreter::Abrupt::Throw(_))));
    assert_eq!(
        state.pc,
        plan.pc + 3,
        "fallback reports the original Add boundary"
    );
    assert_eq!(
        result(&holder).as_engine_text().unwrap().len(),
        MAX_STR_LEN - 1
    );
}

fn prefix(engine: &mut Engine, state: &mut bytecode::VmCoro, stop: usize) {
    assert!(matches!(
        bytecode::run_vm(
            &mut engine.interp,
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
            None
        ),
        Ok(bytecode::VmRunExit::Step(bytecode::VmStep::Done(value))) if matches!(value.view(), ValueView::Undefined)
    ));
    assert_eq!(state.pc, stop);
}

fn continuation(engine: &mut Engine) -> (bytecode::VmCoro, Env, Plan) {
    let chunk = bytecode::compile(&function(
        "function* subject(part){yield result+=part;return result;}",
    ))
    .unwrap();
    let plan = first_plan(&chunk);
    let holder = scope(engine, Value::lstr("A"));
    let state = bytecode::VmCoro::new_generator(
        &mut engine.interp,
        chunk,
        holder,
        Value::Undefined,
        &[Value::lstr("B")],
        &[],
    );
    (state, holder, plan)
}

#[test]
fn every_original_vm_interior_stop_exposes_ordinary_state_before_commit() {
    for interior in 1..=4 {
        let mut engine = Engine::new();
        let (mut state, holder, plan) = continuation(&mut engine);
        let hits = TEST_VM_HITS.with(Cell::get);
        prefix(&mut engine, &mut state, plan.pc + interior);
        assert_eq!(
            TEST_VM_HITS.with(Cell::get),
            hits,
            "interior stop disables atomic region"
        );
        assert_eq!(result(&holder).as_engine_text().unwrap(), "A");
        assert_eq!(
            state.stack.len(),
            match interior {
                1 => 1,
                2 => 2,
                3 => 1,
                4 => 2,
                _ => unreachable!(),
            }
        );
        prefix(&mut engine, &mut state, plan.end);
        assert_eq!(result(&holder).as_engine_text().unwrap(), "AB");
        assert_eq!(state.stack.last().unwrap().as_engine_text().unwrap(), "AB");
    }
    let mut engine = Engine::new();
    let (mut state, holder, plan) = continuation(&mut engine);
    let hits = TEST_VM_HITS.with(Cell::get);
    prefix(&mut engine, &mut state, plan.end);
    assert_eq!(TEST_VM_HITS.with(Cell::get), hits + 1);
    assert_eq!(result(&holder).as_engine_text().unwrap(), "AB");
}

#[test]
fn complete_no_dup_region_commits_without_creating_an_expression_owner() {
    let mut engine = Engine::new();
    let mut chunk =
        bytecode::compile(&function("function* subject(part){yield result+=part;}")).unwrap();
    let original = first_plan(&chunk);
    chunk = Chunk::rewrite_owned_for_test(chunk, |chunk| {
        // Undef supplies the following Yield's independently specified value.
        chunk.ops[original.pc + 3] = chunk.ops[original.pc + 4];
        chunk.ops[original.pc + 4] = Op::Undef;
    });
    let plan = first_plan(&chunk);
    assert!(!plan.keep);
    let holder = scope(&engine, Value::lstr("A"));
    let mut state = bytecode::VmCoro::new_generator(
        &mut engine.interp,
        chunk,
        holder,
        Value::Undefined,
        &[Value::lstr("B")],
        &[],
    );
    prefix(&mut engine, &mut state, plan.end);
    assert!(state.stack.is_empty());
    assert_eq!(result(&holder).as_engine_text().unwrap(), "AB");
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "linux", target_os = "macos", target_os = "windows")
    ))]
    {
        let chunk = state.chunk.clone();
        state.pc = plan.pc;
        holder
            .borrow_mut()
            .vars
            .get_mut("result")
            .unwrap()
            .set_value(Value::lstr("C"));
        let code = crate::jit::continuation_code(&mut engine.interp, &chunk).unwrap();
        let hits = TEST_NATIVE_HITS.with(Cell::get);
        let mut native = bytecode::NativeContinuation {
            chunk: &*chunk,
            env: &mut state.env,
            cap_env: &state.cap_env,
            references: &mut state.references[..],
            stack: &mut state.stack,
            disposal_frames: &mut state.disposal_frames,
            class_states: &mut state.class_states[..],
        };
        let step = unsafe {
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
        }
        .unwrap_or_else(|_| panic!("native continuation must complete without error"));
        assert!(
            matches!(step, bytecode::VmStep::Yield(value) if matches!(value.view(), ValueView::Undefined))
        );
        assert_eq!(TEST_NATIVE_HITS.with(Cell::get), hits + 1);
        assert_eq!(result(&holder).as_engine_text().unwrap(), "CB");
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn borrowed_native_every_original_interior_resumes_without_replaying_the_prefix() {
    for interior in 1..=4 {
        let mut engine = Engine::new();
        let (mut state, holder, plan) = continuation(&mut engine);
        prefix(&mut engine, &mut state, plan.pc + interior);
        let chunk = state.chunk.clone();
        let code = crate::jit::continuation_code(&mut engine.interp, &chunk).unwrap();
        let hits = TEST_NATIVE_HITS.with(Cell::get);
        let mut native = bytecode::NativeContinuation {
            chunk: &*chunk,
            env: &mut state.env,
            cap_env: &state.cap_env,
            references: &mut state.references[..],
            stack: &mut state.stack,
            disposal_frames: &mut state.disposal_frames,
            class_states: &mut state.class_states[..],
        };
        let step = unsafe {
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
        }
        .unwrap_or_else(|_| panic!("native continuation must complete without error"));
        assert!(
            matches!(step,bytecode::VmStep::Yield(value) if value.as_engine_text().as_deref()==Some("AB"))
        );
        assert_eq!(result(&holder).as_engine_text().unwrap(), "AB");
        assert_eq!(
            TEST_NATIVE_HITS.with(Cell::get),
            hits,
            "interior executes original suffix"
        );
        assert!(code.test_residency_marks(false).1 > 0);
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn borrowed_native_preentry_interrupt_leaves_binding_then_commits_before_yield() {
    let mut engine = Engine::new();
    let (mut state, holder, plan) = continuation(&mut engine);
    prefix(&mut engine, &mut state, plan.pc);
    let chunk = state.chunk.clone();
    let code = crate::jit::continuation_code(&mut engine.interp, &chunk).unwrap();
    let hits = TEST_NATIVE_HITS.with(Cell::get);
    let mut native = bytecode::NativeContinuation {
        chunk: &*chunk,
        env: &mut state.env,
        cap_env: &state.cap_env,
        references: &mut state.references[..],
        stack: &mut state.stack,
        disposal_frames: &mut state.disposal_frames,
        class_states: &mut state.class_states[..],
    };
    engine.interrupt_handle().request_user_navigation();
    let failure = unsafe {
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
        failure,
        Err(crate::interpreter::Abrupt::Interrupt(
            crate::InterruptReason::UserNavigation
        ))
    ));
    assert_eq!(TEST_NATIVE_HITS.with(Cell::get), hits);
    assert_eq!(result(&holder).as_engine_text().unwrap(), "A");
    engine.interrupt_handle().begin_user_interaction();
    let step = unsafe {
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
    }
    .unwrap_or_else(|_| panic!("native continuation must complete without error"));
    assert!(
        matches!(step,bytecode::VmStep::Yield(value) if value.as_engine_text().as_deref()==Some("AB"))
    );
    assert_eq!(TEST_NATIVE_HITS.with(Cell::get), hits + 1);
    assert_eq!(result(&holder).as_engine_text().unwrap(), "AB");
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn actual_checked_numeric_chain_h_exec_replay_commits_string_transaction() {
    if std::env::var_os("TRUST_JS_STRING_ACCUMULATION_REPLAY_CHILD").is_none() {
        let name="bytecode::string_accumulation::tests::actual_checked_numeric_chain_h_exec_replay_commits_string_transaction";
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture"])
            .env("TRUST_JS_STRING_ACCUMULATION_REPLAY_CHILD", "1")
            .env("TRUST_JS_JIT_FAST", "24576")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "exact child must execute: {stdout}\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            TEST_REPLAY_ONLY.with(|v| v.set(false));
        }
    }
    TEST_REPLAY_ONLY.with(|v| v.set(true));
    let _reset = Reset;
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let chunk = prepare_callback(&mut engine);
    let hits = TEST_NATIVE_HITS.with(Cell::get);
    assert_eq!(
        eval(
            &mut engine,
            "for(var k=0;k<40;k++)push('x');holder[1]()==='x'.repeat(40);"
        ),
        "true"
    );
    assert!(chunk.jit_runs.get() > 0);
    assert!(
        TEST_NATIVE_HITS.with(Cell::get) > hits,
        "all direct attempts suppressed: only actual H_EXEC replay can commit"
    );
}
