//! Actual x64 machine-entry proofs; run under native x64 or the isolated QEMU runner.
//! ECMA-262 e28783d5 GetIdentifierReference/GetValue/PutValue and ToNumeric.
use super::*;
use crate::interpreter::{new_binding_layout_id, new_scope, Binding, Env};
use crate::value::Value;
use crate::{bytecode::Tier, Completion, Engine};
use std::rc::Rc;

fn compile(engine: &mut Engine, source: &str) -> (Rc<Chunk>, JitCode) {
    let statements = crate::parser::parse_script(source, false)
        .ok()
        .expect("source parses");
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function")
    };
    let chunk = crate::bytecode::compile(function).expect("bytecode");
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    assert!(valid(&layout), "validated native name layout");
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = crate::jit::compile(&chunk, &layout, &ilayout).expect("native code");
    (chunk, code)
}

fn scope(parent: Option<Env>, key: &str, value: f64, layout: u32) -> Env {
    let scope = new_scope(parent);
    scope
        .borrow_mut()
        .vars
        .insert(key, Binding::data(Value::Num(value), true, true));
    scope.borrow_mut().vars.publish_layout(layout);
    scope
}

fn run(engine: &mut Engine, chunk: &Rc<Chunk>, code: &JitCode, env: &Env) -> Value {
    crate::jit::run(&mut engine.interp, chunk, code, env, Value::Undefined, &[])
        .unwrap_or_else(|_| panic!("native fixture throws"))
}

fn number(value: Value) -> f64 {
    match value {
        Value::Num(number) => number,
        _ => panic!("expected Number"),
    }
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

#[test]
fn native_deep_names_use_fresh_holder_layouts_and_live_flags() {
    let mut engine = Engine::new();
    let (chunk, code) = compile(&mut engine, "function f(){return outer;}");
    let layouts = [
        new_binding_layout_id(),
        new_binding_layout_id(),
        new_binding_layout_id(),
    ];
    let holder = scope(None, "outer", 11.0, layouts[0]);
    let child = scope(
        Some(scope(Some(holder.clone()), "middle", 0.0, layouts[1])),
        "local",
        0.0,
        layouts[2],
    );
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 11.0);
    let before = TEST_NATIVE_DEEP_NAMES.with(std::cell::Cell::get);
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 11.0);
    let other = scope(None, "outer", 29.0, layouts[0]);
    let fresh = scope(
        Some(scope(Some(other.clone()), "middle", 0.0, layouts[1])),
        "local",
        0.0,
        layouts[2],
    );
    assert_eq!(number(run(&mut engine, &chunk, &code, &fresh)), 29.0);
    assert_eq!(
        TEST_NATIVE_DEEP_NAMES.with(std::cell::Cell::get),
        before + 2
    );
    let operations = TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get);
    other
        .borrow_mut()
        .vars
        .get_mut("outer")
        .unwrap()
        .initialized = false;
    assert!(crate::jit::run(
        &mut engine.interp,
        &chunk,
        &code,
        &fresh,
        Value::Undefined,
        &[]
    )
    .is_err());
    assert_eq!(
        TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get),
        operations,
        "TDZ is not a fast read"
    );
    other
        .borrow_mut()
        .vars
        .get_mut("outer")
        .unwrap()
        .initialized = true;
    other.borrow_mut().vars.set_generation_for_test(u32::MAX);
    assert_eq!(number(run(&mut engine, &chunk, &code, &fresh)), 29.0);
    assert_eq!(
        TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get),
        operations,
        "exhausted generation must not publish a raw address proof"
    );
}

#[test]
fn native_deep_globals_load_store_and_all_numeric_update_kinds() {
    let mut engine = Engine::new();
    engine.eval("var outer=0;", false).unwrap();
    let (chunk, code) = compile(&mut engine,
        "function f(){outer=10;let a=outer++;let b=++outer;let c=outer--;let d=--outer;outer++;outer--;return a+b+c+d+outer;}");
    let child = new_scope(Some(new_scope(Some(engine.interp.global_env.clone()))));
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 54.0);
    let before = TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get);
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 54.0);
    assert!(
        TEST_NATIVE_NAME_OPS.with(std::cell::Cell::get) >= before + 7,
        "numeric update/load operations must enter native name templates"
    );
    check("var value=-0;function f(){let a=value++;value=Infinity;let b=--value;value=NaN;value++;return [1/a,b,Number.isNaN(value)].join('|')}f()",
        "-Infinity|Infinity|true");
}

#[test]
fn native_name_wide_and_packed_owner_transfers_and_last_owner_fallback() {
    check(
        r#"
        var shared, values=[undefined,null,false,true,-0,NaN,Infinity,-Infinity,42.25,
            'text',Symbol('same'),{identity:7},12345678901234567890n];
        function global(value){shared=value;return shared;}
        function make(){let held;return function(value){held=value;return held;};}
        var capture=make(),okay=true;
        for(let j=0;j<100;j++)for(let value of values){
            if(!Object.is(global(value),value)||!Object.is(capture(value),value))okay=false;
        }
        for(let j=0;j<100;j++){global({last:j});capture({last:j});}
        [okay,global('last'),capture('last')].join('|');
    "#,
        "true|last|last",
    );
}

#[test]
fn native_name_import_mutability_and_parent_changes_stay_live() {
    let mut engine = Engine::new();
    let (chunk, code) = compile(&mut engine, "function f(){return outer;}");
    let holder = scope(None, "outer", 1.0, new_binding_layout_id());
    let middle = new_scope(Some(holder.clone()));
    let child = new_scope(Some(middle.clone()));
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 1.0);
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 1.0);
    let exporter = scope(None, "source", 23.0, new_binding_layout_id());
    holder
        .borrow_mut()
        .vars
        .get_mut("outer")
        .unwrap()
        .set_import_reference(Some((exporter.clone(), "source".into())));
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 23.0);
    exporter.borrow_mut().vars.get_mut("source").unwrap().value = Value::Num(24.0);
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 24.0);
    let replacement = scope(None, "outer", 31.0, new_binding_layout_id());
    middle.borrow_mut().parent = Some(replacement);
    assert_eq!(number(run(&mut engine, &chunk, &code, &child)), 31.0);
    check(
        "const x=1;function f(){x=2;}var output;try{f()}catch(e){output=e.name;}output",
        "TypeError",
    );
}

#[test]
fn native_references_hot_fragment_reuses_capture_load_and_store() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(32);
    TEST_NATIVE_REFERENCES.with(|counts| counts.iter().for_each(|count| count.set(0)));
    crate::jit::TEST_OSR_ENTRIES.with(|count| count.set(0));
    crate::bytecode::TEST_REFERENCE_HELPERS.with(|count| count.set(0));
    assert!(matches!(engine.eval(
        "var total=0;for(var index=0;index<2000;index++)total+=index;total;", false),
        Ok(Completion::Value(value)) if value=="1999000"));
    assert!(crate::jit::TEST_OSR_ENTRIES.with(std::cell::Cell::get) > 0);
    TEST_NATIVE_REFERENCES.with(|counts| {
        for count in counts {
            assert!(
                count.get() > 100,
                "actual native capture/load/store required"
            );
        }
    });
    assert!(
        crate::bytecode::TEST_REFERENCE_HELPERS.with(std::cell::Cell::get) < 16,
        "warm canonical references must not repeat slow-path walks"
    );
}

#[test]
fn native_references_preserve_rhs_coercion_tdz_and_saved_base() {
    check(
        r#"
        var x='outer', trace=[], box={x:1};
        function deleted(){return eval('var x=1;x+=(delete x,4);x;');}
        var first=deleted();
        with(box){x+=(delete box.x,Object.defineProperty(box,Symbol.unscopables,{value:{x:true}}),4);}
        try { t=(trace.push('rhs'),1);let t; } catch(e){trace.push(e.name);}
        const c=1;try{c=(trace.push('const'),2);}catch(e){trace.push(e.name);}
        var coercing={x:{valueOf(){delete coercing.x;return 8;}}};
        with(coercing){x++;}
        [first,box.x,coercing.x,x,trace.join(',')].join('|');
    "#,
        "5|5|9|outer|rhs,ReferenceError,const,TypeError",
    );
}

#[test]
fn native_references_keep_heap_owners_across_collection_and_hint_misses() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(32);
        engine.interp.def_method(
            &engine.interp.global,
            "collectReferenceTest",
            0,
            |interp, _, _| {
                interp.gc_collect();
                Ok(Value::Undefined)
            },
        );
        let source = r#"
            var values=[undefined,null,true,1,-0,NaN,'a',Symbol(),{},1n],held,count=0;
            for(var j=0;j<2000;j++){
                held=values[j%values.length];count++;
                if(j%200===0)collectReferenceTest();
            }
            [count,typeof held].join('|');
        "#;
        assert!(
            matches!(engine.eval(source,false),Ok(Completion::Value(value)) if value=="2000|bigint"),
            "{tier:?}"
        );
    }
}
