//! Count-free overwrites retain property/read semantics (ECMA-262 e28783d5,
//! OrdinarySetWithOwnDescriptor / GetValue / ToNumeric).
use crate::bytecode::Tier;
use crate::value::Callable;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).unwrap() {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

fn native(engine: &mut Engine, name: &str) {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().unwrap();
    let borrowed = value.as_obj().unwrap().borrow();
    let Callable::User(user) = &borrowed.call else {
        panic!("writer")
    };
    let runtime = user.func.runtime();
    let chunk = runtime.code.get().and_then(Option::as_ref).unwrap();
    assert!(chunk.jit.get().flatten().is_some());
    assert!(chunk.jit_runs.get() > 0);
}

fn element_helpers() -> (usize, usize) {
    (
        crate::bytecode::TEST_JIT_SET_ELEM_HELPERS.with(|count| count.get()),
        crate::bytecode::TEST_JIT_EXEC_ELEMENT_HELPERS.with(|count| count.get()),
    )
}

#[test]
fn numeric_element_stores_replace_all_displaced_kinds_without_element_helpers() {
    for (definition, call) in [
        (
            "function sink(){return target;}function writer(v){sink()[0]=v;return 0;}",
            "writer(7)",
        ),
        (
            "function writer(array,index,v){array[index]=v;return 0;}",
            "writer(target,0,7)",
        ),
        (
            "function writer(array,index,v){array[index]=v+1;return 0;}",
            "writer(target,0,6)",
        ),
    ] {
        for initial in ["[0]", "{0:0,named:1}"] {
            let mut engine = engine();
            eval(
                &mut engine,
                &format!("var target={initial};{definition}for(var k=0;k<80;k++){{{call};}}0"),
            );
            native(&mut engine, "writer");
            for displaced in [
                "undefined",
                "null",
                "false",
                "'text'",
                "Symbol('old')",
                "({old:1})",
                "12345678901234567890n",
            ] {
                eval(&mut engine, &format!("target[0]={displaced};0"));
                let before = element_helpers();
                assert_eq!(eval(&mut engine, call), "0");
                assert_eq!(element_helpers(), before, "{definition}: displaced {displaced} must not enter SetElem or exec-element helpers");
                assert_eq!(eval(&mut engine, "target[0]"), "7");
            }
        }
    }
}

#[test]
fn numeric_property_chain_replaces_a_reference_without_property_or_exec_helpers() {
    let mut engine = engine();
    // The chain consumes property feedback at native compilation, so fill that feedback first.
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, "var target={value:0};function writer(receiver,x){receiver.value=x+1;return 0;}for(var k=0;k<80;k++)writer(target,6);0");
    engine.set_tier(Tier::Jit);
    eval(&mut engine, "writer(target,6)");
    native(&mut engine, "writer");
    for displaced in [
        "'old'",
        "Symbol('old')",
        "({old:1})",
        "12345678901234567890n",
    ] {
        eval(&mut engine, &format!("target.value={displaced};0"));
        let before = (
            crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(|count| count.get()),
            crate::bytecode::TEST_JIT_EXEC_PROPERTY_HELPERS.with(|count| count.get()),
            element_helpers(),
        );
        let commits = crate::jit::TEST_CHAIN_PROPERTY_STORES.with(|count| count.get());
        assert_eq!(eval(&mut engine, "writer(target,6)"), "0");
        assert!(
            crate::jit::TEST_CHAIN_PROPERTY_STORES.with(|count| count.get()) > commits,
            "displaced {displaced} must take the committed ChainOp::StoreProp path"
        );
        assert_eq!(
            (
                crate::bytecode::TEST_JIT_SET_PROP_HELPERS.with(|count| count.get()),
                crate::bytecode::TEST_JIT_EXEC_PROPERTY_HELPERS.with(|count| count.get()),
                element_helpers()
            ),
            before,
            "numeric chain store must not inspect displaced {displaced}"
        );
        assert_eq!(eval(&mut engine, "target.value"), "7");
    }
}

#[test]
fn holes_still_call_inherited_setters_and_nonnumeric_reads_keep_conversion() {
    let mut engine = engine();
    eval(&mut engine, "var target=[0],source=[1];function write(array,index,value){array[index]=value;}function mixed(out,input){out[0]=input[0]+1;return out[0];}for(var k=0;k<80;k++){write(target,0,7);mixed(target,source);}0");
    native(&mut engine, "write");
    native(&mut engine, "mixed");
    assert_eq!(eval(&mut engine, "var calls=0;var parent=[];Object.defineProperty(parent,'0',{set(value){calls+=value;}});delete target[0];Object.setPrototypeOf(target,parent);write(target,0,7);calls+':'+Object.hasOwn(target,'0')"), "7:false");
    assert_eq!(
        eval(
            &mut engine,
            "target=[0];source=['text'];mixed(target,source)"
        ),
        "text1"
    );
    assert_eq!(eval(&mut engine, "var conversions=0;source=[{valueOf(){conversions++;return 41;}}];mixed(target,source)+':'+conversions"), "42:1");
}
