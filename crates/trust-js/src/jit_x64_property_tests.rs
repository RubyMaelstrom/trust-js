//! ECMA-262 e28783d5 Get/OrdinaryGet: native own-data reads retain the live result;
//! accessors and exotic receivers retain their observable [[Get]] algorithm.
use super::*;
use crate::value::Value;
use crate::value::ValueView;
use crate::{bytecode::Tier, Engine};
use std::rc::Rc;

fn compile(engine: &mut Engine, source: &str) -> (Rc<Chunk>, JitCode) {
    let statements = crate::parser::parse_script(source, false)
        .ok()
        .expect("fixture parses");
    let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
        panic!("function fixture")
    };
    let chunk = crate::bytecode::compile(function).expect("fixture bytecode");
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let code = crate::jit::compile(&chunk, &layout, &ilayout).expect("fixture native code");
    (chunk, code)
}

fn eval(engine: &mut Engine, source: &str) -> Value {
    let env = engine.interp.global_env;
    engine
        .interp
        .perform_eval(source, &env, false)
        .unwrap_or_else(|_| panic!("fixture throws: {source}"))
}

fn reads(index: usize) -> usize {
    TEST_OWN_PROPERTY_READS.with(|counts| counts[index].get())
}

fn run(
    engine: &mut Engine,
    chunk: &Rc<Chunk>,
    code: &JitCode,
    receiver: &Value,
    this: bool,
) -> Value {
    let env = engine.interp.global_env;
    let (receiver_this, args) = if this {
        (*receiver, Vec::new())
    } else {
        (Value::Undefined, vec![*receiver])
    };
    crate::jit::run(&mut engine.interp, chunk, code, &env, receiver_this, &args)
        .unwrap_or_else(|_| panic!("native read throws"))
}

#[test]
fn own_data_reads_warm_after_native_compilation_and_clone_live_value_kinds() {
    for (this, source) in [
        (false, "function read(receiver){return receiver.value;}"),
        (true, "function read(){'use strict';return this.value;}"),
    ] {
        let mut engine = Engine::new();
        // Compile before any cache fill. Later fills must be read by the existing machine body.
        let (chunk, code) = compile(&mut engine, source);
        let receiver = Value::Obj(engine.interp.new_object());
        engine
            .interp
            .set_member_key(&receiver, "value", Value::Num(1.0))
            .unwrap_or_else(|_| panic!("fixture operation throws"));
        assert!(matches!(
            (run(&mut engine, &chunk, &code, &receiver, this)).view(),
            crate::value::ValueView::Num(1.0)
        ));
        let values = eval(&mut engine,
            "[undefined,null,false,true,0,-0,NaN,Infinity,'owned string',Symbol('owned'),{identity:42},12345678901234567890n]");
        let before = reads(usize::from(this));
        for index in 0..12 {
            let expected = engine
                .interp
                .get_member_key(&values, &index.to_string())
                .unwrap_or_else(|_| panic!("fixture operation throws"));
            engine
                .interp
                .set_member_key(&receiver, "value", expected)
                .unwrap_or_else(|_| panic!("fixture operation throws"));
            let result = run(&mut engine, &chunk, &code, &receiver, this);
            assert!(crate::builtins::same_value_pub(&result, &expected));
            // Removing the property owner must leave the returned value intact through collection.
            engine
                .interp
                .set_member_key(&receiver, "value", Value::Undefined)
                .unwrap_or_else(|_| panic!("fixture operation throws"));
            engine.interp.collect_garbage_for_host();
            assert!(crate::builtins::same_value_pub(&result, &expected));
        }
        assert_eq!(
            reads(usize::from(this)) - before,
            12,
            "all ordinary execution values use native reads"
        );
    }
}

/// The template finds a named field in place and spilled (an object's fifth field spills its
/// fields from the cell to a heap buffer), from a local slot's receiver and from `this`, the
/// packed word at [48].
#[test]
fn own_data_reads_find_fields_in_place_and_spilled() {
    for (this, source) in [
        (false, "function read(receiver){return receiver.value;}"),
        (true, "function read(){'use strict';return this.value;}"),
    ] {
        // `value` is the last of `count` fields.
        for count in [1usize, 4, 5, 9] {
            let mut engine = Engine::new();
            let (chunk, code) = compile(&mut engine, source);
            let fields: String = (1..count).map(|k| format!("f{k}:{k},")).collect();
            let receiver = eval(&mut engine, &format!("({{{fields}value:{count}}})"));
            run(&mut engine, &chunk, &code, &receiver, this);
            let before = reads(usize::from(this));
            let result = run(&mut engine, &chunk, &code, &receiver, this);
            assert!(
                matches!((result).view(), ValueView::Num(n) if n == count as f64),
                "{count} fields (this: {this})"
            );
            assert_eq!(
                reads(usize::from(this)) - before,
                1,
                "{count} fields (this: {this}): the read is native"
            );
        }
    }
}

#[test]
fn live_polymorphic_own_cache_ways_keep_each_receivers_storage() {
    let mut engine = Engine::new();
    let (chunk, code) = compile(
        &mut engine,
        "function read(receiver){return receiver.value;}",
    );
    let receivers = eval(
        &mut engine,
        "[{value:'a'},{first:1,value:'b'},{value:'c',last:1},{first:1,second:2,value:'d'}]",
    );
    let objects: Vec<_> = (0..4)
        .map(|index| {
            engine
                .interp
                .get_member_key(&receivers, &index.to_string())
                .unwrap_or_else(|_| panic!("fixture operation throws"))
        })
        .collect();
    for receiver in &objects {
        run(&mut engine, &chunk, &code, receiver, false);
    }
    let before = reads(0);
    for _ in 0..4 {
        for (index, receiver) in objects.iter().enumerate() {
            let result = run(&mut engine, &chunk, &code, receiver, false);
            assert!(crate::builtins::same_value_pub(
                &result,
                &Value::str(["a", "b", "c", "d"][index])
            ));
        }
    }
    assert_eq!(
        reads(0) - before,
        16,
        "every warmed way must enter native code"
    );
}

#[test]
fn descriptor_exotic_shape_and_value_changes_preserve_get_effects_across_tiers() {
    let source = r#"
        function read(receiver){return receiver.value;}
        function readThis(){'use strict';return this.value;}
        var o={value:'first'},good=true;
        for(var i=0;i<20;i++)good=good&&read(o)==='first'&&readThis.call(o)==='first';
        o.value={identity:1};
        good=good&&read(o)===o.value&&readThis.call(o)===o.value;
        var gets=0,token={};
        Object.defineProperty(o,'value',{get:function(){gets++;return this===o?'getter':'wrong';},configurable:true});
        good=good&&read(o)==='getter'&&readThis.call(o)==='getter'&&gets===2;
        Object.defineProperty(o,'value',{get:function(){gets++;throw token;},configurable:true});
        try{read(o);good=false;}catch(e){good=good&&e===token;}
        try{readThis.call(o);good=false;}catch(e){good=good&&e===token;}
        delete o.value;
        var proto={value:'inherited'};Object.setPrototypeOf(o,proto);
        good=good&&read(o)==='inherited'&&readThis.call(o)==='inherited';
        o.value='restored';o.extra=1;
        good=good&&read(o)==='restored'&&readThis.call(o)==='restored';
        var traps=0,proxy=new Proxy(o,{get:function(target,key,receiver){traps++;return receiver===proxy?'proxy':'wrong';}});
        good=good&&read(proxy)==='proxy'&&readThis.call(proxy)==='proxy'&&traps===2;
        var revoked=Proxy.revocable(o,{});revoked.revoke();
        try{read(revoked.proxy);good=false;}catch(e){good=good&&e instanceof TypeError;}
        var missing={};good=good&&read(missing)===undefined;
        function tdz(){return before.value;let before={value:1};}
        try{tdz();good=false;}catch(e){good=good&&e instanceof ReferenceError;}
        good&&gets===4;
    "#;
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let before = reads(0) + reads(1);
        assert!(
            matches!(
                (eval(&mut engine, source)).view(),
                crate::value::ValueView::Bool(true)
            ),
            "{tier:?}"
        );
        if tier == Tier::Jit {
            assert!(
                reads(0) + reads(1) > before,
                "semantic fixture must use the native own-data path"
            );
        }
    }
}

#[test]
fn one_native_chunk_reads_live_receivers_in_each_realm() {
    use crate::realm_inline_guard_tests::{eval_string, in_realm};
    let mut engine = Engine::new();
    let (chunk, code) = compile(
        &mut engine,
        "function read(receiver){return receiver.value;}",
    );
    let first = eval(&mut engine, "({value:{realm:'main'}})");
    run(&mut engine, &chunk, &code, &first, false);
    let child = engine.interp.create_realm();
    let child_receiver = in_realm(&mut engine.interp, &child, |it| {
        assert_eq!(
            eval_string(
                it,
                "globalThis.receiver={value:{realm:'child'}};receiver.value.realm"
            ),
            "child"
        );
        let env = it.global_env;
        it.get_var("receiver", &env)
            .unwrap_or_else(|_| panic!("fixture operation throws"))
    });
    run(&mut engine, &chunk, &code, &child_receiver, false);
    for receiver in [&first, &child_receiver, &first] {
        let expected = engine
            .interp
            .get_member_key(receiver, "value")
            .unwrap_or_else(|_| panic!("fixture operation throws"));
        let before = reads(0);
        let result = in_realm(&mut engine.interp, &child, |it| {
            let env = it.global_env;
            crate::jit::run(
                it,
                &chunk,
                &code,
                &env,
                Value::Undefined,
                std::slice::from_ref(receiver),
            )
            .unwrap_or_else(|_| panic!("fixture operation throws"))
        });
        assert!(crate::builtins::same_value_pub(&result, &expected));
        assert_eq!(reads(0), before + 1);
    }
}
