//! Warm own-data stores: bounded compile IR, actual native entry and live
//! OrdinarySetWithOwnDescriptor/ownership invalidators (ECMA-262 e28783d5).

use super::*;
use crate::{bytecode::Tier, feedback::ValueClass, value::Callable, Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("store fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn warmed(source: &str, warm: &str) -> (Engine, Rc<crate::ast::Function>, Rc<Chunk>) {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    eval(&mut engine, warm);
    let env = engine.interp.global_env.clone();
    let subject = engine.interp.get_var("subject", &env).ok().unwrap();
    let function = {
        let object = subject.as_obj().unwrap().borrow();
        let Callable::User(user) = &object.call else {
            panic!("author function")
        };
        user.func.clone()
    };
    let chunk = function
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap()
        .clone();
    (engine, function, chunk)
}

fn install(engine: &mut Engine, chunk: Rc<Chunk>, code: JitCode) -> Rc<JitCode> {
    let code = Rc::new(code);
    assert!(code.pc_offsets.is_empty(), "actual optimizing code");
    chunk
        .jit
        .begin_compile()
        .unwrap()
        .commit(Some(code.clone()));
    engine.set_tier(Tier::Jit);
    code.residency.referenced.set(0);
    code
}

fn fixture(source: &str, warm: &str, run: &str, expected: &str) -> usize {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut oracle = Engine::new();
        oracle.set_tier(tier);
        oracle.set_tier_threshold(0);
        eval(&mut oracle, source);
        eval(&mut oracle, warm);
        assert_eq!(eval(&mut oracle, run), expected, "{tier:?}");
        oracle.interp.gc_collect();
    }
    let (mut engine, _function, chunk) = warmed(source, warm);
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    stores::TEST_COMPACT_STORES.with(|count| count.set(0));
    let compiled = compile_with_deopt(&chunk, &values, &layout, None).expect("generic optimizer");
    let sites = stores::TEST_COMPACT_STORES.with(std::cell::Cell::get);
    let code = install(&mut engine, chunk, compiled);
    engine.interp.gc_collect();
    assert_eq!(eval(&mut engine, run), expected);
    assert!(
        code.residency.referenced.get() > 0,
        "native subject entered"
    );
    assert_eq!(code.residency.active.get(), 0);
    assert_eq!(engine.interp.depth, 0);
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
    sites
}

const HOT_SOURCE: &str = "function subject(o,n){'use strict';o.x=n;return n+n;}var receiver={x:0};";

fn compact_active() -> bool {
    [
        "TRUST_JS_OPT_JIT_HEAP_OPS",
        "TRUST_JS_OPT_JIT_COMPACT_STORES",
    ]
    .iter()
    .all(|key| std::env::var(key).as_deref() != Ok("0"))
}

#[test]
fn warmed_own_store_reduces_ir_and_fits_the_unchanged_hot_budget() {
    if !compact_active() {
        return;
    }
    let (mut engine, function, primary) =
        warmed(HOT_SOURCE, "subject(receiver,1);subject(receiver,2)");
    let chunk = bytecode::compile_for_optimizer(
        &function,
        &primary,
        &[(1, ValueClass::NumberInt32.bit())],
        &[],
    )
    .unwrap();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    stores::TEST_FORCE_POLYMORPHIC_STORES.with(|flag| flag.set(true));
    let expanded = compile_with_limits(&chunk, &values, &layout, None, 262_144, 65_536)
        .expect("old emission under a test-only generous diagnostic budget");
    let expanded_size = stores::TEST_BODY_SIZE.with(std::cell::Cell::get);
    stores::TEST_FORCE_POLYMORPHIC_STORES.with(|flag| flag.set(false));
    stores::TEST_COMPACT_STORES.with(|count| count.set(0));
    let compact = compile_with_deopt(&chunk, &values, &layout, None)
        .expect("warmed own store fits actual production candidate budget");
    let compact_size = stores::TEST_BODY_SIZE.with(std::cell::Cell::get);
    assert_eq!(stores::TEST_COMPACT_STORES.with(std::cell::Cell::get), 1);
    assert!(compact_size.0 < expanded_size.0);
    assert!(compact_size.1 < expanded_size.1);
    assert!(compact.len < expanded.len);
    eprintln!(
        "compact-store IR={} blocks={} bytes={}; expanded IR={} blocks={} bytes={}",
        compact_size.0, compact_size.1, compact.len, expanded_size.0, expanded_size.1, expanded.len
    );
    let code = install(&mut engine, chunk.clone(), compact);
    assert!(function.optimizing_code.set(Some(chunk)).is_ok());
    bytecode::invalidate_call_caches();
    assert_eq!(
        eval(&mut engine, "[subject(receiver,21),receiver.x].join('|')"),
        "42|21"
    );
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(code.residency.active.get(), 0);
    engine.interp.gc_collect();
}

#[test]
fn compact_own_stores_observe_descriptors_prototypes_deletion_exotics_and_throws() {
    let sites = fixture(
        HOT_SOURCE,
        "subject(receiver,1);subject(receiver,2)",
        r#"
        var trace=[],results=[];
        Object.setPrototypeOf(receiver,{set x(v){trace.push('proto:'+v);}});
        results.push(subject(receiver,3));results.push(receiver.x);
        Object.defineProperty(receiver,'x',{get:function(){return 11;},set:function(v){trace.push('own:'+v);},configurable:true});
        results.push(subject(receiver,4));
        Object.defineProperty(receiver,'x',{value:5,writable:false,configurable:true});
        try{subject(receiver,6);}catch(e){results.push(e.name);}
        delete receiver.x;
        results.push(subject(receiver,7));results.push(Object.hasOwn(receiver,'x'));
        Object.preventExtensions(receiver);results.push(subject(receiver,8));
        receiver=new Proxy({x:0},{set:function(t,k,v,r){trace.push('proxy:'+v);return Reflect.set(t,k,v,r);}});
        results.push(subject(receiver,9));results.push(receiver.x);
        receiver=[];receiver.x=0;results.push(subject(receiver,10));results.push(receiver.x);
        receiver={x:0};Object.defineProperty(receiver,'x',{set:function(v){trace.push('throw:'+v);throw 'stop';}});
        try{subject(receiver,11);}catch(e){results.push(e);}
        results.join('|')+'|'+trace.join(',');
        "#,
        "6|3|8|TypeError|14|false|16|18|9|20|10|stop|own:4,proto:7,proto:8,proxy:9,throw:11",
    );
    assert_eq!(sites, usize::from(compact_active()));
}

#[test]
fn compact_own_stores_keep_aliases_every_value_category_and_live_storage_through_gc() {
    for (body, call) in [
        ("return o.x=v", "subject(receiver,held)"),
        ("return (0,o).x=v", "subject(receiver,held)"),
        ("return this.x=v", "subject.call(receiver,receiver,held)"),
    ] {
        for held in [
            "receiver",
            "undefined",
            "null",
            "false",
            "-0",
            "NaN",
            "Infinity",
            "'shared'",
            "Symbol('x')",
            "12345678901234567890n",
        ] {
            let source = format!(
                "function subject(o,v){{'use strict';{body};}}var receiver={{x:0}},held={held};"
            );
            let warm = format!("{call};{call};");
            let run =
                format!("var result={call};Object.is(result,held)+'|'+Object.is(receiver.x,held);");
            let sites = fixture(&source, &warm, &run, "true|true");
            assert_eq!(sites, usize::from(compact_active()), "{body}, {held}");
        }
    }
    let sites=fixture(
        HOT_SOURCE,
        "subject(receiver,1);subject(receiver,2)",
        "for(var k=0;k<200;k++)receiver['field'+k]=k;var result=subject(receiver,9);[result,receiver.x,receiver.field199].join('|')",
        "18|9|199",
    );
    assert_eq!(sites, usize::from(compact_active()));
}

#[test]
fn compact_own_store_uses_replaced_storage_with_the_same_receiver_and_shape() {
    let (mut engine, _function, chunk) =
        warmed(HOT_SOURCE, "subject(receiver,1);subject(receiver,2)");
    let env = engine.interp.global_env.clone();
    let receiver = engine
        .interp
        .get_var("receiver", &env)
        .ok()
        .unwrap()
        .as_obj()
        .unwrap()
        .clone();
    let warmed_shape = receiver.borrow().props.shape();
    let values = crate::value::jit_layout(&engine.interp.object_proto);
    let layout = crate::interpreter::interp_layout(&mut engine.interp);
    stores::TEST_COMPACT_STORES.with(|count| count.set(0));
    let compiled = compile_with_deopt(&chunk, &values, &layout, None).unwrap();
    assert_eq!(
        stores::TEST_COMPACT_STORES.with(std::cell::Cell::get),
        usize::from(compact_active())
    );
    let code = install(&mut engine, chunk, compiled);
    let replacement = {
        let object = receiver.borrow();
        let mut replacement = crate::value::Props::with_capacity(256);
        assert!(replacement.try_copy_named_data_from(&object.props));
        assert_eq!(replacement.shape(), object.props.shape());
        // Both allocations are live here. Compare addresses only; retain no raw
        // storage address across replacement, collection or the native call.
        assert_ne!(
            replacement.property_at(0).unwrap() as *const _,
            object.props.property_at(0).unwrap() as *const _
        );
        replacement
    };
    receiver.borrow_mut().replace_properties(replacement);
    assert_eq!(receiver.borrow().props.shape(), warmed_shape);
    engine.interp.gc_collect();
    elements::TEST_HEAP_HELPERS.with(|count| count.set([0; 3]));
    assert_eq!(eval(&mut engine, "subject(receiver,9)"), "18");
    let helpers = elements::TEST_HEAP_HELPERS.with(std::cell::Cell::get);
    assert!(code.residency.referenced.get() > 0);
    assert_eq!(
        helpers,
        [
            usize::from(std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() == Ok("0")),
            0,
            0
        ],
        "same-shape storage replacement must keep the native overwrite path"
    );
    assert_eq!(eval(&mut engine, "receiver.x"), "9");
    assert_eq!(code.residency.active.get(), 0);
    assert!(engine.interp.fn_frames.is_empty());
    engine.interp.gc_collect();
}

#[test]
fn creation_and_polymorphic_store_sites_keep_the_existing_dynamic_paths() {
    for (source,warm,run,expected) in [
        (HOT_SOURCE,"subject({},1);subject({},2)","var row={};[subject(row,7),row.x].join('|')","14|7"),
        ("function subject(o,n){'use strict';o.x=n;return n+n;}var first={x:0},second={pad:11,x:0};",
         "subject(first,1);subject(second,2)","[subject(first,3),subject(second,4),first.x,second.x,second.pad].join('|')","6|8|3|4|11"),
    ] {
        assert_eq!(fixture(source,warm,run,expected),0);
    }
}
