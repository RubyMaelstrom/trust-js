//! Native slices preserve ECMA-262 Await, GeneratorResume[Abrupt], GeneratorYield and
//! AsyncGeneratorUnwrapYieldResumption. Official local snapshot e28783d5 (2026-09-06).

use crate::{bytecode::Tier, Completion, Engine};

fn value(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("continuation fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

fn check(setup: &str, expression: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        super::TEST_NATIVE_SLICES.with(|count| count.set(0));
        value(&mut engine, setup);
        engine.interp.gc_collect();
        assert_eq!(value(&mut engine, expression), expected, "{tier:?}");
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        if tier == Tier::Jit {
            assert!(
                super::TEST_NATIVE_SLICES.with(|count| count.get()) > 0,
                "fixture must execute a native continuation, not silently fall back"
            );
        }
    }
}

#[test]
fn native_continuation_owned_words_survive_yield_gc_and_numeric_loops() {
    check(r#"
        var token={identity:1},symbol=Symbol('id');
        function* run(input){
            var saved=input,copy=token,sum=0;
            for(var k=0;k<45;k++)sum+=k;
            var resumed=yield [saved,copy,sum,symbol,12345678901234567890n];
            for(var k=0;k<45;k++)sum+=k;
            return [saved===input,copy===token,resumed===token,sum,symbol===resumed.symbol];
        }
        token.symbol=symbol;var input={x:1},it=run(input),first=it.next().value;
    "#, "[first[0]===input,first[1]===token,first[2],first[4],it.next(token).value.join(',')].join('|')",
        "true|true|990|12345678901234567890|true,true,true,1980,true");
}

#[test]
fn native_continuation_nested_finally_throw_return_and_reentry() {
    check(r#"
        var trace=[],sentinel={},it;
        function* body(){try{try{yield 1;}finally{trace.push('inner');yield 2;}}
            finally{trace.push('outer');yield 3;}}
        it=body();var a=it.next(),b=it.return(sentinel),c=it.next(),d=it.next();
        function* reentrant(){try{it.next();}catch(e){yield e.name;}}
        it=reentrant();var reentry=it.next().value;
        function* never(){trace.push('body');yield 1;}
        var idle=never();var early=idle.return(9);
    "#, "[a.value,b.value,c.value,d.value===sentinel,d.done,trace.join(','),reentry,early.value].join('|')",
        "1|2|3|true|true|inner,outer|TypeError|9");
}

#[test]
fn native_continuation_repeated_factory_calls_never_use_ordinary_entry() {
    check(
        r#"
        var out=0,asyncOut=0;
        function* generator(input){yield input;return input+1;}
        async function asynchronous(input){await 0;return input;}
        function invoke(callback,input){return callback(input);}
        for(var k=0;k<140;k++){
            var it=invoke(generator,k);out+=it.next().value+it.next().value;
            invoke(asynchronous,k).then(value=>asyncOut+=value);
        }
        var mapped=[1,2,3].map(generator).map(it=>it.next().value).join(',');
    "#,
        "[out,asyncOut,mapped].join('|')",
        "19600|9730|1,2,3",
    );
}

#[test]
fn native_continuation_await_keeps_reaction_order_and_return_finalizers() {
    check(
        r#"
        var trace=[],out='pending';
        async function body(){trace.push('start');try{
            var value=await {then(resolve){trace.push('then');resolve(7);}};
            trace.push('resumed');return value;
        }finally{trace.push('finally');await 0;trace.push('clean');}}
        body().then(value=>out=value);trace.push('caller');
    "#,
        "trace.join(',')+'|'+out",
        "start,caller,then,resumed,finally,clean|7",
    );
}

#[test]
fn native_continuation_async_generator_return_awaits_before_finally() {
    check(
        r#"
        var trace=[],out='pending';
        async function* body(){try{yield 1;return {then(resolve){trace.push('then');resolve(9);}};}
            finally{trace.push('finally');yield 2;}}
        async function drive(){var it=body();var a=await it.next(),b=await it.next(),c=await it.next();
            out=[a.value,b.value,c.value,c.done,trace.join(',')].join('|');}
        drive();
    "#,
        "out",
        "1|2|9|true|then,finally",
    );
}

#[test]
fn native_continuation_scope_reference_and_async_cleanup_state() {
    check(
        r#"
        var trace=[],out='pending';
        async function body(){
            var captured=0;function read(){return captured;}
            {await using resource={[Symbol.asyncDispose](){trace.push('dispose');return Promise.resolve();}};
             for await(var x of [Promise.resolve(2),3]){captured+=x;await 0;}}
            return read();
        }
        body().then(value=>out=value);
        function* scope(){var target={value:1};with(target){value=yield value;yield value;}return target.value;}
        var it=scope(),a=it.next(),b=it.next(8),c=it.next();
    "#,
        "[out,trace.join(','),a.value,b.value,c.value,c.done].join('|')",
        "5|dispose|1|8|8|true",
    );
}

#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn native_continuation_all_suspension_classes_receive_distinct_entry_code() {
    for source in [
        "function* body(input){var a=yield input;yield* [a];return a;}",
        "async function body(input){try{return await input;}finally{await 0;}}",
        "async function* body(input){for await(var x of input)yield x;return input;}",
        "async function body(input){await using r=input;await 0;return 7;}",
    ] {
        let statements = crate::parser::parse_script(source, false)
            .ok()
            .expect("fixture parses");
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("function")
        };
        let chunk = crate::bytecode::compile(function).expect("body compiles");
        let mut engine = Engine::new();
        let layout = crate::value::jit_layout(&engine.interp.object_proto);
        let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
        let code = super::compile(&chunk, &layout, &ilayout).expect("native slice coverage");
        assert!(chunk.jit_is_resumable());
        assert!(!chunk.jit_no_activation());
        assert_eq!(
            chunk.jit_direct_flags(&code),
            0,
            "never enter via ordinary direct ABI"
        );
        for (pc, op) in chunk.jit_ops().iter().enumerate() {
            if crate::bytecode::jit_resume_after(op) && code.resume_depths[pc].is_some() {
                assert!(
                    code.resume_depths[pc + 1].is_some(),
                    "missing resume root {op:?}"
                );
            }
        }
    }
}
