//! ECMA-262 Grouping Evaluation, Optional ChainEvaluation and EvaluateCall preserve
//! References until GetThisValue/WithBaseObject. Official local snapshot e28783d5
//! (2026-09-06), §§13.2.9.3, 13.3.6.2, 13.3.9.

use crate::{bytecode::Tier, Completion, Engine};

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine
        .eval(source, false)
        .expect("call reference fixture parses")
    {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.set((0, 0)));
        assert_eq!(evaluate(&mut engine, source), expected, "{tier:?}");
        let (vm, native) = crate::bytecode::TEST_SCRIPT_ENTRIES.with(|counts| counts.get());
        if tier != Tier::Interp {
            assert!(vm + native > 0, "fixture must enter compiled Script code");
        }
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        if tier == Tier::Jit {
            assert!(native > 0, "fixture must enter native Script code");
        }
    }
}

#[test]
fn grouped_optional_reference_keeps_receiver_but_value_operators_do_not() {
    check(
        r#"
        var object={m(){'use strict';return this===object;}};
        [object.m(),(object.m)(),object?.m(),(object?.m)(),object.m?.(),
         (object.m)?.(),object?.m?.(),(object?.m)?.(),(object?.['m'])(),
         ((object?.m))(),(0,object?.m)(),(true?object?.m:0)(),
         (object?.m||0)()].join(',');
    "#,
        "true,true,true,true,true,true,true,true,true,true,false,false,false",
    );
}

#[test]
fn grouped_optional_boundary_preserves_argument_and_template_abrupt_order() {
    check(
        r#"
        var trace=[],nil=null;
        nil?.m(trace.push('chain'));
        try{(nil?.m)(trace.push('ordinary'));}catch(e){trace.push(e.name);}
        (nil?.m)?.(trace.push('optional'));
        try{(nil?.m).x(trace.push('outer'));}catch(e){trace.push(e.name);}
        try{(nil?.m)`a${trace.push('tag')}`;}catch(e){trace.push(e.name);}
        trace.join(',');
    "#,
        "ordinary,TypeError,TypeError,tag,TypeError",
    );
}

#[test]
fn grouped_optional_computed_getter_and_spread_evaluate_once() {
    check(
        r#"
        var trace=[],key={toString(){trace.push('key');return 'm';}};
        var object={get m(){trace.push('get');return function(a,b){
            'use strict';trace.push(this===object,a,b);return 7;};}};
        function arg(){trace.push('arg');return [2,3];}
        var out=(object?.[key])(...arg());
        [out,trace.join(',')].join('|');
    "#,
        "7|key,get,arg,true,2,3",
    );
}

#[test]
fn optional_environment_call_keeps_with_base_and_indirect_eval() {
    check(
        r#"
        var reads=0,scope={get m(){reads++;return function(){
            'use strict';return this===scope;};}},result;
        with(scope){result=m?.(delete scope.m);}
        function indirect(){var localOnly=4;return eval?.('typeof localOnly');}
        [result,reads,indirect()].join('|');
    "#,
        "true|1|undefined",
    );
}

#[test]
fn grouped_optional_private_super_and_tag_references_keep_actual_receiver() {
    check(
        r#"
        class Base {m(){return this.value;}}
        class Derived extends Base {
            #m(){return this.value+1;}
            constructor(){super();this.value=7;}
            run(){return [(super.m)?.(),(super['m'])?.(),(this?.#m)(),
                (this?.#m)?.()].join(',');}
        }
        var object={value:9,tag(strings,x){return this.value+x+strings[0];}};
        [new Derived().run(),(object?.tag)`:${2}`].join('|');
    "#,
        "7,7,8,8|11:",
    );
}

#[test]
fn grouped_optional_receiver_survives_suspension_and_collection() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        super::TEST_NATIVE_SLICES.with(|count| count.set(0));
        evaluate(
            &mut engine,
            r#"
            var object={value:13,m(x){return this.value+x;}},asyncResult;
            function* generator(){return (object?.m)(yield 1);}
            async function asynchronous(){return (object?.m)(await 4);}
            var iterator=generator(),first=iterator.next();
            asynchronous().then(value=>asyncResult=value);
            object=null;
        "#,
        );
        engine.interp.gc_collect();
        assert_eq!(
            evaluate(
                &mut engine,
                "[first.value,iterator.next(2).value,asyncResult].join('|');"
            ),
            "1|15|17",
            "{tier:?}"
        );
        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "x86_64"),
            any(target_os = "linux", target_os = "macos", target_os = "windows")
        ))]
        if tier == Tier::Jit {
            assert!(super::TEST_NATIVE_SLICES.with(|count| count.get()) > 0);
        }
    }
}

#[test]
fn grouped_optional_hot_tail_calls_keep_receiver() {
    check(
        r#"
        'use strict';
        var object={value:5,m(x){return this.value+x;}};
        function invoke(x){return (object?.m)(x);}
        function optional(x){return (object?.m)?.(x);}
        var sum=0;for(var k=0;k<300;k++)sum+=invoke(k)+optional(k);
        sum;
    "#,
        "92700",
    );
}

#[test]
fn call_reference_tail_failures_and_super_getters_evaluate_once() {
    check(
        r#"
        'use strict';var trace=[];
        function bad(){return ({get m(){trace.push('get');return 0;}}).m(trace.push('arg'));}
        function tag(){return ({get m(){trace.push('tag-get');return null;}}).m`${trace.push('sub')}`;}
        try{bad();}catch(e){trace.push(e.name);}
        try{tag();}catch(e){trace.push(e.name);}
        class Base {get m(){trace.push(this.value);return function(){return this.value;};}}
        class Derived extends Base {constructor(){super();this.value=7;}
            run(){return (super.m)?.();}}
        trace.push(new Derived().run());trace.join(',');
    "#,
        "get,arg,TypeError,tag-get,sub,TypeError,7,7",
    );
}

#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn grouped_optional_reference_receives_native_machine_code() {
    for source in [
        "function run(object,arg){return (object?.m)(arg);}",
        "function run(object,arg){return (object?.m)?.(arg);}",
        "function run(object,key,arg){return (object?.[key])(arg);}",
        "function run(object,arg){return (object?.tag)`text${arg}`;}",
    ] {
        let statements = crate::parser::parse_script(source, false)
            .ok()
            .expect("fixture parses");
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("function")
        };
        let chunk = crate::bytecode::compile(function).expect("reference compiles");
        assert!(chunk.jit_ops().iter().any(|op| matches!(
            op,
            crate::bytecode::Op::CallWithThis(..) | crate::bytecode::Op::TailCall(_, true)
        )));
        let mut engine = Engine::new();
        let layout = crate::value::jit_layout(&engine.interp.object_proto);
        let interpreter_layout = crate::interpreter::interp_layout(&mut engine.interp);
        assert!(
            super::compile(&chunk, &layout, &interpreter_layout).is_some(),
            "reference must receive machine code: {:?}",
            chunk.jit_ops()
        );
    }
}
