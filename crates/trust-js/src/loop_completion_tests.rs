//! DoWhileLoopEvaluation, LoopContinues and UpdateEmpty; local official ECMA-262
//! snapshot e28783d5fc9dc12b3de905961e2c71410b38a202, spec.html 22450–22542.
use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        for threshold in [0, 32] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(threshold);
            match engine.eval(source, false).expect("loop fixture parses") {
                Completion::Value(value) => {
                    assert_eq!(value, expected, "{tier:?}/{threshold}: {source}");
                }
                Completion::Throw { name, message } => {
                    panic!("{tier:?}/{threshold}: {name}: {message}\n{source}");
                }
            }
        }
    }
}

#[test]
fn do_while_observes_one_condition_per_continued_body() {
    check(
        "var tests=0,bodies=0;do{bodies++;}while(++tests<4);bodies+','+tests",
        "4,4",
    );
    check(
        "var trace=[];var gate={get next(){trace.push('test');return trace.length<6;}};do{trace.push('body');}while(gate.next);trace.join(',')",
        "body,test,body,test,body,test",
    );
    check(
        "var bodies=0,coerced=0;var truthy={valueOf(){coerced++;return false;}};do{if(++bodies===3)break;}while(truthy);bodies+','+coerced",
        "3,0",
    );
}

#[test]
fn do_while_continue_labels_and_finalizers_share_the_condition_phase() {
    check(
        "var trace=[],n=0;outer:inner:do{trace.push('b'+(++n));try{if(n===1)continue outer;if(n===2)continue inner;trace.push('body');}finally{trace.push('f'+n);}}while((trace.push('t'+n),n<3));trace.join(',')",
        "b1,f1,t1,b2,f2,t2,b3,body,f3,t3",
    );
    check(
        "var outerTests=0,innerTests=0,bodies=0;outer:do{do{bodies++;continue outer;}while(++innerTests<8);}while(++outerTests<3);[bodies,innerTests,outerTests].join(',')",
        "3,0,3",
    );
    // Block evaluation disposes its resources before the loop consumes the continue
    // completion and evaluates the condition (ECMA-262 Block / DisposeResources).
    check(
        "var trace=[],n=0;do{using resource={[Symbol.dispose](){trace.push('d'+n);}};trace.push('b'+(++n));continue;}while((trace.push('t'+n),n<2));trace.join(',')",
        "b1,d1,t1,b2,d2,t2",
    );
}

#[test]
fn do_while_abrupt_exits_do_not_evaluate_the_condition() {
    check(
        "var tests=0,trace=[];out:do{try{trace.push('body');break out;}finally{trace.push('finally');}}while(++tests);trace.join(',')+'|'+tests",
        "body,finally|0",
    );
    check(
        "var tests=0;function f(){do{try{return 1;}finally{return 7;}}while(++tests);}f()+'|'+tests",
        "7|0",
    );
    check(
        "var tests=0,seen='';try{do{throw 'body';}while(++tests);}catch(e){seen=e;}seen+'|'+tests",
        "body|0",
    );
    check(
        "var tests=0;do{try{continue;}finally{break;}}while(++tests);tests",
        "0",
    );
}

#[test]
fn do_while_condition_errors_follow_body_and_cleanup_exactly_once() {
    check(
        "var trace=[];function test(){trace.push('test');throw 'condition';}try{do{try{trace.push('body');continue;}finally{trace.push('cleanup');}}while(test());}catch(e){trace.push(e);}trace.join(',')",
        "body,cleanup,test,condition",
    );
    check(
        "var trace=[];function test(){trace.push('test');throw 'condition';}try{do{trace.push('body');}while(test());}catch(e){trace.push(e);}trace.join(',')",
        "body,test,condition",
    );
}

#[test]
fn do_while_preserves_statement_completion_after_false_and_continue() {
    check("do{42;}while(false)", "42");
    check("do{42;continue;}while(false)", "42");
    check("do{42;}while((99,false))", "42");
    check("var n=0;do{n++;if(n===1){42;}}while(n<2)", "undefined");
    check("do{}while(false)", "undefined");
    check("var n=0;do{++n;}while(n<4)", "4");
    check("label:do{42;break label;}while(true)", "42");
}
