//! ECMA-262 DestructuringAssignmentEvaluation and its iterator/keyed/reference algorithms.
//! Local official snapshot e28783d5fc9dc12b3de905961e2c71410b38a202 (2026-09-06).

use crate::{bytecode::Tier, Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine
            .eval(source, false)
            .expect("assignment fixture parses")
        {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

#[test]
fn native_assignment_patterns_preserve_values_defaults_and_rest_keys() {
    check(
        r#"
        function run(input,key) {
            var a,b,c,rest,f,unchanged;
            var original=([a,,[b=7],...rest]=input);
            ({[key]:c,missing:f=function(){},nil:unchanged=99,...rest}=rest[0]);
            return [original===input,a,b,c,f.name,unchanged,rest.tail,
                    Object.getOwnPropertySymbols(rest).length,Object.keys(rest).join()].join('|');
        }
        var key=Symbol('take'),keep=Symbol('keep'),object={nil:null,tail:8};
        object[key]=9;object[keep]=10;
        var out;
        for(var i=0;i<50;i++)out=run([1,2,[],object],key);
        out;
    "#,
        "true|1|7|9|f||8|1|tail",
    );
}

#[test]
fn native_assignment_reference_order_and_close_precedence() {
    check(
        r#"
        var trace=[],sentinel={},target={set x(v){trace.push('set'+v);}},key={
            toString(){trace.push('key');return 'x';}
        };
        function base(){trace.push('base');return target;}
        function input(value,closeThrows){return {[Symbol.iterator](){trace.push('iter');return {
            next(){trace.push('next');return {value:value,done:false};},
            return(){trace.push('close');if(closeThrows)throw 'close-error';return {};}
        };}};}
        function assign(value){[base()[key]= (trace.push('default'),4)]=value;}
        assign(input(undefined,false));
        var first=trace.join(',');trace=[];
        function fail(value){[target.x=(function(){throw sentinel;})()]=value;}
        var preserved=false;try{fail(input(undefined,true));}catch(e){preserved=e===sentinel;}
        var second=trace.join(',');trace=[];
        function empty(value){[]=value;}
        empty(input(1,false));
        [first,second,preserved,trace.join(',')].join('|');
    "#,
        "iter,base,next,default,key,set4,close|iter,next,close|true|iter,close",
    );
}

#[test]
fn native_assignment_nullish_reference_rejects_at_put_after_iterator_step() {
    check(
        r#"
        var steps=0,closed=0;
        function run(base){
            var source={[Symbol.iterator](){return {
                next(){steps++;return {done:false,value:1};},
                return(){closed++;return {};}
            };}};
            try{[base['x']]=source;}catch(e){return e.name;}
        }
        [run(null),run(undefined),steps,closed].join('|');
    "#,
        "TypeError|TypeError|2|2",
    );
}

#[test]
fn native_assignment_partial_writes_tdz_const_and_for_heads() {
    check(
        r#"
        function run(){
            var first=0,trace=[];
            const fixed=1;
            try{[first,fixed]=[4,5];}catch(e){trace.push(e.name,first,fixed);}
            try{[first,later]=[6,7];let later;}catch(e){trace.push(e.name,first);}
            var key,item,sum=0;
            for([key,item] of [[1,2],[3,4]])sum+=key+item;
            for({length:item} in {ab:1,cdef:2})sum+=item;
            var captured=0;function get(){return captured;}
            [captured]=[11];
            return trace.join(',')+'|'+sum+'|'+get();
        }
        var out;for(var i=0;i<30;i++)out=run();out;
    "#,
        "TypeError,4,1,ReferenceError,6|16|11",
    );
}

#[test]
fn native_assignment_super_private_and_prepared_parameters() {
    check(
        r#"
        var trace=[];
        class Base {set value(v){trace.push('set'+v);this.saved=v;}}
        class Derived extends Base {
            #private=0;
            run(source,{offset=1}={}) {
                var key={toString(){trace.push('key');return 'value';}};
                [super[key],this.#private]=source;
                return this.saved+this.#private+offset;
            }
        }
        var instance=new Derived();
        var source={[Symbol.iterator](){var k=0;return {
            next(){trace.push('next');return {value:++k,done:false};},
            return(){trace.push('close');return {};}
        };}};
        [instance.run(source),trace.join(',')].join('|');
    "#,
        "4|next,key,set1,next,close",
    );
}

#[test]
#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn native_assignment_patterns_receive_machine_code() {
    for source in [
        "function run(input){var a,b,rest;return ([a,{b=7},...rest]=input);}",
        "function run(input,key){var a,rest;return ({[key]:a,...rest}=input);}",
        "function run(input){var a,b,out=0;for([a,b] of input)out+=a+b;return out;}",
    ] {
        let statements = crate::parser::parse_script(source, false)
            .ok()
            .expect("fixture parses");
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("function")
        };
        let chunk = crate::bytecode::compile(function).expect("ordinary pattern compiles");
        assert!(
            !chunk
                .jit_ops()
                .iter()
                .any(|op| matches!(op, crate::bytecode::Op::AssignTarget(_))),
            "pattern must use explicit native-compatible lowering"
        );
        let mut engine = Engine::new();
        let layout = crate::value::jit_layout(&engine.interp.object_proto);
        let interpreter_layout = crate::interpreter::interp_layout(&mut engine.interp);
        assert!(
            super::compile(&chunk, &layout, &interpreter_layout).is_some(),
            "pattern must receive machine code: {:?}",
            chunk.jit_ops()
        );
    }
}
