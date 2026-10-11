//! Native-coverage checks accompany tier agreement: falling back can hide a lost optimization.

#[test]
fn compact_native_frames_preserve_all_value_kinds_and_receiver_ownership() {
    // ECMA-262 §6.1 and OrdinaryCallBindThis: frame representation changes must
    // preserve identities and strict receivers, including primitive receivers.
    assert_tiers(
        r#"
        var object={identity:1}, symbol=Symbol('identity');
        var values=[undefined,null,false,true,0,-0,NaN,Infinity,-Infinity,
            1.25,'owned string',symbol,object,123456789012345678901234567890n];
        function identity(value){return value;}
        function receiver(){'use strict';return this;}
        function move(value,callback){
            var first={old:1},second={old:2};
            first=value;second=first;first=undefined;
            return callback(second);
        }
        function overwrite(value){
            var local={old:1},copy;
            for(var i=0;i<43;i++){copy=(local=value);local=copy;}
            return local;
        }
        var good=true;
        for(var round=0;round<180;round++)for(var k=0;k<values.length;k++){
            var value=values[k];
            if(!Object.is(move(value,identity),value) ||
               !Object.is(overwrite(value),value) ||
               !Object.is(receiver.call(value),value)) good=false;
        }
        good;
    "#,
        "true",
    );
}

#[test]
fn compact_native_numeric_spills_preserve_signed_zero_and_canonical_nan() {
    // ECMA-262 §6.1.6.1 permits NaN payload canonicalization, not aliasing NaNs
    // with object tags or erasing negative zero. Exercise arithmetic, side exits,
    // local ownership replacement, dense/property stores and call boundaries.
    assert_tiers(
        r#"
        function pass(x){return x;}
        function arithmetic(x,branch){
            var out=x,other=x;
            for(var i=0;i<41;i++){
                if(branch && i===20) other=pass(x);
                else other=x*1;
                out=other;
            }
            return out;
        }
        function store(x){var a=[1,2],o={value:0};
            for(var i=0;i<41;i++){a[i&1]=x*1;o.value=-(-x);}
            return Object.is(a[0],x)&&Object.is(a[1],x)&&Object.is(o.value,x);
        }
        var inputs=[0,-0,1.5,-1.5,Number.MIN_VALUE,Number.MAX_VALUE,
            Infinity,-Infinity,NaN];
        var data=new DataView(new ArrayBuffer(8));
        var high=[0x7ff90000,0x7ffa0000,0x7ffb0000,0x7ffc0000,0x7ffd0000,
            0x7ffe0000,0x7fff0000,0xfff90000,0xfffa0000,0xffffffff];
        for(var h=0;h<high.length;h++){
            data.setUint32(0,0x1234,true);data.setUint32(4,high[h],true);
            inputs.push(data.getFloat64(0,true));
        }
        var good=true;
        for(var round=0;round<110;round++)for(var k=0;k<inputs.length;k++){
            var x=inputs[k];
            if(!Object.is(arithmetic(x,true),x) ||
               !Object.is(arithmetic(x,false),x) || !store(x)) good=false;
        }
        good;
    "#,
        "true",
    );
}

#[test]
fn general_region_bitwise_conversion_boundaries_and_shift_counts() {
    // ECMA-262 ToInt32/ToUint32: truncation precedes modulo; shifts mask to five bits.
    // Include the i64 saturation edge, non-finite values and signed/unsigned results.
    assert_tiers(
        r#"
        function calculate(x, y, mode) {
            var result=0, count=0;
            for(var i=0;i<41;i++) {
                if(mode===0) result=x&y;
                else if(mode===1) result=x|y;
                else if(mode===2) result=x^y;
                else if(mode===3) result=x<<y;
                else if(mode===4) result=x>>y;
                else if(mode===5) result=x>>>y;
                else result=~x;
                count+=i;
            }
            if(count!==820) throw new Error('lost local');
            return result;
        }
        var cases=[[0,0],[-0,-0],[1.9,33.9],[-1.9,-1],[4294967295,4],
            [2147483648,1],[2**63,31],[-(2**63),-32],[Infinity,NaN],
            [NaN,Infinity],[Number.MIN_VALUE,-Infinity],
            [9007199254740991,4294967297],[-9007199254740991,-33]];
        var expected=[[0,0,0,0,0,0,-1],[0,0,0,0,0,0,-1],[1,33,32,2,0,0,-2],
            [-1,-1,0,-2147483648,-1,1,0],[4,-1,-5,-16,-1,268435455,0],
            [0,-2147483647,-2147483647,0,-1073741824,1073741824,2147483647],
            [0,31,31,0,0,0,-1],[0,-32,-32,0,0,0,-1],[0,0,0,0,0,0,-1],
            [0,0,0,0,0,0,-1],[0,0,0,0,0,0,-1],[1,-1,-2,-2,-1,2147483647,0],
            [1,-33,-34,-2147483648,0,0,-2]];
        var good=true;
        for(var c=0;c<cases.length;c++)for(var m=0;m<7;m++)
            good=good && Object.is(calculate(cases[c][0],cases[c][1],m),expected[c][m]);
        var conversions=0;
        var object={valueOf:function(){conversions++;return 3;}};
        good && calculate(object,1,0)===1 && conversions===41;
    "#,
        "true",
    );
}

#[test]
fn general_region_direct_calls_keep_polymorphism_and_throw_resumptions() {
    assert_tiers(
        r#"
        function invoke(callback,n){var sum=0;for(var i=0;i<n;i++)sum+=callback(i);return sum;}
        function ordinary(x){return x+1;}
        var arrow=x=>x+2, count=0, token={};
        function throwing(x){count++;if(x===7)throw token;return x;}
        var good=true;
        for(var j=0;j<100;j++) {
            good=good && invoke(ordinary,100)===5050 && invoke(arrow,100)===5150;
        }
        try{invoke(throwing,100);}catch(error){good=good && error===token;}
        good && count===8 && invoke(ordinary,100)===5050;
    "#,
        "true",
    );
}

#[test]
fn general_region_private_homes_materialize_on_throw_and_local_receiver_reads() {
    assert_tiers(
        r#"
        var token={},trace=[];
        function callback(i){if(i===17)throw token;return i;}
        function run(callback){
            var sum=0,i=0;
            try{for(i=0;i<100;i++){sum+=3;sum+=callback(i);}}
            finally{trace.push(sum,i);}
        }
        for(var round=0;round<80;round++){
            try{run(callback);}catch(error){if(error!==token)throw error;}
        }
        // A property helper can read a numeric local receiver directly. That one
        // local must be current even while unrelated homes remain in registers.
        function numbers(n){var out='',x=0;for(var i=0;i<n;i++){
            x+=1;out+=x.toString();
        }return out;}
        // Captured/eval-visible bindings remain actual Environment Record cells.
        function captured(){var n=0;function change(){n+=10;}
            for(var i=0;i<30;i++){n++;change();}return n;
        }
        var good=true;
        for(var j=0;j<trace.length;j+=2)if(trace[j]!==190||trace[j+1]!==17)good=false;
        good && trace.length===160 && numbers(12)==='123456789101112' && captured()===330;
    "#,
        "true",
    );
}

#[test]
fn general_region_register_pressure_and_effect_aliases() {
    assert_tiers(
        r#"
        function crowded(n,o){
            var a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8,j=9,k=10,s=0;
            for(var i=0;i<n;i++){
                a++;b++;c++;d++;e++;f++;g++;h++;j++;k++;s+=o.value;
            }
            return a+b+c+d+e+f+g+h+j+k+s;
        }
        function aliases(receiver,other,callback){
            var sum=0;
            for(var i=0;i<40;i++){
                receiver.value=i;
                sum+=other.value;
                callback(receiver,i);
                sum+=other.value;
            }
            return sum;
        }
        var o={},good=true;
        function change(o,i){
            if(i&1) Object.defineProperty(o,'value',{get:function(){return i+1;},configurable:true});
            else Object.defineProperty(o,'value',{value:i+1,writable:true,configurable:true});
        }
        var trace=[];
        function setter(v){trace.push(v);}
        for(var round=0;round<80;round++){
            good=good && crowded(40,{value:3})===575;
            Object.defineProperty(o,'value',{value:0,writable:true,configurable:true});
            // The getter installed on odd iterations blocks the following sloppy
            // write. Reads still use the live descriptor and aliased receiver.
            good=good && aliases(o,o,change)===1600;
        }
        good;
    "#,
        "true",
    );
}

#[test]
fn general_region_tagged_equality_keeps_coercion_and_nan_rules() {
    // ECMA-262 IsStrictlyEqual/IsLooselyEqual (local e28783d5): same-type
    // identity, Number equality, nullish equivalence and observable ToPrimitive.
    assert_tiers(
        r#"
        function compare(a,b,mode){var sum=0;for(var i=0;i<41;i++){
            if(mode===0){if(a===b)sum++;}
            else if(mode===1){if(a!==b)sum++;}
            else if(mode===2){if(a==b)sum++;}
            else {if(a!=b)sum++;}
        }return sum;}
        var object={},symbol=Symbol('identity'),calls=0;
        var coercible={valueOf:function(){calls++;return 7;}};
        var pairs=[[NaN,NaN,false,false],[-0,0,true,true],
            [null,undefined,false,true],[false,0,false,true],
            ['same','sa'+'me',true,true],[7,'7',false,true],
            [7n,7,false,true],[8n,7,false,false],
            [object,object,true,true],[{},object,false,false],
            [symbol,symbol,true,true],[Symbol('identity'),symbol,false,false],
            [coercible,7,false,true]];
        var good=true;
        for(var r=0;r<5;r++)for(var k=0;k<pairs.length;k++){
            var pair=pairs[k];
            good=good && compare(pair[0],pair[1],0)===(pair[2]?41:0)
                && compare(pair[0],pair[1],1)===(pair[2]?0:41)
                && compare(pair[0],pair[1],2)===(pair[3]?41:0)
                && compare(pair[0],pair[1],3)===(pair[3]?0:41);
        }
        good && calls===410;
    "#,
        "true",
    );
}

#[test]
fn captured_const_store_inside_a_block_scope_throws() {
    // SetMutableBinding on an immutable binding throws a TypeError (ECMA-262
    // DeclarativeEnvironmentRecord.SetMutableBinding); the captured `const` lives in the body's
    // activation record, not in the loop's per-iteration record running the assignment.
    assert_tiers(
        r#"
        function f() {
            const c = 1;
            var g = function () { return c; };
            for (let j = 0; j < 1; j++) {
                var h = function () { return j; };
                try { c = 2; } catch (e) { return e.constructor.name + ' ' + g() + h(); }
            }
            return 'no error';
        }
        [f(), f(), f()].join(',')
    "#,
        "TypeError 10,TypeError 10,TypeError 10",
    );
}

use crate::{bytecode::Tier, Completion, Engine};

fn assert_tiers(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine
            .eval(source, false)
            .expect("regression source parses")
        {
            Completion::Value(actual) => assert_eq!(actual, expected, "tier {tier:?}"),
            Completion::Throw { name, message } => {
                panic!("unexpected {tier:?} throw: {name}: {message}")
            }
        }
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn assert_native_compiles(source: &str) {
    let statements = crate::parser::parse_script(source, false)
        .ok()
        .expect("source parses");
    let function = statements
        .iter()
        .find_map(|statement| match statement {
            crate::ast::Stmt::FuncDecl(function) => Some(function),
            _ => None,
        })
        .expect("function declaration");
    let chunk = crate::bytecode::compile(function).expect("body compiles to bytecode");
    let mut engine = Engine::new();
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let interpreter_layout = crate::interpreter::interp_layout(&mut engine.interp);
    assert!(
        super::compile(&chunk, &layout, &interpreter_layout).is_some(),
        "body must receive native code: {:?}",
        chunk.jit_ops()
    );
}

#[test]
fn native_object_spread_preserves_keys_getters_and_throw_ownership() {
    // ECMA-262 CopyDataProperties visits keys in order, checks each live descriptor,
    // and invokes Get only for enumerable properties. A getter can remove a later key.
    let source = r#"
        function merge(source) { return {...source, tail: 9}; }
        var key = Symbol('symbol'), trace = [], thrown = {sentinel: 1};
        for (var iteration = 0; iteration < 150; ++iteration) {
            var input = { get a() { delete this.b; trace.push('a'); return 1; }, b: 2 };
            input[key] = 3;
            Object.defineProperty(input, 'hidden', {value: 4, enumerable: false});
            var result = merge(input);
            if (result.a !== 1 || result[key] !== 3 || result.tail !== 9 ||
                'b' in result || 'hidden' in result || Object.keys(result).join() !== 'a,tail')
                throw new Error('incorrect spread');
        }
        var caught = false;
        try { merge({get boom() {throw thrown;}}); } catch (error) {caught = error === thrown;}
        [trace.length, caught, Object.keys(merge(null)).join(),
            Object.keys(merge(undefined)).join(), merge('ab')[1]].join(':');
    "#;
    assert_tiers(source, "150:true:tail:tail:b");
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    assert_native_compiles("function merge(source) { return {...source, tail: 9}; }");
}

#[test]
fn native_shared_closure_inline_guard_checks_the_callers_environment() {
    // One Function AST serves both closures, while the pinned callee stays the
    // first closure's function. Identity alone does not validate lexical lookup.
    assert_tiers(
        r#"
        function factory(value) {
            function read() { return value; }
            function invoke(callee) { return callee(); }
            return {read: read, invoke: invoke};
        }
        var first = factory(11), second = factory(29), total = 0;
        for (var k = 0; k < 400; ++k) total += first.invoke(first.read);
        [total, second.invoke(first.read), first.invoke(second.read)].join(':');
        "#,
        "4400:11:29",
    );
}

#[test]
fn native_finally_preserves_nested_completion_precedence() {
    assert_tiers(
        r#"
        var trace = [];
        function run(mode) {
            try {
                try { if (mode === 0) return 7; if (mode === 1) throw 8; return; }
                finally { trace.push('inner'); if (mode === 2) return 9; }
            } finally { trace.push('outer'); if (mode === 3) throw 10; }
        }
        var out = [];
        for (var i = 0; i < 4; ++i) {
            try { out.push(run(i)); } catch (e) { out.push('throw' + e); }
        }
        function jump() {
            var sum = 0;
            for (var j = 0; j < 5; ++j) {
                try { if (j === 1) continue; if (j === 3) break; sum += j; }
                finally { sum += 10; }
            }
            return sum;
        }
        out.join(':') + '/' + trace.join(',') + '/' + jump();
    "#,
        "7:throw8:9:throw10/inner,outer,inner,outer,inner,outer,inner,outer/42",
    );
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    {
        assert_native_compiles("function f(x) { try { return x + 1; } finally { x = 4; } }");
        assert_native_compiles("function f(n) { var s=0; for(var i=0;i<n;i++){try {if(i===1)continue;if(i===3)break;s+=i;}finally{s+=10;}} return s; }");
    }
}

#[test]
fn native_iterator_close_and_finally_preserve_throw_priority() {
    assert_tiers(
        r#"
        var log = [], original = {}, closeError = {};
        function iterable() {
            var iterator = {next: function(){return {value:1,done:false};},
                return: function(){log.push('close');throw closeError;}};
            iterator[Symbol.iterator] = function(){return this;};
            return iterator;
        }
        function thrower() { for (var item of iterable()) { try {throw original;} finally {log.push('finally');} } }
        function returner() { for (var item of iterable()) { try {return item;} finally {log.push('finally');} } }
        var a=false,b=false;
        try{thrower();}catch(e){a=e===original;}
        try{returner();}catch(e){b=e===closeError;}
        a+':'+b+':'+log.join(',');
    "#,
        "true:true:finally,close,finally,close",
    );
}

#[test]
fn native_stateful_operations_keep_lexical_and_prepared_bindings_live() {
    assert_tiers(
        r#"
        function f({x} = {x:2}, ...rest) {
            var callbacks = [];
            for (let i=0;i<3;++i) callbacks.push(()=>x+i);
            var copy = [0, ...rest, , 9];
            var {a, ...remaining} = {a:1,b:2};
            { let x=20; callbacks.push(()=>x); }
            return callbacks.map(fn=>fn()).join(',')+':'+copy.length+':'+remaining.b;
        }
        f(undefined, 4, 5);
    "#,
        "2,3,4,20:5:2",
    );
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    {
        assert_native_compiles(
            "function f(x) { var a=[...x]; var {p,...rest}=x; return a.length+rest.q; }",
        );
        assert_native_compiles("function f({x}={x:1}) { {let y=x; return ()=>y;} }");
    }
}

#[test]
fn native_classes_and_private_members_keep_home_object_and_abrupt_state() {
    assert_tiers(
        r#"
        function build(v) {
            class Base { read(){return 4;} }
            class Derived extends Base {
                #value = v;
                read(){return super.read()+this.#value;}
                has(o){return #value in o;}
            }
            var object = new Derived();
            return object.read()+':'+object.has(object)+':'+object.has({});
        }
        build(7);
    "#,
        "11:true:false",
    );
    #[cfg(all(
        any(target_arch = "aarch64", target_arch = "x86_64"),
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    assert_native_compiles(
        "function f() { class C { #x=1; get(){return this.#x;} } return new C().get(); }",
    );
}

#[test]
fn optimized_private_backedges_honor_host_deadlines() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let interrupt = engine.interrupt_handle();
    interrupt.set_deadline(Some(
        std::time::Instant::now() + std::time::Duration::from_millis(20),
    ));
    let started = std::time::Instant::now();
    let result = engine
        .eval_interruptible(
            "function spin(n){var sum=0;for(var i=0;i<n;i++)sum+=i;return sum;} spin(1e12);",
            false,
        )
        .expect("parse");
    assert!(matches!(
        result,
        crate::ExecutionOutcome::Interrupted {
            reason: crate::InterruptReason::DeadlineExceeded
        }
    ));
    assert!(started.elapsed() < std::time::Duration::from_secs(2));
}

#[test]
fn general_cfg_regions_preserve_objects_calls_nested_branches_and_bailout_effects() {
    assert_tiers(
        r#"
        function compute(n, object, callback) {
            var sum=0;
            for(var i=0;i<n;i++) {
                if ((i % 3) === 0) { sum += object.value; continue; }
                for(var j=0;j<3;j++) {
                    if(j===1) sum+=callback(i); else sum+=j;
                }
                if(sum>1000) break;
            }
            return sum;
        }
        var reads=0,calls=0;
        var object={get value(){reads++;return reads===3?'5':5;}};
        function callback(value){calls++;return value;}
        var result=compute(20,object,callback);
        result+':'+reads+':'+calls;
    "#,
        "305072:3:5",
    );
}
