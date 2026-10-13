//! Compiled proper-tail-call coverage. Checks the executed tier as well as the JS result.
//! Normative snapshot: ecma262 e28783d5fc9dc12b3de905961e2c71410b38a202,
//! spec.html:19922 (EvaluateCall), :26255 (tail positions), :26617 (PrepareForTailCall).

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn evaluate(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("tail fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn assert_compiled(engine: &mut Engine, name: &str, native: bool) {
    let env = engine.interp.global_env.clone();
    let value = engine
        .interp
        .get_var(name, &env)
        .unwrap_or_else(|_| panic!("missing {name}"));
    let crate::value::Value::Obj(object) = value else {
        panic!("{name} is not an object")
    };
    let object = object.borrow();
    let crate::value::Callable::User(user) = &object.call else {
        panic!("{name} is not a user function")
    };
    let chunk = user
        .func
        .code
        .get()
        .and_then(Option::as_ref)
        .unwrap_or_else(|| panic!("{name} stayed interpreted"));
    assert!(chunk.has_tail_calls(), "{name} did not lower its tail call");
    if native && cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
        assert!(
            chunk.jit.get().is_some_and(|code| code.is_some()),
            "{name} stayed in bytecode"
        );
    }
}

fn bounded_depth(
    interp: &mut crate::interpreter::Interp,
    _this: crate::value::Value,
    _args: &[crate::value::Value],
) -> Result<crate::value::Value, crate::value::Value> {
    // Detect retained logical frames early instead of risking a host stack exhaustion. Shallow
    // tail calls are ordinary calls (`Op::TailDeep`); past that depth every tail call retires
    // its caller, so a chain of any length stays within a fixed bound.
    let bound = crate::bytecode::ordinary_tail_call_depth() + 16;
    assert!(
        interp.depth <= bound,
        "tail chain grew logical call depth to {}",
        interp.depth
    );
    assert!(
        interp.fn_frames.len() as u32 <= bound,
        "tail chain retained function frames"
    );
    Ok(crate::value::Value::Undefined)
}

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    let global = engine.interp.global.clone();
    engine
        .interp
        .def_method(&global, "checkTailDepth", 0, bounded_depth);
    engine
}

#[test]
fn compiled_tail_mutual_recursion_retires_vm_and_native_frames() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            function even(n,a) { 'use strict'; checkTailDepth(); return n === 0 ? a : odd(n-1,a+1); }
            function odd(n,a) { 'use strict'; checkTailDepth(); return n === 0 ? a : even(n-1,a+1); }
            function outer(n) { return 7 + even(n,0); }
            outer(1); outer(2); outer(20000)
        "#
            ),
            "20007"
        );
        assert_compiled(&mut engine, "even", matches!(tier, Tier::Jit));
        assert_compiled(&mut engine, "odd", matches!(tier, Tier::Jit));
        assert!(engine.interp.pending_tail.is_none());
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn tail_dispatch_keeps_stack_growth_for_nested_non_tail_calls() {
    // PrepareForTailCall retires only the caller's context. The resumed callee
    // can still create ordinary nested contexts, including through accessors.
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        std::thread::Builder::new()
            .name(format!("tail-stack-{tier:?}"))
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                let mut engine = engine(tier);
                assert_eq!(
                    evaluate(
                        &mut engine,
                        r#"
                    function tail(n) { 'use strict'; return body(n); }
                    function body(n) { return n === 0 ? 0 : 1 + tail(n - 1); }
                    tail(2048)
                "#
                    ),
                    "2048"
                );
                assert_eq!(crate::interpreter::execution_stack_segments(), 0);
                assert_eq!(
                    evaluate(
                        &mut engine,
                        r#"
                        var remaining = 512;
                        var object = { get value() { 'use strict'; return read(); } };
                        function read() { return remaining-- === 0 ? 0 : 1 + object.value; }
                        object.value
                    "#
                    ),
                    "512"
                );
                assert_eq!(crate::interpreter::execution_stack_segments(), 0);
            })
            .unwrap()
            .join()
            .unwrap();
    }
}

#[test]
fn compiled_tail_expression_forms_and_optional_receivers() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        for expression in [
            "(next(n-1,a+1))",
            "(0, next(n-1,a+1))",
            "true && next(n-1,a+1)",
            "false || next(n-1,a+1)",
            "null ?? next(n-1,a+1)",
            "next?.(n-1,a+1)",
            "holder?.step(n-1,a+1)",
            "holder.step?.(n-1,a+1)",
            "holder['step']?.(n-1,a+1)",
            "next(...[n-1,a+1])",
        ] {
            let mut engine = engine(tier);
            let source = format!(
                r#"
                function next(n,a) {{ 'use strict'; checkTailDepth(); return n === 0 ? a : {expression}; }}
                var holder={{step:next}};
                next(2048,0)
            "#
            );
            assert_eq!(
                evaluate(&mut engine, &source),
                "2048",
                "{tier:?}: {expression}"
            );
            assert_compiled(&mut engine, "next", matches!(tier, Tier::Jit));
        }
        let mut engine = engine(tier);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var calls=0;
            function side(){calls++;return 1;}
            function missing(f){'use strict';return f?.(side());}
            function method(o){'use strict';return o?.step?.(side());}
            var holder={base:40,step(x){return this.base+x;}};
            [missing(null),method(null),method({}),method(holder),calls].join('|')
        "#
            ),
            "|||41|1"
        );
        assert_compiled(&mut engine, "missing", matches!(tier, Tier::Jit));
        assert_compiled(&mut engine, "method", matches!(tier, Tier::Jit));
    }
}

#[test]
fn compiled_tail_call_order_and_throwing_callee() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var order=[];
            var box={get f(){order.push('get');return 3;}};
            function arg(){order.push('arg');return 1;}
            function invoke(){'use strict';return box.f(arg());}
            try{invoke()}catch(e){order.push(e.name)}
            function fail(){throw 42;}
            function transfer(){'use strict';return fail();}
            function caller(){try{return 1+transfer()}catch(e){return e+1}}
            order.join(',')+'|'+caller()+'|'+caller()
        "#
            ),
            "get,arg,TypeError|43|43"
        );
        assert_compiled(&mut engine, "invoke", matches!(tier, Tier::Jit));
        assert_compiled(&mut engine, "transfer", matches!(tier, Tier::Jit));
        assert!(engine.interp.pending_tail.is_none());
    }
}

#[test]
fn compiled_tail_cleanup_contexts_preserve_call_and_cleanup_order() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var log=[];
            function fail(){log.push('call');throw 5;}
            function guarded(){'use strict';try{return fail()}catch(e){log.push('catch');return e}}
            function finalized(){'use strict';try{return fail()}finally{log.push('finally')}}
            function iterated(){'use strict';for(var x of {
                [Symbol.iterator](){return this},next(){return {done:false,value:1}},
                return(){log.push('close');return {done:true}}
            }){return fail()}}
            function disposed(){'use strict';using x={[Symbol.dispose](){log.push('dispose')}};return fail()}
            guarded();
            try{finalized()}catch(e){}
            try{iterated()}catch(e){}
            try{disposed()}catch(e){}
            log.join(',')
        "#
            ),
            "call,catch,call,finally,call,close,call,dispose"
        );
        assert!(engine.interp.pending_tail.is_none());
    }
}

#[test]
fn compiled_tail_finalizers_catches_and_empty_disposal_boundaries_transfer() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            function fromCatch(n){'use strict';checkTailDepth();try{throw n}catch(e){return e===0?0:fromCatch(e-1)}}
            function fromFinally(n){'use strict';checkTailDepth();try{return 10}finally{if(n) return fromFinally(n-1)}}
            function beforeUsing(n){'use strict';checkTailDepth();return n===0?0:beforeUsing(n-1);using absent=null;}
            fromCatch(2048)+fromFinally(2048)+beforeUsing(2048)
        "#
            ),
            "10"
        );
        assert_compiled(&mut engine, "fromCatch", matches!(tier, Tier::Jit));
        assert_compiled(&mut engine, "fromFinally", false);
        assert_compiled(&mut engine, "beforeUsing", false);
    }
}

#[test]
fn compiled_tail_constructor_results_do_not_escape_to_the_caller() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            function result(n){return n===2?{answer:42}:17;}
            function C(n){'use strict';return result(n);}
            C(0);C(2);
            function make(n){return new C(n);}
            var ok=true;
            for(var n=0;n<20;n++){ok=ok && make(1) instanceof C && make(2).answer===42;}
            class Base{constructor(n){return result(n)}}
            ok && new Base(1) instanceof Base && new Base(2).answer===42
        "#
            ),
            "true"
        );
        assert_compiled(&mut engine, "C", matches!(tier, Tier::Jit));
        assert!(engine.interp.pending_tail.is_none());
    }
}

#[test]
fn compiled_tail_spreads_templates_and_cross_tier_calls() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var order=[];
            function collect(...args){return args.join(':');}
            function spread(){'use strict';return collect(...[1,2],(order.push('arg'),3),...[4]);}
            var tag={value:10,f(strings,x){return this.value+x+strings.raw.length;}};
            function tagged(){'use strict';return tag.f`a${4}b`;}
            function rebound(f){'use strict';return f(3);}
            var bound=(function(x){return this.base+x;}).bind({base:8});
            var proxy=new Proxy(bound,{apply(f,t,a){return Reflect.apply(f,t,a)+1}});
            spread()+'|'+tagged()+'|'+rebound(bound)+'|'+rebound(proxy)+'|'+order.join(',')
        "#
            ),
            "1:2:3:4|16|11|12|arg"
        );
        assert_compiled(&mut engine, "spread", false);
        assert_compiled(&mut engine, "tagged", false);
        assert_compiled(&mut engine, "rebound", matches!(tier, Tier::Jit));
        assert!(engine.interp.pending_tail.is_none());
    }
}

#[test]
fn compiled_tail_general_entry_and_overridden_eval_retire_frames() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            'use strict';
            function rest(n,...args){checkTailDepth();return n===0?args[0]:rest(n-1,args[0]+1)}
            function destructure({n,a}){checkTailDepth();return n===0?a:destructure({n:n-1,a:a+1})}
            var original=eval;
            function direct(n){checkTailDepth();return eval(n)}
            direct('1+2');
            globalThis.eval=function(n){checkTailDepth();return n===0?0:direct(n-1)};
            var answer=rest(4096,0)+destructure({n:4096,a:0})+direct(4096);
            globalThis.eval=original;
            answer
        "#
            ),
            "8192"
        );
        assert_compiled(&mut engine, "rest", matches!(tier, Tier::Jit));
        assert_compiled(&mut engine, "destructure", matches!(tier, Tier::Jit));
        assert_compiled(&mut engine, "direct", false);
        assert!(engine.interp.pending_tail.is_none());
    }
}

#[test]
fn compiled_tail_optional_super_calls_preserve_the_reference_receiver() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
            var log=[];
            class Base {
                constructor(){this.value=40}
                get step(){log.push('get');return function(x){log.push(this.value);return this.value+x}}
            }
            class Derived extends Base {
                constructor(){var value=super()?.value;log.push(value)}
                step(...args){return super.step?.(...args)}
                computed(...args){return super[(log.push('key'),'step')]?.(...args)}
                missing(){return super.absent?.(log.push('bad'))}
            }
            var object=new Derived();
            var step=object.step,computed=object.computed,missing=object.missing;
            [step.call(object,2),computed.call(object,3),missing.call(object),log.join(',')].join('|')
        "#
            ),
            "42|43||40,get,40,key,get,40"
        );
        assert_compiled(&mut engine, "step", false);
        assert_compiled(&mut engine, "computed", false);
        assert_compiled(&mut engine, "missing", false);
    }
}

/// Shallow tail calls run as ordinary calls, so their ordinary form shares call caches,
/// direct calls and speculative splices; a splice's guard still falls back when the target
/// changes. Past `ordinary_tail_call_depth` (here below 1500 non-tail frames, and along an
/// unbounded mutual recursion) every tail call retires its caller, keeping the logical depth
/// bounded. Shallow expected values are Node's; Node has no proper tail calls, so the deep
/// results are the closed forms (30000, and 100001 is odd).
#[test]
fn shallow_tail_calls_are_ordinary_and_deep_ones_stay_bounded() {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        assert_eq!(
            evaluate(
                &mut engine,
                r#"
'use strict';
var log = [];
class P { constructor(i) { this.id = i } tw() { return this.id * 2 } twice() { return this.tw() } }
var objs = []; for (var k = 0; k < 16; k++) objs.push(new P(k));
var s = 0; for (var r = 0; r < 300; r++) for (var k = 0; k < 16; k++) s += objs[k].twice();
log.push(s);
P.prototype.tw = function () { return -this.id; };
s = 0; for (var k = 0; k < 16; k++) s += objs[k].twice();
log.push(s);
function idf(x) { return x; }
function viaCond(x) { return idf(x) === x ? idf(x * 3) : -1; }
function viaLogical(x) { return x > 5 && idf(x + 1); }
function viaSeq(x) { return (idf(0), idf(x - 1)); }
function spread(a) { return Math.max(...a); }
var t = 0;
for (var k = 0; k < 400; k++) t += viaCond(k) + viaLogical(k) + viaSeq(k) + spread([k, 2, 3]);
log.push(t);
function thrower() { throw new RangeError('x'); }
function callsThrower() { return thrower(); }
try { callsThrower(); } catch (e) { log.push(e.name); }
function notCallable() { var o = {}; return o.missing(); }
try { notCallable(); } catch (e) { log.push(e.constructor.name); }
function argsOrder() { var order = []; function f(a, b) { return order.join('') + a + b; } return f(order.push('a') && 'x', order.push('b') && 'y'); }
log.push(argsOrder());
{
  function down(n, m) { return n === 0 ? loop(m, 0) : 1 + down(n - 1, m) - 1; }
  function loop(m, acc) { return m === 0 ? acc : loop(m - 1, acc + 1); }
  log.push(down(1500, 30000));
  function even(n) { checkTailDepth(); return n === 0 ? 'even' : odd(n - 1); }
  function odd(n) { return n === 0 ? 'odd' : even(n - 1); }
  log.push(even(100001));
}
log.join(',');
"#
            ),
            "72000,-120,478785,RangeError,TypeError,abxy,30000,odd",
            "{tier:?}"
        );
        assert!(engine.interp.pending_tail.is_none());
    }
}
