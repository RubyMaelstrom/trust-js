//! Callee-local fast paths in Agents containing proxies. ECMA-262 snapshot e28783d5:
//! Proxy [[Call]] / [[Construct]] (spec.html:16541,16568), Object.hasOwn (31428).

use std::rc::Rc;

use crate::bytecode::{Chunk, Tier};
use crate::value::{Callable, Gc, Value};
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("guard fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn object(engine: &mut Engine, name: &str) -> Gc {
    let env = engine.interp.global_env.clone();
    let Value::Obj(object) = engine
        .interp
        .get_var(name, &env)
        .unwrap_or_else(|_| panic!("missing {name}"))
    else {
        panic!("{name} is not an object")
    };
    object
}

fn chunk(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    let object = object(engine, name);
    let object = object.borrow();
    let Callable::User(user) = &object.call else {
        panic!("{name} is not a user function")
    };
    let chunk = user
        .func
        .code2
        .get()
        .or_else(|| user.func.code.get())
        .and_then(Option::as_ref)
        .expect("compiled caller")
        .clone();
    assert!(
        chunk.jit.get().is_some_and(|code| code.is_some()),
        "{name} was not native"
    );
    chunk
}

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine, source), expected, "{tier:?}");
        assert!(engine.interp.pending_tail.is_none());
        assert!(engine.interp.fn_frames.is_empty());
        engine.interp.gc_collect();
        assert_eq!(eval(&mut engine, "1+2"), "3");
    }
}

#[test]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn proxies_do_not_disable_ordinary_call_and_construct_cache_fill() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    assert_eq!(
        eval(
            &mut engine,
            r#"
        var unrelated = new Proxy({}, {});
        function leaf(x) { return x+1; }
        function Box(x) { this.x=x; }
        var native = Math.sqrt;
        function driver(x) { return leaf(x)+native(4); }
        function maker(x) { return new Box(x); }
        function arrays() { return new Array(3).length; }
        var out;
        for(var n=0;n<80;n++) out=driver(n)+':'+maker(n).x+':'+arrays();
        out
    "#
        ),
        "82:79:3"
    );
    let driver = chunk(&mut engine, "driver");
    for name in ["leaf", "native"] {
        let target = object(&mut engine, name);
        let entry = driver
            .cached_call_for(Rc::as_ptr(&target) as usize)
            .unwrap_or_else(|| panic!("{name} did not fill its call cache with a live proxy"));
        assert_eq!(entry.native != 0, name == "native");
    }
    let maker = chunk(&mut engine, "maker");
    let target = object(&mut engine, "Box");
    assert!(
        maker.cached_construct_for(Rc::as_ptr(&target) as usize),
        "ordinary construction remained uncached"
    );
}

#[test]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn fresh_closure_call_cache_uses_the_live_function_identity() {
    // PrepareForOrdinaryCall sets the execution context's Function to the actual
    // callee (ECMA-262 e28783d5, spec.html:13873). Sharing code does not share that
    // identity. Error.stack is an implementation extension that reads this frame.
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            function factory(label) {
                var f = function (x) {
                    var local = x;
                    function capture() { return local; }
                    if (x < 0) Reflect.apply(ArrayBuffer, null, [8]);
                    return capture() + ':' + new Error('frame').stack;
                };
                Object.defineProperty(f, 'name', {value: label});
                return f;
            }
            function drive(f, x) { return f(x); }
            var original = factory('originalClosure');
            for (var i = 0; i < 80; i++) drive(original, i);
            var replacement = factory('replacementClosure');
        "#,
        );
        if matches!(tier, Tier::Jit) {
            let driver = chunk(&mut engine, "drive");
            let original = object(&mut engine, "original");
            let entry = driver
                .cached_call_for(Rc::as_ptr(&original) as usize)
                .expect("original closure filled the call cache");
            assert_ne!(entry.direct & crate::bytecode::CALL_IC_NEEDS_ENV, 0);
        }
        // Keep the old closure alive for a deterministic wrong-identity failure,
        // instead of depending on a use-after-free to crash the test process.
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var result = drive(replacement, 42);
            result.startsWith('42:') && result.includes('replacementClosure') &&
                !result.includes('originalClosure')
        "#
            ),
            "true",
            "{tier:?}"
        );
        assert_eq!(
            eval(
                &mut engine,
                r#"
            original = null;
            var correct = true;
            for (var i = 0; i < 32; i++) {
                var label = 'freshClosure' + i;
                var fresh = factory(label);
                var churn = Array.from({length: 32}, function () {
                    return {values: new Float64Array([1.5, 2.5, 3.5])};
                });
                var stack = drive(fresh, i);
                correct = correct && stack.startsWith(i + ':') && stack.includes(label);
                try { drive(fresh, -1); correct = false; }
                catch (e) {
                    correct = correct && e.name === 'TypeError' && e.stack.includes(label);
                }
            }
            correct
        "#
            ),
            "true",
            "{tier:?}: discarded cached closure and native exception"
        );
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
fn cached_calls_still_dispatch_proxy_traps_and_revocation() {
    check(
        r#"
        var calls=0, log=[];
        function leaf(x){return x+1}
        function drive(f,x){return f(x)}
        var rev=Proxy.revocable(leaf,{apply(t,receiver,a){calls++;return t(a[0])+10}});
        var native=new Proxy(Math.sqrt,{apply(t,receiver,a){return 100+t(a[0])}});
        for(var n=0;n<80;n++){drive(leaf,n);drive(rev.proxy,n)}
        var result=[calls,drive(native,4)];
        rev.revoke();
        try{drive(rev.proxy,(log.push('arg'),1))}catch(e){result.push(e.name)}
        result.push(drive(leaf,2),log.join(','));
        result.join('|')
    "#,
        "80|102|TypeError|3|arg",
    );
}

#[test]
fn cached_constructors_preserve_proxy_new_target_and_live_prototypes() {
    check(
        r#"
        var unrelated=new Proxy({},{}), hits=0;
        function Box(x){this.x=x}
        function make(C,x){return new C(x)}
        for(var n=0;n<160;n++)make(Box,n);
        var rev=Proxy.revocable(Box,{construct(t,a,nt){hits++;return {x:a[0]+10,nt:nt}}});
        var p=make(rev.proxy,2), out=[p.x,p.nt===rev.proxy,hits];
        Box.prototype={tag:8};out.push(make(Box,3).tag);
        var seen=0;
        Box.prototype=new Proxy({}, {set(t,k,v,r){seen++;return Reflect.set(t,k,v,r)}});
        out.push(make(Box,4).x,seen);
        var Native=new Proxy(Array,{construct(t,a,nt){return {length:a[0]+7}}});
        out.push(make(Native,2).length,make(Array,2).length);
        rev.revoke();try{make(rev.proxy,3)}catch(e){out.push(e.name)}
        var bad=new Proxy(Box,{construct(){return 1}});
        try{make(bad,1)}catch(e){out.push(e.name)}
        function Target(){this.nt=new.target}
        var wrapped=new Proxy(new Proxy(Target,{}),{});
        out.push(make(wrapped,0).nt===wrapped);
        out.join('|')
    "#,
        "12|true|1|8|4|1|9|2|TypeError|TypeError|true",
    );
}

#[test]
fn enabled_native_intrinsics_preserve_exotic_receivers_and_lists() {
    check(
        r#"
        var unrelated=new Proxy({},{}), events=[];
        function sum(a,b){return this.base+a+b}
        function apply(f,t,a){return f.apply(t,a)}
        function call(f,t,a){return f.call(t,a,2)}
        function push(a,x){return a.push(x)}
        var t={base:5}, a=[];
        for(var n=0;n<160;n++){apply(sum,t,[1,2]);call(sum,t,1);push(a,n);a.pop()}
        var fn=new Proxy(sum,{apply(target,receiver,args){events.push('apply');return 20+Reflect.apply(target,receiver,args)}});
        var list=new Proxy({length:2,0:1,1:2},{get(t,k,r){events.push(k);return Reflect.get(t,k,r)}});
        var out=[apply(fn,t,list),call(fn,t,1)];
        var p=new Proxy([],{set(t,k,v,r){events.push('set:'+k);return Reflect.set(t,k,v,r)}});
        out.push(push(p,7),p[0],events.join(','));
        out.join('|')
    "#,
        "28|28|1|7|length,0,1,apply,apply,set:0,set:length",
    );
}

#[test]
fn hasown_intrinsic_dispatches_getownproperty_after_warmup() {
    check(r#"
        var unrelated=new Proxy({},{}), reads=0;
        function own(o,k){return Object.hasOwn(o,k)}
        for(var n=0;n<160;n++)own({x:1},'x');
        var rev=Proxy.revocable({x:1},{getOwnPropertyDescriptor(t,k){reads++;return Reflect.getOwnPropertyDescriptor(t,k)}});
        var out=[own(rev.proxy,'x'),own(rev.proxy,'missing'),reads];
        var bytes=new Uint8Array(2);bytes['01']=3;
        out.push(own(bytes,'0'),own(bytes,'2'),own(bytes,'-0'),own(bytes,'01'));
        out.push(own('abc','1'),own('abc','length'),own(1,'x'),own(true,'x'),own(1n,'x'),own(Symbol(),'x'));
        var holes=[1,,3];out.push(own(holes,'0'),own(holes,'1'));
        rev.revoke();try{own(rev.proxy,'x')}catch(e){out.push(e.name)}
        var sealed=Object.freeze({x:1}), liar=new Proxy(sealed,{getOwnPropertyDescriptor(){return undefined}});
        try{own(liar,'x')}catch(e){out.push(e.name)}
        out.join('|')
    "#,"true|false|2|true|false|false|true|true|true|false|false|false|false|true|false|TypeError|TypeError");
}

#[test]
fn hasown_preserves_coercion_order_and_does_not_invoke_data_getters() {
    check(
        r#"
        var log=[], key={toString(){log.push('key');return 'x'}};
        try{Object.hasOwn(null,key)}catch(e){log.push('static:'+e.name)}
        try{Object.prototype.hasOwnProperty.call(null,key)}catch(e){log.push('proto:'+e.name)}
        var o={get x(){throw 'must not read value'}};
        log.push(Object.hasOwn(o,key));
        var s=Symbol();o[s]=2;log.push(Object.hasOwn(o,s));
        log.join('|')
    "#,
        "static:TypeError|key|proto:TypeError|key|true|true",
    );
}

#[test]
fn proxies_do_not_weaken_realm_or_class_call_guards() {
    check(
        r#"
        var unrelated=new Proxy({},{});
        var other=$262.createRealm().global;
        var foreign=other.eval('(function f(){return this})');
        function drive(f){return f()}
        function local(){return this}
        for(var n=0;n<160;n++){drive(local);drive(foreign)}
        var out=[drive(local)===globalThis,drive(foreign)===other];
        var foreignFail=other.eval('(function f(){throw new TypeError("other")})');
        try{drive(foreignFail)}catch(e){out.push(e instanceof other.TypeError,!(e instanceof TypeError))}
        class C{};try{drive(C)}catch(e){out.push(e.name)}
        out.push(drive(local)===globalThis);
        out.join('|')
    "#,
        "true|true|true|true|TypeError|true",
    );
}

/// ECMA-262 ProxyCreate step 5 and Call step 2: a proxy of a non-callable target has no
/// [[Call]], so calling it throws a TypeError without reading the handler's `apply` trap, on
/// every tier and at plain, method and spread call sites.
#[test]
fn calling_a_proxy_of_a_non_callable_target_never_reads_its_trap() {
    check(
        r#"
        var reads=0, handler={get apply(){reads++;return function(){return 'trap'}}};
        var p=new Proxy({}, handler), o={p:p}, out=[];
        function plain(){return p()}
        function method(){return o.p()}
        function spread(){return p(...[1])}
        for(var n=0;n<3;n++){
            for(var f of [plain,method,spread]){
                try{out.push(f())}catch(e){out.push(e instanceof TypeError)}
            }
        }
        out.push(reads, typeof p, typeof new Proxy(function(){}, {}));
        out.join('|')
    "#,
        "true|true|true|true|true|true|true|true|true|0|object|function",
    );
}
