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
fn checked_calls_cache_self_hosted_private_environments_in_multi_realm_agents() {
    // PrepareForOrdinaryCall uses [[Realm]], not the parent of [[Environment]].
    // Self-hosted methods deliberately bind their intrinsics in a parentless record.
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var foreign = $262.createRealm().global;
            var mappingMethod = Array.prototype.map;
            function callback(x) { return x + 1; }
            function mapDrive(values, fn) { return values.map(fn).join(','); }
            var result;
            for (var n=0; n<12; n++) result = mapDrive([1,,3], callback);
            result
        "#
            ),
            "2,,4",
            "{tier:?}"
        );
        let method = object(&mut engine, "mappingMethod");
        let method_record = method.borrow();
        let Callable::User(user) = &method_record.call else {
            panic!("self-hosted map")
        };
        assert!(user.func.self_hosted);
        assert!(user.env.borrow().parent.is_none());
        drop(method_record);
        if tier == Tier::Jit {
            let driver = chunk(&mut engine, "mapDrive");
            let entry = driver
                .cached_call_for(Gc::as_ptr(&method) as usize)
                .expect("private self-hosted method fills the native call cache");
            assert_eq!(entry.native, 0);
            assert_eq!(entry.realm, Gc::as_ptr(&engine.interp.global) as usize);
            assert_eq!(
                entry.global_env,
                Rc::as_ptr(&engine.interp.global_env) as usize
            );
            assert!(!entry.code.is_null());
            assert!(driver.jit_runs.get() > 0);
        }
        assert_eq!(
            eval(
                &mut engine,
                r#"
            var calls = 0;
            try {
                mapDrive([1,2,3], function (x) { calls++; if (x === 2) throw x; return x; });
            } catch (error) { result = error + ':' + calls; }
            var foreignResult = foreign.Array.prototype.map.call([4], callback);
            result + ':' + (Object.getPrototypeOf(foreignResult) === foreign.Array.prototype)
                + ':' + (Object.getPrototypeOf(mapDrive([5], callback)) === String.prototype)
        "#
            ),
            "2:2:true:true",
            "{tier:?}"
        );
        assert!(engine.interp.fn_frames.is_empty());
        engine.interp.gc_collect();
    }
}

#[test]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn checked_calls_use_creation_realm_after_host_environment_replacement() {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine.interp.switch_host_job_context(101);
    assert_eq!(
        eval(
            &mut engine,
            r#"
        var foreign = $262.createRealm().global;
        function realmLeaf(x) { return x + 3; }
        function realmDrive(fn, x) { return fn(x); }
        for (var n=0; n<12; n++) realmDrive(realmLeaf, n);
        'ready'
    "#
        ),
        "ready"
    );
    let leaf = object(&mut engine, "realmLeaf");
    let driver = chunk(&mut engine, "realmDrive");
    let old_environment = engine.interp.global_env.clone();
    let previous = driver
        .cached_call_for(Gc::as_ptr(&leaf) as usize)
        .expect("initial same-Realm cache fill");
    assert_eq!(previous.global_env, Rc::as_ptr(&old_environment) as usize);
    engine.interp.switch_host_job_context(102);
    assert!(!Rc::ptr_eq(&old_environment, &engine.interp.global_env));
    assert_eq!(
        eval(
            &mut engine,
            r#"
        Object.setPrototypeOf(realmLeaf, foreign.Function.prototype);
        realmDrive(realmLeaf, 5)
    "#
        ),
        "8"
    );
    let current = driver
        .cached_call_for(Gc::as_ptr(&leaf) as usize)
        .expect("creation Realm remains cacheable with a replaced global environment");
    assert_ne!(current.global_env, previous.global_env);
    assert_eq!(
        current.global_env,
        Rc::as_ptr(&engine.interp.global_env) as usize
    );
    assert_eq!(current.realm, Gc::as_ptr(&engine.interp.global) as usize);
    assert_eq!(
        eval(
            &mut engine,
            r#"
        var proxyCalls = 0;
        var proxy = new Proxy(realmLeaf, {apply(f,t,a) { proxyCalls++; return Reflect.apply(f,t,a); }});
        class ClassOnly {}
        var checks = realmDrive(proxy, 6) === 9 && proxyCalls === 1
            && realmDrive(realmLeaf.bind(null), 7) === 10;
        try { realmDrive(ClassOnly, 0); checks = false; }
        catch (error) { checks = checks && error instanceof TypeError; }
        foreign.eval('globalThis.foreignLeaf = function(x) { return [x+4, new TypeError()]; };');
        var foreignAnswer = realmDrive(foreign.foreignLeaf, 5);
        checks = checks && foreignAnswer[0] === 9
            && foreignAnswer[1] instanceof foreign.TypeError
            && !(foreignAnswer[1] instanceof TypeError);
        var scoped, scope = {marker: 11};
        with (scope) { scoped = function() { return marker; }; }
        checks = checks && realmDrive(scoped, 0) === 11;
        scope.marker = 12;
        checks && realmDrive(scoped, 0) === 12
    "#
        ),
        "true"
    );
    for name in ["proxy", "ClassOnly", "scoped"] {
        let target = object(&mut engine, name);
        assert!(
            driver
                .cached_call_for(Gc::as_ptr(&target) as usize)
                .is_none(),
            "{name} retains checked generic dispatch"
        );
    }
    assert!(engine.interp.fn_frames.is_empty());
    assert!(engine.interp.pending_tail.is_none());
    engine.interp.gc_collect();
}

#[test]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn checked_calls_borrow_last_callee_owner_through_gc_return_and_throw() {
    for mode in [1, 2] {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        eval(
            &mut engine,
            r#"
            var foreign = $262.createRealm().global;
            var token = {number: 17};
            function make(token) {
                return function (x) {
                    var home = x;
                    function capture() { return home; }
                    var number = token.number;
                    if (x) {
                        victim.prototype = null;
                        victim = null;
                        $262.gc();
                        if (x === 2) throw 'owned throw';
                    }
                    return number + capture();
                };
            }
            var victim = make(token);
            function checkedDrive(f, x) { return f((f=null,x)); }
            for (var n=0; n<12; n++) checkedDrive(victim, 0);
        "#,
        );
        let victim = object(&mut engine, "victim");
        let token = object(&mut engine, "token");
        let victim_weak = Gc::downgrade(&victim);
        let token_weak = Gc::downgrade(&token);
        let driver = chunk(&mut engine, "checkedDrive");
        let entry = driver
            .cached_call_for(Gc::as_ptr(&victim) as usize)
            .expect("capturing callee is cached");
        assert_ne!(
            entry.direct & crate::bytecode::CALL_IC_NEEDS_ENV,
            0,
            "activation-homed callee forces the checked helper rather than direct machine call"
        );
        drop((victim, token));
        eval(&mut engine, "token = null");
        let result = eval(
            &mut engine,
            &format!("try {{ checkedDrive(victim,{mode}) }} catch(error) {{ error }}"),
        );
        assert_eq!(result, if mode == 1 { "18" } else { "owned throw" });
        engine.interp.gc_collect();
        assert_eq!(
            victim_weak.strong_count(),
            0,
            "callee destructor after mode {mode}"
        );
        assert_eq!(
            token_weak.strong_count(),
            0,
            "captured payload destructor after mode {mode}"
        );
        assert!(engine.interp.fn_frames.is_empty());
    }
}

#[test]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn checked_calls_cache_deep_same_realm_closures_without_flattening_environments() {
    let reads = (0..10)
        .map(|n| format!("capture{n}"))
        .collect::<Vec<_>>()
        .join("+");
    let mut expression = format!("function (x) {{ return {reads}+x; }}");
    for n in (0..10).rev() {
        expression = format!("(function () {{ let capture{n}={n}; return {expression}; }})()");
    }
    let source = format!(
        r#"
        var foreign = $262.createRealm().global;
        var deep = {expression};
        function deepDrive(f, x) {{ return f(x); }}
        var answer;
        for(var n=0;n<12;n++) answer = deepDrive(deep, n);
        answer
    "#
    );
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(eval(&mut engine, &source), "56", "{tier:?}");
        let deep = object(&mut engine, "deep");
        let mut current = match &deep.borrow().call {
            Callable::User(user) => user.env.clone(),
            _ => panic!("deep user closure"),
        };
        let mut depth = 0;
        while !Rc::ptr_eq(&current, &engine.interp.global_env) {
            let next = current
                .borrow()
                .parent
                .clone()
                .expect("global root retained");
            current = next;
            depth += 1;
        }
        assert!(
            depth > 8,
            "fixture exercises the former bounded parent walk"
        );
        if tier == Tier::Jit {
            let driver = chunk(&mut engine, "deepDrive");
            let entry = driver
                .cached_call_for(Gc::as_ptr(&deep) as usize)
                .expect("deep same-Realm closure fills the native call cache");
            assert_eq!(
                entry.env,
                match &deep.borrow().call {
                    Callable::User(user) => Rc::as_ptr(&user.env),
                    _ => unreachable!(),
                }
            );
            assert!(driver.jit_runs.get() > 0);
        }
        assert_eq!(eval(&mut engine, "$262.gc(); deepDrive(deep, 20)"), "65");
        assert!(engine.interp.fn_frames.is_empty());
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
            .cached_call_for(Gc::as_ptr(&target) as usize)
            .unwrap_or_else(|| panic!("{name} did not fill its call cache with a live proxy"));
        assert_eq!(entry.native != 0, name == "native");
    }
    let maker = chunk(&mut engine, "maker");
    let target = object(&mut engine, "Box");
    assert!(
        maker.cached_construct_for(Gc::as_ptr(&target) as usize),
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
                .cached_call_for(Gc::as_ptr(&original) as usize)
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
