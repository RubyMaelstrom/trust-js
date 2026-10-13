//! ECMA-262 e28783d5, spec.html: Object Initializer 19189–19283,
//! ToPropertyKey 5752, CreateDataPropertyOrThrow 6311, SetFunctionName 14194.
//! https://tc39.es/ecma262/#sec-runtime-semantics-propertydefinitionevaluation

use crate::bytecode::Tier;
use crate::value::Value;
use crate::{Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        engine.interp.def_method(
            &engine.interp.global,
            "collectLiteralTest",
            0,
            |interp, _, _| {
                interp.gc_collect();
                Ok(Value::Undefined)
            },
        );
        match engine.eval(source, false).expect("literal fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        engine.interp.gc_collect();
        assert!(matches!(engine.eval("1 + 2", false), Ok(Completion::Value(v)) if v == "3"));
    }
}

#[cfg(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
#[test]
fn object_literal_builders_have_real_native_entries() {
    use crate::value::Callable;
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
        .eval(
            r#"
            function create(i) { var o = Object.create(null); o.value = i; return o; }
            function nullLiteral(i) { return {__proto__: null, value: i}; }
            function computedLiteral(i) { return {[i]: i}; }
            function numericLiteral(i) { return {0: i, 1: i + 1}; }
            function protoLiteral(i) { return {__proto__: i, value: i}; }
            "#,
            false,
        )
        .unwrap();
    for name in [
        "create",
        "nullLiteral",
        "computedLiteral",
        "numericLiteral",
        "protoLiteral",
    ] {
        let global = Value::Obj(engine.interp.global.clone());
        let function = engine
            .interp
            .get_member(&global, name)
            .unwrap_or_else(|_| panic!("{name}"));
        // The hot-call counter is updated at compiled call sites, not direct Rust calls.
        let calls = format!("(function() {{ for (var n=0;n<16;n++) {name}(n); return true; }})()");
        assert!(matches!(engine.eval(&calls, false), Ok(Completion::Value(v)) if v == "true"));
        let Value::Obj(object) = function else {
            panic!("function object")
        };
        let chunk = match &object.borrow().call {
            Callable::User(user) => user
                .func
                .code
                .get()
                .and_then(Option::as_ref)
                .unwrap()
                .clone(),
            _ => panic!("ordinary user function"),
        };
        assert!(
            chunk.jit.get().is_some_and(|code| code.is_some()),
            "{name}: must not silently fall back to the VM"
        );
        assert!(
            chunk.jit_runs.get() > 0,
            "{name}: machine code must execute"
        );
    }
}

#[test]
fn object_literal_builder_keys_prototypes_descriptors_and_names() {
    check(
        r#"
        function build(proto, key, sym) {
            return {__proto__: proto, a: 1, [key]: 2, a: 3,
                ['__proto__']: 4, [sym]: function() {},
                f: function() {}, arrow: () => 7, 10: 10, 2: 2};
        }
        var sym = Symbol('method'), proto = {a: 90}, out = [];
        Object.defineProperty(proto, 'a', {set() { throw 'inherited setter'; }});
        for (var n = 0; n < 120; n++) build(proto, 'b', sym);
        var o = build(proto, 'b', sym), d = Object.getOwnPropertyDescriptor(o, 'a');
        out.push(Object.getPrototypeOf(o) === proto, o.a, o.b, o.__proto__,
            d.writable, d.enumerable, d.configurable, o[sym].name, o.f.name, o.arrow.name,
            Reflect.ownKeys(o).map(k => typeof k === 'symbol' ? 'symbol' : k).join(','));
        var nil = build(null, 'b', sym), primitive = build(5, 'b', sym);
        out.push(Object.getPrototypeOf(nil) === null,
            Object.getPrototypeOf(primitive) === Object.prototype);
        function anonymousProto() { return {__proto__: function() {}}; }
        out.push(Object.getPrototypeOf(anonymousProto()).name === '');
        out.join('|')
        "#,
        "true|3|2|4|true|true|true|[method]|f|arrow|2,10,a,b,__proto__,f,arrow,symbol|true|true|true",
    );
}

#[test]
fn object_literal_builder_coercion_order_gc_and_abrupt_cleanup() {
    check(
        r#"
        var log = [], sentinel = {}, failKey = false, failValue = false;
        var key = {[Symbol.toPrimitive](hint) {
            log.push('key:' + hint); collectLiteralTest();
            if (failKey) throw sentinel;
            return 'value';
        }};
        function value() {
            log.push('value');
            if (failValue) throw sentinel;
            var o = {tag: 42}; o.self = o; return o;
        }
        function later() { log.push('later'); collectLiteralTest(); return 7; }
        function build() { return {__proto__: null, [key]: value(), later: later()}; }
        function attempt() {
            try { return build(); }
            catch (e) { log.push(e === sentinel ? 'caught' : 'wrong'); return null; }
        }
        var first = attempt(), out = [log.join(','), first.value.tag,
            first.value.self === first.value, first.later, Object.getPrototypeOf(first) === null];
        failKey = true; log = []; out.push(attempt() === null, log.join(','));
        failKey = false; failValue = true; log = []; out.push(attempt() === null, log.join(','));
        failValue = false; log = []; var last = attempt();
        out.push(log.join(','), last.value.tag, last.value !== first.value);
        out.join('|')
        "#,
        "key:string,value,later|42|true|7|true|true|key:string,caught|true|key:string,value,caught|key:string,value,later|42|true",
    );
}
