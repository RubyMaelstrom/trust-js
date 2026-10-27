use super::*;
use crate::interpreter::Interp;
use crate::value::Value;
use crate::value::ValueView;

// ECMA-262 Object Type / ToPropertyKey (local snapshot e28783d5fc9d): all Strings,
// including leading NULs, are keys distinct from every Symbol. These regressions
// deliberately use the old Symbol/internal-marker spelling in author strings.
const STRING_SYMBOL_KEYS: &str = r#"
function check(ok, label) { if (!ok) throw new Error(label); }
function exercise() {
    var s = Symbol('s'), o = {[s]: 'symbol', [Symbol.iterator]: 'iterator'};
    var names = ['\0', '\0\0', '\0raw_json', '\0marker', '\0+1', '#\0co_locale', '#p\u00011'];
    for (var j = 0; j < 128; ++j) names.push('\0' + j);
    for (var j = 0; j < names.length; ++j) o[names[j]] = j;
    var strings = Object.keys(o);
    check(JSON.stringify(strings) === JSON.stringify(names), 'Object.keys');
    check(JSON.stringify(Object.getOwnPropertyNames(o)) === JSON.stringify(names), 'names');
    check(Object.getOwnPropertySymbols(o).length === 2, 'symbols');
    check(o[s] === 'symbol' && o[Symbol.iterator] === 'iterator', 'identity');
    var keys = Reflect.ownKeys(o);
    check(keys.length === names.length + 2 && keys[names.length] === s, 'Reflect keys');
    var entries = Object.entries(o), values = Object.values(o), iterated = [];
    for (var k in o) iterated.push(k);
    check(JSON.stringify(iterated) === JSON.stringify(names), 'for-in');
    for (var j = 0; j < names.length; ++j) {
        var k = names[j];
        check(o[k] === j && Reflect.get(o, k) === j && k in o, 'read/has');
        check(Object.prototype.hasOwnProperty.call(o, k) && Object.hasOwn(o, k), 'own');
        check(Object.getOwnPropertyDescriptor(o, k).value === j, 'descriptor');
        check(entries[j][0] === k && entries[j][1] === j && values[j] === j, 'entries');
    }
    var spread = {...o}, assigned = Object.assign({}, o);
    var created = Object.create(null, Object.getOwnPropertyDescriptors(o));
    for (var j = 0; j < names.length; ++j) {
        check(spread[names[j]] === j && assigned[names[j]] === j && created[names[j]] === j, 'copy');
    }
    check(spread[s] === 'symbol' && assigned[s] === 'symbol' && created[s] === 'symbol', 'copy symbol');
    var wanted = '\0\0method';
    var literal = {'\0\0method': function(){}, '\0get': 2};
    var incremental = {get '\0get'() { return 3; }, [wanted]() { return 4; }};
    var {'\0\0method': picked, ...rest} = literal;
    check(picked.name === wanted && rest['\0get'] === 2 && !Object.hasOwn(rest, wanted), 'rest/static');
    check(literal[wanted].name === wanted && incremental[wanted].name === wanted, 'function names');
    check(Object.getOwnPropertyDescriptor(incremental, '\0get').get.name === 'get \0get', 'getter name');
    var opaque = '#p\u00011';
    class Named {
        [wanted]() {}
        [opaque]() {}
        field = function() {};
        static [wanted] = function() {};
    }
    check(new Named()[wanted].name === wanted && new Named()[opaque].name === opaque, 'class method names');
    check(Named[wanted].name === wanted, 'class field names');
    var proto = {[wanted]: 7}, derived = Object.create(proto);
    check(derived[wanted] === 7 && wanted in derived, 'prototype');
    for (var holder of [[], new Uint8Array(0), new String('')]) {
        holder[wanted] = 8;
        check(holder[wanted] === 8 && Object.keys(holder)[0] === wanted, 'exotic expandos');
        check(Object.entries(holder)[0][0] === wanted, 'exotic entries');
        delete holder[wanted];
        check(!Object.hasOwn(holder, wanted), 'exotic delete');
    }
    for (var j = 0; j < names.length; ++j) delete o[names[j]];
    check(Object.keys(o).length === 0 && o[s] === 'symbol' && o[Symbol.iterator] === 'iterator', 'delete isolation');
    return 'ok';
}
exercise(); exercise(); exercise()
"#;

const PROXY_STRING_SYMBOL_KEYS: &str = r#"
function check(ok, label) { if (!ok) throw new Error(label); }
function exerciseProxy() {
    var s = Symbol('s'), raw = '\0\0raw', log = [];
    var target = {[s]: 9, [raw]: 1, '\u00000': 2};
    var p = new Proxy(target, {
        get(t,k,r) { log.push(k); return Reflect.get(t,k,r); },
        set(t,k,v,r) { log.push(k); return Reflect.set(t,k,v,r); },
        has(t,k) { log.push(k); return Reflect.has(t,k); },
        getOwnPropertyDescriptor(t,k) { log.push(k); return Reflect.getOwnPropertyDescriptor(t,k); },
        defineProperty(t,k,d) { log.push(k); return Reflect.defineProperty(t,k,d); },
        deleteProperty(t,k) { log.push(k); return Reflect.deleteProperty(t,k); },
        ownKeys(t) { return Reflect.ownKeys(t); }
    });
    check(p[raw] === 1 && Reflect.get(p,raw) === 1 && raw in p, 'proxy get/has');
    check(Object.hasOwn(p,raw) && Object.prototype.hasOwnProperty.call(p,raw), 'proxy own');
    check(Object.getOwnPropertyDescriptor(p,raw).value === 1, 'proxy descriptor');
    p[raw] = 3;
    Object.defineProperty(p,raw,{value:4,writable:true,enumerable:true,configurable:true});
    check(log.every(function(k) { return k === raw; }), 'trap string identity');
    log = [];
    check(p[s] === 9 && log.length === 1 && log[0] === s, 'trap symbol identity');
    var names = Object.keys(p), entries = Object.entries(p), copy = {...p}, iterated = [];
    for (var k in p) iterated.push(k);
    check(JSON.stringify(names) === JSON.stringify([raw,'\u00000']), 'proxy names');
    check(entries[0][0] === raw && entries[0][1] === 4 && copy[raw] === 4 && copy[s] === 9, 'proxy copy');
    check(JSON.stringify(iterated) === JSON.stringify(names), 'proxy for-in');
    delete p[raw];
    check(!Object.hasOwn(target,raw) && target[s] === 9, 'proxy delete');
    return 'ok';
}
exerciseProxy(); exerciseProxy(); exerciseProxy()
"#;

const JSON_STRING_KEYS: &str = r#"
function check(ok, label) { if (!ok) throw new Error(label); }
function exerciseJSON() {
    var names = ['\u00000', '\0\0', '\0raw_json', '#\0co_locale'];
    var o = {}, log = [];
    for (var j = 0; j < names.length; ++j) o[names[j]] = j;
    var text = JSON.stringify(o);
    var parsed = JSON.parse(text, function(k,v,c) {
        if (k !== '') { log.push(k); check(c.source === String(v), 'source'); }
        return v;
    });
    check(JSON.stringify(log) === JSON.stringify(names), 'reviver');
    check(JSON.stringify(Object.keys(parsed)) === JSON.stringify(names), 'parsed keys');
    check(JSON.stringify(parsed) === text && parsed['\u00000'] === 0, 'parse roundtrip');
    var seen = [];
    check(JSON.stringify(o, function(k,v) { if(k !== '') seen.push(k); return v; }) === text, 'replacer text');
    check(JSON.stringify(seen) === JSON.stringify(names), 'replacer names');
    check(JSON.stringify(o,names) === text, 'replacer list');
    check(JSON.stringify(new Proxy(o,{})) === text, 'proxy stringify');
    return 'ok';
}
exerciseJSON(); exerciseJSON(); exerciseJSON()
"#;

#[test]
fn author_nul_strings_and_symbol_keys_are_distinct_in_all_tiers() {
    for tier in [
        crate::bytecode::Tier::Interp,
        crate::bytecode::Tier::Bytecode,
        crate::bytecode::Tier::Jit,
    ] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        for source in [
            STRING_SYMBOL_KEYS,
            PROXY_STRING_SYMBOL_KEYS,
            JSON_STRING_KEYS,
        ] {
            assert_eq!(run_in(&mut engine, source), "ok", "{tier:?}");
        }
    }
}

#[test]
fn encoded_string_keys_never_claim_symbol_ownership() {
    for text in ["\0", "\x000", "\x00123", "\0\0", "\0raw_json", "plain"] {
        let key = crate::value::PropertyKey::string(text.to_string());
        assert!(!Interp::is_sym_key(&key));
        assert_eq!(crate::value::decode_string_key(&key), text);
        assert!(matches!((key.into_value()).view(), ValueView::Str(value) if value.text() == text));
    }
}

#[test]
fn host_property_boundaries_encode_author_nul_names() {
    fn host_getter(_ctx: &mut Interp, _this: Value, _args: &[Value]) -> Result<Value, Value> {
        Ok(Value::Num(17.0))
    }
    let mut engine = Engine::new();
    let global = engine.interp.global_this();
    assert!(engine
        .interp
        .member_set(&global, "\0host", Value::Num(9.0))
        .is_ok());
    assert!(matches!(
        (engine.interp.member_get(&global, "\0host")).map(|__value| (__value).view()),
        Ok(ValueView::Num(9.0))
    ));
    let getter = engine.interp.make_native("host_getter", 0, host_getter);
    engine
        .interp
        .define_accessor_value(&global, "\0getter", Some(Value::Obj(getter)), None, true);
    engine
        .interp
        .def_method(&engine.interp.global.clone(), "\0method", 0, host_getter);
    assert_eq!(
        run_in(
            &mut engine,
            r#"[globalThis['\0host'], globalThis['\0getter'], globalThis['\0method']()].join(',')"#
        ),
        "9,17,17"
    );
}

#[test]
fn module_namespace_nul_names_remain_strings() {
    for tier in [
        crate::bytecode::Tier::Interp,
        crate::bytecode::Tier::Bytecode,
        crate::bytecode::Tier::Jit,
    ] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        let completion = engine.eval_module(
            r#"import * as ns from 'dep';
               globalThis.moduleKeyResult = ns['\u00000'] + ',' + ns['\0\0name'] + ',' +
                 JSON.stringify(Object.getOwnPropertyNames(ns)) + ',' + Object.getOwnPropertySymbols(ns).length;"#,
            "entry",
            |specifier, _| (specifier == "dep").then(|| (
                "dep".to_string(),
                r#"const a = 2, b = 3; export { a as "\u00000", b as "\0\0name" };"#.to_string(),
            )),
        ).expect("module parsed");
        assert!(
            matches!(completion, crate::Completion::Value(_)),
            "module evaluated"
        );
        assert_eq!(
            run_in(&mut engine, "moduleKeyResult"),
            r#"2,3,["\u0000\u0000name","\u00000"],1"#,
            "{tier:?}"
        );
    }
}

#[test]
fn public_host_string_apis_cannot_forge_symbol_property_keys() {
    for tier in [
        crate::bytecode::Tier::Interp,
        crate::bytecode::Tier::Bytecode,
        crate::bytecode::Tier::Jit,
    ] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert_eq!(
            run_in(
                &mut engine,
                r#"
            var hostSymbol = Symbol('host'), hostObject = {[hostSymbol]: 17};
            var hostKeyLog = [];
            var hostProxy = new Proxy(hostObject, {
                get(t, k, r) { hostKeyLog.push(k); return Reflect.get(t, k, r); },
                set(t, k, v, r) { hostKeyLog.push(k); return Reflect.set(t, k, v, r); }
            });
            'ready'
        "#
            ),
            "ready"
        );
        let global = engine.interp.global_this();
        let symbol = engine
            .interp
            .member_get(&global, "hostSymbol")
            .ok()
            .expect("host symbol");
        let ValueView::Sym(symbol) = (symbol).view() else {
            panic!("expected a Symbol");
        };
        let object = engine
            .interp
            .member_get(&global, "hostObject")
            .ok()
            .expect("host object");
        let proxy = engine
            .interp
            .member_get(&global, "hostProxy")
            .ok()
            .expect("host proxy");
        let forged = format!("\0{}", symbol.id);
        assert!(matches!(
            (engine.interp.get_member(&object, &forged)).map(|__value| (__value).view()),
            Ok(ValueView::Undefined)
        ));
        assert!(matches!(
            (engine.interp.get_member(&proxy, &forged)).map(|__value| (__value).view()),
            Ok(ValueView::Undefined)
        ));
        assert!(engine
            .interp
            .set_member(&object, &forged, Value::Num(99.0))
            .is_ok());
        assert!(engine
            .interp
            .set_member(&proxy, &forged, Value::Num(100.0))
            .is_ok());
        assert!(matches!(
            (engine.interp.member_get(&object, &forged)).map(|__value| (__value).view()),
            Ok(ValueView::Num(100.0))
        ));
        assert!(engine
            .interp
            .member_set(&global, "forgedHostKey", Value::lstr(forged))
            .is_ok());
        assert_eq!(
            run_in(
                &mut engine,
                r#"
            [hostObject[hostSymbol], hostObject[forgedHostKey],
             Object.getOwnPropertySymbols(hostObject)[0] === hostSymbol,
             Object.keys(hostObject)[0] === forgedHostKey,
             hostKeyLog.every(k => k === forgedHostKey)].join(',')
        "#
            ),
            "17,100,true,true,true",
            "{tier:?}"
        );
        // A second leading NUL must also round-trip exactly once through public String APIs.
        assert!(engine
            .interp
            .set_member(&object, "\0\0host", Value::Num(8.0))
            .is_ok());
        assert!(matches!(
            (engine.interp.get_member(&object, "\0\0host")).map(|__value| (__value).view()),
            Ok(ValueView::Num(8.0))
        ));
    }
}
