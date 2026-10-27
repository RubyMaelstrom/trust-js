//! Reflect.get and Reflect.has (ECMA-262 §28.1.6, §28.1.9) through the ordinary property paths:
//! keys taken without a key string and reads through the property ICs' stub cache, each with the
//! steps they may not skip.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn check(source: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("Reflect fixture parses") {
            Completion::Value(ref value) if value == "ok" => {}
            Completion::Value(value) => panic!("{tier:?}: {value}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

const ASSERT: &str = "function assert(value, message) { if (!value) throw Error(message); }\n";

#[test]
fn reflect_get_reads_through_the_stub_cache_and_follows_changes() {
    check(&format!(
        "{ASSERT}{}",
        r#"
        var proto = { inherited: 'p', get viaGetter() { return this.tag; } };
        var obj = Object.create(proto);
        obj.own = 1;
        obj.tag = 'obj';
        var other = { tag: 'other' };
        function read(k) {
            assert(Reflect.get(obj, 'own') === 1, 'own');
            assert(Reflect.get(obj, 'inherited') === 'p', 'inherited');
            assert(Reflect.get(obj, 'viaGetter') === 'obj', 'getter on the target');
            assert(Reflect.get(obj, 'viaGetter', other) === 'other', 'getter on the receiver');
            assert(Reflect.get(obj, 'viaGetter', 7) === undefined, 'primitive receiver');
            assert(Reflect.get(obj, 'never a key ' + k) === undefined, 'text with no atom');
            assert(Reflect.get(obj, 'absent') === undefined, 'absent');
        }
        for (var k = 0; k < 40; k++) read(k);
        // A cached resolution follows every change to the chain.
        obj.inherited = 'shadowed';
        assert(Reflect.get(obj, 'inherited') === 'shadowed', 'shadow');
        delete obj.inherited;
        assert(Reflect.get(obj, 'inherited') === 'p', 'unshadow');
        proto.absent = 'added';
        assert(Reflect.get(obj, 'absent') === 'added', 'added to the prototype');
        Object.defineProperty(obj, 'own', { get() { return 'accessor'; }, configurable: true });
        assert(Reflect.get(obj, 'own') === 'accessor', 'redefined as an accessor');
        Object.setPrototypeOf(obj, { inherited: 'swapped' });
        assert(Reflect.get(obj, 'inherited') === 'swapped', 'prototype swapped');
        assert(Reflect.get(obj, 'viaGetter') === undefined, 'gone with the old prototype');
        // Keys of every kind, converted once and only after the target check.
        var s = Symbol('s');
        var keyed = { '\0lead': 'nul', 7: 'seven', [s]: 'sym', '1.5': 'fraction', length: 'own' };
        for (var k = 0; k < 3; k++) {
            assert(Reflect.get(keyed, '\0lead') === 'nul', 'NUL-led key');
            assert(Reflect.get(keyed, 7) === 'seven' && Reflect.get(keyed, '7') === 'seven', 'index');
            assert(Reflect.get(keyed, s) === 'sym' && Reflect.get(keyed, Symbol('s')) === undefined, 'symbol');
            assert(Reflect.get(keyed, 1.5) === 'fraction' && Reflect.get(keyed, 'length') === 'own', 'text');
            assert(Reflect.get({ 0: 'zero' }, -0) === 'zero', '-0 is "0"');
        }
        var log = [];
        var key = { toString() { log.push('key'); return 'own'; } };
        assert(Reflect.get({ own: 'converted' }, key) === 'converted' && log.join() === 'key', 'once');
        try { Reflect.get(1, key); assert(false, 'no TypeError'); } catch (e) { assert(e instanceof TypeError, 'TypeError'); }
        assert(log.join() === 'key', 'no conversion before the target check');
        // Exotic objects and Proxies keep their own [[Get]].
        assert(Reflect.get([1, 2, 3], 'length') === 3 && Reflect.get([1, 2, 3], 1) === 2, 'array');
        assert(Reflect.get([1, 2, 3], 'push') === Array.prototype.push, 'array method');
        assert(Reflect.get(new String('abc'), 'length') === 3 && Reflect.get(new String('abc'), 1) === 'b', 'String');
        assert(Reflect.get(new Uint8Array([5]), 0) === 5 && Reflect.get(new Uint8Array([5]), 'length') === 1, 'typed');
        var receiver = {};
        var proxy = new Proxy({ a: 1 }, { get(t, k, r) { return k + ':' + (r === receiver); } });
        assert(Reflect.get(proxy, 'a', receiver) === 'a:true', 'proxy target');
        var child = Object.create(new Proxy({}, { get(t, k, r) { return r === child ? 'via proxy' : 'wrong'; } }));
        for (var k = 0; k < 3; k++) assert(Reflect.get(child, 'x') === 'via proxy', 'proxy on the chain');
        function args(a) { arguments[0] = 'written'; return Reflect.get(arguments, 0) + a; }
        assert(args('given') === 'writtenwritten', 'arguments');
        'ok'
    "#
    ));
}

#[test]
fn reflect_has_shares_the_in_operator() {
    check(&format!(
        "{ASSERT}{}",
        r#"
        var sym = Symbol();
        var base = { inherited: 1 };
        var o = Object.create(base);
        o.own = 2;
        o[sym] = 3;
        o['\0nul'] = 4;
        for (var k = 0; k < 40; k++) {
            assert(Reflect.has(o, 'own') && Reflect.has(o, 'inherited') && Reflect.has(o, 'toString'), 'present');
            assert(!Reflect.has(o, 'absent') && !Reflect.has(o, 'never a key ' + k), 'absent');
            assert(Reflect.has(o, sym) && !Reflect.has(o, Symbol()) && Reflect.has(o, '\0nul'), 'symbols');
        }
        var holey = [1, , 3];
        assert(Reflect.has(holey, 0) && !Reflect.has(holey, 1) && Reflect.has(holey, '2'), 'elements');
        assert(Reflect.has(holey, 'length') && !Reflect.has(holey, 3) && Reflect.has(holey, 'push'), 'array');
        var str = new String('ab');
        assert(Reflect.has(str, 1) && Reflect.has(str, 'length') && !Reflect.has(str, 2), 'String');
        assert(Reflect.has({ 0: 1 }, -0) && Reflect.has({ '1.5': 1 }, 1.5) && Reflect.has({ '-1': 1 }, -1), 'numbers');
        var log = [];
        var p = new Proxy({}, { has(t, k) { log.push(typeof k + ':' + String(k)); return k === 'yes'; } });
        assert(Reflect.has(p, 'yes') && !Reflect.has(p, 1) && !Reflect.has(Object.create(p), 'no'), 'proxy');
        assert(!Reflect.has(p, sym), 'proxy symbol');
        assert(log.join() === 'string:yes,string:1,string:no,symbol:Symbol()', log.join());
        var converted = 0;
        var key = { toString() { converted++; return 'own'; } };
        try { Reflect.has(1, key); assert(false, 'no TypeError'); } catch (e) { assert(e instanceof TypeError, 'TypeError'); }
        assert(converted === 0 && Reflect.has(o, key) && converted === 1, 'converted once, after the check');
        'ok'
    "#
    ));
}
