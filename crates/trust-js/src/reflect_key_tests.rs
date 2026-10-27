//! Reflect.has (ECMA-262 §28.1.9) through the `in` operator's HasProperty: keys taken without
//! a key string, with the steps it may not skip.

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
