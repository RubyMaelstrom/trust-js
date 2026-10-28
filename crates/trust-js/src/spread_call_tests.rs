//! Spread calls (W3's item 6, `bytecode::jit_call_spread`): ArgumentListEvaluation's final spread
//! (ECMA-262 §13.3.8.1) of an Array whose intrinsic iteration runs no code, read in place and
//! called through the site's call cache, and every case that keeps the iterator protocol: holes,
//! getters, a replaced @@iterator or %ArrayIteratorPrototype%.next, Proxies and other iterables,
//! long lists, and EvaluateCall's order of argument evaluation and IsCallable.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

/// Run `source` at every tier; it must complete with "ok". At the JIT tier, `inert` says whether
/// some spread call must have been made from `jit_call_spread`'s own frame.
fn check(source: &str, inert: bool) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        crate::bytecode::TEST_INERT_SPREAD_CALLS.with(|calls| calls.set(0));
        match engine
            .eval(source, false)
            .expect("spread call fixture parses")
        {
            Completion::Value(ref value) if value == "ok" => {}
            Completion::Value(value) => panic!("{tier:?}: {value}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        if inert && matches!(tier, Tier::Jit) {
            let calls = crate::bytecode::TEST_INERT_SPREAD_CALLS.with(|calls| calls.get());
            assert!(calls > 0, "no spread call read its Array in place");
        }
    }
}

const ASSERT: &str = "function assert(value, message) { if (!value) throw Error(message); }\n";

#[test]
fn spread_calls_pass_an_arrays_elements_as_its_iteration_does() {
    check(
        &format!(
            "{ASSERT}{}",
            r#"
            function add2(a, b) { return a + b; }
            function all() { return Array.prototype.join.call(arguments, '|') + '#' + arguments.length; }
            function rest(a, ...more) { return a + ':' + more.join(); }
            var o = { base: 10, m(a, b) { return this.base + a + b; } };
            var arr3 = [1, 2, 3];
            for (var k = 0; k < 30; k++) {
                assert(add2(...arr3) === 3, 'spread');
                assert(all(0, ...arr3) === '0|1|2|3#4', 'plain arguments first');
                assert(all(...[]) === '#0' && all(1, ...[]) === '1#1', 'empty');
                assert(o.m(...[1, 2]) === 13, 'method');
                assert(all(...['a', undefined, null, -0, NaN]) === 'a|||0|NaN#5', 'values');
                assert(Math.max(...arr3) === 3 && Math.max(...[]) === -Infinity, 'native');
                assert(rest(...[1, 2, 3]) === '1:2,3', 'rest');
            }
            var w = { base: 5, m(a) { return this.base + a; } };
            with (w) { assert(m(...[1]) === 6, 'with'); }
            var bound = all.bind(null, 'b');
            assert(bound(...[1, 2]) === 'b|1|2#3', 'bound');
            var proxied = new Proxy(all, { apply(t, thisArg, a) { return 'trap:' + a.length; } });
            assert(proxied(...[1, 2]) === 'trap:2', 'proxy callee');
            function* g2(a) { yield a; }
            assert(g2(...[5]).next().value === 5, 'generator callee');
            'ok'
        "#
        ),
        true,
    );
}

#[test]
fn other_spreads_keep_the_iterator_protocol() {
    check(
        &format!(
            "{ASSERT}{}",
            r#"
            function args3(a, b, c) { return [a, b, c, arguments.length].join(); }
            for (var k = 0; k < 20; k++) assert(args3(...[1, , 3]) === '1,,3,3', 'hole');
            Array.prototype[1] = 'inherited';
            assert(args3(...[1, , 3]) === '1,inherited,3,3', 'inherited hole');
            delete Array.prototype[1];
            var getterArray = [1, 2, 3], gets = 0;
            Object.defineProperty(getterArray, 1, { get() { gets++; return 'g'; } });
            assert(args3(...getterArray) === '1,g,3,3' && gets === 1, 'getter');
            var own = [1, 2];
            own[Symbol.iterator] = function* () { yield 'own'; };
            assert(args3(...own) === 'own,,,1', 'own iterator');
            var saved = Array.prototype[Symbol.iterator];
            Array.prototype[Symbol.iterator] = function* () { yield 'proto'; };
            assert(args3(...[1, 2]) === 'proto,,,1', 'prototype iterator');
            Array.prototype[Symbol.iterator] = saved;
            var arrayIteratorProto = Object.getPrototypeOf([][Symbol.iterator]());
            var savedNext = arrayIteratorProto.next;
            arrayIteratorProto.next = function () { return { done: true }; };
            assert(args3(...[1, 2]) === ',,,0', 'next');
            arrayIteratorProto.next = savedNext;
            assert(args3(...[1, 2]) === '1,2,,2', 'restored');
            assert(args3(...new Proxy([4, 5], {})) === '4,5,,2', 'proxy');
            assert(args3(...'xy') === 'x,y,,2' && args3(...new Set([7])) === '7,,,1', 'other iterables');
            function spreadArguments() { return args3(...arguments); }
            assert(spreadArguments(8, 9) === '8,9,,2', 'arguments');
            assert(args3(...new Int8Array([1, 2])) === '1,2,,2', 'typed array');
            var long = [];
            for (var k = 0; k < 40; k++) long.push(k);
            function count() { return arguments.length + ':' + arguments[39]; }
            assert(count(...long) === '40:39', 'long');
            'ok'
        "#
        ),
        false,
    );
}

#[test]
fn spread_calls_evaluate_arguments_before_their_callee_check() {
    check(
        &format!(
            "{ASSERT}{}",
            r#"
            var log = [];
            function* gen() { log.push('iterate'); yield 1; }
            try { undefined(log.push('plain'), ...gen()); } catch (e) { log.push(e instanceof TypeError); }
            assert(log.join() === 'plain,iterate,true', log.join());
            var notCallable = {};
            for (var k = 0; k < 3; k++) {
                try { notCallable(...[1]); assert(false, 'called'); } catch (e) { assert(e instanceof TypeError, 'not callable'); }
            }
            class C {}
            for (var k = 0; k < 3; k++) {
                try { C(...[1]); assert(false, 'class called'); } catch (e) { assert(e instanceof TypeError, 'class'); }
            }
            function thrower(a) { throw new Error('boom ' + a); }
            var caught = 0;
            for (var k = 0; k < 20; k++) {
                try { thrower(...[k]); } catch (e) { caught += e.message === 'boom ' + k; }
            }
            assert(caught === 20, 'throws through');
            'ok'
        "#
        ),
        true,
    );
}
