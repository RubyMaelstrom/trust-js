//! Function.prototype.call and apply (ECMA-262 §20.2.3.3, §20.2.3.1) entering a compiled target
//! as the ordinary call they make (W3's item 6, `Interp::call_forwarded`), and every case that
//! keeps the builtin: other targets, lists whose CreateListFromArrayLike (§7.3.20) runs code or
//! reads a prototype, non-callable receivers.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

/// Run `source` at every tier; it must complete with "ok". At the JIT tier, `forwarded` says
/// whether some call must have entered its target through `Interp::call_compiled_target`.
fn check(source: &str, forwarded: bool) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        crate::interpreter::TEST_COMPILED_TARGET_CALLS.with(|count| count.set(0));
        match engine
            .eval(source, false)
            .expect("call/apply fixture parses")
        {
            Completion::Value(ref value) if value == "ok" => {}
            Completion::Value(value) => panic!("{tier:?}: {value}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        if forwarded && matches!(tier, Tier::Jit) {
            let calls = crate::interpreter::TEST_COMPILED_TARGET_CALLS.with(|count| count.get());
            assert!(calls > 0, "no call/apply entered its target directly");
        }
    }
}

const ASSERT: &str = "function assert(value, message) { if (!value) throw Error(message); }\n";

#[test]
fn call_and_apply_enter_their_targets_as_ordinary_calls() {
    check(
        &format!(
            "{ASSERT}{}",
            r#"
            function sum(a, b, c) { return typeof this + ':' + [a, b, c].join() + ':' + arguments.length; }
            function strictThis() { 'use strict'; return this; }
            function sloppyThis() { return this; }
            function fwd() { return sum.apply(this, arguments); }
            function strictFwd() { 'use strict'; return sum.apply(this, arguments); }
            var o = { tag: 'o' };
            for (var k = 0; k < 30; k++) {
                assert(sum.call(o, 1, 2, 3) === 'object:1,2,3:3', 'call');
                assert(sum.call(o, 1) === 'object:1,,:1', 'fewer arguments');
                assert(sum.call(o, 1, 2, 3, 4) === 'object:1,2,3:4', 'more arguments');
                assert(sum.call() === 'object:,,:0', 'no arguments');
                assert(strictThis.call(5) === 5 && strictThis.call() === undefined, 'strict this');
                assert(strictThis.call(null) === null, 'strict null this');
                assert(typeof sloppyThis.call(5) === 'object' && sloppyThis.call(5).valueOf() === 5, 'boxed');
                assert(sloppyThis.call(null) === globalThis && sloppyThis.call() === globalThis, 'global');
                assert(sum.apply(o, [1, 2, 3]) === 'object:1,2,3:3', 'apply');
                assert(sum.apply(o) === 'object:,,:0' && sum.apply(o, null) === 'object:,,:0', 'no list');
                assert(sum.apply(o, undefined) === 'object:,,:0' && sum.apply(o, []) === 'object:,,:0', 'empty');
                assert(fwd.call(o, 7, 8) === 'object:7,8,:2', 'arguments list');
                assert(strictFwd.call(o, 7, 8, 9) === 'object:7,8,9:3', 'strict arguments list');
                assert(strictThis.apply(7, [1]) === 7, 'apply strict this');
            }
            'ok'
        "#
        ),
        true,
    );
}

#[test]
fn forwarded_targets_keep_their_frames_and_bindings() {
    check(
        &format!(
            "{ASSERT}{}",
            r#"
            function thrower() { throw new Error('boom'); }
            var caught = 0;
            for (var k = 0; k < 20; k++) {
                try { thrower.call(null); } catch (e) { caught += e.message === 'boom'; }
                try { thrower.apply(null, [1]); } catch (e) { caught += e.message === 'boom'; }
            }
            assert(caught === 40, 'throws');
            function nt() { return new.target === undefined; }
            function rest(a, ...more) { return a + ':' + more.length; }
            function depth(n) { return n === 0 ? 0 : 1 + depth.call(null, n - 1); }
            function counter() { var c = 0; return function () { return ++c + (this && this.base || 0); }; }
            var inc = counter();
            assert(inc.call({ base: 10 }) === 11 && inc.apply({ base: 20 }) === 22 && inc.call() === 3, 'closure');
            for (var k = 0; k < 20; k++) {
                assert(nt.call({}) && nt.apply({}, []), 'new.target');
                assert(rest.call(null, 1, 2, 3) === '1:2' && rest.apply(null, [1, 2, 3, 4]) === '1:3', 'rest');
                assert(depth(50) === 50, 'recursion');
            }
            function* gen(a) { yield a; yield this.tag; }
            var it = gen.call({ tag: 't' }, 1);
            assert(it.next().value === 1 && it.next().value === 't', 'generator');
            async function af() { return 1; }
            assert(af.call(null) instanceof Promise && af.apply(null, []) instanceof Promise, 'async');
            var arrow = () => typeof this;
            assert(arrow.call(5) === typeof this && arrow.apply(5, []) === typeof this, 'arrow');
            var stackLine = (function named() { try { thrower.call(null); } catch (e) { return String(e.stack); } })();
            assert(stackLine.indexOf('thrower') >= 0, 'stack names the target');
            'ok'
        "#
        ),
        true,
    );
}

#[test]
fn other_targets_and_lists_keep_the_builtins() {
    check(
        &format!(
            "{ASSERT}{}",
            r#"
            var threw = 0;
            try { Function.prototype.call.call(1); } catch (e) { threw += e instanceof TypeError; }
            try { Function.prototype.apply.call({}, null, []); } catch (e) { threw += e instanceof TypeError; }
            assert(threw === 2, 'non-callable');
            class C { constructor() { this.x = 1; } }
            for (var k = 0; k < 3; k++) {
                try { C.call({}); assert(false, 'class call'); } catch (e) { assert(e instanceof TypeError, 'class'); }
                try { C.apply({}, []); assert(false, 'class apply'); } catch (e) { assert(e instanceof TypeError, 'class'); }
            }
            function who() { return this.tag + arguments.length; }
            var bound = who.bind({ tag: 'b' }, 1);
            assert(bound.call({ tag: 'x' }, 2) === 'b2' && bound.apply({ tag: 'x' }, [2, 3]) === 'b3', 'bound');
            assert(Math.max.call(null, 1, 3, 2) === 3 && Math.max.apply(null, [1, 5, 2]) === 5, 'natives');
            var arr = [];
            Array.prototype.push.apply(arr, [1, 2]);
            Array.prototype.push.call(arr, 3);
            assert(arr.join() === '1,2,3', 'native receivers');
            var log = [];
            var p = new Proxy(who, { apply(t, thisArg, args) { log.push(args.length); return Reflect.apply(t, thisArg, args); } });
            assert(p.call({ tag: 'p' }, 1) === 'p1' && p.apply({ tag: 'q' }, [1, 2]) === 'q2' && log.join() === '1,2', 'proxy');
            function args3(a, b, c) { return [a, b, c, arguments.length].join(); }
            for (var k = 0; k < 20; k++) {
                assert(args3.apply(null, [1, , 3]) === '1,,3,3', 'hole');
                assert(args3.apply(null, { length: 2, 0: 'x', 1: 'y' }) === 'x,y,,2', 'array-like');
            }
            Array.prototype[1] = 'inherited';
            assert(args3.apply(null, [1, , 3]) === '1,inherited,3,3', 'inherited hole');
            delete Array.prototype[1];
            var getterList = [1, 2, 3], gets = 0;
            Object.defineProperty(getterList, 1, { get() { gets++; return 'g'; } });
            assert(args3.apply(null, getterList) === '1,g,3,3' && gets === 1, 'getter');
            function mapped(a, b) { a = 'changed'; return args3.apply(null, arguments); }
            assert(mapped(1, 2) === 'changed,2,,2', 'mapped arguments');
            function lengthChanged() { 'use strict'; arguments.length = 1; return args3.apply(null, arguments); }
            assert(lengthChanged(1, 2) === '1,,,1', 'arguments length');
            var long = [];
            for (var k = 0; k < 40; k++) long.push(k);
            function count() { return arguments.length + ':' + arguments[39]; }
            assert(count.apply(null, long) === '40:39', 'long list');
            var lengthGets = 0;
            var lengthy = { get length() { lengthGets++; return 1; }, 0: 'z' };
            assert(args3.apply(null, lengthy) === 'z,,,1' && lengthGets === 1, 'length getter');
            try { args3.apply(null, 1); assert(false, 'primitive list'); } catch (e) { assert(e instanceof TypeError, 'list'); }
            'ok'
        "#
        ),
        false,
    );
}

/// `call_native_entry`'s class match (`native_entry_class`) and the native-entry hit's light
/// check (`native_light_operands`) agree on every class but call's and apply's, which only the
/// former forwards, for receivers and arguments of every kind the light classes test.
#[test]
fn native_entry_classes_agree_with_the_light_check() {
    use crate::bytecode::{
        native_entry_class, native_light_operands, NativeEntryClass, INTRINSIC_FUNCTION_APPLY,
        INTRINSIC_FUNCTION_CALL,
    };
    use crate::value::Value;
    let engine = Engine::new();
    let object = engine.interp.global_this();
    let string = Value::str("s");
    let number = Value::Num(1.0);
    let values = [Value::Undefined, number, string, object];
    let mut lists: Vec<Vec<Value>> = vec![vec![]];
    for first in &values {
        lists.push(vec![*first]);
        for second in &values {
            lists.push(vec![*first, *second]);
        }
    }
    for intrinsic in 0..=u8::MAX {
        for this in &values {
            for args in &lists {
                // SAFETY: a receiver and an argument window of initialized values.
                let (class, light) = unsafe {
                    (
                        native_entry_class(intrinsic, this, args.as_ptr(), args.len()),
                        native_light_operands(intrinsic, this, args.as_ptr(), args.len()),
                    )
                };
                if matches!(
                    intrinsic,
                    INTRINSIC_FUNCTION_CALL | INTRINSIC_FUNCTION_APPLY
                ) {
                    assert_eq!(class, NativeEntryClass::Forward, "class {intrinsic}");
                    assert!(!light, "class {intrinsic}");
                } else {
                    let expected = if light {
                        NativeEntryClass::Light
                    } else {
                        NativeEntryClass::Boundary
                    };
                    assert_eq!(class, expected, "class {intrinsic}");
                }
            }
        }
    }
}
