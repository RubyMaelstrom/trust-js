//! ECMA-262 editor's draft e28783d5, spec.html: String concatenation 1251,
//! ApplyStringOrNumericBinaryOperator 21341, String.prototype.concat 36291.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("concat fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        assert!(engine.interp.fn_frames.is_empty());
        engine.interp.gc_collect();
        assert!(matches!(engine.eval("1+2", false), Ok(Completion::Value(v)) if v == "3"));
    }
}

#[test]
fn string_concat_aliases_temporaries_and_self_append() {
    check(
        r#"
        function run() {
            var s = 'prefix', keep = s, out = [], o = {s: s};
            for (var i = 0; i < 60; i++) {
                if (i === 20 || i === 40) out.push(o.s);
                o.s += 'x';
                s = (s + 'y') + 'z';
            }
            var t = o.s; o.s += o.s;
            return [keep, out[0].length, out[1].length, s.length,
                    t.length, o.s === t + t, ''.concat(t) === t,
                    t.concat('') === t].join('|');
        }
        run()
        "#,
        "prefix|26|46|126|66|true|true|true",
    );
}

#[test]
fn ascii_unit_observation_does_not_pin_unique_accumulators() {
    use crate::lstr::LStr;
    let mut engine = Engine::new();
    let mut s = LStr::from("a".repeat(128)).concat_owned(&LStr::from("b"));
    let pointer = s.buffer_ptr();
    assert_eq!(engine.interp.str_len(&s), 129);
    assert_eq!(engine.interp.unit_at(&s, 128), Some(b'b' as u16));
    assert_eq!(engine.interp.unit_at(&s, 129), None);
    s = s.concat_owned(&LStr::from("c"));
    assert_eq!(
        s.buffer_ptr(),
        pointer,
        "unit observation must not force a prefix copy"
    );
    assert_eq!(engine.interp.str_len(&s), 130);

    // A deliberately materialized unit vector still owns its immutable old view. Bypassing
    // that cache for later ASCII reads must neither mutate the view nor return a stale length.
    let units = engine.interp.units_full(&s);
    let mut next = s.concat_owned(&LStr::from("d"));
    assert_eq!(units.len(), 130);
    assert_eq!(engine.interp.str_len(&next), 131);
    next = next.concat_owned(&LStr::from("𝌆"));
    assert_eq!(engine.interp.str_len(&next), 133);
    assert_eq!(engine.interp.unit_at(&next, 131), Some(0xD834));
    assert_eq!(engine.interp.unit_at(&next, 132), Some(0xDF06));
}

#[test]
fn observed_string_accumulators_keep_aliases_and_utf16_units() {
    check(
        r#"
        function build() {
            var box = {s:'a'.repeat(128)}, aliases = [], sum = 0;
            for (var i=0; i<200; i++) {
                sum += box.s.charCodeAt(0);
                if (i % 50 === 0) aliases.push(box.s);
                if (box.s.charAt(0) !== 'a' || box.s.indexOf('a') !== 0) throw Error('view');
                box.s += i === 100 ? '\uD834\uDF06' : 'b';
            }
            return [box.s.length, sum, aliases.map(x=>x.length).join(','),
                box.s.charCodeAt(228), box.s.charCodeAt(229)].join('|');
        }
        build()
    "#,
        "329|19400|128,178,228,279|55348|57094",
    );
}

#[test]
fn string_concat_preserves_code_units_and_join_canonicalization() {
    check(
        r#"
        function append(o, s) { o.s += s; return o.s; }
        function add(a, b) { return a + b; }
        function run() {
            var out = [];
            for (var i = 0; i < 30; i++) {
                var o = {s: '\uD834'}, alias = o.s;
                var joined = append(o, '\uDF06');
                if (joined !== '\u{1D306}' || joined.length !== 2 ||
                    alias.length !== 1 || alias.charCodeAt(0) !== 0xD834)
                    throw new Error('property join');
            }
            out.push(add('\uD834', '\uDF06') === '\u{1D306}');
            out.push('\uD834'.concat('\uDF06') === '\u{1D306}');
            out.push('x'.concat('\uD834', '\uDF06', 'y') === 'x\u{1D306}y');
            out.push(add('\uDBFE', '\uDC00') === '\u{10F800}');
            out.push(append({s: '\uDBFE'}, '\uDC00') === '\u{10F800}');
            out.push(add('abc', '\u00E9').charCodeAt(3) === 233);
            out.push(add('\uD834', 'x').charCodeAt(0) === 0xD834);
            out.push(add('', '\uDC00').charCodeAt(0) === 0xDC00);
            return out.join('|');
        }
        run()
        "#,
        "true|true|true|true|true|true|true|true",
    );
}

#[test]
fn string_concat_conversion_order_and_abrupt_completion() {
    check(
        r#"
        function run() {
            var log = [], out = [];
            function operand(name, result) {
                return {[Symbol.toPrimitive](hint) { log.push(name + ':' + hint); return result; }};
            }
            out.push(operand('left', 'a') + operand('right', 7));
            out.push(log.join(',')); log.length = 0;
            out.push(String.prototype.concat.call(operand('this', 's'),
                operand('one', 1), operand('two', 2)));
            out.push(log.join(',')); log.length = 0;
            try { operand('left', Symbol()) + operand('right', 's'); }
            catch (e) { out.push(e.name); }
            out.push(log.join(',')); log.length = 0;
            var bad = {[Symbol.toPrimitive]() { log.push('bad'); throw 123; }};
            try { bad + operand('unreached', 's'); } catch (e) { out.push(e); }
            try { 's'.concat(bad, operand('unreached', 's')); } catch (e) { out.push(e); }
            out.push(log.join(','));
            out.push('b' + 2n, 2n + 'b', 'n' + -0, 'u' + undefined, 'o' + null);
            return out.join('|');
        }
        run()
        "#,
        "a7|left:default,right:default|s12|this:string,one:string,two:string|TypeError|left:default,right:default|123|123|bad,bad|b2|2b|n0|uundefined|onull",
    );
}

#[test]
fn string_concat_compound_read_precedes_rhs_and_respects_descriptors() {
    check(
        r#"
        function run() {
            var o = {s: 'before'}, log = [], out = [];
            o.s += (o.s = 'replacement', 'after');
            out.push(o.s);
            Object.defineProperty(o, 's', {
                get() { log.push('get'); return 'G'; },
                set(v) { log.push('set:' + v); }, configurable: true
            });
            o.s += (log.push('rhs'), 'R');
            out.push(log.join(','));
            Object.defineProperty(o, 's', {value: 'fixed', writable: false});
            o.s += 'lost'; out.push(o.s);
            function strict() { 'use strict'; o.s += 'throws'; }
            try { strict(); } catch (e) { out.push(e.name); }
            return out.join('|');
        }
        run()
        "#,
        "beforeafter|get,rhs,set:GR|fixed|TypeError",
    );
}

#[test]
fn string_concat_helpers_reach_compiled_tiers() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        assert!(matches!(engine.eval(r#"
            function add(a, b) { return a + b; }
            function append(o, b) { o.s += b; return o.s; }
            function appendLocal(a, b) { a += b; return a; }
            function appendCaptured(a, b) { function read() { return a; } a += b; return read(); }
            var ok = true;
            for (var i = 0; i < 30; i++) {
                ok = ok && add('\uD834', '\uDF06') === '\u{1D306}' &&
                    append({s: '\uD834'}, '\uDF06') === '\u{1D306}';
                ok = ok && appendLocal('\uD834', '\uDF06') === '\u{1D306}';
                ok = ok && appendCaptured('\uD834', '\uDF06') === '\u{1D306}';
            }
            ok
        "#, false), Ok(Completion::Value(v)) if v == "true"));
        for name in ["add", "append", "appendLocal", "appendCaptured"] {
            let env = engine.interp.global_env.clone();
            let crate::value::Value::Obj(object) = engine
                .interp
                .get_var(name, &env)
                .unwrap_or_else(|_| panic!("missing {name}"))
            else {
                panic!("{name} is not an object");
            };
            let object = object.borrow();
            let crate::value::Callable::User(user) = &object.call else {
                panic!("{name} is not a user function");
            };
            let chunk = user
                .func
                .code
                .get()
                .and_then(Option::as_ref)
                .expect("compiled body");
            if name == "appendLocal" {
                assert!(
                    chunk.test_ops().windows(2).any(|ops| matches!(
                        ops,
                        [crate::bytecode::Op::Add, crate::bytecode::Op::StoreLocal(_)]
                    )),
                    "local string store pattern must actually execute"
                );
            }
            if name == "appendCaptured" {
                assert!(
                    chunk.test_ops().windows(2).any(|ops| matches!(
                        ops,
                        [crate::bytecode::Op::Add, crate::bytecode::Op::StoreCap(_)]
                    )),
                    "captured string store pattern must actually execute"
                );
            }
            if tier == Tier::Jit && cfg!(any(target_arch = "aarch64", target_arch = "x86_64")) {
                assert!(
                    chunk.jit.get().is_some_and(|code| code.is_some()),
                    "{name} stayed in bytecode"
                );
            }
        }
    }
}

#[test]
fn captured_string_append_inside_nested_scopes() {
    // A body's captured `s` stays in its activation record while a loop, block, catch, `with`
    // or class scope is the running LexicalEnvironment (ECMA-262 ForBodyEvaluation and
    // CreatePerIterationEnvironment, BlockDeclarationInstantiation, CatchClauseEvaluation,
    // the with statement, ClassDefinitionEvaluation). The compiled `+=` into it must find the
    // binding there; generator bodies resume on a separate continuation.
    check(
        r#"
        function scopes() {
            var s = '', read = function () { return s; };
            for (let i = 0; i < 40; i++) {
                var seen = function () { return i; };
                s += 'a';
            }
            { let j = 1; var block = function () { return j; }; s = s + 'b'; }
            try { throw 'e'; } catch (e) { var caught = function () { return e; }; s += 'c'; }
            with ({ unrelated: 1 }) { s += 'd'; }
            class A extends (s += 'e', Object) { [(s += 'f', 'k')]() { return A; } }
            return read().length + ':' + read().slice(-6) + ':' + seen() + block() + caught();
        }
        function* generator() {
            var s = '', read = () => s;
            for (let i = 0; i < 3; i++) { var q = () => i; s += 'g'; yield s; }
            return read();
        }
        var out = [];
        for (var n = 0; n < 3; n++) out.push(scopes(), [...generator()].join(','));
        out.join('|')
    "#,
        "45:abcdef:391e|g,gg,ggg|45:abcdef:391e|g,gg,ggg|45:abcdef:391e|g,gg,ggg",
    );
}

#[test]
fn local_string_store_preserves_rhs_changes_aliases_and_exceptions() {
    check(r#"
        function build() {
            var s = 'a'.repeat(128), original = s, aliases = [], out = [];
            for (var k=0; k<400; k++) {
                if (k % 100 === 0) aliases.push(s);
                s += 'x';
            }
            out.push(original.length, s.length, aliases.map(x=>x.length).join(','));
            s += s; out.push(s.length);
            s = 'before'; s += (s = 'replacement', 'after'); out.push(s);
            s = 'keep';
            try { s += Symbol('fail'); } catch(e) { out.push(e.name, s); }
            try { s += (() => { throw Error('rhs'); })(); } catch(e) { out.push(s); }
            s += { [Symbol.toPrimitive]() { out.push(s); s = 'changed'; return 'R'; } };
            out.push(s);
            s = 'left'; s = s + (s = 'right', 'suffix'); out.push(s);
            const fixed = 'const';
            try { fixed += 'x'; } catch(e) { out.push(e.name, fixed); }
            return out.join('|');
        }
        build()
    "#, "128|528|128,228,328,428|1056|beforeafter|TypeError|keep|keep|keep|keepR|leftsuffix|TypeError|const");
    // This separate body has no closure capture of the local under test.
    check(
        r#"
        function local(s) {
            var alias = s, observed;
            s += (observed = s, 'x');
            var first = s;
            s += s;
            s = s + '\uD834'; s += '\uDF06';
            return [alias, observed, first, s].join('|');
        }
        local('a')
    "#,
        "a|a|ax|axax𝌆",
    );
}

#[test]
fn string_trim_ascii_set_and_unicode_boundaries() {
    check(
        r#"
        function run() {
            for (var n = 0; n < 128; n++) {
                var c = String.fromCharCode(n), s = c + 'x' + c;
                var ws = (n >= 9 && n <= 13) || n === 32;
                if (s.trim() !== (ws ? 'x' : s) ||
                    s.trimStart() !== (ws ? 'x' + c : s) ||
                    s.trimEnd() !== (ws ? c + 'x' : s)) throw new Error('ASCII ' + n);
            }
            return [('\v\t\r\n\f x\v\t ').trim(), '\v'.trim().length,
                    '\u00A0\uFEFFx\u2028\u2029'.trim(),
                    '\u0085x\u0085'.trim().length,
                    '\v\uD800\v'.trim().charCodeAt(0),
                    String.prototype.trimLeft === String.prototype.trimStart,
                    String.prototype.trimRight === String.prototype.trimEnd].join('|');
        }
        run()
        "#,
        "x|0|x|3|55296|true|true",
    );
}
