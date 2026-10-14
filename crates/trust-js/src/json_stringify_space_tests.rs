//! JSON.stringify's indentation limit follows UTF-16 code units (ECMA-262
//! snapshot e28783d5, sec-json.stringify; local spec.html:48201-48202).
//! Expected results verified with Node.js.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine
            .eval(source, false)
            .expect("JSON indentation fixture parses")
        {
            Completion::Value(actual) => assert_eq!(actual, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
        engine.interp.gc_collect();
    }
}

#[test]
fn stringify_space_truncates_astral_and_lone_surrogate_code_units() {
    check(
        r#"
        function run() {
            var cases = [
                ['abcdefghijk', 'abcdefghij'],
                ['😀'.repeat(10), '😀'.repeat(5)],
                ['123456789😀', '123456789\ud83d'],
                ['123456789\ud800x', '123456789\ud800'],
                ['123456789\udfffz', '123456789\udfff'],
                [String.fromCodePoint(0x10ffff).repeat(6), String.fromCodePoint(0x10ffff).repeat(5)],
                ['é123456789😀', 'é123456789'],
                ['\ude00' + 'x'.repeat(8) + '😀', '\ude00' + 'x'.repeat(8) + '\ud83d'],
                ['😀', '😀'],
                ['', '']
            ];
            for (var n = 0; n < cases.length; n++) {
                var input = cases[n][0], gap = cases[n][1];
                var expected = gap === '' ? '{"a":[1]}' :
                    '{\n' + gap + '"a": [\n' + gap + gap + '1\n' + gap + ']\n}';
                var actual = JSON.stringify({a: [1]}, null, input);
                if (actual !== expected) return n + ':' + actual.length + ':' + expected.length;
            }
            return cases.length;
        }
        run()
        "#,
        "10",
    );
}

#[test]
fn stringify_space_wrapper_conversion_precedes_serialization() {
    check(
        r#"
        function run() {
            var events = [], space = new String('unused');
            space.toString = function() { events.push('space'); return '123456789😀'; };
            var value = {get a() { events.push('get'); return 1; }};
            var actual = JSON.stringify(value, function(key, item) {
                events.push('replace:' + key); return item;
            }, space);
            var expected = '{\n123456789\ud83d"a": 1\n}';
            var first = actual === expected;
            space.toString = function() { events.push('throw'); throw 'stop'; };
            try { JSON.stringify(value, null, space); } catch (error) { events.push(error); }
            return first + '|' + events.join(',');
        }
        run()
        "#,
        "true|space,replace:,get,replace:a,throw,stop",
    );
}
