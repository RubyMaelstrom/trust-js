//! RegExp matching and the RegExp-based String methods on every tier. Expected values are
//! node's (V8) results for the same scripts.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn check(source: &str, expected: &str) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("regexp fixture parses") {
            Completion::Value(value) => assert_eq!(value, expected, "{tier:?}"),
            Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
        }
    }
}

/// ECMA-262 bounds neither the time nor the depth of a match (§22.2.2): long subjects that
/// keep a choice point per iteration match like any other.
#[test]
fn long_subjects_match_without_a_depth_limit() {
    check(
        r#"
        function run() {
            var quoted = '"' + 'ab\\"c'.repeat(300) + '"';
            var string = /"(?:[^"\\]|\\.)*"/.exec(quoted)[0].length;
            var star = /(a|b)*/.exec('ab'.repeat(500));
            var lazy = /(a|ab)*?c/.exec('ab'.repeat(5000) + 'c');
            return [string, star[0].length, star[1], lazy[0].length, lazy[1]].join();
        }
        run()
        "#,
        "1502,1000,b,10001,ab",
    );
}

/// Backtracking out of an alternative undoes the captures it made, including those of a
/// lookaround inside it (§22.2.2.4) and those cleared by a later quantifier iteration
/// (RepeatMatcher step 4).
#[test]
fn backtracking_restores_lookaround_and_iteration_captures() {
    check(
        r#"
        JSON.stringify([
            /(?:(?=(a))b|a)/.exec("a"),
            /(?:(?<=(a))c|a$)/.exec("aa"),
            /(?:(?=(\w+))\1x|\w)(.)/.exec("ab"),
            /^(?:(a)|b)*$/.exec("ab"),
        ])
        "#,
        r#"[["a",null],["a",null],["ab",null,"b"],["ab",null]]"#,
    );
}

/// Counted repetitions too large to unroll run as loops over one copy of their body, keeping
/// RepeatMatcher's semantics: per-iteration capture resets, the empty-iteration rule beyond
/// `min`, and greedy versus lazy order (§22.2.2.3.1).
#[test]
fn large_counted_repetitions_match_like_unrolled_ones() {
    check(
        r#"
        function show(match) {
            return match ? match[0].length + ':' + match.index + ':' + match.slice(1).join('/') : 'null';
        }
        [
            show(/(?:(?:ab){1000}){1000}/.exec('ab'.repeat(1000))),
            show(/(?:(?:ab){10}){100}/.exec('x' + 'ab'.repeat(1000))),
            show(/(?:(a)|(b)){1500}/.exec('ab'.repeat(800))),
            show(/^(?:(a)|(b)){2,1500}?$/.exec('ab'.repeat(700))),
            show(/(?:a?){1001}b/.exec('aab')),
            show(/(?:(a)|b?){1001,1002}c/.exec('abc')),
            show(/(?:x(?:ab){1001}y){3}/.exec(('x' + 'ab'.repeat(1001) + 'y').repeat(3))),
            show(new RegExp('(?:'.repeat(10) + 'a|b' + '){2}'.repeat(10)).exec('ab'.repeat(600))),
        ].join()
        "#,
        "null,2000:1:,1500:0:/b,1400:0:/b,3:0:,3:0:,6012:0:,1024:0:",
    );
}

/// `split`'s limit is ToUint32 (§7.1.7), which wraps modulo 2^32 for every Number rather than
/// saturating, for string and RegExp separators alike (§22.1.3.23, §22.2.6.14 step 13).
/// RegExp.prototype[@@split] and [@@matchAll] default SpeciesConstructor to the realm's %RegExp%
/// intrinsic and never read the global `RegExp` binding (§22.2.6.14 step 4, §22.2.6.9 step 4).
#[test]
fn split_limits_wrap_and_species_defaults_to_the_intrinsic() {
    check(
        r#"
        function run() {
            var lengths = [
                "a,b,c".split(",", -1e20).length,
                "a,b,c".split(",", 2 ** 64).length,
                "a,b,c".split(",", 2 ** 32 + 1).length,
                "a,b,c".split(",", -1).length,
                "a,b,c".split(",", -4294967294).length,
                "a,b,c".split(/,/, -1e20).length,
                "a,b,c".split(/,/, 2 ** 64).length,
                "a,b,c".split(/,/, 2 ** 32 + 2).length,
                "a,b,c".split(/,/, -4294967294).length,
                "a,b,c".split(/,/, 1e300).length,
            ];
            var intrinsic = RegExp, constructed = [];
            globalThis.RegExp = function () {
                constructed.push(arguments.length);
                return /x/y;
            };
            var splitter = /\d/;
            splitter.constructor = undefined;
            var parts = "a1b2c".split(splitter).join("|");
            var matcher = /\d/g;
            matcher.constructor = undefined;
            var all = Array.from("a1b2c".matchAll(matcher), function (m) { return m[0]; }).join("|");
            var reads = 0;
            Object.defineProperty(globalThis, "RegExp", {
                get: function () { reads++; return intrinsic; },
                configurable: true,
            });
            "x,y".split(/,/);
            Array.from("xy".matchAll(/./g));
            Object.defineProperty(globalThis, "RegExp", { value: intrinsic, writable: true, configurable: true });
            return [lengths.join(" "), parts, all, constructed.length, reads].join(",");
        }
        run()
        "#,
        "3 0 1 3 2 3 0 2 2 0,a|b|c,1|2,0,0",
    );
}

/// A pattern whose alternatives begin with `^` or with an element (`(^|\s)name`) is only
/// attempted where one of them can begin, including the empty end of the input.
#[test]
fn caret_alternatives_only_start_where_caret_holds() {
    check(
        r###"
        (function () {
        function all(re, s) {
            var found = [], m;
            re.lastIndex = 0;
            while ((m = re.exec(s))) {
                found.push(m.index + ':' + m[0].length);
                if (m[0].length === 0) re.lastIndex++;
                if (found.length > 50) break;
            }
            return found.join(' ');
        }
        var cases = [
            [/(^|\s)active(\s|$)/g, 'active btn active-x  active'],
            [/(^|x)/g, ''],
            [/(^|x)/g, 'axbx'],
            [/(?:^|,)(\w*)/g, 'a,,b,'],
            [/(?:^|,)(\w*)/gm, 'a\n,b\nc'],
            [/^|a/gm, 'xa\nya\n'],
            [/(?:^|b)/gm, 'b\n\nb'],
            [/(?:^ +|\s)x/g, '  x x'],
            [/(^\d+|[a-z]+)/g, '12ab 34'],
            [/(^|\W)#(\w+)/g, '#tag and #more #'],
            [/(?:^|[?&])id=(\d+)/g, 'id=1&x=2&id=3'],
            [/(^|\s)a/g, 'a a a'],
            [/(^|\s)a/gu, '😀 a'],
            [/(?:^|\n)b/g, 'b\nb\r\nb'],
            [/(?:^|\n)b/gm, 'b\nb b'],
        ];
        var lines = cases.map(function (c) { return c[0] + ' ' + all(c[0], c[1]); });
        lines.push([/(^|\s)active(\s|$)/.test('btn active'), /(^|\s)active(\s|$)/.test('btnactive'), /(^|x)$/.test(''), /(?:^|b)c/.test('ac')].join());
        return lines.join('\n');
        })()
        "###,
        r###"/(^|\s)active(\s|$)/g 0:7 20:7
/(^|x)/g 0:0
/(^|x)/g 0:0 1:1 3:1
/(?:^|,)(\w*)/g 0:1 1:1 2:2 4:1
/(?:^|,)(\w*)/gm 0:1 2:0 5:1
/^|a/gm 0:0 1:1 3:0 4:1 6:0
/(?:^|b)/gm 0:0 2:0 3:0
/(?:^ +|\s)x/g 0:3 3:2
/(^\d+|[a-z]+)/g 0:2 2:2
/(^|\W)#(\w+)/g 0:4 8:6
/(?:^|[?&])id=(\d+)/g 0:4 8:5
/(^|\s)a/g 0:1 1:2 3:2
/(^|\s)a/gu 2:2
/(?:^|\n)b/g 0:1 1:2 4:2
/(?:^|\n)b/gm 0:1 1:2 4:1
true,false,true,false"###,
    );
}

/// Deeply nested patterns are an implementation limit: past it, a SyntaxError rather than a
/// native stack overflow. Patterns at the limit compile and match (node gives the same results
/// for them).
#[test]
fn deeply_nested_patterns_are_syntax_errors_not_crashes() {
    check(
        r#"
        function attempt(source, flags, subject) {
            try {
                var match = new RegExp(source, flags).exec(subject);
                return match ? match.length + ':' + match.index : 'null';
            } catch (error) {
                return error.name;
            }
        }
        function nest(open, body, close, depth) {
            return open.repeat(depth) + body + close.repeat(depth);
        }
        [
            attempt(nest('(', 'a', ')', 1000), '', 'a'),
            attempt(nest('(?:', 'a', ')*', 1000), '', 'aa'),
            attempt(nest('[', 'a', ']', 1000), 'v', 'a'),
            attempt(nest('(?<=', 'a', ')', 128) + 'b', '', 'ab'),
            attempt(nest('(', 'a', ')', 100000), '', 'a'),
            attempt(nest('(?:', 'a', ')*', 100000), '', 'a'),
            attempt(nest('[', 'a', ']', 100000), 'v', 'a'),
            attempt(nest('(?=', 'a', ')', 100000), '', 'a'),
        ].join()
        "#,
        "1001:0,1:0,1:0,1:1,SyntaxError,SyntaxError,SyntaxError,SyntaxError",
    );
}
