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
