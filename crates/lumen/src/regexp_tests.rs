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
