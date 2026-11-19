//! The template's cached absence across Array levels (P3 stage 3, the megamorphic line's piece
//! 2): a named read of a name no array holds (Vue's `__v_isRef` on arrays) is served by the way
//! probe's absent landing, which admits an Array level where the site's name passed the
//! `arr_ok` gate and compares every level's shape as for any other level. Each change to the
//! chain (an addition on Array.prototype, on Object.prototype, on the array, an accessor, a
//! deletion, a subclass level) is seen at once, and every tier agrees.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

const PROGRAM: &str = r#"
function rd(a) { return a.__v_isRef; }
class Sub extends Array {}
var sub = new Sub(); sub.push(1);
var arrs = [[1, 2], [3], [], [4, 5, 6]];
var calls = 0;
function run(xs) { var r = ""; for (var i = 0; i < 40; i++) r += rd(xs[i % xs.length]) + ","; return r; }
"#;

/// Each step's source, run after the one before.
const STEPS: [&str; 9] = [
    "run(arrs)",
    "Array.prototype.__v_isRef = 1; run(arrs)",
    "delete Array.prototype.__v_isRef; run(arrs)",
    "Object.prototype.__v_isRef = 2; run(arrs)",
    "delete Object.prototype.__v_isRef; arrs[1].__v_isRef = 3; run(arrs)",
    "delete arrs[1].__v_isRef; Object.defineProperty(Array.prototype, '__v_isRef', \
     { configurable: true, get: function () { calls++; return 4; } }); run(arrs) + calls",
    "delete Array.prototype.__v_isRef; run(arrs)",
    "run([sub])",
    "Sub.prototype.__v_isRef = 6; run([sub])",
];

#[test]
fn the_template_serves_an_array_s_cached_absence_and_sees_every_change() {
    // Counts the template JIT's absence hits: with the tier on, `run`'s loop would reach its
    // checkpoints and the tier's code would make the reads.
    crate::opt::pin_tier_off_for_test();
    let mut expected: Vec<String> = Vec::new();
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        eval(&mut engine, PROGRAM);
        let mut values = Vec::new();
        for (k, step) in STEPS.into_iter().enumerate() {
            let before = super::regions::property_paths();
            values.push(eval(&mut engine, step));
            let after = super::regions::property_paths();
            let (absent, checked) = (after[1] - before[1], after[2] - before[2]);
            if tier == Tier::Jit && (k == 0 || k == 2 || k == 6 || k == 7) {
                // A step whose reads are all absent: the first read of each new shape fills
                // the way and the rest are native.
                assert!(
                    absent >= 36 && checked <= 4,
                    "step {k} `{step}`: {absent} native absences, {checked} checked reads"
                );
            }
        }
        if expected.is_empty() {
            expected = values;
        } else {
            assert_eq!(values, expected, "{tier:?}");
        }
    }
    let undefined = "undefined,".repeat(40);
    assert_eq!(expected[0], undefined);
    assert_eq!(expected[1], "1,".repeat(40));
    assert_eq!(expected[2], undefined);
    assert_eq!(expected[3], "2,".repeat(40));
    assert_eq!(expected[4], "undefined,3,undefined,undefined,".repeat(10));
    assert_eq!(expected[5], format!("{}40", "4,".repeat(40)));
    assert_eq!(expected[6], undefined);
    assert_eq!(expected[7], undefined);
    assert_eq!(expected[8], "6,".repeat(40));
}
