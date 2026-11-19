//! A typed array's intrinsic getters answered in line at a read's validated accessor landing
//! (P3 stage 3, the getter line, piece b; `typed_array::emit_named_getter`): `length`,
//! `byteLength` and `byteOffset` of fixed views, of a length-tracking view and of views of a
//! resizable buffer that shrinks under them, then of a view whose buffer is detached, read
//! through every receiver form (a local, the frame's `this`, an operand), from code a cold engine
//! makes at a function's first call and from code a warm one makes once its reads' caches hold
//! their ways (baked). The value is Node's; on ARM64 the reads are answered in generated code
//! (`regions::property_paths` 0) and none takes the checked helper (2).

use crate::bytecode::Tier;
use crate::Engine;

use super::traced_tests::eval;

const READS: &str = r#"
function r(a) { return a.length + a.byteLength * 100 + a.byteOffset * 10000; }
function rThis() { return this.length; }
function rLocal(a) { var t = 0; for (var i = 0; i < 4; i++) t += a.byteLength; return t; }
function rOperand(a) { return [a][0].length; }
var rab = new ArrayBuffer(32, { maxByteLength: 64 });
var vs = [new Uint8Array(10), new Float64Array(new ArrayBuffer(64), 8, 3), new Int16Array(rab, 4, 4), new Int16Array(rab, 6)];
var s = 0, t = 0;
for (var k = 0; k < 200; k++) for (var j = 0; j < 4; j++) s += r(vs[j]) + rThis.call(vs[j]) + rLocal(vs[j]) + rOperand(vs[j]);
rab.resize(9);
for (var k = 0; k < 50; k++) for (var j = 0; j < 4; j++) t += r(vs[j]);
rab.resize(2);
for (var k = 0; k < 50; k++) for (var j = 0; j < 4; j++) t += r(vs[j]);
var moved = vs[1].buffer.transfer();
for (var k = 0; k < 50; k++) t += r(vs[1]) + r(vs[0]);
s + ':' + t + ':' + moved.byteLength
"#;

#[test]
fn typed_array_getters_answer_in_line() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        for threshold in [0, 1] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(threshold);
            if threshold == 1 {
                engine.interp.jit_admit_runs = 20;
            }
            #[cfg(target_arch = "aarch64")]
            let before = super::regions::property_paths();
            assert_eq!(
                eval(&mut engine, READS),
                "37432400:11401850:64",
                "{tier:?} {threshold}"
            );
            #[cfg(target_arch = "aarch64")]
            if tier == Tier::Jit {
                let after = super::regions::property_paths();
                let (hits, checked) = (after[0] - before[0], after[2] - before[2]);
                // 8,700 getter reads (and `[a][0]`'s array element); the warm engine's first
                // calls run on the VM.
                assert!(
                    hits >= 8_000,
                    "{threshold}: {hits} in line, {checked} checked"
                );
                assert!(
                    checked <= 400,
                    "{threshold}: {hits} in line, {checked} checked"
                );
            }
        }
    }
}

/// The in-line answer at a full site's landing is one process-wide stub
/// (`SharedStub::TypedGetter`): two engines alive on one thread, each with its own Realm's
/// getters and its own arrays, read through it in turn, and every read of each answers in line
/// with that engine's values. A stub that reached an engine's state through an address it was
/// made with would serve the engine that made it alone. Both of the template's forms: a cold
/// engine's first-stage code (the shared read stub's landing) and a warm engine's recompiled
/// code (full sites, whose landing calls the stub).
#[test]
fn two_engines_on_one_thread_answer_through_the_one_stub() {
    crate::opt::pin_tier_off_for_test();
    for threshold in [0, 1] {
        let make = |len: u32| {
            let mut engine = Engine::new();
            engine.set_tier(Tier::Jit);
            engine.set_tier_threshold(threshold);
            if threshold == 1 {
                engine.interp.jit_admit_runs = 20;
            }
            let program = format!(
                "function r(a) {{ return a.length + a.byteLength * 100 + a.byteOffset * 10000; }}
                 function run(a, n) {{ var s = 0; for (var i = 0; i < n; i++) s += r(a); return s; }}
                 var a = new Int16Array(new ArrayBuffer(64), 8, {len});
                 var w = 0; for (var k = 0; k < 20; k++) w = run(a, 50); w + ''"
            );
            assert_eq!(
                eval(&mut engine, &program),
                (50 * (201 * len + 80_000)).to_string()
            );
            engine
        };
        let mut first = make(3);
        let mut second = make(5);
        for round in 0..2 {
            for (label, engine, len) in [("first", &mut first, 3u32), ("second", &mut second, 5)] {
                #[cfg(target_arch = "aarch64")]
                let before = super::regions::property_paths();
                assert_eq!(
                    eval(engine, "run(a, 40) + ''"),
                    (40 * (201 * len + 80_000)).to_string(),
                    "{threshold}: round {round}, the {label} engine"
                );
                #[cfg(target_arch = "aarch64")]
                {
                    let after = super::regions::property_paths();
                    assert_eq!(
                        (after[0] - before[0], after[2] - before[2]),
                        (120, 0),
                        "{threshold}: round {round}, the {label} engine: reads in line, checked"
                    );
                }
            }
        }
    }
}
