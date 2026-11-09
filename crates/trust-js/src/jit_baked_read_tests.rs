//! Named reads whose native code baked one cached state: a warm engine runs a loop-free function
//! on the bytecode VM first, so the read's cache held a single shape when its native code was
//! made, as a second stage's seeded caches do. Receivers of other shapes come later. The baked
//! probe's miss searches the site's live ways through the shared read stub before the checked
//! read (JSC's Baseline runs its data IC's handler over every case the cache holds; SpiderMonkey's
//! Baseline stubs fail over to the next in their chain): once the checked read has cached a new
//! shape, the stub serves it natively. Each form (a local, an operand, `this`, a method read, a
//! primitive's wrapper prototype) reads 30 receivers of a second shape and then 30 of a third:
//! one checked read for each, then native hits. Every tier agrees on the values.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

/// The bytecode VM runs a warm engine's loop-free functions make before native code.
const WARM_RUNS: u32 = 20;

/// A warm engine at `tier`: functions run on the bytecode VM a while before native code, so their
/// reads' caches hold what those runs saw when the code is made.
fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(1);
    engine.interp.jit_admit_runs = WARM_RUNS;
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// Native named-read hits and checked named reads (`regions::property_paths` 0 and 2) while
/// `source` runs.
fn named_reads(engine: &mut Engine, source: &str) -> (u64, u64, String) {
    let before = super::regions::property_paths();
    let value = eval(engine, source);
    let after = super::regions::property_paths();
    (after[0] - before[0], after[2] - before[2], value)
}

/// Three receiver classes of different shapes (`x` at slots 0, 1 and 2), sharing `getX` and
/// `m`; the read forms, each warmed on the first class alone. `run` makes no named read of its
/// own.
const PROGRAM: &str = r#"
function getX() { return this.x; }
function m() { return this.x + 1; }
function A(x) { this.x = x; }
function B(x) { this.y = 0; this.x = x; }
function C(x) { this.z = 0; this.w = 0; this.x = x; }
A.prototype.getX = B.prototype.getX = C.prototype.getX = getX;
A.prototype.m = B.prototype.m = C.prototype.m = m;
function local(o) { return o.x; }
function operand(o) { return [o][0].x; }
function self(o) { return o.getX(); }
function method(o) { return o.m(); }
function wrapper(v) { return v.toString(); }
var as = [], bs = [], cs = [];
for (var i = 0; i < 30; i++) { as.push(new A(i)); bs.push(new B(i + 100)); cs.push(new C(i + 200)); }
var prims = { a: [], b: [], c: [] };
for (var i = 0; i < 30; i++) { prims.a.push("s" + i); prims.b.push(i + 0.5); prims.c.push(i % 2 === 0); }
function run(f, xs) { var r = ""; for (var i = 0; i < 30; i++) r += f(xs[i]) + ","; return r; }
var warm = run(local, as) + run(operand, as) + run(self, as) + run(method, as) + run(wrapper, prims.a);
warm.length;
"#;

#[test]
fn a_baked_read_s_miss_probes_the_live_ways_before_the_checked_read() {
    // Counts the template JIT's baked probes and checked reads: with the tier on, `run`'s loop
    // reaches its checkpoints within a few calls (P3 stage 2, step 8), and the tier's code
    // splices the read forms.
    crate::opt::pin_tier_off_for_test();
    let forms = ["local", "operand", "self", "method"];
    let mut expected = Vec::new();
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        eval(&mut engine, PROGRAM);
        let mut values = Vec::new();
        for form in forms {
            for (receivers, label) in [("bs", "second"), ("cs", "third")] {
                let (hits, checked, value) =
                    named_reads(&mut engine, &format!("run({form}, {receivers})"));
                values.push(value);
                if tier == Tier::Jit && cfg!(target_arch = "aarch64") {
                    // `self` and `method` also read `getX`/`m` (baked on A's prototype, a miss
                    // for B's and C's receivers too) and then `this.x` in the callee.
                    let reads = if form == "self" || form == "method" {
                        2
                    } else {
                        1
                    };
                    assert_eq!(
                        checked, reads,
                        "{form}, {label} shape: one checked read a site, the rest native"
                    );
                    assert_eq!(
                        hits,
                        30 * reads - reads,
                        "{form}, {label} shape: native hits"
                    );
                }
            }
        }
        for (receivers, label) in [("prims.b", "Number"), ("prims.c", "Boolean")] {
            let (hits, checked, value) =
                named_reads(&mut engine, &format!("run(wrapper, {receivers})"));
            values.push(value);
            if tier == Tier::Jit && cfg!(target_arch = "aarch64") {
                assert_eq!(checked, 1, "wrapper, {label}: one checked read");
                assert_eq!(hits, 29, "wrapper, {label}: native hits");
            }
        }
        if expected.is_empty() {
            expected = values;
        } else {
            assert_eq!(values, expected, "{tier:?}");
        }
    }
}
