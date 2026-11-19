//! The engine-wide property cache probed by generated code (the megamorphic line's piece 1b): a
//! named read whose site's ways miss probes the cache's entry for the receiver's shape and the
//! site's key in its shared read stub (`emit_mega_way`) and lands as a way does. At a site of
//! eight receiver shapes (four ways) every read is native once the entries are filled: own data,
//! a prototype's data at depths 1 to 3, a cached absence; an accessor entry publishes the getter
//! for the helper, which runs it once a read. Two keys that share a table slot never read each
//! other's property, a prototype changed after the fill is seen, and two engines alive on one
//! thread each probe their own table (the stubs are process-wide). Every tier agrees.

use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}\n{source}"),
    }
}

fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

/// Native named reads (hits and absences) and checked reads (`regions::property_paths` 0, 1
/// and 2) while `source` runs, and its value.
fn reads(engine: &mut Engine, source: &str) -> (u64, u64, String) {
    let before = super::regions::property_paths();
    let value = eval(engine, source);
    let after = super::regions::property_paths();
    (
        after[0] + after[1] - before[0] - before[1],
        after[2] - before[2],
        value,
    )
}

/// Two names whose keys take one slot of the table for every shape, in `engine`'s heap.
fn colliding_names(engine: &mut Engine) -> (String, String) {
    let _ = engine;
    let mix = |name: &str| {
        let n = crate::atom::PropKey::from_text(name).word() as usize >> 3;
        (n ^ (n >> 7)) & (crate::interpreter::STUB_CACHE_SIZE - 1)
    };
    let mut seen = std::collections::HashMap::new();
    for k in 0..20_000 {
        let name = format!("k{k}");
        if let Some(other) = seen.insert(mix(&name), name.clone()) {
            return (other, name);
        }
    }
    panic!("no two of 20,000 names share a slot");
}

/// Eight receivers of eight shapes over one prototype chain P1 -> P2 -> P3; the read sites.
fn program(first: &str, second: &str) -> String {
    format!(
        r#"
var P3 = {{ p3: 3 }};
var P2 = Object.create(P3); P2.p2 = 2;
var P1 = Object.create(P2); P1.p1 = 1;
var calls = 0;
Object.defineProperty(P1, "g", {{ configurable: true, get: function () {{ calls++; return this.id; }} }});
var objs = [];
for (var i = 0; i < 8; i++) {{
  var o = Object.create(P1); o["s" + i] = i; o.id = i; o.x = 10 + i; o.{first} = 100 + i; o.{second} = 200 + i;
  objs.push(o);
}}
function rx(o) {{ return o.x; }}
function r1(o) {{ return o.p1; }}
function r2(o) {{ return o.p2; }}
function r3(o) {{ return o.p3; }}
function rz(o) {{ return o.zz; }}
function rg(o) {{ return o.g; }}
function ra(o) {{ return o.{first}; }}
function rb(o) {{ return o.{second}; }}
function run(f, n) {{ var s = ""; for (var i = 0; i < n; i++) s += f(objs[i & 7]) + ","; return s; }}
var sites = [rx, r1, r2, r3, rz, ra, rb];
for (var k = 0; k < sites.length; k++) run(sites[k], 16);
"#
    )
}

/// Steps after the program: the first five are warm sites, read natively.
const STEPS: [&str; 14] = [
    "run(rx, 80)",
    "run(r1, 80)",
    "run(r2, 80)",
    "run(r3, 80)",
    "run(rz, 80)",
    "run(ra, 80)",
    "run(rb, 80)",
    // The colliding keys, alternating: each read's fill overwrites the other key's entry.
    "var s = ''; for (var k = 0; k < 4; k++) s += run(ra, 16) + run(rb, 16); s",
    // An accessor entry: the getter runs once a read, with its receiver as `this`.
    "run(rg, 16); calls = 0; run(rg, 40) + calls",
    // A prototype changed after the fills: a shadowing level, an addition at the chain's end.
    "P1.p2 = 'shadow'; run(r2, 24)",
    "P2.p3 = 'mid'; run(r3, 24)",
    "Object.prototype.zz = 'end'; run(rz, 24)",
    "delete P1.p2; delete P2.p3; delete Object.prototype.zz; run(r2, 16) + run(r3, 16) + run(rz, 16)",
    "run(rx, 16)",
];

/// Each step's value; a read step (the first five) runs twice, and its second pass must be all
/// native: a site's entries never take each other's slots (one key, distinct shapes), but another
/// site's entry may have taken one since the warm pass (eight shapes and seven keys share 4096
/// slots), which the first pass refills.
fn steps(engine: &mut Engine, tier: Tier) -> Vec<String> {
    let mut values = Vec::new();
    for (k, step) in STEPS.into_iter().enumerate() {
        let (_, _, value) = reads(engine, step);
        if k < 5 {
            let (native, checked, again) = reads(engine, step);
            assert_eq!(again, value);
            if tier == Tier::Jit {
                assert_eq!(
                    (native, checked),
                    (80, 0),
                    "step {k} `{step}`: every read native once the entries are filled"
                );
            }
        }
        values.push(value);
    }
    values
}

#[test]
fn a_megamorphic_read_is_served_by_the_engine_wide_cache_in_generated_code() {
    // Counts the template JIT's reads: with the tier on, `run`'s loop would reach its
    // checkpoints and the tier's code would make them.
    crate::opt::pin_tier_off_for_test();
    let mut jit = engine(Tier::Jit);
    let (first, second) = colliding_names(&mut jit);
    let source = program(&first, &second);
    eval(&mut jit, &source);
    let expected = steps(&mut jit, Tier::Jit);
    for tier in [Tier::Interp, Tier::Bytecode] {
        let mut engine = engine(tier);
        eval(&mut engine, &source);
        assert_eq!(steps(&mut engine, tier), expected, "{tier:?}");
    }
    // A warm engine: the read functions run on the VM first, so their native code is made with
    // the ways the VM filled (four of eight shapes, the site marked generic) and probes the cache
    // through the generic way stub (PropWays) after its in-line ways, where the engine above's
    // code, made at the first call, took the compact read stub (PropRead).
    let mut warm = engine(Tier::Jit);
    warm.set_tier_threshold(1);
    warm.interp.jit_admit_runs = 20;
    let (first, second) = colliding_names(&mut warm);
    eval(&mut warm, &program(&first, &second));
    assert_eq!(steps(&mut warm, Tier::Jit), expected, "warm");
    let cycle = |f: &dyn Fn(usize) -> String, n: usize| (0..n).map(|i| f(i & 7) + ",").collect();
    let x: String = cycle(&|i| (10 + i).to_string(), 80);
    assert_eq!(expected[0], x);
    assert_eq!(expected[1], "1,".repeat(80));
    assert_eq!(expected[4], "undefined,".repeat(80));
    let a: String = cycle(&|i| (100 + i).to_string(), 16);
    let b: String = cycle(&|i| (200 + i).to_string(), 16);
    assert_eq!(expected[7], format!("{a}{b}").repeat(4));
    let g: String = cycle(&|i| i.to_string(), 40);
    assert_eq!(expected[8], format!("{g}40"));
    assert_eq!(expected[9], "shadow,".repeat(24));
    assert_eq!(expected[10], "mid,".repeat(24));
    assert_eq!(expected[11], "end,".repeat(24));
    assert_eq!(
        expected[12],
        format!(
            "{}{}{}",
            "2,".repeat(16),
            "3,".repeat(16),
            "undefined,".repeat(16)
        )
    );
}

#[test]
fn two_engines_alive_on_one_thread_each_probe_their_own_table() {
    // The read stubs are process-wide: each engine's reads must find its own interpreter's
    // table through the context, never an address the stub was made with.
    crate::opt::pin_tier_off_for_test();
    let mut first = engine(Tier::Jit);
    let mut second = engine(Tier::Jit);
    for engine in [&mut first, &mut second] {
        let source = program("fa", "fb");
        eval(engine, &source);
    }
    for round in 0..2 {
        for (label, engine) in [("first", &mut first), ("second", &mut second)] {
            for site in ["rx", "r1", "r3", "rz"] {
                // A first pass refills an entry another site's took; the second is all native.
                reads(engine, &format!("run({site}, 40)"));
                let (native, checked, _) = reads(engine, &format!("run({site}, 40)"));
                assert_eq!(
                    (native, checked),
                    (40, 0),
                    "round {round}, the {label} engine's {site}: native reads"
                );
            }
        }
    }
}

/// An inlined frame's megamorphic read (`gx`'s `o.x` over eight shapes, a generic node of the
/// inlined frame under graph inlining): the node's template calls its read stub before any
/// helper, with nothing of the frame's context switched, and the stub's probe needs none (it
/// reads `ctx.interp`, the site's key and cache words and the receiver; in test builds the lazy
/// switch leaves a trap in `JitCtx::chunk`, which faults any read of the chunk). The reads hit the
/// cache natively; a receiver whose `x` is a throwing getter misses into the helper, whose
/// switched context names the inlined frame in the error's stack as the tier off does.
const INLINED: &str = r#"
var shapes = [];
for (var i = 0; i < 8; i++) { var o = {}; o["s" + i] = i; o.x = i; shapes.push(o); }
var thrower = { get x() { throw new Error("boom"); } };
function gx(o) { return o.x; }
function run(n, bad) {
  var s = 0, st = "";
  for (var i = 0; i < n; i++) {
    s += gx(shapes[i & 7]);
    if (bad && i === n - 1) {
      try { gx(thrower); } catch (e) { st = String(e.stack).split("\n").slice(0, 3).join("|"); }
    }
  }
  return s + ":" + st;
}
var r = ""; for (var k = 0; k < 200; k++) r = run(40, k === 199); r"#;

#[test]
fn an_inlined_frames_megamorphic_read_hits_the_cache_and_its_miss_names_the_frame() {
    crate::opt::TestSwitches::default().set();
    let mut off = engine(Tier::Jit);
    let expected = eval(&mut off, INLINED);
    assert!(expected.contains("gx"), "{expected}");
    crate::opt::TestSwitches {
        on: true,
        eager: true,
        asynchronous: false,
        inline_graph: Some(true),
        ..Default::default()
    }
    .set();
    let mut graph = engine(Tier::Jit);
    crate::opt::a64::TEST_INLINED_GENERIC.with(|ops| ops.borrow_mut().clear());
    assert_eq!(eval(&mut graph, INLINED), expected);
    let ops =
        crate::opt::a64::TEST_INLINED_GENERIC.with(|ops| std::mem::take(&mut *ops.borrow_mut()));
    assert!(
        ops.iter().any(|op| matches!(
            op,
            crate::bytecode::Op::GetProp(..) | crate::bytecode::Op::GetPropLocal(..)
        )),
        "gx's read is an inlined frame's generic node: {ops:?}"
    );
    // The tier's code, warm: every read of the eight shapes native (four through the ways, four
    // through the cache), the getter's one read through the helper.
    let (native, checked, value) = reads(&mut graph, "run(40, true)");
    assert_eq!(value, expected);
    assert_eq!(
        (native, checked),
        (40, 1),
        "native reads and the getter's helper call"
    );
    crate::opt::TestSwitches::default().set();
}

/// Only a megamorphic site (its way 0 holding the generic mark) probes the table. A site of one
/// shape whose ways are still empty takes the helper and fills its own ways, though the table
/// holds another site's entry for the same shape and key; served from the table, it would never
/// fill them, and the tier, which reads a site's ways, would see no feedback for it. Its next read
/// hits its own way.
#[test]
fn a_site_short_of_megamorphic_fills_its_own_ways_and_does_not_probe_the_table() {
    crate::opt::pin_tier_off_for_test();
    let mut jit = engine(Tier::Jit);
    eval(
        &mut jit,
        "var o = { x: 1, y: 2 }; function a(o) { return o.x; } function b(o) { return o.x; }
         a(o); a(o);",
    );
    assert_eq!(
        reads(&mut jit, "b(o)"),
        (0, 1, "1".to_string()),
        "the helper fills b's way"
    );
    assert_eq!(
        reads(&mut jit, "b(o)"),
        (1, 0, "1".to_string()),
        "b's own way"
    );
}

/// The global `name`'s value: a function's compiled chunk, or an object's shape.
fn global(engine: &mut Engine, name: &str) -> crate::value::Value {
    let env = engine.interp.global_env;
    engine.interp.get_var(name, &env).ok().expect("bound name")
}

fn chunk_of(engine: &mut Engine, name: &str) -> std::rc::Rc<crate::bytecode::Chunk> {
    let crate::value::ValueView::Obj(function) = global(engine, name).view() else {
        panic!("{name} is not an object")
    };
    let crate::value::ObjectKind::User(user) = &function.borrow().kind else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    let chunk = runtime.code.get().and_then(Option::as_ref);
    std::rc::Rc::clone(chunk.unwrap_or_else(|| panic!("{name} stayed interpreted")))
}

fn shape_of(engine: &mut Engine, name: &str) -> u32 {
    let crate::value::ValueView::Obj(object) = global(engine, name).view() else {
        panic!("{name} is not an object")
    };
    let shape = object.borrow().props.shape();
    shape
}

/// The receiver shapes of `function`'s one named read's ways, and whether way 0 holds the
/// generic mark.
fn ways_of(engine: &mut Engine, function: &str) -> ([u32; crate::bytecode::PROP_IC_WAYS], bool) {
    use crate::bytecode::Op;
    let chunk = chunk_of(engine, function);
    let caches: Vec<u32> = chunk
        .jit_ops()
        .iter()
        .filter_map(|op| match *op {
            Op::GetProp(_, cache) | Op::GetPropLocal(_, _, cache) => Some(cache),
            _ => None,
        })
        .collect();
    assert_eq!(caches.len(), 1, "one named read in {function}");
    let ways = chunk.jit_cache_ways(caches[0]);
    (
        ways.map(|way| way.recv_shape),
        ways[0].mid_ok & crate::bytecode::IC_MID_GENERIC != 0,
    )
}

/// A read the engine-wide cache serves in generated code is promoted into its site's ways as the
/// helper's hit of the cache is (`Interp::try_ic_get` -> `ic_insert`): way 0 takes the receiver's
/// shape with the site's generic mark and the other ways move down one, at a site made cold (the
/// compact read stub) and at a site made warm (the way stub after its baked ways).
#[test]
fn a_read_the_cache_serves_in_generated_code_is_promoted_into_its_sites_ways() {
    crate::opt::pin_tier_off_for_test();
    for warm in [false, true] {
        let mut jit = engine(Tier::Jit);
        if warm {
            jit.set_tier_threshold(1);
            jit.interp.jit_admit_runs = 20;
        }
        // Sixteen receiver shapes: more than the warm code's four baked ways and four ways hold.
        let source = program("a", "b")
            + "for (var k = 0; k < 16; k++) {
                 var o = k < 8 ? objs[k] : Object.create(P1);
                 if (k >= 8) { o['t' + k] = k; o.x = 10 + k; }
                 globalThis['o' + k] = o;
               }
               for (var k = 0; k < 64; k++) rx(globalThis['o' + (k & 15)]);";
        eval(&mut jit, &source);
        let mut promoted = 0;
        for k in [0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 5, 14, 7] {
            let shape = shape_of(&mut jit, &format!("o{k}"));
            let (before, generic) = ways_of(&mut jit, "rx");
            assert!(generic, "warm {warm}: rx's site is megamorphic");
            let (native, checked, value) = reads(&mut jit, &format!("rx(o{k})"));
            assert_eq!(value, (10 + k).to_string());
            let (after, generic) = ways_of(&mut jit, "rx");
            assert!(generic, "warm {warm}: the mark stays in way 0");
            // A shape in the ways is read there; at the warm site, one of the four the code baked
            // is read in line: neither reaches the cache.
            if before.contains(&shape) || (warm && after == before) {
                continue;
            }
            assert_eq!(
                (native, checked),
                (1, 0),
                "warm {warm}: o{k}'s read is native"
            );
            assert_eq!(
                after,
                [shape, before[0], before[1], before[2]],
                "warm {warm}: o{k}'s entry promoted into way 0"
            );
            promoted += 1;
        }
        assert!(
            promoted >= if warm { 4 } else { 8 },
            "warm {warm}: {promoted} promotions"
        );
    }
}

/// Key-checked entries (`IC_ARR_KEYCHK`) are served by the cache at a megamorphic site: arrays of
/// eight shapes read a method of Array.prototype (one hop to an Array holder) and their own
/// `length` natively once the entries are filled, as every tier reads them.
#[test]
fn a_megamorphic_sites_key_checked_array_entries_are_served_in_generated_code() {
    crate::opt::pin_tier_off_for_test();
    let source = r#"
var arrs = [];
for (var i = 0; i < 8; i++) { var a = [1, 2, i]; a["s" + i] = i; arrs.push(a); }
function rp(a) { return a.push; }
function rl(a) { return a.length; }
var P = Array.prototype.push;
function runa(f, n) { var s = 0; for (var i = 0; i < n; i++) { var v = f(arrs[i & 7]); s += v === P ? 1 : v; } return s; }
runa(rp, 16); runa(rl, 16);
"#;
    let mut values = Vec::new();
    for tier in [Tier::Jit, Tier::Interp, Tier::Bytecode] {
        let mut engine = engine(tier);
        eval(&mut engine, source);
        let mut got = Vec::new();
        for step in ["runa(rp, 80)", "runa(rl, 80)"] {
            let (_, _, first) = reads(&mut engine, step);
            let (native, checked, again) = reads(&mut engine, step);
            assert_eq!(first, again);
            if tier == Tier::Jit && step.contains("rp") {
                assert_eq!(
                    (native, checked),
                    (80, 0),
                    "`{step}`: every read native once the entries are filled"
                );
            }
            got.push(again);
        }
        values.push(got);
    }
    assert_eq!(values[0], ["80", "240"]);
    assert!(values.iter().all(|v| *v == values[0]), "{values:?}");
}
