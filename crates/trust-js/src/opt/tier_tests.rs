//! The optimizing tier end to end (P3 stage 1, milestone 2; stage 2, step 3): programs whose hot
//! functions reach the second-stage checkpoints run with the tier on, eager, and with the
//! forced-exit gate, and their results equal the bytecode VM's; a callee-free loop function runs
//! in the tier's code; repeated exits discard a function's code, and its recompile, past the
//! warm-up, declines what exited.

use super::ir::ExitReason;
use super::profile::{self, ChunkProfile};
use super::TestSwitches;
use crate::bytecode::{Chunk, Tier};
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

fn run(tier: Tier, source: &str) -> String {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    eval(&mut engine, source)
}

/// An engine at the JIT tier that compiles every function at its first call.
fn jit_engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

/// The tier on and eager, nothing forced.
fn eager() -> TestSwitches {
    TestSwitches {
        on: true,
        eager: true,
        ..Default::default()
    }
}

fn compiled() -> Vec<usize> {
    super::TEST_COMPILED.with(|compiled| std::mem::take(&mut *compiled.borrow_mut()))
}

fn exits() -> Vec<(u32, ExitReason)> {
    super::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()))
}

/// The codes discards retired since the last call.
fn discarded() -> Vec<usize> {
    super::TEST_DISCARDED.with(|discarded| std::mem::take(&mut *discarded.borrow_mut()))
}

fn address(chunk: &Rc<Chunk>) -> usize {
    Rc::as_ptr(chunk) as usize
}

/// Whether `code` is the tier's (it has an exit table), not the template JIT's.
fn is_tier_code(code: &crate::jit::JitCode) -> bool {
    super::exit::table_bytes(code.mem_ptr()) != 0
}

/// R: the discards of the function whose recompile `chunk` is.
fn discards(chunk: &Chunk) -> u8 {
    ChunkProfile::get(profile::first_stage(chunk)).map_or(0, ChunkProfile::discards)
}

/// The exit sites keyed on `chunk`: (pc, reason, inlined).
fn sites_on(chunk: &Chunk) -> Vec<(u32, u8, bool)> {
    let mut sites = Vec::new();
    if let Some(profile) = ChunkProfile::get(chunk) {
        for pc in 0..chunk.jit_ops().len() as u32 {
            profile.sites_at(pc, |reason, inlined, _| sites.push((pc, reason, inlined)));
        }
    }
    sites
}

/// The exit sites keyed on a recompile and on its function's first stage.
fn sites(chunk: &Chunk) -> Vec<(u32, u8, bool)> {
    let mut sites = sites_on(chunk);
    sites.extend(sites_on(profile::first_stage(chunk)));
    sites
}

/// Programs whose functions run past the second stage's checkpoints (100, 500, 2,100 and 8,500
/// native runs).
const PROGRAMS: &[&str] = &[
    // A callee-free arithmetic loop.
    "function sum(n) { let s = 0; for (let i = 0; i < n; i++) s = s + i; return s; }\n\
     let t = 0; for (let k = 0; k < 3000; k++) t += sum(k % 50); t",
    // int32 arithmetic that overflows inside its loop.
    "function mulsum(n) { let s = 1; for (let i = 1; i < n; i++) s = (s * 31 + i) | 0; \
     return s; }\n\
     let t = 0; for (let k = 0; k < 3000; k++) t = (t + mulsum(k % 40)) | 0; t",
    // Named reads and writes on constructed objects.
    "function P(x, y) { this.x = x; this.y = y; }\n\
     function len2(p) { return p.x * p.x + p.y * p.y; }\n\
     function bump(p) { p.x = p.x + 1; return p.x; }\n\
     let t = 0; for (let k = 0; k < 3000; k++) { const p = new P(k, k + 1); \
     t += len2(p) + bump(p); } t",
    // Calls, closures and strings.
    "function add(a, b) { return a + b; }\n\
     function twice(f, x) { return f(f(x)); }\n\
     let s = ''; for (let k = 0; k < 3000; k++) \
     s = add(s.length > 100 ? '' : s, String(twice(function (v) { return v + 1; }, k))); s",
    // Branches on values of more than one type: some checks fail.
    "function cls(v) { if (v < 10) return 'small'; if (v < 100) return 'mid'; return 'big'; }\n\
     let out = []; for (let k = 0; k < 3000; k++) out.push(cls(k % 7 === 0 ? String(k) : k));\n\
     out.join(',').length + ':' + out[7] + out[3]",
    // Truthiness of every kind of value.
    "function yes(v) { return v ? 1 : 0; }\n\
     const vals = [0, 1, '', 'a', null, undefined, {}, [], NaN, -0, 2n, 0n];\n\
     let t = 0; for (let k = 0; k < 3000; k++) t = t * 2 % 1000003 + yes(vals[k % vals.length]); t",
    // Throws inside a try.
    "function risky(k) { try { if (k % 13 === 0) throw new Error('e' + k); return k; } \
     catch (e) { return -1; } }\n\
     let t = 0; for (let k = 0; k < 3000; k++) t += risky(k); t",
    // for-of over an Array.
    "function total(a) { let s = 0; for (const v of a) s += v; return s; }\n\
     const arr = [1, 2, 3, 4, 5]; let t = 0; for (let k = 0; k < 3000; k++) t += total(arr); t",
    // ToInt32 of doubles: products and sums past the int32 range, fractions, NaN, the
    // infinities, 2^63 and past it, and `>>>`'s uint32 results.
    "function mix(a, b) { return ((a * b) & 0xffff) ^ ((a + 0.5) | 0) ^ ~(b * 1.5) ^ \
     ((a * 65536 * 65536) >>> 0) ^ (b >>> (a & 31)); }\n\
     const vals = [0, -0, 1.5, -1.5, 2147483648, -2147483649, 4294967296.5, 9007199254740992, \
     1e20, -1e20, 9223372036854777856, Infinity, -Infinity, NaN, 1e300];\n\
     let t = 0; for (let k = 0; k < 3000; k++) t = (t + mix(vals[k % vals.length], k)) % 1000003; t",
    // A shape change after the feedback settles.
    "function getx(o) { return o.x; }\n\
     let t = 0; for (let k = 0; k < 3000; k++) t += getx(k < 2500 ? {x: k} : {y: 0, x: -k}); t",
    // A for-of whose body sometimes throws: the iterator's handler closes it on the way out.
    "function firstBig(a) { for (const v of a) { if (v > 1000) throw new RangeError('big ' + v); } \
     return a.length; }\n\
     function run(k) { try { return firstBig(k % 97 === 0 ? [1, 2, 5000] : [1, 2, 3]); } \
     catch (e) { return e.message; } }\n\
     let out = ''; for (let k = 0; k < 3000; k++) { const r = run(k); \
     out += typeof r === 'string' ? r[4] : r; }\n\
     out.length + out.slice(0, 40)",
    // Element reads (P3 stage 2 step 2): ints, doubles, holes, past the used words, an empty
    // block and new Array(n); -0, negative, fractional, String and past-int32 keys.
    "function rd(a, i) { return a[i]; }\n\
     const arrs = [[1, 2, 3], [1.5, , 3.5], [], new Array(4), ['a', 'b'], {0: 'o', 2: 'p'}];\n\
     const keys = [0, 1, 2, 3, -0, -1, 1.5, '1', 2147483648, 2];\n\
     let out = ''; for (let k = 0; k < 3000; k++) \
     out += String(rd(arrs[k % arrs.length], keys[(k >> 2) % keys.length]))[0];\n\
     out.length + out.slice(0, 64)",
    // Holes read under the prototype proof, and an index getter installed on Array.prototype
    // mid-loop, then removed.
    "function sumHoles(a) { let s = 0; for (let i = 0; i < 6; i++) { const v = a[i]; \
     s += v === undefined ? 1 : v; } return s; }\n\
     const h = [1, , 3, , 5, ,];\n\
     let t = 0; for (let k = 0; k < 3000; k++) { if (k === 1500) \
     Object.defineProperty(Array.prototype, 3, { get() { return 100; }, configurable: true }); \
     if (k === 2500) delete Array.prototype[3]; t += sumHoles(h); } t",
    // Array subclasses, arrays with an index among their named entries, frozen arrays.
    "class MyArr extends Array {}\n\
     function pair(a) { return a[0] * 10 + a[1]; }\n\
     const m = MyArr.from([1, 2]); const f = [3, 4]; \
     Object.defineProperty(f, 5, { get() { return 9; } }); const z = Object.freeze([5, 6]);\n\
     let t = 0; for (let k = 0; k < 3000; k++) t += pair(k % 3 === 0 ? m : k % 3 === 1 ? f : z); t",
    // Crypto's am3 (Octane crypto.js), its digits in two arrays, on a BigInteger-like receiver.
    "function BI(n, seed) { this.array = []; this.t = n; \
     for (let i = 0; i < n; i++) this.array.push((i * seed + 12345) & 0xfffffff); }\n\
     BI.prototype.am = function am3(i, x, w, j, c, n) { var this_array = this.array; \
     var w_array = w.array; var xl = x & 0x3fff, xh = x >> 14; \
     while (--n >= 0) { var l = this_array[i] & 0x3fff; var h = this_array[i++] >> 14; \
     var m = xh * l + h * xl; l = xl * l + ((m & 0x3fff) << 14) + w_array[j] + c; \
     c = (l >> 28) + (m >> 14) + xh * h; w_array[j++] = l & 0xfffffff; } return c; };\n\
     const x = new BI(20, 7919); const w = new BI(40, 104729); let c = 0;\n\
     for (let k = 0; k < 3000; k++) c = (c + x.am(0, (k * 31) & 0xfffffff, w, k % 20, 0, 20)) | 0;\n\
     c + ':' + w.array.slice(0, 6).join(',')",
    // Element stores: int32s, doubles, -0, NaN, past-int32 Numbers and references into int32,
    // double and copy-on-write words, frozen and sealed arrays, holes, appends and gaps.
    "function put(a, i, v) { a[i] = v; return a; }\n\
     function lit() { return [1, 2, 3, 4]; }\n\
     const vals = [7, 2.5, -0, NaN, 'w', null, {t: 1}, 2147483648];\n\
     let out = ''; for (let k = 0; k < 3000; k++) { const s = k % 5; \
     const a = s === 0 ? lit() : s === 1 ? Object.freeze([1, 2, 3, 4]) : \
     s === 2 ? Object.seal([1.5, 2, 3, 4]) : s === 3 ? [1, , 3, 4] : [1, 2, 3, 4]; \
     put(a, k % 6, vals[k % vals.length]); \
     out += a.length + String(Object.is(a[k % 6], -0) ? 'z' : a[k % 6])[0]; }\n\
     out.length + out.slice(0, 80) + lit().join()",
    // Appends from the empty block and across capacities, hole fills, `i < a.length` loops,
    // element updates and compound element assignments.
    "function build(n) { const a = []; for (let i = 0; i < n; i++) a[i] = i * 2; return a; }\n\
     function holes(n) { const h = new Array(n); for (let i = n - 1; i >= 0; i -= 2) h[i] = i; \
     return h; }\n\
     function sum(a) { let s = 0; for (let i = 0; i < a.length; i++) \
     s += a[i] === undefined ? 1000 : a[i]; return s; }\n\
     function bump(a) { for (let i = 0; i < a.length; i++) { a[i]++; a[i] += 2; a[i] *= 1.5; } \
     return a; }\n\
     let t = 0; for (let k = 0; k < 2000; k++) { const a = bump(build(k % 40)); \
     const h = holes(k % 12); t += sum(a) + a.length + sum(h); } t",
    // References into an array of more than 64 words, read back across collections.
    "function fill(a, n, k) { for (let i = 0; i < n; i++) a[i] = {k: k, i: i}; return a; }\n\
     const big = []; for (let i = 0; i < 100; i++) big.push(i * 0.5);\n\
     let t = 0; for (let k = 0; k < 1000; k++) { fill(big, 100, k); \
     if (k % 50 === 0) { const junk = []; for (let j = 0; j < 200; j++) junk.push([j]); } \
     t += big[k % 100].k + big[(k * 7) % 100].i; } t",
    // Typed arrays: one kind per site read and written, with each kind's conversions (fractions,
    // NaN, -0, past the int32 range, clamps, Float32's rounding), the stub at a site of every
    // kind, typed arrays and Arrays at one site, reads past the length, and buffers detached and
    // shrunk mid-loop.
    "function sumF(t) { let s = 0; for (let i = 0; i < 8; i++) s += t[i]; return s; }\n\
     function setU(u, k) { u[k % 8] = k * 2654435761; return u[(k + 3) % 8]; }\n\
     function fill(t, v) { for (let i = 0; i < 8; i++) t[i] = v * (i - 3); return t; }\n\
     function at(t, i) { return t[i]; }\n\
     function count(t) { let n = 0; while (t[n] !== undefined) n++; return n; }\n\
     const kinds = [Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, \
     Int32Array, Uint32Array, Float32Array, Float64Array];\n\
     const all = kinds.map(K => new K(8)); const u = new Uint32Array(8);\n\
     const rab = new ArrayBuffer(64, {maxByteLength: 64}); let f = new Float64Array(rab, 0, 8);\n\
     const vals = [1.5, -0, NaN, 300.5, -2147483649, 1e10, 0.5, 2.5, Infinity, -1.25];\n\
     let t = 0; for (let k = 0; k < 2000; k++) { const v = vals[k % vals.length]; \
     fill(all[k % 9], v); fill(f, v); \
     if (k === 1200) rab.resize(32); if (k === 1300) rab.resize(64); \
     if (k === 1400) f.buffer.transfer(); if (k === 1500) f = new Float64Array(8); \
     t = (t + (sumF(f) || 0) + setU(u, k) + (at(all[k % 9], k % 9) || 0) + \
     (at(k % 2 ? f : [k, 1], 1) || 0) + count(all[k % 9])) % 1000003; }\n\
     String(t) + all.map(a => Array.from(a, x => Object.is(x, -0) ? 'z' : x).join()).join('|')",
];

fn with_switches(switches: TestSwitches, source: &str) -> String {
    switches.set();
    run(Tier::Jit, source)
}

#[test]
fn results_equal_the_vms_with_the_tier_on_eager_and_forcing_exits() {
    let configurations = [
        (
            "on",
            TestSwitches {
                on: true,
                eager: false,
                exit_every: 0,
                ..Default::default()
            },
        ),
        (
            "eager",
            TestSwitches {
                on: true,
                eager: true,
                exit_every: 0,
                ..Default::default()
            },
        ),
        (
            "forced exits",
            TestSwitches {
                on: true,
                eager: true,
                exit_every: 7,
                ..Default::default()
            },
        ),
    ];
    let mut compiles = 0;
    for program in PROGRAMS {
        let expected = run(Tier::Bytecode, program);
        for (label, switches) in configurations {
            compiled();
            assert_eq!(
                with_switches(switches, program),
                expected,
                "{label}: {program}"
            );
            compiles += compiled().len();
        }
    }
    assert!(
        compiles > 0,
        "the tier compiled some of the programs' functions"
    );
}

fn second_stage(engine: &mut Engine, name: &str) -> Option<std::rc::Rc<crate::bytecode::Chunk>> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok()?;
    let object = value.as_obj()?;
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        return None;
    };
    user.func.runtime().code2.get().and_then(Option::clone)
}

#[test]
fn typed_arrays_of_one_kind_run_in_the_tiers_code() {
    // The VM's runs record each site's element kind, which the first stage's in-line hits never
    // do: the tier then reads and writes each kind in line, with the VM's results.
    const TYPED: &str = "function sumF(t) { let s = 0; for (let i = 0; i < 16; i++) s += t[i]; \
         return s; }\n\
         function scale(t, v) { for (let i = 0; i < 16; i++) t[i] = (t[i] * v + 0.5) % 1000; \
         return t; }\n\
         function clamp(b, v) { for (let i = 0; i < 16; i++) b[i] = v + i * 37; return b[3]; }\n\
         function shorts(w, v) { for (let i = 0; i < 16; i++) w[i] = v * i; return w[15]; }\n\
         const f = new Float64Array(16), c = new Uint8ClampedArray(16), w = new Int16Array(16);\n\
         for (let k = 0; k < 3; k++) { scale(f, 1.5); sumF(f); clamp(c, 1.5); shorts(w, 1.5); }";
    const RUN: &str = "let t = 0; for (let k = 0; k < 3000; k++) { scale(f, k % 3 ? 1.5 : -0.5); \
         t = (t + sumF(f) + clamp(c, k * 0.75) + shorts(w, k * 1000.5)) % 1000003; }\n\
         String(t) + f.join() + c.join() + w.join()";
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        eval(&mut vm, TYPED);
        eval(&mut vm, RUN)
    };
    eager().set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, TYPED);
    engine.set_tier(Tier::Jit);
    compiled();
    exits();
    assert_eq!(eval(&mut engine, RUN), expected);
    assert!(exits().is_empty(), "the speculation held");
    let tier = compiled();
    use super::ir::{NodeOp, TypedKind};
    for (name, op) in [
        ("sumF", NodeOp::LoadTyped(TypedKind::F64)),
        ("scale", NodeOp::StoreTyped(TypedKind::F64)),
        ("clamp", NodeOp::StoreTyped(TypedKind::U8Clamped)),
        ("shorts", NodeOp::StoreTyped(TypedKind::I16)),
    ] {
        let chunk = second_stage(&mut engine, name).unwrap_or_else(|| panic!("{name}: code2"));
        assert!(
            tier.contains(&address(&chunk)),
            "{name}: the tier compiled it"
        );
        let graph = graph_now(&chunk);
        assert!(graph.nodes.iter().any(|node| node.op == op), "{name}");
    }
}

#[test]
fn a_callee_free_loop_function_runs_in_the_tiers_code() {
    const LOOP: &str =
        "function sum(n) { let s = 0; for (let i = 0; i < n; i++) s = s + i; return s; }";
    // The tier off: a function with nothing to splice keeps its first stage.
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    TestSwitches::default().set();
    eval(&mut engine, LOOP);
    assert_eq!(
        eval(
            &mut engine,
            "let t = 0; for (let k = 0; k < 1000; k++) t += sum(10); t"
        ),
        "45000"
    );
    assert!(second_stage(&mut engine, "sum").is_none());
    // The tier on: the function's own chunk, compiled by the tier at a checkpoint.
    TestSwitches {
        on: true,
        eager: true,
        exit_every: 0,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    compiled();
    eval(&mut engine, LOOP);
    assert_eq!(
        eval(
            &mut engine,
            "let t = 0; for (let k = 0; k < 1000; k++) t += sum(10); t"
        ),
        "45000"
    );
    let chunk = second_stage(&mut engine, "sum").expect("the tier's chunk was published");
    assert!(
        compiled().contains(&(std::rc::Rc::as_ptr(&chunk) as usize)),
        "the tier compiled it"
    );
    // Nothing spliced: the first stage's operations, its feedback forwarded by the record map.
    let first = {
        let env = engine.interp.global_env;
        let value = engine.interp.get_var("sum", &env).ok().unwrap();
        let object = value.as_obj().unwrap();
        let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
            panic!("an author function")
        };
        user.func
            .runtime()
            .code
            .get()
            .and_then(Option::clone)
            .unwrap()
    };
    assert_eq!(
        format!("{:?}", chunk.jit_ops()),
        format!("{:?}", first.jit_ops())
    );
    assert!(chunk.feedback_vector().record_map().is_some());
    let code = chunk.jit.get().flatten().expect("its code is resident");
    exits();
    assert_eq!(
        eval(
            &mut engine,
            "let u = 0; for (let k = 0; k < 1000; k++) u += sum(k % 20); u"
        ),
        "57000"
    );
    assert!(exits().is_empty(), "the loop runs without leaving the code");
    code.referenced.set(0);
    assert_eq!(eval(&mut engine, "sum(100000)"), "4999950000");
    assert!(code.referenced.get() != 0, "the tier's code ran");
}

#[test]
fn a_for_of_whose_binding_takes_its_handlers_operand_runs_in_the_tiers_code() {
    // The for-of pushes its iterator's handler with the step's value on the operand stack, and
    // the binding's store pops it: the frame states inside the loop are shallower than the
    // handler record, as the VM and the first stage allow.
    TestSwitches {
        on: true,
        eager: true,
        exit_every: 0,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    compiled();
    eval(
        &mut engine,
        "function total(a) { let s = 0; for (const v of a) s += JSON.parse(v); return s; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += total(['1', '2', String(k)]); t",
    );
    let chunk = second_stage(&mut engine, "total").expect("the tier compiled total");
    assert!(compiled().contains(&(std::rc::Rc::as_ptr(&chunk) as usize)));
    let code = chunk.jit.get().flatten().expect("its code is resident");
    code.referenced.set(0);
    exits();
    assert_eq!(eval(&mut engine, "total(['4', '5', '6'])"), "15");
    assert!(code.referenced.get() != 0, "the tier's code ran");
    assert!(exits().is_empty());
    // A throw in the loop's call leaves with the throw pending, below the record's depth: the
    // handler closes the iterator and the caller's catch takes the error.
    assert_eq!(
        eval(
            &mut engine,
            "let r = 'none'; try { total(['1', 'x', '3']); } catch (e) { r = e.name; } r"
        ),
        "SyntaxError"
    );
    assert_eq!(
        exits()
            .into_iter()
            .map(|(_, reason)| reason)
            .collect::<Vec<_>>(),
        vec![super::ir::ExitReason::Throw]
    );
    // A failed check inside the loop: the VM resumes in the region with the handler rebuilt.
    assert_eq!(eval(&mut engine, "total(['1', '\"2\"', '3'])"), "123");
    assert_eq!(
        exits()
            .into_iter()
            .map(|(_, reason)| reason)
            .collect::<Vec<_>>(),
        vec![super::ir::ExitReason::NotNumber]
    );
}

#[test]
fn under_no_reopt_repeated_exits_send_new_entries_to_the_template_jits_code_for_good() {
    // TRUST_JS_OPT_NO_REOPT: stage 1's fallback, whose subject this is.
    TestSwitches {
        on: true,
        eager: true,
        no_reopt: true,
        ..Default::default()
    }
    .set();
    let mut engine = jit_engine();
    compiled();
    eval(
        &mut engine,
        "function add(a, b) { return a + b; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += add(k, 1); t",
    );
    let chunk = second_stage(&mut engine, "add").expect("the tier compiled add");
    assert!(compiled().contains(&address(&chunk)));
    let optimized = chunk.jit.get().flatten().expect("resident");
    exits();
    discarded();
    // Strings fail the Number checks: every call exits until the code is discarded.
    assert_eq!(
        eval(
            &mut engine,
            "let s = ''; for (let k = 0; k < 3000; k++) s = add('', k % 10); s"
        ),
        "9"
    );
    let taken = exits().len();
    assert_eq!(
        taken,
        super::EXIT_LIMIT as usize,
        "no exit after the discard"
    );
    assert_eq!(discarded(), vec![optimized.mem_ptr() as usize]);
    let now = chunk.jit.get().flatten().expect("the template JIT's code");
    assert!(!std::rc::Rc::ptr_eq(&now, &optimized));
    assert!(!is_tier_code(&now));
    // No profile, no R, no recompile.
    assert_eq!(discards(&chunk), 0);
    assert!(!super::reoptimizing(&chunk));
    assert!(sites(&chunk).is_empty());
    assert!(compiled().is_empty(), "no recompile");
    assert_eq!(eval(&mut engine, "add(2, 3) + add('a', 'b')"), "5ab");
    assert!(exits().is_empty());
    drop(optimized);
    super::release_retired(&mut engine.interp);
    assert!(
        engine.interp.opt_retired.is_empty(),
        "no activation runs the retired code"
    );
}

#[test]
fn the_settled_feedback_rule_waits_for_an_unchanged_fingerprint() {
    TestSwitches {
        on: true,
        eager: false,
        exit_every: 0,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "function getx(o) { return o.x; } getx({x: 1});",
    );
    let env = engine.interp.global_env;
    let value = engine.interp.get_var("getx", &env).ok().unwrap();
    let object = value.as_obj().unwrap();
    let chunk = {
        let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
            panic!("an author function")
        };
        user.func
            .runtime()
            .code
            .get()
            .and_then(Option::clone)
            .unwrap()
    };
    assert!(!super::settled(&chunk), "the first checkpoint records");
    assert!(super::settled(&chunk), "unchanged feedback has settled");
    eval(&mut engine, "getx({y: 1, x: 2});");
    assert!(!super::settled(&chunk), "a new receiver shape changes it");
    assert!(super::settled(&chunk));
    super::forget(&chunk);
    assert!(!super::settled(&chunk), "a forgotten chunk starts again");
    TestSwitches {
        on: true,
        eager: true,
        exit_every: 0,
        ..Default::default()
    }
    .set();
    super::forget(&chunk);
    assert!(
        super::settled(&chunk),
        "eager compiles at the first checkpoint"
    );
    super::forget(&chunk);
}

/// The first-stage chunk of the global function `name`.
fn first_stage(engine: &mut Engine, name: &str) -> std::rc::Rc<crate::bytecode::Chunk> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().unwrap();
    let object = value.as_obj().unwrap();
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        panic!("an author function")
    };
    user.func
        .runtime()
        .code
        .get()
        .and_then(Option::clone)
        .unwrap()
}

#[test]
fn feedback_that_changes_at_every_checkpoint_compiles_at_the_last() {
    // getx's receivers show a shape new at each checkpoint (native runs 100, 500, 2,100 and
    // 8,500): the tier waits at the first three and compiles at the fourth, whatever the
    // fingerprint says.
    TestSwitches {
        on: true,
        eager: false,
        exit_every: 0,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    compiled();
    super::TEST_WAITS.with(|waits| waits.borrow_mut().clear());
    assert_eq!(
        eval(
            &mut engine,
            "function getx(o) { return o.x; }\n\
             function receiver(k) { if (k < 300) return {x: k}; \
             if (k < 1200) return {a: 0, x: k}; if (k < 5000) return {b: 0, x: k}; \
             return {c: 0, x: k}; }\n\
             var k = 0, t = 0; for (; k < 2300; k++) t += getx(receiver(k)); t"
        ),
        "2643850"
    );
    let first = first_stage(&mut engine, "getx");
    let waits = || {
        super::TEST_WAITS.with(|waits| {
            waits
                .borrow()
                .iter()
                .filter(|&&(chunk, _)| chunk == std::rc::Rc::as_ptr(&first) as usize)
                .map(|&(_, checkpoint)| checkpoint)
                .collect::<Vec<_>>()
        })
    };
    assert_eq!(waits(), vec![1, 2, 3]);
    assert!(
        second_stage(&mut engine, "getx").is_none(),
        "the tier waits"
    );
    assert_eq!(
        eval(
            &mut engine,
            "for (; k < 9000; k++) t += getx(receiver(k)); t"
        ),
        "40495500"
    );
    let chunk = second_stage(&mut engine, "getx").expect("the last checkpoint compiles");
    assert!(compiled().contains(&(std::rc::Rc::as_ptr(&chunk) as usize)));
    assert_eq!(waits(), vec![1, 2, 3], "no wait at the last checkpoint");
}

/// The pc of `chunk`'s first GetMethod.
fn method_pc(chunk: &crate::bytecode::Chunk) -> usize {
    chunk
        .jit_ops()
        .iter()
        .position(|op| matches!(op, crate::bytecode::Op::GetMethod(..)))
        .expect("a method load")
}

/// The primitive kind of the first way of `chunk`'s named site at `pc`.
fn primitive_at(chunk: &crate::bytecode::Chunk, pc: usize) -> u8 {
    super::snapshot::Snapshot::take(chunk)
        .named(pc)
        .and_then(|named| named.filled().next().copied())
        .expect("a filled way")
        .primitive
}

/// Whether the tier's graph of `chunk` keeps the named site at `pc` an effect, with no shape
/// check anywhere.
fn stays_an_effect(chunk: &crate::bytecode::Chunk, pc: usize) -> bool {
    use super::ir::NodeOp;
    let snapshot = super::snapshot::Snapshot::take(chunk);
    let graph = super::build::build(chunk, &snapshot).expect("the tier builds it");
    !graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, NodeOp::CheckShape(_)))
        && graph
            .nodes
            .iter()
            .any(|node| matches!(node.op, NodeOp::Effect { pc: at, .. } if at as usize == pc))
}

#[test]
fn named_reads_on_primitives_stay_effects_and_never_exit() {
    // Each site's way names String.prototype's (Number's, Boolean's) shape: its fill's receiver
    // was the primitive's wrapper prototype, and the mark keeps the builder from checking the
    // receiver word against it.
    TestSwitches {
        on: true,
        eager: true,
        exit_every: 0,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    compiled();
    const PROGRAM: &str = "function code(s, i) { return s.charCodeAt(i); }\n\
         function fixed(n) { return n.toFixed(1); }\n\
         function shown(b) { return b.toString(); }\n\
         let t = 0; for (let k = 0; k < 1000; k++) \
         t += code('abc', k % 3) + fixed(k).length + shown(k % 2 === 0).length; t";
    let expected = run(Tier::Bytecode, PROGRAM);
    exits();
    // The loop runs on past the first checkpoint, where the tier compiles each function.
    assert_eq!(eval(&mut engine, PROGRAM), expected);
    assert!(exits().is_empty(), "no exit on a primitive receiver");
    let compiles = compiled();
    assert_eq!(
        eval(
            &mut engine,
            "code('xyz', 1) + fixed(2.25) + shown(true) + code('xyz', 2)"
        ),
        "121".to_string() + "2.3" + "true" + "122"
    );
    assert!(exits().is_empty(), "no exit on a primitive receiver");
    use crate::bytecode::{IC_PRIMITIVE_BOOLEAN, IC_PRIMITIVE_NUMBER, IC_PRIMITIVE_STRING};
    for (name, kind) in [
        ("code", IC_PRIMITIVE_STRING),
        ("fixed", IC_PRIMITIVE_NUMBER),
        ("shown", IC_PRIMITIVE_BOOLEAN),
    ] {
        let chunk = second_stage(&mut engine, name).expect("the tier's chunk");
        assert!(
            compiles.contains(&(std::rc::Rc::as_ptr(&chunk) as usize)),
            "the tier compiled {name}"
        );
        let pc = method_pc(&chunk);
        assert_eq!(primitive_at(&chunk, pc), kind, "{name}'s way");
        assert!(stays_an_effect(&chunk, pc), "{name}'s method load");
    }
}

#[test]
fn a_site_that_sees_string_prototype_and_primitive_strings_stays_an_effect() {
    // String.prototype itself and primitive strings share the site's way, whose shape is the
    // prototype's, in either order: the prototype's own fill records the string kind too, since
    // the templates' probe then hits the way with primitives.
    for program in [
        "function first(s) { return s.charAt(0); }\n\
         let out = ''; for (let k = 0; k < 1000; k++) \
         out += first(k % 7 === 0 ? String.prototype : 'abc') + '.'; out.length + out.slice(0, 9)",
        "function first(s) { return s.charAt(0); }\n\
         let out = ''; for (let k = 0; k < 1000; k++) \
         out += first(k % 7 === 3 ? String.prototype : 'abc') + '.'; out.length + out.slice(0, 9)",
    ] {
        TestSwitches {
            on: true,
            eager: true,
            exit_every: 0,
            ..Default::default()
        }
        .set();
        let expected = run(Tier::Bytecode, program);
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        compiled();
        exits();
        assert_eq!(eval(&mut engine, program), expected, "{program}");
        let chunk = second_stage(&mut engine, "first").expect("the tier's chunk");
        assert!(compiled().contains(&(std::rc::Rc::as_ptr(&chunk) as usize)));
        let pc = method_pc(&chunk);
        assert_eq!(
            primitive_at(&chunk, pc),
            crate::bytecode::IC_PRIMITIVE_STRING
        );
        assert!(stays_an_effect(&chunk, pc));
        assert!(exits().is_empty(), "{program}");
        assert_eq!(
            eval(
                &mut engine,
                "first(String.prototype) + first('q') + first('rs')"
            ),
            "qr"
        );
        assert!(exits().is_empty());
    }
}

#[test]
fn a_primitive_mark_survives_seeding_into_a_second_stage() {
    // The tier off: the first stage's way, filled in the bytecode VM from a primitive receiver,
    // keeps its mark in the chunk a recompile seeds from it.
    TestSwitches::default().set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(
        &mut engine,
        "function code(s, i) { return s.charCodeAt(i); }\n\
         let t = 0; for (let k = 0; k < 50; k++) t += code('abc', k % 3); t",
    );
    let first = first_stage(&mut engine, "code");
    let pc = method_pc(&first);
    assert_eq!(
        primitive_at(&first, pc),
        crate::bytecode::IC_PRIMITIVE_STRING
    );
    let env = engine.interp.global_env;
    let value = engine.interp.get_var("code", &env).ok().unwrap();
    let object = value.as_obj().unwrap();
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        panic!("an author function")
    };
    let second = crate::bytecode::compile_with_inlines(&user.func, &Default::default(), &first)
        .expect("the recompile");
    assert_eq!(
        primitive_at(&second, method_pc(&second)),
        crate::bytecode::IC_PRIMITIVE_STRING
    );
}

/// One reason's convergence (the stage 2 design's Tests): `warm` runs the global function
/// `name` until the tier, eager, compiled its second stage; `exit` then makes it exit for
/// `reason` at its calls until the discard (100 exits), after which the template JIT's code and,
/// from the warm-up's first checkpoint (200 native runs at R = 1), the tier's recompile run it
/// without an exit; `after` runs the recompile. Every result equals the VM's. The second stage.
fn converges(name: &str, warm: &str, exit: &str, after: &str, reason: ExitReason) -> Rc<Chunk> {
    converges_with(name, warm, &|_| {}, exit, after, reason)
}

/// [`converges`], with `prepare` run on each engine before `exit`.
fn converges_with(
    name: &str,
    warm: &str,
    prepare: &dyn Fn(&mut Engine),
    exit: &str,
    after: &str,
    reason: ExitReason,
) -> Rc<Chunk> {
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        let warmed = eval(&mut vm, warm);
        prepare(&mut vm);
        [warmed, eval(&mut vm, exit), eval(&mut vm, after)]
    };
    eager().set();
    let mut engine = jit_engine();
    compiled();
    exits();
    discarded();
    assert_eq!(eval(&mut engine, warm), expected[0], "{name}: warm");
    let chunk = second_stage(&mut engine, name).unwrap_or_else(|| panic!("{name}: code2"));
    assert!(compiled().contains(&address(&chunk)), "{name}: compiled");
    let optimized = chunk.jit.get().flatten().expect("resident");
    assert!(is_tier_code(&optimized), "{name}");
    assert!(exits().is_empty(), "{name}: no exit while warm");
    prepare(&mut engine);
    assert_eq!(eval(&mut engine, exit), expected[1], "{name}: exit");
    let taken = exits();
    assert_eq!(
        taken.len(),
        super::EXIT_LIMIT as usize,
        "{name}: exits until the discard and none after: {taken:?}"
    );
    assert!(
        taken.iter().all(|&(_, why)| why == reason),
        "{name}: {taken:?}"
    );
    assert_eq!(
        discarded(),
        vec![optimized.mem_ptr() as usize],
        "{name}: one discard"
    );
    assert_eq!(discards(&chunk), 1, "{name}: R");
    assert!(
        compiled().contains(&address(&chunk)),
        "{name}: the recompile"
    );
    assert!(!super::reoptimizing(&chunk), "{name}: the warm-up ended");
    let now = chunk.jit.get().flatten().expect("resident");
    assert!(
        !Rc::ptr_eq(&now, &optimized) && is_tier_code(&now),
        "{name}: the recompile is resident"
    );
    now.referenced.set(0);
    assert_eq!(eval(&mut engine, after), expected[2], "{name}: after");
    assert!(exits().is_empty(), "{name}: no exit after the recompile");
    assert!(now.referenced.get() != 0, "{name}: the recompile ran");
    drop((optimized, now));
    super::release_retired(&mut engine.interp);
    chunk
}

/// The tier's graph of `chunk` as a compile would build it now.
fn graph_now(chunk: &Chunk) -> super::ir::Graph {
    super::build::build(chunk, &super::snapshot::Snapshot::take(chunk)).expect("built")
}

/// The global `name`'s object, its shape made one no cache way takes, as a map's past the
/// process's shape ids (`Props::force_uncacheable_shape_for_test`).
fn make_uncacheable(engine: &mut Engine, name: &str) {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("a global");
    let object = value.as_obj().expect("an object");
    object.borrow_mut().props.force_uncacheable_shape_for_test();
}

#[test]
fn an_uncacheable_receiver_converges_on_the_generic_access() {
    // Its exits fill no way: the cache's refused fill marks the site generic (W4 step 5), which
    // the recompile reads, the profile's WrongShape site a backstop for compiles that read
    // another site's cache (a splice elsewhere; `exit_sites_key_where_every_compile...`).
    let chunk = converges_with(
        "getx",
        "function getx(o) { return o.x; }\n\
         var p = {x: 1}, u = {x: 5};\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += getx(p); t",
        &|engine| make_uncacheable(engine, "u"),
        "let s = 0; for (let k = 0; k < 600; k++) s += getx(u); s",
        "getx(u) * 10 + getx(p)",
        ExitReason::WrongShape,
    );
    let graph = graph_now(&chunk);
    assert!(!graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, super::ir::NodeOp::CheckShape(_))));
}

#[test]
fn a_new_shape_that_fills_a_way_is_checked_again() {
    // Its exits fill a second way: the recompile checks both shapes.
    let chunk = converges(
        "getx",
        "function getx(o) { return o.x; }\n\
         var p = {x: 1}, q = {x: 2, a: 0};\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += getx(p); t",
        "let s = 0; for (let k = 0; k < 600; k++) s += getx(q); s",
        "getx(q) * 10 + getx(p)",
        ExitReason::WrongShape,
    );
    let graph = graph_now(&chunk);
    assert!(graph.declines.is_empty(), "{:?}", graph.declines);
    assert!(graph.nodes.iter().any(
        |node| matches!(&node.op, super::ir::NodeOp::CheckShape(shapes) if shapes.len() == 2)
    ));
}

#[test]
fn strings_into_add_converge_on_the_generic_add() {
    converges(
        "add",
        "function add(a, b) { return a + b; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += add(k, 1); t",
        "let s = ''; for (let k = 0; k < 600; k++) s = add('', k % 10); s",
        "add('a', 'b') + add(2, 3)",
        ExitReason::NotNumber,
    );
}

#[test]
fn an_int32_overflow_converges_on_f64_arithmetic() {
    // `(a | 0) + (b | 0)` adds proven int32s, whose int32 add ignores the records: only the
    // profile's Overflow flag takes the recompile to f64.
    let chunk = converges(
        "f",
        "function f(a, b) { return (a | 0) + (b | 0); }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = (t + f(k, 1)) % 1000; t",
        "let u = 0; for (let k = 0; k < 600; k++) u = (u + f(2147483647, k)) % 1000003; u",
        "f(2147483647, 2147483647) + f(1, 2)",
        ExitReason::Overflow,
    );
    let graph = graph_now(&chunk);
    use super::ir::NodeOp;
    assert!(graph.nodes.iter().any(|node| node.op == NodeOp::F64Add));
    assert!(!graph.nodes.iter().any(|node| node.op == NodeOp::I32Add));
}

#[test]
fn an_array_that_gains_a_hole_converges_on_reading_holes() {
    // The first stage's in-line hits record nothing, so the site is compiled from an empty
    // record (the kind tag's branch); its hole exits flag Hole, and the recompile reads holes as
    // undefined under the prototype proof.
    let chunk = converges(
        "get",
        "function get(a, i) { return a[i]; }\n\
         var p = [1, 2, 3, 4];\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += get(p, k & 3); t",
        "delete p[1]; let s = 0; for (let k = 0; k < 600; k++) \
         s += get(p, 1) === undefined ? 1 : 0; s",
        "String(get(p, 1)) + get(p, 2)",
        ExitReason::Hole,
    );
    let graph = graph_now(&chunk);
    use super::ir::{Holes, NodeOp};
    assert!(graph
        .nodes
        .iter()
        .any(|node| node.op == NodeOp::LoadElement(Holes::Undefined)));
    assert!(!graph
        .nodes
        .iter()
        .any(|node| node.op == NodeOp::LoadElement(Holes::Exit)));
}

#[test]
fn a_typed_read_past_the_length_converges_on_undefined() {
    // The stub's exits flag OutOfBounds, and the first stage's misses record the array's kind:
    // the recompile reads that kind, undefined past the length.
    let chunk = converges(
        "get",
        "function get(t, i) { return t[i]; }\n\
         var f = new Float64Array([1.5, 2.5, 3.5, 4.5]);\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += get(f, k & 3); t",
        "let s = 0; for (let k = 0; k < 600; k++) \
         s += get(f, 4 + (k & 1)) === undefined ? 1 : 0; s",
        "String(get(f, 4)) + get(f, 1) + get(f, -1)",
        ExitReason::OutOfBounds,
    );
    let graph = graph_now(&chunk);
    use super::ir::{NodeOp, TypedKind};
    assert!(graph
        .nodes
        .iter()
        .any(|node| node.op == NodeOp::LoadTypedOrUndefined(TypedKind::F64)));
}

#[test]
fn a_fifth_shape_converges_on_the_generic_access() {
    // Four shapes share `x`'s slot, which the code checks; a fifth fills no way.
    converges(
        "getx",
        "function getx(o) { return o.x; }\n\
         const four = [{x: 1}, {x: 2, a: 0}, {x: 3, b: 0}, {x: 4, c: 0}];\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += getx(four[k % 4]); t",
        "const fifth = {x: 5, d: 0}; let u = 0; for (let k = 0; k < 600; k++) u += getx(fifth); u",
        "getx({x: 6, e: 0}) + getx(four[0]) + getx(fifth)",
        ExitReason::WrongShape,
    );
}

#[test]
fn a_call_over_four_callees_converges_on_the_guards_mismatch_edge() {
    // Richards' `t.run()`: one callee while the second stage is built, which splices it; four
    // afterwards, which the recompile's guard sends to the generic call in place.
    let chunk = converges(
        "runTask",
        "function A() {} A.prototype.run = function () { return 1; };\n\
         function B() {} B.prototype.run = function () { return 2; };\n\
         function C() {} C.prototype.run = function () { return 3; };\n\
         function D() {} D.prototype.run = function () { return 4; };\n\
         function runTask(t) { return t.run(); }\n\
         const tasks = [new A(), new B(), new C(), new D()];\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += runTask(tasks[0]); t",
        "let u = 0; for (let k = 0; k < 600; k++) u += runTask(tasks[k % 4]); u",
        "runTask(tasks[1]) * 1000 + runTask(tasks[2]) * 100 + runTask(tasks[3]) * 10 + \
         runTask(tasks[0])",
        ExitReason::WrongCallee,
    );
    let graph = graph_now(&chunk);
    use super::ir::NodeOp;
    assert!(graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, NodeOp::CalleeMatches { .. })));
    assert!(!graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, NodeOp::CheckCallee { .. })));
}

#[test]
fn a_block_that_starts_to_run_converges_on_its_code() {
    converges(
        "nr",
        "function nr(o, c) { if (c) { return o.y + 1; } return 1; }\n\
         const p = {y: 1};\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += nr(p, false); t",
        "let u = 0; for (let k = 0; k < 600; k++) u += nr(p, true); u",
        "nr(p, true) * 10 + nr(p, false)",
        ExitReason::NeverRan,
    );
}

/// A run whose int32 sums overflow at every other call, over 20,000 calls after the warm-up.
const FLIPPING: &str = "function f(a, b) { return (a | 0) + (b | 0); }\n\
     let t = 0; for (let k = 0; k < 1000; k++) t = (t + f(k, 1)) % 1000;\n\
     for (let k = 0; k < 20000; k++) t = (t + f(k % 2 ? 2147483647 : k, k)) % 1000003; t";

/// [`FLIPPING`]'s result under `switches`, R and the discards of `f`.
fn flipping(switches: TestSwitches) -> (String, u8, usize) {
    switches.set();
    let mut engine = jit_engine();
    discarded();
    let result = eval(&mut engine, FLIPPING);
    let chunk = second_stage(&mut engine, "f").expect("f's second stage");
    (result, discards(&chunk), discarded().len())
}

#[test]
fn backoff_keeps_the_discards_of_flipping_overflows_logarithmic() {
    let expected = run(Tier::Bytecode, FLIPPING);
    // With the profile, the recompile takes f64 arithmetic: one discard.
    let (result, r, count) = flipping(eager());
    assert_eq!(result, expected);
    assert_eq!((r, count), (1, 1));
    // Without it every recompile overflows again: each discard doubles the next code's limit
    // (100 << R exits) and the warm-up before its recompile, so that 20,000 calls make no more
    // than log2(20,000 / 100) discards, and R counts them.
    let (result, r, count) = flipping(TestSwitches {
        no_exit_profile: true,
        ..eager()
    });
    assert_eq!(result, expected);
    assert!((2..=7).contains(&count), "{count} discards");
    assert_eq!(usize::from(r), count);
}

#[test]
fn a_discard_of_unfixable_exits_only_arms_no_recompile() {
    // `throw k` in a try exits to the VM, which runs the catch: nothing to decline.
    eager().set();
    let mut engine = jit_engine();
    compiled();
    eval(
        &mut engine,
        "function risky(k) { try { if (k < 0) throw k; return k; } catch (e) { return -1; } }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += risky(k); t",
    );
    let chunk = second_stage(&mut engine, "risky").expect("risky's second stage");
    assert!(compiled().contains(&address(&chunk)));
    let optimized = chunk.jit.get().flatten().expect("resident");
    exits();
    discarded();
    assert_eq!(
        eval(
            &mut engine,
            "let u = 0; for (let k = 0; k < 3000; k++) u += risky(-1 - k % 2); u"
        ),
        "-3000"
    );
    let taken = exits();
    assert_eq!(taken.len(), super::EXIT_LIMIT as usize);
    assert!(
        taken
            .iter()
            .all(|&(_, why)| matches!(why, ExitReason::Unsupported | ExitReason::Throw)),
        "{taken:?}"
    );
    assert_eq!(discarded(), vec![optimized.mem_ptr() as usize]);
    assert_eq!(discards(&chunk), 1, "a discard all the same");
    assert!(!super::reoptimizing(&chunk), "no warm-up");
    assert!(sites(&chunk).is_empty(), "nothing flagged");
    assert!(compiled().is_empty(), "no recompile");
    let now = chunk.jit.get().flatten().expect("the template JIT's code");
    assert!(!is_tier_code(&now));
}

#[test]
fn a_long_activation_finishes_in_its_discarded_code_while_later_calls_run_the_new() {
    // f(x, n) for n > 0 calls f(x, 0) n times, which adds 1 to x: the outer activation runs the
    // tier's code while its inner calls exit on a String, until the discard; it finishes in the
    // discarded code, its later inner calls in the template JIT's.
    const SOURCE: &str = "function f(x, n) { if (n > 0) { for (let k = 0; k < n; k++) f(x, 0); \
         return n; } return x + 1; }\n\
         let w = 0; for (let k = 0; k < 300; k++) w += f(k, 3); w";
    eager().set();
    let mut engine = jit_engine();
    compiled();
    eval(&mut engine, SOURCE);
    let chunk = second_stage(&mut engine, "f").expect("f's second stage");
    assert!(compiled().contains(&address(&chunk)));
    let optimized = chunk.jit.get().flatten().expect("resident");
    exits();
    discarded();
    assert_eq!(eval(&mut engine, "f('s', 150)"), "150");
    let taken = exits();
    assert_eq!(taken.len(), super::EXIT_LIMIT as usize, "{taken:?}");
    let add = taken[0].0;
    assert!(
        taken
            .iter()
            .all(|&(pc, why)| pc == add && why == ExitReason::NotNumber),
        "only the inner calls exit, at their Add: {taken:?}"
    );
    assert_eq!(discarded(), vec![optimized.mem_ptr() as usize]);
    // Kept while the outer activation ran it; released once nothing does.
    assert!(engine
        .interp
        .opt_retired
        .iter()
        .any(|code| Rc::ptr_eq(code, &optimized)));
    let retired = optimized.mem_ptr() as usize;
    drop(optimized);
    super::release_retired(&mut engine.interp);
    assert!(engine
        .interp
        .opt_retired
        .iter()
        .all(|code| code.mem_ptr() as usize != retired));
    // Later calls: the template JIT's code, then from the warm-up's checkpoint the recompile,
    // which takes the generic add.
    assert_eq!(
        eval(
            &mut engine,
            "let v = ''; for (let k = 0; k < 400; k++) v = f('t' + k % 3, 0); v"
        ),
        "t01"
    );
    assert!(exits().is_empty());
    assert!(compiled().contains(&address(&chunk)), "the recompile");
    let now = chunk.jit.get().flatten().expect("resident");
    assert!(is_tier_code(&now));
    assert_eq!(eval(&mut engine, "f('u', 0) + f(1, 0) + f(2, 4)"), "u124");
    assert!(exits().is_empty());
}

#[test]
fn forced_exits_record_nothing() {
    TestSwitches {
        exit_every: 3,
        ..eager()
    }
    .set();
    let mut engine = jit_engine();
    compiled();
    exits();
    discarded();
    assert_eq!(
        eval(
            &mut engine,
            "function add(a, b) { return a + b; }\n\
             let t = 0; for (let k = 0; k < 3000; k++) t += add(k, 0.5); t"
        ),
        "4500000"
    );
    let chunk = second_stage(&mut engine, "add").expect("add's second stage");
    assert!(compiled().contains(&address(&chunk)));
    let taken = exits();
    assert!(
        taken.len() > super::EXIT_LIMIT as usize,
        "forced exits past the limit: {}",
        taken.len()
    );
    assert!(taken.iter().all(|&(_, why)| why == ExitReason::Forced));
    assert!(discarded().is_empty(), "forced exits discard nothing");
    assert_eq!(discards(&chunk), 0);
    assert!(sites(&chunk).is_empty(), "nothing flagged");
    let code = chunk.jit.get().flatten().expect("resident");
    assert!(is_tier_code(&code));
}

#[test]
fn a_stale_chains_recompile_is_compiled_by_the_tier() {
    // A discarded second stage runs the template JIT's code, whose operator chains bake the
    // shapes its caches held (`jit::build_chain`); a second class's replays ask for the recompile
    // without chains (`Chunk::note_chain_replay`), code3, which the tier compiles as it does a
    // second stage, before the warm-up's own checkpoint.
    TestSwitches {
        exit_limit: 1,
        ..eager()
    }
    .set();
    let mut engine = jit_engine();
    compiled();
    discarded();
    eval(
        &mut engine,
        "function A(x) { this.x = x; } function B(x) { this.y = 0; this.x = x; }\n\
         function one() { return 1; }\n\
         function staged(o) { var k = one(); if (o.x < 50) return k; return 0; }\n\
         var a = new A(1), b = new B(2), t = 0;\n\
         for (var i = 0; i < 300; i++) t += staged(a); t",
    );
    let code2 = second_stage(&mut engine, "staged").expect("staged's second stage");
    assert!(compiled().contains(&address(&code2)));
    // A primitive receiver fails the shape check: at the limit of 1, the discard.
    assert_eq!(eval(&mut engine, "staged(5)"), "0");
    assert_eq!(discarded().len(), 1);
    assert!(super::reoptimizing(&code2));
    assert!(
        code2.property_chains.get(),
        "the template JIT's code builds chains"
    );
    assert_eq!(
        eval(
            &mut engine,
            "for (var i = 0; i < 150; i++) t += staged(b); t"
        ),
        "450"
    );
    let code3 = {
        let env = engine.interp.global_env;
        let value = engine.interp.get_var("staged", &env).ok().unwrap();
        let object = value.as_obj().unwrap();
        let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
            panic!("an author function")
        };
        user.func.runtime().code3.get().and_then(Option::clone)
    }
    .expect("the recompile without property chains");
    assert!(code3.no_property_chains.get());
    assert!(
        compiled().contains(&address(&code3)),
        "the tier compiled it"
    );
    let code = code3.jit.get().flatten().expect("resident");
    assert!(is_tier_code(&code));
    code.referenced.set(0);
    assert_eq!(
        eval(&mut engine, "staged(a) + staged(b) + staged({x: 99})"),
        "2"
    );
    assert!(code.referenced.get() != 0, "later calls run it");
}

#[test]
fn the_exit_limit_doubles_with_r_up_to_its_cap() {
    eager().set();
    let mut engine = jit_engine();
    eval(&mut engine, "function g(a) { return a; } g(1);");
    let chunk = first_stage(&mut engine, "g");
    assert_eq!(super::exit_limit(&chunk), 100);
    let profile = ChunkProfile::of(&chunk);
    assert_eq!(profile.add_discard(), 1);
    assert_eq!(super::exit_limit(&chunk), 200);
    for _ in 0..30 {
        profile.add_discard();
    }
    // R stops where the warm-up's last checkpoint, 6,400 << R runs away, still fits its word.
    assert_eq!(profile.discards(), profile::MAX_DISCARDS);
    assert_eq!(super::exit_limit(&chunk), 100 << 19);
    assert!(6400u64 << profile::MAX_DISCARDS <= u64::from(u32::MAX));
    TestSwitches {
        exit_limit: 1,
        ..eager()
    }
    .set();
    assert_eq!(super::exit_limit(&chunk), 1 << 19);
}

#[test]
fn an_exit_limit_of_one_cycles_reoptimization_and_results_hold() {
    // TRUST_JS_OPT_EXIT_LIMIT=1, the gates' cycling configuration: each code is discarded at its
    // first exit (the limit 100 << R becomes 1 << R).
    let switches = TestSwitches {
        exit_limit: 1,
        ..eager()
    };
    let mut compiles = 0;
    for program in PROGRAMS {
        let expected = run(Tier::Bytecode, program);
        compiled();
        assert_eq!(with_switches(switches, program), expected, "{program}");
        compiles += compiled().len();
    }
    assert!(compiles > 0);
}

// ---- free names (stage 2, step 4) ---------------------------------------------------------------

/// `source`'s result under the tier eager, with exits forced at every seventh check, and with
/// every code discarded at its first exit (TRUST_JS_OPT_EXIT_LIMIT=1), each against the VM's.
fn names_hold(source: &str) {
    let expected = run(Tier::Bytecode, source);
    for (label, switches) in [
        ("eager", eager()),
        (
            "forced exits",
            TestSwitches {
                exit_every: 7,
                ..eager()
            },
        ),
        (
            "an exit limit of one",
            TestSwitches {
                exit_limit: 1,
                ..eager()
            },
        ),
        (
            "both",
            TestSwitches {
                exit_every: 7,
                exit_limit: 1,
                ..eager()
            },
        ),
    ] {
        compiled();
        assert_eq!(
            with_switches(switches, source),
            expected,
            "{label}: {source}"
        );
        assert!(
            !compiled().is_empty(),
            "{label}: the tier compiled none of {source}"
        );
    }
}

#[test]
fn a_closure_reading_an_outer_let_in_its_tdz_exits_and_the_vm_throws_at_its_read() {
    // `read`'s code checks `x` initialized; an activation that calls it before `let x` runs
    // exits there (Tdz), and the VM's run of the read throws its ReferenceError, caught in the
    // callee's own try, in its caller's, or by no one.
    const SOURCE: &str = "function mk(early, guard) {
           function read() { return x + 1; }
           if (early) {
             if (guard) { try { return read(); } catch (e) { return e.name; } }
             return read();
           }
           let x = 5;
           return read();
         }\n\
         let t = 0; for (let k = 0; k < 2000; k++) t += mk(false, false); t";
    eager().set();
    let mut engine = jit_engine();
    compiled();
    assert_eq!(eval(&mut engine, SOURCE), "12000");
    exits();
    assert_eq!(
        eval(&mut engine, "mk(true, true)"),
        "ReferenceError",
        "caught in the callee's try"
    );
    assert_eq!(
        eval(
            &mut engine,
            "let caught = 'none'; try { mk(true, false); } catch (e) { caught = e.name + ': ' + \
             e.message; } caught"
        ),
        "ReferenceError: cannot access 'x' before initialization"
    );
    assert!(eval(&mut engine, "mk(true, false)").starts_with("throw ReferenceError"));
    // One Tdz exit a call, each at the read itself (a splice of `read` into mk's code guards a
    // closure of another activation, which may exit too: WrongCallee).
    let taken = exits();
    let tdz: Vec<u32> = taken
        .iter()
        .filter(|&&(_, why)| why == ExitReason::Tdz)
        .map(|&(pc, _)| pc)
        .collect();
    assert_eq!(tdz.len(), 3, "{taken:?}");
    assert!(tdz.windows(2).all(|pair| pair[0] == pair[1]), "{taken:?}");
    assert_eq!(
        eval(&mut engine, "mk(false, false) + mk(false, true)"),
        "12"
    );
    names_hold(SOURCE);
}

#[test]
fn a_const_and_a_var_read_in_a_loop_whose_callee_assigns_the_var() {
    names_hold(
        "function mk() {
           const step = 3; var total = 0;
           function bump(k) { total = total + k; }
           return function run(n) {
             let sum = 0;
             for (let i = 0; i < n; i++) { bump(i); sum = sum + step * i + total; }
             return sum + total;
           };
         }
         const run = mk();
         let out = 0; for (let k = 0; k < 600; k++) out = (out + run(k % 17)) % 1000003; out",
    );
}

#[test]
fn static_stores_and_updates_of_closure_variables_hold_their_values() {
    names_hold(
        "function counter() {
           let n = 0; var s = '';
           return function tick(k) { n++; n += k; s = k % 3 === 0 ? s + 'x' : s; ++n; n--;
             return n + s.length; };
         }
         const tick = counter();
         let t = 0; for (let k = 0; k < 3000; k++) t = (t + tick(k)) % 1000003; t",
    );
}

#[test]
fn a_static_store_of_an_object_into_an_old_record_marks_its_card() {
    // `set` stores young objects into mk's activation, which collections promote: a minor
    // collection finds the edge through the record's card, which heap verification (on in the
    // engine's tests) checks at every collection.
    const SOURCE: &str = "function mk() { var held = null;
           return { set: function (o) { held = o; }, get: function () { return held; } }; }
         const box = mk();
         for (let k = 0; k < 1000; k++) box.set({v: k});";
    eager().set();
    let mut engine = jit_engine();
    compiled();
    eval(&mut engine, SOURCE);
    assert!(!compiled().is_empty(), "the tier compiled set");
    {
        let mut core = engine.interp.gc_heap.core.borrow_mut();
        core.set_stress_every(1);
        core.set_incremental(crate::gc_core::IncrementalMode::Off);
    }
    assert_eq!(
        eval(
            &mut engine,
            "let sum = 0; for (let k = 0; k < 400; k++) { box.set({v: k, pad: [k, k]});
               const junk = [{}, {}, {}]; sum += box.get().v + box.get().pad[1] + junk.length; }
             sum"
        ),
        (0..400).map(|k| 2 * k + 3).sum::<i64>().to_string()
    );
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
}

#[test]
fn a_class_whose_heritage_makes_and_calls_a_closure_that_reads_outer_variables() {
    // The class's outer record is re-parented after its heritage and computed keys ran
    // (eval.rs's eval_class_strict): a closure the heritage made and called, which reads the
    // enclosing function's variables, and one that escapes and runs after the class exists,
    // read what the VM reads (the plan's Q3: parent links are value facts within an
    // activation).
    names_hold(
        "function outer(k) {
           let base = k; var seen = 0; let later = null;
           class C extends (function () {
             const peek = () => base + seen;
             for (let i = 0; i < 40; i++) seen = seen + peek() % 7;
             later = () => base * 2 + seen;
             return Object;
           })() {
             [(seen = seen + 1, 'm' + (base % 3))]() { return base + seen; }
           }
           return later() + new C()['m' + (k % 3)]() + seen;
         }
         let t = 0; for (let k = 0; k < 300; k++) t = (t + outer(k)) % 1000003; t",
    );
}
