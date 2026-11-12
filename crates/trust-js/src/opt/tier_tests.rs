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

/// The exit sites keyed on a recompile and on its function's first stage, or on a first stage
/// that carries the tier's code (P3 stage 2, step 14).
fn sites(chunk: &Chunk) -> Vec<(u32, u8, bool)> {
    let mut sites = sites_on(chunk);
    let first = profile::first_stage(chunk);
    if !std::ptr::eq(first, chunk) {
        sites.extend(sites_on(first));
    }
    sites
}

/// Programs whose functions run past the second stage's checkpoints (100, 500, 2,100 and 8,500
/// invocations, then every 500).
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
    // Immediate operands (P3 stage 2 step 7): int32 additions, subtractions, masks, shifts and
    // comparisons with constants at the encodings' bounds and past them, on either side, and
    // f64 constants. The parameters are made int32s first, as asm.js code makes them, so the
    // arithmetic is int32; they reach the int32 range's ends late, so additions with an
    // immediate operand overflow and exit, and the recompiles take f64 arithmetic there.
    "function imm(a, b) { a = a | 0; b = b | 0; \
     let r = ((a + 4095) ^ (a - 4096) ^ (0xfff000 + a) ^ (a - -1) ^ (b - 0x1000000) ^ \
     (a + -2147483648) ^ (4097 - b)) | 0; \
     r = r ^ (a & 0x3fff) ^ (b | 0xfffffff) ^ (a ^ 0x12345) ^ (0x55555555 & b) ^ (a & -8) ^ \
     (b | 0); \
     r = r ^ (a << 14) ^ (b >> 31) ^ (a >>> 28) ^ (b << 33) ^ (a >> 0) ^ (1 << (b & 31)) ^ \
     ((a >>> 0) & 1) ^ (b >>> 32); \
     if (a < 4096) r = r + 1 | 0; if (-4096 >= b) r = r - 3 | 0; if (a == 0xfff000) r = ~r; \
     if (4095 > b) r = r ^ 7; if (b != -1) r = r ^ 0x100; if (a <= -2147483648) r = r ^ 9; \
     return r; }\n\
     function fimm(x) { return x * 0.5 + 1.5 - x / 31 + (x < 0.125 ? 0.1 : -2.5) * 0.1; }\n\
     const vals = [0, 1, -1, 4095, 4096, -4096, 0x3fff, 0xfff000, 123456789, -987654321];\n\
     let t = 0, u = 0; for (let k = 0; k < 3000; k++) { \
     const a = k < 2000 ? vals[k % vals.length] : k % 2 ? 2147483647 : -2147483648; \
     const b = vals[(k * 7) % vals.length]; \
     t = (t + imm(a, b)) | 0; u += fimm(k * 0.25 - 100); }\n\
     t + ':' + u",
    // Bitwise operations on oddballs: a holey array's push/shift queue, whose hole's undefined
    // reaches `| 0` (the calls benchmark's array-shift-queue-holey), and undefined, null and
    // Booleans at every bitwise operator, a String reaching them late (their truncation exits).
    "function queue(n) { const a = []; for (let k = 0; k < 40; k++) a.push(k); delete a[20]; \
     let s = 0; for (let k = 0; k < n; k++) { a.push(k); s = (s + (a.shift() | 0)) | 0; } \
     return s; }\n\
     function bits(v, k) { return (v | 0) ^ ~v ^ (k >>> v) ^ (v << 2) ^ (v & k) ^ (v >> 1); }\n\
     const vals = [undefined, null, true, false, 3, -2.5, 4294967296.5, NaN];\n\
     let t = 0; for (let k = 0; k < 3000; k++) { t = (t + queue(k % 60)) | 0; \
     t = (t + bits(k < 2800 ? vals[k % vals.length] : String(k % 7), k)) | 0; } t",
    // Effects at several operand depths (the small callees' study): x20 moved from the depth
    // the emission knows, so each effect's operands must land where its template reads them:
    // calls of a callee the tier never splices (its finally) under other operands, getters and
    // `arguments` reads, both arms of a branch that merge, loops whose header reloads x20 (a long
    // one whose call leaves x20 two words above the header's depth each iteration); and f64
    // results boxed with a NaN canonicalized, a negated NaN's sign set among them.
    "function g(a, b) { try { return a * 3 - b; } finally { } }\n\
     const o = { v: 2, get w() { return this.v + 0.5; } };\n\
     function eff(k, arr) { let s = k + g(k, 1) * (2 + g(1, k)) - arguments.length; \
     if (k & 1) { s = s - o.w + g(s, o.w); } else { s = g(o.w, s) - o.w + s; } \
     for (let i = 0; i < 3; i++) s = s + g(i, arr[i]) * arguments[0]; \
     return s + -arr[3]; }\n\
     function long(n) { let s = 0; for (let i = 0; i < n; i++) s = s + g(i, 1); return s; }\n\
     let t = 0, nans = 0; for (let k = 0; k < 3000; k++) { \
     const r = eff(k % 17, [k, 2, -1, k % 50 === 0 ? NaN : k % 7]); \
     if (r !== r) nans++; else t += r; if (k % 100 === 0) t += long(4000); } t + ':' + nans",
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

/// The verifier's switch (stage 2, step 10): a debug build verifies every graph the tier builds,
/// a release build only under TRUST_JS_OPT_VERIFY=1, and the library's tests always.
#[test]
fn a_release_build_verifies_only_under_its_switch() {
    assert!(super::verifies_in(true, false), "a debug build verifies");
    assert!(super::verifies_in(true, true));
    assert!(
        !super::verifies_in(false, false),
        "a release build without TRUST_JS_OPT_VERIFY does not"
    );
    assert!(
        super::verifies_in(false, true),
        "a release build under TRUST_JS_OPT_VERIFY=1 does"
    );
    assert!(super::verifies(), "the library's tests verify");
}

/// In the library's tests the tier verifies the graph of each function it compiles.
#[test]
fn the_tier_verifies_each_graph_it_compiles_in_the_tests() {
    let expected = run(Tier::Bytecode, PROGRAMS[0]);
    super::TEST_VERIFIED.with(|verified| verified.set(0));
    compiled();
    assert_eq!(with_switches(eager(), PROGRAMS[0]), expected);
    let compiles = compiled().len();
    let verified = super::TEST_VERIFIED.with(std::cell::Cell::get);
    assert!(compiles > 0, "the tier compiled sum");
    assert!(
        verified >= compiles,
        "{verified} graphs verified for {compiles} compiles"
    );
}

/// The chunk the tier compiled for the global function `name`: its second stage, where its plan
/// splices callees (or a stale chain's recompile made one), else its first stage once the tier's
/// code was installed in it (a plan that splices nothing, P3 stage 2, step 14).
fn tier_chunk(engine: &mut Engine, name: &str) -> Option<std::rc::Rc<crate::bytecode::Chunk>> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok()?;
    let object = value.as_obj()?;
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        return None;
    };
    let runtime = user.func.runtime();
    runtime.code2.get().and_then(Option::clone).or_else(|| {
        runtime
            .code
            .get()
            .and_then(Option::clone)
            .filter(|first| super::in_place(first))
    })
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
        let chunk = tier_chunk(&mut engine, name).unwrap_or_else(|| panic!("{name}: code2"));
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
    assert!(tier_chunk(&mut engine, "sum").is_none());
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
    let chunk = tier_chunk(&mut engine, "sum").expect("the tier's code");
    assert!(
        compiled().contains(&(std::rc::Rc::as_ptr(&chunk) as usize)),
        "the tier compiled it"
    );
    // Nothing spliced: the tier compiled the first stage itself and installed its code there
    // (P3 stage 2, step 14), with no second chunk and no record map.
    let first = {
        let env = engine.interp.global_env;
        let value = engine.interp.get_var("sum", &env).ok().unwrap();
        let object = value.as_obj().unwrap();
        let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
            panic!("an author function")
        };
        assert!(user.func.runtime().code2.get().is_none(), "no second chunk");
        user.func
            .runtime()
            .code
            .get()
            .and_then(Option::clone)
            .unwrap()
    };
    assert!(std::rc::Rc::ptr_eq(&chunk, &first));
    assert!(chunk.feedback_vector().record_map().is_none());
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
    let chunk = tier_chunk(&mut engine, "total").expect("the tier compiled total");
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
    let chunk = tier_chunk(&mut engine, "add").expect("the tier compiled add");
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

#[test]
fn a_sites_refills_past_its_capacity_leave_the_fingerprint_and_its_transitions_change_it() {
    // V8's megamorphic rule (stage 2, step 11): a named site's fifth receiver shape makes it
    // generic, which changes the fingerprint, and the shapes after it refill its ways without
    // changing it; a call site's evictions of live ways make it megamorphic, which changes it,
    // and its ways' refills after an epoch move do not.
    use crate::feedback::{CallState, SiteView};
    TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    let mut engine = jit_engine();
    let mut source = String::from(
        "function getx(o) { return o.x; }\nfunction callf(f) { return f(); }\nvar fs = [];",
    );
    for k in 0..16 {
        source.push_str(&format!("\nfs.push(function () {{ return {k}; }});"));
    }
    // Each function's chunk is made at its first call, the callees' before callf's site sees
    // them (a site fills a way for a callee with native code).
    source.push_str("\nfor (var k = 0; k < fs.length; k++) fs[k]();");
    eval(&mut engine, &source);
    eval(&mut engine, "getx({x: 0}); callf(fs[0]);");
    let getx = first_stage(&mut engine, "getx");
    let callf = first_stage(&mut engine, "callf");
    let print = |chunk: &Chunk| super::snapshot::fingerprint(chunk);
    let named = |chunk: &Chunk| {
        let snapshot = super::snapshot::Snapshot::take(chunk);
        (0..snapshot.len())
            .find_map(|pc| snapshot.named(pc))
            .expect("getx's named site")
    };
    let call = |chunk: &Chunk| {
        let snapshot = super::snapshot::Snapshot::take(chunk);
        (0..snapshot.len())
            .find_map(|pc| match snapshot.site(pc) {
                SiteView::Call(call) => Some(call),
                _ => None,
            })
            .expect("callf's call site")
    };
    // The named site: four shapes fill its four ways, each a change.
    let shapes = [
        "{x: 0}",
        "{a: 0, x: 0}",
        "{b: 0, x: 0}",
        "{c: 0, x: 0}",
        "{d: 0, x: 0}",
        "{e: 0, x: 0}",
        "{f: 0, x: 0}",
    ];
    let mut last = print(&getx);
    for shape in &shapes[1..4] {
        eval(&mut engine, &format!("getx({shape})"));
        assert_ne!(print(&getx), last, "{shape}: a new way");
        last = print(&getx);
    }
    assert!(!named(&getx).generic);
    eval(&mut engine, &format!("getx({})", shapes[4]));
    assert!(named(&getx).generic, "the fifth shape: generic");
    assert_ne!(print(&getx), last, "the transition changes it");
    last = print(&getx);
    let ways = named(&getx).ways;
    for shape in &shapes[5..] {
        eval(&mut engine, &format!("getx({shape})"));
    }
    assert_ne!(named(&getx).ways, ways, "the generic site's ways refilled");
    assert_eq!(print(&getx), last, "its refills change nothing");
    // The call site: one callee, then four ways, then four evictions of live ways in one epoch.
    assert_eq!(call(&callf).state, CallState::Monomorphic);
    let mut last = print(&callf);
    eval(&mut engine, "callf(fs[1])");
    assert_eq!(call(&callf).state, CallState::Polymorphic);
    assert_ne!(print(&callf), last, "its second");
    let polymorphic = print(&callf);
    // The epoch is the process's: another test's thread may move it between two evictions (a
    // discard, a recompile), so a round of evictions repeats until one falls in one epoch.
    for _ in 0..100 {
        if call(&callf).state == CallState::Megamorphic {
            break;
        }
        eval(&mut engine, "for (var k = 2; k < 10; k++) callf(fs[k]);");
    }
    assert_eq!(call(&callf).state, CallState::Megamorphic);
    assert_ne!(print(&callf), polymorphic, "the transition changes it");
    last = print(&callf);
    // A moved epoch: the ways of the old one are refilled as new callees arrive.
    let targets = call(&callf).ways;
    crate::bytecode::invalidate_call_caches();
    eval(&mut engine, "for (var k = 10; k < 16; k++) callf(fs[k]);");
    assert_eq!(call(&callf).state, CallState::Megamorphic);
    let refilled = call(&callf).ways;
    assert!(
        refilled.iter().flatten().any(|target| !targets
            .iter()
            .flatten()
            .any(|old| old.callee == target.callee)),
        "the megamorphic site's ways refilled"
    );
    assert_eq!(print(&callf), last, "its refills change nothing");
}

#[test]
fn a_chunk_over_the_size_limit_is_refused_before_its_snapshot_and_keeps_its_second_stage() {
    // V8's size limit (stage 2, step 11): big, which splices inc, has more operations than the
    // tier takes; its second stage is built and published, and the tier refuses it first thing,
    // so that the template JIT's code runs it, as after any refusal.
    let mut source =
        String::from("function inc(x) { return x + 1; }\nfunction big(x) { x = inc(x);");
    for k in 0..4000 {
        source.push_str(&format!(" x = (x + {}) | 0;", k % 7));
    }
    source.push_str(" return x; }");
    let program =
        format!("{source}\nvar t = 0; for (var k = 0; k < 150; k++) t = (t + big(k)) | 0; t");
    let expected = [
        run(Tier::Bytecode, &program),
        run(Tier::Bytecode, &format!("{source}\nbig(1)")),
    ];
    eager().set();
    let mut engine = jit_engine();
    compiled();
    super::TEST_REFUSED.with(|refused| refused.borrow_mut().clear());
    assert_eq!(eval(&mut engine, &program), expected[0]);
    let chunk = tier_chunk(&mut engine, "big").expect("big's second stage");
    assert!(
        chunk.jit_ops().len() > super::MAX_OPTIMIZED_OPS,
        "{} operations",
        chunk.jit_ops().len()
    );
    let refused = super::TEST_REFUSED.with(|refused| refused.borrow().clone());
    assert!(
        refused.contains(&(address(&chunk), "over the size limit")),
        "{refused:?}"
    );
    assert!(!compiled().contains(&address(&chunk)));
    let code = chunk.jit.get().flatten().expect("the template JIT's code");
    assert!(!is_tier_code(&code));
    code.referenced.set(0);
    assert_eq!(eval(&mut engine, "big(1)"), expected[1]);
    assert!(code.referenced.get() != 0, "the second stage runs");
}

#[test]
fn a_chunk_over_the_size_limit_with_nothing_to_splice_is_never_recompiled() {
    // V8's size limit before the recompile (stage 2, step 11): huge splices nothing, so it has no
    // second stage to keep, and the tier stops at its checkpoint before the bytecode recompile,
    // which only the tier would take: the refusal is the first stage's (a refused recompile's
    // would be another chunk's), the function keeps its first stage, its checkpoints disarmed.
    let mut source = String::from("function huge(x) {");
    for k in 0..4000 {
        source.push_str(&format!(" x = (x + {}) | 0;", k % 7));
    }
    source.push_str(" return x; }");
    let program =
        format!("{source}\nvar t = 0; for (var k = 0; k < 150; k++) t = (t + huge(k)) | 0; t");
    let expected = run(Tier::Bytecode, &program);
    eager().set();
    let mut engine = jit_engine();
    compiled();
    super::TEST_REFUSED.with(|refused| refused.borrow_mut().clear());
    assert_eq!(eval(&mut engine, &program), expected);
    let first = first_stage(&mut engine, "huge");
    assert!(
        first.jit_ops().len() > super::MAX_OPTIMIZED_OPS,
        "{} operations",
        first.jit_ops().len()
    );
    let refused = super::TEST_REFUSED.with(|refused| refused.borrow().clone());
    assert_eq!(
        refused,
        vec![(address(&first), "over the size limit")],
        "refused before its recompile"
    );
    assert!(tier_chunk(&mut engine, "huge").is_none(), "no second stage");
    assert!(compiled().is_empty());
    assert_eq!(first.inline_retry_at.get(), 0, "its checkpoints disarmed");
    assert!(
        first.jit.get().flatten().is_some(),
        "its first stage's code"
    );
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

/// The checkpoints (chunk, number) at which `first` waited since the last clear.
fn waits_of(first: &Rc<Chunk>) -> Vec<u8> {
    super::TEST_WAITS.with(|waits| {
        waits
            .borrow()
            .iter()
            .filter(|&&(chunk, _)| chunk == address(first))
            .map(|&(_, checkpoint)| checkpoint)
            .collect()
    })
}

#[test]
fn feedback_that_changes_until_the_fourth_checkpoint_compiles_at_the_first_quiet_window() {
    // getx's receivers show a shape new before each of the first four checkpoints (100, 500,
    // 2,100 and 8,500 invocations): the tier waits at each, the fourth too, where it compiled
    // whatever the fingerprint said before step 11, and compiles at the next checkpoint, the
    // first 500-invocation window in which the feedback did not change (9,000), settled.
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
             var k = 0, t = 0; for (; k < 8950; k++) t += getx(receiver(k)); t"
        ),
        "40046775"
    );
    let first = first_stage(&mut engine, "getx");
    assert_eq!(
        waits_of(&first),
        vec![1, 2, 3, 4],
        "a wait at the fourth too"
    );
    assert!(
        tier_chunk(&mut engine, "getx").is_none(),
        "nothing forced at the fourth checkpoint"
    );
    assert_eq!(
        eval(
            &mut engine,
            "for (; k < 9050; k++) t += getx(receiver(k)); t"
        ),
        "40946725"
    );
    let chunk = tier_chunk(&mut engine, "getx").expect("the first quiet window compiles");
    assert!(compiled().contains(&address(&chunk)));
    assert_eq!(waits_of(&first), vec![1, 2, 3, 4], "settled at the fifth");
    let code = chunk.jit.get().flatten().expect("resident");
    assert!(is_tier_code(&code));
}

/// A function whose sites run one after another, `phase` choosing the one that runs: each new
/// site's first way changes the fingerprint.
const PHASED: &str = "function g(o, p) { if (p === 0) return o.a; if (p === 1) return o.b; \
     if (p === 2) return o.c; if (p === 3) return o.d; if (p === 4) return o.e; \
     if (p === 5) return o.f; if (p === 6) return o.g; if (p === 7) return o.h; \
     if (p === 8) return o.i; if (p === 9) return o.j; return 0; }\n\
     var o = {a: 1, b: 2, c: 3, d: 4, e: 5, f: 6, g: 7, h: 8, i: 9, j: 10};\n\
     var starts = [0, 300, 1200, 5000, 8700, 9200, 9700, 10200, 10700];\n\
     function phase(k) { var p = 0; while (p + 1 < starts.length && k >= starts[p + 1]) p++; \
     return p; }\n\
     var k = 0, t = 0;";

#[test]
fn feedback_that_changes_in_every_window_never_compiles() {
    // A new site runs between every two checkpoints, past the fourth too (8,500, then every 500
    // invocations): the tier waits at each of the first nine, and nothing forces a compile; once
    // the sites stop changing (from 10,700), the next window (11,500) compiles, settled.
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        eval(&mut vm, PHASED);
        [
            eval(&mut vm, "for (; k < 11450; k++) t += g(o, phase(k)); t"),
            eval(&mut vm, "for (; k < 11550; k++) t += g(o, phase(k)); t"),
        ]
    };
    TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    let mut engine = jit_engine();
    compiled();
    super::TEST_WAITS.with(|waits| waits.borrow_mut().clear());
    eval(&mut engine, PHASED);
    assert_eq!(
        eval(&mut engine, "for (; k < 11450; k++) t += g(o, phase(k)); t"),
        expected[0]
    );
    let first = first_stage(&mut engine, "g");
    assert_eq!(waits_of(&first), (1..=9).collect::<Vec<u8>>());
    assert!(
        tier_chunk(&mut engine, "g").is_none(),
        "no compile while the feedback changes"
    );
    assert_eq!(
        eval(&mut engine, "for (; k < 11550; k++) t += g(o, phase(k)); t"),
        expected[1]
    );
    let chunk = tier_chunk(&mut engine, "g").expect("the quiet window compiles");
    assert!(compiled().contains(&address(&chunk)));
    assert_eq!(waits_of(&first), (1..=9).collect::<Vec<u8>>());
}

#[test]
fn a_loop_free_function_gets_the_tiers_code_at_its_500th_entry() {
    // Its first checkpoint (100 invocations) waits, the settled-feedback rule having no earlier
    // fingerprint, and its second (500) compiles: each entry counts K, each distance is K times
    // as long, so a loop-free function keeps its schedule (step 11).
    TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    let mut engine = jit_engine();
    eval(
        &mut engine,
        "function sq(x) { return x * x; }\nvar k = 0, t = 0;",
    );
    compiled();
    assert_eq!(
        eval(&mut engine, "for (; k < 499; k++) t += sq(k); t"),
        "41292749"
    );
    let first = first_stage(&mut engine, "sq");
    assert_eq!(
        first.jit_runs.get(),
        499 * super::ENTRY_WEIGHT,
        "K an entry"
    );
    assert!(tier_chunk(&mut engine, "sq").is_none(), "499 entries");
    assert_eq!(
        eval(&mut engine, "for (; k < 500; k++) t += sq(k); t"),
        "41541750"
    );
    let chunk = tier_chunk(&mut engine, "sq").expect("compiled at the 500th entry");
    assert!(compiled().contains(&address(&chunk)));
    let code = chunk.jit.get().flatten().expect("resident");
    assert!(is_tier_code(&code));
    code.referenced.set(0);
    assert_eq!(eval(&mut engine, "sq(7)"), "49");
    assert!(
        code.referenced.get() != 0,
        "the 501st entry runs the tier's code"
    );
}

#[test]
fn with_the_tier_off_an_entry_counts_one_and_the_second_stage_comes_at_the_100th() {
    // P2's checkpoints, unchanged by step 11: one an entry, the first at 100 entries, where the
    // template JIT's second stage splices inc.
    TestSwitches::default().set();
    let mut engine = jit_engine();
    eval(
        &mut engine,
        "function inc(x) { return x + 1; }\n\
         function f(x) { return inc(x) * 2; }\nvar k = 0, t = 0;",
    );
    assert_eq!(
        eval(&mut engine, "for (; k < 99; k++) t += f(k); t"),
        "9900"
    );
    let first = first_stage(&mut engine, "f");
    assert_eq!(first.jit_runs.get(), 99, "one an entry");
    assert!(tier_chunk(&mut engine, "f").is_none(), "99 entries");
    assert_eq!(
        eval(&mut engine, "for (; k < 100; k++) t += f(k); t"),
        "10100"
    );
    assert!(
        tier_chunk(&mut engine, "f").is_some(),
        "the second stage at the 100th entry"
    );
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
    let graph = super::build::build(chunk, &snapshot)
        .expect("the tier builds it")
        .graph;
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
        let chunk = tier_chunk(&mut engine, name).expect("the tier's chunk");
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
        let chunk = tier_chunk(&mut engine, "first").expect("the tier's chunk");
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
    let chunk = tier_chunk(&mut engine, name).unwrap_or_else(|| panic!("{name}: code2"));
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
    super::build::build(chunk, &super::snapshot::Snapshot::take(chunk))
        .expect("built")
        .graph
}

/// `live_at`'s index (stage 2, step 10) agrees with a scan of every range at each node of the
/// tier's graphs of the programs' functions: loops, effects, calls, element and named accesses.
#[test]
fn live_at_agrees_with_a_scan_on_the_tiers_graphs() {
    let mut graphs = 0;
    let mut live = 0;
    for program in PROGRAMS {
        eager().set();
        let mut engine = jit_engine();
        eval(&mut engine, program);
        let names: Vec<&str> = program
            .split("function ")
            .skip(1)
            .filter_map(|rest| rest.split('(').next())
            .filter(|name| !name.is_empty() && name.chars().all(|c| c.is_alphanumeric()))
            .collect();
        for name in names {
            let Some(chunk) = tier_chunk(&mut engine, name) else {
                continue;
            };
            let Ok(built) = super::build::build(&chunk, &super::snapshot::Snapshot::take(&chunk))
            else {
                continue;
            };
            let alloc = super::regalloc::allocate(&built.graph, &super::regalloc::ARM64)
                .expect("the tier's graph allocates");
            live += super::regalloc::tests::assert_live_at_agrees(&built.graph, &alloc);
            graphs += 1;
        }
    }
    assert!(graphs >= 10, "{graphs} graphs");
    assert!(live > 0);
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
fn oddballs_into_a_bitwise_operation_converge_on_their_truncation() {
    // Compiled on Numbers, `v | 0` checks a Number and exits for the undefined and null that
    // reach it later (a holey array's hole once the code ran); the VM records them, and the
    // recompile truncates them in line, as V8 reoptimizes on kNumberOrOddball, where the flag
    // alone declined the generic operation.
    let chunk = converges(
        "low",
        "function low(v) { return (v | 0) + 1; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = (t + low(k)) | 0; t",
        "let s = 0; for (let k = 0; k < 600; k++) s = (s + low(k % 3 ? undefined : null)) | 0; s",
        "low(undefined) + low(true) * 10 + low(7.5) * 100",
        ExitReason::NotNumber,
    );
    let graph = graph_now(&chunk);
    use super::ir::NodeOp;
    assert!(graph.declines.is_empty(), "{:?}", graph.declines);
    assert!(graph
        .nodes
        .iter()
        .any(|node| node.op == NodeOp::TruncateNumberOrOddball));
    assert!(!graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, NodeOp::Effect { .. })));
}

#[test]
fn a_symbol_into_an_oddball_truncation_converges_on_the_generic_operation() {
    // Compiled on Numbers and oddballs, `v | 0` truncates them in line and exits for a Symbol,
    // whose TypeError leaves the record as it was: its truncation's flag (TRUNCATION_EXIT)
    // declines the generic operation, which throws there without an exit.
    let chunk = converges(
        "low",
        "function low(v) { return (v | 0) + 1; }\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = (t + low(k % 2 ? k : undefined)) | 0; t",
        "let s = 0; for (let k = 0; k < 600; k++) { try { low(Symbol()); } catch (e) { s++; } } s",
        "low(undefined) + low(5) * 10 + (() => { try { return low(Symbol()); } \
         catch (e) { return e.name; } })()",
        ExitReason::NotNumber,
    );
    let graph = graph_now(&chunk);
    use super::ir::NodeOp;
    let or = chunk
        .jit_ops()
        .iter()
        .position(|op| *op == crate::bytecode::Op::BitOr)
        .expect("the BitOr");
    assert_eq!(
        graph
            .declines
            .iter()
            .map(|decline| (decline.pc as usize, decline.reason))
            .collect::<Vec<_>>(),
        [(or, ExitReason::NotNumber)]
    );
    assert!(!graph
        .nodes
        .iter()
        .any(|node| node.op == NodeOp::TruncateNumberOrOddball));
    assert!(graph
        .nodes
        .iter()
        .any(|node| matches!(node.op, NodeOp::Effect { pc, .. } if pc as usize == or)));
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

/// A late second closure (P3 stage 2, step 15's point 5): a site that met one closure when the
/// second stage was planned splices it under the tier's rule; fresh closures afterwards fail its
/// guard to the exit limit. No plan follows: the reoptimization compiles the same second stage
/// (step 3's departure 1), and the discard's WrongCallee flag sends the closures to the guard's
/// mismatch edge, the generic call in place, which meets them by code.
#[test]
fn a_late_second_closure_converges_on_the_guards_mismatch_edge() {
    let chunk = converges(
        "apply1",
        "function apply1(f, x) { return f(x); }\n\
         function make(k) { return (v) => v + k; }\n\
         const one = make(1);\n\
         let t = 0; for (let k = 0; k < 1000; k++) t += apply1(one, k); t",
        "let u = 0; for (let k = 0; k < 600; k++) u += apply1(make(k), k); u",
        "apply1(make(3), 4) * 100 + apply1(make(5), 6)",
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
    // The splice's generic call met the closures by code, through the second stage's own site.
    let generic = chunk
        .jit_ops()
        .iter()
        .find_map(|op| match *op {
            crate::bytecode::Op::Call(1, c) => Some(c),
            _ => None,
        })
        .expect("the splice's generic call");
    assert!(chunk.test_call_site(generic).met_another_closure());
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
    let chunk = tier_chunk(&mut engine, "f").expect("f's second stage");
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
    let chunk = tier_chunk(&mut engine, "risky").expect("risky's second stage");
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
    let chunk = tier_chunk(&mut engine, "f").expect("f's second stage");
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
    let chunk = tier_chunk(&mut engine, "add").expect("add's second stage");
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
    let code2 = tier_chunk(&mut engine, "staged").expect("staged's second stage");
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
    // R stops where the warm-up's longest distance, its fourth checkpoint's 6,400 invocations
    // << R (K native runs each), still fits its word (step 11), and so does every window after it.
    assert_eq!(profile.discards(), profile::MAX_DISCARDS);
    assert_eq!(profile::MAX_DISCARDS, 13);
    assert_eq!(super::exit_limit(&chunk), 100 << 13);
    let k = u64::from(super::ENTRY_WEIGHT);
    assert!((6400 * k) << profile::MAX_DISCARDS <= u64::from(u32::MAX));
    assert!((6400 * k) << (profile::MAX_DISCARDS + 1) > u64::from(u32::MAX));
    for checkpoint in 0..8 {
        assert!(
            super::warm_up_gap(profile::MAX_DISCARDS, checkpoint).is_some(),
            "checkpoint {checkpoint}"
        );
    }
    assert_eq!(
        super::warm_up_gap(profile::MAX_DISCARDS, 3),
        Some((6400 << 13) * super::ENTRY_WEIGHT)
    );
    assert_eq!(
        super::warm_up_gap(profile::MAX_DISCARDS, 4),
        Some((500 << 13) * super::ENTRY_WEIGHT)
    );
    TestSwitches {
        exit_limit: 1,
        ..eager()
    }
    .set();
    assert_eq!(super::exit_limit(&chunk), 1 << 13);
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

/// `name`'s code now, and whether it is the tier's.
fn tier_code(engine: &mut Engine, name: &str) -> Option<Rc<crate::jit::JitCode>> {
    let chunk = tier_chunk(engine, name)?;
    chunk.jit.get().flatten().filter(|code| is_tier_code(code))
}

#[test]
fn a_global_read_in_a_hot_loop_survives_additions_and_exits_once_a_deletion_moves_the_epoch() {
    // Additions append slots: no exit. A deletion closes its gap and moves the later slots: the
    // epoch moved, the code exits (WrongName) until its discard, and the recompile reads the new
    // slots under the new epoch, without exits.
    const WARM: &str = "globalThis.a = 1; globalThis.b = 2; globalThis.c = 3;
         function sum(n) { var s = 0; for (var i = 0; i < n; i++) s = s + b + c; return s; }
         let t = 0; for (let k = 0; k < 1000; k++) t += sum(5); t";
    const ADD: &str = "for (let k = 0; k < 50; k++) { globalThis['x' + k] = k; t += sum(5); } t";
    const DELETE: &str = "delete globalThis.a; for (let k = 0; k < 900; k++) t += sum(5); t";
    const AFTER: &str = "t + sum(7) + b * 100 + c";
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        [WARM, ADD, DELETE, AFTER].map(|source| eval(&mut vm, source))
    };
    eager().set();
    let mut engine = jit_engine();
    compiled();
    exits();
    discarded();
    assert_eq!(eval(&mut engine, WARM), expected[0]);
    let optimized = tier_code(&mut engine, "sum").expect("the tier compiled sum");
    assert!(exits().is_empty());
    assert_eq!(eval(&mut engine, ADD), expected[1], "additions");
    assert!(exits().is_empty(), "additions append: no exit");
    assert_eq!(eval(&mut engine, DELETE), expected[2], "a deletion");
    let taken = exits();
    assert_eq!(taken.len(), super::EXIT_LIMIT as usize, "{taken:?}");
    assert!(
        taken.iter().all(|&(_, why)| why == ExitReason::WrongName),
        "{taken:?}"
    );
    assert_eq!(discarded(), vec![optimized.mem_ptr() as usize]);
    let now = tier_code(&mut engine, "sum").expect("the recompile is resident");
    assert!(!Rc::ptr_eq(&now, &optimized));
    now.referenced.set(0);
    assert_eq!(eval(&mut engine, AFTER), expected[3]);
    assert!(exits().is_empty(), "the recompile reads the new slots");
    assert!(now.referenced.get() != 0);
    drop((optimized, now));
    super::release_retired(&mut engine.interp);
}

#[test]
fn global_reads_hold_through_redefinitions_shadowing_lexicals_and_late_sites() {
    // A property redefined as an accessor and back, a later script's lexical that shadows a
    // configurable global, and a site that first runs after the compile.
    names_hold(
        "globalThis.p = 1; globalThis.q = 2;
         function read(late) { var r = p + q; if (late) r = r + lateGlobal; return r; }
         let t = 0; for (let k = 0; k < 600; k++) t += read(false);
         Object.defineProperty(globalThis, 'p', { get() { return 40; }, configurable: true });
         for (let k = 0; k < 300; k++) t += read(false);
         Object.defineProperty(globalThis, 'p', { value: 5, writable: true, configurable: true });
         for (let k = 0; k < 300; k++) t += read(false);
         globalThis.lateGlobal = 7;
         for (let k = 0; k < 300; k++) t += read(true);
         t",
    );
    for switches in [
        eager(),
        TestSwitches {
            exit_limit: 1,
            ..eager()
        },
    ] {
        switches.set();
        let mut engine = jit_engine();
        assert_eq!(
            eval(
                &mut engine,
                "globalThis.s = 3; function shadowed() { return s + 1; }
                 let u = 0; for (let k = 0; k < 600; k++) u += shadowed(); u"
            ),
            "2400"
        );
        // A later script's lexical shadows the configurable property: the read sees it.
        assert_eq!(eval(&mut engine, "let s = 10; shadowed()"), "11");
        assert_eq!(
            eval(
                &mut engine,
                "let v = 0; for (let k = 0; k < 600; k++) v += shadowed(); v"
            ),
            "6600"
        );
    }
}

#[test]
fn a_function_of_another_realm_reads_its_own_globals() {
    use crate::realm_inline_guard_tests::{eval_string, in_realm};
    for switches in [
        eager(),
        TestSwitches {
            exit_every: 7,
            ..eager()
        },
    ] {
        switches.set();
        let mut engine = jit_engine();
        let it = &mut *engine.interp;
        let other = it.create_realm();
        eval_string(
            it,
            "var own = 1; function reader() { return own; }
             globalThis.mine = 0; for (var k = 0; k < 600; k++) mine += reader();",
        );
        in_realm(it, &other, |it| {
            eval_string(
                it,
                "var own = 100; function reader() { return own; }
                 var total = 0; for (var k = 0; k < 600; k++) total += reader();",
            )
        });
        let theirs = in_realm(it, &other, |it| eval_string(it, "total"));
        assert_eq!(theirs, "60000");
        assert_eq!(eval_string(it, "mine"), "600");
        // Each realm's reader called from the other: its own realm's global.
        crate::realm_inline_guard_tests::set_global(it, "otherGlobal", other);
        assert_eq!(
            eval_string(
                it,
                "var mixed = 0; for (var k = 0; k < 600; k++) mixed += reader() + \
                 otherGlobal.reader(); mixed"
            ),
            "60600"
        );
    }
}

#[test]
fn typed_arrays_bound_to_a_global_and_to_a_module_variable() {
    // Mandreel's heap views are globals, zlib's an asm.js module's own variables: the views'
    // reads and the element accesses on them (step 2).
    names_hold(
        "var HEAP32 = new Int32Array(256); var HEAPF64 = new Float64Array(64);
         function put(i, v) { HEAP32[i & 255] = v | 0; HEAPF64[i & 63] = v * 0.5; }
         function get(i) { return HEAP32[i & 255] + HEAPF64[(i + 1) & 63]; }
         function module() {
           var heap = new Int32Array(128); var u8 = new Uint8Array(heap.buffer);
           function store(i, v) { heap[i & 127] = v; u8[i & 511] = v & 255; }
           function load(i) { return heap[i & 127] + u8[(i * 3) & 511]; }
           return { store: store, load: load };
         }
         const m = module();
         let t = 0;
         for (let k = 0; k < 3000; k++) { put(k, k * 3); m.store(k, k ^ 0x55);
           t = (t + get(k) + m.load(k + 1)) % 1000003; }
         t",
    );
}

#[test]
fn script_level_lexicals_read_while_scripts_assign_and_declare_them() {
    // Assignments keep the record's generation (no exit); a later script's declarations move it,
    // and the code exits WrongName until its recompile takes the new proof.
    const WARM: &str = "let lex = 1; const k = 3;
         function readLex() { return lex * 2 + k; }
         let t = 0; for (let n = 0; n < 1000; n++) t += readLex(); t";
    const ASSIGN: &str = "for (let n = 0; n < 50; n++) { lex = n; t += readLex(); } t";
    const DECLARE: &str =
        "let another = 4; for (let n = 0; n < 900; n++) t += readLex() + another; t";
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        [WARM, ASSIGN, DECLARE].map(|source| eval(&mut vm, source))
    };
    eager().set();
    let mut engine = jit_engine();
    compiled();
    exits();
    discarded();
    assert_eq!(eval(&mut engine, WARM), expected[0]);
    let optimized = tier_code(&mut engine, "readLex").expect("the tier compiled readLex");
    assert_eq!(eval(&mut engine, ASSIGN), expected[1], "assignments");
    assert!(exits().is_empty(), "an assignment moves no generation");
    assert_eq!(
        eval(&mut engine, DECLARE),
        expected[2],
        "a later declaration"
    );
    let taken = exits();
    assert!(
        !taken.is_empty() && taken.iter().all(|&(_, why)| why == ExitReason::WrongName),
        "{taken:?}"
    );
    assert_eq!(discarded(), vec![optimized.mem_ptr() as usize]);
    let now = tier_code(&mut engine, "readLex").expect("the recompile is resident");
    now.referenced.set(0);
    // `lex` is 49 since the assignments: 49 * 2 + 3 + 4.
    assert_eq!(eval(&mut engine, "readLex() + another"), "105");
    assert!(exits().is_empty());
    assert!(now.referenced.get() != 0);
    drop((optimized, now));
    super::release_retired(&mut engine.interp);
    names_hold(
        "let a = 1; let b = 'x';
         function f(n) { return a + n + b.length; }
         let t = 0; for (let k = 0; k < 600; k++) { t += f(k); if (k % 100 === 0) a = a + 1; }
         t",
    );
}

#[test]
fn a_global_site_that_first_runs_after_the_compile_converges_on_its_read() {
    // Its cache was empty at the compile: its block exits NeverRan when it starts to run, until
    // the discard; the recompile finds the cache filled and reads the global in line.
    let chunk = converges(
        "late",
        "globalThis.lateG = 5;
         function late(c) { if (c) { return lateG + 1; } return 0; }
         let t = 0; for (let k = 0; k < 1000; k++) t += late(false); t",
        "let u = 0; for (let k = 0; k < 600; k++) u += late(true); u",
        "late(true) * 10 + late(false)",
        ExitReason::NeverRan,
    );
    let snapshot = super::snapshot::Snapshot::take(&chunk);
    let read = (0..chunk.jit_ops().len())
        .find(|&pc| matches!(chunk.jit_ops()[pc], crate::bytecode::Op::LoadName(..)))
        .expect("lateG's read");
    assert_eq!(snapshot.ran(read), super::snapshot::Ran::Yes);
}

#[test]
fn a_global_whose_proof_keeps_failing_converges_on_the_effect() {
    // A global another function deletes and adds again every third read: each recompile finds
    // its proof's byte the discarded code's (the effect) or not (the check again); either way
    // the results are the VM's, and the discards back off.
    names_hold(
        "globalThis.flip = 1;
         function readFlip() { return typeof flip === 'number' ? flip : -1; }
         function churn(k) { delete globalThis.flip; globalThis.flip = k; }
         let t = 0;
         for (let k = 0; k < 4000; k++) { t += readFlip(); if (k % 3 === 0) churn(k % 5); }
         t",
    );
}

// ---- untagged phis (stage 2, step 6) ------------------------------------------------------------

#[test]
fn a_catch_reads_a_variable_its_try_overwrites_before_any_other_read() {
    // `tag` is dead on the try body's normal path after `g(k)` (overwritten, never read), but
    // live at the call for the catch, which the VM runs on the frame the effect's throw leaves:
    // the handler keeps it live (the slot liveness's try rule), so the code flushes it there.
    names_hold(
        "function g(k) { if (k % 7 === 3) throw new Error('x' + k); return k; }
         function f(n) { let s = 0;
           for (let k = 0; k < n; k++) { let tag = k * 2 + 1;
             try { s += g(k); tag = -1; } catch (e) { s += tag * 1000; } }
           return s; }
         let t = 0; for (let r = 0; r < 300; r++) t = (t + f(20)) % 1000003; t",
    );
}

/// The representations of the phis the tier builds now for the global function `name`'s second
/// stage, once `source` ran with the tier eager: (I32 phis, F64 phis, tagged phis).
fn phi_reps(source: &str, name: &str) -> (usize, usize, usize) {
    use super::ir::Rep;
    eager().set();
    let mut engine = jit_engine();
    eval(&mut engine, source);
    let chunk = tier_chunk(&mut engine, name).unwrap_or_else(|| panic!("{name}: code2"));
    let graph = graph_now(&chunk);
    let reps: Vec<Rep> = graph
        .blocks
        .iter()
        .flat_map(|block| block.phis.iter().map(|&phi| graph.node(phi).rep))
        .collect();
    let count = |rep: Rep| reps.iter().filter(|&&r| r == rep).count();
    (count(Rep::I32), count(Rep::F64), count(Rep::Tagged))
}

#[test]
fn an_f64_phi_carries_minus_zero_and_nan_into_an_index_and_a_boxing() {
    // `x` is an F64 phi from -0: -0 indexes element 0, NaN and a fraction fail the index's check
    // at the use (NotInt32), and the boxings keep -0 and canonicalize NaN.
    const SOURCE: &str = "function walk(a, n, step) { let s = ''; let x = -0;
           for (let k = 0; k < n; k++) { const v = a[x]; s += v === undefined ? 'u' : String(v);
             x = x + step; }
           return s + (Object.is(x, -0) ? 'z' : String(x)); }
         const arr = [7, 8, 9, 10, 11]; let out = '';
         for (let r = 0; r < 300; r++) out = walk(arr, 4, 1);
         out + walk(arr, 4, -0) + walk(arr, 3, NaN) + walk(arr, 3, 0.5) + walk(arr, 2, 1) +
           walk(arr, 3, -0.5)";
    // `x` is the F64 phi, `k` an I32 one, and the strings' phis are tagged.
    assert_eq!(phi_reps(SOURCE, "walk"), (1, 1, 3));
    names_hold(SOURCE);
}

#[test]
fn a_swap_of_two_locals_moves_a_cycle_on_the_back_edge() {
    // `a` and `b` (F64 phis) and `p` and `q` (I32 phis) take each other's values at the back
    // edge: a cycle of moves in each register class.
    const SOURCE: &str = "function swap(n) { let a = 0.5, b = 1.5, p = 1, q = 2;
           for (let k = 0; k < n; k++) { const t = a; a = b; b = t; const u = p; p = q; q = u; }
           return a * 10 + b + p * 100 + q * 1000; }
         let s = 0; for (let r = 0; r < 300; r++) s += swap(r % 7); s";
    // `a`, `b` and `k` F64, `p` and `q` I32.
    assert_eq!(phi_reps(SOURCE, "swap"), (2, 3, 0));
    names_hold(SOURCE);
}

#[test]
fn exits_with_untagged_phis_in_registers_and_spill_slots_rebuild_their_numbers() {
    // Twenty-four int32 phis and ten f64 phis in one loop take every register a class has and
    // spill the rest; forced exits at every seventh check rebuild each from its home.
    let ints: Vec<String> = (0..24).map(|k| format!("a{k}")).collect();
    let floats: Vec<String> = (0..10).map(|k| format!("f{k}")).collect();
    let declare = ints
        .iter()
        .enumerate()
        .map(|(k, name)| format!("{name} = {k}"))
        .chain(
            floats
                .iter()
                .enumerate()
                .map(|(k, name)| format!("{name} = {k}.5")),
        )
        .collect::<Vec<_>>()
        .join(", ");
    let step = ints
        .iter()
        .enumerate()
        .map(|(k, name)| {
            format!(
                "{name} = ({name} + {} + k) | 0;",
                ints[(k + 1) % ints.len()]
            )
        })
        .chain(floats.iter().enumerate().map(|(k, name)| {
            format!(
                "{name} = {name} * 0.5 + {} * 0.25;",
                floats[(k + 1) % floats.len()]
            )
        }))
        .collect::<Vec<_>>()
        .join(" ");
    let all = ints
        .iter()
        .chain(&floats)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "function many(n) {{ let {declare}; for (let k = 0; k < n; k++) {{ {step} }} \
         return [{all}].join(); }}\n\
         let s = ''; for (let r = 0; r < 300; r++) s = many(r % 9); s + many(40)"
    );
    // The twenty-four int32s, the ten f64s and the counter.
    assert_eq!(phi_reps(&source, "many"), (24, 11, 0));
    names_hold(&source);
}

#[test]
fn untagged_phis_live_across_collections_beside_words() {
    // `s` and `i` live across each `push`, which allocates: with a collection at every
    // allocation their homes beside the array's word survive each one.
    const SOURCE: &str = "function keep(n) { let s = 0.5, i = 0; const objs = [];
           for (let k = 0; k < n; k++) { objs.push({v: k, pad: [k]}); s = s + objs[k].v * 0.5;
             i = (i + 3) | 0; }
           return s + i + objs.length + objs[n - 1].pad[0]; }
         let t = 0; for (let r = 0; r < 300; r++) t += keep(r % 9 + 1); t";
    let expected = run(Tier::Bytecode, SOURCE);
    eager().set();
    let mut engine = jit_engine();
    compiled();
    assert_eq!(eval(&mut engine, SOURCE), expected);
    assert!(!compiled().is_empty(), "the tier compiled keep");
    {
        let mut core = engine.interp.gc_heap.core.borrow_mut();
        core.set_stress_every(1);
        core.set_incremental(crate::gc_core::IncrementalMode::Off);
    }
    let again = "let u = 0; for (let r = 0; r < 200; r++) u += keep(r % 13 + 1); u";
    let mut vm = Engine::new();
    vm.set_tier(Tier::Bytecode);
    vm.set_tier_threshold(0);
    eval(&mut vm, SOURCE);
    let expected = eval(&mut vm, again);
    assert_eq!(eval(&mut engine, again), expected);
    engine.interp.gc_heap.core.borrow_mut().set_stress_every(0);
}

/// A loop whose phi `i` enters from the parameter `start`: an I32 phi (its back edge's `| 0`),
/// its entry checked a Number and an int32 at the loop's entry (stage 2 step 6).
const ENTERED: &str = "function entered(start, n) { let i = start;
       for (let k = 0; k < n; k++) i = (i + 1) | 0; return i; }\n";

/// The representation of the phi of `entered`'s `i` in `graph`.
fn entered_phi(graph: &super::ir::Graph) -> super::ir::Rep {
    let header = graph
        .blocks
        .iter()
        .find(|block| block.loop_header)
        .expect("the loop");
    let poll = graph.node(header.body[0]);
    let state = graph.state(poll.state.expect("a poll's state"));
    // `i` is the function's third slot (start, n, i, k).
    match state.slots[2] {
        super::ir::SlotSource::Value(value) if header.phis.contains(&value) => {
            graph.node(value).rep
        }
        other => panic!("i's phi: {other:?}"),
    }
}

#[test]
fn a_loop_entered_with_a_double_where_int32s_were_seen_converges_on_an_f64_phi() {
    // The entry's CheckInt32 exits at the loop header (PhiNotInt32), whose flag the recompile
    // declines to an F64 phi, checked a Number only.
    let source = format!(
        "{ENTERED}let t = 0; for (let r = 0; r < 1000; r++) t = (t + entered(r, 3)) | 0; t"
    );
    let chunk = converges(
        "entered",
        &source,
        "let u = 0; for (let r = 0; r < 600; r++) u += entered(r + 0.5, 3); u",
        "entered(1.5, 4) + entered(7, 2) + entered(-0.5, 1)",
        ExitReason::PhiNotInt32,
    );
    let graph = graph_now(&chunk);
    assert_eq!(entered_phi(&graph), super::ir::Rep::F64);
    assert!(graph
        .declines
        .iter()
        .any(|decline| decline.reason == ExitReason::PhiNotInt32));
}

#[test]
fn a_loop_entered_with_a_string_where_int32s_were_seen_converges_on_a_tagged_phi() {
    let source = format!(
        "{ENTERED}let t = 0; for (let r = 0; r < 1000; r++) t = (t + entered(r, 3)) | 0; t"
    );
    let chunk = converges(
        "entered",
        &source,
        "let u = ''; for (let r = 0; r < 600; r++) u = entered('s' + r, 3) + u.slice(0, 20); u",
        "entered('t', 4) + ':' + entered(7, 2) + entered(2.5, 1)",
        ExitReason::PhiNotNumber,
    );
    // The VM's runs of the exits recorded the strings at `i + 1` and `| 0`, whose generic
    // operations leave the back edge a word: the feedback alone keeps the phi tagged, as step
    // 3's NotNumber converges (the flag's own decline: build_tests'
    // `a_flag_at_a_loop_header_declines_its_entry_speculation`).
    let graph = graph_now(&chunk);
    assert_eq!(entered_phi(&graph), super::ir::Rep::Tagged);
}

#[test]
fn entry_speculations_back_off_without_an_exit_cycle() {
    // Int32s enter the loop until it is compiled, then int32s, doubles and strings in turn: the
    // results hold in the gates' configurations, and with every code discarded at its first exit
    // the header's two flags (and the VM's records of the exits' runs) settle it after a few
    // discards, not one per call.
    let source = format!(
        "{ENTERED}let t = 0; for (let r = 0; r < 1000; r++) t = (t + entered(r, 3)) | 0;\n\
         let s = String(t); for (let r = 0; r < 600; r++) {{ const k = r % 3; \
         s = String(entered(k === 0 ? r : k === 1 ? r + 0.25 : 'x' + r, 3)).slice(0, 4) + \
         s.slice(0, 30); }} s"
    );
    names_hold(&source);
    let expected = run(Tier::Bytecode, &source);
    TestSwitches {
        exit_limit: 1,
        ..eager()
    }
    .set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, &source), expected);
    let chunk = tier_chunk(&mut engine, "entered").expect("entered's second stage");
    assert!(discards(&chunk) <= 3, "{} discards", discards(&chunk));
}

#[test]
fn a_counter_passing_2_31_exits_on_overflow_and_converges_on_an_f64_phi() {
    // The VM's runs record `i++`'s int32s, so the tier adds int32s with an overflow exit and
    // `i` is an I32 phi; a count past 2^31 - 1 exits there (Overflow), the VM finishes with
    // 2^31 and more, and after the discard the recompile adds f64s into an F64 phi, exact to
    // 2^53 as the VM's.
    const SOURCE: &str = "function count(start, n) { let i = start;
           for (let k = 0; k < n; k++) i++; return i; }\n\
         for (let r = 0; r < 50; r++) count(r, 4);";
    const WARM: &str = "let t = 0; for (let r = 0; r < 600; r++) t += count(r, 4); t";
    const PAST: &str = "let u = 0; for (let r = 0; r < 600; r++) u += count(2147483600, 100); u";
    const AFTER: &str = "[count(2147483647, 3), count(9007199254740989, 5), count(-5, 3)].join()";
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        eval(&mut vm, SOURCE);
        [
            eval(&mut vm, WARM),
            eval(&mut vm, PAST),
            eval(&mut vm, AFTER),
        ]
    };
    eager().set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, SOURCE);
    engine.set_tier(Tier::Jit);
    compiled();
    exits();
    discarded();
    assert_eq!(eval(&mut engine, WARM), expected[0]);
    assert!(exits().is_empty(), "the speculation held");
    let chunk = tier_chunk(&mut engine, "count").expect("count's second stage");
    assert!(compiled().contains(&address(&chunk)));
    let int = graph_now(&chunk);
    assert!(int
        .nodes
        .iter()
        .any(|node| node.op == super::ir::NodeOp::I32Add
            && node.reason == Some(ExitReason::Overflow)));
    assert_eq!(
        int.blocks
            .iter()
            .flat_map(|block| block.phis.iter())
            .filter(|&&phi| int.node(phi).rep == super::ir::Rep::I32)
            .count(),
        2,
        "i and k"
    );
    assert_eq!(eval(&mut engine, PAST), expected[1]);
    let taken = exits();
    assert!(!taken.is_empty() && taken.iter().all(|&(_, why)| why == ExitReason::Overflow));
    assert_eq!(discards(&chunk), 1);
    assert_eq!(eval(&mut engine, AFTER), expected[2]);
    assert!(exits().is_empty(), "no exit after the recompile");
    let graph = graph_now(&chunk);
    let header = graph
        .blocks
        .iter()
        .find(|block| block.loop_header)
        .expect("the loop");
    let state = graph.state(graph.node(header.body[0]).state.expect("a state"));
    let i = match state.slots[2] {
        super::ir::SlotSource::Value(value) => value,
        other => panic!("i: {other:?}"),
    };
    assert_eq!(graph.node(i).rep, super::ir::Rep::F64);
}

#[test]
fn a_poll_interrupted_with_untagged_phis_in_its_state_leaves_and_the_code_runs_again() {
    // `s` (F64), `i` and `j` (I32) are the loop's phis at its header's poll, inside a try: the
    // host's interruption there exits through the poll's recipe (its state's phis read from the
    // registers the slow path restored), and the interruption, no throw, leaves the function.
    const SOURCE: &str = "function spin(n) { let s = 0.5, i = 0, j = 0;
           try { while (i < n) { s = s * 0.5 + 1; i = (i + 1) | 0; j = (j + 3) | 0; } }
           catch (e) { return -1; }
           return s + i + j; }\n\
         let t = 0; for (let r = 0; r < 300; r++) t += spin(r % 20); t";
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        [eval(&mut vm, SOURCE), eval(&mut vm, "spin(10) + spin(3)")]
    };
    assert_eq!(phi_reps(SOURCE, "spin"), (2, 1, 0));
    eager().set();
    let mut engine = jit_engine();
    compiled();
    assert_eq!(eval(&mut engine, SOURCE), expected[0]);
    assert!(!compiled().is_empty(), "the tier compiled spin");
    exits();
    let interrupt = engine.interrupt_handle();
    interrupt.set_deadline(Some(
        std::time::Instant::now() + std::time::Duration::from_millis(100),
    ));
    let outcome = engine.eval_interruptible("spin(3e9)", false).unwrap();
    interrupt.set_deadline(None);
    assert!(
        matches!(outcome, crate::ExecutionOutcome::Interrupted { .. }),
        "the interruption is no throw the catch takes"
    );
    assert!(
        exits().iter().any(|&(_, why)| why == ExitReason::Throw),
        "the poll's exit"
    );
    assert_eq!(eval(&mut engine, "spin(10) + spin(3)"), expected[1]);
}

/// Richards' Scheduler.prototype.schedule in small (stage 2, step 8): a function entered once a
/// run, with the run's loop inside.
const ENTERED_ONCE: &str = "function step(x) { return (x * 7 + 1) % 1009; }\n\
     function drive(n) { let x = 0; for (let i = 0; i < n; i++) x = step(x); return x; }";

#[test]
fn a_function_entered_a_few_times_with_a_long_loop_gets_the_tiers_code_by_its_third_entry() {
    // Its loop's iterations count toward the second stage's checkpoints, each its loop's weight
    // (a share of an entry's K, its loop spanning most of the function): the first entry's loop
    // reaches the first checkpoint (100 invocations' worth), which the second entry takes and
    // waits at (the settled-feedback rule has no earlier fingerprint there), and its loop the
    // second (500), which the third entry takes and compiles at; the fourth runs the tier's
    // code. With the counting off, the four entries count four entries' K: no checkpoint, the
    // first stage kept.
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        eval(&mut vm, ENTERED_ONCE);
        [eval(&mut vm, "drive(5000)"), eval(&mut vm, "drive(10)")]
    };
    for counting in [true, false] {
        TestSwitches {
            on: true,
            no_loop_count: !counting,
            ..Default::default()
        }
        .set();
        let mut engine = jit_engine();
        eval(&mut engine, ENTERED_ONCE);
        compiled();
        super::TEST_WAITS.with(|waits| waits.borrow_mut().clear());
        let mut compiled_at = None;
        for entry in 1..=4 {
            assert_eq!(
                eval(&mut engine, "drive(5000)"),
                expected[0],
                "entry {entry}"
            );
            if compiled_at.is_none() && tier_chunk(&mut engine, "drive").is_some() {
                compiled_at = Some(entry);
            }
        }
        let first = first_stage(&mut engine, "drive");
        let waits: Vec<u8> = super::TEST_WAITS.with(|waits| {
            waits
                .borrow()
                .iter()
                .filter(|&&(chunk, _)| chunk == address(&first))
                .map(|&(_, checkpoint)| checkpoint)
                .collect()
        });
        if !counting {
            assert_eq!(compiled_at, None, "counting off: no checkpoint");
            assert_eq!(
                first.jit_runs.get(),
                4 * super::ENTRY_WEIGHT,
                "the entries alone"
            );
            assert!(waits.is_empty());
            continue;
        }
        assert_eq!(compiled_at, Some(3), "compiled at the third entry");
        assert_eq!(waits, vec![1], "waited at the first checkpoint");
        let chunk = tier_chunk(&mut engine, "drive").expect("drive's second stage");
        assert!(
            compiled().contains(&address(&chunk)),
            "the tier compiled it"
        );
        let code = chunk.jit.get().flatten().expect("resident");
        assert!(is_tier_code(&code));
        code.referenced.set(0);
        assert_eq!(eval(&mut engine, "drive(10)"), expected[1]);
        assert!(code.referenced.get() != 0, "the tier's code runs");
    }
}

/// A long function with a short loop (stage 2, step 11): 48 statements ahead of a loop of a few
/// operations.
fn long_with_short_loop() -> String {
    let mut source = String::from("function longf(a, n) { var s = a | 0;");
    for k in 0..48 {
        source.push_str(&format!(" s = (s * 3 + {k}) | 0;"));
    }
    source.push_str(" for (var i = 0; i < n; i++) s = (s + i) | 0; return s; }");
    source
}

/// The one loop's weight in `chunk`, max(1, round(K x B / L)) (see `jit::loop_count_tests`).
fn the_loop_weight(chunk: &Chunk) -> u32 {
    use crate::bytecode::Op;
    let ops = chunk.jit_ops();
    let spans: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| match *op {
            Op::Jump(target)
            | Op::AbruptJump(target, _)
            | Op::JumpIfFalse(target)
            | Op::JumpIfFalsePeek(target)
            | Op::JumpIfTruePeek(target)
            | Op::JumpIfNotNullishPeek(target)
                if target as usize <= pc =>
            {
                Some(pc - target as usize + 1)
            }
            _ => None,
        })
        .collect();
    assert_eq!(spans.len(), 1, "one back edge");
    let (k, len) = (u64::from(super::ENTRY_WEIGHT), ops.len() as u64);
    ((k * spans[0] as u64 + len / 2) / len).max(1) as u32
}

/// The entry at which a function each of whose entries runs `n` iterations of its one loop, of
/// weight `w`, reaches its second checkpoint, where settled feedback compiles: an entry adds K
/// and takes a due checkpoint, the next one 400 invocations' K later; the loop adds its
/// iterations' weights up to one below the word.
fn second_checkpoint_entry(n: u64, w: u64) -> u32 {
    let k = u64::from(super::ENTRY_WEIGHT);
    let (mut runs, mut word, mut taken) = (0u64, 100 * k, 0);
    for entry in 1u32.. {
        runs += k;
        if runs >= word {
            taken += 1;
            if taken == 2 {
                return entry;
            }
            word = runs + 400 * k;
        }
        runs = (runs + n * w).min(word - 1);
    }
    unreachable!()
}

#[test]
fn a_long_function_with_a_short_loop_compiles_after_500_invocations_worth_of_its_loop() {
    // Its loop's iterations each count the loop's small share of an entry, so the function
    // reaches its second checkpoint after about 500 x L / B iterations, as V8's budget of 500
    // invocations' worth of bytecode would, where counting each iteration as an entry (step 8)
    // compiled it at its third entry.
    let n = 2000u64;
    let source = long_with_short_loop();
    let expected: Vec<String> = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        eval(&mut vm, &source);
        (1..=12)
            .map(|entry| eval(&mut vm, &format!("longf({entry}, {n})")))
            .collect()
    };
    TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    let mut engine = jit_engine();
    eval(&mut engine, &source);
    compiled();
    let mut compiled_at = None;
    let mut w = 0;
    for entry in 1..=12usize {
        assert_eq!(
            eval(&mut engine, &format!("longf({entry}, {n})")),
            expected[entry - 1],
            "entry {entry}"
        );
        if entry == 1 {
            w = the_loop_weight(&first_stage(&mut engine, "longf"));
        }
        if compiled_at.is_none() && tier_chunk(&mut engine, "longf").is_some() {
            compiled_at = Some(entry);
        }
    }
    let first = first_stage(&mut engine, "longf");
    let len = first.jit_ops().len();
    eprintln!("longf: {len} ops, loop weight {w}, compiled at entry {compiled_at:?}");
    assert!(
        len > 200 && (1..=4).contains(&w),
        "a short loop in a long function"
    );
    let at = second_checkpoint_entry(n, u64::from(w));
    assert!(at > 3, "{at}");
    let compiled_at = compiled_at.expect("compiled within 12 entries") as u32;
    assert!(
        (at - 1..=at + 1).contains(&compiled_at),
        "compiled at entry {compiled_at}, the counting's {at}"
    );
    // About 500 invocations' worth of the loop, 500 x K / w iterations, within the two entries'
    // loops the checkpoints' clamps leave uncounted.
    let iterations = u64::from(compiled_at - 1) * n;
    let budget = 500 * u64::from(super::ENTRY_WEIGHT) / u64::from(w);
    assert!(
        iterations + 2 * n >= budget && iterations <= budget + 2 * n,
        "{iterations} iterations before the compile, 500 invocations' worth {budget}"
    );
    let chunk = tier_chunk(&mut engine, "longf").expect("its second stage");
    assert!(compiled().contains(&address(&chunk)));
}

#[test]
fn a_discards_warm_up_reaches_its_recompile_through_a_loops_iterations() {
    // The discarded code's fallback, the template JIT's, counts its loop's iterations toward the
    // warm-up's first checkpoint (200 native runs at R = 1): the entry after a long loop takes
    // it and recompiles, where entries alone, the counting off, need 200.
    const SOURCE: &str = "function f(a, n) { let s = 0; for (let i = 0; i < n; i++) s = s + a; \
                          return s; }";
    for counting in [true, false] {
        TestSwitches {
            no_loop_count: !counting,
            ..eager()
        }
        .set();
        let mut engine = jit_engine();
        eval(&mut engine, SOURCE);
        compiled();
        exits();
        discarded();
        assert_eq!(
            eval(
                &mut engine,
                "let t = 0; for (let k = 0; k < 1000; k++) t += f(1, 3); t"
            ),
            "3000"
        );
        let chunk = tier_chunk(&mut engine, "f").expect("f's second stage");
        assert!(compiled().contains(&address(&chunk)));
        let optimized = chunk.jit.get().flatten().expect("resident");
        assert!(is_tier_code(&optimized));
        // A string fails its Number check: each call exits once, and the 100th discards it.
        assert_eq!(
            eval(
                &mut engine,
                "let u = ''; for (let k = 0; k < 100; k++) u = f('x', 1); u"
            ),
            "0x"
        );
        assert_eq!(exits().len(), super::EXIT_LIMIT as usize);
        assert_eq!(discarded(), vec![optimized.mem_ptr() as usize]);
        assert!(super::reoptimizing(&chunk), "the warm-up is armed");
        assert!(compiled().is_empty());
        assert_eq!(eval(&mut engine, "f(2, 10000)"), "20000");
        assert_eq!(eval(&mut engine, "f(3, 10000)"), "30000");
        if counting {
            assert!(
                compiled().contains(&address(&chunk)),
                "the recompile, at the second entry"
            );
            assert!(!super::reoptimizing(&chunk), "the warm-up ended");
            let now = chunk.jit.get().flatten().expect("resident");
            assert!(is_tier_code(&now), "the recompile is resident");
            now.referenced.set(0);
            assert_eq!(eval(&mut engine, "f('a', 2) + f(1, 2)"), "0aa2");
            assert!(now.referenced.get() != 0, "the recompile runs");
            assert!(exits().is_empty(), "declining the add, it exits no more");
        } else {
            assert!(compiled().is_empty(), "no recompile after two entries");
            assert!(super::reoptimizing(&chunk), "the warm-up waits");
        }
        drop(optimized);
        super::release_retired(&mut engine.interp);
    }
}
