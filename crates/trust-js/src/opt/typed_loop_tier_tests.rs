//! P3 stage 3 step 0c end to end: loops without effects that read and write typed arrays whose
//! facts their entries hold or establish, with the VM's results in the gates' configurations,
//! detached, transferred and shrunk buffers between and before the loops, the entry check's
//! back-off, and an interruption at such a loop's poll.

use super::*;

/// Loops without effects over typed arrays of every kind: parameters read before the loop
/// (`before`) or first in it (`first`), an asm.js-style module's closure bindings (`mod`), a
/// global (`glob`), a loop that stores what it reads, two loops with a call between them that
/// detaches the buffer (`two`), and a driver that transfers, shrinks and grows buffers between
/// calls.
const PROGRAM: &str = "function before(t, n) { let s = t[0] || 0; \
     for (let i = 0; i < n; i++) s = s + t[i] * 0.5; return s; }\n\
     function first(t, n, v) { let s = 0; \
     for (let i = 0; i < n; i++) { t[i] = v + i; s = s + t[i]; } return s; }\n\
     const M = (function (g, buf) { var a = new g.Int8Array(buf); \
     var c = new g.Int32Array(buf); var d = new g.Float64Array(buf); \
     function mix(n) { n = n | 0; let s = c[0] | 0; \
       for (let i = 0; (i | 0) < (n | 0); i = (i + 1) | 0) { \
         s = (s + (a[i & 63] | 0) + (c[(i >> 2) & 15] | 0)) | 0; c[i & 15] = s; } \
       return s | 0; } \
     function fill(n, v) { for (let i = 0; i < n; i++) d[i & 7] = v * i; return d[3]; } \
     function swap() { d = new g.Float64Array(8); } \
     function calls(fs, n) { let s = c[0] | 0; for (let i = 0; i < n; i++) { \
       s = (s + (c[i & 15] | 0)) | 0; fs[i & 1](i); } return s | 0; } \
     function renew(i) { if (i === 5 && (c[0] | 0) === 77) c = new g.Int32Array(16).fill(3); } \
     function cut(i) { if (i === 3 && (c[0] | 0) === 88) c.buffer.transfer(); } \
     function stay(i) { return i; } \
     return { mix: mix, fill: fill, swap: swap, calls: calls, renew: renew, cut: cut, stay: stay, \
       set0: function (v) { c[0] = v; } }; \
     })(globalThis, new ArrayBuffer(64));\n\
     var G = new Uint16Array(16);\n\
     function glob(n) { let s = 0; for (let i = 0; i < n; i++) { G[i & 15] = G[i & 15] + i; \
     s = s + G[(i + 1) & 15]; } return s; }\n\
     function inner(t, f, n) { let s = t[0] || 0; for (let i = 0; i < n; i++) { \
     s = s + (t[i] || 0); f(t, i); } return s; }\n\
     function two(t, f, n) { let s = 0; for (let i = 0; i < n; i++) s = s + (t[i] || 0); \
     f(t); for (let i = 0; i < n; i++) s = s + (t[i] || 0); return s; }\n\
     const kinds = [Int8Array, Uint8Array, Uint8ClampedArray, Int16Array, Uint16Array, \
     Int32Array, Uint32Array, Float32Array, Float64Array];\n\
     const all = kinds.map(K => new K(16));\n\
     const rab = new ArrayBuffer(128, {maxByteLength: 128}); let r = new Float64Array(rab, 0, 16);\n\
     let gone = new Int32Array(16);\n\
     function keep() {} function drop(t) { if (t.length > 0 && t[0] > 1e9) t.buffer.transfer(); }\n\
     function late(t, i) { if (i === 3 && t[0] > 1e9) t.buffer.transfer(); }\n\
     const vals = [1.5, -0, NaN, 300.5, -2147483649, 1e10, 0.5, 2.5, Infinity, -1.25];\n\
     function h(x) { return x === x && x < 1e300 && x > -1e300 ? Math.round(x) % 1000003 : 7; }\n\
     let t = 0; for (let k = 0; k < 2400; k++) { const v = vals[k % vals.length]; \
     const x = all[k % 9]; \
     t = (t + h(first(x, 16, v)) + h(before(x, 16)) + h(first(r, 16, v)) + \
     M.mix(k % 40) + h(M.fill(k % 9, v)) + glob(k % 20)) % 1000003; \
     if (k === 1200) rab.resize(64); if (k === 1300) rab.resize(128); \
     if (k === 1400) r.buffer.transfer(); if (k === 1500) r = new Float64Array(16); \
     if (k === 1600) M.swap(); \
     gone[0] = k === 1700 ? 1.5e9 : k; t = (t + h(two(gone, k === 1700 ? drop : keep, 16))) % 1000003; \
     if (k === 1700) gone = new Int32Array(16); \
     gone.fill((k & 7) + 1); gone[0] = k === 1800 ? 1.5e9 : k; t = (t + h(inner(gone, k === 1800 ? late : keep, 8))) % 1000003; \
     if (k === 1800) gone = new Int32Array(16); \
     if (k === 1900) M.set0(77); if (k === 2001) M.set0(88); \
     t = (t + M.calls(k % 3 ? [M.stay, M.renew] : [M.renew, M.cut], 12)) % 1000003; }\n\
     String(t) + all.map(a => Array.from(a, x => Object.is(x, -0) ? 'z' : x).join()).join('|') \
     + ':' + G.join() + ':' + M.mix(40)";

#[test]
fn typed_loops_without_effects_have_the_vms_results() {
    let expected = run(Tier::Bytecode, PROGRAM);
    let mut compiles = 0;
    for (label, switches) in [
        (
            "on",
            TestSwitches {
                on: true,
                ..Default::default()
            },
        ),
        ("eager", eager()),
        (
            "forced exits 7",
            TestSwitches {
                exit_every: 7,
                ..eager()
            },
        ),
        (
            "forced exits 100",
            TestSwitches {
                exit_every: 100,
                ..eager()
            },
        ),
    ] {
        compiled();
        assert_eq!(with_switches(switches, PROGRAM), expected, "{label}");
        compiles += compiled().len();
    }
    assert!(compiles > 0, "the tier compiled the functions");
}
