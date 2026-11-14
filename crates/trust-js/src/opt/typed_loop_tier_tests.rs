//! P3 stage 3 step 0c end to end: loops without effects that read and write typed arrays whose
//! facts their entries hold or establish, with the VM's results in the gates' configurations,
//! detached, transferred and shrunk buffers between and before the loops, the entry check's
//! back-off, and an interruption at such a loop's poll.

use super::*;

/// Loops without effects over typed arrays of every kind: parameters read before the loop
/// (`before`) or first in it (`first`), an asm.js-style module's closure bindings (`mod`), a
/// global (`glob`), a loop that stores what it reads, two loops with a call between them that
/// detaches the buffer (`two`), a read and a store of one element of a writable or an immutable
/// buffer (`bump`), and a driver that transfers, shrinks and grows buffers between calls.
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
     function bump(t, k) { t[k & 3] = (t[k & 3] | 0) + 1; return t[k & 3] | 0; }\n\
     const IM = new Int32Array(new ArrayBuffer(16).transferToImmutable()); const BM = new Int32Array(4);\n\
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
     t = (t + bump(k % 7 ? BM : IM, k)) % 1000003; \
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

/// A loop without effects whose typed array its entry checks (P3 stage 3 step 0c), a detached
/// view `D` and a view `R` of a resizable buffer.
const ENTRY: &str = "function sum(t, n) { let s = 0; for (let i = 0; i < n; i++) s = s + t[i]; return s; }\n\
     const T = new Float64Array(16); for (let i = 0; i < 16; i++) T[i] = i * 0.5;\n\
     const D = new Float64Array(16); D.buffer.transfer();\n\
     const rab = new ArrayBuffer(128, {maxByteLength: 128}); const R = new Float64Array(rab, 0, 16);\n";

/// The checks `sum`'s graph makes at its loop's entry (HoistingFailed).
fn entry_checks(chunk: &Chunk) -> usize {
    graph_now(chunk)
        .nodes
        .iter()
        .filter(|node| node.reason == Some(ExitReason::HoistingFailed))
        .count()
}

#[test]
fn a_zero_trip_loop_entered_with_a_detached_buffer_exits_at_its_entry_and_backs_off() {
    // The blind speculation's cost: the loop never runs, so the old code never checked the view;
    // the entry's check exits (HoistingFailed) until the discard, whose flag's recompile checks
    // in the loop, which a zero-trip call never reaches: no exit after.
    let warm = format!("{ENTRY}let t = 0; for (let r = 0; r < 1000; r++) t = t + sum(T, 16); t");
    let chunk = converges(
        "sum",
        &warm,
        "let u = 0; for (let r = 0; r < 600; r++) u = u + sum(D, 0); u",
        "sum(D, 0) + sum(T, 16) + sum(R, 16)",
        ExitReason::HoistingFailed,
    );
    assert_eq!(entry_checks(&chunk), 0, "the recompile checks in the loop");
}

#[test]
fn a_zero_trip_loop_entered_with_a_shrunk_resizable_buffer_backs_off_the_same_way() {
    let warm = format!("{ENTRY}let t = 0; for (let r = 0; r < 1000; r++) t = t + sum(R, 16); t");
    let chunk = converges(
        "sum",
        &warm,
        "rab.resize(64); let u = 0; for (let r = 0; r < 600; r++) u = u + sum(R, 0); u",
        "rab.resize(128); sum(R, 16) + sum(T, 16) + sum(R, 0)",
        ExitReason::HoistingFailed,
    );
    assert_eq!(entry_checks(&chunk), 0, "the recompile checks in the loop");
}

#[test]
fn a_loop_entered_with_detached_buffers_converges_without_an_exit_cycle() {
    // Detached and live views in turn, reading in the loop: the entry's check exits first and
    // its flag keeps the checks in the loop, whose Detached exits then decline the accesses to
    // the generic one; a few discards, not one a call, and the VM's results throughout, with the
    // code discarded at every exit too.
    let source = format!(
        "{ENTRY}let t = 0; for (let r = 0; r < 1000; r++) t = t + sum(T, 16);\n\
         for (let r = 0; r < 1200; r++) t = t + (sum(r % 3 ? T : D, 8) || 0) + sum(R, r & 7); t"
    );
    let expected = run(Tier::Bytecode, &source);
    for (label, switches) in [
        ("eager", eager()),
        (
            "exit limit 1",
            TestSwitches {
                exit_limit: 1,
                ..eager()
            },
        ),
    ] {
        switches.set();
        let mut engine = jit_engine();
        assert_eq!(eval(&mut engine, &source), expected, "{label}");
        let chunk = tier_chunk(&mut engine, "sum").expect("sum's second stage");
        assert!(
            discards(&chunk) <= 3,
            "{label}: {} discards",
            discards(&chunk)
        );
    }
}

#[test]
fn an_interruption_at_the_poll_of_a_loop_whose_entry_checked_its_array_leaves_the_function() {
    // The entry checked the array and the loop reads its address through its poll's slow path,
    // in a try region: the host's interruption exits through the poll (its state has the
    // handler) and leaves the function, no throw the catch takes, and the code runs again with
    // the VM's results.
    let source = format!(
        "{ENTRY}function spin(t, n) {{ let s = 0; \
         try {{ for (let i = 0; i < n; i++) s = s + t[i & 15]; }} catch (e) {{ return -1; }} \
         return s; }}\n\
         let t = 0; for (let r = 0; r < 300; r++) t = t + spin(T, r % 40); t"
    );
    let expected = {
        let mut vm = Engine::new();
        vm.set_tier(Tier::Bytecode);
        vm.set_tier_threshold(0);
        [
            eval(&mut vm, &source),
            eval(&mut vm, "spin(T, 100) + spin(T, 3)"),
        ]
    };
    eager().set();
    let mut engine = jit_engine();
    compiled();
    assert_eq!(eval(&mut engine, &source), expected[0]);
    let chunk = tier_chunk(&mut engine, "spin").expect("spin's second stage");
    assert!(entry_checks(&chunk) > 0, "the entry checks the array");
    exits();
    let interrupt = engine.interrupt_handle();
    interrupt.set_deadline(Some(
        std::time::Instant::now() + std::time::Duration::from_millis(100),
    ));
    let outcome = engine.eval_interruptible("spin(T, 3e9)", false).unwrap();
    interrupt.set_deadline(None);
    assert!(
        matches!(outcome, crate::ExecutionOutcome::Interrupted { .. }),
        "the interruption"
    );
    assert!(
        exits().iter().any(|&(_, why)| why == ExitReason::Throw),
        "the poll's exit"
    );
    assert_eq!(eval(&mut engine, "spin(T, 100) + spin(T, 3)"), expected[1]);
}
