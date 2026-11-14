//! P3 stage 3 step R1 end to end: values live across a loop's calls, spilled, and read in an
//! inner loop without calls, which reads them from loop caches loaded at its header; the VM's
//! results with the tier on, eager, with exits forced every 7 and every 100 (an exit inside the
//! inner loop rebuilds the frame from the slots), with graph inlining (a leaf's loop inlined
//! into a loop of its caller), a throw from the outer loop's call caught by a handler outside
//! the loops, and collections inside the call while the cached values are objects.

use super::*;

/// The caches the allocations on this thread made since the last call.
fn caches() -> usize {
    crate::opt::regalloc::TEST_CACHES.with(|made| made.replace(0))
}

/// Fourteen int32s made from the arguments live across the outer loop's call (a polymorphic
/// one: a generic node) and read by the inner loop each iteration, then read after the loops.
const MIX: &str = "function st(s, k) { return (s * 31 + k) & 0xfffff; }\n\
     function st2(s, k) { return (s ^ (k * 7)) & 0xfffff; }\n\
     const fs = [st, st2];\n\
     function mix(n, a, b, c, d) { \
       let a1 = (a * 3) & 0xfff, a2 = (a1 ^ b) & 0xfff, a3 = (a2 + c) & 0xfff, \
         a4 = (a3 - d) & 0xfff, a5 = (a4 * 7) & 0xfff, a6 = (a5 ^ a) & 0xfff, \
         a7 = (a6 + b) & 0xfff, a8 = (a7 >> 1) & 0xfff, a9 = (a8 + 11) & 0xfff, \
         a10 = (a9 ^ c) & 0xfff, a11 = (a10 + d) & 0xfff, a12 = (a11 * 5) & 0xfff, \
         a13 = (a12 ^ a1) & 0xfff, a14 = (a13 + a2) & 0xfff; \
       let s = 0; \
       for (let i = 0; i < n; i++) { \
         s = fs[i & 1](s, i); \
         for (let j = 0; j < 6; j++) { \
           s = (s + ((a1 & j) | a2) + (a3 ^ j) + (a4 >> (j & 3)) + (a5 & a6) + (a7 | j) \
             + (a8 ^ a9) + (a10 + j) + (a11 & a12) + (a13 ^ a14)) & 0xfffff; } } \
       return s + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14; }\n\
     let t = 0; for (let k = 0; k < 400; k++) t = (t + mix(12, k, k * 3, 7, k ^ 5)) % 1000003;\n\
     t + ':' + mix(5, 1e9, 3, 2, 1) + ':' + mix(3, 2.5, 1, 1, 1) + ':' + mix(4, 'x', 1, 1, 1)";

/// MIX's shape, the outer loop's call throwing at its ninth turn where s is odd, caught by a
/// handler around both loops that reads s and the int32s.
const GUARD: &str = "function st(s, k) { return (s * 31 + k) & 0xfffff; }\n\
     function boom(s, k) { if (k === 9 && (s & 1) === 1) throw new RangeError('boom ' + s); \
       return (s + k) & 0xfffff; }\n\
     function guard(n, a, b) { \
       let a1 = (a * 3) & 0xfff, a2 = (a1 ^ b) & 0xfff, a3 = (a2 + 5) & 0xfff, \
         a4 = (a3 - 2) & 0xfff, a5 = (a4 * 7) & 0xfff, a6 = (a5 ^ a) & 0xfff, \
         a7 = (a6 + b) & 0xfff, a8 = (a7 >> 1) & 0xfff, a9 = (a8 + 11) & 0xfff, \
         a10 = (a9 ^ 3) & 0xfff, a11 = (a10 + 9) & 0xfff, a12 = (a11 * 5) & 0xfff; \
       let s = 0; \
       try { \
         for (let i = 0; i < n; i++) { \
           s = (i & 1 ? boom : st)(s, i); \
           for (let j = 0; j < 5; j++) { \
             s = (s + (a1 ^ j) + (a2 & j) + (a3 | j) + (a4 >> (j & 3)) + (a5 & a6) \
               + (a7 ^ a8) + (a9 + j) + (a10 & a11) + a12) & 0xfffff; } } \
       } catch (e) { return e.message + '/' + (s + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 \
         + a10 + a11 + a12); } \
       return s + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12; }\n\
     let t = ''; for (let k = 0; k < 300; k++) { const r = guard(12, k, k * 5); \
       if (k % 37 === 0) t += r + ';'; }\n\
     t + guard(20, 3, 4)";

/// Twelve objects made from the argument, live across the outer loop's call, which allocates
/// (collections run in it), and read by the inner loop each iteration; an array read after.
const OBJECTS: &str = "function mk(s, k) { const o = { v: s + k, w: [s, k, s ^ k] }; \
       return (o.v + o.w[2]) & 0xffff; }\n\
     function objs(n, p) { \
       const o1 = {x: p}, o2 = {x: p + 1}, o3 = {x: p + 2}, o4 = {x: p + 3}, o5 = {x: p + 4}, \
         o6 = {x: p + 5}, o7 = {x: p + 6}, o8 = {x: p + 7}, o9 = {x: p + 8}, o10 = {x: p + 9}, \
         o11 = {x: p + 10}, o12 = {x: p + 11}; \
       const arr = [p, p + 1, p + 2, p + 3]; \
       let s = 0; \
       for (let i = 0; i < n; i++) { \
         s = mk(s, i); \
         for (let j = 0; j < 4; j++) { \
           s = (s + o1.x + o2.x + o3.x + o4.x + o5.x + o6.x + o7.x + o8.x + o9.x + o10.x \
             + o11.x + o12.x + j) & 0xffff; } } \
       return s + o1.x + o6.x + o12.x + arr[3]; }\n\
     let t = 0; for (let k = 0; k < 200; k++) t = (t + objs(10, k)) % 1000003;\n\
     t + ':' + objs(6, 1.5)";

/// A leaf whose loop reads its arguments, inlined at graph into a loop of a caller that keeps
/// a dozen int32s across its own call.
const INLINED: &str = "function st(s, k) { return (s * 31 + k) & 0xfffff; }\n\
     function st2(s, k) { return (s ^ (k * 7)) & 0xfffff; }\n\
     const fs = [st, st2];\n\
     function sumk(t, k, m) { let s = 0; for (let j = 0; j < m; j++) s = (s + t[j & 3] * k) & 0xffff; \
       return s; }\n\
     function outer(n, a) { \
       let a1 = (a * 3) & 0xfff, a2 = (a1 ^ 5) & 0xfff, a3 = (a2 + 7) & 0xfff, \
         a4 = (a3 - 1) & 0xfff, a5 = (a4 * 7) & 0xfff, a6 = (a5 ^ a) & 0xfff, \
         a7 = (a6 + 2) & 0xfff, a8 = (a7 >> 1) & 0xfff, a9 = (a8 + 11) & 0xfff, \
         a10 = (a9 ^ 4) & 0xfff, a11 = (a10 + 6) & 0xfff, a12 = (a11 * 5) & 0xfff; \
       const arr = [a1, a2, a3, a4]; \
       let s = 0; \
       for (let i = 0; i < n; i++) { \
         s = fs[i & 1](s, i); \
         s = (s + sumk(arr, a5, 8) + (a6 ^ i) + (a7 & a8) + (a9 | a10) + (a11 ^ a12)) & 0xfffff; } \
       return s + a1 + a12; }\n\
     let t = 0; for (let k = 0; k < 300; k++) t = (t + outer(10, k)) % 1000003;\n\
     t + ':' + outer(4, 2.5)";

fn configurations() -> Vec<(&'static str, TestSwitches)> {
    let on = TestSwitches {
        on: true,
        ..Default::default()
    };
    let graph = TestSwitches {
        inline_graph: Some(true),
        ..eager()
    };
    vec![
        ("on", on),
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
        ("graph", graph),
        (
            "graph, forced exits 7",
            TestSwitches {
                exit_every: 7,
                ..graph
            },
        ),
    ]
}

/// `program`'s result in each configuration equals the VM's, and the tier made loop caches in
/// the run labelled `cached`.
fn equal_with_caches(program: &str, cached: &str) {
    let expected = run(Tier::Bytecode, program);
    for (label, switches) in configurations() {
        compiled();
        caches();
        assert_eq!(with_switches(switches, program), expected, "{label}");
        let made = caches();
        assert!(!compiled().is_empty(), "{label}: the tier compiled");
        if label == cached {
            assert!(made > 0, "{label}: the allocations made loop caches");
        }
    }
}

#[test]
fn values_spilled_across_an_outer_loops_call_are_cached_through_its_inner_loop() {
    equal_with_caches(MIX, "eager");
}

#[test]
fn a_throw_from_the_outer_loops_call_is_caught_outside_the_loops_with_the_vms_values() {
    equal_with_caches(GUARD, "eager");
}

#[test]
fn objects_cached_through_the_inner_loop_survive_collections_in_the_outer_loops_call() {
    // With the tier on: an eager compile, before the inner loop's property sites ran, makes
    // them generic nodes, a call in the inner loop.
    equal_with_caches(OBJECTS, "on");
    // Collections at every allocation: each call of `mk` collects while the objects are in
    // their slots and callee-saved registers, and the inner loop reloads its caches after it.
    let program = OBJECTS.replace("k < 200", "k < 60");
    let expected = run(Tier::Bytecode, &program);
    TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    {
        let mut core = engine.interp.gc_heap.core.borrow_mut();
        core.set_stress_every(1);
        core.set_incremental(crate::gc_core::IncrementalMode::Off);
    }
    caches();
    let before = engine.interp.gc_heap.core.borrow().collections();
    assert_eq!(eval(&mut engine, &program), expected);
    assert!(engine.interp.gc_heap.core.borrow().collections() > before);
    assert!(caches() > 0, "the allocations made loop caches");
}

#[test]
fn a_leafs_loop_inlined_at_graph_reads_its_callers_spilled_values_from_caches() {
    // At splice the leaf is a call; at graph its loop is the caller's.
    equal_with_caches(INLINED, "graph");
}
