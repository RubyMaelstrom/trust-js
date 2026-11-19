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

/// A leaf whose loop reads its argument k, inlined at graph into a loop of a caller that keeps six
/// int32s across its own call, each read ten times a turn: they outweigh k (read once in the
/// leaf's loop, at depth 2), which is defined after them while every callee-saved register holds
/// one and so is spilled at its definition (rule B), and the leaf's loop reads it from a cache.
/// (Eight reads a turn left k the heavier; twelve made the caller too large for the plan to
/// inline the leaf.)
const INLINED: &str = "function st(s, k) { return (s * 31 + k) & 0xfffff; }\n\
     function st2(s, k) { return (s ^ (k * 7)) & 0xfffff; }\n\
     const fs = [st, st2];\n\
     function sumk(k, m) { let s = 0; for (let j = 0; j < m; j++) s = (s + (j & 3) * k) & 0xffff; \
       return s; }\n\
     function outer(n, a) { \
       let a1 = (a * 3) & 0xfff, a2 = (a1 ^ 5) & 0xfff, a3 = (a2 + 7) & 0xfff, \
         a4 = (a3 - 1) & 0xfff, a5 = (a4 * 7) & 0xfff, a6 = (a5 ^ a) & 0xfff; \
       const k5 = (a * 5 + 1) & 0xfff; \
       let s = 0; \
       for (let i = 0; i < n; i++) { \
         s = fs[i & 1](s, i); \
         s = (s + sumk(k5, 8) + ((a1 + 1) ^ a2) + ((a3 + 1) & a4) + ((a5 + 1) | a6) \
           + ((a1 + 2) ^ a2) + ((a3 + 2) & a4) + ((a5 + 2) | a6) \
           + ((a1 + 3) ^ a2) + ((a3 + 3) & a4) + ((a5 + 3) | a6) \
           + ((a1 + 4) ^ a2) + ((a3 + 4) & a4) + ((a5 + 4) | a6) \
           + ((a1 + 5) ^ a2) + ((a3 + 5) & a4) + ((a5 + 5) | a6) \
           + ((a1 + 6) ^ a2) + ((a3 + 6) & a4) + ((a5 + 6) | a6) \
           + ((a1 + 7) ^ a2) + ((a3 + 7) & a4) + ((a5 + 7) | a6) \
           + ((a1 + 8) ^ a2) + ((a3 + 8) & a4) + ((a5 + 8) | a6) \
           + ((a1 + 9) ^ a2) + ((a3 + 9) & a4) + ((a5 + 9) | a6) \
           + ((a1 + 10) ^ a2) + ((a3 + 10) & a4) + ((a5 + 10) | a6)) & 0xfffff; } \
       return s + a1 + a6; }\n\
     let t = 0; for (let k = 0; k < 300; k++) t = (t + outer(10, k)) % 1000003;\n\
     t + ':' + outer(4, 2.5)";

/// Fourteen int32s live across the loop's two calls and read several times between them in one
/// block: no loop cache (every callee-saved register holds a value across the calls), read spans;
/// then a1, a2 and a3 read once after the second call, outside any span, from their slots.
const BETWEEN: &str = "function st(s, k) { return (s * 31 + k) & 0xfffff; }\n\
     function st2(s, k) { return (s ^ (k * 7)) & 0xfffff; }\n\
     const fs = [st, st2];\n\
     function between(n, a, b) { \
       let a1 = (a * 3) & 0xfff, a2 = (a1 ^ b) & 0xfff, a3 = (a2 + 5) & 0xfff, \
         a4 = (a3 - 2) & 0xfff, a5 = (a4 * 7) & 0xfff, a6 = (a5 ^ a) & 0xfff, \
         a7 = (a6 + b) & 0xfff, a8 = (a7 >> 1) & 0xfff, a9 = (a8 + 11) & 0xfff, \
         a10 = (a9 ^ 3) & 0xfff, a11 = (a10 + 9) & 0xfff, a12 = (a11 * 5) & 0xfff, \
         a13 = (a12 ^ a1) & 0xfff, a14 = (a13 + a2) & 0xfff; \
       let s = 0; \
       for (let i = 0; i < n; i++) { \
         s = fs[i & 1](s, i); \
         s = (s + (a1 ^ a2) + (a1 & a3) + (a1 | a4) + (a2 ^ a5) + (a2 & a6) + (a3 | a7) \
           + (a3 ^ a8) + (a4 & a9) + (a4 | a10) + (a5 ^ a11) + (a6 & a12) + (a7 | a13) \
           + (a8 ^ a14) + (a9 & a1) + (a10 | a2) + (a11 ^ a3) + (a12 & a4) + (a13 | a5) \
           + (a14 ^ a6)) & 0xfffff; \
         s = fs[(i + 1) & 1](s, a1 & 7); \
         s = (s + a1 + (a2 ^ a3)) & 0xfffff; } \
       return s + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14; }\n\
     let t = 0; for (let k = 0; k < 400; k++) t = (t + between(10, k, k * 3)) % 1000003;\n\
     t + ':' + between(5, 2.5, 1) + ':' + between(3, 1e9, 2)";

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
fn values_read_several_times_between_a_loops_calls_read_one_register_each_run() {
    let expected = run(Tier::Bytecode, BETWEEN);
    for (label, switches) in configurations() {
        crate::opt::regalloc::TEST_SPANS.with(|made| made.set(0));
        assert_eq!(with_switches(switches, BETWEEN), expected, "{label}");
        let made = crate::opt::regalloc::TEST_SPANS.with(|made| made.replace(0));
        if label == "eager" {
            assert!(made > 0, "eager: the allocations made read spans");
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

/// A value made first and read only in the loop, pushed out by twenty-four int32s that a nest of
/// two loops reads before the loop (each read weighing 8 to the power 2, rule B), through which
/// it lives unread and so has the cheapest whole range, in a loop that calls `cat`, which graph
/// inlining builds into the loop with its `s + k` a generic node of the inlined frame: a call in
/// the loop, across which few values live, so the cache is a callee-saved register.
const INLINED_CALL: &str = "function cat(s, k) { return s + k; }\n\
     function f(n, a) { \
       const v = (a * 7) & 0xffff; \
       let b1 = (a ^ 1) & 0xfff, b2 = (a ^ 2) & 0xfff, b3 = (a ^ 3) & 0xfff, b4 = (a ^ 4) & 0xfff, \
         b5 = (a ^ 5) & 0xfff, b6 = (a ^ 6) & 0xfff, b7 = (a ^ 7) & 0xfff, b8 = (a ^ 8) & 0xfff, \
         b9 = (a ^ 9) & 0xfff, b10 = (a ^ 10) & 0xfff, b11 = (a ^ 11) & 0xfff, b12 = (a ^ 12) & 0xfff, \
         b13 = (a ^ 13) & 0xfff, b14 = (a ^ 14) & 0xfff, b15 = (a ^ 15) & 0xfff, b16 = (a ^ 16) & 0xfff, \
         b17 = (a ^ 17) & 0xfff, b18 = (a ^ 18) & 0xfff, b19 = (a ^ 19) & 0xfff, b20 = (a ^ 20) & 0xfff, \
         b21 = (a ^ 21) & 0xfff, b22 = (a ^ 22) & 0xfff, b23 = (a ^ 23) & 0xfff, b24 = (a ^ 24) & 0xfff; \
       let r = 0; \
       for (let x = 0; x < 2; x++) for (let y = 0; y < 2; y++) \
         r = (r + (b1 ^ b2) + (b3 ^ b4) + (b5 ^ b6) + (b7 ^ b8) + (b9 ^ b10) + (b11 ^ b12) \
           + (b13 ^ b14) + (b15 ^ b16) + (b17 ^ b18) + (b19 ^ b20) + (b21 ^ b22) + (b23 ^ b24)) & 0xffff; \
       let s = ''; \
       for (let i = 0; i < n; i++) { \
         s = cat(s, i & 3); \
         r = (r + v + (v >> 3)) & 0xffff; } \
       return r + s.length; }\n\
     let t = 0; for (let k = 0; k < 300; k++) t = (t + f(12, k)) % 1000003;\n\
     t + ':' + f(5, 2.5)";

/// INLINED_CALL's loop, `cat`'s `s + k` throwing where k's valueOf does, caught by a handler in
/// the loop around the call: the inlined frame's generic node at a pc with a handler in its
/// chain exits by its recipe, which names the cached value's slot.
const INLINED_THROW: &str = "function cat(s, k) { return s + k; }\n\
     const bad = { valueOf() { throw new RangeError('no'); } };\n\
     function f(n, a, odd) { \
       const v = (a * 7) & 0xffff; \
       let b1 = (a ^ 1) & 0xfff, b2 = (a ^ 2) & 0xfff, b3 = (a ^ 3) & 0xfff, b4 = (a ^ 4) & 0xfff, \
         b5 = (a ^ 5) & 0xfff, b6 = (a ^ 6) & 0xfff, b7 = (a ^ 7) & 0xfff, b8 = (a ^ 8) & 0xfff, \
         b9 = (a ^ 9) & 0xfff, b10 = (a ^ 10) & 0xfff, b11 = (a ^ 11) & 0xfff, b12 = (a ^ 12) & 0xfff, \
         b13 = (a ^ 13) & 0xfff, b14 = (a ^ 14) & 0xfff, b15 = (a ^ 15) & 0xfff, b16 = (a ^ 16) & 0xfff, \
         b17 = (a ^ 17) & 0xfff, b18 = (a ^ 18) & 0xfff, b19 = (a ^ 19) & 0xfff, b20 = (a ^ 20) & 0xfff, \
         b21 = (a ^ 21) & 0xfff, b22 = (a ^ 22) & 0xfff, b23 = (a ^ 23) & 0xfff, b24 = (a ^ 24) & 0xfff; \
       let r = 0; \
       for (let x = 0; x < 2; x++) for (let y = 0; y < 2; y++) \
         r = (r + (b1 ^ b2) + (b3 ^ b4) + (b5 ^ b6) + (b7 ^ b8) + (b9 ^ b10) + (b11 ^ b12) \
           + (b13 ^ b14) + (b15 ^ b16) + (b17 ^ b18) + (b19 ^ b20) + (b21 ^ b22) + (b23 ^ b24)) & 0xffff; \
       let s = ''; \
       for (let i = 0; i < n; i++) { \
         try { s = cat(s, i === 3 && odd ? bad : i & 3); } catch (e) { r = (r + v + e.message.length) & 0xffff; } \
         r = (r + v + (v >> 3)) & 0xffff; } \
       return r + s.length; }\n\
     let t = 0; for (let k = 0; k < 300; k++) t = (t + f(8, k, k % 5 === 4)) % 1000003;\n\
     t + ':' + f(6, 11, true)";

/// `program` in each configuration equals the VM's, and the allocations at graph made loop
/// caches in callee-saved registers.
fn equal_with_callee_caches(program: &str) {
    let expected = run(Tier::Bytecode, program);
    for (label, switches) in configurations() {
        crate::opt::regalloc::TEST_CALLEE_CACHES.with(|made| made.set(0));
        assert_eq!(with_switches(switches, program), expected, "{label}");
        let made = crate::opt::regalloc::TEST_CALLEE_CACHES.with(|made| made.replace(0));
        if label == "graph" {
            assert!(
                made > 0,
                "graph: the allocations made callee-saved loop caches"
            );
        }
    }
}

#[test]
fn a_value_cached_through_a_loop_with_an_inlined_frames_generic_node_takes_a_callee_saved_register()
{
    // The consistency check after every allocation refuses a caller-saved cache through a loop
    // with a call, which an inlined frame's generic node is (NodeOp::Generic).
    equal_with_callee_caches(INLINED_CALL);
}

#[test]
fn an_inlined_frames_throw_exit_in_a_cached_loop_reads_the_cached_values_slot() {
    equal_with_callee_caches(INLINED_THROW);
}

/// INLINED_CALL's loop with a callee that makes a plain call (P3 stage 3 step 3c-2's piece B): at
/// graph `wrap` is built into the loop and its call of `g` (which reads `arguments`, so the plan
/// inlines none of `wrap`'s calls) is a call node of the inlined frame, whose template runs `g`'s
/// code with x21 on the inlined call helpers' table: a call in the loop, across which the value
/// read in the loop and spilled before it lives in a callee-saved cache.
const INLINED_PLAIN_CALL: &str = "function g(k) { return (k * 3 + arguments.length) & 0xff; }\n\
     function wrap(s, k) { var t = g(k); return (s + t) & 0xffff; }\n\
     function f(n, a) { \
       const v = (a * 7) & 0xffff; \
       let b1 = (a ^ 1) & 0xfff, b2 = (a ^ 2) & 0xfff, b3 = (a ^ 3) & 0xfff, b4 = (a ^ 4) & 0xfff, \
         b5 = (a ^ 5) & 0xfff, b6 = (a ^ 6) & 0xfff, b7 = (a ^ 7) & 0xfff, b8 = (a ^ 8) & 0xfff, \
         b9 = (a ^ 9) & 0xfff, b10 = (a ^ 10) & 0xfff, b11 = (a ^ 11) & 0xfff, b12 = (a ^ 12) & 0xfff, \
         b13 = (a ^ 13) & 0xfff, b14 = (a ^ 14) & 0xfff, b15 = (a ^ 15) & 0xfff, b16 = (a ^ 16) & 0xfff, \
         b17 = (a ^ 17) & 0xfff, b18 = (a ^ 18) & 0xfff, b19 = (a ^ 19) & 0xfff, b20 = (a ^ 20) & 0xfff, \
         b21 = (a ^ 21) & 0xfff, b22 = (a ^ 22) & 0xfff, b23 = (a ^ 23) & 0xfff, b24 = (a ^ 24) & 0xfff; \
       let r = 0; \
       for (let x = 0; x < 2; x++) for (let y = 0; y < 2; y++) \
         r = (r + (b1 ^ b2) + (b3 ^ b4) + (b5 ^ b6) + (b7 ^ b8) + (b9 ^ b10) + (b11 ^ b12) \
           + (b13 ^ b14) + (b15 ^ b16) + (b17 ^ b18) + (b19 ^ b20) + (b21 ^ b22) + (b23 ^ b24)) & 0xffff; \
       let s = 0; \
       for (let i = 0; i < n; i++) { \
         s = wrap(s, i & 3); \
         r = (r + v + (v >> 3)) & 0xffff; } \
       return r + s; }\n\
     let t = 0; for (let k = 0; k < 300; k++) t = (t + f(12, k)) % 1000003;\n\
     t + ':' + f(5, 2.5)";

#[test]
fn a_value_cached_through_a_loop_whose_inlined_frame_makes_a_call_takes_a_callee_saved_register() {
    // The call node's template enters another function's code: a call by NodeOp::is_effectful,
    // as every generic node is, so the consistency check after every allocation refuses a
    // caller-saved cache through the loop; the results equal the VM's in every configuration,
    // and at graph an inlined frame's call node was lowered and a callee-saved cache made.
    let expected = run(Tier::Bytecode, INLINED_PLAIN_CALL);
    for (label, switches) in configurations() {
        crate::opt::regalloc::TEST_CALLEE_CACHES.with(|made| made.set(0));
        crate::opt::a64::TEST_INLINED_GENERIC.with(|ops| ops.borrow_mut().clear());
        assert_eq!(
            with_switches(switches, INLINED_PLAIN_CALL),
            expected,
            "{label}"
        );
        let made = crate::opt::regalloc::TEST_CALLEE_CACHES.with(|made| made.replace(0));
        let calls = crate::opt::a64::TEST_INLINED_GENERIC.with(|ops| {
            std::mem::take(&mut *ops.borrow_mut())
                .iter()
                .filter(|op| {
                    matches!(
                        op,
                        crate::bytecode::Op::Call(..) | crate::bytecode::Op::CallWithThis(..)
                    )
                })
                .count()
        });
        if label == "graph" {
            assert!(calls > 0, "graph: an inlined frame's call node");
            assert!(
                made > 0,
                "graph: the allocations made callee-saved loop caches"
            );
        }
    }
}
