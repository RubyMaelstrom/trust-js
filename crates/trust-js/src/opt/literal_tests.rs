//! Piece B's tests (P3 stage 3, step 3g): a closure the compiled function creates and passes to
//! an inlined callee that calls it, built as a by-literal frame with no guard (the design:
//! target/p3/stage3/step3f-impl/design-note-B.txt and step3g-impl/b-note.txt; the conditions:
//! briefs/step3g-resume2.txt), and its callee's word in the activation's frame-callee table,
//! which the stack's readers resolve it through (`Interp::frame_callee`).

use super::ir::ExitReason;
use super::TestSwitches;
use crate::bytecode::Tier;
use crate::{Completion, Engine};

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

/// `source`'s completion with the tier off (the bytecode VM, its own thread's switches off).
fn tier_off(source: &str) -> String {
    TestSwitches::default().set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, source)
}

/// `source`'s completion with the tier on and eager under `switches`, at TRUST_JS_OPT_INLINE=graph.
fn graph_with(switches: TestSwitches, source: &str) -> String {
    TestSwitches {
        on: true,
        eager: true,
        inline_graph: Some(true),
        ..switches
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    eval(&mut engine, source)
}

fn graph(source: &str) -> String {
    graph_with(TestSwitches::default(), source)
}

/// The by-literal frames built since the last call.
fn literal_frames() -> usize {
    super::build::TEST_LITERAL_FRAMES.with(|count| count.replace(0))
}

/// The exits taken since the last call, by reason.
fn exits_for(reason: ExitReason) -> usize {
    super::exit::TEST_EXITS.with(|exits| {
        std::mem::take(&mut *exits.borrow_mut())
            .into_iter()
            .filter(|&(_, r)| r == reason)
            .count()
    })
}

/// React's shape: a function passing a fresh arrow to an inlined helper that calls it in a loop.
const EACH: &str = "function each(xs, f) { let s = 0; \
     for (let i = 0; i < xs.length; i++) s += f(xs[i]); return s; }\n\
     function scaled(xs, k) { return each(xs, (x) => x * k + 1); }\n\
     const xs = [1, 2, 3, 4];\n\
     let t = 0; for (let k = 0; k < 3000; k++) t = (t + scaled(xs, k)) % 1000003; t";

#[test]
fn a_fresh_closure_passed_to_an_inlined_callee_is_a_by_literal_frame() {
    let expected = tier_off(EACH);
    literal_frames();
    exits_for(ExitReason::WrongCallee);
    assert_eq!(graph(EACH), expected);
    assert!(literal_frames() > 0, "a by-literal frame built");
    assert_eq!(exits_for(ExitReason::WrongCallee), 0, "no guard to fail");
}

/// The arrow reads the creating frame's binding through its environment (EnvParent from the
/// creation's), and a sloppy function it calls reads its `caller` and a stack.
const REFLECT: &str = "function probe(v) { \
     const lines = new Error('x').stack.split('\\n').slice(1, 4).map(l => l.trim().split(' ')[1]); \
     return [String(probe.caller).slice(0, 12), lines.join('/'), v].join(':'); }\n\
     function each(xs, f) { let s = ''; for (let i = 0; i < xs.length; i++) s = f(xs[i]); \
     return s; }\n\
     function outer(xs, k) { let base = k * 2; return each(xs, (x) => probe(x + base)); }\n\
     const xs = [1, 2];\n\
     let r = ''; for (let k = 0; k < 3000; k++) r = outer(xs, k); r";

#[test]
fn a_by_literal_frames_callee_is_its_stacks_and_callers_function() {
    let expected = tier_off(REFLECT);
    literal_frames();
    assert_eq!(graph(REFLECT), expected);
    assert!(literal_frames() > 0, "a by-literal frame built");
}

#[test]
fn forced_exits_in_a_by_literal_frame_rebuild_it() {
    for every in [1, 7] {
        let switches = TestSwitches {
            exit_every: every,
            ..Default::default()
        };
        assert_eq!(
            graph_with(switches, EACH),
            tier_off(EACH),
            "exit every {every}"
        );
        assert_eq!(
            graph_with(switches, REFLECT),
            tier_off(REFLECT),
            "exit every {every}"
        );
    }
}

/// Whether every entry of `engine`'s frame-callee table is clear (or it has none).
fn table_clear(engine: &Engine) -> bool {
    let interp = &*engine.interp as *const crate::interpreter::Interp as usize;
    (0..crate::stack_trace::FRAME_CALLEE_WORDS)
        .all(|d| crate::stack_trace::frame_callee_word(interp, d) == 0)
}

/// `source` at graph under `switches` on a fresh engine, which it returns with the completion.
fn graph_engine(switches: TestSwitches, source: &str) -> (Engine, String) {
    TestSwitches {
        on: true,
        eager: true,
        inline_graph: Some(true),
        ..switches
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let result = eval(&mut engine, source);
    (engine, result)
}

/// `each`'s call of `f` met one closure before `scaled`'s compile (the empty arrays reach no
/// call), so that the plan has an identity way there; every later closure is a fresh one.
const PLANNED: &str = "function each(xs, f) { let s = 0; \
     for (let i = 0; i < xs.length; i++) s += f(xs[i]); return s; }\n\
     function scaled(xs, k) { return each(xs, (x) => x * k + 1); }\n\
     const xs = [1, 2, 3, 4], none = [];\n\
     let t = scaled(xs, 1);\n\
     for (let k = 0; k < 3000; k++) t += scaled(none, k);\n\
     for (let k = 0; k < 3000; k++) t = (t + scaled(xs, k)) % 1000003; t";

/// P1 (the closure fact before a nested site's guarded ways): where the plan has an identity way
/// at the call and its callee node is a fresh closure of the literal, the by-literal frame is
/// built and no identity guard: no WrongCallee exit over thousands of fresh closures.
#[test]
fn the_fact_is_taken_before_a_planned_identity_way() {
    let expected = tier_off(PLANNED);
    literal_frames();
    exits_for(ExitReason::WrongCallee);
    assert_eq!(graph(PLANNED), expected);
    assert!(literal_frames() > 0, "a by-literal frame built");
    assert_eq!(
        exits_for(ExitReason::WrongCallee),
        0,
        "no identity guard at the call"
    );
}

/// The compiled frame reads the stack and a sloppy function's `caller` right after its
/// by-literal frame returned (the entry cleared at the return, P2), and after an exit from it.
const AFTER: &str = "function probe() { \
     const top = new Error('x').stack.split('\\n').slice(1, 3).map(l => l.trim().split(' ')[1]); \
     return String(probe.caller).slice(0, 15) + '/' + top.join('/'); }\n\
     function each(xs, f) { let s = 0; for (let i = 0; i < xs.length; i++) s += f(xs[i]); \
     return s; }\n\
     function outer(xs, k) { const s = each(xs, (x) => x * k + 1); return s + ':' + probe(); }\n\
     const xs = [1, 2];\n\
     let r = ''; for (let k = 0; k < 3000; k++) r = outer(xs, k); r";

/// P2: the stack read after a by-literal frame returned and after exits from it equals the tier
/// off's, and no entry of the table outlives its frame (the return's and the exit's clears).
#[test]
fn no_entry_outlives_its_by_literal_frame() {
    for source in [EACH, REFLECT, AFTER] {
        let expected = tier_off(source);
        for every in [0, 1, 7] {
            let switches = TestSwitches {
                exit_every: every,
                ..Default::default()
            };
            literal_frames();
            let (engine, result) = graph_engine(switches, source);
            assert_eq!(result, expected, "exit every {every}");
            assert!(literal_frames() > 0, "a by-literal frame built");
            assert!(
                table_clear(&engine),
                "an entry left after its frame (exit every {every})"
            );
        }
    }
}

thread_local! {
    /// The other engine of `two_engines_on_one_thread_keep_their_own_callees`.
    static OTHER: std::cell::RefCell<Option<Engine>> = const { std::cell::RefCell::new(None) };
}

/// Condition 1: two engines on one thread, each with a by-literal frame active at the same
/// FnFrame index; engine A's `caller` and stack, read inside its frame after engine B's frame at
/// that index returned, name A's closure.
#[test]
fn two_engines_on_one_thread_keep_their_own_callees() {
    const A: &str = "function probe(v) { \
         const top = new Error('x').stack.split('\\n').slice(1, 3).map(l => l.trim().split(' ')[1]); \
         return String(probe.caller).slice(0, 18) + '/' + top.join('/') + '/' + v; }\n\
         function each(xs, f) { let s = ''; for (let i = 0; i < xs.length; i++) s = f(xs[i]); \
         return s; }\n\
         function outer(xs, k) { return each(xs, (aa) => { other(); return probe(aa + k); }); }\n\
         const xs = [1, 2];\n\
         let r = ''; for (let k = 0; k < 2000; k++) r = outer(xs, k); r";
    const B: &str = "function each(xs, f) { let s = 0; \
         for (let i = 0; i < xs.length; i++) s += f(xs[i]); return s; }\n\
         function scaled(xs, k) { return each(xs, (bb) => bb * k + 1); }\n\
         var u = 0; function run() { u = (u + scaled([1, 2, 3], 7)) % 1000; return u; }";
    let expected = {
        TestSwitches::default().set();
        let mut a = Engine::new();
        a.set_tier(Tier::Bytecode);
        a.interp
            .def_method(&a.interp.global, "other", 0, |_, _, _| {
                Ok(crate::value::Value::Undefined)
            });
        eval(&mut a, A)
    };
    TestSwitches {
        on: true,
        eager: true,
        inline_graph: Some(true),
        ..Default::default()
    }
    .set();
    let mut b = Engine::new();
    b.set_tier(Tier::Jit);
    b.set_tier_threshold(0);
    eval(&mut b, B);
    OTHER.with(|other| *other.borrow_mut() = Some(b));
    let mut a = Engine::new();
    a.set_tier(Tier::Jit);
    a.set_tier_threshold(0);
    a.interp
        .def_method(&a.interp.global, "other", 0, |_, _, _| {
            OTHER.with(|other| {
                let mut other = other.borrow_mut();
                let b = other.as_mut().expect("engine B");
                // B's top-level run() calls scaled at FnFrame index 1, A's outer's index too.
                b.eval("run()", false).expect("parses");
            });
            Ok(crate::value::Value::Undefined)
        });
    literal_frames();
    assert_eq!(eval(&mut a, A), expected);
    assert!(
        literal_frames() >= 2,
        "both engines' by-literal frames built"
    );
    OTHER.with(|other| other.borrow_mut().take());
}

thread_local! {
    static HELD: std::cell::RefCell<Option<crate::value::WeakGc>> =
        const { std::cell::RefCell::new(None) };
}

thread_local! {
    /// Whether the held closure was alive at every `alive()` call so far.
    static DURING: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Condition 3: the closure a by-literal frame runs lives while the frame runs: its call out
/// collects and finds it (here on a Rust entry, Reflect.apply, where the collector scans the
/// entry's operand words to its live top), and no entry outlives its frame. Whether it is
/// collected once its activation returned, at the same depth, is not asserted: the collector's
/// conservative scan of the native stack keeps a stale word's closure in every tier (the tier
/// off's bytecode VM, the template JIT and both inlining modes each answered either way on
/// 2026-11-18, step3g-impl/b1/diag-log.log); the table's own part in it, that the trace reads
/// no entry but one a frame above a running by-literal frame was called from, is
/// `the_trace_reads_an_entry_only_below_a_call_from_a_by_literal_frame`.
#[test]
fn a_by_literal_frames_closure_lives_while_it_runs() {
    const SOURCE: &str = "function each(xs, f) { hold(f); let s = 0; \
         for (let i = 0; i < xs.length; i++) s += f(xs[i]); return s; }\n\
         function scaled(xs, k) { \
         return each(xs, (x) => x * k + 1 + (k % 500 === 0 ? alive() : 0)); }\n\
         const xs = [1, 2];\n\
         let t = 0; for (let k = 0; k < 3000; k++) \
         t = (t + Reflect.apply(scaled, null, [xs, k])) % 1000003; t";
    fn install(engine: &mut Engine) {
        DURING.with(|during| during.set(true));
        engine
            .interp
            .def_method(&engine.interp.global, "hold", 1, |_, _, args| {
                let f = args[0].as_obj().expect("a closure");
                HELD.with(|held| *held.borrow_mut() = Some(crate::value::Gc::downgrade(&f)));
                Ok(crate::value::Value::Undefined)
            });
        engine
            .interp
            .def_method(&engine.interp.global, "alive", 0, |interp, _, _| {
                interp.gc_collect();
                let alive = HELD.with(|held| {
                    held.borrow()
                        .as_ref()
                        .is_some_and(|weak| weak.upgrade().is_some())
                });
                DURING.with(|during| during.set(during.get() && alive));
                Ok(crate::value::Value::Num(0.0))
            });
    }
    let expected = {
        TestSwitches::default().set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        install(&mut engine);
        eval(&mut engine, SOURCE)
    };
    assert!(
        DURING.with(std::cell::Cell::get),
        "alive while called, the tier off"
    );
    for every in [0, 7] {
        TestSwitches {
            on: true,
            eager: true,
            inline_graph: Some(true),
            exit_every: every,
            ..Default::default()
        }
        .set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        install(&mut engine);
        literal_frames();
        assert_eq!(eval(&mut engine, SOURCE), expected, "exit every {every}");
        assert!(literal_frames() > 0, "a by-literal frame built");
        assert!(
            DURING.with(std::cell::Cell::get),
            "alive while its frame runs"
        );
        assert!(table_clear(&engine), "an entry left after its frame");
    }
}

/// Condition 3's rule, read directly: the collector reads entry d of an interpreter's table only
/// where frame d + 1's call position names a by-literal frame's callee, never the top frame's
/// entry; and the invariant the clears keep (an entry is nonzero exactly while the by-literal
/// frame that stored it runs), which debug builds' trace asserts: an entry under a frame called
/// from elsewhere, or one at or above the frames' count, fails it.
#[test]
fn the_trace_reads_an_entry_only_below_a_call_from_a_by_literal_frame() {
    struct Words(Vec<u64>);
    impl crate::gc_trace::GcSink for Words {
        fn word(&mut self, word: u64) {
            self.0.push(word);
        }
    }
    let engine = Engine::new();
    let object = crate::value::Value::Obj(engine.interp.global).bits();
    let key = &object as *const u64 as usize;
    let table = crate::stack_trace::frame_callee_table(key, true).expect("a table");
    let (sites, positions) =
        crate::stack_trace::InlineSites::register(vec![crate::stack_trace::InlineSite {
            callee: crate::stack_trace::InlineCallee::Frame,
            inner: 5,
            outer: 7,
        }]);
    let frame = |call_site| crate::interpreter::FnFrame {
        fn_ptr: 0,
        call_site,
        strict: false,
        construct: false,
        extra: None,
    };
    let set = |d: usize, word: u64| unsafe { *(table as *mut u64).add(d) = word };
    let trace = |frames: &[crate::interpreter::FnFrame]| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut words = Words(Vec::new());
            crate::stack_trace::trace_frame_callees(key, frames, &mut words);
            words.0
        }))
    };
    // Entry 0 under a frame called from a by-literal frame, entry 2 the top's: entry 0 alone.
    let frames = [frame(7), frame(positions[0]), frame(9)];
    set(0, object);
    set(2, object);
    let read = trace(&frames).expect("a state the invariant allows");
    assert_eq!(read.len(), 1, "entry 0 alone: {read:x?}");
    // Entry 1 under a frame called from elsewhere; then entry 3, above the three frames.
    for d in [1, 3] {
        set(d, object);
        let read = trace(&frames);
        if cfg!(debug_assertions) {
            assert!(read.is_err(), "entry {d}: the invariant's assertion");
        } else {
            assert_eq!(read.expect("release").len(), 1, "entry {d} not read");
        }
        set(d, 0);
    }
    crate::stack_trace::clear_frame_callees(key);
    drop(sites);
}

/// Condition 4: an activation deeper than the table takes the generic call at its by-literal
/// sites, with the same results and stacks.
#[test]
fn an_activation_past_the_table_takes_the_generic_call() {
    let source = format!(
        "function probe() {{ return new Error('x').stack.split('\\n').length; }}\n\
         function each(xs, f) {{ let s = 0; for (let i = 0; i < xs.length; i++) s += f(xs[i]); \
         return s; }}\n\
         function scaled(xs, k) {{ return each(xs, (x) => x * k + probe()); }}\n\
         function down(n, k) {{ return n === 0 ? scaled([1, 2], k) : down(n - 1, k); }}\n\
         let t = 0; for (let k = 0; k < 1000; k++) t = (t + down(k % 2 ? 3 : {}, k)) % 1000003; t",
        crate::stack_trace::FRAME_CALLEE_WORDS + 40
    );
    assert_eq!(graph(&source), tier_off(&source));
}

/// The planner's rules hold for a literal (`inline_callee_rules`): an arrow that reads its
/// lexical `this` is no by-literal frame (the planner's "arrow lexical observations"), and its
/// call stays the call, with the tier off's result.
#[test]
fn an_arrow_reading_its_lexical_this_is_left_to_the_call() {
    const SOURCE: &str = "function each(xs, f) { let s = 0; \
         for (let i = 0; i < xs.length; i++) s += f(xs[i]); return s; }\n\
         const o = { k: 3, scaled(xs) { return each(xs, (x) => x * this.k + 1); } };\n\
         const xs = [1, 2, 3];\n\
         let t = 0; for (let k = 0; k < 3000; k++) t = (t + o.scaled(xs)) % 1000003; t";
    let expected = tier_off(SOURCE);
    literal_frames();
    assert_eq!(graph(SOURCE), expected);
    assert_eq!(
        literal_frames(),
        0,
        "no by-literal frame for a lexical `this`"
    );
}

/// React's shape through the self-hosted Array.prototype methods, method calls whose callee word
/// lies under the arguments and over the receiver: map, filter and forEach of fresh arrows in a
/// strict function, each a by-literal frame in the inlined method's frame.
#[test]
fn array_methods_of_fresh_arrows_are_by_literal_frames() {
    const SOURCE: &str = "function step(xs, k) { 'use strict'; \
         const ys = xs.map((x) => x * k + 1).filter((y) => y % 3 !== 0); \
         let s = 0; ys.forEach((y) => { s += y; }); return s; }\n\
         const xs = [1, 2, 3, 4, 5];\n\
         let t = 0; for (let k = 0; k < 3000; k++) t = (t + step(xs, k)) % 1000003; t";
    let expected = tier_off(SOURCE);
    literal_frames();
    assert_eq!(graph(SOURCE), expected);
    assert!(
        literal_frames() >= 2,
        "map's and filter's arrows by literal"
    );
    for every in [1, 7] {
        let switches = TestSwitches {
            exit_every: every,
            ..Default::default()
        };
        let (engine, result) = graph_engine(switches, SOURCE);
        assert_eq!(result, expected, "exit every {every}");
        assert!(
            table_clear(&engine),
            "an entry left after its frame (exit every {every})"
        );
    }
}

thread_local! {
    /// Whether every `table_clear_here()` call found the running engine's table clear.
    static CLEAR_EACH_TIME: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// P2: right after a by-literal frame returned, or after an exit left its code, no entry of the
/// table holds a closure: a native the compiled frame calls next finds the table clear (with
/// the return's or the exit's clear removed, the entry still held the closure there).
#[test]
fn the_entry_is_clear_right_after_the_frame_returns_or_exits() {
    // The arrow's shape check fails inside its frame for every 500th array (an exit from the
    // by-literal frame itself), besides the forced exits.
    const SOURCE: &str = "function each(xs, f) { let s = 0; \
         for (let i = 0; i < xs.length; i++) s += f(xs[i]); return s; }\n\
         function outer(xs, k) { const s = each(xs, (x) => x.v * k + 1); \
         return s + clearHere(); }\n\
         const good = [{ v: 1 }, { v: 2 }, { v: 3 }], odd = [{ v: 1 }, { w: 0, v: 2 }];\n\
         let t = 0; for (let k = 0; k < 3000; k++) \
         t = (t + outer(k % 500 === 250 ? odd : good, k)) % 1000003; t";
    fn install(engine: &mut Engine) {
        CLEAR_EACH_TIME.with(|clear| clear.set(true));
        engine
            .interp
            .def_method(&engine.interp.global, "clearHere", 0, |interp, _, _| {
                let at = &*interp as *const crate::interpreter::Interp as usize;
                let clear = (0..crate::stack_trace::FRAME_CALLEE_WORDS)
                    .all(|d| crate::stack_trace::frame_callee_word(at, d) == 0);
                CLEAR_EACH_TIME.with(|each| each.set(each.get() && clear));
                Ok(crate::value::Value::Num(0.0))
            });
    }
    let expected = {
        TestSwitches::default().set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        install(&mut engine);
        eval(&mut engine, SOURCE)
    };
    for every in [0, 1, 3] {
        TestSwitches {
            on: true,
            eager: true,
            inline_graph: Some(true),
            exit_every: every,
            ..Default::default()
        }
        .set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        install(&mut engine);
        literal_frames();
        assert_eq!(eval(&mut engine, SOURCE), expected, "exit every {every}");
        assert!(literal_frames() > 0, "a by-literal frame built");
        assert!(
            CLEAR_EACH_TIME.with(std::cell::Cell::get),
            "an entry held after its frame (exit every {every})"
        );
    }
}

/// Condition 5: an error thrown by a generic node of the by-literal frame itself (not a call:
/// an `in` of a non-object, an operation the tier leaves to its template) names the arrow's
/// frame in its stack, as the tier off's does: the frame's own inline site names its callee
/// word.
#[test]
fn an_error_thrown_in_a_by_literal_frame_names_its_closure() {
    const SOURCE: &str = "function each(xs, f) { let s = 0; \
         for (let i = 0; i < xs.length; i++) s += f(xs[i]); return s; }\n\
         function outer(xs, k) { try { return each(xs, function arrowish(x) { \
         return ('b' in x.a ? 1 : 0) + k; }); } \
         catch (e) { return e.stack.split('\\n').slice(1, 4).map(l => l.trim().split(' ')[1]).join('/'); } }\n\
         const good = [{ a: { b: 1 }, p0: 0 }, { a: { b: 2 }, p1: 0 }, { a: { b: 3 }, p2: 0 }, \
         { a: { b: 4 }, p3: 0 }, { a: { b: 5 }, p4: 0 }, { a: { b: 6 }, p5: 0 }], \
         bad = [{ a: { b: 1 }, p0: 0 }, { p6: 0 }];\n\
         let r = ''; for (let k = 0; k < 3000; k++) { const v = outer(k % 500 === 499 ? bad : good, k); \
         if (typeof v === 'string') r = v; } r";
    let expected = tier_off(SOURCE);
    assert!(expected.starts_with("arrowish/each/outer"), "{expected}");
    literal_frames();
    assert_eq!(graph(SOURCE), expected);
    assert!(literal_frames() > 0, "a by-literal frame built");
}

/// The refusals the oracle recorded for literals since the last call.
fn literal_refusals() -> Vec<&'static str> {
    super::oracle::TEST_LITERAL_REFUSALS
        .with(|refusals| std::mem::take(&mut *refusals.borrow_mut()))
}

/// The planner's rules and its budget hold for a literal (P3: `inline_callee_rules`, the budget
/// replayed with literal ways last): an arrow past the planner's callee size limit is no
/// by-literal frame; one under it is, unless what the plan left of the shared budget is spent,
/// where its call is the generic call, as today, with the tier off's result.
#[test]
fn the_planners_rules_and_budget_decide_a_literal() {
    let source = |arrow: &str, pad: &str| {
        format!(
            "function each(xs, f) {{ let s = 0; \
             for (let i = 0; i < xs.length; i++) s += f(xs[i]); return s; }}\n\
             function outer(xs, k) {{ let a = k; {pad} return each(xs, (x) => {arrow}) + a; }}\n\
             const xs = [1, 2];\n\
             let t = 0; for (let k = 0; k < 3000; k++) t = (t + outer(xs, k)) % 1000003; t"
        )
    };
    let medium = "x * 1 + x * 2 + x * 3 + x * 4 + x * 5 + x * 6 + x * 7 + x * 8 + x * 9 + x * 10 + x * 11 + x * 12";
    let large = format!("{medium} + {medium} + {medium}");
    // Past the size limit (96 operations).
    let past = source(&large, "");
    literal_refusals();
    literal_frames();
    assert_eq!(graph(&past), tier_off(&past));
    assert_eq!(
        literal_frames(),
        0,
        "no by-literal frame past the size limit"
    );
    assert!(literal_refusals().contains(&"callee size/shape"));
    // Under it, with the budget the plan leaves.
    let under = source(medium, "");
    assert_eq!(graph(&under), tier_off(&under));
    assert!(literal_frames() > 0, "a by-literal frame under the limit");
    // Under it, the shared budget spent by the compiled function's own size: the first padding
    // that leaves each's way its budget and the literal's none.
    let spent = (10..80).find_map(|n| {
        let pad: String = (1..=n).map(|k| format!("a = (a + {k}) | 0; ")).collect();
        let spent = source(medium, &pad);
        literal_refusals();
        literal_frames();
        let result = graph(&spent);
        let refused = literal_refusals().contains(&"the planner's budget");
        refused.then(|| (spent, result, literal_frames()))
    });
    let (spent, result, frames) = spent.expect("a padding at which the budget refuses the literal");
    assert_eq!(result, tier_off(&spent));
    assert_eq!(frames, 0, "no by-literal frame once the budget is spent");
}

/// A named function expression passed fresh to an inlined helper that calls it: `W`'s body
/// `literal` (its formals `n` and `k`), its results summed over 4,000 calls, under `prologue`
/// (`'use strict';` or nothing), through a user helper (`via`: `helper`) or the self-hosted
/// `map` (a strict program's: its own helper's call is no plan there).
fn named_literal(prologue: &str, via: &str, literal: &str) -> String {
    let call = match via {
        "helper" => format!("helper({literal}, n)"),
        _ => format!("[n, n].map({literal})[1]"),
    };
    format!(
        "{prologue}function helper(f, a) {{ return f(a); }}\n\
         function W(n, k) {{ return {call}; }}\n\
         let r = 0; for (let i = 0; i < 4000; i++) r += W(i % 4, i % 3); r"
    )
}

/// For each `(literal, refused)` under `prologue` through `via`: the tier off's result at graph,
/// and a by-literal frame built where `refused` is None, or else none, the oracle's refusal
/// that reason (the planner's shared rules, `bytecode::inline_callee_rules`).
fn assert_named_literals(prologue: &str, via: &str, cases: &[(&str, Option<&str>)]) {
    for &(literal, refused) in cases {
        let source = named_literal(prologue, via, literal);
        let expected = tier_off(&source);
        literal_frames();
        literal_refusals();
        assert_eq!(graph(&source), expected, "{literal}");
        assert_eq!(
            literal_frames() > 0,
            refused.is_none(),
            "a by-literal frame: {literal}"
        );
        if let Some(reason) = refused {
            assert!(literal_refusals().contains(&reason), "{reason}: {literal}");
        }
    }
}

/// A named function expression whose name may be observed gets a self-name record as its
/// closure's [[Environment]] (`Interp::make_function_named`), which the by-literal frame skips:
/// the name's read is the frame's callee node, every other name one record fewer from the
/// creation's environment (`Builder::running_walk`, `Builder::self_name`); in test builds the
/// frame's entry checks that assumption against the closure (`NodeOp::CheckFrameEnv`).
#[test]
fn a_named_function_expressions_own_name_is_its_by_literal_frames_callee() {
    assert_named_literals(
        "",
        "helper",
        &[
            // The name called (recursion through the frame's callee) beside a free name.
            (
                "function me(x) { return x <= 0 ? k : me(x - 1) + k; }",
                None,
            ),
            // The name's own property.
            (
                "function me(x) { return (me.length === 1 ? x : -1) + k; }",
                None,
            ),
            // The name read and unused, a free name beside it: the record is there all the same.
            ("function me(x) { var t = me; return x + k; }", None),
            // The name compared, the free name past the record.
            (
                "function me(x) { return me === undefined ? -1000 : x + k; }",
                None,
            ),
            // No self-name record (the name never read): the creation's environment is the
            // closure's.
            ("function me(x) { return x + k; }", None),
            // Shadowed by a parameter or a `var`: a record no code reads, the free name past it.
            ("function me(me) { return me + k; }", None),
            ("function me(x) { var me = x * 2; return me + k; }", None),
        ],
    );
    assert_named_literals(
        "'use strict';\n",
        "map",
        &[("function me(x) { return me.length + x + k; }", None)],
    );
}

/// The name read where the literal's code would make records of its own (a block's captured
/// `let`) or a closure, and written in sloppy code (no effect) and strict code (a TypeError):
/// none of those operations is one an inlined frame runs (`bytecode::inline_capability`, the
/// planner's shared rule), so each literal is refused and its call stays the generic call.
#[test]
fn a_named_function_expression_that_writes_its_name_or_makes_records_is_refused() {
    let context = "callee operation requires an unremapped call context";
    assert_named_literals(
        "",
        "helper",
        &[
            (
                "function me(x) { let s = 0; { let y = x + k; const g = () => y; \
                 s = g() + me.length; } return s; }",
                Some("callee size/shape"),
            ),
            (
                "function me(x) { const g = () => me; return g() === me ? x + k : -1000; }",
                Some(context),
            ),
            (
                "function me(x) { me = 5; return me === 5 ? -1000 : x + k; }",
                Some(context),
            ),
        ],
    );
    assert_named_literals(
        "'use strict';\n",
        "map",
        &[(
            "function me(x) { try { me = 5; } catch (e) { \
             return x + k + (e instanceof TypeError ? 100 : 0); } return -1000; }",
            Some(context),
        )],
    );
}

/// Which of the shapes above the engine gives a self-name record: a name a parameter or a `var`
/// shadows still has one (the query is syntactic and conservative, `bytecode::fn_statements`'
/// watched-name scan), which the code never reads; an unread name has none.
#[test]
fn a_shadowed_named_function_expression_keeps_an_unread_self_name_record() {
    if std::env::var("TRUST_JS_ELIDE_SELF_ENV").as_deref() == Ok("0") {
        return;
    }
    TestSwitches::default().set();
    let mut engine = Engine::new();
    eval(
        &mut engine,
        "var p = function me(me) { return me; }; \
         var v = function me(x) { var me = x; return me; }; \
         var u = function me(x) { return x; }; \
         var r = function me(x) { return me; }; 0",
    );
    let mut record = |name: &str| {
        let env = engine.interp.global_env;
        let value = engine
            .interp
            .get_var(name, &env)
            .unwrap_or_else(|_| panic!("the closure {name}"));
        let object = value.as_obj().expect("an object");
        let borrowed = object.borrow();
        let crate::value::ObjectKind::User(user) = &borrowed.kind else {
            panic!("a user function")
        };
        let has = user.env.borrow().vars.get("me").is_some();
        assert_eq!(has, user.func.needs_self_environment(), "{name}");
        has
    };
    assert!(record("p"), "shadowed by a parameter: a record");
    assert!(record("v"), "shadowed by a var: a record");
    assert!(!record("u"), "never read: no record");
    assert!(record("r"), "read: a record");
}

/// Whether `frame` is `of` or nested in it.
fn in_frame(
    graph: &super::ir::Graph,
    mut frame: super::ir::FrameIx,
    of: super::ir::FrameIx,
) -> bool {
    loop {
        if frame == of {
            return true;
        }
        match graph.inlined(frame) {
            Some(inlined) => frame = graph.state(inlined.parent).frame,
            None => return false,
        }
    }
}

/// `graph`'s allocation: every node of each by-literal frame lies inside its callee's range,
/// whose home is callee-saved or a spill slot across each node that calls. The frames' callees,
/// and the blocks of the frames' nodes that call.
fn assert_callees_homed(
    graph: &super::ir::Graph,
) -> (Vec<super::ir::NodeIx>, Vec<super::ir::BlockIx>) {
    use super::ir::FrameIx;
    use super::regalloc::{Home, ARM64};
    let alloc = super::regalloc::allocate(graph, &ARM64).expect("the graph allocates");
    let (mut callees, mut calls) = (Vec::new(), Vec::new());
    for (k, frame) in graph.frames.iter().enumerate() {
        if !frame.by_literal {
            continue;
        }
        let of = FrameIx(k as u32 + 1);
        let callee = frame.callee;
        callees.push(callee);
        let end = alloc.ranges[callee.index()].1;
        let home = alloc.homes[callee.index()];
        for &b in &alloc.order {
            for &node in &graph.block(b).body {
                if !in_frame(graph, graph.node(node).frame, of) {
                    continue;
                }
                let at = alloc.pos[node.index()];
                assert!(
                    end > at,
                    "the callee's range ends at {end} (its home {home:?}), a node of its frame \
                     ({:?}, b{}) is at {at}",
                    graph.node(node).op,
                    b.0
                );
                if graph.node(node).op.is_effectful() {
                    calls.push(b);
                    assert!(
                        matches!(home, Home::Gp(r) if ARM64.gp_callee.contains(&r))
                            || matches!(home, Home::Spill(_)),
                        "the callee's home across a call: {home:?}"
                    );
                }
            }
        }
    }
    assert!(!callees.is_empty(), "a by-literal frame built");
    assert!(!calls.is_empty(), "a node of the frame that calls");
    (callees, calls)
}

/// `graph` with every read of `callees` outside their by-literal frames taken away (an input
/// another value, the frame's receiver; a state that holds one dropped): the generic call the code
/// takes where the table has no entry reads the callee, as its input and in its state (the
/// helper's slot), after the frame in code order (`regalloc::reverse_postorder` visits a branch's
/// last successor first), which alone keeps it across the frame in today's graphs.
fn without_outside_reads(graph: &mut super::ir::Graph, callees: &[super::ir::NodeIx]) {
    use super::ir::FrameIx;
    let mut replaced = 0;
    for k in 0..graph.nodes.len() {
        let frame = graph.nodes[k].frame;
        let outside =
            graph.frames.iter().enumerate().all(|(f, data)| {
                !data.by_literal || !in_frame(graph, frame, FrameIx(f as u32 + 1))
            });
        if !outside {
            continue;
        }
        for input in 0..graph.nodes[k].inputs.len() {
            if let Some(&callee) = callees.iter().find(|&&c| c == graph.nodes[k].inputs[input]) {
                let receiver = graph
                    .frames
                    .iter()
                    .find(|data| data.callee == callee)
                    .expect("its frame")
                    .receiver;
                graph.nodes[k].inputs[input] = receiver;
                replaced += 1;
            }
        }
        if let Some(state) = graph.nodes[k].state {
            let mut holds = false;
            graph.for_each_exit_value(state, |v| holds |= callees.contains(&v));
            if holds {
                graph.nodes[k].state = None;
                replaced += 1;
            }
        }
    }
    assert!(replaced > 0, "the generic call's read of the callee");
}

/// The allocation of a global function `name` of `source`, run first by template code.
fn graph_of(source: &str, name: &str) -> super::ir::Graph {
    let mut engine = super::inline_tests::warmed(source);
    super::inline_tests::Compile::new(&mut engine, name)
        .build()
        .graph
}

/// Point B's rule (the coordinator's review, 2026-11-18), read in the allocation: a by-literal
/// frame's callee has a home across every node of its frame (`regalloc`'s by-literal extents),
/// a callee-saved register or a spill slot across each node that calls, so that a helper the
/// frame's last node runs, which collects and then reads the stack through the frame-callee
/// table, finds the closure alive; in the graph as built, and with every read of the callee
/// outside its frame taken away, where the rule alone must keep it (the rule's mutation is caught
/// there only). The shape has nothing after the frame's last call to keep the closure: the
/// helper's `return f(a)`, the arrow's `return probe(...)` on two arms.
#[test]
fn a_by_literal_frames_callee_has_a_home_across_every_node_of_its_frame() {
    const SOURCE: &str = "function helper(f, a) { return f(a); }\n\
         function probe(v) { return new Error('x').stack.length + v; }\n\
         function W(n, k) { return helper((x) => (x > 1 ? probe(x + k) : probe(k - x)), n); }\n\
         for (var i = 0; i < 30; i++) W(i % 4, i % 3);";
    let mut graph = graph_of(SOURCE, "W");
    let (callees, _) = assert_callees_homed(&graph);
    without_outside_reads(&mut graph, &callees);
    assert_callees_homed(&graph);
}

/// What the rule's one read after the frame's last node pins (the coordinator, 2026-11-18): it
/// covers an arm of a branching frame that is not on a path to that node only because a home is
/// held over a value's whole range in code order. A frame that branches into two arms, each with
/// a call of its own: every node of both arms lies in the callee's range, with the callee's reads
/// outside the frame taken away, so that an allocator that learns holes in ranges or splits them
/// fails here instead of silently uncovering an arm.
#[test]
fn a_branching_by_literal_frames_callee_has_a_home_across_both_arms() {
    const SOURCE: &str = "function helper(f, a) { return f(a); }\n\
         function probe(v) { return new Error('x').stack.length + v; }\n\
         function probe2(v) { return new Error('y').stack.length - v; }\n\
         function W(n, k) { return helper((x) => (x > 0 ? probe(x + k) : probe2(k - x)), n); }\n\
         for (var i = 0; i < 30; i++) W(i % 4 - 1, i % 3);";
    let mut graph = graph_of(SOURCE, "W");
    let (callees, calls) = assert_callees_homed(&graph);
    let arms: std::collections::BTreeSet<_> = calls.iter().map(|b| b.0).collect();
    assert!(
        arms.len() >= 2,
        "a call on each arm, in blocks of their own: {calls:?}"
    );
    without_outside_reads(&mut graph, &callees);
    assert_callees_homed(&graph);
}

/// The coordinator's point C: a throw out of a by-literal frame that no handler in its chain
/// takes leaves the code through its throw epilogue, which clears the activation's entry
/// (`a64::Gen::emit_throw_clear`): after `try { W() } catch {}` the table is clear, for a throw
/// from a generic node of the frame (`in` of a Number) and one from a function the frame calls;
/// and the next activation at that index finds its entry clear (test builds trap otherwise, at
/// FrameIndex), as the next collection's trace does (debug builds assert it).
#[test]
fn a_throw_out_of_a_by_literal_frame_clears_its_entry() {
    const SOURCE: &str = "function helper(f, a) { return f(a); }\n\
         function thrower(x) { if (x === 3) throw new Error('t'); return x; }\n\
         function W(n, k) { return helper((x) => thrower(x) + k, n); }\n\
         function V(o, k) { return helper((x) => ('p' in x ? 1 : 2) + k, o); }\n\
         const objects = [{ p: 1 }, { q: 1 }, { p: 1, r: 2 }];\n\
         let r = 0, c = 0; for (let i = 0; i < 3000; i++) { \
         try { r += W(i % 5, i % 3); } catch (e) { c++; check(); } \
         try { r += V(i % 50 === 49 ? 7 : objects[i % 3], i % 3); } catch (e) { c += 100; check(); } } \
         r + ':' + c";
    thread_local! {
        static CLEAR: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    }
    fn install(engine: &mut Engine) {
        CLEAR.with(|clear| clear.set(true));
        engine
            .interp
            .def_method(&engine.interp.global, "check", 0, |interp, _, _| {
                let at = interp as *const crate::interpreter::Interp as usize;
                let clear = (0..crate::stack_trace::FRAME_CALLEE_WORDS)
                    .all(|d| crate::stack_trace::frame_callee_word(at, d) == 0);
                CLEAR.with(|all| all.set(all.get() && clear));
                Ok(crate::value::Value::Undefined)
            });
    }
    let expected = {
        TestSwitches::default().set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Bytecode);
        install(&mut engine);
        eval(&mut engine, SOURCE)
    };
    assert!(expected.ends_with(":6600"), "both throw: {expected}");
    for every in [0, 7] {
        TestSwitches {
            on: true,
            eager: true,
            inline_graph: Some(true),
            exit_every: every,
            ..Default::default()
        }
        .set();
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        install(&mut engine);
        literal_frames();
        assert_eq!(eval(&mut engine, SOURCE), expected, "exit every {every}");
        assert!(literal_frames() > 0, "a by-literal frame built");
        assert!(
            CLEAR.with(std::cell::Cell::get),
            "the table clear in each catch, exit every {every}"
        );
        assert!(table_clear(&engine), "an entry left after the run");
    }
}

/// What a frame of this engine takes from its callee object or its creation, and what a
/// by-literal frame does about each (the coordinator's review, point A's list): each shape's
/// result is the tier off's at graph, and its literal is built by literal (None) or refused for
/// the reason given, by the planner's shared rules (`bytecode::inline_callee_rules`,
/// `bytecode::inline_capability`) or the oracle's own filters (`oracle::Prep::literal`); a
/// literal no `Op::MakeClosure` makes (a class) is no candidate at all (Some("")). The own name
/// is `a_named_function_expressions_own_name_is_its_by_literal_frames_callee`'s.
#[test]
fn what_a_by_literal_frame_takes_from_its_closure() {
    let lexical = Some("arrow lexical observations");
    let params = Some("param shape");
    let shape = Some("callee size/shape");
    let context = Some("callee operation requires an unremapped call context");
    let wrap = |literal: &str| {
        format!(
            "function helper(f, a) {{ return f(a); }}\n\
             function W(n, k) {{ return helper({literal}, n); }}\n\
             let r = 0; for (let i = 0; i < 4000; i++) {{ try {{ const v = W(i % 4, i % 3); \
             r += typeof v === 'number' ? v : (typeof v).length; }} catch (e) {{ r += 1000; }} }} r"
        )
    };
    let cases: Vec<(String, Option<&str>)> = vec![
        // [[Environment]]: the free name through the creation's record.
        (wrap("(x) => x + k"), None),
        // Lexical this, arguments and new.target of an arrow, and a direct eval in one.
        (wrap("(x) => (this === undefined ? x : 0) + k"), lexical),
        (wrap("(x) => arguments.length + x + k"), lexical),
        (wrap("(x) => (new.target === undefined ? x : 0) + k"), lexical),
        (wrap("(x) => eval('x + k')"), lexical),
        // A non-arrow's own this (sloppy: the global object), arguments and new.target.
        (
            wrap("function (x) { return (this === undefined ? 1 : 2) + x + k; }"),
            Some("a `this` no guard tests"),
        ),
        (wrap("function (x) { return arguments.length + x + k; }"), shape),
        (
            wrap("function (x) { return (new.target === undefined ? x : 0) + k; }"),
            context,
        ),
        // A parameter scope: a default, a rest, a pattern.
        (wrap("(x = 1) => x + k"), params),
        (wrap("(...xs) => xs.length + k"), params),
        (wrap("({ length }) => length + k"), params),
        // Strictness against the helper's.
        (wrap("(x) => { 'use strict'; return x + k; }"), Some("strictness")),
        // The legacy caller reflection.
        (
            wrap("function f2(x) { return (f2.caller ? 1 : 0) + x + k; }"),
            Some("legacy caller/arguments reflection"),
        ),
        // Generator and async kinds.
        (wrap("function* (x) { yield x + k; }"), shape),
        (wrap("async (x) => x + k"), shape),
        // A class: made by class evaluation, never Op::MakeClosure.
        (wrap("class { }"), Some("")),
        // A home object: `super` in an arrow of a method, through the method's home record.
        (
            "function helper(f, a) { return f(a); }\n\
             const o = { __proto__: { v: 5 }, W(n, k) { return helper((x) => super.v + x + k, n); } };\n\
             let r = 0; for (let i = 0; i < 4000; i++) r += o.W(i % 4, i % 3); r"
                .to_string(),
            lexical,
        ),
    ];
    for (source, refused) in &cases {
        let expected = tier_off(source);
        literal_frames();
        literal_refusals();
        assert_eq!(graph(source), expected, "{source}");
        let (frames, refusals) = (literal_frames(), literal_refusals());
        match refused {
            None => assert!(frames > 0, "a by-literal frame: {source}"),
            Some("") => assert!(frames == 0 && refusals.is_empty(), "no candidate: {source}"),
            Some(reason) => {
                assert_eq!(frames, 0, "refused: {source}");
                assert!(refusals.contains(reason), "{reason}: {refusals:?} {source}");
            }
        }
    }
}
