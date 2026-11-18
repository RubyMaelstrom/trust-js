//! Calls from borrowed frames (W3's item 4): a loop the tree-walker or the bytecode VM moved into
//! native code (a loop fragment, an OSR'd Script loop) and a coroutine's resumed slices make
//! their calls, constructions, getter reads and setter stores through the direct sequence, whose
//! callee runs on a record of its own. The frame's continuation state comes through every call
//! as it was: an abrupt completion from the callee reaches the handlers the frame borrowed from
//! the VM (ECMA-262 §14.15.3 TryStatement evaluation; §27.5.3.3 GeneratorResume), a return's
//! value passes the finalizers that make calls (§14.15.3 step 3, UpdateEmpty), and a generator
//! the callee re-enters stays running (§27.5.3.2 GeneratorValidate step 4). Every tier agrees;
//! the expected values are Node v24's.
//!
//! On ARM64 the named callees complete through the sequence (their chunks' direct return and
//! throw counters) while the frame is borrowed (its native entries or slices are counted).

use crate::bytecode::Tier;
use crate::Engine;

use super::traced_tests::eval;

/// Where a case's calls run in native code: a non-resumable borrowed frame (a loop fragment or
/// an OSR'd loop) or a coroutine's slices.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Borrowed {
    Loop,
    Slices,
}

struct Case {
    source: &'static str,
    expected: &'static str,
    borrowed: Borrowed,
    /// Global functions called only from borrowed frames, which must complete through the direct
    /// sequence on ARM64.
    direct: &'static [&'static str],
}

/// A fresh engine at `tier` with `collect()`. Functions tier up after 32 calls, so a function
/// called once stays with the tree-walker, which moves its hot loop into a native fragment.
fn engine(tier: Tier) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(32);
    let collect = engine.interp.make_native("collect", 0, |interp, _, _| {
        interp.gc_collect();
        Ok(crate::value::Value::Undefined)
    });
    engine.interp.global.borrow_mut().props.insert(
        "collect",
        crate::value::Property::data(crate::value::Value::Obj(collect), true, true, true),
    );
    engine
}

fn check(case: &Case) {
    for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        super::TEST_OSR_ENTRIES.with(|n| n.set(0));
        super::TEST_NATIVE_SLICES.with(|n| n.set(0));
        assert_eq!(
            eval(&mut engine, case.source),
            case.expected,
            "{tier:?}: {}",
            case.source
        );
        #[cfg(target_arch = "aarch64")]
        if tier == Tier::Jit {
            let entries = match case.borrowed {
                Borrowed::Loop => super::TEST_OSR_ENTRIES.with(|n| n.get()),
                Borrowed::Slices => super::TEST_NATIVE_SLICES.with(|n| n.get()),
            };
            assert!(entries > 0, "no borrowed native frame ran: {}", case.source);
            for name in case.direct {
                let completions = direct_completions(&mut engine, name);
                assert!(
                    completions > 0,
                    "{name} never completed through the direct sequence: {}",
                    case.source
                );
            }
        }
    }
}

/// The completions of the global function `name` through the direct sequence, into any of its
/// chunks: with the optimizing tier on, a hot callee's second stage (the tier's chunk, though it
/// splices nothing) takes the calls its first stage took.
#[cfg(target_arch = "aarch64")]
fn direct_completions(engine: &mut Engine, name: &str) -> usize {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("bound name");
    let object_handle = value.as_obj().expect("a function object");
    let object = object_handle.borrow();
    let crate::value::ObjectKind::User(user) = &object.kind else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    [runtime.code.get(), runtime.code2.get(), runtime.code3.get()]
        .into_iter()
        .flatten()
        .flatten()
        .map(|chunk| chunk.test_direct_returns.get() + chunk.test_direct_throws.get())
        .sum()
}

#[test]
fn a_loop_fragment_enters_every_callee_form_directly() {
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
var obj = { v: 1, get(k) { return this.v + k; } };
function mk(x) { return function (y) { return x + y; }; } var add3 = mk(3);
function envCallee(a) { var t = a + 1; if (a < 0) return () => t; return t; }
function rest(...xs) { return xs.length + xs[0]; }
function args() { return arguments.length + arguments[0]; }
function mapped(a, b) { return arguments.length + arguments[1] + a; }
function cls(k) { 'use strict'; class A { constructor() { this.v = k; } } return new A().v; }
function inc(k) { return k + 1; }
function tail(k) { 'use strict'; return inc(k); }
function forms(n) {
  var s = 0;
  for (var k = 0; k < n; k++) {
    s = (s + add(s, k) + obj.get(k) + add3(k) + envCallee(k) + rest(k, 2) + args(k, 1, 2) + mapped(k, 5) + tail(k)) | 0;
    if ((k & 3) === 0) s = (s + cls(k)) | 0;
  }
  return s;
}
forms(3000)"#,
        expected: "324483014",
        borrowed: Borrowed::Loop,
        direct: &[
            "add",
            "add3",
            "envCallee",
            "rest",
            "args",
            "mapped",
            "cls",
            "tail",
        ],
    });
}

#[test]
fn a_script_loop_enters_its_callees_directly() {
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
var obj = { v: 1, get(k) { return this.v + k; } };
function mk(x) { return function (y) { return x + y; }; } var add3 = mk(3);
var s = 0;
for (var k = 0; k < 3000; k++) { s = (s + add(s, k) + obj.get(k) + add3(k)) | 0; }
s"#,
        expected: "-9007",
        borrowed: Borrowed::Loop,
        direct: &["add", "add3"],
    });
}

#[test]
fn a_generators_slices_enter_their_callees_directly() {
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
var obj = { v: 1, get(k) { return this.v + k; } };
function* g(n) { var s = 0; for (var k = 0; k < n; k++) { s = (s + add(s, k) + obj.get(k)) | 0; if ((k & 63) === 0) yield s; } return s; }
var it = g(3000), r, ys = 0; do { r = it.next(); if (!r.done) ys++; } while (!r.done);
[r.value, ys].join()"#,
        expected: "-6003,47",
        borrowed: Borrowed::Slices,
        direct: &["add"],
    });
}

#[test]
fn a_callees_throw_reaches_the_handlers_the_frame_borrowed() {
    // Caught in the fragment, with a finalizer that calls; escaping the fragment, with the
    // thrower's frame in the stack; caught in a generator's slices.
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
function mayThrow(k) { if (k % 97 === 0) throw new Error('t' + k); return k & 7; }
function handlers(n) {
  var s = 0, caught = 0, fin = 0;
  for (var k = 0; k < n; k++) {
    try { s = (s + mayThrow(k)) | 0; } catch (e) { caught++; s = (s + e.message.length) | 0; } finally { fin = add(fin, 1); }
  }
  return [s, caught, fin].join();
}
handlers(3000)"#,
        expected: "10536,31,3000",
        borrowed: Borrowed::Loop,
        direct: &["add", "mayThrow"],
    });
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
function boom(k, n) { if (k === n - 1) throw new Error('boom' + k); return 1; }
function escape(n) { var s = 0; for (var k = 0; k < n; k++) { s = (s + add(s, k) + boom(k, n)) | 0; } return s; }
var m; try { escape(3000); } catch (e) { m = e.message + ':' + (e.stack.indexOf('boom') >= 0); }
m"#,
        expected: "boom2999:true",
        borrowed: Borrowed::Loop,
        direct: &["add", "boom"],
    });
    check(&Case {
        source: r#"
function mayThrow(k) { if (k % 97 === 0) throw new Error('t' + k); return k & 7; }
function* gt(n) { var c = 0, y = 0; for (var k = 0; k < n; k++) { try { if (k % 64 === 0) { y++; yield mayThrow(k); } else mayThrow(k); } catch (e) { c++; } } return [c, y].join(); }
var it = gt(3000), r; do { r = it.next(); } while (!r.done);
r.value"#,
        expected: "31,47",
        borrowed: Borrowed::Slices,
        direct: &["mayThrow"],
    });
}

#[test]
fn a_return_passes_finalizers_that_call() {
    // `cls` builds a class, which materializes its frame's native activation, so its finish
    // takes the helper, which moves the completion through the caller's `ret`.
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
function cls(k) { 'use strict'; class A { constructor() { this.v = k; } } return new A().v; }
function retAfter(n) { var s = 0, t = 0; for (var k = 0; k < n; k++) { try { if (k === n - 7) return [add(s, k), t, cls(k)].join(); s = (s + add(s, k)) | 0; } finally { t = add(t, cls(1)); } } return 'no'; }
retAfter(3000)"#,
        expected: "-1,2993,2993",
        borrowed: Borrowed::Loop,
        direct: &["add", "cls"],
    });
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
function cls(k) { 'use strict'; class A { constructor() { this.v = k; } } return new A().v; }
function* retGen(n) { var s = 0; for (var k = 0; k < n; k++) { s = (s + add(s, cls(k))) | 0; if (k === n - 3) return [cls(k), add(s, 1)].join(); if ((k & 127) === 0) yield k; } }
var it = retGen(3000), r; do { r = it.next(); } while (!r.done);
r.value"#,
        expected: "2997,-2998",
        borrowed: Borrowed::Slices,
        direct: &["add", "cls"],
    });
}

#[test]
fn constructions_getters_and_setters_from_borrowed_frames() {
    check(&Case {
        source: r#"
function P(x, y) { this.x = x; this.y = y; }
class G { constructor(x) { this.x = x; } get d() { return this.x * 2; } set d(v) { this.x = v; } }
var getD = Object.getOwnPropertyDescriptor(G.prototype, 'd').get;
var setD = Object.getOwnPropertyDescriptor(G.prototype, 'd').set;
function objects(n) { var s = 0, g = new G(1); for (var k = 0; k < n; k++) { var p = new P(k, 1); g.d = k; s = (s + p.y + g.d + new G(k).x) | 0; } return s; }
objects(3000)"#,
        expected: "13498500",
        borrowed: Borrowed::Loop,
        direct: &["P", "G", "getD", "setD"],
    });
    check(&Case {
        source: r#"
function P(x, y) { this.x = x; this.y = y; }
class G { constructor(x) { this.x = x; } get d() { return this.x * 2; } set d(v) { this.x = v; } }
var getD = Object.getOwnPropertyDescriptor(G.prototype, 'd').get;
var setD = Object.getOwnPropertyDescriptor(G.prototype, 'd').set;
function* objGen(n) { var s = 0, g = new G(1); for (var k = 0; k < n; k++) { var p = new P(k, 2); g.d = k; s = (s + p.y + g.d) | 0; if ((k & 127) === 0) yield; } return s; }
var it = objGen(3000), r; do { r = it.next(); } while (!r.done);
r.value"#,
        expected: "9003000",
        borrowed: Borrowed::Slices,
        direct: &["P", "getD", "setD"],
    });
}

#[test]
fn borrowed_frames_recurse_past_the_stack_limit() {
    check(&Case {
        source: r#"
function deep(n) { return n === 0 ? 0 : 1 + deep(n - 1); }
function recursion(n) { var s = 0, e = false; for (var k = 0; k < n; k++) { s = (s + deep(k & 15)) | 0; } try { deep(1e6); } catch (x) { e = x instanceof RangeError; } return [s, e].join(); }
recursion(3000)"#,
        expected: "22468,true",
        borrowed: Borrowed::Loop,
        direct: &["deep"],
    });
}

#[test]
fn a_borrowed_frames_operands_survive_its_callees_collections() {
    // `keep` waits on the fragment's operand stack while `alloc` runs and collects.
    check(&Case {
        source: r#"
function alloc(k) { var a = []; for (var i = 0; i < 20; i++) a.push({ i: i, k: k }); if ((k & 255) === 0) collect(); return a.length; }
function pair(o, x) { return o.tag.length + x; }
function collecting(n) { var keep = { tag: 'kept' }, s = 0; for (var k = 0; k < n; k++) { s = (s + pair(keep, alloc(k))) | 0; } return [s, keep.tag].join(); }
collecting(3000)"#,
        expected: "72000,kept",
        borrowed: Borrowed::Loop,
        direct: &["alloc", "pair"],
    });
}

#[test]
fn a_callee_that_resumes_the_running_generator_throws() {
    check(&Case {
        source: r#"
var it; function reenter() { try { it.next(); return 'no'; } catch (e) { return e.name; } }
function* gr(n) { var o = ''; for (var k = 0; k < n; k++) { var v = reenter(); if (k === n - 1) o = v; if ((k & 63) === 0) yield k; } return o; }
it = gr(500); var r; do { r = it.next(); } while (!r.done);
r.value"#,
        expected: "TypeError",
        borrowed: Borrowed::Slices,
        direct: &["reenter"],
    });
}

#[test]
fn a_constructors_fragment_keeps_its_new_target() {
    // The tree-walker runs `Ctor` as a [[Construct]]: the fragment's calls save and restore its
    // construct state, and the arrow keeps the lexical new.target.
    check(&Case {
        source: r#"
function add(a, b) { return a + b; }
function Ctor(n) { var f = () => new.target === Ctor, bad = 0, s = 0; for (var k = 0; k < n; k++) { if (!f()) bad++; s = (s + add(s, k)) | 0; } this.r = [s, bad].join(); }
new Ctor(3000).r"#,
        expected: "-3001,0",
        borrowed: Borrowed::Loop,
        direct: &["add"],
    });
}

/// `o`, named by a word fifteen operands deep in a pooled VM buffer alone: `fill`'s loop runs in
/// a fragment whose buffer holds `o` there and goes back to the pool (`Interp::vm_pool`, which
/// clears a buffer by its length only) when the loop ends; `w` is a WeakRef to it, which keeps it
/// no longer once this job ends.
fn leave_a_stale_word(engine: &mut Engine) {
    eval(
        engine,
        r#"
var w, alive = 0;
function sink() { return 0; }
(function fill() {
    var o = { only: 'a stale word names me' }; w = new WeakRef(o);
    for (var i = 0; i < 200; i++) sink(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, o);
    return 1;
})()"#,
    );
}

/// `holder.o`, which only `holder` names, and `w`, a WeakRef to it that keeps it no longer once this
/// job ends; `sink` takes any arguments.
fn hold_an_object(engine: &mut Engine) {
    eval(
        engine,
        r#"
var holder = { o: { only: 'my own popped word names me' } }, w = new WeakRef(holder.o), alive = 0;
function sink() { return 0; }
for (var i = 0; i < 64; i++) sink(i);
1"#,
    );
}

/// A borrowed frame's own popped words above the top a direct call leaves are no roots, which the
/// clear at its run's entry cannot reach (`jit::entry_live_top` bounds them): at the 1,000th turn
/// the loop pushes `holder.o` fifteen operands deep (the arguments of a direct call to `sink`, compiled) and pops it, and
/// `holder` lets go; at the 1,023rd it waits on `probe`, whose callee collects and reports what
/// marked `o` first: no root (`R`); the native stack's conservative scan (`S`) may keep it (ARM64:
/// x86-64's calls run through Rust entries, whose top the collector does not know).
#[test]
#[cfg(target_arch = "aarch64")]
fn a_borrowed_frames_own_popped_words_are_no_roots_while_it_waits_on_a_call() {
    let mut engine = super::rooting_tests::jit_engine(32);
    super::TEST_OSR_ENTRIES.with(|n| n.set(0));
    super::rooting_tests::run(
        &mut engine,
        "var holder = { o: track({ only: 'my own popped word names me' }) };
         function sink() { return 0; }
         for (var i = 0; i < 64; i++) sink(i);
         1",
    );
    let source = r#"
function inner() { return collect(); }
function probe(k) { return k === 1023 ? inner() : ''; }
(function own() {
    var s = 0, m = '';
    for (var k = 0; k < 3000; k++) {
        if (k === 1000) { s += sink(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, holder.o); holder.o = null; }
        m += probe(k);
    }
    return m;
})()"#;
    let marker = super::rooting_tests::run(&mut engine, source);
    assert!(marker == "S" || marker == "D", "{marker}");
    assert!(
        super::TEST_OSR_ENTRIES.with(|n| n.get()) > 0,
        "no borrowed native frame ran"
    );
    assert!(
        direct_completions(&mut engine, "probe") > 0,
        "probe never completed through the direct sequence"
    );
}

/// A borrowed frame's operand buffer comes from the VM's pool, so the words past the frame's
/// operands can name what an earlier activation of the buffer pushed. While the frame waits on a
/// direct call, the collector scans its operand words up to the top that call left
/// (`jit::entry_live_top`), not the whole buffer its entry registers: an object only such a word
/// names is collected (P3 stage 3 step 3g: the Splay retention test's failure at
/// TRUST_JS_OPT_INLINE=graph, `gc_collect::tests::a_splay_trees_live_data_stays_flat_in_every_tier`,
/// where `measure`'s loop held removed nodes so).
#[test]
fn a_borrowed_frames_stale_operand_words_retain_nothing_while_it_waits_on_a_call() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        super::TEST_OSR_ENTRIES.with(|n| n.set(0));
        leave_a_stale_word(&mut engine);
        // `waits`'s loop takes that buffer and waits on `probe`, which collects.
        eval(
            &mut engine,
            r#"
function inner() { collect(); return w.deref() !== undefined; }
function probe(k) { if ((k & 255) === 255 && inner()) alive++; return 1; }
(function waits() { var s = 0; for (var k = 0; k < 3000; k++) s += probe(k); return s; })()"#,
        );
        assert_eq!(eval(&mut engine, "alive"), "0", "{tier:?}");
        #[cfg(target_arch = "aarch64")]
        if tier == Tier::Jit {
            assert!(
                super::TEST_OSR_ENTRIES.with(|n| n.get()) > 0,
                "no borrowed native frame ran"
            );
            assert!(
                direct_completions(&mut engine, "probe") > 0,
                "probe never completed through the direct sequence"
            );
        }
    }
}

/// The same while the borrowed frame waits on a native that collects, a call into Rust whose top
/// the collector does not know: the frame's run cleared the buffer's words past its entry depth
/// (`jit::run_borrowed_frame`), so no earlier user's word is left to root.
#[test]
fn a_borrowed_frames_stale_operand_words_retain_nothing_while_it_waits_on_a_native() {
    for tier in [Tier::Bytecode, Tier::Jit] {
        let mut engine = engine(tier);
        super::TEST_OSR_ENTRIES.with(|n| n.set(0));
        leave_a_stale_word(&mut engine);
        eval(
            &mut engine,
            r#"
(function waitsOnANative() {
    var s = 0;
    for (var k = 0; k < 3000; k++) { if ((k & 255) === 255) { collect(); if (w.deref() !== undefined) alive++; } s += k; }
    return s;
})()"#,
        );
        assert_eq!(eval(&mut engine, "alive"), "0", "{tier:?}");
        #[cfg(target_arch = "aarch64")]
        if tier == Tier::Jit {
            assert!(
                super::TEST_OSR_ENTRIES.with(|n| n.get()) > 0,
                "no borrowed native frame ran"
            );
        }
    }
}

/// A borrowed frame's own popped words above its poll's top (`JitCtx::poll_top`) are no roots,
/// which the clear at its run's entry cannot reach (`jit::entry_live_top` bounds them): at the
/// 1,000th turn the loop pushes `o` fifteen operands deep (the arguments of a direct call to `sink`, compiled) and pops it, and the
/// global that named it
/// lets go; from the 2,000th turn a collection is due after every allocation, at its polls.
#[test]
fn a_borrowed_frames_own_popped_words_retain_nothing_at_its_polls() {
    let mut engine = engine(Tier::Jit);
    let stress = engine.interp.make_native("stressOn", 0, |interp, _, _| {
        interp.gc_heap.core.borrow_mut().set_stress_every(1);
        Ok(crate::value::Value::Undefined)
    });
    let calm = engine.interp.make_native("stressOff", 0, |interp, _, _| {
        let every = crate::interpreter::gc_collect::gc_stress();
        interp.gc_heap.core.borrow_mut().set_stress_every(every);
        Ok(crate::value::Value::Undefined)
    });
    for (name, f) in [("stressOn", stress), ("stressOff", calm)] {
        engine.interp.global.borrow_mut().props.insert(
            name,
            crate::value::Property::data(crate::value::Value::Obj(f), true, true, true),
        );
    }
    super::TEST_OSR_ENTRIES.with(|n| n.set(0));
    hold_an_object(&mut engine);
    super::take_poll_census();
    eval(
        &mut engine,
        r#"
(function polls() {
    var o = null, s = 0;
    for (var k = 0; k < 2200; k++) {
        if (k === 1000) { s += sink(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, holder.o); holder.o = null; }
        if (k === 2000) stressOn();
        o = { k: k };
    }
    stressOff();
    return o.k;
})()"#,
    );
    let census = super::take_poll_census();
    assert_eq!(eval(&mut engine, "w.deref() === undefined"), "true");
    #[cfg(target_arch = "aarch64")]
    {
        assert!(
            super::TEST_OSR_ENTRIES.with(|n| n.get()) > 0,
            "no borrowed native frame ran"
        );
        assert!(
            census.collections.iter().sum::<u64>() > 0,
            "no collection at a poll: {census:?}"
        );
    }
    let _ = census;
}

/// A borrowed frame's top operand word is a root while the frame waits on a direct call that
/// collects: `inner` takes no parameter, so the object `track` follows stays on the frame's operand
/// stack alone, as its top word, below the top the call left (`jit::entry_live_top`; the record's
/// form is jit_rooting_tests' a_waiting_callers_top_operand_word_is_a_root).
#[test]
fn a_borrowed_frames_top_operand_word_is_a_root_while_it_waits_on_a_call() {
    let mut engine = super::rooting_tests::jit_engine(32);
    super::TEST_OSR_ENTRIES.with(|n| n.set(0));
    let source = r#"
function inner() { return collect(); }
(function waits() {
    var bad = '';
    for (var k = 0; k < 600; k++) { var m = inner(track({ tag: 'T' + k })); if (m !== 'R') bad += k + ':' + m + ' '; }
    return bad;
})()"#;
    assert_eq!(super::rooting_tests::run(&mut engine, source), "");
    #[cfg(target_arch = "aarch64")]
    {
        assert!(
            super::TEST_OSR_ENTRIES.with(|n| n.get()) > 0,
            "no borrowed native frame ran"
        );
        assert!(
            direct_completions(&mut engine, "inner") > 0,
            "inner never completed through the direct sequence"
        );
    }
}

/// A borrowed frame's for-of keeps its iterator across the loop's polls, at which a collection is
/// due after every allocation from the 2,000th turn on: the iterator, which nothing else names, is
/// a root there, and the loop sees every value. `marks()` at the last turn's start reports the
/// previous turn's last collection, which ran at the loop's poll. The iterator and its `next`
/// live in the frame's hidden slots (`%iter%`; `emit_for_of_step`), which `drive_vm` registers,
/// not in its operand words: the poll's top (`JitCtx::poll_top`) does not bound them.
#[test]
fn a_borrowed_frames_iterator_is_a_root_at_its_polls() {
    let mut engine = super::rooting_tests::jit_engine(32);
    // Every stress collection a major one: the iterator is old by then, which a minor one does
    // not mark.
    let stress = engine.interp.make_native("stressOn", 0, |interp, _, _| {
        let mut core = interp.gc_heap.core.borrow_mut();
        core.set_stress_every(1);
        core.set_stress_major_every(1);
        Ok(crate::value::Value::Undefined)
    });
    let calm = engine.interp.make_native("stressOff", 0, |interp, _, _| {
        let mut core = interp.gc_heap.core.borrow_mut();
        core.set_stress_every(crate::interpreter::gc_collect::gc_stress());
        core.set_stress_major_every(crate::interpreter::gc_collect::gc_stress_major());
        Ok(crate::value::Value::Undefined)
    });
    for (name, f) in [("stressOn", stress), ("stressOff", calm)] {
        engine.interp.global.borrow_mut().props.insert(
            name,
            crate::value::Property::data(crate::value::Value::Obj(f), true, true, true),
        );
    }
    super::TEST_OSR_ENTRIES.with(|n| n.set(0));
    super::take_poll_census();
    let source = r#"
(function iterates() {
    var iterable = { [Symbol.iterator]() {
        return track({ i: 0, next() { return this.i < 2200 ? { value: this.i++, done: false } : { value: 0, done: true }; } });
    } };
    var sum = 0, o = null, last = '';
    for (var v of iterable) { if (v === 2199) last = marks(); if (v === 2000) stressOn(); o = { v: v }; sum += v; }
    stressOff();
    return sum + ':' + last;
})()"#;
    assert_eq!(super::rooting_tests::run(&mut engine, source), "2418900:R");
    let census = super::take_poll_census();
    #[cfg(target_arch = "aarch64")]
    {
        assert!(
            super::TEST_OSR_ENTRIES.with(|n| n.get()) > 0,
            "no borrowed native frame ran"
        );
        assert!(
            census.collections.iter().sum::<u64>() > 0,
            "no collection at a poll: {census:?}"
        );
    }
    let _ = census;
}

/// A Rust entry's slots are roots whatever its operand top: `holder`, run through a native
/// callback on an entry of its own (`gc_trace::JitFrameRoots`: its slots and its operand words),
/// waits on a direct call that collects, with the object `track` follows in its ninth slot alone,
/// above the top of its operand words (`jit::entry_live_top` cuts those alone).
#[test]
fn a_rust_entrys_slots_are_roots_while_it_waits_on_a_call() {
    let mut engine = super::rooting_tests::jit_engine(0);
    let source = r#"
function inner() { return collect(); }
function holder(n) {
    var a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8;
    var keep = track({ tag: 'K' + n });
    var m = inner();
    return m + (keep.tag === 'K' + n ? '' : '!') + (a + b + c + d + e + f + g + h === 36 ? '' : '?');
}
var out = '';
for (var k = 0; k < 40; k++) out += [k].map(holder)[0];
out"#;
    assert_eq!(
        super::rooting_tests::run(&mut engine, source),
        "R".repeat(40)
    );
}

/// The engine of the end-poll cases below. `armOwn()` arms the call templates' end-poll request
/// for its own call's completion and `armNext()` for the next call's after it
/// (`jit::arm_call_end_poll_limit_at`, templates emitted with `jit::set_call_done_hook`): that
/// call's end poll runs a major collection, its `gc_check` finding the ceiling passed, and throws
/// the live-object limit's RangeError. `entryCount()` counts the Rust entries registered
/// (`gc_trace::JitFrameRoots`). No GC stress and no incremental marking: `marks()` reports that
/// collection.
#[cfg(target_arch = "aarch64")]
fn end_poll_engine() -> Engine {
    use crate::value::Value;
    let mut engine = super::rooting_tests::jit_engine(0);
    {
        let mut core = engine.interp.gc_heap.core.borrow_mut();
        core.set_stress_every(0);
        core.set_incremental(crate::gc_core::IncrementalMode::Off);
    }
    super::rooting_tests::install(&mut engine, "armOwn", 0, |_, _, _| {
        super::arm_call_end_poll_limit_at(1, false);
        Ok(Value::Undefined)
    });
    super::rooting_tests::install(&mut engine, "armNext", 0, |_, _, _| {
        super::arm_call_end_poll_limit_at(2, false);
        Ok(Value::Undefined)
    });
    super::rooting_tests::install(&mut engine, "entryCount", 0, |_, _, _| {
        Ok(Value::Num(
            crate::gc_trace::jit_frames_registered().len() as f64
        ))
    });
    super::set_call_done_hook(true);
    engine
}

/// Runs `source` on [`end_poll_engine`], checking that the armed request fell due.
#[cfg(target_arch = "aarch64")]
fn end_poll_case(source: &str) -> String {
    let mut engine = end_poll_engine();
    let got = super::rooting_tests::run(&mut engine, source);
    super::set_call_done_hook(false);
    assert!(
        !super::take_end_poll_limit(super::POLL_HELPER_END),
        "the request fell due and no poll took it: {got}"
    );
    got
}

/// A Rust entry's top operand word is a root at its poll's top (`JitCtx::poll_top`,
/// `jit::entry_live_top`): `f`, which `invoke` runs on an entry of its own (`jit::run`; one more
/// entry than the script's), pushes the object `track` returns as `id`'s argument, and the end
/// poll after `track`'s call collects with that object in the top operand word alone (the
/// words of `track`'s argument lie above the top). `marks()` reports a root marking it (`R`), not
/// the native stack's conservative scan (`S`) nor nothing (`D`).
#[test]
#[cfg(target_arch = "aarch64")]
fn a_rust_entrys_top_operand_word_is_a_root_at_its_polls() {
    let got = end_poll_case(
        r#"
function id(o) { return o; }
function f() {
    try { armNext(); return id(track({ tag: 'T' })) ? 'none' : 'none'; }
    catch (e) { return e.name + ':' + marks() + ':' + (entryCount() - outer); }
}
var outer = entryCount();
invoke(f)"#,
    );
    assert_eq!(got, "RangeError:R:1");
}

/// A Rust entry's slots are roots whatever its operand top, its poll's too: `g`, which `invoke`
/// runs on an entry of its own, keeps the object `track` follows in its ninth slot alone and
/// collects at the end poll of `armOwn`'s call, whose top is one operand word, below that slot's
/// index (`jit::entry_live_top` cuts the operand words alone): `marks()` reports a root marking
/// it (`R`).
#[test]
#[cfg(target_arch = "aarch64")]
fn a_rust_entrys_slots_past_its_poll_top_are_roots() {
    let got = end_poll_case(
        r#"
function g() {
    var a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, h = 7, k = 8;
    var keep = track({ tag: 'K' });
    try { armOwn(); return 'none'; }
    catch (x) {
        // `keep` is read only where it was marked: an object the collection missed is freed.
        var m = marks();
        return x.name + ':' + m + ':' + (entryCount() - outer)
            + (a + b + c + d + e + f + h + k === 36 && (m !== 'R' || keep.tag === 'K') ? '' : '?');
    }
}
var outer = entryCount();
invoke(g)"#,
    );
    assert_eq!(got, "RangeError:R:1");
}
