//! The collector roots the native JIT's activations (P1 S5 step 7; gc-heap, ARM64), with
//! collections on and, in debug builds, the heap verifier after every marking.
//!
//! Each program keeps values in one kind of JIT storage and forces a collection from inside the
//! activation: the test native `collect()`, called from generated code, runs the explicit
//! collection. `track(v)` hands an object's weak handle to the test; `collect()` then answers,
//! for each object tracked since its last call, what marked it first: `R` a root, `S` something
//! else (the native stack, a cell), `D` nothing (it died). Roots are marked before the native
//! stack is scanned, so a stale stack copy of a value cannot stand in for the JIT root under
//! test; nothing but a JIT activation (and in (9) the call's FnFrame) holds a tracked object as
//! a root in these programs. The programs then read their values back. `collect()` also records the JIT roots registered at
//! the time (active records, Rust entries' word ranges) and the same marker for the environment
//! of every materialized native activation, and each test checks that its program ran natively
//! the way it is meant to. Once a program's activations have finished, a collection from the
//! test (its stack scrubbed, see `gc_test`) must find every object it tracked dead.

use crate::bytecode::Tier;
use crate::interpreter::Interp;
use crate::value::{Gc, Property, Value, WeakGc};
use crate::Engine;
use std::cell::RefCell;

use super::traced_tests::{assert_native, eval, settle, with_chunk};

thread_local! {
    /// The objects tracked since the last `collect()`.
    static PENDING: RefCell<Vec<WeakGc>> = const { RefCell::new(Vec::new()) };
    /// Every object tracked by the current test.
    static TRACKED: RefCell<Vec<WeakGc>> = const { RefCell::new(Vec::new()) };
    /// What each `collect()` found registered.
    static CENSUS: RefCell<Vec<Census>> = const { RefCell::new(Vec::new()) };
}

/// The JIT roots registered while a `collect()` ran.
#[derive(Clone, Debug, Default)]
struct Census {
    /// Active records of the collecting interpreter.
    records: usize,
    /// Each Rust entry's registered activation: its slot and operand word counts.
    entries: Vec<[usize; 2]>,
    /// A marker for the environment of each materialized native activation of a running JIT
    /// context.
    activations: String,
}

/// `track(value)`: the test follows an object through a weak handle; returns `value`.
pub(super) fn track(_: &mut Interp, _: Value, args: &[Value]) -> Result<Value, Value> {
    let value = args.first().cloned().unwrap_or(Value::Undefined);
    if let Value::Obj(object) = &value {
        let weak = Gc::downgrade(object);
        PENDING.with(|pending| pending.borrow_mut().push(weak.clone()));
        TRACKED.with(|tracked| tracked.borrow_mut().push(weak));
    }
    Ok(value)
}

/// `collect()`: an explicit collection from inside the calling activation, then one marker per
/// object tracked since the last call (see the module documentation), or "deferred".
fn collect(i: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    let before = collections(&i.gc_heap);
    i.gc_collect();
    if collections(&i.gc_heap) == before {
        return Ok(Value::str("deferred"));
    }
    let census = Census {
        records: i.frame_pool.active_records().len(),
        entries: crate::gc_trace::jit_frames_registered(),
        activations: activation_envs(i)
            .into_iter()
            .map(|address| marker(&i.gc_heap, address))
            .collect(),
    };
    CENSUS.with(|all| all.borrow_mut().push(census));
    let pending = PENDING.with(|pending| std::mem::take(&mut *pending.borrow_mut()));
    let marks: String = pending
        .iter()
        .map(|weak| weak_marker(&i.gc_heap, weak))
        .collect();
    Ok(Value::str(marks))
}

/// `marks()`: for each object tracked since the last `collect()` or `marks()`, what marked it
/// first in the most recent collection, whoever ran it (a poll of generated code, say), without
/// collecting: the markers of `collect()`.
pub(super) fn marks(i: &mut Interp, _: Value, _: &[Value]) -> Result<Value, Value> {
    let pending = PENDING.with(|pending| std::mem::take(&mut *pending.borrow_mut()));
    let marks: String = pending
        .iter()
        .map(|weak| weak_marker(&i.gc_heap, weak))
        .collect();
    Ok(Value::str(marks))
}

fn collections(heap: &crate::value::GcHeap) -> u64 {
    heap.core.borrow().stats().collections
}

/// The environment cells of the materialized native activations of every running JIT context:
/// `i`'s active records and this thread's registered Rust entries.
fn activation_envs(i: &Interp) -> Vec<usize> {
    let mut contexts: Vec<*const super::JitCtx> = i
        .frame_pool
        .active_records()
        .into_iter()
        // SAFETY: an active record is allocated.
        .map(|record| unsafe { std::ptr::addr_of!((*record.as_ptr()).ctx) })
        .collect();
    contexts.extend(crate::gc_trace::jit_frame_contexts());
    contexts
        .into_iter()
        // SAFETY: a running context outlives a call made from inside it.
        .filter_map(|ctx| unsafe { (*ctx).activation.as_ref() }.map(|a| a.env_address()))
        .collect()
}

/// What first marked the cell at `address` in the last collection (see the module
/// documentation).
fn marker(heap: &crate::value::GcHeap, address: usize) -> char {
    match heap.core.borrow().retention_path(address).first() {
        Some((_, _, crate::gc_core::MarkSource::Root)) => 'R',
        Some(_) => 'S',
        None => 'D',
    }
}

/// [`marker`] for `weak`'s referent, asked in a frame of its own (see `gc_test::alive`).
#[inline(never)]
fn weak_marker(heap: &crate::value::GcHeap, weak: &WeakGc) -> char {
    use crate::gc_test::WeakCell;
    weak.live_address()
        .map_or('D', |address| marker(heap, address))
}

/// `invoke(f, ...args)`: call `f` from Rust, through the interpreter's generic call (a
/// compiled callee runs on `jit::run`).
fn invoke(i: &mut Interp, _: Value, args: &[Value]) -> Result<Value, Value> {
    let callee = args.first().cloned().unwrap_or(Value::Undefined);
    let rest = args.get(1..).unwrap_or(&[]);
    i.call(callee, Value::Undefined, rest)
        .map_err(crate::interpreter::abrupt_value)
}

pub(super) fn install(engine: &mut Engine, name: &str, length: usize, f: crate::value::NativeFn) {
    let native = engine.interp.make_native(name, length, f);
    engine
        .interp
        .global
        .borrow_mut()
        .props
        .insert(name, Property::data(Value::Obj(native), true, true, true));
}

/// An engine at `Tier::Jit` with tier threshold `threshold`, the test natives, and retention
/// paths recorded for `collect()`'s markers.
pub(super) fn jit_engine(threshold: u32) -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(threshold);
    assert_eq!(engine.interp.tier, Tier::Jit);
    engine
        .interp
        .gc_heap
        .core
        .borrow_mut()
        .set_retention_tracing(true);
    install(&mut engine, "collect", 0, collect);
    install(&mut engine, "marks", 0, marks);
    install(&mut engine, "track", 1, track);
    install(&mut engine, "invoke", 1, invoke);
    PENDING.with(|pending| pending.borrow_mut().clear());
    TRACKED.with(|tracked| tracked.borrow_mut().clear());
    CENSUS.with(|census| census.borrow_mut().clear());
    engine
}

/// Evaluate `source` in a frame of its own: no handle it leaves behind is in the test's frame.
pub(super) fn run(engine: &mut Engine, source: &str) -> String {
    crate::gc_test::in_frame(|| eval(engine, source))
}

/// What the last `collect()` found registered.
fn last_census() -> Census {
    CENSUS.with(|census| census.borrow().last().cloned().expect("collect() ran"))
}

/// Run `source` until a round returns through the direct call sequence at least `direct` times:
/// a concurrent test can exhaust the process-wide call-cache epoch, which sends calls through
/// the checked helper until their sites refill. Returns the last round's result.
pub(super) fn direct_round(engine: &mut Engine, source: &str, direct: usize) -> String {
    let mut result = String::new();
    for _ in 0..100 {
        super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.set(0));
        result = run(engine, source);
        if super::TEST_DIRECT_PACKED_RETURNS.with(|count| count.get()) >= direct {
            return result;
        }
    }
    panic!("{source}: fewer than {direct} direct returns (last result {result})");
}

/// Once its activations have finished, nothing a program tracked survives a collection: pooled
/// records, pooled buffers and finished entries hold nothing.
pub(super) fn assert_tracked_collectable(engine: &mut Engine) {
    assert_eq!(
        crate::gc_trace::jit_frames_registered().len(),
        0,
        "every Rust entry unregistered"
    );
    assert!(engine.interp.frame_pool.active_records().is_empty());
    engine.interp.gc_collect();
    TRACKED.with(|tracked| {
        let tracked = tracked.borrow();
        assert!(!tracked.is_empty());
        for (k, weak) in tracked.iter().enumerate() {
            assert!(
                !crate::gc_test::alive(weak),
                "tracked object {k} of {} outlived its frames",
                tracked.len()
            );
        }
    });
}

/// (1) A direct callee's slots: the callee's own record.
#[test]
fn direct_callee_slots_are_roots() {
    let mut engine = jit_engine(0);
    let source = r#"
        function callee(n) {
            var local = track({tag: 'C' + n});
            var marks = collect();
            return marks + ':' + local.tag;
        }
        function caller(n) { return callee(n); }
        caller(0);
    "#;
    assert_eq!(run(&mut engine, source), "R:C0");
    assert_native(&mut engine, &["callee", "caller"]);
    settle(&mut engine, "callee");
    assert_eq!(direct_round(&mut engine, "caller(1);", 1), "R:C1");
    let census = last_census();
    assert!(census.records >= 2, "caller and callee records: {census:?}");
    assert_tracked_collectable(&mut engine);
}

/// (2) A caller's operand words across a direct call: `pair`'s first argument waits on `outer`'s
/// operand stack while `inner` collects.
#[test]
fn caller_operand_words_are_roots_across_a_direct_call() {
    let mut engine = jit_engine(0);
    let source = r#"
        function inner(n) { return collect(); }
        function pair(a, b) { return b + ':' + a.tag; }
        function outer(n) { return pair(track({tag: 'O' + n}), inner(n)); }
        outer(0);
    "#;
    assert_eq!(run(&mut engine, source), "R:O0");
    assert_native(&mut engine, &["inner", "pair", "outer"]);
    settle(&mut engine, "inner");
    settle(&mut engine, "pair");
    assert_eq!(direct_round(&mut engine, "outer(1);", 2), "R:O1");
    assert!(last_census().records >= 2);
    assert_tracked_collectable(&mut engine);
}

/// (3) A record's `this` binding and return slot. The direct sequence moves the receiver into
/// the callee record's `this_val` (the caller's operand word keeps a stale copy, also scanned)
/// while the method collects; the callee's completion travels back through its record's `ret`
/// and the caller's (the finish stub moves it before any helper runs, so no collection sees it
/// there; it is traced precisely all the same) into a slot of the caller, which collects again.
#[test]
fn record_this_and_return_values_are_roots() {
    let mut engine = jit_engine(0);
    let source = r#"
        function Box(tag) { this.tag = tag; }
        function check() { var marks = collect(); return track({text: marks + ':' + this.tag}); }
        Box.prototype.check = check;
        function method(n) {
            var result = track(new Box('M' + n)).check();
            return result.text + ':' + collect();
        }
        method(0);
    "#;
    assert_eq!(run(&mut engine, source), "R:M0:R");
    assert_native(&mut engine, &["Box", "check", "method"]);
    settle(&mut engine, "check");
    assert_eq!(direct_round(&mut engine, "method(1);", 1), "R:M1:R");
    assert!(last_census().records >= 1);
    assert_tracked_collectable(&mut engine);
}

/// (4) A native activation's state (`NativeActivation`, which bridge operations materialize): the
/// environments JIT code pushed for a block and for a `catch` clause, whose bindings closures it
/// created capture. The environment cell is the activation's own root, so the collection must
/// mark it first from the activation; the objects it binds are reached through it (first marked
/// by a cell, or by the stale operand word they were stored from), and read back.
#[test]
fn native_activation_state_is_a_root() {
    let mut engine = jit_engine(0);
    let source = r#"
        function block(n) {
            var marks, read;
            for (let k = 0; k < 1; k++) {
                let held = track({tag: 'A' + n});
                read = () => held.tag;
                marks = collect();
            }
            return marks + ':' + read();
        }
        function caught(n) {
            try {
                throw track({tag: 'E' + n});
            } catch (e) {
                let read = () => e.tag;
                var marks = collect();
                return marks + ':' + read();
            }
        }
        block(0) + ',' + caught(0);
    "#;
    let first = run(&mut engine, source);
    assert!(
        ["R", "S"]
            .iter()
            .any(|a| ["R", "S"].iter().any(|e| first == format!("{a}:A0,{e}:E0"))),
        "{first}"
    );
    assert_native(&mut engine, &["block", "caught"]);
    for n in 1..3 {
        for (function, tag) in [("block", 'A'), ("caught", 'E')] {
            let result = run(&mut engine, &format!("{function}({n});"));
            let tail = format!(":{tag}{n}");
            assert!(
                result == format!("R{tail}") || result == format!("S{tail}"),
                "{function}({n}): {result}"
            );
            let census = last_census();
            assert_eq!(
                census.activations, "R",
                "{function}({n}): the activation's environment: {census:?}"
            );
        }
    }
    CENSUS.with(|census| {
        for census in census.borrow().iter() {
            assert_eq!(census.activations, "R", "{census:?}");
        }
    });
    assert_tracked_collectable(&mut engine);
}

/// (5) A Rust entry's pooled buffers (`jit::run`): the Script's own operand words (tier
/// threshold 0 runs the whole Script natively), the slots of a function a native calls back
/// (`invoke`), and a callback of `Array.prototype.map` (its first call runs on `jit::run`, the
/// prepared ones on records).
#[test]
fn rust_entry_buffers_are_roots() {
    let mut engine = jit_engine(0);
    assert_eq!(
        run(
            &mut engine,
            "(function (pair) { return pair[1] + ':' + pair[0].tag; })\
             ([track({tag: 'T'}), collect()]);",
        ),
        "R:T"
    );
    assert!(!last_census().entries.is_empty(), "the Script's entry");
    let source = r#"
        function viaRun(n) {
            var local = track({tag: 'V' + n});
            var marks = collect();
            return marks + ':' + local.tag;
        }
        function mapped(x) {
            var local = track({tag: 'P' + x});
            var marks = collect();
            return marks + ':' + local.tag;
        }
        invoke(viaRun, 1);
    "#;
    assert_eq!(run(&mut engine, source), "R:V1");
    assert_native(&mut engine, &["viaRun"]);
    let census = last_census();
    assert!(
        census.entries.len() >= 2 && census.entries.iter().any(|&[slots, _]| slots >= 3),
        "the Script's entry and viaRun's: {census:?}"
    );
    assert_eq!(
        run(&mut engine, "[1, 2, 3].map(mapped).join();"),
        "R:P1,R:P2,R:P3"
    );
    assert_native(&mut engine, &["mapped"]);
    assert_tracked_collectable(&mut engine);
}

/// (6) A frame larger than a record (`jit::run_moved_oversized`): its slots in a pooled buffer.
#[test]
fn oversized_frame_slots_are_roots() {
    let mut engine = jit_engine(0);
    let locals: Vec<String> = (0..300).map(|k| format!("v{k} = {k}")).collect();
    let source = format!(
        "function big(n) {{ var {}; var local = track({{tag: 'G' + n}}); var marks = collect(); \
         return marks + ':' + local.tag + ':' + (v0 + v299); }} \
         function callBig(n) {{ return big(n); }} callBig(0);",
        locals.join(", ")
    );
    assert_eq!(run(&mut engine, &source), "R:G0:299");
    assert_native(&mut engine, &["big", "callBig"]);
    assert_eq!(run(&mut engine, "callBig(1);"), "R:G1:299");
    let census = last_census();
    assert!(
        census.entries.iter().any(|&[slots, _]| slots >= 300),
        "big's oversized entry: {census:?}"
    );
    assert_tracked_collectable(&mut engine);
}

/// (7) Borrowed VM frames (`jit::run_borrowed_frame`): the operand words a hot Script loop's
/// native code (OSR) and a generator's native slice keep in the VM frame's stack, whose length
/// is 0 for the native run.
#[test]
fn borrowed_frame_operands_are_roots() {
    // Tier threshold 32: the Script starts on the tree-walker, and its hot loop enters native
    // code at a backedge.
    let mut engine = jit_engine(32);
    crate::jit::TEST_OSR_ENTRIES.with(|count| count.set(0));
    let source = r#"
        var osr = '';
        for (var k = 0; k < 3000; k++) {
            if (k === 2999) {
                osr = (function (pair) { return pair[1] + ':' + pair[0].tag; })
                    ([track({tag: 'L' + k}), collect()]);
            }
        }
        osr;
    "#;
    assert_eq!(run(&mut engine, source), "R:L2999");
    assert!(crate::jit::TEST_OSR_ENTRIES.with(|count| count.get()) > 0);
    let census = last_census();
    assert!(
        census
            .entries
            .iter()
            .any(|&[slots, stack]| slots == 0 && stack > 0),
        "the loop's borrowed frame: {census:?}"
    );
    assert_tracked_collectable(&mut engine);

    let mut engine = jit_engine(0);
    crate::jit::TEST_NATIVE_SLICES.with(|count| count.set(0));
    let source = r#"
        function* gen(n) {
            for (var k = 0; k < n; k++) {
                yield (function (pair) { return pair[1] + ':' + pair[0].tag; })
                    ([track({tag: 'Y' + k}), collect()]);
            }
        }
        function drain(n) { var out = []; for (var v of gen(n)) out.push(v); return out.join(); }
        drain(3);
    "#;
    assert_eq!(run(&mut engine, source), "R:Y0,R:Y1,R:Y2");
    assert!(crate::jit::TEST_NATIVE_SLICES.with(|count| count.get()) > 0);
    let census = last_census();
    assert!(
        census
            .entries
            .iter()
            .any(|&[slots, stack]| slots == 0 && stack > 0),
        "the generator's borrowed frame: {census:?}"
    );
    assert_tracked_collectable(&mut engine);
}

/// (8) Recursion deeper than the 256-record pool: every level's record, pooled or allocated
/// past the pool, is a root; afterwards the excess records are freed and unregistered.
#[test]
fn records_past_the_pool_are_roots() {
    let mut engine = jit_engine(0);
    let source = r#"
        function deep(n) {
            var held = track({tag: 'D' + n});
            var marks = n === 0 ? collect() : deep(n - 1);
            return held.tag === 'D' + n ? marks : marks + '!' + n;
        }
        deep(3);
    "#;
    assert_eq!(run(&mut engine, source), "RRRR");
    assert_native(&mut engine, &["deep"]);
    settle(&mut engine, "deep");
    assert_eq!(run(&mut engine, "deep(300);"), "R".repeat(301));
    let census = last_census();
    assert!(census.records >= 301, "a record per level: {census:?}");
    let pool = &engine.interp.frame_pool;
    assert!(pool.len() <= super::FRAME_POOL_LIMIT);
    assert_eq!(pool.registered(), pool.len(), "records past the pool freed");
    assert_tracked_collectable(&mut engine);
}

/// (9) A direct callee whose closure (and so its environment) the caller's operand stack holds
/// for the call: a fresh closure of warmed code that nothing else names. The FnFrame the direct
/// sequence pushes names it too (`FnFrame::fn_ptr`, a root).
#[test]
fn a_fresh_callee_closure_survives_its_direct_call() {
    let mut engine = jit_engine(0);
    let source = r#"
        function make(n) {
            var captured = {tag: 'K' + n};
            return function () { var marks = collect(); return marks + ':' + captured.tag; };
        }
        function fresh(n) { return track(make(n))(); }
        var proto = make(0);
        fresh(0);
    "#;
    assert_eq!(run(&mut engine, source), "R:K0");
    assert_native(&mut engine, &["make", "fresh", "proto"]);
    settle(&mut engine, "make");
    with_chunk(&mut engine, "proto", |chunk| {
        chunk.inline_attempted.set(true);
        chunk.inline_retry_at.set(0);
    });
    assert_eq!(direct_round(&mut engine, "fresh(1);", 1), "R:K1");
    assert!(last_census().records >= 2);
    assert_tracked_collectable(&mut engine);
}

/// These programs and step 6's (`jit::traced_tests`) again at `TRUST_JS_GC_STRESS=1`, which
/// collects at every safepoint that follows an allocation (the calls, polls and natives JIT
/// code reaches throughout, besides `collect()`), in a process of their own: the setting is
/// read once per process.
#[test]
fn rooting_holds_under_gc_stress() {
    const CHILD: &str = "TRUST_JS_TEST_ROOTING_STRESS_CHILD";
    if std::env::var_os(CHILD).is_some() || crate::gc_test::stress_active() {
        return;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "jit::rooting_tests::",
            "jit::traced_tests::",
            "--skip",
            "rooting_holds_under_gc_stress",
        ])
        .env(CHILD, "1")
        .env("TRUST_JS_GC_STRESS", "1")
        .output()
        .expect("stressed rooting tests");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success()
            && stdout.contains("test result: ok.")
            && !stdout.contains(" 0 passed"),
        "{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
