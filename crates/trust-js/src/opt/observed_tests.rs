//! The typed stores' book kept only for a buffer an embedder observed (P3 stage 3 step L4): the
//! embed API's generation and ranges exact in every tier from a buffer's first observation on,
//! none kept before it; the tier's TypedData of a store testing the flag (ObservedBuffer, its own
//! exit, a loop's entry check's too) and the recompile after its flag keeping today's book.

use super::build::build;
use super::ir::*;
use super::profile::ChunkProfile;
use super::snapshot::Snapshot;
use super::TestSwitches;
use crate::bytecode::{Chunk, Op, Tier};
use crate::native_typed_array::with_array_buffer;
use crate::value::{ObjectKind, Value, ValueView};
use crate::{Completion, Engine};
use std::cell::Cell;
use std::ops::Range;
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

fn global(engine: &mut Engine, name: &str) -> Value {
    let env = engine.interp.global_env;
    engine
        .interp
        .get_var(name, &env)
        .ok()
        .unwrap_or_else(|| panic!("no global {name}"))
}

/// The buffer the global `name` names (an ArrayBuffer, or a view's).
fn buffer(engine: &mut Engine, name: &str) -> Value {
    let value = global(engine, name);
    let ValueView::Obj(object) = value.view() else {
        panic!("{name} is no object")
    };
    let view = match &object.borrow().kind {
        ObjectKind::TypedArray(view) => Some(view.info.buffer),
        _ => None,
    };
    view.map_or(value, Value::Obj)
}

fn version(engine: &mut Engine, name: &str) -> Option<u64> {
    let buffer = buffer(engine, name);
    engine.interp.array_buffer_version(&buffer)
}

fn take(engine: &mut Engine, name: &str) -> Option<Vec<Range<usize>>> {
    let buffer = buffer(engine, name);
    engine.interp.take_array_buffer_dirty_ranges(&buffer)
}

/// One range, as the embed API returns ranges.
fn one(range: Range<usize>) -> Vec<Range<usize>> {
    std::iter::once(range).collect()
}

/// The bytes `ranges` cover, as ranges of their union.
fn union(ranges: &[Range<usize>]) -> Vec<Range<usize>> {
    let mut bytes: Vec<usize> = ranges.iter().flat_map(|range| range.clone()).collect();
    bytes.sort_unstable();
    bytes.dedup();
    let mut out: Vec<Range<usize>> = Vec::new();
    for byte in bytes {
        match out.last_mut() {
            Some(last) if last.end == byte => last.end += 1,
            _ => out.push(byte..byte + 1),
        }
    }
    out
}

/// The book of the buffer `name` names, as the engine holds it: (the state's flag, the native
/// record's flag, its interval's start, its count, the generation, the ranges held), and that
/// the two flags agree.
fn book(
    engine: &mut Engine,
    name: &str,
) -> (bool, Option<u8>, Option<usize>, Option<u64>, u64, usize) {
    let buffer = buffer(engine, name);
    let object = buffer.as_obj().expect("a buffer object");
    with_array_buffer(&object, |data| {
        assert!(data.observed_agrees(), "{name}: the two flags agree");
        let native = data.native.as_ref();
        (
            data.observed.get(),
            native.map(|buffer| buffer.observed.get()),
            native.map(|buffer| buffer.dirty_start.get()),
            native.map(|buffer| buffer.writes.get()),
            data.version.get(),
            data.dirty.borrow().len(),
        )
    })
    .expect("an ArrayBuffer")
}

/// The global function `name`'s first-stage chunk.
fn chunk_of(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    let value = global(engine, name);
    let ValueView::Obj(function) = value.view() else {
        panic!("{name} is not an object")
    };
    let ObjectKind::User(user) = &function.borrow().kind else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .cloned()
        .unwrap_or_else(|| panic!("{name} stayed interpreted"))
}

/// The pc of `chunk`'s loop header: the target of its one backward jump.
fn header_pc(chunk: &Chunk) -> u32 {
    let ops = chunk.jit_ops();
    let targets: Vec<u32> = ops
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| match *op {
            Op::Jump(target) | Op::JumpIfFalse(target) if (target as usize) <= pc => Some(target),
            _ => None,
        })
        .collect();
    assert_eq!(targets.len(), 1, "one loop: {ops:?}");
    targets[0]
}

/// The exit profile's reasons flagged anywhere in `chunk`, each (pc, reason).
fn all_flags(chunk: &Chunk) -> Vec<(u32, u8)> {
    let mut out = Vec::new();
    if let Some(profile) = ChunkProfile::get(chunk) {
        for pc in 0..chunk.jit_ops().len() as u32 {
            profile.sites_at(pc, |reason, _, _| out.push((pc, reason)));
        }
    }
    out
}

/// The built graph of global function `name` (its snapshot as it stands, read in the engine's
/// realm as a compile reads it: the Agent's protector too), verified.
fn graph_of(engine: &mut Engine, name: &str) -> Graph {
    let chunk = chunk_of(engine, name);
    let snapshot = Snapshot::take_in(&chunk, Some(&engine.interp));
    let built = build(&chunk, &snapshot).unwrap_or_else(|refusal| panic!("{name}: {refusal:?}"));
    super::verify::verify(&built.graph, &chunk, &built.cfg)
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    built.graph
}

/// The graph's TypedData nodes of stores, each (book, moved to a loop's entry).
fn store_data(graph: &Graph) -> Vec<(bool, bool)> {
    graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(k, node)| match node.op {
            NodeOp::TypedData { write: true, book } => {
                let moved = graph.moved.iter().any(|check| check.node.index() == k);
                Some((book, moved))
            }
            _ => None,
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Config {
    Vm,
    Template,
    Splice,
    Graph,
    Forced,
    Poison,
}

const CONFIGS: [Config; 6] = [
    Config::Vm,
    Config::Template,
    Config::Splice,
    Config::Graph,
    Config::Forced,
    Config::Poison,
];

fn engine(config: Config) -> Engine {
    let tier = !matches!(config, Config::Vm | Config::Template);
    TestSwitches {
        on: tier,
        eager: matches!(config, Config::Forced | Config::Poison),
        exit_every: if config == Config::Forced { 100 } else { 0 },
        inline_graph: Some(config == Config::Graph),
        poison: config == Config::Poison,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(if config == Config::Vm {
        Tier::Bytecode
    } else {
        Tier::Jit
    });
    engine
}

const STORES: &str = "var buf = new ArrayBuffer(64), u8 = new Uint8Array(buf);
     function fill8(a, n, v) { for (var i = 0; i < n; i++) a[i] = v; }";

// ---- the contract: exact from the first observation on, in every tier --------------------------

#[test]
fn an_observed_buffers_generation_counts_each_store_and_its_ranges_cover_them_in_every_tier() {
    for config in CONFIGS {
        let mut engine = engine(config);
        eval(&mut engine, STORES);
        // The first observation: the baseline, and the whole buffer in the first take.
        let mut seen = version(&mut engine, "buf").expect("attached");
        assert_eq!(take(&mut engine, "buf"), Some(one(0..64)), "{config:?}");
        assert_eq!(
            version(&mut engine, "buf"),
            Some(seen),
            "{config:?}: read, read"
        );
        for round in 0..40 {
            // 50 calls of 16 stores, through each tier the configuration reaches.
            eval(
                &mut engine,
                "for (var c = 0; c < 50; c++) fill8(u8, 16, c & 7)",
            );
            let now = version(&mut engine, "buf").expect("attached");
            assert_eq!(now, seen + 800, "{config:?} round {round}: one a store");
            seen = now;
            let ranges = take(&mut engine, "buf").expect("attached");
            assert_eq!(union(&ranges), one(0..16), "{config:?} round {round}");
            assert_eq!(
                version(&mut engine, "buf"),
                Some(seen),
                "{config:?}: read, read"
            );
        }
        // The builtins' and a DataView's writes: each moves the generation and covers its bytes.
        for (write, bytes) in [
            ("new DataView(buf).setInt16(20, 7)", 20..22),
            ("u8.fill(3, 30, 34)", 30..34),
            ("u8.set([1, 2, 3], 40)", 40..43),
            ("u8.copyWithin(50, 0, 4)", 50..54),
        ] {
            eval(&mut engine, write);
            let now = version(&mut engine, "buf").expect("attached");
            assert_ne!(now, seen, "{config:?}: {write}");
            seen = now;
            let ranges = take(&mut engine, "buf").expect("attached");
            assert_eq!(union(&ranges), one(bytes), "{config:?}: {write}");
            assert_eq!(
                version(&mut engine, "buf"),
                Some(seen),
                "{config:?}: read, read"
            );
        }
        assert_eq!(
            eval(&mut engine, "Array.from(u8.subarray(0, 4)).join()"),
            "1,1,1,1",
            "{config:?}"
        );
    }
}

#[test]
fn an_unobserved_buffer_keeps_no_book_and_its_first_observation_is_the_baseline() {
    for config in [Config::Template, Config::Splice, Config::Graph] {
        let mut engine = engine(config);
        eval(&mut engine, STORES);
        eval(
            &mut engine,
            "for (var c = 0; c < 2000; c++) fill8(u8, 16, c & 7); u8[20] = 1;
             new DataView(buf).setInt8(30, 2); u8.fill(3, 40, 44)",
        );
        // Nothing kept: no generation, no ranges, no native interval or count.
        assert_eq!(
            book(&mut engine, "buf"),
            (false, Some(0), Some(usize::MAX), Some(0), 0, 0),
            "{config:?}"
        );
        // The first observation by the generation's read: the baseline as it stands.
        assert_eq!(version(&mut engine, "buf"), Some(0), "{config:?}");
        assert_eq!(
            book(&mut engine, "buf").1,
            Some(1),
            "{config:?}: the record's flag set"
        );
        assert_eq!(take(&mut engine, "buf"), Some(one(0..64)), "{config:?}");
        eval(&mut engine, "fill8(u8, 4, 9)");
        assert_eq!(version(&mut engine, "buf"), Some(4), "{config:?}");
        assert_eq!(union(&take(&mut engine, "buf").unwrap()), one(0..4));
        // The first observation by a take: the whole buffer, and the generation as it stands.
        eval(
            &mut engine,
            "var other = new Uint8Array(new ArrayBuffer(24)); fill8(other, 8, 1)",
        );
        assert_eq!(take(&mut engine, "other"), Some(one(0..24)), "{config:?}");
        assert_eq!(version(&mut engine, "other"), Some(0), "{config:?}");
        assert_eq!(eval(&mut engine, "other[7] + u8[3]"), "10", "{config:?}");
    }
}

// ---- the tier's speculation and its recompile -------------------------------------------------

/// `w` warmed in the tier (its loop's typed checks moved to the entry, its stores keeping no
/// book), then the buffer observed by the embedder: the loop's entry check exits ObservedBuffer
/// at the header, one discard flags that alone, and the recompile keeps the checks moved and its
/// stores keep the book; the generation counts every store and the ranges cover them throughout.
/// A `for` loop, whose store the loop's first test may skip (the moved check's site blind: its
/// discard flags the record's own reason at the header), and a do-while, whose store runs
/// whenever the loop is entered (its site runs anyway: a moved check's own record flags that
/// site with its Detached reason, which the ObservedBuffer record must not carry).
#[test]
fn a_loop_whose_buffer_is_observed_after_warm_up_exits_at_its_header_once_and_books_after() {
    for (shape, w) in [
        (
            "for",
            "function w(x, n, v) { for (var i = 0; i < n; i++) x[i] = v; }",
        ),
        (
            "do-while",
            "function w(x, n, v) { var i = 0; do { x[i] = v; i++; } while (i < n); }",
        ),
    ] {
        for graph in [false, true] {
            observed_loop_case(shape, w, graph);
        }
    }
}

fn observed_loop_case(shape: &str, w: &str, graph: bool) {
    {
        let config = if graph { Config::Graph } else { Config::Splice };
        let mut engine = engine(config);
        eval(
            &mut engine,
            &format!("var a = new Uint8Array(new ArrayBuffer(256)); {w}"),
        );
        super::TEST_COMPILED.with(|compiled| compiled.borrow_mut().clear());
        eval(
            &mut engine,
            "for (var c = 0; c < 3000; c++) w(a, 64, c & 7)",
        );
        assert!(
            super::TEST_COMPILED.with(|compiled| !compiled.borrow().is_empty()),
            "{shape} {config:?}: the tier compiled w"
        );
        assert_eq!(
            store_data(&graph_of(&mut engine, "w")),
            vec![(false, true)],
            "{shape} {config:?}"
        );
        let chunk = chunk_of(&mut engine, "w");
        let header = header_pc(&chunk);
        assert!(all_flags(&chunk).is_empty(), "{shape} {config:?}");
        assert!(!book(&mut engine, "a").0);
        let mut seen = version(&mut engine, "a").expect("attached");
        assert_eq!(take(&mut engine, "a"), Some(one(0..256)));
        super::TEST_DISCARDED.with(|discarded| discarded.borrow_mut().clear());
        super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        super::TEST_COMPILED.with(|compiled| compiled.borrow_mut().clear());
        for round in 0..60 {
            eval(&mut engine, "for (var c = 0; c < 100; c++) w(a, 64, c & 7)");
            let now = version(&mut engine, "a").expect("attached");
            assert_eq!(
                now,
                seen + 6400,
                "{shape} {config:?} round {round}: one a store"
            );
            seen = now;
            assert_eq!(
                union(&take(&mut engine, "a").unwrap()),
                one(0..64),
                "{shape} {config:?}"
            );
        }
        let exits = super::exit::TEST_EXITS.with(|exits| exits.borrow().clone());
        assert!(
            !exits.is_empty(),
            "{shape} {config:?}: the entry check exited"
        );
        assert!(
            exits
                .iter()
                .all(|&exit| exit == (header, ExitReason::ObservedBuffer)),
            "{shape} {config:?}: {exits:?}"
        );
        let discards = super::TEST_DISCARDED.with(|discarded| discarded.borrow().len());
        assert_eq!(discards, 1, "{shape} {config:?}");
        assert_eq!(
            all_flags(&chunk),
            vec![(header, ExitReason::ObservedBuffer as u8)],
            "{shape} {config:?}: the header's ObservedBuffer alone, no HoistingFailed, no Detached"
        );
        assert!(
            super::TEST_COMPILED.with(|compiled| !compiled.borrow().is_empty()),
            "{shape} {config:?}: recompiled"
        );
        // The recompile's entry check is still moved, and books.
        assert_eq!(
            store_data(&graph_of(&mut engine, "w")),
            vec![(true, true)],
            "{shape} {config:?}"
        );
        super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        eval(&mut engine, "for (var c = 0; c < 100; c++) w(a, 64, 5)");
        assert!(
            super::exit::TEST_EXITS.with(|exits| exits.borrow().is_empty()),
            "{shape} {config:?}: no exit after the recompile"
        );
        assert_eq!(
            version(&mut engine, "a"),
            Some(seen + 6400),
            "{shape} {config:?}"
        );
        assert_eq!(
            union(&take(&mut engine, "a").unwrap()),
            one(0..64),
            "{shape} {config:?}"
        );
        assert_eq!(
            eval(&mut engine, "a[63] + a[64]"),
            "5",
            "{shape} {config:?}"
        );
    }
}

thread_local! {
    /// The generation `observe(b)` read at its first call of a run (`observing_engine`).
    static FIRST: Cell<Option<u64>> = const { Cell::new(None) };
}

/// An engine with the host function `observe(b)`: the embedder reads `b`'s generation, and at
/// its first call of a run takes `b`'s ranges (the whole buffer), recording the generation.
fn observing_engine(config: Config) -> Engine {
    let engine = engine(config);
    let observe = engine.interp.make_native("observe", 1, |interp, _, args| {
        let version = interp.array_buffer_version(&args[0]).expect("attached");
        if FIRST.with(Cell::get).is_none() {
            interp.take_array_buffer_dirty_ranges(&args[0]);
            FIRST.with(|first| first.set(Some(version)));
        }
        Ok(Value::Undefined)
    });
    engine.interp.global.borrow_mut().props.insert(
        "observe",
        crate::value::Property::data(Value::Obj(observe), true, true, true),
    );
    engine
}

/// A loop that calls a host function observing its store's buffer: every store after the call
/// keeps the book, in the template and in the tier at splice and at graph (the call is an
/// effect, which ends the TypedData fact, so the next store's TypedData tests the flag again).
#[test]
fn a_host_call_in_the_loop_that_observes_the_buffer_books_every_store_after_it() {
    for config in [Config::Template, Config::Splice, Config::Graph] {
        let mut engine = observing_engine(config);
        eval(
            &mut engine,
            "function w(x, n, v) { for (var i = 0; i < n; i++) { x[i] = v; observe(x.buffer); } }
             for (var c = 0; c < 1000; c++) w(new Uint8Array(new ArrayBuffer(48)), 32, 1);",
        );
        for run in 0..300 {
            FIRST.with(|first| first.set(None));
            eval(
                &mut engine,
                &format!(
                    "var a = new Uint8Array(new ArrayBuffer(48)); w(a, 32, {})",
                    run & 7
                ),
            );
            let first = FIRST.with(Cell::get).expect("observed");
            // The first store came before the first observation, the other 31 after it.
            assert_eq!(
                version(&mut engine, "a"),
                Some(first + 31),
                "{config:?} run {run}"
            );
            assert_eq!(
                union(&take(&mut engine, "a").unwrap()),
                one(1..32),
                "{config:?} run {run}"
            );
        }
    }
}

/// Two receivers in one loop, one buffer observed: the header's flag makes both entry checks
/// `book` (the flag names the pc, not the buffer), and the other buffer's stores stay right, and
/// exact once it too is observed.
#[test]
fn a_loop_with_two_receivers_books_both_after_one_buffers_observation() {
    for config in [Config::Splice, Config::Graph] {
        let mut engine = engine(config);
        eval(
            &mut engine,
            "var A = new Uint8Array(new ArrayBuffer(64)), B = new Uint8Array(new ArrayBuffer(64));
             function w2(x, y, n, v) { for (var i = 0; i < n; i++) { x[i] = v; y[i] = v + 1; } }
             for (var c = 0; c < 3000; c++) w2(A, B, 32, c & 7);",
        );
        assert_eq!(
            store_data(&graph_of(&mut engine, "w2")),
            vec![(false, true), (false, true)],
            "{config:?}"
        );
        let mut seen = version(&mut engine, "A").expect("attached");
        take(&mut engine, "A");
        for _ in 0..60 {
            eval(
                &mut engine,
                "for (var c = 0; c < 100; c++) w2(A, B, 32, c & 7)",
            );
            let now = version(&mut engine, "A").expect("attached");
            assert_eq!(now, seen + 3200, "{config:?}");
            seen = now;
            assert_eq!(union(&take(&mut engine, "A").unwrap()), one(0..32));
        }
        assert_eq!(
            store_data(&graph_of(&mut engine, "w2")),
            vec![(true, true), (true, true)],
            "{config:?}: both entry checks book"
        );
        assert_eq!(
            eval(&mut engine, "w2(A, B, 32, 3); A[31] + ',' + B[31]"),
            "3,4"
        );
        // B, observed now: its baseline, then exact.
        let base = version(&mut engine, "B").expect("attached");
        assert_eq!(union(&take(&mut engine, "B").unwrap()), one(0..64));
        eval(
            &mut engine,
            "for (var c = 0; c < 100; c++) w2(A, B, 32, c & 7)",
        );
        assert_eq!(version(&mut engine, "B"), Some(base + 3200), "{config:?}");
        assert_eq!(
            union(&take(&mut engine, "B").unwrap()),
            one(0..32),
            "{config:?}"
        );
    }
}

// ---- the two flags -----------------------------------------------------------------------------

/// The state's flag and the native record's agree wherever both exist: a record made after the
/// observation copies it; a detached buffer answers None; a transfer's and a copy's new buffer
/// (structured clone's `new ArrayBuffer` and `set`, trust-js-web serialize.js) start unobserved;
/// a resize keeps the flag; a SharedArrayBuffer keeps no book and is never flagged.
#[test]
fn the_buffers_two_observed_flags_agree_through_views_detach_transfer_resize_and_copies() {
    for config in [Config::Template, Config::Splice] {
        let mut engine = engine(config);
        eval(
            &mut engine,
            "function put(t, i, v) { t[i] = v; }
             var w = new Uint8Array(8); for (var c = 0; c < 3000; c++) put(w, c & 7, c & 255);
             var early = new ArrayBuffer(16);",
        );
        // Observed before any view made its record: the record copies the flag.
        assert_eq!(book(&mut engine, "early"), (false, None, None, None, 0, 0));
        assert_eq!(version(&mut engine, "early"), Some(0));
        assert_eq!(take(&mut engine, "early"), Some(one(0..16)));
        eval(
            &mut engine,
            "var et = new Uint8Array(early); put(et, 3, 9);",
        );
        assert_eq!(book(&mut engine, "early").1, Some(1), "{config:?}");
        assert_eq!(version(&mut engine, "early"), Some(1), "{config:?}");
        assert_eq!(take(&mut engine, "early"), Some(one(3..4)), "{config:?}");
        // A transfer's new buffer starts unobserved; the old one is detached.
        eval(
            &mut engine,
            "var moved = early.transfer(); var mt = new Uint8Array(moved);",
        );
        assert_eq!(version(&mut engine, "early"), None);
        assert_eq!(take(&mut engine, "early"), None);
        eval(&mut engine, "put(mt, 5, 1); put(et, 0, 1);");
        assert_eq!(
            book(&mut engine, "moved"),
            (false, Some(0), Some(usize::MAX), Some(0), 0, 0)
        );
        assert_eq!(take(&mut engine, "moved"), Some(one(0..16)), "{config:?}");
        assert_eq!(eval(&mut engine, "mt[3] + mt[5] + ',' + et.length"), "10,0");
        // A copy, as structured clone makes it: a new buffer, unobserved.
        eval(
            &mut engine,
            "var copy = new ArrayBuffer(16); new Uint8Array(copy).set(mt); put(new Uint8Array(copy), 1, 2);",
        );
        assert!(!book(&mut engine, "copy").0);
        assert_eq!(book(&mut engine, "copy").4, 0);
        // Detach by the embedder after its observation.
        let moved = buffer(&mut engine, "moved");
        assert!(engine.interp.detach_array_buffer(&moved));
        assert_eq!(version(&mut engine, "moved"), None);
        assert_eq!(eval(&mut engine, "put(mt, 1, 1); mt[1]"), "undefined");
        // A resize keeps the flag; the resize itself is a write the book covers.
        eval(&mut engine, "var rb = new ArrayBuffer(8, { maxByteLength: 64 }); var rt = new Uint8Array(rb, 0, 8);");
        let base = version(&mut engine, "rb").expect("attached");
        take(&mut engine, "rb");
        eval(&mut engine, "rb.resize(32); put(rt, 2, 7);");
        assert_eq!(book(&mut engine, "rb").1, Some(1), "{config:?}");
        assert!(version(&mut engine, "rb").unwrap() > base, "{config:?}");
        assert_eq!(
            union(&take(&mut engine, "rb").unwrap()),
            one(0..32),
            "{config:?}"
        );
        // A SharedArrayBuffer: no generation, no ranges, no flag.
        eval(
            &mut engine,
            "var sab = new SharedArrayBuffer(8); new Uint8Array(sab)[1] = 3;",
        );
        assert_eq!(version(&mut engine, "sab"), None);
        assert_eq!(take(&mut engine, "sab"), None);
        assert!(!book(&mut engine, "sab").0);
    }
}

// ---- the protector: an Agent where an embedder observed some buffer --------------------------

/// An embedder that observed some buffer before the tier compiled: every compile's stores test
/// their buffer's flag themselves (the Agent's protector, as V8's code carries its typed stores'
/// detach test at each access once its ArrayBufferDetaching protector is invalid), the loop's
/// hoisted TypedData testing nothing for it, so nothing exits ObservedBuffer, the observed
/// buffer's book is exact and an unobserved one's stores keep none. On the main thread and on
/// the compile thread (the protector read into the raw snapshot on the main thread).
#[test]
fn an_observation_before_the_first_compile_books_every_store_and_exits_nothing() {
    for (config, asynchronous) in [
        (Config::Splice, false),
        (Config::Graph, false),
        (Config::Splice, true),
    ] {
        let mut engine = engine(config);
        if asynchronous {
            TestSwitches {
                on: true,
                inline_graph: Some(false),
                asynchronous: true,
                ..Default::default()
            }
            .set();
        }
        eval(
            &mut engine,
            "var a = new Uint8Array(new ArrayBuffer(128)), b = new Uint8Array(new ArrayBuffer(128));
             function w(x, n, v) { for (var i = 0; i < n; i++) x[i] = v; }",
        );
        let mut seen = version(&mut engine, "a").expect("attached");
        assert_eq!(take(&mut engine, "a"), Some(one(0..128)));
        super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        super::TEST_COMPILED.with(|compiled| compiled.borrow_mut().clear());
        for round in 0..40 {
            eval(
                &mut engine,
                "for (var c = 0; c < 100; c++) { w(a, 64, c & 7); w(b, 32, 1); }",
            );
            super::finish_compiles_for_test(&mut engine.interp);
            let now = version(&mut engine, "a").expect("attached");
            assert_eq!(now, seen + 6400, "{config:?} {asynchronous} round {round}");
            seen = now;
            assert_eq!(union(&take(&mut engine, "a").unwrap()), one(0..64));
        }
        assert!(
            super::TEST_COMPILED.with(|compiled| !compiled.borrow().is_empty()),
            "{config:?} {asynchronous}: compiled"
        );
        let exits = super::exit::TEST_EXITS.with(|exits| exits.borrow().clone());
        assert!(
            !exits
                .iter()
                .any(|&(_, reason)| reason == ExitReason::ObservedBuffer),
            "{config:?} {asynchronous}: {exits:?}"
        );
        assert_eq!(
            store_data(&graph_of(&mut engine, "w")),
            one_book(),
            "{config:?}"
        );
        // E1's untracked path: b's stores tested its flag and kept no book.
        assert_eq!(
            book(&mut engine, "b"),
            (false, Some(0), Some(usize::MAX), Some(0), 0, 0),
            "{config:?} {asynchronous}"
        );
        assert_eq!(take(&mut engine, "b"), Some(one(0..128)), "{config:?}");
        assert_eq!(eval(&mut engine, "a[63] + b[31]"), "4", "{config:?}");
    }
}

fn one_book() -> Vec<(bool, bool)> {
    std::iter::once((true, true)).collect()
}

/// A function of two loops over two views of one buffer, compiled before any observation, each
/// loop's entry check testing the flag: the observed buffer exits at the first loop's header,
/// and the one discard's recompile builds both loops' stores to test the flag themselves (the
/// protector), not only the pc its exit flagged, so the function is discarded once, not once a
/// site.
#[test]
fn a_late_observation_discards_a_function_once_and_its_recompile_books_every_store() {
    for config in [Config::Splice, Config::Graph] {
        let mut engine = engine(config);
        eval(
            &mut engine,
            "var a = new Uint8Array(new ArrayBuffer(128)), a2 = new Uint8Array(a.buffer, 32, 32);
             function w2(x, y, n, v) {
               for (var i = 0; i < n; i++) x[i] = v;
               for (var j = 0; j < n; j++) y[j] = v + 1;
             }
             for (var c = 0; c < 3000; c++) w2(a, a2, 32, c & 7);",
        );
        assert_eq!(
            store_data(&graph_of(&mut engine, "w2")),
            vec![(false, true), (false, true)],
            "{config:?}: speculating before the observation"
        );
        let mut seen = version(&mut engine, "a").expect("attached");
        take(&mut engine, "a");
        super::TEST_DISCARDED.with(|discarded| discarded.borrow_mut().clear());
        super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        for _ in 0..60 {
            eval(
                &mut engine,
                "for (var c = 0; c < 100; c++) w2(a, a2, 32, c & 7)",
            );
            let now = version(&mut engine, "a").expect("attached");
            assert_eq!(now, seen + 6400, "{config:?}");
            seen = now;
            assert_eq!(union(&take(&mut engine, "a").unwrap()), one(0..64));
        }
        let discards = super::TEST_DISCARDED.with(|discarded| discarded.borrow().len());
        assert_eq!(discards, 1, "{config:?}");
        assert_eq!(
            store_data(&graph_of(&mut engine, "w2")),
            vec![(true, true), (true, true)],
            "{config:?}: both loops book"
        );
        super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        eval(
            &mut engine,
            "for (var c = 0; c < 100; c++) w2(a, a2, 32, 2)",
        );
        assert!(super::exit::TEST_EXITS.with(|exits| exits.borrow().is_empty()));
        assert_eq!(eval(&mut engine, "a[31] + a[63]"), "5", "{config:?}");
    }
}

/// The protector set between a compile's snapshot (main thread, the bit clear) and its install:
/// the installed code speculates, meets the observed buffer, exits ObservedBuffer at the loop's
/// header, is discarded once, and its recompile's stores test the flag themselves; the book
/// exact throughout.
#[test]
fn an_observation_between_a_compiles_snapshot_and_its_install_exits_then_recompiles_once() {
    assert!(
        (0..3).any(|_| snapshot_to_install_attempt()),
        "no compile seen in flight in three engines"
    );
}

fn snapshot_to_install_attempt() -> bool {
    let mut engine = engine(Config::Splice);
    TestSwitches {
        on: true,
        inline_graph: Some(false),
        asynchronous: true,
        ..Default::default()
    }
    .set();
    eval(
        &mut engine,
        "var a = new Uint8Array(new ArrayBuffer(128));
         function w(x, n, v) { for (var i = 0; i < n; i++) x[i] = v; }",
    );
    // Single calls until a compile is in flight: its snapshot taken, its code not installed. A
    // compile made and installed between two of them (the shared compile thread idle, or the
    // code deferred while other tests hold the executable memory) misses the window: the test
    // then starts again with a fresh engine, three times at most.
    let mut calls = 0;
    while engine.interp.opt_pending.len() == 0 {
        eval(&mut engine, "w(a, 64, 1)");
        calls += 1;
        if calls == 20_000 {
            return false;
        }
    }
    assert!(!book(&mut engine, "a").0);
    let mut seen = version(&mut engine, "a").expect("attached");
    take(&mut engine, "a");
    super::TEST_DISCARDED.with(|discarded| discarded.borrow_mut().clear());
    super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
    super::TEST_COMPILED.with(|compiled| compiled.borrow_mut().clear());
    super::finish_compiles_for_test(&mut engine.interp);
    assert_eq!(
        super::TEST_COMPILED.with(|compiled| compiled.borrow().len()),
        1,
        "the speculating compile installed"
    );
    for round in 0..60 {
        eval(&mut engine, "for (var c = 0; c < 100; c++) w(a, 64, c & 7)");
        super::finish_compiles_for_test(&mut engine.interp);
        let now = version(&mut engine, "a").expect("attached");
        assert_eq!(now, seen + 6400, "round {round}");
        seen = now;
        assert_eq!(union(&take(&mut engine, "a").unwrap()), one(0..64));
    }
    let chunk = chunk_of(&mut engine, "w");
    let header = header_pc(&chunk);
    let exits = super::exit::TEST_EXITS.with(|exits| exits.borrow().clone());
    assert!(!exits.is_empty());
    assert!(
        exits
            .iter()
            .all(|&exit| exit == (header, ExitReason::ObservedBuffer)),
        "{exits:?}"
    );
    assert_eq!(
        super::TEST_DISCARDED.with(|discarded| discarded.borrow().len()),
        1
    );
    assert!(
        super::TEST_COMPILED.with(|compiled| compiled.borrow().len()) >= 2,
        "recompiled"
    );
    assert_eq!(store_data(&graph_of(&mut engine, "w")), one_book());
    super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
    eval(&mut engine, "for (var c = 0; c < 100; c++) w(a, 64, 6)");
    assert!(super::exit::TEST_EXITS.with(|exits| exits.borrow().is_empty()));
    assert_eq!(version(&mut engine, "a"), Some(seen + 6400));
    true
}
