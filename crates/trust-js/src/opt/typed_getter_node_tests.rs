//! A typed array's `length`, `byteLength` and `byteOffset` in the optimizing tier (the getter
//! line, (c)): a read whose ways are all typed-array accessor ways at the intrinsic getter
//! becomes the receiver's typed-array check, the walk's checks, the live getter's check and the
//! answer computed from the view (`NodeOp::TypedArrayLength`), with no call; a site that met
//! another receiver, or whose getter the snapshot read replaced, keeps the generic access. The
//! values are Node's.

use super::*;

/// The verified graph of the global function `name`, its snapshot taken in the engine's realm,
/// which reads the holder's live getter (`Snapshot::typed_getter`).
fn graph_in(engine: &mut Engine, name: &str) -> (Rc<Chunk>, Graph) {
    let chunk = chunk_of(engine, name);
    let snapshot = Snapshot::take_in(&chunk, Some(&engine.interp));
    let built = build(&chunk, &snapshot)
        .unwrap_or_else(|refusal| panic!("{name}: {refusal:?}\n{:#?}", chunk.jit_ops()));
    if let Err(error) = super::super::verify::verify(&built.graph, &chunk, &built.cfg) {
        panic!("{name}: {error}\n{}", dump(&built.graph, &chunk));
    }
    (chunk, built.graph)
}

fn lengths(graph: &Graph) -> Vec<u8> {
    graph
        .nodes
        .iter()
        .filter_map(|node| match node.op {
            NodeOp::TypedArrayLength { which } => Some(which),
            _ => None,
        })
        .collect()
}

const WARM: &str = "function f(a) { return a.length + a.byteLength + a.byteOffset; }
     var u8 = new Uint8Array(new ArrayBuffer(16), 4, 8);
     for (var i = 0; i < 20; i++) f(u8);";

#[test]
fn typed_array_reads_at_the_intrinsic_getter_are_the_tiers_checks_and_length_nodes() {
    let mut engine = engine(Tier::Jit);
    eval(&mut engine, WARM);
    let (chunk, graph) = graph_in(&mut engine, "f");
    let text = dump(&graph, &chunk);
    assert_eq!(lengths(&graph), [0, 1, 2], "{text}");
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    for node in &graph.nodes {
        let reason = match node.op {
            NodeOp::CheckTypedShape { .. } | NodeOp::CheckProto { .. } => ExitReason::WrongShape,
            NodeOp::CheckGetter { .. } => ExitReason::WrongCallee,
            _ => continue,
        };
        assert_eq!(node.reason, Some(reason), "{text}");
    }
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckTypedShape { .. })),
        3,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckGetter { .. })),
        3,
        "{text}"
    );
    // No ordinary receiver check: a typed array has its IC-plain flag clear.
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckShape { .. })),
        0,
        "{text}"
    );
}

#[test]
fn a_site_that_met_another_receiver_or_another_getter_keeps_the_generic_access() {
    // Another receiver: a plain object's own `length` fills a data way (kind 0).
    let mut mixed = engine(Tier::Jit);
    eval(
        &mut mixed,
        "function f(a) { return a.length; }
         var u8 = new Uint8Array(3), o = { length: 1 };
         for (var i = 0; i < 20; i++) { f(u8); f(o); }",
    );
    let (chunk, graph) = graph_in(&mut mixed, "f");
    assert!(lengths(&graph).is_empty(), "{}", dump(&graph, &chunk));
    // The getter replaced on %TypedArray.prototype%: the snapshot reads the live getter.
    let mut replaced = engine(Tier::Jit);
    eval(&mut replaced, WARM);
    eval(
        &mut replaced,
        "var TAP = Object.getPrototypeOf(Uint8Array.prototype);
         var d = Object.getOwnPropertyDescriptor(TAP, 'byteLength');
         Object.defineProperty(TAP, 'byteLength', { get: function () { return d.get.call(this); },
           configurable: true });",
    );
    let (chunk, graph) = graph_in(&mut replaced, "f");
    assert_eq!(lengths(&graph), [0, 2], "{}", dump(&graph, &chunk));
    // No realm read, no getter known: the generic access.
    let chunk = chunk_of(&mut replaced, "f");
    let built = build(&chunk, &Snapshot::take(&chunk)).expect("built");
    assert!(lengths(&built.graph).is_empty());
    drop(replaced);
    drop(mixed);
    // An ordinary object on a typed array's prototype shares its ways' walk: a way of kind 0.
    let mut fake = engine(Tier::Jit);
    eval(
        &mut fake,
        "function f(a) { return a.length; }
         var u8 = new Uint8Array(3), o = Object.create(Uint8Array.prototype);
         for (var i = 0; i < 20; i++) { f(u8); try { f(o); } catch (e) {} }",
    );
    let (chunk, graph) = graph_in(&mut fake, "f");
    assert!(lengths(&graph).is_empty(), "{}", dump(&graph, &chunk));
}

/// Fixed, length-tracking, resized, detached and subclass views, each read by functions of their
/// own (`new Function`: a chunk each, so each site sees one walk), BigInt elements (no guard
/// record: the out-of-line call), receivers alternating with an ordinary object, and
/// the getter replaced after the compile (WrongCallee, then the new getter's value), and an
/// ordinary object of a typed array's shape on its prototype (the kind check: the native's
/// TypeError). Node's values.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const PROGRAM: &str = r#"
function mk() { return [new Function('a', 'return a.length'), new Function('a', 'return a.byteLength'), new Function('a', 'return a.byteOffset')]; }
function all(fs, a) { return fs[0](a) + '/' + fs[1](a) + '/' + fs[2](a); }
var out = [];
var rab = new ArrayBuffer(16, { maxByteLength: 64 });
var fixed = new Uint16Array(rab, 4, 4), track = new Int32Array(rab, 8), plain = new Float64Array(3);
var sub = new (class extends Uint8Array {})(7), big = new BigInt64Array(5);
var det = new Uint8Array(new ArrayBuffer(12), 2, 6);
var F = mk(), T = mk(), P = mk(), S = mk(), B = mk(), D = mk(), n = 0;
function sum(fs, a) { return fs[0](a) + fs[1](a) + fs[2](a); }
for (var r = 0; r < 20000; r++) { n += sum(F, fixed) + sum(T, track) + sum(P, plain) + sum(S, sub) + sum(B, big) + sum(D, det); }
out.push(n, all(F, fixed), all(T, track), all(P, plain), all(S, sub), all(B, big), all(D, det));
rab.resize(10);
out.push(all(F, fixed), all(T, track));
rab.resize(6);
out.push(all(T, track));
rab.resize(40);
out.push(all(F, fixed), all(T, track));
det.buffer.transfer();
out.push(all(D, det));
var fake = Object.create(Uint8Array.prototype), te = 0;
for (var r = 0; r < 50; r++) { try { F[0](fake); } catch (e) { if (e instanceof TypeError) te++; } }
out.push(te);
var s = 0; for (var r = 0; r < 300; r++) s += P[0](r % 2 ? plain : { length: 5 }); out.push(s);
var TAP = Object.getPrototypeOf(Uint8Array.prototype);
var orig = Object.getOwnPropertyDescriptor(TAP, 'length');
Object.defineProperty(TAP, 'length', { get: function () { return 42; }, configurable: true });
s = 0; for (var r = 0; r < 300; r++) s += S[0](sub); out.push(s);
Object.defineProperty(TAP, 'length', orig);
s = 0; for (var r = 0; r < 300; r++) s += S[0](sub); out.push(s);
out.join(' ')
"#;

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const NODE: &str = "2680000 4/8/4 2/8/8 3/24/0 7/7/0 5/40/0 6/6/2 0/0/0 0/0/8 0/0/0 4/8/4 \
     8/32/8 0/0/0 50 1200 12600 2100";

#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn typed_array_getters_answer_as_node_at_graph_and_splice() {
    for graph in [true, false] {
        for eager in [false, true] {
            let mut engine = engine(Tier::Jit);
            super::super::TestSwitches {
                on: true,
                eager,
                inline_graph: Some(graph),
                ..Default::default()
            }
            .set();
            assert_eq!(
                eval(&mut engine, PROGRAM),
                NODE,
                "graph {graph} eager {eager}"
            );
        }
    }
}

/// A loop reading a typed array's length: the read is no effect, so the loop's own facts stay
/// (the node writes nothing: `verify::heap_write`).
#[test]
fn a_loop_bounded_by_a_typed_arrays_length_holds_no_effect() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function g(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
         var u8 = new Uint8Array(8); for (var i = 0; i < 20; i++) g(u8);",
    );
    let (chunk, graph) = graph_in(&mut engine, "g");
    let text = dump(&graph, &chunk);
    assert_eq!(lengths(&graph), [0], "{text}");
    assert_eq!(count(&graph, is_effect), 0, "{text}");
}

/// A view with no guard record (BigInt elements) takes the length node's out-of-line call of the
/// intrinsic, which answers.
#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn a_view_with_no_guard_record_takes_the_out_of_line_call() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "var g = new Function('a', 'return a.byteLength + a.length');
         var big = new BigInt64Array(5); for (var i = 0; i < 20; i++) g(big);",
    );
    let (chunk, graph) = graph_in(&mut engine, "g");
    assert_eq!(lengths(&graph), [1, 0], "{}", dump(&graph, &chunk));
    super::super::TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    let calls = super::super::a64::TEST_TYPED_GETTER_CALLS.with(|calls| calls.get());
    super::super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
    assert_eq!(
        eval(
            &mut engine,
            "var s = 0; for (var i = 0; i < 20000; i++) s += g(big); s"
        ),
        "900000"
    );
    assert!(
        super::super::a64::TEST_TYPED_GETTER_CALLS.with(|calls| calls.get()) > calls,
        "the out-of-line call ran"
    );
    // The receiver's typed-array check passes a typed array: the code never exits.
    let exits = super::super::exit::TEST_EXITS.with(|exits| exits.borrow().clone());
    assert!(exits.is_empty(), "{exits:?}");
}

/// An ordinary object given a typed array's prototype (`Object.setPrototypeOf`, which keeps the
/// empty shape a typed array has; `Object.create`'s object has another), reaching
/// compiled code whose ways saw typed arrays alone: the receiver's kind check exits, and the
/// intrinsic getter throws its TypeError, as Node does.
#[test]
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn an_ordinary_object_of_a_typed_arrays_shape_fails_the_kind_check() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "var h = new Function('a', 'return a.length');
         var u8 = new Uint8Array(6); for (var i = 0; i < 20; i++) h(u8);",
    );
    super::super::TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    assert_eq!(
        eval(
            &mut engine,
            "var s = 0; for (var i = 0; i < 20000; i++) s += h(u8);
             var fake = Object.setPrototypeOf({}, Uint8Array.prototype), t = 0;
             for (var i = 0; i < 50; i++) { try { h(fake); } catch (e) { if (e instanceof TypeError) t++; } }
             s + '|' + t"
        ),
        "120000|50"
    );
}
