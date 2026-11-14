//! P3 stage 3 step 0c's builder tests: a typed array's facts (its cell, kind, view, length and
//! data address) kept through a block's edge where a binding's or a field's fact gives its word
//! back (`Builder::prune`), through the header of a loop without effects (`loop_heap_facts`),
//! and established at such a loop's entry edge (`loop_entry_chains`).

use super::*;

/// The typed nodes of `graph`: CheckObject, CheckElements of a typed kind, TypedView,
/// TypedLength and TypedData, by name.
fn typed_counts(graph: &Graph) -> [(&'static str, usize); 5] {
    [
        ("CheckObject", count(graph, |op| *op == NodeOp::CheckObject)),
        (
            "CheckElements(Typed)",
            count(graph, |op| {
                matches!(op, NodeOp::CheckElements(ElemKind::Typed(_)))
            }),
        ),
        ("TypedView", count(graph, |op| *op == NodeOp::TypedView)),
        ("TypedLength", count(graph, |op| *op == NodeOp::TypedLength)),
        (
            "TypedData",
            count(graph, |op| matches!(op, NodeOp::TypedData { .. })),
        ),
    ]
}

#[test]
fn a_bindings_or_a_globals_typed_array_keeps_its_facts_past_a_branch() {
    // `a` is a closure's mutable binding (asm.js code's heap views), `G` a global variable
    // (Mandreel's): the arm's read takes the word from the binding's or the field's fact, and its
    // cell, kind, view, length and data address from that word's facts, which the branch's edge
    // keeps: one of each.
    let mut engine = warmed(
        "var m = (function (g) {
           var a = new g.Int32Array(8);
           function pick(x) { var s = a[0] | 0; if (x) s = (s + (a[1] | 0)) | 0; return s | 0; }
           return { pick: pick };
         })(this);
         var pick = m.pick;
         var G = new Int32Array(8);
         function gpick(x) { var s = G[0] | 0; if (x) s = (s + (G[1] | 0)) | 0; return s | 0; }
         for (var k = 0; k < 20; k++) { pick(k & 1); gpick(k & 1); }",
        "for (var k = 0; k < 20; k++) { pick(k & 1); gpick(k & 1); }",
    );
    for name in ["pick", "gpick"] {
        let chunk = chunk_of(&mut engine, name);
        let graph = graph_in(&engine, &chunk, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        for (what, n) in typed_counts(&graph) {
            assert_eq!(n, 1, "{name} {what}: {text}");
        }
    }
}

/// The nodes of `graph` in the loop whose header starts at `pc` that `which` accepts.
fn in_loop(graph: &Graph, pc: u32, which: impl Fn(&NodeOp) -> bool) -> Vec<&Node> {
    let lp = loop_at(graph, pc);
    graph
        .nodes
        .iter()
        .filter(|node| lp.members[node.block.index()] && which(&node.op))
        .collect()
}

/// Whether `op` is one of a typed access's checks and loads that an established fact serves:
/// CheckObject, CheckElements of a typed kind, TypedView, TypedLength, TypedData, LoadBinding.
fn established(op: &NodeOp) -> bool {
    matches!(
        op,
        NodeOp::CheckObject
            | NodeOp::CheckElements(ElemKind::Typed(_))
            | NodeOp::TypedView
            | NodeOp::TypedLength
            | NodeOp::TypedData { .. }
            | NodeOp::LoadBinding { .. }
    )
}

/// A typed array read and written before an effect-free loop that reads and writes it: a
/// parameter's (`kp`) and an asm.js-style closure binding's (`kb`).
const KEPT: &str = "var mk = (function (g) {
       var a = new g.Int32Array(64);
       function kb(n) { var s = a[0] | 0; a[1] = s;
         for (var i = 0; i < n; i = (i + 1) | 0) { s = (s + (a[i] | 0)) | 0; a[i] = s; }
         return s | 0; }
       return { kb: kb };
     })(this);
     var kb = mk.kb;
     function kp(t, n) { var s = t[0] | 0; t[1] = s;
       for (var i = 0; i < n; i = (i + 1) | 0) { s = (s + (t[i] | 0)) | 0; t[i] = s; }
       return s | 0; }
     var T = new Int32Array(64);
     for (var k = 0; k < 20; k++) { kb(60); kp(T, 60); }";
const KEPT_DRIVE: &str = "for (var k = 0; k < 20; k++) { kb(60); kp(T, 60); }";

#[test]
fn an_effect_free_loop_keeps_a_typed_arrays_facts_its_entry_holds() {
    // The read and the store before the loop checked the array's data address for a store, and
    // the loop has no effect: its accesses read the address, the view, the length, the cell and
    // the kind its entry holds, and the binding's word, with no check or load of them in the loop.
    let mut engine = warmed(KEPT, KEPT_DRIVE);
    for name in ["kb", "kp"] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        let header_pc = the_loop(&graph);
        assert!(
            in_loop(&graph, header_pc, established).is_empty(),
            "{name}: {text}"
        );
        assert_eq!(
            in_loop(&graph, header_pc, |op| matches!(op, NodeOp::LoadTyped(_))).len(),
            1,
            "{name}: {text}"
        );
        assert_eq!(
            in_loop(&graph, header_pc, |op| matches!(op, NodeOp::StoreTyped(_))).len(),
            1,
            "{name}: {text}"
        );
        // The store's check before the loop serves the loop's reads and writes.
        let data = nodes(&graph, |op| matches!(op, NodeOp::TypedData { .. }));
        assert_eq!(data.len(), 2, "{name}: {text}");
        assert!(
            graph
                .nodes
                .iter()
                .all(|node| node.reason != Some(ExitReason::HoistingFailed)),
            "{name}: {text}"
        );
    }
}

#[test]
fn a_loop_with_an_effect_checks_its_typed_arrays_in_the_loop_as_before() {
    let mut engine = warmed(
        "var mk = (function (g) {
           var a = new g.Int32Array(64);
           function eb(f, n) { var s = a[0] | 0;
             for (var i = 0; i < n; i = (i + 1) | 0) { s = (s + (a[i] | 0)) | 0; f(); }
             return s | 0; }
           return { eb: eb };
         })(this);
         var eb = mk.eb;
         function ep(f, t, n) { var s = t[0] | 0;
           for (var i = 0; i < n; i = (i + 1) | 0) { s = (s + (t[i] | 0)) | 0; f(); }
           return s | 0; }
         function nothing() {}
         var T = new Int32Array(64);
         for (var k = 0; k < 20; k++) { eb(nothing, 60); ep(nothing, T, 60); }",
        "for (var k = 0; k < 20; k++) { eb(nothing, 60); ep(nothing, T, 60); }",
    );
    for name in ["eb", "ep"] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        let header_pc = the_loop(&graph);
        assert_eq!(
            in_loop(&graph, header_pc, |op| matches!(
                op,
                NodeOp::TypedData { .. }
            ))
            .len(),
            1,
            "{name}: {text}"
        );
    }
}

#[test]
fn a_loop_with_two_entry_edges_keeps_only_what_both_entries_hold() {
    // `a` is a module's constant, whose word, kind, view and length are value facts both entry
    // edges hold; a call on one of them ended its data address there, so the loop checks the
    // address itself, whichever of the header's predecessors made the call.
    let mut engine = warmed(
        "var mt = (function (g) {
           const a = new g.Int32Array(64);
           function two(n, c, f) { var s = a[0] | 0, i = 0; if (c) { f(); }
             while (i < n) { s = (s + (a[i] | 0)) | 0; i = (i + 1) | 0; } return s | 0; }
           function owt(n, c, f) { var s = a[0] | 0, i = 0; if (c) { i = 1; } else { f(); }
             while (i < n) { s = (s + (a[i] | 0)) | 0; i = (i + 1) | 0; } return s | 0; }
           return { two: two, owt: owt };
         })(this);
         var two = mt.two, owt = mt.owt;
         function nothing() {}
         for (var k = 0; k < 20; k++) { two(60, k & 1, nothing); owt(60, k & 1, nothing); }",
        "for (var k = 0; k < 20; k++) { two(60, k & 1, nothing); owt(60, k & 1, nothing); }",
    );
    for name in ["two", "owt"] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        let header_pc = the_loop(&graph);
        assert_eq!(
            loop_at(&graph, header_pc).entries.len(),
            2,
            "{name}: {text}"
        );
        assert_eq!(
            in_loop(&graph, header_pc, |op| matches!(
                op,
                NodeOp::TypedData { .. }
            ))
            .len(),
            1,
            "{name}: {text}"
        );
        // The view, a value fact both entries hold, is the loop's.
        assert!(
            in_loop(&graph, header_pc, |op| *op == NodeOp::TypedView).is_empty(),
            "{name}: {text}"
        );
    }
}

#[test]
fn a_loop_with_a_modulo_keeps_no_heap_fact() {
    // F64Mod calls out, which ends the heap facts in the builder's model: the loop's read before
    // it checks the address again.
    let mut engine = warmed(
        "function md(t, n) { var s = t[0] | 0;
           for (var i = 0; i < n; i = (i + 1) | 0) { s = (s + (t[i] | 0)) | 0; s = s % 1000; }
           return s | 0; }
         var T = new Int32Array(64);
         for (var k = 0; k < 20; k++) md(T, 60);",
        "for (var k = 0; k < 20; k++) md(T, 60);",
    );
    let (chunk, graph) = graph_of(&mut engine, "md");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Mod), 1, "{text}");
    let header_pc = the_loop(&graph);
    assert_eq!(
        in_loop(&graph, header_pc, |op| matches!(
            op,
            NodeOp::TypedData { .. }
        ))
        .len(),
        1,
        "{text}"
    );
}

#[test]
fn a_binding_the_loop_stores_is_loaded_again_in_it() {
    // `b`'s value before the loop is no fact of the loop, which stores it: each iteration loads
    // it; `a`, which the loop does not store, keeps its value and its facts.
    let mut engine = warmed(
        "var mk = (function (g) {
           var a = new g.Int32Array(64); var b = 0;
           function st(n) { var s = (a[0] | 0) + (b | 0) | 0;
             for (var i = 0; i < n; i = (i + 1) | 0) { s = (s + (a[i] | 0) + (b | 0)) | 0; b = (b + 1) | 0; }
             return s | 0; }
           return { st: st };
         })(this);
         var st = mk.st;
         for (var k = 0; k < 20; k++) st(60);",
        "for (var k = 0; k < 20; k++) st(60);",
    );
    let (chunk, graph) = graph_of(&mut engine, "st");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let header_pc = the_loop(&graph);
    let loads = in_loop(&graph, header_pc, |op| {
        matches!(op, NodeOp::LoadBinding { .. })
    });
    assert_eq!(loads.len(), 1, "{text}");
    let stores = in_loop(&graph, header_pc, |op| {
        matches!(op, NodeOp::StoreBinding { .. })
    });
    assert_eq!(stores.len(), 1, "{text}");
    assert_eq!(
        loads[0].op,
        NodeOp::LoadBinding {
            slot: match stores[0].op {
                NodeOp::StoreBinding { slot } => slot,
                _ => unreachable!(),
            }
        },
        "{text}"
    );
    assert!(
        in_loop(&graph, header_pc, |op| matches!(
            op,
            NodeOp::TypedData { .. }
        ))
        .is_empty(),
        "{text}"
    );
}
