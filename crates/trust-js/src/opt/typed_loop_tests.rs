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
