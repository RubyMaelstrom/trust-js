//! P3 stage 3 step L3: named stores that create their property, the builder's graphs. A site
//! whose ways are creations (bytecode.rs IC_CREATE) of one receiver shape and one landing, whose
//! prototypes' chains the snapshot walked (with the realm: `Snapshot::take_in`), is the receiver's
//! shape check, a CheckProto per level of its chain and CheckNoProto at its end, CheckCreatable
//! (extensible, unmarked) and StoreNew; after it, the receiver's shape is the landing one, and
//! every other value whose shape fact holds the old shape loses it.

use super::*;
use crate::feedback::{NamedWay, NamedWayKind};

/// The verified graph of the global function `name`, its snapshot taken in the engine's realm (a
/// creation's chain is walked through the realm's pins).
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

/// The filled ways of the named site at `pc`.
fn ways_at(chunk: &Chunk, pc: usize) -> Vec<NamedWay> {
    let snapshot = Snapshot::take(chunk);
    let named = snapshot.named(pc).expect("a named view");
    named.filled().copied().collect()
}

fn creations(graph: &Graph) -> Vec<&Node> {
    nodes(graph, |op| matches!(op, NodeOp::StoreNew { .. }))
}

fn proofs(graph: &Graph) -> Vec<&Node> {
    nodes(graph, |op| matches!(op, NodeOp::CheckCreatable { .. }))
}

/// Every check and creation of the operation at `pc` exits WrongShape at `pc`, its state the
/// operation's before-state: nothing exits after a write.
fn assert_creation_exits_at(graph: &Graph, pc: usize, text: &str) {
    for node in graph.nodes.iter().filter(|node| {
        node.pc as usize == pc
            && matches!(
                node.op,
                NodeOp::CheckShape(_)
                    | NodeOp::CheckProto { .. }
                    | NodeOp::CheckNoProto { .. }
                    | NodeOp::CheckCreatable { .. }
                    | NodeOp::StoreNew { .. }
            )
    }) {
        assert_eq!(node.reason, Some(ExitReason::WrongShape), "{text}");
        let state = graph.state(node.state.expect("a check's state"));
        assert_eq!(state.pc as usize, pc, "{text}");
    }
}

#[test]
fn a_constructors_creations_are_its_receivers_checks_once_and_its_stores() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function P(x, y) { this.x = x; this.y = y; }
         for (var i = 0; i < 20; i++) new P(i, i + 1);",
    );
    let (chunk, graph) = graph_in(&mut engine, "P");
    let text = dump(&graph, &chunk);
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| matches!(chunk.jit_ops()[pc], Op::SetPropThisDrop(..)))
        .collect();
    assert_eq!(pcs.len(), 2, "{text}");
    let first = ways_at(&chunk, pcs[0]);
    let second = ways_at(&chunk, pcs[1]);
    assert!(
        first
            .iter()
            .chain(&second)
            .all(|way| way.kind == NamedWayKind::Creation),
        "{text}"
    );
    // No store is an effect: two creations, one receiver check, one proof of the chain.
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let stores = creations(&graph);
    assert_eq!(stores.len(), 2, "{text}");
    assert_eq!(
        stores[0].op,
        NodeOp::StoreNew {
            location: first[0].location,
            shape: first[0].holder_shape,
            key: chunk.freeze().jit_name_key(match chunk.jit_ops()[pcs[0]] {
                Op::SetPropThisDrop(name, _) => name,
                _ => unreachable!(),
            }),
            ways: 1
        },
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckShape(_))),
        1,
        "the second creation's receiver is the first's landing shape: {text}"
    );
    assert_eq!(
        nodes(&graph, |op| matches!(op, NodeOp::CheckShape(_)))[0].op,
        NodeOp::CheckShape(vec![first[0].receiver_shape]),
        "{text}"
    );
    // The second way's receiver is the first's landing.
    assert_eq!(second[0].receiver_shape, first[0].holder_shape, "{text}");
    // P.prototype, then Object.prototype, then null: two levels by their shapes and the end,
    // and the flags, each once.
    assert_eq!(proofs(&graph).len(), 1, "{text}");
    let levels = nodes(&graph, |op| matches!(op, NodeOp::CheckProto { .. }));
    assert_eq!(levels.len(), 2, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckNoProto { .. })),
        1,
        "{text}"
    );
    assert_ne!(first[0].proto, 0, "P.prototype: {text}");
    for &pc in &pcs {
        assert_creation_exits_at(&graph, pc, &text);
    }
}

#[test]
fn after_a_creation_its_field_and_its_landing_shape_are_known() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function P(x) { this.x = x; return this.x + this.x; }
         for (var i = 0; i < 20; i++) new P(i);",
    );
    let (chunk, graph) = graph_in(&mut engine, "P");
    let text = dump(&graph, &chunk);
    assert_eq!(creations(&graph).len(), 1, "{text}");
    // The loads read the stored value: no check, no load.
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckShape(_))),
        1,
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadField { .. })),
        0,
        "{text}"
    );
}

#[test]
fn a_creation_ends_the_shape_facts_that_hold_its_old_shape_and_keeps_the_others() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function same(a, b, v) { var x = a.k; b.z = v; return x + a.k; }
         function other(a, b, v) { var x = a.k; b.z = v; return x + a.k; }
         for (var i = 0; i < 20; i++) {
           same({ k: i }, { k: 0 }, i);
           other({ k: i, j: 1 }, { k: 0 }, i);
         }",
    );
    // `same`: a and b share the shape the creation leaves, so a may be b: its shape is checked
    // again after the creation (its field, at another location, stays known).
    let (chunk, graph) = graph_in(&mut engine, "same");
    let text = dump(&graph, &chunk);
    assert_eq!(creations(&graph).len(), 1, "{text}");
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| matches!(chunk.jit_ops()[pc], Op::GetPropLocal(..)))
        .collect();
    assert_eq!(pcs.len(), 2, "{text}");
    let checked_at = |pc: usize| {
        graph
            .nodes
            .iter()
            .filter(|node| node.pc as usize == pc && matches!(node.op, NodeOp::CheckShape(_)))
            .count()
    };
    assert_eq!(checked_at(pcs[0]), 1, "{text}");
    assert_eq!(checked_at(pcs[1]), 1, "a's shape fact ended: {text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadField { .. })),
        1,
        "a.k's word is known across the creation at another location: {text}"
    );
    // `other`: a's shape is not the creation's receiver's, so a is not b: no check again.
    let (chunk, graph) = graph_in(&mut engine, "other");
    let text = dump(&graph, &chunk);
    assert_eq!(creations(&graph).len(), 1, "{text}");
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| matches!(chunk.jit_ops()[pc], Op::GetPropLocal(..)))
        .collect();
    let checked_at = |pc: usize| {
        graph
            .nodes
            .iter()
            .filter(|node| node.pc as usize == pc && matches!(node.op, NodeOp::CheckShape(_)))
            .count()
    };
    assert_eq!(checked_at(pcs[0]), 1, "{text}");
    assert_eq!(
        checked_at(pcs[1]),
        0,
        "a's fact excludes the old shape: {text}"
    );
}

#[test]
fn a_walk_through_the_creations_receiver_checks_its_level_again() {
    // c's method is on base, two levels up through p; the creation on p, c's prototype, between
    // the two loads ends the fact of p's level (p's shape is the creation's old one), so the
    // second load walks again (and exits where p now shadows the method).
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "var base = { m: function () { return 1; } };
         function g(c, p, v) { var a = c.m; p.n = v; return c.m; }
         for (var i = 0; i < 20; i++) { var p = Object.create(base); g(Object.create(p), p, i); }",
    );
    let (chunk, graph) = graph_in(&mut engine, "g");
    let text = dump(&graph, &chunk);
    assert_eq!(creations(&graph).len(), 1, "{text}");
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| matches!(chunk.jit_ops()[pc], Op::GetPropLocal(..)))
        .collect();
    assert_eq!(pcs.len(), 2, "{text}");
    let hops_at = |pc: usize| {
        graph
            .nodes
            .iter()
            .filter(|node| node.pc as usize == pc && matches!(node.op, NodeOp::CheckProto { .. }))
            .count()
    };
    assert_eq!(hops_at(pcs[0]), 2, "{text}");
    assert_eq!(
        hops_at(pcs[1]),
        2,
        "the walk's facts through p ended: {text}"
    );
}

#[test]
fn what_no_template_creates_and_setters_stay_effects() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function L(v) { this.length = v; }
         function T(v) { this.t = v; }
         Object.defineProperty(T.prototype, 't', { set(v) { this.u = v; }, configurable: true });
         for (var i = 0; i < 20; i++) { new L(i); new T(i); }",
    );
    for name in ["L", "T"] {
        let (chunk, graph) = graph_in(&mut engine, name);
        let text = dump(&graph, &chunk);
        let pc = pc_of(&chunk, |op| matches!(op, Op::SetPropThisDrop(..)));
        assert!(creations(&graph).is_empty(), "{name}: {text}");
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| matches!(node.op, NodeOp::Effect { pc: at, .. } if at as usize == pc)),
            "{name}: {text}"
        );
    }
}
