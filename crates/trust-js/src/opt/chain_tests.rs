//! P3 stage 3 step L2: named loads through the prototype chain, the builder's graphs. A data way
//! at depth 1 to 3 is the receiver's shape check, a CheckProto per level reached by its
//! [[Prototype]] (the levels between by their recorded shapes, the holder's last) and the
//! holder's field; ways that share the walk check their receiver shapes together; the links,
//! shapes and the holder's field are facts of the effect epoch.

use super::*;
use crate::feedback::NamedWay;

/// The one filled way of the named site at `pc`.
fn way_at(chunk: &Chunk, pc: usize) -> NamedWay {
    let snapshot = Snapshot::take(chunk);
    let named = snapshot.named(pc).expect("a named view");
    let mut ways = named.filled();
    let way = *ways.next().expect("a filled way");
    assert!(ways.next().is_none(), "one way");
    way
}

fn protos(graph: &Graph) -> Vec<&Node> {
    nodes(graph, |op| matches!(op, NodeOp::CheckProto { .. }))
}

/// The checks of a load from `pc`: every one exits WrongShape at `pc` with the receiver on the
/// stack (the operation's before-state).
fn assert_exits_at(graph: &Graph, pc: usize, text: &str) {
    for node in graph
        .nodes
        .iter()
        .filter(|node| matches!(node.op, NodeOp::CheckShape(_) | NodeOp::CheckProto { .. }))
    {
        assert_eq!(node.reason, Some(ExitReason::WrongShape), "{text}");
        let state = graph.state(node.state.expect("a check's state"));
        assert_eq!(state.pc as usize, pc, "{text}");
    }
}

#[test]
fn a_method_on_the_prototype_is_its_receivers_check_the_holders_and_its_field() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "class A { m() { return 1; } } function f(o) { return o.m(); }
         var a = new A(); for (var i = 0; i < 20; i++) f(a);",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    let text = dump(&graph, &chunk);
    let pc = pc_of(&chunk, |op| matches!(op, Op::GetMethod(..)));
    let way = way_at(&chunk, pc);
    assert_eq!(way.depth, 1, "{text}");
    // The call stays the effect; the method load is the tier's.
    assert_eq!(count(&graph, is_effect), 1, "{text}");
    assert!(
        !graph
            .nodes
            .iter()
            .any(|node| matches!(node.op, NodeOp::Effect { pc: at, .. } if at as usize == pc)),
        "{text}"
    );
    let checks = nodes(&graph, |op| matches!(op, NodeOp::CheckShape(_)));
    assert_eq!(checks.len(), 1, "{text}");
    assert_eq!(
        checks[0].op,
        NodeOp::CheckShape(vec![way.receiver_shape]),
        "{text}"
    );
    let hops = protos(&graph);
    assert_eq!(hops.len(), 1, "{text}");
    assert_eq!(
        hops[0].op,
        NodeOp::CheckProto {
            shapes: vec![way.holder_shape],
            ways: 1
        },
        "{text}"
    );
    // The hop reads the receiver's checked cell; the field is the holder's.
    let receiver_check = graph
        .nodes
        .iter()
        .position(|node| matches!(node.op, NodeOp::CheckShape(_)))
        .expect("the receiver's check");
    assert_eq!(
        hops[0].inputs,
        vec![NodeIx(receiver_check as u32)],
        "{text}"
    );
    let loads = nodes(&graph, |op| matches!(op, NodeOp::LoadField { .. }));
    assert_eq!(loads.len(), 1, "{text}");
    assert_eq!(
        loads[0].op,
        NodeOp::LoadField {
            location: way.location,
            slot: way.slot
        },
        "{text}"
    );
    let hop = graph
        .nodes
        .iter()
        .position(|node| matches!(node.op, NodeOp::CheckProto { .. }))
        .expect("the hop");
    assert_eq!(loads[0].inputs, vec![NodeIx(hop as u32)], "{text}");
    assert_exits_at(&graph, pc, &text);
}

#[test]
fn deeper_holders_check_every_level_between_by_its_recorded_shape() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "class A { m() { return 1; } } class B extends A { n() { return 2; } }
         class C extends B { k() { return 3; } }
         function two(o) { return o.m(); } function three(o) { return o.m(); }
         var b = new B(), c = new C();
         for (var i = 0; i < 20; i++) { two(b); three(c); }",
    );
    for (name, depth) in [("two", 2), ("three", 3)] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        let pc = pc_of(&chunk, |op| matches!(op, Op::GetMethod(..)));
        let way = way_at(&chunk, pc);
        assert_eq!(way.depth, depth, "{name}: {text}");
        let levels = way.levels().expect("recorded levels");
        assert_eq!(levels.len(), depth as usize, "{text}");
        assert_eq!(*levels.last().unwrap(), way.holder_shape, "{text}");
        let hops: Vec<NodeOp> = protos(&graph).iter().map(|n| n.op.clone()).collect();
        let expected: Vec<NodeOp> = levels
            .iter()
            .map(|&shape| NodeOp::CheckProto {
                shapes: vec![shape],
                ways: 1,
            })
            .collect();
        assert_eq!(hops, expected, "{name}: {text}");
        // Each hop reads the one below it.
        let hop_ixs: Vec<usize> = (0..graph.nodes.len())
            .filter(|&k| matches!(graph.nodes[k].op, NodeOp::CheckProto { .. }))
            .collect();
        for pair in hop_ixs.windows(2) {
            assert_eq!(
                graph.nodes[pair[1]].inputs,
                vec![NodeIx(pair[0] as u32)],
                "{text}"
            );
        }
        assert_eq!(count(&graph, is_effect), 1, "the call: {text}");
        assert_exits_at(&graph, pc, &text);
    }
}

#[test]
fn receivers_of_several_shapes_that_share_the_walk_are_checked_together() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function P() {} P.prototype.m = function () { return 1; };
         function f(o) { return o.m(); }
         var x = new P(), y = new P(); y.extra = 1;
         for (var i = 0; i < 20; i++) { f(x); f(y); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    let text = dump(&graph, &chunk);
    let checks = nodes(&graph, |op| matches!(op, NodeOp::CheckShape(_)));
    assert_eq!(checks.len(), 1, "{text}");
    assert!(
        matches!(&checks[0].op, NodeOp::CheckShape(shapes) if shapes.len() == 2),
        "{text}"
    );
    let hops = protos(&graph);
    assert_eq!(hops.len(), 1, "{text}");
    assert!(
        matches!(&hops[0].op, NodeOp::CheckProto { shapes, ways: 2 } if shapes.len() == 1),
        "{text}"
    );
    assert_eq!(count(&graph, is_effect), 1, "the call: {text}");
}

#[test]
fn several_walks_another_kind_of_way_or_a_primitive_stay_the_effect() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "class A { m() { return 1; } } class B { n() { return 0; } m() { return 2; } }
         class G { get m() { return 3; } }
         function walks(o) { return o.m(); } function getter(o) { return o.m; }
         function absent(o) { return o.nothing; } function prim(s) { return s.charAt(0); }
         function proto(f) { return f.prototype; } function index(o) { return o['3']; }
         function mixed(o) { return o.m; }
         function prim1(s) { return s.hasOwnProperty('length'); }
         function inherited(o) { return typeof o.prototype; }
         var a = new A(), b = new B(), g = new G(), own = {m: 5};
         A.prototype[3] = 1;
         for (var i = 0; i < 20; i++) {
           walks(a); walks(b); getter(g); absent(a); prim('xy'); proto(A); index(a);
           mixed(a); mixed(own); prim1('xy'); inherited(Object.create(function () {}));
         }",
    );
    // A string's way through String.prototype to Object.prototype (depth 1, its primitive
    // marked); a `prototype` a function holds for the objects made from it (a deferred one in
    // its slot until the checked operation materializes it).
    for name in [
        "walks",
        "getter",
        "absent",
        "prim",
        "proto",
        "index",
        "mixed",
        "prim1",
        "inherited",
    ] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(protos(&graph).len(), 0, "{name}: {text}");
        assert!(count(&graph, is_effect) >= 1, "{name}: {text}");
    }
}

#[test]
fn the_links_and_the_holders_field_are_facts_until_an_effect() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function P() { this.v = 1; } P.prototype.a = 2; P.prototype.b = 3;
         function both(o) { return o.a + o.b + o.a; }
         function around(o, g) { var x = o.a; g(); return x + o.a; }
         var p = new P(); function nothing() {}
         for (var i = 0; i < 20; i++) { both(p); around(p, nothing); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "both");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckShape(_))),
        1,
        "one receiver check: {text}"
    );
    assert_eq!(protos(&graph).len(), 1, "one hop: {text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadField { .. })),
        2,
        "a and b, a once: {text}"
    );
    let (chunk, graph) = graph_of(&mut engine, "around");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 1, "the call: {text}");
    assert_eq!(protos(&graph).len(), 2, "checked again after it: {text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadField { .. })),
        2,
        "{text}"
    );
}

#[test]
fn a_lowered_store_at_the_holders_location_ends_its_field() {
    // `P.m = x` stores to an own data way of the prototype object: the holder's field at that
    // location is loaded again after it (any object's field there may be the one stored).
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function P() {} P.prototype.m = 1;
         function f(o, q) { var a = o.m; q.m = a + 1; return o.m; }
         var p = new P();
         for (var i = 0; i < 20; i++) f(p, P.prototype);",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::StoreField { .. })),
        1,
        "{text}"
    );
    assert_eq!(protos(&graph).len(), 1, "the link stays proven: {text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::LoadField { .. })),
        2,
        "the field is loaded again: {text}"
    );
}

#[test]
fn a_flagged_walk_is_declined_unless_its_ways_grew() {
    // A WrongShape exit of the walk flags the site: the recompile keeps the effect, as an own
    // way's flag does (`stale_shapes`), a hop's exit keeping its access's receiver count.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function P() {} P.prototype.m = function () { return 1; };
         function f(o) { return o.m(); }
         var x = new P(); for (var i = 0; i < 20; i++) f(x);",
    );
    let chunk = chunk_of(&mut engine, "f");
    let pc = pc_of(&chunk, |op| matches!(op, Op::GetMethod(..)));
    flag(&chunk, pc, ExitReason::WrongShape, false, 1);
    let graph = graph_for(&chunk, "f flagged");
    let text = dump(&graph, &chunk);
    assert_eq!(protos(&graph).len(), 0, "{text}");
    assert_eq!(count(&graph, is_effect), 2, "the load and the call: {text}");
    assert_eq!(declines(&graph), [(pc, ExitReason::WrongShape)], "{text}");
    // A second receiver shape since, sharing the walk: both checked again.
    eval(&mut engine, "var y = new P(); y.extra = 1; f(y);");
    let graph = graph_for(&chunk, "f with a new way");
    let text = dump(&graph, &chunk);
    assert_eq!(protos(&graph).len(), 1, "{text}");
    assert!(
        matches!(&nodes(&graph, |op| matches!(op, NodeOp::CheckShape(_)))[0].op,
            NodeOp::CheckShape(shapes) if shapes.len() == 2),
        "{text}"
    );
    assert!(graph.declines.is_empty(), "{text}");
}

#[test]
fn prototypes_of_one_shape_share_a_way_and_the_walk_reads_the_one_it_reaches() {
    // A's and B's prototypes have one shape ({constructor, m}), so the receivers of both hit one
    // way: the walk checks the shape of the object the link reaches and loads its slot, no
    // identity compared (as the templates' probe does).
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "class A { m() { return 1; } } class B { m() { return 2; } }
         function f(o) { return o.m(); }
         var a = new A(), b = new B(); for (var i = 0; i < 20; i++) { f(a); f(b); }",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    let text = dump(&graph, &chunk);
    let pc = pc_of(&chunk, |op| matches!(op, Op::GetMethod(..)));
    let way = way_at(&chunk, pc);
    assert_eq!(way.depth, 1, "{text}");
    assert_eq!(protos(&graph).len(), 1, "{text}");
    assert_eq!(eval(&mut engine, "f(a) * 10 + f(b)"), "12");
}

#[test]
fn a_loop_whose_loads_go_through_the_chain_is_free_of_effects() {
    // The rules lower the walk (Rules::named_load_lowers): the loop keeps its slots' values
    // through its header, with no reload from the frame and no flush.
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function P() { this.v = 1; } P.prototype.k = 5;
         function sum(o, n) { var s = 0; for (var i = 0; i < n; i++) s = s + o.k; return s; }
         var p = new P(); for (var i = 0; i < 20; i++) sum(p, 3);",
    );
    let (chunk, graph) = graph_of(&mut engine, "sum");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(protos(&graph).len(), 1, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(
            op,
            NodeOp::FrameSlot(_) | NodeOp::StoreSlot(_)
        )),
        0,
        "{text}"
    );
}
