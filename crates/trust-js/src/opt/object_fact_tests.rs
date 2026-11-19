//! P3 stage 3 (c-tag): the builder's value fact that a word is an object's (`Forms::object`) and
//! the shape checks that take it. A check of a word that a CheckObject or CheckShape proved on
//! every path tests no tag (`NodeOp::CheckShape::object`): an effect does not end the fact (a
//! value's tag never changes), a join keeps it where every path proved it, a loop's body has its
//! entry's, and a phi, a new value, has none. The verifier refuses a check that omits its test on
//! a word no check proved on every path.

use super::*;

/// The graph's shape checks in code order (its blocks in reverse postorder, each body in order):
/// each one's node, input word and whether it tests no tag.
fn shape_checks(graph: &Graph) -> Vec<(NodeIx, NodeIx, bool)> {
    let mut out = Vec::new();
    for block in &graph.blocks {
        for &n in &block.body {
            if let NodeOp::CheckShape { object, .. } = graph.node(n).op {
                out.push((n, graph.node(n).inputs[0], object));
            }
        }
    }
    out
}

/// Whether each shape check tests no tag, in code order.
fn omitted(graph: &Graph) -> Vec<bool> {
    shape_checks(graph).iter().map(|&(_, _, o)| o).collect()
}

const FIXTURE: &str = "
    function P() { this.x = 1; this.y = 2; this.z = 3; }
    function g() {}
    function around(o, g) { var a = o.x; g(); return a + o.y; }
    function el(o, g) { var s = o[0]; g(); return s + o.x; }
    function upd(o, g) { o[0] += 1; g(); return o.x; }
    function numArm(w, c, d, g) {
      var a; if (c) a = w.x; else a = w * 2; g(); if (d) a += w.y; return a;
    }
    function join(o, c, g) { var a; if (c) a = o.x; else a = o.y; g(); return a + o.z; }
    function oneArm(o, c, g) { var a = 0; if (c) a = o.x; g(); return a + o.z; }
    function before(o, n, g) { var s = o.x; for (var i = 0; i < n; i++) { g(); s += o.y; } return s; }
    function inside(o, n, g) { var s = 0; for (var i = 0; i < n; i++) { g(); s += o.y; } return s; }
    function twoInside(o, n, g) {
      var s = 0; for (var i = 0; i < n; i++) { s += o.x; g(); s += o.y; } return s;
    }
    function phi(o, k, c, g) { var a = o.x + k.x; var w = c ? o : k; g(); return a + w.y; }
    var p = new P(), q = new P(), e = {x: 1}; e[0] = 5;
    for (var i = 0; i < 20; i++) {
      around(p, g); el(e, g); upd(e, g); join(p, i & 1, g); oneArm(p, i & 1, g);
      before(p, 3, g); inside(p, 3, g); twoInside(p, 3, g); phi(p, q, i & 1, g);
      numArm(i & 1 ? p : 3, i & 1, i & 1, g);
    }";

#[test]
fn a_check_of_a_word_a_check_proved_tests_no_tag_across_effects_and_joins() {
    let mut engine = engine(Tier::Jit);
    eval(&mut engine, FIXTURE);
    for (name, expected, effects) in [
        // The second read of `o` after the call: proven by the first read's check.
        ("around", &[false, true][..], 1),
        // Proven by the element read's CheckObject before the call.
        ("el", &[true][..], 1),
        // Proven by the compound assignment's CheckObject of its base (ToPropKey) before the call.
        ("upd", &[true][..], 1),
        // Each arm's check proves `o` on its path: the read after the join tests no tag.
        ("join", &[false, false, true][..], 1),
        // One arm alone proves it: the read after the join tests it.
        ("oneArm", &[false, false][..], 1),
        // A check before the loop proves the body's.
        ("before", &[false, true][..], 1),
        // The back edge alone proves nothing at the header.
        ("inside", &[false][..], 1),
        // In the body: the first read's check proves the second's, across the call.
        ("twoInside", &[false, true][..], 1),
        // One arm checks `w` an object's, the other proves it a Number: the read after the join
        // tests the tag (a join keeps the fact only where every predecessor holds it).
        ("numArm", &[false, false][..], 1),
    ] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(
            count(&graph, is_effect),
            effects,
            "{name}: the call\n{text}"
        );
        assert_eq!(omitted(&graph), expected, "{name}\n{text}");
        // Every check of the function reads one word: `o`.
        let checks = shape_checks(&graph);
        assert!(
            checks.iter().all(|&(_, word, _)| word == checks[0].1),
            "{name}\n{text}"
        );
    }
    // A phi of two proven objects is a new value: its check tests the tag.
    let (chunk, graph) = graph_of(&mut engine, "phi");
    let text = dump(&graph, &chunk);
    let checks = shape_checks(&graph);
    assert_eq!(omitted(&graph), [false, false, false], "{text}");
    assert_eq!(graph.node(checks[2].1).op, NodeOp::Phi, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::CheckObject), 0, "{text}");
}

#[test]
fn the_verifier_refuses_a_check_without_its_tag_test_on_a_word_no_check_proved() {
    let mut engine = engine(Tier::Jit);
    eval(&mut engine, FIXTURE);
    // As built, each graph holds the rule (`graph_of` verifies it); each check the builder left
    // testing its tag, made to omit it, is refused: the first read's (nothing before it), the
    // read after a join one arm alone proved, the loop's read the back edge alone proves, and a
    // phi's.
    for (name, check) in [("around", 0), ("oneArm", 1), ("inside", 0), ("phi", 2)] {
        let (chunk, mut graph) = graph_of(&mut engine, name);
        let (n, _, object) = shape_checks(&graph)[check];
        assert!(!object, "{name}");
        let NodeOp::CheckShape { object, .. } = &mut graph.nodes[n.index()].op else {
            unreachable!()
        };
        *object = true;
        let error = verify(&graph, &chunk).expect_err(name);
        assert!(error.contains("no check proved"), "{name}: {error}");
    }
}

/// The guard of a spliced sloppy callee that reads `this` (CheckCallee): `caller`'s receiver is
/// proven by the method load's shape check before the guard, so the guard does not test it again
/// (`this_object`); `callNum`'s, a Number's, is proven by nothing, and its guard tests it. Either
/// guard made to skip a test its receiver needs is refused.
#[test]
fn a_guard_of_a_receiver_a_check_proved_does_not_test_it_again() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function get() { return this.v; }
         function caller(o) { return o.get(); }
         Number.prototype.kind = function () { return typeof this; };
         function callNum(v) { return v.kind(); }
         var p = {v: 1, get: get};
         function drive(n) {
           var t = 0;
           for (var i = 0; i < n; i++) t += caller(p) + callNum(i).length;
           return t;
         }
         drive(150);",
    );
    for (name, proven) in [("caller", true), ("callNum", false)] {
        let Some(transformed) = second_stage(&mut engine, name, "drive(150)") else {
            return;
        };
        let mut graph = graph_for(&transformed, name);
        let text = dump(&graph, &transformed);
        let guards: Vec<NodeIx> = (0..graph.nodes.len() as u32)
            .map(NodeIx)
            .filter(|&n| {
                matches!(
                    graph.node(n).op,
                    NodeOp::CheckCallee { .. } | NodeOp::CalleeMatches { .. }
                ) && graph.node(n).inputs.len() == 2
            })
            .collect();
        let [guard] = guards[..] else {
            panic!("{name}: one guard of `this`\n{text}")
        };
        let (NodeOp::CheckCallee { this_object, .. } | NodeOp::CalleeMatches { this_object, .. }) =
            graph.node(guard).op
        else {
            unreachable!()
        };
        assert_eq!(this_object, proven, "{name}\n{text}");
        if proven {
            // The receiver word is the one the method load's shape check took.
            let receiver = graph.node(guard).inputs[1];
            assert!(
                shape_checks(&graph)
                    .iter()
                    .any(|&(_, word, _)| word == receiver),
                "{name}\n{text}"
            );
        } else {
            if let NodeOp::CheckCallee { this_object, .. }
            | NodeOp::CalleeMatches { this_object, .. } = &mut graph.nodes[guard.index()].op
            {
                *this_object = true;
            }
            let error = verify(&graph, &transformed).expect_err(name);
            assert!(error.contains("no check proved"), "{name}: {error}");
        }
    }
}

/// A guard that tests no receiver proves nothing: in `both`, whose String receiver's method loads
/// keep the receiver's word (the String prototype's check), the guard of `n`, a callee that reads
/// no `this`, does not test that word, so the guard of `l`, which reads it, still tests it (a
/// strict callee's guard tests none either, but no strict callee is spliced into a sloppy
/// function).
#[test]
fn a_guard_that_tests_no_receiver_proves_nothing() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "String.prototype.n = function () { return 'n'; };
         String.prototype.l = function () { return typeof this; };
         function both(v) { return v.n() + v.l(); }
         function drive(k) { var t = ''; for (var i = 0; i < k; i++) t = both('s' + (i & 3)); return t; }
         drive(150);",
    );
    let Some(transformed) = second_stage(&mut engine, "both", "drive(150)") else {
        return;
    };
    let graph = graph_for(&transformed, "both");
    let text = dump(&graph, &transformed);
    let guards: Vec<&Node> = nodes(&graph, |op| {
        matches!(
            op,
            NodeOp::CheckCallee { .. } | NodeOp::CalleeMatches { .. }
        )
    });
    assert_eq!(guards.len(), 2, "{text}");
    for guard in guards {
        let (NodeOp::CheckCallee { this_object, .. } | NodeOp::CalleeMatches { this_object, .. }) =
            guard.op
        else {
            unreachable!()
        };
        assert!(!this_object, "{text}");
    }
    // One guard has a receiver to test, `l`'s, and it is the word both calls take: the slot's.
    let receivers: Vec<NodeIx> = nodes(&graph, |op| {
        matches!(
            op,
            NodeOp::CheckCallee { .. } | NodeOp::CalleeMatches { .. }
        )
    })
    .iter()
    .filter_map(|guard| guard.inputs.get(1).copied())
    .collect();
    let [receiver] = receivers[..] else {
        panic!("one guard tests the receiver: {text}")
    };
    assert_eq!(graph.node(receiver).op, NodeOp::EntrySlot(0), "{text}");
}
