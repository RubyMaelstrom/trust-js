//! P3 stage 3 step L2 meets step 2's graph inliner: a callee whose named loads go through the
//! prototype chain lowers in the rules (Rules::named_load_lowers), so E0 inlines it, and its walk
//! is built in its frame.

use super::*;

const CHAIN_CALLEES: &str = "
function P() { this.v = 1; } P.prototype.k = 5; P.prototype.m = function () { return 2; };
function getk(o) { return o.k; }
function c1(o) { return getk(o) + 1; }
function method(o) { return o.m; }
function slot(o) { return o.x; }
function c3(o) { return slot(o) + 1; }
function c2(o) { return method(o) === P.prototype.m ? 1 : 0; }
var p = new P();
var q1 = {x: 1}, q2 = {y: 2, x: 3};
for (var k = 0; k < 30; k++) { c1(p); c2(p); c3(q1); c3(q2); }
";

#[test]
fn a_callee_whose_loads_go_through_the_chain_is_inlined_with_its_walk_in_its_frame() {
    let mut engine = warmed(CHAIN_CALLEES);
    for name in ["c1", "c2"] {
        let compile = Compile::new(&mut engine, name);
        let built = compile.build();
        assert_eq!(
            outcomes(&built),
            vec![(0, InlineOutcome::Inlined)],
            "{name}: {}",
            dump(&built.graph)
        );
        let hops = nodes(&built.graph, |node| {
            matches!(node.op, NodeOp::CheckProto { .. })
        });
        assert_eq!(hops.len(), 1, "{name}: {}", dump(&built.graph));
        let hop = built.graph.node(hops[0]);
        assert_eq!(hop.frame, FrameIx(1), "{name}: {}", dump(&built.graph));
        assert!(
            nodes(&built.graph, |node| node.frame == FrameIx(1)
                && matches!(node.op, NodeOp::Effect { .. }))
            .is_empty(),
            "{name}: {}",
            dump(&built.graph)
        );
    }
}

#[test]
fn a_callee_whose_load_dispatches_on_the_receivers_shape_is_inlined() {
    let mut engine = warmed(CHAIN_CALLEES);
    let compile = Compile::new(&mut engine, "c3");
    let built = compile.build();
    assert_eq!(
        outcomes(&built),
        vec![(0, InlineOutcome::Inlined)],
        "{}",
        dump(&built.graph)
    );
    let tests = nodes(&built.graph, |node| matches!(node.op, NodeOp::ShapeIs(_)));
    assert_eq!(tests.len(), 1, "{}", dump(&built.graph));
    assert_eq!(
        built.graph.node(tests[0]).frame,
        FrameIx(1),
        "{}",
        dump(&built.graph)
    );
}
