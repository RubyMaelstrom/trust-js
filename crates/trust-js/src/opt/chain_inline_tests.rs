//! P3 stage 3 step L2 meets step 2's graph inliner: a callee whose named loads go through the
//! prototype chain lowers in the rules (Rules::named_load_lowers), so E0 inlines it, and its walk
//! is built in its frame.

use super::*;

const CHAIN_CALLEES: &str = "
function P() { this.v = 1; } P.prototype.k = 5; P.prototype.m = function () { return 2; };
function getk(o) { return o.k; }
function c1(o) { return getk(o) + 1; }
function method(o) { return o.m; }
function c2(o) { return method(o) === P.prototype.m ? 1 : 0; }
var p = new P();
for (var k = 0; k < 30; k++) { c1(p); c2(p); }
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
