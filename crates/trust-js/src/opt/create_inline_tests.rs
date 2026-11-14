//! P3 stage 3 step L3 meets step 2's graph inliner: a callee whose stores create their property,
//! whose updates and appends are on Numbers or whose `length` is a String's lowers in the rules
//! (Rules::lowers), so E0 inlines it, and its nodes are built in its frame.

use super::*;

const CALLEES: &str = "
function setz(o, v) { o.z = v; }
function c1(o, v) { setz(o, v); return 1; }
function bump(o) { o.n++; o.m += 2; }
function c2(o) { bump(o); return o.n; }
function len(s) { return s.length; }
function c3(s) { return len(s) + 1; }
var b = { n: 0, m: 0.5 };
for (var k = 0; k < 30; k++) { c1({ y: k }, k); c2(b); c3('ab' + k); }
";

#[test]
fn callees_that_create_update_and_read_a_strings_length_are_inlined_in_their_frames() {
    let mut engine = warmed(CALLEES);
    for (name, which) in [
        (
            "c1",
            NodeOp::StoreNew {
                location: 0,
                shape: 0,
                key: 0,
                ways: 1,
            },
        ),
        ("c2", NodeOp::F64Add),
        ("c3", NodeOp::StringLength),
    ] {
        let compile = Compile::new(&mut engine, name);
        let built = compile.build();
        assert_eq!(
            outcomes(&built),
            vec![(0, InlineOutcome::Inlined)],
            "{name}: {}",
            dump(&built.graph)
        );
        let found = nodes(&built.graph, |node| {
            std::mem::discriminant(&node.op) == std::mem::discriminant(&which)
        });
        assert!(!found.is_empty(), "{name}: {}", dump(&built.graph));
        assert!(
            found
                .iter()
                .all(|&node| built.graph.node(node).frame == FrameIx(1)),
            "{name}: {}",
            dump(&built.graph)
        );
        assert!(
            nodes(&built.graph, |node| node.frame == FrameIx(1)
                && matches!(
                    node.op,
                    NodeOp::Effect { .. } | NodeOp::Generic { .. }
                ))
            .is_empty(),
            "{name}: {}",
            dump(&built.graph)
        );
    }
}
