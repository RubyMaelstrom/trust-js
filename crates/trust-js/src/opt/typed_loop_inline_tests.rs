//! P3 stage 3 step 0c meets step 2's graph inliner: an inlined callee's loop without effects
//! keeps the typed-array facts its caller's state brings (the fact's node the caller's) and
//! checks at its own entry edge, in its own frame, those it reads first; a caller's loop around
//! an inlined call keeps no value of a binding the callee stores.

use super::*;

/// An inlined callee's effect-free loop over a typed array: `rs` sums it, `ws` writes it; `g1`
/// reads it before the call, `g2` does not.
const CALLEE_LOOPS: &str = "
function rs(t, n) { var s = 0; for (var i = 0; i < n; i = (i + 1) | 0) s = (s + (t[i] | 0)) | 0; return s | 0; }
function ws(t, n) { for (var i = 0; i < n; i = (i + 1) | 0) t[i] = i; return n; }
function g1(t) { var a = t[0] | 0; return (a + rs(t, 16)) | 0; }
function g2(t) { return (ws(t, 16) + rs(t, 8)) | 0; }
var T = new Int32Array(16);
for (var k = 0; k < 30; k++) { g1(T); g2(T); }
";

/// The nodes of `frame` in the natural loop of the loop header of that frame (the one).
fn frame_loop_nodes(graph: &Graph, frame: FrameIx, is: impl Fn(&Node) -> bool) -> Vec<NodeIx> {
    let headers: Vec<usize> = (0..graph.blocks.len())
        .filter(|&b| {
            graph.blocks[b].loop_header && graph.node(graph.blocks[b].body[0]).frame == frame
        })
        .collect();
    let [header] = headers[..] else {
        panic!("one loop of frame {frame:?}: {headers:?}\n{}", dump(graph))
    };
    let mut members = vec![false; graph.blocks.len()];
    members[header] = true;
    let mut work: Vec<usize> = graph.blocks[header]
        .preds
        .iter()
        .map(|p| p.index())
        .filter(|&p| p >= header)
        .collect();
    while let Some(b) = work.pop() {
        if std::mem::replace(&mut members[b], true) {
            continue;
        }
        work.extend(graph.blocks[b].preds.iter().map(|p| p.index()));
    }
    nodes(graph, |node| members[node.block.index()] && is(node))
}

#[test]
fn an_inlined_callees_loop_keeps_the_typed_facts_its_caller_brings() {
    // g1 checked T's data address before the call: the callee's loop, in frame 1, reads it with
    // no check of its own; the fact's TypedData is the caller's (frame 0).
    let mut engine = warmed_on(CALLEE_LOOPS);
    let built = Compile::new(&mut engine, "g1").build();
    let graph = &built.graph;
    assert_eq!(graph.frames.len(), 1, "{}", dump(graph));
    let typed = |node: &Node| {
        matches!(
            node.op,
            NodeOp::TypedData { .. } | NodeOp::TypedView | NodeOp::CheckElements(_)
        )
    };
    assert!(
        frame_loop_nodes(graph, FrameIx(1), typed).is_empty(),
        "{}",
        dump(graph)
    );
    let loads = frame_loop_nodes(graph, FrameIx(1), |node| {
        matches!(node.op, NodeOp::LoadTyped(_))
    });
    let [load] = loads[..] else {
        panic!("one read in the callee's loop\n{}", dump(graph))
    };
    let data = graph.node(graph.node(load).inputs[1]);
    assert!(
        matches!(data.op, NodeOp::TypedData { .. }),
        "{}",
        dump(graph)
    );
    assert_eq!(data.frame, FrameIx::COMPILED, "{}", dump(graph));
}

#[test]
fn an_inlined_callees_loop_checks_its_typed_array_at_its_entry_in_its_frame() {
    // g2's first callee writes T in its loop with no check before: the callee's loop checks T at
    // its entry edge, for a store, each exit HoistingFailed in the callee's frame (1) at its
    // header's pc; the second callee's loop reads T with the facts the first left.
    let mut engine = warmed_on(CALLEE_LOOPS);
    let compile = Compile::new(&mut engine, "g2");
    let built = compile.build();
    let graph = &built.graph;
    assert_eq!(graph.frames.len(), 2, "{}", dump(graph));
    let hoisted = nodes(graph, |node| {
        node.reason == Some(ExitReason::HoistingFailed)
    });
    assert!(!hoisted.is_empty(), "{}", dump(graph));
    for &n in &hoisted {
        let node = graph.node(n);
        let state = graph.state(node.state.expect("a state"));
        assert_eq!(
            (node.frame, state.frame),
            (FrameIx(1), FrameIx(1)),
            "{}",
            dump(graph)
        );
        let header = (0..graph.blocks.len())
            .find(|&b| {
                graph.blocks[b].loop_header
                    && graph.node(graph.blocks[b].body[0]).frame == FrameIx(1)
            })
            .expect("ws's loop");
        assert_eq!(state.pc, graph.blocks[header].first_pc, "{}", dump(graph));
    }
    assert!(
        nodes(graph, |node| node.reason
            == Some(ExitReason::HoistingFailed)
            && node.op == NodeOp::TypedData { write: true })
        .len()
            == 1,
        "{}",
        dump(graph)
    );
    for frame in [FrameIx(1), FrameIx(2)] {
        assert!(
            frame_loop_nodes(graph, frame, |node| matches!(
                node.op,
                NodeOp::TypedData { .. }
            ))
            .is_empty(),
            "frame {frame:?}\n{}",
            dump(graph)
        );
    }
}

/// A caller's loop around an inlined callee that stores a closure binding the caller reads before
/// the loop and in it.
const STORES: &str = "
var mg = (function () {
  var c = 0;
  function bump() { c = (c + 1) | 0; }
  function lp(n) { var s = c | 0; for (var i = 0; i < n; i = (i + 1) | 0) { s = (s + c) | 0; bump(); } return s | 0; }
  return { lp: lp, bump: bump };
})();
var lp = mg.lp, bmp = mg.bump;
for (var k = 0; k < 30; k++) lp(8);
";

#[test]
fn a_callers_loop_reloads_a_binding_its_inlined_callee_stores() {
    // The planner splices no callee that stores a closure binding (a store requires the call's
    // context, bytecode.rs inline_capability), so the plan is made by hand, as the recursion
    // test makes one: bump's store of `c` ends every record's fact of c's slot, so lp's loop,
    // which the inlined call leaves free of effects, reads `c` from its record each iteration
    // (Loop::stored takes the inlined callees' stores, `Rules::inlined_stores`).
    let mut engine = warmed(STORES);
    let (func, realm) = function(&mut engine, "lp");
    let chunk = first_stage(&func);
    let plan = crate::bytecode::plan_inlines(&chunk, &func, realm, &engine.interp.global_env);
    assert!(plan.is_empty(), "the planner splices no storing callee");
    let env = engine.interp.global_env;
    let call = pc_of(&chunk, is_call) as usize;
    let (Op::Call(_, cache) | Op::CallWithThis(_, cache)) = chunk.jit_ops()[call] else {
        unreachable!()
    };
    let (bump, _) = function(&mut engine, "bmp");
    let object = engine
        .interp
        .get_var("bmp", &env)
        .ok()
        .unwrap()
        .as_obj()
        .unwrap();
    let way = crate::bytecode::InlineWay {
        f: bump,
        obj: object,
        check_this: false,
        uses_this: false,
        leaf: true,
        free_names: Vec::new(),
        expected_env: 0,
        expected_genv: crate::interpreter::Env::as_ptr(&env) as usize,
        genv_pin: crate::interpreter::Env::downgrade(&env),
        nested: Default::default(),
    };
    let mut plan = crate::fasthash::FastMap::default();
    plan.insert(cache, crate::bytecode::InlinePlanEntry { ways: vec![way] });
    let built = Compile::with(&mut engine, "lp", Request::Plan(plan)).build();
    let graph = &built.graph;
    assert_eq!(
        graph.frames.len(),
        1,
        "bump inlined: {:?}\n{}",
        outcomes(&built),
        dump(graph)
    );
    assert!(
        nodes(graph, |node| matches!(node.op, NodeOp::Effect { .. })).is_empty(),
        "{}",
        dump(graph)
    );
    // The slot bump stores (`c`'s), read again by lp's loop in lp's frame.
    let stores = nodes(graph, |node| {
        node.frame == FrameIx(1) && matches!(node.op, NodeOp::StoreBinding { .. })
    });
    let [store] = stores[..] else {
        panic!("bump's store\n{}", dump(graph))
    };
    let NodeOp::StoreBinding { slot } = graph.node(store).op else {
        unreachable!()
    };
    let reads = frame_loop_nodes(graph, FrameIx::COMPILED, |node| {
        node.frame == FrameIx::COMPILED && node.op == NodeOp::LoadBinding { slot }
    });
    assert!(!reads.is_empty(), "{}", dump(graph));
}

#[test]
fn typed_loops_and_stores_in_inlined_callees_give_the_tier_offs_results() {
    let source = format!(
        "{CALLEE_LOOPS}{STORES}
         var r = 0; for (var k = 0; k < 600; k++) {{ r = (r + g1(T) + g2(T) + lp(k % 9)) | 0;
           if (k === 400) T.buffer.transfer(); if (k === 450) T = new Int32Array(16); }}
         r + ':' + T.join()"
    );
    for (exit_every, stress) in [(0, false), (7, false), (0, true)] {
        let (expected, got) = both(&source, exit_every, stress);
        assert_eq!(got, expected, "exit every {exit_every}, stress {stress}");
    }
}
