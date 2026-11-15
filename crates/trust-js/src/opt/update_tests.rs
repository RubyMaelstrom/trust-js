//! P3 stage 3 step L3: named updates (UpdateProp, AppendProp) and a String's `length`, the
//! builder's graphs. An update on own writable data ways whose record saw Numbers is the
//! receiver's check, the field, the arithmetic and the store; `o.k += v` adds two Numbers and
//! stores; a `length` read whose site filled no way is StringLength, a value fact of the word.

use super::*;

fn stores(graph: &Graph) -> usize {
    count(graph, |op| matches!(op, NodeOp::StoreField { .. }))
}

#[test]
fn an_update_of_a_numbers_field_is_its_load_arithmetic_and_store() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function post(o) { return o.x++; }
         function pre(o) { return --o.x; }
         function drop(o) { o.x++; return o.x; }
         var o = { x: 1.5 };
         for (var i = 0; i < 20; i++) { post(o); pre(o); drop(o); }",
    );
    for name in ["post", "pre", "drop"] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        let pc = pc_of(&chunk, |op| matches!(op, Op::UpdateProp(..)));
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert_eq!(
            count(&graph, |op| matches!(op, NodeOp::CheckShape { .. })),
            1,
            "{name}: {text}"
        );
        assert_eq!(stores(&graph), 1, "{name}: {text}");
        // The arithmetic: f64 (the record holds nothing, rule U's Numbers: no int32 hint).
        let add = if name == "pre" {
            NodeOp::F64Sub
        } else {
            NodeOp::F64Add
        };
        assert_eq!(count(&graph, |op| *op == add), 1, "{name}: {text}");
        // Every exit before the store, at the update's pc.
        let store = graph
            .nodes
            .iter()
            .position(|node| matches!(node.op, NodeOp::StoreField { .. }))
            .expect("the store");
        for (k, node) in graph.nodes.iter().enumerate() {
            if let Some(state) = node.state.filter(|_| node.pc as usize == pc) {
                assert!(k < store, "{name}: an exit after the store: {text}");
                assert_eq!(graph.state(state).pc as usize, pc, "{name}: {text}");
            }
        }
        // `drop`'s read after the update is the stored word: no load again.
        if name == "drop" {
            assert_eq!(
                count(&graph, |op| matches!(op, NodeOp::LoadField { .. })),
                1,
                "{text}"
            );
        }
    }
}

#[test]
fn an_append_of_two_numbers_adds_and_stores() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function g(o, v) { o.k += v; return o.k; }
         var o = { k: 0.5 };
         for (var i = 0; i < 20; i++) g(o, i);",
    );
    let (chunk, graph) = graph_of(&mut engine, "g");
    let text = dump(&graph, &chunk);
    assert!(
        chunk
            .jit_ops()
            .iter()
            .any(|op| matches!(op, Op::AppendProp(..))),
        "{text}"
    );
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(count(&graph, |op| *op == NodeOp::F64Add), 1, "{text}");
    assert_eq!(stores(&graph), 1, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::CheckShape { .. })),
        1,
        "the read's check serves the store: {text}"
    );
}

#[test]
fn an_update_whose_record_saw_a_string_stays_the_effect() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function f(o) { return o.x++; }
         for (var i = 0; i < 20; i++) f({ x: '7' });",
    );
    let (chunk, graph) = graph_of(&mut engine, "f");
    let text = dump(&graph, &chunk);
    let pc = pc_of(&chunk, |op| matches!(op, Op::UpdateProp(..)));
    assert!(
        graph
            .nodes
            .iter()
            .any(|node| matches!(node.op, NodeOp::Effect { pc: at, .. }
                | NodeOp::Generic { pc: at, .. } if at as usize == pc)),
        "{text}"
    );
}

#[test]
fn a_strings_length_is_its_tag_check_and_header_read_once() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function len(s) { return s.length + s.length; }
         for (var i = 0; i < 20; i++) len('abc' + i);",
    );
    let (chunk, graph) = graph_of(&mut engine, "len");
    let text = dump(&graph, &chunk);
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    let lengths = nodes(&graph, |op| *op == NodeOp::StringLength);
    assert_eq!(lengths.len(), 1, "a value fact: {text}");
    assert_eq!(lengths[0].rep, Rep::I32, "{text}");
    assert_eq!(lengths[0].reason, Some(ExitReason::WrongShape), "{text}");
}

/// P3 stage 3 step L3 D: the rules say these forms lower, so a loop holding them is free of
/// effects: its invariant `a` is one value through the loop, read from no frame slot again.
#[test]
fn loops_of_updates_creations_and_lengths_are_free_of_effects() {
    let mut engine = engine(Tier::Jit);
    eval(
        &mut engine,
        "function up(o, a, n) { let s = 0; for (let i = 0; i < n; i++) { o.x++; s = s + a; } return s; }
         function app(o, a, n) { let s = 0; for (let i = 0; i < n; i++) { o.k += 1; s = s + a; } return s; }
         function len(t, a) { let s = 0; for (let i = 0; i < t.length; i++) s = s + a; return s; }
         function mk(os, a, n) { let s = 0; for (let i = 0; i < n; i++) { os[i].z = i; s = s + a; } return s; }
         for (var r = 0; r < 20; r++) {
           up({ x: 1 }, 0.5, 3); app({ k: 1 }, 0.5, 3); len('abcd', 0.5);
           mk([{ y: 1 }, { y: 2 }, { y: 3 }], 0.5, 3);
         }",
    );
    for name in ["up", "app", "len"] {
        let (chunk, graph) = graph_of(&mut engine, name);
        let text = dump(&graph, &chunk);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert_eq!(
            count(&graph, |op| matches!(op, NodeOp::FrameSlot(_))),
            0,
            "{name}: no reload in the loop: {text}"
        );
    }
    // The creation's chain needs the realm's snapshot.
    let chunk = chunk_of(&mut engine, "mk");
    let snapshot = Snapshot::take_in(&chunk, Some(&engine.interp));
    let graph = build(&chunk, &snapshot).expect("built").graph;
    let text = dump(&graph, &chunk);
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::StoreNew { .. })),
        1,
        "{text}"
    );
    assert_eq!(count(&graph, is_effect), 0, "{text}");
    assert_eq!(
        count(&graph, |op| matches!(op, NodeOp::FrameSlot(_))),
        0,
        "mk: no reload in the loop: {text}"
    );
}
