//! A polymorphic read's arm test branches on its shape compares (a64.rs `fused_tests`): a
//! ShapeIs whose one use is the Branch right after it sets the flags that Branch takes, so no
//! Boolean is materialised between them, as Maglev's arms branch on the map compare itself
//! (BuildCompareMaps). The reads' results are the VM's for every arm and shape count, the arm
//! with several shapes (its compares chained by ccmp) among them.

use super::build::build;
use super::ir::*;
use super::snapshot::Snapshot;
use super::TestSwitches;
use crate::bytecode::{Chunk, Tier};
use crate::value::{ObjectKind, Value, ValueView};
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

fn chunk_of(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    let env = engine.interp.global_env;
    let value: Value = engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("a bound name");
    let ValueView::Obj(function) = value.view() else {
        panic!("{name} is not an object")
    };
    let ObjectKind::User(user) = &function.borrow().kind else {
        panic!("{name} is not a user function")
    };
    let runtime = user.func.runtime();
    runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .cloned()
        .unwrap_or_else(|| panic!("{name} stayed interpreted"))
}

/// The receivers: two shapes with `x` at one slot (S1, S2), one with it at the next slot (S3),
/// one through its prototype (S4); `read` reads `o.x`.
const SHAPES: &str = "
    function S1() { this.x = 1; this.a = 0; } function S2() { this.x = 2; this.b = 0; }
    function S3() { this.c = 0; this.x = 3; } function S4() {} S4.prototype.x = 4;
    function read(o) { return o.x; }";

/// `SHAPES` with a loop reading `pick(r)`'s receivers, its result the VM's at every tier switch.
fn program(pick: &str, rounds: u32) -> String {
    format!(
        "{SHAPES}
         var all = [new S1(), new S2(), new S3(), new S4()];
         var t = 0;
         for (var r = 0; r < {rounds}; r++) t = (t * 31 + read(all[{pick}])) % 1000003;
         t"
    )
}

fn vm(program: &str) -> String {
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, program)
}

#[test]
fn every_arm_of_a_fused_shape_test_reads_the_vms_value() {
    // Two shapes at two slots (one ShapeIs of one shape), three with two of them at one slot (an
    // arm of two shapes: cmp then ccmp), four with a prototype's (an arm that walks): each
    // receiver in turn, and each one alone after the others (the last arm, the fallthrough).
    for pick in [
        "r & 1 ? 2 : 0",
        "r % 3",
        "r & 3",
        "r < 4000 ? r & 3 : 3",
        "r < 4000 ? r % 3 : 2",
    ] {
        let source = program(pick, 12000);
        let expected = vm(&source);
        for (exit_every, graph) in [(0, false), (0, true), (7, false)] {
            TestSwitches {
                on: true,
                eager: exit_every != 0,
                exit_every,
                inline_graph: Some(graph),
                ..Default::default()
            }
            .set();
            let mut engine = Engine::new();
            engine.set_tier(Tier::Jit);
            assert_eq!(
                eval(&mut engine, &source),
                expected,
                "{pick}, exits every {exit_every}, graph {graph}"
            );
        }
    }
}

#[test]
fn a_polymorphic_reads_arm_tests_are_fused_into_their_branches() {
    // The two-slot read's graph: each arm's ShapeIs feeds only the Branch after it, so the back
    // end fuses every one of them; no other node is fused.
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, &program("r & 3", 400));
    let chunk = chunk_of(&mut engine, "read");
    let built = build(&chunk, &Snapshot::take(&chunk)).expect("read builds");
    let graph = built.graph;
    let tests: Vec<NodeIx> = (0..graph.nodes.len() as u32)
        .map(NodeIx)
        .filter(|&n| matches!(graph.node(n).op, NodeOp::ShapeIs(_)))
        .collect();
    assert!(!tests.is_empty(), "a polymorphic read: {:?}", graph.nodes);
    let fused = super::a64::fused_tests(&graph);
    for (index, &on) in fused.iter().enumerate() {
        let shape_is = matches!(graph.nodes[index].op, NodeOp::ShapeIs(_));
        assert_eq!(on, shape_is, "node {index}: {:?}", graph.nodes[index].op);
    }
}
