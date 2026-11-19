//! A loop that keeps heap facts loads the globals it reads once, at its entry edge, and its header
//! keeps their values (F4 part 1; `Rules::global_source`, `Builder::loop_entry_checks`,
//! `loop_heap_facts`), as Ion's and the DFG's LICM move a global's guard and load out of a loop
//! that writes none (jit/LICM.cpp:154-172; DFGLICMPhase.cpp:388-413). A loop with a call keeps
//! its reads; every result is the VM's, a global reassigned between loops, one that is no typed
//! array at an entry, a script-level lexical and a field the loop stores among them.

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

fn agrees_at_every_switch(source: &str) {
    let expected = vm(source);
    for (exit_every, graph) in [(0, false), (0, true), (7, false), (7, true)] {
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
            eval(&mut engine, source),
            expected,
            "exits every {exit_every}, graph {graph}"
        );
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

/// The blocks of `name`'s graph that hold a LoadField of the global object, and the first loop
/// header's block (blocks are in reverse postorder: a loop's entry edge before its header, its
/// body after).
fn global_loads(source: &str, name: &str) -> (Vec<u32>, u32) {
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, source);
    let chunk = chunk_of(&mut engine, name);
    let snapshot = Snapshot::take_in(&chunk, Some(&engine.interp));
    let built = build(&chunk, &snapshot).expect("the function builds");
    let graph = built.graph;
    let header = graph
        .blocks
        .iter()
        .position(|block| block.loop_header)
        .expect("a loop") as u32;
    let blocks = graph
        .nodes
        .iter()
        .filter(|node| {
            matches!(node.op, NodeOp::LoadField { .. })
                && matches!(graph.node(node.inputs[0]).op, NodeOp::GlobalObject)
        })
        .map(|node| node.block.0)
        .collect();
    (blocks, header)
}

const HEAPS: &str = "
    var heap32 = new Int32Array(64), heap16 = new Int16Array(64), limit = 64;
    for (var k = 0; k < 64; k++) { heap32[k] = k * 7; heap16[k] = k & 15; }
    function f1(x) { return (x + 1) | 0; } function f2(x) { return (x ^ 3) | 0; }
    var table = [f1, f2];";

#[test]
fn an_effect_free_loop_loads_its_globals_at_its_entry() {
    let source = format!(
        "{HEAPS}
         function sum(n) {{
             var s = 0;
             for (var i = 0; i < n; i++) s = s ^ heap32[i & 63] ^ heap16[i & 63];
             return s;
         }}
         for (var r = 0; r < 50; r++) sum(200);"
    );
    let (blocks, header) = global_loads(&source, "sum");
    assert!(!blocks.is_empty(), "the loop's globals are read");
    assert!(
        blocks.iter().all(|&block| block < header),
        "every global load before the loop's header {header}: {blocks:?}"
    );
}

#[test]
fn a_loop_with_a_call_keeps_its_global_reads_in_its_body() {
    let source = format!(
        "{HEAPS}
         function sum(n) {{
             var s = 0;
             for (var i = 0; i < n; i++) {{
                 s = s ^ heap32[i & 63];
                 if ((i & 15) === 15) s = table[i & 1](s);
             }}
             return s;
         }}
         for (var r = 0; r < 50; r++) sum(200);"
    );
    let (blocks, header) = global_loads(&source, "sum");
    assert!(
        blocks.iter().any(|&block| block > header),
        "a global load in the loop's body: {blocks:?}, header {header}"
    );
}

#[test]
fn hoisted_globals_read_as_the_vm_does() {
    // A typed array and a Number bound read in an effect-free loop; the globals reassigned
    // between two runs of it (another array, another bound); a global that is no typed array at
    // one entry (its entry check fails: one exit, then the loop as before); nested loops.
    agrees_at_every_switch(&format!(
        "{HEAPS}
         function sum(n) {{
             var s = 0;
             for (var i = 0; i < n && i < limit; i++) s = s ^ heap32[i & 63] ^ heap16[i & 63];
             return s;
         }}
         function nest(n) {{
             var s = 0;
             for (var a = 0; a < n; a++) for (var b = 0; b < 8; b++) s = (s + heap32[(a + b) & 63]) | 0;
             return s;
         }}
         var t = 0;
         for (var r = 0; r < 300; r++) t = (t * 31 + sum(100) + nest(20)) % 1000003;
         heap32 = new Int32Array(64); for (var k = 0; k < 64; k++) heap32[k] = 1000 - k; limit = 40;
         for (var r = 0; r < 300; r++) t = (t * 31 + sum(100) + nest(20)) % 1000003;
         var keep = heap16; heap16 = [5, 6, 7];
         t = (t * 31 + sum(100)) % 1000003;
         heap16 = keep;
         for (var r = 0; r < 300; r++) t = (t * 31 + sum(100)) % 1000003;
         t"
    ));
}

#[test]
fn a_lexical_global_and_a_stored_field_read_as_the_vm_does() {
    // A script-level `let` typed array read in an effect-free loop; a loop that stores a global
    // property through `this` (sloppy, the global object) and reads it back as a free name.
    agrees_at_every_switch(
        "let lex = new Int32Array(32);
         for (let k = 0; k < 32; k++) lex[k] = k * 3;
         var g = 0;
         function lexsum(n) { var s = 0; for (var i = 0; i < n; i++) s = (s + lex[i & 31]) | 0; return s; }
         function store(n) { var s = 0; for (var i = 0; i < n; i++) { this.g = i; s = (s + g) | 0; } return s; }
         var t = 0;
         for (var r = 0; r < 300; r++) t = (t * 31 + lexsum(100) + store(50)) % 1000003;
         lex = new Int32Array(32);
         for (var r = 0; r < 300; r++) t = (t * 31 + lexsum(100) + store(50)) % 1000003;
         t",
    );
}

#[test]
fn a_global_changed_by_a_call_in_the_loop_reads_its_new_value() {
    // The global is read before the loop (its fact made there) and the loop's call reassigns it:
    // a loop with a call keeps no global's value through its header.
    agrees_at_every_switch(&format!(
        "{HEAPS}
         var other = new Int32Array(64); for (var k = 0; k < 64; k++) other[k] = 5000 + k;
         function swap() {{ var t = heap32; heap32 = other; other = t; return 0; }}
         function run(n) {{
             var s = heap32[1];
             for (var i = 0; i < n; i++) {{
                 s = (s + heap32[i & 63]) | 0;
                 if ((i & 31) === 31) s = (s + swap()) | 0;
             }}
             return s;
         }}
         var t = 0;
         for (var r = 0; r < 300; r++) t = (t * 31 + run(100)) % 1000003;
         t"
    ));
}
