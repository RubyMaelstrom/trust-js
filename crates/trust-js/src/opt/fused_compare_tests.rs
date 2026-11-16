//! A Branch on a compare branches on the compare's flags (a64.rs `fused_tests`): an I32Cmp, F64Cmp
//! or TaggedSame whose one use is the Branch right after it sets the flags and the Branch takes the
//! condition or its inverse, no Boolean between, as Maglev's BranchIfInt32Compare and
//! BranchIfFloat64Compare. Every relation in each statement form (an if, a conditional, a negated
//! test, a loop's test at its top and its bottom) gives the VM's result, a NaN, ±0 and the
//! infinities among the Numbers and the int32 range's ends among the integers; a Boolean with
//! another use stays materialised, and one in a local dead after its Branch does not.

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

const RELATIONS: [&str; 8] = ["<", "<=", ">", ">=", "==", "!=", "===", "!=="];

/// Each relation in each statement form, over every pair of `values`, `rounds` times: the forms'
/// results folded into one number (each pair's own digit), so a wrong branch on any pair shows.
fn program(values: &str, rounds: u32) -> String {
    let mut source = String::new();
    let mut calls = Vec::new();
    for (k, op) in RELATIONS.iter().enumerate() {
        source.push_str(&format!(
            "function if{k}(a, b) {{ if (a {op} b) return 1; return 2; }}
             function cond{k}(a, b) {{ return (a {op} b) ? 3 : 4; }}
             function not{k}(a, b) {{ if (!(a {op} b)) return 5; return 6; }}
             function top{k}(a, b) {{ var n = 0; while (n < 3 && a {op} b) n++; return n; }}
             function bottom{k}(a, b) {{ var n = 0; do {{ n++; }} while (n < 3 && a {op} b); return n; }}
             function both{k}(a, b) {{ var n = 0; for (var i = 0; a {op} b && i < 2; i++) n += 2; return n; }}\n"
        ));
        for form in ["if", "cond", "not", "top", "bottom", "both"] {
            calls.push(format!("{form}{k}(a, b)"));
        }
    }
    let sum = calls.join(" * 7 + ");
    source.push_str(&format!(
        "var vs = [{values}];
         var t = 0;
         for (var r = 0; r < {rounds}; r++)
           for (var i = 0; i < vs.length; i++)
             for (var j = 0; j < vs.length; j++) {{
               var a = vs[i], b = vs[j];
               t = (t * 31 + ({sum})) % 1000003;
             }}
         t"
    ));
    source
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
            eval(&mut engine, source),
            expected,
            "exits every {exit_every}, graph {graph}"
        );
    }
}

#[test]
fn int32_compares_branch_as_the_vm_does() {
    agrees_at_every_switch(&program(
        "0, 1, -1, 2, 2147483647, -2147483648, 2147483646",
        60,
    ));
}

#[test]
fn number_compares_branch_as_the_vm_does_with_nan_zeros_and_infinities() {
    // A NaN on either side or both: every order false, == and === false, != and !== true, in
    // either branch sense; -0 equal to +0; the infinities ordered.
    agrees_at_every_switch(&program(
        "NaN, -0, 0, 1.5, -1.5, Infinity, -Infinity, 0.1, 2.5e300",
        40,
    ));
}

#[test]
fn a_constant_on_the_left_branches_as_the_vm_does() {
    // Each relation with a constant on its left. (Where the compare's left operand is an int32
    // constant node, cmp_i32 moves it to the right and swaps the condition, which the Branch
    // then takes: a64_tests' hand-built graph pins that case.)
    agrees_at_every_switch(
        "
        function lhs(a) {
          var n = 0;
          if (2 < a) n += 1; if (5 >= a) n += 2; if (3 > a) n += 4; if (4 <= a) n += 8;
          if (7 === a) n += 16; if (!(6 < a)) n += 32;
          return n;
        }
        var t = 0;
        for (var r = 0; r < 4000; r++) t = (t * 31 + lhs((r % 11) - 1)) % 1000003;
        t",
    );
}

#[test]
fn identity_compares_of_objects_branch_as_the_vm_does() {
    let source = "
        var o = {}, p = {}, q = [];
        function same(a, b) { if (a === b) return 1; return 2; }
        function differ(a, b) { var n = 0; while (a !== b && n < 2) n++; return n; }
        var vs = [o, p, q, o];
        var t = 0;
        for (var r = 0; r < 500; r++)
          for (var i = 0; i < 4; i++)
            for (var j = 0; j < 4; j++)
              t = (t * 31 + same(vs[i], vs[j]) * 3 + differ(vs[i], vs[j])) % 1000003;
        t";
    agrees_at_every_switch(source);
}

#[test]
fn typed_array_tests_callee_tests_and_constant_branches_branch_as_the_vm_does() {
    // An element read whose receivers are typed arrays and Arrays (IsTypedArray's Branch); a
    // method call whose callee is one of two functions, then also a third, whose guard's exit
    // (WrongCallee) makes the recompiled site's guards CalleeMatches and branches, the third
    // taking the call after them; a test the builder proves (`x == null` of a Number, a loop on
    // `true`), whose Branch is on a constant; negations of each.
    let source = "
        var f64 = new Float64Array([1.5, 2.5, 3.5, 4.5]), arr = [10, 20, 30, 40];
        var i32 = new Int32Array([7, 8, 9, 10]);
        function rd(a, i) { return a[i & 3]; }
        function ma() { return 1; } function mb() { return 2; } function mc() { return 4; }
        var oa = {m: ma}, ob = {m: mb}, oc = {m: mc};
        function c7(o) { return o.m(); }
        function nn(x) { var n = 0; if (x == null) n = 5; if (!(x == null)) n += 2; return n; }
        function spin(k) { var n = 0; while (true) { n++; if (n > k) break; } return n; }
        var rs = [f64, arr, i32, arr], os = [oa, ob, oa, ob];
        var t = 0;
        for (var r = 0; r < 12000; r++) {
          t = (t * 31 + rd(rs[r & 3], r)) % 1000003;
          t = (t * 31 + c7(r < 3000 ? os[r & 3] : r % 3 == 0 ? oc : os[r & 3])) % 1000003;
          t = (t * 31 + nn(r) + spin(r & 7)) % 1000003;
        }
        t";
    agrees_at_every_switch(source);
}

/// The fused nodes of `name`'s graph after `source` warmed it at the VM, by op.
fn fused_of(source: &str, name: &str) -> Vec<(NodeOp, bool)> {
    TestSwitches {
        on: false,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    eval(&mut engine, source);
    let chunk = chunk_of(&mut engine, name);
    let built = build(&chunk, &Snapshot::take(&chunk)).expect("the function builds");
    let graph = built.graph;
    let fused = super::a64::fused_tests(&graph);
    graph
        .nodes
        .iter()
        .zip(fused)
        .filter(|(node, _)| matches!(node.op, NodeOp::I32Cmp(_) | NodeOp::F64Cmp(_)))
        .map(|(node, on)| (node.op.clone(), on))
        .collect()
}

#[test]
fn a_compare_used_only_by_its_branch_is_fused_and_one_used_again_is_not() {
    let warm =
        "for (var r = 0; r < 300; r++) { f(r, 150); g(r, 150); h(r + 0.5, 150); k(r, 150); }";
    // The loop's test and the if's: each compare's one use is its Branch.
    let source = format!(
        "function f(a, b) {{ var n = 0; for (var i = 0; i < a; i++) {{ if (i > b) n++; }} return n; }}
         function g(a, b) {{ var c = a < b; if (c) return c; return 0; }}
         function h(a, b) {{ if (a < b) return 1; return 2; }}
         function k(a, b) {{ var c = a < b; if (c) {{ f(1, 0); }} return 1; }}
         {warm}"
    );
    let f = fused_of(&source, "f");
    assert!(
        f.len() >= 2 && f.iter().all(|&(_, on)| on),
        "f's compares: {f:?}"
    );
    let h = fused_of(&source, "h");
    assert!(
        h.iter()
            .any(|(op, on)| matches!(op, NodeOp::F64Cmp(_)) && *on),
        "h's compare: {h:?}"
    );
    // The Boolean the function returns as well: two uses, materialised.
    let g = fused_of(&source, "g");
    assert!(
        !g.is_empty() && g.iter().all(|&(_, on)| !on),
        "g's compare: {g:?}"
    );
    // A Boolean held in a local that nothing reads after the Branch: dead in the call's frame
    // state (`opt::liveness`), so its one use is still the Branch. (A frame state that names the
    // Boolean keeps it materialised: a64_tests' hand-built graph.)
    let k = fused_of(&source, "k");
    assert!(
        !k.is_empty() && k.iter().all(|&(_, on)| on),
        "k's compare: {k:?}"
    );
}
