//! Equality and typeof tests as the tier's own nodes (P3 stage 3, step L1): the builder's forms
//! for each case and what stays the generic operation, and end to end, the tier's results over a
//! table of operand pairs equal the VM's.

use super::build::build;
use super::ir::*;
use super::snapshot::Snapshot;
use crate::bytecode::{Chunk, Tier};
use crate::value::{ObjectKind, TypeofTest, Value, ValueView};
use crate::{Completion, Engine};
use std::fmt::Write as _;
use std::rc::Rc;

/// An engine at `tier` that compiles every function at its first call, the tier's switches off
/// whatever the environment's (the graphs are built from the tier-off pipeline's feedback).
fn engine(tier: Tier) -> Engine {
    #[cfg(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    super::TestSwitches {
        on: false,
        eager: false,
        exit_every: 0,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(tier);
    engine.set_tier_threshold(0);
    engine
}

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => panic!("{name}: {message}"),
    }
}

/// The global function `name`'s first-stage chunk.
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

fn dump(graph: &Graph, chunk: &Chunk) -> String {
    let mut out = String::new();
    for (pc, op) in chunk.jit_ops().iter().enumerate() {
        let _ = writeln!(out, "  {pc:>3}  {op:?}");
    }
    for (index, block) in graph.blocks.iter().enumerate() {
        let _ = writeln!(out, "b{index} (pc {})", block.first_pc);
        for &ix in block.phis.iter().chain(&block.body) {
            let node = graph.node(ix);
            let inputs: Vec<u32> = node.inputs.iter().map(|input| input.0).collect();
            let _ = writeln!(
                out,
                "  n{} = {:?} {inputs:?} {:?} @{} {:?}",
                ix.0, node.op, node.rep, node.pc, node.reason
            );
        }
    }
    out
}

/// The verified graph of the global function `name`, and its listing for messages.
fn graph_of(engine: &mut Engine, name: &str) -> (Graph, String) {
    let chunk = chunk_of(engine, name);
    let snapshot = Snapshot::take(&chunk);
    let built = build(&chunk, &snapshot)
        .unwrap_or_else(|refusal| panic!("{name}: {refusal:?}\n{:#?}", chunk.jit_ops()));
    if let Err(error) = super::verify::verify(&built.graph, &chunk, &built.cfg) {
        panic!("{name}: {error}\n{}", dump(&built.graph, &chunk));
    }
    let text = dump(&built.graph, &chunk);
    (built.graph, text)
}

fn count(graph: &Graph, which: impl Fn(&NodeOp) -> bool) -> usize {
    graph.nodes.iter().filter(|node| which(&node.op)).count()
}

/// The generic operation: a generic node (P3 stage 3 step 3a; an equality or a typeof test the
/// tier does not lower is one) or a canonical-frame effect.
fn is_effect(op: &NodeOp) -> bool {
    matches!(op, NodeOp::Generic { .. } | NodeOp::Effect { .. })
}

/// The engine after `source` ran in the VM (whose runs record), the JIT tier then set.
fn warmed(source: &str) -> Engine {
    let mut engine = engine(Tier::Bytecode);
    eval(&mut engine, source);
    engine.set_tier(Tier::Jit);
    engine
}

#[test]
fn a_loose_equality_with_a_nullish_constant_tests_the_word() {
    // `x == null` and `undefined != x` (the compiler puts the literal on the right): undefined,
    // null and the [[IsHTMLDDA]] object, whatever the other word, with no check and no effect.
    let mut engine = warmed(
        "function nl(x) { return x == null; }
         function nu(x) { return undefined != x; }
         function nf(x) { return (x | 0) == null; }
         function nb(a, b) { return (a < b) != undefined; }
         for (var i = 0; i < 20; i++) { var v = i % 3 ? {} : i % 2 ? null : 'a';
           nl(v); nu(v); nf(i); nb(i, 3); }",
    );
    for (name, nots) in [("nl", 0), ("nu", 1)] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(
            count(&graph, |op| *op == NodeOp::NullOrLikeUndefined),
            1,
            "{name}: {text}"
        );
        assert_eq!(
            count(&graph, |op| *op == NodeOp::BoolNot),
            nots,
            "{name}: {text}"
        );
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert!(
            graph.nodes.iter().all(|node| node.reason.is_none()),
            "{name}: no check: {text}"
        );
    }
    // A proven Number or Boolean is never nullish: the constant false, no test.
    for name in ["nf", "nb"] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(
            count(&graph, |op| *op == NodeOp::NullOrLikeUndefined),
            0,
            "{name}: {text}"
        );
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| node.op == NodeOp::ConstBool(name == "nb")),
            "{name}: the folded result: {text}"
        );
    }
}

#[test]
fn typeof_tests_are_their_own_nodes() {
    let tests = [
        ("number", TypeofTest::Number),
        ("boolean", TypeofTest::Boolean),
        ("string", TypeofTest::String),
        ("symbol", TypeofTest::Symbol),
        ("bigint", TypeofTest::BigInt),
        ("undefined", TypeofTest::Undefined),
        ("object", TypeofTest::Object),
        ("function", TypeofTest::Function),
    ];
    let mut source = String::new();
    for (k, (literal, _)) in tests.iter().enumerate() {
        let _ = write!(
            source,
            "function t{k}(x) {{ return typeof x === '{literal}'; }}
             function n{k}(x) {{ return typeof x !== '{literal}'; }}"
        );
    }
    source.push_str(
        "function never(x) { return typeof x === 'nothing'; }
         function known(x) { return typeof (x | 0) === 'number'; }
         function cs(o) { var k = o.k; return typeof o === 'string'; }
         function co(o) { var k = o.k; return typeof o === 'object'; }
         var vals = [1, 'a', {}, null, undefined, true, function () {}], q = {k: 1};
         for (var i = 0; i < 21; i++) { var v = vals[i % vals.length]; never(v); known(i);
           cs(q); co(q);",
    );
    for k in 0..tests.len() {
        let _ = write!(source, "t{k}(v); n{k}(v);");
    }
    source.push('}');
    let mut engine = warmed(&source);
    for (k, &(literal, test)) in tests.iter().enumerate() {
        for (name, nots) in [(format!("t{k}"), 0), (format!("n{k}"), 1)] {
            let (graph, text) = graph_of(&mut engine, &name);
            assert_eq!(
                count(&graph, |op| *op == NodeOp::TypeofIs(test)),
                1,
                "{literal}: {text}"
            );
            assert_eq!(
                count(&graph, |op| matches!(op, NodeOp::TypeofIs(_))),
                1,
                "{literal}: {text}"
            );
            assert_eq!(
                count(&graph, |op| *op == NodeOp::BoolNot),
                nots,
                "{literal}: {text}"
            );
            assert_eq!(count(&graph, is_effect), 0, "{literal}: {text}");
        }
    }
    // A literal no typeof result is, a proven Number, a proven object's primitive test: folded.
    for (name, value) in [("never", false), ("known", true), ("cs", false)] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(
            count(&graph, |op| matches!(op, NodeOp::TypeofIs(_))),
            0,
            "{name}: {text}"
        );
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert!(
            graph
                .nodes
                .iter()
                .any(|node| node.op == NodeOp::ConstBool(value)),
            "{name}: {text}"
        );
    }
    // A proven object's typeof is "object", "function" or "undefined": the test stays.
    let (graph, text) = graph_of(&mut engine, "co");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::TypeofIs(TypeofTest::Object)),
        1,
        "{text}"
    );
}

#[test]
fn strict_equality_with_a_proven_object_or_boolean_is_identity() {
    // `o.k` checks o's shape, which proves it an object: under `===` it equals only itself,
    // whatever x is; a comparison's Boolean equals only its own word.
    let mut engine = warmed(
        "function so(o, x) { var k = o.k; return o === x; }
         function sb(a, b, x) { return (a < b) !== x; }
         function lo(o, x) { var k = o.k; return o == x; }
         var p = {k: 1};
         for (var i = 0; i < 20; i++) { so(p, i % 2 ? p : 'a'); sb(i, 3, true); lo(p, 1); }",
    );
    for (name, nots) in [("so", 0), ("sb", 1)] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(
            count(&graph, |op| *op == NodeOp::TaggedSame),
            1,
            "{name}: {text}"
        );
        assert_eq!(
            count(&graph, |op| *op == NodeOp::BoolNot),
            nots,
            "{name}: {text}"
        );
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
    }
    // Under `==` an object against a word nothing types may coerce: the generic operation.
    let (graph, text) = graph_of(&mut engine, "lo");
    assert_eq!(count(&graph, is_effect), 1, "{text}");
}

#[test]
fn strict_equality_with_a_number_compares_the_word_as_a_double() {
    // 0 and -0 are equal to more words than their own: F64Cmp of the other's word, unchecked,
    // which every word but a Number's makes unordered; and so a proven Number.
    let mut engine = warmed(
        "function sz(x) { return x === 0; }
         function snz(x) { return x !== -0; }
         function sk(a, x) { return (a * 2) === x; }
         var vals = [0, -0, 'a', null, 1.5, 0 / 0];
         for (var i = 0; i < 20; i++) { var v = vals[i % vals.length]; sz(v); snz(v); sk(i, v); }",
    );
    for (name, cond) in [("sz", Cond::Eq), ("snz", Cond::Ne), ("sk", Cond::Eq)] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(
            count(&graph, |op| *op == NodeOp::WordAsF64),
            1,
            "{name}: {text}"
        );
        assert_eq!(
            count(&graph, |op| *op == NodeOp::F64Cmp(cond)),
            1,
            "{name}: {text}"
        );
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert!(
            graph
                .nodes
                .iter()
                .all(|node| node.op != NodeOp::CheckNumber || name == "sk"),
            "{name}: the word is not checked: {text}"
        );
    }
}

#[test]
fn equalities_nothing_types_stay_the_generic_operation() {
    // A loose compare with a String or Number constant may coerce, and two words nothing types
    // may be Strings: without a record (step L1's B) each is the generic operation.
    let mut engine = warmed(
        "function ls(x) { return x == 'a'; }
         function ln(x) { return x != 0; }
         function uu(a, b) { return a === b; }
         function ss(x) { return x === 'a'; }
         for (var i = 0; i < 20; i++) { ls(i); ln(i); uu(i, i); ss('b'); }",
    );
    for name in ["ls", "ln", "uu", "ss"] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(count(&graph, is_effect), 1, "{name}: {text}");
    }
}

/// The operand table of the end-to-end tests: undefined, null, the Booleans, 0, -0, NaN, 1,
/// 1.5, "", "a", "ab" and an equal String made in another cell, a long String and an equal one
/// made apart, a Symbol, two objects, a function, an Array, a BigInt, an object with valueOf,
/// the [[IsHTMLDDA]] object and two proxies (one callable).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const TABLE: &str = "const dda = $262.IsHTMLDDA; const o = {k: 1}, p = {k: 2};
     const s1 = 'ab', s2 = ['a', 'b'].join(''), l1 = 'x'.repeat(100), l2 = 'x'.repeat(99) + 'x';
     const vals = [undefined, null, true, false, 0, -0, NaN, 1, 1.5, '', 'a', s1, s2, l1, l2,
       Symbol('s'), o, p, function f() {}, [1], 1n, {valueOf() { return 1; }}, dda,
       new Proxy({}, {}), new Proxy(function () {}, {})];
     const n = vals.length;";

/// Results with the tier on, eager and forcing exits equal the VM's for `program`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn results_equal_the_vms(program: &str) {
    use super::TestSwitches;
    fn run(tier: Tier, switches: Option<TestSwitches>, source: &str) -> String {
        if let Some(switches) = switches {
            switches.set();
        }
        let mut engine = Engine::new();
        engine.set_tier(tier);
        engine.set_tier_threshold(0);
        match engine.eval(source, false).expect("the program parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => format!("throw {name}: {message}"),
        }
    }
    let expected = run(
        Tier::Bytecode,
        Some(TestSwitches {
            on: false,
            ..Default::default()
        }),
        program,
    );
    assert!(!expected.starts_with("throw"), "{expected}");
    let mut compiles = 0;
    for (label, exit_every, eager) in [("on", 0, false), ("eager", 0, true), ("forced", 7, true)] {
        super::TEST_COMPILED.with(|compiled| compiled.borrow_mut().clear());
        let switches = TestSwitches {
            on: true,
            eager,
            exit_every,
            ..Default::default()
        };
        assert_eq!(
            run(Tier::Jit, Some(switches), program),
            expected,
            "{label}: {program}"
        );
        compiles += super::TEST_COMPILED.with(|compiled| compiled.borrow().len());
    }
    assert!(compiles > 0, "the tier compiled the program's functions");
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn equality_results_over_the_operand_table_equal_the_vms() {
    // Every pair of the table at each operation, the nullish constants on either side, the
    // strict Number constants, and a proven object and Boolean under `===`.
    results_equal_the_vms(&format!(
        "{TABLE}
         function eq(a, b) {{ return a == b; }} function ne(a, b) {{ return a != b; }}
         function se(a, b) {{ return a === b; }} function sn(a, b) {{ return a !== b; }}
         function nl(a) {{ return a == null; }} function nu(a) {{ return a != undefined; }}
         function un(a) {{ return undefined == a; }} function lnn(a) {{ return null != a; }}
         function sz(a) {{ return a === 0; }} function snz(a) {{ return a !== -0; }}
         function snan(a) {{ return a === NaN; }} function s15(a) {{ return 1.5 !== a; }}
         function so(q, a) {{ var k = q.k; return q === a; }}
         function sbo(x, y, a) {{ return (x < y) === a; }}
         function lz(a) {{ return a == 0; }} function lnz(a) {{ return 1.5 != a; }}
         function tob(q) {{ var k = q.k; return typeof q === 'object'; }}
         function tfn(q) {{ var k = q.k; return typeof q !== 'function'; }}
         let h = 0, t = 0;
         for (let k = 0; k < 5000; k++) {{
           const a = vals[k % n], b = vals[Math.floor(k / n) % n];
           const bits = [eq(a, b), ne(a, b), se(a, b), sn(a, b), nl(a), nu(a), un(b), lnn(b),
             sz(a), snz(a), snan(a), s15(a), so(o, a), sbo(k & 1, 1, a), lz(a), lnz(b),
             tob(k & 1 ? o : p), tfn(o)];
           for (const bit of bits) {{ h = (h * 2 + (bit ? 1 : 0)) % 1000003; t += bit ? 1 : 0; }}
         }}
         h + ':' + t"
    ));
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn typeof_results_over_every_kind_equal_the_vms() {
    // Each test and its negation over every kind of value, the [[IsHTMLDDA]] object and the
    // proxies (the cold calls) among them, and a literal no typeof result is.
    let mut tests = String::new();
    let mut calls = String::new();
    for (k, literal) in [
        "undefined",
        "boolean",
        "number",
        "bigint",
        "string",
        "symbol",
        "object",
        "function",
        "nothing",
    ]
    .iter()
    .enumerate()
    {
        let _ = write!(
            tests,
            "function t{k}(a) {{ return typeof a === '{literal}'; }}
             function n{k}(a) {{ return '{literal}' != typeof a; }}"
        );
        let _ = write!(calls, "t{k}(a), n{k}(a), ");
    }
    results_equal_the_vms(&format!(
        "{TABLE} {tests}
         let h = 0, t = 0;
         for (let k = 0; k < 5000; k++) {{
           const a = vals[k % n];
           const bits = [{calls}];
           for (const bit of bits) {{ h = (h * 2 + (bit ? 1 : 0)) % 1000003; t += bit ? 1 : 0; }}
         }}
         h + ':' + t"
    ));
}
