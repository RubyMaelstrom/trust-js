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

// ---- B: the equality record and its speculations ----------------------------------------------

/// The pc of `chunk`'s one equality.
fn equality_pc(chunk: &Chunk) -> usize {
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| crate::bytecode::feedback_read::is_equality(&chunk.jit_ops()[pc]))
        .collect();
    assert_eq!(pcs.len(), 1, "one equality: {:?}", chunk.jit_ops());
    pcs[0]
}

/// Mark the pair classes `classes` one after another at the global function `name`'s one
/// equality, each merged into its record, as template code's stores and the tier's merges at its
/// checkpoints make it.
fn see(engine: &mut Engine, name: &str, classes: &[u8]) {
    let chunk = chunk_of(engine, name);
    let pc = equality_pc(&chunk);
    for &class in classes {
        super::profile::ChunkProfile::of(&chunk).see_equality(&chunk, pc, class);
        super::profile::merge_element_hits(&chunk);
    }
}

/// Pair classes as template code marks them (feedback.rs EQ_CLASS_*, eq_class_of_tag).
const NUMBERS: u8 = crate::feedback::EQ_CLASS_NUMBERS;
const TWO_TAGS: u8 = crate::feedback::EQ_CLASS_TWO_KINDS;
#[cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    allow(dead_code)
)]
const OBJECT_OR_NULLISH: u8 = crate::feedback::EQ_CLASS_OBJECT_OR_NULLISH;
const STRINGS: u8 = crate::feedback::eq_class_of_tag(crate::value::PACK_STR);
const OBJECTS: u8 = crate::feedback::eq_class_of_tag(crate::value::PACK_OBJ);
const BOOLEANS: u8 = crate::feedback::eq_class_of_tag(crate::value::PACK_BOOL);
#[cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    allow(dead_code)
)]
const UNDEFINEDS: u8 = crate::feedback::eq_class_of_tag(crate::value::PACK_UNDEFINED);
#[cfg_attr(
    not(all(
        target_arch = "aarch64",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    )),
    allow(dead_code)
)]
const NULLS: u8 = crate::feedback::eq_class_of_tag(crate::value::PACK_NULL);
const SYMBOLS: u8 = crate::feedback::eq_class_of_tag(crate::value::PACK_SYM);

/// The checks of `op` in `graph` and their reasons.
fn checks(graph: &Graph, op: &NodeOp) -> Vec<Option<ExitReason>> {
    graph
        .nodes
        .iter()
        .filter(|node| node.op == *op)
        .map(|node| node.reason)
        .collect()
}

const SPECULATED: &str = "function en(a, b) { return a == b; }
     function es(a, b) { return a !== b; }
     function ec(a) { return a === 'lit'; }
     function eo(a, b) { return a == b; }
     function eb(a, b) { return a === b; }
     for (var i = 0; i < 20; i++) { en(i, 2); es('a', 'b'); ec('x'); eo({}, {}); eb(true, false); }";

#[test]
fn an_equality_whose_record_saw_one_pair_speculates_on_it() {
    let mut engine = warmed(SPECULATED);
    for (name, code) in [
        ("en", NUMBERS),
        ("es", STRINGS),
        ("ec", STRINGS),
        ("eo", OBJECTS),
        ("eb", BOOLEANS),
    ] {
        see(&mut engine, name, &[code]);
    }
    let (graph, text) = graph_of(&mut engine, "en");
    assert_eq!(
        checks(&graph, &NodeOp::CheckNumber),
        [Some(ExitReason::NotNumber); 2],
        "{text}"
    );
    assert_eq!(
        count(&graph, |op| *op == NodeOp::F64Cmp(Cond::Eq)),
        1,
        "{text}"
    );
    for (name, strings, nots) in [("es", 2, 1), ("ec", 1, 0)] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(
            checks(&graph, &NodeOp::CheckString),
            vec![Some(ExitReason::NotString); strings],
            "{name}: {text}"
        );
        assert_eq!(count(&graph, |op| *op == NodeOp::StringEqual), 1, "{text}");
        assert_eq!(count(&graph, |op| *op == NodeOp::BoolNot), nots, "{text}");
    }
    for (name, check, reason) in [
        ("eo", NodeOp::CheckObject, ExitReason::NotObject),
        ("eb", NodeOp::CheckBoolean, ExitReason::NotBoolean),
    ] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(checks(&graph, &check), [Some(reason); 2], "{name}: {text}");
        assert_eq!(count(&graph, |op| *op == NodeOp::TaggedSame), 1, "{text}");
    }
    for name in ["en", "es", "ec", "eo", "eb"] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
    }
}

#[test]
fn an_equality_whose_record_saw_several_pairs_or_another_stays_the_generic_operation() {
    let mut engine = warmed(SPECULATED);
    see(&mut engine, "en", &[NUMBERS, STRINGS]);
    see(&mut engine, "es", &[TWO_TAGS]);
    see(&mut engine, "eo", &[OBJECTS, TWO_TAGS]);
    // Two Symbols: one tag, but no speculation's.
    see(&mut engine, "eb", &[SYMBOLS]);
    for name in ["en", "es", "eo", "eb", "ec"] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(count(&graph, is_effect), 1, "{name}: {text}");
    }
}

#[test]
fn a_flagged_speculation_leaves_the_generic_operation() {
    // A discarded code's check exited at the site: its recompile declines the speculation, the
    // record notwithstanding.
    let mut engine = warmed(SPECULATED);
    for (name, code, reason) in [
        ("en", NUMBERS, ExitReason::NotNumber),
        ("es", STRINGS, ExitReason::NotString),
        ("eo", OBJECTS, ExitReason::NotObject),
        ("eb", BOOLEANS, ExitReason::NotBoolean),
    ] {
        see(&mut engine, name, &[code]);
        let chunk = chunk_of(&mut engine, name);
        let pc = equality_pc(&chunk);
        super::profile::ChunkProfile::of(&chunk).add_site(pc as u32, reason as u8, false, 0, false);
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(count(&graph, is_effect), 1, "{name}: {text}");
        assert!(
            graph
                .declines
                .iter()
                .any(|decline| decline.pc == pc as u32 && decline.reason == reason),
            "{name}: the decline is noted: {text}"
        );
    }
    // Another reason's flag declines nothing of it.
    see(&mut engine, "ec", &[STRINGS]);
    let chunk = chunk_of(&mut engine, "ec");
    let pc = equality_pc(&chunk);
    super::profile::ChunkProfile::of(&chunk).add_site(
        pc as u32,
        ExitReason::NotNumber as u8,
        false,
        0,
        false,
    );
    let (graph, text) = graph_of(&mut engine, "ec");
    assert_eq!(count(&graph, is_effect), 0, "{text}");
}

/// Template code's forms of equality, each in a function of its own: a compact site of two
/// operands, compact sites of a Number and a String constant, a loop's local pair feeding a
/// branch, a loop's value equality on a property, and a numeric loop's integer compare.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const FORMS: &str = "function c2(a, b) { return a === b; }
     function cn(a) { return a == 0; }
     function cs(a) { return a === 'a'; }
     function lp(a, b, n) { var c = 0; for (var i = 0; i < n; i++) { if (a == b) c++; } return c; }
     function lv(o, b, n) { var c = 0; for (var i = 0; i < n; i++) { c += o.v === b ? 1 : 0; }
       return c; }
     function li(n) { var c = 0; for (var i = 0; i < n; i++) { if ((i & 3) == 1) c++; } return c; }";

/// An engine at the JIT tier compiling at the first call, under `switches` (forced exits off).
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn jit_engine(on: bool) -> Engine {
    super::TestSwitches {
        on,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

/// Run `call` as template code only: its first call compiles the function, whose checkpoint is
/// then never due, so that the tier compiles nothing.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn template_only(engine: &mut Engine, name: &str, call: &str) {
    eval(engine, call);
    let chunk = chunk_of(engine, name);
    assert!(chunk.jit.get().flatten().is_some(), "{name}: native");
    chunk.set_inline_retry_at(u32::MAX);
    eval(engine, call);
}

/// The pair classes met since the last merge (bit k class k) and the record of the global
/// function `name`'s equality.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn seen(engine: &mut Engine, name: &str) -> (u32, u32) {
    let chunk = chunk_of(engine, name);
    let pc = equality_pc(&chunk);
    (
        super::profile::ChunkProfile::get(&chunk).map_or(0, |profile| profile.equality_seen(pc)),
        chunk.feedback_vector().record_word(pc),
    )
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn template_code_records_the_pair_each_form_met() {
    super::pin_forced_exits_off_for_test();
    let mut engine = jit_engine(true);
    eval(&mut engine, FORMS);
    for (name, call, code) in [
        ("c2", "c2(1, 2.5)", NUMBERS),
        ("c2", "c2('ab', ['a', 'b'].join(''))", STRINGS),
        ("c2", "c2({}, {})", OBJECTS),
        ("c2", "c2(true, false)", BOOLEANS),
        ("c2", "c2(1, 'a')", TWO_TAGS),
        ("cn", "cn(3)", NUMBERS),
        ("cn", "cn('0')", TWO_TAGS),
        ("cs", "cs('b')", STRINGS),
        ("cs", "cs(1)", TWO_TAGS),
        ("lp", "lp(1, 2, 8)", NUMBERS),
        ("lp", "lp('a', 'b', 8)", STRINGS),
        ("lp", "lp({}, null, 8)", OBJECT_OR_NULLISH),
        ("lp", "lp(1, null, 8)", TWO_TAGS),
        ("lv", "lv({v: 1}, 2, 8)", NUMBERS),
        ("lv", "lv({v: 'a'}, 'b', 8)", STRINGS),
        ("lv", "lv({v: {}}, {}, 8)", OBJECTS),
        ("li", "li(8)", NUMBERS),
    ] {
        template_only(&mut engine, name, call);
        assert_eq!(seen(&mut engine, name).0, 1 << code, "{call}: the class");
        // The merge ORs the class's bit into the record and empties the group.
        super::profile::merge_element_hits(&chunk_of(&mut engine, name));
        let (classes, record) = seen(&mut engine, name);
        assert_eq!(classes, 0, "{call}");
        let bit = crate::feedback::eq_class_bit(code);
        assert_ne!(record & bit, 0, "{call}: {record:#x}");
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn template_code_marks_each_pairs_class() {
    // The tag path's classes (P3 stage 3 step L1b): a pair of canonical NaNs is two Numbers;
    // undefined against undefined, null against null, and an object or nullish against another of
    // them are the nullish pairs; a nullish value against another kind, and a Number against a
    // String, two kinds; two Symbols their tag's, the other pairs'.
    super::pin_forced_exits_off_for_test();
    let mut engine = jit_engine(true);
    eval(&mut engine, FORMS);
    template_only(&mut engine, "c2", "c2(1, 2)");
    super::profile::merge_element_hits(&chunk_of(&mut engine, "c2"));
    for (call, class) in [
        ("c2(NaN, NaN)", NUMBERS),
        ("c2(undefined, undefined)", UNDEFINEDS),
        ("c2(null, null)", NULLS),
        ("c2(null, undefined)", OBJECT_OR_NULLISH),
        ("c2({}, undefined)", OBJECT_OR_NULLISH),
        ("c2(null, [])", OBJECT_OR_NULLISH),
        ("c2(1, null)", TWO_TAGS),
        ("c2(true, null)", TWO_TAGS),
        ("c2('a', 1)", TWO_TAGS),
        ("c2(NaN, 1)", TWO_TAGS),
        ("var sy = Symbol(); c2(sy, sy)", SYMBOLS),
    ] {
        eval(&mut engine, call);
        assert_eq!(seen(&mut engine, "c2").0, 1 << class, "{call}");
        super::profile::merge_element_hits(&chunk_of(&mut engine, "c2"));
    }
    use crate::feedback::{EQ_NULLISH, EQ_NUMBERS, EQ_OTHER};
    assert_eq!(
        seen(&mut engine, "c2").1,
        EQ_NUMBERS | EQ_NULLISH | EQ_OTHER
    );
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn the_tdz_marker_and_an_arrays_hole_reach_no_equality_operand() {
    // The class test of two tags takes the TDZ marker (0x7ffa) where CheckObjectOrNullish does not
    // (feedback.rs EQ_CLASS_OBJECT_OR_NULLISH): no path gives an equality the marker. A binding
    // read in its dead zone throws before the compare; an array's hole reads as undefined through
    // an index, a spread, a destructuring, a for-of and an argument spread: two undefineds' class,
    // never the marker's (2).
    super::pin_forced_exits_off_for_test();
    let mut engine = jit_engine(true);
    eval(&mut engine, "function he(x, y) { return x === y; }");
    template_only(&mut engine, "he", "he(1, 2)");
    super::profile::merge_element_hits(&chunk_of(&mut engine, "he"));
    assert_eq!(
        eval(
            &mut engine,
            "function early() { return he(late, late); }
             var thrown; try { early(); } catch (e) { thrown = e instanceof ReferenceError; }
             let late = 1;
             thrown"
        ),
        "true"
    );
    assert_eq!(
        seen(&mut engine, "he").0,
        0,
        "the dead zone's read threw first"
    );
    for call in [
        "var ha = [1, , 3]; he(ha[1], ha[1])",
        "var [hd] = [, 1]; he(hd, hd)",
        "var hs = [...[, 1]]; he(hs[0], hs[0])",
        "for (var hf of [, 2]) { he(hf, hf); break; }",
        "(function (a, b) { return he(a, a); })(...[, 1])",
        "var hm = [, 1].map(function (v, k) { return k; }); he(hm[0], hm[0])",
    ] {
        eval(&mut engine, call);
        assert_eq!(seen(&mut engine, "he").0, 1 << UNDEFINEDS, "{call}");
        super::profile::merge_element_hits(&chunk_of(&mut engine, "he"));
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn template_code_keeps_every_pair_met_between_two_merges() {
    // RayTrace's `shape != exclude`: an object against an object, then against null, then an
    // object again, all before one merge: both classes are kept and both merged (P3 stage 3 step
    // L1b), where a last-seen byte kept the last alone; the same for two Numbers and two Strings
    // at a loop's local pair.
    use crate::feedback::{EQ_NULLISH, EQ_NUMBERS, EQ_OBJECTS, EQ_STRINGS};
    super::pin_forced_exits_off_for_test();
    let mut engine = jit_engine(true);
    eval(&mut engine, FORMS);
    for (name, calls, first, classes, record) in [
        (
            "c2",
            &["c2({}, {})", "c2({}, null)", "c2({}, {})"][..],
            OBJECTS,
            1u32 << OBJECTS | 1 << OBJECT_OR_NULLISH,
            EQ_OBJECTS | EQ_NULLISH,
        ),
        (
            "lp",
            &["lp(1, 2, 8)", "lp('a', 'b', 8)", "lp(3, 4, 8)"][..],
            NUMBERS,
            1u32 << NUMBERS | 1 << STRINGS,
            EQ_NUMBERS | EQ_STRINGS,
        ),
    ] {
        template_only(&mut engine, name, calls[0]);
        super::profile::merge_element_hits(&chunk_of(&mut engine, name));
        assert_eq!(
            seen(&mut engine, name).1,
            crate::feedback::eq_class_bit(first)
        );
        for call in calls {
            eval(&mut engine, call);
        }
        assert_eq!(
            seen(&mut engine, name).0,
            classes,
            "{name}: the classes met"
        );
        super::profile::merge_element_hits(&chunk_of(&mut engine, name));
        assert_eq!(seen(&mut engine, name), (0, record), "{name}: merged");
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn with_the_tier_off_template_code_records_no_pair() {
    super::pin_forced_exits_off_for_test();
    let mut engine = jit_engine(false);
    eval(&mut engine, FORMS);
    for (name, call) in [
        ("c2", "c2(1, 2)"),
        ("cn", "cn(3)"),
        ("cs", "cs('b')"),
        ("lp", "lp(1, 2, 8)"),
        ("lv", "lv({v: 1}, 2, 8)"),
        ("li", "li(8)"),
    ] {
        template_only(&mut engine, name, call);
        assert_eq!(seen(&mut engine, name), (0, 0), "{call}: nothing recorded");
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn an_equality_with_a_nullish_constant_keeps_no_record() {
    // A loop's in-line templates record their equalities' pairs (P3 stage 3 step L1b), but not
    // where the right operand is an undefined or null constant on every path (step L1b, D): such
    // an equality has no group, and its record stays empty, an object against the constant as
    // before; `null === a` is compiled as `a === null`. Where the conditional's join reaches the
    // equality, the constant is one operand of two, and the site records.
    super::pin_forced_exits_off_for_test();
    let mut engine = jit_engine(true);
    let mut source = String::new();
    for (name, test) in [
        ("rn", "a == null"),
        ("ru", "undefined != a"),
        ("rs", "a === null"),
        ("rw", "null === a"),
        ("rv", "a !== void 0"),
        ("rj", "a == (b ? a : null)"),
    ] {
        source.push_str(&format!(
            "function {name}(a, b, n) {{ var c = 0;
               for (var i = 0; i < n; i++) {{ if ({test}) c++; }} return c; }}"
        ));
    }
    eval(&mut engine, &source);
    for name in ["rn", "ru", "rs", "rw", "rv", "rj"] {
        template_only(&mut engine, name, &format!("{name}({{}}, false, 8)"));
        let chunk = chunk_of(&mut engine, name);
        let pc = equality_pc(&chunk);
        let records = name == "rj";
        assert_eq!(
            super::profile::ChunkProfile::of(&chunk)
                .equality_bytes(&chunk, pc)
                .is_some(),
            records,
            "{name}: a group: {:?}",
            chunk.jit_ops()
        );
        let classes = if records { 1 << OBJECT_OR_NULLISH } else { 0 };
        assert_eq!(seen(&mut engine, name), (classes, 0), "{name}: met");
        super::profile::merge_element_hits(&chunk);
        let record = if records {
            crate::feedback::EQ_NULLISH
        } else {
            0
        };
        assert_eq!(seen(&mut engine, name), (0, record), "{name}: merged");
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn a_strict_equality_with_a_boolean_or_number_constant_keeps_no_record() {
    // P3 stage 3 step L1c: under `===` and `!==` a Boolean or a Number constant decides the
    // equality too (identity, the word's double compare), so a loop's in-line templates record
    // nothing there; `0 === a` is compiled as `a === 0`, `-1.5` is a constant. Under `==` a Number
    // and under `===` a String constant keep their records, which the builder reads.
    super::pin_forced_exits_off_for_test();
    let mut engine = jit_engine(true);
    let shapes = [
        ("kt", "a === true", false),
        ("kf", "false !== a", false),
        ("kz", "a !== 0", false),
        ("kw", "0 === a", false),
        ("kn", "a === -1.5", false),
        ("kl", "a == 0", true),
        ("ks", "a === 's'", true),
    ];
    let mut source = String::new();
    for (name, test, _) in shapes {
        source.push_str(&format!(
            "function {name}(a, n) {{ var c = 0;
               for (var i = 0; i < n; i++) {{ if ({test}) c++; }} return c; }}"
        ));
    }
    eval(&mut engine, &source);
    for (name, _, records) in shapes {
        template_only(&mut engine, name, &format!("{name}('x', 8)"));
        let chunk = chunk_of(&mut engine, name);
        let pc = equality_pc(&chunk);
        assert_eq!(
            super::profile::ChunkProfile::of(&chunk)
                .equality_bytes(&chunk, pc)
                .is_some(),
            records,
            "{name}: a group: {:?}",
            chunk.jit_ops()
        );
        let met = seen(&mut engine, name).0;
        assert_eq!(met != 0, records, "{name}: met {met:#x}");
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn speculated_equality_results_equal_the_vms() {
    // Sites that meet one pair for long, then others late (their checks exit), a String pair
    // of equal lengths and contents in other cells, a long String, interned and made Strings,
    // and the table's pairs at sites the records leave generic.
    results_equal_the_vms(&format!(
        "{TABLE}
         function nn(a, b) {{ return a == b; }} function ss(a, b) {{ return a === b; }}
         function sl(a) {{ return a !== 'a'; }} function oo(a, b) {{ return a == b; }}
         function bb(a, b) {{ return a !== b; }} function ln(a) {{ return a == 1.5; }}
         const strs = ['a', 'ab', s2, l1, l2, 'b', 'a'.concat(''), String.fromCharCode(97)];
         let h = 0, t = 0;
         for (let k = 0; k < 6000; k++) {{
           const late = k > 5000, x = vals[k % n], y = vals[(k >> 3) % n];
           const bits = [
             nn(late ? x : k & 7, late ? y : (k >> 2) & 7),
             ss(late ? x : strs[k & 7], late ? y : strs[(k >> 3) & 7]),
             sl(late ? x : strs[k & 7]),
             oo(late ? x : (k & 1 ? o : p), late ? y : (k & 2 ? o : p)),
             bb(late ? x : (k & 1) === 0, late ? y : (k & 2) === 0),
             ln(late ? x : (k & 3) / 2)];
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
fn a_site_that_alternates_its_pairs_settles_within_the_limits() {
    // `a === b` meets Numbers, then Strings, then both, in turns: its speculation exits, its
    // flag declines it, and the recompile keeps the generic operation, within two discards.
    super::pin_forced_exits_off_for_test();
    let program = "function alt(a, b) { return a === b; }
         let t = 0;
         for (let k = 0; k < 40000; k++) {
           const phase = Math.floor(k / 4000) % 3;
           const a = phase === 1 ? 's' + (k & 3) : phase === 2 && k & 1 ? 'x' : k & 3;
           t += alt(a, phase === 0 ? 2 : 's1') ? 1 : 0;
         }
         t";
    let expected = {
        super::TestSwitches {
            on: false,
            ..Default::default()
        }
        .set();
        let mut engine = engine(Tier::Bytecode);
        eval(&mut engine, program)
    };
    super::TEST_DISCARDED.with(|discarded| discarded.borrow_mut().clear());
    let mut engine = jit_engine(true);
    assert_eq!(eval(&mut engine, program), expected);
    let discards = super::TEST_DISCARDED.with(|discarded| discarded.borrow().len());
    assert!(
        (1..=2).contains(&discards),
        "{discards} discards: the speculation exited, then settled"
    );
    let chunk = chunk_of(&mut engine, "alt");
    let first = super::profile::first_stage(&chunk);
    let flagged = super::profile::ChunkProfile::get(first).is_some_and(|profile| {
        let mut flagged = false;
        profile.sites_at(equality_pc(&chunk) as u32, |reason, _, _| {
            flagged |=
                reason == ExitReason::NotNumber as u8 || reason == ExitReason::NotString as u8
        });
        flagged
    });
    assert!(flagged, "the site's flag");
}

// ---- L1b: the object-or-nullish pair ------------------------------------------------------------

const NULLISH_SITES: &str = "function on(a, b) { return a == b; }
     function os(a, b) { return a !== b; }
     function oc(a) { return a == 'x'; }
     function ou(a, b) { return a === b; }
     function ox(a, b) { return a == b; }
     for (var i = 0; i < 20; i++) { on(i & 1 ? {} : null, {}); os(undefined, null); oc({});
       ou({}, i & 1 ? {} : undefined); ox(null, i & 1 ? {} : 1); }";

#[test]
fn a_record_of_objects_and_nullish_values_checks_each_an_object_or_nullish() {
    // An object against null and two objects (`==`): each operand checked an object or nullish
    // [NotObjectOrNullish], then TurboFan's loose compare; two undefineds alone (`!==`): the
    // checks, then identity, negated; `a == 'x'` with such a record: the String constant rules
    // the pair out, the mismatch exit; nullish pairs and another (null against a Number): the
    // generic operation.
    let mut engine = warmed(NULLISH_SITES);
    see(&mut engine, "on", &[OBJECTS, OBJECT_OR_NULLISH]);
    see(&mut engine, "os", &[UNDEFINEDS]);
    see(&mut engine, "oc", &[OBJECT_OR_NULLISH]);
    see(&mut engine, "ou", &[OBJECT_OR_NULLISH, OBJECTS]);
    see(&mut engine, "ox", &[OBJECT_OR_NULLISH, TWO_TAGS]);
    let (graph, text) = graph_of(&mut engine, "ox");
    assert_eq!(count(&graph, is_effect), 1, "{text}");
    for (name, compare) in [
        ("on", NodeOp::ObjectOrNullishEqual),
        ("os", NodeOp::TaggedSame),
        ("ou", NodeOp::TaggedSame),
    ] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert_eq!(
            checks(&graph, &NodeOp::CheckObjectOrNullish),
            vec![Some(ExitReason::NotObjectOrNullish); 2],
            "{name}: {text}"
        );
        assert_eq!(count(&graph, |op| *op == compare), 1, "{name}: {text}");
        assert_eq!(
            count(&graph, |op| *op == NodeOp::BoolNot),
            usize::from(name == "os"),
            "{name}: {text}"
        );
    }
    let (graph, text) = graph_of(&mut engine, "oc");
    assert_eq!(
        count(&graph, |op| *op
            == NodeOp::Exit(ExitReason::NotObjectOrNullish)),
        1,
        "{text}"
    );
    // NotObject's flag (an Objects speculation's exit) leaves the pair; its own flag declines it.
    let chunk = chunk_of(&mut engine, "on");
    let pc = equality_pc(&chunk);
    let profile = super::profile::ChunkProfile::of(&chunk);
    profile.add_site(pc as u32, ExitReason::NotObject as u8, false, 0, false);
    let (graph, text) = graph_of(&mut engine, "on");
    assert_eq!(
        count(&graph, |op| *op == NodeOp::ObjectOrNullishEqual),
        1,
        "{text}"
    );
    profile.add_site(
        pc as u32,
        ExitReason::NotObjectOrNullish as u8,
        false,
        0,
        false,
    );
    let (graph, text) = graph_of(&mut engine, "on");
    assert_eq!(count(&graph, is_effect), 1, "{text}");
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn object_or_nullish_equality_results_equal_the_vms() {
    // Sites that meet objects (an [[IsHTMLDDA]] one among them), undefined and null for long, then
    // the whole table late (their checks exit), under `==`, `!=`, `===` and `!==`.
    results_equal_the_vms(&format!(
        "{TABLE}
         function le(a, b) {{ return a == b; }} function ln(a, b) {{ return a != b; }}
         function se(a, b) {{ return a === b; }} function sn(a, b) {{ return a !== b; }}
         const kin = [o, p, null, undefined, dda, o, [1], null];
         let h = 0, t = 0;
         for (let k = 0; k < 6000; k++) {{
           const late = k > 5000, x = vals[k % n], y = vals[(k >> 3) % n];
           const a = late ? x : kin[k & 7], b = late ? y : kin[(k >> 3) & 7];
           const bits = [le(a, b), ln(b, a), se(a, b), sn(a, b), le(a, a), se(b, null)];
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
fn nullish_constant_equality_results_equal_the_vms() {
    // The table at the equalities of P3 stage 3 step L1b's D: an undefined or null constant
    // against a conditional's phi and an assignment's value (no record, the rules' claim from the
    // constant alone), and the constant one arm of a conditional the equality joins (a record);
    // in loops, whose in-line templates record, and out of them.
    results_equal_the_vms(&format!(
        "{TABLE}
         function pl(a, b, c) {{ return (c ? a : b) == null; }}
         function ps(a, b, c) {{ return (c ? a : b) !== undefined; }}
         function al(a) {{ var t; return (t = a) != null; }}
         function as(a) {{ var t; return (t = a) === null; }}
         function jn(a, c) {{ return a == (c ? a : null); }}
         function jv(a, c) {{ return a === (c ? null : void 0); }}
         function lw(a, b, m) {{ var c = 0;
           for (var i = 0; i < m; i++) {{ if ((i & 1 ? a : b) != null) c++;
             if (a === (i & 2 ? null : b)) c += 2; }} return c; }}
         let h = 0, t = 0;
         for (let k = 0; k < 5000; k++) {{
           const a = vals[k % n], b = vals[Math.floor(k / n) % n], c = k & 1;
           const bits = [pl(a, b, c), ps(a, b, c), al(a), as(b), jn(a, c), jv(b, c),
             lw(a, b, 4)];
           for (const bit of bits) {{ h = (h * 3 + (bit | 0)) % 1000003; t += bit | 0; }}
         }}
         h + ':' + t"
    ));
}

/// Run `program` with the tier on (forced exits off), its result the VM's; the exits taken, and
/// the discards made, while it ran.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn exits_and_discards(program: &str) -> (Engine, Vec<ExitReason>, usize) {
    let expected = {
        super::TestSwitches {
            on: false,
            ..Default::default()
        }
        .set();
        let mut engine = engine(Tier::Bytecode);
        eval(&mut engine, program)
    };
    super::TEST_DISCARDED.with(|discarded| discarded.borrow_mut().clear());
    super::TEST_COMPILED.with(|compiled| compiled.borrow_mut().clear());
    super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
    let mut engine = jit_engine(true);
    assert_eq!(eval(&mut engine, program), expected, "{program}");
    assert!(
        super::TEST_COMPILED.with(|compiled| !compiled.borrow().is_empty()),
        "the tier compiled: {program}"
    );
    let exits = super::exit::TEST_EXITS
        .with(|exits| exits.borrow().iter().map(|&(_, reason)| reason).collect());
    let discards = super::TEST_DISCARDED.with(|discarded| discarded.borrow().len());
    (engine, exits, discards)
}

/// The equality speculations' exit reasons.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
const EQUALITY_REASONS: [ExitReason; 5] = [
    ExitReason::NotNumber,
    ExitReason::NotString,
    ExitReason::NotObject,
    ExitReason::NotBoolean,
    ExitReason::NotObjectOrNullish,
];

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn sites_whose_pairs_alternate_take_no_exit_once_their_record_has_seen_both() {
    // From the first call: RayTrace's `shape != exclude` (an object, null for one call in four,
    // each call's loop meeting one pair), Strings and Numbers in turns, two Strings and a String
    // against an object in turns. The record keeps both pairs (P3 stage 3 step L1b), so the first
    // compile speculates on objects or nullish values, or keeps the generic operation: no exit.
    super::pin_forced_exits_off_for_test();
    for (name, program, compare) in [
        (
            "rt",
            "function rt(shapes, ex) { var h = 0;
               for (var i = 0; i < shapes.length; i++) { if (shapes[i] != ex) h++; } return h; }
             const S = [{}, {}, {}, {}]; let t = 0;
             for (let k = 0; k < 20000; k++) t += rt(S, k & 3 ? S[k & 3] : null);
             t",
            Some(NodeOp::ObjectOrNullishEqual),
        ),
        (
            "sn",
            "function sn(a, b) { return a === b; } let t = 0;
             for (let k = 0; k < 20000; k++) t += (k & 1 ? sn(k & 3, 2) : sn('s' + (k & 3), 's1')) ? 1 : 0;
             t",
            None,
        ),
        (
            "st",
            "function st(a, b) { return a === b; } const o = {}; let t = 0;
             for (let k = 0; k < 20000; k++) t += (k & 1 ? st('a' + (k & 1), 'a1') : st('a', o)) ? 1 : 0;
             t",
            None,
        ),
    ] {
        let (mut engine, exits, _) = exits_and_discards(program);
        let at_equality: Vec<_> = exits
            .iter()
            .filter(|reason| EQUALITY_REASONS.contains(reason))
            .collect();
        assert!(at_equality.is_empty(), "{name}: {at_equality:?}");
        let (graph, text) = graph_of(&mut engine, name);
        match &compare {
            Some(op) => assert_eq!(count(&graph, |node| node == op), 1, "{name}: {text}"),
            None => assert_eq!(count(&graph, is_effect), 1, "{name}: {text}"),
        }
    }
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn a_site_that_meets_a_second_pair_late_converges_in_one_discard() {
    // Objects alone for long, then objects and null in turns: the Objects speculation exits
    // [NotObject], one discard, and the recompile speculates on objects or nullish values with no
    // exit after; Numbers for long, then Strings: one discard, then the generic operation.
    super::pin_forced_exits_off_for_test();
    for (name, program, compare) in [
        (
            "cv",
            "function cv(shapes, ex) { var h = 0;
               for (var i = 0; i < shapes.length; i++) { if (shapes[i] != ex) h++; } return h; }
             const S = [{}, {}, {}, {}]; let t = 0;
             for (let k = 0; k < 40000; k++) t += cv(S, k < 20000 || k & 3 ? S[k & 3] : null);
             t",
            Some(NodeOp::ObjectOrNullishEqual),
        ),
        (
            "nv",
            "function nv(a, b) { return a === b; } let t = 0;
             for (let k = 0; k < 40000; k++) t += (k < 20000 ? nv(k & 3, 2) : nv('s' + (k & 3), 's1')) ? 1 : 0;
             t",
            None,
        ),
    ] {
        let (mut engine, exits, discards) = exits_and_discards(program);
        assert_eq!(discards, 1, "{name}: {exits:?}");
        let (graph, text) = graph_of(&mut engine, name);
        match &compare {
            Some(op) => assert_eq!(count(&graph, |node| node == op), 1, "{name}: {text}"),
            None => assert_eq!(count(&graph, is_effect), 1, "{name}: {text}"),
        }
    }
}

// ---- C: the prepass's claims ------------------------------------------------------------------

/// The blocks of `graph`'s one natural loop: its header and those that reach its back edges.
fn loop_members(graph: &Graph, text: &str) -> Vec<bool> {
    let header = graph
        .blocks
        .iter()
        .position(|block| block.loop_header)
        .unwrap_or_else(|| panic!("a loop: {text}"));
    let mut members = vec![false; graph.blocks.len()];
    members[header] = true;
    let mut work: Vec<usize> = graph.blocks[header]
        .preds
        .iter()
        .map(|pred| pred.index())
        .filter(|&pred| pred >= header)
        .collect();
    while let Some(block) = work.pop() {
        if std::mem::replace(&mut members[block], true) {
            continue;
        }
        work.extend(graph.blocks[block].preds.iter().map(|pred| pred.index()));
    }
    members
}

#[test]
fn a_loop_whose_equalities_and_typeof_tests_lower_is_free_of_effects() {
    // `g()` before the loop leaves `a` in the frame's keeping; in a loop without effects it is
    // loaded once at the entry (stage 3 step 0), as in one whose generic nodes leave it alone
    // (step 3a); the lowered tests leave no generic node in the loop.
    let tests = [
        ("ln", "x == null", None),
        ("lu", "undefined != x", None),
        ("lt", "typeof x === 'number'", None),
        ("lz", "x === 0", None),
        ("ls", "x === y", Some(STRINGS)),
        ("lx", "x == y", Some(NUMBERS)),
    ];
    let mut source = String::new();
    for (name, test, _) in tests {
        source.push_str(&format!(
            "function {name}(g, a, x, y, n) {{ var s = 0; g();
               for (var i = 0; i < n; i++) {{ if ({test}) s = s + a; }} return s; }}"
        ));
    }
    source.push_str(
        "function lo(g, a, x, y, n) { var s = 0; g();
           for (var i = 0; i < n; i++) { if (x == 'a') s = s + a; } return s; }
         function nothing() {}",
    );
    let mut drive = String::from("for (var r = 0; r < 20; r++) {");
    for (name, _, _) in tests {
        drive.push_str(&format!("{name}(nothing, 1.5, r, 'b', 4);"));
    }
    drive.push_str("lo(nothing, 1.5, r, 'b', 4); }");
    // The VM's runs record; the JIT tier's then fill the call caches the builder reads.
    let mut engine = warmed(&format!("{source} {drive}"));
    eval(&mut engine, &drive);
    for (name, test, code) in tests {
        if let Some(code) = code {
            see(&mut engine, name, &[code]);
        }
        let (graph, text) = graph_of(&mut engine, name);
        let members = loop_members(&graph, &text);
        let in_loop = |which: &dyn Fn(&NodeOp) -> bool| {
            graph
                .nodes
                .iter()
                .any(|node| members[node.block.index()] && which(&node.op))
        };
        assert!(
            !in_loop(&is_effect),
            "{test}: no effect in the loop: {text}"
        );
        assert!(
            !in_loop(&|op| matches!(op, NodeOp::FrameSlot(_))),
            "{test}: no reload in the loop: {text}"
        );
    }
    // A loose compare with a String constant may coerce: a generic node.
    let (graph, text) = graph_of(&mut engine, "lo");
    let members = loop_members(&graph, &text);
    assert!(
        graph
            .nodes
            .iter()
            .any(|node| members[node.block.index()] && is_effect(&node.op)),
        "{text}"
    );
}

#[test]
fn a_constants_equality_lowers_whatever_made_the_other_operand() {
    // The other operand a conditional's phi, made in other blocks, or an assignment's value, which
    // no single push made (P3 stage 3 step L1b, D): the rules claim the equality from the
    // constant's producer alone, so the loop is free of effects and keeps its heap facts (its read
    // takes the data address the read before it checked, stage 3 step 0c), and the builder lowers
    // it from the constant: NullOrLikeUndefined under `==`, identity under `===` with undefined
    // or null, the word's double compare with a Number. Such sites keep no record
    // (`feedback_read::records_equality`): the constant is the claim's only source.
    let tests = [
        ("pl", "(y ? x : y) == null", NodeOp::NullOrLikeUndefined),
        ("al", "(y = x) != null", NodeOp::NullOrLikeUndefined),
        ("ps", "(y ? x : y) === null", NodeOp::TaggedSame),
        ("as", "(y = x) !== undefined", NodeOp::TaggedSame),
        ("pn", "(y ? x : y) === 0", NodeOp::F64Cmp(Cond::Eq)),
    ];
    let mut source = String::from("var T = new Int32Array(8);");
    let mut drive = String::from("for (var r = 0; r < 20; r++) {");
    for (name, test, _) in &tests {
        source.push_str(&format!(
            "function {name}(t, x, y, n) {{ var s = t[0] | 0;
               for (var i = 0; i < n; i = (i + 1) | 0) {{
                 if ({test}) s = (s + (t[i] | 0)) | 0; }}
               return s | 0; }}"
        ));
        drive.push_str(&format!("{name}(T, r & 1 ? null : r, 'b', 8);"));
    }
    drive.push('}');
    let mut engine = warmed(&format!("{source} {drive}"));
    for (name, test, lowered) in tests {
        let (graph, text) = graph_of(&mut engine, name);
        let members = loop_members(&graph, &text);
        let in_loop = |which: &dyn Fn(&NodeOp) -> bool| {
            graph
                .nodes
                .iter()
                .any(|node| members[node.block.index()] && which(&node.op))
        };
        assert!(
            !in_loop(&is_effect),
            "{test}: no effect in the loop: {text}"
        );
        assert!(
            !in_loop(&|op| matches!(op, NodeOp::TypedData { .. })),
            "{test}: the data address loaded before the loop alone: {text}"
        );
        assert!(in_loop(&|op| *op == lowered), "{test}: {lowered:?}: {text}");
    }
}

#[test]
fn a_loop_whose_equality_lowers_keeps_its_heap_facts() {
    // A typed array's facts are heap facts, which a loop keeps through its header where nothing in
    // it may end them (stage 3 step 0c): with `x === 0` lowered, the loop's read takes the data
    // address the read before it checked; with `x == 'a'`, a generic node, which may run anything
    // (step 3a), the loop checks and loads it again.
    let mut engine = warmed(
        "function hk(t, x, n) { var s = t[0] | 0;
           for (var i = 0; i < n; i = (i + 1) | 0) { if (x === 0) s = (s + (t[i] | 0)) | 0; }
           return s | 0; }
         function hg(t, x, n) { var s = t[0] | 0;
           for (var i = 0; i < n; i = (i + 1) | 0) { if (x == 'a') s = (s + (t[i] | 0)) | 0; }
           return s | 0; }
         var T = new Int32Array(8);
         for (var r = 0; r < 20; r++) { hk(T, r & 1, 8); hg(T, 'a', 8); }",
    );
    for (name, generic) in [("hk", false), ("hg", true)] {
        let (graph, text) = graph_of(&mut engine, name);
        let members = loop_members(&graph, &text);
        let in_loop = |which: &dyn Fn(&NodeOp) -> bool| {
            graph
                .nodes
                .iter()
                .any(|node| members[node.block.index()] && which(&node.op))
        };
        assert_eq!(in_loop(&is_effect), generic, "{name}: {text}");
        assert_eq!(
            in_loop(&|op| matches!(op, NodeOp::TypedData { .. })),
            generic,
            "{name}: the data address loaded in the loop only past a generic node: {text}"
        );
    }
}

#[test]
fn a_claimed_speculation_its_operands_rule_out_exits() {
    // The record saw two Numbers at `x == 'a'` (a constant String) and two Strings at `(a | 0)
    // == b` (a proven int32): the rules claim them lowered, so the builder exits at each [its
    // speculation's reason] rather than make the effect a loop would not expect, and the exit's
    // flag then declines the claim.
    let mut engine = warmed(
        "function cn(x) { return x == 'a'; } function ci(a, b) { return (a | 0) == b; }
         for (var i = 0; i < 20; i++) { cn(i); ci(i, 'b'); }",
    );
    see(&mut engine, "cn", &[NUMBERS]);
    see(&mut engine, "ci", &[STRINGS]);
    for (name, reason) in [("cn", ExitReason::NotNumber), ("ci", ExitReason::NotString)] {
        let (graph, text) = graph_of(&mut engine, name);
        assert_eq!(count(&graph, is_effect), 0, "{name}: {text}");
        assert_eq!(
            count(&graph, |op| *op == NodeOp::Exit(reason)),
            1,
            "{name}: {text}"
        );
        assert!(
            graph
                .nodes
                .iter()
                .all(|node| !matches!(node.op, NodeOp::CheckString)),
            "{name}: {text}"
        );
    }
    let chunk = chunk_of(&mut engine, "cn");
    let pc = equality_pc(&chunk);
    super::profile::ChunkProfile::of(&chunk).add_site(
        pc as u32,
        ExitReason::NotNumber as u8,
        false,
        0,
        false,
    );
    let (graph, text) = graph_of(&mut engine, "cn");
    assert_eq!(count(&graph, is_effect), 1, "{text}");
}

#[test]
fn the_equality_record_leaves_the_settled_feedback_fingerprint_as_it_was() {
    // A pair met at an equality changes its record, which the merge at the fingerprint takes,
    // and not the fingerprint: the function tiers up when its other feedback settles, as V8's
    // compare feedback delays no tier-up.
    let mut engine = warmed(SPECULATED);
    let chunk = chunk_of(&mut engine, "en");
    let before = super::snapshot::fingerprint(&chunk);
    let pc = equality_pc(&chunk);
    super::profile::ChunkProfile::of(&chunk).see_equality(&chunk, pc, NUMBERS);
    assert_eq!(super::snapshot::fingerprint(&chunk), before);
    assert_eq!(
        chunk.feedback_vector().record_word(pc),
        crate::feedback::EQ_NUMBERS,
        "merged at the fingerprint"
    );
    super::profile::ChunkProfile::of(&chunk).see_equality(&chunk, pc, STRINGS);
    assert_eq!(super::snapshot::fingerprint(&chunk), before);
}

// ---- L1c: the equalities that keep no record ---------------------------------------------------

/// The lowering the builder made of the equality at `pc`: its nodes there, but constants, phis
/// and the boxing of the result, as debug text, sorted.
fn lowering(graph: &Graph, pc: usize) -> Vec<String> {
    let mut ops: Vec<String> = graph
        .nodes
        .iter()
        .filter(|node| node.pc as usize == pc && node.frame == FrameIx::COMPILED)
        .filter(|node| {
            !matches!(
                node.op,
                NodeOp::ConstTagged(_)
                    | NodeOp::ConstF64(_)
                    | NodeOp::ConstI32(_)
                    | NodeOp::ConstBool(_)
                    | NodeOp::Phi
            )
        })
        .map(|node| format!("{:?} {:?}", node.op, node.reason))
        .collect();
    ops.sort();
    ops
}

#[test]
fn the_equalities_that_keep_no_record_are_those_their_constant_decides() {
    // The invariant of P3 stage 3 step L1c (`feedback_read::constant_decides_equality`): for each
    // shape, the equality built under no record and under each pair's is one lowering exactly
    // where `feedback_read::records_equality` keeps no record. A builder case that decides from a
    // constant before the record, or a no-record case the builder reads a record for, breaks it
    // alone. The other operand is a parameter of every kind (no fact proves it).
    use crate::feedback::{EQ_BOOLEANS, EQ_NULLISH, EQ_NUMBERS, EQ_OBJECTS, EQ_STRINGS};
    let shapes = [
        "x === undefined",
        "x !== null",
        "x === void 0",
        "x === true",
        "false !== x",
        "x === 0",
        "x !== -0",
        "x === 1.5",
        "5 === x",
        "x === -2",
        "x === 'a'",
        "x !== ''",
        "x == null",
        "undefined != x",
        "x == true",
        "x == 0",
        "x == 1.5",
        "x != 'a'",
        "x === y",
        "x == y",
        "x === NaN",
        "x == (y ? 1 : null)",
        "x === (y ? 0 : null)",
    ];
    let records = [
        0,
        EQ_NUMBERS,
        EQ_STRINGS,
        EQ_OBJECTS,
        EQ_BOOLEANS,
        EQ_NULLISH,
        EQ_NULLISH | EQ_OBJECTS,
    ];
    for (k, test) in shapes.iter().enumerate() {
        let name = format!("q{k}");
        let program = format!(
            "function {name}(x, y) {{ return {test}; }}
             var kinds = [1, 'a', null, {{}}, true, undefined, 1.5];
             for (var r = 0; r < 21; r++) {name}(kinds[r % 7], r & 1);"
        );
        let mut keeps = None;
        let lowerings: Vec<Vec<String>> = records
            .iter()
            .map(|&pairs| {
                let mut engine = warmed(&program);
                let chunk = chunk_of(&mut engine, &name);
                let pc = equality_pc(&chunk);
                let targets = crate::bytecode::jump_targets(chunk.jit_ops());
                keeps = Some(crate::bytecode::feedback_read::records_equality(
                    &chunk, &targets, pc,
                ));
                if pairs != 0 {
                    chunk.feedback_vector().record(pc, pairs);
                }
                let (graph, _) = graph_of(&mut engine, &name);
                lowering(&graph, pc)
            })
            .collect();
        let decided = lowerings.iter().all(|lowering| *lowering == lowerings[0]);
        assert_eq!(
            keeps,
            Some(!decided),
            "{test}: a record kept where the builder decides without it, or none where it reads \
             one: {lowerings:?}"
        );
    }
}
