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

/// Store `codes` one after another in the last-seen byte of the global function `name`'s one
/// equality, each merged into its record, as template code's stores and the tier's merges at its
/// checkpoints make it.
fn see(engine: &mut Engine, name: &str, codes: &[u8]) {
    let chunk = chunk_of(engine, name);
    let pc = equality_pc(&chunk);
    for &code in codes {
        super::profile::ChunkProfile::of(&chunk).see_equality(chunk.jit_ops().len(), pc, code);
        super::profile::merge_element_hits(&chunk);
    }
}

/// Pair codes as template code stores them (feedback.rs EQ_SEEN_*, eq_seen_tag).
const NUMBERS: u8 = crate::feedback::EQ_SEEN_NUMBERS;
const TWO_TAGS: u8 = crate::feedback::EQ_SEEN_TWO_TAGS;
const STRINGS: u8 = crate::feedback::eq_seen_tag(crate::value::PACK_STR);
const OBJECTS: u8 = crate::feedback::eq_seen_tag(crate::value::PACK_OBJ);
const BOOLEANS: u8 = crate::feedback::eq_seen_tag(crate::value::PACK_BOOL);

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
    // Two undefineds, two Symbols: one tag, but no speculation's.
    see(
        &mut engine,
        "eb",
        &[crate::feedback::eq_seen_tag(crate::value::PACK_UNDEFINED)],
    );
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

/// The equality's last-seen byte and record of the global function `name`.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
fn seen(engine: &mut Engine, name: &str) -> (u8, u32) {
    let chunk = chunk_of(engine, name);
    let pc = equality_pc(&chunk);
    (
        super::profile::ChunkProfile::get(&chunk)
            .map_or(super::profile::UNSEEN, |profile| profile.equality_seen(pc)),
        chunk.feedback_vector().record_word(pc),
    )
}

#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "macos", target_os = "linux", target_os = "windows")
))]
#[test]
fn template_code_records_the_pair_each_form_met() {
    use crate::feedback::{EQ_NUMBERS, EQ_OBJECTS, EQ_OTHER, EQ_STRINGS};
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
        ("lp", "lp({}, null, 8)", TWO_TAGS),
        ("lv", "lv({v: 1}, 2, 8)", NUMBERS),
        ("lv", "lv({v: 'a'}, 'b', 8)", STRINGS),
        ("lv", "lv({v: {}}, {}, 8)", OBJECTS),
        ("li", "li(8)", NUMBERS),
    ] {
        template_only(&mut engine, name, call);
        assert_eq!(seen(&mut engine, name).0, code, "{call}: the byte");
        // The merge ORs the pair's bit into the record and empties the byte.
        super::profile::merge_element_hits(&chunk_of(&mut engine, name));
        let (byte, record) = seen(&mut engine, name);
        assert_eq!(byte, super::profile::UNSEEN, "{call}");
        let bit = match code {
            NUMBERS => EQ_NUMBERS,
            STRINGS => EQ_STRINGS,
            OBJECTS => EQ_OBJECTS,
            BOOLEANS => crate::feedback::EQ_BOOLEANS,
            _ => EQ_OTHER,
        };
        assert_ne!(record & bit, 0, "{call}: {record:#x}");
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
        assert_eq!(
            seen(&mut engine, name),
            (super::profile::UNSEEN, 0),
            "{call}: nothing recorded"
        );
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
