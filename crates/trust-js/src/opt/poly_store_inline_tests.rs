//! Graph-next piece 2 (gnext-impl design note 3, "PIECE 2" and its addition u1 to u5): a named
//! store whose site saw several receiver shapes, some replacing an own property and some
//! creating it, built as an arm per way under TRUST_JS_OPT_INLINE=graph, in an inlined callee's
//! frame and in the compiled frame: its arms' nodes, its results against the tier off under
//! every change of its proof, the loop prepass's reading of its writes (every arm's location, a
//! transitioning arm's creation, the store lowered), the verifier's, its cards, and its exits'
//! teacher.

use super::*;

/// Callees whose store sites see two shapes each: `setl` replaces an own `left` (`old`'s) and
/// creates it on a fresh Node (whose prototype holds `left`: Splay's tree nodes), `setx`
/// replaces `x` at two slots (A's first, B's second), `tag` creates `tag` on two shapes.
const POLY: &str = "
function Node(k) { this.key = k; }
Node.prototype.left = null;
function A(k) { this.x = k; this.y = 0; }
function B(k) { this.y = 0; this.x = k; }
function setl(n, v) { n.left = v; }
function setx(o, v) { o.x = v; }
function tag(o, v) { o.tag = v; }
function m1(a, v) { setl(a, v); return 1; }
function m2(a, v) { setx(a, v); return 1; }
function m3(a, v) { tag(a, v); return 1; }
function d1(a, v) { a.left = v; return 1; }
var old = new Node(0); old.left = old;
for (var k = 0; k < 30; k++) {
  m1((k & 1) ? new Node(k) : old, k);
  m2((k & 1) ? new A(k) : new B(k), k);
  m3((k & 1) ? {p: k} : {q: k}, k);
  d1((k & 1) ? new Node(k) : old, k);
}
";

/// Whether this process runs the test `name` of this module alone (a child of the test binary,
/// which runs the body), else runs it so and checks that it passed: a creation way's proof holds
/// the creation epoch, which other tests' engines move (create_tier_tests' `alone`), and a site
/// whose ways a moved epoch refilled until it turned generic builds no creation.
fn alone(name: &str) -> bool {
    const CHILD: &str = "TRUST_JS_TEST_POLY_STORE_CHILD";
    if std::env::var_os(CHILD).is_some() {
        return true;
    }
    let output = std::process::Command::new(std::env::current_exe().expect("the test binary"))
        .args([
            "--exact",
            &format!("opt::inline_tests::poly_store::{name}"),
            "--test-threads=1",
        ])
        .env(CHILD, "1")
        .output()
        .expect("the child runs");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "the child {}:\n{stdout}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    false
}

fn count(graph: &Graph, frame: FrameIx, is: impl Fn(&NodeOp) -> bool) -> usize {
    nodes(graph, |node| node.frame == frame && is(&node.op)).len()
}

#[test]
fn polymorphic_stores_build_an_arm_per_way_in_their_frames() {
    if !alone("polymorphic_stores_build_an_arm_per_way_in_their_frames") {
        return;
    }
    let mut engine = warmed(POLY);
    // (caller, the store's frame, its Data arms, its Creation arms)
    for (name, frame, data, created) in [
        ("m1", FrameIx(1), 1, 1),
        ("m2", FrameIx(1), 2, 0),
        ("m3", FrameIx(1), 0, 2),
        ("d1", FrameIx(0), 1, 1),
    ] {
        let built = Compile::new(&mut engine, name).build();
        let graph = &built.graph;
        if frame == FrameIx(1) {
            assert_eq!(
                outcomes(&built),
                vec![(0, InlineOutcome::Inlined)],
                "{name}: {}",
                dump(graph)
            );
        }
        let stores = count(graph, frame, |op| matches!(op, NodeOp::StoreField { .. }));
        let news = count(graph, frame, |op| matches!(op, NodeOp::StoreNew { .. }));
        let tests = count(graph, frame, |op| matches!(op, NodeOp::ShapeIs(..)));
        assert_eq!(
            (stores, news, tests),
            (data, created, data + created - 1),
            "{name}: {}",
            dump(graph)
        );
        // Every check of the store keeps the access's two receiver shapes (the exits' teacher).
        for node in nodes(graph, |node| node.frame == frame) {
            match graph.node(node).op {
                NodeOp::CheckShape { ways, .. }
                | NodeOp::CheckProto { ways, .. }
                | NodeOp::CheckNoProto { ways }
                | NodeOp::CheckCreatable { ways }
                | NodeOp::StoreNew { ways, .. } => assert_eq!(ways, 2, "{name}: {}", dump(graph)),
                NodeOp::Effect { .. } | NodeOp::Generic { .. } => {
                    panic!("{name}: the store stays generic\n{}", dump(graph))
                }
                _ => {}
            }
        }
    }
}

/// The polymorphic stores of `POLY` under changes of their proof: a fresh Node frozen, sealed or
/// made non-extensible before its creation, an object holding `left` frozen before its
/// replacement, `left` on Node.prototype turned into an accessor, then read-only, then a data
/// property again; B's objects frozen before their replacement; a creation arm whose object fills
/// its cell's inline slots first (W) and one in dictionary mode; and quiet loops whose facts an
/// arm ends (`loopx`: p.x read, q.x stored through `setx` with q sometimes p; `loopl`: p.left
/// read, q.left created or replaced through `setl` with q sometimes p).
const CHANGES: &str = "
function Node(k) { this.key = k; }
Node.prototype.left = null;
function A(k) { this.x = k; this.y = 0; }
function B(k) { this.y = 0; this.x = k; }
function W(k) { this.a = k; this.b = k; this.c = k; this.d = k; }
function setl(n, v) { n.left = v; }
function setx(o, v) { o.x = v; }
function tag(o, v) { o.tag = v; }
function h(x) { return typeof x === 'number' ? x % 1000003 : x === null ? 3 : x === undefined ? -1 : typeof x === 'object' ? 7 : 9; }
function sum(o) { let s = 0; for (const k in o) s = (s * 7 + h(o[k])) % 1000003; return s; }
function loopx(p, q, n) { var s = p.x; for (var i = 0; i < n; i++) { setx(q, i); s = (s + p.x) | 0; } return s; }
function loopl(p, q, n) { var s = 0; for (var i = 0; i < n; i++) { setl(q, i); s = (s + h(p.left) + p.key) | 0; } return s; }
let t = 0;
const old = new Node(0); old.left = old;
function step(k) {
  const n = new Node(k);
  const f = k > 300 ? k % 4 : 0;
  if (f === 1) Object.freeze(n); else if (f === 2) Object.seal(n); else if (f === 3) Object.preventExtensions(n);
  const o2 = new Node(k); o2.left = 1;
  if (k > 900 && k % 3 === 0) Object.freeze(o2);
  setl((k & 1) ? n : old, k);
  setl(o2, k + 2);
  t = (t * 31 + h(n.left) + h(o2.left) + sum(n) + sum(o2)) % 1000003;
  const a = new A(k), b = new B(k);
  if (k > 500 && k % 5 === 0) Object.freeze(b);
  setx((k & 1) ? a : b, k + 1);
  t = (t * 31 + sum(a) + sum(b)) % 1000003;
  const w = new W(k);
  const d = { u: 1, v: 2 }; if (k > 700) delete d.u;
  tag((k % 3 === 0) ? w : (k % 3 === 1) ? { p: k } : d, k);
  t = (t * 31 + sum(w) + sum(d)) % 1000003;
  t = (t * 31 + h(loopx(b, (k % 3) ? a : b, 5)) + h(loopl(n, (k % 2) ? n : new Node(k), 5))) % 1000003;
}
for (let k = 0; k < 1300; k++) {
  if (k === 600) Object.defineProperty(Node.prototype, 'left',
    { set(v) { this.ll = v * 3; }, get() { return this.ll; }, configurable: true });
  if (k === 800) Object.defineProperty(Node.prototype, 'left', { value: 9, writable: false, configurable: true });
  if (k === 1000) { delete Node.prototype.left; Node.prototype.left = null; }
  step(k);
}
t";

/// `source`'s completion with the tier off, then under `switches`.
fn against_the_tier_off(source: &str, switches: &[TestSwitches]) {
    TestSwitches::default().set();
    let mut off = jit_engine();
    let expected = eval(&mut off, source);
    assert!(!expected.starts_with("throw"), "{expected}");
    for &switch in switches {
        switch.set();
        let mut engine = jit_engine();
        assert_eq!(eval(&mut engine, source), expected, "{switch:?}");
    }
    TestSwitches::default().set();
}

/// On, eager, eager with an exit forced every 7th and 100th check, and an exit limit of 1, at
/// graph and at the splice (whose compiled frames' stores the rule takes too).
fn configurations() -> Vec<TestSwitches> {
    let mut all = Vec::new();
    for graph in [Some(true), Some(false)] {
        let base = TestSwitches {
            on: true,
            asynchronous: false,
            inline_graph: graph,
            ..Default::default()
        };
        all.push(base);
        for (exit_every, exit_limit) in [(0, 0), (7, 0), (100, 0), (0, 1)] {
            all.push(TestSwitches {
                eager: true,
                exit_every,
                exit_limit,
                ..base
            });
        }
    }
    all
}

#[test]
fn polymorphic_stores_have_the_tier_offs_results_under_every_change_of_their_proof() {
    // Alone: its prototypes' changes move the process-wide creation epoch, which turns other
    // tests' creation sites generic (`alone`).
    if !alone("polymorphic_stores_have_the_tier_offs_results_under_every_change_of_their_proof") {
        return;
    }
    against_the_tier_off(CHANGES, &configurations());
}

/// Addition u1: `setx`'s two arms store `x` at two locations; `f1`'s p is always a B and `f2`'s
/// an A (each loop's read of p.x monomorphic, carried through a quiet loop), q sometimes p.
const TWO_LOCATIONS: &str = "
function A(k) { this.x = k; this.y = 0; }
function B(k) { this.y = 0; this.x = k; }
function setx(o, v) { o.x = v; }
function f1(p, q, n) { var s = p.x; for (var i = 0; i < n; i++) { setx(q, i); s = (s + p.x) | 0; } return s; }
function f2(p, q, n) { var s = p.x; for (var i = 0; i < n; i++) { setx(q, i); s = (s + p.x) | 0; } return s; }
var r = 0;
for (var k = 0; k < 400; k++) {
  var a = new A(k), b = new B(k);
  r = (r + f1(b, (k & 1) ? a : b, 6) + f2(a, (k & 2) ? b : a, 6)) % 1000003;
}
r";

#[test]
fn a_quiet_loop_reads_a_field_again_after_a_polymorphic_store_at_any_arms_location() {
    // Each loop is quiet (its store lowered) and keeps no field fact at either arm's location:
    // p.x is read in the loop, and the results are the tier off's.
    if !alone("a_quiet_loop_reads_a_field_again_after_a_polymorphic_store_at_any_arms_location") {
        return;
    }
    let mut engine = warmed(TWO_LOCATIONS);
    for name in ["f1", "f2"] {
        let built = Compile::new(&mut engine, name).build();
        let graph = &built.graph;
        assert_eq!(graph.kept.len(), 1, "{name}: a quiet loop\n{}", dump(graph));
        let stores = nodes(graph, |node| matches!(node.op, NodeOp::StoreField { .. }));
        assert_eq!(stores.len(), 2, "{name}: two arms\n{}", dump(graph));
        for &store in &stores {
            let NodeOp::StoreField { location, .. } = graph.node(store).op else {
                unreachable!()
            };
            assert!(
                !graph.kept[0].fields.contains(&location),
                "{name}: kept {:?}\n{}",
                graph.kept[0],
                dump(graph)
            );
        }
    }
    against_the_tier_off(
        TWO_LOCATIONS,
        &[TestSwitches {
            on: true,
            eager: true,
            asynchronous: false,
            inline_graph: Some(true),
            ..Default::default()
        }],
    );
}

/// Addition u2 and u4: `setl`'s arms replace and create `left`, in `g`'s loop, which reads p.key.
const TRANSITION: &str = "
function Node(k) { this.key = k; }
Node.prototype.left = null;
function setl(n, v) { n.left = v; }
function g(p, q, n) { var s = p.key; for (var i = 0; i < n; i++) { setl(q, i); s = (s + p.key) | 0; } return s; }
var old = new Node(0); old.left = old;
for (var k = 0; k < 40; k++) g(old, (k & 1) ? new Node(k) : old, 5);
";

/// `g`'s graph after `TRANSITION`, its loop quiet: one kept header, its store's arms in the loop.
fn transition_graph() -> Graph {
    let mut engine = warmed(TRANSITION);
    let built = Compile::new(&mut engine, "g").build();
    let graph = built.graph;
    assert_eq!(graph.kept.len(), 1, "a quiet loop\n{}", dump(&graph));
    assert_eq!(
        nodes(&graph, |node| matches!(node.op, NodeOp::StoreNew { .. })).len(),
        1,
        "{}",
        dump(&graph)
    );
    graph
}

#[test]
fn a_quiet_loop_with_a_transitioning_arm_keeps_no_shape_fact() {
    // The loop prepass counts the creation arm (Writes::creates): the header keeps no shape fact,
    // and the verifier (Compile::build's) admits the graph.
    if !alone("a_quiet_loop_with_a_transitioning_arm_keeps_no_shape_fact") {
        return;
    }
    let graph = transition_graph();
    assert!(
        !graph.kept[0].shapes,
        "{:?}\n{}",
        graph.kept[0],
        dump(&graph)
    );
}

#[test]
fn the_verifier_refuses_a_creation_arm_in_a_loop_that_kept_shapes() {
    // Addition u4: the same graph, its header made to keep shapes, is refused at the creation
    // arm's node, which lies in a block of the arm, not on the loop's every path.
    if !alone("the_verifier_refuses_a_creation_arm_in_a_loop_that_kept_shapes") {
        return;
    }
    let mut broken = transition_graph();
    super::super::verify::verify_kept_heap(&broken).expect("as built");
    broken.kept[0].shapes = true;
    let error = super::super::verify::verify_kept_heap(&broken).expect_err("a creation arm");
    assert!(
        error.contains("StoreNew") && error.contains("a creation (a shape transition)"),
        "{error}"
    );
}

/// Addition u3: `setx`'s store (two Data arms, at A's and B's first two slots) is the loop's only
/// store, inlined in `fy` and the function's own in `gy`; p (a C, its `y` in a third slot no arm
/// writes) is read before and in the loop.
const DATA_LOOP: &str = "
function A(k) { this.x = k; this.y = 0; }
function B(k) { this.y = 0; this.x = k; }
function C(k) { this.u = 0; this.w = 0; this.y = k; }
function setx(o, v) { o.x = v; }
function fy(p, q, n) { var s = p.y; for (var i = 0; i < n; i++) { setx(q, i); s = (s + p.y) | 0; } return s; }
function gy(p, q, n) { var s = p.y; for (var i = 0; i < n; i++) { q.x = i; s = (s + p.y) | 0; } return s; }
var a = new A(1), b = new B(2), c = new C(3);
for (var k = 0; k < 40; k++) { fy(c, (k & 1) ? a : b, 5); gy(c, (k & 1) ? a : b, 5); }
";

#[test]
fn a_loop_whose_only_store_is_polymorphic_keeps_its_other_facts() {
    // The rules say the store lowers (Rules::lowers): the loop keeps p's shape and p.y, which it
    // then neither checks nor loads.
    for name in ["fy", "gy"] {
        let mut engine = warmed(DATA_LOOP);
        let built = Compile::new(&mut engine, name).build();
        let graph = &built.graph;
        assert_eq!(graph.kept.len(), 1, "{name}: a quiet loop\n{}", dump(graph));
        assert!(graph.kept[0].shapes, "{name}: {:?}", graph.kept[0]);
        assert_eq!(
            nodes(graph, |node| matches!(node.op, NodeOp::StoreField { .. })).len(),
            2,
            "{name}: two arms\n{}",
            dump(graph)
        );
        // p.y: C's third slot (the global reads of `setx` are field loads too).
        assert_eq!(
            nodes(graph, |node| matches!(
                node.op,
                NodeOp::LoadField { slot: 2, .. }
            ))
            .len(),
            1,
            "{name}: p.y loaded before the loop alone\n{}",
            dump(graph)
        );
    }
}

/// Addition u5: `put` (`setx`'s two Data arms, `setl`'s Data and Creation arms, inlined) warmed
/// until the tier's code holds the arms (its first code, compiled before the sites ran, exits
/// and is recompiled), then, under collections at every allocation, old holders of each arm's
/// shapes given young values through each arm; only the arms' cards keep the values. Each kind
/// of holder lies apart from the others (no card of one holds another's), so that each arm's
/// own mark is what keeps its values.
const CARDS_WARM: &str = "
function A(k) { this.x = k; this.y = 0; }
function B(k) { this.y = 0; this.x = k; }
function Node(k) { this.key = k; }
Node.prototype.left = null;
function setx(o, v) { o.x = v; }
function setl(n, v) { n.left = v; }
function put(o, n, v) { setx(o, v); setl(n, v); }
var m0 = new Node(0); m0.left = 0;
for (var k = 0; k < 3000; k++) put((k & 1) ? new A(k) : new B(k), (k & 1) ? new Node(k) : m0, k);
var as = [], bs = [], ns = [], ms = [], pad = [];
function gap() { for (var k = 0; k < 200; k++) pad.push([k]); }
for (var i = 0; i < 12; i++) as.push(new A(i)); gap();
for (var i = 0; i < 12; i++) bs.push(new B(i)); gap();
for (var i = 0; i < 12; i++) { var m = new Node(i); m.left = 0; ms.push(m); } gap();
for (var i = 0; i < 12; i++) ns.push(new Node(i)); gap();
0";

const CARDS: &str = "
for (var i = 0; i < 12; i++) { put(as[i], ns[i], {v: 'a' + i}); put(bs[i], ms[i], {v: 'b' + i}); }
0";

const CARDS_READ: &str = "
var ok = 0;
for (var i = 0; i < 12; i++) {
  if (as[i].x.v === 'a' + i) ok++; if (bs[i].x.v === 'b' + i) ok++;
  if (ns[i].left.v === 'a' + i) ok++; if (ms[i].left.v === 'b' + i) ok++;
}
ok";

#[test]
fn every_arm_of_a_polymorphic_store_marks_its_card() {
    // The holders made old by a major collection, the stores of young values through the tier's
    // arms, then a minor collection with no conservative native roots (no stale word keeps a
    // value): only the holders' cards reach the values.
    if !alone("every_arm_of_a_polymorphic_store_marks_its_card") {
        return;
    }
    use crate::gc_core::CollectionKind;
    graph_switches(0).set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, CARDS_WARM), "0");
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Major);
    let exits = |_: ()| {
        super::super::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()))
    };
    exits(());
    assert_eq!(eval(&mut engine, CARDS), "0");
    assert!(exits(()).is_empty(), "the stores ran in the tier's arms");
    engine
        .interp
        .gc_collect_precise_for_test(CollectionKind::Minor);
    assert_eq!(eval(&mut engine, CARDS_READ), "48");
    TestSwitches::default().set();
}

/// The always-wrong case's teacher: after the warm-up, every fresh Node `setl` creates on is
/// non-extensible, whose shape its creation arm's receiver check passes and whose flags check
/// (CheckCreatable) fails at every execution, which the VM's store leaves the site's ways as they
/// were: each exit keeps the access's two receiver shapes, so that the recompile after the
/// discard declines the site (its ways did not grow) and each code holding the store is
/// discarded once, not again in each recompile.
const NON_EXTENSIBLE: &str = "
function Node(k) { this.key = k; }
Node.prototype.left = null;
function setl(n, v) { n.left = v; }
function d(n, v) { setl(n, v); return n.left === v ? 1 : 2; }
var old = new Node(0); old.left = old;
var s = 0;
for (var k = 0; k < 400; k++) s = (s + d((k & 1) ? new Node(k) : old, k)) % 1000003;
for (var k = 0; k < 1600; k++) s = (s + d((k & 1) ? Object.preventExtensions(new Node(k)) : old, k)) % 1000003;
s";

#[test]
fn a_creation_arms_failing_flags_check_declines_the_site_after_one_discard() {
    if !alone("a_creation_arms_failing_flags_check_declines_the_site_after_one_discard") {
        return;
    }
    TestSwitches::default().set();
    let mut off = jit_engine();
    let expected = eval(&mut off, NON_EXTENSIBLE);
    TestSwitches {
        on: true,
        eager: true,
        asynchronous: false,
        inline_graph: Some(true),
        ..Default::default()
    }
    .set();
    let mut engine = jit_engine();
    let taken = |_: ()| {
        super::super::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()))
    };
    taken(());
    let got = eval(&mut engine, NON_EXTENSIBLE);
    let exits = taken(());
    TestSwitches::default().set();
    assert_eq!(got, expected);
    let wrong = exits
        .iter()
        .filter(|exit| exit.1 == ExitReason::WrongShape)
        .count();
    // One discard per code holding the store: `d`'s, which inlines `setl`, and `setl`'s own (the
    // template code's calls reach it before `d` is compiled); 800 executions fail the check.
    assert!(
        wrong <= 2 * super::super::EXIT_LIMIT as usize,
        "WrongShape exits {wrong}, more than one discard per code"
    );
}

/// A creation arm's chain: M's objects have Node's own properties but a prototype whose `left`
/// is a setter; they reach `setl`'s site after its compile, and a Node's prototype gains a
/// level (Object.setPrototypeOf) that holds a setter.
const CHAINS: &str = "
function Node(k) { this.key = k; }
Node.prototype.left = null;
function M(k) { this.key = k; }
Object.defineProperty(M.prototype, 'left', { set(v) { this.mm = v * 7; }, get() { return this.mm; }, configurable: true });
function setl(n, v) { n.left = v; }
function d(n, v) { setl(n, v); return n.left === v ? 1 : 2; }
var old = new Node(0); old.left = old;
var t = 0;
for (var k = 0; k < 3000; k++) {
  if (k === 2500) Object.setPrototypeOf(Node.prototype, { set left(v) { this.nn = v * 5; } });
  var n = (k > 1500 && k % 3 === 0) ? new M(k) : (k & 1) ? new Node(k) : old;
  t = (t * 31 + d(n, k) + (n.mm | 0) + (n.nn | 0)) % 1000003;
}
t";

#[test]
fn a_creation_arm_checks_its_receivers_chain() {
    if !alone("a_creation_arm_checks_its_receivers_chain") {
        return;
    }
    against_the_tier_off(CHAINS, &configurations());
}

/// The arms' join: after `o.x = v` through A's and B's Data arms (x in each one's other slot),
/// the receiver's facts are those both arms hold, so that `o.y`'s load takes its own arms.
const JOIN: &str = "
function A(k) { this.x = k; this.y = 100; }
function B(k) { this.y = 200; this.x = k; }
function f(o, v) { o.x = v; return o.x + o.y; }
var t = 0;
for (var k = 0; k < 3000; k++) t = (t + f((k & 1) ? new A(k) : new B(k), k)) % 1000003;
t";

#[test]
fn the_arms_join_with_the_facts_they_all_hold() {
    against_the_tier_off(JOIN, &configurations());
}

// ---- Graph-next piece 2, its second part: compound stores on Data arms ----------------------

/// `acc` appends to `n` (`o.n += v`, AppendProp) and `bump` updates it (`o.n++`, UpdateProp) at
/// two-shape sites (P's `n` in its first slot, Q's in its second); `acc2` appends where R's
/// prototype holds `n` (its store creates on a fresh R, replaces on one that has its own);
/// `loopn` holds `acc` in a quiet loop that reads p.n, q sometimes p.
const COMPOUND: &str = "
function P(k) { this.n = k; this.m = 0; }
function Q(k) { this.m = 0; this.n = k; }
function R(k) { this.k = k; }
R.prototype.n = 5;
function acc(o, v) { o.n += v; }
function acc2(o, v) { o.n += v; }
function bump(o) { o.n++; }
function loopn(p, q, n) { var s = p.n; for (var i = 0; i < n; i++) { acc(q, 1); s = (s + p.n) | 0; } return s; }
";

#[test]
fn compound_stores_take_their_data_arms_and_never_create() {
    let warm = format!(
        "{COMPOUND}
         for (var k = 0; k < 30; k++) {{
           var p = new P(k), q = new Q(k), r = new R(k), r2 = new R(k); r2.n = 1;
           acc((k & 1) ? p : q, k); bump((k & 1) ? q : p); acc2((k & 1) ? r : r2, k);
           loopn(p, (k & 2) ? p : q, 4);
         }}"
    );
    let mut engine = warmed(&warm);
    // (function, its Data arms' stores, whether the store stays the generic operation)
    for (name, stores, generic) in [
        ("acc", 2, false),
        ("bump", 2, false),
        ("acc2", 0, true),
        ("loopn", 2, false),
    ] {
        let built = Compile::new(&mut engine, name).build();
        let graph = &built.graph;
        let all = |is: fn(&NodeOp) -> bool| nodes(graph, |node| is(&node.op)).len();
        assert_eq!(
            all(|op| matches!(op, NodeOp::StoreField { .. })),
            stores,
            "{name}\n{}",
            dump(graph)
        );
        assert_eq!(
            all(|op| matches!(op, NodeOp::StoreNew { .. })),
            0,
            "{name}: a compound store never creates\n{}",
            dump(graph)
        );
        assert_eq!(
            all(|op| matches!(op, NodeOp::Effect { .. } | NodeOp::Generic { .. })) > 0,
            generic,
            "{name}\n{}",
            dump(graph)
        );
        if name == "loopn" {
            // The loop is quiet (the store lowered: Rules::lowers) and keeps no field fact at
            // either arm's location (Rules::own_field_writes); the verifier admitted it.
            assert_eq!(graph.kept.len(), 1, "{}", dump(graph));
            for store in nodes(graph, |node| matches!(node.op, NodeOp::StoreField { .. })) {
                let NodeOp::StoreField { location, .. } = graph.node(store).op else {
                    unreachable!()
                };
                assert!(
                    !graph.kept[0].fields.contains(&location),
                    "{:?}",
                    graph.kept
                );
            }
        }
    }
}

#[test]
fn compound_stores_have_the_tier_offs_results_when_valueof_moves_or_deletes_the_property() {
    if !alone(
        "compound_stores_have_the_tier_offs_results_when_valueof_moves_or_deletes_the_property",
    ) {
        return;
    }
    // A value whose valueOf adds a property to the receiver (its other shape) or deletes `n`,
    // which the numeric lowering never calls (its NotNumber exit, then the generic operation);
    // R's creations through acc2; the quiet loop with q sometimes p.
    let source = format!(
        "{COMPOUND}
         function h(x) {{ return typeof x === 'number' ? (x === x ? x % 1000003 : 7) : x === undefined ? -1 : 9; }}
         let t = 0;
         for (let k = 0; k < 1500; k++) {{
           const p = new P(k), q = new Q(k), o = (k & 1) ? p : q;
           acc(o, k % 7); bump((k & 2) ? p : q);
           let v = 1;
           if (k > 700 && k % 50 === 0) v = {{ valueOf() {{ o.extra = 1; return 3; }} }};
           if (k > 900 && k % 70 === 0) v = {{ valueOf() {{ delete o.n; return 4; }} }};
           acc(o, v); bump(o);
           const r = new R(k); if (k % 3) r.n = 2; acc2(r, k);
           t = (t * 31 + h(p.n) + 3 * h(q.n) + (o.extra | 0) + h(r.n) + h(loopn(p, (k % 3) ? p : q, 4))) % 1000003;
         }}
         t"
    );
    against_the_tier_off(&source, &configurations());
}
