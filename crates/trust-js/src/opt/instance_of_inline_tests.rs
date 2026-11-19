//! Graph line piece 3 (gp3-impl design note section 1): `instanceof` lowered as
//! OrdinaryHasInstance on the right-hand side its site's cell recorded (CheckShape, CheckCtor,
//! the `prototype` slot, HasInPrototypeChain), in an inlined callee's frame and in the compiled
//! frame; its results against the tier off under every change of its proof; the cell's generic
//! mark (the teacher); the realm's Function.prototype read through the context.

use super::*;

/// `m1` reaches `isP`'s InstanceOf inlined, `d1` holds its own; `any` sees P and Q, whose shapes
/// differ (Q's own `extra`), so that its cell is marked generic; `bnd` sees P and a bound P, which
/// no cell can hold (the cell's guards reject it), so that its cell is marked generic too.
const IO: &str = "
function P(a) { this.a = a; }
function Q(a) { this.a = a; }
Q.extra = 1;
var PB = P.bind(null);
class C extends P { constructor(a) { super(a); } }
function isP(x) { return x instanceof P; }
function m1(x) { return isP(x) ? 1 : 0; }
function d1(x) { return x instanceof P ? 1 : 0; }
function any(x, f) { return x instanceof f ? 1 : 0; }
function bnd(x, f) { return x instanceof f ? 1 : 0; }
var xs = [new P(1), { a: 2 }, new C(3), 7, null, 'p'];
var s = 0;
for (var k = 0; k < 60; k++)
  s += m1(xs[k % 6]) + d1(xs[k % 6]) + any(xs[k % 6], (k & 1) ? P : Q) + bnd(xs[k % 6], (k & 1) ? P : PB);
s";

fn count(graph: &Graph, frame: FrameIx, is: impl Fn(&NodeOp) -> bool) -> usize {
    nodes(graph, |node| node.frame == frame && is(&node.op)).len()
}

#[test]
fn instanceof_builds_ordinary_has_instance_in_its_frames() {
    let mut engine = warmed(IO);
    // (function, the InstanceOf's frame, whether its cell's form is lowered)
    for (name, frame, lowered) in [
        ("m1", FrameIx(1), true),
        ("d1", FrameIx(0), true),
        ("any", FrameIx(0), false),
    ] {
        let built = Compile::new(&mut engine, name).build();
        let graph = &built.graph;
        let ctors = count(graph, frame, |op| matches!(op, NodeOp::CheckCtor));
        let walks = count(graph, frame, |op| matches!(op, NodeOp::HasInPrototypeChain));
        let generic = count(graph, frame, |op| {
            matches!(op, NodeOp::Effect { .. } | NodeOp::Generic { .. })
        });
        if lowered {
            assert_eq!(
                (ctors, walks, generic),
                (1, 1, 0),
                "{name}: {}",
                dump(graph)
            );
        } else {
            assert_eq!((ctors, walks), (0, 0), "{name}: {}", dump(graph));
            assert!(generic >= 1, "{name}: {}", dump(graph));
        }
    }
}

/// `instanceof` at two sites under changes of its proof: `prototype` replaced; a primitive, a
/// null, a subclass instance and a plain object on the left; P's [[Prototype]] moved to an
/// object holding a Symbol.hasInstance and back; Symbol.hasInstance defined later on P (its
/// shape) and deleted; an attempt to redefine Function.prototype's (immutable: it throws); then
/// at `via`'s site a bound function, a class with a static Symbol.hasInstance and a primitive
/// right-hand side (TypeError); and last a Proxy on the left whose getPrototypeOf trap counts
/// its calls.
const CHANGES: &str = "
function P(a) { this.a = a; }
class C extends P { constructor(a) { super(a); } }
class S { static [Symbol.hasInstance](v) { return v === 3; } }
var calls = 0;
var trap = { getPrototypeOf(t) { calls++; return P.prototype; } };
var trap2 = { getPrototypeOf(t) { calls += 10; return Q.prototype; } };
var h = { [Symbol.hasInstance](v) { return typeof v === 'number'; } };
Object.setPrototypeOf(h, Function.prototype);
function Q(a) { this.a = a; }
var fake = Object.create(Function.prototype);
Object.defineProperty(fake, 'length', { value: 1, configurable: true });
Object.defineProperty(fake, 'name', { value: 'Q', configurable: true });
Object.defineProperty(fake, 'prototype', { value: Q.prototype, writable: true });
function isP(x) { return x instanceof P; }
function isQ(x) { return x instanceof Q; }
function viaQ(x, f) { try { return x instanceof f ? 1 : 0; } catch (e) { return e instanceof TypeError ? 5 : 6; } }
function via(x, f) { try { return x instanceof f ? 1 : 0; } catch (e) { return e instanceof TypeError ? 5 : 6; } }
function step(k, f) {
  var xs = [new P(k), { a: k }, new C(k), k, null, undefined, 's', k > 1100 ? new Proxy({}, trap) : 0];
  var r = 0;
  var q = k > 1100 && k % 2 ? new Proxy({}, trap2) : k % 3 ? new Q(k) : { a: k };
  r = r * 3 + (isQ(q) ? 1 : 0) + viaQ(q, k >= 1060 && k < 1070 ? fake : Q);
  for (var i = 0; i < xs.length; i++) r = (r * 3 + (isP(xs[i]) ? 1 : 0) + via(xs[i], f)) % 1000003;
  return r;
}
var t = 0;
for (var k = 0; k < 1300; k++) {
  if (k === 200) P.prototype = { b: 1 };
  if (k === 300) P.prototype = C.prototype;
  if (k === 400) Object.setPrototypeOf(P, h);
  if (k === 450) Object.setPrototypeOf(P, Function.prototype);
  if (k === 500) { try { Object.defineProperty(Function.prototype, Symbol.hasInstance, { value: function () { return true; } }); } catch (e) { t += 1000; } }
  if (k === 600) Object.defineProperty(P, Symbol.hasInstance, { value: function (v) { return v === null; }, configurable: true });
  if (k === 700) delete P[Symbol.hasInstance];
  var f = k < 800 ? P : k < 900 ? P.bind(null) : k < 1000 ? S : k < 1050 ? 4 : P;
  t = (t * 7 + step(k, f)) % 1000003;
}
t * 10 + calls";

/// On, eager, eager with an exit forced every 7th and 100th check, and an exit limit of 1, at
/// graph and at the splice.
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
fn instanceof_has_the_tier_offs_results_under_every_change_of_its_proof() {
    TestSwitches::default().set();
    let mut off = jit_engine();
    let expected = eval(&mut off, CHANGES);
    assert!(!expected.starts_with("throw"), "{expected}");
    super::super::build::TEST_INSTANCE_OFS.with(|count| count.set(0));
    for switch in configurations() {
        switch.set();
        let mut engine = jit_engine();
        assert_eq!(eval(&mut engine, CHANGES), expected, "{switch:?}");
    }
    TestSwitches::default().set();
    assert!(
        super::super::build::TEST_INSTANCE_OFS.with(|count| count.get()) > 0,
        "nothing lowered"
    );
}

/// A site whose right-hand side alternates two shapes: the first stage marks its cell generic
/// (`Interp::instanceof_ic`), and the snapshot reads the mark.
#[test]
fn a_right_hand_side_the_cells_guards_reject_marks_the_cell_generic() {
    let mut engine = warmed(IO);
    for (name, generic) in [("d1", false), ("any", true), ("bnd", true)] {
        let chunk = chunk_of(&mut engine, name);
        let snapshot = crate::opt::snapshot::Snapshot::take(&chunk);
        let pc = pc_of(&chunk, |op| matches!(op, Op::InstanceOf(_)));
        let view = snapshot.instance_of(pc as usize).expect("a filled cell");
        assert_eq!(view.generic, generic, "{name}: {view:?}");
    }
}

/// Two engines alive on one thread, each running its own lowered sites: every CheckCtor reads its
/// own realm's Function.prototype, so that neither engine's code exits there.
#[test]
fn two_engines_on_one_thread_check_their_own_function_prototype() {
    let source = "
function P(a) { this.a = a; }
function isP(x) { return x instanceof P; }
function run(n) { var s = 0; for (var i = 0; i < n; i++) s += isP(i & 1 ? new P(i) : i) ? 1 : 0; return s; }
var t = 0; for (var k = 0; k < 200; k++) t += run(20); t";
    for graph in [false, true] {
        TestSwitches {
            on: true,
            asynchronous: false,
            inline_graph: Some(graph),
            ..Default::default()
        }
        .set();
        super::super::build::TEST_INSTANCE_OFS.with(|count| count.set(0));
        let mut a = Engine::new();
        a.set_tier(Tier::Jit);
        let mut b = Engine::new();
        b.set_tier(Tier::Jit);
        crate::opt::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        assert_eq!(eval(&mut a, source), "2000");
        assert_eq!(eval(&mut b, source), "2000");
        for _ in 0..3 {
            assert_eq!(eval(&mut a, "run(400)"), "200", "graph {graph}");
            assert_eq!(eval(&mut b, "run(400)"), "200", "graph {graph}");
        }
        let exits =
            crate::opt::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()));
        assert!(
            !exits
                .iter()
                .any(|&(_, reason)| reason == ExitReason::WrongShape),
            "graph {graph}: {exits:?}"
        );
        assert!(
            super::super::build::TEST_INSTANCE_OFS.with(|count| count.get()) >= 2,
            "graph {graph}: both engines lower"
        );
    }
    TestSwitches::default().set();
}

/// A loop whose only operation that could run code is a lowered InstanceOf: the rules count it
/// lowered (Rules::lowers), so that the loop is quiet and keeps p's shape and p.y.
const LOOP: &str = "
function P(a) { this.a = a; this.y = 2; }
function f(p, q, n) { var s = 0; for (var i = 0; i < n; i++) s = (s + (q instanceof P ? p.y : 1)) | 0; return s; }
var t = 0;
for (var k = 0; k < 40; k++) t += f(new P(k), (k & 1) ? new P(k) : 3, 5);
t";

#[test]
fn a_loop_whose_instanceof_is_lowered_keeps_its_facts() {
    let mut engine = warmed(LOOP);
    let built = Compile::new(&mut engine, "f").build();
    let graph = &built.graph;
    assert_eq!(graph.kept.len(), 1, "a quiet loop\n{}", dump(graph));
    assert!(graph.kept[0].shapes, "{:?}", graph.kept[0]);
    assert_eq!(
        count(graph, FrameIx(0), |op| matches!(
            op,
            NodeOp::HasInPrototypeChain
        )),
        1,
        "{}",
        dump(graph)
    );
}

/// A spliced callee's InstanceOf (TRUST_JS_OPT_INLINE=splice): the transformed chunk's site
/// reads its baseline's cell through the record map, as its value and element sites read their
/// records, and is lowered.
#[test]
fn a_spliced_callees_instanceof_reads_its_baselines_cell() {
    TestSwitches {
        on: true,
        asynchronous: false,
        inline_graph: Some(false),
        ..Default::default()
    }
    .set();
    super::super::build::TEST_INSTANCE_OFS.with(|count| count.set(0));
    let mut engine = jit_engine();
    let source = "
function P(a) { this.a = a; }
function isP(x) { return x instanceof P; }
function run(xs, n) { var r = 0; for (var i = 0; i < n; i++) if (isP(xs[i & 1])) r++; return r; }
var xs = [new P(1), 7];
var t = 0; for (var k = 0; k < 300; k++) t += run(xs, 40); t";
    assert_eq!(eval(&mut engine, source), "6000");
    let lowered = super::super::build::TEST_INSTANCE_OFS.with(|count| count.get());
    assert!(lowered > 0, "no InstanceOf lowered at the splice");
    // `run`'s second stage is the transformed chunk with isP spliced in: its snapshot reads the
    // spliced site's cell through the record map (the transformed chunk's own cell there is
    // empty), and its build lowers the walk in its frame.
    let (func, _) = function(&mut engine, "run");
    let transformed = func
        .runtime()
        .code2
        .get()
        .and_then(Option::clone)
        .expect("run's second stage, isP spliced in");
    let pc = pc_of(&transformed, |op| matches!(op, Op::InstanceOf(_)));
    let snapshot = crate::opt::snapshot::Snapshot::take(&transformed);
    let view = snapshot.instance_of(pc as usize);
    assert!(view.is_some_and(|view| !view.generic), "{view:?}");
    // Only a generic execution of the spliced site fills the transformed chunk's own cell: the
    // compile at the splice read the baseline's cell and lowered the site from the start.
    let Op::InstanceOf(cache) = transformed.jit_ops()[pc as usize] else {
        unreachable!()
    };
    let own = transformed.cache_state(cache);
    assert!(
        own.is_none_or(|state| state.depth != 0),
        "the spliced site ran generic: its own cell is filled"
    );
    let built = super::super::build::build(&transformed, &snapshot)
        .unwrap_or_else(|refusal| panic!("run's second stage: {refusal:?}"));
    TestSwitches::default().set();
    assert_eq!(
        count(&built.graph, FrameIx(0), |op| matches!(
            op,
            NodeOp::HasInPrototypeChain
        )),
        1,
        "{}",
        dump(&built.graph)
    );
}

/// The steady case, lowered: no change of the proof, no exit (a CheckCtor reading another
/// realm's word or another field, or a `prototype` read from another slot, would exit every
/// time and still give the right results).
#[test]
fn lowered_instanceof_runs_without_exits() {
    let source = "
function P(a) { this.a = a; }
class C extends P {}
function isP(x) { return x instanceof P; }
function run(n) { var s = 0; for (var i = 0; i < n; i++) s += isP(i % 3 ? (i & 1 ? new C(i) : { a: i }) : i) ? 1 : 0; return s; }
var t = 0; for (var k = 0; k < 300; k++) t += run(30); t";
    for graph in [false, true] {
        TestSwitches {
            on: true,
            asynchronous: false,
            inline_graph: Some(graph),
            ..Default::default()
        }
        .set();
        super::super::build::TEST_INSTANCE_OFS.with(|count| count.set(0));
        let mut engine = jit_engine();
        // No exit in the warm-up either: a check that fails every time exits there and its flag
        // declines the site, after which the results are the generic operation's.
        crate::opt::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        let first = eval(&mut engine, source);
        assert_eq!(eval(&mut engine, "run(600)"), "200", "graph {graph}");
        // Still the lowered form: a compile now, with every flag the warm-up raised, builds the
        // walk (a CheckCtor or a load that failed every time would have been flagged).
        let built = Compile::new(&mut engine, "isP").build();
        assert_eq!(
            count(&built.graph, FrameIx(0), |op| matches!(
                op,
                NodeOp::HasInPrototypeChain
            )),
            1,
            "graph {graph}: {}",
            dump(&built.graph)
        );
        let exits =
            crate::opt::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()));
        let lowered = super::super::build::TEST_INSTANCE_OFS.with(|count| count.get());
        assert!(lowered > 0, "graph {graph}: nothing lowered ({first})");
        assert!(exits.is_empty(), "graph {graph}: {exits:?}");
    }
    TestSwitches::default().set();
}

/// After the compile, with the lowered code running (asserted first: no exit in the warm-up, and
/// a compile now still builds both sites' walks, so that a check failing every time cannot pass
/// by being declined): P's [[Prototype]] moved to an object with a Symbol.hasInstance (P's shape
/// unchanged: CheckCtor's prototype test sees it), and an ordinary object given P's own
/// properties by Object.defineProperties from an empty literal, which reaches P's shape, with
/// Function.prototype as its [[Prototype]] (CheckCtor's kind and flags tests see that it is no
/// constructor: the generic operation answers false, its inherited @@hasInstance finding it not
/// callable, where a walk would answer true).
#[test]
fn a_constructors_prototype_or_kind_change_after_the_compile_is_seen() {
    let warm = "
function P(a) { this.a = a; }
function isP(x) { return x instanceof P; }
function viaF(x, f) { return x instanceof f ? 1 : 0; }
function run(n, v) { var s = 0; for (var i = 0; i < n; i++) s += isP(v) ? 1 : 0; return s; }
function run2(n, f) { var s = 0; for (var i = 0; i < n; i++) s += viaF(i & 1 ? i : new P(i), f); return s; }
var t = 0; for (var k = 0; k < 300; k++) t += run(30, 5) + run2(30, P); t";
    let change = "
var d = {};
for (var key of Reflect.ownKeys(P)) d[key] = Object.getOwnPropertyDescriptor(P, key);
var fake = {};
Object.defineProperties(fake, d);
Object.setPrototypeOf(fake, Function.prototype);
var r1; try { r1 = run2(10, fake); } catch (e) { r1 = e.constructor.name; }
var h = { [Symbol.hasInstance](v) { return typeof v === 'number'; } };
Object.setPrototypeOf(h, Function.prototype);
Object.setPrototypeOf(P, h);
[r1, run(10, 5)].join()";
    TestSwitches::default().set();
    let mut off = jit_engine();
    let expected = (eval(&mut off, warm), eval(&mut off, change));
    // `fake` inherits Function.prototype's @@hasInstance, whose OrdinaryHasInstance answers false
    // for a C that is not callable (ECMA-262 7.3.21 step 1): a walk would answer true for the
    // P instances.
    assert_eq!(expected.1, "0,10");
    for graph in [false, true] {
        TestSwitches {
            on: true,
            asynchronous: false,
            inline_graph: Some(graph),
            ..Default::default()
        }
        .set();
        super::super::build::TEST_INSTANCE_OFS.with(|count| count.set(0));
        crate::opt::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
        let mut engine = jit_engine();
        assert_eq!(eval(&mut engine, warm), expected.0, "graph {graph}");
        let exits =
            crate::opt::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()));
        assert!(exits.is_empty(), "graph {graph}: {exits:?}");
        for name in ["isP", "viaF"] {
            let built = Compile::new(&mut engine, name).build();
            assert_eq!(
                count(&built.graph, FrameIx(0), |op| matches!(
                    op,
                    NodeOp::HasInPrototypeChain
                )),
                1,
                "graph {graph}, {name}: {}",
                dump(&built.graph)
            );
        }
        assert_eq!(eval(&mut engine, change), expected.1, "graph {graph}");
    }
    TestSwitches::default().set();
}
