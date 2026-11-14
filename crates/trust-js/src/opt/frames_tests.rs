//! Frame-state chains and exits through several frames (P3 stage 3, step 1), on graphs built by
//! hand: no graph the builder makes has an inlined frame yet (step 2's inliner), so these build
//! the compiled function's graph with its callees' frames over the chunks of small functions the
//! bytecode VM ran, and check them against the verifier's rules for frames.

use super::ir::*;
use super::view::FrozenChunk;
use crate::bytecode::{Chunk, Op, Tier};
use crate::jit_ir::Cfg;
use crate::value::Value;
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the fixture parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

/// An engine whose functions compile to bytecode at their first call and run in the VM, until a
/// test installs the tier's code (as `a64_tests`'s fixture: the driver off, no exit forced but
/// the tests' own).
fn fixture(source: &str) -> Engine {
    super::TestSwitches {
        on: false,
        eager: false,
        exit_every: 0,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Bytecode);
    engine.set_tier_threshold(0);
    eval(&mut engine, source);
    engine
}

fn chunk_of(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("a global");
    let object = value.as_obj().expect("a function");
    let crate::value::ObjectKind::User(user) = &object.borrow().kind else {
        panic!("{name} is an author function")
    };
    let runtime = user.func.runtime();
    runtime
        .code
        .get()
        .and_then(Option::as_ref)
        .cloned()
        .expect("the function ran as bytecode")
}

fn ops(chunk: &Chunk) -> String {
    chunk
        .jit_ops()
        .iter()
        .enumerate()
        .map(|(pc, op)| format!("{pc}: {op:?}"))
        .collect::<Vec<_>>()
        .join("\n")
}

const FIXTURES: &str = "
function inner(a, b) { return a + b; }
function outer(f, x, y) { return f(x, y) * 2; }
function mid(g, a, b) { return g(a, b) + 1; }
function top3(f, g, x, y) { return f(g, x, y) * 2; }
function thrower(a) { if (a) throw a; return 1; }
function guarded(f, x) { try { return f(x) * 2; } catch (e) { return 'caught ' + e; } }
function plain(f, x) { return f(x) * 2; }
function stackOf(a) { return new Error('s' + a).stack; }
function outerS(f, x) { return f(x); }
function who(a) { var c = who.caller; return c ? c.name : String(c); }
function outerW(f, x) { return f(x); }
function churn(a) { var t = []; for (var j = 0; j < 50; j++) t.push({j: j}); return t.length + a.v; }
function outerC(f) { var o = {v: 7}; return f(o) + o.v; }
function getk(z) { return this.k + z; }
function meth(o, x) { return o.m(x) - 1; }
function wide(f, h, x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9) {
    h(); return f(x, x) + a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9;
}
function nothing() {}
function rest(...r) { return r.length; }
function strict(a) { 'use strict'; return a; }
function outerP(f, x) { return x + f(x, x); }
function kind(z) { return typeof this; }
function spin(a) { while (a) { a++; } }
function guardS(f, x) { try { return f(x); } catch (e) { return 'caught'; } }
function spin2(n) { while (n) {} }
function elem(a, i) { return a[i]; }
function outerT(f, a, i, c, q0, q1, q2, q3, q4, q5, q6, q7, q8, q9) {
    return f(a, i) + c + q0 + q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8 + q9;
}
outer(inner, 1, 2); top3(mid, inner, 1, 2); thrower(0); guarded(thrower, 0); plain(thrower, 0);
outerS(stackOf, 1); outerW(who, 1); outerC(churn); meth({k: 1, m: getk}, 1);
wide(inner, nothing, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); rest(1); strict(1);
outerP(inner, 1); outerW(kind, 1); guardS(kind, 1);
spin(0); spin2(0);
outerT(elem, new Uint8Array(2), 0, 'c', 'q', 'q', 'q', 'q', 'q', 'q', 'q', 'q', 'q', 'q');
";

/// The test's view of its fixtures' operations: run with `--nocapture` to print them.
#[test]
fn print_fixture_operations() {
    let mut engine = fixture(FIXTURES);
    for name in [
        "inner", "outer", "mid", "top3", "thrower", "guarded", "plain", "stackOf", "outerS", "who",
        "outerW", "churn", "outerC", "getk", "meth", "wide", "outerP", "kind", "guardS", "elem",
        "outerT",
    ] {
        let chunk = chunk_of(&mut engine, name);
        println!("== {name} {:?}\n{}", chunk.jit_frame(), ops(&chunk));
    }
}

fn f64_word(x: f64) -> u64 {
    x.to_bits()
}

/// A hand-built graph with frames.
struct Build {
    g: Graph,
}

const F0: FrameIx = FrameIx::COMPILED;

impl Build {
    fn new(slots: usize) -> Self {
        Build {
            g: Graph::new(slots),
        }
    }

    fn block(&mut self, pc: u32) -> BlockIx {
        let b = self.g.add_block(pc, false);
        if self.g.entry.is_none() {
            self.g.entry = Some(b);
        }
        b
    }

    fn edge(&mut self, from: BlockIx, to: BlockIx) {
        self.g.add_edge(from, to);
    }

    fn node(
        &mut self,
        frame: FrameIx,
        b: BlockIx,
        op: NodeOp,
        inputs: &[NodeIx],
        rep: Rep,
        pc: u32,
    ) -> NodeIx {
        self.g.add_node_in(frame, b, op, inputs, rep, pc)
    }

    fn undefined(&mut self, b: BlockIx) -> NodeIx {
        self.node(
            F0,
            b,
            NodeOp::ConstTagged(Value::Undefined.bits()),
            &[],
            Rep::Tagged,
            0,
        )
    }

    /// A state of `frame` at `pc`.
    fn state(
        &mut self,
        frame: FrameIx,
        pc: u32,
        stack: &[NodeIx],
        slots: Vec<SlotSource>,
        handlers: &[(u32, u32)],
    ) -> StateIx {
        self.g.add_state(FrameState {
            pc,
            slots: slots.into(),
            stack: stack.into(),
            handlers: handlers.into(),
            frame,
        })
    }

    /// A node that can exit, with its state.
    #[allow(clippy::too_many_arguments)]
    fn exiting(
        &mut self,
        frame: FrameIx,
        b: BlockIx,
        op: NodeOp,
        inputs: &[NodeIx],
        rep: Rep,
        pc: u32,
        state: StateIx,
        reason: ExitReason,
    ) -> NodeIx {
        let n = self.node(frame, b, op, inputs, rep, pc);
        self.g.node_mut(n).state = Some(state);
        self.g.node_mut(n).reason = Some(reason);
        n
    }

    fn exit(&mut self, frame: FrameIx, b: BlockIx, pc: u32, state: StateIx, reason: ExitReason) {
        self.exiting(
            frame,
            b,
            NodeOp::Exit(reason),
            &[],
            Rep::None,
            pc,
            state,
            reason,
        );
    }

    fn open(
        &mut self,
        chunk: u32,
        parent: StateIx,
        call_pc: u32,
        callee: NodeIx,
        receiver: NodeIx,
    ) -> FrameIx {
        self.g.open_frame(InlinedFrame {
            chunk,
            parent,
            call_pc,
            callee,
            receiver,
        })
    }
}

fn frame_slots(n: usize) -> Vec<SlotSource> {
    vec![SlotSource::Frame; n]
}

fn values(values: &[NodeIx]) -> Vec<SlotSource> {
    values.iter().map(|&v| SlotSource::Value(v)).collect()
}

/// The chunks and Cfgs of `callees`, in the graph's chunk order.
fn callee_chunks(engine: &mut Engine, callees: &[&str]) -> Vec<(Rc<Chunk>, Cfg)> {
    callees
        .iter()
        .map(|name| {
            let chunk = chunk_of(engine, name);
            let cfg = Cfg::build(&chunk).expect("the chunk's stack depths settle");
            (chunk, cfg)
        })
        .collect()
}

/// `graph` against the verifier, `name`'s chunk compiled and `callees` its inlined chunks.
fn verify(engine: &mut Engine, name: &str, callees: &[&str], graph: &Graph) -> Result<(), String> {
    let chunk = chunk_of(engine, name);
    let cfg = Cfg::build(&chunk).expect("the chunk's stack depths settle");
    let inlined = callee_chunks(engine, callees);
    let views: Vec<FrozenChunk> = inlined.iter().map(|(chunk, _)| chunk.freeze()).collect();
    let frames: Vec<super::verify::FrameChunk<'_>> = views
        .iter()
        .zip(&inlined)
        .map(|(view, (_, cfg))| (view, cfg))
        .collect();
    super::verify::verify_frames(graph, (&chunk.freeze(), &cfg), &frames)
}

/// `outer(f, x, y)` (LoadLocal 0-2, Call(2), Const 2, Mul, Return) with `inner(a, b)` (LoadLocal
/// 0-1, Add, Return) built in its frame: x checked a Number in the compiled frame before the call
/// (a NotNumber exit there rebuilds one frame), y checked in the callee's frame at its Add (a
/// NotNumber exit there rebuilds both, the caller resuming after its call with the result),
/// their sum doubled in the caller's frame. The callee is the call's operand, or with `guard`
/// (its word), a constant behind an identity test whose other side exits WrongCallee.
fn outer_inner(outer: &Chunk, inner: &Chunk, guard: Option<u64>) -> Graph {
    assert!(
        matches!(
            outer.jit_ops(),
            [
                Op::LoadLocal(0),
                Op::LoadLocal(1),
                Op::LoadLocal(2),
                Op::Call(2, _),
                Op::Const(_),
                Op::Mul,
                Op::Return,
                ..
            ]
        ),
        "{}",
        ops(outer)
    );
    assert!(
        matches!(
            inner.jit_ops(),
            [Op::LoadLocal(0), Op::LoadLocal(1), Op::Add, Op::Return, ..]
        ),
        "{}",
        ops(inner)
    );
    let mut b = Build::new(3);
    let entry = b.block(0);
    let f = b.node(F0, entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let x = b.node(F0, entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let y = b.node(F0, entry, NodeOp::EntrySlot(2), &[], Rep::Tagged, 2);
    let undefined = b.undefined(entry);
    let at_call = b.state(F0, 3, &[f, x, y], frame_slots(3), &[]);
    let (body, callee) = match guard {
        None => (entry, f),
        Some(word) => {
            let expected = b.node(F0, entry, NodeOp::ConstTagged(word), &[], Rep::Tagged, 3);
            let same = b.node(F0, entry, NodeOp::TaggedSame, &[f, expected], Rep::Bool, 3);
            let body = b.block(3);
            let wrong = b.block(3);
            b.node(
                F0,
                entry,
                NodeOp::Branch {
                    then_block: body,
                    else_block: wrong,
                },
                &[same],
                Rep::None,
                3,
            );
            b.edge(entry, body);
            b.edge(entry, wrong);
            b.exit(F0, wrong, 3, at_call, ExitReason::WrongCallee);
            (body, expected)
        }
    };
    let fx = b.exiting(
        F0,
        body,
        NodeOp::CheckNumber,
        &[x],
        Rep::F64,
        3,
        at_call,
        ExitReason::NotNumber,
    );
    let after = b.state(F0, 4, &[], frame_slots(3), &[]);
    let f1 = b.open(0, after, 3, callee, undefined);
    let at_add = b.state(f1, 2, &[fx, y], values(&[fx, y]), &[]);
    let fy = b.exiting(
        f1,
        body,
        NodeOp::CheckNumber,
        &[y],
        Rep::F64,
        2,
        at_add,
        ExitReason::NotNumber,
    );
    let sum = b.node(f1, body, NodeOp::F64Add, &[fx, fy], Rep::F64, 2);
    let two = b.node(F0, body, NodeOp::ConstF64(f64_word(2.0)), &[], Rep::F64, 4);
    let product = b.node(F0, body, NodeOp::F64Mul, &[sum, two], Rep::F64, 5);
    let boxed = b.node(F0, body, NodeOp::F64ToTagged, &[product], Rep::Tagged, 5);
    b.node(F0, body, NodeOp::Return, &[boxed], Rep::None, 6);
    b.g
}

/// A function of more locals than a pooled activation record holds.
fn big_function() -> String {
    let vars: Vec<String> = (0..300).map(|k| format!("v{k} = {k}")).collect();
    format!(
        "function big(a, b) {{ var {}; return a + b; }} big(1, 2);",
        vars.join(", ")
    )
}

/// What the verifier says of `graph` (outer's, with `callees` inlined).
fn refusal(engine: &mut Engine, callees: &[&str], graph: &Graph) -> String {
    verify(engine, "outer", callees, graph).expect_err("a refusal")
}

#[test]
fn the_verifier_refuses_frames_no_exit_can_rebuild() {
    let mut engine = fixture(&format!("{FIXTURES}\n{}", big_function()));
    let (outer, inner) = (
        chunk_of(&mut engine, "outer"),
        chunk_of(&mut engine, "inner"),
    );
    let graph = outer_inner(&outer, &inner, None);
    verify(&mut engine, "outer", &["inner"], &graph).expect("the graph itself holds");
    let node = |graph: &Graph, op: NodeOp| {
        NodeIx(
            graph
                .nodes
                .iter()
                .position(|n| n.op == op)
                .expect("the node") as u32,
        )
    };
    let parent = graph.frames[0].parent;
    let cases: Vec<(&str, Box<dyn Fn(&mut Graph)>, &[&str])> = vec![
        (
            "chunk 5, which the compile has not",
            Box::new(|g| g.frames[0].chunk = 5),
            &["inner"],
        ),
        (
            "a parent state of frame",
            Box::new(move |g| g.states[parent.index()].frame = FrameIx(1)),
            &["inner"],
        ),
        (
            "a call at pc 2",
            Box::new(|g| g.frames[0].call_pc = 2),
            &["inner"],
        ),
        (
            "not past its call",
            Box::new(move |g| g.states[parent.index()].pc = 5),
            &["inner"],
        ),
        (
            "a parent state of depth 1",
            Box::new(move |g| {
                let x = node(g, NodeOp::EntrySlot(1));
                g.states[parent.index()].stack = vec![x].into();
            }),
            &["inner"],
        ),
        (
            "a callee that is no Tagged node",
            Box::new(move |g| g.frames[0].callee = node(g, NodeOp::CheckNumber)),
            &["inner"],
        ),
        (
            "a receiver that is no Tagged node",
            Box::new(move |g| g.frames[0].receiver = NodeIx(999)),
            &["inner"],
        ),
        (
            "a Call's receiver that is not undefined",
            Box::new(move |g| g.frames[0].receiver = node(g, NodeOp::EntrySlot(1))),
            &["inner"],
        ),
        (
            "a callee whose call makes more than its frame",
            Box::new(|_| {}),
            &["rest"],
        ),
        (
            "a callee of another strictness",
            Box::new(|_| {}),
            &["strict"],
        ),
        (
            "a callee's frame past a pooled activation record",
            Box::new(|_| {}),
            &["big"],
        ),
    ];
    for (what, break_it, callees) in cases {
        let mut broken = graph.clone();
        break_it(&mut broken);
        let error = refusal(&mut engine, callees, &broken);
        assert!(error.contains(what), "{what}: {error}");
    }
}

#[test]
fn the_verifier_checks_each_frames_nodes_and_states_against_its_chunk() {
    let mut engine = fixture(FIXTURES);
    let (outer, inner) = (
        chunk_of(&mut engine, "outer"),
        chunk_of(&mut engine, "inner"),
    );
    let graph = outer_inner(&outer, &inner, None);
    let find = |graph: &Graph, op: NodeOp| {
        graph
            .nodes
            .iter()
            .position(|n| n.op == op)
            .expect("the node")
    };
    let fy = NodeIx(
        graph
            .nodes
            .iter()
            .rposition(|n| n.op == NodeOp::CheckNumber)
            .expect("inner's check") as u32,
    );
    let fx = NodeIx(find(&graph, NodeOp::CheckNumber) as u32);
    let at_call = graph.node(fx).state.expect("a state");
    let at_add = graph.node(fy).state.expect("a state");
    let parent = graph.frames[0].parent;
    let sum = NodeIx(find(&graph, NodeOp::F64Add) as u32);
    let cases: Vec<(&str, Box<dyn Fn(&mut Graph)>)> = vec![
        (
            "reads the compiled frame's context",
            Box::new(move |g| {
                let f1 = g.node(fy).frame;
                let entry = g.entry.expect("an entry");
                let slot = g.new_node(entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
                g.node_mut(slot).frame = f1;
                g.blocks[entry.index()].body.insert(0, slot);
            }),
        ),
        (
            "a frame that does not exist",
            Box::new(move |g| g.node_mut(sum).frame = FrameIx(9)),
        ),
        (
            "a pc past the chunk",
            Box::new(move |g| g.node_mut(sum).pc = 5),
        ),
        (
            "a state of frame FrameIx(0), not its own FrameIx(1)",
            Box::new(move |g| g.node_mut(fy).state = Some(at_call)),
        ),
        (
            "an inlined frame's parent state as its own",
            Box::new(move |g| g.node_mut(fx).state = Some(parent)),
        ),
        (
            "its frame's chain's",
            Box::new(move |g| {
                let mut slots = g.states[parent.index()].slots.to_vec();
                slots[0] = SlotSource::Value(sum);
                g.states[parent.index()].slots = slots.into();
            }),
        ),
        (
            "an inlined frame's slot read from a frame it has not",
            Box::new(move |g| {
                let mut slots = g.states[at_add.index()].slots.to_vec();
                slots[0] = SlotSource::Frame;
                g.states[at_add.index()].slots = slots.into();
            }),
        ),
        (
            "a handler record in an inlined frame",
            Box::new(move |g| g.states[at_add.index()].handlers = vec![(0, 0)].into()),
        ),
        (
            "block 0 starts past the chunk",
            Box::new(move |g| {
                // The entry's first node made inner's: outer's pc 6 is past inner's chunk.
                let first = g.blocks[0].body[0];
                g.node_mut(first).frame = FrameIx(1);
                g.blocks[0].first_pc = 6;
            }),
        ),
    ];
    for (what, break_it) in cases {
        let mut broken = graph.clone();
        break_it(&mut broken);
        let error = refusal(&mut engine, &["inner"], &broken);
        assert!(error.contains(what), "{what}: {error}");
    }
}
