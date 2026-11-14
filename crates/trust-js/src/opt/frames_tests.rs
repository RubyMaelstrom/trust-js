//! Frame-state chains and exits through several frames (P3 stage 3, step 1), on graphs built by
//! hand: no graph the builder makes has an inlined frame yet (step 2's inliner), so these build
//! the compiled function's graph with its callees' frames over the chunks of small functions the
//! bytecode VM ran, generate its code (`a64::compile_frames_with`) and install it, as `a64_tests`
//! does for one frame, and compare what the functions return with the VM's answers.

use super::a64::Options;
use super::exit::{TEST_EXITS, TEST_INLINED_EXITS};
use super::ir::*;
use super::view::FrozenChunk;
use crate::bytecode::{Chunk, Op, Tier};
use crate::jit::JitCode;
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

/// The VM's answer to `source` after the fixtures, the tier never involved.
fn vm_answer(source: &str) -> String {
    let mut engine = fixture(FIXTURES);
    eval(&mut engine, source)
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

/// The word of global `name`'s value.
fn word_of(engine: &mut Engine, name: &str) -> u64 {
    let env = engine.interp.global_env;
    engine
        .interp
        .get_var(name, &env)
        .ok()
        .expect("a global")
        .bits()
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

/// The pc of the only operation of `chunk` that `is` matches.
fn pc_of(chunk: &Chunk, is: impl Fn(&Op) -> bool) -> u32 {
    let pcs: Vec<usize> = (0..chunk.jit_ops().len())
        .filter(|&pc| is(&chunk.jit_ops()[pc]))
        .collect();
    match pcs[..] {
        [pc] => pc as u32,
        _ => panic!("{pcs:?} match in\n{}", ops(chunk)),
    }
}

fn is_call(op: &Op) -> bool {
    matches!(op, Op::Call(..) | Op::CallWithThis(..))
}

fn depth_at(chunk: &Chunk, pc: u32) -> usize {
    Cfg::build(chunk)
        .expect("the chunk's stack depths settle")
        .stack_depth_at(pc as usize)
        .expect("a reachable pc")
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
var caughtS = 0;
function guardS(f, x) { try { return f(x); } catch (e) { caughtS++; return 'caught'; } }
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

/// The exits taken since the last call: every exit's (pc, reason), and each exit through
/// inlined frames' (chain depth, pc, reason).
fn exits() -> (Vec<(u32, ExitReason)>, Vec<(usize, u32, ExitReason)>) {
    (
        TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut())),
        TEST_INLINED_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut())),
    )
}

/// The engine's activation records are all pooled again, its depth and function frames those of
/// a script between two calls: what each exit through inlined frames made, it released.
fn assert_frames_released(engine: &Engine) {
    assert!(
        engine.interp.frame_pool.active_records().is_empty(),
        "an activation record is still out of the pool"
    );
    assert_eq!(engine.interp.depth, 0, "the depth is given back");
    assert!(
        engine.interp.fn_frames.is_empty(),
        "every FnFrame is popped"
    );
}

/// Collect at the first safepoint after every allocation (`TRUST_JS_GC_STRESS=1` on this
/// engine's heap), marking in one pause.
fn stress(engine: &Engine) {
    let mut core = engine.interp.gc_heap.core.borrow_mut();
    core.set_stress_every(1);
    core.set_incremental(crate::gc_core::IncrementalMode::Off);
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

/// A callee frame's slots at its entry: its parameters the call's arguments, the rest undefined.
fn entry_slots(callee: &Chunk, args: &[NodeIx], undefined: NodeIx) -> Vec<SlotSource> {
    let (n_params, n_slots) = callee.jit_frame();
    (0..n_slots)
        .map(|k| match args.get(k).filter(|_| k < n_params) {
            Some(&arg) => SlotSource::Value(arg),
            None => SlotSource::Value(undefined),
        })
        .collect()
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

/// Generate `graph`'s code for `name`'s chunk with `callees`' chunks inlined, after the verifier
/// and the allocation's consistency checks, install it as the chunk's native code and move the
/// engine to the JIT tier, whose entries then run it.
fn install(
    engine: &mut Engine,
    name: &str,
    callees: &[&str],
    graph: &Graph,
    options: Options,
) -> Rc<JitCode> {
    if let Err(what) = verify(engine, name, callees, graph) {
        panic!("{name}'s graph fails verification: {what}");
    }
    let alloc = super::regalloc::allocate(graph, &super::regalloc::ARM64).unwrap();
    super::regalloc::tests::assert_consistent(graph, &alloc);
    let chunk = chunk_of(engine, name);
    let cfg = Cfg::build(&chunk).expect("the chunk's stack depths settle");
    let inlined = callee_chunks(engine, callees);
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let initialized = cfg.initialized_reads(chunk.jit_ops(), true);
    let code = super::a64::compile_frames_with(
        &chunk,
        graph,
        &cfg,
        &inlined,
        initialized,
        &layout,
        &ilayout,
        options,
    )
    .unwrap_or_else(|refusal| panic!("the tier refused {name}: {refusal:?}"));
    chunk
        .jit
        .begin_compile()
        .expect("an untried code slot")
        .commit(Some(code.clone()));
    engine.set_tier(Tier::Jit);
    code
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

#[test]
fn an_exit_in_an_inlined_frame_resumes_its_caller_after_the_call_with_the_result() {
    let mut engine = fixture(FIXTURES);
    let (outer, inner) = (
        chunk_of(&mut engine, "outer"),
        chunk_of(&mut engine, "inner"),
    );
    install(
        &mut engine,
        "outer",
        &["inner"],
        &outer_inner(&outer, &inner, None),
        Options::default(),
    );
    exits();
    // In the code: (1 + 2) * 2.
    assert_eq!(eval(&mut engine, "outer(inner, 1, 2)"), "6");
    assert_eq!(exits(), (vec![], vec![]));
    // The callee's check: its frame rebuilt at its Add, the caller's after its call; the VM adds
    // '1' + '3' in the callee and doubles '13' in the caller.
    for (call, expected) in [
        ("outer(inner, 1, '3')", "26"),
        ("outer(inner, 0.5, true)", "3"),
        ("outer(inner, 1, null)", "2"),
    ] {
        assert_eq!(eval(&mut engine, call), expected, "{call}");
        assert_eq!(vm_answer(call), expected, "{call} in the VM");
        assert_eq!(
            exits(),
            (
                vec![(2, ExitReason::NotNumber)],
                vec![(1, 2, ExitReason::NotNumber)]
            ),
            "{call}"
        );
        assert_frames_released(&engine);
    }
    // The caller's own check before the call rebuilds its frame alone.
    assert_eq!(eval(&mut engine, "outer(inner, '4', 2)"), "84");
    assert_eq!(exits(), (vec![(3, ExitReason::NotNumber)], vec![]));
    assert_frames_released(&engine);
}

#[test]
fn a_constant_callee_behind_its_identity_test_is_rebuilt_from_the_constant() {
    let mut engine = fixture(FIXTURES);
    let (outer, inner) = (
        chunk_of(&mut engine, "outer"),
        chunk_of(&mut engine, "inner"),
    );
    let word = word_of(&mut engine, "inner");
    let code = install(
        &mut engine,
        "outer",
        &["inner"],
        &outer_inner(&outer, &inner, Some(word)),
        Options::default(),
    );
    let table = super::exit::table_of(&code).expect("the tier's code");
    let frame = &table.frames[0];
    assert_eq!(
        (frame.callee, frame.receiver),
        (
            super::exit::Source::Const(word),
            super::exit::Source::Const(Value::Undefined.bits())
        )
    );
    exits();
    assert_eq!(eval(&mut engine, "outer(inner, 1, 2)"), "6");
    assert_eq!(eval(&mut engine, "outer(inner, 1, '3')"), "26");
    assert_eq!(
        exits().1,
        vec![(1, 2, ExitReason::NotNumber)],
        "the callee's frame rebuilt"
    );
    // Another callee takes the identity test's other side.
    let call = "outer(mid, inner, 5) + ':' + outer(function (a, b) { return a - b; }, 5, 2)";
    assert_eq!(eval(&mut engine, call), vm_answer(call));
    assert_eq!(eval(&mut engine, call), "NaN:6");
    let wrong = exits().0;
    assert!(
        !wrong.is_empty()
            && wrong
                .iter()
                .all(|&exit| exit == (3, ExitReason::WrongCallee)),
        "{wrong:?}"
    );
    assert_frames_released(&engine);
}

/// `top3(f, g, x, y)` (LoadLocal 0-3, Call(3), Const 2, Mul, Return) with `mid(g, a, b)`
/// (LoadLocal 0-2, Call(2), Const 1, Add, Return) in its frame and `inner(a, b)` in mid's: three
/// frames, each with a check that exits with its own state: x a Number in top3's frame at its
/// call, x an int32 in mid's at its call, y a Number in inner's at its Add. Every frame's values
/// are of another representation: mid's slots hold the callee g (a word), x (an f64) and y (a
/// word), inner's the int32 and the word, each receiver the constant undefined.
fn three_frames(top3: &Chunk, mid: &Chunk, inner: &Chunk) -> Graph {
    assert!(
        matches!(
            top3.jit_ops(),
            [
                Op::LoadLocal(0),
                Op::LoadLocal(1),
                Op::LoadLocal(2),
                Op::LoadLocal(3),
                Op::Call(3, _),
                Op::Const(_),
                Op::Mul,
                Op::Return,
                ..
            ]
        ),
        "{}",
        ops(top3)
    );
    assert!(
        matches!(
            mid.jit_ops(),
            [
                Op::LoadLocal(0),
                Op::LoadLocal(1),
                Op::LoadLocal(2),
                Op::Call(2, _),
                Op::Const(_),
                Op::Add,
                Op::Return,
                ..
            ]
        ),
        "{}",
        ops(mid)
    );
    assert!(
        matches!(
            inner.jit_ops(),
            [Op::LoadLocal(0), Op::LoadLocal(1), Op::Add, Op::Return, ..]
        ),
        "{}",
        ops(inner)
    );
    let mut b = Build::new(4);
    let entry = b.block(0);
    let f = b.node(F0, entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let g = b.node(F0, entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let x = b.node(F0, entry, NodeOp::EntrySlot(2), &[], Rep::Tagged, 2);
    let y = b.node(F0, entry, NodeOp::EntrySlot(3), &[], Rep::Tagged, 3);
    let undefined = b.undefined(entry);
    let at_call = b.state(F0, 4, &[f, g, x, y], frame_slots(4), &[]);
    let fx = b.exiting(
        F0,
        entry,
        NodeOp::CheckNumber,
        &[x],
        Rep::F64,
        4,
        at_call,
        ExitReason::NotNumber,
    );
    let after_mid = b.state(F0, 5, &[], frame_slots(4), &[]);
    let f1 = b.open(0, after_mid, 4, f, undefined);
    let mid_slots = values(&[g, fx, y]);
    let at_inner = b.state(f1, 3, &[g, fx, y], mid_slots.clone(), &[]);
    let ix = b.exiting(
        f1,
        entry,
        NodeOp::CheckInt32,
        &[fx],
        Rep::I32,
        3,
        at_inner,
        ExitReason::NotInt32,
    );
    let after_inner = b.state(f1, 4, &[], mid_slots, &[]);
    let f2 = b.open(1, after_inner, 3, g, undefined);
    let at_add = b.state(f2, 2, &[ix, y], values(&[ix, y]), &[]);
    let fy = b.exiting(
        f2,
        entry,
        NodeOp::CheckNumber,
        &[y],
        Rep::F64,
        2,
        at_add,
        ExitReason::NotNumber,
    );
    let wide = b.node(f2, entry, NodeOp::I32ToF64, &[ix], Rep::F64, 2);
    let sum = b.node(f2, entry, NodeOp::F64Add, &[wide, fy], Rep::F64, 2);
    let one = b.node(f1, entry, NodeOp::ConstF64(f64_word(1.0)), &[], Rep::F64, 4);
    let plus = b.node(f1, entry, NodeOp::F64Add, &[sum, one], Rep::F64, 5);
    let two = b.node(F0, entry, NodeOp::ConstF64(f64_word(2.0)), &[], Rep::F64, 5);
    let product = b.node(F0, entry, NodeOp::F64Mul, &[plus, two], Rep::F64, 6);
    let boxed = b.node(F0, entry, NodeOp::F64ToTagged, &[product], Rep::Tagged, 6);
    b.node(F0, entry, NodeOp::Return, &[boxed], Rep::None, 7);
    b.g
}

fn install_three(engine: &mut Engine, options: Options) -> Rc<JitCode> {
    let top3 = chunk_of(engine, "top3");
    let mid = chunk_of(engine, "mid");
    let inner = chunk_of(engine, "inner");
    let graph = three_frames(&top3, &mid, &inner);
    install(engine, "top3", &["mid", "inner"], &graph, options)
}

/// The three-frame calls, their answers and the exit each takes: none, then one at each depth.
const THREE: [(&str, &str, Option<(usize, u32, ExitReason)>); 5] = [
    ("top3(mid, inner, 1, 2)", "8", None),
    (
        "top3(mid, inner, 1, '3')",
        "262",
        Some((2, 2, ExitReason::NotNumber)),
    ),
    (
        "top3(mid, inner, 1.5, 2)",
        "9",
        Some((1, 3, ExitReason::NotInt32)),
    ),
    (
        "top3(mid, inner, '4', 2)",
        "842",
        Some((0, 4, ExitReason::NotNumber)),
    ),
    (
        "top3(mid, inner, -0, 0.25)",
        "2.5",
        Some((1, 3, ExitReason::NotInt32)),
    ),
];

#[test]
fn a_three_frame_chain_exits_at_each_depth_and_resumes_every_caller() {
    let mut engine = fixture(FIXTURES);
    install_three(&mut engine, Options::default());
    exits();
    for (call, expected, exit) in THREE {
        assert_eq!(eval(&mut engine, call), expected, "{call}");
        assert_eq!(vm_answer(call), expected, "{call} in the VM");
        let (all, inlined) = exits();
        match exit {
            None => assert_eq!((all, inlined), (vec![], vec![]), "{call}"),
            Some((0, pc, reason)) => {
                assert_eq!((all, inlined), (vec![(pc, reason)], vec![]), "{call}")
            }
            Some((depth, pc, reason)) => assert_eq!(
                (all, inlined),
                (vec![(pc, reason)], vec![(depth, pc, reason)]),
                "{call}"
            ),
        }
        assert_frames_released(&engine);
    }
}

#[test]
fn the_parent_recipes_are_kept_once_for_every_exit_through_their_frames() {
    let mut engine = fixture(FIXTURES);
    let code = install_three(&mut engine, Options::default());
    let table = super::exit::table_of(&code).expect("the tier's code");
    // Three exits, one per frame; two inlined frames, each recipe once: mid's serves its own
    // exit and inner's, as inner's parent.
    let mut frames: Vec<u16> = table.exits.iter().map(|exit| exit.frame).collect();
    frames.sort_unstable();
    assert_eq!(frames, vec![0, 1, 2]);
    assert_eq!(table.frames.len(), 2);
    assert_eq!(
        table
            .frames
            .iter()
            .map(|frame| (frame.chunk, frame.parent, frame.call_pc, frame.stack.len()))
            .collect::<Vec<_>>(),
        vec![(0, 0, 4, 0), (1, 1, 3, 0)]
    );
    // mid's parent recipe writes no slot of top3's (each is the frame's); inner's writes mid's
    // three.
    assert_eq!(
        (table.frames[0].slots.len(), table.frames[1].slots.len()),
        (0, 3)
    );
    // The table's bytes: each record's and each frame's once.
    let record = std::mem::size_of::<super::exit::ExitRecord>();
    let frame = std::mem::size_of::<super::exit::FrameRecipe>();
    let source = std::mem::size_of::<super::exit::Source>();
    let slot = std::mem::size_of::<(u16, super::exit::Source)>();
    let records: usize = table
        .exits
        .iter()
        .map(|exit| record + exit.slots.len() * slot + exit.stack.len() * source)
        .sum();
    let frames: usize = table
        .frames
        .iter()
        .map(|recipe| frame + recipe.slots.len() * slot + recipe.stack.len() * source)
        .sum();
    assert_eq!(
        table.retained_bytes(),
        std::mem::size_of::<super::exit::ExitTable>()
            + 4 * table.exits.len()
            + records
            + frames
            + table.name.len()
    );
    // A record is no larger than before the frames: its frame in its padding.
    assert_eq!(record, 72);
}

#[test]
fn forced_exits_at_every_check_of_every_frame_resume_in_the_vm() {
    let mut engine = fixture(FIXTURES);
    install_three(
        &mut engine,
        Options {
            forced_exits: true,
            ..Options::default()
        },
    );
    // An interval fixes which check of a call the gate stops at (with the natural exits shifting
    // it): over 1 to 4, every check of every frame.
    exits();
    for every in 1..=4 {
        engine.interp.opt_exit_every = every;
        engine.interp.opt_exit_countdown = every;
        for _ in 0..3 {
            for (call, expected, _) in THREE {
                assert_eq!(eval(&mut engine, call), expected, "{call} every {every}");
                assert_frames_released(&engine);
            }
        }
    }
    let (all, inlined) = exits();
    let forced = |depth: usize| {
        inlined
            .iter()
            .any(|&(d, _, reason)| d == depth && reason == ExitReason::Forced)
    };
    assert!(
        all.contains(&(4, ExitReason::Forced)) && forced(1) && forced(2),
        "forced exits at each frame's check: {all:?} {inlined:?}"
    );
}

/// `caller(f, x)` (`guarded`: PushHandler, LoadLocal 0-1, Call(1), Const 2, Mul, ..., a catch
/// pad; or `plain`, without the try) with `thrower(a)` (LoadLocal 0, JumpIfFalse, LoadLocal 0,
/// Throw, Const 1, Return) in its frame: the callee's throwing side never ran, so it exits
/// NeverRan, and the VM throws in the callee's frame, which the caller's frame takes at its
/// call: to its catch, or out.
fn throwing(caller: &Chunk, thrower: &Chunk, handlers: &[(u32, u32)]) -> Graph {
    let call = pc_of(caller, is_call);
    assert!(
        matches!(caller.jit_ops()[call as usize], Op::Call(1, _))
            && matches!(
                caller.jit_ops()[call as usize + 1..],
                [Op::Const(_), Op::Mul, ..]
            ),
        "{}",
        ops(caller)
    );
    let ret = (call + 3..caller.jit_ops().len() as u32)
        .find(|&pc| caller.jit_ops()[pc as usize] == Op::Return)
        .unwrap_or_else(|| panic!("a Return after the Mul\n{}", ops(caller)));
    let throw = pc_of(thrower, |op| *op == Op::Throw);
    let test = pc_of(thrower, |op| matches!(op, Op::JumpIfFalse(_)));
    let one = pc_of(thrower, |op| matches!(op, Op::Const(_)));
    let (_, n_slots) = caller.jit_frame();
    let mut b = Build::new(n_slots);
    let entry = b.block(0);
    let f = b.node(F0, entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, call - 2);
    let x = b.node(F0, entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, call - 1);
    let undefined = b.undefined(entry);
    let after = b.state(F0, call + 1, &[], frame_slots(n_slots), handlers);
    let f1 = b.open(0, after, call, f, undefined);
    let truthy = b.node(f1, entry, NodeOp::ToBoolean, &[x], Rep::Bool, test);
    let throws = b.block(throw - 1);
    let returns = b.block(one);
    b.node(
        f1,
        entry,
        NodeOp::Branch {
            then_block: throws,
            else_block: returns,
        },
        &[truthy],
        Rep::None,
        test,
    );
    b.edge(entry, throws);
    b.edge(entry, returns);
    let at_throw = b.state(f1, throw - 1, &[], values(&[x]), &[]);
    b.exit(f1, throws, throw - 1, at_throw, ExitReason::NeverRan);
    // The callee returns 1, which its caller doubles.
    let result = b.node(
        f1,
        returns,
        NodeOp::ConstF64(f64_word(1.0)),
        &[],
        Rep::F64,
        one,
    );
    let two = b.node(
        F0,
        returns,
        NodeOp::ConstF64(f64_word(2.0)),
        &[],
        Rep::F64,
        call + 1,
    );
    let product = b.node(
        F0,
        returns,
        NodeOp::F64Mul,
        &[result, two],
        Rep::F64,
        call + 2,
    );
    let boxed = b.node(
        F0,
        returns,
        NodeOp::F64ToTagged,
        &[product],
        Rep::Tagged,
        call + 2,
    );
    b.node(F0, returns, NodeOp::Return, &[boxed], Rep::None, ret);
    b.g
}

#[test]
fn a_throw_in_an_inlined_frame_reaches_its_callers_handler_or_leaves_it() {
    let mut engine = fixture(FIXTURES);
    let guarded = chunk_of(&mut engine, "guarded");
    let plain = chunk_of(&mut engine, "plain");
    let thrower = chunk_of(&mut engine, "thrower");
    let push = pc_of(&guarded, |op| matches!(op, Op::PushHandler(_)));
    let depth = depth_at(&guarded, push) as u32;
    install(
        &mut engine,
        "guarded",
        &["thrower"],
        &throwing(&guarded, &thrower, &[(push, depth)]),
        Options::default(),
    );
    install(
        &mut engine,
        "plain",
        &["thrower"],
        &throwing(&plain, &thrower, &[]),
        Options::default(),
    );
    exits();
    for (call, expected) in [
        ("guarded(thrower, 0)", "2"),
        ("guarded(thrower, 5)", "caught 5"),
        ("guarded(thrower, 'x')", "caught x"),
        ("plain(thrower, 0)", "2"),
        ("try { plain(thrower, 7) } catch (e) { 'out ' + e }", "out 7"),
        (
            "var n = 0; for (var k = 0; k < 20; k++) { try { n += plain(thrower, k % 2) } catch (e) { n += 100 * e } } n",
            "1020",
        ),
    ] {
        assert_eq!(eval(&mut engine, call), expected, "{call}");
        assert_eq!(vm_answer(call), expected, "{call} in the VM");
        assert_frames_released(&engine);
    }
    let (_, inlined) = exits();
    assert_eq!(inlined.len(), 2 + 1 + 10, "{inlined:?}");
}

/// `caller(f, x)` (LoadLocal 0-1, Call(1), ...; `handlers` the records active at the call) with
/// its callee's frame exiting NeverRan at the callee's first operation: the callee runs in the VM
/// on its rebuilt frame, from its entry.
fn at_callee_entry(caller: &Chunk, callee: &Chunk, handlers: &[(u32, u32)]) -> Graph {
    let call = pc_of(caller, is_call);
    assert!(
        call >= 2
            && matches!(
                caller.jit_ops()[call as usize - 2..=call as usize],
                [Op::LoadLocal(0), Op::LoadLocal(1), Op::Call(1, _)]
            ),
        "{}",
        ops(caller)
    );
    let (_, n_slots) = caller.jit_frame();
    let mut b = Build::new(n_slots);
    let entry = b.block(0);
    let f = b.node(F0, entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let x = b.node(F0, entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let undefined = b.undefined(entry);
    let after = b.state(F0, call + 1, &[], frame_slots(n_slots), handlers);
    let f1 = b.open(0, after, call, f, undefined);
    let at_entry = b.state(f1, 0, &[], entry_slots(callee, &[x], undefined), &[]);
    b.exit(f1, entry, 0, at_entry, ExitReason::NeverRan);
    b.g
}

#[test]
fn error_stacks_and_callers_inside_the_inner_run_name_every_frame() {
    let mut engine = fixture(FIXTURES);
    for (caller, callee) in [("outerS", "stackOf"), ("outerW", "who")] {
        let (a, b) = (chunk_of(&mut engine, caller), chunk_of(&mut engine, callee));
        install(
            &mut engine,
            caller,
            &[callee],
            &at_callee_entry(&a, &b, &[]),
            Options::default(),
        );
    }
    exits();
    for call in [
        "outerS(stackOf, 1)",
        "function deeper() { return outerS(stackOf, 2); } deeper()",
        "outerW(who, 1)",
        "(function named() { return outerW(who, 1); })()",
    ] {
        let answer = eval(&mut engine, call);
        assert_eq!(answer, vm_answer(call), "{call}");
        assert_frames_released(&engine);
    }
    assert_eq!(eval(&mut engine, "outerW(who, 1)"), "outerW");
    assert!(
        eval(&mut engine, "outerS(stackOf, 3)").contains("at outerS"),
        "the caller's frame is named"
    );
    assert_eq!(exits().1.len(), 6);
}

/// `outerC(f)` (Const 7, MakeObject, StoreLocal 1, LoadLocal 0-1, Call(1), GetPropLocal 1, Add,
/// Return): the object made by an effect and held in a register only (the graph never stores it),
/// passed to `churn(a)`, whose frame exits at its entry: churn allocates in the VM (collections
/// under stress), the caller's frame reads the object from the slot its recipe wrote.
fn object_across(outer: &Chunk, churn: &Chunk) -> Graph {
    let make = pc_of(outer, |op| matches!(op, Op::MakeObject(..)));
    let call = pc_of(outer, is_call);
    assert!(
        matches!(outer.jit_ops()[make as usize - 1], Op::Const(_))
            && matches!(outer.jit_ops()[call as usize], Op::Call(1, _)),
        "{}",
        ops(outer)
    );
    let (_, n_slots) = outer.jit_frame();
    let mut b = Build::new(n_slots);
    let entry = b.block(0);
    let f = b.node(F0, entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let seven = b.node(
        F0,
        entry,
        NodeOp::ConstTagged(Value::Num(7.0).bits()),
        &[],
        Rep::Tagged,
        make - 1,
    );
    let undefined = b.undefined(entry);
    let at_make = b.state(F0, make, &[seven], frame_slots(n_slots), &[]);
    let effect = b.exiting(
        F0,
        entry,
        NodeOp::Effect {
            pc: make,
            pops: 1,
            pushes: 1,
        },
        &[seven],
        Rep::None,
        make,
        at_make,
        ExitReason::Forced,
    );
    let o = b.node(
        F0,
        entry,
        NodeOp::Projection(0),
        &[effect],
        Rep::Tagged,
        make,
    );
    let mut slots = frame_slots(n_slots);
    slots[1] = SlotSource::Value(o);
    let after = b.state(F0, call + 1, &[], slots, &[]);
    let f1 = b.open(0, after, call, f, undefined);
    let at_entry = b.state(f1, 0, &[], entry_slots(churn, &[o], undefined), &[]);
    b.exit(f1, entry, 0, at_entry, ExitReason::NeverRan);
    b.g
}

#[test]
fn collections_during_the_inner_run_keep_every_frames_values() {
    let mut engine = fixture(FIXTURES);
    let (outer, churn) = (
        chunk_of(&mut engine, "outerC"),
        chunk_of(&mut engine, "churn"),
    );
    install(
        &mut engine,
        "outerC",
        &["churn"],
        &object_across(&outer, &churn),
        Options::default(),
    );
    stress(&engine);
    exits();
    let call = "var s = 0; for (var k = 0; k < 20; k++) s += outerC(churn); s";
    assert_eq!(eval(&mut engine, call), "1280");
    assert_eq!(vm_answer(call), "1280");
    assert_eq!(exits().1.len(), 20);
    assert_frames_released(&engine);
}

/// `meth(o, x)` (LoadLocal 0, the method's load, LoadLocal 1, CallWithThis(1), Const 1, Sub,
/// Return) with the method's frame exiting at its entry: the receiver, the load's first result,
/// is the callee's `this`, which reads its `k`.
fn with_receiver(meth: &Chunk, getk: &Chunk) -> Graph {
    let call = pc_of(meth, is_call);
    assert!(
        matches!(meth.jit_ops()[call as usize], Op::CallWithThis(1, _))
            && meth.jit_ops()[..call as usize]
                .iter()
                .filter(|op| !matches!(op, Op::LoadLocal(_)))
                .count()
                == 1,
        "{}",
        ops(meth)
    );
    let load = (0..call)
        .find(|&pc| !matches!(meth.jit_ops()[pc as usize], Op::LoadLocal(_)))
        .expect("the method's load");
    let mut b = Build::new(2);
    let entry = b.block(0);
    let o = b.node(F0, entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let x = b.node(F0, entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
    let at_load = b.state(F0, load, &[o], frame_slots(2), &[]);
    let effect = b.exiting(
        F0,
        entry,
        NodeOp::Effect {
            pc: load,
            pops: 1,
            pushes: 2,
        },
        &[o],
        Rep::None,
        load,
        at_load,
        ExitReason::Forced,
    );
    let receiver = b.node(
        F0,
        entry,
        NodeOp::Projection(0),
        &[effect],
        Rep::Tagged,
        load,
    );
    let method = b.node(
        F0,
        entry,
        NodeOp::Projection(1),
        &[effect],
        Rep::Tagged,
        load,
    );
    let undefined = b.undefined(entry);
    let after = b.state(F0, call + 1, &[], frame_slots(2), &[]);
    let f1 = b.open(0, after, call, method, receiver);
    let at_entry = b.state(f1, 0, &[], entry_slots(getk, &[x], undefined), &[]);
    b.exit(f1, entry, 0, at_entry, ExitReason::NeverRan);
    b.g
}

#[test]
fn a_method_calls_receiver_is_bound_as_its_frames_this() {
    let mut engine = fixture(FIXTURES);
    let (meth, getk) = (chunk_of(&mut engine, "meth"), chunk_of(&mut engine, "getk"));
    install(
        &mut engine,
        "meth",
        &["getk"],
        &with_receiver(&meth, &getk),
        Options::default(),
    );
    exits();
    for call in [
        "meth({k: 10, m: getk}, 5)",
        "var o = {k: 'a', m: getk}; meth(o, 'b') + meth(o, 1)",
    ] {
        assert_eq!(eval(&mut engine, call), vm_answer(call), "{call}");
        assert_frames_released(&engine);
    }
    assert_eq!(eval(&mut engine, "meth({k: 10, m: getk}, 5)"), "14");
    assert_eq!(exits().1.len(), 4);
}

/// `wide(f, h, x, a0, ..., a9)` (LoadLocal 1, Call(0), Pop, LoadLocal 0, LoadLocal 2 twice,
/// Call(2), then the sum): h's call an effect, across which the callee, x and the ten later
/// summands live, so that most of them live in callee-saved registers and spill slots; inner's
/// frame for f(x, x), whose check exits through it, and the caller's frame exiting after the call
/// with the result, its summands written from the parent recipe's homes.
fn spilled(wide: &Chunk) -> Graph {
    let calls: Vec<u32> = (0..wide.jit_ops().len() as u32)
        .filter(|&pc| is_call(&wide.jit_ops()[pc as usize]))
        .collect();
    let [first, call] = calls[..] else {
        panic!("two calls\n{}", ops(wide))
    };
    assert!(
        matches!(wide.jit_ops()[first as usize], Op::Call(0, _))
            && matches!(wide.jit_ops()[call as usize], Op::Call(2, _)),
        "{}",
        ops(wide)
    );
    let (_, n_slots) = wide.jit_frame();
    let mut b = Build::new(n_slots);
    let entry = b.block(0);
    let f = b.node(F0, entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let h = b.node(F0, entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
    let x = b.node(F0, entry, NodeOp::EntrySlot(2), &[], Rep::Tagged, 0);
    let summands: Vec<NodeIx> = (0..10)
        .map(|k| b.node(F0, entry, NodeOp::EntrySlot(3 + k), &[], Rep::Tagged, 0))
        .collect();
    let undefined = b.undefined(entry);
    let at_h = b.state(F0, first, &[h], frame_slots(n_slots), &[]);
    let effect = b.exiting(
        F0,
        entry,
        NodeOp::Effect {
            pc: first,
            pops: 1,
            pushes: 1,
        },
        &[h],
        Rep::None,
        first,
        at_h,
        ExitReason::Forced,
    );
    b.node(
        F0,
        entry,
        NodeOp::Projection(0),
        &[effect],
        Rep::Tagged,
        first,
    );
    let mut slots = frame_slots(n_slots);
    for (k, &summand) in summands.iter().enumerate() {
        slots[3 + k] = SlotSource::Value(summand);
    }
    let after = b.state(F0, call + 1, &[], slots, &[]);
    let f1 = b.open(0, after, call, f, undefined);
    let at_add = b.state(f1, 2, &[x, x], values(&[x, x]), &[]);
    let fx = b.exiting(
        f1,
        entry,
        NodeOp::CheckNumber,
        &[x],
        Rep::F64,
        2,
        at_add,
        ExitReason::NotNumber,
    );
    let sum = b.node(f1, entry, NodeOp::F64Add, &[fx, fx], Rep::F64, 2);
    let boxed = b.node(
        F0,
        entry,
        NodeOp::F64ToTagged,
        &[sum],
        Rep::Tagged,
        call + 1,
    );
    let at_rest = b.state(F0, call + 1, &[boxed], frame_slots(n_slots), &[]);
    b.exit(F0, entry, call + 1, at_rest, ExitReason::Unsupported);
    b.g
}

#[test]
fn a_parent_recipe_reads_registers_spill_slots_and_constants() {
    let mut engine = fixture(FIXTURES);
    let wide = chunk_of(&mut engine, "wide");
    let code = install(
        &mut engine,
        "wide",
        &["inner"],
        &spilled(&wide),
        Options::default(),
    );
    let table = super::exit::table_of(&code).expect("the tier's code");
    let recipe = &table.frames[0];
    let sources: Vec<super::exit::Source> = recipe
        .slots
        .iter()
        .map(|&(_, source)| source)
        .chain([recipe.callee, recipe.receiver])
        .collect();
    use super::exit::Source;
    assert!(
        sources.iter().any(|s| matches!(s, Source::Spill { .. }))
            && sources.iter().any(|s| matches!(s, Source::Gp { .. }))
            && sources.iter().any(|s| matches!(s, Source::Const(_))),
        "{sources:?}"
    );
    exits();
    for call in [
        "wide(inner, function () {}, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)",
        "wide(inner, function () {}, '2', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)",
        "wide(inner, function () { return 1; }, 0.5, 'a', {}, null, 4, 5, [6], 7, 8, true, 10)",
    ] {
        assert_eq!(eval(&mut engine, call), vm_answer(call), "{call}");
        assert_frames_released(&engine);
    }
    assert_eq!(
        eval(
            &mut engine,
            "wide(inner, function () {}, '2', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)"
        ),
        "2212345678910"
    );
    assert_eq!(exits().1.len(), 2, "inner's check twice");
}

#[test]
fn a_rebuilt_frame_past_the_stack_limit_throws_the_range_error_its_call_would() {
    // As tests.rs's segment-cap test: the stack positioned inside the check's margin above the red
    // zone, in a thread of its own, the live segment count at the cap, so that the exhaustion check
    // at a depth that is a multiple of 8 reports it. The rebuilt frame's entry counts a depth as a
    // call does, and its check is the call's: at depth 8 it throws, at 7 it does not.
    use crate::interpreter::{
        with_execution_stack_segments, EXECUTION_STACK_CHECK_MARGIN, EXECUTION_STACK_RED_ZONE,
        MAX_EXECUTION_STACK_SEGMENTS,
    };
    #[inline(never)]
    fn below_headroom(below: usize, f: &mut dyn FnMut()) {
        let mut frame = [0u8; 1024];
        std::hint::black_box(&mut frame);
        let remaining = stacker::remaining_stack().expect("a thread stack of known bounds");
        if remaining < below {
            f();
        } else {
            below_headroom(below, f);
        }
        std::hint::black_box(&frame);
    }
    std::thread::Builder::new()
        .name(String::from("trust-js-rebuild-depth"))
        .stack_size(6 * 1024 * 1024)
        .spawn(|| {
            let mut engine = fixture(FIXTURES);
            let (outer, inner) = (
                chunk_of(&mut engine, "outer"),
                chunk_of(&mut engine, "inner"),
            );
            install(
                &mut engine,
                "outer",
                &["inner"],
                &outer_inner(&outer, &inner, None),
                Options::default(),
            );
            let call = "outer(inner, 1, '3')";
            assert_eq!(eval(&mut engine, call), "26");
            // Called from Rust inside the region, which parses and compiles nothing there.
            eval(
                &mut engine,
                "var three = '3'; function isRange(e) { return e instanceof RangeError && \
                 e.message === 'Maximum call stack size exceeded'; }",
            );
            let global = |engine: &mut Engine, name: &str| {
                let env = engine.interp.global_env;
                engine.interp.get_var(name, &env).ok().expect("a global")
            };
            let outer_fn = global(&mut engine, "outer");
            let args = [
                global(&mut engine, "inner"),
                Value::Num(1.0),
                global(&mut engine, "three"),
            ];
            let mut answers = Vec::new();
            below_headroom(
                EXECUTION_STACK_RED_ZONE + EXECUTION_STACK_CHECK_MARGIN / 2,
                &mut || {
                    with_execution_stack_segments(MAX_EXECUTION_STACK_SEGMENTS, || {
                        // The call to outer at depth + 1, its callee's rebuilt frame at depth + 2.
                        for depth in [5, 6] {
                            exits();
                            engine.interp.depth = depth;
                            let result = engine.interp.call(outer_fn, Value::Undefined, &args);
                            engine.interp.depth = 0;
                            answers.push((result, exits().1));
                        }
                    })
                },
            );
            let [(at_7, exits_7), (at_8, exits_8)] = &answers[..] else {
                unreachable!()
            };
            assert_eq!(
                at_7.as_ref().ok().map(|value| value.bits()),
                Some(Value::Num(26.0).bits()),
                "the rebuild at depth 7 is no check's"
            );
            let Err(crate::interpreter::Abrupt::Throw(error)) = at_8 else {
                panic!("the rebuild at depth 8 throws");
            };
            let is_range = global(&mut engine, "isRange");
            let range = engine.interp.call(is_range, Value::Undefined, &[*error]);
            assert_eq!(
                range.ok().map(|value| value.bits()),
                Some(Value::Bool(true).bits()),
                "the RangeError a call past the limit throws"
            );
            for inlined in [exits_7, exits_8] {
                assert_eq!(inlined, &vec![(1, 2, ExitReason::NotNumber)]);
            }
            assert_frames_released(&engine);
            assert_eq!(eval(&mut engine, call), "26");
        })
        .expect("spawn the rebuild-depth test")
        .join()
        .expect("the rebuild-depth test completes");
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

#[test]
fn the_back_end_refuses_frames_its_exits_cannot_rebuild() {
    let mut engine = fixture(FIXTURES);
    let (outer, inner) = (
        chunk_of(&mut engine, "outer"),
        chunk_of(&mut engine, "inner"),
    );
    let graph = outer_inner(&outer, &inner, None);
    let fy = NodeIx(
        graph
            .nodes
            .iter()
            .rposition(|n| n.op == NodeOp::CheckNumber)
            .expect("inner's check") as u32,
    );
    let at_add = graph.node(fy).state.expect("a state");
    let cfg = Cfg::build(&outer).expect("settles");
    let layout = crate::value::jit_layout(&engine.interp.object_proto);
    let ilayout = crate::interpreter::interp_layout(&mut engine.interp);
    let compile = |graph: &Graph, callees: &[(Rc<Chunk>, Cfg)]| {
        super::a64::compile_frames_with(
            &outer,
            graph,
            &cfg,
            callees,
            cfg.initialized_reads(outer.jit_ops(), true),
            &layout,
            &ilayout,
            Options::default(),
        )
        .err()
        .map(|refusal| refusal.reason)
    };
    let inlined = callee_chunks(&mut engine, &["inner"]);
    assert_eq!(compile(&graph, &inlined), None, "the graph itself compiles");
    let mut broken = graph.clone();
    broken.frames[0].call_pc = 2;
    assert_eq!(
        compile(&broken, &inlined),
        Some("an inlined frame no exit can rebuild")
    );
    assert_eq!(
        compile(&graph, &callee_chunks(&mut engine, &["rest"])),
        Some("an inlined frame no exit can rebuild")
    );
    let mut broken = graph.clone();
    broken.node_mut(fy).op = NodeOp::This;
    broken.node_mut(fy).inputs.clear();
    broken.node_mut(fy).rep = Rep::Tagged;
    assert_eq!(
        compile(&broken, &inlined),
        Some("an inlined frame's node that reads the compiled frame")
    );
    let mut broken = graph.clone();
    let at_call = broken
        .nodes
        .iter()
        .find_map(|n| (n.frame == FrameIx::COMPILED).then_some(n.state).flatten())
        .expect("the caller's check");
    broken.node_mut(fy).state = Some(at_call);
    assert_eq!(compile(&broken, &inlined), Some("a frame state's frame"));
    let mut broken = graph.clone();
    let mut slots = broken.states[at_add.index()].slots.to_vec();
    slots[1] = SlotSource::Frame;
    broken.states[at_add.index()].slots = slots.into();
    assert_eq!(compile(&broken, &inlined), Some("a frame state's slots"));
}

/// `outerP(f, x)` (LoadLocal 1, LoadLocal 0, LoadLocal 1 twice, Call(2), Add, Return) with inner's
/// frame for f(x, x): the caller's operand below the call (x) is in its parent recipe's stack, and
/// the caller resumes with it under the call's result, its operands gone.
fn below_the_call(outer: &Chunk) -> Graph {
    assert!(
        matches!(
            outer.jit_ops(),
            [
                Op::LoadLocal(1),
                Op::LoadLocal(0),
                Op::LoadLocal(1),
                Op::LoadLocal(1),
                Op::Call(2, _),
                Op::Add,
                Op::Return,
                ..
            ]
        ),
        "{}",
        ops(outer)
    );
    let mut b = Build::new(2);
    let entry = b.block(0);
    let f = b.node(F0, entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 1);
    let x = b.node(F0, entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 0);
    let undefined = b.undefined(entry);
    let after = b.state(F0, 5, &[x], frame_slots(2), &[]);
    let f1 = b.open(0, after, 4, f, undefined);
    let at_add = b.state(f1, 2, &[x, x], values(&[x, x]), &[]);
    let fx = b.exiting(
        f1,
        entry,
        NodeOp::CheckNumber,
        &[x],
        Rep::F64,
        2,
        at_add,
        ExitReason::NotNumber,
    );
    let sum = b.node(f1, entry, NodeOp::F64Add, &[fx, fx], Rep::F64, 2);
    let total = b.node(F0, entry, NodeOp::F64Add, &[fx, sum], Rep::F64, 5);
    let boxed = b.node(F0, entry, NodeOp::F64ToTagged, &[total], Rep::Tagged, 5);
    b.node(F0, entry, NodeOp::Return, &[boxed], Rep::None, 6);
    b.g
}

#[test]
fn a_callers_operands_below_its_call_stay_under_the_result() {
    let mut engine = fixture(FIXTURES);
    let outer = chunk_of(&mut engine, "outerP");
    let code = install(
        &mut engine,
        "outerP",
        &["inner"],
        &below_the_call(&outer),
        Options::default(),
    );
    let table = super::exit::table_of(&code).expect("the tier's code");
    assert_eq!(table.frames[0].stack.len(), 1, "x in the parent's recipe");
    exits();
    for (call, expected) in [
        ("outerP(inner, 2)", "6"),
        ("outerP(inner, '2')", "222"),
        ("outerP(inner, true)", "3"),
    ] {
        assert_eq!(eval(&mut engine, call), expected, "{call}");
        assert_eq!(vm_answer(call), expected, "{call} in the VM");
        assert_frames_released(&engine);
    }
    assert_eq!(exits().1.len(), 2);
}

#[test]
fn a_plain_calls_frame_binds_its_receiver_as_the_call_would() {
    let mut engine = fixture(FIXTURES);
    let (outer, kind) = (
        chunk_of(&mut engine, "outerW"),
        chunk_of(&mut engine, "kind"),
    );
    install(
        &mut engine,
        "outerW",
        &["kind"],
        &at_callee_entry(&outer, &kind, &[]),
        Options::default(),
    );
    exits();
    // A sloppy callee's undefined receiver binds the global object.
    assert_eq!(eval(&mut engine, "outerW(kind, 1)"), "object");
    assert_eq!(vm_answer("outerW(kind, 1)"), "object");
    assert_eq!(exits().1.len(), 1);
    assert_frames_released(&engine);
}

#[test]
fn an_interruption_in_the_inner_run_passes_every_frame() {
    let mut engine = fixture(FIXTURES);
    let (guard, spin) = (
        chunk_of(&mut engine, "guardS"),
        chunk_of(&mut engine, "spin"),
    );
    let push = pc_of(&guard, |op| matches!(op, Op::PushHandler(_)));
    let depth = depth_at(&guard, push) as u32;
    install(
        &mut engine,
        "guardS",
        &["spin"],
        &at_callee_entry(&guard, &spin, &[(push, depth)]),
        Options::default(),
    );
    exits();
    let landed = interrupted(&mut engine, "guardS(spin, 1)", |(_, inlined)| {
        !inlined.is_empty()
    });
    assert_eq!(landed.1, vec![(1, 0, ExitReason::NeverRan)]);
}

#[test]
fn an_exit_through_an_inlined_frame_counts_on_the_code_and_flags_the_callees_first_stage() {
    // The fixtures in the VM, then the tier on with an exit limit of 1, so that the exit's
    // discard keys its site in the exit profile (D12: one count and one limit per code).
    let mut engine = fixture(FIXTURES);
    super::TestSwitches {
        on: true,
        eager: false,
        exit_every: 0,
        asynchronous: false,
        exit_limit: 1,
        ..Default::default()
    }
    .set();
    let (outer, inner) = (
        chunk_of(&mut engine, "outer"),
        chunk_of(&mut engine, "inner"),
    );
    let code = install(
        &mut engine,
        "outer",
        &["inner"],
        &outer_inner(&outer, &inner, None),
        Options::default(),
    );
    exits();
    assert_eq!(eval(&mut engine, "outer(inner, 1, '3')"), "26");
    assert_eq!(exits().1, vec![(1, 2, ExitReason::NotNumber)]);
    let table = super::exit::table_of(&code).expect("the code is retired, not freed");
    assert!(table.discarded.get(), "the code's one limit");
    assert_eq!(table.taken.get(), 1);
    let sites = |chunk: &Chunk, pc: u32| {
        let mut sites = Vec::new();
        if let Some(profile) = super::profile::ChunkProfile::get(chunk) {
            profile.sites_at(pc, |reason, inlined, _| sites.push((reason, inlined)));
        }
        sites
    };
    assert_eq!(
        sites(&inner, 2),
        vec![(ExitReason::NotNumber as u8, true)],
        "inner's first stage, at its Add, inlined"
    );
    assert_eq!(
        sites(&outer, 2),
        vec![],
        "none on the compiled function's pc"
    );
    assert_frames_released(&engine);
    super::pin_tier_off_for_test();
}

/// `guardS(f, x)` (PushHandler, LoadLocal 0-1, Call(1), ..., a catch pad) with `spin2(n)` (`while
/// (n) {}`: its header's test, then the backward Jump) in its frame, built as the loop it is for a
/// truthy n: the loop in the inlined frame, its header's poll there, whose throw a handler of the
/// caller's frame takes, so that the poll's throw exits.
fn poll_in_callee(guard: &Chunk, spin: &Chunk, handlers: &[(u32, u32)]) -> Graph {
    let call = pc_of(guard, is_call);
    assert!(
        matches!(
            guard.jit_ops()[call as usize - 2..=call as usize],
            [Op::LoadLocal(0), Op::LoadLocal(1), Op::Call(1, _)]
        ),
        "{}",
        ops(guard)
    );
    let back = pc_of(spin, |op| matches!(op, Op::Jump(_)));
    let Op::Jump(header_pc) = spin.jit_ops()[back as usize] else {
        unreachable!()
    };
    assert!(header_pc <= back, "a backward jump\n{}", ops(spin));
    let (_, n_slots) = guard.jit_frame();
    let mut b = Build::new(n_slots);
    let entry = b.block(0);
    let f = b.node(F0, entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let x = b.node(F0, entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let undefined = b.undefined(entry);
    let after = b.state(F0, call + 1, &[], frame_slots(n_slots), handlers);
    let f1 = b.open(0, after, call, f, undefined);
    let header = b.g.add_block(header_pc, true);
    b.node(F0, entry, NodeOp::Jump(header), &[], Rep::None, call);
    b.edge(entry, header);
    let at_header = b.state(f1, header_pc, &[], entry_slots(spin, &[x], undefined), &[]);
    b.exiting(
        f1,
        header,
        NodeOp::Poll,
        &[],
        Rep::None,
        header_pc,
        at_header,
        ExitReason::Throw,
    );
    b.node(f1, header, NodeOp::Jump(header), &[], Rep::None, back);
    b.edge(header, header);
    b.g
}

#[test]
fn a_polls_throw_in_an_inlined_frame_exits_where_a_callers_handler_takes_it() {
    let mut engine = fixture(FIXTURES);
    let (guard, spin) = (
        chunk_of(&mut engine, "guardS"),
        chunk_of(&mut engine, "spin2"),
    );
    let push = pc_of(&guard, |op| matches!(op, Op::PushHandler(_)));
    let depth = depth_at(&guard, push) as u32;
    let graph = poll_in_callee(&guard, &spin, &[(push, depth)]);
    let header = graph.blocks[1].first_pc;
    install(
        &mut engine,
        "guardS",
        &["spin2"],
        &graph,
        Options::default(),
    );
    exits();
    let landed = interrupted(&mut engine, "guardS(spin2, 1)", |(all, _)| !all.is_empty());
    // The poll's exit, whose interruption rebuilds no frame.
    assert_eq!(landed, (vec![(header, ExitReason::Throw)], vec![]));
}

/// The exits of `call` run interruptibly under deadlines of 100 ms, 1 s and then 5 s, until one
/// lands where `landed` sees it (inside the code under test: under GC stress, a short deadline
/// can pass in the collections before the call reaches it); every run interrupted, never a
/// throw the caller's catch takes, and every frame it rebuilt released. The outcome alone cannot
/// tell a catch that ran (the host's microtask drain after the script reports the deadline,
/// still passed, as an interruption whatever the script's completion), so `guardS`'s catch
/// counts its runs in `caughtS`, which must stay 0.
fn interrupted(
    engine: &mut Engine,
    call: &str,
    landed: impl Fn(&(Vec<(u32, ExitReason)>, Vec<(usize, u32, ExitReason)>)) -> bool,
) -> (Vec<(u32, ExitReason)>, Vec<(usize, u32, ExitReason)>) {
    let interrupt = engine.interrupt_handle();
    for millis in [100, 1000, 5000] {
        exits();
        interrupt.set_deadline(Some(
            std::time::Instant::now() + std::time::Duration::from_millis(millis),
        ));
        let outcome = engine.eval_interruptible(call, false).unwrap();
        interrupt.set_deadline(None);
        assert!(
            matches!(outcome, crate::ExecutionOutcome::Interrupted { .. }),
            "the interruption is no throw the caller's catch takes"
        );
        assert_eq!(
            eval(engine, "caughtS"),
            "0",
            "the caller's catch never ran: an interruption passes every frame"
        );
        assert_frames_released(engine);
        let exits = exits();
        if landed(&exits) {
            return exits;
        }
    }
    panic!("no interruption of {call} landed in its code");
}

/// `outerT(f, a, i, c, q0, ..., q9)` (LoadLocal 0-2, Call(2), then the sum) with `elem(a, i)`
/// (`a[i]`) in its frame, read by the TypedElement stub (the snapshot knows no kind): the ten
/// summands live past the stub's call (the caller's exit after the call reads them), so that they
/// take the registers it leaves, and `c`, which the parent's recipe writes back to its slot and
/// which nothing reads after the stub's node's exit, takes one the stub clobbers (x11-x15,
/// `regalloc::clobbers`): a declined read exits the callee's frame after the call, the parent
/// rebuilt from that register, which the code must have saved (stage 3 step 0b's
/// `save_live_clobbered`, its exit's reads including the frame's chain).
fn clobbered_chain(outer: &Chunk, elem: &Chunk) -> Graph {
    let call = pc_of(outer, is_call);
    assert!(
        matches!(outer.jit_ops()[call as usize], Op::Call(2, _)),
        "{}",
        ops(outer)
    );
    let get = pc_of(elem, |op| {
        !matches!(op, Op::LoadLocal(_) | Op::Return | Op::ReturnUndef)
    });
    assert!(
        matches!(elem.jit_ops()[..get as usize], [Op::LoadLocal(1)]) && depth_at(elem, get) == 1,
        "{}",
        ops(elem)
    );
    let (_, n_slots) = outer.jit_frame();
    let mut b = Build::new(n_slots);
    let entry = b.block(0);
    // The summands first, then c, so that c takes the next register free: a clobbered one.
    let summands: Vec<NodeIx> = (0..10)
        .map(|k| b.node(F0, entry, NodeOp::EntrySlot(4 + k), &[], Rep::Tagged, 0))
        .collect();
    let c = b.node(F0, entry, NodeOp::EntrySlot(3), &[], Rep::Tagged, 0);
    let f = b.node(F0, entry, NodeOp::EntrySlot(0), &[], Rep::Tagged, 0);
    let a = b.node(F0, entry, NodeOp::EntrySlot(1), &[], Rep::Tagged, 1);
    let i = b.node(F0, entry, NodeOp::EntrySlot(2), &[], Rep::Tagged, 2);
    let undefined = b.undefined(entry);
    let mut parent_slots = frame_slots(n_slots);
    parent_slots[3] = SlotSource::Value(c);
    let after = b.state(F0, call + 1, &[], parent_slots, &[]);
    let f1 = b.open(0, after, call, f, undefined);
    let at_get = b.state(f1, get, &[i], values(&[a, i]), &[]);
    let check = |b: &mut Build, op: NodeOp, inputs: &[NodeIx], rep: Rep, reason: ExitReason| {
        b.exiting(f1, entry, op, inputs, rep, get, at_get, reason)
    };
    let cell = check(
        &mut b,
        NodeOp::CheckObject,
        &[a],
        Rep::Cell,
        ExitReason::NotObject,
    );
    check(
        &mut b,
        NodeOp::CheckElements(ElemKind::Typed(None)),
        &[cell],
        Rep::None,
        ExitReason::WrongElements,
    );
    let number = check(
        &mut b,
        NodeOp::CheckNumber,
        &[i],
        Rep::F64,
        ExitReason::NotNumber,
    );
    let index = check(
        &mut b,
        NodeOp::CheckIndex,
        &[number],
        Rep::I32,
        ExitReason::NotInt32,
    );
    let element = check(
        &mut b,
        NodeOp::TypedElement { write: false },
        &[cell, index],
        Rep::Tagged,
        ExitReason::OutOfBounds,
    );
    // The caller's sum in the VM: its summands from their slots, which the exit writes.
    let mut slots = frame_slots(n_slots);
    for (k, &summand) in summands.iter().enumerate() {
        slots[4 + k] = SlotSource::Value(summand);
    }
    let at_sum = b.state(F0, call + 1, &[element], slots, &[]);
    b.exit(F0, entry, call + 1, at_sum, ExitReason::Unsupported);
    b.g
}

#[test]
fn a_stub_call_in_an_inlined_frame_keeps_the_chain_its_exit_reads() {
    let mut engine = fixture(FIXTURES);
    let (outer, elem) = (
        chunk_of(&mut engine, "outerT"),
        chunk_of(&mut engine, "elem"),
    );
    let code = install(
        &mut engine,
        "outerT",
        &["elem"],
        &clobbered_chain(&outer, &elem),
        Options::default(),
    );
    let table = super::exit::table_of(&code).expect("the tier's code");
    let clobbered = |source: super::exit::Source| {
        matches!(source, super::exit::Source::Gp { reg: 11..=15, .. })
    };
    assert!(
        table.frames[0]
            .slots
            .iter()
            .any(|&(slot, source)| slot == 3 && clobbered(source)),
        "c's home is a register the stub clobbers: {:?}",
        table.frames[0].slots
    );
    exits();
    let qs = "'q', 'q', 'q', 'q', 'q', 'q', 'q', 'q', 'q', 'q'";
    for (index, expected) in [(1, "0cqqqqqqqqqq"), (5, "undefinedcqqqqqqqqqq")] {
        let call = format!("outerT(elem, new Uint8Array(2), {index}, 'c', {qs})");
        assert_eq!(eval(&mut engine, &call), expected, "{call}");
        assert_eq!(vm_answer(&call), expected, "{call} in the VM");
        assert_frames_released(&engine);
    }
    let get = pc_of(&elem, |op| {
        !matches!(op, Op::LoadLocal(_) | Op::Return | Op::ReturnUndef)
    });
    assert_eq!(exits().1, vec![(1, get, ExitReason::OutOfBounds)]);
}
