//! The graph inliner for callees without effects (P3 stage 3, step 2): the builder's graphs of
//! small callers whose plans' callees are leaves, made from the oracle's own output over chunks
//! and feedback the JIT tier warmed, and the inliner end to end under TRUST_JS_OPT_INLINE=graph,
//! its results against the tier off's.

use super::build::{Built, InlineNote, InlineOutcome};
use super::exit::TEST_INLINED_EXITS;
use super::ir::*;
use super::job::{InlineJob, JobMemory};
use super::oracle::{Held, Request};
use super::profile::ChunkProfile;
use super::snapshot::{Raw, Snapshot};
use super::view::FrozenChunk;
use super::TestSwitches;
use crate::bytecode::{Chunk, Op, Tier};
use crate::value::{ObjectKind, Value, ValueView};
use crate::{Completion, Engine};
use std::rc::Rc;

fn eval(engine: &mut Engine, source: &str) -> String {
    match engine.eval(source, false).expect("the program parses") {
        Completion::Value(value) => value,
        Completion::Throw { name, message } => format!("throw {name}: {message}"),
    }
}

/// An engine at the JIT tier that compiles every function at its first call, the tier off, that
/// ran `source`: the builder tests' feedback, template code's.
fn warmed(source: &str) -> Engine {
    TestSwitches::default().set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let out = eval(&mut engine, source);
    assert!(!out.starts_with("throw"), "the fixture runs: {out}");
    engine
}

/// `warmed` with the tier on (not eager): template code records its in-line element hits (rule
/// U), and no function reaches its first checkpoint in so short a run.
fn warmed_on(source: &str) -> Engine {
    TestSwitches {
        on: true,
        ..Default::default()
    }
    .set();
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    let out = eval(&mut engine, source);
    assert!(!out.starts_with("throw"), "the fixture runs: {out}");
    engine
}

/// The global user function `name`: its function and its [[Realm]]'s identity.
fn function(engine: &mut Engine, name: &str) -> (Rc<crate::ast::Function>, usize) {
    let env = engine.interp.global_env;
    let value = engine.interp.get_var(name, &env).ok().expect("a global");
    let ValueView::Obj(object) = value.view() else {
        panic!("{name} is not an object")
    };
    let ObjectKind::User(user) = &object.borrow().kind else {
        panic!("{name} is not a user function")
    };
    (Rc::clone(&user.func), user.realm)
}

fn first_stage(func: &crate::ast::Function) -> Rc<Chunk> {
    func.runtime()
        .code
        .get()
        .and_then(Option::clone)
        .expect("a compiled function")
}

fn chunk_of(engine: &mut Engine, name: &str) -> Rc<Chunk> {
    first_stage(&function(engine, name).0)
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

/// A compile of a global function's first stage under TRUST_JS_OPT_INLINE=graph as a job makes
/// it: the planner's plan, the oracle's preparation (the guards appended to the view's inline
/// targets, each callee's view and raw copy), and the snapshots decoded.
struct Compile {
    chunk: Rc<Chunk>,
    view: FrozenChunk,
    snapshot: Snapshot,
    job: Option<InlineJob>,
    held: Held,
    callee_snapshots: Vec<Snapshot>,
}

impl Compile {
    fn new(engine: &mut Engine, name: &str) -> Self {
        let (func, realm) = function(engine, name);
        let plan = crate::bytecode::plan_inlines(
            &first_stage(&func),
            &func,
            realm,
            &engine.interp.global_env,
        );
        Compile::with(engine, name, Request::Plan(plan))
    }

    fn with(engine: &mut Engine, name: &str, request: Request) -> Self {
        let chunk = chunk_of(engine, name);
        let mut view = chunk.freeze();
        let raw = Raw::take_in(&chunk, Some(&engine.interp));
        let snapshot = Snapshot::decode(&raw, &view);
        let memory = || -> JobMemory { Default::default() };
        let (job, held) =
            match super::oracle::prepare(&engine.interp, &chunk, &mut view, request, memory) {
                Some((job, held)) => (Some(job), held),
                None => (None, Held::default()),
            };
        let callee_snapshots = job.as_ref().map_or_else(Vec::new, |job| {
            job.callees
                .iter()
                .map(|(view, raw)| Snapshot::decode(raw, view))
                .collect()
        });
        Compile {
            chunk,
            view,
            snapshot,
            job,
            held,
            callee_snapshots,
        }
    }

    /// The graph, verified against every frame's chunk as a job verifies it.
    fn build(&self) -> Built {
        let callees: Vec<super::build::Callee<'_>> =
            self.job.as_ref().map_or_else(Vec::new, |job| {
                job.callees
                    .iter()
                    .zip(&self.callee_snapshots)
                    .map(|((view, _), snapshot)| (view, snapshot))
                    .collect()
            });
        let inlining = self.job.as_ref().map(|job| (&job.sites[..], &callees[..]));
        let built = super::build::build_inlined(&self.view, &self.snapshot, inlining)
            .unwrap_or_else(|refusal| panic!("refused: {refusal:?}\n{}", ops(&self.chunk)));
        let job_callees = self.job.as_ref().map_or(&[][..], |job| &job.callees[..]);
        let frames: Vec<super::verify::FrameChunk<'_>> = built
            .callees
            .iter()
            .map(|(chunk, cfg)| (&job_callees[*chunk as usize].0, cfg))
            .collect();
        if let Err(error) =
            super::verify::verify_frames(&built.graph, (&self.view, &built.cfg), &frames)
        {
            panic!("{error}\n{}", dump(&built.graph));
        }
        built
    }

    /// The callee chunk of the compile's frame `frame`.
    fn frame_chunk(&self, built: &Built, frame: FrameIx) -> Rc<Chunk> {
        let inlined = built.graph.inlined(frame).expect("an inlined frame");
        let (index, _) = built.callees[inlined.chunk as usize];
        self.held.chunks[index as usize].clone()
    }
}

fn dump(graph: &Graph) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    for (index, block) in graph.blocks.iter().enumerate() {
        let _ = writeln!(
            out,
            "b{index} (pc {}) preds {:?} succs {:?}",
            block.first_pc,
            block.preds.iter().map(|b| b.0).collect::<Vec<_>>(),
            block.succs.iter().map(|b| b.0).collect::<Vec<_>>()
        );
        for &ix in block.phis.iter().chain(&block.body) {
            let node = graph.node(ix);
            let inputs: Vec<u32> = node.inputs.iter().map(|input| input.0).collect();
            let _ = write!(
                out,
                "  n{} = {:?} {inputs:?} {:?} @{} f{}",
                ix.0, node.op, node.rep, node.pc, node.frame.0
            );
            if let Some(state) = node.state {
                let state = graph.state(state);
                let _ = write!(
                    out,
                    "  [exit {:?} to f{} pc {} slots {:?} stack {:?}]",
                    node.reason, state.frame.0, state.pc, state.slots, state.stack
                );
            }
            out.push('\n');
        }
    }
    out
}

fn nodes(graph: &Graph, is: impl Fn(&Node) -> bool) -> Vec<NodeIx> {
    (0..graph.nodes.len())
        .map(|k| NodeIx(k as u32))
        .filter(|&n| is(graph.node(n)))
        .collect()
}

fn outcomes(built: &Built) -> Vec<(u8, InlineOutcome)> {
    built
        .inlines
        .iter()
        .map(|note| (note.way, note.outcome.clone()))
        .collect()
}

/// Flag `reason` at `pc` of `chunk`'s exit profile, keyed `inlined`, with the byte `byte`.
fn flag(chunk: &Chunk, pc: u32, reason: ExitReason, inlined: bool, byte: u8) {
    ChunkProfile::of(chunk).add_site(pc, reason as u8, inlined, byte, false);
}

// ---- the builder ---------------------------------------------------------------------------

const LEAVES: &str = "
function sq(v) { return v * v; }
function c1(x) { return sq(x) + 1; }
function ab(v) { if (v > 0) return v; return 0 - v; }
function c2(x) { return ab(x) * 2; }
function two(a, b) { return b; }
function c3(x) { return two(x); }
function c4(x) { return two(x, x + 1, x + 2); }
function getx() { return this.x; }
function c5(o) { return o.gx(); }
function gety() { 'use strict'; return this.y; }
function c6(o) { 'use strict'; return o.gy(); }
function c17(f, x) { return f(x); }
var o5 = {x: 1, gx: getx}, o6 = {y: 2, gy: gety};
for (var k = 0; k < 30; k++) { c1(k); c2(k - 15); c3(k); c4(k); c5(o5); c6(o6); c17(sq, k); }
";

#[test]
fn a_leaf_callee_is_built_in_the_graph_in_a_frame_of_its_own() {
    let mut engine = warmed(LEAVES);
    let compile = Compile::new(&mut engine, "c1");
    let built = compile.build();
    let graph = &built.graph;
    let call = pc_of(&compile.chunk, is_call);
    assert_eq!(graph.frames.len(), 1, "{}", dump(graph));
    let frame = graph.frames[0].clone();
    assert_eq!(frame.call_pc, call);
    assert_eq!(outcomes(&built), vec![(0, InlineOutcome::Inlined)]);
    let sq = chunk_of(&mut engine, "sq");
    assert!(Rc::ptr_eq(&compile.frame_chunk(&built, FrameIx(1)), &sq));
    // The guard at the call, of the compiled frame, exiting with the call's operands.
    let guards = nodes(graph, |node| matches!(node.op, NodeOp::CheckCallee { .. }));
    let [guard] = guards[..] else {
        panic!("one guard\n{}", dump(graph))
    };
    let guard = graph.node(guard);
    assert_eq!((guard.pc, guard.frame), (call, FrameIx::COMPILED));
    assert_eq!(guard.reason, Some(ExitReason::WrongCallee));
    let depth = crate::jit_ir::Cfg::build(&compile.chunk)
        .unwrap()
        .stack_depth_at(call as usize)
        .unwrap();
    assert_eq!(graph.state(guard.state.unwrap()).stack.len(), depth);
    assert_eq!(
        guard.inputs[0], frame.callee,
        "the callee is the guard's word"
    );
    // The call's operation is no effect: sq's multiplication is a node of frame 1.
    assert!(nodes(graph, |node| matches!(node.op, NodeOp::Effect { .. })).is_empty());
    assert!(
        !nodes(graph, |node| node.frame == FrameIx(1)
            && matches!(node.op, NodeOp::F64Mul | NodeOp::I32Mul))
        .is_empty(),
        "{}",
        dump(graph)
    );
    // Its parent state: the caller's after the call, without the call's operands.
    let parent = graph.state(frame.parent);
    let (pops, _) = compile.chunk.jit_stack_effect(call as usize).unwrap();
    assert_eq!((parent.pc, parent.frame), (call + 1, FrameIx::COMPILED));
    assert_eq!(parent.stack.len(), depth - pops);
    // A free-name call passes undefined as its receiver.
    assert_eq!(
        graph.node(frame.receiver).op,
        NodeOp::ConstTagged(Value::Undefined.bits())
    );
}

#[test]
fn a_callees_returns_merge_at_the_calls_continuation() {
    let mut engine = warmed(LEAVES);
    let compile = Compile::new(&mut engine, "c2");
    let built = compile.build();
    let graph = &built.graph;
    let call = pc_of(&compile.chunk, is_call);
    // The continuation: a block of the compiled frame at the call's pc + 1 whose predecessors
    // are the callee's two returns, joined by a phi.
    let joins: Vec<&BlockData> = graph
        .blocks
        .iter()
        .filter(|block| {
            block.first_pc == call + 1
                && block.preds.len() == 2
                && block.preds.iter().all(|&pred| {
                    let last = *graph.block(pred).body.last().unwrap();
                    graph.node(last).frame == FrameIx(1)
                        && matches!(graph.node(last).op, NodeOp::Jump(_))
                })
        })
        .collect();
    assert_eq!(joins.len(), 1, "{}", dump(graph));
    assert_eq!(joins[0].phis.len(), 1, "the returned values' phi");
    assert_eq!(graph.node(joins[0].phis[0]).frame, FrameIx::COMPILED);
}

#[test]
fn a_missing_argument_is_undefined_and_extras_are_dropped() {
    let mut engine = warmed(LEAVES);
    // two(x): b is undefined, which c3 returns.
    let built = Compile::new(&mut engine, "c3").build();
    let graph = &built.graph;
    assert_eq!(graph.frames.len(), 1);
    let returns = nodes(graph, |node| node.op == NodeOp::Return);
    let [ret] = returns[..] else { panic!() };
    assert_eq!(
        graph.node(graph.node(ret).inputs[0]).op,
        NodeOp::ConstTagged(Value::Undefined.bits()),
        "{}",
        dump(graph)
    );
    // two(x, x + 1, x + 2): b is the second argument; the third is in no state of frame 1.
    let built = Compile::new(&mut engine, "c4").build();
    let graph = &built.graph;
    let returns = nodes(graph, |node| node.op == NodeOp::Return);
    let [ret] = returns[..] else { panic!() };
    // The second argument, as the call's operands held it (boxed for the return).
    let returned = graph.node(ret).inputs[0];
    let guards = nodes(graph, |node| matches!(node.op, NodeOp::CheckCallee { .. }));
    let guard_state = graph.state(graph.node(guards[0]).state.unwrap());
    let len = guard_state.stack.len();
    let (second, third) = (guard_state.stack[len - 2], guard_state.stack[len - 1]);
    let word = |n: NodeIx| match graph.node(n).op {
        NodeOp::I32ToTagged | NodeOp::F64ToTagged => graph.node(n).inputs[0],
        _ => n,
    };
    assert_eq!(word(returned), word(second), "{}", dump(graph));
    for state in graph
        .states
        .iter()
        .filter(|state| state.frame == FrameIx(1))
    {
        assert_eq!(state.slots.len(), 2, "two's frame: its two formals");
        assert!(!state.slots.contains(&SlotSource::Value(third)));
    }
}

#[test]
fn a_sloppy_callee_that_reads_this_has_its_receiver_checked_a_strict_ones_is_not() {
    let mut engine = warmed(LEAVES);
    for (name, inputs) in [("c5", 2), ("c6", 1)] {
        let compile = Compile::new(&mut engine, name);
        let built = compile.build();
        let graph = &built.graph;
        let guards = nodes(graph, |node| matches!(node.op, NodeOp::CheckCallee { .. }));
        let [guard] = guards[..] else {
            panic!("{name}\n{}", dump(graph))
        };
        assert_eq!(graph.node(guard).inputs.len(), inputs, "{name}");
        // `this` in the callee is the receiver the call passes: its field's load takes the
        // receiver's cell, the caller's shape check of it (the method's load proved the shape,
        // a fact the callee's builder was handed).
        let frame = &graph.frames[0];
        let loads = nodes(graph, |node| {
            node.frame == FrameIx(1) && matches!(node.op, NodeOp::LoadField { .. })
        });
        assert!(
            loads.iter().any(|&load| {
                let cell = graph.node(graph.node(load).inputs[0]);
                matches!(cell.op, NodeOp::CheckShape(_)) && cell.inputs[0] == frame.receiver
            }),
            "{name}: the callee's `this` is its receiver\n{}",
            dump(graph)
        );
    }
}

const WAYS: &str = "
function ma() { return 1; }
function mb() { return 2; }
var oa = {m: ma}, ob = {m: mb};
function c7(o) { return o.m(); }
for (var k = 0; k < 30; k++) { c7(oa); c7(ob); }
";

#[test]
fn a_flagged_guard_branches_to_its_next_way_and_the_last_to_the_call() {
    let mut engine = warmed(WAYS);
    let chunk = chunk_of(&mut engine, "c7");
    let call = pc_of(&chunk, is_call);
    let kinds = |built: &Built| -> (usize, usize, usize) {
        let graph = &built.graph;
        (
            nodes(graph, |node| matches!(node.op, NodeOp::CheckCallee { .. })).len(),
            nodes(graph, |node| {
                matches!(node.op, NodeOp::CalleeMatches { .. })
            })
            .len(),
            nodes(
                graph,
                |node| matches!(node.op, NodeOp::Effect { pc, .. } if pc == call),
            )
            .len(),
        )
    };
    // Unflagged: the first way's guard exits, ending the chain (as a splice's).
    let built = Compile::new(&mut engine, "c7").build();
    assert_eq!(kinds(&built), (1, 0, 0), "{}", dump(&built.graph));
    assert_eq!(
        outcomes(&built),
        vec![(0, InlineOutcome::Inlined), (1, InlineOutcome::Unreached)]
    );
    assert_eq!(built.graph.frames.len(), 1);
    // The first guard exited (its byte 1): it branches, the second exits.
    flag(&chunk, call, ExitReason::WrongCallee, false, 1);
    let built = Compile::new(&mut engine, "c7").build();
    assert_eq!(kinds(&built), (1, 1, 0), "{}", dump(&built.graph));
    assert_eq!(built.graph.frames.len(), 2);
    // Both exited (byte 2): two branches, then the call itself.
    flag(&chunk, call, ExitReason::WrongCallee, false, 2);
    let built = Compile::new(&mut engine, "c7").build();
    assert_eq!(kinds(&built), (0, 2, 1), "{}", dump(&built.graph));
    assert_eq!(
        outcomes(&built),
        vec![(0, InlineOutcome::Inlined), (1, InlineOutcome::Inlined)]
    );
}

#[test]
fn a_flagged_one_way_site_keeps_its_call_after_the_mismatch() {
    let mut engine = warmed(LEAVES);
    let chunk = chunk_of(&mut engine, "c1");
    let call = pc_of(&chunk, is_call);
    flag(&chunk, call, ExitReason::WrongCallee, false, 1);
    let built = Compile::new(&mut engine, "c1").build();
    let graph = &built.graph;
    assert_eq!(
        nodes(graph, |node| matches!(
            node.op,
            NodeOp::CalleeMatches { .. }
        ))
        .len(),
        1
    );
    assert!(nodes(graph, |node| matches!(node.op, NodeOp::CheckCallee { .. })).is_empty());
    assert_eq!(
        nodes(
            graph,
            |node| matches!(node.op, NodeOp::Effect { pc, .. } if pc == call)
        )
        .len(),
        1,
        "{}",
        dump(graph)
    );
}

const LOOPS: &str = "
function tri(n) { var t = 0; for (var i = 0; i < n; i++) t = t + i; return t; }
function c8(n) { var s = 0; for (var j = 0; j < n; j++) s = s + tri(j); return s; }
function spin(t, n) { while (t < n) t = t + 1; return t; }
function c12(o) { return spin(o.v, 20); }
var o12 = {v: 3};
for (var k = 0; k < 30; k++) { c8(5); c12(o12); }
";

#[test]
fn a_callees_loop_inside_its_callers_loop_leaves_the_callers_values_held() {
    let mut engine = warmed(LOOPS);
    let built = Compile::new(&mut engine, "c8").build();
    let graph = &built.graph;
    assert_eq!(graph.frames.len(), 1, "{}", dump(graph));
    // The caller's loop has no effect now: no template, no reload of a slot from the frame.
    assert!(
        nodes(graph, |node| matches!(
            node.op,
            NodeOp::Effect { .. } | NodeOp::FrameSlot(_) | NodeOp::StoreSlot(_)
        ))
        .is_empty(),
        "{}",
        dump(graph)
    );
    // Two loop headers, the callee's of its frame, inside the caller's.
    let headers: Vec<FrameIx> = graph
        .blocks
        .iter()
        .filter(|block| block.loop_header)
        .map(|block| graph.node(block.body[0]).frame)
        .collect();
    assert_eq!(headers.len(), 2);
    assert!(headers.contains(&FrameIx::COMPILED) && headers.contains(&FrameIx(1)));
}

#[test]
fn a_callee_loops_entry_speculation_takes_the_callees_frame_and_flags() {
    let mut engine = warmed(LOOPS);
    let compile = Compile::new(&mut engine, "c12");
    let built = compile.build();
    let graph = &built.graph;
    // spin's loop header is its first operation: its entry edge is the caller's block after the
    // guard, where the selector checks t's word, an exit of spin's frame at its header's pc.
    let specs = nodes(graph, |node| {
        matches!(
            node.reason,
            Some(ExitReason::PhiNotNumber | ExitReason::PhiNotInt32)
        )
    });
    assert!(!specs.is_empty(), "{}", dump(graph));
    for &spec in &specs {
        let node = graph.node(spec);
        assert_eq!((node.frame, node.pc), (FrameIx(1), 0), "{}", dump(graph));
        assert_eq!(graph.state(node.state.unwrap()).frame, FrameIx(1));
        let first = graph.block(node.block).body[0];
        assert_eq!(
            graph.node(first).frame,
            FrameIx::COMPILED,
            "the entry edge's block is the caller's"
        );
    }
    // PhiNotNumber flagged at spin's header by spin's own code (not inlined: a reason that is
    // no IC's status reads from any key): the inlined loop's phi is tagged, with no
    // speculation, and the decline is listed in spin's frame.
    let spin = compile.frame_chunk(&built, FrameIx(1));
    flag(&spin, 0, ExitReason::PhiNotNumber, false, 0);
    let built = Compile::new(&mut engine, "c12").build();
    let specs = nodes(&built.graph, |node| {
        matches!(
            node.reason,
            Some(ExitReason::PhiNotNumber | ExitReason::PhiNotInt32)
        )
    });
    assert!(specs.is_empty(), "{}", dump(&built.graph));
    assert!(built
        .graph
        .declines
        .iter()
        .any(|decline| decline.reason == ExitReason::PhiNotNumber && decline.frame == FrameIx(1)));
}

const MISC: &str = "
function nr(o, f) { if (f) return o.p; return 1; }
function c9(o) { return nr(o, 0); }
function rd(o, a) { return o.k + (a + 1); }
function c10(o, a) { return rd(o, a); }
function cat(s) { return s + 'x'; }
function c11(s) { return cat(s); }
function rec(n) { return n > 0 ? rec(n - 1) : 0; }
var o9 = {p: 1}, o10 = {k: 2};
for (var k = 0; k < 30; k++) { c9(o9); c10(o10, k); c11('s'); rec(2); }
";

#[test]
fn a_callees_never_ran_block_exits_with_the_chained_state() {
    let mut engine = warmed(MISC);
    let built = Compile::new(&mut engine, "c9").build();
    let graph = &built.graph;
    assert_eq!(graph.frames.len(), 1, "{}", dump(graph));
    let exits = nodes(graph, |node| node.op == NodeOp::Exit(ExitReason::NeverRan));
    assert!(
        exits.iter().any(|&exit| {
            let node = graph.node(exit);
            node.frame == FrameIx(1) && graph.state(node.state.unwrap()).frame == FrameIx(1)
        }),
        "{}",
        dump(graph)
    );
}

#[test]
fn an_inlined_callee_reads_its_kind_bound_flags_from_its_inlined_keys_only() {
    // A WrongShape of rd's own compile (not inlined) is not the inlined read's: c10 inlines rd,
    // and rd compiled alone takes it.
    let mut engine = warmed(MISC);
    let rd = chunk_of(&mut engine, "rd");
    let load = pc_of(&rd, |op| {
        matches!(op, Op::GetProp(..) | Op::GetPropLocal(..))
    });
    flag(&rd, load, ExitReason::WrongShape, false, 1);
    let built = Compile::new(&mut engine, "c10").build();
    assert_eq!(outcomes(&built), vec![(0, InlineOutcome::Inlined)]);
    assert!(
        !built
            .graph
            .declines
            .iter()
            .any(|decline| decline.reason == ExitReason::WrongShape),
        "{:?}",
        built.graph.declines
    );
    let alone = Compile::with(&mut engine, "rd", Request::Plan(Default::default())).build();
    assert!(alone
        .graph
        .declines
        .iter()
        .any(|decline| decline.reason == ExitReason::WrongShape && decline.pc == load));
    // An inlined one is the inlined read's, its generic access an effect, which declines the way
    // (E0); rd compiled alone ignores it.
    let mut engine = warmed(MISC);
    let rd = chunk_of(&mut engine, "rd");
    flag(&rd, load, ExitReason::WrongShape, true, 1);
    let built = Compile::new(&mut engine, "c10").build();
    assert!(
        matches!(
            &outcomes(&built)[..],
            [(0, InlineOutcome::Declined { at: Some((pc, Op::GetPropLocal(..))), .. })] if *pc == load
        ),
        "{:?}",
        built.inlines
    );
    assert!(built.graph.frames.is_empty());
    let alone = Compile::with(&mut engine, "rd", Request::Plan(Default::default())).build();
    assert!(
        !alone
            .graph
            .declines
            .iter()
            .any(|decline| decline.reason == ExitReason::WrongShape),
        "{:?}",
        alone.graph.declines
    );
}

#[test]
fn a_callee_with_an_effect_is_declined_and_its_call_stays() {
    let mut engine = warmed(MISC);
    let compile = Compile::new(&mut engine, "c11");
    let built = compile.build();
    let cat = chunk_of(&mut engine, "cat");
    let add = pc_of(&cat, |op| *op == Op::Add);
    assert_eq!(
        outcomes(&built),
        vec![(
            0,
            InlineOutcome::Declined {
                reason: super::build::E0_EFFECT,
                at: Some((add, Op::Add)),
            }
        )]
    );
    let call = pc_of(&compile.chunk, is_call);
    assert!(built.graph.frames.is_empty());
    assert_eq!(
        nodes(
            &built.graph,
            |node| matches!(node.op, NodeOp::Effect { pc, .. } if pc == call)
        )
        .len(),
        1
    );
}

#[test]
fn recursion_is_refused_as_the_planner_refuses_it() {
    let mut engine = warmed(MISC);
    // The planner skips a self-recursive way: rec's plan is empty, its graph has no frame.
    let (func, realm) = function(&mut engine, "rec");
    let chunk = first_stage(&func);
    let plan = crate::bytecode::plan_inlines(&chunk, &func, realm, &engine.interp.global_env);
    assert!(plan.is_empty());
    // A plan naming the function itself at its own call (a leaf by hand) is refused by the oracle
    // as well.
    let env = engine.interp.global_env;
    let object = engine
        .interp
        .get_var("rec", &env)
        .ok()
        .unwrap()
        .as_obj()
        .unwrap();
    let call = pc_of(&chunk, is_call) as usize;
    let (Op::Call(_, cache) | Op::CallWithThis(_, cache)) = chunk.jit_ops()[call] else {
        unreachable!()
    };
    let way = crate::bytecode::InlineWay {
        f: func.clone(),
        obj: object,
        check_this: false,
        uses_this: false,
        leaf: true,
        free_names: Vec::new(),
        expected_env: 0,
        expected_genv: crate::interpreter::Env::as_ptr(&env) as usize,
        genv_pin: crate::interpreter::Env::downgrade(&env),
        nested: Default::default(),
    };
    let mut plan = crate::fasthash::FastMap::default();
    plan.insert(cache, crate::bytecode::InlinePlanEntry { ways: vec![way] });
    let compile = Compile::with(&mut engine, "rec", Request::Plan(plan));
    assert!(compile.job.is_none(), "no site inlined");
}

#[test]
fn an_inlined_callees_element_site_takes_the_hits_its_template_code_saw() {
    // rt's read ran only in template code, in line: its record is empty but for the last-seen
    // byte (rule U), which the oracle's copy of rt merges as the compiled chunk's copy does.
    let mut engine = warmed_on(
        "function rt(a, i) { return a[i]; }
         function c13(a) { return rt(a, 1) + 1; }
         var ta = new Uint8Array(4); for (var k = 0; k < 30; k++) c13(ta);",
    );
    let rt = chunk_of(&mut engine, "rt");
    let c13 = chunk_of(&mut engine, "c13");
    assert!(
        !super::in_place(&rt) && !super::in_place(&c13),
        "both still run template code"
    );
    let built = Compile::new(&mut engine, "c13").build();
    let graph = &built.graph;
    assert_eq!(graph.frames.len(), 1, "{}", dump(graph));
    assert!(
        !nodes(graph, |node| node.frame == FrameIx(1)
            && node.op == NodeOp::LoadTyped(TypedKind::U8))
        .is_empty(),
        "the inlined read is the one kind's, in line\n{}",
        dump(graph)
    );
    assert!(
        nodes(graph, |node| matches!(
            node.op,
            NodeOp::TypedElement { .. } | NodeOp::IsTypedArray
        ))
        .is_empty(),
        "no stub and no kind branch\n{}",
        dump(graph)
    );
}

const FACTS: &str = "
function cn(v) { return v + 1; }
function c14(o) { var a = o.v; var x = a * 3; var y = cn(a); return x + y + a; }
function c18(o, p) { var a = o.v; var x = a * 3; var y = cn(p); return x + y + a; }
function mk() { var z = 5; return function () { var w = 1; return function (v) { return v + z + w; }; }; }
var cz = mk()();
function c15(x) { return cz(x); }
var gk = 3;
function rg(v) { return v + gk; }
function c16(x) { return rg(x); }
var o14 = {v: 2};
for (var k = 0; k < 30; k++) { c14(o14); c15(k); c16(k); c18(o14, k); }
";

#[test]
fn value_facts_flow_into_a_callee_and_back() {
    let mut engine = warmed(FACTS);
    let built = Compile::new(&mut engine, "c14").build();
    let graph = &built.graph;
    assert_eq!(graph.frames.len(), 1, "{}", dump(graph));
    // `a`'s Number is checked once: before the call (a * 3), and neither the callee (v + 1) nor
    // the caller after it (+ a) checks it again.
    // o.v's load (cn's is the global object's).
    let loads = nodes(graph, |node| {
        matches!(node.op, NodeOp::LoadField { .. })
            && matches!(graph.node(node.inputs[0]).op, NodeOp::CheckShape(_))
    });
    let [load] = loads[..] else {
        panic!("{}", dump(graph))
    };
    let checks = nodes(graph, |node| {
        node.op == NodeOp::CheckNumber && node.inputs[0] == load
    });
    assert_eq!(checks.len(), 1, "{}", dump(graph));
    // A caller's value the callee never sees keeps its facts through the callee's build (its
    // prunes keep the callers' values): `a`'s Number, checked before cn(p), is not checked again
    // after it.
    let built = Compile::new(&mut engine, "c18").build();
    let graph = &built.graph;
    assert_eq!(graph.frames.len(), 1, "{}", dump(graph));
    let loads = nodes(graph, |node| {
        matches!(node.op, NodeOp::LoadField { .. })
            && matches!(graph.node(node.inputs[0]).op, NodeOp::CheckShape(_))
    });
    let [load] = loads[..] else {
        panic!("{}", dump(graph))
    };
    let checks = nodes(graph, |node| {
        node.op == NodeOp::CheckNumber && node.inputs[0] == load
    });
    assert_eq!(checks.len(), 1, "{}", dump(graph));
}

#[test]
fn a_callees_free_names_start_from_its_closures_environment_folded() {
    let mut engine = warmed(FACTS);
    // A closure variable one record up: its record folded from the closure's environment, no
    // parent walk.
    let built = Compile::new(&mut engine, "c15").build();
    let graph = &built.graph;
    assert_eq!(graph.frames.len(), 1, "{}", dump(graph));
    // w at hops 0 (the middle function's record), z at hops 1 (mk's): both read from folded
    // records.
    let reads = nodes(graph, |node| {
        node.frame == FrameIx(1) && matches!(node.op, NodeOp::LoadBinding { .. })
    });
    assert_eq!(reads.len(), 2, "{}", dump(graph));
    for read in reads {
        assert!(matches!(
            graph.node(graph.node(read).inputs[0]).op,
            NodeOp::ConstEnv(_)
        ));
    }
    assert!(nodes(graph, |node| node.frame == FrameIx(1)
        && matches!(node.op, NodeOp::EnvParent | NodeOp::Env))
    .is_empty());
    // A global read whose search reaches the realm's global environment from the closure's: no
    // check of the environment, as a splice's pinned search.
    let built = Compile::new(&mut engine, "c16").build();
    let graph = &built.graph;
    assert_eq!(graph.frames.len(), 1, "{}", dump(graph));
    assert!(
        !nodes(graph, |node| node.frame == FrameIx(1)
            && matches!(node.op, NodeOp::LoadField { .. }))
        .is_empty(),
        "gk read in line\n{}",
        dump(graph)
    );
    assert!(
        nodes(graph, |node| node.frame == FrameIx(1)
            && matches!(node.op, NodeOp::CheckEnvIs { .. } | NodeOp::Env))
        .is_empty(),
        "{}",
        dump(graph)
    );
}

#[test]
fn the_verifier_refuses_a_guard_unlike_its_call() {
    let mut engine = warmed(LEAVES);
    for (name, change) in [("c1", "argc"), ("c17", "this on a plain call")] {
        let mut compile = Compile::new(&mut engine, name);
        let built = compile.build();
        let target = built
            .graph
            .nodes
            .iter()
            .find_map(|node| match node.op {
                NodeOp::CheckCallee { target } => Some(target as usize),
                _ => None,
            })
            .unwrap();
        // c17's f(x) is a plain call (a local's), c1's sq(x) a free name's, with a receiver.
        let call = pc_of(&compile.chunk, is_call) as usize;
        match change {
            "argc" => compile.view.inline_targets[target].argc += 1,
            _ => {
                assert!(
                    matches!(compile.view.ops[call], Op::Call(..)),
                    "{}",
                    ops(&compile.chunk)
                );
                compile.view.inline_targets[target].check_this = true;
            }
        }
        let callees: Vec<super::build::Callee<'_>> = compile
            .job
            .as_ref()
            .unwrap()
            .callees
            .iter()
            .zip(&compile.callee_snapshots)
            .map(|((view, _), snapshot)| (view, snapshot))
            .collect();
        let frames: Vec<super::verify::FrameChunk<'_>> = built
            .callees
            .iter()
            .map(|(chunk, cfg)| (callees[*chunk as usize].0, cfg))
            .collect();
        let verdict =
            super::verify::verify_frames(&built.graph, (&compile.view, &built.cfg), &frames);
        assert!(
            verdict.as_ref().is_err_and(|error| error.contains("guard")),
            "{change}: {verdict:?}"
        );
    }
}

// ---- end to end ----------------------------------------------------------------------------

/// The tier on, eager, compiling on the main thread, graph inlining: an install happens in the
/// call whose checkpoint asks for it.
fn graph_switches(exit_every: u32) -> TestSwitches {
    TestSwitches {
        on: true,
        eager: true,
        exit_every,
        asynchronous: false,
        inline_graph: Some(true),
        ..Default::default()
    }
}

fn jit_engine() -> Engine {
    let mut engine = Engine::new();
    engine.set_tier(Tier::Jit);
    engine.set_tier_threshold(0);
    engine
}

fn inlines() -> Vec<InlineNote> {
    super::TEST_INLINES.with(|log| std::mem::take(&mut *log.borrow_mut()))
}

fn inlined_exits() -> Vec<(usize, u32, ExitReason)> {
    TEST_INLINED_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()))
}

/// Inlining shapes (the brief's difftest seeds): leaf callees at depths 1 to 3 (a non-leaf plan
/// splices), guards failing late, holes and int32 overflow in a callee, a callee's loop, a
/// throw, never-ran blocks, receivers and arguments.
const SHAPES: &[&str] = &[
    // Depth 1, a callee's int32 overflow late.
    "function add(a, b) { return a + b; }
     function f(n) { var t = 0; for (var i = 0; i < n; i++) t = add(t, i * 1000003); return t; }
     var r = 0; for (var k = 0; k < 400; k++) r = f(k % 50); r + ',' + f(5000)",
    // Depths 2 and 3: the outer plans call, and splice; the inner leaf's caller inlines it.
    "function l3(x) { return x + 1; }
     function l2(x) { return l3(x) * 2; }
     function l1(x) { return l2(x) - 1; }
     function g(n) { var t = 0; for (var i = 0; i < n; i++) t += l1(i); return t; }
     var r = 0; for (var k = 0; k < 400; k++) r += g(10); r",
    // A guard failing late: another callee at a warmed site.
    "function one() { return 1; } function other() { return 2; }
     var o = {m: one};
     function h(o) { return o.m() + 1; }
     var r = 0; for (var k = 0; k < 600; k++) { if (k == 500) o.m = other; r += h(o); } r",
    // Holes and reads past the length in a callee.
    "function at(a, i) { return a[i]; }
     function s(a) { var t = 0; for (var i = 0; i < 8; i++) { var v = at(a, i); t += v === undefined ? 100 : v; } return t; }
     var full = [1, 2, 3, 4, 5, 6, 7, 8], holey = [1, , 3, , 5]; var r = 0;
     for (var k = 0; k < 400; k++) r += s(full); r + ',' + s(holey)",
    // A callee's loop, a throw in a callee, a never-ran block, receivers.
    "function sum(a) { var t = 0; for (var i = 0; i < a.length; i++) t += a[i]; return t; }
     function chk(x) { if (x < 0) throw x; return x; }
     function px(o) { return o.k; }
     function u(a, o) { return sum(a) + chk(a.length) + px(o); }
     var r = 0; for (var k = 0; k < 400; k++) r += u([1, 2, 3], {k: k});
     var e; try { u([], {k: 1}); chk(-1); } catch (x) { e = x; }
     r + ',' + u([1.5, 2], {j: 1, k: 2}) + ',' + e",
    // Missing and extra arguments, a sloppy method's receiver, a strict function.
    "function two(a, b) { return b === undefined ? a : a + b; }
     function me() { return this.v * 2; }
     var ob = {v: 3, m: me};
     function w(x) { return two(x) + two(x, 1, 2) + ob.m(); }
     var r = 0; for (var k = 0; k < 400; k++) r += w(k); r + ',' + w(0.5)",
];

/// The program's completion with the tier off, and under TRUST_JS_OPT_INLINE=graph forcing an
/// exit every `exit_every` checks (0: none), with `stress` collections at every allocation.
fn both(source: &str, exit_every: u32, stress: bool) -> (String, String) {
    TestSwitches::default().set();
    let mut off = jit_engine();
    let expected = eval(&mut off, source);
    graph_switches(exit_every).set();
    let mut engine = jit_engine();
    if stress {
        let mut core = engine.interp.gc_heap.core.borrow_mut();
        core.set_stress_every(1);
        core.set_incremental(crate::gc_core::IncrementalMode::Off);
    }
    let got = eval(&mut engine, source);
    (expected, got)
}

#[test]
fn inlining_shapes_give_the_tier_offs_results() {
    let mut inlined = 0;
    for source in SHAPES {
        inlines();
        let (expected, got) = both(source, 0, false);
        assert_eq!(got, expected, "{source}");
        inlined += inlines()
            .iter()
            .filter(|note| note.outcome == InlineOutcome::Inlined)
            .count();
    }
    assert!(
        inlined >= SHAPES.len(),
        "the shapes inline ({inlined} ways)"
    );
}

#[test]
fn inlining_shapes_give_the_tier_offs_results_with_an_exit_forced_every_seventh_check() {
    for source in SHAPES {
        inlined_exits();
        let (expected, got) = both(source, 7, false);
        assert_eq!(got, expected, "{source}");
    }
}

#[test]
fn inlining_shapes_give_the_tier_offs_results_under_collections_at_every_allocation() {
    for source in &SHAPES[..2] {
        let (expected, got) = both(source, 7, true);
        assert_eq!(got, expected, "{source}");
    }
}

#[test]
fn an_inlined_callees_exit_rebuilds_both_frames_and_flags_the_callees_first_stage_inlined() {
    graph_switches(0).set();
    let mut engine = jit_engine();
    eval(
        &mut engine,
        "function px(o) { return o.k; }
         function cx(o) { return px(o) + 1; }
         var a = {k: 1}; for (var i = 0; i < 400; i++) cx(a);",
    );
    let px = chunk_of(&mut engine, "px");
    let cx = chunk_of(&mut engine, "cx");
    assert!(super::in_place(&cx), "cx compiled in place");
    assert!(
        inlines()
            .iter()
            .any(|note| note.outcome == InlineOutcome::Inlined),
        "px inlined"
    );
    let code = cx.jit.get().flatten().unwrap();
    let cx_code = code.mem_ptr() as usize;
    let limit = super::exit::table_of(&code).unwrap().limit;
    assert_eq!(limit, super::EXIT_LIMIT, "cx's limit, at its R of 0");
    drop(code);
    inlined_exits();
    super::TEST_DISCARDED.with(|discarded| discarded.borrow_mut().clear());
    // Another shape: px's shape check exits in its inlined frame, each exit rebuilding px's frame
    // and cx's, until the code's limit discards it.
    let out = eval(
        &mut engine,
        "var t = 0; for (var i = 0; i < 300; i++) t += cx({j: 0, k: i}); t",
    );
    assert_eq!(out, (0..300).map(|i| i + 1).sum::<i32>().to_string());
    let exits = inlined_exits();
    let discarded =
        super::TEST_DISCARDED.with(|discarded| std::mem::take(&mut *discarded.borrow_mut()));
    assert!(
        exits.len() as u32 >= limit && exits.iter().all(|&(depth, _, _)| depth == 1),
        "{} exits through px's frame (limit {limit})",
        exits.len()
    );
    let load = pc_of(&px, |op| {
        matches!(op, Op::GetProp(..) | Op::GetPropLocal(..))
    });
    assert!(exits
        .iter()
        .any(|&(_, pc, reason)| pc == load && reason == ExitReason::WrongShape));
    // The flag: on px's first stage at its pc, inlined.
    let mut keys = Vec::new();
    ChunkProfile::get(&px)
        .expect("px's record")
        .sites_at(load, |reason, inlined, _| keys.push((reason, inlined)));
    assert!(
        keys.contains(&(ExitReason::WrongShape as u8, true)),
        "{keys:?}"
    );
    // R and the limits are cx's: its code, compiled with cx's limit, was discarded at it, and
    // the discard counts on cx (px's own code, which runs once cx's is discarded, keeps its own).
    assert!(
        discarded.contains(&cx_code),
        "cx's code discarded ({discarded:?})"
    );
    assert!(ChunkProfile::get(&cx).map_or(0, ChunkProfile::discards) >= 1);
    // cx inlining px again takes the inlined flag (E0 declines the way).
    let again = Compile::new(&mut engine, "cx").build();
    assert!(
        matches!(&outcomes(&again)[..], [(0, InlineOutcome::Declined { .. })]),
        "{:?}",
        again.inlines
    );
}

#[test]
fn an_exit_through_an_inlined_frame_that_loops_takes_the_loop_limit_at_once() {
    super::TEST_LIMITS.with(|limits| limits.borrow_mut().clear());
    graph_switches(0).set();
    let mut engine = jit_engine();
    eval(
        &mut engine,
        "function lp(a) { var t = 0; for (var i = 0; i < a.length; i++) t = t + a[i]; return t; }
         function cl(a) { return lp(a) * 2; }
         var a = [1, 2, 3, 4]; for (var i = 0; i < 400; i++) cl(a);",
    );
    let cl = chunk_of(&mut engine, "cl");
    let lp = chunk_of(&mut engine, "lp");
    assert!(super::in_place(&cl));
    let code = cl.jit.get().flatten().unwrap();
    let table = super::exit::table_of(&code).unwrap();
    let loop_limit = table.loop_limit;
    assert!(loop_limit < table.limit);
    // A string element: lp's add exits in its loop, and its rebuilt frame runs the rest of the
    // loop in the VM, which tries its borrowed-frame code at the back edge.
    let out = eval(
        &mut engine,
        "var t = ''; for (var i = 0; i < 20; i++) t = cl([1, 'x', 2]); t",
    );
    assert_eq!(out, "NaN");
    assert!(
        lp.osr.get().is_some(),
        "lp's loop tried its borrowed-frame code"
    );
    let limits = super::TEST_LIMITS.with(|limits| std::mem::take(&mut *limits.borrow_mut()));
    assert!(
        limits
            .iter()
            .any(|&(limit, taken)| limit == super::Limit::InlinedLoop && taken == loop_limit),
        "{limits:?}"
    );
}

#[test]
fn error_stack_and_caller_from_an_exit_inside_a_callee_are_the_tier_offs() {
    let source = "
        var seen = '';
        var g = { get p() { seen = new Error('e').stack + '|' + (nr.caller && nr.caller.name); return 1; } };
        function nr(o, f) { if (f) return o.p; return 1; }
        function cs(o, f) { return nr(o, f) + 1; }
        for (var k = 0; k < 400; k++) cs(g, 0);
        cs(g, 1); seen";
    let (expected, got) = both(source, 0, false);
    assert!(
        expected.contains("nr") && expected.contains("cs"),
        "{expected}"
    );
    assert_eq!(got, expected);
}

#[test]
fn a_guards_exits_flag_its_call_and_the_reoptimized_code_branches_on_it() {
    graph_switches(0).set();
    let mut engine = jit_engine();
    eval(
        &mut engine,
        "function one() { return 1; } function other() { return 2; }
         var o = {m: one};
         function h(o) { return o.m() + 1; }
         for (var k = 0; k < 400; k++) h(o);",
    );
    let h = chunk_of(&mut engine, "h");
    assert!(super::in_place(&h));
    let limit = super::exit::table_of(&h.jit.get().flatten().unwrap())
        .unwrap()
        .limit;
    super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
    let out = eval(
        &mut engine,
        "o.m = other; var t = 0; for (var k = 0; k < 3000; k++) { if (k % 2) o.m = one; else o.m = other; t += h(o); } t",
    );
    assert_eq!(out, (3000 / 2 * 2 + 3000 / 2 * 3).to_string());
    let call = pc_of(&h, is_call);
    let wrong = super::exit::TEST_EXITS.with(|exits| {
        std::mem::take(&mut *exits.borrow_mut())
            .into_iter()
            .filter(|&(pc, reason)| pc == call && reason == ExitReason::WrongCallee)
            .count()
    });
    // The first code's limit, then the reoptimized code's guards branch: one's flagged (its
    // byte), other's in its chain.
    assert!(
        wrong as u32 <= limit + limit / 2,
        "{wrong} WrongCallee exits (limit {limit})"
    );
    let mut bytes = Vec::new();
    ChunkProfile::get(&h)
        .unwrap()
        .sites_at(call, |reason, inlined, byte| {
            bytes.push((reason, inlined, byte))
        });
    assert!(
        bytes.contains(&(ExitReason::WrongCallee as u8, false, 1)),
        "{bytes:?}"
    );
}

#[test]
fn a_function_the_tier_refuses_splices_at_its_next_checkpoint() {
    graph_switches(0).set();
    let mut engine = jit_engine();
    eval(
        &mut engine,
        "function lf(x) { return x + 1; }
         function rf(x) { var t = 0; try { t = lf(x); } finally { t = t + 1; } return t; }
         for (var i = 0; i < 2000; i++) rf(i);",
    );
    // rf's plan inlines lf, a leaf; the tier refuses rf (its `finally`), and its next checkpoint
    // splices, as a function the tier refuses keeps a second stage's splices.
    let (func, _) = function(&mut engine, "rf");
    let first = first_stage(&func);
    assert!(super::graph_refused(&first), "the graph compile refused");
    assert!(!super::in_place(&first));
    assert!(
        func.runtime().code2.get().and_then(Option::clone).is_some(),
        "a second stage with its splice"
    );
    assert_eq!(eval(&mut engine, "rf(5)"), "7");
}

#[test]
fn a_reoptimization_inlines_the_ways_its_first_compile_took() {
    // Richards' `t.run()`: one callee at the first compile, four afterwards. The reoptimization
    // inlines the first compile's way again (`ChunkProfile::inline_ways`), its guard flagged, and
    // the others take the generic call after it, as a second stage's reoptimization keeps its
    // splices: one discard, then no exit.
    graph_switches(0).set();
    let mut engine = jit_engine();
    eval(
        &mut engine,
        "function A() {} A.prototype.run = function () { return 1; };
         function B() {} B.prototype.run = function () { return 2; };
         function C() {} C.prototype.run = function () { return 3; };
         function D() {} D.prototype.run = function () { return 4; };
         function runTask(t) { return t.run(); }
         var tasks = [new A(), new B(), new C(), new D()];
         for (var k = 0; k < 1000; k++) runTask(tasks[0]);",
    );
    let chunk = chunk_of(&mut engine, "runTask");
    assert!(super::in_place(&chunk));
    assert_eq!(
        ChunkProfile::get(&chunk).map(|profile| profile.inline_ways().len()),
        Some(1)
    );
    inlines();
    super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
    let out = eval(
        &mut engine,
        "var u = 0; for (var k = 0; k < 2000; k++) u += runTask(tasks[k % 4]); u",
    );
    assert_eq!(out, "5000");
    let exits = super::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()));
    assert_eq!(
        exits.len(),
        super::EXIT_LIMIT as usize,
        "one discard's exits, then none"
    );
    let notes = inlines();
    assert_eq!(
        notes
            .iter()
            .map(|note| (note.way, note.outcome.clone()))
            .collect::<Vec<_>>(),
        vec![(0, InlineOutcome::Inlined)],
        "the reoptimization's one way, the first compile's"
    );
}

#[test]
fn a_sites_later_guard_leaves_its_own_byte_and_the_reoptimization_branches_there_too() {
    // Two ways at the first compile: the first's guard exits for the second's callee (its byte
    // 1); the reoptimization branches there and checks the second; a third callee then exits at
    // the second guard (its byte 2), and the next reoptimization branches at both, the generic
    // call after them, as a splice's guards converge one at a time.
    graph_switches(0).set();
    let mut engine = jit_engine();
    eval(
        &mut engine,
        "function A() {} A.prototype.m = function () { return 1; };
         function B() {} B.prototype.m = function () { return 2; };
         function C() {} C.prototype.m = function () { return 3; };
         function site(o) { return o.m(); }
         var a = new A(), b = new B(), c = new C();
         var t = 0; for (var k = 0; k < 3000; k++) t += site(k % 2 ? a : b);",
    );
    let chunk = chunk_of(&mut engine, "site");
    let call = pc_of(&chunk, is_call);
    let byte = |chunk: &Chunk| {
        let mut most = None;
        if let Some(profile) = ChunkProfile::get(chunk) {
            profile.sites_at(call, |reason, _, byte| {
                if reason == ExitReason::WrongCallee as u8 {
                    most = Some(byte);
                }
            });
        }
        most
    };
    assert_eq!(
        ChunkProfile::get(&chunk).map(|profile| profile.inline_ways().len()),
        Some(2)
    );
    assert_eq!(byte(&chunk), Some(1), "the first guard's exits");
    eval(
        &mut engine,
        "for (var k = 0; k < 6000; k++) t += site(k % 3 == 0 ? c : k % 3 == 1 ? a : b); t",
    );
    assert_eq!(
        byte(&chunk),
        Some(2),
        "the second guard's exits leave its byte"
    );
    super::exit::TEST_EXITS.with(|exits| exits.borrow_mut().clear());
    let out = eval(
        &mut engine,
        "var u = 0; for (var k = 0; k < 3000; k++) u += site(k % 3 == 0 ? c : k % 3 == 1 ? a : b); u",
    );
    assert_eq!(out, "6000");
    let exits = super::exit::TEST_EXITS.with(|exits| std::mem::take(&mut *exits.borrow_mut()));
    assert!(exits.is_empty(), "converged: {exits:?}");
}
