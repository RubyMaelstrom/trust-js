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
    assert!(nodes(graph, |node| matches!(
        node.op,
        NodeOp::Effect { .. } | NodeOp::Generic { .. }
    ))
    .is_empty());
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
                |node| matches!(node.op, NodeOp::Effect { pc, .. } | NodeOp::Generic { pc, .. } if pc == call),
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
            |node| matches!(node.op, NodeOp::Effect { pc, .. } | NodeOp::Generic { pc, .. } if pc == call)
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
            NodeOp::Effect { .. }
                | NodeOp::Generic { .. }
                | NodeOp::FrameSlot(_)
                | NodeOp::StoreSlot(_)
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
function spr(a) { return [...a].length; }
function c12(a) { return spr(a); }
var o9 = {p: 1}, o10 = {k: 2};
for (var k = 0; k < 30; k++) { c9(o9); c10(o10, k); c11('s'); rec(2); c12([k]); }
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
    // An inlined one is the inlined read's: its access is the generic node of the inlined frame
    // (P3 stage 3 step 3c; E0 declined the way before); rd compiled alone ignores it.
    let mut engine = warmed(MISC);
    let rd = chunk_of(&mut engine, "rd");
    flag(&rd, load, ExitReason::WrongShape, true, 1);
    let built = Compile::new(&mut engine, "c10").build();
    assert_eq!(outcomes(&built), vec![(0, InlineOutcome::Inlined)]);
    assert_eq!(
        nodes(&built.graph, |node| node.frame == FrameIx(1)
            && matches!(node.op, NodeOp::Generic { pc, .. } if pc == load))
        .len(),
        1,
        "{}",
        dump(&built.graph)
    );
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
fn a_callee_with_a_generic_operation_is_inlined_and_one_needing_the_whole_frame_declined() {
    // P3 stage 3 step 3c: cat's `s + 'x'` (no Numbers) is a generic node of its inlined frame,
    // and c11's call is gone; spr's array spread runs the VM on its frame's slots in place (a
    // canonical-frame effect), which an inlined frame has none of: E0 declines it, and c12's
    // call stays.
    let mut engine = warmed(MISC);
    let compile = Compile::new(&mut engine, "c11");
    let built = compile.build();
    let cat = chunk_of(&mut engine, "cat");
    let add = pc_of(&cat, |op| *op == Op::Add);
    assert_eq!(outcomes(&built), vec![(0, InlineOutcome::Inlined)]);
    let call = pc_of(&compile.chunk, is_call);
    assert_eq!(built.graph.frames.len(), 1);
    assert!(nodes(&built.graph, |node| node.frame.is_compiled()
        && matches!(node.op, NodeOp::Effect { pc, .. } | NodeOp::Generic { pc, .. } if pc == call))
    .is_empty());
    assert_eq!(
        nodes(&built.graph, |node| node.frame == FrameIx(1)
            && matches!(node.op, NodeOp::Generic { pc, .. } if pc == add))
        .len(),
        1,
        "{}",
        dump(&built.graph)
    );
    let compile = Compile::new(&mut engine, "c12");
    let built = compile.build();
    let spr = chunk_of(&mut engine, "spr");
    assert!(
        matches!(
            &outcomes(&built)[..],
            [(0, InlineOutcome::Declined { reason: super::build::E0_EFFECT, at: Some((pc, op)) })]
                if crate::bytecode::jit_runs_on_vm_copy(op) && spr.jit_ops()[*pc as usize] == *op
        ),
        "{:?}",
        built.inlines
    );
    let call = pc_of(&compile.chunk, is_call);
    assert!(built.graph.frames.is_empty());
    assert_eq!(
        nodes(
            &built.graph,
            |node| matches!(node.op, NodeOp::Effect { pc, .. } | NodeOp::Generic { pc, .. } if pc == call)
        )
        .len(),
        1
    );
}

#[test]
fn callees_whose_equalities_and_typeof_tests_lower_are_inlined() {
    // P3 stage 3 step L1: `x == null`, typeof tests and `x === 0` lower whatever the facts, an
    // equality whose record names one pair speculates on it (C's claims in Rules::lowers, which
    // E0 reads): no decline; a loose compare with a String constant still declines its callee.
    let mut engine = warmed(
        "function isn(x) { return x == null; }
         function isnum(x) { return typeof x === 'number'; }
         function isz(x) { return x === 0; }
         function eqn(a, b) { return a == b; }
         function lit(x) { return x == 'a'; }
         function c(x) { return isn(x) || isnum(x) || isz(x); }
         function d(a, b) { return eqn(a, b) ? 1 : 0; }
         function f(x) { return lit(x); }
         for (var k = 0; k < 30; k++) { c(k & 1 ? k : 's'); d(k, k + 1); f(k); }",
    );
    let eqn = chunk_of(&mut engine, "eqn");
    let pc = pc_of(&eqn, |op| *op == Op::EqEq) as usize;
    ChunkProfile::of(&eqn).see_equality(&eqn, pc, crate::feedback::EQ_CLASS_NUMBERS);
    let built = Compile::new(&mut engine, "c").build();
    assert_eq!(
        outcomes(&built),
        vec![
            (0, InlineOutcome::Inlined),
            (0, InlineOutcome::Inlined),
            (0, InlineOutcome::Inlined)
        ],
        "{}",
        dump(&built.graph)
    );
    let built = Compile::new(&mut engine, "d").build();
    assert_eq!(outcomes(&built), vec![(0, InlineOutcome::Inlined)]);
    assert!(
        built
            .graph
            .nodes
            .iter()
            .any(|node| !node.frame.is_compiled()
                && node.op == NodeOp::CheckNumber
                && node.reason == Some(ExitReason::NotNumber)),
        "the callee's speculation: {}",
        dump(&built.graph)
    );
    // A loose compare with a String constant lowers to nothing: its callee is inlined with the
    // compare as its frame's generic node (P3 stage 3 step 3c; E0 declined it before).
    let lit = chunk_of(&mut engine, "lit");
    let eq = pc_of(&lit, |op| *op == Op::EqEq);
    let built = Compile::new(&mut engine, "f").build();
    assert_eq!(outcomes(&built), vec![(0, InlineOutcome::Inlined)]);
    assert_eq!(
        nodes(&built.graph, |node| node.frame == FrameIx(1)
            && matches!(node.op, NodeOp::Generic { pc, .. } if pc == eq))
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
    // cx inlining px again: the flag, px's inlined key, would decline its one-shape check, but
    // its ways grew (the new shape filled one, at another slot), so the load is checked again,
    // dispatching on the two shapes in px's frame (P3 stage 3 step L2: `Rules::poly_ways`), as a
    // grown site's own check is (`stale_ways`).
    let again = Compile::new(&mut engine, "cx").build();
    assert!(
        matches!(&outcomes(&again)[..], [(0, InlineOutcome::Inlined)]),
        "{:?}",
        again.inlines
    );
    assert!(
        again
            .graph
            .nodes
            .iter()
            .any(|node| node.frame == FrameIx(1) && matches!(node.op, NodeOp::ShapeIs(_))),
        "px's load dispatches"
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

/// P3 stage 3 step 0c under TRUST_JS_OPT_INLINE=graph: typed arrays' facts in inlined callees'
/// loops and in callers' loops around inlined calls.
#[path = "typed_loop_inline_tests.rs"]
mod typed_loop;

/// P3 stage 3 step L2 under TRUST_JS_OPT_INLINE=graph: callees whose loads go through the chain.
#[path = "chain_inline_tests.rs"]
mod chain;

/// P3 stage 3 step L3 under TRUST_JS_OPT_INLINE=graph: callees that create, update and read a
/// String's length.
#[path = "create_inline_tests.rs"]
mod create;

// ---- P3 stage 3 step 3c: generic nodes in inlined frames --------------------------------------

/// The inlined frames' generic nodes the back end lowered since the last call, by operation.
fn inlined_generic() -> Vec<Op> {
    super::a64::TEST_INLINED_GENERIC.with(|ops| std::mem::take(&mut *ops.borrow_mut()))
}

/// `both` under `switches`' graph inlining (`graph_switches` with the poison switch where
/// `poison`), with `stress` collections at every allocation.
fn both_poisoned(source: &str, exit_every: u32, stress: bool, poison: bool) -> (String, String) {
    TestSwitches::default().set();
    let mut off = jit_engine();
    let expected = eval(&mut off, source);
    TestSwitches {
        poison,
        ..graph_switches(exit_every)
    }
    .set();
    let mut engine = jit_engine();
    if stress {
        let mut core = engine.interp.gc_heap.core.borrow_mut();
        core.set_stress_every(1);
        core.set_incremental(crate::gc_core::IncrementalMode::Off);
    }
    let got = eval(&mut engine, source);
    (expected, got)
}

/// Leaf callees whose operations the tier does not lower, each a kind E0 declined (the census's:
/// generic named loads through `this`, a local and the stack, stores and an update through
/// setters, element reads by name, instanceof, template strings, array literals, loose equality,
/// string concatenation, delete), each through accessors or megamorphic sites so that it stays
/// a generic node.
const GENERIC_KINDS: &str = "
var shapes = [{a: 1, x: 1}, {b: 1, x: 2}, {c: 1, x: 3}, {d: 1, x: 4}, {e: 1, x: 5}, {f: 1, x: 6}];
function gx(o) { return o.x; }
function M(v) { this.v = v; }
Object.defineProperty(M.prototype, 'w', {
  get: function () { return this.v + 1; },
  set: function (z) { this.v = z * 2; }
});
M.prototype.rw = function () { return this.w; };
M.prototype.ww = function (z) { this.w = z; };
function sw(o, z) { o.w = z; }
function at(o, k) { return o[k]; }
function isM(o) { return o instanceof M; }
function str(v) { return `${v}!`; }
function arr(a, b) { return [a, b]; }
function loc(o) { var q = o; return q.w; }
function sloc(o, z) { var q = o; q.w = z; }
function eqs(a, b) { return a == b; }
function cat(a, b) { return a + b; }
function upd(o) { o.w++; }
function del(o) { delete o.tmp; return o.v; }
function incs(s) { var q = s; q++; return q; }
function run(n) {
  var m = new M(1), s = 0, t = '';
  for (var i = 0; i < n; i++) {
    s += gx(shapes[i % 6]);
    s += m.rw(); m.ww(i); sw(m, i + 1);
    s += at(m, i & 1 ? 'v' : 'w');
    if (isM(m)) s++;
    t = str(i) + cat('a', i);
    s += arr(i, 2).length + loc(m);
    sloc(m, 3);
    if (eqs('1', i % 2)) s++;
    upd(m);
    m.tmp = i; s += del(m);
    s += incs(i & 1 ? '5' : '6');
  }
  return s + ':' + t + ':' + m.v;
}
var r = ''; for (var k = 0; k < 200; k++) r = run(20); r";

/// Each of an inlined frame's context reads, by a program whose compiled frame holds another
/// value there: `this` (the method's caller has a `w` of its own), the environment (the caller's
/// closure binds the name the callee reads from the global object's accessor), a named slot (the
/// caller's slot of the callee's local's number holds another object).
const CONTEXT_READS: &[&str] = &[
    "function M(v) { this.v = v; }
     Object.defineProperty(M.prototype, 'w', { get: function () { return this.v + 1; } });
     M.prototype.rw = function () { return this.w; };
     var holder = { v: 1000, sum: function (m, n) {
       var s = 0; for (var i = 0; i < n; i++) s += m.rw() + this.v + this.w; return s; } };
     Object.defineProperty(holder, 'w', { get: function () { return -1; } });
     var m = new M(2), r = 0; for (var k = 0; k < 300; k++) r += holder.sum(m, 10); r",
    "Object.defineProperty(globalThis, 'gz', { get: function () { return 3; }, configurable: true });
     function rg() { return gz; }
     function mk() { var gz = 5;
       return function inner(n) { var s = 0;
         for (var i = 0; i < n; i++) { s += rg() + gz; var f = function () { return gz; }; s += f(); }
         return s; }; }
     var g = mk(), r = 0; for (var k = 0; k < 300; k++) r += g(10); r",
    "function M(v) { this.v = v; }
     Object.defineProperty(M.prototype, 'w', { get: function () { return this.v + 1; } });
     function loc(o) { var q = o; return q.w; }
     function rl(n) { var a = new M(50), b = new M(7), s = 0;
       for (var i = 0; i < n; i++) s += loc(b) + a.w; return s; }
     var r = 0; for (var k = 0; k < 300; k++) r += rl(10); r",
];

/// Error.stack, a getter's `caller` and a thrown error's position from an inlined frame's node:
/// a getter its generic read calls (its frame's caller the inlined callee, which reports its
/// start), and a TypeError its generic element read raises (the callee's frame at the read, the
/// caller's at the call).
const POSITIONS: &str = "
function G() {}
var stk = '', cal, cal2;
Object.defineProperty(G.prototype, 'p', { get: function getter() {
  stk = new Error('e').stack; cal = getter.caller; return 1; } });
Object.defineProperty(G.prototype, 'q', { get: function getter2() { cal2 = getter2.caller; return 2; } });
var cal3, o2 = { toString: function ts() { cal3 = ts.caller; return 'x'; } };
function viaGet(o) { return o.p; }
function outer(o) { var s = 0;
  for (var i = 0; i < 20; i++) s += viaGet(o) + o.q + `${o2}`.length; return s; }
var g = new G(), r = 0; for (var k = 0; k < 300; k++) r += outer(g);
function ge(a, k) { return a[k]; }
function gouter(a) { var s = 0; for (var i = 0; i < 20; i++) s += ge(a, i & 1 ? 'x' : 'y'); return s; }
var ga = {x: 1, y: 2}; for (var k = 0; k < 300; k++) r += gouter(ga);
var err = ''; try { gouter(null); } catch (e) { err = e.stack.split('\\n').slice(0, 3).join('|'); }
r + '|' + stk.split('\\n').slice(0, 4).join('|') + '|' + (cal === viaGet) + (cal2 === outer) + (cal3 === outer) + '|' + err";

/// A throw from an inlined frame's node, caught by its caller's handler around the call and by
/// none (the throw leaves the compiled function).
const THROWS: &str = "
function G() {}
var calls = 0, msg = '';
Object.defineProperty(G.prototype, 'p', { get: function () { calls++;
  if (this.bad) throw new TypeError('bad ' + this.bad); return 1; } });
function viaGet(o) { return o.p; }
function caught(o) { var s = 0;
  for (var i = 0; i < 20; i++) { try { s += viaGet(o); } catch (e) { s += 1000; msg = msg || e.message; } }
  return s; }
function leaves(o) { var s = 0; for (var i = 0; i < 20; i++) s += viaGet(o); return s; }
var g = new G(), r = 0;
for (var k = 0; k < 300; k++) { r += caught(g); r += leaves(g); }
g.bad = 'x';
var e2 = ''; try { leaves(g); } catch (e) { e2 = e.message + ':' + e.stack.split('\\n').slice(0, 3).join('|'); }
r + ',' + caught(g) + ',' + e2 + ',' + calls + ',' + msg";

/// Collections while an inlined frame's helper runs, with objects live in both frames: the
/// getter allocates, the callee and the caller hold objects across it.
const COLLECTIONS: &str = "
function M(v) { this.v = v; }
Object.defineProperty(M.prototype, 'w', { get: function () {
  var junk = []; for (var j = 0; j < 8; j++) junk.push({j: j, s: 'x' + j}); return this.v + junk.length; } });
function pair(m, o) { var keep = {a: o.a + 1}; return m.w + keep.a; }
function hold(n) { var m = new M(3), o = {a: 1}, s = 0;
  for (var i = 0; i < n; i++) { var t = {b: i}; s += pair(m, o) + t.b + o.a; } return s; }
var r = 0; for (var k = 0; k < 100; k++) r += hold(10); r";

/// The graph counterparts of the splice subjects step 2's record lists before the flip: a callee
/// of another realm at a warmed site, a closure callee that outlives its last reference (the
/// site meets another closure after a collection), a callee's string constants.
const COUNTERPARTS: &[&str] = &[
    "var other = $262.createRealm().global;
     other.eval(\"function fr(o) { return o.w; }\");
     function M(v) { this.v = v; }
     Object.defineProperty(M.prototype, 'w', { get: function () { return this.v + 1; } });
     function fl(o) { return o.w; }
     function cr(f, o) { var s = 0; for (var i = 0; i < 10; i++) s += f(o); return s; }
     var m = new M(4), r = 0; for (var k = 0; k < 300; k++) r += cr(fl, m);
     r + ',' + cr(other.fr, m) + ',' + cr(fl, m)",
    "function M(v) { this.v = v; }
     Object.defineProperty(M.prototype, 'w', { get: function () { return this.v + 1; } });
     function mkc(k) { return function (o) { return o.w + k; }; }
     var f = mkc(1);
     function cc(o) { var s = 0; for (var i = 0; i < 10; i++) s += f(o); return s; }
     var m = new M(4), r = 0; for (var k = 0; k < 300; k++) r += cc(m);
     f = mkc(2); var junk = []; for (var j = 0; j < 20000; j++) junk.push({j: j}); junk = null;
     r + ',' + cc(m)",
    "function M(v) { this.v = v; }
     Object.defineProperty(M.prototype, 'w', { get: function () { return this.v + 1; } });
     function sc(o) { return o.w + 'tail' + o.v; }
     function cs(o) { var t = ''; for (var i = 0; i < 10; i++) t = sc(o) + i; return t; }
     var m = new M(4), r = ''; for (var k = 0; k < 300; k++) r = cs(m);
     var junk = []; for (var j = 0; j < 20000; j++) junk.push('s' + j); junk = null; r + cs(m)",
];

#[test]
fn an_inlined_frames_generic_node_takes_every_named_slot_and_leaves_it_a_value() {
    // lw's two reads of `q.w` (an accessor: generic) each take q's value as an input: an inlined
    // frame has no frame, so the first node's write of its slot window holds nothing the second
    // can read, and q stays a value (no state of the frame names the frame).
    let mut engine = warmed(
        "function M(v) { this.v = v; }
         Object.defineProperty(M.prototype, 'w', { get: function () { return this.v; } });
         function lw(o) { var q = o; var a = q.w; var b = q.w; return a + b; }
         function clw(o) { return lw(o); }
         var m = new M(1); for (var k = 0; k < 30; k++) clw(m);",
    );
    let built = Compile::new(&mut engine, "clw").build();
    assert_eq!(outcomes(&built), vec![(0, InlineOutcome::Inlined)]);
    let graph = &built.graph;
    let reads = nodes(graph, |node| {
        node.frame == FrameIx(1)
            && matches!(node.op, NodeOp::Generic { pops: 0, .. })
            && node.inputs.len() == 1
    });
    assert_eq!(reads.len(), 2, "{}", dump(graph));
    assert_eq!(
        graph.node(reads[0]).inputs,
        graph.node(reads[1]).inputs,
        "both read q's value: {}",
        dump(graph)
    );
    for &read in &reads {
        let state = graph.state(graph.node(read).state.unwrap());
        assert!(!state.slots.contains(&SlotSource::Frame));
    }
}

#[test]
fn a_call_whose_inlined_callee_runs_a_generic_node_is_no_lowered_operation() {
    // G's value, which `bump`'s setter changes, is read in the loop after each call: the loop
    // prepass must count the inlined call an effect (its callee's generic store may run any
    // JavaScript), so that no fact of G crosses it or the header.
    let source = "var G = 1;
         function M() {}
         Object.defineProperty(M.prototype, 'g', { set: function (z) { G = z; } });
         function bump(m, i) { m.g = i; }
         function loop(m, n) { var s = 0; for (var i = 0; i < n; i++) { s += G; bump(m, i); } return s; }
         var m = new M(), r = 0; for (var k = 0; k < 300; k++) r += loop(m, 10); r";
    inlines();
    inlined_generic();
    let (expected, got) = both(source, 0, false);
    assert_eq!(got, expected);
    assert!(inlines()
        .iter()
        .any(|note| note.outcome == InlineOutcome::Inlined));
    assert!(!inlined_generic().is_empty());
    // A loop free of effects keeps a typed array's data address through its header (step 0c):
    // the inlined callee's setter detaches the array, so the loop must count the call an effect
    // and check the array again after it.
    let typed = "var buf = new ArrayBuffer(32), t = new Int32Array(buf), arm = false;
         for (var j = 0; j < 8; j++) t[j] = j + 1;
         function M() {}
         Object.defineProperty(M.prototype, 'g', { set: function (z) {
           if (arm && z === 4) buf.transfer(); } });
         function bump(m, i) { m.g = i; }
         function loop(m, a, n) { var s = 0;
           for (var i = 0; i < n; i++) { s += a[i] | 0; bump(m, i); } return s; }
         var m = new M(), r = 0; for (var k = 0; k < 300; k++) r += loop(m, t, 8);
         arm = true; r + ',' + loop(m, t, 8) + ',' + t.length";
    inlined_generic();
    let (expected, got) = both(typed, 0, false);
    assert_eq!(got, expected);
    assert_eq!(expected, "10800,15,0");
    assert!(!inlined_generic().is_empty());
}

#[test]
fn the_inliner_declines_a_callee_past_the_compiled_frames_record() {
    // cbig's 225 locals and big's 30 (the planner's limit is 32): big's generic node's slot
    // window above its operand words would put cbig's frame past a pooled record (P3 stage 3
    // step 3c), where a callee without a generic node fits.
    let locals = |prefix: &str, n: usize| -> String {
        (0..n)
            .map(|k| format!("{prefix}{k}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let source = format!(
        "function M(v) {{ this.v = v; }}
         Object.defineProperty(M.prototype, 'w', {{ get: function () {{ return this.v; }} }});
         function big(o) {{ var {}; return o.w; }}
         function cbig(o) {{ var {}; return big(o); }}
         var m = new M(1); for (var k = 0; k < 30; k++) cbig(m);",
        locals("v", 29),
        locals("u", 225)
    );
    let mut engine = warmed(&source);
    let built = Compile::new(&mut engine, "cbig").build();
    assert_eq!(
        outcomes(&built),
        vec![(
            0,
            InlineOutcome::Declined {
                reason: super::build::PAST_RECORD,
                at: None,
            }
        )]
    );
}

#[test]
fn generic_operations_of_inlined_callees_give_the_tier_offs_results() {
    inlines();
    inlined_generic();
    let (expected, got) = both(GENERIC_KINDS, 0, false);
    assert_eq!(got, expected);
    let inlined = inlines()
        .iter()
        .filter(|note| note.outcome == InlineOutcome::Inlined)
        .count();
    let ops = inlined_generic();
    assert!(inlined >= 10, "the callees inline ({inlined} ways)");
    for kind in [
        |op: &Op| matches!(op, Op::GetPropThis(..)),
        |op: &Op| matches!(op, Op::SetPropThisDrop(..)),
        |op: &Op| matches!(op, Op::GetPropLocal(..)),
        |op: &Op| matches!(op, Op::SetPropLocalDrop(..)),
        |op: &Op| matches!(op, Op::GetElem | Op::GetElemLocal(_)),
        |op: &Op| matches!(op, Op::InstanceOf(_)),
        |op: &Op| matches!(op, Op::EqEq),
        |op: &Op| matches!(op, Op::UpdateProp(..)),
        |op: &Op| matches!(op, Op::UpdateLocal(..)),
    ] {
        assert!(ops.iter().any(kind), "{ops:?}");
    }
}

#[test]
fn generic_operations_of_inlined_callees_give_the_tier_offs_results_with_exits_forced() {
    for every in [7, 100] {
        let (expected, got) = both(GENERIC_KINDS, every, false);
        assert_eq!(got, expected, "an exit every {every} checks");
    }
}

#[test]
fn generic_operations_of_inlined_callees_give_the_tier_offs_results_under_the_poison() {
    for source in std::iter::once(GENERIC_KINDS).chain(CONTEXT_READS.iter().copied()) {
        super::take_poison_faults();
        let (expected, got) = both_poisoned(source, 0, false, true);
        assert_eq!(got, expected, "{source}");
        assert_eq!(
            super::take_poison_faults(),
            0,
            "no template read or wrote a poisoned word: {source}"
        );
    }
}

#[test]
fn an_inlined_generic_node_reads_its_frames_this_environment_and_slots() {
    for source in CONTEXT_READS {
        inlined_generic();
        let (expected, got) = both(source, 0, false);
        assert_eq!(got, expected, "{source}");
        assert!(
            !inlined_generic().is_empty(),
            "a generic node inlined: {source}"
        );
        // An exit after the node resumes the compiled frame in the VM on its context: `this`,
        // the environment and the slots restored (the compiled frame reads its own bindings
        // through constants, so only its exits read env_raw again).
        let (expected, got) = both(source, 7, false);
        assert_eq!(got, expected, "with exits forced: {source}");
    }
}

#[test]
fn error_stack_caller_and_positions_from_an_inlined_frames_node_equal_the_tier_offs() {
    inlined_generic();
    let (expected, got) = both(POSITIONS, 0, false);
    assert_eq!(got, expected);
    assert!(
        expected.contains("|truetruetrue|"),
        "the getters' callers: {expected}"
    );
    assert!(!inlined_generic().is_empty());
    let (expected, got) = both(POSITIONS, 7, false);
    assert_eq!(got, expected, "with exits forced");
}

#[test]
fn a_throw_from_an_inlined_frames_node_is_caught_by_its_callers_handler_or_leaves() {
    inlined_generic();
    inlined_exits();
    let (expected, got) = both(THROWS, 0, false);
    assert_eq!(got, expected);
    assert!(!inlined_generic().is_empty());
    assert!(
        inlined_exits()
            .iter()
            .any(|&(depth, _, reason)| depth == 1 && reason == ExitReason::Throw),
        "the caught throw exits through the inlined frame"
    );
    let (expected, got) = both(THROWS, 7, false);
    assert_eq!(got, expected, "with exits forced");
}

#[test]
fn collections_during_an_inlined_frames_helper_keep_both_frames_values() {
    let (expected, got) = both(COLLECTIONS, 0, true);
    assert_eq!(got, expected);
    let (expected, got) = both(THROWS, 7, true);
    assert_eq!(got, expected);
}

#[test]
fn the_graph_counterparts_of_the_splice_subjects_give_the_tier_offs_results() {
    for source in COUNTERPARTS {
        let (expected, got) = both(source, 0, false);
        assert_eq!(got, expected, "{source}");
        let (expected, got) = both(source, 7, false);
        assert_eq!(got, expected, "with exits forced: {source}");
    }
}

/// The poison switch's writer check over an inlined frame's generic node: a test hook makes the
/// exact-PC helper of dl's `delete` write its local x's slot, which `delete` does not name: in
/// the inlined frame the helper writes the node's slot window, which the check reports; the
/// tier's code still reads x from its home. Without the hook, no fault.
#[test]
fn the_poison_check_reports_an_inlined_frames_template_that_writes_a_slot_it_does_not_name() {
    let program = "function dl(o, n) { var x = n * 3; delete o.c; return x + o.a; }
         function cdl(o, n) { return dl(o, n); }
         var t = 0; for (var k = 0; k < 300; k++) t += cdl({a: k, c: 1}, k); t";
    TestSwitches::default().set();
    let mut off = jit_engine();
    let expected = eval(&mut off, program);
    TestSwitches {
        poison: true,
        ..graph_switches(0)
    }
    .set();
    let mut engine = jit_engine();
    inlined_generic();
    assert_eq!(eval(&mut engine, program), expected);
    assert!(
        inlined_generic()
            .iter()
            .any(|op| matches!(op, Op::DeleteProp(..))),
        "dl's delete is an inlined frame's generic node"
    );
    let dl = chunk_of(&mut engine, "dl");
    let pc = pc_of(&dl, |op| matches!(op, Op::DeleteProp(..)));
    let slot = dl
        .jit_slot_names()
        .iter()
        .position(|name| &**name == "x")
        .expect("x has a slot");
    super::take_poison_faults();
    assert_eq!(eval(&mut engine, "cdl({a: 1, c: 2}, 4)"), "13");
    assert_eq!(super::take_poison_faults(), 0, "no hidden writer, no fault");
    crate::bytecode::TEST_HIDDEN_SLOT_WRITE.with(|write| {
        write.set(Some((Rc::as_ptr(&dl) as usize, pc, slot)));
    });
    let result = eval(&mut engine, "cdl({a: 1, c: 2}, 4)");
    crate::bytecode::TEST_HIDDEN_SLOT_WRITE.with(|write| write.set(None));
    assert_eq!(result, "13", "the tier's code reads x from its home");
    assert_eq!(
        super::take_poison_faults(),
        1,
        "the check reports the write"
    );
}

/// The collector's extent of an active record (`JitCtx::frame_words`): its slots and its chunk's
/// operand capacity, and the whole record while its slots are not its own first words, as while
/// an inlined frame's generic node runs on its slot window (P3 stage 3 step 3c).
#[test]
fn a_record_whose_slots_are_a_window_is_scanned_whole() {
    let mut engine = warmed(MISC);
    let chunk = chunk_of(&mut engine, "rd");
    let i = &mut engine.interp;
    let record = crate::jit::JitFrame::alloc(i);
    // SAFETY: a record just allocated, which no code runs on.
    let ctx = unsafe { &mut (*record.as_ptr()).ctx };
    let own = ctx.slots;
    ctx.chunk = Rc::as_ptr(&chunk);
    ctx.n_slots = 2;
    let capacity = chunk.jit.max_stack().expect("rd has template code");
    assert!(ctx.slots_are_own());
    assert_eq!(ctx.frame_words(), 2 + capacity);
    // SAFETY: within the record's words.
    ctx.slots = unsafe { own.add(2 + capacity) };
    assert!(!ctx.slots_are_own());
    assert_eq!(ctx.frame_words(), crate::jit::FRAME_BUF);
    ctx.slots = own;
    ctx.chunk = std::ptr::null();
    ctx.n_slots = 0;
    // SAFETY: the record runs nothing; it goes to the pool as any finished activation's does.
    unsafe { crate::jit::JitFrame::release(i, record) };
}

/// The code of a function with an inlined frame's generic node declares the operand words its
/// node takes, the callee's capacity and slot window above the compiled frame's operand base,
/// which every entry sizes its stack from (P3 stage 3 step 3c).
#[test]
fn an_inlined_frames_operand_words_and_window_bound_the_codes_stack() {
    let locals = (0..30)
        .map(|k| format!("v{k}"))
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "function M(v) {{ this.v = v; }}
         Object.defineProperty(M.prototype, 'w', {{ get: function () {{ return this.v; }} }});
         function wide(o) {{ var {locals}; return o.w; }}
         function cw(o) {{ return wide(o); }}
         var m = new M(1), r = 0; for (var k = 0; k < 300; k++) r += cw(m); r"
    );
    TestSwitches::default().set();
    let expected = eval(&mut jit_engine(), &source);
    graph_switches(0).set();
    let mut engine = jit_engine();
    inlined_generic();
    assert_eq!(eval(&mut engine, &source), expected);
    assert!(!inlined_generic().is_empty());
    let cw = chunk_of(&mut engine, "cw");
    let stack = cw.jit.max_stack().expect("cw has code");
    assert!(
        stack > 30,
        "the code's operand words cover wide's window: {stack}"
    );
}

/// The graph counterpart of the splice's `spliced_callee_outlives_dropping_its_last_reference`
/// (P3 stage 3 step 3c; step 2's record asks for it before the flip): the inlined callee drops the
/// last reference to itself, then its generic node's getter allocates (a collection at every
/// allocation), then it reads its captured bindings, whose records the code embeds. The frame's
/// callee is a node every state of the frame names, in a home the collector scans, which keeps
/// the closure and its environment alive; the callee's chunk, which ctx.chunk names while the
/// node runs, the code holds (`exit::Inlined::chunks`).
#[test]
fn an_inlined_callee_that_drops_its_last_reference_runs_on_through_collections() {
    let source = "function M() {}
         Object.defineProperty(M.prototype, 'w', { get: function () {
           var junk = []; for (var j = 0; j < 4; j++) junk.push({j: j}); return junk.length; } });
         function makeWarm() { var secret = {v: 42}; var other = [1, 2, 3];
           return function target(holder) { var m = holder.m; if (holder.drop) holder.fn = null;
             var k = m.w; return secret.v + other.length + k; }; }
         function invoke(holder) { return holder.fn(holder); }
         function drive(make) {
           var holder = { fn: make(), drop: false, m: new M() }, last;
           for (var i = 0; i < 300; i++) last = invoke(holder);
           holder.drop = true;
           return last + '->' + invoke(holder) + ',' + holder.fn;
         }
         drive(makeWarm)";
    inlines();
    inlined_generic();
    let (expected, got) = both(source, 0, true);
    assert_eq!(got, expected);
    assert_eq!(expected, "49->49,null");
    assert!(inlines()
        .iter()
        .any(|note| note.outcome == InlineOutcome::Inlined));
    assert!(!inlined_generic().is_empty());
}

/// A slot window above its node's operand top is scanned where the collector knows the record's
/// top (P3 stage 3 step 3c; the coordinator's review): inc's `x++` of a BigInt is a generic node
/// that writes x with a fresh cell into its window, which its slot result reads after the
/// template's end poll; under a collection at every allocation that poll collects with the
/// record's top its x20, below the window. Every such trace scans the window
/// (`interpreter::TEST_WINDOW_TRACES`), and the results are the tier off's.
#[test]
fn a_slot_window_is_scanned_at_its_nodes_end_poll() {
    let source = "function inc(x) { x++; return x; }
         function run(n) { var out = [], b = 2n ** 80n;
           for (var i = 0; i < n; i++) { b = inc(b); out.push(b); } return out; }
         var r; for (var k = 0; k < 40; k++) r = run(20);
         String(r[0]) + ',' + String(r[19]) + ',' + r.length";
    crate::interpreter::TEST_WINDOW_TRACES.with(|traces| traces.set(Default::default()));
    inlined_generic();
    let (expected, got) = both(source, 0, true);
    assert_eq!(got, expected);
    assert!(
        inlined_generic()
            .iter()
            .any(|op| matches!(op, Op::UpdateLocal(..))),
        "inc's update is an inlined frame's generic node"
    );
    let traces = crate::interpreter::TEST_WINDOW_TRACES.with(std::cell::Cell::get);
    assert!(
        traces.poll_tops > 0,
        "a collection ran at the node's end poll while its window was live: {traces:?}"
    );
    assert_eq!(
        traces.missed, 0,
        "every window traced was scanned: {traces:?}"
    );
}

// ---- P3 stage 3 step 3c-2: an inlined frame's direct accessor calls ---------------------------

/// `source`'s completion with the tier off, at TRUST_JS_OPT_INLINE=splice and at graph (an exit
/// forced every `exit_every` checks; collections at every allocation where `stress`, at graph),
/// each with the direct accessor calls inlined frames' helper stubs made
/// (`jit::take_inlined_accessor_calls`: getter, setter and trap calls that returned, then those
/// that threw), and the poison switch at graph where `poison`.
fn counted(source: &str, exit_every: u32, stress: bool, poison: bool) -> [(String, [u64; 6]); 3] {
    let run = |switches: TestSwitches, stress: bool| {
        switches.set();
        let mut engine = jit_engine();
        if stress {
            let mut core = engine.interp.gc_heap.core.borrow_mut();
            core.set_stress_every(1);
            core.set_incremental(crate::gc_core::IncrementalMode::Off);
        }
        crate::jit::take_inlined_accessor_calls();
        let got = eval(&mut engine, source);
        (got, crate::jit::take_inlined_accessor_calls())
    };
    let splice = TestSwitches {
        inline_graph: Some(false),
        ..graph_switches(exit_every)
    };
    let graph = TestSwitches {
        poison,
        ..graph_switches(exit_every)
    };
    [
        run(TestSwitches::default(), false),
        run(splice, false),
        run(graph, stress),
    ]
}

/// The direct calls' counts of `counted`: none with the tier off or at splice (the variant stub
/// is an inlined frame's only), and at graph at least one returned call of each accessor kind
/// `returned` marks ([getter, setter, trap]) and one thrown call of each `threw` marks.
fn assert_direct_calls(runs: &[(String, [u64; 6]); 3], returned: [bool; 3], threw: [bool; 3]) {
    assert_eq!(
        runs[0].1, [0; 6],
        "no direct call of the variant with the tier off"
    );
    assert_eq!(runs[1].1, [0; 6], "no direct call of the variant at splice");
    let calls = runs[2].1;
    for (kind, name) in ["getter", "setter", "trap"].iter().enumerate() {
        if returned[kind] {
            assert!(calls[kind] > 0, "a direct {name} call returned: {calls:?}");
        }
        if threw[kind] {
            assert!(calls[3 + kind] > 0, "a direct {name} call threw: {calls:?}");
        }
    }
}

/// A getter, a setter and a Proxy `get` trap that inlined frames' generic reads and stores call
/// (GetPropThis, SetPropThisDrop, GetPropLocal, SetPropLocalDrop, GetProp), each recording what
/// it sees once the code runs: its caller (`f.caller`), its Error.stack, the stack of a TypeError
/// its own operation raises, and the caller a toString it calls implicitly sees; then the same
/// below a recursion past the direct calls' depth (`Interp::direct_call_depth`), where every
/// direct sequence declines at its depth gate and the stub takes the plain helper.
const DIRECT_ACCESSORS: &str = "
var probe = false, log = [];
var o2 = { toString: function ts() { if (probe) log.push('ts<' + (ts.caller && ts.caller.name)); return 'x'; } };
function seen(tag, f) {
  return tag + '<' + (f.caller && f.caller.name) + '|' + new Error(tag).stack.split('\\n').slice(0, 4).join('/');
}
function M(v) { this.v = v; }
Object.defineProperty(M.prototype, 'w', {
  get: function getter() {
    if (probe) { var e = ''; try { null.boom; } catch (x) { e = x.stack.split('\\n').slice(0, 4).join('/'); }
      log.push(seen('g', getter) + '|' + e + '|' + o2); }
    return this.v + 1; },
  set: function setter(z) {
    if (probe) { var e = ''; try { null.boom; } catch (x) { e = x.stack.split('\\n').slice(0, 4).join('/'); }
      log.push(seen('s', setter) + '|' + e + '|' + o2); }
    this.v = z * 2; } });
M.prototype.rw = function rw() { return this.w; };
M.prototype.ww = function ww(z) { this.w = z; };
function loc(o) { var q = o; return q.w; }
function sloc(o, z) { var q = o; q.w = z; }
var ph = { get: function trap(t, k, r) {
  if (probe) { var e = ''; try { null.boom; } catch (x) { e = x.stack.split('\\n').slice(0, 4).join('/'); }
    log.push(seen('t', trap) + '|' + e + '|' + o2); }
  return k === 'w' ? 7 : 0; } };
var px = new Proxy({}, ph), h = {p: px};
function gp(h) { return h.p.w; }
function run(m, n) { var s = 0;
  for (var i = 0; i < n; i++) { s += m.rw(); m.ww(i); s += loc(m); sloc(m, i + 1); s += gp(h) + loc(px); }
  return s; }
function deep(n) { return n > 0 ? deep(n - 1) : run(m, 1); }
var m = new M(1), r = 0;
for (var k = 0; k < 300; k++) r += run(m, 10);
probe = true; r += run(m, 1); log.push('deep'); r += deep(140); probe = false;
r + '\\n' + log.join('\\n')";

/// A throw from each accessor kind an inlined frame calls directly, caught by its caller's
/// handler around the call and by none (the throw leaves the compiled function: each leaving
/// function twice, since the accessor's first throw exits its own code, whose next call the way
/// refills through the helper, and the second call is direct again); and the [[Get]]
/// invariant's TypeError (ECMA-262 §10.5.8 step 10) after a trap's direct call returned, which
/// captures its stack from `Interp::call_site`.
const DIRECT_ACCESSOR_THROWS: &str = "
function M(v) { this.v = v; this.bad = ''; }
Object.defineProperty(M.prototype, 'w', {
  get: function () { if (this.bad) throw new TypeError('get ' + this.bad); return this.v; },
  set: function (z) { if (this.bad) throw new RangeError('set ' + this.bad); this.v = z; } });
M.prototype.rw = function rw() { return this.w; };
M.prototype.ww = function ww(z) { this.w = z; };
function loc(o) { var q = o; return q.w; }
function sloc(o, z) { var q = o; q.w = z; }
var tgt = {bad: ''};
var px = new Proxy(tgt, { get: function (t, k) { if (t.bad) throw new SyntaxError('trap ' + t.bad); return 5; } });
var h = {p: px};
function gp(h) { return h.p.w; }
function caught(m, n) { var s = 0, msg = '';
  for (var i = 0; i < n; i++) {
    try { s += m.rw(); } catch (e) { s += 1000; msg += e.message + ';'; }
    try { m.ww(i); } catch (e) { s += 2000; msg += e.message + ';'; }
    try { s += loc(m); } catch (e) { s += 3000; msg += e.name + ';'; }
    try { sloc(m, i); } catch (e) { s += 4000; msg += e.name + ';'; }
    try { s += gp(h); } catch (e) { s += 5000; msg += e.message + ';'; }
  }
  return s + ':' + msg; }
function l1(m, n) { var s = 0; for (var i = 0; i < n; i++) s += m.rw(); return s; }
function l2(m, n) { for (var i = 0; i < n; i++) m.ww(i); return m.v; }
function l3(m, n) { var s = 0; for (var i = 0; i < n; i++) s += loc(m); return s; }
function l4(m, n) { for (var i = 0; i < n; i++) sloc(m, i); return m.v; }
function l5(h, n) { var s = 0; for (var i = 0; i < n; i++) s += gp(h); return s; }
var frozen = {}, lie = false;
Object.defineProperty(frozen, 'w', {value: 1, writable: false, configurable: false});
var liar = new Proxy(frozen, { get: function (t, k) { return lie ? 2 : 1; } });
function l6(n) { var s = 0; for (var i = 0; i < n; i++) s += loc(liar); return s; }
function stack(e) { return e.name + ':' + e.message + ':' + e.stack.split('\\n').slice(0, 4).join('/'); }
var m = new M(1), r = '';
for (var k = 0; k < 300; k++) { r = caught(m, 10); l1(m, 10); l2(m, 10); l3(m, 10); l4(m, 10); l5(h, 10); l6(10); }
var out = [];
lie = true; try { out.push(l6(3)); } catch (e) { out.push(stack(e)); } lie = false;
m.bad = 'x'; tgt.bad = 'y'; out.push(caught(m, 2));
var leaves = [function () { return l1(m, 3); }, function () { return l2(m, 3); }, function () { return l3(m, 3); },
              function () { return l4(m, 3); }, function () { return l5(h, 3); }];
for (var j = 0; j < 2 * leaves.length; j++) { try { out.push(leaves[j >> 1]()); } catch (e) { out.push(stack(e)); } }
m.bad = ''; tgt.bad = '';
r + '|' + out.join('|') + '|' + caught(m, 2)";

/// Collections inside the accessors an inlined frame's windowed nodes (GetPropLocal,
/// SetPropLocalDrop) call directly, with objects live in the window (the local receiver) and in
/// both frames (the callee's `keep`, the caller's `o` and `t`).
const DIRECT_ACCESSOR_COLLECTIONS: &str = "
function M(v) { this.v = v; }
Object.defineProperty(M.prototype, 'w', {
  get: function () { var junk = {j: this.v}; return junk.j + 1; },
  set: function (z) { var junk = [z, z]; this.v = junk[0] + junk.length; } });
var px = new Proxy({}, { get: function (t, k) { var junk = {k: k}; return junk.k.length; } });
function loc(o, z) { var q = o, keep = {a: z}; var w = q.w; return w + keep.a; }
function sloc(o, z) { var q = o, keep = {a: z}; q.w = keep.a; return keep.a; }
function lpx(z) { var q = px, keep = [z]; return q.w + keep[0]; }
function hold(n) { var m = new M(3), o = {a: 1}, s = 0;
  for (var i = 0; i < n; i++) { var t = {b: i}; s += loc(m, i) + sloc(m, i) + lpx(i) + t.b + o.a; }
  return s + ':' + m.v; }
var r = ''; for (var k = 0; k < 100; k++) r = hold(10); r";

/// Accessors replaced (a new pair of new functions, the holder's shape kept or not) and
/// recompiled (the getter's loop makes it hot enough for the tier) while inlined frames call them
/// directly.
const DIRECT_ACCESSOR_RECOMPILES: &str = "
function M(v) { this.v = v; }
var gen = 0;
function install() { var g = gen++;
  Object.defineProperty(M.prototype, 'w', { configurable: true,
    get: function () { var s = 0; for (var j = 0; j < 4; j++) s += j; return this.v + g + s; },
    set: function (z) { this.v = z + g; } }); }
install();
M.prototype.rw = function () { return this.w; };
M.prototype.ww = function (z) { this.w = z; };
function run(m, n) { var s = 0; for (var i = 0; i < n; i++) { s += m.rw(); m.ww(i & 3); } return s; }
var m = new M(1), r = [];
for (var k = 0; k < 600; k++) { r.push(run(m, 10)); if (k % 150 == 75) install(); }
r.slice(-3).join(',') + ':' + r.reduce(function (a, b) { return a + b; }, 0) + ':' + gen";

#[test]
fn an_inlined_frames_accessors_are_called_directly_with_the_tier_offs_positions_and_stacks() {
    inlined_generic();
    let runs = counted(DIRECT_ACCESSORS, 0, false, false);
    let expected = &runs[0].0;
    assert_eq!(&runs[1].0, expected, "splice");
    assert_eq!(&runs[2].0, expected, "graph");
    for seen in [
        "g<rw|",
        "s<ww|",
        "g<loc|",
        "s<sloc|",
        "t<gp|",
        "t<loc|",
        "ts<getter",
        "ts<setter",
        "ts<trap",
    ] {
        assert!(expected.contains(seen), "{seen}: {expected}");
    }
    let ops = inlined_generic();
    for kind in [
        |op: &Op| matches!(op, Op::GetPropThis(..)),
        |op: &Op| matches!(op, Op::SetPropThisDrop(..)),
        |op: &Op| matches!(op, Op::GetPropLocal(..)),
        |op: &Op| matches!(op, Op::SetPropLocalDrop(..)),
        |op: &Op| matches!(op, Op::GetProp(..)),
    ] {
        assert!(ops.iter().any(kind), "{ops:?}");
    }
    assert_direct_calls(&runs, [true; 3], [false; 3]);
}

#[test]
fn an_inlined_frames_direct_accessor_calls_give_the_tier_offs_results_with_exits_forced_and_poisoned(
) {
    for source in [DIRECT_ACCESSORS, DIRECT_ACCESSOR_THROWS] {
        let runs = counted(source, 7, false, false);
        assert_eq!(runs[2].0, runs[0].0, "an exit every 7 checks: {source}");
        assert_eq!(
            runs[1].0, runs[0].0,
            "splice, an exit every 7 checks: {source}"
        );
        super::take_poison_faults();
        let runs = counted(source, 0, false, true);
        assert_eq!(runs[2].0, runs[0].0, "poisoned: {source}");
        assert_eq!(
            super::take_poison_faults(),
            0,
            "no template read or wrote a poisoned word: {source}"
        );
        assert!(
            runs[2].1[..3].iter().all(|&calls| calls > 0),
            "{:?}",
            runs[2].1
        );
    }
}

#[test]
fn a_throw_from_an_accessor_an_inlined_frame_calls_directly_is_caught_by_its_callers_handler_or_leaves(
) {
    inlined_exits();
    let runs = counted(DIRECT_ACCESSOR_THROWS, 0, false, false);
    let expected = &runs[0].0;
    assert_eq!(&runs[1].0, expected, "splice");
    assert_eq!(&runs[2].0, expected, "graph");
    for thrown in [
        "TypeError:get x:",
        "RangeError:set x:",
        "SyntaxError:trap y:",
    ] {
        assert!(expected.contains(thrown), "{thrown}: {expected}");
    }
    assert_direct_calls(&runs, [true; 3], [true; 3]);
    assert!(
        inlined_exits()
            .iter()
            .any(|&(depth, _, reason)| depth == 1 && reason == ExitReason::Throw),
        "a caught throw exits through the inlined frame"
    );
}

/// A slot window above its node's operand top is scanned where a direct accessor call records the
/// record's top (P3 stage 3 step 3c-2): the getter, setter and trap that GetPropLocal and
/// SetPropLocalDrop call directly collect at every allocation, with the record's top the call's
/// (its callee's `final_sp`, the node's x20 and the call's words), below the window.
#[test]
fn a_slot_window_is_scanned_at_a_direct_accessor_calls_top() {
    crate::interpreter::TEST_WINDOW_TRACES.with(|traces| traces.set(Default::default()));
    let runs = counted(DIRECT_ACCESSOR_COLLECTIONS, 0, true, false);
    assert_eq!(runs[2].0, runs[0].0);
    assert_eq!(runs[1].0, runs[0].0, "splice");
    assert_direct_calls(&runs, [true; 3], [false; 3]);
    let traces = crate::interpreter::TEST_WINDOW_TRACES.with(std::cell::Cell::get);
    assert!(
        traces.call_tops > 0,
        "a collection ran at a direct call's top while the node's window was live: {traces:?}"
    );
    assert_eq!(
        traces.missed, 0,
        "every window traced was scanned: {traces:?}"
    );
}

#[test]
fn an_accessor_replaced_or_recompiled_while_an_inlined_frame_calls_it_directly_keeps_the_tier_offs_results(
) {
    let runs = counted(DIRECT_ACCESSOR_RECOMPILES, 0, false, false);
    assert_eq!(runs[2].0, runs[0].0);
    assert_eq!(runs[1].0, runs[0].0, "splice");
    assert!(runs[0].0.ends_with(":5"), "{}", runs[0].0);
    assert_direct_calls(&runs, [true, true, false], [false; 3]);
    let runs = counted(DIRECT_ACCESSOR_RECOMPILES, 7, false, false);
    assert_eq!(runs[2].0, runs[0].0, "an exit every 7 checks");
}

/// Callers of inlined accessor reads and stores (GetPropThis, SetPropThisDrop, GetPropLocal,
/// SetPropLocalDrop), warmed so that each calls its getter or setter directly; a trigger that
/// reports the error a call of one of them throws, with its stack; and a reference whose
/// accessor throws its own error (`bad`), whose stack places the caller's frame at the inlined
/// call.
const END_POLL_WARM: &str = "
function M(v) { this.v = v; }
Object.defineProperty(M.prototype, 'w', {
  get: function getter() { if (this.boom) throw new TypeError('boom'); return this.v + 1; },
  set: function setter(z) { if (this.boom) throw new TypeError('boom'); this.v = z; } });
M.prototype.rw = function rw() { return this.w; };
M.prototype.ww = function ww(z) { this.w = z; };
function loc(o) { var q = o; return q.w; }
function sloc(o, z) { var q = o; q.w = z; }
function c1(m, n) { var s = 0; for (var i = 0; i < n; i++) s += m.rw(); return s; }
function c2(m, n) { for (var i = 0; i < n; i++) m.ww(i); return m.v; }
function c3(m, n) { var s = 0; for (var i = 0; i < n; i++) s += loc(m); return s; }
function c4(m, n) { for (var i = 0; i < n; i++) sloc(m, i); return m.v; }
function c5(m, n) { var s = 0;
  for (var i = 0; i < n; i++) { try { s += m.rw(); } catch (e) { return 'caught ' + e.name + '|' + e.stack.split('\\n').slice(0, 4).join('/'); } }
  return s; }
var m = new M(1), r = 0;
for (var k = 0; k < 300; k++) r += c1(m, 10) + c2(m, 10) + c3(m, 10) + c4(m, 10) + c5(m, 10);
function trig(f) { try { return 'none:' + f(m, 3); }
  catch (e) { return e.name + ':' + e.message + '|' + e.stack.split('\\n').slice(0, 4).join('/'); } }
var bad = new M(1); bad.boom = true;
function ref(f) { try { return 'none:' + f(bad, 1); }
  catch (e) { return e.name + ':' + e.message + '|' + e.stack.split('\\n').slice(0, 4).join('/'); } }
r";

/// The live-object limit's RangeError raised at the end poll of an inlined frame's property node
/// after its getter or setter returned through the helper stub's direct sequence (P3 stage 3 step
/// 3c-2's review; `jit::arm_end_poll_limit`, a test-only request that makes that poll's
/// `gc_check` find the ceiling passed, since no real shape pins which poll first sees it). The
/// poll helper places no error (`jit_interrupt`): the stack is what `capture_stack_trace` finds
/// from `Interp::call_site`. After the variant put the node's site back there it names the
/// inlined callee's frame, reported at its start as a frame whose position is pending is, and the
/// compiled frame at the inlined call, the position the reference's error (the accessor's own,
/// with the tier off) gives the caller's frame. Neither the tier off nor the splice gives the
/// same stack to compare: the tier off's template code for these callees has no end poll at the
/// site (the request raises the error at its next poll, only checked to be caught), and the
/// splice's spliced body publishes nothing at its end poll (its stack starts at the compiled
/// frame, reported at its start: shown, not asserted).
#[test]
fn an_error_at_the_end_poll_after_a_direct_accessor_call_has_the_inlined_frame() {
    const TRIGGERS: [&str; 5] = ["trig(c1)", "trig(c2)", "trig(c3)", "trig(c4)", "c5(m, 3)"];
    let run = |switches: TestSwitches, any: bool, refs: bool| -> Vec<String> {
        switches.set();
        let mut engine = jit_engine();
        engine
            .interp
            .gc_heap
            .core
            .borrow_mut()
            .set_incremental(crate::gc_core::IncrementalMode::Off);
        eval(&mut engine, END_POLL_WARM);
        if refs {
            return ["ref(c1)", "ref(c2)", "ref(c3)", "ref(c4)"]
                .iter()
                .map(|reference| eval(&mut engine, reference))
                .collect();
        }
        TRIGGERS
            .iter()
            .map(|trigger| {
                crate::jit::arm_end_poll_limit(any);
                let got = eval(&mut engine, trigger);
                assert!(
                    !crate::jit::take_end_poll_limit(crate::jit::POLL_HELPER_END),
                    "the request fell due and no poll took it: {got}"
                );
                got
            })
            .collect()
    };
    let refs = run(TestSwitches::default(), false, true);
    let off = run(TestSwitches::default(), true, false);
    for got in &off {
        assert!(got.contains("RangeError"), "the tier off: {got}");
    }
    let splice = run(
        TestSwitches {
            inline_graph: Some(false),
            ..graph_switches(0)
        },
        false,
        false,
    );
    for got in &splice {
        assert!(got.contains("RangeError"), "splice: {got}");
    }
    inlined_generic();
    let graph = run(graph_switches(0), false, false);
    let lines = |text: &str| -> Vec<String> {
        text.split('|')
            .nth(1)
            .unwrap_or("")
            .split('/')
            .map(str::to_string)
            .collect()
    };
    for (k, (callee, caller)) in [("rw", "c1"), ("ww", "c2"), ("loc", "c3"), ("sloc", "c4")]
        .iter()
        .enumerate()
    {
        let (got, reference) = (lines(&graph[k]), lines(&refs[k]));
        assert!(
            graph[k].starts_with("RangeError:allocation limit exceeded|"),
            "graph: {}",
            graph[k]
        );
        assert!(
            got[1].starts_with(&format!("    at {callee} (")),
            "graph names the inlined callee: {} (splice: {})",
            graph[k],
            splice[k]
        );
        let call = reference
            .iter()
            .find(|line| line.starts_with(&format!("    at {caller} (")))
            .unwrap_or_else(|| panic!("the reference: {}", refs[k]));
        assert_eq!(
            &got[2], call,
            "the compiled frame at the inlined call: {}",
            graph[k]
        );
    }
    assert!(graph[4].starts_with("caught RangeError|"), "{}", graph[4]);
    assert!(
        lines(&graph[4])[1].starts_with("    at rw ("),
        "{}",
        graph[4]
    );
    let ops = inlined_generic();
    for kind in [
        |op: &Op| matches!(op, Op::GetPropThis(..)),
        |op: &Op| matches!(op, Op::SetPropThisDrop(..)),
        |op: &Op| matches!(op, Op::GetPropLocal(..)),
        |op: &Op| matches!(op, Op::SetPropLocalDrop(..)),
    ] {
        assert!(ops.iter().any(kind), "{ops:?}");
    }
    // The splice's and the tier off's, for the record (`--nocapture`).
    println!("splice {splice:?}\noff {off:?}\ngraph {graph:?}");
}

// ---- P3 stage 3 step 3c-2: calls from inlined frames ------------------------------------------

/// Functions an inlined frame calls (P3 stage 3 step 3c-2), each recording what it sees once the
/// code runs: its `caller`, `arguments.callee.caller` and Error.stack, by every path a call takes
/// from an inlined frame's call node: a plain call to a compiled function (the direct sequence),
/// a method call, a megamorphic site (the call helpers), the function-call intrinsic
/// (`f.call`), a callee that builds an environment (`envy`), a bound function (`boundee`: no user
/// function), a Proxy of a function (`proxied`: the call helper makes its call, and the target's
/// frame takes the position the helper publishes), a native that throws (JSON.parse), a
/// function that throws (caught by the
/// compiled frame's handler and by none) and a call back into the compiled function itself; and
/// the compiled frame's own calls after the inlined frames' (a megamorphic site's, `envy`'s and
/// the bound function's, which the call helper makes: its position the compiled chunk's, x21
/// restored).
const INLINED_CALLS: &str = "
var probe = false, log = [];
function stk(e, from, n) { return e.stack.split('\\n').slice(from, from + n).join('/'); }
function callee(x) {
  if (probe) log.push('c<' + (callee.caller && callee.caller.name) + '|'
    + (arguments.callee.caller && arguments.callee.caller.name) + '|' + stk(new Error('c'), 0, 4));
  return x + 1;
}
function mk(k) { return function mega(x) {
  if (probe) log.push('m' + k + '<' + (mega.caller && mega.caller.name) + '|' + stk(new Error('m'), 1, 3));
  return x + k; }; }
var megas = [mk(1), mk(2), mk(3), mk(4), mk(5), mk(6)];
var o = { m: function meth(x) {
  if (probe) log.push('t<' + (meth.caller && meth.caller.name) + '|' + stk(new Error('t'), 1, 3));
  return x * 2; } };
function thrower(x) { if (x < 0) throw new RangeError('neg ' + x + ':' + arguments.length); return x; }
function envy(x) {
  if (probe) log.push('e<' + (envy.caller && envy.caller.name) + '|' + stk(new Error('e'), 1, 3));
  var f = function () { return x; }; return f(); }
function viaEnv(x) { return envy(x) + 1; }
var boundee = function bnd(x) {
  if (probe) log.push('bnd<' + (bnd.caller && bnd.caller.name) + '|' + stk(new Error('bd'), 1, 3));
  return x; }.bind(null);
function viaBound(x) { return boundee(x) + 1; }
var proxied = new Proxy(function pt(x) {
  if (probe) log.push('pt<' + (pt.caller && pt.caller.name) + '|' + stk(new Error('pt'), 1, 3));
  return x; }, {});
function viaProxy(x) { return proxied(x) + 1; }
function viaCall(x) { var s = callee(x); return s + 1; }
function viaMethod(o, x) { return o.m(x); }
function viaMega(i, x) { return megas[i % 6](x); }
function viaApply(x) { return callee.call(null, x); }
function viaThrow(x) { return thrower(x) + 1; }
function viaNative(s) { return JSON.parse(s); }
function run(n) { var s = 0;
  for (var i = 0; i < n; i++) {
    s += viaCall(i) + viaMethod(o, i) + viaMega(i, i) + viaApply(i) + viaThrow(i) + viaNative('1') + viaEnv(i) + viaBound(i) + viaProxy(i);
    s += megas[(i + 1) % 6](i) + envy(i) + boundee(i) + proxied(i);
  }
  return s; }
function caughtT(n, bad) { var s = 0;
  for (var i = 0; i < n; i++) {
    try { s += viaThrow(i === bad ? -i : i); } catch (e) { s += 1000; if (probe) log.push('caught|' + stk(e, 0, 4)); }
  }
  return s; }
function reenter(n) { return n > 0 ? viaRe(n - 1) : (probe ? (log.push('re|' + stk(new Error('re'), 1, 5)), 0) : 0); }
function viaRe(n) { return reenter(n) + 1; }
var r = 0;
for (var k = 0; k < 300; k++) { r += run(6); r += reenter(3); r += caughtT(4, -1); }
probe = true; r += run(2); r += reenter(2); r += caughtT(4, 3);
var out = [];
try { viaThrow(-5); } catch (e) { out.push('left|' + stk(e, 0, 4)); }
function leaves(x) { return viaThrow(x) + viaNative(x < 0 ? '{' : '2'); }
for (var k = 0; k < 300; k++) r += leaves(k);
try { leaves(-1); } catch (e) { out.push('left2|' + e.name + '|' + stk(e, 0, 4)); }
probe = false;
r + '\\n' + log.join('\\n') + '\\n' + out.join('\\n')";

/// Collections inside functions an inlined frame calls, the call's arguments and the inlined
/// frame's values live only in the call's operand words and the frames' homes (alloc's own
/// plan, the splice would inline in viaAlloc's body, is not taken: viaAlloc's call is a call).
const INLINED_CALL_COLLECTIONS: &str = "
function alloc(a, b) { var junk = [a, b, {x: a.v}]; return junk.length + a.v + b.k + arguments.length; }
function viaAlloc(o, i) { var keep = {k: i}; return alloc({v: o.v + i}, keep) + keep.k; }
function hold(n) { var o = {v: 3}, s = 0;
  for (var i = 0; i < n; i++) { var t = {b: i}; s += viaAlloc(o, i) + t.b; } return s; }
var r = 0; for (var k = 0; k < 100; k++) r += hold(10); r";

/// A recursion through an inlined frame (rec calls mid, inlined, which calls rec) until its
/// RangeError: the depth it reached.
const INLINED_CALL_DEPTH: &str = "
var deepest = 0;
function rec(n, lim) { if (n > deepest) deepest = n; if (n >= lim) return n; return mid(n, lim); }
function mid(n, lim) { return rec(n + 1, lim); }
for (var k = 0; k < 300; k++) rec(0, 8);
var e = ''; try { rec(0, 1e9); } catch (x) { e = x.name; }
e + ':' + deepest";

#[test]
fn a_function_an_inlined_frame_calls_sees_the_tier_offs_callers_and_stacks() {
    inlines();
    inlined_generic();
    let runs = counted(INLINED_CALLS, 0, false, false);
    let expected = &runs[0].0;
    assert_eq!(&runs[1].0, expected, "splice");
    assert_eq!(&runs[2].0, expected, "graph");
    for seen in [
        "c<viaCall|viaCall|",
        "t<viaMethod|",
        "<viaMega|",
        "c<viaApply",
        "e<viaEnv|",
        "bnd<",
        "pt<",
        "caught|",
        "re|",
        "left|",
        "left2|RangeError",
    ] {
        assert!(expected.contains(seen), "{seen}: {expected}");
    }
    // A call of a free name is a CallWithThis (its receiver undefined), as a method call is.
    let ops = inlined_generic();
    assert!(
        ops.iter()
            .filter(|op| matches!(op, Op::CallWithThis(..)))
            .count()
            >= 4,
        "{ops:?}"
    );
    assert!(inlines()
        .iter()
        .any(|note| note.outcome == InlineOutcome::Inlined));
}

#[test]
fn calls_from_inlined_frames_give_the_tier_offs_results_with_exits_forced_poisoned_and_collected() {
    for source in [INLINED_CALLS, INLINED_CALL_COLLECTIONS] {
        let runs = counted(source, 7, false, false);
        assert_eq!(runs[2].0, runs[0].0, "an exit every 7 checks: {source}");
        super::take_poison_faults();
        let runs = counted(source, 0, false, true);
        assert_eq!(runs[2].0, runs[0].0, "poisoned: {source}");
        assert_eq!(
            super::take_poison_faults(),
            0,
            "no poisoned word read or written: {source}"
        );
    }
    inlined_generic();
    let runs = counted(INLINED_CALL_COLLECTIONS, 0, true, false);
    assert_eq!(runs[2].0, runs[0].0, "collections at every allocation");
    assert!(inlined_generic()
        .iter()
        .any(|op| matches!(op, Op::Call(..) | Op::CallWithThis(..))));
}

/// The depth (P3 stage 3 step 3c-2): the tier off runs an inlined callee in a frame of its own,
/// the splice and the graph do not, and neither counts it in `Interp::depth`; the RangeError
/// comes where the native stack runs out (`interpreter::execution_stack_exhausted`, as the
/// exemplars bound theirs), so a recursion through an inlined frame reaches a depth its frames'
/// sizes decide: within 2% of the splice's at graph (on this program off 64,126 levels, splice
/// 64,118, graph 63,606 when written).
#[test]
fn a_recursion_through_an_inlined_frame_reaches_the_splices_depth() {
    let runs = counted(INLINED_CALL_DEPTH, 0, false, false);
    let depth = |run: &(String, [u64; 6])| -> u64 {
        let (error, depth) = run.0.split_once(':').expect("error:depth");
        assert_eq!(error, "RangeError", "{}", run.0);
        depth.parse().expect("a depth")
    };
    let (off, splice, graph) = (depth(&runs[0]), depth(&runs[1]), depth(&runs[2]));
    assert!(
        graph * 100 >= splice * 98 && graph * 100 <= splice * 102,
        "off {off}, splice {splice}, graph {graph}"
    );
}

/// A callee an inlined frame calls through the direct sequence (P3 stage 3 step 3c-2's review:
/// the call node's published site must not stay in `Interp::call_site` while it runs), from a
/// call outside a loop (`outerS`: the DirectCall stub) and inside one (`outerL`: the in-line
/// sequence): `inner` is too large to inline, so `mid`'s call to it is a call of the inlined
/// frame. What its own operations see: the caller a toString and a getter they call implicitly
/// see (`ts.caller`, `gg.caller`, the getter's stack) and the TypeError `o.x` raises; and what
/// the getter `mid`'s own read calls before the call sees (the inlined frame's site, not its
/// call's).
const DIRECT_CALLEE: &str = "
var probe = false, log = [];
var o2 = { toString: function ts() { if (probe) log.push('ts<' + (ts.caller && ts.caller.name)); return 'x'; } };
var G = {};
Object.defineProperty(G, 'g', { get: function gg() {
  if (probe) log.push('gg<' + (gg.caller && gg.caller.name) + '|' + new Error('g').stack.split('\\n').slice(1, 4).join('/'));
  return 1; } });
function inner(o) {
  if (o === -1) {
    var y = 0;
    y = y * 3 + 0; if (y > 1e9) y = y % 7;
    y = y * 3 + 1; if (y > 1e9) y = y % 7;
    y = y * 3 + 2; if (y > 1e9) y = y % 7;
    y = y * 3 + 3; if (y > 1e9) y = y % 7;
    y = y * 3 + 4; if (y > 1e9) y = y % 7;
    y = y * 3 + 5; if (y > 1e9) y = y % 7;
    y = y * 3 + 6; if (y > 1e9) y = y % 7;
    y = y * 3 + 7; if (y > 1e9) y = y % 7;
    y = y * 3 + 8; if (y > 1e9) y = y % 7;
    y = y * 3 + 9; if (y > 1e9) y = y % 7;
    y = y * 3 + 10; if (y > 1e9) y = y % 7;
    y = y * 3 + 11; if (y > 1e9) y = y % 7;
    y = y * 3 + 12; if (y > 1e9) y = y % 7;
    y = y * 3 + 13; if (y > 1e9) y = y % 7;
    y = y * 3 + 14; if (y > 1e9) y = y % 7;
    y = y * 3 + 15; if (y > 1e9) y = y % 7;
    y = y * 3 + 16; if (y > 1e9) y = y % 7;
    y = y * 3 + 17; if (y > 1e9) y = y % 7;
    y = y * 3 + 18; if (y > 1e9) y = y % 7;
    y = y * 3 + 19; if (y > 1e9) y = y % 7;
    y = y * 3 + 20; if (y > 1e9) y = y % 7;
    y = y * 3 + 21; if (y > 1e9) y = y % 7;
    y = y * 3 + 22; if (y > 1e9) y = y % 7;
    y = y * 3 + 23; if (y > 1e9) y = y % 7;
    y = y * 3 + 24; if (y > 1e9) y = y % 7;
    y = y * 3 + 25; if (y > 1e9) y = y % 7;
    y = y * 3 + 26; if (y > 1e9) y = y % 7;
    y = y * 3 + 27; if (y > 1e9) y = y % 7;
    y = y * 3 + 28; if (y > 1e9) y = y % 7;
    y = y * 3 + 29; if (y > 1e9) y = y % 7;
    y = y * 3 + 30; if (y > 1e9) y = y % 7;
    y = y * 3 + 31; if (y > 1e9) y = y % 7;
    y = y * 3 + 32; if (y > 1e9) y = y % 7;
    y = y * 3 + 33; if (y > 1e9) y = y % 7;
    y = y * 3 + 34; if (y > 1e9) y = y % 7;
    y = y * 3 + 35; if (y > 1e9) y = y % 7;
    y = y * 3 + 36; if (y > 1e9) y = y % 7;
    y = y * 3 + 37; if (y > 1e9) y = y % 7;
    y = y * 3 + 38; if (y > 1e9) y = y % 7;
    y = y * 3 + 39; if (y > 1e9) y = y % 7;
    y = y * 3 + 40; if (y > 1e9) y = y % 7;
    y = y * 3 + 41; if (y > 1e9) y = y % 7;
    y = y * 3 + 42; if (y > 1e9) y = y % 7;
    y = y * 3 + 43; if (y > 1e9) y = y % 7;
    y = y * 3 + 44; if (y > 1e9) y = y % 7;
    y = y * 3 + 45; if (y > 1e9) y = y % 7;
    y = y * 3 + 46; if (y > 1e9) y = y % 7;
    y = y * 3 + 47; if (y > 1e9) y = y % 7;
    y = y * 3 + 48; if (y > 1e9) y = y % 7;
    y = y * 3 + 49; if (y > 1e9) y = y % 7;
    y = y * 3 + 50; if (y > 1e9) y = y % 7;
    y = y * 3 + 51; if (y > 1e9) y = y % 7;
    y = y * 3 + 52; if (y > 1e9) y = y % 7;
    y = y * 3 + 53; if (y > 1e9) y = y % 7;
    y = y * 3 + 54; if (y > 1e9) y = y % 7;
    y = y * 3 + 55; if (y > 1e9) y = y % 7;
    y = y * 3 + 56; if (y > 1e9) y = y % 7;
    y = y * 3 + 57; if (y > 1e9) y = y % 7;
    y = y * 3 + 58; if (y > 1e9) y = y % 7;
    y = y * 3 + 59; if (y > 1e9) y = y % 7;
    return y;
  }
  var t = '' + o2;
  var u = G.g;
  return o.x + t.length + u;
}
function mid(o) { var g = G.g; var r = inner(o); return r + g; }
function outerS(o) { return mid(o); }
function outerL(o, n) { var r = 0; for (var i = 0; i < n; i++) r = mid(o); return r; }
function mid2(o) { var r = inner(o); return o.k.z + r; }
function outer2(o) { return mid2(o); }
var shapes = [{x: 1, a: 1, k: {z: 1, a: 1}}, {x: 1, b: 1, k: {z: 1, b: 1}}, {x: 1, c: 1, k: {z: 1, c: 1}},
              {x: 1, d: 1, k: {z: 1, d: 1}}, {x: 1, e: 1, k: {z: 1, e: 1}}, {x: 1, f: 1, k: {z: 1, f: 1}}];
function mid3(o) { var r = inner(o); return o.k.z + r; }
function outer3(o) { return mid3(o); }
function stk(e) { return e.name + '|' + e.stack.split('\\n').slice(0, 4).join('/'); }
var r = 0;
for (var k = 0; k < 300; k++) { r += outerS({x: 1}) + outerL({x: 1}, 3) + outer2({x: 1, k: {z: 1}}) + outer3(shapes[k % 6]); }
probe = true; r += outerS({x: 1}) + outerL({x: 1}, 1); probe = false;
var out = [];
try { outerS(null); } catch (e) { out.push(stk(e)); }
try { outerL(null, 1); } catch (e) { out.push(stk(e)); }
r + '\\n' + log.join('\\n') + '\\n' + out.join('\\n')";

#[test]
fn a_callee_an_inlined_frame_calls_directly_runs_without_the_published_site() {
    inlined_generic();
    let runs = counted(DIRECT_CALLEE, 0, false, false);
    let expected = &runs[0].0;
    assert_eq!(&runs[1].0, expected, "splice");
    assert_eq!(&runs[2].0, expected, "graph");
    for seen in [
        "ts<inner",
        "gg<inner|",
        "gg<mid|",
        "TypeError|TypeError: cannot read property 'x' of null/    at inner (",
    ] {
        assert!(expected.contains(seen), "{seen}: {expected}");
    }
    assert!(
        inlined_generic()
            .iter()
            .any(|op| matches!(op, Op::Call(..) | Op::CallWithThis(..))),
        "mid's call is a call node of its inlined frame"
    );
    let runs = counted(DIRECT_CALLEE, 7, false, false);
    assert_eq!(runs[2].0, runs[0].0, "an exit every 7 checks");
}

/// The live-object limit's RangeError at the end poll of an inlined frame's call node, after the
/// call returned (P3 stage 3 step 3c-2's review), by the DirectCall stub (`outerS`) and the
/// in-line sequence (`outerL`), and an error in the inlined frame's next node after the call
/// (`mid2`'s `o.k.z` of an `o` without `k`, and `mid3`'s, whose megamorphic reads are generic
/// nodes, so that its helper creates the error from the frame's published site): their stacks
/// name the inlined frame. A test-only
/// request makes the call's end poll find the ceiling passed (`jit::set_call_done_hook`,
/// `jit::arm_call_end_poll_limit`). The poll places no error, so the stack is what the capture
/// finds in `Interp::call_site`: at graph the inlined frame `mid`, pending (reported at its
/// start), and the compiled frame at the inlined call, the line the callee's own error's stack
/// gives it (the reference: `outerS(null)`'s TypeError, the tier off's). The tier off's template
/// code for `mid` has no end poll at its call (the request meets its next poll, only checked to
/// throw), and the splice's spliced body publishes nothing at that poll (its stack starts at the
/// compiled frame): printed, not asserted. The next node's error equals the tier off's.
#[test]
fn errors_after_an_inlined_frames_call_returned_name_the_inlined_frame() {
    let run = |switches: TestSwitches, any: bool| -> Vec<String> {
        switches.set();
        crate::jit::set_call_done_hook(true);
        let mut engine = jit_engine();
        engine
            .interp
            .gc_heap
            .core
            .borrow_mut()
            .set_incremental(crate::gc_core::IncrementalMode::Off);
        eval(&mut engine, DIRECT_CALLEE);
        let mut got: Vec<String> = ["stk2(outerS)", "stk2(outerL)"]
            .iter()
            .map(|trigger| {
                crate::jit::arm_call_end_poll_limit(any);
                let source = format!(
                    "function stk2(f) {{ try {{ return 'none:' + f({{x: 1}}, 1); }} catch (e) {{ return stk(e); }} }} {trigger}"
                );
                let got = eval(&mut engine, &source);
                assert!(
                    !crate::jit::take_end_poll_limit(crate::jit::POLL_HELPER_END),
                    "the request fell due and no poll took it: {got}"
                );
                got
            })
            .collect();
        got.push(eval(
            &mut engine,
            "try { outer2({x: 1}); } catch (e) { stk(e) }",
        ));
        got.push(eval(
            &mut engine,
            "try { outer3({x: 1, g: 1}); } catch (e) { stk(e) }",
        ));
        got.push(eval(
            &mut engine,
            "try { outerS(null); } catch (e) { stk(e) }",
        ));
        crate::jit::set_call_done_hook(false);
        got
    };
    let off = run(TestSwitches::default(), true);
    let splice = run(
        TestSwitches {
            inline_graph: Some(false),
            ..graph_switches(0)
        },
        false,
    );
    inlined_generic();
    let graph = run(graph_switches(0), false);
    let lines = |text: &str| -> Vec<String> {
        text.split_once('|')
            .map_or("", |(_, stack)| stack)
            .split('/')
            .map(str::to_string)
            .collect()
    };
    let reference = lines(&off[4]);
    let caller = reference
        .iter()
        .find(|line| line.starts_with("    at outerS ("))
        .unwrap_or_else(|| panic!("the reference: {}", off[4]));
    for k in 0..2 {
        assert!(
            off[k].starts_with("RangeError|"),
            "the tier off: {}",
            off[k]
        );
        assert!(
            splice[k].starts_with("RangeError|"),
            "splice: {}",
            splice[k]
        );
        let got = lines(&graph[k]);
        assert!(
            graph[k].starts_with("RangeError|RangeError: allocation limit exceeded/"),
            "graph: {}",
            graph[k]
        );
        assert!(
            got[1].starts_with("    at mid ("),
            "graph: {} (splice: {})",
            graph[k],
            splice[k]
        );
        let outer = if k == 0 {
            caller.clone()
        } else {
            caller.replace("outerS", "outerL")
        };
        assert_eq!(
            got[2].split(" (").next(),
            outer.split(" (").next(),
            "the compiled frame: {}",
            graph[k]
        );
    }
    assert_eq!(
        &got_line(&graph[0], 2),
        caller,
        "the compiled frame at the inlined call"
    );
    assert_eq!(graph[2], off[2], "the next node");
    assert_eq!(
        graph[3], off[3],
        "the next node, a generic one (megamorphic reads)"
    );
    assert_eq!(graph[4], off[4], "the reference");
    assert!(inlined_generic()
        .iter()
        .any(|op| matches!(op, Op::Call(..) | Op::CallWithThis(..))));
    println!("splice {splice:?}\noff {off:?}\ngraph {graph:?}");
}

/// Line `k` of the stack a `stk` result reports.
fn got_line(text: &str, k: usize) -> String {
    text.split_once('|')
        .map_or("", |(_, stack)| stack)
        .split('/')
        .nth(k)
        .unwrap_or("")
        .to_string()
}

/// A re-entrant call from an inlined frame's call node (P3 stage 3 step 3c-2's review): `innerR`,
/// which `midR`'s inlined frame calls, calls the compiled function `outerR` again, whose own
/// activation runs `midR` inlined, its call node publishing and consuming its own site; on the
/// way back each outer node's state is what its next node (`o.k[n].z`) and its end poll need.
const REENTRANT_CALLS: &str = "
var probe = false, log = [];
function stk(e) { return e.name + '|' + e.stack.split('\\n').slice(0, 6).join('/'); }
function innerR(o, n) {
  if (o === -1) {
    var y = 0;
    y = y * 3 + 0; if (y > 1e9) y = y % 7;
    y = y * 3 + 1; if (y > 1e9) y = y % 7;
    y = y * 3 + 2; if (y > 1e9) y = y % 7;
    y = y * 3 + 3; if (y > 1e9) y = y % 7;
    y = y * 3 + 4; if (y > 1e9) y = y % 7;
    y = y * 3 + 5; if (y > 1e9) y = y % 7;
    y = y * 3 + 6; if (y > 1e9) y = y % 7;
    y = y * 3 + 7; if (y > 1e9) y = y % 7;
    y = y * 3 + 8; if (y > 1e9) y = y % 7;
    y = y * 3 + 9; if (y > 1e9) y = y % 7;
    y = y * 3 + 10; if (y > 1e9) y = y % 7;
    y = y * 3 + 11; if (y > 1e9) y = y % 7;
    y = y * 3 + 12; if (y > 1e9) y = y % 7;
    y = y * 3 + 13; if (y > 1e9) y = y % 7;
    y = y * 3 + 14; if (y > 1e9) y = y % 7;
    y = y * 3 + 15; if (y > 1e9) y = y % 7;
    y = y * 3 + 16; if (y > 1e9) y = y % 7;
    y = y * 3 + 17; if (y > 1e9) y = y % 7;
    y = y * 3 + 18; if (y > 1e9) y = y % 7;
    y = y * 3 + 19; if (y > 1e9) y = y % 7;
    y = y * 3 + 20; if (y > 1e9) y = y % 7;
    y = y * 3 + 21; if (y > 1e9) y = y % 7;
    y = y * 3 + 22; if (y > 1e9) y = y % 7;
    y = y * 3 + 23; if (y > 1e9) y = y % 7;
    y = y * 3 + 24; if (y > 1e9) y = y % 7;
    y = y * 3 + 25; if (y > 1e9) y = y % 7;
    y = y * 3 + 26; if (y > 1e9) y = y % 7;
    y = y * 3 + 27; if (y > 1e9) y = y % 7;
    y = y * 3 + 28; if (y > 1e9) y = y % 7;
    y = y * 3 + 29; if (y > 1e9) y = y % 7;
    y = y * 3 + 30; if (y > 1e9) y = y % 7;
    y = y * 3 + 31; if (y > 1e9) y = y % 7;
    y = y * 3 + 32; if (y > 1e9) y = y % 7;
    y = y * 3 + 33; if (y > 1e9) y = y % 7;
    y = y * 3 + 34; if (y > 1e9) y = y % 7;
    y = y * 3 + 35; if (y > 1e9) y = y % 7;
    y = y * 3 + 36; if (y > 1e9) y = y % 7;
    y = y * 3 + 37; if (y > 1e9) y = y % 7;
    y = y * 3 + 38; if (y > 1e9) y = y % 7;
    y = y * 3 + 39; if (y > 1e9) y = y % 7;
    y = y * 3 + 40; if (y > 1e9) y = y % 7;
    y = y * 3 + 41; if (y > 1e9) y = y % 7;
    y = y * 3 + 42; if (y > 1e9) y = y % 7;
    y = y * 3 + 43; if (y > 1e9) y = y % 7;
    y = y * 3 + 44; if (y > 1e9) y = y % 7;
    y = y * 3 + 45; if (y > 1e9) y = y % 7;
    y = y * 3 + 46; if (y > 1e9) y = y % 7;
    y = y * 3 + 47; if (y > 1e9) y = y % 7;
    y = y * 3 + 48; if (y > 1e9) y = y % 7;
    y = y * 3 + 49; if (y > 1e9) y = y % 7;
    y = y * 3 + 50; if (y > 1e9) y = y % 7;
    y = y * 3 + 51; if (y > 1e9) y = y % 7;
    y = y * 3 + 52; if (y > 1e9) y = y % 7;
    y = y * 3 + 53; if (y > 1e9) y = y % 7;
    y = y * 3 + 54; if (y > 1e9) y = y % 7;
    y = y * 3 + 55; if (y > 1e9) y = y % 7;
    y = y * 3 + 56; if (y > 1e9) y = y % 7;
    y = y * 3 + 57; if (y > 1e9) y = y % 7;
    y = y * 3 + 58; if (y > 1e9) y = y % 7;
    y = y * 3 + 59; if (y > 1e9) y = y % 7;
    return y;
  }
  if (n > 0) { var v = outerR(o, n - 1); if (probe) log.push('back' + n + '|' + stk(new Error('b'))); return v + 1; }
  if (probe) log.push('bottom|' + stk(new Error('r')));
  return o.x;
}
function midR(o, n) { var r = innerR(o, n); return o.k[n].z + r; }
function outerR(o, n) { return midR(o, n); }
var full = {x: 1, k: [{z: 1}, {z: 1}, {z: 1}]}, r = 0;
for (var k = 0; k < 300; k++) r += outerR(full, 2) + outerR(full, k % 3);
probe = true; r += outerR(full, 2); probe = false;
var out = [];
try { outerR({x: 1, k: [{z: 1}, {z: 1}]}, 2); } catch (e) { out.push(stk(e)); }
r + '\\n' + log.join('\\n') + '\\n' + out.join('\\n')";

/// Two scripts for the active script's lookup (ECMA-262 GetActiveScriptOrModule;
/// `Interp::active_runtime_origin`): the compiled function and its inlined frame are script A's,
/// the callee the inlined frame calls script B's, and the callee's `import()` and an indirect
/// eval's `import()` resolve against B, as a stale inline site in `Interp::call_site` would make
/// them resolve against A (the site names A's function).
const ORIGIN_A: &str = "
function midO(o) { var r = innerO(o); return r; }
function outerOS(o) { return midO(o); }
function outerOL(o, n) { var r; for (var i = 0; i < n; i++) r = midO(o); return r; }";
const ORIGIN_B: &str = "
var probeO = false;
function innerO(o) {
  if (o === -1) {
    var y = 0;
    y = y * 3 + 0; if (y > 1e9) y = y % 7;
    y = y * 3 + 1; if (y > 1e9) y = y % 7;
    y = y * 3 + 2; if (y > 1e9) y = y % 7;
    y = y * 3 + 3; if (y > 1e9) y = y % 7;
    y = y * 3 + 4; if (y > 1e9) y = y % 7;
    y = y * 3 + 5; if (y > 1e9) y = y % 7;
    y = y * 3 + 6; if (y > 1e9) y = y % 7;
    y = y * 3 + 7; if (y > 1e9) y = y % 7;
    y = y * 3 + 8; if (y > 1e9) y = y % 7;
    y = y * 3 + 9; if (y > 1e9) y = y % 7;
    y = y * 3 + 10; if (y > 1e9) y = y % 7;
    y = y * 3 + 11; if (y > 1e9) y = y % 7;
    y = y * 3 + 12; if (y > 1e9) y = y % 7;
    y = y * 3 + 13; if (y > 1e9) y = y % 7;
    y = y * 3 + 14; if (y > 1e9) y = y % 7;
    y = y * 3 + 15; if (y > 1e9) y = y % 7;
    y = y * 3 + 16; if (y > 1e9) y = y % 7;
    y = y * 3 + 17; if (y > 1e9) y = y % 7;
    y = y * 3 + 18; if (y > 1e9) y = y % 7;
    y = y * 3 + 19; if (y > 1e9) y = y % 7;
    y = y * 3 + 20; if (y > 1e9) y = y % 7;
    y = y * 3 + 21; if (y > 1e9) y = y % 7;
    y = y * 3 + 22; if (y > 1e9) y = y % 7;
    y = y * 3 + 23; if (y > 1e9) y = y % 7;
    y = y * 3 + 24; if (y > 1e9) y = y % 7;
    y = y * 3 + 25; if (y > 1e9) y = y % 7;
    y = y * 3 + 26; if (y > 1e9) y = y % 7;
    y = y * 3 + 27; if (y > 1e9) y = y % 7;
    y = y * 3 + 28; if (y > 1e9) y = y % 7;
    y = y * 3 + 29; if (y > 1e9) y = y % 7;
    y = y * 3 + 30; if (y > 1e9) y = y % 7;
    y = y * 3 + 31; if (y > 1e9) y = y % 7;
    y = y * 3 + 32; if (y > 1e9) y = y % 7;
    y = y * 3 + 33; if (y > 1e9) y = y % 7;
    y = y * 3 + 34; if (y > 1e9) y = y % 7;
    y = y * 3 + 35; if (y > 1e9) y = y % 7;
    y = y * 3 + 36; if (y > 1e9) y = y % 7;
    y = y * 3 + 37; if (y > 1e9) y = y % 7;
    y = y * 3 + 38; if (y > 1e9) y = y % 7;
    y = y * 3 + 39; if (y > 1e9) y = y % 7;
    y = y * 3 + 40; if (y > 1e9) y = y % 7;
    y = y * 3 + 41; if (y > 1e9) y = y % 7;
    y = y * 3 + 42; if (y > 1e9) y = y % 7;
    y = y * 3 + 43; if (y > 1e9) y = y % 7;
    y = y * 3 + 44; if (y > 1e9) y = y % 7;
    y = y * 3 + 45; if (y > 1e9) y = y % 7;
    y = y * 3 + 46; if (y > 1e9) y = y % 7;
    y = y * 3 + 47; if (y > 1e9) y = y % 7;
    y = y * 3 + 48; if (y > 1e9) y = y % 7;
    y = y * 3 + 49; if (y > 1e9) y = y % 7;
    y = y * 3 + 50; if (y > 1e9) y = y % 7;
    y = y * 3 + 51; if (y > 1e9) y = y % 7;
    y = y * 3 + 52; if (y > 1e9) y = y % 7;
    y = y * 3 + 53; if (y > 1e9) y = y % 7;
    y = y * 3 + 54; if (y > 1e9) y = y % 7;
    y = y * 3 + 55; if (y > 1e9) y = y % 7;
    y = y * 3 + 56; if (y > 1e9) y = y % 7;
    y = y * 3 + 57; if (y > 1e9) y = y % 7;
    y = y * 3 + 58; if (y > 1e9) y = y % 7;
    y = y * 3 + 59; if (y > 1e9) y = y % 7;
    return y;
  }
  if (probeO) { import('./i.js'); (0, eval)(\"import('./e.js')\"); }
  return o.x;
}";
const ORIGIN_WARM: &str = "
for (var k = 0; k < 300; k++) { outerOS({x: 1}); outerOL({x: 1}, 2); }
probeO = true; outerOS({x: 1}); outerOL({x: 1}, 1); probeO = false;";

#[test]
fn a_reentrant_call_from_an_inlined_frame_keeps_each_activations_published_site() {
    inlined_generic();
    let runs = counted(REENTRANT_CALLS, 0, false, false);
    let expected = &runs[0].0;
    assert_eq!(&runs[1].0, expected, "splice");
    assert_eq!(&runs[2].0, expected, "graph");
    assert!(
        expected.contains("bottom|Error|Error: r/    at innerR ("),
        "{expected}"
    );
    assert!(expected.contains("TypeError|"), "{expected}");
    assert!(inlined_generic()
        .iter()
        .any(|op| matches!(op, Op::Call(..) | Op::CallWithThis(..))));
    let runs = counted(REENTRANT_CALLS, 7, false, false);
    assert_eq!(runs[2].0, runs[0].0, "an exit every 7 checks");
    // The end poll of the outermost activation's call node, the fifth call to complete (the
    // innermost midR's, innerR's of outerR, the middle midR's, innerR's again, then this one).
    let run = |switches: TestSwitches, any: bool| -> String {
        switches.set();
        crate::jit::set_call_done_hook(true);
        let mut engine = jit_engine();
        engine
            .interp
            .gc_heap
            .core
            .borrow_mut()
            .set_incremental(crate::gc_core::IncrementalMode::Off);
        eval(&mut engine, REENTRANT_CALLS);
        crate::jit::arm_call_end_poll_limit_at(5, any);
        let got = eval(
            &mut engine,
            "try { 'none:' + outerR(full, 2); } catch (e) { stk(e) }",
        );
        assert!(
            !crate::jit::take_end_poll_limit(crate::jit::POLL_HELPER_END),
            "the request fell due and no poll took it: {got}"
        );
        crate::jit::set_call_done_hook(false);
        got
    };
    let off = run(TestSwitches::default(), true);
    let graph = run(graph_switches(0), false);
    assert!(off.starts_with("RangeError|"), "the tier off: {off}");
    assert!(
        graph.starts_with("RangeError|RangeError: allocation limit exceeded/    at midR ("),
        "graph: {graph}"
    );
    // The outermost outerR's line, the call of midR: the innermost activation's stack has the
    // same call's (its third frame: innerR, midR, outerR).
    let bottom = expected
        .lines()
        .find(|line| line.starts_with("bottom|"))
        .unwrap_or_else(|| panic!("{expected}"));
    let reference = got_line(&format!("x|{bottom}"), 3);
    assert!(reference.starts_with("    at outerR ("), "{reference}");
    assert_eq!(got_line(&graph, 2), reference, "graph: {graph}");
    println!("off {off}\ngraph {graph}\nreference {reference}");
}

/// A direct call of an inlined frame's call node that declines at each of its gates (P3 stage 3
/// step 3c-2's review): the depth (`Interp::direct_call_depth`, a recursion past it), the callee's
/// code changed (its entry word moved, `Chunk::note_entry_change`) and the call caches' epoch
/// (`bytecode::invalidate_call_caches`). The helper that runs the call next still finds the
/// node's publication, so the callee's frame takes the call's site: the callee's own error's
/// stack and what its implicit callbacks see equal the tier off's and the splice's.
#[test]
fn a_declined_direct_call_from_an_inlined_frame_leaves_the_helper_the_published_site() {
    const TRIGGER: &str = "probe = true; log = [];
        var t = (function () { try { return 'none:' + outerS(null); } catch (e) { return stk(e); } })();
        probe = false; t + '|' + log.join(';')";
    const DEEP: &str = "function deepD(n) { return n > 0 ? deepD(n - 1) : outerS(null); }
        probe = true; log = [];
        var t = (function () { try { return 'none:' + deepD(140); } catch (e) { return stk(e); } })();
        probe = false; t + '|' + log.join(';')";
    let run = |switches: TestSwitches| -> Vec<String> {
        switches.set();
        let mut engine = jit_engine();
        eval(&mut engine, DIRECT_CALLEE);
        let mut got = vec![eval(&mut engine, DEEP)];
        chunk_of(&mut engine, "inner").note_entry_change();
        got.push(eval(&mut engine, TRIGGER));
        crate::bytecode::invalidate_call_caches();
        got.push(eval(&mut engine, TRIGGER));
        got
    };
    let off = run(TestSwitches::default());
    let splice = run(TestSwitches {
        inline_graph: Some(false),
        ..graph_switches(0)
    });
    inlined_generic();
    let graph = run(graph_switches(0));
    for k in 0..3 {
        assert!(off[k].starts_with("TypeError|"), "the tier off: {}", off[k]);
        assert!(
            off[k].contains("ts<inner") && off[k].contains("gg<inner"),
            "{}",
            off[k]
        );
        assert_eq!(splice[k], off[k], "splice, case {k}");
        assert_eq!(graph[k], off[k], "graph, case {k}");
    }
    assert!(inlined_generic()
        .iter()
        .any(|op| matches!(op, Op::Call(..) | Op::CallWithThis(..))));
}

#[test]
fn a_callee_an_inlined_frame_calls_imports_against_its_own_script() {
    let run = |switches: TestSwitches| -> Vec<String> {
        switches.set();
        let mut engine = jit_engine();
        let seen = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let log = seen.clone();
        engine.set_module_loader(move |specifier, referrer| {
            log.borrow_mut().push(format!("{specifier} <- {referrer}"));
            Some((
                format!("{referrer}#{specifier}"),
                "export default 1;".into(),
            ))
        });
        engine.set_import_base("https://a.test/a.js");
        eval(&mut engine, ORIGIN_A);
        engine.set_import_base("https://b.test/b.js");
        eval(&mut engine, ORIGIN_B);
        engine.set_import_base("https://c.test/c.js");
        eval(&mut engine, ORIGIN_WARM);
        engine.set_import_base("https://doc.test/page.html");
        engine.run_microtasks();
        let got = seen.borrow().clone();
        got
    };
    let off = run(TestSwitches::default());
    let splice = run(TestSwitches {
        inline_graph: Some(false),
        ..graph_switches(0)
    });
    inlined_generic();
    let graph = run(graph_switches(0));
    assert!(
        !off.is_empty() && off.iter().all(|seen| seen.ends_with("b.test/b.js")),
        "the tier off: {off:?}"
    );
    assert_eq!(splice, off, "splice");
    assert_eq!(graph, off, "graph");
    assert!(inlined_generic()
        .iter()
        .any(|op| matches!(op, Op::Call(..) | Op::CallWithThis(..))));
}

/// The rule for a way whose callee calls (P3 stage 3 step 3c-2, `InlineWay::graph_inlines`):
/// `midN` calls a small leaf the plan inlines in turn, so `outerN`'s plan splices (a second
/// stage, as at splice); `midP` calls a callee too large to plan, so `outerP` compiles in place
/// at graph with `midP` inlined and its call a call node of the inlined frame.
const NESTED_RULE: &str = "
function leafy(x) { return x + 1; }
function midN(x) { return leafy(x) * 2; }
function outerN(n) { var s = 0; for (var i = 0; i < n; i++) s += midN(i); return s; }
function bigc(x) {
  if (x === -1) {
    var y = 0;
    y = y * 3 + 0; if (y > 1e9) y = y % 7;
    y = y * 3 + 1; if (y > 1e9) y = y % 7;
    y = y * 3 + 2; if (y > 1e9) y = y % 7;
    y = y * 3 + 3; if (y > 1e9) y = y % 7;
    y = y * 3 + 4; if (y > 1e9) y = y % 7;
    y = y * 3 + 5; if (y > 1e9) y = y % 7;
    y = y * 3 + 6; if (y > 1e9) y = y % 7;
    y = y * 3 + 7; if (y > 1e9) y = y % 7;
    y = y * 3 + 8; if (y > 1e9) y = y % 7;
    y = y * 3 + 9; if (y > 1e9) y = y % 7;
    y = y * 3 + 10; if (y > 1e9) y = y % 7;
    y = y * 3 + 11; if (y > 1e9) y = y % 7;
    y = y * 3 + 12; if (y > 1e9) y = y % 7;
    y = y * 3 + 13; if (y > 1e9) y = y % 7;
    y = y * 3 + 14; if (y > 1e9) y = y % 7;
    y = y * 3 + 15; if (y > 1e9) y = y % 7;
    y = y * 3 + 16; if (y > 1e9) y = y % 7;
    y = y * 3 + 17; if (y > 1e9) y = y % 7;
    y = y * 3 + 18; if (y > 1e9) y = y % 7;
    y = y * 3 + 19; if (y > 1e9) y = y % 7;
    y = y * 3 + 20; if (y > 1e9) y = y % 7;
    y = y * 3 + 21; if (y > 1e9) y = y % 7;
    y = y * 3 + 22; if (y > 1e9) y = y % 7;
    y = y * 3 + 23; if (y > 1e9) y = y % 7;
    y = y * 3 + 24; if (y > 1e9) y = y % 7;
    y = y * 3 + 25; if (y > 1e9) y = y % 7;
    y = y * 3 + 26; if (y > 1e9) y = y % 7;
    y = y * 3 + 27; if (y > 1e9) y = y % 7;
    y = y * 3 + 28; if (y > 1e9) y = y % 7;
    y = y * 3 + 29; if (y > 1e9) y = y % 7;
    y = y * 3 + 30; if (y > 1e9) y = y % 7;
    y = y * 3 + 31; if (y > 1e9) y = y % 7;
    y = y * 3 + 32; if (y > 1e9) y = y % 7;
    y = y * 3 + 33; if (y > 1e9) y = y % 7;
    y = y * 3 + 34; if (y > 1e9) y = y % 7;
    y = y * 3 + 35; if (y > 1e9) y = y % 7;
    y = y * 3 + 36; if (y > 1e9) y = y % 7;
    y = y * 3 + 37; if (y > 1e9) y = y % 7;
    y = y * 3 + 38; if (y > 1e9) y = y % 7;
    y = y * 3 + 39; if (y > 1e9) y = y % 7;
    y = y * 3 + 40; if (y > 1e9) y = y % 7;
    y = y * 3 + 41; if (y > 1e9) y = y % 7;
    y = y * 3 + 42; if (y > 1e9) y = y % 7;
    y = y * 3 + 43; if (y > 1e9) y = y % 7;
    y = y * 3 + 44; if (y > 1e9) y = y % 7;
    y = y * 3 + 45; if (y > 1e9) y = y % 7;
    y = y * 3 + 46; if (y > 1e9) y = y % 7;
    y = y * 3 + 47; if (y > 1e9) y = y % 7;
    y = y * 3 + 48; if (y > 1e9) y = y % 7;
    y = y * 3 + 49; if (y > 1e9) y = y % 7;
    y = y * 3 + 50; if (y > 1e9) y = y % 7;
    y = y * 3 + 51; if (y > 1e9) y = y % 7;
    y = y * 3 + 52; if (y > 1e9) y = y % 7;
    y = y * 3 + 53; if (y > 1e9) y = y % 7;
    y = y * 3 + 54; if (y > 1e9) y = y % 7;
    y = y * 3 + 55; if (y > 1e9) y = y % 7;
    y = y * 3 + 56; if (y > 1e9) y = y % 7;
    y = y * 3 + 57; if (y > 1e9) y = y % 7;
    y = y * 3 + 58; if (y > 1e9) y = y % 7;
    y = y * 3 + 59; if (y > 1e9) y = y % 7;
    return y;
  }
  return x + 1;
}
function midP(x) { return bigc(x) * 2; }
function outerP(n) { var s = 0; for (var i = 0; i < n; i++) s += midP(i); return s; }
var r = 0; for (var k = 0; k < 300; k++) r += outerN(10) + outerP(10); r";

#[test]
fn a_way_whose_callees_calls_the_plan_inlines_keeps_its_plan_spliced() {
    inlines();
    inlined_generic();
    TestSwitches::default().set();
    let mut off = jit_engine();
    let expected = eval(&mut off, NESTED_RULE);
    graph_switches(0).set();
    let mut engine = jit_engine();
    assert_eq!(eval(&mut engine, NESTED_RULE), expected);
    let second_stage = |engine: &mut Engine, name: &str| {
        function(engine, name)
            .0
            .runtime()
            .code2
            .get()
            .is_some_and(Option::is_some)
    };
    assert!(second_stage(&mut engine, "outerN"), "outerN splices");
    assert!(
        !second_stage(&mut engine, "outerP"),
        "outerP compiles in place"
    );
    assert!(inlines()
        .iter()
        .any(|note| note.outcome == InlineOutcome::Inlined));
    assert!(inlined_generic()
        .iter()
        .any(|op| matches!(op, Op::Call(..) | Op::CallWithThis(..))));
}
