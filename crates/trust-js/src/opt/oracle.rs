//! The graph inliner's oracle (P3 stage 3, step 2; the design draft's D1): on the main thread, at a
//! compile's preparation, the inline decisions for the function and a snapshot of each inlined
//! callee, as SpiderMonkey's WarpOracle decides and snapshots each inlined script on the main
//! thread for WarpBuilder to build off it (jit/WarpOracle.cpp:1404-1554). TRust's feedback cells
//! are the main thread's (stage 2, step 12), so that a decision made on the compile thread, as
//! Maglev's and JSC's are, would need every cell atomic.
//!
//! - The decisions are the splice planner's (`bytecode::plan_inlines`: the same ways, so that the
//!   step measures the mechanism alone; the policy is step 8's): the plan the checkpoint made, and
//!   at a reoptimization the ways that compile took (`ChunkProfile::inline_ways`, weak pins), as a
//!   second stage's reoptimization recompiles the splices its recompile made, so that a site that
//!   meets other callees later converges as a splice does, on its guard's mismatch edge. Kept of
//!   each site: the ways whose callees are leaves (no call or `new`), the splice's own emission's
//!   filters (at most eight arguments; no `this` from a plain call), recursion refused as the
//!   planner refuses it (D13).
//! - Each distinct callee first-stage chunk, once per job: its frozen view and raw copy in job
//!   memory, as the compiled chunk's (`Chunk::freeze_into`, `Raw::take_inlined_into`, which reads
//!   its exit profile as an inlined frame's).
//! - Each way: its guard's data appended to the compiled chunk's view's inline targets (its place
//!   in the site + 1 the guard's byte), its closure's environment and the records its callee's
//!   free names reach from it, folded as a splice's holders are; the callee's pin, which keeps
//!   the identity the guard embeds from reuse (`exit::Inlined::pins`).

use super::build::{InlineSite, InlineWay, LiteralWay};
use super::job::{InlineJob, JobMemory};
use super::profile::{ChunkProfile, KeptWay};
use super::view::{FrozenChunk, InlineTargetView};
use crate::bytecode::{Chunk, InlinePlanEntry, Op};
use crate::fasthash::FastMap;
use crate::interpreter::{Env, Interp};
use std::rc::Rc;

#[cfg(test)]
thread_local! {
    /// Why the oracle made no by-literal way for a candidate, since the tests last took them.
    pub(crate) static TEST_LITERAL_REFUSALS: std::cell::RefCell<Vec<&'static str>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// What a compile's preparation asks the oracle for: the plan its checkpoint made (a first
/// compile), or the ways that compile took (a reoptimization: `ChunkProfile::inline_ways`).
pub(crate) enum Request {
    Plan(FastMap<u32, InlinePlanEntry>),
    Kept,
}

/// What the main thread keeps of a compile's inlining until its install: the callee chunks, by
/// the oracle's index (`InlineWay::chunk`), the pins of the callees the code embeds, and the
/// ways' inline sites (`InlineWay::position`, P3 stage 3 step 3c), registered in the thread's
/// table and released with the code or with a job dropped before its install.
#[derive(Default)]
pub(crate) struct Held {
    pub(crate) chunks: Vec<Rc<Chunk>>,
    pub(crate) pins: Vec<crate::value::WeakGc>,
    pub(crate) sites: crate::stack_trace::InlineSites,
}

impl Held {
    /// What the code holds (`exit::Inlined`): the chunks its inlined frames run, by
    /// `InlinedFrame::chunk` (`frame_chunks`, the oracle's indices), and every pin.
    pub(crate) fn into_inlined(self, frame_chunks: &[u32]) -> super::exit::Inlined {
        super::exit::Inlined {
            chunks: frame_chunks
                .iter()
                .map(|&chunk| self.chunks[chunk as usize].clone())
                .collect(),
            pins: self.pins.into_boxed_slice(),
            sites: self.sites,
        }
    }

    /// The revisions of the callee chunks' exit profiles (an install drops a job a discard
    /// changed one of since its snapshot, as a second stage's record map's chunks are counted).
    pub(crate) fn revisions(&self) -> u64 {
        self.chunks
            .iter()
            .map(|chunk| {
                super::profile::ChunkProfile::get(chunk).map_or(0, |p| u64::from(p.revision()))
            })
            .sum()
    }
}

/// The oracle for a compile of `chunk` (see the module's doc), its frozen view `view`, whose
/// inline targets the ways' guards are appended to; `memory` gives a job memory for each callee
/// chunk's view and raw copy. None where nothing is inlined.
pub(crate) fn prepare(
    i: &Interp,
    chunk: &Chunk,
    view: &mut FrozenChunk,
    request: Request,
    mut memory: impl FnMut() -> JobMemory,
) -> Option<(InlineJob, Held)> {
    let plan = match request {
        Request::Plan(plan) => plan,
        Request::Kept => kept_plan(chunk),
    };
    if plan.is_empty() {
        return None;
    }
    // What the planner's budget leaves for the closure literals this chunk creates (P3 stage 3,
    // piece B), where it creates any.
    let mut budget = chunk
        .jit_ops()
        .iter()
        .any(|op| matches!(op, Op::MakeClosure(..)))
        .then(|| crate::bytecode::InlineBudgetLeft::after(chunk, &plan));
    let compiled_cfg = budget
        .as_ref()
        .and_then(|_| crate::jit_ir::Cfg::build(chunk).ok());
    let mut entries: Vec<(u32, InlinePlanEntry)> = plan.into_iter().collect();
    entries.sort_unstable_by_key(|&(cache, _)| cache);
    // Each call site's call: its argument count, whether it passes a receiver, and its pc.
    let calls: FastMap<u32, (u16, bool, usize)> = chunk
        .jit_ops()
        .iter()
        .enumerate()
        .filter_map(|(pc, op)| match *op {
            Op::Call(argc, cache) => Some((cache, (argc, false, pc))),
            Op::CallWithThis(argc, cache) => Some((cache, (argc, true, pc))),
            _ => None,
        })
        .collect();
    let mut sites = Vec::new();
    let mut callees: Vec<JobMemory> = Vec::new();
    let mut held = Held::default();
    let mut kept: Vec<KeptWay> = Vec::new();
    // Each way's inline site, in the order of the ways (`InlineWay::position`): the callee, no
    // position of its own (its frame's operations place a thrown error, `Interp::place_thrown`,
    // and a callback's frame reports its start, as a callee entered implicitly does), and the
    // call's position in this chunk, as a splice's body site (bytecode.rs's `inline_position`
    // with NO_POSITION); registered here, on the main thread, since the code embeds them.
    let mut inline_sites: Vec<crate::stack_trace::InlineSite> = Vec::new();
    // A non-leaf way's calls (P3 stage 3 step 3c-2): (its view's index, the callee's call cache,
    // the call's site among `inline_sites`), each call's position in the way's own view.
    let mut call_sites: Vec<(usize, u32, usize)> = Vec::new();
    // Its plain `new`s likewise (P3 stage 3 step 3e): (the view, the construct cache, the site).
    let mut construct_sites: Vec<(usize, u32, usize)> = Vec::new();
    for (cache, entry) in entries {
        let Some(&(argc, with_this, pc)) = calls.get(&cache) else {
            continue;
        };
        // The splice's emission declines a call of more than eight arguments
        // (`try_emit_inline`).
        if argc > 8 {
            continue;
        }
        // The closure literals the call passes, by argument (piece B): an argument word an
        // Op::MakeClosure of this chunk pushed.
        let literal_args: Vec<Option<u32>> = match (&budget, &compiled_cfg) {
            (Some(_), Some(cfg)) => (0..usize::from(argc))
                .map(|a| {
                    let at = producer(chunk, cfg, pc, usize::from(argc) - 1 - a)?;
                    match chunk.jit_ops()[at] {
                        Op::MakeClosure(literal, _) => Some(literal),
                        _ => None,
                    }
                })
                .collect(),
            _ => Vec::new(),
        };
        let mut ways = Vec::new();
        for way in &entry.ways {
            let mut prep = Prep {
                literal_args: &literal_args,
                budget: &mut budget,
                i,
                compiled: chunk,
                view,
                held: &mut held,
                callees: &mut callees,
                inline_sites: &mut inline_sites,
                call_sites: &mut call_sites,
                construct_sites: &mut construct_sites,
                memory: &mut memory,
            };
            // The way's frame is the compiled frame's child: its site's caller position is this
            // chunk's call's, never an inline position (a nested frame's chains its caller's).
            debug_assert!(!crate::stack_trace::is_inline_position(
                chunk.call_position(cache)
            ));
            let site = Call {
                cache,
                argc,
                with_this,
                place: ways.len(),
                caller_position: chunk.call_position(cache),
            };
            if let Some((taken, kept_way)) = prep.way(way, site, &[]) {
                ways.push(taken);
                kept.push(kept_way);
            }
        }
        if !ways.is_empty() {
            sites.push(InlineSite { cache, ways });
        }
    }
    let (registered, positions) = crate::stack_trace::InlineSites::register(inline_sites);
    held.sites = registered;
    for (view, call, site) in call_sites {
        if let Some(position) = callees[view].0.call_positions.get_mut(call as usize) {
            *position = positions[site];
        }
    }
    for (view, construct, site) in construct_sites {
        if let Some(position) = callees[view]
            .0
            .construct_positions
            .get_mut(construct as usize)
        {
            *position = positions[site];
        }
    }
    fn place(sites: &mut [InlineSite], positions: &[u32]) {
        for way in sites.iter_mut().flat_map(|site| site.ways.iter_mut()) {
            way.position = positions[way.position as usize];
            for literal in &mut way.literals {
                literal.position = positions[literal.position as usize];
            }
            // The frames nested in it (step 3d).
            place(&mut way.nested, positions);
        }
    }
    place(&mut sites, &positions);
    // The ways a reoptimization takes again (see the module's doc).
    ChunkProfile::of(chunk).set_inline_ways(kept);
    (!sites.is_empty()).then_some((InlineJob { sites, callees }, held))
}

/// A call site whose ways `Prep::way` takes (P3 stage 3 step 3d): its cache in its caller's
/// chunk, its call's argument count and receiver form, the way's place among the site's taken
/// ways, and the caller's position at the call: the compiled chunk's call's byte offset, or the
/// inline position (this job's list's index, tagged) of the caller frame's call site.
#[derive(Clone, Copy)]
struct Call {
    cache: u32,
    argc: u16,
    with_this: bool,
    place: usize,
    caller_position: u32,
}

/// What `prepare` grows while it takes ways (P3 stage 3 step 3d: at any depth).
struct Prep<'p, 'c, M: FnMut() -> JobMemory> {
    i: &'p Interp,
    /// The compiled chunk's closure literals its call passes, by argument (piece B).
    literal_args: &'p [Option<u32>],
    /// What the planner's budget leaves for them (`InlineBudgetLeft`), where the chunk creates
    /// any.
    budget: &'p mut Option<crate::bytecode::InlineBudgetLeft>,
    compiled: &'p Chunk,
    view: &'p mut FrozenChunk,
    held: &'p mut Held,
    callees: &'p mut Vec<JobMemory>,
    inline_sites: &'p mut Vec<crate::stack_trace::InlineSite>,
    call_sites: &'p mut Vec<(usize, u32, usize)>,
    construct_sites: &'p mut Vec<(usize, u32, usize)>,
    memory: &'c mut M,
}

impl<M: FnMut() -> JobMemory> Prep<'_, '_, M> {
    /// Way `way` of `call` (its callee's chain of callers `chain`, the compiled chunk excluded),
    /// as the graph builds it: its callee chunk's view (a leaf's shared, a calling way's its own,
    /// whose calls' positions are its sites), its guard's data (the compiled view's inline
    /// targets, a nested way's too), its pin, its inline site (the callee, no position of its
    /// own, the caller's position at the call), a site per call of its callee (piece B; its outer
    /// the same caller position), and the ways the plan inlines at its callee's own calls in turn
    /// (step 3d), each nested in its frame, their caller position its call's site; None for a way
    /// the graph does not take.
    fn way(
        &mut self,
        way: &crate::bytecode::InlineWay,
        call: Call,
        chain: &[*const Chunk],
    ) -> Option<(InlineWay, KeptWay)> {
        // Leaves, and callees making plain and method calls (`InlineWay::graph_inlines`); a
        // plain call passes no `this` its callee could read; a nested guard reads no caller
        // environment (the verifier's frame-free guard).
        if !way.graph_inlines()
            || (!call.with_this && way.uses_this)
            || (!chain.is_empty() && way.expected_env != 0)
        {
            return None;
        }
        let callee = way.f.runtime().code.get().and_then(Option::clone)?;
        // Recursion, refused as the planner refuses it (D13; maglev-reducer-inl.h:2234-2237): a
        // callee that is its caller's chunk. A recursion through another callee the planner nests
        // to its depth, and so does the graph (a frame of the compiled chunk's own in another's).
        let caller = chain
            .last()
            .copied()
            .unwrap_or(self.compiled as *const Chunk);
        if std::ptr::eq(&*callee, caller) {
            return None;
        }
        let env = match &way.obj.borrow().kind {
            crate::value::ObjectKind::User(user) => user.env,
            _ => return None,
        };
        // A way whose callee calls gets a view of its own: its calls' positions are the way's
        // inline sites (below), which another way of the same callee does not share.
        let shared = self
            .held
            .chunks
            .iter()
            .position(|held| Rc::ptr_eq(held, &callee))
            .filter(|_| way.leaf);
        let index = match shared {
            Some(index) => index,
            None => {
                let (mut callee_view, mut raw) = (self.memory)();
                raw.take_inlined_into(&callee, Some(self.i));
                callee.freeze_into(&mut callee_view);
                #[cfg(debug_assertions)]
                super::snapshot::check_decode(&callee, &raw, &callee_view);
                self.callees.push((callee_view, raw));
                self.held.chunks.push(callee.clone());
                self.held.chunks.len() - 1
            }
        };
        let mut holders: Vec<(u16, usize)> = Vec::new();
        for hops in self.callees[index].1.running_hops() {
            if !holders.iter().any(|&(at, _)| at == hops) {
                if let Some(record) = super::snapshot::parent_record(env, hops) {
                    holders.push((hops, record));
                }
            }
        }
        let target = self.view.inline_targets.len() as u32;
        self.view.inline_targets.push(InlineTargetView {
            argc: call.argc,
            check_this: call.with_this && way.check_this,
            expected_env: way.expected_env,
            expected_genv: way.expected_genv,
            callee: Some(crate::bytecode::stored_word(way.obj)),
            env: Some(Env::as_ptr(&env) as usize),
            guard: (call.place + 1) as u8,
        });
        self.held.pins.push(crate::value::Gc::downgrade(&way.obj));
        self.inline_sites.push(crate::stack_trace::InlineSite {
            callee: crate::stack_trace::InlineCallee::of(&way.obj),
            inner: crate::stack_trace::NO_POSITION,
            outer: call.caller_position,
        });
        let way_site = self.inline_sites.len() - 1;
        // Each call the callee makes, from its inlined frame (step 3c-2): an inline site of its
        // own, the callee at the call's position in the callee's source and the caller's
        // position at the way's call, as a spliced body's call carries; the way's view gives it
        // to the call's template and its node publishes it; by cache, for the nested ways.
        let mut own_calls: FastMap<u32, (u16, bool, usize)> = FastMap::default();
        // The calls of the callee's formals the compiled chunk's literals are bound to (piece B):
        // (the call's cache, its argument count and receiver form, its site, the literal).
        let mut literal_calls: Vec<(u32, u16, bool, usize, u32)> = Vec::new();
        let callee_cfg = (chain.is_empty() && self.literal_args.iter().any(Option::is_some))
            .then(|| crate::jit_ir::Cfg::build(&callee).ok())
            .flatten();
        if !way.leaf {
            for (callee_pc, op) in callee.jit_ops().iter().enumerate() {
                let (argc, inner_call, with_this) = match *op {
                    Op::Call(argc, inner_call) => (argc, inner_call, false),
                    Op::CallWithThis(argc, inner_call) => (argc, inner_call, true),
                    // A plain `new` (step 3e): a site of its own, as a call's, the `new`'s
                    // position in the callee's source; its node's construct cache is the view's,
                    // the callee chunk's own cell. No way is planned at a construct site.
                    Op::New(_, construct) => {
                        let inner = callee.construct_position(construct);
                        debug_assert!(!crate::stack_trace::is_inline_position(inner));
                        self.inline_sites.push(crate::stack_trace::InlineSite {
                            callee: crate::stack_trace::InlineCallee::of(&way.obj),
                            inner,
                            outer: call.caller_position,
                        });
                        self.construct_sites
                            .push((index, construct, self.inline_sites.len() - 1));
                        continue;
                    }
                    _ => continue,
                };
                let inner = callee.call_position(inner_call);
                debug_assert!(!crate::stack_trace::is_inline_position(inner));
                self.inline_sites.push(crate::stack_trace::InlineSite {
                    callee: crate::stack_trace::InlineCallee::of(&way.obj),
                    inner,
                    outer: call.caller_position,
                });
                let site = self.inline_sites.len() - 1;
                self.call_sites.push((index, inner_call, site));
                own_calls.insert(inner_call, (argc, with_this, site));
                // A call whose callee word is a formal the compiled frame's call binds to one
                // of its closure literals (piece B: the compiled frame's own calls only).
                if let Some(cfg) = &callee_cfg {
                    // The callee word lies under the arguments, the receiver under it.
                    let from_top = usize::from(argc);
                    if let Some(Op::LoadLocal(formal)) =
                        producer(&callee, cfg, callee_pc, from_top).map(|at| callee.jit_ops()[at])
                    {
                        if usize::from(formal) < callee.jit_frame().0 {
                            if let Some(&Some(literal)) = self.literal_args.get(usize::from(formal))
                            {
                                literal_calls.push((inner_call, argc, with_this, site, literal));
                            }
                        }
                    }
                }
            }
        }
        // The callee's own calls the plan inlines in turn (step 3d): each a site of the way's
        // frame, its ways' guards in it, each way a frame nested in it, whose caller position is
        // its call's site (an inline position of this job's list, which `InlineSites::register`
        // maps), to the planner's depth.
        let mut nested_sites: Vec<InlineSite> = Vec::new();
        let mut nested_kept: Vec<KeptWay> = Vec::new();
        let mut entries: Vec<(&u32, &InlinePlanEntry)> = way.nested.iter().collect();
        entries.sort_unstable_by_key(|&(inner_call, _)| *inner_call);
        let mut inner_chain = chain.to_vec();
        inner_chain.push(Rc::as_ptr(&callee));
        for (&inner_call, entry) in entries {
            let Some(&(argc, with_this, site)) = own_calls.get(&inner_call) else {
                continue;
            };
            if argc > 8 {
                continue;
            }
            let mut inner_ways = Vec::new();
            for inner_way in &entry.ways {
                let inner = Call {
                    cache: inner_call,
                    argc,
                    with_this,
                    place: inner_ways.len(),
                    caller_position: crate::stack_trace::inline_position(site as u32),
                };
                if let Some((taken, kept_way)) = self.way(inner_way, inner, &inner_chain) {
                    inner_ways.push(taken);
                    nested_kept.push(kept_way);
                }
            }
            if !inner_ways.is_empty() {
                nested_sites.push(InlineSite {
                    cache: inner_call,
                    ways: inner_ways,
                });
            }
        }
        let mut literals = Vec::new();
        for (inner_call, argc, with_this, site, literal) in literal_calls {
            let in_loop = crate::bytecode::call_site_in_loop(self.compiled, call.cache)
                || crate::bytecode::call_site_in_loop(&callee, inner_call);
            if let Some(literal_way) = self.literal(
                way, &callee, inner_call, argc, with_this, site, literal, in_loop,
            ) {
                literals.push(literal_way);
            }
        }
        Some((
            InlineWay {
                target,
                callee: crate::bytecode::stored_word(way.obj),
                literals,
                chunk: index as u32,
                env: Env::as_ptr(&env) as usize,
                holders: holders.into_boxed_slice(),
                // Its site's index among this job's, until they are registered.
                position: way_site as u32,
                nested: nested_sites,
            },
            KeptWay {
                cache: call.cache,
                callee: crate::value::Gc::downgrade(&way.obj),
                check_this: way.check_this,
                uses_this: way.uses_this,
                leaf: way.leaf,
                expected_genv: way.expected_genv,
                genv_pin: way.genv_pin.clone(),
                nested: nested_kept,
            },
        ))
    }
}

impl<M: FnMut() -> JobMemory> Prep<'_, '_, M> {
    /// The by-literal way of closure literal `literal` of the compiled chunk at call `inner_call`
    /// of `w`'s callee chunk `w_chunk` (argument count `argc`, a receiver if `with_this`, its
    /// inline site `site` among this job's, in a loop if `in_loop`), whose callee word is a
    /// formal the compiled frame's call binds to the literal (P3 stage 3, piece B; the design
    /// note's section 1): the planner's rules for the literal as a callee of `w`'s function
    /// (`inline_callee_rules`) and its budget after the plan, the oracle's own filters for any
    /// way; its chunk's view and snapshot as a way's, its frame's inline site and one per call
    /// and plain `new` of its chunk, each naming the frame's callee word
    /// (`InlineCallee::Frame`). No guard, pin, environment or holders: the frame's callee is the
    /// creation's node, its environment the creation's (`build_inline`).
    #[allow(clippy::too_many_arguments)]
    fn literal(
        &mut self,
        w: &crate::bytecode::InlineWay,
        w_chunk: &Chunk,
        inner_call: u32,
        argc: u16,
        with_this: bool,
        site: usize,
        literal: u32,
        in_loop: bool,
    ) -> Option<LiteralWay> {
        let f = self.compiled.function_literal(literal)?.clone();
        let realm = match &w.obj.borrow().kind {
            crate::value::ObjectKind::User(user) => user.realm,
            _ => return None,
        };
        let n_params = u16::try_from(f.params.len()).ok()?;
        let refused = |why: &'static str| {
            #[cfg(test)]
            TEST_LITERAL_REFUSALS.with(|refusals| refusals.borrow_mut().push(why));
            if super::log_enabled() {
                eprintln!(
                    "[opt] no by-literal way for literal {literal} at call {inner_call}: {why}"
                );
            }
        };
        let chunk = match crate::bytecode::inline_callee_rules(
            w_chunk,
            &w.f,
            realm,
            realm,
            &f,
            n_params,
            crate::bytecode::inline_max_ops(),
        ) {
            Ok(chunk) => chunk,
            Err(why) => {
                refused(why);
                return None;
            }
        };
        // Recursion (D13): the literal's chunk is the compiled chunk's.
        if std::ptr::eq(&*chunk, self.compiled) || argc > 8 {
            return None;
        }
        // The oracle's filters for any way (`way`'s): the callee's calls plain or method calls
        // or plain `new`s; a `this` it reads passed by the call and, where it is sloppy, tested
        // by a guard, which a by-literal way has none of.
        let leaf = !chunk.jit_ops().iter().any(|op| {
            matches!(
                op,
                Op::Call(..)
                    | Op::CallWithThis(..)
                    | Op::CallApplyFrameArgs(..)
                    | Op::CallSpread(..)
                    | Op::CallSpreadThis(..)
                    | Op::CallArgsArray
                    | Op::CallArgsArrayThis
                    | Op::New(..)
                    | Op::NewArgsArray
            )
        });
        if !leaf && !crate::bytecode::makes_plain_calls_only(&chunk) {
            refused("a call other than a plain or method call or a plain `new`");
            return None;
        }
        if chunk.uses_this() && (!with_this || !f.is_strict) {
            refused("a `this` no guard tests");
            return None;
        }
        if !self
            .budget
            .as_mut()
            .is_some_and(|budget| budget.take(chunk.jit_ops().len(), leaf, in_loop))
        {
            refused("the planner's budget");
            return None;
        }
        let shared = self
            .held
            .chunks
            .iter()
            .position(|held| Rc::ptr_eq(held, &chunk))
            .filter(|_| leaf);
        let index = match shared {
            Some(index) => index,
            None => {
                let (mut view, mut raw) = (self.memory)();
                raw.take_inlined_into(&chunk, Some(self.i));
                chunk.freeze_into(&mut view);
                #[cfg(debug_assertions)]
                super::snapshot::check_decode(&chunk, &raw, &view);
                self.callees.push((view, raw));
                self.held.chunks.push(chunk.clone());
                self.held.chunks.len() - 1
            }
        };
        let outer = crate::stack_trace::inline_position(site as u32);
        self.inline_sites.push(crate::stack_trace::InlineSite {
            callee: crate::stack_trace::InlineCallee::Frame,
            inner: crate::stack_trace::NO_POSITION,
            outer,
        });
        let frame_site = self.inline_sites.len() - 1;
        for op in chunk.jit_ops() {
            match *op {
                Op::Call(_, own) | Op::CallWithThis(_, own) => {
                    self.inline_sites.push(crate::stack_trace::InlineSite {
                        callee: crate::stack_trace::InlineCallee::Frame,
                        inner: chunk.call_position(own),
                        outer,
                    });
                    self.call_sites
                        .push((index, own, self.inline_sites.len() - 1));
                }
                Op::New(_, construct) => {
                    self.inline_sites.push(crate::stack_trace::InlineSite {
                        callee: crate::stack_trace::InlineCallee::Frame,
                        inner: chunk.construct_position(construct),
                        outer,
                    });
                    self.construct_sites
                        .push((index, construct, self.inline_sites.len() - 1));
                }
                _ => {}
            }
        }
        // The interpreter's frame-callee table, made here on the main thread (its first compile
        // with a by-literal way).
        let interp = self.i as *const Interp as usize;
        let table = crate::stack_trace::frame_callee_table(interp, true)?;
        Some(LiteralWay {
            cache: inner_call,
            literal,
            chunk: index as u32,
            position: frame_site as u32,
            self_name: f.needs_self_environment(),
            interp,
            table,
        })
    }
}

/// The operation whose push is the operand word `from_top` words below the top of the stack
/// before the operation at `pc` of `chunk` (0: the top): a backward walk of its operations' stack
/// effects through the blocks of `cfg` (`chunk`'s), from a block's start into its one
/// predecessor (a `return f(...)`'s call is reached from its tail-call test's branch), stopping
/// at a join (P3 stage 3, piece B). The builder decides by the node, so a walk that named
/// another producer than the one that ran would cost a prepared way that is never built.
fn producer(
    chunk: &Chunk,
    cfg: &crate::jit_ir::Cfg,
    pc: usize,
    mut from_top: usize,
) -> Option<usize> {
    let blocks = cfg.blocks();
    let mut block = &blocks[cfg.block_at(pc)?.0 as usize];
    let mut at = pc;
    // A bound on the blocks walked (a cycle of single predecessors is unreachable code).
    for _ in 0..blocks.len() {
        while at > block.start {
            at -= 1;
            let (pops, pushes) = chunk.jit_stack_effect(at)?;
            if from_top < pushes {
                return Some(at);
            }
            from_top = from_top - pushes + pops;
        }
        let [only] = block.predecessors[..] else {
            return None;
        };
        block = &blocks[only.0 as usize];
        at = block.end;
    }
    None
}

/// The plan of the ways `chunk`'s first compile took (`ChunkProfile::inline_ways`), in their
/// sites and order, each while its callee lives: a reoptimization's, as a second stage's
/// reoptimization recompiles the splices its recompile made (stage 2, step 3's departure 1).
fn kept_plan(chunk: &Chunk) -> FastMap<u32, InlinePlanEntry> {
    let Some(profile) = ChunkProfile::get(chunk) else {
        return FastMap::default();
    };
    kept_ways(profile.inline_ways())
}

/// The plan of `ways`, by their sites, each with the ways kept at its callee's own sites (P3
/// stage 3 step 3d), while its callee lives.
fn kept_ways(ways: Vec<KeptWay>) -> FastMap<u32, InlinePlanEntry> {
    let mut plan: FastMap<u32, InlinePlanEntry> = FastMap::default();
    for way in ways {
        let Some(obj) = way.callee.upgrade() else {
            continue;
        };
        let f = match &obj.borrow().kind {
            crate::value::ObjectKind::User(user) => Rc::clone(&user.func),
            _ => continue,
        };
        plan.entry(way.cache)
            .or_insert_with(|| InlinePlanEntry { ways: Vec::new() })
            .ways
            .push(crate::bytecode::InlineWay {
                f,
                obj,
                check_this: way.check_this,
                uses_this: way.uses_this,
                leaf: way.leaf,
                free_names: Vec::new(),
                expected_env: 0,
                expected_genv: way.expected_genv,
                genv_pin: way.genv_pin,
                nested: kept_ways(way.nested),
            });
    }
    plan
}
