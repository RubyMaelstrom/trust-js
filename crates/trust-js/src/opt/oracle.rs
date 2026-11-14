//! The graph inliner's oracle (P3 stage 3, step 2; the design draft's D1): on the main thread, at a
//! compile's preparation, the inline decisions for the function and a snapshot of each inlined
//! callee, as SpiderMonkey's WarpOracle decides and snapshots each inlined script on the main
//! thread for WarpBuilder to build off it (jit/WarpOracle.cpp:1404-1554). TRust's feedback cells
//! are the main thread's (stage 2, step 12), so that a decision made on the compile thread, as
//! Maglev's and JSC's are, would need every cell atomic.
//!
//! - The decisions are the splice planner's (`bytecode::plan_inlines`: the same ways, so that the
//!   step measures the mechanism alone; the policy is step 8's): the plan the checkpoint made, or
//!   at a reoptimization a fresh one, as every exemplar decides at each compile (a plan holds
//!   unrooted function cells, which no chunk may keep). Kept of each site: the ways whose callees
//!   are leaves (no call or `new`), the splice's own emission's filters (at most eight arguments;
//!   no `this` from a plain call), recursion refused as the planner refuses it (D13).
//! - Each distinct callee first-stage chunk, once per job: its frozen view and raw copy in job
//!   memory, as the compiled chunk's (`Chunk::freeze_into`, `Raw::take_inlined_into`, which reads
//!   its exit profile as an inlined frame's).
//! - Each way: its guard's data appended to the compiled chunk's view's inline targets (its place
//!   in the site + 1 the guard's byte), its closure's environment and the records its callee's
//!   free names reach from it, folded as a splice's holders are; the callee's pin, which keeps
//!   the identity the guard embeds from reuse (`exit::Inlined::pins`).

use super::build::{InlineSite, InlineWay};
use super::job::{InlineJob, JobMemory};
use super::view::{FrozenChunk, InlineTargetView};
use crate::bytecode::{Chunk, InlinePlanEntry, Op};
use crate::fasthash::FastMap;
use crate::interpreter::{Env, Interp};
use std::rc::Rc;

/// What a compile's preparation asks the oracle for: the plan its checkpoint made (a first
/// compile), or a fresh plan of the function (a reoptimization).
pub(crate) enum Request {
    Plan(FastMap<u32, InlinePlanEntry>),
    Replan {
        func: Rc<crate::ast::Function>,
        /// The function's [[Realm]] (`UserCallable::realm`), as its call cache recorded it.
        realm: usize,
    },
}

/// What the main thread keeps of a compile's inlining until its install: the callee chunks, by
/// the oracle's index (`InlineWay::chunk`), and the pins of the callees the code embeds.
#[derive(Default)]
pub(crate) struct Held {
    pub(crate) chunks: Vec<Rc<Chunk>>,
    pub(crate) pins: Vec<crate::value::WeakGc>,
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
        Request::Replan { func, realm } => {
            crate::bytecode::plan_inlines(chunk, &func, realm, &i.global_env)
        }
    };
    if plan.is_empty() {
        return None;
    }
    let mut entries: Vec<(u32, InlinePlanEntry)> = plan.into_iter().collect();
    entries.sort_unstable_by_key(|&(cache, _)| cache);
    // Each call site's call: its argument count and whether it passes a receiver.
    let calls: FastMap<u32, (u16, bool)> = chunk
        .jit_ops()
        .iter()
        .filter_map(|op| match *op {
            Op::Call(argc, cache) => Some((cache, (argc, false))),
            Op::CallWithThis(argc, cache) => Some((cache, (argc, true))),
            _ => None,
        })
        .collect();
    let mut sites = Vec::new();
    let mut callees: Vec<JobMemory> = Vec::new();
    let mut held = Held::default();
    for (cache, entry) in entries {
        let Some(&(argc, with_this)) = calls.get(&cache) else {
            continue;
        };
        // The splice's emission declines a call of more than eight arguments
        // (`try_emit_inline`).
        if argc > 8 {
            continue;
        }
        let mut ways = Vec::new();
        for way in &entry.ways {
            // Leaves only (E0's callees); a plain call passes no `this` its callee could read.
            if !way.leaf || (!with_this && way.uses_this) {
                continue;
            }
            let Some(callee) = way.f.runtime().code.get().and_then(Option::clone) else {
                continue;
            };
            // Recursion, refused as the planner refuses it (D13; maglev-reducer-inl.h:2234-2237).
            if std::ptr::eq(&*callee, chunk) {
                continue;
            }
            let env = match &way.obj.borrow().kind {
                crate::value::ObjectKind::User(user) => user.env,
                _ => continue,
            };
            let index = match held
                .chunks
                .iter()
                .position(|held| Rc::ptr_eq(held, &callee))
            {
                Some(index) => index,
                None => {
                    let (mut callee_view, mut raw) = memory();
                    raw.take_inlined_into(&callee, Some(i));
                    callee.freeze_into(&mut callee_view);
                    #[cfg(debug_assertions)]
                    super::snapshot::check_decode(&callee, &raw, &callee_view);
                    callees.push((callee_view, raw));
                    held.chunks.push(callee);
                    held.chunks.len() - 1
                }
            };
            let mut holders: Vec<(u16, usize)> = Vec::new();
            for hops in callees[index].1.running_hops() {
                if !holders.iter().any(|&(at, _)| at == hops) {
                    if let Some(record) = super::snapshot::parent_record(env, hops) {
                        holders.push((hops, record));
                    }
                }
            }
            let target = view.inline_targets.len() as u32;
            view.inline_targets.push(InlineTargetView {
                argc,
                check_this: with_this && way.check_this,
                expected_env: way.expected_env,
                expected_genv: way.expected_genv,
                callee: Some(crate::bytecode::stored_word(way.obj)),
                env: Some(Env::as_ptr(&env) as usize),
                guard: (ways.len() + 1) as u8,
            });
            held.pins.push(crate::value::Gc::downgrade(&way.obj));
            ways.push(InlineWay {
                target,
                chunk: index as u32,
                env: Env::as_ptr(&env) as usize,
                holders: holders.into_boxed_slice(),
            });
        }
        if !ways.is_empty() {
            sites.push(InlineSite { cache, ways });
        }
    }
    (!sites.is_empty()).then_some((InlineJob { sites, callees }, held))
}
