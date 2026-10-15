//! Bounded opt-in policy for an observed native part of a large ordinary body.
//!
//! ECMA-262 e28783d5 Execution Contexts / GetValue / TryStatement: coverage is
//! workload feedback, never a semantic dead-code proof. Original analyses,
//! handlers, owning slots and exact terminal before-effect VM recovery survive.
//! The opportunity sum is a policy heuristic across observed native branches;
//! it is NOT a minimum saved per invocation or a measured repayment claim.

use super::{frame, observed_regions, stack};
use crate::{
    bytecode::{Chunk, Op},
    jit_ir::Cfg,
};

pub(super) const MAX_OPS: usize = 4096;
pub(super) const MAX_BLOCKS: usize = MAX_OPS + 1;
// Provisional opt-in policy coefficient. Release compile and application results
// must validate repayment before promotion; this does not waive any semantic guard.
const REPAYMENT_WORK: u64 = 32;

pub(super) fn enabled() -> bool {
    std::env::var("TRUST_JS_OPT_JIT_LARGE_REGIONS").as_deref() == Ok("1")
        && observed_regions::enabled()
        && crate::execution_coverage::enabled()
}

pub(super) fn source_size(ops: usize) -> bool {
    enabled() && (257..=MAX_OPS).contains(&ops)
}

pub(super) fn unsupported(chunk: &Chunk) -> bool {
    chunk.jit_ops().iter().any(|op| {
        matches!(
            op,
            Op::FragmentExit(_)
                | Op::Await
                | Op::Yield
                | Op::YieldStar
                | Op::AsyncIterStepL(..)
                | Op::AsyncIterResumeL(..)
                | Op::AsyncIterCloseL(..)
                | Op::DisposeNormal
                | Op::DisposeThrow
                | Op::DisposeReturn
                | Op::DisposeBareReturn
                | Op::DisposeResumeReturn
                | Op::DisposeJump
        )
    })
}

/// This only allows bounded baseline sampling. It cannot publish a large body;
/// a later checkpoint and candidate preparation need the same frozen certificate.
pub(super) fn sampling_candidate(chunk: &Chunk) -> bool {
    source_size(chunk.jit_ops().len())
        && chunk.execution_coverage.source().is_some()
        && crate::bytecode::native_deopt::supported(chunk)
        && !unsupported(chunk)
}

pub(super) fn certificate(
    chunk: &Chunk,
    cfg: &Cfg,
    regions: Option<&observed_regions::Plan>,
) -> bool {
    source_size(chunk.jit_ops().len())
        && cfg.blocks().len() <= MAX_BLOCKS
        && !unsupported(chunk)
        && regions.is_some_and(|regions| {
            regions.cut_blocks != 0
                && chunk.execution_coverage.source_version() == Some(regions.source_version)
        })
}

pub(super) struct Work {
    pub(super) native_expansion: u64,
    pub(super) original_labels: u64,
    pub(super) original_frame: u64,
    pub(super) cold_recovery: u64,
    pub(super) call_imports: u64,
    pub(super) literal_analysis: u64,
    pub(super) literal_dispatch_lowering: u64,
    pub(super) borrowed_operands: u64,
}

impl Work {
    pub(super) fn total(&self) -> Option<u64> {
        self.native_expansion
            .checked_add(self.original_labels)?
            .checked_add(self.original_frame)?
            .checked_add(self.cold_recovery)?
            .checked_add(self.call_imports)?
            .checked_add(self.literal_analysis)?
            .checked_add(self.literal_dispatch_lowering)?
            .checked_add(self.borrowed_operands)
    }

    pub(super) fn admission_entries(&self, opportunities: u32) -> Option<u64> {
        if opportunities == 0 {
            return None;
        }
        Some(
            self.total()?
                .checked_mul(REPAYMENT_WORK)?
                .div_ceil(u64::from(opportunities))
                .max(4096),
        )
    }
}

/// Prices both retained native expansion and full-source scaffold/analysis work.
/// It predicts policy work, not actual IR. All successful candidates still pass
/// the actual unchanged16K instruction/65536 block checks before publication.
pub(super) fn work(
    chunk: &Chunk,
    cfg: &Cfg,
    stack: &stack::StackPlan,
    frame: &frame::FramePlan,
    regions: &observed_regions::Plan,
) -> Option<Work> {
    if !certificate(chunk, cfg, Some(regions)) {
        return None;
    }
    let shared_reads = super::read_guards::requested();
    let mut native_expansion = 128u64;
    let mut call_imports = 0u64;
    let mut cold_recovery = u64::try_from(cfg.handler_roots().len() + cfg.resume_roots().len())
        .ok()?
        .checked_mul(16)?;
    for (pc, op) in chunk.jit_ops().iter().enumerate() {
        match regions.emission(pc) {
            observed_regions::Emission::Native => {
                if !stack.reachable[pc] {
                    continue;
                }
                native_expansion = native_expansion.checked_add(40)?;
                if super::names::enabled()
                    && matches!(op, Op::LoadName(..) | Op::LoadNameForCall(..))
                {
                    native_expansion =
                        native_expansion.checked_add(if shared_reads { 32 } else { 384 })?;
                }
                if matches!(op, Op::LoadCap(_))
                    && super::names::enabled()
                    && !chunk.jit_needs_activation_state()
                {
                    native_expansion =
                        native_expansion.checked_add(if shared_reads { 32 } else { 384 })?;
                }
                if matches!(op, Op::Call(..) | Op::CallWithThis(..)) {
                    call_imports = call_imports.checked_add(128)?;
                }
            }
            observed_regions::Emission::Cut => {
                // No optimistic sharing credit: every cut pays for its exact
                // dirty local slice and actual known owning operand suffix.
                let words = frame.dirty_at[pc].len().checked_add(stack.at[pc].known())?;
                cold_recovery =
                    cold_recovery.checked_add(16 + u64::try_from(words).ok()?.checked_mul(2)?)?;
            }
            observed_regions::Emission::OmittedInterior => {}
        }
    }
    let original_labels = u64::try_from(chunk.jit_ops().len().checked_add(6)?)
        .ok()?
        .checked_mul(2)?
        .checked_add(super::native_pcs::analysis_work(chunk.jit_ops().len())?)?;
    let frame_words = frame.tracked.len().div_ceil(64).max(1);
    let original_frame = u64::try_from(
        chunk
            .jit_ops()
            .len()
            .checked_mul(frame_words)?
            .checked_mul(2)?
            .checked_add(chunk.jit_frame().1.checked_mul(2)?)?,
    )
    .ok()?;
    Some(Work {
        native_expansion,
        original_labels,
        original_frame,
        cold_recovery,
        call_imports,
        literal_analysis: 0,
        literal_dispatch_lowering: 0,
        borrowed_operands: 0,
    })
}

pub(super) fn work_with_literals(
    chunk: &Chunk,
    cfg: &Cfg,
    stack: &stack::StackPlan,
    frame: &frame::FramePlan,
    regions: &observed_regions::Plan,
) -> Option<Work> {
    let mut work = work(chunk, cfg, stack, frame, regions)?;
    // Even a no-match Plan initializes/scans the original graph, including all VM cuts.
    work.literal_analysis = super::string_literals::analysis_work(chunk.jit_ops().len())?;
    Some(work)
}

/// Normal-edge native reachability stops at every VM cut. Handler roots remain
/// in the compiler graph but provide no speculative normal-path opportunity.
/// Entered is a conservative union/superset; mutually exclusive branches may
/// contribute to this explicitly optimistic opportunity heuristic.
pub(super) fn opportunity_tests(
    chunk: &Chunk,
    cfg: &Cfg,
    regions: &observed_regions::Plan,
    numeric: Option<&super::specialization::Plan>,
    structural: Option<&super::generic_admission::Plan>,
) -> Option<u32> {
    opportunity_work(chunk, cfg, regions, numeric, structural, None)
}

/// Adds actual selected literal-owner work to the existing branch-union heuristic.
/// Handler-only roots and operations after a VM cut provide no credit.
pub(super) fn opportunity_work(
    chunk: &Chunk,
    cfg: &Cfg,
    regions: &observed_regions::Plan,
    numeric: Option<&super::specialization::Plan>,
    structural: Option<&super::generic_admission::Plan>,
    literals: Option<&super::string_literals::Plan>,
) -> Option<u32> {
    if !certificate(chunk, cfg, Some(regions)) {
        return None;
    }
    let mut reached = vec![false; cfg.blocks().len()];
    let mut pending = vec![cfg.block_at(0)?.0 as usize];
    while let Some(id) = pending.pop() {
        if std::mem::replace(&mut reached[id], true) || regions.cuts_block(id) {
            continue;
        }
        pending.extend(
            cfg.blocks()[id]
                .successors
                .iter()
                .map(|target| target.0 as usize),
        );
    }
    let mut numeric_pc = vec![0u32; chunk.jit_ops().len()];
    if let Some(numeric) = numeric {
        for opportunity in &numeric.opportunities {
            let at = numeric_pc.get_mut(opportunity.pc)?;
            *at = at.checked_add(opportunity.removed_tests)?;
        }
    }
    let mut saved = 0u32;
    for (id, block) in cfg.blocks().iter().enumerate() {
        if !reached[id] || regions.cuts_block(id) {
            continue;
        }
        for pc in block.start..block.end {
            if regions.emission(pc) != observed_regions::Emission::Native {
                continue;
            }
            if let Some(structural) = structural {
                saved = saved.checked_add(structural.per_pc[pc].unwrap_or(0))?;
            }
            saved = saved.checked_add(numeric_pc[pc])?;
            saved = saved.checked_add(literals.map_or(0, |plan| plan.credit_at(pc)))?;
        }
    }
    let guards = numeric.map_or(0, |plan| plan.inputs.len() + plan.properties.len());
    let saved = saved.checked_sub(u32::try_from(guards).ok()?)?;
    (saved != 0).then_some(saved)
}
