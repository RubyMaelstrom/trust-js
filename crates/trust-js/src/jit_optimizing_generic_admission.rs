//! Bounded hot selection for repeated ordinary own-data metadata checks.
//!
//! IC observations choose a guarded version; they never prove a live descriptor,
//! value category, owner count or future call count. The exact selected scalar
//! states are frozen into that version. Only the minimum removed structural
//! checks on normal and abrupt native paths pays the existing work estimate.
//! Host interruption/resource termination and later specialization misses are
//! separate outcomes; no historical entry count guarantees future repayment.
//! Numeric composition retains the older static-body numeric estimate.
//! ECMA-262 e28783d5 OrdinaryGet / OrdinarySetWithOwnDescriptor: reads and writes
//! keep their current descriptor guards, actual Values and distinct ownership.

use super::{frame, generic_own, heap_proofs, observed_regions, stack};
use crate::{
    bytecode::{Chunk, IcState, Op},
    jit_ir::Cfg,
};
use std::cell::Cell;

use crate::feedback::StructuralObservation as Observation;
pub(super) const MAX_SITES: usize = 32;

pub(super) fn enabled() -> bool {
    #[cfg(test)]
    let requested = generic_own::TEST_ENABLED
        .with(Cell::get)
        .unwrap_or_else(|| std::env::var("TRUST_JS_OPT_JIT_GENERIC_OWN").as_deref() == Ok("1"));
    #[cfg(not(test))]
    let requested = std::env::var("TRUST_JS_OPT_JIT_GENERIC_OWN").as_deref() == Ok("1");
    requested
        && std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() != Ok("0")
        && std::env::var("TRUST_JS_OPT_JIT_COMPACT_STORES").as_deref() != Ok("0")
}

fn cache(op: &Op) -> Option<(u32, u32)> {
    match *op {
        Op::GetPropLocal(_, name, cache)
        | Op::GetPropThis(name, cache)
        | Op::SetPropLocalDrop(_, name, cache)
        | Op::SetPropThisDrop(name, cache) => Some((name, cache)),
        _ => None,
    }
}

pub(super) fn has_source(chunk: &Chunk) -> bool {
    enabled()
        && chunk.jit_ops().iter().any(|op| {
            cache(op).is_some_and(|(name, _)| {
                crate::value::canonical_index(chunk.jit_name(name)).is_none()
            })
        })
}

fn snapshot(chunk: &Chunk, pc: usize) -> Option<IcState> {
    let (name, cache) = cache(chunk.jit_ops().get(pc)?)?;
    if crate::value::canonical_index(chunk.jit_name(name)).is_some() {
        return None;
    }
    chunk
        .jit_cache_preferred(cache)
        .filter(|state| state.depth == 0 && state.has_cacheable_shapes())
}

pub(super) fn same(left: IcState, right: IcState) -> bool {
    left.recv_shape == right.recv_shape
        && left.holder_shape == right.holder_shape
        && left.slot == right.slot
        && left.depth == right.depth
        && left.mid_ok == right.mid_ok
        && left.mid_shape == right.mid_shape
        && left.mid2_shape == right.mid2_shape
        && left.mid3_shape == right.mid3_shape
        && left.mid4_shape == right.mid4_shape
}

pub(super) fn observations(chunk: &Chunk) -> Box<[Observation]> {
    if !enabled() {
        return Box::default();
    }
    chunk
        .jit_ops()
        .iter()
        .enumerate()
        .filter(|(_, op)| {
            cache(op).is_some_and(|(name, _)| {
                crate::value::canonical_index(chunk.jit_name(name)).is_none()
            })
        })
        .take(MAX_SITES)
        .map(|(pc, _)| {
            let state = snapshot(chunk, pc);
            (pc as u32, Cell::new(state), Cell::new(state.is_some()))
        })
        .collect()
}

pub(super) fn observe(chunk: &Chunk, observations: &[Observation]) {
    for (pc, old, eligible) in observations {
        // A polymorphic, absent, accessor, indexed or changed site stays rejected
        // for this bounded version. No author operation or Value materialization.
        if !eligible.get()
            || !old
                .get()
                .zip(snapshot(chunk, *pc as usize))
                .is_some_and(|(old, current)| same(old, current))
        {
            eligible.set(false);
        }
    }
}

pub(super) fn selected(observations: &[Observation]) -> Vec<(u32, IcState)> {
    observations
        .iter()
        .filter_map(|(pc, state, eligible)| {
            eligible
                .get()
                .then(|| state.get().map(|state| (*pc, state)))
                .flatten()
        })
        .collect()
}

pub(super) struct Plan {
    pub(super) selected: Vec<(u32, IcState)>,
    pub(super) minimum_tests: u32,
    pub(super) per_pc: Vec<Option<u32>>,
}

impl Plan {
    #[cfg(test)]
    pub(super) fn build(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        selected: &[(u32, IcState)],
    ) -> Option<Self> {
        Self::with_objects_and_regions(chunk, cfg, stack, selected, None, None)
    }

    #[cfg(test)]
    pub(super) fn with_objects(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        selected: &[(u32, IcState)],
        objects: Option<&super::object_facts::Plan>,
    ) -> Option<Self> {
        Self::with_objects_and_regions(chunk, cfg, stack, selected, objects, None)
    }

    #[cfg(test)]
    pub(super) fn build_with_regions(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        selected: &[(u32, IcState)],
        regions: Option<&observed_regions::Plan>,
    ) -> Option<Self> {
        Self::with_objects_and_regions(chunk, cfg, stack, selected, None, regions)
    }

    pub(super) fn with_objects_and_regions(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        selected: &[(u32, IcState)],
        objects: Option<&super::object_facts::Plan>,
        regions: Option<&observed_regions::Plan>,
    ) -> Option<Self> {
        let large = super::large_regions::certificate(chunk, cfg, regions);
        if !enabled()
            || chunk.jit_ops().len() > 256 && !large
            || selected.is_empty()
            || selected.len() > MAX_SITES
            || selected
                .iter()
                .any(|&(pc, _)| pc as usize >= chunk.jit_ops().len())
        {
            return None;
        }
        let frame = frame::FramePlan::build(
            chunk,
            cfg,
            std::env::var("TRUST_JS_OPT_JIT_LOCAL_EFFECTS").as_deref() != Ok("0"),
        )?;
        // Planning a baseline uses precisely the same frozen selection seam as
        // lowering its candidate. It must not consult newer caches for extra sites.
        let selected: Vec<_> = selected
            .iter()
            .copied()
            .filter(|&(pc, _)| {
                regions.is_none_or(|plan| {
                    plan.emission(pc as usize) == observed_regions::Emission::Native
                })
            })
            .collect();
        if selected.is_empty() {
            return None;
        }
        let proof = generic_own::Plan::build_hot_selection_with_regions(
            chunk, cfg, &frame, stack, &selected, regions,
        )?;
        let mut weights = vec![0u32; cfg.blocks().len()];
        let mut per_pc = vec![None; chunk.jit_ops().len()];
        for &(pc, _) in &selected {
            let site = proof
                .read(pc as usize)
                .or_else(|| proof.write(pc as usize))?;
            let block = cfg.block_at(pc as usize)?.0 as usize;
            let credit = removed_tests(site.removed);
            weights[block] = weights[block].checked_add(credit)?;
            per_pc[pc as usize] = Some(credit);
        }
        // per_pc remains raw structural credit. The combined region estimate
        // recomputes its paths and charges this same emitted guard once itself.
        // Large observed bodies have separate opportunity accounting. This
        // certificate validates their scalar guards/proofs, never a minimum.
        let minimum_tests = if large {
            0
        } else {
            minimum_path(cfg, chunk, &weights, &per_pc, &[], objects, regions)?
                .saturating_sub(objects.map_or(0, |plan| plan.guard_cost))
        };
        (large || minimum_tests != 0).then_some(Self {
            selected,
            minimum_tests,
            per_pc,
        })
    }
}

fn removed_tests(removed: heap_proofs::ReadElisions) -> u32 {
    // Ordinary reuse omits the receiver tag, ic_plain and ordinary-exotic checks.
    // Some read templates also omit a duplicate exotic branch; no credit for it,
    // boxing, reference-count changes, storage pointers or hypothetical helpers.
    3 * u32::from(removed.ordinary)
        + u32::from(removed.shape)
        + u32::from(removed.bounds)
        + u32::from(removed.data)
}

pub(super) fn minimum_path(
    cfg: &Cfg,
    chunk: &Chunk,
    weights: &[u32],
    per_pc: &[Option<u32>],
    numeric_per_pc: &[u32],
    objects: Option<&super::object_facts::Plan>,
    regions: Option<&observed_regions::Plan>,
) -> Option<u32> {
    let ops = chunk.jit_ops();
    let blocks = cfg.blocks();
    if blocks.is_empty()
        || blocks.len() > 257
        || weights.len() != blocks.len()
        || per_pc.len() != ops.len()
        || (!numeric_per_pc.is_empty() && numeric_per_pc.len() != ops.len())
        || !cfg.handler_roots().is_empty()
        || !cfg.resume_roots().is_empty()
    {
        return None;
    }
    let entry = cfg.block_at(0)?.0 as usize;
    let mut distance = vec![u32::MAX; blocks.len()];
    let mut settled = vec![false; blocks.len()];
    distance[entry] = weights[entry];
    // Nonnegative shortest paths count no repeated backedge work. At most one
    // bounded quadratic scan per <=256-op body; no assumed loop trip count.
    for _ in 0..blocks.len() {
        let next = (0..blocks.len())
            .filter(|&b| !settled[b])
            .min_by_key(|&b| distance[b])?;
        if distance[next] == u32::MAX {
            break;
        }
        settled[next] = true;
        if regions.is_some_and(|plan| plan.cuts_block(next)) {
            // The native invocation ends here; the VM remainder cannot pay for it.
            continue;
        }
        for target in &blocks[next].successors {
            let target = target.0 as usize;
            if objects.is_some_and(|plan| {
                !plan.accepts(chunk, blocks[next].end - 1, blocks[target].start)
            }) {
                continue;
            }
            let total = distance[next].checked_add(*weights.get(target)?)?;
            distance[target] = distance[target].min(total);
        }
    }
    let terminal = blocks
        .iter()
        .enumerate()
        .filter(|(id, block)| {
            distance[*id] != u32::MAX
                && (regions.is_some_and(|plan| plan.cuts_block(*id))
                    || block.successors.is_empty()
                        && (block.start == ops.len()
                            || block
                                .end
                                .checked_sub(1)
                                .and_then(|pc| ops.get(pc))
                                .is_some_and(|op| {
                                    matches!(
                                        op,
                                        Op::Return | Op::ReturnBare | Op::ReturnUndef | Op::Throw
                                    )
                                })))
        })
        .map(|(id, _)| distance[id])
        .min();
    let mut minimum = terminal?;
    for (id, block) in blocks
        .iter()
        .enumerate()
        .filter(|(id, _)| distance[*id] != u32::MAX)
    {
        if regions.is_some_and(|plan| plan.cuts_block(id)) {
            continue;
        }
        let mut prefix = distance[id].checked_sub(weights[id])?;
        for pc in block.start..block.end {
            if let Some(credit) = per_pc[pc] {
                prefix = prefix.checked_add(credit)?;
            } else if may_throw(chunk, pc, objects) {
                // A checked call/getter/conversion may exit before the block's
                // final opcode. Later field savings cannot pay that entry.
                minimum = minimum.min(prefix);
            }
            // Numeric credit never treats a potentially abrupt operation as an
            // ordinary proven data access. Count its conservative early completion
            // before credit for that operation, not after it.
            prefix = prefix.checked_add(numeric_per_pc.get(pc).copied().unwrap_or(0))?;
        }
    }
    Some(minimum)
}

fn may_throw(chunk: &Chunk, pc: usize, objects: Option<&super::object_facts::Plan>) -> bool {
    let op = &chunk.jit_ops()[pc];
    // Use the IDENTICAL settled normal-edge initialization fact consumed by
    // ownership::LoadLocal. Object remains an owner: only its TDZ test disappears.
    if matches!(op, Op::LoadLocal(_))
        && objects.is_some_and(|plan| plan.values.at[pc].local.initialized())
    {
        return false;
    }
    !matches!(
        op,
        Op::Const(_)
            | Op::Undef
            | Op::LoadThis
            | Op::StoreLocal(_)
            | Op::Pop
            | Op::Dup
            | Op::Dup2
            | Op::Not
            | Op::Typeof
            | Op::StrictEq
            | Op::StrictNotEq
            | Op::Jump(_)
            | Op::JumpIfFalse(_)
            | Op::JumpIfFalsePeek(_)
            | Op::JumpIfTruePeek(_)
            | Op::JumpIfNotNullishPeek(_)
            | Op::InlineGuard(..)
            | Op::Return
            | Op::ReturnBare
            | Op::ReturnUndef
    ) && !matches!(op, Op::LoadLocal(slot) if usize::from(*slot) < chunk.jit_frame().0)
}

pub(super) fn admission_entries(
    chunk: &Chunk,
    numeric_tests: u32,
    structural_tests: u32,
) -> Option<u64> {
    let saved = numeric_tests.checked_add(structural_tests)?;
    (saved != 0).then(|| {
        (chunk.jit_ops().len() as u64)
            .saturating_mul(40)
            .saturating_add(super::names::instruction_allowance(chunk) as u64)
            .saturating_mul(1024)
            .div_ceil(u64::from(saved))
    })
}
