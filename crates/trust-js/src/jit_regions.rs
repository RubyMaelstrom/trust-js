//! General CFG loop and bounded acyclic lowering. Numeric homes are selected from SSA uses, not opcode
//! sequences. Objects remain owned in canonical frame/operand slots: property ICs can
//! borrow those roots. Effects materialize operand owners and only the private locals
//! they can inspect; unknown effects conservatively materialize and reload every home.
//! A post-effect type guard resumes *after* the effect, so getters/calls never replay.

use super::*;
use crate::bytecode::{Op, UpdKind};
use crate::jit_ir::{Cfg, FrameLoc, InstKind, RegionIr, ValueDef};

#[cfg(test)]
thread_local! {
    static EXECUTED_REGIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static BORROWED_READS: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
    static GUARDED_REGIONS: std::cell::Cell<[u64; 4]> = const { std::cell::Cell::new([0; 4]) };
    static FUSED_PREDICATES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static REUSED_CHAINS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    // Fast local-source own-element results: three-op GetElem DReg/slot, In DReg/slot,
    // then the older two-op local GetElem region fusion.
    static LOCAL_ARRAY_HITS: std::cell::Cell<[u64; 5]> = const { std::cell::Cell::new([0; 5]) };
    // Actual successful cross-block HasProperty/TDZ/Get admission, split by numeric-home key.
    static TDZ_ARRAY_PAIR_HITS: std::cell::Cell<[u64; 2]> = const { std::cell::Cell::new([0; 2]) };
    // Actual region executions of Tdz grouped by effect/home contract:
    // tagged Write with another numeric home, tagged Write without one, untagged Write,
    // numeric-home Full, other Full.
    static TDZ_EFFECT_PATHS: std::cell::Cell<[u64; 5]> = const { std::cell::Cell::new([0; 5]) };
    // In / CreateDataPropertyOrThrow sites emitted as frame-independent while this
    // region carries numeric homes.
    static INDEPENDENT_EFFECT_LIVE_HOMES: std::cell::Cell<[u64; 2]> =
        const { std::cell::Cell::new([0; 2]) };
    // Native data hit, native absence hit, checked property fallback, region property
    // continuation, numeric result, post-result miss, scalar prefix/local reuse.
    static PROPERTY_PATHS: std::cell::Cell<[u64; 7]> = const { std::cell::Cell::new([0; 7]) };
}

#[cfg(test)]
extern "C" fn record_property_path(event: usize) {
    PROPERTY_PATHS.with(|counts| {
        let mut current = counts.get();
        current[event] += 1;
        counts.set(current);
    });
}

/// The test-only event callbacks are C calls: keep the virtual stack's registers (d16..d31 and
/// the integer operand registers x2..x7) across them.
#[cfg(test)]
fn event_save(a: &mut asm::Asm) {
    a.sub_imm(31, 31, 176);
    for index in (16..32).step_by(2) {
        a.stp_d_off(index, index + 1, (index - 16) as i32 * 8);
    }
    for index in (2..8).step_by(2) {
        a.stp_off(index, index + 1, 128 + (index as i32 - 2) * 8);
    }
}

#[cfg(test)]
fn event_restore(a: &mut asm::Asm) {
    for index in (16..32).step_by(2) {
        a.ldp_d_off(index, index + 1, (index - 16) as i32 * 8);
    }
    for index in (2..8).step_by(2) {
        a.ldp_off(index, index + 1, 128 + (index as i32 - 2) * 8);
    }
    a.add_imm(31, 31, 176);
}

#[cfg(test)]
pub(super) fn emit_property_event(a: &mut asm::Asm, event: usize) {
    event_save(a);
    a.mov_imm64(0, event as u64);
    a.mov_imm64(16, record_property_path as *const () as u64);
    a.blr(16);
    event_restore(a);
}

#[cfg(test)]
extern "C" fn record_guarded_region(event: usize) {
    GUARDED_REGIONS.with(|counts| {
        let mut current = counts.get();
        current[event] += 1;
        counts.set(current);
    });
}

#[cfg(test)]
fn emit_guarded_event(a: &mut asm::Asm, event: usize) {
    event_save(a);
    a.mov_imm64(0, event as u64);
    a.mov_imm64(16, record_guarded_region as *const () as u64);
    a.blr(16);
    event_restore(a);
}

#[cfg(test)]
extern "C" fn record_region_entry() {
    EXECUTED_REGIONS.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
extern "C" fn record_fused_predicate() {
    FUSED_PREDICATES.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
fn emit_predicate_event(a: &mut asm::Asm) {
    emit_lowering_event(a, record_fused_predicate as *const () as u64);
}

#[cfg(test)]
extern "C" fn record_reused_chain() {
    REUSED_CHAINS.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
pub(super) extern "C" fn record_local_array_hit(kind: usize) {
    LOCAL_ARRAY_HITS.with(|counts| {
        let mut current = counts.get();
        current[kind] += 1;
        counts.set(current);
    });
}

#[cfg(test)]
pub(super) extern "C" fn record_local_array_get() {
    record_local_array_hit(0);
}

#[cfg(test)]
pub(super) extern "C" fn record_local_array_get_slot() {
    record_local_array_hit(1);
}

#[cfg(test)]
pub(super) extern "C" fn record_local_array_in() {
    record_local_array_hit(2);
}

#[cfg(test)]
pub(super) extern "C" fn record_local_array_in_slot() {
    record_local_array_hit(3);
}

#[cfg(test)]
pub(super) extern "C" fn record_local_array_legacy_get() {
    record_local_array_hit(4);
}

/// Register regions entered so far on this thread (tests only).
#[cfg(test)]
pub(super) fn executed_regions() -> u64 {
    EXECUTED_REGIONS.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(super) fn local_array_hits() -> [u64; 5] {
    LOCAL_ARRAY_HITS.with(std::cell::Cell::get)
}

#[cfg(test)]
extern "C" fn record_tdz_array_pair_hit(kind: usize) {
    TDZ_ARRAY_PAIR_HITS.with(|counts| {
        let mut current = counts.get();
        current[kind] += 1;
        counts.set(current);
    });
}

#[cfg(test)]
extern "C" fn record_tdz_array_pair_dreg() {
    record_tdz_array_pair_hit(0);
}

#[cfg(test)]
extern "C" fn record_tdz_array_pair_slot() {
    record_tdz_array_pair_hit(1);
}

#[cfg(test)]
pub(super) fn tdz_array_pair_hits() -> [u64; 2] {
    TDZ_ARRAY_PAIR_HITS.with(std::cell::Cell::get)
}

#[cfg(test)]
extern "C" fn record_tdz_effect_path(kind: usize) {
    TDZ_EFFECT_PATHS.with(|counts| {
        let mut current = counts.get();
        current[kind] += 1;
        counts.set(current);
    });
}

#[cfg(test)]
pub(super) fn tdz_effect_paths() -> [u64; 5] {
    TDZ_EFFECT_PATHS.with(std::cell::Cell::get)
}

#[cfg(test)]
fn emit_tdz_effect_event(a: &mut asm::Asm, kind: usize) {
    // The diagnostic callback preserves the registers live across an operation:
    // a Tdz can occur with both a live operand prefix and numeric/tagged local homes.
    // d8-d15 are preserved by the AAPCS64 callee; d16-d31 and x0-x17 are saved here.
    // The d0-d7 scratch registers have no live values at this boundary.
    const GPR_BYTES: i32 = 144;
    const FP_BYTES: i32 = 128;
    const SAVE_BYTES: i32 = GPR_BYTES + FP_BYTES;
    a.sub_imm(31, 31, SAVE_BYTES as u32);
    for index in (0..18).step_by(2) {
        a.stp_off(index, index + 1, index as i32 * 8);
    }
    for index in (16..32).step_by(2) {
        a.stp_d_off(index, index + 1, GPR_BYTES + (index - 16) as i32 * 8);
    }
    a.mov_imm64(0, kind as u64);
    a.mov_imm64(16, record_tdz_effect_path as *const () as u64);
    a.blr(16);
    for index in (16..32).step_by(2).rev() {
        a.ldp_d_off(index, index + 1, GPR_BYTES + (index - 16) as i32 * 8);
    }
    for index in (0..18).step_by(2).rev() {
        a.ldp_off(index, index + 1, index as i32 * 8);
    }
    a.add_imm(31, 31, SAVE_BYTES as u32);
}

#[cfg(test)]
extern "C" fn record_independent_effect_live_home(kind: usize) {
    INDEPENDENT_EFFECT_LIVE_HOMES.with(|counts| {
        let mut current = counts.get();
        current[kind] += 1;
        counts.set(current);
    });
}

#[cfg(test)]
pub(super) fn independent_effect_live_home_paths() -> [u64; 2] {
    INDEPENDENT_EFFECT_LIVE_HOMES.with(std::cell::Cell::get)
}

#[cfg(test)]
fn emit_independent_effect_live_home_event(a: &mut asm::Asm, kind: usize) {
    // This test-only event runs at an Independent In/CreateDataProperty site with
    // numeric homes still resident in ABI-preserved D registers. Preserve the
    // rest of the live machine state across the diagnostic callback.
    const GPR_BYTES: i32 = 144;
    const FP_BYTES: i32 = 128;
    const SAVE_BYTES: i32 = GPR_BYTES + FP_BYTES;
    a.sub_imm(31, 31, SAVE_BYTES as u32);
    for index in (0..18).step_by(2) {
        a.stp_off(index, index + 1, index as i32 * 8);
    }
    for index in (16..32).step_by(2) {
        a.stp_d_off(index, index + 1, GPR_BYTES + (index - 16) as i32 * 8);
    }
    a.mov_imm64(0, kind as u64);
    a.mov_imm64(16, record_independent_effect_live_home as *const () as u64);
    a.blr(16);
    for index in (16..32).step_by(2).rev() {
        a.ldp_d_off(index, index + 1, GPR_BYTES + (index - 16) as i32 * 8);
    }
    for index in (0..18).step_by(2).rev() {
        a.ldp_off(index, index + 1, index as i32 * 8);
    }
    a.add_imm(31, 31, SAVE_BYTES as u32);
}

#[cfg(test)]
pub(super) fn emit_lowering_event(a: &mut asm::Asm, function: u64) {
    // Record before computing the predicate; a C call clobbers NZCV and volatile GPRs.
    event_save(a);
    a.mov_imm64(16, function);
    a.blr(16);
    event_restore(a);
}

#[cfg(test)]
extern "C" fn record_borrowed_read(reused: u64) {
    BORROWED_READS.with(|counts| {
        let (reads, reuses) = counts.get();
        counts.set((reads + 1, reuses + reused));
    });
}

pub(super) struct Plan {
    head: usize,
    blocks: Vec<(usize, usize)>,
    acyclic: bool,
    homes: Vec<u16>,
    /// Borrowed compact words. Their canonical slots retain the sole owning
    /// references and remain roots across helpers and collection.
    tagged_homes: Vec<u16>,
    /// Demand is a profitability hint, never an entry/type proof. Every checked
    /// producer executes first and guards its actual result at the post-effect PC.
    numeric_results: crate::fasthash::FastSet<usize>,
    borrowed_methods: crate::fasthash::FastSet<usize>,
}

#[derive(Clone, Copy)]
struct TdzArrayPair {
    key: u16,
    receiver: u16,
    value: u16,
    mapped: u16,
    continuation: usize,
}

#[derive(Clone, Copy)]
struct TdzArrayEmission {
    pattern: TdzArrayPair,
    pre_fail: usize,
    post_fail: usize,
    continuation: usize,
}

/// The self-hosted map/filter loop has a HasProperty branch followed by two iteration-local
/// TDZ clears and a GetElem. Fuse only the exact source/CFG shape: misses before either clear
/// replay the two local loads and In from their first source PC; no landing or side entry may
/// enter the clear/read sequence; and its true edge is the body's sole predecessor. The emitter
/// later provides a private post-StoreLocal join.
fn tdz_array_pair_at(
    ops: &[Op],
    cfg: &Cfg,
    plan: &Plan,
    source_targets: &[bool],
    pc: usize,
) -> Option<TdzArrayPair> {
    let end = pc.checked_add(10)?;
    if end >= ops.len() || end >= source_targets.len() {
        return None;
    }
    let (
        Op::LoadLocal(key),
        Op::LoadLocal(receiver),
        Op::In,
        Op::JumpIfFalse(false_target),
        Op::Tdz(value),
        Op::Tdz(mapped),
        Op::LoadLocal(body_receiver),
        Op::LoadLocal(body_key),
        Op::GetElem,
        Op::StoreLocal(body_value),
    ) = (
        &ops[pc],
        &ops[pc + 1],
        &ops[pc + 2],
        &ops[pc + 3],
        &ops[pc + 4],
        &ops[pc + 5],
        &ops[pc + 6],
        &ops[pc + 7],
        &ops[pc + 8],
        &ops[pc + 9],
    )
    else {
        return None;
    };
    if receiver != body_receiver
        || key != body_key
        || value != body_value
        || receiver == key
        || receiver == value
        || receiver == mapped
        || key == value
        || key == mapped
        || value == mapped
    {
        return None;
    }

    let key_off = *key as u32 * 8;
    let receiver_off = *receiver as u32 * 8;
    if key_off + 8 >= 4096 || receiver_off + 8 >= 4096 {
        return None;
    }
    // The two Tdz targets cannot retain a stale DReg when cleared. The receiver likewise
    // must remain the canonical GC root; a numeric key home is safe and is preserved through
    // the release-only Tdz helpers. Checked numeric-result recovery after GetElem is a
    // separate continuation shape, so leave those sites to the established emitter.
    if plan.reg(*receiver).is_some()
        || plan.reg(*value).is_some()
        || plan.reg(*mapped).is_some()
        || plan.numeric_results.contains(&(pc + 8))
    {
        return None;
    }
    for slot in [*value, *mapped] {
        if (slot as u32) * 8 + 8 >= 4096 || plan.reg(slot).is_some() {
            return None;
        }
    }

    let continuation = pc + 10;
    let false_target = *false_target as usize;
    if (pc + 4..continuation).contains(&false_target) {
        return None;
    }
    // Each of the first four ops is one block ending at its conditional branch; the true
    // fallthrough body is one block through StoreLocal. Its only normal predecessor is that
    // branch. Handler/resume/OSR entries and bytecode targets cannot land in an interior op.
    let in_block = cfg.block_at(pc)?;
    let body_block = cfg.block_at(pc + 4)?;
    let in_cfg = cfg.blocks().get(in_block.0 as usize)?;
    let body_cfg = cfg.blocks().get(body_block.0 as usize)?;
    if in_cfg.end != pc + 4
        || cfg.block_at(pc + 3) != Some(in_block)
        || body_cfg.start != pc + 4
        || body_cfg.end < continuation
        || cfg.block_at(pc + 9) != Some(body_block)
        || body_cfg.predecessors.as_slice() != [in_block]
        || plan.head == body_cfg.start
        || cfg.block_at(false_target) == Some(body_block)
    {
        return None;
    }
    if (pc + 4..continuation).any(|target| source_targets[target])
        || cfg
            .handler_roots()
            .iter()
            .any(|root| root.target == body_block)
        || cfg
            .resume_roots()
            .iter()
            .any(|root| root.target == body_block)
        || cfg.osr_entry_depth(pc + 4).is_some()
    {
        return None;
    }
    let expected_depths = [0, 1, 2, 1, 0, 0, 0, 1, 2, 1, 0];
    if expected_depths
        .iter()
        .enumerate()
        .any(|(offset, depth)| cfg.stack_depth_at(pc + offset) != Some(*depth))
    {
        return None;
    }
    let in_plan = |at: usize| {
        plan.blocks
            .iter()
            .any(|&(start, end)| start <= at && at < end)
    };
    if !(pc..=continuation).all(in_plan) {
        return None;
    }

    Some(TdzArrayPair {
        key: *key,
        receiver: *receiver,
        value: *value,
        mapped: *mapped,
        continuation,
    })
}

fn checked_numeric_result(op: &Op) -> bool {
    matches!(
        op,
        Op::GetProp(..) | Op::GetPropLocal(..) | Op::GetPropThis(..)
        | Op::GetElem | Op::GetElemLocal(..) | Op::Call(..) | Op::CallWithThis(..)
        // A reused chain has a checked suffix as well as a numeric fast path.
        // Its canonical output therefore needs the same live post-result proof.
        | Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Neg | Op::Plus
        | Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr | Op::UShr | Op::BitNot
        // A frame slot read for arithmetic is guarded as a Number where it is read.
        | Op::LoadLocal(..)
    )
}

/// Backward SSA demand through local copies and real CFG phi edges. Each value
/// is visited once; the region's existing value/edge limits bound the whole walk.
/// A demand through a phi does not assert anything about an independent entry.
fn result_demands(
    ir: &RegionIr,
    chunk: &Chunk,
) -> (
    crate::fasthash::FastSet<usize>,
    crate::fasthash::FastSet<usize>,
) {
    let ops = chunk.jit_ops();
    let mut aliases = vec![Vec::new(); ir.values.len()];
    let mut work = Vec::new();
    let mut methods = Vec::new();
    let blocks: crate::fasthash::FastMap<_, _> = ir
        .blocks
        .iter()
        .map(|block| (block.cfg_block, block))
        .collect();
    for block in &ir.blocks {
        for inst in &block.insts {
            if let Op::InlineGuard(target, _) = ops[inst.pc] {
                if let Some(callee) = inst
                    .inputs
                    .len()
                    .checked_sub(chunk.jit_inline_target(target).argc as usize + 1)
                {
                    methods.push(inst.inputs[callee]);
                }
            }
            if matches!(inst.kind, InstKind::CheckLocal(_) | InstKind::Clone) {
                for output in &inst.outputs {
                    aliases[output.index()].extend(inst.inputs.iter().copied());
                }
            }
            if matches!(
                ops[inst.pc],
                Op::Add
                    | Op::Sub
                    | Op::Mul
                    | Op::Div
                    | Op::Lt
                    | Op::Le
                    | Op::Gt
                    | Op::Ge
                    | Op::BitAnd
                    | Op::BitOr
                    | Op::BitXor
                    | Op::Shl
                    | Op::Shr
                    | Op::UShr
                    | Op::BitNot
                    | Op::Neg
            ) {
                work.extend(inst.inputs.iter().copied());
            }
        }
        for edge in &block.successors {
            for ((_, parameter), argument) in blocks[&edge.target].params.iter().zip(&edge.args) {
                aliases[parameter.index()].push(*argument);
            }
        }
    }
    let mut seen = vec![false; ir.values.len()];
    let mut results = crate::fasthash::FastSet::default();
    while let Some(value) = work.pop() {
        if std::mem::replace(&mut seen[value.index()], true) {
            continue;
        }
        if let ValueDef::OpResult { pc, result: 0 } = ir.values[value.index()].def {
            if checked_numeric_result(&ops[pc]) {
                results.insert(pc);
            }
        }
        work.extend(aliases[value.index()].iter().copied());
    }
    seen.fill(false);
    let mut borrowed_methods = crate::fasthash::FastSet::default();
    while let Some(value) = methods.pop() {
        if std::mem::replace(&mut seen[value.index()], true) {
            continue;
        }
        if let ValueDef::OpResult { pc, .. } = ir.values[value.index()].def {
            if matches!(ops[pc], Op::GetMethod(..)) {
                borrowed_methods.insert(pc);
            }
        }
        methods.extend(aliases[value.index()].iter().copied());
    }
    (results, borrowed_methods)
}

impl Plan {
    pub(super) fn build(chunk: &Chunk, cfg: &Cfg, head: usize) -> Option<Self> {
        Self::build_kind(chunk, cfg, head, false, None)
    }

    pub(super) fn build_acyclic(
        chunk: &Chunk,
        cfg: &Cfg,
        head: usize,
        budget: &mut crate::jit_ir::AcyclicBudget,
    ) -> Option<Self> {
        Self::build_kind(chunk, cfg, head, true, Some(budget))
    }

    pub(super) fn ranges(&self) -> &[(usize, usize)] {
        &self.blocks
    }

    fn build_kind(
        chunk: &Chunk,
        cfg: &Cfg,
        head: usize,
        acyclic: bool,
        budget: Option<&mut crate::jit_ir::AcyclicBudget>,
    ) -> Option<Self> {
        if chunk.jit_detailed_feedback_enabled() {
            return None;
        }
        let ir = if acyclic {
            RegionIr::build_acyclic(chunk, cfg, head, budget?)
        } else {
            RegionIr::build_loop(chunk, cfg, head)
        }
        .ok()?;
        let live_entry = ir.live_entry_locals();
        let ops = chunk.jit_ops();
        let (numeric_results, borrowed_methods) = result_demands(&ir, chunk);
        let mut candidates: Vec<(u16, usize)> = Vec::new();
        let mut tagged_candidates: Vec<(u16, usize)> = Vec::new();
        let mut blocks = Vec::new();
        let mut effects = false;
        let mut opportunities = 0;
        let mut add = |slot| {
            if (slot as usize) < 512 {
                if let Some((_, weight)) = candidates.iter_mut().find(|(s, _)| *s == slot) {
                    *weight += 1;
                } else {
                    candidates.push((slot, 1));
                }
            }
        };
        for block in &ir.blocks {
            let range = &cfg.blocks()[block.cfg_block.0 as usize];
            blocks.push((range.start, range.end));
            for inst in &block.insts {
                let pc = inst.pc;
                // Native continuations leave the machine stack at every suspend
                // or completion boundary. Register regions must not route one
                // through an ordinary checked-op fallback or retain borrowed homes
                // across that boundary, even if CFG root construction changes.
                if crate::bytecode::jit_slice_exit_op(&ops[pc])
                    && !matches!(
                        ops[pc],
                        Op::Return | Op::ReturnBare | Op::ReturnUndef | Op::Throw
                    )
                {
                    return None;
                }
                if matches!(
                    ops[pc],
                    Op::LoadLocal(_)
                        | Op::Const(_)
                        | Op::GetPropLocal(..)
                        | Op::GetPropThis(..)
                        | Op::GetMethod(..)
                        | Op::InlineGuard(..)
                        | Op::Add
                        | Op::Sub
                        | Op::Mul
                        | Op::Div
                        | Op::StrictEq
                        | Op::Lt
                        | Op::Gt
                ) {
                    opportunities += 1;
                }
                if let Op::LoadLocal(slot) = ops[pc] {
                    if slot < 512 {
                        if let Some((_, weight)) =
                            tagged_candidates.iter_mut().find(|(s, _)| *s == slot)
                        {
                            *weight += 1;
                        } else {
                            tagged_candidates.push((slot, 1));
                        }
                    }
                }
                if cfg.stack_depth_at(pc)? > 15 {
                    return None;
                }
                match ops[pc] {
                    Op::UpdateLocal(slot, _) => add(slot),
                    Op::Add
                    | Op::Sub
                    | Op::Mul
                    | Op::Div
                    | Op::Lt
                    | Op::Gt
                    | Op::Le
                    | Op::Ge
                    | Op::BitAnd
                    | Op::BitOr
                    | Op::BitXor
                    | Op::Shl
                    | Op::Shr
                    | Op::UShr
                    | Op::BitNot => {
                        for input in &inst.inputs {
                            match ir.values[input.index()].def {
                                ValueDef::RegionInput(FrameLoc::Local(slot))
                                | ValueDef::BlockParam {
                                    loc: FrameLoc::Local(slot),
                                    ..
                                } => add(slot),
                                ValueDef::OpResult { pc: source, .. } => {
                                    if let Op::LoadLocal(slot) = ops[source] {
                                        add(slot);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    Op::PushHandler(..)
                    | Op::PushFinally(..)
                    | Op::PushIterator(..)
                    | Op::PopHandler
                    | Op::AbruptJump(..)
                    | Op::ResumeReturn
                    | Op::ResumeJump => return None,
                    Op::GetProp(..)
                    | Op::GetPropThis(..)
                    | Op::GetPropLocal(..)
                    | Op::SetProp(..)
                    | Op::SetPropDrop(..)
                    | Op::SetPropThisDrop(..)
                    | Op::SetPropLocalDrop(..)
                    | Op::GetElem
                    | Op::GetElemLocal(..)
                    | Op::SetElem
                    | Op::SetElemDrop
                    | Op::SetElemLocal(..)
                    | Op::SetElemLocalDrop(..)
                    | Op::Call(..)
                    | Op::CallWithThis(..) => effects = true,
                    _ => {}
                }
                if let InstKind::StoreLocal(slot) = inst.kind {
                    if inst.inputs.iter().any(|input| matches!(ir.values[input.index()].def,
                        ValueDef::OpResult { pc: source, .. } if matches!(ops[source], Op::Add | Op::Sub | Op::Mul | Op::Div
                            | Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr | Op::UShr | Op::BitNot))) {
                        add(slot);
                    }
                }
            }
        }
        // Keep the established straight-line chain for pure linear loops. The CFG tier
        // covers arbitrary diamonds, nested loops, and loops containing checked effects.
        if (acyclic && opportunities < 4)
            || (!acyclic && candidates.is_empty() && tagged_candidates.is_empty())
            || (!acyclic && blocks.len() <= 2 && !effects)
        {
            return None;
        }
        // Register pressure is not a reason to abandon the entire region. Choose
        // the most frequently used numeric candidates; the others retain their
        // ordinary canonical-slot templates. Stable ties keep compilation deterministic.
        candidates.sort_by_key(|(_, weight)| std::cmp::Reverse(*weight));
        let homes: Vec<u16> = candidates
            .into_iter()
            // A local defined inside a region may hold Undefined or an owner on entry.
            // Its canonical store must release that owner; don't impose a Number guard
            // on a dead initial value merely because its later value is arithmetic.
            .filter(|(slot, _)| !acyclic && live_entry.get(*slot as usize) == Some(&true))
            .take(8)
            .map(|(slot, _)| slot)
            .collect();
        tagged_candidates.sort_by_key(|(_, weight)| std::cmp::Reverse(*weight));
        let tagged_homes = tagged_candidates
            .into_iter()
            .filter(|(slot, _)| !homes.contains(slot))
            .take(8 - homes.len())
            .map(|(slot, _)| slot)
            .collect();
        blocks.sort_unstable();
        Some(Self {
            head,
            blocks,
            acyclic,
            homes,
            tagged_homes,
            numeric_results,
            borrowed_methods,
        })
    }

    fn reg(&self, slot: u16) -> Option<u32> {
        self.homes
            .iter()
            .position(|candidate| *candidate == slot)
            .map(|n| 8 + n as u32)
    }

    fn tagged_reg(&self, slot: u16) -> Option<u32> {
        self.tagged_homes
            .iter()
            .position(|candidate| *candidate == slot)
            .map(|index| 8 + self.homes.len() as u32 + index as u32)
    }

    pub(super) fn flush_locals(&self, a: &mut asm::Asm) {
        for (index, slot) in self.homes.iter().enumerate() {
            emit_exec_number_store(a, 8 + index as u32, 22, *slot as i32 * 8, 9);
        }
    }

    fn flush_local(&self, a: &mut asm::Asm, slot: u16) {
        if let Some(reg) = self.reg(slot) {
            emit_exec_number_store(a, reg, 22, slot as i32 * 8, 9);
        }
    }

    pub(super) fn reload(&self, a: &mut asm::Asm, fail: usize) {
        // Never write canonical memory while reloading. Failed post-effect guards resume
        // without flushing these registers, preserving locals that now hold object owners.
        for (index, slot) in self.homes.iter().enumerate() {
            emit_exec_word_load(a, 9, 22, *slot as i32 * 8);
            emit_exec_number_guard(a, 9, 8 + index as u32, 10, fail);
        }
        for slot in &self.tagged_homes {
            a.ldr_d_imm(self.tagged_reg(*slot).unwrap(), 22, *slot as u32 * 8);
        }
    }
}

/// All optional code and deopt destinations remain private until admission.
/// Cost units describe eliminated canonical transfers/owner operations, not CPU
/// cycles. Byte estimates deliberately allow the full mature property/call
/// templates; the function-wide hard byte budget remains authoritative.
pub(super) struct Emission {
    pub(super) plain: usize,
    pub(super) targets: Vec<usize>,
    pub(super) saved_work: usize,
    entry_work: usize,
    baseline_bytes: usize,
    pub(super) bytes: usize,
    acyclic: bool,
}

impl Emission {
    pub(super) fn profitable(&self) -> bool {
        let amortization = if self.acyclic { 1 } else { 16 };
        self.saved_work.saturating_mul(amortization) > self.entry_work
            && self.bytes <= self.baseline_bytes.saturating_mul(2).saturating_add(1024)
    }

    pub(super) fn publish(&self, targeted: &mut [bool]) {
        for &pc in &self.targets {
            targeted[pc] = true;
        }
    }
}

fn baseline_bytes(op: &Op) -> usize {
    match op {
        // These are conservative template-family code charges, not timing
        // predictions. Complex checked templates must not look artificially cheap.
        Op::GetProp(..)
        | Op::GetPropLocal(..)
        | Op::GetPropThis(..)
        | Op::GetMethod(..)
        | Op::GetElem
        | Op::GetElemLocal(..)
        | Op::GetMethodElem
        | Op::SetElem
        | Op::SetElemDrop
        | Op::SetElemLocal(..)
        | Op::SetElemLocalDrop(..) => 1536,
        Op::Call(..) | Op::CallWithThis(..) | Op::New(..) => 2048,
        Op::LoadName(..)
        | Op::LoadNameForCall(..)
        | Op::LoadNameIn(..)
        | Op::LoadNameForCallIn(..)
        | Op::LoadCap(..)
        | Op::StoreNameCached(..)
        | Op::UpdateNameCached(..)
        | Op::StoreCap(..) => 768,
        Op::LoadLocal(..)
        | Op::StoreLocal(..)
        | Op::UpdateLocal(..)
        | Op::StrictEq
        | Op::StrictNotEq
        | Op::EqEq
        | Op::NotEq => 192,
        Op::Jump(..) | Op::JumpIfFalse(..) | Op::Const(..) | Op::Undef => 64,
        _ => 128,
    }
}

/// Which canonical local values an effect helper can inspect or modify. This is
/// deliberately an allowlist: projected evaluator/stateful operations retain a
/// full barrier unless their complete frame contract is established.
///
/// CaptureScan homes bindings observable through closures, direct eval or mapped
/// arguments in Environment Records, not local slots (ECMA-262 §9.1.1.1 and
/// §10.2.11). Consequently author code entered by a Call/Get/Set cannot read or
/// change the caller's private slots. Its heap effects still invalidate object
/// facts; that is distinct from invalidating these numeric register homes.
/// Borrowed AST activations are the exception: their ordinary/legacy argument maps may
/// alias these slots. `region_frame_effect` strengthens their observer boundary to publish
/// all homes before author effects and reload cached representations after writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameEffect {
    Independent,
    Read(u16),
    /// The canonical template reads/replaces exactly this private slot. Author conversion
    /// (UpdateLocal) still cannot mutate another unexposed caller local.
    Write(u16),
    Full,
}

fn frame_effect(op: &Op) -> FrameEffect {
    match *op {
        Op::Call(..)
        | Op::CallWithThis(..)
        | Op::GetProp(..)
        | Op::GetPropThis(..)
        | Op::GetMethod(..)
        | Op::GetMethodElem
        | Op::SetProp(..)
        | Op::SetPropDrop(..)
        | Op::SetPropThisDrop(..)
        | Op::UpdateProp(..)
        | Op::GetElem
        | Op::SetElem
        | Op::SetElemDrop
        | Op::LoadName(..)
        | Op::LoadNameForCall(..)
        | Op::LoadNameIn(..)
        | Op::LoadNameForCallIn(..)
        | Op::UpdateNameCached(..)
        | Op::StoreNameCached(..)
        | Op::LoadCap(..)
        | Op::StoreCap(..)
        // These may invoke author code (proxy traps, coercion, or species
        // constructors), but consume only stack operands and never inspect a
        // caller's private slots. Their heap effects still clear own_facts.
        | Op::In
        | Op::Abstract(..)
        | Op::MakeObject(..)
        | Op::MakeArray(..)
        | Op::MakeRegExp(..)
        | Op::New(..)
        | Op::InstanceOf(..)
        | Op::Void
        | Op::Typeof
        | Op::TypeofIs(..)
        | Op::TypeofName(..)
        | Op::ToStr
        | Op::ToPropKey
        | Op::DestructureGuard
        | Op::Not
        | Op::Pop
        | Op::Dup
        | Op::LoadThis
        | Op::Undef
        | Op::Const(..) => FrameEffect::Independent,
        Op::GetPropLocal(slot, ..)
        | Op::SetPropLocalDrop(slot, ..)
        | Op::GetElemLocal(slot)
        | Op::SetElemLocal(slot)
        | Op::SetElemLocalDrop(slot)
        | Op::ToPropKeyLocal(slot) => FrameEffect::Read(slot),
        Op::StoreLocal(slot) | Op::UpdateLocal(slot, _) => FrameEffect::Write(slot),
        _ => FrameEffect::Full,
    }
}

/// Tdz changes only its target local. Keep the full barrier when that local is a
/// numeric register home: writing Empty would invalidate the DReg proof, so the
/// established full reload must deopt at the post-Tdz boundary. Tagged homes are
/// refreshed by the ordinary Write postlude after the canonical owner is released.
fn region_frame_effect(op: &Op, plan: &Plan, parameter_aliases: bool) -> FrameEffect {
    if parameter_aliases
        && !matches!(
            op,
            Op::Void
                | Op::Typeof
                | Op::TypeofIs(..)
                | Op::Not
                | Op::Pop
                | Op::Dup
                | Op::LoadThis
                | Op::Undef
                | Op::Const(..)
                | Op::StoreLocal(..)
                | Op::Tdz(..)
                | Op::DestructureGuard
        )
    {
        // A live AST ParameterMap can read/write formal slots from any author callback.
        // Publish the canonical cells before conversion/property/call effects, and reload
        // every cached home afterwards. Guarded pure Number operations never reach here.
        return FrameEffect::Full;
    }
    match *op {
        Op::Tdz(slot) if plan.reg(slot).is_none() => FrameEffect::Write(slot),
        _ => frame_effect(op),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarKind {
    Number,
    /// An exact integer strictly inside the signed i64 conversion range.
    Integer,
    /// Packed Boolean bits, never a floating-point Number payload.
    Boolean,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScalarState {
    kind: ScalarKind,
    /// The precise logical operand slot containing this value's canonical word.
    /// A duplicate/new result must not inherit a different slot's clean proof.
    canonical_slot: Option<u8>,
    /// d16+logical_index holds the value. Checked effects clobber these registers
    /// but leave unconsumed canonical scalar words intact.
    register_valid: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operand {
    Canonical,
    /// Non-owning view of a tagged home. LoadLocal rejects Empty/BigInt before
    /// creating this view. Every effect or control-flow edge clones it into a
    /// canonical owner before any source local can be replaced or observed by GC.
    Borrowed(u32),
    /// A live data-property value, rooted by a canonical receiver and its data
    /// descriptor until the next effect. Unlike a home, this uses its own stack
    /// temporary register, so Dup must copy its word into the new temporary.
    BorrowedValue,
    Scalar(ScalarState),
    /// A virtual int32 Number constant that has neither a register nor a canonical word yet.
    Constant(i32),
    /// An int32 result of a bitwise operator held in the low 32 bits of x`reg` (x2..x6, by
    /// operand index), with no floating-point register and no canonical word yet: `unsigned`
    /// is a `>>>` result, whose Number is the register read as a uint32. Whatever needs the
    /// double (an arithmetic operand, a store) converts it (`scvtf`/`ucvtf`); the bitwise
    /// operators, comparisons and branches use the register as it is.
    Int {
        reg: u8,
        unsigned: bool,
    },
}

impl Operand {
    fn temporary(kind: ScalarKind) -> Self {
        Self::Scalar(ScalarState {
            kind,
            canonical_slot: None,
            register_valid: true,
        })
    }

    /// The Number constant `value`, if it is an int32 (not -0): such a constant stays virtual
    /// until a consumer needs its register or canonical word, so one feeding a bitwise
    /// operation costs an immediate instead of a convert/compare/convert sequence.
    fn int32_constant(bits: u64) -> Option<Self> {
        let value = f64::from_bits(bits);
        let int = value as i32;
        (f64::from(int) == value && bits != (-0.0f64).to_bits()).then_some(Self::Constant(int))
    }

    /// The integer register pool: operand index -> x2..x6 (x7 belongs to the name probe).
    fn int_pool(index: usize) -> Option<u32> {
        (index < 5 && !env_flag!("TRUST_JS_JIT_NO_INT_REGS")).then_some(2 + index as u32)
    }

    fn immediate(self) -> Option<i32> {
        match self {
            Self::Constant(value) => Some(value),
            _ => None,
        }
    }

    fn clean(kind: ScalarKind, index: usize, register_valid: bool) -> Self {
        assert!(index < 16, "bounded region operand register bank");
        Self::Scalar(ScalarState {
            kind,
            canonical_slot: Some(index as u8),
            register_valid,
        })
    }

    fn scalar(self) -> bool {
        matches!(self, Self::Scalar(_) | Self::Constant(_) | Self::Int { .. })
    }

    fn numeric(self) -> bool {
        matches!(
            self,
            Self::Scalar(ScalarState {
                kind: ScalarKind::Number | ScalarKind::Integer,
                ..
            }) | Self::Constant(_)
                | Self::Int { .. }
        )
    }

    fn kind(self) -> Option<ScalarKind> {
        match self {
            Self::Scalar(state) => Some(state.kind),
            Self::Constant(_) | Self::Int { .. } => Some(ScalarKind::Integer),
            _ => None,
        }
    }
}

type Bailout = (usize, usize, Vec<Operand>, usize, bool);

fn preserve_scalar_prefix(
    a: &mut asm::Asm,
    before: &[Operand],
    after: &mut [Operand],
    physical: usize,
) {
    assert_eq!(
        after.len(),
        physical,
        "effect must publish its exact canonical extent"
    );
    let mut restored = false;
    for (index, value) in before.iter().copied().enumerate().take(after.len()) {
        if let Some(kind) = value.kind() {
            // GetValue/Call already ran once with the whole prefix published.
            // Preserve its immutable language type and exact canonical slot,
            // not volatile register contents. Reload only at a real consumer.
            // Borrowed heap/local views are deliberately lost across effects.
            after[index] = Operand::clean(kind, index, false);
            restored = true;
        }
    }
    #[cfg(test)]
    if restored {
        emit_property_event(a, 6);
    }
    #[cfg(not(test))]
    let _ = (a, restored);
}

/// Establish register residency without converting, cloning, or guarding a
/// previously proved language scalar. Canonical identity never follows a Dup
/// or an overwritten result slot: those operations construct a new dirty state.
/// The floating-point register of the operand at `index`.
fn reg_d(index: usize) -> u32 {
    16 + index as u32
}

fn scalar_register(a: &mut asm::Asm, stack: &mut [Operand], physical: usize, index: usize) {
    let reg = 16 + index as u32;
    if let Operand::Constant(value) = stack[index] {
        // A virtual constant becomes an ordinary dirty Integer temporary in its register.
        if value == 0 {
            a.fmov_d_x(reg, 31); // XZR
        } else if value == 1 {
            a.fmov_one(reg);
        } else {
            a.mov_imm64(9, f64::from(value).to_bits());
            a.fmov_d_x(reg, 9);
        }
        stack[index] = Operand::temporary(ScalarKind::Integer);
        return;
    }
    if let Operand::Int { reg, unsigned } = stack[index] {
        // The double of an integer operand is made when something needs it.
        if unsigned {
            a.ucvtf_d_w(reg_d(index), u32::from(reg));
        } else {
            a.scvtf_d_w(reg_d(index), u32::from(reg));
        }
        stack[index] = Operand::temporary(ScalarKind::Integer);
        return;
    }
    let Operand::Scalar(state) = &mut stack[index] else {
        unreachable!("scalar register requires a scalar proof")
    };
    if !state.register_valid {
        assert_eq!(state.canonical_slot.map(usize::from), Some(index));
        emit_exec_word_load(a, 9, 20, (index as i32 - physical as i32) * 8);
        a.fmov_d_x(reg, 9);
        state.register_valid = true;
    }
}

fn duplicate_operand(a: &mut asm::Asm, stack: &mut Vec<Operand>, physical: usize) {
    let depth = stack.len();
    let duplicate = match *stack.last().expect("Dup has an operand") {
        // A virtual constant duplicates for free.
        value @ Operand::Constant(_) => value,
        Operand::Int { .. } => {
            scalar_register(a, stack, physical, depth - 1);
            a.fmov_d_d(16 + depth as u32, 15 + depth as u32);
            Operand::temporary(ScalarKind::Integer)
        }
        Operand::Scalar(state) => {
            scalar_register(a, stack, physical, depth - 1);
            a.fmov_d_d(16 + depth as u32, 15 + depth as u32);
            // The new slot contains neither a canonical word nor an owner yet.
            // Never copy the source's canonical identity to a different index.
            Operand::temporary(state.kind)
        }
        value @ Operand::Borrowed(_) => value,
        value @ Operand::BorrowedValue => {
            a.fmov_d_d(16 + depth as u32, 15 + depth as u32);
            value
        }
        Operand::Canonical => unreachable!("owning Dup uses the checked template"),
    };
    stack.push(duplicate);
}

fn propagate_checked_result(
    a: &mut asm::Asm,
    plan: &Plan,
    pc: usize,
    stack: &mut [Operand],
    physical: usize,
    bails: &mut Vec<Bailout>,
    result_misses: &mut Vec<usize>,
) {
    if !plan.numeric_results.contains(&pc) || stack.is_empty() {
        return;
    }
    // OrdinaryGet/Call has already completed once. Every earlier value and the
    // actual returned owner are canonical at this guard. A miss resumes after
    // the producer, never at it (ECMA-262 OrdinaryGet / ToNumeric).
    let miss = a.new_label();
    bails.push((miss, pc + 1, stack.to_vec(), physical, true));
    result_misses.push(miss);
    emit_exec_word_load(a, 9, 20, -8);
    let index = stack.len() - 1;
    emit_exec_number_guard(a, 9, 16 + index as u32, 10, miss);
    stack[index] = Operand::clean(ScalarKind::Number, index, true);
    #[cfg(test)]
    emit_property_event(a, 4);
}

fn materialize(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    stack: &[Operand],
    physical: usize,
) {
    for (index, value) in stack.iter().enumerate() {
        let offset = (index as i32 - physical as i32) * 8;
        let reg = 16 + index as u32;
        match value {
            Operand::Canonical => {}
            Operand::Borrowed(_) | Operand::BorrowedValue => {
                let source = match value {
                    Operand::Borrowed(home) => *home,
                    _ => reg,
                };
                a.fmov_x_d(9, source);
                // The originating read guard excludes BigInt, Empty and all
                // property-only tags such as lazy function prototypes.
                // All remaining compact frame kinds are immediate or shared Rc.
                // Clone cannot call author code, allocate, or invalidate another
                // virtual register; the canonical local continues to own the source.
                let cloned = a.new_label();
                emit_exec_clone(a, layout, 9, 10, 11, cloned);
                a.bind(cloned);
                emit_exec_word_store(a, 9, 20, offset);
            }
            Operand::Int { reg, unsigned } => {
                // An integer is always a finite Number: no NaN to canonicalize.
                if *unsigned {
                    a.ucvtf_d_w(0, u32::from(*reg));
                } else {
                    a.scvtf_d_w(0, u32::from(*reg));
                }
                a.stur_d(0, 20, offset);
            }
            Operand::Constant(value) => {
                // Never produced a register: publish the constant's own word.
                a.mov_imm64(9, f64::from(*value).to_bits());
                emit_exec_word_store(a, 9, 20, offset);
            }
            Operand::Scalar(state) => {
                if let Some(slot) = state.canonical_slot {
                    assert_eq!(
                        usize::from(slot),
                        index,
                        "clean scalar belongs to a different operand"
                    );
                    continue;
                }
                assert!(
                    state.register_valid,
                    "dirty scalar must have a live register"
                );
                match state.kind {
                    ScalarKind::Number | ScalarKind::Integer => {
                        emit_exec_number_store(a, reg, 20, offset, 9);
                    }
                    ScalarKind::Boolean => {
                        a.fmov_x_d(9, reg);
                        emit_exec_word_store(a, 9, 20, offset);
                    }
                }
            }
        }
    }
    if stack.len() > physical {
        a.add_imm(20, 20, (stack.len() - physical) as u32 * 8);
    } else if stack.len() < physical {
        a.sub_imm(20, 20, (physical - stack.len()) as u32 * 8);
    }
}

/// Read a tagged operand without creating another owner. A virtual Number/Boolean cannot
/// satisfy an object identity/receiver guard, even if its floating payload resembles a tag.
fn guard_operand(a: &mut asm::Asm, stack: &[Operand], physical: usize, index: usize, fail: usize) {
    match stack[index] {
        Operand::Canonical => emit_exec_word_load(a, 9, 20, (index as i32 - physical as i32) * 8),
        Operand::Borrowed(home) => a.fmov_x_d(9, home),
        Operand::BorrowedValue => a.fmov_x_d(9, 16 + index as u32),
        Operand::Scalar(_) | Operand::Constant(_) | Operand::Int { .. } => a.b(fail),
    }
}

fn number(a: &mut asm::Asm, stack: &mut [Operand], physical: usize, index: usize, fail: usize) {
    match stack[index] {
        Operand::Scalar(ScalarState {
            kind: ScalarKind::Number | ScalarKind::Integer,
            ..
        }) => {
            scalar_register(a, stack, physical, index);
        }
        Operand::Constant(_) | Operand::Int { .. } => scalar_register(a, stack, physical, index),
        Operand::Scalar(_) => a.b(fail), // Boolean coercion remains in the checked operation
        Operand::Borrowed(_) | Operand::BorrowedValue => {
            let source = match stack[index] {
                Operand::Borrowed(home) => home,
                _ => 16 + index as u32,
            };
            a.fmov_x_d(9, source);
            emit_exec_number_guard(a, 9, 16 + index as u32, 10, fail);
            stack[index] = Operand::temporary(ScalarKind::Number);
        }
        Operand::Canonical => {
            let offset = (index as i32 - physical as i32) * 8;
            emit_exec_word_load(a, 9, 20, offset);
            emit_exec_number_guard(a, 9, 16 + index as u32, 10, fail);
            stack[index] = Operand::clean(ScalarKind::Number, index, true);
        }
    }
}

/// A shape proves a descriptor location, never its value (ECMA-262
/// OrdinaryGet, §10.1.8.1, local snapshot e28783d5). Facts survive only an
/// effect-free part of one basic block. In particular no assumption of distinct
/// objects is made from distinct local slots: every possible heap effect clears
/// the entire set. IC cells and their chunk are pinned for native execution.
#[derive(Default)]
struct OwnReadFacts(Vec<(PropRecv, usize)>);

impl OwnReadFacts {
    fn clear(&mut self) {
        self.0.clear();
    }

    fn record(&mut self, receiver: PropRecv, cache: usize) -> Option<usize> {
        if let Some((_, previous)) = self.0.iter_mut().find(|(r, _)| *r == receiver) {
            Some(std::mem::replace(previous, cache))
        } else {
            self.0.push((receiver, cache));
            None
        }
    }
}

/// Borrow an ordinary own data value without touching its reference count. All
/// guards fail at the original Get PC, before any observable operation commits.
/// Accessors, exotics, inherited properties, BigInts and lazy-prototype sentinels
/// keep the complete existing checked templates. Reload the *live* property even
/// when a preceding read proved the receiver's type/shape.
fn emit_borrowed_own_read(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    receiver: PropRecv,
    cache: usize,
    previous_cache: Option<usize>,
    output: u32,
    fail: usize,
) {
    use crate::bytecode::{IC_OFF_DEPTH, IC_OFF_RECV_SHAPE, IC_OFF_SLOT};
    let reused = previous_cache.is_some();
    match receiver {
        PropRecv::This => {
            a.ldr_imm(14, 19, 48); // wide ctx.this_raw; owned by this activation
            if !reused {
                a.ldurb(9, 14, 0);
                a.cmp_imm_w(9, 8);
                a.b_cond(C_NE, fail);
            }
            a.ldur(10, 14, 8);
        }
        PropRecv::Slot(offset) => {
            emit_exec_word_load(a, 9, 22, offset as i32);
            if !reused {
                emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, fail);
            }
            emit_exec_payload(a, 9, 10);
        }
        PropRecv::Stack => unreachable!("a borrowed read requires a canonical receiver"),
    }
    a.add_imm(11, 10, layout.obj_from_rc as u32);
    a.mov_imm64(12, cache as u64);
    a.ldrb_imm(9, 12, IC_OFF_DEPTH);
    a.cbnz(9, false, fail); // only depth zero is authoritative for own reads
    if let Some(previous) = previous_cache {
        if previous != cache {
            // No helper or author code could have changed either cache since
            // the previous successful read. Equal feedback shapes extend its
            // receiver proof to this site's descriptor, not to its value.
            a.mov_imm64(14, previous as u64);
            a.ldr_w_imm(14, 14, IC_OFF_RECV_SHAPE);
            a.ldr_w_imm(16, 12, IC_OFF_RECV_SHAPE);
            a.cmp_reg_w(14, 16);
            a.b_cond(C_NE, fail);
        }
    } else {
        a.ldrb_imm(14, 11, layout.obj_exotic as u32);
        a.cmp_imm_w(14, layout.exotic_none_tag as u32);
        a.b_cond(C_NE, fail);
        a.ldrb_imm(14, 11, layout.obj_ic_plain as u32);
        a.cbz(14, false, fail);
        a.ldr_w_imm(14, 11, (layout.obj_props + layout.props_shape) as u32);
        a.ldr_w_imm(16, 12, IC_OFF_RECV_SHAPE);
        a.cmp_reg_w(14, 16);
        a.b_cond(C_NE, fail);
    }
    a.ldr_w_imm(13, 12, IC_OFF_SLOT);
    a.ldr_imm(
        16,
        11,
        (layout.obj_props + layout.props_entries + layout.vec_len_off) as u32,
    );
    a.cmp_reg_x(13, 16);
    a.b_cond(C_HS, fail);
    a.ldr_imm(
        15,
        11,
        (layout.obj_props + layout.props_entries + layout.vec_ptr_off) as u32,
    );
    a.mov_imm64(16, layout.entry_size as u64);
    a.madd(15, 13, 16, 15);
    guard_prop_data(a, 9, 15, layout.entry_accessor as u32, fail);
    a.ldur(12, 15, layout.entry_value as i32);
    a.lsr_imm(9, 12, 48);
    for tag in [crate::value::PACK_BIGINT, crate::value::PACK_EMPTY] {
        a.movz(14, (tag >> 48) as u32, 0);
        a.cmp_reg_w(9, 14);
        a.b_cond(C_EQ, fail);
    }
    // Negative tagged words above Obj are property-only internal values, not
    // cloneable execution words. Ordinary numbers (including -NaN canonicalized
    // on heap stores) and every public compact kind remain eligible.
    a.movz(14, (crate::value::PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_w(9, 14);
    a.b_cond(C_HI, fail);
    a.fmov_d_x(output, 12);
    #[cfg(test)]
    {
        // Runtime evidence, without assuming TLS addresses at compile time or
        // clobbering any live caller-saved virtual operand register.
        a.sub_imm(31, 31, 128);
        for index in (16..32).step_by(2) {
            a.stp_d_off(index, index + 1, (index - 16) as i32 * 8);
        }
        a.mov_imm64(0, u64::from(reused));
        a.mov_imm64(16, record_borrowed_read as *const () as u64);
        a.blr(16);
        for index in (16..32).step_by(2) {
            a.ldp_d_off(index, index + 1, (index - 16) as i32 * 8);
        }
        a.add_imm(31, 31, 128);
    }
}

fn borrowed_method_state(chunk: &Chunk, cache: u32) -> Option<crate::bytecode::IcState> {
    chunk.jit_cache_preferred(cache).filter(|state| {
        state.has_cacheable_shapes()
            && state.depth <= 3
            && (state.depth < 2
                || state.mid_ok & ((1 << (state.depth - 1)) - 1) == (1 << (state.depth - 1)) - 1)
    })
}

fn borrowed_own_state(chunk: &Chunk, op: &Op) -> Option<crate::bytecode::IcState> {
    let cache = match *op {
        Op::GetPropLocal(_, _, cache) | Op::GetPropThis(_, cache) => cache,
        _ => return None,
    };
    // Array holders carry IC_ARR_KEYCHK, inherited/negative states have their
    // own depth tags. Unknown/polymorphic sites retain the complete baseline
    // native template rather than paying a guaranteed own-only side exit.
    chunk
        .jit_cache_preferred(cache)
        .filter(|state| state.depth == 0 && state.has_cacheable_shapes())
}

/// OrdinaryGet may return a borrowed data word while the method receiver remains live.
/// Every prototype pointer is followed again, not pinned from the feedback. No getter,
/// allocation, helper or owner overwrite occurs between this read and its materialization.
/// Consequently the canonical receiver also retains the holder through its live chain.
fn emit_borrowed_method(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    state: crate::bytecode::IcState,
    stack: &[Operand],
    physical: usize,
    fail: usize,
) {
    let depth = stack.len();
    guard_operand(a, stack, physical, depth - 1, fail);
    emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 11, fail);
    emit_exec_payload(a, 9, 10);
    a.add_imm(11, 10, layout.obj_from_rc as u32);
    for hop in 0..=state.depth {
        if hop != 0 {
            a.ldr_imm(17, 11, layout.obj_proto as u32);
            a.cbz(17, true, fail);
            a.add_imm(11, 17, layout.obj_from_rc as u32);
        }
        a.ldrb_imm(14, 11, layout.obj_exotic as u32);
        a.cmp_imm_w(14, layout.exotic_none_tag as u32);
        a.b_cond(C_NE, fail);
        a.ldrb_imm(14, 11, layout.obj_ic_plain as u32);
        a.cbz(14, false, fail);
        a.ldr_w_imm(14, 11, (layout.obj_props + layout.props_shape) as u32);
        let shape = if hop == 0 {
            state.recv_shape
        } else if hop == state.depth {
            state.holder_shape
        } else if hop == 1 {
            state.mid_shape
        } else {
            state.mid2_shape
        };
        a.mov_imm64(16, shape as u64);
        a.cmp_reg_w(14, 16);
        a.b_cond(C_NE, fail);
    }
    a.mov_imm64(13, state.slot as u64);
    a.ldr_imm(
        16,
        11,
        (layout.obj_props + layout.props_entries + layout.vec_len_off) as u32,
    );
    a.cmp_reg_x(13, 16);
    a.b_cond(C_HS, fail);
    a.ldr_imm(
        15,
        11,
        (layout.obj_props + layout.props_entries + layout.vec_ptr_off) as u32,
    );
    a.mov_imm64(16, layout.entry_size as u64);
    a.madd(15, 13, 16, 15);
    guard_prop_data(a, 9, 15, layout.entry_accessor as u32, fail);
    a.ldur(12, 15, layout.entry_value as i32);
    a.lsr_imm(9, 12, 48);
    for tag in [crate::value::PACK_BIGINT, crate::value::PACK_EMPTY] {
        a.movz(14, (tag >> 48) as u32, 0);
        a.cmp_reg_w(9, 14);
        a.b_cond(C_EQ, fail);
    }
    a.movz(14, (crate::value::PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_w(9, 14);
    a.b_cond(C_HI, fail);
    a.fmov_d_x(16 + depth as u32, 12);
    #[cfg(test)]
    emit_guarded_event(a, 3);
}

/// ECMA-262 ToInt32/ToUint32 and NumberBitwiseOp (local snapshot e28783d5):
/// truncate, then wrap modulo 2^32. FJCVTZS (FEAT_JSCVT) is exactly this conversion for every
/// Number, including NaN, the infinities and values beyond the i64 range, so it never leaves
/// the region. Without it an exact i64 conversion has the same low bits: fractional finite
/// inputs are valid, while NaN/infinities and saturation must take the checked opcode with its
/// original operands (including the +2^63 round-trip edge).
fn to_int32(
    a: &mut asm::Asm,
    stack: &mut [Operand],
    physical: usize,
    index: usize,
    output: u32,
    fail: usize,
) {
    let reg = 16 + index as u32;
    if stack[index].kind() == Some(ScalarKind::Boolean) {
        // ToNumber(false/true) is 0/1. Boolean temporaries hold packed bits, not
        // an f64 payload, so converting the FP register would convert a NaN tag.
        scalar_register(a, stack, physical, index);
        a.fmov_x_d(output, reg);
        a.logic_imm_w(0, output, output, asm::logical_imm_w(1).unwrap());
        return;
    }
    number(a, stack, physical, index, fail);
    if jscvt_available() {
        a.fjcvtzs_w_d(output, reg);
        return;
    }
    a.fcvtzs_x_d(output, reg);
    if stack[index].kind() != Some(ScalarKind::Integer) {
        a.scvtf_d_x(0, output);
        a.frintz(1, reg);
        a.fcmp(0, 1);
        a.b_cond(C_NE, fail);
        a.cmn_imm_x(output, 1);
        a.b_cond(C_VS, fail);
    }
}

/// The attribution family of the code a region emits for `op` (see `TRUST_JS_JIT_MAP`): one
/// interned string per operation kind. Only called when profiling marks are enabled.
fn region_mark(op: &Op) -> &'static str {
    use std::sync::Mutex;
    static MARKS: Mutex<Vec<(String, &'static str)>> = Mutex::new(Vec::new());
    let debug = format!("{op:?}");
    let name = debug.split('(').next().unwrap_or("?").to_owned();
    let mut marks = MARKS.lock().unwrap();
    if let Some((_, interned)) = marks.iter().find(|(known, _)| *known == name) {
        return interned;
    }
    let interned: &'static str = Box::leak(format!("region:{name}").into_boxed_str());
    marks.push((name, interned));
    interned
}

/// Branch to `fail` unless the frame slot at `offset` owns nothing: its word is a Number or one
/// of the immediate kinds (undefined, empty, null, Boolean), never a BigInt, String, Symbol or
/// Object reference whose release a plain overwrite would skip.
fn emit_slot_owns_nothing(a: &mut asm::Asm, offset: i32, fail: usize) {
    emit_exec_word_load(a, 9, 22, offset);
    a.lsr_imm(10, 9, 48);
    a.movz(11, (crate::value::PACK_BIGINT >> 48) as u32, 0);
    a.sub_reg(11, 10, 11);
    a.cmp_imm_x(11, 2);
    a.b_cond(C_LS, fail); // BigInt, String, Symbol
    a.movz(11, (crate::value::PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_x(10, 11);
    a.b_cond(C_HS, fail); // Object, lazy prototype
}

/// The value domain in which two integer operands compare exactly (ECMA-262 Number::lessThan and
/// Number::equal on integral values): the operands of a bitwise operator are int32 (signed) or,
/// for `>>>`, uint32 (unsigned); a constant is whichever domain holds it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum IntDomain {
    /// Both operands read as int32: a 32-bit signed compare.
    Signed,
    /// Both read as uint32: a 32-bit unsigned compare.
    Unsigned,
    /// One of each: compare the values widened to 64 bits.
    Wide,
}

impl IntDomain {
    /// The condition that is true when the comparison `op` holds.
    fn condition(self, op: &Op) -> u32 {
        const C_LT: u32 = 11;
        const C_LE: u32 = 13;
        match (self, op) {
            (_, Op::EqEq | Op::StrictEq) => C_EQ,
            (_, Op::NotEq | Op::StrictNotEq) => C_NE,
            (Self::Unsigned, Op::Lt) => C_LO,
            (Self::Unsigned, Op::Le) => C_LS,
            (Self::Unsigned, Op::Gt) => C_HI,
            (Self::Unsigned, _) => C_HS,
            (_, Op::Lt) => C_LT,
            (_, Op::Le) => C_LE,
            (_, Op::Gt) => C_GT,
            (_, _) => C_GE,
        }
    }
}

/// The exact integer domain of a comparison whose operands are integer registers or constants
/// (at least one a register), else None: the operands are then Numbers in double registers.
fn int_comparison(op: &Op, operands: &[Operand]) -> Option<IntDomain> {
    if !matches!(
        op,
        Op::Lt | Op::Le | Op::Gt | Op::Ge | Op::EqEq | Op::StrictEq | Op::NotEq | Op::StrictNotEq
    ) || !operands
        .iter()
        .all(|o| matches!(o, Operand::Int { .. } | Operand::Constant(_)))
        || !operands.iter().any(|o| matches!(o, Operand::Int { .. }))
    {
        return None;
    }
    // Signed-compatible: int32 results and constants. Unsigned-compatible: uint32 results and
    // non-negative constants.
    let signed = operands.iter().all(|o| match o {
        Operand::Int { unsigned, .. } => !unsigned,
        _ => true,
    });
    let unsigned = operands.iter().all(|o| match o {
        Operand::Int { unsigned, .. } => *unsigned,
        Operand::Constant(value) => *value >= 0,
        _ => false,
    });
    Some(if signed {
        IntDomain::Signed
    } else if unsigned {
        IntDomain::Unsigned
    } else {
        IntDomain::Wide
    })
}

/// Set NZCV for `operands[0] ? operands[1]` in the exact domain of `int_comparison`. Scratch
/// x12/x13 for widened or immediate operands.
fn emit_int_compare(a: &mut asm::Asm, domain: IntDomain, operands: [Operand; 2]) {
    let widen = |a: &mut asm::Asm, operand: Operand, scratch: u32| -> u32 {
        match operand {
            Operand::Int { reg, unsigned } => {
                let reg = u32::from(reg);
                if unsigned {
                    a.mov_w(scratch, reg); // zero-extends
                } else {
                    a.sxtw(scratch, reg);
                }
            }
            Operand::Constant(value) => a.mov_imm64(scratch, value as i64 as u64),
            _ => unreachable!("integer comparison operands"),
        }
        scratch
    };
    if domain == IntDomain::Wide {
        let left = widen(a, operands[0], 12);
        let right = widen(a, operands[1], 13);
        a.cmp_reg_x(left, right);
        return;
    }
    let narrow = |a: &mut asm::Asm, operand: Operand, scratch: u32| -> u32 {
        match operand {
            Operand::Int { reg, .. } => u32::from(reg),
            Operand::Constant(value) => {
                a.mov_imm64(scratch, u64::from(value as u32));
                scratch
            }
            _ => unreachable!("integer comparison operands"),
        }
    };
    match (operands[0], operands[1]) {
        (Operand::Int { reg, .. }, Operand::Constant(value)) if (0..4096).contains(&value) => {
            a.cmp_imm_w(u32::from(reg), value as u32);
        }
        _ => {
            let left = narrow(a, operands[0], 12);
            let right = narrow(a, operands[1], 13);
            a.cmp_reg_w(left, right);
        }
    }
}

/// A ToInt32 result: a compile-time constant (ToInt32 applied while compiling, so it costs no
/// register, conversion or guard) or the low 32 bits of an x register.
#[derive(Clone, Copy)]
enum IntOperand {
    Imm(i32),
    Reg(u32),
}

fn int32_operand(
    a: &mut asm::Asm,
    stack: &mut [Operand],
    physical: usize,
    index: usize,
    output: u32,
    fail: usize,
) -> IntOperand {
    if let Some(value) = stack[index].immediate() {
        return IntOperand::Imm(value); // an int32 constant is its own ToInt32
    }
    if let Operand::Int { reg, .. } = stack[index] {
        return IntOperand::Reg(u32::from(reg)); // already ToInt32 bits (also for a uint32)
    }
    to_int32(a, stack, physical, index, output, fail);
    IntOperand::Reg(output)
}

/// NumberBitwiseOp for the six binary operators (ECMA-262 §6.1.6.1.9-§6.1.6.1.11 and
/// §13.9-§13.12): the operands are already ToInt32 values (ToUint32 has the same bits), shift
/// counts use their low five bits, and `>>>` alone yields its unsigned result. Returns the
/// x register whose low 32 bits are the result; identity operations (`x | 0`, `x & -1`,
/// `x ^ 0`, `x << 0`, `x >> 0`, `x >>> 0`) emit nothing. `scratch` holds an unencodable
/// immediate and `out` receives any computed result.
fn int32_binary(
    a: &mut asm::Asm,
    op: &Op,
    lhs: IntOperand,
    rhs: IntOperand,
    out: u32,
    scratch: u32,
) -> u32 {
    use IntOperand::{Imm, Reg};
    let load = |a: &mut asm::Asm, reg: u32, value: i32| a.mov_imm64(reg, u64::from(value as u32));
    let commutative = matches!(op, Op::BitAnd | Op::BitOr | Op::BitXor);
    let (lhs, rhs) = match (lhs, rhs) {
        (Imm(_), Reg(_)) if commutative => (rhs, lhs),
        _ => (lhs, rhs),
    };
    match (lhs, rhs) {
        (Imm(x), Imm(y)) => {
            let value = match op {
                Op::BitAnd => x & y,
                Op::BitOr => x | y,
                Op::BitXor => x ^ y,
                Op::Shl => x.wrapping_shl(y as u32 & 31),
                Op::Shr => x >> (y as u32 & 31),
                _ => ((x as u32) >> (y as u32 & 31)) as i32,
            };
            load(a, out, value);
            out
        }
        (Reg(l), Imm(y)) => {
            // `l` is x12 or, after a commutative swap, x13: keep the immediate clear of it.
            let scratch = if l == scratch { 12 } else { scratch };
            let logical = |a: &mut asm::Asm, code: u32| match asm::logical_imm_w(y as u32) {
                Some(field) => a.logic_imm_w(code, out, l, field),
                None => {
                    load(a, scratch, y);
                    a.logic_w(code, out, l, scratch);
                }
            };
            match op {
                Op::BitAnd if y == -1 => return l,
                Op::BitOr | Op::BitXor if y == 0 => return l,
                Op::BitAnd => logical(a, 0),
                Op::BitOr => logical(a, 1),
                Op::BitXor => logical(a, 2),
                _ => {
                    let count = y as u32 & 31;
                    if count == 0 {
                        return l;
                    }
                    match op {
                        Op::Shl => a.lsl_imm_w(out, l, count),
                        Op::Shr => a.asr_imm_w(out, l, count),
                        _ => a.lsr_imm_w(out, l, count),
                    }
                }
            }
            out
        }
        (lhs, rhs) => {
            let l = match lhs {
                Reg(register) => register,
                Imm(value) => {
                    load(a, out, value);
                    out
                }
            };
            let r = match rhs {
                Reg(register) => register,
                Imm(value) => {
                    load(a, scratch, value);
                    scratch
                }
            };
            match op {
                Op::BitAnd => a.logic_w(0, out, l, r),
                Op::BitOr => a.logic_w(1, out, l, r),
                Op::BitXor => a.logic_w(2, out, l, r),
                Op::Shl => a.shift_w(0, out, l, r),
                Op::UShr => a.shift_w(1, out, l, r),
                _ => a.shift_w(2, out, l, r),
            }
            out
        }
    }
}

/// `LoadName K; <key>; GetElem` and `LoadName K; <key>; <value>; SetElemDrop` where the operands
/// between the name and the element operation are pure Number arithmetic over numeric homes and
/// constants. Nothing there can observe or change the binding, throw, or run author code (a
/// guard that fails resumes at the LoadName with the stack as it was), so the name is read after
/// them (ECMA-262 §13.3.2 EvaluatePropertyAccessWithExpressionKey, TypedArrayGetElement §10.4.5.15,
/// TypedArraySetElement §10.4.5.16): a typed array bound to the name is accessed straight from
/// its binding, with no owning copy on the operand stack, and everything else (a different
/// kind of receiver, a stale name cache, an out-of-bounds or detached view, a throw) runs the
/// ordinary LoadName and element templates in a cold continuation.
#[derive(Clone, Copy)]
struct NameElementPlan {
    elem_pc: usize,
    set: bool,
}

#[allow(clippy::too_many_arguments)]
fn plan_name_element(
    chunk: &Chunk,
    ops: &[Op],
    name_pc: usize,
    end: usize,
    source_targets: &[bool],
    plan: &Plan,
    layout: &crate::value::JitLayout,
    fast: u32,
    pre_len: usize,
) -> Option<NameElementPlan> {
    if env_flag!("TRUST_JS_JIT_NO_NAME_ELEM")
        || !layout.valid
        || layout.native_typed_array.is_none()
        || layout.entry_accessor != layout.entry_value + 8
        || fast & 8192 == 0
        || !load_name_inlinable(layout)
    {
        return None;
    }
    let mut depth = 0usize; // operands above the (absent) name
    for pc in name_pc + 1..end {
        if source_targets[pc] {
            return None;
        }
        match ops[pc] {
            Op::GetElem if depth == 1 && fast & 1024 != 0 && get_elem_inlinable(layout) => {
                return Some(NameElementPlan {
                    elem_pc: pc,
                    set: false,
                });
            }
            Op::SetElemDrop if depth == 2 && fast & 2048 != 0 && elem_inlinable(layout) => {
                return Some(NameElementPlan {
                    elem_pc: pc,
                    set: true,
                });
            }
            // A numeric home, or a frame slot read as a Number (a miss replays the LoadName).
            Op::LoadLocal(slot) if slot < 256 && plan.tagged_reg(slot).is_none() => depth += 1,
            Op::Const(k) if chunk.jit_const_num(k).is_some() => depth += 1,
            Op::Add
            | Op::Sub
            | Op::Mul
            | Op::Div
            | Op::BitAnd
            | Op::BitOr
            | Op::BitXor
            | Op::Shl
            | Op::Shr
            | Op::UShr
                if depth >= 2 =>
            {
                depth -= 1
            }
            Op::BitNot | Op::Neg if depth >= 1 => {}
            _ => return None,
        }
        if pre_len + 1 + depth > 15 {
            return None;
        }
    }
    None
}

/// Whether fused sites call the process-wide routine (`TRUST_JS_JIT_INLINE_STUBS` and
/// `TRUST_JS_JIT_NO_NAME_ELEM_STUB` carry the sequence in line).
fn name_element_stub(a: &asm::Asm) -> bool {
    let _ = a;
    shared_stubs_enabled() && !env_flag!("TRUST_JS_JIT_NO_NAME_ELEM_STUB")
}

/// The access of a fused site once its key (d3) and value (d1) are in place and x12 holds the
/// name cache: the binding, an ordinary non-exotic Object, an exact index, and the native
/// typed-array element access. Registers: x7, x9-x17, d0-d3 (x13 and the status labels are the
/// caller's). `leaf` carries the name probe and the element access in line: the shared routine
/// calls nothing, so its link register stays the caller's.
///
/// * `hit`: the element is read (x12 = packed Number, d1 = value) or written.
/// * `value_miss`: the name validated (x14 = its value, x7 = storage kind) but the receiver or
///   index is not for the native path.
/// * `name_miss`: the name cache did not validate or a typed-array guard failed.
pub(super) fn emit_name_element_body(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    write: bool,
    leaf: bool,
    hit: usize,
    name_miss: usize,
    value_miss: usize,
) {
    // The binding: x14 = its value, x7 = 0 for a wide scope binding, 1 for a packed global
    // property (the shared probe of LoadName).
    if leaf || !shared_stubs_enabled() {
        emit_name_ic_value_ptr_body(a, layout, name_miss, true);
    } else {
        let stub = a.shared_stub(SharedStub::NameValuePtr { packed_ok: true }.key());
        a.bl_label(stub);
        a.cbz(9, false, name_miss);
    }
    // x10 = the Object's allocation. A wide scope binding holds its Value (tag byte, then the
    // payload word); a packed global property holds the encoded word.
    let have = a.new_label();
    let wide = a.new_label();
    a.cbz(7, false, wide);
    a.ldur(12, 14, 0);
    emit_exec_tag_guard(a, 12, crate::value::PACK_OBJ, 13, value_miss);
    emit_exec_payload(a, 12, 10);
    a.b(have);
    a.bind(wide);
    a.ldurb(13, 14, 0);
    a.cmp_imm_w(13, 8); // Value::Obj
    a.b_cond(C_NE, value_miss);
    a.ldur(10, 14, 8);
    a.bind(have);
    // An ordinary Object that is not exotic (a typed array is neither Array nor plain).
    a.add_imm(11, 10, layout.obj_from_rc as u32);
    a.ldrb_imm(12, 11, layout.obj_exotic as u32);
    a.cmp_imm_w(12, layout.exotic_none_tag as u32);
    a.b_cond(C_NE, value_miss);
    a.ldrb_imm(12, 11, layout.obj_ic_plain as u32);
    a.cbnz(12, false, value_miss);
    // The index: exactly a u32 (negative and fractional keys, NaN and -0 miss).
    if jscvt_available() {
        // FJCVTZS reports in Z that the key is an integer within int32 (so -0, fractions,
        // NaN and anything beyond decline); a negative key is then a u32 above 2^31, which
        // no native view's element count (at most 2^31) admits.
        a.fjcvtzs_w_d(9, 3);
        a.b_cond(C_NE, value_miss);
    } else {
        a.fcvtzu_w_d(9, 3);
        a.ucvtf_d_w(0, 9);
        a.fcmp(3, 0);
        a.b_cond(C_NE, value_miss);
    }
    // The value (d1) is in place: the access reads it as a register operand.
    match layout.native_typed_array {
        Some(native) if leaf => typed_array::emit_access(a, &native, write, hit, name_miss),
        _ => {
            let value = write.then_some(typed_array::WriteValue::Register(1));
            typed_array::emit(a, layout, value, hit, name_miss);
        }
    }
}

/// The cold continuation of one fused name/element site (see [`NameElementPlan`]).
struct NameElementFallback {
    /// The name cache did not validate, or a typed-array guard failed: run the whole LoadName.
    name_miss: usize,
    /// The name validated (x14 = its value, x7 = storage kind) but is not a typed array: clone
    /// the value without probing again.
    value_miss: usize,
    join: usize,
    /// The operand stack below the name, as the site found it. With `lazy` set some of it is
    /// still virtual (registers only) and `physical` words are in memory: the continuation
    /// publishes all of it, runs the ordinary templates, and restores the virtual picture.
    pre: Vec<Operand>,
    physical: usize,
    lazy: bool,
    cache_ptr: usize,
    preferred: Option<u64>,
    name_pc: usize,
    elem_pc: usize,
    set: bool,
    key: Operand,
    value: Option<Operand>,
}

/// Where an operand awaits, virtual, above the stack pointer: constants are published as their
/// own word, registers canonicalized like any numeric stack store.
fn stash_operand(a: &mut asm::Asm, operand: Operand, index: usize, offset: i32) {
    match operand {
        Operand::Constant(value) => {
            a.mov_imm64(9, f64::from(value).to_bits());
            emit_exec_word_store(a, 9, 20, offset);
        }
        _ => emit_exec_number_store(a, 16 + index as u32, 20, offset, 9),
    }
}

/// The fast path of a fused site, entered with the operand stack canonical up to `pre_len` and
/// the key (and value) virtual above it. On a hit the result of a get is on the stack (canonical
/// word); a set leaves nothing.
#[allow(clippy::too_many_arguments)]
fn emit_name_element(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    chunk: &Chunk,
    stack: &mut [Operand],
    physical: usize,
    pre: Vec<Operand>,
    lazy: bool,
    name_pc: usize,
    plan: NameElementPlan,
) -> NameElementFallback {
    let ops = chunk.jit_ops();
    let Op::LoadName(_, cache) = ops[name_pc] else {
        unreachable!("planned at a LoadName")
    };
    let cache_ptr = chunk.jit_name_cache_ptr(cache);
    let name_miss = a.new_label();
    let value_miss = a.new_label();
    let join = a.new_label();
    let hit = a.new_label();
    a.mark_push("name-elem");
    let key_index = pre.len();
    // Registers for the operands (a constant key needs none: it is an immediate below).
    for index in [key_index, key_index + 1] {
        if index < key_index + 1 + usize::from(plan.set)
            && (!matches!(stack[index], Operand::Constant(_)) || index != key_index)
        {
            number(a, stack, physical, index, name_miss);
        }
    }
    let key = stack[key_index];
    let value = plan.set.then(|| stack[key_index + 1]);

    // The key (d3) and the value to store (d1) for the shared sequence.
    match key {
        Operand::Constant(index) if index >= 0 => {
            a.mov_imm64(9, index as u64);
            a.scvtf_d_w(3, 9);
        }
        Operand::Constant(_) => a.b(name_miss), // a negative index is never an element
        _ => a.fmov_d_d(3, 16 + key_index as u32),
    }
    if plan.set {
        a.fmov_d_d(1, 16 + key_index as u32 + 1);
    }
    a.mov_imm64(12, cache_ptr as u64);
    if name_element_stub(a) {
        // The sequence below, in one process-wide routine: W13 = 1 after the access, 2 when the
        // binding validated but is not a typed array (x14/x7 still describe it), else 0.
        let stub = a.shared_stub(SharedStub::NameElement { write: plan.set }.key());
        a.bl_label(stub);
        a.cbz(13, false, name_miss);
        a.cmp_imm_w(13, 1);
        a.b_cond(C_NE, value_miss);
    } else {
        emit_name_element_body(a, layout, plan.set, false, hit, name_miss, value_miss);
    }
    a.bind(hit);
    if !plan.set {
        if lazy {
            // The result stays a register: nothing touches the operand stack.
            a.fmov_d_d(16 + key_index as u32, 1);
        } else {
            emit_exec_word_store(a, 12, 20, 0);
            a.add_imm(20, 20, 8);
        }
    }
    a.bind(join);
    a.mark_pop();
    NameElementFallback {
        name_miss,
        value_miss,
        join,
        pre,
        physical,
        lazy,
        cache_ptr,
        preferred: chunk.jit_name_number(cache),
        name_pc,
        elem_pc: plan.elem_pc,
        set: plan.set,
        key,
        value,
    }
}

fn emit_name_element_fallback(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    f: &NameElementFallback,
    unwind: usize,
    bails: &mut Vec<Bailout>,
) {
    a.mark_push("name-elem-fallback");
    let pre_len = f.pre.len();
    let full = a.new_label();
    let common = a.new_label();
    // Entered with the operand stack as the fast path left it: publish what is virtual, then
    // set the key (and value) in their slots above the name's.
    let prefix = |a: &mut asm::Asm| {
        if f.lazy {
            materialize(a, layout, &f.pre, f.physical);
        }
        stash_operand(a, f.key, pre_len, 8);
        if let Some(value) = f.value {
            stash_operand(a, value, pre_len + 1, 16);
        }
    };
    a.bind(f.name_miss);
    prefix(a);
    a.bind(full);
    emit_load_name_inline(
        a,
        layout,
        f.cache_ptr,
        f.preferred,
        f.name_pc as u32,
        unwind,
        false,
    );
    a.b(common);
    a.bind(f.value_miss);
    prefix(a);
    emit_load_name_value(a, layout, f.preferred, full, false);
    a.bind(common);
    a.add_imm(20, 20, 8 * (1 + u32::from(f.set)));
    if f.set {
        emit_set_elem_inline(a, layout, f.elem_pc as u32, unwind, false);
    } else {
        emit_get_elem_inline(a, layout, f.elem_pc as u32, unwind);
    }
    if f.lazy {
        // Back to the register picture of the fast path: the sums and offsets that were only
        // in registers are reloaded from the words just published, the result (a get demanded
        // as a Number) is guarded, and the stack pointer returns to the memory extent the
        // fast path left.
        let top = pre_len + usize::from(!f.set);
        for (index, operand) in f.pre.iter().enumerate() {
            let dirty = matches!(operand, Operand::Scalar(state) if state.canonical_slot.is_none());
            if dirty {
                emit_exec_word_load(a, 9, 20, (index as i32 - top as i32) * 8);
                a.fmov_d_x(16 + index as u32, 9);
            }
        }
        if !f.set {
            let miss = a.new_label();
            // A result that is not a Number resumes after the element operation with the
            // whole operand stack in memory, which is where it is.
            bails.push((
                miss,
                f.elem_pc + 1,
                vec![Operand::Canonical; top],
                top,
                true,
            ));
            emit_exec_word_load(a, 9, 20, -8);
            emit_exec_number_guard(a, 9, 16 + pre_len as u32, 10, miss);
        }
        // The fast path leaves the stack pointer where the site found it, which may lie beyond
        // the words that are still live (popped numeric operands).
        if top > f.physical {
            a.sub_imm(20, 20, ((top - f.physical) * 8) as u32);
        } else if top < f.physical {
            a.add_imm(20, 20, ((f.physical - top) * 8) as u32);
        }
    }
    a.b(f.join);
    a.mark_pop();
}

/// Route only edges whose operand owners are already canonical. Internal forward
/// edges need no stub; backedges retain the poll and external exits retain local
/// publication. This does not apply to the unique InlineGuard success edge, which
/// can carry a virtual operand map instead of the ordinary block-entry contract.
struct CanonicalEdges<'a> {
    labels: &'a std::collections::HashMap<usize, usize>,
    deferred: Vec<(usize, usize, usize)>,
}

impl<'a> CanonicalEdges<'a> {
    fn new(labels: &'a std::collections::HashMap<usize, usize>) -> Self {
        Self {
            labels,
            deferred: Vec::new(),
        }
    }

    fn target(&mut self, a: &mut asm::Asm, from: usize, target: usize) -> usize {
        if target > from {
            if let Some(&label) = self.labels.get(&target) {
                return label;
            }
        }
        let label = a.new_label();
        self.deferred.push((label, from, target));
        label
    }

    fn transfer(
        &mut self,
        a: &mut asm::Asm,
        from: usize,
        target: usize,
        physical_next: Option<usize>,
    ) {
        // Actual emission adjacency, not merely adjacent source PCs. Even a
        // physically adjacent self/backedge still owes its canonical poll.
        if target > from && physical_next == Some(target) && self.labels.contains_key(&target) {
            return;
        }
        let label = self.target(a, from, target);
        a.b(label);
    }
}

/// Emit a private CFG and cold deoptimization stubs, retaining all original bytecode
/// entries as exact resumptions. This is a fixed-home register allocator with explicit
/// safepoints, not a linear trace whose branch conditions must match a pattern.
pub(super) fn emit(
    a: &mut asm::Asm,
    chunk: &Chunk,
    cfg: &Cfg,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    plan: &Plan,
    pc_labels: &[usize],
    source_targets: &[bool],
    stronger_header_cache: &mut [Option<bool>],
    fast: u32,
    array_intrinsics_on: bool,
    function_call_intrinsic_on: bool,
    unwind: usize,
    direct_finish: usize,
    entry_kind: NativeEntryKind,
) -> Emission {
    let parameter_aliases =
        entry_kind == NativeEntryKind::BorrowedFrame && chunk.jit_borrowed_parameter_aliases();
    let emission_start = a.checkpoint();
    #[cfg(feature = "optimizing-jit")]
    emit_coverage_ranges(a, chunk, plan.ranges());

    let plain = a.new_label();
    let mut saved_work = 0usize;
    let mut targets = Vec::new();
    let stronger_headers: Vec<_> = plan
        .blocks
        .iter()
        .filter_map(|&(head, _)| {
            if head == plan.head || cfg.loop_at_header(head).is_none() {
                return None;
            }
            let stronger = *stronger_header_cache[head].get_or_insert_with(|| {
                plan_loop(
                    chunk,
                    chunk.jit_ops(),
                    head,
                    source_targets,
                    layout,
                    fast,
                    cfg,
                )
                .is_some()
                    || plan_linked_scan(chunk, chunk.jit_ops(), head, cfg, layout, fast).is_some()
            });
            stronger.then_some(head)
        })
        .collect();
    let labels: std::collections::HashMap<usize, usize> = plan
        .blocks
        .iter()
        .map(|(start, _)| (*start, a.new_label()))
        .collect();
    let mut exits = Vec::new();
    let mut bails: Vec<Bailout> = Vec::new();
    let mut fused: Vec<NameElementFallback> = Vec::new();
    let mut edges = CanonicalEdges::new(&labels);
    let mut guard_misses = Vec::new();
    let mut result_misses = Vec::new();
    let mut tdz_array_pairs = HashMap::<usize, TdzArrayEmission>::new();
    let mut tdz_array_skips = crate::fasthash::FastSet::default();
    let mut tdz_array_exit_blocks = crate::fasthash::FastSet::default();
    let mut tdz_array_continuations = HashMap::<usize, Vec<usize>>::new();
    if fast & (16 | 1024) == (16 | 1024) && elem_inlinable(layout) {
        for &(start, end) in &plan.blocks {
            for pc in start..end {
                if tdz_array_skips.contains(&pc) {
                    continue;
                }
                let Some(pattern) =
                    tdz_array_pair_at(chunk.jit_ops(), cfg, plan, source_targets, pc)
                else {
                    continue;
                };
                let Some(block_id) = cfg.block_at(pc) else {
                    continue;
                };
                let block_start = cfg.blocks()[block_id.0 as usize].start;
                let pre_fail = a.new_label();
                let post_fail = a.new_label();
                let continuation = a.new_label();
                bails.push((pre_fail, pc, Vec::new(), 0, true));
                bails.push((post_fail, pc + 6, Vec::new(), 0, true));
                tdz_array_exit_blocks.insert(block_start);
                tdz_array_skips.extend(pc + 1..pattern.continuation);
                tdz_array_continuations
                    .entry(pattern.continuation)
                    .or_default()
                    .push(continuation);
                tdz_array_pairs.insert(
                    pc,
                    TdzArrayEmission {
                        pattern,
                        pre_fail,
                        post_fail,
                        continuation,
                    },
                );
            }
        }
    }
    // A frame-independent helper can throw after consuming operands. Its exact
    // operand pointer is already updated by the ordinary template; only private
    // numeric homes need materializing before the shared handler unwinder runs.
    let private_unwind = a.new_label();
    plan.reload(a, plain);
    #[cfg(test)]
    {
        // Test-only runtime evidence after entry guards, not merely successful
        // compilation. The fixed homes are ABI-preserved d8..d15.
        a.mov_imm64(16, record_region_entry as *const () as u64);
        a.blr(16);
        if plan.acyclic {
            emit_guarded_event(a, 0);
        }
    }
    a.b(labels[&plan.head]);
    let ops = chunk.jit_ops();
    let mut carried: Option<(usize, Vec<Operand>, usize, OwnReadFacts)> = None;
    for (block_index, &(start, end)) in plan.blocks.iter().enumerate() {
        let physical_next = plan.blocks.get(block_index + 1).map(|&(next, _)| next);
        a.bind(labels[&start]);
        if stronger_headers.contains(&start) {
            // All incoming ordinary edges already materialized the operand stack. A loop
            // header cannot be a unique guard-success fallthrough (it has a backedge).
            // Transfer canonical locals to the stronger original header; its exit follows
            // original labels, and any outer reentry reloads homes and polls normally.
            debug_assert!(carried.is_none());
            plan.flush_locals(a);
            a.b(pc_labels[start]);
            continue;
        }
        let (mut stack, mut physical, mut own_facts) =
            if let Some((target, stack, physical, facts)) = carried.take() {
                assert_eq!(target, start);
                (stack, physical, facts)
            } else {
                let depth = cfg.stack_depth_at(start).unwrap();
                (
                    vec![Operand::Canonical; depth],
                    depth,
                    OwnReadFacts::default(),
                )
            };
        let mut consumed = 0;
        // A pending `LoadName`: its pc, the fused site's plan and the canonical stack below it.
        let mut fusing: Option<(usize, NameElementPlan, Vec<Operand>, bool)> = None;
        // Block-local facts are established by executed stores, never imported
        // from SSA demand or a different predecessor/OSR entry. CaptureScan keeps
        // author-visible bindings out of fresh-frame slots. Borrowed activation aliases
        // invalidate these facts through their full observer barrier.
        let mut numeric_locals = crate::fasthash::FastSet::default();
        for pc in start..end {
            if let Some(labels) = tdz_array_continuations.get(&pc) {
                for &label in labels {
                    a.bind(label);
                }
            }
            if tdz_array_skips.contains(&pc) {
                continue;
            }
            if consumed != 0 {
                consumed -= 1;
                continue;
            }
            if let Some(candidate) = tdz_array_pairs.get(&pc).copied() {
                let pattern = candidate.pattern;
                debug_assert!(stack.is_empty());
                debug_assert_eq!(physical, 0);
                // Avoid two owning local loads, operand/condition traffic, and the
                // repeated receiver/index/descriptor proof across HasProperty and Get.
                saved_work += 8;
                own_facts.clear();
                numeric_locals.remove(&pattern.value);
                numeric_locals.remove(&pattern.mapped);

                let key_off = pattern.key as u32 * 8;
                let receiver_off = pattern.receiver as u32 * 8;
                let key_src = plan
                    .reg(pattern.key)
                    .map_or(KeySrc::Slot(key_off), KeySrc::DReg);
                emit_local_own_dense_data_guard(
                    a,
                    layout,
                    receiver_off,
                    key_src,
                    candidate.pre_fail,
                );
                emit_tdz_objects_nonfinal(
                    a,
                    layout,
                    pattern.value as u32 * 8,
                    pattern.mapped as u32 * 8,
                    candidate.pre_fail,
                );

                // The helper on a last-owner packed drop may clobber every caller-saved
                // register. No borrowed descriptor or object pointer may survive either
                // ordered release; the rooted receiver and preserved numeric key are
                // reloaded only after both TDZ stores below.
                for register in 9..=17 {
                    a.movz(register, 0, 0);
                }
                for slot in [pattern.value, pattern.mapped] {
                    let offset = slot as i32 * 8;
                    emit_exec_release_word(a, layout, 22, offset);
                    a.mov_imm64(9, crate::value::PACK_EMPTY);
                    emit_exec_word_store(a, 9, 22, offset);
                    if let Some(home) = plan.tagged_reg(slot) {
                        a.ldr_d_imm(home, 22, slot as u32 * 8);
                    }
                }

                emit_local_assumed_dense_data_value(
                    a,
                    layout,
                    receiver_off,
                    key_src,
                    candidate.post_fail,
                );
                emit_exec_clone(a, layout, 12, 13, 16, candidate.post_fail);
                // The first Tdz made this destination Empty, so the cloned result can be
                // transferred directly into its canonical owner slot.
                emit_exec_word_store(a, 12, 22, pattern.value as i32 * 8);
                if let Some(home) = plan.tagged_reg(pattern.value) {
                    a.ldr_d_imm(home, 22, pattern.value as u32 * 8);
                }
                #[cfg(test)]
                emit_lowering_event(
                    a,
                    if matches!(key_src, KeySrc::DReg(_)) {
                        record_tdz_array_pair_dreg as *const () as u64
                    } else {
                        record_tdz_array_pair_slot as *const () as u64
                    },
                );
                a.b(candidate.continuation);
                continue;
            }
            // A region keeps its private CFG, but does not discard the baseline's
            // stronger ownership-aware local templates. Spans stay within one block;
            // independent original entries remain separately emitted in the baseline.
            if !parameter_aliases && pc + 3 < end && fast & 2 != 0 && eq_inlinable(layout) {
                if let (
                    Op::LoadLocal(lhs),
                    Op::LoadLocal(rhs),
                    cmp @ (Op::StrictEq | Op::StrictNotEq | Op::EqEq | Op::NotEq),
                    Op::JumpIfFalse(target),
                ) = (&ops[pc], &ops[pc + 1], &ops[pc + 2], &ops[pc + 3])
                {
                    if *lhs as u32 * 8 + 8 < 4096 && *rhs as u32 * 8 + 8 < 4096 {
                        own_facts.clear();
                        materialize(a, layout, &stack, physical);
                        physical = stack.len();
                        stack.fill(Operand::Canonical);
                        plan.flush_local(a, *lhs);
                        plan.flush_local(a, *rhs);
                        let edge = edges.target(a, pc + 3, *target as usize);
                        #[cfg(test)]
                        emit_predicate_event(a);
                        emit_local_eq_branch(
                            a,
                            layout,
                            *lhs as u32 * 8,
                            *rhs as u32 * 8,
                            pc as u32,
                            private_unwind,
                            matches!(cmp, Op::StrictEq | Op::StrictNotEq),
                            matches!(cmp, Op::StrictNotEq | Op::NotEq),
                            edge,
                        );
                        consumed = 3;
                        continue;
                    }
                }
            }
            if pc + 1 < end {
                if fast & (8 | 64) == (8 | 64) {
                    if let Some(slot) = local_read_discard_pair(ops, pc, source_targets) {
                        // No produced owner, no heap effect. Numeric homes cannot be Empty.
                        if plan.reg(slot).is_none() {
                            materialize(a, layout, &stack, physical);
                            physical = stack.len();
                            stack.fill(Operand::Canonical);
                            emit_discard_local_read(a, slot as u32 * 8, pc as u32, private_unwind);
                        }
                        consumed = 1;
                        continue;
                    }
                }
                if fast & (8 | 16 | 64) == (8 | 16 | 64) {
                    if let Some(slot) = local_store_pair(ops, pc, source_targets)
                        .filter(|slot| plan.reg(*slot).is_none())
                    {
                        own_facts.clear();
                        let numeric = stack.last().is_some_and(|value| value.numeric());
                        let before = stack.clone();
                        // Materialize aliases before replacing their canonical source owner.
                        materialize(a, layout, &stack, physical);
                        physical = stack.len();
                        stack.fill(Operand::Canonical);
                        emit_store_local(
                            a,
                            layout,
                            slot as u32 * 8,
                            &[pc as u32, pc as u32 + 1],
                            private_unwind,
                            true,
                            true,
                        );
                        if let Some(home) = plan.tagged_reg(slot) {
                            a.ldr_d_imm(home, 22, slot as u32 * 8);
                        }
                        if numeric {
                            numeric_locals.insert(slot);
                        } else {
                            numeric_locals.remove(&slot);
                        }
                        preserve_scalar_prefix(a, &before, &mut stack, physical);
                        consumed = 1;
                        continue;
                    }
                }
                // Local-source array reads avoid materializing the receiver and key as
                // owning stack Values. Numeric index homes stay live on the dense own-data
                // hit; every other case flushes and replays from the first LoadLocal.
                if fast & 1024 != 0 && get_elem_inlinable(layout) && pc + 2 < end {
                    if let (Op::LoadLocal(receiver), Op::LoadLocal(key), Op::GetElem) =
                        (&ops[pc], &ops[pc + 1], &ops[pc + 2])
                    {
                        let receiver_off = *receiver as u32 * 8;
                        let key_off = *key as u32 * 8;
                        if receiver_off + 8 < 4096
                            && key_off + 8 < 4096
                            // A numeric home may have changed since its canonical slot
                            // was last published. Such a slot cannot be the current receiver.
                            && plan.reg(*receiver).is_none()
                            && !source_targets[pc + 1]
                            && !source_targets[pc + 2]
                        {
                            own_facts.clear();
                            numeric_locals.clear();
                            let before = stack.clone();
                            materialize(a, layout, &stack, physical);
                            physical = stack.len();
                            stack.fill(Operand::Canonical);
                            let slow_reload_fail = a.new_label();
                            bails.push((slow_reload_fail, pc + 3, Vec::new(), 0, false));
                            let key_src =
                                plan.reg(*key).map_or(KeySrc::Slot(key_off), KeySrc::DReg);
                            #[cfg(test)]
                            let event = Some(if matches!(key_src, KeySrc::DReg(_)) {
                                record_local_array_get as *const () as u64
                            } else {
                                record_local_array_get_slot as *const () as u64
                            });
                            #[cfg(not(test))]
                            let event = None;
                            emit_elem_local_keyed(
                                a,
                                layout,
                                receiver_off,
                                &[pc as u32, pc as u32 + 1, pc as u32 + 2],
                                unwind,
                                ElemLocalKind::Get,
                                key_src,
                                Some(plan),
                                slow_reload_fail,
                                event,
                            );
                            physical += 1;
                            stack.push(Operand::Canonical);
                            preserve_scalar_prefix(a, &before, &mut stack, physical);
                            propagate_checked_result(
                                a,
                                plan,
                                pc + 2,
                                &mut stack,
                                physical,
                                &mut bails,
                                &mut result_misses,
                            );
                            consumed = 2;
                            continue;
                        }
                    }
                }
                // `in` is true without a key coercion or prototype walk only when a plain
                // own dense element is present. Holes and every uncertain form replay the
                // complete source sequence, retaining ToPropertyKey/HasProperty semantics.
                if fast & 1024 != 0 && get_elem_inlinable(layout) && pc + 2 < end {
                    if let (Op::LoadLocal(key), Op::LoadLocal(receiver), Op::In) =
                        (&ops[pc], &ops[pc + 1], &ops[pc + 2])
                    {
                        let key_off = *key as u32 * 8;
                        let receiver_off = *receiver as u32 * 8;
                        if receiver_off + 8 < 4096
                            && key_off + 8 < 4096
                            && plan.reg(*receiver).is_none()
                            && !source_targets[pc + 1]
                            && !source_targets[pc + 2]
                        {
                            own_facts.clear();
                            numeric_locals.clear();
                            let before = stack.clone();
                            materialize(a, layout, &stack, physical);
                            physical = stack.len();
                            stack.fill(Operand::Canonical);
                            let slow_reload_fail = a.new_label();
                            bails.push((slow_reload_fail, pc + 3, Vec::new(), 0, false));
                            let key_src =
                                plan.reg(*key).map_or(KeySrc::Slot(key_off), KeySrc::DReg);
                            #[cfg(test)]
                            let event = Some(if matches!(key_src, KeySrc::DReg(_)) {
                                record_local_array_in as *const () as u64
                            } else {
                                record_local_array_in_slot as *const () as u64
                            });
                            #[cfg(not(test))]
                            let event = None;
                            emit_in_local_inline(
                                a,
                                layout,
                                receiver_off,
                                key_src,
                                &[pc as u32, pc as u32 + 1, pc as u32 + 2],
                                unwind,
                                plan,
                                slow_reload_fail,
                                event,
                            );
                            physical += 1;
                            stack.push(Operand::Canonical);
                            preserve_scalar_prefix(a, &before, &mut stack, physical);
                            consumed = 2;
                            continue;
                        }
                    }
                }
                if fast & 1024 != 0 && get_elem_inlinable(layout) {
                    if let (Op::LoadLocal(key), Op::GetElemLocal(receiver)) =
                        (&ops[pc], &ops[pc + 1])
                    {
                        if *key as u32 * 8 + 8 < 4096
                            && *receiver as u32 * 8 + 8 < 4096
                            && plan.reg(*receiver).is_none()
                        {
                            own_facts.clear();
                            numeric_locals.clear();
                            let before = stack.clone();
                            materialize(a, layout, &stack, physical);
                            physical = stack.len();
                            stack.fill(Operand::Canonical);
                            let slow_reload_fail = a.new_label();
                            bails.push((slow_reload_fail, pc + 2, Vec::new(), 0, false));
                            let key_off = *key as u32 * 8;
                            let key_src =
                                plan.reg(*key).map_or(KeySrc::Slot(key_off), KeySrc::DReg);
                            #[cfg(test)]
                            let event = Some(record_local_array_legacy_get as *const () as u64);
                            #[cfg(not(test))]
                            let event = None;
                            emit_elem_local_keyed(
                                a,
                                layout,
                                *receiver as u32 * 8,
                                &[pc as u32, pc as u32 + 1],
                                unwind,
                                ElemLocalKind::Get,
                                key_src,
                                Some(plan),
                                slow_reload_fail,
                                event,
                            );
                            physical += 1;
                            stack = vec![Operand::Canonical; physical];
                            preserve_scalar_prefix(a, &before, &mut stack, physical);
                            propagate_checked_result(
                                a,
                                plan,
                                pc + 1,
                                &mut stack,
                                physical,
                                &mut bails,
                                &mut result_misses,
                            );
                            consumed = 1;
                            continue;
                        }
                    }
                }
            }
            if fast & 16384 != 0 && plan.homes.is_empty() && stack.is_empty() && fusing.is_none() {
                // The established scalar/property/element chain has stronger local
                // representation propagation than independent opcode templates. Reuse
                // it only at an empty operand boundary with no private numeric locals:
                // its d8..d15 temporaries may overwrite borrowed tagged caches, which
                // are reloaded from their owning canonical slots on fallthrough. Any
                // terminal branch exits through an exact original canonical entry.
                if let Some((chain, count)) =
                    build_chain(chunk, &ops[..end], pc, source_targets, layout, fast)
                {
                    own_facts.clear();
                    numeric_locals.clear(); // checked chain suffixes may replace any written slot
                    materialize(a, layout, &stack, physical);
                    #[cfg(test)]
                    emit_lowering_event(a, record_reused_chain as *const () as u64);
                    emit_chain(a, layout, &chain, pc_labels, private_unwind);
                    let mut depth = 0;
                    for at in pc..pc + count {
                        let (pops, pushes) = chunk.jit_stack_effect(at).unwrap();
                        depth = depth - pops + pushes;
                    }
                    physical = depth;
                    stack = vec![Operand::Canonical; depth];
                    for slot in &plan.tagged_homes {
                        a.ldr_d_imm(plan.tagged_reg(*slot).unwrap(), 22, *slot as u32 * 8);
                    }
                    let last = pc + count - 1;
                    if chunk
                        .jit_stack_effect(last)
                        .is_some_and(|(_, pushes)| pushes == 1)
                    {
                        propagate_checked_result(
                            a,
                            plan,
                            last,
                            &mut stack,
                            physical,
                            &mut bails,
                            &mut result_misses,
                        );
                    }
                    consumed = count - 1;
                    continue;
                }
            }
            // A fused name/element access (see `NameElementPlan`): the LoadName waits for the
            // element operation, which reads the binding itself.
            if !parameter_aliases
                && fusing.is_none()
                && tdz_array_pairs.is_empty()
                && matches!(ops[pc], Op::LoadName(..))
            {
                if let Some(found) = plan_name_element(
                    chunk,
                    ops,
                    pc,
                    end,
                    source_targets,
                    plan,
                    layout,
                    fast,
                    stack.len(),
                ) {
                    own_facts.clear();
                    // A set, or a get consumed as a Number, keeps the operands below the name
                    // in their registers; any other result is published, so is what a borrowed
                    // view would depend on.
                    let lazy = (found.set || plan.numeric_results.contains(&found.elem_pc))
                        && !stack
                            .iter()
                            .any(|o| matches!(o, Operand::Borrowed(_) | Operand::BorrowedValue));
                    let pre = if lazy {
                        // The continuation clobbers the integer registers: below the name only
                        // double registers survive.
                        for index in 0..stack.len() {
                            if matches!(stack[index], Operand::Int { .. }) {
                                scalar_register(a, &mut stack, physical, index);
                            }
                        }
                        stack.clone()
                    } else {
                        let before = stack.clone();
                        materialize(a, layout, &stack, physical);
                        physical = stack.len();
                        stack = vec![Operand::Canonical; physical];
                        preserve_scalar_prefix(a, &before, &mut stack, physical);
                        stack.clone()
                    };
                    saved_work += 6;
                    fusing = Some((pc, found, pre, lazy));
                    continue;
                }
            }
            if fusing
                .as_ref()
                .is_some_and(|(_, found, _, _)| found.elem_pc == pc)
            {
                let (name_pc, found, pre, lazy) = fusing.take().unwrap();
                saved_work += 8;
                fused.push(emit_name_element(
                    a,
                    layout,
                    chunk,
                    &mut stack,
                    physical,
                    pre.clone(),
                    lazy,
                    name_pc,
                    found,
                ));
                if lazy {
                    // The fast path left every register where it was and (for a get) the
                    // result in the key's register; the continuation restores the same
                    // picture. Words the continuation published are valid, registers are not.
                    stack = pre;
                    for (index, operand) in stack.iter_mut().enumerate() {
                        if let Operand::Scalar(state) = *operand {
                            if state.canonical_slot.is_some() {
                                *operand = Operand::clean(state.kind, index, false);
                            }
                        }
                    }
                    if !found.set {
                        stack.push(Operand::temporary(ScalarKind::Number));
                    }
                } else {
                    physical = pre.len() + usize::from(!found.set);
                    stack = vec![Operand::Canonical; physical];
                    preserve_scalar_prefix(a, &pre, &mut stack, physical);
                    if !found.set {
                        propagate_checked_result(
                            a,
                            plan,
                            found.elem_pc,
                            &mut stack,
                            physical,
                            &mut bails,
                            &mut result_misses,
                        );
                    }
                }
                continue;
            }
            let bailout = a.new_label();
            bails.push((bailout, pc, stack.clone(), physical, true));
            if jit_map_enabled() {
                // Profiler attribution (`TRUST_JS_JIT_MAP`): the code of each region operation
                // belongs to `region:<operation>`, not to the region as a whole.
                a.mark_op(region_mark(&ops[pc]));
            }
            if let Some((name_pc, _, pre, _)) = &fusing {
                // Inside the pure window a miss replays from the LoadName with the stack as it
                // was: nothing between has any effect.
                let bail = bails.last_mut().unwrap();
                bail.1 = *name_pc;
                bail.2 = pre.clone();
            }
            let depth = stack.len();
            match ops[pc] {
                Op::InlineGuard(target, mismatch) => {
                    // GetValue/argument evaluation already happened. A failed guard resumes
                    // at the original call-alternative PC with those exact captured owners;
                    // it never re-reads a method or replays an argument (EvaluateCall).
                    bails.last_mut().unwrap().1 = mismatch as usize;
                    guard_misses.push(bailout);
                    let guard = chunk.jit_inline_target(target);
                    let stored = guard.pin.upgrade().filter(|_| layout.valid).map(|object| {
                        let owner: Option<crate::value::Gc> = Some(crate::value::Gc::from(object));
                        unsafe { *(&owner as *const Option<crate::value::Gc> as *const usize) }
                    });
                    if let Some(stored) = stored {
                        if guard.expected_env != 0 {
                            a.ldr_imm(9, 19, 40);
                            a.mov_imm64(10, guard.expected_env as u64);
                            a.cmp_reg_x(9, 10);
                            a.b_cond(C_NE, bailout);
                        }
                        // Same-Realm proof (see `InlineTarget::expected_genv`).
                        a.ldr_imm(9, 19, std::mem::offset_of!(crate::jit::JitCtx, genv) as u32);
                        a.mov_imm64(10, guard.expected_genv as u64);
                        a.cmp_reg_x(9, 10);
                        a.b_cond(C_NE, bailout);
                        let callee = depth - guard.argc as usize - 1;
                        guard_operand(a, &stack, physical, callee, bailout);
                        a.mov_imm64(10, crate::value::PACK_OBJ | stored as u64);
                        a.cmp_reg_x(9, 10);
                        a.b_cond(C_NE, bailout);
                        if guard.check_this {
                            guard_operand(a, &stack, physical, callee - 1, bailout);
                            emit_exec_tag_guard(a, 9, crate::value::PACK_OBJ, 10, bailout);
                        }
                        #[cfg(test)]
                        emit_guarded_event(a, 1);
                    } else {
                        a.b(bailout);
                    }
                }
                Op::Return | Op::ReturnBare | Op::ReturnUndef | Op::Throw => {
                    // A region is not a completion driver. Retain the exact ordinary/borrrowed
                    // return or throw template, including pending finally/iterator semantics.
                    a.b(bailout);
                }
                Op::GetMethod(_, cache)
                    if fast & 256 != 0
                        && get_prop_inlinable(layout)
                        && plan.borrowed_methods.contains(&pc)
                        && borrowed_method_state(chunk, cache).is_some() =>
                {
                    saved_work += 2; // no owning push/pop for the borrowed method
                    emit_borrowed_method(
                        a,
                        layout,
                        borrowed_method_state(chunk, cache).unwrap(),
                        &stack,
                        physical,
                        bailout,
                    );
                    stack.push(Operand::BorrowedValue);
                }
                Op::GetPropLocal(_, _, _) | Op::GetPropThis(_, _)
                    if fast & 256 != 0
                        && get_prop_inlinable(layout)
                        && layout.entry_accessor == layout.entry_value + 8
                        && borrowed_own_state(chunk, &ops[pc]).is_some()
                        && !matches!(ops[pc], Op::GetPropLocal(slot, ..)
                            if plan.reg(slot).is_some() || slot as u32 * 8 + 8 >= 4096) =>
                {
                    saved_work += 2; // no property owner clone/drop
                    let (receiver, cache) = match ops[pc] {
                        Op::GetPropLocal(slot, _, cache) => {
                            (PropRecv::Slot(slot as u32 * 8), chunk.jit_cache_ptr(cache))
                        }
                        Op::GetPropThis(_, cache) => (PropRecv::This, chunk.jit_cache_ptr(cache)),
                        _ => unreachable!(),
                    };
                    let previous = own_facts.record(receiver, cache);
                    emit_borrowed_own_read(
                        a,
                        layout,
                        receiver,
                        cache,
                        previous,
                        16 + depth as u32,
                        bailout,
                    );
                    stack.push(Operand::BorrowedValue);
                }
                Op::LoadLocal(slot) if plan.reg(slot).is_some() => {
                    saved_work += 2;
                    a.fmov_d_d(16 + depth as u32, plan.reg(slot).unwrap());
                    stack.push(Operand::temporary(ScalarKind::Number));
                }
                Op::LoadLocal(slot) if fusing.is_some() && slot < 256 => {
                    // Inside a fused name/element window: a pure Number read of the frame slot
                    // (no owner, no stack word). Anything else replays from the LoadName.
                    saved_work += 2;
                    emit_exec_word_load(a, 9, 22, slot as i32 * 8);
                    emit_exec_number_guard(a, 9, 16 + depth as u32, 10, bailout);
                    stack.push(Operand::temporary(ScalarKind::Number));
                }
                Op::LoadLocal(slot) if numeric_locals.contains(&slot) => {
                    saved_work += 2;
                    if let Some(home) = plan.tagged_reg(slot) {
                        a.fmov_d_d(16 + depth as u32, home);
                    } else {
                        emit_exec_word_load(a, 9, 22, slot as i32 * 8);
                        a.fmov_d_x(16 + depth as u32, 9);
                    }
                    stack.push(Operand::temporary(ScalarKind::Number));
                    #[cfg(test)]
                    emit_property_event(a, 6);
                }
                Op::Const(k) if chunk.jit_const_num(k).is_some() => {
                    saved_work += 1;
                    let bits = chunk.jit_const_num(k).unwrap();
                    let value = f64::from_bits(bits);
                    if let Some(constant) = Operand::int32_constant(bits) {
                        // Emitted only if a consumer needs a register or the canonical word.
                        stack.push(constant);
                    } else {
                        a.mov_imm64(9, bits);
                        a.fmov_d_x(16 + depth as u32, 9);
                        stack.push(
                            if value.is_finite()
                                && value.fract() == 0.0
                                && value.abs() < 9223372036854775808.0
                            {
                                Operand::temporary(ScalarKind::Integer)
                            } else {
                                Operand::temporary(ScalarKind::Number)
                            },
                        );
                    }
                }
                Op::LoadLocal(slot)
                    if slot < 256
                        && plan.tagged_reg(slot).is_none()
                        && plan.numeric_results.contains(&pc) =>
                {
                    // A frame slot that is not a numeric home but feeds arithmetic: read it as
                    // a Number straight into a register, with no owner and no stack word. A
                    // slot that is not a Number leaves the region at this read.
                    saved_work += 2;
                    emit_exec_word_load(a, 9, 22, slot as i32 * 8);
                    emit_exec_number_guard(a, 9, 16 + depth as u32, 10, bailout);
                    stack.push(Operand::temporary(ScalarKind::Number));
                }
                Op::LoadLocal(slot) if plan.tagged_reg(slot).is_some() => {
                    saved_work += 2;
                    let home = plan.tagged_reg(slot).unwrap();
                    a.fmov_x_d(10, home);
                    a.mov_imm64(9, crate::value::PACK_EMPTY);
                    a.cmp_reg_x(10, 9);
                    a.b_cond(C_EQ, bailout);
                    a.lsr_imm(9, 10, 48);
                    a.movz(11, (crate::value::PACK_BIGINT >> 48) as u32, 0);
                    a.cmp_reg_x(9, 11);
                    a.b_cond(C_EQ, bailout);
                    stack.push(Operand::Borrowed(home));
                }
                Op::LoadLocal(slot) if slot < 256 => {
                    // Make the local's one canonical stack owner without a helper. No author
                    // code executes, so numeric homes need no spill/reload. BigInt cloning
                    // and TDZ errors retain the exact original-op fallback.
                    let off = slot as i32 * 8;
                    emit_exec_word_load(a, 10, 22, off);
                    a.mov_imm64(9, crate::value::PACK_EMPTY);
                    a.cmp_reg_x(10, 9);
                    a.b_cond(C_EQ, bailout);
                    emit_exec_clone(a, layout, 10, 11, 12, bailout);
                    let dest = (depth as i32 - physical as i32) * 8;
                    emit_exec_word_store(a, 10, 20, dest);
                    stack.push(Operand::Canonical);
                }
                Op::StoreLocal(slot) if plan.reg(slot).is_some() => {
                    saved_work += 2;
                    // This slot's numeric-home guard proves it cannot be a
                    // borrowed receiver. No property owner is displaced here,
                    // and a guarded Number PutValue into a private local cannot
                    // invoke author code or change an OrdinaryGet descriptor.
                    // A non-number leaves the region before this store commits.
                    if let Operand::Int { reg, unsigned } = stack[depth - 1] {
                        let home = plan.reg(slot).unwrap();
                        if unsigned {
                            a.ucvtf_d_w(home, u32::from(reg));
                        } else {
                            a.scvtf_d_w(home, u32::from(reg));
                        }
                    } else {
                        number(a, &mut stack, physical, depth - 1, bailout);
                        a.fmov_d_d(plan.reg(slot).unwrap(), 15 + depth as u32);
                    }
                    stack.pop();
                }
                Op::StoreLocal(slot)
                    if slot < 256
                        && plan.reg(slot).is_none()
                        && plan.tagged_reg(slot).is_none()
                        && stack[depth - 1].numeric() =>
                {
                    // A Number into a frame slot that is not a numeric home. The old word owns
                    // nothing (a slot holding a reference leaves the region instead), so the
                    // store is a plain overwrite; no author code runs, and the slot's own-read
                    // facts end here.
                    saved_work += 3;
                    own_facts.clear();
                    number(a, &mut stack, physical, depth - 1, bailout);
                    emit_slot_owns_nothing(a, slot as i32 * 8, bailout);
                    emit_exec_number_store(a, 15 + depth as u32, 22, slot as i32 * 8, 9);
                    stack.pop();
                    numeric_locals.insert(slot);
                }
                Op::UpdateLocal(slot, kind) if plan.reg(slot).is_some() => {
                    saved_work += 2;
                    // ECMA-262 UpdateExpression: this Number-only local home
                    // needs no ToPrimitive/ToNumeric call or property [[Set]].
                    // The guarded receiver/data-descriptor facts remain valid.
                    let reg = plan.reg(slot).unwrap();
                    let post = matches!(kind, UpdKind::PostInc | UpdKind::PostDec);
                    let pushes = !matches!(kind, UpdKind::IncDiscard | UpdKind::DecDiscard);
                    if post {
                        a.fmov_d_d(16 + depth as u32, reg);
                    }
                    a.fmov_one(0);
                    a.f_arith(
                        u32::from(matches!(
                            kind,
                            UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
                        )),
                        reg,
                        reg,
                        0,
                    );
                    if pushes {
                        if !post {
                            a.fmov_d_d(16 + depth as u32, reg);
                        }
                        stack.push(Operand::temporary(ScalarKind::Number));
                    }
                }
                Op::StrictEq | Op::StrictNotEq | Op::EqEq | Op::NotEq
                    if !parameter_aliases
                        && !stack[depth - 2..].iter().all(|value| value.numeric()) =>
                {
                    // Equality is not exclusively numeric. Preserve the ordinary
                    // identity/nullish/string/coercion template inside the region
                    // rather than leaving the region on every object comparison.
                    // IsLooselyEqual may call author conversion methods; all
                    // operand owners are canonical before entering that fallback.
                    own_facts.clear();
                    let before = stack.clone();
                    materialize(a, layout, &stack, physical);
                    let branch = if pc + 1 < end {
                        match ops[pc + 1] {
                            Op::JumpIfFalse(target) => {
                                let edge = edges.target(a, pc + 1, target as usize);
                                Some(edge)
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    #[cfg(test)]
                    if branch.is_some() {
                        emit_predicate_event(a);
                    }
                    if fast & 2 != 0 && eq_inlinable(layout) {
                        emit_eq_inline(
                            a,
                            layout,
                            pc as u32,
                            private_unwind,
                            matches!(ops[pc], Op::StrictEq | Op::StrictNotEq),
                            matches!(ops[pc], Op::StrictNotEq | Op::NotEq),
                            branch,
                        );
                    } else {
                        emit_exec(a, pc as u32, private_unwind);
                        if let Some(edge) = branch {
                            emit_cond(a, COND_POP_TRUTHY, private_unwind);
                            a.cbz(1, false, edge);
                        }
                    }
                    if branch.is_some() {
                        physical = depth - 2;
                        stack = vec![Operand::Canonical; physical];
                        consumed = 1;
                        continue;
                    }
                    physical = depth - 1;
                    stack = vec![Operand::Canonical; physical];
                    preserve_scalar_prefix(a, &before[..depth - 2], &mut stack, physical);
                    // The checked abstract operation always publishes Boolean.
                    // Its type is known, but a register is needed only by a
                    // consumer; intervening effects need no load/store traffic.
                    stack[physical - 1] = Operand::clean(ScalarKind::Boolean, physical - 1, false);
                }
                Op::Add
                | Op::Sub
                | Op::Mul
                | Op::Div
                | Op::Lt
                | Op::Gt
                | Op::Le
                | Op::Ge
                | Op::StrictEq
                | Op::StrictNotEq
                | Op::EqEq
                | Op::NotEq => {
                    saved_work += stack[depth - 2..]
                        .iter()
                        .filter(|value| value.numeric())
                        .count();
                    let integers = int_comparison(&ops[pc], &stack[depth - 2..]);
                    if integers.is_none() {
                        number(a, &mut stack, physical, depth - 2, bailout);
                        number(a, &mut stack, physical, depth - 1, bailout);
                    }
                    let left = 14 + depth as u32;
                    let right = left + 1;
                    let arithmetic = match ops[pc] {
                        Op::Add => Some(0),
                        Op::Sub => Some(1),
                        Op::Mul => Some(2),
                        Op::Div => Some(3),
                        _ => None,
                    };
                    let operands = [stack[depth - 2], stack[depth - 1]];
                    stack.pop();
                    stack.pop();
                    if let Some(op) = arithmetic {
                        a.f_arith(op, left, left, right);
                        stack.push(Operand::temporary(ScalarKind::Number));
                    } else {
                        let cond = match (integers, &ops[pc]) {
                            (Some(domain), op) => domain.condition(op),
                            (None, Op::Lt) => C_MI,
                            (None, Op::Gt) => C_GT,
                            (None, Op::Le) => C_LS,
                            (None, Op::Ge) => C_GE,
                            (None, Op::EqEq | Op::StrictEq) => C_EQ,
                            (None, _) => C_NE,
                        };
                        if let Some(Op::JumpIfFalse(target)) = (pc + 1 < end).then(|| &ops[pc + 1])
                        {
                            // Number::equal treats NaN as unequal and signed zeroes as equal.
                            // Materialization canonicalizes NaN/clobbers NZCV, so compare LAST.
                            #[cfg(test)]
                            emit_predicate_event(a);
                            materialize(a, layout, &stack, physical);
                            match integers {
                                Some(domain) => emit_int_compare(a, domain, operands),
                                None => a.fcmp(left, right),
                            }
                            let edge = edges.target(a, pc + 1, *target as usize);
                            a.b_cond(cond ^ 1, edge);
                            physical = stack.len();
                            stack.fill(Operand::Canonical);
                            consumed = 1;
                        } else {
                            match integers {
                                Some(domain) => emit_int_compare(a, domain, operands),
                                None => a.fcmp(left, right),
                            }
                            a.cset_w(9, cond);
                            a.mov_imm64(10, crate::value::PACK_BOOL);
                            a.logic_x(1, 9, 9, 10);
                            a.fmov_d_x(left, 9);
                            stack.push(Operand::temporary(ScalarKind::Boolean));
                        }
                    }
                }
                Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr | Op::UShr => {
                    let index = depth - 2;
                    let pool = Operand::int_pool(index);
                    let lhs =
                        int32_operand(a, &mut stack, physical, index, pool.unwrap_or(12), bailout);
                    let rhs = int32_operand(a, &mut stack, physical, depth - 1, 13, bailout);
                    let unsigned = matches!(ops[pc], Op::UShr);
                    let source = int32_binary(a, &ops[pc], lhs, rhs, pool.unwrap_or(11), 13);
                    stack.truncate(depth - 2);
                    if let Some(pool) = pool {
                        if source != pool {
                            a.mov_w(pool, source); // an identity operation of another register
                        }
                        stack.push(Operand::Int {
                            reg: pool as u8,
                            unsigned,
                        });
                    } else {
                        let result = 14 + depth as u32;
                        if unsigned {
                            a.ucvtf_d_w(result, source);
                        } else {
                            a.scvtf_d_w(result, source);
                        }
                        stack.push(Operand::temporary(ScalarKind::Integer));
                    }
                }
                Op::BitNot => {
                    // ~x is ToInt32(x) XOR -1 (ECMA-262 §13.5.6): one MVN.
                    let pool = Operand::int_pool(depth - 1);
                    let target = pool.unwrap_or(9);
                    match int32_operand(a, &mut stack, physical, depth - 1, target, bailout) {
                        IntOperand::Imm(value) => a.mov_imm64(target, u64::from(!value as u32)),
                        IntOperand::Reg(register) => a.mvn_w(target, register),
                    }
                    if let Some(pool) = pool {
                        stack[depth - 1] = Operand::Int {
                            reg: pool as u8,
                            unsigned: false,
                        };
                    } else {
                        a.scvtf_d_w(15 + depth as u32, target);
                        stack[depth - 1] = Operand::temporary(ScalarKind::Integer);
                    }
                }
                Op::Neg | Op::Plus => {
                    number(a, &mut stack, physical, depth - 1, bailout);
                    if matches!(ops[pc], Op::Neg) {
                        a.fneg(15 + depth as u32, 15 + depth as u32);
                        stack[depth - 1] = Operand::temporary(ScalarKind::Number);
                    }
                }
                Op::Pop if !matches!(stack.last(), Some(Operand::Canonical)) => {
                    stack.pop();
                }
                Op::Dup if !matches!(stack.last(), Some(Operand::Canonical)) => {
                    duplicate_operand(a, &mut stack, physical);
                }
                Op::Jump(target) => {
                    own_facts.clear();
                    materialize(a, layout, &stack, physical);
                    edges.transfer(a, pc, target as usize, physical_next);
                }
                Op::JumpIfFalse(target) => {
                    own_facts.clear();
                    if stack[depth - 1].kind() == Some(ScalarKind::Boolean) {
                        scalar_register(a, &mut stack, physical, depth - 1);
                        stack.pop();
                        materialize(a, layout, &stack, physical);
                        a.fmov_x_d(9, 15 + depth as u32);
                        a.logic_imm_w(0, 9, 9, asm::logical_imm_w(1).unwrap());
                        let edge = edges.target(a, pc, target as usize);
                        a.cbz(9, false, edge);
                    } else if let Operand::Int { reg, .. } = stack[depth - 1] {
                        // ToBoolean of an integral Number: false only for zero.
                        stack.pop();
                        materialize(a, layout, &stack, physical);
                        let edge = edges.target(a, pc, target as usize);
                        a.cbz(u32::from(reg), false, edge);
                    } else if stack[depth - 1].numeric() {
                        scalar_register(a, &mut stack, physical, depth - 1);
                        stack.pop();
                        materialize(a, layout, &stack, physical);
                        // Numeric stores canonicalize NaNs and clobber NZCV. The condition's
                        // virtual register remains live until this comparison after spilling.
                        a.fcmp_zero(15 + depth as u32);
                        let edge = edges.target(a, pc, target as usize);
                        a.b_cond(C_EQ, edge);
                        a.b_cond(C_VS, edge);
                    } else {
                        materialize(a, layout, &stack, physical);
                        emit_pop_cond(a, layout, private_unwind);
                        let edge = edges.target(a, pc, target as usize);
                        a.cbz(1, false, edge);
                        stack.pop();
                    }
                    physical = stack.len();
                    stack.fill(Operand::Canonical);
                }
                Op::JumpIfFalsePeek(target)
                | Op::JumpIfTruePeek(target)
                | Op::JumpIfNotNullishPeek(target) => {
                    own_facts.clear();
                    let value = *stack.last().unwrap();
                    let nullish = matches!(ops[pc], Op::JumpIfNotNullishPeek(_));
                    if value.scalar() && !nullish {
                        scalar_register(a, &mut stack, physical, depth - 1);
                    }
                    materialize(a, layout, &stack, physical);
                    // Preserve the ordinary native truthiness/nullish template. ToBoolean
                    // has no author effects, so private homes need no spill or reload.
                    let edge = edges.target(a, pc, target as usize);
                    let when_false = matches!(ops[pc], Op::JumpIfFalsePeek(_));
                    if nullish && value.scalar() {
                        a.b(edge); // these exact language types are never nullish
                    } else if value.kind() == Some(ScalarKind::Boolean) {
                        a.fmov_x_d(9, 15 + depth as u32);
                        a.logic_imm_w(0, 9, 9, asm::logical_imm_w(1).unwrap());
                        if when_false {
                            a.cbz(9, false, edge);
                        } else {
                            a.cbnz(9, false, edge);
                        }
                    } else if value.numeric() {
                        a.fcmp_zero(15 + depth as u32);
                        if when_false {
                            a.b_cond(C_EQ, edge);
                            a.b_cond(C_VS, edge);
                        } else {
                            // Nonzero AND ordered; ARM NE alone also includes NaN.
                            let no = a.new_label();
                            a.b_cond(C_VS, no);
                            a.b_cond(C_NE, edge);
                            a.bind(no);
                        }
                    } else {
                        emit_peek_cond_inline(a, layout, nullish, private_unwind);
                        if when_false {
                            a.cbz(1, false, edge);
                        } else {
                            a.cbnz(1, false, edge);
                        }
                    }
                    physical = stack.len();
                    stack.fill(Operand::Canonical);
                }
                _ => {
                    own_facts.clear();
                    let before = stack.clone();
                    materialize(a, layout, &stack, physical);
                    // Unlike an ordinary free-name read, the complete accumulation also
                    // reads a local RHS and can leave this region after its store. Publish all
                    // private homes before the helper, retaining exact canonical exit state.
                    let accumulation = a.string_accumulation(pc as u32).is_some();
                    let effect = if accumulation {
                        FrameEffect::Full
                    } else {
                        region_frame_effect(&ops[pc], plan, parameter_aliases)
                    };
                    match effect {
                        FrameEffect::Independent => {}
                        FrameEffect::Read(slot) | FrameEffect::Write(slot) => {
                            plan.flush_local(a, slot)
                        }
                        FrameEffect::Full => plan.flush_locals(a),
                    }
                    if accumulation {
                        emit_string_accumulation(a, pc as u32, false);
                    }
                    #[cfg(test)]
                    if let Op::Tdz(slot) = ops[pc] {
                        let path = match effect {
                            FrameEffect::Write(slot)
                                if plan.tagged_reg(slot).is_some() && !plan.homes.is_empty() =>
                            {
                                0
                            }
                            FrameEffect::Write(slot) if plan.tagged_reg(slot).is_some() => 1,
                            FrameEffect::Write(_) => 2,
                            FrameEffect::Full if plan.reg(slot).is_some() => 3,
                            _ => 4,
                        };
                        emit_tdz_effect_event(a, path);
                    }
                    #[cfg(test)]
                    if effect == FrameEffect::Independent && !plan.homes.is_empty() {
                        match ops[pc] {
                            Op::In => emit_independent_effect_live_home_event(a, 0),
                            Op::Abstract(
                                crate::bytecode::AbstractOp::CreateDataPropertyOrThrow,
                            ) => emit_independent_effect_live_home_event(a, 1),
                            _ => {}
                        }
                    }
                    emit_effect(
                        a,
                        chunk,
                        layout,
                        ilayout,
                        pc,
                        fast,
                        array_intrinsics_on,
                        function_call_intrinsic_on,
                        if effect == FrameEffect::Full {
                            unwind
                        } else {
                            private_unwind
                        },
                        direct_finish,
                    );
                    let (pops, pushes) = chunk.jit_stack_effect(pc).unwrap();
                    physical = depth - pops + pushes;
                    stack = vec![Operand::Canonical; physical];
                    if effect == FrameEffect::Full {
                        numeric_locals.clear();
                        let fail = a.new_label();
                        bails.push((fail, pc + 1, Vec::new(), 0, false));
                        plan.reload(a, fail);
                    } else if let FrameEffect::Write(slot) = effect {
                        // Numeric-home writes have their dedicated guarded templates above.
                        // Do not reload unrelated homes after binding an inline parameter.
                        debug_assert!(plan.reg(slot).is_none());
                        if let Some(home) = plan.tagged_reg(slot) {
                            a.ldr_d_imm(home, 22, slot as u32 * 8);
                        }
                        if matches!(ops[pc], Op::StoreLocal(_))
                            && before.last().is_some_and(|value| value.numeric())
                        {
                            numeric_locals.insert(slot);
                        } else {
                            numeric_locals.remove(&slot);
                        }
                    }
                    if effect != FrameEffect::Full {
                        preserve_scalar_prefix(a, &before[..depth - pops], &mut stack, physical);
                    }
                    if matches!(
                        ops[pc],
                        Op::GetProp(..) | Op::GetPropLocal(..) | Op::GetPropThis(..)
                    ) {
                        #[cfg(test)]
                        emit_property_event(a, 3);
                    }
                    if pushes == 1 {
                        propagate_checked_result(
                            a,
                            plan,
                            pc,
                            &mut stack,
                            physical,
                            &mut bails,
                            &mut result_misses,
                        );
                    }
                }
            }
        }
        if tdz_array_exit_blocks.contains(&start) {
            // The fused branch above either deoptimizes at the original In or jumps
            // directly to the post-StoreLocal continuation. It owns both CFG edges.
            continue;
        }
        let guarded_fallthrough = matches!(ops[end - 1], Op::InlineGuard(..))
            && plan
                .blocks
                .get(block_index + 1)
                .is_some_and(|(next, _)| *next == end)
            && cfg.block_at(end).is_some_and(|next| {
                let block = &cfg.blocks()[next.0 as usize];
                block.predecessors.len() == 1 && cfg.dominates(cfg.block_at(start).unwrap(), next)
            });
        if guarded_fallthrough {
            // A unique dominated success edge retains the same virtual owner map. All
            // independent entries still target the separate baseline labels below.
            carried = Some((end, stack, physical, own_facts));
        } else if !matches!(
            ops[end - 1],
            Op::Jump(_) | Op::Return | Op::ReturnBare | Op::ReturnUndef | Op::Throw
        ) {
            materialize(a, layout, &stack, physical);
            // ECMA-262 If/Logical/Conditional Evaluation: same selected value,
            // owners and completion, with no extra control-flow-only trampoline.
            edges.transfer(a, end - 1, end, physical_next);
        }
    }
    for (label, from, target) in edges.deferred {
        a.bind(label);
        if let Some(&to) = labels.get(&target) {
            if target <= from {
                let poll = a.new_label();
                exits.push((poll, target));
                #[cfg(feature = "optimizing-jit")]
                if entry_kind == NativeEntryKind::FreshFrame {
                    if let Some(counter) = optimizing::loop_entry::counter(chunk, target) {
                        let done = a.new_label();
                        a.mov_imm64(9, counter as usize as u64);
                        a.ldr_w_imm(10, 9, 0);
                        a.cbz(10, false, done);
                        // Leave the due visit for the canonical header: its
                        // checkpoint sees every private home flushed exactly once.
                        a.cmp_imm_w(10, 1);
                        a.b_cond(C_EQ, poll);
                        a.sub_imm(10, 10, 1);
                        a.str_w_imm(10, 9, 0);
                        a.bind(done);
                    }
                }
                emit_region_poll_guard(a, ilayout, poll, true);
            }
            a.b(to);
        } else {
            plan.flush_locals(a);
            a.b(pc_labels[target]);
        }
    }
    for (label, target) in exits {
        a.bind(label);
        plan.flush_locals(a);
        a.b(pc_labels[target]);
    }
    for site in &fused {
        emit_name_element_fallback(a, layout, site, private_unwind, &mut bails);
    }
    let referenced: crate::fasthash::FastSet<_> = a.referenced_since(emission_start).collect();
    for (label, pc, stack, physical, flush) in bails {
        if !referenced.contains(&label) {
            continue;
        }
        a.bind(label);
        #[cfg(test)]
        if guard_misses.contains(&label) {
            emit_guarded_event(a, 2);
        }
        #[cfg(test)]
        if result_misses.contains(&label) {
            emit_property_event(a, 5);
        }
        if flush {
            materialize(a, layout, &stack, physical);
            plan.flush_locals(a);
        }
        targets.push(pc);
        a.b(if pc == plan.head {
            plain
        } else {
            pc_labels[pc]
        });
    }
    a.bind(private_unwind);
    plan.flush_locals(a);
    a.b(unwind);
    Emission {
        plain,
        targets,
        saved_work,
        entry_work: 1 + plan.homes.len() * 2 + plan.tagged_homes.len(),
        baseline_bytes: plan
            .blocks
            .iter()
            .flat_map(|&(start, end)| &ops[start..end])
            .map(baseline_bytes)
            .sum(),
        bytes: (a.buf.len() - emission_start.0) * 4,
        acyclic: plan.acyclic,
    }
}

/// ToBoolean followed by consuming the canonical owner. The peek emitter preserves w1;
/// shared-owner dropping only uses x9..x11/NZCV. The rare last-owner/BigInt fallback
/// repeats only ToBoolean (never author code), before any ownership mutation commits.
fn emit_pop_cond(a: &mut asm::Asm, layout: &crate::value::JitLayout, unwind: usize) {
    let slow = a.new_label();
    let done = a.new_label();
    emit_peek_cond_inline(a, layout, false, unwind);
    emit_exec_word_load(a, 9, 20, -8);
    emit_exec_drop_shared(a, layout, 9, 10, 11, slow);
    a.sub_imm(20, 20, 8);
    a.b(done);
    a.bind(slow);
    emit_cond(a, COND_POP_TRUTHY, unwind);
    a.bind(done);
}

fn emit_effect(
    a: &mut asm::Asm,
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    pc: usize,
    fast: u32,
    array_intrinsics_on: bool,
    function_call_intrinsic_on: bool,
    unwind: usize,
    direct_finish: usize,
) {
    // Keep an isolated ablation of the old region vocabulary. Baseline emission
    // always uses the same operation implementation; normal builds add no runtime
    // flag check to generated code.
    static SHARED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *SHARED
        .get_or_init(|| std::env::var("TRUST_JS_SHARED_NATIVE_OPERATIONS").as_deref() != Ok("0"))
        && operations::emit(
            a,
            chunk,
            layout,
            ilayout,
            pc,
            fast,
            array_intrinsics_on,
            function_call_intrinsic_on,
            unwind,
            direct_finish,
        )
    {
        return;
    }
    let op = &chunk.jit_ops()[pc];
    let pc_u32 = pc as u32;
    if fast & 8192 != 0 && names::emit_reference_op(a, layout, ilayout, chunk, op, pc_u32, unwind) {
        return;
    }
    match *op {
        Op::LoadThis if fast & 32768 != 0 => {
            let slow = a.new_label();
            let done = a.new_label();
            a.ldr_imm(9, 19, 48); // wide ctx.this_raw, as in the baseline template
            a.ldr_imm(10, 9, 0);
            a.ldr_imm(11, 9, 8);
            emit_exec_encode_wide(a, 10, 11, 12, 13, 14, 0, slow);
            emit_exec_clone(a, layout, 12, 13, 14, slow);
            emit_exec_word_store(a, 12, 20, 0);
            a.add_imm(20, 20, 8);
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc_u32, unwind);
            a.bind(done);
        }
        Op::Undef if fast & 128 != 0 => {
            a.mov_imm64(9, crate::value::PACK_UNDEFINED);
            emit_exec_word_store(a, 9, 20, 0);
            a.add_imm(20, 20, 8);
        }
        Op::Const(k) if fast & 128 != 0 && chunk.jit_const_copyable(k) => {
            a.mov_imm64(9, chunk.jit_const_packed_bits(k).unwrap());
            emit_exec_word_store(a, 9, 20, 0);
            a.add_imm(20, 20, 8);
        }
        Op::Const(k)
            if fast & 128 != 0 && layout.rc_strong_off == 0 && chunk.jit_const_is_str(k) =>
        {
            a.mov_imm64(9, chunk.jit_const_ptr(k) as u64);
            a.ldr_imm(11, 9, 8);
            a.mov_imm64(10, crate::value::PACK_STR);
            a.logic_x(1, 10, 10, 11);
            emit_exec_word_store(a, 10, 20, 0);
            a.ldur(13, 11, 0);
            a.add_imm(13, 13, 1);
            a.stur(13, 11, 0);
            a.add_imm(20, 20, 8);
        }
        Op::GetProp(name, cache)
        | Op::GetPropThis(name, cache)
        | Op::GetPropLocal(_, name, cache)
        | Op::GetMethod(name, cache)
            if fast
                & if matches!(op, Op::GetMethod(..)) {
                    512
                } else {
                    256
                }
                != 0
                && get_method_inlinable(layout)
                && !matches!(op, Op::GetPropLocal(slot, ..) if *slot as u32 * 8 + 8 >= 4096) =>
        {
            let receiver = match *op {
                Op::GetPropThis(..) => PropRecv::This,
                Op::GetPropLocal(slot, ..) => PropRecv::Slot(slot as u32 * 8),
                _ => PropRecv::Stack,
            };
            emit_prop_load_inline(
                a,
                layout,
                ilayout,
                chunk.jit_cache_ptr(cache),
                chunk.jit_cache_preferred(cache),
                chunk.jit_name(name),
                pc as u32,
                unwind,
                matches!(op, Op::GetMethod(..)),
                !chunk
                    .jit_name(name)
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_digit),
                receiver,
            );
        }
        Op::GetMethodElem if fast & 1024 != 0 && get_method_inlinable(layout) && ilayout.valid => {
            emit_computed_method_inline(a, layout, ilayout, pc_u32, unwind);
        }
        Op::GetElem if fast & 1024 != 0 && get_elem_inlinable(layout) => {
            emit_get_elem_inline(a, layout, pc_u32, unwind);
        }
        Op::SetElem | Op::SetElemDrop
            if fast
                & if matches!(op, Op::SetElem) {
                    4096
                } else {
                    2048
                }
                != 0
                && elem_inlinable(layout) =>
        {
            emit_set_elem_inline(a, layout, pc_u32, unwind, matches!(op, Op::SetElem));
        }
        Op::GetElemLocal(slot)
            if fast & 1024 != 0 && get_elem_inlinable(layout) && slot as u32 * 8 + 8 < 4096 =>
        {
            emit_elem_local_inline(
                a,
                layout,
                slot as u32 * 8,
                pc as u32,
                unwind,
                ElemLocalKind::Get,
            )
        }
        Op::SetElemLocal(slot)
            if fast & 4096 != 0 && elem_inlinable(layout) && slot as u32 * 8 + 8 < 4096 =>
        {
            emit_elem_local_inline(
                a,
                layout,
                slot as u32 * 8,
                pc_u32,
                unwind,
                ElemLocalKind::SetKeep,
            );
        }
        Op::SetElemLocalDrop(slot)
            if fast & 2048 != 0
                && elem_inlinable(layout)
                && slot as u32 * 8 + 8 < 4096
                && (layout.entry_accessor != layout.entry_value + 8
                    || fast & (1024 | 8192 | 32768 | 262144) == (1024 | 8192 | 32768 | 262144)) =>
        {
            // Keep the baseline emitter's diagnostic-mask eligibility identical.
            emit_elem_local_inline(
                a,
                layout,
                slot as u32 * 8,
                pc_u32,
                unwind,
                ElemLocalKind::SetDrop,
            );
        }
        Op::SetProp(name, cache)
        | Op::SetPropDrop(name, cache)
        | Op::SetPropThisDrop(name, cache)
        | Op::SetPropLocalDrop(_, name, cache)
            if fast & 65536 != 0
                && set_prop_inlinable(layout)
                && !matches!(op, Op::SetPropLocalDrop(slot, ..) if *slot as u32 * 8 + 8 >= 4096) =>
        {
            let receiver = match *op {
                Op::SetPropThisDrop(..) => PropRecv::This,
                Op::SetPropLocalDrop(slot, ..) => PropRecv::Slot(slot as u32 * 8),
                _ => PropRecv::Stack,
            };
            emit_set_prop_inline(
                a,
                layout,
                chunk.jit_cache_ptr(cache),
                chunk.jit_name(name),
                pc_u32,
                unwind,
                receiver,
                matches!(op, Op::SetProp(..)),
            );
        }
        Op::UpdateProp(_, cache, kind) if fast & 65536 != 0 && set_prop_inlinable(layout) => {
            emit_update_prop_inline(a, layout, chunk.jit_cache_ptr(cache), kind, pc_u32, unwind);
        }
        Op::LoadName(_, cache) | Op::LoadNameForCall(_, cache)
            if fast & 8192 != 0 && load_name_inlinable(layout) =>
        {
            emit_load_name_inline(
                a,
                layout,
                chunk.jit_name_cache_ptr(cache),
                chunk.jit_name_number(cache),
                pc_u32,
                unwind,
                matches!(op, Op::LoadNameForCall(..)),
            );
        }
        Op::UpdateNameCached(_, cache, kind)
            if fast & 8192 != 0 && update_name_inlinable(layout) =>
        {
            emit_update_name_inline(
                a,
                layout,
                chunk.jit_name_cache_ptr(cache),
                kind,
                pc_u32,
                unwind,
            );
        }
        Op::StoreNameCached(_, cache) if fast & 8192 != 0 && update_name_inlinable(layout) => {
            emit_store_name_inline(a, layout, chunk.jit_name_cache_ptr(cache), pc_u32, unwind);
        }
        Op::LoadCap(name)
            if fast & 8192 != 0
                && load_name_inlinable(layout)
                && !chunk.jit_needs_activation_state() =>
        {
            emit_load_name_inline(
                a,
                layout,
                chunk.jit_cap_cache_ptr(name),
                None,
                pc_u32,
                unwind,
                false,
            );
        }
        Op::StoreCap(name)
            if fast & 8192 != 0
                && update_name_inlinable(layout)
                && !chunk.jit_needs_activation_state() =>
        {
            emit_store_name_inline(a, layout, chunk.jit_cap_cache_ptr(name), pc_u32, unwind);
        }
        Op::StoreLocal(slot) if fast & 16 != 0 && slot as u32 * 8 + 8 < 4096 => {
            emit_store_local(a, layout, slot as u32 * 8, &[pc_u32], unwind, false, true);
        }
        Op::UpdateLocal(slot, kind) if fast & 32 != 0 && slot as u32 * 8 + 8 < 4096 => {
            emit_update_local(a, slot, kind, pc_u32, unwind);
        }
        Op::Pop if fast & 64 != 0 && layout.valid && layout.rc_strong_off < 256 => {
            emit_exec_release_word(a, layout, 20, -8);
            a.sub_imm(20, 20, 8);
        }
        Op::Pop | Op::Dup if fast & 64 != 0 => {
            let slow = a.new_label();
            let done = a.new_label();
            emit_exec_word_load(a, 9, 20, -8);
            if matches!(op, Op::Pop) {
                emit_exec_drop_shared(a, layout, 9, 10, 11, slow);
                a.sub_imm(20, 20, 8);
            } else {
                emit_exec_clone(a, layout, 9, 10, 11, slow);
                emit_exec_word_store(a, 9, 20, 0);
                a.add_imm(20, 20, 8);
            }
            a.b(done);
            a.bind(slow);
            emit_exec(a, pc_u32, unwind);
            a.bind(done);
        }
        Op::Not if fast & 131072 != 0 && eq_inlinable(layout) => {
            emit_not_inline(a, layout, pc_u32, unwind);
        }
        Op::InstanceOf(cache) if instanceof_inlinable(layout, ilayout) => {
            emit_instanceof_inline(
                a,
                layout,
                ilayout,
                chunk.jit_cache_ptr(cache),
                pc_u32,
                unwind,
            );
        }
        Op::Call(..) | Op::CallWithThis(..) => emit_call_inline(
            a,
            chunk,
            layout,
            ilayout,
            pc,
            fast,
            array_intrinsics_on,
            function_call_intrinsic_on,
            unwind,
            direct_finish,
        ),
        Op::MakeRegExp(..) => emit_op_helper(a, H_MAKE_REGEXP, pc_u32, unwind),
        Op::MakeObject(..) => emit_op_helper(a, H_MAKE_OBJECT, pc_u32, unwind),
        Op::MakeArray(..) => emit_op_helper(a, H_MAKE_ARRAY, pc_u32, unwind),
        Op::New(argc, _) => {
            a.mov(0, 19);
            a.movz(1, pc_u32, 0);
            a.movk(1, argc as u32, 1);
            a.mov(2, 20);
            a.ldr_imm(16, 21, (H_NEW * 8) as u32);
            a.blr(16);
            a.mov(20, 0);
            a.cbnz(1, false, unwind);
        }
        Op::SetProp(..)
        | Op::SetPropDrop(..)
        | Op::SetPropThisDrop(..)
        | Op::SetPropLocalDrop(..) => emit_op_helper(a, H_SET_PROP, pc_u32, unwind),
        Op::GetProp(..) | Op::GetPropThis(..) | Op::GetPropLocal(..) | Op::GetMethod(..) => {
            emit_op_helper(a, H_GET_PROP, pc_u32, unwind);
        }
        Op::GetMethodElem => emit_op_helper(a, H_GET_METHOD_ELEM, pc_u32, unwind),
        Op::GetElem | Op::GetElemLocal(_) => emit_op_helper(a, H_GET_ELEM, pc_u32, unwind),
        Op::DestructureArr(_)
        | Op::TryArrayBinding(..)
        | Op::ArrayBindingStep(..)
        | Op::IterStepL(..) => {
            emit_op_helper(a, H_ITERATION, pc_u32, unwind);
        }
        _ => emit_exec(a, pc_u32, unwind),
    }
}

#[cfg(test)]
#[path = "jit_region_predicate_tests.rs"]
mod predicate_tests;

#[cfg(test)]
#[path = "jit_native_operations_tests.rs"]
mod native_operations_tests;

#[cfg(test)]
#[path = "jit_region_result_tests.rs"]
mod result_tests;

#[cfg(test)]
#[path = "jit_region_scalar_tests.rs"]
mod scalar_tests;

#[cfg(test)]
#[path = "jit_region_edge_tests.rs"]
mod edge_tests;

#[cfg(test)]
#[path = "jit_region_effect_tests.rs"]
mod effect_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn check_guarded(source: &str, required: &[usize]) {
        use crate::{bytecode::Tier, value::Value, Completion, Engine};
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            engine.interp.def_method(
                &engine.interp.global,
                "collectGuardedTest",
                0,
                |interp, _, _| {
                    interp.gc_collect();
                    Ok(Value::Undefined)
                },
            );
            let before = GUARDED_REGIONS.with(std::cell::Cell::get);
            match engine
                .eval(source, false)
                .expect("guarded region fixture parses")
            {
                Completion::Value(value) => assert_eq!(value, "true", "{tier:?}"),
                Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
            }
            if tier == Tier::Jit {
                let after = GUARDED_REGIONS.with(std::cell::Cell::get);
                for &event in required {
                    assert!(
                        after[event] > before[event],
                        "missing actual region event {event}: before={before:?}, after={after:?}"
                    );
                }
            }
            engine.interp.gc_collect();
        }
    }

    #[test]
    fn guarded_regions_plain_calls_and_acyclic_inline_temporaries() {
        check_guarded(
            r#"
            function inc(x){return x+1;}
            function different(x){return x+7;}
            function straight(f,x){var a=f(x);return (a*2)+3;}
            function looping(f,n){var s=0;for(var i=0;i<n;i++){s+=f(i);if(i===3)s+=1;}return s;}
            var good=true;
            for(var r=0;r<160;r++){
                good=straight(inc,5)===15&&good;
                good=looping(inc,9)===46&&good;
            }
            good=straight(different,5)===27&&looping(different,9)===100&&good;
            good;
        "#,
            &[0, 1, 2],
        );
    }

    #[test]
    fn guarded_regions_borrow_inherited_method_and_preserve_mutating_arguments() {
        check_guarded(
            r#"
            function original(x){return this.bias+x;}
            function replacement(x){return this.bias+x+100;}
            var proto={method:original},o=Object.create(proto);o.bias=2;
            function straight(o,x){var a=o.method(x);return a*2+1;}
            function loop(o,n){var s=0;for(var i=0;i<n;i++){s+=o.method(i);if(i===3)s++;}return s;}
            var good=true;
            for(var r=0;r<160;r++)good=straight(o,5)===15&&loop(o,9)===55&&good;
            proto.method=replacement;
            good=straight(o,5)===215&&loop(o,9)===955&&good;
            proto.method=original;
            var changes=0;
            function argument(o){changes++;proto.method=replacement;collectGuardedTest();return 5;}
            function withArgument(o){return o.method(argument(o));}
            for(var r=0;r<140;r++){
                proto.method=original;
                good=withArgument(o)===7&&good;
            }
            var gets=0;
            Object.defineProperty(proto,'method',{configurable:true,get(){gets++;return original;}});
            good=straight(o,5)===15&&gets===1&&changes===140&&good;
            var proxied=new Proxy(o,{get(t,k){return t[k];}});
            good=straight(proxied,5)===15&&gets===2&&good;
            var alternate={method:replacement};
            function swapPrototype(o){Object.setPrototypeOf(o,alternate);collectGuardedTest();return 5;}
            function withPrototype(o){return o.method(swapPrototype(o));}
            for(var r=0;r<140;r++){
                Object.setPrototypeOf(o,{method:original});
                good=withPrototype(o)===7&&good;
            }
            function throwingArgument(){changes++;throw 'argument';}
            function withThrow(o){return o.method(throwingArgument());}
            try{withThrow(o);good=false;}catch(e){good=e==='argument'&&changes===141&&good;}
            good;
        "#,
            &[0, 1, 2, 3],
        );
    }

    #[test]
    fn guarded_regions_preserve_closure_environment_throws_and_owner_replacement() {
        check_guarded(
            r#"
            function make(k){return function(x){return x+k;};}
            function call(f,x){var a=f(x);return a*2+1;}
            var one=make(1),seven=make(7),good=true;
            for(var r=0;r<160;r++)good=call(one,2)===7&&good;
            good=call(seven,2)===19&&good;
            var calls=0;
            function throws(x){calls++;throw x;}
            try{call(throws,'unique');good=false;}catch(e){good=e==='unique'&&calls===1&&good;}
            function method(value){return value.id+this.bias;}
            function use(o){return o.method(o.child,(o.child=null,collectGuardedTest()));}
            for(var r=0;r<150;r++)good=use({method:method,bias:4,child:{id:r}})===r+4&&good;
            function branch(o,x){var a=o.a; if(x){a=a+2;}else{a=a*3;}return a+o.b;}
            for(var r=0;r<120;r++)good=branch({a:4,b:5},true)===11&&branch({a:4,b:5},false)===17&&good;
            function localAlias(o){var a=o.child;return call(function(v){return v.id;},(o=null,collectGuardedTest(),a));}
            for(var r=0;r<120;r++)good=localAlias({child:{id:r}})===r*2+1&&good;
            function tdz(flag){if(flag){return missing+1;}let missing=2;return missing*2+1;}
            for(var r=0;r<120;r++)good=tdz(false)===5&&good;
            try{tdz(true);good=false;}catch(e){good=e instanceof ReferenceError&&good;}
            good;
        "#,
            // Captured closures keep their real execution context rather than requiring
            // an inline guard. Other fixture groups require an actual guard-miss event;
            // here require real region/guard-success execution plus exact context results.
            &[0, 1],
        );
    }

    fn check_borrowed_reads(source: &str, require_reuse: bool) {
        use crate::{bytecode::Tier, value::Value, Completion, Engine};
        for (tier, warmed) in [
            (Tier::Interp, false),
            (Tier::Bytecode, false),
            (Tier::Jit, false),
            (Tier::Jit, true),
        ] {
            let mut engine = Engine::new();
            engine.set_tier(if warmed { Tier::Bytecode } else { tier });
            engine.set_tier_threshold(0);
            engine.interp.def_method(
                &engine.interp.global,
                "collectOwnReadTest",
                0,
                |interp, _, _| {
                    interp.gc_collect();
                    Ok(Value::Undefined)
                },
            );
            // Cold all-tier runs above remain unchanged. The additional JIT run
            // warms the same function chunks up to the explicit fixture boundary:
            // feedback chooses borrowed-own lowering, live guards prove each use.
            engine.interp.def_method(
                &engine.interp.global,
                "beginOwnReadNativeTest",
                0,
                if warmed {
                    |interp, _, _| {
                        interp.tier = Tier::Jit;
                        Ok(Value::Undefined)
                    }
                } else {
                    |_, _, _| Ok(Value::Undefined)
                },
            );
            let before = BORROWED_READS.with(std::cell::Cell::get);
            match engine.eval(source, false).expect("property fixture parses") {
                Completion::Value(value) => assert_eq!(value, "true", "{tier:?}"),
                Completion::Throw { name, message } => panic!("{tier:?}: {name}: {message}"),
            }
            if warmed && require_reuse {
                let after = BORROWED_READS.with(std::cell::Cell::get);
                assert!(after.0 > before.0, "must execute borrowed property reads");
                assert!(after.1 > before.1, "must execute reused receiver guards");
            }
            engine.interp.gc_collect();
        }
    }

    #[test]
    fn borrowed_own_read_guards_survive_only_guarded_numeric_local_updates() {
        check_borrowed_reads(
            r#"
            function sum(o,n){var s=0;for(var i=0;i<n;i++){s+=o.a;s++;s+=o.b;}return s;}
            var o={a:2,b:3},good=sum(o,30)===180;
            beginOwnReadNativeTest();good=sum(o,30)===180&&good;
            // An intervening coercion is not a numeric local update. It must
            // leave the guarded region before mutating the other descriptor.
            var conversions=0;
            var changing={a:{valueOf(){conversions++;changing.b=9;return 4;}},b:2};
            good=sum(changing,30)===420&&conversions===30&&good;
            var gets=0,accessor={get a(){gets++;this.b=7;return 2;},b:3};
            good=sum(accessor,30)===300&&gets===30&&good;
            var string={a:'2',b:3};good=sum(string,2)===66&&good;
            var threw=false;try{sum({a:1n,b:3},1);}catch(e){threw=e instanceof TypeError;}
            good=threw&&sum(o,30)===180&&good;
            good;
        "#,
            true,
        );
    }

    #[test]
    fn borrowed_own_reads_reuse_guards_but_observe_live_values_and_descriptors() {
        check_borrowed_reads(
            r#"
            function sum(o,n){var s=0;for(var i=0;i<n;i++)s+=o.a+o.b;return s;}
            var o={a:2,b:3}, good=sum(o,30)===150;
            beginOwnReadNativeTest();good=sum(o,30)===150&&good;
            o.a=7;good=sum(o,30)===300&&good;
            var gets=0;
            Object.defineProperty(o,'a',{get(){gets++;return 11;},configurable:true});
            good=sum(o,30)===420&&gets===30&&good;
            delete o.a;o.a=5;good=sum(o,30)===240&&good;
            delete o.b;Object.setPrototypeOf(o,{b:17});good=sum(o,30)===660&&good;
            Object.defineProperty(o,'b',{value:19,writable:true});
            good=sum(o,30)===720&&good;
            function aliased(o,alias){var s=0;for(var i=0;i<30;i++){s+=o.a+o.b;alias.a++;}return s;}
            var a={a:1,b:2};good=aliased(a,a)===525&&good;
            function receiver(n){var s=0;for(var i=0;i<n;i++)s+=this.a+this.b;return s;}
            good=receiver.call({a:5,b:7},30)===360&&good;
            function numericKey(o){var s=0;for(var i=0;i<30;i++)s+=o['0']+o.a;return s;}
            good=numericKey({'0':4,a:3})===210&&good;
            var proxied=new Proxy({a:2,b:3},{get(t,k){gets++;return t[k];}});
            var before=gets;good=sum(proxied,30)===150&&gets-before===60&&good;
            var arr=[];arr.a=2;arr.b=3;good=sum(arr,30)===150&&good;
            var boxed=new Number(1);boxed.a=2;boxed.b=3;good=sum(boxed,30)===150&&good;
            var changes=0;
            var shifting={a:1,get b(){changes++;this.a++;return 3;}};
            good=sum(shifting,24)===372&&changes===24&&good;
            good;
        "#,
            true,
        );
    }

    #[test]
    fn borrowed_own_reads_materialize_before_alias_writes_receiver_replacement_and_gc() {
        check_borrowed_reads(
            r#"
            function same(a,b){return a===b;}
            function check(o,keep){
                var good=true,last;
                for(var i=0;i<24;i++){
                    o.keep=keep;
                    // Dup's temporary and the original receiver's last owner
                    // must both survive replacement and a reentrant collection.
                    last=o.keep;
                    good=same(o.keep,(o={keep:keep},collectOwnReadTest(),keep))&&good;
                    good=same(o.keep,(o.keep=null,collectOwnReadTest(),keep))&&good;
                    good=last===keep&&good;
                }
                return good;
            }
            var inputs=[{},Symbol('keep'),'owned value',true,null,undefined,123n];
            var good=true;
            good=check({keep:inputs[0]},inputs[0])&&good;
            good=numeric({a:2,b:3})===120&&good;
            beginOwnReadNativeTest();
            for(var k=0;k<inputs.length;k++)good=check({keep:inputs[k]},inputs[k])&&good;
            function onlyOwner(value,ignored){return value.id;}
            function droppedReceiver(){
                var total=0;
                for(var i=0;i<24;i++){
                    var o={keep:{id:i}};
                    total+=onlyOwner(o.keep,(o=null,collectOwnReadTest()));
                }
                return total;
            }
            good=droppedReceiver()===276&&good;
            function numeric(o){var s=0;for(var i=0;i<24;i++)s+=o.a+o.b;return s;}
            good=numeric({a:2,b:3})===120&&good;
            good;
        "#,
            true,
        );
    }

    #[test]
    fn borrowed_own_reads_preserve_sentinels_nan_signed_zero_and_exact_throw_resume() {
        check_borrowed_reads(
            r#"
            function numbers(o){var s=0;for(var i=0;i<24;i++)s+=o.a+o.b;return s;}
            var good=numbers({a:2,b:3})===120;
            beginOwnReadNativeTest();good=numbers({a:2,b:3})===120&&good;
            good=Number.isNaN(numbers({a:NaN,b:2}))&&good;
            function zeros(o){var good=true;for(var i=0;i<24;i++)good=Object.is(o.a,-0)&&good;return good;}
            good=zeros({a:-0})&&good;
            function big(o){var s=0n;for(var i=0;i<24;i++)s+=o.a+o.b;return s;}
            good=big({a:123n,b:456n})===13896n&&good;
            function lazy(f){var good=true;for(var i=0;i<24;i++)good=f.prototype.constructor===f&&good;return good;}
            good=lazy(function Fresh(){})&&good;
            var reads=0,calls=0;
            function effect(o){calls++;o.a=100;if(calls===13)throw 'stop';return 2;}
            function throwing(o){var s=0;for(var i=0;i<24;i++)s+=o.a+effect(o);return s;}
            var o={get a(){reads++;return 5;}};
            try{throwing(o);good=false;}catch(e){good=e==='stop'&&good;}
            good=reads===13&&calls===13&&good;
            // Coercion is also an effect; it can replace descriptors after a read.
            var target={a:{valueOf(){target.b=9;return 4;}},b:2};
            function coercion(o){var s=0;for(var i=0;i<24;i++)s+=(o.a+1)+o.b;return s;}
            good=coercion(target)===336&&good;
            good;
        "#,
            true,
        );
    }

    #[test]
    fn general_region_admits_nested_control_flow_and_checked_objects() {
        let statements = crate::parser::parse_script(
            "function f(n,o,callback){var s=0;for(var i=0;i<n;i++){if(i<5){s+=o.value;continue;}for(var j=0;j<3;j++){if(j===1)s+=callback(i);else s+=j;}if(s>1000)break;}return s;}", false).ok().unwrap();
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("function");
        };
        let chunk = crate::bytecode::compile(function).unwrap();
        let cfg = Cfg::build(&chunk).unwrap();
        assert!(cfg.loops().iter().any(|lp| {
            let head = cfg.blocks()[lp.header.0 as usize].start;
            Plan::build(&chunk, &cfg, head).is_some_and(|plan| plan.blocks.len() >= 5)
        }), "general lowering must cover the outer object/call loop, not just its numeric inner loop");
    }

    #[test]
    fn register_pressure_selects_a_useful_bounded_subset() {
        let statements = crate::parser::parse_script(
            "function f(n,o){var a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8,j=9,k=10,s=0;for(var i=0;i<n;i++){a++;b++;c++;d++;e++;f++;g++;h++;j++;k++;s+=o.value;}return a+b+c+d+e+f+g+h+j+k+s;}", false).ok().unwrap();
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("function")
        };
        let chunk = crate::bytecode::compile(function).unwrap();
        let cfg = Cfg::build(&chunk).unwrap();
        let plan = cfg
            .loops()
            .iter()
            .find_map(|lp| Plan::build(&chunk, &cfg, cfg.blocks()[lp.header.0 as usize].start))
            .expect("register pressure must retain a useful region");
        assert_eq!(plan.homes.len(), 8);
    }

    #[test]
    fn general_region_tagged_homes_execute_with_owned_gc_roots() {
        use crate::{bytecode::Tier, value::Value, Completion, Engine};
        let source = r#"
            var keep={}, spare={}, symbol=Symbol('retained');
            function collect(o){collectRegionTest();return o.value;}
            function execute(a,b,callback){
                var sum=0,last=a;
                for(var i=0;i<45;i++){
                    // A duplicated borrowed view must become independently owned
                    // before replacing its canonical source or entering a callback.
                    last=(a=b); b=last;
                    sum+=callback({value:i,retained:a});
                    if(!Object.is(last,b))throw 'lost alias';
                }
                return sum===990 && Object.is(a,b) && Object.is(last,b);
            }
            var inputs=[keep,spare,symbol,'retained',false,null,undefined,
                NaN,-0,Infinity,12345678901234567890n];
            var good=true;
            for(var r=0;r<4;r++)for(var k=0;k<inputs.length;k++)
                good=execute(keep,inputs[k],collect)&&good;
            good;
        "#;
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        engine.interp.def_method(
            &engine.interp.global,
            "collectRegionTest",
            0,
            |interp, _, _| {
                interp.gc_collect();
                Ok(Value::Undefined)
            },
        );
        let before = EXECUTED_REGIONS.with(std::cell::Cell::get);
        assert!(matches!(engine.eval(source, false), Ok(Completion::Value(v)) if v == "true"));
        assert!(
            EXECUTED_REGIONS.with(std::cell::Cell::get) > before,
            "typed guards must actually admit native region execution"
        );
        engine.interp.gc_collect();
    }
}
