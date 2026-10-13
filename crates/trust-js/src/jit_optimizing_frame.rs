//! Whole-function local-state publication analysis for the optimizing backend.
//!
//! A dirty local is an owning word in SSA, not yet published in its canonical frame location.
//! No call, allocation/GC poll, exception transfer, or other observer may see that stale memory.
//! Checked operations publish incoming dirty locals; only needed, possibly overwritten
//! register copies must be reloaded. Publication and private-word writes are distinct.
//! Canonical owners remain intact even when their register copies are not needed again.
//! Guarded fast paths may avoid the call; their outgoing dirty state therefore joins BOTH paths.
//! Exception/finalizer dispatch starts from canonical memory, not a normal CFG predecessor.
//!
//! ECMA-262 e28783d5: Execution Contexts, GetValue/PutValue, and WeakRef Liveness. This analysis
//! changes storage only; it never treats a heap word as a root across an unpublished safepoint.

use crate::bytecode::{Chunk, Op};
use crate::jit_ir::Cfg;
use std::collections::VecDeque;

use super::effects::{checked_writes, SlotWrites};

/// Must describe every native local write and every path that can bypass a checked call in
/// `Lowering::lower`. Unknown operations are full publication barriers. A guarded
/// slow call can clear dirtiness on that edge, but cannot clear it on the sibling fast edge.
/// This describes canonical publication, NOT private-word replacement by helpers.
#[derive(Clone, Copy)]
enum Effect {
    Preserve,
    Write(u16),
    Barrier,
}

fn effect(op: &Op) -> Effect {
    match op {
        Op::StoreLocal(slot) | Op::UpdateLocal(slot, _) => Effect::Write(*slot),
        // The native name-cache hit performs no call/publication. Carry all
        // incoming dirty owners across BOTH fast and checked successors.
        Op::LoadName(..) | Op::LoadNameForCall(..) | Op::LoadCap(_) if super::names::enabled() => {
            Effect::Preserve
        }
        Op::Const(_)
        | Op::Undef
        | Op::LoadLocal(_)
        | Op::Dup
        | Op::Pop
        | Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::BitAnd
        | Op::BitOr
        | Op::BitXor
        | Op::Shl
        | Op::Shr
        | Op::UShr
        | Op::Lt
        | Op::Le
        | Op::Gt
        | Op::Ge
        | Op::StrictEq
        | Op::StrictNotEq
        | Op::GetProp(..)
        | Op::GetPropThis(..)
        | Op::GetPropLocal(..)
        | Op::GetMethod(..)
        | Op::SetProp(..)
        | Op::SetPropDrop(..)
        | Op::SetPropThisDrop(..)
        | Op::SetPropLocalDrop(..)
        | Op::GetElem
        | Op::GetElemLocal(_)
        | Op::GetMethodElem
        | Op::SetElem
        | Op::SetElemDrop
        | Op::SetElemLocal(_)
        | Op::SetElemLocalDrop(_)
        | Op::Jump(_)
        | Op::JumpIfFalse(_)
        | Op::JumpIfFalsePeek(_)
        | Op::JumpIfTruePeek(_)
        | Op::JumpIfNotNullishPeek(_)
        | Op::InlineGuard(..) => Effect::Preserve,
        _ => Effect::Barrier,
    }
}

pub(super) struct FramePlan {
    pub(super) tracked: Vec<u16>,
    pub(super) dirty_at: Vec<Vec<u16>>,
    /// Conservative publication set at implicit function fall-through.
    pub(super) written: Vec<u16>,
    /// SSA words needed before an operation, including canonical publication.
    /// This is NOT JavaScript binding/GC liveness: unneeded register copies
    /// leave every canonical frame owner and environment binding untouched.
    pub(super) live_at: Vec<Vec<u16>>,
    /// Normal successors only. Exceptional/dynamic-completion destinations
    /// reload their own live_at set from the actual canonical frame.
    pub(super) live_after: Vec<Vec<u16>>,
}

fn write(mask: &mut [u64], slot: u16) {
    mask[slot as usize / 64] |= 1 << (slot as usize % 64);
}

fn members(mask: &[u64]) -> Vec<u16> {
    let mut result = Vec::new();
    for (word, &bits) in mask.iter().enumerate() {
        let mut bits = bits;
        while bits != 0 {
            result.push((word * 64 + bits.trailing_zeros() as usize) as u16);
            bits &= bits - 1;
        }
    }
    result
}

impl FramePlan {
    pub(super) fn build(chunk: &Chunk, cfg: &Cfg, local_effects: bool) -> Option<Self> {
        let ops = chunk.jit_ops();
        let (_, slots) = chunk.jit_frame();
        let words = slots.div_ceil(64);
        let blocks = cfg.blocks();
        // These are compiler-work limits, not JavaScript limits. The caller retains complete
        // baseline execution if a pathological function exceeds bounded analysis/IR space.
        if slots > u16::MAX as usize + 1
            || blocks.len().checked_mul(words)?.checked_mul(3)? > 1_048_576
        {
            return None;
        }
        let mut tracked = vec![0; words];
        let mut written = vec![0; words];
        for op in ops {
            if let Op::LoadLocal(slot)
            | Op::StoreLocal(slot)
            | Op::UpdateLocal(slot, _)
            | Op::GetPropLocal(slot, ..)
            | Op::SetPropLocalDrop(slot, ..)
            | Op::GetElemLocal(slot)
            | Op::SetElemLocal(slot)
            | Op::SetElemLocalDrop(slot) = op
            {
                if *slot as usize >= slots {
                    return None;
                }
                write(&mut tracked, *slot);
            }
            if let Effect::Write(slot) = effect(op) {
                write(&mut written, slot);
            }
        }
        let tracked = members(&tracked);
        if ops.len().checked_mul(tracked.len())? > 1_000_000 {
            return None;
        }
        let mut incoming = vec![vec![0; words]; blocks.len()];
        let mut outgoing = incoming.clone();
        let mut generated = incoming.clone();
        let mut kills = vec![false; blocks.len()];
        for (id, block) in blocks.iter().enumerate() {
            for op in &ops[block.start..block.end] {
                match effect(op) {
                    Effect::Write(slot) => write(&mut generated[id], slot),
                    Effect::Barrier => {
                        generated[id].fill(0);
                        kills[id] = true;
                    }
                    Effect::Preserve => {}
                }
            }
        }
        // Monotone may-dirty fixed point. Normal backedges and joins carry SSA values;
        // exceptional entries contribute the empty set because every checked call publishes.
        let mut queue: VecDeque<_> = (0..blocks.len()).collect();
        let mut queued = vec![true; blocks.len()];
        let mut work = 8_000_000usize;
        while let Some(id) = queue.pop_front() {
            queued[id] = false;
            work = work.checked_sub((blocks[id].predecessors.len() + 2).checked_mul(words + 1)?)?;
            for pred in &blocks[id].predecessors {
                for (dst, src) in incoming[id].iter_mut().zip(&outgoing[pred.0 as usize]) {
                    *dst |= src;
                }
            }
            let mut changed = false;
            for word in 0..words {
                let next = generated[id][word] | if kills[id] { 0 } else { incoming[id][word] };
                changed |= next != outgoing[id][word];
                outgoing[id][word] = next;
            }
            if changed {
                for next in &blocks[id].successors {
                    let next = next.0 as usize;
                    if !queued[next] {
                        queued[next] = true;
                        queue.push_back(next);
                    }
                }
            }
        }
        let mut dirty_at = vec![Vec::new(); ops.len()];
        let mut locations = 0usize;
        for (id, block) in blocks.iter().enumerate() {
            let mut dirty = incoming[id].clone();
            for pc in block.start..block.end {
                let entries = members(&dirty);
                locations = locations.checked_add(entries.len())?;
                if locations > 4_000_000 {
                    return None;
                }
                dirty_at[pc] = entries;
                match effect(&ops[pc]) {
                    Effect::Write(slot) => write(&mut dirty, slot),
                    Effect::Barrier => dirty.fill(0),
                    Effect::Preserve => {}
                }
            }
        }
        let written = members(&written);
        let live = reload_liveness(ops, &tracked, &dirty_at, &written, slots, local_effects)?;
        Some(Self {
            tracked,
            dirty_at,
            written,
            live_at: live.at,
            live_after: live.after,
        })
    }
}

struct ReloadLiveness {
    at: Vec<Vec<u16>>,
    after: Vec<Vec<u16>>,
}

/// Backward register-use fixed point. A concrete helper's private-word write set
/// kills those register copies; publication by itself does not. Unknown checked
/// operations still define all copies from canonical storage. The diagnostic
/// ablation retains the original all-clobber barrier analysis.
///
/// Every may-dirty local is a use at EVERY bytecode boundary, even if ordinary
/// lowering has no helper there. This protects loop polls, before-effect
/// deoptimization and conservative fast/slow joins from stale republishing.
/// StoreLocal reads its old owner before replacing it, so that slot is a use
/// even when the JavaScript value is otherwise dead. Completion landings are
/// independent entries, never ordinary predecessors of the throwing helper.
fn reload_liveness(
    ops: &[Op],
    tracked: &[u16],
    dirty_at: &[Vec<u16>],
    written: &[u16],
    slots: usize,
    local_effects: bool,
) -> Option<ReloadLiveness> {
    let words = tracked.len().div_ceil(64);
    if (ops.len() + 1).checked_mul(words)? > 1_048_576 {
        return None;
    }
    // Compact bits follow TRACKED slots, not the largest physical slot number.
    let mut compact = vec![usize::MAX; slots];
    for (index, &slot) in tracked.iter().enumerate() {
        compact[usize::from(slot)] = index;
    }
    let bit = |slot: u16| -> Option<usize> {
        compact
            .get(usize::from(slot))
            .copied()
            .filter(|&index| index != usize::MAX)
    };
    let mut uses = vec![vec![0u64; words]; ops.len() + 1];
    for (pc, dirty) in dirty_at.iter().enumerate() {
        for &slot in dirty {
            let index = bit(slot)?;
            uses[pc][index / 64] |= 1 << (index % 64);
        }
        if let Op::LoadLocal(slot)
        | Op::StoreLocal(slot)
        | Op::UpdateLocal(slot, _)
        | Op::GetPropLocal(slot, ..)
        | Op::SetPropLocalDrop(slot, ..)
        | Op::GetElemLocal(slot)
        | Op::SetElemLocal(slot)
        | Op::SetElemLocalDrop(slot) = ops[pc]
        {
            let index = bit(slot)?;
            uses[pc][index / 64] |= 1 << (index % 64);
        }
    }
    for &slot in written {
        let index = bit(slot)?;
        uses[ops.len()][index / 64] |= 1 << (index % 64);
    }

    // Precompute finite write masks instead of repeatedly scanning tracked slots
    // in loop fixed-point iterations. FramePlan's ops*tracked/work bounds cover
    // this additional matrix and construction work as well.
    let mut definitions = vec![vec![0u64; words]; ops.len()];
    for (pc, op) in ops.iter().enumerate() {
        let writes = if local_effects {
            checked_writes(op)
        } else {
            match effect(op) {
                Effect::Barrier => SlotWrites::All,
                Effect::Preserve => SlotWrites::None,
                Effect::Write(slot) => SlotWrites::One(slot),
            }
        };
        for (index, &slot) in tracked.iter().enumerate() {
            if writes.contains(slot) {
                definitions[pc][index / 64] |= 1 << (index % 64);
            }
        }
    }

    let mut predecessors = vec![Vec::new(); ops.len() + 1];
    for (pc, op) in ops.iter().enumerate() {
        let (successors, count) = super::stack::successors(op, pc);
        for &target in &successors[..count] {
            predecessors.get_mut(target)?.push(pc);
        }
    }
    let mut at = uses.clone();
    let mut after = vec![vec![0u64; words]; ops.len() + 1];
    let mut queue: VecDeque<_> = (0..ops.len()).rev().collect();
    let mut queued = vec![true; ops.len()];
    let mut budget = 8_000_000usize;
    while let Some(pc) = queue.pop_front() {
        queued[pc] = false;
        let (successors, count) = super::stack::successors(&ops[pc], pc);
        budget = budget.checked_sub((count + 3).checked_mul(words.max(1))?)?;
        let mut changed = false;
        for word in 0..words {
            let outgoing = successors[..count]
                .iter()
                .fold(0, |mask, &target| mask | at[target][word]);
            after[pc][word] = outgoing;
            let kept = outgoing & !definitions[pc][word];
            let incoming = uses[pc][word] | kept;
            changed |= incoming != at[pc][word];
            at[pc][word] = incoming;
        }
        if changed {
            for &previous in &predecessors[pc] {
                if !queued[previous] {
                    queued[previous] = true;
                    queue.push_back(previous);
                }
            }
        }
    }
    let expand = |sets: Vec<Vec<u64>>| -> Vec<Vec<u16>> {
        sets.into_iter()
            .map(|mask| {
                members(&mask)
                    .into_iter()
                    .map(|index| tracked[usize::from(index)])
                    .collect()
            })
            .collect()
    };
    Some(ReloadLiveness {
        at: expand(at),
        after: expand(after),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::{self, UpdKind};

    fn analyze(ops: &[Op], tracked: &[u16]) -> ReloadLiveness {
        let slots = tracked.last().map_or(0, |slot| usize::from(*slot) + 1);
        reload_liveness(
            ops,
            tracked,
            &vec![Vec::new(); ops.len()],
            &[],
            slots,
            false,
        )
        .unwrap()
    }

    fn analyze_effects(ops: &[Op], tracked: &[u16]) -> ReloadLiveness {
        let slots = tracked.last().map_or(0, |slot| usize::from(*slot) + 1);
        reload_liveness(ops, tracked, &vec![Vec::new(); ops.len()], &[], slots, true).unwrap()
    }

    #[test]
    fn private_slot_liveness_flows_through_reentrant_calls_and_loop_joins() {
        let ops = [
            Op::Call(0, 0),
            Op::JumpIfFalse(6),
            Op::LoadLocal(0),
            Op::CallWithThis(0, 1),
            Op::LoadLocal(1),
            Op::Jump(1),
            Op::LoadLocal(2),
            Op::Return,
        ];
        let ReloadLiveness { at, after } = analyze_effects(&ops, &[0, 1, 2, 3]);
        assert_eq!(at[0], [0, 1, 2], "calls must not kill preserved SSA uses");
        assert_eq!(after[3], [0, 1, 2], "loop carries the next iteration too");
        assert!(at.iter().all(|live| !live.contains(&3)));
    }

    #[test]
    fn private_slot_liveness_kills_only_written_words_but_keeps_unknown_bridges() {
        let ops = [
            Op::Call(0, 0),
            Op::ResetSlots(0, 1),
            Op::LoadLocal(0),
            Op::LoadLocal(1),
            Op::Return,
        ];
        let ReloadLiveness { at, after } = analyze_effects(&ops, &[0, 1]);
        assert_eq!(at[0], [1]);
        assert_eq!(at[1], [1]);
        assert_eq!(after[1], [0, 1]);
        let ops = [
            Op::Call(0, 0),
            Op::EvalExpr(0),
            Op::LoadLocal(0),
            Op::Return,
        ];
        let ReloadLiveness { at, after } = analyze_effects(&ops, &[0]);
        assert!(at[0].is_empty());
        assert!(at[1].is_empty());
        assert_eq!(after[1], [0], "bridge writeback must be reloaded");
    }

    #[test]
    fn private_slot_liveness_preserves_publications_after_mixed_helper_effects() {
        let ops = [Op::Call(0, 0), Op::Tdz(0), Op::Call(0, 1), Op::Return];
        let dirty = vec![vec![], vec![1], vec![], vec![1]];
        let ReloadLiveness { at, after } =
            reload_liveness(&ops, &[0, 1], &dirty, &[], 2, true).unwrap();
        assert_eq!(at[0], [1]);
        assert_eq!(after[0], [1]);
        assert_eq!(at[2], [1]);
        assert_eq!(after[2], [1], "later publication is a register use");
    }

    #[test]
    fn reload_liveness_does_not_carry_register_copies_across_checked_barriers() {
        let ops = [
            Op::LoadLocal(0),
            Op::Call(0, 0),
            Op::Call(0, 1),
            Op::LoadLocal(1),
            Op::Return,
        ];
        let ReloadLiveness { at, after } = analyze(&ops, &[0, 1, 2]);
        assert_eq!(at[0], [0]);
        assert!(at[1].is_empty());
        assert!(after[1].is_empty());
        assert_eq!(after[2], [1]);
        assert_eq!(at[3], [1]);
        assert!(at.iter().all(|live| !live.contains(&2)));
    }

    #[test]
    fn reload_liveness_counts_overwritten_owners_and_guarded_receiver_reads() {
        for op in [Op::StoreLocal(0), Op::UpdateLocal(0, UpdKind::PostInc)] {
            let ops = [Op::Call(0, 0), op, Op::Return];
            let ReloadLiveness { at, after } = analyze(&ops, &[0, 1]);
            assert_eq!(after[0], [0], "replacing a value must read its old owner");
            assert_eq!(at[1], [0]);
        }
        let ops = [
            Op::Call(0, 0),
            Op::GetPropLocal(0, 0, 0),
            Op::LoadLocal(1),
            Op::Return,
        ];
        let ReloadLiveness { at, after } = analyze(&ops, &[0, 1, 2]);
        assert_eq!(at[1], [0, 1], "the guarded fast edge preserves locals");
        assert_eq!(after[0], [0, 1]);
        assert_eq!(after[1], [1]);
    }

    #[test]
    fn reload_liveness_includes_publication_at_every_boundary_and_fallthrough() {
        let ops = [Op::Call(0, 0), Op::Undef, Op::Add, Op::Return];
        let dirty = vec![vec![], vec![], vec![0], vec![0]];
        let ReloadLiveness { at, after } =
            reload_liveness(&ops, &[0, 1], &dirty, &[0], 2, false).unwrap();
        assert_eq!(after[0], [0]);
        assert_eq!(at[1], [0]);
        assert_eq!(
            at[2],
            [0],
            "poll/deopt publication is a use without a JS read"
        );
        assert_eq!(
            at[ops.len()],
            [0],
            "implicit return publishes all may-writes"
        );
        assert!(
            after[3].is_empty(),
            "explicit return has no ordinary successor"
        );
    }

    #[test]
    fn reload_liveness_merges_branches_and_reaches_a_loop_fixed_point() {
        let ops = [
            Op::Call(0, 0),
            Op::JumpIfFalse(5),
            Op::LoadLocal(0),
            Op::LoadLocal(1),
            Op::Jump(1),
            Op::LoadLocal(2),
            Op::Return,
        ];
        let ReloadLiveness { at, after } = analyze(&ops, &[0, 1, 2, 3]);
        assert!(at[0].is_empty());
        assert_eq!(after[0], [0, 1, 2]);
        assert_eq!(at[1], [0, 1, 2]);
        assert_eq!(
            at[4],
            [0, 1, 2],
            "backedge needs another fixed-point iteration"
        );
        assert_eq!(at[5], [2]);
    }

    #[test]
    fn reload_liveness_keeps_completion_entries_separate_and_high_slots_compact() {
        let ops = [
            Op::Call(0, 0),
            Op::AbruptJump(3, 0),
            Op::Throw,
            Op::LoadLocal(u16::MAX),
            Op::Return,
        ];
        let ReloadLiveness { at, after } = analyze(&ops, &[7, u16::MAX]);
        assert!(after[0].is_empty());
        assert!(
            after[1].is_empty(),
            "finally dispatch supplies a new canonical entry"
        );
        assert!(after[2].is_empty());
        assert_eq!(at[3], [u16::MAX]);
        assert_eq!(at.len(), ops.len() + 1);
    }

    #[test]
    fn reload_liveness_rejects_invalid_normal_edges_instead_of_inventing_state() {
        assert!(reload_liveness(&[Op::Jump(3)], &[], &[vec![]], &[], 0, false).is_none());
    }

    #[test]
    fn compiled_frame_reload_sets_cover_owners_without_reloading_every_binding() {
        let statements = crate::parser::parse_script(
            r#"
            function subject() {
                var first={x:1}, second={x:2}, third={x:3};
                var n=first.x+second.x+third.x;
                tick(); tick(); tick();
                for(var i=0;i<20;i++) {
                    try { n+=i; tick(); if(i===4)continue; }
                    finally { n+=1; }
                }
                return n;
            }
            "#,
            false,
        )
        .ok()
        .unwrap();
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("function fixture")
        };
        let chunk = bytecode::compile(function).unwrap();
        let cfg = Cfg::build(&chunk).unwrap();
        let plan = FramePlan::build(&chunk, &cfg, false).unwrap();
        assert!(plan.tracked.len() >= 5);
        let mut smaller_calls = 0;
        for (pc, op) in chunk.jit_ops().iter().enumerate() {
            for slot in &plan.dirty_at[pc] {
                assert!(plan.live_at[pc].contains(slot), "publication at PC {pc}");
            }
            if let Op::StoreLocal(slot) | Op::UpdateLocal(slot, _) = op {
                assert!(plan.live_at[pc].contains(slot), "old owner at PC {pc}");
            }
            // Global identifier calls retain Reference/this semantics and use
            // CallWithThis even when the receiver ultimately is undefined.
            if matches!(op, Op::Call(..) | Op::CallWithThis(..))
                && plan.live_after[pc].len() < plan.tracked.len()
            {
                smaller_calls += 1;
            }
        }
        assert!(
            smaller_calls >= 3,
            "real bytecode must have smaller post-call reload sets"
        );
        assert!(!cfg.handler_roots().is_empty());
        for root in cfg.handler_roots() {
            let pc = cfg.blocks()[root.target.0 as usize].start;
            assert!(
                !plan.live_at[pc].is_empty(),
                "finally needs canonical locals"
            );
        }
    }
}
