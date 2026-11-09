//! The frame slots live at each pc (P3 stage 2 step 6; the plan's "Step 6, untagged phis", its
//! Liveness): a backward dataflow of the slots each operation may read before any write, over
//! the chunk's Cfg, as Maglev's bytecode liveness and the DFG's prune their phis and Ion's
//! EliminatePhis removes unobservable ones (target/p3/stage2/phis-input.txt 5.1).
//!
//! The builder makes a loop header's phi only for a slot live at the header and written in the
//! loop, merges no dead slot at a join, flushes no dead slot before an effect, and marks a slot
//! dead at a frame state's pc [`SlotSource::Dead`](super::ir::SlotSource): its exit leaves the
//! frame's word, which the VM writes before any read (`opt::exit`). The verifier checks the Dead
//! slots against a liveness of its own in tests and debug builds.
//!
//! - An operation reads and writes the slots it names (LoadLocal, StoreLocal, UpdateLocal, Tdz,
//!   ResetSlots, the fused local accesses, the for-of and destructuring steps' hidden slots, the
//!   elided `arguments` and rest reads). An operation that reads no slot is listed as such; any
//!   other counts as reading every slot (a template that may read any slot, as EvalExpr's and
//!   AssignTarget's projections of the frame do), and none counts as writing one it does not
//!   name, which only keeps more slots live.
//! - A try region's handlers keep the slots they read: at every pc where a handler record is
//!   active (its push earlier on the path, its pop not yet), the slots live where each of its
//!   targets starts are live too (the catch pad; a `for…of` region's close pads). Handler records
//!   change only at the last operation of a Cfg block (`jit_ir::semantic_block_starts`), so each
//!   block has one stack of them, propagated from the entry and from each push to its targets.
//!   Where two paths bring different stacks to a block (which the builder refuses) every slot is
//!   live everywhere.
//! - Callees cannot read a caller's slots: the bindings closures, direct eval and mapped
//!   `arguments` observe live in Environment Records (`jit_regions::FrameEffect`'s contract),
//!   and the tier refuses the chunks whose entry is prepared for them.

use crate::bytecode::Op;
use crate::jit_ir::{BlockId, Cfg};

/// The slots live before each operation of a chunk (see the module's doc).
pub(crate) struct SlotLiveness {
    /// u64 words per pc.
    words: usize,
    /// The live-in bits of each pc, `words` per pc.
    bits: Vec<u64>,
}

impl SlotLiveness {
    /// The liveness of `n_slots` frame slots over `ops`, whose Cfg `cfg` is.
    pub(crate) fn compute(ops: &[Op], cfg: &Cfg, n_slots: usize) -> SlotLiveness {
        let words = n_slots.div_ceil(64).max(1);
        let blocks = cfg.blocks();
        let index = |id: BlockId| id.0 as usize;
        let all: Vec<u64> = (0..words)
            .map(|word| {
                let bits = n_slots.saturating_sub(64 * word).min(64);
                if bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                }
            })
            .collect();
        let everything = |ops: &[Op]| SlotLiveness {
            words,
            bits: all.iter().copied().cycle().take(words * ops.len()).collect(),
        };
        // Each block's handler records at its start (push pcs, innermost last), from the entry
        // and from each push to its targets; None for a block nothing reaches.
        let mut stacks: Vec<Option<Vec<u32>>> = vec![None; blocks.len()];
        let Some(entry) = cfg.block_at(0) else {
            return everything(ops);
        };
        stacks[index(entry)] = Some(Vec::new());
        let mut work = vec![entry];
        let mut consistent = true;
        // Whether the block `to` starts with `stack`, which it is given if nothing reached it.
        let reach = |stacks: &mut Vec<Option<Vec<u32>>>,
                     work: &mut Vec<BlockId>,
                     to: BlockId,
                     stack: &[u32]| match &stacks[index(to)] {
            Some(known) => known.as_slice() == stack,
            None => {
                stacks[index(to)] = Some(stack.to_vec());
                work.push(to);
                true
            }
        };
        while let Some(id) = work.pop() {
            let block = &blocks[index(id)];
            let mut stack = stacks[index(id)].clone().expect("reached");
            let last = block.end - 1;
            match ops[last] {
                Op::PushHandler(_) | Op::PushIterator(..) | Op::PushFinally(..) => {
                    for target in handler_targets(&ops[last]) {
                        if let Some(to) = cfg.block_at(target) {
                            consistent &= reach(&mut stacks, &mut work, to, &stack);
                        }
                    }
                    stack.push(last as u32);
                }
                Op::PopHandler => consistent &= stack.pop().is_some(),
                _ => {}
            }
            for &succ in &block.successors {
                consistent &= reach(&mut stacks, &mut work, succ, &stack);
            }
        }
        if !consistent {
            return everything(ops);
        }
        // Each block's live-in slots, to a fixed point.
        let mut live_in = vec![0u64; words * blocks.len()];
        let mut live = vec![0u64; words];
        let mut changed = true;
        while changed {
            changed = false;
            for b in (0..blocks.len()).rev() {
                let Some(stack) = &stacks[b] else {
                    continue;
                };
                let block = &blocks[b];
                live.fill(0);
                for &succ in &block.successors {
                    union(&mut live, &live_in[index(succ) * words..][..words]);
                }
                let handlers = handler_live(ops, cfg, stack, &live_in, words);
                for pc in (block.start..block.end).rev() {
                    transfer(&ops[pc], &mut live, &all);
                    union(&mut live, &handlers);
                }
                let known = &mut live_in[b * words..][..words];
                if known != live.as_slice() {
                    known.copy_from_slice(&live);
                    changed = true;
                }
            }
        }
        // Each pc's live-in slots.
        let mut bits = all.iter().copied().cycle().take(words * ops.len()).collect::<Vec<_>>();
        for (b, block) in blocks.iter().enumerate() {
            let Some(stack) = &stacks[b] else {
                continue;
            };
            live.fill(0);
            for &succ in &block.successors {
                union(&mut live, &live_in[index(succ) * words..][..words]);
            }
            let handlers = handler_live(ops, cfg, stack, &live_in, words);
            for pc in (block.start..block.end).rev() {
                transfer(&ops[pc], &mut live, &all);
                union(&mut live, &handlers);
                bits[pc * words..][..words].copy_from_slice(&live);
            }
        }
        SlotLiveness { words, bits }
    }

    /// Whether `slot` is live before the operation at `pc`: some path from there reads it
    /// before any write (a pc nothing reaches keeps every slot).
    pub(crate) fn live(&self, pc: usize, slot: usize) -> bool {
        self.bits
            .get(pc * self.words + slot / 64)
            .is_none_or(|word| word & (1 << (slot % 64)) != 0)
    }
}

/// The slots live where the targets of the handler records `stack` start.
fn handler_live(ops: &[Op], cfg: &Cfg, stack: &[u32], live_in: &[u64], words: usize) -> Vec<u64> {
    let mut live = vec![0u64; words];
    for &push in stack {
        for target in handler_targets(&ops[push as usize]) {
            if let Some(to) = cfg.block_at(target) {
                union(&mut live, &live_in[to.0 as usize * words..][..words]);
            }
        }
    }
    live
}

/// The pcs a handler record's push sends its completions to (`jit_ir::handler_targets`'s).
fn handler_targets(op: &Op) -> Vec<usize> {
    match *op {
        Op::PushHandler(t) => vec![t as usize],
        Op::PushFinally(t, r, b, s, j) => [t, r, b, s, j].map(|pc| pc as usize).to_vec(),
        Op::PushIterator(t, r, b, s) => [t, r, b, s].map(|pc| pc as usize).to_vec(),
        _ => Vec::new(),
    }
}

fn union(into: &mut [u64], from: &[u64]) {
    for (a, b) in into.iter_mut().zip(from) {
        *a |= *b;
    }
}

fn set(live: &mut [u64], slot: u16) {
    if let Some(word) = live.get_mut(usize::from(slot) / 64) {
        *word |= 1 << (slot % 64);
    }
}

fn clear(live: &mut [u64], slot: u16) {
    if let Some(word) = live.get_mut(usize::from(slot) / 64) {
        *word &= !(1 << (slot % 64));
    }
}

/// The slots live before `op` from those live after it (`all`: every slot).
fn transfer(op: &Op, live: &mut [u64], all: &[u64]) {
    match *op {
        // Writes.
        Op::StoreLocal(slot) | Op::Tdz(slot) => clear(live, slot),
        Op::ResetSlots(start, count) => {
            for slot in start..start.saturating_add(count) {
                clear(live, slot);
            }
        }
        // Reads (an update's write follows its read, so the slot is live before it).
        Op::LoadLocal(slot)
        | Op::UpdateLocal(slot, _)
        | Op::GetPropLocal(slot, ..)
        | Op::SetPropLocalDrop(slot, ..)
        | Op::GetElemLocal(slot)
        | Op::SetElemLocal(slot)
        | Op::SetElemLocalDrop(slot)
        | Op::ToPropKeyLocal(slot)
        | Op::StoreConstLocal(slot, _)
        | Op::IterCloseL(slot)
        | Op::IterAbortL(slot)
        | Op::IterNextBuiltin(slot)
        | Op::ArgElem(slot)
        | Op::RestElem(slot)
        | Op::ArgLength(slot, ..)
        | Op::RestLength(slot, ..) => set(live, slot),
        Op::IterStepL(a, b) | Op::IterCloseIfNotDoneL(a, b) | Op::IterAbortIfNotDoneL(a, b) => {
            set(live, a);
            set(live, b);
        }
        Op::ForInStepL(a, b, c)
        | Op::DestructureStepL(a, b, c)
        | Op::DestructureRestL(a, b, c)
        | Op::TryArrayBinding(a, b, c)
        | Op::ArrayBindingStep(a, b, c) => {
            set(live, a);
            set(live, b);
            set(live, c);
        }
        // Operations that read no slot: their operands are the operand stack's, `this`, the
        // environment's or the heap's.
        Op::Const(_)
        | Op::Undef
        | Op::Dup
        | Op::Dup2
        | Op::Pop
        | Op::LoadCap(_)
        | Op::StoreCap(_)
        | Op::StoreCapInit(_)
        | Op::UpdateCap(..)
        | Op::UpdateNameCached(..)
        | Op::MakeClosure(..)
        | Op::LoadName(..)
        | Op::LoadNameForCall(..)
        | Op::LoadNameIn(..)
        | Op::LoadNameForCallIn(..)
        | Op::StoreName(_)
        | Op::StoreGlobalName(_)
        | Op::StoreNameCached(..)
        | Op::StoreConstCap(_)
        | Op::UpdateConst(..)
        | Op::LoadThis
        | Op::GetProp(..)
        | Op::GetPropThis(..)
        | Op::GetMethod(..)
        | Op::GetMethodElem
        | Op::SetProp(..)
        | Op::SetPropDrop(..)
        | Op::SetPropThisDrop(..)
        | Op::UpdateProp(..)
        | Op::AppendProp(..)
        | Op::GetElem
        | Op::SetElem
        | Op::SetElemDrop
        | Op::UpdateElem(_)
        | Op::ToPropKey
        | Op::ToStr
        | Op::GetIter
        | Op::ForInKeys
        | Op::DestructureGuard
        | Op::DestructureArr(_)
        | Op::ObjectRest(_)
        | Op::DeleteProp(..)
        | Op::DeleteElem(_)
        | Op::DeleteName(_)
        | Op::Add
        | Op::Sub
        | Op::Mul
        | Op::Div
        | Op::Mod
        | Op::BitAnd
        | Op::BitOr
        | Op::BitXor
        | Op::Shl
        | Op::Shr
        | Op::UShr
        | Op::Lt
        | Op::Gt
        | Op::Le
        | Op::Ge
        | Op::EqEq
        | Op::NotEq
        | Op::StrictEq
        | Op::StrictNotEq
        | Op::InstanceOf(_)
        | Op::In
        | Op::GenBin(_)
        | Op::Neg
        | Op::Plus
        | Op::Not
        | Op::BitNot
        | Op::Typeof
        | Op::TypeofIs(..)
        | Op::TypeofName(_)
        | Op::Void
        | Op::Jump(_)
        | Op::JumpIfFalse(_)
        | Op::JumpIfFalsePeek(_)
        | Op::JumpIfTruePeek(_)
        | Op::JumpIfNotNullishPeek(_)
        | Op::Call(..)
        | Op::CallWithThis(..)
        | Op::CallSpread(..)
        | Op::CallSpreadThis(..)
        | Op::CallArgsArray
        | Op::CallArgsArrayThis
        | Op::TailDeep
        | Op::TailCall(..)
        | Op::InlineGuard(..)
        | Op::New(..)
        | Op::NewArgsArray
        | Op::MakeRegExp(..)
        | Op::MakeArray(_)
        | Op::MakeArrayLiteral(_)
        | Op::NewArray
        | Op::ArrayPush
        | Op::ArrayHole
        | Op::ArraySpread
        | Op::MakeObject(..)
        | Op::NewObject
        | Op::ObjectData(_)
        | Op::ObjectSpread
        | Op::ObjectProto
        | Op::ObjectMethod(..)
        | Op::ObjectDataName(..)
        | Op::ObjectMethodName(..)
        | Op::TemplateObject(_)
        | Op::PushLex(_)
        | Op::PushCatchLex(_)
        | Op::CloneLex(_)
        | Op::InitLex(_)
        | Op::PopEnv
        | Op::PushHandler(_)
        | Op::PushIterator(..)
        | Op::PopHandler
        | Op::RequireIterResult
        | Op::Throw
        | Op::Return
        | Op::ReturnBare
        | Op::ReturnUndef => {}
        // Any other operation may read any slot.
        _ => live.copy_from_slice(all),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::UpdKind;

    /// The stack effects of the operations these tests use.
    fn effect(op: &Op) -> Option<(usize, usize)> {
        Some(match op {
            Op::Const(_) | Op::Undef | Op::LoadLocal(_) | Op::EvalExpr(_) => (0, 1),
            Op::StoreLocal(_) | Op::Pop | Op::Return => (1, 0),
            Op::Lt => (2, 1),
            Op::Call(argc, _) => (usize::from(*argc) + 1, 1),
            Op::Jump(_) | Op::PushHandler(_) | Op::PopHandler | Op::ReturnUndef => (0, 0),
            Op::UpdateLocal(..) => (0, 0),
            Op::JumpIfFalse(_) => (1, 0),
            _ => return None,
        })
    }

    fn liveness(ops: &[Op], n_slots: usize) -> SlotLiveness {
        let cfg = Cfg::test_build_with(ops, |pc| effect(&ops[pc])).expect("a Cfg");
        SlotLiveness::compute(ops, &cfg, n_slots)
    }

    fn live_at(liveness: &SlotLiveness, pc: usize, n_slots: usize) -> Vec<usize> {
        (0..n_slots)
            .filter(|&slot| liveness.live(pc, slot))
            .collect()
    }

    #[test]
    fn a_slot_written_before_any_read_is_dead_at_the_loop_header() {
        // i = 0; header (2): i < 10 or out; body: t = i; i++; read t; back. out: return.
        let ops = [
            Op::Const(0),
            Op::StoreLocal(0),
            Op::LoadLocal(0),
            Op::Const(1),
            Op::Lt,
            Op::JumpIfFalse(12),
            Op::LoadLocal(0),
            Op::StoreLocal(1),
            Op::UpdateLocal(0, UpdKind::IncDiscard),
            Op::LoadLocal(1),
            Op::Pop,
            Op::Jump(2),
            Op::ReturnUndef,
        ];
        let live = liveness(&ops, 2);
        // At the header i is live and t is not: the body writes t before it reads it.
        assert_eq!(live_at(&live, 2, 2), [0]);
        // t is live from its store to its read.
        assert_eq!(live_at(&live, 8, 2), [0, 1]);
        assert_eq!(live_at(&live, 10, 2), [0]);
        // Nothing is live before i's first store.
        assert_eq!(live_at(&live, 0, 2), Vec::<usize>::new());
        assert_eq!(live_at(&live, 12, 2), Vec::<usize>::new());
    }

    #[test]
    fn a_try_regions_handler_keeps_the_slots_its_catch_reads() {
        // x = 0; try { y = 1; call } catch { return y } return x.
        let ops = [
            Op::Const(0),
            Op::StoreLocal(0),
            Op::PushHandler(10),
            Op::Const(1),
            Op::StoreLocal(1),
            Op::Undef,
            Op::Call(0, 0),
            Op::Pop,
            Op::PopHandler,
            Op::Jump(13),
            Op::Pop,
            Op::LoadLocal(1),
            Op::Return,
            Op::LoadLocal(0),
            Op::Return,
        ];
        let live = liveness(&ops, 2);
        // At the call, which may throw, y is live for the catch and x for the normal path.
        assert_eq!(live_at(&live, 6, 2), [0, 1]);
        // Before its store in the region y is live too: the handler may run from there.
        assert_eq!(live_at(&live, 3, 2), [0, 1]);
        // After the region only x is.
        assert_eq!(live_at(&live, 9, 2), [0]);
        assert_eq!(live_at(&live, 13, 2), [0]);
        // The pad reads y.
        assert_eq!(live_at(&live, 10, 2), [1]);
    }

    #[test]
    fn an_operation_that_may_read_any_slot_keeps_every_slot_live() {
        let ops = [
            Op::Const(0),
            Op::StoreLocal(0),
            Op::EvalExpr(0),
            Op::Pop,
            Op::ReturnUndef,
        ];
        let live = liveness(&ops, 3);
        assert_eq!(live_at(&live, 2, 3), [0, 1, 2]);
        // Slot 0 is written first, the others are live from the entry.
        assert_eq!(live_at(&live, 0, 3), [1, 2]);
        assert_eq!(live_at(&live, 3, 3), Vec::<usize>::new());
    }
}
