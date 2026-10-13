//! Proven-live operand suffixes for whole-function SSA.
//!
//! The shared completion router truncates to min(saved depth, remaining depth). Therefore a
//! handler's CFG depth is only a logical upper bound: reloading its whole prefix would revive
//! moved owners. A state tracks only the suffix known to exist above that opaque prefix.
//! Labels are addressed relative to the ACTUAL returned sp, never the original stack base.
//!
//! ECMA-262 e28783d5: Execution Contexts, TryStatement Evaluation, GetValue/PutValue, WeakRef
//! Liveness. This is storage analysis, not permission to omit retains or roots at safepoints.

use crate::bytecode::{self, Chunk, Op};
use crate::jit_ir::Cfg;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct StackState {
    pub(super) depth: usize,
    pub(super) floor: usize,
}

impl StackState {
    pub(super) fn known(self) -> usize {
        self.depth - self.floor
    }

    fn after(self, pops: usize, pushes: usize) -> Option<Self> {
        let kept = self.depth.checked_sub(pops)?;
        Some(Self {
            depth: kept.checked_add(pushes)?,
            floor: self.floor.min(kept),
        })
    }
}

pub(super) struct StackPlan {
    pub(super) at: Vec<StackState>,
    pub(super) after: Vec<StackState>,
    /// An edge with a less precise successor must first publish the owners it forgets.
    /// A branch conservatively publishes the union required by either successor.
    pub(super) forget_to: Vec<usize>,
    pub(super) reachable: Vec<bool>,
    pub(super) capacity: usize,
}

pub(super) fn successors(op: &Op, pc: usize) -> ([usize; 2], usize) {
    match op {
        Op::Jump(target) => ([*target as usize, 0], 1),
        Op::JumpIfFalse(target)
        | Op::JumpIfFalsePeek(target)
        | Op::JumpIfTruePeek(target)
        | Op::JumpIfNotNullishPeek(target)
        | Op::InlineGuard(_, target) => ([*target as usize, pc + 1], 2),
        // AbruptJump is a DYNAMIC completion entry, even if the ordinary CFG records its
        // eventual target as an edge. Intermediate finally pads may shorten its prefix.
        op if bytecode::jit_slice_exit_op(op) => ([0, 0], 0),
        _ => ([pc + 1, 0], 1),
    }
}

impl StackPlan {
    pub(super) fn build(chunk: &Chunk, cfg: &Cfg) -> Option<Self> {
        let ops = chunk.jit_ops();
        let capacity = cfg.max_settled_stack();
        // Compiler/IR budget only. Declining preserves the complete baseline implementation.
        if (ops.len() + 1).checked_mul(capacity + 1)? > 1_000_000 {
            return None;
        }
        let mut at = vec![StackState::default(); ops.len() + 1];
        let reachable: Vec<_> = (0..=ops.len())
            .map(|pc| {
                if let Some(depth) = cfg.stack_depth_at(pc) {
                    at[pc].depth = depth;
                    true
                } else {
                    false
                }
            })
            .collect();
        let mut handler_entry = vec![false; ops.len() + 1];
        for root in cfg.handler_roots() {
            let target = cfg.blocks()[root.target.0 as usize].start;
            handler_entry[target] = true;
            let installed = cfg.stack_depth_at(root.push_pc)?;
            let added = root.stack_depth.checked_sub(installed)?;
            // Only the words explicitly supplied by the completion router are guaranteed.
            let floor = at[target].depth.checked_sub(added)?;
            at[target].floor = at[target].floor.max(floor);
        }
        for op in ops {
            if let Op::AbruptJump(target, _) = op {
                let state = &mut at[*target as usize];
                state.floor = state.depth;
            }
        }
        for block in cfg.blocks() {
            if block.start != 0
                && block.predecessors.is_empty()
                && block.stack_in.is_some()
                && !handler_entry[block.start]
            {
                at[block.start].floor = at[block.start].depth;
            }
        }

        // Least fixed point of the maximum incoming opaque floor. Known-result pushes can
        // grow the suffix even after an operation consumes part of an opaque prefix.
        let mut queue: VecDeque<_> = (0..ops.len()).filter(|&pc| reachable[pc]).collect();
        let mut queued = reachable.clone();
        let mut budget = 8_000_000usize;
        while let Some(pc) = queue.pop_front() {
            budget = budget.checked_sub(1)?;
            queued[pc] = false;
            let (pops, pushes) = chunk.jit_stack_effect(pc)?;
            let out = at[pc].after(pops, pushes)?;
            let (targets, count) = successors(&ops[pc], pc);
            for &target in &targets[..count] {
                if !reachable[target] || at[target].depth != out.depth {
                    return None;
                }
                if out.floor > at[target].floor {
                    at[target].floor = out.floor;
                    if target < ops.len() && !queued[target] {
                        queued[target] = true;
                        queue.push_back(target);
                    }
                }
            }
        }
        let mut after = vec![StackState::default(); ops.len()];
        let mut forget_to = vec![0; ops.len()];
        for (pc, op) in ops.iter().enumerate() {
            if !reachable[pc] {
                continue;
            }
            let (pops, pushes) = chunk.jit_stack_effect(pc)?;
            after[pc] = at[pc].after(pops, pushes)?;
            let (targets, count) = successors(op, pc);
            for &target in &targets[..count] {
                forget_to[pc] = forget_to[pc].max(at[target].floor);
            }
        }
        Some(Self {
            at,
            after,
            forget_to,
            reachable,
            capacity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_suffix_transfer_does_not_restore_consumed_opaque_owners() {
        let incoming = StackState { depth: 8, floor: 6 };
        assert_eq!(incoming.known(), 2);
        assert_eq!(
            incoming.after(1, 2),
            Some(StackState { depth: 9, floor: 6 })
        );
        assert_eq!(
            incoming.after(5, 1),
            Some(StackState { depth: 4, floor: 3 })
        );
        assert_eq!(incoming.after(9, 0), None);
    }

    fn plan(source: &str) -> (std::rc::Rc<Chunk>, Cfg, StackPlan) {
        let statements = crate::parser::parse_script(source, false).ok().unwrap();
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("function fixture")
        };
        let chunk = bytecode::compile(function).unwrap();
        let cfg = Cfg::build(&chunk).unwrap();
        let plan = StackPlan::build(&chunk, &cfg).unwrap();
        (chunk, cfg, plan)
    }

    #[test]
    fn ordinary_diamond_and_loop_keep_the_entire_stack_known() {
        let (_, _, plan) =
            plan("function subject(a,b){for(var k=0;k<20;k++)a=(k?b:a)+a;return a;}");
        for (state, live) in plan.at.iter().zip(&plan.reachable) {
            if *live {
                assert_eq!(state.floor, 0);
            }
        }
    }

    #[test]
    fn completion_entries_never_assume_the_saved_prefix_still_exists() {
        let (chunk, cfg, plan) = plan(
            r#"
            function subject(a) {
                outer:for(var k=0;k<3;k++) {
                    try { let [x]=a; if(x)continue outer;return x; }
                    catch(e) { a=e; }
                    finally { a=a; }
                }
                return a;
            }
        "#,
        );
        assert!(!cfg.handler_roots().is_empty());
        for root in cfg.handler_roots() {
            let target = cfg.blocks()[root.target.0 as usize].start;
            let pushed = root.stack_depth - cfg.stack_depth_at(root.push_pc).unwrap();
            assert!(plan.at[target].known() <= pushed);
        }
        for op in chunk.jit_ops() {
            if let Op::AbruptJump(target, _) = op {
                assert_eq!(plan.at[*target as usize].known(), 0);
            }
        }
    }
}
