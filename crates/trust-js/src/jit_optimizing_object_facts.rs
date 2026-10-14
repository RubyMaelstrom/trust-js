//! Guarded own-data Object results and certificates for exact-null strict equality.
//!
//! ECMA-262 e28783d5 OrdinaryGet/IsStrictlyEqual: only the live data value's PACK_OBJ
//! guard creates a fact. A miss resumes BEFORE the read. Copied owning words retain
//! their category according to ValuePlan's slot/effect rules, never a heap-content
//! identity. IsHTMLDDA objects are still unequal to Null under strict equality;
//! this plan makes no truthiness, loose-equality or typeof assumption.
//! https://tc39.es/ecma262/#sec-isstrictlyequal

use super::*;
use crate::{bytecode::IcState, feedback::ValueClass};

#[cfg(test)]
thread_local! {
    pub(super) static TEST_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    pub(super) static TEST_EMITTED: std::cell::Cell<[usize; 3]> = const { std::cell::Cell::new([0; 3]) };
}

pub(super) fn enabled() -> bool {
    #[cfg(test)]
    let requested = TEST_ENABLED
        .with(std::cell::Cell::get)
        .unwrap_or_else(|| std::env::var("TRUST_JS_OPT_JIT_OBJECT_FIELDS").as_deref() == Ok("1"));
    #[cfg(not(test))]
    let requested = std::env::var("TRUST_JS_OPT_JIT_OBJECT_FIELDS").as_deref() == Ok("1");
    requested
        && generic_admission::enabled()
        && std::env::var("TRUST_JS_OPT_JIT_VALUE_FACTS").as_deref() != Ok("0")
}

pub(super) fn selected(
    properties: &[(u32, u32)],
    structural: &[(u32, IcState)],
) -> Vec<(u32, IcState)> {
    if !enabled() {
        return Vec::new();
    }
    structural
        .iter()
        .filter(|&&(pc, _)| {
            properties
                .iter()
                .any(|&(site, bits)| site == pc && bits == ValueClass::Object.bit())
        })
        .copied()
        .collect()
}

pub(super) struct Plan {
    pub(super) values: value_facts::ValuePlan,
    pub(super) properties: Vec<(u32, value_facts::Types)>,
    comparisons: Vec<Option<bool>>,
    branches: Vec<Option<bool>>,
    pub(super) reachable: Vec<bool>,
    pub(super) guard_cost: u32,
}

impl Plan {
    pub(super) fn build(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        objects: &[(u32, IcState)],
        structural: &[(u32, IcState)],
        inputs: &[(u16, value_facts::Types)],
        numeric_properties: &[(u32, value_facts::Types)],
    ) -> Option<Self> {
        let ops = chunk.jit_ops();
        if !enabled()
            || objects.is_empty()
            || objects.len() > 16
            || ops.len() > 256
            || !cfg.handler_roots().is_empty()
            || !cfg.resume_roots().is_empty()
            || ops.iter().any(|op| matches!(op, Op::AbruptJump(..)))
        {
            return None;
        }
        let mut properties = numeric_properties.to_vec();
        let mut previous = None;
        for &(pc, state) in objects {
            if previous.is_some_and(|old| pc <= old) {
                return None;
            }
            previous = Some(pc);
            let (name, cache) = match *ops.get(pc as usize)? {
                Op::GetPropLocal(slot, name, cache)
                    if usize::from(slot) < chunk.jit_frame().0.min(16) =>
                {
                    (name, cache)
                }
                Op::GetPropThis(name, cache) => (name, cache),
                _ => return None,
            };
            if crate::value::canonical_index(chunk.jit_name(name)).is_some()
                || state.depth != 0
                || !state.has_cacheable_shapes()
                || !structural
                    .iter()
                    .any(|&(site, other)| site == pc && generic_admission::same(state, other))
                || !chunk
                    .jit_cache_preferred(cache)
                    .is_some_and(|live| generic_admission::same(state, live))
                || properties.iter().any(|&(site, _)| site == pc)
            {
                return None;
            }
            // Debit one result guard per invocation. It is sound only when each selected
            // guard is outside every native cycle, so it cannot execute repeatedly.
            let block = cfg.block_at(pc as usize)?.0 as usize;
            if cyclic(cfg, block) {
                return None;
            }
            properties.push((pc, value_facts::Types::OBJECT));
        }
        let values = value_facts::ValuePlan::with_inputs(chunk, cfg, stack, inputs, &properties)?;
        let mut comparisons = vec![None; ops.len()];
        let mut branches = vec![None; ops.len()];
        for (pc, op) in ops.iter().enumerate() {
            if !matches!(op, Op::StrictEq | Op::StrictNotEq) || stack.at[pc].known() < 2 {
                continue;
            }
            let input = values.at[pc];
            if !(input.top.is_object() && input.second.is_null()
                || input.top.is_null() && input.second.is_object())
            {
                continue;
            }
            comparisons[pc] = Some(matches!(op, Op::StrictNotEq));
            if matches!(
                ops.get(pc + 1),
                Some(Op::JumpIfFalse(_) | Op::JumpIfFalsePeek(_) | Op::JumpIfTruePeek(_))
            ) && cfg.block_at(pc) == cfg.block_at(pc + 1)
                && stack.at[pc + 1].known() != 0
            {
                branches[pc + 1] = comparisons[pc];
            }
        }
        // Solve from the original BOTH-edge value facts, not from facts invented by
        // pruning an edge. This avoids cyclic speculation and stale stored predicates.
        let mut reachable = vec![false; ops.len() + 1];
        let mut queue = std::collections::VecDeque::from([0usize]);
        while let Some(pc) = queue.pop_front() {
            if pc > ops.len() || reachable[pc] {
                continue;
            }
            reachable[pc] = true;
            if pc == ops.len() {
                continue;
            }
            let (targets, count) = stack::successors(&ops[pc], pc);
            for &target in &targets[..count] {
                if accepts(&ops[pc], pc, branches[pc], target) {
                    queue.push_back(target);
                }
            }
        }
        Some(Self {
            values,
            properties,
            comparisons,
            branches,
            reachable,
            guard_cost: objects.len() as u32,
        })
    }

    pub(super) fn comparison(&self, pc: usize) -> Option<bool> {
        self.comparisons.get(pc).copied().flatten()
    }
    pub(super) fn branch(&self, pc: usize) -> Option<bool> {
        self.branches.get(pc).copied().flatten()
    }
    pub(super) fn accepts(&self, chunk: &Chunk, pc: usize, target: usize) -> bool {
        accepts(&chunk.jit_ops()[pc], pc, self.branch(pc), target)
    }
}

fn accepts(op: &Op, pc: usize, truth: Option<bool>, target: usize) -> bool {
    let Some(truth) = truth else {
        return true;
    };
    let selected = match *op {
        Op::JumpIfFalse(destination) | Op::JumpIfFalsePeek(destination) => {
            if truth {
                pc + 1
            } else {
                destination as usize
            }
        }
        Op::JumpIfTruePeek(destination) => {
            if truth {
                destination as usize
            } else {
                pc + 1
            }
        }
        _ => return true,
    };
    target == selected
}

fn cyclic(cfg: &Cfg, source: usize) -> bool {
    let mut seen = vec![false; cfg.blocks().len()];
    let mut queue: std::collections::VecDeque<_> = cfg.blocks()[source]
        .successors
        .iter()
        .map(|b| b.0 as usize)
        .collect();
    while let Some(block) = queue.pop_front() {
        if block == source {
            return true;
        }
        if seen[block] {
            continue;
        }
        seen[block] = true;
        queue.extend(cfg.blocks()[block].successors.iter().map(|b| b.0 as usize));
    }
    false
}

#[cfg(test)]
pub(super) fn record(index: usize) {
    TEST_EMITTED.with(|cell| {
        let mut counts = cell.get();
        counts[index] += 1;
        cell.set(counts);
    });
}

impl Lowering<'_, '_> {
    pub(super) fn object_comparison(&mut self, pc: usize, next: Block) -> bool {
        let Some(truth) = self.object_plan.and_then(|plan| plan.comparison(pc)) else {
            return false;
        };
        let slow = self.b.create_block();
        let input = self.input_types();
        let owner = if input.top.is_object() {
            self.stack_read(1)
        } else {
            self.stack_read(2)
        };
        // The Object word remains an owner. Last-owner destruction must execute the
        // real helper with both original operands and counts unchanged.
        self.transfer_owners(&[(owner, false)], slow);
        let result = self.b.ins().iconst(
            types::I64,
            (crate::value::PACK_BOOL | u64::from(truth)) as i64,
        );
        self.stack_write(2, result);
        self.change_top(-8);
        self.jump_normal(next);
        self.slow(pc, slow, next);
        #[cfg(test)]
        record(1);
        true
    }
}
