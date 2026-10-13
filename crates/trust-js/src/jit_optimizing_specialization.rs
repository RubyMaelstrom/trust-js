//! Live feedback → guarded semantic facts → effect/owner-aware lowering.
//!
//! ECMA-262 e28783d5: FunctionDeclarationInstantiation, GetValue, Number operations,
//! ToBoolean and WeakRef liveness. Entry samples contain no owners and invoke no user
//! operations. They justify selecting a version, never assuming its inputs. Formal facts
//! have guards before PC 0; own numeric field facts have live receiver/descriptor/value
//! guards before each access. A miss continues that same invocation in the VM with its
//! canonical owners. Captured/eval-visible bindings remain environment operations, and
//! the existing effect solver kills private facts on unknown writes.

use super::{
    stack,
    value_facts::{Inputs, Types, ValuePlan},
};
use crate::{
    bytecode::{Chunk, Op},
    feedback::SiteId,
    jit_ir::Cfg,
};

pub(super) struct Opportunity {
    pub(super) pc: usize,
    pub(super) site: Option<SiteId>,
    pub(super) removed_tests: u32,
}

pub(super) struct Plan {
    pub(super) inputs: Vec<(u16, Types)>,
    pub(super) properties: Vec<(u32, Types)>,
    pub(super) values: ValuePlan,
    pub(super) opportunities: Vec<Opportunity>,
    pub(super) saved_tests: u32,
}

impl Plan {
    pub(super) fn build(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        observations: &[(u16, u32)],
        property_observations: &[(u32, u32)],
    ) -> Option<Self> {
        let inputs: Vec<_> = observations
            .iter()
            .filter_map(|&(slot, bits)| {
                let used = chunk
                    .jit_ops()
                    .iter()
                    .any(|op| matches!(op, Op::LoadLocal(s) | Op::UpdateLocal(s, _) if *s == slot));
                used.then(|| Types::guarded_class(bits).map(|ty| (slot, ty)))
                    .flatten()
            })
            .collect();
        let properties: Vec<_> = property_observations
            .iter()
            .filter_map(|&(pc, bits)| {
                let (Op::GetPropLocal(_, _, cache) | Op::GetPropThis(_, cache)) =
                    chunk.jit_ops().get(pc as usize)?
                else {
                    return None;
                };
                // Only the warmed own-data case has this result guard. Prototype/accessor
                // and polymorphic resolver bodies keep their complete ordinary lowering.
                (chunk.jit_feedback_site(pc as usize).is_some()
                    && chunk
                        .jit_cache_preferred(*cache)
                        .is_some_and(|state| state.depth == 0))
                .then(|| {
                    Types::guarded_class(bits)
                        .filter(|ty| ty.is_number())
                        .map(|ty| (pc, ty))
                })
                .flatten()
            })
            .collect();
        if inputs.is_empty() && properties.is_empty() {
            return None;
        }
        let generic = ValuePlan::build(chunk, cfg, stack)?;
        let values = ValuePlan::with_inputs(chunk, cfg, stack, &inputs, &properties)?;
        let mut opportunities = Vec::new();
        let mut removed = 0u32;
        for (pc, op) in chunk.jit_ops().iter().enumerate() {
            let before = tests(op, generic.at[pc]);
            let after = tests(op, values.at[pc]);
            let count = before.saturating_sub(after)
                + u32::from(properties.iter().any(|&(site, _)| site as usize == pc));
            if count != 0 {
                removed = removed.saturating_add(count);
                opportunities.push(Opportunity {
                    pc,
                    site: chunk.jit_feedback_site(pc),
                    removed_tests: count,
                });
            }
        }
        // Each guarded formal pays one class test on every entry. Require an actual net
        // simplification, not just a sampled category or a hotter counter.
        // Each numeric field-result guard replaces its dynamic owner-retain test.
        let saved_tests = removed.saturating_sub((inputs.len() + properties.len()) as u32);
        (saved_tests != 0).then_some(Self {
            inputs,
            properties,
            values,
            opportunities,
            saved_tests,
        })
    }

    pub(super) fn admission_entries(&self, chunk: &Chunk) -> u64 {
        // Diagnostic19 measured median 36.7 IR instructions/op, ~5.9 us of codegen/IR
        // instruction. Use a 40-instruction envelope and 1024 previously seen removable
        // tests per predicted IR instruction as a conservative initial work budget.
        // This is an admission estimate, not a speed claim; controlled workloads must
        // validate repayment. It is intentionally much stricter than the old 512 calls.
        (chunk.jit_ops().len() as u64)
            .saturating_mul(40)
            .saturating_add(super::names::instruction_allowance(chunk) as u64)
            .saturating_mul(1024)
            .div_ceil(u64::from(self.saved_tests))
    }
}

pub(super) fn tests(op: &Op, inputs: Inputs) -> u32 {
    let number = |ty: Types| u32::from(!ty.is_number());
    let owner = |ty: Types| u32::from(!ty.is_copyable());
    match op {
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
        | Op::Lt
        | Op::Le
        | Op::Gt
        | Op::Ge
        | Op::StrictEq
        | Op::StrictNotEq => number(inputs.top) + number(inputs.second),
        Op::LoadLocal(_) => owner(inputs.local) + u32::from(!inputs.local.initialized()),
        Op::UpdateLocal(..) => number(inputs.local),
        Op::StoreLocal(_) => owner(inputs.local) + owner(inputs.top),
        Op::Dup | Op::Pop => owner(inputs.top),
        Op::JumpIfFalse(_) | Op::JumpIfFalsePeek(_) | Op::JumpIfTruePeek(_) => {
            u32::from(!inputs.top.is_boolean() && !inputs.top.is_number())
        }
        _ => 0,
    }
}
