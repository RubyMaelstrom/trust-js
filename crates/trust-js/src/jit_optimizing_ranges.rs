//! Integer intervals proved over normal whole-function CFG edges.
//!
//! ECMA-262 e28783d5, Number operations and ToInt32: a range is a finite,
//! exactly integral Number and never minus zero. Samples do not establish a
//! range. Constants and successful language operations do; mixed Number/BigInt
//! operations retain their checked path, and only their possible normal Number
//! results get integer facts. Unknown effects kill the actual private writes.
//! This analysis changes neither owning words nor recovery boundaries.

use super::{effects, integers, stack, value_facts};
use crate::bytecode::{Chunk, Op, UpdKind};
use crate::jit_ir::Cfg;
use std::collections::VecDeque;

const SAFE_MAX: i64 = 9_007_199_254_740_991;

#[cfg(test)]
thread_local! {
    pub(super) static TEST_PROOFS: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
}

#[cfg(test)]
pub(super) fn record_conversion_proof() {
    TEST_PROOFS.with(|counts| {
        let mut values = counts.get();
        values[0] += 1;
        counts.set(values);
    });
}

#[cfg(test)]
pub(super) fn record_index_proof() {
    TEST_PROOFS.with(|counts| {
        let mut values = counts.get();
        values[1] += 1;
        counts.set(values);
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IntegerRange {
    pub(super) min: i64,
    pub(super) max: i64,
}

impl IntegerRange {
    const I32: Self = Self {
        min: i32::MIN as i64,
        max: i32::MAX as i64,
    };
    const U32: Self = Self {
        min: 0,
        max: u32::MAX as i64,
    };

    fn new(min: i64, max: i64) -> Option<Self> {
        (min <= max && min >= -SAFE_MAX && max <= SAFE_MAX).then_some(Self { min, max })
    }

    fn constant(number: f64) -> Option<Self> {
        (number.is_finite()
            && number.fract() == 0.0
            && number.abs() <= SAFE_MAX as f64
            && !(number == 0.0 && number.is_sign_negative()))
        .then_some(Self {
            min: number as i64,
            max: number as i64,
        })
    }

    pub(super) fn signed32(self) -> bool {
        self.min >= i32::MIN as i64 && self.max <= i32::MAX as i64
    }

    pub(super) fn unsigned32(self) -> bool {
        self.min >= 0 && self.max <= u32::MAX as i64
    }

    pub(super) fn index(self) -> bool {
        self.min >= 0 && self.max < u32::MAX as i64
    }

    fn singleton(self) -> Option<i64> {
        (self.min == self.max).then_some(self.min)
    }

    fn hull(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn intersect(self, other: Self) -> Option<Self> {
        Self::new(self.min.max(other.min), self.max.min(other.max))
    }

    fn arithmetic(self, other: Self, op: &Op) -> Option<Self> {
        match op {
            Op::Add => Self::new(
                self.min.checked_add(other.min)?,
                self.max.checked_add(other.max)?,
            ),
            Op::Sub => Self::new(
                self.min.checked_sub(other.max)?,
                self.max.checked_sub(other.min)?,
            ),
            Op::Mul => {
                // Zero multiplied by a negative integer is minus zero.
                if (self.min <= 0 && self.max >= 0 && other.min < 0)
                    || (other.min <= 0 && other.max >= 0 && self.min < 0)
                {
                    return None;
                }
                let products = [
                    self.min.checked_mul(other.min)?,
                    self.min.checked_mul(other.max)?,
                    self.max.checked_mul(other.min)?,
                    self.max.checked_mul(other.max)?,
                ];
                Self::new(*products.iter().min()?, *products.iter().max()?)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Inputs {
    pub(super) top: Option<IntegerRange>,
    pub(super) second: Option<IntegerRange>,
    pub(super) local: Option<IntegerRange>,
}

pub(super) struct NumericPlan {
    pub(super) at: Vec<Inputs>,
    pub(super) integers: Option<integers::Facts>,
}

#[derive(Clone, Copy)]
enum Relation {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

impl Relation {
    fn from_op(op: &Op) -> Option<Self> {
        Some(match op {
            Op::Lt => Self::Lt,
            Op::Le => Self::Le,
            Op::Gt => Self::Gt,
            Op::Ge => Self::Ge,
            Op::StrictEq | Op::EqEq => Self::Eq,
            Op::StrictNotEq | Op::NotEq => Self::Ne,
            _ => return None,
        })
    }

    fn negate(self) -> Self {
        match self {
            Self::Lt => Self::Ge,
            Self::Le => Self::Gt,
            Self::Gt => Self::Le,
            Self::Ge => Self::Lt,
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
        }
    }

    fn refine(self, lhs: IntegerRange, rhs: IntegerRange) -> Option<(IntegerRange, IntegerRange)> {
        match self {
            Self::Gt => Self::Lt.refine(rhs, lhs).map(|(r, l)| (l, r)),
            Self::Ge => Self::Le.refine(rhs, lhs).map(|(r, l)| (l, r)),
            Self::Lt | Self::Le => {
                let distance = i64::from(matches!(self, Self::Lt));
                Some((
                    IntegerRange::new(lhs.min, lhs.max.min(rhs.max - distance))?,
                    IntegerRange::new(rhs.min.max(lhs.min + distance), rhs.max)?,
                ))
            }
            Self::Eq => {
                let common = lhs.intersect(rhs)?;
                Some((common, common))
            }
            Self::Ne => {
                if lhs.singleton().is_some() && lhs == rhs {
                    return None;
                }
                let exclude = |range: IntegerRange, point: Option<i64>| {
                    let min = range.min + i64::from(point == Some(range.min));
                    let max = range.max - i64::from(point == Some(range.max));
                    IntegerRange::new(min, max)
                };
                Some((
                    exclude(lhs, rhs.singleton())?,
                    exclude(rhs, lhs.singleton())?,
                ))
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Comparison {
    pc: usize,
    relation: Relation,
    lhs: IntegerRange,
    rhs: IntegerRange,
    left_local: Option<u16>,
    right_local: Option<u16>,
}

fn local_operand(op: &Op) -> Option<u16> {
    match *op {
        Op::LoadLocal(slot) | Op::StoreLocal(slot) | Op::UpdateLocal(slot, _) => Some(slot),
        _ => None,
    }
}

fn numeric_bitwise_result(op: &Op, inputs: value_facts::Inputs, ranges: Inputs) -> bool {
    matches!(op, Op::UShr)
        || inputs.top.is_number()
        || inputs.second.is_number()
        || ranges.top.is_some()
        || ranges.second.is_some()
}

fn bitwise_range(op: &Op, lhs: Option<IntegerRange>, rhs: Option<IntegerRange>) -> IntegerRange {
    let constant = rhs.and_then(IntegerRange::singleton);
    if let (Some(left), Some(right)) = (lhs.and_then(IntegerRange::singleton), constant) {
        let left = left as i32;
        let right = right as i32;
        let result = match op {
            Op::BitAnd => i64::from(left & right),
            Op::BitOr => i64::from(left | right),
            Op::BitXor => i64::from(left ^ right),
            Op::Shl => i64::from(left.wrapping_shl(right as u32 & 31)),
            Op::Shr => i64::from(left.wrapping_shr(right as u32 & 31)),
            Op::UShr => i64::from((left as u32).wrapping_shr(right as u32 & 31)),
            _ => unreachable!(),
        };
        return IntegerRange {
            min: result,
            max: result,
        };
    }
    match op {
        Op::BitAnd => {
            let mask = constant.or_else(|| lhs.and_then(IntegerRange::singleton));
            if let Some(mask) = mask.filter(|&mask| (mask as i32) >= 0) {
                IntegerRange {
                    min: 0,
                    max: i64::from(mask as i32),
                }
            } else {
                IntegerRange::I32
            }
        }
        Op::Shr => {
            if let (Some(lhs), Some(shift)) = (lhs.filter(|range| range.signed32()), constant) {
                let shift = shift as u32 & 31;
                IntegerRange {
                    min: lhs.min >> shift,
                    max: lhs.max >> shift,
                }
            } else {
                IntegerRange::I32
            }
        }
        Op::UShr => {
            let shift = constant.map_or(0, |shift| shift as u32 & 31);
            if let Some(lhs) = lhs.filter(|range| range.min >= 0 && range.max <= i32::MAX as i64) {
                IntegerRange {
                    min: constant.map_or(0, |_| lhs.min >> shift),
                    max: lhs.max >> shift,
                }
            } else {
                IntegerRange {
                    max: IntegerRange::U32.max >> shift,
                    ..IntegerRange::U32
                }
            }
        }
        _ => IntegerRange::I32,
    }
}

impl NumericPlan {
    pub(super) fn build(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        values: Option<&value_facts::ValuePlan>,
    ) -> Option<Self> {
        Self::solve(chunk, cfg, stack, values, 1_000_000, 8_000_000)
    }

    pub(super) fn with_integer_demand(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        values: Option<&value_facts::ValuePlan>,
        demand: Option<&integers::Demand>,
    ) -> Option<Self> {
        Self::solve_with_demand(chunk, cfg, stack, values, 1_000_000, 8_000_000, demand)
    }

    fn solve(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        values: Option<&value_facts::ValuePlan>,
        cells_limit: usize,
        work: usize,
    ) -> Option<Self> {
        Self::solve_with_demand(chunk, cfg, stack, values, cells_limit, work, None)
    }

    fn solve_with_demand(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        values: Option<&value_facts::ValuePlan>,
        cells_limit: usize,
        mut work: usize,
        demand: Option<&integers::Demand>,
    ) -> Option<Self> {
        let ops = chunk.jit_ops();
        let slots = chunk.jit_frame().1;
        let mut integer_facts = demand.map(integers::Facts::new);
        let mut integer_work = 8_000_000usize;
        let width = slots.checked_add(stack.capacity)?;
        let blocks = cfg.blocks();
        let cells = width.checked_mul(blocks.len())?;
        if cells > cells_limit {
            return None;
        }
        work = work.checked_sub(ops.len())?;
        // Widen growing loops toward syntactic constants, never inventing an I32
        // bound. If safe-integer precision is lost the fact becomes unknown.
        let mut thresholds = vec![
            -SAFE_MAX,
            i32::MIN as i64,
            -1,
            0,
            1,
            i32::MAX as i64,
            u32::MAX as i64,
            SAFE_MAX,
        ];
        for op in ops {
            if let Op::Const(index) = *op {
                if let Some(range) = chunk
                    .jit_const_num(index)
                    .and_then(|bits| IntegerRange::constant(f64::from_bits(bits)))
                {
                    thresholds.push(range.min);
                    if thresholds.len() == 64 {
                        break;
                    }
                }
            }
        }
        thresholds.sort_unstable();
        thresholds.dedup();
        let mut loop_headers = vec![false; blocks.len()];
        for natural in cfg.loops() {
            loop_headers[natural.header.0 as usize] = true;
        }
        let mut entries: Vec<Option<IntegerRange>> = vec![None; cells];
        let mut growth = vec![0u8; cells];
        let mut reached = vec![false; blocks.len()];
        let mut queued = vec![false; blocks.len()];
        let mut queue = VecDeque::new();
        let mut roots = vec![false; blocks.len()];
        roots[cfg.block_at(0)?.0 as usize] = true;
        for root in cfg.handler_roots() {
            roots[root.target.0 as usize] = true;
        }
        for root in cfg.resume_roots() {
            roots[root.target.0 as usize] = true;
        }
        for (index, block) in blocks.iter().enumerate() {
            if block.stack_in.is_some() && block.predecessors.is_empty() {
                roots[index] = true;
            }
        }
        for op in ops {
            if let Op::AbruptJump(target, _) = *op {
                roots[cfg.block_at(target as usize)?.0 as usize] = true;
            }
        }
        for (index, root) in roots.into_iter().enumerate() {
            if root {
                reached[index] = true;
                queued[index] = true;
                queue.push_back(index);
            }
        }
        let mut scratch = vec![None; width];
        // Origins exist only within one basic block. A stored comparison or loop
        // phi cannot retain the identity of an older dynamic local value.
        let mut origins = vec![None; stack.capacity];
        let mut at = vec![Inputs::default(); ops.len()];
        while let Some(index) = queue.pop_front() {
            queued[index] = false;
            let block = &blocks[index];
            work = work.checked_sub(width + 1)?;
            scratch.copy_from_slice(&entries[index * width..(index + 1) * width]);
            origins.fill(None);
            let mut comparison = None;
            for pc in block.start..block.end {
                work = work.checked_sub(width + 1)?;
                let before = stack.at[pc];
                scratch[slots..slots + before.floor].fill(None);
                origins[..before.floor].fill(None);
                let (pops, pushes) = chunk.jit_stack_effect(pc)?;
                let kept = before.depth.checked_sub(pops)?;
                let input = |back| {
                    if before.known() >= back {
                        scratch[slots + before.depth - back]
                    } else {
                        None
                    }
                };
                let inputs = Inputs {
                    top: input(1),
                    second: input(2),
                    local: local_operand(&ops[pc]).and_then(|slot| scratch[usize::from(slot)]),
                };
                at[pc] = inputs;
                if let (Some(facts), Some(demand)) = (&mut integer_facts, demand) {
                    let used = facts.record(demand, pc, integers::Point::Before, &scratch);
                    if let Some(remaining) = integer_work.checked_sub(used) {
                        integer_work = remaining;
                    } else {
                        integer_facts = None;
                    }
                }
                let types = values.map_or_else(value_facts::Inputs::default, |plan| plan.at[pc]);
                let mut results = [None; 4];
                let mut result_origins = [None; 4];
                if let (Some(relation), Some(lhs), Some(rhs)) =
                    (Relation::from_op(&ops[pc]), inputs.second, inputs.top)
                {
                    comparison = Some(Comparison {
                        pc,
                        relation,
                        lhs,
                        rhs,
                        left_local: origins[before.depth - 2],
                        right_local: origins[before.depth - 1],
                    });
                }
                match ops[pc] {
                    Op::Const(k) => {
                        results[0] = chunk
                            .jit_const_num(k)
                            .and_then(|bits| IntegerRange::constant(f64::from_bits(bits)))
                    }
                    Op::LoadLocal(slot) => {
                        results[0] = inputs.local;
                        result_origins[0] = Some(slot);
                    }
                    Op::Dup => {
                        results[..2].fill(inputs.top);
                        result_origins[..2].fill(origins[before.depth - 1]);
                    }
                    Op::Dup2 => {
                        results = [inputs.second, inputs.top, inputs.second, inputs.top];
                        let left = origins[before.depth - 2];
                        let right = origins[before.depth - 1];
                        result_origins = [left, right, left, right];
                    }
                    ref op @ (Op::Add | Op::Sub | Op::Mul) => {
                        results[0] = inputs
                            .second
                            .zip(inputs.top)
                            .and_then(|(left, right)| left.arithmetic(right, op))
                    }
                    ref
                    op @ (Op::BitAnd | Op::BitOr | Op::BitXor | Op::Shl | Op::Shr | Op::UShr)
                        if numeric_bitwise_result(op, types, inputs) =>
                    {
                        results[0] = Some(bitwise_range(op, inputs.second, inputs.top))
                    }
                    Op::BitNot if inputs.top.is_some() || types.top.is_number() => {
                        results[0] = Some(inputs.top.and_then(IntegerRange::singleton).map_or(
                            IntegerRange::I32,
                            |value| {
                                let result = i64::from(!(value as i32));
                                IntegerRange {
                                    min: result,
                                    max: result,
                                }
                            },
                        ));
                    }
                    Op::Neg => {
                        results[0] = inputs
                            .top
                            .filter(|range| range.min > 0 || range.max < 0)
                            .and_then(|range| IntegerRange::new(-range.max, -range.min))
                    }
                    Op::ToPropKey => {
                        // This ABI preserves BOTH the receiver and the key.
                        // A numeric key is never the receiver's value fact.
                        results[..2].copy_from_slice(&[inputs.second, inputs.top]);
                        result_origins[..2].copy_from_slice(&[
                            origins[before.depth - 2],
                            origins[before.depth - 1],
                        ]);
                    }
                    Op::Plus
                    | Op::ToPropKeyLocal(_)
                    | Op::RequireObject
                    | Op::DestructureGuard
                    | Op::JumpIfFalsePeek(_)
                    | Op::JumpIfTruePeek(_)
                    | Op::JumpIfNotNullishPeek(_)
                    | Op::SetProp(..)
                    | Op::SetElem
                    | Op::SetElemLocal(_) => results[0] = inputs.top,
                    Op::UpdateLocal(_, kind) => {
                        let delta = if matches!(
                            kind,
                            UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
                        ) {
                            -1
                        } else {
                            1
                        };
                        let updated = inputs.local.and_then(|range| {
                            range.arithmetic(
                                IntegerRange {
                                    min: delta,
                                    max: delta,
                                },
                                &Op::Add,
                            )
                        });
                        results[0] = if matches!(kind, UpdKind::PostInc | UpdKind::PostDec) {
                            inputs.local
                        } else {
                            updated
                        };
                    }
                    _ => {}
                }
                let writes = effects::checked_writes(&ops[pc]);
                for (slot, value) in scratch[..slots].iter_mut().enumerate() {
                    if writes.contains(slot as u16) {
                        *value = None;
                    }
                }
                for origin in &mut origins {
                    if origin.is_some_and(|slot| writes.contains(slot)) {
                        *origin = None;
                    }
                }
                match ops[pc] {
                    Op::StoreLocal(slot) => scratch[usize::from(slot)] = inputs.top,
                    Op::UpdateLocal(slot, kind) => {
                        let delta = if matches!(
                            kind,
                            UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
                        ) {
                            -1
                        } else {
                            1
                        };
                        scratch[usize::from(slot)] = inputs.local.and_then(|range| {
                            range.arithmetic(
                                IntegerRange {
                                    min: delta,
                                    max: delta,
                                },
                                &Op::Add,
                            )
                        });
                    }
                    _ => {}
                }
                scratch[slots + kept..slots + kept + pushes].fill(None);
                origins[kept..kept + pushes].fill(None);
                let count = pushes.min(4);
                scratch[slots + kept..slots + kept + count].copy_from_slice(&results[..count]);
                origins[kept..kept + count].copy_from_slice(&result_origins[..count]);
                if let (Some(facts), Some(demand)) = (&mut integer_facts, demand) {
                    let used = facts.record(demand, pc, integers::Point::After, &scratch);
                    if let Some(remaining) = integer_work.checked_sub(used) {
                        integer_work = remaining;
                    } else {
                        integer_facts = None;
                    }
                }
            }
            let last = block.end - 1;
            let (targets, count) = stack::successors(&ops[last], last);
            for &target in &targets[..count] {
                let Some(target_block) = cfg.block_at(target) else {
                    if target == ops.len() {
                        continue;
                    }
                    return None;
                };
                work = work.checked_sub(width + 1)?;
                let mut edge = scratch.clone();
                if let Some(comparison) = comparison.filter(|comparison| comparison.pc + 1 == last)
                {
                    let truth = match ops[last] {
                        Op::JumpIfFalse(destination) | Op::JumpIfFalsePeek(destination)
                            if destination as usize != last + 1 =>
                        {
                            Some(target != destination as usize)
                        }
                        Op::JumpIfTruePeek(destination) if destination as usize != last + 1 => {
                            Some(target == destination as usize)
                        }
                        _ => None,
                    };
                    if let Some(truth) = truth {
                        let relation = if truth {
                            comparison.relation
                        } else {
                            comparison.relation.negate()
                        };
                        let Some((left, right)) = relation.refine(comparison.lhs, comparison.rhs)
                        else {
                            continue;
                        };
                        for (local, narrowed) in [
                            (comparison.left_local, left),
                            (comparison.right_local, right),
                        ] {
                            if let Some(local) = local {
                                let Some(range) = edge[usize::from(local)]
                                    .and_then(|range| range.intersect(narrowed))
                                else {
                                    continue;
                                };
                                edge[usize::from(local)] = Some(range);
                            }
                        }
                    }
                }
                let destination = target_block.0 as usize;
                let state = stack.at[target];
                let used = slots + state.depth;
                work = work.checked_sub(used + 1)?;
                edge[slots..slots + state.floor].fill(None);
                let mut changed = !reached[destination];
                for word in 0..used {
                    let cell = destination * width + word;
                    let next = if !reached[destination] {
                        edge[word]
                    } else {
                        entries[cell].zip(edge[word]).map(|(old, new)| {
                            let joined = old.hull(new);
                            if joined != old && growth[cell] >= 2 && loop_headers[destination] {
                                IntegerRange {
                                    min: if joined.min < old.min {
                                        *thresholds
                                            .iter()
                                            .rev()
                                            .find(|&&value| value <= joined.min)
                                            .unwrap()
                                    } else {
                                        joined.min
                                    },
                                    max: if joined.max > old.max {
                                        *thresholds
                                            .iter()
                                            .find(|&&value| value >= joined.max)
                                            .unwrap()
                                    } else {
                                        joined.max
                                    },
                                }
                            } else {
                                joined
                            }
                        })
                    };
                    if next != entries[cell] {
                        changed = true;
                        growth[cell] = growth[cell].saturating_add(1);
                        entries[cell] = next;
                    }
                }
                reached[destination] = true;
                if changed && !queued[destination] {
                    queued[destination] = true;
                    queue.push_back(destination);
                }
            }
        }
        if let Some(facts) = &mut integer_facts {
            facts.finish(slots);
        }
        Some(Self {
            at,
            integers: integer_facts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode;
    use std::rc::Rc;

    fn plan(source: &str) -> (Rc<Chunk>, NumericPlan) {
        let statements = crate::parser::parse_script(source, false)
            .ok()
            .expect("range fixture parses");
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("function fixture");
        };
        let chunk = bytecode::compile(function).unwrap();
        let cfg = Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        let values = value_facts::ValuePlan::build(&chunk, &cfg, &stack).unwrap();
        let ranges = NumericPlan::build(&chunk, &cfg, &stack, Some(&values)).unwrap();
        (chunk, ranges)
    }

    fn at(source: &str, predicate: impl Fn(&Op) -> bool) -> Vec<Inputs> {
        let (chunk, plan) = plan(source);
        chunk
            .jit_ops()
            .iter()
            .enumerate()
            .filter_map(|(pc, op)| predicate(op).then_some(plan.at[pc]))
            .collect()
    }

    #[test]
    fn constants_ranges_and_signed_zero_have_exact_boundaries() {
        for number in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.0,
            0.5,
            9_007_199_254_740_992.0,
        ] {
            assert_eq!(IntegerRange::constant(number), None);
        }
        for number in [
            0.0,
            -1.0,
            i32::MIN as f64,
            i32::MAX as f64,
            u32::MAX as f64,
            SAFE_MAX as f64,
        ] {
            let range = IntegerRange::constant(number).unwrap();
            assert_eq!(range.min as f64, number);
            assert_eq!(range.max as f64, number);
        }
        assert!(IntegerRange::I32.signed32());
        assert!(!IntegerRange::U32.signed32());
        assert!(IntegerRange::U32.unsigned32());
        assert!(!IntegerRange::U32.index());
        assert!(IntegerRange::new(0, u32::MAX as i64 - 1).unwrap().index());
        assert!(!IntegerRange::new(-1, 255).unwrap().index());
    }

    #[test]
    fn arithmetic_drops_overflow_precision_and_negative_zero_proofs() {
        let exact = |value| IntegerRange {
            min: value,
            max: value,
        };
        assert_eq!(
            exact(i32::MAX as i64).arithmetic(exact(1), &Op::Add),
            Some(exact(2_147_483_648))
        );
        assert_eq!(exact(SAFE_MAX).arithmetic(exact(1), &Op::Add), None);
        assert_eq!(exact(SAFE_MAX).arithmetic(exact(2), &Op::Mul), None);
        assert_eq!(exact(0).arithmetic(exact(-1), &Op::Mul), None);
        assert_eq!(exact(-1).arithmetic(exact(0), &Op::Mul), None);
        assert_eq!(exact(0).arithmetic(exact(1), &Op::Mul), Some(exact(0)));
        assert_eq!(exact(-2).arithmetic(exact(-3), &Op::Mul), Some(exact(6)));
        assert_eq!(
            at("function f(){var n=0;return(-n)|0;}", |op| matches!(
                op,
                Op::BitOr
            ))[0]
                .second,
            None
        );
    }

    #[test]
    fn bitwise_normal_results_keep_number_bigint_and_unsigned_distinctions() {
        let bigint = at("function f(a,b){var n=a&b;return n|0;}", |op| {
            matches!(op, Op::BitOr)
        });
        assert_eq!(
            bigint[0].second, None,
            "two unknown inputs may yield BigInt"
        );
        let integer = at("function f(a){var n=a|0;return n&255;}", |op| {
            matches!(op, Op::BitAnd)
        });
        assert_eq!(
            integer[0].second,
            Some(IntegerRange::I32),
            "a normal result with Number RHS is Number"
        );
        let unsigned = at("function f(a){var n=a>>>0;return n|0;}", |op| {
            matches!(op, Op::BitOr)
        });
        assert_eq!(unsigned[0].second, Some(IntegerRange::U32));
        for shift in [0, 1, 31, 32, 33, 63, -1] {
            let left = IntegerRange::constant(-1.0);
            let right = IntegerRange::constant(shift as f64);
            let range = bitwise_range(&Op::UShr, left, right);
            assert_eq!(
                range.singleton(),
                Some(i64::from(u32::MAX >> (shift as u32 & 31)))
            );
        }
        let range = bitwise_range(&Op::UShr, IntegerRange::new(1, 255), None);
        assert_eq!(range, IntegerRange { min: 0, max: 255 });
    }

    #[test]
    fn computed_compound_assignment_preserves_receiver_and_key_facts_separately() {
        let (chunk, plan) = plan("function f(object){var key=1;return((0,object)[key]+=3);}");
        let pc = chunk
            .jit_ops()
            .iter()
            .position(|op| matches!(op, Op::ToPropKey))
            .expect("generic computed receiver");
        assert!(matches!(chunk.jit_ops()[pc + 1], Op::Dup2));
        assert_eq!(
            plan.at[pc + 1].second,
            None,
            "the Object receiver is not its numeric key"
        );
        assert_eq!(plan.at[pc + 1].top, IntegerRange::new(1, 1));
    }

    #[test]
    fn edge_refinement_and_loop_widening_feed_real_index_reads() {
        let reads = at(
            "function f(a){var sum=0;for(var i=0;i<256;i++)sum+=a[i];return sum;}",
            |op| matches!(op, Op::GetElem | Op::GetElemLocal(_)),
        );
        assert_eq!(reads.len(), 1);
        assert_eq!(reads[0].top, IntegerRange::new(0, 255));
        let unbounded = at(
            "function f(a,n){var i=0;while(n){i=i+1;}return i|0;}",
            |op| matches!(op, Op::BitOr),
        );
        assert_eq!(
            unbounded[0].second, None,
            "widening may not invent an I32 bound"
        );
        let phi = at(
            "function f(flag,a){var i;if(flag)i=3;else i=7;return a[i];}",
            |op| matches!(op, Op::GetElem | Op::GetElemLocal(_)),
        );
        assert_eq!(phi[0].top, IntegerRange::new(3, 7));
    }

    #[test]
    fn stored_comparisons_and_observer_bridges_do_not_refine_reassigned_locals() {
        let stored = at(
            "function f(a,n){var i=n|0;var yes=i>=0;i=-1;if(yes)return a[i];return 0;}",
            |op| matches!(op, Op::GetElem | Op::GetElemLocal(_)),
        );
        assert_eq!(stored[0].top, IntegerRange::new(-1, -1));
        let eval = at("function f(a){var i=3;eval('i=1.5');return a[i];}", |op| {
            matches!(op, Op::GetElem | Op::GetElemLocal(_))
        });
        assert_eq!(eval[0].top, None);
        let private = at(
            "function f(a,observe){var i=3;observe();return a[i];}",
            |op| matches!(op, Op::GetElem | Op::GetElemLocal(_)),
        );
        assert_eq!(
            private[0].top,
            IntegerRange::new(3, 3),
            "ordinary callbacks cannot replace private words"
        );
        let captured = at(
            "function f(a,observe){var i=3;var read=()=>i;observe(read);return a[i];}",
            |op| matches!(op, Op::GetElem | Op::GetElemLocal(_)),
        );
        assert_eq!(
            captured[0].top, None,
            "environment-backed reads remain unknown"
        );
    }

    #[test]
    fn phi_false_edges_and_rhs_writes_preserve_current_value_identity() {
        for other in ["1n", "({valueOf(){return 3;}})"] {
            let source =
                format!("function f(flag,a){{var i;if(flag)i=3;else i={other};return a[i];}}");
            let reads = at(&source, |op| {
                matches!(op, Op::GetElem | Op::GetElemLocal(_))
            });
            assert_eq!(reads[0].top, None, "mixed phi {other}");
        }
        for (condition, expected) in [
            ("if(i>=10)return 0;", IntegerRange::new(0, 9)),
            ("if(i===255)return 0;", IntegerRange::new(0, 254)),
            ("if(i===0)return 0;", IntegerRange::new(1, 255)),
        ] {
            let source = format!("function f(a,x){{var i=x&255;{condition}return a[i];}}");
            let reads = at(&source, |op| {
                matches!(op, Op::GetElem | Op::GetElemLocal(_))
            });
            assert_eq!(reads[0].top, expected, "false edge {condition}");
        }
        let reads = at(
            "function f(a,n){var i=n|0;if(i<(i=-1))return a[i];return 0;}",
            |op| matches!(op, Op::GetElem | Op::GetElemLocal(_)),
        );
        assert_eq!(
            reads[0].top,
            IntegerRange::new(-1, -1),
            "RHS assignment invalidates the old LHS origin"
        );
    }

    #[test]
    fn exception_and_cleanup_landings_do_not_inherit_normal_edge_ranges() {
        for source in [
            "function f(a){var i=1;try{i={};throw 0;}catch(e){}return a[i];}",
            "function f(a){var i=1;outer:for(var k=0;k<1;k++){try{break outer;}finally{i={};}}return a[i];}",
        ] {
            let reads = at(source, |op| matches!(op, Op::GetElem | Op::GetElemLocal(_)));
            assert_eq!(reads[0].top, None, "completion root: {source}");
        }
    }

    #[test]
    fn analysis_budget_exhaustion_disables_only_range_optimization() {
        let (chunk, _) = plan("function f(a){var i=0;for(;i<256;i++)a[i]=i;return i;}");
        let cfg = Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        assert!(NumericPlan::solve(&chunk, &cfg, &stack, None, 0, 8_000_000).is_none());
        assert!(NumericPlan::solve(&chunk, &cfg, &stack, None, 1_000_000, 0).is_none());
        assert!(NumericPlan::build(&chunk, &cfg, &stack, None).is_some());
    }

    #[test]
    fn native_range_lowering_elides_conversions_and_index_guards_with_exact_results() {
        use crate::{bytecode::Tier, Completion, Engine};
        fn eval(engine: &mut Engine, source: &str) -> String {
            match engine.eval(source, false).unwrap() {
                Completion::Value(value) => value,
                Completion::Throw { name, message } => panic!("{name}: {message}"),
            }
        }
        let source = r#"
            function subject(values,input){
                var sum=0;
                for(var i=0;i<16;i++){
                    var index=(i+32)&15;
                    values[index]=i;
                    sum+=(values[index]<<2)>>>1;
                }
                return sum+':'+(input>>>0)+':'+(input|0);
            }
            var items=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],coercions=0;
            var mixed={valueOf(){coercions++;return -1;}};
        "#;
        let cases = [
            ("0", "240:0:0"),
            ("-0", "240:0:0"),
            ("-1", "240:4294967295:-1"),
            ("2147483647", "240:2147483647:2147483647"),
            ("2147483648", "240:2147483648:-2147483648"),
            ("4294967295", "240:4294967295:-1"),
            ("4294967296", "240:0:0"),
            ("NaN", "240:0:0"),
            ("Infinity", "240:0:0"),
            ("-Infinity", "240:0:0"),
            ("1.5", "240:1:1"),
            ("9007199254740992", "240:0:0"),
        ];
        for tier in [Tier::Interp, Tier::Bytecode, Tier::Jit] {
            let mut engine = Engine::new();
            engine.set_tier(tier);
            engine.set_tier_threshold(0);
            eval(&mut engine, source);
            let code = if tier == Tier::Jit {
                TEST_PROOFS.with(|counts| counts.set([0; 2]));
                let code = super::super::call_tests::optimized(&mut engine, "subject", None);
                let proofs = TEST_PROOFS.with(|counts| counts.get());
                if std::env::var("TRUST_JS_OPT_JIT_RANGES").as_deref() == Ok("0") {
                    assert_eq!(proofs, [0; 2]);
                } else {
                    assert!(proofs[0] > 0, "range proof must remove actual ToInt32 CFGs");
                    if std::env::var("TRUST_JS_OPT_JIT_HEAP_OPS").as_deref() != Ok("0") {
                        assert!(proofs[1] > 0, "range proof must remove actual index guards");
                    }
                }
                code.residency.referenced.set(0);
                Some(code)
            } else {
                None
            };
            for (argument, expected) in cases {
                assert_eq!(
                    eval(&mut engine, &format!("subject(items,{argument})")),
                    expected,
                    "{tier:?} {argument}"
                );
            }
            assert_eq!(
                eval(&mut engine, "subject(items,mixed)+'|'+coercions"),
                "240:4294967295:-1|2"
            );
            assert_eq!(
                eval(
                    &mut engine,
                    "try{subject(items,1n)}catch(error){error.name}"
                ),
                "TypeError"
            );
            if let Some(code) = code {
                assert_eq!(
                    code.residency.referenced.get(),
                    1,
                    "exact optimizing body must execute"
                );
                assert_eq!(code.residency.active.get(), 0);
            }
            assert_eq!(engine.interp.depth, 0);
            assert!(engine.interp.fn_frames.is_empty());
        }
        TEST_PROOFS.with(|counts| counts.set([0; 2]));
        super::super::tests::check(
            r#"
            function subject(){
                var calls=0,object={1:{valueOf(){calls++;return 7;}}};
                var key={toString(){calls++;return '1';}};
                var a=((0,object)[1]+=3),b=((0,object)[key]*=2);
                var c=((0,object)[1]>>>=1);
                return [a,b,c,object[1],calls].join('|');
            }
            "#,
            "10|20|10|10|2",
        );
    }
}
