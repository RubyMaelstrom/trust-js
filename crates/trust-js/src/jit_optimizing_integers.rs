//! Sparse semantic integer mirrors of canonical packed Numbers.
//!
//! ECMA-262 e28783d5, ToInt32/ToUint32 and Number shifts: the mirror is the
//! actual safe integral Number value, not its signed 32-bit bit pattern. Thus
//! >>> results zero-extend and signed results sign-extend before a CFG join.
//! Mirrors own nothing. Packed/F64 words remain eager and authoritative for
//! observers, checked fallbacks, collection and before-effect VM recovery.

use super::{effects, ranges, stack};
use crate::bytecode::{Chunk, Op};
use std::collections::VecDeque;

#[derive(Clone, Copy)]
pub(super) enum Point {
    Before,
    After,
}

/// Backward register demand, not language/GC liveness. Locations are private
/// locals followed by physical cells in the proven owning operand suffix.
pub(super) struct Demand {
    before: Vec<Vec<u64>>,
    after: Vec<Vec<u64>>,
    width: usize,
    slots: usize,
}

fn has(mask: &[u64], location: usize) -> bool {
    mask[location / 64] & (1 << (location % 64)) != 0
}

fn add(mask: &mut [u64], location: usize) {
    mask[location / 64] |= 1 << (location % 64);
}

impl Demand {
    pub(super) fn build(chunk: &Chunk, stack: &stack::StackPlan) -> Option<Self> {
        Self::solve(chunk, stack, 1_000_000, 8_000_000)
    }

    fn solve(
        chunk: &Chunk,
        stack: &stack::StackPlan,
        cells: usize,
        mut work: usize,
    ) -> Option<Self> {
        let ops = chunk.jit_ops();
        let slots = chunk.jit_frame().1;
        let width = slots.checked_add(stack.capacity)?;
        if width.checked_mul(ops.len() + 1)? > cells {
            return None;
        }
        let words = width.div_ceil(64);
        let mut before = vec![vec![0; words]; ops.len() + 1];
        let mut after = before.clone();
        let mut seeds = before.clone();
        let mut predecessors = vec![Vec::new(); ops.len() + 1];
        for (pc, op) in ops.iter().enumerate() {
            if !stack.reachable[pc] {
                continue;
            }
            let state = stack.at[pc];
            let inspected = match op {
                Op::BitAnd
                | Op::BitOr
                | Op::BitXor
                | Op::Shl
                | Op::Shr
                | Op::UShr
                | Op::EqEq
                | Op::NotEq
                | Op::StrictEq
                | Op::StrictNotEq
                | Op::Lt
                | Op::Le
                | Op::Gt
                | Op::Ge => {
                    if state.known() >= 2 {
                        add(&mut seeds[pc], slots + state.depth - 1);
                        add(&mut seeds[pc], slots + state.depth - 2);
                    }
                    0
                }
                Op::Not
                | Op::JumpIfFalse(_)
                | Op::JumpIfFalsePeek(_)
                | Op::JumpIfTruePeek(_)
                | Op::JumpIfNotNullishPeek(_) => {
                    if state.known() > 0 {
                        add(&mut seeds[pc], slots + state.depth - 1);
                    }
                    0
                }
                Op::GetElemLocal(_) => 1,
                Op::GetElem | Op::GetMethodElem => 2,
                Op::SetElemLocal(_) | Op::SetElemLocalDrop(_) => 2,
                Op::SetElem | Op::SetElemDrop => 3,
                _ => 0,
            };
            if inspected != 0 && state.known() >= inspected {
                let key_back = usize::from(matches!(
                    op,
                    Op::SetElem | Op::SetElemDrop | Op::SetElemLocal(_) | Op::SetElemLocalDrop(_)
                )) + 1;
                add(&mut seeds[pc], slots + state.depth - key_back);
            }
            let (targets, count) = stack::successors(op, pc);
            for &target in &targets[..count] {
                predecessors.get_mut(target)?.push(pc);
            }
        }
        let mut queue: VecDeque<_> = (0..ops.len()).rev().collect();
        let mut queued = vec![true; ops.len() + 1];
        while let Some(pc) = queue.pop_front() {
            queued[pc] = false;
            if !stack.reachable[pc] {
                continue;
            }
            work = work.checked_sub(width.checked_mul(2)?.checked_add(words * 3 + 1)?)?;
            let op = &ops[pc];
            let (targets, count) = stack::successors(op, pc);
            let mut outgoing = vec![0; words];
            for &target in &targets[..count] {
                for (dst, &source) in outgoing.iter_mut().zip(&before[target]) {
                    *dst |= source;
                }
            }
            let state = stack.at[pc];
            let result = stack.after[pc];
            let (pops, pushes) = chunk.jit_stack_effect(pc)?;
            let kept = state.depth.checked_sub(pops)?;
            let mut incoming = seeds[pc].clone();
            let writes = effects::checked_writes(op);
            for slot in 0..slots {
                if !writes.contains(slot as u16) && has(&outgoing, slot) {
                    add(&mut incoming, slot);
                }
            }
            for cell in state.floor..kept {
                if cell >= result.floor && has(&outgoing, slots + cell) {
                    add(&mut incoming, slots + cell);
                }
            }
            let wanted = |output: usize| output < pushes && has(&outgoing, slots + kept + output);
            let mut operand = |back: usize| {
                if back <= state.known() {
                    add(&mut incoming, slots + state.depth - back);
                }
            };
            match op {
                Op::LoadLocal(slot) if wanted(0) => add(&mut incoming, usize::from(*slot)),
                Op::StoreLocal(slot) if has(&outgoing, usize::from(*slot)) => operand(1),
                Op::Dup if wanted(0) || wanted(1) => operand(1),
                Op::Dup2 => {
                    if wanted(0) || wanted(2) {
                        operand(2);
                    }
                    if wanted(1) || wanted(3) {
                        operand(1);
                    }
                }
                Op::ToPropKey => {
                    if wanted(0) {
                        operand(2);
                    }
                    if wanted(1) {
                        operand(1);
                    }
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
                | Op::SetElemLocal(_)
                    if wanted(0) =>
                {
                    operand(1)
                }
                _ => {}
            }
            after[pc] = outgoing;
            if before[pc] != incoming {
                before[pc] = incoming;
                for &previous in &predecessors[pc] {
                    if !queued[previous] {
                        queued[previous] = true;
                        queue.push_back(previous);
                    }
                }
            }
        }
        Some(Self {
            before,
            after,
            width,
            slots,
        })
    }

    fn mask(&self, pc: usize, point: Point) -> &[u64] {
        match point {
            Point::Before => &self.before[pc],
            Point::After => &self.after[pc],
        }
    }
}

/// Only demanded exact integer facts are retained after the range fixed point.
/// Unknown definitions have no fact, and must explicitly define a dummy mirror.
pub(super) struct Facts {
    before: Vec<Vec<(usize, ranges::IntegerRange)>>,
    after: Vec<Vec<(usize, ranges::IntegerRange)>>,
    pub(super) locals: Vec<bool>,
    pub(super) operands: Vec<bool>,
}

impl Facts {
    pub(super) fn new(demand: &Demand) -> Self {
        Self {
            before: vec![Vec::new(); demand.before.len()],
            after: vec![Vec::new(); demand.after.len()],
            locals: vec![false; demand.slots],
            operands: vec![false; demand.width - demand.slots],
        }
    }

    pub(super) fn record(
        &mut self,
        demand: &Demand,
        pc: usize,
        point: Point,
        ranges: &[Option<ranges::IntegerRange>],
    ) -> usize {
        let destination = match point {
            Point::Before => &mut self.before[pc],
            Point::After => &mut self.after[pc],
        };
        destination.clear();
        let mask = demand.mask(pc, point);
        let mut work = mask.len();
        for (word, &bits) in mask.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let location = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                work += 1;
                if let Some(range) = ranges[location] {
                    destination.push((location, range));
                }
            }
        }
        work
    }

    pub(super) fn finish(&mut self, slots: usize) {
        // Intermediate solver iterations may have optimistic facts. Declare
        // mirrors only from the final settled per-PC snapshots.
        for &(location, _) in self.before.iter().chain(&self.after).flatten() {
            if location < slots {
                self.locals[location] = true;
            } else {
                self.operands[location - slots] = true;
            }
        }
    }

    pub(super) fn at(
        &self,
        pc: usize,
        point: Point,
        location: usize,
    ) -> Option<ranges::IntegerRange> {
        let values = match point {
            Point::Before => self.before.get(pc)?,
            Point::After => self.after.get(pc)?,
        };
        values
            .binary_search_by_key(&location, |&(location, _)| location)
            .ok()
            .map(|index| values[index].1)
    }

    /// Exact selected helper-bypass sites for canonical publication planning.
    /// Generic equality/unary fallbacks retain their original barriers, including
    /// when INT_SSA is disabled or range/demand analysis exhausts its budget.
    pub(super) fn scalar_sites(&self, chunk: &Chunk, stack: &stack::StackPlan) -> Vec<bool> {
        let slots = chunk.jit_frame().1;
        chunk
            .jit_ops()
            .iter()
            .enumerate()
            .map(|(pc, op)| {
                let inputs = match op {
                    Op::EqEq | Op::NotEq => 2,
                    Op::Not => 1,
                    _ => return false,
                };
                let state = stack.at[pc];
                state.known() >= inputs
                    && (1..=inputs).all(|back| {
                        self.at(pc, Point::Before, slots + state.depth - back)
                            .is_some()
                    })
            })
            .collect()
    }
}

#[cfg(test)]
thread_local! {
    /// Emitted mirror ToInt32 reads / index reads / forwarded copies / integer definitions.
    pub(super) static TEST_MIRRORS: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
}

#[cfg(test)]
pub(super) fn record(kind: usize) {
    TEST_MIRRORS.with(|counts| {
        let mut values = counts.get();
        values[kind] += 1;
        counts.set(values);
    });
}

use super::*;

impl Lowering<'_, '_> {
    fn integer_fact(&self, point: Point, location: usize) -> Option<ranges::IntegerRange> {
        self.numeric_plan?
            .integers
            .as_ref()?
            .at(self.pc, point, location)
    }

    pub(super) fn stack_integer(&mut self, back: usize) -> Option<Value> {
        assert!(back > 0 && back <= self.stack_state.known());
        let cell = self.stack_state.depth - back;
        self.integer_fact(Point::Before, self.locals.len() + cell)?;
        Some(self.b.use_var(self.stack_integers[cell]?))
    }

    pub(super) fn local_integer(&mut self, slot: u16) -> Option<Value> {
        self.integer_fact(Point::Before, usize::from(slot))?;
        Some(self.b.use_var(self.local_integers[usize::from(slot)]?))
    }

    pub(super) fn integer_stack_output(&self, cell: usize) -> bool {
        self.stack_integers[cell].is_some()
            && self
                .integer_fact(Point::After, self.locals.len() + cell)
                .is_some()
    }

    fn integer_value(
        &mut self,
        word: Value,
        integer: Option<Value>,
        range: Option<ranges::IntegerRange>,
    ) -> Value {
        if let Some(range) = range {
            #[cfg(test)]
            record(3);
            if let Some(integer) = integer {
                #[cfg(test)]
                record(2);
                return integer;
            }
            if range.min == range.max {
                return self.b.ins().iconst(types::I64, range.min);
            }
            // The range proof is finite, integral, safe and excludes -0.
            // Conversion therefore preserves its exact mathematical value.
            let number = self.b.ins().bitcast(types::F64, MemFlags::new(), word);
            return self.b.ins().fcvt_to_sint_sat(types::I64, number);
        }
        // Every unknown definition/reload invalidates the previous mirror.
        // Consumers require a settled proof at their exact incoming PC, so
        // this dummy may merge at an unused phi but can never be consumed.
        self.b.ins().iconst(types::I64, 0)
    }

    pub(super) fn define_stack_integer(
        &mut self,
        cell: usize,
        word: Value,
        integer: Option<Value>,
        point: Point,
    ) {
        if let Some(var) = self.stack_integers[cell] {
            let range = self.integer_fact(point, self.locals.len() + cell);
            let integer = self.integer_value(word, integer, range);
            self.b.def_var(var, integer);
        }
    }

    pub(super) fn define_local_integer(
        &mut self,
        slot: u16,
        word: Value,
        integer: Option<Value>,
        point: Point,
    ) {
        if let Some(var) = self.local_integers[usize::from(slot)] {
            let range = self.integer_fact(point, usize::from(slot));
            let integer = self.integer_value(word, integer, range);
            self.b.def_var(var, integer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demand_budget_exhaustion_retains_the_existing_range_plan() {
        let statements =
            crate::parser::parse_script("function subject(a){var x=a>>>0;return(x^3)&7;}", false)
                .ok()
                .expect("fixture parses");
        let crate::ast::Stmt::FuncDecl(function) = &statements[0] else {
            panic!("function fixture")
        };
        let chunk = crate::bytecode::compile(function).unwrap();
        let cfg = crate::jit_ir::Cfg::build(&chunk).unwrap();
        let stack = stack::StackPlan::build(&chunk, &cfg).unwrap();
        assert!(Demand::solve(&chunk, &stack, 0, 8_000_000).is_none());
        assert!(Demand::solve(&chunk, &stack, 1_000_000, 0).is_none());
        assert!(ranges::NumericPlan::build(&chunk, &cfg, &stack, None).is_some());
        assert!(Demand::build(&chunk, &stack).is_some());
    }
}
