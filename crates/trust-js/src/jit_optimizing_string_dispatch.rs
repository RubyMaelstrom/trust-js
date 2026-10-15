//! Ordered literal-string case dispatch over the original immutable bytecode graph.
//!
//! ECMA-262 e28783d5 IsStrictlyEqual / SameValueNonNumber / CaseBlockEvaluation:
//! compare in source order, select the first match, then execute the original case body
//! and its fallthrough. Only certified pure comparisons are replaced. No JS owner, author
//! effect, poll, handler/cleanup landing, or observer is removed from a native entry path.

use super::*;
use crate::value::{PackedValue, Value as JsValue, PACK_EMPTY};

// Bounds each non-observing leaf and switch construction; longer chains keep ordered groups.
const MAX_CASES: usize = 128;
const MIN_CASES: usize = 2;

/// Provisional compiler-work units for every original operation, including cold cuts.
/// Price the five dense scalar arrays, original edge/root scans and bounded chain
/// certification at estimate, final checkpoint and preparation. An attempted plan
/// returning None still performed this work; selected cases never shrink its input.
pub(super) fn analysis_work(ops: usize) -> Option<u64> {
    u64::try_from(ops.checked_add(1)?).ok()?.checked_mul(36)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Site {
    start: usize,
    slot: u16,
    first_load: usize,
    count: usize,
}

impl Site {
    pub(super) fn end(self) -> usize {
        self.start + self.count * 5
    }

    pub(super) fn branch(self, index: usize) -> usize {
        self.start + index * 5 + 3
    }

    pub(super) fn count(self) -> usize {
        self.count
    }

    pub(super) fn native_identity(self) -> (usize, u16, usize, usize) {
        (self.start, self.slot, self.first_load, self.count)
    }

    /// The same conservative switch/helper/publication price used before IR emission.
    /// This is additional compiler work, not credit for a removed case body or helper.
    fn lowering_work(self, stack: &stack::StackPlan) -> Option<usize> {
        let floor = stack.at[self.start].floor;
        let publications = (0..self.count)
            .try_fold(0usize, |total, index| {
                total.checked_add(stack.forget_to[self.branch(index)].saturating_sub(floor))
            })?
            .checked_add(stack.forget_to[self.branch(self.count - 1)].saturating_sub(floor))?;
        self.count
            .checked_add(1)?
            .checked_mul(20)?
            .checked_add(publications)?
            .checked_add(64)
    }
}

#[derive(Clone, Copy)]
enum Incoming {
    None,
    One(usize),
    Many,
}

pub(super) struct Plan {
    sites: Box<[Option<Site>]>,
    interior: Box<[bool]>,
}

impl Plan {
    pub(super) fn build(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        regions: Option<&observed_regions::Plan>,
        native_reachable: Option<&[bool]>,
        deopt: Option<usize>,
        literals: &string_literals::Plan,
    ) -> Option<Self> {
        #[cfg(test)]
        if !TEST_ENABLED.with(std::cell::Cell::get) {
            return None;
        }
        let ops = chunk.jit_ops();
        // One bounded full scan, rather than scanning the full graph per candidate chain.
        let mut incoming = vec![Incoming::None; ops.len() + 1];
        let mut poll = vec![false; ops.len() + 1];
        for (pc, op) in ops.iter().enumerate() {
            if let Some(target) = crate::jit_ir::jump_target(op) {
                let entry = incoming.get_mut(target)?;
                *entry = match *entry {
                    Incoming::None => Incoming::One(pc),
                    _ => Incoming::Many,
                };
                if target <= pc {
                    poll[target] = true;
                }
            }
        }
        let mut roots = vec![false; ops.len() + 1];
        for root in cfg.handler_roots().iter().chain(cfg.resume_roots()) {
            roots[cfg.blocks()[root.target.0 as usize].start] = true;
        }
        let native = |pc: usize| {
            stack.reachable.get(pc).copied().unwrap_or(false)
                && native_reachable.is_none_or(|reachable| reachable.get(pc) == Some(&true))
                && regions
                    .is_none_or(|plan| plan.emission(pc) == observed_regions::Emission::Native)
        };
        let case = |pc: usize| {
            let (slot, _, negate, mismatch) = literals.site(pc)?.identity();
            if negate || mismatch != pc.checked_add(5)? || pc + 4 >= ops.len() {
                return None;
            }
            let Op::Jump(target) = ops[pc + 4] else {
                return None;
            };
            // The comparison consumes only four operations. Its fifth, original case
            // Jump remains an independently labeled operation and can be a before-effect
            // cold cut. Never manufacture a label for an omitted interior operation.
            let jump_pc = pc + 4;
            let retained_jump = stack.reachable[jump_pc]
                && native_reachable.is_none_or(|reachable| reachable.get(jump_pc) == Some(&true))
                && regions.is_none_or(|plan| {
                    plan.emission(jump_pc) != observed_regions::Emission::OmittedInterior
                });
            ((pc..pc + 4).all(&native) && retained_jump).then_some((slot, target as usize))
        };
        let mut sites = vec![None; ops.len()];
        let mut interior = vec![false; ops.len()];
        let mut groups = 0;
        let mut pc = 0;
        while pc + 4 < ops.len() {
            let Some((slot, _)) = case(pc) else {
                pc += 1;
                continue;
            };
            let mut count = 1;
            while count < MAX_CASES {
                let at = pc + count * 5;
                if at + 4 >= ops.len() || case(at).is_none_or(|(next_slot, _)| next_slot != slot) {
                    break;
                }
                count += 1;
            }
            let site = Site {
                start: pc,
                slot,
                first_load: pc + usize::from(!matches!(ops[pc], Op::LoadLocal(_))),
                count,
            };
            let end = site.end();
            let state = stack.at[pc];
            let valid = count >= MIN_CASES
                && !deopt.is_some_and(|at| pc < at && at < end)
                && (pc + 1..end).all(|at| !poll[at] && !roots[at])
                && (0..count).all(|index| {
                    let at = pc + index * 5;
                    let Some((_, target)) = case(at) else {
                        return false;
                    };
                    let Some(compare) = cfg.block_at(at) else {
                        return false;
                    };
                    let Some(jump) = cfg.block_at(at + 4) else {
                        return false;
                    };
                    if (pc..end).contains(&target)
                        || stack.at[at] != state
                        || stack.after[at + 3] != state
                        || stack.at[at + 4] != state
                        || stack.after[at + 4] != state
                        || cfg.blocks()[jump.0 as usize].start != at + 4
                        || cfg.blocks()[jump.0 as usize].predecessors.as_slice() != [compare]
                        || !matches!(incoming[at + 4], Incoming::None)
                    {
                        return false;
                    }
                    index == 0
                        || (cfg.blocks()[compare.0 as usize].start == at
                            && cfg.blocks()[compare.0 as usize].predecessors.as_slice()
                                == [cfg.block_at(at - 5).expect("previous certified comparison")]
                            && matches!(incoming[at], Incoming::One(source) if source == at - 2))
                });
            if valid {
                sites[pc] = Some(site);
                for index in 0..count {
                    let at = pc + index * 5;
                    interior[at..at + 4].fill(true);
                }
                interior[pc] = false;
                groups += 1;
            }
            // A rejected chain retains every original/fused comparison, including arbitrary
            // interior entry. Do not attempt suffixes through its rejected certificates.
            pc = end;
        }
        (groups != 0).then(|| Self {
            sites: sites.into_boxed_slice(),
            interior: interior.into_boxed_slice(),
        })
    }

    pub(super) fn site(&self, pc: usize) -> Option<Site> {
        self.sites[pc]
    }

    pub(super) fn interior(&self, pc: usize) -> bool {
        self.interior[pc]
    }

    pub(super) fn selected_groups(&self) -> usize {
        self.sites.iter().filter(|site| site.is_some()).count()
    }

    pub(super) fn selected_cases(&self) -> usize {
        self.sites.iter().flatten().map(|site| site.count).sum()
    }

    pub(super) fn lowering_work(&self, stack: &stack::StackPlan) -> Option<u64> {
        self.sites.iter().flatten().try_fold(0u64, |total, site| {
            total.checked_add(u64::try_from(site.lowering_work(stack)?).ok()?)
        })
    }

    pub(super) fn payload_bytes(&self) -> usize {
        std::mem::size_of_val(&*self.sites) + std::mem::size_of_val(&*self.interior)
    }
}

/// `word` is lent by a live canonical or SSA owner; the exact Chunk's immutable operations
/// and constant addresses are pinned through preparation, code publication and execution.
/// No owner is cloned/dropped, no cached text or owner count is retained, and no observer,
/// allocation, collection, poll, materialization or USV conversion occurs here.
unsafe extern "C" fn first_match(word: u64, chunk: *const Chunk, start: u32, count: u32) -> u64 {
    #[cfg(test)]
    {
        TEST_LEAF_ENTRIES.with(|entries| entries.set(entries.get().saturating_add(1)));
        if word & 0xffff_0000_0000_0000 == crate::value::PACK_STR {
            // Test-only current count, using the same validated String Header as owners.
            // Distinguishes borrowing a sole canonical view owner from temporary clones.
            let strong = unsafe {
                &*((word & 0x0000_ffff_ffff_ffff) as usize as *const std::cell::Cell<usize>)
            }
            .get();
            // PackedValue is repr(transparent) over Cell<u64>. This read-only test
            // borrow creates no owner and performs no Cell write through the reference.
            // Inspect the actual native input, rather than infer a view from slice syntax.
            let packed = unsafe { &*std::ptr::from_ref(&word).cast::<PackedValue>() };
            if packed.with_string(|string| string.view_owner().is_some()) == Some(true) {
                TEST_REAL_VIEWS.with(|views| views.set(views.get() + 1));
            }
            TEST_STRING_OWNERS.with(|counts| {
                let mut next = counts.get();
                let index = usize::from(strong != 1);
                next[index] = next[index].saturating_add(1);
                counts.set(next);
            });
        }
    }
    unsafe {
        PackedValue::with_raw_string_text(word, |text| {
            let chunk = &*chunk;
            let ops = chunk.jit_ops();
            for index in 0..count as usize {
                let at = start as usize + index * 5;
                let literal = match (&ops[at], &ops[at + 1]) {
                    (_, Op::Const(literal)) | (Op::Const(literal), _) => *literal,
                    _ => unreachable!("immutable certified literal case"),
                };
                let JsValue::Str(literal) = &*chunk.jit_const_ptr(literal) else {
                    unreachable!("immutable certified String constant")
                };
                if text == literal.as_str() {
                    return index as u64 + 1;
                }
            }
            0
        })
        .unwrap_or(0)
    }
}

impl Lowering<'_, '_> {
    pub(super) fn string_case_dispatch(
        &mut self,
        pc: usize,
        site: Site,
        instruction_limit: usize,
        block_limit: usize,
    ) -> Result<(), CompileError> {
        // Dense ordinal switches are bounded before allocating their cases. Price every
        // outcome's full possible suffix publication and diagnostic count conservatively.
        let state = self.stack_plan.at[pc];
        let outcomes = site
            .count
            .checked_add(1)
            .ok_or(CompileError::LoweringBudget)?;
        let work = site
            .lowering_work(self.stack_plan)
            .ok_or(CompileError::LoweringBudget)?;
        let blocks = outcomes
            .checked_add(8)
            .ok_or(CompileError::LoweringBudget)?;
        if self
            .b
            .func
            .dfg
            .num_insts()
            .checked_add(work)
            .is_none_or(|n| n > instruction_limit)
            || self
                .b
                .func
                .dfg
                .num_blocks()
                .checked_add(blocks)
                .is_none_or(|n| n > block_limit)
        {
            return Err(CompileError::LoweringBudget);
        }
        let word = self.local(site.slot);
        // This is exactly the original first LoadLocal proof, including reversed Const/load
        // order. No sample-derived String fact is introduced for the discriminant.
        let initialized = self
            .value_plan
            .is_some_and(|plan| plan.at[site.first_load].local.initialized());
        let empty = if initialized {
            None
        } else {
            #[cfg(test)]
            TEST_EMPTY_GUARDS.with(|guards| guards.set(guards.get() + 1));
            let fast = self.b.create_block();
            let empty = self.b.create_block();
            let ready = self
                .b
                .ins()
                .icmp_imm_s(IntCC::NotEqual, word, PACK_EMPTY as i64);
            self.b.ins().brif(ready, fast, &[], empty, &[]);
            self.b.switch_to_block(fast);
            Some(empty)
        };
        let chunk = self
            .b
            .ins()
            .iconst(types::I64, self.chunk as *const Chunk as usize as i64);
        let start = self.b.ins().iconst(types::I32, pc as i64);
        let count = self.b.ins().iconst(types::I32, site.count as i64);
        let callee = self
            .b
            .ins()
            .iconst(types::I64, first_match as *const () as usize as i64);
        if let Some(record) = self.diagnostics {
            record.emitted_helpers.set(record.emitted_helpers.get() + 1);
            record.emitted_string_dispatch();
        }
        let call = self.b.ins().call_indirect(
            self.string_dispatch_sig
                .expect("selected String dispatch signature"),
            callee,
            &[word, chunk, start, count],
        );
        let result = self.b.inst_results(call)[0];
        // One successful ordered dispatch, never a count for each skipped comparison.
        self.count_native_site_success(pc, "literal_string_case_dispatch");
        let default = self.b.create_block();
        let mut switch = Switch::new();
        let cases: Vec<_> = (0..site.count)
            .map(|index| {
                let block = self.b.create_block();
                switch.set_entry(index as u128 + 1, block);
                (index, block)
            })
            .collect();
        switch.emit(&mut self.b, result, default);
        for (index, block) in cases {
            self.b.switch_to_block(block);
            self.pc = site.branch(index);
            self.stack_state = state;
            self.dirty = &self.frame_plan.dirty_at[self.pc];
            self.finish_normal();
            let target = self.loop_exit(self.pc_block(self.pc + 1));
            self.b.ins().jump(target, &[]);
        }
        self.b.switch_to_block(default);
        self.pc = site.branch(site.count - 1);
        self.stack_state = state;
        self.dirty = &self.frame_plan.dirty_at[self.pc];
        self.finish_normal();
        let target = self.loop_exit(self.pc_block(site.end()));
        self.b.ins().jump(target, &[]);
        if let Some(empty) = empty {
            self.b.switch_to_block(empty);
            self.b.set_cold_block(empty);
            self.pc = pc;
            self.stack_state = state;
            self.dirty = &self.frame_plan.dirty_at[pc];
            self.deopt_before(pc, self.error);
        }
        #[cfg(test)]
        TEST_SELECTED.with(|sites| sites.borrow_mut().push((pc, site.count)));
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static TEST_LEAF_ENTRIES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    pub(super) static TEST_SELECTED: std::cell::RefCell<Vec<(usize, usize)>> = const { std::cell::RefCell::new(Vec::new()) };
    pub(super) static TEST_STRING_OWNERS: std::cell::Cell<[u64; 2]> = const { std::cell::Cell::new([0; 2]) };
    pub(super) static TEST_EMPTY_GUARDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static TEST_REAL_VIEWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
#[path = "jit_optimizing_string_dispatch_tests.rs"]
mod tests;
