//! Closed SSA loan windows for private local copies and call arguments.
//!
//! ECMA-262 e28783d5 GetValue/PutValue/ArgumentListEvaluation: each original TDZ
//! read and argument order survive. Nothing observes a temporary clone before the
//! first effect. Canonical/SSA local homes keep their owners throughout the window.
//! ALL guards precede one atomic owner transaction; a miss resumes before the
//! original first load with its unchanged owning stack and dirty SSA homes.

use super::*;
use crate::value::{PACK_BIGINT, PACK_EMPTY, PACK_OBJ, PACK_STR, PACK_SYM};
use std::cell::Cell;

const MAX_LOANS: usize = 8;
const PAYLOAD: u64 = 0x0000_ffff_ffff_ffff;
const SUCCESS: u64 = 1;
const RETAINS_SHIFT: u32 = 8;
const RELEASES_SHIFT: u32 = 16;

pub(super) fn requested() -> bool {
    std::env::var("TRUST_JS_OPT_JIT_BORROWED_OPERANDS").as_deref() == Ok("1")
}

/// Three bounded original-graph attempts (estimate/checkpoint/preparation), each
/// with three scalar arrays, original edge/root and linear run scans, plus
/// numeric opportunity retain/fold scans. This is a conservative work coefficient,
/// not a measured compiler-instruction count.
pub(super) fn analysis_work(ops: usize) -> Option<u64> {
    if !requested() {
        return Some(0);
    }
    u64::try_from(ops.checked_add(1)?).ok()?.checked_mul(24)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Site {
    start: usize,
    end: usize,
    slots: [u16; MAX_LOANS],
    count: usize,
    initialized: u8,
    copyable: u8,
    destination: Option<u16>,
    destination_copyable: bool,
}

pub(super) struct Plan {
    sites: Box<[Option<Site>]>,
    interior: Box<[bool]>,
    minimum_path_work: u64,
}

impl Site {
    /// Original semantic endpoint; a call is still emitted there by the head.
    pub(super) fn native_end(self) -> usize {
        self.end
    }

    pub(super) fn native_matches(self, ops: &[Op], head: usize) -> bool {
        self.start == head
            && (1..=MAX_LOANS).contains(&self.count)
            && self.end < ops.len()
            && self.end.checked_sub(head) == Some(self.count)
            && (0..self.count).all(|index|
                matches!(ops[head + index], Op::LoadLocal(slot) if slot == self.slots[index]))
            && match self.destination {
                Some(slot) => self.count == 1 && matches!(ops[self.end], Op::StoreLocal(actual) if actual == slot),
                None => matches!(ops[self.end], Op::CallWithThis(..)),
            }
    }
}

impl Plan {
    #[allow(clippy::too_many_arguments)] // Exact original graph, selection, entry and value certificates.
    pub(super) fn build(
        chunk: &Chunk,
        cfg: &Cfg,
        stack: &stack::StackPlan,
        regions: Option<&observed_regions::Plan>,
        reachable: Option<&[bool]>,
        deopt: Option<usize>,
        origin: NativeEntryKind,
        facts: Option<&value_facts::ValuePlan>,
    ) -> Option<Self> {
        if !requested()
            || origin != NativeEntryKind::FreshFrame
            || chunk.prepared_entry
            || chunk.jit_is_resumable()
            || !bytecode::native_deopt::supported(chunk)
        {
            return None;
        }
        let ops = chunk.jit_ops();
        let mut blocked = vec![false; ops.len() + 1];
        for op in ops {
            if let Some(target) = crate::jit_ir::jump_target(op) {
                *blocked.get_mut(target)? = true;
            }
        }
        for root in cfg.handler_roots().iter().chain(cfg.resume_roots()) {
            blocked[cfg.blocks()[root.target.0 as usize].start] = true;
        }
        let native = |pc: usize| {
            stack.reachable.get(pc) == Some(&true)
                && reachable.is_none_or(|r| r.get(pc) == Some(&true))
                && regions.is_none_or(|p| p.emission(pc) == observed_regions::Emission::Native)
        };
        let mut sites = vec![None; ops.len()];
        let mut interior = vec![false; ops.len()];
        let mut pc = 0;
        let mut selected = 0;
        while pc < ops.len() {
            if !matches!(ops[pc], Op::LoadLocal(_)) {
                pc += 1;
                continue;
            }
            let run = pc;
            while pc < ops.len() && matches!(ops[pc], Op::LoadLocal(_)) {
                pc += 1;
            }
            let end = pc;
            let destination = match ops.get(end) {
                Some(Op::StoreLocal(s)) => Some(*s),
                _ => None,
            };
            let count = if destination.is_some() {
                1
            } else if matches!(ops.get(end), Some(Op::CallWithThis(..))) {
                (end - run).min(MAX_LOANS)
            } else {
                continue;
            };
            let start = end - count;
            let Some(block) = cfg.block_at(start) else {
                continue;
            };
            let state = stack.at[start];
            if cfg.block_at(end) != Some(block)
                || !(start..=end).all(native)
                || (start + 1..=end).any(|at| blocked[at] || deopt == Some(at))
                || (start..=end)
                    .any(|at| crate::stack_trace::is_inline_position(chunk.position(at)))
                || stack.at[end].depth != state.depth.checked_add(count)?
                || stack.at[end].floor != state.floor
                || (destination.is_some() && stack.after[end] != state)
            {
                continue;
            }
            let mut site = Site {
                start,
                end,
                slots: [0; MAX_LOANS],
                count,
                initialized: 0,
                copyable: 0,
                destination,
                destination_copyable: destination
                    .is_some_and(|_| facts.is_some_and(|f| f.at[end].local.is_copyable())),
            };
            for index in 0..count {
                let Op::LoadLocal(slot) = ops[start + index] else {
                    unreachable!("certified local run")
                };
                if usize::from(slot) >= chunk.jit_frame().1 {
                    return None;
                }
                site.slots[index] = slot;
                if let Some(f) = facts {
                    site.initialized |= u8::from(f.at[start + index].local.initialized()) << index;
                    site.copyable |= u8::from(f.at[start + index].local.is_copyable()) << index;
                }
            }
            if site.copyable.count_ones() as usize == count
                && (destination.is_none() || site.destination_copyable)
            {
                continue;
            }
            sites[start] = Some(site);
            interior[start + 1..=end].fill(true);
            selected += 1;
        }
        // New small-body abrupt minima use the bounded quadratic CFG solver.
        // Price three actual policy/preparation attempts, not only native IR.
        let minimum_path_work = if ops.len() <= 256 {
            u64::try_from(cfg.blocks().len())
                .ok()?
                .checked_pow(2)?
                .checked_add(u64::try_from(ops.len()).ok()?)?
                .checked_mul(3)?
        } else {
            0
        };
        (selected != 0).then(|| Self {
            sites: sites.into_boxed_slice(),
            interior: interior.into_boxed_slice(),
            minimum_path_work,
        })
    }

    pub(super) fn site(&self, pc: usize) -> Option<Site> {
        self.sites.get(pc).copied().flatten()
    }
    pub(super) fn interior(&self, pc: usize) -> bool {
        self.interior.get(pc) == Some(&true)
    }
    fn covers(&self, pc: usize) -> bool {
        self.site(pc).is_some() || self.interior(pc)
    }

    /// No new benefit credit. The original per-operation owner checks have been
    /// replaced by the boundary transaction, so neither numeric plan may claim them.
    pub(super) fn remove_numeric_credit(&self, plan: &mut specialization::Plan) {
        plan.opportunities
            .retain(|opportunity| !self.covers(opportunity.pc));
        let removed = plan.opportunities.iter().fold(0u32, |total, opportunity| {
            total.saturating_add(opportunity.removed_tests)
        });
        plan.saved_tests =
            removed.saturating_sub((plan.inputs.len() + plan.properties.len()) as u32);
    }

    pub(super) fn lowering_work(&self, chunk: &Chunk, stack: &stack::StackPlan) -> Option<u64> {
        self.sites
            .iter()
            .flatten()
            .try_fold(self.minimum_path_work, |total, site| {
                // Conservative full-frame publication upper bound on before-start
                // failure, plus actual bounded word/guard/scratch/helper installation.
                let publication = chunk
                    .jit_frame()
                    .1
                    .checked_add(stack.at[site.start].known())?
                    .checked_mul(2)?;
                let work = site
                    .count
                    .checked_add(1)?
                    .checked_mul(24)?
                    .checked_add(publication)?
                    .checked_add(64)?;
                total.checked_add(u64::try_from(work).ok()?)
            })
    }
}

/// # Safety
/// Caller validates the same live packed-owner ABI as the existing scalar leaf.
/// The words are borrowed canonical/SSA owners, not arbitrary pointer integers;
/// storage stays initialized and immutable through this non-observing call.
/// Count Cells are single-threaded. No allocation, Rust owner clone/drop, GC,
/// callback, canonical frame write or exception can occur here. All count/layout
/// failures precede EVERY mutation, including aliases and mixed retain/release.
unsafe fn transaction<const COUNTED: bool>(words: *const u64, len: u32, retains: u32) -> u64 {
    if len == 0 || len as usize > MAX_LOANS || retains >> len != 0 || words.is_null() {
        return 0;
    }
    let mut owners = [0usize; MAX_LOANS];
    let mut additions = [0usize; MAX_LOANS];
    let mut subtractions = [0usize; MAX_LOANS];
    let mut after = [0usize; MAX_LOANS];
    let mut unique = 0;
    let mut retained = 0u64;
    let mut released = 0u64;
    for index in 0..len as usize {
        let word = unsafe { *words.add(index) };
        let tag = word & !PAYLOAD;
        if !matches!(tag, PACK_OBJ | PACK_STR | PACK_SYM | PACK_BIGINT) {
            if word <= (PACK_BOOL | 1) || (word & 0x7fff_0000_0000_0000) <= PACK_CANON_NAN {
                continue;
            }
            return 0;
        }
        let owner = (word & PAYLOAD) as usize;
        if owner == 0 || !owner.is_multiple_of(std::mem::align_of::<Cell<usize>>()) {
            return 0;
        }
        let at = owners[..unique]
            .iter()
            .position(|&other| other == owner)
            .unwrap_or_else(|| {
                let at = unique;
                owners[at] = owner;
                unique += 1;
                at
            });
        if retains & (1 << index) != 0 {
            additions[at] += 1;
            retained += 1;
        } else {
            subtractions[at] += 1;
            released += 1;
        }
    }
    for index in 0..unique {
        let before = unsafe { &*(owners[index] as *const Cell<usize>) }.get();
        let Some(sum) = before.checked_add(additions[index]) else {
            return 0;
        };
        if before == 0 || sum <= subtractions[index] {
            return 0;
        }
        after[index] = sum - subtractions[index];
    }
    // Commit once per distinct allocation. No subsequent failure or observer.
    for index in 0..unique {
        unsafe { &*(owners[index] as *const Cell<usize>) }.set(after[index]);
    }
    #[cfg(test)]
    TEST_TRANSACTIONS.with(|calls| calls.set(calls.get() + 1));
    if COUNTED {
        SUCCESS | (retained << RETAINS_SHIFT) | (released << RELEASES_SHIFT)
    } else {
        SUCCESS
    }
}

unsafe extern "C" fn uncounted(words: *const u64, len: u32, retains: u32) -> u64 {
    unsafe { transaction::<false>(words, len, retains) }
}
unsafe extern "C" fn counted(words: *const u64, len: u32, retains: u32) -> u64 {
    unsafe { transaction::<true>(words, len, retains) }
}

impl Lowering<'_, '_> {
    fn borrowed_transaction(&mut self, changes: &[(Value, bool)], miss: Block) -> bool {
        if changes.is_empty() {
            return true;
        }
        if !self.owners_valid {
            self.b.ins().jump(miss, &[]);
            return false;
        }
        assert!(changes.len() <= MAX_LOANS);
        let signature = *self.borrowed_sig.get_or_insert_with(|| {
            let mut sig = Signature::new(self.b.func.signature.call_conv);
            sig.params.extend([
                AbiParam::new(types::I64),
                AbiParam::new(types::I32),
                AbiParam::new(types::I32),
            ]);
            sig.returns.push(AbiParam::new(types::I64));
            self.b.func.import_signature(sig)
        });
        let slot = *self.borrowed_words.get_or_insert_with(|| {
            self.b
                .create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    (MAX_LOANS * 8) as u32,
                    3,
                ))
        });
        let words = self.b.ins().stack_addr(types::I64, slot, 0);
        let mut mask = 0u32;
        for (index, &(word, retain)) in changes.iter().enumerate() {
            self.store(word, words, (index * 8) as i32);
            mask |= u32::from(retain) << index;
        }
        let full = self
            .diagnostics
            .is_some_and(|record| record.runtime_counters);
        let helper = if full { counted } else { uncounted };
        let callee = self
            .b
            .ins()
            .iconst(types::I64, helper as *const () as usize as i64);
        let len = self.b.ins().iconst(types::I32, changes.len() as i64);
        let mask = self.b.ins().iconst(types::I32, i64::from(mask));
        if let Some(record) = self.diagnostics {
            record.emitted_helpers.set(record.emitted_helpers.get() + 1);
        }
        let call = self
            .b
            .ins()
            .call_indirect(signature, callee, &[words, len, mask]);
        let status = self.b.inst_results(call)[0];
        let pass = self.b.ins().band_imm_s(status, SUCCESS as i64);
        self.count_heap_guard();
        let committed = self.b.create_block();
        self.b.ins().brif(pass, committed, &[], miss, &[]);
        self.b.switch_to_block(committed);
        if full {
            for (counter, shift) in [
                (diagnostics::RETAIN, RETAINS_SHIFT),
                (diagnostics::RELEASE, RELEASES_SHIFT),
            ] {
                let amount = self.b.ins().ushr_imm_u(status, i64::from(shift));
                let amount = self.b.ins().band_imm_s(amount, 0xff);
                self.owner_leaf_count(counter, amount);
            }
        }
        true
    }

    pub(super) fn borrowed_operands(
        &mut self,
        site: Site,
        instruction_limit: usize,
        block_limit: usize,
    ) -> Result<(), CompileError> {
        // Provisional local reservation; unchanged call lowering and the exact
        // before-start publication still pass the existing per-op/final DFG cap.
        let upper = site
            .count
            .checked_add(1)
            .and_then(|n| n.checked_mul(24))
            .and_then(|n| n.checked_add(64))
            .ok_or(CompileError::LoweringBudget)?;
        if self
            .b
            .func
            .dfg
            .num_insts()
            .checked_add(upper)
            .is_none_or(|n| n > instruction_limit)
            || self
                .b
                .func
                .dfg
                .num_blocks()
                .checked_add(site.count * 2 + 8)
                .is_none_or(|n| n > block_limit)
        {
            return Err(CompileError::LoweringBudget);
        }
        let incoming = self.stack_state;
        let miss = self.b.create_block();
        let mut words = Vec::with_capacity(site.count);
        let mut integers = Vec::with_capacity(site.count);
        let mut changes = Vec::with_capacity(MAX_LOANS);
        for index in 0..site.count {
            self.pc = site.start + index;
            let word = self.local(site.slots[index]);
            words.push(word);
            integers.push(self.local_integer(site.slots[index]));
            if site.initialized & (1 << index) == 0 {
                let valid = self
                    .b
                    .ins()
                    .icmp_imm_s(IntCC::NotEqual, word, PACK_EMPTY as i64);
                let next = self.b.create_block();
                self.b.ins().brif(valid, next, &[], miss, &[]);
                self.b.switch_to_block(next);
            }
            if site.copyable & (1 << index) == 0 {
                changes.push((word, true));
            }
        }
        if let Some(destination) = site.destination {
            if !site.destination_copyable {
                changes.push((self.local(destination), false));
            }
        }
        if self.borrowed_transaction(&changes, miss) {
            // No failing guard/observer between the owner commit and these exact
            // owner moves. Incoming operand owners and the source homes stay intact.
            if let Some(destination) = site.destination {
                self.pc = site.end;
                self.set_local_integer(destination, words[0], integers[0]);
                self.count_native_site_success(site.start, "borrowed_local_copy");
                self.jump_normal(self.pc_block(site.end + 1));
            } else {
                for index in 0..site.count {
                    self.pc = site.start + index;
                    self.push_integer(words[index], integers[index]);
                }
                self.pc = site.end;
                self.dirty = &self.frame_plan.dirty_at[site.end];
                self.count_native_site_success(site.start, "borrowed_call_materialization");
                self.lower(site.end, &self.chunk.jit_ops()[site.end]);
            }
        }
        self.b.switch_to_block(miss);
        self.b.set_cold_block(miss);
        self.pc = site.start;
        self.stack_state = incoming;
        self.dirty = &self.frame_plan.dirty_at[site.start];
        self.deopt_before(site.start, self.error);
        #[cfg(test)]
        TEST_SELECTED.with(|sites| sites.borrow_mut().push((site.start, site.end)));
        Ok(())
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_TRANSACTIONS:Cell<usize>=const{Cell::new(0)};
    pub(super) static TEST_SELECTED:std::cell::RefCell<Vec<(usize,usize)>>=const{std::cell::RefCell::new(Vec::new())};
}

#[cfg(test)]
#[path = "jit_optimizing_borrowed_operand_tests.rs"]
mod tests;
