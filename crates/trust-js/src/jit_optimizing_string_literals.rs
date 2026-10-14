//! Borrowed literal-string comparisons with their original branch destinations.
//!
//! ECMA-262 e28783d5 IsStrictlyEqual / SameValueNonNumber / CaseBlockEvaluation:
//! String comparison neither converts nor observes other values. Four pure bytecodes may
//! therefore borrow the local and immutable constant instead of creating and consuming two
//! temporary owners. This is per-comparison fusion; the case chain and case bodies survive.

use super::*;
use crate::value::{PackedValue, Value as JsValue, PACK_EMPTY};

/// Guaranteed constant String retain/release work, less one borrowed String tag test.
/// Unknown-local owners, existing comparison tests and emitted helper/SSA work get no credit.
pub(super) const OWNER_WORK_PER_SITE: u32 = 1;

/// Sampling eligibility only. The complete CFG/stack/selection certificate decides benefit.
/// No value observation, owner, environment lookup or AST materialization occurs here.
pub(super) fn has_source(chunk: &Chunk) -> bool {
    chunk.jit_ops().windows(4).any(|ops| {
        let literal = match (&ops[0], &ops[1]) {
            (Op::LoadLocal(_), Op::Const(k)) | (Op::Const(k), Op::LoadLocal(_)) => *k,
            _ => return false,
        };
        matches!(ops[2], Op::StrictEq | Op::StrictNotEq)
            && matches!(ops[3], Op::JumpIfFalse(_))
            && chunk.jit_const_is_str(literal)
    })
}

/// Price the full original graph, including VM cuts: supported scan, two dense array
/// initializations, the certificate scan and the retained-site scan. Conservatively charge
/// estimate, final checkpoint and preparation. These are provisional compiler-work units,
/// not measured CPU instructions; no native/VM work is credited for omitted operations.
pub(super) fn analysis_work(ops: usize) -> Option<u64> {
    u64::try_from(ops).ok()?.checked_mul(15)
}

/// Object-result reachability adds one full-source filter at each analysis phase.
pub(super) fn analysis_work_with_filter(ops: usize, object_filter: bool) -> Option<u64> {
    analysis_work(ops)?.checked_add(if object_filter {
        u64::try_from(ops).ok()?.checked_mul(3)?
    } else {
        0
    })
}

/// Two additional bounded hot-checkpoint solvers. The attempted analysis is priced even
/// when budget exhaustion returns no proof; that result must retain ordinary Empty guards.
/// Candidate preparation already performed generic value analysis before literal fusion.
pub(super) fn checkpoint_value_work(
    ops: usize,
    slots: usize,
    capacity: usize,
    attempted: bool,
) -> Option<u64> {
    if !attempted {
        return Some(0);
    }
    let width = slots.checked_add(capacity)?.checked_add(1)?;
    u64::try_from(ops)
        .ok()?
        .checked_mul(u64::try_from(width).ok()?)?
        .checked_mul(2)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Site {
    slot: u16,
    literal: u32,
    negate: bool,
    mismatch: usize,
}

impl Site {
    /// Scalar certificate access for a stronger compiler transformation. The grouped
    /// dispatcher must independently validate all additional original CFG/stack entries.
    pub(super) fn identity(self) -> (u16, u32, bool, usize) {
        (self.slot, self.literal, self.negate, self.mismatch)
    }
}

/// Compiler-only scalar data. Constants are borrowed from the exact candidate Chunk; no
/// objects, environments, string owners or additional runtime metadata are retained.
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
        deopt: Option<usize>,
    ) -> Option<Self> {
        #[cfg(test)]
        if !TEST_ENABLED.with(std::cell::Cell::get) {
            return None;
        }
        if !bytecode::native_deopt::supported(chunk) {
            return None;
        }
        let ops = chunk.jit_ops();
        let mut sites = vec![None; ops.len()];
        let mut interior = vec![false; ops.len()];
        let mut pc = 0;
        let mut count = 0;
        while pc + 3 < ops.len() {
            let selected = (|| {
                let (slot, literal) = match (&ops[pc], &ops[pc + 1]) {
                    (Op::LoadLocal(slot), Op::Const(literal))
                    | (Op::Const(literal), Op::LoadLocal(slot)) => (*slot, *literal),
                    _ => return None,
                };
                let negate = match ops[pc + 2] {
                    Op::StrictEq => false,
                    Op::StrictNotEq => true,
                    _ => return None,
                };
                let Op::JumpIfFalse(mismatch) = ops[pc + 3] else {
                    return None;
                };
                let block = cfg.block_at(pc)?;
                // One original semantic block excludes every interior branch, handler,
                // dynamic completion and coroutine resume entry. Independent optimizing
                // continuations are excluded by the caller before constructing this plan.
                if cfg.block_at(pc + 3)? != block
                    || !chunk.jit_const_is_str(literal)
                    || !stack.reachable[pc]
                    || stack.at[pc] != stack.after[pc + 3]
                    || deopt.is_some_and(|at| pc < at && at < pc + 4)
                    || (pc..pc + 4).any(|at| {
                        regions.is_some_and(|plan| {
                            plan.emission(at) != observed_regions::Emission::Native
                        })
                    })
                {
                    return None;
                }
                Some(Site {
                    slot,
                    literal,
                    negate,
                    mismatch: mismatch as usize,
                })
            })();
            if let Some(site) = selected {
                sites[pc] = Some(site);
                interior[pc + 1..pc + 4].fill(true);
                count += 1;
                pc += 4;
            } else {
                pc += 1;
            }
        }
        (count != 0).then(|| Self {
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

    pub(super) fn restrict_reachable(mut self, reachable: Option<&[bool]>) -> Option<Self> {
        if let Some(reachable) = reachable {
            for pc in 0..self.sites.len() {
                if self.sites[pc].is_some()
                    && (pc..pc + 4).any(|at| !reachable.get(at).copied().unwrap_or(false))
                {
                    self.sites[pc] = None;
                    self.interior[pc + 1..pc + 4].fill(false);
                }
            }
        }
        self.sites.iter().any(Option::is_some).then_some(self)
    }

    /// Credit follows both original reads; an earlier TDZ/author effect cannot earn it.
    pub(super) fn credit_at(&self, pc: usize) -> u32 {
        pc.checked_sub(2)
            .and_then(|at| self.sites.get(at))
            .filter(|site| site.is_some())
            .map_or(0, |_| OWNER_WORK_PER_SITE)
    }

    pub(super) fn selected_sites(&self) -> usize {
        self.sites.iter().filter(|site| site.is_some()).count()
    }

    pub(super) fn payload_bytes(&self) -> usize {
        std::mem::size_of_val(&*self.sites) + std::mem::size_of_val(&*self.interior)
    }
}

/// Safety: `word` is borrowed from a live canonical/SSA owner. `literal` points into the
/// exact executing Chunk's immutable constants, pinned by preparation and native execution.
/// The worker sees only these numeric addresses. No owner, allocation, callback, collection,
/// polling, USV conversion or JavaScript object inspection occurs here, including for views.
unsafe extern "C" fn equal(word: u64, literal: *const JsValue) -> u64 {
    let JsValue::Str(literal) = (unsafe { &*literal }) else {
        return 0;
    };
    #[cfg(test)]
    {
        TEST_LEAF_ENTRIES.with(|count| count.set(count.get().saturating_add(1)));
        if word & 0xffff_0000_0000_0000 == crate::value::PACK_STR {
            // LStr's validated Header starts with strong: Cell<usize>, including views.
            // This test-only read distinguishes an actual sole-local borrowed owner from
            // a route accidentally retaining the old two temporary execution owners.
            let strong = unsafe {
                &*((word & 0x0000_ffff_ffff_ffff) as usize as *const std::cell::Cell<usize>)
            }
            .get();
            TEST_STRING_OWNERS.with(|counts| {
                let mut next = counts.get();
                let index = usize::from(strong != 1);
                next[index] = next[index].saturating_add(1);
                counts.set(next);
            });
        }
    }
    u64::from(unsafe { PackedValue::raw_string_equals(word, literal.as_str()) })
}

impl Lowering<'_, '_> {
    pub(super) fn string_literal_comparison(&mut self, pc: usize, site: Site) {
        let word = self.local(site.slot);
        let load_pc = pc + usize::from(!matches!(self.chunk.jit_ops()[pc], Op::LoadLocal(_)));
        // Reuse EXACTLY the original LoadLocal proof. In reversed order, Const at
        // pc has no queried local input; the successful-read fact lives at pc+1.
        // An absent/budget-exhausted proof retains the before-start TDZ recovery.
        let known_initialized = self
            .value_plan
            .and_then(|plan| plan.at.get(load_pc))
            .is_some_and(|inputs| inputs.local.initialized());
        let empty = (!known_initialized).then(|| {
            let initialized = self
                .b
                .ins()
                .icmp_imm_s(IntCC::NotEqual, word, PACK_EMPTY as i64);
            let fast = self.b.create_block();
            let empty = self.b.create_block();
            self.b.ins().brif(initialized, fast, &[], empty, &[]);
            self.b.switch_to_block(fast);
            #[cfg(test)]
            TEST_EMPTY_GUARDS.with(|count| count.set(count.get() + 1));
            empty
        });
        let address = self.b.ins().iconst(
            types::I64,
            self.chunk.jit_const_ptr(site.literal) as usize as i64,
        );
        let callee = self
            .b
            .ins()
            .iconst(types::I64, equal as *const () as usize as i64);
        if let Some(record) = self.diagnostics {
            record.emitted_helpers.set(record.emitted_helpers.get() + 1);
        }
        if let Some(record) = self.diagnostics {
            record.emitted_literal();
        }
        let call = self.b.ins().call_indirect(
            self.string_literal_sig.expect("selected literal signature"),
            callee,
            &[word, address],
        );
        let result = self.b.inst_results(call)[0];
        let comparison = if site.negate {
            IntCC::Equal
        } else {
            IntCC::NotEqual
        };
        let matches = self.b.ins().icmp_imm_s(comparison, result, 0);
        self.count_native_site_success(pc, "literal_string_comparison");
        // Finish the unchanged opaque prefix at the terminal branch, not the producing
        // LoadLocal/Const. No temporary stack owner was created on the native path.
        self.pc = pc + 3;
        self.dirty = &self.frame_plan.dirty_at[self.pc];
        self.finish_normal();
        let yes = self.loop_exit(self.pc_block(pc + 4));
        let no = self.loop_exit(self.pc_block(site.mismatch));
        self.b.ins().brif(matches, yes, &[], no, &[]);

        if let Some(empty) = empty {
            self.b.switch_to_block(empty);
            self.b.set_cold_block(empty);
            // Reversed operands originally create an owning Const before LoadLocal. Resuming
            // at the group start reproduces that exact stack and original TDZ error position.
            self.pc = pc;
            self.stack_state = self.stack_plan.at[pc];
            self.dirty = &self.frame_plan.dirty_at[pc];
            self.deopt_before(pc, self.error);
        }
        #[cfg(test)]
        TEST_SELECTED.with(|sites| sites.borrow_mut().push(pc));
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    pub(super) static TEST_LEAF_ENTRIES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    pub(super) static TEST_EMPTY_GUARDS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static TEST_SELECTED: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    pub(super) static TEST_STRING_OWNERS: std::cell::Cell<[u64; 2]> = const { std::cell::Cell::new([0; 2]) };
}

#[cfg(test)]
#[path = "jit_optimizing_string_literal_tests.rs"]
mod tests;
