//! Ownership-preserving packed-word operations for the whole-function tier.
//!
//! ECMA-262 e28783d5: GetValue / PutValue and WeakRef liveness. These operations change
//! storage, not value identity. All failing guards precede the single retain/release effect.
//! Last-owner drops stay checked: destruction may update weak metadata or owned host state.
//! No safepoint may separate a native refcount change from its corresponding SSA/stack move.

use super::*;
use cranelift_codegen::cursor::Cursor;

#[cfg(test)]
thread_local! {
    pub(super) static TEST_CALLEE_CACHE:std::cell::Cell<bool>=const{std::cell::Cell::new(true)};
    pub(super) static TEST_CALLEE_CONSTANTS:std::cell::Cell<usize>=const{std::cell::Cell::new(0)};
}
use crate::value::{PACK_BIGINT, PACK_OBJ, PACK_STR, PACK_SYM};

#[cfg(test)]
thread_local! {
    pub(super) static TEST_OWNERSHIP_HELPERS: std::cell::Cell<[usize; 4]> =
        const { std::cell::Cell::new([0; 4]) };
    pub(super) static TEST_OWNERSHIP_SITES: std::cell::Cell<[usize; 4]> =
        const { std::cell::Cell::new([0; 4]) };
    // retain calls/IR/blocks; release calls/IR/blocks; multi-owner calls/IR/blocks.
    pub(super) static TEST_OWNER_IR: std::cell::Cell<[usize; 9]> =
        const { std::cell::Cell::new([0; 9]) };
}

#[cfg(test)]
fn record_owner_ir(kind: usize, before: (usize, usize), after: (usize, usize)) {
    TEST_OWNER_IR.with(|cell| {
        let mut counts = cell.get();
        counts[kind * 3] += 1;
        counts[kind * 3 + 1] += after.0 - before.0;
        counts[kind * 3 + 2] += after.1 - before.1;
        cell.set(counts);
    });
}

impl Lowering<'_, '_> {
    /// The two scalar leaf functions are immutable engine code, with no Realm or
    /// JS owner. Insert a pure constant before the existing entry prologue only
    /// when its call is actually emitted. That definition dominates every native
    /// branch, handler and loop without adding unused constants to numeric bodies.
    /// FunctionBuilder::cursor permits SSA insertion that touches no locals/CFG;
    /// every inserted instruction remains charged to the ordinary DFG budget.
    fn owner_leaf_callee(&mut self, retain: bool) -> Value {
        let address = if retain {
            owner_leaf::retain
        } else {
            owner_leaf::release
        } as usize as i64;
        #[cfg(test)]
        if !TEST_CALLEE_CACHE.with(std::cell::Cell::get) {
            return self.b.ins().iconst(types::I64, address);
        }
        let index = usize::from(retain);
        if let Some(callee) = self.owner_leaf_callees[index] {
            return callee;
        }
        let first = self
            .b
            .func
            .layout
            .entry_block()
            .and_then(|entry| self.b.func.layout.first_inst(entry));
        let Some(first) = first else {
            // The normal lowering always has an entry prologue. A future caller
            // lacking it retains the complete original per-call emission.
            return self.b.ins().iconst(types::I64, address);
        };
        let callee = self
            .b
            .cursor()
            .at_inst(first)
            .ins()
            .iconst(types::I64, address);
        self.owner_leaf_callees[index] = Some(callee);
        #[cfg(test)]
        TEST_CALLEE_CONSTANTS.with(|count| count.set(count.get() + 1));
        callee
    }

    pub(super) fn owner_leaf_count(&mut self, counter: usize, amount: Value) {
        if let Some(record) = self.diagnostics.filter(|record| record.runtime_counters) {
            let address = self
                .b
                .ins()
                .iconst(types::I64, record.counter(counter) as i64);
            let previous = self.load(address, 0);
            let next = self.b.ins().iadd(previous, amount);
            let no_overflow = self
                .b
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, next, previous);
            let maximum = self.b.ins().iconst(types::I64, -1);
            let next = self.b.ins().select(no_overflow, next, maximum);
            self.store(next, address, 0);
        }
    }

    /// Pure scalar C call, not an observing helper: canonical owners, private
    /// words, storage and metadata remain live. Cranelift preserves every live
    /// SSA value across the ABI caller-clobbered registers. No final_sp reload
    /// or JavaScript frame publication is needed because the leaf cannot GC.
    fn shared_owner(&mut self, value: Value, retain: bool, slow: Block) -> bool {
        let Some(signature) = self.owner_leaf_sig else {
            return false;
        };
        if let Some(record) = self.diagnostics {
            record.emitted_helpers.set(record.emitted_helpers.get() + 1);
            record
                .emitted_owner_leaf_calls
                .set(record.emitted_owner_leaf_calls.get() + 1);
        }
        if self
            .diagnostics
            .is_some_and(|record| record.runtime_counters)
        {
            let one = self.b.ins().iconst(types::I64, 1);
            self.owner_leaf_count(diagnostics::OWNER_LEAF, one);
        }
        let callee = self.owner_leaf_callee(retain);
        let call = self.b.ins().call_indirect(signature, callee, &[value]);
        let status = self.b.inst_results(call)[0];
        if self
            .diagnostics
            .is_some_and(|record| record.runtime_counters)
        {
            let guards = self
                .b
                .ins()
                .ushr_imm_u(status, i64::from(owner_leaf::GUARD_SHIFT));
            self.owner_leaf_count(diagnostics::OWNER_LEAF_GUARDS, guards);
        }
        let success = self.b.ins().band_imm_s(status, owner_leaf::SUCCESS as i64);
        self.owner_guard(success, slow);
        if self
            .diagnostics
            .is_some_and(|record| record.runtime_counters)
        {
            let transfer = self.b.ins().band_imm_s(status, owner_leaf::TRANSFER as i64);
            let transfer = self.b.ins().ushr_imm_u(transfer, 1);
            self.owner_leaf_count(
                if retain {
                    diagnostics::RETAIN
                } else {
                    diagnostics::RELEASE
                },
                transfer,
            );
        }
        true
    }

    fn owner_guard(&mut self, pass: Value, miss: Block) {
        self.count_heap_guard();
        let next = self.b.create_block();
        self.b.ins().brif(pass, next, &[], miss, &[]);
        self.b.switch_to_block(next);
    }

    /// Only execution owners with a live, validated strong-count ABI are accepted. In
    /// particular a lazy function-prototype thunk must materialize through the checked path.
    fn owned_pointer(&mut self, value: Value, slow: Block) -> Value {
        let tag = self.b.ins().ushr_imm_u(value, 48);
        let mut supported = self
            .b
            .ins()
            .icmp_imm_s(IntCC::Equal, tag, (PACK_OBJ >> 48) as i64);
        for tag_bits in [PACK_STR, PACK_SYM, PACK_BIGINT] {
            let kind = self
                .b
                .ins()
                .icmp_imm_s(IntCC::Equal, tag, (tag_bits >> 48) as i64);
            supported = self.b.ins().bor(supported, kind);
        }
        self.owner_guard(supported, slow);
        self.b.ins().band_imm_s(value, 0x0000_ffff_ffff_ffff)
    }

    pub(super) fn retain_word(&mut self, value: Value, slow: Block) {
        #[cfg(test)]
        let before = (self.b.func.dfg.num_insts(), self.b.func.dfg.num_blocks());
        self.retain_word_body(value, slow);
        #[cfg(test)]
        record_owner_ir(
            0,
            before,
            (self.b.func.dfg.num_insts(), self.b.func.dfg.num_blocks()),
        );
    }

    fn retain_word_body(&mut self, value: Value, slow: Block) {
        let immediate = self.immediate(value);
        if !self.owners_valid {
            self.owner_guard(immediate, slow);
            return;
        }
        let heap = self.b.create_block();
        let done = self.b.create_block();
        self.b.ins().brif(immediate, done, &[], heap, &[]);
        self.b.switch_to_block(heap);
        if self.shared_owner(value, true, slow) {
            self.b.ins().jump(done, &[]);
            self.b.switch_to_block(done);
            return;
        }
        let owner = self.owned_pointer(value, slow);
        let count = self.load(owner, self.values.rc_strong_off as i32);
        let incremented = self.b.ins().iadd_imm_s(count, 1);
        let valid = self
            .b
            .ins()
            .icmp_imm_s(IntCC::UnsignedGreaterThan, incremented, 1);
        self.owner_guard(valid, slow);
        self.store(incremented, owner, self.values.rc_strong_off as i32);
        self.count_runtime(diagnostics::RETAIN, 1);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
    }

    fn release_word(&mut self, value: Value, slow: Block) {
        #[cfg(test)]
        let before = (self.b.func.dfg.num_insts(), self.b.func.dfg.num_blocks());
        self.release_word_body(value, slow);
        #[cfg(test)]
        record_owner_ir(
            1,
            before,
            (self.b.func.dfg.num_insts(), self.b.func.dfg.num_blocks()),
        );
    }

    fn release_word_body(&mut self, value: Value, slow: Block) {
        let immediate = self.immediate(value);
        if !self.owners_valid {
            self.owner_guard(immediate, slow);
            return;
        }
        let heap = self.b.create_block();
        let done = self.b.create_block();
        self.b.ins().brif(immediate, done, &[], heap, &[]);
        self.b.switch_to_block(heap);
        if self.shared_owner(value, false, slow) {
            self.b.ins().jump(done, &[]);
            self.b.switch_to_block(done);
            return;
        }
        let owner = self.owned_pointer(value, slow);
        let count = self.load(owner, self.values.rc_strong_off as i32);
        let shared = self
            .b
            .ins()
            .icmp_imm_s(IntCC::UnsignedGreaterThan, count, 1);
        self.owner_guard(shared, slow);
        let decremented = self.b.ins().iadd_imm_s(count, -1);
        self.store(decremented, owner, self.values.rc_strong_off as i32);
        self.count_runtime(diagnostics::RELEASE, 1);
        self.b.ins().jump(done, &[]);
        self.b.switch_to_block(done);
    }

    /// An indivisible set of owner transfers. A property overwrite can release BOTH its
    /// previous value and the consumed receiver, and retain its expression result. Those
    /// words and a consumed string index may name the same allocation. Independent `strong > 1` checks are
    /// unsound: together two releases could destroy that allocation without its destructor.
    ///
    /// First classify and combine aliases, then validate every resulting count, and only
    /// then commit. No guard, helper, allocation or safepoint follows the first count store.
    /// Last-owner destruction and unrecognized owner ABIs stay on the checked path, with
    /// ALL original counts and operand owners unchanged. This preserves the production
    /// nursery's Rc-based conservative old-to-young remembered set as well as strong owners.
    pub(super) fn transfer_owners(&mut self, changes: &[(Value, bool)], slow: Block) {
        assert!(!changes.is_empty() && changes.len() <= 4);
        if let [(value, retain)] = changes {
            if *retain {
                self.retain_word(*value, slow);
            } else {
                self.release_word(*value, slow);
            }
            return;
        }
        #[cfg(test)]
        let before = (self.b.func.dfg.num_insts(), self.b.func.dfg.num_blocks());
        let zero = self.b.ins().iconst(types::I64, 0);
        let mut owners = Vec::with_capacity(changes.len());
        for &(word, _) in changes {
            let immediate = self.immediate(word);
            if !self.owners_valid {
                self.owner_guard(immediate, slow);
                owners.push(zero);
                continue;
            }
            let heap = self.b.create_block();
            let done = self.b.create_block();
            self.b.append_block_param(done, types::I64);
            self.b
                .ins()
                .brif(immediate, done, &[zero.into()], heap, &[]);
            self.b.switch_to_block(heap);
            let owner = self.owned_pointer(word, slow);
            self.owner_guard(owner, slow);
            self.b.ins().jump(done, &[owner.into()]);
            self.b.switch_to_block(done);
            owners.push(self.b.block_params(done)[0]);
        }

        let mut writes = Vec::with_capacity(changes.len());
        for (index, &owner) in owners.iter().enumerate() {
            let mut first = self.b.ins().icmp_imm_s(IntCC::NotEqual, owner, 0);
            for &previous in &owners[..index] {
                let distinct = self.b.ins().icmp(IntCC::NotEqual, owner, previous);
                first = self.b.ins().band(first, distinct);
            }
            let inspect = self.b.create_block();
            let done = self.b.create_block();
            self.b.append_block_param(done, types::I64); // zero for a duplicate/immediate
            self.b.append_block_param(done, types::I64); // the final strong count
            self.b
                .ins()
                .brif(first, inspect, &[], done, &[zero.into(), zero.into()]);
            self.b.switch_to_block(inspect);
            let mut retains = zero;
            let mut releases = zero;
            for (&other, &(_, retain)) in owners[index..].iter().zip(&changes[index..]) {
                let same = self.b.ins().icmp(IntCC::Equal, owner, other);
                let one = self.b.ins().uextend(types::I64, same);
                if retain {
                    retains = self.b.ins().iadd(retains, one);
                } else {
                    releases = self.b.ins().iadd(releases, one);
                }
            }
            let before = self.load(owner, self.values.rc_strong_off as i32);
            self.owner_guard(before, slow);
            let retained = self.b.ins().iadd(before, retains);
            let no_overflow =
                self.b
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, retained, before);
            self.owner_guard(no_overflow, slow);
            let survives = self
                .b
                .ins()
                .icmp(IntCC::UnsignedGreaterThan, retained, releases);
            self.owner_guard(survives, slow);
            let after = self.b.ins().isub(retained, releases);
            self.b.ins().jump(done, &[owner.into(), after.into()]);
            self.b.switch_to_block(done);
            writes.push((self.b.block_params(done)[0], self.b.block_params(done)[1]));
        }

        // Commit exactly once per distinct allocation, including alias-cancelled transfers.
        for (owner, count) in writes {
            let write = self.b.create_block();
            let done = self.b.create_block();
            self.b.ins().brif(owner, write, &[], done, &[]);
            self.b.switch_to_block(write);
            self.store(count, owner, self.values.rc_strong_off as i32);
            self.b.ins().jump(done, &[]);
            self.b.switch_to_block(done);
        }
        // Count semantic transfers only after every guard succeeded, including transfers
        // that cancel through aliasing. Immediate words do not own an allocation.
        if let Some(record) = self.diagnostics.filter(|record| record.runtime_counters) {
            for (&owner, &(_, retain)) in owners.iter().zip(changes) {
                let heap = self.b.ins().icmp_imm_s(IntCC::NotEqual, owner, 0);
                let amount = self.b.ins().uextend(types::I64, heap);
                let counter = if retain {
                    diagnostics::RETAIN
                } else {
                    diagnostics::RELEASE
                };
                let address = self
                    .b
                    .ins()
                    .iconst(types::I64, record.counter(counter) as i64);
                let previous = self.load(address, 0);
                let next = self.b.ins().iadd(previous, amount);
                self.store(next, address, 0);
            }
        }
        #[cfg(test)]
        record_owner_ir(
            2,
            before,
            (self.b.func.dfg.num_insts(), self.b.func.dfg.num_blocks()),
        );
    }

    pub(super) fn ownership(&mut self, pc: usize, op: &Op, next: Block) {
        #[cfg(test)]
        TEST_OWNERSHIP_SITES.with(|count| {
            let kind = match op {
                Op::LoadLocal(_) => 0,
                Op::Dup => 1,
                Op::Pop => 2,
                Op::StoreLocal(_) => 3,
                _ => unreachable!("packed ownership operation"),
            };
            let mut counts = count.get();
            counts[kind] += 1;
            count.set(counts);
        });
        let value = match op {
            Op::LoadLocal(slot) | Op::StoreLocal(slot) => self.local(*slot),
            _ => self.stack_read(1),
        };
        let inputs = self.input_types();
        let known = match op {
            Op::LoadLocal(_) | Op::StoreLocal(_) => inputs.local,
            _ => inputs.top,
        };
        let integer = match op {
            Op::LoadLocal(slot) => self.local_integer(*slot),
            Op::Dup | Op::StoreLocal(_) => self.stack_integer(1),
            _ => None,
        };
        // A whole-function proof can remove the entire owner/slow-path graph. BigInt,
        // String and Symbol are NOT copyable here, even though ECMAScript calls them
        // primitive values. Empty is copyable storage but never a successful local read.
        if known.is_copyable() && (!matches!(op, Op::LoadLocal(_)) || known.initialized()) {
            #[cfg(test)]
            Self::count_value_proof(2, 1);
            match op {
                Op::LoadLocal(_) | Op::Dup => self.push_integer(value, integer),
                Op::StoreLocal(slot) => {
                    let incoming = self.stack_read(1);
                    self.set_local_integer(*slot, incoming, integer);
                    self.change_top(-8);
                }
                Op::Pop => self.change_top(-8),
                _ => unreachable!("packed ownership operation"),
            }
            self.jump_normal(next);
            return;
        }
        let slow = self.b.create_block();
        if matches!(op, Op::LoadLocal(_)) && !known.initialized() {
            // GetBindingValue must still throw for the temporal dead zone, even though the
            // internal Empty word needs no retain/drop and is safe for storage initialization.
            let initialized = self
                .b
                .ins()
                .icmp_imm_s(IntCC::NotEqual, value, PACK_EMPTY as i64);
            self.owner_guard(initialized, slow);
        }
        match op {
            Op::LoadLocal(_) | Op::Dup => {
                self.retain_word(value, slow);
                self.push_integer(value, integer);
            }
            Op::StoreLocal(slot) => {
                // The input stack already OWNS incoming. Self-assignment therefore has at
                // least two references and cannot destroy the value while moving it home.
                let incoming = self.stack_read(1);
                self.release_word(value, slow);
                self.set_local_integer(*slot, incoming, integer);
                self.change_top(-8);
            }
            Op::Pop => {
                self.release_word(value, slow);
                self.change_top(-8);
            }
            _ => unreachable!("packed ownership operation"),
        }
        self.jump_normal(next);
        self.slow(pc, slow, next);
    }
}
