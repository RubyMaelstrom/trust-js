//! Exact Null strict comparisons without coercion or a field/type specialization.
//!
//! ECMA-262 e28783d5, IsStrictlyEqual / SameValueNonNumber and the HTMLDDA annex:
//! only another Null is strictly equal to Null. IsHTMLDDA, Proxy traps and
//! ToPrimitive do not participate. A settled exact-Null fact proves one operand
//! owns nothing; arbitrary peers retain their ordinary destruction boundaries.

use super::*;

fn enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = TEST_ENABLED.with(std::cell::Cell::get) {
        return enabled;
    }
    std::env::var("TRUST_JS_OPT_JIT_STRICT_NULL").as_deref() != Ok("0")
}

/// Select from the same settled facts used by lowering. An unavailable plan or
/// opaque operand prefix supplies no exact-Null proof, including whole-function
/// VALUE_FACTS=0. Continuations can have their own available guarded entry plan.
pub(super) fn selected_peer(
    op: &Op,
    inputs: Option<value_facts::Inputs>,
    known_suffix: usize,
) -> Option<(usize, value_facts::Types)> {
    if !matches!(op, Op::StrictEq | Op::StrictNotEq) || known_suffix < 2 || !enabled() {
        return None;
    }
    let facts = inputs?;
    if facts.top.is_null() {
        Some((2, facts.second))
    } else if facts.second.is_null() {
        Some((1, facts.top))
    } else {
        None
    }
}

impl Lowering<'_, '_> {
    pub(super) fn strict_null_comparison(&mut self, pc: usize, op: &Op, next: Block) -> bool {
        let facts = self.value_plan.and_then(|plan| plan.at.get(pc)).copied();
        let Some((peer_back, peer_type)) = selected_peer(op, facts, self.stack_state.known())
        else {
            return false;
        };
        #[cfg(test)]
        if TEST_CAPTURE.with(std::cell::Cell::get) {
            TEST_SITES.with(|sites| sites.borrow_mut().push((pc, peer_type.is_copyable())));
        }
        // Read the canonical packed owner, including the boxed word accompanying
        // any Number/integer SSA mirror. Comparing an unboxed scalar is invalid.
        let peer = self.stack_read(peer_back);
        let predicate = if matches!(op, Op::StrictEq) {
            IntCC::Equal
        } else {
            IntCC::NotEqual
        };
        let bit = self.b.ins().icmp_imm_s(predicate, peer, PACK_NULL as i64);
        let slow = if peer_type.is_copyable() {
            None
        } else {
            let slow = self.b.create_block();
            // No stack/owner mutation precedes this guard. Last-owner destruction
            // or an unsupported owner layout resumes checked execution at this
            // comparison, preserving the already evaluated operands and effects.
            self.transfer_owners(&[(peer, false)], slow);
            Some(slow)
        };
        let bit = self.b.ins().uextend(types::I64, bit);
        let boolean = self.b.ins().bor_imm_s(bit, PACK_BOOL as i64);
        self.stack_write(2, boolean);
        self.change_top(-8);
        self.jump_normal(next);
        if let Some(slow) = slow {
            self.slow(pc, slow, next);
        }
        true
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_CAPTURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    pub(super) static TEST_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    pub(super) static TEST_SITES: std::cell::RefCell<Vec<(usize, bool)>> = const { std::cell::RefCell::new(Vec::new()) };
    pub(super) static TEST_CHECKED: std::cell::RefCell<Vec<(usize, usize, [u64; 2])>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(super) unsafe fn record_checked(ctx: *mut JitCtx, pc: u32, sp: *mut PackedValue) {
    if !TEST_CAPTURE.with(std::cell::Cell::get) {
        return;
    }
    let depth = sp.offset_from((*ctx).stack_base) as usize;
    assert!(depth >= 2);
    TEST_CHECKED.with(|checked| {
        checked.borrow_mut().push((
            pc as usize,
            depth,
            [(*sp.sub(2)).bits(), (*sp.sub(1)).bits()],
        ));
    });
    tests::observe_checked_owner();
}

#[cfg(test)]
#[path = "jit_optimizing_strict_null_tests.rs"]
mod tests;
