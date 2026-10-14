//! Native consumers of exact semantic integer mirrors.
//!
//! ECMA-262 e28783d5 IsLooselyEqual first delegates equal-type Numbers to
//! IsStrictlyEqual/Number::equal. Range proofs exclude NaN, -0 and precision
//! loss, so signed I64 mathematical comparisons are exact, including UInt32
//! mirrors. Unknown/mixed inputs retain full checked conversions and equality.
//! ToBoolean is false precisely at integer zero; no heap owner is consumed here.

use super::*;

#[cfg(test)]
thread_local! {
    /// Emitted loose equality / strict-or-relational / branch / unary-not sites.
    pub(super) static TEST_SCALARS: std::cell::Cell<[usize; 4]> = const { std::cell::Cell::new([0; 4]) };
}

#[cfg(test)]
pub(super) fn record(kind: usize) {
    TEST_SCALARS.with(|counts| {
        let mut values = counts.get();
        values[kind] += 1;
        counts.set(values);
    });
}

impl Lowering<'_, '_> {
    pub(super) fn integer_scalar(&mut self, op: &Op, next: Block) -> bool {
        let (back, cc, kind) = match op {
            Op::EqEq => (2, IntCC::Equal, 0),
            Op::NotEq => (2, IntCC::NotEqual, 0),
            Op::StrictEq => (2, IntCC::Equal, 1),
            Op::StrictNotEq => (2, IntCC::NotEqual, 1),
            Op::Lt => (2, IntCC::SignedLessThan, 1),
            Op::Le => (2, IntCC::SignedLessThanOrEqual, 1),
            Op::Gt => (2, IntCC::SignedGreaterThan, 1),
            Op::Ge => (2, IntCC::SignedGreaterThanOrEqual, 1),
            Op::Not => (1, IntCC::Equal, 3),
            _ => return false,
        };
        if self.stack_state.known() < back {
            return false;
        }
        let Some(right) = self.stack_integer(1) else {
            return false;
        };
        let result = if back == 2 {
            let Some(left) = self.stack_integer(2) else {
                return false;
            };
            self.b.ins().icmp(cc, left, right)
        } else {
            self.b.ins().icmp_imm_s(cc, right, 0)
        };
        #[cfg(test)]
        record(kind);
        #[cfg(not(test))]
        let _ = kind;
        let bit = self.b.ins().uextend(types::I64, result);
        let word = self.b.ins().bor_imm_s(bit, PACK_BOOL as i64);
        // Both operands are immediate Numbers. Their canonical words remain
        // eager until this exact non-observing operation consumes them.
        self.stack_write(back, word);
        self.change_top(-8 * (back - 1) as i64);
        self.jump_normal(next);
        true
    }
}
