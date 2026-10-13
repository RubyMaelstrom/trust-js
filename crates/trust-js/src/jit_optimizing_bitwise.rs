//! ECMA-262 e28783d5 #sec-toint32, #sec-tofixedsizeinteger,
//! #sec-numberbitwiseop and Number shift methods. No host conversion or
//! coercion occurs here: callers first prove both operands are Numbers.

use super::*;

impl Lowering<'_, '_> {
    /// Truncate and reduce modulo 2^32, including huge finite Numbers, signed
    /// zero, infinities and NaN. The common range takes one machine conversion;
    /// the rare large range extracts the low integer bits from the IEEE value.
    pub(super) fn number_i32(&mut self, word: Value, number: Value) -> Value {
        let magnitude = self.b.ins().band_imm_s(word, i64::MAX);
        let small = self.b.ins().icmp_imm_s(
            IntCC::UnsignedLessThan,
            magnitude,
            (1086u64 << 52) as i64, // 2^63; signed conversion is exact below it.
        );
        let fast = self.b.create_block();
        let large = self.b.create_block();
        let done = self.b.create_block();
        self.b.append_block_param(done, types::I32);
        self.b.ins().brif(small, fast, &[], large, &[]);
        self.b.switch_to_block(fast);
        let integer = self.b.ins().fcvt_to_sint_sat(types::I64, number);
        let integer = self.b.ins().ireduce(types::I32, integer);
        self.b.ins().jump(done, &[integer.into()]);
        self.b.switch_to_block(large);
        let exponent = self.b.ins().ushr_imm_u(magnitude, 52);
        // At 2^84 and above every finite double is a multiple of 2^32.
        // The same zero result applies to exponent 2047 (NaN and infinities).
        let useful = self
            .b
            .ins()
            .icmp_imm_s(IntCC::UnsignedLessThan, exponent, 1107);
        let shift = self.b.ins().iadd_imm_s(exponent, -1075);
        let mantissa = self.b.ins().band_imm_s(word, 0x000f_ffff_ffff_ffff);
        let mantissa = self.b.ins().bor_imm_s(mantissa, 1i64 << 52);
        let integer = self.b.ins().ishl(mantissa, shift);
        let integer = self.b.ins().ireduce(types::I32, integer);
        let negative = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, word, 0);
        let negated = self.b.ins().ineg(integer);
        let integer = self.b.ins().select(negative, negated, integer);
        let zero = self.b.ins().iconst(types::I32, 0);
        let integer = self.b.ins().select(useful, integer, zero);
        self.b.ins().jump(done, &[integer.into()]);
        self.b.switch_to_block(done);
        self.b.block_params(done)[0]
    }

    pub(super) fn bitwise_number(
        &mut self,
        op: &Op,
        left_word: Value,
        right_word: Value,
        left: Value,
        right: Value,
    ) -> Value {
        let left = self.number_i32(left_word, left);
        let right = self.number_i32(right_word, right);
        let integer = match op {
            Op::BitAnd => self.b.ins().band(left, right),
            Op::BitOr => self.b.ins().bor(left, right),
            Op::BitXor => self.b.ins().bxor(left, right),
            // I32 shifts mask their amount modulo 32, as required by Number.
            Op::Shl => self.b.ins().ishl(left, right),
            Op::Shr => self.b.ins().sshr(left, right),
            Op::UShr => self.b.ins().ushr(left, right),
            _ => unreachable!("Number bitwise operation"),
        };
        if matches!(op, Op::UShr) {
            self.b.ins().fcvt_from_uint(types::F64, integer)
        } else {
            self.b.ins().fcvt_from_sint(types::F64, integer)
        }
    }
}
