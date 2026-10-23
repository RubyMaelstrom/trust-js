//! ARM64 compact execution-word primitives. Canonical local/operand words and a context's return
//! slot (`ret`), `this`, lexical Binding cells and object data values share Value words.
//! ECMA-262 §6.1.6.1 (snapshot e28783d5): preserve signed zero/infinities; NaN payload
//! canonicalization is allowed and prevents numeric bits from aliasing internal tags.

use super::*;
use crate::value::{PACK_CANON_NAN, PACK_LAZY_PROTO, PACK_OBJ, PACK_SYM, PACK_UNDEFINED};

/// Decode the ValueView semantic discriminant (0..=8). Preserve raw, clobber
/// kind/scratch/NZCV, reject property-only tags.
pub(super) fn emit_exec_kind(a: &mut asm::Asm, raw: u32, kind: u32, scratch: u32, slow: usize) {
    assert!(raw != kind && raw != scratch && kind != scratch);
    let object = a.new_label();
    let number = a.new_label();
    let done = a.new_label();
    a.lsr_imm(kind, raw, 48);
    a.movz(scratch, (PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_x(kind, scratch);
    a.b_cond(C_EQ, object);
    a.b_cond(C_HI, slow);
    a.movz(scratch, (PACK_UNDEFINED >> 48) as u32, 0);
    a.cmp_reg_x(kind, scratch);
    a.b_cond(C_LO, number);
    a.movz(scratch, (PACK_SYM >> 48) as u32, 0);
    a.cmp_reg_x(kind, scratch);
    a.b_cond(C_HI, number);
    a.movz(scratch, (PACK_UNDEFINED >> 48) as u32, 0);
    a.sub_reg(kind, kind, scratch);
    // Positive tags map to 0..3,5..7; Number occupies discriminant 4.
    a.cmp_imm_w(kind, 4);
    a.b_cond(C_LO, done);
    a.add_imm(kind, kind, 1);
    a.b(done);
    a.bind(number);
    a.movz(kind, 4, 0);
    a.b(done);
    a.bind(object);
    a.movz(kind, 8, 0);
    a.bind(done);
}

/// No scratch registers or flags changed. Far-negative addresses need a caller-computed base.
pub(super) fn emit_exec_word_load(a: &mut asm::Asm, dst: u32, base: u32, off: i32) {
    if off >= 0 && off & 7 == 0 {
        a.ldr_imm(dst, base, off as u32);
    } else {
        assert!((-256..256).contains(&off));
        a.ldur(dst, base, off);
    }
}

pub(super) fn emit_exec_word_store(a: &mut asm::Asm, src: u32, base: u32, off: i32) {
    if off >= 0 && off & 7 == 0 {
        a.str_imm(src, base, off as u32);
    } else {
        assert!((-256..256).contains(&off));
        a.stur(src, base, off);
    }
}

/// Borrow a Number from an encoded word. Output dreg is live on success; scratch and
/// NZCV are clobbered. All tagged words are NaNs, but the canonical numeric NaN is valid.
pub(super) fn emit_exec_number_guard(
    a: &mut asm::Asm,
    raw: u32,
    dreg: u32,
    scratch: u32,
    fail: usize,
) {
    assert_ne!(raw, scratch);
    let number = a.new_label();
    a.fmov_d_x(dreg, raw);
    a.fcmp(dreg, dreg);
    a.b_cond(7, number); // VC: every finite value, signed zero and infinity.
    a.mov_imm64(scratch, PACK_CANON_NAN);
    a.cmp_reg_x(raw, scratch);
    a.b_cond(C_NE, fail);
    a.bind(number);
}

/// Compare only the encoded high tag; preserve raw and clobber exactly scratch (no NZCV).
pub(super) fn emit_exec_tag_guard(a: &mut asm::Asm, raw: u32, tag: u64, scratch: u32, fail: usize) {
    assert_ne!(raw, scratch);
    assert_eq!(tag & 0x0000_ffff_ffff_ffff, 0);
    a.mov_imm64(scratch, tag);
    a.logic_x(2, scratch, raw, scratch);
    a.lsr_imm(scratch, scratch, 48);
    a.cbnz(scratch, true, fail);
}

/// Branch to `slow` if `raw`, a word just read from a property, has a tag above PACK_OBJ: a
/// property-only word (a function's deferred `prototype`, PACK_LAZY_PROTO), which must not
/// become an execution word. The site's checked operation reads the property in Rust, which
/// materializes the prototype and leaves an ordinary Object word in the property. This is a
/// materialization boundary, not ownership: a site emits it where the property read can be
/// keyed "prototype" (no other routing sends the word to the checked path). Every other
/// word (scalars, BigInt, String, Symbol, Object) falls through with its payload unread. Raw
/// is preserved; scratch and NZCV are clobbered.
pub(super) fn property_word_guard(a: &mut asm::Asm, raw: u32, scratch: u32, slow: usize) {
    assert_ne!(raw, scratch);
    // One unsigned compare: PACK_LAZY_PROTO's tag is the first above Object's, so every word
    // with a higher tag is at least PACK_LAZY_PROTO and every other word is below it.
    const _: () = assert!(PACK_LAZY_PROTO == PACK_OBJ + (1 << 48));
    a.mov_imm64(scratch, PACK_LAZY_PROTO);
    a.cmp_reg_x(raw, scratch);
    a.b_cond(C_HS, slow);
}

/// Extract a borrowed 48-bit heap payload; raw==dst is allowed, flags are preserved.
pub(super) fn emit_exec_payload(a: &mut asm::Asm, raw: u32, dst: u32) {
    a.ubfx(dst, raw, 0, 48);
}

/// Store a Number as one owned execution word. No ownership traffic. Canonicalize
/// *all* numerical NaNs, including hostile DataView payloads and negative NaN.
/// Only scratch and NZCV are clobbered; dreg is preserved. Callers with a live branch
/// condition must compare after materialization, not across this operation.
pub(super) fn emit_exec_number_store(
    a: &mut asm::Asm,
    dreg: u32,
    base: u32,
    off: i32,
    scratch: u32,
) {
    assert_ne!(base, scratch);
    let number = a.new_label();
    let done = a.new_label();
    a.fcmp(dreg, dreg);
    a.b_cond(7, number);
    a.mov_imm64(scratch, PACK_CANON_NAN);
    emit_exec_word_store(a, scratch, base, off);
    a.b(done);
    a.bind(number);
    if off >= 0 && off & 7 == 0 {
        a.str_d_imm(dreg, base, off as u32);
    } else {
        assert!((-256..256).contains(&off));
        a.stur_d(dreg, base, off);
    }
    a.bind(done);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::{PACK_BIGINT, PACK_BOOL, PACK_EMPTY, PACK_NULL, PACK_STR};

    fn executable(a: asm::Asm) -> ExecutableBuffer {
        let bytes: Vec<u8> = a.finish().into_iter().flat_map(u32::to_le_bytes).collect();
        ExecutableBuffer::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn bitfield_extract_encodes_every_range_and_register_pair() {
        let mut a = asm::Asm::new();
        for lsb in 0..64 {
            for width in 1..=64 - lsb {
                for rn in 0..32 {
                    for rd in 0..32 {
                        a.buf.clear();
                        a.ubfx(rd, rn, lsb, width);
                        assert_eq!(a.buf.len(), 1);
                        let word = a.buf[0];
                        // Independently check every architected field of 64-bit UBFM.
                        assert_eq!(word >> 31, 1); // sf
                        assert_eq!((word >> 29) & 3, 2); // opc: unsigned, no flag update
                        assert_eq!((word >> 23) & 0x3f, 0b100110);
                        assert_eq!((word >> 22) & 1, 1); // N
                        assert_eq!((word >> 16) & 63, lsb); // immr
                        assert_eq!((word >> 10) & 63, lsb + width - 1); // imms
                        assert_eq!((word >> 5) & 31, rn);
                        assert_eq!(word & 31, rd);
                    }
                }
            }
        }
    }

    #[test]
    fn bitfield_extract_rejects_invalid_fields_before_emitting() {
        for (rd, rn, lsb, width) in [
            (32, 0, 0, 1),
            (0, 32, 0, 1),
            (0, 0, 64, 1),
            (0, 0, 0, 0),
            (0, 0, 0, 65),
            (0, 0, 63, 2),
            (0, 0, u32::MAX, 1),
            (0, 0, 1, u32::MAX),
        ] {
            let mut a = asm::Asm::new();
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || a.ubfx(rd, rn, lsb, width)
            ))
            .is_err());
            assert!(a.buf.is_empty());
        }
    }

    // x0=input, x1/x2=flag-generating operands, x3=output. x4 is an optional
    // separate destination, x5 records NZCV via CSET (which does not alter flags).
    // Keep all stubs in one executable allocation; register31 is tested as XZR.
    fn append_extract_probe(
        a: &mut asm::Asm,
        rn: u32,
        rd: u32,
        lsb: u32,
        width: u32,
        payload: bool,
    ) -> usize {
        let offset = a.buf.len() * 4;
        a.cmp_reg_x(1, 2);
        for (index, condition) in [C_EQ, C_HS, C_MI, C_VS].into_iter().enumerate() {
            a.cset_w(5, condition);
            a.str_imm(5, 3, (3 + index as u32) * 8);
        }
        let start = a.buf.len();
        if payload {
            assert_eq!((lsb, width), (0, 48));
            emit_exec_payload(a, rn, rd);
        } else {
            a.ubfx(rd, rn, lsb, width);
        }
        assert_eq!(a.buf.len() - start, 1, "extraction must be one instruction");
        a.str_imm(rd, 3, 0);
        a.str_imm(0, 3, 8);
        a.str_imm(1, 3, 16);
        for (index, condition) in [C_EQ, C_HS, C_MI, C_VS].into_iter().enumerate() {
            a.cset_w(5, condition);
            a.str_imm(5, 3, (7 + index as u32) * 8);
        }
        a.ret();
        offset
    }

    const FLAG_INPUTS: [(u64, u64); 6] = [
        (0, 0),
        (0, 1),
        (1, 0),
        (1 << 63, 1),
        ((1 << 63) - 1, u64::MAX),
        (1 << 63, 0),
    ];

    fn check_extract_probe(
        code: &ExecutableBuffer,
        offset: usize,
        raw: u64,
        rn: u32,
        rd: u32,
        lsb: u32,
        width: u32,
        flag_operands: (u64, u64),
    ) {
        let run: unsafe extern "C" fn(u64, u64, u64, *mut u64) =
            unsafe { std::mem::transmute(code.as_ptr().add(offset)) };
        let (left, right) = flag_operands;
        let mut output = [u64::MAX; 11];
        unsafe { run(raw, left, right, output.as_mut_ptr()) };
        let source = if rn == 31 { 0 } else { raw };
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let expected = if rd == 31 { 0 } else { (source >> lsb) & mask };
        assert_eq!(
            output[0], expected,
            "extract {raw:016x}, {lsb}:{width}, x{rn}→x{rd}"
        );
        assert_eq!(
            output[1],
            if rd == 0 { expected } else { raw },
            "source preservation"
        );
        assert_eq!(output[2], left, "unrelated register preservation");
        let difference = left.wrapping_sub(right);
        let flags = [
            u64::from(difference == 0),
            u64::from(left >= right),
            difference >> 63,
            ((left ^ right) & (left ^ difference)) >> 63,
        ];
        assert_eq!(output[3..7], flags, "probe's incoming CMP flags");
        assert_eq!(
            output[7..11],
            flags,
            "extract must preserve every NZCV flag"
        );
    }

    #[test]
    fn native_bitfield_extract_all_ranges_preserve_source_flags_and_aliasing() {
        let mut a = asm::Asm::new();
        let mut cases = Vec::new();
        for lsb in 0..64 {
            for width in 1..=64 - lsb {
                for (rn, rd) in [(0, 4), (0, 0), (31, 4), (0, 31)] {
                    let offset = append_extract_probe(&mut a, rn, rd, lsb, width, false);
                    cases.push((offset, rn, rd, lsb, width));
                }
            }
        }
        let code = executable(a);
        let mut random = 0x1319_8a2e_0370_7344u64;
        for (offset, rn, rd, lsb, width) in cases {
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            for raw in [
                0,
                u64::MAX,
                1 << lsb,
                1 << (lsb + width - 1),
                0x5555_5555_5555_5555,
                random,
            ] {
                for flags in FLAG_INPUTS {
                    check_extract_probe(&code, offset, raw, rn, rd, lsb, width, flags);
                }
            }
        }
    }

    #[test]
    fn native_payload_extract_exhausts_tags_and_preserves_flags() {
        let mut a = asm::Asm::new();
        let separate = append_extract_probe(&mut a, 0, 4, 0, 48, true);
        let in_place = append_extract_probe(&mut a, 0, 0, 0, 48, true);
        let code = executable(a);
        let mut random = 0xa409_3822_299f_31d0u64;
        for high in 0..=u16::MAX as u64 {
            for (index, payload) in [0, 1, 0x0000_ffff_ffff_ffff, random]
                .into_iter()
                .enumerate()
            {
                let raw = (high << 48) | (payload & 0x0000_ffff_ffff_ffff);
                let flags = FLAG_INPUTS[(high as usize + index) % FLAG_INPUTS.len()];
                check_extract_probe(&code, separate, raw, 0, 4, 0, 48, flags);
                check_extract_probe(&code, in_place, raw, 0, 0, 0, 48, flags);
            }
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
        }
    }

    #[test]
    fn native_execution_number_guard_rejects_tags_and_preserves_numeric_bits() {
        let mut a = asm::Asm::new();
        let fail = a.new_label();
        emit_exec_number_guard(&mut a, 0, 0, 2, fail);
        emit_exec_number_store(&mut a, 0, 1, 0, 3);
        a.movz(0, 1, 0);
        a.ret();
        a.bind(fail);
        a.movz(0, 0, 0);
        a.ret();
        let code = executable(a);
        let run: unsafe extern "C" fn(u64, *mut u64) -> u64 =
            unsafe { std::mem::transmute(code.as_ptr()) };
        for number in [
            0.0f64,
            -0.0,
            1.5,
            -1.5,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let raw = (Value::Num(number)).bits();
            let mut output = u64::MAX;
            assert_eq!(unsafe { run(raw, &mut output) }, 1);
            assert_eq!(output, raw);
        }
        for tag in [
            PACK_UNDEFINED,
            PACK_EMPTY,
            PACK_NULL,
            PACK_BOOL,
            PACK_BIGINT,
            PACK_STR,
            PACK_SYM,
            PACK_OBJ,
            crate::value::PACK_LAZY_PROTO,
        ] {
            let mut output = 123;
            assert_eq!(unsafe { run(tag, &mut output) }, 0);
            assert_eq!(output, 123, "failed guard changed canonical memory");
        }
    }

    #[test]
    fn native_execution_number_store_canonicalizes_hostile_nan_payloads() {
        let mut a = asm::Asm::new();
        a.fmov_d_x(0, 0);
        emit_exec_number_store(&mut a, 0, 1, 0, 2);
        a.ret();
        let code = executable(a);
        let run: unsafe extern "C" fn(u64, *mut u64) =
            unsafe { std::mem::transmute(code.as_ptr()) };
        for raw in [
            0x7ff9_0000_0000_1234,
            0xfff9_0000_0000_1234,
            0xffff_ffff_ffff_ffff,
            0x7ff0_0000_0000_0001,
            (-f64::NAN).to_bits(),
        ] {
            let mut output = 0;
            unsafe { run(raw, &mut output) };
            assert_eq!(output, PACK_CANON_NAN);
        }
    }
}
