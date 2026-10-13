//! Exact integer constant materialization, independent of JS value interpretation.
//! Arm 100076_0100_00_en D2.106–108: MOVK changes one 16-bit lane; MOVZ/MOVN
//! initialize every bit. ECMA-262 e28783d5 Number semantics require preserving
//! signed zero/infinities; this assembler also preserves arbitrary NaN/pointer bits.

use super::{asm, ExecutableBuffer};

fn words(register: u32, value: u64) -> Vec<u32> {
    let mut assembler = asm::Asm::new();
    assembler.mov_imm64(register, value);
    assembler.finish()
}

// Decode the architectural instruction fields rather than reusing emitter helpers.
fn emulate(code: &[u32], register: u32, initial: u64) -> u64 {
    let mut value = initial;
    for (index, &word) in code.iter().enumerate() {
        assert_eq!(word & 31, register, "wrong destination: {word:08x}");
        let shift = ((word >> 21) & 3) * 16;
        let immediate = u64::from((word >> 5) & 0xffff) << shift;
        match word & 0xff80_0000 {
            0xd280_0000 => {
                assert_eq!(index, 0, "MOVZ must initialize, not discard earlier lanes");
                value = immediate;
            }
            0x9280_0000 => {
                assert_eq!(index, 0, "MOVN must initialize, not discard earlier lanes");
                value = !immediate;
            }
            0xf280_0000 => {
                assert_ne!(index, 0, "MOVK must not depend on incoming register bits");
                value = (value & !(0xffff_u64 << shift)) | immediate;
            }
            _ => panic!("not a 64-bit move-wide instruction: {word:08x}"),
        }
    }
    if register == 31 {
        0
    } else {
        value
    } // XZR discards writes, not SP.
}

fn check(register: u32, value: u64) {
    let code = words(register, value);
    let zeros = (0..4)
        .filter(|lane| value >> (lane * 16) & 0xffff == 0)
        .count();
    let ones = (0..4)
        .filter(|lane| value >> (lane * 16) & 0xffff == 0xffff)
        .count();
    // A base instruction establishes one arbitrary lane plus three all-zero/all-one lanes.
    // Each remaining non-default lane requires a MOVK: a lower bound achieved by this emitter.
    assert_eq!(
        code.len(),
        (4 - zeros.max(ones)).max(1),
        "nonminimal {value:016x}"
    );
    let expected = if register == 31 { 0 } else { value };
    assert_eq!(emulate(&code, register, 0), expected, "{value:016x}");
    assert_eq!(emulate(&code, register, u64::MAX), expected, "{value:016x}");
}

#[test]
fn mov_imm64_exhaustive_lanes_and_sparse_patterns_are_exact_and_minimal() {
    for lane in 0..4 {
        for bits in 0..=u16::MAX {
            let value = u64::from(bits) << (lane * 16);
            check(9, value);
            check(16, !value);
        }
    }
    let patterns = [0_u64, 0xffff, 1, 0x8000, 0x1234, 0xabcd];
    for a in patterns {
        for b in patterns {
            for c in patterns {
                for d in patterns {
                    check(0, a | b << 16 | c << 32 | d << 48);
                }
            }
        }
    }
}

fn representative_values() -> Vec<u64> {
    use crate::value::*;
    let mut values = vec![
        0,
        1,
        u64::MAX,
        u64::MAX - 1,
        0x1234_0000_5678_0000,
        0xffff_1234_ffff_5678,
        0x1234_5678_9abc_def0,
        0x0000_ffff_ffff_ffff,
        0x0000_1234_5678_9abc,
        0.0_f64.to_bits(),
        (-0.0_f64).to_bits(),
        1.0_f64.to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        f64::MAX.to_bits(),
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        0x7ff0_0000_0000_0001,
        0xffff_ffff_ffff_ffff,
        PACK_CANON_NAN,
        PACK_UNDEFINED,
        PACK_EMPTY,
        PACK_NULL,
        PACK_BOOL,
        PACK_BOOL | 1,
        PACK_BIGINT,
        PACK_STR,
        PACK_SYM,
        PACK_OBJ,
        PACK_LAZY_PROTO,
    ];
    for tag in [PACK_BIGINT, PACK_STR, PACK_SYM, PACK_OBJ] {
        values.push(tag | 0x1234_5678_9abc);
    }
    // Stable, dependency-free bit coverage; no timing or OS entropy controls the test.
    let mut random = 0x5b1c_983d_4e72_a601_u64;
    for _ in 0..8192 {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        values.push(random);
    }
    values
}

#[test]
fn mov_imm64_all_registers_and_random_words_preserve_all_bits() {
    for (index, value) in representative_values().into_iter().enumerate() {
        check(index as u32 % 32, value);
    }
    for register in 0..32 {
        for value in [0, u64::MAX, 0x0000_abcd_0000_0000, 0xffff_ffff_0000_ffff] {
            check(register, value);
        }
    }
}

#[test]
fn mov_imm64_generated_native_functions_preserve_bits_and_condition_flags() {
    let values = representative_values();
    let mut bytes = Vec::new();
    let mut offsets = Vec::new();
    for &value in &values {
        offsets.push(bytes.len());
        let mut assembler = asm::Asm::new();
        // Return the word in x0 and a pre-existing EQ flag through the output pointer x1.
        assembler.cmp_reg_x(0, 0);
        assembler.mov_imm64(0, value);
        assembler.cset_w(2, 0); // EQ must survive every MOVZ/MOVN/MOVK instruction.
        assembler.str_w_imm(2, 1, 0);
        assembler.ret();
        bytes.extend(assembler.finish().into_iter().flat_map(u32::to_le_bytes));
    }
    let code = ExecutableBuffer::from_bytes(&bytes).expect("native constant test mapping");
    for (offset, value) in offsets.into_iter().zip(values) {
        let run: unsafe extern "C" fn(u64, *mut u32) -> u64 =
            unsafe { std::mem::transmute(code.as_ptr().add(offset)) };
        let mut flag = 0;
        assert_eq!(unsafe { run(!value, &mut flag) }, value, "{value:016x}");
        assert_eq!(flag, 1, "condition flags changed for {value:016x}");
    }
}
