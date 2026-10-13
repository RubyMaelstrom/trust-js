//! A64 encodings of the instructions the inline Math intrinsics and Uint8Clamped stores use, against the Arm ARM
//! (100076_0100): FRINTM/FRINTP/FABS (C7.2), FMIN/FMAX (C7.2.51-52), CLZ (C6.2.57), MUL (MADD
//! with WZR, C6.2.194). The expected words are the assembler's output for the same operands.

use super::asm;

fn word(build: impl FnOnce(&mut asm::Asm)) -> u32 {
    let mut a = asm::Asm::new();
    build(&mut a);
    let words = a.finish();
    assert_eq!(words.len(), 1);
    words[0]
}

#[test]
fn math_intrinsic_instruction_encodings() {
    assert_eq!(word(|a| a.frintm(3, 5)), 0x1e65_40a3); // frintm d3, d5
    assert_eq!(word(|a| a.frintp(3, 5)), 0x1e64_c0a3); // frintp d3, d5
    assert_eq!(word(|a| a.fabs(3, 5)), 0x1e60_c0a3); // fabs d3, d5
    assert_eq!(word(|a| a.fminmax(false, 3, 5, 7)), 0x1e67_58a3); // fmin d3, d5, d7
    assert_eq!(word(|a| a.fminmax(true, 3, 5, 7)), 0x1e67_48a3); // fmax d3, d5, d7
    assert_eq!(word(|a| a.clz_w(3, 5)), 0x5ac0_10a3); // clz w3, w5
    assert_eq!(word(|a| a.mul_w(3, 5, 7)), 0x1b07_7ca3); // mul w3, w5, w7
    assert_eq!(word(|a| a.fcvtns_w_d(3, 5)), 0x1e60_00a3); // fcvtns w3, d5 (Uint8Clamped stores)
}
