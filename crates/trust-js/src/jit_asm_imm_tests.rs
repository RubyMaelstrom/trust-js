//! Decision 23: the ARM64 encoders take any offset or immediate. An operand an instruction's
//! own form encodes gives the word it always did; one past it is legalized (`asm::Asm::mem`,
//! `pair`, `add_far`, `flags`). These tests check both against llvm-mc's encodings and run the
//! legalized sequences on the CPU.

use super::asm::{legalized, Asm};
use super::ExecutableBuffer;

fn words(emit: impl FnOnce(&mut Asm)) -> Vec<u32> {
    let mut a = Asm::new();
    emit(&mut a);
    assert_eq!(a.failed(), None);
    a.finish()
}

fn executable(a: Asm) -> ExecutableBuffer {
    let bytes: Vec<u8> = a.finish().into_iter().flat_map(u32::to_le_bytes).collect();
    ExecutableBuffer::from_bytes(&bytes).unwrap()
}

#[test]
fn operands_in_range_encode_as_before() {
    let before = legalized();
    let scaled =
        |op: u32, size: u32, off: u32, rn: u32, rt: u32| op | (off / size) << 10 | rn << 5 | rt;
    let unscaled =
        |op: u32, off: i32, rn: u32, rt: u32| op | ((off as u32) & 0x1FF) << 12 | rn << 5 | rt;
    for (rt, rn) in [(9, 22), (0, 31), (16, 17), (22, 22), (31, 9)] {
        for off in [0, 8, 4088, 4096, 32752, 32760] {
            assert_eq!(
                words(|a| a.ldr_imm(rt, rn, off)),
                [scaled(0xF940_0000, 8, off, rn, rt)]
            );
            assert_eq!(
                words(|a| a.str_imm(rt, rn, off)),
                [scaled(0xF900_0000, 8, off, rn, rt)]
            );
            assert_eq!(
                words(|a| a.ldr_d_imm(rt, rn, off)),
                [scaled(0xFD40_0000, 8, off, rn, rt)]
            );
            assert_eq!(
                words(|a| a.str_d_imm(rt, rn, off)),
                [scaled(0xFD00_0000, 8, off, rn, rt)]
            );
        }
        for off in [0, 4, 16376, 16380] {
            assert_eq!(
                words(|a| a.ldr_w_imm(rt, rn, off)),
                [scaled(0xB940_0000, 4, off, rn, rt)]
            );
            assert_eq!(
                words(|a| a.str_w_imm(rt, rn, off)),
                [scaled(0xB900_0000, 4, off, rn, rt)]
            );
        }
        for off in [0, 2, 8190] {
            assert_eq!(
                words(|a| a.ldrh_imm(rt, rn, off)),
                [scaled(0x7940_0000, 2, off, rn, rt)]
            );
        }
        for off in [0, 1, 4095] {
            assert_eq!(
                words(|a| a.ldrb_imm(rt, rn, off)),
                [scaled(0x3940_0000, 1, off, rn, rt)]
            );
            assert_eq!(
                words(|a| a.strb_imm(rt, rn, off)),
                [scaled(0x3900_0000, 1, off, rn, rt)]
            );
        }
        for off in [-256, -255, -8, -1, 0, 8, 254, 255] {
            assert_eq!(
                words(|a| a.ldur(rt, rn, off)),
                [unscaled(0xF840_0000, off, rn, rt)]
            );
            assert_eq!(
                words(|a| a.stur(rt, rn, off)),
                [unscaled(0xF800_0000, off, rn, rt)]
            );
            assert_eq!(
                words(|a| a.ldur_d(rt, rn, off)),
                [unscaled(0xFC40_0000, off, rn, rt)]
            );
            assert_eq!(
                words(|a| a.stur_d(rt, rn, off)),
                [unscaled(0xFC00_0000, off, rn, rt)]
            );
            assert_eq!(
                words(|a| a.ldurb(rt, rn, off)),
                [unscaled(0x3840_0000, off, rn, rt)]
            );
            assert_eq!(
                words(|a| a.sturh(rt, rn, off)),
                [unscaled(0x7800_0000, off, rn, rt)]
            );
        }
    }
    for imm in [0, 1, 4094, 4095] {
        let field = imm << 10;
        assert_eq!(
            words(|a| a.add_imm(9, 31, imm)),
            [0x9100_0000 | field | 31 << 5 | 9]
        );
        assert_eq!(
            words(|a| a.add_imm(31, 31, imm)),
            [0x9100_0000 | field | 31 << 5 | 31]
        );
        assert_eq!(
            words(|a| a.sub_imm(20, 20, imm)),
            [0xD100_0000 | field | 20 << 5 | 20]
        );
        assert_eq!(
            words(|a| a.cmp_imm_w(9, imm)),
            [0x7100_001F | field | 9 << 5]
        );
        assert_eq!(
            words(|a| a.cmp_imm_x(9, imm)),
            [0xF100_001F | field | 9 << 5]
        );
        assert_eq!(
            words(|a| a.cmn_imm_w(9, imm)),
            [0x3100_001F | field | 9 << 5]
        );
        assert_eq!(
            words(|a| a.cmn_imm_x(9, imm)),
            [0xB100_001F | field | 9 << 5]
        );
        assert_eq!(
            words(|a| a.adds_imm_w(9, 10, imm)),
            [0x3100_0000 | field | 10 << 5 | 9]
        );
        assert_eq!(
            words(|a| a.subs_imm_w(28, 28, imm)),
            [0x7100_0000 | field | 28 << 5 | 28]
        );
    }
    for off in [-512, -504, -8, 0, 8, 496, 504] {
        let field = ((off / 8) as u32 & 0x7F) << 15;
        let pair = |op: u32, rt1: u32, rt2: u32, rn: u32| op | field | rt2 << 10 | rn << 5 | rt1;
        assert_eq!(
            words(|a| a.stp_pre(29, 30, off)),
            [pair(0xA980_0000, 29, 30, 31)]
        );
        assert_eq!(
            words(|a| a.ldp_post(29, 30, off)),
            [pair(0xA8C0_0000, 29, 30, 31)]
        );
        assert_eq!(
            words(|a| a.stp_off(19, 20, off)),
            [pair(0xA900_0000, 19, 20, 31)]
        );
        assert_eq!(
            words(|a| a.ldp_off(19, 20, off)),
            [pair(0xA940_0000, 19, 20, 31)]
        );
        assert_eq!(
            words(|a| a.stp_base(9, 10, 22, off)),
            [pair(0xA900_0000, 9, 10, 22)]
        );
        assert_eq!(
            words(|a| a.ldp_base(9, 10, 9, off)),
            [pair(0xA940_0000, 9, 10, 9)]
        );
        assert_eq!(
            words(|a| a.stp_d_off(8, 9, off)),
            [pair(0x6D00_0000, 8, 9, 31)]
        );
        assert_eq!(
            words(|a| a.ldp_d_off(16, 17, off)),
            [pair(0x6D40_0000, 16, 17, 31)]
        );
        assert_eq!(
            words(|a| a.ldp_d_restore(8, 9, off)),
            [pair(0x6D40_0000, 8, 9, 31)]
        );
    }
    assert_eq!(
        words(|a| a.movz(9, 0xFFFF, 3)),
        [0xD2E0_0000 | 0xFFFF << 5 | 9]
    );
    assert_eq!(
        words(|a| a.movk(9, 0xFFFF, 3)),
        [0xF2E0_0000 | 0xFFFF << 5 | 9]
    );
    assert_eq!(
        words(|a| a.lsl_imm(9, 10, 63)),
        [0xD340_0000 | 1 << 16 | 10 << 5 | 9]
    );
    assert_eq!(
        words(|a| a.lsr_imm(9, 10, 63)),
        [0xD340_FC00 | 63 << 16 | 10 << 5 | 9]
    );
    assert_eq!(
        words(|a| a.add_shifted(9, 10, 11, 63)),
        [0x8B00_0000 | 11 << 16 | 63 << 10 | 10 << 5 | 9]
    );
    assert_eq!(
        words(|a| a.asr_imm_w(9, 10, 31)),
        [0x1300_7C00 | 31 << 16 | 10 << 5 | 9]
    );
    assert_eq!(
        words(|a| a.lsr_imm_w(9, 10, 31)),
        [0x5300_7C00 | 31 << 16 | 10 << 5 | 9]
    );
    assert_eq!(
        words(|a| a.lsl_imm_w(9, 10, 31)),
        [0x5300_0000 | 1 << 16 | 10 << 5 | 9]
    );
    assert_eq!(
        words(|a| a.logic_imm_w(0, 9, 10, 0xFFF)),
        [0x1200_0000 | 0xFFF << 10 | 10 << 5 | 9]
    );
    assert_eq!(legalized(), before, "no operand in range is legalized");
}

/// Each legalized form, as llvm-mc 23 encodes its instructions
/// (`llvm-mc -triple=aarch64 -show-encoding`).
#[test]
fn legalized_forms_are_the_intended_instructions() {
    let cases: Vec<(Vec<u32>, &[u32], &str)> = vec![
        // A load: through its destination, the offset's high part added (rung 2).
        (words(|a| a.ldr_imm(9, 22, 32768)), &[0x9140_22C9, 0xF940_0129], "add x9, x22, #8, lsl #12; ldr x9, [x9]"),
        (words(|a| a.ldr_imm(9, 22, 40000)), &[0x9140_26C9, 0xF946_2129], "add x9, x22, #9, lsl #12; ldr x9, [x9, #3136]"),
        (words(|a| a.ldr_imm(22, 22, 40000)), &[0x9140_26D6, 0xF946_22D6], "add x22, x22, #9, lsl #12; ldr x22, [x22, #3136]"),
        (words(|a| a.ldur(9, 20, -264)), &[0xD140_0689, 0xF947_7D29], "sub x9, x20, #1, lsl #12; ldr x9, [x9, #3832]"),
        (words(|a| a.ldrb_imm(9, 22, 4096)), &[0x9140_06C9, 0x3940_0129], "add x9, x22, #1, lsl #12; ldrb w9, [x9]"),
        (words(|a| a.ldrh_imm(9, 22, 8192)), &[0x9140_0AC9, 0x7940_0129], "add x9, x22, #2, lsl #12; ldrh w9, [x9]"),
        // The other one-instruction form (rung 1).
        (words(|a| a.ldr_imm(9, 22, 4)), &[0xF840_42C9], "ldur x9, [x22, #4]"),
        (words(|a| a.ldur(9, 22, 256)), &[0xF940_82C9], "ldr x9, [x22, #256]"),
        (words(|a| a.ldurb(9, 22, -1)), &[0x385F_F2C9], "ldurb w9, [x22, #-1]"),
        // A load whose offset splits neither way: the whole offset in the destination.
        (words(|a| a.ldr_imm(9, 22, 1001)), &[0xD280_7D29, 0xF869_6AC9], "mov x9, #1001; ldr x9, [x22, x9]"),
        (words(|a| a.ldr_imm(9, 22, 1 << 24)), &[0xD2A0_2009, 0xF869_6AC9], "mov x9, #16777216; ldr x9, [x22, x9]"),
        // A store: the base moved and moved back.
        (words(|a| a.str_imm(9, 22, 40000)), &[0x9140_26D6, 0xF906_22C9, 0xD140_26D6], "add x22, x22, #9, lsl #12; str x9, [x22, #3136]; sub x22, x22, #9, lsl #12"),
        (words(|a| a.stur(9, 20, -264)), &[0xD140_0694, 0xF907_7E89, 0x9140_0694], "sub x20, x20, #1, lsl #12; str x9, [x20, #3832]; add x20, x20, #1, lsl #12"),
        (words(|a| a.str_d_imm(0, 22, 40000)), &[0x9140_26D6, 0xFD06_22C0, 0xD140_26D6], "add x22, x22, #9, lsl #12; str d0, [x22, #3136]; sub x22, x22, #9, lsl #12"),
        (words(|a| a.strb_imm(9, 22, 4096)), &[0x9140_06D6, 0x3900_02C9, 0xD140_06D6], "add x22, x22, #1, lsl #12; strb w9, [x22]; sub x22, x22, #1, lsl #12"),
        // A store of its own base, or based on SP: a recycled register (rung 3).
        (words(|a| a.str_imm(22, 22, 40000)), &[0xF81F_0FF0, 0x9140_26D0, 0xF906_2216, 0xF841_07F0], "str x16, [sp, #-16]!; add x16, x22, #9, lsl #12; str x22, [x16, #3136]; ldr x16, [sp], #16"),
        (words(|a| a.str_imm(16, 31, 40000)), &[0xF81F_0FF1, 0x9140_27F1, 0xF906_2A30, 0xF841_07F1], "str x17, [sp, #-16]!; add x17, sp, #9, lsl #12; str x16, [x17, #3152]; ldr x17, [sp], #16"),
        (words(|a| a.ldr_d_imm(0, 31, 40000)), &[0xF81F_0FF0, 0x9140_27F0, 0xFD46_2A00, 0xF841_07F0], "str x16, [sp, #-16]!; add x16, sp, #9, lsl #12; ldr d0, [x16, #3152]; ldr x16, [sp], #16"),
        // Adds and subtracts through their destination.
        (words(|a| a.add_imm(20, 20, 4096)), &[0x9140_0694], "add x20, x20, #1, lsl #12"),
        (words(|a| a.add_imm(20, 20, 4100)), &[0x9140_0694, 0x9100_1294], "add x20, x20, #1, lsl #12; add x20, x20, #4"),
        (words(|a| a.sub_imm(31, 31, 5000)), &[0xD140_07FF, 0xD10E_23FF], "sub sp, sp, #1, lsl #12; sub sp, sp, #904"),
        (words(|a| a.add_imm(9, 10, 1 << 24)), &[0xD2A0_2009, 0x8B09_0149], "mov x9, #16777216; add x9, x10, x9"),
        (words(|a| a.add_imm(9, 31, 1 << 24)), &[0xD2A0_2009, 0x8B29_63E9], "mov x9, #16777216; add x9, sp, x9"),
        (words(|a| a.add_imm(9, 9, 1 << 24)), &[0xF81F_0FF0, 0xD2A0_2010, 0x8B10_0129, 0xF841_07F0], "str x16, [sp, #-16]!; mov x16, #16777216; add x9, x9, x16; ldr x16, [sp], #16"),
        // Flag-setting immediates: shifted by 12, through the destination, or recycled.
        (words(|a| a.cmp_imm_w(1, 4096)), &[0x7140_043F], "cmp w1, #1, lsl #12"),
        (words(|a| a.cmp_imm_w(1, 5000)), &[0xF81F_0FF0, 0xD282_7110, 0x6B10_003F, 0xF841_07F0], "str x16, [sp, #-16]!; mov x16, #5000; cmp w1, w16; ldr x16, [sp], #16"),
        (words(|a| a.cmp_imm_w(16, 5000)), &[0xF81F_0FF1, 0xD282_7111, 0x6B11_021F, 0xF841_07F1], "str x17, [sp, #-16]!; mov x17, #5000; cmp w16, w17; ldr x17, [sp], #16"),
        (words(|a| a.subs_imm_w(9, 10, 5000)), &[0xD282_7109, 0x6B09_0149], "mov x9, #5000; subs w9, w10, w9"),
        (words(|a| a.subs_imm_w(9, 9, 5000)), &[0xF81F_0FF0, 0xD282_7110, 0x6B10_0129, 0xF841_07F0], "str x16, [sp, #-16]!; mov x16, #5000; subs w9, w9, w16; ldr x16, [sp], #16"),
        (words(|a| a.cmn_imm_x(9, 0x12_3456)), &[0xF81F_0FF0, 0xD286_8AD0, 0xF2A0_0250, 0xAB10_013F, 0xF841_07F0], "str x16, [sp, #-16]!; mov x16, #13398; movk x16, #18, lsl #16; cmn x9, x16; ldr x16, [sp], #16"),
        // Pairs.
        (words(|a| a.ldp_off(9, 10, 1024)), &[0x9110_03E9, 0xA940_2929], "add x9, sp, #1024; ldp x9, x10, [x9]"),
        (words(|a| a.ldp_base(9, 10, 9, 4096)), &[0x9140_0529, 0xA940_2929], "add x9, x9, #1, lsl #12; ldp x9, x10, [x9]"),
        (words(|a| a.ldp_base(9, 10, 10, 4096)), &[0x9140_0549, 0xA940_2929], "add x9, x10, #1, lsl #12; ldp x9, x10, [x9]"),
        (words(|a| a.stp_base(9, 10, 22, 1024)), &[0x9110_02D6, 0xA900_2AC9, 0xD110_02D6], "add x22, x22, #1024; stp x9, x10, [x22]; sub x22, x22, #1024"),
        (words(|a| a.stp_base(9, 10, 22, 4096 + 16)), &[0x9140_06D6, 0xA901_2AC9, 0xD140_06D6], "add x22, x22, #1, lsl #12; stp x9, x10, [x22, #16]; sub x22, x22, #1, lsl #12"),
        (words(|a| a.stp_off(29, 30, 1024)), &[0xF81F_0FF0, 0x9110_43F0, 0xA900_7A1D, 0xF841_07F0], "str x16, [sp, #-16]!; add x16, sp, #1040; stp x29, x30, [x16]; ldr x16, [sp], #16"),
        (words(|a| a.stp_d_off(8, 9, 1024)), &[0xF81F_0FF0, 0x9110_43F0, 0x6D00_2608, 0xF841_07F0], "str x16, [sp, #-16]!; add x16, sp, #1040; stp d8, d9, [x16]; ldr x16, [sp], #16"),
        (words(|a| a.stp_pre(29, 30, -1024)), &[0xD110_03FF, 0xA900_7BFD], "sub sp, sp, #1024; stp x29, x30, [sp]"),
        (words(|a| a.ldp_post(29, 30, 1024)), &[0xA940_7BFD, 0x9110_03FF], "ldp x29, x30, [sp]; add sp, sp, #1024"),
    ];
    for (emitted, expected, text) in cases {
        assert_eq!(emitted, expected, "{text}");
    }
}

#[test]
fn legalizing_is_counted_only_past_the_form() {
    let before = legalized();
    words(|a| a.ldr_imm(9, 22, 32760));
    assert_eq!(legalized(), before);
    words(|a| a.ldr_imm(9, 22, 32768));
    assert_eq!(legalized(), before + 1);
    // The other one-instruction form is no legalization either.
    words(|a| a.ldr_imm(9, 22, 4));
    assert_eq!(legalized(), before + 1);
}

/// What no sequence encodes fails the assembly, which the compilers then refuse: an SP compared
/// with an immediate past 12 bits (the register form reads register 31 as XZR) and SP moved by
/// 16 MB or more. A rewound emission takes its failure with it.
#[test]
fn operands_no_sequence_encodes_fail_the_assembly() {
    for emit in [
        (|a: &mut Asm| a.cmp_imm_x(31, 5000)) as fn(&mut Asm),
        |a| a.adds_imm_w(9, 31, 5000),
        |a| a.add_imm(31, 31, 1 << 24),
        |a| a.sub_imm(31, 9, 1 << 24),
        |a| a.stp_pre(29, 30, -(1 << 24)),
        |a| a.ldp_post(29, 30, 1 << 24),
    ] {
        let mut a = Asm::new();
        a.movz(0, 1, 0);
        emit(&mut a);
        assert!(a.failed().is_some());
        assert!(a.buf.contains(&0), "UDF #0 in the failing operand's place");
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| a.finish())).is_err());
    }
    let mut a = Asm::new();
    a.movz(0, 1, 0);
    let checkpoint = a.checkpoint();
    a.cmp_imm_x(31, 5000);
    assert!(a.failed().is_some());
    a.rewind(checkpoint);
    assert_eq!(
        a.failed(),
        None,
        "the failure went with the rewound emission"
    );
    a.cmp_imm_x(31, 5000);
    let checkpoint = a.checkpoint();
    a.movz(1, 2, 0);
    a.rewind(checkpoint);
    assert!(
        a.failed().is_some(),
        "a failure before the checkpoint stays"
    );
}

/// Field widths out of range (a shift, a MOVZ lane, a prepared logical field) are template
/// bugs: debug builds report them, release builds fail the assembly.
#[test]
fn field_widths_out_of_range_are_reported() {
    for emit in [
        (|a: &mut Asm| a.lsl_imm(0, 0, 64)) as fn(&mut Asm),
        |a| a.lsr_imm(0, 0, 64),
        |a| a.add_shifted(0, 0, 1, 64),
        |a| a.asr_imm_w(0, 0, 32),
        |a| a.lsr_imm_w(0, 0, 32),
        |a| a.lsl_imm_w(0, 0, 32),
        |a| a.movz(0, 0x1_0000, 0),
        |a| a.movz(0, 1, 4),
        |a| a.movk(0, 0x1_0000, 0),
        |a| a.logic_imm_w(0, 0, 0, 1 << 12),
    ] {
        let mut a = Asm::new();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| emit(&mut a)));
        if cfg!(debug_assertions) {
            assert!(caught.is_err());
        } else {
            assert!(caught.is_ok() && a.failed().is_some());
        }
    }
}

// ---- the legalized sequences on the CPU ----

/// The registers a probe runs with and leaves: x0..x28 (x18, the platform register, untouched),
/// d0 and d1, SP (the probe's own stack) and NZCV.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
struct Machine {
    x: [u64; 29],
    d: [u64; 2],
    sp: u64,
    nzcv: u64,
}

const D_OFF: u32 = 29 * 8;
const SP_OFF: u32 = 31 * 8;
const NZCV_OFF: u32 = 32 * 8;

/// x0..x28 but x18, in loading pairs.
const LOADED: [(u32, u32); 13] = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
];

/// Append a probe: `extern "C" fn(*mut Machine)` running `emit` with the machine's registers,
/// SP and NZCV, and storing them back. x29 keeps the probe's frame and x30 serves the
/// epilogue, so `emit` names neither. Returns the probe's byte offset.
fn append_probe(a: &mut Asm, emit: impl FnOnce(&mut Asm)) -> usize {
    let start = a.buf.len() * 4;
    a.stp_pre(29, 30, -112);
    a.add_imm(29, 31, 0);
    for (k, r) in [19, 21, 23, 25, 27].into_iter().enumerate() {
        a.stp_base(r, r + 1, 29, 16 + 16 * k as i32);
    }
    a.str_imm(0, 29, 96);
    a.ldr_imm(30, 0, SP_OFF);
    a.add_imm(31, 30, 0); // mov sp, x30
    a.ldr_imm(30, 0, NZCV_OFF);
    a.word(0xD51B_4200 | 30); // msr nzcv, x30
    for (r1, r2) in LOADED {
        a.ldp_base(r1, r2, 0, r1 as i32 * 8);
    }
    a.ldr_imm(17, 0, 17 * 8);
    a.ldr_d_imm(0, 0, D_OFF);
    a.ldr_d_imm(1, 0, D_OFF + 8);
    a.ldr_imm(0, 0, 0);
    emit(a);
    a.ldr_imm(30, 29, 96);
    a.str_imm(0, 30, 0);
    for (r1, r2) in LOADED {
        a.stp_base(r1, r2, 30, r1 as i32 * 8);
    }
    a.str_imm(17, 30, 17 * 8);
    a.str_d_imm(0, 30, D_OFF);
    a.str_d_imm(1, 30, D_OFF + 8);
    a.add_imm(0, 31, 0); // mov x0, sp
    a.str_imm(0, 30, SP_OFF);
    a.word(0xD53B_4200); // mrs x0, nzcv
    a.str_imm(0, 30, NZCV_OFF);
    a.add_imm(31, 29, 0); // mov sp, x29
    for (k, r) in [19, 21, 23, 25, 27].into_iter().enumerate() {
        a.ldp_base(r, r + 1, 29, 16 + 16 * k as i32);
    }
    a.ldp_post(29, 30, 112);
    a.ret();
    start
}

fn run(code: &ExecutableBuffer, offset: usize, machine: &mut Machine) {
    let probe: unsafe extern "C" fn(*mut Machine) =
        unsafe { std::mem::transmute(code.as_ptr().add(offset)) };
    unsafe { probe(machine) };
}

/// A 16-byte-aligned region of memory with a middle that bases point at.
struct Region {
    bytes: Vec<u8>,
    middle: usize,
}

impl Region {
    fn new(span: usize) -> Region {
        let bytes = vec![0u8; 2 * span + 64];
        let middle = (bytes.as_ptr() as usize + span + 15) & !15;
        Region { bytes, middle }
    }
    fn at(&self, address: u64) -> usize {
        (address as usize) - self.bytes.as_ptr() as usize
    }
}

const NZCV_IN: u64 = 0xA000_0000; // N and C set

fn machine(stack: &Region) -> Machine {
    let mut machine = Machine {
        x: [0; 29],
        d: [0x4045_0000_0000_0000, 0x4046_0000_0000_0000],
        sp: stack.middle as u64,
        nzcv: NZCV_IN,
    };
    for (r, value) in machine.x.iter_mut().enumerate() {
        *value = 0x5A5A_0000_0000_0000 | (r as u64) << 32 | 0xC0DE_0000 | r as u64;
    }
    machine
}

#[derive(Clone, Copy, Debug)]
enum Access {
    LdrX,
    StrX,
    LdrW,
    StrW,
    Ldrh,
    Ldurb,
    Strb,
    Sturh,
    LdrD,
    StrD,
}

impl Access {
    fn size(self) -> usize {
        match self {
            Access::LdrX | Access::StrX | Access::LdrD | Access::StrD => 8,
            Access::LdrW | Access::StrW => 4,
            Access::Ldrh | Access::Sturh => 2,
            Access::Ldurb | Access::Strb => 1,
        }
    }
    fn load(self) -> bool {
        matches!(
            self,
            Access::LdrX | Access::LdrW | Access::Ldrh | Access::Ldurb | Access::LdrD
        )
    }
    fn fp(self) -> bool {
        matches!(self, Access::LdrD | Access::StrD)
    }
    fn emit(self, a: &mut Asm, rt: u32, rn: u32, off: i64) {
        match self {
            Access::LdrX => a.ldr_imm(rt, rn, off as u32),
            Access::StrX => a.str_imm(rt, rn, off as u32),
            Access::LdrW => a.ldr_w_imm(rt, rn, off as u32),
            Access::StrW => a.str_w_imm(rt, rn, off as u32),
            Access::Ldrh => a.ldrh_imm(rt, rn, off as u32),
            Access::Ldurb => a.ldurb(rt, rn, off as i32),
            Access::Strb => a.strb_imm(rt, rn, off as u32),
            Access::Sturh => a.sturh(rt, rn, off as i32),
            Access::LdrD => a.ldr_d_imm(rt, rn, off as u32),
            Access::StrD => a.str_d_imm(rt, rn, off as u32),
        }
    }
    /// The encoder takes an unsigned offset.
    fn unsigned(self) -> bool {
        !matches!(self, Access::Ldurb | Access::Sturh)
    }
}

/// Offsets at and past every boundary of `size`'s forms, within `span`.
fn offsets(size: i64, span: i64, unsigned: bool) -> Vec<i64> {
    let mut offsets = vec![
        0,
        size,
        255,
        256,
        4088,
        4095,
        4096,
        4096 * size - size,
        4096 * size,
        4096 * size + size,
        4096 * size + 1,
        40000,
        40001,
        65536 - size,
        (1 << 20) + 4093,
        span - 16,
    ];
    if !unsigned {
        offsets.extend([
            -256,
            -257,
            -264,
            -4095,
            -4096,
            -4104,
            -40000,
            -40001,
            -(span - 16),
        ]);
    }
    offsets.retain(|off| off.unsigned_abs() as i64 <= span - 16);
    offsets
}

/// Every single-register access, at offsets at and past each boundary, with bases and
/// transfer registers in every role the ladder distinguishes (the base stored, a recycling
/// candidate named, SP), run on the CPU: the access lands at base + offset with its size, the
/// loaded value reaches the destination, and every other register, SP and NZCV is unchanged.
#[test]
fn legalized_accesses_run_on_the_cpu() {
    const SPAN: usize = 1 << 21;
    let mut data = Region::new(SPAN);
    let mut stack = Region::new(SPAN);
    let accesses = [
        Access::LdrX,
        Access::StrX,
        Access::LdrW,
        Access::StrW,
        Access::Ldrh,
        Access::Ldurb,
        Access::Strb,
        Access::Sturh,
        Access::LdrD,
        Access::StrD,
    ];
    let roles: &[(u32, u32)] = &[
        (9, 22),
        (22, 22),
        (16, 22),
        (17, 22),
        (9, 16),
        (16, 17),
        (17, 16),
        (15, 16),
        (14, 15),
        (9, 31),
        (16, 31),
        (17, 31),
        (0, 1),
        (28, 0),
    ];
    let mut a = Asm::new();
    let mut cases = Vec::new();
    for access in accesses {
        for &(rt, rn) in roles {
            let rt = if access.fp() { rt % 2 } else { rt };
            for off in offsets(access.size() as i64, SPAN as i64, access.unsigned()) {
                // An access of the probe's own stack below SP would meet the recycled push.
                if rn == 31 && off < 16 {
                    continue;
                }
                let at = append_probe(&mut a, |a| access.emit(a, rt, rn, off));
                cases.push((at, access, rt, rn, off));
            }
        }
    }
    assert_eq!(a.failed(), None);
    let code = executable(a);
    // Loads read a pattern that names each byte's address; stores write into zeros, which a
    // comparison with `zeros` scans for stray writes.
    let pattern = |k: usize| ((k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 56) as u8 | 1;
    let zeros = vec![0u8; data.bytes.len()];
    for loads in [true, false] {
        for region in [&mut data, &mut stack] {
            for (k, byte) in region.bytes.iter_mut().enumerate() {
                *byte = if loads { pattern(k) } else { 0 };
            }
        }
        for &(at, access, rt, rn, off) in cases.iter().filter(|case| case.1.load() == loads) {
            let mut before = machine(&stack);
            if rn != 31 {
                before.x[rn as usize] = data.middle as u64;
            }
            let base = if rn == 31 {
                before.sp
            } else {
                data.middle as u64
            };
            let address = base.wrapping_add(off as u64);
            let mut after = before;
            let what = format!("{access:?} rt {rt} rn {rn} off {off}");
            run(&code, at, &mut after);
            let size = access.size();
            let mut expected = before;
            let region = if rn == 31 { &mut stack } else { &mut data };
            let from = region.at(address);
            if loads {
                let mut value = [0u8; 8];
                value[..size].copy_from_slice(&region.bytes[from..from + size]);
                let value = u64::from_le_bytes(value);
                if access.fp() {
                    expected.d[rt as usize] = value;
                } else {
                    expected.x[rt as usize] = value;
                }
            } else {
                let value = if access.fp() {
                    before.d[rt as usize]
                } else {
                    before.x[rt as usize]
                };
                assert_eq!(
                    region.bytes[from..from + size],
                    value.to_le_bytes()[..size],
                    "{what}"
                );
                region.bytes[from..from + size].fill(0);
                // A recycled register's push, just below the probe's SP, is the only other write.
                let push = stack.at(before.sp - 16);
                stack.bytes[push..push + 16].fill(0);
                assert!(
                    data.bytes == zeros && stack.bytes == zeros,
                    "{what}: a stray write"
                );
            }
            assert_eq!(after, expected, "{what}");
        }
    }
}

/// Pairs: signed offsets past imm7 with every base and register role, and pre- and
/// post-indexed SP moves past it, run on the CPU.
#[test]
fn legalized_pairs_run_on_the_cpu() {
    const SPAN: usize = 1 << 21;
    let mut data = Region::new(SPAN);
    let mut stack = Region::new(SPAN);
    #[derive(Clone, Copy, Debug)]
    enum Kind {
        Ldp,
        Stp,
        LdpD,
        StpD,
    }
    let roles: &[(u32, u32, u32)] = &[
        (9, 10, 22),
        (9, 10, 9),
        (9, 10, 10),
        (16, 17, 15),
        (14, 15, 16),
        (16, 17, 31),
        (9, 10, 31),
        (0, 1, 2),
    ];
    let pair_offsets: Vec<i64> = vec![
        -520,
        -512,
        504,
        512,
        1001,
        1024,
        4088,
        4096,
        4096 + 16,
        4096 + 504,
        4096 + 512,
        40000,
        -40000,
        1 << 20,
    ];
    let mut a = Asm::new();
    let mut cases = Vec::new();
    for kind in [Kind::Ldp, Kind::Stp, Kind::LdpD, Kind::StpD] {
        for &(rt1, rt2, rn) in roles {
            let (rt1, rt2) = match kind {
                Kind::LdpD | Kind::StpD => (0, 1),
                _ => (rt1, rt2),
            };
            for &off in &pair_offsets {
                if rn == 31 && off < 16 {
                    continue;
                }
                let at = append_probe(&mut a, |a| match kind {
                    Kind::Ldp => a.ldp_base(rt1, rt2, rn, off as i32),
                    Kind::Stp => a.stp_base(rt1, rt2, rn, off as i32),
                    Kind::LdpD => a.pair_d(true, rt1, rt2, rn, off as i32),
                    Kind::StpD => a.pair_d(false, rt1, rt2, rn, off as i32),
                });
                cases.push((at, kind, rt1, rt2, rn, off));
            }
        }
    }
    let code = executable(a);
    let zeros = vec![0u8; data.bytes.len()];
    for loads in [true, false] {
        for region in [&mut data, &mut stack] {
            for (k, byte) in region.bytes.iter_mut().enumerate() {
                *byte = if loads { (k as u8) | 1 } else { 0 };
            }
        }
        let pass = cases
            .iter()
            .filter(|case| matches!(case.1, Kind::Ldp | Kind::LdpD) == loads);
        for &(at, kind, rt1, rt2, rn, off) in pass {
            let fp = matches!(kind, Kind::LdpD | Kind::StpD);
            let mut before = machine(&stack);
            if rn != 31 {
                before.x[rn as usize] = data.middle as u64;
            }
            let base = if rn == 31 {
                before.sp
            } else {
                data.middle as u64
            };
            let address = base.wrapping_add(off as u64);
            let mut after = before;
            let what = format!("{kind:?} {rt1} {rt2} [{rn}, #{off}]");
            run(&code, at, &mut after);
            let region = if rn == 31 { &mut stack } else { &mut data };
            let from = region.at(address);
            let mut expected = before;
            if loads {
                let word =
                    |k: usize| u64::from_le_bytes(region.bytes[k..k + 8].try_into().unwrap());
                if fp {
                    expected.d = [word(from), word(from + 8)];
                } else {
                    expected.x[rt1 as usize] = word(from);
                    expected.x[rt2 as usize] = word(from + 8);
                }
            } else {
                let (first, second) = if fp {
                    (before.d[0], before.d[1])
                } else {
                    (before.x[rt1 as usize], before.x[rt2 as usize])
                };
                assert_eq!(region.bytes[from..from + 8], first.to_le_bytes(), "{what}");
                assert_eq!(
                    region.bytes[from + 8..from + 16],
                    second.to_le_bytes(),
                    "{what}"
                );
                region.bytes[from..from + 16].fill(0);
                let push = stack.at(before.sp - 16);
                stack.bytes[push..push + 16].fill(0);
                assert!(
                    data.bytes == zeros && stack.bytes == zeros,
                    "{what}: a stray write"
                );
            }
            assert_eq!(after, expected, "{what}");
        }
    }

    // Pre- and post-indexed SP past imm7: SP moves by the offset, the pair at the moved SP
    // (pre) or the old one (post).
    let mut a = Asm::new();
    let mut cases = Vec::new();
    for off in [-528i64, -1024, -4096 - 16, -40000 - 16] {
        cases.push((
            append_probe(&mut a, |a| a.stp_pre(9, 10, off as i32)),
            true,
            off,
        ));
    }
    for off in [528i64, 1024, 4096 + 16, 40000 + 16] {
        cases.push((
            append_probe(&mut a, |a| a.ldp_post(9, 10, off as i32)),
            false,
            off,
        ));
    }
    let code = executable(a);
    for (at, pre, off) in cases {
        for (k, byte) in stack.bytes.iter_mut().enumerate() {
            *byte = if pre { 0 } else { (k as u8) | 1 };
        }
        let before = machine(&stack);
        let mut after = before;
        run(&code, at, &mut after);
        let mut expected = before;
        expected.sp = before.sp.wrapping_add(off as u64);
        if pre {
            let from = stack.at(expected.sp);
            assert_eq!(
                stack.bytes[from..from + 8],
                before.x[9].to_le_bytes(),
                "pre {off}"
            );
            assert_eq!(stack.bytes[from + 8..from + 16], before.x[10].to_le_bytes());
        } else {
            let from = stack.at(before.sp);
            let word = |k: usize| u64::from_le_bytes(stack.bytes[k..k + 8].try_into().unwrap());
            expected.x[9] = word(from);
            expected.x[10] = word(from + 8);
        }
        assert_eq!(
            after,
            expected,
            "{} {off}",
            if pre { "pre" } else { "post" }
        );
    }
}

/// Adds and subtracts past 12 bits, into another register, onto their own source and onto SP,
/// run on the CPU.
#[test]
fn legalized_adds_run_on_the_cpu() {
    let stack = Region::new(1 << 21);
    let imms: [u32; 10] = [
        4095,
        4096,
        4097,
        5000,
        0xF_FFF0,
        0xFF_F000,
        0xFF_FFFF,
        0x100_0000,
        0x100_0001,
        u32::MAX,
    ];
    let mut a = Asm::new();
    let mut cases = Vec::new();
    for imm in imms {
        for (rd, rn) in [(9u32, 10u32), (9, 9), (16, 16), (16, 17), (9, 31), (31, 31)] {
            // SP stays within the probe's stack: 16-byte multiples below 1 MB.
            if rd == 31 && (imm % 16 != 0 || imm >= 1 << 20) {
                continue;
            }
            for sub in [false, true] {
                let at = append_probe(&mut a, |a| {
                    if sub {
                        a.sub_imm(rd, rn, imm)
                    } else {
                        a.add_imm(rd, rn, imm)
                    }
                });
                cases.push((at, rd, rn, imm, sub));
            }
        }
    }
    assert_eq!(a.failed(), None);
    let code = executable(a);
    for (at, rd, rn, imm, sub) in cases {
        let before = machine(&stack);
        let mut after = before;
        run(&code, at, &mut after);
        let source = if rn == 31 {
            before.sp
        } else {
            before.x[rn as usize]
        };
        let result = if sub {
            source.wrapping_sub(imm.into())
        } else {
            source.wrapping_add(imm.into())
        };
        let mut expected = before;
        if rd == 31 {
            expected.sp = result;
        } else {
            expected.x[rd as usize] = result;
        }
        assert_eq!(
            after,
            expected,
            "x{rd} = x{rn} {} {imm}",
            if sub { "-" } else { "+" }
        );
    }
}

/// Flag-setting immediates past 12 bits (CMP, CMN, ADDS, SUBS), at the flags' edges, run on
/// the CPU: the result and NZCV are the operation's, every other register unchanged.
#[test]
fn legalized_flag_setting_immediates_run_on_the_cpu() {
    let stack = Region::new(1 << 16);
    #[derive(Clone, Copy, Debug)]
    enum Op {
        CmpW,
        CmpX,
        CmnW,
        CmnX,
        AddsW,
        SubsW,
    }
    let imms: [u32; 9] = [
        4095,
        4096,
        4097,
        5000,
        0x7000,
        0xFF_F000,
        0x100_0000,
        0x12_3456,
        u32::MAX,
    ];
    let mut a = Asm::new();
    let mut cases = Vec::new();
    for op in [Op::CmpW, Op::CmpX, Op::CmnW, Op::CmnX, Op::AddsW, Op::SubsW] {
        for imm in imms {
            for (rd, rn) in [
                (31u32, 1u32),
                (31, 16),
                (31, 17),
                (9, 10),
                (9, 9),
                (16, 16),
                (17, 16),
            ] {
                let rd = match op {
                    Op::AddsW | Op::SubsW => rd,
                    _ => 31,
                };
                let at = append_probe(&mut a, |a| match op {
                    Op::CmpW => a.cmp_imm_w(rn, imm),
                    Op::CmpX => a.cmp_imm_x(rn, imm),
                    Op::CmnW => a.cmn_imm_w(rn, imm),
                    Op::CmnX => a.cmn_imm_x(rn, imm),
                    Op::AddsW => a.adds_imm_w(rd, rn, imm),
                    Op::SubsW => a.subs_imm_w(rd, rn, imm),
                });
                cases.push((at, op, rd, rn, imm));
            }
        }
    }
    assert_eq!(a.failed(), None);
    let code = executable(a);
    for (at, op, rd, rn, imm) in cases {
        let imm64 = u64::from(imm);
        for value in [
            imm64,
            imm64.wrapping_sub(1),
            imm64 + 1,
            0,
            u64::MAX,
            0x8000_0000,
            0x7FFF_FFFF,
            0x8000_0000_0000_0000,
            0x7FFF_FFFF_FFFF_FFFF,
            u64::from(imm.wrapping_neg()),
        ] {
            let mut before = machine(&stack);
            before.x[rn as usize] = value;
            let mut after = before;
            run(&code, at, &mut after);
            let wide = matches!(op, Op::CmpX | Op::CmnX);
            let add = matches!(op, Op::CmnW | Op::CmnX | Op::AddsW);
            let (result, nzcv) = flags(value, imm64, wide, add);
            let mut expected = before;
            expected.nzcv = nzcv;
            if rd != 31 {
                expected.x[rd as usize] = result;
            }
            assert_eq!(after, expected, "{op:?} x{rd}, x{rn}={value:#x}, #{imm:#x}");
        }
    }
}

/// The result (zero-extended for 32 bits) and NZCV of `value` plus or minus `imm`.
fn flags(value: u64, imm: u64, wide: bool, add: bool) -> (u64, u64) {
    let bits = if wide { 64 } else { 32 };
    let mask = if wide { u64::MAX } else { 0xFFFF_FFFF };
    let (x, y) = (value & mask, imm & mask);
    let (result, carry) = if add {
        let sum = u128::from(x) + u128::from(y);
        ((sum as u64) & mask, sum >> bits != 0)
    } else {
        (x.wrapping_sub(y) & mask, x >= y)
    };
    let sign = |v: u64| (v >> (bits - 1)) & 1 == 1;
    let overflow = if add {
        sign(x) == sign(y) && sign(result) != sign(x)
    } else {
        sign(x) != sign(y) && sign(result) != sign(x)
    };
    let nzcv = u64::from(sign(result)) << 31
        | u64::from(result == 0) << 30
        | u64::from(carry) << 29
        | u64::from(overflow) << 28;
    (result, nzcv)
}
