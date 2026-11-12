//! The direct sequence's compare of the callee chunk's entry word (P3 stage 2, step 14): with
//! the optimizing tier on, an entry weighing K, it loads the way's copy of the word by a load of
//! its own beside the chunk's LDUR and compares the chunk's word, four instructions; with the
//! tier off, an entry weighing one, it emits the sequence as before: the LDUR and no compare.
//! No load of the way spans its byte 64, a cache line's boundary where the way starts on one:
//! the LDP of bytes 56-71 that took the copy with the chunk paid the X925's and the A725's
//! alignment latency there once a call (the small callees' study, 2026-11-12).

use super::*;

/// The direct sequence's words for a call of one argument, its entries weighing `weight`.
fn sequence(weight: u32) -> Vec<u32> {
    let mut engine = crate::Engine::new();
    let i = &mut *engine.interp;
    let layout = crate::value::jit_layout(&i.object_proto);
    let ilayout = crate::interpreter::interp_layout(i);
    let mut a = asm::Asm::new();
    a.set_entry_weight(weight);
    let hit_slow = a.new_label();
    let l_unwind = a.new_label();
    let done = a.new_label();
    let finish = a.new_label();
    assert!(emit_direct_sequence(
        &mut a,
        &ilayout,
        &layout,
        std::mem::offset_of!(crate::bytecode::Chunk, inline_attempted),
        std::mem::offset_of!(crate::bytecode::Chunk, jit_runs),
        std::mem::offset_of!(crate::bytecode::Chunk, inline_retry_at),
        1,
        DirectForm::Call { with_this: false },
        hit_slow,
        l_unwind,
        done,
        finish,
        DirectCallSite::Position(0),
    ));
    for label in [hit_slow, l_unwind, done, finish] {
        a.bind(label);
    }
    a.finish()
}

/// The one word `emit` assembles.
fn word(emit: impl FnOnce(&mut asm::Asm)) -> u32 {
    let mut a = asm::Asm::new();
    emit(&mut a);
    let words = a.finish();
    assert_eq!(words.len(), 1);
    words[0]
}

#[test]
fn the_tier_on_compares_the_entry_word_and_the_tier_off_emits_the_sequence_as_before() {
    let ldur = word(|a| a.ldur(4, 12, 64));
    let copy = word(|a| a.ldr_w_imm(6, 12, 60));
    let ldrb = word(|a| a.ldrb_imm(5, 4, crate::bytecode::CHUNK_ENTRY_OFF as u32));
    let cmp = word(|a| a.cmp_reg_w(5, 6));
    assert_eq!(copy, 0xB940_3D86, "ldr w6, [x12, #60]");
    assert_eq!(cmp, 0x6B06_00BF, "cmp w5, w6");
    let count = |words: &[u32], w: u32| words.iter().filter(|&&x| x == w).count();
    let off = sequence(1);
    let on = sequence(crate::opt::ENTRY_WEIGHT);
    // The tier off: no compare, the chunk loaded alone (here and at the install).
    assert_eq!(
        (count(&off, copy), count(&off, ldrb), count(&off, cmp)),
        (0, 0, 0)
    );
    assert_eq!(count(&off, ldur), 2);
    // The tier on: the chunk's LDUR as before, then the copy's load and the compare.
    assert_eq!(
        (count(&on, copy), count(&on, ldrb), count(&on, cmp)),
        (1, 1, 1)
    );
    assert_eq!(count(&on, ldur), 2);
    assert_eq!(on.len(), off.len() + 4);
    let at = on.iter().position(|&w| w == ldur).unwrap();
    assert_eq!(off[at], ldur, "the chunk's LDUR keeps its place");
    for words in [&off, &on] {
        assert_eq!(
            way_loads_across_64(words),
            0,
            "a load of the way spans its byte 64"
        );
    }
}

/// The loads based on x12, the way, whose bytes span its byte 64: an LDP of X registers or a
/// 64-bit LDR/LDUR whose bytes [o, o + size) contain both 63 and 64.
fn way_loads_across_64(words: &[u32]) -> usize {
    words
        .iter()
        .filter(|&&w| {
            let rn = (w >> 5) & 31;
            if rn != 12 {
                return false;
            }
            // LDP Xt1, Xt2, [Xn, #imm7 * 8] (signed offset form).
            let (offset, size) = if w & 0xFFC0_0000 == 0xA940_0000 {
                let imm7 = ((w >> 15) & 0x7F) as i32;
                (((imm7 << 25) >> 25) * 8, 16)
            // LDR Xt, [Xn, #imm12 * 8].
            } else if w & 0xFFC0_0000 == 0xF940_0000 {
                (((w >> 10) & 0xFFF) as i32 * 8, 8)
            // LDUR Xt, [Xn, #simm9].
            } else if w & 0xFFE0_0C00 == 0xF840_0000 {
                let imm9 = ((w >> 12) & 0x1FF) as i32;
                ((imm9 << 23) >> 23, 8)
            } else {
                return false;
            };
            offset < 64 && offset + size > 64
        })
        .count()
}

#[test]
fn the_check_finds_the_ldp_that_took_the_copy_with_the_chunk() {
    // The form the compare replaced: one LDP of bytes 56-71.
    let ldp = word(|a| a.ldp_base(6, 4, 12, 56));
    assert_eq!(way_loads_across_64(&[ldp]), 1);
    assert_eq!(
        way_loads_across_64(&[word(|a| a.ldp_base(4, 6, 12, 72))]),
        0
    );
    assert_eq!(way_loads_across_64(&[word(|a| a.ldur(4, 12, 64))]), 0);
    assert_eq!(way_loads_across_64(&[word(|a| a.ldr_imm(4, 12, 64))]), 0);
    assert_eq!(way_loads_across_64(&[word(|a| a.ldur(4, 12, 60))]), 1);
}
