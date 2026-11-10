//! The direct sequence's compare of the callee chunk's entry word (P3 stage 2, step 14): with
//! the optimizing tier on, an entry weighing K, it loads the way's epoch and its copy of the
//! word with the callee's chunk in one LDP, in place of the chunk's LDUR, and compares the
//! chunk's word, three instructions; with the tier off, an entry weighing one, it emits the
//! sequence as before: the LDUR and no compare.

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
    let ldp = word(|a| a.ldp_base(6, 4, 12, 56));
    let ldrb = word(|a| a.ldrb_imm(5, 4, crate::bytecode::CHUNK_ENTRY_OFF as u32));
    let cmp = word(|a| a.cmp_lsr_x(5, 6, 32));
    assert_eq!(cmp, 0xEB46_80BF, "cmp x5, x6, lsr #32");
    let count = |words: &[u32], w: u32| words.iter().filter(|&&x| x == w).count();
    let off = sequence(1);
    let on = sequence(crate::opt::ENTRY_WEIGHT);
    // The tier off: no compare, the chunk loaded alone (here and at the install).
    assert_eq!(
        (count(&off, ldp), count(&off, ldrb), count(&off, cmp)),
        (0, 0, 0)
    );
    assert_eq!(count(&off, ldur), 2);
    // The tier on: the LDP in place of the first LDUR, and the compare's three instructions.
    assert_eq!(
        (count(&on, ldp), count(&on, ldrb), count(&on, cmp)),
        (1, 1, 1)
    );
    assert_eq!(count(&on, ldur), 1);
    assert_eq!(on.len(), off.len() + 3);
    let at = on.iter().position(|&w| w == ldp).unwrap();
    assert_eq!(off[at], ldur, "the LDP takes the LDUR's place");
}
