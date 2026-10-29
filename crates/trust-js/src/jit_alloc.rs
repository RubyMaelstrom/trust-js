//! W2e: inline allocation in generated code, ARM64 (the plan's "cursor take, byte charge and
//! template copy, keeping the poll after allocation"). Its first use is a fresh frame's
//! activation, which the prologue makes in line instead of calling the frame setup helper (W3
//! item 3's step 7), as SpiderMonkey's Baseline makes its CallObject from a template in its
//! prologue and checks its stack after (jit/BaselineCodeGen.cpp:1506-1606, 7059-7080).
//! [`emit_inline_alloc`] serves any small cell; [`emit_inline_activation`] writes a record.

use super::*;
use crate::bytecode::activation::InlineActivation;
use crate::gc_core::{inline_alloc as off, InlineAllocSite};

/// rbit xd, xn
fn rbit_x(a: &mut asm::Asm, rd: u32, rn: u32) {
    a.word(0xDAC0_0000 | (rn << 5) | rd);
}

/// clz xd, xn
fn clz_x(a: &mut asm::Asm, rd: u32, rn: u32) {
    a.word(0xDAC0_1000 | (rn << 5) | rd);
}

/// lslv xd, xn, xm
fn lsl_reg_x(a: &mut asm::Asm, rd: u32, rn: u32, rm: u32) {
    a.word(0x9AC0_2000 | (rm << 16) | (rn << 5) | rd);
}

/// Take a cell of `site`'s size class into `dst` in line, or branch to `slow` having changed
/// nothing ([`InlineAllocSite`]): when the heap counts each allocation, the cursor's word has no
/// free cell, or the cell's bytes would pass the byte check (the slow path's Rust allocation
/// then charges them and recomputes the heap's due bit). The cell is neither zeroed nor
/// initialized: the caller writes every word a trace reads before the next safepoint, which it
/// then reaches, so a collection the charge made due runs there. No barrier: the cell is young.
/// Clobbers x10-x15.
pub(super) fn emit_inline_alloc(a: &mut asm::Asm, site: &InlineAllocSite, dst: u32, slow: usize) {
    debug_assert!(!(10..=15).contains(&dst));
    a.mov_imm64(10, site.heap as u64);
    a.ldrb_imm(11, 10, off::HEAP_FAST_ALLOC as u32);
    a.cbz(11, false, slow);
    a.ldr_imm(12, 10, off::HEAP_CURSOR_BASE as u32);
    a.add_imm(12, 12, (site.cursor * off::CURSOR_SIZE) as u32);
    a.ldr_imm(13, 12, off::CURSOR_FREE as u32);
    a.cbz(13, true, slow);
    // The byte charge (`Heap::charge_bytes`), unless it would pass the check.
    a.ldr_imm(14, 10, off::HEAP_ALLOCATED_BYTES as u32);
    a.ldr_imm(15, 10, off::HEAP_DUE_CHECK_BYTES as u32);
    a.add_imm(14, 14, site.cell_size as u32);
    a.cmp_reg_x(14, 15);
    a.b_cond(C_HI, slow);
    a.str_imm(14, 10, off::HEAP_ALLOCATED_BYTES as u32);
    // The cursor's lowest free cell (`Cursor::take`): its bit in x15, the word without it.
    rbit_x(a, 15, 13);
    clz_x(a, 15, 15);
    a.sub_imm(14, 13, 1);
    a.logic_x(0, 13, 13, 14);
    a.str_imm(13, 12, off::CURSOR_FREE as u32);
    // The page's allocation bitmap: bits[word] |= 1 << bit.
    a.ldr_w_imm(14, 12, off::CURSOR_WORD as u32);
    a.ldr_imm(11, 12, off::CURSOR_BITS as u32);
    a.add_shifted(11, 11, 14, 3);
    a.ldr_imm(13, 11, 0);
    a.movz(dst, 1, 0);
    lsl_reg_x(a, dst, dst, 15);
    a.logic_x(1, 13, 13, dst);
    a.str_imm(13, 11, 0);
    // The cell's index, word * 64 + bit; then `taken` and `high`.
    a.add_shifted(14, 15, 14, 6);
    a.ldr_w_imm(11, 12, off::CURSOR_TAKEN as u32);
    a.add_imm(11, 11, 1);
    a.str_w_imm(11, 12, off::CURSOR_TAKEN as u32);
    a.add_imm(11, 14, 1);
    a.str_w_imm(11, 12, off::CURSOR_HIGH as u32);
    // The cell: the page's first cell plus index cells.
    a.ldr_imm(11, 12, off::CURSOR_PAGE_BASE as u32);
    a.mov_imm64(13, site.cell_size as u64);
    a.madd(dst, 14, 13, 11);
}

/// Make the activation in line and install it as the frame's environment (what
/// `bytecode::jit_frame_setup` does for such a chunk), or branch to `slow` having changed
/// nothing: the cell (x9), its header's words, the closure's environment (`env_raw`) as its
/// outer record, then the template's words, each run of equal words from one register (as
/// SpiderMonkey's `initGCSlots` fills a run, MacroAssembler.cpp:859-924); then `env_parent_raw`
/// and `env_raw`. The caller copies the captured parameters and `this`, then polls. Clobbers
/// x9-x15.
pub(super) fn emit_inline_activation(a: &mut asm::Asm, activation: &InlineActivation, slow: usize) {
    emit_inline_alloc(a, &activation.site, 9, slow);
    a.ldr_imm(10, 19, std::mem::offset_of!(JitCtx, env_raw) as u32);
    let mut held = None;
    let mut store = |a: &mut asm::Asm, offset: usize, word: u64| {
        if word == 0 {
            a.str_imm(31, 9, offset as u32);
            return;
        }
        if held != Some(word) {
            a.mov_imm64(11, word);
            held = Some(word);
        }
        a.str_imm(11, 9, offset as u32);
    };
    for (k, &word) in activation.header.iter().enumerate() {
        if k == activation.parent_word {
            a.str_imm(10, 9, (8 * k) as u32);
        } else {
            store(a, 8 * k, word);
        }
    }
    for (slot, &word) in activation.slots.iter().enumerate() {
        store(a, crate::interpreter::RECORD_HEADER + 8 * slot, word);
    }
    a.str_imm(10, 19, std::mem::offset_of!(JitCtx, env_parent_raw) as u32);
    a.str_imm(9, 19, std::mem::offset_of!(JitCtx, env_raw) as u32);
}
