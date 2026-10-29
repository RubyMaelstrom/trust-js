//! W2e: inline allocation in generated code, x86-64, as ARM64's (`jit::alloc`): its first use is
//! a fresh frame's activation, which the prologue makes in line instead of calling the frame
//! setup helper (W3 item 3's step 7).

use super::*;
use crate::bytecode::activation::InlineActivation;
use crate::gc_core::{inline_alloc as off, InlineAllocSite};

/// Take a cell of `site`'s size class into r9 in line, or jump to `slow` having changed
/// nothing, as ARM64's `jit::alloc::emit_inline_alloc` does (see `gc_core::InlineAllocSite`).
/// Clobbers rax, rcx, rdx, r8, r10 and r11.
pub(super) fn emit_inline_alloc(a: &mut Asm, site: &InlineAllocSite, slow: usize) {
    a.nr_imm(R10, site.heap as u64);
    a.nr_load8(AX, R10, off::HEAP_FAST_ALLOC);
    a.nr_zero_guard(AX, slow);
    a.nr_load(R11, R10, off::HEAP_CURSOR_BASE);
    a.nr_op_imm(0, R11, (site.cursor * off::CURSOR_SIZE) as i32); // add r11,cursor
    a.nr_load(AX, R11, off::CURSOR_FREE);
    a.nr_zero_guard(AX, slow);
    // The byte charge (`Heap::charge_bytes`), unless it would pass the check.
    a.nr_load(CX, R10, off::HEAP_ALLOCATED_BYTES);
    a.nr_op_imm(0, CX, site.cell_size as i32); // add rcx,cell_size
    a.nr_load(DX, R10, off::HEAP_DUE_CHECK_BYTES);
    a.nr_cmp(CX, DX);
    a.jcc(ABOVE, slow);
    a.nr_store(R10, off::HEAP_ALLOCATED_BYTES, CX);
    // The cursor's lowest free cell (`Cursor::take`): its bit in rcx, the word without it.
    a.nr_rex(true, CX, AX);
    a.bytes(&[0x0f, 0xbc, 0xc8]); // bsf rcx,rax
    a.nr_lea(DX, AX, -1);
    a.nr_reg(0x21, AX, DX); // and rax,rdx
    a.nr_store(R11, off::CURSOR_FREE, AX);
    // The page's allocation bitmap: bits[word] |= 1 << bit.
    a.nr_load32(DX, R11, off::CURSOR_WORD);
    a.nr_load(R8, R11, off::CURSOR_BITS);
    a.nr_shift(false, DX, 3);
    a.nr_reg(0x01, R8, DX); // add r8,rdx: &bits[word]
    a.nr_imm(R9, 1);
    a.nr_rex(true, 0, R9);
    a.bytes(&[0xd3, 0xe0 | (R9 & 7)]); // shl r9,cl
    a.nr_load(AX, R8, 0);
    a.nr_reg(0x09, AX, R9); // or rax,r9
    a.nr_store(R8, 0, AX);
    // The cell's index, word * 64 + bit; then `taken` and `high`.
    a.nr_shift(false, DX, 3);
    a.nr_reg(0x01, DX, CX); // add rdx,rcx
    a.nr_mem(false, &[0xff], 0, R11, off::CURSOR_TAKEN as i32); // inc dword [r11+taken]
    a.nr_lea(AX, DX, 1);
    a.nr_store32(R11, off::CURSOR_HIGH, AX);
    // The cell: the page's first cell plus index cells.
    a.nr_mul_imm(DX, site.cell_size);
    a.nr_load(R9, R11, off::CURSOR_PAGE_BASE);
    a.nr_reg(0x01, R9, DX); // add r9,rdx
}

/// Make the activation in line and install it as the frame's environment, or jump to `slow`
/// having changed nothing, as ARM64's `jit::alloc::emit_inline_activation`: the cell (r9), its
/// header's words with the closure's environment as its outer record, the template's words
/// (each run of equal words from one register), then `env_parent_raw` and `env_raw`.
pub(super) fn emit_inline_activation(a: &mut Asm, activation: &InlineActivation, slow: usize) {
    emit_inline_alloc(a, &activation.site, slow);
    a.nr_load(
        R10,
        CTX,
        std::mem::offset_of!(super::super::JitCtx, env_raw),
    );
    let mut held = None;
    let mut store = |a: &mut Asm, offset: usize, word: u64| {
        if held != Some(word) {
            a.nr_imm(AX, word);
            held = Some(word);
        }
        a.nr_store(R9, offset, AX);
    };
    for (k, &word) in activation.header.iter().enumerate() {
        if k == activation.parent_word {
            a.nr_store(R9, 8 * k, R10);
        } else {
            store(a, 8 * k, word);
        }
    }
    for (slot, &word) in activation.slots.iter().enumerate() {
        store(a, crate::interpreter::RECORD_HEADER + 8 * slot, word);
    }
    a.nr_store(
        CTX,
        std::mem::offset_of!(super::super::JitCtx, env_parent_raw),
        R10,
    );
    a.nr_store(CTX, std::mem::offset_of!(super::super::JitCtx, env_raw), R9);
}
