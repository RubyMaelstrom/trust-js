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
/// free cell, the cell's bytes would pass the in-line check (zero while a collection is due; the
/// slow path's Rust allocation then charges them, recomputes the heap's due bit, and its helper
/// polls), or one more counted cell would pass the live check. The cell is neither zeroed nor
/// initialized: the caller writes every word a trace reads before the next safepoint. No poll
/// follows (W2e's design 1), and no barrier: the cell is young. Clobbers x10-x15, and x16-x17
/// for a counted descriptor or an allocation count.
pub(super) fn emit_inline_alloc(a: &mut asm::Asm, site: &InlineAllocSite, dst: u32, slow: usize) {
    debug_assert!(!(10..=17).contains(&dst));
    a.mov_imm64(10, site.heap as u64);
    a.ldrb_imm(11, 10, off::HEAP_FAST_ALLOC as u32);
    a.cbz(11, false, slow);
    a.ldr_imm(12, 10, off::HEAP_CURSOR_BASE as u32);
    a.add_imm(12, 12, (site.cursor * off::CURSOR_SIZE) as u32);
    a.ldr_imm(13, 12, off::CURSOR_FREE as u32);
    a.cbz(13, true, slow);
    // The byte charge (`Heap::charge_bytes`), unless it would pass the check.
    a.ldr_imm(14, 10, off::HEAP_ALLOCATED_BYTES as u32);
    a.ldr_imm(15, 10, off::HEAP_INLINE_CHECK_BYTES as u32);
    a.add_imm(14, 14, site.cell_size as u32);
    a.cmp_reg_x(14, 15);
    a.b_cond(C_HI, slow);
    if site.live_count != 0 {
        // The embedder's live count (`Heap::count_live`): one more cell, unless it would pass
        // the live check (a signed count; `i64::MAX` while the heap bit is set).
        a.mov_imm64(16, site.live_count as u64);
        a.ldr_imm(17, 16, 0);
        a.add_imm(17, 17, 1);
        a.ldr_imm(15, 10, off::HEAP_LIVE_CHECK as u32);
        a.cmp_reg_x(17, 15);
        a.b_cond(C_GT, slow);
        a.str_imm(17, 16, 0);
    }
    if site.allocated != 0 {
        // The embedder's count of the cells it allocated (objects': the task boundary's policy).
        a.mov_imm64(16, site.allocated as u64);
        a.ldr_imm(17, 16, 0);
        a.add_imm(17, 17, 1);
        a.str_imm(17, 16, 0);
    }
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

/// Make an object literal's instance in line (W2e step 3), its `slots` values taken from the
/// operand stack's top, and leave it on the stack in their place; or take the helper `helper`,
/// whose call makes the whole object, having changed nothing: when the site's `image` is not ready
/// (its first execution makes it, as V8's boilerplate and JSC's profile come from an earlier run),
/// or the allocation's checks fail. The cell (x9), the image's header words, the realm's
/// %Object.prototype% (the interpreter's `object_proto`) in its word, then the slots. No barrier
/// (the object is young) and no poll (W2e's design 1). Clobbers x9-x17.
pub(super) fn emit_inline_object(
    a: &mut asm::Asm,
    site: &InlineAllocSite,
    image: &crate::bytecode::ObjectImage,
    slots: usize,
    helper: usize,
    pc: u32,
    unwind: usize,
) {
    emit_inline_cell(
        a,
        site,
        image,
        INTERP_OBJECT_PROTO,
        0,
        slots,
        helper,
        pc,
        unwind,
    );
}

/// Make an Array in line from `image` (W2e step 4: `[]` and a constant literal's copy, which
/// shares its site's block): the image's header and its `length` slot, with the realm's
/// %Array.prototype%, pushed; or the helper, as [`emit_inline_object`]. Clobbers x9-x17.
pub(super) fn emit_inline_array(
    a: &mut asm::Asm,
    site: &InlineAllocSite,
    image: &crate::bytecode::ObjectImage,
    helper: usize,
    pc: u32,
    unwind: usize,
) {
    emit_inline_cell(a, site, image, INTERP_ARRAY_PROTO, 1, 0, helper, pc, unwind);
}

/// [`emit_inline_object`] and [`emit_inline_array`]: the image's `fixed` slots (an Array's
/// `length`) then `stacked` slots from the operand stack, the prototype at `proto` in the
/// interpreter.
#[allow(clippy::too_many_arguments)]
fn emit_inline_cell(
    a: &mut asm::Asm,
    site: &InlineAllocSite,
    image: &crate::bytecode::ObjectImage,
    proto: usize,
    fixed: usize,
    slots: usize,
    helper: usize,
    pc: u32,
    unwind: usize,
) {
    debug_assert!(fixed <= 1);
    let slow = a.new_label();
    let done = a.new_label();
    let image = std::ptr::from_ref(image) as u64;
    a.mov_imm64(9, image);
    a.ldr_imm(
        10,
        9,
        std::mem::offset_of!(crate::bytecode::ObjectImage, ready) as u32,
    );
    a.cbz(10, true, slow);
    emit_inline_alloc(a, site, 9, slow);
    a.mov_imm64(
        15,
        image + std::mem::offset_of!(crate::bytecode::ObjectImage, words) as u64,
    );
    a.ldr_imm(10, 19, std::mem::offset_of!(JitCtx, interp) as u32);
    a.ldr_imm(10, 10, proto as u32);
    for k in 0..crate::value::OBJECT_HEADER / 8 {
        let offset = (8 * k) as u32;
        if k == crate::value::OBJECT_PROTO_WORD {
            a.str_imm(10, 9, offset);
        } else {
            a.ldr_imm(11, 15, offset);
            a.str_imm(11, 9, offset);
        }
    }
    if fixed == 1 {
        let slot0 = std::mem::offset_of!(crate::bytecode::ObjectImage, slot0)
            - std::mem::offset_of!(crate::bytecode::ObjectImage, words);
        a.ldr_imm(11, 15, slot0 as u32);
        a.str_imm(11, 9, crate::value::OBJECT_HEADER as u32);
    }
    for slot in 0..slots {
        emit_exec_word_load(a, 12, 20, -(8 * (slots - slot) as i32));
        a.str_imm(
            12,
            9,
            (crate::value::OBJECT_HEADER + 8 * (fixed + slot)) as u32,
        );
    }
    // The object's word over the first value's (or pushed, for none).
    a.mov_imm64(12, crate::value::PACK_OBJ);
    a.logic_x(1, 9, 9, 12);
    emit_exec_word_store(a, 9, 20, -(8 * slots as i32));
    match slots {
        0 => a.add_imm(20, 20, 8),
        1 => {}
        _ => a.sub_imm(20, 20, (8 * (slots - 1)) as u32),
    }
    #[cfg(test)]
    record_inline_object(a);
    a.b(done);
    a.bind(slow);
    emit_op_helper(a, helper, pc, unwind);
    a.bind(done);
}

#[cfg(test)]
thread_local! {
    /// Objects generated code made in line ([`emit_inline_object`]), for tests.
    pub(crate) static INLINE_OBJECTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Count an in-line object for tests, through the test thread's counter's address (no call;
/// x10 and x11 are scratch here).
#[cfg(test)]
fn record_inline_object(a: &mut asm::Asm) {
    let address = INLINE_OBJECTS.with(|count| count.as_ptr() as usize);
    a.mov_imm64(10, address as u64);
    a.ldr_imm(11, 10, 0);
    a.add_imm(11, 11, 1);
    a.str_imm(11, 10, 0);
}
