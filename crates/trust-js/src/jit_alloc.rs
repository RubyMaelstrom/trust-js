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

/// An array literal with `count` elements in line (W2e step 5; MakeArray, what
/// `Interp::make_array_from_raw` does): the values from the operand stack's top into an exact
/// element block with their kind bits (`Words::from_words`'s), then the Array's cell from the
/// chunk's MakeArray `image` with that block and `length`, pushed in the values' place; or the
/// helper, having written nothing but possibly an element block that nothing names (plain data in
/// a storage cell, which the next sweep frees): the block comes first so that no object cell is
/// ever left unwritten. Clobbers x5-x17, d0 and d1.
#[allow(clippy::too_many_arguments)]
pub(super) fn emit_inline_array_elements(
    a: &mut asm::Asm,
    cell: &InlineAllocSite,
    block: &InlineAllocSite,
    image: &crate::bytecode::ObjectImage,
    count: usize,
    helper: usize,
    pc: u32,
    unwind: usize,
) {
    use crate::value::word_block::{KIND_ALL, KIND_INT32, KIND_NO_HOLES};
    let slow = a.new_label();
    let done = a.new_label();
    let image_addr = std::ptr::from_ref(image) as u64;
    a.mov_imm64(9, image_addr);
    a.ldr_imm(
        10,
        9,
        std::mem::offset_of!(crate::bytecode::ObjectImage, ready) as u32,
    );
    a.cbz(10, true, slow);
    // The values' kind bits (w5), as `word_kind` folds them; a hole's tag (the TDZ marker) is
    // the helper's.
    a.movz(5, KIND_ALL, 0);
    for k in 0..count {
        let next = a.new_label();
        let not_number = a.new_label();
        emit_exec_word_load(a, 12, 20, -(8 * (count - k) as i32));
        emit_exec_number_guard(a, 12, 0, 13, not_number);
        a.fcvtzs_w_d(13, 0);
        a.scvtf_d_w(1, 13);
        a.fmov_x_d(13, 1);
        a.cmp_reg_x(13, 12);
        a.b_cond(C_EQ, next);
        a.logic_imm_w(0, 5, 5, asm::logical_imm_w(!KIND_INT32).unwrap());
        a.b(next);
        a.bind(not_number);
        a.lsr_imm(13, 12, 48);
        a.movz(14, (crate::value::PACK_EMPTY >> 48) as u32, 0);
        a.cmp_reg_w(13, 14);
        a.b_cond(C_EQ, slow);
        a.logic_imm_w(0, 5, 5, asm::logical_imm_w(KIND_NO_HOLES).unwrap());
        a.bind(next);
    }
    // The element block (x6): its header {used, capacity, shifted, flags}, then the words.
    emit_inline_alloc(a, block, 6, slow);
    a.mov_imm64(7, count as u64 | ((count as u64) << 32));
    a.str_imm(7, 6, 0);
    a.lsl_imm(7, 5, 32);
    a.str_imm(7, 6, 8);
    for k in 0..count {
        emit_exec_word_load(a, 12, 20, -(8 * (count - k) as i32));
        a.str_imm(12, 6, (16 + 8 * k) as u32);
    }
    a.add_imm(6, 6, 16);
    // The Array's cell (x9): the image's header with the block and the realm's %Array.prototype%,
    // then `length`.
    emit_inline_alloc(a, cell, 9, slow);
    a.mov_imm64(
        15,
        image_addr + std::mem::offset_of!(crate::bytecode::ObjectImage, words) as u64,
    );
    a.ldr_imm(10, 19, std::mem::offset_of!(JitCtx, interp) as u32);
    a.ldr_imm(10, 10, INTERP_ARRAY_PROTO as u32);
    for k in 0..crate::value::OBJECT_HEADER / 8 {
        let offset = (8 * k) as u32;
        if k == crate::value::OBJECT_PROTO_WORD {
            a.str_imm(10, 9, offset);
        } else if k == crate::value::CELL_ELEMS_OFFSET / 8 {
            a.str_imm(6, 9, offset);
        } else {
            a.ldr_imm(11, 15, offset);
            a.str_imm(11, 9, offset);
        }
    }
    a.mov_imm64(11, crate::value::Value::Num(count as f64).bits());
    a.str_imm(11, 9, crate::value::OBJECT_HEADER as u32);
    a.mov_imm64(12, crate::value::PACK_OBJ);
    a.logic_x(1, 9, 9, 12);
    emit_exec_word_store(a, 9, 20, -(8 * count as i32));
    if count > 1 {
        a.sub_imm(20, 20, (8 * (count - 1)) as u32);
    }
    #[cfg(test)]
    record_inline_object(a);
    a.b(done);
    a.bind(slow);
    emit_op_helper(a, helper, pc, unwind);
    a.bind(done);
}

/// [`emit_inline_alloc`] of a class chosen at run time (W2e step 6): the cursor's byte offset
/// among the heap's cursors in x`cursor` and the cell's size in x`size` (neither in x10-x17);
/// `heap`'s heap, live count and allocation count are the code's own. Clobbers x10-x17.
pub(super) fn emit_inline_alloc_dynamic(
    a: &mut asm::Asm,
    heap: &InlineAllocSite,
    cursor: u32,
    size: u32,
    dst: u32,
    slow: usize,
) {
    debug_assert!(![dst, cursor, size].iter().any(|r| (10..=17).contains(r)));
    a.mov_imm64(10, heap.heap as u64);
    a.ldrb_imm(11, 10, off::HEAP_FAST_ALLOC as u32);
    a.cbz(11, false, slow);
    a.ldr_imm(12, 10, off::HEAP_CURSOR_BASE as u32);
    a.add_shifted(12, 12, cursor, 0);
    a.ldr_imm(13, 12, off::CURSOR_FREE as u32);
    a.cbz(13, true, slow);
    a.ldr_imm(14, 10, off::HEAP_ALLOCATED_BYTES as u32);
    a.ldr_imm(15, 10, off::HEAP_INLINE_CHECK_BYTES as u32);
    a.add_shifted(14, 14, size, 0);
    a.cmp_reg_x(14, 15);
    a.b_cond(C_HI, slow);
    if heap.live_count != 0 {
        a.mov_imm64(16, heap.live_count as u64);
        a.ldr_imm(17, 16, 0);
        a.add_imm(17, 17, 1);
        a.ldr_imm(15, 10, off::HEAP_LIVE_CHECK as u32);
        a.cmp_reg_x(17, 15);
        a.b_cond(C_GT, slow);
        a.str_imm(17, 16, 0);
    }
    if heap.allocated != 0 {
        a.mov_imm64(16, heap.allocated as u64);
        a.ldr_imm(17, 16, 0);
        a.add_imm(17, 17, 1);
        a.str_imm(17, 16, 0);
    }
    a.str_imm(14, 10, off::HEAP_ALLOCATED_BYTES as u32);
    rbit_x(a, 15, 13);
    clz_x(a, 15, 15);
    a.sub_imm(14, 13, 1);
    a.logic_x(0, 13, 13, 14);
    a.str_imm(13, 12, off::CURSOR_FREE as u32);
    a.ldr_w_imm(14, 12, off::CURSOR_WORD as u32);
    a.ldr_imm(11, 12, off::CURSOR_BITS as u32);
    a.add_shifted(11, 11, 14, 3);
    a.ldr_imm(13, 11, 0);
    a.movz(dst, 1, 0);
    lsl_reg_x(a, dst, dst, 15);
    a.logic_x(1, 13, 13, dst);
    a.str_imm(13, 11, 0);
    a.add_shifted(14, 15, 14, 6);
    a.ldr_w_imm(11, 12, off::CURSOR_TAKEN as u32);
    a.add_imm(11, 11, 1);
    a.str_w_imm(11, 12, off::CURSOR_TAKEN as u32);
    a.add_imm(11, 14, 1);
    a.str_w_imm(11, 12, off::CURSOR_HIGH as u32);
    a.ldr_imm(11, 12, off::CURSOR_PAGE_BASE as u32);
    a.madd(dst, 14, size, 11);
}

/// The `this` of a direct construction in line (W2e step 6; [[Construct]] step 3,
/// OrdinaryCreateFromConstructor(F, "%Object.prototype%"), what `bytecode::jit_construct_this`
/// does): with x0 the construct site (a `Cell<ConstructSite>`) and x1 the constructor's cell,
/// leave the instance's word in x0 and branch to `made`, or branch to `helper` having changed
/// nothing, with x0 and x1 as they came. It reads the callee's chunk's instance image (ready
/// once the constructor's learning is over), the constructor's `prototype` through the site's
/// cached shape and slot (in the cell, an Object: not the deferred prototype, which the helper
/// makes), then allocates the image's class, as V8's FastNewObject and SpiderMonkey's call stub
/// make `this` on the caller's side (target/p2/w2e/exemplars.txt, 1.4, 2.4). Clobbers x2-x17.
pub(super) fn emit_inline_construct_this(
    a: &mut asm::Asm,
    heap: &InlineAllocSite,
    helper: usize,
    made: usize,
) {
    use crate::bytecode::{CallIc, Chunk, ConstructSite, InstanceImage};
    use std::mem::offset_of;
    let chunk = offset_of!(ConstructSite, call) + offset_of!(CallIc, chunk_raw);
    a.ldr_imm(2, 0, chunk as u32);
    a.cbz(2, true, helper);
    a.mov_imm64(3, offset_of!(Chunk, instance_image) as u64);
    a.add_shifted(3, 2, 3, 0);
    a.ldr_imm(4, 3, offset_of!(InstanceImage, ready) as u32);
    a.cbz(4, true, helper);
    a.ldr_w_imm(5, 1, crate::value::CELL_SHAPE_OFFSET as u32);
    a.ldr_w_imm(6, 0, offset_of!(ConstructSite, prototype_shape) as u32);
    a.cmp_reg_w(5, 6);
    a.b_cond(C_NE, helper);
    a.ldr_w_imm(6, 0, offset_of!(ConstructSite, prototype_slot) as u32);
    a.ldrb_imm(7, 1, crate::value::CELL_SLOTS_OFFSET as u32);
    a.cmp_reg_w(6, 7);
    a.b_cond(C_HS, helper);
    a.add_shifted(7, 1, 6, 3);
    a.ldr_imm(8, 7, crate::value::OBJECT_HEADER as u32);
    a.lsr_imm(5, 8, 48);
    a.movz(6, (crate::value::PACK_OBJ >> 48) as u32, 0);
    a.cmp_reg_w(5, 6);
    a.b_cond(C_NE, helper);
    a.ubfx(8, 8, 0, 48);
    a.ldr_imm(5, 3, offset_of!(InstanceImage, cursor) as u32);
    a.ldr_imm(6, 3, offset_of!(InstanceImage, cell_size) as u32);
    emit_inline_alloc_dynamic(a, heap, 5, 6, 9, helper);
    let words = offset_of!(InstanceImage, words);
    for k in 0..crate::value::OBJECT_HEADER / 8 {
        let offset = (8 * k) as u32;
        if k == crate::value::OBJECT_PROTO_WORD {
            a.str_imm(8, 9, offset);
        } else {
            a.ldr_imm(11, 3, (words + 8 * k) as u32);
            a.str_imm(11, 9, offset);
        }
    }
    a.mov_imm64(12, crate::value::PACK_OBJ);
    a.logic_x(1, 0, 9, 12);
    #[cfg(test)]
    record_inline_construct(a);
    a.b(made);
}

#[cfg(test)]
thread_local! {
    /// The `this` objects generated code made in line ([`emit_inline_construct_this`]).
    pub(crate) static INLINE_CONSTRUCTS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// Count an in-line `this` for tests (x10 and x11 are scratch here).
#[cfg(test)]
fn record_inline_construct(a: &mut asm::Asm) {
    let address = INLINE_CONSTRUCTS.with(|count| count.as_ptr() as usize);
    a.mov_imm64(10, address as u64);
    a.ldr_imm(11, 10, 0);
    a.add_imm(11, 11, 1);
    a.str_imm(11, 10, 0);
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
