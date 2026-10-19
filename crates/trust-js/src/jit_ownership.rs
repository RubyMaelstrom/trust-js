//! ARM64 emitters for the one reference count generated code still maintains: a
//! `PropertyLayout` key buffer's.
//!
//! Generated code keeps no value alive by counting. Every String, Symbol, Object and BigInt is a
//! cell of the Agent's heap, and the collector traces the cells reachable from the words
//! generated code stores (frame slots, operand words, property words, the context's packed
//! `this`), so copying, moving or dropping such a word emits no instruction, and an execution
//! word owns nothing. No generated code calls a drop helper: a store overwrites its old word, a
//! function's deferred prototype included (a cell, which dies once no word names it). A property
//! word that can be a function's deferred `prototype` is kept out of execution by
//! [`property_word_guard`]: a materialization boundary, not ownership.
//!
//! A `PropertyLayout` key buffer is an `Rc<PropertyLayoutData>`, never a value owner. Its count
//! changes only through the `layout_rc_*` emitters below, which read their own probed offset
//! (`JitRepr::layout_rc_strong_off`).

use super::*;

// ---------------------------------------------------------------------------------------------
// PropertyLayout key buffers (an `Rc<PropertyLayoutData>`, never a value owner)
// ---------------------------------------------------------------------------------------------

/// The strong-count offset of a stored `PropertyLayout` Rc pointer
/// (`JitRepr::layout_rc_strong_off`): `jit_repr` locates it on a live `Rc<PropertyLayoutData>`
/// and verifies its count before a layout is reported valid. Sites gate on [`layout_rc_fits`].
#[inline]
fn layout_rc_strong_off(layout: &crate::value::JitLayout) -> i32 {
    layout.repr.layout_rc_strong_off as i32
}

/// Whether the `layout_rc_*` emitters can address the PropertyLayout count with the unscaled
/// 9-bit LDUR/STUR offset.
pub(super) fn layout_rc_fits(layout: &crate::value::JitLayout) -> bool {
    layout.repr.layout_rc_strong_off < 256
}

/// Branch to `miss` unless the PropertyLayout at `ptr` has another owner, so replacing it
/// releases a non-final count (a last owner's destructor stays with Rust). Clobbers count and
/// NZCV.
#[inline]
pub(super) fn layout_rc_guard_shared(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    ptr: u32,
    count: u32,
    miss: usize,
) {
    a.ldur(count, ptr, layout_rc_strong_off(layout));
    a.cmp_imm_x(count, 1);
    a.b_cond(C_LS, miss);
}

/// Acquire one owner of the PropertyLayout at `ptr`. Clobbers count; flags are preserved.
#[inline]
pub(super) fn layout_rc_inc(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    ptr: u32,
    count: u32,
) {
    let strong = layout_rc_strong_off(layout);
    a.ldur(count, ptr, strong);
    a.add_imm(count, count, 1);
    a.stur(count, ptr, strong);
}

/// Release one owner of the PropertyLayout at `ptr`, proven non-final by
/// [`layout_rc_guard_shared`]. Clobbers count; flags are preserved.
#[inline]
pub(super) fn layout_rc_dec_shared(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    ptr: u32,
    count: u32,
) {
    let strong = layout_rc_strong_off(layout);
    a.ldur(count, ptr, strong);
    a.sub_imm(count, count, 1);
    a.stur(count, ptr, strong);
}
