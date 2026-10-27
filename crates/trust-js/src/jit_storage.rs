//! ARM64 storage accessors: where generated code finds an object's named-field values, a binding
//! map's entries and a slot memo, in the build's layout (probed into the [`JitLayout`]).
//!
//! Named fields and binding-map entries are an `InlineVec` whose first elements stay in the
//! object or environment cell ([`JitVecLayout::Inline`]), and a slot memo is 2 bytes wide.
//! Templates never address these words themselves: they call the emitters below, which also
//! still handle an earlier layout's `Vec` and 4-byte memo. Each site gates on the matching
//! `*_fits` predicate, which holds wherever every offset the emitter bakes fits its instruction's
//! immediate.
//!
//! [`JitLayout`]: crate::value::JitLayout

use super::*;
use crate::value::JitVecLayout;

#[cfg(test)]
#[path = "jit_storage_tests.rs"]
mod tests;

/// Whether `off` is a `width`-aligned offset the scaled unsigned LDR/STR immediate reaches.
fn fits_scaled(off: usize, width: usize) -> bool {
    off.is_multiple_of(width) && off / width < 4096
}

// ---------------------------------------------------------------------------------------------
// Growable sequences, `off` bytes into the structure at `base`
// ---------------------------------------------------------------------------------------------

/// Load the sequence's element count into `dst`, zero-extended. Nothing else changes.
pub(super) fn vec_len(a: &mut asm::Asm, seq: JitVecLayout, dst: u32, base: u32, off: usize) {
    match seq {
        JitVecLayout::Vec { len, .. } => a.ldr_imm(dst, base, (off + len) as u32),
        JitVecLayout::Inline { len, .. } => a.ldr_w_imm(dst, base, (off + len) as u32),
    }
}

/// Store `src` as the sequence's element count. The caller has proven that the elements below it
/// are initialized and that it does not exceed the capacity. Nothing else changes.
pub(super) fn vec_set_len(a: &mut asm::Asm, seq: JitVecLayout, src: u32, base: u32, off: usize) {
    match seq {
        JitVecLayout::Vec { len, .. } => a.str_imm(src, base, (off + len) as u32),
        JitVecLayout::Inline { len, .. } => a.str_w_imm(src, base, (off + len) as u32),
    }
}

/// Load the number of elements the sequence has room for into `dst`: a `Vec`'s capacity; an
/// `InlineVec`'s heap capacity, or its inline capacity while the elements are in place.
/// Clobbers only `dst` (NZCV is preserved).
pub(super) fn vec_capacity(a: &mut asm::Asm, seq: JitVecLayout, dst: u32, base: u32, off: usize) {
    match seq {
        JitVecLayout::Vec { cap, .. } => a.ldr_imm(dst, base, (off + cap) as u32),
        JitVecLayout::Inline {
            cap,
            inline_capacity,
            ..
        } => {
            let done = a.new_label();
            a.ldr_w_imm(dst, base, (off + cap) as u32);
            a.cbnz(dst, false, done);
            a.movz(dst, inline_capacity as u32, 0);
            a.bind(done);
        }
    }
}

/// Load the address of the sequence's first element into `dst`, which must differ from `base`:
/// a `Vec`'s data pointer; an `InlineVec`'s in-place elements exactly while its capacity word is
/// 0, else its heap buffer pointer. Clobbers only `dst` (NZCV is preserved).
pub(super) fn vec_data(a: &mut asm::Asm, seq: JitVecLayout, dst: u32, base: u32, off: usize) {
    match seq {
        JitVecLayout::Vec { ptr, .. } => a.ldr_imm(dst, base, (off + ptr) as u32),
        JitVecLayout::Inline {
            cap, inline, heap, ..
        } => {
            assert_ne!(
                dst, base,
                "the in-place elements are addressed from the base"
            );
            let in_place = a.new_label();
            let done = a.new_label();
            a.ldr_w_imm(dst, base, (off + cap) as u32);
            a.cbz(dst, false, in_place);
            a.ldr_imm(dst, base, (off + heap) as u32);
            a.b(done);
            a.bind(in_place);
            a.add_imm(dst, base, (off + inline) as u32);
            a.bind(done);
        }
    }
}

/// Whether [`vec_len`] and [`vec_set_len`] can address the sequence `off` bytes into their base.
pub(super) fn vec_len_fits(seq: JitVecLayout, off: usize) -> bool {
    match seq {
        JitVecLayout::Vec { len, .. } => fits_scaled(off + len, 8),
        JitVecLayout::Inline { len, .. } => fits_scaled(off + len, 4),
    }
}

/// Whether [`vec_capacity`] can address the sequence `off` bytes into its base.
pub(super) fn vec_capacity_fits(seq: JitVecLayout, off: usize) -> bool {
    match seq {
        JitVecLayout::Vec { cap, .. } => fits_scaled(off + cap, 8),
        JitVecLayout::Inline {
            cap,
            inline_capacity,
            ..
        } => fits_scaled(off + cap, 4) && inline_capacity <= 0xFFFF,
    }
}

/// Whether [`vec_data`] can address the sequence `off` bytes into its base.
pub(super) fn vec_data_fits(seq: JitVecLayout, off: usize) -> bool {
    match seq {
        JitVecLayout::Vec { ptr, .. } => fits_scaled(off + ptr, 8),
        JitVecLayout::Inline {
            cap, inline, heap, ..
        } => fits_scaled(off + cap, 4) && fits_scaled(off + heap, 8) && off + inline < 4096,
    }
}

// ---------------------------------------------------------------------------------------------
// Named fields: the `Property` values of the Object at `obj` (the stored handle word plus
// `obj_from_cell`), in key-layout order
// ---------------------------------------------------------------------------------------------

/// The named-field storage within an Object.
fn fields_off(layout: &crate::value::JitLayout) -> usize {
    layout.obj_props + layout.props_entries
}

/// Load the Object's named-field count into `dst` ([`vec_len`]).
pub(super) fn fields_len(a: &mut asm::Asm, layout: &crate::value::JitLayout, dst: u32, obj: u32) {
    vec_len(a, layout.repr.props_fields, dst, obj, fields_off(layout));
}

/// Store `src` as the Object's named-field count ([`vec_set_len`]).
pub(super) fn fields_set_len(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    src: u32,
    obj: u32,
) {
    vec_set_len(a, layout.repr.props_fields, src, obj, fields_off(layout));
}

/// Load the number of named fields the Object has room for into `dst` ([`vec_capacity`]).
pub(super) fn fields_capacity(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    dst: u32,
    obj: u32,
) {
    vec_capacity(a, layout.repr.props_fields, dst, obj, fields_off(layout));
}

/// Load the address of the Object's first named-field `Property` into `dst` ([`vec_data`]).
/// Field `slot` is `entry_size * slot` bytes on.
pub(super) fn fields_data(a: &mut asm::Asm, layout: &crate::value::JitLayout, dst: u32, obj: u32) {
    vec_data(a, layout.repr.props_fields, dst, obj, fields_off(layout));
}

/// Whether [`fields_len`] and [`fields_set_len`] can address an Object's count.
pub(super) fn fields_len_fits(layout: &crate::value::JitLayout) -> bool {
    vec_len_fits(layout.repr.props_fields, fields_off(layout))
}

/// Whether [`fields_capacity`] can address an Object's capacity.
pub(super) fn fields_capacity_fits(layout: &crate::value::JitLayout) -> bool {
    vec_capacity_fits(layout.repr.props_fields, fields_off(layout))
}

/// Whether [`fields_data`] can address an Object's field storage.
pub(super) fn fields_data_fits(layout: &crate::value::JitLayout) -> bool {
    vec_data_fits(layout.repr.props_fields, fields_off(layout))
}

// ---------------------------------------------------------------------------------------------
// Binding-map entries: the ordered `(Rc<str>, Binding)` elements of a Small or Indexed map, in
// the scope at `scope` (`Env::as_ptr`)
// ---------------------------------------------------------------------------------------------

/// How the probed binding storage holds its entries (`JitRepr::scope_small`). Only sites that
/// passed [`scope_entries_fit`] ask.
fn scope_entries(layout: &crate::value::JitLayout) -> JitVecLayout {
    layout
        .repr
        .scope_small
        .expect("scope_entries_fit gates every binding-entry read")
        .storage
}

/// Load the scope's binding count into `dst` ([`vec_len`]): every record keeps its bindings'
/// values, slot by slot, in one vector.
pub(super) fn scope_entries_len(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    dst: u32,
    scope: u32,
) {
    vec_len(a, scope_entries(layout), dst, scope, layout.scope_small_vec);
}

/// Load the address of the scope's first entry into `dst` ([`vec_data`]). Entry `slot`'s
/// `Binding` is `scope_binding_stride * slot + scope_binding_offset` bytes on.
pub(super) fn scope_entries_data(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    dst: u32,
    scope: u32,
) {
    vec_data(a, scope_entries(layout), dst, scope, layout.scope_small_vec);
}

/// Whether the probed small binding storage exists and [`scope_entries_len`] and
/// [`scope_entries_data`] can address it.
pub(super) fn scope_entries_fit(layout: &crate::value::JitLayout) -> bool {
    layout.scope_small_valid
        && layout.repr.scope_small.is_some_and(|small| {
            vec_len_fits(small.storage, layout.scope_small_vec)
                && vec_data_fits(small.storage, layout.scope_small_vec)
        })
}

// ---------------------------------------------------------------------------------------------
// Slot memos (`Props::len_slot`): a narrow slot number, or the all-ones "no slot" value
// ---------------------------------------------------------------------------------------------

/// Load the slot memo `off` bytes into `base` into `dst`, zero-extended. Nothing else changes.
pub(super) fn slot_memo_load(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    dst: u32,
    base: u32,
    off: usize,
) {
    match layout.repr.slot_memo_width {
        4 => a.ldr_w_imm(dst, base, off as u32),
        2 => a.ldrh_imm(dst, base, off as u32),
        width => unreachable!("slot_memo_fits admits no {width}-byte slot memo"),
    }
}

/// Branch to `none` when the slot memo loaded into `memo` is the "no slot" value. Clobbers NZCV,
/// and `scratch` for a memo narrower than a word (the all-ones 16-bit value is no compare
/// immediate).
pub(super) fn slot_memo_branch_none(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    memo: u32,
    scratch: u32,
    none: usize,
) {
    match (layout.repr.slot_memo_width, layout.repr.slot_memo_none) {
        // ADDS WZR, Wmemo, #1 is zero exactly at u32::MAX.
        (4, u32::MAX) => a.cmn_imm_w(memo, 1),
        (2, 0xFFFF) => {
            a.movz(scratch, 0xFFFF, 0);
            a.cmp_reg_w(memo, scratch);
        }
        (width, sentinel) => {
            unreachable!("slot_memo_fits admits no {width}-byte memo with sentinel {sentinel:#x}")
        }
    }
    a.b_cond(C_EQ, none);
}

/// Whether [`slot_memo_load`] can address the memo `off` bytes into its base and
/// [`slot_memo_branch_none`] knows its "no slot" value.
pub(super) fn slot_memo_fits(layout: &crate::value::JitLayout, off: usize) -> bool {
    matches!(
        (layout.repr.slot_memo_width, layout.repr.slot_memo_none),
        (4, u32::MAX) | (2, 0xFFFF)
    ) && fits_scaled(off, layout.repr.slot_memo_width)
}
