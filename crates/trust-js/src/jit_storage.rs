//! ARM64 storage accessors: where generated code finds an object's named-field values, a binding
//! map's entries and a slot memo, in the build's layout (probed into the [`JitLayout`]).
//!
//! Named slots are one word each, the first [`crate::value::INLINE_NAMED_SLOTS`] in the object's
//! cell and the rest in its out-of-line record (W2d step 2); binding-map entries are an
//! `InlineVec` whose first elements stay in the environment cell ([`JitVecLayout::Inline`]), and a
//! slot memo is 2 bytes wide.
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

/// Store `src` as the sequence's element count. The caller has proven that the elements below it
/// are initialized and that it does not exceed the capacity. Nothing else changes.
#[cfg(test)]
pub(super) fn vec_set_len(a: &mut asm::Asm, seq: JitVecLayout, src: u32, base: u32, off: usize) {
    match seq {
        JitVecLayout::Vec { len, .. } => a.str_imm(src, base, (off + len) as u32),
        JitVecLayout::Inline { len, .. } => a.str_w_imm(src, base, (off + len) as u32),
    }
}

/// Load the number of elements the sequence has room for into `dst`: a `Vec`'s capacity; an
/// `InlineVec`'s heap capacity, or its inline capacity while the elements are in place.
/// Clobbers only `dst` (NZCV is preserved).
#[cfg(test)]
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

/// Whether [`vec_capacity`] can address the sequence `off` bytes into its base.
#[cfg(test)]
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

/// Whether [`vec_len`] and [`vec_set_len`] can address the sequence `off` bytes into their base.
pub(super) fn vec_len_fits(seq: JitVecLayout, off: usize) -> bool {
    match seq {
        JitVecLayout::Vec { len, .. } => fits_scaled(off + len, 8),
        JitVecLayout::Inline { len, .. } => fits_scaled(off + len, 4),
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
// Named slots (W2d step 2): the value words of the Object at `obj` (the stored handle word plus
// `obj_from_cell`), in key-layout order, [`INLINE_NAMED_SLOTS`] in the cell and the rest in the
// out-of-line record. A slot holds only its value word: its attributes are the layout's, which
// the shape proves.
// ---------------------------------------------------------------------------------------------

use crate::value::INLINE_NAMED_SLOTS;

/// The cell's first slot, from the Object.
fn slots_off(layout: &crate::value::JitLayout) -> usize {
    layout.obj_props + layout.props_slots
}

/// The record pointer, from the Object.
fn record_off(layout: &crate::value::JitLayout) -> usize {
    layout.obj_props + layout.props_record
}

/// Load the Object's named count (a `u32`) into `dst`, zero-extended.
pub(super) fn named_len(a: &mut asm::Asm, layout: &crate::value::JitLayout, dst: u32, obj: u32) {
    a.ldr_w_imm(dst, obj, (layout.obj_props + layout.props_len) as u32);
}

/// Store `src` as the Object's named count. The caller has written the slots below it and proven
/// that the Object has room for them.
pub(super) fn named_set_len(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    src: u32,
    obj: u32,
) {
    a.str_w_imm(src, obj, (layout.obj_props + layout.props_len) as u32);
}

/// Load the address of slot `slot` (a register holding an index below the count) into `dst`,
/// which must differ from `obj` and `slot`: in the cell below [`INLINE_NAMED_SLOTS`], else in the
/// record (which a count past the cell's slots implies). Clobbers NZCV.
pub(super) fn slot_addr(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    dst: u32,
    obj: u32,
    slot: u32,
) {
    assert!(
        dst != obj && dst != slot,
        "the slot address is built from both"
    );
    let record = a.new_label();
    let done = a.new_label();
    a.cmp_imm_x(slot, INLINE_NAMED_SLOTS as u32);
    a.b_cond(C_HS, record);
    a.add_imm(dst, obj, slots_off(layout) as u32);
    a.add_shifted(dst, dst, slot, 3);
    a.b(done);
    a.bind(record);
    a.ldr_imm(dst, obj, record_off(layout) as u32);
    a.add_shifted(dst, dst, slot, 3);
    // Slot INLINE + i is record_slots + 8i into the block: dst = record + 8 * slot - the cell's.
    a.sub_imm(
        dst,
        dst,
        (8 * INLINE_NAMED_SLOTS - layout.record_slots) as u32,
    );
    a.bind(done);
}

/// Load the address of slot `slot`, known at compile time, into `dst` (which may be `obj` only in
/// the cell's case): one instruction in the cell, two through the record.
pub(super) fn slot_addr_const(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    dst: u32,
    obj: u32,
    slot: usize,
) {
    if slot < INLINE_NAMED_SLOTS {
        a.add_imm(dst, obj, (slots_off(layout) + 8 * slot) as u32);
    } else {
        a.ldr_imm(dst, obj, record_off(layout) as u32);
        a.add_imm(
            dst,
            dst,
            (layout.record_slots + 8 * (slot - INLINE_NAMED_SLOTS)) as u32,
        );
    }
}

/// Branch to `miss` unless the Object at `obj` has room for one more slot past `len` (a register
/// holding its count): the cell's, or the record's capacity. Clobbers `scratch` and NZCV.
pub(super) fn guard_named_room(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    len: u32,
    obj: u32,
    scratch: u32,
    miss: usize,
) {
    let room = a.new_label();
    a.cmp_imm_x(len, INLINE_NAMED_SLOTS as u32);
    a.b_cond(C_LO, room);
    a.ldr_imm(scratch, obj, record_off(layout) as u32);
    a.cbz(scratch, true, miss);
    a.ldr_w_imm(scratch, scratch, layout.record_cap as u32);
    a.add_imm(scratch, scratch, INLINE_NAMED_SLOTS as u32);
    a.cmp_reg_x(len, scratch);
    a.b_cond(C_HS, miss);
    a.bind(room);
}

/// Branch to `miss` if the Object at `obj` keeps a key index of its own (in its record). Clobbers
/// `scratch`.
pub(super) fn guard_no_key_index(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    obj: u32,
    scratch: u32,
    miss: usize,
) {
    let none = a.new_label();
    a.ldr_imm(scratch, obj, record_off(layout) as u32);
    a.cbz(scratch, true, none);
    a.ldr_imm(scratch, scratch, layout.record_index as u32);
    a.cbnz(scratch, true, miss);
    a.bind(none);
}

/// Branch to `miss` unless slot `slot` (a register holding an index below the count) of the Object
/// at `obj` is a writable data property, by its layout's attribute byte (W2d step 2; the shape
/// proves the attributes, but a cache way records no writability): a layout whose keys are all
/// plain answers after one count load. `s1` and `s2` must differ from the inputs; both and NZCV
/// are clobbered.
pub(super) fn guard_slot_writable_data(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    obj: u32,
    slot: u32,
    s1: u32,
    s2: u32,
    miss: usize,
) {
    debug_assert!(s1 != obj && s1 != slot && s2 != obj && s2 != slot && s1 != s2);
    let ok = a.new_label();
    a.ldr_imm(s1, obj, (layout.obj_props + layout.props_layout) as u32);
    a.ldr_w_imm(s2, s1, layout.layout_attrs_len as u32);
    a.cbz(s2, false, ok);
    a.ldr_imm(s1, s1, layout.layout_attrs_ptr as u32);
    a.ldrb_reg(s1, s1, slot);
    a.tbnz(s1, crate::value::PROP_ACCESSOR.trailing_zeros(), miss);
    a.tbz(s1, crate::value::PROP_WRITABLE.trailing_zeros(), miss);
    a.bind(ok);
}

/// x7 = the owner kind of slot `slot` (a register) of the Object at `obj`, for the name and
/// Reference paths: 1 for a writable data property, 2 for any other (W2d step 2: the layout's
/// attribute byte, the meta word's role before). Stores take only kind 1; reads test the slot's
/// word for an accessor. `s1` and `s2` must differ from the inputs and from x7; both are
/// clobbered.
pub(super) fn emit_property_kind(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    obj: u32,
    slot: u32,
    s1: u32,
    s2: u32,
) {
    debug_assert!(![obj, slot, s2, 7].contains(&s1) && ![obj, slot, 7].contains(&s2));
    let done = a.new_label();
    a.movz(7, 1, 0);
    a.ldr_imm(s1, obj, (layout.obj_props + layout.props_layout) as u32);
    a.ldr_w_imm(s2, s1, layout.layout_attrs_len as u32);
    a.cbz(s2, false, done);
    a.ldr_imm(s1, s1, layout.layout_attrs_ptr as u32);
    a.ldrb_reg(s1, s1, slot);
    // Writable data: of the accessor and writable bits, exactly the writable one.
    let bits =
        asm::logical_imm_w((crate::value::PROP_ACCESSOR | crate::value::PROP_WRITABLE) as u32)
            .expect("a run of attribute bits");
    a.logic_imm_w(0, s1, s1, bits);
    a.cmp_imm_w(s1, crate::value::PROP_WRITABLE as u32);
    a.b_cond(C_EQ, done);
    a.movz(7, 2, 0);
    a.bind(done);
}

/// Whether the named-slot emitters above can address an Object's slots, count, record and its
/// record's capacity and index with their instructions' immediates.
pub(super) fn named_slots_fit(layout: &crate::value::JitLayout) -> bool {
    layout.valid
        && slots_off(layout) + 8 * INLINE_NAMED_SLOTS < 4096
        && slots_off(layout).is_multiple_of(8)
        && fits_scaled(record_off(layout), 8)
        && fits_scaled(layout.obj_props + layout.props_len, 4)
        && fits_scaled(layout.record_cap, 4)
        && fits_scaled(layout.record_index, 8)
        && layout.record_slots.is_multiple_of(8)
        && layout.record_slots <= 8 * INLINE_NAMED_SLOTS
        && fits_scaled(layout.obj_props + layout.props_layout, 8)
        && fits_scaled(layout.layout_attrs_len, 4)
        && fits_scaled(layout.layout_attrs_ptr, 8)
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
