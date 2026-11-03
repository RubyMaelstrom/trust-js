//! x86-64 storage accessors: where the x64 templates find an object's named slots and a binding
//! map's entries, in the build's layout (probed into the [`JitLayout`]), as the ARM64 accessors
//! (`jit_storage.rs`) do. Named slots are one word each, in the cell and then the record (W2d
//! step 2).
//!
//! Binding entries are an `InlineVec` ([`JitVecLayout::Inline`]): 4-byte length
//! and capacity words, with the first elements in the object or environment cell exactly while
//! the capacity word is 0, else behind a heap pointer. A length is loaded as the 32-bit word it
//! is: a 64-bit load would read `len | cap << 32`, which passes any bounds check of a spilled
//! sequence. A `Vec` keeps its 8-byte words. Templates never address these words themselves: they
//! call the emitters below, gated by the matching `*_fits` predicate, which holds wherever every
//! offset the emitter bakes fits its disp32.
//!
//! [`JitLayout`]: crate::value::JitLayout

use super::*;
use crate::value::{JitLayout, JitVecLayout};

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "jit_x64_storage_tests.rs"]
mod tests;

/// Whether `off + field` is a displacement the `[base+disp32]` forms reach.
fn fits_disp32(off: usize, field: usize) -> bool {
    off.checked_add(field)
        .is_some_and(|at| at <= i32::MAX as usize)
}

// ---------------------------------------------------------------------------------------------
// Growable sequences, `off` bytes into the structure at `base`
// ---------------------------------------------------------------------------------------------

/// Load the sequence's element count into `dst`, zero-extended: a `Vec`'s 8-byte length word, an
/// `InlineVec`'s 4-byte one. Nothing else changes.
pub(super) fn vec_len(a: &mut Asm, seq: JitVecLayout, dst: u8, base: u8, off: usize) {
    match seq {
        JitVecLayout::Vec { len, .. } => a.nr_load(dst, base, off + len),
        JitVecLayout::Inline { len, .. } => a.nr_load32(dst, base, off + len),
    }
}

/// Load the address of the sequence's first element into `dst`: a `Vec`'s data pointer; an
/// `InlineVec`'s in-place elements exactly while its capacity word is 0, else its heap buffer
/// pointer. Clobbers `dst` and the flags; `dst` may be `base`.
pub(super) fn vec_data(a: &mut Asm, seq: JitVecLayout, dst: u8, base: u8, off: usize) {
    match seq {
        JitVecLayout::Vec { ptr, .. } => a.nr_load(dst, base, off + ptr),
        JitVecLayout::Inline {
            cap, inline, heap, ..
        } => {
            let in_place = a.label();
            let done = a.label();
            a.nr_mem(false, &[0x83], 7, base, (off + cap) as i32); // cmp dword [base+cap],0
            a.bytes(&[0]);
            a.jcc(EQ, in_place);
            a.nr_load(dst, base, off + heap);
            a.jmp(done);
            a.bind(in_place);
            a.nr_lea(dst, base, (off + inline) as i32);
            a.bind(done);
        }
    }
}

/// Whether [`vec_len`] can address the sequence `off` bytes into its base.
pub(super) fn vec_len_fits(seq: JitVecLayout, off: usize) -> bool {
    match seq {
        JitVecLayout::Vec { len, .. } | JitVecLayout::Inline { len, .. } => fits_disp32(off, len),
    }
}

/// Whether [`vec_data`] can address the sequence `off` bytes into its base.
pub(super) fn vec_data_fits(seq: JitVecLayout, off: usize) -> bool {
    match seq {
        JitVecLayout::Vec { ptr, .. } => fits_disp32(off, ptr),
        JitVecLayout::Inline {
            cap, inline, heap, ..
        } => fits_disp32(off, cap) && fits_disp32(off, inline) && fits_disp32(off, heap),
    }
}

// ---------------------------------------------------------------------------------------------
// Named slots (W2d step 3): the value words of the Object at `obj` (a stored cell word plus
// `obj_from_cell`), in key-layout order, the first in the cell after its header and the rest in
// the out-of-line record, where each slot's location word says, as the ARM64 emitters address
// them
// ---------------------------------------------------------------------------------------------

/// The cell's first slot, from the Object.
fn slots_off(layout: &JitLayout) -> usize {
    layout.obj_props + layout.props_slots
}

/// The record pointer, from the Object.
fn record_off(layout: &JitLayout) -> usize {
    layout.obj_props + layout.props_record
}

/// Load the Object's named count (a `u32`) into `dst`, zero-extended.
pub(super) fn named_len(a: &mut Asm, layout: &JitLayout, dst: u8, obj: u8) {
    a.nr_load32(dst, obj, layout.obj_props + layout.props_len);
}

/// Load the address of the slot whose location word (W2d step 3: `IcState::loc`,
/// [`crate::value::location_word`]) is in register `loc`, zero-extended, into `dst`, which must
/// differ from `obj` and `loc`: the Object plus the word's offset in the cell, or the record plus
/// its offset there (bit 0 set), as ARM64's `slot_addr_loc`. Clobbers the flags.
pub(super) fn slot_addr_loc(a: &mut Asm, layout: &JitLayout, dst: u8, obj: u8, loc: u8) {
    assert!(
        dst != obj && dst != loc,
        "the slot address is built from both"
    );
    let record = a.label();
    let done = a.label();
    a.nr_mov(dst, loc);
    a.nr_shift(true, dst, 1); // shr dst, 1: CF = bit 0, the record's
    a.jcc(BELOW, record);
    a.nr_reg(0x01, dst, obj); // add dst, obj
    a.jmp(done);
    a.bind(record);
    a.nr_mem(true, &[0x03], dst, obj, record_off(layout) as i32); // add dst, [obj+record]
    a.bind(done);
}

/// Load the word of the slot whose location word `loc` is known at compile time into `dst`
/// (which may be `obj`): one load in the cell, two through the record.
pub(super) fn load_slot_loc_const(a: &mut Asm, layout: &JitLayout, dst: u8, obj: u8, loc: u16) {
    let off = usize::from(loc >> 1);
    if loc & 1 == 0 {
        a.nr_load(dst, obj, off);
    } else {
        a.nr_load(dst, obj, record_off(layout));
        a.nr_load(dst, dst, off);
    }
}

/// R11 = the owner kind of slot `slot` (a register) of the Object at `obj`, as ARM64's
/// `emit_property_kind` makes x7: 1 for a writable data property, 2 for any other, from the
/// layout's attribute byte (W2d step 2). Clobbers AX and the flags; `obj` and `slot` must not be
/// AX or R11.
pub(super) fn property_kind(a: &mut Asm, layout: &JitLayout, obj: u8, slot: u8) {
    debug_assert!(![AX, R11].contains(&obj) && ![AX, R11].contains(&slot));
    let done = a.label();
    a.nr_imm(R11, 1);
    a.nr_load(AX, obj, layout.obj_props + layout.props_layout);
    a.nr_mem(false, &[0x83], 7, AX, layout.layout_attrs_len as i32); // cmp dword [rax+len],0
    a.bytes(&[0]);
    a.jcc(EQ, done);
    a.nr_load(AX, AX, layout.layout_attrs_ptr);
    a.nr_reg(0x01, AX, slot); // add rax, slot
    a.nr_load8(AX, AX, 0);
    a.nr_op_imm(
        4,
        AX,
        (crate::value::PROP_ACCESSOR | crate::value::PROP_WRITABLE) as i32,
    ); // and
    a.nr_op_imm(7, AX, crate::value::PROP_WRITABLE as i32); // cmp
    a.jcc(EQ, done);
    a.nr_imm(R11, 2);
    a.bind(done);
}

/// Whether the named-slot emitters above can address an Object's slots, count and record, and a
/// layout's attribute bytes, with their disp32 forms.
pub(super) fn named_slots_fit(layout: &JitLayout) -> bool {
    layout.valid
        && [
            slots_off(layout) + 8 * crate::value::MAX_CELL_SLOTS,
            record_off(layout),
            layout.obj_props + layout.props_len,
            layout.record_slots,
            layout.obj_props + layout.props_layout,
            layout.layout_attrs_len,
            layout.layout_attrs_ptr,
        ]
        .iter()
        .all(|&off| fits_disp32(off, 0))
}

// ---------------------------------------------------------------------------------------------
// Binding-map entries: the ordered `(Rc<str>, Binding)` elements of a Small or Indexed map, in
// the scope at `scope` (`Env::as_ptr`: the environment cell's address)
// ---------------------------------------------------------------------------------------------

/// How the probed binding storage holds its entries (`JitRepr::scope_small`). Only sites that
/// passed [`scope_entries_fit`] ask.
fn scope_entries(layout: &JitLayout) -> JitVecLayout {
    layout
        .repr
        .scope_small
        .expect("scope_entries_fit gates every binding-entry read")
        .storage
}

/// Load the scope's binding count into `dst` ([`vec_len`]): every record keeps its bindings'
/// values, slot by slot, in one vector.
pub(super) fn scope_entries_len(a: &mut Asm, layout: &JitLayout, dst: u8, scope: u8) {
    vec_len(a, scope_entries(layout), dst, scope, layout.scope_small_vec);
}

/// Load the address of the scope's first entry into `dst` ([`vec_data`]). Entry `slot`'s
/// `Binding` is `scope_binding_stride * slot + scope_binding_offset` bytes on.
pub(super) fn scope_entries_data(a: &mut Asm, layout: &JitLayout, dst: u8, scope: u8) {
    vec_data(a, scope_entries(layout), dst, scope, layout.scope_small_vec);
}

/// Whether the probed small binding storage exists and [`scope_entries_len`] and
/// [`scope_entries_data`] can address it.
pub(super) fn scope_entries_fit(layout: &JitLayout) -> bool {
    layout.scope_small_valid
        && layout.repr.scope_small.is_some_and(|small| {
            vec_len_fits(small.storage, layout.scope_small_vec)
                && vec_data_fits(small.storage, layout.scope_small_vec)
        })
}
