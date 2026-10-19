//! x86-64 storage accessors: where the x64 templates find an object's named-field values and a
//! binding map's entries, in the build's layout (probed into the [`JitLayout`]), as the ARM64
//! accessors (`jit_storage.rs`) do.
//!
//! Under gc-heap both sequences are an `InlineVec` ([`JitVecLayout::Inline`]): 4-byte length
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
// Named fields: the `Property` values of the Object at `obj` (a stored cell word plus
// `obj_from_rc`), in key-layout order
// ---------------------------------------------------------------------------------------------

/// The named-field storage within an Object.
fn fields_off(layout: &JitLayout) -> usize {
    layout.obj_props + layout.props_entries
}

/// Load the Object's named-field count into `dst` ([`vec_len`]).
pub(super) fn fields_len(a: &mut Asm, layout: &JitLayout, dst: u8, obj: u8) {
    vec_len(a, layout.repr.props_fields, dst, obj, fields_off(layout));
}

/// Load the address of the Object's first named-field `Property` into `dst` ([`vec_data`]).
/// Field `slot` is `entry_size * slot` bytes on.
pub(super) fn fields_data(a: &mut Asm, layout: &JitLayout, dst: u8, obj: u8) {
    vec_data(a, layout.repr.props_fields, dst, obj, fields_off(layout));
}

/// Whether [`fields_len`] can address an Object's count.
pub(super) fn fields_len_fits(layout: &JitLayout) -> bool {
    vec_len_fits(layout.repr.props_fields, fields_off(layout))
}

/// Whether [`fields_data`] can address an Object's field storage.
pub(super) fn fields_data_fits(layout: &JitLayout) -> bool {
    vec_data_fits(layout.repr.props_fields, fields_off(layout))
}

// ---------------------------------------------------------------------------------------------
// Binding-map entries: the ordered `(Rc<str>, Binding)` elements of a Small or Indexed map, in
// the scope at `scope` (`Env::as_ptr`: under gc-heap the environment cell's address)
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

/// Load the scope's entry count into `dst` ([`vec_len`]). The caller has checked that the map's
/// tag (`JitLayout::scope_small_tag`) is Small or Indexed.
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
