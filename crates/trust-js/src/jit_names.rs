//! Native declarative/global name proofs. No observable operation occurs before the final
//! checked load/store: every intermediate environment, current parent, generation/layout and
//! binding/property flag stays authoritative.

use super::*;
use crate::bytecode::lexical_cache::{
    NativeNameDescriptor, NAME_GUARD_GENERATION, NAME_GUARD_IDENTITY, NAME_GUARD_LAYOUT,
    NAME_GUARD_SIZE,
};

#[cfg(test)]
thread_local! {
    pub(super) static TEST_NATIVE_DEEP_NAMES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    // Successful value load, value store, numeric update, numeric chain load, loop preload.
    pub(super) static TEST_NATIVE_NAME_COMMITS: [std::cell::Cell<usize>; 5] = const {
        [const { std::cell::Cell::new(0) }; 5]
    };
    pub(super) static TEST_NATIVE_REFERENCES: [std::cell::Cell<usize>; 3] = const {
        [std::cell::Cell::new(0), std::cell::Cell::new(0), std::cell::Cell::new(0)]
    };
}

/// Count an in-line activation in a test build (x10 and x11 scratch, before the poll).
#[cfg(all(test, target_arch = "aarch64"))]
pub(super) fn record_inline_activation(a: &mut asm::Asm) {
    let address = super::TestAddresses::current().inline_activations;
    a.mov_imm64(10, address as u64);
    a.ldr_imm(11, 10, 0);
    a.add_imm(11, 11, 1);
    a.str_imm(11, 10, 0);
}

/// A successful operation, after its final guard and value/stack mutation. No call or poll;
/// x9/x10 are scratch at these commit sites and all live words stay in their original homes.
#[cfg(all(test, target_arch = "aarch64"))]
pub(super) fn record_name_commit(a: &mut asm::Asm, kind: usize) {
    let address = super::TestAddresses::current().native_name_commits[kind];
    a.mov_imm64(9, address as u64);
    a.ldr_imm(10, 9, 0);
    a.add_imm(10, 10, 1);
    a.str_imm(10, 9, 0);
}

#[cfg(test)]
#[path = "jit_names_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "jit_packed_name_tests.rs"]
mod packed_tests;

/// x12: descriptor; x9: the current scope's `Env::as_ptr`. On success x14 is Value
/// address, x7 identifies Property(1) or Binding(0). Both store words; the flag preserves
/// descriptor/initialization/mutability rules and the actual owner. Clobbers x7,x9..x17 and NZCV
/// only.
#[cfg(all(
    target_arch = "aarch64",
    any(target_os = "linux", target_os = "macos", target_os = "windows")
))]
fn emit_deep_name_target_ptr(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    slow: usize,
    property_ok: bool,
) {
    use std::mem::offset_of;
    let loop_head = a.new_label();
    let layout_guard = a.new_label();
    let checked = a.new_label();
    let holder = a.new_label();
    let global = a.new_label();
    let holder_slot = a.new_label();
    let binding_ready = a.new_label();
    let done = a.new_label();
    if !deep_layout_supported(layout) {
        a.b(slow);
        return;
    }
    a.cbz(12, true, slow);
    a.ldr_imm(10, 12, offset_of!(NativeNameDescriptor, guards) as u32);
    a.ldr_imm(11, 12, offset_of!(NativeNameDescriptor, len) as u32);
    a.cbz(11, true, slow);
    a.cmp_imm_x(11, 16);
    a.b_cond(C_HI, slow);
    a.bind(loop_head);
    a.ldr_imm(13, 9, layout.scope_with as u32);
    a.mov_imm64(14, layout.scope_with_none);
    a.cmp_reg_x(13, 14);
    a.b_cond(C_NE, slow);
    a.ldr_imm(13, 10, NAME_GUARD_IDENTITY as u32);
    a.cmp_reg_x(13, 9);
    a.b_cond(C_NE, layout_guard);
    a.ldr_w_imm(13, 10, NAME_GUARD_GENERATION as u32);
    a.cmn_imm_w(13, 1);
    a.b_cond(C_EQ, layout_guard);
    a.ldr_w_imm(14, 9, layout.scope_gen as u32);
    a.cmp_reg_w(13, 14);
    a.b_cond(C_EQ, checked);
    a.bind(layout_guard);
    // A layout identity (a shared descriptor's address) proves the key set; zero proves none.
    a.ldr_imm(13, 10, NAME_GUARD_LAYOUT as u32);
    a.cbz(13, true, slow);
    a.ldr_imm(14, 9, layout.scope_layout as u32);
    a.cmp_reg_x(13, 14);
    a.b_cond(C_NE, slow);
    a.bind(checked);
    a.sub_imm(11, 11, 1);
    a.cbz(11, true, holder);
    a.ldr_imm(9, 9, layout.scope_parent as u32);
    a.cbz(9, true, slow);
    a.add_imm(10, 10, NAME_GUARD_SIZE as u32);
    a.b(loop_head);
    a.bind(holder);
    a.ldr_w_imm(13, 12, offset_of!(NativeNameDescriptor, kind) as u32);
    a.cbnz(13, false, global);
    // The exact holder can reuse the raw address. Fresh fixed-layout holders must derive
    // their own slot; never dereference a previous invocation's Binding pointer.
    a.ldr_imm(13, 10, NAME_GUARD_IDENTITY as u32);
    a.cmp_reg_x(13, 9);
    a.b_cond(C_NE, holder_slot);
    a.ldr_w_imm(13, 10, NAME_GUARD_GENERATION as u32);
    a.cmn_imm_w(13, 1);
    a.b_cond(C_EQ, holder_slot);
    a.ldr_w_imm(14, 9, layout.scope_gen as u32);
    a.cmp_reg_w(13, 14);
    a.b_cond(C_NE, holder_slot);
    a.ldr_imm(14, 12, offset_of!(NativeNameDescriptor, binding) as u32);
    a.b(binding_ready);
    a.bind(holder_slot);
    if scope_entries_fit(layout) {
        a.ldr_imm(13, 12, offset_of!(NativeNameDescriptor, slot) as u32);
        scope_entries_len(a, layout, 14, 9);
        a.cmp_reg_x(13, 14);
        a.b_cond(C_HS, slow);
        scope_entries_data(a, layout, 14, 9);
        a.add_shifted(14, 14, 13, 3); // one word a slot
    } else {
        a.b(slow);
    }
    a.bind(binding_ready);
    a.cbz(14, true, slow);
    a.movz(7, 0, 0);
    a.b(done);
    a.bind(global);
    if !property_ok {
        a.b(slow);
    }
    a.cmp_imm_w(13, 1);
    a.b_cond(C_NE, slow);
    a.ldr_imm(13, 19, offset_of!(JitCtx, genv) as u32);
    a.cmp_reg_x(13, 9);
    a.b_cond(C_NE, slow);
    a.ldr_imm(14, 19, offset_of!(JitCtx, global_body) as u32);
    a.cbz(14, true, slow);
    a.ldrb_imm(13, 14, layout.obj_kind as u32);
    guard_exotic_none(a, 13, slow);
    a.ldrb_imm(13, 14, layout.obj_flags as u32);
    a.tbz(13, flag_bit(crate::value::FLAG_IC_PLAIN), slow);
    a.ldr_w_imm(13, 14, (layout.obj_props + layout.props_shape) as u32);
    a.ldr_w_imm(15, 12, offset_of!(NativeNameDescriptor, shape) as u32);
    a.cmp_reg_w(13, 15);
    a.b_cond(C_NE, slow);
    a.ldr_imm(13, 12, offset_of!(NativeNameDescriptor, slot) as u32);
    named_len(a, layout, 15, 14);
    a.cmp_reg_x(13, 15);
    a.b_cond(C_HS, slow);
    // The slot's location word (W2d step 3), which the descriptor's fill recorded.
    a.ldr_w_imm(16, 12, offset_of!(NativeNameDescriptor, loc) as u32);
    slot_addr_loc(a, layout, 15, 14, 16);
    emit_property_kind(a, layout, 14, 13, 16, 17); // x7 = 1 (writable) or 2
    a.mov(14, 15);
    a.bind(done);
}

pub(super) fn deep_layout_supported(layout: &crate::value::JitLayout) -> bool {
    layout.scope_parent_valid
        && layout.scope_with_valid
        && layout.scope_parent.is_multiple_of(8)
        && layout.scope_parent / 8 < 4096
        && layout.scope_with < 4096
        && layout.scope_layout.is_multiple_of(8)
        && layout.scope_layout / 8 < 4096
        && (!layout.scope_small_valid
            || (scope_entries_fit(layout)
                && layout.scope_binding_stride == 8
                && layout.scope_binding_offset == 0))
}

/// Value-read variant of the resolution proof. Capturing a Reference uses the target proof
/// above directly: TDZ, imports and accessors are deliberately tested only at GetValue.
pub(super) fn emit_deep_name_value_ptr(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    slow: usize,
    property_ok: bool,
) {
    emit_deep_name_target_ptr(a, layout, slow, property_ok);
    emit_target_value_ptr(a, layout, slow);
    #[cfg(test)]
    {
        let address = super::TestAddresses::current().native_deep_names;
        a.mov_imm64(15, address as u64);
        a.ldr_imm(16, 15, 0);
        a.add_imm(16, 16, 1);
        a.str_imm(16, 15, 0);
    }
}

/// x14 = a binding slot (x7 0) or a property's slot (x7 1 or 2) → its value's address, after the
/// live checks: a binding's TDZ, the empty word (the proof that found it covers its attributes: a
/// deep cache's fill refused an import, a Reference names a plain binding only), or a data
/// property's word (an accessor's, under PACK_ACCESSOR, takes the checked path; W2d step 2).
/// Clobbers x13 and x15.
fn emit_target_value_ptr(a: &mut asm::Asm, _layout: &crate::value::JitLayout, slow: usize) {
    let property = a.new_label();
    let done = a.new_label();
    a.cbnz(7, false, property);
    emit_tdz_guard(a, 13, 14, 15, slow);
    a.b(done);
    a.bind(property);
    a.ldr_imm(13, 14, 0);
    branch_accessor_word(a, 13, 13, slow);
    a.bind(done);
}

/// Probe one captured base without resolving the name again. x8 points at PreparedReference;
/// x14 returns Binding/Property and x7 its kind. Failed hints retain exact owners for the helper.
fn emit_reference_target_ptr(a: &mut asm::Asm, layout: &crate::value::JitLayout, slow: usize) {
    use crate::eval::PreparedReference as R;
    use std::mem::offset_of;
    let property = a.new_label();
    let done = a.new_label();
    a.ldr_w_imm(7, 8, offset_of!(R, kind) as u32);
    a.cbnz(7, false, property);
    a.ldr_imm(9, 8, offset_of!(R, scope) as u32);
    a.cbz(9, true, slow);
    a.ldr_w_imm(10, 8, offset_of!(R, generation) as u32);
    a.cmn_imm_w(10, 1);
    a.b_cond(C_EQ, slow);
    a.ldr_w_imm(11, 9, layout.scope_gen as u32);
    a.cmp_reg_w(10, 11);
    a.b_cond(C_NE, slow);
    a.ldr_imm(14, 8, offset_of!(R, binding) as u32);
    a.cbz(14, true, slow);
    a.b(done);
    a.bind(property);
    a.cmp_imm_w(7, 1);
    a.b_cond(C_NE, slow);
    a.ldr_imm(9, 8, offset_of!(R, object) as u32);
    a.cbz(9, true, slow);
    a.add_imm(9, 9, layout.obj_from_cell as u32);
    a.ldrb_imm(10, 9, layout.obj_kind as u32);
    guard_exotic_ordinary(a, layout, 10, 10, slow);
    a.ldrb_imm(10, 9, layout.obj_flags as u32);
    a.tbz(10, flag_bit(crate::value::FLAG_IC_PLAIN), slow);
    a.ldr_w_imm(10, 8, offset_of!(R, shape) as u32);
    a.cbz(10, false, slow);
    a.cmn_imm_w(10, 1);
    a.b_cond(C_EQ, slow);
    a.ldr_w_imm(11, 9, (layout.obj_props + layout.props_shape) as u32);
    a.cmp_reg_w(10, 11);
    a.b_cond(C_NE, slow);
    a.ldr_imm(10, 8, offset_of!(R, slot) as u32);
    named_len(a, layout, 11, 9);
    a.cmp_reg_x(10, 11);
    a.b_cond(C_HS, slow);
    // The slot's location word (W2d step 3), which the Reference's hint recorded.
    a.ldr_w_imm(11, 8, offset_of!(R, loc) as u32);
    slot_addr_loc(a, layout, 14, 9, 11);
    emit_property_kind(a, layout, 9, 10, 11, 13); // x7 = 1 (writable) or 2
    a.bind(done);
}

/// Whether the Reference `reference` read at `pc` can name `prototype`, the one key whose value
/// can be a function's deferred prototype. The compiler gives every `ResolveNameRef` a fresh
/// Reference, emitted before the Reference's uses, so the nearest one before `pc` names it; a
/// Reference whose `ResolveNameRef` is not found is treated as if it could.
fn reference_may_name_prototype<C: JitChunk + ?Sized>(
    chunk: &C,
    pc: usize,
    reference: u16,
) -> bool {
    chunk.jit_ops()[..pc]
        .iter()
        .rev()
        .find_map(|op| match *op {
            crate::bytecode::Op::ResolveNameRef(name, resolved) if resolved == reference => {
                Some(chunk.jit_name(name) == "prototype")
            }
            _ => None,
        })
        .unwrap_or(true)
}

/// Whole-operation native Reference template. All proof/owner guards precede mutation. The
/// canonical slot owns the base/name across RHS effects; no hint miss restarts resolution.
pub(super) fn emit_reference_op<C: JitChunk + ?Sized>(
    a: &mut asm::Asm,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    chunk: &C,
    op: &crate::bytecode::Op,
    pc: u32,
    unwind: usize,
) -> bool {
    use crate::bytecode::Op;
    use crate::eval::{PreparedReference as R, PreparedReferenceSlot as S};
    use std::mem::{offset_of, size_of};
    let reference = match *op {
        Op::ResolveNameRef(_, reference) | Op::LoadRef(reference) | Op::StoreRef(reference) => {
            reference
        }
        _ => return false,
    };
    if !update_name_inlinable(layout)
        || !layout.key_probe_ok
        || !deep_layout_supported(layout)
        || !ilayout.valid
    {
        return false;
    }
    let slow = a.new_label();
    let done = a.new_label();
    let fast_done = a.new_label();
    a.ldr_imm(8, 19, offset_of!(JitCtx, references_raw) as u32);
    a.cbz(8, true, slow);
    a.mov_imm64(9, reference as u64 * size_of::<S>() as u64);
    a.add_shifted(8, 8, 9, 0);
    a.ldr_imm(9, 8, offset_of!(S, present) as u32);
    a.cmp_imm_x(9, 1);
    a.b_cond(C_NE, slow);
    a.add_imm(8, 8, offset_of!(S, reference) as u32);
    match *op {
        Op::ResolveNameRef(name, _) => {
            // Site and shared-name addresses remain stable for this pinned Chunk. A helper
            // refill can replace its descriptor, so load that descriptor afresh each time.
            a.mov_imm64(12, chunk.jit_resolution_cache_ptr(name) as u64);
            a.ldr_imm(12, 12, 0);
            a.ldr_imm(9, 19, offset_of!(JitCtx, env_raw) as u32);
            emit_deep_name_target_ptr(a, layout, slow, true);
            a.ldr_w_imm(13, 8, offset_of!(R, kind) as u32);
            a.cmp_reg_w(13, 7);
            a.b_cond(C_NE, slow);
            a.mov_imm64(15, chunk.jit_shared_name_ptr(name) as u64);
            for word in [layout.str_ptr_word, layout.str_len_word] {
                a.ldr_imm(13, 8, (offset_of!(R, name) + word) as u32);
                a.ldr_imm(16, 15, word as u32);
                a.cmp_reg_x(13, 16);
                a.b_cond(C_NE, slow);
            }
            let property = a.new_label();
            let strict = a.new_label();
            a.cbnz(7, false, property);
            a.ldr_imm(13, 8, offset_of!(R, scope) as u32);
            a.cbz(13, true, slow);
            a.cmp_reg_x(13, 9);
            a.b_cond(C_NE, slow);
            // Commit only non-owning address hints; exact record/name owners stay unchanged.
            a.ldr_w_imm(13, 9, layout.scope_gen as u32);
            a.str_w_imm(13, 8, offset_of!(R, generation) as u32);
            a.str_imm(14, 8, offset_of!(R, binding) as u32);
            a.b(strict);
            a.bind(property);
            a.ldr_imm(13, 8, offset_of!(R, object) as u32);
            a.cbz(13, true, slow);
            a.add_imm(13, 13, layout.obj_from_cell as u32);
            a.ldr_imm(15, 19, offset_of!(JitCtx, global_body) as u32);
            a.cmp_reg_x(13, 15);
            a.b_cond(C_NE, slow);
            a.ldr_w_imm(13, 12, offset_of!(NativeNameDescriptor, shape) as u32);
            a.str_w_imm(13, 8, offset_of!(R, shape) as u32);
            a.ldr_imm(13, 12, offset_of!(NativeNameDescriptor, slot) as u32);
            a.str_imm(13, 8, offset_of!(R, slot) as u32);
            a.bind(strict);
            a.ldr_imm(13, 19, offset_of!(JitCtx, interp) as u32);
            let (strict_base, strict_offset) = byte_field_address(a, 13, ilayout.strict, 16);
            a.ldrb_imm(13, strict_base, strict_offset);
            a.str_w_imm(13, 8, offset_of!(R, strict) as u32);
            a.b(fast_done);
        }
        Op::LoadRef(_) => {
            emit_reference_target_ptr(a, layout, slow);
            emit_target_value_ptr(a, layout, slow);
            let prototype_key = reference_may_name_prototype(chunk, pc as usize, reference);
            emit_load_name_value(a, slow, false, prototype_key);
            a.b(fast_done);
        }
        Op::StoreRef(_) => {
            emit_reference_target_ptr(a, layout, slow);
            emit_target_value_ptr(a, layout, slow);
            emit_store_name_value(
                a,
                layout,
                NameStoreOwner::Reference(reference),
                slow,
                fast_done,
            );
        }
        _ => unreachable!(),
    }
    a.bind(fast_done);
    #[cfg(test)]
    {
        let index = match op {
            Op::ResolveNameRef(..) => 0,
            Op::LoadRef(_) => 1,
            _ => 2,
        };
        let address = super::TestAddresses::current().native_references[index];
        a.mov_imm64(9, address as u64);
        a.ldr_imm(10, 9, 0);
        a.add_imm(10, 10, 1);
        a.str_imm(10, 9, 0);
    }
    a.b(done);
    a.bind(slow);
    emit_op_helper(a, H_REFERENCE_OP, pc, unwind);
    a.bind(done);
    true
}
