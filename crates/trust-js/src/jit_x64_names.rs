//! ECMA-262 e28783d5: GetIdentifierReference, GetValue and PutValue.
//! Resolution is a distinct operation: capture must not inspect TDZ/value/import flags.
//! A later load/store follows the captured owner, never repeats name resolution.
use super::*;
use crate::bytecode::lexical_cache::{
    NativeNameDescriptor, DEEP_NAME_IC, NAME_GUARD_GENERATION, NAME_GUARD_IDENTITY,
    NAME_GUARD_LAYOUT, NAME_GUARD_SIZE,
};
use crate::bytecode::UpdKind;
use crate::eval::{PreparedReference, PreparedReferenceSlot};
use crate::value::JitLayout;
use std::mem::{offset_of, size_of};

#[cfg(test)]
thread_local! {
    pub(super) static TEST_NATIVE_DEEP_NAMES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static TEST_NATIVE_NAME_OPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    // Successful load, store and numeric update, after the value/stack mutation.
    pub(super) static TEST_NATIVE_NAME_COMMITS: [std::cell::Cell<usize>; 3] = const {
        [const { std::cell::Cell::new(0) }; 3]
    };
    pub(super) static TEST_NATIVE_REFERENCES: [std::cell::Cell<usize>; 3] = const {
        [std::cell::Cell::new(0), std::cell::Cell::new(0), std::cell::Cell::new(0)]
    };
}

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "jit_x64_names_tests.rs"]
mod tests;

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "jit_packed_name_tests.rs"]
mod packed_tests;

/// Thread-affine test instrumentation only; production code contains no counter operations.
/// AX/DX are volatile after an operation commits; no owner or stack pointer is changed.
#[cfg(test)]
fn record(a: &mut Asm, address: usize) {
    a.nr_imm(AX, address as u64);
    a.nr_mem(true, &[0xff], 0, AX, 0);
}

fn valid(l: &JitLayout) -> bool {
    l.valid
        && l.binding_value == 0
        && l.scope_binding_stride == 8
        && l.scope_binding_offset == 0
        && storage::fields_len_fits(l)
        && storage::fields_data_fits(l)
        && l.scope_parent_valid
        && l.scope_with_valid
        && l.key_probe_ok
        && l.entry_accessor == l.entry_value + 8
        && [
            l.scope_gen,
            l.scope_layout,
            l.scope_parent,
            l.scope_with,
            l.scope_small_vec,
            l.scope_binding_stride,
            l.scope_binding_offset,
            l.binding_value,
            l.obj_props + l.props_shape,
            l.obj_ic_plain,
            l.obj_exotic,
            l.entry_size,
            l.entry_value,
            l.entry_accessor,
            l.entry_writable,
            l.obj_from_cell,
            l.str_ptr_word,
            l.str_len_word,
        ]
        .iter()
        .all(|off| *off <= i32::MAX as usize)
}

fn scope_plain(a: &mut Asm, l: &JitLayout, slow: usize) {
    a.nr_load(AX, R9, l.scope_with);
    a.nr_imm(DX, l.scope_with_none);
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
}

/// R9=Object (a stored cell word plus `obj_from_cell`), AX=shape, DX=slot. Output CX=Property,
/// R11=1. No value flags yet. `valid` admitted the layout. A global object is never a Map or Set
/// (the VM checks `Exotic::None` there); a Reference's object base (`ordinary`) may be any object
/// with ordinary internal methods, as ARM64's Reference template admits.
fn global_target(a: &mut Asm, l: &JitLayout, ordinary: bool, slow: usize) {
    a.nr_zero_guard(R9, slow);
    a.nr_zero_guard(AX, slow);
    a.nr_max_guard(AX, CX, slow);
    a.nr_load32(CX, R9, l.obj_props + l.props_shape);
    a.nr_cmp(CX, AX);
    a.jcc(NE, slow);
    if ordinary {
        a.guard_exotic_ordinary(l, R9, CX, slow);
    } else {
        a.nr_load8(CX, R9, l.obj_exotic);
        a.nr_op_imm(7, CX, l.exotic_none_tag as i32);
        a.jcc(NE, slow);
    }
    a.nr_load8(CX, R9, l.obj_ic_plain);
    a.nr_zero_guard(CX, slow);
    storage::fields_len(a, l, CX, R9);
    a.nr_cmp(DX, CX);
    a.jcc(AE, slow);
    storage::fields_data(a, l, CX, R9);
    a.nr_mul_imm(DX, l.entry_size);
    a.nr_reg(0x01, CX, DX);
    a.nr_imm(R11, 1);
}

/// R8 supplies a validated layout's slot index; R9 is its actual current scope. Address
/// the live binding slots (one word each) instead of retaining an identity-specific slot address.
fn layout_binding_target(a: &mut Asm, l: &JitLayout, slot_offset: usize, slow: usize) {
    if storage::scope_entries_fit(l) {
        a.nr_load(DX, R8, slot_offset);
        storage::scope_entries_len(a, l, CX, R9);
        a.nr_cmp(DX, CX);
        a.jcc(AE, slow);
        storage::scope_entries_data(a, l, CX, R9);
        a.nr_shift(false, DX, 3);
        a.nr_reg(0x01, CX, DX);
    } else {
        a.jmp(slow);
    }
}

/// R8=live descriptor, R9=the current scope (`Env::as_ptr`: the environment cell's
/// address). Output CX=Binding/Property, R11=0/1, R9=holder scope/Object. Preserve descriptor
/// through commit.
fn deep_target(a: &mut Asm, l: &JitLayout, slow: usize) {
    let loop_head = a.label();
    let layout = a.label();
    let checked = a.label();
    let holder = a.label();
    let fresh = a.label();
    let binding = a.label();
    let global = a.label();
    let done = a.label();
    a.nr_zero_guard(R8, slow);
    a.nr_load(R10, R8, offset_of!(NativeNameDescriptor, guards));
    a.nr_load(R11, R8, offset_of!(NativeNameDescriptor, len));
    a.nr_zero_guard(R11, slow);
    a.nr_op_imm(7, R11, 16);
    a.jcc(ABOVE, slow);
    a.bind(loop_head);
    scope_plain(a, l, slow);
    a.nr_load(AX, R10, NAME_GUARD_IDENTITY);
    a.nr_cmp(AX, R9);
    a.jcc(NE, layout);
    a.nr_load32(AX, R10, NAME_GUARD_GENERATION);
    a.nr_max_guard(AX, DX, layout);
    a.nr_load32(DX, R9, l.scope_gen);
    a.nr_cmp(AX, DX);
    a.jcc(EQ, checked);
    a.bind(layout);
    // A layout identity (a shared descriptor's address) proves the key set; zero proves none.
    a.nr_load(AX, R10, NAME_GUARD_LAYOUT);
    a.nr_zero_guard(AX, slow);
    a.nr_load(DX, R9, l.scope_layout);
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
    a.bind(checked);
    a.nr_op_imm(5, R11, 1);
    a.jcc(EQ, holder);
    a.nr_load(R9, R9, l.scope_parent);
    a.nr_zero_guard(R9, slow);
    a.nr_op_imm(0, R10, NAME_GUARD_SIZE as i32);
    a.jmp(loop_head);
    a.bind(holder);
    a.nr_load32(AX, R8, offset_of!(NativeNameDescriptor, kind));
    a.nr_test(AX);
    a.jcc(NE, global);
    a.nr_load(AX, R10, NAME_GUARD_IDENTITY);
    a.nr_cmp(AX, R9);
    a.jcc(NE, fresh);
    a.nr_load32(AX, R10, NAME_GUARD_GENERATION);
    a.nr_max_guard(AX, DX, fresh);
    a.nr_load32(DX, R9, l.scope_gen);
    a.nr_cmp(AX, DX);
    a.jcc(NE, fresh);
    a.nr_load(CX, R8, offset_of!(NativeNameDescriptor, binding));
    a.jmp(binding);
    a.bind(fresh);
    layout_binding_target(a, l, offset_of!(NativeNameDescriptor, slot), slow);
    a.bind(binding);
    a.nr_zero_guard(CX, slow);
    a.nr_imm(R11, 0);
    a.jmp(done);
    a.bind(global);
    a.nr_op_imm(7, AX, 1);
    a.jcc(NE, slow);
    a.nr_load(DX, CTX, offset_of!(super::super::JitCtx, genv));
    a.nr_cmp(DX, R9);
    a.jcc(NE, slow);
    a.nr_load(R9, CTX, offset_of!(super::super::JitCtx, global_body));
    a.nr_load32(AX, R8, offset_of!(NativeNameDescriptor, shape));
    a.nr_load(DX, R8, offset_of!(NativeNameDescriptor, slot));
    global_target(a, l, false, slow);
    a.bind(done);
    #[cfg(test)]
    record(
        a,
        TEST_NATIVE_DEEP_NAMES.with(|count| count.as_ptr() as usize),
    );
}

/// Same result as deep_target, accepting the own/fixed/depth-one/global compact IC ABI.
fn name_target(a: &mut Asm, l: &JitLayout, cache: usize, slow: usize) {
    use crate::bytecode::{
        FIXED_NAME_IC, NAME_IC_OFF_ACT_GEN, NAME_IC_OFF_BINDING, NAME_IC_OFF_ENV, NAME_IC_OFF_GEN,
        NAME_IC_OFF_LAYOUT,
    };
    let deep = a.label();
    let fixed = a.label();
    let direct = a.label();
    let depth_one = a.label();
    let generation = a.label();
    let parent = a.label();
    let binding = a.label();
    let done = a.label();
    a.nr_imm(R8, cache as u64);
    a.nr_load(R9, CTX, offset_of!(super::super::JitCtx, env_raw));
    scope_plain(a, l, slow);
    a.nr_load(R10, R8, NAME_IC_OFF_ENV as usize);
    a.nr_op_imm(7, R10, DEEP_NAME_IC as i32);
    a.jcc(EQ, deep);
    a.nr_op_imm(7, R10, FIXED_NAME_IC as i32);
    a.jcc(EQ, fixed);
    a.nr_load32(R11, R8, NAME_IC_OFF_GEN as usize);
    a.nr_max_guard(R11, AX, slow);
    a.nr_cmp(R10, R9);
    a.jcc(EQ, direct);
    a.nr_mov(AX, R10);
    a.nr_op_imm(4, AX, 2);
    a.nr_test(AX);
    a.jcc(NE, depth_one);
    a.nr_load32(AX, R9, l.scope_gen);
    a.nr_cmp(AX, R11);
    a.jcc(NE, slow);
    a.nr_lea(AX, R9, 1);
    a.nr_cmp(AX, R10);
    a.jcc(NE, slow);
    a.nr_load(R9, CTX, offset_of!(super::super::JitCtx, global_body));
    a.nr_load(AX, R8, NAME_IC_OFF_BINDING as usize);
    a.nr_mov(DX, AX);
    a.nr_shift(true, AX, 32);
    // A 32-bit self move zero-extends the compact slot without sign extension.
    a.bytes(&[0x89, 0xd2]);
    global_target(a, l, false, slow);
    a.jmp(done);
    a.bind(direct);
    a.nr_load32(AX, R9, l.scope_gen);
    a.nr_cmp(AX, R11);
    a.jcc(NE, slow);
    a.jmp(binding);
    a.bind(depth_one);
    a.nr_mov(AX, R10);
    a.nr_op_imm(4, AX, 4);
    a.nr_test(AX);
    a.jcc(EQ, generation);
    // Layout mode: the activation's layout identity, a shared descriptor's address.
    a.nr_load(DX, R8, NAME_IC_OFF_LAYOUT as usize);
    a.nr_zero_guard(DX, slow);
    a.nr_load(AX, R9, l.scope_layout);
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
    a.jmp(parent);
    a.bind(generation);
    a.nr_load32(DX, R8, NAME_IC_OFF_ACT_GEN as usize);
    a.nr_max_guard(DX, AX, slow);
    a.nr_load32(AX, R9, l.scope_gen);
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
    a.bind(parent);
    a.nr_load(R9, R9, l.scope_parent);
    a.nr_zero_guard(R9, slow);
    a.nr_op_imm(4, R10, -8);
    a.nr_cmp(R10, R9);
    a.jcc(NE, slow);
    scope_plain(a, l, slow);
    a.nr_load32(AX, R9, l.scope_gen);
    a.nr_cmp(AX, R11);
    a.jcc(NE, slow);
    a.bind(binding);
    a.nr_load(CX, R8, NAME_IC_OFF_BINDING as usize);
    a.nr_zero_guard(CX, slow);
    a.nr_imm(R11, 0);
    a.jmp(done);
    a.bind(fixed);
    a.nr_load(AX, R8, NAME_IC_OFF_LAYOUT as usize);
    a.nr_zero_guard(AX, slow);
    a.nr_load(DX, R9, l.scope_layout);
    // Fixed layout identity, not a generation or an identity-specific slot address.
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
    layout_binding_target(a, l, NAME_IC_OFF_BINDING as usize, slow);
    a.nr_imm(R11, 0);
    a.jmp(done);
    a.bind(deep);
    a.nr_load(R8, R8, NAME_IC_OFF_BINDING as usize);
    deep_target(a, l, slow);
    a.bind(done);
}

/// Convert Binding/Property address to its value address, after live semantic flags: a
/// binding's TDZ (the proof that found it covers its attributes: a cache's fill refused an
/// import, and a writing site's an immutable binding; a Reference names a plain binding only),
/// a property's descriptor.
fn value_address(a: &mut Asm, l: &JitLayout, store: bool, slow: usize) {
    let property = a.label();
    let done = a.label();
    a.nr_test(R11);
    a.jcc(NE, property);
    if store {
        // TDZ, the empty word: a load's execution word guard refuses it too.
        a.nr_load(AX, CX, 0);
        a.nr_imm(DX, crate::value::PACK_EMPTY);
        a.nr_cmp(AX, DX);
        a.jcc(EQ, slow);
    }
    a.jmp(done);
    a.bind(property);
    a.nr_load(AX, CX, l.entry_accessor);
    a.nr_op_imm(4, AX, crate::value::PROP_ACCESSOR as i32);
    a.nr_test(AX);
    a.jcc(NE, slow);
    if store {
        a.nr_load(AX, CX, l.entry_writable);
        a.nr_op_imm(4, AX, crate::value::PROP_WRITABLE as i32);
        a.nr_zero_guard(AX, slow);
    }
    a.nr_lea(CX, CX, l.entry_value as i32);
    a.bind(done);
}

/// Binding and Property values share one execution word. CX/R8..R11 remain unchanged;
/// the following execution guard preserves the EMPTY and property-only boundaries.
fn read_word(a: &mut Asm) {
    a.nr_load(AX, CX, 0);
}

/// Values crossing into execution storage cannot carry TDZ/EMPTY or a deferred prototype.
/// References need no ownership decoding. The checked path materializes property-only words.
fn execution_word_guard(a: &mut Asm, slow: usize) {
    a.nr_mov(DX, AX);
    a.nr_shift(true, DX, 48);
    a.nr_op_imm(7, DX, (crate::value::PACK_EMPTY >> 48) as i32);
    a.jcc(EQ, slow);
    a.nr_op_imm(7, DX, (crate::value::PACK_LAZY_PROTO >> 48) as i32);
    a.jcc(BELOW ^ 1, slow);
}

fn load_value(a: &mut Asm, for_call: bool, slow: usize) {
    read_word(a);
    execution_word_guard(a, slow);
    if for_call {
        a.nr_imm(DX, PACK_UNDEFINED);
        a.nr_store(SP, 0, DX);
        a.add_sp(8);
    }
    a.nr_store(SP, 0, AX);
    a.add_sp(8);
}

/// Record the actual owner's card (S6 step10b, `gc_core::card_mark`). R9 still holds that
/// Env/Object after the word store: no RHS width conversion or spill can overwrite it.
/// An Object's body is inside the same card as its cell start. Clobbers AX and R9.
fn card_mark(a: &mut Asm) {
    use crate::gc_core::{CARD_SHIFT, CARD_TABLE_OFFSET, PAGE_SIZE};
    // The 16-bit move takes the owner's offset in its page.
    const _: () = assert!(PAGE_SIZE == 1 << 16 && CARD_TABLE_OFFSET < i32::MAX as usize);
    a.bytes(&[0x41, 0x0f, 0xb7, 0xc1]); // movzx eax, r9w: the owner's offset in its page
    a.bytes(&[0xc1, 0xe8, CARD_SHIFT as u8]); // shr eax, 7: its card
    a.nr_op_imm(4, R9, -(PAGE_SIZE as i32)); // and r9, -65536: the page base
    a.bytes(&[0x4a, 0x8d, 0x84, 0x08]); // lea rax, [rax + r9 + 65024]: the card's byte
    a.bytes(&(CARD_TABLE_OFFSET as i32).to_le_bytes());
    a.bytes(&[0xc6, 0x00, 0x01]); // mov byte [rax], 1
                                  // mov byte [r9 + 65535], 1: the page's card summary (`gc_core::CARD_SUMMARY`).
    const _: () = assert!(crate::gc_core::CARD_SUMMARY == 0xffff);
    a.bytes(&[0x41, 0xc6, 0x81, 0xff, 0xff, 0x00, 0x00, 0x01]);
}

/// All semantic guards precede the store commit. The write barrier records the store in its
/// actual owner's card before any safepoint; R11 keeps its Binding/Property meaning.
fn store_value(a: &mut Asm, slow: usize) {
    a.nr_mem(true, &[0x8b], AX, SP, -8);
    execution_word_guard(a, slow);
    a.nr_store(CX, 0, AX);
    card_mark(a);
    a.add_sp(-8);
}

/// Numeric ++/--, after live mutability/writability guards. Non-Numbers and NaNs
/// stay on the checked ToNumeric boundary. The only mutation follows every guard.
fn update_value(a: &mut Asm, kind: UpdKind, slow: usize) {
    a.nr_load(R8, CX, 0);
    a.bytes(&[0x66, 0x49, 0x0f, 0x6e, 0xc0]); // movq xmm0,r8 (old Number)
    a.bytes(&[0x66, 0x0f, 0x2e, 0xc0]); // ucomisd xmm0,xmm0
    a.jcc(0x8a, slow); // NaN uses checked canonicalization
    a.nr_imm(AX, 1.0f64.to_bits());
    a.bytes(&[0x66, 0x48, 0x0f, 0x6e, 0xc8]); // movq xmm1,rax
    let dec = matches!(
        kind,
        UpdKind::PreDec | UpdKind::PostDec | UpdKind::DecDiscard
    );
    a.bytes(&[0xf2, 0x0f, if dec { 0x5c } else { 0x58 }, 0xc1]);
    a.bytes(&[0x66, 0x48, 0x0f, 0x7e, 0xc0]); // movq rax,xmm0
    a.nr_store(CX, 0, AX);
    match kind {
        UpdKind::PreInc | UpdKind::PreDec => {
            a.nr_store(SP, 0, AX);
            a.add_sp(8);
        }
        UpdKind::PostInc | UpdKind::PostDec => {
            a.nr_store(SP, 0, R8);
            a.add_sp(8);
        }
        UpdKind::IncDiscard | UpdKind::DecDiscard => {}
    }
}

/// Captured same-record target, even after intervening RHS code changes the scope chain.
fn reference_target(a: &mut Asm, l: &JitLayout, slot: u16, slow: usize) {
    let object = a.label();
    let done = a.label();
    a.nr_load(R8, CTX, offset_of!(super::super::JitCtx, references_raw));
    a.nr_zero_guard(R8, slow);
    a.nr_lea(
        R8,
        R8,
        (slot as usize * size_of::<PreparedReferenceSlot>()) as i32,
    );
    a.nr_load(AX, R8, offset_of!(PreparedReferenceSlot, present));
    a.nr_op_imm(7, AX, 1);
    a.jcc(NE, slow);
    a.nr_lea(R8, R8, offset_of!(PreparedReferenceSlot, reference) as i32);
    a.nr_load32(R11, R8, offset_of!(PreparedReference, kind));
    a.nr_test(R11);
    a.jcc(NE, object);
    a.nr_load(R9, R8, offset_of!(PreparedReference, scope));
    a.nr_zero_guard(R9, slow);
    a.nr_load32(AX, R8, offset_of!(PreparedReference, generation));
    a.nr_max_guard(AX, DX, slow);
    a.nr_load32(DX, R9, l.scope_gen);
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
    a.nr_load(CX, R8, offset_of!(PreparedReference, binding));
    a.nr_zero_guard(CX, slow);
    a.jmp(done);
    a.bind(object);
    a.nr_op_imm(7, R11, 1);
    a.jcc(NE, slow);
    a.nr_load(R9, R8, offset_of!(PreparedReference, object));
    a.nr_zero_guard(R9, slow);
    a.nr_lea(R9, R9, l.obj_from_cell as i32);
    a.nr_load32(AX, R8, offset_of!(PreparedReference, shape));
    a.nr_load(DX, R8, offset_of!(PreparedReference, slot));
    global_target(a, l, true, slow);
    a.bind(done);
}

fn capture(
    a: &mut Asm,
    l: &JitLayout,
    il: &crate::interpreter::InterpLayout,
    chunk: &Chunk,
    name: u32,
    slot: u16,
    slow: usize,
) {
    a.nr_imm(R8, chunk.jit_resolution_cache_ptr(name) as u64);
    a.nr_load(R8, R8, 0);
    a.nr_load(R9, CTX, offset_of!(super::super::JitCtx, env_raw));
    deep_target(a, l, slow);
    a.nr_load(R10, CTX, offset_of!(super::super::JitCtx, references_raw));
    a.nr_zero_guard(R10, slow);
    a.nr_lea(
        R10,
        R10,
        (slot as usize * size_of::<PreparedReferenceSlot>()) as i32,
    );
    a.nr_load(AX, R10, offset_of!(PreparedReferenceSlot, present));
    a.nr_op_imm(7, AX, 1);
    a.jcc(NE, slow);
    a.nr_lea(
        R10,
        R10,
        offset_of!(PreparedReferenceSlot, reference) as i32,
    );
    a.nr_load32(AX, R10, offset_of!(PreparedReference, kind));
    a.nr_cmp(AX, R11);
    a.jcc(NE, slow);
    a.nr_imm(AX, chunk.jit_shared_name_ptr(name) as u64);
    for off in [l.str_ptr_word, l.str_len_word] {
        a.nr_load(DX, AX, off);
        a.nr_mem(
            true,
            &[0x3b],
            DX,
            R10,
            (offset_of!(PreparedReference, name) + off) as i32,
        );
        a.jcc(NE, slow);
    }
    let object = a.label();
    let commit = a.label();
    let hints_done = a.label();
    a.nr_test(R11);
    a.jcc(NE, object);
    a.nr_mov(AX, R9);
    a.nr_load(DX, R10, offset_of!(PreparedReference, scope));
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
    a.jmp(commit);
    a.bind(object);
    a.nr_lea(AX, R9, -(l.obj_from_cell as i32));
    a.nr_load(DX, R10, offset_of!(PreparedReference, object));
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
    a.bind(commit);
    // Same owners and key already installed. Refresh only hints and captured strictness.
    a.nr_load(AX, CTX, offset_of!(super::super::JitCtx, interp));
    a.nr_load8(DX, AX, il.strict);
    a.nr_store32(R10, offset_of!(PreparedReference, strict), DX);
    let object_hint = a.label();
    a.nr_test(R11);
    a.jcc(NE, object_hint);
    a.nr_load32(AX, R9, l.scope_gen);
    a.nr_store32(R10, offset_of!(PreparedReference, generation), AX);
    a.nr_store(R10, offset_of!(PreparedReference, binding), CX);
    a.jmp(hints_done);
    a.bind(object_hint);
    a.nr_load32(AX, R8, offset_of!(NativeNameDescriptor, shape));
    a.nr_store32(R10, offset_of!(PreparedReference, shape), AX);
    a.nr_load(AX, R8, offset_of!(NativeNameDescriptor, slot));
    a.nr_store(R10, offset_of!(PreparedReference, slot), AX);
    a.bind(hints_done);
}

/// A fixed activation's captured parameters and `this` (`Chunk::jit_prologue_copies`), which its
/// frame setup helper left at their template words: generated stores copy them from the frame's
/// parameter slots (r15) and record, then, after the helper (`mark`), mark the record's card
/// once (the helper's closing poll may have aged it). W3 item 3's step 5 (Decision 20c), as
/// ARM64's prologue.
pub(super) fn emit_activation_copies(a: &mut Asm, chunk: &Chunk, mark: bool) {
    let Some((parameters, this_slot)) = chunk.jit_prologue_copies() else {
        return;
    };
    let offset = |slot: u32| crate::interpreter::RECORD_HEADER + 8 * slot as usize;
    a.nr_load(R9, CTX, offset_of!(super::super::JitCtx, env_raw));
    for (slot, parameter) in parameters {
        a.nr_load(AX, 15, usize::from(parameter) * 8);
        a.nr_store(R9, offset(slot), AX);
    }
    if let Some(slot) = this_slot {
        a.nr_load(AX, CTX, offset_of!(super::super::JitCtx, this_word));
        a.nr_store(R9, offset(slot), AX);
    }
    // An in-line record is young until the poll that follows (W3 item 3's step 7).
    if mark {
        card_mark(a);
    }
}

/// Count an in-line activation in a test build (rax scratch, before the poll).
#[cfg(test)]
pub(super) fn record_inline_activation(a: &mut Asm) {
    record(
        a,
        super::super::TEST_INLINE_ACTIVATIONS.with(|count| count.as_ptr() as usize),
    );
}

/// LoadCap, StoreCap, StoreCapInit and UpdateCap at fixed slots of the running activation
/// (`Chunk::jit_cap_binding`, W3 item 3's step 5): a load tests the word for TDZ (the empty
/// word) unless the binding is never in TDZ or its basic block checked it already
/// (`Chunk::jit_cap_initialized`); a store marks the record's card. `false`: not such an
/// operation here.
fn emit_fixed_cap(a: &mut Asm, chunk: &Chunk, op: &Op, pc: u32, unwind: usize) -> bool {
    let name = match *op {
        Op::LoadCap(n) | Op::StoreCap(n) | Op::StoreCapInit(n) | Op::UpdateCap(n, _) => n,
        _ => return false,
    };
    // A borrowed frame's activation may be another tier's, without the plan's fixed slots.
    let Some((slot, attrs)) = a
        .fixed_activation
        .then(|| chunk.jit_cap_binding(name))
        .flatten()
    else {
        return false;
    };
    if matches!(op, Op::StoreCap(_) | Op::UpdateCap(..)) && !attrs.mutable() {
        return false;
    }
    let offset = crate::interpreter::RECORD_HEADER + 8 * slot as usize;
    let tdz = attrs.lexical() && !chunk.jit_cap_initialized(pc as usize);
    let slow = a.label();
    let done = a.label();
    a.nr_load(R9, CTX, offset_of!(super::super::JitCtx, env_raw));
    match *op {
        Op::LoadCap(_) => {
            a.nr_load(AX, R9, offset);
            if tdz {
                execution_word_guard(a, slow);
            }
            a.nr_store(SP, 0, AX);
            a.add_sp(8);
        }
        Op::StoreCap(_) | Op::StoreCapInit(_) => {
            if tdz && matches!(op, Op::StoreCap(_)) {
                a.nr_load(AX, R9, offset);
                a.nr_imm(DX, crate::value::PACK_EMPTY);
                a.nr_cmp(AX, DX);
                a.jcc(EQ, slow);
            }
            a.nr_mem(true, &[0x8b], AX, SP, -8);
            execution_word_guard(a, slow);
            a.nr_store(R9, offset, AX);
            card_mark(a);
            a.add_sp(-8);
        }
        Op::UpdateCap(_, kind) => {
            // The empty word (TDZ) and every non-Number are NaN words: the checked path.
            a.nr_lea(CX, R9, offset as i32);
            update_value(a, kind, slow);
        }
        _ => unreachable!(),
    }
    #[cfg(test)]
    {
        // As the cached name templates count their commits.
        record(
            a,
            TEST_NATIVE_NAME_OPS.with(|count| count.as_ptr() as usize),
        );
        let index = match op {
            Op::LoadCap(_) => 0,
            Op::UpdateCap(..) => 2,
            _ => 1,
        };
        record(
            a,
            TEST_NATIVE_NAME_COMMITS.with(|counts| counts[index].as_ptr() as usize),
        );
    }
    a.jmp(done);
    a.bind(slow);
    a.helper_spflag(H_EXEC, pc, unwind);
    a.bind(done);
    true
}

pub(super) fn emit_op(
    a: &mut Asm,
    chunk: &Chunk,
    l: &JitLayout,
    il: &crate::interpreter::InterpLayout,
    op: &Op,
    pc: u32,
    unwind: usize,
) -> bool {
    if emit_fixed_cap(a, chunk, op, pc, unwind) {
        return true;
    }
    let reference = matches!(
        op,
        Op::ResolveNameRef(..) | Op::LoadRef(_) | Op::StoreRef(_)
    );
    let (cache, for_call, store) = match *op {
        Op::LoadName(_, cache) | Op::LoadNameForCall(_, cache) => (
            Some(chunk.jit_name_cache_ptr(cache)),
            matches!(op, Op::LoadNameForCall(..)),
            false,
        ),
        Op::StoreNameCached(_, cache) => (Some(chunk.jit_name_cache_ptr(cache)), false, true),
        Op::UpdateNameCached(_, cache, _) => (Some(chunk.jit_name_cache_ptr(cache)), false, true),
        Op::LoadCap(name) if !chunk.jit_needs_activation_state() => {
            (Some(chunk.jit_cap_cache_ptr(name)), false, false)
        }
        Op::StoreCap(name) if !chunk.jit_needs_activation_state() => {
            (Some(chunk.jit_cap_cache_ptr(name)), false, true)
        }
        _ if reference => (None, false, matches!(op, Op::StoreRef(_))),
        _ => return false,
    };
    let slow = a.label();
    let done = a.label();
    if valid(l) && il.valid && il.strict <= i32::MAX as usize {
        match *op {
            Op::ResolveNameRef(name, slot) => capture(a, l, il, chunk, name, slot, slow),
            _ => {
                if let Some(cache) = cache {
                    name_target(a, l, cache, slow);
                } else {
                    let slot = match *op {
                        Op::LoadRef(slot) | Op::StoreRef(slot) => slot,
                        _ => unreachable!(),
                    };
                    reference_target(a, l, slot, slow);
                }
                value_address(a, l, store, slow);
                if let Op::UpdateNameCached(_, _, kind) = *op {
                    update_value(a, kind, slow);
                } else if store {
                    store_value(a, slow);
                } else {
                    load_value(a, for_call, slow);
                }
            }
        }
        #[cfg(test)]
        {
            let address = if reference {
                let index = match op {
                    Op::ResolveNameRef(..) => 0,
                    Op::LoadRef(_) => 1,
                    _ => 2,
                };
                TEST_NATIVE_REFERENCES.with(|counts| counts[index].as_ptr() as usize)
            } else {
                TEST_NATIVE_NAME_OPS.with(|count| count.as_ptr() as usize)
            };
            record(a, address);
            if !matches!(op, Op::ResolveNameRef(..)) {
                let index = if matches!(op, Op::UpdateNameCached(..)) {
                    2
                } else if store {
                    1
                } else {
                    0
                };
                record(
                    a,
                    TEST_NATIVE_NAME_COMMITS.with(|counts| counts[index].as_ptr() as usize),
                );
            }
        }
        a.jmp(done);
    }
    a.bind(slow);
    a.helper_spflag(
        if reference {
            super::super::H_REFERENCE_OP
        } else {
            H_EXEC
        },
        pc,
        unwind,
    );
    a.bind(done);
    true
}
