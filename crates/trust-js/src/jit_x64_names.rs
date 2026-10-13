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
use crate::value::{JitLayout, PACK_NULL, PACK_STR, PACK_SYM};
use std::mem::{offset_of, size_of};

const AX: u8 = 0;
const CX: u8 = 1;
const DX: u8 = 2;
const R8: u8 = 8;
const R9: u8 = 9;
const R10: u8 = 10;
const R11: u8 = 11;
const CTX: u8 = 12;
const SP: u8 = 13;
const EQ: u8 = 0x84;
const NE: u8 = 0x85;
const BELOW: u8 = 0x82;
const ABOVE: u8 = 0x87;
const AE: u8 = 0x83;
const BE: u8 = 0x86;

#[cfg(test)]
thread_local! {
    pub(super) static TEST_NATIVE_DEEP_NAMES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static TEST_NATIVE_NAME_OPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static TEST_NATIVE_REFERENCES: [std::cell::Cell<usize>; 3] = const {
        [std::cell::Cell::new(0), std::cell::Cell::new(0), std::cell::Cell::new(0)]
    };
}

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "jit_x64_names_tests.rs"]
mod tests;

/// Thread-affine test instrumentation only; production code contains no counter operations.
/// AX/DX are volatile after an operation commits; no owner or stack pointer is changed.
#[cfg(test)]
fn record(a: &mut Asm, address: usize) {
    a.nr_imm(AX, address as u64);
    a.nr_mem(true, &[0xff], 0, AX, 0);
}

// Only ABI-volatile registers are scratch (also on Win64). No push/pop changes
// the established helper-call alignment. Always use a full disp32 addressing form.
impl Asm {
    fn nr_rex(&mut self, wide: bool, reg: u8, base: u8) {
        self.code
            .push(0x40 | (u8::from(wide) << 3) | ((reg >> 3) << 2) | (base >> 3));
    }
    fn nr_mem(&mut self, wide: bool, opcode: &[u8], reg: u8, base: u8, off: i32) {
        self.nr_rex(wide, reg, base);
        self.bytes(opcode);
        self.code.push(0x80 | ((reg & 7) << 3) | (base & 7));
        if base & 7 == 4 {
            self.code.push(0x24);
        }
        self.bytes(&off.to_le_bytes());
    }
    fn nr_load(&mut self, reg: u8, base: u8, off: usize) {
        self.nr_mem(true, &[0x8b], reg, base, off as i32);
    }
    fn nr_load32(&mut self, reg: u8, base: u8, off: usize) {
        self.nr_mem(false, &[0x8b], reg, base, off as i32);
    }
    fn nr_load8(&mut self, reg: u8, base: u8, off: usize) {
        self.nr_mem(false, &[0x0f, 0xb6], reg, base, off as i32);
    }
    fn nr_store(&mut self, base: u8, off: usize, reg: u8) {
        self.nr_mem(true, &[0x89], reg, base, off as i32);
    }
    fn nr_store32(&mut self, base: u8, off: usize, reg: u8) {
        self.nr_mem(false, &[0x89], reg, base, off as i32);
    }
    fn nr_lea(&mut self, reg: u8, base: u8, off: i32) {
        self.nr_mem(true, &[0x8d], reg, base, off);
    }
    fn nr_reg(&mut self, opcode: u8, dst: u8, src: u8) {
        self.nr_rex(true, src, dst);
        self.bytes(&[opcode, 0xc0 | ((src & 7) << 3) | (dst & 7)]);
    }
    fn nr_mov(&mut self, dst: u8, src: u8) {
        self.nr_reg(0x89, dst, src);
    }
    fn nr_cmp(&mut self, lhs: u8, rhs: u8) {
        self.nr_reg(0x39, lhs, rhs);
    }
    fn nr_test(&mut self, reg: u8) {
        self.nr_reg(0x85, reg, reg);
    }
    fn nr_imm(&mut self, reg: u8, value: u64) {
        self.nr_rex(true, 0, reg);
        self.code.push(0xb8 | (reg & 7));
        self.bytes(&value.to_le_bytes());
    }
    fn nr_op_imm(&mut self, group: u8, reg: u8, value: i32) {
        self.nr_rex(true, 0, reg);
        self.bytes(&[0x81, 0xc0 | (group << 3) | (reg & 7)]);
        self.bytes(&value.to_le_bytes());
    }
    fn nr_shift(&mut self, right: bool, reg: u8, count: u8) {
        self.nr_rex(true, 0, reg);
        self.bytes(&[
            0xc1,
            0xc0 | ((if right { 5 } else { 4 }) << 3) | (reg & 7),
            count,
        ]);
    }
    fn nr_mul_imm(&mut self, reg: u8, stride: usize) {
        self.nr_rex(true, reg, reg);
        self.bytes(&[0x69, 0xc0 | ((reg & 7) << 3) | (reg & 7)]);
        self.bytes(&(stride as i32).to_le_bytes());
    }
    fn nr_zero_guard(&mut self, reg: u8, slow: usize) {
        self.nr_test(reg);
        self.jcc(EQ, slow);
    }
    fn nr_max_guard(&mut self, reg: u8, scratch: u8, slow: usize) {
        self.nr_imm(scratch, u32::MAX as u64);
        self.nr_cmp(reg, scratch);
        self.jcc(EQ, slow);
    }
}

fn valid(l: &JitLayout) -> bool {
    l.valid
        && l.scope_parent_valid
        && l.scope_with_valid
        && l.key_probe_ok
        && l.entry_accessor == l.entry_value + 8
        && [
            l.scope_gen,
            l.scope_layout,
            l.scope_parent,
            l.scope_with,
            l.scope_small_tag,
            l.scope_small_vec,
            l.scope_binding_stride,
            l.scope_binding_offset,
            l.binding_value,
            l.binding_init,
            l.binding_import,
            l.binding_mutable,
            l.obj_props + l.props_entries + l.vec_len_off,
            l.obj_props + l.props_entries + l.vec_ptr_off,
            l.obj_props + l.props_shape,
            l.obj_ic_plain,
            l.obj_exotic,
            l.entry_size,
            l.entry_value,
            l.entry_accessor,
            l.entry_writable,
            l.rc_strong_off,
            l.scope_data_off,
            l.obj_from_rc,
            l.str_ptr_word,
            l.str_len_word,
        ]
        .iter()
        .all(|off| *off <= i32::MAX as usize)
}

fn scope_plain(a: &mut Asm, l: &JitLayout, slow: usize) {
    a.nr_load8(AX, R9, l.scope_with);
    a.nr_op_imm(7, AX, l.scope_with_none as i32);
    a.jcc(NE, slow);
}

/// R9=body, AX=shape, DX=slot. Output CX=Property, R11=1. No value flags yet.
fn global_target(a: &mut Asm, l: &JitLayout, slow: usize) {
    a.nr_zero_guard(R9, slow);
    a.nr_zero_guard(AX, slow);
    a.nr_max_guard(AX, CX, slow);
    a.nr_load32(CX, R9, l.obj_props + l.props_shape);
    a.nr_cmp(CX, AX);
    a.jcc(NE, slow);
    a.nr_load8(CX, R9, l.obj_exotic);
    a.nr_op_imm(7, CX, l.exotic_none_tag as i32);
    a.jcc(NE, slow);
    a.nr_load8(CX, R9, l.obj_ic_plain);
    a.nr_zero_guard(CX, slow);
    a.nr_load(CX, R9, l.obj_props + l.props_entries + l.vec_len_off);
    a.nr_cmp(DX, CX);
    a.jcc(AE, slow);
    a.nr_load(CX, R9, l.obj_props + l.props_entries + l.vec_ptr_off);
    a.nr_mul_imm(DX, l.entry_size);
    a.nr_reg(0x01, CX, DX);
    a.nr_imm(R11, 1);
}

/// R8=live descriptor, R9=current Rc::as_ptr(Scope). Output CX=Binding/Property,
/// R11=0/1, R9=holder Scope pointer/Object body. Preserve descriptor through commit.
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
    a.nr_load32(AX, R10, NAME_GUARD_LAYOUT);
    a.nr_zero_guard(AX, slow);
    a.nr_load32(DX, R9, l.scope_layout);
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
    a.bind(checked);
    a.nr_op_imm(5, R11, 1);
    a.jcc(EQ, holder);
    a.nr_load(R9, R9, l.scope_parent);
    a.nr_zero_guard(R9, slow);
    a.nr_lea(R9, R9, l.scope_data_off as i32);
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
    if l.scope_small_valid {
        a.nr_load8(AX, R9, l.scope_small_tag);
        a.nr_imm(DX, 2);
        a.nr_reg(0x09, AX, DX); // tag | 2 == 2 accepts Small(0) and Indexed(2)
        a.nr_cmp(AX, DX);
        a.jcc(NE, slow);
        a.nr_load(DX, R8, offset_of!(NativeNameDescriptor, slot));
        a.nr_load(CX, R9, l.scope_small_vec + l.vec_len_off);
        a.nr_cmp(DX, CX);
        a.jcc(AE, slow);
        a.nr_load(CX, R9, l.scope_small_vec + l.vec_ptr_off);
        a.nr_mul_imm(DX, l.scope_binding_stride);
        a.nr_reg(0x01, CX, DX);
        a.nr_lea(CX, CX, l.scope_binding_offset as i32);
    } else {
        a.jmp(slow);
    }
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
    global_target(a, l, slow);
    a.bind(done);
    #[cfg(test)]
    record(
        a,
        TEST_NATIVE_DEEP_NAMES.with(|count| count.as_ptr() as usize),
    );
}

/// Same result as deep_target, accepting the existing own/depth-one/global compact IC ABI.
fn name_target(a: &mut Asm, l: &JitLayout, cache: usize, slow: usize) {
    use crate::bytecode::{
        NAME_IC_OFF_ACT_GEN, NAME_IC_OFF_BINDING, NAME_IC_OFF_ENV, NAME_IC_OFF_GEN,
    };
    let deep = a.label();
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
    global_target(a, l, slow);
    a.jmp(done);
    a.bind(direct);
    a.nr_load32(AX, R9, l.scope_gen);
    a.nr_cmp(AX, R11);
    a.jcc(NE, slow);
    a.jmp(binding);
    a.bind(depth_one);
    a.nr_load32(DX, R8, NAME_IC_OFF_ACT_GEN as usize);
    a.nr_mov(AX, R10);
    a.nr_op_imm(4, AX, 4);
    a.nr_test(AX);
    a.jcc(EQ, generation);
    a.nr_zero_guard(DX, slow);
    a.nr_load32(AX, R9, l.scope_layout);
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
    a.jmp(parent);
    a.bind(generation);
    a.nr_max_guard(DX, AX, slow);
    a.nr_load32(AX, R9, l.scope_gen);
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
    a.bind(parent);
    a.nr_load(R9, R9, l.scope_parent);
    a.nr_zero_guard(R9, slow);
    a.nr_lea(R9, R9, l.scope_data_off as i32);
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
    a.bind(deep);
    a.nr_load(R8, R8, NAME_IC_OFF_BINDING as usize);
    deep_target(a, l, slow);
    a.bind(done);
}

/// Convert Binding/Property address to its value address, after live semantic flags.
fn value_address(a: &mut Asm, l: &JitLayout, store: bool, slow: usize) {
    let packed = a.label();
    let done = a.label();
    a.nr_test(R11);
    a.jcc(NE, packed);
    a.nr_load8(AX, CX, l.binding_init);
    a.nr_zero_guard(AX, slow);
    a.nr_load8(AX, CX, l.binding_import);
    a.nr_test(AX);
    a.jcc(NE, slow);
    if store {
        a.nr_load8(AX, CX, l.binding_mutable);
        a.nr_zero_guard(AX, slow);
    }
    a.nr_lea(CX, CX, l.binding_value as i32);
    a.jmp(done);
    a.bind(packed);
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

/// Read wide Binding.value to compact RAX, or copy a packed property. No ownership change.
/// CX/R8..R11 preserved. Empty/BigInt/reserved words take the authoritative helper.
fn read_word(a: &mut Asm, slow: usize) {
    let packed = a.label();
    let done = a.label();
    let number = a.label();
    let boolean = a.label();
    let pointer = a.label();
    let null = a.label();
    let undefined = a.label();
    a.nr_test(R11);
    a.jcc(NE, packed);
    a.nr_load8(AX, CX, 0);
    for (tag, label) in [(0, undefined), (2, null), (3, boolean), (4, number)] {
        a.nr_op_imm(7, AX, tag);
        a.jcc(EQ, label);
    }
    a.nr_op_imm(7, AX, 6);
    a.jcc(BELOW, slow);
    // Sym/Object plus String use an explicit tag map; no invalid wide discriminant admitted.
    a.jmp(pointer);
    a.bind(undefined);
    a.nr_imm(AX, PACK_UNDEFINED);
    a.jmp(done);
    a.bind(null);
    a.nr_imm(AX, PACK_NULL);
    a.jmp(done);
    a.bind(boolean);
    // Value::Bool's byte sits at the payload offset; the rest of that word is padding.
    a.nr_load8(DX, CX, 8);
    a.nr_op_imm(4, DX, 1);
    a.nr_imm(AX, PACK_BOOL);
    a.nr_reg(0x09, AX, DX);
    a.jmp(done);
    a.bind(number);
    a.nr_load(AX, CX, 8);
    a.bytes(&[0x66, 0x48, 0x0f, 0x6e, 0xc0, 0x66, 0x0f, 0x2e, 0xc0]); // movq xmm0,rax; ucomisd xmm0,xmm0
    a.jcc(0x8b, done);
    a.nr_imm(AX, 0x7ff8_0000_0000_0000);
    a.jmp(done);
    a.bind(pointer);
    let string = a.label();
    let symbol = a.label();
    let object = a.label();
    for (tag, label) in [(6, string), (7, symbol), (8, object)] {
        a.nr_op_imm(7, AX, tag);
        a.jcc(EQ, label);
    }
    a.jmp(slow);
    for (label, tag) in [(string, PACK_STR), (symbol, PACK_SYM), (object, PACK_OBJ)] {
        a.bind(label);
        a.nr_load(AX, CX, 8);
        a.nr_imm(DX, tag);
        a.nr_reg(0x09, AX, DX);
        a.jmp(done);
    }
    a.bind(packed);
    a.nr_load(AX, CX, 0);
    a.bind(done);
}

/// Keep RAX intact; DX becomes zero for immediate values or the shared owner pointer.
fn owner_pointer(a: &mut Asm, slow: usize) {
    let owner = a.label();
    let scalar = a.label();
    let done = a.label();
    a.nr_mov(DX, AX);
    a.nr_shift(true, DX, 48);
    for tag in [0x7ffa, 0x7ffd] {
        a.nr_op_imm(7, DX, tag);
        a.jcc(EQ, slow);
    }
    for tag in [0x7ffe, 0x7fff, 0xfff9] {
        a.nr_op_imm(7, DX, tag);
        a.jcc(EQ, owner);
    }
    a.jcc(ABOVE, slow);
    a.jmp(scalar);
    a.bind(owner);
    a.nr_mov(DX, AX);
    a.nr_shift(false, DX, 16);
    a.nr_shift(true, DX, 16);
    a.jmp(done);
    a.bind(scalar);
    a.nr_imm(DX, 0);
    a.bind(done);
}

fn load_value(a: &mut Asm, l: &JitLayout, for_call: bool, slow: usize) {
    read_word(a, slow);
    owner_pointer(a, slow);
    let copied = a.label();
    a.nr_test(DX);
    a.jcc(EQ, copied);
    a.nr_mem(true, &[0xff], 0, DX, l.rc_strong_off as i32);
    a.bind(copied);
    if for_call {
        a.nr_imm(DX, PACK_UNDEFINED);
        a.nr_store(SP, 0, DX);
        a.add_sp(8);
    }
    a.nr_store(SP, 0, AX);
    a.add_sp(8);
}

/// Decode compact RHS without cloning. R8=wide tag word, R9=payload; R10 stays raw.
fn decode_rhs(a: &mut Asm, slow: usize) {
    let number = a.label();
    let tagged = a.label();
    let done = a.label();
    a.nr_mov(DX, R10);
    a.nr_shift(true, DX, 48);
    a.nr_op_imm(7, DX, 0x7ff9);
    a.jcc(BELOW, number);
    a.nr_op_imm(7, DX, 0x7fff);
    a.jcc(BE, tagged);
    a.nr_op_imm(7, DX, 0xfff9);
    a.jcc(BELOW, number);
    a.bind(tagged);
    let undefined = a.label();
    let null = a.label();
    let boolean = a.label();
    let string = a.label();
    let symbol = a.label();
    let object = a.label();
    for (tag, label) in [
        (0x7ff9, undefined),
        (0x7ffb, null),
        (0x7ffc, boolean),
        (0x7ffe, string),
        (0x7fff, symbol),
        (0xfff9, object),
    ] {
        a.nr_op_imm(7, DX, tag);
        a.jcc(EQ, label);
    }
    a.jmp(slow);
    for (label, tag) in [(undefined, 0), (null, 2)] {
        a.bind(label);
        a.nr_imm(R8, tag);
        a.nr_imm(R9, 0);
        a.jmp(done);
    }
    a.bind(boolean);
    a.nr_imm(R8, 3);
    a.nr_mov(R9, R10);
    a.nr_op_imm(4, R9, 1);
    a.jmp(done);
    for (label, tag) in [(string, 6), (symbol, 7), (object, 8)] {
        a.bind(label);
        a.nr_imm(R8, tag);
        a.nr_mov(R9, R10);
        a.nr_shift(false, R9, 16);
        a.nr_shift(true, R9, 16);
        a.jmp(done);
    }
    a.bind(number);
    a.nr_imm(R8, 4);
    a.nr_mov(R9, R10);
    a.bind(done);
}

/// All semantic and ownership guards precede the sole refcount decrement/store commit.
fn store_value(a: &mut Asm, l: &JitLayout, slow: usize) {
    a.nr_mem(true, &[0x8b], R10, SP, -8);
    decode_rhs(a, slow);
    read_word(a, slow);
    owner_pointer(a, slow);
    let commit = a.label();
    a.nr_test(DX);
    a.jcc(EQ, commit);
    a.nr_load(AX, DX, l.rc_strong_off);
    a.nr_op_imm(7, AX, 1);
    a.jcc(BE, slow);
    // No further guard/helper after this mutation. R8/R9/R10 hold the transferred new owner.
    a.nr_mem(true, &[0xff], 1, DX, l.rc_strong_off as i32);
    a.bind(commit);
    let packed = a.label();
    let done = a.label();
    a.nr_test(R11);
    a.jcc(NE, packed);
    a.nr_store(CX, 0, R8);
    a.nr_store(CX, 8, R9);
    a.jmp(done);
    a.bind(packed);
    a.nr_store(CX, 0, R10);
    a.bind(done);
    a.add_sp(-8);
}

/// Numeric ++/--, after live mutability/writability guards. Non-Numbers and NaNs
/// stay on the checked ToNumeric boundary. The only mutation follows every guard.
fn update_value(a: &mut Asm, kind: UpdKind, slow: usize) {
    let packed = a.label();
    let number = a.label();
    a.nr_test(R11);
    a.jcc(NE, packed);
    a.nr_load8(AX, CX, 0);
    a.nr_op_imm(7, AX, 4);
    a.jcc(NE, slow);
    a.nr_lea(CX, CX, 8);
    a.jmp(number);
    a.bind(packed);
    a.nr_load(AX, CX, 0);
    a.nr_shift(true, AX, 48);
    a.nr_op_imm(7, AX, 0x7ff9);
    a.jcc(BELOW, number);
    a.nr_op_imm(7, AX, 0x7fff);
    a.jcc(BE, slow);
    a.nr_op_imm(7, AX, 0xfff9);
    a.jcc(AE, slow);
    a.bind(number);
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
    a.nr_lea(R9, R9, l.scope_data_off as i32);
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
    a.nr_lea(R9, R9, l.obj_from_rc as i32);
    a.nr_load32(AX, R8, offset_of!(PreparedReference, shape));
    a.nr_load(DX, R8, offset_of!(PreparedReference, slot));
    global_target(a, l, slow);
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
    a.nr_lea(AX, R9, -(l.scope_data_off as i32));
    a.nr_load(DX, R10, offset_of!(PreparedReference, scope));
    a.nr_cmp(AX, DX);
    a.jcc(NE, slow);
    a.jmp(commit);
    a.bind(object);
    a.nr_lea(AX, R9, -(l.obj_from_rc as i32));
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

pub(super) fn emit_op(
    a: &mut Asm,
    chunk: &Chunk,
    l: &JitLayout,
    il: &crate::interpreter::InterpLayout,
    op: &Op,
    pc: u32,
    unwind: usize,
) -> bool {
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
                    store_value(a, l, slow);
                } else {
                    load_value(a, l, for_call, slow);
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
