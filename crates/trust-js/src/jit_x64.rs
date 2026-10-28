//! Baseline x86-64 template backend (Intel macOS, x64 Linux, and x64 Windows).
//!
//! This deliberately starts below the mature ARM64 backend: native branches remove bytecode
//! dispatch, while individual operations enter the existing checked helpers. Hot inline templates
//! can then be ported without duplicating runtime semantics or compromising deoptimization.

use super::{
    JitCode, COND_PEEK_TRUTHY, COND_POP_TRUTHY, H_CALL, H_CALL_SPREAD, H_COMPLETE, H_COND, H_EXEC,
    H_GET_ELEM, H_GET_METHOD_ELEM, H_GET_PROP, H_INTERRUPT, H_NEW, H_POP_HANDLER, H_PUSH_HANDLER,
    H_SET_PROP, H_SLICE_OP, H_UNWIND,
};
use crate::bytecode::{Chunk, Op};
use crate::value::{PACK_BOOL, PACK_OBJ, PACK_UNDEFINED};

#[path = "jit_x64_names.rs"]
mod names;
#[path = "jit_x64_storage.rs"]
mod storage;

// JavaScript slots are owned Values; C ABI stack alignment is independent.
const SLOT_BYTES: i32 = 8;

// Register numbers and condition codes of the register-generic forms (`Asm::nr_*`).
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

#[derive(Clone, Copy)]
enum PropRecv {
    This,
    Slot(u16),
}

struct Asm {
    code: Vec<u8>,
    labels: Vec<Option<usize>>,
    patches: Vec<(usize, usize)>,
}

impl Asm {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            labels: Vec::new(),
            patches: Vec::new(),
        }
    }
    fn label(&mut self) -> usize {
        self.labels.push(None);
        self.labels.len() - 1
    }
    fn bind(&mut self, label: usize) {
        self.labels[label] = Some(self.code.len());
    }
    fn bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }
    fn rel32(&mut self, label: usize) {
        self.patches.push((self.code.len(), label));
        self.code.extend_from_slice(&[0; 4]);
    }
    fn jmp(&mut self, label: usize) {
        self.code.push(0xe9);
        self.rel32(label);
    }
    fn jcc(&mut self, cc: u8, label: usize) {
        self.bytes(&[0x0f, cc]);
        self.rel32(label);
    }
    #[cfg(not(target_os = "windows"))]
    fn call_helper_pair(&mut self, helper: usize, imm: u32) {
        // rdi=ctx(r12), esi=imm, rdx=sp(r13); call [helpers(r14)+index*8].
        self.bytes(&[0x4c, 0x89, 0xe7, 0xbe]);
        self.bytes(&imm.to_le_bytes());
        self.bytes(&[0x4c, 0x89, 0xea, 0x41, 0xff, 0x96]);
        self.bytes(&((helper * 8) as i32).to_le_bytes());
    }
    #[cfg(not(target_os = "windows"))]
    fn call_helper_ptr(&mut self, helper: usize, imm: u32) {
        self.call_helper_pair(helper, imm);
    }
    #[cfg(target_os = "windows")]
    fn call_helper_pair(&mut self, helper: usize, imm: u32) {
        // Win64 returns the 16-byte SpFlag through a hidden pointer. Reserve the mandatory
        // 32-byte shadow area plus the result, preserving 16-byte alignment around the call.
        self.bytes(&[0x48, 0x83, 0xec, 0x30]); // sub rsp,48
        self.bytes(&[0x48, 0x8d, 0x4c, 0x24, 0x20]); // rcx=&result
        self.bytes(&[0x4c, 0x89, 0xe2, 0x41, 0xb8]); // rdx=ctx; r8d=imm
        self.bytes(&imm.to_le_bytes());
        self.bytes(&[0x4d, 0x89, 0xe9, 0x41, 0xff, 0x96]); // r9=sp; call helper
        self.bytes(&((helper * 8) as i32).to_le_bytes());
        self.bytes(&[
            0x48, 0x8b, 0x44, 0x24, 0x20, // rax=result.sp
            0x48, 0x8b, 0x54, 0x24, 0x28, // rdx=result.flag
            0x48, 0x83, 0xc4, 0x30, // add rsp,48
        ]);
    }
    #[cfg(target_os = "windows")]
    fn call_helper_ptr(&mut self, helper: usize, imm: u32) {
        self.bytes(&[0x48, 0x83, 0xec, 0x20]); // Win64 shadow space
        self.bytes(&[0x4c, 0x89, 0xe1, 0xba]); // rcx=ctx; edx=imm
        self.bytes(&imm.to_le_bytes());
        self.bytes(&[0x4d, 0x89, 0xe8, 0x41, 0xff, 0x96]); // r8=sp; call helper
        self.bytes(&((helper * 8) as i32).to_le_bytes());
        self.bytes(&[0x48, 0x83, 0xc4, 0x20]);
    }
    fn cmp_tag_r13(&mut self, disp: i32, tag: u16) {
        self.bytes(&[0x66, 0x41, 0x81, 0xbd]); // cmp word [r13+disp+6],tag
        self.bytes(&(disp + 6).to_le_bytes());
        self.bytes(&tag.to_le_bytes());
    }
    /// The completion router has already handled finalizers and iterator pads. Publish the
    /// new word even when an earlier result occupies the destination; GC owns displaced cells.
    fn return_value(&mut self, mode: u32, ret_ok: usize) {
        let ret = std::mem::offset_of!(super::JitCtx, ret) as i32;
        if mode == 1 {
            self.add_sp(-SLOT_BYTES as i8);
            self.load_word_r13(0);
        } else {
            self.mov_word_imm(PACK_UNDEFINED);
        }
        self.bytes(&[0x49, 0x89, 0x84, 0x24]); // mov [r12+ret],rax
        self.bytes(&ret.to_le_bytes());
        self.jmp(ret_ok);
    }
    fn cmp_qword_r13_rax(&mut self, disp: i32) {
        self.bytes(&[0x49, 0x39, 0x85]);
        self.bytes(&disp.to_le_bytes());
    }
    fn load_tag_r13(&mut self, disp: i32) {
        self.bytes(&[0x41, 0x0f, 0xb7, 0x85]); // movzx eax,word [r13+disp+6]
        self.bytes(&(disp + 6).to_le_bytes());
    }
    fn load_tag_r15(&mut self, disp: i32) {
        self.bytes(&[0x41, 0x0f, 0xb7, 0x87]);
        self.bytes(&(disp + 6).to_le_bytes());
    }
    fn load_word_r13(&mut self, disp: i32) {
        self.bytes(&[0x49, 0x8b, 0x85]);
        self.bytes(&disp.to_le_bytes());
    }
    fn load_word_r15(&mut self, disp: i32) {
        self.bytes(&[0x49, 0x8b, 0x87]);
        self.bytes(&disp.to_le_bytes());
    }
    fn store_word_r13(&mut self, disp: i32) {
        self.bytes(&[0x49, 0x89, 0x85]);
        self.bytes(&disp.to_le_bytes());
    }
    fn store_word_r15(&mut self, disp: i32) {
        self.bytes(&[0x49, 0x89, 0x87]);
        self.bytes(&disp.to_le_bytes());
    }
    fn add_sp(&mut self, amount: i8) {
        self.bytes(&[
            0x49,
            0x83,
            if amount >= 0 { 0xc5 } else { 0xed },
            amount.unsigned_abs(),
        ]);
    }
    fn mov_word_imm(&mut self, word: u64) {
        self.bytes(&[0x48, 0xb8]);
        self.bytes(&word.to_le_bytes());
    }
    fn cmp_eax(&mut self, value: u32) {
        self.code.push(0x3d);
        self.bytes(&value.to_le_bytes());
    }
    fn payload_rax_to_rdx(&mut self) {
        self.bytes(&[
            0x48, 0x89, 0xc2, // rdx=rax
            0x48, 0xc1, 0xe2, 0x10, // shl rdx,16
            0x48, 0xc1, 0xea, 0x10, // shr rdx,16 (unsigned low48)
        ]);
    }
    /// Property-only deferred words must materialize before entering execution storage.
    fn guard_execution_tag(&mut self, slow: usize) {
        self.cmp_eax((crate::value::PACK_LAZY_PROTO >> 48) as u32);
        self.jcc(0x83, slow);
    }
    fn guard_number_r13(&mut self, disp: i32, slow: usize) {
        self.load_tag_r13(disp);
        let number = self.label();
        self.cmp_eax(0x7ff9);
        self.jcc(0x82, number);
        self.cmp_eax(0x7fff);
        self.jcc(0x86, slow);
        self.cmp_eax(0xfff9);
        self.jcc(0x83, slow);
        self.bind(number);
    }
    fn numeric_compare(&mut self, setcc: u8, reject_unordered: bool) {
        // Packed Numbers are raw f64 bits; ucomisd supplies ordered scalar flags.
        self.bytes(&[0xf2, 0x41, 0x0f, 0x10, 0x85]);
        self.bytes(&(-2 * SLOT_BYTES).to_le_bytes());
        self.bytes(&[0xf2, 0x41, 0x0f, 0x10, 0x8d]);
        self.bytes(&(-8i32).to_le_bytes());
        self.bytes(&[0x66, 0x0f, 0x2e, 0xc1, 0x0f, setcc, 0xc0]); // setcc al
        if reject_unordered {
            self.bytes(&[0x0f, 0x9b, 0xc2, 0x20, 0xd0]); // setnp dl; and al,dl
        } else if setcc == 0x95 {
            self.bytes(&[0x0f, 0x9a, 0xc2, 0x08, 0xd0]); // setp dl; or al,dl (NaN !=)
        }
        self.bytes(&[0x0f, 0xb6, 0xd0]); // movzx edx,al
        self.mov_word_imm(PACK_BOOL);
        self.bytes(&[0x48, 0x09, 0xd0]); // or rax,rdx
        self.store_word_r13(-2 * SLOT_BYTES);
        self.add_sp(-8);
    }

    /// Equality for Undefined/Null/Bool pairs, after the numeric path declined.
    /// String/BigInt content and coercion keep their semantic checked paths.
    fn scalar_equality(&mut self, strict: bool, negate: bool, slow: usize) {
        let booleans = self.label();
        let compare = self.label();
        let have = self.label();
        self.load_tag_r13(-2 * SLOT_BYTES);
        self.bytes(&[0x89, 0xc1]); // ecx=lhs tag
        self.load_tag_r13(-SLOT_BYTES);
        self.bytes(&[0x81, 0xf9, 0xfc, 0x7f, 0x00, 0x00]);
        self.jcc(0x84, booleans);
        // Undefined (7ff9) and Null (7ffb) differ by bit1; Empty is NOT nullish.
        self.bytes(&[0x83, 0xc9, 0x02, 0x81, 0xf9, 0xfb, 0x7f, 0x00, 0x00]);
        self.jcc(0x85, slow);
        self.bytes(&[0x83, 0xc8, 0x02]);
        self.cmp_eax(0x7ffb);
        self.jcc(0x85, slow);
        if strict {
            self.jmp(compare);
        } else {
            self.bytes(&[0xb0, 0x01]); // al=true
            self.jmp(have);
        }
        self.bind(booleans);
        self.cmp_eax(0x7ffc);
        self.jcc(0x85, slow);
        self.bind(compare);
        self.load_word_r13(-2 * SLOT_BYTES);
        self.cmp_qword_r13_rax(-SLOT_BYTES);
        self.bytes(&[0x0f, 0x94, 0xc0]); // sete al
        self.bind(have);
        self.bytes(&[0x0f, 0xb6, 0xd0]); // edx=Boolean
        if negate {
            self.bytes(&[0x83, 0xf2, 0x01]);
        }
        self.mov_word_imm(PACK_BOOL);
        self.bytes(&[0x48, 0x09, 0xd0]);
        self.store_word_r13(-2 * SLOT_BYTES);
        self.add_sp(-8);
    }

    /// ToBoolean into edx without changing the operand. String/Symbol/ordinary Object work
    /// for both consuming and peek operations; BigInt and possible HTMLDDA keep checked paths.
    fn truthy_r13(&mut self, layout: &crate::value::JitLayout, slow: usize) {
        let tagged = self.label();
        let boolean = self.label();
        let falsy = self.label();
        let truthy = self.label();
        let string = self.label();
        let object = self.label();
        let done = self.label();
        self.guard_number_r13(-SLOT_BYTES, tagged);
        self.bytes(&[0xf2, 0x41, 0x0f, 0x10, 0x85]);
        self.bytes(&(-SLOT_BYTES).to_le_bytes());
        self.bytes(&[
            0x66, 0x0f, 0xef, 0xc9, // pxor xmm1,xmm1: +0
            0x66, 0x0f, 0x2e, 0xc1, // ucomisd xmm0,xmm1
            0x0f, 0x95, 0xc2, // setne dl
            0x0f, 0x9b, 0xc0, // setnp al
            0x20, 0xc2, // and dl,al: both zero and NaN are false
            0x0f, 0xb6, 0xd2,
        ]);
        self.jmp(done);
        self.bind(tagged);
        for (tag, label) in [(0x7ffc, boolean), (0x7ff9, falsy), (0x7ffb, falsy)] {
            self.cmp_eax(tag);
            self.jcc(0x84, label);
        }
        let object_flag = layout
            .obj_from_cell
            .checked_add(layout.obj_ic_plain)
            .filter(|offset| *offset <= i32::MAX as usize);
        if layout.valid {
            self.cmp_eax(0x7fff);
            self.jcc(0x84, truthy);
            self.cmp_eax(0x7ffe);
            self.jcc(0x84, string);
            if object_flag.is_some() {
                self.cmp_eax(0xfff9);
                self.jcc(0x84, object);
            }
        }
        self.jmp(slow);
        self.bind(boolean);
        self.load_word_r13(-SLOT_BYTES);
        self.bytes(&[0x89, 0xc2, 0x83, 0xe2, 0x01]); // edx=low Boolean bit
        self.jmp(done);
        self.bind(string);
        self.load_word_r13(-SLOT_BYTES);
        self.payload_rax_to_rdx();
        self.bytes(&[0x8b, 0x8a]); // ecx=String length
        self.bytes(&(crate::lstr::LEN_OFF as i32).to_le_bytes());
        self.bytes(&[0x85, 0xc9, 0x0f, 0x95, 0xc2, 0x0f, 0xb6, 0xd2]);
        self.jmp(done);
        self.bind(object);
        self.load_word_r13(-SLOT_BYTES);
        self.payload_rax_to_rdx();
        self.bytes(&[0x80, 0xba]); // cmp byte [rdx+obj.ic_plain],0
        self.bytes(&(object_flag.unwrap_or(0) as i32).to_le_bytes());
        self.bytes(&[0]);
        self.jcc(0x84, slow);
        self.bind(truthy);
        self.bytes(&[0xba, 0x01, 0x00, 0x00, 0x00]);
        self.jmp(done);
        self.bind(falsy);
        self.bytes(&[0x31, 0xd2]);
        self.bind(done);
    }
    fn numeric_bitop(&mut self, opcode: u8, slow: usize) {
        // Accept only exactly representable i32 operands. Fractional/out-of-range/NaN values
        // retain full ToInt32 semantics through the checked helper.
        self.bytes(&[0xf2, 0x41, 0x0f, 0x10, 0x85]);
        self.bytes(&(-2 * SLOT_BYTES).to_le_bytes()); // xmm0=lhs
        self.bytes(&[0xf2, 0x41, 0x0f, 0x10, 0x8d]);
        self.bytes(&(-8i32).to_le_bytes()); // xmm1=rhs
        self.bytes(&[
            0xf2, 0x0f, 0x2c, 0xc0, // cvttsd2si eax,xmm0
            0xf2, 0x0f, 0x2a, 0xd0, // cvtsi2sd xmm2,eax
            0x66, 0x0f, 0x2e, 0xc2, // ucomisd xmm0,xmm2
        ]);
        a_jne_or_unordered(self, slow);
        self.bytes(&[
            0xf2, 0x0f, 0x2c, 0xc9, // cvttsd2si ecx,xmm1
            0xf2, 0x0f, 0x2a, 0xd1, // cvtsi2sd xmm2,ecx
            0x66, 0x0f, 0x2e, 0xca, // ucomisd xmm1,xmm2
        ]);
        a_jne_or_unordered(self, slow);
        self.bytes(&[opcode, 0xc8]); // eax op= ecx
        self.bytes(&[0xf2, 0x0f, 0x2a, 0xc0]); // cvtsi2sd xmm0,eax
        self.bytes(&[0xf2, 0x41, 0x0f, 0x11, 0x85]);
        self.bytes(&(-2 * SLOT_BYTES).to_le_bytes());
        self.add_sp(-8);
    }
    /// Call `helper` with the `(ctx, imm, sp)` contract: r13 takes the returned sp, and a set flag
    /// (a throw) goes to `unwind`. A call of an allocating helper then ends in the explicit poll
    /// (see [`Asm::helper_end_poll`]).
    fn helper_spflag(&mut self, helper: usize, imm: u32, unwind: usize) {
        self.call_helper_pair(helper, imm);
        self.bytes(&[0x49, 0x89, 0xc5]); // r13 = rax (new sp)
        self.bytes(&[0x48, 0x85, 0xd2]); // test rdx, rdx (throw flag)
        self.jcc(0x85, unwind); // jne
        if super::helper_ends_in_poll(helper) {
            self.helper_end_poll(unwind);
        }
    }
    /// The explicit poll that ends a call to an allocating helper (see
    /// `jit::helper_ends_in_poll`), and the calls of `H_CALL` and `H_NEW` and string
    /// accumulation's committed exit: the recorded S5 deviation, in place of an allocation slow
    /// path that collects. It runs once the call has returned, with its results in canonical
    /// places (operand words below r13, slots, context fields): this backend keeps no value in a
    /// caller-saved register across a helper call, only the context, operand top, helper table
    /// and slot base in r12-r15. The poll word, as entry and loop polls test it
    /// ([`Asm::interrupt_poll`]).
    fn helper_end_poll(&mut self, unwind: usize) {
        #[cfg(test)]
        if super::TEST_NO_END_POLLS.with(std::cell::Cell::get) {
            return;
        }
        self.interrupt_poll(super::POLL_HELPER_END, unwind);
    }
    /// Generated code's safepoint (`kind` is `POLL_ENTRY`, `POLL_LOOP` or `POLL_HELPER_END`):
    /// the interpreter's poll word (`interrupt::PollWord`), set while a collection is due or a
    /// host interruption may be pending. Three instructions while it is clear: the context's
    /// pointer, a compare of the word with zero and a branch. The slow path is `H_INTERRUPT`
    /// (`jit_interrupt`: the host interrupt, then `Interp::gc_check`), whose throw unwinds from
    /// here. While collections are suppressed or deferred the heap's bit stays set and every
    /// poll calls the helper until one can collect.
    fn interrupt_poll(&mut self, kind: u32, unwind: usize) {
        let done = self.label();
        self.bytes(&[0x49, 0x8b, 0x8c, 0x24]); // mov rcx,[r12+poll_word]
        self.bytes(&(std::mem::offset_of!(super::JitCtx, poll_word) as i32).to_le_bytes());
        self.bytes(&[0x48, 0x83, 0x39, 0x00]); // cmp qword [rcx],0
        self.jcc(0x84, done); // je: nothing pending
        self.helper_spflag(H_INTERRUPT, kind, unwind);
        self.bind(done);
    }
    fn finish(mut self) -> Vec<u8> {
        for (at, label) in self.patches {
            let target = self.labels[label].expect("unbound x64 JIT label");
            let delta = target as i64 - (at + 4) as i64;
            let delta: i32 = delta.try_into().expect("x64 JIT branch exceeds rel32");
            self.code[at..at + 4].copy_from_slice(&delta.to_le_bytes());
        }
        self.code
    }
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
    /// Branch to `slow` unless the Object at `obj` has ordinary internal methods
    /// ([`crate::value::Exotic::is_ordinary`]), as ARM64's `guard_exotic_ordinary`: its exotic
    /// tag is None's, or, where `jit::exotic_collection_admitted` holds, a Map's or Set's
    /// Collection's, which one unsigned range test from None's tag admits. Clobbers `scratch`
    /// and the flags.
    fn guard_exotic_ordinary(
        &mut self,
        layout: &crate::value::JitLayout,
        obj: u8,
        scratch: u8,
        slow: usize,
    ) {
        self.nr_load8(scratch, obj, layout.obj_exotic);
        if super::exotic_collection_admitted(layout) {
            self.nr_op_imm(5, scratch, i32::from(layout.exotic_none_tag)); // sub
            self.nr_op_imm(7, scratch, 1);
            self.jcc(ABOVE, slow);
        } else {
            self.nr_op_imm(7, scratch, i32::from(layout.exotic_none_tag));
            self.jcc(NE, slow);
        }
    }
}

fn a_jne_or_unordered(a: &mut Asm, slow: usize) {
    a.jcc(0x85, slow);
    a.jcc(0x8a, slow);
}

// ECMA-262 e28783d5, OrdinaryGet: an ordinary own data descriptor returns its live value.
// No getter, prototype walk, conversion or JavaScript call occurs on this path. Test-only counts
// of the reads `emit_prop_own` committed, by receiver form: [slot, `this`].
#[cfg(test)]
thread_local! {
    pub(super) static TEST_OWN_PROPERTY_READS: [std::cell::Cell<usize>; 2] = const {
        [std::cell::Cell::new(0), std::cell::Cell::new(0)]
    };
}

/// r10 = the receiver's cell word, untagged, once its tag is Object's: other receivers (Empty, a
/// primitive) keep the checked helper. Either form reads a packed word as a slot holds one: the
/// local slot's, or the context's [r12+0x30] (`JitCtx::this_word`), the packed `this` itself.
/// That slot or context keeps the receiver alive throughout the read.
fn receiver_cell(a: &mut Asm, recv: PropRecv, slow: usize) {
    match recv {
        PropRecv::This => {
            a.nr_mem(false, &[0x0f, 0xb7], AX, CTX, 0x30 + 6); // eax=ctx.this_word tag
            a.cmp_eax((PACK_OBJ >> 48) as u32);
            a.jcc(0x85, slow);
            a.nr_load(R10, CTX, 0x30); // r10=ctx.this_word
        }
        PropRecv::Slot(slot) => {
            let off = i32::from(slot) * SLOT_BYTES;
            a.load_tag_r15(off);
            a.cmp_eax((PACK_OBJ >> 48) as u32);
            a.jcc(0x85, slow);
            a.bytes(&[0x4d, 0x8b, 0x97]); // r10=[r15+off]
            a.bytes(&off.to_le_bytes());
        }
    }
    a.bytes(&[0x49, 0xc1, 0xe2, 0x10, 0x49, 0xc1, 0xea, 0x10]); // untag r10
}

/// Probe the live property IC, including fills made after this machine body was compiled.
/// Receiver, cache, bounds, descriptor and value-tag checks precede the single push.
fn emit_prop_own(
    a: &mut Asm,
    layout: &crate::value::JitLayout,
    cache: usize,
    recv: PropRecv,
    slow: usize,
) -> bool {
    use crate::bytecode::{IC_OFF_DEPTH, IC_OFF_RECV_SHAPE, IC_OFF_SLOT, PROP_IC_WAYS};
    let offsets = [
        layout.obj_from_cell,
        layout.obj_exotic,
        layout.obj_ic_plain,
        layout.obj_props + layout.props_shape,
        layout.entry_size,
        layout.entry_value,
        layout.entry_accessor,
    ];
    if !layout.valid
        || !storage::fields_len_fits(layout)
        || !storage::fields_data_fits(layout)
        || layout.entry_accessor != layout.entry_value + 8
        || cache == 0
        || offsets.iter().any(|offset| *offset > i32::MAX as usize)
    {
        return false;
    }
    receiver_cell(a, recv, slow);
    a.bytes(&[0x4d, 0x89, 0xd3, 0x49, 0x81, 0xc3]); // r11=Object
    a.bytes(&(layout.obj_from_cell as i32).to_le_bytes());
    a.guard_exotic_ordinary(layout, R11, AX, slow); // ordinary internal methods
    a.bytes(&[0x41, 0x80, 0xbb]);
    a.bytes(&(layout.obj_ic_plain as i32).to_le_bytes());
    a.bytes(&[0]);
    a.jcc(0x84, slow);
    a.bytes(&[0x45, 0x8b, 0x83]); // r8d=live receiver shape
    a.bytes(&((layout.obj_props + layout.props_shape) as i32).to_le_bytes());
    a.bytes(&[0x49, 0xb9]); // r9=first live cache way, pinned by this Chunk
    a.bytes(&(cache as u64).to_le_bytes());
    a.bytes(&[0xb9]); // ecx=ways remaining
    a.bytes(&(PROP_IC_WAYS as u32).to_le_bytes());
    let probe = a.label();
    let next = a.label();
    let found = a.label();
    a.bind(probe);
    a.bytes(&[0x41, 0x80, 0xb9]); // cmp byte [r9+depth],0: own data only
    a.bytes(&(IC_OFF_DEPTH as i32).to_le_bytes());
    a.bytes(&[0]);
    a.jcc(0x85, next);
    a.bytes(&[0x45, 0x3b, 0x81]); // cmp r8d,[r9+recv_shape]
    a.bytes(&(IC_OFF_RECV_SHAPE as i32).to_le_bytes());
    a.jcc(0x85, next);
    a.bytes(&[0x41, 0x8b, 0x91]); // edx=live cached slot (zero extended)
    a.bytes(&(IC_OFF_SLOT as i32).to_le_bytes());
    a.jmp(found);
    a.bind(next);
    a.bytes(&[0x49, 0x81, 0xc1]); // advance to the next Cell<IcState>
    a.bytes(
        &(std::mem::size_of::<std::cell::Cell<crate::bytecode::IcState>>() as i32).to_le_bytes(),
    );
    a.bytes(&[0xff, 0xc9]); // dec ecx
    a.jcc(0x85, probe);
    a.jmp(slow);
    a.bind(found);
    storage::fields_len(a, layout, CX, R11); // rcx=live entry count
    a.nr_cmp(DX, CX);
    a.jcc(AE, slow);
    storage::fields_data(a, layout, AX, R11); // rax=live entries
    a.bytes(&[0x48, 0x69, 0xd2]); // imul rdx,rdx,entry_size
    a.bytes(&(layout.entry_size as i32).to_le_bytes());
    a.bytes(&[0x48, 0x01, 0xd0]); // rax=Property address
    a.bytes(&[0xf6, 0x80]); // reject live accessor, even if its cached shape still matches
    a.bytes(&(layout.entry_accessor as i32).to_le_bytes());
    a.bytes(&[crate::value::PROP_ACCESSOR as u8]);
    a.jcc(0x85, slow);
    a.bytes(&[0x4c, 0x8b, 0x90]); // r10=borrowed live packed property word
    a.bytes(&(layout.entry_value as i32).to_le_bytes());
    a.bytes(&[0x4c, 0x89, 0xd0, 0x48, 0xc1, 0xe8, 0x30]); // eax=value tag
    a.cmp_eax((crate::value::PACK_EMPTY >> 48) as u32);
    a.jcc(0x84, slow);
    a.guard_execution_tag(slow);
    a.bytes(&[0x4c, 0x89, 0xd0]); // rax=the live property word, including BigInt
    a.store_word_r13(0);
    a.add_sp(8);
    #[cfg(test)]
    TEST_OWN_PROPERTY_READS.with(|counts| {
        let index = usize::from(matches!(recv, PropRecv::This));
        a.mov_word_imm(counts[index].as_ptr() as usize as u64);
        a.bytes(&[0x48, 0xff, 0x00]); // inc qword [rax], no callback or production instructions
    });
    true
}

#[allow(clippy::too_many_arguments)]
fn emit_prop_read(
    a: &mut Asm,
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    cache: u32,
    recv: PropRecv,
    pc: u32,
    unwind: usize,
) {
    let slow = a.label();
    let numeric = a.label();
    let done = a.label();
    if emit_prop_own(a, layout, chunk.jit_cache_ptr(cache), recv, numeric) {
        a.jmp(done);
    }
    a.bind(numeric);
    // Retain the established warmed Number path for inherited reads and unsupported layouts.
    if chunk
        .jit_cache_preferred(cache)
        .is_some_and(|state| emit_prop_num(a, layout, state, recv, slow))
    {
        a.jmp(done);
    }
    a.bind(slow);
    a.helper_spflag(H_GET_PROP, pc, unwind);
    a.bind(done);
}

/// Compact monomorphic property-number load. Every cached structural fact is rechecked against
/// the live object graph; a miss reaches H_GET_PROP before the operand stack changes.
fn emit_prop_num(
    a: &mut Asm,
    layout: &crate::value::JitLayout,
    st: crate::bytecode::IcState,
    recv: PropRecv,
    slow: usize,
) -> bool {
    if !layout.valid
        || !storage::fields_data_fits(layout)
        || layout.obj_exotic > i32::MAX as usize
        || st.depth > 2
        || (st.depth == 2 && st.mid_ok & 1 == 0)
        || layout.entry_size.checked_mul(st.slot as usize).is_none()
    {
        return false;
    }
    let Ok(obj_off) = i32::try_from(layout.obj_from_cell) else {
        return false;
    };
    let Ok(plain) = i32::try_from(layout.obj_ic_plain) else {
        return false;
    };
    let Ok(shape) = i32::try_from(layout.obj_props + layout.props_shape) else {
        return false;
    };
    let Ok(proto) = i32::try_from(layout.obj_proto) else {
        return false;
    };
    let Some(entry_base) = layout.entry_size.checked_mul(st.slot as usize) else {
        return false;
    };
    let Ok(value_off) = i32::try_from(entry_base + layout.entry_value) else {
        return false;
    };
    let Ok(accessor_off) = i32::try_from(entry_base + layout.entry_accessor) else {
        return false;
    };

    receiver_cell(a, recv, slow);

    let guard_object = |a: &mut Asm, expected: u32, slow: usize| {
        // r11 = the Object in the cell whose word is in r10.
        a.bytes(&[0x4d, 0x89, 0xd3, 0x49, 0x81, 0xc3]);
        a.bytes(&obj_off.to_le_bytes());
        a.guard_exotic_ordinary(layout, R11, AX, slow);
        a.bytes(&[0x41, 0x80, 0xbb]);
        a.bytes(&plain.to_le_bytes());
        a.bytes(&[0]);
        a.jcc(0x84, slow);
        a.bytes(&[0x41, 0x81, 0xbb]);
        a.bytes(&shape.to_le_bytes());
        a.bytes(&expected.to_le_bytes());
        a.jcc(0x85, slow);
    };
    guard_object(a, st.recv_shape, slow);
    if st.depth >= 1 {
        a.bytes(&[0x4d, 0x8b, 0x93]); // r10=[r11+proto], the prototype's cell word or null
        a.bytes(&proto.to_le_bytes());
        a.bytes(&[0x4d, 0x85, 0xd2]);
        a.jcc(0x84, slow);
        guard_object(
            a,
            if st.depth == 1 {
                st.holder_shape
            } else {
                st.mid_shape
            },
            slow,
        );
    }
    if st.depth == 2 {
        a.bytes(&[0x4d, 0x8b, 0x93]);
        a.bytes(&proto.to_le_bytes());
        a.bytes(&[0x4d, 0x85, 0xd2]);
        a.jcc(0x84, slow);
        guard_object(a, st.holder_shape, slow);
    }

    // rax=the holder's entries; reject a live accessor descriptor, then decode only packed
    // Numbers.
    storage::fields_data(a, layout, AX, R11);
    a.bytes(&[0xf6, 0x80]);
    a.bytes(&accessor_off.to_le_bytes());
    a.bytes(&[crate::value::PROP_ACCESSOR as u8]);
    a.jcc(0x85, slow);
    a.bytes(&[0x48, 0x8b, 0x90]); // rdx=packed value
    a.bytes(&value_off.to_le_bytes());
    a.bytes(&[0x48, 0x89, 0xd0, 0x48, 0xc1, 0xe8, 0x30]); // eax=upper 16 bits
    let number = a.label();
    a.bytes(&[0x3d]);
    a.bytes(&0x7ff9u32.to_le_bytes());
    a.jcc(0x82, number);
    a.bytes(&[0x3d]);
    a.bytes(&0x7fffu32.to_le_bytes());
    a.jcc(0x86, slow); // boxed positive tag
    a.bytes(&[0x3d]);
    a.bytes(&0xfff9u32.to_le_bytes());
    a.jcc(0x83, slow); // PACK_OBJ or a non-canonical negative NaN
    a.bind(number);
    a.bytes(&[0x48, 0x89, 0xd0]); // packed Number rax=rdx
    a.store_word_r13(0);
    a.add_sp(8);
    true
}

pub(super) fn compile(
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
) -> Option<JitCode> {
    compile_entry(
        chunk,
        layout,
        ilayout,
        if chunk.jit_is_resumable() {
            super::NativeEntryKind::BorrowedFrame
        } else {
            super::NativeEntryKind::FreshFrame
        },
    )
}

pub(super) fn compile_entry(
    chunk: &Chunk,
    layout: &crate::value::JitLayout,
    ilayout: &crate::interpreter::InterpLayout,
    entry_kind: super::NativeEntryKind,
) -> Option<JitCode> {
    if !ilayout.valid || ilayout.strict > i32::MAX as usize {
        return None;
    }
    let strict_offset = ilayout.strict as i32;
    let interp_offset = std::mem::offset_of!(crate::jit::JitCtx, interp) as i32;
    let ops = chunk.jit_ops();
    if ops.is_empty() || ops.len() > u32::MAX as usize {
        return None;
    }
    let resumable = chunk.jit_is_resumable();
    let borrowed_entry = entry_kind == super::NativeEntryKind::BorrowedFrame;
    if !borrowed_entry && ops.iter().any(|op| matches!(op, Op::FragmentExit(_))) {
        return None;
    }
    let return_needs_unwind = ops
        .iter()
        .any(|op| matches!(op, Op::PushFinally(..) | Op::PushIterator(..)));
    // Suspension is native only when the caller owns a canonical resumable VM activation.
    if !resumable
        && ops.iter().any(|op| {
            matches!(
                op,
                Op::Await
                    | Op::Yield
                    | Op::YieldStar
                    | Op::AsyncIterStepL(..)
                    | Op::AsyncIterResumeL(..)
                    | Op::AsyncIterCloseL(..)
            )
        })
    {
        return None;
    }
    let cfg = if borrowed_entry && !resumable {
        crate::jit_ir::Cfg::build_osr(chunk)
    } else {
        crate::jit_ir::Cfg::build(chunk)
    }
    .ok()?;
    let max_stack = cfg.jit_stack_capacity();
    let mut a = Asm::new();
    let pcs: Vec<_> = (0..=ops.len()).map(|_| a.label()).collect();
    let unwind = a.label();
    let ret_ok = a.label();
    let ret_throw = a.label();
    let mut interrupt_targets = vec![false; ops.len()];
    for (pc, op) in ops.iter().enumerate() {
        match op {
            Op::Jump(target)
            | Op::AbruptJump(target, _)
            | Op::JumpIfFalse(target)
            | Op::JumpIfFalsePeek(target)
            | Op::JumpIfTruePeek(target)
            | Op::JumpIfNotNullishPeek(target)
                if (*target as usize) <= pc =>
            {
                interrupt_targets[*target as usize] = true;
            }
            _ => {}
        }
    }

    // Preserve five registers so the stack is 16-byte aligned at every helper call.
    a.bytes(&[
        0x55, // push rbp
        0x48, 0x89, 0xe5, // mov rbp,rsp
        0x41, 0x54, // push r12
        0x41, 0x55, // push r13
        0x41, 0x56, // push r14
        0x41, 0x57, // push r15
    ]);
    #[cfg(not(target_os = "windows"))]
    a.bytes(&[0x49, 0x89, 0xfc]); // r12 = rdi (ctx)
    #[cfg(target_os = "windows")]
    a.bytes(&[0x49, 0x89, 0xcc]); // r12 = rcx (ctx)
    a.bytes(&[
        0x4d, 0x8b, 0x34, 0x24, // r14 = [r12] (helpers)
    ]);
    a.bytes(&[
        0x4d,
        0x8b,
        0x6c,
        0x24,
        if borrowed_entry { 0x10 } else { 0x08 },
    ]);
    // A resumed slice starts with the owned live operand prefix, not an empty stack.
    a.bytes(&[0x4d, 0x8b, 0x7c, 0x24, 0x18]); // r15 = [r12+24] (slots)
                                              // ECMA-262 Strict Mode Code / PutValue. Preserve the caller's ambient mode on this
                                              // native frame and install the body's mode for every semantic helper and nested call.
    a.bytes(&[0x48, 0x83, 0xec, 0x10]); // sub rsp,16 (keep helper-call alignment)
    a.bytes(&[0x49, 0x8b, 0x84, 0x24]); // rax = [r12+interp_offset]
    a.bytes(&interp_offset.to_le_bytes());
    a.bytes(&[0x0f, 0xb6, 0x88]); // ecx = byte [rax+strict_offset]
    a.bytes(&strict_offset.to_le_bytes());
    a.bytes(&[0x88, 0x0c, 0x24]); // [rsp] = cl
    if !borrowed_entry {
        a.bytes(&[0xc6, 0x80]); // byte [rax+strict_offset] = body's strictness
        a.bytes(&strict_offset.to_le_bytes());
        a.code.push(u8::from(chunk.jit_is_strict()));
    }
    // Function entry is a safepoint, like a loop back-edge: allocation pressure and host
    // interruption are observed here, through the poll word. Code that never takes a
    // back-edge (loop-free recursion makes billions of calls) must still be interruptible, since
    // a user agent may abort a running script at any time (HTML "Killing scripts"). The frame
    // is fully established and the operand stack empty, as at a back-edge target at pc 0.
    if !borrowed_entry {
        a.interrupt_poll(super::POLL_ENTRY, unwind);
    }

    if borrowed_entry {
        // The shared slice entry validates resume_pc before exposing canonical storage to
        // machine code. Every x64 bytecode PC has an explicit label (no fused-away targets).
        a.bytes(&[0x49, 0x8b, 0x84, 0x24]); // rax = ctx.resume_pc
        a.bytes(&(std::mem::offset_of!(super::JitCtx, resume_pc) as i32).to_le_bytes());
        a.bytes(&[0x49, 0x8b, 0x8c, 0x24]); // rcx = ctx.pc_offsets
        a.bytes(&(std::mem::offset_of!(super::JitCtx, pc_offsets) as i32).to_le_bytes());
        a.bytes(&[0x8b, 0x04, 0x81]); // eax = u32 [rcx+rax*4]
        a.bytes(&[0x49, 0x03, 0x84, 0x24]); // rax += ctx.code_base
        a.bytes(&(std::mem::offset_of!(super::JitCtx, code_base) as i32).to_le_bytes());
        a.bytes(&[0xff, 0xe0]); // jmp rax
    }

    let mut boolean_bit_paths = Vec::new();
    let mut pc_offsets = Vec::with_capacity(ops.len() + 1);
    for (pc, op) in ops.iter().enumerate() {
        a.bind(pcs[pc]);
        pc_offsets.push(a.code.len() as u32);
        if interrupt_targets[pc] {
            a.interrupt_poll(super::POLL_LOOP, unwind);
        }
        if borrowed_entry && crate::bytecode::jit_slice_exit_op(op) {
            a.helper_spflag(H_SLICE_OP, pc as u32, ret_throw);
            a.jmp(ret_ok);
            continue;
        }
        if let Some(plan) = crate::bytecode::string_accumulation::Plan::at(chunk, pc) {
            // All original interior labels remain below. RHS Number/Empty/objects retain
            // their ordinary templates; this private miss flag never enters exception unwind.
            let ordinary = a.label();
            a.load_tag_r15(i32::from(plan.slot) * SLOT_BYTES);
            a.cmp_eax((crate::value::PACK_STR >> 48) as u32);
            a.jcc(0x85, ordinary);
            a.call_helper_pair(super::H_STRING_ACCUMULATION, pc as u32);
            a.bytes(&[0x49, 0x89, 0xc5]); // r13 = canonical sp
            a.bytes(&[0x48, 0x85, 0xd2]); // flag0 is complete success
            a.jcc(0x85, ordinary); // an unchanged miss
            a.helper_end_poll(unwind); // the committed exit ends in the explicit poll
            a.jmp(pcs[plan.end]);
            a.bind(ordinary);
        }
        if names::emit_op(&mut a, chunk, layout, ilayout, op, pc as u32, unwind) {
            continue;
        }
        match op {
            Op::Const(k) if chunk.jit_const_copyable(*k) => {
                a.mov_word_imm(chunk.jit_const_scalar_bits(*k)?);
                a.store_word_r13(0);
                a.add_sp(8);
            }
            // The constant's word owns nothing and names a cell the chunk's pool roots while
            // this code, which the chunk owns, can run: an immediate (see `jit_const_str_word`).
            Op::Const(k) if chunk.jit_const_is_str(*k) => {
                a.mov_word_imm(chunk.jit_const_str_word(*k)?);
                a.store_word_r13(0);
                a.add_sp(8);
            }
            Op::Undef => {
                a.mov_word_imm(PACK_UNDEFINED);
                a.store_word_r13(0);
                a.add_sp(8);
            }
            Op::LoadLocal(slot) => {
                let slow = a.label();
                let exit = a.label();
                let off = i32::from(*slot) * SLOT_BYTES;
                a.load_tag_r15(off);
                a.cmp_eax(0x7ffa); // Empty is a TDZ throw
                a.jcc(0x84, slow);
                a.guard_execution_tag(slow);
                a.load_word_r15(off);
                a.store_word_r13(0);
                a.add_sp(8);
                a.jmp(exit);
                a.bind(slow);
                a.helper_spflag(H_EXEC, pc as u32, unwind);
                a.bind(exit);
            }
            Op::StoreLocal(slot) => {
                // The incoming execution word replaces the old local without reading it.
                a.load_word_r13(-SLOT_BYTES);
                a.store_word_r15(i32::from(*slot) * SLOT_BYTES);
                a.add_sp(-8);
            }
            Op::Pop => a.add_sp(-8),
            Op::Dup => {
                a.load_word_r13(-SLOT_BYTES);
                a.store_word_r13(0);
                a.add_sp(8);
            }
            Op::EqEq | Op::StrictEq | Op::NotEq | Op::StrictNotEq => {
                let slow = a.label();
                let scalar = a.label();
                let done = a.label();
                a.guard_number_r13(-2 * SLOT_BYTES, scalar);
                a.guard_number_r13(-SLOT_BYTES, scalar);
                let ne = matches!(op, Op::NotEq | Op::StrictNotEq);
                a.numeric_compare(if ne { 0x95 } else { 0x94 }, !ne);
                a.jmp(done);
                a.bind(scalar);
                a.scalar_equality(matches!(op, Op::StrictEq | Op::StrictNotEq), ne, slow);
                a.jmp(done);
                a.bind(slow);
                a.helper_spflag(H_EXEC, pc as u32, unwind);
                a.bind(done);
            }
            Op::Lt | Op::Gt | Op::Le | Op::Ge => {
                let slow = a.label();
                let done = a.label();
                a.guard_number_r13(-2 * SLOT_BYTES, slow);
                a.guard_number_r13(-SLOT_BYTES, slow);
                let (setcc, reject_unordered) = match op {
                    Op::Lt => (0x92, true),
                    Op::Gt => (0x97, false),
                    Op::Le => (0x96, true),
                    _ => (0x93, false),
                };
                a.numeric_compare(setcc, reject_unordered);
                a.jmp(done);
                a.bind(slow);
                a.helper_spflag(H_EXEC, pc as u32, unwind);
                a.bind(done);
            }
            Op::BitAnd | Op::BitOr | Op::BitXor => {
                let slow = a.label();
                let done = a.label();
                let booleans = a.label();
                a.guard_number_r13(-2 * SLOT_BYTES, booleans);
                a.guard_number_r13(-SLOT_BYTES, booleans);
                a.numeric_bitop(
                    match op {
                        Op::BitAnd => 0x21,
                        Op::BitOr => 0x09,
                        _ => 0x31,
                    },
                    slow,
                );
                a.jmp(done);
                let opcode = match op {
                    Op::BitAnd => 0x21,
                    Op::BitOr => 0x09,
                    _ => 0x31,
                };
                boolean_bit_paths.push((booleans, slow, done, opcode));
                a.bind(slow);
                a.helper_spflag(H_EXEC, pc as u32, unwind);
                a.bind(done);
            }
            Op::Jump(target) => a.jmp(pcs[*target as usize]),
            Op::JumpIfFalse(target) => {
                let slow = a.label();
                let have = a.label();
                a.truthy_r13(layout, slow);
                a.add_sp(-SLOT_BYTES as i8);
                a.jmp(have);
                a.bind(slow);
                a.call_helper_pair(H_COND, COND_POP_TRUTHY);
                a.bytes(&[0x49, 0x89, 0xc5]);
                a.bind(have);
                a.bytes(&[0x48, 0x85, 0xd2]);
                a.jcc(0x84, pcs[*target as usize]);
            }
            Op::JumpIfFalsePeek(target) | Op::JumpIfTruePeek(target) => {
                let slow = a.label();
                let have = a.label();
                a.truthy_r13(layout, slow);
                a.jmp(have);
                a.bind(slow);
                a.call_helper_pair(H_COND, COND_PEEK_TRUTHY);
                a.bytes(&[0x49, 0x89, 0xc5]);
                a.bind(have);
                a.bytes(&[0x48, 0x85, 0xd2]);
                a.jcc(
                    if matches!(op, Op::JumpIfFalsePeek(_)) {
                        0x84
                    } else {
                        0x85
                    },
                    pcs[*target as usize],
                );
            }
            Op::JumpIfNotNullishPeek(target) => {
                let nullish = a.label();
                a.load_tag_r13(-SLOT_BYTES);
                a.cmp_eax(0x7ff9);
                a.jcc(0x84, nullish);
                a.cmp_eax(0x7ffb);
                a.jcc(0x85, pcs[*target as usize]);
                a.bind(nullish);
            }
            Op::Not => {
                let slow = a.label();
                let done = a.label();
                a.truthy_r13(layout, slow);
                a.bytes(&[0x83, 0xf2, 0x01]); // negate truthiness
                a.mov_word_imm(PACK_BOOL);
                a.bytes(&[0x48, 0x09, 0xd0]);
                a.store_word_r13(-SLOT_BYTES);
                a.jmp(done);
                a.bind(slow);
                a.helper_spflag(H_EXEC, pc as u32, unwind);
                a.bind(done);
            }
            Op::InlineGuard(t, target) => {
                let it = chunk.jit_inline_target(*t);
                let stored = it.pin.upgrade().filter(|_| layout.valid).map(|o| {
                    let some: Option<crate::value::Gc> = Some(o);
                    unsafe { *(&some as *const Option<crate::value::Gc> as *const usize) }
                });
                match stored {
                    None => a.jmp(pcs[*target as usize]),
                    Some(stored) => {
                        if it.expected_env != 0 {
                            // ResolveBinding in a shared-closure splice uses the
                            // caller's environment. Match the live activation,
                            // just as the bytecode and ARM64 guards do.
                            a.bytes(&[0x48, 0xb8]); // movabs rax, expected environment
                            a.bytes(&(it.expected_env as u64).to_le_bytes());
                            a.bytes(&[0x49, 0x39, 0x44, 0x24, 0x28]); // cmp [r12+40],rax
                            a.jcc(0x85, pcs[*target as usize]);
                        }
                        // Same-Realm proof: the splice runs in the planning Realm only
                        // (see `InlineTarget::expected_genv`). JitCtx::genv is at [r12+64].
                        const _: () = assert!(std::mem::offset_of!(crate::jit::JitCtx, genv) == 64);
                        a.bytes(&[0x48, 0xb8]); // movabs rax, expected global environment
                        a.bytes(&(it.expected_genv as u64).to_le_bytes());
                        a.bytes(&[0x49, 0x39, 0x44, 0x24, 0x40]); // cmp [r12+64],rax
                        a.jcc(0x85, pcs[*target as usize]);
                        let callee = -((it.argc as i32 + 1) * SLOT_BYTES);
                        a.mov_word_imm(PACK_OBJ | stored as u64);
                        a.cmp_qword_r13_rax(callee); // exact tag and callee identity
                        a.jcc(0x85, pcs[*target as usize]);
                        if it.check_this {
                            a.cmp_tag_r13(callee - SLOT_BYTES, 0xfff9);
                            a.jcc(0x85, pcs[*target as usize]);
                        }
                    }
                }
            }
            Op::Return | Op::ReturnBare if !return_needs_unwind => {
                a.return_value(u32::from(matches!(op, Op::Return)), ret_ok);
            }
            Op::Return
            | Op::ReturnBare
            | Op::ResumeReturn
            | Op::AbruptJump(..)
            | Op::ResumeJump => {
                a.call_helper_pair(H_COMPLETE, pc as u32);
                a.bytes(&[0x49, 0x89, 0xc5]);
                a.bytes(&[0x48, 0x85, 0xd2]); // test rdx,rdx
                a.jcc(0x84, ret_ok);
                a.bytes(&[0xff, 0xe2]); // jmp rdx
            }
            Op::ReturnUndef => {
                a.return_value(0, ret_ok);
            }
            Op::PushHandler(..) | Op::PushFinally(..) | Op::PushIterator(..) => {
                a.call_helper_ptr(H_PUSH_HANDLER, pc as u32);
                a.bytes(&[0x49, 0x89, 0xc5]);
            }
            Op::PopHandler => {
                a.call_helper_ptr(H_POP_HANDLER, 0);
                a.bytes(&[0x49, 0x89, 0xc5]);
            }
            Op::Throw => {
                a.helper_spflag(H_EXEC, pc as u32, unwind);
                a.jmp(unwind);
            }
            Op::Call(..) | Op::CallWithThis(..) => {
                a.helper_spflag(H_CALL, pc as u32, unwind);
                // A native callee can allocate after the call's last safepoint.
                a.helper_end_poll(unwind);
            }
            // Ends in the helper end poll (`jit::helper_ends_in_poll`).
            Op::CallSpread(..) | Op::CallSpreadThis(..) => {
                a.helper_spflag(H_CALL_SPREAD, pc as u32, unwind)
            }
            Op::New(argc, _) => {
                a.helper_spflag(H_NEW, pc as u32 | ((*argc as u32) << 16), unwind);
                // A native constructor allocates after the call's last safepoint.
                a.helper_end_poll(unwind);
            }
            Op::GetPropThis(_, cache) => {
                emit_prop_read(
                    &mut a,
                    chunk,
                    layout,
                    *cache,
                    PropRecv::This,
                    pc as u32,
                    unwind,
                );
            }
            Op::GetPropLocal(slot, _, cache) => {
                emit_prop_read(
                    &mut a,
                    chunk,
                    layout,
                    *cache,
                    PropRecv::Slot(*slot),
                    pc as u32,
                    unwind,
                );
            }
            Op::GetProp(..) | Op::GetMethod(..) => a.helper_spflag(H_GET_PROP, pc as u32, unwind),
            Op::GetMethodElem => a.helper_spflag(H_GET_METHOD_ELEM, pc as u32, unwind),
            Op::GetElem | Op::GetElemLocal(_) => a.helper_spflag(H_GET_ELEM, pc as u32, unwind),
            Op::SetProp(..)
            | Op::SetPropDrop(..)
            | Op::SetPropThisDrop(..)
            | Op::SetPropLocalDrop(..) => a.helper_spflag(H_SET_PROP, pc as u32, unwind),
            _ => a.helper_spflag(H_EXEC, pc as u32, unwind),
        }
    }
    a.bind(pcs[ops.len()]);
    pc_offsets.push(a.code.len() as u32);
    if borrowed_entry {
        a.helper_spflag(H_SLICE_OP, ops.len() as u32, ret_throw);
        a.jmp(ret_ok);
    } else {
        a.return_value(0, ret_ok);
    }

    for (booleans, slow, done, opcode) in boolean_bit_paths {
        a.bind(booleans);
        // Keep the Number/Number hot path unchanged. Primitive boolean pairs avoid
        // the helper; mixed primitives use jit_bin_i32's checked conversion.
        a.cmp_tag_r13(-2 * SLOT_BYTES, 0x7ffc);
        a.jcc(0x85, slow);
        a.cmp_tag_r13(-SLOT_BYTES, 0x7ffc);
        a.jcc(0x85, slow);
        a.load_word_r13(-2 * SLOT_BYTES);
        a.bytes(&[0x83, 0xe0, 0x01, 0x89, 0xc1]); // ecx=lhs & 1
        a.load_word_r13(-SLOT_BYTES);
        a.bytes(&[0x83, 0xe0, 0x01]); // eax=rhs & 1; operations commute
        a.bytes(&[opcode, 0xc8]);
        a.bytes(&[0xf2, 0x0f, 0x2a, 0xc0]); // cvtsi2sd xmm0,eax
        a.bytes(&[0xf2, 0x41, 0x0f, 0x11, 0x85]);
        a.bytes(&(-2 * SLOT_BYTES).to_le_bytes());
        a.add_sp(-8);
        a.jmp(done);
    }

    a.bind(unwind);
    if borrowed_entry {
        // VmCoro's driver owns abrupt routing and handler state across slices. Never unwind
        // or pop those handlers as if this Rust-visible slice were a complete JS call.
        a.jmp(ret_throw);
    } else {
        a.call_helper_pair(H_UNWIND, 0);
        a.bytes(&[0x48, 0x85, 0xc0]); // test returned catch address
        a.jcc(0x84, ret_throw);
        a.bytes(&[0x49, 0x89, 0xd5, 0xff, 0xe0]); // r13=rdx; jmp rax
    }

    let epilogue = |a: &mut Asm, ok: bool| {
        a.bytes(&[0x4d, 0x89, 0x6c, 0x24, 0x10]); // ctx.final_sp = r13
        a.bytes(&[0x49, 0x8b, 0x84, 0x24]); // rax = [r12+interp_offset]
        a.bytes(&interp_offset.to_le_bytes());
        a.bytes(&[0x8a, 0x0c, 0x24]); // cl = [rsp]
        a.bytes(&[0x88, 0x88]); // byte [rax+strict_offset] = cl
        a.bytes(&strict_offset.to_le_bytes());
        a.bytes(&[0x48, 0x83, 0xc4, 0x10]); // add rsp,16
        if ok {
            a.bytes(&[0xb8, 1, 0, 0, 0]);
        } else {
            a.bytes(&[0x31, 0xc0]);
        }
        a.bytes(&[0x41, 0x5f, 0x41, 0x5e, 0x41, 0x5d, 0x41, 0x5c, 0x5d, 0xc3]);
    };
    a.bind(ret_ok);
    epilogue(&mut a, true);
    a.bind(ret_throw);
    epilogue(&mut a, false);

    let code = a.finish();
    let executable = crate::jit::ExecutableBuffer::from_bytes(&code)?;
    let mem = executable.as_ptr() as *mut u8;
    let len = executable.len();
    Some(JitCode {
        entry_kind,
        osr_entry_depths: if borrowed_entry && !resumable {
            (0..ops.len()).map(|pc| cfg.osr_entry_depth(pc)).collect()
        } else {
            Vec::new()
        },
        mem,
        len,
        pc_offsets,
        resume_depths: if borrowed_entry {
            (0..ops.len()).map(|pc| cfg.stack_depth_at(pc)).collect()
        } else {
            Vec::new()
        },
        max_stack,
        needs_global: ops.iter().any(|o| {
            matches!(
                o,
                Op::LoadName(..)
                    | Op::LoadNameForCall(..)
                    | Op::LoadNameIn(..)
                    | Op::LoadNameForCallIn(..)
            )
        }),
        executable,
        referenced: std::cell::Cell::new(0),
    })
}

#[cfg(test)]
mod encoding_tests {
    use super::Asm;

    #[test]
    fn compact_word_load_store_and_tag_offsets_encode_eight_byte_slots() {
        let mut a = Asm::new();
        a.load_word_r13(-8);
        a.store_word_r15(24);
        a.cmp_tag_r13(-8, 0xfff9);
        assert_eq!(
            a.finish(),
            [
                0x49, 0x8b, 0x85, 0xf8, 0xff, 0xff, 0xff, 0x49, 0x89, 0x87, 0x18, 0, 0, 0, 0x66,
                0x41, 0x81, 0xbd, 0xfe, 0xff, 0xff, 0xff, 0xf9, 0xff,
            ]
        );
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use crate::value::ValueView;
    use crate::{bytecode::Tier, value::Callable, Completion, Engine};

    fn engine() -> Engine {
        let mut engine = Engine::new();
        engine.set_tier(Tier::Jit);
        engine.set_tier_threshold(0);
        engine
    }

    fn evaluate(engine: &mut Engine, source: &str) -> String {
        match engine.eval(source, false).expect("fixture parses") {
            Completion::Value(value) => value,
            Completion::Throw { name, message } => panic!("{name}: {message}"),
        }
    }

    fn assert_native_code(engine: &mut Engine, name: &str, expected: bool) {
        let env = engine.interp.global_env;
        let ValueView::Obj(object) = (engine
            .interp
            .get_var(name, &env)
            .unwrap_or_else(|_| panic!("missing {name}")))
        .view() else {
            panic!("{name} is not an object");
        };
        let object = object.borrow();
        let Callable::User(user) = &object.call else {
            panic!("{name} is not a user function");
        };
        let runtime = user.func.runtime();
        let chunk = runtime.code.get().and_then(Option::as_ref).unwrap();
        assert_eq!(
            chunk.jit.get().map(|code| code.is_some()),
            Some(expected),
            "unexpected native compilation state for {name}"
        );
    }

    #[test]
    fn finally_completions_remain_native_and_preserve_abrupt_precedence() {
        // ECMA-262 e28783d5, sec-try-statement-runtime-semantics-evaluation:
        // normal finally completion preserves the body completion; abrupt finally replaces it.
        let mut engine = engine();
        assert_eq!(
            evaluate(
                &mut engine,
                "var events = [];
                 function plain(x) { return x + 1; }
                 function bare() { return; }
                 function cleanup(mode) {
                     try {
                         if (mode === 0) return plain(6);
                         if (mode === 1) throw 'body';
                         return 1;
                     } finally {
                         events.push(mode);
                         if (mode === 2) return 9;
                         if (mode === 3) throw 'finally';
                     }
                 }
                 var results = [cleanup(0)];
                 try { cleanup(1); } catch (e) { results.push(e); }
                 results.push(cleanup(2));
                 try { cleanup(3); } catch (e) { results.push(e); }
                 results.push(bare() === undefined);
                 results.join(',') + '|' + events.join(',')"
            ),
            "7,body,9,finally,true|0,1,2,3"
        );
        assert_native_code(&mut engine, "cleanup", true);
        assert_native_code(&mut engine, "plain", true);
        assert_native_code(&mut engine, "bare", true);
    }

    #[test]
    fn labelled_break_runs_finally_then_closes_iterators_inside_out() {
        // ECMA-262 e28783d5, sec-iteratorclose and TryStatement Evaluation: preserve
        // the pending break while running every intervening cleanup in nesting order.
        let mut engine = engine();
        assert_eq!(
            evaluate(
                &mut engine,
                "var events = [];
                 function iterable(name) {
                     return { [Symbol.iterator]() { return {
                         next() { return {value: 1, done: false}; },
                         return() { events.push(name); return {}; }
                     }; } };
                 }
                 function leave() {
                     outer: for (var x of iterable('outer')) {
                         for (var y of iterable('inner')) {
                             try { break outer; } finally { events.push('finally'); }
                         }
                     }
                     return events.join(',');
                 }
                 leave()"
            ),
            "finally,inner,outer"
        );
        assert_native_code(&mut engine, "leave", true);
    }

    #[test]
    fn packed_numeric_and_boolean_operands_preserve_number_semantics() {
        let mut engine = engine();
        assert_eq!(evaluate(&mut engine,
            "function comparisons(a,b) {
                return [a===b,a!==b,a<b,a<=b,a>b,a>=b].join(',');
             }
             function bits(a,b) { return [a&b,a|b,a^b].join(','); }
             [comparisons(NaN,NaN), comparisons(-0,0), comparisons(-Infinity,Infinity),
              bits(true,false), bits(-2147483648,2147483647), bits(4294967297,3)].join('|')"),
            "false,true,false,false,false,false|true,false,false,true,false,true|false,true,true,true,false,false|0,1,1|0,-1,-1|1,3,2");
        assert_native_code(&mut engine, "comparisons", true);
        assert_native_code(&mut engine, "bits", true);
    }

    #[test]
    fn packed_heap_owners_survive_local_overwrites_calls_and_unwind() {
        let mut engine = engine();
        assert_eq!(
            evaluate(
                &mut engine,
                "function transfer(a,b,c,d,e,f) {
                let old = {discard:true}; old = a;
                let twice; twice = old;
                try { if (f) throw twice; return [old===a,b===c,d,e].join(','); }
                catch (caught) { return caught===a; }
             }
             var object = {value:42}, symbol=Symbol('identity');
             var first=transfer(object,symbol,symbol,'string',12345678901234567890n,false);
             var second=transfer(object,symbol,symbol,'string',12345678901234567890n,true);
             first+'|'+second+'|'+object.value"
            ),
            "true,true,string,12345678901234567890|true|42"
        );
        assert_native_code(&mut engine, "transfer", true);
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "jit_x64_property_tests.rs"]
mod property_tests;
