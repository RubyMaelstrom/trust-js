//! The native RegExp tier's x86-64 backend (System V ABI; see `regex_native.rs`): the [`Masm`]
//! operations as x86-64 instructions, with their own small assembler.
//!
//! Registers: rbx the subject, r12 the position, r13 the slots, r14 the backtrack stack's top,
//! r15 the context, rbp the choice count (callee-saved); r8 the length, r9 the root floor's
//! logged slots, r10 the poll counter, eax the element, rdx the popped operand, rcx a loop's run
//! start, rsi and rdi a scan's limit and stretch end, r11 scratch (caller-saved, which every
//! helper call's routine saves but for rax); the leading loop's run end and the stack pointer
//! after the entry's pushes are kept in the context.

use super::{ctx, Cond, Data, Helpers, Label, Masm, Operand, Value, Width};
use super::{CANDIDATE_RUN, POLL_INTERRUPT_BIT, POLL_PERIOD, SCAN_STRETCH, STATUS_EXHAUSTED};

const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;
const RBX: u8 = 3;
const RSP: u8 = 4;
const RBP: u8 = 5;
const RSI: u8 = 6;
const RDI: u8 = 7;
const R8: u8 = 8;
const R9: u8 = 9;
const R10: u8 = 10;
const R11: u8 = 11;
const R12: u8 = 12;
const R13: u8 = 13;
const R14: u8 = 14;
const R15: u8 = 15;

const SUBJ: u8 = RBX;
const POS: u8 = R12;
const SLOTS: u8 = R13;
const TOP: u8 = R14;
const CTX: u8 = R15;
const CHOICES: u8 = RBP;
const LEN: u8 = R8;
const LOGGED: u8 = R9;
const POLL: u8 = R10;
const E: u8 = RAX;
const OP: u8 = RDX;
const RS: u8 = RCX;
const LIM: u8 = RSI;
const CLIM: u8 = RDI;
const SCRATCH: u8 = R11;
/// A loop's held table address (see [`Masm::hold_lut`]): the popped operand's register, which
/// no loop's scan or candidate test reads.
const HELD: u8 = RDX;

/// The caller-saved registers the code keeps values in, which helper calls' routines save.
const SAVED: [u8; 8] = [RCX, RDX, RSI, RDI, R8, R9, R10, R11];

/// The condition nibble of a Jcc or CMOVcc.
fn cc(cond: Cond) -> u8 {
    match cond {
        Cond::Lo => 0x2,
        Cond::Hs => 0x3,
        Cond::Eq => 0x4,
        Cond::Ne => 0x5,
        Cond::Ls => 0x6,
        Cond::Hi => 0x7,
    }
}

/// Out-of-line code that sites request, emitted after the program.
enum Outline {
    Poll { slow: Label, back: Label },
    Grow { slow: Label, retry: Label },
    CandidateStatus { slow: Label, no_match: Label },
}

pub(super) struct Asm {
    width: Width,
    helpers: Helpers,
    code: Vec<u8>,
    labels: Vec<Option<usize>>,
    /// rel32 fields to patch: (offset of the field, label).
    fixups: Vec<(usize, Label)>,
    data: Vec<u8>,
    /// RIP-relative disp32 fields of data references: (offset of the field, data offset).
    data_refs: Vec<(usize, usize)>,
    luts: std::collections::HashMap<[u8; 256], Data>,
    literals: std::collections::HashMap<u64, usize>,
    outlines: Vec<Outline>,
    poll_routine: Label,
    grow_routine: Label,
    candidate_routine: Label,
    exit: Label,
    /// The table whose address rdx holds.
    held: Option<Data>,
    /// An operand past its field (a LOGGED bit from 64 on): `finish` refuses the code.
    unencodable: bool,
}

impl Asm {
    pub(super) fn new(width: Width, helpers: Helpers) -> Self {
        let mut asm = Asm {
            width,
            helpers,
            code: Vec::new(),
            labels: Vec::new(),
            fixups: Vec::new(),
            data: Vec::new(),
            data_refs: Vec::new(),
            luts: Default::default(),
            literals: Default::default(),
            outlines: Vec::new(),
            poll_routine: 0,
            grow_routine: 0,
            candidate_routine: 0,
            exit: 0,
            held: None,
            unencodable: false,
        };
        asm.poll_routine = asm.label();
        asm.grow_routine = asm.label();
        asm.candidate_routine = asm.label();
        asm.exit = asm.label();
        asm
    }

    fn byte(&mut self, byte: u8) {
        self.code.push(byte);
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    fn imm32(&mut self, value: i32) {
        self.bytes(&value.to_le_bytes());
    }

    /// A REX prefix, when one is needed.
    fn rex(&mut self, wide: bool, reg: u8, index: u8, base: u8) {
        let rex = 0x40
            | u8::from(wide) << 3
            | ((reg >> 3) & 1) << 2
            | ((index >> 3) & 1) << 1
            | ((base >> 3) & 1);
        if rex != 0x40 {
            self.byte(rex);
        }
    }

    /// `opcode` with a register-direct operand: ModRM `11 reg rm`.
    fn op_rr(&mut self, wide: bool, opcode: &[u8], reg: u8, rm: u8) {
        self.rex(wide, reg, 0, rm);
        self.bytes(opcode);
        self.byte(0xc0 | (reg & 7) << 3 | (rm & 7));
    }

    /// `opcode` with a memory operand `[base + disp32]`.
    fn op_mem(&mut self, wide: bool, opcode: &[u8], reg: u8, base: u8, disp: i32) {
        self.rex(wide, reg, 0, base);
        self.bytes(opcode);
        if base & 7 == 4 {
            self.byte(0x80 | (reg & 7) << 3 | 4);
            self.byte(0x24);
        } else {
            self.byte(0x80 | (reg & 7) << 3 | (base & 7));
        }
        self.imm32(disp);
    }

    /// `opcode` with a memory operand `[base + index << scale + disp32]`.
    fn op_sib(
        &mut self,
        wide: bool,
        opcode: &[u8],
        reg: u8,
        base: u8,
        index: u8,
        scale: u8,
        disp: i32,
    ) {
        debug_assert!(index != RSP);
        self.rex(wide, reg, index, base);
        self.bytes(opcode);
        self.byte(0x80 | (reg & 7) << 3 | 4);
        self.byte(scale << 6 | (index & 7) << 3 | (base & 7));
        self.imm32(disp);
    }

    /// `opcode` with a memory operand `[rip + disp32]`, the disp32 last in the instruction.
    fn op_rip(&mut self, wide: bool, opcode: &[u8], reg: u8) -> usize {
        self.rex(wide, reg, 0, 0);
        self.bytes(opcode);
        self.byte((reg & 7) << 3 | 5);
        let field = self.code.len();
        self.imm32(0);
        field
    }

    fn mov(&mut self, dst: u8, src: u8) {
        self.op_rr(true, &[0x89], src, dst);
    }

    fn load(&mut self, dst: u8, base: u8, disp: usize) {
        self.op_mem(true, &[0x8b], dst, base, disp as i32);
    }

    fn store(&mut self, base: u8, disp: usize, src: u8) {
        self.op_mem(true, &[0x89], src, base, disp as i32);
    }

    /// `mov r32, imm32`, which zero-extends.
    fn mov_imm32(&mut self, dst: u8, value: u32) {
        self.rex(false, 0, 0, dst);
        self.byte(0xb8 | (dst & 7));
        self.imm32(value as i32);
    }

    /// An ALU operation (`/digit`: 0 add, 1 or, 4 and, 5 sub, 7 cmp) of `dst` with `imm`.
    fn alu_imm(&mut self, wide: bool, digit: u8, dst: u8, imm: i32) {
        if (-128..128).contains(&imm) {
            self.op_rr(wide, &[0x83], digit, dst);
            self.byte(imm as i8 as u8);
        } else {
            self.op_rr(wide, &[0x81], digit, dst);
            self.imm32(imm);
        }
    }

    /// An ALU operation (`opcode`: 0x01 add, 0x09 or, 0x29 sub, 0x39 cmp, 0x85 test) of `dst`
    /// with `src`.
    fn alu(&mut self, wide: bool, opcode: u8, dst: u8, src: u8) {
        self.op_rr(wide, &[opcode], src, dst);
    }

    fn shift(&mut self, right: bool, dst: u8, by: u8) {
        self.op_rr(true, &[0xc1], if right { 5 } else { 4 }, dst);
        self.byte(by);
    }

    fn cmov(&mut self, cond: Cond, dst: u8, src: u8) {
        self.op_rr(true, &[0x0f, 0x40 | cc(cond)], dst, src);
    }

    fn jcc(&mut self, cond: Cond, label: Label) {
        self.bytes(&[0x0f, 0x80 | cc(cond)]);
        self.rel32(label);
    }

    fn rel32(&mut self, label: Label) {
        self.fixups.push((self.code.len(), label));
        self.imm32(0);
    }

    fn call(&mut self, label: Label) {
        self.byte(0xe8);
        self.rel32(label);
    }

    fn lea_label(&mut self, dst: u8, label: Label) {
        let field = self.op_rip(true, &[0x8d], dst);
        self.fixups.push((field, label));
    }

    fn lea_data(&mut self, dst: u8, offset: usize) {
        let field = self.op_rip(true, &[0x8d], dst);
        self.data_refs.push((field, offset));
    }

    /// `call [rip + <literal>]` of the 64-bit `value` in the data.
    fn call_literal(&mut self, value: u64) {
        let offset = match self.literals.get(&value) {
            Some(&offset) => offset,
            None => {
                while !self.data.len().is_multiple_of(8) {
                    self.data.push(0);
                }
                let offset = self.data.len();
                self.data.extend_from_slice(&value.to_le_bytes());
                self.literals.insert(value, offset);
                offset
            }
        };
        let field = self.op_rip(false, &[0xff], 2);
        self.data_refs.push((field, offset));
    }

    fn push_reg(&mut self, reg: u8) {
        self.rex(false, 0, 0, reg);
        self.byte(0x50 | (reg & 7));
    }

    fn pop_reg(&mut self, reg: u8) {
        self.rex(false, 0, 0, reg);
        self.byte(0x58 | (reg & 7));
    }

    fn ret_instruction(&mut self) {
        self.byte(0xc3);
    }

    /// Save the caller-saved registers the code keeps (all but rax), aligning the stack for a
    /// call from a routine entered by `call`.
    fn save_scratch(&mut self) {
        for reg in SAVED {
            self.push_reg(reg);
        }
        self.alu_imm(true, 5, RSP, 8);
    }

    fn restore_scratch(&mut self) {
        self.alu_imm(true, 0, RSP, 8);
        for reg in SAVED.iter().rev() {
            self.pop_reg(*reg);
        }
    }

    fn slot_disp(slot: usize) -> usize {
        slot * 8
    }

    fn emit_outlines(&mut self) {
        for outline in std::mem::take(&mut self.outlines) {
            match outline {
                Outline::Poll { slow, back } => {
                    self.bind(slow);
                    self.call(self.poll_routine);
                    self.jump(back);
                }
                Outline::Grow { slow, retry } => {
                    self.bind(slow);
                    self.call(self.grow_routine);
                    self.jump(retry);
                }
                Outline::CandidateStatus { slow, no_match } => {
                    self.bind(slow);
                    self.alu_imm(false, 4, RAX, 0xff);
                    self.jcc(Cond::Eq, no_match);
                    self.jump(self.exit);
                }
            }
        }
    }

    fn referenced(&self, label: Label) -> bool {
        self.fixups.iter().any(|&(_, target)| target == label)
    }

    fn emit_routines(&mut self) {
        let poll = self.poll_routine;
        if self.referenced(poll) {
            let call = self.label();
            self.bind(poll);
            self.mov_imm32(POLL, POLL_PERIOD);
            self.load(SCRATCH, CTX, ctx::POLL_WORD);
            // test byte [r11], interrupt bit
            self.op_mem(false, &[0xf6], 0, SCRATCH, 0);
            self.byte(1 << POLL_INTERRUPT_BIT);
            self.jcc(Cond::Ne, call);
            self.ret_instruction();
            self.bind(call);
            self.save_scratch();
            self.mov(RDI, CTX);
            self.call_literal(self.helpers.poll as u64);
            self.alu(true, 0x85, RAX, RAX);
            self.jcc(Cond::Ne, self.exit);
            self.restore_scratch();
            self.ret_instruction();
        }
        let grow = self.grow_routine;
        if self.referenced(grow) {
            let exhausted = self.label();
            self.bind(grow);
            self.save_scratch();
            self.mov(RDI, CTX);
            self.mov(RSI, TOP);
            self.call_literal(self.helpers.grow as u64);
            self.alu(true, 0x85, RAX, RAX);
            self.jcc(Cond::Eq, exhausted);
            self.mov(TOP, RAX);
            self.restore_scratch();
            self.ret_instruction();
            self.bind(exhausted);
            self.mov_imm32(RAX, STATUS_EXHAUSTED as u32);
            self.jump(self.exit);
        }
        let candidate = self.candidate_routine;
        if self.referenced(candidate) {
            self.bind(candidate);
            self.save_scratch();
            self.mov(RDI, CTX);
            self.mov(RSI, POS);
            self.call_literal(self.helpers.next_candidate as u64);
            self.restore_scratch();
            self.ret_instruction();
        }
        // Leave with rax's status, from wherever the routines left the stack.
        self.bind(self.exit);
        self.load(RSP, CTX, ctx::SAVED_SP);
        self.alu_imm(true, 0, RSP, 8);
        for reg in [R15, R14, R13, R12, RBX, RBP] {
            self.pop_reg(reg);
        }
        self.ret_instruction();
    }
}

impl Masm for Asm {
    fn label(&mut self) -> Label {
        self.labels.push(None);
        self.labels.len() - 1
    }

    fn bind(&mut self, label: Label) {
        debug_assert!(self.labels[label].is_none(), "label bound twice");
        self.labels[label] = Some(self.code.len());
    }

    fn jump(&mut self, label: Label) {
        self.byte(0xe9);
        self.rel32(label);
    }

    fn prologue(&mut self, sentinel: Label) {
        for reg in [RBP, RBX, R12, R13, R14, R15] {
            self.push_reg(reg);
        }
        self.alu_imm(true, 5, RSP, 8);
        self.mov(CTX, RDI);
        self.store(CTX, ctx::SAVED_SP, RSP);
        self.load(SUBJ, CTX, ctx::SUBJECT);
        self.load(LEN, CTX, ctx::LEN);
        self.load(POS, CTX, ctx::START);
        self.load(SLOTS, CTX, ctx::SLOTS);
        self.load(TOP, CTX, ctx::STACK);
        self.lea_label(SCRATCH, sentinel);
        self.store(TOP, 0, SCRATCH);
        // mov qword [r14 + 8], 0
        self.op_mem(true, &[0xc7], 0, TOP, 8);
        self.imm32(0);
        self.alu_imm(true, 0, TOP, 16);
        self.mov_imm32(CHOICES, 0);
        self.mov_imm32(POLL, POLL_PERIOD);
    }

    fn ret(&mut self, status: u64) {
        self.mov_imm32(RAX, status as u32);
        self.jump(self.exit);
    }

    fn poll_tick(&mut self) {
        let slow = self.label();
        let back = self.label();
        self.alu_imm(false, 5, POLL, 1);
        self.jcc(Cond::Eq, slow);
        self.bind(back);
        self.outlines.push(Outline::Poll { slow, back });
    }

    fn poll_now(&mut self) {
        self.call(self.poll_routine);
    }

    fn set_logged(&mut self, mask: u64) {
        if mask == u64::MAX {
            self.op_rr(true, &[0xc7], 0, LOGGED);
            self.imm32(-1);
        } else if let Ok(mask) = u32::try_from(mask) {
            self.mov_imm32(LOGGED, mask);
        } else {
            // mov r64, imm64
            self.rex(true, 0, 0, LOGGED);
            self.byte(0xb8 | (LOGGED & 7));
            self.bytes(&mask.to_le_bytes());
        }
    }

    fn fill_unset(&mut self, slots: std::ops::Range<usize>) {
        for slot in slots {
            self.op_mem(true, &[0xc7], 0, SLOTS, Self::slot_disp(slot) as i32);
            self.imm32(-1);
        }
    }

    fn rearm_sentinel(&mut self) {
        self.alu_imm(true, 0, TOP, 16);
    }

    fn branch_room_lt(&mut self, k: u32, label: Label) {
        self.mov(SCRATCH, LEN);
        self.alu(true, 0x29, SCRATCH, POS);
        self.alu_imm(true, 7, SCRATCH, k as i32);
        self.jcc(Cond::Lo, label);
    }

    fn branch_at_start(&mut self, at: bool, label: Label) {
        self.alu(true, 0x85, POS, POS);
        self.jcc(if at { Cond::Eq } else { Cond::Ne }, label);
    }

    fn branch_at_end(&mut self, at: bool, label: Label) {
        self.alu(true, 0x39, POS, LEN);
        self.jcc(if at { Cond::Hs } else { Cond::Ne }, label);
    }

    fn advance(&mut self, k: u32) {
        self.alu_imm(true, 0, POS, k as i32);
    }

    fn retreat(&mut self, k: u32) {
        self.alu_imm(true, 5, POS, k as i32);
    }

    fn load_elem(&mut self, offset: i32) {
        match self.width {
            Width::One => self.op_sib(false, &[0x0f, 0xb6], E, SUBJ, POS, 0, offset),
            Width::Two => self.op_sib(false, &[0x0f, 0xb7], E, SUBJ, POS, 1, offset * 2),
        }
    }

    fn branch_elem(&mut self, cond: Cond, value: u32, label: Label) {
        self.alu_imm(false, 7, E, value as i32);
        self.jcc(cond, label);
    }

    fn branch_elem_in(&mut self, lo: u32, hi: u32, inside: bool, label: Label) {
        // lea r11d, [rax - lo]; cmp r11d, hi - lo
        self.op_mem(false, &[0x8d], SCRATCH, E, -(lo as i32));
        self.alu_imm(false, 7, SCRATCH, (hi - lo) as i32);
        self.jcc(if inside { Cond::Ls } else { Cond::Hi }, label);
    }

    fn branch_elem_pair(&mut self, lo: u32, bit: u32, member: bool, label: Label) {
        debug_assert!(bit.is_power_of_two() && lo & bit == 0);
        // lea r11d, [rax - lo]; test r11d, ~bit: zero for lo and lo + bit alone.
        self.op_mem(false, &[0x8d], SCRATCH, E, -(lo as i32));
        self.op_rr(false, &[0xf7], 0, SCRATCH);
        self.imm32(!bit as i32);
        self.jcc(if member { Cond::Eq } else { Cond::Ne }, label);
    }

    fn lut(&mut self, table: &[u8; 256]) -> Data {
        if let Some(&offset) = self.luts.get(table) {
            return offset;
        }
        let offset = self.data.len();
        self.data.extend_from_slice(table);
        self.luts.insert(*table, offset);
        offset
    }

    fn branch_lut(&mut self, lut: Data, member: bool, label: Label) {
        let base = if self.held == Some(lut) {
            HELD
        } else {
            self.lea_data(SCRATCH, lut);
            SCRATCH
        };
        // cmp byte [base + rax], 0
        self.op_sib(false, &[0x80], 7, base, E, 0, 0);
        self.byte(0);
        self.jcc(if member { Cond::Ne } else { Cond::Eq }, label);
    }

    fn hold_lut(&mut self, lut: Data) {
        debug_assert!(self.held.is_none());
        self.lea_data(HELD, lut);
        self.held = Some(lut);
    }

    fn release_lut(&mut self) {
        self.held = None;
    }

    fn push(&mut self, stub: Label, operand: Operand) {
        let retry = self.label();
        let slow = self.label();
        self.bind(retry);
        // cmp r14, [r15 + stack_end]
        self.op_mem(true, &[0x3b], TOP, CTX, ctx::STACK_END as i32);
        self.jcc(Cond::Hs, slow);
        self.outlines.push(Outline::Grow { slow, retry });
        self.lea_label(SCRATCH, stub);
        self.store(TOP, 0, SCRATCH);
        let value = match operand {
            Operand::Pos => POS,
            Operand::Slot(slot) => {
                self.load(RAX, SLOTS, Self::slot_disp(slot));
                RAX
            }
            Operand::Pack(add) => {
                // lea rax, [rcx + add]; shl rax, 32; or rax, r12
                self.op_mem(true, &[0x8d], RAX, RS, add as i32);
                self.shift(false, RAX, 32);
                self.alu(true, 0x09, RAX, POS);
                RAX
            }
        };
        self.store(TOP, 8, value);
        self.alu_imm(true, 0, TOP, 16);
    }

    fn pop_dispatch(&mut self) {
        self.alu_imm(true, 5, TOP, 16);
        self.load(OP, TOP, 8);
        // jmp [r14]
        self.op_mem(false, &[0xff], 4, TOP, 0);
    }

    fn choices(&mut self, delta: i32) {
        self.alu_imm(true, if delta > 0 { 0 } else { 5 }, CHOICES, delta.abs());
    }

    fn branch_choices(&mut self, zero: bool, label: Label) {
        self.alu(true, 0x85, CHOICES, CHOICES);
        self.jcc(if zero { Cond::Eq } else { Cond::Ne }, label);
    }

    fn branch_logged(&mut self, slot: usize, label: Label) {
        // bt r9, slot: the bit lands in the carry flag. The CPU takes the imm8 modulo 64.
        self.unencodable |= slot >= 64;
        self.op_rr(true, &[0x0f, 0xba], 4, LOGGED);
        self.byte(slot as u8);
        self.jcc(Cond::Lo, label);
    }

    fn log_slot(&mut self, slot: usize) {
        // bts r9, slot
        self.unencodable |= slot >= 64;
        self.op_rr(true, &[0x0f, 0xba], 5, LOGGED);
        self.byte(slot as u8);
    }

    fn store_slot(&mut self, slot: usize, value: Value) {
        match value {
            Value::Pos => self.store(SLOTS, Self::slot_disp(slot), POS),
            Value::Op => self.store(SLOTS, Self::slot_disp(slot), OP),
            Value::Unset => {
                self.op_mem(true, &[0xc7], 0, SLOTS, Self::slot_disp(slot) as i32);
                self.imm32(-1);
            }
        }
    }

    fn branch_slot_unset(&mut self, slot: usize, label: Label) {
        // cmp qword [r13 + disp], -1
        self.op_mem(true, &[0x83], 7, SLOTS, Self::slot_disp(slot) as i32);
        self.byte(0xff);
        self.jcc(Cond::Eq, label);
    }

    fn branch_slot_eq_pos(&mut self, slot: usize, label: Label) {
        // cmp [r13 + disp], r12
        self.op_mem(true, &[0x39], POS, SLOTS, Self::slot_disp(slot) as i32);
        self.jcc(Cond::Eq, label);
    }

    fn pos_from_op(&mut self) {
        self.mov(POS, OP);
    }

    fn unpack_op(&mut self) {
        // mov r12d, edx (zero-extends); mov rcx, rdx; shr rcx, 32
        self.op_rr(false, &[0x89], OP, POS);
        self.mov(RS, OP);
        self.shift(true, RS, 32);
    }

    fn run_start(&mut self) {
        self.mov(RS, POS);
    }

    fn set_limit(&mut self, max: Option<u32>) {
        match max {
            None => self.mov(LIM, LEN),
            Some(max) => {
                self.op_mem(true, &[0x8d], LIM, POS, max as i32);
                self.alu(true, 0x39, LIM, LEN);
                self.cmov(Cond::Hi, LIM, LEN);
            }
        }
    }

    fn set_stretch_limit(&mut self) {
        self.op_mem(true, &[0x8d], CLIM, POS, SCAN_STRETCH as i32);
        self.alu(true, 0x39, CLIM, LIM);
        self.cmov(Cond::Hi, CLIM, LIM);
    }

    fn set_candidate_bound(&mut self, window: bool) {
        self.op_mem(true, &[0x8d], CLIM, POS, CANDIDATE_RUN as i32);
        self.alu(true, 0x39, CLIM, LEN);
        self.cmov(Cond::Hi, CLIM, LEN);
        if window {
            self.load(SCRATCH, CTX, ctx::WINDOW);
            self.alu(true, 0x39, CLIM, SCRATCH);
            self.cmov(Cond::Hi, CLIM, SCRATCH);
        }
    }

    fn branch_pos_stretch(&mut self, cond: Cond, label: Label) {
        self.alu(true, 0x39, POS, CLIM);
        self.jcc(cond, label);
    }

    fn branch_stretch_is_limit(&mut self, label: Label) {
        self.alu(true, 0x39, CLIM, LIM);
        self.jcc(Cond::Eq, label);
    }

    fn pos_to_limit(&mut self) {
        self.mov(POS, LIM);
    }

    fn branch_run(&mut self, cond: Cond, n: u32, label: Label) {
        self.mov(SCRATCH, POS);
        self.alu(true, 0x29, SCRATCH, RS);
        self.alu_imm(true, 7, SCRATCH, n as i32);
        self.jcc(cond, label);
    }

    fn branch_pos_rs(&mut self, cond: Cond, label: Label) {
        self.alu(true, 0x39, POS, RS);
        self.jcc(cond, label);
    }

    fn store_lead(&mut self) {
        self.store(CTX, ctx::LEAD, POS);
    }

    fn branch_window(&mut self, label: Label) {
        // cmp r12, [r15 + window]
        self.op_mem(true, &[0x3b], POS, CTX, ctx::WINDOW as i32);
        self.jcc(Cond::Hs, label);
    }

    fn next_candidate(&mut self, no_match: Label) {
        let slow = self.label();
        self.call(self.candidate_routine);
        self.alu(true, 0x85, RAX, RAX);
        // js: bit 63 carries a status.
        self.bytes(&[0x0f, 0x88]);
        self.rel32(slow);
        self.mov(POS, RAX);
        self.outlines
            .push(Outline::CandidateStatus { slow, no_match });
    }

    fn next_start(&mut self, lead: bool, no_match: Label) {
        self.load(SCRATCH, SLOTS, 0);
        self.alu(true, 0x39, SCRATCH, LEN);
        self.jcc(Cond::Hs, no_match);
        self.op_mem(true, &[0x8d], POS, SCRATCH, 1);
        if lead {
            self.load(SCRATCH, CTX, ctx::LEAD);
            self.alu(true, 0x39, POS, SCRATCH);
            self.cmov(Cond::Lo, POS, SCRATCH);
        }
    }

    fn finish(mut self) -> Option<Vec<u8>> {
        self.emit_outlines();
        self.emit_routines();
        if self.unencodable {
            return None;
        }
        let data_start = self.code.len().next_multiple_of(8);
        for (field, label) in std::mem::take(&mut self.fixups) {
            let target = self.labels.get(label).copied().flatten()?;
            let delta = i32::try_from(target as isize - (field as isize + 4)).ok()?;
            self.code[field..field + 4].copy_from_slice(&delta.to_le_bytes());
        }
        for (field, offset) in std::mem::take(&mut self.data_refs) {
            let delta =
                i32::try_from((data_start + offset) as isize - (field as isize + 4)).ok()?;
            self.code[field..field + 4].copy_from_slice(&delta.to_le_bytes());
        }
        let mut bytes = self.code;
        // int3 up to the data.
        bytes.resize(data_start, 0xcc);
        bytes.extend_from_slice(&self.data);
        Some(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The encoders, each case checked against GNU objdump's disassembly of its bytes
    /// (`objdump -D -b binary -m i386:x86-64 -M intel`, binutils 2.42).
    #[test]
    fn encodings_disassemble_as_intended() {
        type Emit = fn(&mut Asm);
        let cases: &[(Emit, &str, &str)] = &[
            (|a| a.push_reg(R12), "41 54", "push r12"),
            (|a| a.pop_reg(RBX), "5b", "pop rbx"),
            (|a| a.mov(R15, RDI), "49 89 ff", "mov r15,rdi"),
            (
                |a| a.load(RBX, R15, 0),
                "49 8b 9f 00 00 00 00",
                "mov rbx,QWORD PTR [r15+0x0]",
            ),
            (
                |a| a.store(R15, 96, RSP),
                "49 89 a7 60 00 00 00",
                "mov QWORD PTR [r15+0x60],rsp",
            ),
            (
                |a| a.load(R11, R12, 8),
                "4d 8b 9c 24 08 00 00 00",
                "mov r11,QWORD PTR [r12+0x8]",
            ),
            (
                |a| a.mov_imm32(R10, 16384),
                "41 ba 00 40 00 00",
                "mov r10d,0x4000",
            ),
            (|a| a.alu_imm(true, 5, RSP, 8), "48 83 ec 08", "sub rsp,0x8"),
            (
                |a| a.alu_imm(false, 7, RAX, 0x2028),
                "81 f8 28 20 00 00",
                "cmp eax,0x2028",
            ),
            (|a| a.alu_imm(true, 7, R11, 9), "49 83 fb 09", "cmp r11,0x9"),
            (|a| a.alu(true, 0x39, R12, R8), "4d 39 c4", "cmp r12,r8"),
            (|a| a.alu(true, 0x29, R11, R12), "4d 29 e3", "sub r11,r12"),
            (|a| a.alu(true, 0x85, RBP, RBP), "48 85 ed", "test rbp,rbp"),
            (|a| a.alu(true, 0x09, RAX, R12), "4c 09 e0", "or rax,r12"),
            (|a| a.shift(false, RAX, 32), "48 c1 e0 20", "shl rax,0x20"),
            (|a| a.shift(true, RCX, 32), "48 c1 e9 20", "shr rcx,0x20"),
            (|a| a.cmov(Cond::Hi, RSI, R8), "49 0f 47 f0", "cmova rsi,r8"),
            (
                |a| a.cmov(Cond::Lo, R12, R11),
                "4d 0f 42 e3",
                "cmovb r12,r11",
            ),
            (
                |a| a.op_sib(false, &[0x0f, 0xb6], RAX, RBX, R12, 0, 5),
                "42 0f b6 84 23 05 00 00 00",
                "movzx eax,BYTE PTR [rbx+r12*1+0x5]",
            ),
            (
                |a| a.op_sib(false, &[0x0f, 0xb7], RAX, RBX, R12, 1, -2),
                "42 0f b7 84 63 fe ff ff ff",
                "movzx eax,WORD PTR [rbx+r12*2-0x2]",
            ),
            (
                |a| {
                    a.op_sib(false, &[0x80], 7, R11, RAX, 0, 0);
                    a.byte(0);
                },
                "41 80 bc 03 00 00 00 00 00",
                "cmp BYTE PTR [r11+rax*1+0x0],0x0",
            ),
            (
                |a| a.op_mem(false, &[0x8d], R11, RAX, -0x2028),
                "44 8d 98 d8 df ff ff",
                "lea r11d,[rax-0x2028]",
            ),
            (
                |a| a.op_mem(true, &[0x8d], RSI, R12, 100),
                "49 8d b4 24 64 00 00 00",
                "lea rsi,[r12+0x64]",
            ),
            (
                |a| a.op_mem(true, &[0x3b], R14, R15, 40),
                "4d 3b b7 28 00 00 00",
                "cmp r14,QWORD PTR [r15+0x28]",
            ),
            (
                |a| a.op_mem(true, &[0x39], R12, R13, 16),
                "4d 39 a5 10 00 00 00",
                "cmp QWORD PTR [r13+0x10],r12",
            ),
            (
                |a| {
                    a.op_mem(true, &[0x83], 7, R13, 24);
                    a.byte(0xff);
                },
                "49 83 bd 18 00 00 00 ff",
                "cmp QWORD PTR [r13+0x18],0xffffffffffffffff",
            ),
            (
                |a| {
                    a.op_mem(true, &[0xc7], 0, R13, 32);
                    a.imm32(-1);
                },
                "49 c7 85 20 00 00 00 ff ff ff ff",
                "mov QWORD PTR [r13+0x20],0xffffffffffffffff",
            ),
            (
                |a| a.op_mem(false, &[0xff], 4, R14, 0),
                "41 ff a6 00 00 00 00",
                "jmp QWORD PTR [r14+0x0]",
            ),
            (
                |a| {
                    a.op_rr(true, &[0x0f, 0xba], 4, R9);
                    a.byte(5);
                },
                "49 0f ba e1 05",
                "bt r9,0x5",
            ),
            (
                |a| {
                    a.op_rr(true, &[0x0f, 0xba], 5, R9);
                    a.byte(40);
                },
                "49 0f ba e9 28",
                "bts r9,0x28",
            ),
            (
                |a| a.op_rr(false, &[0x89], RDX, R12),
                "41 89 d4",
                "mov r12d,edx",
            ),
            (
                |a| {
                    a.op_mem(false, &[0xf6], 0, R11, 0);
                    a.byte(2);
                },
                "41 f6 83 00 00 00 00 02",
                "test BYTE PTR [r11+0x0],0x2",
            ),
            (
                |a| {
                    a.op_rr(true, &[0xc7], 0, R9);
                    a.imm32(-1);
                },
                "49 c7 c1 ff ff ff ff",
                "mov r9,0xffffffffffffffff",
            ),
            (
                |a| a.alu_imm(false, 4, RAX, 0xff),
                "81 e0 ff 00 00 00",
                "and eax,0xff",
            ),
            (
                |a| {
                    a.op_rip(false, &[0xff], 2);
                },
                "ff 15 00 00 00 00",
                "call QWORD PTR [rip+0x0]",
            ),
            (
                |a| {
                    a.op_rip(true, &[0x8d], R11);
                },
                "4c 8d 1d 00 00 00 00",
                "lea r11,[rip+0x0]",
            ),
            (
                |a| a.alu_imm(true, 0, R14, 16),
                "49 83 c6 10",
                "add r14,0x10",
            ),
            (
                |a| a.op_mem(true, &[0x8d], R12, R11, 1),
                "4d 8d a3 01 00 00 00",
                "lea r12,[r11+0x1]",
            ),
            (|a| a.ret_instruction(), "c3", "ret"),
            (
                |a| a.alu_imm(false, 5, R10, 1),
                "41 83 ea 01",
                "sub r10d,0x1",
            ),
            (
                |a| a.op_mem(true, &[0x8d], RAX, RCX, 3),
                "48 8d 81 03 00 00 00",
                "lea rax,[rcx+0x3]",
            ),
            (|a| a.set_logged(0b11), "41 b9 03 00 00 00", "mov r9d,0x3"),
            (
                |a| a.set_logged(0x1_0000_0001),
                "49 b9 01 00 00 00 01 00 00 00",
                "movabs r9,0x100000001",
            ),
            (|a| a.log_slot(63), "49 0f ba e9 3f", "bts r9,0x3f"),
            (
                |a| a.op_mem(false, &[0x8d], R11, RAX, -0x41),
                "44 8d 98 bf ff ff ff",
                "lea r11d,[rax-0x41]",
            ),
            (
                |a| {
                    a.op_rr(false, &[0xf7], 0, R11);
                    a.imm32(!0x20);
                },
                "41 f7 c3 df ff ff ff",
                "test r11d,0xffffffdf",
            ),
        ];
        let print = std::env::var_os("TRUST_JS_REGEXP_X64_ENCODINGS").is_some();
        for (emit, expected, text) in cases {
            let mut a = Asm::new(
                Width::One,
                Helpers {
                    poll: 0,
                    grow: 0,
                    next_candidate: 0,
                },
            );
            emit(&mut a);
            let hex: Vec<String> = a.code.iter().map(|byte| format!("{byte:02x}")).collect();
            let hex = hex.join(" ");
            if print {
                eprintln!("{hex}\t{text}");
            } else {
                assert_eq!(hex, *expected, "{text}");
            }
        }
    }

    /// A LOGGED bit from 64 on, which `bt` and `bts` would take modulo 64, refuses the program
    /// (Decision 23).
    #[test]
    fn logged_bits_past_the_register_refuse_the_program() {
        let helpers = || Helpers {
            poll: 0,
            grow: 0,
            next_candidate: 0,
        };
        for slot in [63, 64, 300] {
            let mut a = Asm::new(Width::One, helpers());
            a.log_slot(slot);
            let mut b = Asm::new(Width::One, helpers());
            let label = b.label();
            b.branch_logged(slot, label);
            b.bind(label);
            assert_eq!((a.unencodable, b.unencodable), (slot >= 64, slot >= 64));
            if slot >= 64 {
                assert!(a.finish().is_none() && b.finish().is_none());
            }
        }
    }
}
