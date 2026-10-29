//! The native RegExp tier's AArch64 backend (see `regex_native.rs`): the [`Masm`] operations as
//! A64 instructions, with their own small assembler.
//!
//! Registers: x19 the subject, x20 its length, x21 the position, x22 the slots, x23 the backtrack
//! stack's top, x24 the root floor's logged slots, x25 the context, x26 the leading loop's run
//! end, x27 the choice count, x28 the poll counter (all callee-saved); w9 the element, x10 the
//! popped operand, x11 a loop's run start, x12 and x13 a scan's limit and stretch end, x14 and
//! x15 scratch (x15 caches the address of the position's element), x16 and x17 the
//! intra-procedure scratch. The code keeps a frame record (x29, x30), so frame-pointer unwinding
//! walks through it. Helper calls go through routines that keep x9-x15 and x30, so the
//! operations' registers survive them; AAPCS64 leaves the flags undefined across a call, so
//! results are compared after it, never assumed.

use super::{ctx, Cond, Data, Helpers, Label, Masm, Operand, Value, Width};
use super::{CANDIDATE_RUN, POLL_INTERRUPT_BIT, POLL_PERIOD, SCAN_STRETCH, STATUS_EXHAUSTED};

const SUBJ: u32 = 19;
const LEN: u32 = 20;
const POS: u32 = 21;
const SLOTS: u32 = 22;
const TOP: u32 = 23;
const LOGGED: u32 = 24;
const CTX: u32 = 25;
const LEAD: u32 = 26;
const CHOICES: u32 = 27;
const POLL: u32 = 28;
const E: u32 = 9;
const OP: u32 = 10;
const RS: u32 = 11;
const LIM: u32 = 12;
const CLIM: u32 = 13;
/// A loop's held table address (see [`Masm::hold_lut`]).
const HELD: u32 = 14;
const ADDR: u32 = 15;
const IP0: u32 = 16;
const IP1: u32 = 17;
const FP: u32 = 29;
const LR: u32 = 30;
const ZR: u32 = 31;
const SP: u32 = 31;

/// The condition field of a conditional branch or select.
fn cond_code(cond: Cond) -> u32 {
    match cond {
        Cond::Eq => 0,
        Cond::Ne => 1,
        Cond::Hs => 2,
        Cond::Lo => 3,
        Cond::Hi => 8,
        Cond::Ls => 9,
    }
}

// Encodings (Arm ARM, A64 base instructions), checked against llvm-mc in the tests below.

/// What an encoder returns for a field past its range (UDF #0, permanently undefined), as no
/// instruction it emits ever is: `Asm::emit` notes it and `finish` refuses the program, so the
/// pattern stays on the matcher instead of running a wrong word (Decision 23).
const UNENCODABLE: u32 = 0;

fn add_imm(sf: bool, rd: u32, rn: u32, imm12: u32, shift12: bool) -> u32 {
    if imm12 >= 1 << 12 {
        return UNENCODABLE;
    }
    (if sf { 0x9100_0000 } else { 0x1100_0000 })
        | u32::from(shift12) << 22
        | imm12 << 10
        | rn << 5
        | rd
}

fn sub_imm(sf: bool, rd: u32, rn: u32, imm12: u32) -> u32 {
    if imm12 >= 1 << 12 {
        return UNENCODABLE;
    }
    (if sf { 0xd100_0000 } else { 0x5100_0000 }) | imm12 << 10 | rn << 5 | rd
}

fn subs_imm(sf: bool, rd: u32, rn: u32, imm12: u32) -> u32 {
    if imm12 >= 1 << 12 {
        return UNENCODABLE;
    }
    (if sf { 0xf100_0000 } else { 0x7100_0000 }) | imm12 << 10 | rn << 5 | rd
}

fn adds_imm(sf: bool, rd: u32, rn: u32, imm12: u32) -> u32 {
    if imm12 >= 1 << 12 {
        return UNENCODABLE;
    }
    (if sf { 0xb100_0000 } else { 0x3100_0000 }) | imm12 << 10 | rn << 5 | rd
}

fn add_reg(rd: u32, rn: u32, rm: u32, lsl: u32) -> u32 {
    if lsl >= 64 {
        return UNENCODABLE;
    }
    0x8b00_0000 | rm << 16 | lsl << 10 | rn << 5 | rd
}

fn sub_reg(sf: bool, rd: u32, rn: u32, rm: u32) -> u32 {
    (if sf { 0xcb00_0000 } else { 0x4b00_0000 }) | rm << 16 | rn << 5 | rd
}

fn subs_reg(sf: bool, rd: u32, rn: u32, rm: u32) -> u32 {
    (if sf { 0xeb00_0000 } else { 0x6b00_0000 }) | rm << 16 | rn << 5 | rd
}

fn orr_reg(sf: bool, rd: u32, rn: u32, rm: u32, lsl: u32) -> u32 {
    if lsl >= if sf { 64 } else { 32 } {
        return UNENCODABLE;
    }
    (if sf { 0xaa00_0000 } else { 0x2a00_0000 }) | rm << 16 | lsl << 10 | rn << 5 | rd
}

fn movz(sf: bool, rd: u32, imm16: u32, hw: u32) -> u32 {
    if imm16 > 0xffff || hw >= if sf { 4 } else { 2 } {
        return UNENCODABLE;
    }
    (if sf { 0xd280_0000 } else { 0x5280_0000 }) | hw << 21 | imm16 << 5 | rd
}

fn movk(sf: bool, rd: u32, imm16: u32, hw: u32) -> u32 {
    if imm16 > 0xffff || hw >= if sf { 4 } else { 2 } {
        return UNENCODABLE;
    }
    (if sf { 0xf280_0000 } else { 0x7280_0000 }) | hw << 21 | imm16 << 5 | rd
}

fn movn(rd: u32, imm16: u32) -> u32 {
    if imm16 > 0xffff {
        return UNENCODABLE;
    }
    0x9280_0000 | imm16 << 5 | rd
}

fn csel(rd: u32, rn: u32, rm: u32, cond: Cond) -> u32 {
    0x9a80_0000 | rm << 16 | cond_code(cond) << 12 | rn << 5 | rd
}

fn lsr_imm(rd: u32, rn: u32, shift: u32) -> u32 {
    if shift >= 64 {
        return UNENCODABLE;
    }
    0xd340_0000 | shift << 16 | 63 << 10 | rn << 5 | rd
}

/// `lsr wd, wn, #shift`.
fn lsr_w(rd: u32, rn: u32, shift: u32) -> u32 {
    if shift >= 32 {
        return UNENCODABLE;
    }
    0x5300_0000 | shift << 16 | 31 << 10 | rn << 5 | rd
}

/// `csinc xd, xn, xm, cond`: `cinc xd, xn, cond'` and `cset xd, cond'` for the inverse `cond'`.
fn csinc(rd: u32, rn: u32, rm: u32, cond: Cond) -> u32 {
    0x9a80_0400 | rm << 16 | cond_code(cond) << 12 | rn << 5 | rd
}

/// `cmp wn, #imm12, lsl #12`.
fn cmp_imm_lsl12(rn: u32, imm12: u32) -> u32 {
    if imm12 >= 1 << 12 {
        return UNENCODABLE;
    }
    0x7140_0000 | imm12 << 10 | rn << 5 | ZR
}

/// `orr xd, xn, #(1 << bit)`: a logical immediate of one set bit.
fn orr_bit(rd: u32, rn: u32, bit: u32) -> u32 {
    if bit >= 64 {
        return UNENCODABLE;
    }
    0xb240_0000 | ((64 - bit) % 64) << 16 | rn << 5 | rd
}

/// `and xd, xn, #0xff`.
fn and_low_byte(rd: u32, rn: u32) -> u32 {
    0x9240_1c00 | rn << 5 | rd
}

fn ldr_imm(rt: u32, rn: u32, offset: usize) -> u32 {
    if !scaled_x(offset) {
        return UNENCODABLE;
    }
    0xf940_0000 | (offset as u32 / 8) << 10 | rn << 5 | rt
}

fn str_imm(rt: u32, rn: u32, offset: usize) -> u32 {
    if !scaled_x(offset) {
        return UNENCODABLE;
    }
    0xf900_0000 | (offset as u32 / 8) << 10 | rn << 5 | rt
}

/// Whether a 64-bit load's or store's scaled 12-bit offset field encodes `offset`.
fn scaled_x(offset: usize) -> bool {
    offset.is_multiple_of(8) && offset / 8 < 1 << 12
}

fn ldrb_imm(rt: u32, rn: u32, offset: u32) -> u32 {
    if offset >= 1 << 12 {
        return UNENCODABLE;
    }
    0x3940_0000 | offset << 10 | rn << 5 | rt
}

fn ldrh_imm(rt: u32, rn: u32, offset: u32) -> u32 {
    if !offset.is_multiple_of(2) || offset / 2 >= 1 << 12 {
        return UNENCODABLE;
    }
    0x7940_0000 | (offset / 2) << 10 | rn << 5 | rt
}

fn ldurb(rt: u32, rn: u32, offset: i32) -> u32 {
    if !(-256..256).contains(&offset) {
        return UNENCODABLE;
    }
    0x3840_0000 | ((offset as u32) & 0x1ff) << 12 | rn << 5 | rt
}

fn ldurh(rt: u32, rn: u32, offset: i32) -> u32 {
    if !(-256..256).contains(&offset) {
        return UNENCODABLE;
    }
    0x7840_0000 | ((offset as u32) & 0x1ff) << 12 | rn << 5 | rt
}

/// `ldrb wt, [xn, xm]`.
fn ldrb_reg(rt: u32, rn: u32, rm: u32) -> u32 {
    0x3860_6800 | rm << 16 | rn << 5 | rt
}

/// `ldrh wt, [xn, xm, lsl #1]`.
fn ldrh_reg_scaled(rt: u32, rn: u32, rm: u32) -> u32 {
    0x7860_7800 | rm << 16 | rn << 5 | rt
}

/// `ldrh wt, [xn, xm]`.
fn ldrh_reg(rt: u32, rn: u32, rm: u32) -> u32 {
    0x7860_6800 | rm << 16 | rn << 5 | rt
}

/// `ldrb wt, [xn], #1` or `ldrh wt, [xn], #2`: the element at xn, and xn past it.
fn ldr_elem_post(width: Width, rt: u32, rn: u32) -> u32 {
    match width {
        Width::One => 0x3840_0400 | 1 << 12 | rn << 5 | rt,
        Width::Two => 0x7840_0400 | 2 << 12 | rn << 5 | rt,
    }
}

#[derive(Clone, Copy)]
enum PairMode {
    Pre,
    Post,
    Offset,
}

fn stp(mode: PairMode, rt1: u32, rt2: u32, rn: u32, offset: i32) -> u32 {
    if offset % 8 != 0 || !(-512..512).contains(&offset) {
        return UNENCODABLE;
    }
    let base = match mode {
        PairMode::Pre => 0xa980_0000,
        PairMode::Post => 0xa880_0000,
        PairMode::Offset => 0xa900_0000,
    };
    base | (((offset / 8) as u32) & 0x7f) << 15 | rt2 << 10 | rn << 5 | rt1
}

fn ldp(mode: PairMode, rt1: u32, rt2: u32, rn: u32, offset: i32) -> u32 {
    match stp(mode, rt1, rt2, rn, offset) {
        UNENCODABLE => UNENCODABLE,
        word => word | 1 << 22,
    }
}

const RET: u32 = 0xd65f_03c0;

fn br(rn: u32) -> u32 {
    0xd61f_0000 | rn << 5
}

fn blr(rn: u32) -> u32 {
    0xd63f_0000 | rn << 5
}

/// A branch to resolve once its label is bound.
#[derive(Clone, Copy)]
enum Fixup {
    /// B, BL: imm26.
    Imm26,
    /// B.cond, CBZ, CBNZ: imm19.
    Imm19,
    /// TBZ, TBNZ: imm14.
    Imm14,
    /// ADR: imm21, in bytes.
    Adr,
}

/// A reference to the data that follows the code.
#[derive(Clone, Copy)]
enum DataRef {
    Adr,
    Literal,
}

/// Out-of-line code that sites request, emitted after the program.
enum Outline {
    /// The poll counter ran out: `bl poll; b back`.
    Poll { slow: Label, back: Label },
    /// The stack is full: `bl grow; b retry`.
    Grow { slow: Label, retry: Label },
    /// The candidate scan returned a status: none left (0), or an interruption.
    CandidateStatus { slow: Label, no_match: Label },
    /// A helper returned an interruption's status: leave with it.
    Status { slow: Label },
}

pub(super) struct Asm {
    width: Width,
    helpers: Helpers,
    code: Vec<u32>,
    labels: Vec<Option<usize>>,
    fixups: Vec<(usize, Label, Fixup)>,
    data: Vec<u8>,
    data_refs: Vec<(usize, usize, DataRef)>,
    luts: std::collections::HashMap<[u8; 256], Data>,
    literals: std::collections::HashMap<u64, usize>,
    outlines: Vec<Outline>,
    /// Shared routines: the poll's slow path, the stack's growth, a backreference's helper
    /// call and the exit with x0's status.
    poll_routine: Label,
    grow_routine: Label,
    backref_routine: Label,
    exit: Label,
    /// Whether x15 holds the address of the position's element.
    addr_valid: bool,
    /// The table whose address x14 holds.
    held: Option<Data>,
    /// An encoder met a field past its range (see [`UNENCODABLE`]): `finish` refuses the code.
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
            backref_routine: 0,
            exit: 0,
            addr_valid: false,
            held: None,
            unencodable: false,
        };
        asm.poll_routine = asm.label();
        asm.grow_routine = asm.label();
        asm.backref_routine = asm.label();
        asm.exit = asm.label();
        asm
    }

    fn emit(&mut self, word: u32) {
        self.unencodable |= word == UNENCODABLE;
        self.code.push(word);
    }

    fn branch(&mut self, base: u32, label: Label, fixup: Fixup) {
        self.fixups.push((self.code.len(), label, fixup));
        self.emit(base);
    }

    fn b(&mut self, label: Label) {
        self.branch(0x1400_0000, label, Fixup::Imm26);
    }

    fn bl(&mut self, label: Label) {
        self.branch(0x9400_0000, label, Fixup::Imm26);
    }

    fn b_cond(&mut self, cond: Cond, label: Label) {
        self.branch(0x5400_0000 | cond_code(cond), label, Fixup::Imm19);
    }

    fn cbz(&mut self, sf: bool, nonzero: bool, rt: u32, label: Label) {
        let base = match (sf, nonzero) {
            (true, false) => 0xb400_0000,
            (true, true) => 0xb500_0000,
            (false, false) => 0x3400_0000,
            (false, true) => 0x3500_0000,
        };
        self.branch(base | rt, label, Fixup::Imm19);
    }

    fn tbnz(&mut self, rt: u32, bit: u32, label: Label) {
        if bit >= 64 {
            return self.emit(UNENCODABLE);
        }
        self.branch(
            0x3700_0000 | (bit >> 5) << 31 | (bit & 31) << 19 | rt,
            label,
            Fixup::Imm14,
        );
    }

    fn adr(&mut self, rd: u32, label: Label) {
        self.branch(0x1000_0000 | rd, label, Fixup::Adr);
    }

    fn adr_data(&mut self, rd: u32, offset: usize) {
        self.data_refs.push((self.code.len(), offset, DataRef::Adr));
        self.emit(0x1000_0000 | rd);
    }

    /// `ldr xt, <literal>` of the 64-bit `value` in the data.
    fn ldr_literal(&mut self, rt: u32, value: u64) {
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
        self.data_refs
            .push((self.code.len(), offset, DataRef::Literal));
        self.emit(0x5800_0000 | rt);
    }

    fn mov(&mut self, rd: u32, rm: u32) {
        self.emit(orr_reg(true, rd, ZR, rm, 0));
    }

    /// rd = `value`.
    fn mov_imm(&mut self, sf: bool, rd: u32, value: u64) {
        if value == u64::MAX && sf {
            self.emit(movn(rd, 0));
            return;
        }
        let halves = if sf { 4 } else { 2 };
        self.emit(movz(sf, rd, (value & 0xffff) as u32, 0));
        for hw in 1..halves {
            let half = ((value >> (16 * hw)) & 0xffff) as u32;
            if half != 0 {
                self.emit(movk(sf, rd, half, hw));
            }
        }
    }

    /// xd = xn + `value`, through x17 when it does not fit an immediate.
    fn add_any(&mut self, rd: u32, rn: u32, value: u32) {
        if value < 1 << 12 {
            self.emit(add_imm(true, rd, rn, value, false));
        } else if value & 0xfff == 0 && value < 1 << 24 {
            self.emit(add_imm(true, rd, rn, value >> 12, true));
        } else {
            self.mov_imm(true, IP1, u64::from(value));
            self.emit(add_reg(rd, rn, IP1, 0));
        }
    }

    /// Compare register `rn` (`sf`: 64 bits) with `value`, through x17 when it does not fit an
    /// immediate.
    fn cmp_any(&mut self, sf: bool, rn: u32, value: u32) {
        if value < 1 << 12 {
            self.emit(subs_imm(sf, ZR, rn, value));
        } else {
            self.mov_imm(sf, IP1, u64::from(value));
            self.emit(subs_reg(sf, ZR, rn, IP1));
        }
    }

    fn slot_offset(slot: usize) -> usize {
        slot * 8
    }

    /// `ldr xt, [xn, #offset]` at any offset below 16 MB (Decision 23; MAX_SLOTS keeps the slots
    /// within the scaled form): past the form, through xt, which the load overwrites, the
    /// offset's high part added (`add xt, xn, #high, lsl #12`).
    fn ldr_x(&mut self, rt: u32, rn: u32, offset: usize) {
        if scaled_x(offset) || offset >= 1 << 24 || rt == ZR {
            return self.emit(ldr_imm(rt, rn, offset));
        }
        self.emit(add_imm(true, rt, rn, (offset >> 12) as u32, true));
        self.emit(ldr_imm(rt, rt, offset & 0xfff));
    }

    /// `str xt, [xn, #offset]` at any offset below 16 MB: past the scaled form, through x17,
    /// the intra-procedure scratch that `add_any` and `cmp_any` use, which no store's register
    /// or base is.
    fn str_x(&mut self, rt: u32, rn: u32, offset: usize) {
        if scaled_x(offset) || offset >= 1 << 24 || rt == IP1 || rn == IP1 {
            return self.emit(str_imm(rt, rn, offset));
        }
        self.emit(add_imm(true, IP1, rn, (offset >> 12) as u32, true));
        self.emit(str_imm(rt, IP1, offset & 0xfff));
    }

    /// Save x30 and x9-x15 for a call from a shared routine.
    fn save_scratch(&mut self) {
        self.emit(stp(PairMode::Pre, LR, E, SP, -64));
        self.emit(stp(PairMode::Offset, OP, RS, SP, 16));
        self.emit(stp(PairMode::Offset, LIM, CLIM, SP, 32));
        self.emit(stp(PairMode::Offset, 14, ADDR, SP, 48));
    }

    fn restore_scratch(&mut self) {
        self.emit(ldp(PairMode::Offset, 14, ADDR, SP, 48));
        self.emit(ldp(PairMode::Offset, LIM, CLIM, SP, 32));
        self.emit(ldp(PairMode::Offset, OP, RS, SP, 16));
        self.emit(ldp(PairMode::Post, LR, E, SP, 64));
    }

    fn emit_outlines(&mut self) {
        let outlines = std::mem::take(&mut self.outlines);
        for outline in outlines {
            match outline {
                Outline::Poll { slow, back } => {
                    self.bind(slow);
                    self.bl(self.poll_routine);
                    self.b(back);
                }
                Outline::Grow { slow, retry } => {
                    self.bind(slow);
                    self.bl(self.grow_routine);
                    self.b(retry);
                }
                Outline::CandidateStatus { slow, no_match } => {
                    self.bind(slow);
                    self.emit(and_low_byte(0, 0));
                    self.cbz(true, false, 0, no_match);
                    self.b(self.exit);
                }
                Outline::Status { slow } => {
                    self.bind(slow);
                    self.emit(and_low_byte(0, 0));
                    self.b(self.exit);
                }
            }
        }
    }

    fn emit_routines(&mut self) {
        // The poll counter's end: reload it, and while the word's interrupt bit is set, ask the
        // handle (`helper_poll`), leaving with its status if it reports an interruption.
        let poll = self.poll_routine;
        if self.referenced(poll) {
            let call = self.label();
            self.bind(poll);
            self.mov_imm(true, POLL, u64::from(POLL_PERIOD));
            self.emit(ldr_imm(IP0, CTX, ctx::POLL_WORD));
            self.emit(ldr_imm(IP0, IP0, 0));
            self.tbnz(IP0, POLL_INTERRUPT_BIT, call);
            self.emit(RET);
            self.bind(call);
            self.save_scratch();
            self.mov(0, CTX);
            self.ldr_literal(IP0, self.helpers.poll as u64);
            self.emit(blr(IP0));
            self.cbz(true, true, 0, self.exit);
            self.restore_scratch();
            self.emit(RET);
        }
        // The stack is full: grow it (`helper_grow`), which answers the new top, or null at the
        // state limit.
        let grow = self.grow_routine;
        if self.referenced(grow) {
            let exhausted = self.label();
            self.bind(grow);
            self.save_scratch();
            self.mov(0, CTX);
            self.mov(1, TOP);
            self.ldr_literal(IP0, self.helpers.grow as u64);
            self.emit(blr(IP0));
            self.cbz(true, false, 0, exhausted);
            self.mov(TOP, 0);
            self.restore_scratch();
            self.emit(RET);
            self.bind(exhausted);
            self.mov_imm(true, 0, STATUS_EXHAUSTED);
            self.b(self.exit);
        }
        // A backreference under `i`: the helper compares the x11 units at x10 with the subject
        // at the position, backward if x12 is nonzero (`helper_backref_fold`), answering in x0.
        let backref = self.backref_routine;
        if self.referenced(backref) {
            self.bind(backref);
            self.save_scratch();
            self.mov(0, CTX);
            self.mov(1, OP);
            self.mov(2, POS);
            self.mov(3, RS);
            self.mov(4, LIM);
            self.ldr_literal(IP0, self.helpers.backref_fold as u64);
            self.emit(blr(IP0));
            self.restore_scratch();
            self.emit(RET);
        }
        // Leave with x0's status: the frame record restores sp, whatever routines pushed.
        self.bind(self.exit);
        self.emit(add_imm(true, SP, FP, 0, false));
        self.emit(ldp(PairMode::Offset, 19, 20, SP, 16));
        self.emit(ldp(PairMode::Offset, 21, 22, SP, 32));
        self.emit(ldp(PairMode::Offset, 23, 24, SP, 48));
        self.emit(ldp(PairMode::Offset, 25, 26, SP, 64));
        self.emit(ldp(PairMode::Offset, 27, 28, SP, 80));
        self.emit(ldp(PairMode::Post, FP, LR, SP, 96));
        self.emit(RET);
    }

    fn referenced(&self, label: Label) -> bool {
        self.fixups.iter().any(|&(_, target, _)| target == label)
    }

    /// x15 = the address of the position's element.
    fn element_address(&mut self) {
        if !self.addr_valid {
            let shift = match self.width {
                Width::One => 0,
                Width::Two => 1,
            };
            self.emit(add_reg(ADDR, SUBJ, POS, shift));
            self.addr_valid = true;
        }
    }

    fn pos_changed(&mut self) {
        self.addr_valid = false;
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
        self.addr_valid = false;
    }

    fn jump(&mut self, label: Label) {
        self.b(label);
    }

    fn prologue(&mut self, sentinel: Label) {
        self.emit(stp(PairMode::Pre, FP, LR, SP, -96));
        self.emit(add_imm(true, FP, SP, 0, false));
        self.emit(stp(PairMode::Offset, 19, 20, SP, 16));
        self.emit(stp(PairMode::Offset, 21, 22, SP, 32));
        self.emit(stp(PairMode::Offset, 23, 24, SP, 48));
        self.emit(stp(PairMode::Offset, 25, 26, SP, 64));
        self.emit(stp(PairMode::Offset, 27, 28, SP, 80));
        self.mov(CTX, 0);
        self.emit(ldr_imm(SUBJ, CTX, ctx::SUBJECT));
        self.emit(ldr_imm(LEN, CTX, ctx::LEN));
        self.emit(ldr_imm(POS, CTX, ctx::START));
        self.emit(ldr_imm(SLOTS, CTX, ctx::SLOTS));
        self.emit(ldr_imm(TOP, CTX, ctx::STACK));
        self.adr(IP0, sentinel);
        self.emit(stp(PairMode::Post, IP0, ZR, TOP, 16));
        self.mov_imm(true, CHOICES, 0);
        self.mov_imm(true, POLL, u64::from(POLL_PERIOD));
        self.mov_imm(true, LEAD, 0);
        self.pos_changed();
    }

    fn ret(&mut self, status: u64) {
        self.mov_imm(true, 0, status);
        self.b(self.exit);
    }

    fn poll_tick(&mut self) {
        let slow = self.label();
        let back = self.label();
        self.emit(subs_imm(true, POLL, POLL, 1));
        self.b_cond(Cond::Eq, slow);
        let addr_valid = self.addr_valid;
        self.bind(back);
        self.addr_valid = addr_valid;
        self.outlines.push(Outline::Poll { slow, back });
    }

    fn poll_now(&mut self) {
        let addr_valid = self.addr_valid;
        self.bl(self.poll_routine);
        self.addr_valid = addr_valid;
    }

    fn set_logged(&mut self, mask: u64) {
        self.mov_imm(true, LOGGED, mask);
    }

    fn fill_unset(&mut self, slots: std::ops::Range<usize>) {
        self.emit(movn(IP0, 0));
        for slot in slots {
            self.str_x(IP0, SLOTS, Self::slot_offset(slot));
        }
    }

    fn rearm_sentinel(&mut self) {
        self.emit(add_imm(true, TOP, TOP, 16, false));
    }

    fn branch_room_lt(&mut self, k: u32, label: Label) {
        self.emit(sub_reg(true, IP0, LEN, POS));
        self.cmp_any(true, IP0, k);
        self.b_cond(Cond::Lo, label);
    }

    fn branch_room_back_lt(&mut self, k: u32, label: Label) {
        self.cmp_any(true, POS, k);
        self.b_cond(Cond::Lo, label);
    }

    fn branch_at_start(&mut self, at: bool, label: Label) {
        self.cbz(true, !at, POS, label);
    }

    fn branch_at_end(&mut self, at: bool, label: Label) {
        self.emit(subs_reg(true, ZR, POS, LEN));
        self.b_cond(if at { Cond::Hs } else { Cond::Ne }, label);
    }

    fn advance(&mut self, k: u32) {
        self.add_any(POS, POS, k);
        self.pos_changed();
    }

    fn retreat(&mut self, k: u32) {
        if k < 1 << 12 {
            self.emit(sub_imm(true, POS, POS, k));
        } else {
            // Through x17, as `add_any` advances.
            self.mov_imm(true, IP1, u64::from(k));
            self.emit(sub_reg(true, POS, POS, IP1));
        }
        self.pos_changed();
    }

    fn load_elem(&mut self, offset: i32) {
        let two = self.width == Width::Two;
        if offset == 0 {
            self.emit(if two {
                ldrh_reg_scaled(E, SUBJ, POS)
            } else {
                ldrb_reg(E, SUBJ, POS)
            });
            return;
        }
        self.element_address();
        let bytes = if two { offset * 2 } else { offset };
        if bytes > 0 && bytes < (1 << 12) * if two { 2 } else { 1 } {
            self.emit(if two {
                ldrh_imm(E, ADDR, bytes as u32)
            } else {
                ldrb_imm(E, ADDR, bytes as u32)
            });
        } else if (-256..256).contains(&bytes) {
            self.emit(if two {
                ldurh(E, ADDR, bytes)
            } else {
                ldurb(E, ADDR, bytes)
            });
        } else {
            self.mov_imm(true, IP1, bytes as i64 as u64);
            self.emit(if two {
                ldrh_reg(E, ADDR, IP1)
            } else {
                ldrb_reg(E, ADDR, IP1)
            });
        }
    }

    fn decode_elem(&mut self, back: bool) {
        debug_assert_eq!(self.width, Width::Two);
        // A lead surrogate's unit is 0x36 << 10 up, a trail's 0x37 << 10 up; a pair's element is
        // 0x10000 + (lead - 0xd800) << 10 + (trail - 0xdc00), (lead << 10) + trail - 0x35fdc00.
        let done = self.label();
        if back {
            // sub x16, x21, #1; ldrh w9, [x19, x16, lsl #1]: the unit before POS, a trail?
            self.emit(sub_imm(true, IP0, POS, 1));
            self.emit(ldrh_reg_scaled(E, SUBJ, IP0));
            self.emit(lsr_w(IP1, E, 10));
            self.emit(subs_imm(false, ZR, IP1, 0x37));
            self.b_cond(Cond::Ne, done);
            // A lead before it?
            self.cbz(true, false, IP0, done);
            self.emit(sub_imm(true, IP0, IP0, 1));
            self.emit(ldrh_reg_scaled(IP1, SUBJ, IP0));
            self.emit(lsr_w(IP0, IP1, 10));
            self.emit(subs_imm(false, ZR, IP0, 0x36));
            self.b_cond(Cond::Ne, done);
            // add x9, x9, x17, lsl #10
            self.emit(add_reg(E, E, IP1, 10));
        } else {
            // ldrh w9, [x19, x21, lsl #1]: the unit at POS, a lead?
            self.emit(ldrh_reg_scaled(E, SUBJ, POS));
            self.emit(lsr_w(IP0, E, 10));
            self.emit(subs_imm(false, ZR, IP0, 0x36));
            self.b_cond(Cond::Ne, done);
            // A trail after it?
            self.emit(add_imm(true, IP0, POS, 1, false));
            self.emit(subs_reg(true, ZR, IP0, LEN));
            self.b_cond(Cond::Hs, done);
            self.emit(ldrh_reg_scaled(IP1, SUBJ, IP0));
            self.emit(lsr_w(IP0, IP1, 10));
            self.emit(subs_imm(false, ZR, IP0, 0x37));
            self.b_cond(Cond::Ne, done);
            // add x9, x17, x9, lsl #10
            self.emit(add_reg(E, IP1, E, 10));
        }
        self.mov_imm(true, IP0, 0x35f_dc00);
        self.emit(sub_reg(true, E, E, IP0));
        self.bind(done);
    }

    fn step_elem(&mut self, back: bool) {
        // Two units for an element past the BMP (E >= 0x10000), else one.
        self.emit(cmp_imm_lsl12(E, 0x10));
        if back {
            self.emit(csinc(IP0, ZR, ZR, Cond::Lo));
            self.emit(sub_reg(true, POS, POS, IP0));
            self.emit(sub_imm(true, POS, POS, 1));
        } else {
            self.emit(csinc(POS, POS, POS, Cond::Lo));
            self.emit(add_imm(true, POS, POS, 1, false));
        }
        self.pos_changed();
    }

    fn branch_elem(&mut self, cond: Cond, value: u32, label: Label) {
        self.cmp_any(false, E, value);
        self.b_cond(cond, label);
    }

    fn branch_elem_in(&mut self, lo: u32, hi: u32, inside: bool, label: Label) {
        debug_assert!(lo <= hi);
        if lo < 1 << 12 {
            self.emit(sub_imm(false, IP0, E, lo));
        } else {
            self.mov_imm(false, IP1, u64::from(lo));
            self.emit(sub_reg(false, IP0, E, IP1));
        }
        self.cmp_any(false, IP0, hi - lo);
        self.b_cond(if inside { Cond::Ls } else { Cond::Hi }, label);
    }

    fn branch_elem_pair(&mut self, lo: u32, bit: u32, member: bool, label: Label) {
        debug_assert!(bit.is_power_of_two() && lo & bit == 0);
        // orr x16, x9, #bit; cmp w16, #(lo | bit)
        self.emit(orr_bit(IP0, E, bit.trailing_zeros()));
        self.cmp_any(false, IP0, lo | bit);
        self.b_cond(if member { Cond::Eq } else { Cond::Ne }, label);
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
        if self.held == Some(lut) {
            self.emit(ldrb_reg(IP0, HELD, E));
        } else {
            self.adr_data(IP0, lut);
            self.emit(ldrb_reg(IP0, IP0, E));
        }
        self.cbz(false, member, IP0, label);
    }

    fn hold_lut(&mut self, lut: Data) {
        debug_assert!(self.held.is_none());
        self.adr_data(HELD, lut);
        self.held = Some(lut);
    }

    fn release_lut(&mut self) {
        self.held = None;
    }

    fn push(&mut self, stub: Label, operand: Operand) {
        let retry = self.label();
        let slow = self.label();
        let addr_valid = self.addr_valid;
        self.bind(retry);
        self.addr_valid = addr_valid;
        self.emit(ldr_imm(IP0, CTX, ctx::STACK_END));
        self.emit(subs_reg(true, ZR, TOP, IP0));
        self.b_cond(Cond::Hs, slow);
        self.outlines.push(Outline::Grow { slow, retry });
        self.adr(IP0, stub);
        let value = match operand {
            Operand::Pos => POS,
            Operand::Slot(slot) => {
                self.ldr_x(IP1, SLOTS, Self::slot_offset(slot));
                IP1
            }
            Operand::Pack(add) => {
                let high = if add == 0 {
                    RS
                } else {
                    self.add_any(IP1, RS, add);
                    IP1
                };
                self.emit(orr_reg(true, IP1, POS, high, 32));
                IP1
            }
            Operand::PosChoices => {
                self.emit(orr_reg(true, IP1, POS, CHOICES, 32));
                IP1
            }
        };
        self.emit(stp(PairMode::Post, IP0, value, TOP, 16));
    }

    fn pop_dispatch(&mut self) {
        self.emit(ldp(PairMode::Pre, IP0, OP, TOP, -16));
        self.emit(br(IP0));
    }

    fn choices(&mut self, delta: i32) {
        if delta > 0 {
            self.emit(add_imm(true, CHOICES, CHOICES, delta as u32, false));
        } else {
            self.emit(sub_imm(true, CHOICES, CHOICES, (-delta) as u32));
        }
    }

    fn branch_choices(&mut self, zero: bool, label: Label) {
        self.cbz(true, !zero, CHOICES, label);
    }

    fn branch_logged(&mut self, slot: usize, label: Label) {
        self.tbnz(LOGGED, slot as u32, label);
    }

    fn log_slot(&mut self, slot: usize) {
        self.emit(orr_bit(LOGGED, LOGGED, slot as u32));
    }

    fn store_slot(&mut self, slot: usize, value: Value) {
        let register = match value {
            Value::Pos => POS,
            Value::Op => OP,
            Value::Zero => ZR,
            Value::Unset => {
                self.emit(movn(IP0, 0));
                IP0
            }
            Value::Next => {
                self.ldr_x(IP0, SLOTS, Self::slot_offset(slot));
                self.emit(add_imm(true, IP0, IP0, 1, false));
                IP0
            }
        };
        self.str_x(register, SLOTS, Self::slot_offset(slot));
    }

    fn branch_slot_unset(&mut self, slot: usize, label: Label) {
        self.ldr_x(IP0, SLOTS, Self::slot_offset(slot));
        self.emit(adds_imm(true, ZR, IP0, 1));
        self.b_cond(Cond::Eq, label);
    }

    fn branch_slot_eq_pos(&mut self, slot: usize, label: Label) {
        self.ldr_x(IP0, SLOTS, Self::slot_offset(slot));
        self.emit(subs_reg(true, ZR, IP0, POS));
        self.b_cond(Cond::Eq, label);
    }

    fn branch_slot_lt(&mut self, slot: usize, value: u32, lt: bool, label: Label) {
        self.ldr_x(IP0, SLOTS, Self::slot_offset(slot));
        self.cmp_any(true, IP0, value);
        self.b_cond(if lt { Cond::Lo } else { Cond::Hs }, label);
    }

    fn mark_top(&mut self, slot: usize) {
        // An offset, as V8 keeps a lookaround's stack pointer: growing the stack moves it.
        self.emit(ldr_imm(IP0, CTX, ctx::STACK));
        self.emit(sub_reg(true, IP0, TOP, IP0));
        self.str_x(IP0, SLOTS, Self::slot_offset(slot));
    }

    fn cut_to(&mut self, slot: usize) {
        self.emit(ldr_imm(IP0, CTX, ctx::STACK));
        self.ldr_x(IP1, SLOTS, Self::slot_offset(slot));
        self.emit(add_reg(TOP, IP0, IP1, 0));
        self.emit(ldr_imm(OP, TOP, 8));
    }

    fn unpack_pos_choices(&mut self) {
        self.emit(orr_reg(false, POS, ZR, OP, 0));
        self.emit(lsr_imm(CHOICES, OP, 32));
        self.pos_changed();
    }

    fn backref(
        &mut self,
        slot: usize,
        icase: bool,
        back: bool,
        paired: bool,
        reversed: bool,
        fail: Label,
    ) {
        // The loop registers: no loop holds them across a backreference.
        debug_assert!(self.held.is_none());
        let done = self.label();
        // x10 = the capture's start, x11 = its end: unset (all ones), it matches empty.
        self.ldr_x(OP, SLOTS, Self::slot_offset(slot));
        self.ldr_x(RS, SLOTS, Self::slot_offset(slot + 1));
        self.emit(adds_imm(true, ZR, OP, 1));
        self.b_cond(Cond::Eq, done);
        self.emit(adds_imm(true, ZR, RS, 1));
        self.b_cond(Cond::Eq, done);
        if reversed {
            // A lookbehind captures backward, its start past its end: x10 = the span's low end,
            // x11 = its length; empty, it matches empty.
            self.emit(subs_reg(true, ZR, RS, OP));
            self.b_cond(Cond::Eq, done);
            self.emit(sub_reg(true, IP0, RS, OP));
            self.emit(sub_reg(true, IP1, OP, RS));
            self.emit(csel(OP, OP, RS, Cond::Hi));
            self.emit(csel(RS, IP0, IP1, Cond::Hi));
        } else {
            // x11 = its length (a capture made reading forward ends at or past its start):
            // empty, it matches empty.
            self.emit(subs_reg(true, RS, RS, OP));
            self.b_cond(Cond::Eq, done);
        }
        if icase {
            // The helper answers the position past the repetition plus one, 0 if it does not
            // repeat, or bit 63 and a status if the host interrupted a long compare; compared
            // after the call, as AAPCS64 leaves the flags undefined across it.
            let status = self.label();
            self.mov_imm(true, LIM, u64::from(back));
            self.bl(self.backref_routine);
            self.emit(subs_imm(true, ZR, 0, 0));
            self.branch(0x5400_0004, status, Fixup::Imm19); // b.mi
            self.b_cond(Cond::Eq, fail);
            self.outlines.push(Outline::Status { slow: status });
            self.emit(sub_imm(true, POS, 0, 1));
        } else {
            // The subject must have as many units left past the position (before it).
            if back {
                self.emit(subs_reg(true, ZR, POS, RS));
            } else {
                self.emit(sub_reg(true, IP0, LEN, POS));
                self.emit(subs_reg(true, ZR, IP0, RS));
            }
            self.b_cond(Cond::Lo, fail);
            // x12 = the capture's first unit, x13 = the subject's, and POS past the repetition
            // (before it; a failure backtracks, which sets it again). Stretches of at most
            // SCAN_STRETCH units, polled between: x15 counts the stretch's units, x11 those after
            // it.
            let shift = match self.width {
                Width::One => 0,
                Width::Two => 1,
            };
            let stretch = self.label();
            let unit = self.label();
            let end = self.label();
            self.emit(add_reg(LIM, SUBJ, OP, shift));
            if back {
                self.emit(sub_reg(true, POS, POS, RS));
                self.emit(add_reg(CLIM, SUBJ, POS, shift));
            } else {
                self.emit(add_reg(CLIM, SUBJ, POS, shift));
                self.emit(add_reg(POS, POS, RS, 0));
            }
            self.bind(stretch);
            self.mov_imm(true, IP1, u64::from(SCAN_STRETCH));
            self.emit(subs_reg(true, ZR, RS, IP1));
            self.emit(csel(ADDR, RS, IP1, Cond::Lo));
            self.emit(sub_reg(true, RS, RS, ADDR));
            self.bind(unit);
            self.emit(ldr_elem_post(self.width, E, LIM));
            self.emit(ldr_elem_post(self.width, HELD, CLIM));
            self.emit(subs_reg(false, ZR, E, HELD));
            self.b_cond(Cond::Ne, fail);
            self.emit(subs_imm(true, ADDR, ADDR, 1));
            self.b_cond(Cond::Ne, unit);
            self.cbz(true, false, RS, end);
            self.poll_now();
            self.b(stretch);
            self.bind(end);
            if paired {
                // The repetition must not end inside a surrogate pair (as the matcher's
                // `boundary` check and V8's `CheckNotInSurrogatePair` have it): no trail at POS
                // after a lead.
                self.cbz(true, false, POS, done);
                self.emit(subs_reg(true, ZR, POS, LEN));
                self.b_cond(Cond::Hs, done);
                self.emit(ldrh_reg_scaled(IP0, SUBJ, POS));
                self.emit(lsr_w(IP0, IP0, 10));
                self.emit(subs_imm(false, ZR, IP0, 0x37));
                self.b_cond(Cond::Ne, done);
                self.emit(sub_imm(true, IP1, POS, 1));
                self.emit(ldrh_reg_scaled(IP0, SUBJ, IP1));
                self.emit(lsr_w(IP0, IP0, 10));
                self.emit(subs_imm(false, ZR, IP0, 0x36));
                self.b_cond(Cond::Eq, fail);
            }
        }
        self.bind(done);
        self.pos_changed();
    }

    fn pos_from_op(&mut self) {
        self.mov(POS, OP);
        self.pos_changed();
    }

    fn unpack_op(&mut self) {
        self.emit(orr_reg(false, POS, ZR, OP, 0));
        self.emit(lsr_imm(RS, OP, 32));
        self.pos_changed();
    }

    fn run_start(&mut self) {
        self.mov(RS, POS);
    }

    fn set_limit(&mut self, max: Option<u32>) {
        match max {
            None => self.mov(LIM, LEN),
            Some(max) => {
                self.add_any(LIM, POS, max);
                self.emit(subs_reg(true, ZR, LIM, LEN));
                self.emit(csel(LIM, LIM, LEN, Cond::Lo));
            }
        }
    }

    fn set_limit_back(&mut self, max: Option<u32>) {
        match max {
            None => self.mov_imm(true, LIM, 0),
            Some(max) => {
                // subs x12, x21, max; csel x12, x12, xzr, hs
                self.mov_imm(true, IP1, u64::from(max));
                self.emit(subs_reg(true, LIM, POS, IP1));
                self.emit(csel(LIM, LIM, ZR, Cond::Hs));
            }
        }
    }

    fn set_stretch_limit(&mut self) {
        self.add_any(CLIM, POS, SCAN_STRETCH);
        self.emit(subs_reg(true, ZR, CLIM, LIM));
        self.emit(csel(CLIM, CLIM, LIM, Cond::Lo));
    }

    fn set_stretch_limit_back(&mut self) {
        self.mov_imm(true, IP1, u64::from(SCAN_STRETCH));
        self.emit(subs_reg(true, CLIM, POS, IP1));
        self.emit(csel(CLIM, CLIM, ZR, Cond::Hs));
        self.emit(subs_reg(true, ZR, CLIM, LIM));
        self.emit(csel(CLIM, CLIM, LIM, Cond::Hs));
    }

    fn set_count(&mut self, count: u32) {
        self.mov_imm(true, LIM, u64::from(count));
    }

    fn branch_count(&mut self, zero: bool, label: Label) {
        self.cbz(true, !zero, LIM, label);
    }

    fn count_down(&mut self) {
        self.emit(sub_imm(true, LIM, LIM, 1));
    }

    fn set_candidate_bound(&mut self, window: bool) {
        self.emit(add_imm(true, CLIM, POS, CANDIDATE_RUN, false));
        self.emit(subs_reg(true, ZR, CLIM, LEN));
        self.emit(csel(CLIM, CLIM, LEN, Cond::Lo));
        if window {
            self.emit(ldr_imm(IP0, CTX, ctx::WINDOW));
            self.emit(subs_reg(true, ZR, CLIM, IP0));
            self.emit(csel(CLIM, CLIM, IP0, Cond::Lo));
        }
    }

    fn branch_pos_stretch(&mut self, cond: Cond, label: Label) {
        self.emit(subs_reg(true, ZR, POS, CLIM));
        self.b_cond(cond, label);
    }

    fn branch_stretch_is_limit(&mut self, label: Label) {
        self.emit(subs_reg(true, ZR, CLIM, LIM));
        self.b_cond(Cond::Eq, label);
    }

    fn pos_to_limit(&mut self) {
        self.mov(POS, LIM);
        self.pos_changed();
    }

    fn branch_run(&mut self, cond: Cond, n: u32, back: bool, label: Label) {
        if back {
            self.emit(sub_reg(true, IP0, RS, POS));
        } else {
            self.emit(sub_reg(true, IP0, POS, RS));
        }
        self.cmp_any(true, IP0, n);
        self.b_cond(cond, label);
    }

    fn rs_add(&mut self, k: u32, back: bool) {
        if back {
            self.mov_imm(true, IP1, u64::from(k));
            self.emit(sub_reg(true, RS, RS, IP1));
        } else {
            self.add_any(RS, RS, k);
        }
    }

    fn set_rs(&mut self, value: u32) {
        self.mov_imm(true, RS, u64::from(value));
    }

    fn branch_rs(&mut self, cond: Cond, n: u32, label: Label) {
        self.cmp_any(true, RS, n);
        self.b_cond(cond, label);
    }

    fn branch_pos_rs(&mut self, cond: Cond, label: Label) {
        self.emit(subs_reg(true, ZR, POS, RS));
        self.b_cond(cond, label);
    }

    fn store_lead(&mut self) {
        self.mov(LEAD, POS);
    }

    fn branch_window(&mut self, label: Label) {
        self.emit(ldr_imm(IP0, CTX, ctx::WINDOW));
        self.emit(subs_reg(true, ZR, POS, IP0));
        self.b_cond(Cond::Hs, label);
    }

    fn next_candidate(&mut self, no_match: Label) {
        // Nothing but the callee-saved registers is live at a candidate. A status (bit 63) goes
        // to an outline after the program, past TBNZ's 32 KB: through the sign's B.MI.
        let slow = self.label();
        self.mov(0, CTX);
        self.mov(1, POS);
        self.ldr_literal(IP0, self.helpers.next_candidate as u64);
        self.emit(blr(IP0));
        self.emit(subs_imm(true, ZR, 0, 0));
        self.branch(0x5400_0004, slow, Fixup::Imm19); // b.mi
        self.mov(POS, 0);
        self.pos_changed();
        self.outlines
            .push(Outline::CandidateStatus { slow, no_match });
    }

    fn next_start(&mut self, lead: bool, paired: bool, no_match: Label) {
        self.emit(ldr_imm(IP0, SLOTS, 0));
        self.emit(subs_reg(true, ZR, IP0, LEN));
        self.b_cond(Cond::Hs, no_match);
        self.emit(add_imm(true, POS, IP0, 1, false));
        if paired {
            // Past a surrogate pair's two units: a lead at the start, a trail after it.
            let single = self.label();
            self.emit(ldrh_reg_scaled(IP1, SUBJ, IP0));
            self.emit(lsr_w(IP1, IP1, 10));
            self.emit(subs_imm(false, ZR, IP1, 0x36));
            self.b_cond(Cond::Ne, single);
            self.emit(subs_reg(true, ZR, POS, LEN));
            self.b_cond(Cond::Hs, single);
            self.emit(ldrh_reg_scaled(IP1, SUBJ, POS));
            self.emit(lsr_w(IP1, IP1, 10));
            self.emit(subs_imm(false, ZR, IP1, 0x37));
            self.b_cond(Cond::Ne, single);
            self.emit(add_imm(true, POS, POS, 1, false));
            self.bind(single);
        }
        if lead {
            self.emit(subs_reg(true, ZR, POS, LEAD));
            self.emit(csel(POS, POS, LEAD, Cond::Hs));
        }
        self.pos_changed();
    }

    fn finish(mut self) -> Option<Vec<u8>> {
        self.emit_outlines();
        self.emit_routines();
        if self.unencodable {
            return None;
        }
        let code_bytes = self.code.len() * 4;
        let data_start = code_bytes.next_multiple_of(8);
        for (at, label, fixup) in std::mem::take(&mut self.fixups) {
            let target = self.labels.get(label).copied().flatten()?;
            let delta = target as isize - at as isize;
            let patched = match fixup {
                Fixup::Imm26 => {
                    if !(-(1 << 25)..(1 << 25)).contains(&delta) {
                        return None;
                    }
                    (delta as u32) & 0x03ff_ffff
                }
                Fixup::Imm19 => {
                    if !(-(1 << 18)..(1 << 18)).contains(&delta) {
                        return None;
                    }
                    ((delta as u32) & 0x7ffff) << 5
                }
                Fixup::Imm14 => {
                    if !(-(1 << 13)..(1 << 13)).contains(&delta) {
                        return None;
                    }
                    ((delta as u32) & 0x3fff) << 5
                }
                Fixup::Adr => adr_field(delta * 4)?,
            };
            self.code[at] |= patched;
        }
        for (at, offset, kind) in std::mem::take(&mut self.data_refs) {
            let delta = (data_start + offset) as isize - (at * 4) as isize;
            self.code[at] |= match kind {
                DataRef::Adr => adr_field(delta)?,
                DataRef::Literal => {
                    if delta % 4 != 0 || !(-(1 << 20)..(1 << 20)).contains(&delta) {
                        return None;
                    }
                    (((delta / 4) as u32) & 0x7ffff) << 5
                }
            };
        }
        let mut bytes: Vec<u8> = self
            .code
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        bytes.resize(data_start, 0);
        bytes.extend_from_slice(&self.data);
        Some(bytes)
    }
}

/// ADR's immediate fields for a byte offset, if it reaches.
fn adr_field(delta: isize) -> Option<u32> {
    if !(-(1 << 20)..(1 << 20)).contains(&delta) {
        return None;
    }
    let imm = delta as u32;
    Some((imm & 3) << 29 | ((imm >> 2) & 0x7ffff) << 5)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The encoders against llvm-mc 23's (`llvm-mc -triple=aarch64 -show-encoding`).
    #[test]
    fn encodings_agree_with_llvm_mc() {
        let cases: &[(u32, u32, &str)] = &[
            (
                add_imm(true, 21, 21, 3, false),
                0x9100_0eb5,
                "add x21, x21, #3",
            ),
            (
                add_imm(true, 13, 21, 4, true),
                0x9140_12ad,
                "add x13, x21, #4, lsl #12",
            ),
            (
                add_imm(false, 9, 9, 1, false),
                0x1100_0529,
                "add w9, w9, #1",
            ),
            (sub_imm(true, 21, 21, 1), 0xd100_06b5, "sub x21, x21, #1"),
            (sub_imm(false, 14, 9, 48), 0x5100_c12e, "sub w14, w9, #48"),
            (subs_imm(true, 28, 28, 1), 0xf100_079c, "subs x28, x28, #1"),
            (subs_imm(true, ZR, 16, 9), 0xf100_261f, "cmp x16, #9"),
            (subs_imm(false, ZR, 9, 10), 0x7100_293f, "cmp w9, #10"),
            (adds_imm(true, ZR, 16, 1), 0xb100_061f, "cmn x16, #1"),
            (add_reg(16, 19, 21, 0), 0x8b15_0270, "add x16, x19, x21"),
            (
                add_reg(16, 19, 21, 1),
                0x8b15_0670,
                "add x16, x19, x21, lsl #1",
            ),
            (sub_reg(true, 16, 20, 21), 0xcb15_0290, "sub x16, x20, x21"),
            (subs_reg(true, ZR, 21, 20), 0xeb14_02bf, "cmp x21, x20"),
            (subs_reg(false, ZR, 9, 16), 0x6b10_013f, "cmp w9, w16"),
            (
                orr_reg(true, 17, 21, 17, 32),
                0xaa11_82b1,
                "orr x17, x21, x17, lsl #32",
            ),
            (orr_reg(true, 25, ZR, 0, 0), 0xaa00_03f9, "mov x25, x0"),
            (orr_reg(false, 21, ZR, 10, 0), 0x2a0a_03f5, "mov w21, w10"),
            (movz(true, 28, 16384, 0), 0xd288_001c, "mov x28, #16384"),
            (movz(false, 0, 5, 0), 0x5280_00a0, "mov w0, #5"),
            (
                movk(true, 16, 0x1234, 1),
                0xf2a2_4690,
                "movk x16, #0x1234, lsl #16",
            ),
            (movn(16, 0), 0x9280_0010, "mov x16, #-1"),
            (
                csel(12, 12, 20, Cond::Lo),
                0x9a94_318c,
                "csel x12, x12, x20, lo",
            ),
            (
                csel(21, 21, 26, Cond::Hs),
                0x9a9a_22b5,
                "csel x21, x21, x26, hs",
            ),
            (lsr_imm(11, 10, 32), 0xd360_fd4b, "lsr x11, x10, #32"),
            (orr_bit(24, 24, 4), 0xb27c_0318, "orr x24, x24, #0x10"),
            (orr_bit(24, 24, 63), 0xb241_0318, "orr x24, x24, #1 << 63"),
            (orr_bit(24, 24, 2), 0xb27e_0318, "orr x24, x24, #0x4"),
            (orr_bit(16, 9, 5), 0xb27b_0130, "orr x16, x9, #0x20"),
            // W5b's 3c: lookahead frames, counted loops' counts, backreferences.
            (
                ldr_elem_post(Width::One, 9, 12),
                0x3840_1589,
                "ldrb w9, [x12], #1",
            ),
            (
                ldr_elem_post(Width::Two, 14, 13),
                0x7840_25ae,
                "ldrh w14, [x13], #2",
            ),
            (subs_reg(false, ZR, 9, 14), 0x6b0e_013f, "cmp w9, w14"),
            (
                csel(15, 11, 17, Cond::Lo),
                0x9a91_316f,
                "csel x15, x11, x17, lo",
            ),
            (
                orr_reg(true, 17, 21, 27, 32),
                0xaa1b_82b1,
                "orr x17, x21, x27, lsl #32",
            ),
            (orr_reg(false, 21, ZR, 10, 0), 0x2a0a_03f5, "mov w21, w10"),
            (lsr_imm(27, 10, 32), 0xd360_fd5b, "lsr x27, x10, #32"),
            (str_imm(ZR, 22, 64), 0xf900_22df, "str xzr, [x22, #64]"),
            (add_reg(23, 16, 17, 0), 0x8b11_0217, "add x23, x16, x17"),
            (ldr_imm(10, 23, 8), 0xf940_06ea, "ldr x10, [x23, #8]"),
            (sub_reg(true, 16, 23, 16), 0xcb10_02f0, "sub x16, x23, x16"),
            (adds_imm(true, ZR, 10, 1), 0xb100_055f, "cmn x10, #1"),
            (
                add_reg(12, 19, 10, 1),
                0x8b0a_066c,
                "add x12, x19, x10, lsl #1",
            ),
            (
                add_imm(true, 16, 16, 1, false),
                0x9100_0610,
                "add x16, x16, #1",
            ),
            // W5b's 3d: surrogate pairs decoded and stepped, backward scans.
            (lsr_w(16, 9, 10), 0x530a_7d30, "lsr w16, w9, #10"),
            (lsr_w(17, 17, 10), 0x530a_7e31, "lsr w17, w17, #10"),
            (
                add_reg(9, 17, 9, 10),
                0x8b09_2a29,
                "add x9, x17, x9, lsl #10",
            ),
            (
                add_reg(9, 9, 17, 10),
                0x8b11_2929,
                "add x9, x9, x17, lsl #10",
            ),
            (
                csinc(21, 21, 21, Cond::Lo),
                0x9a95_36b5,
                "cinc x21, x21, hs",
            ),
            (csinc(16, ZR, ZR, Cond::Lo), 0x9a9f_37f0, "cset x16, hs"),
            (cmp_imm_lsl12(9, 0x10), 0x7140_413f, "cmp w9, #16, lsl #12"),
            (
                csel(12, 12, ZR, Cond::Hs),
                0x9a9f_218c,
                "csel x12, x12, xzr, hs",
            ),
            (
                csel(10, 10, 11, Cond::Hi),
                0x9a8b_814a,
                "csel x10, x10, x11, hi",
            ),
            (ldr_imm(19, 25, 8), 0xf940_0733, "ldr x19, [x25, #8]"),
            (str_imm(21, 22, 32), 0xf900_12d5, "str x21, [x22, #32]"),
            (ldrb_imm(9, 16, 5), 0x3940_1609, "ldrb w9, [x16, #5]"),
            (ldrh_imm(9, 16, 6), 0x7940_0e09, "ldrh w9, [x16, #6]"),
            (ldurb(9, 16, -1), 0x385f_f209, "ldurb w9, [x16, #-1]"),
            (ldurh(9, 16, -2), 0x785f_e209, "ldurh w9, [x16, #-2]"),
            (ldrb_reg(9, 19, 21), 0x3875_6a69, "ldrb w9, [x19, x21]"),
            (
                ldrh_reg_scaled(9, 19, 21),
                0x7875_7a69,
                "ldrh w9, [x19, x21, lsl #1]",
            ),
            (ldrb_reg(16, 16, 9), 0x3869_6a10, "ldrb w16, [x16, x9]"),
            (
                stp(PairMode::Pre, 29, 30, SP, -96),
                0xa9ba_7bfd,
                "stp x29, x30, [sp, #-96]!",
            ),
            (
                stp(PairMode::Offset, 19, 20, SP, 16),
                0xa901_53f3,
                "stp x19, x20, [sp, #16]",
            ),
            (
                ldp(PairMode::Offset, 19, 20, SP, 16),
                0xa941_53f3,
                "ldp x19, x20, [sp, #16]",
            ),
            (
                ldp(PairMode::Post, 29, 30, SP, 96),
                0xa8c6_7bfd,
                "ldp x29, x30, [sp], #96",
            ),
            (
                stp(PairMode::Post, 16, 17, 23, 16),
                0xa881_46f0,
                "stp x16, x17, [x23], #16",
            ),
            (
                ldp(PairMode::Pre, 16, 10, 23, -16),
                0xa9ff_2af0,
                "ldp x16, x10, [x23, #-16]!",
            ),
            (
                stp(PairMode::Pre, 30, 9, SP, -64),
                0xa9bc_27fe,
                "stp x30, x9, [sp, #-64]!",
            ),
            (
                ldp(PairMode::Post, 30, 9, SP, 64),
                0xa8c4_27fe,
                "ldp x30, x9, [sp], #64",
            ),
            (br(16), 0xd61f_0200, "br x16"),
            (blr(16), 0xd63f_0200, "blr x16"),
            (RET, 0xd65f_03c0, "ret"),
            (and_low_byte(0, 0), 0x9240_1c00, "and x0, x0, #0xff"),
        ];
        for &(ours, theirs, text) in cases {
            assert_eq!(ours, theirs, "{text}: {ours:08x} against {theirs:08x}");
        }
    }

    /// Branch fields as the assembler patches them, against llvm-mc's.
    #[test]
    fn branches_agree_with_llvm_mc() {
        let mut a = Asm::new(
            Width::One,
            Helpers {
                poll: 0,
                grow: 0,
                next_candidate: 0,
                backref_fold: 0,
            },
        );
        let ahead = a.label();
        let behind = a.label();
        a.bind(behind);
        a.emit(0xd503_201f); // nop
        a.cbz(true, false, 21, ahead); // cbz x21, .+8 (to `ahead`, bound two words on)
        a.b_cond(Cond::Hs, behind); // b.hs .-8
        a.bind(ahead);
        a.tbnz(24, 40, behind); // tbnz x24, #40, .-12
        a.emit(RET);
        let words: Vec<u32> = a.code.clone();
        let mut fixed = words.clone();
        for (at, label, fixup) in &a.fixups {
            let delta = a.labels[*label].unwrap() as isize - *at as isize;
            fixed[*at] |= match fixup {
                Fixup::Imm19 => ((delta as u32) & 0x7ffff) << 5,
                Fixup::Imm14 => ((delta as u32) & 0x3fff) << 5,
                _ => unreachable!(),
            };
        }
        // llvm-mc: `cbz x21, .+8` = b4000055, `b.hs .-8` = 54ffffc2, `tbnz x24, #40, .-12` =
        // b747ffb8.
        assert_eq!(fixed[1], 0xb400_0055);
        assert_eq!(fixed[2], 0x54ff_ffc2);
        assert_eq!(fixed[3], 0xb747_ffb8);
    }

    /// Decision 23: a slot access or a retreat past its form legalizes (a load through its
    /// destination, a store and a retreat through x17); a field no sequence encodes makes
    /// `finish` refuse the program instead of emitting a wrong word.
    #[test]
    fn operands_past_their_forms_legalize_or_refuse_the_program() {
        let helpers = || Helpers {
            poll: 0,
            grow: 0,
            next_candidate: 0,
            backref_fold: 0,
        };
        let mut a = Asm::new(Width::One, helpers());
        a.ldr_x(IP0, SLOTS, 32760);
        a.ldr_x(IP0, SLOTS, 40000);
        a.str_x(POS, SLOTS, 40000);
        a.retreat(4095);
        a.retreat(5000);
        assert!(!a.unencodable);
        // llvm-mc: `ldr x16, [x22, #32760]`; `add x16, x22, #9, lsl #12; ldr x16, [x16, #3136]`;
        // `add x17, x22, #9, lsl #12; str x21, [x17, #3136]`; `sub x21, x21, #4095`;
        // `mov x17, #5000; sub x21, x21, x17`.
        assert_eq!(
            a.code,
            [
                0xf97f_fed0,
                0x9140_26d0,
                0xf946_2210,
                0x9140_26d1,
                0xf906_2235,
                0xd13f_feb5,
                0xd282_7111,
                0xcb11_02b5,
            ]
        );
        for emit in [
            (|a: &mut Asm| a.log_slot(64)) as fn(&mut Asm),
            |a| {
                let label = a.label();
                a.branch_logged(64, label);
                a.bind(label);
            },
            |a| a.emit(add_imm(true, 0, 0, 1 << 12, false)),
            |a| a.emit(ldp(PairMode::Offset, 0, 1, SP, 512)),
            |a| a.ldr_x(IP0, SLOTS, 1 << 24),
            |a| a.str_x(IP1, SLOTS, 40000),
        ] {
            let mut a = Asm::new(Width::One, helpers());
            emit(&mut a);
            assert!(a.unencodable);
            assert!(a.finish().is_none());
        }
    }
}
