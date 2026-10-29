//! The RegExp native tier (W5b's item 3): a pattern's [`Inst`] program compiled to machine
//! code, one template per instruction, for one subject width, in the shape of V8's
//! assemble-from-bytecode generator (`--regexp-assemble-from-bytecode`,
//! src/regexp/regexp-code-generator.cc): the generator below walks the program and emits through
//! a macro assembler ([`Masm`]) that each backend implements, as V8's per-architecture
//! `RegExpMacroAssembler`s do. [`Matcher::run`](super::Matcher::run) stays the reference, and
//! the matcher runs whatever a program uses that the tier does not.
//!
//! The tier takes the instruction set but `v`-mode class strings (characters, `.`, classes,
//! captures and their undo rules, alternation, `^`, `$` and `\b` with the flags resolved at
//! compile time, which inline modifier groups fix per instruction, greedy and lazy loops of one
//! element, atomic loops, group loops' progress marks and capture clears, counted loops,
//! lookaheads, lookbehinds reading backward, and backreferences) over one-byte and two-byte
//! subjects, under `i` once the program's characters and classes are closed over case. In
//! Unicode mode a two-byte subject's surrogate pair is one element, as the matcher's `Paired`
//! input steps it: the code decodes an element where a test may match a surrogate or one past
//! the BMP, and tests units where it cannot. The candidate loop runs
//! in the generated code, with the first-element filter's table tested in line; the prefilter's
//! literal and the filter's vector scans run in Rust ([`Regex::next_candidate`]), which the code
//! calls when its own test rules a start out.
//!
//! Machine state: an entry `fn(*mut NativeCtx) -> u64` returning a status ([`STATUS_MATCH`] and
//! the others). Callee-saved registers hold the subject, its length, the position, the slots
//! (captures, then marks and lookaheads' frames), the backtrack stack's top, the root floor's
//! logged slots, the context, the leading loop's run end, the choice count and the poll counter
//! (AArch64 x19-x28; x86-64 keeps some in caller-saved registers that every helper call saves,
//! and the context).
//! The backtrack stack is the program's [`MatchScratch`] storage taken as 16-byte records: the
//! address of a resume point or an undo stub, and an operand. A pop jumps to the record's address
//! with the operand in a register; undo stubs restore their slot and pop again, as V8's undo
//! code does (regexp-compiler.cc, `Trace::Flush`). A sentinel record at the stack's base ends a
//! failed attempt, as V8 pushes its fail label. A Rust helper grows the stack and enforces
//! [`MAX_BACKTRACK_STATE`](super::MAX_BACKTRACK_STATE), counted in records.
//!
//! Polling: a counter in a register, decremented at each backtrack, candidate, loop back-edge
//! and retry step, loads the interpreter's poll word (`interrupt::PollWord`) when it reaches
//! zero, as a loop's scan does after each stretch of elements, and calls the Rust helper, which
//! asks the interrupt handle, only while the word's interrupt bit is set: V8's generated code
//! compares the stack pointer with a limit that interrupt requests lower (`CheckPreemption`).
//!
//! `TRUST_JS_REGEXP_NATIVE_VERIFY=1` runs the matcher after every native execution and aborts on
//! any difference in result, capture spans or error kind (see [`verify`]).

// Targets without a backend (Windows' x64 ABI, wasm32) keep every pattern on the matcher.
#![cfg_attr(
    not(all(any(target_arch = "aarch64", target_arch = "x86_64"), unix)),
    allow(dead_code)
)]

use super::{
    backtrack_state_limit, class_elements, regexp_prof_enabled, regexp_prof_native_code,
    regexp_prof_native_exec, FirstFilter, Inst, MatchError, MatchScratch, ReInput, Regex, Rep,
    REGEXP_PROF,
};
use crate::interrupt::PollWord;
use std::rc::Rc;

/// Whether this target has a backend: AArch64 and x86-64 on Unix (Windows' x64 ABI stays on
/// the matcher).
pub(super) const BACKEND: bool = cfg!(all(
    any(target_arch = "aarch64", target_arch = "x86_64"),
    unix
));

/// A label in code being generated.
pub(super) type Label = usize;

/// A table in the data the generated code reads (see [`Masm::lut`]).
pub(super) type Data = usize;

/// The statuses the generated code returns.
pub(super) const STATUS_NO_MATCH: u64 = 0;
pub(super) const STATUS_MATCH: u64 = 1;
pub(super) const STATUS_CANCELLED: u64 = 2;
pub(super) const STATUS_NAVIGATION: u64 = 3;
pub(super) const STATUS_DEADLINE: u64 = 4;
pub(super) const STATUS_EXHAUSTED: u64 = 5;

/// The poll counter's period: events (backtracks, candidates, loop back-edges, retry steps,
/// loop stretches) between two loads of the poll word. Between two events the code runs at most
/// the program once through, so this bounds the time to an interrupt as the matcher's
/// `INTERRUPT_POLL_MASK` does.
pub(super) const POLL_PERIOD: u32 = 1 << 14;

/// Elements a loop's scan takes between two polls.
pub(super) const SCAN_STRETCH: u32 = 1 << 14;

/// Starts the candidate loop tests in line, by the first-element filter's table, before it
/// calls the Rust scan, whose vector loops take longer stretches.
pub(super) const CANDIDATE_RUN: u32 = 16;

/// A subject's positions fit in 32 bits below this, as the tier's packed record operands need:
/// longer subjects (engine strings never are) stay on the matcher.
const MAX_SUBJECT_LEN: usize = 1 << 31;

/// A program's code-size estimate (see [`code_units`]) past which it stays on the matcher, so
/// that every branch in its code reaches: AArch64's conditional branches and ADR take a
/// megabyte either way, and a unit is at most 100 instructions of code and stubs.
const MAX_CODE_UNITS: usize = 2048;

/// Capture and mark slots a program may have (AArch64's scaled 12-bit offsets reach 4,095).
const MAX_SLOTS: usize = 4000;

/// Ranges of two-byte elements above 255 a class may have.
const MAX_HIGH_RANGES: usize = 256;

/// The subject widths the tier compiles for: one byte a code unit, or two.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Width {
    One,
    Two,
}

impl Width {
    fn index(self) -> usize {
        self as usize
    }

    /// The greatest code unit of the width.
    fn max_unit(self) -> u32 {
        match self {
            Width::One => 0xff,
            Width::Two => 0xffff,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Width::One => "one-byte",
            Width::Two => "two-byte",
        }
    }
}

/// Unsigned conditions of the generated code's compares.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Cond {
    Eq,
    Ne,
    /// Lower (unsigned `<`).
    Lo,
    /// Higher or same (unsigned `>=`).
    Hs,
    /// Higher (unsigned `>`).
    Hi,
    /// Lower or same (unsigned `<=`).
    Ls,
}

/// What a pushed record carries besides its address.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Operand {
    /// The position.
    Pos,
    /// Slot `k`'s value (a capture or a mark).
    Slot(usize),
    /// The position in the low 32 bits and the run start plus `k` in the high ones.
    Pack(u32),
    /// The position in the low 32 bits and CHOICES in the high ones: a lookahead's frame.
    PosChoices,
}

/// What a slot store writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Value {
    Pos,
    Unset,
    /// The operand of the record just popped.
    Op,
    /// Zero: a counted loop's count, reset.
    Zero,
    /// The slot's value plus one: a counted loop's count, after an iteration.
    Next,
}

/// The macro assembler the generator emits through, as V8's `RegExpMacroAssembler`. Its
/// operations name the machine state by role:
/// - POS, the position; LEN, the subject's length; E, the element register, 32 bits wide, which
///   a helper call clobbers;
/// - SLOTS, the capture slots (unset is all ones), then the marks;
/// - TOP, the backtrack stack's top; OP, the operand of the record popped last;
/// - CHOICES, the choice records on the stack; LOGGED, the slots whose first write at the root
///   floor has an undo record this attempt (one bit a slot,
///   [`Matcher::logged`](super::Matcher::logged));
/// - RS, a loop's run start (or, packed in a record, its least end); LIM and CLIM, a loop scan's
///   limit and the end of its current stretch; LEAD, where the leading loop's run ended.
///
/// Every register but E keeps its value across the helper calls an operation makes.
pub(super) trait Masm {
    fn label(&mut self) -> Label;
    fn bind(&mut self, label: Label);
    fn jump(&mut self, label: Label);
    /// Enter: keep the caller's state, set the registers from the context, POS from its start,
    /// write the sentinel record (`sentinel`'s address) at the stack's base and point TOP past it,
    /// CHOICES to zero and the poll counter to its period.
    fn prologue(&mut self, sentinel: Label);
    /// Leave, returning `status`.
    fn ret(&mut self, status: u64);
    /// Count an event; at the period's end, poll the word and, while its interrupt bit is set,
    /// the handle, leaving with its status if it reports an interruption.
    fn poll_tick(&mut self);
    /// Poll the word now, as [`Self::poll_tick`] does at the period's end.
    fn poll_now(&mut self);
    fn set_logged(&mut self, mask: u64);
    fn fill_unset(&mut self, slots: std::ops::Range<usize>);
    /// TOP past the sentinel record again, which the failed attempt's last pop took.
    fn rearm_sentinel(&mut self);
    /// Branch if fewer than `k` units are left past POS.
    fn branch_room_lt(&mut self, k: u32, label: Label);
    /// Branch if fewer than `k` units are left before POS (POS below `k`).
    fn branch_room_back_lt(&mut self, k: u32, label: Label);
    /// Branch if POS is 0 (`at`), or is not.
    fn branch_at_start(&mut self, at: bool, label: Label);
    /// Branch if POS is LEN (`at`), or is not.
    fn branch_at_end(&mut self, at: bool, label: Label);
    /// Branch if no element is left past POS (before it, reading backward).
    fn branch_at_edge(&mut self, back: bool, label: Label) {
        if back {
            self.branch_at_start(true, label);
        } else {
            self.branch_at_end(true, label);
        }
    }
    fn advance(&mut self, k: u32);
    fn retreat(&mut self, k: u32);
    /// E = the unit at POS + `offset`, which the caller has checked is in the subject.
    fn load_elem(&mut self, offset: i32);
    /// E = the element at POS (`back`: the one ending at POS), which the caller has checked the
    /// subject holds, in a paired mode: a lead surrogate's unit and the trail surrogate's after it
    /// make one element past the BMP, as the matcher's `Paired::next_elem` (`prev_elem`) has it;
    /// any other unit is an element of its own. Uses the intra-procedure scratch.
    fn decode_elem(&mut self, back: bool);
    /// POS past the element in E (before it, `back`): two units for one past the BMP, else one.
    fn step_elem(&mut self, back: bool);
    /// Branch if E `cond` `value`.
    fn branch_elem(&mut self, cond: Cond, value: u32, label: Label);
    /// Branch if E is in `lo..=hi` (`inside`), or is not.
    fn branch_elem_in(&mut self, lo: u32, hi: u32, inside: bool, label: Label);
    /// Branch if E is `lo` or `lo + bit` (`member`), or is neither: `bit` is a power of two that
    /// `lo` lacks, so one compare under a mask decides (V8's `ShortCutEmitCharacterPair`).
    fn branch_elem_pair(&mut self, lo: u32, bit: u32, member: bool, label: Label);
    /// A byte table for [`Self::branch_lut`].
    fn lut(&mut self, table: &[u8; 256]) -> Data;
    /// Branch if `lut` holds a nonzero byte at E (`member`), or a zero one; E is below 256.
    fn branch_lut(&mut self, lut: Data, member: bool, label: Label);
    /// Keep `lut`'s address in a register for the [`Self::branch_lut`]s of a loop, emitted
    /// until [`Self::release_lut`]; the loop's code is entered only through what follows.
    fn hold_lut(&mut self, lut: Data);
    fn release_lut(&mut self);
    /// Push the record (`stub`'s address, `operand`), growing the stack if it is full.
    fn push(&mut self, stub: Label, operand: Operand);
    /// Pop a record into OP and jump to its address.
    fn pop_dispatch(&mut self);
    /// CHOICES += `delta` (1 or -1).
    fn choices(&mut self, delta: i32);
    /// Branch if CHOICES is zero (`zero`), or is not.
    fn branch_choices(&mut self, zero: bool, label: Label);
    /// Branch if `slot`'s bit of LOGGED is set (`slot` below 64).
    fn branch_logged(&mut self, slot: usize, label: Label);
    fn log_slot(&mut self, slot: usize);
    fn store_slot(&mut self, slot: usize, value: Value);
    fn branch_slot_unset(&mut self, slot: usize, label: Label);
    fn branch_slot_eq_pos(&mut self, slot: usize, label: Label);
    /// Branch if `slot`'s value (a count) is below `value` (`lt`), or is not; `value` is below
    /// 2^31.
    fn branch_slot_lt(&mut self, slot: usize, value: u32, lt: bool, label: Label);
    /// `slot` = the offset from the stack's base of the record pushed next: a lookahead's frame.
    fn mark_top(&mut self, slot: usize);
    /// Cut the stack back to the record whose offset `slot` keeps (see [`Self::mark_top`]),
    /// discarding it and every record above it, with OP = its operand.
    fn cut_to(&mut self, slot: usize);
    /// POS = OP's low 32 bits, CHOICES = its high ones (see [`Operand::PosChoices`]).
    fn unpack_pos_choices(&mut self);
    /// A backreference to the capture in `slot` and `slot + 1`, a span whichever way round (a
    /// lookbehind captures backward; unset or empty, it matches empty), as the matcher's
    /// `backref_end` compares it: unit by unit, or under `icase` through the helper, which
    /// compares case-insensitively (element by element in a paired mode). To `fail` if the
    /// subject at POS (`back`: before it) does not repeat the capture, or, in a paired mode
    /// (`paired`), if the repetition would end inside a surrogate pair, as V8's
    /// `CheckNotInSurrogatePair` checks; else POS past the repetition (before it). A capture made
    /// in a lookbehind's body (`reversed`) may have its start past its end. The compare polls
    /// between stretches, as a scan does.
    fn backref(
        &mut self,
        slot: usize,
        icase: bool,
        back: bool,
        paired: bool,
        reversed: bool,
        fail: Label,
    );
    /// POS = OP.
    fn pos_from_op(&mut self);
    /// POS = OP's low 32 bits, RS = its high ones.
    fn unpack_op(&mut self);
    /// RS = POS.
    fn run_start(&mut self);
    /// LIM = min(LEN, POS + `max`).
    fn set_limit(&mut self, max: Option<u32>);
    /// LIM = POS - `max`, or 0 if that is below it (`None`: 0): a backward scan's limit.
    fn set_limit_back(&mut self, max: Option<u32>);
    /// CLIM = min(LIM, POS + [`SCAN_STRETCH`]).
    fn set_stretch_limit(&mut self);
    /// CLIM = max(LIM, POS - [`SCAN_STRETCH`]): a backward scan's stretch.
    fn set_stretch_limit_back(&mut self);
    /// LIM = `count`: elements a decoded scan may still take.
    fn set_count(&mut self, count: u32);
    /// Branch if LIM is zero (`zero`), or is not.
    fn branch_count(&mut self, zero: bool, label: Label);
    /// LIM -= 1.
    fn count_down(&mut self);
    /// CLIM = min(LEN, POS + [`CANDIDATE_RUN`]), and with `window` the context's window.
    fn set_candidate_bound(&mut self, window: bool);
    /// Branch if POS `cond` CLIM.
    fn branch_pos_stretch(&mut self, cond: Cond, label: Label);
    fn branch_stretch_is_limit(&mut self, label: Label);
    /// POS = LIM.
    fn pos_to_limit(&mut self);
    /// Branch if POS - RS (`back`: RS - POS) `cond` `n`: the run's units.
    fn branch_run(&mut self, cond: Cond, n: u32, back: bool, label: Label);
    /// RS += `k` (`back`: RS -= `k`).
    fn rs_add(&mut self, k: u32, back: bool);
    /// RS = `value`: a decoded lazy loop's count of elements.
    fn set_rs(&mut self, value: u32);
    /// Branch if RS `cond` `n`.
    fn branch_rs(&mut self, cond: Cond, n: u32, label: Label);
    /// Branch if POS `cond` RS.
    fn branch_pos_rs(&mut self, cond: Cond, label: Label);
    /// LEAD = POS.
    fn store_lead(&mut self);
    /// Branch if POS is at or past the context's window (one past the last start in range of
    /// the prefilter literal's latest occurrence; 0 before the scan has looked for it).
    fn branch_window(&mut self, label: Label);
    /// POS = the next start from POS that the Rust scan finds ([`Regex::next_candidate`]); to
    /// `no_match` when none is left, or leave with the status of an interruption.
    fn next_candidate(&mut self, no_match: Label);
    /// After a failed attempt from the start in slot 0: to `no_match` if that start was LEN,
    /// else POS = the start plus one element (in a paired mode, two units for a surrogate
    /// pair's), and with `lead`, at least LEAD.
    fn next_start(&mut self, lead: bool, paired: bool, no_match: Label);
    /// The finished code and data, or `None` if a branch does not reach.
    fn finish(self) -> Option<Vec<u8>>;
}

/// The addresses of the helpers the generated code calls.
#[derive(Clone, Copy)]
pub(super) struct Helpers {
    pub(super) poll: usize,
    pub(super) grow: usize,
    pub(super) next_candidate: usize,
    pub(super) backref_fold: usize,
}

/// The context the generated code runs with (offsets in [`ctx`]).
#[repr(C)]
pub(super) struct NativeCtx {
    subject: *const u8,
    len: usize,
    start: usize,
    slots: *mut usize,
    stack: *mut [usize; 2],
    stack_end: *mut [usize; 2],
    poll_word: *const std::sync::atomic::AtomicU64,
    /// One past the last start in range of the prefilter literal's latest occurrence, 0 before
    /// the scan has looked for one; `usize::MAX` without a prefilter.
    window: usize,
    /// LEAD, on backends that keep it here.
    lead: usize,
    regex: *const Regex,
    control: *const crate::RuntimeInterrupt,
    stack_vec: *mut Vec<[usize; 2]>,
    /// Records the stack may hold.
    limit: usize,
    /// The stack pointer after the entry's pushes, on backends that restore it from here.
    saved_sp: usize,
}

/// The context's field offsets, for the backends.
pub(super) mod ctx {
    use super::NativeCtx;
    use std::mem::offset_of;
    pub(in crate::regex) const SUBJECT: usize = offset_of!(NativeCtx, subject);
    pub(in crate::regex) const LEN: usize = offset_of!(NativeCtx, len);
    pub(in crate::regex) const START: usize = offset_of!(NativeCtx, start);
    pub(in crate::regex) const SLOTS: usize = offset_of!(NativeCtx, slots);
    pub(in crate::regex) const STACK: usize = offset_of!(NativeCtx, stack);
    pub(in crate::regex) const STACK_END: usize = offset_of!(NativeCtx, stack_end);
    pub(in crate::regex) const POLL_WORD: usize = offset_of!(NativeCtx, poll_word);
    pub(in crate::regex) const WINDOW: usize = offset_of!(NativeCtx, window);
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(in crate::regex) const LEAD: usize = offset_of!(NativeCtx, lead);
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    pub(in crate::regex) const SAVED_SP: usize = offset_of!(NativeCtx, saved_sp);
}

/// The interrupt bit of the poll word, as the generated code tests it.
pub(super) const POLL_INTERRUPT_BIT: u32 = crate::interrupt::POLL_INTERRUPT.trailing_zeros();

/// Records the stack starts with (and keeps between executions).
const INITIAL_RECORDS: usize = 64;

/// A program compiled for one width.
pub(super) struct NativeCode {
    /// Ownership keeps the W^X mapping alive as long as the pattern.
    #[cfg_attr(not(test), allow(dead_code))]
    executable: crate::jit::ExecutableBuffer,
    width: Width,
    /// Made for Unicode mode's two-byte subjects, stepping surrogate pairs as one element: it
    /// runs on the matcher's `Paired` input only.
    paired: bool,
    /// The slots the code keeps: captures, marks, then its lookarounds' frames.
    slots: usize,
    entry: unsafe extern "C" fn(*mut NativeCtx) -> u64,
}

/// A pattern's native code, compiled for each width on its first use once the pattern is hot.
#[derive(Default)]
pub(super) struct NativeTiers {
    codes: [std::cell::OnceCell<Option<NativeCode>>; 2],
}

impl NativeTiers {
    fn get(&self, width: Width) -> Option<&NativeCode> {
        self.codes[width.index()].get()?.as_ref()
    }

    /// Whether a compilation for `width` has been tried, and whether it made code.
    #[cfg(test)]
    pub(super) fn state(&self, width: Width) -> (bool, bool) {
        let cell = self.codes[width.index()].get();
        (cell.is_some(), cell.is_some_and(Option::is_some))
    }
}

/// Executions over a subject this long tier up at once, as V8 compiles a pattern for a subject
/// of 1,000 units or more (`regexp.cc`, `RegExpImpl::IrregexpExec`'s tier-up).
pub(super) const TIER_UP_LENGTH: usize = 1000;

/// Count an execution of `re` over a subject of `width` and `len` units toward the tier, and
/// compile the pattern for `width` once it is hot: after the pattern's threshold of executions
/// (`TRUST_JS_REGEXP_TIER_UP_AT`, 64 by default; 0 keeps every pattern on the matcher), or at
/// once over a subject of [`TIER_UP_LENGTH`] units or more.
#[inline]
pub(super) fn note_execution(re: &Regex, width: Width, len: usize) {
    if re.tier_up_at == 0 {
        return;
    }
    let ticks = re.tier_ticks.get().saturating_add(1);
    re.tier_ticks.set(ticks);
    if ticks < re.tier_up_at && len < TIER_UP_LENGTH {
        return;
    }
    if re.native.codes[width.index()].get().is_none() {
        tier_up(re, width);
    }
}

/// Compile `re` for `width`, once: out of line, so that executions pay only the tests above.
#[cold]
#[inline(never)]
fn tier_up(re: &Regex, width: Width) {
    let code = compile(re, width);
    super::regexp_prof_tier_up(code.is_some());
    let _ = re.native.codes[width.index()].set(code);
}

/// The code to run for `input`, if the pattern has some for its width.
#[inline]
pub(super) fn code_for<'a, I: ReInput>(re: &'a Regex, input: &I) -> Option<&'a NativeCode> {
    let width = I::NATIVE_WIDTH?;
    let code = re.native.get(width)?;
    (input.len() < MAX_SUBJECT_LEN).then_some(code)
}

/// Why `re` stays on the matcher for `width`, or `None` if the tier takes it.
fn decline(re: &Regex, width: Width) -> Option<&'static str> {
    // Under `i`, and in the groups of `(?i:…)`, the program's characters, classes and loops
    // match with no case folding once they are closed over case (`Regex::case_closed`); `\b`
    // tests its word characters under `i` (Unicode mode's take in the long s and the Kelvin
    // sign), and a backreference compares case-insensitively through a helper.
    if !re.case_closed && (re.ignore_case || super::changes_flags(&re.prog)) {
        return Some("i flag");
    }
    if re.single.is_some() {
        return Some("single element");
    }
    if width == Width::One && (re.literal_ascii.is_some() || re.literal_fold.is_some()) {
        return Some("literal");
    }
    let mode = Mode::of(re, width);
    let prog = &re.prog;
    let n = prog.len();
    if code_units(prog, mode) > MAX_CODE_UNITS {
        return Some("code size");
    }
    if 2 * (re.ngroups + 1) + re.nmarks + look_count(prog) > MAX_SLOTS {
        return Some("slots");
    }
    if n < 3
        || !matches!(prog[0], Inst::Save(0))
        || !matches!(prog[n - 2], Inst::Save(1))
        || !matches!(prog[n - 1], Inst::Match)
    {
        return Some("program shape");
    }
    if !program_taken(prog, mode, true) {
        return Some("instruction");
    }
    None
}

/// The subjects a program's code is made for: their width, and in Unicode mode over two-byte
/// subjects (`paired`), elements of one unit or of a surrogate pair's two, as the matcher's
/// `Paired` input steps them.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Mode {
    width: Width,
    paired: bool,
}

impl Mode {
    fn of(re: &Regex, width: Width) -> Self {
        Mode {
            width,
            paired: re.unicode && width == Width::Two,
        }
    }

    /// The greatest element a subject holds.
    fn max_element(self) -> u32 {
        if self.paired {
            super::MAX_ELEMENT
        } else {
            self.width.max_unit()
        }
    }
}

/// Whether the tier takes every instruction of program `prog`: the pattern's (`root`), or a
/// lookaround's body. `v`-mode class strings stay on the matcher.
fn program_taken(prog: &[Inst], mode: Mode, root: bool) -> bool {
    let n = prog.len();
    prog.iter().enumerate().all(|(pc, inst)| match inst {
        Inst::Save(0) => root && pc == 0,
        Inst::Save(1) => root && pc == n - 2,
        Inst::Match => pc == n - 1,
        Inst::Class(class) => class_plan(class, mode).is_some(),
        Inst::Many { rep, .. } => match rep {
            Rep::Class(class) => class_plan(class, mode).is_some(),
            Rep::Char(_) | Rep::Any => true,
        },
        Inst::Look { prog: body, .. } | Inst::LookBehind { prog: body, .. } => {
            program_taken(body, mode, false)
        }
        // The code compares counts below 2^31 (a count past it takes as many iterations to
        // reach).
        Inst::CountedBranch { min, max, .. } => {
            *min < 1 << 31 && max.is_none_or(|max| max < 1 << 31)
        }
        Inst::Char(_)
        | Inst::Any
        | Inst::Save(_)
        | Inst::Split(..)
        | Inst::Jmp(_)
        | Inst::AssertStart
        | Inst::AssertEnd
        | Inst::WordBoundary(_)
        | Inst::SetMark(_)
        | Inst::CheckProgress(_)
        | Inst::ClearCaps(..)
        | Inst::Backref(_)
        | Inst::BackrefAlt(_)
        | Inst::PushFlags(..)
        | Inst::PopFlags
        | Inst::CounterReset(_)
        | Inst::CounterIncr(_)
        | Inst::CheckCountedProgress { .. } => true,
        Inst::StringSet(_) | Inst::StringSetRepeat { .. } => false,
    })
}

/// The lookarounds of `prog`, those of their bodies included: the frames the code may keep.
fn look_count(prog: &[Inst]) -> usize {
    prog.iter()
        .map(|inst| match inst {
            Inst::Look { prog: body, .. } | Inst::LookBehind { prog: body, .. } => {
                1 + look_count(body)
            }
            _ => 0,
        })
        .sum()
}

/// An upper bound on the size of `prog`'s code for `mode`, in units of 100 instructions: one
/// an instruction, but a class's (or a class loop's) table and ranges, which two-byte code tests
/// in line, each slot a capture clear restores, a lookaround's frame, body and captures, and each
/// group a backreference may compare.
fn code_units(prog: &[Inst], mode: Mode) -> usize {
    let class = |class: &super::CharClass| {
        3 + match mode.width {
            Width::One => 0,
            Width::Two => class_elements(class).map_or(0, |set| set.len()),
        }
    };
    prog.iter()
        .map(|inst| match inst {
            Inst::Class(c) => class(c),
            Inst::Many { rep, .. } => {
                2 + match rep {
                    Rep::Class(c) => 2 * class(c),
                    Rep::Char(_) | Rep::Any => 0,
                }
            }
            Inst::ClearCaps(lo, hi) => 1 + (hi + 1 - lo),
            Inst::Look { prog: body, .. } | Inst::LookBehind { prog: body, .. } => {
                2 + code_units(body, mode) + groups_inside(body).map_or(0, |(lo, hi)| hi + 1 - lo)
            }
            Inst::BackrefAlt(groups) => 1 + groups.len(),
            _ => 1,
        })
        .sum()
}

/// Compile `re` for `width`, or `None` if the tier does not take it or its code cannot be made.
fn compile(re: &Regex, width: Width) -> Option<NativeCode> {
    if !BACKEND || decline(re, width).is_some() {
        return None;
    }
    let started = regexp_prof_enabled().then(std::time::Instant::now);
    let plan = Plan::new(re, width);
    let bytes = assemble(&plan)?;
    let executable = crate::jit::ExecutableBuffer::from_bytes(&bytes)?;
    // The code's one entry is at byte zero, and the mapping is immutable once published. It
    // embeds no pointer: tables and helper addresses are in its own data.
    let entry = unsafe {
        std::mem::transmute::<*const u8, unsafe extern "C" fn(*mut NativeCtx) -> u64>(
            executable.as_ptr(),
        )
    };
    crate::jit::profiler::perf_map_code(executable.as_ptr(), executable.len(), || {
        format!("RegExp:/{}/{}", crate::jstr::to_text(&re.source), re.flags)
    });
    if let Some(pattern) = env_value!("TRUST_JS_REGEXP_CODEDUMP") {
        if pattern == "*" || re.source.contains(pattern) {
            codedump(re, width, &bytes, executable.as_ptr());
        }
    }
    regexp_prof_native_code(executable.len());
    if let Some(started) = started {
        let nanos = started.elapsed().as_nanos() as u64;
        REGEXP_PROF.with(|prof| prof.borrow_mut().native_compile_ns += nanos);
    }
    Some(NativeCode {
        executable,
        width,
        paired: plan.paired,
        slots: plan.slots,
        entry,
    })
}

/// Print a compiled program (`TRUST_JS_REGEXP_CODEDUMP=<source text>`, `*` for every pattern):
/// its words as `TRUST_JS_JIT_CODEDUMP` prints them on AArch64, its bytes on x86-64.
fn codedump(re: &Regex, width: Width, bytes: &[u8], base: *const u8) {
    eprintln!(
        "[regexp-codedump] /{}/{} {} base={:#x} {} bytes",
        re.source,
        re.flags,
        width.name(),
        base as usize,
        bytes.len()
    );
    if cfg!(target_arch = "aarch64") {
        for word in bytes.chunks(4) {
            let mut padded = [0u8; 4];
            padded[..word.len()].copy_from_slice(word);
            eprintln!("[regexp-codedump] {:08x}", u32::from_le_bytes(padded));
        }
    } else {
        for line in bytes.chunks(16) {
            let hex: Vec<String> = line.iter().map(|byte| format!("{byte:02x}")).collect();
            eprintln!("[regexp-codedump] {}", hex.join(" "));
        }
    }
}

/// Emit `plan` through the backend of this target.
#[allow(unreachable_code)]
fn assemble(plan: &Plan) -> Option<Vec<u8>> {
    let helpers = Helpers {
        poll: helper_poll as *const () as usize,
        grow: helper_grow as *const () as usize,
        next_candidate: match (plan.width, plan.paired) {
            (Width::One, _) => helper_next_candidate::<&[u8]> as *const () as usize,
            (Width::Two, false) => helper_next_candidate::<&[u16]> as *const () as usize,
            (Width::Two, true) => helper_next_candidate::<super::Paired> as *const () as usize,
        },
        backref_fold: match (plan.width, plan.paired) {
            (Width::One, _) => helper_backref_fold::<&[u8]> as *const () as usize,
            (Width::Two, false) => helper_backref_fold::<&[u16]> as *const () as usize,
            (Width::Two, true) => helper_backref_fold::<super::Paired> as *const () as usize,
        },
    };
    #[cfg(all(target_arch = "aarch64", unix))]
    {
        let mut masm = a64::Asm::new(plan.width, helpers);
        Generator::new(&mut masm, plan).emit();
        return masm.finish();
    }
    #[cfg(all(target_arch = "x86_64", unix))]
    {
        let mut masm = x64::Asm::new(plan.width, helpers);
        Generator::new(&mut masm, plan).emit();
        return masm.finish();
    }
    let _ = (plan, helpers);
    None
}

#[cfg(all(target_arch = "aarch64", unix))]
#[path = "regex_native_a64.rs"]
mod a64;

#[cfg(all(target_arch = "x86_64", unix))]
#[path = "regex_native_x64.rs"]
mod x64;

/// An element test the templates make.
#[derive(Clone)]
enum Test {
    Char(u32),
    /// `.`, with the dotall flag in effect where it is.
    Any {
        dotall: bool,
    },
    Class(Rc<ClassPlan>),
}

/// A class's tests: the byte table of its elements below 256, its ranges of elements above them
/// (two-byte units, or in a paired mode any element), and its elements themselves when a subject
/// can hold at most two.
struct ClassPlan {
    lut: [u8; 256],
    high: Vec<(u32, u32)>,
    few: Option<Vec<u32>>,
    /// The class holds no surrogate and no element past the BMP, so in a paired mode it matches
    /// only elements of one unit that is no surrogate: a unit test decides it.
    units: bool,
}

/// The elements of a class its code may compare one by one, as V8 compares a character's case-
/// independent letters (`TextNode::EmitAtomLetter`, regexp-compiler.cc): past two, the class's
/// tables take fewer instructions.
const MAX_FEW: usize = 2;

/// `class`'s tests for `mode`, or `None` if it has too many ranges to test in line.
fn class_plan(class: &super::CharClass, mode: Mode) -> Option<ClassPlan> {
    let mut lut = [0u8; 256];
    for (element, member) in lut.iter_mut().enumerate() {
        *member = u8::from(class.matches(element as u32, false, false));
    }
    let elements = class_elements(class);
    let max = mode.max_element();
    let high = match mode.width {
        Width::One => Vec::new(),
        Width::Two => elements
            .as_ref()?
            .iter()
            .copied()
            .filter(|&(_, hi)| hi >= 0x100)
            .filter(|&(lo, _)| lo <= max)
            .map(|(lo, hi)| (lo.max(0x100), hi.min(max)))
            .collect(),
    };
    let units = elements
        .as_ref()
        .is_some_and(|set| super::elements_disjoint(set, &[(0xd800, 0xdfff), (0x1_0000, max)]));
    // The elements a subject of the width can hold: a class closed over case keeps only those of
    // its letters the width has, as V8's `GetCaseIndependentLetters` drops the ones a one-byte
    // subject cannot hold.
    let few = elements.and_then(|set| {
        let mut few = Vec::new();
        for (lo, hi) in set {
            for element in lo..=hi.min(max) {
                if few.len() == MAX_FEW {
                    return None;
                }
                few.push(element);
            }
        }
        Some(few)
    });
    (high.len() <= MAX_HIGH_RANGES).then_some(ClassPlan {
        lut,
        high,
        few,
        units,
    })
}

/// How a test of E against a set of elements `few` (see [`ClassPlan::few`]) is made, if not by
/// the class's tables: none can match; one is a compare; two that differ in one bit are one
/// compare under a mask (V8's `ShortCutEmitCharacterPair`), the case pairs of `i` among them;
/// two others, on two-byte subjects, where the tables need a range test first, a compare each.
/// One-byte code tests two others by the table, in fewer instructions.
fn few_test(few: &[u32], width: Width) -> Option<FewTest> {
    match *few {
        [] => Some(FewTest::None),
        [element] => Some(FewTest::One(element)),
        [lo, hi] if (lo ^ hi).is_power_of_two() && lo & (lo ^ hi) == 0 => {
            Some(FewTest::Pair { lo, bit: lo ^ hi })
        }
        [_, _] if width == Width::Two => Some(FewTest::Each),
        _ => None,
    }
}

/// See [`few_test`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FewTest {
    None,
    One(u32),
    Pair { lo: u32, bit: u32 },
    Each,
}

/// How a test of `class` is made: by its elements (see [`few_test`]), or by its tables. A loop
/// that holds the table (`held`) tests one-byte elements by it in two instructions, fewer than a
/// pair's three.
enum ClassTest<'c> {
    Few(FewTest, &'c [u32]),
    Table,
}

fn class_test(class: &ClassPlan, width: Width, held: bool) -> ClassTest<'_> {
    let Some(few) = class.few.as_deref() else {
        return ClassTest::Table;
    };
    match few_test(few, width) {
        Some(FewTest::Pair { .. } | FewTest::Each) if held && width == Width::One => {
            ClassTest::Table
        }
        Some(test) => ClassTest::Few(test, few),
        None => ClassTest::Table,
    }
}

/// The flags in effect at an instruction: the pattern's, or those of the innermost inline
/// modifier group around it. A group's `PushFlags` and `PopFlags` enclose its body in the
/// program's order, and no jump crosses them, so the flags are fixed per instruction at compile
/// time; a lookahead's body begins with the flags of its `Look`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Flags {
    icase: bool,
    multiline: bool,
    dotall: bool,
}

/// A program the generator emits: the pattern's, or a lookaround's body.
struct Program<'a> {
    prog: &'a [Inst],
    /// The pcs a branch or a resumed record reaches, which begin a run of templates.
    targets: Vec<bool>,
    /// The flags in effect at each pc.
    flags: Vec<Flags>,
    /// A lookbehind's body, compiled from the reversed pattern: it reads the subject backward,
    /// each element ending where the position is, as the matcher's `back` direction does.
    back: bool,
}

impl<'a> Program<'a> {
    fn new(prog: &'a [Inst], entry: Flags, back: bool) -> Self {
        let mut targets = vec![false; prog.len() + 1];
        let mut flags = Vec::with_capacity(prog.len());
        let mut scopes = vec![entry];
        for (pc, inst) in prog.iter().enumerate() {
            let outer = *scopes.last().expect("the program's own flags");
            flags.push(outer);
            match inst {
                Inst::Split(a, b) => {
                    targets[*a] = true;
                    targets[*b] = true;
                }
                Inst::Jmp(target) => targets[*target] = true,
                Inst::CountedBranch { body, exit, .. } => {
                    targets[*body] = true;
                    targets[*exit] = true;
                }
                // Loops resume after themselves, and a negative lookaround's failed body at the
                // instruction after it.
                Inst::Many { .. } | Inst::Look { .. } | Inst::LookBehind { .. } => {
                    targets[pc + 1] = true
                }
                Inst::PushFlags(icase, multiline, dotall) => scopes.push(Flags {
                    icase: icase.unwrap_or(outer.icase),
                    multiline: multiline.unwrap_or(outer.multiline),
                    dotall: dotall.unwrap_or(outer.dotall),
                }),
                Inst::PopFlags => {
                    scopes.pop();
                }
                _ => {}
            }
        }
        Program {
            prog,
            targets,
            flags,
            back,
        }
    }
}

/// A lookaround's plan.
struct LookPlan {
    /// Its body's program.
    body: usize,
    /// The units the body tests from the position on (before it, for a lookbehind), when it
    /// tests elements of fixed widths and does nothing else: the lookaround tests them in place,
    /// with no frame.
    simple: Option<u32>,
    /// The slot that keeps the offset of its frame record in the stack.
    frame: usize,
    /// The capture groups inside the body, `lo..=hi`: unset whenever the lookahead is entered
    /// (each iteration of a loop around it unsets them first), so that restoring them, as the
    /// matcher's snapshot does, is unsetting them, as V8's lookarounds clear their registers
    /// (`ActionNode::PositiveSubmatchSuccess`, `NegativeSubmatchSuccess`).
    groups: Option<(usize, usize)>,
}

/// Units a lookaround tests in place (see [`Plan::simple_body`]).
const MAX_SIMPLE_RUN: usize = 16;

/// How an element test reads the subject (see [`Plan::shape`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    /// Units at fixed offsets: one, or a character past the BMP's two in a paired mode.
    Units(u32),
    /// An element of one unit or of a surrogate pair's two, decoded where it is (a paired mode).
    Decoded,
}

/// The capture groups whose slots `prog` writes, `lo..=hi`, its lookarounds' included.
fn groups_inside(prog: &[Inst]) -> Option<(usize, usize)> {
    let mut span: Option<(usize, usize)> = None;
    let mut add = |lo: usize, hi: usize| {
        span = Some(span.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
    };
    for inst in prog {
        match inst {
            Inst::Save(slot) => add(slot / 2, slot / 2),
            Inst::ClearCaps(lo, hi) => add(*lo, *hi),
            Inst::Look { prog: body, .. } | Inst::LookBehind { prog: body, .. } => {
                if let Some((lo, hi)) = groups_inside(body) {
                    add(lo, hi);
                }
            }
            _ => {}
        }
    }
    span
}

/// What the generator needs to know of a pattern, for one width.
struct Plan<'a> {
    width: Width,
    /// Unicode mode over two-byte subjects: an element is a unit or a surrogate pair's two.
    paired: bool,
    /// Unicode mode: `\b` under `i` takes in the characters that fold into word characters.
    unicode: bool,
    /// The pattern's program (0) and its lookarounds' bodies.
    progs: Vec<Program<'a>>,
    /// The lookarounds, by program and pc.
    looks: std::collections::HashMap<(usize, usize), LookPlan>,
    /// The capture groups a lookbehind's body captures, reading backward: their start may be
    /// past their end.
    reversed: std::collections::HashSet<usize>,
    /// Capture slots (two per group, group 0's included); the marks follow them, then the
    /// lookaheads' frames.
    nslots: usize,
    /// Every slot the code keeps.
    slots: usize,
    /// More capture slots than LOGGED has bits: each attempt unsets them all instead.
    wide: bool,
    /// The scan tries one start (sticky, or anchored at the input's start).
    only: bool,
    /// The pattern's own multiline flag, which the first-element filter's line starts follow.
    multiline: bool,
    lead_run: bool,
    /// The pattern has a prefilter, whose window the candidate loop keeps.
    window: bool,
    /// The first-element filter's table, which the candidate loop tests in line.
    first: Option<[u8; 256]>,
    /// A match may also begin where `^` holds, whatever the element there.
    first_line_start: bool,
    /// Tests by class, made once per class (by address).
    classes: std::cell::RefCell<std::collections::HashMap<usize, Rc<ClassPlan>>>,
}

impl<'a> Plan<'a> {
    fn new(re: &'a Regex, width: Width) -> Self {
        let nslots = 2 * (re.ngroups + 1);
        let first = match (&re.first, &re.first_lut) {
            (FirstFilter::Atoms(_), Some(lut)) => {
                let mut table = [0u8; 256];
                for (slot, &member) in table.iter_mut().zip(lut.iter()) {
                    *slot = u8::from(member);
                }
                Some(table)
            }
            _ => None,
        };
        let mut plan = Plan {
            width,
            paired: Mode::of(re, width).paired,
            unicode: re.unicode,
            progs: Vec::new(),
            looks: Default::default(),
            reversed: Default::default(),
            nslots,
            slots: nslots + re.nmarks,
            wide: nslots > 64,
            only: re.sticky || matches!(re.first, FirstFilter::Anchored),
            multiline: re.multiline,
            lead_run: re.lead_run,
            window: re.prefilter.is_some(),
            first,
            first_line_start: re.first_line_start,
            classes: Default::default(),
        };
        let entry = Flags {
            icase: re.ignore_case,
            multiline: re.multiline,
            dotall: re.dotall,
        };
        plan.add_program(&re.prog, entry, false);
        plan
    }

    fn mode(&self) -> Mode {
        Mode {
            width: self.width,
            paired: self.paired,
        }
    }

    /// Add `prog`, entered with `entry`'s flags and reading backward if `back`, and its
    /// lookarounds' bodies, returning its index.
    fn add_program(&mut self, prog: &'a [Inst], entry: Flags, back: bool) -> usize {
        let id = self.progs.len();
        self.progs.push(Program::new(prog, entry, back));
        if back {
            for inst in prog {
                if let Inst::Save(slot) = inst {
                    self.reversed.insert(slot / 2);
                }
            }
        }
        for (pc, inst) in prog.iter().enumerate() {
            let (body, body_back) = match inst {
                Inst::Look { prog: body, .. } => (body, false),
                Inst::LookBehind { prog: body, .. } => (body, true),
                _ => continue,
            };
            let flags = self.progs[id].flags[pc];
            let body_id = self.add_program(&body[..], flags, body_back);
            let simple = self.simple_body(body_id);
            let frame = self.slots;
            if simple.is_none() {
                self.slots += 1;
            }
            self.looks.insert(
                (id, pc),
                LookPlan {
                    body: body_id,
                    simple,
                    frame,
                    groups: groups_inside(body),
                },
            );
        }
        id
    }

    /// The units a lookaround's body `id` tests in place when it tests only elements of fixed
    /// widths (a loop of one element with a fixed count is as many), or `None`.
    fn simple_body(&self, id: usize) -> Option<u32> {
        let program = &self.progs[id];
        let (last, elements) = program.prog.split_last()?;
        if !matches!(last, Inst::Match) {
            return None;
        }
        let mut units = 0u32;
        for (pc, inst) in elements.iter().enumerate() {
            let (test, count) = match inst {
                Inst::Char(_) | Inst::Any | Inst::Class(_) => (self.element_test(id, pc), 1),
                Inst::Many { rep, min, max, .. }
                    if Some(*min) == *max && *min <= MAX_SIMPLE_RUN =>
                {
                    (self.rep_test(rep, program.flags[pc]), *min as u32)
                }
                _ => return None,
            };
            let Shape::Units(width) = self.shape(&test) else {
                return None;
            };
            units += width * count;
        }
        (units <= MAX_SIMPLE_RUN as u32).then_some(units)
    }

    /// How `test` reads the subject: a unit, unless in a paired mode it may match a surrogate
    /// or an element past the BMP (a character past it is its pair's two units, fixed), which it
    /// decodes. A test that matches only BMP elements other than surrogates fails on a
    /// surrogate's unit, as on the pair it begins or ends, so its unit decides it, as V8's BMP
    /// alternative of a class tests units (`AddBmpCharacters`, regexp-compiler-tonode.cc).
    fn shape(&self, test: &Test) -> Shape {
        if !self.paired {
            return Shape::Units(1);
        }
        match test {
            Test::Char(c) if *c > 0xffff => Shape::Units(2),
            Test::Char(c) if (0xd800..=0xdfff).contains(c) => Shape::Decoded,
            Test::Char(_) => Shape::Units(1),
            Test::Any { .. } => Shape::Decoded,
            Test::Class(class) if class.units => Shape::Units(1),
            Test::Class(_) => Shape::Decoded,
        }
    }

    fn class(&self, class: &Rc<super::CharClass>) -> Rc<ClassPlan> {
        let key = Rc::as_ptr(class) as usize;
        self.classes
            .borrow_mut()
            .entry(key)
            .or_insert_with(|| {
                Rc::new(class_plan(class, self.mode()).expect("declined classes never compile"))
            })
            .clone()
    }

    fn rep_test(&self, rep: &Rep, flags: Flags) -> Test {
        match rep {
            Rep::Char(c) => Test::Char(*c),
            Rep::Any => Test::Any {
                dotall: flags.dotall,
            },
            Rep::Class(class) => Test::Class(self.class(class)),
        }
    }

    /// The element test of the element instruction at `pc` of program `id`.
    fn element_test(&self, id: usize, pc: usize) -> Test {
        let program = &self.progs[id];
        match &program.prog[pc] {
            Inst::Char(c) => Test::Char(*c),
            Inst::Any => Test::Any {
                dotall: program.flags[pc].dotall,
            },
            Inst::Class(class) => Test::Class(self.class(class)),
            _ => unreachable!("an element instruction"),
        }
    }

    /// The test the branch at `pc` of program `id` begins with, past capture and mark
    /// bookkeeping: what [`Matcher::branch_may_match`](super::Matcher::branch_may_match) tests
    /// at run time, resolved at compile time. `None` when the branch may match without one.
    fn first_test(&self, id: usize, mut pc: usize) -> Option<Test> {
        let program = &self.progs[id];
        for _ in 0..4 {
            match &program.prog[pc] {
                Inst::Char(_) | Inst::Any | Inst::Class(_) => {
                    return Some(self.element_test(id, pc))
                }
                Inst::Many { rep, min, .. } if *min > 0 => {
                    return Some(self.rep_test(rep, program.flags[pc]))
                }
                Inst::Save(_)
                | Inst::ClearCaps(..)
                | Inst::SetMark(_)
                | Inst::CounterReset(_)
                | Inst::CounterIncr(_) => pc += 1,
                Inst::Jmp(target) => pc = *target,
                _ => return None,
            }
        }
        None
    }
}

/// Stubs the templates refer to, emitted after the program.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Stub {
    /// Resume the branch at a program's pc with POS from the record.
    Alt(usize, usize),
    /// Restore a slot from the record, and backtrack on.
    Undo(usize),
    /// Retry the greedy loop at a program's pc with a shorter run.
    Greedy(usize, usize),
    /// Retry the lazy loop at a program's pc with a longer run.
    Lazy(usize, usize),
    /// The body of the lookahead at a program's pc failed: POS and CHOICES from its frame record.
    LookFail(usize, usize),
    /// Unset capture groups `lo..=hi`, those of a positive lookahead backtracked past, and
    /// backtrack on.
    Clear(usize, usize),
}

/// The template generator: one pass over each program, in order, a lookahead's body in line.
struct Generator<'p, 'a, M: Masm> {
    m: &'p mut M,
    plan: &'p Plan<'a>,
    /// Each program's pcs.
    pcs: Vec<Vec<Label>>,
    bt: Label,
    /// Where the pattern's `Match` goes.
    matched: Label,
    stubs: Vec<(Stub, Label)>,
    stub_labels: std::collections::HashMap<Stub, Label>,
    /// Lazy loops' extension code, emitted after the program: (program, pc, check label, extend
    /// label).
    lazy: Vec<(usize, usize, Label, Label)>,
    /// Each lazy loop's extension, which its retry stub jumps to.
    extend: std::collections::HashMap<(usize, usize), Label>,
    /// The class whose table a loop holds (see [`Masm::hold_lut`]).
    held: Option<*const ClassPlan>,
}

impl<'p, 'a, M: Masm> Generator<'p, 'a, M> {
    fn new(m: &'p mut M, plan: &'p Plan<'a>) -> Self {
        let pcs = plan
            .progs
            .iter()
            .map(|program| (0..=program.prog.len()).map(|_| m.label()).collect())
            .collect();
        let bt = m.label();
        let matched = m.label();
        Generator {
            m,
            plan,
            pcs,
            bt,
            matched,
            stubs: Vec::new(),
            stub_labels: Default::default(),
            lazy: Vec::new(),
            extend: Default::default(),
            held: None,
        }
    }

    fn stub(&mut self, stub: Stub) -> Label {
        if let Some(&label) = self.stub_labels.get(&stub) {
            return label;
        }
        let label = self.m.label();
        self.stub_labels.insert(stub, label);
        self.stubs.push((stub, label));
        label
    }

    fn emit(mut self) {
        let plan = self.plan;
        let sentinel = self.m.label();
        let matched = self.matched;
        let no_match = self.m.label();
        let attempt = self.m.label();
        let cand = self.m.label();
        self.m.prologue(sentinel);
        if !plan.only {
            // The candidate loop: a start the filter's table admits, inside the prefilter's
            // window, is tried at once; the Rust scan finds the others. Up to `CANDIDATE_RUN`
            // starts the table rules out are skipped in line first, unless a match may begin at
            // a line start the table does not show: outside multiline mode only 0 is one, where
            // a match may always begin.
            self.m.bind(cand);
            self.m.poll_tick();
            let scan = self.m.label();
            let mut scans = false;
            if plan.window {
                self.m.branch_window(scan);
                scans = true;
            }
            if let Some(table) = &plan.first {
                let lut = self.m.lut(table);
                if plan.first_line_start && !plan.multiline {
                    self.m.branch_at_start(true, attempt);
                }
                if plan.first_line_start && plan.multiline {
                    self.m.branch_at_end(true, scan);
                    self.m.load_elem(0);
                    if plan.width == Width::Two {
                        self.m.branch_elem(Cond::Hs, 0x100, scan);
                    }
                    self.m.branch_lut(lut, false, scan);
                    self.m.jump(attempt);
                } else {
                    let next = self.m.label();
                    self.m.hold_lut(lut);
                    self.m.set_candidate_bound(plan.window);
                    self.m.bind(next);
                    self.m.branch_pos_stretch(Cond::Hs, scan);
                    self.m.load_elem(0);
                    if plan.width == Width::Two {
                        self.m.branch_elem(Cond::Hs, 0x100, scan);
                    }
                    self.m.branch_lut(lut, true, attempt);
                    self.m.advance(1);
                    self.m.jump(next);
                    self.m.release_lut();
                }
                scans = true;
            } else if scans {
                self.m.jump(attempt);
            }
            if scans {
                self.m.bind(scan);
                self.m.next_candidate(no_match);
            }
        }
        self.m.bind(attempt);
        if plan.wide {
            self.m.fill_unset(2..plan.nslots);
            self.m.set_logged(u64::MAX);
        } else {
            self.m.set_logged(0b11);
        }
        self.emit_program(0);
        // Backtrack: pop a record and jump to it.
        self.m.bind(self.bt);
        self.m.poll_tick();
        self.m.pop_dispatch();
        // The sentinel ends a failed attempt.
        self.m.bind(sentinel);
        self.m.rearm_sentinel();
        if plan.only {
            self.m.jump(no_match);
        } else {
            self.m.next_start(plan.lead_run, plan.paired, no_match);
            self.m.jump(cand);
        }
        let mut emitted = 0;
        while emitted < self.stubs.len() || !self.lazy.is_empty() {
            if let Some((id, pc, check, extend)) = self.lazy.pop() {
                self.emit_lazy_extend(id, pc, check, extend);
                continue;
            }
            let (stub, label) = self.stubs[emitted];
            emitted += 1;
            self.m.bind(label);
            self.emit_stub(stub);
        }
        self.m.bind(matched);
        self.m.ret(STATUS_MATCH);
        self.m.bind(no_match);
        self.m.ret(STATUS_NO_MATCH);
    }

    /// Emit program `id`'s templates, in order.
    fn emit_program(&mut self, id: usize) {
        let mut pc = 0;
        while pc < self.plan.progs[id].prog.len() {
            pc = self.emit_at(id, pc);
        }
    }

    /// Emit the template of `pc` in program `id` (or of the run of element instructions it
    /// begins), returning the pc after it.
    fn emit_at(&mut self, id: usize, pc: usize) -> usize {
        let plan = self.plan;
        let program = &plan.progs[id];
        let flags = program.flags[pc];
        let bt = self.bt;
        self.m.bind(self.pcs[id][pc]);
        match &program.prog[pc] {
            Inst::Char(_) | Inst::Any | Inst::Class(_) => {
                let first = plan.element_test(id, pc);
                if plan.shape(&first) == Shape::Decoded {
                    self.emit_decoded(&first, program.back, bt);
                    return pc + 1;
                }
                // A straight run of elements of fixed widths checks the room once and tests each
                // at its offset.
                let mut tests = vec![first];
                let mut end = pc + 1;
                while end < program.prog.len()
                    && !program.targets[end]
                    && matches!(
                        program.prog[end],
                        Inst::Char(_) | Inst::Any | Inst::Class(_)
                    )
                {
                    let test = plan.element_test(id, end);
                    if plan.shape(&test) == Shape::Decoded {
                        break;
                    }
                    tests.push(test);
                    end += 1;
                }
                self.emit_run(&tests, program.back, bt);
                return end;
            }
            Inst::Save(slot) => {
                let slot = *slot;
                // Slot 0 is written first in every attempt, at the root floor, and slot 1 only
                // before the match: neither needs an undo record. In a lookahead's body CHOICES
                // counts its frame record, so every write there has one.
                if slot > 1 {
                    let record = self.m.label();
                    let store = self.m.label();
                    let undo = self.stub(Stub::Undo(slot));
                    if plan.wide {
                        self.m.branch_choices(true, store);
                    } else {
                        // Above the root floor every write has a record; at it, the first write
                        // of each slot (`Matcher::log_root_write`).
                        self.m.branch_choices(false, record);
                        self.m.branch_logged(slot, store);
                        self.m.log_slot(slot);
                        self.m.bind(record);
                    }
                    self.m.push(undo, Operand::Slot(slot));
                    self.m.bind(store);
                }
                self.m.store_slot(slot, Value::Pos);
            }
            Inst::Split(a, b) => self.emit_split(id, pc, *a, *b),
            Inst::Jmp(target) => {
                if *target <= pc {
                    self.m.poll_tick();
                }
                self.m.jump(self.pcs[id][*target]);
            }
            Inst::Match => {
                // A lookaround's body ends in its `Match`, which falls through to the
                // lookaround's success, emitted next (see `emit_look`).
                if id == 0 {
                    self.m.jump(self.matched);
                }
            }
            Inst::AssertStart => {
                if flags.multiline {
                    let ok = self.m.label();
                    self.m.branch_at_start(true, ok);
                    self.m.load_elem(-1);
                    self.emit_line_terminator(ok);
                    self.m.jump(bt);
                    self.m.bind(ok);
                } else {
                    self.m.branch_at_start(false, bt);
                }
            }
            Inst::AssertEnd => {
                if flags.multiline {
                    let ok = self.m.label();
                    self.m.branch_at_end(true, ok);
                    self.m.load_elem(0);
                    self.emit_line_terminator(ok);
                    self.m.jump(bt);
                    self.m.bind(ok);
                } else {
                    self.m.branch_at_end(false, bt);
                }
            }
            Inst::WordBoundary(want) => self.emit_word_boundary(*want, flags.icase && plan.unicode),
            Inst::Many {
                rep,
                min,
                max,
                greedy,
                atomic,
            } => {
                let test = plan.rep_test(rep, flags);
                // A count past any subject's length bounds nothing, or cannot be reached.
                let max = max.and_then(|max| u32::try_from(max).ok().filter(|&m| m < 1 << 31));
                let Some(min) = u32::try_from(*min).ok().filter(|&m| m < 1 << 31) else {
                    self.m.jump(bt);
                    return pc + 1;
                };
                // A loop of decoded elements counts them (a surrogate pair is one); a character
                // past the BMP repeats as such an element too.
                let decoded = plan.shape(&test) != Shape::Units(1);
                match (*greedy, decoded) {
                    (true, false) => self.emit_greedy(id, pc, &test, min, max, *atomic),
                    (true, true) => self.emit_greedy_decoded(id, pc, &test, min, max, *atomic),
                    (false, false) => self.emit_lazy(id, pc, &test, min, max),
                    (false, true) => self.emit_lazy_decoded(id, pc, &test, min, max),
                }
            }
            Inst::SetMark(mark) => self.emit_mark_write(plan.nslots + mark, Value::Pos),
            Inst::CheckProgress(mark) => {
                // An iteration that consumed nothing fails (RepeatMatcher's empty check).
                self.m.branch_slot_eq_pos(plan.nslots + mark, bt);
            }
            Inst::ClearCaps(lo, hi) => {
                for slot in 2 * lo..2 * hi + 2 {
                    let skip = self.m.label();
                    let clear = self.m.label();
                    let undo = self.stub(Stub::Undo(slot));
                    self.m.branch_slot_unset(slot, skip);
                    self.m.branch_choices(true, clear);
                    self.m.push(undo, Operand::Slot(slot));
                    self.m.bind(clear);
                    self.m.store_slot(slot, Value::Unset);
                    self.m.bind(skip);
                }
            }
            Inst::Look { negate, .. } | Inst::LookBehind { negate, .. } => {
                self.emit_look(id, pc, *negate)
            }
            Inst::Backref(group) => {
                // A reference to a group the pattern does not have matches empty, as the
                // matcher's does.
                if *group != 0 && 2 * group + 1 < plan.nslots {
                    let reversed = plan.reversed.contains(group);
                    self.m.backref(
                        2 * group,
                        flags.icase,
                        program.back,
                        plan.paired,
                        reversed,
                        bt,
                    );
                }
            }
            Inst::BackrefAlt(groups) => {
                // `\k<name>` of a name several groups share: through the one that captured.
                let done = self.m.label();
                for &group in groups.iter() {
                    if 2 * group + 1 >= plan.nslots {
                        continue;
                    }
                    let next = self.m.label();
                    self.m.branch_slot_unset(2 * group, next);
                    self.m.branch_slot_unset(2 * group + 1, next);
                    let reversed = plan.reversed.contains(&group);
                    self.m.backref(
                        2 * group,
                        flags.icase,
                        program.back,
                        plan.paired,
                        reversed,
                        bt,
                    );
                    self.m.jump(done);
                    self.m.bind(next);
                }
                self.m.bind(done);
            }
            // Inline modifier groups emit nothing: each instruction's flags are fixed.
            Inst::PushFlags(..) | Inst::PopFlags => {}
            Inst::CounterReset(counter) => self.emit_mark_write(plan.nslots + counter, Value::Zero),
            Inst::CounterIncr(counter) => self.emit_mark_write(plan.nslots + counter, Value::Next),
            Inst::CountedBranch {
                id: counter,
                min,
                max,
                greedy,
                body,
                exit,
            } => {
                // As the matcher: a mandatory iteration while the count is below `min`, the
                // exit once it reaches `max`, else a choice, greedy or lazy first.
                let slot = plan.nslots + counter;
                if *min > 0 {
                    self.m
                        .branch_slot_lt(slot, *min as u32, true, self.pcs[id][*body]);
                }
                if let Some(max) = max {
                    self.m
                        .branch_slot_lt(slot, *max as u32, false, self.pcs[id][*exit]);
                }
                let (first, second) = if *greedy {
                    (*body, *exit)
                } else {
                    (*exit, *body)
                };
                self.emit_split(id, pc, first, second);
            }
            Inst::CheckCountedProgress { mark, counter, min } => {
                // Only an optional iteration, one past `min`, fails when it consumed nothing.
                let skip = self.m.label();
                if *min > 0 {
                    self.m
                        .branch_slot_lt(plan.nslots + counter, *min as u32, true, skip);
                }
                self.m.branch_slot_eq_pos(plan.nslots + mark, bt);
                self.m.bind(skip);
            }
            Inst::StringSet(_) | Inst::StringSetRepeat { .. } => {
                unreachable!("declined instruction in a native program")
            }
        }
        pc + 1
    }

    /// A choice between the branches at `first` and `second` of program `id`, from `pc`: as
    /// `Matcher::branch`, a first branch that fails at once is skipped, and the second is
    /// recorded only if it may match here.
    fn emit_split(&mut self, id: usize, pc: usize, first: usize, second: usize) {
        let back = self.plan.progs[id].back;
        if let Some(test) = self.plan.first_test(id, first) {
            self.emit_may_match(&test, self.pcs[id][second], back);
        }
        let skip = self.m.label();
        if let Some(test) = self.plan.first_test(id, second) {
            self.emit_may_match(&test, skip, back);
        }
        let alt = self.stub(Stub::Alt(id, second));
        self.m.push(alt, Operand::Pos);
        self.m.choices(1);
        self.m.bind(skip);
        if first != pc + 1 {
            self.m.jump(self.pcs[id][first]);
        }
    }

    /// Write a mark or a loop's count (`value`), with an undo record above the root floor: a
    /// failure at it unwinds the whole attempt, and a path reads a mark only after writing it.
    fn emit_mark_write(&mut self, slot: usize, value: Value) {
        let store = self.m.label();
        let undo = self.stub(Stub::Undo(slot));
        self.m.branch_choices(true, store);
        self.m.push(undo, Operand::Slot(slot));
        self.m.bind(store);
        self.m.store_slot(slot, value);
    }

    /// A lookaround at `pc` of program `id`, as the matcher's `LookFrame` runs it: its body in
    /// line (a lookbehind's reading backward), above a frame record of the position and CHOICES,
    /// which counts it as a choice so that every write in the body has an undo record. The
    /// body's success cuts the stack back to the record, discarding the body's records (a
    /// lookaround is atomic), and restores the position; its failure pops the frame record. A
    /// positive lookaround's captures stay set, and are unset when the code backtracks past it;
    /// a negative one's are always unset after it. A body of elements of fixed widths alone is
    /// tested in place.
    fn emit_look(&mut self, id: usize, pc: usize, negate: bool) {
        let plan = self.plan;
        let look = &plan.looks[&(id, pc)];
        let bt = self.bt;
        if let Some(units) = look.simple {
            let body = &plan.progs[look.body];
            let fail = if negate { self.m.label() } else { bt };
            if units > 0 {
                self.emit_room(units, body.back, fail);
            }
            let mut offset = 0u32;
            for (bpc, inst) in body.prog[..body.prog.len() - 1].iter().enumerate() {
                let (test, repeat) = match inst {
                    Inst::Many { rep, min, .. } => (plan.rep_test(rep, body.flags[bpc]), *min),
                    _ => (plan.element_test(look.body, bpc), 1),
                };
                for _ in 0..repeat {
                    offset += self.emit_test_at(&test, offset, body.back, fail);
                }
            }
            if negate {
                self.m.jump(bt);
                self.m.bind(fail);
            }
            return;
        }
        let failed = self.stub(Stub::LookFail(id, pc));
        self.m.mark_top(look.frame);
        self.m.push(failed, Operand::PosChoices);
        self.m.choices(1);
        self.emit_program(look.body);
        // The body matched.
        self.m.cut_to(look.frame);
        self.m.unpack_pos_choices();
        if negate {
            if let Some((lo, hi)) = look.groups {
                self.m.fill_unset(2 * lo..2 * hi + 2);
            }
            self.m.jump(bt);
        } else if let Some((lo, hi)) = look.groups {
            // Its captures stay set, and are unset by a record when the code backtracks past
            // it. At the root floor the attempt's first success leaves the record, as the
            // matcher logs a root write once (`Matcher::log_root_write`): the lookahead's slots
            // are logged together, and nothing else logs them (its body runs above its frame
            // record). A wide program's attempt unsets every slot itself.
            let clear = self.stub(Stub::Clear(lo, hi));
            let record = self.m.label();
            let done = self.m.label();
            if id == 0 {
                if plan.wide {
                    self.m.branch_choices(true, done);
                } else {
                    self.m.branch_choices(false, record);
                    self.m.branch_logged(2 * lo, done);
                    for slot in 2 * lo..2 * hi + 2 {
                        self.m.log_slot(slot);
                    }
                }
            }
            self.m.bind(record);
            self.m.push(clear, Operand::Pos);
            self.m.bind(done);
        }
    }

    /// Test E with `test`, going to `fail` if it does not match.
    fn emit_test(&mut self, test: &Test, fail: Label) {
        match test {
            Test::Char(c) => {
                if *c > self.plan.mode().max_element() {
                    self.m.jump(fail);
                } else {
                    self.m.branch_elem(Cond::Ne, *c, fail);
                }
            }
            Test::Any { dotall } => {
                if !dotall {
                    self.m.branch_elem(Cond::Eq, 0x0a, fail);
                    self.m.branch_elem(Cond::Eq, 0x0d, fail);
                    if self.plan.width == Width::Two {
                        self.m.branch_elem_in(0x2028, 0x2029, true, fail);
                    }
                }
            }
            Test::Class(class) => {
                let width = self.plan.width;
                let held = self.held == Some(Rc::as_ptr(class));
                match class_test(class, width, held) {
                    ClassTest::Few(FewTest::None, _) => return self.m.jump(fail),
                    ClassTest::Few(FewTest::One(element), _) => {
                        return self.m.branch_elem(Cond::Ne, element, fail)
                    }
                    ClassTest::Few(FewTest::Pair { lo, bit }, _) => {
                        return self.m.branch_elem_pair(lo, bit, false, fail)
                    }
                    ClassTest::Few(FewTest::Each, few) => {
                        let ok = self.m.label();
                        let (last, rest) = few.split_last().expect("a set of elements");
                        for &element in rest {
                            self.m.branch_elem(Cond::Eq, element, ok);
                        }
                        self.m.branch_elem(Cond::Ne, *last, fail);
                        return self.m.bind(ok);
                    }
                    ClassTest::Table => {}
                }
                if width == Width::One {
                    let lut = self.m.lut(&class.lut);
                    self.m.branch_lut(lut, false, fail);
                    return;
                }
                let high = self.m.label();
                let ok = self.m.label();
                self.m.branch_elem(Cond::Hs, 0x100, high);
                let lut = self.m.lut(&class.lut);
                self.m.branch_lut(lut, false, fail);
                self.m.jump(ok);
                self.m.bind(high);
                self.emit_ranges(&class.high, ok, fail);
                self.m.bind(ok);
            }
        }
    }

    /// Go to `ok` if E is in one of the sorted, disjoint `ranges`, else to `fail`: linearly for a
    /// few, else by halves.
    fn emit_ranges(&mut self, ranges: &[(u32, u32)], ok: Label, fail: Label) {
        if ranges.len() <= 4 {
            for &(lo, hi) in ranges {
                self.m.branch_elem_in(lo, hi, true, ok);
            }
            self.m.jump(fail);
            return;
        }
        let mid = ranges.len() / 2;
        let lower = self.m.label();
        self.m.branch_elem(Cond::Lo, ranges[mid].0, lower);
        self.emit_ranges(&ranges[mid..], ok, fail);
        self.m.bind(lower);
        self.emit_ranges(&ranges[..mid], ok, fail);
    }

    /// Go to `fail` unless an element at POS (before it, reading backward) passes `test`.
    fn emit_may_match(&mut self, test: &Test, fail: Label, back: bool) {
        match self.plan.shape(test) {
            Shape::Units(1) => {
                if back {
                    self.m.branch_at_start(true, fail);
                    self.m.load_elem(-1);
                } else {
                    self.m.branch_at_end(true, fail);
                    self.m.load_elem(0);
                }
                self.emit_test(test, fail);
            }
            Shape::Units(units) => {
                self.emit_room(units, back, fail);
                self.emit_test_at(test, 0, back, fail);
            }
            Shape::Decoded => {
                self.m.branch_at_edge(back, fail);
                self.m.decode_elem(back);
                self.emit_test(test, fail);
            }
        }
    }

    /// Go to `fail` if fewer than `units` units are left past POS (before it, reading backward).
    fn emit_room(&mut self, units: u32, back: bool, fail: Label) {
        if back {
            self.m.branch_room_back_lt(units, fail);
        } else {
            self.m.branch_room_lt(units, fail);
        }
    }

    /// Test the element of fixed width `test` at `offset` units from POS (the element ending
    /// that far before it, reading backward), which the caller has checked the subject holds,
    /// returning its width: a unit, or a character past the BMP's two.
    fn emit_test_at(&mut self, test: &Test, offset: u32, back: bool, fail: Label) -> u32 {
        let at = |unit: u32| -> i32 {
            if back {
                -(unit as i32) - 1
            } else {
                unit as i32
            }
        };
        match (self.plan.shape(test), test) {
            (Shape::Units(2), Test::Char(c)) => {
                // A character past the BMP is its surrogate pair: lead, then trail.
                let lead = 0xd800 + ((c - 0x1_0000) >> 10);
                let trail = 0xdc00 + ((c - 0x1_0000) & 0x3ff);
                let (first, second) = if back {
                    (at(offset + 1), at(offset))
                } else {
                    (at(offset), at(offset + 1))
                };
                self.m.load_elem(first);
                self.m.branch_elem(Cond::Ne, lead, fail);
                self.m.load_elem(second);
                self.m.branch_elem(Cond::Ne, trail, fail);
                2
            }
            _ => {
                self.m.load_elem(at(offset));
                self.emit_test(test, fail);
                1
            }
        }
    }

    /// A straight run of elements of fixed widths: the room checked once, each tested at its
    /// offset, then the position past them all (before them, reading backward).
    fn emit_run(&mut self, tests: &[Test], back: bool, fail: Label) {
        let units: u32 = tests
            .iter()
            .map(|test| match self.plan.shape(test) {
                Shape::Units(units) => units,
                Shape::Decoded => unreachable!("a decoded element in a run"),
            })
            .sum();
        self.emit_room(units, back, fail);
        let mut offset = 0;
        for test in tests {
            offset += self.emit_test_at(test, offset, back, fail);
        }
        if back {
            self.m.retreat(units);
        } else {
            self.m.advance(units);
        }
    }

    /// A decoded element (a paired mode): the element at POS, a unit or a surrogate pair (the
    /// one ending at POS, reading backward), tested, then POS past it.
    fn emit_decoded(&mut self, test: &Test, back: bool, fail: Label) {
        self.m.branch_at_edge(back, fail);
        self.m.decode_elem(back);
        self.emit_test(test, fail);
        self.m.step_elem(back);
    }

    /// Go to `hit` if E is a line terminator.
    fn emit_line_terminator(&mut self, hit: Label) {
        self.m.branch_elem(Cond::Eq, 0x0a, hit);
        self.m.branch_elem(Cond::Eq, 0x0d, hit);
        if self.plan.width == Width::Two {
            self.m.branch_elem_in(0x2028, 0x2029, true, hit);
        }
    }

    /// Go to `hit` if E is a word character: `[0-9A-Za-z_]`, IsWordChar outside Unicode mode
    /// with `i` or without, and in Unicode mode under `i` (`unicode_icase`) also the long s and
    /// the Kelvin sign, which fold into it (WordCharacters, as `is_word_ic` has it).
    fn emit_word(&mut self, hit: Label, unicode_icase: bool) {
        let mut table = [0u8; 256];
        for (element, member) in table.iter_mut().enumerate() {
            *member = u8::from(super::is_word(element as u32));
        }
        let skip = self.m.label();
        if self.plan.width == Width::Two && unicode_icase {
            let low = self.m.label();
            self.m.branch_elem(Cond::Lo, 0x100, low);
            self.m.branch_elem(Cond::Eq, 0x17f, hit);
            self.m.branch_elem(Cond::Eq, 0x212a, hit);
            self.m.jump(skip);
            self.m.bind(low);
        } else if self.plan.width == Width::Two {
            self.m.branch_elem(Cond::Hs, 0x100, skip);
        }
        let lut = self.m.lut(&table);
        self.m.branch_lut(lut, true, hit);
        self.m.bind(skip);
    }

    /// `\b` (`want`) or `\B`: whether the units before and after POS differ in being word
    /// characters (a surrogate is none, as the matcher reads units here).
    fn emit_word_boundary(&mut self, want: bool, unicode_icase: bool) {
        let ok = self.m.label();
        let (boundary, none) = if want { (ok, self.bt) } else { (self.bt, ok) };
        let before_word = self.m.label();
        let before_not = self.m.label();
        self.m.branch_at_start(true, before_not);
        self.m.load_elem(-1);
        self.emit_word(before_word, unicode_icase);
        self.m.bind(before_not);
        self.m.branch_at_end(true, none);
        self.m.load_elem(0);
        self.emit_word(boundary, unicode_icase);
        self.m.jump(none);
        self.m.bind(before_word);
        self.m.branch_at_end(true, boundary);
        self.m.load_elem(0);
        self.emit_word(none, unicode_icase);
        self.m.jump(boundary);
        self.m.bind(ok);
    }

    /// Scan from POS while units pass `test`, at most `max` of them (`None`: to the subject's
    /// edge), leaving POS past the run (before it, reading backward), polling between stretches.
    fn emit_scan(&mut self, test: &Test, max: Option<u32>, back: bool) {
        if back {
            self.m.set_limit_back(max);
        } else {
            self.m.set_limit(max);
        }
        if matches!(test, Test::Any { dotall: true }) {
            self.m.pos_to_limit();
            return;
        }
        let stretch = self.m.label();
        let scan = self.m.label();
        let bound = self.m.label();
        let done = self.m.label();
        let held = match test {
            Test::Class(class)
                if matches!(class_test(class, self.plan.width, true), ClassTest::Table) =>
            {
                let lut = self.m.lut(&class.lut);
                self.m.hold_lut(lut);
                self.held = Some(Rc::as_ptr(class));
                true
            }
            _ => false,
        };
        self.m.bind(stretch);
        if back {
            // Downward: POS above CLIM, the stretch's end, while the run goes on.
            self.m.set_stretch_limit_back();
            self.m.branch_pos_stretch(Cond::Ls, bound);
            self.m.bind(scan);
            self.m.load_elem(-1);
            self.emit_test(test, done);
            self.m.retreat(1);
            self.m.branch_pos_stretch(Cond::Hi, scan);
        } else {
            self.m.set_stretch_limit();
            self.m.branch_pos_stretch(Cond::Hs, bound);
            self.m.bind(scan);
            self.m.load_elem(0);
            self.emit_test(test, done);
            self.m.advance(1);
            self.m.branch_pos_stretch(Cond::Lo, scan);
        }
        self.m.bind(bound);
        self.m.branch_stretch_is_limit(done);
        self.m.poll_now();
        self.m.jump(stretch);
        self.m.bind(done);
        if held {
            self.m.release_lut();
            self.held = None;
        }
    }

    /// Scan decoded elements (a paired mode) from POS while they pass `test`, at most `count`
    /// of them (`None`: to the subject's edge), leaving POS past the run (before it, reading
    /// backward) and LIM counting down the elements `count` left untaken. Each element counts
    /// as an event toward the poll.
    fn emit_elem_scan(&mut self, test: &Test, count: Option<u32>, back: bool) {
        let next = self.m.label();
        let done = self.m.label();
        if let Some(count) = count {
            self.m.set_count(count);
        }
        self.m.bind(next);
        if count.is_some() {
            self.m.branch_count(true, done);
        }
        self.m.branch_at_edge(back, done);
        self.m.decode_elem(back);
        self.emit_test(test, done);
        self.m.step_elem(back);
        if count.is_some() {
            self.m.count_down();
        }
        self.m.poll_tick();
        self.m.jump(next);
        self.m.bind(done);
    }

    /// A greedy loop at `pc` of program `id`: the longest run, and a record to retry shorter ones
    /// unless they are futile ([`Matcher`](super::Matcher)'s `Backtrack::Greedy`).
    fn emit_greedy(
        &mut self,
        id: usize,
        pc: usize,
        test: &Test,
        min: u32,
        max: Option<u32>,
        atomic: bool,
    ) {
        let bt = self.bt;
        let back = self.plan.progs[id].back;
        self.m.run_start();
        self.emit_scan(test, max, back);
        // The program's leading loop records where its run ended (see `Regex::lead_run`).
        if self.plan.lead_run && id == 0 && pc == 1 {
            self.m.store_lead();
        }
        if min > 0 {
            self.m.branch_run(Cond::Lo, min, back, bt);
        }
        if !atomic {
            let skip = self.m.label();
            self.m.branch_run(Cond::Ls, min, back, skip);
            let retry = self.stub(Stub::Greedy(id, pc));
            if back {
                // The record keeps the least end: `min` units before the run's start.
                self.m.rs_add(min, true);
                self.m.push(retry, Operand::Pack(0));
            } else {
                self.m.push(retry, Operand::Pack(min));
            }
            self.m.choices(1);
            self.m.bind(skip);
        }
    }

    /// A greedy loop of decoded elements (a paired mode): `min` elements, where its least end
    /// is, then the longest run past it, and a record to retry shorter ones, one element (one
    /// or two units) at a time, unless they are futile.
    fn emit_greedy_decoded(
        &mut self,
        id: usize,
        pc: usize,
        test: &Test,
        min: u32,
        max: Option<u32>,
        atomic: bool,
    ) {
        let bt = self.bt;
        let back = self.plan.progs[id].back;
        if min > 0 {
            self.emit_elem_scan(test, Some(min), back);
            self.m.branch_count(false, bt);
        }
        self.m.run_start();
        self.emit_elem_scan(test, max.map(|max| max - min), back);
        if self.plan.lead_run && id == 0 && pc == 1 {
            self.m.store_lead();
        }
        if !atomic {
            let skip = self.m.label();
            self.m.branch_pos_rs(Cond::Eq, skip);
            let retry = self.stub(Stub::Greedy(id, pc));
            self.m.push(retry, Operand::Pack(0));
            self.m.choices(1);
            self.m.bind(skip);
        }
    }

    /// A lazy loop at `pc` of program `id`: the shortest run whose continuation may match, and a
    /// record to retry longer ones ([`Matcher::lazy_next`](super::Matcher::lazy_next)).
    fn emit_lazy(&mut self, id: usize, pc: usize, test: &Test, min: u32, max: Option<u32>) {
        let bt = self.bt;
        let back = self.plan.progs[id].back;
        self.m.run_start();
        if min > 0 {
            self.emit_scan(test, Some(min), back);
            self.m.branch_run(Cond::Lo, min, back, bt);
        }
        let check = self.m.label();
        let extend = self.m.label();
        self.m.bind(check);
        if let Some(next) = self.plan.first_test(id, pc + 1) {
            self.emit_may_match(&next, extend, back);
        }
        let skip = self.m.label();
        if let Some(max) = max {
            self.m.branch_run(Cond::Hs, max, back, skip);
        }
        let retry = self.stub(Stub::Lazy(id, pc));
        self.m.push(retry, Operand::Pack(0));
        self.m.choices(1);
        self.m.bind(skip);
        self.lazy.push((id, pc, check, extend));
        self.extend.insert((id, pc), extend);
    }

    /// A lazy loop of decoded elements (a paired mode): `min` elements, then the shortest run
    /// whose continuation may match, RS counting its elements (a surrogate pair is one), and a
    /// record to retry longer ones.
    fn emit_lazy_decoded(&mut self, id: usize, pc: usize, test: &Test, min: u32, max: Option<u32>) {
        let bt = self.bt;
        let back = self.plan.progs[id].back;
        if min > 0 {
            self.emit_elem_scan(test, Some(min), back);
            self.m.branch_count(false, bt);
        }
        self.m.set_rs(min);
        let check = self.m.label();
        let extend = self.m.label();
        self.m.bind(check);
        if let Some(next) = self.plan.first_test(id, pc + 1) {
            self.emit_may_match(&next, extend, back);
        }
        let skip = self.m.label();
        if let Some(max) = max {
            self.m.branch_rs(Cond::Hs, max, skip);
        }
        let retry = self.stub(Stub::Lazy(id, pc));
        self.m.push(retry, Operand::Pack(0));
        self.m.choices(1);
        self.m.bind(skip);
        self.lazy.push((id, pc, check, extend));
        self.extend.insert((id, pc), extend);
    }

    /// A lazy loop's extension: one more element, then the continuation's test again.
    fn emit_lazy_extend(&mut self, id: usize, pc: usize, check: Label, extend: Label) {
        let bt = self.bt;
        let program = &self.plan.progs[id];
        let back = program.back;
        let Inst::Many { rep, max, .. } = &program.prog[pc] else {
            unreachable!("lazy extension outside its loop");
        };
        let test = self.plan.rep_test(rep, program.flags[pc]);
        let max = max.and_then(|max| u32::try_from(max).ok().filter(|&m| m < 1 << 31));
        self.m.bind(extend);
        if self.plan.shape(&test) != Shape::Units(1) {
            if let Some(max) = max {
                self.m.branch_rs(Cond::Hs, max, bt);
            }
            self.emit_decoded(&test, back, bt);
            self.m.rs_add(1, false);
        } else {
            if let Some(max) = max {
                self.m.branch_run(Cond::Hs, max, back, bt);
            }
            if back {
                self.m.branch_at_start(true, bt);
                self.m.load_elem(-1);
            } else {
                self.m.branch_at_end(true, bt);
                self.m.load_elem(0);
            }
            self.emit_test(&test, bt);
            if back {
                self.m.retreat(1);
            } else {
                self.m.advance(1);
            }
        }
        self.m.poll_tick();
        self.m.jump(check);
    }

    fn emit_stub(&mut self, stub: Stub) {
        let bt = self.bt;
        match stub {
            Stub::Alt(id, target) => {
                self.m.pos_from_op();
                self.m.choices(-1);
                self.m.jump(self.pcs[id][target]);
            }
            Stub::Undo(slot) => {
                self.m.store_slot(slot, Value::Op);
                self.m.jump(bt);
            }
            Stub::Greedy(id, pc) => {
                // As `Matcher::greedy_retry`: one element shorter at a time, down to the least
                // end, until the continuation's first test may match. Reading backward, the run
                // ends below its start, and a shorter one ends above.
                let program = &self.plan.progs[id];
                let back = program.back;
                let Inst::Many { rep, .. } = &program.prog[pc] else {
                    unreachable!("greedy retry outside its loop");
                };
                let decoded =
                    self.plan.shape(&self.plan.rep_test(rep, program.flags[pc])) != Shape::Units(1);
                self.m.choices(-1);
                self.m.unpack_op();
                let retry = self.m.label();
                let next = self.m.label();
                self.m.bind(retry);
                match (decoded, back) {
                    // The element before POS, a unit or a pair, comes off the run's end.
                    (true, _) => {
                        self.m.decode_elem(!back);
                        self.m.step_elem(!back);
                    }
                    (false, false) => self.m.retreat(1),
                    (false, true) => self.m.advance(1),
                }
                self.m.poll_tick();
                if let Some(test) = self.plan.first_test(id, pc + 1) {
                    self.emit_may_match(&test, next, back);
                }
                let skip = self.m.label();
                self.m
                    .branch_pos_rs(if back { Cond::Hs } else { Cond::Ls }, skip);
                let again = self.stub(Stub::Greedy(id, pc));
                self.m.push(again, Operand::Pack(0));
                self.m.choices(1);
                self.m.bind(skip);
                self.m.jump(self.pcs[id][pc + 1]);
                self.m.bind(next);
                self.m
                    .branch_pos_rs(if back { Cond::Lo } else { Cond::Hi }, retry);
                self.m.jump(bt);
            }
            Stub::Lazy(id, pc) => {
                self.m.choices(-1);
                self.m.unpack_op();
                let extend = self.extend[&(id, pc)];
                self.m.jump(extend);
            }
            Stub::LookFail(id, pc) => {
                // The body failed: a positive lookaround fails, a negative one holds.
                self.m.unpack_pos_choices();
                match self.plan.progs[id].prog[pc] {
                    Inst::Look { negate: true, .. } | Inst::LookBehind { negate: true, .. } => {
                        self.m.jump(self.pcs[id][pc + 1])
                    }
                    _ => self.m.jump(bt),
                }
            }
            Stub::Clear(lo, hi) => {
                self.m.fill_unset(2 * lo..2 * hi + 2);
                self.m.jump(bt);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Running the code
// ---------------------------------------------------------------------------------------------

/// Run `code` over `input` from `start` (`only` alone, when it is set), as
/// [`Regex::run_matcher`] runs the matcher: `Ok(true)` leaves the match's capture slots in
/// `scratch.caps`.
#[allow(clippy::too_many_arguments)]
pub(super) fn run<I: ReInput>(
    re: &Regex,
    code: &NativeCode,
    input: I,
    start: usize,
    only: Option<usize>,
    control: &crate::RuntimeInterrupt,
    poll: &PollWord,
    scratch: &mut MatchScratch,
) -> Result<bool, MatchError> {
    let matched = run_native(re, code, input, start, only, control, poll, scratch);
    if env_flag!("TRUST_JS_REGEXP_NATIVE_VERIFY") {
        verify(re, input, start, only, control, scratch, &matched);
    }
    matched
}

#[allow(clippy::too_many_arguments)]
fn run_native<I: ReInput>(
    re: &Regex,
    code: &NativeCode,
    input: I,
    start: usize,
    only: Option<usize>,
    control: &crate::RuntimeInterrupt,
    poll: &PollWord,
    scratch: &mut MatchScratch,
) -> Result<bool, MatchError> {
    debug_assert_eq!(I::NATIVE_WIDTH, Some(code.width));
    debug_assert_eq!(I::PAIRED, code.paired);
    let nslots = 2 * (re.ngroups + 1);
    let slots = code.slots;
    if scratch.native_slots.len() != slots || scratch.native_stack.capacity() < INITIAL_RECORDS {
        prepare(scratch, slots);
    }
    let stack = &mut scratch.native_stack;
    debug_assert!(stack.is_empty());
    let base = stack.as_mut_ptr();
    let (subject, len) = input.native_parts();
    let mut ctx = NativeCtx {
        subject,
        len,
        start: only.unwrap_or(start),
        slots: scratch.native_slots.as_mut_ptr(),
        stack: base,
        stack_end: base.wrapping_add(stack.capacity()),
        poll_word: poll.as_ptr(),
        window: if re.prefilter.is_some() {
            0
        } else {
            usize::MAX
        },
        lead: 0,
        regex: re,
        control,
        stack_vec: stack,
        limit: backtrack_state_limit(),
        saved_sp: 0,
    };
    regexp_prof_native_exec();
    // SAFETY: the code was generated for this width and reads the subject only below its length;
    // the context's pointers live for the call, and the helpers it calls take the context only.
    let status = unsafe { (code.entry)(&mut ctx) };
    let stack = &mut scratch.native_stack;
    debug_assert!(stack.is_empty());
    if stack.capacity() > super::RETAINED_BACKTRACK_STATE.max(INITIAL_RECORDS) {
        release(stack);
    }
    let native = &mut scratch.native_slots[..nslots];
    let result = match status {
        STATUS_MATCH => {
            for (slot, &value) in scratch.caps.iter_mut().zip(native.iter()) {
                *slot = super::unpack_slot(value);
            }
            Ok(true)
        }
        STATUS_NO_MATCH => Ok(false),
        STATUS_CANCELLED => Err(MatchError::Interrupted(crate::InterruptReason::Cancelled)),
        STATUS_NAVIGATION => Err(MatchError::Interrupted(
            crate::InterruptReason::UserNavigation,
        )),
        STATUS_DEADLINE => Err(MatchError::Interrupted(
            crate::InterruptReason::DeadlineExceeded,
        )),
        _ => Err(MatchError::ResourceExhausted),
    };
    // The slots stay unset between executions: a failed attempt's undo records unset what it
    // set, but for slot 0 and a wide program's slots, which each attempt sets anew.
    if nslots > 64 || !matches!(result, Ok(false)) {
        native.fill(usize::MAX);
    } else {
        native[0] = usize::MAX;
    }
    result
}

/// Size the scratch's native slots for a program of `slots`, all unset, and give its stack its
/// first records: once a program, out of line.
#[cold]
#[inline(never)]
fn prepare(scratch: &mut MatchScratch, slots: usize) {
    if scratch.native_slots.len() != slots {
        scratch.native_slots = vec![usize::MAX; slots].into_boxed_slice();
    }
    scratch.native_stack.reserve(INITIAL_RECORDS);
}

/// Release a large match's backtrack stack, keeping its first records.
#[cold]
#[inline(never)]
fn release(stack: &mut Vec<[usize; 2]>) {
    stack.shrink_to(INITIAL_RECORDS);
}

/// `TRUST_JS_REGEXP_NATIVE_VERIFY=1`: run the matcher where the native code ran and abort, with
/// the pattern, flags, subject and start, on any difference in the result, the capture spans or
/// the error kind. An interruption only one of the two saw is no difference: each polls at its
/// own points.
#[cold]
#[inline(never)]
fn verify<I: ReInput>(
    re: &Regex,
    input: I,
    start: usize,
    only: Option<usize>,
    control: &crate::RuntimeInterrupt,
    scratch: &mut MatchScratch,
    native: &Result<bool, MatchError>,
) {
    let native_caps: Vec<Option<usize>> = scratch.caps.to_vec();
    scratch.caps.fill(None);
    let reference = re.run_matcher(input, start, only, control, scratch);
    let reference_caps: Vec<Option<usize>> = scratch.caps.to_vec();
    // Leave the native result for the caller to project.
    scratch.caps.copy_from_slice(&native_caps);
    let same = match (native, &reference) {
        (Ok(true), Ok(true)) => native_caps == reference_caps,
        (Ok(a), Ok(b)) => a == b,
        (Err(MatchError::Interrupted(a)), Err(MatchError::Interrupted(b))) => a == b,
        (Err(MatchError::Interrupted(_)), Ok(_)) | (Ok(_), Err(MatchError::Interrupted(_))) => true,
        (Err(a), Err(b)) => a == b,
        _ => false,
    };
    if same {
        return;
    }
    let subject: Vec<u32> = (0..input.len()).map(|at| input.at(at)).collect();
    let shown = if subject.iter().all(|&unit| unit < 0x80) {
        format!(
            "{:?}",
            subject
                .iter()
                .map(|&unit| unit as u8 as char)
                .collect::<String>()
        )
    } else {
        format!("{subject:x?}")
    };
    eprintln!(
        "[regexp-native-verify] DIFFERENCE /{}/{} {} start={start} only={only:?}\n  subject ({} units): {shown}\n  native:  {native:?} {native_caps:?}\n  matcher: {reference:?} {reference_caps:?}",
        re.source,
        re.flags,
        I::NATIVE_WIDTH.map_or("?", Width::name),
        input.len()
    );
    std::process::abort();
}

/// The poll counter's slow path: the poll word's interrupt bit is set, so ask the handle.
extern "C" fn helper_poll(ctx: *mut NativeCtx) -> u64 {
    // SAFETY: the generated code passes the context of the running call.
    let ctx = unsafe { &*ctx };
    let control = unsafe { &*ctx.control };
    match control.current_reason() {
        None => 0,
        Some(reason) => status_of(MatchError::Interrupted(reason)),
    }
}

fn status_of(error: MatchError) -> u64 {
    match error {
        MatchError::Interrupted(crate::InterruptReason::Cancelled) => STATUS_CANCELLED,
        MatchError::Interrupted(crate::InterruptReason::UserNavigation) => STATUS_NAVIGATION,
        MatchError::Interrupted(crate::InterruptReason::DeadlineExceeded) => STATUS_DEADLINE,
        MatchError::ResourceExhausted => STATUS_EXHAUSTED,
    }
}

/// The stack is full at `top`: grow it, up to
/// [`MAX_BACKTRACK_STATE`](super::MAX_BACKTRACK_STATE) records (the context's
/// limit), returning the new top, or null when the limit is reached.
extern "C" fn helper_grow(ctx: *mut NativeCtx, top: *mut [usize; 2]) -> *mut [usize; 2] {
    // SAFETY: the generated code passes the context of the running call and its stack's top.
    let ctx = unsafe { &mut *ctx };
    let stack = unsafe { &mut *ctx.stack_vec };
    let used = (top as usize - ctx.stack as usize) / std::mem::size_of::<[usize; 2]>();
    if used >= ctx.limit {
        return std::ptr::null_mut();
    }
    let wanted = used
        .saturating_mul(2)
        .clamp(INITIAL_RECORDS, ctx.limit.max(1));
    // The records below the top are initialized: keep them across the reallocation.
    unsafe { stack.set_len(used) };
    if stack.try_reserve_exact(wanted - used).is_err() {
        unsafe { stack.set_len(0) };
        return std::ptr::null_mut();
    }
    let base = stack.as_mut_ptr();
    let end = base.wrapping_add(stack.capacity());
    unsafe { stack.set_len(0) };
    ctx.stack = base;
    ctx.stack_end = end;
    base.wrapping_add(used)
}

/// The candidate loop's scan: the next start from `from` ([`Regex::next_candidate`]), or
/// `1 << 63` with a status (0: none is left).
extern "C" fn helper_next_candidate<I: ReInput + NativeInput>(
    ctx: *mut NativeCtx,
    from: usize,
) -> u64 {
    // SAFETY: the generated code passes the context of the running call, whose subject has
    // `I`'s width.
    let ctx = unsafe { &mut *ctx };
    let re = unsafe { &*ctx.regex };
    let control = unsafe { &*ctx.control };
    let input = unsafe { I::from_parts(ctx.subject, ctx.len) };
    let mut window = match ctx.window {
        0 => None,
        usize::MAX => None,
        end => Some(end - 1),
    };
    let found = re.next_candidate(input, from, &mut window, control);
    if re.prefilter.is_some() {
        ctx.window = window.map_or(0, |end| end + 1);
    }
    match found {
        Ok(Some(at)) => at as u64,
        Ok(None) => 1 << 63,
        Err(error) => 1 << 63 | status_of(error),
    }
}

/// A backreference under `i`, as the matcher's `backref_end` compares it: the capture's `n`
/// units at `lo` against the subject's at `pos` (`back`: those ending at `pos`), by
/// Canonicalize's, or in Unicode mode by simple case folding's, element by element over a paired
/// subject. Answers the position past the repetition (before it) plus one, 0 when the subject
/// does not repeat the capture there, or `1 << 63` with a status when the host interrupts the
/// compare, which polls as the matcher's does.
extern "C" fn helper_backref_fold<I: ReInput + NativeInput>(
    ctx: *mut NativeCtx,
    lo: usize,
    pos: usize,
    n: usize,
    back: u64,
) -> u64 {
    // SAFETY: the generated code passes the context of the running call, whose subject has
    // `I`'s width, and a capture span it holds.
    let ctx = unsafe { &*ctx };
    let control = unsafe { &*ctx.control };
    let re = unsafe { &*ctx.regex };
    let input = unsafe { I::from_parts(ctx.subject, ctx.len) };
    match backref_fold(input, lo, lo + n, pos, back != 0, re.unicode, control) {
        Ok(Some(next)) => next as u64 + 1,
        Ok(None) => 0,
        Err(error) => 1 << 63 | status_of(error),
    }
}

/// [`Matcher::backref_end`](super::Matcher::backref_end) under `i`, for `input` with no matcher:
/// where a repetition of the capture `a..b` from `pos` ends, if the subject has one.
fn backref_fold<I: ReInput>(
    input: I,
    a: usize,
    b: usize,
    pos: usize,
    back: bool,
    unicode: bool,
    control: &crate::RuntimeInterrupt,
) -> Result<Option<usize>, MatchError> {
    let poll = |index: usize| {
        if index == 0 || index & super::INTERRUPT_POLL_MASK != 0 {
            Ok(())
        } else {
            super::poll_interrupt(control)
        }
    };
    let same = |x: u32, y: u32| super::chars_eq(x, y, true, unicode);
    if I::PAIRED {
        // `backref_end_by_elements`: case-insensitively equal code points need not take as
        // many units.
        let mut len = 0usize;
        let mut at = a;
        while at < b {
            at = input.next_elem(at).1;
            len += 1;
        }
        let start = if back {
            let mut start = pos;
            for _ in 0..len {
                if start == 0 {
                    return Ok(None);
                }
                start = input.prev_elem(start).1;
            }
            start
        } else {
            pos
        };
        let (mut subject, mut captured) = (start, a);
        for index in 0..len {
            poll(index)?;
            if subject >= input.len() {
                return Ok(None);
            }
            let (found, after) = input.next_elem(subject);
            let (expected, next) = input.next_elem(captured);
            if !same(found, expected) {
                return Ok(None);
            }
            (subject, captured) = (after, next);
        }
        return Ok(Some(if back { start } else { subject }));
    }
    let n = b - a;
    let start = if back {
        let Some(start) = pos.checked_sub(n) else {
            return Ok(None);
        };
        start
    } else {
        if pos + n > input.len() {
            return Ok(None);
        }
        pos
    };
    for index in 0..n {
        poll(index)?;
        if !same(input.at(start + index), input.at(a + index)) {
            return Ok(None);
        }
    }
    Ok(Some(if back { pos - n } else { pos + n }))
}

/// The subjects the tier runs on, rebuilt from the parts the context keeps.
pub(super) trait NativeInput: Sized {
    /// # Safety
    /// `subject` and `len` are the parts of a live subject of this width.
    unsafe fn from_parts(subject: *const u8, len: usize) -> Self;
}

impl NativeInput for &[u8] {
    unsafe fn from_parts(subject: *const u8, len: usize) -> Self {
        std::slice::from_raw_parts(subject, len)
    }
}

impl NativeInput for &[u16] {
    unsafe fn from_parts(subject: *const u8, len: usize) -> Self {
        std::slice::from_raw_parts(subject as *const u16, len)
    }
}

impl NativeInput for super::Paired<'_> {
    unsafe fn from_parts(subject: *const u8, len: usize) -> Self {
        super::Paired(std::slice::from_raw_parts(subject as *const u16, len))
    }
}

#[cfg(test)]
#[path = "regex_native_tests.rs"]
mod tests;
