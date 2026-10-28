//! The RegExp native tier (W5b's item 3a): a pattern's [`Inst`] program compiled to machine
//! code, one template per instruction, for one subject width, in the shape of V8's
//! assemble-from-bytecode generator (`--regexp-assemble-from-bytecode`,
//! src/regexp/regexp-code-generator.cc): the generator below walks the program and emits through
//! a macro assembler ([`Masm`]) that each backend implements, as V8's per-architecture
//! `RegExpMacroAssembler`s do. [`Matcher::run`] stays the reference, and the matcher runs
//! whatever a program uses that the tier does not.
//!
//! The tier takes the core of the instruction set (characters, `.`, classes, captures and their
//! undo rules, alternation, `^`, `$` and `\b` with the flags resolved at compile time, greedy and
//! lazy loops of one element, atomic loops, group loops' progress marks and capture clears) over
//! one-byte and two-byte subjects outside Unicode mode and without `i`. The candidate loop runs
//! in the generated code, with the first-element filter's table tested in line; the prefilter's
//! literal and the filter's vector scans run in Rust ([`Regex::next_candidate`]), which the code
//! calls when its own test rules a start out.
//!
//! Machine state: an entry `fn(*mut NativeCtx) -> u64` returning a status ([`STATUS_MATCH`] and
//! the others). Callee-saved registers hold the subject, its length, the position, the capture
//! slots (the marks follow them), the backtrack stack's top, the root floor's logged slots, the
//! context, the leading loop's run end, the choice count and the poll counter (AArch64 x19-x28;
//! x86-64 keeps some in caller-saved registers that every helper call saves, and the context).
//! The backtrack stack is the program's [`MatchScratch`] storage taken as 16-byte records: the
//! address of a resume point or an undo stub, and an operand. A pop jumps to the record's address
//! with the operand in a register; undo stubs restore their slot and pop again, as V8's undo
//! code does (regexp-compiler.cc, `Trace::Flush`). A sentinel record at the stack's base ends a
//! failed attempt, as V8 pushes its fail label. A Rust helper grows the stack and enforces
//! [`MAX_BACKTRACK_STATE`], counted in records.
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

/// A subject's positions fit in 32 bits below this, as the tier's packed record operands need:
/// longer subjects (engine strings never are) stay on the matcher.
const MAX_SUBJECT_LEN: usize = 1 << 31;

/// Programs longer than this stay on the matcher: their code's branches might not reach.
const MAX_PROGRAM_LEN: usize = 4096;

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
}

/// What a slot store writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Value {
    Pos,
    Unset,
    /// The operand of the record just popped.
    Op,
}

/// The macro assembler the generator emits through, as V8's `RegExpMacroAssembler`. Its
/// operations name the machine state by role:
/// - POS, the position; LEN, the subject's length; E, the element register, 32 bits wide, which
///   a helper call clobbers;
/// - SLOTS, the capture slots (unset is all ones), then the marks;
/// - TOP, the backtrack stack's top; OP, the operand of the record popped last;
/// - CHOICES, the choice records on the stack; LOGGED, the slots whose first write at the root
///   floor has an undo record this attempt (one bit a slot, [`Matcher::logged`]);
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
    /// Branch if fewer than `k` elements are left past POS.
    fn branch_room_lt(&mut self, k: u32, label: Label);
    /// Branch if POS is 0 (`at`), or is not.
    fn branch_at_start(&mut self, at: bool, label: Label);
    /// Branch if POS is LEN (`at`), or is not.
    fn branch_at_end(&mut self, at: bool, label: Label);
    fn advance(&mut self, k: u32);
    fn retreat(&mut self, k: u32);
    /// E = the element at POS + `offset`, which the caller has checked is in the subject.
    fn load_elem(&mut self, offset: i32);
    /// Branch if E `cond` `value`.
    fn branch_elem(&mut self, cond: Cond, value: u32, label: Label);
    /// Branch if E is in `lo..=hi` (`inside`), or is not.
    fn branch_elem_in(&mut self, lo: u32, hi: u32, inside: bool, label: Label);
    /// A byte table for [`Self::branch_lut`].
    fn lut(&mut self, table: &[u8; 256]) -> Data;
    /// Branch if `lut` holds a nonzero byte at E (`member`), or a zero one; E is below 256.
    fn branch_lut(&mut self, lut: Data, member: bool, label: Label);
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
    /// POS = OP.
    fn pos_from_op(&mut self);
    /// POS = OP's low 32 bits, RS = its high ones.
    fn unpack_op(&mut self);
    /// RS = POS.
    fn run_start(&mut self);
    /// LIM = min(LEN, POS + `max`).
    fn set_limit(&mut self, max: Option<u32>);
    /// CLIM = min(LIM, POS + [`SCAN_STRETCH`]).
    fn set_stretch_limit(&mut self);
    fn branch_pos_ge_stretch(&mut self, label: Label);
    fn branch_stretch_is_limit(&mut self, label: Label);
    /// POS = LIM.
    fn pos_to_limit(&mut self);
    /// Branch if POS - RS `cond` `n`.
    fn branch_run(&mut self, cond: Cond, n: u32, label: Label);
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
    /// else POS = the start plus one, and with `lead`, at least LEAD.
    fn next_start(&mut self, lead: bool, no_match: Label);
    /// The finished code and data, or `None` if a branch does not reach.
    fn finish(self) -> Option<Vec<u8>>;
}

/// The addresses of the helpers the generated code calls.
#[derive(Clone, Copy)]
pub(super) struct Helpers {
    pub(super) poll: usize,
    pub(super) grow: usize,
    pub(super) next_candidate: usize,
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
pub(super) fn note_execution(re: &Regex, width: Width, len: usize) {
    if re.tier_up_at == 0 {
        return;
    }
    let ticks = re.tier_ticks.get().saturating_add(1);
    re.tier_ticks.set(ticks);
    if ticks < re.tier_up_at && len < TIER_UP_LENGTH {
        return;
    }
    let cell = &re.native.codes[width.index()];
    if cell.get().is_some() {
        return;
    }
    let code = compile(re, width);
    super::regexp_prof_tier_up(code.is_some());
    let _ = cell.set(code);
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
    if re.ignore_case {
        return Some("i flag");
    }
    if re.unicode {
        return Some("Unicode mode");
    }
    if re.single.is_some() {
        return Some("single element");
    }
    if width == Width::One && (re.literal_ascii.is_some() || re.literal_fold.is_some()) {
        return Some("literal");
    }
    let prog = &re.prog;
    let n = prog.len();
    if n > MAX_PROGRAM_LEN {
        return Some("program length");
    }
    if 2 * (re.ngroups + 1) + re.nmarks > MAX_SLOTS {
        return Some("slots");
    }
    if n < 3
        || !matches!(prog[0], Inst::Save(0))
        || !matches!(prog[n - 2], Inst::Save(1))
        || !matches!(prog[n - 1], Inst::Match)
    {
        return Some("program shape");
    }
    for (pc, inst) in prog.iter().enumerate() {
        let core = match inst {
            Inst::Save(0) => pc == 0,
            Inst::Save(1) => pc == n - 2,
            Inst::Match => pc == n - 1,
            Inst::Class(class) => class_plan(class, width).is_some(),
            Inst::Many { rep, .. } => match rep {
                Rep::Class(class) => class_plan(class, width).is_some(),
                Rep::Char(_) | Rep::Any => true,
            },
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
            | Inst::ClearCaps(..) => true,
            Inst::StringSet(_)
            | Inst::StringSetRepeat { .. }
            | Inst::Backref(_)
            | Inst::BackrefAlt(_)
            | Inst::Look { .. }
            | Inst::LookBehind { .. }
            | Inst::PushFlags(..)
            | Inst::PopFlags
            | Inst::CounterReset(_)
            | Inst::CounterIncr(_)
            | Inst::CountedBranch { .. }
            | Inst::CheckCountedProgress { .. } => false,
        };
        if !core {
            return Some("instruction");
        }
    }
    None
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
        next_candidate: match plan.width {
            Width::One => helper_next_candidate::<&[u8]> as *const () as usize,
            Width::Two => helper_next_candidate::<&[u16]> as *const () as usize,
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
    Any,
    Class(Rc<ClassPlan>),
}

/// A class's tests: the byte table of its elements below 256, and its ranges of two-byte elements
/// above them.
struct ClassPlan {
    lut: [u8; 256],
    high: Vec<(u32, u32)>,
}

/// `class`'s tests for `width`, or `None` if it has too many ranges to test in line.
fn class_plan(class: &super::CharClass, width: Width) -> Option<ClassPlan> {
    let mut lut = [0u8; 256];
    for (element, member) in lut.iter_mut().enumerate() {
        *member = u8::from(class.matches(element as u32, false, false));
    }
    let high = match width {
        Width::One => Vec::new(),
        Width::Two => class_elements(class)?
            .into_iter()
            .filter(|&(_, hi)| hi >= 0x100)
            .filter(|&(lo, _)| lo <= 0xffff)
            .map(|(lo, hi)| (lo.max(0x100), hi.min(0xffff)))
            .collect(),
    };
    (high.len() <= MAX_HIGH_RANGES).then_some(ClassPlan { lut, high })
}

/// What the generator needs to know of a program, for one width.
struct Plan<'a> {
    prog: &'a [Inst],
    width: Width,
    /// Capture slots (two per group, group 0's included); the marks follow them.
    nslots: usize,
    /// More capture slots than LOGGED has bits: each attempt unsets them all instead.
    wide: bool,
    /// The scan tries one start (sticky, or anchored at the input's start).
    only: bool,
    multiline: bool,
    dotall: bool,
    lead_run: bool,
    /// The pattern has a prefilter, whose window the candidate loop keeps.
    window: bool,
    /// The first-element filter's table, which the candidate loop tests in line.
    first: Option<[u8; 256]>,
    /// The pcs a branch or a resumed record reaches, which begin a run of templates.
    targets: Vec<bool>,
    /// Tests by class, made once per class (by address).
    classes: std::cell::RefCell<std::collections::HashMap<usize, Rc<ClassPlan>>>,
}

impl<'a> Plan<'a> {
    fn new(re: &'a Regex, width: Width) -> Self {
        let prog = &re.prog[..];
        let mut targets = vec![false; prog.len() + 1];
        for (pc, inst) in prog.iter().enumerate() {
            match inst {
                Inst::Split(a, b) => {
                    targets[*a] = true;
                    targets[*b] = true;
                }
                Inst::Jmp(target) => targets[*target] = true,
                // Loops resume after themselves.
                Inst::Many { .. } => targets[pc + 1] = true,
                _ => {}
            }
        }
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
        Plan {
            prog,
            width,
            nslots,
            wide: nslots > 64,
            only: re.sticky || matches!(re.first, FirstFilter::Anchored),
            multiline: re.multiline,
            dotall: re.dotall,
            lead_run: re.lead_run,
            window: re.prefilter.is_some(),
            first,
            targets,
            classes: Default::default(),
        }
    }

    fn class(&self, class: &Rc<super::CharClass>) -> Rc<ClassPlan> {
        let key = Rc::as_ptr(class) as usize;
        self.classes
            .borrow_mut()
            .entry(key)
            .or_insert_with(|| {
                Rc::new(class_plan(class, self.width).expect("declined classes never compile"))
            })
            .clone()
    }

    fn rep_test(&self, rep: &Rep) -> Test {
        match rep {
            Rep::Char(c) => Test::Char(*c),
            Rep::Any => Test::Any,
            Rep::Class(class) => Test::Class(self.class(class)),
        }
    }

    /// The element test of an element instruction.
    fn element_test(&self, inst: &Inst) -> Option<Test> {
        match inst {
            Inst::Char(c) => Some(Test::Char(*c)),
            Inst::Any => Some(Test::Any),
            Inst::Class(class) => Some(Test::Class(self.class(class))),
            _ => None,
        }
    }

    /// The test the branch at `pc` begins with, past capture and mark bookkeeping: what
    /// [`Matcher::branch_may_match`] tests at run time, resolved at compile time. `None` when the
    /// branch may match without one.
    fn first_test(&self, mut pc: usize) -> Option<Test> {
        for _ in 0..4 {
            match &self.prog[pc] {
                Inst::Char(_) | Inst::Any | Inst::Class(_) => {
                    return self.element_test(&self.prog[pc])
                }
                Inst::Many { rep, min, .. } if *min > 0 => return Some(self.rep_test(rep)),
                Inst::Save(_) | Inst::ClearCaps(..) | Inst::SetMark(_) => pc += 1,
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
    /// Resume the branch at a pc with POS from the record.
    Alt(usize),
    /// Restore a slot from the record, and backtrack on.
    Undo(usize),
    /// Retry the greedy loop at a pc with a shorter run.
    Greedy(usize),
    /// Retry the lazy loop at a pc with a longer run.
    Lazy(usize),
}

/// The template generator: one pass over the program, in order.
struct Generator<'p, 'a, M: Masm> {
    m: &'p mut M,
    plan: &'p Plan<'a>,
    pcs: Vec<Label>,
    bt: Label,
    stubs: Vec<(Stub, Label)>,
    stub_labels: std::collections::HashMap<Stub, Label>,
    /// Lazy loops' extension code, emitted after the program: (pc, check label, extend label).
    lazy: Vec<(usize, Label, Label)>,
    /// Each lazy loop's extension, which its retry stub jumps to.
    extend: std::collections::HashMap<usize, Label>,
}

impl<'p, 'a, M: Masm> Generator<'p, 'a, M> {
    fn new(m: &'p mut M, plan: &'p Plan<'a>) -> Self {
        let pcs = (0..=plan.prog.len()).map(|_| m.label()).collect();
        let bt = m.label();
        Generator {
            m,
            plan,
            pcs,
            bt,
            stubs: Vec::new(),
            stub_labels: Default::default(),
            lazy: Vec::new(),
            extend: Default::default(),
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
        let matched = self.m.label();
        let no_match = self.m.label();
        let attempt = self.m.label();
        let cand = self.m.label();
        self.m.prologue(sentinel);
        if !plan.only {
            // The candidate loop: a start the filter's table admits, inside the prefilter's
            // window, is tried at once; any other goes to the Rust scan.
            self.m.bind(cand);
            self.m.poll_tick();
            let scan = self.m.label();
            let mut scans = false;
            if plan.window {
                self.m.branch_window(scan);
                scans = true;
            }
            if let Some(table) = &plan.first {
                self.m.branch_at_end(true, scan);
                self.m.load_elem(0);
                if plan.width == Width::Two {
                    self.m.branch_elem(Cond::Hs, 0x100, scan);
                }
                let lut = self.m.lut(table);
                self.m.branch_lut(lut, false, scan);
                scans = true;
            }
            if scans {
                self.m.jump(attempt);
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
        let mut pc = 0;
        while pc < plan.prog.len() {
            pc = self.emit_at(pc, matched);
        }
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
            self.m.next_start(plan.lead_run, no_match);
            self.m.jump(cand);
        }
        let mut emitted = 0;
        while emitted < self.stubs.len() || !self.lazy.is_empty() {
            if let Some((pc, check, extend)) = self.lazy.pop() {
                self.emit_lazy_extend(pc, check, extend);
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

    /// Emit the template of `pc` (or of the run of element instructions it begins), returning
    /// the pc after it.
    fn emit_at(&mut self, pc: usize, matched: Label) -> usize {
        let plan = self.plan;
        let bt = self.bt;
        self.m.bind(self.pcs[pc]);
        match &plan.prog[pc] {
            Inst::Char(_) | Inst::Any | Inst::Class(_) => {
                // A straight run of elements checks the room once and tests each at its offset.
                let mut end = pc + 1;
                while end < plan.prog.len()
                    && !plan.targets[end]
                    && matches!(plan.prog[end], Inst::Char(_) | Inst::Any | Inst::Class(_))
                {
                    end += 1;
                }
                let k = (end - pc) as u32;
                self.m.branch_room_lt(k, bt);
                for (offset, inst) in plan.prog[pc..end].iter().enumerate() {
                    let test = plan.element_test(inst).expect("an element instruction");
                    self.m.load_elem(offset as i32);
                    self.emit_test(&test, bt);
                }
                self.m.advance(k);
                return end;
            }
            Inst::Save(slot) => {
                let slot = *slot;
                // Slot 0 is written first in every attempt, at the root floor, and slot 1 only
                // before the match: neither needs an undo record.
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
            Inst::Split(a, b) => {
                let (a, b) = (*a, *b);
                // As `Matcher::branch`: a first branch that fails at once is skipped, and the
                // second is recorded only if it may match here.
                if let Some(test) = plan.first_test(a) {
                    self.emit_may_match(&test, self.pcs[b]);
                }
                let skip = self.m.label();
                if let Some(test) = plan.first_test(b) {
                    self.emit_may_match(&test, skip);
                }
                let alt = self.stub(Stub::Alt(b));
                self.m.push(alt, Operand::Pos);
                self.m.choices(1);
                self.m.bind(skip);
                if a != pc + 1 {
                    self.m.jump(self.pcs[a]);
                }
            }
            Inst::Jmp(target) => {
                if *target <= pc {
                    self.m.poll_tick();
                }
                self.m.jump(self.pcs[*target]);
            }
            Inst::Match => self.m.jump(matched),
            Inst::AssertStart => {
                if plan.multiline {
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
                if plan.multiline {
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
            Inst::WordBoundary(want) => self.emit_word_boundary(*want),
            Inst::Many {
                rep,
                min,
                max,
                greedy,
                atomic,
            } => {
                let test = plan.rep_test(rep);
                let min = u32::try_from(*min).unwrap_or(u32::MAX);
                // A count past any subject's length bounds nothing.
                let max = max.and_then(|max| u32::try_from(max).ok().filter(|&m| m < 1 << 31));
                if *greedy {
                    self.emit_greedy(pc, &test, min, max, *atomic);
                } else {
                    self.emit_lazy(pc, &test, min, max);
                }
            }
            Inst::SetMark(id) => {
                let slot = plan.nslots + id;
                let store = self.m.label();
                let undo = self.stub(Stub::Undo(slot));
                self.m.branch_choices(true, store);
                self.m.push(undo, Operand::Slot(slot));
                self.m.bind(store);
                self.m.store_slot(slot, Value::Pos);
            }
            Inst::CheckProgress(id) => {
                // An iteration that consumed nothing fails (RepeatMatcher's empty check).
                self.m.branch_slot_eq_pos(plan.nslots + id, bt);
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
            _ => unreachable!("declined instruction in a native program"),
        }
        pc + 1
    }

    /// Test E with `test`, going to `fail` if it does not match.
    fn emit_test(&mut self, test: &Test, fail: Label) {
        match test {
            Test::Char(c) => {
                if *c > self.plan.width.max_unit() {
                    self.m.jump(fail);
                } else {
                    self.m.branch_elem(Cond::Ne, *c, fail);
                }
            }
            Test::Any => {
                if !self.plan.dotall {
                    self.m.branch_elem(Cond::Eq, 0x0a, fail);
                    self.m.branch_elem(Cond::Eq, 0x0d, fail);
                    if self.plan.width == Width::Two {
                        self.m.branch_elem_in(0x2028, 0x2029, true, fail);
                    }
                }
            }
            Test::Class(class) => {
                if self.plan.width == Width::One {
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

    /// Go to `fail` unless an element at POS passes `test`.
    fn emit_may_match(&mut self, test: &Test, fail: Label) {
        self.m.branch_at_end(true, fail);
        self.m.load_elem(0);
        self.emit_test(test, fail);
    }

    /// Go to `hit` if E is a line terminator.
    fn emit_line_terminator(&mut self, hit: Label) {
        self.m.branch_elem(Cond::Eq, 0x0a, hit);
        self.m.branch_elem(Cond::Eq, 0x0d, hit);
        if self.plan.width == Width::Two {
            self.m.branch_elem_in(0x2028, 0x2029, true, hit);
        }
    }

    /// Go to `hit` if E is a word character (`[0-9A-Za-z_]`, IsWordChar outside Unicode mode
    /// and `i`).
    fn emit_word(&mut self, hit: Label) {
        let mut table = [0u8; 256];
        for (element, member) in table.iter_mut().enumerate() {
            *member = u8::from(super::is_word(element as u32));
        }
        let skip = self.m.label();
        if self.plan.width == Width::Two {
            self.m.branch_elem(Cond::Hs, 0x100, skip);
        }
        let lut = self.m.lut(&table);
        self.m.branch_lut(lut, true, hit);
        self.m.bind(skip);
    }

    /// `\b` (`want`) or `\B`: whether the elements before and after POS differ in being word
    /// characters.
    fn emit_word_boundary(&mut self, want: bool) {
        let ok = self.m.label();
        let (boundary, none) = if want { (ok, self.bt) } else { (self.bt, ok) };
        let before_word = self.m.label();
        let before_not = self.m.label();
        self.m.branch_at_start(true, before_not);
        self.m.load_elem(-1);
        self.emit_word(before_word);
        self.m.bind(before_not);
        self.m.branch_at_end(true, none);
        self.m.load_elem(0);
        self.emit_word(boundary);
        self.m.jump(none);
        self.m.bind(before_word);
        self.m.branch_at_end(true, boundary);
        self.m.load_elem(0);
        self.emit_word(none);
        self.m.jump(boundary);
        self.m.bind(ok);
    }

    /// Scan from POS while elements pass `test`, at most `max` of them (`None`: to the end),
    /// leaving POS past the run, polling between stretches.
    fn emit_scan(&mut self, test: &Test, max: Option<u32>) {
        self.m.set_limit(max);
        if matches!(test, Test::Any) && self.plan.dotall {
            self.m.pos_to_limit();
            return;
        }
        let stretch = self.m.label();
        let scan = self.m.label();
        let bound = self.m.label();
        let done = self.m.label();
        self.m.bind(stretch);
        self.m.set_stretch_limit();
        self.m.bind(scan);
        self.m.branch_pos_ge_stretch(bound);
        self.m.load_elem(0);
        self.emit_test(test, done);
        self.m.advance(1);
        self.m.jump(scan);
        self.m.bind(bound);
        self.m.branch_stretch_is_limit(done);
        self.m.poll_now();
        self.m.jump(stretch);
        self.m.bind(done);
    }

    /// A greedy loop: the longest run, and a record to retry shorter ones unless they are
    /// futile ([`Matcher`]'s `Backtrack::Greedy`).
    fn emit_greedy(&mut self, pc: usize, test: &Test, min: u32, max: Option<u32>, atomic: bool) {
        let bt = self.bt;
        self.m.run_start();
        self.emit_scan(test, max);
        // The program's leading loop records where its run ended (see `Regex::lead_run`).
        if self.plan.lead_run && pc == 1 {
            self.m.store_lead();
        }
        if min > 0 {
            self.m.branch_run(Cond::Lo, min, bt);
        }
        if !atomic {
            let skip = self.m.label();
            self.m.branch_run(Cond::Ls, min, skip);
            let retry = self.stub(Stub::Greedy(pc));
            self.m.push(retry, Operand::Pack(min));
            self.m.choices(1);
            self.m.bind(skip);
        }
    }

    /// A lazy loop: the shortest run whose continuation may match, and a record to retry longer
    /// ones ([`Matcher::lazy_next`]).
    fn emit_lazy(&mut self, pc: usize, test: &Test, min: u32, max: Option<u32>) {
        let bt = self.bt;
        self.m.run_start();
        if min > 0 {
            self.emit_scan(test, Some(min));
            self.m.branch_run(Cond::Lo, min, bt);
        }
        let check = self.m.label();
        let extend = self.m.label();
        self.m.bind(check);
        if let Some(next) = self.plan.first_test(pc + 1) {
            self.emit_may_match(&next, extend);
        }
        let skip = self.m.label();
        if let Some(max) = max {
            self.m.branch_run(Cond::Hs, max, skip);
        }
        let retry = self.stub(Stub::Lazy(pc));
        self.m.push(retry, Operand::Pack(0));
        self.m.choices(1);
        self.m.bind(skip);
        self.lazy.push((pc, check, extend));
        self.extend.insert(pc, extend);
    }

    /// A lazy loop's extension: one more element, then the continuation's test again.
    fn emit_lazy_extend(&mut self, pc: usize, check: Label, extend: Label) {
        let bt = self.bt;
        let Inst::Many { rep, max, .. } = &self.plan.prog[pc] else {
            unreachable!("lazy extension outside its loop");
        };
        let test = self.plan.rep_test(rep);
        let max = max.and_then(|max| u32::try_from(max).ok().filter(|&m| m < 1 << 31));
        self.m.bind(extend);
        if let Some(max) = max {
            self.m.branch_run(Cond::Hs, max, bt);
        }
        self.m.branch_at_end(true, bt);
        self.m.load_elem(0);
        self.emit_test(&test, bt);
        self.m.advance(1);
        self.m.poll_tick();
        self.m.jump(check);
    }

    fn emit_stub(&mut self, stub: Stub) {
        let bt = self.bt;
        match stub {
            Stub::Alt(target) => {
                self.m.pos_from_op();
                self.m.choices(-1);
                self.m.jump(self.pcs[target]);
            }
            Stub::Undo(slot) => {
                self.m.store_slot(slot, Value::Op);
                self.m.jump(bt);
            }
            Stub::Greedy(pc) => {
                // As `Matcher::greedy_retry`: one element shorter at a time, down to the least
                // end, until the continuation's first test may match.
                self.m.choices(-1);
                self.m.unpack_op();
                let retry = self.m.label();
                let next = self.m.label();
                self.m.bind(retry);
                self.m.retreat(1);
                self.m.poll_tick();
                if let Some(test) = self.plan.first_test(pc + 1) {
                    self.emit_may_match(&test, next);
                }
                let skip = self.m.label();
                self.m.branch_pos_rs(Cond::Ls, skip);
                let again = self.stub(Stub::Greedy(pc));
                self.m.push(again, Operand::Pack(0));
                self.m.choices(1);
                self.m.bind(skip);
                self.m.jump(self.pcs[pc + 1]);
                self.m.bind(next);
                self.m.branch_pos_rs(Cond::Hi, retry);
                self.m.jump(bt);
            }
            Stub::Lazy(pc) => {
                self.m.choices(-1);
                self.m.unpack_op();
                let extend = self.extend[&pc];
                self.m.jump(extend);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Running the code
// ---------------------------------------------------------------------------------------------

/// The slots the tier keeps for a program: its capture slots, then its marks.
fn slot_count(re: &Regex) -> usize {
    2 * (re.ngroups + 1) + re.nmarks
}

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
    let nslots = 2 * (re.ngroups + 1);
    let slots = slot_count(re);
    if scratch.native_slots.len() != slots {
        scratch.native_slots = vec![usize::MAX; slots].into_boxed_slice();
    }
    let stack = &mut scratch.native_stack;
    if stack.capacity() < INITIAL_RECORDS {
        stack.reserve(INITIAL_RECORDS);
    }
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
        stack.shrink_to(INITIAL_RECORDS);
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

/// `TRUST_JS_REGEXP_NATIVE_VERIFY=1`: run the matcher where the native code ran and abort, with
/// the pattern, flags, subject and start, on any difference in the result, the capture spans or
/// the error kind. An interruption only one of the two saw is no difference: each polls at its
/// own points.
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

/// The stack is full at `top`: grow it, up to [`MAX_BACKTRACK_STATE`] records (the context's
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

#[cfg(test)]
#[path = "regex_native_tests.rs"]
mod tests;
